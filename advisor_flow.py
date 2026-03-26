import re
from concurrent.futures import ThreadPoolExecutor

from advisor_tools import (
    geocode_to_point,
    lookup_fema_flood_hazard,
    lookup_vulnerability_index,
    combine_tiers,
    score_site,
)
from advisor_llm import llm_fill_slots


# -------------------------
# Term definitions (build-related vocabulary)
# -------------------------
BUILDING_TERM_DEFS = {
    "warehouse": (
        "A **warehouse** is a large building or facility used primarily for **storing goods and materials**.\n"
        "- Typically used for **inventory storage**, **distribution**, and **logistics** (shipping/receiving).\n"
        "- Often includes **loading docks**, **racking/shelving**, and space for **forklifts/material handling**.\n"
        "- May be used for **cold storage**, **fulfillment**, or **light assembly**, depending on the operation."
    ),
    "hospital": (
        "A **hospital** is a healthcare facility that provides **medical treatment and emergency services**.\n"
        "- Includes **inpatient** care (overnight stays) and often **emergency**, **surgery**, and **diagnostics**.\n"
        "- Considered a **critical facility**, so safety standards (including flood risk) are stricter."
    ),
    "clinic": (
        "A **clinic** is a healthcare facility focused on **outpatient** services.\n"
        "- Usually smaller than a hospital and often does not provide overnight stays.\n"
        "- Can include primary care, specialty care, imaging, labs, etc."
    ),
    "school": (
        "A **school** is an educational facility for teaching and learning.\n"
        "- Often treated as a **sensitive/critical occupancy** for hazard screening and emergency planning."
    ),
    "residential": (
        "**Residential** refers to housing used for people to live in.\n"
        "- Examples: **single-family homes**, **townhomes**, **apartments**, **condos**.\n"
        "- Different building codes/standards apply compared to warehouses or industrial facilities."
    ),
    "commercial": (
        "**Commercial** refers to buildings used for business activities.\n"
        "- Examples: **retail**, **offices**, **restaurants**, **clubs**, **services**.\n"
        "- Occupancy and code requirements vary by use and crowd size."
    ),
    "industrial": (
        "**Industrial** refers to facilities used for manufacturing or heavy processing.\n"
        "- Examples: factories, plants, processing facilities.\n"
        "- Often involves higher utility demand and stricter environmental/site constraints."
    ),
    "club": (
        "A **club** (nightclub/venue) is a **commercial** place for entertainment and social gathering.\n"
        "- Often has **high occupancy**, so fire/life-safety and emergency access requirements are important."
    ),
}

DEFN_RE = re.compile(r"^\s*(what is|define|meaning of|mean by)\b", re.IGNORECASE)
HAS_BUILD_RE = re.compile(r"\b(build|construction|construct|feasibility)\b", re.IGNORECASE)
ADDR_HINT_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\- ]+(st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|way|pl|place)\b",
    re.IGNORECASE,
)
LOCATION_PREFIX_RE = re.compile(
    r"^\s*(my\s+location\s+is|location\s+is|it's\s+at|it\s+is\s+at|at|in|near|around)\s+",
    re.IGNORECASE,
)

def _clean_text(s: str) -> str:
    return (s or "").strip().rstrip("?.!")

def _extract_address_from_text(raw: str) -> str:
    if not raw:
        return ""
    s = _clean_text(raw)
    s = LOCATION_PREFIX_RE.sub("", s).strip()
    # if the whole message contains an address substring, prefer that
    m = ADDR_HINT_RE.search(s)
    if m:
        # If user wrote: "can I build X at 123 Main St, Tuscaloosa AL"
        # cut to the last " at "
        if " at " in s.lower():
            parts = re.split(r"\bat\b", s, flags=re.IGNORECASE)
            tail = parts[-1].strip()
            if any(ch.isdigit() for ch in tail):
                return _clean_text(tail)
        return s
    # city/state fallback
    if re.search(r"\b[A-Za-z .'-]+,\s*[A-Z]{2}\b", s):
        return _clean_text(s)
    return ""

def _infer_bt_from_text(raw: str):
    s = (raw or "").lower()

    # map common uses
    if "warehouse" in s: return "warehouse"
    if "hospital" in s: return "hospital"
    if "clinic" in s: return "medical_clinic"
    if "school" in s: return "school"
    if "club" in s or "nightclub" in s or "bar" in s: return "commercial"
    if "apartment" in s or "multifamily" in s or "multi family" in s: return "multi_family_residential"
    if "house" in s or "home" in s or "residential" in s: return "residential"
    if "industrial" in s or "factory" in s: return "industrial"
    if "commercial" in s: return "commercial"
    return None

def _definition_handler(raw: str) -> str | None:
    """
    If user is asking for the meaning/definition of a building term, answer that
    instead of running build-site tools.
    """
    t = raw.lower()

    # Only treat as definition if it looks definitional AND NOT a build-site question
    if not (DEFN_RE.match(raw) or "meaning" in t):
        return None
    if HAS_BUILD_RE.search(raw):
        return None
    if ADDR_HINT_RE.search(raw):
        return None  # if they provided an address, likely screening not definition

    # find which term they mean
    for term in ["warehouse", "hospital", "clinic", "school", "residential", "commercial", "industrial", "club"]:
        if term in t:
            # return from map, fallback to short generic if missing
            return BUILDING_TERM_DEFS.get(term, f"**{term}** is a type of building/use category.")
    return None

def _last_address_from_history(history):
    for m in reversed(history or []):
        if m.get("role") != "user":
            continue
        a = _extract_address_from_text(m.get("content", ""))
        if a:
            return a
    return None

def _last_building_from_history(history):
    for m in reversed(history or []):
        if m.get("role") != "user":
            continue
        bt = _infer_bt_from_text(m.get("content", ""))
        if bt:
            return bt
    return None


def run_build_site_advisor(user_message: str, history=None) -> dict:
    history = history or []
    raw = _clean_text(user_message)

    # 0) Definition shortcut (warehouse meaning, etc.)
    defn = _definition_handler(raw)
    if defn:
        return {"text": defn, "lat": None, "lon": None}

    # 1) LLM slot fill, then hard guardrails
    slots = llm_fill_slots(raw, history)
    building_type = slots.get("building_type") or _infer_bt_from_text(raw)
    address = _clean_text(slots.get("address") or "")

    # Guardrail: if user text contains a real address, prefer that
    addr_from_raw = _extract_address_from_text(raw)
    if addr_from_raw:
        address = addr_from_raw

    # 2) History fill (only when missing)
    if building_type and not address:
        prev_addr = _last_address_from_history(history)
        if prev_addr:
            address = prev_addr

    if address and not building_type:
        prev_bt = _last_building_from_history(history)
        if prev_bt:
            building_type = prev_bt

    # 3) Ask only what's missing
    if building_type and not address:
        return {
            "text": f"What is the location/address for the {building_type}? Example: 302 Reed St, Tuscaloosa AL",
            "lat": None,
            "lon": None,
        }

    if address and not building_type:
        return {
            "text": f"What building type is it at {address} (warehouse, home, hospital, school, club, etc.)?",
            "lat": None,
            "lon": None,
        }

    if not building_type and not address:
        return {
            "text": slots.get("followup_question", "What building type and what location/address should I screen?"),
            "lat": None,
            "lon": None,
        }

    # City/state only: ask street address or allow city-level
    if slots.get("mode") in {"city", "state"} and address and not any(ch.isdigit() for ch in address):
        return {
            "text": (
                f"I can do a coarse screening for: {address}\n"
                "But block-level vulnerability works best with a street address.\n\n"
                "Reply with a street address (best) OR reply 'use city-level' to proceed (FEMA-only)."
            ),
            "lat": None,
            "lon": None,
        }

    # 4) Geocode
    geo = geocode_to_point(address)
    if geo.get("error"):
        # One retry: LLM repair
        slots2 = llm_fill_slots(raw, history, geocode_error=geo["error"])
        new_addr = _extract_address_from_text(slots2.get("address", "")) or addr_from_raw
        if new_addr and new_addr != address and not slots2.get("needs_followup", False):
            address = new_addr
            geo = geocode_to_point(address)

    if geo.get("error"):
        return {"text": f"I could not geocode the address: {geo['error']}", "lat": None, "lon": None}

    lat = geo.get("lat")
    lon = geo.get("lon")
    geoid10 = geo.get("geoid10") or ""

    # 5) Parallel: FEMA + VI
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_fema = ex.submit(lookup_fema_flood_hazard, lat, lon)
        fut_vi = ex.submit(lookup_vulnerability_index, geoid10)
        fema = fut_fema.result()
        vi = fut_vi.result()

    fema_ok = bool(fema.get("found"))
    vi_ok = bool(vi.get("found"))

    if fema_ok and vi_ok:
        combined_tier = combine_tiers(fema["tier"], vi["vi_tier"])
        combined_label = f"Combined tier = max(FEMA {fema['tier']}, VI {vi['vi_tier']})"
    elif vi_ok and not fema_ok:
        combined_tier = int(vi["vi_tier"])
        combined_label = "Combined tier = VI only (FEMA unavailable)"
    elif fema_ok and not vi_ok:
        combined_tier = int(fema["tier"])
        combined_label = "Combined tier = FEMA only (VI unavailable)"
    else:
        combined_tier = 3
        combined_label = "Combined tier = 3 (fallback)"

    sc = score_site(combined_tier, building_type)

    lines = [
        f"Screening result for the proposed {building_type} at {address}:",
        "",
        f"Coordinates: ({lat}, {lon})",
        f"GEOID10 (Census Block): {geoid10 if geoid10 else 'N/A'}",
        "",
        "Flood hazard (FEMA NFHL):",
    ]

    if fema_ok:
        lines += [
            f"FEMA zone: {fema.get('zone') or 'N/A'}",
            f"FEMA subtype: {fema.get('zone_subtype') or 'N/A'}",
            f"SFHA flag (1% annual chance): {fema.get('sfha_tf') or 'N/A'}",
            f"FEMA tier: {fema['tier']} ({fema.get('tier_label','N/A')})",
        ]
    else:
        lines += [f"FEMA status: unavailable ({fema.get('error')})"]

    lines += ["", "Social vulnerability (Block-level table):"]
    if vi_ok:
        lines += [
            f"Vulnerability_Index: {vi['vi']} (scaled ~ {vi['vi_pct']:.2f})",
            f"VI tier: {vi['vi_tier']} ({vi['vi_label']})",
        ]
    else:
        lines += [f"VI lookup: not available ({vi.get('error')})"]

    lines += [
        "",
        f"Overall risk tier: {combined_tier}",
        combined_label,
        f"Feasibility score: {sc['score']}/100",
        f"Assessment label: {sc['label']}",
        "",
        "Recommended next steps:",
    ]
    lines += [f"- {t}" for t in sc["triggers"]]

    if fema_ok and fema.get("fema_notes"):
        lines += ["", "FEMA notes:"]
        lines += [f"- {n}" for n in fema["fema_notes"]]

    lines += [
        "",
        "Limitation: Screening-level only. FEMA NFHL + block-level vulnerability index do not replace a survey, "
        "local floodplain determination, or site-specific engineering.",
    ]

    return {"text": "\n".join(lines), "lat": lat, "lon": lon}