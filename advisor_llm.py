import json
import re
from typing import Dict, List, Optional, Any

import torch
import model_runtime


CANONICAL_TYPES = [
    "warehouse",
    "residential",
    "multi_family_residential",
    "commercial",
    "industrial",
    "school",
    "medical_clinic",
    "hospital",
]

TYPE_SYNONYMS = {
    "home": "residential",
    "house": "residential",
    "apartment": "multi_family_residential",
    "multifamily": "multi_family_residential",
    "multi-family": "multi_family_residential",
    "factory": "industrial",
    "plant": "industrial",
    "clinic": "medical_clinic",
    "club": "commercial",
    "nightclub": "commercial",
    "bar": "commercial",
}

ADDR_HINT_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\- ]+(?:st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|way|pl|place)\b",
    re.IGNORECASE,
)


def _get_model_and_tokenizer():
    ret = model_runtime.get_model_and_tokenizer()
    if isinstance(ret, (tuple, list)):
        if len(ret) < 2:
            raise ValueError("get_model_and_tokenizer must return at least (model, tokenizer)")
        return ret[0], ret[1]
    if isinstance(ret, dict) and "model" in ret and "tokenizer" in ret:
        return ret["model"], ret["tokenizer"]
    raise ValueError(f"Unsupported get_model_and_tokenizer() return type: {type(ret)}")


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def _truncate_history(history: List[Dict[str, str]], max_turns: int = 10) -> List[Dict[str, str]]:
    return [{"role": m.get("role", ""), "content": m.get("content", "")} for m in (history or [])[-max_turns:]]


def _extract_final_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    m = re.search(r"<FINAL_JSON>\s*(\{.*?\})\s*</FINAL_JSON>", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            return None

    dec = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for s in reversed(starts):
        try:
            obj, _ = dec.raw_decode(text[s:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _validate_slots(slots: Any, user_text: str) -> Dict[str, Any]:
    out = {
        "intent": "advisor",
        "building_type": None,
        "address": None,
        "mode": "block",
        "needs_followup": False,
        "followup_question": "",
        "confidence": 0.5,
        "notes": "",
    }

    if not isinstance(slots, dict):
        slots = {}

    bt = slots.get("building_type", None)
    if bt is not None:
        bt = str(bt).strip().lower()
        bt = TYPE_SYNONYMS.get(bt, bt)
        if bt in CANONICAL_TYPES:
            out["building_type"] = bt

    if re.search(r"\bnot\s+(a\s+)?hospital\b", (user_text or "").lower()):
        if out["building_type"] == "hospital":
            out["building_type"] = None

    addr = slots.get("address", None)
    if addr is not None:
        addr = str(addr).strip()
        if ADDR_HINT_RE.search(addr) or re.search(r"\b[A-Za-z .'-]+,\s*[A-Z]{2}\b", addr):
            out["address"] = addr

    mode = str(slots.get("mode", "block")).strip().lower()
    if mode in {"block", "city", "state"}:
        out["mode"] = mode

    out["needs_followup"] = bool(slots.get("needs_followup", False))
    fq = slots.get("followup_question", "")
    out["followup_question"] = str(fq).strip()[:300] if fq else ""

    if not out["building_type"] and not out["address"]:
        out["needs_followup"] = True
        if not out["followup_question"]:
            out["followup_question"] = (
                "I can screen a build site, but I need building type + address. "
                "Example: 'club at 123 Main St, Tuscaloosa AL' or 'hospital at 302 Reed St, Tuscaloosa AL'."
            )

    return out


def _generate_text(system_prompt: str, user_payload: str, max_new_tokens: int = 220) -> str:
    model, tokenizer = _get_model_and_tokenizer()

    if _has_chat_template(tokenizer):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{system_prompt}\n\nUSER:\n{user_payload}\nASSISTANT:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


def llm_fill_slots(user_text: str, history: List[Dict[str, str]], geocode_error: str = "") -> Dict[str, Any]:
    model, tokenizer = _get_model_and_tokenizer()
    hist = _truncate_history(history, 10)

    system = (
        "You extract building_type and address for a Build-Site Feasibility Advisor.\n"
        "Return STRICT JSON ONLY inside <FINAL_JSON>...</FINAL_JSON>.\n"
        "Rules:\n"
        "1) building_type must be one of: " + ", ".join(CANONICAL_TYPES) + "\n"
        "2) Recognize synonyms: house/home->residential; club/nightclub/bar->commercial.\n"
        "3) If user says 'not a hospital', do NOT set building_type=hospital.\n"
        "4) Only set address if it clearly contains a street address (number + street suffix) or City, ST.\n"
        "5) If missing info, ask ONE short follow-up question.\n"
        "6) Never invent FEMA/VI/tiers/scores.\n\n"
        "Schema:\n"
        "<FINAL_JSON>{"
        "\"intent\":\"advisor\","
        "\"building_type\":null,"
        "\"address\":null,"
        "\"mode\":\"block\","
        "\"needs_followup\":false,"
        "\"followup_question\":\"\","
        "\"confidence\":0.5,"
        "\"notes\":\"\""
        "}</FINAL_JSON>\n"
    )

    payload = {
        "user_message": user_text,
        "geocode_error": geocode_error or "",
        "conversation_tail": hist,
    }

    if _has_chat_template(tokenizer):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = system + "\nUSER_JSON:\n" + json.dumps(payload, ensure_ascii=False) + "\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    slots = _extract_final_json(text)
    return _validate_slots(slots, user_text)


def interpret_rainfall_risk_llm(rainfall: Dict[str, Any], building_type: str = "", address: str = "") -> str:
    d2 = rainfall.get("2yr_24hr")
    d10 = rainfall.get("10yr_24hr")
    d50 = rainfall.get("50yr_24hr")
    d100 = rainfall.get("100yr_24hr")
    avg = rainfall.get("avg_intensity_100yr_24hr")

    if all(v is None for v in [d2, d10, d50, d100, avg]):
        return (
            "NOAA Atlas 14 precipitation frequency values were not successfully retrieved for this location, "
            "so rainfall intensity risk could not be interpreted from Atlas 14 in this run."
        )

    system = (
        "You are a hydrology and stormwater engineering assistant.\n\n"
        "Rules:\n"
        "1. Do NOT make a yes/no build decision from rainfall alone.\n"
        "2. Treat the 100-year 24-hour rainfall depth as the main screening indicator.\n"
        "3. Focus on runoff, drainage demand, and extreme-event stormwater implications.\n"
        "4. Do not invent numbers. Only use provided values.\n"
        "5. Keep it short: 3-4 sentences max.\n"
    )

    payload = {
        "building_type": building_type,
        "address": address,
        "rainfall_statistics": {
            "2yr_24hr_in": d2,
            "10yr_24hr_in": d10,
            "50yr_24hr_in": d50,
            "100yr_24hr_in": d100,
            "avg_100yr_24hr_in_per_hr": avg,
            "risk_label": rainfall.get("risk_label"),
        },
    }

    try:
        text = _generate_text(system, json.dumps(payload, ensure_ascii=False), max_new_tokens=170).strip()
        if text:
            return text
    except Exception:
        pass

    if d100 is None:
        return (
            "NOAA Atlas 14 precipitation frequency data was partially available, "
            "but the 100-year 24-hour depth could not be interpreted."
        )

    return (
        f"NOAA Atlas 14 indicates a 100-year 24-hour rainfall depth of {d100:.2f} inches at this site. "
        f"That is a strong extreme-event runoff signal for screening purposes and suggests substantial drainage demand during severe storms. "
        f"For a {building_type or 'proposed site'}, stormwater capacity and runoff management should be evaluated carefully."
    )


def interpret_terrain_risk_llm(terrain: Dict[str, Any], building_type: str = "", address: str = "") -> str:
    if not terrain.get("found"):
        return "Terrain and local drainage-context screening was unavailable in this run."

    system = (
        "You are a terrain and drainage screening assistant.\n\n"
        "You are given local terrain metrics derived from ring-sampled elevation points around a site.\n"
        "Interpret them conservatively for site screening.\n"
        "Focus on whether the site appears locally low, convergent, flat, depressional, or relatively elevated.\n"
        "Do not claim full hydrologic modeling. Keep it to 3-4 sentences.\n"
    )

    payload = {
        "building_type": building_type,
        "address": address,
        "terrain": terrain,
    }

    try:
        text = _generate_text(system, json.dumps(payload, ensure_ascii=False), max_new_tokens=170).strip()
        if text:
            return text
    except Exception:
        pass

    cls = terrain.get("terrain_class", "mixed local terrain")
    delta = terrain.get("center_minus_inner_mean_m")
    slope = terrain.get("mean_abs_slope_pct_inner")

    if terrain.get("depression_flag"):
        return (
            f"Local terrain screening suggests the site sits in a relatively low or depressional position compared with nearby sampled points. "
            f"That can increase ponding sensitivity and make extreme-rainfall runoff consequences worse even outside a mapped FEMA floodplain."
        )
    if terrain.get("convergence_flag"):
        return (
            f"Local terrain screening suggests a convergent setting with nearby ground tending to drain toward the site. "
            f"With flat-to-gentle slopes, that can increase localized runoff concentration and drainage demand."
        )
    if terrain.get("ridge_flag"):
        return (
            f"Local terrain screening suggests the site is somewhat elevated relative to nearby sampled points, which is generally more favorable for local drainage than a low-lying setting. "
            f"That does not remove flood or stormwater risk, but it reduces concern about local topographic ponding."
        )

    return (
        f"Local terrain screening indicates {cls.lower()}, with center-to-neighbor elevation difference of about {delta} m "
        f"and mean local slope proxy around {slope}%. This is useful as a first-pass terrain context, but it should be followed by detailed grading and drainage analysis for design decisions."
    )


def interpret_site_risk_llm(
    building_type: str,
    address: str,
    fema: Dict[str, Any],
    noaa: Dict[str, Any],
    vi: Dict[str, Any],
    terrain: Dict[str, Any],
    risk: Dict[str, Any],
) -> str:
    system = (
        "You are a hydrology, terrain, and vulnerability screening assistant.\n\n"
        "Rules:\n"
        "1. Treat FEMA and NOAA as physical hazard signals.\n"
        "2. Treat local terrain/drainage context as a key site-condition signal.\n"
        "3. Treat SEIV as a vulnerability/consequence amplifier, not a direct no-build indicator.\n"
        "4. Do not say 'cannot build' unless direct prohibitive evidence exists.\n"
        "5. Keep it concise: 4-6 sentences.\n"
        "6. Do not invent numbers or facts.\n"
    )

    payload = {
        "building_type": building_type,
        "address": address,
        "fema": {
            "found": fema.get("found"),
            "zone": fema.get("zone"),
            "tier": fema.get("tier"),
            "tier_label": fema.get("tier_label"),
        },
        "noaa": {
            "found": noaa.get("found"),
            "100yr_24hr": noaa.get("100yr_24hr"),
            "risk_label": noaa.get("risk_label"),
        },
        "vi": {
            "found": vi.get("found"),
            "vi": vi.get("vi"),
            "vi_tier": vi.get("vi_tier"),
            "vi_label": vi.get("vi_label"),
        },
        "terrain": terrain,
        "composite_risk": risk,
    }

    try:
        text = _generate_text(system, json.dumps(payload, ensure_ascii=False), max_new_tokens=230).strip()
        if text:
            return text
    except Exception:
        pass

    pieces = []

    if fema.get("found"):
        pieces.append(
            f"FEMA maps the site in zone {fema.get('zone') or 'unknown'}, which indicates {str(fema.get('tier_label', 'lower mapped flood hazard')).lower()}."
        )
    else:
        pieces.append("Mapped FEMA flood hazard was unavailable in this run.")

    if noaa.get("found") and isinstance(noaa.get("100yr_24hr"), (int, float)):
        pieces.append(
            f"NOAA Atlas 14 indicates about {noaa['100yr_24hr']:.2f} inches for the 100-year 24-hour rainfall event, which is a strong runoff-demand signal."
        )
    else:
        pieces.append("NOAA Atlas 14 rainfall intensity was unavailable in this run.")

    if terrain.get("found"):
        pieces.append(
            f"Local terrain screening indicates {str(terrain.get('terrain_class', 'mixed local terrain')).lower()}, which affects how readily water may concentrate or drain away from the site."
        )
    else:
        pieces.append("Terrain and local drainage-context screening was unavailable in this run.")

    if vi.get("found"):
        pieces.append(
            f"The block-level SEIV is {str(vi.get('vi_label', 'unknown')).lower()} (tier {vi.get('vi_tier', 'N/A')}), meaning disruption and recovery impacts may be more severe than in lower-vulnerability blocks."
        )

    if building_type == "hospital":
        pieces.append(
            "For a hospital, the main implication is consequence sensitivity: service continuity, emergency access, and utility reliability matter more than for ordinary occupancy."
        )
    elif building_type in {"medical_clinic", "school"}:
        pieces.append(
            f"For a {building_type}, operational and safety impacts can be disproportionate even when mapped flood hazard is not extreme."
        )
    else:
        pieces.append(
            f"For a {building_type}, this should be read as a screening-level indication of hazard, terrain, and vulnerability context."
        )

    pieces.append(
        f"Overall, this supports a composite screening score of {risk.get('final_score', 'N/A')}/100 with an assessment of {risk.get('label', 'N/A')}."
    )

    return " ".join(pieces)
