"""
Shared utility helpers for the HydroUA-GPT advisor workflow.

These helpers were originally embedded in advisor_flow.py. Keeping them here lets the
LangGraph/multi-agent workflow avoid depending on the old procedural advisor_flow module.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


BUILDING_TERM_DEFS = {
    "warehouse": (
        "A **warehouse** is a large building or facility used primarily for **storing goods and materials**.\n"
        "- Typically used for inventory storage, distribution, and logistics."
    ),
    "hospital": (
        "A **hospital** is a healthcare facility that provides medical treatment and emergency services.\n"
        "- It is a **critical facility**, so continuity of operations and emergency access are especially important."
    ),
    "clinic": (
        "A **clinic** is a healthcare facility focused on outpatient services.\n"
        "- Usually smaller than a hospital and often does not provide overnight stays."
    ),
    "school": (
        "A **school** is an educational facility for teaching and learning.\n"
        "- Often treated as a sensitive occupancy for hazard screening and emergency planning."
    ),
    "residential": (
        "**Residential** refers to housing used for people to live in.\n"
        "- Examples: single-family homes, apartments, condos, townhomes."
    ),
    "commercial": (
        "**Commercial** refers to buildings used for business activities.\n"
        "- Examples: offices, retail, restaurants, and service businesses."
    ),
    "industrial": (
        "**Industrial** refers to facilities used for manufacturing or heavy processing.\n"
        "- These often have stricter operational and environmental site constraints."
    ),
    "club": (
        "A **club** is a commercial place for entertainment and social gathering.\n"
        "- High occupancy can make emergency access and life-safety issues important."
    ),
}

DEFN_RE = re.compile(r"^\s*(what is|define|meaning of|mean by)\b", re.IGNORECASE)
HAS_BUILD_RE = re.compile(r"\b(build|construction|construct|feasibility)\b", re.IGNORECASE)
ADDR_HINT_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\- ]+(st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|way|pl|place|south|north|east|west)\b",
    re.IGNORECASE,
)
LOCATION_PREFIX_RE = re.compile(
    r"^\s*(my\s+location\s+is|location\s+is|it's\s+at|it\s+is\s+at|at|in|near|around)\s+",
    re.IGNORECASE,
)


def clean_text(s: str) -> str:
    return (s or "").strip().rstrip("?.!")


def extract_address_from_text(raw: str) -> str:
    if not raw:
        return ""
    s = clean_text(raw)
    s = LOCATION_PREFIX_RE.sub("", s).strip()

    m = re.search(r"\b(?:at|in|near|around)\s+(.+)$", s, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if any(ch.isdigit() for ch in candidate):
            return clean_text(candidate)

    m2 = ADDR_HINT_RE.search(s)
    if m2:
        idx = m2.start()
        return clean_text(s[idx:])

    if re.search(r"\b[A-Za-z .'-]+,\s*[A-Z]{2}\b", s):
        return clean_text(s)

    return ""


def infer_building_type_from_text(raw: str) -> Optional[str]:
    s = (raw or "").lower()
    if "warehouse" in s:
        return "warehouse"
    if "hospital" in s:
        return "hospital"
    if "clinic" in s:
        return "medical_clinic"
    if "school" in s:
        return "school"
    if "club" in s or "nightclub" in s or "bar" in s:
        return "commercial"
    if "apartment" in s or "multifamily" in s or "multi family" in s:
        return "multi_family_residential"
    if "house" in s or "home" in s or "residential" in s:
        return "residential"
    if "industrial" in s or "factory" in s:
        return "industrial"
    if "commercial" in s:
        return "commercial"
    return None


def definition_handler(raw: str) -> Optional[str]:
    t = (raw or "").lower()

    if not (DEFN_RE.match(raw or "") or "meaning" in t):
        return None
    if HAS_BUILD_RE.search(raw or ""):
        return None
    if ADDR_HINT_RE.search(raw or ""):
        return None

    for term in ["warehouse", "hospital", "clinic", "school", "residential", "commercial", "industrial", "club"]:
        if term in t:
            return BUILDING_TERM_DEFS.get(term, f"**{term}** is a type of building/use category.")
    return None


def last_address_from_history(history: List[Dict[str, str]]) -> Optional[str]:
    for m in reversed(history or []):
        if m.get("role") != "user":
            continue
        a = extract_address_from_text(m.get("content", ""))
        if a:
            return a
    return None


def last_building_from_history(history: List[Dict[str, str]]) -> Optional[str]:
    for m in reversed(history or []):
        if m.get("role") != "user":
            continue
        bt = infer_building_type_from_text(m.get("content", ""))
        if bt:
            return bt
    return None


def fmt_inches(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"


def fmt_intensity(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"
