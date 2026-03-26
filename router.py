import re

# Advisor building keywords
BUILDING_TYPES = {
    "warehouse", "hospital", "clinic", "school",
    "house", "home", "residential", "apartment",
    "industrial", "factory", "commercial",
}

# Hydrology terms (include rain/weather)
HYDRO_TERMS = {
    "hydrology", "soil", "runoff", "run off", "flood", "drought",
    "water", "watershed", "basin", "river", "stream", "lake",
    "groundwater", "aquifer", "precipitation", "rainfall", "rain",
    "discharge", "streamflow", "hydrograph", "infiltration",
    "evapotranspiration", "erosion", "sediment", "stormwater"
}

DEFN_RE = re.compile(r"^\s*(what is|define|explain|how does|how do|why)\b", re.IGNORECASE)

ADDRESS_RE = re.compile(r"\b\d{1,6}\s+[A-Za-z0-9.\- ]+", re.IGNORECASE)
STREET_SUFFIX_RE = re.compile(
    r"\b(st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|way|pl|place)\b",
    re.IGNORECASE
)
CITY_STATE_RE = re.compile(r"\b[A-Za-z .'-]+,\s*[A-Z]{2}\b")
ZIP_RE = re.compile(r"\b\d{5}\b")

# Follow-up elaboration short forms
ELAB_RE = re.compile(
    r"^\s*(define|explain)\s*(it|that)?\s*(more|again|better|elaborately)?\s*$",
    re.IGNORECASE
)

# Weather intent (must override advisor even if it includes an address)
WEATHER_RE = re.compile(
    r"\b(will it rain|rain tonight|rain tomorrow|forecast|weather)\b",
    re.IGNORECASE
)

def looks_like_location_reply(text: str) -> bool:
    q = (text or "").strip()
    t = q.lower()
    if not q:
        return False
    if t in {"use city-level", "use city level", "city-level", "city level"}:
        return True
    if t.startswith(("for ", "in ", "at ", "near ", "around ")):
        return True
    if CITY_STATE_RE.search(q) or ZIP_RE.search(q):
        return True
    if ADDRESS_RE.search(q) and STREET_SUFFIX_RE.search(q):
        return True
    return False

def route_intent_rules(text: str) -> str:
    q = (text or "").strip()
    t = q.lower()

    # 1) WEATHER must be hydrology even if an address is present
    if WEATHER_RE.search(q):
        return "hydrology"

    # 2) Elaborate/define follow-ups should stay hydrology (app.py will also reinforce)
    if ELAB_RE.match(q):
        return "hydrology"

    # 3) Advisor: build intent / building types / location fragments
    if "build" in t or "can i build" in t:
        return "advisor"
    if any(bt in t for bt in BUILDING_TYPES):
        return "advisor"
    if looks_like_location_reply(q):
        return "advisor"

    # 4) Hydrology: any hydro term
    if any(ht in t for ht in HYDRO_TERMS):
        return "hydrology"

    return "offtopic"

def is_ambiguous(text: str) -> bool:
    # Don't block conceptual hydrology questions.
    if not text:
        return True
    if DEFN_RE.match(text.strip()):
        return False
    if ELAB_RE.match(text.strip()):
        return False
    return False