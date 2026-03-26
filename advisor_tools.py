import os
import re
import json
import ssl
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import duckdb

try:
    import certifi
except Exception:
    certifi = None

# =========================
# Settings
# =========================
VULN_PARQUET_PATH = os.environ.get(
    "VULN_PARQUET_PATH",
    "/icebox/data/shares/mh2/shassan6/gradio/data/vulnerability.parquet",
)

# Use 2010 blocks to match GEOID10 table
CENSUS_BENCHMARK = "Public_AR_Current"
CENSUS_VINTAGE = "Census2010_Current"

LOCAL_TMP = os.environ.get("SLURM_TMPDIR", "/tmp")
GEOCODE_CACHE_DB = str(Path(LOCAL_TMP) / f"hydroua_geocode_cache_{os.environ.get('USER','user')}.sqlite")
FEMA_CACHE_DB = str(Path(LOCAL_TMP) / f"hydroua_fema_cache_{os.environ.get('USER','user')}.sqlite")

FEMA_NFHL_LAYER28_QUERY = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

FEMA_TIMEOUT = int(os.environ.get("FEMA_TIMEOUT", "60"))
FEMA_RETRIES = int(os.environ.get("FEMA_RETRIES", "2"))
FEMA_CACHE_TTL_SECONDS = int(os.environ.get("FEMA_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))
FEMA_CACHE_DECIMALS = int(os.environ.get("FEMA_CACHE_DECIMALS", "6"))

# =========================
# VI quantile thresholds (your bins)
# =========================
VI_Q20 = 0.374852
VI_Q40 = 0.413375
VI_Q60 = 0.441290
VI_Q80 = 0.496553

BUILDING_TYPE_MAP = {
    "warehouse": "warehouse",
    "hospital": "hospital",
    "clinic": "medical_clinic",
    "school": "school",
    "house": "residential",
    "home": "residential",
    "residential": "residential",
    "apartment": "multi_family_residential",
    "factory": "industrial",
    "industrial": "industrial",
    "commercial": "commercial",
}

TIER_BASE = {1: 90, 2: 75, 3: 55, 4: 35, 5: 15}
BUILDING_PENALTY = {
    "residential": 5,
    "commercial": 10,
    "warehouse": 10,
    "industrial": 15,
    "multi_family_residential": 12,
    "school": 20,
    "medical_clinic": 25,
    "hospital": 35,
}

# =========================
# SSL context
# =========================
def _ssl_context():
    cafile = os.environ.get("SSL_CERT_FILE")
    if not cafile and certifi is not None:
        try:
            cafile = certifi.where()
        except Exception:
            cafile = None
    try:
        if cafile:
            return ssl.create_default_context(cafile=cafile)
        return ssl.create_default_context()
    except Exception:
        return ssl.create_default_context()

SSL_CONTEXT = _ssl_context()

# =========================
# DuckDB connection
# =========================
_DUCK = None

def _duck():
    global _DUCK
    if _DUCK is None:
        con = duckdb.connect(database=":memory:")
        con.execute("PRAGMA threads=8;")
        con.execute("PRAGMA enable_object_cache=true;")
        _DUCK = con
    return _DUCK

# =========================
# Helpers
# =========================
def _norm_addr_key(addr: str) -> str:
    return re.sub(r"\s+", " ", addr.strip().lower())

def _digits(x) -> str:
    return re.sub(r"\D", "", str(x or ""))

def _digits_pad15(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if "e" in s.lower():
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    s = re.sub(r"\D", "", s)
    if 0 < len(s) < 15:
        s = s.zfill(15)
    if len(s) > 15:
        s = s[-15:]
    return s

def _open_cache_conn(path: str):
    con = sqlite3.connect(path, timeout=15, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=15000;")
    return con

# =========================
# Geocode cache (AUTO-MIGRATING)
# =========================
def _ensure_geocode_cache():
    con = _open_cache_conn(GEOCODE_CACHE_DB)

    # Create minimal table if missing
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS geocode_cache (
            address_key TEXT PRIMARY KEY,
            address_raw TEXT,
            geoid10 TEXT,
            lat REAL,
            lon REAL,
            match_quality TEXT
        )
        """
    )

    # MIGRATION: add columns if an older table exists without them
    cols = [r[1] for r in con.execute("PRAGMA table_info(geocode_cache)").fetchall()]

    if "address_raw" not in cols:
        con.execute("ALTER TABLE geocode_cache ADD COLUMN address_raw TEXT")
    if "geoid10" not in cols:
        con.execute("ALTER TABLE geocode_cache ADD COLUMN geoid10 TEXT")
    if "lat" not in cols:
        con.execute("ALTER TABLE geocode_cache ADD COLUMN lat REAL")
    if "lon" not in cols:
        con.execute("ALTER TABLE geocode_cache ADD COLUMN lon REAL")
    if "match_quality" not in cols:
        con.execute("ALTER TABLE geocode_cache ADD COLUMN match_quality TEXT")

    return con

def _geo_cache_read(address_key: str):
    con = _ensure_geocode_cache()
    row = con.execute(
        "SELECT geoid10, lat, lon, match_quality FROM geocode_cache WHERE address_key = ?",
        (address_key,),
    ).fetchone()
    con.close()
    return row

def _geo_cache_write(address_key: str, address_raw: str, geoid10: str, lat, lon, match_quality: str):
    con = _ensure_geocode_cache()
    con.execute(
        "INSERT OR REPLACE INTO geocode_cache(address_key, address_raw, geoid10, lat, lon, match_quality) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (address_key, address_raw, geoid10, lat, lon, match_quality),
    )
    con.close()

# =========================
# FEMA cache
# =========================
def _ensure_fema_cache():
    con = _open_cache_conn(FEMA_CACHE_DB)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fema_cache (
            key TEXT PRIMARY KEY,
            ts INTEGER,
            payload_json TEXT
        )
        """
    )
    return con

def _fema_cache_key(lat: float, lon: float) -> str:
    return f"{round(float(lat), FEMA_CACHE_DECIMALS)},{round(float(lon), FEMA_CACHE_DECIMALS)}"

def _fema_cache_get(lat: float, lon: float):
    con = _ensure_fema_cache()
    k = _fema_cache_key(lat, lon)
    row = con.execute("SELECT ts, payload_json FROM fema_cache WHERE key = ?", (k,)).fetchone()
    con.close()
    if not row:
        return None
    ts, payload_json = row
    if (int(time.time()) - int(ts)) > FEMA_CACHE_TTL_SECONDS:
        return None
    return json.loads(payload_json)

def _fema_cache_put(lat: float, lon: float, payload: dict):
    con = _ensure_fema_cache()
    k = _fema_cache_key(lat, lon)
    con.execute(
        "INSERT OR REPLACE INTO fema_cache(key, ts, payload_json) VALUES (?, ?, ?)",
        (k, int(time.time()), json.dumps(payload)),
    )
    con.close()

# =========================
# Extract building type + address
# =========================
def extract_site_inputs(text: str):
    raw = text or ""
    t = raw.lower()

    building_type = None
    for k, v in BUILDING_TYPE_MAP.items():
        if k in t:
            building_type = v
            break

    # accept at/in/near/around/for
    m = re.search(r"\b(?:at|in|near|around|for)\s+(.+)$", raw, re.IGNORECASE)
    if m:
        address = m.group(1).strip().rstrip("?.!")
        return {"address": address, "building_type": building_type}

    # street address fallback
    m2 = re.search(
        r"\b\d{1,6}\s+[A-Za-z0-9.\- ]+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln|Dr|Drive|Ct|Court|Way|Pl|Place)\b.*",
        raw,
        re.IGNORECASE,
    )
    if m2:
        return {"address": m2.group(0).strip().rstrip("?.!"), "building_type": building_type}

    # coarse location fallback (state/city words)
    if any(s in t for s in ["alabama", "tuscaloosa"]):
        return {"address": raw.strip().rstrip("?.!"), "building_type": building_type}

    return {"address": None, "building_type": building_type}
# =========================
# Geocode -> lat/lon + GEOID10 (2010)
# =========================
def geocode_to_point(address: str):
    if not address:
        return {"error": "Missing address"}

    key = f"{CENSUS_VINTAGE}:{_norm_addr_key(address)}"
    row = _geo_cache_read(key)
    if row:
        geoid10, lat, lon, match_quality = row
        return {"geoid10": geoid10 or "", "lat": lat, "lon": lon, "match_quality": match_quality}

    base = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
    params = {
        "address": address,
        "benchmark": CENSUS_BENCHMARK,
        "vintage": CENSUS_VINTAGE,
        "format": "json",
    }
    url = f"{base}?{urlencode(params)}"

    try:
        req = Request(url, headers={"User-Agent": "HydroUA-GPT/1.0"})
        with urlopen(req, timeout=20, context=SSL_CONTEXT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"Geocoding request failed: {e}"}

    try:
        matches = payload.get("result", {}).get("addressMatches", [])
        if not matches:
            return {"error": "No geocoding match found (try adding city/state or ZIP)"}

        match = matches[0]
        coords = match.get("coordinates", {}) or {}
        lon = coords.get("x")
        lat = coords.get("y")

        geoid10 = ""
        geos = match.get("geographies", {}) or {}
        for k, v in geos.items():
            if "Block" in k and isinstance(v, list) and v:
                rec = v[0] or {}
                cand = rec.get("GEOID") or rec.get("GEOID20") or ""
                cand_digits = _digits(cand)
                if len(cand_digits) >= 14:
                    geoid10 = _digits_pad15(cand_digits)
                    break

        _geo_cache_write(key, address, geoid10, lat, lon, "census_geocoder")
        return {"geoid10": geoid10, "lat": lat, "lon": lon, "match_quality": "census_geocoder"}

    except Exception as e:
        return {"error": f"Geocoder parse error: {e}"}
# =========================
# FEMA query with retries + cache
# =========================
def _fema_query_point(lon: float, lat: float):
    cached = _fema_cache_get(lat, lon)
    if cached is not None:
        return cached

    params = {
        "f": "json",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "false",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DEPTH,VELOCITY,V_DATUM,SOURCE_CIT",
    }
    url = FEMA_NFHL_LAYER28_QUERY + "?" + urlencode(params, quote_via=quote)
    req = Request(url, headers={"User-Agent": "HydroUA-GPT/1.0"})

    last_err = None
    for attempt in range(FEMA_RETRIES + 1):
        try:
            with urlopen(req, timeout=FEMA_TIMEOUT, context=SSL_CONTEXT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if "error" in payload:
                out = {"error": payload["error"]}
            else:
                out = {"features": payload.get("features", [])}
            _fema_cache_put(lat, lon, out)
            return out
        except Exception as e:
            last_err = e
            time.sleep(min(8, 2 ** attempt))

    return {"error": f"Timeout after retries. Last error: {last_err}"}

def _clean_fema_num(v):
    if v is None:
        return None
    try:
        x = float(v)
        if x <= -9999:
            return None
        return x
    except Exception:
        return None

def _derive_fema_tier(features):
    if not features:
        return {
            "tier": 2,
            "tier_label": "No mapped flood polygon at point (screening caution)",
            "zone": None,
            "zone_subtype": None,
            "sfha_tf": None,
            "bfe": None,
            "depth": None,
            "velocity": None,
            "fema_notes": ["No FEMA polygon intersected; verify FEMA MSC and local maps."],
        }

    a = (features[0] or {}).get("attributes", {}) or {}
    zone = (a.get("FLD_ZONE") or "").strip().upper()
    zone_sub = (a.get("ZONE_SUBTY") or "").strip().upper()
    sfha = (a.get("SFHA_TF") or "").strip().upper()

    def rank():
        if zone.startswith("V"):
            return 5
        if sfha == "T":
            return 4
        if zone in {"A", "AE", "AH", "AO", "A99", "AR"}:
            return 4
        if "1 PCT ANNUAL CHANCE" in zone_sub:
            return 4
        if "0.2 PCT ANNUAL CHANCE" in zone_sub:
            return 3
        if zone == "X":
            return 2
        return 3

    tier = rank()
    tier_label_map = {
        5: "Very High (Coastal / wave-action flood hazard)",
        4: "High (1% annual chance flood hazard / SFHA)",
        3: "Moderate (0.2% annual chance or uncertain mapped condition)",
        2: "Lower mapped risk at point (not zero risk)",
        1: "Very Low",
    }

    notes = []
    if tier >= 4:
        notes.append("Point intersects FEMA SFHA (1% annual chance).")
    else:
        notes.append("Point is not in mapped SFHA at this point; local drainage checks recommended.")

    return {
        "tier": tier,
        "tier_label": tier_label_map.get(tier, f"Tier {tier}"),
        "zone": zone,
        "zone_subtype": zone_sub,
        "sfha_tf": sfha,
        "bfe": _clean_fema_num(a.get("STATIC_BFE")),
        "depth": _clean_fema_num(a.get("DEPTH")),
        "velocity": _clean_fema_num(a.get("VELOCITY")),
        "fema_notes": notes,
    }

def lookup_fema_flood_hazard(lat: float, lon: float):
    q = _fema_query_point(float(lon), float(lat))
    if q.get("error"):
        return {"found": False, "error": q["error"]}
    d = _derive_fema_tier(q.get("features", []))
    d["found"] = True
    return d

# =========================
# VI lookup (Parquet query, correct .0 stripping)
# =========================
def lookup_vulnerability_index(geoid10: str):
    g = _digits_pad15(geoid10)
    if len(g) != 15:
        return {"found": False, "error": "Invalid GEOID10", "vi": None}

    if not os.path.exists(VULN_PARQUET_PATH):
        return {"found": False, "error": f"Missing parquet: {VULN_PARQUET_PATH}", "vi": None}

    con = _duck()
    row = con.execute(
        r"""
        SELECT Vulnerability_Index
        FROM read_parquet(?)
        WHERE lpad(
                regexp_replace(
                  regexp_replace(CAST(GEOID10 AS VARCHAR), '\.0$', ''),
                  '[^0-9]', '', 'g'
                ),
                15, '0'
              ) = ?
        LIMIT 1
        """,
        [VULN_PARQUET_PATH, g],
    ).fetchone()

    if not row:
        return {"found": False, "error": "GEOID10 not found in VI table", "vi": None}

    vi = float(row[0])
    vi_pct = vi * 100.0

    if vi < VI_Q20:
        vi_tier, vi_label = 1, "Very Low"
    elif vi < VI_Q40:
        vi_tier, vi_label = 2, "Low"
    elif vi < VI_Q60:
        vi_tier, vi_label = 3, "Medium"
    elif vi < VI_Q80:
        vi_tier, vi_label = 4, "High"
    else:
        vi_tier, vi_label = 5, "Very High"

    return {"found": True, "vi": vi, "vi_pct": vi_pct, "vi_tier": vi_tier, "vi_label": vi_label}

def combine_tiers(fema_tier: int, vi_tier: int):
    return max(int(fema_tier), int(vi_tier))

def score_site(combined_tier: int, building_type: str):
    tier = int(combined_tier)
    base = TIER_BASE.get(tier, 50)
    penalty = BUILDING_PENALTY.get(building_type, 10)
    score = max(0, min(100, base - penalty))

    if score >= 75:
        label = "Low Concern"
    elif score >= 50:
        label = "Moderate Concern"
    elif score >= 30:
        label = "Conditional"
    else:
        label = "High Concern"

    triggers = []
    if tier >= 3:
        triggers.append("Drainage assessment recommended")
    if tier >= 4:
        triggers.append("Floodplain review required if applicable; verify FEMA panel and local ordinance")
    if building_type in {"hospital", "school"}:
        triggers.append("Critical facility standard review required")
    if label in {"Conditional", "High Concern"}:
        triggers.append("Site-specific engineering study required")
    if not triggers:
        triggers.append("Verify local drainage, grading, and stormwater compliance before design.")

    return {"score": score, "label": label, "triggers": triggers}