import os
import re
import json
import ssl
import math
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

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
    "/icebox/data/shares/mh2/mkarimiziarani/agent/data/vulnerability.parquet",
)

CENSUS_BENCHMARK = "Public_AR_Current"
CENSUS_VINTAGE = "Census2010_Current"

LOCAL_TMP = os.environ.get("SLURM_TMPDIR", "/tmp")
GEOCODE_CACHE_DB = str(Path(LOCAL_TMP) / f"hydroua_geocode_cache_{os.environ.get('USER','user')}.sqlite")
FEMA_CACHE_DB = str(Path(LOCAL_TMP) / f"hydroua_fema_cache_{os.environ.get('USER','user')}.sqlite")
NOAA_CACHE_DB = str(Path(LOCAL_TMP) / f"hydroua_noaa_cache_{os.environ.get('USER','user')}.sqlite")
TERRAIN_CACHE_DB = str(Path(LOCAL_TMP) / f"hydroua_terrain_cache_{os.environ.get('USER','user')}.sqlite")

FEMA_NFHL_LAYER28_QUERY = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
NOAA_ATLAS14_ENDPOINT = os.environ.get(
    "NOAA_ATLAS14_ENDPOINT",
    "https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text.csv",
)
USGS_EPQS_ENDPOINT = os.environ.get(
    "USGS_EPQS_ENDPOINT",
    "https://epqs.nationalmap.gov/v1/json",
)

FEMA_TIMEOUT = int(os.environ.get("FEMA_TIMEOUT", "60"))
FEMA_RETRIES = int(os.environ.get("FEMA_RETRIES", "2"))
FEMA_CACHE_TTL_SECONDS = int(os.environ.get("FEMA_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))
FEMA_CACHE_DECIMALS = int(os.environ.get("FEMA_CACHE_DECIMALS", "6"))

NOAA_TIMEOUT = int(os.environ.get("NOAA_TIMEOUT", "45"))
NOAA_RETRIES = int(os.environ.get("NOAA_RETRIES", "2"))
NOAA_CACHE_TTL_SECONDS = int(os.environ.get("NOAA_CACHE_TTL_SECONDS", str(30 * 24 * 3600)))
NOAA_CACHE_DECIMALS = int(os.environ.get("NOAA_CACHE_DECIMALS", "4"))

EPQS_TIMEOUT = int(os.environ.get("EPQS_TIMEOUT", "30"))
EPQS_RETRIES = int(os.environ.get("EPQS_RETRIES", "2"))
EPQS_CACHE_TTL_SECONDS = int(os.environ.get("EPQS_CACHE_TTL_SECONDS", str(30 * 24 * 3600)))
EPQS_CACHE_DECIMALS = int(os.environ.get("EPQS_CACHE_DECIMALS", "6"))

DEBUG_NOAA = os.environ.get("DEBUG_NOAA", "0") == "1"
DEBUG_TERRAIN = os.environ.get("DEBUG_TERRAIN", "0") == "1"

# =========================
# VI quantile thresholds
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

# Small consequence adjustment only
BUILDING_CRITICALITY_ADDON = {
    "residential": 0.0,
    "commercial": 0.0,
    "warehouse": 0.0,
    "industrial": 2.0,
    "multi_family_residential": 2.0,
    "school": 4.0,
    "medical_clinic": 4.0,
    "hospital": 6.0,
}

RISK_BAND_LABELS = [
    (25, "Low Concern"),
    (45, "Moderate Concern"),
    (65, "Elevated Concern"),
    (80, "High Concern"),
    (101, "Severe Concern"),
]


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


def _label_from_score(score: float) -> str:
    for threshold, label in RISK_BAND_LABELS:
        if score < threshold:
            return label
    return "Severe Concern"


# =========================
# Geocode cache
# =========================
def _ensure_geocode_cache():
    con = _open_cache_conn(GEOCODE_CACHE_DB)
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
# NOAA cache
# =========================
def _ensure_noaa_cache():
    con = _open_cache_conn(NOAA_CACHE_DB)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS noaa_cache (
            key TEXT PRIMARY KEY,
            ts INTEGER,
            payload_json TEXT
        )
        """
    )
    return con


def _noaa_cache_key(lat: float, lon: float) -> str:
    return f"{round(float(lat), NOAA_CACHE_DECIMALS)},{round(float(lon), NOAA_CACHE_DECIMALS)}"


def _noaa_cache_get(lat: float, lon: float):
    con = _ensure_noaa_cache()
    k = _noaa_cache_key(lat, lon)
    row = con.execute("SELECT ts, payload_json FROM noaa_cache WHERE key = ?", (k,)).fetchone()
    con.close()
    if not row:
        return None
    ts, payload_json = row
    if (int(time.time()) - int(ts)) > NOAA_CACHE_TTL_SECONDS:
        return None
    return json.loads(payload_json)


def _noaa_cache_put(lat: float, lon: float, payload: dict):
    con = _ensure_noaa_cache()
    k = _noaa_cache_key(lat, lon)
    con.execute(
        "INSERT OR REPLACE INTO noaa_cache(key, ts, payload_json) VALUES (?, ?, ?)",
        (k, int(time.time()), json.dumps(payload)),
    )
    con.close()


# =========================
# Terrain cache
# =========================
def _ensure_terrain_cache():
    con = _open_cache_conn(TERRAIN_CACHE_DB)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS terrain_cache (
            key TEXT PRIMARY KEY,
            ts INTEGER,
            payload_json TEXT
        )
        """
    )
    return con


def _terrain_cache_key(lat: float, lon: float) -> str:
    return f"{round(float(lat), EPQS_CACHE_DECIMALS)},{round(float(lon), EPQS_CACHE_DECIMALS)}"


def _terrain_cache_get(lat: float, lon: float):
    con = _ensure_terrain_cache()
    k = _terrain_cache_key(lat, lon)
    row = con.execute("SELECT ts, payload_json FROM terrain_cache WHERE key = ?", (k,)).fetchone()
    con.close()
    if not row:
        return None
    ts, payload_json = row
    if (int(time.time()) - int(ts)) > EPQS_CACHE_TTL_SECONDS:
        return None
    return json.loads(payload_json)


def _terrain_cache_put(lat: float, lon: float, payload: dict):
    con = _ensure_terrain_cache()
    k = _terrain_cache_key(lat, lon)
    con.execute(
        "INSERT OR REPLACE INTO terrain_cache(key, ts, payload_json) VALUES (?, ?, ?)",
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

    m = re.search(r"\b(?:at|in|near|around|for)\s+(.+)$", raw, re.IGNORECASE)
    if m:
        address = m.group(1).strip().rstrip("?.!")
        return {"address": address, "building_type": building_type}

    m2 = re.search(
        r"\b\d{1,6}\s+[A-Za-z0-9.\- ]+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln|Dr|Drive|Ct|Court|Way|Pl|Place)\b.*",
        raw,
        re.IGNORECASE,
    )
    if m2:
        return {"address": m2.group(0).strip().rstrip("?.!"), "building_type": building_type}

    if any(s in t for s in ["alabama", "tuscaloosa", "birmingham"]):
        return {"address": raw.strip().rstrip("?.!"), "building_type": building_type}

    return {"address": None, "building_type": building_type}


# =========================
# Geocode
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
# FEMA
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
# NOAA Atlas 14
# =========================
_DEFAULT_ARIS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]


def _parse_frequency_header(text: str):
    for line in text.splitlines():
        line_s = line.strip()
        if "ARI" in line_s and "years" in line_s:
            rhs = line_s.split(":", 1)[1] if ":" in line_s else line_s
            vals = []
            for tok in rhs.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    vals.append(int(tok))
            if vals:
                return vals
    return None


def _find_duration_line(text: str, duration_label: str):
    for line in text.splitlines():
        if re.match(rf"^\s*{re.escape(duration_label)}\b", line.strip(), flags=re.IGNORECASE):
            return line.strip()
    return None


def _extract_numeric_values_from_line(line: str):
    if not line:
        return []
    if ":" in line:
        line = line.split(":", 1)[1]

    vals = []
    for tok in line.split(","):
        tok = tok.strip()
        if not tok or tok == "-":
            continue
        tok = tok.strip('"').strip("'")
        try:
            vals.append(float(tok))
        except Exception:
            pass
    return vals


def _safe_get_by_ari(ari_list, value_list, target_ari):
    if not ari_list or not value_list:
        return None
    if len(ari_list) != len(value_list):
        return None
    try:
        idx = ari_list.index(int(target_ari))
        return float(value_list[idx])
    except Exception:
        return None


def _derive_rainfall_risk_bucket(depth_24hr_100yr: float):
    if depth_24hr_100yr is None:
        return {
            "bucket": "unknown",
            "label": "Unknown",
            "engineering_note": "Atlas 14 rainfall depth unavailable.",
        }

    if depth_24hr_100yr < 4.0:
        return {
            "bucket": "low",
            "label": "Low extreme rainfall intensity",
            "engineering_note": "Lower 24-hour extreme rainfall burden at screening level.",
        }
    if depth_24hr_100yr < 6.0:
        return {
            "bucket": "moderate",
            "label": "Moderate extreme rainfall intensity",
            "engineering_note": "Drainage design should account for meaningful extreme-event runoff.",
        }
    if depth_24hr_100yr < 8.0:
        return {
            "bucket": "high",
            "label": "High extreme rainfall intensity",
            "engineering_note": "Large impervious developments may require stronger stormwater controls.",
        }
    return {
        "bucket": "very_high",
        "label": "Very high extreme rainfall intensity",
        "engineering_note": "Stormwater design should explicitly consider major 24-hour extreme rainfall events and runoff management capacity.",
    }


def _noaa_fetch_text(lat: float, lon: float, units: str = "english", series: str = "pds"):
    cached = _noaa_cache_get(lat, lon)
    if cached is not None:
        return cached

    params = {
        "lat": f"{float(lat):.6f}",
        "lon": f"{float(lon):.6f}",
        "data": "depth",
        "units": units,
        "series": series,
    }
    url = f"{NOAA_ATLAS14_ENDPOINT}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "HydroUA-GPT/1.0"})

    last_err = None
    for attempt in range(NOAA_RETRIES + 1):
        try:
            with urlopen(req, timeout=NOAA_TIMEOUT, context=SSL_CONTEXT) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")
            out = {"raw_text": raw_text}
            _noaa_cache_put(lat, lon, out)
            return out
        except Exception as e:
            last_err = e
            time.sleep(min(8, 2 ** attempt))

    return {"error": f"NOAA Atlas 14 request failed after retries. Last error: {last_err}"}


def lookup_noaa_atlas14(lat: float, lon: float):
    q = _noaa_fetch_text(float(lat), float(lon), units="english", series="pds")
    if q.get("error"):
        return {"found": False, "error": q["error"]}

    text = q.get("raw_text", "")
    if not text:
        return {"found": False, "error": "Empty NOAA Atlas 14 response"}

    if DEBUG_NOAA:
        print("NOAA endpoint:", NOAA_ATLAS14_ENDPOINT)
        print("NOAA first 1200 chars:")
        print(text[:1200])

    aris = _parse_frequency_header(text)
    row_line = _find_duration_line(text, "24-hr")
    row_vals = _extract_numeric_values_from_line(row_line)

    if not row_vals:
        return {
            "found": False,
            "error": "Could not parse NOAA Atlas 14 24-hour precipitation frequency row",
        }

    if len(row_vals) == len(_DEFAULT_ARIS):
        if not aris or len(aris) != len(row_vals):
            aris = list(_DEFAULT_ARIS)

    if not aris or len(aris) != len(row_vals):
        return {
            "found": False,
            "error": f"NOAA Atlas 14 parse mismatch for 24-hour row (aris={len(aris) if aris else 0}, values={len(row_vals)})",
            "debug_line": row_line,
        }

    d2 = _safe_get_by_ari(aris, row_vals, 2)
    d10 = _safe_get_by_ari(aris, row_vals, 10)
    d50 = _safe_get_by_ari(aris, row_vals, 50)
    d100 = _safe_get_by_ari(aris, row_vals, 100)

    avg_intensity_100yr = round(d100 / 24.0, 3) if d100 is not None else None
    risk = _derive_rainfall_risk_bucket(d100)

    return {
        "found": True,
        "source": "NOAA Atlas 14",
        "series": "partial_duration",
        "units_depth": "inches",
        "units_intensity": "in/hr",
        "2yr_24hr": d2,
        "10yr_24hr": d10,
        "50yr_24hr": d50,
        "100yr_24hr": d100,
        "avg_intensity_100yr_24hr": avg_intensity_100yr,
        "risk_bucket": risk["bucket"],
        "risk_label": risk["label"],
        "engineering_note": risk["engineering_note"],
    }


# =========================
# SEIV lookup
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


# =========================
# Terrain / drainage context via sampled 3DEP point elevations
# =========================
def _epqs_query_raw(lat: float, lon: float):
    params = {
        "x": f"{float(lon):.8f}",
        "y": f"{float(lat):.8f}",
        "units": "Meters",
        "output": "json",
        "wkid": "4326",
        "includeDate": "false",
    }
    url = f"{USGS_EPQS_ENDPOINT}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "HydroUA-GPT/1.0"})

    last_err = None
    for attempt in range(EPQS_RETRIES + 1):
        try:
            with urlopen(req, timeout=EPQS_TIMEOUT, context=SSL_CONTEXT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(min(8, 2 ** attempt))

    return {"error": f"EPQS request failed after retries. Last error: {last_err}"}


def _epqs_parse_elevation(payload: dict):
    if not isinstance(payload, dict):
        return None

    # v1/json style
    try:
        val = payload.get("value")
        if val is not None:
            return float(val)
    except Exception:
        pass

    # older EPQS style
    try:
        elev = payload["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
        if elev is not None:
            return float(elev)
    except Exception:
        pass

    # alternate style
    try:
        elev = payload["elevation"]
        if elev is not None:
            return float(elev)
    except Exception:
        pass

    return None


def _offset_latlon(lat: float, lon: float, dx_m: float, dy_m: float):
    dlat = dy_m / 111320.0
    dlon = dx_m / (111320.0 * max(0.1, math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def _terrain_sample_points(lat: float, lon: float):
    pts = [("center", 0.0, 0.0)]

    # Inner ring ~30m
    inner_r = 30.0
    # Outer ring ~90m
    outer_r = 90.0
    dirs = [
        ("N", 0.0, 1.0),
        ("NE", math.sqrt(0.5), math.sqrt(0.5)),
        ("E", 1.0, 0.0),
        ("SE", math.sqrt(0.5), -math.sqrt(0.5)),
        ("S", 0.0, -1.0),
        ("SW", -math.sqrt(0.5), -math.sqrt(0.5)),
        ("W", -1.0, 0.0),
        ("NW", -math.sqrt(0.5), math.sqrt(0.5)),
    ]

    for name, ux, uy in dirs:
        pts.append((f"inner_{name}", inner_r * ux, inner_r * uy))
    for name, ux, uy in dirs:
        pts.append((f"outer_{name}", outer_r * ux, outer_r * uy))

    out = []
    for name, dx, dy in pts:
        plat, plon = _offset_latlon(lat, lon, dx, dy)
        out.append({"name": name, "lat": plat, "lon": plon, "radius_m": math.sqrt(dx * dx + dy * dy)})
    return out


def _epqs_elevation_at_point(lat: float, lon: float):
    payload = _epqs_query_raw(lat, lon)
    if payload.get("error"):
        return {"found": False, "error": payload["error"]}
    elev = _epqs_parse_elevation(payload)
    if elev is None:
        return {"found": False, "error": "Could not parse elevation from EPQS response"}
    return {"found": True, "elevation_m": elev}


def analyze_local_terrain(lat: float, lon: float):
    """
    Terrain/drainage screening proxy using point elevations sampled in rings around the site.
    This is a practical step-1 proxy, not full raster flow accumulation.
    """
    cached = _terrain_cache_get(lat, lon)
    if cached is not None:
        return cached

    pts = _terrain_sample_points(lat, lon)

    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_epqs_elevation_at_point, p["lat"], p["lon"]): p for p in pts}
        for fut, p in futs.items():
            try:
                res = fut.result()
            except Exception as e:
                res = {"found": False, "error": str(e)}
            results[p["name"]] = {"meta": p, "res": res}

    center = results["center"]["res"]
    if not center.get("found"):
        out = {"found": False, "error": center.get("error", "Center elevation unavailable")}
        _terrain_cache_put(lat, lon, out)
        return out

    center_elev = float(center["elevation_m"])

    inner_vals = []
    outer_vals = []
    inner_slopes = []

    for name, obj in results.items():
        if name == "center":
            continue
        r = obj["meta"]["radius_m"]
        rr = obj["res"]
        if not rr.get("found"):
            continue
        elev = float(rr["elevation_m"])
        if name.startswith("inner_"):
            inner_vals.append(elev)
            inner_slopes.append(abs((elev - center_elev) / max(1.0, r)) * 100.0)
        elif name.startswith("outer_"):
            outer_vals.append(elev)

    if len(inner_vals) < 5:
        out = {"found": False, "error": "Insufficient valid terrain samples from EPQS"}
        _terrain_cache_put(lat, lon, out)
        return out

    inner_mean = sum(inner_vals) / len(inner_vals)
    outer_mean = (sum(outer_vals) / len(outer_vals)) if outer_vals else inner_mean

    higher_neighbors_inner = sum(1 for v in inner_vals if v > center_elev + 0.25)
    lower_neighbors_inner = sum(1 for v in inner_vals if v < center_elev - 0.25)

    center_minus_inner_mean = center_elev - inner_mean
    center_minus_outer_mean = center_elev - outer_mean

    relief_inner = max(inner_vals) - min(inner_vals) if inner_vals else 0.0
    relief_outer = max(outer_vals) - min(outer_vals) if outer_vals else relief_inner
    mean_abs_slope_pct_inner = sum(inner_slopes) / len(inner_slopes) if inner_slopes else 0.0

    depression_flag = (higher_neighbors_inner >= 6 and center_minus_inner_mean <= -0.75)
    convergence_flag = (higher_neighbors_inner >= 5 and center_minus_inner_mean <= -0.40)
    ridge_flag = (lower_neighbors_inner >= 5 and center_minus_inner_mean >= 0.75)
    flat_flag = mean_abs_slope_pct_inner < 1.0

    if depression_flag:
        terrain_class = "Locally depressional / potential ponding setting"
    elif convergence_flag and flat_flag:
        terrain_class = "Flat, convergent local terrain"
    elif ridge_flag:
        terrain_class = "Locally elevated / divergent setting"
    elif flat_flag:
        terrain_class = "Flat to gently sloping terrain"
    else:
        terrain_class = "Mixed local terrain"

    out = {
        "found": True,
        "source": "USGS 3DEP EPQS ring-sampled point elevations",
        "center_elevation_m": round(center_elev, 2),
        "inner_ring_mean_elevation_m": round(inner_mean, 2),
        "outer_ring_mean_elevation_m": round(outer_mean, 2),
        "center_minus_inner_mean_m": round(center_minus_inner_mean, 2),
        "center_minus_outer_mean_m": round(center_minus_outer_mean, 2),
        "relief_inner_m": round(relief_inner, 2),
        "relief_outer_m": round(relief_outer, 2),
        "mean_abs_slope_pct_inner": round(mean_abs_slope_pct_inner, 2),
        "higher_neighbors_inner": int(higher_neighbors_inner),
        "lower_neighbors_inner": int(lower_neighbors_inner),
        "depression_flag": bool(depression_flag),
        "convergence_flag": bool(convergence_flag),
        "ridge_flag": bool(ridge_flag),
        "flat_flag": bool(flat_flag),
        "terrain_class": terrain_class,
    }

    if DEBUG_TERRAIN:
        print("TERRAIN DEBUG:", json.dumps(out, indent=2))

    _terrain_cache_put(lat, lon, out)
    return out


# =========================
# Composite scoring
# =========================
def score_fema_component(fema: dict) -> float:
    # Range: 0 to 25
    if not fema.get("found"):
        return 8.0
    tier = int(fema.get("tier", 2))
    mapping = {
        1: 3.0,
        2: 8.0,
        3: 15.0,
        4: 21.0,
        5: 25.0,
    }
    return mapping.get(tier, 8.0)


def score_rainfall_component(noaa: dict) -> float:
    # Range: 0 to 20
    if not noaa.get("found"):
        return 8.0

    d100 = noaa.get("100yr_24hr")
    if d100 is None:
        return 8.0

    if d100 < 4.0:
        return 4.0
    if d100 < 6.0:
        return 8.0
    if d100 < 8.0:
        return 13.0
    if d100 < 10.0:
        return 17.0
    return 20.0


def score_vi_component(vi: dict) -> float:
    # Range: 0 to 10
    if not vi.get("found"):
        return 4.0

    tier = int(vi.get("vi_tier", 3))
    mapping = {
        1: 1.0,
        2: 3.0,
        3: 5.0,
        4: 8.0,
        5: 10.0,
    }
    return mapping.get(tier, 4.0)


def score_terrain_component(terrain: dict) -> float:
    """
    Terrain/drainage context score.
    Range: 0 to 25
    """
    if not terrain.get("found"):
        return 8.0

    score = 4.0

    delta_inner = float(terrain.get("center_minus_inner_mean_m", 0.0))
    slope_pct = float(terrain.get("mean_abs_slope_pct_inner", 0.0))

    if terrain.get("depression_flag"):
        score += 9.0
    elif terrain.get("convergence_flag"):
        score += 6.0

    if slope_pct < 1.0:
        score += 5.0
    elif slope_pct < 2.0:
        score += 3.0
    elif slope_pct < 4.0:
        score += 1.0

    if delta_inner <= -1.5:
        score += 7.0
    elif delta_inner <= -0.75:
        score += 4.0
    elif delta_inner >= 1.0 and terrain.get("ridge_flag"):
        score -= 2.0

    return max(0.0, min(25.0, round(score, 1)))


def compute_composite_site_risk(building_type: str, fema: dict, noaa: dict, vi: dict, terrain: dict) -> dict:
    """
    Step-1 upgraded screening model:
    - FEMA + NOAA = physical hazard
    - Terrain = local site/drainage context
    - SEIV = vulnerability/consequence context
    - building type = small consequence-sensitive add-on
    """
    fema_score = score_fema_component(fema)        # /25
    noaa_score = score_rainfall_component(noaa)    # /20
    terrain_score = score_terrain_component(terrain)  # /25
    vi_score = score_vi_component(vi)              # /10
    criticality_addon = BUILDING_CRITICALITY_ADDON.get(building_type, 0.0)  # /6

    base_score = fema_score + noaa_score + terrain_score + vi_score

    interaction_bonus = 0.0

    # High rainfall + convergent/depressional terrain
    if noaa.get("found") and terrain.get("found"):
        d100 = noaa.get("100yr_24hr")
        if d100 is not None and d100 >= 8.0:
            if terrain.get("depression_flag"):
                interaction_bonus += 8.0
            elif terrain.get("convergence_flag"):
                interaction_bonus += 5.0

    # FEMA high + terrain low spot
    if fema.get("found") and terrain.get("found"):
        if int(fema.get("tier", 2)) >= 4 and terrain.get("depression_flag"):
            interaction_bonus += 8.0
        elif int(fema.get("tier", 2)) >= 4 and terrain.get("convergence_flag"):
            interaction_bonus += 5.0

    # High vulnerability amplifies consequence, but modestly
    if vi.get("found") and terrain.get("found"):
        if int(vi.get("vi_tier", 3)) >= 4 and terrain.get("depression_flag"):
            interaction_bonus += 3.0
        elif int(vi.get("vi_tier", 3)) >= 4 and terrain.get("convergence_flag"):
            interaction_bonus += 2.0

    final_score = min(100.0, base_score + interaction_bonus + criticality_addon)
    label = _label_from_score(final_score)

    drivers = []
    if fema_score >= 16:
        drivers.append("mapped flood hazard")
    if noaa_score >= 13:
        drivers.append("extreme rainfall intensity")
    if terrain_score >= 16:
        drivers.append("low-lying / convergent local terrain")
    if vi_score >= 8:
        drivers.append("high socio-economic-infrastructure vulnerability")
    if criticality_addon >= 4:
        drivers.append("critical facility sensitivity")
    if interaction_bonus > 0:
        drivers.append("compounding hazard-terrain-vulnerability effects")

    recommendations = []
    if final_score >= 45:
        recommendations.append("Site-specific engineering study required")
    if terrain_score >= 16:
        recommendations.append("Detailed site grading, local drainage, and ponding analysis recommended")
    if noaa_score >= 13:
        recommendations.append("Stormwater and drainage design should be checked against extreme-event runoff")
    if fema_score >= 16:
        recommendations.append("Floodplain review required if applicable; verify FEMA panel and local ordinance")
    if vi_score >= 8:
        recommendations.append("Consider access, continuity-of-operations, and emergency service reliability")
    if building_type in {"hospital", "medical_clinic", "school"}:
        recommendations.append("Critical facility standard review required")
    if not recommendations:
        recommendations.append("Verify local drainage, grading, and stormwater compliance before design.")

    return {
        "fema_score": round(fema_score, 1),
        "noaa_score": round(noaa_score, 1),
        "terrain_score": round(terrain_score, 1),
        "vi_score": round(vi_score, 1),
        "base_score": round(base_score, 1),
        "interaction_bonus": round(interaction_bonus, 1),
        "criticality_addon": round(criticality_addon, 1),
        "final_score": round(final_score, 1),
        "label": label,
        "drivers": drivers,
        "recommendations": recommendations,
    }
