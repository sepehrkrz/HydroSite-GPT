"""
True multi-agent LangGraph workflow for HydroUA-GPT.

This replaces the single procedural advisor pipeline with explicit cooperating agents:

1) Supervisor/Router Agent
2) Planner Agent
3) Geocoder Agent
4) FEMA Flood Hazard Agent
5) Rainfall/Hydrology Agent
6) Terrain/Drainage Agent
7) SEIV Vulnerability Agent
8) Scoring/Synthesis Agent
9) Critic/Verifier Agent
10) Report Writer Agent

The agents are intentionally grounded in deterministic tools. The LLM is used for routing fallback,
slot extraction, domain interpretation, and final explanation, but the hazard data and scores come
from the existing FEMA/NOAA/SEIV/terrain/scoring functions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

import router
import llm_router

from advisor_utils import (
    clean_text,
    definition_handler,
    extract_address_from_text,
    infer_building_type_from_text,
    last_address_from_history,
    last_building_from_history,
    fmt_inches,
    fmt_intensity,
)
from advisor_llm import (
    llm_fill_slots,
    interpret_rainfall_risk_llm,
    interpret_terrain_risk_llm,
    interpret_site_risk_llm,
)
from advisor_tools import (
    geocode_to_point,
    lookup_fema_flood_hazard,
    lookup_vulnerability_index,
    lookup_noaa_atlas14,
    analyze_local_terrain,
    compute_composite_site_risk,
)


WEATHER_Q = re.compile(r"\b(will it rain|rain tonight|rain tomorrow|forecast|weather)\b", re.IGNORECASE)
FLOOD_FORECAST_Q = re.compile(r"\b(will it flood|flood tonight|flood tomorrow|flood anytime soon)\b", re.IGNORECASE)
FOLLOWUP_INHERIT_RE = re.compile(
    r"^\s*(please\s+)?(explain|define)\b.*\b(example|examples|more|again|better|elaborately)\b\s*$"
    r"|^\s*(example|examples|with example|with examples|give an example|give examples)\s*$",
    re.IGNORECASE,
)
YES_FOLLOWUP_RE = re.compile(r"^\s*(yes|yeah|yep|ok|okay|sure)\b", re.IGNORECASE)

SAFE_FORECAST_RESPONSE = (
    "I can explain rainfall/flooding and how forecasts work, but I don't have a guaranteed real-time feed here.\n\n"
    "Tell me your location (city/ZIP) and I'll tell you what to check (NWS forecast, probability of precipitation, "
    "watches/warnings, river gauge forecast points) and how to interpret it. "
    "If you paste a forecast here, I can interpret it."
)

OFFTOPIC_RESPONSE = (
    "I can help with hydrology questions and build-site feasibility screening.\n\n"
    "For build-site screening, include a building type + location/address.\n"
    "Example: 'Can I build a warehouse at 123 Main St, Tuscaloosa AL?'"
)


class HydroAgentState(TypedDict, total=False):
    # Inputs
    user_message: str
    history: List[Dict[str, str]]

    # Supervisor routing
    inherited_intent: str
    route_rule: str
    intent: str
    stream_required: bool

    # Planner output
    plan: Dict[str, Any]
    agent_trace: List[str]
    definition_answer: str
    building_type: Optional[str]
    address: Optional[str]
    mode: str
    slots: Dict[str, Any]
    needs_fema: bool
    needs_noaa: bool
    needs_terrain: bool
    needs_vi: bool
    needs_scoring: bool

    # Geocoder output
    lat: Optional[float]
    lon: Optional[float]
    geoid10: str
    match_quality: str

    # Specialist agent outputs
    fema: Dict[str, Any]
    fema_analysis: Dict[str, Any]
    noaa: Dict[str, Any]
    rainfall_analysis: Dict[str, Any]
    terrain: Dict[str, Any]
    terrain_analysis: Dict[str, Any]
    vi: Dict[str, Any]
    vulnerability_analysis: Dict[str, Any]

    # Synthesis / critic / report
    risk: Dict[str, Any]
    scoring_analysis: Dict[str, Any]
    rainfall_interp: str
    terrain_interp: str
    integrated_interp: str
    verifier_report: Dict[str, Any]
    final_answer: str
    map_lat: Optional[float]
    map_lon: Optional[float]
    error: str


def _append_trace(state: HydroAgentState, message: str) -> List[str]:
    return list(state.get("agent_trace") or []) + [message]


def _last_user_intent(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return router.route_intent_rules(m.get("content", ""))
    return ""


# -----------------------------------------------------------------------------
# 1) Supervisor / Router Agent
# -----------------------------------------------------------------------------
def supervisor_router_agent(state: HydroAgentState) -> Dict[str, Any]:
    """
    Decides which high-level agent path should handle the user request.
    Uses fast rules first; LLM router can only upgrade offtopic -> hydrology/advisor.
    """
    q = (state.get("user_message") or "").strip()
    history = state.get("history") or []
    inherited = _last_user_intent(history)

    if FOLLOWUP_INHERIT_RE.match(q) and inherited in {"hydrology", "advisor"}:
        return {
            "inherited_intent": inherited,
            "route_rule": inherited,
            "intent": inherited,
            "agent_trace": _append_trace(state, f"Supervisor/Router Agent: inherited intent '{inherited}'."),
        }

    route_rule = router.route_intent_rules(q)
    route_final = route_rule
    warning = ""

    if route_rule == "offtopic":
        try:
            llm_choice = llm_router.route_intent_llm(q, history)
            if llm_choice in {"hydrology", "advisor"}:
                route_final = llm_choice
        except Exception as e:
            warning = f"LLM router failed; continued with rule-based routing: {e}"

    if route_final == "offtopic" and inherited == "hydrology" and (YES_FOLLOWUP_RE.match(q) or len(q) <= 28):
        route_final = "hydrology"

    out = {
        "inherited_intent": inherited,
        "route_rule": route_rule,
        "intent": route_final,
        "agent_trace": _append_trace(state, f"Supervisor/Router Agent: rule='{route_rule}', final='{route_final}'."),
    }
    if warning:
        out["error"] = warning
    return out


def route_after_supervisor(state: HydroAgentState) -> str:
    return state.get("intent", "offtopic")


# -----------------------------------------------------------------------------
# Hydrology and offtopic paths
# -----------------------------------------------------------------------------
def hydrology_chat_agent_gate(state: HydroAgentState) -> Dict[str, Any]:
    """
    Gates hydrology chat. Real-time weather/flood forecasts are guarded because the local
    model has no guaranteed live feed. Other hydrology questions stream through hydrology_chat.py.
    """
    q = state.get("user_message") or ""
    if WEATHER_Q.search(q) or FLOOD_FORECAST_Q.search(q):
        return {
            "final_answer": SAFE_FORECAST_RESPONSE,
            "stream_required": False,
            "agent_trace": _append_trace(state, "Hydrology Chat Agent: forecast guard returned safe response."),
        }

    return {
        "final_answer": "",
        "stream_required": True,
        "agent_trace": _append_trace(state, "Hydrology Chat Agent: delegated to streaming hydrology model."),
    }


def offtopic_agent(state: HydroAgentState) -> Dict[str, Any]:
    return {
        "final_answer": OFFTOPIC_RESPONSE,
        "stream_required": False,
        "agent_trace": _append_trace(state, "Off-topic Agent: returned scope reminder."),
    }


# -----------------------------------------------------------------------------
# 2) Planner Agent
# -----------------------------------------------------------------------------
def planner_agent(state: HydroAgentState) -> Dict[str, Any]:
    """
    Extracts required site inputs and decides which specialist agents must run.
    This is the multi-agent supervisor for the advisor path.
    """
    history = state.get("history") or []
    raw = clean_text(state.get("user_message") or "")

    defn = definition_handler(raw)
    if defn:
        return {
            "definition_answer": defn,
            "final_answer": defn,
            "stream_required": False,
            "agent_trace": _append_trace(state, "Planner Agent: detected definition-only request; no site tools needed."),
        }

    slots = llm_fill_slots(raw, history)
    building_type = slots.get("building_type") or infer_building_type_from_text(raw)
    address = clean_text(slots.get("address") or "")

    addr_from_raw = extract_address_from_text(raw)
    if addr_from_raw:
        address = addr_from_raw

    if building_type and not address:
        prev_addr = last_address_from_history(history)
        if prev_addr:
            address = prev_addr

    if address and not building_type:
        prev_bt = last_building_from_history(history)
        if prev_bt:
            building_type = prev_bt

    if building_type and not address:
        answer = f"What is the location/address for the {building_type}? Example: 302 Reed St, Tuscaloosa AL"
        return {
            "slots": slots,
            "building_type": building_type,
            "address": address,
            "final_answer": answer,
            "stream_required": False,
            "agent_trace": _append_trace(state, "Planner Agent: missing address; requested follow-up."),
        }

    if address and not building_type:
        answer = f"What building type is it at {address} (warehouse, home, hospital, school, club, etc.)?"
        return {
            "slots": slots,
            "building_type": building_type,
            "address": address,
            "final_answer": answer,
            "stream_required": False,
            "agent_trace": _append_trace(state, "Planner Agent: missing building type; requested follow-up."),
        }

    if not building_type and not address:
        answer = slots.get("followup_question", "What building type and what location/address should I screen?")
        return {
            "slots": slots,
            "final_answer": answer,
            "stream_required": False,
            "agent_trace": _append_trace(state, "Planner Agent: missing building type and address; requested follow-up."),
        }

    mode = slots.get("mode", "block")
    if mode in {"city", "state"} and address and not any(ch.isdigit() for ch in address):
        answer = (
            f"I can do a coarse screening for: {address}\n"
            "But block-level vulnerability works best with a street address.\n\n"
            "Reply with a street address (best) OR reply 'use city-level' to proceed (FEMA-only)."
        )
        return {
            "slots": slots,
            "building_type": building_type,
            "address": address,
            "mode": mode,
            "final_answer": answer,
            "stream_required": False,
            "agent_trace": _append_trace(state, "Planner Agent: city/state-only address; requested street address or city-level confirmation."),
        }

    plan = {
        "goal": "screen_build_site_feasibility",
        "building_type": building_type,
        "address": address,
        "agents": [
            "geocoder_agent",
            "fema_flood_hazard_agent",
            "rainfall_hydrology_agent",
            "terrain_drainage_agent",
            "seiv_vulnerability_agent",
            "scoring_synthesis_agent",
            "critic_verifier_agent",
            "report_writer_agent",
        ],
        "rationale": (
            "A defensible screening needs geocoding, mapped flood hazard, extreme rainfall, "
            "local terrain/drainage context, vulnerability/consequence context, composite scoring, "
            "verification, and a final report."
        ),
    }

    return {
        "slots": slots,
        "building_type": building_type,
        "address": address,
        "mode": mode,
        "plan": plan,
        "needs_fema": True,
        "needs_noaa": True,
        "needs_terrain": True,
        "needs_vi": True,
        "needs_scoring": True,
        "final_answer": "",
        "stream_required": False,
        "agent_trace": _append_trace(state, "Planner Agent: created specialist execution plan."),
    }


def advisor_after_planner(state: HydroAgentState) -> str:
    if (state.get("final_answer") or "").strip():
        return "done"
    return "geocode"


# -----------------------------------------------------------------------------
# 3) Geocoder Agent
# -----------------------------------------------------------------------------
def geocoder_agent(state: HydroAgentState) -> Dict[str, Any]:
    raw = clean_text(state.get("user_message") or "")
    history = state.get("history") or []
    address = state.get("address") or ""

    geo = geocode_to_point(address)
    if geo.get("error"):
        slots2 = llm_fill_slots(raw, history, geocode_error=geo["error"])
        addr_from_raw = extract_address_from_text(raw)
        new_addr = extract_address_from_text(slots2.get("address", "")) or addr_from_raw
        if new_addr and new_addr != address and not slots2.get("needs_followup", False):
            address = new_addr
            geo = geocode_to_point(address)

    if geo.get("error"):
        return {
            "address": address,
            "final_answer": f"I could not geocode the address: {geo['error']}",
            "stream_required": False,
            "map_lat": None,
            "map_lon": None,
            "agent_trace": _append_trace(state, "Geocoder Agent: geocoding failed."),
        }

    lat = geo.get("lat")
    lon = geo.get("lon")
    return {
        "address": address,
        "lat": lat,
        "lon": lon,
        "geoid10": geo.get("geoid10") or "",
        "match_quality": geo.get("match_quality") or "",
        "map_lat": lat,
        "map_lon": lon,
        "agent_trace": _append_trace(state, f"Geocoder Agent: resolved site to lat/lon=({lat}, {lon})."),
    }


def advisor_after_geocode(state: HydroAgentState) -> str:
    if (state.get("final_answer") or "").strip():
        return "done"
    return "fema"


# -----------------------------------------------------------------------------
# 4) FEMA Flood Hazard Agent
# -----------------------------------------------------------------------------
def fema_flood_hazard_agent(state: HydroAgentState) -> Dict[str, Any]:
    if not state.get("needs_fema", True):
        return {"agent_trace": _append_trace(state, "FEMA Flood Hazard Agent: skipped by plan.")}

    fema = lookup_fema_flood_hazard(state["lat"], state["lon"])
    if fema.get("found"):
        tier = fema.get("tier")
        label = fema.get("tier_label")
        zone = fema.get("zone") or "N/A"
        summary = f"FEMA zone {zone}; tier {tier} ({label})."
    else:
        summary = f"FEMA unavailable: {fema.get('error')}"

    return {
        "fema": fema,
        "fema_analysis": {
            "agent": "FEMA Flood Hazard Agent",
            "found": bool(fema.get("found")),
            "summary": summary,
            "confidence": "medium" if fema.get("found") else "low",
        },
        "agent_trace": _append_trace(state, "FEMA Flood Hazard Agent: completed mapped flood hazard lookup."),
    }


# -----------------------------------------------------------------------------
# 5) Rainfall / Hydrology Agent
# -----------------------------------------------------------------------------
def rainfall_hydrology_agent(state: HydroAgentState) -> Dict[str, Any]:
    if not state.get("needs_noaa", True):
        return {"agent_trace": _append_trace(state, "Rainfall/Hydrology Agent: skipped by plan.")}

    noaa = lookup_noaa_atlas14(state["lat"], state["lon"])
    rainfall_interp = interpret_rainfall_risk_llm(
        noaa if noaa.get("found") else {},
        building_type=state.get("building_type") or "",
        address=state.get("address") or "",
    )

    if noaa.get("found"):
        summary = (
            f"100-year 24-hour rainfall = {fmt_inches(noaa.get('100yr_24hr'))} inches; "
            f"class = {noaa.get('risk_label') or 'N/A'}."
        )
    else:
        summary = f"NOAA Atlas 14 unavailable: {noaa.get('error')}"

    return {
        "noaa": noaa,
        "rainfall_interp": rainfall_interp,
        "rainfall_analysis": {
            "agent": "Rainfall/Hydrology Agent",
            "found": bool(noaa.get("found")),
            "summary": summary,
            "interpretation": rainfall_interp,
            "confidence": "medium" if noaa.get("found") else "low",
        },
        "agent_trace": _append_trace(state, "Rainfall/Hydrology Agent: completed Atlas 14 analysis."),
    }


# -----------------------------------------------------------------------------
# 6) Terrain / Drainage Agent
# -----------------------------------------------------------------------------
def terrain_drainage_agent(state: HydroAgentState) -> Dict[str, Any]:
    if not state.get("needs_terrain", True):
        return {"agent_trace": _append_trace(state, "Terrain/Drainage Agent: skipped by plan.")}

    terrain = analyze_local_terrain(state["lat"], state["lon"])
    terrain_interp = interpret_terrain_risk_llm(
        terrain if terrain.get("found") else {},
        building_type=state.get("building_type") or "",
        address=state.get("address") or "",
    )

    if terrain.get("found"):
        summary = (
            f"Terrain class = {terrain.get('terrain_class')}; "
            f"center-minus-inner = {terrain.get('center_minus_inner_mean_m')} m; "
            f"depression={terrain.get('depression_flag')}; convergence={terrain.get('convergence_flag')}."
        )
    else:
        summary = f"Terrain unavailable: {terrain.get('error')}"

    return {
        "terrain": terrain,
        "terrain_interp": terrain_interp,
        "terrain_analysis": {
            "agent": "Terrain/Drainage Agent",
            "found": bool(terrain.get("found")),
            "summary": summary,
            "interpretation": terrain_interp,
            "confidence": "medium" if terrain.get("found") else "low",
        },
        "agent_trace": _append_trace(state, "Terrain/Drainage Agent: completed local drainage-context analysis."),
    }


# -----------------------------------------------------------------------------
# 7) SEIV Vulnerability Agent
# -----------------------------------------------------------------------------
def seiv_vulnerability_agent(state: HydroAgentState) -> Dict[str, Any]:
    if not state.get("needs_vi", True):
        return {"agent_trace": _append_trace(state, "SEIV Vulnerability Agent: skipped by plan.")}

    vi = lookup_vulnerability_index(state.get("geoid10") or "")
    if vi.get("found"):
        summary = f"SEIV tier {vi.get('vi_tier')} ({vi.get('vi_label')}); VI={vi.get('vi')}."
    else:
        summary = f"SEIV unavailable: {vi.get('error')}"

    return {
        "vi": vi,
        "vulnerability_analysis": {
            "agent": "SEIV Vulnerability Agent",
            "found": bool(vi.get("found")),
            "summary": summary,
            "confidence": "medium" if vi.get("found") else "low",
        },
        "agent_trace": _append_trace(state, "SEIV Vulnerability Agent: completed block-level vulnerability lookup."),
    }


# -----------------------------------------------------------------------------
# 8) Scoring / Synthesis Agent
# -----------------------------------------------------------------------------
def scoring_synthesis_agent(state: HydroAgentState) -> Dict[str, Any]:
    risk = compute_composite_site_risk(
        building_type=state.get("building_type") or "",
        fema=state.get("fema") or {},
        noaa=state.get("noaa") or {},
        vi=state.get("vi") or {},
        terrain=state.get("terrain") or {},
    )

    integrated_interp = interpret_site_risk_llm(
        building_type=state.get("building_type") or "",
        address=state.get("address") or "",
        fema=state.get("fema") or {},
        noaa=state.get("noaa") or {},
        vi=state.get("vi") or {},
        terrain=state.get("terrain") or {},
        risk=risk,
    )

    scoring_analysis = {
        "agent": "Scoring/Synthesis Agent",
        "final_score": risk.get("final_score"),
        "label": risk.get("label"),
        "drivers": risk.get("drivers", []),
        "recommendations": risk.get("recommendations", []),
        "interpretation": integrated_interp,
    }

    return {
        "risk": risk,
        "integrated_interp": integrated_interp,
        "scoring_analysis": scoring_analysis,
        "agent_trace": _append_trace(state, "Scoring/Synthesis Agent: computed composite risk and integrated interpretation."),
    }


# -----------------------------------------------------------------------------
# 9) Critic / Verifier Agent
# -----------------------------------------------------------------------------
def critic_verifier_agent(state: HydroAgentState) -> Dict[str, Any]:
    """
    Checks whether the report has enough evidence and flags missing/weak components.
    This does not block the report; it adds warnings and prevents overclaiming.
    """
    checks = []
    warnings = []

    if state.get("lat") is not None and state.get("lon") is not None:
        checks.append("Geocoding produced coordinates.")
    else:
        warnings.append("Geocoding did not produce coordinates.")

    if (state.get("fema") or {}).get("found"):
        checks.append("FEMA flood hazard lookup succeeded.")
    else:
        warnings.append("FEMA flood hazard was unavailable or did not return a usable record.")

    if (state.get("noaa") or {}).get("found"):
        checks.append("NOAA Atlas 14 rainfall lookup succeeded.")
    else:
        warnings.append("NOAA Atlas 14 rainfall was unavailable or could not be parsed.")

    if (state.get("terrain") or {}).get("found"):
        checks.append("Terrain/drainage context lookup succeeded.")
    else:
        warnings.append("Terrain/drainage context was unavailable.")

    if (state.get("vi") or {}).get("found"):
        checks.append("SEIV vulnerability lookup succeeded.")
    else:
        warnings.append("SEIV vulnerability was unavailable for the geocoded block.")

    risk = state.get("risk") or {}
    if isinstance(risk.get("final_score"), (int, float)):
        checks.append("Composite risk score was computed.")
    else:
        warnings.append("Composite risk score was not computed.")

    if state.get("building_type") in {"hospital", "medical_clinic", "school"}:
        checks.append("Critical/sensitive facility status was considered in scoring and recommendations.")

    verifier_report = {
        "agent": "Critic/Verifier Agent",
        "passed": len(warnings) == 0,
        "checks": checks,
        "warnings": warnings,
        "reporting_instruction": (
            "Use screening-level language. Do not make final permitting/build/no-build claims. "
            "Mention unavailable components explicitly."
        ),
    }

    return {
        "verifier_report": verifier_report,
        "agent_trace": _append_trace(state, f"Critic/Verifier Agent: completed verification with {len(warnings)} warning(s)."),
    }


# -----------------------------------------------------------------------------
# 10) Report Writer Agent
# -----------------------------------------------------------------------------
def report_writer_agent(state: HydroAgentState) -> Dict[str, Any]:
    building_type = state.get("building_type") or "proposed building"
    address = state.get("address") or "the selected site"
    lat = state.get("lat")
    lon = state.get("lon")
    geoid10 = state.get("geoid10") or ""
    fema = state.get("fema") or {}
    vi = state.get("vi") or {}
    noaa = state.get("noaa") or {}
    terrain = state.get("terrain") or {}
    risk = state.get("risk") or {}
    verifier = state.get("verifier_report") or {}

    fema_ok = bool(fema.get("found"))
    vi_ok = bool(vi.get("found"))
    noaa_ok = bool(noaa.get("found"))
    terrain_ok = bool(terrain.get("found"))

    lines = [
        f"Screening result for the proposed {building_type} at {address}:",
        "",
        "Multi-agent workflow used:",
        "- Supervisor/Router Agent",
        "- Planner Agent",
        "- Geocoder Agent",
        "- FEMA Flood Hazard Agent",
        "- Rainfall/Hydrology Agent",
        "- Terrain/Drainage Agent",
        "- SEIV Vulnerability Agent",
        "- Scoring/Synthesis Agent",
        "- Critic/Verifier Agent",
        "- Report Writer Agent",
        "",
        f"Coordinates: ({lat}, {lon})",
        f"GEOID10 (Census Block): {geoid10 if geoid10 else 'N/A'}",
        "",
        "Planner decision:",
        f"- Building type: {building_type}",
        f"- Address/location: {address}",
        f"- Specialist agents selected: {', '.join((state.get('plan') or {}).get('agents', [])) or 'N/A'}",
        "",
        "Flood hazard analysis — FEMA Flood Hazard Agent:",
    ]

    if fema_ok:
        lines += [
            f"FEMA zone: {fema.get('zone') or 'N/A'}",
            f"FEMA subtype: {fema.get('zone_subtype') or 'N/A'}",
            f"SFHA flag (1% annual chance): {fema.get('sfha_tf') or 'N/A'}",
            f"FEMA tier: {fema.get('tier', 'N/A')} ({fema.get('tier_label', 'N/A')})",
        ]
    else:
        lines += [f"FEMA status: unavailable ({fema.get('error')})"]

    lines += ["", "Rainfall intensity analysis — Rainfall/Hydrology Agent:"]
    if noaa_ok:
        lines += [
            f"2-year 24-hour rainfall: {fmt_inches(noaa.get('2yr_24hr'))} inches",
            f"10-year 24-hour rainfall: {fmt_inches(noaa.get('10yr_24hr'))} inches",
            f"50-year 24-hour rainfall: {fmt_inches(noaa.get('50yr_24hr'))} inches",
            f"100-year 24-hour rainfall: {fmt_inches(noaa.get('100yr_24hr'))} inches",
            f"Estimated average intensity (100-year storm over 24 hours): {fmt_intensity(noaa.get('avg_intensity_100yr_24hr'))} inches/hour",
            f"Rainfall intensity class: {noaa.get('risk_label') or 'N/A'}",
        ]
    else:
        lines += [f"NOAA Atlas 14 status: unavailable ({noaa.get('error')})"]

    lines += ["", "Terrain and local drainage analysis — Terrain/Drainage Agent:"]
    if terrain_ok:
        lines += [
            f"Center elevation: {terrain.get('center_elevation_m', 'N/A')} m",
            f"Center minus inner-ring mean elevation: {terrain.get('center_minus_inner_mean_m', 'N/A')} m",
            f"Inner-ring local relief: {terrain.get('relief_inner_m', 'N/A')} m",
            f"Mean local slope proxy (inner ring): {terrain.get('mean_abs_slope_pct_inner', 'N/A')} %",
            f"Higher nearby sample points (inner ring): {terrain.get('higher_neighbors_inner', 'N/A')} / 8",
            f"Terrain class: {terrain.get('terrain_class', 'N/A')}",
            f"Depression flag: {terrain.get('depression_flag')}",
            f"Convergence flag: {terrain.get('convergence_flag')}",
        ]
    else:
        lines += [f"Terrain status: unavailable ({terrain.get('error')})"]

    lines += ["", "Socio-Economic-Infrastructure Vulnerability — SEIV Vulnerability Agent:"]
    if vi_ok:
        lines += [
            f"Vulnerability_Index: {vi.get('vi')} (scaled ~ {vi.get('vi_pct', 0):.2f})",
            f"VI tier: {vi.get('vi_tier', 'N/A')} ({vi.get('vi_label', 'N/A')})",
        ]
    else:
        lines += [f"VI lookup: not available ({vi.get('error')})"]

    lines += ["", "Agent interpretations:"]
    lines += ["Rainfall/Hydrology Agent:", state.get("rainfall_interp") or "N/A"]
    lines += ["", "Terrain/Drainage Agent:", state.get("terrain_interp") or "N/A"]

    lines += [
        "",
        "Composite Site Risk Assessment — Scoring/Synthesis Agent:",
        f"FEMA hazard component score: {risk.get('fema_score', 'N/A')}/25",
        f"Rainfall intensity component score: {risk.get('noaa_score', 'N/A')}/20",
        f"Terrain/drainage context score: {risk.get('terrain_score', 'N/A')}/25",
        f"SEIV vulnerability component score: {risk.get('vi_score', 'N/A')}/10",
        f"Base combined score: {risk.get('base_score', 'N/A')}/80",
        f"Interaction bonus: {risk.get('interaction_bonus', 'N/A')}/14",
        f"Building criticality add-on: {risk.get('criticality_addon', 'N/A')}/6",
        f"Final composite site risk score: {risk.get('final_score', 'N/A')}/100",
        f"Assessment label: {risk.get('label', 'N/A')}",
    ]

    if risk.get("drivers"):
        lines += ["", "Main risk drivers:"]
        lines += [f"- {d}" for d in risk["drivers"]]

    lines += ["", "Integrated interpretation — Scoring/Synthesis Agent:", state.get("integrated_interp") or "N/A"]

    lines += ["", "Critic/Verifier Agent:"]
    lines += [f"Verification passed: {verifier.get('passed', False)}"]
    if verifier.get("checks"):
        lines += ["Checks passed:"]
        lines += [f"- {c}" for c in verifier.get("checks", [])]
    if verifier.get("warnings"):
        lines += ["Warnings:"]
        lines += [f"- {w}" for w in verifier.get("warnings", [])]

    lines += ["", "Recommended next steps:"]
    lines += [f"- {t}" for t in risk.get("recommendations", [])]

    if fema_ok and fema.get("fema_notes"):
        lines += ["", "FEMA notes:"]
        lines += [f"- {n}" for n in fema["fema_notes"]]

    if noaa_ok and noaa.get("engineering_note"):
        lines += ["", "NOAA engineering note:"]
        lines.append(f"- {noaa['engineering_note']}")

    lines += [
        "",
        "Agent trace:",
        *[f"- {t}" for t in state.get("agent_trace", [])],
        "",
        "Limitation: Screening-level only. This multi-agent workflow improves organization, cross-checking, and explainability, but it is still not a substitute for full DEM-based flow accumulation, detailed grading design, storm sewer analysis, permitting review, or hydrologic/hydraulic modeling.",
    ]

    return {
        "final_answer": "\n".join(lines),
        "stream_required": False,
        "agent_trace": _append_trace(state, "Report Writer Agent: generated final multi-agent screening report."),
    }


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(HydroAgentState)

    graph.add_node("supervisor_router_agent", supervisor_router_agent)
    graph.add_node("hydrology_chat_agent_gate", hydrology_chat_agent_gate)
    graph.add_node("offtopic_agent", offtopic_agent)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("geocoder_agent", geocoder_agent)
    graph.add_node("fema_flood_hazard_agent", fema_flood_hazard_agent)
    graph.add_node("rainfall_hydrology_agent", rainfall_hydrology_agent)
    graph.add_node("terrain_drainage_agent", terrain_drainage_agent)
    graph.add_node("seiv_vulnerability_agent", seiv_vulnerability_agent)
    graph.add_node("scoring_synthesis_agent", scoring_synthesis_agent)
    graph.add_node("critic_verifier_agent", critic_verifier_agent)
    graph.add_node("report_writer_agent", report_writer_agent)

    graph.set_entry_point("supervisor_router_agent")

    graph.add_conditional_edges(
        "supervisor_router_agent",
        route_after_supervisor,
        {
            "hydrology": "hydrology_chat_agent_gate",
            "advisor": "planner_agent",
            "offtopic": "offtopic_agent",
        },
    )

    graph.add_edge("hydrology_chat_agent_gate", END)
    graph.add_edge("offtopic_agent", END)

    graph.add_conditional_edges(
        "planner_agent",
        advisor_after_planner,
        {
            "done": END,
            "geocode": "geocoder_agent",
        },
    )

    graph.add_conditional_edges(
        "geocoder_agent",
        advisor_after_geocode,
        {
            "done": END,
            "fema": "fema_flood_hazard_agent",
        },
    )

    # Specialist agents run as explicit cooperating agents.
    # This keeps the flow simple and debuggable. You can later parallelize these nodes using Send/reducers.
    graph.add_edge("fema_flood_hazard_agent", "rainfall_hydrology_agent")
    graph.add_edge("rainfall_hydrology_agent", "terrain_drainage_agent")
    graph.add_edge("terrain_drainage_agent", "seiv_vulnerability_agent")
    graph.add_edge("seiv_vulnerability_agent", "scoring_synthesis_agent")
    graph.add_edge("scoring_synthesis_agent", "critic_verifier_agent")
    graph.add_edge("critic_verifier_agent", "report_writer_agent")
    graph.add_edge("report_writer_agent", END)

    return graph.compile()


HYDRO_GRAPH = build_graph()


def run_hydro_graph(user_message: str, history: Optional[List[Dict[str, str]]] = None) -> HydroAgentState:
    """Public entry point used by app.py."""
    initial: HydroAgentState = {
        "user_message": (user_message or "").strip(),
        "history": history or [],
        "stream_required": False,
        "final_answer": "",
        "map_lat": None,
        "map_lon": None,
        "agent_trace": [],
    }
    return HYDRO_GRAPH.invoke(initial)
