import json
import re
from typing import Any, Dict, List, Optional

import torch
import model_runtime

INTENTS = {"hydrology", "advisor", "offtopic"}

# Treat these as strong signals for hydrology (LLM should be typo-tolerant too)
HYDRO_SIGNAL_RE = re.compile(
    r"\b(hydro\w*|aqua\w*|flood\w*|rain\w*|rainfall\w*|precip\w*|runoff\w*|"
    r"watershed\w*|basin\w*|stream\w*|river\w*|groundwater\w*|aquifer\w*|"
    r"soil\w*|smap\w*|evap\w*|et\b|drought\w*|storm\w*|drain\w*|hydraulic\w*)\b",
    re.IGNORECASE,
)

# Minor typo normalizer (cheap)
TYPO_MAP = {
    "fainfall": "rainfall",
    "rainfal": "rainfall",
    "ranfall": "rainfall",
    "hydrolgy": "hydrology",
    "hydrolgoy": "hydrology",
    "preciptation": "precipitation",
    "precipitaion": "precipitation",
}

def _normalize(q: str) -> str:
    t = (q or "").strip()
    for bad, good in TYPO_MAP.items():
        t = re.sub(bad, good, t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t


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


def route_intent_llm(user_text: str, history: List[Dict[str, str]]) -> str:
    """
    Hydrology-biased router:
      - If uncertain between hydrology vs offtopic, choose hydrology.
      - Anything with hydro*/flood*/rain*/runoff*/soil*/smap* should route hydrology.
      - Still route obvious build-site feasibility to advisor.
    """
    model, tokenizer = _get_model_and_tokenizer()
    hist = _truncate_history(history, 10)

    q_norm = _normalize(user_text)

    # VERY FAST pre-check: if it looks hydrology-adjacent, force hydrology
    if HYDRO_SIGNAL_RE.search(q_norm):
        return "hydrology"

    system = (
        "You are an INTENT ROUTER for a single-chat app. Choose exactly one intent:\n\n"
        "A) hydrology\n"
        "- Anything about water/soil/precipitation/rainfall/weather concepts/runoff/floods/droughts/river systems/groundwater.\n"
        "- Remote sensing hydrology (e.g., SMAP soil moisture, saturation maps).\n"
        "- IMPORTANT: If a term looks hydrology-adjacent (especially starts with 'hydro' like 'hydropump'), choose hydrology.\n"
        "- If you are unsure between hydrology vs offtopic, choose hydrology.\n\n"
        "B) advisor\n"
        "- Build-site feasibility screening: can I build X at Y, building type + address, follow-ups like 'hospital' or an address.\n\n"
        "C) offtopic\n"
        "- Clearly unrelated topics.\n\n"
        "Return STRICT JSON only:\n"
        "<FINAL_JSON>{\"intent\":\"hydrology|advisor|offtopic\"}</FINAL_JSON>\n"
    )

    payload = {"user_message": q_norm, "conversation_tail": hist}

    if _has_chat_template(tokenizer):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = system + "\nUSER_JSON:\n" + json.dumps(payload, ensure_ascii=False) + "\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    obj = _extract_final_json(text) or {}

    intent = str(obj.get("intent", "offtopic")).strip().lower()
    if intent not in INTENTS:
        intent = "offtopic"
    return intent