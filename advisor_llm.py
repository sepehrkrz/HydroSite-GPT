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
    # NEW
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
    starts = [i for i, ch in enumerate(text) if ch == "{"]  # last JSON fallback
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

    # building_type
    bt = slots.get("building_type", None)
    if bt is not None:
        bt = str(bt).strip().lower()
        bt = TYPE_SYNONYMS.get(bt, bt)
        if bt in CANONICAL_TYPES:
            out["building_type"] = bt

    # NEGATION: "not a hospital" should not end up hospital
    if re.search(r"\bnot\s+(a\s+)?hospital\b", (user_text or "").lower()):
        if out["building_type"] == "hospital":
            out["building_type"] = None

    # address: only accept if it looks like an address (prevents geocoding whole question)
    addr = slots.get("address", None)
    if addr is not None:
        addr = str(addr).strip()
        if ADDR_HINT_RE.search(addr) or re.search(r"\b[A-Za-z .'-]+,\s*[A-Z]{2}\b", addr):
            out["address"] = addr

    # mode
    mode = str(slots.get("mode", "block")).strip().lower()
    if mode in {"block", "city", "state"}:
        out["mode"] = mode

    out["needs_followup"] = bool(slots.get("needs_followup", False))
    fq = slots.get("followup_question", "")
    out["followup_question"] = str(fq).strip()[:300] if fq else ""

    # If missing both, force follow-up
    if not out["building_type"] and not out["address"]:
        out["needs_followup"] = True
        if not out["followup_question"]:
            out["followup_question"] = (
                "I can screen a build site, but I need building type + address. "
                "Example: 'club at 123 Main St, Tuscaloosa AL' or 'hospital at 302 Reed St, Tuscaloosa AL'."
            )

    return out

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

    payload = {"user_message": user_text, "geocode_error": geocode_error or "", "conversation_tail": hist}

    if _has_chat_template(tokenizer):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = system + "\nUSER_JSON:\n" + json.dumps(payload, ensure_ascii=False) + "\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    slots = _extract_final_json(text)
    return _validate_slots(slots, user_text)