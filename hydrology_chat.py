import torch
from threading import Thread
from typing import List, Dict
from transformers import TextIteratorStreamer

import model_runtime

BASE_SYSTEM_PROMPT = (
    "HydroUA-GPT is a helpful assistant specialized in hydrology and water resources.\n"
    "Answer hydrology-related questions clearly and conversationally.\n"
    "If the question is not hydrology-related, politely refuse.\n"
)

STYLE_HINT = (
    "Write a short direct answer first, then add a few bullet points if helpful.\n"
    "If the user asks a vague term (e.g., 'hydropump'), explain likely meanings and ask one clarifying question.\n"
)

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n" + STYLE_HINT
TRUNCATION_MAX_LEN = 4096
MAX_NEW_TOKENS = 768


def _get_model_and_tokenizer():
    ret = model_runtime.get_model_and_tokenizer()
    if isinstance(ret, (tuple, list)):
        return ret[0], ret[1]
    if isinstance(ret, dict):
        return ret["model"], ret["tokenizer"]
    raise ValueError("model_runtime.get_model_and_tokenizer() returned unexpected type")


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def _build_prompt(messages: List[Dict[str, str]], tokenizer) -> str:
    if _has_chat_template(tokenizer):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Fallback plain format
    conv = SYSTEM_PROMPT + "\n\n"
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            conv += f"User: {content}\n"
        else:
            conv += f"Assistant: {content}\n"
    conv += "Assistant:"
    return conv


def stream_reply(messages: List[Dict[str, str]]):
    """
    IMPORTANT: no 'is_on_topic' rejection here.
    Routing already decided this is hydrology.
    """
    model, tokenizer = _get_model_and_tokenizer()

    prompt_text = _build_prompt(messages, tokenizer)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=TRUNCATION_MAX_LEN,
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    t = Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        messages[-1]["content"] = partial
        yield messages