import os
import importlib
import re
from typing import List, Dict

import gradio as gr

import model_runtime
model_runtime.get_model_and_tokenizer()

import router
import llm_router
import advisor_flow
import hydrology_chat

BASE_MAP_URL = os.environ.get("MAP_BASE_URL", "https://homes-design-brave-anna.trycloudflare.com/")
DEFAULT_ZOOM = 16
DEBUG_ROUTER = os.environ.get("DEBUG_ROUTER", "1") == "1"

WEATHER_Q = re.compile(r"\b(will it rain|rain tonight|rain tomorrow|forecast|weather)\b", re.IGNORECASE)
FLOOD_FORECAST_Q = re.compile(r"\b(will it flood|flood tonight|flood tomorrow|flood anytime soon)\b", re.IGNORECASE)

# Follow-up messages that should inherit last intent
FOLLOWUP_INHERIT_RE = re.compile(
    r"^\s*(please\s+)?(explain|define)\b.*\b(example|examples|more|again|better|elaborately)\b\s*$"
    r"|^\s*(example|examples|with example|with examples|give an example|give examples)\s*$",
    re.IGNORECASE,
)

YES_FOLLOWUP_RE = re.compile(r"^\s*(yes|yeah|yep|ok|okay|sure)\b", re.IGNORECASE)


def hot_reload_logic():
    global router, llm_router, advisor_flow, hydrology_chat
    router = importlib.reload(router)
    llm_router = importlib.reload(llm_router)
    advisor_flow = importlib.reload(advisor_flow)
    hydrology_chat = importlib.reload(hydrology_chat)


def make_map_url(lat: float, lon: float, zoom: int = DEFAULT_ZOOM) -> str:
    return f"{BASE_MAP_URL}?lat={lat}&lng={lon}&z={zoom}"


def make_embed_html(url: str) -> str:
    return f"""
<div style="width:100%; height:560px; border:1px solid #d0d0d0; border-radius:10px; overflow:hidden;">
  <iframe src="{url}" style="width:100%; height:100%; border:0;" loading="lazy"></iframe>
</div>
"""


DEFAULT_WEB_HTML = """
<div style="width:100%; height:560px; border:1px dashed #d0d0d0; border-radius:10px; padding:12px;">
  <div style="font-size:14px; color:#666;">
    Map will appear here after you run a build-site feasibility query.
  </div>
</div>
"""


if DEBUG_ROUTER:
    print("APP:", __file__)
    print("router:", router.__file__)
    print("llm_router:", llm_router.__file__)
    print("advisor_flow:", advisor_flow.__file__)
    print("hydrology_chat:", hydrology_chat.__file__)


def _last_user_intent(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return router.route_intent_rules(m.get("content", ""))
    return ""


def preprocess_and_enqueue(user_message: str, messages: List[Dict[str, str]], current_web_html: str):
    hot_reload_logic()
    messages = messages or []
    q = (user_message or "").strip()

    if not q:
        return gr.update(value=""), messages, messages, current_web_html

    # Follow-up inheritance (no keywords needed)
    inherited = _last_user_intent(messages)
    if FOLLOWUP_INHERIT_RE.match(q) and inherited in {"hydrology", "advisor"}:
        route_rule = inherited
        route_final = inherited
        if DEBUG_ROUTER:
            print(f"ROUTE: rule={route_rule} final={route_final} | q={q!r} (followup-inherit)")
    else:
        # RULES decide first
        route_rule = router.route_intent_rules(q)
        route_final = route_rule

        # LLM can ONLY upgrade offtopic -> hydrology/advisor (never demote)
        if route_rule == "offtopic":
            try:
                llm_choice = llm_router.route_intent_llm(q, messages)
                if llm_choice in {"hydrology", "advisor"}:
                    route_final = llm_choice
            except Exception as e:
                if DEBUG_ROUTER:
                    print("LLM_ROUTER_ERROR:", repr(e))

        # Extra positivity for short confirmations in an ongoing hydrology thread
        if route_final == "offtopic" and inherited == "hydrology" and (YES_FOLLOWUP_RE.match(q) or len(q) <= 28):
            route_final = "hydrology"

        if DEBUG_ROUTER:
            print(f"ROUTE: rule={route_rule} final={route_final} | q={q!r}")

    # --- Dispatch ---
    if route_final == "advisor":
        out = advisor_flow.run_build_site_advisor(q, messages)
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": out["text"]})

        new_web_html = current_web_html
        if out.get("lat") is not None and out.get("lon") is not None:
            new_web_html = make_embed_html(make_map_url(out["lat"], out["lon"], DEFAULT_ZOOM))

        return gr.update(value=""), messages, messages, new_web_html

    if route_final == "hydrology":
        # Forecast safety guard (don't hallucinate)
        if WEATHER_Q.search(q) or FLOOD_FORECAST_Q.search(q):
            safe = (
                "I can explain rainfall/flooding and how forecasts work, but I don't have a guaranteed real-time feed here.\n\n"
                "Tell me your location (city/ZIP) and I'll tell you what to check (NWS forecast, probability of precipitation, "
                "watches/warnings, river gauge forecast points) and how to interpret it. "
                "If you paste a forecast here, I can interpret it."
            )
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": safe})
            return gr.update(value=""), messages, messages, current_web_html

        # IMPORTANT: add blank assistant so stream_wrapper knows to generate
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": ""})
        return gr.update(value=""), messages, messages, current_web_html

    # Offtopic
    refusal = (
        "I can help with hydrology questions and build-site feasibility screening.\n\n"
        "For build-site screening, include a building type + location/address.\n"
        "Example: 'Can I build a warehouse at 123 Main St, Tuscaloosa AL?'"
    )
    messages.append({"role": "user", "content": q})
    messages.append({"role": "assistant", "content": refusal})
    return gr.update(value=""), messages, messages, current_web_html


def stream_wrapper(messages: List[Dict[str, str]]):
    """
    CRITICAL FIX:
    Only stream the LLM if the last assistant message is blank ("").
    That means it's a hydrology answer waiting to be generated.
    For advisor/offtopic, assistant already has text -> do nothing.
    """
    if not messages:
        yield messages
        return

    last = messages[-1]
    if last.get("role") != "assistant":
        yield messages
        return

    if (last.get("content") or "").strip() != "":
        # advisor/offtopic already filled
        yield messages
        return

    # hydrology: stream model output into the blank assistant message
    yield from hydrology_chat.stream_reply(messages)


with gr.Blocks(title="HydroUA-GPT") as demo:
    gr.Markdown("## HydroUA-GPT\nHydrology + Build-Site Feasibility")

    with gr.Row():
        with gr.Column(scale=6):
            chat = gr.Chatbot(height=520, type="messages", label="HydroUA-GPT")
            msg = gr.Textbox(
                placeholder="Ask hydrology, or: Can I build a warehouse at 123 Main St, Tuscaloosa AL?",
                lines=3,
                autofocus=True,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=5):
            gr.Markdown("### Location View")
            webview = gr.HTML(value=DEFAULT_WEB_HTML)

    state = gr.State([])

    msg.submit(preprocess_and_enqueue, [msg, state, webview], [msg, state, chat, webview]).then(stream_wrapper, [state], [chat])
    send_btn.click(preprocess_and_enqueue, [msg, state, webview], [msg, state, chat, webview]).then(stream_wrapper, [state], [chat])

    def clear_all():
        return [], DEFAULT_WEB_HTML

    clear_btn.click(clear_all, outputs=[chat, webview]).then(lambda: [], outputs=[state])

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "7860"))
demo.queue(max_size=32).launch(server_name=HOST, server_port=PORT, share=False, show_error=True)
