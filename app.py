import os
import importlib
from typing import List, Dict

import gradio as gr

import model_runtime
model_runtime.get_model_and_tokenizer()

import advisor_graph
import hydrology_chat

BASE_MAP_URL = os.environ.get("MAP_BASE_URL", "https://homes-design-brave-anna.trycloudflare.com/")
DEFAULT_ZOOM = 16
DEBUG_ROUTER = os.environ.get("DEBUG_ROUTER", "1") == "1"


def hot_reload_logic():
    global advisor_graph, hydrology_chat
    advisor_graph = importlib.reload(advisor_graph)
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
    print("advisor_graph:", advisor_graph.__file__)
    print("hydrology_chat:", hydrology_chat.__file__)


def preprocess_and_enqueue(user_message: str, messages: List[Dict[str, str]], current_web_html: str):
    hot_reload_logic()
    messages = messages or []
    q = (user_message or "").strip()

    if not q:
        return gr.update(value=""), messages, messages, current_web_html

    out = advisor_graph.run_hydro_graph(q, messages)

    if DEBUG_ROUTER:
        print(
            f"ROUTE: rule={out.get('route_rule')} final={out.get('intent')} "
            f"stream={out.get('stream_required')} | q={q!r}"
        )
        if out.get("error"):
            print("GRAPH_WARNING:", out.get("error"))
        if out.get("agent_trace"):
            print("AGENT_TRACE:")
            for item in out.get("agent_trace", []):
                print(" -", item)

    messages.append({"role": "user", "content": q})

    # Hydrology answers still stream through the existing hydrology_chat module.
    if out.get("stream_required"):
        messages.append({"role": "assistant", "content": ""})
        return gr.update(value=""), messages, messages, current_web_html

    messages.append({"role": "assistant", "content": out.get("final_answer", "")})

    new_web_html = current_web_html
    if out.get("map_lat") is not None and out.get("map_lon") is not None:
        new_web_html = make_embed_html(make_map_url(out["map_lat"], out["map_lon"], DEFAULT_ZOOM))

    return gr.update(value=""), messages, messages, new_web_html


def stream_wrapper(messages: List[Dict[str, str]]):
    """
    Only stream the LLM if the last assistant message is blank.
    Advisor/offtopic/forecast-guard responses are already filled by the multi-agent LangGraph workflow.
    """
    if not messages:
        yield messages
        return

    last = messages[-1]
    if last.get("role") != "assistant":
        yield messages
        return

    if (last.get("content") or "").strip() != "":
        yield messages
        return

    yield from hydrology_chat.stream_reply(messages)


with gr.Blocks(title="HydroUA-GPT") as demo:
    gr.Markdown("## HydroUA-GPT\nHydrology + Multi-Agent Build-Site Feasibility")

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
demo.queue(max_size=32).launch(server_name=HOST, server_port=PORT, share=True, show_error=True)
