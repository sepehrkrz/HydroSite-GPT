import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#MODEL_PATH = "/icebox/data/shares/mh1/mkarimiziarani/research/gpu/2-generate/QAParagraphs/hydrology_aqua_llm_strict_11B"
MODEL_PATH = "/icebox/data/shares/mh2/shassan6/gradio/models/hydrology_aqua_llm_strict_11B"


_model = None
_tokenizer = None
_has_chat_template = False


def get_model_and_tokenizer():
    global _model, _tokenizer, _has_chat_template
    if _model is not None:
        return _model, _tokenizer, _has_chat_template

    try:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
        )
        base_id = getattr(model.config, "base_model_name_or_path", None) or MODEL_PATH
        tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
        )

    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    model.eval()

    _model = model
    _tokenizer = tok
    _has_chat_template = bool(getattr(tok, "chat_template", None))
    return _model, _tokenizer, _has_chat_template