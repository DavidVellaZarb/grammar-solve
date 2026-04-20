import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


# flash_attention_2 only supports head_dim <= 256; these models exceed that
_NO_FLASH_ATTN = {"gemma4", "gemma4_text"}

# These models inject <|channel>thought\n<channel|> into the generation prompt;
# training data must include it so the token boundary aligns with inference
_NEEDS_THOUGHT_CHANNEL_PREFIX = {"gemma4", "gemma4_text"}

_VLM_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_5",
    "qwen3_5_vl",
    "gemma3",
    "gemma3_text",
    "gemma4",
    "gemma4_text",
    "llava",
    "llava_next",
    "llava_onevision",
    "idefics2",
    "idefics3",
    "paligemma",
    "mllama",
}


def is_vlm(model_name: str) -> bool:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if getattr(config, "model_type", None) in _VLM_MODEL_TYPES:
        return True
    if hasattr(config, "vision_config"):
        return True
    return False


def load_base_model(
    model_name: str,
    *,
    torch_dtype=torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    device_map="auto",
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device found. Refusing to run inference on CPU. "
            "Set CUDA_VISIBLE_DEVICES or run on a GPU machine."
        )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type in _NO_FLASH_ATTN and attn_implementation == "flash_attention_2":
        attn_implementation = "sdpa"
    vlm = model_type in _VLM_MODEL_TYPES or hasattr(config, "vision_config")
    model_cls = AutoModelForImageTextToText if vlm else AutoModelForCausalLM
    return model_cls.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )


def load_processor(model_name: str):
    if is_vlm(model_name):
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(model_name)


def get_tokenizer(processing_class):
    return getattr(processing_class, "tokenizer", processing_class)
