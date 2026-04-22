import sys

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

# Hybrid-attention models (Gated DeltaNet + causal conv1d token mixer). Without
# the fused kernels from flash-linear-attention and causal-conv1d, HF falls
# back to a pure-PyTorch reference implementation. It still runs on GPU but is
# kernel-launch-bound: one CPU core pins at 100%, GPU util drops to ~2%, and
# decoding is 10-20x slower (a 30-min eval becomes 6+ hours). The only signal
# HF emits is a single warning line, easy to miss. Check at load time instead.
_NEEDS_LINEAR_ATTN_KERNELS = {"qwen3_5", "qwen3_5_vl", "qwen3_next"}


def _warn_if_linear_attn_kernels_missing():
    missing = []
    try:
        import causal_conv1d  # noqa: F401
    except ImportError:
        missing.append("causal-conv1d")
    try:
        import fla  # noqa: F401
    except ImportError:
        missing.append("flash-linear-attention")
    if not missing:
        return
    bar = "=" * 80
    print(
        f"\n{bar}\n"
        f"WARNING: hybrid-attention model loaded without fused kernels: "
        f"{', '.join(missing)}.\n"
        f"Inference will run ~10-20x slower on the PyTorch fallback path.\n"
        f"Install with:\n"
        f"    uv pip install {' '.join(missing)}\n"
        f"{bar}\n",
        file=sys.stderr,
        flush=True,
    )

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
    if model_type in _NEEDS_LINEAR_ATTN_KERNELS:
        _warn_if_linear_attn_kernels_missing()
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
