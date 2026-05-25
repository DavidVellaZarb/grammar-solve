from __future__ import annotations

import os
import sys

import fire
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import cache_key, load_cache, save_cache
from model_loading import get_tokenizer, load_base_model, load_processor
from predict_utils import write_output
from rag_grammar import (
    _build_messages,
    _get_system_prompt,
    _load_grammar_as_bnf,
    _load_knn,
)


def predict(
    test_path: str = "data/smcalflow/test.json",
    train_path: str = "data/smcalflow/train.json",
    grammar_path: str = "grammars/smcalflow.lark",
    output_path: str = "outputs/predicted_grammars/openweight_rag/test.json",
    model: str = "Qwen/Qwen3.5-9B-Instruct",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    k: int = 64,
    cache_path: str = "cache/rag_local_cache.json",
    cache_dir: str = "cache/knn",
    batch_size: int = 4,
    embed_batch_size: int = 256,
    max_new_tokens: int = 4096,
    prompt_style: str = "cot",
    exclude_self: bool = False,
    attn_implementation: str = "flash_attention_2",
):
    print(f"Model: {model}, Embedding: {embedding_model}")

    full_grammar = _load_grammar_as_bnf(grammar_path)
    system_prompt = _get_system_prompt(grammar_path, full_grammar, prompt_style=prompt_style)

    train_data, test_data, knn_indices = _load_knn(
        test_path, train_path, embedding_model, cache_dir, k, embed_batch_size,
        exclude_self=exclude_self,
    )
    cache = load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")

    all_messages: list[list[dict]] = []
    all_keys: list[str] = []
    for i, ex in enumerate(test_data):
        neighbors = [train_data[idx] for idx in knn_indices[i]]
        messages = _build_messages(
            ex["query"], neighbors, system_prompt, prompt_style=prompt_style,
        )
        all_messages.append(messages)
        all_keys.append(cache_key(messages, model))

    pending_idx = [i for i in range(len(test_data)) if all_keys[i] not in cache]
    print(f"Pending: {len(pending_idx)} / Total: {len(test_data)}")

    if pending_idx:
        m = load_base_model(model, attn_implementation=attn_implementation)
        m.eval()
        processing_class = load_processor(model)
        tokenizer = get_tokenizer(processing_class)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        chat_kwargs: dict = {}
        if "qwen3" in model.lower():
            chat_kwargs["enable_thinking"] = False

        for start in tqdm(range(0, len(pending_idx), batch_size), desc="Local RAG predict"):
            batch_idx = pending_idx[start:start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    all_messages[i], tokenize=False, add_generation_prompt=True,
                    **chat_kwargs,
                )
                for i in batch_idx
            ]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=False,
            ).to(m.device)
            with torch.no_grad():
                outputs = m.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            responses = tokenizer.batch_decode(
                outputs[:, prompt_len:], skip_special_tokens=True,
            )
            for i, resp in zip(batch_idx, responses):
                cache[all_keys[i]] = resp.strip()
            save_cache(cache, cache_path)

        del m
        torch.cuda.empty_cache()

    results = []
    n_missing = 0
    for i, ex in enumerate(test_data):
        key = all_keys[i]
        if key in cache:
            results.append({**ex, "minimal_grammar": cache[key]})
        else:
            results.append({**ex, "minimal_grammar": None})
            n_missing += 1
    if n_missing:
        print(f"Warning: {n_missing} examples missing from cache")
    write_output(results, output_path)


if __name__ == "__main__":
    fire.Fire({"predict": predict})
