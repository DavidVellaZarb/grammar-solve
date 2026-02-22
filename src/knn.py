import hashlib
import json
import os

import fire
import numpy as np
from sentence_transformers import SentenceTransformer

from data import load_raw_data
from grammar_utils import parse_minimal_grammar, reconstruct_minimal_grammar


def _compute_embeddings(
    texts: list[str], model: SentenceTransformer, batch_size: int = 256
) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)


def _load_or_compute_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    cache_path: str,
    model_name: str,
    batch_size: int = 256,
) -> np.ndarray:
    query_hash = hashlib.md5("".join(texts).encode()).hexdigest()[:8]
    safe_model = model_name.replace("/", "_")
    cache_file = os.path.join(cache_path, f"{safe_model}_{query_hash}.npy")

    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = _compute_embeddings(texts, model, batch_size)
    os.makedirs(cache_path, exist_ok=True)
    np.save(cache_file, embeddings)
    print(f"Cached embeddings to {cache_file}")
    return embeddings


def _find_knn(
    query_embeddings: np.ndarray, train_embeddings: np.ndarray, k: int
) -> np.ndarray:
    scores = query_embeddings @ train_embeddings.T
    if k >= scores.shape[1]:
        return np.argsort(-scores, axis=1)
    indices = np.argpartition(-scores, k, axis=1)[:, :k]
    row_idx = np.arange(indices.shape[0])[:, None]
    sorted_order = np.argsort(-scores[row_idx, indices], axis=1)
    return indices[row_idx, sorted_order]


def merge_grammars(grammars: list[str], strategy: str = "union") -> str:
    parsed = [parse_minimal_grammar(g) for g in grammars]

    if strategy == "union":
        merged: dict[str, list[str]] = {}
        for rules in parsed:
            for name, alts in rules.items():
                if name not in merged:
                    merged[name] = []
                for alt in alts:
                    if alt not in merged[name]:
                        merged[name].append(alt)
        return reconstruct_minimal_grammar(merged)

    elif strategy == "intersection":
        if not parsed:
            return ""
        all_names = set(parsed[0].keys())
        for rules in parsed[1:]:
            all_names &= set(rules.keys())
        merged = {}
        for name in all_names:
            common_alts = set(parsed[0].get(name, []))
            for rules in parsed[1:]:
                common_alts &= set(rules.get(name, []))
            if common_alts:
                merged[name] = [
                    a for a in parsed[0][name] if a in common_alts
                ]
        return reconstruct_minimal_grammar(merged)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def predict(
    test_path: str = "data/smcalflow/test.json",
    train_path: str = "data/smcalflow/train_generic.json",
    output_path: str = "outputs/predicted_grammars/knn_generic.json",
    model_name: str = "BAAI/bge-large-en-v1.5",
    k: int = 1,
    strategy: str = "union",
    cache_dir: str = "cache/knn",
    batch_size: int = 256,
) -> None:
    train_data = load_raw_data(train_path)
    test_data = load_raw_data(test_path)

    train_queries = [ex["query"] for ex in train_data]
    test_queries = [ex["query"] for ex in test_data]

    print(f"Train: {len(train_queries)} queries, Test: {len(test_queries)} queries")
    print(f"Model: {model_name}, k={k}, strategy={strategy}")

    model = SentenceTransformer(model_name)

    train_embeddings = _load_or_compute_embeddings(
        train_queries, model, cache_dir, model_name, batch_size
    )
    test_embeddings = _load_or_compute_embeddings(
        test_queries, model, cache_dir, model_name, batch_size
    )

    knn_indices = _find_knn(test_embeddings, train_embeddings, k)
    print(f"Found {k}-NN for {len(test_queries)} test queries")

    results = []
    for i, ex in enumerate(test_data):
        neighbor_grammars = [
            train_data[idx]["minimal_grammar"] for idx in knn_indices[i]
        ]
        merged = merge_grammars(neighbor_grammars, strategy)
        results.append(
            {
                "query": ex["query"],
                "minimal_grammar": merged,
                "program": ex["program"],
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"data": results}, f, indent=2)

    print(f"Wrote {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    fire.Fire({"predict": predict})
