import json
import os
import re
from datetime import datetime

import fire
import numpy as np
import torch
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data import load_raw_data
from grammar_utils import ENUM_TERMINALS, parse_lark_grammar
from predict_utils import write_output

load_dotenv()


def _normalize_alternative(text: str) -> str:
    tokens = re.findall(r'"(?:[^"\\]|\\.)*"|[^\s"]+', text)
    return " ".join(tokens)


def _expand_terminal_in_alt(alt: str, terminal: str, value: str) -> str:
    parts = re.split(r'("(?:[^"\\]|\\.)*")', alt)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            result.append(part)
        else:
            result.append(re.sub(rf"\b{terminal}\b", value, part))
    return "".join(result)


def _expand_alternative(alt: str, enum_values: dict[str, list[str]]) -> list[str]:
    stripped = re.sub(r'"[^"]*"', "", alt)
    referenced = [t for t in enum_values if re.search(rf"\b{t}\b", stripped)]

    if not referenced:
        return [alt]

    results = [alt]
    for terminal in referenced:
        new_results = []
        for current in results:
            for value in enum_values[terminal]:
                expanded = _expand_terminal_in_alt(current, terminal, value)
                new_results.append(expanded)
        results = new_results

    return results


def build_label_index(
    grammar_path: str,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    with open(grammar_path) as f:
        grammar_text = f.read()
    rules = parse_lark_grammar(grammar_text)

    enum_values: dict[str, list[str]] = {}
    for term in ENUM_TERMINALS:
        if term in rules:
            enum_values[term] = rules[term]

    labels: list[tuple[str, str]] = []
    label_to_idx: dict[tuple[str, str], int] = {}

    for name, alts in rules.items():
        if name.isupper():
            continue
        for alt in alts:
            normalized = _normalize_alternative(alt)
            expanded_alts = _expand_alternative(normalized, enum_values)

            if len(expanded_alts) > 1 or _normalize_alternative(expanded_alts[0]) != normalized:
                for exp_alt in expanded_alts:
                    exp_norm = _normalize_alternative(exp_alt)
                    label = (name, exp_norm)
                    if label not in label_to_idx:
                        label_to_idx[label] = len(labels)
                        labels.append(label)
            else:
                label = (name, normalized)
                if label not in label_to_idx:
                    label_to_idx[label] = len(labels)
                    labels.append(label)

    alt_to_label_idx: dict[tuple[str, str], int] = {}
    alt_to_label_idx.update(label_to_idx)
    for label_idx, (name, alt) in enumerate(labels):
        merged = re.sub(r'"\s+"', "", alt)
        merged = " ".join(merged.split())
        if merged != alt:
            key_merged = (name, merged)
            if key_merged not in alt_to_label_idx:
                alt_to_label_idx[key_merged] = label_idx

    return labels, label_to_idx, alt_to_label_idx


def minimal_grammar_to_labels(
    grammar_text: str,
    alt_to_label_idx: dict[tuple[str, str], int],
    n_labels: int,
) -> np.ndarray:
    from grammar_utils import parse_minimal_grammar

    rules = parse_minimal_grammar(grammar_text)
    vector = np.zeros(n_labels, dtype=np.float32)
    for name, alts in rules.items():
        for alt in alts:
            key = (name, _normalize_alternative(alt))
            if key in alt_to_label_idx:
                vector[alt_to_label_idx[key]] = 1.0
    return vector


def labels_to_minimal_grammar(
    label_vector: np.ndarray,
    labels: list[tuple[str, str]],
) -> str:
    from grammar_utils import reconstruct_minimal_grammar

    rules: dict[str, list[str]] = {}
    for i, (name, alt) in enumerate(labels):
        if label_vector[i]:
            if name not in rules:
                rules[name] = []
            rules[name].append(alt)
    return reconstruct_minimal_grammar(rules)


def _build_dataset(
    data: list[dict],
    alt_to_label_idx: dict[tuple[str, str], int],
    n_labels: int,
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    queries = [ex["query"] for ex in data]
    label_vectors = [
        minimal_grammar_to_labels(ex["minimal_grammar"], alt_to_label_idx, n_labels).tolist()
        for ex in data
    ]

    ds = Dataset.from_dict({"text": queries, "labels": label_vectors})
    ds = ds.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=max_seq_length),
        batched=True,
        remove_columns=["text"],
    )
    ds.set_format("torch")
    return ds


def train(
    train_path: str = "data/smcalflow/train_generic.json",
    val_path: str = "data/smcalflow/valid_generic.json",
    grammar_path: str = "grammars/smcalflow_pruned.lark",
    output_dir: str = "outputs/classifier",
    model_name: str = "microsoft/deberta-v3-base",
    num_train_epochs: int = 5,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 32,
    max_seq_length: int = 128,
    seed: int = 42,
    push_to_hub: bool = True,
    hub_model_id: str | None = None,
) -> None:
    model_alias = model_name.split("/")[-1].lower()
    hf_namespace = os.getenv("HF_NAMESPACE", "")
    hub_repo = hub_model_id or (
        f"{hf_namespace}/{model_alias}_smcalflow-classifier" if hf_namespace else None
    )

    labels, label_to_idx, alt_to_label_idx = build_label_index(grammar_path)
    n_labels = len(labels)
    print(f"Built label index with {n_labels} labels")

    train_data = load_raw_data(train_path)
    print(f"Loaded {len(train_data)} training examples")

    Y = np.stack([
        minimal_grammar_to_labels(ex["minimal_grammar"], alt_to_label_idx, n_labels)
        for ex in train_data
    ])
    active_mask = Y.sum(axis=0) > 0
    n_active = int(active_mask.sum())
    if n_active < n_labels:
        print(f"Filtering {n_labels - n_active} labels with no positive examples ({n_active} active)")
        labels = [l for l, m in zip(labels, active_mask) if m]
        old_to_new = {}
        for old_idx, active in enumerate(active_mask):
            if active:
                old_to_new[old_idx] = len(old_to_new)
        alt_to_label_idx = {
            k: old_to_new[v] for k, v in alt_to_label_idx.items() if v in old_to_new
        }
        n_labels = n_active

    print(f"Active labels: {n_labels}, avg per example: {Y[:, active_mask.astype(bool)].sum(axis=1).mean():.1f}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    val_data = load_raw_data(val_path)
    print(f"Loaded {len(val_data)} validation examples")

    train_ds = _build_dataset(train_data, alt_to_label_idx, n_labels, tokenizer, max_seq_length)
    val_ds = _build_dataset(val_data, alt_to_label_idx, n_labels, tokenizer, max_seq_length)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_labels,
        problem_type="multi_label_classification",
        torch_dtype=torch.float32,
    )

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        y_pred = (probs > 0.5).astype(int)
        return {
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "exact_match": float((y_pred == y_true).all(axis=1).mean()),
        }

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    run_name = f"classifier_{model_alias}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        run_name=run_name,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        seed=seed,
        report_to="wandb" if use_wandb else "none",
        push_to_hub=push_to_hub and hub_repo is not None,
        hub_model_id=hub_repo,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f)
    print(f"Saved model and labels to {output_dir}")

    if push_to_hub and hub_repo:
        trainer.push_to_hub()
        tokenizer.push_to_hub(hub_repo)
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=labels_path,
            path_in_repo="labels.json",
            repo_id=hub_repo,
        )
        print(f"Pushed to {hub_repo}")


def predict(
    test_path: str = "data/smcalflow/test_generic.json",
    output_path: str = "outputs/predicted_grammars/classifier_generic.json",
    classifier: str = "deberta-v3-base_smcalflow-classifier",
    threshold: float = 0.5,
    batch_size: int = 64,
    max_seq_length: int = 128,
) -> None:
    if "/" not in classifier:
        hf_namespace = os.getenv("HF_NAMESPACE", "")
        if not hf_namespace:
            raise ValueError("No namespace in classifier name and HF_NAMESPACE not set")
        classifier = f"{hf_namespace}/{classifier}"

    model = AutoModelForSequenceClassification.from_pretrained(classifier)
    tokenizer = AutoTokenizer.from_pretrained(classifier)
    from huggingface_hub import hf_hub_download
    labels_file = hf_hub_download(repo_id=classifier, filename="labels.json")
    with open(labels_file) as f:
        labels = [tuple(l) for l in json.load(f)]
    print(f"Loaded classifier from {classifier} with {len(labels)} labels, threshold={threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_data = load_raw_data(test_path)
    queries = [ex["query"] for ex in test_data]
    print(f"Loaded {len(test_data)} test examples")

    all_preds = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        encodings = tokenizer(
            batch, truncation=True, max_length=max_seq_length,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encodings).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append((probs >= threshold).astype(np.float32))

    predictions = np.concatenate(all_preds)

    results = []
    for i, ex in enumerate(test_data):
        grammar = labels_to_minimal_grammar(predictions[i], labels)
        results.append({
            "query": ex["query"],
            "minimal_grammar": grammar,
            "program": ex["program"],
        })

    write_output(results, output_path)


if __name__ == "__main__":
    fire.Fire({"train": train, "predict": predict})
