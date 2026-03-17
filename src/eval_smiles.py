import json
import re

import fire
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftConfig, PeftModel
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt_messages, load_raw_data
from eval_utils import save_results


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compute_fingerprint_similarity(s1: str, s2: str) -> float | None:
    m1 = Chem.MolFromSmiles(s1)
    m2 = Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp1 = gen.GetFingerprint(m1)
    fp2 = gen.GetFingerprint(m2)
    return TanimotoSimilarity(fp1, fp2)


def smiles_to_tokens(smiles: str) -> list[str]:
    pattern = r"Br|Cl|%\d{2}|@@|[A-Z][a-z]?|[a-z]|[^A-Za-z]"
    return re.findall(pattern, smiles)


def extract_smiles(prediction: str) -> str:
    return prediction.strip().split("\n")[0].strip()


def evaluate(
    adapter: str,
    test_path: str = "data/smiles/test.json",
    model_name: str | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    output_path: str | None = None,
    attn_implementation: str = "flash_attention_2",
    grammar_file: str | None = None,
    include_grammar: bool = True,
    task: str = "program",
):
    peft_config = PeftConfig.from_pretrained(adapter)
    base_model_name = model_name or peft_config.base_model_name_or_path
    assert base_model_name is not None, "No model_name provided and adapter config has no base_model_name_or_path"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    examples = load_raw_data(test_path)

    if grammar_file:
        print(f"Using predicted grammars from {grammar_file}")
        with open(grammar_file) as f:
            grammar_data = json.load(f)["data"]
        assert len(grammar_data) == len(examples), (
            f"Grammar file has {len(grammar_data)} entries but test data has {len(examples)}"
        )
        for ex, gex in zip(examples, grammar_data):
            assert ex["query"] == gex["query"], (
                f"Query mismatch: {ex['query']!r} vs {gex['query']!r}"
            )
            ex["minimal_grammar"] = gex["minimal_grammar"]
    else:
        print("Using gold grammars from test data")

    prompts = []
    for ex in examples:
        messages = format_prompt_messages(ex, include_grammar=include_grammar, task=task)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        predictions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()

    results = []
    for ex, prompt, pred in zip(examples, prompts, predictions):
        gold = ex["program"]
        pred_smiles = extract_smiles(pred)

        exact_match = gold in pred

        gold_canon = canonicalize_smiles(gold)
        pred_canon = canonicalize_smiles(pred_smiles)
        canonical_match = (
            gold_canon is not None
            and pred_canon is not None
            and gold_canon == pred_canon
        )

        valid = pred_canon is not None

        fp_sim = compute_fingerprint_similarity(gold, pred_smiles)

        gold_tokens = smiles_to_tokens(gold)
        pred_tokens = smiles_to_tokens(pred_smiles)
        bleu = sentence_bleu(
            [gold_tokens],
            pred_tokens,
            smoothing_function=SmoothingFunction().method1,
        )

        results.append({
            "prompt": prompt,
            "gold": gold,
            "prediction": pred,
            "pred_smiles": pred_smiles,
            "exact_match": exact_match,
            "canonical_match": canonical_match,
            "valid": valid,
            "fingerprint_similarity": fp_sim,
            "bleu": bleu,
        })

    total = len(results)
    exact_count = sum(1 for r in results if r["exact_match"])
    canon_count = sum(1 for r in results if r["canonical_match"])
    valid_count = sum(1 for r in results if r["valid"])
    fp_sims = [r["fingerprint_similarity"] for r in results if r["fingerprint_similarity"] is not None]
    bleus = [r["bleu"] for r in results]

    metrics = {
        "accuracy": canon_count / total if total > 0 else 0.0,
        "exact_match": exact_count / total if total > 0 else 0.0,
        "canonical_exact_match": canon_count / total if total > 0 else 0.0,
        "validity": valid_count / total if total > 0 else 0.0,
        "fingerprint_similarity": sum(fp_sims) / len(fp_sims) if fp_sims else 0.0,
        "bleu": sum(bleus) / len(bleus) if bleus else 0.0,
        "correct": canon_count,
        "total": total,
    }

    print(f"Exact match:           {metrics['exact_match']:.4f} ({exact_count}/{total})")
    print(f"Canonical exact match: {metrics['canonical_exact_match']:.4f} ({canon_count}/{total})")
    print(f"Validity:              {metrics['validity']:.4f} ({valid_count}/{total})")
    print(f"Fingerprint sim:       {metrics['fingerprint_similarity']:.4f}")
    print(f"BLEU:                  {metrics['bleu']:.4f}")

    if output_path:
        save_results(metrics, results, output_path)


if __name__ == "__main__":
    fire.Fire(evaluate)
