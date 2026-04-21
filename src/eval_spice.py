import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import networkx as nx
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from model_loading import get_tokenizer, load_base_model, load_processor

from data import format_prompt_messages, load_raw_data
from eval_utils import save_results
from grammar_parser import _build_parser
from grammar_utils import extract_grammar_from_output

GRAMMAR_PATH = "grammars/spice.lark"

COMPONENT_PREFIXES = set("RCLVIDQMXKJGEFHBSWrclvidqmxkjgefhbsw")


def extract_netlist(prediction: str) -> str:
    """Extract SPICE netlist from model prediction, up to and including .end."""
    lines = prediction.strip().split("\n")
    result_lines = []
    for line in lines:
        result_lines.append(line)
        if line.strip().lower() == ".end":
            break
    return "\n".join(result_lines).strip()


def check_syntax_validity(netlist: str) -> bool:
    """Check if the netlist can be parsed by our Lark grammar."""
    from load_spice import _preprocess_netlist
    try:
        preprocessed = _preprocess_netlist(netlist)
        parser = _build_parser(GRAMMAR_PATH, start="netlist")
        parser.parse(preprocessed)
        return True
    except Exception:
        return False


_COMP_NODE_COUNT = {
    "R": 2, "C": 2, "L": 2,
    "V": 2, "I": 2,
    "D": 2,
    "Q": 3,
    "M": 4,
    "J": 3,
    "K": 0,
    "X": -1,
    "G": 4, "E": 4,
    "F": 2, "H": 2,
    "B": 2,
    "S": 4, "W": 2,
}


def _netlist_to_graph(netlist: str) -> nx.Graph:
    """Parse a netlist into a graph: nodes=components+nets, edges=connections."""
    G = nx.Graph()
    for line in netlist.strip().split("\n"):
        parts = line.strip().split()
        if not parts:
            continue
        name = parts[0]
        if not name or name[0].upper() not in COMPONENT_PREFIXES:
            continue
        comp_type = name[0].upper()
        comp_node = f"comp:{name.lower()}"
        G.add_node(comp_node, type=comp_type)

        n_nodes = _COMP_NODE_COUNT.get(comp_type, 2)
        if n_nodes == -1:
            node_parts = []
            for part in parts[1:]:
                if "=" in part:
                    break
                node_parts.append(part)
            if len(node_parts) > 1:
                node_parts = node_parts[:-1]
        else:
            node_parts = parts[1:1 + n_nodes]

        for part in node_parts:
            net_node = f"net:{part.lower()}"
            if not G.has_node(net_node):
                G.add_node(net_node, type="net")
            G.add_edge(comp_node, net_node)
    return G


def compute_ged_similarity(gold_netlist: str, pred_netlist: str, timeout: float = 5.0) -> float:
    """Compute GED-based similarity: S = (1 - GED / GED_max) * 100.

    Uses Masala-CHAI formulation where GED_max = |V1|+|E1|+|V2|+|E2|.
    Uses networkx optimize_graph_edit_distance for tractability.
    """
    g1 = _netlist_to_graph(gold_netlist)
    g2 = _netlist_to_graph(pred_netlist)

    ged_max = g1.number_of_nodes() + g1.number_of_edges() + g2.number_of_nodes() + g2.number_of_edges()
    if ged_max == 0:
        return 1.0

    def node_subst_cost(attrs1, attrs2):
        return 0.0 if attrs1.get("type") == attrs2.get("type") else 1.0

    try:
        best_ged = ged_max
        import signal

        class TimeoutError(Exception):
            pass

        def handler(signum, frame):
            raise TimeoutError()

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            for ged in nx.optimize_graph_edit_distance(
                g1, g2, node_subst_cost=node_subst_cost
            ):
                best_ged = ged
        except TimeoutError:
            pass
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

        similarity = max(0.0, 1.0 - best_ged / ged_max)
        return similarity
    except Exception:
        return 0.0


def extract_component_types(netlist: str) -> set[str]:
    """Extract set of component type prefixes from a netlist."""
    types = set()
    for line in netlist.strip().split("\n"):
        line = line.strip()
        if line and line[0].upper() in COMPONENT_PREFIXES:
            types.add(line[0].upper())
    return types


def compute_component_f1(gold_types: set[str], pred_types: set[str]) -> dict:
    """Compute precision, recall, F1 for component types."""
    if not gold_types and not pred_types:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(gold_types & pred_types)
    precision = tp / len(pred_types) if pred_types else 0.0
    recall = tp / len(gold_types) if gold_types else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def run_ngspice_simulation(netlist: str, timeout: float = 10.0) -> bool:
    """Run netlist through ngspice -b, return True if no errors."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False) as f:
        f.write(netlist)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["ngspice", "-b", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        has_error = result.returncode != 0 or "Error" in result.stderr
        return not has_error
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        os.unlink(tmp_path)


def _evaluate_single(args: tuple) -> dict:
    """Evaluate a single prediction (top-level for ProcessPoolExecutor)."""
    idx, gold, prompt, pred, ged_timeout, ngspice_timeout = args
    pred_netlist = extract_netlist(pred)

    exact_match = gold in pred
    valid = check_syntax_validity(pred_netlist)

    ged_sim = compute_ged_similarity(gold, pred_netlist, timeout=ged_timeout)
    sim_success = run_ngspice_simulation(pred_netlist, timeout=ngspice_timeout)

    gold_tokens = gold.split()
    pred_tokens = pred_netlist.split()
    bleu = sentence_bleu(
        [gold_tokens],
        pred_tokens,
        smoothing_function=SmoothingFunction().method1,
    )

    gold_comp = extract_component_types(gold)
    pred_comp = extract_component_types(pred_netlist)
    comp_metrics = compute_component_f1(gold_comp, pred_comp)

    return {
        "idx": idx,
        "prompt": prompt,
        "gold": gold,
        "prediction": pred,
        "pred_netlist": pred_netlist,
        "exact_match": exact_match,
        "valid": valid,
        "ged_similarity": ged_sim,
        "simulation_success": sim_success,
        "bleu": bleu,
        "component_precision": comp_metrics["precision"],
        "component_recall": comp_metrics["recall"],
        "component_f1": comp_metrics["f1"],
    }


def evaluate(
    adapter: str,
    test_path: str = "data/spice/test.json",
    model_name: str | None = None,
    batch_size: int = 32,
    max_new_tokens: int = 2048,
    output_path: str | None = None,
    attn_implementation: str = "flash_attention_2",
    grammar_file: str | None = None,
    include_grammar: bool = True,
    task: str = "program",
    ngspice_timeout: float = 10.0,
    ged_timeout: float = 5.0,
):
    peft_config = PeftConfig.from_pretrained(adapter)
    base_model_name = model_name or peft_config.base_model_name_or_path
    assert base_model_name is not None, "No model_name provided and adapter config has no base_model_name_or_path"

    model = load_base_model(base_model_name, attn_implementation=attn_implementation)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    processing_class = load_processor(base_model_name)
    tokenizer = get_tokenizer(processing_class)
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
        skip_indices = set()
        for i, (ex, gex) in enumerate(zip(examples, grammar_data)):
            assert ex["query"] == gex["query"], (
                f"Query mismatch: {ex['query']!r} vs {gex['query']!r}"
            )
            if gex["minimal_grammar"] is None:
                skip_indices.add(i)
            else:
                ex["minimal_grammar"] = extract_grammar_from_output(gex["minimal_grammar"])
        if skip_indices:
            print(f"WARNING: Skipping {len(skip_indices)} examples with missing grammar predictions")
            examples = [ex for i, ex in enumerate(examples) if i not in skip_indices]
    else:
        print("Using gold grammars from test data")

    if shutil.which("ngspice") is None:
        raise RuntimeError("ngspice not found on PATH (install with: apt-get install -y ngspice)")

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

    eval_args = [
        (i, ex["program"], prompt, pred, ged_timeout, ngspice_timeout)
        for i, (ex, prompt, pred) in enumerate(zip(examples, prompts, predictions))
    ]

    results_unordered = []
    num_workers = min(16, os.cpu_count() or 1, len(eval_args))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_evaluate_single, args): args[0] for args in eval_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results_unordered.append(future.result())

    results = sorted(results_unordered, key=lambda r: r.pop("idx"))

    total = len(results)
    exact_count = sum(1 for r in results if r["exact_match"])
    valid_count = sum(1 for r in results if r["valid"])
    sim_count = sum(1 for r in results if r["simulation_success"])
    ged_sims = [r["ged_similarity"] for r in results]
    bleus = [r["bleu"] for r in results]
    comp_f1s = [r["component_f1"] for r in results]

    metrics = {
        "ged_similarity": sum(ged_sims) / len(ged_sims) if ged_sims else 0.0,
        "simulation_success": sim_count / total if total > 0 else 0.0,
        "syntax_validity": valid_count / total if total > 0 else 0.0,
        "exact_match": exact_count / total if total > 0 else 0.0,
        "bleu": sum(bleus) / len(bleus) if bleus else 0.0,
        "component_f1": sum(comp_f1s) / len(comp_f1s) if comp_f1s else 0.0,
        "correct": exact_count,
        "total": total,
    }

    print(f"\n--- Primary metrics ---")
    print(f"GED similarity:       {metrics['ged_similarity']:.4f}")
    print(f"Simulation success:   {metrics['simulation_success']:.4f} ({sim_count}/{total})")
    print(f"Syntax validity:      {metrics['syntax_validity']:.4f} ({valid_count}/{total})")
    print(f"\n--- Secondary metrics ---")
    print(f"Exact match:          {metrics['exact_match']:.4f} ({exact_count}/{total})")
    print(f"BLEU:                 {metrics['bleu']:.4f}")
    print(f"Component F1:         {metrics['component_f1']:.4f}")

    if output_path:
        save_results(metrics, results, output_path)


def evaluate_predictions(
    predictions_path: str,
    output_path: str,
    ngspice_timeout: float = 10.0,
    ged_timeout: float = 5.0,
    num_workers: int | None = None,
):
    """Evaluate a predictions JSON (from icl.py) for SPICE."""
    have_ngspice = shutil.which("ngspice") is not None
    if not have_ngspice:
        print("ngspice not found on PATH — simulation_success will be skipped")

    with open(predictions_path) as f:
        preds = json.load(f)["data"]

    eval_args = [
        (
            i,
            entry["gold_program"],
            "",
            entry.get("raw_prediction") or "",
            ged_timeout,
            ngspice_timeout,
        )
        for i, entry in enumerate(preds)
    ]

    if num_workers is None:
        num_workers = min(16, os.cpu_count() or 1, max(1, len(eval_args)))
    results_unordered = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_evaluate_single, args): args[0] for args in eval_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results_unordered.append(future.result())
    results = sorted(results_unordered, key=lambda r: r.pop("idx"))

    if not have_ngspice:
        for r in results:
            r["simulation_success"] = None

    total = len(results)
    exact_count = sum(1 for r in results if r["exact_match"])
    valid_count = sum(1 for r in results if r["valid"])
    ged_sims = [r["ged_similarity"] for r in results]
    bleus = [r["bleu"] for r in results]
    comp_f1s = [r["component_f1"] for r in results]

    metrics = {
        "ged_similarity": sum(ged_sims) / len(ged_sims) if ged_sims else 0.0,
        "syntax_validity": valid_count / total if total > 0 else 0.0,
        "exact_match": exact_count / total if total > 0 else 0.0,
        "accuracy": exact_count / total if total > 0 else 0.0,
        "bleu": sum(bleus) / len(bleus) if bleus else 0.0,
        "component_f1": sum(comp_f1s) / len(comp_f1s) if comp_f1s else 0.0,
        "correct": exact_count,
        "total": total,
    }
    if have_ngspice:
        sim_count = sum(1 for r in results if r["simulation_success"])
        metrics["simulation_success"] = sim_count / total if total > 0 else 0.0

    print(f"GED similarity:     {metrics['ged_similarity']:.4f}")
    print(f"Syntax validity:    {metrics['syntax_validity']:.4f} ({valid_count}/{total})")
    print(f"Exact match:        {metrics['exact_match']:.4f} ({exact_count}/{total})")
    print(f"Component F1:       {metrics['component_f1']:.4f}")
    print(f"BLEU:               {metrics['bleu']:.4f}")
    if "simulation_success" in metrics:
        sim_count = sum(1 for r in results if r["simulation_success"])
        print(f"Simulation success: {metrics['simulation_success']:.4f} ({sim_count}/{total})")

    save_results(metrics, results, output_path)


if __name__ == "__main__":
    fire.Fire(evaluate)
