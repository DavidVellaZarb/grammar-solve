import json
import os
import re
import shutil
import subprocess
import tempfile

import fire
import networkx as nx
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt_messages, load_raw_data
from eval_utils import save_results
from grammar_parser import _build_parser

GRAMMAR_PATH = "grammars/spice.lark"

COMPONENT_PREFIXES = set("RCLVIDQMXKJrclvidqmxkj")


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


# Number of net/node connections per component type (before value/model fields)
_COMP_NODE_COUNT = {
    "R": 2, "C": 2, "L": 2,
    "V": 2, "I": 2,
    "D": 2,
    "Q": 3,  # collector, base, emitter (optional substrate handled by +1 heuristic)
    "M": 4,  # drain, gate, source, bulk
    "J": 3,  # drain, gate, source
    "K": 0,  # coupled inductor: K1 L1 L2 value (L1/L2 are inductor names, not nets)
    "X": -1, # subcircuit: variable number of nodes
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
            # Subcircuit call: all tokens until last non-param token are nodes,
            # last one is subcircuit name. Take all except name and last non-'=' token.
            node_parts = []
            for part in parts[1:]:
                if "=" in part:
                    break
                node_parts.append(part)
            # Last one is subcircuit name, rest are nets
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
        # optimize_graph_edit_distance yields increasingly better upper bounds
        best_ged = ged_max  # worst case
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


def evaluate(
    adapter: str,
    test_path: str = "data/spice/test.json",
    model_name: str | None = None,
    batch_size: int = 4,
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

    # Check ngspice availability
    has_ngspice = shutil.which("ngspice") is not None
    if not has_ngspice:
        print("Warning: ngspice not found on PATH, simulation_success will be 0")

    prompts = []
    for ex in examples:
        messages = format_prompt_messages(ex, include_grammar=include_grammar, task=task)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating"):
        batch_prompts = prompts[i : i + batch_size]
        batch_examples = examples[i : i + batch_size]

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
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for ex, prompt, pred in zip(batch_examples, batch_prompts, predictions):
            gold = ex["program"]
            pred_netlist = extract_netlist(pred)

            exact_match = gold in pred
            valid = check_syntax_validity(pred_netlist)

            # GED similarity (primary metric)
            ged_sim = compute_ged_similarity(gold, pred_netlist, timeout=ged_timeout)

            # Simulation
            sim_success = False
            if has_ngspice:
                sim_success = run_ngspice_simulation(pred_netlist, timeout=ngspice_timeout)

            # BLEU (secondary)
            gold_tokens = gold.split()
            pred_tokens = pred_netlist.split()
            bleu = sentence_bleu(
                [gold_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method1,
            )

            # Component type match (secondary)
            gold_comp = extract_component_types(gold)
            pred_comp = extract_component_types(pred_netlist)
            comp_metrics = compute_component_f1(gold_comp, pred_comp)

            results.append({
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
            })

    total = len(results)
    exact_count = sum(1 for r in results if r["exact_match"])
    valid_count = sum(1 for r in results if r["valid"])
    sim_count = sum(1 for r in results if r["simulation_success"])
    ged_sims = [r["ged_similarity"] for r in results]
    bleus = [r["bleu"] for r in results]
    comp_f1s = [r["component_f1"] for r in results]

    metrics = {
        "accuracy": sum(ged_sims) / len(ged_sims) if ged_sims else 0.0,
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


if __name__ == "__main__":
    fire.Fire(evaluate)
