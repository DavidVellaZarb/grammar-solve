import heapq
import json
import os
import statistics
import tempfile

import fire
import matplotlib.pyplot as plt
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt_messages, load_raw_data
from eval_utils import save_results
from grammar_utils import extract_grammar_from_output


def _reconstruct_task(domain_file, problem_pddl_str):
    """Parse and ground a PDDL problem. Returns (task, heuristic)."""
    from pyperplan.grounding import ground
    from pyperplan.heuristics.relaxation import hFFHeuristic
    from pyperplan.pddl.parser import Parser

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pddl", delete=False
    ) as f:
        f.write(problem_pddl_str)
        problem_path = f.name

    try:
        parser = Parser(domain_file, problem_path)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        task = ground(problem)
        heuristic = hFFHeuristic(task)
        return task, heuristic
    finally:
        os.unlink(problem_path)


def seeded_gbfs(task, heuristic, seed_actions=None, max_nodes=100_000):
    """GBFS with optional queue seeding from a predicted plan."""
    from pyperplan.search.searchspace import make_child_node, make_root_node

    root = make_root_node(task.initial_state)
    init_h = heuristic(root)

    if init_h == float("inf"):
        seed_total = len(seed_actions) if seed_actions else 0
        return {
            "solved": False, "nodes_created": 1, "nodes_expanded": 0,
            "plan_length": None, "seed_valid": 0, "seed_total": seed_total,
        }

    open_list = []
    tiebreaker = 0
    heapq.heappush(open_list, (init_h, tiebreaker, root))
    tiebreaker += 1

    state_cost = {task.initial_state: 0}
    nodes_created = 1
    nodes_expanded = 0

    op_map = {}
    for op in task.operators:
        op_map[op.name.lower().strip()] = op

    seed_valid = 0
    seed_total = len(seed_actions) if seed_actions else 0
    if seed_actions:
        node = root
        for action_str in seed_actions:
            key = action_str.lower().strip()
            op = op_map.get(key)
            if op is None or not op.applicable(node.state):
                break
            succ_state = op.apply(node.state)
            child = make_child_node(node, op, succ_state)
            child_h = heuristic(child)
            if child_h == float("inf"):
                break
            seed_valid += 1
            if succ_state not in state_cost or state_cost[succ_state] > child.g:
                heapq.heappush(open_list, (child_h, tiebreaker, child))
                tiebreaker += 1
                state_cost[succ_state] = child.g
                nodes_created += 1
            node = child

    while open_list and nodes_expanded < max_nodes:
        _, _, node = heapq.heappop(open_list)

        if state_cost.get(node.state, float("inf")) < node.g:
            continue

        nodes_expanded += 1

        if task.goal_reached(node.state):
            plan = node.extract_solution()
            return {
                "solved": True,
                "nodes_created": nodes_created,
                "nodes_expanded": nodes_expanded,
                "plan_length": len(plan),
                "seed_valid": seed_valid,
                "seed_total": seed_total,
            }

        for op, succ_state in task.get_successor_states(node.state):
            succ_g = node.g + 1
            if succ_state not in state_cost or state_cost[succ_state] > succ_g:
                child = make_child_node(node, op, succ_state)
                child_h = heuristic(child)
                if child_h != float("inf"):
                    heapq.heappush(open_list, (child_h, tiebreaker, child))
                    tiebreaker += 1
                    state_cost[succ_state] = succ_g
                    nodes_created += 1

    return {
        "solved": False,
        "nodes_created": nodes_created,
        "nodes_expanded": nodes_expanded,
        "plan_length": None,
        "seed_valid": seed_valid,
        "seed_total": seed_total,
    }


def _parse_plan(prediction_str):
    """Parse LLM prediction into list of action strings."""
    actions = []
    for line in prediction_str.strip().split("\n"):
        line = line.strip()
        if line and line.startswith("("):
            actions.append(line)
    return actions


def evaluate_gbfs_only(
    test_path: str = "data/pddl_blocksworld/test.json",
    domain_file: str = "pddl_domains/blocksworld/domain.pddl",
    output_path: str | None = None,
    max_nodes: int = 100_000,
):
    """Run GBFS without any LLM seeding. Reports baseline search effort."""
    assert os.path.exists(domain_file), f"Domain file not found: {domain_file}"

    examples = load_raw_data(test_path)
    results = []

    for ex in tqdm(examples, desc="GBFS-only"):
        assert "problem_pddl" in ex, "Test data must include 'problem_pddl' field"
        task, heuristic = _reconstruct_task(domain_file, ex["problem_pddl"])
        gbfs_result = seeded_gbfs(task, heuristic, seed_actions=None, max_nodes=max_nodes)
        results.append({
            "query": ex["query"],
            "gold": ex["program"],
            "solved": gbfs_result["solved"],
            "nodes_created": gbfs_result["nodes_created"],
            "nodes_expanded": gbfs_result["nodes_expanded"],
            "plan_length": gbfs_result["plan_length"],
        })

    metrics = _compute_metrics(results)
    _print_metrics("GBFS-only", metrics)

    if output_path:
        save_results(metrics, results, output_path)


def evaluate(
    adapter: str,
    test_path: str = "data/pddl_blocksworld/test.json",
    domain_file: str = "pddl_domains/blocksworld/domain.pddl",
    model_name: str | None = None,
    batch_size: int = 32,
    max_new_tokens: int = 1024,
    output_path: str | None = None,
    attn_implementation: str = "flash_attention_2",
    grammar_file: str | None = None,
    include_grammar: bool = True,
    task: str = "program",
    max_nodes: int = 100_000,
):
    """Evaluate LLM plans by seeding GBFS."""
    assert os.path.exists(domain_file), f"Domain file not found: {domain_file}"

    peft_config = PeftConfig.from_pretrained(adapter)
    base_model_name = model_name or peft_config.base_model_name_or_path
    assert base_model_name is not None

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
        assert len(grammar_data) == len(examples)
        skip_indices = set()
        for i, (ex, gex) in enumerate(zip(examples, grammar_data)):
            assert ex["query"] == gex["query"]
            if gex["minimal_grammar"] is None:
                skip_indices.add(i)
            else:
                ex["minimal_grammar"] = extract_grammar_from_output(gex["minimal_grammar"])
        if skip_indices:
            print(f"WARNING: Skipping {len(skip_indices)} examples with missing grammar predictions")
            examples = [ex for i, ex in enumerate(examples) if i not in skip_indices]
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
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        predictions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()

    results = []
    for ex, prompt, pred in tqdm(
        zip(examples, prompts, predictions), total=len(examples), desc="Seeded GBFS"
    ):
        assert "problem_pddl" in ex, "Test data must include 'problem_pddl' field"
        task_obj, heuristic = _reconstruct_task(domain_file, ex["problem_pddl"])

        pred_actions = _parse_plan(pred)

        seeded_result = seeded_gbfs(
            task_obj, heuristic, seed_actions=pred_actions, max_nodes=max_nodes
        )

        results.append({
            "prompt": prompt,
            "gold": ex["program"],
            "prediction": pred,
            "pred_actions": pred_actions,
            "solved": seeded_result["solved"],
            "nodes_created": seeded_result["nodes_created"],
            "nodes_expanded": seeded_result["nodes_expanded"],
            "plan_length": seeded_result["plan_length"],
            "seed_valid": seeded_result["seed_valid"],
            "seed_total": seeded_result["seed_total"],
        })

    metrics = _compute_metrics(results)
    _print_metrics("Seeded GBFS", metrics)

    if output_path:
        save_results(metrics, results, output_path)


def _compute_metrics(results):
    total = len(results)
    solved = [r for r in results if r["solved"]]

    nodes_created = [r["nodes_created"] for r in results]
    nodes_expanded = [r["nodes_expanded"] for r in results]

    metrics = {
        "success_rate": len(solved) / total if total > 0 else 0.0,
        "total": total,
        "solved": len(solved),
        "avg_nodes_created": statistics.mean(nodes_created) if nodes_created else 0.0,
        "avg_nodes_expanded": statistics.mean(nodes_expanded) if nodes_expanded else 0.0,
        "median_nodes_created": statistics.median(nodes_created) if nodes_created else 0.0,
        "median_nodes_expanded": statistics.median(nodes_expanded) if nodes_expanded else 0.0,
    }

    if any("seed_total" in r for r in results):
        seed_ratios = [
            r["seed_valid"] / r["seed_total"]
            for r in results
            if r.get("seed_total", 0) > 0
        ]
        if seed_ratios:
            metrics["avg_seed_valid_ratio"] = statistics.mean(seed_ratios)

    return metrics


def _print_metrics(label, metrics):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Success rate:          {metrics['success_rate']:.4f} "
          f"({metrics['solved']}/{metrics['total']})")
    print(f"  Avg nodes created:     {metrics['avg_nodes_created']:.1f}")
    print(f"  Avg nodes expanded:    {metrics['avg_nodes_expanded']:.1f}")
    print(f"  Median nodes created:  {metrics['median_nodes_created']:.1f}")
    print(f"  Median nodes expanded: {metrics['median_nodes_expanded']:.1f}")
    if "avg_seed_valid_ratio" in metrics:
        print(f"  Avg seed valid ratio:  {metrics['avg_seed_valid_ratio']:.4f}")


def plot(
    result_files: list[str],
    labels: list[str],
    output_path: str = "results/pddl/comparison.png",
    title: str | None = None,
):
    """Plot PDDL evaluation results: success rate + median nodes expanded."""
    all_results = []
    for path in result_files:
        with open(path) as f:
            all_results.append(json.load(f))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    num = len(result_files)
    x = range(num)
    bar_width = 0.5

    success_rates = [d.get("success_rate", 0.0) for d in all_results]
    bars1 = ax1.bar(x, success_rates, bar_width, color=plt.cm.tab10.colors[:num])
    for bar, val in zip(bars1, success_rates):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.1%}", ha="center", va="bottom", fontsize=9,
        )
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Success Rate")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Success Rate")

    nodes = [d.get("median_nodes_expanded", 0.0) for d in all_results]
    bars2 = ax2.bar(x, nodes, bar_width, color=plt.cm.tab10.colors[:num])
    for bar, val in zip(bars2, nodes):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(nodes) * 0.02,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9,
        )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Median Nodes Expanded")
    ax2.set_title("Search Effort")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    fire.Fire({
        "evaluate": evaluate,
        "evaluate_gbfs_only": evaluate_gbfs_only,
        "plot": plot,
    })
