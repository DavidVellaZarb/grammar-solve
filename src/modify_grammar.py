import json
import random

import fire

from grammar_utils import (
    ENUM_TERMINALS,
    GENERIC_TERMINALS,
    filter_rules,
    parse_lark_grammar,
    parse_minimal_grammar,
    reconstruct_minimal_grammar,
)

KNOWN_OPERATIONS = {"add", "remove", "add_remove"}
PROTECTED_RULES = {"string", "number"}


def add_alternative(
    minimal_rules: dict[str, list[str]],
    lark_rules: dict[str, list[str]],
    rng: random.Random,
) -> dict:
    minimal_set = {(rule, alt) for rule, alts in minimal_rules.items() for alt in alts}
    candidates = [
        (rule, alt)
        for rule, alts in lark_rules.items()
        for alt in alts
        if (rule, alt) not in minimal_set
    ]
    if not candidates:
        raise ValueError("No candidates available to add")

    rule_name, alt = rng.choice(candidates)
    if rule_name in minimal_rules:
        minimal_rules[rule_name].append(alt)
    else:
        minimal_rules[rule_name] = [alt]
    return {"rule": rule_name, "alternative": alt}


def remove_alternative(
    minimal_rules: dict[str, list[str]],
    rng: random.Random,
) -> dict:
    eligible = [r for r in minimal_rules if r not in PROTECTED_RULES]
    if not eligible:
        raise ValueError("No eligible rules available to remove from")

    rule_name = rng.choice(eligible)
    alts = minimal_rules[rule_name]
    idx = rng.randrange(len(alts))
    removed_alt = alts.pop(idx)
    if not alts:
        del minimal_rules[rule_name]
    return {"rule": rule_name, "alternative": removed_alt}


def modify_grammar(
    output_path: str,
    input_path: str = "data/smcalflow/test.json",
    operations: list[str] = ["add"],
    proportion: float = 0.1,
    grammar_file: str = "grammars/smcalflow.lark",
    exclude_enum_terminals: bool = True,
    seed: int | None = None,
    balanced: bool = False,
    n_ops: int | list[int] = 1,
):
    ops = set(operations)
    if not ops:
        raise ValueError("operations must be non-empty")
    unknown = ops - KNOWN_OPERATIONS
    if unknown:
        raise ValueError(f"Unknown operations: {unknown}. Known: {KNOWN_OPERATIONS}")

    if isinstance(n_ops, int):
        if n_ops < 1:
            raise ValueError(f"n_ops must be >= 1, got {n_ops}")
        n_ops_range = (n_ops, n_ops + 1)
    elif isinstance(n_ops, list):
        if len(n_ops) != 2 or not all(isinstance(x, int) for x in n_ops):
            raise ValueError(f"n_ops list must be [low, high] of ints, got {n_ops}")
        low, high = n_ops
        if low < 1:
            raise ValueError(f"n_ops low must be >= 1, got {low}")
        if high <= low:
            raise ValueError(f"n_ops high must be > low, got [{low}, {high})")
        n_ops_range = (low, high)
    else:
        raise ValueError(f"n_ops must be int or list[int], got {type(n_ops)}")

    with open(input_path) as f:
        data = json.load(f)["data"]

    lark_rules: dict[str, list[str]] = {}
    if "add" in ops or "add_remove" in ops:
        with open(grammar_file) as f:
            all_rules = parse_lark_grammar(f.read())
        excluded = GENERIC_TERMINALS | ENUM_TERMINALS if exclude_enum_terminals else GENERIC_TERMINALS
        lark_rules = filter_rules(all_rules, exclude=excluded)

    rng = random.Random(seed)
    n_modify = round(len(data) * proportion)
    indices = set(rng.sample(range(len(data)), min(n_modify, len(data))))

    op_list = sorted(operations)
    stats = {op: 0 for op in op_list}

    if balanced and len(op_list) > 1:
        shuffled_indices = list(indices)
        rng.shuffle(shuffled_indices)
        n = len(shuffled_indices)
        per_op = n // len(op_list)
        op_assignments = {}
        for i, op in enumerate(op_list):
            start = i * per_op
            end = (i + 1) * per_op if i < len(op_list) - 1 else n
            for j in range(start, end):
                op_assignments[shuffled_indices[j]] = op
    else:
        op_assignments = None

    for example in data:
        example["modifications"] = {"added": [], "removed": []}

    for idx in indices:
        example = data[idx]
        minimal_rules = parse_minimal_grammar(example["minimal_grammar"])
        op = op_assignments[idx] if op_assignments is not None else rng.choice(op_list)

        if op == "add":
            n = rng.randrange(*n_ops_range)
            for _ in range(n):
                try:
                    result = add_alternative(minimal_rules, lark_rules, rng)
                    example["modifications"]["added"].append(result)
                except ValueError:
                    break
        elif op == "remove":
            n = rng.randrange(*n_ops_range)
            for _ in range(n):
                try:
                    result = remove_alternative(minimal_rules, rng)
                    example["modifications"]["removed"].append(result)
                except ValueError:
                    break
        elif op == "add_remove":
            n_add = rng.randrange(*n_ops_range)
            n_remove = rng.randrange(*n_ops_range)
            for _ in range(n_add):
                try:
                    result = add_alternative(minimal_rules, lark_rules, rng)
                    example["modifications"]["added"].append(result)
                except ValueError:
                    break
            for _ in range(n_remove):
                try:
                    result = remove_alternative(minimal_rules, rng)
                    example["modifications"]["removed"].append(result)
                except ValueError:
                    break

        example["minimal_grammar"] = reconstruct_minimal_grammar(minimal_rules)
        stats[op] += 1

    metadata = {
        "total": len(data),
        "modified": len(indices),
        "proportion": len(indices) / len(data),
        "n_ops": n_ops,
        "operations": {
            op: {"count": count, "proportion": count / len(data) if len(data) else 0}
            for op, count in stats.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump({"metadata": metadata, "data": data}, f, indent=2)

    print(f"Input: {input_path} ({len(data)} examples)")
    print(f"Output: {output_path}")
    print(f"Modified {len(indices)} / {len(data)} examples (proportion={proportion})")
    for op, count in stats.items():
        print(f"  {op}: {count}")


if __name__ == "__main__":
    fire.Fire(modify_grammar)
