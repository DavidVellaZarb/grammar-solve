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

KNOWN_OPERATIONS = {"add", "remove", "add_remove", "add_specific"}
PROTECTED_RULES = {"string", "number"}


def _parse_ops_range(value: int | list[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
        return (value, value + 1)
    elif isinstance(value, list):
        if len(value) != 2 or not all(isinstance(x, int) for x in value):
            raise ValueError(f"{name} list must be [low, high] of ints, got {value}")
        low, high = value
        if low < 1:
            raise ValueError(f"{name} low must be >= 1, got {low}")
        if high <= low:
            raise ValueError(f"{name} high must be > low, got [{low}, {high})")
        return (low, high)
    else:
        raise ValueError(f"{name} must be int or list[int], got {type(value)}")


def build_alternative_pool(data: list[dict]) -> dict[str, list[str]]:
    pool: dict[str, set[str]] = {}
    for ex in data:
        rules = parse_minimal_grammar(ex["minimal_grammar"])
        for rule_name, alts in rules.items():
            if rule_name not in pool:
                pool[rule_name] = set()
            pool[rule_name].update(alts)
    return {name: sorted(alts) for name, alts in pool.items()}


def add_specific_alternative(
    minimal_rules: dict[str, list[str]],
    pool: dict[str, list[str]],
    rng: random.Random,
) -> dict:
    minimal_set = {(rule, alt) for rule, alts in minimal_rules.items() for alt in alts}
    candidates = [
        (rule, alt)
        for rule, alts in pool.items()
        for alt in alts
        if (rule, alt) not in minimal_set
    ]
    if not candidates:
        raise ValueError("No candidates available to add from pool")
    rule_name, alt = rng.choice(candidates)
    if rule_name in minimal_rules:
        minimal_rules[rule_name].append(alt)
    else:
        minimal_rules[rule_name] = [alt]
    return {"rule": rule_name, "alternative": alt}


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
    n_ops: int | list = 1,
):
    ops = set(operations)
    if not ops:
        raise ValueError("operations must be non-empty")
    unknown = ops - KNOWN_OPERATIONS
    if unknown:
        raise ValueError(f"Unknown operations: {unknown}. Known: {KNOWN_OPERATIONS}")

    if "add_specific" in ops and "spice" not in grammar_file.lower():
        raise ValueError(
            f"add_specific is only supported for SPICE "
            f"(grammar_file must contain 'spice', got '{grammar_file}')"
        )

    if isinstance(n_ops, int) or (
        isinstance(n_ops, list)
        and len(n_ops) == 2
        and all(isinstance(x, int) for x in n_ops)
    ):
        shared_range = _parse_ops_range(n_ops, "n_ops")
        op_ranges = {op: shared_range for op in operations}
    else:
        if not isinstance(n_ops, list) or len(n_ops) != len(operations):
            raise ValueError(
                f"n_ops must be a single range or a list of {len(operations)} ranges "
                f"(one per operation), got {n_ops}"
            )
        op_ranges = {
            op: _parse_ops_range(r, f"n_ops[{i}]")
            for i, (op, r) in enumerate(zip(operations, n_ops))
        }

    with open(input_path) as f:
        data = json.load(f)["data"]

    lark_rules: dict[str, list[str]] = {}
    if "add" in ops or "add_remove" in ops:
        with open(grammar_file) as f:
            all_rules = parse_lark_grammar(f.read())
        excluded = GENERIC_TERMINALS | ENUM_TERMINALS if exclude_enum_terminals else GENERIC_TERMINALS
        lark_rules = filter_rules(all_rules, exclude=excluded)

    pool: dict[str, list[str]] = {}
    if "add_specific" in ops:
        pool = build_alternative_pool(data)

    rng = random.Random(seed)
    n_modify = round(len(data) * proportion)
    indices = set(rng.sample(range(len(data)), min(n_modify, len(data))))

    op_list = sorted(operations)
    stats = {op: 0 for op in op_list}

    for example in data:
        example["modifications"] = {"added": [], "removed": []}

    for idx in indices:
        example = data[idx]
        minimal_rules = parse_minimal_grammar(example["minimal_grammar"])

        for op in op_list:
            n = rng.randrange(*op_ranges[op])

            if op == "add":
                for _ in range(n):
                    try:
                        result = add_alternative(minimal_rules, lark_rules, rng)
                        example["modifications"]["added"].append(result)
                    except ValueError:
                        break
            elif op == "remove":
                for _ in range(n):
                    try:
                        result = remove_alternative(minimal_rules, rng)
                        example["modifications"]["removed"].append(result)
                    except ValueError:
                        break
            elif op == "add_remove":
                n_add = rng.randrange(*op_ranges[op])
                n_remove = rng.randrange(*op_ranges[op])
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
            elif op == "add_specific":
                for _ in range(n):
                    try:
                        result = add_specific_alternative(minimal_rules, pool, rng)
                        example["modifications"]["added"].append(result)
                    except ValueError:
                        break

            stats[op] += 1

        example["minimal_grammar"] = reconstruct_minimal_grammar(minimal_rules)

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
