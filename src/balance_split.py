import json
import os
import random

import fire

from grammar_utils import parse_minimal_grammar


def _extract_rules(example: dict) -> set[str]:
    rules = parse_minimal_grammar(example["minimal_grammar"])
    return {f"{name} ::= {alt}" for name, alts in rules.items() for alt in alts}


def _has_any_rule(example: dict, target_rules: set[str]) -> bool:
    return bool(_extract_rules(example) & target_rules)


def balance(
    train_path: str = "data/smcalflow/train.json",
    test_path: str = "data/smcalflow/test.json",
    valid_path: str = "data/smcalflow/valid.json",
    output_dir: str = "data/smcalflow/",
    max_train_df: float = 0.05,
    min_test_df: float = 0.05,
    seed: int = 42,
):
    with open(train_path) as f:
        train_data = json.load(f)["data"]
    with open(test_path) as f:
        test_data = json.load(f)["data"]
    with open(valid_path) as f:
        valid_data = json.load(f)["data"]

    orig_train_size = len(train_data)
    orig_test_size = len(test_data)
    orig_valid_size = len(valid_data)

    # Extract rule sets for each example
    train_rule_sets = [_extract_rules(ex) for ex in train_data]
    test_rule_sets = [_extract_rules(ex) for ex in test_data]

    # Collect all unique rules
    all_rules = set()
    for rs in train_rule_sets:
        all_rules |= rs
    for rs in test_rule_sets:
        all_rules |= rs

    # Compute document frequencies
    train_df = {}
    test_df = {}
    for rule in all_rules:
        train_count = sum(1 for rs in train_rule_sets if rule in rs)
        test_count = sum(1 for rs in test_rule_sets if rule in rs)
        train_df[rule] = train_count / len(train_data) if train_data else 0
        test_df[rule] = test_count / len(test_data) if test_data else 0

    # Identify imbalanced rules
    imbalanced_rules = set()
    for rule in all_rules:
        # Rare in train but common in test
        if train_df[rule] < max_train_df and test_df[rule] >= min_test_df:
            imbalanced_rules.add(rule)
        # Test-only rules (never seen in train)
        train_count = sum(1 for rs in train_rule_sets if rule in rs)
        test_count = sum(1 for rs in test_rule_sets if rule in rs)
        if train_count == 0 and test_count > 0:
            imbalanced_rules.add(rule)

    print(f"Found {len(imbalanced_rules)} imbalanced rules:")
    for rule in sorted(imbalanced_rules):
        print(f"  {rule}")
        print(f"    train_df={train_df[rule]:.4f}, test_df={test_df[rule]:.4f}")

    if not imbalanced_rules:
        print("No imbalanced rules found. No changes needed.")
        return

    # Partition each split into clean/problem
    test_clean = [ex for ex in test_data if not _has_any_rule(ex, imbalanced_rules)]
    test_problem = [ex for ex in test_data if _has_any_rule(ex, imbalanced_rules)]

    valid_clean = [ex for ex in valid_data if not _has_any_rule(ex, imbalanced_rules)]
    valid_problem = [ex for ex in valid_data if _has_any_rule(ex, imbalanced_rules)]

    train_clean = [ex for ex in train_data if not _has_any_rule(ex, imbalanced_rules)]
    train_problem = [ex for ex in train_data if _has_any_rule(ex, imbalanced_rules)]

    print(f"\nPartition sizes:")
    print(f"  Train: {len(train_clean)} clean, {len(train_problem)} problem")
    print(f"  Valid: {len(valid_clean)} clean, {len(valid_problem)} problem")
    print(f"  Test:  {len(test_clean)} clean, {len(test_problem)} problem")

    # Need enough clean train examples to backfill test + valid
    backfill_needed = len(test_problem) + len(valid_problem)
    if len(train_clean) < backfill_needed:
        print(
            f"\nWarning: Not enough clean train examples ({len(train_clean)}) "
            f"to backfill test+valid ({backfill_needed}). "
            f"Will use all available ({len(train_clean)})."
        )

    # Swap: problem test/valid → train, clean train → test/valid
    rng = random.Random(seed)

    # Shuffle clean train for random backfill sampling
    train_clean_shuffled = list(train_clean)
    rng.shuffle(train_clean_shuffled)

    # Backfill test first, then valid
    test_backfill_count = min(len(test_problem), len(train_clean_shuffled))
    test_backfill = train_clean_shuffled[:test_backfill_count]
    remaining_clean = train_clean_shuffled[test_backfill_count:]

    valid_backfill_count = min(len(valid_problem), len(remaining_clean))
    valid_backfill = remaining_clean[:valid_backfill_count]
    remaining_clean = remaining_clean[valid_backfill_count:]

    # Build new splits
    new_test = test_clean + test_backfill
    new_valid = valid_clean + valid_backfill
    new_train = remaining_clean + train_problem + test_problem + valid_problem

    # Shuffle to avoid ordering artifacts
    rng.shuffle(new_train)
    rng.shuffle(new_test)
    rng.shuffle(new_valid)

    print(f"\nNew split sizes:")
    print(f"  Train: {len(new_train)} (was {orig_train_size})")
    print(f"  Valid: {len(new_valid)} (was {orig_valid_size})")
    print(f"  Test:  {len(new_test)} (was {orig_test_size})")
    print(f"\nExamples moved:")
    print(f"  Test problem → Train: {len(test_problem)}")
    print(f"  Valid problem → Train: {len(valid_problem)}")
    print(f"  Train clean → Test: {test_backfill_count}")
    print(f"  Train clean → Valid: {valid_backfill_count}")

    # Verify sizes preserved
    total_orig = orig_train_size + orig_test_size + orig_valid_size
    total_new = len(new_train) + len(new_test) + len(new_valid)
    assert total_new == total_orig, (
        f"Total examples changed: {total_orig} → {total_new}"
    )

    metadata = {
        "max_train_df": max_train_df,
        "min_test_df": min_test_df,
        "seed": seed,
        "imbalanced_rules": sorted(imbalanced_rules),
        "n_imbalanced_rules": len(imbalanced_rules),
        "original_sizes": {
            "train": orig_train_size,
            "valid": orig_valid_size,
            "test": orig_test_size,
        },
        "new_sizes": {
            "train": len(new_train),
            "valid": len(new_valid),
            "test": len(new_test),
        },
        "swapped": {
            "test_problem_to_train": len(test_problem),
            "valid_problem_to_train": len(valid_problem),
            "train_clean_to_test": test_backfill_count,
            "train_clean_to_valid": valid_backfill_count,
        },
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [
        ("train_balanced", new_train),
        ("valid_balanced", new_valid),
        ("test_balanced", new_test),
    ]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump({"metadata": metadata, "data": data}, f, indent=2)
        print(f"Saved {path} ({len(data)} examples)")


if __name__ == "__main__":
    fire.Fire(balance)
