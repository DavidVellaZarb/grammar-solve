import json
import os
import random
import string
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
from lark import Lark
from tqdm import tqdm

from grammar_parser import _walk_tree

SKIP_RULES = {"plan"}

DOMAIN_CONFIGS = {
    "blocksworld": {
        "domain_file": "pddl_domains/blocksworld/domain.pddl",
        "grammar_path": "grammars/pddl_blocksworld.lark",
        "start_rule": "plan",
        "train_sizes": {"n_blocks": (4, 8)},
        "valid_sizes": {"n_blocks": (7, 10)},
        "test_sizes": {"n_blocks": (9, 14)},
    },
    "depot": {
        "domain_file": "pddl_domains/depot/domain.pddl",
        "grammar_path": "grammars/pddl_depot.lark",
        "start_rule": "plan",
        "train_sizes": {
            "n_depots": (1, 2), "n_distributors": (1, 2),
            "n_trucks": (1, 2), "n_crates": (2, 4), "n_pallets": (2, 3),
        },
        "valid_sizes": {
            "n_depots": (1, 2), "n_distributors": (1, 2),
            "n_trucks": (1, 2), "n_crates": (3, 5), "n_pallets": (3, 4),
        },
        "test_sizes": {
            "n_depots": (2, 3), "n_distributors": (1, 2),
            "n_trucks": (1, 2), "n_crates": (4, 7), "n_pallets": (3, 5),
        },
    },
    "satellite": {
        "domain_file": "pddl_domains/satellite/domain.pddl",
        "grammar_path": "grammars/pddl_satellite.lark",
        "start_rule": "plan",
        "train_sizes": {
            "n_satellites": (1, 2), "n_instruments_per_sat": (1, 2),
            "n_modes": (1, 3), "n_directions": (3, 6), "n_observations": (1, 3),
        },
        "valid_sizes": {
            "n_satellites": (1, 2), "n_instruments_per_sat": (1, 3),
            "n_modes": (2, 3), "n_directions": (4, 7), "n_observations": (2, 4),
        },
        "test_sizes": {
            "n_satellites": (2, 3), "n_instruments_per_sat": (2, 3),
            "n_modes": (2, 3), "n_directions": (5, 9), "n_observations": (3, 6),
        },
    },
}

BLOCK_NAMES = list(string.ascii_lowercase)


def _random_tower_state(blocks, rng):
    """Generate a random configuration of blocks as a list of towers."""
    blocks = list(blocks)
    rng.shuffle(blocks)
    towers = []
    i = 0
    while i < len(blocks):
        tower_len = rng.randint(1, max(1, len(blocks) - i))
        towers.append(blocks[i:i + tower_len])
        i += tower_len
    return towers


def _towers_to_predicates(towers):
    """Convert tower config to PDDL predicates."""
    preds = ["(handempty)"]
    for tower in towers:
        preds.append(f"(ontable {tower[0]})")
        preds.append(f"(clear {tower[-1]})")
        for i in range(1, len(tower)):
            preds.append(f"(on {tower[i]} {tower[i-1]})")
    return preds


def _generate_blocksworld(n_blocks, seed):
    rng = random.Random(seed)
    blocks = BLOCK_NAMES[:n_blocks]

    for _ in range(100):
        init_towers = _random_tower_state(blocks, rng)
        goal_towers = _random_tower_state(blocks, rng)
        init_preds = set(_towers_to_predicates(init_towers))
        goal_preds_full = set(_towers_to_predicates(goal_towers))
        goal_preds = {p for p in goal_preds_full if not p.startswith("(handempty")}
        if init_preds != goal_preds_full:
            break
    else:
        return None

    objects_str = " ".join(blocks)
    init_str = " ".join(sorted(init_preds))
    goal_str = " ".join(sorted(goal_preds))

    query = (
        f"(:objects {objects_str} - block)\n"
        f"(:init {init_str})\n"
        f"(:goal (and {goal_str}))"
    )

    problem_pddl = (
        f"(define (problem bw-{seed})\n"
        f"  (:domain blocksworld)\n"
        f"  (:objects {objects_str})\n"
        f"  (:init {init_str})\n"
        f"  (:goal (and {goal_str}))\n"
        f")"
    )

    return {"query": query, "problem_pddl": problem_pddl, "domain": "blocksworld"}


def _generate_depot(n_depots, n_distributors, n_trucks, n_crates, n_pallets, seed):
    rng = random.Random(seed)

    depots = [f"depot{i}" for i in range(n_depots)]
    distributors = [f"distributor{i}" for i in range(n_distributors)]
    places = depots + distributors
    trucks = [f"truck{i}" for i in range(n_trucks)]
    pallets = [f"pallet{i}" for i in range(n_pallets)]
    crates = [f"crate{i}" for i in range(n_crates)]
    n_hoists = len(places)
    hoists = [f"hoist{i}" for i in range(n_hoists)]

    obj_parts = []
    obj_parts.append(" ".join(depots) + " - depot")
    obj_parts.append(" ".join(distributors) + " - distributor")
    obj_parts.append(" ".join(trucks) + " - truck")
    obj_parts.append(" ".join(pallets) + " - pallet")
    obj_parts.append(" ".join(crates) + " - crate")
    obj_parts.append(" ".join(hoists) + " - hoist")

    init_preds = []

    for i, place in enumerate(places):
        init_preds.append(f"(at {hoists[i]} {place})")
        init_preds.append(f"(available {hoists[i]})")

    for i, pallet in enumerate(pallets):
        place = places[i % len(places)]
        init_preds.append(f"(at {pallet} {place})")
        init_preds.append(f"(clear {pallet})")

    pallet_tops = {p: p for p in pallets}
    for crate in crates:
        pallet = rng.choice(pallets)
        top = pallet_tops[pallet]
        place = None
        for pred in init_preds:
            if pred.startswith(f"(at {pallet} "):
                place = pred.split()[-1].rstrip(")")
                break
        assert place is not None, f"Could not find location of pallet {pallet}"
        init_preds.append(f"(at {crate} {place})")
        init_preds.append(f"(on {crate} {top})")
        init_preds = [p for p in init_preds if p != f"(clear {top})"]
        init_preds.append(f"(clear {crate})")
        pallet_tops[pallet] = crate

    for truck in trucks:
        place = rng.choice(places)
        init_preds.append(f"(at {truck} {place})")

    goal_preds = []
    moved = set()
    target_moves = max(1, n_crates // 2)
    crates_shuffled = list(crates)
    rng.shuffle(crates_shuffled)
    for crate in crates_shuffled[:target_moves]:
        target_pallet = rng.choice(pallets)
        current_surface = None
        for pred in init_preds:
            if pred.startswith(f"(on {crate} "):
                current_surface = pred.split()[-1].rstrip(")")
                break
        if current_surface != target_pallet:
            goal_preds.append(f"(on {crate} {target_pallet})")
            moved.add(crate)

    if not goal_preds:
        crate = crates[0]
        current_surface = None
        for pred in init_preds:
            if pred.startswith(f"(on {crate} "):
                current_surface = pred.split()[-1].rstrip(")")
                break
        other_pallets = [p for p in pallets if p != current_surface]
        if other_pallets:
            goal_preds.append(f"(on {crate} {rng.choice(other_pallets)})")
        else:
            return None

    objects_line = " ".join(obj_parts)
    init_str = " ".join(sorted(init_preds))
    goal_str = " ".join(sorted(goal_preds))

    query = (
        f"(:objects {objects_line})\n"
        f"(:init {init_str})\n"
        f"(:goal (and {goal_str}))"
    )

    problem_pddl = (
        f"(define (problem depot-{seed})\n"
        f"  (:domain depot)\n"
        f"  (:objects {objects_line})\n"
        f"  (:init {init_str})\n"
        f"  (:goal (and {goal_str}))\n"
        f")"
    )

    return {"query": query, "problem_pddl": problem_pddl, "domain": "depot"}


def _generate_satellite(n_satellites, n_instruments_per_sat, n_modes, n_directions,
                         n_observations, seed):
    rng = random.Random(seed)

    satellites = [f"satellite{i}" for i in range(n_satellites)]
    all_modes = ["thermograph", "image", "spectrograph", "infrared"][:n_modes]
    directions = [f"star{i}" for i in range(n_directions)]
    n_gs = max(1, n_satellites)
    groundstations = [f"groundstation{i}" for i in range(n_gs)]
    all_directions = directions + groundstations

    instruments = []
    instrument_assignments = {}
    for sat_idx, sat in enumerate(satellites):
        for j in range(n_instruments_per_sat):
            inst = f"instrument{sat_idx * n_instruments_per_sat + j}"
            instruments.append(inst)
            instrument_assignments[inst] = sat

    obj_parts = []
    obj_parts.append(" ".join(satellites) + " - satellite")
    obj_parts.append(" ".join(instruments) + " - instrument")
    obj_parts.append(" ".join(all_modes) + " - mode")
    obj_parts.append(" ".join(all_directions) + " - direction")

    init_preds = []

    for inst, sat in instrument_assignments.items():
        init_preds.append(f"(on_board {inst} {sat})")

    inst_modes = {}
    for inst in instruments:
        n_supported = rng.randint(1, min(2, len(all_modes)))
        supported = rng.sample(all_modes, n_supported)
        inst_modes[inst] = supported
        for mode in supported:
            init_preds.append(f"(supports {inst} {mode})")

    for inst in instruments:
        cal_target = rng.choice(groundstations)
        init_preds.append(f"(calibration_target {inst} {cal_target})")

    for sat in satellites:
        init_dir = rng.choice(all_directions)
        init_preds.append(f"(pointing {sat} {init_dir})")
        init_preds.append(f"(power_avail {sat})")

    goal_preds = []
    possible_goals = [(d, m) for d in directions for m in all_modes]
    rng.shuffle(possible_goals)
    n_obs = min(n_observations, len(possible_goals))
    for d, m in possible_goals[:n_obs]:
        goal_preds.append(f"(have_image {d} {m})")

    if not goal_preds:
        return None

    objects_line = " ".join(obj_parts)
    init_str = " ".join(sorted(init_preds))
    goal_str = " ".join(sorted(goal_preds))

    query = (
        f"(:objects {objects_line})\n"
        f"(:init {init_str})\n"
        f"(:goal (and {goal_str}))"
    )

    problem_pddl = (
        f"(define (problem sat-{seed})\n"
        f"  (:domain satellite)\n"
        f"  (:objects {objects_line})\n"
        f"  (:init {init_str})\n"
        f"  (:goal (and {goal_str}))\n"
        f")"
    )

    return {"query": query, "problem_pddl": problem_pddl, "domain": "satellite"}


def _gbfs_with_timeout(task, heuristic, timeout_sec=30):
    """GBFS with heuristic + timeout. Returns list of operators or None."""
    import heapq
    from pyperplan.search.searchspace import make_child_node, make_root_node

    root = make_root_node(task.initial_state)
    init_h = heuristic(root)
    if init_h == float("inf"):
        return None

    open_list = []
    tiebreaker = 0
    heapq.heappush(open_list, (init_h, tiebreaker, root))
    tiebreaker += 1

    state_cost = {task.initial_state: 0}
    start_time = time.time()

    while open_list:
        if time.time() - start_time > timeout_sec:
            return None

        _, _, node = heapq.heappop(open_list)

        if state_cost.get(node.state, float("inf")) < node.g:
            continue

        if task.goal_reached(node.state):
            return node.extract_solution()

        for op, succ_state in task.get_successor_states(node.state):
            succ_g = node.g + 1
            if succ_state not in state_cost or state_cost[succ_state] > succ_g:
                child = make_child_node(node, op, succ_state)
                child_h = heuristic(child)
                if child_h != float("inf"):
                    heapq.heappush(open_list, (child_h, tiebreaker, child))
                    tiebreaker += 1
                    state_cost[succ_state] = succ_g

    return None


def _solve_problem(domain_path, problem_pddl_str, timeout_sec=30):
    """Solve a PDDL problem. Returns list of operator name strings or None."""
    from pyperplan.grounding import ground
    from pyperplan.heuristics.relaxation import hFFHeuristic
    from pyperplan.pddl.parser import Parser

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pddl", delete=False
    ) as f:
        f.write(problem_pddl_str)
        problem_path = f.name

    try:
        parser = Parser(domain_path, problem_path)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        task = ground(problem)
        heuristic = hFFHeuristic(task)
        solution = _gbfs_with_timeout(task, heuristic, timeout_sec)
        if solution is None:
            return None
        return [op.name for op in solution]
    except Exception:
        return None
    finally:
        os.unlink(problem_path)


def _extract_grammar(plan_str, parser):
    """Parse plan string and extract minimal grammar."""
    try:
        tree = parser.parse(plan_str)
    except Exception as e:
        raise AssertionError(f"Failed to parse plan with Lark grammar: {e}\nPlan: {plan_str}")

    rules = {}
    _walk_tree(tree, rules)
    rules = {k: v for k, v in rules.items() if k not in SKIP_RULES}
    lines = [f"{name} ::= {' | '.join(alts)}" for name, alts in rules.items()]
    return "\n".join(lines)


def _generate_one(args):
    """Generate one problem, solve it, extract grammar. Returns dict or None."""
    domain_name, size_params, seed, domain_file, grammar_path, timeout_sec = args

    if domain_name == "blocksworld":
        rng = random.Random(seed)
        n_blocks = rng.randint(*size_params["n_blocks"])
        result = _generate_blocksworld(n_blocks, seed)
    elif domain_name == "depot":
        rng = random.Random(seed)
        n_depots = rng.randint(*size_params["n_depots"])
        n_distributors = rng.randint(*size_params["n_distributors"])
        n_trucks = rng.randint(*size_params["n_trucks"])
        n_crates = rng.randint(*size_params["n_crates"])
        n_pallets = rng.randint(*size_params["n_pallets"])
        result = _generate_depot(n_depots, n_distributors, n_trucks, n_crates, n_pallets, seed)
    elif domain_name == "satellite":
        rng = random.Random(seed)
        n_sats = rng.randint(*size_params["n_satellites"])
        n_inst = rng.randint(*size_params["n_instruments_per_sat"])
        n_modes = rng.randint(*size_params["n_modes"])
        n_dirs = rng.randint(*size_params["n_directions"])
        n_obs = rng.randint(*size_params["n_observations"])
        result = _generate_satellite(n_sats, n_inst, n_modes, n_dirs, n_obs, seed)
    else:
        raise ValueError(f"Unknown domain: {domain_name}")

    if result is None:
        return None

    solution = _solve_problem(domain_file, result["problem_pddl"], timeout_sec)
    if solution is None:
        return None

    plan_str = "\n".join(solution)

    lark_parser = Lark(
        open(grammar_path).read(),
        start="plan", parser="earley", keep_all_tokens=True,
    )
    grammar = _extract_grammar(plan_str, lark_parser)

    return {
        "query": result["query"],
        "minimal_grammar": grammar,
        "program": plan_str,
        "problem_pddl": result["problem_pddl"],
        "domain": domain_name,
    }


def _load_domain(
    domain_name: str,
    output_dir: str,
    n_train: int = 10000,
    n_valid: int = 500,
    n_test: int = 500,
    timeout_sec: int = 30,
    n_workers: int | None = None,
    max_attempts_factor: int = 5,
):
    config = DOMAIN_CONFIGS[domain_name]
    domain_file = config["domain_file"]
    grammar_path = config["grammar_path"]

    assert os.path.exists(domain_file), f"Domain file not found: {domain_file}"
    assert os.path.exists(grammar_path), f"Grammar file not found: {grammar_path}"

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    os.makedirs(output_dir, exist_ok=True)

    splits = {
        "train": (n_train, config["train_sizes"]),
        "valid": (n_valid, config["valid_sizes"]),
        "test": (n_test, config["test_sizes"]),
    }

    seed_offset = 0

    for split_name, (n_target, size_params) in splits.items():
        print(f"\n{'='*60}")
        print(f"Generating {split_name} split ({n_target} examples) for {domain_name}")
        print(f"Size params: {size_params}")
        print(f"{'='*60}")

        results = []
        n_attempts = 0
        n_timeouts = 0
        max_attempts = n_target * max_attempts_factor

        pbar = tqdm(total=n_target, desc=f"  {split_name}")

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            while len(results) < n_target and n_attempts < max_attempts:
                batch_size = min(
                    (n_target - len(results)) * 2,
                    n_workers * 8,
                )
                args_list = [
                    (domain_name, size_params, seed_offset + n_attempts + i,
                     domain_file, grammar_path, timeout_sec)
                    for i in range(batch_size)
                ]
                n_attempts += batch_size

                futures = {pool.submit(_generate_one, a): a for a in args_list}
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=timeout_sec + 10)
                    except Exception:
                        n_timeouts += 1
                        continue

                    if result is not None and len(results) < n_target:
                        results.append(result)
                        pbar.update(1)
                    elif result is None:
                        n_timeouts += 1

        pbar.close()
        seed_offset += n_attempts

        assert len(results) >= n_target, (
            f"Only generated {len(results)}/{n_target} examples for {split_name}. "
            f"{n_timeouts} failures/timeouts out of {n_attempts} attempts. "
            f"Consider adjusting problem sizes or timeout."
        )

        results = results[:n_target]

        out_path = os.path.join(output_dir, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump({"data": results}, f, indent=2)
        print(f"  Wrote {len(results)} entries to {out_path}")
        print(f"  Failures/timeouts: {n_timeouts}/{n_attempts} "
              f"({100 * n_timeouts / max(1, n_attempts):.1f}%)")

    print(f"\nDone! Data saved to {output_dir}/")


def load_blocksworld(
    output_dir: str = "data/pddl_blocksworld",
    n_train: int = 10000,
    n_valid: int = 500,
    n_test: int = 500,
    timeout_sec: int = 30,
    n_workers: int | None = None,
):
    _load_domain("blocksworld", output_dir, n_train, n_valid, n_test, timeout_sec, n_workers)


def load_depot(
    output_dir: str = "data/pddl_depot",
    n_train: int = 10000,
    n_valid: int = 500,
    n_test: int = 500,
    timeout_sec: int = 30,
    n_workers: int | None = None,
):
    _load_domain("depot", output_dir, n_train, n_valid, n_test, timeout_sec, n_workers)


def load_satellite(
    output_dir: str = "data/pddl_satellite",
    n_train: int = 10000,
    n_valid: int = 500,
    n_test: int = 500,
    timeout_sec: int = 30,
    n_workers: int | None = None,
):
    _load_domain("satellite", output_dir, n_train, n_valid, n_test, timeout_sec, n_workers)


def load(
    n_train: int = 10000,
    n_valid: int = 500,
    n_test: int = 500,
    timeout_sec: int = 30,
    n_workers: int | None = None,
):
    """Load all three PDDL domains."""
    load_blocksworld(n_train=n_train, n_valid=n_valid, n_test=n_test,
                     timeout_sec=timeout_sec, n_workers=n_workers)
    load_depot(n_train=n_train, n_valid=n_valid, n_test=n_test,
               timeout_sec=timeout_sec, n_workers=n_workers)
    load_satellite(n_train=n_train, n_valid=n_valid, n_test=n_test,
                   timeout_sec=timeout_sec, n_workers=n_workers)


if __name__ == "__main__":
    fire.Fire({
        "load": load,
        "load_blocksworld": load_blocksworld,
        "load_depot": load_depot,
        "load_satellite": load_satellite,
    })
