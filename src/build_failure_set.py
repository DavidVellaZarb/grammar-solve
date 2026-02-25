import json

import fire

from data import load_raw_data
from predict_utils import write_output


def build_failure_set(
    results_path: str = "results/baseline/baseline.json",
    test_path: str = "data/smcalflow/test.json",
    generic_grammar: bool = False,
    output_path: str | None = None,
) -> None:
    with open(results_path) as f:
        results = json.load(f)["results"]

    test_data = load_raw_data(test_path)
    assert len(results) == len(test_data), (
        f"Length mismatch: {len(results)} results vs {len(test_data)} test entries"
    )

    if generic_grammar:
        generic_path = test_path.replace(".json", "_generic.json")
        generic_data = load_raw_data(generic_path)
        assert len(generic_data) == len(test_data)
    else:
        generic_data = test_data

    failures = []
    for result, test_entry, grammar_entry in zip(results, test_data, generic_data):
        if not result["match"]:
            failures.append(
                {
                    "query": test_entry["query"],
                    "minimal_grammar": grammar_entry["minimal_grammar"],
                    "program": test_entry["program"],
                }
            )

    if output_path is None:
        output_path = "data/smcalflow/test_failures.json"

    write_output(failures, output_path)
    print(f"Failures: {len(failures)}/{len(results)}")


if __name__ == "__main__":
    fire.Fire(build_failure_set)
