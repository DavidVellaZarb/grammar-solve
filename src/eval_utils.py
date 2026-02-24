import json
import os


def check_match(gold: str, prediction: str) -> bool:
    return gold in prediction


def compute_metrics(results: list[dict]) -> dict:
    correct = sum(1 for r in results if r["match"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def save_results(metrics: dict, results: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output = {**metrics, "results": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")
