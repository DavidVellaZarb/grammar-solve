import json
import os


def write_output(results: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"data": results}, f, indent=2)
    print(f"Wrote {len(results)} predictions to {output_path}")
