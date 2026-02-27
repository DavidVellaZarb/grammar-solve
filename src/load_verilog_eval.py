import json
import os
import urllib.request

import fire

MG_VERILOG_BASE_URL = (
    "https://raw.githubusercontent.com/GATECH-EIC/mg-verilog/main/verilog_eval"
)

PROBLEM_FILES = [
    "VerilogEval_Machine.jsonl",
    "VerilogEval_Human.jsonl",
]

DESCRIPTION_FILES = {
    "VerilogEval_Machine.jsonl": "VerilogDescription_Machine.jsonl",
    "VerilogEval_Human.jsonl": "VerilogDescription_Human.jsonl",
}


def _download_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done: {dest}")


def _load_descriptions(path: str) -> dict[str, dict[str, str]]:
    descriptions = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            desc = {"description": entry["detail_description"]}
            if "simple_description" in entry:
                desc["simple_description"] = entry["simple_description"]
            descriptions[entry["task_id"]] = desc
    return descriptions


def _merge_descriptions(
    problem_path: str, descriptions: dict[str, dict[str, str]]
) -> None:
    entries = []
    with open(problem_path) as f:
        for line in f:
            entries.append(json.loads(line))

    problem_ids = {e["task_id"] for e in entries}
    desc_ids = set(descriptions.keys())
    if problem_ids != desc_ids:
        only_problems = problem_ids - desc_ids
        only_descs = desc_ids - problem_ids
        raise ValueError(
            f"Task ID mismatch in {problem_path}:\n"
            f"  In problems but not descriptions: {sorted(only_problems)}\n"
            f"  In descriptions but not problems: {sorted(only_descs)}"
        )

    for entry in entries:
        entry.update(descriptions[entry["task_id"]])

    with open(problem_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def download(output_dir: str = "data/verilog_eval") -> None:
    os.makedirs(output_dir, exist_ok=True)

    for filename in PROBLEM_FILES:
        _download_file(
            f"{MG_VERILOG_BASE_URL}/data/{filename}",
            os.path.join(output_dir, filename),
        )

    for problem_file, desc_file in DESCRIPTION_FILES.items():
        problem_path = os.path.join(output_dir, problem_file)
        desc_path = os.path.join(output_dir, desc_file)

        with open(problem_path) as f:
            first = json.loads(f.readline())
        if "description" in first:
            print(f"Descriptions already merged in {problem_path}")
            continue

        _download_file(
            f"{MG_VERILOG_BASE_URL}/descriptions/{desc_file}",
            desc_path,
        )

        descriptions = _load_descriptions(desc_path)
        print(f"Merging {len(descriptions)} descriptions into {problem_path}...")
        _merge_descriptions(problem_path, descriptions)
        print(f"  Merged. Removing {desc_path}")
        os.remove(desc_path)

    print("\nAll files downloaded.")


if __name__ == "__main__":
    fire.Fire({"download": download})
