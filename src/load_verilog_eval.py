import os
import urllib.request

import fire

BASE_URL = (
    "https://raw.githubusercontent.com/NVlabs/verilog-eval/release/1.0.0/data"
)

FILES = [
    "VerilogEval_Machine.jsonl",
    "VerilogEval_Human.jsonl",
]


def download(output_dir: str = "data/verilog_eval") -> None:
    os.makedirs(output_dir, exist_ok=True)

    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        dest = os.path.join(output_dir, filename)

        if os.path.exists(dest):
            print(f"Already exists: {dest}")
            continue

        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  Done: {dest}")

    print("\nAll files downloaded.")


if __name__ == "__main__":
    fire.Fire({"download": download})
