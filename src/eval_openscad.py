import json
import os
import re
import shutil
import subprocess
import tempfile

import fire
import numpy as np
import torch
import trimesh
from peft import PeftConfig, PeftModel
from scipy.spatial import KDTree
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt_messages, load_raw_data
from eval_utils import save_results


def extract_openscad_code(prediction: str) -> str:
    fence = re.search(r"```(?:openscad|scad)?\s*\n(.*?)```", prediction, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return prediction.strip()


def compile_openscad(code: str, timeout: float = 30.0) -> trimesh.Trimesh | None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".scad", delete=False) as f_in:
        f_in.write(code)
        scad_path = f_in.name

    stl_path = scad_path.replace(".scad", ".stl")
    try:
        result = subprocess.run(
            ["openscad", "--export-format", "stl", "-o", stl_path, scad_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0 or not os.path.exists(stl_path):
            return None
        if os.path.getsize(stl_path) == 0:
            return None
        mesh = trimesh.load(stl_path, file_type="stl")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            return None
        return mesh
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None
    finally:
        for p in [scad_path, stl_path]:
            if os.path.exists(p):
                os.unlink(p)


def check_syntax_validity(code: str, timeout: float = 30.0) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".scad", delete=False) as f_in:
        f_in.write(code)
        scad_path = f_in.name

    stl_path = scad_path.replace(".scad", ".stl")
    try:
        result = subprocess.run(
            ["openscad", "--export-format", "stl", "-o", stl_path, scad_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        for p in [scad_path, stl_path]:
            if os.path.exists(p):
                os.unlink(p)


def compute_chamfer_distance(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh,
                             n_points: int = 10000) -> float:
    pts1 = mesh1.sample(n_points)
    pts2 = mesh2.sample(n_points)

    tree1 = KDTree(pts1)
    tree2 = KDTree(pts2)

    d1, _ = tree2.query(pts1)
    d2, _ = tree1.query(pts2)

    return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))


def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    extents = mesh.extents
    max_extent = max(extents)
    if max_extent > 0:
        mesh.apply_scale(1.0 / max_extent)
    return mesh


def compute_iou(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh,
                pitch: float = 0.02) -> float:
    m1 = _normalize_mesh(mesh1)
    m2 = _normalize_mesh(mesh2)

    try:
        v1 = m1.voxelized(pitch)
        v2 = m2.voxelized(pitch)

        s1 = set(map(tuple, v1.sparse_indices))
        s2 = set(map(tuple, v2.sparse_indices))

        intersection = len(s1 & s2)
        union = len(s1 | s2)

        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def evaluate(
    adapter: str,
    test_path: str = "data/openscad/test.json",
    model_name: str | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 2048,
    output_path: str | None = None,
    attn_implementation: str = "flash_attention_2",
    grammar_file: str | None = None,
    include_grammar: bool = True,
    task: str = "program",
    compile_timeout: float = 30.0,
    n_sample_points: int = 10000,
    voxel_pitch: float = 0.02,
    cache_dir: str | None = None,
):
    assert shutil.which("openscad") is not None, (
        "openscad CLI not found on PATH. Install OpenSCAD and ensure 'openscad' is available."
    )

    peft_config = PeftConfig.from_pretrained(adapter)
    base_model_name = model_name or peft_config.base_model_name_or_path
    assert base_model_name is not None, "No model_name provided and adapter config has no base_model_name_or_path"

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
        assert len(grammar_data) == len(examples), (
            f"Grammar file has {len(grammar_data)} entries but test data has {len(examples)}"
        )
        for ex, gex in zip(examples, grammar_data):
            assert ex["query"] == gex["query"], (
                f"Query mismatch: {ex['query']!r} vs {gex['query']!r}"
            )
            ex["minimal_grammar"] = gex["minimal_grammar"]
    else:
        print("Using gold grammars from test data")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    prompts = []
    for ex in examples:
        messages = format_prompt_messages(ex, include_grammar=include_grammar, task=task)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating"):
        batch_prompts = prompts[i : i + batch_size]
        batch_examples = examples[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for ex, prompt, pred in zip(batch_examples, batch_prompts, predictions):
            gold = ex["program"]
            pred_code = extract_openscad_code(pred)

            valid = check_syntax_validity(pred_code, timeout=compile_timeout)

            cd = None
            iou = None

            gold_mesh = compile_openscad(gold, timeout=compile_timeout)
            pred_mesh = compile_openscad(pred_code, timeout=compile_timeout) if valid else None

            if gold_mesh is not None and pred_mesh is not None:
                cd = compute_chamfer_distance(gold_mesh, pred_mesh, n_points=n_sample_points)
                iou = compute_iou(gold_mesh, pred_mesh, pitch=voxel_pitch)

            results.append({
                "prompt": prompt,
                "gold": gold,
                "prediction": pred,
                "pred_code": pred_code,
                "valid": valid,
                "chamfer_distance": cd,
                "iou": iou,
            })

    total = len(results)
    valid_count = sum(1 for r in results if r["valid"])
    cds = [r["chamfer_distance"] for r in results if r["chamfer_distance"] is not None]
    ious = [r["iou"] for r in results if r["iou"] is not None]

    metrics = {
        "syntax_validity": valid_count / total if total > 0 else 0.0,
        "chamfer_distance": sum(cds) / len(cds) if cds else None,
        "iou": sum(ious) / len(ious) if ious else None,
        "geometry_evaluated": len(cds),
        "total": total,
    }

    print(f"\n--- Metrics ---")
    print(f"Syntax validity:    {metrics['syntax_validity']:.4f} ({valid_count}/{total})")
    if metrics["chamfer_distance"] is not None:
        print(f"Chamfer Distance:   {metrics['chamfer_distance']:.6f} (n={len(cds)})")
    else:
        print(f"Chamfer Distance:   N/A (no valid geometry pairs)")
    if metrics["iou"] is not None:
        print(f"Volumetric IoU:     {metrics['iou']:.4f} (n={len(ious)})")
    else:
        print(f"Volumetric IoU:     N/A (no valid geometry pairs)")

    if output_path:
        save_results(metrics, results, output_path)


if __name__ == "__main__":
    fire.Fire(evaluate)
