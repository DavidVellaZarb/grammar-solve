from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any

from smoke_test.common import clean_code_block

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:  # pragma: no cover - dependency is declared, keep CLI robust.
    SmoothingFunction = None
    sentence_bleu = None

try:
    import selfies as sf
except Exception:  # pragma: no cover - handled by SELFIES metrics.
    sf = None

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import TanimotoSimilarity
except Exception:  # pragma: no cover - dependency is declared, keep CLI robust.
    Chem = None
    TanimotoSimilarity = None
    rdFingerprintGenerator = None


MetricValues = dict[str, float]


DOMAIN_METRICS: dict[str, list[str]] = {
    "text_to_sql": ["exact_match", "component_f1"],
    "sparql": ["exact_match", "component_f1"],
    "graphql": ["exact_match", "selection_f1"],
    "vega_lite": ["exact_match", "spec_validity", "mark_f1", "encoding_f1"],
    "vhdl": ["exact_match", "token_bleu"],
    "restricted_graphics": ["exact_match", "xml_validity", "dom_f1"],
    "selfies": [
        "exact_match",
        "molecule_validity",
        "canonical_molecule_exact_match",
        "fingerprint_similarity",
    ],
}

METRIC_LABELS: dict[str, str] = {
    "exact_match": "Exact",
    "component_f1": "Component F1",
    "selection_f1": "Selection F1",
    "spec_validity": "Spec Valid",
    "mark_f1": "Mark F1",
    "encoding_f1": "Encoding F1",
    "token_bleu": "BLEU",
    "xml_validity": "XML Valid",
    "dom_f1": "DOM F1",
    "molecule_validity": "Mol Valid",
    "canonical_molecule_exact_match": "Mol Exact",
    "fingerprint_similarity": "FP Tanimoto",
}

_SQL_CLAUSES = [
    ("select", re.compile(r"\bselect\b", re.IGNORECASE)),
    ("from", re.compile(r"\bfrom\b", re.IGNORECASE)),
    ("where", re.compile(r"\bwhere\b", re.IGNORECASE)),
    ("group_by", re.compile(r"\bgroup\s+by\b", re.IGNORECASE)),
    ("having", re.compile(r"\bhaving\b", re.IGNORECASE)),
    ("order_by", re.compile(r"\border\s+by\b", re.IGNORECASE)),
    ("limit", re.compile(r"\blimit\b", re.IGNORECASE)),
]

_TOKEN_RE = re.compile(
    r"""
    <[^<>"{}|^`\\]*>
    |\$?[A-Za-z_][A-Za-z0-9_.:$-]*
    |[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?
    |"([^"\\]|\\.)*"
    |'([^'\\]|\\.)*'
    |[{}()[\],.;:+\-*/=<>!?|^]
    """,
    re.VERBOSE,
)

_FEATURE_TOKEN_RE = re.compile(
    r"<[^<>\s]+>|\$?[A-Za-z_][A-Za-z0-9_.:$-]*|[+-]?\d+(?:\.\d+)?|[=<>!]+"
)


def compute_domain_metrics(domain: str, results: list[dict[str, Any]]) -> MetricValues:
    if domain not in DOMAIN_METRICS:
        raise ValueError(f"Unsupported smoke-test domain: {domain}")
    if not results:
        return {metric: 0.0 for metric in DOMAIN_METRICS[domain]}

    if domain == "text_to_sql":
        return _compute_text_to_sql(results)
    if domain == "sparql":
        return _compute_sparql(results)
    if domain == "graphql":
        return _compute_graphql(results)
    if domain == "vega_lite":
        return _compute_vega_lite(results)
    if domain == "vhdl":
        return _compute_vhdl(results)
    if domain == "restricted_graphics":
        return _compute_restricted_graphics(results)
    if domain == "selfies":
        return _compute_selfies(results)
    raise AssertionError(domain)


def metric_order(domain: str) -> list[str]:
    return DOMAIN_METRICS[domain]


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _compute_text_to_sql(results: list[dict[str, Any]]) -> MetricValues:
    return {
        "exact_match": _mean(_exact_matches(results)),
        "component_f1": _mean(
            _multiset_f1(
                _sql_component_features(_gold_text(r)),
                _sql_component_features(_prediction_text(r)),
            )
            for r in results
        ),
    }


def _compute_sparql(results: list[dict[str, Any]]) -> MetricValues:
    return {
        "exact_match": _mean(_exact_matches(results)),
        "component_f1": _mean(
            _multiset_f1(_sparql_features(_gold_text(r)), _sparql_features(_prediction_text(r)))
            for r in results
        ),
    }


def _compute_graphql(results: list[dict[str, Any]]) -> MetricValues:
    return {
        "exact_match": _mean(_exact_matches(results)),
        "selection_f1": _mean(
            _multiset_f1(_graphql_features(_gold_text(r)), _graphql_features(_prediction_text(r)))
            for r in results
        ),
    }


def _compute_vega_lite(results: list[dict[str, Any]]) -> MetricValues:
    gold_specs = [_parse_json(_gold_text(r)) for r in results]
    pred_specs = [_parse_json(_prediction_text(r)) for r in results]
    return {
        "exact_match": _mean(_exact_matches(results)),
        "spec_validity": _mean(_is_vega_lite_spec(s) for s in pred_specs),
        "mark_f1": _mean(
            _multiset_f1(_vega_marks(g), _vega_marks(p))
            for g, p in zip(gold_specs, pred_specs, strict=True)
        ),
        "encoding_f1": _mean(
            _multiset_f1(_vega_encoding_features(g), _vega_encoding_features(p))
            for g, p in zip(gold_specs, pred_specs, strict=True)
        ),
    }


def _compute_vhdl(results: list[dict[str, Any]]) -> MetricValues:
    return {
        "exact_match": _mean(_exact_matches(results)),
        "token_bleu": _mean(
            _bleu(_program_tokens(_gold_text(r)), _program_tokens(_prediction_text(r)))
            for r in results
        ),
    }


def _compute_restricted_graphics(results: list[dict[str, Any]]) -> MetricValues:
    gold_roots = [_parse_xml(_gold_text(r)) for r in results]
    pred_roots = [_parse_xml(_prediction_text(r)) for r in results]
    return {
        "exact_match": _mean(_exact_matches(results)),
        "xml_validity": _mean(root is not None for root in pred_roots),
        "dom_f1": _mean(
            _multiset_f1(_svg_dom_features(g), _svg_dom_features(p))
            for g, p in zip(gold_roots, pred_roots, strict=True)
        ),
    }


def _compute_selfies(results: list[dict[str, Any]]) -> MetricValues:
    gold_mols = [_selfies_to_mol(_gold_text(r)) for r in results]
    pred_mols = [_selfies_to_mol(_prediction_text(r)) for r in results]
    return {
        "exact_match": _mean(_exact_matches(results)),
        "molecule_validity": _mean(mol is not None for mol in pred_mols),
        "canonical_molecule_exact_match": _mean(
            g is not None and p is not None and _canonical_smiles(g) == _canonical_smiles(p)
            for g, p in zip(gold_mols, pred_mols, strict=True)
        ),
        "fingerprint_similarity": _mean(
            _fingerprint_similarity(g, p) or 0.0
            for g, p in zip(gold_mols, pred_mols, strict=True)
        ),
    }


def _exact_matches(results: list[dict[str, Any]]) -> list[bool]:
    return [
        bool(
            r.get("match")
            if "match" in r
            else _gold_text(r).strip() == _prediction_text(r).strip()
        )
        for r in results
    ]


def _gold_text(result: dict[str, Any]) -> str:
    return str(result.get("gold_normalized") or result.get("gold") or "")


def _prediction_text(result: dict[str, Any]) -> str:
    return str(result.get("prediction_normalized") or result.get("prediction") or "")


def _mean(values: Any) -> float:
    materialized = [float(v) for v in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def _program_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(clean_code_block(text))]


def _feature_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _FEATURE_TOKEN_RE.finditer(clean_code_block(text))]


def _multiset_f1(gold: list[str], pred: list[str]) -> float:
    gold_counts = Counter(gold)
    pred_counts = Counter(pred)
    if not gold_counts and not pred_counts:
        return 1.0
    if not gold_counts or not pred_counts:
        return 0.0
    overlap = sum((gold_counts & pred_counts).values())
    precision = overlap / sum(pred_counts.values()) if pred_counts else 0.0
    recall = overlap / sum(gold_counts.values()) if gold_counts else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _bleu(gold_tokens: list[str], pred_tokens: list[str]) -> float:
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    if sentence_bleu is None or SmoothingFunction is None:
        return _multiset_f1(gold_tokens, pred_tokens)
    try:
        return float(
            sentence_bleu(
                [gold_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except Exception:
        return 0.0


def _sql_component_features(sql: str) -> list[str]:
    text = clean_code_block(sql)
    lower = text.lower()
    starts: list[tuple[int, str, int]] = []
    for name, pattern in _SQL_CLAUSES:
        for match in pattern.finditer(lower):
            starts.append((match.start(), name, match.end()))
    starts.sort()
    if not starts:
        return [f"token:{t}" for t in _feature_tokens(text)]

    features: list[str] = []
    for i, (start, name, end) in enumerate(starts):
        next_start = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        span = text[end:next_start]
        features.append(f"clause:{name}")
        for token in _feature_tokens(span):
            if token not in {"as", ",", "(", ")"}:
                features.append(f"{name}:{token}")
    return features


def _sparql_features(text: str) -> list[str]:
    tokens = _feature_tokens(text)
    features: list[str] = []
    for token in tokens:
        if token in {"select", "ask", "construct", "describe"}:
            features.append(f"form:{token}")
        elif token.startswith("?") or token.startswith("$"):
            features.append(f"var:{token}")
        elif token.startswith("<") or ":" in token:
            features.append(f"term:{token}")
        elif token in {"where", "optional", "filter", "union", "bind", "values"}:
            features.append(f"op:{token}")
        else:
            features.append(f"tok:{token}")
    return features


def _graphql_features(text: str) -> list[str]:
    text = clean_code_block(text)
    features: list[str] = []
    for name in re.findall(r"\b[_A-Za-z][_0-9A-Za-z]*\b", text):
        if name in {"query", "mutation", "subscription", "fragment", "on", "true", "false", "null"}:
            features.append(f"kw:{name}")
        else:
            features.append(f"name:{name}")
    for arg in re.findall(r"\b([_A-Za-z][_0-9A-Za-z]*)\s*:", text):
        features.append(f"arg:{arg}")
    for var in re.findall(r"\$[_A-Za-z][_0-9A-Za-z]*", text):
        features.append(f"var:{var}")
    return features


def _parse_json(text: str) -> Any | None:
    try:
        return json.loads(clean_code_block(text))
    except Exception:
        return None


def _is_vega_lite_spec(spec: Any | None) -> bool:
    if not isinstance(spec, dict):
        return False
    return bool(
        {"mark", "encoding", "layer", "concat", "vconcat", "hconcat", "facet", "spec", "repeat"}
        & set(spec)
    )


def _vega_marks(value: Any | None) -> list[str]:
    marks: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            mark = node.get("mark")
            if isinstance(mark, str):
                marks.append(mark.lower())
            elif isinstance(mark, dict) and isinstance(mark.get("type"), str):
                marks.append(mark["type"].lower())
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return marks


def _vega_encoding_features(value: Any | None) -> list[str]:
    features: list[str] = []

    def add_encoding(encoding: dict[str, Any]) -> None:
        for channel, spec in encoding.items():
            if not isinstance(spec, dict):
                continue
            channel_l = str(channel).lower()
            features.append(f"channel:{channel_l}")
            for key in ("field", "type", "aggregate", "bin", "timeUnit", "sort"):
                if key in spec:
                    features.append(f"{channel_l}:{key.lower()}:{_json_atom(spec[key])}")

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            encoding = node.get("encoding")
            if isinstance(encoding, dict):
                add_encoding(encoding)
            for transform in node.get("transform", []) if isinstance(node.get("transform"), list) else []:
                if isinstance(transform, dict):
                    for key in transform:
                        features.append(f"transform:{key.lower()}")
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return features


def _json_atom(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value).lower()
    return json.dumps(value, sort_keys=True, separators=(",", ":")).lower()


def _parse_xml(text: str) -> ET.Element | None:
    try:
        return ET.fromstring(clean_code_block(text))
    except Exception:
        return None


def _svg_dom_features(root: ET.Element | None) -> list[str]:
    if root is None:
        return []
    features: list[str] = []
    for elem in root.iter():
        tag = _strip_namespace(elem.tag).lower()
        features.append(f"tag:{tag}")
        text = (elem.text or "").strip()
        if text:
            normalized_text = re.sub(r"\s+", " ", text).lower()
            features.append(f"text:{normalized_text}")
        for name, value in sorted(elem.attrib.items()):
            clean_name = _strip_namespace(name).lower()
            features.append(f"attr:{tag}:{clean_name}")
            if clean_name in {
                "fill",
                "stroke",
                "width",
                "height",
                "x",
                "y",
                "cx",
                "cy",
                "r",
                "points",
                "d",
                "text-anchor",
                "font-size",
            }:
                features.append(f"attr:{tag}:{clean_name}={value.strip().lower()}")
    return features


def _strip_namespace(name: str) -> str:
    if "}" in name:
        return name.rsplit("}", 1)[1]
    return name


def _selfies_to_mol(text: str) -> Any | None:
    if sf is None or Chem is None:
        return None
    try:
        smiles = sf.decoder(re.sub(r"\s+", "", clean_code_block(text)))
        if not smiles:
            return None
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _canonical_smiles(mol: Any) -> str | None:
    if Chem is None or mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _fingerprint_similarity(gold_mol: Any | None, pred_mol: Any | None) -> float | None:
    if (
        gold_mol is None
        or pred_mol is None
        or rdFingerprintGenerator is None
        or TanimotoSimilarity is None
    ):
        return None
    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return float(TanimotoSimilarity(gen.GetFingerprint(gold_mol), gen.GetFingerprint(pred_mol)))
    except Exception:
        return None


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))
