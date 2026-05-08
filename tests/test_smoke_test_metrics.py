from __future__ import annotations

import pytest

from smoke_test.metrics import compute_domain_metrics


def _result(gold: str, pred: str, match: bool = False) -> dict[str, object]:
    return {
        "gold": gold,
        "prediction": pred,
        "gold_normalized": gold,
        "prediction_normalized": pred,
        "match": match,
    }


@pytest.mark.parametrize(
    ("domain", "gold", "pred", "required_metric", "similarity_metric"),
    [
        (
            "text_to_sql",
            "SELECT name FROM users WHERE age > 30",
            "SELECT name FROM users WHERE age > 30",
            "component_f1",
            "component_f1",
        ),
        (
            "sparql",
            "SELECT ?x WHERE { ?x <p> <o> . }",
            "SELECT ?x WHERE { ?x <p> <o> . }",
            "component_f1",
            "component_f1",
        ),
        (
            "graphql",
            "{ user(id: 1) { name email } }",
            "{ user(id: 1) { name } }",
            "selection_f1",
            "selection_f1",
        ),
        (
            "vega_lite",
            '{"mark":"bar","encoding":{"x":{"field":"a","type":"quantitative"}}}',
            '{"mark":"bar","encoding":{"x":{"field":"a","type":"quantitative"}}}',
            "spec_validity",
            "encoding_f1",
        ),
        (
            "vhdl",
            "entity foo is end foo;",
            "entity foo is end foo;",
            "token_bleu",
            "token_bleu",
        ),
        (
            "restricted_graphics",
            '<svg width="10" height="10"><text x="1" y="2">Hi</text></svg>',
            '<svg width="10" height="10"><text x="1" y="2">Hi</text></svg>',
            "xml_validity",
            "dom_f1",
        ),
    ],
)
def test_compute_domain_metrics(domain, gold, pred, required_metric, similarity_metric):
    metrics = compute_domain_metrics(domain, [_result(gold, pred, match=gold == pred)])

    assert "syntax_validity" not in metrics
    assert required_metric in metrics
    assert 0.0 <= metrics[similarity_metric] <= 1.0


def test_selfies_metrics_decode_to_molecules():
    sf = pytest.importorskip("selfies")
    gold = sf.encoder("CCO")
    pred = sf.encoder("CCO")

    metrics = compute_domain_metrics("selfies", [_result(gold, pred, match=True)])

    assert metrics["exact_match"] == 1.0
    assert metrics["molecule_validity"] == 1.0
    assert metrics["canonical_molecule_exact_match"] == 1.0
    assert metrics["fingerprint_similarity"] == 1.0


def test_invalid_predictions_score_zero_for_validity():
    metrics = compute_domain_metrics(
        "vega_lite",
        [
            _result(
                '{"mark":"bar","encoding":{"x":{"field":"a","type":"quantitative"}}}',
                '{"mark":',
            )
        ],
    )

    assert metrics["spec_validity"] == 0.0
    assert metrics["mark_f1"] == 0.0
    assert metrics["encoding_f1"] == 0.0
