import json

import pytest
from lark import Lark

PRUNED_GRAMMAR_PATH = "grammars/smcalflow_pruned.lark"
TRAIN_PATH = "data/smcalflow/train.json"
TEST_PATH = "data/smcalflow/test.json"


@pytest.fixture(scope="module")
def parser():
    with open(PRUNED_GRAMMAR_PATH) as f:
        text = f.read()
    return Lark(text, start="call", parser="earley", keep_all_tokens=True)


@pytest.fixture(scope="module")
def train_data():
    with open(TRAIN_PATH) as f:
        return json.load(f)["data"]


@pytest.fixture(scope="module")
def test_data():
    with open(TEST_PATH) as f:
        return json.load(f)["data"]


def test_pruned_grammar_parses_train(parser, train_data):
    failures = []
    for i, entry in enumerate(train_data):
        try:
            parser.parse(entry["program"])
        except Exception as e:
            failures.append(f"[{i}] {entry['program'][:100]}... -> {e}")
    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])


def test_pruned_grammar_parses_test(parser, test_data):
    failures = []
    for i, entry in enumerate(test_data):
        try:
            parser.parse(entry["program"])
        except Exception as e:
            failures.append(f"[{i}] {entry['program'][:100]}... -> {e}")
    assert not failures, f"{len(failures)} failures:\n" + "\n".join(failures[:10])
