import json

import pytest

MACHINE_PATH = "data/verilog_eval/VerilogEval_Machine.jsonl"
HUMAN_PATH = "data/verilog_eval/VerilogEval_Human.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


@pytest.fixture
def machine_entries():
    return load_jsonl(MACHINE_PATH)


@pytest.fixture
def human_entries():
    return load_jsonl(HUMAN_PATH)


class TestMachineDescriptions:
    def test_has_entries(self, machine_entries):
        assert len(machine_entries) > 0

    def test_each_entry_has_description(self, machine_entries):
        for entry in machine_entries:
            assert "description" in entry, (
                f"Missing 'description' for task_id={entry['task_id']}"
            )
            assert entry["description"].strip(), (
                f"Empty 'description' for task_id={entry['task_id']}"
            )


class TestHumanDescriptions:
    def test_has_entries(self, human_entries):
        assert len(human_entries) > 0

    def test_each_entry_has_description(self, human_entries):
        for entry in human_entries:
            assert "description" in entry, (
                f"Missing 'description' for task_id={entry['task_id']}"
            )
            assert entry["description"].strip(), (
                f"Empty 'description' for task_id={entry['task_id']}"
            )
