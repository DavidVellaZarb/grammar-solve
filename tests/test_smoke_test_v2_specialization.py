from pathlib import Path

from lark import Lark

from smoke_test.common import COMMON_GENERIC_TERMINALS, minimal_grammar_for_program
from smoke_test.validate_specialized import find_generic_placeholders


def test_specialized_smoke_grammar_replaces_generic_terminals():
    grammar_path = Path("smoke_test/text_to_sql/text_to_sql.lark")
    parser = Lark(
        grammar_path.read_text(),
        start="query",
        parser="earley",
        keep_all_tokens=True,
    )
    program = "SELECT name FROM users WHERE age = 3"

    generic = minimal_grammar_for_program(
        program,
        parser,
        grammar_path,
        generic_terminals=COMMON_GENERIC_TERMINALS,
    )
    specialized = minimal_grammar_for_program(
        program,
        parser,
        grammar_path,
        generic_terminals=frozenset(),
    )

    assert "IDENTIFIER" in generic
    assert "NUMBER" in generic
    assert "IDENTIFIER" not in specialized
    assert "NUMBER" not in specialized
    assert '"name"' in specialized
    assert '"users"' in specialized
    assert '"age"' in specialized
    assert '"3"' in specialized


def test_specialization_validator_ignores_quoted_terminal_names():
    terminals = frozenset({"IDENTIFIER"})

    assert find_generic_placeholders("identifier ::= IDENTIFIER", terminals) == {
        "IDENTIFIER"
    }
    assert find_generic_placeholders('identifier ::= "IDENTIFIER"', terminals) == set()
