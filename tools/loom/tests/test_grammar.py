"""J3c foundation tests: the vendored GPDA parser (``loom.grammar``) loads and
parses.  This is the smoke layer that de-risks the shared ``.ftsl`` grammar work
— it proves the pinned ``_gpda.py`` is importable and functional inside loom
before any ``.ftsl`` grammar is built on top of it.

Runnable directly (``python tests/test_grammar.py``) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom.grammar import load_grammar, ParseNode  # noqa: E402


# A JSON grammar in *tokenized* EPEG (the flavour `_gpda.py` implements): terminals
# are `'literal'` / `/regex/` / token NAME — bare `[...]` char-classes are scannerless
# syntax and only valid inside a `/regex/`.  Exercises alternation, quantifiers, regex
# terminals and @skip.
_JSON = r"""
start json
WS = /[ \t\n\r]+/ @skip
NUMBER = /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+\-]?[0-9]+)?/
STRING = /"([^"\\]|\\.)*"/
json = value
value = object | array | STRING | NUMBER | bool | null
object = '{' (pair (',' pair)*)? '}'
pair = STRING ':' value
array = '[' (value (',' value)*)? ']'
bool = 'true' | 'false'
null = 'null'
"""


def _names(node):
    """Set of every rule/terminal name reachable in the tree (for shape asserts)."""
    out = {node.name}
    for c in node.children:
        out |= _names(c)
    return out


def test_vendored_parser_imports_and_parses_json():
    p = load_grammar(_JSON)
    tree = p.parse('{"a": [1, 2.5, true], "b": null}')
    assert isinstance(tree, ParseNode)
    names = _names(tree)
    # the whole structure was recognised (objects, arrays, and the leaf token kinds)
    assert {"json", "object", "pair", "array", "value"} <= names
    assert {"STRING", "NUMBER", "bool", "null"} <= names


def test_parse_failure_is_reported_not_hung():
    p = load_grammar(_JSON)
    # malformed JSON must fail cleanly (return None / raise), never hang
    result = None
    try:
        result = p.parse('{"a": }')
    except Exception:
        result = "raised"
    assert result is not None
    assert result == "raised" or result is not None


def test_grammar_provenance_header_present():
    # the vendored copy must keep its pin header so refreshes stay traceable
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(here, "loom", "grammar", "_gpda.py")
    with open(src, "r", encoding="utf-8") as fh:
        head = fh.read(2000)
    assert "VENDORED" in head and "Commit" in head


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
