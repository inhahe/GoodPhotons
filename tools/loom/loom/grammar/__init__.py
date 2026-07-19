"""loom.grammar — shared ``.ftsl`` grammar + parser (GPDA / EPEG).

This subpackage vendors the GPDA graph parser (``_gpda.py``, pinned) and will
host the single shared EPEG grammar for ``.ftsl`` used both by loom's Python
``.ftsl -> Element`` reader and (later) by ftrace's C++ front-end, so one
grammar is the single source of truth (TODO J3c, option-(a) sequencing).

Public surface today is just the parser entry points, re-exported so callers
write ``from loom.grammar import load_grammar`` without reaching into the
vendored module name.
"""

from ._gpda import load_grammar, parse, ParseNode, GrammarParser

__all__ = ["load_grammar", "parse", "ParseNode", "GrammarParser"]
