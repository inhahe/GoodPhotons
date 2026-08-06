"""FTCL -- the creature-language front-end of the FTSL family.

Generic machinery only: lexing, a schema-driven block parser, a symbolic expression
sublanguage and a units system. It knows nothing about creatures; the vocabulary lives in
`creaturelab.schema`.
"""
from .errors import FtclError, Loc
from .expr import Env, Expr, Num, Ref
from .lexer import Token, tokenize
from .parser import BlockSpec, Node, Parser, Prop, VARIADIC, parse_file, parse_string

__all__ = ["FtclError", "Loc", "Env", "Expr", "Num", "Ref", "Token", "tokenize",
           "BlockSpec", "Node", "Parser", "Prop", "VARIADIC", "parse_file", "parse_string"]
