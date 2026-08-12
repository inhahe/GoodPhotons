"""Schema-driven block/statement parser for the FTSL language family.

Why schema-driven rather than newline-terminated: FTSL's compact style puts several
statements on one line --

    geom capsule { a 0 0 -0.05  b 0 0 0.05  radius 0.06 }

-- and there is no way to know that `a` takes three values and `radius` takes one without
knowing the shape of the thing being parsed. Carrying a schema also buys the two things
that actually matter when a human is editing rig files:

* real diagnostics -- `bone has no property 'lenght'; did you mean 'length'?`, with the
  source line and a caret, instead of a generic syntax error twenty lines later; and
* `required` checking at parse time, so a bone with no geometry is caught here rather than
  as a NaN in the physics an hour into training.

The parser is generic: it knows blocks, properties, arities and label styles, and nothing
about creatures. The creature vocabulary lives in `creaturelab/schema.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import FtclError, Loc, did_you_mean
from .expr import Expr, is_value_start, parse_atom, parse_expr
from .lexer import EOF, IDENT, NEWLINE, NUMBER, PUNCT_T, STRING, Token, tokenize

VARIADIC = "*"


@dataclass
class Prop:
    """A statement inside a block: `keyword <arity> values`."""
    arity: int | str = 1
    kind: str = "num"          # 'num' (expressions) | 'str' (quoted) | 'word' (bare ident)
    required: bool = False
    choices: tuple[str, ...] | None = None    # for kind='word'
    doc: str = ""


@dataclass
class BlockSpec:
    label: str | None = None                  # None | 'name' (quoted) | 'word' (bare)
    label_required: bool = True
    choices: tuple[str, ...] | None = None     # allowed values when label == 'word'
    props: dict[str, Prop] = field(default_factory=dict)
    blocks: dict[str, "BlockSpec"] = field(default_factory=dict)
    required_blocks: tuple[str, ...] = ()
    # When set, unknown keys are accepted as properties of this Prop shape instead of
    # being an error. Used by `derive { shank = femur*0.95 }`, where the whole point is
    # that the author invents the names -- so a fixed vocabulary cannot work. Everywhere
    # else the fixed vocabulary is what gives good typo diagnostics, so this stays opt-in.
    free_props: Prop | None = None
    doc: str = ""


@dataclass
class Node:
    """A parsed block. `props` values are lists of `Expr` (num) or `str` (str/word)."""
    kw: str
    loc: Loc
    name: str | None = None
    word: str | None = None                   # the bare-identifier label, e.g. `capsule`
    props: dict[str, list] = field(default_factory=dict)
    prop_locs: dict[str, Loc] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    free_keys: list[str] = field(default_factory=list)   # author-invented props, in order

    # -- convenience accessors used by the builder ---------------------------------------
    def has(self, key: str) -> bool:
        return key in self.props

    def child(self, kw: str) -> "Node | None":
        for c in self.children:
            if c.kw == kw:
                return c
        return None

    def all(self, kw: str) -> list["Node"]:
        return [c for c in self.children if c.kw == kw]


class Parser:
    def __init__(self, src: str, filename: str, root: BlockSpec):
        self.src = src
        self.toks = tokenize(src, filename)
        self.i = 0
        self.root_spec = root

    # -- token helpers --------------------------------------------------------------------
    def peek(self, k: int = 0) -> Token:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def take(self) -> Token:
        tok = self.toks[self.i]
        if tok.kind != EOF:
            self.i += 1
        return tok

    def skip_newlines(self) -> None:
        while self.peek().kind == NEWLINE:
            self.i += 1

    def err(self, msg: str, tok: Token | None = None) -> FtclError:
        tok = tok or self.peek()
        return FtclError(msg, tok.loc, self.src)

    def expect_punct(self, text: str) -> Token:
        tok = self.take()
        if tok.kind != PUNCT_T or tok.text != text:
            raise self.err(f"expected '{text}', found {tok.text!r}", tok)
        return tok

    # -- entry point ----------------------------------------------------------------------
    def parse(self) -> list[Node]:
        """Parse a whole file as a sequence of top-level blocks."""
        out: list[Node] = []
        self.skip_newlines()
        while self.peek().kind != EOF:
            kw = self.peek()
            if kw.kind != IDENT:
                raise self.err(f"expected a block keyword, found {kw.text!r}")
            if kw.text not in self.root_spec.blocks:
                raise self.err(f"unknown top-level block '{kw.text}'"
                               + did_you_mean(kw.text, self.root_spec.blocks))
            out.append(self.parse_block(kw.text, self.root_spec.blocks[kw.text]))
            self.skip_newlines()
        return out

    # -- blocks ---------------------------------------------------------------------------
    def parse_block(self, kw: str, spec: BlockSpec) -> Node:
        kw_tok = self.take()
        node = Node(kw=kw, loc=kw_tok.loc)

        if spec.label == "name":
            tok = self.peek()
            if tok.kind == STRING:
                node.name = str(self.take().value)
            elif spec.label_required:
                raise self.err(f"'{kw}' needs a quoted name, e.g. {kw} \"something\" {{")
        elif spec.label == "word":
            tok = self.peek()
            if tok.kind == IDENT:
                node.word = self.take().text
                if spec.choices and node.word not in spec.choices:
                    raise self.err(
                        f"'{kw}' kind must be one of {', '.join(spec.choices)}, "
                        f"found '{node.word}'" + did_you_mean(node.word, spec.choices),
                        tok)
            elif spec.label_required:
                raise self.err(
                    f"'{kw}' needs a kind, one of {', '.join(spec.choices or ())}")

        self.expect_punct("{")
        self.parse_body(node, spec)
        self.expect_punct("}")
        self.check_required(node, spec)
        return node

    def parse_body(self, node: Node, spec: BlockSpec) -> None:
        while True:
            self.skip_newlines()
            tok = self.peek()
            if tok.kind == EOF:
                raise self.err(f"unterminated '{node.kw}' block "
                               f"(opened at {node.loc})", tok)
            if tok.kind == PUNCT_T and tok.text == "}":
                return
            if tok.kind != IDENT:
                raise self.err(f"expected a property or block inside '{node.kw}', "
                               f"found {tok.text!r}", tok)

            key = tok.text
            if key in spec.blocks:
                node.children.append(self.parse_block(key, spec.blocks[key]))
                continue
            if key in spec.props:
                self.parse_prop(node, key, spec.props[key], spec)
                continue
            if spec.free_props is not None:
                self.parse_prop(node, key, spec.free_props, spec)
                node.free_keys.append(key)
                continue

            known = sorted(set(spec.props) | set(spec.blocks))
            raise self.err(
                f"'{node.kw}' has no property or block '{key}'"
                + (did_you_mean(key, known) or f" (known: {', '.join(known)})"), tok)

    # -- statements -----------------------------------------------------------------------
    def parse_prop(self, node: Node, key: str, prop: Prop,
                   spec: "BlockSpec | None" = None) -> None:
        key_tok = self.take()
        if key in node.props:
            raise self.err(f"'{key}' set twice in '{node.kw}'", key_tok)
        node.prop_locs[key] = key_tok.loc

        # `key = <full infix expression to end of line>`
        if self.peek().kind == PUNCT_T and self.peek().text == "=":
            if prop.kind != "num" or prop.arity != 1:
                raise self.err(f"'{key}' cannot use '=' (it takes "
                               f"{prop.arity} {prop.kind} value(s))")
            self.take()
            e, self.i = parse_expr(self.toks, self.i, self.src)
            node.props[key] = [e]
            self.end_of_statement(key)
            return

        if prop.kind == "str":
            vals = []
            count = 0
            while True:
                tok = self.peek()
                if tok.kind != STRING:
                    break
                vals.append(str(self.take().value))
                count += 1
                if prop.arity != VARIADIC and count >= prop.arity:
                    break
            self.require_count(key, prop, len(vals), key_tok)
            node.props[key] = vals
            self.end_of_statement(key)
            return

        if prop.kind == "word":
            vals = []
            count = 0
            while True:
                tok = self.peek()
                if tok.kind != IDENT:
                    break
                if prop.choices and tok.text not in prop.choices:
                    raise self.err(
                        f"'{key}' must be one of {', '.join(prop.choices)}, "
                        f"found '{tok.text}'" + did_you_mean(tok.text, prop.choices), tok)
                vals.append(self.take().text)
                count += 1
                if prop.arity != VARIADIC and count >= prop.arity:
                    break
            self.require_count(key, prop, len(vals), key_tok)
            node.props[key] = vals
            self.end_of_statement(key)
            return

        # numeric: a fixed number of atoms, or variadic to end of line
        vals = []
        siblings = (set(spec.props) | set(spec.blocks)) if spec else set()
        while True:
            if prop.arity != VARIADIC and len(vals) >= prop.arity:
                break
            tok = self.peek()
            if not is_value_start(tok):
                break
            # Do not swallow the NEXT statement's keyword as a value. Only applied once at
            # least one value is in hand, so a rig may still legitimately name a morph
            # param after a property (`radius mass`) as the first argument.
            if vals and tok.kind == IDENT and tok.text in siblings:
                break
            e, self.i = parse_atom(self.toks, self.i, self.src)
            vals.append(e)
        self.require_count(key, prop, len(vals), key_tok)
        node.props[key] = vals
        self.end_of_statement(key)

    def require_count(self, key: str, prop: Prop, got: int, tok: Token) -> None:
        if prop.arity == VARIADIC:
            if got == 0:
                raise self.err(f"'{key}' needs at least one value", tok)
            return
        if got != prop.arity:
            raise self.err(f"'{key}' takes {prop.arity} value(s), got {got}", tok)

    def end_of_statement(self, key: str) -> None:
        """A statement ends at a newline, a '}', or the start of the next keyword.

        The last case is what allows FTSL's one-line style; it is safe precisely because
        arity told us we already have every value this statement wanted.
        """
        tok = self.peek()
        if tok.kind in (NEWLINE, EOF):
            return
        if tok.kind == PUNCT_T and tok.text == "}":
            return
        if tok.kind == IDENT:
            return
        raise self.err(f"unexpected {tok.text!r} after '{key}'", tok)

    def check_required(self, node: Node, spec: BlockSpec) -> None:
        for key, prop in spec.props.items():
            if prop.required and key not in node.props:
                raise FtclError(
                    f"'{node.kw}'{f' \"{node.name}\"' if node.name else ''} "
                    f"is missing required property '{key}'", node.loc, self.src)
        for key in spec.required_blocks:
            if node.child(key) is None:
                raise FtclError(
                    f"'{node.kw}'{f' \"{node.name}\"' if node.name else ''} "
                    f"is missing required block '{key}'", node.loc, self.src)


def parse_file(path: str, root: BlockSpec) -> tuple[list[Node], str]:
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    return Parser(src, path, root).parse(), src


def parse_string(src: str, root: BlockSpec, filename: str = "<string>") -> list[Node]:
    return Parser(src, filename, root).parse()
