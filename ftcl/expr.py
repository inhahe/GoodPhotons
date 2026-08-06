"""The expression sublanguage.

Expressions are kept **symbolic** rather than folded to numbers at parse time, and that is
the single most important property in this file. `radius = 0.03 * limb_gracility` has to
re-evaluate every time the morph vector changes -- during domain randomisation that is
thousands of times per training run. Constant-folding at parse time would silently destroy
the parameterisation that the whole project depends on (see design.md, "Why a layer above
MJCF at all").

Two entry points, and the distinction is deliberate:

* `parse_expr`  -- full infix, used after `=` where the statement runs to end of line.
* `parse_atom`  -- a single value: optional unary `-`, then a number / reference / call /
  parenthesised expression. Positional arguments use *atoms*, so `origin 0.05 -0.02 0.03`
  is unambiguously three values and never a subtraction. Any real arithmetic in positional
  position just needs parentheses: `origin (femur_len*0.2) -0.02 0.03`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import FtclError, Loc, did_you_mean
from .lexer import EOF, IDENT, NEWLINE, NUMBER, PUNCT_T, Token


# --------------------------------------------------------------------------- AST nodes
class Expr:
    loc: Loc

    def eval(self, env: "Env") -> float:
        raise NotImplementedError

    def refs(self) -> set[str]:
        """Free variable names, for dependency ordering and cycle detection."""
        return set()


@dataclass
class Num(Expr):
    value: float
    loc: Loc

    def eval(self, env): return self.value


@dataclass
class Ref(Expr):
    name: str
    loc: Loc

    def eval(self, env): return env.lookup(self.name, self.loc)
    def refs(self): return {self.name}


@dataclass
class Unary(Expr):
    op: str
    operand: Expr
    loc: Loc

    def eval(self, env):
        v = self.operand.eval(env)
        return -v if self.op == "-" else (0.0 if v else 1.0) if self.op == "!" else v

    def refs(self): return self.operand.refs()


@dataclass
class Binary(Expr):
    op: str
    lhs: Expr
    rhs: Expr
    loc: Loc

    def eval(self, env):
        a = self.lhs.eval(env)
        # short-circuit so `cond && (1/x)` can guard a division
        if self.op == "&&":
            return 1.0 if (a and self.rhs.eval(env)) else 0.0
        if self.op == "||":
            return 1.0 if (a or self.rhs.eval(env)) else 0.0
        b = self.rhs.eval(env)
        try:
            match self.op:
                case "+": return a + b
                case "-": return a - b
                case "*": return a * b
                case "/": return a / b
                case "%": return math.fmod(a, b)
                case "^": return a ** b
                case "<": return float(a < b)
                case ">": return float(a > b)
                case "<=": return float(a <= b)
                case ">=": return float(a >= b)
                case "==": return float(a == b)
                case "!=": return float(a != b)
        except ZeroDivisionError:
            raise FtclError(f"division by zero in '{self.op}'", self.loc, env.src) from None
        raise FtclError(f"unknown operator '{self.op}'", self.loc, env.src)

    def refs(self): return self.lhs.refs() | self.rhs.refs()


@dataclass
class Call(Expr):
    name: str
    args: list[Expr]
    loc: Loc

    def eval(self, env):
        fn = FUNCS.get(self.name)
        if fn is None:
            raise FtclError(f"unknown function '{self.name}'"
                            + did_you_mean(self.name, FUNCS), self.loc, env.src)
        arity, impl = fn
        vals = [a.eval(env) for a in self.args]
        if arity is not None and len(vals) != arity:
            raise FtclError(f"'{self.name}' takes {arity} argument(s), got {len(vals)}",
                            self.loc, env.src)
        try:
            return float(impl(*vals))
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            raise FtclError(f"'{self.name}': {e}", self.loc, env.src) from None

    def refs(self):
        out = set()
        for a in self.args:
            out |= a.refs()
        return out


def _clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x


FUNCS: dict[str, tuple[int | None, object]] = {
    "abs": (1, abs), "sqrt": (1, math.sqrt), "exp": (1, math.exp), "log": (1, math.log),
    "sin": (1, math.sin), "cos": (1, math.cos), "tan": (1, math.tan),
    "asin": (1, math.asin), "acos": (1, math.acos), "atan": (1, math.atan),
    "atan2": (2, math.atan2), "pow": (2, pow), "hypot": (None, math.hypot),
    "floor": (1, math.floor), "ceil": (1, math.ceil), "round": (1, round),
    "sign": (1, lambda x: (x > 0) - (x < 0)),
    "min": (None, min), "max": (None, max), "clamp": (3, _clamp),
    "lerp": (3, lambda a, b, t: a + (b - a) * t),
    "deg": (1, math.degrees), "rad": (1, math.radians),
}

CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


# ------------------------------------------------------------------------- environment
class Env:
    """Lazy, cycle-detecting symbol table.

    Values may be plain floats or unevaluated `Expr`s that reference other names, so a rig
    can say `radius = 0.03 * limb_gracility` where `limb_gracility` is itself derived. The
    resolution is memoised and guarded, because a cyclic definition otherwise blows the
    stack with a traceback that says nothing useful about the rig file.
    """

    def __init__(self, values: dict[str, float | Expr] | None = None,
                 src: str | None = None, parent: "Env | None" = None):
        self.values: dict[str, float | Expr] = dict(values or {})
        self.src = src if src is not None else (parent.src if parent else None)
        self.parent = parent
        self._cache: dict[str, float] = {}
        self._resolving: set[str] = set()

    def child(self, values: dict[str, float | Expr]) -> "Env":
        return Env(values, parent=self)

    def define(self, name: str, value: float | Expr) -> None:
        self.values[name] = value
        self._cache.pop(name, None)

    def lookup(self, name: str, loc: Loc | None = None) -> float:
        if name in self._cache:
            return self._cache[name]
        if name in self.values:
            if name in self._resolving:
                chain = " -> ".join(sorted(self._resolving)) + f" -> {name}"
                raise FtclError(f"cyclic definition: {chain}", loc, self.src)
            self._resolving.add(name)
            try:
                v = self.values[name]
                out = float(v) if isinstance(v, (int, float)) else v.eval(self)
            finally:
                self._resolving.discard(name)
            self._cache[name] = out
            return out
        if name in CONSTS:
            return CONSTS[name]
        if self.parent is not None:
            return self.parent.lookup(name, loc)
        known = sorted(set(self.values) | set(CONSTS))
        raise FtclError(f"undefined name '{name}'" + did_you_mean(name, known),
                        loc, self.src)

    def names(self) -> set[str]:
        out = set(self.values) | set(CONSTS)
        if self.parent:
            out |= self.parent.names()
        return out


# ----------------------------------------------------------------------------- parsing
# Lowest binding first. `^` is right-associative and binds tighter than unary minus, so
# `-x^2` is -(x^2), matching every other language a rig author will have used.
_BINOPS = [
    ("||",), ("&&",), ("==", "!="), ("<", ">", "<=", ">="),
    ("+", "-"), ("*", "/", "%"),
]


class _P:
    def __init__(self, toks: list[Token], src: str | None):
        self.t = toks
        self.i = 0
        self.src = src

    def peek(self) -> Token: return self.t[self.i]

    def take(self) -> Token:
        tok = self.t[self.i]
        if tok.kind is not EOF:
            self.i += 1
        return tok

    def skip_nl(self) -> None:
        """Newlines are insignificant *inside* an expression once it has committed."""
        while self.t[self.i].kind == NEWLINE:
            self.i += 1

    def peek_past_newlines(self) -> tuple[Token, int]:
        """Look ahead over blank lines without consuming them.

        This is what allows an operator-leading continuation line:

            hind_drop = L_femur * cos(a)
                      + L_tibia * cos(b)

        The continuation is only accepted when the first real token *is* a binary
        operator, so a newline still reliably ends a statement in every other case --
        the property that keeps the block grammar unambiguous.
        """
        k = self.i
        while self.t[k].kind == NEWLINE:
            k += 1
        return self.t[k], k

    def err(self, msg: str, tok: Token | None = None) -> FtclError:
        tok = tok or self.peek()
        return FtclError(msg, tok.loc, self.src)


def _parse_binary(p: _P, level: int) -> Expr:
    if level >= len(_BINOPS):
        return _parse_unary(p)
    lhs = _parse_binary(p, level + 1)
    while True:
        nxt, at = p.peek_past_newlines()
        if not (nxt.kind == PUNCT_T and nxt.text in _BINOPS[level]):
            break
        p.i = at                      # commit to the continuation line
        op = p.take()
        p.skip_nl()
        rhs = _parse_binary(p, level + 1)
        lhs = Binary(op.text, lhs, rhs, op.loc)
    return lhs


def _parse_unary(p: _P) -> Expr:
    tok = p.peek()
    if tok.kind == PUNCT_T and tok.text in ("-", "+", "!"):
        p.take()
        p.skip_nl()
        operand = _parse_unary(p)
        return operand if tok.text == "+" else Unary(tok.text, operand, tok.loc)
    return _parse_power(p)


def _parse_power(p: _P) -> Expr:
    base = _parse_primary(p)
    if p.peek().kind == PUNCT_T and p.peek().text == "^":
        op = p.take()
        p.skip_nl()
        return Binary("^", base, _parse_unary(p), op.loc)   # right-assoc
    return base


def _parse_primary(p: _P) -> Expr:
    tok = p.take()
    if tok.kind == NUMBER:
        return Num(float(tok.value), tok.loc)
    if tok.kind == IDENT:
        if p.peek().kind == PUNCT_T and p.peek().text == "(":
            p.take()
            p.skip_nl()
            args: list[Expr] = []
            if not (p.peek().kind == PUNCT_T and p.peek().text == ")"):
                while True:
                    args.append(_parse_binary(p, 0))
                    p.skip_nl()
                    if p.peek().kind == PUNCT_T and p.peek().text == ",":
                        p.take()
                        p.skip_nl()
                        continue
                    break
            close = p.take()
            if not (close.kind == PUNCT_T and close.text == ")"):
                raise p.err("expected ')' to close function call", close)
            return Call(tok.text, args, tok.loc)
        return Ref(tok.text, tok.loc)
    if tok.kind == PUNCT_T and tok.text == "(":
        p.skip_nl()
        inner = _parse_binary(p, 0)
        p.skip_nl()
        close = p.take()
        if not (close.kind == PUNCT_T and close.text == ")"):
            raise p.err("expected ')'", close)
        return inner
    raise p.err(f"expected a value, found {tok.text!r}", tok)


def parse_expr(toks: list[Token], pos: int, src: str | None = None) -> tuple[Expr, int]:
    """Full infix expression starting at `pos`. Returns (expr, next_pos)."""
    p = _P(toks, src)
    p.i = pos
    e = _parse_binary(p, 0)
    return e, p.i


def parse_atom(toks: list[Token], pos: int, src: str | None = None) -> tuple[Expr, int]:
    """A single positional value: optional unary sign, then number/ref/call/'(' expr ')'.

    Never consumes an infix operator, which is what keeps `0.05 -0.02` two values.
    """
    p = _P(toks, src)
    p.i = pos
    tok = p.peek()
    if tok.kind == PUNCT_T and tok.text in ("-", "+"):
        p.take()
        inner = _parse_primary(p)
        e: Expr = inner if tok.text == "+" else Unary("-", inner, tok.loc)
    else:
        e = _parse_primary(p)
    return e, p.i


def is_value_start(tok: Token) -> bool:
    """Could this token begin a positional value?"""
    if tok.kind in (NUMBER, IDENT):
        return True
    return tok.kind == PUNCT_T and tok.text in ("-", "+", "(")


__all__ = ["Expr", "Num", "Ref", "Unary", "Binary", "Call", "Env",
           "parse_expr", "parse_atom", "is_value_start", "FUNCS", "CONSTS",
           "EOF", "NEWLINE"]
