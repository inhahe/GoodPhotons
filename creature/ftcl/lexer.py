"""FTCL lexer -- shared with the FTSL family: `#` comments, braces, bare identifiers,
quoted names, numbers with optional unit suffixes.

Two decisions worth stating, because they shape the grammar above:

* **Numbers may carry a unit suffix written with no space** (`0.22m`, `15deg`, `800N`).
  Everything normalises to SI *at lex time*, so nothing downstream ever has to ask what
  units a value is in. Angles in particular become radians immediately, which is why the
  MJCF emitter can declare `angle="radian"` and never convert again. A unit mistake that
  survives to the simulator is a silently wrong creature, so it is caught here.

* **A leading `-` is NOT folded into the number.** `origin 0.05 -0.02 0.03` has to mean
  three values, not a subtraction, and the only robust way to get that is to make
  positional arguments *atoms* (see parser.py) and let unary minus be part of atom
  parsing. Folding the sign in the lexer would make `a -b` ambiguous forever.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import FtclError, Loc

# Suffix -> multiplier into SI. Angles land in radians.
UNITS: dict[str, float] = {
    "m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0,
    "kg": 1.0, "g": 0.001,
    "s": 1.0, "ms": 0.001,
    "n": 1.0,                      # newton (lexer lowercases suffixes)
    "rad": 1.0, "deg": math.pi / 180.0,
    "hz": 1.0,
    "pct": 0.01,
}

PUNCT = set("{}()[],=+-*/%^<>!:")
# Multi-character operators must be tried before their single-character prefixes.
PUNCT2 = ("<=", ">=", "==", "!=", "&&", "||")

NUMBER, STRING, IDENT, PUNCT_T, NEWLINE, EOF = "num", "str", "ident", "punct", "nl", "eof"


@dataclass
class Token:
    kind: str
    text: str
    loc: Loc
    value: float | str | None = None   # numeric value (SI) for NUMBER, body for STRING
    unit: str | None = None            # original unit suffix, kept for diagnostics

    def __repr__(self) -> str:         # pragma: no cover - debugging aid
        return f"Token({self.kind},{self.text!r})"


def tokenize(src: str, filename: str = "<string>") -> list[Token]:
    toks: list[Token] = []
    i, line, col = 0, 1, 1
    n = len(src)

    def loc() -> Loc:
        return Loc(filename, line, col)

    def err(msg: str) -> FtclError:
        return FtclError(msg, loc(), src)

    while i < n:
        c = src[i]

        # --- newlines are significant: they terminate variadic statements -------------
        if c == "\n":
            toks.append(Token(NEWLINE, "\\n", loc()))
            i += 1
            line += 1
            col = 1
            continue
        if c in " \t\r":
            i += 1
            col += 1
            continue

        # --- comments ------------------------------------------------------------------
        if c == "#" or src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
            continue
        if src.startswith("/*", i):
            start = loc()
            i += 2
            col += 2
            while i < n and not src.startswith("*/", i):
                if src[i] == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1
            if i >= n:
                raise FtclError("unterminated /* comment", start, src)
            i += 2
            col += 2
            continue

        # --- quoted string -------------------------------------------------------------
        if c == '"':
            start = loc()
            i += 1
            col += 1
            buf = []
            while i < n and src[i] != '"':
                if src[i] == "\n":
                    raise FtclError("unterminated string", start, src)
                if src[i] == "\\" and i + 1 < n:
                    esc = src[i + 1]
                    buf.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(esc, esc))
                    i += 2
                    col += 2
                    continue
                buf.append(src[i])
                i += 1
                col += 1
            if i >= n:
                raise FtclError("unterminated string", start, src)
            i += 1
            col += 1
            toks.append(Token(STRING, '"' + "".join(buf) + '"', start, value="".join(buf)))
            continue

        # --- number (with optional unit suffix) ----------------------------------------
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            start = loc()
            j = i
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == ".":
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
            if j < n and src[j] in "eE":
                k = j + 1
                if k < n and src[k] in "+-":
                    k += 1
                if k < n and src[k].isdigit():
                    j = k
                    while j < n and src[j].isdigit():
                        j += 1
            num_text = src[i:j]
            # unit suffix: letters/percent glued directly to the number
            k = j
            while k < n and (src[k].isalpha() or src[k] == "%"):
                k += 1
            unit_text = src[j:k]
            mult, unit = 1.0, None
            if unit_text:
                key = "pct" if unit_text == "%" else unit_text.lower()
                if key not in UNITS:
                    col += j - i
                    raise FtclError(
                        f"unknown unit '{unit_text}' (known: {', '.join(sorted(UNITS))})",
                        Loc(filename, line, col), src)
                mult, unit = UNITS[key], key
            toks.append(Token(NUMBER, src[i:k], start,
                              value=float(num_text) * mult, unit=unit))
            col += k - i
            i = k
            continue

        # --- identifier ----------------------------------------------------------------
        if c.isalpha() or c == "_":
            start = loc()
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_."):
                j += 1
            toks.append(Token(IDENT, src[i:j], start))
            col += j - i
            i = j
            continue

        # --- punctuation ---------------------------------------------------------------
        two = src[i:i + 2]
        if two in PUNCT2:
            toks.append(Token(PUNCT_T, two, loc()))
            i += 2
            col += 2
            continue
        if c in PUNCT:
            toks.append(Token(PUNCT_T, c, loc()))
            i += 1
            col += 1
            continue

        raise err(f"unexpected character {c!r}")

    toks.append(Token(EOF, "", Loc(filename, line, col)))
    return toks
