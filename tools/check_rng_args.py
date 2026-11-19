"""Source check: no expression evaluates two RNG draws in an order the compiler chooses.

    python tools/check_rng_args.py [--src src]

C++ leaves unsequenced the arguments of a call and the operands of most operators, so
`f(rng.uniform(), rng.uniform())` or `rng.uniform() * rng.uniform()` makes an image depend on the
compiler that built it, and a host twin and its device twin may take one stream's draws in opposite
orders (design.md, "The draws a render takes"). A draw is `<x>.uniform()` (or .uniformOpen / .next /
.nextFloat / .uni / .u01 / .uniformF / .nextDouble / .next64 / .nextU32 / .uniformD, and `->`), or a
call of a callable named for one (`rng()`, `prng()`, `frand()`, `uni()`, `rnd()`, `u01()`).

Flagged:
  - a call whose arguments hold draws in two of them, a draw counting for every bracket around it up
    to its statement: `f((k + rng.uniform()) / n, rng.uniform())`, `f(Vec3{rng.uniform()}, rnd())`;
  - one operand (a call's argument, a parenthesis, a braced clause, a statement's expression) holding
    two draws, through parentheses, with no `&&`, `||`, `?` or `:` between them to sequence them:
    `a * rng.uniform() + b * (rng.uniform() * 2)`.
Sequenced, and passed: a declaration list (`const double u = rng.uniform(), v = rng.uniform();`), a
braced initializer's clauses (`Vec3{rng.uniform(), rng.uniform()}`), the comma operator. A brace is
a block, not an initializer, where it holds a statement (a `;`, or `if`, `return` and the like): a
function body after `const`, a lambda's after its return type. A helper that draws inside itself (a
`unitDir()` lambda over rng) cannot be seen here: write it as the rule says. Comments, strings,
character literals and digit separators (`500'000`) are skipped.

Lists each offender as file:line and exits 1 when there is one; exits 2 when the scanner fails its
own cases (CASES), which it runs first.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
BS = chr(92)
DRAW = re.compile(r"\b\w+(?:\.|->)(?:uniform|uniformOpen|next|nextFloat|uni|u01|uniformF|nextDouble|next64|nextU32"
                  r"|uniformD)\(\)"
                  r"|\b(?:rng|prng|frand|uni|rnd|u01)\(\)")
SEQ = re.compile(r"&&|\|\||\?|(?<!:):(?!:)")   # a conditional's `:` too, not a scope's `::`
STMT_WORD = re.compile(r"\b(?:if|for|while|switch|return|do|case|break|continue)\b")
NOT_CALL = re.compile(r"\b(?:if|while|for|switch|return|sizeof|catch)\s*$")

# (source, whether the scanner must flag it)
CASES = [
    ("x = f(rng.uniform(), rng.uniform());", True),
    ("x = f((k + rng.uniform()) / n, rng.uniform());", True),
    ("x = f(Vec3{rng.uniform(), 0.0, 0.0}, rnd());", True),
    ("x = rng.uniform() * rng.uniform();", True),
    ("x = a + b * rng.uniform() + c * (rng.uniform() * 2);", True),
    ("x = rng.uniform() * std::sqrt(rng.uniform());", True),
    ("x = rng.uniformOpen() * rng.uniform();", True),
    ("x = (*fn)(rng.uniform(), rnd());", True),
    ("x = table[i](rng.uniform(), rnd());", True),
    ("void g() const { x = rnd() * rnd(); }", True),
    ("auto l = [&]() -> Vec3 { return f(rng.uniform(), rng.uniform()); };", True),
    ("auto l = [&] { return rng.uniform() + rng.uniform(); };", True),
    ("const double u = rng.uniform(), v = rng.uniform();", False),
    ("Vec3 o{rng.uniform() - 0.5, rng.uniform() - 0.5, rng.uniform()};", False),
    ("x = Vec3{rng.uniform() * 4 - 2, rng.uniform() * 4 - 2, rng.uniform()};", False),
    ("x = f(Vec3{rng.uniform(), rng.uniform(), 0.0});", False),
    ("if (a && rng.uniform() < p && rng.uniform() < q) x = 1;", False),
    ("x = c ? rng.uniform() : rng.uniform();", False),
    ("for (int i = 0; i < n; ++i) x += rng.uniform();", False),
    ("void g() const { const int vi = pick(rng.uniform()); y = rng.uniform(); }", False),
    ("int n = 500'000; x = f(rng.uniform()); y = g(rng.uniform());", False),
    ("// f(rng.uniform(), rng.uniform())" + NL + "x = 1;", False),
    ("const char* s = \"rng.uniform() * rng.uniform()\";", False),
]


def strip(text):
    """Blank comments, string and character literals (not digit separators), keeping offsets."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find(NL, i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != NL:
                    out[k] = " "
            i = j
        elif text[i] == "'" and i > 0 and text[i - 1].isalnum() and i + 1 < n and text[i + 1].isalnum():
            i += 1   # a digit separator, 500'000
        elif text[i] in "\"'":
            q = text[i]
            j = i + 1
            while j < n and text[j] != q and text[j] != NL:
                j += 2 if text[j] == BS else 1
            for k in range(i + 1, min(j, n)):
                out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


class Frame:
    """A bracket, a block or a statement, and the draws in each of its comma-separated slots."""

    def __init__(self, kind, pos, call):
        self.kind, self.pos, self.call = kind, pos, call   # kind: "(", "[", "{" (an initializer),
        self.slot = 0                                      # "block" or "stmt"
        self.draws = {}                 # slot -> [positions of its draws, through parentheses]
        self.slots_with_draws = set()   # the slots a draw sits in, through any bracket


def offenders(raw):
    """The offending lines of source text `raw`: sorted (line number, the line's text) pairs."""
    s = strip(raw)
    draws = {m.start(): m.end() for m in DRAW.finditer(s)}
    hits = set()
    n = len(s)

    def close(fr):
        if fr.call and len(fr.slots_with_draws) >= 2:
            hits.add(fr.pos)
        for ps in fr.draws.values():
            if len(ps) >= 2 and not SEQ.search(s[ps[0]:ps[-1]]):
                hits.add(ps[0])

    def holds_statement(at):
        """Whether the brace at `at` holds a statement -- a `;` or a statement keyword at its own
        depth -- so is a block rather than a braced initializer, which holds neither."""
        depth = 0
        for k in range(at + 1, n):
            ch = s[k]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    return False
                depth -= 1
            elif depth == 0 and (ch == ";" or STMT_WORD.match(s, k)):
                return True
        return False

    def before(at):
        """The position of the last non-blank character before `at`, or -1."""
        j = at - 1
        while j >= 0 and s[j] in " \t\r" + NL:
            j -= 1
        return j

    stack = [Frame("stmt", -1, False)]
    i = 0
    while i < n:
        if i in draws:
            # The call rule counts the draw for every bracket up to its statement (a braced
            # initializer's clauses, sequenced among themselves, still sit in one argument of an
            # enclosing call); the one-operand rule counts it only through parentheses.
            through_parens = True
            for fr in reversed(stack):
                fr.slots_with_draws.add(fr.slot)
                if through_parens:
                    fr.draws.setdefault(fr.slot, []).append(i)
                if fr.kind == "{":
                    through_parens = False
                if fr.kind == "stmt":
                    break
            i = draws[i]
            continue
        c = s[i]
        if c in "([":
            j = before(i)
            # a call: after a name, a template's `>`, or a callee that is itself an expression --
            # `(*fn)(...)`, `table[i](...)`; a lambda's parameters (`[&](...)`) hold no draws
            call = c == "(" and j >= 0 and (s[j].isalnum() or s[j] in "_>)]") and \
                not NOT_CALL.search(s[max(0, j - 8):j + 1])
            stack.append(Frame(c, i, call))
        elif c == "{":
            j = before(i)
            word = re.search(r"(\w+)\s*$", s[max(0, j - 10):j + 1])
            if (j < 0 or s[j] in ");{}:]" or (word and word.group(1) in ("else", "do", "try"))
                    or holds_statement(i)):
                close(stack[-1])           # a block: the statement before it ends, and its own
                stack.append(Frame("block", i, False))   # statements are frames of their own
                stack.append(Frame("stmt", i, False))
            else:
                stack.append(Frame("{", i, False))
        elif c in ")]":
            if len(stack) > 1 and stack[-1].kind in "([":
                close(stack.pop())
        elif c == "}":
            while len(stack) > 1 and stack[-1].kind not in ("{", "block"):
                close(stack.pop())
            if len(stack) > 1:
                fr = stack.pop()
                close(fr)
                if fr.kind == "block":
                    stack.append(Frame("stmt", i, False))
        elif c == ";":
            # a statement ends; a for header's `;` separates its parenthesis's slots
            if stack[-1].kind == "stmt":
                close(stack.pop())
                stack.append(Frame("stmt", i, False))
            elif stack[-1].kind in "([":
                stack[-1].slot += 1
        elif c == ",":
            stack[-1].slot += 1
        i += 1
    close(stack[-1])
    lines = raw.split(NL)
    return sorted({(ln, lines[ln - 1].strip()[:120])
                   for ln in (s.count(NL, 0, max(pos, 0)) + 1 for pos in hits)})


def selftest():
    """The cases the scanner gets wrong, as (source, expected) pairs."""
    return [(src, want) for src, want in CASES if bool(offenders(src)) != want]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--src", default=os.path.join(ROOT, "src"))
    a = ap.parse_args()
    wrong = selftest()
    for src, want in wrong:
        print("scanner %s: %s" % ("misses" if want else "wrongly flags", src.replace(NL, " / ")))
    if wrong:
        print("FAIL: the scanner gets %d of its %d cases wrong" % (len(wrong), len(CASES)))
        return 2
    total = 0
    for name in sorted(os.listdir(a.src)):
        if not name.endswith((".h", ".hpp", ".cpp", ".cu", ".cuh")):
            continue
        with open(os.path.join(a.src, name), encoding="utf-8", errors="replace") as f:
            raw = f.read()
        for ln, text in offenders(raw):
            print("%s:%d: %s" % (name, ln, text))
            total += 1
    print("%d expression(s) evaluate two RNG draws in an order the compiler chooses" % total if total else
          "ok: every expression's RNG draws are sequenced (and the scanner its %d cases)" % len(CASES))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
