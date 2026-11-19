"""Source check: no call takes two or more RNG draws as its arguments.

    python tools/check_rng_args.py [--src src]

C++ leaves the order in which a call evaluates its arguments to the compiler, so
`f(rng.uniform(), rng.uniform())` makes an image depend on the compiler that built it, and a host
twin and its device twin may take one stream's draws in opposite orders (design.md, "The draws a
render takes"). A draw is `<x>.uniform()` (or .next / .nextFloat / .uni / .u01 / .uniformF /
.nextDouble / .next64 / .nextU32 / .uniformD, and `->`), or a call of a callable named for one
(`rng()`, `prng()`, `frand()`, `uni()`, `rnd()`, `u01()`). A call is flagged when draws sit in two of
its arguments; a declaration list (`const double u = rng.uniform(), v = rng.uniform();`) and a braced
initializer (`Vec3{...}`) are sequenced and pass. Comments and strings are skipped. Lists each
offender as file:line and exits 1 when there is one.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
BS = chr(92)
DRAW = re.compile(r"\b\w+(?:\.|->)(?:uniform|next|nextFloat|uni|u01|uniformF|nextDouble|next64|nextU32|uniformD)\(\)"
                  r"|\b(?:rng|prng|frand|uni|rnd|u01)\(\)")


def strip(text):
    """Blank comments, string and character literals, keeping offsets."""
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


def offenders(path):
    raw = open(path, encoding="utf-8", errors="replace").read()
    s = strip(raw)
    draws = {m.start(): m.end() for m in DRAW.finditer(s)}
    stack, found = [], {}
    i, n = 0, len(s)
    while i < n:
        if i in draws:
            if stack and stack[-1][0] == "(":
                found.setdefault(stack[-1][1], set()).add(stack[-1][2])
            i = draws[i]
            continue
        c = s[i]
        if c in "({[":
            stack.append([c, i, 0])
        elif c in ")}]":
            if stack:
                stack.pop()
        elif c == "," and stack:
            stack[-1][2] += 1
        i += 1
    lines = raw.split(NL)
    hits = []
    for key, slots in found.items():
        if len(slots) < 2:
            continue
        j = key - 1
        while j >= 0 and s[j] in " \t\r" + NL:
            j -= 1
        if j < 0 or not (s[j].isalnum() or s[j] in "_>"):
            continue   # a cast or a grouping, not a call
        ln = s.count(NL, 0, key) + 1
        hits.append((ln, lines[ln - 1].strip()[:120]))
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--src", default=os.path.join(ROOT, "src"))
    a = ap.parse_args()
    total = 0
    for name in sorted(os.listdir(a.src)):
        if not name.endswith((".h", ".hpp", ".cpp", ".cu", ".cuh")):
            continue
        for ln, text in offenders(os.path.join(a.src, name)):
            print("%s:%d: %s" % (name, ln, text))
            total += 1
    print("%d call(s) take two or more RNG draws as arguments" % total if total else
          "ok: no call takes two RNG draws as arguments")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
