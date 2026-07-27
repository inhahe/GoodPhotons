#!/usr/bin/env python3
"""Cross-check ftrace's record-stop parser against loom's twin of it.

``loom/record.py``'s ``Record._parse_stops`` and ftrace's ``Parser::parseChannelStops``
(``src/ftsl.h``, driven by ``src/record_ladder.h``) are declared **twins**: loom emits
the ``.ftsl`` that ftrace reads, so the two must agree exactly about where a channel
line's stop boundaries fall.  A disagreement is not a parse error on either side — both
happily produce *a* record — it is a silent wrong-render bug, which is precisely the
kind that survives a test suite.  Hence this checker.

The comparison is on **stop count per channel**, which is the thing the delimiter
precedence ladder decides (``0 0 0, 1 1 1`` is 2 stops, not 6; ``rgb .5 .5 .5`` is 1,
not 3).  ftrace has no "dump records" mode, so the count is probed indirectly: append a
throwaway material whose slot reads ``rec.chan[999]`` and read the loader's reply,
``stop index 999 out of range (0..N-1)``.  Colour-kind and scalar-kind channels are
reached through different slots, so both are tried and whichever answers wins.

Usage::

    python tools/check_record_twins.py [scene.ftsl ...]      # default: scenes/_record_*.ftsl

Exit status is non-zero if any channel disagrees (so it can gate a commit).
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "tools", "loom"))

from loom.record import Record  # noqa: E402

_FTRACE = os.path.join(_REPO, "ftrace.exe")
_RANGE = re.compile(r"out of range \(0\.\.(-?\d+)\)")

# The two probe materials.  A record channel is either colour-kind or scalar-kind and
# ftrace routes them through different slot parsers, so a channel that rejects one is
# retried on the other; only "no channel called that" fails both.
_PROBES = (
    'material "__twincheck" {{ type diffuse  reflect {sel} }}',
    'material "__twincheck" {{ type glossy   reflect whitewall  roughness {sel} }}',
)


def _stop_count(scene_text: str, sel: str) -> int | None:
    """ftrace's stop count for ``rec.chan``, or None if it never answered."""
    for tmpl in _PROBES:
        probe = scene_text + "\n" + tmpl.format(sel=sel + "[999]") + "\n"
        fd, path = tempfile.mkstemp(suffix=".ftsl", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(probe)
            r = subprocess.run([_FTRACE, path, "-parseonly"],
                               capture_output=True, text=True)
        finally:
            os.unlink(path)
        m = _RANGE.search(r.stdout + r.stderr)
        if m:
            return int(m.group(1)) + 1
    return None


def check(path: str) -> int:
    """Report every channel of every record in ``path``; return the disagreement count."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        recs = Record.parse_all(text)
    except ValueError as e:
        print(f"BAD {os.path.basename(path)}: loom could not parse it: {e}")
        return 1
    bad = 0
    for rec in recs:
        for ch in rec.channels:
            got = _stop_count(text, f"{rec.name}.{ch.name}")
            want = len(ch.stops)
            ok = got == want
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {os.path.basename(path)}"
                  f"  {rec.name + '.' + ch.name:24s} ftrace={got} loom={want}")
    return bad


def main(argv: list[str]) -> int:
    paths = argv[1:] or sorted(glob.glob(os.path.join(_REPO, "scenes", "_record_*.ftsl")))
    if not os.path.exists(_FTRACE):
        print(f"ftrace.exe not found at {_FTRACE} — build it first", file=sys.stderr)
        return 2
    bad = sum(check(p) for p in paths)
    print(f"\n{'ALL AGREE' if not bad else f'{bad} DISAGREEMENT(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
