"""Source-located errors.

Every failure in this front-end carries a file/line/column and prints the offending line
with a caret. That is not politeness -- a rig file is edited by hand and by parameter
sweeps, and "unexpected token" with no location is the difference between a five-second
fix and a hunt.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Loc:
    file: str
    line: int          # 1-based
    col: int           # 1-based

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


class FtclError(Exception):
    """A parse/build error with a source location and a rendered source excerpt."""

    def __init__(self, msg: str, loc: Loc | None = None, src: str | None = None):
        self.msg = msg
        self.loc = loc
        self.src = src
        super().__init__(self.render())

    def render(self) -> str:
        if self.loc is None:
            return self.msg
        head = f"{self.loc}: {self.msg}"
        if not self.src:
            return head
        lines = self.src.splitlines()
        if not (1 <= self.loc.line <= len(lines)):
            return head
        text = lines[self.loc.line - 1]
        gutter = f"{self.loc.line:>5} | "
        caret = " " * (len(gutter) + max(0, self.loc.col - 1)) + "^"
        return f"{head}\n{gutter}{text}\n{caret}"


def did_you_mean(word: str, options) -> str:
    """Suggest the closest option, or '' when nothing is close enough.

    Cheap edit-distance ratio; the point is only to catch typos like `lenght`.
    """
    import difflib
    hit = difflib.get_close_matches(word, list(options), n=1, cutoff=0.6)
    return f"; did you mean '{hit[0]}'?" if hit else ""
