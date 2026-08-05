"""Atomic file replacement — the one implementation every loom writer uses.

Loom hands files to a *concurrent reader* in several places: the live viewer channel
(§F4) re-emits a scene on a worker thread while ftrace is still loading the previous
emission's assets out of the same directory, and an editor polls a ``CurveDrive``
sidecar while loom rewrites it.  A plain ``open(path, "w")`` truncates first and fills
in afterwards, so a reader in that window sees an empty or half-written file — an
``.obj`` whose ``f`` lines reference vertices that do not exist yet, or a ``.json``
that fails to parse.  Writing a temp file in the *same directory* and then
``os.replace``-ing it makes the swap a single filesystem operation: a reader sees
either the whole old file or the whole new one, never a splice.

**Why the retry.**  ``os.replace`` is atomic on Windows too, but it is not immune to
*sharing*: ``MoveFileEx`` fails with ``ERROR_ACCESS_DENIED`` (WinError 5) or
``ERROR_SHARING_VIOLATION`` (WinError 32) if anything holds a handle to the source or
destination at that instant — which on Windows includes the antivirus scanner and the
search indexer opportunistically opening a file loom just closed, not only ftrace
reading it.  That is a transient, sub-second condition, and it showed up as an
intermittent failure of loom's own test suite (``write_obj`` into a temp dir), so it is
certainly reachable in the live channel this helper exists for.  Retrying briefly
turns a spurious hard failure into a few milliseconds of delay; a genuine permission
problem still raises after the budget, unchanged.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

#: Total time to keep retrying a transiently-blocked replace, in seconds.  Long enough
#: to outlast a scanner's handle, short enough that a real failure is still prompt.
_RETRY_BUDGET = 1.0
_RETRY_START = 0.002        # first backoff; doubles up to a 64 ms ceiling
_RETRY_MAX = 0.064


def replace_atomic(tmp: str, path: str) -> None:
    """``os.replace(tmp, path)``, retrying briefly past transient sharing errors."""
    deadline = time.monotonic() + _RETRY_BUDGET
    delay = _RETRY_START
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # WinError 5 / 32 — someone (ftrace, an editor, a scanner) has a handle.
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, _RETRY_MAX)


def write_atomic(path, text: str, *, suffix: str = ".tmp",
                 encoding: Optional[str] = "utf-8") -> None:
    """Write ``text`` to ``path`` so a concurrent reader never sees a partial file.

    The temp file is created in ``path``'s own directory, because ``os.replace`` is
    only atomic within a single filesystem.  On any failure the temp file is removed,
    so a crashed write leaves the old file intact and no litter behind.
    """
    path = str(path)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir=d)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        replace_atomic(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
