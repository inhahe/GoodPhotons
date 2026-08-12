"""Make this process's stdout/stderr able to carry the characters the tools actually print.

Every tool in `tools/` prints non-ASCII as a matter of course -- `theta` as the Greek letter in
the morph docstrings, en-dashes and arrows in the help text, degree signs and micro prefixes in
the reports. On Windows, `sys.stdout` defaults to the console's ANSI code page (cp1252 here),
which cannot encode any of them, and Python's default error handler for that is `strict`. The
result is not mojibake, it is an **exception**:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u03b8'

and it is raised from inside `argparse._print_message`, so the crash is `--help`. `morph_sweep.py
--help` and `fit_selftest.py --help` both died this way -- the first thing anyone types about a
tool was the one thing guaranteed to fail, while the tool itself ran fine.

Two things make this worth a module rather than a one-liner per tool. First, it must be fixed for
*all* output, not just help: a three-hour run that prints a theta in its final summary would crash
at the finish line, having done all the work. Second, the tempting fix -- delete the offending
characters from the docstrings -- is not a fix at all. It leaves the process unable to print them,
so it regresses silently the moment someone writes a degree sign in a new message, and it makes
the docs worse to avoid a bug that is not in the docs.

This is deliberately a function to call, not an import side effect. `creaturelab` is a library;
importing it should not reach out and mutate the interpreter's streams underneath a caller who
never asked -- notably pytest, which installs its own capture objects, and the FTSL/orchestrator
processes that embed this package.
"""
from __future__ import annotations

import sys


def use_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8. Safe to call more than once, and safe to call
    when the streams are not real files.

    `errors="replace"` is belt and braces rather than expected to fire: with UTF-8 there is
    no character it cannot encode. It is there so that a stream this function could not
    reconfigure at all -- a redirect into an odd wrapper, an embedding host -- degrades to a
    `?` rather than taking the process down, which is the whole point of the module.

    Streams that pytest or another host has replaced with objects lacking `reconfigure` are
    left exactly as they are: they are that host's to manage, and they are not the console
    that has the cp1252 problem.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Already detached, or a stream that reports the method but cannot honour it.
            # Printing a warning here would need the very stream that just failed.
            pass
