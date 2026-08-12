"""Every tool must be able to print its own help on a Windows console.

This exists because of a bug that was live in the tree and invisible to the whole rest of the
suite: `tools/morph_sweep.py --help` and `tools/fit_selftest.py --help` crashed with

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u03b8'

Their docstrings mention the morph vector as Greek theta, argparse echoes the docstring into the
help text, and `sys.stdout` on a Windows console defaults to the ANSI code page (cp1252 here)
with `errors="strict"`. So the *first* thing anyone types about a tool was the one thing certain
to fail, while the tool itself ran fine. Nothing caught it because pytest replaces stdout with a
UTF-8-capable capture object, and every existing test imports the modules rather than running
them -- the failure needs a real subprocess with a narrow encoding to appear at all.

Hence both unusual things here. These run the tools as **subprocesses**, because the bug is in
the interpreter's stream setup and cannot exist in-process under pytest. And they force
`PYTHONIOENCODING=cp1252`, so the test reproduces the Windows console on any machine and this
file keeps its teeth when someone runs the suite on Linux.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = sorted(p.name for p in (ROOT / "tools").glob("*.py") if not p.name.startswith("_"))


def _run(args: list[str], encoding: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING=encoding, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, cwd=ROOT, timeout=300)


def test_the_tools_directory_is_not_empty():
    """A glob that silently matched nothing would make every test below vacuously pass."""
    assert len(TOOLS) >= 6, TOOLS


@pytest.mark.parametrize("tool", TOOLS)
def test_help_survives_a_narrow_console(tool):
    """`--help` on a cp1252 stdout. This is the actual regression."""
    r = _run([str(ROOT / "tools" / tool), "--help"], "cp1252")
    assert r.returncode == 0, f"{tool} --help failed on cp1252:\n{r.stderr}"
    assert "UnicodeEncodeError" not in r.stderr
    assert r.stdout.strip(), f"{tool} --help printed nothing"


@pytest.mark.parametrize("tool", TOOLS)
def test_help_is_the_same_text_on_a_wide_console(tool):
    """The fix must widen the console, not narrow the docs.

    Deleting the non-ASCII characters from the docstrings would pass the test above while
    leaving the tools unable to print them -- so it would regress the moment someone wrote a
    degree sign in a new message. Comparing cp1252 output against UTF-8 output catches that:
    under a `?`-replacing fallback the two differ, and under a real fix they are identical.
    """
    narrow = _run([str(ROOT / "tools" / tool), "--help"], "cp1252")
    wide = _run([str(ROOT / "tools" / tool), "--help"], "utf-8")
    assert narrow.returncode == wide.returncode == 0
    assert narrow.stdout == wide.stdout, f"{tool}: help text degrades on a cp1252 console"


def test_a_tool_that_prints_theta_really_does_print_it():
    """Guards the guard: if no tool's help contained non-ASCII any more, the tests above
    would keep passing while testing nothing, and the next person would not know."""
    joined = "".join(_run([str(ROOT / "tools" / t), "--help"], "utf-8").stdout for t in TOOLS)
    assert not joined.isascii(), (
        "no tool's --help contains non-ASCII any more -- either the docstrings were stripped "
        "(the wrong fix; see creaturelab/console.py) or this suite has lost its subject")
