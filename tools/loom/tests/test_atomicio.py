"""`loom.atomicio` — the single atomic-replace writer every loom file writer uses.

Loom hands files to a concurrent reader (the live viewer channel re-emits a scene
while ftrace is still loading the previous emission out of the same directory), so a
half-written ``.ftsl``/``.obj``/``.json`` is a real failure mode, not a theoretical
one.  Two properties are tested here:

- **atomicity** — a failed write leaves the *old* file untouched and no temp litter.
- **the Windows retry** — ``os.replace`` is atomic but not immune to sharing: it
  raises ``PermissionError`` (WinError 5 / 32) when anything holds a handle at that
  instant, including an antivirus scanner or the search indexer.  That was observed
  as an intermittent failure of ``write_obj`` in loom's own suite, so the helper
  retries briefly rather than failing the write.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import atomicio  # noqa: E402
from loom.atomicio import write_atomic  # noqa: E402
from loom.sweep import write_obj  # noqa: E402


def _tmp():
    return tempfile.mkdtemp(prefix="loom_atomicio_")


def _leftovers(d):
    return [n for n in os.listdir(d) if n.endswith(".tmp") or ".tmp" in n]


def test_writes_the_text_and_leaves_no_temp_file():
    d = _tmp()
    p = os.path.join(d, "a.txt")
    write_atomic(p, "hello\n")
    with open(p, encoding="utf-8") as f:
        assert f.read() == "hello\n"
    assert _leftovers(d) == []


def test_replacing_an_existing_file_is_a_whole_file_swap():
    d = _tmp()
    p = os.path.join(d, "a.txt")
    write_atomic(p, "old\n")
    write_atomic(p, "a much longer new body\n")
    with open(p, encoding="utf-8") as f:
        assert f.read() == "a much longer new body\n"
    assert _leftovers(d) == []


def test_a_transient_sharing_error_is_retried_not_raised():
    """Two WinError-5-style failures then success — exactly the scanner-holds-a-handle
    case that made write_obj flake."""
    d = _tmp()
    p = os.path.join(d, "a.txt")
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(13, "Access is denied", dst, 5)
        return real(src, dst)

    os.replace = flaky
    try:
        write_atomic(p, "survived\n")
    finally:
        os.replace = real
    assert calls["n"] == 3
    with open(p, encoding="utf-8") as f:
        assert f.read() == "survived\n"
    assert _leftovers(d) == []


def test_a_persistent_permission_error_still_raises_and_cleans_up():
    """The retry must not paper over a genuine permission problem — and the old file
    must survive it intact, which is the whole point of writing out of place."""
    d = _tmp()
    p = os.path.join(d, "a.txt")
    write_atomic(p, "original\n")
    real = os.replace
    budget = atomicio._RETRY_BUDGET
    atomicio._RETRY_BUDGET = 0.02          # keep the test fast

    def always(src, dst):
        raise PermissionError(13, "Access is denied", dst, 5)

    os.replace = always
    try:
        write_atomic(p, "never lands\n")
    except PermissionError:
        pass
    else:
        raise AssertionError("a persistent PermissionError must propagate")
    finally:
        os.replace = real
        atomicio._RETRY_BUDGET = budget
    with open(p, encoding="utf-8") as f:
        assert f.read() == "original\n"
    assert _leftovers(d) == []


def test_a_non_sharing_error_is_not_retried():
    d = _tmp()
    p = os.path.join(d, "a.txt")
    real = os.replace
    calls = {"n": 0}

    def broken(src, dst):
        calls["n"] += 1
        raise OSError(22, "invalid")

    os.replace = broken
    try:
        write_atomic(p, "x")
    except OSError:
        pass
    else:
        raise AssertionError("a non-sharing OSError must propagate")
    finally:
        os.replace = real
    assert calls["n"] == 1
    assert _leftovers(d) == []


def test_write_obj_goes_through_the_shared_helper():
    """The OBJ writer is the call site the flake was observed on."""
    d = _tmp()
    p = os.path.join(d, "s.obj")
    write_obj(p, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    with open(p, encoding="utf-8") as f:
        body = f.read()
    assert body.startswith("v 0 0 0\n") and body.endswith("f 1 2 3\n")
    assert _leftovers(d) == []


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
