"""M12 tests: the resident preview-server client (:mod:`loom.preview`).

The heavy end-to-end path (spawning a real ``ftrace -serve`` and opening a GDI
window) is exercised by ``scraps/preview_smoke.py``, not here — these tests cover
the client logic in isolation with a *fake* server that speaks the same
line-oriented protocol, so they run fast and headless.  Covered: command-line
assembly (budget / res / window / extra flags), the request→"[serve] done"
handshake, frame emission + naming, and clean shutdown.  Runnable directly or
under pytest.
"""

from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import Scene, Camera, Sphere, Material, Sine, vec, Const  # noqa: E402
from loom import preview as preview_mod  # noqa: E402
from loom.preview import PreviewServer  # noqa: E402


class _FakeStdin:
    """Collects written lines and, per line, enqueues a matching done marker."""
    def __init__(self, out_queue):
        self.lines = []
        self._out = out_queue
        self.closed = False

    def write(self, s):
        self.lines.append(s)
        line = s.strip()
        if line and line not in ("quit", "exit"):
            self._out.append(f"[serve] done {line}\n")

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _FakeStdout:
    def __init__(self, queue):
        self._q = queue

    def readline(self):
        return self._q.pop(0) if self._q else ""  # "" == EOF


class _FakeProc:
    """Minimal Popen stand-in speaking the -serve protocol in-memory."""
    def __init__(self):
        self._q = ["[serve] ready\n"]
        self.stdout = _FakeStdout(self._q)
        self.stdin = _FakeStdin(self._q)
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def kill(self):
        self._alive = False


def _scene():
    y = Sine(cycles=1, amp=0.3)
    sc = Scene(Camera(eye=(0, 0.6, 2.2), look_at=(0, 0, 0), up=(0, 1, 0),
                      fov_y=40, mode="R", res=(64, 64)))
    sc.add(Material("ball", "diffuse", reflect=0.8),
           Sphere(center=vec(Const(0.0), y, Const(0.0)), radius=0.4, material="ball"))
    return sc


def test_build_cmd_assembles_flags():
    outdir = _tmpdir("cmd")
    srv = PreviewServer(outdir=outdir, name="p", res=128, noise=5.0,
                        interval=3.0, extra_args=["-mode", "R"])
    _patch_ftrace(srv)
    cmd = srv._build_cmd(outdir + os.sep + "p000.ftsl")
    assert "-serve" in cmd and "-in" in cmd
    assert cmd[cmd.index("-in") + 1].endswith("p000.ftsl")
    assert "-noise" in cmd and cmd[cmd.index("-noise") + 1] == "5"
    assert "-r" in cmd and cmd[cmd.index("-r") + 1] == "128"
    assert "-window" in cmd
    assert cmd[-2:] == ["-mode", "R"]


def test_explicit_n_overrides_default_noise():
    srv = PreviewServer(outdir=_tmpdir("n"), name="p", n=250000)
    assert "-n" in srv._budget and "-noise" not in srv._budget


def test_default_budget_is_noise():
    srv = PreviewServer(outdir=_tmpdir("d"), name="p")
    assert srv._budget[0] == "-noise"


def test_protocol_handshake_and_frames():
    outdir = _tmpdir("proto")
    scene = _scene()
    srv = PreviewServer(outdir=outdir, name="pv", res=64, n=1000)
    # Boot on a fake process instead of a real ftrace.
    fake = _FakeProc()
    srv.proc = fake
    srv._wait_for("[serve] ready")  # consume the ready line the fake emits
    # Emit + stream three frames; each returns a written .ftsl and gets a done marker.
    for k in range(3):
        fp = srv.render_frame(scene, k, 3)
        assert os.path.exists(fp)
    # The three requests were sent verbatim (one per frame).
    reqs = [ln.strip() for ln in fake.stdin.lines]
    assert len(reqs) == 3 and all(r.endswith(".ftsl") for r in reqs)
    # Frame naming is zero-padded to at least 3 digits.
    assert reqs[0].endswith("pv000.ftsl") and reqs[2].endswith("pv002.ftsl")


def test_close_sends_quit():
    srv = PreviewServer(outdir=_tmpdir("close"), name="pv")
    fake = _FakeProc()
    srv.proc = fake
    srv.close()
    assert any(ln.strip() == "quit" for ln in fake.stdin.lines)
    assert fake.stdin.closed
    assert srv.proc is None


def test_wait_for_raises_on_eof():
    srv = PreviewServer(outdir=_tmpdir("eof"), name="pv")

    class _DeadProc:
        stdout = _FakeStdout([])  # immediate EOF
        def poll(self):
            return 3
    srv.proc = _DeadProc()
    try:
        srv._wait_for("[serve] done")
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on premature EOF")


# --- helpers ---------------------------------------------------------------
def _tmpdir(tag):
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "loom_preview_test", tag)
    os.makedirs(d, exist_ok=True)
    return d


def _patch_ftrace(srv):
    # _build_cmd calls find_ftrace(); stub it so tests don't need a built binary.
    preview_mod.find_ftrace = lambda: "ftrace.exe"


def _run_all():
    _patch_ftrace(None)
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
