"""Regression check: an image write survives the output file being held by another process.

    D:/python314/python.exe -u tools/check_write_locked.py [--out <dir>] [--exe ftrace.exe]

On Windows a backup program or virus scanner can hold a just-written image for seconds, and a
viewer can keep one open; a render's next write to it (a periodic one, or the final one) used to
fail outright: `error: could not write <path>`. ftrace writes a temporary file beside the target and
moves it into place, retrying the move for up to 10 s while the target is held and, from 2.5 s on,
writing in place when the holder shares writing (src/file_replace.h). Checked through `-topng`,
which writes through the same function a render does:
  1. SCANNER, 3 s: the target opened with NO sharing (as a scanner may). The write must WAIT for
     the release (it finishes after the release) and then succeed.
  2. VIEWER, 15 s: opened sharing reads and writes but not deletion, so the move cannot replace
     it. The write must succeed in place well before the hold ends (under 5 s).
  3. SCANNER, 15 s: longer than the retry window. The write must fail cleanly: exit 1, the old
     file untouched, no temporary file left behind.
  4. READ-ONLY target: it can never be replaced, so the write must fail at once (under 2 s).
  5. HELD TEMPORARY: a stale `<target>.tmp` held with no sharing (a scanner still on the last
     one). The write must step around it and succeed at once, leaving the held one alone.
Windows only (the holds are Win32 share modes). Exits 1 on any failure.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import os
import stat
import subprocess
import sys
import threading
import time

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
SHARE_NONE, SHARE_READ_WRITE = 0, 0x1 | 0x2
INVALID = ctypes.c_void_p(-1).value

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
k32.CloseHandle.argtypes = [wt.HANDLE]       # a 64-bit handle, not a C int
k32.CloseHandle.restype = wt.BOOL


def hold(path, share, seconds, ready, released):
    """Open `path` with share mode `share`, signal `ready`, keep it `seconds`, close it, and
    record in `released` when."""
    h = k32.CreateFileW(path, GENERIC_READ, share, None, OPEN_EXISTING, 0, None)
    if h in (None, INVALID):
        ready.set()
        raise OSError("CreateFileW failed: %d" % ctypes.get_last_error())
    ready.set()
    time.sleep(seconds)
    k32.CloseHandle(h)
    released.append(time.perf_counter())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_write_locked"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)          # CreateProcess wants a full path for a relative one
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):                 # a previous run's files must not pass for this run's
        if f.startswith("locked_"):
            fp = os.path.join(d, f)
            os.chmod(fp, stat.S_IWRITE)
            os.remove(fp)
    scene = os.path.join(d, "s.ftsl")
    with open(scene, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join([
            'camera "c" { eye 0 0.6 4  look_at 0 0.4 0  up 0 1 0  fov_y 35  film { res 64 48 } }',
            'material "w" { type diffuse reflect 0.6 }',
            'light sun { dir 0.3 0.8 0.5  angle 0.53  spd blackbody 5800  intensity 1e-13 }',
            'sphere { center 0 0.5 0  radius 0.5  material w }',
            'quad { corner -3 0 -3  u 6 0 0  v 0 0 6  material w }']) + NL)
    src = os.path.join(d, "src.png")
    r = subprocess.run([a.exe, "-in", scene, "-mode", "D", "-device", "cpu", "-spp", "2", "-checkpoint",
                        "-window-min", "-o", src], capture_output=True, text=True, encoding="utf-8", errors="replace")
    ftbuf = src + ".ftbuf"
    if r.returncode != 0 or not os.path.exists(ftbuf):
        print(r.stdout[-1500:], r.stderr[-1500:]); print("FAIL: no checkpoint to develop"); return 1

    def develop(target):
        t0 = time.perf_counter()
        r = subprocess.run([a.exe, "-topng", ftbuf, target], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return r.returncode, time.perf_counter() - t0, time.perf_counter()

    def size_of(target):
        img = Image.open(target); sz = img.size; img.close()
        return sz

    ok = True
    for name, share, secs in (("scanner_3s", SHARE_NONE, 3.0), ("viewer_15s", SHARE_READ_WRITE, 15.0),
                              ("scanner_15s", SHARE_NONE, 15.0)):
        target = os.path.join(d, "locked_%s.png" % name)
        Image.new("RGB", (8, 8), (255, 0, 255)).save(target)            # the "old" file: 8x8 magenta
        before = os.path.getmtime(target)
        ready, released = threading.Event(), []
        t = threading.Thread(target=hold, args=(target, share, secs, ready, released)); t.start(); ready.wait()
        rc, took, t1 = develop(target)
        t.join()
        if not released:
            print("FAIL: %s -> the hold never took" % name); ok = False; continue
        new = size_of(target) == (64, 48)
        leftovers = [f for f in os.listdir(d) if f.startswith("locked_%s.png." % name)]
        if name == "scanner_3s":
            # Finishing AFTER the release proves the write waited for the lock, rather than the
            # lock happening to be gone before ftrace got to the file.
            waited = t1 >= released[0]
            good = rc == 0 and new and not leftovers and waited
            what = "exit %d after %.1f s, %s, finished %s the release" % (
                rc, took, "new image" if new else "OLD image", "after" if waited else "BEFORE")
        elif name == "viewer_15s":
            good = rc == 0 and new and not leftovers and took < 5.0
            what = "exit %d after %.1f s, %s" % (rc, took, "new image" if new else "OLD image")
        else:
            good = (rc == 1 and not new and os.path.getmtime(target) == before
                    and not leftovers and took >= 9.5)
            what = "exit %d after %.1f s, the old file %s" % (rc, took, "kept" if not new else "REPLACED")
        print("%s: %s -> %s%s" % ("ok" if good else "FAIL", name, what,
                                  (", leftovers %s" % leftovers) if leftovers else ""))
        ok &= good

    # 4. read-only target
    target = os.path.join(d, "locked_readonly.png")
    Image.new("RGB", (8, 8), (255, 0, 255)).save(target)
    os.chmod(target, stat.S_IREAD)
    rc, took, _ = develop(target)
    os.chmod(target, stat.S_IWRITE)
    new = size_of(target) == (64, 48)
    leftovers = [f for f in os.listdir(d) if f.startswith("locked_readonly.png.")]
    good = rc == 1 and not new and not leftovers and took < 2.0
    print("%s: readonly -> exit %d after %.1f s, the old file %s%s" % (
        "ok" if good else "FAIL", rc, took, "kept" if not new else "REPLACED",
        (", leftovers %s" % leftovers) if leftovers else ""))
    ok &= good

    # 5. a held stale temporary
    target = os.path.join(d, "locked_heldtmp.png")
    Image.new("RGB", (8, 8), (255, 0, 255)).save(target)
    stale = target + ".tmp"
    with open(stale, "wb") as f:
        f.write(b"stale")
    ready, released = threading.Event(), []
    t = threading.Thread(target=hold, args=(stale, SHARE_NONE, 6.0, ready, released)); t.start(); ready.wait()
    rc, took, _ = develop(target)
    t.join()
    new = size_of(target) == (64, 48)
    others = sorted(f for f in os.listdir(d) if f.startswith("locked_heldtmp.png.") and f != "locked_heldtmp.png.tmp")
    good = rc == 0 and new and took < 3.0 and not others and open(stale, "rb").read() == b"stale"
    print("%s: heldtmp -> exit %d after %.1f s, %s%s" % (
        "ok" if good else "FAIL", rc, took, "new image" if new else "OLD image",
        (", leftovers %s" % others) if others else ""))
    ok &= good
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
