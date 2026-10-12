"""Regression check: an image write survives the output file being locked for a moment.

    D:/python314/python.exe -u tools/check_write_locked.py [--out <dir>] [--exe ftrace.exe]

On Windows a virus scanner or a viewer can hold a just-written image open for a moment, and a
render's next write to it (the periodic one, or the final one) used to fail outright:
`error: could not write <path>`. ftrace now writes a temporary file beside the target and moves it
into place, retrying while the target is held. Checked here through `-topng`, which writes through
the same function a render does:
  1. the target is opened with NO sharing (as a scanner may) and released after 1 s, while
     `-topng` writes to it: the write must succeed, and the file must be the new image;
  2. the target is held for longer than the retry window: the write must fail cleanly (exit 1, the
     old file untouched, no temporary file left behind).
Windows only (the lock is a Win32 share mode). Exits 1 on any failure.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import threading
import time

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
INVALID = ctypes.c_void_p(-1).value

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]


def hold(path, seconds, ready):
    """Open `path` with no sharing at all, signal `ready`, keep it `seconds`, close it."""
    h = k32.CreateFileW(path, GENERIC_READ, 0, None, OPEN_EXISTING, 0, None)
    if h in (None, INVALID):
        ready.set()
        raise OSError("CreateFileW failed: %d" % ctypes.get_last_error())
    ready.set()
    time.sleep(seconds)
    k32.CloseHandle(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_write_locked"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)          # CreateProcess wants a full path for a relative one
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
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

    ok = True
    for name, secs, expect_ok in (("short", 1.0, True), ("long", 12.0, False)):
        target = os.path.join(d, "locked_%s.png" % name)
        Image.new("RGB", (8, 8), (255, 0, 255)).save(target)            # the "old" file: 8x8 magenta
        before = os.path.getmtime(target)
        ready = threading.Event()
        t = threading.Thread(target=hold, args=(target, secs, ready)); t.start(); ready.wait()
        t0 = time.time()
        r = subprocess.run([a.exe, "-topng", ftbuf, target], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        took = time.time() - t0
        t.join()
        img = Image.open(target); size = img.size; img.close()
        leftovers = [f for f in os.listdir(d) if f.startswith("locked_%s.png." % name)]
        if expect_ok:
            good = r.returncode == 0 and size == (64, 48) and not leftovers
            print("%s: held 1 s -> exit %d after %.1f s, file is %dx%d%s" % (
                "ok" if good else "FAIL", r.returncode, took, size[0], size[1],
                (", leftovers %s" % leftovers) if leftovers else ""))
        else:
            good = r.returncode == 1 and size == (8, 8) and os.path.getmtime(target) == before and not leftovers
            print("%s: held 12 s -> exit %d after %.1f s, the old file %s%s" % (
                "ok" if good else "FAIL", r.returncode, took, "kept" if size == (8, 8) else "REPLACED",
                (", leftovers %s" % leftovers) if leftovers else ""))
        ok &= good
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
