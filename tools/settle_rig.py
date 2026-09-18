"""The hair settle (`settle { }`), measured on the groom it was built for.

    python tools/settle_rig.py [--keep] [--iters N]

known-issues HAIR-PENETRATION established the target with a control: 72.67 % of Alice's 1.2 M
segments lie inside another strand, against 0.97 % for `fur` scattered at the SAME 20 000 strands,
so the cause is the curve-of-curves blend rather than density. This rig checks that the load-time
relaxation actually removes that, and -- the part that matters more -- that it does not buy the
number by wrecking the groom.

  0. RIG SEES THE SWITCH -- `settle { iterations 0 }` must leave the strands BYTE-IDENTICAL to no
                            settle block at all. Without this, every number below could be the
                            scene reloading differently rather than the solver doing anything, and
                            this file's own history is full of exactly that trap.
  1. DETERMINISTIC        -- two runs of the same settle must be byte-identical. The CPU and CUDA
                            backends have to trace the same geometry, and a flyby re-loads the
                            scene once per frame; a solver that drifts is unusable regardless of
                            how good one frame looks. (This is why the separation pass is
                            one-sided: no atomics, no thread-count dependence.)
  2. PENETRATION FALLS    -- the headline. Measured with tools/hair_penetration.py, the same
                            instrument and the same sampling as the entry that set the target.
  3. THE GROOM SURVIVES   -- mean particle displacement must stay small against the strand length.
                            A solver that "fixes" penetration by collapsing the hair onto the
                            scalp would pass check 2 and be worthless; this is the check that
                            makes check 2 mean something.
"""
import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join("scraps", "_settlerig")
EXE = os.path.join(ROOT, "ftrace.exe")
BASE = "alice/alice_real_hair.ftsl"
TEMP = []          # scenes written into alice/, removed on the way out


def scene(tag, settle_body):
    """alice_real_hair.ftsl plus (optionally) one settle block.

    Written INSIDE alice/, not into scraps/: `include` and the model's own
    `file "assets/alice_segmented.glb"` both resolve relative to the including file, so a scene
    living anywhere else is one path rule away from silently loading a different groom -- and a
    rig that measures a different scene than it thinks is worse than no rig.
    """
    p = os.path.join("alice", "_rig_" + tag + ".ftsl")
    txt = 'include "alice_real_hair.ftsl"\n'
    if settle_body is not None:
        txt += "settle {\n%s\n}\n" % settle_body
    io.open(p, "w", encoding="utf-8", newline="\n").write(txt)
    TEMP.append(p)
    return p


def dump(tag, settle_body):
    sc = scene(tag, settle_body)
    out = os.path.join(OUT, tag + ".txt")
    r = subprocess.run([EXE, "-in", sc, "-dumpcurves", out], capture_output=True, text=True, check=False)
    if not os.path.exists(out):
        raise RuntimeError("dump failed for %s:\n%s" % (tag, (r.stdout + r.stderr)[-2000:]))
    line = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("[settle]")]
    return out, (line[-1] if line else "")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def displacement(a, b):
    """Mean / max particle displacement between two dumps, and the strand length, in mm."""
    import numpy as np

    def pts(p):
        v = []
        for ln in io.open(p, encoding="utf-8"):
            if ln[0] in "#s":
                continue
            w = ln.split()
            if len(w) == 4:
                v.append((float(w[0]), float(w[1]), float(w[2])))
        return np.asarray(v)
    A, B = pts(a), pts(b)
    n = min(len(A), len(B))
    d = np.linalg.norm(A[:n] - B[:n], axis=1)
    seg = np.linalg.norm(A[1:n] - A[:n - 1], axis=1)
    return 1e3 * float(d.mean()), 1e3 * float(d.max()), 1e3 * float(np.median(seg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--stiff", type=float, default=0.20)
    ap.add_argument("--stiff-tip", type=float, default=0.04)
    ap.add_argument("--droop", type=float, default=0.5)
    ap.add_argument("--sep", type=float, default=1.0)
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print("   %-16s %s%s" % (name, "PASS" if cond else "FAIL  <--", ("   " + detail) if detail else ""))

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    from hair_penetration import measure

    body = ("    iterations %d\n    stiffness %g\n    stiffness_tip %g\n    droop %g\n    separation %g"
            % (args.iters, args.stiff, args.stiff_tip, args.droop, args.sep))

    print("  loading Alice three times (no settle / zero sweeps / %d sweeps) ..." % args.iters)
    sys.stdout.flush()
    fBase, _ = dump("base", None)
    fZero, logZero = dump("zero", "    iterations 0")
    fOn, logOn = dump("on", body)
    fOn2, _ = dump("on2", body)
    if logOn:
        print("  " + logOn)

    check("rig sees switch", sha(fBase) == sha(fZero),
          "settle{iterations 0} == no settle block (%s vs %s)" % (sha(fBase), sha(fZero)))
    check("deterministic", sha(fOn) == sha(fOn2),
          "two identical settles agree byte-for-byte (%s)" % sha(fOn))

    before = measure(fBase, verbose=False)
    after = measure(fOn, verbose=False)
    mean, mx, seg = displacement(fBase, fOn)
    print("\n  penetrating segments : %.2f %%  ->  %.2f %%   (a BINARY threshold: a pair 1 um apart counts the same as one fully merged)" % (before["pen_pct"], after["pen_pct"]))
    print("  median overlap depth : %.1f um  ->  %.1f um" % (before["median_pen_um"], after["median_pen_um"]))
    print("  median clearance     : %+.3f mm  ->  %+.3f mm" % (before["median_clear_mm"], after["median_clear_mm"]))
    print("  the groom moved      : mean %.3f mm, max %.3f mm (segment %.3f mm)" % (mean, mx, seg))

    # The headline stays the binary count, and it stays a FAIL while it is one. Softening a
    # check until the thing passes is the exact failure mode this project has a rule about;
    # the depth check below is reported ALONGSIDE it, never instead of it.
    check("penetration falls", after["pen_pct"] < before["pen_pct"] * 0.5,
          "%.2f %% -> %.2f %% -- reduces overlap, does not remove it" % (before["pen_pct"], after["pen_pct"]))
    check("overlap shrinks", after["median_pen_um"] < before["median_pen_um"] * 0.9,
          "median depth %.1f -> %.1f um, clearance %+.3f -> %+.3f mm"
          % (before["median_pen_um"], after["median_pen_um"],
             before["median_clear_mm"], after["median_clear_mm"]))
    check("groom survives", mean < 2.0 * seg,
          "mean displacement %.3f mm against a %.3f mm segment -- a collapse onto the scalp would"
          " pass the check above and be worthless" % (mean, seg))

    for t in TEMP:                      # the rig leaves nothing in alice/
        try:
            os.remove(t)
        except OSError:
            pass
    print("\n-> settle rig %s" % ("PASSED" if ok else "FAILED"))
    if not args.keep and ok:
        shutil.rmtree(OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
