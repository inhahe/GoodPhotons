"""`emit_orient` rig: an emissive ENCLOSURE must be able to glow inward (TODO item 3, 0.338.0).

    python tools/emit_orient_rig.py [--keep]

The scene is a closed emissive cube (an OBJ written here, wound so its faces point INWARD -- a
lampshade interior) with a grey probe quad inside it, viewed from inside. Four checks:

  1. AUTO IS DARK      -- the default `auto` reorients the shell outward (its own diagnosis says
                          so on stderr), so the interior is unlit: the probe is near black.
  2. KEEP LIGHTS IT    -- `emit_orient keep` leaves the winding alone and the probe is lit.
  3. KEEP == 6 QUADS   -- `keep` matches the workaround the furnace rig uses (one `mesh` block per
                          planar face, which the volume test never touches): same probe radiance
                          to within the renders' own noise.
  4. FLIP / ERROR      -- `flip` on an OUTWARD-wound cube reproduces case 2, and a misspelled
                          value is refused by name rather than silently ignored.

Cited by known-issues ("An emissive mesh that is not PLANAR is silently re-oriented outward").
"""
import argparse
import io
import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

OUT = os.path.join("scraps", "_emitorient")
EXE = os.path.join(ROOT, "ftrace.exe")

# A unit cube centred at the origin, side 2. `inward=True` winds every face toward the interior.
CUBE_V = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
          (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
# faces as outward-wound quads (CCW seen from outside)
CUBE_F = [(1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]


def write_cube(path, inward):
    L = ["# emit_orient rig: closed cube, %s-wound\n" % ("inward" if inward else "outward")]
    for v in CUBE_V:
        L.append("v %g %g %g\n" % v)
    for f in CUBE_F:
        a, b, c, d = f
        if inward:
            a, b, c, d = d, c, b, a
        L.append("f %d %d %d\n" % (a, b, c))
        L.append("f %d %d %d\n" % (a, c, d))
    io.open(path, "w", encoding="utf-8", newline="\n").writelines(L)


# Six separate PLANAR mesh blocks: the volume test never touches a planar mesh, so this is the
# workaround the furnace rig uses and the reference case 3 compares against.
def write_faces(dirpath, inward):
    names = []
    for i, f in enumerate(CUBE_F):
        p = os.path.join(dirpath, "face%d.obj" % i)
        idx = {}
        L = []
        a, b, c, d = f
        quad = (d, c, b, a) if inward else (a, b, c, d)
        for j, vi in enumerate(quad):
            idx[vi] = j + 1
            L.append("v %g %g %g\n" % CUBE_V[vi - 1])
        L.append("f 1 2 3\n")
        L.append("f 1 3 4\n")
        io.open(p, "w", encoding="utf-8", newline="\n").writelines(L)
        names.append(p.replace("\\", "/"))
    return names


HEAD = """camera "cam" { eye 0 0 0.5  look_at 0 0 -1  up 0 1 0  fov_y 60  mode R  film { res 64 64 } }
material "emit" { type diffuse  reflect 0.0  emit blackbody 6504 intensity 3e-15 }
material "probe" { type diffuse  reflect 0.6 }
quad { origin -0.4 -0.4 -0.6  u 0.8 0 0  v 0 0.8 0  material probe }
"""


def scene(path, body):
    io.open(path, "w", encoding="utf-8", newline="\n").write(HEAD + body)
    return path


def render(scene_path, tag):
    out = os.path.join(OUT, tag + ".png")
    r = subprocess.run([EXE, "-in", scene_path, "-mode", "R", "-spp", "64", "-r", "64", "64",
                        "-hdr", "-o", out, "-window-min", "-interval", "60"],
                       capture_output=True, text=True, check=False)
    return out[:-4] + ".pfm", r.stdout + r.stderr


def probe_mean(pfm):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from grade_hdr import read_pfm
    import numpy as np
    a = read_pfm(pfm)
    h, w = a.shape[:2]
    return float(np.asarray(a)[h // 3:2 * h // 3, w // 3:2 * w // 3].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print("   %-12s %s%s" % (name, "PASS" if cond else "FAIL  <--", ("   " + detail) if detail else ""))

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    cube_in = os.path.join(OUT, "cube_in.obj")
    cube_out = os.path.join(OUT, "cube_out.obj")
    write_cube(cube_in, inward=True)
    write_cube(cube_out, inward=False)
    faces = write_faces(OUT, inward=True)

    mesh = 'mesh "shell" { file "%s"  material emit%s }\n'
    p_auto = scene(os.path.join(OUT, "auto.ftsl"), mesh % (cube_in.replace("\\", "/"), ""))
    p_keep = scene(os.path.join(OUT, "keep.ftsl"), mesh % (cube_in.replace("\\", "/"), "  emit_orient keep"))
    p_flip = scene(os.path.join(OUT, "flip.ftsl"), mesh % (cube_out.replace("\\", "/"), "  emit_orient flip"))
    p_six = scene(os.path.join(OUT, "six.ftsl"),
                  "".join('mesh "f%d" { file "%s"  material emit }\n' % (i, f) for i, f in enumerate(faces)))
    p_bad = scene(os.path.join(OUT, "bad.ftsl"), mesh % (cube_in.replace("\\", "/"), "  emit_orient sideways"))

    pf_auto, log_auto = render(p_auto, "auto")
    pf_keep, _ = render(p_keep, "keep")
    pf_flip, _ = render(p_flip, "flip")
    pf_six, log_six = render(p_six, "six")

    m_auto, m_keep, m_flip, m_six = (probe_mean(p) for p in (pf_auto, pf_keep, pf_flip, pf_six))
    check("auto dark", m_auto < 0.02 * m_keep, "probe %.4g vs keep's %.4g" % (m_auto, m_keep))
    check("auto warns", "emit_orient keep" in log_auto, "the flip announces itself and names the opt-out")
    check("keep lit", m_keep > 1e-6, "probe %.4g" % m_keep)
    rel = abs(m_keep - m_six) / max(m_six, 1e-30)
    check("keep==6quads", rel < 0.02, "keep %.5g vs six planar meshes %.5g (%.2f%%)" % (m_keep, m_six, 100 * rel))
    relf = abs(m_flip - m_keep) / max(m_keep, 1e-30)
    check("flip", relf < 0.02, "flip on an outward cube %.5g == keep %.5g (%.2f%%)" % (m_flip, m_keep, 100 * relf))
    r = subprocess.run([EXE, "-in", p_bad, "-mode", "R", "-spp", "1", "-r", "8", "8",
                        "-o", os.path.join(OUT, "bad.png")], capture_output=True, text=True, check=False)
    msg = r.stdout + r.stderr
    check("bad value", r.returncode != 0 and "emit_orient" in msg, msg.strip().splitlines()[-1][:80] if msg.strip() else "no message")

    print("\n-> emit_orient rig %s" % ("PASSED" if ok else "FAILED"))
    if not args.keep and ok:
        shutil.rmtree(OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
