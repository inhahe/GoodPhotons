"""Bit-identity check: render a fixed set of scenes with two ftrace builds and compare the HDR
output byte for byte -- the proof that a change meant to alter nothing alters nothing.

    python tools/check_bitid.py run --before <old ftrace.exe> [--after ftrace.exe] [--tol <rel>]
    python tools/check_bitid.py render <label> [--exe ftrace.exe]
    python tools/check_bitid.py compare <labelA> <labelB> [--tol <rel>]
    (all take [--out png/check_bitid])

`run` renders every case with both executables and compares them; build.bat replaces
ftrace.exe, so copy the old binary aside before rebuilding (e.g. to scraps/ftrace_before.exe).
`render` and `compare` do the halves, for a change that is easier to stage that way. Under the
first executable `run` renders every case twice: a case whose two renders differ is not
deterministic and proves nothing, and is reported so.

The cases are scenes this script writes into <out>/scenes, each rendered on the CPU and the GPU
in modes R and D, B (forward), J and U (the bidirectional merges), M and S (the photon maps), at
a few samples:
  flat     quads and analytic spheres -- diffuse, glossy, mirror, glass, translucent, a mix --
           whose shading and geometric normals coincide;
  mapped   a normal-mapped wall and a normal-mapped translucent panel, lit at a grazing angle;
  smooth   a smooth-normalled sphere mesh, and the same tilted and squashed to a tenth of its
           height by mesh_instance (its interpolated normals turned across its faces' planes),
           lit from above and from below;
  shapes   curve tubes and an isosurface under a pattern.

`compare` prints, per case, "identical", or how it differs: the pixels that changed, the
largest change relative to the pixel, and the change of the image's mean. --tol <rel> passes
a GPU case whose every pixel is within that relative tolerance: a GPU kernel the change only
re-compiled can differ in the last bits (instruction scheduling, fused multiply-adds) -- 1e-3,
say. The CPU cases are always held to bit identity. GPU modes M and S are not bit-deterministic
at all (atomic deposits), so they are always judged at a floor of 1e-4 (NONDET_FLOOR), the
determinism pass included. A change meant to alter some renders shows up here as differing
cases, with the mean change to judge them by. Exits 0 when every case is identical (or within
its tolerance), 1 otherwise.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
MODES = [  # (tag, ftrace arguments)
    ("R_cpu", ["-mode", "R", "-spp", "4", "-device", "cpu"]),
    ("R_gpu", ["-mode", "R", "-spp", "16", "-device", "gpu"]),
    ("D_cpu", ["-mode", "D", "-spp", "2", "-device", "cpu"]),
    ("D_gpu", ["-mode", "D", "-spp", "8", "-device", "gpu"]),
    ("B_cpu", ["-mode", "B", "-n", "200000", "-device", "cpu"]),
    ("B_gpu", ["-mode", "B", "-n", "1000000", "-device", "gpu"]),
    ("J_cpu", ["-mode", "J", "-spp", "1", "-device", "cpu"]),
    ("J_gpu", ["-mode", "J", "-spp", "2", "-device", "gpu"]),
    ("U_cpu", ["-mode", "U", "-spp", "2", "-device", "cpu"]),
    ("U_gpu", ["-mode", "U", "-spp", "4", "-device", "gpu"]),
    ("M_cpu", ["-mode", "M", "-n", "200000", "-spp", "2", "-device", "cpu"]),
    ("M_gpu", ["-mode", "M", "-n", "200000", "-spp", "2", "-device", "gpu"]),
    ("S_cpu", ["-mode", "S", "-spp", "2", "-device", "cpu"]),
    ("S_gpu", ["-mode", "S", "-spp", "2", "-device", "gpu"]),
]
# GPU modes M and S deposit their photons through atomics, so their order -- and the last bits of
# every sum over them -- varies from run to run: one build differs from itself there by up to a
# few 1e-5 of a pixel. Those cases are judged at this relative floor, in the determinism pass too.
NONDET_FLOOR = {"M_gpu": 1e-4, "S_gpu": 1e-4}


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def uv_sphere(path, nu=24, nv=16, tilt=0.0):
    """A unit sphere with outward vertex normals, tilted about x by `tilt` radians."""
    c, s = np.cos(tilt), np.sin(tilt)
    out = []
    for j in range(nv + 1):
        for i in range(nu):
            th, ph = np.pi * j / nv, 2 * np.pi * i / nu
            x, y0, z0 = np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)
            y, z = c * y0 - s * z0, s * y0 + c * z0
            out += ["v %.17g %.17g %.17g" % (x, y, z), "vn %.17g %.17g %.17g" % (x, y, z)]

    def idx(j, i):
        return j * nu + (i % nu) + 1
    for j in range(nv):
        for i in range(nu):
            a, b, cc, d = idx(j, i), idx(j, i + 1), idx(j + 1, i + 1), idx(j + 1, i)
            if j > 0:
                out.append("f %d//%d %d//%d %d//%d" % (a, a, cc, cc, b, b))
            if j < nv - 1:
                out.append("f %d//%d %d//%d %d//%d" % (a, a, d, d, cc, cc))
    write(path, NL.join(out) + NL)


def ripple_map(path, w=64):
    """A tangent-space normal map of ripples along u, as a binary PPM."""
    u = (np.arange(w) + 0.5) / w
    slope = 0.6 * np.sin(2 * np.pi * 4 * u)
    n = np.stack([slope, np.zeros(w), np.ones(w)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    rgb = np.round((n * 0.5 + 0.5) * 255).astype(np.uint8)
    img = np.broadcast_to(rgb[None, :, :], (w, w, 3))
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, w))
        f.write(np.ascontiguousarray(img).tobytes())


def scenes(out):
    d = os.path.join(out, "scenes")
    os.makedirs(d, exist_ok=True)
    p = d.replace(chr(92), "/")
    uv_sphere(os.path.join(d, "sphere.obj"))
    uv_sphere(os.path.join(d, "tilted.obj"), tilt=0.7)
    ripple_map(os.path.join(d, "ripple.ppm"))
    cam = 'camera "c" { eye 0 1.2 4  look_at 0 0.6 0  up 0 1 0  fov_y 45  film { res 96 72 } }\n'
    room = ('quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }\n'
            'quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }\n'
            'quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material red }\n'
            'quad { origin 2 0 -2  u 0 0 4  v 0 3 0  material green }\n')
    mats = ('material "white" { type diffuse reflect 0.7 }\n'
            'material "red"   { type diffuse reflect rgb 0.7 0.1 0.1 }\n'
            'material "green" { type diffuse reflect rgb 0.1 0.7 0.1 }\n')
    lamp = 'light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }\n'
    text = {
        "flat": "scene { units meters }\n" + mats + (
            'material "gloss" { type glossy reflect 0.8  roughness 0.2 }\n'
            'material "mirror" { type mirror reflect 0.9 }\n'
            'material "glass" { type dielectric ior glass:BK7 }\n'
            'material "trans" { type translucent reflect 0.3  transmit 0.4 }\n'
            'material "mixed" { type mix  layer "white" 0.5  layer "gloss" 0.5 }\n'
            'sphere { center -1 0.4 -0.5  radius 0.4  material gloss }\n'
            'sphere { center 0 0.4 0  radius 0.4  material glass }\n'
            'sphere { center 1 0.4 -0.5  radius 0.4  material mirror }\n'
            'sphere { center 0.6 0.25 0.8  radius 0.25  material mixed }\n'
            'quad { origin -1.4 0 0.6  u 0.6 0 0  v 0 1 0  material trans }\n') + room + lamp + cam,
        "mapped": "scene { units meters }\n" + mats + (
            'texture "rip" { file "%s/ripple.ppm"  encoding linear  filter bilinear  wrap repeat }\n'
            'material "wall" { type diffuse reflect 0.7  normal_map texture:rip }\n'
            'material "panel" { type translucent reflect 0.3  transmit 0.4  normal_map texture:rip }\n'
            'quad { origin -1.5 0.1 -1.9  u 3 0 0  v 0 2.5 0  material wall }\n'
            'quad { origin -0.5 0 0  u 1 0 0.3  v 0 1.2 0  material panel }\n'
            'light area { origin -1.9 0.2 -1.5  u 0 0 0.6  v 0 2.4 0  normal 1 0 0.2  spd preset:bb6500  power 40 }\n'
            % p) + room + cam,
        "smooth": "scene { units meters }\n" + mats + (
            'material "clay" { type diffuse reflect 0.7 }\n'
            'material "gloss" { type glossy reflect 0.8  roughness 0.3 }\n'
            'mesh_asset "ball" { file "%s/sphere.obj"  material gloss }\n'
            'mesh_asset "dome" { file "%s/tilted.obj"  material clay }\n'
            'mesh_instance { of "ball"  scale 0.45  translate -0.8 0.45 0 }\n'
            'mesh_instance { of "dome"  scale 0.9 0.09 0.9  translate 0.7 0.8 0 }\n'
            'quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }\n'
            'light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }\n'
            'light area { origin 0.4 0.3 -0.3  u 0 0 0.6  v 0.6 0 0  normal 0 1 0  spd preset:bb6500  power 20 }\n'
            % (p, p)) + cam,
        "shapes": "scene { units meters }\n" + mats + (
            'pattern "stripes" { expr "step(0.5, fract(4*y))" }\n'
            'material "tube" { type diffuse reflect rgb 0.8 0.6 0.2 }\n'
            'material "iso" { type mix  layer "white" 0.5  layer "red" 0.5  weight_map pattern:stripes }\n'
            'curve { material tube  basis linear  radius 0.06  point -1.2 0.1 0  point -0.6 1.2 -0.4  point 0 0.3 -0.8 }\n'
            'curve { material tube  basis linear  radius 0.04  point 0.2 0.1 0.6  point 0.8 1.0 0.2 }\n'
            'isosurface { material iso  sphere { center 0.9 0.5 -0.6  radius 0.45 } }\n') + room + lamp + cam,
    }
    paths = {}
    for name, t in text.items():
        paths[name] = os.path.join(d, name + ".ftsl")
        write(paths[name], t)
    return paths


def render_all(exe, out, label, paths, log):
    dest = os.path.join(out, label)
    os.makedirs(dest, exist_ok=True)
    ok = True
    for name, scene in paths.items():
        for tag, args in MODES:
            png = os.path.join(dest, "%s_%s.png" % (name, tag))
            for stale in (png, png[:-4] + ".pfm"):
                if os.path.exists(stale):
                    os.remove(stale)
            cmd = [exe, "-in", scene, "-hdr", "-no-meter", "-seed", "7", "-o", png] + args
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0 or not os.path.exists(png[:-4] + ".pfm"):
                log("  %s %s: render failed (exit %d)%s%s" % (name, tag, r.returncode, NL,
                                                             (r.stdout + r.stderr)[-500:]))
                ok = False
    return ok


def read_pfm(path):
    with open(path, "rb") as f:
        ch = 3 if f.readline().strip() == b"PF" else 1
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        return np.fromfile(f, "<f4" if scale < 0 else ">f4").reshape(h, w, ch)


def compare(out, a, b, tol, log):
    da, db = os.path.join(out, a), os.path.join(out, b)
    names = sorted(f for f in os.listdir(da) if f.endswith(".pfm"))
    if not names:
        log("  nothing rendered under %s" % a)
        return False
    ok = True
    for n in names:
        pa, pb = os.path.join(da, n), os.path.join(db, n)
        if not os.path.exists(pb):
            log("  %-22s missing under %s FAIL" % (n[:-4], b))
            ok = False
            continue
        if open(pa, "rb").read() == open(pb, "rb").read():
            log("  %-22s identical" % n[:-4])
            continue
        x, y = read_pfm(pa).astype(np.float64), read_pfm(pb).astype(np.float64)
        if x.shape != y.shape:
            log("  %-22s differs in size FAIL" % n[:-4])
            ok = False
            continue
        px = np.any(x != y, axis=-1)
        rel = float((np.abs(y - x) / np.maximum(np.abs(x), 1e-6)).max())
        mean = (y.sum() - x.sum()) / max(abs(x.sum()), 1e-30)
        tag = "_".join(n[:-4].rsplit("_", 2)[-2:])         # "flat_M_gpu.pfm" -> "M_gpu"
        allowed = max(tol if tag.endswith("_gpu") else 0.0, NONDET_FLOOR.get(tag, 0.0))
        if rel <= allowed:
            log("  %-22s within %.0e: %d of %d pixels, largest change %.3g of the pixel"
                % (n[:-4], allowed, int(px.sum()), px.size, rel))
            continue
        ok = False
        log("  %-22s differs: %d of %d pixels, largest change %.3g of the pixel, mean %+.3f%% FAIL"
            % (n[:-4], int(px.sum()), px.size, rel, 100.0 * mean))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("what", choices=["run", "render", "compare"])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_bitid"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--before")
    ap.add_argument("--after", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    def log(s):
        print(s, flush=True)
    paths = scenes(out)
    if a.what == "render":
        if len(a.labels) != 1:
            ap.error("render takes one label")
        ok = render_all(os.path.abspath(a.exe), out, a.labels[0], paths, log)
    elif a.what == "compare":
        if len(a.labels) != 2:
            ap.error("compare takes two labels")
        ok = compare(out, a.labels[0], a.labels[1], a.tol, log)
    else:
        if not a.before:
            ap.error("run needs --before <the old ftrace.exe>")
        ok = render_all(os.path.abspath(a.before), out, "before", paths, log)
        ok &= render_all(os.path.abspath(a.before), out, "before_again", paths, log)
        log("determinism (the old build against itself):")
        det = compare(out, "before", "before_again", 0.0, log)
        if not det:
            log("  a case above is not deterministic: its comparison below proves nothing")
        ok &= render_all(os.path.abspath(a.after), out, "after", paths, log)
        log("the new build against the old:")
        ok &= compare(out, "before", "after", a.tol, log) and det
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
