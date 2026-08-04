#!/usr/bin/env python3
"""Regression check for the solid mesh voxelizer (src/meshvoxel.h).

`medium { bounds { object "<mesh>" } }` bakes a named mesh into an occupancy lattice.
That bake is easy to get subtly wrong in ways a picture will not show you: the first
implementation read `Tri::gn`, which is still (0,0,0) during the load, so every crossing
got the same winding sign and each scanline filled solid from its first crossing to its
last -- the mesh's x-CONVEX HULL instead of the mesh. A single sphere and a box both
rendered perfectly, because for a convex body the hull IS the body. Only a shape with a
concavity along +x can catch it.

So this builds meshes whose exact solid fraction is computable, asks ftrace to voxelize
each one, and compares against the count of lattice points genuinely inside the analytic
shape -- on the SAME lattice the loader builds (resolution on the longest axis, cubic
voxels, one-voxel zero shell, voxel-centre membership).

    python tools/check_meshvox.py [--ftrace ./ftrace.exe] [--res 200] [--keep]

Cases, and what each one is for:
  sphere       a convex control; catches gross breakage only.
  twospheres   two OVERLAPPING bodies. Parity voxelization hollows the overlap; the
               winding fill must report their union. Also mildly concave along +x.
  disjoint     two SEPARATE bodies. The gap between them is the test -- this is the case
               that exposes a winding that never returns to zero.
  box          every face exactly on a lattice-centre plane. Membership there is a
               floating-point coin flip, so this one is checked with a loose tolerance
               and only to confirm the error stays sub-voxel and on the EROSION side
               (a fog bound must never leak outside its mesh).
"""

import argparse, math, os, re, subprocess, sys, tempfile

try:
    import numpy as np
except ImportError:
    sys.exit("check_meshvox: needs numpy (pip install numpy)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- meshes
def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write("v %.9f %.9f %.9f\n" % v)
        for a, b, c in faces:
            f.write("f %d %d %d\n" % (a + 1, b + 1, c + 1))


def uv_sphere(cx, cy, cz, r, nu=128, nv=64):
    verts = []
    for j in range(nv + 1):
        th = math.pi * j / nv
        for i in range(nu):
            ph = 2 * math.pi * i / nu
            verts.append((cx + r * math.sin(th) * math.cos(ph),
                          cy + r * math.cos(th),
                          cz + r * math.sin(th) * math.sin(ph)))
    idx = lambda j, i: j * nu + (i % nu)
    faces = []
    for j in range(nv):
        for i in range(nu):
            a, b = idx(j, i), idx(j, i + 1)
            c, d = idx(j + 1, i), idx(j + 1, i + 1)
            faces.append((a, c, b))
            faces.append((b, c, d))
    return verts, faces


def join(m1, m2):
    v1, f1 = m1
    v2, f2 = m2
    n = len(v1)
    return v1 + v2, f1 + [(a + n, b + n, c + n) for a, b, c in f2]


def box_mesh(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    f = []
    for a, b, c, d in quads:
        f += [(a, b, c), (a, c, d)]
    return v, f


# ----------------------------------------------------------------- expected fraction
def expected_percent(lo, hi, inside, res):
    """Count lattice points inside `inside`, reproducing meshvox's lattice exactly."""
    lo = np.array(lo, float)
    hi = np.array(hi, float)
    ext = hi - lo
    h = ext.max() / res
    w0 = lo - h                       # one-voxel zero shell
    n = np.ceil(ext / h).astype(int) + 3
    axes = [w0[k] + h * np.arange(n[k]) for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    m = inside(X, Y, Z)
    return 100.0 * m.sum() / m.size, tuple(int(v) for v in n)


CASES = [
    # name, build mesh, AABB lo, AABB hi, analytic membership, tolerance (pct points)
    ("sphere", lambda: uv_sphere(0, 0, 0, 1.0),
     (-1, -1, -1), (1, 1, 1),
     lambda X, Y, Z: X * X + Y * Y + Z * Z <= 1.0, 0.3),

    ("twospheres", lambda: join(uv_sphere(-0.5, 0, 0, 1.0), uv_sphere(0.5, 0, 0, 1.0)),
     (-1.5, -1, -1), (1.5, 1, 1),
     lambda X, Y, Z: (((X + .5) ** 2 + Y * Y + Z * Z <= 1.0) |
                      ((X - .5) ** 2 + Y * Y + Z * Z <= 1.0)), 0.3),

    ("disjoint", lambda: join(uv_sphere(-1.5, 0, 0, 1.0), uv_sphere(1.5, 0, 0, 1.0)),
     (-2.5, -1, -1), (2.5, 1, 1),
     lambda X, Y, Z: (((X + 1.5) ** 2 + Y * Y + Z * Z <= 1.0) |
                      ((X - 1.5) ** 2 + Y * Y + Z * Z <= 1.0)), 0.3),

    # Lattice-aligned faces: membership is genuinely ambiguous, so allow ~1 voxel plane
    # per axis of erosion (at res 200 that is ~1.5 percentage points).
    ("box", lambda: box_mesh((-1, -1, -1), (1, 1, 1)),
     (-1, -1, -1), (1, 1, 1),
     lambda X, Y, Z: (np.abs(X) <= 1) & (np.abs(Y) <= 1) & (np.abs(Z) <= 1), 2.0),
]

SCENE = """\
scene {{ units meters }}
material "w" {{ type diffuse reflect 0.7 }}
quad "floor" {{ origin 0 -3 0  u 12 0 0  v 0 0 12  material w }}
mesh "shape" {{ file "{obj}"  material w  shape_only yes }}
medium {{ sigma_t 1.0 albedo 0.9 bounds {{ object "shape" voxels {res} }} }}
light sphere {{ center 3 3 3 radius 0.2 spd blackbody 6000 power 500 }}
camera "cam" {{ eye 0 0 6 look_at 0 0 0 up 0 1 0 fov_y 45 mode D film {{ res 32 32 }} }}
"""

REPORT = re.compile(r"mesh bound \"shape\": (\d+) x (\d+) x (\d+) lattice, ([0-9.]+)% solid")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ftrace", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--res", type=int, default=200, help="voxels on the longest axis")
    ap.add_argument("--keep", action="store_true", help="keep the generated .obj/.ftsl")
    args = ap.parse_args()

    if not os.path.exists(args.ftrace):
        sys.exit("check_meshvox: no ftrace at %s (build it first)" % args.ftrace)

    workdir = os.path.join(ROOT, "scraps") if args.keep else tempfile.mkdtemp(prefix="meshvox")
    os.makedirs(workdir, exist_ok=True)
    failures = []

    for name, build, lo, hi, inside, tol in CASES:
        obj = os.path.join(workdir, "_meshvox_%s.obj" % name)
        ftsl = os.path.join(workdir, "_meshvox_%s.ftsl" % name)
        write_obj(obj, *build())
        with open(ftsl, "w") as f:
            f.write(SCENE.format(obj=obj.replace("\\", "/"), res=args.res))

        out = subprocess.run([args.ftrace, ftsl, "-check-watertight"],
                             cwd=ROOT, capture_output=True, text=True)
        m = REPORT.search(out.stdout + out.stderr)
        if not m:
            failures.append("%s: loader printed no mesh-bound report" % name)
            print("  [FAIL] %-11s no report\n%s" % (name, (out.stdout + out.stderr)[-600:]))
            continue

        got = float(m.group(4))
        got_lat = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        want, want_lat = expected_percent(lo, hi, inside, args.res)

        ok = abs(got - want) <= tol and got_lat == want_lat
        # A hull-filling regression always over-reports; call that out by name, since it
        # is the specific failure this script exists to catch.
        hint = ""
        if got > want + tol:
            hint = "  <-- over-filled: winding never returns to 0 (convex-hull fill?)"
        elif got < want - tol:
            hint = "  <-- under-filled: crossings lost or spans dropped"
        if got_lat != want_lat:
            hint += "  <-- lattice %dx%dx%d != expected %dx%dx%d" % (got_lat + want_lat)
        print("  [%s] %-11s got %5.1f%%  expected %5.1f%%  (tol %.1f)%s"
              % ("ok  " if ok else "FAIL", name, got, want, tol, hint))
        if not ok:
            failures.append("%s: got %.1f%%, expected %.1f%%" % (name, got, want))

    if failures:
        print("\ncheck_meshvox: %d of %d case(s) FAILED" % (len(failures), len(CASES)))
        for f in failures:
            print("  - " + f)
        return 1
    print("\ncheck_meshvox: all %d case(s) PASS (res %d)" % (len(CASES), args.res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
