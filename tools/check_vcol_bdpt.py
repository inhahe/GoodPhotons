"""Regression check: the device's BDPT-family modes shade per-hit reflectance as the host does.

    python tools/check_vcol_bdpt.py [--out png/check_vcol_bdpt] [--exe ftrace.exe]

A sphere whose PLY vertices carry a hue ramp, under an area light on black. Modes D, J and U
rebuild a hit from each stored path vertex for their connections (dVertHit), and until 0.433.4 the
device's vertex record (DVertex, DVcmLV) did not carry the hit's vertex colour, so every connection
on the GPU shaded the mesh uncoloured while its own walk shaded it in colour (known-issues
VCOL-BDPT-GPU). The host's vertex keeps the whole hit, so the CPU was right.

Each mode on the GPU, and mode D on the CPU, must read the sphere's mean colour within 5 % of mode R
(GPU) per channel -- of the channel, or of a tenth of the three together where it is smaller -- and
its mean saturation within 5 % too. Two seeds. On 0.433.3 the GPU's read the sphere near grey
(saturation 0.06-0.08 of mode R's in modes D and U).

And a grating floor whose reflectance ramps from 0.2 to 1 along it (`reflect [0.2 1](u)`), mirroring
a lit wall: GPU modes D and J, and CPU mode D, must read its near and far halves within 5 % of mode
R (GPU). Until 0.433.8 the device's walk for modes D and J weighted a grating by its bare reflect
spectrum, not its slot (pattern, texture, record, vertex colour), and read the halves at 2.63 and
1.30 of mode R. Exits 0 when every case holds, 1 otherwise.
"""
import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_motion as cm  # noqa: E402

NL = chr(10)


def hue(t):
    """An RGB hue ramp, t in [0, 1)."""
    k = (np.array([5.0, 3.0, 1.0]) + t * 6.0) % 6.0
    return np.clip(1.0 - np.maximum(0.0, np.minimum(np.minimum(k, 4.0 - k), 1.0)), 0, 1) * 0.8 + 0.1


def sphere_ply(path, r=0.5, nu=48, nv=24):
    verts, faces = [], []
    for j in range(nv + 1):
        th = math.pi * j / nv
        for i in range(nu):
            ph = 2 * math.pi * i / nu
            c = (hue(i / nu) * 255).astype(int)
            verts.append("%.6f %.6f %.6f %d %d %d" % (r * math.sin(th) * math.cos(ph), r * math.cos(th),
                                                      r * math.sin(th) * math.sin(ph), c[0], c[1], c[2]))
    for j in range(nv):
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            c, d = a + nu, b + nu
            faces += ["3 %d %d %d" % (a, c, b), "3 %d %d %d" % (b, c, d)]
    with open(path, "w", encoding="ascii", newline="") as f:
        f.write(NL.join(["ply", "format ascii 1.0", "element vertex %d" % len(verts),
                         "property float x", "property float y", "property float z",
                         "property uchar red", "property uchar green", "property uchar blue",
                         "element face %d" % len(faces), "property list uchar int vertex_indices",
                         "end_header"] + verts + faces) + NL)


REF = ["-mode", "R", "-device", "gpu", "-spp", "1024"]
CASES = [("D_gpu", ["-mode", "D", "-device", "gpu", "-spp", "256"]),
         # without its surface merges, which on a sphere this small are too noisy at a check's budget
         # to hold to 5 % (CPU 64 spp 0.83 / 0.96 / 1.16, GPU 1024 spp 1.00 / 0.99 / 0.95 of mode R,
         # each off its own way; check_rough_irid leaves mode J out for the same reason): its
         # connections are what rebuild a hit from a vertex
         ("J_gpu", ["-mode", "J", "-device", "gpu", "-spp", "256", "-nojsurf"]),
         ("U_gpu", ["-mode", "U", "-device", "gpu", "-spp", "256"]),
         ("D_cpu", ["-mode", "D", "-device", "cpu", "-spp", "64"])]
TOL = 0.05

# The grating floor (see the docstring), and where on the film it is seen: (eye, look_at, fov_y).
G_W, G_H = 96, 72
G_CAM = ((0.0, 0.5, -1.6), (0.0, 0.0, 0.2), 40.0)
G_CASES = [("D_gpu", ["-mode", "D", "-device", "gpu", "-spp", "256"]),
           ("J_gpu", ["-mode", "J", "-device", "gpu", "-spp", "256", "-nojsurf"]),
           ("D_cpu", ["-mode", "D", "-device", "cpu", "-spp", "64"])]


def grating_scene():
    (ex, ey, ez), (lx, ly, lz), fov = G_CAM
    return NL.join([
        "scene { units meters }",
        'camera "c" { eye %g %g %g  look_at %g %g %g  up 0 1 0  fov_y %g  film { res %d %d } }' % (
            ex, ey, ez, lx, ly, lz, fov, G_W, G_H),
        'material "white" { type diffuse reflect 0.8 }',
        'material "gr" { type grating  reflect [0.2 1](u)  groove_spacing 1600 }',
        'quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material gr }',
        'quad { origin -1 0 1  u 0 1.5 0  v 2 0 0  material white }',
        'light area { origin -0.4 1.4 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  power 100  spd preset:d65 }',
    ]) + NL


def grating_halves():
    """The film's view of the grating's near half (z < -0.1) and far half (z > 0.1), away from its
    edges: a ray cast of the camera onto the floor."""
    eye, look, fov = np.array(G_CAM[0]), np.array(G_CAM[1]), G_CAM[2]
    f = (look - eye) / np.linalg.norm(look - eye)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    th = math.tan(math.radians(fov) / 2)
    z = np.full((G_H, G_W), np.nan)
    for y in range(G_H):
        for x in range(G_W):
            d = f + (2 * (x + 0.5) / G_W - 1) * th * (G_W / G_H) * r + (1 - 2 * (y + 0.5) / G_H) * th * u
            if d[1] >= 0:
                continue
            p = eye + d * (-eye[1] / d[1])
            if abs(p[0]) < 0.85 and abs(p[2]) < 0.85:
                z[y, x] = p[2]
    return z < -0.1, z > 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_vcol_bdpt"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ply = os.path.join(out, "ramp.ply")
    sphere_ply(ply)
    path = os.path.join(out, "vcol.ftsl")
    cm.write(path, NL.join([
        "scene { units meters }",
        'camera "c" { eye 0 0.4 2.2  look_at 0 0 0  up 0 1 0  fov_y 32  film { res 96 72 } }',
        'material "white" { type diffuse reflect 0.9 }',
        'mesh { file "%s"  material white }' % ply.replace(chr(92), "/"),
        "light area { origin -0.6 1.6 -0.2  u 1.2 0 0  v 0 0 1.2  normal 0 -1 0  power 60  spd preset:d65 }",
    ]) + NL)

    def image(tag, args, gpu):
        return np.mean([cm.render(exe, out, path, "%s_%d" % (tag, s), args, seed=s, want_gpu=gpu)[0]
                        for s in (1, 2)], axis=0)

    R = image("R_gpu", REF, True)
    lum = cm.lum(R)
    m = lum > 0.05 * lum.max()                       # the lit sphere, on black
    rc = R[m].mean(axis=0)
    rs = (R[m].max(axis=-1) - R[m].min(axis=-1)).mean()
    ok = True
    for name, args in CASES:
        X = image(name, args, "gpu" in name)
        xc = X[m].mean(axis=0)
        xs = (X[m].max(axis=-1) - X[m].min(axis=-1)).mean()
        floor = 0.1 * rc.sum()
        good = all(abs(x - r) <= TOL * max(abs(r), floor) for x, r in zip(xc, rc)) and abs(xs / rs - 1.0) <= TOL
        ok &= good
        print("%-6s sphere colour, of mode R: %s  saturation %.3f  %s" % (
            name, " ".join("%.3f" % (x / r) for x, r in zip(xc, rc)), xs / rs, "ok" if good else "FAIL"))

    gpath = os.path.join(out, "grating.ftsl")
    cm.write(gpath, grating_scene())
    near, far = grating_halves()

    def gimage(tag, args, gpu):
        return np.mean([cm.lum(cm.render(exe, out, gpath, "g%s_%d" % (tag, s), args, seed=s, want_gpu=gpu)[0])
                        for s in (1, 2)], axis=0)

    G = gimage("R_gpu", REF, True)
    for name, args in G_CASES:
        X = gimage(name, args, "gpu" in name)
        q = [X[h].mean() / G[h].mean() for h in (near, far)]
        good = all(abs(v - 1.0) <= TOL for v in q)
        ok &= good
        print("%-6s grating, of mode R: near %.3f  far %.3f  %s" % (name, q[0], q[1], "ok" if good else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
