"""Regression check: a mesh-bound medium's `feather` ramps its density in from the silhouette.

    python tools/check_feather.py [--out png/check_feather] [--exe ftrace.exe] [--devices cpu,gpu]

`medium { bounds { object "<mesh>"  feather <m> } }` bakes the mesh into an occupancy lattice and
feathers it (meshvox::featherGrid): each solid voxel holds a smoothstep of its distance to the
outside over the feather, 0 on the silhouette's voxels to 1 that far in. The density must follow
that ramp -- occupancy times the medium's own density (Medium::meshWeight, the device's
dMeshWeight). Read as a threshold at 0.5 instead, the ramp only moves a hard edge half the feather
inward (known-issues FEATHER-IS-A-THRESHOLD).

The scene: a 2 m cube of purely absorbing fog (albedo 0), feathered 0.4 m, seen from far off
against a uniform sky, so that each pixel is the sky dimmed by the fog's transmittance along its
ray. Mode R renders it on each device (the GPU's run must say it ran there). The optical depth -ln(I / I_sky), averaged over the
rows across the middle and over three columns at a time, is compared with the documented ramp integrated along the same rays.
The lattice puts its centres on the box's faces, where membership is a coin flip, so the ramp
starts within a voxel of the face; the model takes the middle, half a voxel in, and the voxels
are fine enough (320 across) that either way moves the optical depth by 0.02 at most:

  - outside the box: no fog;
  - across the feather band: the ramp, within 0.08 (a threshold leaves the outer half of the
    band clear, about 0.5 short there);
  - inside: the full density, through 1.2 m of it between the front and back bands (optical
    depth ~2.4; a density 5 % short reads 0.09 low);
  - and a feather under one voxel, which has no ramp to resolve, leaves the bound as it is
    unfeathered: the CPU's image byte for byte.

Exits 1 on any failure.
"""
import argparse
import math
import os
import re
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

SIGMA = 1.5        # the fog's extinction, 1/m
FEATHER = 0.4      # m
HALF = (1.0, 1.0, 1.0)   # the box's half-extents, m
EYE = 30.0         # the camera's distance, on +z
FOV = 5.0          # degrees: the frame spans about +-1.31 m at the box
RES = 160
TOL = 0.08         # in optical depth


def box_obj(path):
    hx, hy, hz = HALF
    v = [(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    ix = {p: i + 1 for i, p in enumerate(v)}
    def q(a, b, c, d):   # one face, wound counter-clockwise seen from outside
        return ["f %d %d %d" % (ix[a], ix[b], ix[c]), "f %d %d %d" % (ix[a], ix[c], ix[d])]
    X, Y, Z = hx, hy, hz
    faces = (q((X, -Y, -Z), (X, Y, -Z), (X, Y, Z), (X, -Y, Z)) +
             q((-X, -Y, -Z), (-X, -Y, Z), (-X, Y, Z), (-X, Y, -Z)) +
             q((-X, Y, -Z), (-X, Y, Z), (X, Y, Z), (X, Y, -Z)) +
             q((-X, -Y, -Z), (X, -Y, -Z), (X, -Y, Z), (-X, -Y, Z)) +
             q((-X, -Y, Z), (X, -Y, Z), (X, Y, Z), (-X, Y, Z)) +
             q((-X, -Y, -Z), (-X, Y, -Z), (X, Y, -Z), (X, -Y, -Z)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(["v %.6f %.6f %.6f" % p for p in v] + faces) + NL)


def scene_text(obj, feather=FEATHER):
    fk = "  feather %g" % feather if feather else ""
    return NL.join([
        "scene { units meters }",
        "light env { spd preset:d65 }",
        'material "w" { type diffuse reflect 0.5 }',
        'mesh "box" { file "%s"  material w  shape_only yes }' % obj.replace(chr(92), "/"),
        'medium { sigma_t %g  albedo 0  bounds { object "box"  voxels 320%s } }' % (SIGMA, fk),
        'camera "c" { eye 0 0 %g  look_at 0 0 0  up 0 1 0  fov_y %g  film { res %d %d } }'
        % (EYE, FOV, RES, RES)]) + NL


def read_pfm(path):
    with open(path, "rb") as f:
        hdr = f.readline().strip()
        nc = 3 if hdr == b"PF" else 1
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        scale = float(f.readline().strip())
        n = w * h * nc
        data = struct.unpack("%s%df" % ("<" if scale < 0 else ">", n), f.read(4 * n))
    return w, h, nc, data


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def expected_tau(x, voxel):
    """The ramp's optical depth along the central row's ray through x (m, at the box's centre
    plane): density smoothstep((d - voxel/2) / feather) at true depth d inside the box."""
    n = 2000
    hz = HALF[2]
    dz = 2.0 * hz / n
    stretch = math.sqrt(1.0 + (x / EYE) ** 2)
    tau = 0.0
    for k in range(n):
        z = -hz + (k + 0.5) * dz
        X = x * (EYE - z) / EYE
        d = min(HALF[0] - abs(X), HALF[1], hz - abs(z))
        if d > voxel / 2:
            tau += SIGMA * smoothstep((d - voxel / 2) / FEATHER) * dz * stretch
    return tau


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_feather"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--devices", default="cpu,gpu")
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    obj = os.path.join(d, "box.obj")
    box_obj(obj)
    ftsl = os.path.join(d, "feather.ftsl")
    with open(ftsl, "w", encoding="utf-8", newline="") as f:
        f.write(scene_text(obj))
    fails = []

    def check(name, ok, detail=""):
        print("%-58s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    half_w = EYE * math.tan(math.radians(FOV) / 2)
    for dev in a.devices.split(","):
        png = os.path.join(d, "feather_%s.png" % dev)
        cmd = [exe, "-in", ftsl, "-mode", "R", "-device", dev, "-seed", "1", "-no-meter",
               "-spp", "1024", "-hdr", "-o", png]
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        log = p.stdout + p.stderr
        if p.returncode != 0:
            print(log[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        m = re.search(r"feathered ([0-9.eE+-]+) m \(([0-9.]+) voxels\)", log)
        if not m:
            print(log[-1500:])
            raise SystemExit("the loader printed no feather report")
        if dev == "gpu" and "on GPU" not in log:
            check("gpu: the render ran on the GPU", False, "the log does not say so")
            continue
        voxel = float(m.group(1)) / float(m.group(2))
        w, h, nc, data = read_pfm(png[:-4] + ".pfm")
        rows = [r for r in range(h) if abs((2.0 * (r + 0.5) / h - 1.0) * half_w) < 0.4]

        def lum(r, c):
            return data[(r * w + c) * nc + (1 if nc == 3 else 0)]
        xs = [(2.0 * (c + 0.5) / w - 1.0) * half_w for c in range(w)]
        col = [sum(lum(r, c) for r in rows) / len(rows) for c in range(w)]
        sky = [v for x, v in zip(xs, col) if abs(x) > HALF[0] + 0.1]
        i_sky = sum(sky) / len(sky)
        worst = {"outside": (0.0, 0.0), "band": (0.0, 0.0), "inside": (0.0, 0.0)}
        # Columns in threes, against the model's transmittance averaged over the same three.
        for c0 in range(0, w - 2, 3):
            x = xs[c0 + 1]
            if abs(x) > HALF[0] + 0.1:
                continue
            tau = -math.log(max(sum(col[c0:c0 + 3]) / 3, 1e-12) / i_sky)
            want = -math.log(sum(math.exp(-expected_tau(xs[c], voxel)) for c in range(c0, c0 + 3)) / 3)
            depth = HALF[0] - abs(x)
            part = "outside" if depth < 0 else ("band" if depth < FEATHER else "inside")
            err = tau - want
            if abs(err) > abs(worst[part][0]):
                worst[part] = (err, x)
        for part, what in (("outside", "no fog outside the box"),
                           ("band", "the feather band follows the ramp"),
                           ("inside", "the full density inside")):
            err, x = worst[part]
            check("%s: %s" % (dev, what), abs(err) <= TOL,
                  "worst optical-depth error %+.3f at x = %+.3f m" % (err, x))
    # Under one voxel (320 across 2 m: 6.25 mm) there is nothing to ramp.
    imgs = []
    for tag, fe in (("hard", 0.0), ("subvoxel", 0.003)):
        path = os.path.join(d, tag + ".ftsl")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(scene_text(obj, fe))
        png = os.path.join(d, tag + ".png")
        p = subprocess.run([exe, "-in", path, "-mode", "R", "-device", "cpu", "-seed", "1",
                            "-no-meter", "-r", "48", "-spp", "16", "-hdr", "-o", png],
                           cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-1500:])
            raise SystemExit("render failed: " + tag)
        with open(png[:-4] + ".pfm", "rb") as f:
            imgs.append(f.read())
    check("cpu: a feather under one voxel leaves the edge hard", imgs[0] == imgs[1],
          "" if imgs[0] == imgs[1] else "the images differ")
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
