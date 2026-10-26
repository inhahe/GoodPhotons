"""Regression check: no light leaks through a surface whose shading normal leans away from it.

    python tools/check_shading_leak.py [--out png/check_shading_leak] [--exe ftrace.exe]

A continuation drawn about a shading normal that a normal map or smooth shading has turned away
from the true surface can leave on the surface's wrong side -- a reflection behind it, a
transmission back -- travel inside or behind the geometry, and bring back light from there;
and a merge can take a photon from the surface's other face for a reflection. Every walk absorbs
such a continuation (geometry.h leavesWrongSide) and every merge refuses such a photon
(mergeAcrossSurface). Two parts.

SHEETS. An opaque horizontal sheet, seen from above, lit ONLY from below -- no path brings that
light to its top side except through it -- against the same sheet lit from above, which sets the
scale. Lit from below three ways: by a lamp, wide and close, so a continuation that dips just
under the sheet reaches it too; by a floor a lamp hidden under the sheet lights, so a
continuation that crosses brings back what the floor's own light sampling finds there -- the
path a kernel that lights by NEE alone follows; and by the same with the lamp 5 cm under the
sheet, within the merge radius of modes J and U, where a photon stored on the lamp itself was
merged by the sheet (bdpt.h surfMergeSite). The sheets:

  mapped   a quad under a ripple normal map tilting its normal up to about 40 degrees;
  tipped   the same under a map tilting it past 90 degrees (texels with blue below 0.5), so
           the shading normal itself points into the surface;
  smooth   a mesh sheet with vertex normals tilted +-35 degrees from its faces.

The top lit from below must read under 1/1000 of the top lit from above.

MIRRORS. A large mirror sheet seen at a grazing 4 degrees, its normals leaning toward and away
from the camera (a 40-degree ripple map, or smooth normals at +-35 degrees), over a lamp facing
up: where n leans toward the camera, a camera ray arrives below n's plane and its reflection
about n points down through the mirror. With nothing above it to reflect, the image must read
under 1/1000 of the same view with the mirror taken away (the lamp seen directly).

In modes R, D, J, U, M and S on the CPU and the GPU, and B (forward) on the GPU. Mode M runs at
its fixed radius (-nopmauto): its adaptive one outgrows the gap to the floor below and gathers
the floor's own photons (known-issues MERGE-PROXIMITY), which is not what this checks. Exits 0
when every case holds, 1 otherwise.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
MODES = [("R_cpu", ["-mode", "R", "-spp", "16", "-device", "cpu"]),
         ("R_gpu", ["-mode", "R", "-spp", "64", "-device", "gpu"]),
         ("D_cpu", ["-mode", "D", "-spp", "4", "-device", "cpu"]),
         ("D_gpu", ["-mode", "D", "-spp", "16", "-device", "gpu"]),
         ("B_gpu", ["-mode", "B", "-n", "4000000", "-device", "gpu"]),
         ("J_cpu", ["-mode", "J", "-spp", "2", "-device", "cpu"]),
         ("J_gpu", ["-mode", "J", "-spp", "4", "-device", "gpu"]),
         ("U_cpu", ["-mode", "U", "-spp", "2", "-device", "cpu"]),
         ("U_gpu", ["-mode", "U", "-spp", "4", "-device", "gpu"]),
         ("M_cpu", ["-mode", "M", "-nopmauto", "-n", "1000000", "-spp", "4", "-device", "cpu"]),
         ("M_gpu", ["-mode", "M", "-nopmauto", "-n", "1000000", "-spp", "4", "-device", "gpu"]),
         ("S_cpu", ["-mode", "S", "-spp", "2", "-device", "cpu"]),
         ("S_gpu", ["-mode", "S", "-spp", "4", "-device", "gpu"])]


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def normal_map(path, max_angle, w=64):
    """A tangent-space map tilting the normal along u by max_angle * sin(8 pi u), as a PPM."""
    u = (np.arange(w) + 0.5) / w
    a = max_angle * np.sin(2 * np.pi * 4 * u)
    n = np.stack([np.sin(a), np.zeros(w), np.cos(a)], axis=-1)
    rgb = np.round((n * 0.5 + 0.5) * 255).astype(np.uint8)
    img = np.ascontiguousarray(np.broadcast_to(rgb[None, :, :], (w, w, 3)))
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, w))
        f.write(img.tobytes())


def smooth_sheet(path, half=1.0, lean="x", k=6, tilt=np.radians(35)):
    """A flat k x k grid in y = 0 spanning +-half, whose vertex normals lean +-tilt toward +-x
    (alternating along x) or toward +-z (alternating along z)."""
    out = []
    for j in range(k + 1):
        for i in range(k + 1):
            x, z = half * (-1 + 2 * i / k), half * (-1 + 2 * j / k)
            if lean == "x":
                s = tilt if i % 2 == 0 else -tilt
                nrm = (np.sin(s), np.cos(s), 0.0)
            else:
                s = tilt if j % 2 == 0 else -tilt
                nrm = (0.0, np.cos(s), np.sin(s))
            out += ["v %.9g 0 %.9g" % (x, z), "vn %.9g %.9g %.9g" % nrm]

    def idx(i, j):
        return j * (k + 1) + i + 1
    for j in range(k):
        for i in range(k):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            out += ["f %d//%d %d//%d %d//%d" % (a, a, d, d, c, c), "f %d//%d %d//%d %d//%d" % (a, a, c, c, b, b)]
    write(path, NL.join(out) + NL)


def scenes(out):
    d = os.path.join(out, "scenes")
    os.makedirs(d, exist_ok=True)
    p = d.replace(chr(92), "/")
    normal_map(os.path.join(d, "mapped.ppm"), np.radians(40))
    normal_map(os.path.join(d, "tipped.ppm"), np.radians(110))
    smooth_sheet(os.path.join(d, "sheet.obj"))
    smooth_sheet(os.path.join(d, "mirror.obj"), half=10.0, lean="z")
    head = 'scene { units meters }\nmaterial "lit" { type diffuse reflect 0.8 }\n'
    below = 'light area { origin -10 -0.3 -10  u 20 0 0  v 0 0 20  normal 0 1 0  spd preset:bb6500  power 10000 }\n'

    # SHEETS, seen from straight above.
    cam = 'camera "c" { eye 0 2.5 0.01  look_at 0 0 0  up 0 0 -1  fov_y 40  film { res 64 64 } }\n'
    # Below: wide and close, so even a continuation that dips just under the sheet finds it.
    lamps = {"below": below,
             # a lit floor below instead: a crossing continuation brings back what a lamp under
             # the sheet lights there (paths a kernel lighting by NEE alone also follow).
             "floor": ('quad { origin -10 -0.8 -10  u 0 0 20  v 20 0 0  material lit }\n'
                       'light area { origin -0.3 -0.4 -0.3  u 0 0 0.6  v 0.6 0 0  normal 0 -1 0  spd preset:bb6500  power 200 }\n'),
             # the same lamp 5 cm under the sheet, well within the merge radius (0.28 here): the
             # light the floor sends back up lands on the lamp's back face, and a photon stored
             # there was merged by the sheet's top in mode J, whatever its shading normal
             # (known-issues J-LAMP-PHOTONS).
             "near": ('quad { origin -10 -0.5 -10  u 0 0 20  v 20 0 0  material lit }\n'
                      'light area { origin -0.3 -0.05 -0.3  u 0 0 0.6  v 0.6 0 0  normal 0 -1 0  spd preset:bb6500  power 200 }\n'),
             # off to the side, out of the camera's view, aimed at the sheet
             "above":'light area { origin 1.2 1.2 -0.6  u 0 0 1.2  v 0.6 0 0  normal -0.6 -0.8 0  spd preset:bb6500  power 50 }\n'}
    quad = 'quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material m }\n'          # faces up (u x v)
    bodies = {
        "mapped": ('texture "nm" { file "%s/mapped.ppm"  encoding linear  filter bilinear  wrap repeat }\n'
                   'material "m" { type diffuse reflect 0.8  normal_map texture:nm }\n' % p) + quad,
        "tipped": ('texture "nm" { file "%s/tipped.ppm"  encoding linear  filter bilinear  wrap repeat }\n'
                   'material "m" { type diffuse reflect 0.8  normal_map texture:nm }\n' % p) + quad,
        "smooth": 'material "m" { type diffuse reflect 0.8 }\nmesh "s" { file "%s/sheet.obj"  material m }\n' % p,
    }
    paths = {}
    for name, body in bodies.items():
        for side, lamp in lamps.items():
            paths[(name, side)] = os.path.join(d, "%s_%s.ftsl" % (name, side))
            write(paths[(name, side)], head + body + lamp + cam)

    # MIRRORS, seen at a grazing angle: 20 m sheets whose normals lean along the view (z).
    gcam = 'camera "c" { eye 0 0.4 6  look_at 0 0 0  up 0 1 0  fov_y 30  film { res 64 48 } }\n'
    big = 'quad { origin -10 0 -10  u 0 0 20  v 20 0 0  material m }\n'      # faces up; u along z
    mirrors = {
        "mirror_mapped": ('texture "nm" { file "%s/mapped.ppm"  encoding linear  filter bilinear  wrap repeat }\n'
                          'material "m" { type mirror reflect 0.9  normal_map texture:nm }\n' % p) + big,
        "mirror_smooth": ('material "m" { type mirror reflect 0.9 }\n'
                          'mesh "s" { file "%s/mirror.obj"  material m }\n' % p),
        "mirror_open": "",                                    # the lamp seen directly: the scale
    }
    for name, body in mirrors.items():
        paths[(name, "below")] = os.path.join(d, "%s.ftsl" % name)
        write(paths[(name, "below")], head + body + below + gcam)
    return paths


def read_pfm(path):
    with open(path, "rb") as f:
        ch = 3 if f.readline().strip() == b"PF" else 1
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        a = np.fromfile(f, "<f4" if scale < 0 else ">f4").reshape(h, w, ch).astype(np.float64)
    return a.mean(axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_shading_leak"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    exe = os.path.abspath(a.exe)
    paths = scenes(out)
    ok = True

    def render(name, side, tag, args):
        """The render's HDR image, or None (and the failure printed)."""
        png = os.path.join(out, "%s_%s_%s.png" % (name, side, tag))
        for stale in (png, png[:-4] + ".pfm"):
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [exe, "-in", paths[(name, side)], "-hdr", "-no-meter", "-seed", "5", "-o", png] + args
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not os.path.exists(png[:-4] + ".pfm"):
            print("  %s %s %s: render failed%s%s" % (name, side, tag, NL, (r.stdout + r.stderr)[-400:]))
            return None
        return read_pfm(png[:-4] + ".pfm")

    for tag, args in MODES:
        for name in ("mapped", "tipped", "smooth"):
            top = {}
            for side in ("below", "floor", "near", "above"):
                img = render(name, side, tag, args)
                if img is None:
                    ok = False
                    break
                h, w = img.shape
                top[side] = float(img[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean())   # the sheet's middle
            if len(top) < 4:
                continue
            for side in ("below", "floor", "near"):
                ratio = top[side] / max(top["above"], 1e-30)
                good = top["above"] > 0 and ratio < 1e-3
                ok &= good
                print("  %-13s %-6s %-5s: the top lit from below reads %.2e of the top lit from above  %s"
                      % (name, tag, side, ratio, "ok" if good else "FAIL (light leaks through the surface)"))
        imgs = {name: render(name, "below", tag, args) for name in ("mirror_open", "mirror_mapped", "mirror_smooth")}
        if any(v is None for v in imgs.values()):
            ok = False
            continue
        scale = float(imgs["mirror_open"].mean())
        for name in ("mirror_mapped", "mirror_smooth"):
            ratio = float(imgs[name].mean()) / max(scale, 1e-30)
            good = scale > 0 and ratio < 1e-3
            ok &= good
            print("  %-13s %-6s grazing: the mirror over the lamp reads %.2e of the lamp seen directly  %s"
                  % (name, tag, ratio, "ok" if good else "FAIL (light leaks through the mirror)"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
