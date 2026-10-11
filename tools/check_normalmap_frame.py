"""Regression check: the raster preview decodes normal maps with the tracers' frame handedness.

    D:/python314/python.exe -u tools/check_normalmap_frame.py [--out <dir>] [--exe ftrace.exe]

Every normal-mapped surface wears a constant tangent-space map tilted toward +B (the bitangent): RGB
(0.5, 0.8, 0.9), i.e. n = (0, 0.6, 0.8). The light is a sun from above, so a tilt up brightens a
surface and a tilt down darkens it.

  1. MIRRORED UVs. Two quads side by side, facing the camera, the right one's UVs mirrored in u
     (its frame is left-handed: Tri::bitangentSign -1), and a plain reference quad below. With the
     handedness honoured both tilt the same way and shade alike; ignoring it (the preview before
     0.374.2) the mirrored one tilts the other way. Checked in the CPU and GPU previews: each
     quad's brightness relative to the reference must agree to 3% (and differ from it by 10%).
  2. AN ANALYTIC SPHERE. The tracers use +1 for it; the preview tessellates it (wound inward, its
     vertex normals outward), and must use +1 too. Checked against the path tracer: the map darkens
     or brightens the sphere's front, against a plain sphere beside it, the same way in the preview
     as in mode R.
Comparisons are within one image, so each renderer's own exposure cancels. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(lines) + NL)


def write_inputs(d):
    W = 64
    Image.fromarray(np.full((W, W, 3), np.round(np.array((0.5, 0.8, 0.9)) * 255), np.uint8)).save(os.path.join(d, "tilt.ppm"))
    # top left: u grows with x (right-handed); top right: u grows against x (mirrored)
    write_lines(os.path.join(d, "quads.obj"), [
        "v -1.1 0.05 0", "v -0.1 0.05 0", "v -0.1 1.05 0", "v -1.1 1.05 0",
        "v 0.1 0.05 0", "v 1.1 0.05 0", "v 1.1 1.05 0", "v 0.1 1.05 0",
        "vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1",
        "vt 1 0", "vt 0 0", "vt 0 1", "vt 1 1",
        "vn 0 0 1",
        "f 1/1/1 2/2/1 3/3/1", "f 1/1/1 3/3/1 4/4/1",
        "f 5/5/1 6/6/1 7/7/1", "f 5/5/1 7/7/1 8/8/1"])
    # the reference below them, without a normal map
    write_lines(os.path.join(d, "ref.obj"), [
        "v -0.5 -1.05 0", "v 0.5 -1.05 0", "v 0.5 -0.05 0", "v -0.5 -0.05 0", "vn 0 0 1",
        "f 1//1 2//1 3//1", "f 1//1 3//1 4//1"])
    common = ['texture "nm" { file "%s/tilt.ppm"  encoding linear  filter bilinear  wrap repeat }' % d,
              'material "m" { type diffuse reflect 0.3  normal_map texture:nm }',
              'material "plain" { type diffuse reflect 0.3 }',
              'light sun { dir 0 0.8 0.6  angle 0.53  spd blackbody 5800  intensity 1e-13 }']
    write_lines(os.path.join(d, "quads.ftsl"),
                ['camera "c" { eye 0 0 6  look_at 0 0 0  up 0 1 0  fov_y 25  film { res 400 400 } }'] + common +
                ['mesh "q" { file "%s/quads.obj"  material m  uv use_mesh }' % d,
                 'mesh "r" { file "%s/ref.obj"  material plain }' % d])
    write_lines(os.path.join(d, "spheres.ftsl"),
                ['camera "c" { eye 0 0 6  look_at 0 0 0  up 0 1 0  fov_y 25  film { res 400 200 } }'] + common +
                ['sphere { center -0.6 0 0  radius 0.5  material m }',
                 'sphere { center 0.6 0 0  radius 0.5  material plain }'])


def run(exe, scene, out, args):
    """Render `scene` to `out` with the live window minimized (the repo's rule). A raster preview
    then stays open in its viewer once its frame is written, so it is stopped the clean way,
    `ftrace -stop <pid>` -- never killed. Returns the image as the mean of its RGB channels."""
    if os.path.exists(out):
        os.remove(out)
    log = out + ".log"
    cmd = [exe, "-in", scene, "-o", out] + args + ["-window-min"]
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while p.poll() is None:
        text = open(log, encoding="utf-8", errors="replace").read()
        if "[raster] done" in text and os.path.exists(out):
            subprocess.run([exe, "-stop", str(p.pid)], capture_output=True, timeout=200)
            p.wait(timeout=200)
            break
        if time.time() - t0 > 900:
            subprocess.run([exe, "-stop", str(p.pid)], capture_output=True, timeout=200)
            raise RuntimeError("render did not finish in 15 minutes: %s" % " ".join(cmd))
        time.sleep(0.5)
    if not os.path.exists(out):
        print(open(log, encoding="utf-8", errors="replace").read()[-2000:])
        raise RuntimeError("render failed: %s" % " ".join(cmd))
    return np.asarray(Image.open(out).convert("RGB")).astype(np.float64).mean(2)


def box(img, x0, x1, y0, y1):
    """The mean over a box given in fractions of the image (x left to right, y top to bottom)."""
    h, w = img.shape
    return img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_normalmap"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    d = os.path.abspath(a.out).replace(chr(92), "/")
    os.makedirs(d, exist_ok=True)
    write_inputs(d)
    ok = True
    # 1) mirrored UVs, both preview backends. At z = 0 the 25-degree camera at 6 sees +-1.33, so the
    #    top quads span x 0.09-0.46 / 0.54-0.91 and y 0.11-0.48 of the frame, the reference x
    #    0.31-0.69, y 0.52-0.89.
    for dev in ("cpu", "gpu"):
        img = run(a.exe, d + "/quads.ftsl", "%s/quads_%s.png" % (d, dev), ["-raster", "-device", dev])
        ref = box(img, 0.40, 0.60, 0.58, 0.85)
        el = box(img, 0.13, 0.42, 0.16, 0.43) / max(1e-9, ref)
        er = box(img, 0.58, 0.87, 0.16, 0.43) / max(1e-9, ref)
        good = abs(el - er) < 0.03 * max(el, er) and abs(el - 1.0) > 0.10
        ok &= good
        print("mirrored UVs, %s preview: the plain quad %.2f of the reference, the mirrored one %.2f  %s"
              % (dev.upper(), el, er, "ok" if good else "FAIL (the mirrored quad decodes the other way)"))
    # 2) the analytic sphere: its front (the frame is 400 x 200, so x spans +-2.66: the spheres'
    #    centres sit at x 0.387 and 0.613 of the frame), against the plain sphere's
    eff = {}
    for name, args in (("cpu", ["-raster", "-device", "cpu"]), ("gpu", ["-raster", "-device", "gpu"]),
                       ("traced", ["-mode", "R", "-device", "gpu", "-spp", "64"])):
        img = run(a.exe, d + "/spheres.ftsl", "%s/spheres_%s.png" % (d, name), args)
        eff[name] = box(img, 0.37, 0.405, 0.44, 0.56) / max(1e-9, box(img, 0.595, 0.63, 0.44, 0.56))
    for name in ("cpu", "gpu"):
        good = (eff[name] - 1.0) * (eff["traced"] - 1.0) > 0 and abs(eff["traced"] - 1.0) > 0.05
        ok &= good
        print("sphere, %s preview: the map makes its front %.2f of the plain sphere's, the path tracer %.2f  %s"
              % (name.upper(), eff[name], eff["traced"], "ok" if good else "FAIL (the preview tilts it the other way)"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
