"""Regression check: modes R and W draw camera wavelengths through a visible-band window, unbiased.

    D:/python314/python.exe -u tools/check_cam_vis.py [--out <dir>] [--exe ftrace.exe] [--spp N]
                                                      [--only cornell|fire|modew|gpu]

A camera path draws its wavelengths from Scene::camDraw -- the illuminant's g(lambda) times a
window over the visible band (mode R: pbrt-v4's 1/cosh^2; mode W: flat over 400..700 nm) -- while
the path keeps the illuminant's own weight camG / g and the film multiplies each wavelength by
Scene::camFilmWeight, the ratio of the two densities. FTRACE_CAM_VIS=0 draws from the illuminant
alone, as before 0.377.0, so the two must render the same image; only the noise differs.
Compared:
  cornell  mode R, the Cornell box under a spiky triphosphor light and a narrow blue band, with a
           dispersive SF10 sphere (a bundle de-heros there) and a gold one; -heroc 4 and 1;
  fire     mode R, tools/check_fire_r.py's fire, the only light: a 1500 K blackbody, most of whose
           power lies in the near infrared, where the old draw spent most of its wavelengths;
  modew    mode W on the same Cornell box: deterministic, so at 256 spp, by tolerance alone;
  gpu      mode R on the device (kBackward), the box with both spheres diffuse and its lights in
           lumens -- on against off, and on against the host's own render, since an error in the
           device's film weight alone would scale both of its renders alike -- and mode W likewise.
A tile of a 4 x 4 grid fails when the two differ by more than 5 standard errors (from the seed
pairs, tile- and pixel-wise, the larger kept; none for mode W) AND 1 % (3 % device against host);
the frame when by more than 5 standard errors AND 0.3 % (1 % device against host). Also prints
the summed per-pixel variance, old draw over new. Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from check_glossy_nee_media import read_pfm, tile_stats  # noqa: E402
from check_fire_r import write_fire  # noqa: E402

NL = chr(10)
CORNELL = [
    'material "white" { type diffuse  reflect whitewall 0.75 }',
    'material "red"   { type diffuse  reflect redwall }',
    'material "green" { type diffuse  reflect greenwall }',
    'material "glass" { type dielectric  ior glass:SF10 }',
    'material "gold"  { type glossy  reflect metal:gold  roughness 0.1 }',
    'quad { origin 0 0 0   u 1 0 0   v 0 0 1   material white }',
    'quad { origin 0 1 0   u 0 0 1   v 1 0 0   material white }',
    'quad { origin 0 0 0   u 0 1 0   v 1 0 0   material white }',
    'quad { origin 0 0 0   u 0 0 1   v 0 1 0   material red }',
    'quad { origin 1 0 0   u 0 1 0   v 0 0 1   material green }',
    'sphere { center 0.33 0.22 0.45  radius 0.2  material glass }',
    'sphere { center 0.72 0.18 0.6  radius 0.16  material gold }',
    'light area { origin 0.35 0.999 0.35  u 0.3 0 0  v 0 0 0.3  normal 0 -1 0  spd preset:f11 }',
    'light area { origin 0.05 0.6 0.9  u 0 0 -0.2  v 0 0.2 0  normal 1 0 0'
    '  spd gaussian center=450 sigma=6 amp=4 }',
    'camera "c" { eye 0.5 0.5 2.7  look_at 0.5 0.5 0.5  up 0 1 0  fov_y 40  film { res 64 64 } }',
]
# The device case: both spheres diffuse (the device's mode-R scope), and the lights in lumens, so
# the device's photometric scaling is covered too.
CORNELL_GPU = [l.replace("material glass }", "material white }")
                .replace("material gold }", "material white }")
                .replace("spd preset:f11 }", "spd preset:f11  lumens 2000 }")
                .replace("amp=4 }", "amp=4  lumens 200 }")
               for l in CORNELL]


def verdict(on, off, z_max, rel_min):
    """Ratio on / off, z-scores and failing cells (both z above `z_max` and the relative
    difference above `rel_min`), from tile_stats (mean, standard error) pairs."""
    (m1, e1), (m0, e0) = on, off
    z = np.abs(m1 - m0) / np.maximum(np.hypot(e1, e0), 1e-30)
    ratio = m1 / np.maximum(m0, 1e-30)
    live = m0 > 0.02 * m0.max()
    return ratio, np.where(live, z, 0.0), live & (z > z_max) & (np.abs(ratio - 1.0) > rel_min)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_cam_vis"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    write_fire(d)
    for name, lines in (("cornell.ftsl", CORNELL), ("cornell_gpu.ftsl", CORNELL_GPU)):
        with open(os.path.join(d, name), "w", encoding="utf-8", newline="") as f:
            f.write(NL.join(lines) + NL)

    def render(tag, scene, args, vis):
        imgs = []
        for seed in (1, 2):
            png = "%s_%s_s%d.png" % (tag, "on" if vis else "off", seed)
            env = dict(os.environ)
            if not vis:
                env["FTRACE_CAM_VIS"] = "0"
            cmd = [a.exe, "-in", scene, "-spp", str(a.spp), "-seed", str(seed), "-hdr",
                   "-no-meter", "-o", png] + args
            p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            pfm = os.path.join(d, png[:-4] + ".pfm")
            if p.returncode != 0 or not os.path.exists(pfm):
                print(p.stdout[-1500:], p.stderr[-1500:])
                raise SystemExit("render failed: " + " ".join(cmd))
            imgs.append(read_pfm(pfm).sum(-1))
        return imgs

    def compare(name, x1, x0, tile_tol, frame_tol, stochastic):
        ratio, z, bad = verdict(tile_stats(*x1), tile_stats(*x0), 5.0, tile_tol)
        fr, fz, fbad = verdict(tile_stats(*x1, 1), tile_stats(*x0, 1), 5.0, frame_tol)
        good = not bad.any() and not fbad.any()
        i, j = np.unravel_index(int(np.argmax(np.abs(ratio - 1.0) * (z > 0))), z.shape)
        var = lambda im: float(((im[0] - im[1]) ** 2).sum())   # noqa: E731
        if stochastic:
            print("%s: %s, frame x%.4f (%.1f sigma); worst tile x%.4f (%.1f sigma); pixel variance "
                  "x%.2f lower" % ("ok" if good else "FAIL", name, fr[0, 0], fz[0, 0], ratio[i, j],
                                   z[i, j], var(x0) / max(var(x1), 1e-300)), flush=True)
        else:   # mode W: no seed noise, so no standard error -- the tolerances alone decide
            print("%s: %s, frame x%.4f; worst tile x%.4f (deterministic)" % (
                "ok" if good else "FAIL", name, fr[0, 0], ratio[i, j]), flush=True)
        return good

    R4, R1 = ["-mode", "R", "-heroc", "4"], ["-mode", "R", "-heroc", "1"]
    W = ["-mode", "W"]
    CPU, GPU = ["-device", "cpu"], ["-device", "gpu"]
    ok = True
    for group, label, scene, args in (("cornell", "cornell R c4", "cornell.ftsl", R4 + CPU),
                                      ("cornell", "cornell R c1", "cornell.ftsl", R1 + CPU),
                                      ("fire", "fire R c4", "fire.ftsl", R4 + CPU),
                                      ("modew", "cornell W", "cornell.ftsl", W + CPU),
                                      ("gpu", "gpu R c4", "cornell_gpu.ftsl", R4 + GPU),
                                      ("gpu", "gpu W", "cornell_gpu.ftsl", W + GPU)):
        if a.only and a.only != group:
            continue
        tag = label.replace(" ", "_")
        on, off = render(tag, scene, args, True), render(tag, scene, args, False)
        ok &= compare("%s: windowed / illuminant draw" % label, on, off, 0.01, 0.003, "R" in args)
        if group == "gpu":
            host = render(tag + "_host", scene, args[:-2] + CPU, True)
            ok &= compare("%s: device / host" % label, on, host, 0.03, 0.01, "R" in args)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
