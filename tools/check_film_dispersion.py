"""Regression check: a thin film's index is read at each wavelength -- `film_ior` a spectrum.

    D:/python314/python.exe -u tools/check_film_dispersion.py [--out <dir>] [--exe ftrace.exe]
                                                               [--no-gpu]

`film_ior` takes a spectrum (glass:titania, TiO2 as a thin film: 2.92 at 360 nm, 2.44 at 550,
2.34 at 830). Under a narrow-band light at lambda0 (a 1 nm gaussian), a film with
`film_ior glass:titania` must render as one with the constant titania(lambda0); with the
constant taken at another wavelength it must not -- the control, without which agreement would
prove nothing. Three films, the three code paths that read the index, each a sphere in front of a
white wall lit by the narrow-band panel:
  * `clear`: titania on BK7, the specular coating (interfaceOptics);
  * `coat`: a `layered` material whose thin-film coat is titania over a white body
    (layeredCoatReflectance);
  * `rough`: titania on titanium with a roughness, the rough interference reflector
    (coatingReflectance).
Each renders in mode R, and the rough one in mode B too (the forward camera sees no specular
coating directly), on the CPU and, unless --no-gpu, the GPU, at lambda0 = 420, 550 and 680 nm.
The sphere's mean (a disc of 8 pixels' radius about its centre, at the frame's centre) must agree
with the constant's to 1 %, and differ from the control's by more than 3 %. Exits 1 on any
failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
GPU_TOKENS = ("-> GPU", "[gpu]")
FILMS = {
    "clear": 'type thinfilm  ior glass:BK7  film_ior %s  film_thickness 250',
    "coat": 'type layered  ior 1.5  coat { reflectance thinfilm  ior 1.5  film_ior %s'
            '  film_thickness 250  roughness 0.05 }  layer "white" 1',
    "rough": 'type thinfilm  ior 2.5  substrate_k 3.0  film_ior %s  film_thickness 80'
             '  roughness 0.15',
}
LAMBDAS = (420, 550, 680)
CONTROL = {420: 680, 550: 420, 680: 420}     # where the control's constant is read
MODES = [("R", ["-mode", "R", "-spp", "256"]), ("B", ["-mode", "B", "-n", "20000000"])]


def titania():
    rows = [l.split() for l in open(os.path.join(ROOT, "data", "glass", "titania.glass"),
                                    encoding="utf-8") if l[:1].isdigit()]
    lam = np.array([float(r[0]) for r in rows])
    n = np.array([float(r[1]) for r in rows])
    return lambda L: float(np.interp(L, lam, n))


def scene(film, ior, lam0):
    return NL.join([
        "scene { units meters }",
        'material "white" { type diffuse reflect 0.8 }',
        'material "film" { %s }' % (FILMS[film] % ior),
        "sphere { center 0 0.5 0  radius 0.5  material film }",
        "quad { origin -2 0 -1  u 4 0 0  v 0 3 0  material white }",
        "quad { origin -2 0 -1  u 0 0 3  v 4 0 0  material white }",
        "light area { origin -0.5 2.5 0.5  u 1 0 0  v 0 0 1  normal 0 -1 0"
        "  spd gaussian center=%d sigma=1  power 50 }" % lam0,
        'camera "c" { eye 0 1.0 3.0  look_at 0 0.5 0  up 0 1 0  fov_y 40  film { res 64 48 } }',
    ]) + NL


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        return np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, 3)


def render(exe, out, name, text, args):
    path = os.path.join(out, name + ".ftsl")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    png = os.path.join(out, name + ".png")
    pfm = png[:-4] + ".pfm"
    if os.path.exists(pfm):
        os.remove(pfm)
    cmd = [exe, "-in", path, "-seed", "3", "-hdr", "-no-meter", "-o", png] + args
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(pfm):
        print(p.stdout[-1500:], p.stderr[-1500:])
        raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
    if "gpu" in args and not any(t in p.stdout + p.stderr for t in GPU_TOKENS):
        raise SystemExit("did not run on the GPU: " + " ".join(cmd))
    img = read_pfm(pfm).astype(np.float64).sum(-1)
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]
    return float(img[(x + 0.5 - w / 2) ** 2 + (y + 0.5 - h / 2) ** 2 < 64].sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_film_dispersion"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    n = titania()
    ok = True
    for dev in ("cpu",) if a.no_gpu else ("cpu", "gpu"):
        for mtag, margs in MODES:
            for film in FILMS:
                if mtag == "B" and film != "rough":
                    continue
                for lam0 in LAMBDAS:
                    args = margs + ["-device", dev]
                    tag = "%s_%s_%s_%d" % (film, mtag, dev, lam0)
                    disp = render(exe, out, tag + "_disp", scene(film, "glass:titania", lam0), args)
                    same = render(exe, out, tag + "_const",
                                  scene(film, "%.6f" % n(lam0), lam0), args)
                    ctrl = render(exe, out, tag + "_ctrl",
                                  scene(film, "%.6f" % n(CONTROL[lam0]), lam0), args)
                    r_same, r_ctrl = disp / same, disp / ctrl
                    good = abs(r_same - 1.0) <= 0.01 and abs(r_ctrl - 1.0) > 0.03
                    ok &= good
                    print("%-22s dispersive / n(%d) %.4f   / control n(%d) %.4f%s"
                          % (tag, lam0, r_same, CONTROL[lam0], r_ctrl, "" if good else "  FAIL"),
                          flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
