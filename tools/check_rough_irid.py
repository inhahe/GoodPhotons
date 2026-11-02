"""Regression check: a rough reflector renders in the forward modes, and a rough interference
reflector renders the same in every mode.

    D:/python314/python.exe -u tools/check_rough_irid.py [--out <dir>] [--exe ftrace.exe]
                                                          [--no-gpu] [--only B_cpu,D_gpu,...]

A box lit by a ceiling panel holds three spheres: a plain rough `glossy`, a rough beetle stack (a
`multilayer` given a `roughness`) and a rough anodized-titanium film (a `thinfilm` given one) --
the rough interference reflectors of 0.401.0, glossy lobes weighted by the coating's reflectance at
the half-vector (Material::iridescence). Each sphere's mean colour, over the pixels that see it
first (an analytic ray cast of the camera, eroded from the rim), is held per channel against
mode R on the CPU, in modes A, B, D and M on the CPU and A, B, D, M and R on the GPU -- the
forward modes A and B included, where a rough glossy is visible only through the camera splat's
glossy lobe (render.h GlossySplat). Modes B, R and D run twice, the second time with `-heroc 1`, for
the walks that carry one wavelength. Mode U is left out: its merges read the camera path's colours
at the camera's wavelength, which washes any coloured glossy (known-issues VCM-SPEC). So is mode J:
its walk is mode D's, and its surface merges on these small curved glossy spheres are too noisy to
resolve a bias at a check's budget (at 1102 spp its sphere means still scattered by 5-12 %, the CPU
and GPU in opposite directions). A sphere channel fails when it differs from the reference by more
than 5 standard errors (two seeds each) AND by more than 3 %.

Mode A renders the same box through a 1 mm thin lens focused on the spheres. It records film
irradiance, not radiance: the camera equation scales it, and natural vignetting (cos^4 of the
field angle) darkens it away from the centre, by 0.69x at this frame's corners. So its image is
divided by that cos^4 and scaled to match the reference over the pixels that see no sphere, and
its spheres are then held to the reference like the others. Mode M on the GPU is scaled to the
reference over those pixels too, without the cos^4: it reads this box's walls 2-3 % bright, a
bias of its own that predates the spheres (known-issues MODEM-GPU-OFFSET), so its spheres are
judged against its own background. Drop that once the offset is fixed.

It also requires a CLEAR coating given a `roughness` (a soap film) to be refused at load, with
the message saying why, and the same film given `roughness 0` -- the specular coating, as with no
roughness at all -- to load. And it requires mode B to render a `layered` car-paint sphere the
same on the GPU as on the CPU (sphere mean per channel, 5 standard errors and 3 %): the device
gives a layered coat a synthetic glossy lobe, which must not take the glossy camera splat the
host never gives a coat (known-issues FWD-GLOSSY-INDIRECT).
Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
W, H = 128, 96
EYE, LOOK, FOV = (0.0, 1.2, 4.0), (0.0, 0.6, 0.0), 45.0
STACK = "  ".join(["layer 1.75 0 85.7  layer 1.35 0 111.1"] * 6)
SPHERES = [("glossy", (-1.0, 0.4, -0.5), 0.4), ("beetle", (0.0, 0.4, 0.0), 0.4),
           ("titanium", (1.0, 0.4, -0.5), 0.4)]
SCENE = NL.join([
    "scene { units meters }",
    'material "white" { type diffuse reflect 0.7 }',
    'material "red"   { type diffuse reflect rgb 0.7 0.1 0.1 }',
    'material "green" { type diffuse reflect rgb 0.1 0.7 0.1 }',
    'material "glossy" { type glossy reflect 0.8  roughness 0.2 }',
    'material "beetle" { type multilayer  ior 1.6  substrate_k 0.4  %s  roughness 0.2 }' % STACK,
    'material "titanium" { type thinfilm  ior 2.5  substrate_k 3.0  film_ior 2.30'
    '  film_thickness 85  roughness 0.15 }',
] + ["sphere { center %g %g %g  radius %g  material %s }" % (c + (r, n)) for n, c, r in SPHERES] + [
    "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
    "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
    "quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material red }",
    "quad { origin 2 0 -2  u 0 0 4  v 0 3 0  material green }",
    "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
    "  power 60 }",
    'camera "c" { eye %g %g %g  look_at %g %g %g  up 0 1 0  fov_y %g  film { res %d %d } }'
    % (EYE + LOOK + (FOV, W, H)),
]) + NL
SCENE_A = SCENE.replace("fov_y %g  film" % FOV, "fov_y %g  aperture 0.001  focus 4.04  film" % FOV)
assert SCENE_A != SCENE
CLEAR = NL.join([
    "scene { units meters }",
    'material "soap" { preset soap-bubble  roughness %s }',
    "sphere { center 0 0.4 0  radius 0.4  material soap }",
    "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
    "  power 60 }",
    'camera "c" { eye 0 1.2 4  look_at 0 0.6 0  up 0 1 0  fov_y 45  film { res 32 24 } }',
]) + NL
COAT = SCENE.replace('material "glossy" { type glossy reflect 0.8  roughness 0.2 }',
                     'material "base" { type diffuse reflect rgb 0.6 0.05 0.05 }' + NL +
                     'material "glossy" { type layered  ior 1.5  coat { roughness 0.05 }'
                     '  layer "base" 1 }')
assert COAT != SCENE
REF = ("R_cpu", ["-mode", "R", "-device", "cpu", "-spp", "4096"])
CASES = [   # the A cases render SCENE_A, through the lens
    ("A_cpu", ["-mode", "A", "-device", "cpu", "-n", "120000000"]),
    ("B_cpu", ["-mode", "B", "-device", "cpu", "-n", "120000000"]),
    ("D_cpu", ["-mode", "D", "-device", "cpu", "-spp", "256"]),
    ("M_cpu", ["-mode", "M", "-device", "cpu", "-spp", "256"]),
    ("R_gpu", ["-mode", "R", "-device", "gpu", "-spp", "4096"]),
    ("A_gpu", ["-mode", "A", "-device", "gpu", "-n", "120000000"]),
    ("B_gpu", ["-mode", "B", "-device", "gpu", "-n", "120000000"]),
    ("D_gpu", ["-mode", "D", "-device", "gpu", "-spp", "512"]),
    ("M_gpu", ["-mode", "M", "-device", "gpu", "-spp", "512"]),
    # one wavelength per path: the scalar walks, beside the hero bundles above
    ("B1_cpu", ["-mode", "B", "-device", "cpu", "-n", "120000000", "-heroc", "1"]),
    ("R1_cpu", ["-mode", "R", "-device", "cpu", "-spp", "2048", "-heroc", "1"]),
    ("D1_cpu", ["-mode", "D", "-device", "cpu", "-spp", "256", "-heroc", "1"]),
    ("B1_gpu", ["-mode", "B", "-device", "gpu", "-n", "120000000", "-heroc", "1"]),
    ("R1_gpu", ["-mode", "R", "-device", "gpu", "-spp", "4096", "-heroc", "1"]),
    ("D1_gpu", ["-mode", "D", "-device", "gpu", "-spp", "512", "-heroc", "1"]),
]
GPU_TOKENS = ("-> GPU", "[gpu]", "VCM/UPS on ", "SPPM on ")


def read_pfm(path):
    """A PFM as h x w x 3, top row first."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, 3)
    return data[::-1].astype(np.float64)


def frame():
    """Per pixel, the unit direction of its camera ray, and cos^4 of its angle to the axis."""
    eye, look = np.array(EYE), np.array(LOOK)
    f = (look - eye) / np.linalg.norm(look - eye)
    r = np.cross(f, [0.0, 1.0, 0.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    t = np.tan(np.radians(FOV) / 2)
    i, j = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = (f + ((2 * i / W - 1) * t * W / H)[..., None] * r + ((1 - 2 * j / H) * t)[..., None] * u)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d, (d * f).sum(-1) ** 4


def masks():
    """Per sphere, the pixels whose camera ray hits it first, through the inner 85 % of its
    radius (so no pixel mixes it with what lies behind its rim); and last, ("rest", the pixels
    that see no sphere at all)."""
    eye = np.array(EYE)
    d, _ = frame()
    best, lab, inner = np.full((H, W), np.inf), np.full((H, W), -1), np.zeros((H, W), bool)
    for k, (_, c, rad) in enumerate(SPHERES):
        oc = eye - np.array(c)
        b = (d * oc).sum(-1)
        cc = (oc * oc).sum() - rad * rad
        disc = b * b - cc
        tt = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
        closer = (tt > 0) & (tt < best)
        best = np.where(closer, tt, best)
        lab = np.where(closer, k, lab)
        inner = np.where(closer, np.sqrt(np.maximum(-disc + rad * rad, 0)) < 0.85 * rad, inner)
    return [(n, (lab == k) & inner) for k, (n, _, _) in enumerate(SPHERES)] + [("rest", lab == -1)]


def render(exe, out, scene, tag, args, seeds=(1, 2)):
    imgs = []
    for seed in seeds:
        png = os.path.join(out, "%s_s%d.png" % (tag, seed))
        pfm = png[:-4] + ".pfm"
        for f in (png, pfm):
            if os.path.exists(f):
                os.remove(f)
        cmd = [exe, "-in", scene, "-seed", str(seed), "-hdr", "-no-meter", "-o", png] + args
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.exists(pfm):
            print(p.stdout[-1500:], p.stderr[-1500:])
            raise SystemExit("render failed (exit %d): %s" % (p.returncode, " ".join(cmd)))
        if "gpu" in args and not any(t in p.stdout + p.stderr for t in GPU_TOKENS):
            raise SystemExit("did not run on the GPU: " + " ".join(cmd))
        imgs.append(read_pfm(pfm))
    return imgs


def stats(imgs, m):
    """The mean colour over mask `m` of the seeds' average, and its standard error per channel:
    the larger of the estimate from the seed means and the per-pixel one."""
    a, b = imgs[0][m], imgs[1][m]
    d = a - b
    mean = (a.mean(0) + b.mean(0)) / 2
    err = np.maximum(np.abs(a.mean(0) - b.mean(0)) / 2, np.sqrt((d * d).sum(0)) / (2.0 * len(d)))
    return mean, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_rough_irid"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True

    clear = os.path.join(out, "clear.ftsl")
    for rough, want in (("0.2", False), ("0", True)):
        with open(clear, "w", encoding="utf-8", newline="") as f:
            f.write(CLEAR % rough)
        p = subprocess.run([exe, "-in", clear, "-parseonly"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        loaded = p.returncode == 0
        good = loaded if want else ("needs an opaque" in p.stdout + p.stderr and not loaded)
        print("clear coating, roughness %s: %s%s" % (rough, "loads" if loaded else "refused",
                                                    "" if good else " -- FAIL"))
        ok &= good

    if not a.no_gpu and (not a.only or "coat" in a.only.split(",")):
        coat = os.path.join(out, "coat.ftsl")
        with open(coat, "w", encoding="utf-8", newline="") as f:
            f.write(COAT)
        m = masks()[0][1]                    # the first sphere, now car paint
        args = ["-mode", "B", "-n", "60000000"]
        mc = stats(render(exe, out, coat, "coat_B_cpu", args + ["-device", "cpu"]), m)
        mg = stats(render(exe, out, coat, "coat_B_gpu", args + ["-device", "gpu"]), m)
        ratio = mg[0] / np.maximum(mc[0], 1e-30)
        z = np.abs(mg[0] - mc[0]) / np.maximum(np.hypot(mg[1], mc[1]), 1e-30)
        bad = (z > 5.0) & (np.abs(ratio - 1.0) > 0.03)
        ok &= not bad.any()
        print("coat   car-paint sphere, mode B, GPU / CPU %s%s"
              % (" ".join("%.3f" % x for x in ratio), " FAIL" if bad.any() else ""))

    scene, scene_a = os.path.join(out, "spheres.ftsl"), os.path.join(out, "spheres_lens.ftsl")
    for path, text in ((scene, SCENE), (scene_a, SCENE_A)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    ms = masks()
    rest = ms[-1][1]
    cos4 = frame()[1][..., None]
    ref = render(exe, out, scene, REF[0], REF[1])
    rs = [stats(ref, m) for _, m in ms]
    print("%-6s %s" % ("ref", "   ".join("%s %s" % (n, np.array2string(s[0], precision=5))
                                         for (n, _), s in zip(ms, rs))))
    for tag, args in CASES:
        if (a.only and tag not in a.only.split(",")) or (a.no_gpu and "gpu" in args):
            continue
        lens = "A" in args
        rescale = lens or tag == "M_gpu"   # MODEM-GPU-OFFSET: see the docstring
        imgs = render(exe, out, scene_a if lens else scene, tag, args)
        if lens:   # irradiance to radiance: undo the vignetting
            imgs = [x / cos4 for x in imgs]
        if rescale:   # then the camera equation (mode A), or the device's offset (GPU mode M)
            k = sum(ref[i][rest].sum() for i in (0, 1)) / sum(x[rest].sum() for x in imgs)
            imgs = [x * k for x in imgs]
        cells = []
        for (name, m), (m0, e0) in zip(ms, rs):
            if rescale and name == "rest":
                continue           # matched by construction
            m1, e1 = stats(imgs, m)
            ratio = m1 / np.maximum(m0, 1e-30)
            z = np.abs(m1 - m0) / np.maximum(np.hypot(e1, e0), 1e-30)
            bad = (z > 5.0) & (np.abs(ratio - 1.0) > 0.03)
            ok &= not bad.any()
            cells.append("%s %s%s" % (name, " ".join("%.3f" % x for x in ratio),
                                      " FAIL" if bad.any() else ""))
        print("%-6s %s" % (tag, "   ".join(cells)), flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
