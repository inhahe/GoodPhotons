"""Regression check: mode J's colour matches mode R's in a medium whose extinction varies with
wavelength.

    python tools/check_jchroma.py [--out png/check_jchroma] [--exe ftrace.exe]
                                  [--variants gpu,gpu-jsurf,cpu] [--seeds 3] [--tol 0.008]
                                  [--jspp 2048] [--epochs 256] [--reuse-ref]

A mode-J merge weight is a ratio of the densities with which the competing techniques would make
one path, and must read them all at that path's one wavelength. Its light half is the beam's or
photon's; its camera half carries a medium's sigma_t * Tr in every beam eta, and read those at the
camera's, so in a chromatic medium the weights of a merged path did not sum to one and the frame
tilted in colour (known-issues UPBP-W-CHROMATIC). Since 0.434.19 the camera half is read at the
merged light's wavelength (bdpt.h ChromaMis, the device's DChromaRun).

The scene is scraps/jspec2/hetmed's, written here: a grey box filled with a heterogeneous fog
whose absorption rises from red to blue (`sigma_a rgb 0.15 0.45 0.9`), lit by a small ceiling
light, at 64x64. The reference is mode R at 32768 spp, which agrees with mode D there to 0.1-0.4 %
per channel and costs a minute or two where mode D costs a quarter of an hour; mode J renders 2048
spp (beam merges only) on each of --seeds seeds. The colour must not tilt: X and Z of J/R, each
over Y's, within --tol -- before the fix X over Y read 0.987 and Z over Y 1.023 against mode D.
Mode J's own small bias (about -0.4 %, J-FOGCORNELL's kernel blur) is shared by the three channels
and cancels there; the frame's Y must agree within 1.5 %. About 25 minutes on the GPU at the
defaults: run it after a change to mode J's weights.

On the GPU a J render's light-side epochs are pinned (--epochs, `-beamepochs`; 256 is about what
the clock chose at 2048 spp): the device ends them on the clock otherwise, so the verdict moved
with the machine's speed -- one binary failed one run and passed another (known-issues
JCHROMA-CLOCK). Pinned, a binary's verdict is the same every run. It is still a sample: a seed's X
and Z over Y scatter by 0.5-1 %, the camera side's noise (neither 1024 epochs nor radii held still,
`-vcmalpha 1`, move it), so the check prints each seed and the seeds' standard error, and says so
when a ratio lies within two of them of --tol: more --seeds decide, since a change that moves the
random streams draws again.
--variants picks what is held: `gpu` (the default; beam merges), `gpu-jsurf` (point merges too,
whose weights read the same tables: SurfMergeWeight, DSurfMergeW) and `cpu` (the host's
ChromaMis; beam merges), each against mode R on its own device, about 25 minutes apiece. In this
scene the point merges carry about 1e-5 of the frame, so `gpu-jsurf` reads what `gpu` does to the
fourth digit and does not hold their weights (known-issues JCHROMA-CLOCK). The CPU's
mode J reads X over Y about 1.3 % lower than the GPU's here with the fix on or off -- a CPU-only
bias of its own (known-issues J-FOG-REALCAM) -- so `cpu` is held to 1.5 %: with the fix it reads
0.990 / 1.003, without 0.971 / 1.019 (the GPU: 1.002 / 0.994 and 0.984 / 1.016).
`--reuse-ref` keeps the references from an earlier run in --out, so a second run -- with
FTRACE_JCHROMA=0 in the environment, say, which reads the weights at the camera's wavelength as
before the fix and fails -- renders only mode J.

Exits 1 on any failure.
"""
import argparse
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

SCENE = NL.join([
    "scene { units meters  spectral 360 830 1 }",
    'material "grey" { type diffuse reflect 0.6 }',
    "quad { origin 0 0 0   u 1 0 0   v 0 0 1   material grey }",
    "quad { origin 0 1 0   u 0 0 1   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 1 0   v 1 0 0   material grey }",
    "quad { origin 0 0 0   u 0 0 1   v 0 1 0   material grey }",
    "quad { origin 1 0 0   u 0 1 0   v 0 0 1   material grey }",
    "medium {",
    "    sigma_a rgb 0.15 0.45 0.9   sigma_s 0.5   g 0.0",
    "    bounds { min 0 0 0   max 1 1 1 }",
    '    density "0.4 + 0.6*saturate(1.2 - 2.0*sqrt((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2))"',
    "}",
    "light area {",
    "    origin 0.35 0.999 0.35   u 0.3 0 0   v 0 0 0.3   normal 0 -1 0",
    "    spd preset:bb6500",
    "}",
    'camera "cam" {',
    "    eye 0.5 0.5 2.7   look_at 0.5 0.5 0.5   up 0 1 0   fov_y 40",
    "    film { res 64 64 }",
    "}"]) + NL


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


def channel_means(path):
    w, h, nc, data = read_pfm(path)
    return [sum(data[c::nc]) / (w * h) for c in range(nc)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_jchroma"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--variants", default="gpu")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.008)
    ap.add_argument("--rspp", type=int, default=32768)
    ap.add_argument("--jspp", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=256,
                    help="light-side epochs of a GPU mode-J render (-beamepochs)")
    ap.add_argument("--reuse-ref", action="store_true")
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    ftsl = os.path.join(d, "jchroma.ftsl")
    with open(ftsl, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)

    def render(dev, tag, mode, seed, spp, extra=()):
        png = os.path.join(d, "%s_%s_%d.png" % (tag, mode, seed))
        cmd = [exe, "-in", ftsl, "-mode", mode, "-device", dev, "-max-bounce", "8",
               "-no-meter", "-hdr", "-seed", str(seed), "-spp", str(spp), "-window-min",
               "-o", png] + list(extra)
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        log = p.stdout + p.stderr
        if p.returncode != 0:
            print(log[-1500:])
            raise SystemExit("render failed: " + " ".join(cmd))
        if dev == "gpu" and "on GPU" not in log:
            print(log[-1500:])
            raise SystemExit("mode %s did not run on the GPU" % mode)
        print("%s: mode %s seed %d done" % (tag, mode, seed), flush=True)
        return channel_means(png[:-4] + ".pfm")

    # device, mode-J flags, colour tolerance (the CPU's: J-FOG-REALCAM, above)
    VARIANTS = {"gpu": ("gpu", ("-nojsurf",), a.tol), "gpu-jsurf": ("gpu", ("-jsurf",), a.tol),
                "cpu": ("cpu", ("-nojsurf",), max(a.tol, 0.015))}
    fails = []
    for tag in a.variants.split(","):
        dev, jflags, ctol = VARIANTS[tag]
        ref_pfm = os.path.join(d, "%s_R_1.pfm" % dev)
        ref = (channel_means(ref_pfm) if a.reuse_ref and os.path.exists(ref_pfm)
               else render(dev, dev, "R", 1, a.rspp))
        # The device ends mode J's epochs on the clock unless they are pinned (JCHROMA-CLOCK).
        epochs = ("-beamepochs", str(a.epochs)) if dev == "gpu" else ()
        per = [render(dev, tag, "J", sd, a.jspp,
                      ("-beamcount", "100000", "-beamk", "2") + jflags + epochs)
               for sd in range(1, a.seeds + 1)]
        j = [sum(m[c] for m in per) / len(per) for c in range(3)]
        ratios = [j[c] / ref[c] for c in range(3)]
        print("%s: J/R  X %.4f  Y %.4f  Z %.4f" % ((tag,) + tuple(ratios)))

        def of(m, c):   # one J render's ratio to R: X or Z over Y, or Y itself
            r = [m[k] / ref[k] for k in range(3)]
            return r[c] / r[1] if c != 1 else r[1]

        for sd, m in enumerate(per, 1):
            print("%s: seed %d  X over Y %.4f  Z over Y %.4f  Y %.4f"
                  % (tag, sd, of(m, 0), of(m, 2), of(m, 1)))
        for name, c, tol in (("X over Y", 0, ctol), ("Z over Y", 2, ctol), ("Y", 1, 0.015)):
            r = ratios[c] / ratios[1] if c != 1 else ratios[1]
            ok = abs(r - 1.0) <= tol
            v = [of(m, c) for m in per]
            mu = sum(v) / len(v)
            se = (sum((x - mu) ** 2 for x in v) / (len(v) - 1) / len(v)) ** 0.5 if len(v) > 1 else 0.0
            print("%s: %-9s %.4f  (tol %.3f, the seeds' standard error %.4f)  %s"
                  % (tag, name, r, tol, se, "ok" if ok else "FAIL"))
            if abs(abs(r - 1.0) - tol) < 2.0 * se:
                print("%s: %-9s ...within two standard errors of the tolerance: more --seeds decide"
                      % (tag, name))
            if not ok:
                fails.append("%s %s" % (tag, name))
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
