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

On the GPU a J render's light-side epochs are pinned (--epochs, `-beamepochs`; by default
--jspp / 8, about what the clock chose: 234-329 at 2048 spp), so a binary's verdict is the same
every run and stays where it is should the device's work-count constants be refitted, which would
move the epochs the counts lay (known-issues JCHROMA-CLOCK, GPU-J-CLOCK). It is
still a sample: a render's X and Z over Y scatter by 0.5-1 %, from its camera samples and its
light-side realizations alike (one seed's ratios move by up to 2 % between 64, 256 and 1024
epochs, in no consistent direction; holding the radii still, `-vcmalpha 1`, moves them by 0.1 %).
So the check prints each seed and the seeds' standard error, and says when the tolerance lies
inside the seeds' 95 % interval: more --seeds decide, since a change that moves the random
streams draws again.
--variants picks what is held: `gpu` (the default; beam merges), `gpu-jsurf` (point merges too,
whose weights read the same tables: SurfMergeWeight, DSurfMergeW) and `cpu` (the host's
ChromaMis; beam merges), each against mode R on its own device, about 25 minutes apiece. In this
scene the point merges carry about 1e-5 of the frame, so `gpu-jsurf` reads what `gpu` does to the
fourth digit and does not hold their weights: tools/check_jsurfw.py holds those directly,
device against host (known-issues JCHROMA-JSURF-WEAK). The CPU's
mode J reads X over Y about 1.3 % lower than the GPU's here with the fix on or off -- a CPU-only
bias of its own (known-issues J-FOG-REALCAM) -- so `cpu` is held to 1.5 %: with the fix it reads
0.990 / 1.003, without 0.971 / 1.019 (the GPU at the defaults: 0.998 / 0.997 and 0.981 / 1.011).
`--reuse-ref` keeps the references from an earlier run in --out, so a second run -- with
FTRACE_JCHROMA=0 in the environment, say, which reads the weights at the camera's wavelength as
before the fix and fails -- renders only mode J.

Exits 1 on any failure.
"""
import argparse
import os
import statistics
import subprocess
import sys
from pfm import read_pfm_flat  # noqa: E402

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


def channel_means(path):
    w, h, nc, data = read_pfm_flat(path)
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
    ap.add_argument("--epochs", type=int, default=None,
                    help="light-side epochs of a GPU mode-J render (-beamepochs; default "
                         "--jspp / 8, about what the clock chose)")
    ap.add_argument("--reuse-ref", action="store_true")
    a = ap.parse_args()
    if a.epochs is None:
        a.epochs = max(1, a.jspp // 8)
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    ftsl = os.path.join(d, "jchroma.ftsl")
    with open(ftsl, "w", encoding="utf-8", newline="") as f:
        f.write(SCENE)

    def render(dev, tag, mode, seed, spp, extra=()):
        png = os.path.join(d, "%s_%s_%d.png" % (tag, mode, seed))
        for stale in (png, png[:-4] + ".pfm"):   # a render that writes nothing must not pass
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [exe, "-in", ftsl, "-mode", mode, "-device", dev, "-max-bounce", "8",
               "-no-meter", "-hdr", "-seed", str(seed), "-spp", str(spp), "-window-min",
               "-o", png] + list(extra)
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        log = p.stdout + p.stderr
        why = ("exit code %d" % p.returncode if p.returncode != 0 else
               "it was stopped (a closed window, or ftrace -stop)" if "[stop] interrupted" in log else
               "it ran on the clock: -beamepochs was ignored" if "-beamepochs ignored" in log else
               "it did not run on the GPU" if dev == "gpu" and "on GPU" not in log else
               "it wrote no image" if not os.path.exists(png[:-4] + ".pfm") else None)
        if why:
            print(log[-1500:])
            raise SystemExit("mode %s seed %d: %s: %s" % (mode, seed, why, " ".join(cmd)))
        print("%s: mode %s seed %d done" % (tag, mode, seed), flush=True)
        return channel_means(png[:-4] + ".pfm")

    # device, mode-J flags, colour tolerance (the CPU's: J-FOG-REALCAM, above)
    VARIANTS = {"gpu": ("gpu", ("-nojsurf",), a.tol), "gpu-jsurf": ("gpu", ("-jsurf",), a.tol),
                "cpu": ("cpu", ("-nojsurf",), max(a.tol, 0.015))}
    # Two-sided 95 % t quantiles by degrees of freedom (the seeds less one), for the standard
    # error of a handful of seeds.
    T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}
    refs = {}   # a device's mode-R reference, rendered once a run
    fails = []
    for tag in a.variants.split(","):
        dev, jflags, ctol = VARIANTS[tag]
        if dev not in refs:
            ref_pfm = os.path.join(d, "%s_R_1.pfm" % dev)
            refs[dev] = (channel_means(ref_pfm) if a.reuse_ref and os.path.exists(ref_pfm)
                         else render(dev, dev, "R", 1, a.rspp))
        ref = refs[dev]
        # The device ends mode J's epochs on the clock unless they are pinned (JCHROMA-CLOCK).
        epochs = ("-beamepochs", str(a.epochs)) if dev == "gpu" else ()
        per = [render(dev, tag, "J", sd, a.jspp,
                      ("-beamcount", "100000", "-beamk", "2") + jflags + epochs)
               for sd in range(1, a.seeds + 1)]
        j = [sum(m[c] for m in per) / len(per) for c in range(3)]
        if tag == "gpu-jsurf":
            print("gpu-jsurf: the point merges carry ~1e-5 of this frame, so this holds no more "
                  "than `gpu` does; tools/check_jsurfw.py holds their weights")

        def of(m, c):   # J channel means m against R: X or Z over Y, or Y itself
            r = [m[k] / ref[k] for k in range(3)]
            return r[c] / r[1] if c != 1 else r[1]

        print("%s: J/R  X %.4f  Y %.4f  Z %.4f" % ((tag,) + tuple(j[k] / ref[k] for k in range(3))))
        for sd, m in enumerate(per, 1):
            print("%s: seed %d  X over Y %.4f  Z over Y %.4f  Y %.4f"
                  % (tag, sd, of(m, 0), of(m, 2), of(m, 1)))
        for name, c, tol in (("X over Y", 0, ctol), ("Z over Y", 2, ctol), ("Y", 1, 0.015)):
            r = of(j, c)
            ok = abs(r - 1.0) <= tol
            v = [of(m, c) for m in per]
            se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
            print("%s: %-9s %.4f  (tol %.3f, the seeds' standard error %.4f)  %s"
                  % (tag, name, r, tol, se, "ok" if ok else "FAIL"))
            t = T95.get(len(v) - 1, 2.0)
            if len(v) > 1 and abs(abs(r - 1.0) - tol) < t * se:
                print("%s: %-9s ...the tolerance lies inside the seeds' 95 %% interval (%.2f "
                      "standard errors; the reference's own noise not counted): more --seeds "
                      "decide" % (tag, name, t))
            if not ok:
                fails.append("%s %s" % (tag, name))
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
