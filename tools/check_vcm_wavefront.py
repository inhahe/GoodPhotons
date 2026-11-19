"""Regression check: GPU mode U's queued merges render what the inline merge renders, in waves the
queue holds.

    python tools/check_vcm_wavefront.py [--out png/check_vcm_wavefront] [--exe ftrace.exe]

kVcmCameraT queues each camera vertex's merge (DVcmMergeQueue) for kVcmMergeGather, which gathers a
site with a whole warp and adds it to the accumulator (render_cuda.cu, VCM-WAVEFRONT). The light
vertices, their weights, the footprint's probes and the pixel's terms are the inline merge's
(FTRACE_NOVCMWF=1); only the order of the sums differs, which moves a float32 pixel by a rounding
at most (ORDER_TOL). So it requires, against the inline merge's image of the same scene:

  - in scenes/_merge_glossy_bowl.ftsl at five times its merge radius (`-pmradius 0.05`, a wide
    merge, each site gathering many light vertices): every site through the queue (the
    run's merge line, none inline) and the inline merge's image; the inline run itself printing no
    merge line, i.e. bypassing the queue;
  - the same image with the queue shrunk to 32 requests (FTRACE_VCMWFCAP), which turns sites away
    to the inline merge every wave;
  - the waves' sizes (vcmSessionPass): with the queue shrunk to 4096 requests and
    FTRACE_WAVE_DEBUG's line a wave, every pass split into several waves, and no wave after the
    first pass turning a site away -- each is sized by the last pass's profile of its pixels;
  - the same of a frame whose top half is sky over a glossy floor (512 requests): a wave of sky
    queues nothing, and the floor's waves beside it must still hold their sites, every pass after
    the first -- a wave sized by the wave before it would take the rest of the pass after the sky;
  - the bowl with a ball of ruby glass in it, seen through by the camera, at one wavelength a
    path (`-heroc 1`: the kernels' other instantiation; the bowl's runs take the default hero
    bundle, whose light vertices' secondary rows the merge reads): the walks behind the glass carry
    a VCM-SPEC carrier, which rides in the queue's side table -- a site whose table the queue
    lacked would be merged inline -- and the inline merge's image.

Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fogbox  # noqa: E402
from pfm import ORDER_TOL, image_diff, read_pfm  # noqa: E402

LINE = re.compile(r"mode U merges: (\d+) sites gathered a warp each \(([\d.]+) a path\) "
                  r"in [\d.]+ s of GPU time, (\d+) inline")
WAVE = re.compile(r"\[wave\] U pass (\d+) paths (\d+)\.\.(\d+) \(\d+\), (\d+) merge sites "
                  r"\((\d+) turned away\)")

RUBY = """scene { units meters }
material "rough" { type glossy reflect 0.7 roughness 1.2 }
material "ruby"  { type dielectric ior 1.5  absorb rgb 0.4 6.0 7.0 }
sphere { center 0 0 0  radius 0.3  material rough }
sphere { center 0 -0.1 -0.12  radius 0.08  material ruby }
light area { origin -0.05 0.2 -0.05  u 0.1 0 0  v 0 0 0.1  normal 0 -1 0  power 5  spd preset:d65 }
camera "c" { eye 0 0.05 0.2  look_at 0 -0.1 -0.2  up 0 1 0  fov_y 60  film { res 96 72 } }
"""
# A glossy floor a metre square, lit from above, seen level from its edge: the frame's top half is sky.
SKY = """scene { units meters }
material "rough" { type glossy reflect 0.7 roughness 1.2 }
quad { origin -0.5 0 -0.5  u 0 0 1  v 1 0 0  material rough }
light area { origin -0.1 0.6 -0.1  u 0.2 0 0  v 0 0 0.2  normal 0 -1 0  power 5  spd preset:d65 }
camera "c" { eye 0 0.25 1.2  look_at 0 0.25 0  up 0 1 0  fov_y 50  film { res 96 72 } }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_vcm_wavefront"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    ruby, sky = os.path.join(d, "ruby.ftsl"), os.path.join(d, "sky.ftsl")
    for path, text in ((ruby, RUBY), (sky, SKY)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    BOWL = [os.path.join(ROOT, "scenes", "_merge_glossy_bowl.ftsl"), "-spp", "4", "-max-bounce", "8",
            "-pmradius", "0.05"]
    RUBYARGS = [ruby, "-spp", "4", "-max-bounce", "8", "-pmradius", "0.03", "-heroc", "1"]
    check = fogbox.Checks(62)
    logs = []

    def run(tag, args, env):
        png = os.path.join(d, tag + ".png")
        for stale in (png, png[:-4] + ".pfm"):
            if os.path.exists(stale):
                os.remove(stale)
        cmd = [exe, "-in"] + args + ["-mode", "U", "-device", "gpu", "-seed", "3", "-hdr", "-no-meter",
                                     "-window-min", "-o", png]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=dict(os.environ, **env))
        log = p.stdout + p.stderr
        logs.append("--- " + tag + NL + log[-600:])
        pfm = png[:-4] + ".pfm"
        return p.returncode, log, (read_pfm(pfm, top_down=False) if os.path.exists(pfm) else None)

    def pair(name, args):
        """The inline merge's image and the queue's; checks both ran as asked. Returns the first."""
        rc0, log0, inline = run(name + "_inline", args, {"FTRACE_NOVCMWF": "1"})
        rc1, log1, queued = run(name + "_queued", args, {})
        m = LINE.search(log1)
        check(name + ": the inline merge renders, bypassing the queue",
              rc0 == 0 and inline is not None and LINE.search(log0) is None,
              "rc %d%s" % (rc0, ", a merge line" if LINE.search(log0) else ""))
        check(name + ": the queue renders, every site through it",
              rc1 == 0 and queued is not None and m is not None and int(m.group(1)) > 0
              and m.group(3) == "0",
              ("%s sites, %s a path, %s inline" % m.groups()) if m else "rc %d, no merge line" % rc1)
        if inline is not None and queued is not None:
            r, n = image_diff(inline, queued)
            check(name + ": ...the image the inline merge's", r < ORDER_TOL,
                  "largest change %.2g of the pixel, %d values differ" % (r, n))
        return inline

    inline = pair("bowl", BOWL)
    rc2, log2, tiny = run("bowl_tiny", BOWL, {"FTRACE_VCMWFCAP": "32"})
    m2 = LINE.search(log2)
    check("bowl: a 32-request queue turns sites away inline",
          rc2 == 0 and m2 is not None and int(m2.group(3)) > 0,
          ("%s sites, %s inline" % (m2.group(1), m2.group(3))) if m2 else "rc %d" % rc2)
    if inline is not None and tiny is not None:
        r, n = image_diff(inline, tiny)
        check("bowl: ...and renders the same image", r < ORDER_TOL,
              "largest change %.2g of the pixel, %d values differ" % (r, n))

    def sized(name, args, cap):
        """With the queue shrunk to `cap` requests: every pass in waves, none after the first
        turning a site away."""
        rc, log, _ = run(name + "_sized", args, {"FTRACE_VCMWFCAP": str(cap), "FTRACE_WAVE_DEBUG": "1"})
        waves = [tuple(int(g) for g in m.groups()) for m in WAVE.finditer(log)]
        passes = sorted({w[0] for w in waves})
        check("%s: a %d-request queue splits every pass into waves" % (name, cap),
              rc == 0 and len(passes) == 4 and all(sum(1 for w in waves if w[0] == p) >= 2 for p in passes),
              "rc %d, %d waves in %d passes" % (rc, len(waves), len(passes)))
        later = [w for w in waves if w[0] > 0]
        check("...none after the first pass turning a site away",
              len(later) > 0 and sum(w[4] for w in later) == 0 and sum(w[3] for w in later) > 0,
              "%d of %d sites turned away after the first pass (in it: %d of %d)"
              % (sum(w[4] for w in later), sum(w[3] for w in later),
                 sum(w[4] for w in waves if w[0] == 0), sum(w[3] for w in waves if w[0] == 0)))

    sized("bowl", BOWL, 4096)
    sized("sky", [sky, "-spp", "4", "-max-bounce", "8", "-pmradius", "0.1"], 512)

    pair("ruby", RUBYARGS)
    return check.verdict(NL.join(logs[-3:]))


if __name__ == "__main__":
    sys.exit(main())
