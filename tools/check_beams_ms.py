"""Measure the fog-ball disc in scenes/_beams_ms.ftsl renders.

The whole-frame mean hides the thing under test: multiple scattering only changes what
comes out of the medium, and the medium is a 36-px disc in a 192-px frame. This reports
the mean inside a 25-px disc centred on the ball's projection, the frame outside it, and
the ratio to the reference, so `mode M -beams` vs `mode D` is a single number.

Usage:  python scraps/_ms_ball.py ref.pfm a.pfm [b.pfm ...]
"""
import numpy as np, sys


def read_pfm(path):
    with open(path, "rb") as f:
        hdr = f.readline().strip()
        assert hdr in (b"PF", b"Pf"), hdr
        ch = 3 if hdr == b"PF" else 1
        w, h = map(int, f.readline().split())
        scale = float(f.readline().strip())
        data = np.frombuffer(f.read(w * h * ch * 4),
                             dtype="<f4" if scale < 0 else ">f4")
        return np.flipud(data.reshape(h, w, ch)).astype(np.float64)


imgs = [(p, read_pfm(p)) for p in sys.argv[1:]]
h, w, _ = imgs[0][1].shape
# Ball at (0.5,0.42,0.5) r=0.30; eye (0.5,0.5,2.7) -> look_at (0.5,0.5,0.5); fov_y 40.
yy, xx = np.mgrid[0:h, 0:w]
cx, cy = w * 0.5, h * 0.5 + h * (np.tan(np.radians(2.08)) / np.tan(np.radians(20.0))) * 0.5
disc = ((xx - cx) ** 2 + (yy - cy) ** 2) < (0.65 * 36) ** 2   # inner 65% of the ball
# Pixels that are NaN in ANY image are dropped from every image, so all rows compare the
# same pixel set. (Mode D on this scene emits a small number of NaN samples -- logged in
# known-issues.md; a silent nanmean would let the images disagree about their own support.)
good = np.ones((h, w), bool)
for p, im in imgs:
    n = np.isnan(im).any(axis=2)
    if n.any():
        print(f"note: {p.split('/')[-1]} has {n.sum()} NaN pixels -- excluded from every row")
    good &= ~n
disc &= good
rest = (~disc) & good
ref = imgs[0][1].mean(axis=2)
print(f"disc: centre ({cx:.0f},{cy:.0f}) r={0.65*36:.0f}px, {disc.sum()} px of {h*w}")
print(f"{'image':28s} {'ball disc':>11s} {'vs ref':>8s} {'rest of frame':>14s} {'vs ref':>8s}")
for p, im in imgs:
    L = im.mean(axis=2)
    din, dout = L[disc].mean(), L[rest].mean()
    rin, rout = ref[disc].mean(), ref[rest].mean()
    print(f"{p.split('/')[-1]:28s} {din:11.5g} {din/rin:8.4f} {dout:14.5g} {dout/rout:8.4f}")
