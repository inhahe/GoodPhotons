# Measure the raincloud in a gallery_rain render: crown vs underside vs the rain shaft.
#
# Used to choose the 0.199.0 cloud retune (see the `medium` comment in gallery_rain.ftsl).
# Usage:  python tools/check_cloud.py <ref.pfm> [<cmp.pfm> ...]   -- every row is ratioed
# against the FIRST file, and every row shares one pixel mask taken from the first file, so
# no image gets to define the cloud's silhouette differently from the others.
#
# The cloud is the only lit thing in the upper-right quadrant against black sky, so a
# fixed box plus a "not black" mask isolates it without needing the projection matrix.
# The point of the comparison is NOT the whole-frame mean (the cloud is a few percent of
# the frame) but the CROWN/BASE CONTRAST: the scene's header requires the underside to
# stay dark enough to serve as the bow's backdrop, so the number that decides whether the
# retune is acceptable is base brightness, and crown/base ratio, not total energy.
import sys, numpy as np


def readpfm(path):
    with open(path, "rb") as f:
        hdr = f.readline().strip()
        assert hdr in (b"PF", b"Pf"), hdr
        ch = 3 if hdr == b"PF" else 1
        while True:
            l = f.readline().strip()
            if l and not l.startswith(b"#"):
                break
        w, h = map(int, l.split())
        sc = float(f.readline().strip())
        d = np.fromfile(f, "<f4" if sc < 0 else ">f4", w * h * ch)
    return d.reshape(h, w, ch)[::-1]


def lum(im):
    return 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]


paths = sys.argv[1:]
imgs = [(p, readpfm(p)) for p in paths]
h, w, _ = imgs[0][1].shape

# Cloud box, in fractions of the frame (read off a coarse luminance grid of the 480x270
# render: the cloud occupies x 288..420, y 54..106, and nothing else in that box is lit).
x0, x1 = int(0.60 * w), int(0.88 * w)
y0, y1 = int(0.19 * h), int(0.40 * h)
box = np.zeros((h, w), bool)
box[y0:y1, x0:x1] = True

# "Lit" = above a floor set from the FIRST image's own frame statistics, applied to every
# image, so all rows share one pixel set and neither image gets to define the cloud's
# silhouette differently from the other. The floor is 5% of the first image's cloud-box
# 95th percentile -- well above the black sky, well below anything actually cloud.
ref0 = lum(imgs[0][1])
floor = 0.05 * np.nanpercentile(ref0[box], 95)
lit = box & (ref0 > floor)
for p, im in imgs:
    lit &= np.isfinite(im).all(axis=2)
ym = (y0 + y1) // 2
crown = lit.copy(); crown[ym:, :] = False
base = lit.copy(); base[:ym, :] = False

# The rain shaft directly under the cloud -- the bow's own volume, which the cloud lights.
shaft = np.zeros((h, w), bool)
shaft[int(0.41 * h):int(0.58 * h), int(0.63 * w):int(0.83 * w)] = True
for p, im in imgs:
    shaft &= np.isfinite(im).all(axis=2)

print(f"{w}x{h}   cloud px {lit.sum()} (crown {crown.sum()} / base {base.sum()})"
      f"   shaft px {shaft.sum()}")
# Chroma outliers: a lit cloud pixel whose max/min channel ratio is huge is not a noisy
# sample, it is a divide by a vanishing density -- a single-wavelength path that returned an
# absurd value and painted one channel. A white cloud lit by a blackbody should sit near 1;
# anything past 3:1 is suspect and anything past 10:1 is a firefly. This column is what
# known-issues.md's "1e29 fireflies" entry is measured with, so it lives here rather than in
# a throwaway script.
def chroma(im, mask):
    px = im[mask]
    mx = px.max(axis=1)
    mn = px.min(axis=1)
    ok = mn > 0.0
    r = np.full(mx.shape, 1.0)
    r[ok] = mx[ok] / mn[ok]
    r[~ok & (mx > 0.0)] = np.inf          # a channel at exactly 0 beside a lit one
    return int((r > 3.0).sum()), int((r > 10.0).sum()), float(r.max())


print(f"{'image':<26}{'crown':>12}{'base':>12}{'crown/base':>12}"
      f"{'shaft':>12}{'frame':>12}{'chroma>3':>10}{'>10':>7}{'peak':>12}")
ref = None
for p, im in imgs:
    L = lum(im)
    c, b = L[crown].mean(), L[base].mean()
    s = L[shaft].mean()
    fr = L[np.isfinite(L)].mean()
    n3, n10, peak = chroma(im, lit)
    print(f"{p.split('/')[-1]:<26}{c:12.4e}{b:12.4e}{c / b:12.3f}{s:12.4e}{fr:12.4e}"
          f"{n3:10d}{n10:7d}{peak:12.3g}")
    if ref is None:
        ref = (c, b, s, fr)
    else:
        print(f"{'  vs first':<26}{c / ref[0]:12.3f}{b / ref[1]:12.3f}"
              f"{'':>12}{s / ref[2]:12.3f}{fr / ref[3]:12.3f}")
