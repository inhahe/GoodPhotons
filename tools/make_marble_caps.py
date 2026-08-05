"""Turn the raw `marble texture *.{jpg,jpeg,avif}` drops into cap textures the gallery can use.

WHY THIS EXISTS AT ALL, i.e. why the raw images cannot just be pointed at:

  A cap in `gallery_rain` is not decoration, it is the caustic SCREEN, and its albedo is the
  single most-tuned number in the scene (0.15 — see the `capwhite` comment in the scene, and
  the clip analysis in known-issues.md).  The raw marble photos mean 0.62-0.93 in sRGB, i.e.
  0.34-0.85 LINEAR: dropping one in as-is raises a cap's albedo by up to 5.7x, which is two
  and a half stops back INTO the clip that the 0.15 was chosen to escape.  Every caustic in
  the scene would go back to printing as flat #FFFFFF.  So the one non-negotiable operation
  here is: rescale each image so its MEAN LINEAR REFLECTANCE equals the albedo the cap had
  before it was textured.  The stone's pattern survives; its brightness does not.

  Two further knobs exist because a caustic on a diffuse surface is MULTIPLICATIVE
  (pixel = albedo(x) * irradiance(x)), so the stone modulates the caustic directly:

    contrast `k`   -- albedo swing about the mean, in linear light.  A vein at half the mean
                      albedo halves the caustic that lands on it; a caustic crossing a dark
                      vein is literally erased there.  k < 1 compresses the swing.
    saturation `s` -- a coloured stone TINTS the caustic, which attacks the exact quantity
                      (`sat`, `spread`) the axicon exists to produce.  s < 1 pulls the stone
                      toward neutral so the colour in the picture is the light's, not the rock's.

  Both default to 1 (untouched).  They are turned down only on the three caps that actually
  catch a measured caustic, and left alone everywhere else — see CAPS below, where the
  assignment is by measured caustic strength, not by taste.

Usage:  python tools/make_marble_caps.py            # write textures/marble_*.png
        python tools/make_marble_caps.py --report   # just print what it would do
"""
import os, sys, glob

from PIL import Image

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SRC_DIR, 'textures')

TARGET_ALBEDO = 0.15      # `capwhite`'s reflect — the number the whole caustic study rests on
MAXDIM = 1024             # a 1.6 m cap spans <650 px on screen; 4650^2 sources are pure waste


def srgb_to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin_to_srgb(c):
    c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


_S2L = [srgb_to_lin(i / 255.0) for i in range(256)]

# name          source                     cap                    crop            k     s
# `crop` is (left, top, right, bottom) as FRACTIONS, applied before anything else.
# k = linear contrast about the mean, s = saturation.  1.0 = untouched.
#
# The three caps that catch a MEASURED caustic get the flattest, most neutral stone in the
# set and get it further calmed (k/s well below 1).  Ranking is from the float re-measurement
# in design.md: axicon 0.30 % coverage at sat 0.275 (and the only piece that makes COLOUR),
# crystal orb 0.72 % / 0.246, solid gyroid k=10 0.37 % / 0.208.  Everything else measures at
# or near zero — gold/brass/chrome are opaque, the heart and the jack are opaque iridescent,
# and the Klein bottle's 2.4 mm wall is optically a window (0.12 %, nothing at the 1.5x bar).
# Those caps are free, so they get the dramatic gold-veined stone, where it costs nothing and
# rhymes with the metal standing on it.
#
# Sources are also matched to cap SCREEN SIZE, because two of the drops are only ~200 px
# across and the caps span 72-634 px in the still.  Cap footprints (1280x720, still camera):
# gyroid 634x293, axicon 411x109, diamond 288x75, glass 225x70, klein 153x37, dumbbell
# 105x28, chrome 105x21, brass 91x19, heart 84x19, oil 72x16.  So the two 197x256 drops go
# to caps under 160 px and the 4650^2 and 1600x1067 sheets go to the two that matter.
CAPS = [
    # ---- caustic-critical: calm, pale, near-neutral -------------------------------------
    # #6 is the flattest AND most neutral sheet in the set (contrast 0.18, chroma 0.003), so
    # it goes under the one piece whose entire purpose is to put COLOUR on a cap.
    ('marble_axicon',  'marble texture 6.jpg',    'stand_axicon_cap',   None,           0.45, 0.30),
    ('marble_orb',     'marble texture 7.jpg',    'stand_glass_cap',    None,           0.50, 0.35),
    ('marble_gyroidx', 'marble texture 3.jpg',    'stand_diamond_cap',  None,           0.55, 0.40),
    # ---- mild: no measured caustic, but keep them quiet enough to read as a set ---------
    ('marble_klein',   'marble texture 4.jpg',    'stand_klein_cap',    None,           0.75, 0.70),
    ('marble_heart',   'marble texture 5.jpg',    'stand_heart_cap',    None,           0.85, 0.80),
    ('marble_oil',     'marble texture 3.6.jpg',  'stand_oil_cap',      None,           0.85, 0.80),
    # ---- free: opaque / metal pieces, full drama ----------------------------------------
    # The gold-veined sheets go under the GOLD gyroid and the BRASS cluster on purpose: the
    # centrepiece cap is the biggest surface in the frame and the vein colour rhymes with the
    # metal standing on it, at zero caustic cost because both pieces are opaque.
    ('marble_gold',    'marble texture 1.jpeg',   'stand_gyroid_cap',   None,           1.00, 1.00),
    ('marble_brass',   'marble texture 8.jpg',    'stand_brass_cap',    None,           1.00, 1.00),
    # #2 carries a near-black grey field beside its gold; at full contrast it bottoms out at
    # 0.006 linear (sRGB 19), which reads as a HOLE in the tabletop rather than dark stone.
    # k=0.75 lifts the floor without touching the veins.
    ('marble_chrome',  'marble texture 2.jpg',    'stand_chrome_cap',   None,           0.75, 1.00),
    # `marble texture 3.5.avif` is a WATERMARKED VectorStock preview: a black footer bar with
    # the agency name and stock id runs across the bottom ~9 % (it is why that file measured
    # p5 = 0.000 while every other sheet floors around 0.45).  The stone above the bar is
    # clean, so the bar is cropped off here — but the file is still a watermarked comp, so
    # this is flagged in the scene and in known-issues.md rather than silently shipped.
    ('marble_dumbbell', 'marble texture 3.5.avif', 'stand_dumbbell_cap', (0, 0, 1, 0.905), 1.00, 1.00),
]


def build(name, src, crop, k, s, report=False):
    path = os.path.join(SRC_DIR, src)
    try:
        im = Image.open(path).convert('RGB')
    except Exception as e:                       # AVIF needs pillow-avif-plugin
        print(f'  !! {src}: {e}')
        return None
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    w, h = im.size
    if max(w, h) > MAXDIM:                       # box-filter down; keeps veins, drops the bulk
        sc = MAXDIM / max(w, h)
        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)

    px = list(im.getdata())
    lin = [(_S2L[r], _S2L[g], _S2L[b]) for r, g, b in px]
    n = len(lin)

    def luma(p):
        return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]

    mean0 = sum(luma(p) for p in lin) / n

    # 1) desaturate toward each texel's own luminance, 2) compress contrast about the MEAN
    #    (both in linear light, so they compose with the rescale below as pure multiplies),
    #    3) rescale the whole image so the mean linear luminance is exactly TARGET_ALBEDO.
    out = []
    for p in lin:
        y = luma(p)
        q = [y + (c - y) * s for c in p]          # saturation
        q = [mean0 + (c - mean0) * k for c in q]  # contrast about the original mean
        out.append(q)
    mean1 = sum(luma(p) for p in out) / n
    g = TARGET_ALBEDO / max(mean1, 1e-9)
    out = [[max(0.0, c * g) for c in p] for p in out]

    fin = [luma(p) for p in out]
    fin.sort()
    lo, hi = fin[int(0.02 * n)], fin[int(0.98 * n)]
    clipped = sum(1 for p in out for c in p if c > 1.0)
    stats = dict(name=name, src=src, size=im.size, mean_src=mean0,
                 lo=lo, hi=hi, swing=hi / max(lo, 1e-9), clipped=clipped)
    if report:
        return stats, None

    enc = Image.new('RGB', im.size)
    enc.putdata([tuple(int(255 * lin_to_srgb(c) + 0.5) for c in p) for p in out])
    return stats, enc


def main():
    report = '--report' in sys.argv
    if not report:
        os.makedirs(OUT_DIR, exist_ok=True)
    print(f'target mean LINEAR reflectance {TARGET_ALBEDO} (= capwhite), max dim {MAXDIM}\n')
    print(f'{"texture":18s} {"source":24s} {"size":11s} {"srcmean":>7s} '
          f'{"p2":>6s} {"p98":>6s} {"swing":>6s}  cap')
    for name, src, cap, crop, k, s, in CAPS:
        r = build(name, src, crop, k, s, report)
        if r is None:
            continue
        st, enc = r
        print(f'{name:18s} {src:24s} {st["size"][0]:4d}x{st["size"][1]:<5d} '
              f'{st["mean_src"]:7.3f} {st["lo"]:6.3f} {st["hi"]:6.3f} '
              f'{st["swing"]:6.2f}  {cap}')
        if enc is not None:
            dst = os.path.join(OUT_DIR, name + '.png')
            enc.save(dst)
    if not report:
        print(f'\nwrote {len(CAPS)} textures to textures/')


if __name__ == '__main__':
    main()
