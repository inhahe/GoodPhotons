"""LOOK at a cap's caustic instead of only metering it.

`_capchroma.py` says the axicon's caustic is coloured (spread 0.185 on a 0.048 speckle floor,
fan 0.46).  By eye, in the finished 1280x720 frame, the cusps read whitish.  Both can be true:
the cap is ~380 px wide on screen and the caustic is a pair of thin cusps inside it, so at 1:1
the colour is there but subtends a few pixels.  This crops the cap's screen footprint out of
the FLOAT buffer and prints it three ways, stacked, upscaled:

  1. AS SHIPPED      -- linear x GAIN, sRGB encode.  Exactly what the PNG shows.
  2. UNDER-EXPOSED   -- gain chosen so the cap's PEAK lands just under white.  If row 1 is
                       white and row 2 is coloured, the colour is real and the tone map is
                       eating it; if row 2 is white too, there is no colour to eat.
  3. CHROMATICITY    -- every pixel renormalised to the same luminance and its saturation
                       stretched.  This is `fan` made visible: dispersed colour varies
                       SMOOTHLY with position, speckle is confetti.  Luminance is discarded,
                       so this row says nothing about brightness by design.

Usage:  python scraps/_capcrop.py png/rain_axicon.pfm axicon [more caps ...]
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw                                                   # noqa: E402
import _capchroma as C                                                             # noqa: E402
from _pfm import Linear                                                            # noqa: E402

ZOOM = int(os.environ.get('CAPZOOM', '3'))
SAT = float(os.environ.get('CAPSAT', '3.0'))     # chromaticity-row saturation stretch


def srgb(v):
    v = max(0.0, min(1.0, v))
    return int(255 * (1.055 * v ** (1 / 2.4) - 0.055 if v > 0.0031308 else 12.92 * v) + 0.5)


def footprint(name, w, h):
    """Screen bounding box of the cap's top face, padded a little.

    `CAPBOX=x0,z0,x1,z1` narrows it to a world-space window on that same plane, which is how
    you get onto a single cusp: the whole cap is 388 px wide here and the caustic is a pair of
    thin arcs inside it, so the full footprint at any readable zoom is mostly bare screen.
    """
    c, sz = C.caps[name]
    top = c[1] + sz[1] / 2
    corners = ((-.5, -.5), (-.5, .5), (.5, .5), (.5, -.5))
    if os.environ.get('CAPBOX'):
        a, b, d, e = (float(v) for v in os.environ['CAPBOX'].split(','))
        pts = [C.project((x, top, z), w, h) for x in (a, d) for z in (b, e)]
    else:
        pts = [C.project((c[0] + sx * sz[0], top, c[2] + sz_ * sz[2]), w, h)
               for sx, sz_ in corners]
    x0 = max(0, int(min(p[0] for p in pts)) - 4)
    x1 = min(w, int(max(p[0] for p in pts)) + 5)
    y0 = max(0, int(min(p[1] for p in pts)) - 4)
    y1 = min(h, int(max(p[1] for p in pts)) + 5)
    return x0, y0, x1, y1


def rows(im, name):
    w, h = im.w, im.h
    x0, y0, x1, y1 = footprint(name, w, h)
    cw, ch = x1 - x0, y1 - y0
    px = [[im.at(x, y) for x in range(x0, x1)] for y in range(y0, y1)]

    # The peak is taken from a 2x2 box, not a raw pixel: one firefly would otherwise set the
    # under-exposed row's gain and black out the whole crop.
    peak = 1e-9
    for y in range(ch - 1):
        for x in range(cw - 1):
            peak = max(peak, sum(0.2126 * px[y+j][x+i][0] + 0.7152 * px[y+j][x+i][1]
                                 + 0.0722 * px[y+j][x+i][2]
                                 for j in (0, 1) for i in (0, 1)) / 4.0)

    out = []
    for label, gain in (('as shipped (gain 1.5)', C.GAIN),
                        (f'under-exposed (gain {0.95/peak:.2f})', 0.95 / peak)):
        img = Image.new('RGB', (cw, ch))
        img.putdata([tuple(srgb(c * gain) for c in px[y][x])
                     for y in range(ch) for x in range(cw)])
        out.append((label, img))

    img = Image.new('RGB', (cw, ch))
    data = []
    for y in range(ch):
        for x in range(cw):
            p = px[y][x]
            s = sum(p)
            if s <= 1e-9:
                data.append((0, 0, 0))
                continue
            # renormalise to equal luminance, then push each channel away from grey by SAT
            q = [c / s * 3.0 for c in p]
            data.append(tuple(srgb(0.5 * (1.0 + SAT * (c - 1.0))) for c in q))
    img.putdata(data)
    out.append((f'chromaticity only (sat x{SAT:g})', img))
    return out


if __name__ == '__main__':
    src = sys.argv[1]
    im = Linear(src)
    if im.clipped:
        print(im.banner())
    for arg in sys.argv[2:]:
        name = arg if arg in C.caps else f'stand_{arg}_cap'
        rs = rows(im, name)
        cw, ch = rs[0][1].size
        pad, bar = 6, 16
        sheet = Image.new('RGB', (cw * ZOOM + 2 * pad,
                                  len(rs) * (ch * ZOOM + bar + pad) + pad), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for i, (label, img) in enumerate(rs):
            y = pad + i * (ch * ZOOM + bar + pad)
            d.text((pad, y + 3), f'{name}  --  {label}', fill=(210, 210, 210))
            sheet.paste(img.resize((cw * ZOOM, ch * ZOOM), Image.NEAREST), (pad, y + bar))
        dst = f'png/crop_{name}.png'
        sheet.save(dst)
        print(f'wrote {dst}  ({cw}x{ch} px cap footprint, {ZOOM}x)')
