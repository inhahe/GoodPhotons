"""Meter a caustic ON A CAP IN THE FINISHED SCENE, not over a bare cap in a test rig.

`_gemsweep.py` floats one piece at a time over a bare cap lit ONLY by the sun.  The scene is
not that: `gallery_rain` also carries an 11000 K sky panel for fill, so every cap sits on a
pedestal of white light that the rig never had.  A caustic is only ever visible as a RATIO to
its screen, so fill raises the denominator and can wash out colour that measured beautifully
in isolation.  The only way to know how much is to meter the shipped frame.

Usage:  python scraps/_capchroma.py png/rain_axicon.pfm [cap ...]

FEED IT THE .pfm, NOT THE .png.  A PNG clamps at white, and a caustic is by definition the
brightest thing in frame, so its core prints as #FFFFFF with all three channels equal --
its colour is destroyed by the tone map before any meter sees it.  (In the first render of
the axicon exhibit 596 of a cap's 22639 pixels were pure white, more than half the caustic's
area.)  Render with `-hdr` and point this at the float sidecar; a PNG still works, but every
number it gives for a bright caustic is the tone map's opinion rather than the render's.

It reads the cap boxes straight out of the scene (spread-aware, same accumulation as
`_standaudit.py`), projects each one through the still camera's basis, and runs the same
metric `_gemsweep.py` uses -- 4x box in LINEAR light, then coverage above 2x the cap's own
median, excess-weighted saturation, and chromaticity `spread`.  Reporting several caps from
ONE frame is the point: they share an exposure, a sun and a fill, so the comparison between
them is exact even though the absolute numbers are not comparable with the rig's.
"""
import math, re, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pfm import Linear, scatter, fan, fmt                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))

DOWN = 4
CUT = float(os.environ.get('GEMCUT', '2.0'))

# What the tone map multiplies scene-linear radiance by before gamma, so `clip` below can say
# how much of each cap the DISPLAYED image throws away.  Metering the float sidecar fixed the
# measurement but not the picture: a caustic core blown to white looks white on screen too, and
# the whole reason the caps were darkened is to keep the peak under this.  1.5 = ftrace's
# ABS_EXPOSURE_GAIN (6.0) x the film's `exposure 0.25`; it is also confirmed empirically, by
# dividing unclipped PNG pixels by their .pfm counterparts.  Re-derive it if the scene's
# exposure changes -- ftrace prints it ("exposure=1.5") on every write.
GAIN = float(os.environ.get('CAPGAIN', '1.5'))

# The still camera, copied from the scene's prefer{} block.
EYE = (5.0, 2.95, 9.35)
LOOK = (5.0, 1.74, 3.3)
UP = (0.0, 1.0, 0.0)
FOV_Y = 52.0


def _norm(v):
    m = math.sqrt(sum(c * c for c in v))
    return tuple(c / m for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


FWD = _norm(tuple(LOOK[i] - EYE[i] for i in range(3)))
RIGHT = _norm(_cross(FWD, UP))
CAMUP = _norm(_cross(RIGHT, FWD))


def project(p, w, h):
    d = tuple(p[i] - EYE[i] for i in range(3))
    zc = sum(d[i] * FWD[i] for i in range(3))
    if zc <= 1e-6:
        return None
    ty = math.tan(math.radians(FOV_Y) / 2)
    xc = sum(d[i] * RIGHT[i] for i in range(3))
    yc = sum(d[i] * CAMUP[i] for i in range(3))
    return (w / 2 * (1 + xc / (zc * ty * (w / h))), h / 2 * (1 - yc / (zc * ty)))


# ---- pull the cap boxes out of the scene, spread-aware -------------------------------
src = open('scenes/gallery_rain.ftsl', encoding='utf-8').read()
BOX = re.compile(r'box\s*\{\s*center\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)'
                 r'\s+size\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)')
HDR = re.compile(r'isosurface\s+"([^"]+_cap)"\s*\{')
GRP = re.compile(r'group\s*\{')
TRN = re.compile(r'translate\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)')


def block(text, o):
    d, i = 0, o
    while i < len(text):
        if text[i] == '{':
            d += 1
        elif text[i] == '}':
            d -= 1
            if d == 0:
                return text[o + 1:i], i + 1
        i += 1
    raise ValueError('unbalanced braces')


groups, gp = [], 0
while (g := GRP.search(src, gp)):
    gs = g.end() - 1
    gb, ge = block(src, gs)
    gp = g.end()
    t = TRN.search(gb)
    b = gb.find('{')
    if t and (b < 0 or t.start() < b):
        groups.append((gs, ge, tuple(float(t.group(i)) for i in (1, 2, 3))))

caps, pos = {}, 0
while (h := HDR.search(src, pos)):
    name, hs = h.group(1), h.start()
    body, pos = block(src, h.end() - 1)
    off = [0.0, 0.0, 0.0]
    for s, e, d in groups:
        if s <= hs < e:
            for i in range(3):
                off[i] += d[i]
    m = BOX.search(body)
    c = [float(m.group(i)) + off[i - 1] for i in (1, 2, 3)]
    sz = [float(m.group(i)) for i in (4, 5, 6)]
    caps[name] = (c, sz)


# ---- the pieces themselves, which sit over the caps and so occlude them ---------------
# A cap's screen footprint is not all cap: the piece stands in the middle of it, and a gold
# gyroid or an oilslick jack is far more saturated than any caustic.  Metering the footprint
# raw measures the PIECE, not its caustic -- that is what first made stand_gyroid_cap read
# sat 0.462 / spread 0.426 (gold, not light).  So mask each piece out.
#
# The mask is a world-space solid tested against the EYE->cap-point segment, and the SHAPE
# of that solid matters more than it looks.  Sight lines to a cap graze low, just over the
# cap's own surface, so a mask is only sound where the piece is actually THICK down there:
#
#   * a bounding SPHERE is right for the blobby pieces, but wrong for the axicon -- r 0.63
#     is loose for a cone (the rim is far from the centre) and it projects to a 111 px disc
#     that swallows the middle 222 px of a 380 px cap, masking the caustic itself;
#   * the axicon's `contained_by` BOX is worse still: the cone is a needle at its apex but
#     the box is full width all the way down, exactly where the sight lines pass;
#   * so the axicon gets a real CONE, and only its own narrow lower half is masked.
#
# Spheres are the collider table from `scraps/_flyplan.py`; keep the two in step.
SPHERES = [
    ((5.00, 1.33, 6.05), 0.78),   # gold gyroid
    ((2.60, 1.75, 4.15), 0.50),   # crystal gyroid
    ((7.70, 1.70, 3.95), 0.50),   # glass orb
    ((2.60, 1.30, 1.55), 0.44),   # chrome ring
    ((5.90, 1.25, 2.58), 0.42),   # klein bottle
    ((4.13, 0.95, 2.29), 0.40),   # brass dumbbell
    ((5.63, 1.10, 1.65), 0.45),   # brass cluster
    ((4.13, 1.13, 1.45), 0.35),   # morpho heart
    ((4.99, 1.20, 1.19), 0.45),   # oilslick jack
]
# apex, slope (radius per metre of rise), height -- the axicon: apex y 1.55-0.28 = 1.27,
# radius = y - apex_y, up to the 0.56 m rim at y 1.83.
CONES = [((2.60, 1.27, 5.45), 1.0, 0.56)]


def occluded(p):
    """Does the segment EYE -> p pass through any piece?"""
    d = tuple(p[i] - EYE[i] for i in range(3))
    dd = sum(v * v for v in d)
    for c, r in SPHERES:
        e = tuple(EYE[i] - c[i] for i in range(3))
        b = sum(e[i] * d[i] for i in range(3))
        disc = b * b - dd * (sum(v * v for v in e) - r * r)
        if disc >= 0:
            s = math.sqrt(disc)
            if 0.0 < (-b - s) / dd < 1.0 or 0.0 < (-b + s) / dd < 1.0:
                return True
    for a, s, hh in CONES:
        e = tuple(EYE[i] - a[i] for i in range(3))
        qa = d[0] * d[0] + d[2] * d[2] - s * s * d[1] * d[1]
        qb = 2 * (e[0] * d[0] + e[2] * d[2] - s * s * e[1] * d[1])
        qc = e[0] * e[0] + e[2] * e[2] - s * s * e[1] * e[1]
        disc = qb * qb - 4 * qa * qc
        if abs(qa) < 1e-12 or disc < 0:
            continue
        rt = math.sqrt(disc)
        for t in ((-qb - rt) / (2 * qa), (-qb + rt) / (2 * qa)):
            if 0.0 < t < 1.0 and 0.0 <= e[1] + t * d[1] <= hh:
                return True
    return False


def unproject(x, y, w, h, ylevel):
    """Screen pixel -> the world point where its ray crosses the plane y = ylevel."""
    ty = math.tan(math.radians(FOV_Y) / 2)
    a = (2 * x / w - 1) * ty * (w / h)
    b = (1 - 2 * y / h) * ty
    d = tuple(FWD[i] + a * RIGHT[i] + b * CAMUP[i] for i in range(3))
    if abs(d[1]) < 1e-9:
        return None
    t = (ylevel - EYE[1]) / d[1]
    return None if t <= 0 else tuple(EYE[i] + t * d[i] for i in range(3))


def meter(im, name):
    c, sz = caps[name]
    w, h = im.w, im.h

    # Walk the cap's screen footprint, but decide membership in WORLD space: back-project
    # each cell centre onto the plane of the cap's top face and keep it only if it lands on
    # the cap rectangle.  A cap seen from 11 degrees above is a squashed trapezium, so its
    # screen bounding box is mostly floor, rain and neighbouring stand; that background is
    # darker than the cap, drags the median down, and then every cap pixel counts as
    # "caustic" (the diamond cap read 34.8% that way).  Back-projection also hands us the
    # world position of every cell, which is what the patch geometry below is measured in.
    top = c[1] + sz[1] / 2
    poly = []
    for sx, sz_ in ((-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)):
        q = project((c[0] + sx * sz[0], top, c[2] + sz_ * sz[2]), w, h)
        if q is None:
            print(f'  {name}: behind the camera')
            return
        poly.append(q)
    x0 = max(0, int(min(p[0] for p in poly)))
    x1 = min(w, int(max(p[0] for p in poly)) + 1)
    y0 = max(0, int(min(p[1] for p in poly)))
    y1 = min(h, int(max(p[1] for p in poly)) + 1)
    if x1 - x0 < 2 * DOWN or y1 - y0 < 2 * DOWN:
        print(f'  {name}: only {x1-x0}x{y1-y0} px on screen -- too small to meter')
        return

    cells, box, px, clip = [], 0, 0, 0
    for by in range(y0, y1 - DOWN + 1, DOWN):
        for bx in range(x0, x1 - DOWN + 1, DOWN):
            p3 = unproject(bx + DOWN / 2, by + DOWN / 2, w, h, top)
            if p3 is None or abs(p3[0] - c[0]) > sz[0] / 2 or abs(p3[2] - c[2]) > sz[2] / 2:
                continue
            box += 1
            if occluded(p3):
                continue
            acc = [0.0, 0.0, 0.0]
            for y in range(by, by + DOWN):
                for x in range(bx, bx + DOWN):
                    p = im.at(x, y)
                    px += 1
                    if max(p) * GAIN >= 1.0:
                        clip += 1
                    for k in range(3):
                        acc[k] += p[k]
            cells.append((tuple(a / (DOWN * DOWN) for a in acc), p3[0], p3[2]))
    if len(cells) < 16:
        print(f'  {name}: only {len(cells)} of {box} cap cells survive the piece mask')
        return

    lums = sorted(0.2126 * q[0] + 0.7152 * q[1] + 0.0722 * q[2] for q, _, _ in cells)
    bg = lums[len(lums) // 2]
    if bg <= 1e-9:
        print(f'  {name}: screen is black here')
        return

    # The cap's own pedestal, per channel.  Every cell is CAUSTIC + PEDESTAL, and in this
    # scene the pedestal is not small: the 11000 K sky panel lights every cap with broad
    # white fill that the isolation rig (sun only) never had.  So report chromaticity twice:
    # `spread` as seen on screen, and `xspread` after subtracting the pedestal, which is the
    # colour of the LIGHT THE PIECE ADDED and so the number comparable with the rig's.  If
    # spread collapses but xspread does not, the caustic is coloured and the fill is
    # washing it out -- a lighting problem, not an optics one.
    ped = tuple(sorted(q[k] for q, _, _ in cells)[len(cells) // 2] for k in range(3))

    wsum = satsum = 0.0
    n = 0
    chroma, xchroma, patch, ctrl = [], [], [], []
    peak = 0.0
    for q, wx, wz in cells:
        lum = 0.2126 * q[0] + 0.7152 * q[1] + 0.0722 * q[2]
        peak = max(peak, lum / bg)
        # CONTROL BAND: cells at 1.0-1.2x the median are plain sunlit cap -- no caustic, so
        # whatever chromatic scatter they show is the SPECKLE FLOOR of this render, and it is
        # not small.  Mode D carries one hero wavelength per sample, so an unconverged pixel is
        # randomly coloured, and `spread` (an RMS radius in chromaticity) cannot tell random
        # colour from dispersed colour -- both are a wide cloud.  A 4x box only divides the
        # speckle by 4.  So `spread` is meaningless unless it clears `noise` by a good margin,
        # and this is the number that says whether it does.
        if bg <= lum < 1.2 * bg:
            s0 = sum(q)
            if s0 > 1e-9:
                ctrl.append((1.0, q[0] / s0, q[2] / s0, wx, wz))
        if lum < CUT * bg:
            continue
        n += 1
        e = lum - bg
        mx = max(q)
        satsum += e * ((mx - min(q)) / mx if mx > 1e-9 else 0.0)
        wsum += e
        patch.append((e, wx, wz))
        s = sum(q)
        if s > 1e-9:
            chroma.append((e, q[0] / s, q[2] / s, wx, wz))
        x = tuple(max(0.0, q[k] - ped[k]) for k in range(3))
        sx = sum(x)
        if sx > 1e-9:
            xchroma.append((e, x[0] / sx, x[2] / sx))
    sat = satsum / wsum if wsum > 0 else 0.0

    spread, xspread = scatter(chroma), scatter(xchroma)
    noise = scatter(ctrl)

    print(f'  {name:22s} {len(cells):5d}/{box:5d} cells   coverage {100.0*n/len(cells):5.2f}%   '
          f'sat {sat:.3f}   spread {spread:.3f}   xspread {xspread:.3f}   peak {peak:.2f}x   '
          f'clip {100.0*clip/px if px else 0.0:5.2f}%   '
          f'noise {noise:.3f}/fan {fmt(fan(ctrl))}   fan {fmt(fan(chroma))}')
    if patch:
        cx_ = sum(t[0] * t[1] for t in patch) / wsum
        cz_ = sum(t[0] * t[2] for t in patch) / wsum
        print(f'  {"":22s} caustic core at world {cx_:.2f} {cz_:.2f}, '
              f'{cx_-c[0]:+.2f} {cz_-c[2]:+.2f} from the cap centre')


if __name__ == '__main__':
    src = sys.argv[1]
    want = sys.argv[2:] or sorted(caps)
    im = Linear(src)
    print(f'{src}, caustic = >{CUT:g}x each cap\'s OWN median, {DOWN}x box in linear light')
    if im.clipped:
        print(im.banner())
    print()
    for nm in want:
        if nm not in caps:
            nm = f'stand_{nm}_cap'
        meter(im, nm)
