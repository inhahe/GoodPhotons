"""Read a render as scene-linear RGB floats -- shared by `_gemsweep.py` and `_capchroma.py`.

WHY THIS EXISTS.  Both rigs measure caustics, and a caustic is by definition the brightest
thing in frame, so metering one off a PNG measures the tone map instead of the render: 8-bit
sRGB clamps at white, and at the clip point all three channels are EQUAL, which deletes
exactly the two quantities a caustic study cares about -- its hue and its peak-to-screen
ratio.  (Found the hard way: 596 of one cap's 22639 pixels in `gallery_rain` were pure
white, more than half the caustic's area, and a whole ranking table had been computed
through that clamp.)  Render with ftrace's `-hdr` and read the `.pfm` sidecar it writes.

A PNG still loads, so old outputs remain readable, but `.clipped` is set and callers are
expected to say so out loud.
"""
import array
import math
import sys


def srgb_to_lin(v):
    v /= 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


class Linear:
    """An image as scene-linear RGB floats, from a .pfm (exact) or a .png (clipped)."""

    def __init__(self, path):
        self.path = path
        self.clipped = not path.lower().endswith('.pfm')
        if self.clipped:
            from PIL import Image
            im = Image.open(path).convert('RGB')
            self.w, self.h = im.size
            lut = [srgb_to_lin(float(i)) for i in range(256)]
            px = im.load()
            self.buf = array.array('f', (lut[px[x, y][k]]
                                         for y in range(self.h)
                                         for x in range(self.w) for k in range(3)))
            return
        with open(path, 'rb') as f:
            if f.readline().strip() != b'PF':
                raise ValueError(f'{path} is not a colour PFM')
            self.w, self.h = (int(v) for v in f.readline().split())
            scale = float(f.readline())
            self.buf = array.array('f')
            self.buf.frombytes(f.read(self.w * self.h * 12))
        if (scale < 0) != (sys.byteorder == 'little'):
            self.buf.byteswap()
        # PFM rasters run BOTTOM-to-top; flip to image order so (x, y) means the same thing
        # here as it does for a PNG and for the camera projection.
        row = self.w * 3
        self.buf = array.array('f', (v for y in range(self.h - 1, -1, -1)
                                     for v in self.buf[y * row:(y + 1) * row]))

    def at(self, x, y):
        i = (y * self.w + x) * 3
        return self.buf[i], self.buf[i + 1], self.buf[i + 2]

    def banner(self):
        return ('' if not self.clipped else
                '  !! 8-bit sRGB: every caustic core is clamped to white and its colour with'
                ' it.\n     Re-render with -hdr and meter the .pfm.')


# ---------------------------------------------------------------------------------------
# The chromaticity statistics, shared for the same reason the reader is: both rigs rank
# pieces on them, so they have to be ONE implementation or the two rankings aren't
# comparable.  Points are `(weight, r, b, x, z)` where (r, b) = (R, B)/(R+G+B) and (x, z)
# is the world position on the screen.


def scatter(pts):
    """Twice the weighted RMS radius of the (r, b) chromaticity cloud.

    White or uniformly tinted collapses to a point; a spectrum is a long streak.  On its own
    this number is NOT evidence of dispersion -- see `fan` -- because a loud enough sampler
    makes an equally wide cloud out of nothing.
    """
    if not pts:
        return 0.0
    tw = sum(t[0] for t in pts)
    mr = sum(t[0] * t[1] for t in pts) / tw
    mb = sum(t[0] * t[2] for t in pts) / tw
    return 2.0 * math.sqrt(sum(t[0] * ((t[1] - mr) ** 2 + (t[2] - mb) ** 2)
                               for t in pts) / tw)


def fan(pts):
    """Fraction of the chromatic variance explained by a quadratic in position (adjusted R^2).

    `scatter` above a measured noise floor says the colour is real; it does not say the colour
    is ARRANGED into anything, and mode D's per-sample hero wavelength produces a loud,
    genuinely-coloured, completely random cloud that scores identically.  What a dispersing
    piece claims is that chromaticity is a FUNCTION OF POSITION, so fit each channel over
    (x, z) by weighted least squares and report how much of the variance the fit explains.
    Speckle scores ~0 however loud; a rainbow scores high however quiet.

    The basis is QUADRATIC {1, x, z, x^2, z^2, xz}, not linear: a first draft fitted a plane
    and scored the shipped axicon 0.09, because an axicon disperses RADIALLY about its own
    axis -- red outside, violet inside, on both flanking cusps at once -- and a plane is blind
    to that by symmetry.  The quadratic basis covers ramps, rings and saddles without needing
    a guess about which the piece makes.  Returns None (not 0.0) when there are too few cells
    to fit, since "unmeasurable" and "not organised" are different answers.
    """
    if len(pts) < 20:
        return None
    tw = sum(t[0] for t in pts)
    mx = sum(t[0] * t[3] for t in pts) / tw
    mz = sum(t[0] * t[4] for t in pts) / tw
    # Centre and scale to O(1), or the quadratic terms wreck the normal equations.
    sc = math.sqrt(sum(t[0] * ((t[3] - mx) ** 2 + (t[4] - mz) ** 2) for t in pts) / tw)
    if sc < 1e-9:
        return None
    rows = []
    for t in pts:
        u, v = (t[3] - mx) / sc, (t[4] - mz) / sc
        rows.append((t[0], (1.0, u, v, u * u, v * v, u * v), t[1], t[2]))
    m, tot, res = 6, 0.0, 0.0
    for ch in (2, 3):
        a = [[sum(r[0] * r[1][i] * r[1][j] for r in rows) for j in range(m)] +
             [sum(r[0] * r[1][i] * r[ch] for r in rows)] for i in range(m)]
        for i in range(m):                                        # Gaussian elimination
            p = max(range(i, m), key=lambda k: abs(a[k][i]))
            if abs(a[p][i]) < 1e-12:
                return None
            a[i], a[p] = a[p], a[i]
            for k in range(i + 1, m):
                f = a[k][i] / a[i][i]
                for j in range(i, m + 1):
                    a[k][j] -= f * a[i][j]
        co = [0.0] * m
        for i in range(m - 1, -1, -1):
            co[i] = (a[i][m] - sum(a[i][j] * co[j] for j in range(i + 1, m))) / a[i][i]
        mc = sum(r[0] * r[ch] for r in rows) / tw
        for r in rows:
            e = r[ch] - sum(co[j] * r[1][j] for j in range(m))
            res += r[0] * e * e
            tot += r[0] * (r[ch] - mc) ** 2
    dof = len(rows) - m
    if tot <= 0 or dof < 4:
        return None
    return max(0.0, 1.0 - (res / dof) / (tot / (len(rows) - 1)))       # adjusted R^2


def fmt(v):
    return ' -- ' if v is None else f'{v:.2f}'
