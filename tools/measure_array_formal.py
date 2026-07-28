"""Measure the 12 tiles of scenes/_array_formal.ftsl.

Each tile is a flat camera-facing quad at z=0.006, so its screen box follows
directly from the pinhole camera in the scene (eye 0.5 0.5 2.0, look_at along
-z, fov_y 32, square film).  For every tile we report the mean luminance plus
the horizontal / vertical gradient measured as (mean of right half - mean of
left half) and (mean of top half - mean of bottom half), normalised by the
tile mean so the row's lighting profile mostly cancels.

The numbers this prints are recorded in the scene's own header comment, so a later change
that shifts them shows up as a diff against that block.  Note this is only the QUALITATIVE
confirmation: a Monte-Carlo image agrees only to within its own noise and the room's
lighting profile.  The identities themselves are pinned exactly by `ftrace -checkarray`.

Usage:  python tools/measure_array_formal.py png/_array_formal.png
"""
import sys, math
from PIL import Image

RES = 320
FOV_Y = 32.0
EYE = (0.5, 0.5, 2.0)
ZPLANE = 0.006

dist = EYE[2] - ZPLANE
half = math.tan(math.radians(FOV_Y) * 0.5) * dist


def to_px(x, y):
    px = RES * 0.5 + (x - EYE[0]) / half * (RES * 0.5)
    py = RES * 0.5 - (y - EYE[1]) / half * (RES * 0.5)
    return px, py


TILE_W, TILE_H = 0.20, 0.22
XS = [0.04, 0.28, 0.52, 0.76]
ROWS = [("A", 0.68), ("B", 0.39), ("C", 0.10)]
LABELS = {
    "A1": "[0 1](u)              inline",
    "A2": "arr_a.reflect(a=u)    keyword rebind",
    "A3": "arr_a.reflect(u)      positional rebind",
    "A4": "[0 1](v)              CONTROL",
    "B1": "[[0 .3][.6 1]](u,v)   as authored",
    "B2": "[[0 .3][.6 1]](v,u)   transposed by hand",
    "B3": "arr_uv.reflect(u=v,v=u)  simultaneous swap",
    "B4": "arr_uv.reflect        bare property ref",
    "C1": "[0 1](0.25+0.5*v)     inline expression",
    "C2": "arr_a.reflect(a=0.25+0.5*v)",
    "C3": "material arr_a(a=0.25+0.5*v)  geometry field",
    "C4": "[0 1](1-u)            CONTROL",
}


def main(path):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    assert (W, H) == (RES, RES), f"expected {RES}x{RES}, got {W}x{H}"
    px = im.load()

    def lum(x, y):
        r, g, b = px[x, y]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    inset = 3  # skip the tile's own edge pixels (AA / light leak)
    results = {}
    for tag, y0 in ROWS:
        for i, x0 in enumerate(XS):
            l, b = to_px(x0, y0)
            r, t = to_px(x0 + TILE_W, y0 + TILE_H)
            x_lo, x_hi = int(round(l)) + inset, int(round(r)) - inset
            y_lo, y_hi = int(round(t)) + inset, int(round(b)) - inset
            vals = [[lum(x, y) for x in range(x_lo, x_hi)] for y in range(y_lo, y_hi)]
            n = len(vals) * len(vals[0])
            mean = sum(sum(row) for row in vals) / n
            midx = len(vals[0]) // 2
            midy = len(vals) // 2
            left = sum(sum(row[:midx]) for row in vals) / (len(vals) * midx)
            right = sum(sum(row[midx:]) for row in vals) / (len(vals) * (len(vals[0]) - midx))
            top = sum(sum(row) for row in vals[:midy]) / (midy * len(vals[0]))
            bot = sum(sum(row) for row in vals[midy:]) / ((len(vals) - midy) * len(vals[0]))
            results[f"{tag}{i+1}"] = (mean, (right - left) / mean, (top - bot) / mean,
                                      (x_lo, y_lo, x_hi, y_hi))

    # The room is not perfectly uniform in x: the two strip lights are full width, but the
    # grey side walls bounce, so a tile's measured du carries an AMBIENT term that depends on
    # its column.  Four tiles have an albedo that is constant in u (C1/C2/C3 drive on v only,
    # A4 drives on v), one per column — so they measure that ambient du directly.  Subtract it
    # before comparing, or a pair of identical spellings in different columns reads as a
    # difference ~= the ambient drift (which is what the raw numbers below show).
    amb_du = {1: results["C1"][1], 2: results["C2"][1], 3: results["C3"][1], 4: results["A4"][1]}
    print(f"ambient du by column (from u-constant tiles C1 C2 C3 A4): "
          + "  ".join(f"{c}:{v:+.3f}" for c, v in sorted(amb_du.items())))
    print()

    print(f"{'tile':5s} {'mean':>7s} {'du':>8s} {'du-amb':>8s} {'dv':>8s}   spelling")
    for k in sorted(results):
        m, du, dv, box = results[k]
        print(f"{k:5s} {m:7.2f} {du:+8.3f} {du - amb_du[int(k[1])]:+8.3f} {dv:+8.3f}   {LABELS[k]}")

    print()
    def cmp(a, b, want_same=True):
        _, dua, dva, _ = results[a]
        _, dub, dvb, _ = results[b]
        dua -= amb_du[int(a[1])]
        dub -= amb_du[int(b[1])]
        d = max(abs(dua - dub), abs(dva - dvb))
        ok = (d < 0.05) == want_same
        rel = "==" if want_same else "!="
        print(f"  {'OK  ' if ok else 'FAIL'} {a} {rel} {b}   max|dgrad| = {d:.3f}")

    print("row A — three spellings of 'sample at u':")
    cmp("A1", "A2"); cmp("A1", "A3"); cmp("A1", "A4", want_same=False)
    print("row B — 2-D literal and a simultaneous swap:")
    cmp("B2", "B3"); cmp("B1", "B4"); cmp("B1", "B2", want_same=False)
    # NOTE row C's correction is self-referential — C1/C2/C3 are themselves the ambient
    # estimators for columns 1-3, so `C1 == C2` cannot fail here by construction.  Row C's
    # real evidence is the RAW table: C1/C2/C3 all carry dv ~ +0.176 and |du| <= 0.042 while
    # the C4 control is dv -0.048 / du -0.527, and C2/C3 agree to the reported precision on
    # mean as well.  Rows A and B are corrected by tiles OUTSIDE the pair being compared, so
    # their margins are not circular.
    print("row C — a driver expression at three binding sites:")
    cmp("C1", "C2"); cmp("C1", "C3"); cmp("C1", "C4", want_same=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "png/_array_formal.png")
