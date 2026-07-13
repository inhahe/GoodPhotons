"""Digitize a curve from a plot image into `wavelength_nm,value` (or any x,y) CSV.

Turns a raster spectral plot (SPD, transmittance, reflectance, n/k, ...) into
numeric samples, so we no longer have to eyeball heights off a datasheet graph.

Two extraction modes:
  * ``fill-top``  (default) — the curve is the TOP EDGE of a filled area. Works
    for the rainbow-filled spectral-power plots common in lamp datasheets (GE CMH,
    LED dies, ...), where the fill colour varies along the curve so colour-matching
    fails. For each pixel column it scans down from the top of the plot box and
    takes the first "ink" pixel (coloured OR dark enough), i.e. the curve.
  * ``color``     — the curve is a constant-colour LINE. For each column it takes
    the median y of pixels within ``--tol`` of ``--color``. Best for simple line
    graphs (single or multiple traces distinguished by colour).

Axes are calibrated by two reference points each (pixel -> data value); linear or
log10 per axis. The mapping is affine in pixel space, so calibration points need
not be the plot corners — any two ticks with known values work, and the image's
top-down y direction is handled automatically.

Always writes an ``*_annotated.png`` next to the output (unless --no-annotated)
overlaying the detected samples on the source image — VERIFY THE TRACE with it.

Example (GE CMH 3000 K SPD, rainbow-filled, resampled 390-700 @ 10 nm):
  python tools/plot_digitizer.py --image scraps/cmh3000_crop.png \
      --xcal 62:390 705:750 --ycal 470:0 40:1 \
      --plot-box 62 40 705 470 --mode fill-top \
      --resample 390 700 10 --normalize --out data/illuminant/cmh-3000k.csv \
      --xlabel wavelength_nm --ylabel relative_power

Run ``python tools/plot_digitizer.py --help`` for all options.
"""
import argparse
import math
import sys

import numpy as np
from PIL import Image


def _parse_pair(s):
    """'PIXEL:DATA' -> (float pixel, float data)."""
    a, b = s.split(":")
    return float(a), float(b)


def _axis_map(p0, d0, p1, d1, log):
    """Return f(pixel)->data for a linear or log10 axis from two ref points."""
    if log:
        l0, l1 = math.log10(d0), math.log10(d1)
        slope = (l1 - l0) / (p1 - p0)
        return lambda p: 10.0 ** (l0 + slope * (p - p0))
    slope = (d1 - d0) / (p1 - p0)
    return lambda p: d0 + slope * (p - p0)


def _inv_axis_map(p0, d0, p1, d1, log):
    """Return f(data)->pixel (inverse of _axis_map), for annotation."""
    if log:
        l0, l1 = math.log10(d0), math.log10(d1)
        slope = (p1 - p0) / (l1 - l0)
        return lambda d: p0 + slope * (math.log10(d) - l0)
    slope = (p1 - p0) / (d1 - d0)
    return lambda d: p0 + slope * (d - d0)


def _is_ink(rgb, bg_lum, sat_min, dark_max):
    """A pixel is 'ink' if coloured (saturated) or dark — not background/grid.

    rgb: (H,W,3) uint8 array slice-friendly; returns bool array.
    """
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    lum = (r + g + b) / 3.0
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)
    coloured = sat >= sat_min
    dark = lum <= dark_max
    return (coloured | dark) & (lum < bg_lum)


def extract_fill_top(arr, box, sat_min, dark_max, bg_lum):
    """For each column in box, first ink pixel scanning top->down. -> {col: row}."""
    left, top, right, bottom = box
    sub = arr[top:bottom, left:right, :3]
    ink = _is_ink(sub, bg_lum, sat_min, dark_max)
    cols = {}
    for x in range(ink.shape[1]):
        rows = np.nonzero(ink[:, x])[0]
        if rows.size:
            cols[left + x] = top + int(rows[0])
    return cols


def extract_color(arr, box, color, tol):
    """For each column, median row of pixels within tol of color. -> {col: row}."""
    left, top, right, bottom = box
    sub = arr[top:bottom, left:right, :3].astype(int)
    target = np.array(color, dtype=int)
    dist = np.sqrt(((sub - target) ** 2).sum(axis=2))
    mask = dist <= tol
    cols = {}
    for x in range(mask.shape[1]):
        rows = np.nonzero(mask[:, x])[0]
        if rows.size:
            cols[left + x] = top + int(np.median(rows))
    return cols


def main(argv=None):
    ap = argparse.ArgumentParser(description="Digitize a curve from a plot image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--xcal", nargs=2, required=True, metavar="PIXEL:DATA",
                    help="two x-axis refs, e.g. --xcal 62:390 705:750")
    ap.add_argument("--ycal", nargs=2, required=True, metavar="PIXEL:DATA",
                    help="two y-axis refs, e.g. --ycal 470:0 40:1")
    ap.add_argument("--xlog", action="store_true", help="x axis is log10")
    ap.add_argument("--ylog", action="store_true", help="y axis is log10")
    ap.add_argument("--mode", choices=["fill-top", "color"], default="fill-top")
    ap.add_argument("--color", help="R,G,B target for --mode color")
    ap.add_argument("--tol", type=float, default=40.0, help="colour distance tol")
    ap.add_argument("--plot-box", nargs=4, type=int, metavar=("L", "T", "R", "B"),
                    help="pixel box to search (excludes axes/legend); default full image")
    ap.add_argument("--sat-min", type=float, default=0.25,
                    help="min saturation to count as coloured ink (fill-top)")
    ap.add_argument("--dark-max", type=float, default=110.0,
                    help="max luminance to count as dark ink (fill-top)")
    ap.add_argument("--bg-lum", type=float, default=245.0,
                    help="pixels brighter than this are background")
    ap.add_argument("--resample", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                    help="resample onto regular grid via linear interp")
    ap.add_argument("--normalize", action="store_true", help="scale so peak y = 1")
    ap.add_argument("--clamp-min", type=float, help="clamp output y to >= this")
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--xlabel", default="wavelength_nm")
    ap.add_argument("--ylabel", default="value")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-annotated", action="store_true")
    args = ap.parse_args(argv)

    arr = np.asarray(Image.open(args.image).convert("RGB"))
    h, w = arr.shape[:2]
    box = tuple(args.plot_box) if args.plot_box else (0, 0, w, h)

    xp0, xd0 = _parse_pair(args.xcal[0]); xp1, xd1 = _parse_pair(args.xcal[1])
    yp0, yd0 = _parse_pair(args.ycal[0]); yp1, yd1 = _parse_pair(args.ycal[1])
    fx = _axis_map(xp0, xd0, xp1, xd1, args.xlog)
    fy = _axis_map(yp0, yd0, yp1, yd1, args.ylog)

    if args.mode == "color":
        if not args.color:
            ap.error("--mode color requires --color R,G,B")
        color = [int(c) for c in args.color.split(",")]
        cols = extract_color(arr, box, color, args.tol)
    else:
        cols = extract_fill_top(arr, box, args.sat_min, args.dark_max, args.bg_lum)

    if not cols:
        print("ERROR: no curve pixels found; check --plot-box / thresholds.",
              file=sys.stderr)
        return 1

    # pixel samples -> data, sorted & deduped by x (median y per identical data-x bin)
    xs = np.array(sorted(cols))
    data_x = np.array([fx(x) for x in xs])
    data_y = np.array([fy(cols[x]) for x in xs])
    order = np.argsort(data_x)
    data_x, data_y = data_x[order], data_y[order]

    if args.resample:
        start, stop, step = args.resample
        n = int(round((stop - start) / step)) + 1
        grid = start + step * np.arange(n)
        # np.interp needs strictly increasing x; collapse dup x by mean
        ux, inv = np.unique(data_x, return_inverse=True)
        uy = np.zeros_like(ux)
        np.add.at(uy, inv, data_y)
        counts = np.bincount(inv)
        uy = uy / counts
        out_x = grid
        out_y = np.interp(grid, ux, uy)
    else:
        out_x, out_y = data_x, data_y

    if args.clamp_min is not None:
        out_y = np.maximum(out_y, args.clamp_min)
    if args.normalize:
        peak = out_y.max()
        if peak > 0:
            out_y = out_y / peak

    fmt = f"{{:.{args.decimals}f}}"
    with open(args.out, "w", newline="\n") as fh:
        fh.write(f"# digitized from {args.image} via tools/plot_digitizer.py "
                 f"(mode={args.mode}).\n")
        fh.write(f"{args.xlabel},{args.ylabel}\n")
        for x, y in zip(out_x, out_y):
            xs_ = f"{x:.0f}" if float(x).is_integer() else fmt.format(x)
            fh.write(f"{xs_},{fmt.format(y)}\n")
    print(f"wrote {len(out_x)} samples -> {args.out} "
          f"(x {out_x.min():.1f}..{out_x.max():.1f}, y {out_y.min():.3g}..{out_y.max():.3g})")

    if not args.no_annotated:
        ann = arr.copy()
        # overlay the raw detected curve pixels in magenta so the trace is verifiable
        for x, r in cols.items():
            for dy in (-1, 0, 1):
                ry = r + dy
                if 0 <= ry < h and 0 <= x < w:
                    ann[ry, x] = (255, 0, 255)
        ann_path = args.out.rsplit(".", 1)[0] + "_annotated.png"
        Image.fromarray(ann).save(ann_path)
        print(f"wrote overlay -> {ann_path}  (magenta = detected curve; verify it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
