#!/usr/bin/env python3
"""csv_to_table.py — turn a CSV/TSV of spectral data into an FTSL `table { }` block.

The renderer's scene language (FTSL) ingests measured spectra via
    spectrum "name" = table { 400:0.05 450:0.12 500:0.31 ... }
where each token is `wavelength_nm:value`, linearly interpolated between samples
(src/spectrum.h `tabulatedSpectrum`). This tool converts a two-column data file
(wavelength, value) — e.g. a reflectance curve, an illuminant SPD, or an n(λ)
dispersion column exported from refractiveindex.info — into that block, so real
published datasets can be dropped straight into a scene.

Examples
--------
  # a reflectance CSV (nm, reflectance) -> a named spectrum block
  python tools/csv_to_table.py gold_reflectance.csv -n gold_measured

  # refractiveindex.info export in micrometres, second column is n
  python tools/csv_to_table.py bk7.csv --x-scale 1000 --y-col 1 -n bk7_ior

  # resample to a uniform 5 nm grid over the visible, clamp to [0,1]
  python tools/csv_to_table.py leaf.csv --resample 5 --min 380 --max 780 --clamp01

  # just the bare `table { ... }` expression (paste into any <spectrum> slot)
  python tools/csv_to_table.py skin.csv --bare

By default the input is auto-sniffed for comma/semicolon/tab/whitespace delimiters,
header and comment (`#`) lines are skipped, and samples are sorted by wavelength.
Output goes to stdout unless -o is given. stdlib only — no numpy required.
"""
import argparse
import os
import re
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a CSV/TSV of (wavelength, value) into an FTSL table {} block.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[1] if "Examples" in __doc__ else None,
    )
    p.add_argument("input", help="input CSV/TSV file, or '-' for stdin")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    p.add_argument("-n", "--name", help="spectrum block name (default: derived from filename)")
    p.add_argument("--bare", action="store_true",
                   help="emit only `table { ... }` without the `spectrum \"name\" =` wrapper")
    p.add_argument("--x-col", type=int, default=0, help="0-based wavelength column (default 0)")
    p.add_argument("--y-col", type=int, default=1, help="0-based value column (default 1)")
    p.add_argument("--x-scale", type=float, default=1.0,
                   help="multiply wavelength by this to get nanometres (e.g. 1000 for micrometres)")
    p.add_argument("--delimiter", help="field delimiter (default: auto-detect)")
    p.add_argument("--min", type=float, default=None, help="drop samples below this wavelength (nm)")
    p.add_argument("--max", type=float, default=None, help="drop samples above this wavelength (nm)")
    p.add_argument("--resample", type=float, default=None, metavar="STEP",
                   help="resample onto a uniform grid of STEP nm (linear interpolation)")
    p.add_argument("--normalize", choices=["none", "max", "area"], default="none",
                   help="scale so the maximum is 1 ('max') or the integral is 1 ('area')")
    p.add_argument("--clip-negative", action="store_true", help="clamp negative values to 0")
    p.add_argument("--clamp01", action="store_true", help="clamp values into [0,1]")
    p.add_argument("--decimals", type=int, default=4, help="value decimal places (default 4)")
    p.add_argument("--per-line", type=int, default=8, help="table entries per output line (default 8)")
    return p.parse_args(argv)


def read_rows(text, delimiter, x_col, y_col):
    """Yield (x, y) float pairs from raw text, skipping headers/comments/blank lines."""
    pairs = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if delimiter:
            fields = line.split(delimiter)
        else:
            # Auto: try comma/semicolon/tab, else any run of whitespace.
            if "," in line:
                fields = line.split(",")
            elif ";" in line:
                fields = line.split(";")
            elif "\t" in line:
                fields = line.split("\t")
            else:
                fields = re.split(r"\s+", line)
        fields = [f.strip() for f in fields]
        if len(fields) <= max(x_col, y_col):
            continue
        try:
            x = float(fields[x_col])
            y = float(fields[y_col])
        except ValueError:
            # header row or non-numeric line -> skip
            continue
        pairs.append((x, y))
    return pairs


def lerp_sample(pairs, w):
    """Linear interpolation of the sorted (x,y) list at wavelength w (clamped at ends)."""
    if w <= pairs[0][0]:
        return pairs[0][1]
    if w >= pairs[-1][0]:
        return pairs[-1][1]
    lo, hi = 0, len(pairs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if pairs[mid][0] <= w:
            lo = mid
        else:
            hi = mid
    x0, y0 = pairs[lo]
    x1, y1 = pairs[lo + 1]
    t = (w - x0) / (x1 - x0) if x1 > x0 else 0.0
    return y0 + (y1 - y0) * t


def trapezoid(pairs):
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area


def main(argv=None):
    args = parse_args(argv)

    if args.input == "-":
        text = sys.stdin.read()
        stem = "spectrum"
    else:
        with open(args.input, "r", encoding="utf-8-sig") as f:
            text = f.read()
        stem = os.path.splitext(os.path.basename(args.input))[0]

    pairs = read_rows(text, args.delimiter, args.x_col, args.y_col)
    if not pairs:
        sys.stderr.write("error: no numeric (wavelength, value) rows found\n")
        return 1

    # x scale -> nanometres
    if args.x_scale != 1.0:
        pairs = [(x * args.x_scale, y) for (x, y) in pairs]

    # sort by wavelength, drop exact-duplicate wavelengths (keep first)
    pairs.sort(key=lambda p: p[0])
    dedup = []
    seen = set()
    for x, y in pairs:
        if x in seen:
            continue
        seen.add(x)
        dedup.append((x, y))
    pairs = dedup

    # wavelength window
    if args.min is not None:
        pairs = [(x, y) for (x, y) in pairs if x >= args.min]
    if args.max is not None:
        pairs = [(x, y) for (x, y) in pairs if x <= args.max]
    if not pairs:
        sys.stderr.write("error: no samples left after --min/--max filtering\n")
        return 1

    # resample onto a uniform grid
    if args.resample:
        step = args.resample
        lo = args.min if args.min is not None else pairs[0][0]
        hi = args.max if args.max is not None else pairs[-1][0]
        grid = []
        # inclusive of hi within a small epsilon
        n = int(round((hi - lo) / step))
        for i in range(n + 1):
            w = lo + i * step
            grid.append((w, lerp_sample(pairs, w)))
        pairs = grid

    # value clamping
    if args.clip_negative:
        pairs = [(x, max(0.0, y)) for (x, y) in pairs]
    if args.clamp01:
        pairs = [(x, min(1.0, max(0.0, y))) for (x, y) in pairs]

    # normalization
    if args.normalize == "max":
        peak = max((y for _, y in pairs), default=0.0)
        if peak > 0:
            pairs = [(x, y / peak) for (x, y) in pairs]
    elif args.normalize == "area":
        area = trapezoid(pairs)
        if area > 0:
            pairs = [(x, y / area) for (x, y) in pairs]

    # format entries: integer wavelength if it round-trips, else trimmed float
    def fmt_x(x):
        return str(int(round(x))) if abs(x - round(x)) < 1e-6 else ("%g" % x)

    def fmt_y(y):
        return ("%.*f" % (args.decimals, y)).rstrip("0").rstrip(".") or "0"

    entries = ["%s:%s" % (fmt_x(x), fmt_y(y)) for (x, y) in pairs]

    per = max(1, args.per_line)
    if len(entries) <= per:
        body = "table { " + " ".join(entries) + " }"
    else:
        lines = ["table {"]
        for i in range(0, len(entries), per):
            lines.append("    " + " ".join(entries[i:i + per]))
        lines.append("}")
        body = "\n".join(lines)

    if args.bare:
        out = body
    else:
        name = args.name or re.sub(r"[^A-Za-z0-9_]", "_", stem)
        out = 'spectrum "%s" = %s' % (name, body)

    out += "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        sys.stderr.write("wrote %d samples to %s\n" % (len(entries), args.output))
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
