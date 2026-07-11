#!/usr/bin/env python3
"""splib_to_reflectance.py — USGS Spectral Library v7 spectrum -> reflectance table.

The USGS Spectral Library Version 7 (splib07, public domain, DOI 10.5066/F7RR1WDJ)
distributes each spectrum as an ASCII file of one reflectance value per line,
index-matched to a separate wavelength file (values in micrometres). The first
line of each file is a header; deleted/absent channels are flagged with the
sentinel -1.23e34.

This tool pairs a spectrum file with its wavelength file, drops the sentinels,
converts micrometres to nanometres, and emits the result as either an FTSL
`table { }` block (paste into a diffuse material's `reflect`) or a C++
tabulatedSpectrum initializer for baking into src/materials.h. This is exactly
how the built-in `reflectance:<name>` curves for measured materials were made.

Example
-------
  python tools/splib_to_reflectance.py \\
      splib07a_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt \\
      --wavelengths splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt \\
      --format cpp --resample 10 --min 360 --max 830

The ASD wavelength grid (2151 ch, 0.35-2.5 um at 1 nm) is the high-resolution
option; BECK/AVIRIS grids are coarser. stdlib only.
"""
import argparse
import os
import re
import sys

SENTINEL = -1.0   # splib deleted-channel flag is -1.23e34; anything this negative is invalid


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a USGS splib07 spectrum + wavelength file into a reflectance table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Example")[1] if "Example" in __doc__ else None,
    )
    p.add_argument("spectrum", help="splib07 spectrum ASCII file (reflectance per line)")
    p.add_argument("-w", "--wavelengths", required=True,
                   help="matching splib07 wavelength file (micrometres, index-aligned)")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    p.add_argument("-n", "--name", help="spectrum block name (default: derived from filename)")
    p.add_argument("--format", choices=["ftsl", "cpp"], default="ftsl",
                   help="'ftsl' -> spectrum block; 'cpp' -> tabulatedSpectrum initializer list")
    p.add_argument("--bare", action="store_true",
                   help="ftsl: emit only `table { ... }` without the `spectrum \"name\" =` wrapper")
    p.add_argument("--min", type=float, default=360.0, help="drop samples below this wavelength (nm)")
    p.add_argument("--max", type=float, default=830.0, help="drop samples above this wavelength (nm)")
    p.add_argument("--resample", type=float, default=None, metavar="STEP",
                   help="resample onto a uniform grid of STEP nm (linear interpolation)")
    p.add_argument("--clamp01", action="store_true", help="clamp reflectance into [0,1]")
    p.add_argument("--decimals", type=int, default=3, help="reflectance decimal places (default 3)")
    p.add_argument("--per-line", type=int, default=8, help="entries per output line (default 8)")
    return p.parse_args(argv)


def read_column(path):
    """Read a splib07 ASCII file: skip the header line, return the float column."""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    out = []
    for raw in lines[1:]:            # first line is the record header
        s = raw.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def lerp_sample(pairs, w):
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


def main(argv=None):
    args = parse_args(argv)

    refl = read_column(args.spectrum)
    wl_um = read_column(args.wavelengths)
    if len(refl) != len(wl_um):
        sys.stderr.write("error: spectrum has %d values but wavelength file has %d "
                         "(are they the same spectrometer?)\n" % (len(refl), len(wl_um)))
        return 1

    # pair, drop deleted-channel sentinels, convert um -> nm
    pairs = []
    for w, r in zip(wl_um, refl):
        if r <= SENTINEL:
            continue
        pairs.append((w * 1000.0, r))
    pairs.sort(key=lambda p: p[0])

    pairs = [(w, r) for (w, r) in pairs if args.min <= w <= args.max]
    if not pairs:
        sys.stderr.write("error: no valid samples in the %g..%g nm window\n" % (args.min, args.max))
        return 1

    if args.resample:
        step = args.resample
        lo = args.min
        hi = min(args.max, pairs[-1][0])
        n = int(round((hi - lo) / step))
        pairs = [(lo + i * step, lerp_sample(pairs, lo + i * step)) for i in range(n + 1)]

    if args.clamp01:
        pairs = [(w, min(1.0, max(0.0, r))) for (w, r) in pairs]

    def fmt_x(x):
        return str(int(round(x))) if abs(x - round(x)) < 1e-6 else ("%g" % x)

    def fmt_y(y):
        return ("%.*f" % (args.decimals, y)).rstrip("0").rstrip(".") or "0"

    per = max(1, args.per_line)

    if args.format == "cpp":
        cells = ["{%s,%s}" % (fmt_x(x), fmt_y(y)) for (x, y) in pairs]
        lines = ["return tabulatedSpectrum({"]
        for i in range(0, len(cells), per):
            lines.append("    " + ",".join(cells[i:i + per]) + ("," if i + per < len(cells) else ""))
        lines.append("});")
        out = "\n".join(lines) + "\n"
    else:
        entries = ["%s:%s" % (fmt_x(x), fmt_y(y)) for (x, y) in pairs]
        if len(entries) <= per:
            body = "table { " + " ".join(entries) + " }"
        else:
            blk = ["table {"]
            for i in range(0, len(entries), per):
                blk.append("    " + " ".join(entries[i:i + per]))
            blk.append("}")
            body = "\n".join(blk)
        if args.bare:
            out = body + "\n"
        else:
            stem = os.path.splitext(os.path.basename(args.spectrum))[0]
            name = args.name or re.sub(r"[^A-Za-z0-9_]", "_", stem)
            out = 'spectrum "%s" = %s\n' % (name, body)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        sys.stderr.write("wrote %d samples to %s\n" % (len(pairs), args.output))
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
