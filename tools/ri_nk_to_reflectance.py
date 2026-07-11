#!/usr/bin/env python3
"""ri_nk_to_reflectance.py — refractiveindex.info n,k data -> reflectance table.

refractiveindex.info publishes measured complex refractive indices n(lambda),
k(lambda) as small public-domain (CC0) YAML files, e.g.

    DATA:
      - type: tabulated nk
        data: |
            0.4133 1.46 1.958
            0.4305 1.45 1.948
            ...

where each row is `wavelength_um  n  k`. For an opaque metal the normal-incidence
reflectance is

    R(lambda) = ((n-1)^2 + k^2) / ((n+1)^2 + k^2)

This tool reads such a file, converts to R(lambda) in nanometres, and emits it as
either an FTSL `table { }` block (paste into any <spectrum> slot, e.g. a metal's
`reflect`) or a C++ initializer list for baking into src/materials.h.

Examples
--------
  # gold reflectance as an FTSL table, windowed to the sensor range
  python tools/ri_nk_to_reflectance.py Au_Johnson.yml --name gold_measured

  # as a C++ tabulatedSpectrum initializer, resampled to a 10 nm grid
  python tools/ri_nk_to_reflectance.py Al_Rakic.yml --format cpp --resample 10

The visible/near-IR window defaults to 350..900 nm (a margin around the renderer's
360..830 sensor range so endpoint clamping is accurate). stdlib only.
"""
import argparse
import os
import re
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a refractiveindex.info tabulated-nk YAML into a normal-incidence reflectance table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[1] if "Examples" in __doc__ else None,
    )
    p.add_argument("input", help="refractiveindex.info YAML file, or '-' for stdin")
    p.add_argument("-o", "--output", help="output file (default: stdout)")
    p.add_argument("-n", "--name", help="spectrum block name (default: derived from filename)")
    p.add_argument("--format", choices=["ftsl", "cpp"], default="ftsl",
                   help="'ftsl' -> spectrum block; 'cpp' -> tabulatedSpectrum initializer list")
    p.add_argument("--bare", action="store_true",
                   help="ftsl: emit only `table { ... }` without the `spectrum \"name\" =` wrapper")
    p.add_argument("--min", type=float, default=350.0, help="drop samples below this wavelength (nm)")
    p.add_argument("--max", type=float, default=900.0, help="drop samples above this wavelength (nm)")
    p.add_argument("--resample", type=float, default=None, metavar="STEP",
                   help="resample R onto a uniform grid of STEP nm (linear interpolation)")
    p.add_argument("--decimals", type=int, default=3, help="reflectance decimal places (default 3)")
    p.add_argument("--per-line", type=int, default=6, help="entries per output line (default 6)")
    return p.parse_args(argv)


def read_nk(text):
    """Yield (lambda_um, n, k) triples from the `data:` block of a refractiveindex YAML."""
    triples = []
    in_data = False
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("data:"):
            in_data = True
            continue
        if not in_data:
            continue
        if not stripped:
            continue
        # A new top-level YAML key (no leading indent, ends the literal block).
        if re.match(r"^[A-Za-z_]", line):
            break
        fields = re.split(r"\s+", stripped)
        if len(fields) < 3:
            continue
        try:
            lam = float(fields[0]); n = float(fields[1]); k = float(fields[2])
        except ValueError:
            continue
        triples.append((lam, n, k))
    return triples


def reflectance(n, k):
    return ((n - 1.0) ** 2 + k * k) / ((n + 1.0) ** 2 + k * k)


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

    if args.input == "-":
        text = sys.stdin.read()
        stem = "spectrum"
    else:
        with open(args.input, "r", encoding="utf-8-sig") as f:
            text = f.read()
        stem = os.path.splitext(os.path.basename(args.input))[0]

    triples = read_nk(text)
    if not triples:
        sys.stderr.write("error: no `tabulated nk` rows found (expected a refractiveindex.info YAML)\n")
        return 1

    # (lambda_nm, R), sorted and de-duplicated by wavelength
    pairs = []
    seen = set()
    for lam_um, n, k in sorted(triples, key=lambda t: t[0]):
        w = lam_um * 1000.0
        if w in seen:
            continue
        seen.add(w)
        pairs.append((w, reflectance(n, k)))

    pairs = [(w, r) for (w, r) in pairs if args.min <= w <= args.max]
    if not pairs:
        sys.stderr.write("error: no samples left in the %g..%g nm window\n" % (args.min, args.max))
        return 1

    if args.resample:
        step = args.resample
        # Grid runs on clean multiples of STEP from --min. lerp_sample clamps at the
        # data ends, so a grid point past the measured range just repeats the endpoint
        # (never extrapolates); cap the top at the last measured point to avoid a long
        # flat tail of invented values.
        lo = args.min
        hi = min(args.max, pairs[-1][0])
        n = int(round((hi - lo) / step))
        grid = [(lo + i * step, lerp_sample(pairs, lo + i * step)) for i in range(n + 1)]
        pairs = grid

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
