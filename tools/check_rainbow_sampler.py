"""Regression check: the rainbow phase sampler draws from the density it reports, host and device.

    D:/python314/python.exe -u tools/check_rainbow_sampler.py [--exe ftrace.exe] [--no-gpu]

A `phase rainbow` medium samples its scattered direction from a tabulated (wavelength x cos theta)
phase and reports, as the direction's density, the table's bilinear value. Every analog walk takes
that value for the sampling density, and so does the spectral MIS at a rainbow scatter, which
weighs each wavelength by its density over the hero's. Until 0.380.1 the sampler drew from the
NEAREST wavelength row's histogram instead: halfway between two rows a 40 M-sample histogram sat
up to 11 sigma off the reported density, on the bows (chi^2/dof 1.54).

`ftrace -rainbow-selftest` histograms 40 M draws per test wavelength -- halfway between two table
rows, a quarter of the way, on a row -- one bin per table cell, against the reported density's mass
in each, for the host's sampler and (when there is a GPU) the device's. This requires every chi^2
per degree of freedom under 1.15 (about 5 standard deviations over ~2000 bins), and the device's
lines to be there unless --no-gpu. Exits 1 on failure.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    p = subprocess.run([os.path.abspath(a.exe), "-rainbow-selftest"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    log = p.stdout + p.stderr
    rows = re.findall(r"sampler \((\w+)\) at ([0-9.]+) nm: chi2/dof ([0-9.]+) over (\d+) bins, "
                      r"worst bin ([0-9.]+) sigma", log)
    ok = p.returncode == 0
    seen = set()
    for who, lam, chi, dof, worst in rows:
        good = float(chi) < 1.15
        ok &= good
        seen.add(who)
        print("%s: %s sampler at %s nm: chi2/dof %s over %s bins, worst bin %s sigma" % (
            "ok" if good else "FAIL", who, lam, chi, dof, worst), flush=True)
    for who in ("host",) + (() if a.no_gpu else ("device",)):
        if who not in seen:
            print("FAIL: no %s sampler lines in the self-test's output" % who)
            ok = False
    if not ok and not rows:
        print(log[-1500:])
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
