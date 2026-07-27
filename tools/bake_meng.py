#!/usr/bin/env python3
"""Bake ftrace's RGB->reflectance "smoothest spectrum" table (src/meng_table.h).

METHOD (not data) after Meng, Simon, Hanika & Dachsbacher, "Physically Meaningful
Rendering using Tristimulus Colours", EGSR 2015 / CGF 34(4): tabulate, over a grid
of *chromaticities*, the smoothest (minimum-roughness) reflectance that realises
that chromaticity at the highest attainable brightness, then interpolate the table
at render time and rescale to the requested luminance.

EVERYTHING HERE IS COMPUTED FROM SCRATCH.  The paper's published supplemental
`spectra_xyz_5nm_380_780_0.97.h` / `spectrum_grid.h` carry no licence of any kind,
so none of it is vendored or transcribed; we re-solve the same optimisation with
our own solver against ftrace's own CIE observer + D65, and the emitted table is
therefore our own data.  See TODO.md K1.

TWO DELIBERATE DEPARTURES from the paper's grid, both because ftrace only ever
upsamples an *sRGB* triple:

  1. GRID DOMAIN.  The paper grids the whole chromaticity diagram in a rotated
     `xy*` space and needs per-cell inside/boundary classification, because an
     arbitrary XYZ may sit anywhere in (or outside) the spectral locus.  Every
     colour ftrace upsamples comes from `rgb r g b` with r,g,b in [0,1], so its
     chromaticity is always inside the *sRGB primary triangle*.  We therefore use
     a barycentric lattice over that triangle: the enclosing cell is found in
     closed form from the colour itself (no search, no rotation, no locus
     polygon), and the gamut boundary is exactly the lattice boundary.

  2. EXACT CHROMATICITY.  The paper interpolates tabulated spectra directly, so
     the reconstructed chromaticity carries a small interpolation error.  We
     instead weight vertex k by (bary_k / T_k) where T_k = X+Y+Z of that vertex's
     spectrum.  Because a chromaticity is a (X+Y+Z)-weighted mean, this makes the
     mix land on the requested chromaticity *exactly*; the only residual error is
     the brightness clamp for colours brighter than the smooth spectrum can be.

USAGE
    python tools/bake_meng.py [--order N] [-o src/meng_table.h]

Needs numpy + scipy.  Takes a couple of minutes at the default order.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# ftrace's colour pipeline, ported exactly (src/color.h, src/spectrum.h,
# src/lights.h, src/upsample.h).  These must stay bit-compatible with the C++ or
# the baked table will not round-trip through `reflectanceToLinearSrgbD65`.
# ---------------------------------------------------------------------------

LAMBDA_MIN = 360.0          # color.h
LAMBDA_MAX = 830.0
BASIS_N = 95                # upsample::Basis::N  = (830-360)/5 + 1
BASIS_STEP = 5.0            # upsample::Basis::step

# The tabulated support.  Outside it the C++ holds the endpoint value, which the
# weight folding below reproduces exactly.
TAB_MIN = 380.0
TAB_STEP = 5.0
TAB_N = 81                  # 380..780 nm inclusive


def gauss_piece(x, mu, s1, s2):
    """color.h gaussPiece — a Gaussian with a different width either side of mu."""
    t = (x - mu) * np.where(x < mu, s1, s2)
    return np.exp(-0.5 * t * t)


def cie_x(w):
    return (0.362 * gauss_piece(w, 442.0, 0.0624, 0.0374)
            + 1.056 * gauss_piece(w, 599.8, 0.0264, 0.0323)
            - 0.065 * gauss_piece(w, 501.1, 0.0490, 0.0382))


def cie_y(w):
    return (0.821 * gauss_piece(w, 568.8, 0.0213, 0.0247)
            + 0.286 * gauss_piece(w, 530.9, 0.0613, 0.0322))


def cie_z(w):
    return (1.217 * gauss_piece(w, 437.0, 0.0845, 0.0278)
            + 0.681 * gauss_piece(w, 459.0, 0.0385, 0.0725))


def blackbody_radiance(kelvin, lambda_nm):
    """spectrum.h blackbodyRadiance — Planck's law, absolute."""
    h, c, kb = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    l = lambda_nm * 1e-9
    e = np.exp((h * c) / (l * kb * kelvin)) - 1.0
    return (2.0 * h * c * c) / (l ** 5 * e)


# linSrgbToXyz (upsample.h) — columns are the XYZ of unit R, G and B.
M_RGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
])


def tabulated_weights():
    """The D65 x CIE integration weights of upsample::Basis, folded onto the
    tabulated 380..780 nm support.

    `Basis` integrates 95 samples over 360..830 nm; the emitted spectrum is only
    tabulated over 380..780 and *held* outside.  Folding each Basis sample's
    weight onto the tabulated index it would read (clamped) makes the Python
    solve and the C++ integration agree to the last bit, instead of merely
    approximately.
    """
    lam = LAMBDA_MIN + BASIS_STEP * np.arange(BASIS_N)
    d65 = np.maximum(0.0, blackbody_radiance(6504.0, lam))
    wx = d65 * cie_x(lam) * BASIS_STEP
    wy = d65 * cie_y(lam) * BASIS_STEP
    wz = d65 * cie_z(lam) * BASIS_STEP
    k = 1.0 / wy.sum()                      # unit reflectance -> Y = 1
    wx, wy, wz = wx * k, wy * k, wz * k

    idx = np.clip(np.rint((lam - TAB_MIN) / TAB_STEP).astype(int), 0, TAB_N - 1)
    W = np.zeros((3, TAB_N))
    np.add.at(W[0], idx, wx)
    np.add.at(W[1], idx, wy)
    np.add.at(W[2], idx, wz)
    return W


# ---------------------------------------------------------------------------
# The per-chromaticity solve.
# ---------------------------------------------------------------------------

def smoothest(W, target_xyz, x0, upper=None):
    """Minimum-roughness non-negative reflectance with exactly `target_xyz`.

    Roughness is the sum of squared first differences (Smits 1999's term, which
    is what the paper adopts).  Convex QP over 81 variables with three linear
    equalities and bounds; SLSQP with an analytic gradient is plenty.

    NOTE THE MISSING UPPER BOUND, which is the whole reason the table can be
    indexed by chromaticity alone.  The tabulated spectra are normalised to
    Y = 1 and the renderer scales them by the requested luminance; that scaling
    is only *exactly* optimal if the feasible set is scale-invariant.  The
    non-negative orthant {s >= 0} is a cone, so it is; the box {0 <= s <= 1} is
    not.  Solving with an upper bound of 1 (the obvious thing, and what a
    "brightest attainable spectrum" normalisation forces) makes that bound
    active, and the scaled-down result is then measurably rougher than the true
    optimum for the same colour -- i.e. it loses the one property being
    tabulated.  Dropping it means a vertex spectrum may exceed 1 before scaling;
    the renderer clamps after scaling, which only bites for colours too bright
    for any smooth reflectance of that hue (the paper's own fix-up case).
    """
    D = np.diff(np.eye(TAB_N), axis=0)      # (N-1, N) first-difference operator
    DtD2 = 2.0 * (D.T @ D)

    def f(s):
        d = np.diff(s)
        return float(d @ d)

    def g(s):
        return DtD2 @ s

    cons = [{"type": "eq",
             "fun": lambda s: W @ s - target_xyz,
             "jac": lambda s: W}]
    r = minimize(f, x0, jac=g, bounds=[(0.0, upper)] * TAB_N,
                 constraints=cons, method="SLSQP",
                 options={"maxiter": 500, "ftol": 1e-14})
    resid = np.abs(W @ r.x - target_xyz).max()
    if resid > 1e-9:
        raise RuntimeError(f"smoothness QP missed its target by {resid:.3e}")
    return np.maximum(r.x, 0.0)


# ---------------------------------------------------------------------------
# The barycentric lattice over the sRGB primary triangle.
# ---------------------------------------------------------------------------

def vertex_index(order, a, b):
    """Row-major index of lattice point (a, b) with a + b <= order."""
    return a * (order + 1) - (a * (a - 1)) // 2 + b


def lattice(order):
    """Yield (index, a, b, c) for every point of the order-N barycentric lattice."""
    for a in range(order + 1):
        for b in range(order - a + 1):
            yield vertex_index(order, a, b), a, b, order - a - b


def bake(order):
    W = tabulated_weights()
    n_verts = (order + 1) * (order + 2) // 2
    spectra = np.zeros((n_verts, TAB_N))
    sums = np.zeros(n_verts)

    # Chromaticity of each sRGB primary (the lattice corners).
    prim = M_RGB_TO_XYZ / M_RGB_TO_XYZ.sum(axis=0, keepdims=True)   # xyz per primary
    t0 = time.time()
    prev = np.full(TAB_N, 0.5)
    for k, a, b, c in lattice(order):
        # Barycentric mix of the primary chromaticities, at unit luminance.
        w = np.array([a, b, c], dtype=float) / order
        xyz = prim @ w
        x, y = xyz[0], xyz[1]
        target = np.array([x / y, 1.0, (1.0 - x - y) / y])

        # Warm start from the previous vertex: a near neighbour in chromaticity.
        # (The QP is strictly convex on its feasible set, so the optimum is
        # unique and start-independent -- this only buys iterations.)
        s = smoothest(W, target, prev)
        prev = s
        spectra[k] = s
        sums[k] = float(W.sum(axis=0) @ s)
        if k % 25 == 0:
            print(f"  vertex {k:4d}/{n_verts}  ({a:2d},{b:2d},{c:2d})  "
                  f"peak={s.max():.3f}  {time.time()-t0:6.1f}s", flush=True)
    print(f"  {n_verts} vertices in {time.time()-t0:.1f}s", flush=True)
    return spectra, sums


# ---------------------------------------------------------------------------
# Emit the C++ header.
# ---------------------------------------------------------------------------

BANNER = """\
// meng_table.h -- GENERATED by tools/bake_meng.py.  DO NOT EDIT BY HAND.
//
// Minimum-roughness ("smoothest") reflectance spectra tabulated over a
// barycentric lattice of the sRGB primary chromaticity triangle.  Consumed by
// rgbToReflectanceMeng() in upsample.h; see that function for the query maths
// and tools/bake_meng.py for how these numbers were solved.
//
// METHOD after Meng, Simon, Hanika & Dachsbacher, "Physically Meaningful
// Rendering using Tristimulus Colours", EGSR 2015 (Computer Graphics Forum
// 34(4)).  DATA is ours: every value below was re-solved from scratch by
// tools/bake_meng.py against ftrace's own CIE observer and D65, because the
// paper's supplemental table ships with no licence.  Nothing is transcribed
// from the authors' distribution.
//
// Generated %(when)s -- order=%(order)d, %(nverts)d vertices,
// %(nsamples)d samples over [%(lmin).0f, %(lmax).0f] nm at %(step).0f nm,
// each normalised to unit luminance (Y = 1) so the renderer's scale-by-luminance
// is exactly optimal, not merely approximate -- see bake_meng.py's smoothest().
#pragma once

namespace upsample {

// Lattice geometry.  A colour's barycentric coordinates in this triangle are
// just its linear-sRGB components weighted by each primary's X+Y+Z, so locating
// the enclosing cell needs no search (see rgbToReflectanceMeng).
inline constexpr int    MENG_ORDER     = %(order)d;   // subdivisions per triangle edge
inline constexpr int    MENG_VERTS     = %(nverts)d;
inline constexpr int    MENG_N         = %(nsamples)d;   // spectral samples per vertex
inline constexpr double MENG_LAMBDA_MIN = %(lmin).1f;
inline constexpr double MENG_LAMBDA_STEP = %(step).1f;

// Per-vertex X+Y+Z under D65 (the weight that makes barycentric interpolation
// land on the requested chromaticity exactly).  Y is 1 for every vertex, so this
// is 1 + (x + z)/y of that vertex's chromaticity.
inline constexpr float MENG_SUM[MENG_VERTS] = {
%(sums)s};

// Per-vertex reflectance, row-major [vertex][sample].
inline constexpr float MENG_SPECTRA[MENG_VERTS][MENG_N] = {
%(spectra)s};

}  // namespace upsample
"""


def fmt_rows(spectra):
    out = []
    for row in spectra:
        vals = ", ".join(f"{v:.6f}f" for v in row)
        out.append("    {" + vals + "},\n")
    return "".join(out)


def fmt_sums(sums):
    out, line = [], "   "
    for v in sums:
        tok = f" {v:.6f}f,"
        if len(line) + len(tok) > 96:
            out.append(line + "\n")
            line = "   "
        line += tok
    if line.strip():
        out.append(line + "\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", type=int, default=16,
                    help="lattice subdivisions per triangle edge (default 16 -> 153 vertices)")
    ap.add_argument("-o", "--out", default="src/meng_table.h")
    args = ap.parse_args()

    print(f"baking order-{args.order} Meng table -> {args.out}", flush=True)
    spectra, sums = bake(args.order)
    n_verts = spectra.shape[0]
    text = BANNER % {
        "when": datetime.date.today().isoformat(),
        "order": args.order,
        "nverts": n_verts,
        "nsamples": TAB_N,
        "lmin": TAB_MIN,
        "lmax": TAB_MIN + TAB_STEP * (TAB_N - 1),
        "step": TAB_STEP,
        "sums": fmt_sums(sums),
        "spectra": fmt_rows(spectra),
    }
    with open(args.out, "w", newline="\n") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(text)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
