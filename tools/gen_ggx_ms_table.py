"""Generate src/ggx_ms_table.h: the GGX lobe's single-scattering albedo, for its multiple-scattering
compensation (src/ggx.h, Kulla & Conty 2017).

    python tools/gen_ggx_ms_table.py [--k 512] [--nm 65] [--out src/ggx_ms_table.h]

E(mu, alpha) is the directional albedo of a white GGX reflector (height-correlated Smith) for an arrival
at cosine mu to the normal: the mean of the visible-normal draw's weight G2 / G1 (ggx.h weight), taken
here over a K x K midpoint grid of the draw's two uniforms -- the same spherical-cap sampler ggx.h
uses, so the integrand is smooth but for the edge where a draw leaves below the surface. Eavg(alpha)
is 2 int_0^1 E(mu) mu dmu of the tabulated E itself, read as msE reads it (linear in sqrt(mu) between
its nodes): f_ms = (1 - E(a)) (1 - E(b)) / (pi (1 - Eavg)) then integrates to exactly 1 - E(cos_a), at
every alpha (both are linear in the rows), where an Eavg integrated afresh on finer cosines disagreed
with the nodes' E by 11 % of 1 - Eavg on a polished lobe's grazing dip. The table spans alpha in [0, ALPHA_MAX] (roughness 0 to 2
at 0.646 alpha per roughness) uniform in sqrt(alpha), and mu in [0, 1] uniform in sqrt(mu). With
height-correlated masking E returns to 1 at grazing (G2 / G1 -> 1 as the arrival's cosine -> 0) and
dips below its bulk value within a cosine of about alpha, so a near-mirror lobe's E is a function of
mu / alpha there: rows uniform in alpha (the first at 0.040) left every lobe under roughness 0.06 to an
interpolation from the all-ones row at alpha 0, and a white roughness-0.03 lobe returned 0.94 at 88.6
degrees. NA x NM nodes, read bilinearly in (sqrt(alpha), sqrt(mu)) (ggx.h msE / msEavg).
Deterministic; rerun only to change the grid.
"""
import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NM = 33, 65
ALPHA_MAX = 0.646 * 2.0
MU_MIN = 1e-4          # mu = 0 itself has no visible normal: its node holds the limit


def albedo(alpha, mu, k):
    """E for each mu in the array `mu` at one alpha."""
    if alpha <= 0.0:
        return np.ones_like(mu)
    a2 = alpha * alpha
    u = (np.arange(k) + 0.5) / k
    u1, u2 = np.meshgrid(u, u, indexing="ij")
    u1 = u1.ravel()
    u2 = u2.ravel()
    out = np.empty_like(mu)
    for j, m in enumerate(mu):
        a = np.array([np.sqrt(max(0.0, 1.0 - m * m)), 0.0, m])
        vs = np.array([alpha * a[0], alpha * a[1], a[2]])
        vs /= np.linalg.norm(vs)
        phi = 2.0 * np.pi * u2
        z = u1 * (1.0 + vs[2]) - vs[2]
        st = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
        hs = np.stack([st * np.cos(phi) + vs[0], st * np.sin(phi) + vs[1], z + vs[2]], -1)
        hl = np.stack([alpha * hs[:, 0], alpha * hs[:, 1], np.clip(hs[:, 2], 0.0, None)], -1)
        hl /= np.linalg.norm(hl, axis=-1, keepdims=True)
        b = 2.0 * (hl @ a)[:, None] * hl - a
        cb = b[:, 2]
        ca = m
        A = lambda c: np.sqrt(c * c + a2 * (1.0 - c * c))
        cbc = np.clip(cb, 1e-300, 1.0)
        w = np.where(cb > 0.0, cbc * (ca + A(ca)) / (cbc * A(ca) + ca * A(cbc)), 0.0)
        out[j] = w.mean()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--nm", type=int, default=NM)
    ap.add_argument("--out", default=os.path.join(ROOT, "src", "ggx_ms_table.h"))
    a = ap.parse_args()
    nm = a.nm
    alphas = ALPHA_MAX * (np.arange(NA) / (NA - 1)) ** 2
    mus = np.maximum((np.arange(nm) / (nm - 1)) ** 2, MU_MIN)
    E = np.empty((NA, nm))
    Eavg = np.empty(NA)
    for i, al in enumerate(alphas):
        E[i] = albedo(al, mus, a.k)
        print("alpha %.4f  E(0) %.4f  E(1) %.4f" % (al, E[i, 0], E[i, -1]), flush=True)
    # The multiple-scattering lobe's draw (ggx.h msSampleCos): per alpha row, the cumulative share of each
    # of the NT cells of the cosine grid (cell k spans t in [k, k+1] / NT, mu = t^2) under the density
    # (1 - E(mu)) mu, E the row's own values as the header holds them, linear in t as msE reads them;
    # the shares' sum is int (1 - E) mu dmu = (1 - Eavg) / 2, which is Eavg.
    # A draw picks a cell by it and, within it, a cosine in proportion to mu, so its weight
    # f_ms cos / pdf is nearly 1 - E(cos_a) whatever the exit.
    Ef = E.astype(np.float32).astype(np.float64)
    NT = nm - 1
    sub = (np.arange(256) + 0.5) / 256
    cdf = np.zeros((NA, NT + 1))
    for i in range(NA):
        share = np.empty(NT)
        for k in range(NT):
            t = (k + sub) / NT
            e = Ef[i, k] + (Ef[i, k + 1] - Ef[i, k]) * sub
            share[k] = np.mean((1.0 - e) * t * t * 2.0 * t) / NT     # (1 - E) mu dmu, dmu = 2t dt
        Eavg[i] = min(1.0, 1.0 - 2.0 * share.sum())
        share = np.clip(share, 0.0, None)
        tot = share.sum()
        if tot <= 1e-12:                    # no multiple scattering (alpha 0): never drawn
            share = np.full(NT, 1.0 / NT)
            tot = 1.0
        cdf[i, 1:] = np.cumsum(share) / tot
        cdf[i, -1] = 1.0
    for i, al in enumerate(alphas):
        print("alpha %.4f  Eavg %.6f" % (al, Eavg[i]), flush=True)
    NL = chr(10)
    lines = ["#pragma once",
             "// GENERATED by tools/gen_ggx_ms_table.py (k %d) -- do not edit by hand." % a.k,
             "// The GGX lobe's single-scattering albedo E(mu, alpha) for its multiple-scattering",
             "// compensation (ggx.h msE / msEavg): GGX_MS_NA alphas uniform in sqrt(alpha) over",
             "// [0, GGX_MS_ALPHA_MAX] (rows) by GGX_MS_NM cosines mu uniform in sqrt(mu) over [0, 1]",
             "// (columns), and",
             "// Eavg(alpha) = 2 int E mu dmu of these E; and the multiple-scattering lobe's draw",
             "// (ggx.h msSampleCos):",
             "// per row, the cumulative share of each of GGX_MS_NT cells of that cosine grid under the",
             "// density (1 - E(mu)) mu.",
             "// The initialisers are macros so the host and the device copies (ggx.h) hold the same",
             "// numbers.",
             "#define GGX_MS_ROWS_SQRT_ALPHA 1   // row i is alpha = GGX_MS_ALPHA_MAX (i / (NA - 1))^2",
             "#define GGX_MS_NA %d" % NA,
             "#define GGX_MS_NM %d" % nm,
             "#define GGX_MS_NT %d" % NT,
             "#define GGX_MS_ALPHA_MAX %.17g" % ALPHA_MAX,
             "#define GGX_MS_E_INIT { \\"]
    for i in range(NA):
        row = ", ".join("%.8ff" % v for v in E[i])
        lines.append("    %s%s \\" % (row, "," if i + 1 < NA else ""))
    lines.append("}")
    lines.append("#define GGX_MS_EAVG_INIT { \\")
    lines.append("    %s \\" % ", ".join("%.8ff" % v for v in Eavg))
    lines.append("}")
    lines.append("#define GGX_MS_CDF_INIT { \\")
    for i in range(NA):
        row = ", ".join("%.8ff" % v for v in cdf[i])
        lines.append("    %s%s \\" % (row, "," if i + 1 < NA else ""))
    lines.append("}")
    path = os.path.abspath(a.out)
    with open(path, "w", encoding="ascii", newline="") as f:
        f.write(NL.join(lines) + NL)
    print("wrote", path)


if __name__ == "__main__":
    main()
