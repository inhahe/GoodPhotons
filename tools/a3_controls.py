"""Controls for the A3 simulation, run BEFORE its numbers are believed.

Two answers are known in advance:

  1. n = 1.0 -- there is no coat. No refraction, no Fresnel, no trapped light, F_dr = 0 so
     a_eff = a. The brute-force simulation and the model must then agree EXACTLY (to sampling
     noise). If they do not, the difference measured at n = 1.5 is my own machinery, not the coat.

  2. A LAMBERTIAN body under a real coat. This is the case a_eff was DERIVED for -- the body's
     outgoing distribution inside the coat is cosine-weighted however the light arrived, which is
     exactly the assumption F_dr averages over. So truth and model must agree here too, and the
     A3 numbers are only meaningful to the extent that a DIRECTIONAL body departs from this.

The second control is the sharper one: it separates "the simulation is broken" from "the model is
wrong for directional bodies specifically", which is the whole claim.
"""
import math

import os
import sys

import numpy as np

# tools/ and scraps/ are both one level below the repo root, so a sibling import works
# from either -- see known-issues, "design.md's measurement rigs lived in git-ignored
# scraps/", which records why a home one level DEEPER would silently break this.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3_snell as S


def simulate_lambert(theta_i_deg, a, n=1.5, samples=400000, max_bounce=40):
    """Same brute force, but the body scatters COSINE-weighted (Lambertian) with albedo a."""
    rng = np.random.default_rng(11)
    cos_tc = math.sqrt(max(0.0, 1.0 - 1.0 / (n * n)))
    ti = math.radians(theta_i_deg)
    r_enter = float(S.fresnel_dielectric(math.cos(ti), n))
    w = np.full(samples, 1.0 - r_enter)
    tot = 0.0
    for _ in range(max_bounce):
        if not np.any(w > 1e-6):
            break
        u1, u2 = rng.random(samples), rng.random(samples)
        cz = np.sqrt(u1)                                  # cosine-weighted, independent of arrival
        w = w * a
        r_out = S.fresnel_dielectric(cz, 1.0 / n)         # 1 beyond the critical angle
        tot += float((w * (1.0 - r_out)).sum()) / samples
        w = w * r_out
    return r_enter + tot


def model_lambert(theta_i_deg, a, n=1.5):
    fdr = S.internal_fresnel_diffuse(n)
    a_eff = a * (1.0 - fdr) / (1.0 - a * fdr)
    r = float(S.fresnel_dielectric(math.cos(math.radians(theta_i_deg)), n))
    return r + (1.0 - r) * a_eff


print("CONTROL 1 -- n = 1.0, i.e. NO COAT. Truth and model must be identical.")
S.N, S.COS_TC = 1.0, 0.0
print("  %-5s %-6s %-5s | %-10s %-10s %s" % ("t_i", "alpha", "a", "truth", "model", "diff"))
worst = 0.0
for alpha in (0.05, 0.2, 0.5):
    for ti in (0, 45, 70):
        at, _, _ = S.simulate(ti, alpha, 0.9)
        am, _, _ = S.model(ti, alpha, 0.9)
        d = 100.0 * (am / max(at, 1e-9) - 1.0)
        worst = max(worst, abs(d))
        print("  %-5d %-6.2f %-5.2f | %-10.4f %-10.4f %+.2f %%" % (ti, alpha, 0.9, at, am, d))
print("  -> worst |diff| = %.2f %%  %s\n"
      % (worst, "OK: the machinery agrees with itself" if worst < 2.0 else
         "BROKEN: the simulation and the model disagree with NO coat present"))

S.N, S.COS_TC = 1.5, math.sqrt(1.0 - 1.0 / 2.25)
print("CONTROL 2 -- a LAMBERTIAN body under a real coat (n = 1.5).")
print("This is the case a_eff was derived for, so it must agree. If it does, a departure at a")
print("DIRECTIONAL body is about directionality and not about the coat model in general.")
print("  %-5s %-5s | %-10s %-10s %s" % ("t_i", "a", "truth", "model", "diff"))
worst2 = 0.0
for a in (0.9, 0.5, 0.2):
    for ti in (0, 45, 70):
        at, am = simulate_lambert(ti, a), model_lambert(ti, a)
        d = 100.0 * (am / max(at, 1e-9) - 1.0)
        worst2 = max(worst2, abs(d))
        print("  %-5d %-5.2f | %-10.4f %-10.4f %+.2f %%" % (ti, a, at, am, d))
print("  -> worst |diff| = %.2f %%  %s" % (worst2, "OK" if worst2 < 3.0 else "a_eff is NOT exact even for a Lambertian body"))
