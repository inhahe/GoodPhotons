"""A3: how much does Snell refraction into the body actually change a coated DIRECTIONAL body?

Two questions the plan needs answered before any code is written:

 1. There is an exact result worth knowing first. Refract into the coat, reflect off a body whose
    microfacet normal IS the coat's normal (a SMOOTH body), refract back out:

        sin t_t = sin t_i / n   ->  mirror about n (polar unchanged)  ->  sin t_out = n sin t_t = sin t_i

    so the outgoing direction is EXACTLY the mirror of the incoming one -- identical to doing no
    refraction at all. Snell in-and-out of a smooth body is the IDENTITY. So A3 is not "a coat bends
    the light"; it is specifically about the mismatch between the body's MICROFACET normal and the
    coat's normal, which only exists for a ROUGH body. That is also precisely the case where some of
    the body's lobe lands beyond the critical angle and is trapped by total internal reflection --
    which has no closed form. The two are co-extensive: you cannot have the refraction effect
    without the TIR problem.

 2. So the real question is how WRONG the current model is for a rough body under a coat, since the
    fix is architectural. This measures it against a brute-force simulation of the actual layered
    system: refract in, bounce on a GGX body, escape if inside the critical cone else TIR back down
    and bounce again, to convergence.

Compared quantities:
   * directional albedo (energy) -- is a_eff = a(1-F_dr)/(1-a F_dr) the right energy for a
     DIRECTIONAL body, or only for a Lambertian one?
   * the width of the escaping lobe -- refraction compresses angles toward the normal, so a rough
     body under a coat should look SMOOTHER than it is. How much smoother?
"""
import math

import numpy as np

N = 1.5
COS_TC = math.sqrt(1.0 - 1.0 / (N * N))          # cos of the critical angle, 0.745356
RNG = np.random.default_rng(7)


def fresnel_dielectric(cos_i, eta):
    """Unpolarised Fresnel reflectance for cos_i measured in the medium the ray starts in.

    `eta` is n_transmitted / n_incident. Returns 1.0 on total internal reflection."""
    c = np.clip(np.abs(cos_i), 0.0, 1.0)
    s2t = (1.0 - c * c) / (eta * eta)
    tir = s2t >= 1.0
    ct = np.sqrt(np.clip(1.0 - s2t, 0.0, 1.0))
    rs = (c - eta * ct) / np.maximum(c + eta * ct, 1e-12)
    rp = (eta * c - ct) / np.maximum(eta * c + ct, 1e-12)
    r = 0.5 * (rs * rs + rp * rp)
    return np.where(tir, 1.0, np.clip(r, 0.0, 1.0))


def internal_fresnel_diffuse(n):
    return -1.440 / (n * n) + 0.710 / n + 0.668 + 0.0636 * n


def refract_dir(d, n_vec, eta):
    """Refract d (unit, pointing INTO the surface) about n_vec. eta = n_i / n_t. None if TIR."""
    ci = -np.dot(d, n_vec)
    s2 = eta * eta * (1.0 - ci * ci)
    if s2 >= 1.0:
        return None
    return eta * d + (eta * ci - math.sqrt(1.0 - s2)) * n_vec


def sample_ggx_half(alpha, size):
    """Half-vectors from the GGX distribution D(m)|m.n|, in the local frame (z = up)."""
    u1, u2 = RNG.random(size), RNG.random(size)
    ct = np.sqrt((1.0 - u1) / (1.0 + (alpha * alpha - 1.0) * u1))
    st = np.sqrt(np.clip(1.0 - ct * ct, 0.0, 1.0))
    ph = 2.0 * math.pi * u2
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=-1)


def smith_g1(v_z, alpha):
    """Smith GGX masking for a direction with |cos| = v_z."""
    c = np.clip(np.abs(v_z), 1e-6, 1.0)
    t2 = (1.0 - c * c) / (c * c)
    return 2.0 / (1.0 + np.sqrt(1.0 + alpha * alpha * t2))


def simulate(theta_i_deg, alpha, a, samples=400000, max_bounce=24):
    """Brute-force the real layered system. Returns (albedo, mean escape cos, escape-cos spread)."""
    ti = math.radians(theta_i_deg)
    wi = np.array([math.sin(ti), 0.0, math.cos(ti)])          # pointing away from the surface
    up = np.array([0.0, 0.0, 1.0])

    # --- the coat interface, entered from air -------------------------------------------------
    r_enter = float(fresnel_dielectric(math.cos(ti), N))
    d_in = refract_dir(-wi, up, 1.0 / N)                      # into the coat, pointing down
    if d_in is None:
        return r_enter, 0.0, 0.0
    v = -d_in                                                 # the body sees this, pointing up
    v = np.broadcast_to(v, (samples, 3)).copy()
    w = np.full(samples, (1.0 - r_enter))                     # what got in
    esc_w, esc_c = [], []

    for _ in range(max_bounce):
        alive = w > 1e-5
        if not np.any(alive):
            break
        idx = np.flatnonzero(alive)
        m = sample_ggx_half(alpha, len(idx))
        vv = v[idx]
        vdotm = np.einsum("ij,ij->i", vv, m)
        o = 2.0 * vdotm[:, None] * m - vv                     # reflect about the microfacet
        # GGX sampling weight for D|m.n| sampling: F * G2 / G1 ... use the standard
        #   weight = F * G(v,o,m) * |v.m| / (|v.n| * |m.n|)
        g = smith_g1(vv[:, 2], alpha) * smith_g1(o[:, 2], alpha)
        wt = a * g * np.abs(vdotm) / np.maximum(np.abs(vv[:, 2]) * np.abs(m[:, 2]), 1e-9)
        wnew = w[idx] * np.clip(wt, 0.0, 4.0)
        below = o[:, 2] <= 1e-6                               # absorbed into the body's back side
        wnew = np.where(below, 0.0, wnew)

        cz = np.clip(o[:, 2], 0.0, 1.0)
        # meet the coat from INSIDE: reflect back with R, escape with 1-R
        r_out = fresnel_dielectric(cz, 1.0 / N)               # 1.0 where cz < cos_tc (TIR)
        esc = wnew * (1.0 - r_out)
        keep = esc > 1e-6
        if np.any(keep):
            # refract the escaping directions out into air: sin_out = n sin_in
            s_in = np.sqrt(np.clip(1.0 - cz[keep] ** 2, 0.0, 1.0))
            s_out = np.clip(N * s_in, 0.0, 1.0)
            esc_w.append(esc[keep])
            esc_c.append(np.sqrt(np.clip(1.0 - s_out ** 2, 0.0, 1.0)))
        # the trapped part carries on, mirrored about the coat normal
        o_ref = o.copy()
        o_ref[:, 2] = -o_ref[:, 2]
        v[idx] = -o_ref                                       # heading back down = seen from below
        v[idx, 2] = np.abs(v[idx, 2])
        w[idx] = wnew * r_out
        w[idx[below]] = 0.0

    if not esc_w:
        return r_enter, 0.0, 0.0
    ew = np.concatenate(esc_w)
    ec = np.concatenate(esc_c)
    tot = float(ew.sum()) / samples
    mc = float((ew * ec).sum() / max(ew.sum(), 1e-12))
    sd = float(np.sqrt((ew * (ec - mc) ** 2).sum() / max(ew.sum(), 1e-12)))
    return r_enter + tot, mc, sd


def model(theta_i_deg, alpha, a, samples=400000):
    """What ftrace renders today: coat lobe F(t_i), plus the body lobe with UNREFRACTED directions
    and albedo a_eff. Same estimator, so the two numbers are comparable."""
    ti = math.radians(theta_i_deg)
    fdr = internal_fresnel_diffuse(N)
    a_eff = a * (1.0 - fdr) / (1.0 - a * fdr)
    r_enter = float(fresnel_dielectric(math.cos(ti), N))
    v = np.broadcast_to(np.array([math.sin(ti), 0.0, math.cos(ti)]), (samples, 3)).copy()
    m = sample_ggx_half(alpha, samples)
    vdotm = np.einsum("ij,ij->i", v, m)
    o = 2.0 * vdotm[:, None] * m - v
    g = smith_g1(v[:, 2], alpha) * smith_g1(o[:, 2], alpha)
    wt = a_eff * g * np.abs(vdotm) / np.maximum(np.abs(v[:, 2]) * np.abs(m[:, 2]), 1e-9)
    w = np.where(o[:, 2] <= 1e-6, 0.0, np.clip(wt, 0.0, 4.0)) * (1.0 - r_enter)
    cz = np.clip(o[:, 2], 0.0, 1.0)
    tot = float(w.sum()) / samples
    mc = float((w * cz).sum() / max(w.sum(), 1e-12))
    sd = float(np.sqrt((w * (cz - mc) ** 2).sum() / max(w.sum(), 1e-12)))
    return r_enter + tot, mc, sd


print("A3 -- Snell into a coated DIRECTIONAL body: how wrong is the current model?")
print("Smooth coat n = 1.5 over a GGX body. 'truth' brute-forces the real layered system")
print("(refract in, bounce, escape inside the critical cone else TIR back down and bounce again).")
print("'model' is what ftrace renders today: unrefracted body directions, albedo a_eff.\n")
print("  a=body albedo, alpha=body roughness.  Lobe width = std dev of the escaping cos(theta).\n")
hdr = "  %-5s %-6s %-5s | %-17s | %-17s | %s"
print(hdr % ("t_i", "alpha", "a", "albedo truth/model", "lobe width t/m", "verdict"))
print("  " + "-" * 78)
for a in (0.9, 0.5):
    for alpha in (0.05, 0.2, 0.5):
        for ti in (0, 45, 70):
            at, mct, sdt = simulate(ti, alpha, a)
            am, mcm, sdm = model(ti, alpha, a)
            de = 100.0 * (am / max(at, 1e-9) - 1.0)
            dw = 100.0 * (sdm / max(sdt, 1e-9) - 1.0)
            print("  %-5d %-6.2f %-5.2f | %.4f / %.4f %+6.1f%% | %.4f / %.4f %+6.1f%% |"
                  % (ti, alpha, a, at, am, de, sdt, sdm, dw))
