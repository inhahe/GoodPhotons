"""Closed-form single-scattering reference for scenes/_slab_ss.ftsl (mode-J gate 3).

Why this exists
---------------
Mode `J` (UPBP)'s other validation gates all compare it to another ftrace mode: gate 1 to
mode `D` with beams off, gate 2 to a second implementation of its own MIS weight, gate 4 to
a converged mode `D`. Every one of them would pass unchanged if modes `D` and `J` shared a
common-mode error in the volumetric physics they both call — a factor of 2 in sigma_s, an
isotropic phase function normalised to 1 instead of 1/(4pi), a point-source intensity of
Phi/(2pi). This script computes the answer from the rendering equation directly, with a
deterministic 1-D quadrature and no Monte Carlo anywhere, so mode `J` is checked against
physics instead of against its own siblings.

The scene is built (see its header comment) so that with `-max-bounce 1` the ONLY path that
exists is eye -> one medium scatter -> emitter. For a camera ray o + t*d, with an emitter of
radiant intensity I at P (a uniformly-emitting Lambertian sphere is exactly a point source of
intensity Phi/(4pi) at every external point), an isotropic phase function p = 1/(4pi), and a
homogeneous medium clipped to an axis-aligned box:

    L = integral over the in-box span of
            Tr_cam(t) * sigma_s * p * I * Tr_lit(t) / r(t)^2   dt

    Tr_cam(t) = exp(-sigma_t * (in-box path length from o to o+t*d))
    Tr_lit(t) = exp(-sigma_t * (in-box path length from o+t*d to P))
    r(t)      = |P - (o + t*d)|

`I` and the emitter's spectrum only ever multiply the result, so this script computes the
integral with I*sigma_s*p folded into 1 and reports the answer up to ONE global constant per
channel. That constant is what the two halves of the gate are about:

  (a) SHAPE  -- image / reference must be FLAT across the frame. No free parameter can absorb
      an error in Tr_cam, Tr_lit, 1/r^2, the merge kernel's radius dependence or the beam x
      ray 1/sin(theta), because all of those vary pixel to pixel and the fit is one number.
  (b) SCALE  -- the fitted constant must agree across modes R, D and J. Mode R shares none of
      the bidirectional or beam machinery, so if all three land on the same constant the
      absolute level is pinned too.

Rendering the inputs -- mode R MUST use `-device gpu`
-----------------------------------------------------
    ftrace scenes/_slab_ss.ftsl -mode R -device gpu -r 96 -max-bounce 1 -time 60 -hdr \
           -o png/slab_r.png -window-min -interval 30
    ftrace scenes/_slab_ss.ftsl -mode D -device cpu -r 96 -max-bounce 1 -time 90 -hdr \
           -o png/slab_d.png -window-min -interval 45
    ftrace scenes/_slab_ss.ftsl -mode J -device cpu -r 96 -max-bounce 1 -n 200000 -time 90 \
           -hdr -o png/slab_j.png -window-min -interval 45

The CPU backward tracer collapses every authored medium into ONE global homogeneous haze and
ignores `bounds` regions (it prints a `[medium]` warning saying so at startup). This scene is
nothing *but* a bounded box medium, so `-mode R -device cpu` renders an UNBOUNDED fog -- a
different scene, not a noisier one -- and fits at 304x the right level with a noise rms of 51.
That looks exactly like a catastrophic gate-3(b) failure and is not one. So: mode R on the GPU.
Modes D and J honour the bounds on either device.

`-max-bounce 1` is load-bearing rather than a speed knob: it is what leaves single scattering
as the only transport in the frame, which is the whole premise of the closed form below.

Usage:  python tools/slab_ss_ref.py png/slab_r.pfm [png/slab_d.pfm ...]
        (scene parameters are hard-coded below to match scenes/_slab_ss.ftsl)
"""

import sys

import numpy as np

# ---- scene parameters: must match scenes/_slab_ss.ftsl -------------------------------
BOX_MIN = np.array([-1.0, -1.0, -1.0])
BOX_MAX = np.array([3.0, 1.0, 1.0])
SIGMA_T = 2.0
ALBEDO = 1.0
LIGHT_P = np.array([2.5, 0.0, 0.5])
EYE = np.array([0.0, 0.0, 5.0])
LOOK_AT = np.array([0.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])
FOV_Y = 30.0
RES = 96

NQUAD = 4096  # Simpson intervals along each camera ray (see convergence note at the bottom)


def read_pfm(path):
    with open(path, "rb") as f:
        hdr = f.readline().strip()
        assert hdr in (b"PF", b"Pf"), hdr
        ch = 3 if hdr == b"PF" else 1
        w, h = map(int, f.readline().split())
        scale = float(f.readline().strip())
        data = np.frombuffer(f.read(w * h * ch * 4),
                             dtype="<f4" if scale < 0 else ">f4")
        return np.flipud(data.reshape(h, w, ch)).astype(np.float64)


def slab_span(o, d, tmax=None):
    """Ray/AABB entry+exit parameters, vectorised over leading axes of `o`/`d`.

    Returns (t0, t1, hit). `d` need not be normalised for the algebra, but every caller here
    passes a unit direction so t is a true distance.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        ta = (BOX_MIN - o) * inv
        tb = (BOX_MAX - o) * inv
    tlo = np.minimum(ta, tb)
    thi = np.maximum(ta, tb)
    # A component with d == 0 gives +-inf or nan; nanmax/nanmin drop the nan slabs, which is
    # the right answer only if the origin is inside that slab. Handle it explicitly instead.
    parallel = d == 0.0
    inside = (o >= BOX_MIN) & (o <= BOX_MAX)
    tlo = np.where(parallel, np.where(inside, -np.inf, np.inf), tlo)
    thi = np.where(parallel, np.where(inside, np.inf, -np.inf), thi)
    t0 = np.maximum(np.max(tlo, axis=-1), 0.0)
    t1 = np.min(thi, axis=-1)
    if tmax is not None:
        t1 = np.minimum(t1, tmax)
    return t0, t1, t1 > t0


def camera_rays():
    """ftrace's pinhole convention: +x right, +y up, look_at down -z of the camera basis."""
    fwd = LOOK_AT - EYE
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, UP)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    th = np.tan(np.radians(FOV_Y) * 0.5)
    tw = th * 1.0  # square film
    j, i = np.mgrid[0:RES, 0:RES]            # j = row from the TOP, i = column from the LEFT
    ndc_x = (i + 0.5) / RES * 2.0 - 1.0
    ndc_y = 1.0 - (j + 0.5) / RES * 2.0
    d = (fwd[None, None, :]
         + (ndc_x * tw)[..., None] * right[None, None, :]
         + (ndc_y * th)[..., None] * up[None, None, :])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d


def reference(nquad=NQUAD):
    """The single-scattering field, up to one global constant. Shape (RES, RES).

    Evaluated a row at a time: the full (RES, RES, nquad+1, 3) sample-position array would be
    ~900 MB at the default nquad, and one row of it is 9 MB.
    """
    dirs = camera_rays()
    sigma_s = SIGMA_T * ALBEDO
    n = nquad
    u = np.linspace(0.0, 1.0, n + 1)                       # (n+1,)
    wts = np.ones(n + 1)
    wts[1:-1:2] = 4.0
    wts[2:-1:2] = 2.0

    out = np.zeros((RES, RES))
    for row in range(RES):
        d = dirs[row]                                      # (W, 3)
        o = np.broadcast_to(EYE, d.shape)
        t0, t1, hit = slab_span(o, d)
        t0 = np.where(hit, t0, 0.0)
        t1 = np.where(hit, t1, 0.0)

        # Composite Simpson over [t0, t1] per pixel, on a per-pixel-uniform grid. `n` must be
        # even. The integrand is smooth (a product of exponentials and 1/r^2, with r bounded
        # well away from 0 because the emitter is outside the frustum), so Simpson converges
        # fast -- see the convergence check in __main__.
        span = (t1 - t0)[:, None]                          # (W, 1)
        t = t0[:, None] + span * u                         # (W, n+1)
        x = o[:, None, :] + t[..., None] * d[:, None, :]   # (W, n+1, 3)

        # Camera-side transmittance: the ray enters the box at t0, so the in-medium path
        # length from the eye to x is simply t - t0 (the eye is outside, the box is convex).
        tau_cam = SIGMA_T * (t - t0[:, None])

        # Light-side transmittance: clip the x -> P segment against the box. The emitter is
        # outside the box, so the segment leaves the medium at a wall and travels the rest in
        # vacuum; x is inside, so the entry parameter is 0 and the in-medium length is just
        # the exit parameter.
        w = LIGHT_P - x
        r = np.linalg.norm(w, axis=-1)                     # (W, n+1)
        wn = w / r[..., None]
        _, s1, _ = slab_span(x, wn, tmax=r)
        tau_lit = SIGMA_T * np.maximum(s1, 0.0)

        f = np.exp(-(tau_cam + tau_lit)) * sigma_s / (r * r)
        L = (f * wts).sum(axis=-1) * (span[:, 0] / n) / 3.0
        out[row] = np.where(hit, L, 0.0)
    return out


def main():
    paths = sys.argv[1:]

    ref = reference()
    if not paths or paths[0] == "--selftest":
        # Quadrature convergence: halving the interval count must not move the answer by more
        # than the tolerance the gate is quoted to. If it does, NQUAD is too small and every
        # number below is measuring the reference's own error rather than the renderer's.
        half = reference(NQUAD // 2)
        m = ref > 0
        drift = np.abs(half[m] / ref[m] - 1.0)
        print(f"quadrature self-test: n={NQUAD} vs n={NQUAD//2}, "
              f"max drift {drift.max():.3e}, median {np.median(drift):.3e}")
        if not paths:
            print(__doc__)
            return 1
        paths = paths[1:]
        if not paths:
            return 0
    inside = ref > 0.0
    print(f"reference: {inside.sum()} of {RES*RES} px hit the medium; "
          f"dynamic range {ref[inside].max()/ref[inside].min():.4g}x "
          f"(max {ref[inside].max():.6g}, min {ref[inside].min():.6g}, arbitrary units)")

    # Quartiles of the REFERENCE, so every image is scored on the same partition. A shape
    # error that only shows in the dim, deep part of the slab -- exactly where the merges are
    # supposed to matter -- would otherwise be swamped by the bright near wall.
    q = np.quantile(ref[inside], [0.25, 0.5, 0.75])
    band = np.digitize(ref, q) * inside

    print()
    print(f"{'image':22s} {'scale':>12s} {'Q1':>9s} {'Q2':>9s} {'Q3':>9s} {'Q4':>9s} "
          f"{'noise rms':>10s}")
    base = None
    for p in paths:
        im = read_pfm(p)
        assert im.shape[0] == RES and im.shape[1] == RES, (p, im.shape)
        L = im.mean(axis=2)
        good = inside & np.isfinite(L)
        if good.sum() != inside.sum():
            print(f"note: {p} has {inside.sum()-good.sum()} non-finite px inside the medium")

        # The one constant a shape comparison is allowed to absorb, fitted as an ENERGY ratio
        # (sum of image over sum of reference) rather than as a median or a least-squares fit
        # on L. That matters: a Monte Carlo image's per-pixel ratio is strongly right-skewed
        # (most pixels land below the mean, a few far above), so a median ratio reads several
        # times too LOW on a noisy render and would look exactly like a missing-energy bug.
        # A sum is linear in the samples, so it is unbiased at any sample count.
        def energy(mask):
            return float(L[mask].sum() / ref[mask].sum())

        scale = energy(good)
        rel = L[good] / ref[good] / scale - 1.0
        rms = float(np.sqrt(np.mean(rel ** 2)))
        qs = [energy(good & (band == b)) / scale if (good & (band == b)).any() else float("nan")
              for b in range(4)]
        tag = f"{scale:12.6g}" if base is None else f"{scale/base:11.4f}x"
        if base is None:
            base = scale
        name = p.replace("\\", "/").split("/")[-1]
        print(f"{name:22s} {tag} {qs[0]:9.4f} {qs[1]:9.4f} {qs[2]:9.4f} {qs[3]:9.4f} "
              f"{rms:10.4f}")
    print()
    print("scale:      first image absolute (arbitrary units), the rest RELATIVE to it -- these")
    print("            must all be 1.0000x, that is gate 3(b), the absolute-level check.")
    print("Q1..Q4:     the same energy ratio restricted to one quartile of REFERENCE brightness")
    print("            (Q1 = the dimmest quarter, deep in the slab), divided by the frame's own")
    print("            scale. A flat 1.0000 row is gate 3(a); a trend across the row is a shape")
    print("            error no global scale can hide.")
    print("noise rms:  rms of image/(scale*reference) - 1, per pixel. This one IS just the")
    print("            render's noise; it is here to say how much to trust the row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
