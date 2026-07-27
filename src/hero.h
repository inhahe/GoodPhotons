// Hero-wavelength spectral sampling (Wilkie et al. 2014, "Hero Wavelength
// Spectral Sampling"). One path carries a HERO wavelength plus C-1 stratified
// SECONDARY wavelengths drawn from the same emission-importance CDF. The path
// GEOMETRY is decided by the hero wavelength alone; the secondaries "ride along"
// the identical vertices/directions, each accumulating its own per-wavelength
// throughput and radiance. This slashes chromatic (colour) noise at essentially
// unchanged traversal cost, because C wavelengths share one BVH walk.
//
// At a DISPERSIVE / wavelength-switching event (a dielectric refraction, a
// diffraction grating, a fluorescent Stokes shift, …) the secondaries can no
// longer follow the hero's path, so they are TERMINATED ("de-hero'd"): the hero
// throughput is boosted ×C and the secondaries stop contributing further. The
// portion of the estimate before de-hero is the low-variance C-wavelength average;
// the portion after gracefully degrades to a full-weight single-wavelength
// estimate — both unbiased, matching PBRT-v4's TerminateSecondary() convention.
//
// SPLIT-AT-DISPERSION (`-herosplit`, opt-in) is the alternative policy: instead of
// terminating the secondaries at a dispersive event, all C wavelengths CONTINUE,
// each along its own per-λ direction, turning one bundle into C now-monochromatic
// sub-paths that fan out through the glass. Same mean (both estimators are
// unbiased), but the chromatic spread is resolved geometrically instead of being
// averaged over one shared direction — which is what makes prism / rainbow /
// dispersive-caustic shots crisp. It is off by default purely because it costs C×
// the traversal work past the split (linear, not exponential: once monochromatic a
// sub-path never re-splits).
//
// ---------------------------------------------------------------------------------
// THE FOUR POLICIES. Every hero tracer — forward `tracePhotonHero` (render.h) and its
// device twin `shadeStepHero` (render_cuda.cu), backward `radianceHero` (backward.h) /
// `bkRadianceHero`, and BDPT `randomWalk`/`connectBDPT` (bdpt.h) / `dRandomWalk` —
// implements the SAME four rules. They are stated ONCE here; the tracers reference
// this block rather than each restating it, because the prose used to be duplicated
// near-verbatim in three files and every fix had to be made three times (see the
// hero section of known-issues.md for the two bugs that cost).
//
// (1) STRATIFICATION. The C wavelengths come from ONE uniform variate: the hero takes
//     `u`, secondary i takes `u + i/C` wrapped into [0,1), all pushed through the same
//     emission-importance CDF. `sampleBundle()` below is that draw, shared by all
//     three CPU tracers. Only the HERO is required to have a positive pdf; a secondary
//     that lands in a zero-mass bin simply carries weight 0.
//
// (2) DE-HERO at λ-dependent DIRECTIONS. A lobe collapses the bundle to the hero
//     (boosted ×C) iff its outgoing direction — or the outgoing wavelength itself —
//     depends on λ: Dielectric, ThinFilm, Multilayer, Grating, HalfMirror, Fluorescent.
//     A lobe that is merely delta does NOT de-hero: Mirror reflects and a Filter gel
//     passes straight through regardless of λ, so the bundle rides on with only its
//     per-λ coefficient differing. (Glossy likewise; in BDPT it isn't even delta.)
//     Also de-hero'd, for a different reason, are the λ-dependent *decisions* Mix and
//     Layered — see known-issues.md.
//
// (3) ANALOG RR IS MAX-OVER-LIVE-λ, NEVER HERO-ONLY. Where the scalar tracer survives a
//     coefficient c by analog Russian roulette, the bundle uses `q = max_i c_i` and the
//     survivors reweight by `c_i / q ≤ 1`. Rolling the coin on the hero alone is wrong
//     twice over: it kills live secondaries whenever `c_hero == 0` (a Wratten gel is 0
//     across most of the spectrum), and it AMPLIFIES survivors by `c_i / c_hero`, which
//     on the built-in coloured walls (ρ ∈ 0.05 … 0.75) is a 15× weight spike per diffuse
//     bounce. At `nUp == 1` this reduces to `q == c[0]`, weight 1.0 — the scalar code
//     verbatim, same rng draws in the same order, so `-heroc 1` stays bit-identical.
//
// (4) PER-λ FACTORS ARE ABSOLUTE, NOT RATIOS TO THE HERO. Store `f_i` (or `f_i·cos/pdf`),
//     never `f_i / f_hero`: the ratio is undefined exactly where the bundle helps most,
//     and every "is it black yet?" early-out must therefore be a max-over-live-λ test —
//     the hero can legitimately be zero where a secondary is wide open.
//
// Two rules that are NOT shared, and deliberately so: the FORWARD tracers keep an
// energy ledger, so rule (3)'s reweight — which is deterministic absorption — must be
// booked there (`e.absorbed += beta[i]*(1-w)`) and must not be anywhere else; and BDPT
// normalises a connection by `1/min(nUp_light, nUp_eye)` at the splat instead of folding
// a ×C de-hero boost into the vertex throughputs, since two independently de-hero'd
// subpaths would square it.
//
// Beyond those, this header carries only the small shared constants and the bundle
// draw; the per-renderer plumbing (per-λ throughput arrays, de-hero handling, multi-λ
// NEE) genuinely differs per estimator and lives in each tracer.
#pragma once

namespace hero {
// DEFAULT number of wavelengths carried per path (hero + C-1 secondaries). C == 1
// reduces the estimator bit-identically to the classic single-wavelength tracer.
// 4 is the usual sweet spot; the runtime `-heroc N` flag overrides it per render.
constexpr int kHeroC = 4;
// COMPILE-TIME upper bound for the runtime count, so the per-path throughput/λ arrays
// can live on the stack as fixed `[kHeroMax]` buffers regardless of the `-heroc` value.
// `-heroc N` is clamped to [1, kHeroMax]. Bump this if a larger bundle is ever wanted.
constexpr int kHeroMax = 8;

// Runtime opt-in for the split-at-dispersion policy (`-herosplit`). Unlike heroC —
// which the drivers vary per pass (the meter pre-pass, the media/GRIN/lens gate) and
// therefore thread explicitly through every renderer entry point — this is a single
// global policy choice: main() sets it once while parsing argv, before any render
// starts, and every tracer only ever reads it. `Renderer::heroSplit` default-
// initialises from it, so no call site needs a new parameter.
inline bool gSplit = false;

// POLICY (1): draw one bundle's C wavelengths from `sampler`'s emission-importance CDF
// using a single uniform variate `u` (see the policy block above). Templated on the
// sampler so this header stays dependency-free and both callers work: the forward
// tracer samples a single emitter's `Emitter::spd`, while the backward/BDPT tracers
// sample the scene-wide `Scene::emitSampler` — both `EmissionSampler`.
//
// Writes `lam[0..C)` and `pdf[0..C)`. Returns FALSE iff the HERO's pdf is non-positive,
// in which case the caller must discard the whole sample (there is nothing to carry).
// Secondaries are allowed to come back with pdf 0; each tracer zeroes that λ's weight in
// its own currency (invPdf, throughput, or beta), so they are not checked here.
template <class Sampler>
inline bool sampleBundle(const Sampler& sampler, double u, int C,
                         double* lam, double* pdf) {
    pdf[0] = 0.0;
    lam[0] = sampler.sampleAt(u, pdf[0]);
    if (pdf[0] <= 0.0) return false;
    for (int i = 1; i < C; ++i) {
        double uu = u + (double)i / (double)C;
        if (uu >= 1.0) uu -= 1.0;            // wrap into [0,1)
        pdf[i] = 0.0;
        lam[i] = sampler.sampleAt(uu, pdf[i]);
    }
    return true;
}

// POLICIES (3) and (4): the max over the LIVE λ of a per-λ array. Used both as an analog
// RR survival probability `q` and as the "is the whole bundle black?" early-out test.
// With n == 1 it returns a[0], so every scalar test it replaces is unchanged.
inline double maxOf(const double* a, int n) {
    double m = a[0];
    for (int i = 1; i < n; ++i) if (a[i] > m) m = a[i];
    return m;
}
}
