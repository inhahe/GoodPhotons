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
// This header only carries the small shared constant; the per-renderer plumbing
// (per-λ throughput arrays, de-hero handling, multi-λ NEE) lives in each tracer.
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
}
