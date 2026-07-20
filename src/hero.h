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
// This header only carries the small shared constant; the per-renderer plumbing
// (per-λ throughput arrays, de-hero handling, multi-λ NEE) lives in each tracer.
#pragma once

namespace hero {
// Number of wavelengths carried per path (hero + C-1 secondaries). C == 1 reduces
// the estimator bit-identically to the classic single-wavelength tracer, so this is
// the single knob that turns hero sampling on. 4 is the usual sweet spot.
constexpr int kHeroC = 4;
}
