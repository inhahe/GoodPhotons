// Backward path tracer — an INDEPENDENT reference renderer for validating the
// forward light tracer (model B). It shoots rays from the camera and estimates
// incident spectral radiance per pixel using next-event estimation (NEE) to the
// area light. The forward and backward estimators sample the same scene with
// opposite transport; at convergence they must produce the same image (up to a
// single global scale set by each side's measurement convention). A structured
// residual after best-fit scaling therefore flags a transport/camera bug — which
// energy conservation alone cannot catch.
//
// It deliberately REUSES the material primitives (Fresnel via Renderer::
// refractOrReflect, the glossy lobe via sampleGlossy) so the two renderers agree
// on materials by construction. That isolates the validation to the part that
// actually differs — the transport and camera math (connect / G / We).
//
// Material treatment mirrors the forward tracer exactly:
//   Diffuse    : NEE to the light + cosine-sampled continuation, Russian roulette
//                on the albedo (throughput unchanged on survival).
//   Mirror     : specular reflect, RR on reflectance; emission added on hit.
//   Glossy     : power-cosine lobe around the mirror dir, RR on reflectance.
//   HalfMirror : stochastic reflect/transmit, lossless.
//   Dielectric : Fresnel-weighted reflect/refract, lossless.
//   Fluorescent: bispectral reradiation (Stokes shift). The elastic base reflects
//                at the output wavelength; the fluorescent channel excites at a
//                separately-sampled input wavelength lambdaIn and reradiates at the
//                output wavelength (colour ~ M). Direct excitation is NEE'd; indirect
//                excitation is carried by a stochastic wavelength-switching
//                continuation. This is the unbiased backward adjoint of the forward
//                tracer's fluoroInteract(), so -scene fluoro now validates with modes
//                R/V (previously fluoro was forward-only).
// Participating media (scene.backwardMedium() / -fog) IS supported here: camera and
// scattered rays sample volume free-flight, and volume vertices do phase-function
// NEE to the light (neeVolume). So -fog CAN be combined with modes R/V, which is
// how the forward fog transport is cross-validated.
// Emission is added only when a light is reached via the camera ray or a
// specular/near-specular bounce; diffuse arrivals are covered by NEE (no double
// counting).
// An environment light (scene.envIndex >= 0) is treated as an infinitely-distant
// light: diffuse and fog-scatter vertices do env NEE (neeEnv / neeEnvVolume) by
// importance-sampling the sky's luminance CDF, MIS-combined (balance heuristic) with
// the BSDF-sampled continuation that reaches the sky on a ray miss — the miss term
// is added at full weight only on a camera/specular arrival and MIS-weighted after a
// diffuse/volume bounce. All of this is skipped when the scene has no env light, so
// non-env scenes keep a bit-identical RNG stream / backward image.
#pragma once
#include "scene.h"
#include "camera.h"
#include "render.h"   // sampleGlossy, Renderer::refractOrReflect, clamp01, PI
#include "medium_stack.h" // nested-dielectric priority stack
#include "grin.h"     // shared gradient-index (GRIN) Eikonal marcher
#include "hero.h"     // hero-wavelength spectral sampling (kHeroC)

struct BackwardRenderer {
    int maxBounce = 32;
    // Direct-only (Whitted) preview (CLI -direct-only): after a non-specular
    // (diffuse / diffuse-transmit / fluorescent-elastic / fog-scatter) vertex does its
    // direct-lighting NEE, terminate instead of spawning the indirect continuation.
    // Specular chains (mirror / dielectric / glossy / filter) still recurse, so the
    // image keeps sharp reflections/refractions and direct+specular caustics but drops
    // diffuse GI — converging in ~1 spp. Left false = full unbiased path tracing.
    bool directOnly = false;
    // Deterministic Whitted preview (CLI -mode W). `directOnly` alone still leaves every
    // ESTIMATOR stochastic -- the area light is one random point per sample, a glossy
    // lobe one random direction, and specular survival a Russian-roulette coin -- so it
    // needs tens of spp to look clean and only buys ~3x. `whitted` additionally replaces
    // all three with their deterministic equivalents (fixed light grid, mirror direction,
    // reflectance-weighted continuation), which is what POV-Ray does and why it converges
    // in ONE sample per pixel instead of tens. Implies directOnly.
    bool whitted = false;
    int  lightGrid = 4;        // Whitted: NxN deterministic shadow rays per area light
    // Whitted: flat ambient radiance added at diffuse vertices (POV-Ray `ambient`), the
    // cheap stand-in for the GI this mode drops. ABSOLUTE spectral radiance -- the CLI's
    // dimensionless -ambient is multiplied by Scene::ambientRef() before it lands here.
    double ambient = 0.0;

    // ---- deterministic one-bounce gather ("radiosity" for mode W) -------------------
    // `ambient` alone cannot reproduce two things real bounce light does, and both are
    // measurable on the gold-gyroid scene:
    //   * CONTACT DARKENING. A constant lights a deep crevice exactly as much as an
    //     exposed face. Measured against a converged mode-R reference, the darkest 10%
    //     of the frame (i.e. the occluded interior of the lattice) carries 13.8 units of
    //     luminance error at the best flat ambient, and the error is SIGNED (-8.2) --
    //     still too dark -- so no single constant fixes it: raising it to close the
    //     crevices blows out the open faces by the same amount.
    //   * COLOUR BLEEDING. Real GI more than doubles the red on the side wall
    //     (48.7 -> 102.5) while barely moving the blue (4.4 -> 9.7), because the light
    //     reaching that wall has bounced off gold. A grey constant lifts all three
    //     channels together, so it buys exposure but not colour.
    // `giDirs > 0` replaces the flat term with a real single-bounce hemisphere gather: a
    // FIXED lattice of world-space directions (giDir) is traced from every diffuse
    // vertex, each carrying whatever deterministic Whitted radiance it finds -- which is
    // occlusion-aware and spectral, so both effects come back for real.
    //
    // Why a fixed lattice rather than POV-Ray's radiosity: POV-Ray caches irradiance at
    // an ADAPTIVELY chosen sparse point set and interpolates. Which points get sampled
    // depends on render order and on the local geometry, so on animated geometry the
    // cache's low-frequency blotches pop in and out between frames. POV-Ray's answer is
    // to pre-bake one cache and reuse it, which only works if nothing moves. This
    // gather has NO cache and no adaptivity: the direction set is a pure function of
    // (lattice index, sample index), never of the scene, so two frames of a rotating
    // object are lit by the identical estimator and a seamless loop cannot flicker.
    // The price is that the residual error appears as low-frequency BANDING instead of
    // noise -- but banding is a smooth function of the surface normal, so it slides
    // smoothly as geometry turns, where cache splotches jump. Raise `giDirs` (or -spp,
    // which rotates the lattice, see giDir) to push it down.
    int giDirs = 0;     // gather rays per diffuse vertex (0 = off, flat `ambient` only)
    int giGrid = 1;     // NxN shadow-ray grid at a GATHER vertex (cheaper than lightGrid)
    int giBounce = 4;   // max bounces along a gather ray. Bounds the cost of a specular
                        // chain: gold is ~0.9 reflective, so kWhittedCutoff alone would
                        // let a gather ray ricochet ~60 times inside a gold lattice.

    // Firefly clamp on ONE gather ray's returned radiance, per wavelength; 0 = off.
    // Absolute spectral radiance, pre-scaled by Scene::ambientRef() at the call site
    // exactly as `ambient` is (main.cpp), so the user-facing `-gi-clamp x` means "x times
    // one light's own radiance" in any scene.
    //
    // What it is for: a gather ray that reaches a lamp THROUGH a specular surface (glass
    // ball, mirror) carries that lamp's full radiance, while a gather ray that merely
    // bounced off a wall carries ~rho/pi times a small solid angle of it -- two orders of
    // magnitude less. That caustic is real and NEE cannot sample it (the lamp is behind a
    // refracting surface, so the emitter hit is its only estimator), but `giDirs` fixed
    // directions cannot resolve where it lands. Because every pixel shares those directions
    // (the invariant that makes this mode noise-free), the "does direction k reach the lamp
    // through the ball" boundary is a coherent CONTOUR in the image rather than the grain a
    // stochastic renderer would show -- i.e. thin, blown-out, dashed curves at 1 spp. This
    // caps them. See known-issues.md.
    //
    // Per-wavelength, not per-bundle: the scalar twin giGather() has one lambda and nothing
    // to take a max over, so a bundle-wide clamp would make the hero and scalar paths
    // disagree on the same scene -- and this file works hard to keep those two from
    // drifting apart. Costs a hue shift on a clamped ray, which is the point (it is being
    // pulled toward its neighbourhood).
    //
    // Keep it WELL ABOVE `ambient`. A gather ray that escapes the scene returns the flat
    // `ambient` far-field tail (see radianceHero), and the clamp is applied to that too, so
    // the gather's own fill level is effectively min(ambient, giClamp): a clamp below
    // `-ambient` darkens the whole scene uniformly instead of only capping fireflies. This
    // is exact, not approximate -- scraps/gi_collapse.ftsl pins it down: in a scene where
    // every gather ray escapes, `-gi 32 -gi-clamp c` is PIXEL-IDENTICAL to
    // `-gi 0 -ambient min(ambient, c)` on both the CPU and the GPU.
    double giClamp = 0.0;

    // Where a path sits relative to the gather. `depth == 0` is a camera path (it does
    // the gather); `depth == 1` is a gather ray (it does NOT recurse, uses `giGrid`, and
    // terminates its own diffuse vertices on the flat `ambient` tail). `sIdx` is the
    // absolute sample index, which rotates the lattice so that -spp progressively
    // refines the gather instead of re-rendering the same banding. `bounce` is the current
    // bounce index along the path, so a deterministic per-vertex decision (the mode-W glossy
    // lobe -- see whittedGlossyDir) can pick a different sequence at each vertex instead of
    // driving every glossy bounce off the same 1-D lattice.
    struct GiCtx {
        int depth = 0;
        unsigned long long sIdx = 0;
        int bounce = 0;
    };

    // One direction of the fixed gather lattice: point `j` of an `n`-point Fibonacci
    // spiral on the WHOLE sphere, Cranley-Patterson-rotated by (p1, p2).
    //
    // The lattice is built in WORLD space and the caller keeps the ~half of it with
    // cos > 0, weighting by cos and normalising by the realised sum. Two reasons that
    // beats a cosine-weighted lattice built in a local frame around the normal:
    //   * No tangent frame is needed, so there is no orthonormal-basis discontinuity to
    //     show up as a seam where the frame construction flips.
    //   * A direction entering or leaving the hemisphere does so at cos == 0, i.e. with
    //     zero weight, so the estimate is continuous in the normal -- which is what
    //     makes a rotating object's shading slide instead of pop.
    // Normalising by the realised sum of cosines makes the estimator EXACT for constant
    // incident radiance, so with nothing in the scene the gather reduces bit-sensibly to
    // the flat `ambient` term it replaces (no exposure step when -gi is switched on).
    static Vec3 giDir(int j, int n, double p1, double p2) {
        double t = ((double)j + 0.5) / (double)n + p2;
        t -= std::floor(t);                          // CP rotation of the z-strata
        const double z = 1.0 - 2.0 * t;
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double kGolden = 2.399963229728653;    // pi * (3 - sqrt 5)
        const double a = kGolden * (double)j + 2.0 * PI * p1;
        return Vec3(r * std::cos(a), r * std::sin(a), z);
    }

    bool diffraction = true;   // mirrors Renderer::diffraction for MatType::Grating
    int  heroC = hero::kHeroC;  // wavelengths bundled per camera path when hero is on
                                // (runtime -heroc N, clamped to [1, kHeroMax]; 1 = single-λ)
    // SPLIT-AT-DISPERSION (-herosplit; hero.h's alternative to de-hero'ing). At a
    // dispersive vertex, fan the bundle into C monochromatic sub-paths -- each refracting
    // along its OWN Snell direction -- instead of terminating the secondaries and boosting
    // the hero ×C. Both estimators are unbiased; this one resolves the chromatic spread
    // GEOMETRICALLY, at C× the traversal work past the split (linear, not exponential: a
    // sub-path is already monochromatic and so can never split again).
    //
    // It matters far more here than in the forward tracer, and specifically for mode W.
    // Mode W's λ lattice is SHARED by every pixel (that shared-offset property is exactly
    // what makes it noise-free), so when de-hero collapses the bundle to the hero the
    // ENTIRE FRAME collapses onto ONE wavelength and every dispersive object comes out
    // strongly mistinted -- an error no amount of spatial sampling fixes, because it is not
    // noise. Splitting keeps all C wavelengths live through the glass, so mode W is
    // colour-correct on a dielectric at -spp 1. Hence `whitted` turns this on by default
    // (main.cpp); mode R is stochastic and averages the collapse away, so it stays opt-in.
    bool heroSplit = hero::gSplit;

    // Per-sample cache of emitter-SPD evaluations at the path's wavelengths. The
    // wavelengths are fixed for the whole camera path, but every NEE connection
    // used to re-evaluate em.spdFn(λ) — a std::function typically wrapping the
    // full Planck formula (exp + pow per call) — per emitter per bounce, and
    // Scene::invPdfLambda re-evaluated ALL emitters' SPDs per wavelength again at
    // sample setup. Profiling mode R showed the spdFn dispatch + blackbody math
    // at ~40% of CPU time. renderRows now evaluates spdFn once per (emitter,
    // wavelength) per sample; NEE and the invPdf setup read the table. `lam`
    // points at the pristine per-sample wavelength array; a fluorescent bounce
    // can REWRITE the path's local wavelength (Stokes shift), so readers must
    // check matches() and fall back to a live spdFn evaluation when the path's
    // wavelengths no longer equal the cached ones. Cached values are the exact
    // doubles spdFn returned and the consumers keep the same iteration order and
    // arithmetic shape, so images stay bit-identical.
    struct SpdCache {
        const double* lam = nullptr;  // wavelengths the table was built for
        const double* spd = nullptr;  // emitter-major: spd[e*C + i] = emitters[e].spdFn(lam[i])
        int C = 0;
        bool matches(const double* l, int nUp) const {
            for (int i = 0; i < nUp; ++i) if (l[i] != lam[i]) return false;
            return true;
        }
    };

    // Next-event estimation: connect a surface vertex to each area emitter (the
    // integral splits by light, summed unbiasedly). `invPdfLambda` = emitG/g(lambda)
    // is the reciprocal of the sampled-wavelength pdf; multiplied by an emitter's
    // SPD(lambda) it yields that emitter's Le/pdf weight (= its SPD integral for a
    // single light, matching the forward tracer's photon-weight convention).
    //
    // Shared per-emitter geometry sample for a SURFACE next-event connection. Takes the
    // emitter's two sample coordinates EXPLICITLY (`u1`,`u2`) rather than drawing them,
    // so the scalar and hero NEE can share one visibility sample -- and so the
    // deterministic Whitted preview (`whitted`) can drive the same body from a fixed
    // stratified grid instead of the rng. Callers draw 2 uniforms per emitter to keep
    // the Monte Carlo modes' rng stream (and therefore their images) bit-identical;
    // a collimated beam or point-spot ignores both coordinates.
    // On success returns the λ-INDEPENDENT geometry weight `w` and
    // the connection distance `dist`; the diffuse contribution at wavelength λ is then
    //   (rho(λ)/PI) * SPD(λ)*invPdfλ * w      (× medium transmittance, added by caller).
    // Returns false to skip this emitter (back-facing, shadowed, or behind geometry).
    bool emitterGeom(const Scene& scene, const Hit& h, const Vec3& ngo,
                     const Emitter& em, double u1, double u2, double& dist, double& w) const {
        if (em.collimated) return false;                  // beams aren't area-samplable
        if (em.shape == EmitterShape::Spot) {
            // Point spot: deterministic connect to the light point, weighted by the
            // cone falloff toward the surface (peak intensity/SPD = 1). No rng draw.
            Vec3 toL = em.origin - h.p;
            double dist2 = dot(toL, toL);
            dist = std::sqrt(dist2);
            Vec3 wi = toL / dist;
            double cosSurf = dot(h.n, wi);
            if (cosSurf <= 0) return false;
            double stG = shadowTerminatorG(wi, h.n, ngo);   // Chiang soft terminator (1 if flat)
            if (stG <= 0.0) return false;                    // behind true geometry: hard shadow
            double fall = spotFalloff(dot(-wi, em.beamDir), em.spotCosInner, em.spotCosOuter);
            if (fall <= 0) return false;
            if (scene.occluded(h.p + ngo * 1e-6, wi, dist - 2e-6)) return false;
            w = fall * cosSurf / dist2 * stG;                // I(w)/dist^2 (× BRDF & SPD by caller)
            return true;
        }
        if (em.shape == EmitterShape::Sun) {
            // Distant directional sun: sample wi uniformly in the solar cone about
            // -beamDir (solid-angle pdf 1/Omega) and shadow-ray it to the scene exit —
            // there is no finite light distance, so no 1/dist^2 and no cosLight. In
            // solid-angle measure w = cos(surf)/pdfW = cos(surf)*Omega, and since spdFn
            // is the sun's RADIANCE the caller's (rho/PI)*SPD*w reproduces the textbook
            // (rho/PI)*E_perp*cos(surf) for the irradiance the light was authored with.
            // Two rng draws, matching the area-light path below, so adding a sun does
            // not reshuffle any other emitter's stream.
            Vec3 wi = em.sampleCone(-em.beamDir, u1, u2);
            double cosSurf = dot(h.n, wi);
            if (cosSurf <= 0) return false;
            double stG = shadowTerminatorG(wi, h.n, ngo);   // Chiang soft terminator (1 if flat)
            if (stG <= 0.0) return false;                    // behind true geometry: hard shadow
            dist = length(scene.sceneCenter - h.p) + scene.sceneRadius;   // to the scene exit
            if (scene.occluded(h.p + ngo * 1e-6, wi, dist)) return false;
            w = cosSurf * em.spotOmega * stG;
            return true;
        }
        Vec3 y, nLight, wi;
        double pdfW = 0.0;
        // Sphere: cone/solid-angle importance sampling of only the visible cap toward
        // `h.p` (low variance, no wasted back-facing draws). The estimator is in
        // solid-angle measure, so the cosLight/dist^2/area area-measure Jacobian is
        // replaced by 1/pdfW. Quads (and a receiver inside a sphere) keep the uniform
        // area-measure estimator.
        bool coneSampled = (em.shape == EmitterShape::Sphere) &&
                           em.sampleSphereCone(h.p, u1, u2, y, nLight, wi, dist, pdfW);
        // Cylinder: importance-sample only the front-facing lateral arc toward `h.p`
        // (area measure). Every draw is front-facing, so effArea = 1/pdfArea (the
        // visible area) replaces em.area and no samples land on the back side.
        double effArea = em.area, pdfAreaCyl = 0.0;
        bool cylVisible = !coneSampled && em.shape == EmitterShape::Cylinder &&
                          !em.caps &&   // capped tubes: uniform samplePoint covers the caps too
                          em.sampleCylinderVisible(h.p, u1, u2, y, nLight, pdfAreaCyl);
        if (cylVisible) effArea = 1.0 / pdfAreaCyl;
        if (coneSampled) {
            double cosSurf = dot(h.n, wi);
            if (cosSurf <= 0) return false;
            double stG = shadowTerminatorG(wi, h.n, ngo);   // Chiang soft terminator (1 if flat)
            if (stG <= 0.0) return false;                    // behind true geometry: hard shadow
            if (scene.occluded(h.p + ngo * 1e-6, wi, dist - 2e-6)) return false;
            w = cosSurf / pdfW * stG;                        // solid-angle measure
            return true;
        }
        // quad / mesh / interior-sphere / cylinder fallback. emitterSamplePoint also
        // returns this point's `emit pattern:` multiplier (1.0 when there is none, so
        // every unpatterned scene stays bit-identical); folding it into the λ-independent
        // geometry weight `w` makes both the scalar and hero NEE pick it up at once, and
        // matches the emitSlot() factor the emission-on-hit side applies at the same
        // surface point — which is what keeps the MIS pair consistent.
        double epat = 1.0;
        if (!cylVisible) epat = emitterSamplePoint(scene, em, u1, u2, y, nLight);
        Vec3 toL = y - h.p;
        double dist2 = dot(toL, toL);
        dist = std::sqrt(dist2);
        wi = toL / dist;
        double cosSurf = dot(h.n, wi);
        if (cosSurf <= 0) return false;
        double stG = shadowTerminatorG(wi, h.n, ngo);   // Chiang soft terminator (1 if flat)
        if (stG <= 0.0) return false;                    // behind true geometry: hard shadow
        double cosLight = dot(nLight, -wi);              // light is one-sided
        if (cosLight <= 0) return false;
        if (scene.occluded(h.p + ngo * 1e-6, wi, dist - 2e-6)) return false;
        double G = cosSurf * cosLight / dist2;           // geometry term
        w = G * effArea * stG;                           // pdf_area = 1/effArea (visible area for cylinder)
        if (epat != 1.0) w *= epat;                      // no-op (and bit-identical) without a pattern
        return true;
    }

    // Radical inverses (van der Corput), for the Whitted preview's deterministic sample
    // placement. Base 2 gets the bit-reversal fast path; the odd bases use the generic
    // digit loop (called once per pixel-sample, i.e. nothing beside a BVH walk).
    static double radicalInverse2(uint64_t i) {
        i = (i << 32) | (i >> 32);
        i = ((i & 0x0000ffff0000ffffULL) << 16) | ((i & 0xffff0000ffff0000ULL) >> 16);
        i = ((i & 0x00ff00ff00ff00ffULL) <<  8) | ((i & 0xff00ff00ff00ff00ULL) >>  8);
        i = ((i & 0x0f0f0f0f0f0f0f0fULL) <<  4) | ((i & 0xf0f0f0f0f0f0f0f0ULL) >>  4);
        i = ((i & 0x3333333333333333ULL) <<  2) | ((i & 0xccccccccccccccccULL) >>  2);
        i = ((i & 0x5555555555555555ULL) <<  1) | ((i & 0xaaaaaaaaaaaaaaaaULL) >>  1);
        return (double)i * (1.0 / 18446744073709551616.0);
    }

    // DIGIT-SCRAMBLED radical inverse (Faure's fix for high-dimensional Halton), needed by
    // every lattice below that uses a LARGE base. A plain radical inverse in base b returns
    // exactly i/b for i < b, so its first N points cover only the prefix [0, N/b) of the unit
    // interval — well distributed *within* that prefix and blind to the rest. With b = 61 that
    // means a 16-spp preview never leaves the first quarter of the sequence's range. That is
    // not academic: it made a `fluorescent` dye whose absorption edge sits at 480 nm contribute
    // EXACTLY NOTHING until `-spp 64` (its λ_in coordinate was pinned to [0.5, 0.75], i.e. to
    // wavelengths well past the edge), and it skewed every `glossy` lobe towards its mirror
    // direction until `-spp 13`.
    //
    // The fix is to permute the DIGITS: r = Σ π(dₖ) b^-(k+1). Any bijection π of {0..b-1}
    // leaves the sequence a permutation of the same b-point grid, so it is exactly as
    // low-discrepancy asymptotically, but the *order* the grid is visited in is scattered
    // instead of monotone — N points now spread over the whole interval for any N.
    //
    // π here is multiplicative, π(d) = (d·m) mod b with m ≈ b/φ (golden ratio). That needs no
    // permutation tables (so the device twin is trivially bit-identical), is a bijection for
    // every prime b (gcd(m, b) = 1 since 0 < m < b), spreads consecutive digits about as
    // evenly as a 1-D sequence can, and — the load-bearing property — satisfies **π(0) = 0**,
    // so `radicalInverseScr(b, 0) == 0` exactly, in every base. Every "sample 0 is the
    // canonical outcome" contract below (mirror direction, specular grating order, median λ)
    // therefore survives untouched, and only spp > 1 changes.
    static unsigned goldenDigitMul(unsigned base) {
        // round(base / φ); φ⁻¹ = 0.6180339887498949. Clamped to a valid multiplier for the
        // degenerate small bases (b = 2 gives 1, i.e. the identity — base 2 needs no scramble).
        unsigned m = (unsigned)((double)base * 0.6180339887498949 + 0.5);
        return (m == 0u || m >= base) ? 1u : m;
    }
    static double radicalInverseScr(unsigned base, uint64_t i) {
        const unsigned mul = goldenDigitMul(base);
        const double invB = 1.0 / (double)base;
        double f = invB, r = 0.0;
        while (i) {
            r += (double)((unsigned)(i % base) * mul % base) * f;
            i /= base; f *= invB;
        }
        return r;
    }
    // Cranley-Patterson rotation by 1/2, so that sample 0 of every sequence lands dead
    // centre (0.5) rather than at 0 — a 1-spp preview is then the classic un-antialiased
    // pixel-centre ray, and the wavelength lattice sits mid-stratum.
    static double rot05(double x) { x += 0.5; return (x >= 1.0) ? x - 1.0 : x; }

    // Deterministic sample placement for the Whitted preview. EVERY PIXEL USES THE SAME
    // offsets -- that is precisely what makes the mode noise-free, since neighbouring
    // pixels then differ only by their geometry and never by their luck. The sequence is
    // indexed by the ABSOLUTE sample index and is *progressive* (any prefix of a radical
    // inverse is well distributed), so — exactly like the rng stream — the pattern does
    // not depend on how the budget was split into chunks. That matters in practice: a
    // `-window` render chunks into 1-spp batches, and a per-chunk `(s + .5)/spp` lattice
    // would collapse to "sample 0" forever and make `-spp` a no-op on the image.
    //
    // In this mode `spp` therefore stops meaning "more Monte Carlo samples" and starts
    // meaning "finer edge antialiasing and a denser fixed wavelength set" -- the image
    // gets sharper, never less grainy, because it was never grainy.
    static void whittedSample(uint64_t idx, double& u, double& v) {
        u = rot05(radicalInverse2(idx));
        v = rot05(radicalInverseScr(3, idx));
    }
    // The bundle's / scalar path's base wavelength coordinate; a third, decorrelated
    // radical inverse so λ placement does not lock to the subpixel position.
    static double whittedLambdaU(uint64_t idx) { return rot05(radicalInverseScr(5, idx)); }

    // The two Cranley-Patterson phases of the one-bounce-gather lattice (see giGatherHero),
    // from the ABSOLUTE sample index. Bases 7 and 11 collide with neither the subpixel
    // lattice (2, 3), the wavelength lattice (5), nor the glossy/discrete lattices (>= 13).
    // Every pixel shares them -- the invariant that makes this mode noise-free -- so raising
    // -spp rotates the whole frame's lattice coherently and the banding averages out.
    // (Device twin: dGiPhases. Must stay bit-identical; probed by `-checklattice`.)
    static void giPhases(uint64_t sIdx, double& p1, double& p2) {
        p1 = rot05(radicalInverseScr(7, sIdx));
        p2 = rot05(radicalInverseScr(11, sIdx));
    }

    // Deterministic ROUGH-SPECULAR direction for the Whitted preview: point `sIdx` of a
    // fixed 2-D lattice on the power-cosine lobe around `mdir`, instead of the lobe's single
    // mirror direction.
    //
    // Why this is not just a polish item: collapsing every sample onto the mirror direction
    // makes mode W INCONSISTENT on rough specular -- raising -spp changed nothing at all,
    // because each extra sample re-traced the identical direction, so satin metal previewed
    // crisper than it renders and stayed that way at any budget. Driving the lobe from the
    // sample index makes -spp converge on the true lobe while keeping BOTH mode-W
    // invariants: every pixel uses the same offsets (so it is noise-free, not grainy -- at a
    // given spp the whole frame shares one lobe direction per vertex), and the sequence is
    // indexed by the ABSOLUTE sample index (so the image is chunk-split-independent).
    //
    // The polar coordinate is COMPLEMENTED rather than rot05'd, because glossyDirUV maps
    // u1 == 1 to the mirror direction and radicalInverse(0) == 0 in every base: sample 0 is
    // therefore *exactly* the old mirror direction, so a 1-spp preview is bit-identical to
    // 0.107.0 and only spp > 1 changes. (sinT == 0 there makes the azimuth moot too.)
    //
    // Each bounce depth takes its own prime pair so two glossy vertices on one path are not
    // driven by the same 1-D sequence -- which would correlate their offsets and fold a
    // double-bounce lobe into a line. Bases 2/3 are the subpixel lattice, 5 the wavelength,
    // 7/11 the gather, so these start at 13. Those bases are all LARGER than a typical preview's
    // `-spp`, which is exactly why they go through the digit-SCRAMBLED radical inverse: unscrambled,
    // base 13 pinned u1 to [1 - spp/13, 1] and kept the lobe hugging its mirror direction until
    // `-spp 13`.
    //
    // The two lattice COORDINATES are factored out of the direction so `-checklattice`
    // (N4a) can probe exactly the numbers the renderer uses. They are pure integer-and-
    // `double` arithmetic and so are bit-comparable host-vs-device; the direction itself
    // goes through `glossyDirUV`'s trig in `Real` and is not.
    static void whittedGlossyUV(uint64_t sIdx, int bounce, double& u1, double& u2) {
        static const unsigned kBases[4][2] = {{13, 17}, {19, 23}, {29, 31}, {37, 41}};
        const unsigned* pb = kBases[bounce & 3];
        u1 = 1.0 - radicalInverseScr(pb[0], sIdx);   // 1 at sIdx 0 => mirror
        u2 = radicalInverseScr(pb[1], sIdx);
    }
    static Vec3 whittedGlossyDir(const Vec3& mdir, double roughness, uint64_t sIdx, int bounce) {
        double u1, u2;
        whittedGlossyUV(sIdx, bounce, u1, u2);
        return glossyDirUV(mdir, roughness, u1, u2);
    }

    // Deterministic DISCRETE-CHOICE coordinate for the Whitted preview. Same job as
    // whittedGlossyDir, but for a pick out of a finite weighted SET rather than a direction on
    // a lobe: one scalar in [0,1) off the (sIdx, bounce) lattice, which the caller inverts
    // against its own CDF. Keeping ONE choice per sample (rather than summing the set) is what
    // preserves the estimator -- the existing pick is analog (order i with probability
    // wgt[i]/wsum, throughput unchanged), so feeding it a stratified u instead of an rng draw
    // is still unbiased and merely removes the per-pixel luck.
    //
    // NOT rot05'd, deliberately: `gratingDiffract` walks its candidates in DESCENDING
    // efficiency on this path, so u == 0 -- which radicalInverseB returns at sIdx 0 in every
    // base -- selects the specular order m = 0. That is the exact analogue of
    // whittedGlossyDir's "sample 0 is the mirror direction": a 1-spp preview shows the
    // undiffracted image, and extra spp fan the spectrum out into the higher orders.
    // The scramble is what makes that fan-out *gradual*: unscrambled, base 43 confined u to
    // [0, spp/43), so the higher orders arrived in a lump only once `-spp` passed the base.
    static double whittedOrderU(uint64_t sIdx, int bounce) {
        static const unsigned kBases[4] = {43, 47, 53, 59};
        return radicalInverseScr(kBases[bounce & 3], sIdx);
    }
    // Deterministic Stokes-shift EXCITATION wavelength coordinate for the Whitted preview.
    // Unlike a grating order there is no "specular" outcome worth preferring here, so this one
    // IS rot05'd like the other wavelength lattices: sample 0 lands at the MEDIAN of the
    // excitation CDF -- the most representative single λ_in -- instead of at its short-λ
    // extreme, which is what an unrotated u == 0 would pick.
    //
    // This dimension is where the unscrambled radical inverse hurt most, and it is why
    // radicalInverseScr exists: with base 61 and rot05, u was confined to [0.5, 0.5 + spp/61)
    // for any preview budget below 61 spp, i.e. to the LONG half of the illuminant's CDF. A dye
    // absorbing only below 480 nm was therefore never excited at all and rendered as its bare
    // elastic lobe until `-spp 64`, at which point the sequence finally wrapped and the dye
    // switched on in one step. Scrambled, the same 4 samples straddle the whole CDF.
    // Since v0.115.0 the CDF this indexes is the DYE'S OWN excitation distribution
    // (Material::fluoInSampler = absorb(λ) × illuminant), not the scene-wide illuminant, so
    // sample 0's median draw now lands inside the absorption band by construction and even a
    // narrow-band dye fluoresces correctly at `-spp 1`.
    static double whittedFluoroU(uint64_t sIdx, int bounce) {
        static const unsigned kBases[4] = {61, 67, 71, 73};
        return rot05(radicalInverseScr(kBases[bounce & 3], sIdx));
    }

    // Whitted: replace a Russian-roulette survival test with a throughput WEIGHT.
    // Same expected value, zero variance. Returns false once the path is too dim to
    // matter -- POV-Ray's `adc_bailout` (default 1/255), which is what stops a
    // mirror-lined box from recursing to maxBounce on literally every pixel.
    static constexpr double kWhittedCutoff = 1.0 / 512.0;
    static bool whittedAttenuate(double& thr, double w) {
        if (w <= 0.0) return false;
        thr *= w;
        return thr > kWhittedCutoff;
    }

    // Does this emitter consume its two sample coordinates?  A collimated beam and a
    // point-spot are deterministic connections and draw nothing; every area shape (and
    // the sun's cone) draws two. The Monte Carlo callers must match this exactly, or a
    // scene that merely *adds* a spot light would reshuffle every other emitter's rng
    // stream and change an unrelated image.
    static bool emitterNeedsUV(const Emitter& em) {
        return !em.collimated && em.shape != EmitterShape::Spot;
    }

    // Deterministic stratified sample coordinates for the Whitted preview: the centre
    // of cell (gx,gy) of a G x G grid over the emitter's [0,1)^2 sample domain. This is
    // POV-Ray's `area_light` model -- a fixed lattice of shadow rays whose average is a
    // soft shadow with NO variance, rather than one random point per camera sample whose
    // average only becomes smooth after many spp.
    static void gridUV(int g, int G, double& u1, double& u2) {
        u1 = ((double)(g % G) + 0.5) / (double)G;
        u2 = ((double)(g / G) + 0.5) / (double)G;
    }

    double neeLight(const Scene& scene, const Hit& h, double rho, double invPdfLambda,
                    double lambda, Pcg32& rng, const SpdCache* spdCache = nullptr,
                    GiCtx gi = GiCtx{}) const {
        double total = 0.0;
        // Geometric normal on the shading-normal side: every light connection must lie
        // in this hemisphere too, else a smoothed shading normal would leak light in
        // through the geometric back face (shading-normal problem). No-op for flat
        // tris / analytic spheres (ngo == h.n there). The shadow ray is also offset
        // along ngo so it clears the true surface rather than the shading normal.
        const Vec3 ngo = orientedGeoN(h);
        const bool med = scene.backwardMedium().enabled;
        // The cache is keyed on wavelength slot 0, so a scalar caller inside a hero
        // path (post-de-hero interactMaterial) reuses the hero table's i==0 column;
        // a fluorescent λ-switch fails matches() and falls back to a live spdFn call.
        const bool cached = spdCache && spdCache->matches(&lambda, 1);
        const int nEm = (int)scene.emitters.size();
        for (int e = 0; e < nEm; ++e) {
            const Emitter& em = scene.emitters[e];
            const bool uv = emitterNeedsUV(em);
            // Whitted: G x G deterministic shadow rays per area light, averaged. A
            // deterministic emitter (spot/beam) has nothing to stratify, so it stays at 1.
            // A GATHER vertex uses the coarser giGrid: its soft-shadow detail is about to
            // be averaged over giDirs directions anyway, so paying lightGrid^2 there
            // multiplies the gather's cost for no visible return.
            const int G = (whitted && uv) ? (gi.depth ? giGrid : lightGrid) : 1;
            const int nS = G * G;
            double acc = 0.0, spdV = 0.0;
            bool haveSpd = false;
            for (int s = 0; s < nS; ++s) {
                double u1 = 0.0, u2 = 0.0;
                if (whitted) { if (uv) gridUV(s, G, u1, u2); }
                else if (uv) { u1 = rng.uniform(); u2 = rng.uniform(); }
                double dist = 0.0, w = 0.0;
                if (!emitterGeom(scene, h, ngo, em, u1, u2, dist, w)) continue;
                if (!haveSpd) {   // evaluated at most once per emitter, as before
                    spdV = cached ? spdCache->spd[(size_t)e * (size_t)spdCache->C]
                                  : em.spdFn(lambda);
                    haveSpd = true;
                }
                double contrib = (rho / PI) * (spdV * invPdfLambda) * w;
                if (med)                                          // Beer-Lambert on the shadow ray
                    contrib *= std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
                acc += contrib;
            }
            total += (nS > 1) ? acc / (double)nS : acc;
        }
        return total;
    }

    // Hero-wavelength surface NEE: one shared visibility sample per emitter (identical
    // rng to neeLight), evaluated for all `nUp` active wavelengths. Accumulates
    // thr[i]·(rho[i]/PI)·SPD(λ_i)·invPdf[i]·w into L[i]. Only called on the fog-free
    // hero fast path, so there is no medium transmittance term.
    void neeLightHero(const Scene& scene, const Hit& h, const double* rho, double* L,
                      const double* thr, const double* lam, const double* invPdf,
                      int nUp, Pcg32& rng, const SpdCache* spdCache,
                      GiCtx gi = GiCtx{}) const {
        const Vec3 ngo = orientedGeoN(h);
        const bool cached = spdCache && spdCache->matches(lam, nUp);
        const int nEm = (int)scene.emitters.size();
        for (int e = 0; e < nEm; ++e) {
            const Emitter& em = scene.emitters[e];
            const bool uv = emitterNeedsUV(em);
            const int G = (whitted && uv) ? (gi.depth ? giGrid : lightGrid) : 1;
            const int nS = G * G;
            const double invS = 1.0 / (double)nS;
            for (int s = 0; s < nS; ++s) {
                double u1 = 0.0, u2 = 0.0;
                if (whitted) { if (uv) gridUV(s, G, u1, u2); }
                else if (uv) { u1 = rng.uniform(); u2 = rng.uniform(); }
                double dist = 0.0, w = 0.0;
                if (!emitterGeom(scene, h, ngo, em, u1, u2, dist, w)) continue;
                const double ws = (nS > 1) ? w * invS : w;
                if (cached) {
                    const double* spdE = spdCache->spd + (size_t)e * (size_t)spdCache->C;
                    for (int i = 0; i < nUp; ++i)
                        L[i] += thr[i] * (rho[i] / PI) * (spdE[i] * invPdf[i]) * ws;
                } else {
                    for (int i = 0; i < nUp; ++i)
                        L[i] += thr[i] * (rho[i] / PI) * (em.spdFn(lam[i]) * invPdf[i]) * ws;
                }
            }
        }
    }

    // ---- deterministic one-bounce gather ---------------------------------------------
    // Estimate the cosine-weighted mean INCIDENT radiance over the hemisphere above a
    // diffuse vertex by tracing the fixed lattice (see giDirs / giDir), then add the
    // Lambertian response rho * that. This is the term flat `ambient` was standing in
    // for, computed instead of assumed.
    //
    // Each gather ray runs the same deterministic Whitted radiance the camera ray does,
    // just with GiCtx::depth == 1, which (a) stops it gathering again -- single bounce --
    // (b) drops it to `giGrid` shadow rays, (c) makes it terminate its own diffuse
    // vertices on the flat `ambient` tail, and (d) starts it NON-specular so that a ray
    // landing straight on a light adds nothing (the vertex's own NEE already counted
    // that; adding it here would double the direct light). A ray that reaches a light
    // *via* a mirror still counts, because a specular bounce re-arms specularArrival --
    // so gold-bounced light, the whole point of this, is carried at full weight.
    //
    void giGatherHero(const Scene& scene, const Hit& h, const double* rho, double* L,
                      const double* thr, const double* lam, const double* invPdf,
                      int nUp, Pcg32& rng, const SpdCache* spdCache, GiCtx gi) const {
        const Vec3 ngo = orientedGeoN(h);
        const int n = giDirs * 2;      // full-sphere lattice; ~half of it faces outward
        // Cranley-Patterson phases from the ABSOLUTE sample index, on two decorrelated
        // radical inverses (bases 7 and 11, so they collide with neither the subpixel
        // lattice (2,3) nor the wavelength lattice (5)). Every pixel shares them -- the
        // invariant that makes this mode noise-free -- so raising -spp rotates the whole
        // frame's lattice coherently and the banding averages out progressively.
        double p1, p2;
        giPhases(gi.sIdx, p1, p2);
        double acc[hero::kHeroMax];
        for (int i = 0; i < nUp; ++i) acc[i] = 0.0;
        double wSum = 0.0;
        const GiCtx sub{gi.depth + 1, gi.sIdx};
        for (int j = 0; j < n; ++j) {
            const Vec3 d = giDir(j, n, p1, p2);
            const double c = dot(h.n, d);
            if (c <= 0.0) continue;
            // Also require the GEOMETRIC hemisphere, or a smoothed shading normal would
            // gather through the true back face (the shading-normal problem again).
            if (dot(ngo, d) <= 0.0) continue;
            wSum += c;
            double Lg[hero::kHeroMax];
            radianceHero(scene, Ray{h.p + ngo * 1e-6, d}, lam, invPdf, nUp, Lg, rng,
                         spdCache, sub);
            // Firefly clamp (see giClamp). NOT applied to wSum: the weight of a clamped
            // direction stays c, so the estimator still normalises by the realised sum of
            // cosines and an unclamped gather is untouched bit-for-bit.
            if (giClamp > 0.0)
                for (int i = 0; i < nUp; ++i) if (Lg[i] > giClamp) Lg[i] = giClamp;
            for (int i = 0; i < nUp; ++i) acc[i] += c * Lg[i];
        }
        if (wSum <= 0.0) return;
        const double inv = 1.0 / wSum;
        for (int i = 0; i < nUp; ++i) L[i] += thr[i] * rho[i] * (acc[i] * inv);
    }

    // Scalar twin of giGatherHero, for the paths that cannot use the hero bundle (fog,
    // GRIN, a physical lens) and for a de-hero'd vertex inside interactMaterial.
    double giGather(const Scene& scene, const Hit& h, double rho, double lambda,
                    double invPdfLambda, Pcg32& rng, const SpdCache* spdCache,
                    GiCtx gi) const {
        const Vec3 ngo = orientedGeoN(h);
        const int n = giDirs * 2;
        double p1, p2;
        giPhases(gi.sIdx, p1, p2);
        double acc = 0.0, wSum = 0.0;
        const GiCtx sub{gi.depth + 1, gi.sIdx};
        for (int j = 0; j < n; ++j) {
            const Vec3 d = giDir(j, n, p1, p2);
            const double c = dot(h.n, d);
            if (c <= 0.0) continue;
            if (dot(ngo, d) <= 0.0) continue;
            wSum += c;
            double Lg = radiance(scene, Ray{h.p + ngo * 1e-6, d}, lambda, invPdfLambda,
                                 rng, spdCache, sub);
            if (giClamp > 0.0 && Lg > giClamp) Lg = giClamp;   // see giClamp
            acc += c * Lg;
        }
        return (wSum > 0.0) ? rho * (acc / wSum) : 0.0;
    }

    // The Whitted indirect-diffuse term at a vertex, in one place so the scalar and hero
    // diffuse cases cannot drift apart: the real gather on a camera path when -gi is on,
    // otherwise the flat `ambient` constant. A gather ray (depth > 0) always takes the
    // flat branch -- that constant is what terminates the single bounce.
    bool giUseGather(GiCtx gi) const { return giDirs > 0 && gi.depth == 0; }

    // Volume next-event estimation: connect a fog scattering vertex `p` (photon
    // arriving along `wIn`) to a uniformly-sampled light point. The surface BRDF
    // and cosine are replaced by the single-scattering albedo and the Henyey-
    // Greenstein phase function; the shadow ray carries fog transmittance. This is
    // the backward mirror of the forward tracer's connectVolume().
    double neeVolume(const Scene& scene, const Vec3& p, const Vec3& wIn,
                     double lambda, double invPdfLambda, Pcg32& rng,
                     const SpdCache* spdCache = nullptr) const {
        double total = 0.0;
        const bool cached = spdCache && spdCache->matches(&lambda, 1);
        const int nEmV = (int)scene.emitters.size();
        for (int e = 0; e < nEmV; ++e) {
            const Emitter& em = scene.emitters[e];
            const double spdV = cached ? spdCache->spd[(size_t)e * (size_t)spdCache->C]
                                       : 0.0;   // live spdFn below when not cached
            if (em.collimated) continue;
            if (em.shape == EmitterShape::Spot) {
                // Point spot at a volume vertex: no surface cosine, cone falloff only.
                Vec3 toL = em.origin - p;
                double dist2 = dot(toL, toL);
                double dist = std::sqrt(dist2);
                Vec3 wi = toL / dist;
                double fall = spotFalloff(dot(-wi, em.beamDir), em.spotCosInner, em.spotCosOuter);
                if (fall <= 0) continue;
                if (scene.occluded(p + wi * 1e-6, wi, dist - 2e-6)) continue;
                double phase  = scene.backwardMedium().phaseValue(dot(wIn, wi), lambda);
                double albedo = scene.backwardMedium().albedo(lambda);
                double T = std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
                double emitW = (cached ? spdV : em.spdFn(lambda)) * invPdfLambda;
                total += albedo * phase * emitW * fall / dist2 * T;
                continue;
            }
            if (em.shape == EmitterShape::Sun) {
                // Distant sun at a volume vertex: cone-sampled direction (pdf 1/Omega),
                // no surface cosine, transmittance to the scene exit. 1/pdfW = Omega.
                double s1 = rng.uniform(), s2 = rng.uniform();
                Vec3 wi = em.sampleCone(-em.beamDir, s1, s2);
                double dist = length(scene.sceneCenter - p) + scene.sceneRadius;
                if (scene.occluded(p + wi * 1e-6, wi, dist)) continue;
                double phase  = scene.backwardMedium().phaseValue(dot(wIn, wi), lambda);
                double albedo = scene.backwardMedium().albedo(lambda);
                double T = std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
                double emitW = (cached ? spdV : em.spdFn(lambda)) * invPdfLambda;
                total += albedo * phase * emitW * em.spotOmega * T;
                continue;
            }
            double u1 = rng.uniform(), u2 = rng.uniform();
            Vec3 y, nLight, wi;
            double dist = 0.0, pdfW = 0.0;
            bool coneSampled = (em.shape == EmitterShape::Sphere) &&
                               em.sampleSphereCone(p, u1, u2, y, nLight, wi, dist, pdfW);
            // Cylinder: front-facing lateral-arc sampling (area measure) toward `p`.
            double effArea = em.area, pdfAreaCyl = 0.0;
            bool cylVisible = !coneSampled && em.shape == EmitterShape::Cylinder &&
                              !em.caps &&   // capped tubes: uniform samplePoint covers the caps too
                              em.sampleCylinderVisible(p, u1, u2, y, nLight, pdfAreaCyl);
            if (cylVisible) effArea = 1.0 / pdfAreaCyl;
            double albedo = scene.backwardMedium().albedo(lambda);
            double emitW = (cached ? spdV : em.spdFn(lambda)) * invPdfLambda;
            double contrib;
            if (coneSampled) {
                if (scene.occluded(p + wi * 1e-6, wi, dist - 2e-6)) continue;
                double phase = scene.backwardMedium().phaseValue(dot(wIn, wi), lambda);
                contrib = albedo * phase * emitW / pdfW;   // solid-angle measure
            } else {
                // quad / mesh / interior-sphere / cylinder fallback; also returns the
                // sampled point's emission-pattern factor (1.0 when unpatterned).
                double epat = 1.0;
                if (!cylVisible) epat = emitterSamplePoint(scene, em, u1, u2, y, nLight);
                Vec3 toL = y - p;
                double dist2 = dot(toL, toL);
                dist = std::sqrt(dist2);
                wi = toL / dist;
                double cosLight = dot(nLight, -wi);        // light is one-sided
                if (cosLight <= 0) continue;
                if (scene.occluded(p + wi * 1e-6, wi, dist - 2e-6)) continue;
                double phase = scene.backwardMedium().phaseValue(dot(wIn, wi), lambda);
                double G = cosLight / dist2;               // no surface cosine at a volume vertex
                contrib = albedo * phase * emitW * G * effArea;
                if (epat != 1.0) contrib *= epat;
            }
            contrib *= std::exp(-scene.backwardMedium().sigmaT(lambda) * dist);
            total += contrib;
        }
        return total;
    }

    // Environment next-event estimation at a diffuse surface vertex. Importance-
    // samples a direction from the env map's luminance CDF (a uniform sphere
    // direction for a constant env), connects with a shadow ray out past the scene
    // bounds, and MIS-weights (balance heuristic) against the BSDF-sampled
    // continuation that also reaches the env on a ray miss — so bright, concentrated
    // skies (a sun disk) are low-variance without being double-counted. Only invoked
    // when the scene actually has an env light (guarded by the caller so non-env
    // scenes keep a bit-identical RNG stream). Mirrors neeLight's transmittance /
    // shadow-bias conventions.
    // Shared geometry for environment NEE at a surface vertex: one env-direction
    // importance sample (draws rng once, identical to the old neeEnv), the shading /
    // shadow-terminator gate, the shadow ray, and the balance-heuristic MIS weight
    // against the cosine-sampled continuation. All λ-independent. Returns false to
    // skip; on success fills `wi`, `cosSurf`, `stG`, `pdfW`, `wMis`, `farDist`.
    bool envGeom(const Scene& scene, const Hit& h, Pcg32& rng, Vec3& wi,
                 double& cosSurf, double& stG, double& pdfW, double& wMis,
                 double& farDist) const {
        wi = scene.sampleEnvDir(rng, pdfW);
        if (pdfW <= 0.0) return false;
        cosSurf = dot(h.n, wi);
        const Vec3 ngo = orientedGeoN(h);
        if (cosSurf <= 0.0) return false;                       // below the shading horizon
        stG = shadowTerminatorG(wi, h.n, ngo);                  // Chiang soft terminator (1 if flat)
        if (stG <= 0.0) return false;                           // behind true geometry: hard shadow
        farDist = length(scene.sceneCenter - h.p) + scene.sceneRadius;
        if (scene.occluded(h.p + ngo * 1e-6, wi, farDist)) return false;
        double pdfBsdf = cosSurf / PI;                          // cosine-hemisphere pdf for wi
        wMis = pdfW / (pdfW + pdfBsdf);                         // balance heuristic
        return true;
    }

    double neeEnv(const Scene& scene, const Hit& h, double rho, double invPdfLambda,
                  double lambda, Pcg32& rng) const {
        Vec3 wi; double cosSurf, stG, pdfW, wMis, farDist;
        if (!envGeom(scene, h, rng, wi, cosSurf, stG, pdfW, wMis, farDist)) return 0.0;
        double Lenv = scene.envRadiance(wi, lambda);
        if (Lenv <= 0.0) return 0.0;
        double contrib = (rho / PI) * Lenv * cosSurf * invPdfLambda / pdfW * wMis * stG;
        if (scene.backwardMedium().enabled)                       // Beer-Lambert to the scene exit
            contrib *= std::exp(-scene.backwardMedium().sigmaT(lambda) * farDist);
        return contrib;
    }

    // Hero-wavelength environment NEE: one shared env-direction sample, evaluated for
    // all `nUp` active wavelengths (fog-free hero fast path, no transmittance term).
    void neeEnvHero(const Scene& scene, const Hit& h, const double* rho, double* L,
                    const double* thr, const double* lam, const double* invPdf,
                    int nUp, Pcg32& rng) const {
        Vec3 wi; double cosSurf, stG, pdfW, wMis, farDist;
        if (!envGeom(scene, h, rng, wi, cosSurf, stG, pdfW, wMis, farDist)) return;
        for (int i = 0; i < nUp; ++i) {
            double Lenv = scene.envRadiance(wi, lam[i]);
            if (Lenv <= 0.0) continue;
            L[i] += thr[i] * (rho[i] / PI) * Lenv * cosSurf * invPdf[i] / pdfW * wMis * stG;
        }
    }

    // Environment NEE at a fog scattering vertex: same as neeEnv but the surface
    // BRDF/cosine is replaced by the single-scattering albedo and the HG phase
    // function (which is also the pdf used for the MIS weight against the phase-
    // sampled continuation). Only invoked when the scene has an env light.
    double neeEnvVolume(const Scene& scene, const Vec3& p, const Vec3& wIn,
                        double lambda, double invPdfLambda, Pcg32& rng) const {
        double pdfW;
        Vec3 wi = scene.sampleEnvDir(rng, pdfW);
        if (pdfW <= 0.0) return 0.0;
        double farDist = length(scene.sceneCenter - p) + scene.sceneRadius;
        if (scene.occluded(p + wi * 1e-6, wi, farDist)) return 0.0;
        double Lenv = scene.envRadiance(wi, lambda);
        if (Lenv <= 0.0) return 0.0;
        double phase  = scene.backwardMedium().phaseValue(dot(wIn, wi), lambda);  // == BSDF pdf here
        double albedo = scene.backwardMedium().albedo(lambda);
        double wMis   = pdfW / (pdfW + phase);          // balance heuristic
        double T = std::exp(-scene.backwardMedium().sigmaT(lambda) * farDist);
        return albedo * phase * Lenv * invPdfLambda / pdfW * wMis * T;
    }

    // Handle ONE surface material interaction on a single wavelength — the whole
    // material switch, factored out of radiance() so the scalar tracer and the hero
    // tracer (which de-heros before calling this) share one copy. `m` is the resolved
    // leaf material (Mix/Layered already peeled by the caller). All path state is
    // in/out; the surface's own emission is handled by the caller BEFORE this call.
    // Returns true if the path continues (ray + state updated), false if it terminated
    // (L already holds this path's final value). A `break` in the old switch maps to
    // `return true`, a `return L` maps to `return false`.
    bool interactMaterial(const Scene& scene, const Material& m, const Hit& h, Renderer& mats,
                          Ray& ray, double& lambda, double& invPdfLambda, double& thr, double& L,
                          bool& specularArrival, double& contBsdfPdf, MediumStack& stk,
                          Pcg32& rng, const SpdCache* spdCache = nullptr,
                          GiCtx gi = GiCtx{}) const {
        switch (m.type) {
            case MatType::Dielectric: {
                // Nested-dielectric PRIORITY resolution (Schmidt & Budge 2002). The
                // exterior IOR at this interface is the medium the ray is currently
                // travelling through (the highest-priority stack entry), not a hardcoded
                // 1.0 -- so a glass surface inside water refracts 1.33<->1.52. Where two
                // dielectrics overlap, the higher priority wins and the lower one's
                // boundary is suppressed (the ray passes straight through, unrefracted).
                // SAFE FALLBACK: the priority rule applies only when BOTH sides carry an
                // explicit priority; otherwise the interface degrades to the flat
                // air<->glass model (extIor 1.0), so priority-free scenes are unchanged.
                bool entering = dot(ray.d, h.ng) < 0.0;
                const int mi = (int)(&m - scene.mats.data());   // true index (Mix/Layered aware)
                const int pr = m.priority;               // INT_MIN if unset
                specularArrival = true;
                if (entering) {
                    const int outMat = stk.topMat();     // -1 == air
                    const int outPri = stk.topPri();     // INT_MIN == air
                    const bool ranked = m.hasPriority() &&
                        (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                    if (ranked && !stk.empty() && pr <= outPri) {   // suppressed entry
                        stk.push(mi, pr);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                        return true;
                    }
                    const double extIor = (ranked && outMat >= 0)
                        ? scene.mats[outMat].ior(lambda) : 1.0;
                    bool transmitted = false;
                    // Mode W: dominant Fresnel branch weighted into the throughput instead of
                    // a coin flip (see refractOrReflect's whittedWeight). Attenuating AFTER
                    // the call is safe because the ray has not been traced yet -- returning
                    // false here just ends the path at this vertex, as elsewhere in mode W.
                    double wW = 1.0;
                    ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted,
                                                extIor, whitted ? &wW : nullptr);
                    if (whitted && !whittedAttenuate(thr, wW)) return false;
                    if (transmitted) stk.push(mi, pr);
                    return true;
                } else {
                    MediumStack after = stk;             // exiting solid mi
                    after.popMat(mi);
                    const int newMat = after.topMat();   // -1 == air underneath
                    const int newPri = after.topPri();
                    const bool ranked = m.hasPriority() &&
                        (after.empty() || (newMat >= 0 && scene.mats[newMat].hasPriority()));
                    if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed exit
                        stk.popMat(mi);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                        return true;
                    }
                    const double extIor = (ranked && newMat >= 0)
                        ? scene.mats[newMat].ior(lambda) : 1.0;
                    bool transmitted = false;
                    double wW = 1.0;                      // mode W: see the entering branch
                    ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted,
                                                extIor, whitted ? &wW : nullptr);
                    if (whitted && !whittedAttenuate(thr, wW)) return false;
                    if (transmitted) stk.popMat(mi);      // TIR stays inside mi
                    return true;
                }
            }
            case MatType::ThinFilm: {
                // Mode W: dominant interference branch weighted into the throughput instead
                // of a coin flip, exactly as Dielectric does above (see thinFilmInterface's
                // whittedWeight). Attenuating AFTER the call is safe -- the ray has not been
                // traced yet, so returning false just ends the path at this vertex.
                Ray nr; double wW = 1.0;
                if (!mats.thinFilmInterface(scene, m, h, ray.d, lambda, rng, nr,
                                            whitted ? &wW : nullptr)) return false;
                if (whitted && !whittedAttenuate(thr, wW)) return false;
                ray = nr; specularArrival = true; return true;
            }
            case MatType::Multilayer: {
                Ray nr; double wW = 1.0;                  // mode W: see ThinFilm above
                if (!mats.multilayerInterface(m, h, ray.d, lambda, rng, nr,
                                              whitted ? &wW : nullptr)) return false;
                if (whitted && !whittedAttenuate(thr, wW)) return false;
                ray = nr; specularArrival = true; return true;
            }
            case MatType::Mirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // Whitted: carry the reflectance as WEIGHT instead of rolling for
                // survival. Same expected value, zero variance -- the whole reason a
                // deterministic preview converges at 1 spp where RR needs tens.
                if (whitted) { if (!whittedAttenuate(thr, r)) return false; }
                else if (rng.uniform() >= r) return false;      // RR absorb
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                specularArrival = true; return true;
            }
            case MatType::Grating: {
                // The grating equation is reciprocal, so backward tracing reuses the
                // same diffraction (m <-> -m symmetric). Specular per order.
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (whitted) { if (!whittedAttenuate(thr, r)) return false; }
                else if (rng.uniform() >= r) return false;      // RR absorb
                bool absorbedG;
                // Mode W: the diffraction ORDER comes off the (sIdx, bounce) lattice instead of
                // the rng, so every pixel picks the same order and the preview is noise-free
                // (see whittedOrderU / gratingDiffract).
                const double uOrd = whitted ? whittedOrderU(gi.sIdx, gi.bounce) : 0.0;
                Ray nr = mats.gratingDiffract(m, h, ray.d, lambda, rng, absorbedG,
                                              whitted ? &uOrd : nullptr);
                if (absorbedG) return false;
                ray = nr; specularArrival = true; return true;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // Whitted: a true beam splitter needs the path to FORK, which this
                // iterative loop cannot do. Take the dominant branch and weight it, so a
                // preview is stable rather than a 50/50 coin flipped per pixel. (The
                // minority branch is dropped, not just dimmed -- a half mirror previews
                // as whichever of reflection/transmission is stronger.)
                if (whitted) {
                    const bool refl = (r >= 0.5);
                    if (!whittedAttenuate(thr, refl ? r : 1.0 - r)) return false;
                    ray = refl ? Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)}
                               : Ray{h.p + ray.d * 1e-6, ray.d};
                } else if (rng.uniform() < r) ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                else                          ray = Ray{h.p + ray.d * 1e-6, ray.d};
                specularArrival = true; return true;
            }
            case MatType::Filter: {
                // Colored gel filter: pass straight through, survive with prob T(lambda).
                double t = clamp01(transmitSlot(scene, m, h, lambda));
                if (whitted) { if (!whittedAttenuate(thr, t)) return false; }
                else if (rng.uniform() >= t) return false;      // absorbed
                ray = Ray{h.p + ray.d * 1e-6, ray.d};      // direction unchanged
                specularArrival = true; return true;
            }
            case MatType::Glossy: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // Whitted: the lobe off a deterministic lattice rather than the rng, so the
                // direction is the same for every pixel (noise-free) but varies with the
                // sample index (so -spp actually resolves the lobe). At -spp 1 this IS the
                // mirror direction, which is exact for a near-mirror and over-sharpens as
                // roughness grows; the fix for that is more spp, which now works.
                if (whitted) {
                    if (!whittedAttenuate(thr, r)) return false;
                    Vec3 o = whittedGlossyDir(reflect(ray.d, h.n),
                                              materialRoughness(scene, m, h), gi.sIdx, gi.bounce);
                    if (dot(o, h.n) <= 0) return false;
                    ray = Ray{h.p + h.n * 1e-6, o};
                    specularArrival = true; return true;
                }
                if (rng.uniform() >= r) return false;
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0) return false;
                ray = Ray{h.p + h.n * 1e-6, o};
                specularArrival = true; return true;
            }
            case MatType::Fluorescent: {
                // Bispectral reradiation — backward adjoint of the forward tracer's
                // fluoroInteract(). Elastic base reflects at the output wavelength; the
                // fluorescent channel excites at a separately-sampled lambdaIn (Stokes
                // shift). Both channels NEE; one stochastic continuation carries indirect.
                double rhoEl = clamp01(m.reflect(lambda));   // elastic base at lambda(out)
                // `gi` matters (it selects giGrid over lightGrid at a gather vertex) —
                // omitting it here used to make a fluorescent surface pay lightGrid^2 shadow
                // rays inside a -gi gather while every Diffuse vertex paid giGrid^2, so the
                // two materials disagreed on the shadow lattice for no reason.
                L += thr * neeLight(scene, h, rhoEl, invPdfLambda, lambda, rng, spdCache, gi);
                if (scene.envIndex >= 0)
                    L += thr * neeEnv(scene, h, rhoEl, invPdfLambda, lambda, rng);
                double Mint = m.fluoEmitSampler.integral;
                bool haveFluoro = (Mint > 0.0 && m.fluoYield > 0.0);
                double gOut = 0.0, rhoFluo = 0.0, lambdaIn = 0.0, invPdfIn = 0.0;
                if (haveFluoro) {
                    gOut = (m.fluoEmit(lambda) / Mint) * invPdfLambda;
                    double pin = 0.0;
                    // Mode W: the Stokes-shift EXCITATION wavelength comes off the
                    // (sIdx, bounce) lattice rather than the rng -- the same CDF inversion
                    // (sampleAt), just a stratified u, so the estimator is untouched and only
                    // the per-pixel luck goes away. This was mode W's last rng draw here; the
                    // elastic/fluoro continuation coin below is unreachable because mode W
                    // implies directOnly, which returns first.
                    //
                    // The CDF is the material's own excitation sampler (absorb x illuminant),
                    // so every draw lands inside the dye's absorption band -- see
                    // Material::fluoInSampler. Scenes whose dye cannot be excited at all by
                    // this illuminant have an empty sampler; fall back to the illuminant so
                    // the branch still terminates (rhoFluo will be 0 anyway).
                    const EmissionSampler& inS = (m.fluoInSampler.integral > 0.0)
                                                     ? m.fluoInSampler : scene.emitSampler;
                    lambdaIn = whitted
                        ? inS.sampleAt(whittedFluoroU(gi.sIdx, gi.bounce), pin)
                        : inS.sample(rng, pin);
                    if (pin > 0.0) {
                        // 1/pdf of the sampler we actually drew from. (Pre-0.115.0 this
                        // read scene.invPdfLambda(lambdaIn) -- correct only while the
                        // sampler WAS the illuminant, and even then it mixed an analytic
                        // emitG/g against a bin-discretised CDF draw.)
                        invPdfIn = 1.0 / pin;
                        double rhoIn, aEffIn;
                        fluoroWeights(m, lambdaIn, rhoIn, aEffIn);   // shared with forward
                        rhoFluo = aEffIn * m.fluoYield;              // reradiation albedo @lambdaIn
                        if (rhoFluo > 0.0) {                          // fluoro DIRECT NEE
                            // (lambdaIn ≠ the cached wavelengths → matches() fails and
                            // this evaluates spdFn live, exactly as before.)
                            L += thr * gOut * neeLight(scene, h, rhoFluo, invPdfIn, lambdaIn,
                                                       rng, spdCache, gi);
                            if (scene.envIndex >= 0)
                                L += thr * gOut * neeEnv(scene, h, rhoFluo, invPdfIn, lambdaIn, rng);
                        }
                    }
                }
                if (directOnly) return false;                // Whitted: no indirect (elastic or fluoro)
                double wFluo = gOut * rhoFluo;               // natural indirect-fluoro weight
                double pF = (wFluo > 0.0) ? std::min(std::max(0.0, 1.0 - rhoEl), wFluo) : 0.0;
                double u = rng.uniform();
                if (u < rhoEl) {                             // elastic continuation
                    Vec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                    ray = Ray{h.p + h.n * 1e-6, wOut};
                    specularArrival = false; return true;
                } else if (u < rhoEl + pF) {                 // fluoro (wavelength-switched)
                    thr *= wFluo / pF;
                    lambda = lambdaIn;                       // Stokes shift (to the input wl)
                    invPdfLambda = invPdfIn;
                    Vec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                    ray = Ray{h.p + h.n * 1e-6, wOut};
                    specularArrival = false; return true;
                }
                return false;                                // absorbed / terminated
            }
            case MatType::DiffuseTransmit: {
                // Two-lobe Lambertian: NEE the reflect lobe in the front hemisphere and
                // the transmit lobe in the back (a normal-flipped Hit reuses neeLight/env).
                double rhoR = clamp01(diffuseReflectance(scene, m, h, lambda));
                double rhoT = clamp01(transmitSlot(scene, m, h, lambda));
                double sum = rhoR + rhoT;
                if (sum > 1.0) { rhoR /= sum; rhoT /= sum; sum = 1.0; }   // energy guard
                L += thr * neeLight(scene, h, rhoR, invPdfLambda, lambda, rng, spdCache, gi);
                if (scene.envIndex >= 0)
                    L += thr * neeEnv(scene, h, rhoR, invPdfLambda, lambda, rng);
                Hit hb = h; hb.n = -h.n;                 // back hemisphere for the transmit lobe
                L += thr * neeLight(scene, hb, rhoT, invPdfLambda, lambda, rng, spdCache, gi);
                if (scene.envIndex >= 0)
                    L += thr * neeEnv(scene, hb, rhoT, invPdfLambda, lambda, rng);
                if (whitted) {   // both lobes gather, each into its own hemisphere
                    if (giUseGather(gi)) {
                        L += thr * giGather(scene, h,  rhoR, lambda, invPdfLambda, rng, spdCache, gi);
                        L += thr * giGather(scene, hb, rhoT, lambda, invPdfLambda, rng, spdCache, gi);
                    } else if (ambient > 0.0) {
                        L += thr * (rhoR + rhoT) * ambient;
                    }
                }
                if (directOnly) return false;            // Whitted: no diffuse indirect
                double u = rng.uniform();
                if (u < rhoR) {                          // reflect continuation (front)
                    Vec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                    ray = Ray{h.p + h.n * 1e-6, wOut};
                    specularArrival = false; return true;
                } else if (u < sum) {                    // transmit continuation (back)
                    Vec3 wOut = cosineHemisphere(-h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, -h.n)) / PI;
                    ray = Ray{h.p - h.n * 1e-6, wOut};
                    specularArrival = false; return true;
                }
                return false;                            // absorbed / terminated
            }
            case MatType::Diffuse:
            default: {
                double rho = clamp01(diffuseReflectance(scene, m, h, lambda));
                L += thr * neeLight(scene, h, rho, invPdfLambda, lambda, rng, spdCache, gi);
                if (scene.envIndex >= 0)   // env-NEE toward the sky (MIS'd on miss)
                    L += thr * neeEnv(scene, h, rho, invPdfLambda, lambda, rng);
                if (whitted) {             // indirect diffuse: real gather, or the flat stand-in
                    if (giUseGather(gi))
                        L += thr * giGather(scene, h, rho, lambda, invPdfLambda, rng, spdCache, gi);
                    else if (ambient > 0.0)
                        L += thr * rho * ambient;
                }
                if (directOnly) return false;   // Whitted: no diffuse indirect
                // Russian roulette on the albedo (throughput unchanged on survival).
                if (rng.uniform() >= rho) return false;
                Vec3 wOut = cosineHemisphere(h.n, rng);
                contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                ray = Ray{h.p + h.n * 1e-6, wOut};
                specularArrival = false; return true;
            }
        }
        return true;   // unreachable (Diffuse/default covers every type)
    }

    // Estimate spectral-weighted radiance for a single wavelength along `ray`.
    // `invPdfLambda` = emitG/g(lambda), the reciprocal of the sampled-wavelength
    // pdf; an emitter's Le/pdf weight is its SPD(lambda) * invPdfLambda.
    double radiance(const Scene& scene, Ray ray, double lambda, double invPdfLambda,
                    Pcg32& rng, const SpdCache* spdCache = nullptr,
                    GiCtx gi = GiCtx{}) const {
        double L = 0.0, thr = 1.0;
        // Camera ray may see the light directly; a gather ray may not (see radianceHero).
        bool specularArrival = (gi.depth == 0);
        const int maxB = gi.depth ? std::min(maxBounce, giBounce) : maxBounce;
        double contBsdfPdf = 0.0;      // solid-angle pdf of the current continuation
                                       // ray (for env-miss MIS after a diffuse/volume
                                       // bounce; unused while specularArrival)
        Renderer mats;                 // shared material sampling (stateless)
        mats.diffraction = diffraction; // grating order count follows the CLI toggle

        // Nested-dielectric medium stack: the solids the ray is currently inside. The
        // current medium (for Beer-Lambert absorption + the exterior IOR at the next
        // interface) is the highest-priority entry. Replaces the old single `interior`
        // pointer; behaves identically for a lone dielectric.
        MediumStack stk;
        auto curAbsorb = [&](double lam) -> double {           // sigma_a of the current medium
            int mi = stk.topMat();
            return (mi >= 0) ? scene.mats[mi].absorb(lam) : 0.0;
        };

        // GRADIENT-INDEX (GRIN) support. Any medium carrying an `ior` field bends
        // rays that pass through its bound. `grinAny` gates the shared marcher off so
        // `ior`-free scenes stay bit-identical. The marcher itself now lives in grin.h
        // and is shared verbatim by the forward and bidirectional tracers.
        bool grinAny = grin::sceneHasGrin(scene);

        for (int b = 0; b < maxB; ++b) {
            // Publish the bounce index so a deterministic per-vertex choice (mode W's glossy
            // lobe) can pick a decorrelated sequence at each depth. Costs nothing otherwise.
            gi.bounce = b;
            // GRIN curved marching pre-pass: advance the ray through any gradient-index
            // region it enters, integrating the Eikonal equation d/ds(n·dr/ds)=∇n in
            // small steps so the path bends. Pure marching does NOT consume a bounce;
            // when the ray reaches a surface (within one step) or leaves all GRIN regions
            // it stops and we fall through to the straight-ray body.
            if (grinAny) grin::march(scene, ray);

            Hit h = scene.closestHit(ray);
            double dSurf = h.valid ? h.t : 1e30;

            // Homogeneous fog: sample a free-flight collision that competes with
            // the surface. On a volume collision, estimate direct light via phase-
            // function NEE, then scatter (HG) or absorb — analog, throughput
            // unchanged. Mirrors the forward tracer exactly, so the two agree.
            if (scene.backwardMedium().enabled) {
                double st = scene.backwardMedium().sigmaT(lambda);
                if (st > 0.0) {
                    double tMed = -std::log(1.0 - rng.uniform()) / st;
                    if (tMed < dSurf) {
                        Vec3 p = ray.o + ray.d * tMed;
                        // Beer-Lambert attenuation over the in-glass free-flight leg.
                        {
                            double a = curAbsorb(lambda);
                            if (a > 0.0) thr *= std::exp(-a * tMed);
                        }
                        L += thr * neeVolume(scene, p, ray.d, lambda, invPdfLambda, rng, spdCache);
                        if (scene.envIndex >= 0)   // env-NEE at the volume vertex
                            L += thr * neeEnvVolume(scene, p, ray.d, lambda, invPdfLambda, rng);
                        if (directOnly) return L;  // Whitted: single-scatter only, no indirect
                        if (rng.uniform() >= scene.backwardMedium().albedo(lambda)) return L; // absorbed
                        Vec3 wOut = scene.backwardMedium().phaseSample(ray.d, lambda, rng, contBsdfPdf);
                        ray = Ray{p, wOut};
                        specularArrival = false;   // phase-NEE covered the direct light
                        continue;
                    }
                }
            }

            // Ray escaped the scene: pick up the environment radiance from the escape
            // direction (0 if no env light; constant env ignores the direction, an
            // image env samples the lat-long map). On a camera/specular arrival there
            // is no competing env-NEE, so it is added at full weight (the directly-
            // viewed background and specular-chain sky). On a diffuse/volume arrival
            // the env is also sampled by neeEnv/neeEnvVolume at the previous vertex,
            // so this BSDF-sampled hit is MIS-weighted (balance heuristic) against
            // that NEE to avoid double-counting. Same spdFn*invPdfLambda form as
            // surface emission, so forward and backward agree on env illumination.
            // Beer-Lambert attenuation over the in-glass segment up to the surface
            // (only when the ray actually reached a surface inside a dielectric).
            if (h.valid) {
                double a = curAbsorb(lambda);
                if (a > 0.0) thr *= std::exp(-a * dSurf);
            }

            if (!h.valid) {
                if (scene.envIndex >= 0) {
                    double Lenv = scene.envRadiance(ray.d, lambda) * invPdfLambda;
                    if (specularArrival) {
                        L += thr * Lenv;
                    } else {
                        double pdfEnv = scene.envPdfDir(ray.d);
                        double wMis = (contBsdfPdf + pdfEnv > 0.0)
                                          ? contBsdfPdf / (contBsdfPdf + pdfEnv) : 0.0;
                        L += thr * Lenv * wMis;
                    }
                }
                // Directly-viewed solar disc: camera / specular arrivals only. A diffuse
                // or volume vertex already spent its one estimator on the sun via
                // NEE (emitterGeom / neeVolume) and sets specularArrival = false, so
                // this is a clean single-strategy split, not a missing MIS weight.
                if (scene.sunCount > 0 && specularArrival)
                    L += thr * scene.sunRadiance(ray.d, lambda) * invPdfLambda;
                // Escaped gather ray → the far-field `ambient` fill (see radianceHero).
                if (gi.depth && ambient > 0.0) L += thr * ambient;
                return L;
            }
            const Material* mp = &scene.mats[h.matId];
            // Stochastic mix: resolve to a child material (or terminate on the
            // leftover absorption slice) before the switch, mirroring the forward
            // tracer so the two agree on the blended surface by construction.
            if (mp->type == MatType::Mix) {
                int child = whitted ? mixResolveDominant(scene, *mp, h)
                                    : mixResolveChild(scene, *mp, h, rng.uniform());
                if (child < 0) return L;   // absorbed
                mp = &scene.mats[child];
            }
            // Physical layered stack: reflect off the coat interface with prob R
            // (a glossy lobe about the mirror direction), else enter and pick one
            // body lobe. Mirrors the forward tracer so both split the photon budget
            // identically; the coat reflection is lossless (throughput unchanged).
            if (mp->type == MatType::Layered) {
                const Material& cm = *mp;
                double R = layeredCoatReflectance(scene, cm, h, ray.d, lambda);
                // Whitted: no fork, so take the dominant layer and weight it. A real
                // clearcoat has R ~ 0.04-0.1 at normal incidence, so this previews the
                // BODY (the paint) and drops the coat sheen -- the right trade, since the
                // sheen is the part you can least see at 1 spp anyway.
                if (whitted) {
                    if (R >= 0.5) {
                        if (!whittedAttenuate(thr, R)) return L;
                        Vec3 o = whittedGlossyDir(reflect(ray.d, h.n),
                                                  materialRoughness(scene, cm, h), gi.sIdx, b);
                        if (dot(o, h.n) <= 0) return L;
                        ray = Ray{h.p + h.n * 1e-6, o};
                        specularArrival = true;
                        continue;
                    }
                    if (!whittedAttenuate(thr, 1.0 - R)) return L;
                    int child = mixDominantChild(cm);
                    if (child < 0) return L;
                    mp = &scene.mats[child];
                } else {
                if (rng.uniform() < R) {
                    Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, cm, h), rng);
                    if (dot(o, h.n) <= 0) return L;
                    ray = Ray{h.p + h.n * 1e-6, o};
                    specularArrival = true;
                    continue;
                }
                int child = mixPickChild(cm, rng.uniform());   // body lobe
                if (child < 0) return L;                        // leftover absorbs
                mp = &scene.mats[child];
                }
            }
            const Material& m = *mp;

            // Emission (add only on specular/camera arrival; NEE covers diffuse).
            // The surface's own emitted radiance Le=emitSlot(...), weighted by the
            // reciprocal wavelength pdf (= its SPD integral for a single light).
            // emitSlot applies any `emit pattern:` at this hit; the NEE side below
            // applies the SAME profile via emitterSamplePoint, which is what keeps the
            // two estimators consistent (see Material::emitPat).
            if (m.isLight && specularArrival && dot(ray.d, h.ng) < 0.0)
                L += thr * emitSlot(scene, m, h, lambda) * invPdfLambda;

            if (!interactMaterial(scene, m, h, mats, ray, lambda, invPdfLambda, thr, L,
                                  specularArrival, contBsdfPdf, stk, rng, spdCache, gi))
                return L;                                 // path terminated in the interaction
        }
        return L;
    }

    // Hero-wavelength variant of radiance(): carries C wavelengths (hero + C-1
    // stratified secondaries) down ONE camera path. Index 0 is the hero; it drives
    // every sampling decision with the same rng stream a single-wavelength path would,
    // while the secondaries ride along and are reweighted per-λ. At a dispersive /
    // wavelength-switching material (anything but Diffuse/DiffuseTransmit) the
    // secondaries de-hero (terminate) and the hero is boosted ×C so it alone carries an
    // unbiased single-λ estimate onward (PBRT-v4's TerminateSecondary convention). The
    // caller restricts this to scenes WITHOUT participating media / GRIN / a physical
    // lens, so those branches are absent here. Fills Lout[0..C).
    void radianceHero(const Scene& scene, Ray ray, const double* lamIn,
                      const double* invPdfIn, int C, double* Lout, Pcg32& rng,
                      const SpdCache* spdCache = nullptr, GiCtx gi = GiCtx{}) const {
        // A fresh camera/gather bundle: unit throughput, empty medium stack, at bounce 0.
        // A camera ray may see a light directly; a GATHER ray may not -- the vertex it
        // left already NEE'd the direct light, so counting the emitter again here would
        // double it. A specular bounce re-arms this, so gold-bounced light still lands.
        double thr[hero::kHeroMax];
        for (int i = 0; i < C; ++i) thr[i] = 1.0;
        radianceHeroLoop(scene, ray, MediumStack{}, lamIn, invPdfIn, thr, C,
                         /*secAlive=*/(C > 1), /*specularArrival=*/(gi.depth == 0),
                         /*contBsdfPdf=*/0.0, /*bounce0=*/0, Lout, rng, spdCache, gi);
    }

    // Bounce loop for a hero bundle already sitting at (`ray`, `stk`) with
    // `secAlive ? C : 1` live wavelengths carrying `lamIn[]`/`invPdfIn[]`/`thrIn[]`,
    // resuming at bounce index `bounce0`. Split out of radianceHero so the `heroSplit`
    // policy can RE-ENTER it once per monochromatic sub-path that a dispersive interface
    // fans out (see the dispersive case below) -- the direct twin of the forward tracer's
    // tracePhotonHeroLoop. Every sub-path is spawned with `secAlive == false` and the split
    // branch is guarded on `secAlive`, so a sub-path can never split again: recursion is at
    // most one level deep and the per-frame footprint (a MediumStack plus four kHeroMax
    // double arrays) is bounded.
    //
    // `Lout` is ASSIGNED, not accumulated, exactly as before -- a split parent therefore
    // adds each sub-path's returned radiance into its OWN L[i] slot (see below), which is
    // what keeps wavelength i's radiance attributed to wavelength i.
    void radianceHeroLoop(const Scene& scene, Ray ray, MediumStack stk, const double* lamIn,
                          const double* invPdfIn, const double* thrIn, int C, bool secAlive,
                          bool specularArrival, double contBsdfPdf, int bounce0,
                          double* Lout, Pcg32& rng, const SpdCache* spdCache,
                          GiCtx gi) const {
        double lam[hero::kHeroMax], invPdf[hero::kHeroMax], thr[hero::kHeroMax], L[hero::kHeroMax];
        // Copy only the LIVE entries: a monochromatic sub-path spawned by the split fills
        // only slot 0 of its lamIn/invPdfIn/thrIn, so reading all C would read
        // indeterminate values (harmless while nUp==1 ignores them, but still UB).
        const int nLive = secAlive ? C : 1;
        for (int i = 0; i < nLive; ++i) {
            lam[i] = lamIn[i]; invPdf[i] = invPdfIn[i]; thr[i] = thrIn[i]; L[i] = 0.0;
        }
        for (int i = nLive; i < C; ++i) { lam[i] = 0.0; invPdf[i] = 0.0; thr[i] = 0.0; L[i] = 0.0; }
        // Gather rays are bounce-capped (see giBounce) so a highly reflective lattice
        // cannot turn one gather direction into a 60-deep ricochet.
        const int maxB = gi.depth ? std::min(maxBounce, giBounce) : maxBounce;
        Renderer mats; mats.diffraction = diffraction;

        auto finish = [&]() { for (int i = 0; i < C; ++i) Lout[i] = L[i]; };
        auto deHero = [&]() {            // terminate secondaries, boost hero ×C
            if (!secAlive) return;
            thr[0] *= (double)C;
            secAlive = false;
        };

        for (int b = bounce0; b < maxB; ++b) {
            int nUp = secAlive ? C : 1;   // wavelengths still being propagated
            gi.bounce = b;                // see the scalar twin: mode W's per-vertex lattice
            Hit h = scene.closestHit(ray);
            double dSurf = h.valid ? h.t : 1e30;

            // Beer-Lambert over the in-glass segment. A non-empty stack implies we've
            // already de-hero'd (dielectric entry de-heros), so nUp == 1 whenever
            // absorption is non-zero; the loop still handles the general case.
            if (h.valid) {
                int mi = stk.topMat();
                if (mi >= 0) {
                    for (int i = 0; i < nUp; ++i) {
                        double a = scene.mats[mi].absorb(lam[i]);
                        if (a > 0.0) thr[i] *= std::exp(-a * dSurf);
                    }
                }
            }

            if (!h.valid) {              // env-miss (full weight on specular arrival, else MIS)
                if (scene.envIndex >= 0) {
                    if (specularArrival) {
                        for (int i = 0; i < nUp; ++i)
                            L[i] += thr[i] * scene.envRadiance(ray.d, lam[i]) * invPdf[i];
                    } else {
                        double pdfEnv = scene.envPdfDir(ray.d);
                        double wMis = (contBsdfPdf + pdfEnv > 0.0)
                                          ? contBsdfPdf / (contBsdfPdf + pdfEnv) : 0.0;
                        for (int i = 0; i < nUp; ++i)
                            L[i] += thr[i] * scene.envRadiance(ray.d, lam[i]) * invPdf[i] * wMis;
                    }
                }
                if (scene.sunCount > 0 && specularArrival)   // directly-viewed solar disc
                    for (int i = 0; i < nUp; ++i)
                        L[i] += thr[i] * scene.sunRadiance(ray.d, lam[i]) * invPdf[i];
                // A GATHER ray that escaped the scene picks up `ambient` as the far-field
                // fill. This is what makes -ambient and -gi compose instead of compete:
                // the gather supplies the near field (occlusion + bleeding) and the
                // constant supplies whatever lies beyond the geometry -- and in an empty
                // scene every direction escapes, so the gather collapses exactly back to
                // the flat `rho * ambient` term it replaced (no exposure step).
                if (gi.depth && ambient > 0.0)
                    for (int i = 0; i < nUp; ++i) L[i] += thr[i] * ambient;
                finish(); return;
            }

            const Material* mp = &scene.mats[h.matId];
            if (mp->type == MatType::Mix) {
                int child = whitted ? mixResolveDominant(scene, *mp, h)
                                    : mixResolveChild(scene, *mp, h, rng.uniform());
                if (child < 0) { finish(); return; }
                mp = &scene.mats[child];
            }
            if (mp->type == MatType::Layered) {
                // Physical layered stack (specular coat over a weighted body). The coat
                // interface is NOT dispersive in DIRECTION -- the sheen is a glossy lobe
                // about the mirror direction and the body-lobe pick is a material index --
                // so the ONLY λ dependence here is the scalar coat reflectance R(λ). That
                // means the bundle can normally ride straight through carrying a per-λ
                // weight, with no de-hero at all.
                //
                // Through v0.115.0 this de-hero'd UNCONDITIONALLY, which at -spp 1 collapsed
                // every layered surface onto the bundle's single shared λ and rendered it
                // saturated-monochromatic -- exactly the failure N1 fixed for glass, and the
                // reason the viewer used to force 16 passes on any layered scene.
                const Material& cm = *mp;
                double Rl[hero::kHeroMax];
                for (int i = 0; i < nUp; ++i)
                    Rl[i] = layeredCoatReflectance(scene, cm, h, ray.d, lam[i]);
                // Stochastic: ONE shared coat coin, compared per-λ. u is uniform, so
                // P(u < R_i) == R_i exactly for every live λ -- common-random-number analog
                // splitting, unbiased per λ with NO reweight needed (the probability IS the
                // weight, as in the scalar twin). At nUp == 1 that is the scalar code
                // verbatim: same single draw, same order, throughput untouched.
                // Whitted: no coin at all, just the dominant branch weighted per λ.
                const double uCoat = whitted ? 0.0 : rng.uniform();
                const bool refl0 = whitted ? (Rl[0] >= 0.5) : (uCoat < Rl[0]);
                bool agree = true;
                for (int i = 1; i < nUp; ++i)
                    if ((whitted ? (Rl[i] >= 0.5) : (uCoat < Rl[i])) != refl0) { agree = false; break; }
                if (!agree) {
                    // A genuinely CHROMATIC coat: a thin-film Airy stack, or a Fresnel coat
                    // sitting right on mode W's R >= 0.5 dominant-branch threshold. Some λ
                    // take the sheen while the rest enter the body, so one shared continuation
                    // cannot serve them all. Fan out exactly like the split-at-dispersion case
                    // below: each secondary re-enters THIS vertex as its own monochromatic
                    // sub-path, makes its own coat decision, and lands in its own L[i] slot.
                    //
                    // Re-entering at bounce `b` (not b + 1) is safe because nUp > 1 implies an
                    // EMPTY medium stack -- every dielectric entry de-heros or splits, so a
                    // bundle wider than 1 is never inside glass -- which makes the
                    // Beer-Lambert step at the top of the loop a no-op that cannot be
                    // double-applied. closestHit is deterministic, so the sub-path lands on
                    // this same vertex; the cost is one redundant trace on a rare path.
                    for (int i = 1; i < nUp; ++i) {
                        if (invPdf[i] == 0.0) continue;      // dead secondary (zero-mass λ bin)
                        double sLam = lam[i], sInv = invPdf[i], sThr = thr[i];
                        SpdCache sCache;                     // re-point at λ_i's COLUMN
                        const SpdCache* sCp = nullptr;
                        if (spdCache && spdCache->lam && spdCache->spd && i < spdCache->C) {
                            sCache.lam = spdCache->lam + i;
                            sCache.spd = spdCache->spd + i;
                            sCache.C   = spdCache->C;
                            sCp = &sCache;
                        }
                        double sub[hero::kHeroMax];
                        radianceHeroLoop(scene, ray, stk, &sLam, &sInv, &sThr,
                                         /*C=*/1, /*secAlive=*/false, specularArrival,
                                         contBsdfPdf, b, sub, rng, sCp, gi);
                        L[i] += sub[0];      // this wavelength's own estimate, own slot
                        thr[i] = 0.0;        // now that sub-path's business, not ours
                    }
                    secAlive = false; nUp = 1;   // hero carries on alone, UNBOOSTED
                }
                if (refl0) {                     // coat sheen: every live λ reflects
                    // Whitted only: the deterministic branch carries R as a WEIGHT (the
                    // stochastic coin above already paid for itself). Stop only once the
                    // WHOLE bundle is under the bailout, as in the Glossy case below.
                    if (whitted) {
                        for (int i = 0; i < nUp; ++i) thr[i] *= Rl[i];
                        if (hero::maxOf(thr, nUp) <= kWhittedCutoff) { finish(); return; }
                    }
                    Vec3 o = whitted
                                 ? whittedGlossyDir(reflect(ray.d, h.n),
                                                    materialRoughness(scene, cm, h), gi.sIdx, b)
                                 : sampleGlossy(reflect(ray.d, h.n),
                                                materialRoughness(scene, cm, h), rng);
                    if (dot(o, h.n) <= 0) { finish(); return; }
                    ray = Ray{h.p + h.n * 1e-6, o};
                    specularArrival = true;
                    continue;
                }
                if (whitted) {                   // body: every live λ enters the stack
                    for (int i = 0; i < nUp; ++i) thr[i] *= 1.0 - Rl[i];
                    if (hero::maxOf(thr, nUp) <= kWhittedCutoff) { finish(); return; }
                }
                int child = whitted ? mixDominantChild(cm) : mixPickChild(cm, rng.uniform());
                if (child < 0) { finish(); return; }     // leftover slice absorbs
                mp = &scene.mats[child];
            }
            const Material& m = *mp;

            // Surface emission on a specular/camera arrival (NEE covers diffuse).
            // An `emit pattern:` is achromatic, so evaluate it ONCE for the whole hero
            // bundle rather than per-wavelength inside emitSlot.
            if (m.isLight && specularArrival && dot(ray.d, h.ng) < 0.0) {
                double ep = (m.emitPat < 0) ? 1.0 : slotPatMul(scene, m.emitPat, h);
                for (int i = 0; i < nUp; ++i)
                    L[i] += thr[i] * m.emit(lam[i]) * ep * invPdf[i];
            }

            switch (m.type) {
                case MatType::DiffuseTransmit: {
                    double rhoR[hero::kHeroMax], rhoT[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i) {
                        double rr = clamp01(diffuseReflectance(scene, m, h, lam[i]));
                        double rt = clamp01(transmitSlot(scene, m, h, lam[i]));
                        double s = rr + rt;
                        if (s > 1.0) { rr /= s; rt /= s; }       // per-λ energy guard
                        rhoR[i] = rr; rhoT[i] = rt;
                    }
                    neeLightHero(scene, h, rhoR, L, thr, lam, invPdf, nUp, rng, spdCache, gi);
                    if (scene.envIndex >= 0)
                        neeEnvHero(scene, h, rhoR, L, thr, lam, invPdf, nUp, rng);
                    Hit hb = h; hb.n = -h.n;                     // back hemisphere (transmit lobe)
                    neeLightHero(scene, hb, rhoT, L, thr, lam, invPdf, nUp, rng, spdCache, gi);
                    if (scene.envIndex >= 0)
                        neeEnvHero(scene, hb, rhoT, L, thr, lam, invPdf, nUp, rng);
                    if (whitted) {   // both lobes gather, each into its own hemisphere
                        if (giUseGather(gi)) {
                            giGatherHero(scene, h,  rhoR, L, thr, lam, invPdf, nUp, rng, spdCache, gi);
                            giGatherHero(scene, hb, rhoT, L, thr, lam, invPdf, nUp, rng, spdCache, gi);
                        } else if (ambient > 0.0) {
                            for (int i = 0; i < nUp; ++i)
                                L[i] += thr[i] * (rhoR[i] + rhoT[i]) * ambient;
                        }
                    }
                    if (directOnly) { finish(); return; }        // Whitted: no diffuse indirect
                    // Lobe pick + RR over the whole bundle (see the Diffuse case): the
                    // reflect/transmit probabilities are the per-lobe MAX over live λ, so no
                    // secondary is ever amplified. The maxima can sum past 1 (each λ alone is
                    // guarded), in which case both shrink proportionally. At nUp == 1 the two
                    // maxima are rhoR[0]/rhoT[0], their sum is already <= 1, and every
                    // reweight is *= 1.0 — the scalar code verbatim.
                    double qR = hero::maxOf(rhoR, nUp), qT = hero::maxOf(rhoT, nUp);
                    double sumHero = qR + qT;
                    if (nUp > 1 && sumHero > 1.0) { qR /= sumHero; qT /= sumHero; sumHero = qR + qT; }
                    double u = rng.uniform();
                    if (u < qR) {                                // reflect (front)
                        for (int i = 0; i < nUp; ++i) thr[i] *= rhoR[i] / qR;
                        Vec3 wOut = cosineHemisphere(h.n, rng);
                        contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                        ray = Ray{h.p + h.n * 1e-6, wOut};
                        specularArrival = false; break;
                    } else if (u < sumHero) {                    // transmit (back)
                        for (int i = 0; i < nUp; ++i) thr[i] *= rhoT[i] / qT;
                        Vec3 wOut = cosineHemisphere(-h.n, rng);
                        contBsdfPdf = std::max(0.0, dot(wOut, -h.n)) / PI;
                        ray = Ray{h.p - h.n * 1e-6, wOut};
                        specularArrival = false; break;
                    }
                    finish(); return;                            // absorbed
                }
                case MatType::Mirror:
                case MatType::Filter:
                case MatType::Glossy: {
                    // ACHROMATIC delta lobes: specular (so no NEE, specularArrival stays
                    // true) but the outgoing DIRECTION does not depend on λ — a mirror
                    // reflects, a gel passes straight through, a glossy lobe is the mirror
                    // direction blurred by a λ-independent roughness. So the bundle keeps
                    // riding; only the per-λ coefficient differs. Without this the hero
                    // bundle died at the first chrome/gel surface, which on a mirror-heavy
                    // scene left hero buying almost nothing (see known-issues.md).
                    //
                    // The scalar path survives by ANALOG Russian roulette on the hero's own
                    // coefficient with thr unchanged. Rolling that coin on the hero alone
                    // would kill live secondaries whenever c_hero == 0 (a Wratten gel is 0
                    // across most of the spectrum), so the survival probability is the MAX
                    // over live λ and the survivors reweight by c_i/q. With nUp == 1,
                    // q == c[0] and thr[0] *= 1.0, so this is the scalar code verbatim —
                    // same rng draws, same order, bit-identical.
                    double c[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i)
                        c[i] = (m.type == MatType::Filter) ? clamp01(transmitSlot(scene, m, h, lam[i]))
                                                           : clamp01(reflectSlot(scene, m, h, lam[i]));
                    const double q = hero::maxOf(c, nUp);
                    if (whitted) {
                        // Deterministic: carry every live λ's coefficient as weight (no
                        // coin, no c_i/q reweight) and stop only when the whole bundle
                        // has fallen under the bailout.
                        for (int i = 0; i < nUp; ++i) thr[i] *= c[i];
                        if (hero::maxOf(thr, nUp) <= kWhittedCutoff) { finish(); return; }
                    } else {
                        if (rng.uniform() >= q) { finish(); return; }   // RR absorb (q == 0 always absorbs)
                        for (int i = 0; i < nUp; ++i) thr[i] *= c[i] / q;
                    }
                    if (m.type == MatType::Mirror) {
                        ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                    } else if (m.type == MatType::Filter) {
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};       // direction unchanged
                    } else if (whitted) {
                        // Glossy: the lobe off the deterministic lattice (mirror at sample 0).
                        Vec3 o = whittedGlossyDir(reflect(ray.d, h.n),
                                                  materialRoughness(scene, m, h), gi.sIdx, b);
                        if (dot(o, h.n) <= 0) { finish(); return; }
                        ray = Ray{h.p + h.n * 1e-6, o};
                    } else {
                        Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                        if (dot(o, h.n) <= 0) { finish(); return; }
                        ray = Ray{h.p + h.n * 1e-6, o};
                    }
                    specularArrival = true;
                    break;
                }
                case MatType::Dielectric:
                case MatType::ThinFilm:
                case MatType::Multilayer:
                case MatType::Grating:
                case MatType::HalfMirror:
                case MatType::Fluorescent: {
                    // Dispersive / wavelength-switching: the outgoing direction (and, for a
                    // grating/fluorophore, the wavelength itself) depends on λ, so the bundle
                    // cannot keep riding one shared direction past this interface.
                    if (heroSplit && secAlive && nUp > 1) {
                        // SPLIT-AT-DISPERSION: fan out instead of de-hero'ing. Each secondary
                        // runs the SAME interaction with its OWN λ -- refracting along its own
                        // Snell direction / diffracting into its own grating order -- and then
                        // continues as an independent monochromatic sub-path from this vertex.
                        // Its radiance lands in L[i], the slot for ITS wavelength, so the
                        // caller's per-λ cieXYZ splat stays correctly attributed.
                        //
                        // No ×C boost anywhere: each of the C wavelengths now carries its own
                        // unboosted estimate and the caller averages them (Lh[i]/C), whereas
                        // de-hero boosts the lone survivor to stand in for all C. Both are
                        // unbiased; this one is simply not collapsed.
                        for (int i = 1; i < nUp; ++i) {
                            if (invPdf[i] == 0.0) continue;   // dead secondary (zero-mass λ bin)
                            // Sub-path state: mutable per-λ copies (interactMaterial takes
                            // lambda/invPdf by reference -- a fluorescent Stokes shift rewrites
                            // them) and a private medium stack, since sub-paths diverge here.
                            double sLam = lam[i], sInv = invPdf[i], sThr = thr[i], sL = 0.0;
                            bool sSpec = specularArrival;
                            double sPdf = contBsdfPdf;
                            MediumStack sStk = stk;
                            Ray sRay = ray;
                            // Re-point the emitter-SPD cache at wavelength i's COLUMN. The
                            // table is emitter-major with stride C (spd[e*C + i]), so offsetting
                            // the base by i and KEEPING the stride makes the sub-path's
                            // spd[e*C + 0] read exactly emitter e at λ_i -- zero-copy, and
                            // matches(&sLam, 1) still validates against lam[i]. Without this the
                            // cache would silently hand the sub-path the HERO's SPD values.
                            SpdCache sCache;
                            const SpdCache* sCp = nullptr;
                            if (spdCache && spdCache->lam && spdCache->spd && i < spdCache->C) {
                                sCache.lam = spdCache->lam + i;
                                sCache.spd = spdCache->spd + i;
                                sCache.C   = spdCache->C;
                                sCp = &sCache;
                            }
                            if (interactMaterial(scene, m, h, mats, sRay, sLam, sInv, sThr, sL,
                                                 sSpec, sPdf, sStk, rng, sCp, gi)) {
                                double sub[hero::kHeroMax];
                                radianceHeroLoop(scene, sRay, sStk, &sLam, &sInv, &sThr,
                                                 /*C=*/1, /*secAlive=*/false, sSpec, sPdf,
                                                 b + 1, sub, rng, sCp, gi);
                                sL += sub[0];
                            }
                            L[i] += sL;      // this wavelength's own estimate, own slot
                            thr[i] = 0.0;    // it is now that sub-path's business, not ours
                        }
                        secAlive = false;    // hero carries on alone, UNBOOSTED
                        if (!interactMaterial(scene, m, h, mats, ray, lam[0], invPdf[0], thr[0],
                                              L[0], specularArrival, contBsdfPdf, stk, rng,
                                              spdCache, gi)) { finish(); return; }
                        break;
                    }
                    // Default policy: terminate secondaries, then run the shared scalar
                    // interaction on the (boosted) hero channel.
                    deHero();
                    if (!interactMaterial(scene, m, h, mats, ray, lam[0], invPdf[0], thr[0], L[0],
                                          specularArrival, contBsdfPdf, stk, rng, spdCache, gi)) { finish(); return; }
                    break;
                }
                case MatType::Diffuse:
                default: {
                    double rho[hero::kHeroMax];
                    for (int i = 0; i < nUp; ++i)
                        rho[i] = clamp01(diffuseReflectance(scene, m, h, lam[i]));
                    neeLightHero(scene, h, rho, L, thr, lam, invPdf, nUp, rng, spdCache, gi);
                    if (scene.envIndex >= 0)
                        neeEnvHero(scene, h, rho, L, thr, lam, invPdf, nUp, rng);
                    // The Whitted indirect-diffuse term. With -gi this is a real
                    // single-bounce hemisphere gather (occlusion-aware and spectral); with
                    // -gi 0 it falls back to POV-Ray's flat `ambient` -- physically a lie,
                    // but without it a CLOSED room previews with black shadows, since
                    // every non-key-lit surface there is lit purely by bounce.
                    if (whitted) {
                        if (giUseGather(gi))
                            giGatherHero(scene, h, rho, L, thr, lam, invPdf, nUp, rng, spdCache, gi);
                        else if (ambient > 0.0)
                            for (int i = 0; i < nUp; ++i) L[i] += thr[i] * rho[i] * ambient;
                    }
                    if (directOnly) { finish(); return; }         // Whitted: no diffuse indirect
                    // Continuation RR over the WHOLE bundle: the survival probability is
                    // max_i rho_i, not the hero's own albedo, and every live λ reweights by
                    // rho_i/q <= 1. Rolling the coin on the hero alone (thr[i] *= rho_i/rho_0)
                    // amplifies a secondary by up to rho_max/rho_hero — on a saturated wall
                    // (redWall spans 0.05..0.75) that is a 15x weight spike, and the noise it
                    // injects grew once the bundle started surviving mirrors/gels. With
                    // nUp == 1, q == rho[0] and thr[0] *= 1.0 — the scalar code verbatim.
                    const double q = hero::maxOf(rho, nUp);
                    if (rng.uniform() >= q) { finish(); return; }         // RR absorb
                    for (int i = 0; i < nUp; ++i) thr[i] *= rho[i] / q;   // bounded reweight
                    Vec3 wOut = cosineHemisphere(h.n, rng);
                    contBsdfPdf = std::max(0.0, dot(wOut, h.n)) / PI;
                    ray = Ray{h.p + h.n * 1e-6, wOut};
                    specularArrival = false; break;
                }
            }
        }
        finish();
    }

    // Render `spp` samples per pixel into `film` (accumulates cieXYZ * radiance,
    // exactly like the forward film, so writeFilm with N=spp displays it). Renders
    // the pixel rows [y0, y1) — the caller partitions rows across threads. On a
    // pinhole camera in a scene without fog / GRIN, hero-wavelength sampling is used
    // (C wavelengths per camera path); otherwise the single-wavelength radiance().
    //
    // `sampleBase` is the ABSOLUTE index of the first sample this call renders (a
    // chunked/progressive render passes its running spp count; a resume passes the
    // checkpointed count). Each (pixel, absolute sample) pair seeds its own RNG
    // stream via seedUnit(), so the rendered realization is independent of the
    // chunk split, the row banding, and the thread count.
    void renderRows(const Scene& scene, const Camera& cam, Film& film,
                    int y0, int y1, long long spp, unsigned long long sampleBase) const {
        const int C = heroC;
        const bool useHero = (C > 1) && !scene.backwardMedium().enabled &&
                             !grin::sceneHasGrin(scene) && !cam.hasLens();
        const uint64_t nPix = (uint64_t)film.resX * (uint64_t)film.resY;
        // Per-sample emitter-SPD table (see SpdCache): nEm×C (nEm×1 on the scalar
        // path), allocated once per renderRows call (i.e. per thread) and refilled
        // for every sample.
        const int nEm = (int)scene.emitters.size();
        std::vector<double> spdBuf((size_t)nEm * (size_t)(useHero ? C : 1));
        for (int py = y0; py < y1; ++py) {
            for (int px = 0; px < film.resX; ++px) {
                const uint64_t pixIdx = (uint64_t)py * (uint64_t)film.resX + (uint64_t)px;
                for (long long s = 0; s < spp; ++s) {
                    const uint64_t sIdx = sampleBase + (uint64_t)s;   // absolute sample index
                    Pcg32 rng;
                    seedUnit(rng, sIdx * nPix + pixIdx, 0xD1B54A32D192ED03ULL);
                    if (useHero) {
                        // One stratified base draw → hero + C-1 secondary wavelengths,
                        // all from the emission CDF (hero.h policy 1). The hero (index 0)
                        // must have a valid pdf; dead secondaries (pdf 0) carry invPdf 0
                        // and splat 0.
                        double lamA[hero::kHeroMax], invA[hero::kHeroMax];
                        double pdfA[hero::kHeroMax];
                        // Whitted: the bundle's base coordinate comes off the progressive
                        // deterministic sequence instead of the rng, so the C wavelengths
                        // land on a FIXED lattice of the emission CDF. Without this the
                        // mode would still be noise-free in geometry and shading and yet
                        // visibly speckled in COLOUR, because λ was the last random draw.
                        double uLam = whitted ? whittedLambdaU(sIdx) : rng.uniform();
                        if (!hero::sampleBundle(scene.emitSampler, uLam, C,
                                                lamA, pdfA)) continue;
                        // Fill the per-sample SPD table, then derive invA from it by
                        // replicating Scene::invPdfLambda on the cached values (same
                        // emitter order, same zero guard — bit-identical). The NEE
                        // connections down the path then reuse the table instead of
                        // re-dispatching spdFn per emitter per bounce.
                        for (int e = 0; e < nEm; ++e) {
                            const Emitter& em = scene.emitters[e];
                            for (int i = 0; i < C; ++i)
                                spdBuf[(size_t)e * C + i] = em.spdFn(lamA[i]);
                        }
                        for (int i = 0; i < C; ++i) {
                            if (i > 0 && pdfA[i] <= 0.0) { invA[i] = 0.0; continue; }
                            double g = 0.0;
                            for (int e = 0; e < nEm; ++e)
                                g += scene.emitters[e].geomWeight() * spdBuf[(size_t)e * C + i];
                            invA[i] = (g > 0.0) ? scene.emitG / g : 0.0;
                        }
                        SpdCache spdCache{lamA, spdBuf.data(), C};
                        double jx, jy;
                        if (whitted) whittedSample(sIdx, jx, jy);
                        else { jx = rng.uniform(); jy = rng.uniform(); }
                        Ray ray = cam.genRay(px, py, jx, jy);
                        double Lh[hero::kHeroMax];
                        // GiCtx carries the ABSOLUTE sample index down to the gather, which
                        // rotates its direction lattice by it -- so, exactly like the
                        // subpixel and wavelength lattices, the gather is progressive and
                        // independent of how the budget was chunked.
                        radianceHero(scene, ray, lamA, invA, C, Lh, rng, &spdCache,
                                     GiCtx{0, sIdx});
                        for (int i = 0; i < C; ++i)
                            film.add(px, py, Vec3(cieX(lamA[i]), cieY(lamA[i]), cieZ(lamA[i]))
                                             * (Lh[i] / C));
                        continue;
                    }
                    // Sample lambda from the combined emission distribution g(lambda).
                    double pdf = 0.0;
                    // Whitted: stratified λ, as in the hero path above. Note this scalar
                    // path carries ONE wavelength per sample, so a deterministic spectral
                    // preview here needs spp raised to cover the spectrum (the hero path,
                    // which is the usual one, gets C=heroC of them per sample for free).
                    double lambda = whitted
                        ? scene.emitSampler.sampleAt(whittedLambdaU(sIdx), pdf)
                        : scene.emitSampler.sample(rng, pdf);
                    if (pdf <= 0) continue;
                    // Fill the per-sample SPD table (C=1) and derive invPdfLambda from
                    // it, replicating Scene::invPdfLambda on the cached values (same
                    // emitter order, same zero guard — bit-identical, = emitG/g(λ)).
                    for (int e = 0; e < nEm; ++e)
                        spdBuf[e] = scene.emitters[e].spdFn(lambda);
                    double gSum = 0.0;
                    for (int e = 0; e < nEm; ++e)
                        gSum += scene.emitters[e].geomWeight() * spdBuf[e];
                    double invPdfLambda = (gSum > 0.0) ? scene.emitG / gSum : 0.0;
                    SpdCache spdCache{&lambda, spdBuf.data(), 1};
                    if (cam.hasLens()) {
                        // Physical multi-element lens: trace the camera ray from the
                        // film out through the real glass interfaces at this wavelength
                        // (chromatic aberration + DoF + vignetting emerge). A vignetted
                        // ray contributes nothing; survivors carry a radiometric weight.
                        double jx = rng.uniform(), jy = rng.uniform();
                        double u1 = rng.uniform(), u2 = rng.uniform();
                        Ray ray; double wLens = 0.0;
                        if (!cam.genLensRay(px, py, jx, jy, u1, u2, lambda, ray, wLens))
                            continue;                       // clipped by an element / the stop
                        double L = radiance(scene, ray, lambda, invPdfLambda, rng, &spdCache,
                                            GiCtx{0, sIdx});
                        film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * (L * wLens));
                        continue;
                    }
                    double sjx, sjy;
                    if (whitted) whittedSample(sIdx, sjx, sjy);
                    else { sjx = rng.uniform(); sjy = rng.uniform(); }
                    Ray ray = cam.genRay(px, py, sjx, sjy);
                    double L = radiance(scene, ray, lambda, invPdfLambda, rng, &spdCache,
                                        GiCtx{0, sIdx});
                    film.add(px, py, Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * L);
                }
            }
        }
    }
};
