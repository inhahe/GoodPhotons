// Scene container: triangles, materials, one area light, one contact sensor.
#pragma once
#include <vector>
#include <algorithm>
#include <memory>
#include <climits>
#include "geometry.h"
#include "bvh.h"
#include "implicit.h"
#include "curve.h"       // curve / fiber primitive (hair, fur, grass, wire) — TODO §P1
#include "pattern.h"
#include "spectrum.h"
#include "scene_film.h"
#include "texture.h"
#include "envmap.h"
#include "vdbgrid.h"
#include "phase.h"       // hgPhase/sampleHG + rainbow::RainbowPhase (Medium phase dispatch)
#include "record.h"      // parametric records (§records): named per-channel LUTs

enum class MatType { Diffuse, Dielectric, Mirror, HalfMirror, Glossy, Fluorescent, ThinFilm, Grating, Mix, Multilayer, Layered, DiffuseTransmit, Filter };

// Materials whose last-vertex-before-camera cannot connect to the pinhole in
// model B (a delta or near-delta BSDF has ~zero connection pdf): the forward
// light tracer renders them BLACK from the camera (the SDS limitation). The
// camera-side ray path (mode P) is what fills these pixels in. Diffuse and
// Fluorescent connect in model B, so they are NOT specular-side.
inline bool isSpecularType(MatType t) {
    return t == MatType::Dielectric || t == MatType::Mirror ||
           t == MatType::HalfMirror || t == MatType::ThinFilm ||
           t == MatType::Glossy     || t == MatType::Grating ||
           t == MatType::Multilayer || t == MatType::Filter;
}

// The authored `type` keyword for a material, for diagnostics. Mix/Layered resolve to
// one of the others before they ever reach a BSDF switch, but can still be the type
// recorded on a hit, so they are named too.
inline const char* matTypeName(MatType t) {
    switch (t) {
        case MatType::Dielectric:      return "dielectric";
        case MatType::Mirror:          return "mirror";
        case MatType::HalfMirror:      return "half_mirror";
        case MatType::Glossy:          return "glossy";
        case MatType::Fluorescent:     return "fluorescent";
        case MatType::ThinFilm:        return "thin_film";
        case MatType::Grating:         return "grating";
        case MatType::Mix:             return "mix";
        case MatType::Multilayer:      return "multilayer";
        case MatType::Layered:         return "layered";
        case MatType::DiffuseTransmit: return "translucent";
        case MatType::Filter:          return "filter";
        case MatType::Diffuse:         default: return "diffuse";
    }
}

struct Material {
    MatType type = MatType::Diffuse;
    // reflect means: diffuse albedo / mirror tint / glossy tint / half-mirror
    // reflect-probability, depending on type. For Fluorescent it is the elastic
    // (wavelength-preserving) diffuse albedo.
    Spectrum reflect = constantSpectrum(0.5);
    Spectrum emit    = constantSpectrum(0.0); // emitted radiance vs lambda
    Spectrum ior     = iorConstant(1.5);      // dielectric index vs lambda
    // Nested-dielectric PRIORITY (Schmidt & Budge 2002). Where two dielectric solids
    // overlap in space, the one with the HIGHER priority "wins" that region: its medium
    // fills the overlap and the loser's surface there is a no-op (a coincident/interior
    // face is skipped). This resolves glass-in-water, coatings, and coincident surfaces
    // without a full per-path medium stack. INT_MIN = "unset" — used by the ahead-of-time
    // audit to warn when two overlapping dielectrics both lack an explicit priority (the
    // exterior IOR is then ambiguous). Only consulted for dielectric-like materials.
    int priority = INT_MIN;
    bool hasPriority() const { return priority != INT_MIN; }
    // Interior absorption coefficient sigma_a(lambda) in units of 1/metre, applied
    // Beer-Lambert along the path a photon travels INSIDE a dielectric (colored /
    // attenuating glass; also the `absorb` target a field_material can drive). 0 =
    // colorless (default, bit-identical to before). Only consulted for Dielectric.
    Spectrum absorb  = constantSpectrum(0.0);
    // Diffuse TRANSMISSION albedo vs lambda (MatType::DiffuseTransmit only). The
    // translucent material is a two-lobe Lambertian: `reflect` scatters cosine-
    // distributed into the FRONT hemisphere (+n), `transmit` into the BACK hemisphere
    // (-n). reflect+transmit must be <= 1 per wavelength (the rest is absorbed). Because
    // both lobes are non-specular, a directly-viewed translucent solid CONNECTS to the
    // pinhole and is visible in mode B (unlike clear dielectric, which stays black).
    // Also the per-wavelength TRANSMITTANCE T(lambda) in [0,1] of a MatType::Filter
    // (colored gel / Wratten): the photon passes straight through, surviving with
    // probability T(lambda) and absorbed otherwise — no scattering, no refraction.
    Spectrum transmit = constantSpectrum(0.0);
    double roughness = 0.1;                    // glossy lobe width [0,1]; on a Dielectric it
                                               // roughens the reflected+refracted lobes (frosted)
    bool isLight = false;
    // Spatially-varying diffuse albedo: index into Scene::textures (-1 = use the
    // constant `reflect` spectrum). When set, the reflectance at a hit is the
    // texture's per-texel Jakob-Hanika reflectance sampled at the surface (u,v).
    int reflectTex = -1;
    // Triplanar (box) projection: when > 0, a bound reflectTex is sampled by
    // world-space triplanar projection (three axis planes blended by the surface
    // normal) instead of the per-vertex (u,v) — the value is the world-to-texture
    // scale (repeats per world unit). Set by `uv triplanar [scale <s>]` on the
    // geometry block (spec §9.2). 0 => use the interpolated per-vertex UVs.
    double triplanarScale = 0.0;
    // Spatially-varying NON-albedo scalar parameters (spec §9.4): a bound texture's
    // grayscale value (Texture::scalarAt at the hit u,v) overrides the constant field
    // when >= 0. roughnessTex drives the glossy lobe width; filmThicknessTex drives
    // the thin-film coating thickness (nm) for spatially-varying iridescence
    // (peacock/beetle). -1 => use the constant `roughness` / `filmThickness`. Both use
    // the interpolated per-vertex UVs (no triplanar for scalar params yet).
    int roughnessTex = -1;
    int filmThicknessTex = -1;
    // Tangent-space NORMAL MAP (C6): index into Scene::textures, or -1. When set, the
    // shading normal at a hit is perturbed by the texel's tangent-space normal rotated
    // through the surface TBN frame (see Scene::closestHit). `normalStrength` scales the
    // tangential (x,y) perturbation — 0 disables, 1 is the authored map, >1 exaggerates.
    // The bound texture must be `encoding linear` (a normal map is raw vector data).
    int normalTex = -1;
    double normalStrength = 1.0;
    // Procedural (math-driven) scalar drives (§4): index into Scene::patterns, or -1.
    // A bound pattern is evaluated at the hit point (x,y,z,f,normal,r) and OVERRIDES
    // the constant/texture value — this is how implicit surfaces (which carry no UVs)
    // get spatially-varying roughness, film thickness, and A/B material selection.
    int roughnessPat = -1;
    int filmThicknessPat = -1;
    int mixWeightPat = -1;   // drives child-0 selection prob of a 2-child Mix (see mixResolveChild)
    // Procedural drive on the REFLECT slot: a per-hit MULTIPLIER on whatever the slot
    // otherwise evaluates to (a driven record, a bound reflectTex, or the constant
    // `reflect` spectrum), clamped to [0,1]. -1 = unmodulated. Two spellings collapse
    // here: `reflect pattern:<name>` (also what an inline literal `reflect [0 1](u)`
    // desugars to) puts the pattern in the slot ALONE, so the base is a flat 1.0 and the
    // albedo is greyscale straight from the pattern; `reflect_map pattern:<name>` written
    // beside a spectrum or texture modulates THAT, which is how a pattern gets a tint.
    // Because it is a scalar multiplier it is wavelength-flat by construction — colour
    // still comes from the spectrum/texture, never from the pattern.
    int reflectPat = -1;
    // Same idea on the TRANSMIT slot, read through transmitSlot(): a per-hit multiplier
    // on the constant `transmit` spectrum, clamped to [0,1]. `transmit pattern:<name>`
    // (or `transmit [0 1](u)`) puts the pattern in the slot alone over a flat-1.0 base;
    // `transmit_map pattern:<name>` beside a spectrum modulates that. On a Filter this
    // makes the gel's transmittance vary across its face; on a translucent (two-lobe
    // Lambertian) it varies the back-hemisphere albedo, still under the rhoR+rhoT <= 1
    // energy guard, which is applied AFTER the multiplier at every call site.
    int transmitPat = -1;
    // Same idea on the EMIT slot, read through emitSlot(): a per-point multiplier on the
    // emitted radiance, clamped to [0,1] — a gobo / stained-glass / video-wall profile on
    // an area light. `emit pattern:<name>` (or `emit [0 1](u)`) puts the pattern in the
    // slot alone over a flat-1.0 base; `emit_map pattern:<name>` beside an SPD modulates
    // that, so colour still comes from the spectrum.
    //
    // Emission is the one slot read from BOTH sides of the light transport — as
    // emission-on-hit (a path lands on the emissive surface, PatCtx from the Hit) and as
    // Le at a point drawn by the emitter sampler (NEE / BDPT / forward photon birth,
    // PatCtx from Emitter::samplePointUV). MIS combines those two estimators, so they
    // MUST agree pointwise or the image is biased, not just noisy. That is why the
    // pattern is only accepted where the emitter sampler's (u,v) provably equals the
    // geometry's hit (u,v): a rectangular area light (whose two tris now carry the same
    // corner UVs addQuad uses) and a mesh area light (whose EmitTri carries the source
    // triangle's UVs). Sphere / tube / spot / env emitters are rejected at load.
    //
    // The multiplier is deliberately NOT folded into Emitter::power, so no selection or
    // positional pdf changes anywhere: it is a pure post-multiplier on radiance and on a
    // born photon's beta, which keeps every estimator unbiased by construction. The cost
    // is variance — a mostly-dark pattern still gets sampled as if it were fully on.
    int emitPat = -1;

    // --- Parametric-record drive (§records) ---------------------------------
    // A material's slots can be driven by parametric records (Scene::records). Each
    // RecBinding fills ONE slot from a per-hit source (a record channel sampled at a
    // driver, a constant stop selector, or a direct scalar expression). The inline
    // `material R(driver)` form and a `material "m" { from R(d) … slot = … }` block's
    // ordered statements both collapse (last-write-wins) into at most one binding per
    // slot here. Empty => no record drive (the material uses its constant slots).
    std::vector<RecBinding> recBindings;
    bool hasRecordBinding() const { return !recBindings.empty(); }
    const RecBinding* recBindingFor(int slot) const {
        for (const auto& rb : recBindings) if (rb.slot == slot) return &rb;
        return nullptr;
    }

    // --- Thin-film / iridescence (MatType::ThinFilm) ------------------------
    // A thin dielectric coating of index filmIor and thickness filmThickness (in
    // nanometres) over a substrate whose index is `ior`. Interference between the
    // two coating interfaces yields an angle/wavelength-dependent reflectance
    // (structural colour). With a transparent (real-index) substrate, transport is
    // lossless specular reflect-or-refract, exactly like Dielectric. `substrateK` is
    // the substrate's extinction coefficient kappa (spectral); when non-zero the
    // substrate is absorbing/metallic (complex index n+i*kappa), giving OPAQUE
    // structural colour (oil-on-asphalt, anodised metal, heat-tempered steel): the
    // film reflects the interference fraction R and the transmitted rest is absorbed
    // (no refracted ray). Default 0 -> the exact lossless behaviour is preserved
    // bit-for-bit.
    double filmIor = 1.30;                      // coating refractive index n1
    double filmThickness = 300.0;              // coating thickness in nanometres
    Spectrum substrateK = constantSpectrum(0.0); // substrate extinction kappa (0 = transparent)

    // --- Multilayer thin-film stack (MatType::Multilayer) -------------------
    // An ordered stack of thin dielectric/absorbing layers between the incident
    // medium (air) and the substrate (`ior` + `substrateK`), evaluated with the
    // Abeles characteristic-matrix method (render.h multilayerReflectance). This is
    // the true model for Bragg-stack structural colour: beetle elytra, Morpho
    // wings, nacre, and dichroic/dielectric mirrors. layerN[j]/layerK[j] are the
    // (constant) real/imaginary index of layer j; layerThick[j] its thickness in
    // nanometres. Layer 0 is the outermost (nearest the incident medium).
    std::vector<double> layerN, layerK, layerThick;

    // --- Diffraction grating (MatType::Grating) -----------------------------
    // A reflective diffraction grating with groove period `grooveSpacing` (nm) and
    // grooves running along `grooveDir` (world, projected into the surface plane).
    // A photon of wavelength lambda is diffracted into one of the orders m in
    // [-gratingMaxOrder, gratingMaxOrder], chosen stochastically by an idealised
    // per-order efficiency; the outgoing direction obeys the EXACT vector grating
    // equation  v_t = u_t + m*(lambda/grooveSpacing)*t_hat  (t_hat perpendicular to
    // the grooves, in the surface). So the diffraction ANGLES are physically exact
    // and wavelength-dependent (the rainbow), while the split of energy across
    // orders is a model. m=0 is specular reflection, so with diffraction disabled
    // the grating is a plain mirror. `reflect` is the overall grating reflectivity.
    double grooveSpacing = 1000.0;             // groove period d in nanometres
    Vec3   grooveDir = {1.0, 0.0, 0.0};        // groove direction (world), projected to surface
    int    gratingMaxOrder = 3;                // highest |m| diffraction order considered

    // --- Fluorescence (MatType::Fluorescent) --------------------------------
    // A photon at lambda excites the dye with probability fluoAbsorb(lambda); the
    // dye then re-radiates (quantum yield fluoYield) at a Stokes-shifted lambda'
    // drawn from the normalized emission SPD fluoEmit. Single-wavelength forward
    // tracing handles this naturally: sample lambda' ~ fluoEmit and the M/pdf
    // ratio cancels, so the throughput weight is just the branch probability.
    Spectrum fluoAbsorb = constantSpectrum(0.0);  // excitation prob epsilon(lambda)
    Spectrum fluoEmit   = constantSpectrum(0.0);  // emission SPD M(lambda') (shape)
    EmissionSampler fluoEmitSampler;              // built from fluoEmit
    double fluoYield = 1.0;                        // quantum yield Q in [0,1]
    // Excitation-wavelength sampler for the BACKWARD tracer, built by
    // finalizeEmitters() from the product absorb(lambda) * g(lambda) where g is the
    // combined illuminant (see Scene::emitSampler). Backward transport has to pick
    // lambda_in *before* it knows anything about it, and drawing it from the
    // illuminant alone (what pre-0.115.0 did) wastes most samples on a narrow-band
    // dye: with `absorb shortpass 480` under a 6500 K illuminant across 360..830 nm
    // most draws land above the absorption edge, return aEff = 0 and contribute
    // literally nothing, while the few that do land in the band carry a large weight.
    // Sampling the product puts every draw inside the band -- an unbiased variance
    // reduction (`-checkfluoro` measures both estimators and their variances), and in
    // mode W it makes the ONE deterministic lambda_in the median of the *excitation*
    // band, the physically meaningful single excitation wavelength, so a narrow-band
    // dye is correct at -spp 1 instead of needing -spp 2 (v0.114.0) or -spp 64
    // (v0.113.x, before the radical inverses were digit-scrambled).
    // Empty (integral == 0) whenever the product vanishes or the material is not
    // fluorescent; callers then fall back to Scene::emitSampler.
    EmissionSampler fluoInSampler;

    // --- Stochastic mix (MatType::Mix) --------------------------------------
    // A probabilistic blend of other materials: a photon (or camera path) picks
    // child k with probability mixWeights[k], then behaves exactly as that child.
    // Weights are constants that must sum to <= 1; any leftover (1 - sum) is the
    // probability the photon is absorbed at the surface. This is the "same
    // machinery" the spec's `layered`/`mix` design calls for — per-photon lobe
    // selection — implemented by resolving the child BEFORE the material switch,
    // so every transport path (forward, backward, CUDA) shares one code path.
    // mixChildren holds indices into Scene::mats; a child may itself be any
    // non-Mix material (nested Mix is disallowed by the parser to keep resolve
    // single-step and the CDF bounded).
    std::vector<int>    mixChildren;               // indices into Scene::mats
    std::vector<double> mixWeights;                // selection probs, sum <= 1

    // Does any program bound to this material (or, for a `mix`, to one of its layers)
    // read the `cavity` variable? Set once at load by ftsl::setupCavity. This is the
    // per-hit gate on the cavity probe — the ONLY pattern input that spends rays — so
    // a patterned material that never says `cavity` fires none. See the long note on
    // setupCavity for why a scene-wide flag would not do.
    bool readsCavity = false;

    // --- Layered (MatType::Layered): a specular coat over a weighted body -------
    // Physical two-layer stack (spec §3.2). The COAT is a reflect-or-enter interface
    // that reuses roughness/roughnessTex for its glossiness (0 = mirror), ior for its
    // index, and filmIor/filmThickness[Tex] for a thin-film (Airy) coat. coatModel
    // picks the interface reflectance: 0 = Fresnel (from ior), 1 = thin-film Airy
    // (iridescent), 2 = manual constant coatSpecular. On entry the BODY selects one
    // lobe from mixChildren/mixWeights (the same unbiased selector as `mix`; leftover
    // 1-Sum absorbs), which then behaves exactly as that child material. The coat R and
    // the body weights partition each incident photon — energy-consistent by design.
    int    coatModel    = 0;      // 0 fresnel, 1 thinfilm, 2 manual(coatSpecular)
    double coatSpecular = -1.0;   // manual constant reflectance (used iff coatModel==2)
    // Optional per-hit blend mask (spec §9.4): a grayscale texture that drives the
    // selection weight of a 2-child mix. When set (and exactly 2 children), the map
    // value t at the hit is the probability of child 0 (child 1 gets 1-t, no leftover
    // absorption). -1 => use the constant mixWeights above. Resolved via scalarAt.
    int mixWeightTex = -1;
};

// Resolve a Mix material to one of its child material indices using a single
// uniform u in [0,1). Returns the chosen child index, or -1 if the photon falls
// in the leftover (1 - sum weights) absorption slice. Non-Mix materials never
// call this. Kept in the header so forward/backward transport share it verbatim.
inline int mixPickChild(const Material& m, double u) {
    double acc = 0.0;
    for (size_t k = 0; k < m.mixChildren.size(); ++k) {
        acc += m.mixWeights[k];
        if (u < acc) return m.mixChildren[k];
    }
    return -1;   // leftover slice -> absorbed
}

// Deterministic counterpart of mixPickChild for the Whitted preview (-mode W), which
// cannot flip a coin without reintroducing the per-pixel noise the mode exists to
// avoid: return the HEAVIEST child (the lobe that carries most of the material's
// response), or -1 if the leftover absorption slice outweighs every child. Ties go to
// the first child, so the choice is stable frame to frame.
inline int mixDominantChild(const Material& m) {
    int best = -1;
    double bestW = 0.0;
    for (size_t k = 0; k < m.mixChildren.size(); ++k) {
        if (m.mixWeights[k] > bestW) { bestW = m.mixWeights[k]; best = m.mixChildren[k]; }
    }
    double sum = 0.0;
    for (size_t k = 0; k < m.mixWeights.size(); ++k) sum += m.mixWeights[k];
    if (1.0 - sum > bestW) return -1;   // leftover absorbs more than any single lobe
    return best;
}

// A classic "green highlighter" fluorophore: absorbs blue/violet strongly, glows
// green (~560 nm). Shared by the fluoro demo scene and the -checkfluoro self-test
// so both exercise the exact same material definition (single source of truth).
inline Material makeFluoroMaterial() {
    Material f;
    f.type = MatType::Fluorescent;
    f.reflect     = constantSpectrum(0.05);         // small elastic base reflectance
    f.fluoAbsorb  = shortPass(480.0, 0.06, 0.85);   // excite below ~480 nm
    f.fluoEmit    = gaussianBand(560.0, 25.0, 1.0); // emit green-yellow
    f.fluoEmitSampler.build(f.fluoEmit, 1.0);
    f.fluoYield   = 0.9;
    return f;
}

// A homogeneous participating medium filling the whole scene (fog / haze). A
// photon travelling a distance travels freely until a collision sampled from
// exp(-sigma_t * t); at the collision it scatters (prob albedo = sigma_s/sigma_t,
// new direction from the Henyey-Greenstein phase function) or is absorbed. Beer-
// Lambert transmittance is captured implicitly by the free-flight sampling (analog
// Monte Carlo), so photon throughput stays unchanged — matching the rest of the
// renderer. Coefficients are spectral, so wavelength-dependent (e.g. Rayleigh
// ~1/lambda^4) fog that scatters blue and transmits red works for free.
// Shape of a medium's optional spatial bound: an axis-aligned box or a sphere. A
// sphere bound fills exactly an object-shaped region (e.g. "the whole inside of a
// glass sphere") — author the same center/radius as the sphere geometry.
// How a bounded medium decides which points it occupies. `Mesh` is true containment of
// an imported triangle mesh, carried as a baked occupancy lattice (see meshvoxel.h) —
// NOT the mesh's AABB, which is what a named mesh used to degrade to.
enum class MediumBound { Box, Sphere, Implicit, Mesh };

struct Medium {
    bool enabled = false;
    Spectrum sigma_a = constantSpectrum(0.0); // absorption coefficient vs lambda
    Spectrum sigma_s = constantSpectrum(0.0); // scattering coefficient vs lambda
    double g = 0.0;                            // HG anisotropy [-1,1] (0 = isotropic)

    // --- Scattering phase model ---------------------------------------------
    // By default a medium scatters via the smooth single-parameter Henyey-Greenstein
    // lobe above. A medium can instead opt into a physically-based WATER-DROPLET
    // phase (Airy theory, rainbow.h) via `phase rainbow { .. }` in FTSL, which adds
    // the wavelength-dependent rainbow fine structure (primary/secondary bows,
    // supernumeraries, fogbow limit). When `rainbowPhase` is set it OVERRIDES `g`.
    // The shared_ptr keeps Medium copies cheap (the table is a few MB) and is null
    // for the common HG case, so HG media stay bit-identical.
    std::shared_ptr<rainbow::RainbowPhase> rainbowPhase;
    bool rainbow() const { return (bool)rainbowPhase; }

    // Phase value p(cos) at wavelength lambda (nm) — equals the solid-angle pdf when
    // the scatter direction is importance-sampled from the phase (both models below).
    double phaseValue(double cosTheta, double lambda) const {
        if (rainbowPhase) return rainbowPhase->eval(cosTheta, lambda);
        return hgPhase(cosTheta, g);
    }
    // Importance-sample a scattered direction about propagation `wi` at wavelength
    // lambda; sets pdfOut to the solid-angle pdf p(cos) of the chosen direction.
    Vec3 phaseSample(const Vec3& wi, double lambda, Pcg32& rng, double& pdfOut) const {
        if (rainbowPhase) return rainbowPhase->sample(wi, lambda, rng, pdfOut);
        Vec3 d = sampleHG(wi, g, rng);
        pdfOut = hgPhase(dot(wi, d), g);
        return d;
    }

    // --- Optional heterogeneous density field (fuzzy / bounded fog) ----------
    // When `density` is non-empty, the base coefficients sigma_a/sigma_s are
    // MULTIPLIED by a dimensionless scalar field density(x,y,z) >= 0 evaluated per
    // point (a compiled pattern program over x y z r, §6.1 of FTSL.md). This shapes
    // the haze into blobs with soft, formula-defined boundaries. Empty => density
    // is 1 everywhere (the classic homogeneous medium; unchanged behaviour).
    std::vector<PatNode> density;
    double densityMax = 1.0;   // majorant: sup of density over `bmin..bmax` (delta/ratio tracking)

    // --- Optional gradient-index (GRIN) refractive field n(x,y,z) ------------
    // When `ior` is non-empty, this region is a GRADIENT-INDEX medium: light
    // rays do NOT travel straight through it — they bend continuously, obeying
    // the Eikonal ray equation d/ds(n · dr/ds) = ∇n. `ior` is a compiled pattern
    // program over world `x y z r` (same VM as `density`), giving the local
    // refractive index n(x,y,z) (≥ ~1). The tracer integrates the ray in small
    // steps of `iorStep` world units inside the region's bound (so a GRIN medium
    // needs a `bounds{}`), using central differences of n for ∇n. This produces
    // mirages, gradient lenses, hot-air shimmer, etc. A GRIN region may also be
    // absorbing/scattering, but the classic use is a clear bending field
    // (sigma_a = sigma_s = 0). The one canonical marcher lives in grin.h and is
    // shared by the CPU backward tracer (mode R), the CPU forward light tracer
    // (modes A/B/C) and the GPU forward megakernel/wavefront (dGrinMarch) — all
    // bend rays identically. BDPT (mode D) REFUSES GRIN scenes (its straight-line
    // connection geometry / MIS would be biased); use mode A/B/C or R instead.
    std::vector<PatNode> ior;   // compiled n(x,y,z) program; empty => not GRIN
    double iorStep = 0.0;       // Eikonal march step (world units); 0 => auto from bound

    // --- Optional imported .vdb/.nvdb sparse volume (baked to a dense grid) -----
    // When set, the density multiplier is TRILINEARLY sampled from a real NanoVDB
    // FloatGrid (`density vdb:"cloud.nvdb"`) instead of a formula. Shared so copies
    // of the Medium stay cheap. The grid's world AABB seeds the medium bound and
    // its peak value seeds densityMax. Takes precedence over the `density` formula.
    std::shared_ptr<VdbGrid> vdb;

    // --- Optional MESH containment lattice (boundShape == MediumBound::Mesh) ----
    // Occupancy (1 inside / 0 outside) baked from a named mesh's triangles by
    // meshvox::voxelizeSolid, sampled trilinearly so the 0.5 threshold lands on a
    // smooth interpolated isosurface rather than on voxel faces. Distinct from `vdb`
    // above on purpose: `vdb` REPLACES the density field, whereas this only decides
    // membership, so a `density` formula still multiplies on top and shapes the fog
    // WITHIN the mesh silhouette — the same semantics a named isosurface bound has.
    std::shared_ptr<VdbGrid> boundGrid;
    bool insideMesh(const Vec3& p) const {
        return boundGrid && boundGrid->sample(p) >= 0.5;
    }

    // --- Optional volumetric blackbody EMISSION (fire) ----------------------
    // When `temperature` is set (a second imported grid giving T in Kelvin per
    // voxel, e.g. the "temperature" grid of a multi-grid fire `.vdb`), the medium
    // EMITS thermal radiation: the local emission SOURCE radiance is
    //   L_e(x,λ) = emissionScale · blackbodyEmissionRadiance(T(x), λ)
    // added to the volume rendering equation as the emission term σ_a·L_e
    // (Kirchhoff's law: the same soot that absorbs also radiates), so hot dense
    // regions glow. This turns an imported fire `.vdb` into a self-illuminating
    // volumetric emitter. Null `temperature` => no emission (unchanged media).
    // Imported temperature grids typically store RELATIVE temperature in arbitrary
    // units (e.g. this OpenVDB fire sample peaks at ~46, not ~1500 K), so we map the
    // raw grid value to a physical Kelvin by peak-normalising: T(x) =
    // emitKelvin · grid(x)/tempPeak. `emitKelvin` (grammar `emission_kelvin`) is the
    // temperature of the HOTTEST voxel (default 1500 K — a yellow flame); cooler
    // voxels scale down linearly (redder + dimmer, Wien + Stefan-Boltzmann). This is
    // robust to whatever units the grid was authored in. `tempPeak` is the grid's raw
    // maxVal, captured at load.
    std::shared_ptr<VdbGrid> temperature;   // raw relative temperature grid (0 => cold)
    double tempPeak    = 1.0;               // raw grid peak (for peak-normalisation)
    double emitKelvin  = 1500.0;            // Kelvin of the hottest voxel
    double emissionScale = 1.0;             // brightness multiplier on the Planck term

    bool emissive() const { return (bool)temperature; }

    // Physical temperature (Kelvin) at a world point; 0 outside the grid / cold.
    double temperatureAt(const Vec3& p) const {
        if (!temperature) return 0.0;
        double raw = temperature->sample(p);
        if (raw <= 0.0) return 0.0;
        return emitKelvin * (raw / (tempPeak > 0.0 ? tempPeak : 1.0));
    }
    // Volumetric emission SOURCE radiance L_e(x,λ) (>= 0). This is the emitted
    // radiance BEFORE the σ_a weighting of the RTE emission term; callers that
    // want the full emission coefficient multiply by the local σ_a(x,λ).
    double emissionAt(const Vec3& p, double lambda) const {
        double T = temperatureAt(p);
        if (T <= 0.0) return 0.0;
        double Le = emissionScale * blackbodyEmissionRadiance(T, lambda);
        return Le > 0.0 ? Le : 0.0;
    }

    // --- Optional spatial bound (localized / per-object fog) ----------------
    // When `bounded`, the medium exists only inside a region: an axis-aligned box
    // [bmin,bmax] (`boundShape == Box`) or a sphere centered `bcenter` radius
    // `bradius` (`boundShape == Sphere`, e.g. the interior of a glass sphere). A
    // photon's fog interaction and connect-transmittance are clipped to the ray's
    // overlap with the region. Unbounded => the medium fills the whole scene. For a
    // sphere bound, bmin/bmax hold the sphere's AABB (used by the density majorant
    // grid estimate) so heterogeneous density fields work inside a sphere too.
    bool bounded = false;
    MediumBound boundShape = MediumBound::Box;
    Vec3 bmin{0, 0, 0}, bmax{0, 0, 0};
    Vec3 bcenter{0, 0, 0};
    double bradius = 0.0;

    // --- Optional implicit/isosurface bound (fog shaped by a named field) ------
    // When `boundShape == Implicit`, the medium fills the region inside a compiled
    // scalar field program: a point p is INSIDE when fieldEval(p) < 0 (if
    // boundInsideNeg) or > 0 (otherwise). bmin/bmax hold the field's AABB (for the
    // majorant grid and ray clipping). This lets fog take the exact shape of a
    // metaball / SDF isosurface authored elsewhere in the scene by name.
    std::vector<FieldNode> boundField;      // compiled field nodes (world-space)
    std::vector<PatNode>   boundFieldExpr;  // shared expression pool for the field
    bool boundInsideNeg = true;             // inside test: fieldEval < 0 (true) or > 0

    double sigmaT(double lambda) const {
        return std::max(0.0, sigma_a(lambda) + sigma_s(lambda));
    }
    double albedo(double lambda) const {       // single-scattering albedo sigma_s/sigma_t
        double s = std::max(0.0, sigma_s(lambda));
        double t = s + std::max(0.0, sigma_a(lambda));
        return t > 0.0 ? s / t : 0.0;
    }

    // Inside-test for an implicit-shaped bound: is world point p within the field?
    bool insideField(const Vec3& p, const PatTables* tabs = nullptr) const {
        double f = fieldEval(boundField.data(), (int)boundField.size(), p,
                             boundFieldExpr.data(), tabs);
        return boundInsideNeg ? (f < 0.0) : (f > 0.0);
    }

    // A medium is "heterogeneous" (needs delta/ratio tracking rather than an exact
    // analytic free-flight) when it has a density field OR an implicit bound, since
    // an implicit membership makes the effective density spatially varying (1 inside,
    // 0 outside) even when the base coefficients are constant.
    bool heterogeneous() const {
        return !density.empty() || vdb || boundShape == MediumBound::Implicit ||
               boundShape == MediumBound::Mesh;
    }

    // Dimensionless density multiplier at a world point (>= 0). 1 for a homogeneous
    // medium. Evaluated by the shared pattern VM (x y z r live; f/normal/uv read 0).
    // For an implicit bound the multiplier is 0 outside the field (the medium simply
    // does not exist there), so delta/ratio tracking carves out the exact iso-shape.
    // `tabs` publishes the scene's `grid:`/`scatter:` tables, so a density field can be
    // a SAMPLED volume (`density "grid:rho(x, y, z)"`) rather than only a formula. It is
    // a parameter, not a member, because it points into Scene's vectors — see PatTables.
    double densityAt(const Vec3& p, const PatTables* tabs = nullptr) const {
        if (boundShape == MediumBound::Implicit && !insideField(p, tabs)) return 0.0;
        if (boundShape == MediumBound::Mesh && !insideMesh(p)) return 0.0;
        return densityFieldAt(p, tabs);
    }

    // The density FIELD alone, with no membership carve — i.e. what `densityAt` would
    // return if every point were inside the bound. This exists for majorant estimation:
    // an implicit/mesh membership multiplies the field by 0 or 1, so the field's own
    // peak is always a valid (conservative) majorant, whereas sampling `densityAt` on a
    // coarse grid can miss a thin or low-volume-fraction shape entirely and majorise to
    // ~0, which would make delta/ratio tracking silently drop the medium.
    double densityFieldAt(const Vec3& p, const PatTables* tabs = nullptr) const {
        if (vdb) return vdb->sample(p);   // imported .nvdb volume (trilinear)
        if (density.empty()) return 1.0;
        PatCtx c = makePatCtx(p, 0.0, Vec3(0, 0, 0));
        patBindTables(c, tabs);
        double d = patternEval(density.data(), (int)density.size(), c);
        return d > 0.0 ? d : 0.0;
    }

    // --- Gradient-index (GRIN) helpers ---------------------------------------
    bool grin() const { return !ior.empty(); }

    // Local refractive index n at a world point (>= a small floor). 1 when this
    // is not a GRIN medium. Evaluated by the shared pattern VM (x y z r live).
    double nAt(const Vec3& p, const PatTables* tabs = nullptr) const {
        if (ior.empty()) return 1.0;
        PatCtx c = makePatCtx(p, 0.0, Vec3(0, 0, 0));
        patBindTables(c, tabs);
        double n = patternEval(ior.data(), (int)ior.size(), c);
        return n > 1e-3 ? n : 1e-3;
    }
    // ∇n at a world point via central differences with step h (world units).
    Vec3 gradNAt(const Vec3& p, double h, const PatTables* tabs = nullptr) const {
        double inv = 0.5 / h;
        double gx = nAt(p + Vec3(h, 0, 0), tabs) - nAt(p - Vec3(h, 0, 0), tabs);
        double gy = nAt(p + Vec3(0, h, 0), tabs) - nAt(p - Vec3(0, h, 0), tabs);
        double gz = nAt(p + Vec3(0, 0, h), tabs) - nAt(p - Vec3(0, 0, h), tabs);
        return Vec3(gx, gy, gz) * inv;
    }
    // Point-in-bound test (a GRIN region must be bounded). Mirrors clipToBounds'
    // membership: sphere chord / AABB / implicit field. Unbounded => everywhere.
    bool insideBound(const Vec3& p, const PatTables* tabs = nullptr) const {
        if (!bounded) return true;
        if (boundShape == MediumBound::Sphere) {
            Vec3 d = p - bcenter;
            return dot(d, d) <= bradius * bradius;
        }
        if (boundShape == MediumBound::Implicit) return insideField(p, tabs);
        if (boundShape == MediumBound::Mesh) return insideMesh(p);
        return p.x >= bmin.x && p.x <= bmax.x && p.y >= bmin.y &&
               p.y <= bmax.y && p.z >= bmin.z && p.z <= bmax.z;
    }

    // Clip a ray (o + t*d, t in [t0,t1]) to the bound, returning the sub-interval
    // [ta,tb] that lies inside the medium. Returns false if the ray misses the box.
    // Unbounded media pass the interval through unchanged.
    bool clipToBounds(const Vec3& o, const Vec3& d, double t0, double t1,
                      double& ta, double& tb) const {
        if (!bounded) { ta = t0; tb = t1; return t1 > t0; }
        if (boundShape == MediumBound::Sphere) {
            // Ray (o + t*d) ∩ sphere → the [ta,tb] chord inside the sphere, intersected
            // with [t0,t1]. Origins inside the sphere give a negative near root (clamped
            // to t0). No hit / chord outside [t0,t1] => the ray never enters the fog.
            Vec3 oc = o - bcenter;
            double A = dot(d, d);
            double B = 2.0 * dot(oc, d);
            double C = dot(oc, oc) - bradius * bradius;
            double disc = B * B - 4.0 * A * C;
            if (disc <= 0.0 || A <= 0.0) return false;
            double sd = std::sqrt(disc);
            double s0 = (-B - sd) / (2.0 * A), s1 = (-B + sd) / (2.0 * A);
            double lo = std::max(t0, s0), hi = std::min(t1, s1);
            if (lo > hi) return false;
            ta = lo; tb = hi; return tb > ta;
        }
        double lo = t0, hi = t1;
        for (int a = 0; a < 3; ++a) {
            double oa = (&o.x)[a], da = (&d.x)[a];
            double mn = (&bmin.x)[a], mx = (&bmax.x)[a];
            if (std::fabs(da) < 1e-12) { if (oa < mn || oa > mx) return false; continue; }
            double inv = 1.0 / da;
            double s0 = (mn - oa) * inv, s1 = (mx - oa) * inv;
            if (s0 > s1) std::swap(s0, s1);
            lo = std::max(lo, s0); hi = std::min(hi, s1);
            if (lo > hi) return false;
        }
        ta = lo; tb = hi; return tb > ta;
    }
};

// A flat rectangular contact sensor (model A) spanning origin + s*uAxis + t*vAxis.
struct Sensor {
    Vec3 origin, uAxis, vAxis; // uAxis/vAxis are full edge vectors
    Film film;
    void alloc() { film.alloc(); }
};

// Emitter surface shape. A Quad is the rectangle origin + s*u + t*v (s,t in
// [0,1]) with one-sided Lambertian emission along `normal`. A Sphere is a solid
// glowing ball of radius `radius` centred at `origin`, emitting Lambertian from
// every surface point about that point's outward normal (so exactly the
// hemisphere facing a receiver contributes — handled by the per-sample normal).
// A Spot is a point at `origin` radiating only into a cone about `beamDir`, with
// a smoothstep penumbra between the inner and outer half-angles (spotCosInner /
// spotCosOuter); it has no surface area, so its "geometric weight" is the
// falloff-weighted solid angle spotOmega instead of area*PI.
// An Env is an infinitely-distant environment: a constant radiance L_env(lambda)
// arriving from every direction. Its "geometric weight" is the emitted-power
// phase-space volume 4*PI^2*R^2 (R = scene bounding radius), so total power =
// emitIntegral*4*PI^2*R^2; forward photons are emitted from a disk of radius R on
// the bounding sphere and the backward tracer picks it up on ray misses.
// A Mesh emitter is an arbitrary emissive triangle soup sharing one SPD/material:
// samplePoint area-samples uniformly across all its triangles (pick a tri by a
// cumulative-area CDF, then barycentric point), so a glowing OBJ / tessellated shape
// acts as one area light. Its "geometric weight" is the same area*PI as a quad.
// A Sun emitter is a DISTANT DIRECTIONAL light: an infinitely-far disc of angular
// radius `theta` about `beamDir`, so every point of the scene sees it in the same
// direction and at the same radiance. It is the counterpart of Env for a *small*
// bright feature: forward emission fires PARALLEL photons across the scene's
// projected cross-section (every photon enters the scene — none are wasted aiming
// at a 6.8e-5 sr feature from a uniform sphere), and the backward/photon-map tracers
// next-event-estimate it inside its cone with pdf 1/Omega. That is what separates a
// ~1e5x-brighter-than-sky sun from the env importance sampler, which is why a daylight
// scene lit by `light sun` converges like any single-light scene while the same sun
// baked into an HDRI produces fireflies (see known-issues, K2 follow-up).
enum class EmitterShape { Quad, Sphere, Spot, Env, Cylinder, Mesh, Sun };

// Smoothstep spotlight falloff as a function of cos(angle-off-axis). 1 inside the
// inner cone, 0 outside the outer cone, cubic-smooth (3t^2-2t^3) in the penumbra.
inline double spotFalloff(double ct, double cosInner, double cosOuter) {
    if (ct >= cosInner) return 1.0;
    if (ct <= cosOuter) return 0.0;
    double t = (ct - cosOuter) / (cosInner - cosOuter);
    return t * t * (3.0 - 2.0 * t);
}

// A single emitter. Each carries its own SPD; `power` = emitIntegral * geomWeight
// is the emitter's total emitted power and doubles as the selection weight for the
// power-weighted CDF. The geometric weight is area*PI for area/sphere lights and
// the falloff-weighted solid angle spotOmega for a spot. For a collimated Quad
// every photon fires along `beamDir` from that quad (the prism demo).
// One triangle of a Mesh emitter, precomputed for uniform area sampling: v0 is a
// vertex, e1/e2 are the two edge vectors from it, nrm is the unit geometric normal,
// and cumArea is the running (inclusive) sum of triangle areas up to and including
// this one — so a binary search over cumArea picks a triangle in proportion to area.
struct EmitTri {
    Vec3 v0, e1, e2, nrm;
    double cumArea = 0.0;
    // The source triangle's texture coordinates, so a sampled point can report the SAME
    // (u,v) the ray-hit path interpolates (geometry.h: uv = b0*uv0 + b1*uv1 + b2*uv2).
    // Only read when an emission pattern is bound; stored as uv0 + the two uv edges to
    // mirror the v0/e1/e2 layout above.
    Vec3 uv0{0, 0, 0}, uvE1{1, 0, 0}, uvE2{1, 1, 0};
};

struct Emitter {
    Vec3 origin, u, v, normal;
    double area = 0.0;
    EmitterShape shape = EmitterShape::Quad;
    // Index into Scene::mats of the emissive material on this light's GEOMETRY
    // (area/sphere lights add both an emitter and a matching emissive surface), or
    // -1 for lights with no geometry (spot/env/collimated). BDPT needs this to map a
    // camera-ray hit on a light surface back to its emitter for the s=0 MIS term
    // (pdfLightOrigin = selection prob * 1/area). Set by the scene builders.
    int matId = -1;
    double radius = 0.0;      // sphere radius (Sphere); tube radius (Cylinder)
    // Cylinder (fluorescent-tube) light: `origin` is the base-cap center, `v` is the
    // axis vector (its length = the tube length), and `u`/`normal` are an orthonormal
    // radial basis; the lateral surface is sampled uniformly (area = 2*PI*radius*|v|).
    // When `caps` is set the two circular end discs also emit (a closed glowing
    // capsule): area = 2*PI*r*|v| + 2*PI*r^2 and samplePoint draws all three regions.
    bool caps = false;        // Cylinder: also emit from the two end-cap discs
    bool collimated = false;
    Vec3 beamDir{1, 0, 0};    // collimated fire direction / spot axis
    double spotCosInner = 1.0, spotCosOuter = 1.0; // spot penumbra cosines (Spot)
    double spotOmega = 0.0;   // spot falloff-weighted solid angle = PI*(2-ci-co)
    // Env: phase-space weight 4*PI^2*R^2. Sun: the same quantity for a cone light,
    // Omega*PI*R^2 (cone solid angle x the scene's projected disc). Both set in build()
    // because both depend on the scene bounding sphere.
    double envGeom = 0.0;
    // Sun only: the directly-viewed XYZ of this light, integral(CIE(lam)*L(lam) dlam).
    // Precomputed in build() because the direct-view path (a camera/specular ray that
    // escapes into the sun's cone) is evaluated per pixel and must not re-integrate.
    Vec3 viewXYZ{0, 0, 0};
    std::vector<EmitTri> meshTris; // Mesh: per-triangle area CDF for uniform sampling
    EmissionSampler spd;      // for forward per-emitter lambda importance sampling
    Spectrum spdFn = constantSpectrum(0.0); // raw SPD, for backward per-lambda eval
    double emitIntegral = 0.0;
    double power = 0.0;       // emitIntegral * geomWeight (selection weight)
    // Index into Scene::patterns of an emission profile over this emitter's surface,
    // copied from the emissive material's `emitPat` at registration; -1 = uniform.
    // Deliberately absent from `power` above — see Material::emitPat for why.
    int emitPat = -1;

    // Per-emitter spectral/geometric weight fed into the combined backward
    // wavelength sampler and the power law: area*PI for surfaces, spotOmega for a
    // spot. (Area/sphere keep the exact area*PI expression for bit-identity.)
    double geomWeight() const {
        if (shape == EmitterShape::Spot) return spotOmega;
        if (shape == EmitterShape::Env || shape == EmitterShape::Sun) return envGeom;
        return area * PI;
    }

    // Sample a surface point `y` and its outward unit normal `nOut` from two
    // uniforms. Quad: the bilinear point with the constant face normal (identical
    // draws to the pre-sphere engine, so quad scenes stay bit-identical). Sphere:
    // a uniformly-distributed surface point (pdf = 1/area for both shapes). Not
    // used for Spot (a point light — see the forward/backward spot paths).
    //
    // `uuOut`/`vvOut` optionally report the sampled point's TEXTURE coordinates, which
    // an emission pattern needs (emitterPatMul). They are filled only for the two shapes
    // that can carry one — Quad (the bilinear parameters, which addAreaLight's two tris
    // are UV'd to match) and Mesh (the chosen EmitTri's barycentric UV, the same
    // interpolation geometry.h does at a hit) — and left at 0 elsewhere, since sphere /
    // tube / spot / env emitters reject `emit pattern:` at load. Passing null (the
    // default) keeps every existing caller's arithmetic untouched.
    void samplePoint(double u1, double u2, Vec3& y, Vec3& nOut,
                     double* uuOut = nullptr, double* vvOut = nullptr) const {
        if (uuOut) *uuOut = 0.0;
        if (vvOut) *vvOut = 0.0;
        if (shape == EmitterShape::Sphere) {
            double z = 1.0 - 2.0 * u1;                 // cos(theta) uniform in [-1,1]
            double r = std::sqrt(std::max(0.0, 1.0 - z * z));
            double phi = 2.0 * PI * u2;
            Vec3 d{r * std::cos(phi), r * std::sin(phi), z};
            nOut = d;                                  // unit outward normal
            y = origin + d * radius;
        } else if (shape == EmitterShape::Cylinder) {
            // Uniform over the lateral surface: u1 slides along the axis (v), u2 picks
            // the angle around it. u/normal are the precomputed radial basis, so the
            // outward radial direction is rad = u*cos + normal*sin (a unit vector).
            double phi = 2.0 * PI * u2;
            Vec3 rad = u * std::cos(phi) + normal * std::sin(phi);
            if (caps) {
                // Closed capsule: pick lateral wall or one of the two end discs with
                // probability proportional to area, then reuse u1 (remapped to [0,1))
                // within the chosen region so the combined density is uniform over the
                // whole surface (pdf = 1/area still holds for the caller's 1/area law).
                double len = length(v);
                Vec3 a = (len > 0.0) ? v / len : Vec3{0, 1, 0};
                double latA = 2.0 * PI * radius * len;     // lateral wall
                double capA = PI * radius * radius;        // one end disc
                double total = latA + 2.0 * capA;
                double pLat = latA / total, pCap = capA / total;
                if (u1 < pLat) {                            // lateral wall
                    double uu = u1 / pLat;
                    y = origin + v * uu + rad * radius;
                    nOut = rad;
                } else if (u1 < pLat + pCap) {              // base cap (normal -a)
                    double rr = radius * std::sqrt((u1 - pLat) / pCap);
                    y = origin + rad * rr;
                    nOut = a * -1.0;
                } else {                                    // top cap (normal +a)
                    double rr = radius * std::sqrt((u1 - pLat - pCap) / pCap);
                    y = origin + v + rad * rr;
                    nOut = a;
                }
            } else {
                y = origin + v * u1 + rad * radius;
                nOut = rad;                            // unit outward normal
            }
        } else if (shape == EmitterShape::Mesh) {
            // Uniform over the whole triangle soup: pick a triangle with probability
            // proportional to its area (binary-search u1*area over the cumulative-area
            // CDF), remap the leftover to a fresh [0,1) uniform, then sample the chosen
            // triangle barycentrically. Combined density is 1/area over the surface, so
            // the caller's pdf = 1/area law holds exactly as for a quad.
            double target = u1 * area;
            // Lower-bound: first triangle whose inclusive cumArea >= target.
            size_t lo = 0, hi = meshTris.size();
            while (lo < hi) {
                size_t mid = (lo + hi) >> 1;
                if (meshTris[mid].cumArea < target) lo = mid + 1;
                else hi = mid;
            }
            if (lo >= meshTris.size()) lo = meshTris.size() - 1;
            const EmitTri& t = meshTris[lo];
            double prev = (lo == 0) ? 0.0 : meshTris[lo - 1].cumArea;
            double span = t.cumArea - prev;
            double uu = (span > 0.0) ? (target - prev) / span : u1; // remap within tri
            double su = std::sqrt(std::max(0.0, uu));
            double b1 = 1.0 - su;
            double b2 = u2 * su;
            y = t.v0 + t.e1 * b1 + t.e2 * b2;
            nOut = t.nrm;
            // Same barycentric weights the ray-hit path uses, so a bound emission
            // pattern reads identically from either side of the transport.
            if (uuOut) *uuOut = t.uv0.x + t.uvE1.x * b1 + t.uvE2.x * b2;
            if (vvOut) *vvOut = t.uv0.y + t.uvE1.y * b1 + t.uvE2.y * b2;
        } else {
            y = origin + u * u1 + v * u2;
            nOut = normal;
            if (uuOut) *uuOut = u1;
            if (vvOut) *vvOut = u2;
        }
    }

    // Sun: sample a direction uniformly inside the angular cone about `axis`
    // (solid-angle pdf 1/spotOmega, exact for any half-angle). Called with
    // axis = beamDir for forward emission (the direction a photon travels) and
    // axis = -beamDir for a next-event connection (the direction a shading point
    // looks toward the sun). A Sun stores spotCosInner == spotCosOuter == cos(theta),
    // which makes the shared spotOmega = PI*(2-ci-co) expression evaluate to exactly
    // the cone solid angle 2*PI*(1-cos theta) — so no extra field is needed.
    Vec3 sampleCone(const Vec3& axis, double u1, double u2) const {
        double ct = spotCosOuter + u1 * (1.0 - spotCosOuter);
        double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
        double phi = 2.0 * PI * u2;
        Vec3 t, b; onb(axis, t, b);
        return t * (st * std::cos(phi)) + b * (st * std::sin(phi)) + axis * ct;
    }
    // Sun: does the escaping ray direction `d` (unit) look back into the solar disc?
    bool inCone(const Vec3& d) const { return dot(d, beamDir) <= -spotCosOuter; }

    // Solid-angle (cone) importance sampling of a sphere emitter as seen from a
    // reference point `ref` (PBRT's Sphere::Sample_Li). Samples a direction `wi`
    // uniformly inside the cone the sphere subtends at `ref`, then finds the near
    // intersection point `y`/normal `nOut`; `pdfW` is the solid-angle-measure pdf.
    // Only the visible cap is sampled, so cosLight = dot(nOut,-wi) is always > 0 —
    // no draws are wasted on the far, self-occluded, back-facing hemisphere.
    // Returns false (and does not sample) when `ref` is inside the sphere, where the
    // subtended cone is the whole sphere; the caller then falls back to samplePoint.
    bool sampleSphereCone(const Vec3& ref, double u1, double u2,
                          Vec3& y, Vec3& nOut, Vec3& wi, double& dist,
                          double& pdfW) const {
        Vec3 toC = origin - ref;
        double dc2 = dot(toC, toC);
        double r2 = radius * radius;
        if (dc2 <= r2) return false;                   // ref inside sphere: use area sampling
        double dc = std::sqrt(dc2);
        Vec3 wc = toC / dc;                            // axis toward the sphere centre
        double sin2Max = r2 / dc2;
        double cosMax = std::sqrt(std::max(0.0, 1.0 - sin2Max));
        double cosT = 1.0 - u1 * (1.0 - cosMax);      // uniform cos in [cosMax, 1]
        double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
        double phi = 2.0 * PI * u2;
        Vec3 t, b; onb(wc, t, b);
        wi = t * (sinT * std::cos(phi)) + b * (sinT * std::sin(phi)) + wc * cosT;
        // Near intersection of the ray (ref, wi) with the sphere (guaranteed to hit).
        double tca = dot(toC, wi);
        double d2 = dc2 - tca * tca;
        double thc = std::sqrt(std::max(0.0, r2 - d2));
        dist = tca - thc;
        y = ref + wi * dist;
        nOut = (y - origin) / radius;
        pdfW = 1.0 / (2.0 * PI * (1.0 - cosMax));      // uniform over the cone
        return true;
    }

    // Area-measure importance sampling of a CYLINDER emitter's lateral surface as
    // seen from a reference point `ref`: draw only the front-facing (visible) part,
    // the analog of sampleSphereCone's visible cap. Because the lateral outward
    // normal N(phi) is perpendicular to the axis, the front-facing test
    //   dot(N, ref - Y) = rho*cos(phi) - r > 0   (rho = |ref's perpendicular offset|)
    // depends only on the azimuth phi, NOT the axial position z. So the visible
    // region is the simple strip z in [0,L], phi in (-phiMax, phiMax) with
    // cos(phiMax) = r/rho, of area 2*r*L*phiMax. We sample it uniformly in area:
    // z uniform along the axis, phi uniform in the visible arc, so every draw is
    // front-facing (no wasted back-side samples). `pdfArea` = 1/visibleArea.
    // Returns false when `ref` is within the tube radius (rho <= r), where the arc
    // is undefined; the caller then falls back to the uniform samplePoint().
    bool sampleCylinderVisible(const Vec3& ref, double u1, double u2,
                               Vec3& y, Vec3& nOut, double& pdfArea) const {
        double len = length(v);
        if (len <= 0.0) return false;
        Vec3 a = v / len;                              // axis unit
        Vec3 p = ref - origin;
        double pa = dot(p, a);
        Vec3 pPerp = p - a * pa;
        double rho = length(pPerp);
        if (rho <= radius) return false;               // ref inside tube radius
        Vec3 e1 = pPerp / rho;                          // toward ref's perpendicular projection
        Vec3 e2 = cross(a, e1);                         // completes the radial frame
        double phiMax = std::acos(std::min(1.0, radius / rho));
        double phi = (2.0 * u2 - 1.0) * phiMax;         // uniform in (-phiMax, phiMax)
        double z = u1 * len;                            // uniform along the axis
        Vec3 nrm = e1 * std::cos(phi) + e2 * std::sin(phi);  // outward radial (front-facing)
        y = origin + a * z + nrm * radius;
        nOut = nrm;
        double visibleArea = 2.0 * radius * len * phiMax;
        pdfArea = (visibleArea > 0.0) ? 1.0 / visibleArea : 0.0;
        return pdfArea > 0.0;
    }
};

// ---------------------------------------------------------------------------
// Instancing: a bottom-level acceleration structure (BLAS) is a mesh asset held
// ONCE in its own local (authored) space with its own BVH. A MeshInstance places
// that shared geometry into the world via an affine, WITHOUT baking a private
// copy of the triangles into Scene::tris — this is the memory win over group{}
// (which bakes every copy). The top-level BVH (Scene::bvh) carries one leaf box
// per instance; traversal transforms the ray into the BLAS's local space, walks
// the shared BLAS BVH, and transforms the hit back. The parametric ray distance
// `t` is PRESERVED by the affine ray transform because Affine::applyDir does NOT
// normalize the direction — so the local hit's t equals the world t directly and
// can be compared against the shared world tMax with no rescaling.
// ---------------------------------------------------------------------------
struct Blas {
    std::vector<Tri> tris;   // in local (authored) space
    Bvh bvh;                 // over the local tris
    Aabb localBounds;        // union of the local triangle boxes

    void build() {
        const double pad = 1e-6;
        std::vector<Aabb> boxes;
        boxes.reserve(tris.size());
        localBounds = Aabb{};
        for (auto& t : tris) {
            t.finalize();     // BLAS tris live outside Scene::tris, so finalize here
            Aabb b; b.expand(t.v0); b.expand(t.v1); b.expand(t.v2);
            b.lo = b.lo - Vec3{pad, pad, pad}; b.hi = b.hi + Vec3{pad, pad, pad};
            localBounds.expand(b);
            boxes.push_back(b);
        }
        bvh.build(boxes);
    }
    // Closest hit in local space. `h.t` carries the running (world==local) tMax on
    // entry; intersectTri only accepts a closer hit. Returns true if `h` was updated.
    bool intersectLocal(const Ray& lr, double tmin, Hit& h) const {
        bool found = false;
        double tMax = h.t;
        const TriShear sh = makeTriShear(lr.d);   // watertight shear: once per ray
        bvh.traverseClosest(lr, tmin, tMax, [&](int prim, double& tm) {
            if (intersectTri(sh, lr, tris[prim], tmin, h)) { tm = h.t; found = true; }
        });
        return found;
    }
    bool occludedLocal(const Ray& lr, double tmin, double maxDist) const {
        const TriShear sh = makeTriShear(lr.d);   // watertight shear: once per ray
        return bvh.traverseAny(lr, tmin, maxDist, [&](int prim) {
            Hit h; h.t = maxDist;
            return intersectTri(sh, lr, tris[prim], tmin, h);
        });
    }
};

struct MeshInstance {
    int blasId = -1;
    Affine toWorld = Affine::identity();   // local -> world
    Affine toLocal = Affine::identity();   // world -> local (= toWorld.inverse())
    int matOverride = -1;                  // >=0 replaces the BLAS triangles' matId
    // Derived from toWorld: the factor a LOCAL curvature (1/length) is multiplied by to
    // become a WORLD curvature (O3). Curvature is inverse-length, so it scales by
    // 1/|det(linear)|^(1/3) — the average linear scale. Cached rather than recomputed
    // per hit: instanceHitToWorld is on the shading hot path and this needs a cbrt.
    // Exact for a uniform scale; an approximation under a non-uniform one (see
    // instanceHitToWorld). 1.0 for any rigid transform.
    double curvScale = 1.0;
    // The ONLY way to set the transform, so the derived curvScale (and toLocal) can
    // never silently go stale behind a direct `inst.toWorld = …` assignment.
    void setToWorld(const Affine& xf) {
        toWorld = xf;
        toLocal = xf.inverse();
        const double* m = xf.m;
        double det = m[0] * (m[4] * m[8] - m[5] * m[7])
                   - m[1] * (m[3] * m[8] - m[5] * m[6])
                   + m[2] * (m[3] * m[7] - m[4] * m[6]);
        double a = std::fabs(det);
        double s = (a > 1e-30) ? std::cbrt(a) : 1.0;   // average linear scale
        curvScale = 1.0 / s;
    }
};

// A named mesh object as authored (one `mesh` or `mesh_asset` block), kept so tools
// like `-check-watertight` can report per-object (which the flattened Scene::tris /
// blasList otherwise lose). Either a contiguous run of world triangles in Scene::tris
// (blasId < 0) or a shared BLAS asset (blasId >= 0).
struct MeshGroup {
    std::string name;
    size_t triStart = 0, triCount = 0;   // range into Scene::tris  (blasId < 0)
    int    blasId   = -1;                // >=0: geometry lives in Scene::blasList[blasId]
    int    matId    = 0;                 // representative material (dielectric emphasis)
    // `mesh { shape_only yes }`: the triangles were loaded only to DEFINE a shape (a
    // `medium { bounds { object … } }` containment lattice) and were removed from
    // Scene::tris before the BVH was built, so this group renders nothing and its
    // triStart/triCount are both 0. The entry survives so the object still has a name
    // and reports can say what happened to it rather than silently omitting it.
    bool   shapeOnly = false;
};

struct Scene {
    std::vector<Tri> tris;
    std::vector<Sphere> spheres;
    std::vector<Implicit> implicits;   // isosurfaces / metaballs / (smooth) CSG
    // Curve / fiber primitives (hair, fur, grass, wire). `curves` is one entry per
    // authored strand and exists for diagnostics and future per-strand data; the thing
    // actually traced — and the thing the BVH indexes — is the flat `curveSegs` pool of
    // round cones each strand was flattened into at load time (see curve.h).
    std::vector<Curve>    curves;
    std::vector<CurveSeg> curveSegs;
    std::vector<Blas> blasList;        // shared instanced mesh assets (local space)
    std::vector<MeshInstance> instances; // placements of blasList into the world
    std::vector<MeshGroup> meshGroups;   // named mesh objects (for -check-watertight)
    std::vector<Material> mats;
    std::vector<Texture> textures;   // image textures referenced by materials (Phase 3b)
    std::vector<Pattern> patterns;   // procedural scalar fields for math-driven material props (§4)
    std::vector<Record>  records;    // parametric records: named per-channel LUTs (§records)
    // N-D data tables, sampled from a pattern expression: regular lattices as
    // `grid:<name>(c0, …)` and scattered samples as `scatter:<name>(c0, …)`. Headers
    // and numbers are split so ALL of them share ONE flat pool: a table refers to its
    // run by offset (never by pointer, which would dangle when the pool grows), and
    // that is also the exact shape the GPU uploads — one array however many tables.
    std::vector<PatGrid>    grids;
    std::vector<PatScatter> scatters;
    std::vector<float>      dataPool;
    // Non-owning view of the three vectors above, for evaluators that only forward the
    // tables onward (implicit fields, medium density/ior programs). Cheap enough to
    // build once per ray/traversal, which is the ONLY correct lifetime: a Scene is
    // copied and moved (buildCornell returns by value), so a stored PatTables would
    // dangle. Never cache it in a member — pass it as a parameter.
    PatTables patTables() const {
        PatTables t;
        t.grids     = grids.empty()    ? nullptr : grids.data();
        t.nGrids    = (int)grids.size();
        t.scatters  = scatters.empty() ? nullptr : scatters.data();
        t.nScatters = (int)scatters.size();
        t.dataPool  = dataPool.empty() ? nullptr : dataPool.data();
        t.dataPoolN = (int)dataPool.size();
        return t;
    }
    // ---- `cavity` probe settings (O3 stage 2) --------------------------------------
    // How far a cavity probe ray reaches. Cavity is a NON-LOCAL measure, so unlike
    // `curv` it has no intrinsic scale — "is this enclosed?" is only meaningful
    // relative to a distance, and the same corner reads deeply enclosed at 1 cm and
    // wide open at 1 m. Authored as `scene { cavity_radius <len> }`; when left at 0 the
    // loader derives 2% of the scene's AABB diagonal, which is scale-robust (the same
    // model reads the same whether it was authored in metres or millimetres) and beats
    // any fixed default, since scenes here range from fibres to rooms.
    double cavityRadius = 0.0;
    // Probe ray count. Deliberately a fixed, DETERMINISTIC direction set (see cavityAt),
    // not a Monte-Carlo estimate: a mask that changed sample to sample would inject its
    // own variance into every material it drives and smear under MIS.
    int cavitySamples = 16;
    // Load-time gate: true only if some bound pattern actually reads `cavity`. Every
    // probe is skipped otherwise, so a scene that does not use the feature pays exactly
    // nothing — which matters, because this is the one pattern input that costs rays.
    bool needsCavity = false;
    Sensor sensor;
    // Participating media. Zero or more independent regions (global haze, bounded
    // boxes/spheres, heterogeneous blobs) that may overlap. The forward tracer treats
    // them as superposed: extinction adds (sigma_t = sum over media containing the
    // point), so transmittance is the product of per-medium transmittances and a
    // collision is the earliest of the media's independent free-flight samples (with
    // the scattering medium chosen by the Poisson superposition theorem). Empty =>
    // vacuum. BDPT (mode D, both devices) and the GPU backward megakernel superpose the
    // full vector too; only the CPU backward tracer still degrades to backwardMedium().
    std::vector<Medium> media;

    // The CPU backward tracer (src/backward.h — modes R/W/V and the P composite's
    // camera-side layer) supports only a single GLOBAL HOMOGENEOUS haze and ignores
    // density/bounds. This returns the medium it uses as that haze — the first authored
    // medium — or a disabled default if there is none.
    //
    // NOTE this is now a CPU-only limitation, and a source of CPU/GPU divergence: the
    // device backward megakernel (render_cuda.cu dMediaSampleCollision / bkNeeVolume)
    // superposes the whole `media` vector, bounds + density fields + per-medium phase
    // functions included, so a GPU mode-R/W render of a multi-medium scene looks different
    // (and more correct) than the CPU one. main.cpp warns when a render's backward layer
    // actually lands on this degraded path. Tracked in known-issues.md; the fix is to port
    // the superposition into backward.h and delete this accessor.
    const Medium& backwardMedium() const {
        static const Medium none;   // disabled (enabled=false) sentinel
        return media.empty() ? none : media.front();
    }
    bool anyMedium() const { return !media.empty(); }

    // Emitters. Forward tracing selects one per photon with probability
    // proportional to power (so every photon carries beta = totalPower, keeping
    // the estimator unbiased); backward tracing samples wavelengths from the
    // combined emission distribution and sums NEE over all emitters.
    std::vector<Emitter> emitters;
    std::vector<double> emitterCdf;   // cumulative power, normalised to [0,1]
    double totalPower = 0.0;
    // Set true when at least one emitter authored an absolute flux (`power <watts>`
    // or `lumens <lm>`): the emitter SPDs are then scaled to real radiant power, so
    // the film's radiometric scale is physically meaningful and writeFilm uses a
    // fixed photographic exposure instead of the per-image auto-exposure anchor.
    bool absolute = false;
    // Combined emission wavelength sampler over g(lambda)=sum_k area_k*PI*SPD_k,
    // with emitG = its integral. invPdfLambda(lambda) = emitG / g(lambda) is the
    // per-lambda weight the backward reference needs (see backward.h).
    EmissionSampler emitSampler;
    double emitG = 0.0;

    // --- Volumetric blackbody emitters (fire) --------------------------------
    // Any medium carrying a `temperature` grid is ALSO a self-illuminating
    // isotropic volume emitter. Forward tracing treats each as a pseudo-emitter:
    // with probability totalEmissionPower/(totalPower+totalEmissionPower) a photon
    // is born inside the volume (uniform position in the grid's world AABB, uniform
    // wavelength, isotropic direction) carrying beta = grandTotal·κ_e(x,λ)/meanKe,
    // where κ_e = the medium's emissionAt(). The `power`/`meanKe` are estimated at
    // build() by Monte-Carlo sampling the emission field; `power` only tunes the
    // photon light-vs-fire split (it cancels in the physics), so the absolute fire
    // brightness is set purely by emissionAt (i.e. by `emission_scale`).
    struct EmissiveVolume {
        int    mediumIndex = -1;
        Vec3   bmin{0,0,0}, bmax{0,0,0};   // uniform-sampling AABB (temperature grid)
        double meanKe = 0.0;               // mean emissionAt over bbox×band (β normaliser)
        double power  = 0.0;               // 4π·V·meanKe·Δλ (selection weight)
        // Blackbody wavelength importance sampler (at the medium's peak temperature,
        // emitKelvin). Sampling λ ~ Planck(emitKelvin) instead of uniformly makes the
        // per-photon β nearly constant across λ — collapsing the spectral colour
        // speckle a uniform draw leaves in the hot core (variance-only; unbiased).
        EmissionSampler lamSampler;
    };
    std::vector<EmissiveVolume> emissiveVolumes;
    double totalEmissionPower = 0.0;

    // Estimate each emissive medium's mean emission and selection power. Called by
    // build() after finalizeEmitters(). Cheap Monte-Carlo over the grid AABB × band.
    void finalizeEmissiveVolumes() {
        emissiveVolumes.clear();
        totalEmissionPower = 0.0;
        const double lamMin = LAMBDA_MIN, lamMax = LAMBDA_MAX, dLam = lamMax - lamMin;
        for (size_t mi = 0; mi < media.size(); ++mi) {
            const Medium& m = media[mi];
            if (!m.emissive() || !m.temperature) continue;
            const VdbGrid& g = *m.temperature;
            Vec3 lo = g.wmin, hi = g.wmax;
            double V = (hi.x-lo.x) * (hi.y-lo.y) * (hi.z-lo.z);
            if (V <= 0.0) continue;
            // Stratified-ish MC of emissionAt over the AABB × spectral band.
            Pcg32 rng(0x1234abcdu ^ (uint32_t)mi, 0x9e3779b9u);
            const int NS = 20000;
            double sum = 0.0;
            for (int s = 0; s < NS; ++s) {
                Vec3 p{ lo.x + (hi.x-lo.x)*rng.uniform(),
                        lo.y + (hi.y-lo.y)*rng.uniform(),
                        lo.z + (hi.z-lo.z)*rng.uniform() };
                double lam = lamMin + dLam * rng.uniform();
                sum += m.emissionAt(p, lam);
            }
            double meanKe = sum / NS;
            if (meanKe <= 0.0) continue;   // grid is entirely cold → no emission
            EmissiveVolume ev;
            ev.mediumIndex = (int)mi;
            ev.bmin = lo; ev.bmax = hi;
            ev.meanKe = meanKe;
            ev.power = 4.0 * PI * V * meanKe * dLam;
            // Build the λ importance sampler from a representative blackbody at the
            // medium's peak temperature. Sampling λ ~ Planck(emitKelvin) makes the
            // per-photon β nearly constant across the band (collapses colour speckle).
            ev.lamSampler.build(blackbody(m.emitKelvin), 1.0);
            totalEmissionPower += ev.power;
            emissiveVolumes.push_back(ev);
        }
    }

    // Environment lighting. envIndex is the index into `emitters` of the single Env
    // emitter (or -1 if none). The scene bounding sphere (sceneCenter, sceneRadius,
    // from the BVH root) sizes forward env photon emission. `envMap` is non-null for
    // an image-based (lat-long) environment and null for a constant one; when present
    // it supplies the direction-dependent radiance/background/sampler. envXYZ is the
    // directly-viewed background colour for the CONSTANT case = integral of
    // L_env(lambda)*CIE(lambda) dlambda (the image case uses envMap->xyz(dir)).
    int envIndex = -1;
    Vec3 sceneCenter{0, 0, 0};
    double sceneRadius = 0.0;
    Vec3 envXYZ{0, 0, 0};
    std::shared_ptr<EnvMap> envMap;   // image-based env (null => constant env)
    // Number of EmitterShape::Sun emitters, recounted by finalizeEmitters(). Every
    // sun-aware hot path (ray miss, background pass) tests this first so a scene
    // without a sun pays one integer compare.
    int sunCount = 0;

    // Environment radiance from direction `d` at wavelength lambda (0 if no env).
    // Constant env ignores `d`; an image env samples the lat-long map.
    double envRadiance(const Vec3& d, double lambda) const {
        if (envIndex < 0) return 0.0;
        return envMap ? envMap->radiance(d, lambda) : emitters[envIndex].spdFn(lambda);
    }
    // Directly-viewed background XYZ in direction `d` (integral of CIE*L dlambda).
    Vec3 envXYZForDir(const Vec3& d) const {
        if (envIndex < 0) return Vec3{0, 0, 0};
        return envMap ? envMap->xyz(d) : envXYZ;
    }
    // Radiance of every distant sun whose disc contains direction `d` (0 when the ray
    // escapes into empty sky). Added ONLY on a camera / specular arrival: at a diffuse
    // or volume vertex the sun is covered by NEE (emitterGeom / neeVolume) and the
    // continuation ray must not count it a second time — the same single-estimator split
    // the Spot light already uses, which is why no MIS weight appears anywhere for a sun.
    // (MIS'ing the two would be a pure variance win over a 6.8e-5 sr target, not a
    // correctness fix; logged as a follow-up rather than done here.)
    double sunRadiance(const Vec3& d, double lambda) const {
        if (sunCount == 0) return 0.0;
        double L = 0.0;
        for (const auto& e : emitters)
            if (e.shape == EmitterShape::Sun && e.inCone(d)) L += e.spdFn(lambda);
        return L;
    }
    // Directly-viewed sun XYZ in direction `d`, for the forward tracers' background
    // pass (which composites colour, not per-wavelength radiance).
    Vec3 sunXYZForDir(const Vec3& d) const {
        Vec3 s{0, 0, 0};
        if (sunCount == 0) return s;
        for (const auto& e : emitters)
            if (e.shape == EmitterShape::Sun && e.inCone(d)) s += e.viewXYZ;
        return s;
    }
    // Reciprocal of the sampled-wavelength pdf-weighted mean env radiance shape used
    // by the forward emission reweight (== the env emitter's spdFn).
    double envAvgSpd(double lambda) const {
        return (envIndex >= 0) ? emitters[envIndex].spdFn(lambda) : 0.0;
    }

    // Register one area (or collimated) light. Terse helper for the C++ builders
    // and the FTSL loader; call finalizeEmitters() (via build()) afterwards.
    void addAreaLight(const Vec3& o, const Vec3& U, const Vec3& V, const Vec3& n,
                      double area, const Spectrum& spd, double stepNm,
                      bool collimated = false, const Vec3& beamDir = {1, 0, 0},
                      int matId = -1) {
        Emitter e;
        e.origin = o; e.u = U; e.v = V; e.normal = n; e.area = area;
        e.collimated = collimated; e.beamDir = beamDir; e.matId = matId;
        e.spd.build(spd, stepNm); e.spdFn = spd; e.emitIntegral = e.spd.integral;
        emitters.push_back(std::move(e));
    }

    // Register a spherical area light: a glowing ball of radius r at center c.
    // area = 4*PI*r^2 feeds the same power law (power = emitIntegral*area*PI) and
    // the same 1/area point-sampling pdf as a quad. u/v/normal are unused.
    void addSphereLight(const Vec3& c, double r, const Spectrum& spd, double stepNm,
                        int matId = -1) {
        Emitter e;
        e.origin = c; e.radius = r; e.area = 4.0 * PI * r * r;
        e.shape = EmitterShape::Sphere; e.matId = matId;
        e.spd.build(spd, stepNm); e.spdFn = spd; e.emitIntegral = e.spd.integral;
        emitters.push_back(std::move(e));
    }

    // Register a cylindrical area light: a glowing tube (fluorescent lamp) whose
    // LATERAL surface emits. `base` is the center of one end cap and `axis` points
    // to the other (|axis| = the tube length); `r` is the radius. area = 2*PI*r*|axis|
    // feeds the same power law (power = emitIntegral*area*PI) and the same 1/area
    // uniform-surface pdf as a quad. With `caps` the two end discs also emit (a closed
    // capsule): area += 2*PI*r^2, and samplePoint draws all three regions uniformly.
    // The default (caps=false) omits the caps from both the sampling area and the
    // emissive geometry the loader tessellates (a real fluorescent tube's ends are
    // non-emissive metal end-caps).
    void addCylinderLight(const Vec3& base, const Vec3& axis, double r,
                          const Spectrum& spd, double stepNm, int matId = -1,
                          bool caps = false) {
        Emitter e;
        double len = length(axis);
        Vec3 a = (len > 0.0) ? axis / len : Vec3{0, 1, 0};
        Vec3 t, b; onb(a, t, b);                 // orthonormal radial basis
        e.origin = base; e.v = axis; e.u = t; e.normal = b; e.radius = r;
        e.area = 2.0 * PI * r * len + (caps ? 2.0 * PI * r * r : 0.0);
        e.caps = caps;
        e.shape = EmitterShape::Cylinder; e.matId = matId;
        e.spd.build(spd, stepNm); e.spdFn = spd; e.emitIntegral = e.spd.integral;
        emitters.push_back(std::move(e));
    }

    // Register a mesh area light over the triangles Scene::tris[triStart, triStart+
    // triCount): the whole emissive triangle soup acts as one area light with a shared
    // SPD. Builds a cumulative-area CDF from the triangles (skipping any degenerate
    // zero-area ones) so samplePoint draws uniformly over the total surface; area = sum
    // of triangle areas feeds the same power law (power = emitIntegral*area*PI) and the
    // same 1/area point-sampling pdf as a quad. Call after the triangles are appended
    // to Scene::tris. If every triangle is degenerate, registers nothing.
    void addMeshLight(size_t triStart, size_t triCount, const Spectrum& spd,
                      double stepNm, int matId = -1) {
        Emitter e;
        e.shape = EmitterShape::Mesh; e.matId = matId;
        double total = 0.0;
        size_t end = triStart + triCount;
        if (end > tris.size()) end = tris.size();
        for (size_t i = triStart; i < end; ++i) {
            const Tri& t = tris[i];
            Vec3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0;
            Vec3 nc = cross(e1, e2);
            double a = 0.5 * length(nc);
            if (a <= 0.0) continue;               // skip degenerate triangles
            total += a;
            EmitTri et;
            et.v0 = t.v0; et.e1 = e1; et.e2 = e2;
            et.nrm = nc / (2.0 * a);              // == normalize(cross(e1,e2))
            et.cumArea = total;
            // Carry the source triangle's UVs as uv0 + edges, so a sampled point can
            // report the same (u,v) the ray-hit path interpolates — required for an
            // emission pattern to agree across NEE and emission-on-hit.
            et.uv0 = t.uv0; et.uvE1 = t.uv1 - t.uv0; et.uvE2 = t.uv2 - t.uv0;
            e.meshTris.push_back(et);
        }
        if (e.meshTris.empty() || total <= 0.0) return; // nothing emissive
        e.area = total;
        e.spd.build(spd, stepNm); e.spdFn = spd; e.emitIntegral = e.spd.integral;
        emitters.push_back(std::move(e));
    }

    // Map a hit surface's material index back to the emitter registered on that
    // geometry (or nullptr if none). Linear scan over the few emitters; used by the
    // BDPT s=0 MIS term when a camera subpath lands on a light surface directly.
    const Emitter* emitterForMat(int matId) const {
        if (matId < 0) return nullptr;
        for (const auto& e : emitters)
            if (e.matId == matId) return &e;
        return nullptr;
    }

    // Register a spotlight: a point at `pos` radiating into a cone about unit
    // `axis`, cubic-smooth falloff between the inner and outer half-angles.
    // geomWeight = spotOmega = PI*(2-cosInner-cosOuter) (the falloff-weighted solid
    // angle), so power = emitIntegral*spotOmega and peak intensity per unit SPD = 1.
    void addSpotLight(const Vec3& pos, const Vec3& axis, double cosInner,
                      double cosOuter, const Spectrum& spd, double stepNm) {
        Emitter e;
        e.origin = pos; e.beamDir = normalize(axis);
        e.shape = EmitterShape::Spot;
        e.spotCosInner = cosInner; e.spotCosOuter = cosOuter;
        e.spotOmega = PI * (2.0 - cosInner - cosOuter);
        e.spd.build(spd, stepNm); e.spdFn = spd; e.emitIntegral = e.spd.integral;
        emitters.push_back(std::move(e));
    }

    // Register a distant directional sun. `toSun` points FROM the scene TOWARD the sun
    // (the natural authoring convention, matching sky::SunDisk::dir and the sky block's
    // `sun_dir`); it is negated into `beamDir`, which everywhere else in the engine is
    // the direction light TRAVELS. `halfAngle` is the angular radius of the solar
    // disc in radians (the real sun is 0.00465 rad = 0.53 deg across). `irradiance`
    // is the spectrum of the irradiance falling on a surface FACING the sun, in the
    // same per-nm units every other emitter's `spd` uses.
    //
    // The stored `spdFn` is the sun's RADIANCE, irradiance/Omega, because every
    // consumer (NEE, the direct view, the forward reweight) wants radiance. Dividing
    // here rather than at each site is also what makes the light's brightness
    // INDEPENDENT of `halfAngle`: widening the disc to soften shadows spreads the same
    // irradiance over a larger cone instead of scaling the scene's exposure.
    // geomWeight (envGeom = Omega*PI*R^2) depends on the scene bounds, so build()
    // fills it in — exactly like the env light.
    void addSunLight(const Vec3& toSun, double halfAngle, const Spectrum& irradiance,
                     double stepNm) {
        Emitter e;
        e.shape = EmitterShape::Sun;
        e.beamDir = normalize(toSun) * -1.0;
        double ct = std::cos(halfAngle);
        e.spotCosInner = e.spotCosOuter = ct;
        e.spotOmega = PI * (2.0 - ct - ct);          // == 2*PI*(1-cos theta), the cone
        const double invOmega = (e.spotOmega > 0.0) ? 1.0 / e.spotOmega : 0.0;
        Spectrum rad = [irradiance, invOmega](double lambda) {
            return irradiance(lambda) * invOmega;
        };
        e.spd.build(rad, stepNm); e.spdFn = rad; e.emitIntegral = e.spd.integral;
        emitters.push_back(std::move(e));
    }

    // Register a constant environment light: uniform radiance `spd` arriving from
    // every direction (an infinitely-distant sphere). geomWeight (envGeom) and the
    // background colour (envXYZ) depend on the scene bounds, so they are filled in
    // by build() once the BVH exists. Only one env emitter is supported (the last
    // one registered wins envIndex).
    void addEnvLight(const Spectrum& spd, double stepNm) {
        Emitter e;
        e.shape = EmitterShape::Env;
        e.spd.build(spd, stepNm); e.spdFn = spd; e.emitIntegral = e.spd.integral;
        envIndex = (int)emitters.size();
        emitters.push_back(std::move(e));
    }

    // Register an image-based environment. The emitter's SPD is the map's mean
    // radiance spectrum (sin(theta)-weighted average), which drives the power
    // (emitIntegral*envGeom) and the wavelength importance CDF exactly like a
    // constant env; the per-direction radiance comes from `map` at trace time.
    void addEnvLight(std::shared_ptr<EnvMap> map, double stepNm) {
        Emitter e;
        e.shape = EmitterShape::Env;
        Spectrum mean = [map](double lambda) { return map->avgSpd(lambda); };
        e.spd.build(mean, stepNm); e.spdFn = mean; e.emitIntegral = e.spd.integral;
        envIndex = (int)emitters.size();
        envMap = std::move(map);
        emitters.push_back(std::move(e));
    }

    // Importance-sample an env emission/NEE direction. For an image env this draws
    // from the map's luminance CDF (pdfW = solid-angle density); for a constant env
    // it is a uniform sphere direction (pdfW = 1/4pi).
    Vec3 sampleEnvDir(Pcg32& rng, double& pdfW) const {
        if (envMap) return envMap->sample(rng.uniform(), rng.uniform(), pdfW);
        double u1 = rng.uniform(), u2 = rng.uniform();
        double z = 1.0 - 2.0 * u1;
        double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        double phi = 2.0 * PI * u2;
        pdfW = 1.0 / (4.0 * PI);
        return Vec3{r * std::cos(phi), r * std::sin(phi), z};
    }
    double envPdfDir(const Vec3& d) const {
        return envMap ? envMap->pdf(d) : 1.0 / (4.0 * PI);
    }

    // Compute per-emitter power, the selection CDF, and the combined backward
    // wavelength sampler. Idempotent; called by build().
    void finalizeEmitters(double stepNm = 1.0) {
        totalPower = 0.0;
        // Adopt each emitter's emission profile from the material on its geometry. Done
        // here — one place, after every registration path — rather than threading an
        // extra argument through addAreaLight / addMeshLight / the built-in scenes.
        // NOT folded into `power` below: the pattern is a pure post-multiplier on
        // radiance, so no selection or positional pdf anywhere has to change.
        for (auto& e : emitters)
            e.emitPat = (e.matId >= 0 && e.matId < (int)mats.size()) ? mats[e.matId].emitPat : -1;
        // Recount the distant suns here (not in addSunLight) so the flag survives every
        // path that rebuilds the emitter list, including applyIgnoreFlags' filtering.
        sunCount = 0;
        for (const auto& e : emitters) if (e.shape == EmitterShape::Sun) ++sunCount;
        emitterCdf.assign(emitters.size(), 0.0);
        for (size_t i = 0; i < emitters.size(); ++i) {
            // Area/sphere keep the exact emitIntegral*area*PI expression so those
            // scenes stay bit-identical; a spot uses its solid-angle weight.
            emitters[i].power =
                (emitters[i].shape == EmitterShape::Spot)
                    ? emitters[i].emitIntegral * emitters[i].spotOmega
                : (emitters[i].shape == EmitterShape::Env ||
                   emitters[i].shape == EmitterShape::Sun)
                    ? emitters[i].emitIntegral * emitters[i].envGeom
                : emitters[i].emitIntegral * emitters[i].area * PI;
            totalPower += emitters[i].power;
            emitterCdf[i] = totalPower;
        }
        if (totalPower > 0) for (auto& c : emitterCdf) c /= totalPower;
        // Combined g(lambda) = sum_k geomWeight_k*SPD_k(lambda); by value capture.
        std::vector<std::pair<double, Spectrum>> parts;
        for (const auto& e : emitters) parts.push_back({e.geomWeight(), e.spdFn});
        Spectrum g = [parts](double w) {
            double s = 0.0; for (const auto& p : parts) s += p.first * p.second(w); return s;
        };
        emitSampler.build(g, stepNm);
        emitG = emitSampler.integral;
        // Per-material excitation samplers: absorb(lambda) * g(lambda). Built here
        // rather than at parse time because it needs the finished illuminant, and
        // rebuilt on every finalizeEmitters() so -ignoreenv (which drops an emitter
        // and re-finalizes) keeps host and sampler consistent. See
        // Material::fluoInSampler for why the product and not g alone.
        for (auto& mm : mats) {
            if (mm.type != MatType::Fluorescent) continue;
            Spectrum prod = [&mm, g](double w) {
                double e = mm.fluoAbsorb(w);            // clamp01 lives in render.h
                e = (e < 0.0) ? 0.0 : (e > 1.0 ? 1.0 : e);
                return e * g(w);
            };
            mm.fluoInSampler.build(prod, stepNm);
        }
    }

    // Select an emitter index for the power-weighted CDF. For a single emitter
    // this consumes no randomness (index 0), preserving the RNG stream so
    // single-light scenes render bit-identically to the pre-multi-light engine.
    int selectEmitter(Pcg32& rng) const {
        if (emitters.size() <= 1) return 0;
        double u = rng.uniform();
        int lo = 0, hi = (int)emitterCdf.size() - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (emitterCdf[mid] < u) lo = mid + 1; else hi = mid; }
        return lo;
    }

    // Reference radiance for the Whitted preview's flat ambient term (backward.h).
    // The geomWeight-weighted mean emitter radiance, expressed in exactly the units
    // an NEE connection carries (spd(lambda) * invPdfLambda(lambda)): for a single
    // light that product is emitG/geomWeight identically, independent of lambda, so
    // this is a wavelength-flat "one light's worth of radiance".
    //
    // Why it exists: this renderer works in absolute spectral radiance, where a
    // plausible fill level can be 1e13, so an ambient given as a raw radiance would
    // be unusable and scene-specific. Scaling by this makes `-ambient 0.1` mean
    // "fill the scene uniformly with a tenth of a light's own radiance" in ANY
    // scene. A key light usually subtends well under a steradian as seen from the
    // surfaces it lights, so useful values live in roughly 0.01 .. 0.3.
    double ambientRef() const {
        double wSum = 0.0;
        for (const auto& e : emitters) wSum += e.geomWeight();
        return (wSum > 0.0) ? emitG / wSum : 0.0;
    }

    // Per-lambda weight for the backward reference: emitG / g(lambda), i.e. the
    // reciprocal of the sampled wavelength pdf. Reduces to a single light's
    // emitIntegral once multiplied by that light's SPD(lambda).
    double invPdfLambda(double lambda) const {
        // reconstruct g(lambda) = emitG * pdf; but we stored the sampler, so
        // recompute g directly from emitters (cheap: few evaluations).
        double g = 0.0;
        for (const auto& e : emitters) g += e.geomWeight() * e.spdFn(lambda);
        return (g > 0.0) ? emitG / g : 0.0;
    }

    Bvh bvh;   // acceleration structure over tris (0..nTris) then spheres.

    // Finalize triangle normals and build the BVH. Call after all geometry is
    // added. Primitive index i: i < tris.size() -> tris[i]; else spheres[i-nTris].
    void build() {
        for (auto& t : tris) t.finalize();
        for (auto& bl : blasList) bl.build();   // shared instanced assets (local space)
        buildBvh();
        // Scene bounding sphere from the BVH root AABB: center = box center, radius
        // = half the box diagonal (the box circumradius, guaranteed to enclose all
        // geometry). Sizes forward environment photon emission (disk radius) and
        // the env phase-space weight envGeom = 4*PI^2*R^2.
        if (!bvh.nodes.empty()) {
            const Aabb& b = bvh.nodes[0].box;
            sceneCenter = b.center();
            sceneRadius = length(b.hi - b.lo) * 0.5 * 1.0001; // tiny margin
        }
        // Distant suns are sized by the same bounding sphere: a photon is born on a
        // disc of radius R perpendicular to its (cone-sampled) travel direction, so the
        // phase-space weight is the cone solid angle times that disc's area.
        for (auto& e : emitters)
            if (e.shape == EmitterShape::Sun) {
                e.envGeom = e.spotOmega * PI * sceneRadius * sceneRadius;
                e.viewXYZ = Vec3{0, 0, 0};
                for (double lam = LAMBDA_MIN; lam <= LAMBDA_MAX; lam += 1.0)
                    e.viewXYZ += Vec3(cieX(lam), cieY(lam), cieZ(lam)) * e.spdFn(lam);
            }
        if (envIndex >= 0) {
            emitters[envIndex].envGeom = 4.0 * PI * PI * sceneRadius * sceneRadius;
            // Directly-viewed background colour: integral of L_env(lambda)*CIE dlambda.
            envXYZ = Vec3{0, 0, 0};
            for (double lam = LAMBDA_MIN; lam <= LAMBDA_MAX; lam += 1.0)
                envXYZ += Vec3(cieX(lam), cieY(lam), cieZ(lam))
                          * emitters[envIndex].spdFn(lam);
        }
        finalizeEmitters();
        finalizeEmissiveVolumes();
    }
    void finalizeTris() { build(); }   // kept for existing call sites

    // Scene-ignore flags (Stage 3): cheaply strip expensive features from an
    // already-built scene so the previewer / -explore can trade physical
    // completeness for speed, "like the rasterizer does". Call AFTER build().
    // Pure scene mutation (maxBounce / directOnly are render params, threaded
    // separately). Returns a human-readable summary of what was removed (empty
    // if nothing changed) so the caller can log it.
    //   noMedia  — drop all participating media (haze/volumes) -> vacuum.
    //   noEnv    — remove the environment light (sky/IBL); scene keeps its
    //              local emitters only. Rebuilds the emitter CDF / sampler.
    //   noFluoro — demote fluorescent materials to plain diffuse (their elastic
    //              `reflect` albedo), dropping the Stokes-shift reradiation.
    std::string applyIgnoreFlags(bool noMedia, bool noEnv, bool noFluoro) {
        std::string summary;
        auto note = [&](const std::string& s) {
            if (!summary.empty()) summary += ", ";
            summary += s;
        };
        if (noMedia && !media.empty()) {
            note(std::to_string(media.size()) + " medium/media");
            media.clear();
        }
        if (noEnv && envIndex >= 0) {
            note("environment light");
            emitters.erase(emitters.begin() + envIndex);
            envIndex = -1;
            envMap.reset();
            envXYZ = Vec3{0, 0, 0};
            finalizeEmitters();   // rebuild CDF / emission sampler without the env
        }
        if (noFluoro) {
            int n = 0;
            for (auto& m : mats)
                if (m.type == MatType::Fluorescent) { m.type = MatType::Diffuse; ++n; }
            if (n) note(std::to_string(n) + " fluorescent material" + (n > 1 ? "s" : ""));
        }
        return summary;
    }

    void buildBvh() {
        const double pad = 1e-6;       // avoid zero-thickness slabs on flat prims
        std::vector<Aabb> boxes;
        boxes.reserve(tris.size() + spheres.size());
        for (const auto& t : tris) {
            Aabb b; b.expand(t.v0); b.expand(t.v1); b.expand(t.v2);
            b.lo = b.lo - Vec3{pad, pad, pad}; b.hi = b.hi + Vec3{pad, pad, pad};
            boxes.push_back(b);
        }
        for (const auto& s : spheres) {
            Aabb b; b.expand(s.c - Vec3{s.r, s.r, s.r}); b.expand(s.c + Vec3{s.r, s.r, s.r});
            boxes.push_back(b);
        }
        boxes.reserve(boxes.size() + implicits.size() + curveSegs.size() + instances.size());
        for (const auto& im : implicits) boxes.push_back(im.bounds);
        // One leaf per round cone, NOT per strand: a whole hair's box is mostly empty,
        // and a BVH over long thin near-collinear boxes is exactly the degeneracy
        // TODO §P1 warned about. Per-segment bounds are also exact for what is tested.
        for (const auto& cs : curveSegs) boxes.push_back(curveSegBounds(cs));
        // One TLAS leaf per instance: the BLAS's local bounding box transformed into
        // world space (union of its 8 transformed corners — the tightest world AABB
        // of a rotated box short of re-bounding the actual triangles).
        for (const auto& inst : instances) {
            const Aabb& lb = blasList[inst.blasId].localBounds;
            Aabb wb;
            for (int c = 0; c < 8; ++c) {
                Vec3 corner{ (c & 1) ? lb.hi.x : lb.lo.x,
                             (c & 2) ? lb.hi.y : lb.lo.y,
                             (c & 4) ? lb.hi.z : lb.lo.z };
                wb.expand(inst.toWorld.apply(corner));
            }
            boxes.push_back(wb);
        }
        bvh.build(boxes);
    }

    // Transform a BLAS-local hit (from Blas::intersectLocal) back into world space for
    // instance `inst` under the world ray `r`. Positions map by toWorld; shading and
    // geometric normals map by the inverse-transpose (toWorld.applyNormal) and the
    // shading normal is re-oriented against the world ray (matching the primitive path).
    static void instanceHitToWorld(const MeshInstance& inst, const Ray& r, Hit& lh) {
        lh.p  = r.o + r.d * lh.t;                       // world t == local t (see Blas)
        Vec3 wn  = normalize(inst.toWorld.applyNormal(lh.n));
        Vec3 wng = normalize(inst.toWorld.applyNormal(lh.ng));
        lh.ng = wng;
        lh.n  = (dot(r.d, wn) < 0.0) ? wn : -wn;
        // Map the surface tangent through the instance transform too (C6 normal maps
        // on instanced meshes). The tangent is a direction, so it uses applyDir (not
        // applyNormal); handedness (bitangentSign) is preserved for proper transforms.
        Vec3 wt = inst.toWorld.applyDir(lh.tangent);
        double wtl = std::sqrt(dot(wt, wt));
        if (wtl > 1e-12) lh.tangent = wt * (1.0 / wtl);
        // Curvature is 1/LENGTH, so an instance that scales the mesh must scale it too:
        // blow a sphere up 10x and it gets ten times flatter. The factor is the average
        // linear scale |det(linear)|^(1/3), which is exact for a uniform scale (the case
        // that matters) and a reasonable mean under a mild non-uniform one — a genuinely
        // anisotropic scale changes the two principal curvatures by DIFFERENT amounts, so
        // no single scalar can be right and this is documented as an approximation.
        // Also re-flip if the world-space re-orientation flipped the shading normal.
        lh.curv *= inst.curvScale;
        if (dot(r.d, wn) >= 0.0) lh.curv = -lh.curv;
        if (inst.matOverride >= 0) lh.matId = inst.matOverride;
    }

    // Perturb the shading normal by a bound tangent-space normal map (C6). Builds a
    // TBN frame from the (ray-oriented) shading normal and the hit's surface tangent,
    // rotates the sampled tangent-space normal into world, and replaces h.n. A no-op
    // unless the hit's material carries a valid normalTex. Applied at the single
    // closestHit choke point so every CPU renderer (backward, forward, BDPT, VCM,
    // SPPM, photon map, GRIN) sees the perturbed normal identically.
    void applyNormalMap(Hit& h) const {
        if (!h.valid || h.matId < 0 || h.matId >= (int)mats.size()) return;
        const Material& m = mats[h.matId];
        if (m.normalTex < 0 || m.normalTex >= (int)textures.size()) return;
        const Texture& tx = textures[m.normalTex];
        if (!tx.valid()) return;
        Vec3 tn = tx.sampleNormalTS(h.u, h.v);          // tangent-space normal
        Vec3 N = h.n;                                   // oriented against the ray
        Vec3 T = h.tangent - N * dot(N, h.tangent);     // re-orthogonalize per-hit
        double tl = std::sqrt(dot(T, T));
        if (tl < 1e-9) return;                          // degenerate frame: leave n as-is
        T = T * (1.0 / tl);
        Vec3 B = cross(N, T) * h.bitangentSign;
        double s = m.normalStrength;
        Vec3 pert = T * (tn.x * s) + B * (tn.y * s) + N * tn.z;
        double pl = std::sqrt(dot(pert, pert));
        if (pl > 1e-12) h.n = pert * (1.0 / pl);
    }

    Hit closestHit(const Ray& r, double tmin = 1e-6, TraversalStats* stats = nullptr) const {
        Hit h;
        double tMax = DBL_MAX;
        const size_t nT = tris.size();
        const size_t nS = spheres.size();
        const size_t nI = implicits.size();
        const size_t nC = curveSegs.size();
        const TriShear sh = makeTriShear(r.d);   // watertight shear for world tris: once per ray
        // Unit ray direction + 1/|d| for the round-cone algebra: once per ray, not once
        // per segment (a fur render tests thousands of segments per ray). See curve.h.
        // Guarded on nC so a curve-free scene does not pay a sqrt on every single ray —
        // verified bit-identical against the pre-curve binary on the sample scenes.
        const CurveRay cray = nC ? makeCurveRay(r.d) : CurveRay{};
        // Sampled tables, in case an implicit's field formula reads `grid:`/`scatter:`.
        // Built once per ray, not per implicit hit: three pointer copies either way, and
        // the lambda is called many times.
        const PatTables tabs = patTables();
        bvh.traverseClosest(r, tmin, tMax, [&](int prim, double& tm) {
            if (prim < (int)nT)            { if (intersectTri(sh, r, tris[prim], tmin, h)) tm = h.t; }
            else if (prim < (int)(nT + nS)){ if (intersectSphere(r, spheres[prim - nT], tmin, h)) tm = h.t; }
            else if (prim < (int)(nT + nS + nI)) { if (intersectImplicit(r, implicits[prim - nT - nS], tmin, h, &tabs)) tm = h.t; }
            else if (prim < (int)(nT + nS + nI + nC)) { if (intersectCurveSeg(cray, r, curveSegs[prim - nT - nS - nI], tmin, h)) tm = h.t; }
            else {
                const MeshInstance& inst = instances[prim - nT - nS - nI - nC];
                Ray lr{inst.toLocal.apply(r.o), inst.toLocal.applyDir(r.d)};
                Hit lh; lh.t = h.t;                    // running world tMax == local tMax
                if (blasList[inst.blasId].intersectLocal(lr, tmin, lh)) {
                    instanceHitToWorld(inst, r, lh);
                    h = lh; tm = h.t;
                }
            }
        }, stats);
        applyNormalMap(h);
        return h;
    }

    // Is anything blocking the segment from o toward dir, before maxDist?
    // Used by model-B camera connections (shadow ray to the pinhole).
    // NOTE: dielectrics block connections (can't connect through specular) — the
    // SDS limitation. Glass therefore appears dark in model B; caustics it casts
    // onto diffuse surfaces still render, since those diffuse vertices connect.
    bool occluded(const Vec3& o, const Vec3& dir, double maxDist, double tmin = 1e-6) const {
        Ray r{o, dir};
        const size_t nT = tris.size();
        const size_t nS = spheres.size();
        const size_t nI = implicits.size();
        const size_t nC = curveSegs.size();
        const double seg = maxDist - tmin;
        const TriShear sh = makeTriShear(r.d);   // watertight shear for world tris: once per ray
        const CurveRay cray = nC ? makeCurveRay(r.d) : CurveRay{};   // see closestHit
        const PatTables tabs = patTables();      // see closestHit
        return bvh.traverseAny(r, tmin, seg, [&](int prim) {
            Hit h; h.t = seg;
            if (prim < (int)nT)             return intersectTri(sh, r, tris[prim], tmin, h);
            if (prim < (int)(nT + nS))      return intersectSphere(r, spheres[prim - nT], tmin, h);
            if (prim < (int)(nT + nS + nI)) return intersectImplicit(r, implicits[prim - nT - nS], tmin, h, &tabs, /*anyHit=*/true);
            if (prim < (int)(nT + nS + nI + nC))
                return intersectCurveSeg(cray, r, curveSegs[prim - nT - nS - nI], tmin, h, /*anyHit=*/true);
            const MeshInstance& inst = instances[prim - nT - nS - nI - nC];
            Ray lr{inst.toLocal.apply(r.o), inst.toLocal.applyDir(r.d)};
            return blasList[inst.blasId].occludedLocal(lr, tmin, seg);  // world seg == local seg
        });
    }

    // --- deterministic sampling helpers for emitterSeal ---------------------------
    // Van der Corput radical inverse in base `b`: a low-discrepancy 1D sequence.
    static double vdc(int i, int b) {
        double f = 1.0 / b, r = 0.0;
        for (int n = i; n > 0; n /= b) { r += f * (n % b); f /= b; }
        return r;
    }
    // Uniform direction in the cone of half-angle acos(cosMin) about `axis`, given a
    // cosine already drawn in [cosMin, 1] and an azimuth uniform `u`.
    static Vec3 coneDir(const Vec3& axis, double cz, double u) {
        const double sz = std::sqrt(std::max(0.0, 1.0 - cz * cz)), phi = 2.0 * PI * u;
        Vec3 t, b; onb(axis, t, b);
        return t * (sz * std::cos(phi)) + b * (sz * std::sin(phi)) + axis * cz;
    }
    // Uniform over the hemisphere about `n` (cz uniform in [0,1] is the solid-angle
    // -uniform draw; we want directions, not a cosine-weighted estimator).
    static Vec3 hemiDir(const Vec3& n, double u1, double u2) { return coneDir(n, u1, u2); }

    // What a light-seal probe found (see emitterSeal below).
    struct EmitterSeal {
        double sealed = 0.0;   // fraction of probed directions blocked by a specular surface
        int    probes = 0;     // directions that produced evidence (self-hits excluded)
        int    blockMat = -1;  // the material that blocked the most of them, or -1
    };

    // Diagnostic: is this emitter SEALED inside specular geometry?
    //
    // Next-event estimation is the only way a surface is lit in the Whitted preview
    // (mode W), and NEE is performed at exactly the material types for which
    // isSpecularType() is false — backward.h calls neeLight() from the Diffuse,
    // DiffuseTransmit and Fluorescent cases and nowhere else. A shadow ray is
    // furthermore blocked by ANY geometry, dielectrics very much included (see
    // occluded() above — "can't connect through specular", the SDS limitation).
    //
    // So when every direction leaving a light lands immediately on a specular surface
    // — an arc lamp sealed in its quartz envelope, a filament inside a mirrored
    // reflector — there is no vertex anywhere in the scene that this light can reach by
    // NEE, and mode W renders the whole thing pure black. Nothing is physically wrong
    // with the scene: it simply needs a transport that can refract *out* of the
    // enclosure (modes D/B/M), which is why such scenes select those modes.
    //
    // Deterministic (a stratified/van-der-Corput lattice, no rng), so the answer is
    // reproducible run to run. Hits on the emitter's own surface yield no evidence and
    // are excluded — a concave mesh light seeing itself still lights a room perfectly
    // well. Env/Sun emitters have no enclosure to speak of and report 0.
    EmitterSeal emitterSeal(const Emitter& e, int nSamples = 256) const {
        EmitterSeal out;
        if (e.shape == EmitterShape::Env || e.shape == EmitterShape::Sun) return out;
        std::vector<int> blockCount(mats.size(), 0);
        int blocked = 0;
        for (int i = 0; i < nSamples; ++i) {
            const double s1 = (i + 0.5) / nSamples;      // stratified along the surface
            const double s2 = vdc(i, 2), s3 = vdc(i, 3), s4 = vdc(i, 5);
            Vec3 y, nOut;
            if (e.shape == EmitterShape::Spot) { y = e.origin; nOut = e.beamDir; }
            else                                 e.samplePoint(s1, s2, y, nOut);
            Vec3 d;
            if (e.collimated) d = e.beamDir;
            else if (e.shape == EmitterShape::Spot) {
                // Uniform inside the spot's outer cone — outside it the light emits
                // nothing, so a blocker there is no evidence of a seal.
                const double cz = 1.0 - s3 * (1.0 - e.spotCosOuter);
                d = coneDir(nOut, cz, s4);
            } else d = hemiDir(nOut, s3, s4);            // uniform over the outgoing hemisphere
            const Hit h = closestHit(Ray{y + nOut * 1e-5, d});
            if (!h.valid) { ++out.probes; continue; }    // escapes into open space
            if (e.matId >= 0 && h.matId == e.matId) continue;   // the light's own surface
            ++out.probes;
            if (h.matId >= 0 && h.matId < (int)mats.size() &&
                isSpecularType(mats[h.matId].type)) { ++blocked; ++blockCount[h.matId]; }
        }
        if (out.probes > 0) out.sealed = (double)blocked / out.probes;
        int best = 0;
        for (size_t m = 0; m < blockCount.size(); ++m)
            if (blockCount[m] > best) { best = blockCount[m]; out.blockMat = (int)m; }
        return out;
    }

    // The authored name of the mesh object carrying material `matId`, or nullptr. Only
    // meshes record a name (MeshGroup), which is enough for the seal diagnostic: an
    // enclosure is a shell, and shells are modelled as meshes.
    const char* meshNameForMat(int matId) const {
        for (const auto& g : meshGroups) if (g.matId == matId) return g.name.c_str();
        return nullptr;
    }

    // Linear-scan reference (pre-BVH), kept for the -checkbvh self-test.
    Hit closestHitLinear(const Ray& r, double tmin = 1e-6) const {
        Hit h;
        const TriShear sh = makeTriShear(r.d);
        for (const auto& t : tris)     intersectTri(sh, r, t, tmin, h);
        for (const auto& s : spheres)  intersectSphere(r, s, tmin, h);
        const PatTables tabs = patTables();
        for (const auto& im : implicits) intersectImplicit(r, im, tmin, h, &tabs);
        // Curve segments are part of the reference too. Omitting them did not make
        // `-checkbvh` weaker, it made it WRONG: the BVH found every strand the linear
        // scan couldn't, so the cross-check reported thousands of "mismatches" on any
        // scene with fibers (curve_basics: 4390/2M) and a real BVH regression would
        // have been invisible in the noise.
        if (!curveSegs.empty()) {
            const CurveRay cray = makeCurveRay(r.d);
            for (const auto& cs : curveSegs) intersectCurveSeg(cray, r, cs, tmin, h);
        }
        for (const auto& inst : instances) {
            Ray lr{inst.toLocal.apply(r.o), inst.toLocal.applyDir(r.d)};
            Hit lh; lh.t = h.t;
            if (blasList[inst.blasId].intersectLocal(lr, tmin, lh)) {
                instanceHitToWorld(inst, r, lh);
                h = lh;
            }
        }
        applyNormalMap(h);
        return h;
    }
};

// PatOp::Tex sampler: the LINEAR grayscale value of one of the Scene's image
// textures (Texture::scalarAt — the same sampler roughness / film-thickness maps
// use, and the exact twin of the device's dTexScalarAt). Installed into a PatCtx by
// bindPatTex so pattern.h itself never has to know what a Texture is.
inline double scenePatTexSample(const void* self, int idx, double u, double v) {
    const Scene& s = *static_cast<const Scene*>(self);
    if (idx < 0 || idx >= (int)s.textures.size()) return 0.0;
    return s.textures[idx].scalarAt(u, v);
}
inline void bindPatTex(PatCtx& c, const Scene& s) {
    c.texFn = &scenePatTexSample;
    c.texSelf = &s;
}

// PatOp::Grid / PatOp::Scatter tables. Unlike textures these need no callback: the
// samplers live in pattern.h and read plain POD (headers + one flat float pool), which
// is exactly the layout the GPU uploads, so host and device share one code path.
//
// Two shapes of the same three pointers, because there are two kinds of caller:
//  - `Scene::patTables()` for code that only forwards the tables onward — the field /
//    isosurface / medium evaluators, which take a `const PatTables*` and build their
//    PatCtx internally.
//  - `bindPatData(c, scene)` for code holding a PatCtx it built itself.
inline void bindPatData(PatCtx& c, const Scene& s) {
    PatTables t = s.patTables();
    patBindTables(c, &t);
}

// Publish every scene-owned pattern table into a context. Call this (not the
// individual binders) wherever a PatCtx is built by hand, so a newly added table
// can't be silently missed at one site.
inline void bindPatScene(PatCtx& c, const Scene& s) {
    bindPatTex(c, s);
    bindPatData(c, s);
}

// ---------------------------------------------------------------------------
// `cavity` — the blocked fraction of a short hemispherical probe at a hit (O3 s2).
// ---------------------------------------------------------------------------
// Fires `scene.cavitySamples` short occlusion rays into the hemisphere around the
// shaded-side normal and returns the fraction blocked: 0 on a lone plane, ~0.5 in a
// right-angled interior corner, ->1 deep in a crevice.
//
// The direction set is a FIXED cosine-distributed Fibonacci spiral, not a random draw,
// and this is the load-bearing design decision rather than an optimisation. A pattern
// input is not a light-transport estimator: it is read many times per pixel by
// different tracers (a mix weight here, a roughness there, again on the light subpath),
// and every one of those reads must agree or the material itself becomes a source of
// variance that no amount of sampling averages away cleanly. A deterministic set makes
// `cavity` a true function of position — noise-free, identical on CPU and GPU, and
// stable frame to frame in an animation.
//
// The cost of determinism is BANDING: a fixed direction set can only produce
// cavitySamples+1 distinct values, so a smooth gradient becomes visible steps. That is
// why the count is authorable, and why the natural way to use `cavity` is through a
// `smoothstep` (which quantises anyway) or multiplied by a noise field (which hides the
// steps entirely) — exactly how the crevice-grime idiom already reads.
//
// Cosine weighting, not uniform: cavity stands in for how much ambient light reaches
// the point, and that is a cosine-weighted integral over the hemisphere. It also puts
// samples where the geometry actually occludes rather than wasting them near the
// grazing ring.
inline double cavityAt(const Scene& scene, const Hit& h) {
    const int N = scene.cavitySamples;
    if (N <= 0 || scene.cavityRadius <= 0.0) return 0.0;
    // Probe about the GEOMETRIC normal on the shaded side: using the shading normal
    // would let an interpolated normal tilt the hemisphere into the surface on a
    // coarse mesh and self-report occlusion that is not there.
    const Vec3 n = orientedGeoN(h);
    Vec3 t, b;
    onb(n, t, b);
    // Offset along the normal by a hair to avoid re-hitting the surface we sit on.
    const Vec3 o = h.p + n * 1e-6;
    const double R = scene.cavityRadius;
    // Golden-angle spiral: the standard low-discrepancy hemisphere set, and the reason
    // a mere 16 rays already look even rather than clumped.
    const double golden = 3.14159265358979323846 * (3.0 - std::sqrt(5.0));
    int blocked = 0;
    for (int i = 0; i < N; ++i) {
        // Cosine-weighted: sin(theta) = sqrt(u) with u stratified at bin centres.
        const double u  = (i + 0.5) / (double)N;
        const double sr = std::sqrt(u);          // radius in the projected disc
        const double cz = std::sqrt(1.0 - u);    // cos(theta) — the cosine weight
        const double ph = golden * i;
        const Vec3 d = t * (sr * std::cos(ph)) + b * (sr * std::sin(ph)) + n * cz;
        if (scene.occluded(o, d, R)) ++blocked;
    }
    return (double)blocked / (double)N;
}

// Build a procedural-pattern evaluation context from a hit: world point (x,y,z),
// implicit field value f (0 on non-implicit surfaces), oriented normal, radius, and
// the scene's pattern tables (so `tex:<name>(u,v)` and `grid:<name>(…)` sampling
// inside a pattern work).
inline PatCtx patCtxFromHit(const Scene& scene, const Hit& h) {
    // `cavity` is filled here rather than by the intersector because it needs the whole
    // scene, and cached on the Hit because one shading point builds several PatCtxs.
    // Doubly gated — scene-wide (one compare for the overwhelming majority of scenes,
    // which never mention `cavity`) and then per-material, so a noise-textured surface
    // sitting next to a cavity-driven one is not charged cavitySamples occlusion rays
    // for a variable its own pattern never reads.
    if (scene.needsCavity && !h.cavityDone &&
        h.matId >= 0 && h.matId < (int)scene.mats.size() &&
        scene.mats[h.matId].readsCavity) {
        h.cavity = cavityAt(scene, h);
        h.cavityDone = true;
    }
    PatCtx c = makePatCtx(h.p, h.fieldVal, h.n, h.u, h.v, h.curv, h.cavity);
    bindPatScene(c, scene);
    return c;
}

// Reflect-slot reflectance from a bound parametric record, if the material has a
// REC_SLOT_REFLECT binding. Returns true and sets `out` (constant-stop selector, or
// the driven sample at this hit); false when no record drives the reflect slot. The
// single point of truth so diffuse albedo AND specular tint (mirror/glossy/…) see
// identical record-driven reflectance.
inline bool recordReflectBound(const Scene& scene, const Material& m,
                               const Hit& h, double lambda, double& out) {
    const RecBinding* rb = m.recBindingFor(REC_SLOT_REFLECT);
    if (!rb || rb->recordIndex < 0 || rb->recordIndex >= (int)scene.records.size())
        return false;
    const Record& rec = scene.records[rb->recordIndex];
    const RecChannel& ch = rec.channels[rb->channel];
    if (rb->selStop >= 0 && rb->selStop < (int)ch.stops.size()) {
        out = ch.stops[rb->selStop].color(lambda);            // constant stop selector
    } else {
        double d = patternEval(rb->driver.data(), (int)rb->driver.size(),
                               patCtxFromHit(scene, h));
        out = recReflectanceAt(rec, ch, d, lambda);
    }
    return true;
}

// Per-hit multiplier from a scalar pattern bound to a SPECTRAL slot (`reflectPat` /
// `transmitPat`), clamped to [0,1] so a runaway formula can never manufacture energy.
// 1.0 when unbound, which is why the slot accessors can apply it unconditionally.
inline double slotPatMul(const Scene& scene, int pat, const Hit& h) {
    if (pat < 0 || pat >= (int)scene.patterns.size()) return 1.0;
    double p = scene.patterns[pat].eval(patCtxFromHit(scene, h));
    return p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p);
}

inline double reflectPatMul(const Scene& scene, const Material& m, const Hit& h) {
    return slotPatMul(scene, m.reflectPat, h);
}

// Reflect-slot reflectance for the SPECULAR families (Mirror / Glossy / Grating /
// HalfMirror) whose tint reads the reflect slot directly: a bound record if present,
// else the constant `reflect` spectrum — either way scaled by a bound reflect pattern.
// (These types never bind a reflect texture, so — unlike diffuseReflectance — there is
// no texture path.)
inline double reflectSlot(const Scene& scene, const Material& m,
                          const Hit& h, double lambda) {
    double v;
    if (!recordReflectBound(scene, m, h, lambda, v)) v = m.reflect(lambda);
    return m.reflectPat < 0 ? v : v * reflectPatMul(scene, m, h);
}

// Diffuse albedo at a hit: a bound parametric record (highest priority), else the
// material's spatially-varying texture reflectance if one is bound (Phase 3b), else
// its constant `reflect` spectrum — and then scaled by a bound reflect pattern, which
// is what makes `reflect [0 1](u)` a greyscale ramp (its base spectrum is a flat 1.0).
// Shared by the forward tracer and the backward reference so both see identical albedo.
inline double diffuseReflectance(const Scene& scene, const Material& m,
                                 const Hit& h, double lambda) {
    double rv;
    if (!recordReflectBound(scene, m, h, lambda, rv)) {
        if (m.reflectTex >= 0 && m.reflectTex < (int)scene.textures.size()) {
            const Texture& tx = scene.textures[m.reflectTex];
            rv = (m.triplanarScale > 0.0)
                     ? tx.reflectanceTriplanar(h.p, h.ng, m.triplanarScale, lambda)
                     : tx.reflectanceAt(h.u, h.v, lambda);
        } else {
            rv = m.reflect(lambda);
        }
    }
    return m.reflectPat < 0 ? rv : rv * reflectPatMul(scene, m, h);
}

// Transmit-slot value at a hit — the single point of truth for BOTH readings of the
// slot: a Filter's per-wavelength gel transmittance T(lambda), and a DiffuseTransmit's
// back-hemisphere Lambertian albedo rhoT. The constant `transmit` spectrum scaled by a
// bound transmit pattern (there is no record channel and no texture on this slot, so
// unlike diffuseReflectance there is only the one base path). Callers still clamp01 and,
// for the two-lobe case, still apply the rhoR+rhoT <= 1 energy guard afterwards.
inline double transmitSlot(const Scene& scene, const Material& m,
                           const Hit& h, double lambda) {
    double v = m.transmit(lambda);
    return m.transmitPat < 0 ? v : v * slotPatMul(scene, m.transmitPat, h);
}

// Emitted radiance at a hit ON an emissive surface, i.e. emission-on-hit: the material's
// `emit` SPD scaled by a bound emission pattern. The single point of truth for that half
// of the emission slot; the other half — Le at a point the emitter SAMPLER drew — goes
// through emitterSamplePoint() below, and the two are constructed to agree pointwise
// because MIS combines them (see Material::emitPat).
inline double emitSlot(const Scene& scene, const Material& m,
                       const Hit& h, double lambda) {
    double v = m.emit(lambda);
    return m.emitPat < 0 ? v : v * slotPatMul(scene, m.emitPat, h);
}

// The emission-pattern multiplier at a point on `em`, given the point's own texture
// coordinates. Clamped to [0,1] like the reflect/transmit slot patterns, so a runaway
// formula can never manufacture light.
inline double emitterPatMulAt(const Scene& scene, const Emitter& em,
                              const Vec3& y, const Vec3& nOut, double uu, double vv) {
    if (em.emitPat < 0 || em.emitPat >= (int)scene.patterns.size()) return 1.0;
    PatCtx c = makePatCtx(y, 0.0, nOut, uu, vv);
    bindPatScene(c, scene);
    double p = scene.patterns[em.emitPat].eval(c);
    return p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p);
}

// Draw a point on `em` and return the emission-pattern multiplier there, so a caller can
// write `Le = em.spdFn(lambda) * invPdfLambda * pmul`. 1.0 whenever no pattern is bound,
// which is every scene that does not use the feature — those keep bit-identical draws
// (samplePoint's uv outputs are the only extra work and they do not touch the RNG).
inline double emitterSamplePoint(const Scene& scene, const Emitter& em,
                                 double u1, double u2, Vec3& y, Vec3& nOut) {
    if (em.emitPat < 0) { em.samplePoint(u1, u2, y, nOut); return 1.0; }
    double uu = 0.0, vv = 0.0;
    em.samplePoint(u1, u2, y, nOut, &uu, &vv);
    return emitterPatMulAt(scene, em, y, nOut, uu, vv);
}

// Evaluate a bound scalar pattern at the hit (index checked). Returns the pattern
// value, or `dflt` if `pat` is out of range.
inline double patternScalarAt(const Scene& scene, int pat, const Hit& h, double dflt) {
    if (pat >= 0 && pat < (int)scene.patterns.size())
        return scene.patterns[pat].eval(patCtxFromHit(scene, h));
    return dflt;
}

// Per-hit glossy roughness: a bound roughness pattern (highest priority, for
// implicit surfaces) or roughnessTex's grayscale value at the hit, else the
// constant. Shared by every tracer so sampling and (in BDPT) the MIS pdf see the
// SAME roughness at a hit — otherwise the density and the sample diverge.
inline double materialRoughness(const Scene& scene, const Material& m, const Hit& h) {
    if (const RecBinding* rb = m.recBindingFor(REC_SLOT_ROUGHNESS)) {
        PatCtx c = patCtxFromHit(scene, h);
        double r = 0.0;
        if (rb->recordIndex < 0) {
            // direct scalar expression, e.g. `roughness = sin(v*3.14159)`
            r = patternEval(rb->driver.data(), (int)rb->driver.size(), c);
        } else if (rb->recordIndex < (int)scene.records.size()) {
            const Record& rec = scene.records[rb->recordIndex];
            const RecChannel& ch = rec.channels[rb->channel];
            if (rb->selStop >= 0 && rb->selStop < (int)ch.stops.size()) {
                const std::vector<PatNode>& e = ch.stops[rb->selStop].expr;   // constant stop selector
                r = e.empty() ? 0.0 : patternEval(e.data(), (int)e.size(), c);
            } else {
                double d = patternEval(rb->driver.data(), (int)rb->driver.size(), c);
                r = recSampleScalar(rec, ch, d, c);
            }
        }
        return r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
    }
    if (m.roughnessPat >= 0 && m.roughnessPat < (int)scene.patterns.size()) {
        double r = scene.patterns[m.roughnessPat].eval(patCtxFromHit(scene, h));
        return r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
    }
    if (m.roughnessTex >= 0 && m.roughnessTex < (int)scene.textures.size())
        return scene.textures[m.roughnessTex].scalarAt(h.u, h.v);
    return m.roughness;
}

// Per-hit thin-film coating thickness (nm): a bound thickness pattern or
// filmThicknessTex's grayscale value (both scaled to nm by the constant
// `filmThickness`, so the map is a 0..1 profile of the authored thickness) at the
// hit, else the constant thickness. Spatially varies §3.2 iridescence.
inline double materialFilmThickness(const Scene& scene, const Material& m, const Hit& h) {
    if (m.filmThicknessPat >= 0 && m.filmThicknessPat < (int)scene.patterns.size())
        return scene.patterns[m.filmThicknessPat].eval(patCtxFromHit(scene, h)) * m.filmThickness;
    if (m.filmThicknessTex >= 0 && m.filmThicknessTex < (int)scene.textures.size())
        return scene.textures[m.filmThicknessTex].scalarAt(h.u, h.v) * m.filmThickness;
    return m.filmThickness;
}

// Resolve a stochastic Mix to a child index, honouring an optional per-hit blend
// mask. With a bound mixWeightPat or mixWeightTex (2 children), the value t at the
// hit is the probability of child 0 (child 1 = 1-t, no absorption) — a spatial A/B
// selection that lets colour AND material type vary across an implicit surface.
// Otherwise this is the constant-weight CDF pick (mixPickChild), with the leftover
// (1 - sum) slice absorbed. Mix weight is a stochastic (RR-style) selection that does
// not enter the BSDF pdf, so a per-hit weight stays unbiased in every tracer.
inline int mixResolveChild(const Scene& scene, const Material& m, const Hit& h, double u) {
    if (m.mixChildren.size() == 2 &&
        (m.mixWeightPat >= 0 || m.mixWeightTex >= 0)) {
        double t;
        if (m.mixWeightPat >= 0 && m.mixWeightPat < (int)scene.patterns.size())
            t = scene.patterns[m.mixWeightPat].eval(patCtxFromHit(scene, h));
        else
            t = scene.textures[m.mixWeightTex].scalarAt(h.u, h.v);
        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
        return (u < t) ? m.mixChildren[0] : m.mixChildren[1];
    }
    return mixPickChild(m, u);
}

// Deterministic mixResolveChild for the Whitted preview (-mode W). A pattern/texture
// driven two-way mix picks whichever child dominates AT THIS POINT, so the blend
// becomes a hard threshold at t == 0.5 rather than a stochastic dither: the preview
// shows a crisp boundary where the render shows a smooth gradient. That is the honest
// cost of one deterministic sample per pixel, and it stays put frame to frame.
inline int mixResolveDominant(const Scene& scene, const Material& m, const Hit& h) {
    if (m.mixChildren.size() == 2 &&
        (m.mixWeightPat >= 0 || m.mixWeightTex >= 0)) {
        double t;
        if (m.mixWeightPat >= 0 && m.mixWeightPat < (int)scene.patterns.size())
            t = scene.patterns[m.mixWeightPat].eval(patCtxFromHit(scene, h));
        else
            t = scene.textures[m.mixWeightTex].scalarAt(h.u, h.v);
        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
        return (t >= 0.5) ? m.mixChildren[0] : m.mixChildren[1];
    }
    return mixDominantChild(m);
}
