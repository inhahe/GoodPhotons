// Scene container: triangles, materials, one area light, one contact sensor.
#pragma once
#include <vector>
#include <algorithm>
#include <memory>
#include "geometry.h"
#include "bvh.h"
#include "implicit.h"
#include "pattern.h"
#include "spectrum.h"
#include "scene_film.h"
#include "texture.h"
#include "envmap.h"

enum class MatType { Diffuse, Dielectric, Mirror, HalfMirror, Glossy, Fluorescent, ThinFilm, Grating, Mix, Multilayer, Layered, DiffuseTransmit };

// Materials whose last-vertex-before-camera cannot connect to the pinhole in
// model B (a delta or near-delta BSDF has ~zero connection pdf): the forward
// light tracer renders them BLACK from the camera (the SDS limitation). The
// camera-side ray path (mode P) is what fills these pixels in. Diffuse and
// Fluorescent connect in model B, so they are NOT specular-side.
inline bool isSpecularType(MatType t) {
    return t == MatType::Dielectric || t == MatType::Mirror ||
           t == MatType::HalfMirror || t == MatType::ThinFilm ||
           t == MatType::Glossy     || t == MatType::Grating ||
           t == MatType::Multilayer;
}

struct Material {
    MatType type = MatType::Diffuse;
    // reflect means: diffuse albedo / mirror tint / glossy tint / half-mirror
    // reflect-probability, depending on type. For Fluorescent it is the elastic
    // (wavelength-preserving) diffuse albedo.
    Spectrum reflect = constantSpectrum(0.5);
    Spectrum emit    = constantSpectrum(0.0); // emitted radiance vs lambda
    Spectrum ior     = iorConstant(1.5);      // dielectric index vs lambda
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
    // Procedural (math-driven) scalar drives (§4): index into Scene::patterns, or -1.
    // A bound pattern is evaluated at the hit point (x,y,z,f,normal,r) and OVERRIDES
    // the constant/texture value — this is how implicit surfaces (which carry no UVs)
    // get spatially-varying roughness, film thickness, and A/B material selection.
    int roughnessPat = -1;
    int filmThicknessPat = -1;
    int mixWeightPat = -1;   // drives child-0 selection prob of a 2-child Mix (see mixResolveChild)

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
enum class MediumBound { Box, Sphere, Implicit };

struct Medium {
    bool enabled = false;
    Spectrum sigma_a = constantSpectrum(0.0); // absorption coefficient vs lambda
    Spectrum sigma_s = constantSpectrum(0.0); // scattering coefficient vs lambda
    double g = 0.0;                            // HG anisotropy [-1,1] (0 = isotropic)

    // --- Optional heterogeneous density field (fuzzy / bounded fog) ----------
    // When `density` is non-empty, the base coefficients sigma_a/sigma_s are
    // MULTIPLIED by a dimensionless scalar field density(x,y,z) >= 0 evaluated per
    // point (a compiled pattern program over x y z r, §6.1 of FTSL.md). This shapes
    // the haze into blobs with soft, formula-defined boundaries. Empty => density
    // is 1 everywhere (the classic homogeneous medium; unchanged behaviour).
    std::vector<PatNode> density;
    double densityMax = 1.0;   // majorant: sup of density over `bmin..bmax` (delta/ratio tracking)

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
    bool insideField(const Vec3& p) const {
        double f = fieldEval(boundField.data(), (int)boundField.size(), p,
                             boundFieldExpr.data());
        return boundInsideNeg ? (f < 0.0) : (f > 0.0);
    }

    // A medium is "heterogeneous" (needs delta/ratio tracking rather than an exact
    // analytic free-flight) when it has a density field OR an implicit bound, since
    // an implicit membership makes the effective density spatially varying (1 inside,
    // 0 outside) even when the base coefficients are constant.
    bool heterogeneous() const {
        return !density.empty() || boundShape == MediumBound::Implicit;
    }

    // Dimensionless density multiplier at a world point (>= 0). 1 for a homogeneous
    // medium. Evaluated by the shared pattern VM (x y z r live; f/normal/uv read 0).
    // For an implicit bound the multiplier is 0 outside the field (the medium simply
    // does not exist there), so delta/ratio tracking carves out the exact iso-shape.
    double densityAt(const Vec3& p) const {
        if (boundShape == MediumBound::Implicit && !insideField(p)) return 0.0;
        if (density.empty()) return 1.0;
        PatCtx c = makePatCtx(p, 0.0, Vec3(0, 0, 0));
        double d = patternEval(density.data(), (int)density.size(), c);
        return d > 0.0 ? d : 0.0;
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
enum class EmitterShape { Quad, Sphere, Spot, Env, Cylinder };

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
    double envGeom = 0.0;     // env phase-space weight 4*PI^2*R^2 (Env; set in build())
    EmissionSampler spd;      // for forward per-emitter lambda importance sampling
    Spectrum spdFn = constantSpectrum(0.0); // raw SPD, for backward per-lambda eval
    double emitIntegral = 0.0;
    double power = 0.0;       // emitIntegral * geomWeight (selection weight)

    // Per-emitter spectral/geometric weight fed into the combined backward
    // wavelength sampler and the power law: area*PI for surfaces, spotOmega for a
    // spot. (Area/sphere keep the exact area*PI expression for bit-identity.)
    double geomWeight() const {
        if (shape == EmitterShape::Spot) return spotOmega;
        if (shape == EmitterShape::Env)  return envGeom;
        return area * PI;
    }

    // Sample a surface point `y` and its outward unit normal `nOut` from two
    // uniforms. Quad: the bilinear point with the constant face normal (identical
    // draws to the pre-sphere engine, so quad scenes stay bit-identical). Sphere:
    // a uniformly-distributed surface point (pdf = 1/area for both shapes). Not
    // used for Spot (a point light — see the forward/backward spot paths).
    void samplePoint(double u1, double u2, Vec3& y, Vec3& nOut) const {
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
        } else {
            y = origin + u * u1 + v * u2;
            nOut = normal;
        }
    }

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

struct Scene {
    std::vector<Tri> tris;
    std::vector<Sphere> spheres;
    std::vector<Implicit> implicits;   // isosurfaces / metaballs / (smooth) CSG
    std::vector<Material> mats;
    std::vector<Texture> textures;   // image textures referenced by materials (Phase 3b)
    std::vector<Pattern> patterns;   // procedural scalar fields for math-driven material props (§4)
    Sensor sensor;
    // Participating media. Zero or more independent regions (global haze, bounded
    // boxes/spheres, heterogeneous blobs) that may overlap. The forward tracer treats
    // them as superposed: extinction adds (sigma_t = sum over media containing the
    // point), so transmittance is the product of per-medium transmittances and a
    // collision is the earliest of the media's independent free-flight samples (with
    // the scattering medium chosen by the Poisson superposition theorem). Empty =>
    // vacuum. (The backward/BDPT modes are homogeneous-only and use backwardMedium().)
    std::vector<Medium> media;

    // Backward/BDPT (modes R/V/D and the P composite) support only a single GLOBAL
    // HOMOGENEOUS haze; they ignore density/bounds. This returns the medium they use
    // as that haze — the first authored medium — or a disabled default if there is
    // none. main.cpp warns when an authored medium carries density/bounds for these modes.
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
        emitterCdf.assign(emitters.size(), 0.0);
        for (size_t i = 0; i < emitters.size(); ++i) {
            // Area/sphere keep the exact emitIntegral*area*PI expression so those
            // scenes stay bit-identical; a spot uses its solid-angle weight.
            emitters[i].power =
                (emitters[i].shape == EmitterShape::Spot)
                    ? emitters[i].emitIntegral * emitters[i].spotOmega
                : (emitters[i].shape == EmitterShape::Env)
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
        if (envIndex >= 0) {
            emitters[envIndex].envGeom = 4.0 * PI * PI * sceneRadius * sceneRadius;
            // Directly-viewed background colour: integral of L_env(lambda)*CIE dlambda.
            envXYZ = Vec3{0, 0, 0};
            for (double lam = LAMBDA_MIN; lam <= LAMBDA_MAX; lam += 1.0)
                envXYZ += Vec3(cieX(lam), cieY(lam), cieZ(lam))
                          * emitters[envIndex].spdFn(lam);
        }
        finalizeEmitters();
    }
    void finalizeTris() { build(); }   // kept for existing call sites

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
        boxes.reserve(boxes.size() + implicits.size());
        for (const auto& im : implicits) boxes.push_back(im.bounds);
        bvh.build(boxes);
    }

    Hit closestHit(const Ray& r, double tmin = 1e-6, TraversalStats* stats = nullptr) const {
        Hit h;
        double tMax = DBL_MAX;
        const size_t nT = tris.size();
        const size_t nS = spheres.size();
        bvh.traverseClosest(r, tmin, tMax, [&](int prim, double& tm) {
            if (prim < (int)nT)            { if (intersectTri(r, tris[prim], tmin, h)) tm = h.t; }
            else if (prim < (int)(nT + nS)){ if (intersectSphere(r, spheres[prim - nT], tmin, h)) tm = h.t; }
            else                           { if (intersectImplicit(r, implicits[prim - nT - nS], tmin, h)) tm = h.t; }
        }, stats);
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
        return bvh.traverseAny(r, tmin, maxDist - tmin, [&](int prim) {
            Hit h; h.t = maxDist - tmin;
            if (prim < (int)nT)             return intersectTri(r, tris[prim], tmin, h);
            if (prim < (int)(nT + nS))      return intersectSphere(r, spheres[prim - nT], tmin, h);
            return intersectImplicit(r, implicits[prim - nT - nS], tmin, h);
        });
    }

    // Linear-scan reference (pre-BVH), kept for the -checkbvh self-test.
    Hit closestHitLinear(const Ray& r, double tmin = 1e-6) const {
        Hit h;
        for (const auto& t : tris)     intersectTri(r, t, tmin, h);
        for (const auto& s : spheres)  intersectSphere(r, s, tmin, h);
        for (const auto& im : implicits) intersectImplicit(r, im, tmin, h);
        return h;
    }
};

// Diffuse albedo at a hit: the material's spatially-varying texture reflectance if
// one is bound (Phase 3b), else its constant `reflect` spectrum. Shared by the
// forward tracer and the backward reference so both see identical albedo.
inline double diffuseReflectance(const Scene& scene, const Material& m,
                                 const Hit& h, double lambda) {
    if (m.reflectTex >= 0 && m.reflectTex < (int)scene.textures.size()) {
        const Texture& tx = scene.textures[m.reflectTex];
        if (m.triplanarScale > 0.0)
            return tx.reflectanceTriplanar(h.p, h.ng, m.triplanarScale, lambda);
        return tx.reflectanceAt(h.u, h.v, lambda);
    }
    return m.reflect(lambda);
}

// Build a procedural-pattern evaluation context from a hit: world point (x,y,z),
// implicit field value f (0 on non-implicit surfaces), oriented normal, and radius.
inline PatCtx patCtxFromHit(const Hit& h) { return makePatCtx(h.p, h.fieldVal, h.n, h.u, h.v); }

// Evaluate a bound scalar pattern at the hit (index checked). Returns the pattern
// value, or `dflt` if `pat` is out of range.
inline double patternScalarAt(const Scene& scene, int pat, const Hit& h, double dflt) {
    if (pat >= 0 && pat < (int)scene.patterns.size())
        return scene.patterns[pat].eval(patCtxFromHit(h));
    return dflt;
}

// Per-hit glossy roughness: a bound roughness pattern (highest priority, for
// implicit surfaces) or roughnessTex's grayscale value at the hit, else the
// constant. Shared by every tracer so sampling and (in BDPT) the MIS pdf see the
// SAME roughness at a hit — otherwise the density and the sample diverge.
inline double materialRoughness(const Scene& scene, const Material& m, const Hit& h) {
    if (m.roughnessPat >= 0 && m.roughnessPat < (int)scene.patterns.size()) {
        double r = scene.patterns[m.roughnessPat].eval(patCtxFromHit(h));
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
        return scene.patterns[m.filmThicknessPat].eval(patCtxFromHit(h)) * m.filmThickness;
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
            t = scene.patterns[m.mixWeightPat].eval(patCtxFromHit(h));
        else
            t = scene.textures[m.mixWeightTex].scalarAt(h.u, h.v);
        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
        return (u < t) ? m.mixChildren[0] : m.mixChildren[1];
    }
    return mixPickChild(m, u);
}
