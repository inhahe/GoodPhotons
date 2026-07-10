// Scene container: triangles, materials, one area light, one contact sensor.
#pragma once
#include <vector>
#include <algorithm>
#include "geometry.h"
#include "bvh.h"
#include "spectrum.h"
#include "scene_film.h"

enum class MatType { Diffuse, Dielectric, Mirror, HalfMirror, Glossy, Fluorescent, ThinFilm, Grating, Mix };

// Materials whose last-vertex-before-camera cannot connect to the pinhole in
// model B (a delta or near-delta BSDF has ~zero connection pdf): the forward
// light tracer renders them BLACK from the camera (the SDS limitation). The
// camera-side ray path (mode P) is what fills these pixels in. Diffuse and
// Fluorescent connect in model B, so they are NOT specular-side.
inline bool isSpecularType(MatType t) {
    return t == MatType::Dielectric || t == MatType::Mirror ||
           t == MatType::HalfMirror || t == MatType::ThinFilm ||
           t == MatType::Glossy     || t == MatType::Grating;
}

struct Material {
    MatType type = MatType::Diffuse;
    // reflect means: diffuse albedo / mirror tint / glossy tint / half-mirror
    // reflect-probability, depending on type. For Fluorescent it is the elastic
    // (wavelength-preserving) diffuse albedo.
    Spectrum reflect = constantSpectrum(0.5);
    Spectrum emit    = constantSpectrum(0.0); // emitted radiance vs lambda
    Spectrum ior     = iorConstant(1.5);      // dielectric index vs lambda
    double roughness = 0.1;                    // glossy lobe width [0,1]
    bool isLight = false;

    // --- Thin-film / iridescence (MatType::ThinFilm) ------------------------
    // A thin dielectric coating of index filmIor and thickness filmThickness (in
    // nanometres) over a dielectric substrate whose index is `ior`. Interference
    // between the two coating interfaces yields an angle/wavelength-dependent
    // reflectance (structural colour). Transport is lossless specular reflect-or-
    // refract, exactly like Dielectric (so the backward tracer handles it too).
    double filmIor = 1.30;                      // coating refractive index n1
    double filmThickness = 300.0;              // coating thickness in nanometres

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
struct Medium {
    bool enabled = false;
    Spectrum sigma_a = constantSpectrum(0.0); // absorption coefficient vs lambda
    Spectrum sigma_s = constantSpectrum(0.0); // scattering coefficient vs lambda
    double g = 0.0;                            // HG anisotropy [-1,1] (0 = isotropic)

    double sigmaT(double lambda) const {
        return std::max(0.0, sigma_a(lambda) + sigma_s(lambda));
    }
    double albedo(double lambda) const {       // single-scattering albedo sigma_s/sigma_t
        double s = std::max(0.0, sigma_s(lambda));
        double t = s + std::max(0.0, sigma_a(lambda));
        return t > 0.0 ? s / t : 0.0;
    }
};

// A flat rectangular contact sensor (model A) spanning origin + s*uAxis + t*vAxis.
struct Sensor {
    Vec3 origin, uAxis, vAxis; // uAxis/vAxis are full edge vectors
    Film film;
    void alloc() { film.alloc(); }
};

// A single emitter. An area light is a quad (origin + s*u + t*v, s,t in [0,1])
// with one-sided Lambertian emission along `normal`. A collimated emitter fires
// every photon along `beamDir` from that same quad (the prism demo). Each emitter
// carries its own SPD; `power` = emitIntegral * area * PI is the emitter's total
// emitted power and doubles as the selection weight for the power-weighted CDF.
struct Emitter {
    Vec3 origin, u, v, normal;
    double area = 0.0;
    bool collimated = false;
    Vec3 beamDir{1, 0, 0};
    EmissionSampler spd;      // for forward per-emitter lambda importance sampling
    Spectrum spdFn = constantSpectrum(0.0); // raw SPD, for backward per-lambda eval
    double emitIntegral = 0.0;
    double power = 0.0;       // emitIntegral * area * PI (selection weight)
};

struct Scene {
    std::vector<Tri> tris;
    std::vector<Sphere> spheres;
    std::vector<Material> mats;
    Sensor sensor;
    Medium medium;   // optional global fog / participating medium (disabled by default)

    // Emitters. Forward tracing selects one per photon with probability
    // proportional to power (so every photon carries beta = totalPower, keeping
    // the estimator unbiased); backward tracing samples wavelengths from the
    // combined emission distribution and sums NEE over all emitters.
    std::vector<Emitter> emitters;
    std::vector<double> emitterCdf;   // cumulative power, normalised to [0,1]
    double totalPower = 0.0;
    // Combined emission wavelength sampler over g(lambda)=sum_k area_k*PI*SPD_k,
    // with emitG = its integral. invPdfLambda(lambda) = emitG / g(lambda) is the
    // per-lambda weight the backward reference needs (see backward.h).
    EmissionSampler emitSampler;
    double emitG = 0.0;

    // Register one area (or collimated) light. Terse helper for the C++ builders
    // and the FTSL loader; call finalizeEmitters() (via build()) afterwards.
    void addAreaLight(const Vec3& o, const Vec3& U, const Vec3& V, const Vec3& n,
                      double area, const Spectrum& spd, double stepNm,
                      bool collimated = false, const Vec3& beamDir = {1, 0, 0}) {
        Emitter e;
        e.origin = o; e.u = U; e.v = V; e.normal = n; e.area = area;
        e.collimated = collimated; e.beamDir = beamDir;
        e.spd.build(spd, stepNm); e.spdFn = spd; e.emitIntegral = e.spd.integral;
        emitters.push_back(std::move(e));
    }

    // Compute per-emitter power, the selection CDF, and the combined backward
    // wavelength sampler. Idempotent; called by build().
    void finalizeEmitters(double stepNm = 1.0) {
        totalPower = 0.0;
        emitterCdf.assign(emitters.size(), 0.0);
        for (size_t i = 0; i < emitters.size(); ++i) {
            emitters[i].power = emitters[i].emitIntegral * emitters[i].area * PI;
            totalPower += emitters[i].power;
            emitterCdf[i] = totalPower;
        }
        if (totalPower > 0) for (auto& c : emitterCdf) c /= totalPower;
        // Combined g(lambda) = sum_k area_k*PI*SPD_k(lambda); by value capture.
        std::vector<std::pair<double, Spectrum>> parts;
        for (const auto& e : emitters) parts.push_back({e.area * PI, e.spdFn});
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
        for (const auto& e : emitters) g += e.area * PI * e.spdFn(lambda);
        return (g > 0.0) ? emitG / g : 0.0;
    }

    Bvh bvh;   // acceleration structure over tris (0..nTris) then spheres.

    // Finalize triangle normals and build the BVH. Call after all geometry is
    // added. Primitive index i: i < tris.size() -> tris[i]; else spheres[i-nTris].
    void build() {
        for (auto& t : tris) t.finalize();
        buildBvh();
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
        bvh.build(boxes);
    }

    Hit closestHit(const Ray& r, double tmin = 1e-6, TraversalStats* stats = nullptr) const {
        Hit h;
        double tMax = DBL_MAX;
        const size_t nT = tris.size();
        bvh.traverseClosest(r, tmin, tMax, [&](int prim, double& tm) {
            if (prim < (int)nT) { if (intersectTri(r, tris[prim], tmin, h)) tm = h.t; }
            else                { if (intersectSphere(r, spheres[prim - nT], tmin, h)) tm = h.t; }
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
        return bvh.traverseAny(r, tmin, maxDist - tmin, [&](int prim) {
            Hit h; h.t = maxDist - tmin;
            if (prim < (int)nT) return intersectTri(r, tris[prim], tmin, h);
            return intersectSphere(r, spheres[prim - nT], tmin, h);
        });
    }

    // Linear-scan reference (pre-BVH), kept for the -checkbvh self-test.
    Hit closestHitLinear(const Ray& r, double tmin = 1e-6) const {
        Hit h;
        for (const auto& t : tris)    intersectTri(r, t, tmin, h);
        for (const auto& s : spheres) intersectSphere(r, s, tmin, h);
        return h;
    }
};
