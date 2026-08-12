// Bridge between the renderer's Scene / Material / Hit and the standalone fiber BCSDF.
//
// `hair.h` deliberately knows nothing about this renderer — that is what lets
// `-checkhair` run nine sections of physics with no scene, no camera and no BVH. Every
// piece that needs a Scene therefore lives here instead: reading the material's spectral
// slots, inverting an authored colour into an absorption, building the local fiber frame,
// and turning a BCSDF value into the quantity the light transport actually wants.
//
// THE ONE THING TO UNDERSTAND HERE is the projection factor, because it is not the
// surface one and getting it wrong is invisible until you check energy.
//
// A surface vertex splats to the camera with `beta * f * dot(n, w)`: the cosine is the
// foreshortening of the surface patch. A fiber has no such patch. What is foreshortened
// is the STRAND, and a round strand's projected width is the same from every azimuth, so
// the only foreshortening is longitudinal: `cos theta = sqrt(1 - w_x^2)` in the fiber
// frame. That is also exactly the measure the BCSDF is normalised against — the furnace
// identity `-checkhair` §1 asserts is `integral f cos(theta) dw = 1` with THAT cosine —
// so the two agree by construction. Multiplying by `dot(n, w)` instead would both
// double-count the tube's curvature and break energy conservation, and (worse) it would
// look almost right: the error is a smooth function of angle, so it reads as "my hair
// shader is a bit dark at grazing angles" rather than as a bug.
//
// `hairFCos()` therefore returns `f * cos(theta_long)`, which is the complete
// "BSDF times projection" factor a connection needs. PBRT reaches the same number from
// the other side — it divides `f` by the renderer's `AbsCosTheta` so the integrator's
// own cosine cancels — but doing it explicitly here means the fiber path never quietly
// depends on a cancellation happening somewhere else.
#pragma once
#include <memory>
#include <mutex>
#include <unordered_map>
#include "hair.h"
#include "scene.h"
#include "fur_grid.h"   // -dual-grid: Zinke's §4.1.2 voxel density field

// Everything needed to evaluate or sample the fiber BCSDF at one hit, for one wavelength.
struct HairShade {
    hair::Bcsdf b;
    hair::Frame fr;
    Vec3   woLocal{1, 0, 0};   // the direction the path ARRIVED from, in the fiber frame
    double radius = 0.0;       // world fiber radius; 0 = this hit is not on a strand
    // This is a VIRTUAL fiber, drawn from a density grid's orientation distribution at a
    // volumetric collision (`-fur-volume`, the far LOD tier), not a strand the BVH found.
    // Physically it is the same fiber and the same BCSDF; what changes is VISIBILITY. The
    // strands around it were never traced, so a shadow ray from here must skip fibers in the
    // BVH (they are not there) and pick the coat up as a TRANSMITTANCE instead. Every
    // connection therefore has to know which kind of fiber it is standing on — see
    // backward.h's emitterGeom/envGeom.
    bool   aggregate = false;
};

// Build the BCSDF at a hit. `wPrev` points away from the surface, back along the path
// that got here: toward the light for the forward tracer, toward the eye for the backward
// one. It is the model's reference direction — the impact parameter `h` is measured
// relative to it (as it is in PBRT, where `h` comes from the ribbon that was turned to
// face the incident ray), so it must be the arrival direction and not the sampled one.
inline HairShade hairShadeAt(const Scene& scene, const Material& m, const Hit& hit,
                             double lambda, const Vec3& wPrev) {
    hair::Params pr;
    pr.eta   = m.hairEta;
    pr.betaM = m.hairBetaM;
    pr.betaN = m.hairBetaN;
    pr.alpha = m.hairAlpha;
    // Medulla (stage 3). kappa = 0 is the default, which makes every one of these inert
    // and leaves the fiber a solid stage-1 cylinder.
    pr.kappa   = m.hairKappa;
    pr.mG      = m.hairMedullaG;
    pr.mSigmaS = std::max(0.0, m.hairMedullaSigmaS(lambda));
    pr.mSigmaA = std::max(0.0, m.hairMedullaSigmaA(lambda));

    double sigmaA;
    if (m.hairSigmaAFromReflect) {
        // Chiang eq. 9 run per-wavelength: the authored `reflect` spectrum is read as the
        // reflectance curve the fiber should END UP with, and inverted into the absorption
        // that produces it. Going through diffuseReflectance() (rather than the constant
        // spectrum) is what lets a groom carry a texture, a record or a pattern on its
        // colour — root-to-tip darkening is just `reflect [0.1 0.6](u)`.
        double c = diffuseReflectance(scene, m, hit, lambda);
        c = c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c);
        sigmaA = hair::sigmaAFromReflectance(c, m.hairBetaN);
    } else {
        sigmaA = std::max(0.0, m.hairSigmaA(lambda));
    }

    HairShade s;
    // hit.n is oriented against the arriving ray, so it is the normal on the side the path
    // came from — which is the side `h` and `gamma_o` are defined from.
    s.fr      = hair::frameFromHit(hit.n, hit.tangent);
    s.woLocal = hair::toLocal(s.fr, wPrev);
    s.b       = hair::make(pr, hair::hFromHit(hit.n, hit.tangent, wPrev), sigmaA);
    s.radius  = hit.fiberRadius;
    return s;
}

// The complete "BCSDF times projection" factor toward a world direction — see the header
// comment. This is what multiplies the throughput in a connection, in place of the
// surface path's `f * dot(n, w)`.
inline double hairFCos(const HairShade& s, const Vec3& wWorld) {
    const Vec3 wl = hair::toLocal(s.fr, wWorld);
    const double cosLong = hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
    return hair::f(s.b, s.woLocal, wl) * cosLong;
}

// How far a connection or a scattered ray must step to clear the strand's own body.
//
// The near-field model puts the TT and TRT exits at the ENTRY point, but the geometry is a
// real solid tube: a ray leaving toward the far side would immediately re-hit the strand
// it just came from and be reported as occluded, deleting the forward glow that dominates
// a light-coloured coat. Stepping a couple of diameters clears the tube (strand radii are
// microns, so this displaces nothing visible), and on a non-fiber hit — where fiberRadius
// is 0 — it degrades to the ordinary 1e-6 surface offset.
inline double hairExitOffset(const HairShade& s, const Vec3& n, const Vec3& w) {
    if (dot(n, w) >= 0.0) return 1e-6;              // leaving on the arrival side: normal case
    return (s.radius > 0.0) ? 2.5 * s.radius + 1e-9 : 1e-6;
}

// ---- Dual scattering (P3 stage 4) --------------------------------------------------
//
// `hair::Dual` is a pure function of the fiber's parameters at ONE wavelength, and
// building one costs ~50k BCSDF samples — far too much to redo per shading point and far
// too little to be worth a disk format. So they are memoised here, in the only place that
// knows both the Scene and the fiber model.
//
// THE KEY IS THE INTERESTING PART. A table depends on the material, on lambda (through
// three spectral slots: the cortex absorption and the medulla's sigma_s / sigma_a) and,
// when the colour is textured or pattern-driven, on WHERE on the strand we are. Keying on
// (material, lambda bin, sigma_a bin) covers all three: a plainly-coloured fiber produces
// exactly one table per lambda bin, and a textured one produces a few more. Two shading
// points that fall in the same bin share a table built at whichever lambda got there
// first — that quantisation IS the approximation, and 10 nm is far finer than the scale
// on which a keratin absorption curve moves.
struct HairDualCache {
    std::mutex mu;
    std::unordered_map<uint64_t, std::unique_ptr<hair::Dual>> tab;
};
inline HairDualCache& hairDualCache() { static HairDualCache c; return c; }

// The table for this material at this wavelength. Never null.
inline const hair::Dual* hairDualFor(const Scene& scene, const Material& m, int matId,
                                     const Hit& hit, double lambda) {
    hair::Params pr;
    pr.eta = m.hairEta; pr.betaM = m.hairBetaM; pr.betaN = m.hairBetaN; pr.alpha = m.hairAlpha;
    pr.kappa = m.hairKappa; pr.mG = m.hairMedullaG;
    pr.mSigmaS = std::max(0.0, m.hairMedullaSigmaS(lambda));
    pr.mSigmaA = std::max(0.0, m.hairMedullaSigmaA(lambda));
    double sigmaA;
    if (m.hairSigmaAFromReflect) {
        double c = diffuseReflectance(scene, m, hit, lambda);
        c = c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c);
        sigmaA = hair::sigmaAFromReflectance(c, m.hairBetaN);
    } else {
        sigmaA = std::max(0.0, m.hairSigmaA(lambda));
    }
    // 10 nm in lambda; sqrt-spaced in sigma_a so the bins are fine where the colour is
    // (sigma_a under ~1) and coarse out in the near-black tail where nothing changes.
    const uint64_t lbin = uint64_t(std::max(0.0, std::min(4095.0, lambda * 0.1)));
    const uint64_t sbin = uint64_t(std::min(63.0, std::sqrt(sigmaA) * 8.0 + 0.5));
    const uint64_t key  = (uint64_t(uint32_t(matId)) << 32) | (lbin << 8) | sbin;

    HairDualCache& c = hairDualCache();
    {
        std::lock_guard<std::mutex> lk(c.mu);
        auto it = c.tab.find(key);
        if (it != c.tab.end()) return it->second.get();
    }
    // Built OUTSIDE the lock: two threads racing on a cold key each build one and the
    // loser's copy is dropped, which is cheaper (and far less deadlock-prone) than making
    // every other thread wait behind one 50k-sample integration.
    auto built = std::make_unique<hair::Dual>(hair::makeDual(pr, sigmaA, 1024));
    std::lock_guard<std::mutex> lk(c.mu);
    auto& slot = c.tab[key];
    if (!slot) slot = std::move(built);
    return slot.get();
}

// What a shadow ray learned on its way to the light through a coat: Zinke's forward
// scattering transmittance `T_f` (eq. 5, without the density factor — the shader applies
// that once) and spread variance `sigma_f^2` (eq. 8), plus whether anything was crossed
// at all.
struct HairShadow {
    double Tf   = 1.0;      // product of a_f(theta_d) over the fibers crossed
    double varF = 0.0;      // sum of beta_f(theta_d)^2 over the same
    int    n    = 0;        // fibers crossed; 0 means directly lit
    bool   blocked = false; // an opaque (non-fiber) surface stopped the ray
};

// Accumulate one crossing. `wLight` points from the shading point toward the light, so it
// is already the away-pointing direction at the crossed fiber and its longitudinal
// component in that fiber's frame is sin(theta_d) directly.
inline void hairShadowCross(HairShadow& sh, const hair::Dual& d, const Hit& fiberHit,
                            const Vec3& wLight) {
    const hair::Frame fr = hair::frameFromHit(fiberHit.n, fiberHit.tangent);
    const double sinT = hair::clampd(hair::toLocal(fr, wLight).x, -1.0, 1.0);
    const double th   = std::asin(sinT);
    sh.Tf   *= hair::dualLookup(d.af, th);
    sh.varF += hair::sqr(hair::dualLookup(d.betaF, th));
    ++sh.n;
}

// The dual-scattering replacement for `hairFCos` at a next-event connection: Zinke's
// F(x, w_d, w_o) (his figure 5), already multiplied by the longitudinal projection, so it
// drops into exactly the place the single-scattering `PI * hairFCos` occupies.
//
//   `wLight`   direction to the light (world), the shadow ray that produced `sh`.
//   `wSpread`  a direction drawn from the forward-scattering spread S_f around `wLight`;
//              only read when the point is indirectly lit. The paper convolves the
//              azimuthal part of the BCSDF against an isotropic front half analytically
//              (its table N^G); a path tracer can just draw one sample of that same
//              convolution per connection and let the pixel average do the integral,
//              which is unbiased and needs no second table.
//   `db`,`df`  the backward / forward density factors (both 0.7 in the paper).
inline double hairDualFCos(const HairShade& s, const hair::Dual& d, const HairShadow& sh,
                           const Vec3& wLight, const Vec3& wSpread, double db, double df) {
    if (sh.blocked) return 0.0;
    // The direction the light ACTUALLY arrives from: the light itself when the point is
    // directly lit, one draw from the forward spread S_f when it is not. EVERYTHING below
    // is evaluated at that direction, `cosLong` very much included.
    //
    // Getting that cosine from `wLight` instead — which is what this did first — is a
    // 13x error, not a subtlety. The shading integral is int f(w_o,w_i) L_in(w_i) cos(th_i)
    // dw_i with L_in = T_f d_f S_f, so the cosine belongs to w_i; and `hair::f` is in
    // Chiang's normalisation, which ends in `/= |cos th_i|`. Pair it with its own cosine
    // and the two cancel exactly. Pair it with the light's and nothing cancels: sigma_f
    // after a dozen crossings is over a radian, so the Gaussian draw lands near the fiber
    // axis constantly, |cos th_i| hits its 1e-4 clamp, and f is handed back amplified by up
    // to 1e4 with no cosine to undo it. That is where the 67x 99th percentile came from.
    const Vec3 wIn = hair::toLocal(s.fr, sh.n == 0 ? wLight : wSpread);
    const double sinO = hair::clampd(s.woLocal.x, -1.0, 1.0);
    const double sinI = hair::clampd(wIn.x, -1.0, 1.0);
    const double thO = std::asin(sinO), thI = std::asin(sinI);
    const double cosLong = hair::safeSqrt(1.0 - hair::sqr(sinI));
    // theta_D is Marschner's DIFFERENCE angle, which is what indexes the backscattering
    // tables (eq. 10's `theta = (theta_o - theta_i)/2`); `dev` is the deviation from the
    // specular cone, the variable every longitudinal lobe is a Gaussian in, and therefore
    // the variable the accumulated forward variance widens.
    const double thetaD = 0.5 * (thO - thI);
    const double dev    = thO + thI;
    // `sh.varF` does NOT widen this lobe: `wIn` is already one Monte-Carlo draw of exactly
    // the spread it measures, so `dev` carries the blur and adding sigma_f^2 to the Gaussian
    // would apply it twice. See hair::fBack.
    const double fb = hair::fBack(d, thetaD, dev, cosLong);

    // `s_b` (eq. 15) is 1/pi over BACKWARD azimuths and ZERO forward, which makes f_back a
    // BCSDF like any other: a lobe of density 1/pi spread over the pi-wide retro half. The
    // 1/pi lives inside fBack; the restriction is here, where phi is known. It applies to
    // BOTH branches — backscattered light returns toward wherever the light came from,
    // whether that is the source itself or the direction it forward-scattered in.
    //
    // This is where the implementation parts company with Zinke's Figure 5, which writes
    // `f_s^scatter + PI d_b f_back` and applies no azimuthal test to the indirect branch.
    // That PI cancels f_back's own 1/pi, so the paper adds the backscatter lobe to an
    // indirectly-lit fiber ~2 PI times more strongly than to a directly-lit one. The
    // discrepancy is in the paper itself: its f_s^scatter is a precomputed azimuthal
    // convolution (eq. 25's N^G), where here it is a Monte-Carlo draw from the same spread
    // (hairSpreadDir), so the azimuth of the arriving light is a KNOWN direction rather
    // than something already integrated away, and f_back can be — and has to be — treated
    // as the BCSDF it is.
    const bool back = (s.woLocal.y * wIn.y + s.woLocal.z * wIn.z) > 0.0;
    const double fs = hair::f(s.b, s.woLocal, wIn) + (back ? db * fb : 0.0);
    return (sh.n == 0 ? 1.0 : sh.Tf * df) * fs * cosLong;
}

// One draw from the forward-scattering spread S_f (Zinke eq. 7) about the light
// direction, in the shading fiber's frame: the azimuth is uniform over the FRONT half
// (`s_f = 1/pi` there, zero behind — after a few crossings the coat has forgotten which
// way round the light was) and the longitudinal angle is the light's, jittered by a
// Gaussian of the accumulated variance.
inline Vec3 hairSpreadDir(const HairShade& s, const HairShadow& sh, const Vec3& wLight,
                          double u0, double u1, double u2) {
    const Vec3 wl = hair::toLocal(s.fr, wLight);
    const double thI = std::asin(hair::clampd(wl.x, -1.0, 1.0));
    // Box-Muller, one normal per connection.
    const double g = std::sqrt(-2.0 * std::log(std::max(1e-12, u0))) * std::cos(2.0 * hair::kPi * u1);
    const double th = hair::clampd(thI + g * std::sqrt(std::max(0.0, sh.varF)),
                                   -0.5 * hair::kPi + 1e-4, 0.5 * hair::kPi - 1e-4);
    const double phiL = std::atan2(wl.z, wl.y);
    const double phi  = phiL + hair::kPi * (u2 - 0.5);   // uniform over the front half
    const double ct = std::cos(th);
    return hair::toWorld(s.fr, Vec3{std::sin(th), ct * std::cos(phi), ct * std::sin(phi)});
}

// Everything a fiber next-event connection needs beyond the single-scattering path, so
// the integrator can hand one pointer down instead of six arguments. Null = stage 1-3
// behaviour, i.e. every existing scene renders exactly as before.
struct HairDualCtx {
    const Scene* scene = nullptr;
    const hair::Dual* dual = nullptr;   // the table for the SHADING fiber
    double lambda = 0.0;
    double db = 0.7, df = 0.7;          // Zinke's backward / forward density factors
    int    maxCross = 64;               // cap on strands counted along one shadow ray
    double u0 = 0.0, u1 = 0.0, u2 = 0.0;// uniforms for the one spread draw
    double u3 = 0.0;                    // and one more, for the grid's Poisson crossing count
    // Non-null selects Zinke's §4.1.2 density grid over his §4.1.1 ray shooting: the coat's
    // attenuation is integrated out of a voxel field instead of by crossing every strand.
    const FurGrid* grid = nullptr;
};

// Measure the coat along a shadow ray by MARCHING A DENSITY FIELD rather than crossing
// strands — Zinke et al. 2008 §4.1.2, the alternative to the §4.1.1 walk below.
//
// The whole substitution rests on one identity, which `-checkfurgrid` §4 verifies against
// the shipping curve intersector: the grid's optical depth IS the expected number of fiber
// crossings, because a fiber of radius r and length l presents `2 r l sin(theta)` of
// cross-section and `sigma_t` is exactly that summed per unit volume. So `tau` is the same
// `n` the walk counts, and `<sin^2 theta>` — which the orientation tensor gives for free,
// since `d^T T d` is the mean of `(d.tangent)^2` and that is `sin^2(theta_i)` in Marschner's
// frame — is the inclination to look `a_f` and `beta_f` up at.
//
// THE COUNT IS SAMPLED, NOT ROUNDED, and that is the whole design of this function. `tau` is
// a MEAN; the walk it replaces returns a random draw, and the shader downstream is not
// linear in the draw. Substituting the mean gets three things wrong at once:
//
//  * `T_f` would become `a_f^tau` when the quantity wanted is `E[a_f^N]`. For Poisson N that
//    is `exp(lambda(a-1))`, larger by Jensen — at a_f = 0.8, lambda = 10 it is 0.135 against
//    0.107, a 26% error, all of it darkening.
//  * The "directly lit" branch would never fire. `hairDualFCos` treats `n == 0` specially
//    (no `d_f`, no spread), and it should: a strand on the lit side of the coat really is
//    unshadowed. Rounding `tau = 0.3` to `n = 1` replaces `P(N=0) = 74%` with 0% and puts a
//    0.7 density factor on light that never met a fiber, so the coat's rim goes dull.
//  * The spread variance `n * beta^2` would be a fixed width instead of a distribution of
//    widths, which is visible as a missing soft-to-sharp gradient across the coat.
//
// Drawing `N ~ Poisson(tau)` from one uniform (CDF inversion — one `exp` and `tau` or so
// multiplies) fixes all three at once and makes this an exact drop-in for the walk: same
// estimator, same variance, same distribution of `n`, only the SOURCE of the count differs.
// Everything after the draw is byte-for-byte the shader the walk feeds.
//
// The remaining approximations are the two the grid genuinely cannot avoid: the inclination
// is a single tau-weighted mean rather than a per-crossing angle, and the table read is
// SIGN-MARGINALISED (`furAvgSigned`) because a second moment cannot tell a tangent from its
// reverse.
//
// Visibility is a separate query here, which is the other half of why this is faster:
// `walkFibers` must visit every primitive overlapping the segment because a fiber does not
// stop it, while `occludedSkipHair` early-outs on the first wall.
// `shadingHit` is the fiber being shaded, not a crossed one — the grid does not know which
// strands it summarised, so a material whose absorption comes from a TEXTURE (`hairSigmaA
// FromReflect`) has to be sampled somewhere, and the nearest fiber whose surface parameters
// are actually known is the shading vertex itself. For a coat that is one texture varying
// slowly across the body this is right to within the texture's own gradient over a shadow
// ray; for a coat with high-frequency per-strand colour it is a genuine loss the walk does
// not have, and is the reason `-dual-grid` is opt-in rather than the default.
inline bool hairShadowGrid(HairShadow& sh, const HairDualCtx& ctx, const Scene& sc,
                           const Hit& shadingHit, const Vec3& o, const Vec3& wi, double len) {
    if (sc.occludedSkipHair(o, wi, len)) { sh.blocked = true; return false; }
    const FurMarch fm = ctx.grid->march(o, wi, len);
    if (!(fm.tau > 0.0) || fm.matId < 0 || fm.matId >= (int)sc.mats.size()) return true;
    // `maxCross` keeps its meaning: past this many crossings there is nothing left to carry,
    // and capping the mean also bounds the inversion loop below.
    const double tau = std::min(fm.tau, (double)ctx.maxCross);
    // N ~ Poisson(tau) by CDF inversion. `p` is the pmf, stepped by the `tau/n` recurrence;
    // `u < cdf` terminates in `tau + O(sqrt(tau))` iterations on average. The `n` cap is the
    // same safety net as the walk's, and doubles as the guard against a `u` that rounds to 1.
    int n = 0;
    {
        double p = std::exp(-tau), cdf = p;
        const double u = ctx.u3;
        while (u > cdf && n < ctx.maxCross) { ++n; p *= tau / n; cdf += p; }
    }
    sh.n = n;
    if (!n) return true;                                  // directly lit: T_f = 1, no spread
    const hair::Dual* d = hairDualFor(sc, sc.mats[fm.matId], fm.matId, shadingHit, ctx.lambda);
    if (!d) { sh.n = 0; return true; }
    const double th = std::asin(hair::clampd(std::sqrt(fm.sin2), -1.0, 1.0));
    const double af = furAvgSigned([&](double t) { return hair::dualLookup(d->af, t); }, th);
    const double bf = furAvgSigned([&](double t) { return hair::sqr(hair::dualLookup(d->betaF, t)); }, th);
    sh.Tf   = std::pow(hair::clampd(af, 0.0, 1.0), (double)n);
    sh.varF = n * bf;
    return true;
}

// The dual-scattering surface response toward `wi`, INCLUDING visibility: the shadow ray
// is the thing that measures T_f, so unlike the single-scattering path the connection
// cannot be split into "response" and "occluded" any more. Returns 0 and sets
// `blockedOut` when something solid is in the way.
inline double hairDualResponse(const HairDualCtx& ctx, const HairShade& s, const Hit& h,
                               const Vec3& wi, double dist, bool& blockedOut) {
    blockedOut = true;
    const Scene& sc = *ctx.scene;
    const double off = hairExitOffset(s, h.n, wi);
    const double len = dist - off - 1e-6;
    if (!(len > 0.0)) return 0.0;
    HairShadow sh;
    const Vec3 o = h.p + wi * off;
    bool reached;
    if (ctx.grid && ctx.grid->valid) {
        reached = hairShadowGrid(sh, ctx, sc, h, o, wi, len);
    } else {
        reached = sc.walkFibers(o, wi, len, [&](const Hit& fh) {
            const hair::Dual* d = hairDualFor(sc, sc.mats[fh.matId], fh.matId, fh, ctx.lambda);
            hairShadowCross(sh, *d, fh, wi);
        }, ctx.maxCross);
    }
    if (!reached) return 0.0;
    blockedOut = false;
    const Vec3 ws = (sh.n > 0) ? hairSpreadDir(s, sh, wi, ctx.u0, ctx.u1, ctx.u2) : wi;
    return hairDualFCos(s, *ctx.dual, sh, wi, ws, ctx.db, ctx.df);
}
