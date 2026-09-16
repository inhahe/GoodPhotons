#pragma once
// VOLCACHE -- a volumetric radiance cache for the order >= 2 part of the beam gather.
//
// WHAT IS STORED. Per cell, the DIRECTIONAL distribution of photon path-length density,
// projected onto real spherical harmonics:
//
//     L(x, w) ~= sum_lm c_lm(x) Y_lm(w),     w = the photon's propagation direction
//
// The camera reconstructs in-scattered radiance from it at march time:
//
//     L_s(x, w_out) = sigma_s(x) * (p * L)(x, w_out)
//
// WHY SH, AND WHY THIS IS EXACT RATHER THAN A FIT. In-scattering is a CONVOLUTION of the
// directional radiance with the phase function, and a phase function is rotationally
// symmetric -- it depends only on the angle between incoming and outgoing directions. A
// convolution with a rotationally-symmetric kernel is DIAGONAL in the SH basis (Funk-Hecke):
// band l is scaled by the kernel's l-th Legendre moment, nothing more. For Henyey-Greenstein
// that moment is exactly g^l, so
//
//     L_s(w_out) = sigma_s * sum_lm g^l c_lm Y_lm(w_out)
//
// is not an approximation OF the anisotropic phase -- it IS the anisotropic phase, to the
// truncation order of the series. The earlier scalar cache could not do this at all: it had
// averaged the incoming directions away, so only the isotropic mean was recoverable from it.
//
// THE ISOTROPIC CASE IS THE l = 0 TERM, EXACTLY. At g = 0 every g^l with l > 0 vanishes and
// only c_00 Y_00 survives, which equals fluence/(4 pi) -- the scalar formula this cache used
// before it had directions. So `nSH` is 1 for an isotropic scene: the old behaviour is not a
// special case bolted on, it is what the series reduces to, and it costs the memory it used to.
//
// SCOPE, ENFORCED RATHER THAN ASSUMED. `build()` refuses:
//   * a RAINBOW phase -- a wavelength-dependent Airy table, not HG, so its Legendre moments
//     are neither g^l nor achromatic. They are computable by quadrature per CIE channel; that
//     is real work and is not done here.
//   * a HETEROGENEOUS medium -- the splat attenuates with exp(-sigma_t s), which is the
//     transmittance only a homogeneous medium has.
//   * media that DISAGREE on sigma_t or g -- one grid carries one pair of both. The previous
//     version silently took the LAST enabled medium's sigma_t and applied it to every beam,
//     which is the kind of error that reads as a soft bias rather than as a failure.
//
// THE DIRECTIONAL BINS, MEASURED TWICE -- and ON by default for g != 0 since 0.311.0. The SH
// machinery below reduces to the scalar cache exactly at g = 0 and is energy-correct to 0.8% at
// g = 0.5. The FIRST measurement said it buys nothing: `_beams_ms` at 96^2 -- a sphere of fog in a
// closed box under an area light -- against the uncached render, per-pixel median error:
//
//     medium          SH order 2      l = 0 only
//     g = 0.5           3.03 %          1.88 %
//     g = 0.85          2.31 %          2.13 %
//
// There the l = 0 reconstruction is as good or better even at g = 0.85, and the reason given was
// that this cache only holds the order >= 2 component, whose field diffusion has made nearly
// isotropic. That is true of an ENCLOSURE lit diffusely from above. It is not true of a
// collimated backlight. The SECOND measurement, gallery_rain frame 555 (a g = 0.46 cloud with the
// sun behind it, camera looking into the sun; device march, 320x180 spp 32, the 3 sun-disc pixels
// masked -- see known-issues "VOLCACHE DEVICE MARCH"), frame ratio cache / uncached:
//
//     reconstruction   frame ratio    top-rim tiles
//     l = 0 only         0.934        0.86  0.87  0.83
//     SH order 2         0.996        0.985 0.988 0.928
//
// A scalar cache is 7% dark on a backlit cloud -- systematic, direction-dependent, and so a
// brightness that would drift with the camera across a flyby -- while the bins cost nothing
// measurable (86 s vs 105 s for the frame; a 48^3 grid at 9 bands x 3 channels is 12 MB). The
// order-2 term of a sun through a thin cloud keeps a g^2 forward lobe; the enclosure measurement
// was not wrong, it was local. Hence the default: bins whenever g != 0. The enclosure case pays
// the +0.8% clamp bias for it (ringing clamped non-negative reads bright).
// FTRACE_VOLCACHE_SH=0 restores the scalar reconstruction; =1 forces the bins.
//
// This is a PROTOTYPE for a go/no-go decision, not the production cache: no confidence gate,
// no validation paths, no adaptive resolution.
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "linalg.h"
#include "photonbeams.h"

// Real SH, bands 0..2 (9 coefficients). Order 2 because the series converges geometrically in
// g: at the g = 0.46 of gallery_rain's cloud the bands carry 1, 0.46, 0.21, 0.097, so band 3
// is under a tenth of band 0 and bands 0..2 hold ~96% of the weight. Probe-based real-time GI
// settled on the same order, for the same reason.
inline constexpr int kVolShN = 9;
inline void volShBasis(const Vec3& d, double* y) {
    const double x = d.x, u = d.y, z = d.z;
    y[0] = 0.2820947917738781;
    y[1] = 0.4886025119029199 * u;
    y[2] = 0.4886025119029199 * z;
    y[3] = 0.4886025119029199 * x;
    y[4] = 1.0925484305920792 * x * u;
    y[5] = 1.0925484305920792 * u * z;
    y[6] = 0.3153915652525200 * (3.0 * z * z - 1.0);
    y[7] = 1.0925484305920792 * x * z;
    y[8] = 0.5462742152960396 * (x * x - u * u);
}
// Which band coefficient k belongs to, so it can be scaled by g^l.
inline int volShBand(int k) { return (k == 0) ? 0 : (k <= 3 ? 1 : 2); }

struct VolCache {
    Vec3  lo{0, 0, 0}, hi{0, 0, 0};
    int   nx = 0, ny = 0, nz = 0;
    double cellVol = 0.0;
    int   nSH = 1;              // 1 (isotropic, l = 0 only) or kVolShN
    double gHG = 0.0;           // the HG anisotropy this grid was built for
    int   med = -1;             // the medium index this grid serves
    // RAINBOW (zonal, per-CIE-channel) MODE. A Henyey-Greenstein medium has one scalar phase, so
    // the beam's CIE triple is a constant that can be folded into the coefficients and the kernel
    // is the single number g^l. A RAINBOW does not work that way: its colour is a function of the
    // SCATTERING ANGLE, so the CIE belongs in the KERNEL and the coefficients must stay scalar.
    // That is the structural reason the CIE-weighted cache could not represent a bow at all, and
    // it is what `bowMode` switches.
    bool  bowMode = false;
    Vec3  kMom[3];              // Legendre moment of the bow kernel, per band, per CIE channel
    int   emPick = -1;          // the emitter whose bow LUT kMom came from
    std::vector<float> sh;      // nSH * nCh per cell
    int   nCh = 3;              // 3 = CIE per coefficient (HG), 1 = scalar (bow)
    bool  ready = false;

    int idx(int ix, int iy, int iz) const { return ((iz * ny) + iy) * nx + ix; }

    // ONE predicate, used by build() AND by the deposit split's erase. They must agree exactly:
    // erasing a beam the cache did not splat deletes its energy outright, which is how the split
    // lost 72 % of a cloud once already.
    bool eligible(const PhotonBeam& b, int medOnly, int minOrder) const {
        if (b.med != medOnly) return false;
        if (b.order == kBeamOrderUnknown || (int)b.order < minOrder) return false;
        // In bow mode only the GATHER-TIME FOLD beams qualify (`achro == 2`): their spectral
        // integral is precisely what the bow LUT tabulates, and it is decidable only once the
        // scattering angle is known -- which is exactly what the convolution supplies. A
        // monochromatic or bundled beam carries its own lambda and would need its own kernel.
        if (bowMode && !(b.achro == 2 && (int)b.emIdx == emPick)) return false;
        return true;
    }
    bool eligible(const PhotonBeam& b) const { return eligible(b, med, 2); }

    // The CIE triple of one chord, duplicating BeamMap::build's rule. Needed because the
    // DEPOSIT SPLIT builds this cache from the RAW pre-split chords, before `bm.cie` exists.
    static Vec3 chordCie(const PhotonBeam& b) {
        if (b.achro) return Vec3(b.cieA[0], b.cieA[1], b.cieA[2]);
        const double lam = (double)b.lambda;
        return Vec3(cieX(lam), cieY(lam), cieZ(lam));
    }

    // `sigmaAt(p)` is the TOTAL extinction at p, summed over every enabled medium, exactly as
    // the gather's marched transmittance sees it. Passing a function rather than a constant is
    // what lets a HETEROGENEOUS medium in: the splat no longer assumes exp(-sigma_t s), it
    // integrates the real optical depth along the chord it is already walking.
    // `medOnly` is the medium index this grid serves. ONE GRID PER MEDIUM, because a medium is
    // the unit that owns a phase function, an extinction and a density field -- a shared grid
    // has to pick one of each and is then wrong for every other medium present. It is also what
    // lets a scene be PARTIALLY cached: `gallery_rain` can cache its HG cloud while its
    // `phase rainbow` curtain stays as real beams, which a single grid could not express.
    // `bowMom` non-null selects RAINBOW mode: scalar coefficients convolved with a per-CIE-channel
    // zonal kernel, restricted to the gather-time-fold beams of emitter `emitter`. Null selects
    // the Henyey-Greenstein path (CIE-weighted coefficients, g^l kernel).
    template <class SigmaFn>
    void build(const BeamMap& bm, int res, int minOrder, bool ok, const SigmaFn& sigmaAt,
               double g, int medOnly, const Vec3* bowMom = nullptr, int emitter = -1) {
        ready = false;
        // Every early-out says WHY under FTRACE_VOLCACHE_DIAG. A cache that silently declines to
        // build is indistinguishable from one that built and did nothing, and telling those apart
        // by bisection costs far more than this does.
        const bool diag = std::getenv("FTRACE_VOLCACHE_DIAG") != nullptr;
        if (!ok) {
            if (diag) std::fprintf(stderr, "[vcdiag] med %d: gate refused\n", medOnly);
            return;
        }
        if (bm.empty() || bm.nEmitted <= 0 || res < 2) {
            if (diag) std::fprintf(stderr, "[vcdiag] med %d: empty=%d nEmitted=%lld res=%d\n",
                                   medOnly, (int)bm.empty(), (long long)bm.nEmitted, res);
            return;
        }
        med = medOnly;
        // Bound the grid to THIS medium's own cacheable chords, so its cells are not stretched
        // across a scene-sized box by a medium that happens to be far away.
        Aabb box;
        bool any = false;
        for (size_t i = 0; i < bm.beams.size(); ++i) {
            const PhotonBeam& b = bm.beams[i];
            if (!eligible(b, medOnly, minOrder)) continue;
            box.expand(b.o + b.d * (double)b.s0);
            box.expand(b.o + b.d * ((double)b.s0 + (double)b.len));
            any = true;
        }
        if (!any) {
            if (diag) {
                long long nMed = 0, nOrd = 0, nAch = 0;
                for (size_t i = 0; i < bm.beams.size(); ++i) {
                    const PhotonBeam& q = bm.beams[i];
                    if (q.med == medOnly) ++nMed;
                    if (q.med == medOnly && q.order != kBeamOrderUnknown &&
                        (int)q.order >= minOrder) ++nOrd;
                    if (q.med == medOnly && q.achro == 2) ++nAch;
                }
                std::fprintf(stderr, "[vcdiag] med %d: NO eligible chords of %lld beams "
                             "(this medium %lld, order>=%d %lld, achro2 %lld, bowMode %d)\n",
                             medOnly, (long long)bm.beams.size(), nMed, minOrder, nOrd, nAch,
                             (int)(bowMom != nullptr));
            }
            return;
        }
        if (!(box.hi.x > box.lo.x)) return;
        const Vec3 pad = (box.hi - box.lo) * 0.01 + Vec3{1e-6, 1e-6, 1e-6};
        lo = box.lo - pad; hi = box.hi + pad;
        nx = ny = nz = res;
        // DIRECTIONAL BINS ARE ON BY DEFAULT FOR ANISOTROPIC MEDIA (0.311.0). They were opt-in
        // through 0.310.x on the strength of an enclosure measurement that does not transfer to
        // a collimated backlight: on gallery_rain's sun-lit cloud the scalar reconstruction is
        // 7% dark. See the header note "THE DIRECTIONAL BINS, MEASURED TWICE".
        //   FTRACE_VOLCACHE_SH unset : bins whenever g != 0 (at g = 0 they ARE the scalar cache)
        //   FTRACE_VOLCACHE_SH=0     : the scalar (l = 0) reconstruction regardless of g
        //   FTRACE_VOLCACHE_SH=1     : the bins (same as unset; kept for the old spelling)
        static const int shMode = [] {
            const char* e = std::getenv("FTRACE_VOLCACHE_SH");
            if (!e || !*e) return -1;
            return std::atoi(e) != 0 ? 1 : 0;
        }();
        bowMode = (bowMom != nullptr);
        emPick  = emitter;
        if (bowMode) {
            for (int l = 0; l < 3; ++l) kMom[l] = bowMom[l];
            // A bow is a SHARP angular feature, so its kernel keeps real weight well past band 2
            // and the directional bins are not optional here the way they were for HG. Always
            // full order in this mode.
            gHG = 0.0;
            nSH = kVolShN;
        } else {
            gHG = (shMode == 0) ? 0.0 : g;
            nSH = (gHG == 0.0) ? 1 : kVolShN;
        }
        const Vec3 ext = hi - lo;
        cellVol = (ext.x / nx) * (ext.y / ny) * (ext.z / nz);
        nCh = bowMode ? 1 : 3;
        sh.assign((size_t)nx * ny * nz * nSH * nCh, 0.0f);

        const double step = std::min(ext.x / nx, std::min(ext.y / ny, ext.z / nz)) * 0.5;
        long long nSplat = 0;
        const double invN = 1.0 / (double)bm.nEmitted;
        double Y[kVolShN];
        for (size_t i = 0; i < bm.beams.size(); ++i) {
            const PhotonBeam& b = bm.beams[i];
            if (!eligible(b, medOnly, minOrder)) continue;
            ++nSplat;
            const double len = (double)b.len;
            if (!(len > 0.0)) continue;
            const int ns = (int)std::ceil(len / step);
            const double ds = len / (double)ns;
            const Vec3 cie = (i < bm.cie.size()) ? bm.cie[i] : chordCie(b);
            // OPTICAL DEPTH IS INTEGRATED, NOT ASSUMED. Transmittance is measured from the
            // PARENT chord's origin, so a sub-beam (s0 > 0) must first march the prefix it did
            // not splat -- otherwise every sub-beam past the first is too bright by exactly the
            // attenuation it skipped. The deposit-split path always has s0 == 0; the query-only
            // path does not, and that is the one this prefix exists for.
            const double sStart = (double)b.s0;
            double tau = 0.0;
            if (sStart > 0.0) {
                const int np = (int)std::ceil(sStart / step);
                const double dp = sStart / (double)np;
                for (int j = 0; j < np; ++j)
                    tau += sigmaAt(b.o + b.d * (dp * (j + 0.5))) * dp;
            }
            // The photon's propagation direction IS the direction its radiance travels in, and
            // it is exactly what the gather feeds the phase function (beamgather.h evaluates
            // phaseValue(dot(b.d, -dc))). Projected once per chord: the direction is constant
            // along a straight chord, which is the whole reason a beam is storable at all.
            volShBasis(b.d, Y);
            for (int kk = 0; kk < ns; ++kk) {
                const Vec3 p = b.o + b.d * ((double)b.s0 + ds * (kk + 0.5));
                const int ix = (int)((p.x - lo.x) / ext.x * nx);
                const int iy = (int)((p.y - lo.y) / ext.y * ny);
                const int iz = (int)((p.z - lo.z) / ext.z * nz);
                const bool inGrid = !(ix < 0 || iy < 0 || iz < 0 ||
                                      ix >= nx || iy >= ny || iz >= nz);
                // Midpoint rule: one extinction evaluation per step, with the transmittance
                // taken at the sample point (half a step in) and tau carried to the step end.
                const double sig = sigmaAt(p);
                const double atten = std::exp(-(tau + sig * ds * 0.5));
                tau += sig * ds;
                if (!inGrid) continue;   // tau has already advanced: depth accrues outside too
                const double wgt = (double)b.power * atten * ds * invN / cellVol;
                float* c = &sh[(size_t)idx(ix, iy, iz) * nSH * nCh];
                if (bowMode) {
                    // SCALAR photon-direction density: the colour is in the kernel, not here.
                    for (int k = 0; k < nSH; ++k) c[k] += (float)(wgt * Y[k]);
                } else {
                    for (int k = 0; k < nSH; ++k) {
                        const double wy = wgt * Y[k];
                        c[k * 3 + 0] += (float)(cie.x * wy);
                        c[k * 3 + 1] += (float)(cie.y * wy);
                        c[k * 3 + 2] += (float)(cie.z * wy);
                    }
                }
            }
        }
        ready = true;
        if (std::getenv("FTRACE_VOLCACHE_DIAG"))
            std::fprintf(stderr, "[vcdiag] splatted %lld of %lld beams, nEmitted %lld, "
                         "totalY %.6g, nSH %d, g %.3f, medium %d\n",
                         (long long)nSplat, (long long)bm.beams.size(),
                         (long long)bm.nEmitted, totalY(), nSH, gHG, med);
    }

    // The [t0, t1] span of this grid's box along the ray, clipped to [tLo, tHi]. The march MUST
    // be bounded by this and not by the camera ray's length: a ray that hits nothing is handed
    // tMax = 1e30 (photonmap_render.h), so a march that divides tMax into a fixed number of
    // steps puts every sample ~1e28 away and contributes NOTHING. That is not hypothetical --
    // it silently cost a thick cloud 72 % of its energy on gallery_rain, and stayed invisible
    // through every earlier test because those scenes are closed boxes where every ray hits a
    // wall. Clipping to the grid also concentrates the samples where the medium actually is,
    // which is strictly better sampling for the same cost.
    bool raySpan(const Vec3& o, const Vec3& d, double tLo, double tHi,
                 double& t0, double& t1) const {
        if (!ready) return false;
        t0 = tLo; t1 = tHi;
        const double oa[3] = {o.x, o.y, o.z}, da[3] = {d.x, d.y, d.z};
        const double la[3] = {lo.x, lo.y, lo.z}, ha[3] = {hi.x, hi.y, hi.z};
        for (int a = 0; a < 3; ++a) {
            if (std::fabs(da[a]) < 1e-12) {
                if (oa[a] < la[a] || oa[a] > ha[a]) return false;
                continue;
            }
            double ta = (la[a] - oa[a]) / da[a], tb = (ha[a] - oa[a]) / da[a];
            if (ta > tb) { const double tmp = ta; ta = tb; tb = tmp; }
            if (ta > t0) t0 = ta;
            if (tb < t1) t1 = tb;
            if (t0 >= t1) return false;
        }
        return t1 > t0;
    }

    // In-scattered radiance toward `wOut`, WITHOUT sigma_s (the caller applies it with the
    // local density). `wOut` must be the direction the scattered light travels toward the
    // camera, i.e. -dc, matching the gather's phaseValue(dot(b.d, -dc)).
    bool inScatter(const Vec3& p, const Vec3& wOut, Vec3& out) const {
        if (!ready) return false;
        const Vec3 ext = hi - lo;
        const int ix = (int)((p.x - lo.x) / ext.x * nx);
        const int iy = (int)((p.y - lo.y) / ext.y * ny);
        const int iz = (int)((p.z - lo.z) / ext.z * nz);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) return false;
        const float* c = &sh[(size_t)idx(ix, iy, iz) * nSH * nCh];
        double Y[kVolShN];
        volShBasis(wOut, Y);
        double r[3] = {0.0, 0.0, 0.0};
        if (bowMode) {
            // Same convolution, but the zonal kernel differs PER CIE CHANNEL, so the moment
            // multiplies on the way out instead of being folded into the coefficients.
            for (int k = 0; k < nSH; ++k) {
                const Vec3& m = kMom[volShBand(k)];
                const double a = (double)c[k] * Y[k];
                r[0] += m.x * a; r[1] += m.y * a; r[2] += m.z * a;
            }
        } else {
            // The per-band Legendre moment of Henyey-Greenstein is exactly g^l (Funk-Hecke).
            const double gp[3] = {1.0, gHG, gHG * gHG};
            for (int k = 0; k < nSH; ++k) {
                const double w = gp[volShBand(k)] * Y[k];
                r[0] += (double)c[k * 3 + 0] * w;
                r[1] += (double)c[k * 3 + 1] * w;
                r[2] += (double)c[k * 3 + 2] * w;
            }
        }
        // SH truncation rings, and a NEGATIVE in-scattered radiance is unphysical -- it would
        // subtract energy from the march. Clamp. Windowing the series would remove the ringing
        // instead of bounding it, at the cost of a band of angular sharpness; not worth it at
        // order 2, where the clamp fires on the tail of an already-smooth reconstruction.
        out = Vec3{r[0] > 0.0 ? r[0] : 0.0, r[1] > 0.0 ? r[1] : 0.0, r[2] > 0.0 ? r[2] : 0.0};
        return true;
    }

    // Total CIE-Y energy in the grid. Only the l = 0 band carries net energy -- every higher
    // band integrates to zero over the sphere -- so this reads c_00 * Y_00 * 4 pi per cell.
    double totalY() const {
        if (!ready || nSH < 1) return 0.0;
        const double k = 0.2820947917738781 * 4.0 * 3.14159265358979323846;
        double s = 0.0;
        const size_t stride = (size_t)nSH * nCh;
        // Bow mode stores a scalar, so its l=0 coefficient IS the density; HG mode stores CIE and
        // the Y channel is offset 1. Weighted by the kernel's l=0 moment either way.
        const size_t off = bowMode ? 0 : 1;
        const double kc = bowMode ? kMom[0].y : 1.0;
        for (size_t i = 0; i + stride <= sh.size(); i += stride) s += (double)sh[i + off];
        return s * k * kc * cellVol;
    }
};

// `FTRACE_VOLCACHE_SPLIT=1` turns the prototype from a QUERY-side skip into a DEPOSIT SPLIT:
// the cache is built from the raw chords BEFORE BeamMap::build, and those chords are then
// ERASED, so the SAH split, the CIE table, the box array and the BVH all see only the
// order < 2 remainder. That is where the gain is -- a query-side skip still pays for the beams
// it declines to shade.
// THE SPLIT IS HOST-ONLY, AND THIS FLAG IS WHY IT HAS TO BE ENFORCED RATHER THAN ASSUMED.
// `volCacheSplit` runs from `buildBeamMap`, which is host code that executes whatever backend
// will gather; the MARCH that puts the energy back lives in `beamgather.h` and has no device
// twin (zero volcache symbols in render_cuda.cu). So on a `-device gpu` render the split would
// erase the order >= 2 chords and nothing would ever add them back -- a silent 72 % loss of a
// thick cloud, measured on gallery_rain before this guard existed. Default true; the CUDA
// gather branches clear it.
inline bool& volCacheHostGather() { static bool b = true; return b; }

inline bool volCacheSplitEnabled() {
    static const bool b = [] {
        const char* e = std::getenv("FTRACE_VOLCACHE_SPLIT");
        return e && *e && std::atoi(e) != 0;
    }();
    return b;
}

inline int volCacheRes() {
    static const int n = [] {
        const char* e = std::getenv("FTRACE_VOLCACHE");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return n;
}
