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
// WHY THE DIRECTIONAL BINS ARE OFF BY DEFAULT, HAVING BEEN BUILT AND MEASURED. The SH machinery
// below is correct -- it reduces to the scalar cache exactly at g = 0, and it is energy-correct
// to 0.8% at g = 0.5 -- and it buys NOTHING, which is the useful result. Measured on
// `_beams_ms` at 96^2 against the uncached render, per-pixel median error:
//
//     medium          SH order 2      l = 0 only
//     g = 0.5           3.03 %          1.88 %
//     g = 0.85          2.31 %          2.13 %
//
// The l = 0 reconstruction is as good or BETTER at both, including a strongly forward medium.
// The reason is that this cache only ever holds the order >= 2 component, and by the second
// scattering event diffusion has made the radiance field nearly isotropic -- the anisotropy is
// spent in the FIRST scattering, which stays a real beam and is never cached. So the l >= 1
// bands carry little signal and a full share of estimator noise, and clamping their ringing to
// non-negative biases the result bright (+0.8% against the scalar path's -1.0%).
//
// The real fix for anisotropic media was therefore not directional bins at all: it was to STOP
// REFUSING THEM, because the scalar cache was already adequate. `FTRACE_VOLCACHE_SH=1` keeps
// the SH path available -- it would matter for a cache that held order 1, where the anisotropy
// is real -- at 9x the memory.
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
    std::vector<float> sh;      // nSH * 3 per cell, CIE XYZ per coefficient
    bool  ready = false;

    int idx(int ix, int iy, int iz) const { return ((iz * ny) + iy) * nx + ix; }

    // The CIE triple of one chord, duplicating BeamMap::build's rule. Needed because the
    // DEPOSIT SPLIT builds this cache from the RAW pre-split chords, before `bm.cie` exists.
    static Vec3 chordCie(const PhotonBeam& b) {
        if (b.achro) return Vec3(b.cieA[0], b.cieA[1], b.cieA[2]);
        const double lam = (double)b.lambda;
        return Vec3(cieX(lam), cieY(lam), cieZ(lam));
    }

    void build(const BeamMap& bm, int res, int minOrder, bool ok, double sigmaT, double g) {
        ready = false;
        if (!ok) return;
        if (bm.empty() || bm.nEmitted <= 0 || res < 2) return;
        Aabb box;
        for (size_t i = 0; i < bm.beams.size(); ++i) {
            const PhotonBeam& b = bm.beams[i];
            box.expand(b.o + b.d * (double)b.s0);
            box.expand(b.o + b.d * ((double)b.s0 + (double)b.len));
        }
        if (!(box.hi.x > box.lo.x)) return;
        const Vec3 pad = (box.hi - box.lo) * 0.01 + Vec3{1e-6, 1e-6, 1e-6};
        lo = box.lo - pad; hi = box.hi + pad;
        nx = ny = nz = res;
        // DIRECTIONAL BINS ARE OPT-IN, AND THE MEASUREMENT SAYS THEY ARE NOT WORTH IT.
        // `FTRACE_VOLCACHE_SH=1` enables them; the default is the l = 0 reconstruction even on
        // an anisotropic medium. See the header note "WHY THE DIRECTIONAL BINS ARE OFF".
        static const bool useSH = [] {
            const char* e = std::getenv("FTRACE_VOLCACHE_SH");
            return e && *e && std::atoi(e) != 0;
        }();
        gHG = useSH ? g : 0.0;
        nSH = (gHG == 0.0) ? 1 : kVolShN;
        const Vec3 ext = hi - lo;
        cellVol = (ext.x / nx) * (ext.y / ny) * (ext.z / nz);
        sh.assign((size_t)nx * ny * nz * nSH * 3, 0.0f);

        const double step = std::min(ext.x / nx, std::min(ext.y / ny, ext.z / nz)) * 0.5;
        long long nSplat = 0;
        const double invN = 1.0 / (double)bm.nEmitted;
        double Y[kVolShN];
        for (size_t i = 0; i < bm.beams.size(); ++i) {
            const PhotonBeam& b = bm.beams[i];
            if (b.order == kBeamOrderUnknown || (int)b.order < minOrder) continue;
            ++nSplat;
            const double len = (double)b.len;
            if (!(len > 0.0)) continue;
            const int ns = (int)std::ceil(len / step);
            const double ds = len / (double)ns;
            const Vec3 cie = (i < bm.cie.size()) ? bm.cie[i] : chordCie(b);
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
                if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) continue;
                const double sAlong = (double)b.s0 + ds * (kk + 0.5);
                const double atten  = (sigmaT > 0.0) ? std::exp(-sigmaT * sAlong) : 1.0;
                const double wgt = (double)b.power * atten * ds * invN / cellVol;
                float* c = &sh[(size_t)idx(ix, iy, iz) * nSH * 3];
                for (int k = 0; k < nSH; ++k) {
                    const double wy = wgt * Y[k];
                    c[k * 3 + 0] += (float)(cie.x * wy);
                    c[k * 3 + 1] += (float)(cie.y * wy);
                    c[k * 3 + 2] += (float)(cie.z * wy);
                }
            }
        }
        ready = true;
        if (std::getenv("FTRACE_VOLCACHE_DIAG"))
            std::fprintf(stderr, "[vcdiag] splatted %lld of %lld beams, nEmitted %lld, "
                         "totalY %.6g, nSH %d, g %.3f\n",
                         (long long)nSplat, (long long)bm.beams.size(),
                         (long long)bm.nEmitted, totalY(), nSH, gHG);
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
        const float* c = &sh[(size_t)idx(ix, iy, iz) * nSH * 3];
        double Y[kVolShN];
        volShBasis(wOut, Y);
        // The per-band Legendre moment of Henyey-Greenstein is exactly g^l (Funk-Hecke).
        const double gp[3] = {1.0, gHG, gHG * gHG};
        double r[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < nSH; ++k) {
            const double w = gp[volShBand(k)] * Y[k];
            r[0] += (double)c[k * 3 + 0] * w;
            r[1] += (double)c[k * 3 + 1] * w;
            r[2] += (double)c[k * 3 + 2] * w;
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
        const size_t stride = (size_t)nSH * 3;
        for (size_t i = 0; i + stride <= sh.size(); i += stride) s += (double)sh[i + 1];
        return s * k * cellVol;
    }
};

// `FTRACE_VOLCACHE_SPLIT=1` turns the prototype from a QUERY-side skip into a DEPOSIT SPLIT:
// the cache is built from the raw chords BEFORE BeamMap::build, and those chords are then
// ERASED, so the SAH split, the CIE table, the box array and the BVH all see only the
// order < 2 remainder. That is where the gain is -- a query-side skip still pays for the beams
// it declines to shade.
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
