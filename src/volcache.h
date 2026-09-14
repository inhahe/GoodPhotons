#pragma once
// VOLCACHE PROTOTYPE — a volumetric fluence cache for the order >= 2 part of the beam gather.
//
// WHY A FLUENCE CACHE AND NOT A RADIANCE ONE. The surface `-radcache` stores E/pi keyed by
// (cell, normal bucket), which is the outgoing radiance of a Lambertian surface. A volume has
// no normal and its in-scattered radiance depends on the viewing angle through the phase
// function, so the quantity that CAN be cached direction-independently is the photon
// path-length density (fluence):
//
//     fluence(x) = sum over beams of (power * length inside the cell) / (nEmitted * cellVolume)
//
// The camera then reconstructs radiance from it exactly as the beam estimator does, by
// multiplying at march time rather than at build time:
//
//     L += sigma_s(x) * phase(theta) * fluence(x) * T(x) * dx
//
// The transmittance stays on the camera side, where it is view-dependent and must be.
//
// CORRECTION TO THIS FILE'S FIRST VERSION, because the distinction matters. That version
// claimed the phase function also stays on the camera side, so that "the approximation is
// spatial binning, not an isotropy assumption". **That is wrong.** `phase(theta)` is the angle
// between the INCOMING photon direction and the outgoing camera direction, and a scalar fluence
// has averaged the incoming directions away -- so only the isotropic average is recoverable
// from it. A scalar fluence cache *does* assume an isotropic phase function.
//
// It is therefore EXACT only for `g == 0` media, and `build()` refuses anything else rather
// than silently averaging. Every media scene in this repo happens to be `g = 0.0` (which is
// also the default), so the prototype is exact where it can be tested -- but the constraint is
// real and it shapes the feature: an anisotropic medium would need DIRECTIONAL bins (spherical
// harmonics, or a small discrete direction set) per cell, which multiplies the memory by the
// bin count and changes the cost model this entry has been pricing.
//
// This is a PROTOTYPE for a go/no-go decision, not the production cache. It has no confidence
// gate, no validation paths and no adaptive resolution -- the surface cache's machinery exists
// because those turned out to be necessary there, and whether they are necessary here is one of
// the things a prototype is supposed to find out.
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "linalg.h"
#include "photonbeams.h"

struct VolCache {
    Vec3  lo{0, 0, 0}, hi{0, 0, 0};
    int   nx = 0, ny = 0, nz = 0;
    double cellVol = 0.0;
    std::vector<float> xyz;              // 3 per cell: CIE-weighted fluence
    bool  ready = false;

    int idx(int ix, int iy, int iz) const { return ((iz * ny) + iy) * nx + ix; }

    // Splat every chord of order >= minOrder into the grid, by walking it in steps no longer
    // than a cell. A chord contributes `power * segment length` to each cell it crosses, which
    // is the discrete form of the path-length integral the fluence is defined by.
    // `gOK` must be true: the caller checks every medium this map touches has g == 0.
    // `sigmaT` is the medium's extinction; the splat must attenuate along the chord exactly
    // as the gather does (`mediaTransmittance(b.o, b.d, sBeam)`), or every cached cell is
    // too bright by the transmittance it skipped. Homogeneous only -- a heterogeneous
    // medium needs the marched transmittance and the caller gates on that too.
    // The CIE triple of one chord, duplicating BeamMap::build's rule. Needed because the
    // DEPOSIT SPLIT builds this cache from the RAW pre-split chords, before `bm.cie` exists --
    // and it has to, since the whole point of the split is that those chords never reach the
    // split/CIE/BVH stage at all.
    static Vec3 chordCie(const PhotonBeam& b) {
        if (b.achro) return Vec3(b.cieA[0], b.cieA[1], b.cieA[2]);
        const double lam = (double)b.lambda;
        return Vec3(cieX(lam), cieY(lam), cieZ(lam));
    }

    void build(const BeamMap& bm, int res, int minOrder, bool gOK, double sigmaT) {
        ready = false;
        if (!gOK) return;                      // anisotropic: a scalar cache cannot serve it
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
        const Vec3 ext = hi - lo;
        cellVol = (ext.x / nx) * (ext.y / ny) * (ext.z / nz);
        xyz.assign((size_t)nx * ny * nz * 3, 0.0f);

        const double step = std::min(ext.x / nx, std::min(ext.y / ny, ext.z / nz)) * 0.5;
        long long nSplat = 0;
        const double invN = 1.0 / (double)bm.nEmitted;
        for (size_t i = 0; i < bm.beams.size(); ++i) {
            const PhotonBeam& b = bm.beams[i];
            if (b.order == kBeamOrderUnknown || (int)b.order < minOrder) continue;
            ++nSplat;
            const double len = (double)b.len;
            if (!(len > 0.0)) continue;
            const int ns = (int)std::ceil(len / step);
            const double ds = len / (double)ns;
            const Vec3 cie = (i < bm.cie.size()) ? bm.cie[i] : chordCie(b);
            for (int k = 0; k < ns; ++k) {
                const Vec3 p = b.o + b.d * ((double)b.s0 + ds * (k + 0.5));
                const int ix = (int)((p.x - lo.x) / ext.x * nx);
                const int iy = (int)((p.y - lo.y) / ext.y * ny);
                const int iz = (int)((p.z - lo.z) / ext.z * nz);
                if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) continue;
                const double sAlong = (double)b.s0 + ds * (k + 0.5);
                const double atten  = (sigmaT > 0.0) ? std::exp(-sigmaT * sAlong) : 1.0;
                const double wgt = (double)b.power * atten * ds * invN / cellVol;
                const size_t o = (size_t)idx(ix, iy, iz) * 3;
                xyz[o + 0] += (float)(cie.x * wgt);
                xyz[o + 1] += (float)(cie.y * wgt);
                xyz[o + 2] += (float)(cie.z * wgt);
            }
        }
        ready = true;
        if (std::getenv("FTRACE_VOLCACHE_DIAG"))
            std::fprintf(stderr, "[vcdiag] splatted %lld of %lld beams, nEmitted %lld, "
                         "totalY %.6g, step %.4g, ext %.4g x %.4g x %.4g\n",
                         (long long)nSplat, (long long)bm.beams.size(),
                         (long long)bm.nEmitted, totalY(), step,
                         ext.x, ext.y, ext.z);
    }

    bool lookup(const Vec3& p, Vec3& out) const {
        if (!ready) return false;
        const Vec3 ext = hi - lo;
        const int ix = (int)((p.x - lo.x) / ext.x * nx);
        const int iy = (int)((p.y - lo.y) / ext.y * ny);
        const int iz = (int)((p.z - lo.z) / ext.z * nz);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) return false;
        const size_t o = (size_t)idx(ix, iy, iz) * 3;
        out = Vec3{(double)xyz[o], (double)xyz[o + 1], (double)xyz[o + 2]};
        return true;
    }

    // Total CIE-Y energy in the grid, for the conservation check that has to pass before any
    // of this is worth rendering with.
    double totalY() const {
        double s = 0.0;
        for (size_t i = 1; i < xyz.size(); i += 3) s += (double)xyz[i];
        return s * cellVol;
    }
};

// `FTRACE_VOLCACHE_SPLIT=1` turns the prototype from a QUERY-side skip into a DEPOSIT SPLIT.
//
// v0.299.0 built the cache from the finished beam map and then declined to traverse the
// order >= 2 beams -- but they were still deposited, split, CIE-folded, boxed and BVH-built, so
// the only cost it removed was traversal-and-shading, and it added a march. Measured: ~7 %,
// against a 51-58 % ceiling from dropping those chords outright (`-beams-minorder 2`).
//
// The split closes that gap by routing instead of skipping: the cache is built from the raw
// chords BEFORE `BeamMap::build`, and those chords are then ERASED, so the split, the CIE table,
// the box array and the SAH BVH all see only the order < 2 remainder. `-beams-minorder 2` is the
// same erase without the cache -- i.e. exactly this minus the energy -- which is what makes it a
// calibrated ceiling rather than a guess.
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
