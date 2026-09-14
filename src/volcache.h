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
// That keeps the phase function and the transmittance on the CAMERA side, where they are
// direction-dependent, and puts only the direction-independent part in the grid. The
// approximation is therefore *spatial binning*, not an isotropy assumption -- which matters,
// because known-issues.md records that the cached part carries 99.6 % of the energy at
// sigma_t 20, so an isotropy assumption there would be an assumption about the whole image.
//
// This is a PROTOTYPE for a go/no-go decision, not the production cache. It has no confidence
// gate, no validation paths and no adaptive resolution -- the surface cache's machinery exists
// because those turned out to be necessary there, and whether they are necessary here is one of
// the things a prototype is supposed to find out.
#include <vector>
#include <cmath>
#include <cstdlib>
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
    void build(const BeamMap& bm, int res, int minOrder) {
        ready = false;
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
        const double invN = 1.0 / (double)bm.nEmitted;
        for (size_t i = 0; i < bm.beams.size(); ++i) {
            const PhotonBeam& b = bm.beams[i];
            if (b.order == kBeamOrderUnknown || (int)b.order < minOrder) continue;
            const double len = (double)b.len;
            if (!(len > 0.0)) continue;
            const int ns = (int)std::ceil(len / step);
            const double ds = len / (double)ns;
            const Vec3 cie = bm.cie[i];
            for (int k = 0; k < ns; ++k) {
                const Vec3 p = b.o + b.d * ((double)b.s0 + ds * (k + 0.5));
                const int ix = (int)((p.x - lo.x) / ext.x * nx);
                const int iy = (int)((p.y - lo.y) / ext.y * ny);
                const int iz = (int)((p.z - lo.z) / ext.z * nz);
                if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) continue;
                const double wgt = (double)b.power * ds * invN / cellVol;
                const size_t o = (size_t)idx(ix, iy, iz) * 3;
                xyz[o + 0] += (float)(cie.x * wgt);
                xyz[o + 1] += (float)(cie.y * wgt);
                xyz[o + 2] += (float)(cie.z * wgt);
            }
        }
        ready = true;
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

inline int volCacheRes() {
    static const int n = [] {
        const char* e = std::getenv("FTRACE_VOLCACHE");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return n;
}
