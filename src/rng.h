// PCG32 RNG + sampling helpers. Small state -> trivially GPU-portable later.
#pragma once
#include <cstdint>
#include "linalg.h"

struct Pcg32 {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc = 0xda3e39cb94b95bdbULL;

    void seed(uint64_t seq, uint64_t s) {
        state = 0; inc = (seq << 1u) | 1u;
        next(); state += s; next();
    }
    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }
    // Uniform double in [0,1).
    double uniform() { return (next() >> 8) * (1.0 / 16777216.0); }
};

// SplitMix64 finalizer: bijective avalanche mix for turning structured indices
// (pixel number, absolute sample/photon index) into decorrelated seed material.
inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Seed `rng` for the k-th unit of work (absolute photon index, or a pixel-sample
// pair folded into one index) of the estimator family `salt`. Every CPU estimator
// seeds per WORK UNIT through this one helper, so a render's realization depends
// only on (unit index, salt) — never on how units are chunked into progressive
// batches or banded across threads. That is what makes fixed-parameter CPU renders
// bit-identical across -t values, -window/-time chunking, and resume boundaries.
inline void seedUnit(Pcg32& rng, uint64_t k, uint64_t salt) {
    rng.seed(mix64(k ^ salt), mix64(k + salt));
}

// Cosine-weighted hemisphere direction around unit normal n. pdf = cos/pi.
inline Vec3 cosineHemisphere(const Vec3& n, Pcg32& rng) {
    double u1 = rng.uniform(), u2 = rng.uniform();
    double r = std::sqrt(u1);
    double phi = 6.283185307179586 * u2;
    double lx = r * std::cos(phi), ly = r * std::sin(phi), lz = std::sqrt(1.0 - u1);
    Vec3 t, b; onb(n, t, b);
    return normalize(lx * t + ly * b + lz * n);
}
