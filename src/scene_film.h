// Shared accumulation buffer for both the contact sensor (model A) and the
// pinhole camera film (model B). XYZ + hit count per pixel.
#pragma once
#include <vector>
#include "linalg.h"

struct Film {
    int resX = 256, resY = 256;
    std::vector<Vec3> xyz;     // accumulated XYZ per pixel
    std::vector<double> hits;  // photon contributions per pixel (diagnostics)
    void alloc() { xyz.assign((size_t)resX * resY, {}); hits.assign((size_t)resX * resY, 0.0); }
    void add(int px, int py, const Vec3& c) {
        size_t i = (size_t)py * resX + px;
        xyz[i] += c; hits[i] += 1.0;
    }
    // Merge another film of the same size (for per-thread accumulation).
    void merge(const Film& o) {
        for (size_t i = 0; i < xyz.size(); ++i) { xyz[i] += o.xyz[i]; hits[i] += o.hits[i]; }
    }
};
