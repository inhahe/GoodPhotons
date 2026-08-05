// parallel.h — a minimal work-stealing parallel_for for LOAD-TIME loops.
//
// The renderers each own a bespoke thread pool tuned to their traversal (raster.h keeps
// a persistent pool, photonmap.h bands by photon block, isomesh.h splits the lattice),
// and none of those is reusable from a loader. This header covers the other case: a
// one-shot, embarrassingly parallel pass over an array during scene setup — per-texel
// spectral upsampling, per-texel envmap integration — where the loop is pure, the
// iterations are independent, and the only thing wanted is "use the cores".
//
// Chunks are handed out from a single atomic cursor rather than pre-sliced, because the
// per-item cost of these loops is wildly non-uniform (a Gauss-Newton fit converges in 3
// iterations for a grey texel and burns all 40 for a saturated one), so a static split
// leaves cores idle at the tail. `grain` bounds the fixed cost: below it the call runs
// serially on the caller's thread and spawns nothing.
#pragma once
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>

namespace ft {

// Run fn(i) for i in [0, n), in parallel when that is worth the thread setup.
// `grain` is the chunk size handed to a worker at a time AND the serial cutoff: fewer
// than 2*grain items runs inline. fn must be safe to call concurrently for distinct i.
template <class Fn>
inline void parallelFor(size_t n, size_t grain, Fn&& fn) {
    if (grain == 0) grain = 1;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    if (n < 2 * grain || hw < 2) {                 // too small to be worth a thread
        for (size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    unsigned nt = (unsigned)std::min<size_t>(hw, (n + grain - 1) / grain);
    std::atomic<size_t> cursor{0};
    std::vector<std::thread> pool;
    pool.reserve(nt - 1);
    auto worker = [&] {
        for (;;) {
            size_t b = cursor.fetch_add(grain, std::memory_order_relaxed);
            if (b >= n) return;
            size_t e = std::min(n, b + grain);
            for (size_t i = b; i < e; ++i) fn(i);
        }
    };
    for (unsigned t = 0; t + 1 < nt; ++t) pool.emplace_back(worker);
    worker();                                       // the caller is a worker too
    for (auto& t : pool) t.join();
}

} // namespace ft
