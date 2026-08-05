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
//
// The same cursor is also the program's cancellation seam: a render loop polls the
// stop flag between chunks, and so does this, which is what lets `ftrace -stop <pid>`
// land on a process that is still in scene load rather than being silently deferred
// until the first render chunk boundary. See setStopProbe below.
#pragma once
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>

namespace ft {

// --- Cooperative cancellation (`ftrace -stop <pid>`, Ctrl-C, window close) -------
// The stop flag itself lives in main.cpp: it is a `volatile sig_atomic_t` (the only
// state a signal handler may portably touch) with internal linkage, so this header
// cannot name it. Instead the program installs a PROBE once at startup, before the
// scene loads, and parallelFor calls it at the chunk cursor — once per `grain` items,
// never per item, so it is free even at grain 64.
//
// A cancelled parallelFor returns false and leaves its output range PARTIALLY
// written. Callers must treat that as a failed load and discard the result: half a
// coefficient table renders as garbage, which is worse than not rendering at all.
// Hence [[nodiscard]] — "abandon the load" is the only correct response.
using StopProbe = bool (*)();

inline std::atomic<StopProbe>& stopProbeSlot() {
    static std::atomic<StopProbe> p{nullptr};
    return p;
}
inline void setStopProbe(StopProbe p) { stopProbeSlot().store(p, std::memory_order_relaxed); }

// True once a clean stop has been requested. Also worth calling directly from the
// serial stretches of a loader (ftsl.h polls it between top-level blocks) so a stop
// lands between assets, not just inside a threaded pass.
inline bool stopRequested() {
    StopProbe p = stopProbeSlot().load(std::memory_order_relaxed);
    return p && p();
}

// Run fn(i) for i in [0, n), in parallel when that is worth the thread setup.
// `grain` is the chunk size handed to a worker at a time AND the serial cutoff: fewer
// than 2*grain items runs inline. fn must be safe to call concurrently for distinct i.
// Returns true if every i ran; false if a stop was requested and the remaining range
// was abandoned (see above — the output is then partial and must be thrown away).
template <class Fn>
[[nodiscard]] inline bool parallelFor(size_t n, size_t grain, Fn&& fn) {
    if (grain == 0) grain = 1;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    if (n < 2 * grain || hw < 2) {                 // too small to be worth a thread
        for (size_t b = 0; b < n; b += grain) {    // still chunked, to poll the stop flag
            if (stopRequested()) return false;
            size_t e = std::min(n, b + grain);
            for (size_t i = b; i < e; ++i) fn(i);
        }
        return true;
    }
    unsigned nt = (unsigned)std::min<size_t>(hw, (n + grain - 1) / grain);
    std::atomic<size_t> cursor{0};
    std::atomic<bool> cancelled{false};
    std::vector<std::thread> pool;
    pool.reserve(nt - 1);
    auto worker = [&] {
        for (;;) {
            // Poll before claiming work, so a stop drains the cursor: every worker
            // finishes at most the chunk it already holds and then returns, which
            // bounds the stop latency at one chunk rather than the whole range.
            if (stopRequested()) { cancelled.store(true, std::memory_order_relaxed); return; }
            size_t b = cursor.fetch_add(grain, std::memory_order_relaxed);
            if (b >= n) return;
            size_t e = std::min(n, b + grain);
            for (size_t i = b; i < e; ++i) fn(i);
        }
    };
    for (unsigned t = 0; t + 1 < nt; ++t) pool.emplace_back(worker);
    worker();                                       // the caller is a worker too
    for (auto& t : pool) t.join();
    return !cancelled.load(std::memory_order_relaxed);
}

} // namespace ft
