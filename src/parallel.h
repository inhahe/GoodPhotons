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
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
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

// The DEVICE's copy of the stop flag: an int in mapped pinned host memory that GPU kernels
// read (render_cuda.cu, deviceStopFlag), registered once by the CUDA backend and unregistered
// before its context is torn down. Whoever raises or clears the stop (main.cpp's raiseStop /
// clearStop, a signal handler among them) writes it too, with a plain store, so the
// kernels see a stop the moment it is raised. Null when no device has asked for it.
inline std::atomic<volatile int*>& stopMirrorSlot() {
    static std::atomic<volatile int*> p{nullptr};
    return p;
}
// Writes to it in flight (writeStopMirror): clearing the slot waits for them, so a write that read
// the old pointer lands before its memory is freed. Lock-free counts, so a signal handler may write.
inline std::atomic<int>& stopMirrorWriters() {
    static std::atomic<int> n{0};
    return n;
}
inline void writeStopMirror(int v) {
    stopMirrorWriters().fetch_add(1);
    if (volatile int* m = stopMirrorSlot().load()) *m = v;
    stopMirrorWriters().fetch_sub(1);
}
inline void setStopMirror(volatile int* m) {
    stopMirrorSlot().store(m);
    if (!m)
        while (stopMirrorWriters().load() != 0) std::this_thread::yield();
}

// Run fn(i) for i in [0, n), in parallel when that is worth the thread setup.
// `grain` is the chunk size handed to a worker at a time AND the serial cutoff: fewer
// than 2*grain items runs inline. fn must be safe to call concurrently for distinct i.
// Returns true if every i ran; false if a stop was requested and the remaining range
// was abandoned (see above — the output is then partial and must be thrown away).
namespace detail {
// The loop on the caller's thread alone, still chunked to poll the stop flag (`stoppable`).
template <class Fn>
inline bool serialFor(size_t n, size_t grain, Fn& fn, bool stoppable = true) {
    for (size_t b = 0; b < n; b += grain) {
        if (stoppable && stopRequested()) return false;
        size_t e = std::min(n, b + grain);
        for (size_t i = b; i < e; ++i) fn(i);
    }
    return true;
}
// One worker's share: chunks claimed from `cursor` until the range or a stop runs out.
template <class Fn>
inline void cursorWorker(size_t n, size_t grain, Fn& fn, std::atomic<size_t>& cursor,
                         std::atomic<bool>& cancelled, bool stoppable = true) {
    for (;;) {
        // Poll before claiming work, so a stop drains the cursor: every worker
        // finishes at most the chunk it already holds and then returns, which
        // bounds the stop latency at one chunk rather than the whole range.
        if (stoppable && stopRequested()) { cancelled.store(true, std::memory_order_relaxed); return; }
        size_t b = cursor.fetch_add(grain, std::memory_order_relaxed);
        if (b >= n) return;
        size_t e = std::min(n, b + grain);
        for (size_t i = b; i < e; ++i) fn(i);
    }
}
template <class Fn>
inline bool threadedFor(size_t n, size_t grain, Fn& fn, bool stoppable) {
    if (grain == 0) grain = 1;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    if (n < 2 * grain || hw < 2) return serialFor(n, grain, fn, stoppable);   // not worth a thread
    unsigned nt = (unsigned)std::min<size_t>(hw, (n + grain - 1) / grain);
    std::atomic<size_t> cursor{0};
    std::atomic<bool> cancelled{false};
    std::vector<std::thread> pool;
    pool.reserve(nt - 1);
    auto worker = [&] { cursorWorker(n, grain, fn, cursor, cancelled, stoppable); };
    for (unsigned t = 0; t + 1 < nt; ++t) pool.emplace_back(worker);
    worker();                                       // the caller is a worker too
    for (auto& t : pool) t.join();
    return !cancelled.load(std::memory_order_relaxed);
}
}  // namespace detail

template <class Fn>
[[nodiscard]] inline bool parallelFor(size_t n, size_t grain, Fn&& fn) {
    return detail::threadedFor(n, grain, fn, /*stoppable=*/true);
}
// parallelFor that runs every i whatever a stop says: for a bounded pass whose partial result
// would be worse than finishing it (the sky a stopped render's last write still has to show).
template <class Fn>
inline void parallelForAll(size_t n, size_t grain, Fn&& fn) {
    (void)detail::threadedFor(n, grain, fn, /*stoppable=*/false);
}

// A persistent band pool: N workers that sleep on a condition variable and execute one
// broadcast job at a time (each worker gets its own index and derives its slice), so a pass
// costs one notify_all and N wakeups rather than N thread creations. For loops run over and
// over: the rasterizer's frame pipeline runs seven parallel passes back to back every frame
// (raster.h; with plain std::thread that was ~7*N creations per frame, several milliseconds,
// and scheduler jitter that showed directly in -raster-bench's min-to-median spread), and a
// solver runs a few each iteration (parallelFor below). Not nestable (run() must not be called
// from inside a job), nor to be run from two threads at once.
class BandPool {
public:
    explicit BandPool(int n) : nW_(n < 1 ? 1 : n) {
        workers_.reserve(nW_);
        for (int i = 0; i < nW_; ++i)
            workers_.emplace_back([this, i] {
                uint64_t seen = 0;
                std::unique_lock<std::mutex> lk(m_);
                for (;;) {
                    cvJob_.wait(lk, [&] { return quit_ || gen_ != seen; });
                    if (quit_) return;
                    seen = gen_;
                    const std::function<void(int)>* j = job_;
                    lk.unlock();
                    (*j)(i);
                    lk.lock();
                    if (--pending_ == 0) cvDone_.notify_one();
                }
            });
    }
    ~BandPool() {
        { std::lock_guard<std::mutex> lk(m_); quit_ = true; }
        cvJob_.notify_all();
        for (auto& t : workers_) t.join();
    }
    BandPool(const BandPool&) = delete;
    BandPool& operator=(const BandPool&) = delete;
    int size() const { return nW_; }
    // Run body(workerIndex) on every worker and wait for all of them.
    void run(const std::function<void(int)>& body) {
        std::unique_lock<std::mutex> lk(m_);
        job_ = &body;
        pending_ = nW_;
        ++gen_;
        cvJob_.notify_all();
        cvDone_.wait(lk, [&] { return pending_ == 0; });
        job_ = nullptr;
    }

private:
    std::vector<std::thread>       workers_;
    std::mutex                     m_;
    std::condition_variable        cvJob_, cvDone_;
    const std::function<void(int)>* job_ = nullptr;
    uint64_t                       gen_ = 0;
    int                            pending_ = 0;
    int                            nW_ = 1;
    bool                           quit_ = false;
};

// parallelFor on `pool`'s workers rather than on threads made for the call: the same chunks,
// cursor and stop polling, for a loop run so often that making and joining its threads would
// cost more than its work. The caller waits.
template <class Fn>
[[nodiscard]] inline bool parallelFor(BandPool& pool, size_t n, size_t grain, Fn&& fn) {
    if (grain == 0) grain = 1;
    if (n < 2 * grain || pool.size() < 2) return detail::serialFor(n, grain, fn);
    std::atomic<size_t> cursor{0};
    std::atomic<bool> cancelled{false};
    const std::function<void(int)> body = [&](int) { detail::cursorWorker(n, grain, fn, cursor, cancelled); };
    pool.run(body);
    return !cancelled.load(std::memory_order_relaxed);
}

} // namespace ft
