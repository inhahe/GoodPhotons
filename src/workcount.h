// workcount.h -- deterministic work counts for the light-side refresh (modes J and M; mode J's on
// the device too, which counts its camera pass itself: cudaLastBdptWork).
//
// An epoch of a fixed -spp render lasts as many camera samples as keep its light side's rebuild
// near -beamrefresh's share of the time -- the rule the clock applies under a -time/-noise/-forever
// budget -- but measured in counts, which a machine's speed cannot change, so the render repeats
// exactly, at any thread count (known-issues J-CPU-NONDET, MODEM-CPU-NONDET, GPU-J-CLOCK). One unit: a vertex traced (a light subpath's or a camera sample's), a
// segment of a mode-M camera path traced, a connection evaluated, a beam or photon a camera-side
// gather weighs; a photon-map query's candidate is a sixteenth of one (tlCandidates). A rebuild
// adds kJWorkRebuildFixed, and each map it builds kJWorkPerBeamLog per item times the log of the
// map's size. Both sides accumulate per thread and add once per pass. CountedEpochs turns the
// counts into epoch lengths.
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>

namespace bdpt {

struct JWork {
    std::atomic<long long> light{0};   // the light passes' (the renderers add each map's build)
    std::atomic<long long> cam{0};     // the camera passes'
    std::atomic<long long> camCand{0}; // ...their photon-map query candidates, a sixteenth each
    // The camera side since `cam0` / `cand0` were read: the candidates divided once, over the
    // whole span, so no division of the frame among threads moves it.
    long long camSince(long long cam0, long long cand0) const {
        return (cam.load(std::memory_order_relaxed) - cam0) +
               (camCand.load(std::memory_order_relaxed) - cand0) / 16;
    }
};
inline JWork& jWork() { static JWork w; return w; }

// A camera thread's count where no weight object carries a pointer to one (mode M's camera
// segments and beam gathers): CamWorkFlush adds it to jWork().cam when the thread's pass ends. A
// photon map's query candidates go to tlCandidates and camCand instead, at a sixteenth of a unit
// each (JWork::camSince): a candidate is one distance test, and mode M's photon trace measured
// some 18x a candidate's cost per unit on scenes/cornell.ftsl (FTRACE_JWORK=1), where a beam hit
// costs about what a light unit does.
inline long long& tlWork() { thread_local long long w = 0; return w; }
inline long long& tlCandidates() { thread_local long long c = 0; return c; }
// A pass's count kept locally and added to tlWork once, as the pass's scope ends, whichever way.
struct LocalWork {
    long long n = 0;
    ~LocalWork() { tlWork() += n; }
};
struct CamWorkFlush {
    ~CamWorkFlush() {
        jWork().cam.fetch_add(tlWork(), std::memory_order_relaxed);
        jWork().camCand.fetch_add(tlCandidates(), std::memory_order_relaxed);
        tlWork() = 0;
        tlCandidates() = 0;
    }
};

// Fitted against the clock (FTRACE_JWORK=1, the CPU at 32 threads) on hetmed's small map, a map of
// 100 000 beams and _fog_cornell's of 10 million, whose rebuilds took 0.11 s, 0.26-0.5 s and 1.8 s:
// a rebuild costs about 0.1 s of fixed start-up (its passes' threads, the BVH builder) plus the
// beam build, against a camera unit's 0.6-2 e-7 s. Off by up to 2-3x on any one scene -- the
// camera units' own cost moves that much with the medium -- which moves the light side's share
// between about 5 and 20 % at the default -beamrefresh 0.10.
constexpr long long kJWorkRebuildFixed = 500000;
// The device's camera pass in these units (GPU-J-CLOCK): a device unit (cudaLastBdptWork: a vertex,
// a connection it could evaluate, a photon or beam hit a merge weighs) weighs kJWorkDeviceUnit of
// the CPU's, and each renderBdptCuda call -- one an epoch -- adds kJWorkDeviceCall to the light
// side, its fixed cost (the maps' upload and grid, the queues). Fitted like the CPU's
// (FTRACE_JWORK=1, scraps/_jwork_fit.py), on an RTX 4090 against 32 CPU threads: a device unit
// against a light unit read 1.7 (_fog_cornell), 2.9 (_fog_thick), 0.93 (gallery_rain) and 6.8
// (check_jchroma's heterogeneous fog, whose vertices cost more than their count), a call's setup
// 10-52 ms, 0.8-15e5 light units -- geometric means taken, so the light side lands within a few
// times of its share, as on the CPU. A card faster or slower against its CPU moves the share,
// never the repeatability.
constexpr double    kJWorkDeviceUnit = 2.0;
constexpr long long kJWorkDeviceCall = 500000;
// A map's build in JWork units: its sort (and a beam map's SAH BVH), n log n in its items.
constexpr double kJWorkPerBeamLog = 0.17;
inline long long jWorkBeamBuild(size_t nBeams) {
    return nBeams ? (long long)(kJWorkPerBeamLog * (double)nBeams * std::log2((double)nBeams + 1.0))
                  : 0;
}
// ...and where the device builds the tree (an LBVH: 7.7 M sub-beams in 28 ms on an RTX 4090,
// some 0.4 units a beam) and the host only converts and uploads the map: a unit a beam.
inline long long jWorkBeamBuildDevice(size_t nBeams) { return (long long)nBeams; }

// The counted refresh rule: how many samples each render call of a counted refresh loop takes.
// A light side of `units` buys the samples whose camera work keeps it at -beamrefresh's share of
// the two, a sample priced at the camera work of every sample rendered so far. Epoch 0's call is
// asked for every sample left; its first chunk, one sample (SppProgress::span), prices a sample,
// and first() cuts the call to its own build's share (`build0`), as the clock would let it run on
// its maps -- in one call, not a second that would set the maps up again. Each later epoch runs its
// rebuild's share. A loop asks next() before each call, first() from the unpriced call's span,
// and tells rendered() what each call did.
struct CountedEpochs {
    long long target = 0;   // the frame's samples
    double share = 0.1;     // -beamrefresh
    long long build0 = 0;   // epoch 0's light side, in units (0: epoch 0 stops at its first chunk)

    // Whether a sample has been priced. Until then the next call is epoch 0's, which prices one.
    bool priced() const { return camSpp > 0; }
    // The next call's samples, `done` of the target rendered and `rebuild` the units of the light
    // side drawn for it (unread for epoch 0's call, which may take every sample left).
    long long next(long long rebuild, long long done) const {
        return priced() ? lenFor(rebuild, done, camWork, camSpp) : target - done;
    }
    // Epoch 0's call, `done` of the target rendered before it, has rendered its first `spp`
    // samples for `units` of camera work: the samples it renders in all, those at least.
    long long first(long long units, long long spp, long long done) const {
        return std::max(spp, lenFor(build0, done, units, spp));
    }
    // A call rendered `spp` samples (none: the loop ends) for `units` of camera work.
    void rendered(long long units, long long spp) {
        if (spp <= 0) return;
        camWork += units;
        camSpp  += spp;
    }

private:
    long long camWork = 0, camSpp = 0;   // camera units so far, and the samples they bought
    // The samples `units` of light side buy, `work` the camera units of `spp` samples: at least
    // one and at most what is left, clamped as a double, since a tiny share makes the quotient huge.
    long long lenFor(long long units, long long done, long long work, long long spp) const {
        const long long left = target - done;
        if (spp <= 0 || work <= 0) return std::min(1LL, left);
        const double q = (double)units / (share * (double)work / (double)spp);
        if (!(q < (double)left)) return left;
        return std::max(1LL, (long long)std::ceil(q));
    }
};

}  // namespace bdpt
