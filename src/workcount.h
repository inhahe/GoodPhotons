// workcount.h -- deterministic work counts for the CPU's light-side refresh (modes J and M).
//
// An epoch of a fixed -spp render on the CPU lasts as many camera samples as keep its light side's
// rebuild near -beamrefresh's share of the time -- the rule the clock applies on the device and
// under a -time/-noise/-forever budget -- but measured in counts, which a machine's speed cannot
// change, so the render repeats exactly, at any thread count (known-issues J-CPU-NONDET,
// MODEM-CPU-NONDET). One unit: a vertex traced (a light subpath's or a camera sample's), a
// connection evaluated, a beam, photon or photon-map candidate a camera-side gather weighs; a
// rebuild adds kJWorkRebuildFixed, and each map it builds kJWorkPerBeamLog per item times the log
// of the map's size. Both sides accumulate per thread and add once per pass.
#pragma once
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

// A camera thread's count where no weight object carries a pointer to one (mode M's camera rays
// and beam gathers): CamWorkFlush adds it to jWork().cam when the thread's pass ends. A photon
// map's query candidates go to tlCandidates and camCand instead, at a sixteenth of a unit each
// (JWork::camSince): a candidate
// is one distance test, and mode M's photon trace measured some 18x a candidate's cost per unit
// on scenes/cornell.ftsl (FTRACE_JWORK=1), where a beam hit costs about what a light unit does.
inline long long& tlWork() { thread_local long long w = 0; return w; }
inline long long& tlCandidates() { thread_local long long c = 0; return c; }
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
// A map's build in JWork units: its sort (and a beam map's SAH BVH), n log n in its items.
constexpr double kJWorkPerBeamLog = 0.17;
inline long long jWorkBeamBuild(size_t nBeams) {
    return nBeams ? (long long)(kJWorkPerBeamLog * (double)nBeams * std::log2((double)nBeams + 1.0))
                  : 0;
}

}  // namespace bdpt
