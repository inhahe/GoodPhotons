// Progress hook for chunked samples-per-pixel renderers (mode R backward reference,
// mode D BDPT). These renderers accumulate radiance as a SUM over samples, so the
// requested spp can be split into chunks that accumulate into the same buffer without
// changing the result: brightness tracks the sample count and only graininess falls.
//
// A renderer that is handed a non-null SppProgress runs its work in chunks and, after
// each chunk, calls report() with the running SUM film and the spp completed so far.
// The host (main.cpp) uses that to rewrite the output image, print a status line / ANSI
// preview, and decide when to stop (wall-clock budget, noise target, or Ctrl-C). When
// report() returns true the renderer stops after the current chunk. `final` is true on
// the chunk that reaches the requested spp (the converged frame) so the host can do its
// exposure-anchored final write there.
//
// A null SppProgress (or one with no report) means "render all spp in a single shot",
// which is the historical, bit-identical path.
#pragma once
#include <functional>
#include "scene_film.h"

struct SppProgress {
    // report(sumFilm, sppDone, final) -> stop?
    //   sumFilm : accumulated SUM over sppDone samples (display divides by sppDone)
    //   sppDone : samples-per-pixel completed so far
    //   final   : true when sppDone has reached the requested spp target
    // Return true to stop after the current chunk (host requested an early stop).
    std::function<bool(const Film& sumFilm, long long sppDone, bool final)> report;

    // Resume seed bias (mode R / D disk resume). When a render continues from a saved
    // checkpoint that already holds `sampleBase` samples-per-pixel, the freshly-traced
    // samples must draw an INDEPENDENT noise realization, or averaging them into the
    // loaded film wouldn't reduce variance. The chunked renderers fold this into their
    // RNG seed (CPU: added to the per-chunk seed offset; GPU: mixed into the seed base)
    // so the continued samples are decorrelated from the loaded ones. 0 = fresh render.
    long long sampleBase = 0;
};

// Progress through a phase that has NO pixels yet.
//
// SppProgress can only exist once there is a film to hand back, which in mode M is the
// GATHER — but a showcase mode-M render spends most of its wall clock BEFORE that, in the
// photon deposit and the map/beam builds. Those phases used to be one long silence: the
// live window sat on a frozen "tracing photons…" caption for however many minutes the
// deposit took, which is indistinguishable from a wedged render, and the console said
// nothing either. StageProgress is the hook that makes them legible.
//
// `text` names the phase; `done`/`total` measure progress within it and are both 0 when
// the phase has no meaningful measure (a BVH build). Reporting is best-effort and purely
// informational — unlike SppProgress::report there is no return value, because a stage is
// stopped through the ordinary ft::stopRequested() flag, not by the reporter.
struct StageProgress {
    // `partial`, when non-null, is a PARTIALLY-FILLED film the host should draw instead of
    // the dark placeholder, normalised by `divisor` (the samples every COVERED pixel holds;
    // the pixels the phase has not reached yet are still zero, so the image fills in as it
    // goes). Pass nullptr from a phase that has no pixels at all — a deposit or a BVH build.
    //
    // This exists because "no pixels yet" was only ever true of the deposit and the map
    // builds. A mode-M GATHER has a film from its first launch, but the SppProgress that
    // would show it only fires on a COMPLETE chunk, and a chunk is one whole spp: measured
    // on gallery_rain with -beams that is ~30 minutes, so the window sat on a dark
    // placeholder for half an hour with a render in perfect health behind it. Naming the
    // phase in the title fixed the "is it wedged?" question; it did not give back the thing
    // the live window is FOR, which is watching the image arrive.
    std::function<void(const char* text, long long done, long long total,
                       const Film* partial, double divisor)> report;

    // Progress from INSIDE a single unit of work, where no honest rate exists.
    //
    // Same caption and same throttles as `report`, but it prints no rate and no ETA, and — the
    // reason it is a separate entry point rather than a flag — it does not feed the trailing
    // rate window at all. Call it while a long operation is still running; call `report` when
    // one completes.
    //
    // The mode-M gather is why. At 960x540 it is ONE kernel launch per spp, so `report` fired
    // exactly once per spp and the caption sat at 0% for the whole launch. A device-side
    // counter now exposes retired samples mid-launch — but that curve is steeply CONVEX,
    // because the kernel is wildly divergent and most threads retire long before the handful
    // of rays crossing the thickest cloud do. Measured on gallery_rain: 39% of the frame in
    // the first minute, 10% in the second, 1% in the third. Fed to the trailing window that
    // collapsed the reported rate to 145/s and put the ETA at ~59 min with about one minute
    // actually left; fed to a cumulative average it under-reads instead, because extrapolating
    // linearly from the fast opening cannot see the tail coming. NEITHER estimator survives a
    // convex curve, and the same convexity would poison the rate for the per-launch reports
    // that come after it. So the honest thing — the same judgement the `rate > 0` branch of
    // `report` already makes — is to show the percentage and the clock and admit there is no
    // usable ETA. The percentage is what answers "is it wedged?", which is the whole point.
    std::function<void(const char* text, long long done, long long total)> reportLive;

    // True when a `partial` would actually be DRAWN if one were supplied now.
    //
    // The renderer must ask before assembling one, because on the GPU that means a
    // device->host copy of the whole film. The window refreshes at ~4 Hz and a slice can be
    // far shorter than that, so without this gate the copy would run many times per frame
    // painted. A host with no live window answers false forever and pays nothing.
    std::function<bool()> wantFilm;

    // Re-base the elapsed clock (and the log/window throttles) to NOW.
    //
    // `report`'s rate and ETA are done/elapsed and (total-done)/rate, measured from when the
    // StageProgress was made — which is correct for the FIRST phase it covers and wrong for
    // every one after it, because each later phase starts with `done` back at 0 while
    // `elapsed` still carries all its predecessors. One StageProgress spans the deposit, the
    // map builds and the gather, so without this the gather's first line would report a rate
    // divided by the deposit's minutes and an ETA to match. Call it when a phase begins.
    std::function<void()> reset;
};
