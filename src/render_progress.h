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
