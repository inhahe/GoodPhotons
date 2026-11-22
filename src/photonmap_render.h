// Photon-mapped rendering (ROADMAP item 1, mode M).
//
// Two passes:
//   1. tracePhotonPass(): forward light-trace N photons with the camera splat OFF and
//      the deposit path ON, filling a PhotonMap (view-independent radiance cache).
//   2. renderPhotonCamera(): a backward camera pass that, at the first diffuse hit of
//      each camera ray, estimates reflected radiance either by a DIRECT radius density
//      query into the map (default) or, when final gather is enabled (fgRays > 0), by an
//      indirect Jensen final gather. Direct + specular reach the diffuse surface normally;
//      the map supplies the (direct + indirect) diffuse illumination.
//
// Final gather (fgRays > 0): instead of reading the density estimate AT the visible point
// x — which inherits the estimate's low-frequency blur right at the surface, softening
// contact shadows and small-scale detail — we shoot K cosine-weighted hemisphere sub-rays
// from x, trace one bounce to y_k, and query the map THERE. This decouples the visible-
// surface sharpness from the gather radius (the blur now lives one bounce away, at y),
// exactly the standard Jensen photon-map final gather. Direct light at x is recovered by
// gather rays that strike an emitter/environment directly; indirect by gather rays that
// strike another diffuse surface and read its (converged) outgoing radiance from the map.
// Costs ~K density queries per camera sample, so pair a larger fgRays with fewer spp.
//
// The map is built ONCE and can be reused for many cameras of a static scene — the
// flythrough win (build once, gather per frame): the multi-camera driver in main.cpp
// (runSharedPhotonMap) traces/builds one map, then calls renderPhotonCamera below for
// each frame's camera.
#pragma once
#include "gafootprint.h"
#include "gathercoverage.h" // gatherCoverage: the footprint a gather divides by
#include <vector>
#include <thread>
#include <mutex>
#include <string>
#include <cstdint>
#include "render.h"
#include "photonmap.h"
#include "photonbeams.h"   // volume single-scatter cache (mode M with -beams)
#include "beamgather.h"    // gatherPhotonBeams — the Beam x Ray estimator (shared with mode J)
#include "causticaim.h"    // Jensen projection map: aimed emission for the caustic pass
#include "allocreport.h"   // OOM that names the buffer, its size and the flag that sizes it
#include "backward.h"      // BackwardRenderer::neeLight / neeEnv for final-gather direct lighting
#include "scene_film.h"
#include "camera.h"
#include "color.h"
#include "geometry.h"
#include "parallel.h"      // ft::stopRequested — cooperative `-stop` inside the pixel loop
#include "render_progress.h"   // StageProgress — deposit progress for the live window/title
#include <algorithm>
#include <atomic>

#include <chrono>

// The albedo a gather point (material `m`, hit `h`) weighs a photon by at `lambda`: a translucent
// one's reflect albedo for a photon on its near side, its transmit albedo for one on its far side
// (`farSide`, onGatherSurfaceSided), the pair guarded as everywhere (diffuseTransmitAlbedos); any other's
// diffuse reflectance.
inline double gatherAlbedo(const Scene& scene, const Material& m, const Hit& h, double lambda, bool farSide) {
    if (m.type == MatType::DiffuseTransmit) {
        double rhoR, rhoT;
        bdpt::diffuseTransmitAlbedos(m, lambda, scene, &h, rhoR, rhoT);
        return farSide ? rhoT : rhoR;
    }
    return clamp01(diffuseReflectance(scene, m, h, lambda));
}

// HAIR-NEE (0.334.0): mode M's camera walk connects to the lights at every fiber vertex it
// scatters through (see the Hair case of photonGather). FTRACE_HAIR_NEE=0 turns it off, for
// paired measurements only.
inline bool hairNeeOn() {
    static const bool on = [] { const char* e = std::getenv("FTRACE_HAIR_NEE"); return !(e && e[0] == '0'); }();
    return on;
}
// SPECTRAL NEE (0.341.0): evaluate a glossy vertex's light connection over the SpecThr grid
// rather than at the camera's single wavelength. `-no-spec-nee` restores the scalar term;
// the switch itself lives in lighttree.h so the CUDA upload reads the same object.
inline bool specNeeOn() { return lt::gSpecNee; }
// The fiber half of the same idea, off by default -- see the note in lighttree.h for the
// measurements that put it there rather than on.
inline bool specNeeHairOn() { return lt::gSpecNee && lt::gSpecNeeHair; }

// ---- MODE-M PHASE PROFILE (`-mstats`) -------------------------------------------------------
// VOLCACHE asks for the split inside a mode-M frame's camera gather: how much is the SURFACE
// density estimate and how much is the BEAM gather, since only the latter is what a volumetric
// radiance cache would remove. Per-thread accumulators, summed and printed once.
struct MStats {
    std::atomic<long long> surfNs{0}, beamNs{0};
    std::atomic<long long> surfN{0}, beamN{0};
    void report(double wallSec) const {
        const double s = (double)surfNs.load() * 1e-9, b = (double)beamNs.load() * 1e-9;
        if (s <= 0.0 && b <= 0.0) return;
        std::fprintf(stderr,
            "[mstats] camera gather: surface estimate %.2f s over %lld calls | beam gather %.2f s "
            "over %lld probes | %.0f%% of the gather is beams | wall %.1f s (thread-seconds, so "
            "the two sum to more than the wall on %d threads)\n",
            s, surfN.load(), b, beamN.load(), (s + b) > 0.0 ? 100.0 * b / (s + b) : 0.0,
            wallSec, (int)std::thread::hardware_concurrency());
    }
};
inline MStats& mStats() { static MStats m; return m; }
inline bool mStatsOn() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_MSTATS");
        return e && std::atoi(e) != 0;
    }();
    return on;
}
struct MStatTimer {          // RAII: adds its lifetime to one accumulator, only when enabled
    std::atomic<long long>* ns; std::atomic<long long>* n;
    std::chrono::steady_clock::time_point t0;
    MStatTimer(std::atomic<long long>* nsAcc, std::atomic<long long>* nAcc)
        : ns(mStatsOn() ? nsAcc : nullptr), n(nAcc) {
        if (ns) t0 = std::chrono::steady_clock::now();
    }
    ~MStatTimer() {
        if (!ns) return;
        ns->fetch_add((long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t0).count(),
                      std::memory_order_relaxed);
        n->fetch_add(1, std::memory_order_relaxed);
    }
};

// ---- Forward photon pass: deposit into the map, no camera splat ---------------------
// Traces N photons across nThreads, each depositing into a private bank, then
// concatenates into pm.photons and records pm.nEmitted (= N). Does NOT build the grid;
// the caller picks the gather radius and calls pm.build(radius).
//
// `seedBase` is the ABSOLUTE index of the first photon of this pass: photon i draws
// from its own stream seeded by seedBase+i (seedUnit), so the deposited set is
// thread-count independent — and, crucially, a repeated-pass caller (SPPM) that
// passes its cumulative emitted count gets FRESH photons every pass. (Before this
// parameter existed every SPPM pass re-traced the identical photon set — a real
// correctness bug: progressive photon mapping's convergence needs independent
// passes, so mode S was re-averaging the same deposits at shrinking radii.)
//
// `bm` (optional) turns on the PHOTON-BEAM deposit: the same photons additionally store
// every medium crossing as a segment, giving mode M a view-independent volume cache (see
// photonbeams.h). Passing it changes how photons traverse media — straight, attenuated by
// the crossing, instead of the analog scatter-or-absorb free flight — so the SURFACE map
// changes too (it loses multiply-scattered volume light and gains correct single-scatter
// transmission). That is the documented `-beams` trade, now available to mode M.
// `beamTarget` is a budget on the stored beam count, applied as Russian roulette per beam;
// <= 0 keeps every crossing.
// `stage` (optional) reports deposit progress — how many of the N photons have been
// traced — so the caller can keep the live window's title bar moving through what is
// otherwise the longest silent phase of a mode-M render. Purely informational.
//
// `pmCaustic` (optional) turns on Jensen's TWO-MAP split: a deposit whose path reads
// L·S⁺·D — at least one focusing vertex, no scattering one (see photonVertexKind in
// render.h) — goes to *pmCaustic INSTEAD OF pm. The split is strict, so the two maps
// partition the same deposits: nothing is duplicated, nothing is lost, and a gather that
// sums the two estimates is exactly the one-map estimate would have been IF one radius
// suited both. It does not, which is the whole point — a caustic is a thin high-contrast
// concentration whose photons are orders of magnitude denser than the diffuse background,
// so buildAuto picks each map its own radius and the caustic stops being smeared away by
// a kernel sized for the ambient illumination. Null keeps every deposit in `pm` (mode S
// and every pre-0.199.7 caller).
//
// `aim` + `nAimed` (optional) add Jensen's PROJECTION-MAP half of the two-map scheme: a
// SECOND pass of nAimed photons whose emission is importance-sampled towards the scene's
// focusing geometry (causticaim.h), depositing into *pmCaustic only. Without it the caustic
// map is sharp and nearly empty — 0.12 % of deposits on gallery_rain — because a uniformly
// emitting sky almost never happens to hit a gem. The two passes are combined with the
// balance heuristic: both passes' caustic deposits are scaled by the SAME per-photon weight
// w = 1/(1 + (N_c/N_m)·rho) computed at emission (Renderer::applyCausticAim), and the caustic
// map's nEmitted stays at the main pass's count. So the aimed pass is a pure variance
// reduction — an incomplete or over-eager target set costs efficiency, never correctness —
// and nAimed = 0 leaves every deposit bit-for-bit what it was.
// `depositSurfaces` = false traces the pass for its BEAMS ALONE and leaves `pm` empty. This
// is mode J (UPBP), where surface transport is BDPT's job and a surface photon map would be
// both unused and, at the photon counts a beam map wants, the largest allocation in the
// process. Every other aspect of the pass — emission, media crossing, Russian roulette, the
// RNG stream — is untouched, so the beams a beams-only pass deposits are bit-identical to the
// ones a full mode-M pass would have deposited at the same seed. (Nothing branches on
// `photonDeposit` except `Renderer::depositPhoton`, which is a no-op when it is null.)
inline void tracePhotonPass(const Scene& scene, long long N, int nThreads,
                            bool diffraction, PhotonMap& pm, int heroC = hero::kHeroC,
                            uint64_t seedBase = 0, BeamMap* bm = nullptr,
                            long long beamTarget = 0,
                            const StageProgress* stage = nullptr,
                            PhotonMap* pmCaustic = nullptr,
                            const caim::AimMap* aim = nullptr,
                            long long nAimed = 0,
                            bool depositSurfaces = true) {
    if (nThreads < 1) nThreads = 1;
    // The aimed pass needs somewhere caustic to deposit and something to aim at.
    const bool doAimed = pmCaustic && aim && !aim->empty() && nAimed > 0 && N > 0;
    if (!doAimed) { aim = nullptr; nAimed = 0; }
    const double aimRatio = doAimed ? (double)nAimed / (double)N : 0.0;
    std::vector<PhotonBank> banks(nThreads);
    std::vector<PhotonBank> cbanks(pmCaustic ? nThreads : 0);
    std::vector<BeamBank>   bbanks(nThreads);
    std::vector<long long> emitted(nThreads, 0);
    // Beam budget: give each thread its share of the target as a self-thinning CAP and let
    // it keep everything until it gets there (BeamBank halves itself past the cap). Do NOT
    // precompute a survival rate from N — the beams-per-photon ratio is a property of the
    // scene's media volume fraction, not of the photon count, and guessing it undershot by
    // 131x on gallery_rain and overshot by 2.3x on _fog_cornell. See photonbeams.h.
    //
    // The 2x headroom lets a thread overshoot its exact share before halving, so the banks
    // still sum to at least the target when the final decimateTo trims to it; without it,
    // one halving per thread would land the total at ~half the budget.
    for (int t = 0; t < nThreads; ++t) {
        if (beamTarget > 0)
            bbanks[t].cap = std::max<size_t>(1024, (size_t)(2 * beamTarget / nThreads));
        // Private, per-thread stream for the self-thinning draws, so which beams get dropped
        // never depends on — or perturbs — the photon tracer's own RNG sequence. Salted by
        // `-seed` like every other stream: WHICH beams the roulette drops is part of the
        // realization, so leaving it fixed would have left mode M's beam map partly frozen
        // across seeds, which is the opposite of what the flag is for.
        bbanks[t].rng.seed(seedBase + 0x9E3779B97F4A7C15ULL * (uint64_t)(t + 1),
                           0xBF58476D1CE4E5B9ULL ^ g_rngSalt);
    }

    // Hero-wavelength deposit (modes M/S): each traced path deposits its live wavelengths
    // as per-λ photon records via tracePhotonHero (render.h). nEmitted still counts PATHS
    // (below), so the density estimate is energy-identical to single-λ but with far lower
    // chroma noise. Same gate as the forward tracers: no media / no GRIN (those stay C=1).
    const bool heroOn = (heroC > 1) && scene.media.empty() && !grin::sceneHasGrin(scene);

    // SPECTRAL BEAMS. Where `heroOn` above is gated OFF by the presence of media, this one is
    // gated on exactly the opposite thing — it is the media cache's own spectral widening, and
    // it applies precisely when there ARE media. `-beamspec` (pbeams::gSpecC) asks for it; the
    // scene has to be able to honour it (beamSpectralOK, above); and there must be a beam map
    // to deposit into at all.
    const int beamSpecC = (bm && pbeams::gSpecC > 1 && beamSpectralOK(scene))
                              ? std::min(pbeams::gSpecC, kBeamSpecMax) : 1;
    // ACHROMATIC-PATH BEAMS. Same scene-wide extinction test, asked without the `-beamspec`
    // condition — the mean-CIE fold stores no extra wavelengths, so `-beamspec 1` gets it too
    // (photonbeams.h, ACHROMATIC-PATH BEAMS).
    const bool beamAchroOK = bm && pbeams::gAchro && beamSpectralOK(scene);

    // Published photon count for `stage`. Written by the workers on the SAME 4096-photon
    // cadence as the stop poll (one relaxed fetch_add per 4096 photons is unmeasurable next
    // to 4096 path traces) and read by the monitor thread below. Relaxed ordering is right:
    // nothing is synchronised through it, it only feeds a title bar.
    std::atomic<long long> tracedTotal{0};

    auto worker = [&](int tid, bool aimed) {
        const MotionTimeScope passAt(g_passMotionU);   // a blurred pass's one time, if any
        Renderer r; r.diffraction = diffraction; r.maxBounce = photonMaxBounce();
        if (aimed) {
            // Caustic-only pass: no global deposits (the global map is the main pass's and
            // is normalised by ITS nEmitted), and no beam deposits for the same reason — but
            // media must still be crossed straight, or this pass would be transporting by
            // different rules than the pass it is being combined with.
            r.causticDeposit  = &cbanks[tid];
            r.aimEmission     = true;
            r.beamStraightOnly = (bm != nullptr);
        } else {
            if (depositSurfaces) r.photonDeposit = &banks[tid];
            if (pmCaustic) r.causticDeposit = &cbanks[tid];
            if (bm) r.beamDeposit = &bbanks[tid];
        }
        // Bound on BOTH passes: the main pass does not aim, but it still has to MEASURE its
        // own samples' aimed density to weight its caustic deposits. That measurement draws
        // no randomness, so the main pass's photon set is untouched.
        r.aimMap = aim; r.aimMisRatio = aimRatio;
        r.useHero = heroOn; r.heroC = heroC;
        r.beamSpecC = aimed ? 1 : beamSpecC;
        r.beamAchroOK = !aimed && beamAchroOK;
        // Mode M never consults this one — its photon is born at the chosen emitter's own
        // spectral density, so it needs no conversion (see bdpt.h, BeamSpectral). It is set
        // anyway so the field never reads as "this scene's extinction is chromatic" in a
        // scene where it is not.
        r.beamSpecOK = !aimed && bm && beamSpectralOK(scene);
        Pcg32 rng;
        const long long Np = aimed ? nAimed : N;
        const uint64_t salt = aimed ? 0x94D049BB133111EBULL : 0xEB44ACCAB455D165ULL;
        long long lo = Np * tid / nThreads, hi = Np * (tid + 1) / nThreads;
        EnergyReport e;
        long long done = 0;
        for (long long i = lo; i < hi; ++i) {
            // Cooperative `-stop` / Ctrl-C. Until this poll existed the photon DEPOSIT was
            // completely uninterruptible: v0.194.0 taught the mode-M camera *gather* to stop,
            // but nothing polled here, so `ftrace -stop` on a deposit had to wait out the
            // entire `-n` before the flag was even looked at. On a large `-n` that is many
            // minutes of a process that ignores every stop request while its photon map keeps
            // growing (a 2e9-photon CPU pass was sitting on 10 GB and climbing) — i.e. exactly
            // the situation that tempts a `taskkill /F`, which is what `-stop` exists to
            // prevent. Checked every 4096 photons rather than every photon: a photon is
            // microseconds, so a per-iteration atomic load would be measurable in the hottest
            // loop of a mode-M/S build, while 4096 of them still lands the stop in well under
            // a tenth of a second.
            if ((done & 0xFFF) == 0) {
                if (done) tracedTotal.fetch_add(0x1000, std::memory_order_relaxed);
                if (ft::stopRequested()) break;
            }
            seedUnit(rng, seedBase + (uint64_t)i, salt);
            r.tracePhoton(scene, (const Camera*)nullptr, (Film*)nullptr, (Film*)nullptr, rng, e);
            ++done;
        }
        tracedTotal.fetch_add(done & 0xFFF, std::memory_order_relaxed);   // the tail
        // Count what was ACTUALLY emitted, not what was asked for. pm.nEmitted normalises the
        // density estimate, so reporting the full share after an early break would scale a
        // truncated pass down by the fraction it never traced and darken the image.
        // The aimed pass deliberately does NOT count: it emits no light of its own, it
        // re-estimates the main pass's caustic term with a different sampler.
        if (!aimed) emitted[tid] = done;
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t, false);
    // The deposit is a join-and-wait, so progress has to be sampled from OUTSIDE it: a
    // monitor thread polls the published count while the workers run. It is only started
    // when someone asked for progress, so a headless render spawns nothing extra.
    PollStop monitorStop;
    std::thread monitor;
    if (stage && stage->report) {
        const long long nTotal = N + nAimed;
        monitor = std::thread([&, nTotal] {
            do {
                stage->report("tracing photons",
                              tracedTotal.load(std::memory_order_relaxed), nTotal,
                              nullptr, 0.0);
            } while (!monitorStop.wait(std::chrono::milliseconds(200)));
        });
    }
    for (auto& th : pool) th.join();
    // --- Aimed caustic pass (causticaim.h) --------------------------------------------
    // Runs after the main pass rather than alongside it so both can use every core, and so
    // an `ftrace -stop` during the main pass skips it entirely (the caustic map is then
    // simply the un-aimed one, which is still correct — just noisier).
    if (doAimed && !ft::stopRequested()) {
        std::vector<std::thread> apool;
        for (int t = 0; t < nThreads; ++t) apool.emplace_back(worker, t, true);
        for (auto& th : apool) th.join();
    }
    if (monitor.joinable()) { monitorStop.stop(); monitor.join(); }

    size_t total = 0;
    for (auto& b : banks) total += b.size();
    pm.photons.clear();  ftalloc::reserve(pm.photons, total, "the photon map payloads", "-n");
    pm.pos.clear();      ftalloc::reserve(pm.pos, total, "the photon map positions", "-n");
    pm.nEmitted = 0;
    for (int t = 0; t < nThreads; ++t) {
        // Append both halves in the same thread order, so pos[k] stays the position of
        // photons[k] (PhotonMap's split layout — see photonmap.h).
        pm.photons.insert(pm.photons.end(), banks[t].payload.begin(), banks[t].payload.end());
        pm.pos.insert(pm.pos.end(), banks[t].pos.begin(), banks[t].pos.end());
        pm.nEmitted += emitted[t];
    }
    if (pmCaustic) {
        size_t ctotal = 0;
        for (auto& b : cbanks) ctotal += b.size();
        pmCaustic->photons.clear();
        ftalloc::reserve(pmCaustic->photons, ctotal, "the caustic map payloads", "-n");
        pmCaustic->pos.clear();
        ftalloc::reserve(pmCaustic->pos, ctotal, "the caustic map positions", "-n");
        for (int t = 0; t < nThreads; ++t) {
            pmCaustic->photons.insert(pmCaustic->photons.end(),
                                      cbanks[t].payload.begin(), cbanks[t].payload.end());
            pmCaustic->pos.insert(pmCaustic->pos.end(),
                                  cbanks[t].pos.begin(), cbanks[t].pos.end());
        }
        // SAME normalisation as the global map: nEmitted counts PATHS EMITTED, not photons
        // stored, and both maps were filled by the one pass. Using the caustic map's own
        // stored count here would be the classic two-map bug — it would rescale a rare
        // caustic up to the brightness of the whole light source.
        pmCaustic->nEmitted = pm.nEmitted;
    }
    if (bm) {
        size_t nb = 0;
        for (auto& b : bbanks) nb += b.size();
        bm->beams.clear();
        ftalloc::reserve(bm->beams, nb, "the photon-beam map",
                         "-beamcount (or -n, which feeds it)");
        for (int t = 0; t < nThreads; ++t)
            bm->beams.insert(bm->beams.end(), bbanks[t].beams.begin(), bbanks[t].beams.end());
        bm->nEmitted   = pm.nEmitted;   // same pass, same normalisation
        bm->nDeposited = bm->beams.size();
        // Exact trim. The per-thread banks each self-thinned to their own cap, so the total
        // lands somewhere in [target, 2*target] rather than on the number the user asked
        // for; this is the one unbiased cut that makes -beamcount mean what it says.
        if (beamTarget > 0) bm->decimateTo((size_t)beamTarget);
    }
}

// The Beam x Ray 1D volume gather (`gatherPhotonBeams`) used to be spelled out here. It moved
// to **beamgather.h** in 0.215.0 so mode J (UPBP, bdpt.h) could call it too without bdpt.h
// having to include this whole header to reach one function. Same name, same signature, same
// output — the call sites below are untouched — and it gained a template hook for a per-hit
// MIS weight that mode M instantiates as the constant 1.

// ---- Final-gather sub-ray: one INDIRECT bounce from a visible point into the map -----
// A gather ray shot from a diffuse visible point (visHit/visMat). It follows specular
// surfaces (monochromatic at `lambda`) exactly like photonGather, and terminates at:
//   * the first diffuse/translucent hit y -> a radius density query at y, where EACH
//     photon is reflected off BOTH y (its own material) AND the visible point (visMat),
//     evaluated at the photon's wavelength so the two-bounce colour bleed stays spectral
//     (the same per-photon-wavelength XYZ trick mode M already uses at the visible point).
//     The map at y already holds direct+indirect at y, so this one gather bounce captures
//     the full indirect illumination of the visible point.
//   * a finite EMITTER reached AFTER a specular bounce -> a monochromatic (camera-lambda)
//     sample reflected off the visible point. Reached WITHOUT any specular bounce (a
//     straight hemisphere ray onto a light) it returns 0: that direct term is supplied
//     instead by low-variance next-event estimation at the visible point (see
//     photonGather), so counting it here too would double-count. This specular-arrival
//     gate mirrors backward.h's MIS `specularArrival`.
//   * the ENVIRONMENT on ANY escape -> a monochromatic sample reflected off the visible
//     point (env has no finite-light NEE in mode M, so gather rays carry env's direct
//     term; its indirect bounces come from the map query at a diffuse hit above).
// Returns the XYZ radiance leaving the visible point toward the gather-ray origin for this
// single sampled direction (the caller averages over K samples). Because the sub-ray is
// cosine-weighted (pdf = cos/pi) and the visible BRDF is Lambertian (f_r = rho/pi), the
// cosine and 1/pi cancel: the visible-point weight reduces to rho(vis), folded per photon
// (diffuse hit) or applied once (specular-arrival emitter/env). `norm` = 1/(pi r^2
// nEmitted) as in the caller. Mirrors photonGather's specular walk; keep the two in sync.
//
// `pmC`/`normC` are the optional CAUSTIC map and its own normalisation (see tracePhotonPass).
// When present the density estimate is the SUM of the two maps' estimates — the maps hold
// disjoint deposits, so this is the same estimator with each population read at the radius
// that suits it.
// ---- SPECGATHER: the camera walk's spectral throughput --------------------------------------
// Mode M's camera walk is MONOCHROMATIC at the camera sample's wavelength while the photon map is
// POLYCHROMATIC -- every photon carries its own. The density estimate gets this right at the gather
// point (it evaluates the BRDF at `ph.lambda`, per photon), but every reflectance the ray collected
// on the way there used to be folded into the SCALAR `thr` at the camera's lambda, and that scalar
// then multiplied a photon sum spanning every wavelength. So the estimator computed
//     r(lambda_c) * sum_p rho(lambda_p) P_p CIE(lambda_p)
// where it owes
//     sum_p r(lambda_p) rho(lambda_p) P_p CIE(lambda_p).
// Identical when r is flat; otherwise biased by however r correlates with the photon spectrum, in
// EITHER direction -- measured against mode D on one Cornell sphere: flat 1.005, JH grey 0.985,
// JH white 0.880, JH red 1.142 (see known-issues). gallery_rain's gold gyroid and chrome ring are
// exactly this case.
//
// So the walk carries the same product sampled over a coarse wavelength grid beside the scalar, and
// the estimate reweights each photon by the ratio at ITS wavelength. Two properties make this cheap
// and safe: the ratio is identically 1 when every factor is flat (so an uncoloured scene renders
// BIT-IDENTICALLY, which is the regression test), and a factor is probed at three wavelengths first
// so a flat one costs three evaluations rather than K.
struct SpecThr {
    static constexpr int K = kSpecCellsMax;      // room for the most cells a scene's grid has
    const SpecCells* cells;                      // the scene's grid (Scene::specCells), never null
    int n;                                       // its cells: the arrays' first n entries are live
    double v[K];                                 // the product over the cells (the numerator)
    // The SAME product at the camera's wavelength, accumulated from the very scalars the walk
    // multiplied into `thr`. It is NOT recomputed from the grid, and that is the whole point:
    // `thr` already contains this factor, so `thr * ratio` collapses to (everything else) *
    // v[lambda_p] exactly. The first version divided by the nearest BIN CENTRE instead, which
    // agrees for a smooth spectrum and does not where one plunges near the band edge -- there
    // the cancellation failed and the estimate detonated (a JH white measured 6e5 x mode D).
    double cam = 1.0;
    bool   any = false;                          // anything non-flat folded in? (else ratio == 1)
    // The coins the walk tossed (mulCoin), apart from their weights: per bin, the product of the
    // probabilities of the branches taken (`p`), and the same at the walk's own wavelength
    // (`pcam`). Folding p_L / p_walk per coin, as `v` / `cam` do, re-expresses the walk at L only
    // where every branch it took is possible at L and at the walk's wavelength alike -- a walk
    // cannot sample a branch its own wavelength forbids. Where some bin cannot take the walk's
    // branches (`partial`: a coat opaque at some wavelengths, total internal reflection at some),
    // ratio() divides instead by the probability of the branches averaged over the density the
    // walk's wavelength is drawn from (sum_k Q_k p[k], Q_k its mass in cell k), which every walk wavelength
    // shares, so the walks that can take them stand in for the ones that cannot
    // (MODE-M-FOLD-SUPPORT). Whether a walk is partial depends only on its branches, never on its
    // wavelength, so the two rules combine without bias.
    double p[K];
    double pcam = 1.0;
    bool   partial = false;
    double pscale = 1.0;                         // pcam / sum_k Q_k p[k] once partial
    // `c`: the scene's cells, whose masses are the walk's wavelength density's (the scene's
    // emission sampler: SpecCells::setMass).
    explicit SpecThr(const SpecCells& c) : cells(&c), n(c.n) {
        for (int k = 0; k < n; ++k) { v[k] = 1.0; p[k] = 1.0; }
    }
    double lamOf(int k) const { return cells->lam[k]; }      // cell k's wavelength
    int binOf(double lam) const { return cells->cellOf(lam); }   // the cell `lam` falls in
    // Fold in one spectral factor, `f(lambda)`, at every bin: a band between any few probe
    // wavelengths -- a notch, a band-pass gel -- is as much the factor as its ends, and a zero in it
    // is a hole in the support. A factor equal at every bin folds as the constant it is and leaves
    // `any` alone (it cancels in the ratio). `atCam` is the value the scalar throughput just used at
    // the camera's wavelength, passed in rather than recomputed so the two can never disagree (see
    // `cam`).
    template <class F> void mul(const F& f, double atCam) {
        cam *= atCam;
        double a = 0.0;
        bool flat = true, zero = false;
        for (int k = 0; k < n; ++k) {
            const double fk = f(lamOf(k));
            if (k == 0) a = fk;
            else if (std::fabs(fk - a) > 1e-12 * (1.0 + std::fabs(a))) flat = false;
            v[k] *= fk;
            if (!(fk > 0.0)) { p[k] = 0.0; zero = true; }
        }
        if (!flat) any = true;
        // A weight zero in some bins leaves a walk drawn there with nothing to carry, at any
        // wavelength: the same support problem as a coin's (see `p`), so the walks drawn where it
        // is not zero stand in for them.
        if (zero) { partial = true; rescale(); }
    }
    // Fold in a factor already evaluated ON the grid, plus its value at the camera's wavelength.
    // The media term comes this way: its per-wavelength values fall out of one shared
    // ratio-tracking walk rather than from a function that can be called per bin.
    // A zero here is NOT a hole in the support, as a zero weight in mul is: the media term is a
    // ratio-tracking estimate, so a zero is one realization's event, and whether it happens depends
    // on the walk's wavelength (the majorant is taken over it), which `partial` must never.
    void mulVec(const double* vk, double atCam) {
        cam *= atCam;
        for (int k = 0; k < n; ++k) {
            if (!any && std::fabs(vk[k] - vk[0]) > 1e-12 * (1.0 + std::fabs(vk[0]))) any = true;
            v[k] *= vk[k];
        }
    }
    // Fold in a coin the walk tossed: f(lambda, prob) returns the branch's factor at lambda (its
    // probability times any weight -- Renderer::stepFactorAt) and sets `prob` to its probability
    // alone; factorAtCam / probAtCam are the same at the walk's wavelength. Without `exhaustive`,
    // mul's three-point probe: a factor flat there is folded as the constant it is (it cancels).
    // `exhaustive` evaluates every bin, where a branch impossible in a narrow band could slip past
    // the probe (a coated surface, a half mirror).
    template <class F> void mulCoin(const F& f, double factorAtCam, double probAtCam,
                                    bool exhaustive) {
        double pr;
        if (!exhaustive) {
            const double a = f(lamOf(0), pr), b = f(lamOf(n / 2), pr), c = f(lamOf(n - 1), pr);
            const double tol = 1e-12 * (1.0 + std::fabs(a));
            if (std::fabs(a - b) <= tol && std::fabs(a - c) <= tol) {
                cam *= factorAtCam;
                for (int k = 0; k < n; ++k) v[k] *= a;
                return;
            }
        }
        // Every bin, once. A coin flat over the bins AND equal to the walk's own value cancels: it
        // is folded like any other but leaves `any` alone, so ratio() stays exactly 1 if nothing
        // else varies. Flat over the bins but not at the walk's wavelength (a change inside one
        // bin) does not cancel.
        cam *= factorAtCam;
        pcam *= probAtCam;
        double a = 0.0, pLo = 0.0, pHi = 0.0;
        bool flat = true;
        for (int k = 0; k < n; ++k) {
            const double fk = f(lamOf(k), pr);
            if (k == 0) a = fk;
            else if (std::fabs(fk - a) > 1e-12 * (1.0 + std::fabs(a))) flat = false;
            v[k] *= fk;
            p[k] *= pr;
            if (k == 0) pLo = pHi = p[k];
            pLo = std::min(pLo, p[k]); pHi = std::max(pHi, p[k]);
        }
        if (!(pLo > kCoinRare * pHi)) partial = true;   // the branches impossible, or as good as, somewhere
        if (!flat || std::fabs(factorAtCam - a) > 1e-12 * (1.0 + std::fabs(a))) any = true;
        if (partial) rescale();
    }
    // pscale = pcam / sum_k Q_k p[k], Q_k the walk density's mass in cell k. The sum is zero only if
    // no cell's wavelength the walk could be drawn at takes these branches, which the walk itself
    // just did: then the cells cannot resolve the support, and the walk's own probability serves.
    void rescale() {
        double D = 0.0;
        for (int k = 0; k < n; ++k) D += cells->mass[k] * p[k];
        pscale = (D > 0.0) ? pcam / D : 1.0;
    }
    // The correction a photon of wavelength `lamP` needs. Denominator is `cam`, so this
    // multiplied by `thr` leaves the walk's spectral factors evaluated at lamP and everything
    // else untouched; past a partial coin, times pscale (see `p`). A dead path (cam == 0)
    // contributes nothing either way.
    double ratio(double lamP) const { return ratioAt(binOf(lamP)); }
    double ratioAt(int k) const {                                  // cell k's
        if (!any) return 1.0;
        return (cam > 1e-300) ? v[k] / cam * pscale : 0.0;
    }
    // The largest ratio over the bins (1 when nothing chromatic was folded). Where coloured glass
    // absorbs at the walk's own wavelength and passes others, `cam` and with it the scalar
    // throughput fall as far as the ratio climbs -- past a float's range (1e+38) behind a few
    // centimetres of dark gem. Whatever stores or sums the ratio in float takes ratio / peak and
    // multiplies the throughput, a double, by the peak (mode S's vpRatio, the device's gathers).
    double peak() const {
        if (!any || !(cam > 1e-300)) return 1.0;
        double m = 0.0;
        for (int k = 0; k < n; ++k) m = std::max(m, v[k] / cam * pscale);
        return (m > 0.0) ? m : 1.0;
    }
    // The carrier as the beam gather reads it (JSpec, beamgather.h: mode J's carrier, read between
    // bin centres), the ratio over its peak and the peak in logS, so neither leaves a float: on
    // JSpec's uniform bins, each the cell its centre falls in (the bin itself where it is not split).
    // False, and `js` untouched, while nothing chromatic was folded: the gather then takes no carrier.
    bool toJSpec(JSpec& js) const {
        static_assert(JSpec::K == kBandProbeBins, "the beam gather's carrier is on the uniform bins");
        if (!any) return false;
        const double pk = peak();
        const bool live = cam > 1e-300;
        js.any = true;
        js.disp = false; js.lamC = 0.0f; js.invPdfC = 0.0f;
        js.logS = std::log(pk);
        for (int k = 0; k < JSpec::K; ++k) {
            const int c = cells->cellOf(bandBinLambda(k));
            js.r[k] = live ? (float)(v[c] / cam * pscale / pk) : 0.0f;
        }
        return true;
    }
};

// Says, where the scene's tables split the carrier's bins (SpecCells), where: a mode M or S render's
// opening line beside its others, once a scene (as CudaSceneHold tells scenes apart), however many of
// the routes that start one call it.
inline void reportSpecCells(const Scene& scene) {
    const SpecCells& c = scene.specCells;
    if (!c.split()) return;
    static std::mutex m;
    static const Scene* said = nullptr;
    static uint64_t saidGen = 0, saidBuild = 0;
    {
        std::lock_guard<std::mutex> lk(m);
        if (said == &scene && saidGen == scene.generation && saidBuild == scene.buildCount) return;
        said = &scene; saidGen = scene.generation; saidBuild = scene.buildCount;
    }
    std::string at;
    char buf[32];
    for (int k = 0; k < kBandProbeBins; ++k)
        for (int j = c.first[k] + 1; j < c.first[k] + c.count[k]; ++j) {
            std::snprintf(buf, sizeof buf, " %.2f", c.lo[j]);
            at += buf;
        }
    std::printf("[specgather] the spectral carrier's %d bins split at%s nm (%d cells), where a spectrum "
                "the camera walk folds is not near linear across a bin\n", kBandProbeBins, at.c_str(), c.n);
}

// SPECGATHER: a layered coat's coat-or-body coin at every bin, as modes J and U fold it, for mode M's
// walk and mode S's. A thin-film coat's reflectance swings across the band, and the body under it
// keeps 1 - R(lambda). `R` is the coat's reflectance at the walk's wavelength for the direction `d`
// the walk met it along, `coatPick` the branch it took.
inline void foldCoatCoin(SpecThr& sthr, const Scene& scene, const Material& lm, const Hit& h,
                         const Vec3& d, bool coatPick, double R) {
    const double pc = coatPick ? R : 1.0 - R;
    sthr.mulCoin([&](double L, double& pr) {
        const double rL = layeredCoatReflectance(scene, lm, h, d, L);
        return pr = coatPick ? rL : 1.0 - rL;
    }, pc, pc, false);
}

// The media transmittance of one straight CAMERA segment -- one ratio-tracking sample, or
// pbeams::kSunDiscTrSamples of them when the segment points into a sun's disc (see the constant's
// note) -- at each of the carrier `st`'s cells' wavelengths, Tv[0..st.n), plus (Tv[st.n]) the camera's
// own, all from one shared walk (device twin: dCamMediaTrSpec). The camera's slot is what the
// scalar `thr` takes, so the scalar and the carrier's denominator are the same number by
// construction -- the lesson from SPECGATHER's first attempt, which recomputed it and detonated.
inline void camMediaTrSpec(const Scene& scene, const Renderer& mats, const Vec3& o, const Vec3& d,
                           double len, double lambda, Pcg32& rng, const SpecThr& st, double* Tv) {
    const int K = st.n;
    double lams[SpecThr::K + 1];
    for (int k = 0; k < K; ++k) lams[k] = st.lamOf(k);
    lams[K] = lambda;
    for (int k = 0; k <= K; ++k) Tv[k] = 1.0;
    // Same sun-disc multi-sampling the scalar version does (see pbeams::kSunDiscTrSamples): a
    // directly-viewed disc is 10^4 times its surroundings, so one sample of the cloud in front of it
    // twinkles. Averaging whole VECTORS keeps the wavelengths correlated within each sample.
    const int n = (scene.sunCount > 0 && scene.sunRadiance(d, lambda) > 0.0)
                      ? pbeams::kSunDiscTrSamples : 1;
    if (n == 1) {
        mats.mediaTransmittanceSpec(scene, o, d, len, lams, K + 1, Tv, rng);
        return;
    }
    double acc[SpecThr::K + 1] = {0.0};
    for (int i = 0; i < n; ++i) {
        double one[SpecThr::K + 1];
        for (int k = 0; k <= K; ++k) one[k] = 1.0;
        mats.mediaTransmittanceSpec(scene, o, d, len, lams, K + 1, one, rng);
        for (int k = 0; k <= K; ++k) acc[k] += one[k];
    }
    for (int k = 0; k <= K; ++k) Tv[k] = acc[k] / (double)n;
}

// The GiCtx a light sample takes at a vertex shading with `m` that the walk reached from `wa`
// (toward the previous vertex): a coated vertex's coat (render.h CoatCtx), else nothing.
inline BackwardRenderer::GiCtx coatGi(const Material& m, const Vec3& wa) {
    BackwardRenderer::GiCtx g;
    g.coat = CoatCtx(m, wa);
    return g;
}

// A coated vertex's camera-walk continuation along `d`, the walk having arrived from `wa` (toward
// the previous vertex), takes the coat (render.h coatF: a body's way out, a coat's reciprocal
// reflection), into `thr` and, if given, the carrier at each grid wavelength. False: nothing is
// left to carry.
inline bool coatWalkOut(const Scene& scene, const Material& m, const Hit& h, const Vec3& wa,
                        const Vec3& d, double lambda, double& thr, SpecThr* sthr) {
    return coatContinueWalk(scene, m, h, wa, d, lambda, thr, nullptr, nullptr, 0,
                            [&](const auto& f, double s) { if (sthr) sthr->mul(f, s); });
}

// CURVE-GATHER: a camera walk's step at a strand that holds no photon (gatherlessStrand), for every
// walk that would gather: its direct light by NEE at the walk's wavelength, returned times `thr` as
// it arrived; its reflectance into `thr` and, if given, the carrier; and `ray` turned along its
// diffuse lobe. `cont` comes back false where the walk has nothing left to carry. The caller marks
// the arrival, so the emitter the continuation meets is not counted again (as after a hair strand).
inline double strandStep(const Scene& scene, const BackwardRenderer& bw, const Material& m,
                         const Hit& h, double lambda, double invPdfL, Pcg32& rng, double& thr,
                         SpecThr* sthr, Ray& ray, bool& cont) {
    const double rho = clamp01(diffuseReflectance(scene, m, h, lambda));
    const Vec3 wa = ray.d * -1.0;   // the arrival, for the coat (coatWalkOut)
    const double d = thr * bw.neeLight(scene, h, rho, invPdfL, lambda, rng, nullptr, coatGi(m, wa));
    thr *= rho;
    if (sthr) sthr->mul([&](double L) { return clamp01(diffuseReflectance(scene, m, h, L)); }, rho);
    cont = continueAlong(h, cosineHemisphere(h.n, rng), Lobe::Reflect, ray, scene.hitEps(h)) && thr > 0.0 &&
           coatWalkOut(scene, m, h, wa, ray.d, lambda, thr, sthr);
    return d;
}

inline Vec3 photonGatherSub(const Scene& scene, const PhotonMap& pm, Ray ray, Pcg32& rng,
                            bool diffraction, int maxBounce, double lambda, double invPdfL,
                            double norm, const Hit& visHit, const Material& visMat,
                            const Vec3& visWa, const MediumStack& stk0, const PhotonMap* pmC = nullptr,
                            double normC = 0.0) {
    bdpt::LocalWork work;   // its segments, for the CPU refresh (MODEM-CPU-NONDET)
    Vec3 L{0, 0, 0};
    double thr = 1.0;
    bool specularSeen = false;                           // any specular bounce so far?
    Renderer mats; mats.diffraction = diffraction;
    // nested-dielectric medium stack, from the visible point's (`stk0`)
    MediumStack stk = stk0;
    const bool grinAny = grin::sceneHasGrin(scene);      // final-gather rays bend too

    // GLOSSY-NEE. `bwNee` is the shared direct-lighting estimator (its `neeLight` is what the
    // final gather already uses); `gmis` carries the lobe density of a glossy continuation to
    // whichever emitter site it reaches, and is cleared at the top of every bounce so a mirror
    // or a dielectric can never inherit one and halve the emission behind it.
    BackwardRenderer bwNee; bwNee.diffraction = diffraction;
    const bool gneeOn = BackwardRenderer::glossyNeeOn();
    BackwardRenderer::GlossyMis gmis;
    bool hairArrival = false;   // HAIR-NEE: the previous vertex was a fiber that already took its direct light
    // The softening the last vertex lends what it reaches (contSoftening): first the visible point's.
    double contSoft = contSoftening(visMat, visHit, ray.d);
    // A coated visible point's lobe takes the coat on the gather ray's way in (render.h coatF; the
    // camera walk reached it from `visWa`), at each wavelength a term reads the visible point's
    // reflectance at: it rides in rhoV. Prepared once (CoatEval) and, read per photon, memoised by
    // nm (the rounding the coat's T-bar table already takes).
    const bool visCoat = coatAny(visMat);
    const CoatEval visCe = visCoat ? coatEval(scene, visMat, visHit, visWa, ray.d) : CoatEval{};
    constexpr int kVisN = (int)(LAMBDA_MAX - LAMBDA_MIN) + 1;
    double sVisMemo[kVisN];
    if (visCoat) std::fill(sVisMemo, sVisMemo + kVisN, -1.0);
    const auto sVis = [&](double Lw) {
        if (!visCoat) return 1.0;
        int i = (int)std::lround(Lw - LAMBDA_MIN);
        i = i < 0 ? 0 : (i >= kVisN ? kVisN - 1 : i);
        if (sVisMemo[i] < 0.0) sVisMemo[i] = visCe.at(LAMBDA_MIN + (double)i);
        return sVisMemo[i];
    };

    // A coverage pass-through is no vertex: its bounce is refunded (capped: 256 is far above
    // anything real fur hands back), as backward.h's walks refund it.
    int nullSteps = 0;
    // The tail (b == maxBounce): the ray the last vertex sampled, for the light it reaches and
    // nothing past it (backward.h radianceLoop, GLOSSY-DEPTH-CAP).
    for (int b = 0; b <= maxBounce; ++b) {
        const bool tail = (b == maxBounce);
        ++work.n;
        if (grinAny) {
            double arc = 0.0;
            grin::MarchInfo mInfo;
            grin::marchSegments(scene, ray,
                [&](const Vec3&, const Vec3&, double slen, double&) { arc += slen; return false; },
                false, &mInfo);
            int cm = stk.topMat();                       // Beer-Lambert over the marched arc
            double a = (cm >= 0) ? scene.mats[cm].absorb(lambda) : 0.0;
            if (a > 0.0 && arc > 0.0) thr *= std::exp(-a * arc);
            // Radiance over n^2 along the bend; past a bend the light the walk meets was reached by
            // no light sample (grin.h MarchInfo), so it counts as one met past a specular bounce.
            thr *= mInfo.radScale;
            if (mInfo.bent) { specularSeen = true; hairArrival = false; gmis.clear(); }
        }
        Hit h = closestHitIn(scene, stk, ray);
        if (h.valid) {                                   // Beer-Lambert in current medium
            int cm = h.extMat;
            double a = (cm >= 0) ? scene.mats[cm].absorb(lambda) : 0.0;
            if (a > 0.0) thr *= std::exp(-a * h.t);
        }
        if (!h.valid) {                                  // escaped -> environment
            // Env is collected by the gather rays directly (mode M has no separate env
            // NEE / MIS at the visible point), so add it on ANY escape — the map at a
            // diffuse hit already supplies env's INDIRECT bounces, this supplies direct.
            if (scene.envIndex >= 0) {
                double rhoV = (clamp01(diffuseReflectance(scene, visMat, visHit, lambda)) * sVis(lambda));
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * rhoV * scene.envRadiance(ray.d, lambda) * invPdfL * contSoft);
            }
            return L;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return L;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            // Resolve to a LOBE and let the material switch shade it -- the coat with
            // probability R (the Fresnel reflectance at this angle), else a body lobe. The
            // coat is a material of its own (Material::coatChild), so it goes through the
            // switch's Glossy case and gets that case's glossy-NEE, throughput and MIS. An
            // earlier version reflected the ray inline here and skipped all three, which
            // broke the identity that a coat of reflectance 1 IS a glossy material: it
            // rendered 30 % bright against one (scraps/ident_*.ftsl). Same convention as
            // bdpt.h's randomWalk and the device's dResolveCompound.
            const double R = layeredCoatReflectance(scene, *mp, h, ray.d, lambda);
            int c = (rng.uniform() < R) ? mp->coatChild
                                        : mixResolveChild(scene, *mp, h, rng.uniform());  // body lobe: honours a bound weight map (device twin: dResolveCompound)
            if (c < 0) return L;
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        // Self-emission on a SPECULAR arrival (a diffuse arrival's direct term comes from
        // NEE at the visible point, so adding it here too would double-count). One-sided by
        // the geometric normal, matching Vertex::Le / bkRadiance.
        //
        // This no longer RETURNS: an emissive material still has a BSDF, so a glowing
        // diffuse surface both emits and reflects, and the walk has to go on to the density
        // estimate below. See photonGather for the measurement.
        // A glow no emitter samples lights nothing, so what reaches the visible point from one here
        // is not counted (Scene::isGlow; BackwardRenderer::GiCtx::pastDiffuse).
        const int eIdx = m.isLight ? BackwardRenderer::emitterIndexOfResolved(scene, h, m) : -1;
        if (m.isLight && eIdx >= 0 && specularSeen && dot(ray.d, h.ng) < 0.0 && !hairArrival) {
            double rhoV = (clamp01(diffuseReflectance(scene, visMat, visHit, lambda)) * sVis(lambda));
            // GLOSSY-NEE's lobe-sampling half; 1 (and bit-identical) unless the last bounce was
            // a glossy one that already connected to this emitter.
            const double wMis = (gmis.pdf > 0.0) ? bwNee.glossyHitWeight(scene, gmis, eIdx, ray.d, &h) : 1.0;
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                 * (thr * rhoV * emitSlot(scene, m, h, lambda, ray.d * -1.0) * invPdfL * wMis * contSoft);
        }
        if (tail) {                                      // the light it reached, and no more --
            // past a strand it passes, which is no vertex (BackwardRenderer::tailPassesStrand)
            if (m.type == MatType::Hair && nullSteps < 256 &&
                bwNee.tailPassesStrand(scene, m, h, lambda, ray, rng)) { --b; ++nullSteps; continue; }
            return L;
        }

        // GLOSSY-NEE: cleared HERE and not at the top of the loop. `gmis` is written by the
        // PREVIOUS bounce's glossy branch and read by THIS bounce's emitter/sun sites above, so
        // a clear at the loop top erases it a few lines before the only code that wants it --
        // which leaves the connection in place with no compensating weight on the lobe-sampling
        // side, i.e. double counting wherever both strategies reach the same light. See the
        // twin note in backward.h. A hair pass-through is no vertex and puts all three back.
        const BackwardRenderer::GlossyMis gmisIn = gmis;
        const bool hairArrivalIn = hairArrival, specularSeenIn = specularSeen;
        const double contSoftIn = contSoft;
        bool nullHere = false;
        gmis.clear();
        contSoft = 1.0;   // a specular vertex lends nothing; the glossy case sets its own
        hairArrival = false;
        const Vec3 wArr = ray.d * -1.0;   // the arrival, for the coat (coatWalkOut)

        // CURVE-GATHER: no density estimate on a strand that holds no photon (strandStep). Every
        // term this sub-walk reports carries the visible point's reflectance.
        if (gatherlessStrand(m, h.fiberRadius)) {
            const double rhoV = (clamp01(diffuseReflectance(scene, visMat, visHit, lambda)) * sVis(lambda));
            bool cont;
            const double d = strandStep(scene, bwNee, m, h, lambda, invPdfL, rng, thr, nullptr, ray, cont);
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * (d * rhoV);
            hairArrival = true;
            if (!cont) return L;
            continue;
        }

        switch (m.type) {
            case MatType::Diffuse:
            case MatType::DiffuseTransmit:
            case MatType::Fluorescent: {
                // Density estimate at y, folding the visible-point reflectance per photon
                // wavelength: L_o(vis) += rho(vis,l_p) * [rho(y,l_p)/pi] * Phi_p / (pi r^2 N).
                // `nrmOut` comes back as the map's fixed normalisation unless the map gathers
                // at a PER-QUERY radius (PhotonMap::adaptiveRadius — the caustic map does),
                // in which case 1/(pi r_q^2 N) is recomputed for this query's own radius.
                auto est = [&](const PhotonMap& M, double normFixed, double& nrmOut) {
                    MStatTimer _t(&mStats().surfNs, &mStats().surfN);
                    // The true normal, as the photons carry it (depositPhoton): one for the
                    // radius, the coverage and the photon test alike.
                    const Vec3 gn = orientedGeoN(h);
                    const double slack = M.gatherSlack(h.fiberRadius);
                    const double rq = M.adaptiveRadius(h.p, gn, h.fiberRadius, m.type == MatType::DiffuseTransmit);
                    nrmOut = normFixed;
                    if (rq != M.radius) {
                        const double a = PI * rq * rq;
                        nrmOut = (M.nEmitted > 0 && a > 0.0)
                                     ? 1.0 / (a * (double)M.nEmitted) : 0.0;
                    }
                    // M-GATHERAREA: divide by the area actually gathered from, not by the whole
                    // disc; `-gatherarea 0` leaves nrmOut the whole disc's.
                    if (const int gaM = gatherAreaSamples())
                        nrmOut *= gatherAreaScale(
                            gatherCoverage(scene, h.p, gn, rq, slack, rng, gaM, h.matId,
                                           h.fiberRadius));
                    Vec3 g{0, 0, 0};
                    M.queryR(h.p, rq, [&](const Photon& ph, double d2, int k) {
                        bool farSide;
                        if (!onGatherSurfaceSided(h.p, gn, M.pos[k], ph.n, d2, slack,
                                                  m.type == MatType::DiffuseTransmit, farSide)) return;   // another surface's
                        double rhoY = gatherAlbedo(scene, m, h, ph.lambda, farSide);
                        double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, ph.lambda)) * sVis(ph.lambda);
                        double f = rhoY * (1.0 / PI);
                        g += M.cie[k] * (f * rhoV * (double)ph.power);  // == cie(lambda_p), precomputed
                    });
                    return g;
                };
                double nA = norm;
                L += est(pm, norm, nA) * (nA * thr);
                if (pmC && !pmC->photons.empty()) {
                    double nB = normC;
                    L += est(*pmC, normC, nB) * (nB * thr);
                }
                return L;
            }
            case MatType::Mirror: {
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray, scene.hitEps(h))) return L;   // across the surface
                break;
            }
            case MatType::Glossy: {
                if (!glossyCameraSide(ray.d, h.n)) return L;   // met from below ns (glossyCameraSide)
                // NEXT-EVENT ESTIMATION AT A GLOSSY VERTEX (GLOSSY-NEE). The gather ray folds
                // the VISIBLE point's reflectance into everything it reports, so `rhoV`
                // multiplies the connection exactly as it multiplies the emission above.
                // Taken before `thr *= r`: the connection carries `r` inside bsdfF.
                if (gneeOn) {
                    const double rhoV = (clamp01(diffuseReflectance(scene, visMat, visHit, lambda)) * sVis(lambda));
                    const BackwardRenderer::NeeBsdf nb{&m, ray.d * -1.0};
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                         * (thr * rhoV * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng,
                                                        nullptr, coatGi(m, ray.d * -1.0), nullptr, nullptr, &nb));
                }
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                const GlossyDraw gd = drawGlossy(scene, m, h, h.n, ray.d * -1.0, rng);
                const Vec3& o = gd.o;
                if (dot(o, h.n) <= 0.0) return L;
                // The draw's weight, and an iridescent lobe's interference reflectance at the
                // pair's half-vector.
                thr *= gd.w * gd.R(lambda);
                if (gneeOn) {
                    gmis.pdf = bdpt::bsdfPdf(m, h.n, ray.d * -1.0, o, lambda, scene, &h);
                    gmis.from = h.p;
                    gmis.n = h.n;
                }
                if (!continueAlong(h, o, Lobe::Reflect, ray, scene.hitEps(h))) return L;
                contSoft = contSoftening(m, h, ray.d);
                break;
            }
            case MatType::Dielectric:
            case MatType::ThinFilm:
            case MatType::Multilayer: {
                // A dielectric body's surface, plain or coated (Renderer::interfaceStep); this
                // camera-side walk carries radiance across a refraction.
                const Renderer::InterfaceStep st =
                    mats.interfaceStep(scene, m, h, ray.d, lambda, rng, stk,
                                       Renderer::Transport::Radiance);
                if (st.absorbed) return L;
                ray = st.ray;
                thr *= st.scale;
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() < r) { if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray, scene.hitEps(h))) return L; }
                else                   ray = passThrough(h, ray.d, scene.hitEps(h));
                break;
            }
            case MatType::Filter: {
                thr *= clamp01(transmitSlot(scene, m, h, lambda));
                ray = passThrough(h, ray.d, scene.hitEps(h));
                break;
            }
            case MatType::Hair: {
                // Fiber BCSDF — scatter through it, but do NOT gather here: the photon
                // payload carries no incident direction, so a directional BCSDF density
                // estimate is impossible (see sppm_render.h's Hair case and
                // known-issues.md). Treated like the glossy/specular cases around it.
                const Vec3 wPrev{-ray.d.x, -ray.d.y, -ray.d.z};
                const HairShade hs = hairShadeAt(scene, m, h, lambda, wPrev);
                if (hairNeeOn()) {                          // HAIR-NEE: see photonGather's Hair case
                    // every term this sub-walk reports carries the visible point's albedo
                    const double rhoV = (clamp01(diffuseReflectance(scene, visMat, visHit, lambda)) * sVis(lambda));
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                         * (thr * rhoV * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng, nullptr,
                                                 BackwardRenderer::GiCtx(), &hs, nullptr, nullptr));   // a fibre has no coat
                    hairArrival = true;
                }
                double pdfH = 0.0, fv = 0.0;
                const Vec3 wl = hair::sampleWith(hs.b, hs.woLocal, rng, pdfH, fv);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return L;
                const double cosLong =
                    hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                thr *= clamp01(fv * cosLong / pdfH);       // == T = sum_p A_p
                // A coverage pass-through (exactly -wo) is no vertex: what the last real one
                // decided about counting an emitter hit stands, so light seen straight through
                // the fiber is weighed as that vertex would weigh it.
                if (hs.b.opacity < 1.0 && wl.x == -hs.woLocal.x && wl.y == -hs.woLocal.y &&
                    wl.z == -hs.woLocal.z) {
                    nullHere = true; hairArrival = hairArrivalIn; gmis = gmisIn; contSoft = contSoftIn;
                    if (nullSteps < 256) { --b; ++nullSteps; }   // ...and its bounce refunded
                }
                const Vec3 wo = hair::toWorld(hs.fr, wl);
                ray = Ray{h.p + wo * scene.hitEps(h), wo, hairExitOffset(hs, h.n, wo, scene.hitEps(h))};
                break;
            }
            default: {                                   // Grating: approx reflect
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray, scene.hitEps(h))) return L;   // across the surface
                break;
            }
        }
        // A coated vertex's continuation takes the coat; a pass-through is no vertex.
        if (!nullHere && !coatWalkOut(scene, m, h, wArr, ray.d, lambda, thr, nullptr)) return L;
        // only specular cases reach here (diffuse/light returned above); a pass-through inherits
        specularSeen = nullHere ? specularSeenIn : true;
        if (thr <= 0.0) return L;
    }
    return L;
}

// ---- Camera gather: radiance along one camera ray via the photon map ----------------
// Returns the XYZ contribution. The ray is followed through specular surfaces
// (monochromatic at a sampled lambda); at the first diffuse/translucent hit the reflected
// radiance is estimated by a radius query, with each photon reflected at ITS OWN
// wavelength (density estimate built directly in XYZ). Directly-viewed emitters and the
// environment are added as a monochromatic estimate at the sampled lambda.
//
// PARTICIPATING MEDIA: when `bm` is non-null the walk also (a) gathers volume single scatter
// from the beam map along every segment it travels and (b) attenuates its throughput by the
// media transmittance of that segment, so fog correctly dims the surfaces and sky behind it.
// With `bm` null the walk is media-blind — which is what mode M has always been, and why a
// fog / rain / cloud scene renders its volume as nothing without -beams.
//
// `pmC` (optional) is the CAUSTIC map (see tracePhotonPass): a disjoint half of the same
// deposits, gathered at its own much finer radius and simply added.
// `lambdaU`, in [0, 1), is the hero wavelength's uniform: renderPhotonCamera's stratified one.
inline Vec3 photonGather(const Scene& scene, const PhotonMap& pm, Ray ray, double lambdaU,
                         Pcg32& rng, bool diffraction, int maxBounce, int fgRays,
                         const BeamMap* bm, const PhotonMap* pmC, const MediumStack& stk0) {
    bdpt::LocalWork work;   // its segments, for the CPU refresh (MODEM-CPU-NONDET)
    Vec3 L{0, 0, 0};
    double thr = 1.0;
    double pdfL = 0.0;
    double lambda = scene.emitSampler.sampleAt(lambdaU, pdfL);
    if (pdfL <= 0.0) return L;
    const double invPdfL = scene.invPdfLambda(lambda);

    Renderer mats; mats.diffraction = diffraction;
    // nested-dielectric medium stack, from the camera's (`stk0`, Camera::enclosure)
    MediumStack stk = stk0;
    const double area = PI * pm.radius * pm.radius;
    const double norm = (pm.nEmitted > 0 && area > 0.0)
                            ? 1.0 / (area * (double)pm.nEmitted) : 0.0;
    // The caustic map carries its OWN radius (chosen by its own buildAuto over its own,
    // far denser, population) and therefore its own 1/(pi r^2 N). When it also gathers
    // PER QUERY (PhotonMap::kGather > 0) this is only the fallback: each gather recomputes
    // the normalisation for the radius it actually used. See PhotonMap::adaptiveRadius.
    const bool causOn = (pmC != nullptr) && !pmC->photons.empty();
    const double areaC = causOn ? PI * pmC->radius * pmC->radius : 0.0;
    const double normC = (causOn && pmC->nEmitted > 0 && areaC > 0.0)
                            ? 1.0 / (areaC * (double)pmC->nEmitted) : 0.0;

    // SPECGATHER: the walk's spectral factors, beside the scalar `thr`; its wavelength came from
    // the emission sampler (above).
    SpecThr sthr(scene.specCells);
    const bool volOn = (bm != nullptr) && !bm->empty() && !scene.media.empty();
    // Beams asked for, in a scene with a medium: the camera ray takes the media's extinction whether or
    // not a beam was stored -- a medium that scatters nothing stores none, and still dims what is behind
    // it. (Without -beams mode M is blind to media, and says so.)
    const bool medOn = (bm != nullptr) && !scene.media.empty();
    // -sunnee: the sun's single scatter is marched, not gathered (beamgather.h sunNeeMarch).
    // Not gated on the map being non-empty: erasing the direct-sun chords may have emptied it.
    const bool sunOn = (bm != nullptr) && pbeams::gSunNee && scene.sunCount > 0 && !scene.media.empty();
    // GRADIENT-INDEX: the CAMERA ray has to bend too. Mode M's forward deposit has marched
    // since GRIN landed, but this gather called closestHit directly, so a GRIN lens bent the
    // photons and not the view: the lens rendered dead flat while mode R lensed the same
    // scene into a radial disc, and nothing warned. Marching here is what makes mode M see
    // its own geometry.
    const bool grinAny = grin::sceneHasGrin(scene);


    // GLOSSY-NEE. `bwNee` is the shared direct-lighting estimator (its `neeLight` is what the
    // final gather already uses); `gmis` carries the lobe density of a glossy continuation to
    // whichever emitter site it reaches, and is cleared at the top of every bounce so a mirror
    // or a dielectric can never inherit one and halve the emission behind it.
    BackwardRenderer bwNee; bwNee.diffraction = diffraction;
    const bool gneeOn = BackwardRenderer::glossyNeeOn();
    BackwardRenderer::GlossyMis gmis;
    bool hairArrival = false;   // HAIR-NEE: the previous vertex was a fiber that already took its direct light
    // The walk has gone on past a fibre's scattering (it does nowhere else but at a specular vertex):
    // a glow no emitter samples is seen only before (BackwardRenderer::GiCtx::pastDiffuse).
    bool pastDiffuse = false;
    double contSoft = 1.0;      // the softening the last vertex lends what it reaches (contSoftening)
    // The beam gather over one straight camera segment, times the throughput. The walk's spectral
    // factors and the glass the segment runs through are read at each beam's own wavelength
    // (MODEM-BEAM-CARRIER), as the surface gather reads them at each photon's.
    const auto gatherBeams = [&](const Vec3& o, const Vec3& d, double len, double aG, int glass) {
        JSpec js;
        const bool jsOn = sthr.toJSpec(js);
        return gatherPhotonBeams(scene, mats, *bm, o, d, len, aG, rng, jsOn ? &js : nullptr,
                                 glass >= 0 ? &scene.mats[glass] : nullptr) *
               (jsOn ? js.beta(thr) : thr);
    };
    // A coverage pass-through is no vertex: its bounce is refunded (capped: 256 is far above
    // anything real fur hands back), as backward.h's walks refund it.
    int nullSteps = 0;
    // The tail (b == maxBounce): the ray the last vertex sampled, for the light it reaches --
    // the volume's in-scattering along it is a vertex past the cap -- and nothing past it
    // (backward.h radianceLoop, GLOSSY-DEPTH-CAP).
    for (int b = 0; b <= maxBounce; ++b) {
        const bool tail = (b == maxBounce);
        ++work.n;
        const int    cmIdx  = stk.topMat();
        const double aGlass = (cmIdx >= 0) ? scene.mats[cmIdx].absorb(lambda) : 0.0;
        // Curved pre-pass. The volume estimator runs PER STRAIGHT SUB-SEGMENT of the curve:
        // Beam x Ray is a closest-approach between two straight lines, so a curved camera
        // ray has to be fed to it one Eikonal step at a time. That is exact rather than an
        // approximation — the radiance integral along a path is additive over its pieces,
        // and `thr` carries each piece's transmittance forward, which is precisely what
        // gatherPhotonBeams' own per-beam Tr_cam(0 -> tCam) expects (it measures from the
        // sub-segment start; the accumulated `thr` supplies everything before it).
        if (grinAny) {
            grin::MarchInfo mInfo;
            double radIn = 1.0;   // the part of mInfo.radScale already in thr
            grin::marchSegments(scene, ray,
                [&](const Vec3& so, const Vec3& sd, double slen, double&) -> bool {
                    // This step's radiance factor before what the walk gathers along it (grin.h
                    // MarchInfo; device twin dPhotonGather).
                    if (mInfo.radScale != radIn) { thr *= mInfo.radScale / radIn; radIn = mInfo.radScale; }
                    if (medOn) {
                        if (volOn && !tail) { MStatTimer _t(&mStats().beamNs, &mStats().beamN);
                          L += gatherBeams(so, sd, slen, aGlass, cmIdx); }
                        if (sunOn && !tail) L += sunNeeMarch(scene, mats, so, sd, slen, aGlass, lambda, invPdfL, rng) * thr;
                        {   double Tv[SpecThr::K + 1];
                            camMediaTrSpec(scene, mats, so, sd, slen, lambda, rng, sthr, Tv);
                            thr *= Tv[sthr.n];
                            sthr.mulVec(Tv, Tv[sthr.n]);
                        }
                    }
                    // Inside a dielectric whatever it absorbs at the walk's own wavelength: a
                    // glass clear there may absorb at a photon's (clear glass folds exactly 1).
                    if (cmIdx >= 0) {
                        thr *= std::exp(-aGlass * slen);
                        const int cmA = cmIdx; const double sl = slen;
                        sthr.mul([&](double L) { return std::exp(-scene.mats[cmA].absorb(L) * sl); },
                                 std::exp(-aGlass * slen));
                    }
                    return false;   // a camera ray never terminates in the volume here:
                                    // mode M's volume answer IS the beam gather above
                },
                // b == 0 is the camera ray; see Material::hideCamera. (photonGatherSub's
                // march above is a final-gather sub-ray and keeps the default `false`.)
                /*camHide=*/(b == 0), &mInfo);
            // What is left of the radiance factor (the straight remainder to a surface within the
            // last step); past a bend the light the walk meets was reached by no light sample (grin.h
            // MarchInfo). A factor flat in wavelength leaves the spectral carrier as it is.
            thr *= mInfo.radScale / radIn;
            if (mInfo.bent) { hairArrival = false; gmis.clear(); }
            if (thr <= 0.0) return L;
        }

        // b == 0 is the camera ray photonGather was handed (mode M's eye pass); see
        // Material::hideCamera. photonGatherSub's walk is NOT given this: a final-gather
        // sub-ray leaves a visible point, so it is an indirect ray and must see the flat.
        Hit h = closestHitIn(scene, stk, ray, Scene::kTminEps, nullptr, /*skipHair=*/false,
                                         /*skipCamHidden=*/(b == 0));
        // --- Participating media along this segment (mode M with -beams) ---------------
        // Done BEFORE `thr` takes the segment's attenuation, because each gathered beam
        // needs the transmittance to ITS OWN closest-approach point, not to the segment end.
        if (medOn) {
            const double dSeg = h.valid ? h.t : 1e30;
            if (volOn && !tail) { MStatTimer _t(&mStats().beamNs, &mStats().beamN);
              L += gatherBeams(ray.o, ray.d, dSeg, aGlass, cmIdx);
            }
            if (sunOn && !tail) L += sunNeeMarch(scene, mats, ray.o, ray.d, dSeg, aGlass, lambda, invPdfL, rng) * thr;
            // Extinction along the camera segment: what is behind the fog gets dimmed.
            {   // SPECGATHER: extinction is spectral too, and with a coloured medium badly so.
                double Tv[SpecThr::K + 1];
                camMediaTrSpec(scene, mats, ray.o, ray.d, dSeg, lambda, rng, sthr, Tv);
                thr *= Tv[sthr.n];
                sthr.mulVec(Tv, Tv[sthr.n]);
            }
            if (thr <= 0.0) return L;
        }
        if (h.valid) {                                   // Beer-Lambert in current medium
            if (cmIdx >= 0) {                        // (see the curved pre-pass's note)
                thr *= std::exp(-aGlass * h.t);      // coloured glass is spectral: SPECGATHER too
                const int cmA = cmIdx; const double tt = h.t;
                sthr.mul([&](double L) { return std::exp(-scene.mats[cmA].absorb(L) * tt); },
                         std::exp(-aGlass * h.t));
            }
        }
        if (!h.valid) {                                  // escaped -> environment
            // The env as SEEN (a camera ray or a specular chain -- see the sun note below; the
            // walk has no env NEE, so weight 1): the bilinear lookup, Scene::envRadianceSeen.
            if (scene.envIndex >= 0)
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * scene.envRadianceSeen(ray.d, lambda) * invPdfL * contSoft);
            // Directly-viewed solar disc. This walk terminates at the first diffuse
            // vertex (the density estimate returns there), so an escape reaching here is a
            // camera ray, a specular chain, or a fiber's continuation -- and a fiber's own
            // light sampling (HAIR-NEE) already took the sun, as it took every emitter the
            // emitter-hit site above skips after it (`hairArrival`). The sky is not among them:
            // the walk has no sky light sampling, so the escape is the sky's only estimate.
            if (scene.sunCount > 0 && !hairArrival)
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * bwNee.sunRadianceMis(scene, gmis, ray.d, lambda) * invPdfL * contSoft);
            return L;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return L;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            // Resolve to a LOBE and let the material switch shade it -- the coat with
            // probability R (the Fresnel reflectance at this angle), else a body lobe. The
            // coat is a material of its own (Material::coatChild), so it goes through the
            // switch's Glossy case and gets that case's glossy-NEE, throughput and MIS. An
            // earlier version reflected the ray inline here and skipped all three, which
            // broke the identity that a coat of reflectance 1 IS a glossy material: it
            // rendered 30 % bright against one (scraps/ident_*.ftsl). Same convention as
            // bdpt.h's randomWalk and the device's dResolveCompound.
            const double R = layeredCoatReflectance(scene, *mp, h, ray.d, lambda);
            const bool coatPick = rng.uniform() < R;
            int c = coatPick ? mp->coatChild
                             : mixResolveChild(scene, *mp, h, rng.uniform());  // body lobe: honours a bound weight map (device twin: dResolveCompound)
            if (c < 0) return L;
            foldCoatCoin(sthr, scene, *mp, h, ray.d, coatPick, R);   // SPECGATHER
            mp = &scene.mats[c];
        }
        const Material& m = *mp;

        // Self-emission of a directly-viewed (or specularly-seen) emitter, one-sided by the
        // geometric normal to match Vertex::Le / bkRadiance — the surface glows only from
        // the face cross(u,v) points out of.
        //
        // NOT a `return`. An emissive material still has a BSDF: a glowing DIFFUSE surface
        // both emits and reflects, so the walk falls through to the density estimate below.
        // Returning here is what made gallery_rain's grid floor render as GRID-ONLY on the
        // CPU (host `m.isLight` hit, emission returned, body dropped) and as BODY-ONLY on the
        // GPU (its emitter lookup missed the unregistered quad, so the emitter branch was never
        // taken and the grid vanished) — two mode-M paths disagreeing with each other and
        // both disagreeing with modes R and D. Measured on scraps/mini_grid.ftsl.
        const int eIdx = m.isLight ? BackwardRenderer::emitterIndexOfResolved(scene, h, m) : -1;
        if (m.isLight && dot(ray.d, h.ng) < 0.0 && !hairArrival && (eIdx >= 0 || !pastDiffuse)) {
            const double wMis = (gmis.pdf > 0.0)          // GLOSSY-NEE, as in the sub-walk
                ? bwNee.glossyHitWeight(scene, gmis, eIdx, ray.d, &h) : 1.0;
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                 * (thr * emitSlot(scene, m, h, lambda, ray.d * -1.0) * invPdfL * wMis * contSoft);
        }
        if (tail) {                                      // the light it reached, and no more --
            // past a strand it passes, which is no vertex (BackwardRenderer::tailPassesStrand)
            if (m.type == MatType::Hair && nullSteps < 256 &&
                bwNee.tailPassesStrand(scene, m, h, lambda, ray, rng)) { --b; ++nullSteps; continue; }
            return L;
        }

        // GLOSSY-NEE: cleared HERE and not at the top of the loop. `gmis` is written by the
        // PREVIOUS bounce's glossy branch and read by THIS bounce's emitter/sun sites above, so
        // a clear at the loop top erases it a few lines before the only code that wants it --
        // which leaves the connection in place with no compensating weight on the lobe-sampling
        // side, i.e. double counting wherever both strategies reach the same light. See the
        // twin note in backward.h. A hair pass-through is no vertex and puts both back.
        const BackwardRenderer::GlossyMis gmisIn = gmis;
        const bool hairArrivalIn = hairArrival;
        const double contSoftIn = contSoft;
        gmis.clear();
        hairArrival = false;
        contSoft = 1.0;   // a specular vertex lends nothing; the glossy case sets its own

        // CURVE-GATHER: a `type diffuse` strand (fur, grass, wire: any curve) is no surface to gather
        // on. The photons in a ball around a point on it lie on its neighbours, and dividing them by
        // a disc's area read a diffuse coat at 0.71-0.87 of mode R (check_fur_gather). As a hair
        // strand is, it is lit here by NEE, and the walk goes on along its diffuse lobe to the
        // surface that gathers (strandStep); the emitter that continuation meets is not counted
        // again (`hairArrival`).
        if (gatherlessStrand(m, h.fiberRadius)) {
            bool cont;
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                 * strandStep(scene, bwNee, m, h, lambda, invPdfL, rng, thr, &sthr, ray, cont);
            hairArrival = true;
            pastDiffuse = true;
            if (!cont) return L;
            continue;
        }

        const Vec3 wArr = ray.d * -1.0;   // the arrival, for the coat (coatWalkOut)
        switch (m.type) {
            case MatType::Diffuse:
            case MatType::DiffuseTransmit:
            case MatType::Fluorescent: {
                if (fgRays > 0 && m.type == MatType::Diffuse) {
                    // --- Jensen final gather (decouples visible-surface sharpness from
                    // the gather radius: the density estimate's blur now lives one bounce
                    // away, at y, not on this directly-seen surface). ---
                    // (a) DIRECT lighting from finite emitters via low-variance next-event
                    //     estimation (shadow rays), so we avoid the high variance of gather
                    //     rays randomly striking a small area light.
                    //     `neeLight` carries the shadow leg's media transmittance over the
                    //     WHOLE superposed `scene.media` vector. Until 0.254.0 it applied one
                    //     unbounded homogeneous haze built from media.front() instead, which
                    //     in a scene whose first medium is a dense bounded cloud (gallery_rain:
                    //     sigma_t 2.78, a 3 m box) multiplied every 10-30 m shadow ray by
                    //     exp(-28)..exp(-83) and deleted mode M's ENTIRE direct term (M-FGDARK).
                    BackwardRenderer bw; bw.diffraction = diffraction;
                    double rhoVis = clamp01(diffuseReflectance(scene, m, h, lambda));
                    double direct = bw.neeLight(scene, h, rhoVis, invPdfL, lambda, rng, nullptr, coatGi(m, ray.d * -1.0));
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * (thr * direct);
                    // (b) INDIRECT (+ env + specular-direct) via K cosine-weighted
                    //     hemisphere sub-rays, each querying the map ONE bounce away. The
                    //     cosine/pdf and Lambertian 1/pi cancel to rho(x), folded inside
                    //     photonGatherSub; those rays skip non-specular emitter hits so the
                    //     NEE direct term above is not double-counted.
                    Vec3 fg{0, 0, 0};
                    for (int k = 0; k < fgRays; ++k) {
                        Ray gr;   // a gather ray across the surface gathers nothing
                        if (!continueAlong(h, cosineHemisphere(h.n, rng), Lobe::Reflect, gr, scene.hitEps(h))) continue;
                        fg += photonGatherSub(scene, pm, gr, rng, diffraction, maxBounce,
                                              lambda, invPdfL, norm, h, m, ray.d * -1.0, stk,
                                              causOn ? pmC : nullptr, normC);
                    }
                    L += fg * (thr * (1.0 / (double)fgRays));
                    return L;
                }
                // Direct radius density estimate (default; also DiffuseTransmit/Fluorescent
                // visible points, which fall back here rather than final-gathering):
                //   L_r(x) = (1/N) sum_p f_r * Phi_p / (pi r^2), f_r = rho/pi (Lambertian),
                // accumulated in XYZ per photon wavelength.
                // Per-query adaptive radius on any map that asks for one (the caustic map);
                // see the twin in photonGatherSub and PhotonMap::adaptiveRadius.
                auto est = [&](const PhotonMap& M, double normFixed, double& nrmOut) {
                    MStatTimer _t(&mStats().surfNs, &mStats().surfN);
                    // The true normal, as the photons carry it (depositPhoton): one for the
                    // radius, the coverage and the photon test alike.
                    const Vec3 gn = orientedGeoN(h);
                    const double slack = M.gatherSlack(h.fiberRadius);
                    const double rq = M.adaptiveRadius(h.p, gn, h.fiberRadius, m.type == MatType::DiffuseTransmit);
                    nrmOut = normFixed;
                    if (rq != M.radius) {
                        const double a = PI * rq * rq;
                        nrmOut = (M.nEmitted > 0 && a > 0.0)
                                     ? 1.0 / (a * (double)M.nEmitted) : 0.0;
                    }
                    // M-GATHERAREA: divide by the area actually gathered from, not by the whole
                    // disc; `-gatherarea 0` leaves nrmOut the whole disc's.
                    if (const int gaM = gatherAreaSamples())
                        nrmOut *= gatherAreaScale(
                            gatherCoverage(scene, h.p, gn, rq, slack, rng, gaM, h.matId,
                                           h.fiberRadius));
                    Vec3 g{0, 0, 0};
                    M.queryR(h.p, rq, [&](const Photon& ph, double d2, int k) {
                        bool farSide;
                        if (!onGatherSurfaceSided(h.p, gn, M.pos[k], ph.n, d2, slack,
                                                  m.type == MatType::DiffuseTransmit, farSide)) return;   // another surface's
                        double rho = gatherAlbedo(scene, m, h, ph.lambda, farSide);
                        double f = rho * (1.0 / PI);
                        // SPECGATHER: the walk's reflectances belong at THIS photon's wavelength,
                        // not the camera's. Exactly 1 when they were all flat.
                        f *= sthr.ratio((double)ph.lambda);
                        g += M.cie[k] * (f * (double)ph.power);       // == cie(lambda_p), precomputed
                    });
                    return g;
                };
                double nA = norm;
                L += est(pm, norm, nA) * (nA * thr);
                if (causOn) { double nB = normC; L += est(*pmC, normC, nB) * (nB * thr); }
                return L;
            }
            case MatType::Mirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                sthr.mul([&](double L) { return clamp01(reflectSlot(scene, m, h, L)); }, r);
                if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray, scene.hitEps(h))) return L;   // across the surface
                break;
            }
            case MatType::Glossy: {
                if (!glossyCameraSide(ray.d, h.n)) return L;   // met from below ns (glossyCameraSide)
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // NEXT-EVENT ESTIMATION AT A GLOSSY VERTEX (GLOSSY-NEE). See backward.h's twin
                // for why this is MIS and not the single-estimator split the rest of mode M's
                // walk uses: a glossy lobe can be narrower than the light as easily as wider.
                if (gneeOn) {
                    const BackwardRenderer::NeeBsdf nb{&m, ray.d * -1.0};
                    if (specNeeOn()) {
                        // SPECTRAL (0.341.0): one shadow ray, evaluated over the carrier's cells,
                        // so a gold lobe under a warm light stops painting each sample one
                        // colour. The walk's own spectral throughput rides in through `ratio`.
                        double lamG[SpecThr::K], ratG[SpecThr::K], xyz[3] = {0, 0, 0};
                        for (int k = 0; k < sthr.n; ++k) {
                            lamG[k] = sthr.lamOf(k);
                            ratG[k] = sthr.ratio(lamG[k]);
                        }
                        bwNee.neeLightSpecGlossy(scene, h, nb, lamG, ratG, sthr.n, lambda, thr, rng, xyz);
                        L += Vec3(xyz[0], xyz[1], xyz[2]);
                    } else {
                        L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                             * (thr * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng,
                                                     nullptr, coatGi(m, ray.d * -1.0), nullptr, nullptr, &nb));
                    }
                }
                thr *= r;
                sthr.mul([&](double Lw) { return clamp01(reflectSlot(scene, m, h, Lw)); }, r);
                const GlossyDraw gd = drawGlossy(scene, m, h, h.n, ray.d * -1.0, rng);
                const Vec3& o = gd.o;
                if (dot(o, h.n) <= 0.0) return L;
                thr *= gd.w;   // the draw's weight: the same at every wavelength, so not sthr's
                // An iridescent lobe's interference reflectance at the pair's half-vector, into
                // the scalar and the spectral throughput alike. Where it is 0 at the walk's own
                // wavelength the path carries nothing (and sthr's ratio would divide by it).
                if (gd.irid) {
                    const double Rh = gd.R(lambda);
                    if (!(Rh > 0.0)) return L;
                    thr *= Rh;
                    sthr.mul([&](double Lw) { return gd.R(Lw); }, Rh);
                }
                if (gneeOn) {
                    gmis.pdf = bdpt::bsdfPdf(m, h.n, ray.d * -1.0, o, lambda, scene, &h);
                    gmis.from = h.p;
                    gmis.n = h.n;
                }
                if (!continueAlong(h, o, Lobe::Reflect, ray, scene.hitEps(h))) return L;
                contSoft = contSoftening(m, h, ray.d);
                break;
            }
            case MatType::Dielectric:
            case MatType::ThinFilm:
            case MatType::Multilayer: {
                // A dielectric body's surface, plain or coated (Renderer::interfaceStep); this
                // camera-side walk carries radiance across a refraction. SPECGATHER re-expresses
                // the step at each photon's wavelength (stepFactorAt) against its value here
                // (st.factor): the branch the coin took, whose probability is a coat's whole
                // interference colour, and the radiance factor.
                const Vec3 d0 = ray.d;
                const Renderer::InterfaceStep st =
                    mats.interfaceStep(scene, m, h, d0, lambda, rng, stk,
                                       Renderer::Transport::Radiance);
                if (st.absorbed) return L;
                ray = st.ray;
                thr *= st.scale;
                sthr.mulCoin([&](double Lw, double& pr) {
                    return mats.stepFactorAt(scene, m, st, h, d0, Lw, &pr);
                }, st.factor, st.prob, m.type != MatType::Dielectric);
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                const bool refl = rng.uniform() < r;
                if (refl) { if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray, scene.hitEps(h))) return L; }
                else      ray = passThrough(h, ray.d, scene.hitEps(h));
                // SPECGATHER: the coin at each photon's wavelength -- a coloured half mirror
                // reflects each wavelength by its own reflectance.
                const double pc = refl ? r : 1.0 - r;
                sthr.mulCoin([&](double Lw, double& pr) {
                    const double rL = clamp01(reflectSlot(scene, m, h, Lw));
                    return pr = refl ? rL : 1.0 - rL;
                }, pc, pc, true);
                break;
            }
            case MatType::Filter: {
                double t = clamp01(transmitSlot(scene, m, h, lambda));
                thr *= t;
                sthr.mul([&](double L) { return clamp01(transmitSlot(scene, m, h, L)); }, t);
                ray = passThrough(h, ray.d, scene.hitEps(h));
                break;
            }
            case MatType::Hair: {
                // Fiber BCSDF — scattered through, never a gather site (the photon
                // payload has no incident direction; see photonGatherSub's Hair case).
                const Vec3 wPrev{-ray.d.x, -ray.d.y, -ray.d.z};
                const HairShade hs = hairShadeAt(scene, m, h, lambda, wPrev);
                // HAIR-NEE (0.334.0). This walk collects radiance only where it finally gathers
                // -- a diffuse surface -- and a chain that scatters strand to strand through a
                // hair mass lands on the shadowed scalp or dress beneath it, so the mass's own
                // lit glow (direct light scattered by the fibers toward the eye) was never
                // counted: Alice's strands rendered 30 % darker than mode D's at 20 M photons
                // while the molded base matched. Mode R's Hair case connects to the lights at
                // every fiber vertex with the BCSDF (`neeLight` with `hs`, rho == 1: the colour
                // lives in sigma_a) and does not count an emitter its continuation then hits;
                // the same split here. The photon map is untouched (fibers are never deposited
                // on), and light that scatters off fibers onto a surface is in the map already.
                if (hairNeeOn()) {
                    bool didSpec = false;
                    if (specNeeHairOn()) {
                        // SPECTRAL (0.343.0): one shadow ray over the carrier's cells, so a blonde
                        // fiber under a warm light stops painting each sample one colour.
                        double lamG[SpecThr::K], ratG[SpecThr::K], xyz[3] = {0, 0, 0};
                        for (int k = 0; k < sthr.n; ++k) {
                            lamG[k] = sthr.lamOf(k);
                            ratG[k] = sthr.ratio(lamG[k]);
                        }
                        const double* sigB = hairSigmaCells(scene, m, h);
                        didSpec = bwNee.neeLightSpecHair(scene, h, m, hs, lamG, ratG, sthr.n,
                                                         lambda, thr, sigB, rng, xyz);
                        if (didSpec) L += Vec3(xyz[0], xyz[1], xyz[2]);
                    }
                    if (!didSpec)
                        L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                             * (thr * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng, nullptr,
                                                     BackwardRenderer::GiCtx(), &hs, nullptr, nullptr));   // a fibre has no coat
                    hairArrival = true;
                }
                double pdfH = 0.0, fv = 0.0;
                hair::LobeAngular la;                              // filled by the sample's own f()
                const Vec3 wl = hair::sampleWith(hs.b, hs.woLocal, rng, pdfH, fv, &la);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return L;
                const double cosLong =
                    hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                // A coverage pass-through returns EXACTLY -wo (negation is exact in IEEE), carries
                // weight 1 and is achromatic, so it must not be spectrally reweighted: its pdf is
                // the branch probability, not a scatter density, and dividing by it is meaningless.
                const bool passedThru = (hs.b.opacity < 1.0) && wl.x == -hs.woLocal.x &&
                                        wl.y == -hs.woLocal.y && wl.z == -hs.woLocal.z;
                const double wCam = clamp01(fv * cosLong / pdfH);   // == T = sum_p A_p
                thr *= wCam;
                // ...and no vertex: the last real one's emitter-hit bookkeeping stands (as in the
                // gather sub-walk), so light seen straight through the fiber is counted as it
                // would weigh it.
                if (passedThru) {   // ...and its bounce refunded
                    hairArrival = hairArrivalIn; gmis = gmisIn; contSoft = contSoftIn;
                    if (nullSteps < 256) { --b; ++nullSteps; }
                } else {
                    pastDiffuse = true;   // a fibre's scattering: no glow is seen past it
                }
                // SPECGATHER (0.333.0): the fiber's transmission is the most coloured factor a
                // camera walk meets -- a blonde fiber absorbs blue and green -- and until now the
                // walk folded it in at the camera's wavelength ALONE, so every photon of every
                // wavelength was scaled by one wavelength's attenuation: coloured speckle per
                // sample, and in expectation a flat AVERAGE transmission that greys the hair's
                // multiple-scatter colour. The sampled direction and its pdf are fixed by the
                // sample; only the BCSDF value moves with the absorption, so the same sample's
                // weight is evaluated at every grid wavelength (hairShadeAt inverts the
                // authored reflectance at that wavelength) and each photon is reweighted at ITS
                // wavelength, exactly as coloured glass and media already are (see SpecThr).
                // FTRACE_HAIR_SPECGATHER=0 turns the fold off (the scalar path of 0.332.0), for
                // paired A/B measurements only -- read once per process.
                static const bool hairSpecOn = [] { const char* e = std::getenv("FTRACE_HAIR_SPECGATHER"); return !(e && e[0] == '0'); }();
                if (hairSpecOn && !passedThru) {
                    // Per-lobe form (0.336.0): the angular products once, then per grid wavelength
                    // one absorption inversion, one exp and one Ap() -- a solid fiber's f() at any
                    // wavelength is exactly that (hair.h lobeAngular / fFromLobes). A bin whose own
                    // medulla is live (or a hero whose was) keeps the full rebuild: whether the
                    // medulla is live depends on the wavelength.
                    // the absorption per cell: a per-material table when the colour is constant
                    // (hairSigmaCells), else inverted at each wavelength
                    const double* bins = hairSigmaCells(scene, m, h);
                    sthr.mul([&](double lamK) {
                        // this wavelength's shade, folded from the sample's angular half where
                        // both fibers are solid and evaluated in full where not (hairFoldFCos)
                        const HairShade hk = hairShadeRetint(scene, m, h, lamK, hs,
                                                             bins ? bins[sthr.binOf(lamK)] : -1.0);
                        return clamp01(hairFoldFCos(la, hk, wl, cosLong, nullptr) / pdfH);
                    }, wCam);
                }
                const Vec3 wo = hair::toWorld(hs.fr, wl);
                ray = Ray{h.p + wo * scene.hitEps(h), wo, hairExitOffset(hs, h.n, wo, scene.hitEps(h))};
                break;
            }
            default: {                                   // Grating: approx reflect
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                sthr.mul([&](double Lw) { return clamp01(reflectSlot(scene, m, h, Lw)); }, r);
                if (!continueAlong(h, reflect(ray.d, h.n), Lobe::Reflect, ray, scene.hitEps(h))) return L;   // across the surface
                break;
            }
        }
        // A coated vertex's continuation takes the coat, at each photon's wavelength too
        // (SPECGATHER). A fibre has no coat, so its pass-through never reaches this.
        if (!coatWalkOut(scene, m, h, wArr, ray.d, lambda, thr, &sthr)) return L;
        if (thr <= 0.0) return L;
    }
    return L;
}

// ---- Camera pass driver (single camera) ---------------------------------------------
// Accumulates a SUM over spp (display divides by spp via writeFilm), matching the
// backward/BDPT convention so a chunked/progressive render sums batches.
// `sampleBase` = absolute index of the first sample rendered here; each
// (pixel, absolute sample) seeds its own stream via seedUnit(), so the
// realization is chunk-split / banding / thread-count independent (see
// BackwardRenderer::renderRows).
inline Film renderPhotonCamera(const Scene& scene, const Camera& cam, int resX, int resY,
                               const PhotonMap& pm, long long spp, int nThreads,
                               bool diffraction, int maxBounce = 32,
                               unsigned long long sampleBase = 0, int fgRays = 0,
                               const BeamMap* bm = nullptr,
                               const PhotonMap* pmC = nullptr) {
    if (nThreads < 1) nThreads = 1;
    Film out; out.resX = resX; out.resY = resY; out.alloc();
    std::vector<Film> bands(nThreads);
    std::atomic<bool> stopped{false};   // a band stopped part-way (`-stop`)
    const uint64_t nPix = (uint64_t)resX * (uint64_t)resY;
    // Under a blur: a moving scene's pass is at one time (g_passMotionU, the camera already posed
    // there by the caller); a still scene seen by a moving camera keeps its maps timeless, and
    // each sample poses the camera at a time of its own, as mode R does.
    const bool perSample = cam.timed() && cam.moves() && g_passMotionU < 0.0;
    auto worker = [&](int tid) {
        const MotionTimeScope passAt(g_passMotionU);
        const bdpt::CamWorkFlush workFlush;   // this thread's photon and beam gathers, at the end
        PosedCamera posed(cam, perSample);
        Film& f = bands[tid]; f.resX = resX; f.resY = resY; f.alloc();
        int y0 = resY * tid / nThreads, y1 = resY * (tid + 1) / nThreads;
        for (int py = y0; py < y1; ++py) {
            for (int px = 0; px < resX; ++px) {
                // Cooperative `-stop`, polled once per PIXEL. Without this the gather is
                // uninterruptible: the callers only test the stop flag between whole frames,
                // so a single-camera mode-M render could not be stopped at all, and a
                // pathologically slow gather had to be force-killed — precisely what this
                // project forbids, because tearing down a live CUDA context that way can
                // wedge the display driver. Per PIXEL rather than per scanline because the
                // thing that makes a gather slow enough to want stopping is a slow gather:
                // with an oversized `-beams` kernel radius one scanline can be half a minute,
                // and the poll (one relaxed atomic load) is free against even a fast one.
                // A partially filled band is fine: the film says it was abandoned, and every
                // caller discards it.
                if (ft::stopRequested()) { stopped = true; return; }
                const uint64_t pixIdx = (uint64_t)py * (uint64_t)resX + (uint64_t)px;
                // STRATIFIED hero wavelength (0.314.0; device twin and the reasoning: kGather
                // in render_cuda.cu): a golden-ratio sequence over the pixel's samples by their
                // ABSOLUTE index, under a per-pixel random rotation. The rotation makes every
                // sample uniform, so the estimate is unbiased; and every run of the sequence --
                // one call's samples, an epoch's, a resumed frame's, a stopped one's -- spreads
                // evenly over the spectrum, so neither the calls' sizes, which follow the clock,
                // nor where the frame stops can move or tilt it (MODEM-CPU-NONDET).
                Pcg32 rrot; seedUnit(rrot, pixIdx, 0x9E3779B97F4A7C15ULL);
                const double lambdaRot = rrot.uniform();
                for (long long s = 0; s < spp; ++s) {
                    Pcg32 rng;
                    const uint64_t unit = (sampleBase + (uint64_t)s) * nPix + pixIdx;
                    seedUnit(rng, unit, 0xA24BAED4963EE407ULL);
                    const Camera& c = perSample ? posed.at(cam.shutterTime(unitTime(unit, 0x5851F42D4C957F2DULL)))
                                                : cam;
                    const double pjx = rng.uniform(), pjy = rng.uniform();   // in order
                    Ray ray = c.genRay(px, py, pjx, pjy);
                    double u = lambdaRot + goldenFrac(sampleBase + (uint64_t)s);
                    u -= std::floor(u);
                    const double lambdaU = (u < 1.0) ? u : 0.999999999;
                    f.add(px, py, photonGather(scene, pm, ray, lambdaU, rng, diffraction, maxBounce,
                                               fgRays, bm, pmC, c.enclosure));
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    for (int t = 0; t < nThreads; ++t) out.merge(bands[t]);
    out.abandoned = stopped.load();
    return out;
}
