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
#include <vector>
#include <thread>
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
#include <atomic>

#include <chrono>

// ---- GATHER FOOTPRINT (M-GATHERAREA, `-gatherarea <M>`) --------------------------------------
// The direct density estimate divides by pi*r^2, the area of the full gather disc, while
// collecting only from the part of that disc that is real, same-facing surface. Where the disc
// overhangs -- a cap edge, a fold of cloth, a hair strand -- the divisor is too big and the
// estimate is dark in proportion. Measured on `gallery_rain`: flat ground 0 %, a cap edge -33 %,
// Alice's dress -42 %, her hair -71 %.
//
// `gatherCoverage` measures the fraction of the tangent-plane disc that has same-facing surface
// under it, by probing M points along -n. Returns 1.0 when the feature is off, so the estimate
// is bit-identical then.
//
// WHY A PROBE AND NOT AN ANALYTIC CLIP: the entry prescribes clipping each same-facing primitive
// to the disc, which is exact for triangles and IMPOSSIBLE for everything else in this scene --
// fur (the biggest single loss, mats 38-41), isosurfaces, CSG solids. One intersector call
// handles them all, and it is the same intersector the render already trusts.
// ON BY DEFAULT at 8 probes since v0.268.0. `-gatherarea 0` restores the pre-0.267 estimator.
// 8 is where the sweep plateaus: it recovers 91 % of `alice_hair`'s -68 % for 1.3-1.7x the
// gather cost, and 16 buys only a few more points. Lower is NOT better despite scoring well on
// cloth -- see the Jensen note in known-issues.md.
inline int gatherAreaSamples() {
    static const int m = [] {
        const char* e = std::getenv("FTRACE_GATHERAREA");
        return e ? std::atoi(e) : 8;
    }();
    return m;
}
inline double gatherCoverage(const Scene& scene, const Vec3& p, const Vec3& n,
                             double r, Pcg32& rng, int M) {
    if (M <= 0 || !(r > 0.0)) return 1.0;
    Vec3 t, b; onb(n, t, b);
    double area = 0.0;                 // in units of the full disc, so 1.0 == fully covered
    // THE SILHOUETTE GATE, as an adaptive early-out rather than a separate heuristic. The entry
    // proposes "only worth doing when the gather is near a silhouette or a small-feature
    // primitive", and the honest way to know that is to ask the same estimator with fewer
    // samples: probe a quarter of the budget first, and if every one of them lands on surface
    // that is flat-on (cos ~ 1), this disc is in the interior of a plane and the remaining
    // probes can only confirm it. That costs 4 rays instead of 16 on the ground plane and the
    // caps -- which is most of a frame -- while any disc that is actually truncated shows a
    // miss almost immediately and pays the full budget.
    //
    // Deliberately NOT a photon-count test: this entry already establishes that no photon
    // statistic can separate geometry from illumination, and a gate built on one would skip
    // exactly the dim truncated gathers that need correcting most.
    // `max(2, M/4)` and never M itself: at M = 4 the old form set probe0 = 4, so the check sat
    // at an index the loop never reaches and the early-out silently never fired -- which is why
    // M = 4 cost as much as M = 8 in the first sweep.
    const int probe0 = (M >= 4) ? ((M / 4 < 2) ? 2 : M / 4) : M;
    for (int i = 0; i < M; ++i) {
        if (i == probe0 && area >= (double)probe0 * 0.995)
            return 1.0;                // interior of a flat patch: nothing to correct
        // STRATIFIED in the disc, and the stratification is not a refinement -- it attacks a
        // BIAS. The estimate divides by the measured coverage, and E[1/cov] > 1/E[cov] by
        // Jensen, so noise in `cov` makes the correction too BRIGHT, the more so the fewer
        // samples. Measured: `alice_dress` reads -5.9 % at M = 4 against -15.2 % at M = 16, and
        // the M = 4 figure is not the better one -- it is a bias cancelling the layering
        // under-count below. Cutting the variance of `cov` at fixed M shrinks that bias for
        // free, and a disc stratifies exactly: equal-area rings x equal angle sectors, jittered
        // inside each cell so it stays unbiased.
        //
        // sqrt(u) within the ring puts equal expected samples per unit AREA; a linear radius
        // would over-weight the middle and report a truncated disc as fuller than it is.
        // INDEPENDENT, not stratified, and that is a decision with a measurement behind it.
        // Stratifying the radius to fight the Jensen bias below is incompatible with the
        // early-out above: `u1 = (i + xi)/M` walks the rings from the centre outwards, so the
        // gate's first M/4 probes all land in the MIDDLE of the disc, which is covered almost
        // by definition -- the gate then fires on nearly every gather and the correction stops
        // happening. Measured at M = 16: `cap_gyroid` -4.3 % independent against -16.9 %
        // radius-stratified, `alice_hair` +1.5 % against -14.8 %. (Stratifying BOTH dimensions
        // off one index is worse still, -32.9 %, because it correlates radius with angle and
        // puts every sample on a spiral.) Independent samples are spread over the whole disc by
        // construction, which is exactly what the gate needs to see.
        //
        // sqrt(u) puts equal expected samples per unit AREA; a linear radius would over-weight
        // the middle and report a truncated disc as fuller than it is.
        const double rr = r * std::sqrt(rng.uniform());
        const double ph = 2.0 * PI * rng.uniform();
        const Vec3 q = p + t * (rr * std::cos(ph)) + b * (rr * std::sin(ph));
        // Probe from r ABOVE the tangent plane straight down. `2r` of travel is what lets a
        // curved surface still count: within the disc it deviates from the plane by at most
        // ~r^2/(2R), far inside this window for any radius worth gathering at.
        const Hit h = scene.closestHit(Ray{q + n * r, n * -1.0});
        // Same 60-degree acceptance the photon query uses (dot(ph.n, h.n) < 0.5 rejects), so the
        // footprint and the estimator agree on what surface is "here".
        if (h.valid && h.t <= 2.0 * r) {
            const double c = dot(h.n, n);
            // THE PROJECTION JACOBIAN, and it is not a refinement -- without it the correction
            // overshoots badly on exactly the geometry it is for. The probe samples uniformly in
            // the TANGENT PLANE, so it measures PROJECTED area; the estimator needs SURFACE
            // area, and dA = dq / cos(tilt). A patch tilted 60 degrees carries twice the surface
            // its shadow suggests. Measured on gallery_rain without this term: alice_hair went
            // from -68.0 % to +31.1 % -- past zero, because hair is nearly all steeply-tilted
            // surface and every bit of it was counted at its projected size. Flat ground has
            // cos = 1 and is untouched either way, which is why the null control could not have
            // caught this and the truncated elements could.
            if (c >= 0.5) area += 1.0 / c;
        }
    }
    return area / (double)M;
}
// Never divide by a coverage so small that one stray probe inflates a pixel into a firefly. A
// gather that finds under a twentieth of its disc is not a measurement worth rescaling.
inline double gatherAreaScale(double cov) {
    return (cov >= 0.05) ? 1.0 / cov : 1.0;
}

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
        Renderer r; r.diffraction = diffraction;
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
    std::atomic<bool> monitorStop{false};
    std::thread monitor;
    if (stage && stage->report) {
        const long long nTotal = N + nAimed;
        monitor = std::thread([&, nTotal] {
            while (!monitorStop.load(std::memory_order_relaxed)) {
                stage->report("tracing photons",
                              tracedTotal.load(std::memory_order_relaxed), nTotal,
                              nullptr, 0.0);
                for (int i = 0; i < 20 && !monitorStop.load(std::memory_order_relaxed); ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
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
    if (monitor.joinable()) { monitorStop.store(true, std::memory_order_relaxed); monitor.join(); }

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
inline Vec3 photonGatherSub(const Scene& scene, const PhotonMap& pm, Ray ray, Pcg32& rng,
                            bool diffraction, int maxBounce, double lambda, double invPdfL,
                            double norm, const Hit& visHit, const Material& visMat,
                            const PhotonMap* pmC = nullptr, double normC = 0.0) {
    Vec3 L{0, 0, 0};
    double thr = 1.0;
    bool specularSeen = false;                           // any specular bounce so far?
    Renderer mats; mats.diffraction = diffraction;
    MediumStack stk;                                     // nested-dielectric medium stack
    const bool grinAny = grin::sceneHasGrin(scene);      // final-gather rays bend too

    // GLOSSY-NEE. `bwNee` is the shared direct-lighting estimator (its `neeLight` is what the
    // final gather already uses); `gmis` carries the lobe density of a glossy continuation to
    // whichever emitter site it reaches, and is cleared at the top of every bounce so a mirror
    // or a dielectric can never inherit one and halve the emission behind it.
    BackwardRenderer bwNee; bwNee.diffraction = diffraction;
    const bool gneeOn = BackwardRenderer::glossyNeeOn();
    BackwardRenderer::GlossyMis gmis;

    for (int b = 0; b < maxBounce; ++b) {
        if (grinAny) {
            double arc = 0.0;
            grin::marchSegments(scene, ray,
                [&](const Vec3&, const Vec3&, double slen, double&) { arc += slen; return false; });
            int cm = stk.topMat();                       // Beer-Lambert over the marched arc
            double a = (cm >= 0) ? scene.mats[cm].absorb(lambda) : 0.0;
            if (a > 0.0 && arc > 0.0) thr *= std::exp(-a * arc);
        }
        Hit h = scene.closestHit(ray);
        if (h.valid) {                                   // Beer-Lambert in current medium
            int cm = stk.topMat();
            double a = (cm >= 0) ? scene.mats[cm].absorb(lambda) : 0.0;
            if (a > 0.0) thr *= std::exp(-a * h.t);
        }
        if (!h.valid) {                                  // escaped -> environment
            // Env is collected by the gather rays directly (mode M has no separate env
            // NEE / MIS at the visible point), so add it on ANY escape — the map at a
            // diffuse hit already supplies env's INDIRECT bounces, this supplies direct.
            if (scene.envIndex >= 0) {
                double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, lambda));
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * rhoV * scene.envRadiance(ray.d, lambda) * invPdfL);
            }
            return L;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return L;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            int c = mixPickChild(*mp, rng.uniform());
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
        if (m.isLight && specularSeen && dot(ray.d, h.ng) < 0.0) {
            double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, lambda));
            // GLOSSY-NEE's lobe-sampling half; 1 (and bit-identical) unless the last bounce was
            // a glossy one that already connected to this emitter.
            const double wMis = (gmis.pdf > 0.0)
                ? bwNee.glossyHitWeight(scene, gmis,
                        BackwardRenderer::emitterIndexOfResolved(scene, m), ray.d, &h.p, &h.n)
                : 1.0;
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                 * (thr * rhoV * emitSlot(scene, m, h, lambda) * invPdfL * wMis);
        }

        // GLOSSY-NEE: cleared HERE and not at the top of the loop. `gmis` is written by the
        // PREVIOUS bounce's glossy branch and read by THIS bounce's emitter/sun sites above, so
        // a clear at the loop top erases it a few lines before the only code that wants it --
        // which leaves the connection in place with no compensating weight on the lobe-sampling
        // side, i.e. double counting wherever both strategies reach the same light. See the
        // twin note in backward.h.
        gmis.clear();

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
                    const double rq = M.adaptiveRadius(h.p, h.n);
                    nrmOut = normFixed;
                    if (rq != M.radius) {
                        const double a = PI * rq * rq;
                        nrmOut = (M.nEmitted > 0 && a > 0.0)
                                     ? 1.0 / (a * (double)M.nEmitted) : 0.0;
                    }
                    // M-GATHERAREA: divide by the area actually gathered from, not by the whole
                    // disc. `gatherAreaSamples() == 0` (the default) returns coverage 1 and
                    // leaves nrmOut untouched, so every existing render is bit-identical.
                    if (const int gaM = gatherAreaSamples())
                        nrmOut *= gatherAreaScale(gatherCoverage(scene, h.p, h.n, rq, rng, gaM));
                    Vec3 g{0, 0, 0};
                    M.queryR(h.p, rq, [&](const Photon& ph, double, int k) {
                        if (dot(ph.n, h.n) < 0.5) return;    // reject cross-surface leakage
                        double rhoY = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                        double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, ph.lambda));
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
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
            case MatType::Glossy: {
                // NEXT-EVENT ESTIMATION AT A GLOSSY VERTEX (GLOSSY-NEE). The gather ray folds
                // the VISIBLE point's reflectance into everything it reports, so `rhoV`
                // multiplies the connection exactly as it multiplies the emission above.
                // Taken before `thr *= r`: the connection carries `r` inside bsdfF.
                if (gneeOn) {
                    const double rhoV = clamp01(diffuseReflectance(scene, visMat, visHit, lambda));
                    const BackwardRenderer::NeeBsdf nb{&m, ray.d * -1.0};
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                         * (thr * rhoV * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng,
                                                        nullptr, BackwardRenderer::GiCtx{}, nullptr, nullptr, &nb));
                }
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0.0) return L;
                if (gneeOn) {
                    gmis.pdf = bdpt::bsdfPdf(m, h.n, ray.d * -1.0, o, lambda, scene, &h);
                    gmis.from = h.p;
                    gmis.n = h.n;
                }
                ray = Ray{h.p + h.n * 1e-6, o};
                break;
            }
            case MatType::Dielectric: {
                // Nested-dielectric PRIORITY resolution (Schmidt & Budge 2002): exterior
                // IOR = the medium the photon is currently inside (highest-priority stack
                // entry). Overlapping dielectrics ranked by `priority` (higher wins; lower
                // suppressed -> straight pass-through). SAFE FALLBACK to flat air<->glass
                // unless BOTH sides carry an explicit priority (priority-free scenes stay
                // bit-identical).
                bool entering = dot(ray.d, h.ng) < 0.0;
                const int mi = (int)(&m - scene.mats.data());   // true index (Mix/Layered aware)
                const int pr = m.priority;
                if (entering) {
                    const int outMat = stk.topMat();
                    const int outPri = stk.topPri();
                    const bool ranked = m.hasPriority() &&
                        (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                    if (ranked && !stk.empty() && pr <= outPri) {   // suppressed inner surface
                        stk.push(mi, pr);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && outMat >= 0)
                            ? scene.mats[outMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.push(mi, pr);
                    }
                } else {
                    MediumStack after = stk; after.popMat(mi);
                    const int newMat = after.topMat();
                    const int newPri = after.topPri();
                    const bool ranked = m.hasPriority() &&
                        (after.empty() || (newMat >= 0 && scene.mats[newMat].hasPriority()));
                    if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed: still enclosed
                        stk.popMat(mi);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && newMat >= 0)
                            ? scene.mats[newMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.popMat(mi);            // TIR stays inside mi
                    }
                }
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() < r) ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                else                   ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Filter: {
                thr *= clamp01(transmitSlot(scene, m, h, lambda));
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Hair: {
                // Fiber BCSDF — scatter through it, but do NOT gather here: the photon
                // payload carries no incident direction, so a directional BCSDF density
                // estimate is impossible (see sppm_render.h's Hair case and
                // known-issues.md). Treated like the glossy/specular cases around it.
                const Vec3 wPrev{-ray.d.x, -ray.d.y, -ray.d.z};
                const HairShade hs = hairShadeAt(scene, m, h, lambda, wPrev);
                double pdfH = 0.0, fv = 0.0;
                const Vec3 wl = hair::sample(hs.b, hs.woLocal, rng.uniform(), rng.uniform(),
                                             rng.uniform(), rng.uniform(), pdfH, fv);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return L;
                const double cosLong =
                    hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                thr *= clamp01(fv * cosLong / pdfH);       // == T = sum_p A_p
                const Vec3 wo = hair::toWorld(hs.fr, wl);
                ray = Ray{h.p + wo * hairExitOffset(hs, h.n, wo), wo};
                break;
            }
            default: {                                   // ThinFilm/Multilayer/Grating: approx reflect
                thr *= clamp01(reflectSlot(scene, m, h, lambda));
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
        }
        specularSeen = true;   // only specular cases reach here (diffuse/light returned above)
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
inline Vec3 photonGather(const Scene& scene, const PhotonMap& pm, Ray ray,
                         Pcg32& rng, bool diffraction, int maxBounce, int fgRays = 0,
                         const BeamMap* bm = nullptr, const PhotonMap* pmC = nullptr) {
    Vec3 L{0, 0, 0};
    double thr = 1.0;
    double pdfL = 0.0;
    double lambda = scene.emitSampler.sample(rng, pdfL);
    if (pdfL <= 0.0) return L;
    const double invPdfL = scene.invPdfLambda(lambda);

    Renderer mats; mats.diffraction = diffraction;
    MediumStack stk;                                     // nested-dielectric medium stack
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

    const bool volOn = (bm != nullptr) && !bm->empty() && !scene.media.empty();
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
    for (int b = 0; b < maxBounce; ++b) {
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
            grin::marchSegments(scene, ray,
                [&](const Vec3& so, const Vec3& sd, double slen, double&) -> bool {
                    if (volOn) {
                        { MStatTimer _t(&mStats().beamNs, &mStats().beamN);
                          L += gatherPhotonBeams(scene, mats, *bm, so, sd, slen, aGlass, rng) * thr; }
                        thr *= mats.mediaTransmittance(scene, so, sd, slen, lambda, rng);
                    }
                    if (aGlass > 0.0) thr *= std::exp(-aGlass * slen);
                    return false;   // a camera ray never terminates in the volume here:
                                    // mode M's volume answer IS the beam gather above
                },
                // b == 0 is the camera ray; see Material::hideCamera. (photonGatherSub's
                // march above is a final-gather sub-ray and keeps the default `false`.)
                /*camHide=*/(b == 0));
            if (thr <= 0.0) return L;
        }

        // b == 0 is the camera ray photonGather was handed (mode M's eye pass); see
        // Material::hideCamera. photonGatherSub's walk is NOT given this: a final-gather
        // sub-ray leaves a visible point, so it is an indirect ray and must see the flat.
        Hit h = scene.closestHit(ray, 1e-6, nullptr, /*skipHair=*/false,
                                 /*skipCamHidden=*/(b == 0));
        // --- Participating media along this segment (mode M with -beams) ---------------
        // Done BEFORE `thr` takes the segment's attenuation, because each gathered beam
        // needs the transmittance to ITS OWN closest-approach point, not to the segment end.
        if (volOn) {
            const double dSeg = h.valid ? h.t : 1e30;
            { MStatTimer _t(&mStats().beamNs, &mStats().beamN);
              L += gatherPhotonBeams(scene, mats, *bm, ray.o, ray.d, dSeg, aGlass, rng)
                   * thr;
              }
            // Extinction along the camera segment: what is behind the fog gets dimmed.
            thr *= mats.mediaTransmittance(scene, ray.o, ray.d, dSeg, lambda, rng);
            if (thr <= 0.0) return L;
        }
        if (h.valid) {                                   // Beer-Lambert in current medium
            if (aGlass > 0.0) thr *= std::exp(-aGlass * h.t);
        }
        if (!h.valid) {                                  // escaped -> environment
            if (scene.envIndex >= 0)
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * scene.envRadiance(ray.d, lambda) * invPdfL);
            // Directly-viewed solar disc. This walk terminates at the first diffuse
            // vertex (the density estimate returns there), so any escape reaching here
            // is a camera ray or a specular chain — never a diffuse continuation that
            // the map / NEE already credited with the sun.
            if (scene.sunCount > 0)
                L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                     * (thr * bwNee.sunRadianceMis(scene, gmis, ray.d, lambda) * invPdfL);
            return L;
        }
        const Material* mp = &scene.mats[h.matId];
        if (mp->type == MatType::Mix) {
            int c = mixResolveChild(scene, *mp, h, rng.uniform());
            if (c < 0) return L;
            mp = &scene.mats[c];
        } else if (mp->type == MatType::Layered) {
            int c = mixPickChild(*mp, rng.uniform());    // approximate: gather the body lobe
            if (c < 0) return L;
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
        // GPU (dEmitterForMat missed the unregistered quad, so the emitter branch was never
        // taken and the grid vanished) — two mode-M paths disagreeing with each other and
        // both disagreeing with modes R and D. Measured on scraps/mini_grid.ftsl.
        if (m.isLight && dot(ray.d, h.ng) < 0.0) {
            const double wMis = (gmis.pdf > 0.0)          // GLOSSY-NEE, as in the sub-walk
                ? bwNee.glossyHitWeight(scene, gmis,
                        BackwardRenderer::emitterIndexOfResolved(scene, m), ray.d, &h.p, &h.n)
                : 1.0;
            L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                 * (thr * emitSlot(scene, m, h, lambda) * invPdfL * wMis);
        }

        // GLOSSY-NEE: cleared HERE and not at the top of the loop. `gmis` is written by the
        // PREVIOUS bounce's glossy branch and read by THIS bounce's emitter/sun sites above, so
        // a clear at the loop top erases it a few lines before the only code that wants it --
        // which leaves the connection in place with no compensating weight on the lobe-sampling
        // side, i.e. double counting wherever both strategies reach the same light. See the
        // twin note in backward.h.
        gmis.clear();

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
                    double direct = bw.neeLight(scene, h, rhoVis, invPdfL, lambda, rng);
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda)) * (thr * direct);
                    // (b) INDIRECT (+ env + specular-direct) via K cosine-weighted
                    //     hemisphere sub-rays, each querying the map ONE bounce away. The
                    //     cosine/pdf and Lambertian 1/pi cancel to rho(x), folded inside
                    //     photonGatherSub; those rays skip non-specular emitter hits so the
                    //     NEE direct term above is not double-counted.
                    Vec3 fg{0, 0, 0};
                    for (int k = 0; k < fgRays; ++k) {
                        Ray gr{h.p + h.n * 1e-6, cosineHemisphere(h.n, rng)};
                        fg += photonGatherSub(scene, pm, gr, rng, diffraction, maxBounce,
                                              lambda, invPdfL, norm, h, m,
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
                    const double rq = M.adaptiveRadius(h.p, h.n);
                    nrmOut = normFixed;
                    if (rq != M.radius) {
                        const double a = PI * rq * rq;
                        nrmOut = (M.nEmitted > 0 && a > 0.0)
                                     ? 1.0 / (a * (double)M.nEmitted) : 0.0;
                    }
                    // M-GATHERAREA: divide by the area actually gathered from, not by the whole
                    // disc. `gatherAreaSamples() == 0` (the default) returns coverage 1 and
                    // leaves nrmOut untouched, so every existing render is bit-identical.
                    if (const int gaM = gatherAreaSamples())
                        nrmOut *= gatherAreaScale(gatherCoverage(scene, h.p, h.n, rq, rng, gaM));
                    Vec3 g{0, 0, 0};
                    M.queryR(h.p, rq, [&](const Photon& ph, double, int k) {
                        if (dot(ph.n, h.n) < 0.5) return;    // reject cross-surface leakage
                        double rho = clamp01(diffuseReflectance(scene, m, h, ph.lambda));
                        double f = rho * (1.0 / PI);
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
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
            case MatType::Glossy: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                // NEXT-EVENT ESTIMATION AT A GLOSSY VERTEX (GLOSSY-NEE). See backward.h's twin
                // for why this is MIS and not the single-estimator split the rest of mode M's
                // walk uses: a glossy lobe can be narrower than the light as easily as wider.
                if (gneeOn) {
                    const BackwardRenderer::NeeBsdf nb{&m, ray.d * -1.0};
                    L += Vec3(cieX(lambda), cieY(lambda), cieZ(lambda))
                         * (thr * bwNee.neeLight(scene, h, 1.0, invPdfL, lambda, rng,
                                                 nullptr, BackwardRenderer::GiCtx{}, nullptr, nullptr, &nb));
                }
                thr *= r;
                Vec3 o = sampleGlossy(reflect(ray.d, h.n), materialRoughness(scene, m, h), rng);
                if (dot(o, h.n) <= 0.0) return L;
                if (gneeOn) {
                    gmis.pdf = bdpt::bsdfPdf(m, h.n, ray.d * -1.0, o, lambda, scene, &h);
                    gmis.from = h.p;
                    gmis.n = h.n;
                }
                ray = Ray{h.p + h.n * 1e-6, o};
                break;
            }
            case MatType::Dielectric: {
                // Nested-dielectric PRIORITY resolution (Schmidt & Budge 2002): exterior
                // IOR = the medium the photon is currently inside (highest-priority stack
                // entry). Overlapping dielectrics ranked by `priority` (higher wins; lower
                // suppressed -> straight pass-through). SAFE FALLBACK to flat air<->glass
                // unless BOTH sides carry an explicit priority (priority-free scenes stay
                // bit-identical).
                bool entering = dot(ray.d, h.ng) < 0.0;
                const int mi = (int)(&m - scene.mats.data());   // true index (Mix/Layered aware)
                const int pr = m.priority;
                if (entering) {
                    const int outMat = stk.topMat();
                    const int outPri = stk.topPri();
                    const bool ranked = m.hasPriority() &&
                        (stk.empty() || (outMat >= 0 && scene.mats[outMat].hasPriority()));
                    if (ranked && !stk.empty() && pr <= outPri) {   // suppressed inner surface
                        stk.push(mi, pr);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && outMat >= 0)
                            ? scene.mats[outMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.push(mi, pr);
                    }
                } else {
                    MediumStack after = stk; after.popMat(mi);
                    const int newMat = after.topMat();
                    const int newPri = after.topPri();
                    const bool ranked = m.hasPriority() &&
                        (after.empty() || (newMat >= 0 && scene.mats[newMat].hasPriority()));
                    if (ranked && newMat >= 0 && pr <= newPri) {    // suppressed: still enclosed
                        stk.popMat(mi);
                        ray = Ray{h.p + ray.d * 1e-6, ray.d};
                    } else {
                        const double extIor = (ranked && newMat >= 0)
                            ? scene.mats[newMat].ior(lambda) : 1.0;
                        bool transmitted = false;
                        ray = mats.refractOrReflect(scene, m, h, ray.d, lambda, rng, &transmitted, extIor);
                        if (transmitted) stk.popMat(mi);            // TIR stays inside mi
                    }
                }
                break;
            }
            case MatType::HalfMirror: {
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                if (rng.uniform() < r) ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                else                   ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Filter: {
                double t = clamp01(transmitSlot(scene, m, h, lambda));
                thr *= t;
                ray = Ray{h.p + ray.d * 1e-6, ray.d};
                break;
            }
            case MatType::Hair: {
                // Fiber BCSDF — scattered through, never a gather site (the photon
                // payload has no incident direction; see photonGatherSub's Hair case).
                const Vec3 wPrev{-ray.d.x, -ray.d.y, -ray.d.z};
                const HairShade hs = hairShadeAt(scene, m, h, lambda, wPrev);
                double pdfH = 0.0, fv = 0.0;
                const Vec3 wl = hair::sample(hs.b, hs.woLocal, rng.uniform(), rng.uniform(),
                                             rng.uniform(), rng.uniform(), pdfH, fv);
                if (!(pdfH > 0.0) || !(fv > 0.0)) return L;
                const double cosLong =
                    hair::safeSqrt(1.0 - hair::sqr(hair::clampd(wl.x, -1.0, 1.0)));
                thr *= clamp01(fv * cosLong / pdfH);       // == T = sum_p A_p
                const Vec3 wo = hair::toWorld(hs.fr, wl);
                ray = Ray{h.p + wo * hairExitOffset(hs, h.n, wo), wo};
                break;
            }
            default: {                                   // ThinFilm/Multilayer/Grating: approx reflect
                double r = clamp01(reflectSlot(scene, m, h, lambda));
                thr *= r;
                ray = Ray{h.p + h.n * 1e-6, reflect(ray.d, h.n)};
                break;
            }
        }
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
    const uint64_t nPix = (uint64_t)resX * (uint64_t)resY;
    auto worker = [&](int tid) {
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
                // A partially filled band is fine — every caller discards the film on a stop.
                if (ft::stopRequested()) return;
                const uint64_t pixIdx = (uint64_t)py * (uint64_t)resX + (uint64_t)px;
                for (long long s = 0; s < spp; ++s) {
                    Pcg32 rng;
                    seedUnit(rng, (sampleBase + (uint64_t)s) * nPix + pixIdx,
                             0xA24BAED4963EE407ULL);
                    Ray ray = cam.genRay(px, py, rng.uniform(), rng.uniform());
                    f.add(px, py, photonGather(scene, pm, ray, rng, diffraction, maxBounce,
                                               fgRays, bm, pmC));
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    for (int t = 0; t < nThreads; ++t) out.merge(bands[t]);
    return out;
}
