#pragma once
// ---------------------------------------------------------------------------------------
// Radiance cache — biased early path termination (opt-in; see -radcache in REFERENCE.md).
//
// WHAT IT IS. A world-space cache of *indirect* irradiance at diffuse surfaces. Once a path
// has bounced far enough that its remaining contribution is a slowly-varying wash rather
// than recognisable detail, it stops tracing and reads the answer out of the cache instead.
// That trades a small, bounded bias for an unbounded cut in path length: the cost of deep
// GI stops growing with bounce depth.
//
// It is BIASED, on purpose, and therefore OFF by default and never enabled implicitly. Mode
// `R` is this renderer's unbiased reference — every other mode is validated by comparing
// against it — so a mode `R` render with no `-radcache` on the command line must remain
// bit-for-bit what it always was.
//
// ---------------------------------------------------------------------------------------
// WHERE THE DATA COMES FROM: A DEDICATED UPDATE PASS
//
// The first version of this trained the table purely off the camera paths already being
// traced: a finished path measured its own incident radiance for free (see the identity
// below) and deposited it. Elegant, and it cost nothing — but measurement killed it:
//
//   * DATA STARVATION. Camera paths only deposit where camera paths happen to go. On
//     fur_creature.ftsl the whole render finished at 0.4 samples per bin and 0% termination:
//     pure overhead, no win. The cache's data rate was chained to the image's sampling
//     density, which is exactly the wrong coupling — the sparsely-visited corners that most
//     need a cheap answer are the ones that never get one.
//   * A SELECTION-BIAS MINEFIELD. "Which paths may deposit" turned out to be load-bearing to
//     within a couple of percent, because any rule that keys on where a path GOT TO also
//     keys on where it WENT (measured: 1.8% dark on cornell.ftsl for the natural-looking
//     rule, plus a separate 1.3% from the hero/secondary split at a de-hero).
//
// So the data now comes from a DEDICATED UPDATE PASS, the shape Epic's LumenRef reference
// path tracer uses for its world radiance cache. Per progressive chunk:
//
//   1. RENDER. Camera paths read the table (confident cell => stop and read; no data => keep
//      tracing, exactly as an uncached render would) and MARK every diffuse cell they touch.
//   2. MERGE MARKS. Marked cells are created in the table, carrying a representative surface
//      point and normal but no radiance yet, and the work list is rebuilt.
//   3. UPDATE ROUNDS. Cells that still need data shoot `rays` cosine-hemisphere rays from
//      their representative point and average what comes back. Those rays read the table
//      too, so light propagates one further bounce per round.
//   4. APPLY. Each cell folds the round's samples into a cumulative mean and variance, and
//      re-tests its confidence gate.
//
// Three things fall out of that split, and they are the whole reason for it:
//
//   * The cache's data rate is now PER CELL, not per pixel. A cell that one camera path
//     grazed gets the same `rays` samples as one a million paths hammered.
//   * Camera paths no longer train the table, so their termination decision cannot select
//     the training set. The entire class of bias above simply does not arise.
//   * "No data yet" can therefore fall back to TRACING NORMALLY, for free. The first chunk
//     is exact, there is no dark start, no warm-up ramp, and no special-cased first pass.
//
// ---------------------------------------------------------------------------------------
// WHY A CUMULATIVE MEAN AND NOT LumenRef'S CAPPED MOVING AVERAGE
//
// LumenRef folds each round in with `lerp(history, new, 1/min(n+1, 12))`, i.e. an exponential
// moving average whose window is capped at twelve frames. It has to: it is a real-time
// renderer whose cells return BLACK before they have data, so the first rounds are wrong-dark
// and must be actively forgotten, and its camera moves, so a cell's true value changes.
//
// Neither applies here. A cell with no data does not answer at all — the reader keeps tracing
// — so an update ray fired in round 1 is a COMPLETE, unbiased, full-length path-tracer sample
// of the very same quantity a ray fired in round 50 measures. There is nothing to forget, and
// capping the window at twelve rounds throws away the one thing that actually fixes this
// estimator's error: more samples. (Measured, and this was the whole bug: with the capped
// window a cell's value carried the standard error of ~12 samples, roughly 30% on a smooth
// scene and several hundred percent on fur. That error is FROZEN INTO THE CELL, so every path
// that reads it gets the same wrong number — it is not noise the sampler averages away, it is
// per-cell speckle that no amount of spp removes. fur_creature.ftsl came out at 48% RMS.)
//
// So a cell accumulates `mean` and `sumsq` over every sample it has ever taken, without a cap,
// and the error falls as 1/sqrt(n) for as long as the render runs.
//
// ---------------------------------------------------------------------------------------
// THE CONFIDENCE GATE: MEASURED STANDARD ERROR, NOT A ROUND COUNT
//
// A cell may answer only when its own measured relative standard error is below `tol` in
// EVERY bin. That is the difference between a cache that degrades gracefully and one that
// explodes: on a smooth wall a cell reaches 5% in a few dozen samples and starts paying
// immediately, while on a fur coat — where a hemisphere sample varies by orders of magnitude
// between "saw the sky through a gap" and "hit the neighbouring strand" — it never does, and
// the render simply falls back to exact tracing.
//
// Two consequences make this cheap as well as safe:
//
//   * A CONFIDENT CELL STOPS BEING UPDATED. Its estimate is already a complete unbiased
//     multi-bounce answer (see above), so refining it further buys nothing. The update pass's
//     cost is therefore FRONT-LOADED and amortises to zero: in a long render almost every
//     path terminates for free.
//   * A HOPELESS CELL IS RETIRED EARLY, on a projection rather than by exhaustion. After
//     `minSamples` samples the sample variance already says how many samples the gate would
//     need, n_req = var / (tol*ref)^2; if that exceeds `maxSamples` the cell gives up on the
//     spot. Fur cells die after ~8 samples per bin instead of burning thousands each.
//
// ---------------------------------------------------------------------------------------
// THE BUDGET GOVERNOR: THE UPDATE PASS MAY NOT EAT THE RENDER
//
// The update pass is not free, and without a cap it is unbounded: it costs `rays` rays per
// live cell per round no matter how many cells there are. On fur_creature.ftsl that came to
// 8.6 M update samples against a 27 M ray budget — the update pass consumed essentially the
// entire render, and the image was starved to 48% RMS while the cache learned nothing.
//
// So a chunk may spend at most `budget` update samples per CACHE CONSULT that the chunk's
// camera paths actually made (default 0.25, i.e. a hard 25%-ish overhead ceiling), with a
// multiplier on the first chunk (`warm`) because a cold table is where the propagation
// happens. Cells are served from a rotating cursor over the work list, so a table far larger
// than one chunk's budget still gets fully covered over successive chunks instead of the
// first few thousand cells hogging every round.
//
// Tying the budget to CONSULTS rather than to wall-clock keeps the whole thing deterministic:
// the same command line produces the same table regardless of machine speed or thread count.
//
// ---------------------------------------------------------------------------------------
// WHAT EXACTLY IS STORED, AND WHY IT NEEDS NO EXTRA ESTIMATOR
//
// At a Lambertian vertex the backward tracer continues with a cosine-sampled direction and
// multiplies throughput by the albedo alone (backward.h, `thr[i] *= rho[i] / q`), because
// f*cos/pdf = (rho/pi)*cos / (cos/pi) = rho exactly. So every contribution the path adds
// *after* that vertex is
//
//     dL[i] = thr_after[i] * L_i(omega)
//
// where L_i(omega) is the radiance arriving along the sampled direction, and its
// cosine-weighted average over the hemisphere is precisely
//
//     mean of L_i(omega)  =  (1/pi) * INT L_i cos dw  =  E / pi
//
// which is the one number a terminating path has to multiply by rho to close out its
// estimate. So a cell stores E/pi and the lookup is a multiply.
//
// THE UPDATE RAY MEASURES THAT NUMBER DIRECTLY. A ray launched from the cell with unit
// throughput along a cosine-sampled direction returns exactly one sample of L_i(omega). It is
// launched at `bounce0 = 1` with `specularArrival = false` and `contBsdfPdf = cos/pi`, which
// is what makes an emitter hit on the first segment carry its correct MIS weight against the
// NEE the READER will do.
//
// DIRECT LIGHT IS INCLUDED, and must be: an update ray does NEE at every vertex it reaches,
// starting one vertex past the cell. What it excludes is the NEE at the cell itself, which
// is exactly right — a terminating path does its own NEE first and then adds the cached
// term, so the two partition the integral with no gap and no double count.
//
// UNLIKE LumenRef, cells do NOT store surface albedo. LumenRef caches outgoing radiance and
// so must multiply by the cell's own BaseColor; this cache stores incident E/pi and lets the
// READER apply its own rho, evaluated at the reader's actual shading point with its actual
// textures. One less quantity to cache, and no texture detail lost to cell quantisation.
//
// ---------------------------------------------------------------------------------------
// KEYING: POSITION x NORMAL, ON A CAMERA-CENTRED CLIPMAP
//
// Irradiance is a function of position *and* orientation — the two faces of a wall sit in
// one voxel and see opposite worlds — so a cell is keyed on a quantised normal as well as a
// quantised position. Six major-axis buckets separate the faces of any box and, more to the
// point, always separate a surface from its own back side. (Limit: a curved surface averages
// its irradiance over a 90-degree normal cone within a cell. Indirect irradiance varies
// slowly enough over such a cone that this is not visible in practice, and the cell size
// bounds it further. Documented in known-issues.md.)
//
// Cell size grows with distance from the camera, doubling per clipmap level, so a cell stays
// roughly constant in *screen* size. A fixed world-space cell is the wrong shape for this
// job: sized for the foreground it explodes in cell count across a large scene, and sized
// for the whole scene it smears the foreground.
//
// The grid is an OPEN-ADDRESSED HASH, not the dense array `photonmap.h` uses. That map is
// built once from a bounded photon pass and can afford nx*ny*nz; this one is sparse (only
// surfaces the camera actually reached), lives across a whole progressive render, and spans
// several clipmap levels at once, so a dense allocation per level is not affordable.
//
// ---------------------------------------------------------------------------------------
// SPECTRAL, NOT RGB
//
// Cells store irradiance in `kBins` equal-width wavelength bins across the render's spectral
// range, NOT XYZ. Collapsing to XYZ and upsampling back would round-trip every cached bounce
// through a metameric identification, which is precisely the error this renderer exists to
// avoid: a sodium lamp, a fluorescent gel or a narrowband LED bouncing off a saturated wall
// would come back as a smooth reconstruction with the same tristimulus and a different
// spectrum, and every subsequent spectral interaction would compound it. Bins cost memory
// and sample rate, and that is the right thing to spend them on.
//
// The update pass STRATIFIES its wavelengths across the bins rather than drawing them from
// the emission CDF: a cell's round r fires ray k into bin (r*rays + k) mod kBins. The counter
// is PER CELL (not a global round index), which is what keeps the rotation from aliasing
// against however many rounds a chunk happens to run.
//
// UPDATE RAYS ARE MONOCHROMATIC (one wavelength, not a hero bundle), and that is a
// correctness requirement rather than a simplification. A bundle that meets a dispersive
// interface or a fur fiber DE-HEROS: the secondaries are killed and the hero is boosted x C,
// which keeps the BUNDLE AVERAGE unbiased (that is all the film needs) but leaves each
// individual channel wrong — the hero too bright past the boost, the secondaries truncated.
// A cache bins per wavelength, so bundle-average correctness is not enough, and "drop the
// rays that de-hero'd" would select on where the ray went. One wavelength per ray cannot
// de-hero at all, so the whole class of error is absent by construction.
//
// What a cell stores is PHYSICAL radiance: update rays run with invPdf == 1, and the reader
// restores its own 1/pdf(lambda) weight at its own wavelength. This matters under a
// narrowband source, where pdf(lambda) can swing by orders of magnitude across one ~29 nm
// bin: a cell that held L*invPdf would hand the reader a number weighted by somebody else's
// wavelength sampling.
//
// ---------------------------------------------------------------------------------------
// VERIFYING THE CACHE AGAINST THE READERS THEMSELVES
//
// Everything above decides whether a cell may answer using only the update pass's own
// samples. That is not enough, and the reason is worth stating plainly, because it is the
// single largest source of error this cache ever had.
//
// A confidence gate built on measured variance can only see the part of the distribution its
// samples reached. Indirect radiance routinely has energy in events rarer than 1/n: a caustic
// through a glass sphere, a specular highlight reflected off a small bright object, a sliver
// of sky through a gap in a canopy. A cell that draws a few thousand cosine-distributed rays
// and misses such an event reports a small mean AND a small variance, and the gate waves it
// through. It is then confidently, permanently, and invisibly too dark. Measured with
// -radcache-audit on scenes/cornell.ftsl (a dispersive SF10 sphere under an area light):
// -14.7% systematic error per read, while the same scene with the sphere made diffuse showed
// -0.3%. No amount of samples per PIXEL removes it, because every path that reads the cell
// receives the same wrong number.
//
// The fix uses the one source of information the update pass does not have: the readers. Cache
// consults outnumber update samples by three or four orders of magnitude, and every consult
// stands at a vertex where the exact tail COULD be traced. So a small random fraction of them
// (`-radcache-validate`, default 5%) do exactly that -- they read the cell, remember what it
// offered, and then IGNORE the offer and trace the path to full length anyway. The difference
// between the path's final radiance and its radiance at that moment is precisely the quantity
// the cached number claimed to equal. Per cell, the running ratio
//
//     r = sum(tail actually delivered) / sum(value offered)
//
// is an unbiased estimate of the cell's total systematic error AS USED -- not merely its
// sampling error, but its cell-averaging error, its normal-cone error, and its unsampled
// tails, all at once, weighted exactly as the readers weight them, because the validation
// paths ARE readers drawn at random from the population that reads the cell.
//
// Three things follow from a cell's verification record, applied at the next merge (never
// within the chunk that produced it, so a correction is never correlated with the paths it is
// applied to):
//
//   * ratio well determined (its own relative standard error <= tol/2): adopt it. `corr`
//     multiplies everything the cell offers from then on, and the cell is now right for a
//     reason that does not depend on the update pass having found the rare events.
//   * ratio poorly determined but provably far from 1 (|r-1| > tol + 2 SE): the cell is
//     wrong and cannot be pinned down -- retire it. Readers fall back to exact tracing, which
//     costs speed and nothing else.
//   * not enough validation paths yet: leave corr at 1 and carry on. This is what keeps the
//     warm-up fast; the cache starts answering on the update pass's word and is audited into
//     correctness as the render proceeds.
//
// Validation paths are themselves EXACT (they never terminate on the cache), so they are not
// merely a diagnostic overhead -- their pixels are unbiased samples. The cost is 5% of the
// work the cache was saving, which is nothing next to being wrong.
//
// Selection is by an independent coin flip at the vertex, BEFORE anything about the tail is
// known. That distinction is the whole ballgame: an earlier design deposited from camera
// paths chosen by their fate ("deposit unless you terminated on the cache") and measured 1.8%
// dark on cornell.ftsl. Here the coin cannot see where the path is going.
//
// ---------------------------------------------------------------------------------------
// THREADING: READ-ONLY PASSES, PER-THREAD BANKS, MERGE AT THE CHUNK BOUNDARY
//
// During a chunk the table is immutable, so every lookup is a plain read with no atomics and
// no false sharing. Marks land in a per-thread `RadCacheBank` (which dedups against a small
// direct-mapped filter, so a million paths crossing one cell cost one mark, not a million)
// and are merged between chunks. The update pass likewise writes only to a per-slot SCRATCH
// buffer, never to the live values, so the rays of a round all see the same table no matter
// how the OS scheduled them. `apply()` then folds scratch into the cell single-threaded.
//
// That double-buffering is not just tidiness: it is what makes a chunked render and a
// one-shot render of the same budget produce the same cache, and it is the same shape the
// GPU port wants (append buffer + one merge kernel + one apply kernel).
// ---------------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "linalg.h"

// Wavelength bins per cell. 16 spans 360-830 nm at ~29 nm, which resolves a sodium doublet
// as a spike in one bin rather than as a broadband lift. Raising it costs memory per cell and,
// far more importantly, SAMPLE RATE: a cell's gate is per bin, so doubling the bins doubles
// the rays needed before the cell may answer.
static constexpr int kRadCacheBins = 16;

// Normal buckets: six cube faces, each split 3x3, so a bucket accepts a ~30-degree cone
// instead of a 90-degree one. See normalBucket() for why the split is 3x3 and not 2x2.
static constexpr int kRadCacheNormalBuckets = 54;   // 6 cube faces x a 3x3 split (normalBucket)

// Cosine-hemisphere rays one update round shoots per cell — each monochromatic, each taking
// the next bin of the cell's own rotating schedule. At the default a round sweeps the whole
// spectrum once.
static constexpr int kRadCacheRays = kRadCacheBins;

// Samples per bin a cell must have before its measured standard error is believed at all. A
// standard error estimated from three samples is itself mostly noise and will happily claim
// confidence a cell has not earned.
static constexpr int kRadCacheMinSamples = 8;

// Samples per bin a cell may be projected to need before it is written off as hopeless (see
// the confidence-gate note above). Cells that would need more than this never answer and are
// never updated again, which is what stops a fur coat from consuming the render.
static constexpr int kRadCacheMaxSamples = 4096;

// Required relative standard error, per bin, before a cell may answer. This is the knob that
// trades bias for speed, and it is very nearly the systematic error the cached term carries
// AT THE MOMENT A CELL FIRST ANSWERS -- the error keeps falling from there (tier 2, below).
// 0.2 is chosen to be reachable inside a normal render's budget: a hemisphere sample of
// indirect radiance typically has a relative standard deviation around 2, so 0.2 needs of the
// order of 100 samples per bin, and 0.05 would need 1600.
static constexpr double kRadCacheTol = 0.2;

// Update samples a chunk may spend per cache consult its camera paths made. The overhead
// ceiling: at 0.25 the update pass can never cost much more than a quarter of the trace it is
// trying to shorten, whatever the cell count.
static constexpr double kRadCacheBudget = 0.25;

// First-chunk multiplier on that budget. The cold table is where multi-bounce propagation
// happens and where every subsequent chunk's saving comes from, so it is worth front-loading.
static constexpr int kRadCacheWarm = 8;

// Chunks a cell may go unmarked before its slot is up for reuse. A still camera never
// reaches this; a moving one (the interactive viewer, a flyby) recycles what left the frame.
static constexpr int kRadCacheMaxUnused = 16;

// One cache cell.
//
// `mean` is the live, readable value (E/pi per bin) and is a plain CUMULATIVE mean over every
// update sample the cell has ever taken — see the note above on why this is not LumenRef's
// capped moving average. `sumsq`/`cnt` carry the running second moment, which is what the
// confidence gate is computed from.
//
// `p`/`n` are a representative surface point and normal, taken from the first path that marked
// the cell, and are where the update pass launches from. Storing a REAL surface point rather
// than the cell centre matters: a cell centre is usually inside or behind geometry, and a ray
// from there starts occluded.
struct RadCacheCell {
    float    mean[kRadCacheBins];   // E/pi per wavelength bin (the value a reader multiplies)
    float    sumsq[kRadCacheBins];  // sum of squares, for the standard-error gate
    float    mx[kRadCacheBins];     // largest single sample, for the heavy-tail guard
    uint16_t cnt[kRadCacheBins];    // samples folded into each bin (bounded by maxSamples)
    double   px, py, pz;            // representative surface point (double: it is a ray origin)
    float    nx, ny, nz;            // representative shading normal, unit
    uint32_t lastMark = 0;          // pass index at which a path last marked this cell
    uint32_t upRound  = 0;          // update rounds absorbed; drives the per-cell bin schedule
    uint8_t  ok   = 0;              // 1 = confident: may answer, and needs no more samples
    uint8_t  dead = 0;              // 1 = retired: too noisy to ever pass the gate (see above)
    // ---- verification (see "VERIFYING THE CACHE AGAINST THE READERS THEMSELVES") --------
    float    corr  = 1.0f;          // multiplier the verification has learned for this cell
    double   vOffer = 0.0;          // sum of the RAW values this cell offered validation paths
    double   vTail  = 0.0;          // sum of what those paths' traced tails actually delivered
    double   vTail2 = 0.0;          // ...and of their squares, for the ratio's standard error
    uint32_t vN     = 0;            // validation paths scored against this cell
    Vec3 point()  const { return Vec3{px, py, pz}; }
    Vec3 normal() const { return Vec3{(double)nx, (double)ny, (double)nz}; }
};

// Per-cell scratch for one round: the round's samples, kept out of the cell so every ray of a
// round reads the same pre-round table.
struct RadCacheScratch {
    float    sum[kRadCacheBins];
    float    sumsq[kRadCacheBins];
    float    mx[kRadCacheBins];
    uint16_t cnt[kRadCacheBins];
};

// One verification record: "a camera path read cell `key`, was offered `offer`, and then went
// on to actually collect `tail`". Keyed rather than slot-indexed because a slot can be
// recycled between the chunk that produced the record and the merge that consumes it.
struct RadCacheVal {
    uint64_t key;
    double   offer, tail;
};

// One mark: "a path touched this cell, here is a surface point on it". Carries the key so the
// merge does not re-derive it (and cannot disagree with the thread that produced it).
struct RadCacheMark {
    uint64_t key;
    Vec3     p, n;
};

// Per-thread bank, mirroring photonmap.h's PhotonBank: plain appendable vectors that the
// merge concatenates, so the trace never synchronises.
//
// The mark filter is the reason marking is affordable at all. Every diffuse vertex of every
// camera path wants to mark, which is tens of millions of pushes per chunk for a few tens of
// thousands of distinct cells. A direct-mapped table of recently-seen keys collapses that to
// (very nearly) one push per cell per thread; a filter MISS costs only a duplicate mark,
// which the merge folds away, so the filter never has to be exact.
struct RadCacheBank {
    static constexpr int kFilterBits = 14;                    // 16 k slots = 128 KB/thread
    static constexpr size_t kFilterSize = (size_t)1 << kFilterBits;
    std::vector<uint64_t>     seen;     // direct-mapped recently-marked keys
    std::vector<RadCacheMark> marks;
    std::vector<RadCacheVal>  vals;     // verification records (see RadCacheVal)
    // Per-thread tallies for the -radcache status line AND for the budget governor, folded
    // into the table at the merge. Counted here rather than in the table because the table is
    // read-only during a pass.
    long long nTerm = 0;    // paths that ended by reading the cache
    long long nMiss = 0;    // vertices that consulted the cache and had to keep tracing

    // -radcache-audit only (see BackwardRenderer::radAudit). The audit answers the one
    // question a whole-image RMS cannot: when the cache says "the rest of this path is worth
    // X", is X right? It runs the cache read and the real tail SIDE BY SIDE on the same
    // vertex -- reading the cell but NOT terminating -- and accumulates both. Their ratio is
    // the cache's systematic error PER READ, measured directly, independently of how much of
    // the image the cached term happens to carry and without the render noise that swamps an
    // image-space comparison. Off by default and free when off.
    // The traced tail is the noisy half of the comparison -- on a scene with caustics a
    // single path can carry thousands of times the mean -- so its sum of squares is carried
    // too, and the report prints the audit's OWN standard error. Without that the audit
    // silently reports its noise as if it were bias (measured: the same build swung from
    // -26% to +27% across cell sizes on a dispersive Cornell box, all of it sampling error).
    double    auditCache  = 0.0;  // sum of thr*rho*E/pi*invPdf the cache offered
    double    auditTrace  = 0.0;  // sum of what the traced remainder actually delivered
    double    auditTrace2 = 0.0;  // sum of squares of the same, for that sum's standard error
    long long auditN      = 0;    // audited vertices (first cache-readable vertex per path)

    void ensure() { if (seen.empty()) seen.assign(kFilterSize, ~0ull); }

    // Returns true if this key was not obviously marked already by this thread.
    bool mark(uint64_t k, size_t slot, const Vec3& p, const Vec3& n) {
        ensure();
        uint64_t& s = seen[slot & (kFilterSize - 1)];
        if (s == k) return false;
        s = k;
        marks.push_back(RadCacheMark{k, p, n});
        return true;
    }
    // The filter must be reset at the CHUNK boundary, not at every merge. It is a pure
    // optimisation only as long as forgetting a key is possible: a cell can be recycled out
    // of the table (see insertMark), and a thread that still believed it had marked it would
    // never mark it again, so the cell would never come back. Once per chunk is often enough
    // to be free and rare enough to keep the dedup rate high within a chunk.
    void resetFilter() { if (!seen.empty()) std::fill(seen.begin(), seen.end(), ~0ull); }
    size_t size() const { return marks.size(); }
};

struct RadianceCache {
    // ---- configuration (set once, before the first pass) ----
    double baseCell   = 0.05;   // level-0 cell edge, world units
    double baseDist   = 1.0;    // distance at which level 0 applies; level doubles per 2x
    int    maxLevel   = 8;      // clipmap depth; beyond this the cell size stops growing
    double lambdaLo   = 360.0;  // spectral range the bins span (mirrors the render range)
    double lambdaHi   = 830.0;
    int    minBounce  = 2;      // no termination before this bounce index
    int    rays       = kRadCacheRays;        // update rays per cell per round
    int    minSamples = kRadCacheMinSamples;  // per bin, before the gate is trusted
    int    maxSamples = kRadCacheMaxSamples;  // per bin, before a cell is retired
    double tol        = kRadCacheTol;         // required relative standard error per bin
    double budget     = kRadCacheBudget;      // update samples per cache consult
    int    warm       = kRadCacheWarm;        // first-chunk budget multiplier
    int    maxUnused  = kRadCacheMaxUnused;   // chunks before a slot may be recycled
    double jitter     = 0.0;    // lookup position dither, in cell widths (see lookupBundle)
    Vec3   camera{0, 0, 0};     // clipmap centre

    // A bin whose mean is this far below the cell's brightest bin is held to an ABSOLUTE
    // accuracy instead of a relative one. Without it a bin that is legitimately near zero (a
    // wavelength the scene simply has no light at) could never pass a relative-error test, and
    // one such bin vetoes the whole cell. Its absolute error contributes nothing to the image
    // anyway, which is exactly why the floor is measured against the cell's own peak.
    static constexpr double kFloorFrac = 0.05;
    // Heavy-tail guard (see apply()): the largest single sample in a bin may be at most this
    // share of the bin's total. 0.1 means "no one sample may carry a tenth of the answer",
    // i.e. an effective sample count of at least ten -- loose enough that ordinary indirect
    // light, whose samples are within an order of magnitude of one another, never trips it.
    static constexpr double kDomFrac = 0.1;
    // Validation paths a cell needs before its measured ratio is allowed to decide anything.
    static constexpr int    kMinValid  = 64;
    // Bounds on the learned correction. A cell that wants to be multiplied by more than this
    // is not mis-estimated, it is measuring something else entirely; retire it instead.
    static constexpr double kCorrLo = 0.25, kCorrHi = 4.0;

    // ---- table ----
    // Open addressing with linear probing. `key` is the full 64-bit cell key; kEmpty marks a
    // free slot. Power-of-two capacity so the modulo is a mask.
    static constexpr uint64_t kEmpty = ~0ull;
    std::vector<uint64_t>        key;
    std::vector<RadCacheCell>    cell;
    std::vector<RadCacheScratch> scratch;
    // Slots marked recently enough to still be in play, split into the two service tiers the
    // budget is spent on in order (see the two-tier note above). Rebuilt by mergeMarks().
    std::vector<uint32_t> live;
    std::vector<uint32_t> work;     // tier 1: not confident yet, not retired
    std::vector<uint32_t> refine;   // tier 2: already answering, but still improving
    size_t cursor = 0;          // rotating service point into `work`
    size_t rcursor = 0;         // rotating service point into `refine`
    size_t mask = 0;
    size_t used = 0;
    uint32_t pass = 0;          // monotonic chunk counter, stamped into cell.lastMark

    // Stats, for the -radcache status line.
    long long nSample = 0, nEvicted = 0, nTerm = 0, nMiss = 0, nUpdated = 0;
    long long nOk = 0, nDead = 0;      // cells that reached the gate / were retired
    long long nCorrected = 0;          // cells carrying a verification-learned correction
    long long nRejected  = 0;          // cells the verification retired this merge
    long long chunkConsults = 0;       // consults made by the chunk just rendered
    long long chunkTerm     = 0;       // ...of which terminated (status line / diagnostics)
    // NO SECOND "payoff" GOVERNOR, and the measurement that ruled one out. The obvious idea is
    // to taper the update budget on a scene where the table never starts earning -- on
    // scenes/fur_creature.ftsl only 0.3% of consults ever terminate, because 8.7 k cells
    // cannot cover a fur coat inside one render. It was implemented and thrown away, for two
    // reasons, both worth recording so it does not get reinvented:
    //
    //  1. There is nothing to recover. At a FIXED 128 spp on that scene the update pass costs
    //     210 k rays out of 23.86 M -- 0.9%, ~3.7 rays per update sample. The 15% "loss" that
    //     motivated the governor was WALL-CLOCK, and repeat runs of the identical cache-off
    //     render on this machine came back 18.5 s / 25.8 s / 27.2 s. The ray count is the
    //     deterministic measure; the clock was measuring the machine, not the cache.
    //  2. It cannot tell "not paying" from "not paying YET". A cold table terminates nothing
    //     by construction, so a taper keyed on the termination rate fires hardest exactly
    //     during the warm-up it needs to survive: on scenes/cornell.ftsl it drove terminations
    //     from 39% to 0% and the ray count from 133.7 M straight back to the cache-off 146.7 M.
    //
    // The existing budget governor (samples per consult, warm-multiplied on the cold chunk) is
    // already the bound that matters, because it is proportional to the work the cache is
    // being asked to do rather than to the size of the table.
    double    auditCache = 0.0, auditTrace = 0.0, auditTrace2 = 0.0;   // -radcache-audit totals
    long long auditN = 0;

    // Derived, rebuilt by prepare(): 1/cellSize per clipmap level and 1/baseDist^2. The
    // lookup is on the hot path of every diffuse vertex, so the key derivation must not
    // contain a divide or a log -- see levelOf/cellKey below.
    double invCell[16] = {0};
    double cellSize[16] = {0};
    double invBaseDist2 = 1.0;
    double invLamSpan = 1.0 / (830.0 - 360.0);

    // Call after changing baseCell / baseDist / maxLevel (renderBackward does, every pass).
    void prepare() {
        if (maxLevel > 15) maxLevel = 15;
        for (int l = 0; l <= maxLevel; ++l) {
            cellSize[l] = baseCell * (double)(1 << l);
            invCell[l]  = 1.0 / cellSize[l];
        }
        invBaseDist2 = 1.0 / (baseDist * baseDist);
        invLamSpan   = 1.0 / (lambdaHi - lambdaLo);
        if (minSamples < 1) minSamples = 1;
        if (maxSamples < minSamples) maxSamples = minSamples;
        if (maxSamples > 60000) maxSamples = 60000;   // cell.cnt is uint16
    }

    // Identity hash of every setting a SAVED cell's contents depend on, so a checkpointed
    // table is only ever reloaded into a render that would have produced the same cells.
    // Split into two groups, because getting the split wrong in either direction is a real
    // failure:
    //
    //   * KEY-DEFINING (baseCell, baseDist, maxLevel, camera): change any of these and the
    //     same surface point hashes to a different cell. Reloading across such a change does
    //     not merely misplace cells, it puts a value measured over one volume of space under
    //     the key of another -- silent, systematic, and invisible in the output.
    //   * VALUE-DEFINING (lambda range, minSamples/maxSamples/tol): these decide what a bin
    //     MEANS and when a cell is allowed to answer. A cell that passed a loose `tol` is
    //     frozen at `ok`, so reloading it into a run with a tighter tol would let the old
    //     tolerance quietly govern the new render.
    //
    // Deliberately NOT included: `rays`, `budget`, `warm`, `jitter`, `minBounce`, `maxUnused`
    // and the table CAPACITY. Those change how fast the table is fed, how it is read, or
    // where a cell lives -- none of which makes an already-stored value wrong -- so resuming
    // with a bigger update budget or a larger table is allowed, and is a thing one actually
    // wants to do. (A changed capacity is handled by re-inserting through loadCell(), which
    // rehomes every cell by key.) `kRadCacheBins` and the cell's own size are folded in so a
    // rebuild that changes the record layout rejects the old file instead of misreading it.
    uint64_t configGuard() const {
        uint64_t h = 14695981039346656037ULL;                 // FNV-1a offset basis
        auto mix64 = [&](uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
        auto mixd  = [&](double d) { uint64_t b; std::memcpy(&b, &d, 8); mix64(b); };
        mixd(baseCell); mixd(baseDist); mix64((uint64_t)(unsigned)maxLevel);
        mixd(camera.x); mixd(camera.y); mixd(camera.z);
        mixd(lambdaLo); mixd(lambdaHi);
        mix64((uint64_t)(unsigned)minSamples); mix64((uint64_t)(unsigned)maxSamples);
        mixd(tol);
        mix64((uint64_t)kRadCacheBins);
        mix64((uint64_t)sizeof(RadCacheCell));
        return h;
    }

    void init(size_t capacityPow2) {
        size_t cap = 1;
        while (cap < capacityPow2) cap <<= 1;
        key.assign(cap, kEmpty);
        cell.assign(cap, RadCacheCell{});
        scratch.assign(cap, RadCacheScratch{});
        live.clear();
        work.clear();
        cursor = 0;
        mask = cap - 1;
        used = 0;
        pass = 0;
        nSample = nEvicted = nTerm = nMiss = nUpdated = 0;
        nOk = nDead = chunkConsults = chunkTerm = 0;
        nCorrected = nRejected = 0;
        prepare();
    }

    bool ready() const { return !key.empty(); }

    // ---- keying ---------------------------------------------------------------------

    // Clipmap level for a point: cells double every doubling of distance from the camera, so
    // a cell subtends a roughly constant solid angle at the eye. Computed from the SQUARED
    // distance via ilogb (which just reads the double's exponent field) rather than
    // sqrt + log2 -- floor(log2(d/D)) == floor(ilogb(d^2/D^2)/2) for positive arguments, and
    // this runs at every diffuse vertex of every path.
    int levelOf(const Vec3& p) const {
        const Vec3 v = p - camera;
        const double d2 = (dot(v, v)) * invBaseDist2;
        if (!(d2 > 1.0)) return 0;
        int l = std::ilogb(d2) >> 1;
        return l < 0 ? 0 : (l > maxLevel ? maxLevel : l);
    }

    double cellSizeAt(int level) const { return baseCell * (double)(1 << level); }

    // Direction bucket for the cell key: which cube face the normal points at, subdivided
    // 3x3 within that face. 54 buckets, which is exactly the six spare bits left in the key.
    //
    // LumenRef stops at the six faces. Six is too coarse HERE, for a reason specific to this
    // design: a face bucket accepts a full 90-degree cone, so a sphere, a fillet or a piece
    // of foliage inside one cell folds normals up to 90 degrees apart into a single average,
    // and irradiance across 90 degrees of normal is not remotely constant. That is a
    // SYSTEMATIC error -- every path reading the cell gets the same wrong average, so it
    // survives any number of samples per pixel -- and it does not shrink when the confidence
    // gate tightens, because the cell is confidently reporting the average of the wrong set.
    //
    // Splitting each face 3x3 costs far less than the obvious alternative of halving the cell
    // edge: a FLAT surface still lands in exactly one bucket (its normal is at the face
    // centre), so walls, floors and ground planes pay nothing at all, while curved and
    // scattering geometry -- the only geometry that had the error -- pays for the extra
    // buckets it actually needs. Halving the cell edge, by contrast, multiplies cells
    // everywhere, including on the flat surfaces that were already exact.
    //
    // The 3x3 split (rather than 2x2) exists to keep the face CENTRE in one bucket. A 2x2
    // split cuts at u == 0, which is precisely where an axis-aligned normal sits, so the
    // sign of a component that ought to be zero -- and in practice is +-1e-17 of rounding
    // from a transform or an interpolated vertex normal -- would decide the bucket. A flat
    // floor would then shatter into four buckets along invisible seams and quadruple its
    // cell count for nothing. With cuts at +-kNormalSplit the exact-axis normal is a comfortable
    // distance inside the centre bucket and rounding cannot move it.
    static constexpr double kNormalSplit = 0.4;
    static int normalBucket(const Vec3& n) {
        const double ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
        int face; double u, v, w;
        if (ax >= ay && ax >= az) { face = n.x >= 0.0 ? 0 : 1; u = n.y; v = n.z; w = ax; }
        else if (ay >= az)        { face = n.y >= 0.0 ? 2 : 3; u = n.z; v = n.x; w = ay; }
        else                      { face = n.z >= 0.0 ? 4 : 5; u = n.x; v = n.y; w = az; }
        const double t  = kNormalSplit * w;              // scale-free: compares u/w to the split
        const int    qu = u < -t ? 0 : (u > t ? 2 : 1);
        const int    qv = v < -t ? 0 : (v > t ? 2 : 1);
        return face * 9 + qu * 3 + qv;
    }

    // Bin index for a wavelength, clamped to the table's range.
    int binOf(double lambda) const {
        const double t = (lambda - lambdaLo) * invLamSpan;
        int b = (int)(t * kRadCacheBins);
        return b < 0 ? 0 : (b >= kRadCacheBins ? kRadCacheBins - 1 : b);
    }

    // Representative wavelength of a bin: `u` in [0,1) picks a point inside it. The update
    // pass uses this to stratify a round's wavelengths one per bin.
    double lambdaOfBin(int b, double u) const {
        const double w = (lambdaHi - lambdaLo) / (double)kRadCacheBins;
        return lambdaLo + ((double)b + u) * w;
    }

    // Which bin ray `k` of a cell's round `r` samples. `r` is the CELL's own update-round
    // counter, never a global one: a global counter advances by however many rounds a chunk
    // happened to run, and with rays < kBins that stride aliases against the bin count and
    // starves the bins it never lands on (measured: rays=4 left 8 of 16 bins permanently
    // empty, so no cell ever became readable).
    int scheduleBin(uint32_t r, int k) const {
        const uint64_t i = (uint64_t)r * (uint64_t)(rays > 0 ? rays : 1) + (uint64_t)k;
        return (int)(i % (uint64_t)kRadCacheBins);
    }

    // Pack (level, ix, iy, iz, normal bucket) into one 64-bit key, then mix it. The pack is
    // lossy above +-2^17 cells from the origin on a level, which at level 0's default 5 cm is
    // +-6.5 km — past that two far-apart cells can collide, which costs a little smearing and
    // nothing else. Kept explicit rather than hashing the doubles so an identical point
    // always produces an identical key across CPU and (later) GPU.
    uint64_t cellKey(const Vec3& p, const Vec3& n, int level) const {
        const double ics = invCell[level];
        const int64_t ix = (int64_t)std::floor(p.x * ics);
        const int64_t iy = (int64_t)std::floor(p.y * ics);
        const int64_t iz = (int64_t)std::floor(p.z * ics);
        const uint64_t ux = (uint64_t)(ix & 0x3FFFF);   // 18 bits each
        const uint64_t uy = (uint64_t)(iy & 0x3FFFF);
        const uint64_t uz = (uint64_t)(iz & 0x3FFFF);
        uint64_t k = ux | (uy << 18) | (uz << 36)
                   | ((uint64_t)level << 54) | ((uint64_t)normalBucket(n) << 58);
        // Never produce kEmpty for a real cell.
        return k == kEmpty ? (k - 1) : k;
    }

    uint64_t cellKey(const Vec3& p, const Vec3& n) const { return cellKey(p, n, levelOf(p)); }

    // splitmix64 finalizer — cheap, well-distributed, and identical on any device.
    static uint64_t mix(uint64_t z) {
        z += 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // ---- lookup (hot path; read-only, called from every terminating vertex) -----------

    // Find an existing cell. Returns -1 on miss. Linear probing with a bounded scan: giving
    // up after kProbe slots costs an occasional duplicate cell, never a hang.
    static constexpr int kProbe = 8;

    int findSlot(uint64_t k) const { return findSlotMixed((size_t)mix(k), k); }

    // Probe with the mix already in hand. The reader's call site computes mix(k) anyway for
    // the mark filter's slot, so the hot path hands it in instead of paying splitmix twice.
    int findSlotMixed(size_t h, uint64_t k) const {
        size_t i = h & mask;
        for (int probe = 0; probe < kProbe; ++probe) {
            const uint64_t slot = key[i];
            if (slot == kEmpty) return -1;
            if (slot == k) return (int)i;
            i = (i + 1) & mask;
        }
        return -1;
    }

    const RadCacheCell* find(const Vec3& p, const Vec3& n) const {
        if (key.empty()) return nullptr;
        const int s = findSlot(cellKey(p, n));
        return s < 0 ? nullptr : &cell[s];
    }

    // The value a terminating path multiplies by rho: E/pi in this cell for `lambda`.
    // Returns false when the cell is absent or has not passed its confidence gate, in which
    // case the caller MUST keep tracing. That fallback is FREE in this design (the update
    // pass, not the reader, feeds the table), so it is the honest default rather than a
    // compromise: an unresolved corner renders exactly, it does not answer with noise.
    bool lookup(const Vec3& p, const Vec3& n, double lambda, double& out) const {
        const RadCacheCell* c = find(p, n);
        if (!c || !c->ok) return false;
        out = (double)c->mean[binOf(lambda)];
        return true;
    }

    // Bundle lookup: ONE hash probe for all `n` wavelengths of a hero bundle. Worth having as
    // its own entry point: the naive per-lambda loop repeats the key derivation and, far
    // worse, a random-access probe into a table far larger than L2, four times for one answer.
    //
    // The gate is on the CELL, i.e. every bin, not just the bins this bundle happens to read.
    // Answering some wavelengths of a bundle and not others would let part of a spectral path
    // terminate and part continue, which tints the pixel rather than merely adding noise -- a
    // colour error no amount of spp cleans up.
    //
    // `jitter` (in cell widths, default 0) dithers the lookup POSITION before keying, which
    // turns the cell grid's hard boundaries into noise the sampler averages away instead of a
    // visible discontinuity. LumenRef carries the identical hook. It is off by default because
    // dithering also widens the set of cells a pixel reads from, and the boundaries are
    // usually invisible at the smoothness indirect irradiance actually has.
    // `outCorr` and `outKey` serve the verification pass. `out[]` is deliberately the RAW
    // stored radiance, NOT premultiplied by the correction: a validation record has to be an
    // absolute measurement of what the cell's own samples say, or the ratio it feeds back
    // would be relative to a correction that is itself being updated, and the loop would
    // chase its own tail instead of converging. The reader applies `outCorr` when it uses
    // the value and records the raw product when it validates.
    bool lookupBundle(const Vec3& p, const Vec3& n, const double* lambda, int nLam,
                      double* out, double* outCorr = nullptr, uint64_t* outKey = nullptr,
                      double j0 = 0.0, double j1 = 0.0, double j2 = 0.0) const {
        if (key.empty()) return false;
        Vec3 q = p;
        if (jitter > 0.0) {
            const double h = jitter * cellSize[levelOf(p)];
            q = Vec3{p.x + (j0 - 0.5) * h, p.y + (j1 - 0.5) * h, p.z + (j2 - 0.5) * h};
        }
        const uint64_t k = cellKey(q, n);
        return lookupBundleAt(k, mix(k), lambda, nLam, out, outCorr, outKey);
    }

    // The no-jitter fast path: key and mix precomputed by the caller, which already derived
    // BOTH for the mark filter (backward.h marks first, then reads). With jitter off the
    // lookup key is identical to the mark key, so recomputing levelOf + normalBucket + three
    // floors + splitmix here was pure duplicate work. Same key, same probe, same answer —
    // bit-identical to routing through lookupBundle.
    bool lookupBundleAt(uint64_t k, uint64_t kmix, const double* lambda, int nLam,
                        double* out, double* outCorr = nullptr,
                        uint64_t* outKey = nullptr) const {
        if (key.empty()) return false;
        const int s = findSlotMixed((size_t)kmix, k);
        if (s < 0) return false;
        const RadCacheCell& c = cell[s];
        if (!c.ok) return false;
        for (int i = 0; i < nLam; ++i) out[i] = (double)c.mean[binOf(lambda[i])];
        if (outCorr) *outCorr = (double)c.corr;
        if (outKey)  *outKey  = k;
        return true;
    }

    // ---- merge / update / apply (between chunks; merge and apply are single-threaded) ----

    // Insert a marked cell, or refresh an existing one's mark stamp. Returns the slot, or -1
    // when the probe window is full of live, recently-used cells.
    //
    // A slot whose cell has not been marked for `maxUnused` chunks is RECYCLED rather than
    // counted as a collision. Without that a long flyby would silently fill the table with
    // geometry that left the frame chunks ago and then start dropping the cells actually on
    // screen -- the failure looks like "table full" at a fraction of nominal occupancy,
    // because linear probing needs an empty slot in the window, not merely a low load factor.
    int insertMark(const RadCacheMark& m) {
        if (key.empty()) return -1;
        size_t i = (size_t)mix(m.key) & mask;
        int recycle = -1;
        for (int probe = 0; probe < kProbe; ++probe) {
            const uint64_t k = key[i];
            if (k == m.key) { cell[i].lastMark = pass; return (int)i; }
            if (k == kEmpty) { birth(i, m); ++used; return (int)i; }
            if (recycle < 0 && (uint32_t)(pass - cell[i].lastMark) > (uint32_t)maxUnused)
                recycle = (int)i;
            i = (i + 1) & mask;
        }
        if (recycle >= 0) {
            birth((size_t)recycle, m);
            return recycle;
        }
        ++nEvicted;
        return -1;
    }

    void birth(size_t i, const RadCacheMark& m) {
        key[i] = m.key;
        RadCacheCell& c = cell[i];
        c = RadCacheCell{};
        c.px = m.p.x; c.py = m.p.y; c.pz = m.p.z;
        c.nx = (float)m.n.x; c.ny = (float)m.n.y; c.nz = (float)m.n.z;
        c.lastMark = pass;
        scratch[i] = RadCacheScratch{};
    }

    // Re-insert a cell deserialized from a checkpoint (see the .ftbuf radiance-cache section
    // in main.cpp). Probes exactly as insertMark does, which is what lets a resumed render use
    // a DIFFERENT table capacity: every cell is rehomed by its key rather than by its old slot
    // index, so growing `-radcache-cells` between runs keeps the table instead of scrambling
    // it. A probe window with no free slot simply drops the cell -- it is a cache entry, and a
    // dropped one costs one miss, which the update pass fills back in.
    //
    // `lastMark` is loaded verbatim, and the caller restores `pass` alongside, so cell AGES
    // survive the round trip: a resumed flyby recycles the cells that had gone stale before
    // the interruption rather than treating the whole table as freshly touched.
    bool loadCell(uint64_t k, const RadCacheCell& c) {
        if (key.empty() || k == kEmpty) return false;
        size_t i = (size_t)mix(k) & mask;
        for (int probe = 0; probe < kProbe; ++probe) {
            if (key[i] == kEmpty || key[i] == k) {
                if (key[i] == kEmpty) ++used;
                key[i]     = k;
                cell[i]    = c;
                scratch[i] = RadCacheScratch{};   // a round's scratch is never checkpointed
                return true;
            }
            i = (i + 1) & mask;
        }
        ++nEvicted;
        return false;
    }

    // Add one radiance sample to a cell's scratch, addressed by slot. (The update pass owns
    // disjoint slot ranges, which is what makes this need no atomics.)
    void addSampleAt(size_t slot, int bin, double value) {
        RadCacheScratch& s = scratch[slot];
        s.sum[bin]   += (float)value;
        s.sumsq[bin] += (float)(value * value);
        s.mx[bin]     = std::max(s.mx[bin], (float)value);
        s.cnt[bin]   += 1;
    }

    // Fold every thread's marks into the table and rebuild the work lists. Single-threaded and
    // after the join, which is what makes the table's state a pure function of the samples
    // rendered so far rather than of the thread interleaving.
    void mergeMarks(std::vector<RadCacheBank>& banks) {
        ++pass;
        chunkConsults = 0;
        chunkTerm     = 0;
        for (RadCacheBank& b : banks) {
            for (const RadCacheMark& m : b.marks) insertMark(m);
            nTerm += b.nTerm;
            nMiss += b.nMiss;
            chunkTerm     += b.nTerm;
            chunkConsults += b.nTerm + b.nMiss;
            auditCache  += b.auditCache;
            auditTrace  += b.auditTrace;
            auditTrace2 += b.auditTrace2;
            auditN      += b.auditN;
            b.marks.clear();
            b.nTerm = b.nMiss = 0;
            b.auditCache = b.auditTrace = b.auditTrace2 = 0.0; b.auditN = 0;
        }
        // Verification records fold in AFTER the marks, so a record for a cell that was born
        // (or reborn) this merge still lands on the right slot. Records for a key that is no
        // longer in the table are simply dropped -- the cell they described is gone.
        for (RadCacheBank& b : banks) {
            for (const RadCacheVal& v : b.vals) {
                const int s = findSlot(v.key);
                if (s < 0) continue;
                RadCacheCell& c = cell[s];
                c.vOffer += v.offer;
                c.vTail  += v.tail;
                c.vTail2 += v.tail * v.tail;
                ++c.vN;
            }
            b.vals.clear();
        }
        applyValidation();
        rebuildWork();
    }

    // Turn each cell's accumulated verification record into a correction, a retirement, or
    // nothing. See "VERIFYING THE CACHE AGAINST THE READERS THEMSELVES" at the top of the
    // file for why this is the gate that actually bounds the error.
    void applyValidation() {
        nCorrected = 0;
        for (size_t i = 0; i <= mask; ++i) {
            if (key[i] == kEmpty) continue;
            RadCacheCell& c = cell[i];
            if (c.vN < (uint32_t)kMinValid || !(c.vOffer > 0.0)) {
                if (c.corr != 1.0f) ++nCorrected;
                continue;
            }
            const double N  = (double)c.vN;
            const double r  = c.vTail / c.vOffer;
            // Relative standard error of the numerator. The denominator is the sum of values
            // the cache offered, which is a smooth low-variance quantity (it varies only with
            // the readers' throughput), so the ratio's error is the tail's error.
            const double mT   = c.vTail / N;
            const double varT = std::max(0.0, c.vTail2 / N - mT * mT);
            const double seR  = (mT != 0.0) ? std::sqrt(varT / N) / std::fabs(mT) : 1e30;
            if (seR <= 0.5 * tol) {
                c.corr = (float)std::clamp(r, kCorrLo, kCorrHi);
                if (c.corr != 1.0f) ++nCorrected;
            } else if (std::fabs(r - 1.0) > tol + 2.0 * seR) {
                // Provably wrong and not pinnable: stop answering, stop spending on it.
                if (!c.dead) ++nRejected;
                c.dead = 1; c.ok = 0;
            }
        }
    }

    void rebuildWork() {
        live.clear();
        work.clear();
        refine.clear();
        nDead = 0;
        for (size_t i = 0; i <= mask; ++i) {
            if (key[i] == kEmpty) continue;
            if ((uint32_t)(pass - cell[i].lastMark) > (uint32_t)maxUnused) continue;
            live.push_back((uint32_t)i);
            if (cell[i].dead)   { ++nDead; continue; }
            if (cell[i].ok)       refine.push_back((uint32_t)i);
            else                  work.push_back((uint32_t)i);
        }
        nOk = (long long)refine.size();
        // The cursors index lists that have just been rebuilt, so they are only ever fairness
        // hints. Keep them in range and let them keep rotating rather than resetting to 0,
        // which would re-serve the same head of the list every chunk and starve the tail.
        cursor  = work.empty()   ? 0 : cursor  % work.size();
        rcursor = refine.empty() ? 0 : rcursor % refine.size();
    }

    // How many update SAMPLES this chunk may spend. See the budget-governor note above.
    long long chunkBudget() const {
        const double mult = (pass <= 1) ? (double)std::max(1, warm) : 1.0;
        double n = budget * mult * (double)chunkConsults;
        if (n < 0.0) n = 0.0;
        return (long long)n;
    }

    // Take the next `nCells` cells to serve. Tier 1 (`work` — cells that cannot answer yet)
    // is drained first, because a cell that cannot answer is saving nothing; only once it is
    // empty does the budget flow into tier 2 (`refine` — cells that already answer).
    //
    // TIER 2 IS WHY THE BIAS FADES. A confident cell is not frozen: it keeps taking samples
    // whenever there is budget left over, so its standard error keeps falling as 1/sqrt(n)
    // while the image's own noise falls as 1/sqrt(spp). Both are fed by the same growing
    // stream of consults, so the cache does not become the error floor of a long render --
    // which a freeze-on-confidence table would, and visibly, since a cell's error is frozen
    // speckle that no amount of spp averages away.
    bool takeWork(std::vector<uint32_t>& out, size_t nCells) {
        out.clear();
        if (nCells == 0) return false;
        if (!work.empty()) {
            const size_t n = std::min(nCells, work.size());
            out.reserve(n);
            for (size_t i = 0; i < n; ++i) out.push_back(work[(cursor + i) % work.size()]);
            cursor = (cursor + n) % work.size();
            return true;
        }
        if (!refine.empty()) {
            const size_t n = std::min(nCells, refine.size());
            out.reserve(n);
            for (size_t i = 0; i < n; ++i) out.push_back(refine[(rcursor + i) % refine.size()]);
            rcursor = (rcursor + n) % refine.size();
            return true;
        }
        return false;
    }

    // End of chunk: let the mark filters forget. Marks themselves are NOT dropped -- the
    // update rounds mark too, and those marks are how the cache grows into geometry no camera
    // path ever reached, so they ride to the next chunk's merge.
    void clearBanks(std::vector<RadCacheBank>& banks) {
        for (RadCacheBank& b : banks) b.resetFilter();
    }

    // Fold this round's scratch into the served cells and re-test their confidence gate.
    //
    // The gate: every bin needs `minSamples` samples and a measured standard error inside
    // `tol` of that bin's own mean (floored against the cell's peak -- see kFloorFrac). Once a
    // cell passes it is FROZEN and leaves the work list: its estimate is already a complete
    // unbiased multi-bounce answer, so more samples would only shave an error the reader has
    // agreed to tolerate.
    //
    // The retirement test is a PROJECTION, not an exhaustion: from the sample variance we know
    // the gate would need var/(tol*ref)^2 samples, so a cell that would need more than
    // `maxSamples` is written off after `minSamples` rather than after thousands. This is the
    // whole of the graceful degradation on high-variance geometry (fur, foliage, caustic
    // pools): those cells simply never answer and stop costing anything.
    // Returns how many cells LEFT the work list (passed the gate or were retired), so the
    // caller can skip the work-list rebuild -- a full table scan -- when nothing changed.
    size_t apply(const std::vector<uint32_t>& served) {
        size_t retired = 0;
        for (uint32_t slot : served) {
            RadCacheScratch& s = scratch[slot];
            RadCacheCell&    c = cell[slot];
            bool any = false;
            for (int b = 0; b < kRadCacheBins; ++b) {
                if (!s.cnt[b]) continue;
                any = true;
                const double n0 = (double)c.cnt[b];
                const double n1 = n0 + (double)s.cnt[b];
                c.mean[b]  = (float)(((double)c.mean[b] * n0 + (double)s.sum[b]) / n1);
                c.sumsq[b] = (float)((double)c.sumsq[b] + (double)s.sumsq[b]);
                c.mx[b]    = std::max(c.mx[b], s.mx[b]);
                c.cnt[b]   = (uint16_t)std::min(n1, 65535.0);
            }
            s = RadCacheScratch{};
            if (!any) continue;
            ++c.upRound;
            ++nUpdated;

            double peak = 0.0;
            for (int b = 0; b < kRadCacheBins; ++b) peak = std::max(peak, (double)c.mean[b]);
            const double floorV = kFloorFrac * peak;
            bool   pass_ = true, enough = true;
            double needMax = 0.0;
            for (int b = 0; b < kRadCacheBins; ++b) {
                const double n = (double)c.cnt[b];
                if (n < (double)minSamples) { pass_ = false; enough = false; continue; }
                const double m   = (double)c.mean[b];
                const double var = std::max(0.0, (double)c.sumsq[b] / n - m * m);
                const double ref = std::max(std::fabs(m), floorV);
                if (!(ref > 0.0)) continue;              // a genuinely black bin is exact
                // Conservative standard error. A variance estimated from a handful of samples
                // is itself mostly noise, and the failure mode is not symmetric: a cell whose
                // few samples all MISSED the bright direction under-reports its variance,
                // declares confidence it has not earned, and freezes a dark value into the
                // image. The (1 + 2/sqrt(n)) inflation is a cheap one-sided allowance for
                // that -- 1.7x at 8 samples, 1.2x at 100, vanishing thereafter.
                const double se  = std::sqrt(var / n) * (1.0 + 2.0 / std::sqrt(n));
                if (se > tol * ref) pass_ = false;
                // HEAVY-TAIL GUARD. The standard error above is only meaningful for a
                // distribution the samples have actually explored, and indirect radiance
                // often is not one: a cell that sees a caustic, a small bright specular
                // highlight or a sliver of sky through a gap draws its energy from events
                // rarer than 1/n, so a run of n samples that all missed them reports a small
                // mean AND a small variance, and the gate above waves it through with
                // confidence it has not earned. That failure is the dangerous direction --
                // it freezes a too-dark value into every path that reads the cell, which no
                // number of samples per pixel can undo.
                //
                // The test that catches it needs no extra rays: if the single largest sample
                // in a bin is a large share of that bin's whole sum, the mean is being
                // carried by a handful of events, so the tail is real, under-sampled, and
                // the mean is not to be trusted. Measured on cornell.ftsl (a dispersive
                // glass sphere, so genuinely caustic): -14.7% systematic error per read
                // without this, and the same scene made all-diffuse shows -0.3%, which is
                // what identified the tail as the culprit rather than the estimator.
                //
                // Cells that fail only this test are NOT retired -- they keep taking samples
                // and may pass later, once enough of the tail has been seen for the mean to
                // be real. Retirement is decided by the variance projection below, which for
                // a genuinely heavy-tailed cell will call it hopeless soon enough anyway.
                if ((double)c.mx[b] > kDomFrac * m * n) pass_ = false;
                // Samples this bin would need for se <= tol*ref, from the variance we have.
                const double need = var / (tol * ref * tol * ref);
                needMax = std::max(needMax, need);
            }
            // A cell may fall BACK out of confidence: more samples can reveal a variance the
            // first few hid, and the honest response is to stop answering and take more.
            const uint8_t wasOk = c.ok, wasDead = c.dead;
            c.ok = pass_ ? 1 : 0;
            if (!pass_ && enough && needMax > (double)maxSamples) c.dead = 1;
            if (c.ok != wasOk || c.dead != wasDead) ++retired;
        }
        return retired;
    }

    double occupancy() const { return key.empty() ? 0.0 : (double)used / (double)key.size(); }
};
