#pragma once
// photonmap_io.h — persistence for the mode-M photon map AND its photon-beam volume cache.
//
// Why this file exists at all. Mode M's whole argument is "trace once, gather many": the
// deposited map is view-independent, so it is the expensive part of the render and the part
// worth keeping. `-savemap` / `-loadmap` are that promise made durable — re-gather a saved
// map from new angles, at a new radius, with a new spp, without re-tracing a photon.
//
// Two defects in the previous arrangement are the reason it moved here:
//
//   1. It lived inside `render_cuda.cu` as file-static helpers, so `-savemap` / `-loadmap`
//      were GPU-only *by accident of placement* rather than by design — none of the code is
//      CUDA, it is ordinary host serialisation. The CPU mode-M path simply had no access to
//      it and ignored both flags in silence.
//   2. It stored SURFACE photons only. Once `-beams` gave mode M a volume (photonbeams.h),
//      that omission became a hole in the feature's headline claim: you could bank the
//      surface half of a trace and not the volume half. And because `-beams` forces the CPU
//      path, the two flags were silently mutually exclusive — `-savemap -beams` exited 0 and
//      wrote nothing at all.
//
// Format. `FTPMP09\n` = header, surface block, beam block, CAUSTIC block; its beam record is the
// live PhotonBeam with chord PROVENANCE valid (srcEm / surf, 0.312.0). `FTPMP08\n` = the same
// width with those two fields unset (the loader resets them to "unknown"). `FTPMP07\n`,
// `FTPMP06\n`, `FTPMP05\n` and `FTPMP04\n` have the same block layout but successively NARROWER
// beam records (v7 lacks the bundle's per-member weights, v6 lacks those AND the gather-time
// spectral fold, v5 lacks all that AND the deposit-time achromatic fold, v4 lacks all four —
// see PhotonBeamV7 / PhotonBeamV6 / PhotonBeamV5 / PhotonBeamV4 below); `FTPMP03\n` (header +
// surface + beam) and `FTPMP02\n` (surface only) are older still.
// Every one of them loads: a v7 file's beams widen to wS == 1, which is exactly the
// equal-weight bundle they were saved as; a v6 file's widen to emIdx == -1, which is exactly the
// deposit-time-folded beam they were saved as; a v5 file's widen to achro == 0, which is
// exactly the per-wavelength beam they were saved as; a v4 file's additionally widen to
// nSec == 0, which is exactly the monochromatic beam they were saved as; a v3 file simply has
// every deposit in the global map, which is the pre-0.199.7 single-map behaviour; a v2 file
// additionally reports no beams. So no existing cache is invalidated by any of these changes —
// each just re-gathers as the render it was saved from.
//
// NOTE ON THE wS WIDENING SPECIFICALLY: unlike the achromatic one below, `wS = 1` is exact and
// not merely safe. A v7 bundle could only exist on a path along which every member's spectral
// throughput was IDENTICAL — that was the condition the tracer required before 0.257.0 — so its
// members' relative weights were all 1 by construction, and writing 1 reproduces the file's own
// estimator bit for bit rather than approximating it.
//
// NOTE ON THE ACHROMATIC WIDENING SPECIFICALLY: `achro == 0` is not merely a safe default, it is
// the CORRECT one. A cached beam has no record of the emitter it came from, so the fold constant
// (Emitter::cieMean) cannot be recovered after the fact; falling back to CIE(lambda) reproduces
// the exact estimator the file was traced with. A loaded v5 cache therefore renders as noisy as
// it always did, and re-tracing is what buys the fold. That is a deliberate accuracy-preserving
// choice, not an oversight — inventing a fold constant for a beam whose provenance is unknown
// would silently change the image a cache is supposed to reproduce.
//
// What is NOT stored, deliberately: every derived structure. The photon grid is rebuilt by
// `PhotonMap::build(radius)` and the beam BVH by `BeamMap::buildAuto(K)`, so ONE file serves
// any `-pmradius` / `-beamk` / `-beamradius` you later ask for. For beams this is load-bearing
// rather than merely tidy: `buildAuto` splits long beams at a length derived from the radius
// it just chose, so persisting post-split sub-beams would freeze the radius into the file and
// silently ignore a later `-beamk`. The beams written are therefore the RAW crossings, exactly
// as `tracePhotonPass` deposited them (after `-beamcount` thinning, which is a property of the
// trace, not of the gather).
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "photonmap.h"
#include "photonbeams.h"
#include "scene.h"

// The FTPMP04 beam record, frozen. The beam block is a raw fwrite of the live `PhotonBeam`,
// so widening that struct for the spectral bundle (0.202.0) changed its size and would have
// made every existing .pmap unreadable — silently and catastrophically, since a size mismatch
// on a raw array read is not a parse error, it is garbage beams. Keeping the old layout here
// verbatim, and reading a v4 file through IT, is what makes the widening free: the fields are
// copied across by name and the bundle is set empty, which is precisely what a v4 beam meant.
//
// This struct must NEVER be edited again. It is not "the beam record"; it is "what a v4 file
// contains", and that is now a historical fact. The next widening adds a PhotonBeamV5.
struct PhotonBeamV4 {
    Vec3  o;
    Vec3  d;
    float s0, len, power, lambda, absorb;
    int   med;
};

// The FTPMP05 beam record, frozen — the same story one version on. Widening `PhotonBeam` for
// the achromatic-path fold (0.210.0: `float cieA[3]; unsigned char achro;`) changed its size
// again, so a v5 file read as an array of the live struct would be garbage beams, not a parse
// error. Reading a v5 file through THIS layout and setting `achro = 0` is exact: it is precisely
// what a v5 beam meant.
//
// Same rule as above: never edit this. The next widening adds a PhotonBeamV6, and the widening
// loop below generalises to it by adding one case, not by touching a frozen struct.
// The `3` is written as a LITERAL, not as kBeamSecMax, for the same reason the struct is frozen:
// what a v5 file contains is history, and history does not track a constant someone may raise
// later. The static_assert below is the alarm for exactly that — if `-beamspec`'s ceiling ever
// moves, this stops compiling and whoever moves it must add a PhotonBeamV6 rather than silently
// reinterpret every v5 cache on disk at the wrong stride.
struct PhotonBeamV5 {
    Vec3  o;
    Vec3  d;
    float s0, len, power, lambda, absorb;
    int   med;
    float lamS[3];
    int   nSec;
};
// The FTPMP06 beam record, frozen — the same story one version on again. Widening
// `PhotonBeam` for the GATHER-time spectral fold (0.256.0: `short emIdx;`) changed its size a
// third time. A v6 file read at the live stride would be garbage beams, not a parse error.
// Reading it through THIS layout and setting `emIdx = -1` is exact: a v6 file cannot contain
// achro == 2, and -1 is what "this file predates the gather-time fold" means.
//
// Same rule as the two above: never edit this. The next widening adds a PhotonBeamV7.
struct PhotonBeamV6 {
    Vec3  o;
    Vec3  d;
    float s0, len, power, lambda, absorb;
    int   med;
    float lamS[3];
    int   nSec;
    float cieA[3];
    unsigned char achro;
};
// The FTPMP07 beam record, frozen — the same story one version on again. Widening `PhotonBeam`
// for the spectral bundle's PER-MEMBER WEIGHTS (0.257.0: `float wS[kBeamSecMax];`) changed its
// size a fourth time. A v7 file read at the live stride would be garbage beams, not a parse
// error. Reading it through THIS layout and setting every `wS` to 1 is exact, not approximate:
// see the note at the top of this file.
//
// Same rule as the three above: never edit this. The next widening adds a PhotonBeamV8.
struct PhotonBeamV7 {
    Vec3  o;
    Vec3  d;
    float s0, len, power, lambda, absorb;
    int   med;
    float lamS[3];
    int   nSec;
    float cieA[3];
    unsigned char achro;
    short emIdx;
};
static_assert(kBeamSecMax == 3,
              "kBeamSecMax changed: PhotonBeam's on-disk width moved, so FTPMP08 no longer "
              "describes the live record. Freeze the old layout as PhotonBeamV8 and bump the "
              "magic to FTPMP10 — do NOT edit PhotonBeamV5, V6 or V7.");
// FTPMP09 (0.312.0) put chord provenance INTO the tail padding, so it reads FTPMP08 files with
// the same stride and merely resets those two fields. That only holds while the record is
// exactly 120 bytes; a wider record needs a frozen PhotonBeamV9 and an FTPMP10 magic.
static_assert(sizeof(PhotonBeam) == 120,
              "PhotonBeam's on-disk width moved: FTPMP09 and the FTPMP08 files it still reads "
              "assume a 120-byte record with srcEm/surf in the tail padding.");

// Scene-identity guard: refuses to blend a stale cache into a different scene. Cheap and
// coarse on purpose — it catches "wrong file", not "same scene, one triangle moved".
inline uint64_t photonMapGuard(const Scene& scene, bool diffraction) {
    uint64_t h = 14695981039346656037ULL;                 // FNV-1a offset basis
    auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
    mix((uint64_t)scene.tris.size());
    mix((uint64_t)scene.spheres.size());
    mix((uint64_t)scene.emitters.size());
    uint64_t tp; std::memcpy(&tp, &scene.totalPower, sizeof tp); mix(tp);
    mix(diffraction ? 0x9E37ULL : 0x1234ULL);
    return h;
}

// `bm` may be null (no -beams) or empty (no photon crossed a medium); either writes a beam
// block with count 0, so a reader can always tell "this trace had no volume" apart from
// "this file predates volume support".
// `pmCaustic` may be null (the caustic split was off) — a count-0 caustic block is written,
// which reloads as "no caustic photons", the same thing a v3 file means.
inline bool savePhotonMap(const char* path, const PhotonMap& pm,
                          const EnergyReport& e, uint64_t guard,
                          const BeamMap* bm = nullptr,
                          const PhotonMap* pmCaustic = nullptr) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "[savemap] cannot open %s for writing\n", path); return false; }
    const char magic[8] = {'F','T','P','M','P','0','9','\n'};
    long long nPh = (long long)pm.photons.size();
    double en[5] = {e.emitted, e.absorbed, e.sensor, e.escaped, e.residual};
    bool ok = true;
    ok = ok && std::fwrite(magic, 1, 8, f) == 8;
    ok = ok && std::fwrite(&guard, sizeof guard, 1, f) == 1;
    ok = ok && std::fwrite(&pm.nEmitted, sizeof pm.nEmitted, 1, f) == 1;
    ok = ok && std::fwrite(en, sizeof en, 1, f) == 1;
    ok = ok && std::fwrite(&nPh, sizeof nPh, 1, f) == 1;
    if (ok && nPh > 0) {
        ok = std::fwrite(pm.pos.data(), sizeof(Vec3), (size_t)nPh, f) == (size_t)nPh;
        ok = ok && std::fwrite(pm.photons.data(), sizeof(Photon), (size_t)nPh, f) == (size_t)nPh;
    }
    // --- beam block ---------------------------------------------------------------------
    // nEmitted is stored separately from the surface map's because a beam map normalises by
    // the emitted count of ITS pass; they are equal today but the field is the estimator's,
    // not the file's, and tying them would be a landmine if the passes ever split.
    long long nBm  = (bm && !bm->empty()) ? (long long)bm->beams.size() : 0;
    long long bEm  = bm ? bm->nEmitted : 0;
    long long bDep = bm ? (long long)bm->nDeposited : 0;
    ok = ok && std::fwrite(&nBm,  sizeof nBm,  1, f) == 1;
    ok = ok && std::fwrite(&bEm,  sizeof bEm,  1, f) == 1;
    ok = ok && std::fwrite(&bDep, sizeof bDep, 1, f) == 1;
    if (ok && nBm > 0)
        ok = std::fwrite(bm->beams.data(), sizeof(PhotonBeam), (size_t)nBm, f) == (size_t)nBm;
    // --- caustic block (FTPMP04) ----------------------------------------------------------
    // Its own nEmitted for the same reason the beam block has one: the normalisation belongs
    // to the estimator, not to the file. Today it equals the surface map's (one pass fills
    // both), and a future dedicated caustic emission pass would make it differ — at which
    // point a file that had tied them would be silently wrong.
    long long nCa = (pmCaustic ? (long long)pmCaustic->photons.size() : 0);
    long long cEm = (pmCaustic ? pmCaustic->nEmitted : 0);
    ok = ok && std::fwrite(&nCa, sizeof nCa, 1, f) == 1;
    ok = ok && std::fwrite(&cEm, sizeof cEm, 1, f) == 1;
    if (ok && nCa > 0) {
        ok = std::fwrite(pmCaustic->pos.data(), sizeof(Vec3), (size_t)nCa, f) == (size_t)nCa;
        ok = ok && std::fwrite(pmCaustic->photons.data(), sizeof(Photon), (size_t)nCa, f)
                   == (size_t)nCa;
    }
    std::fclose(f);
    if (!ok) std::fprintf(stderr, "[savemap] write to %s failed\n", path);
    return ok;
}

// Returns false (leaving `pm` untouched) on any problem, always with a printed reason: a
// silently ignored cache is worse than a re-trace, because the render still produces an image
// and you cannot see which one you got.
//
// `bm` may be null when the caller did not ask for -beams; the beam block is then skipped.
// `beamsMissing` (optional out) reports "the caller wanted beams and this file has none", so
// the caller can say so rather than rendering a volumeless image that looks like a beams one.
//
// `pmCaustic` may be null (the caller does not want the split). A v2/v3 file, or a v4 one
// whose caustic block is empty, leaves it cleared — which gathers as the single-map render
// the file was saved from.
inline bool loadPhotonMap(const char* path, PhotonMap& pm,
                          EnergyReport& e, uint64_t guard,
                          BeamMap* bm = nullptr, bool* beamsMissing = nullptr,
                          PhotonMap* pmCaustic = nullptr) {
    if (beamsMissing) *beamsMissing = false;
    if (pmCaustic) { pmCaustic->photons.clear(); pmCaustic->pos.clear(); pmCaustic->nEmitted = 0; }
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "[loadmap] cannot open %s\n", path); return false; }
    char magic[8] = {0};
    long long nEmitted = 0, nPh = 0; double en[5] = {0,0,0,0,0}; uint64_t g = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8;
    const bool v9 = ok && std::memcmp(magic, "FTPMP09\n", 8) == 0;
    const bool v8 = v9 || (ok && std::memcmp(magic, "FTPMP08\n", 8) == 0);   // v9 = v8 + provenance
    const bool v7 = v8 || (ok && std::memcmp(magic, "FTPMP07\n", 8) == 0);   // v8 ⊃ v7 blocks
    const bool v6 = v7 || (ok && std::memcmp(magic, "FTPMP06\n", 8) == 0);   // v7 ⊃ v6 blocks
    const bool v5 = v6 || (ok && std::memcmp(magic, "FTPMP05\n", 8) == 0);   // v6 ⊃ v5 blocks
    const bool v4 = v5 || (ok && std::memcmp(magic, "FTPMP04\n", 8) == 0);   // v5 ⊃ v4 blocks
    const bool v3 = v4 || (ok && std::memcmp(magic, "FTPMP03\n", 8) == 0);   // v4 ⊃ v3 layout
    const bool v2 = ok && std::memcmp(magic, "FTPMP02\n", 8) == 0;
    // Only the BEAM RECORD differs between v4..v8; the block structure is identical from v4 on,
    // which is why the flags above nest and only the record WIDTH distinguishes them. One
    // number carries that, so the read and the skip-seek cannot disagree about the stride:
    //   v4 -> PhotonBeamV4 (no bundle, no fold), v5 -> PhotonBeamV5 (bundle, no fold),
    //   v6 -> PhotonBeamV6 (deposit-time fold), v7 -> PhotonBeamV7 (gather-time fold),
    //   v8 -> the live PhotonBeam (per-member bundle weights).
    const long long beamRec = v8 ? (long long)sizeof(PhotonBeam)
                                 : v7 ? (long long)sizeof(PhotonBeamV7)
                                      : v6 ? (long long)sizeof(PhotonBeamV6)
                                           : v5 ? (long long)sizeof(PhotonBeamV5)
                                                : (long long)sizeof(PhotonBeamV4);
    if (!v3 && !v2) {
        // Name the stale-version case explicitly: a user with a cache from before the
        // split layout should be told to re-deposit, not left guessing.
        if (ok && std::memcmp(magic, "FTPMP01\n", 8) == 0)
            std::fprintf(stderr, "[loadmap] %s is an old FTPMP01 map (pre split-layout); "
                                 "re-run with -savemap to rebuild it. Ignoring.\n", path);
        else
            std::fprintf(stderr, "[loadmap] %s is not a recognised photon-map file; ignoring\n", path);
        std::fclose(f); return false;
    }
    ok = ok && std::fread(&g, sizeof g, 1, f) == 1;
    ok = ok && std::fread(&nEmitted, sizeof nEmitted, 1, f) == 1;
    ok = ok && std::fread(en, sizeof en, 1, f) == 1;
    ok = ok && std::fread(&nPh, sizeof nPh, 1, f) == 1;
    if (!ok) { std::fprintf(stderr, "[loadmap] %s truncated header; ignoring\n", path); std::fclose(f); return false; }
    if (g != guard) {
        std::fprintf(stderr, "[loadmap] %s was built for a different scene; ignoring\n", path);
        std::fclose(f); return false;
    }
    if (nPh > 0) {
        ftalloc::resize(pm.pos, (size_t)nPh, "the photon map positions (-loadmap)",
                        "the photon count the map was saved with");
        ftalloc::resize(pm.photons, (size_t)nPh, "the photon map payloads (-loadmap)",
                        "the photon count the map was saved with");
        ok = std::fread(pm.pos.data(), sizeof(Vec3), (size_t)nPh, f) == (size_t)nPh;
        ok = ok && std::fread(pm.photons.data(), sizeof(Photon), (size_t)nPh, f) == (size_t)nPh;
    }
    if (!ok) {
        std::fprintf(stderr, "[loadmap] %s truncated photon data; ignoring\n", path);
        pm.photons.clear(); pm.pos.clear(); std::fclose(f); return false;
    }
    // --- beam block (FTPMP03 only) --------------------------------------------------------
    if (v3) {
        long long nBm = 0, bEm = 0, bDep = 0;
        bool bok = std::fread(&nBm,  sizeof nBm,  1, f) == 1
                && std::fread(&bEm,  sizeof bEm,  1, f) == 1
                && std::fread(&bDep, sizeof bDep, 1, f) == 1;
        if (!bok) {
            std::fprintf(stderr, "[loadmap] %s truncated beam header; ignoring\n", path);
            pm.photons.clear(); pm.pos.clear(); std::fclose(f); return false;
        }
        if (bm && nBm > 0) {
            ftalloc::resize(bm->beams, (size_t)nBm, "the photon-beam map (-loadmap)",
                            "the beam count the map was saved with");
            bool rok;
            if (v8) {
                rok = std::fread(bm->beams.data(), sizeof(PhotonBeam), (size_t)nBm, f) == (size_t)nBm;
                // An FTPMP08 file has the same width, but its tail bytes predate chord provenance
                // (0.312.0): say "unknown" rather than trust whatever the old writer left there.
                if (rok && !v9)
                    for (auto& b : bm->beams) { b.srcEm = -1; b.surf = 255; b.pad2 = 0; }
            } else {
                // Widen an older file in place. Read into the frozen old layout a chunk at a
                // time rather than allocating a second full array: the beam map is routinely
                // the largest thing in the process, and doubling it to load a file would turn
                // a working -loadmap into an OOM on exactly the caches big enough to be worth
                // saving. 64 K records is a few MB of scratch and one fread per 64 K beams.
                //
                // v4..v7 share this loop because widening is cumulative: a v4 record is a v5
                // record minus the bundle, a v6 record is a v7 one minus `emIdx`, and a v7
                // record is the live one minus `wS`. `common` fills the fields every generation
                // has, and each absent generation's fields get the value that MEANS "this file
                // predates it" (nSec = 0 for the bundle, achro = 0 for the deposit-time fold,
                // emIdx = -1 for the gather-time one, wS = 1 for the per-member weights).
                //
                // The two source vectors are read into by separate branches rather than aliasing
                // one buffer through both struct types: the layouts share a prefix, but punning
                // a PhotonBeamV5* to a PhotonBeamV4* is undefined behaviour, and an optimiser is
                // entitled to reorder around it. The duplication is three lines; the alternative
                // is a load that works until the day someone raises the optimisation level.
                std::vector<PhotonBeamV7> chunk7;
                std::vector<PhotonBeamV6> chunk6;
                std::vector<PhotonBeamV5> chunk5;
                std::vector<PhotonBeamV4> chunk4;
                auto common = [](PhotonBeam& b, const Vec3& o, const Vec3& d, float s0, float len,
                                 float power, float lambda, float absorb, int med) {
                    b.o = o; b.d = d; b.s0 = s0; b.len = len; b.power = power;
                    b.lambda = lambda; b.absorb = absorb; b.med = med;
                    // v6 added the achromatic-path fold; nothing older can supply it, and 0 is
                    // the right answer (see the note at the top of this file) — the beam then
                    // re-gathers at CIE(lambda), exactly as the file was traced.
                    b.achro = 0;
                    for (int k = 0; k < 3; ++k) b.cieA[k] = 0.0f;
                    // v7 added the GATHER-time fold. Nothing older can name an emitter, and -1
                    // is the right answer: the beam re-gathers exactly as the file was traced.
                    b.emIdx = -1;
                    // v9 added chord provenance (0.312.0). Nothing older recorded it: unknown.
                    b.srcEm = -1; b.surf = 255; b.pad2 = 0;
                    // v8 added the bundle's per-member weights. Every pre-v8 bundle was an
                    // EQUAL-weight one by construction, so 1 is exact rather than merely safe.
                    for (int k = 0; k < kBeamSecMax; ++k) b.wS[k] = 1.0f;
                };
                rok = true;
                for (long long done = 0; done < nBm && rok; ) {
                    const size_t n = (size_t)std::min<long long>(65536, nBm - done);
                    if (v7) {
                        chunk7.resize(n);
                        rok = std::fread(chunk7.data(), sizeof(PhotonBeamV7), n, f) == n;
                        for (size_t i = 0; rok && i < n; ++i) {
                            PhotonBeam& b = bm->beams[(size_t)done + i];
                            const PhotonBeamV7& s = chunk7[i];
                            common(b, s.o, s.d, s.s0, s.len, s.power, s.lambda, s.absorb, s.med);
                            b.nSec = (s.nSec < 0) ? 0 : (s.nSec > kBeamSecMax ? kBeamSecMax : s.nSec);
                            for (int k = 0; k < kBeamSecMax; ++k)
                                b.lamS[k] = (k < b.nSec) ? s.lamS[k] : 0.0f;
                            // A v7 file carries BOTH folds, so `common`'s achro = 0 / emIdx = -1
                            // are not the answer here. Values above 2 cannot occur in a v7 file
                            // and are clamped rather than trusted; an `achro == 2` whose emIdx is
                            // out of range is demoted to the deposit-time fold, which `cieA`
                            // supports on its own, so a corrupt byte cannot index bowLuts wrongly.
                            unsigned char a = (s.achro > 2) ? 0 : s.achro;
                            if (a == 2 && s.emIdx < 0) a = 1;
                            b.achro = a;
                            b.emIdx = (a == 2) ? s.emIdx : (short)-1;
                            for (int k = 0; k < 3; ++k) b.cieA[k] = s.cieA[k];
                        }
                    } else if (v6) {
                        chunk6.resize(n);
                        rok = std::fread(chunk6.data(), sizeof(PhotonBeamV6), n, f) == n;
                        for (size_t i = 0; rok && i < n; ++i) {
                            PhotonBeam& b = bm->beams[(size_t)done + i];
                            const PhotonBeamV6& s = chunk6[i];
                            common(b, s.o, s.d, s.s0, s.len, s.power, s.lambda, s.absorb, s.med);
                            b.nSec = (s.nSec < 0) ? 0 : (s.nSec > kBeamSecMax ? kBeamSecMax : s.nSec);
                            for (int k = 0; k < kBeamSecMax; ++k)
                                b.lamS[k] = (k < b.nSec) ? s.lamS[k] : 0.0f;
                            // A v6 file DOES carry the deposit-time fold, so unlike the older
                            // generations `common`'s achro = 0 is not the answer here. Values
                            // above 1 cannot occur in a v6 file and are clamped rather than
                            // trusted, so a corrupt byte cannot make the gather index bowLuts
                            // with the -1 that `common` just wrote.
                            b.achro = (s.achro > 1) ? 1 : s.achro;
                            for (int k = 0; k < 3; ++k) b.cieA[k] = s.cieA[k];
                        }
                    } else if (v5) {
                        chunk5.resize(n);
                        rok = std::fread(chunk5.data(), sizeof(PhotonBeamV5), n, f) == n;
                        for (size_t i = 0; rok && i < n; ++i) {
                            PhotonBeam& b = bm->beams[(size_t)done + i];
                            const PhotonBeamV5& s = chunk5[i];
                            common(b, s.o, s.d, s.s0, s.len, s.power, s.lambda, s.absorb, s.med);
                            b.nSec = (s.nSec < 0) ? 0 : (s.nSec > kBeamSecMax ? kBeamSecMax : s.nSec);
                            for (int k = 0; k < kBeamSecMax; ++k)
                                b.lamS[k] = (k < b.nSec) ? s.lamS[k] : 0.0f;
                        }
                    } else {
                        chunk4.resize(n);
                        rok = std::fread(chunk4.data(), sizeof(PhotonBeamV4), n, f) == n;
                        for (size_t i = 0; rok && i < n; ++i) {
                            PhotonBeam& b = bm->beams[(size_t)done + i];
                            const PhotonBeamV4& s = chunk4[i];
                            common(b, s.o, s.d, s.s0, s.len, s.power, s.lambda, s.absorb, s.med);
                            b.nSec = 0;                       // v4 predates the spectral bundle
                            for (int k = 0; k < kBeamSecMax; ++k) b.lamS[k] = 0.0f;
                        }
                    }
                    done += (long long)n;
                }
            }
            if (!rok) {
                std::fprintf(stderr, "[loadmap] %s truncated beam data; ignoring\n", path);
                pm.photons.clear(); pm.pos.clear(); bm->beams.clear();
                std::fclose(f); return false;
            }
            bm->nEmitted   = bEm;
            bm->nDeposited = (size_t)bDep;
        } else if (bm && nBm == 0 && beamsMissing) {
            *beamsMissing = true;                 // file is v3 but its trace crossed no medium
        } else if (!bm && nBm > 0) {
            // The caller does not want beams, but the caustic block comes AFTER them, so the
            // beam data still has to be stepped over. Without this seek the caustic header was
            // read out of the middle of the beam array — a `-loadmap` without `-beams` on a
            // file that HAS beams produced a nonsense caustic count and then either bailed on
            // a truncation or allocated whatever those bytes happened to spell.
            // Stepped in 1 GB hops because `fseek`'s offset is a `long`, which is 32-bit on
            // Windows: a beam block past 2 GB (entirely reachable with a raised -beamcount)
            // would otherwise seek to a truncated, wrong offset.
            long long rem = nBm * beamRec;
            bool sok = true;
            while (rem > 0 && sok) {
                const long hop = (long)std::min<long long>(rem, 1LL << 30);
                sok = std::fseek(f, hop, SEEK_CUR) == 0;
                rem -= hop;
            }
            if (!sok) {
                std::fprintf(stderr, "[loadmap] %s: cannot skip the beam block; ignoring\n", path);
                pm.photons.clear(); pm.pos.clear(); std::fclose(f); return false;
            }
        }
    } else if (bm && beamsMissing) {
        *beamsMissing = true;                     // v2: predates volume support entirely
    }
    // --- caustic block (FTPMP04 only) -----------------------------------------------------
    if (v4) {
        long long nCa = 0, cEm = 0;
        bool cok = std::fread(&nCa, sizeof nCa, 1, f) == 1
                && std::fread(&cEm, sizeof cEm, 1, f) == 1;
        if (!cok) {
            std::fprintf(stderr, "[loadmap] %s truncated caustic header; ignoring\n", path);
            pm.photons.clear(); pm.pos.clear(); std::fclose(f); return false;
        }
        if (nCa > 0) {
            // Read it even when the caller passed no map: skipping would need a seek past a
            // variable-size block, and the block is the last thing in the file anyway. With
            // pmCaustic null the photons are simply dropped, which is the caller's request.
            if (pmCaustic) {
                ftalloc::resize(pmCaustic->pos, (size_t)nCa, "the caustic map positions (-loadmap)",
                                "the photon count the map was saved with");
                ftalloc::resize(pmCaustic->photons, (size_t)nCa, "the caustic map payloads (-loadmap)",
                                "the photon count the map was saved with");
                cok = std::fread(pmCaustic->pos.data(), sizeof(Vec3), (size_t)nCa, f) == (size_t)nCa;
                cok = cok && std::fread(pmCaustic->photons.data(), sizeof(Photon), (size_t)nCa, f)
                             == (size_t)nCa;
                if (!cok) {
                    std::fprintf(stderr, "[loadmap] %s truncated caustic data; ignoring\n", path);
                    pm.photons.clear(); pm.pos.clear();
                    pmCaustic->photons.clear(); pmCaustic->pos.clear();
                    std::fclose(f); return false;
                }
                pmCaustic->nEmitted = cEm;
            }
        }
    }
    std::fclose(f);
    pm.nEmitted = nEmitted;
    e.emitted += en[0]; e.absorbed += en[1]; e.sensor += en[2]; e.escaped += en[3]; e.residual += en[4];
    return true;
}
