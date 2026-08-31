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
// Format. `FTPMP03\n` = header, surface block, then an OPTIONAL beam block. `FTPMP02\n`
// (surface-only) still loads, so existing caches keep working and simply report no beams.
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
#include "photonmap.h"
#include "photonbeams.h"
#include "scene.h"

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
inline bool savePhotonMap(const char* path, const PhotonMap& pm,
                          const EnergyReport& e, uint64_t guard,
                          const BeamMap* bm = nullptr) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "[savemap] cannot open %s for writing\n", path); return false; }
    const char magic[8] = {'F','T','P','M','P','0','3','\n'};
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
inline bool loadPhotonMap(const char* path, PhotonMap& pm,
                          EnergyReport& e, uint64_t guard,
                          BeamMap* bm = nullptr, bool* beamsMissing = nullptr) {
    if (beamsMissing) *beamsMissing = false;
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "[loadmap] cannot open %s\n", path); return false; }
    char magic[8] = {0};
    long long nEmitted = 0, nPh = 0; double en[5] = {0,0,0,0,0}; uint64_t g = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8;
    const bool v3 = ok && std::memcmp(magic, "FTPMP03\n", 8) == 0;
    const bool v2 = ok && std::memcmp(magic, "FTPMP02\n", 8) == 0;
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
        pm.pos.resize((size_t)nPh);
        pm.photons.resize((size_t)nPh);
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
            bm->beams.resize((size_t)nBm);
            if (std::fread(bm->beams.data(), sizeof(PhotonBeam), (size_t)nBm, f) != (size_t)nBm) {
                std::fprintf(stderr, "[loadmap] %s truncated beam data; ignoring\n", path);
                pm.photons.clear(); pm.pos.clear(); bm->beams.clear();
                std::fclose(f); return false;
            }
            bm->nEmitted   = bEm;
            bm->nDeposited = (size_t)bDep;
        } else if (bm && nBm == 0 && beamsMissing) {
            *beamsMissing = true;                 // file is v3 but its trace crossed no medium
        }
    } else if (bm && beamsMissing) {
        *beamsMissing = true;                     // v2: predates volume support entirely
    }
    std::fclose(f);
    pm.nEmitted = nEmitted;
    e.emitted += en[0]; e.absorbed += en[1]; e.sensor += en[2]; e.escaped += en[3]; e.residual += en[4];
    return true;
}
