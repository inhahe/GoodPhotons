// The dynamic elastic (ROADMAP 11 stage E): an as-rigid-as-possible surface with inertia,
// stepped through a shot in frame order and kept, frame by frame, in a cache the frames read.
//
// `deform elastic { dynamic { cache "png/shot/jelly_%04d.ftec"  frame 12 ... } handle { ... } }`
// on a mesh. `ftrace -bake-elastic <scene%04d.ftsl> <first> <last>` loads each frame's scene in
// order; at a dynamic elastic the loader then continues the simulation from the frame before's
// record (or from rest, the handles' first targets solved quasi-statically, when there is none),
// through the frame's motion keys, and writes the frame's record -- unless the record there was
// simulated from exactly this (chainKeyOf: the record before's chain, this frame's mesh, handles,
// path and keys), when it is kept. So a bake over the whole shot re-simulates from the first frame
// that changed, and only from there. A render reads its frame's record and takes the mesh at each
// key from it: frames stay independently renderable once baked.
//
// The step is implicit (projective-dynamics style): the inertial guess y = x + h v, then the
// ARAP energy plus (mass / (stiffness h^2)) * sum of m_i |x_i - y_i|^2 minimised by
// deform::Arap::solve, the handles' vertices held at their targets; the velocity is the step's
// move over h, less `damping` of it per second. h is in seconds (frames / fps), so `mass` and
// `stiffness` are a material's: a larger mesh of one material wobbles more slowly, as it should.
//
// A record holds the mesh at each of its frame's keys and the state at its last key (positions,
// velocities, the held vertices' targets), in float32, the key of everything it was baked from
// (rest mesh, handles, settings), of the path its frame's handles take over its keys, and of its
// whole history (chainKey): a render refuses a record baked for another mesh, handles, path or
// keys of its own frame. (It cannot see a change to an earlier frame's scene: the bake can.) The
// state a bake carries from one frame to the next is the record's own, rounded to float32, so a
// bake run in one go and one resumed part-way write the same files.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "deform.h"
#include "geometry.h"

namespace elastic {

// Whether this process is baking (`-bake-elastic`): the loader then simulates and writes records
// rather than reading them.
inline bool g_baking = false;

struct Params {
    double mass = 1.0;        // per unit area
    // The wobble's frequency grows as sqrt(stiffness / mass) and falls with the mesh's size: at
    // the defaults a 2 m bar rings at about 2 Hz, and 2e5 makes it about 6 Hz.
    double stiffness = 2.0e4; // the ARAP energy's weight
    double damping = 0.9;     // the share of velocity lost per second
    double fps = 24.0;        // frames per second: a frame's length in time
    int substeps = 8;         // steps per frame
    int iterations = 8;       // ARAP rounds per step
};

struct Record {
    uint64_t key = 0;                       // what it was baked from (see keyOf)
    uint64_t pathKey = 0;                   // the handles' path over its keys (see pathKeyOf)
    uint64_t chainKey = 0;                  // all it was simulated from (see chainKeyOf)
    long long frame = 0;
    std::vector<double> times;              // its keys' times, frames from its instant
    std::vector<std::vector<Vec3>> pos;     // the mesh at each key
    std::vector<Vec3> vel;                  // the velocities at its last key, m/s
    std::vector<Vec3> target;               // the held vertices' targets at its last key
    // The rest it was simulated on at its last key, where deformers before the dynamic elastic
    // move it (bakeFrame's `moving`); empty where its rest is the mesh as loaded.
    std::vector<Vec3> restLast;
};

// FNV-1a over bytes, for the record's key.
struct Hasher {
    uint64_t h = 14695981039346656037ULL;
    void bytes(const void* p, size_t n) {
        const unsigned char* c = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ULL;
    }
    void d(double v) { bytes(&v, sizeof v); }
    void i(long long v) { bytes(&v, sizeof v); }
};

// The key of a dynamic elastic: its rest mesh, its triangles, which vertices each handle holds,
// and its settings. Targets are not in it -- they are what changes from frame to frame.
inline uint64_t keyOf(const std::vector<Vec3>& rest, const std::vector<std::array<int, 3>>& tris,
                      const std::vector<char>& held, const Params& P) {
    Hasher h;
    h.i((long long)rest.size());
    for (const Vec3& v : rest) { h.d(v.x); h.d(v.y); h.d(v.z); }
    h.i((long long)tris.size());
    for (const auto& t : tris) { h.i(t[0]); h.i(t[1]); h.i(t[2]); }
    h.bytes(held.data(), held.size());
    h.d(P.mass); h.d(P.stiffness); h.d(P.damping); h.d(P.fps); h.i(P.substeps); h.i(P.iterations);
    return h.h;
}

inline Vec3 f32(const Vec3& v) { return Vec3{(double)(float)v.x, (double)(float)v.y, (double)(float)v.z}; }

// The key of a frame's handles' path: the held vertices' targets at each of its keys, as the
// record keeps them (float32) -- and, where deformers before the dynamic elastic move its rest
// (`moving`, null where they do not), that rest at each key too. Not in keyOf, since it changes
// from frame to frame: a render compares its own frame's (a change to an earlier frame's handles
// needs that frame re-baked, and the frames after it).
inline uint64_t pathKeyOf(const std::vector<char>& held, const std::vector<std::vector<Vec3>>& targets,
                          const std::vector<std::vector<Vec3>>* moving = nullptr) {
    Hasher h;
    h.i((long long)targets.size());
    for (const auto& tk : targets)
        for (size_t i = 0; i < tk.size(); ++i)
            if (held[i]) { const Vec3 q = f32(tk[i]); h.d(q.x); h.d(q.y); h.d(q.z); }
    if (moving)
        for (const auto& rk : *moving)
            for (const Vec3& v : rk) { const Vec3 q = f32(v); h.d(q.x); h.d(q.y); h.d(q.z); }
    return h.h;
}

// The key of everything a record was simulated from: the record before's chain key (0 for one
// started at rest), and this frame's key, path, number and key times. A bake leaves a record whose
// chain key still matches alone, so a rerun re-simulates from the first frame that changed.
inline uint64_t chainKeyOf(uint64_t prevChain, uint64_t key, uint64_t pathKey, long long frame,
                           const std::vector<double>& times) {
    Hasher h;
    h.bytes(&prevChain, sizeof prevChain);
    h.bytes(&key, sizeof key);
    h.bytes(&pathKey, sizeof pathKey);
    h.i(frame);
    for (double t : times) h.d(t);
    return h.h;
}

// `pat` with frame number `k` in its one integer conversion (%d, %04d, ...), or false for a
// pattern with none, more than one, or another conversion.
inline bool framePath(const std::string& pat, long long k, std::string& out) {
    int convs = 0;
    for (size_t i = 0; i < pat.size(); ++i) {
        if (pat[i] != '%') continue;
        if (i + 1 < pat.size() && pat[i + 1] == '%') { ++i; continue; }
        size_t j = i + 1;
        while (j < pat.size() && pat[j] >= '0' && pat[j] <= '9') ++j;
        if (j >= pat.size() || pat[j] != 'd') return false;
        ++convs;
    }
    if (convs != 1) return false;
    std::vector<char> buf(pat.size() + 32);
    std::snprintf(buf.data(), buf.size(), pat.c_str(), (int)k);
    out = buf.data();
    return true;
}

// ---- the file ----------------------------------------------------------------------------
// "FTEC", u32 version, u64 key, u64 pathKey, u64 chainKey, i64 frame, u32 vertices, u32 keys,
// f64 times[keys],
// f32 xyz pos[keys][vertices], f32 xyz vel[vertices], f32 xyz target[vertices], and in version 3
// only, f32 xyz restLast[vertices]. A record with no moving rest is written as version 2, so the
// records of a shot whose dynamic elastic comes first are what they always were.
constexpr uint32_t kVersion = 3;
inline bool write(const std::string& path, const Record& r, std::string& err) {
    const std::string tmp = path + ".part";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) { err = "cannot write " + tmp; return false; }
    const uint32_t ver = r.restLast.empty() ? 2 : 3, nv = (uint32_t)r.vel.size(), nk = (uint32_t)r.times.size();
    const int64_t fr = r.frame;
    bool ok = std::fwrite("FTEC", 1, 4, f) == 4 && std::fwrite(&ver, 4, 1, f) == 1 &&
              std::fwrite(&r.key, 8, 1, f) == 1 && std::fwrite(&r.pathKey, 8, 1, f) == 1 &&
              std::fwrite(&r.chainKey, 8, 1, f) == 1 && std::fwrite(&fr, 8, 1, f) == 1 &&
              std::fwrite(&nv, 4, 1, f) == 1 && std::fwrite(&nk, 4, 1, f) == 1 &&
              std::fwrite(r.times.data(), 8, nk, f) == nk;
    std::vector<float> buf((size_t)nv * 3);
    const auto put = [&](const std::vector<Vec3>& v) {
        for (size_t i = 0; i < nv; ++i) { buf[3 * i] = (float)v[i].x; buf[3 * i + 1] = (float)v[i].y; buf[3 * i + 2] = (float)v[i].z; }
        return std::fwrite(buf.data(), sizeof(float), buf.size(), f) == buf.size();
    };
    for (uint32_t k = 0; ok && k < nk; ++k) ok = put(r.pos[k]);
    ok = ok && put(r.vel) && put(r.target);
    if (ver == 3) ok = ok && put(r.restLast);
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) { std::remove(tmp.c_str()); err = "writing " + path + " failed"; return false; }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { err = "cannot rename " + tmp + " to " + path; return false; }
    return true;
}

// False with `err` empty when the file does not exist, else with the reason it is unreadable.
inline bool read(const std::string& path, Record& r, std::string& err) {
    err.clear();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    uint32_t ver = 0, nv = 0, nk = 0;
    int64_t fr = 0;
    const bool ftec = std::fread(magic, 1, 4, f) == 4 && !std::memcmp(magic, "FTEC", 4) &&
                      std::fread(&ver, 4, 1, f) == 1;
    if (ftec && ver != 2 && ver != kVersion) {
        std::fclose(f);
        err = path + " is an elastic cache record of another ftrace (version " + std::to_string(ver) +
              ", this one reads 2 and " + std::to_string(kVersion) + ") -- bake the shot again";
        return false;
    }
    bool ok = ftec && std::fread(&r.key, 8, 1, f) == 1 && std::fread(&r.pathKey, 8, 1, f) == 1 &&
              std::fread(&r.chainKey, 8, 1, f) == 1 && std::fread(&fr, 8, 1, f) == 1 &&
              std::fread(&nv, 4, 1, f) == 1 && std::fread(&nk, 4, 1, f) == 1 && nk >= 1 && nk <= 4096 &&
              nv <= 100000000u;
    if (ok) {
        r.frame = fr;
        r.times.resize(nk);
        ok = std::fread(r.times.data(), 8, nk, f) == nk;
    }
    // Sized only once the header has passed: a corrupt vertex count must not allocate.
    std::vector<float> buf(ok ? (size_t)nv * 3 : 0);
    const auto get = [&](std::vector<Vec3>& v) {
        if (std::fread(buf.data(), sizeof(float), buf.size(), f) != buf.size()) return false;
        v.resize(nv);
        for (size_t i = 0; i < nv; ++i) v[i] = Vec3{buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]};
        return true;
    };
    if (ok) {
        r.pos.resize(nk);
        for (uint32_t k = 0; ok && k < nk; ++k) ok = get(r.pos[k]);
        ok = ok && get(r.vel) && get(r.target);
        if (ver == 3) ok = ok && get(r.restLast);
        else r.restLast.clear();
    }
    std::fclose(f);
    if (!ok) err = path + " is not an elastic cache record (or is truncated)";
    return ok;
}

// ---- the simulation ----------------------------------------------------------------------
// Advance `x`, `v` by `h` seconds towards the held vertices' targets `tgt`: one implicit step,
// on the rest `restNow` (null: the mesh the solver was built on). False if a stop arrived part-way.
[[nodiscard]] inline bool step(const deform::Arap& arap, const std::vector<char>& held, const std::vector<Vec3>& tgt,
                 double h, const Params& P, std::vector<Vec3>& x, std::vector<Vec3>& v,
                 const std::vector<Vec3>* restNow = nullptr) {
    const size_t n = x.size();
    std::vector<Vec3> y(n), xn(n);
    for (size_t i = 0; i < n; ++i) y[i] = x[i] + v[i] * h;
    xn = y;
    if (!arap.solve(held, tgt, xn, P.iterations, P.mass / (P.stiffness * h * h), &y, false, restNow))
        return false;
    const double keep = std::pow(std::max(0.0, 1.0 - P.damping), h);
    for (size_t i = 0; i < n; ++i) {
        v[i] = (xn[i] - x[i]) * (keep / h);
        x[i] = xn[i];
    }
    return true;
}

// Run from state (`x`, `v`, targets `t0`) at absolute time `a` (frames) to `b`, the targets moving
// linearly to `t1`, in about P.substeps steps a frame -- and the rest likewise from `r0` to `r1`
// where deformers before the dynamic elastic move it (both null where they do not). False if a
// stop arrived part-way.
[[nodiscard]] inline bool run(const deform::Arap& arap, const std::vector<char>& held, const std::vector<Vec3>& t0,
                const std::vector<Vec3>& t1, double a, double b, const Params& P,
                std::vector<Vec3>& x, std::vector<Vec3>& v,
                const std::vector<Vec3>* r0 = nullptr, const std::vector<Vec3>* r1 = nullptr) {
    if (!(b > a)) return true;
    const int n = std::max(1, (int)std::lround((b - a) * P.substeps));
    const double h = (b - a) / n / P.fps;
    std::vector<Vec3> tgt(t0.size()), rest(r0 && r1 ? r0->size() : 0);
    for (int s = 1; s <= n; ++s) {
        const double f = (double)s / n;
        for (size_t i = 0; i < tgt.size(); ++i) tgt[i] = t0[i] + (t1[i] - t0[i]) * f;
        for (size_t i = 0; i < rest.size(); ++i) rest[i] = (*r0)[i] + ((*r1)[i] - (*r0)[i]) * f;
        if (!step(arap, held, tgt, h, P, x, v, rest.empty() ? nullptr : &rest)) return false;
    }
    return true;
}

// Bake one frame: `rest` and `tris` the mesh, `held` the handles' vertices, `targets[k]` their
// targets at each of the frame's keys (`times`, frames from its instant), `prev` the frame
// before's record or null. `moving[k]` (null: none) is the rest at each key where deformers before
// the dynamic elastic move it -- their output -- and the simulation rests on it, linearly between
// keys, the solver's weights and masses staying `rest`'s. Fills `out`; false (`err`) when the
// frames' intervals overlap.
inline bool bakeFrame(const std::vector<Vec3>& rest, const std::vector<std::array<int, 3>>& tris,
                      const std::vector<char>& held, const std::vector<std::vector<Vec3>>& targets,
                      const std::vector<double>& times, long long frame, const Params& P,
                      const Record* prev, Record& out, std::string& err,
                      const std::vector<std::vector<Vec3>>* moving = nullptr) {
    deform::Arap arap;
    arap.build(rest, tris);
    const size_t K = times.size();
    std::vector<Vec3> x, v;
    std::vector<Vec3> lastT;
    double t = (double)frame + times[0];
    if (prev) {
        const double tPrev = (double)prev->frame + prev->times.back();
        if (tPrev > t + 1e-9) {
            err = "frame " + std::to_string(frame) + "'s keys begin before frame " +
                  std::to_string(prev->frame) + "'s end: a dynamic elastic needs a motion interval no "
                  "longer than a frame (-0.5 0.5, say)";
            return false;
        }
        x = prev->pos.back();
        v = prev->vel;
        lastT = prev->target;
        // Across the gap the rest moves from the frame before's last to this frame's first (the
        // mesh as loaded where either has no moving rest).
        const std::vector<Vec3>* rPrev = !prev->restLast.empty() ? &prev->restLast : (moving ? &rest : nullptr);
        const std::vector<Vec3>* rNow = moving ? &(*moving)[0] : (rPrev ? &rest : nullptr);
        if (!run(arap, held, lastT, targets[0], tPrev, t, P, x, v, rPrev, rNow)) {   // the gap between frames, if any
            err = "stopped by request";
            return false;
        }
    } else {
        // From rest, still: the handles' first targets, met quasi-statically.
        x = moving ? (*moving)[0] : rest;
        if (!arap.solve(held, targets[0], x, std::max(1, P.iterations * 4), 0.0, nullptr, /*fromRest*/true,
                        moving ? &(*moving)[0] : nullptr)) {
            err = "stopped by request";
            return false;
        }
        v.assign(rest.size(), Vec3{0, 0, 0});
    }
    out = Record{};
    out.frame = frame;
    out.times = times;
    out.pos.resize(K);
    out.pos[0] = x;
    for (size_t k = 1; k < K; ++k) {
        if (!run(arap, held, targets[k - 1], targets[k], (double)frame + times[k - 1], (double)frame + times[k],
                 P, x, v, moving ? &(*moving)[k - 1] : nullptr, moving ? &(*moving)[k] : nullptr)) {
            err = "stopped by request";
            return false;
        }
        out.pos[k] = x;
    }
    // Rounded as the file keeps it, so a bake resumed from the file continues from the same state.
    for (auto& p : out.pos) for (Vec3& q : p) q = f32(q);
    out.vel.resize(v.size());
    for (size_t i = 0; i < v.size(); ++i) out.vel[i] = f32(v[i]);
    out.target.resize(targets[K - 1].size());
    for (size_t i = 0; i < out.target.size(); ++i) out.target[i] = f32(targets[K - 1][i]);
    if (moving)
        for (const Vec3& q : (*moving)[K - 1]) out.restLast.push_back(f32(q));
    return true;
}

}  // namespace elastic
