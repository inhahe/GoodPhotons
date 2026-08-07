// assetbytes.h — where a scene's file-backed assets actually come from.
//
// Two problems, one seam.
//
// (1) THE LIVE CHANNEL SHOULDN'T USE FILES AT ALL. The loom viewer re-derives its
//     scene every frame and used to hand ftrace the result through `%TEMP%`. Measured
//     2026-08-06 (`scraps/freshread.py`): on this machine, `open()`ing a file that
//     another process wrote a millisecond ago costs a flat **~8 ms before the first
//     byte is read** — Windows Defender's on-access scan. It is a size *threshold*,
//     not a throughput: under ~8 KB it is ~1 ms, and from 32 KB to 1 MB it sits flat
//     at 8–10 ms. Two files per frame cleared that threshold, so ~17 ms of a 130 ms
//     frame was antivirus reading back what our own child had just written.
//     `Overlay` is the fix: loom sends the bytes down the pipe the two processes
//     already share, and the loader takes them from here instead of from a path.
//
// (2) WHEN THERE *IS* A FILE, ITS SCAN SHOULD OVERLAP WITH THE PARSE. An ordinary
//     `.ftsl` load spends ~16 ms parsing text before it opens the first mesh, and
//     more between meshes. That is idle I/O time, and the gate above is otherwise
//     charged serially in front of every open. `Warmer` reads every asset the scene
//     text mentions on one background thread, so the scan happens during work the
//     loader was doing anyway. It reads and DISCARDS: the point is the OS/AV cache,
//     so this costs constant memory and works for every format, including the ones
//     (glTF, FBX) whose loaders insist on a path.
//
//     What makes this work is a property that took three wrong benchmarks to pin
//     down (see `known-issues.md`): the gate is **per path and stubbornly linear**
//     — 24 fresh files cost 234 ms of pure scan, and *waiting* does not help (a full
//     second of sleep between writing a file and reading it saves nothing) — but it
//     **is overlappable**, because touching a file early lets its scan drain
//     concurrently. Measured: 24 files touched early cost 1.2 ms each to open
//     afterwards versus 7.6 ms each untouched. Prefetching is therefore the only
//     lever that exists here, and it is a real one: −21.5 % on a 24-mesh scene,
//     −6.2 % on a 27 MB one, and nothing lost when the files are already warm.
//
//     Beware when benchmarking this: preparing a "cold" file by writing to it is
//     itself the early touch, so a harness that freshens its own control warms it.
//     That is exactly how this optimization first measured as a 2 % regression.
//
// Neither mechanism can change what gets loaded. The overlay is keyed by the exact
// path the scene names, and warming is pure prefetch whose worst case is a wasted
// read.
#pragma once
#include <cctype>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace assetbytes {

// Read a whole file. Returns false if it cannot be opened; `out` is replaced.
inline bool readFile(const char* path, std::string& out) {
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    out.clear();
    char tmp[1 << 16];
    size_t got;
    while ((got = std::fread(tmp, 1, sizeof tmp, fp)) > 0) out.append(tmp, got);
    std::fclose(fp);
    return true;
}

// Paths reach us from two directions — loom writes posix separators into the `.ftsl`
// it emits, ftrace's own scenes are authored by hand — so a key is normalised to
// forward slashes and lowercase before it is stored or looked up. Windows paths are
// case-insensitive anyway, and a `\` vs `/` mismatch silently defeating the overlay
// would show up only as "the speedup didn't happen", which is exactly the kind of
// failure that goes unnoticed for months.
inline std::string normKey(const std::string& path) {
    std::string k = path;
    for (char& c : k) {
        if (c == '\\') c = '/';
        else if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    }
    return k;
}

// Asset contents supplied out-of-band for one scene load. Not owned by the loader:
// the caller (the viewer's live channel) fills it, passes a pointer into
// `ftsl::loadSource`, and keeps it alive for the duration of the call.
struct Overlay {
    std::map<std::string, std::string> byPath;

    void put(const std::string& path, std::string bytes) {
        byPath[normKey(path)] = std::move(bytes);
    }
    // Pointer into the overlay, or null. Returning a pointer rather than a copy
    // matters: these are megabyte-scale and a scene may name the same file twice.
    const std::string* get(const std::string& path) const {
        auto it = byPath.find(normKey(path));
        return it == byPath.end() ? nullptr : &it->second;
    }
    bool empty() const { return byPath.empty(); }
    size_t bytes() const {
        size_t n = 0;
        for (const auto& kv : byPath) n += kv.second.size();
        return n;
    }
};

// Background prefetch of asset files, so an on-access scanner runs during the parse
// instead of after it. Reads and discards — see the header comment.
class Warmer {
public:
    Warmer() = default;
    Warmer(const Warmer&) = delete;
    Warmer& operator=(const Warmer&) = delete;
    ~Warmer() { join(); }

    // Files bigger than this are skipped. The cost being hidden is a per-file scan
    // gate that does not grow with size, so there is nothing extra to win on a huge
    // asset — while the doubled read traffic would be real. (Scanners typically skip
    // very large files outright, so the gate isn't there to hide anyway.)
    static constexpr long long kMaxBytes = 64ll << 20;

    // `paths` is a best-effort list; anything that doesn't open is simply skipped.
    // Starting twice without joining is a no-op on the second call.
    void start(std::vector<std::string> paths) {
        if (th_.joinable() || paths.empty()) return;
        th_ = std::thread([paths = std::move(paths)] {
            char tmp[1 << 16];
            for (const std::string& p : paths) {
                std::FILE* fp = std::fopen(p.c_str(), "rb");
                if (!fp) continue;
                if (std::fseek(fp, 0, SEEK_END) == 0) {
                    long long sz = std::ftell(fp);
                    if (sz > kMaxBytes) { std::fclose(fp); continue; }
                    std::fseek(fp, 0, SEEK_SET);
                }
                // Touch every byte: the scanner gates the *open*, but a lazily
                // mapped read can defer work, and we want none of it left.
                while (std::fread(tmp, 1, sizeof tmp, fp) > 0) {}
                std::fclose(fp);
            }
        });
    }
    void join() { if (th_.joinable()) th_.join(); }

private:
    std::thread th_;
};

// Every asset path a scene text names, in source order, deduplicated.
//
// This is a deliberate TEXT SCAN, not a parse: its whole value is running *before*
// the parse it is meant to overlap with. It looks for `file` followed by a quoted
// string, which is the one spelling every asset-bearing block uses (`mesh`,
// `mesh_asset`, `texture`, `medium`, …). A false positive costs one wasted read and
// nothing else — the result is never used as geometry, only as a hint about which
// files to touch — so the scan can afford to be simple.
inline std::vector<std::string> scanAssetPaths(const std::string& src) {
    std::vector<std::string> out;
    std::map<std::string, bool> seen;
    const size_t n = src.size();
    for (size_t i = 0; i + 4 < n; ++i) {
        if (src.compare(i, 4, "file") != 0) continue;
        // `file` must be its own word: `profile "x"` is not an asset statement.
        if (i > 0) {
            char p = src[i - 1];
            if (std::isalnum((unsigned char)p) || p == '_') continue;
        }
        size_t j = i + 4;
        while (j < n && (src[j] == ' ' || src[j] == '\t')) ++j;
        if (j >= n || src[j] != '"') continue;
        size_t e = src.find('"', ++j);
        if (e == std::string::npos) continue;
        std::string path = src.substr(j, e - j);
        i = e;
        if (path.empty()) continue;
        if (seen.emplace(normKey(path), true).second) out.push_back(std::move(path));
    }
    return out;
}

} // namespace assetbytes
