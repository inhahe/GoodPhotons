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
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace assetbytes {

// A narrow path string is UTF-8 here (ftrace sets the console and its own I/O to
// UTF-8), which is NOT what `std::filesystem::path(std::string)` assumes on Windows —
// it decodes the native ANSI code page and mangles any non-ASCII asset name. The
// correct spelling is `u8path`, which C++20 deprecated in favour of a `char8_t`
// overload that this header cannot rely on: it is also fed to nvcc. So the deprecated
// call is made in exactly one place, with the warning silenced there rather than
// project-wide.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
inline std::filesystem::path toPath(const std::string& s) {
    return std::filesystem::u8path(s);
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

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

// ---------------------------------------------------------------------------
// WHERE A RELATIVE ASSET PATH IS LOOKED FOR (0.192.0)
//
// Until now there was no answer to that question at all: every authored path went
// straight to `fopen`, so it resolved against the PROCESS WORKING DIRECTORY and a
// scene was loadable from exactly one directory — the one its paths happened to be
// written relative to. `ftrace scenes/gallery.ftsl` worked from the repo root and
// `cd scenes && ..\ftrace gallery.ftsl` did not, because `textures/marble.png` is
// relative to the project, not to wherever the shell happens to be standing. The
// failure was worse than an error message: a missing texture failed one branch of a
// `prefer` block, the fallback branch produced an empty scene, and the empty scene
// crashed the GPU (see known-issues.md, 0.191.2).
//
// The fix is a short, ordered search path, applied by `resolve()` at every point a
// path becomes a file handle:
//
//   1. THE PATH AS AUTHORED (i.e. relative to the cwd).  First, on purpose — every
//      invocation that works today keeps resolving to exactly the file it resolves
//      to today, so this change can only turn failures into successes, never move a
//      load from one file to another.
//   2. THE DIRECTORY OF THE SCENE FILE.  A scene's assets belong to the scene, which
//      is the rule every other tool that references external files uses (glTF's own
//      buffer URIs, below, already worked this way).
//   3. THE SCENE'S ANCESTOR DIRECTORIES, up to three levels.  The reason this exists
//      rather than stopping at (2): the natural layout puts scenes in a subdirectory
//      of the project and assets in sibling subdirectories — `proj/scenes/x.ftsl`
//      naming `textures/y.png` means `proj/textures/y.png`. That is this repo's own
//      layout, and stopping at the scene directory would not fix the bug that
//      prompted the work. Bounded so a stray name can't be answered from the root of
//      the drive.
//   4. THE DIRECTORY OF ftrace.exe.  Engine data (`data/glass/*`, `data/metal/*`) is
//      shipped beside the binary and has nothing to do with the scene, so it must
//      resolve no matter where either the cwd or the scene is.
//
// A path that matches nothing is handed back UNCHANGED, so the error message names
// what the author wrote rather than the last candidate tried; `describeOpenFailure`
// then lists every directory that was searched.
//
// `resolve()` is deliberately safe to call on a path that is already resolved (rule 1
// returns it untouched) and on directories as well as files, so it can be applied
// blindly at any open site without the caller needing to know which it has.
inline std::string& exeDirRef()   { static std::string d; return d; }
inline std::string& sceneDirRef() { static std::string d; return d; }

// Set once at startup from the real module path (argv[0] is not dependable).
inline void setExeDir(const std::string& d) { exeDirRef() = d; }

// The scene directory is a property of the load in progress, not of the process, so
// it is scoped: a nested or subsequent load of a different scene must not inherit it,
// and a `prefer` block try-building several branches must see the same one throughout.
struct ScopedSceneDir {
    std::string prev;
    explicit ScopedSceneDir(const std::string& dir) : prev(sceneDirRef()) { sceneDirRef() = dir; }
    ~ScopedSceneDir() { sceneDirRef() = prev; }
    ScopedSceneDir(const ScopedSceneDir&) = delete;
    ScopedSceneDir& operator=(const ScopedSceneDir&) = delete;
};

// Join a search-path root to a relative path. The empty root means "as authored".
inline std::string joinDir(const std::string& dir, const std::string& rel) {
    if (dir.empty()) return rel;
    return (toPath(dir) / toPath(rel)).string();
}

// The roots of the search path, in order, deduplicated. "" (the cwd) is always first.
inline std::vector<std::string> searchRoots() {
    namespace fs = std::filesystem;
    std::vector<std::string> v;
    v.push_back(std::string());
    auto add = [&v](const fs::path& p) {
        std::string s = p.string();
        if (s.empty()) return;
        for (const std::string& e : v)
            if (normKey(e) == normKey(s)) return;
        v.push_back(std::move(s));
    };
    if (!sceneDirRef().empty()) {
        fs::path p = toPath(sceneDirRef());
        add(p);
        for (int up = 0; up < 3; ++up) {
            if (!p.has_parent_path()) break;
            fs::path q = p.parent_path();
            if (q == p) break;              // hit the root
            p = q;
            add(p);
        }
    }
    if (!exeDirRef().empty()) add(toPath(exeDirRef()));
    return v;
}

// The path to actually open for an authored asset reference. Absolute paths and
// paths that already exist are returned untouched; see the comment above for order.
inline std::string resolve(const std::string& path) {
    namespace fs = std::filesystem;
    if (path.empty()) return path;
    std::error_code ec;
    if (toPath(path).is_absolute()) return path;
    for (const std::string& root : searchRoots()) {
        std::string cand = joinDir(root, path);
        if (fs::exists(toPath(cand), ec)) return cand;
    }
    return path;
}

// Read a whole file. Returns false if it cannot be opened; `out` is replaced.
// Goes through `resolve`, so every loader that reads bytes through this function
// (mesh.h's OBJ/PLY/STL/FBX readers, among others) gets the search path for free.
inline bool readFile(const char* path, std::string& out) {
    const std::string real = resolve(path ? path : "");
    std::FILE* fp = std::fopen(real.c_str(), "rb");
    if (!fp) return false;
    out.clear();
    char tmp[1 << 16];
    size_t got;
    while ((got = std::fread(tmp, 1, sizeof tmp, fp)) > 0) out.append(tmp, got);
    std::fclose(fp);
    return true;
}

// Why an open failure gets a whole function (0.191.1). "cannot open <path>" answers
// the wrong question. The three ways it actually fails want three different reactions
// from the user, and the message must say which one happened:
//
//   * the path names nothing        -> it's a typo or a stale path; look at the name
//   * the path names a directory    -> you passed the folder, not the file in it
//   * the path exists but won't open -> permissions, or something holds it open
//
// The typo case is worth more than a label, because a mistyped asset name is nearly
// always ONE edit away from a file sitting in the same directory — and the user cannot
// see that, since their eyes read the name they meant. So we list the directory and
// offer the near misses by Levenshtein distance. This is not a nicety: the bug that
// prompted it (see known-issues.md) had a dropped leading character in a 30-character
// filename produce a completely silent grey render, and even after the load was made
// to fail loudly, `cannot open` still didn't point at the one-character difference.
//
// Bounded on purpose: directories are read at most once per failure, capped at 4000
// entries, and this runs only on a path that has ALREADY failed — so it is never on
// any hot path and can afford to be thorough.
inline int editDistance(const std::string& a, const std::string& b) {
    const size_t n = a.size(), m = b.size();
    std::vector<int> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = (int)j;
    for (size_t i = 1; i <= n; ++i) {
        cur[0] = (int)i;
        for (size_t j = 1; j <= m; ++j) {
            const char ca = (char)std::tolower((unsigned char)a[i - 1]);
            const char cb = (char)std::tolower((unsigned char)b[j - 1]);
            int del = prev[j] + 1, ins = cur[j - 1] + 1, sub = prev[j - 1] + (ca != cb ? 1 : 0);
            cur[j] = del < ins ? (del < sub ? del : sub) : (ins < sub ? ins : sub);
        }
        prev.swap(cur);
    }
    return prev[m];
}

// Human-readable reason a path could not be read, with "did you mean" when the name
// looks like a typo. Returns a phrase meant to follow the path in an error message.
inline std::string describeOpenFailure(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path p = toPath(resolve(path));

    if (fs::is_directory(p, ec))
        return "that is a directory, not a file";

    if (fs::exists(p, ec)) {
        // It's there and we still couldn't read it. Size is a useful tell: a
        // zero-byte file is usually a half-finished write, not a permission problem.
        const uintmax_t sz = fs::file_size(p, ec);
        if (!ec && sz == 0) return "the file exists but is empty (0 bytes)";
        return "the file exists but could not be opened for reading — check "
               "permissions, or whether another program is holding it open";
    }

    std::string msg = "no such file";

    // The search path (0.192.0) means "the directory" is now several directories, and
    // the answer the user needs is which ones were consulted — otherwise a path that
    // resolves for one invocation and not another looks like the renderer's whim. So
    // the near-miss scan runs under EVERY root, and the roots are named in the message.
    const std::vector<std::string> roots =
        toPath(path).is_absolute() ? std::vector<std::string>{std::string()} : searchRoots();
    const std::string want = toPath(path).filename().string();

    std::vector<std::pair<int, std::string>> cand;   // (edit distance, directory/name)
    std::vector<std::string> dirsTried;
    bool anyDirExisted = false;
    for (const std::string& root : roots) {
        const fs::path full = toPath(joinDir(root, path));
        const fs::path dir  = full.has_parent_path() ? full.parent_path() : fs::path(".");
        const std::string dirStr = fs::absolute(dir, ec).lexically_normal().string();
        bool dup = false;
        for (const std::string& d : dirsTried) if (normKey(d) == normKey(dirStr)) dup = true;
        if (dup) continue;
        dirsTried.push_back(dirStr);
        if (!fs::is_directory(dir, ec)) continue;
        anyDirExisted = true;
        int scanned = 0;
        for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
             it != end && scanned < 4000; it.increment(ec), ++scanned) {
            if (ec) break;
            if (!it->is_regular_file(ec)) continue;
            const std::string have = it->path().filename().string();
            const int d = editDistance(want, have);
            // Accept a quarter of the name's length in edits, at least 1 and at most 6 —
            // enough for a dropped char, a case slip or a wrong extension, not enough to
            // "suggest" an unrelated file in a directory of similar names.
            int budget = (int)want.size() / 4;
            if (budget < 1) budget = 1;
            if (budget > 6) budget = 6;
            // One root: the bare name reads better, and it is unambiguous. Several:
            // the name alone would not say which directory it was found in.
            if (d <= budget)
                cand.emplace_back(d, roots.size() == 1 ? have : (dir / toPath(have)).string());
        }
    }

    if (!anyDirExisted) {
        if (dirsTried.size() == 1) {
            msg += " (and its directory does not exist either: " + dirsTried[0] + ")";
        } else {
            msg += " (and its directory does not exist under any of the places searched: ";
            for (size_t i = 0; i < dirsTried.size(); ++i)
                msg += (i ? ", " : "") + dirsTried[i];
            msg += ")";
        }
        return msg;
    }

    std::sort(cand.begin(), cand.end());
    if (!cand.empty()) {
        msg += " — did you mean ";
        const size_t show = cand.size() < 3 ? cand.size() : 3;
        for (size_t i = 0; i < show; ++i) {
            if (i) msg += (i + 1 == show) ? " or " : ", ";
            msg += "'" + cand[i].second + "'";
        }
        msg += "?";
    }
    if (dirsTried.size() > 1) {
        msg += " [searched: ";
        for (size_t i = 0; i < dirsTried.size(); ++i)
            msg += (i ? ", " : "") + dirsTried[i];
        msg += "]";
    }
    return msg;
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
