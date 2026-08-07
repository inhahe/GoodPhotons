// Micro-benchmark for the vendored minijson parser (src/third_party/json.h).
//
// Why this exists: the viewer's played-frame profile showed `sidecar 135 ms =
// json 129 + geom 1 + dag 1 + skins 0`, i.e. parsing the ~900 KB introspection
// JSON is ~96 % of sidecar adoption and ~half the whole frame. ~7 MB/s is one to
// two orders of magnitude below a competent JSON parser, so the parser -- not the
// format -- is the thing to fix. Iterating on that through a 40-second viewer run
// is hopeless; this parses the same file in a loop and prints MB/s.
//
// Permanent tooling (tools/), not part of the ftrace build -- built on demand by
// tools/jsonbench.bat. This is the only guard against a *performance* regression
// here; the static_assert in json.h guards the noexcept property itself.
//
//   cl /std:c++17 /O2 /EHsc scraps\jsonbench.cpp /Fe:scraps\jsonbench.exe
//   scraps\jsonbench.exe out\scatter_sweep.json 10

#include "../src/third_party/json.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>

// Canonical, unambiguous re-serialisation of a parsed tree. Used to prove the
// sorted-vector rewrite produces exactly the tree the std::map version did: dump with
// each build and diff. Objects are emitted in key order (both implementations sort by
// key, so this is order-insensitive by construction), and numbers get %.17g so no
// difference can hide in the formatting.
static void dump(const minijson::Value& v, std::string& out, int depth = 0) {
    char buf[64];
    switch (v.type) {
        case minijson::Value::Null:   out += "null"; break;
        case minijson::Value::Bool:   out += v.b ? "true" : "false"; break;
        case minijson::Value::Number:
            std::snprintf(buf, sizeof buf, "%.17g", v.num); out += buf; break;
        case minijson::Value::String: out += '"'; out += v.str; out += '"'; break;
        case minijson::Value::Array:
            out += "[\n";
            for (size_t i = 0; i < v.arr.size(); ++i) {
                out.append(size_t(depth + 1) * 2, ' ');
                dump(v.arr[i], out, depth + 1);
                out += (i + 1 < v.arr.size()) ? ",\n" : "\n";
            }
            out.append(size_t(depth) * 2, ' '); out += ']';
            break;
        case minijson::Value::Object:
            out += "{\n";
            {
                size_t n = 0, total = 0;
                for (const auto& kv : v.obj) { (void)kv; ++total; }
                for (const auto& kv : v.obj) {
                    out.append(size_t(depth + 1) * 2, ' ');
                    out += '"'; out += kv.first; out += "\": ";
                    dump(kv.second, out, depth + 1);
                    out += (++n < total) ? ",\n" : "\n";
                }
            }
            out.append(size_t(depth) * 2, ' '); out += '}';
            break;
    }
}

int main(int argc, char** argv) {
    // --dump <file>: print the canonical tree instead of benchmarking.
    if (argc > 2 && std::string(argv[1]) == "--dump") {
        std::ifstream df(argv[2], std::ios::binary);
        if (!df) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        std::ostringstream dss; dss << df.rdbuf();
        const std::string dsrc = dss.str();
        minijson::Value droot; std::string derr;
        minijson::Parser dp(dsrc);
        if (!dp.parse(droot, derr)) { std::fprintf(stderr, "parse failed: %s\n", derr.c_str()); return 2; }
        std::string out;
        dump(droot, out);
        std::fwrite(out.data(), 1, out.size(), stdout);
        return 0;
    }

    const char* path = argc > 1 ? argv[1] : "out/scatter_sweep.json";
    const int   reps = argc > 2 ? std::atoi(argv[2]) : 10;

    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();
    const double mb = double(src.size()) / (1024.0 * 1024.0);
    std::printf("file %s  %.2f MB\n", path, mb);

    // The suspected mechanism. std::vector reallocation only MOVES its elements when
    // the element's move constructor is noexcept; otherwise it must COPY, to preserve
    // the strong exception guarantee. A JSON Value tree copies DEEPLY, so a non-noexcept
    // move turns every array growth into a full recursive clone of everything parsed so
    // far. On MSVC std::map's move ctor is not noexcept (it can allocate a sentinel
    // node), which would poison Value's implicit move.
    std::printf("sizeof(Value)            = %zu bytes\n", sizeof(minijson::Value));
    std::printf("nothrow_move_constructible = %d   <-- 0 means vector COPIES on growth\n",
                (int)std::is_nothrow_move_constructible<minijson::Value>::value);

    double best = 1e30, total = 0.0;
    for (int r = 0; r < reps; ++r) {
        minijson::Value root;
        std::string err;
        auto t0 = std::chrono::steady_clock::now();
        minijson::Parser p(src);
        bool ok = p.parse(root, err);
        auto t1 = std::chrono::steady_clock::now();
        if (!ok) { std::fprintf(stderr, "parse failed: %s\n", err.c_str()); return 2; }
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total += ms;
        if (ms < best) best = ms;
        std::printf("  rep %2d  %7.1f ms   %6.1f MB/s\n", r, ms, mb / (ms / 1000.0));
    }
    std::printf("best %7.1f ms  %6.1f MB/s   |  mean %7.1f ms\n",
                best, mb / (best / 1000.0), total / reps);
    return 0;
}
