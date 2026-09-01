// Turning an out-of-memory into a message the user can act on.
//
// The bulk buffers of a photon render are sized by the command line, not by the scene:
// `-n` decides how many photons are deposited, `-beamcount` how many beams are stored,
// `-beamsplitmax` how many sub-beams the BVH split may produce. When one of those asks
// for more than the machine has, `std::vector::resize` throws `std::bad_alloc`, whose
// `what()` is the bare string "bad allocation" — so the render dies between two progress
// lines having named neither the buffer that failed, nor how big it was, nor which flag
// would make it smaller. The only way to find out was to bisect the flag by hand.
//
// These helpers wrap exactly the allocations that scale with a flag, and convert the
// failure into a `std::runtime_error` that says all three things. Everything else keeps
// the default behaviour; main()'s top-level handler has a `std::bad_alloc` catch that
// gives a generic version of the same advice for allocations nobody wrapped.
#pragma once
#include <cstddef>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <string>

namespace ftalloc {

// "48.83 GiB" / "912 MiB" / "4096 B" — binary units, because that is what the OS and the
// task manager report, and a memory ceiling is the one place a decimal GB misleads.
inline std::string humanBytes(double b) {
    const char* unit[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };
    int u = 0;
    while (b >= 1024.0 && u + 1 < (int)(sizeof(unit) / sizeof(unit[0]))) { b /= 1024.0; ++u; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), u == 0 ? "%.0f %s" : "%.2f %s", b, unit[u]);
    return std::string(buf);
}

// The message. `what` names the buffer ("the photon map"), `flags` names the knobs that
// size it ("-n"), and the count x element size is spelled out so the user can scale the
// flag by arithmetic instead of by bisection.
[[noreturn]] inline void reportOom(const char* what, size_t count, size_t elemBytes,
                                   const char* flags) {
    const double bytes = (double)count * (double)elemBytes;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "out of memory allocating %s: %zu x %zu B = %s.\n"
                  "       Lower %s and re-run. (Host RAM, not VRAM — this buffer lives on "
                  "the CPU side\n       even when the trace ran on the GPU.)",
                  what, count, elemBytes, humanBytes(bytes).c_str(), flags);
    throw std::runtime_error(buf);
}

// resize()/reserve() that report instead of throwing a bare "bad allocation". A vector
// whose growth is bounded by a command-line flag should go through one of these; a small
// or scene-sized one need not.
template <class V>
inline void resize(V& v, size_t n, const char* what, const char* flags) {
    try { v.resize(n); }
    catch (const std::bad_alloc&) { reportOom(what, n, sizeof(typename V::value_type), flags); }
    catch (const std::length_error&) { reportOom(what, n, sizeof(typename V::value_type), flags); }
}

template <class V>
inline void reserve(V& v, size_t n, const char* what, const char* flags) {
    try { v.reserve(n); }
    catch (const std::bad_alloc&) { reportOom(what, n, sizeof(typename V::value_type), flags); }
    catch (const std::length_error&) { reportOom(what, n, sizeof(typename V::value_type), flags); }
}

}  // namespace ftalloc
