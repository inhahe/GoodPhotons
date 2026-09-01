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
//
// Naming the buffer is still only half a diagnosis, because the message it produced —
// "out of memory allocating the photon map positions: 201.03 MiB. Lower -n" — is actively
// MISLEADING when 201 MiB is not the problem. A machine whose commit limit has been eaten
// by *other* programs fails the next allocation whatever its size, so the render that dies
// is simply the one that asked next, and lowering `-n` treats a symptom that was never
// ours. The two cases are told apart by one number the process already knows: how much
// memory THIS render is holding. A 201 MiB failure in a process holding 5 GiB is the
// machine being full; a 40 GiB failure in a process holding 40 GiB is the flag being too
// big. So `reportOom` prints the system's memory state alongside the buffer, and says
// which of the two it is looking at. The query itself lives behind a hook (`memStat`)
// installed by main.cpp: it needs <windows.h>, and this header is included by four other
// translation units — including a .cu — that have no business pulling that in.
#pragma once
#include <cstddef>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <string>

namespace ftalloc {

// A snapshot of where memory actually went. `ok` is false when the platform hook is not
// installed (or failed), in which case reportOom simply omits the whole paragraph rather
// than guessing. "Commit" is the number that matters on Windows: an allocation fails when
// the system-wide commit charge hits the limit (RAM + page file), which can happen with
// gigabytes of physical RAM still free.
struct MemStat {
    bool   ok          = false;
    double procBytes   = 0;   // this process's private commit
    double availCommit = 0;   // system commit still available
    double totalCommit = 0;   // system commit limit (RAM + page file)
    double availPhys   = 0;
    double totalPhys   = 0;
};

// Installed by main() at startup. Null in any TU/build that never sets it.
inline MemStat (*memStat)() = nullptr;

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

// The paragraph that says whether the machine was full or the flags were too big. Empty
// when no hook is installed. Pass `failedBytes < 0` when the size of the failed request is
// unknown (the bad_alloc backstop) — the verdict then rests on the second test alone.
//
// Two independent ways for the failure to be OURS, either of which is enough:
//   * the request is a large fraction of what we already hold — it really could be the
//     straw. 25% is well clear of the noise while still below a doubling vector's own
//     growth step (~50-100% of what it holds), which must always count.
//   * we are most of the system's committed memory — then whatever else is running is a
//     rounding error and the limit is ours to respect.
// Failing both, the machine filled up around us and shrinking our flags is the wrong move.
inline std::string memAdvice(double failedBytes) {
    if (!memStat) return std::string();
    const MemStat m = memStat();
    if (!m.ok) return std::string();
    const double sysUsed = m.totalCommit - m.availCommit;
    const bool ours = (m.procBytes <= 0.0) ||
                      (failedBytes >= 0.0 && failedBytes >= 0.25 * m.procBytes) ||
                      (sysUsed > 0.0 && m.procBytes >= 0.5 * sysUsed);
    char buf[768];
    std::snprintf(buf, sizeof buf,
        "\n       This render was holding %s; the system had %s of commit left out of %s\n"
        "       (%s of %s physical free).\n%s",
        humanBytes(m.procBytes).c_str(), humanBytes(m.availCommit).c_str(),
        humanBytes(m.totalCommit).c_str(), humanBytes(m.availPhys).c_str(),
        humanBytes(m.totalPhys).c_str(),
        ours ? ""
             : "       This render is only a small part of what the machine has committed, so\n"
               "       the MACHINE ran out rather than this render — other programs are using\n"
               "       it. Close something and re-run; shrinking the flags below only makes\n"
               "       this render smaller than a limit that was never really about it.\n");
    return std::string(buf);
}

// The message. `what` names the buffer ("the photon map"), `flags` names the knobs that
// size it ("-n"), and the count x element size is spelled out so the user can scale the
// flag by arithmetic instead of by bisection.
[[noreturn]] inline void reportOom(const char* what, size_t count, size_t elemBytes,
                                   const char* flags) {
    const double bytes = (double)count * (double)elemBytes;
    const std::string note = memAdvice(bytes);
    char buf[1536];
    std::snprintf(buf, sizeof(buf),
                  "out of memory allocating %s: %zu x %zu B = %s.%s\n"
                  "       Lower %s and re-run. (Host RAM, not VRAM — this buffer lives on "
                  "the CPU side\n       even when the trace ran on the GPU.)",
                  what, count, elemBytes, humanBytes(bytes).c_str(), note.c_str(), flags);
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
