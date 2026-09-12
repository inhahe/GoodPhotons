// Human-readable durations: `[[[dd:]hh:]mm:]ss`.
//
// Every progress line in ftrace used to print elapsed time and ETAs as a raw count of
// seconds ("%.1fs"), which is fine for the first minute and unreadable after it: a
// gallery_rain gather reports "~111612s left", and nobody reads that as "1 day 7 hours".
// A renderer whose long runs are its NORMAL runs cannot make the user do that arithmetic.
//
// The format is the familiar clock one, most-significant unit first, leading unit
// unpadded and every following unit zero-padded to two digits:
//
//     42.30s        under a minute — keep the sub-second precision, it is the regime
//     9.5s          where it matters (a BVH build, one repaint)
//     1:45          one minute forty-five
//     3:20:05       three hours
//     1:07:00:12    one day seven hours
//
// Under a minute the unit suffix `s` is kept, because a bare "42" in the middle of a
// status line reads as a count of something rather than a time. From a minute up the
// colons carry that meaning themselves, so no suffix is needed (and appending one would
// wrongly suggest the whole "1:45" is a number of seconds).
#pragma once
#include <cstdio>
#include <string>

// `sec` seconds -> "[[[dd:]hh:]mm:]ss". Negative and NaN inputs clamp to 0.
inline std::string humanDur(double sec) {
    char b[64];
    if (!(sec > 0.0)) return "0.00s";          // also catches NaN (all comparisons false)
    if (sec < 60.0) {
        // Sub-minute is where fractions are the whole point (a slice, a repaint, a small
        // BVH), so keep two digits of precision below ten seconds and one above it.
        std::snprintf(b, sizeof b, sec < 10.0 ? "%.2fs" : "%.1fs", sec);
        return b;
    }
    const long long t = (long long)(sec + 0.5);
    const long long s = t % 60, m = (t / 60) % 60, h = (t / 3600) % 24, d = t / 86400;
    if (d)      std::snprintf(b, sizeof b, "%lld:%02lld:%02lld:%02lld", d, h, m, s);
    else if (h) std::snprintf(b, sizeof b, "%lld:%02lld:%02lld", h, m, s);
    else        std::snprintf(b, sizeof b, "%lld:%02lld", m, s);
    return b;
}
