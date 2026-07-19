// Parametric records — a named FTSL data structure (see ROADMAP_records.md).
//
// A record is a bank of per-channel look-up tables over a shared scalar domain
// [lo,hi]. A per-hit driver scalar samples every channel at once; each channel whose
// name matches a real material slot fills that slot at the driven value. Channel
// names are arbitrary (name-by-destination): a name that matches no slot is simply
// not auto-bound (still addressable by dot), and a slot with no channel is ignored.
//
// STAGE 1: the STRUCTURAL model — channels, their stops, and each stop's
// redistributed domain position + raw value token.
// STAGE 2 (this file): stop COMPILATION + SAMPLING — a scalar channel's stops carry
// compiled pattern programs (evaluated per-hit), a colour channel's stops carry
// resolved reflectance spectra + their linear-RGB; recSampleScalar/recSampleSpectrum
// index a channel at a driver value with nearest / linear / smooth (Fritsch-Carlson
// monotone cubic) interpolation. Driver binding into material slots lands in stage 3.
#pragma once
#include <string>
#include <vector>
#include <cmath>
#include "pattern.h"    // PatNode / PatCtx / patternEval — scalar stop programs
#include "spectrum.h"   // Spectrum + Vec3 (via color.h) — colour stops
#include "upsample.h"   // rgbToReflectanceJH / reflectanceToLinearSrgbD65 — colour lerp

enum class RecInterp { Nearest, Linear, Smooth };
enum class ChanKind  { Scalar, Spectrum };   // a scalar LUT vs a colour LUT

// One stop in a channel LUT.
struct RecStop {
    double      pos    = 0.0;    // domain position in [lo,hi] after redistribution
    bool        pinned = false;  // author gave an explicit p:<pos> prefix
    std::string token;           // raw value token (number / expression / spectrum:ref)
    // --- compiled forms (stage 2) ---
    std::vector<PatNode> expr;   // Scalar channel: program evaluated per-hit (Const for a literal)
    Spectrum             color;  // Spectrum channel: resolved reflectance
    Vec3                 rgb{0, 0, 0}; // Spectrum channel: precomputed linear-RGB for interpolation
};

// One channel: a named LUT (its name is matched to a material slot at bind time).
struct RecChannel {
    std::string          name;
    ChanKind             kind = ChanKind::Scalar;
    std::vector<RecStop> stops;  // author order == ascending pos after redistribution
};

struct Record {
    std::string              name;
    double                   lo = 0.0, hi = 1.0;
    RecInterp                interp = RecInterp::Linear;
    std::vector<RecChannel>  channels;

    int channelIndex(const std::string& n) const {
        for (size_t i = 0; i < channels.size(); ++i)
            if (channels[i].name == n) return (int)i;
        return -1;
    }
};

// ---- sampling ---------------------------------------------------------------
// Locate the interval [i, i+1] of `ch.stops` bracketing driver `d` (already clamped
// to the stop-position range). Returns the left index i in [0, n-2] and the local
// parameter t in [0,1] across that interval. Coincident stops (pos gap ~0) give t=0.
inline int recLocate(const RecChannel& ch, double d, double& t) {
    const int n = (int)ch.stops.size();
    int i = 0;
    while (i < n - 2 && d > ch.stops[i + 1].pos) ++i;
    double p0 = ch.stops[i].pos, p1 = ch.stops[i + 1].pos;
    double span = p1 - p0;
    t = (span > 1e-12) ? (d - p0) / span : 0.0;
    if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
    return i;
}

// Fritsch-Carlson monotone cubic Hermite tangent for interior/endpoint node k given
// per-interval secant slopes. `sec[j]` is the slope of interval [j, j+1] (length
// n-1). Guarantees no overshoot / monotonicity-preservation.
inline double recFCTangent(const double* pos, const double* val, const double* sec,
                           int n, int k) {
    if (k == 0)     return sec[0];
    if (k == n - 1) return sec[n - 2];
    double s0 = sec[k - 1], s1 = sec[k];
    if (s0 * s1 <= 0.0) return 0.0;                 // local extremum -> flat, no overshoot
    // weighted harmonic mean (Fritsch-Carlson 1980, eq. via interval widths)
    double h0 = pos[k]     - pos[k - 1];
    double h1 = pos[k + 1] - pos[k];
    double w0 = 2.0 * h1 + h0, w1 = h1 + 2.0 * h0;
    return (w0 + w1) / (w0 / s0 + w1 / s1);
}

// Sample a Scalar channel at driver `d`, evaluating each stop's expression against
// `ctx` first, then interpolating the resulting scalars per the record's interp mode.
inline double recSampleScalar(const Record& rec, const RecChannel& ch,
                              double d, const PatCtx& ctx) {
    const int n = (int)ch.stops.size();
    auto stopVal = [&](int i) {
        const std::vector<PatNode>& e = ch.stops[i].expr;
        return e.empty() ? 0.0 : patternEval(e.data(), (int)e.size(), ctx);
    };
    if (n == 0) return 0.0;
    if (n == 1) return stopVal(0);
    // clamp to the stop-position range (== [lo,hi] since ends anchor there unless pinned)
    double lo = ch.stops[0].pos, hi = ch.stops[n - 1].pos;
    if (d < lo) d = lo; else if (d > hi) d = hi;

    if (rec.interp == RecInterp::Nearest) {
        double t; int i = recLocate(ch, d, t);
        return stopVal(t < 0.5 ? i : i + 1);
    }
    double t; int i = recLocate(ch, d, t);
    double v0 = stopVal(i), v1 = stopVal(i + 1);
    if (rec.interp == RecInterp::Linear) return v0 + (v1 - v0) * t;

    // Smooth: monotone cubic Hermite. Evaluate all stop values + secants once.
    double vs[64], ps[64], sec[64];
    int m = n < 64 ? n : 64;
    for (int k = 0; k < m; ++k) { vs[k] = stopVal(k); ps[k] = ch.stops[k].pos; }
    for (int k = 0; k < m - 1; ++k) {
        double h = ps[k + 1] - ps[k];
        sec[k] = (h > 1e-12) ? (vs[k + 1] - vs[k]) / h : 0.0;
    }
    if (i > m - 2) i = m - 2;
    double mk  = recFCTangent(ps, vs, sec, m, i);
    double mk1 = recFCTangent(ps, vs, sec, m, i + 1);
    double h   = ps[i + 1] - ps[i];
    double t2 = t * t, t3 = t2 * t;
    double h00 =  2 * t3 - 3 * t2 + 1;
    double h10 =      t3 - 2 * t2 + t;
    double h01 = -2 * t3 + 3 * t2;
    double h11 =      t3 -     t2;
    return h00 * vs[i] + h10 * h * mk + h01 * vs[i + 1] + h11 * h * mk1;
}

// Sample a Spectrum channel at driver `d`: interpolate the stops' precomputed linear
// RGB, then Jakob-Hanika upsample the result back to a reflectance spectrum. Nearest
// returns the picked stop's spectrum verbatim (no re-upsample round-trip).
inline Spectrum recSampleSpectrum(const Record& rec, const RecChannel& ch, double d) {
    const int n = (int)ch.stops.size();
    if (n == 0) return constantSpectrum(0.0);
    if (n == 1) return ch.stops[0].color;
    double lo = ch.stops[0].pos, hi = ch.stops[n - 1].pos;
    if (d < lo) d = lo; else if (d > hi) d = hi;

    if (rec.interp == RecInterp::Nearest) {
        double t; int i = recLocate(ch, d, t);
        return ch.stops[t < 0.5 ? i : i + 1].color;
    }
    double t; int i = recLocate(ch, d, t);
    const Vec3& a = ch.stops[i].rgb;
    const Vec3& b = ch.stops[i + 1].rgb;
    double w = t;
    if (rec.interp == RecInterp::Smooth) w = t * t * (3.0 - 2.0 * t);   // smoothstep on RGB
    Vec3 c{ a.x + (b.x - a.x) * w, a.y + (b.y - a.y) * w, a.z + (b.z - a.z) * w };
    return rgbToReflectanceJH(c.x, c.y, c.z);
}
