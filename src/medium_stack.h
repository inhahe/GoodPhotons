// Per-path medium stack for nested-dielectric PRIORITY resolution (Schmidt & Budge 2002).
//
// As a ray/photon path enters and leaves dielectric solids it carries this small stack of
// the media it is currently inside — each entry is (material index, priority). The
// "current medium" (what the ray is optically travelling through) is the highest-priority
// entry; where solids overlap, the higher priority wins, so an inner LOWER-priority
// surface is suppressed (passed straight through) while a higher-priority one refracts.
//
// SAFE FALLBACK: the priority rule is only applied when BOTH sides of an interface carry
// an EXPLICIT priority. If either is unset (INT_MIN), the interface is treated exactly as
// the old flat model did — a real air<->glass interface — so scenes that never mention
// `priority` render bit-identically to before. Those are precisely the cases the
// ahead-of-time audit (priority_audit.h) warns about.
#pragma once
#include <climits>

struct MediumStack {
    static constexpr int CAP = 32;   // deep dielectric nesting is rare; overflow degrades gracefully
    int matIdx[CAP];
    int pri[CAP];
    int n = 0;

    bool empty() const { return n == 0; }

    // Priority of the current (highest-priority) medium, or INT_MIN if the ray is in air.
    int topPri() const {
        int bp = INT_MIN;
        for (int i = 0; i < n; ++i) if (pri[i] > bp) bp = pri[i];
        return n ? bp : INT_MIN;
    }
    // Material index of the current medium (highest priority; last-pushed breaks ties),
    // or -1 if the ray is in air/vacuum.
    int topMat() const {
        int bp = INT_MIN, bm = -1;
        for (int i = 0; i < n; ++i) if (pri[i] >= bp) { bp = pri[i]; bm = matIdx[i]; }
        return bm;
    }
    void push(int mi, int p) { if (n < CAP) { matIdx[n] = mi; pri[n] = p; ++n; } }
    // Remove one entry whose material index == mi (the solid the ray is exiting).
    void popMat(int mi) {
        for (int i = n - 1; i >= 0; --i)
            if (matIdx[i] == mi) {
                for (int j = i; j < n - 1; ++j) { matIdx[j] = matIdx[j + 1]; pri[j] = pri[j + 1]; }
                --n; return;
            }
    }
};
