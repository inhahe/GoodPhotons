// N4a — the shared layout of the mode-W deterministic-lattice bit-exactness probe.
//
// Mode W has no Monte-Carlo noise for a CPU/GPU disagreement to hide behind, so any
// difference in its sample lattices is a *visible deterministic* difference — a stricter
// bar than every stochastic port in §M, despite the simpler logic. Whole-image
// bit-exactness is nevertheless not achievable (the device's `Real` is fp32 by default,
// `RAY_EPS` is 1e-4f there vs 1e-6 on the host, and CUDA libdevice's transcendentals differ
// from the MSVC CRT's in the last places), so N4 splits acceptance in two:
//
//   (a) BIT-EXACT on the lattice helpers — they are pure integer-and-`double` arithmetic on
//       both sides, with no transcendentals and no `Real`, so they genuinely can be, and
//       must be, bit-identical. That is what this probe tests (`ftrace -checklattice`).
//   (b) image agreement to fp32 tolerance with no structural difference — scraps/n3_check.py
//       and scraps/n3b_check.py.
//
// This header is deliberately dependency-free so BOTH sides can share one definition of the
// column layout: main.cpp (host, always) and render_cuda.cu (device, CUDA builds only). A
// re-declaration on either side would let the two drift, which is precisely the class of bug
// the probe exists to catch.
//
// One row per index, `kLatticeProbeCols` doubles wide:
//
//    col  0        radicalInverse2(i)
//    col  1        rot05(radicalInverse2(i))            — rot05 on a non-trivial argument
//    col  2, 3     whittedSample(i)      -> u, v        — the subpixel lattice
//    col  4        whittedLambdaU(i)                    — the wavelength lattice
//    col  5        whittedOrderU(i, i&3)                — discrete choice (grating order)
//    col  6        whittedFluoroU(i, i&3)               — Stokes-shift excitation λ
//    col  7, 8     whittedGlossyUV(i, i&3) -> u1, u2    — the rough-specular lobe lattice
//    col  9,10     giPhases(i)           -> p1, p2      — the one-bounce-gather CP rotation
//    col 11,12     gridUV(i % (G*G), G), G = 4 + i%5    — the area-light shadow lattice.
//                                                         The device hands this back through
//                                                         `Real`, so the host narrows to
//                                                         cudaRealBytes() before comparing;
//                                                         every other column is double on
//                                                         both sides and compares unadjusted.
//    col 13..32    radicalInverseScr(b, i) for each b in kLatticeProbeBases
#pragma once

inline constexpr int kLatticeProbeNBases = 20;
inline constexpr int kLatticeProbeCols   = 13 + kLatticeProbeNBases;

// Every base any mode-W lattice actually uses: 3 (subpixel v), 5 (λ), 7/11 (gather phases),
// 13..41 (glossy, 4 bounce depths x 2), 43..59 (discrete choice), 61..73 (fluorescence).
// Base 2 has its own bit-reversal fast path and is column 0, not one of these.
inline constexpr unsigned kLatticeProbeBases[kLatticeProbeNBases] = {
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73
};
