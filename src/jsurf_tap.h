// -checkjsurfw (known-issues JCHROMA-JSURF-WEAK): the device's record of mode J's point-merge
// weights, in plain types both halves read.
//
// A point merge's MIS weight is DSurfMergeW on the device and SurfMergeWeight on the host. Where
// a medium's extinction varies with wavelength its camera half carries two rows of JSpec's bins
// (UPBP-W-CHROMATIC: segSumM and the merge at eye[k-1], read at the photon's wavelength), built
// along the camera subpath by DChromaRun and cameraSideSumsChroma. No frame-level check can hold
// them: where they matter, beams and point merges competing for one path, the point merges carry
// well under a seed's scatter of the image. So the probe holds them directly. kBdptT records the
// camera subpaths of a render's first samples and the camera half it built at each point-merge
// site, bins included; the host rebuilds the same halves from the same vertices with its own
// code, and both evaluate the weight at chosen photon states and wavelengths (main.cpp
// checkJSurfTap).
//
// Dependency-free so the two sides share one definition (lattice_probe.h's reasoning).
#pragma once

inline constexpr int kJSurfTapK = 24;      // JSpec::K and DJSPEC_K, asserted on each side
inline constexpr int kJSurfTapMaxV = 67;   // vertices kept a sample: the deep kernel's MAXV

// An eye vertex: what the camera half reads of it. `type` numbers as VType and BV_* do (camera,
// light, surface, medium); `matId` indexes Scene::mats.
struct JSurfTapVertex {
    double p[3], ns[3], ng[3];
    double pdfFwd, pdfRev, beta;
    double fiberR;           // a strand's radius (0 off one): the hit's fiberRadius on the host
    int type, delta, matId, mediumId;
};

// The camera half of a point merge's weight at eye[camVert], as kBdptT built it (DSurfMergeW).
// `chroma`: the bins hold DChromaRun's mPrev (segSumM at row k-1) and e1 (the merge at eye[k-1]).
struct JSurfTapHalf {
    double kappaB, kappaS, invPdfFwdK, gateD1, cosPrev, invDistPrev2, invPdfFwdKm1, etaKm1Coef;
    double segSumC, segSumM;
    double mPrev[kJSurfTapK], e1[kJSurfTapK];
    int camVert;             // k; 0 where no merge was weighed at this vertex
    int maxDepth;
    int chroma;
    int pad_;
};

// A photon's half of the weight (SurfMis's fields), the camera vertex's two BSDF densities toward
// and from it, and its wavelength: one evaluation of a half (`half`, an index into the halves
// handed to cudaJSurfWEval).
struct JSurfTapState {
    double sumC, sumMb, sumMs, pdfFwdA;
    double pdfDirRev, pdfDirFwd, lam;
    float rCoef;
    int gateC1, vert, half;
};
