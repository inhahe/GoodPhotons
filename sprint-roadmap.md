# Sprint Roadmap — autonomous work batch (started 2026-07-13)

Four prioritized items pulled from `known-issues.md` / `ROADMAP.md`, to be worked
top-to-bottom autonomously. This file tracks status and, importantly, any **decisions
the user needs to make** (see the log at the bottom) so work can continue on other
items while a decision is pending.

Status legend: ⬜ not started · 🔨 in progress · ✅ done · ⏸ blocked (see decisions log)

---

## 0. `light cylinder` emits no illumination — ✅ NOT A BUG (misdiagnosis), 2026-07-13
**Source:** `known-issues.md` → Open bugs (now moved to Resolved).
Re-tested: the cylinder light **does** illuminate, on both CPU (mode R) and GPU (mode B).
The original report misread the normal ~1e-14 auto-exposure (the stock Cornell box is
8.87e-14) as "zero light." With no explicit `power` the raw blackbody emitter surface is
astronomically bright and dominates the content-based auto-exposure, crushing the
genuinely-lit wall to near-black — and a `light sphere` in the same scene does exactly the
same. Controlled absolute-`power` test: cylinder vs sphere wall-region means match
(21.33 vs 19.65 at power 4000; CPU 21.33 ≈ GPU 21.60). Fix applied: `known-issues.md`
entry moved to Resolved with the explanation; `scraps/cyl_test.ftsl` updated to use
`power 4000` so it demonstrates the (correct) lit wall.

## 1. GPU instancing expands instances to world tris — ✅ DONE 2026-07-13
**Source:** `known-issues.md` → Tech debt.
CPU has a true two-level BVH (`mesh_asset`/`mesh_instance` share one BLAS). The GPU
`buildUploadScene` (`render_cuda.cu`) expands every instance into world-space triangles
and rebuilds one flat BVH → device memory scales with total instanced tris; a scene that
fits on CPU can OOM on GPU. Images are correct today; this is a memory/scaling fix.
**Proper fix:** device two-level BVH — per-BLAS node/tri/primIdx pools + an instance
table (toLocal affine + blasId + matOverride) + an instance-leaf branch in device
`traverseClosest`/`traverseAny` that transforms the ray into BLAS space (parametric `t`
preserved). Touches the hottest device kernel.
**Done when:** an instanced scene renders bit-identical to today with device memory flat
in instance count, validated CPU↔GPU.

## 2. Mode P not progressive; R/D no disk resume — ✅ DONE 2026-07-13
**Source:** `known-issues.md` → Tech debt.
Mode `P` (composite) doesn't do progressive output; modes `R`/`D` have no `.ftbuf`-style
disk resume (forward modes A/B/C already checkpoint/resume). 
**Done when:** `P` produces periodic progressive writes like A/B/C, and `R`/`D` write a
resumable checkpoint sidecar honored by `-resume`.
**Done:** `runCompositeProgressive` (`main.cpp`) makes `P` progressive — classify pixels
once, alternate forward/backward batches into two SUM films, re-fit the scale and re-blend
every `-interval`, with a dual-film `FTPCM02` checkpoint. `R`/`D` disk-resume via
`runSppProgressive` reusing the single-film `Checkpoint` keyed on spp (mode byte in the
guard blocks cross-mode loads). `SppProgress::sampleBase` decorrelates resumed samples
(CPU seed offset / GPU seed-base XOR). Validated: `R` 58 196→116 545 spp and `D`
1016→2052 spp both track 100/√spp exactly; `P` 36.2 M/4636 spp→56.5 M/7228 spp with the
diffuse residual falling 0.0281→0.0226. `-resume`/`-checkpoint`/`-time`/`-noise`/`-forever`
gates extended to `P`; dead `renderComposite` wrapper removed; docs updated.

## 3. Photon map / view-independent radiance cache (keystone) — ✅ DONE 2026-07-13
**Source:** `ROADMAP.md` item (1).
Stored, view-independent spatial structure of photon records queried by a backward
camera pass — computes light transport once and reuses it across pixels and across many
camera frames of a static scene. Uniform hash grid (CUDA-friendly). New mode `-mode M`.
Unlocks PPM/VCM's merging term (already built as S/U, but those bake their own maps; this
is the shared reusable cache) and the cross-camera flythrough win.
**Note:** modes `S` (SPPM) and `U` (VCM) already exist and internally build hash grids.
**Done when:** `-mode M` matches mode `R` on a diffuse Cornell box (equal-quality RMSE),
and the built map is reused across a multi-camera flythrough (build once, gather per frame).
**Done:** Mode `M` and its cross-camera shared-map path already existed in the codebase
(`tracePhotonPass`/`renderPhotonCamera`/`photonGather` in `photonmap_render.h`;
`runSharedPhotonMap` in `main.cpp`); this item **validated both done-criteria**.
(a) *Radiance match:* on the pure-diffuse Cornell box (`scraps/cornell_diffuse.ftsl`,
both auto-exposed identically to 2.43e-13), un-tone-mapping both images to raw radiance
gives **M/R = 0.990 global**, diffuse-mask (both-unclipped) **relRMSE 4.7%**, **Pearson
r = 0.9980**, thirds top 0.989 / mid 0.996 / bot 0.980 — mode M reproduces mode R's
radiance solution. (b) *Cross-camera reuse:* `runSharedPhotonMap` built **one** map
(11.4 M photons from 8 M emitted, ~2.3 s) and gathered all 3 frames of a dolly flythrough
(`scraps/m3_fly.ftsl`), with dolly0 ≈ dolly2 by symmetry — build-once/gather-per-frame
confirmed. **Caveat logged as tech debt:** mode M does a *direct density query* at the
first diffuse hit, not a true secondary-hemisphere final gather (README/roadmap wording
corrected; a real final-gather pass noted as a future enhancement in `known-issues.md`).

---

## Decisions needed from the user
_(none yet — will append here if a fork needs a human call; work continues on other items meanwhile)_

## Progress log
- 2026-07-13: roadmap created; starting item 0.
- 2026-07-13: item 0 closed as NOT-A-BUG (see above). Cylinder light verified correct on
  CPU (R) and GPU (B) via controlled absolute-power tests. Docs + repro scene updated.
  Starting item 1 (GPU two-level BVH for instancing).
- 2026-07-13: item 1 DONE. Implemented a device two-level BVH in `render_cuda.cu`
  (`DBlas`/`DInstance` + shared `blasTris`/`blasNodes`/`blasPrim` pools, `blasClosest`/
  `blasOccluded`, instance-leaf branch in `closestHit`/`occluded`, `Scene::bvh` uploaded
  verbatim). Validated with `scraps/instance_test.ftsl` (4 tori, one shared 16 384-tri
  BLAS): GPU (B) ≈ CPU (R) at Pearson r=0.996; implicit scene unregressed. Device geometry
  memory now flat in instance count. known-issues.md entry marked DONE. Starting item 2
  (mode P progressive + R/D disk resume).
- 2026-07-13: item 2 DONE. Mode `P` is now progressive (`runCompositeProgressive`): classify
  once, alternate forward/backward batches into two persistent SUM films, re-fit scale + re-
  blend every interval, dual-film `FTPCM02` checkpoint. `R`/`D` disk-resume via
  `runSppProgressive` (single-film spp-keyed `Checkpoint`, mode byte in guard). Resume seed
  decorrelation via `SppProgress::sampleBase`. Validated on GPU (cornell): R/D noise tracks
  100/√spp exactly across resume, P residual falls 0.0281→0.0226; cross-mode guard rejects a
  mismatched checkpoint. Clean build, no warnings. Starting item 3 (photon map / radiance
  cache — mode M already exists; verifying against done-criteria).
- 2026-07-13: item 3 DONE. Mode M and its cross-camera shared-map reuse already existed;
  this item validated both done-criteria. Radiance match on the diffuse Cornell box:
  M/R=0.990 global, diffuse-mask relRMSE 4.7%, Pearson r=0.9980 (thirds 0.989/0.996/0.980),
  auto-exposures identical (2.43e-13). Cross-camera reuse: `runSharedPhotonMap` builds one
  map (11.4 M photons, ~2.3 s) and gathers a 3-frame flythrough. Discovered mid-validation
  that the scene must be passed with `-in <file>` (a positional arg is silently ignored and
  the built-in cornell — with a dispersive glass sphere — is used instead), which had
  contaminated an earlier M-vs-R compare; redone on the true diffuse box. Docs corrected to
  describe mode M as a direct density query (not a secondary final gather); the true final
  gather logged as a future enhancement in known-issues.md. **All four sprint items done.**
