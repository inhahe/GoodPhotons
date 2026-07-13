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

## 1. GPU instancing expands instances to world tris — ⬜
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

## 2. Mode P not progressive; R/D no disk resume — ⬜
**Source:** `known-issues.md` → Tech debt.
Mode `P` (composite) doesn't do progressive output; modes `R`/`D` have no `.ftbuf`-style
disk resume (forward modes A/B/C already checkpoint/resume). 
**Done when:** `P` produces periodic progressive writes like A/B/C, and `R`/`D` write a
resumable checkpoint sidecar honored by `-resume`.

## 3. Photon map / view-independent radiance cache (keystone) — ⬜
**Source:** `ROADMAP.md` item (1).
Stored, view-independent spatial structure of photon records queried by a backward
camera final-gather pass — computes light transport once and reuses it across pixels and
across many camera frames of a static scene. Uniform hash grid (CUDA-friendly). New mode
(e.g. `-mode M` "photon-mapped final gather"). Unlocks PPM/VCM's merging term (already
built as S/U, but those bake their own maps; this is the shared reusable cache) and the
cross-camera flythrough win.
**Note:** modes `S` (SPPM) and `U` (VCM) already exist and internally build hash grids;
reuse/refactor that machinery rather than duplicating it.
**Done when:** `-mode M` matches mode `R` on a diffuse Cornell box (equal-quality RMSE),
and the built map is reused across a multi-camera flythrough (build once, gather per frame).

---

## Decisions needed from the user
_(none yet — will append here if a fork needs a human call; work continues on other items meanwhile)_

## Progress log
- 2026-07-13: roadmap created; starting item 0.
- 2026-07-13: item 0 closed as NOT-A-BUG (see above). Cylinder light verified correct on
  CPU (R) and GPU (B) via controlled absolute-power tests. Docs + repro scene updated.
  Starting item 1 (GPU two-level BVH for instancing).
