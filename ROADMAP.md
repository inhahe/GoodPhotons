# Roadmap

Planned rendering features, ordered by how they build on each other. This file is
forward-looking (what we intend to build); `known-issues.md` tracks bugs/tech-debt and
`README.md` documents what already ships. Mark items **DONE** (with a date) as they land
and migrate the user-facing description into `README.md`.

---

## What exists today (the baseline these build on)

Being precise, because the names overlap and it's easy to assume we have more than we do:

- **Forward light tracing / photon *splatting*** — modes A/B/C. Photons are shot from the
  lights; at each surface/emitter/volume vertex the path is **next-event connected to the
  camera(s) and splatted** onto the film. A photon's flight is camera-independent until the
  splat, so one photon flight can splat to **all cameras at once** (the shared multi-camera
  pass, `main.cpp:1145`). **There is no stored photon map** — nothing is deposited into a
  queryable spatial structure; energy goes straight onto the sensor. This is the
  light-tracing half of BDPT, *not* Jensen-style photon mapping.
- **Backward path tracer** — mode R. Camera-origin rays, unbiased, best specular/GI quality;
  **shares no work across cameras** (each frame re-traces from scratch).
- **BDPT** — mode D (`bdpt.h`). Full bidirectional **vertex connection** with balance-heuristic
  MIS (Veach / PBRT-v3 structure). Has the connection strategies and the MIS machinery, but
  **no vertex merging** (no density-estimation sampling technique).
- **Composite** — mode P (forward light tracing + a camera-side path).
- **Participating media** — homogeneous + heterogeneous (procedural `density` expression /
  pattern fields), Henyey–Greenstein phase, delta/ratio tracking, box/sphere/implicit bounds,
  multi-medium superposition. Full on forward (A/B/C) + BDPT (D); **global-homogeneous only**
  on backward (R). Plus Beer–Lambert absorption inside dielectrics.

**Triangle meshes already exist** (correction): `src/mesh.h` + `src/bvh.h` + `Tri` (geometry.h)
give a working OBJ loader, a binned-SAH BVH, Möller–Trumbore intersection on **both CPU and GPU**,
per-vertex UVs, per-face `usemtl` materials, and transforms — meshes render in **all** modes
(A/B/C/R/D/P). What the mesh path *lacks* is **smooth per-vertex normals** (OBJ `vn` is dropped, so
everything is flat-shaded — the biggest quality gap), plus glTF, instancing, emissive triangles,
and normal maps. Item (5) is therefore scoped to those gaps, not a from-scratch primitive.

**So we do NOT yet have:** a stored photon map, progressive photon mapping (PPM/SPPM), VCM/UPS
merging, external volume-asset (VDB) import, or **smooth-shaded / glTF meshes**. Those are the
items below.

### Dependency graph

```
        ┌──────────────────────────────────────────┐
        │  (1) Photon map / view-independent cache   │   ← keystone
        └──────────────────────────────────────────┘
             │                         │
             ▼                         ▼
   (2) Progressive PM (PPM/SPPM)   (3) VCM / UPS
                                   (needs BDPT ✔ + merging from (1))

   (4) .vdb / NanoVDB volumes    ── independent, can land anytime
   (5) Triangle-mesh primitive   ── independent, can land anytime
```

Item **(1) is the keystone**: it's the data structure that unlocks both progressive photon
mapping and VCM's merging term, *and* it's the direct answer to "share GI work across hundreds
of cameras while keeping backward-tracer quality." Build it first.

---

## (1) Photon map / view-independent radiance cache

**Goal.** A stored, **view-independent** spatial structure of photon records that a backward
camera pass can query by radius (density estimation) — so light transport is computed once and
reused, both across pixels within a frame and **across many camera frames of a static scene**
(exactly the flythrough case: scene fixed, only the camera moves).

**Why.** Three payoffs from one structure:
1. **Share work across cameras with R-grade quality.** Build the map once; each camera does a
   cheap backward **final-gather** pass against it instead of re-tracing full GI. This is the
   "sharing *plus* backward specular quality" combination that plain photon splatting (A/B)
   can't give and mode R doesn't share.
2. Foundation for **PPM/SPPM** (item 2).
3. Foundation for **VCM merging** (item 3).

**Builds on.** The forward photon tracer already *generates* the photon flights — today it
splats them; here it **also (or instead) deposits a record at each diffuse/volume vertex**.
The backward tracer (R) already provides the camera pass we'll bolt final-gather onto.

**Design sketch.**
- **Photon record:** world position, power/throughput (spectral — single-wavelength per photon
  in this renderer, so store λ + monochromatic power), incident direction, surface/volume flag.
- **Acceleration structure:** a **uniform hash grid** (cell size ≈ query radius), *not* a
  kd-tree — the hash grid is far friendlier to the CUDA backend (flat arrays, sortable by cell,
  no pointer chasing) and is the standard choice for GPU PPM/VCM. Build = bin photons into cells,
  sort/counting-sort by cell id, prefix-sum offsets.
- **Radiance estimate:** at a camera-ray diffuse hit, gather photons within radius *r*, weight by
  the BSDF and a kernel (constant or a smoothing kernel), divide by π r² × N_photons.
- **Volumes:** a beam/point density estimate for the medium (photon-in-volume records) gives
  volumetric caustics later; can be a follow-up.
- **New mode:** e.g. `-mode M` ("photon-mapped final gather"): forward photon pass → backward
  camera pass with one-bounce final gather into the map.

**Steps.**
1. Photon-record struct + a "deposit" path in `tracePhoton` (gated so A/B/C splat behavior is
   unchanged when the map is off).
2. Hash-grid build on CPU; then mirror to CUDA (`render_cuda.cu`).
3. Range query + radiance estimate; validate against mode R on a diffuse Cornell box (equal-time
   / equal-quality RMSE).
4. Backward final-gather camera pass; wire `-mode M`.
5. **Cross-camera reuse:** keep the built map resident and run the shared multi-camera final-gather
   over all flythrough frames (build once, gather per frame). This is the concrete flythrough win.

**Open questions.** Spectral density estimation (photons are monochromatic here — gather across a
wavelength band or per-λ maps?); radius selection (fixed vs adaptive); how to keep the direct +
specular components sharp (final gather only the *indirect* diffuse, trace direct + specular normally).

---

## (2) Progressive photon mapping (PPM / SPPM) — **DONE 2026-07-12** (mode `S`)

**Shipped** as `-mode S` (`src/sppm_render.h`): repeated bounded photon passes with a
per-pixel shrinking gather radius (Hachisuka 2008/2009 shared-statistics form). Per-pixel
`SPPMPixel` state (tau/radius/nAcc/directSum + a re-sampled visible point) lives across
passes; each pass re-traces camera visible points (stochastic PPM), traces `-n` photons
into a fresh bounded `PhotonMap`, gathers at the current radius, and applies the
`R'² = R²(N+αM)/(N+M)` / flux-rescale update. `-n` = photons per pass, `-spp` = pass
count (or a `-time`/`-noise`/`-forever` budget), `-sppmalpha` = shrink rate (default 0.7),
initial radius from `-pmradius`/`-pmradiusfrac`. A single pass reduces algebraically to
mode `M` (verified). Plugs into the existing progressive driver (`runSppProgressive`) by
reporting `L·passes` so the divide-by-sppDone recovers the resolved radiance. CPU only.

**Original plan below.**

**Goal.** Converge the photon estimate without unbounded memory and without the bias of a fixed
radius: run repeated photon passes and **shrink the density-estimation radius** over iterations
(Hachisuka et al. 2008). **SPPM** (stochastic PPM) also draws fresh camera samples each pass, so
it's robust for distributed effects (DOF, motion, glossy) and converges to the correct result.

**Why.** Unbiased-in-the-limit convergence with bounded memory, and it **nails caustics and
SDS paths** — the specular gold / glass focusing that pure path tracing (R) and even BDPT
resolve slowly. Progressive = usable preview that keeps refining.

**Builds on.** Item (1)'s photon map + a **progressive radius schedule** and **per-pixel
statistics** (accumulated flux, current radius R_i, accumulated photon count N_i; the classic
R_{i+1}² = R_i² (N_i + αM)/(N_i + M) update).

**Steps.**
1. Per-pixel PPM state buffers (flux, radius, photon count).
2. Iteration loop: photon pass → gather → radius/flux update → repeat, with live progressive
   output (fits the existing `-window` / `-interval` progressive infra).
3. SPPM variant: re-sample camera subpaths each pass (shares the item-1 gather).
4. Validate caustic convergence on a glass-sphere / metal scene vs a long mode-R reference.

**Open questions.** α (radius-shrink rate) default; shared vs per-pixel radius; interaction with
the checkpoint/resume system (forward modes only today).

---

## (3) VCM / UPS (Vertex Connection and Merging / Unified Path Sampling) — ✅ DONE (mode `U`)

**Status.** Implemented as render mode `U` in `src/vcm.h` (wired into `main.cpp`; CLI `-vcmalpha`,
default `0.75`). Each pass traces a light subpath + a camera subpath per pixel and combines all
BDPT connection strategies (emission, NEE, camera↔paired-light-vertex connection, connect-to-camera
splat) with SPPM photon **merging** under one SmallVCM-style balance-heuristic weight (dVCM/dVC/dVM
partial-MIS recursion). Progressive merge-radius shrink `r_i = R0·i^((alpha-1)/2)`; unbiased in the
limit; CPU-threaded (per-thread light-vertex/splat buffers, a counting-sort hash grid rebuilt each
pass). Single-wavelength handling: connections pair a camera path with its **own** light path
(shared λ → exact); merges use the standard spectral-photon-mapping XYZ estimate (MIS pdfs are
wavelength-independent so the balance weights stay a valid partition of unity).

**Validated.** Absolute-exposure Cornell boxes (`scraps/cornell_diffuse_abs.ftsl`,
`scraps/cornell_caustic_abs.ftsl`): mode `U` matches the mode `R` ground truth in absolute scale
(diffuse mean ratio ≈1.009; caustic mean ratio ≈1.003 — unbiased) and, at equal wall-clock time,
has **lower** RMSE-vs-`R` than SPPM both overall (3.93 vs 5.97) and in the caustic region
(3.26 vs 5.31). Fixed a latent bug found during validation: built-in (`-scene`, non-`-in`) scenes
never built a camera for modes `M`/`S`/`U` (they were absent from the `useCamera` list), leaving a
zero camera — now included.

**Goal.** Combine **BDPT vertex connections** (we have these — mode D) with **photon-map vertex
merging** (density estimation reinterpreted as an extra *sampling technique*), all weighted
together under **MIS** (Georgiev et al. 2012 "Light Transport Simulation with Vertex Connection
and Merging"; Hachisuka et al. "Unifying Points, Beams, and Paths in Volumetric Light Transport"
/ UPS). The single most robust unbiased estimator: connections handle what merging is bad at
(and vice-versa), MIS picks the best per path.

**Why.** Each technique covers the others' weak spots — diffuse GI, glossy interreflection,
caustics, and SDS all in one estimator. This is the "have it all" mode.

**Builds on.** **We already have the connection half** (BDPT + its MIS `MISWeight`) and **item (1)
gives the merging half** (the photon map). VCM is largely *gluing these together*: add merging
vertices as a strategy and extend the MIS weights to include the **merge densities** alongside the
connection densities (the delicate part — the "vertex merging as sampling technique with its own
pdf" accounting, including the acceptance-radius → pdf conversion and the number-of-photons factor).

**Steps.**
1. Reuse item (1)'s hash grid to store the light-subpath vertices per iteration.
2. Add the **merge** strategy: for each camera-subpath vertex, gather nearby light-subpath
   vertices and merge.
3. Extend `MISWeight` (in `bdpt.h`) to fold the merge pdfs into the balance heuristic so
   connections and merges are combined without double counting.
4. Progressive radius (shared with item 2).
5. Validate: VCM should match/beat both BDPT and SPPM on a scene with mixed caustics + diffuse GI
   (e.g. the gallery's gold gyroid + glass) at equal time.

**Open questions.** Full MIS derivation for this renderer's cosine-free volumetric phase densities
(the BDPT notes already flag these — merging must stay consistent with them); CPU-first then GPU,
or GPU from the start; memory budget for storing light vertices each pass.

---

## (4) `.vdb` / external volume-asset support — ✅ DONE (`density vdb:<file>`)

**Status.** Implemented. `medium { density vdb:cloud.nvdb  sigma_t … }` imports a NanoVDB
FloatGrid as the heterogeneous density field. The single self-contained `NanoVDB.h` is vendored
into `src/third_party/nanovdb/` and confined to ONE loader TU (`src/vdbgrid.cpp`); everywhere
else sees only the NanoVDB-free `VdbGrid` POD (`src/vdbgrid.h`). On load the sparse grid is baked
into a **dense float lattice** + a world→index affine (recovered by probing `indexToWorld`), so
the *identical* trilinear sampler runs on CPU (`VdbGrid::sample`) and GPU (`dMedDensityAt` over an
uploaded lattice). The grid's world AABB auto-seeds the medium bound and its peak the
delta-tracking majorant. Validated: CPU and GPU renders of the imported fog sphere give
**bit-identical energy balance** (absorbed 0.9454 / escaped 0.0542 / residual 0.0004). Test asset
generator: `scraps/make_nvdb.cpp` → `scraps/cloud.nvdb`; probe scene `scraps/vdb_cloud.ftsl`.
Only float grids; dense bake (safety-capped) — a native sparse device sampler and fp16/emission
grids remain as follow-ups (see `known-issues.md`).

**Goal.** Load external VDB volumes into the existing heterogeneous `medium` path so authored
clouds/smoke/explosions (Houdini/Blender/EmberGen) can be rendered, not just procedural
`density "expr"` fields.

**Approach (decided).** Use **NanoVDB** — a single, header-only, **zero-dependency, CPU+CUDA
-native** VDB reader (the standard renderer-side VDB path). It reads `.nvdb`; `.vdb → .nvdb` is a
one-step conversion (Houdini / Blender / the `nanovdb`/`vdb_tool` CLIs). Chosen over (a) linking
full **OpenVDB** — a heavy TBB/Blosc/zlib/boost dependency chain we don't want in the CUDA build —
and (b) a from-scratch native `.vdb` parser, which is large *and un-validatable here* (no OpenVDB
and no sample `.vdb` files in this environment, so I couldn't verify it). NanoVDB volumes can be
generated with the same header, so the whole pipeline is testable end-to-end. A native `.vdb`
front-end (validated against a downloaded official OpenVDB sample) can be added later if raw `.vdb`
ingestion is ever required.

**Builds on.** `Medium` already has the heterogeneous machinery: a density multiplier
(`densityAt`, `scene.h:272`), a majorant (`densityMax`) for delta/ratio tracking, and box/sphere
bounds. This slots a **grid-backed density sampler** in beside the existing pattern-program one.

**Steps.**
1. Vendor `NanoVDB.h` (+ its minimal headers) into `src/third_party/`.
2. Add a grid-backed density to `Medium` (a NanoVDB grid handle, or a dense float grid sampled
   trilinearly) + its world transform; `densityAt` samples the grid when present.
3. FTSL grammar: `medium { density vdb:"cloud.nvdb"  sigma_t … albedo … g … }`; derive the medium
   bounds and `densityMax` from the grid's active AABB + max value.
4. CUDA: upload the grid buffer; sample on device (NanoVDB is CUDA-native).
5. Test: generate a fog-sphere `.nvdb` with NanoVDB's own tools, render it (mode D for the general
   case; mode B for a fast preview), verify against a matching procedural `density` field.

**Open questions.** Dense-grid (simple, memory-heavy) vs NanoVDB sparse tree (native, GPU sampler)
for the on-device representation; float vs fp16 grids; temperature/emission grids for fire (a
follow-up); level-set vs fog-volume grid types.

---

## (5) Triangle-mesh gaps: smooth normals, glTF, instancing

**Already done** (don't rebuild): OBJ loading (`src/mesh.h`), a binned-SAH BVH (`src/bvh.h`),
Möller–Trumbore intersection on **CPU and GPU**, per-vertex UVs, per-face `usemtl` materials, mesh
transforms (translate/rotate/scale, composed through `group{}`), and rendering in **every** mode
(A/B/C/R/D/P). The `mesh { file "asset.obj" … }` grammar exists. So a Fab/Megascans/Blender OBJ
already drops into a scene, scaled/rotated as a transform. This item is the **remaining gaps**.

**Goal.** Close the quality/format gaps so authored models look right and more formats load:
1. **Smooth per-vertex normals** — ✅ **DONE** (2026-07-12). The OBJ loader now reads `vn`, `Tri`
   stores three per-vertex shading normals (`n0/n1/n2`), and both the CPU and GPU `intersectTri`
   barycentric-interpolate a shading normal at the hit (geometric normal kept as `hit.ng`). Normals
   transform by the inverse-transpose of the mesh transform (`Affine::applyNormal`). A mesh without
   `vn` falls each per-vertex normal back to the geometric normal in `Tri::finalize()`, so untouched
   meshes stay exactly flat-shaded (bit-identical). *Not yet done:* auto-generating smoothed normals
   from a crease-angle threshold when `vn` is absent (a mesh with no `vn` stays flat) — see follow-ups.
2. **glTF/GLB** — ✅ **DONE** (2026-07-12). A second loader (`src/gltf.h` + a self-contained JSON
   parser `src/third_party/json.h`) handles `.gltf` (embedded/external/base64 buffers) and `.glb`
   (binary container), bakes the node transform hierarchy (matrix or TRS quaternion), reads
   POSITION/NORMAL/TEXCOORD_0 + indexed or non-indexed triangles, and maps `pbrMetallicRoughness`
   onto the spectral BSDFs (baseColor→upsampled reflectance, metallic→glossy tint, roughness→lobe).
   Dispatched by extension in `addMesh`; `import_materials no` forces the FTSL material. *Not yet:*
   textures, KHR extensions, skinning/morph, sparse accessors, animation.
3. **Instancing** — ✅ **DONE** (2026-07-12). A two-level BVH (TLAS over instances → shared BLAS).
   `mesh_asset "name" { file … }` loads a mesh once into local space as a `Blas` (its own tris + BVH,
   `scene.h`); `mesh_instance { of "name"  translate/rotate/scale  [material …] }` adds a `MeshInstance`
   (toWorld/toLocal affine + optional material override) as an extra leaf of `Scene::bvh`. `closestHit`/
   `occluded` transform the ray into BLAS-local space and traverse the shared BLAS — the parametric `t`
   is preserved because `Affine::applyDir` doesn't normalize, so local `t` == world `t` and the shared
   `tMax` needs no rescaling. CPU shares geometry (N copies = N affines); the GPU expands instances to
   world tris and rebuilds a flat BVH at upload (identical images, flat device memory — a follow-up).
4. Follow-ups: emissive triangles (mesh area lights), tangent-space **normal maps**, and a watertight
   ray–triangle test to kill grazing-edge cracks.

**Why.** Smooth normals are the single biggest visual win — flat-shaded organic/curved meshes and
faceted glass are the obvious "this looks CG" tell. glTF unlocks the bulk of freely-available assets
(most Fab/Sketchfab/Blender exports). Instancing makes "shrink/enlarge/duplicate an asset" the cheap
transform it should be.

**Builds on.** The whole existing mesh path — `Tri` (geometry.h:15), `loadObj` (mesh.h:112),
`intersectTri` (geometry.h:42, CPU) and its GPU mirror (`DTri` / device `intersectTri` in
render_cuda.cu), and `addMesh` in ftsl.h.

**Steps.**
1. ✅ **DONE** — Added `Vec3 n0,n1,n2` to `Tri` (geometry.h) and `DVec3 n0,n1,n2` to `DTri`
   (render_cuda.cu); parse OBJ `vn` in `loadObj` (index via the 3rd face field, `objNormalIndex`),
   transform each by `Affine::applyNormal` (inverse-transpose), and fill the three `Tri` normals.
   Both `intersectTri` (CPU + GPU) interpolate `hit.n = normalize(w0*n0+u*n1+v*n2)` and orient it
   against the ray, keeping `hit.ng` geometric. `Tri::finalize()` falls absent normals back to `gn`
   so non-`vn` meshes stay flat-shaded. Validated: low-poly UV sphere renders smooth (mode R/B, CPU
   and GPU) vs the flat version's facets; energy balance bit-identical CPU↔GPU. *Follow-up:* crease-
   angle auto-smoothing when `vn` is absent; shading-normal hemisphere clamp for transmission.
2. ✅ **DONE** — glTF loader (`src/gltf.h`) + minimal JSON parser (`src/third_party/json.h`): parses
   nodes/meshes/accessors/bufferViews/buffers (GLB BIN chunk, external `.bin`, base64 data URIs),
   bakes node transforms (matrix or TRS), maps metallic-roughness → spectral BSDFs; `addMesh`
   dispatches `.gltf`/`.glb` by extension. Validated: a metallic sphere + rotated diffuse cube scene
   (both `.gltf` embedded-buffer and `.glb`) renders correctly on CPU (mode R) and GPU (mode B),
   node transforms + materials + smooth normals all correct.
3. Two-level BVH for instancing: keep per-mesh BLAS, add a TLAS over `{blasId, Affine}` instances;
   `mesh_instance { of "name" translate … }` grammar; transform the ray into BLAS space at traversal.
4. Validate: render a smooth sphere-mesh vs the analytic sphere (should match), a glass Stanford
   bunny (smooth refraction, no facets), and an instanced grid (memory stays flat).

**Open questions.** Whether to reuse an existing scene BVH/accel or give meshes their own two-level
structure; shading-normal vs geometric-normal handling for transmission and shadow terminators;
texture-format scope (PNG/EXR/JPG) and sRGB handling; how much glTF material model to map onto this
renderer's spectral materials (metallic-roughness → the existing BSDFs); mesh memory budget on GPU.

---

## Suggested build order

1. **(4) `.vdb`/NanoVDB** — independent, self-contained, immediately useful, and a good warm-up
   that touches the volume path without the MIS complexity.
2. **(1) Photon map / radiance cache** — the keystone; also delivers the cross-camera flythrough
   win on its own.
3. **(2) PPM/SPPM** — progressive convergence + caustics on top of (1).
4. **(3) VCM/UPS** — glue (1)'s merging to the existing BDPT connections under MIS; the capstone.

**(5) Triangle mesh** is independent of the photon chain and can slot in wherever it's wanted — it's
the enabler for using external/authored assets (Fab, Megascans, Blender), so prioritize it whenever
polygon geometry is needed rather than treating it as part of the (1)→(3) sequence.

GPU note for (1)–(3): prefer a **uniform hash grid** over a kd-tree throughout — it's the
CUDA-friendly structure the shared photon/light-vertex passes all reuse.

---

# User requests — 2026-07-14

Five requests captured from a planning conversation. Status: **IN PROGRESS** /
**TODO** / **NEEDS DECISION** (blocked on a question) / **DONE**.

## (6) Isosurface airtightness audit — ray-parity test — DONE 2026-07-14

**Shipped** as `-check-airtight` (`src/airtight.h`, wired in `main.cpp`; chord count via
`-check-airtight-rays`, default 4000). Fires random exterior→exterior chords through
each isosurface's container and counts the boundary crossings the renderer's *own*
marcher (`intersectImplicit`) reports — odd parity ⇒ leak. Also samples the container
boundary for interior (f<0) area (the definitive open-cap signature on `open` surfaces)
and runs a dense reference sampling to flag marcher overshoot even when parity stays
even. Validated: a `capped` box-clipped gyroid and the CSG sphere-clipped showcase ball
report `[OK]`; the same gyroid marked `open` is correctly flagged (≈49% odd-parity
chords, 50% of boundary inside the solid). Documented in README under *Auditing the
marched field directly*. Original request/analysis below.



**Request.** A tool that detects whether an isosurface is air-tight *as the renderer
actually marches it* (the analytic field), not just via a polygonised proxy like
`-check-watertight`. Proposed method: shoot chords across the container bbox/sphere and
flag any chord whose inside/outside **parity** is inconsistent (crosses the zero
level-set an odd number of times between two exterior endpoints, or is inside the solid
at a container face).

**Answers to the questions raised:**
- *Can an isosurface self-intersect?* No. The rendered surface is a level set {f = 0} of
  a continuous scalar field; at every regular point (grad f ≠ 0) it is locally a smooth
  manifold and cannot cross itself. Self-intersection is a *mesh* pathology (two
  triangles passing through each other) — an analytic level set has no triangles to
  cross. (grad f = 0 points can pinch/degenerate, but that's not self-intersection.)
- *Would marching cubes catch everything?* No. MC samples the field on a finite grid, so
  any leak/thin-wall/spike narrower than a cell is missed or mis-resolved — it audits a
  *resampled copy*, not the marched field.
- *Analytic test?* Not for arbitrary `expr` fields — no closed form for closedness of the
  zero set. Monte-Carlo ray parity probes the exact field the renderer marches, so it's
  the right pragmatic tool.

**Leak risks it must catch:** (a) `contained_by` clipping the solid into an open cap
(solid region touches a container face); (b) march overshoot from a wrong
Lipschitz/`max_gradient` bound; (c) features thinner than the marcher step.

**Plan.** New read-only `-check-airtight` audit in C++ (reuses the field evaluator so it
probes the true marched field). For N random exterior→exterior chords across the
container: densely sample f, count zero-crossings, compare crossing parity against the
sign of f at both boundary endpoints; separately sample f on the container faces to
catch the cap-clipping case. Report leak fraction + worst offenders. Non-destructive.

## (7) Nested / overlapping dielectrics (medium stack) — LEVEL 0 DONE

**Request.** Replace the hardcoded exterior IOR 1.0 so glass-in-water and intersecting
dielectrics are modeled correctly, for **all** modes.

**Status.** Level 0 (priority field) is **implemented and validated across every
integrator**: CPU modes R/A/B/C/D/M/S/U (`backward.h`, `render.h`, `bdpt.h`,
`photonmap_render.h`, `sppm_render.h`, `vcm.h`) and the GPU forward/backward/BDPT/photon
backends (`render_cuda.cu`). Each path now carries a tiny LIFO medium stack
(`medium_stack.h`, device `DMediumStack`); at every dielectric hit the exterior IOR is
taken from the enclosing (highest-priority) medium instead of hardcoded 1.0, and the
lower-priority surface inside an overlap is suppressed (ray passes straight through).
**Safe fallback:** the priority rule only fires when *both* sides of an interface carry an
explicit priority (air/empty stack is always valid at IOR 1.0), so priority-free scenes
render bit-identically to before. Validation: mode-R priority vs no-priority scenes differ
across 33.6% of pixels; GPU matches the CPU reference to mean 1.3/255; modes C/M stay
energy-conserving. The ahead-of-time missing-priority warning (below) is also implemented.

Levels 1/2 (true physical stacking of co-located media / interpenetrating volumes) remain
future opt-in tiers; Level 0 already fixes the common nested/overlap cases "for all modes"
at essentially zero cost.

**Decision (tiered — expose all three as options).** A convenience/speed/ability ladder:
- **Level 0 — priority field (Schmidt & Budge 2002).** Integer `priority` per dielectric;
  where solids overlap the highest priority wins that region, so each boundary hit deduces
  the true from/to IOR from the two sides' priorities — no per-ray state. **Essentially
  free** (a couple of int compares per hit). Covers nested gem-in-water + coatings. This is
  the **default** fix. *Limitation:* can't stack genuinely co-located media (resolves
  overlap by ranking, not physical stacking).
- **Level 1 — per-path medium stack, nested only.** Each ray/photon path carries a tiny
  LIFO of the media it is inside (push on entry, pop on exit). Trivial on CPU; on GPU the
  cost is register/local-mem pressure → **low single-digit %** on scenes that use it.
- **Level 2 — per-path stack with overlap support.** Same stack but allows interpenetrating
  volumes, so each crossing must be resolved against all active media → GPU divergence →
  **~5–15%** on affected scenes only.

Default Level 0 (fixes the common cases "for all modes" at zero cost); scenes opt into
Level 1/2 when they need true nesting/overlap. Only paths touching dielectrics pay anything.

**Missing-priority warning (required).** If a place needs a priority definition and none is
given, warn — both ahead-of-time and at render time:
- *Ahead-of-time (scene analysis, cheap, primary UX):* pairwise-test dielectric bounding
  volumes for overlap; if two overlap and either lacks an explicit `priority`, warn naming
  both materials ("exterior IOR is ambiguous — add `priority N`"). Conservative
  (bounds-overlap ⊇ surface-overlap) so it's a warning, not an error.
- *At render time (definitive safety net):* when a refraction crosses an ambiguous overlap
  (equal/undefined priorities), accumulate a per-material-pair counter and print one summary
  at the end ("N rays hit ambiguous dielectric overlap X↔Y — set priorities").

**Isosurface overlap detection (for the same warning).** Isosurfaces don't have plain box
bounds — they're clipped to a `contained_by` container (box or sphere). Two ways to detect
that two isosurfaces overlap / nest:
- **Bounding-volume comparison (chosen for the warning):** compare the `contained_by`
  boxes/spheres. Cheap, and *conservative in the right direction* — because each field is
  clipped to its container, the surfaces can only overlap where the **containers** overlap,
  so container-overlap never misses a real function-overlap. If one container sits inside
  another, the functions almost certainly overlap somewhere, so warn. May over-warn
  (containers overlap but the zero-sets happen not to), which is acceptable for a warning.
- **Marching cubes (rejected for the warning):** mesh both fields and test mesh
  intersection. More precise on false positives but expensive *and itself resolution-limited*
  (can miss a thin overlap between grid cells) — the wrong trade for a cheap conservative
  warning. Bounding-volume wins here.

## (8) Mesh/animation formats + OBJ-sequence → video — PARTIALLY DECIDED

**Already shipped (correction):** `.obj` **and** `.gltf`/`.glb` import already work
(roadmap item 5.2, done 2026-07-12) — static-mesh loading with node transforms,
smooth normals, and PBR material mapping. So the new asks are the animation/video
pipeline and the two heavy formats.
- **OBJ-sequence → MP4 driver** — ✅ **DONE 2026-07-15** (`tools/obj_sequence_to_video.py`).
  Self-contained Python: renders a sequence of per-frame OBJ files with ftrace (a template
  scene with a `{obj}` placeholder substituted per frame) and encodes the frames to MP4 with
  ffmpeg — no new C++ deps. Supports directory/glob frame input (natural sort), per-frame
  `--time`/`--spp`/`--noise` budgets, `--resume`, `--start/--end/--step` sub-ranges,
  `--encode-only`/`--no-encode`, `--write-template`, `--dry-run`, and `--crf/--codec/--pix-fmt`
  encode controls. Validated end-to-end on a 5-frame rotating/growing-cube sequence (frames
  differ as expected; H.264 yuv420p MP4 produced; resume + encode-only paths confirmed).
  Documented in README ("Animated geometry (OBJ sequences) → video").
- **FBX — DECIDED: vendor ufbx (MIT).** For a *renderer's import path* the proprietary
  Autodesk FBX SDK offers nothing we'd use: its exclusive strengths are FBX *writing*,
  evaluating **authored constraint rigs** (artist-built IK/aim/parent-constraint control
  networks), and DCC round-tripping — all authoring concerns. An importer consumes the
  **baked** result (geometry, normals/UVs, materials, skinning/blend-shapes, animation
  curves), which **ufbx** (MIT, single-file, zero-dep) reads and samples itself. So vendor
  ufbx into `src/third_party/` like the glTF/JSON headers — no Autodesk EULA, no manual
  install. (Optional future: an SDK-backed build behind a compile flag only if ftrace ever
  needs to *write* FBX or evaluate live rigs. Low priority.)
- **Alembic (.abc)** — heavy SDK (Imath + HDF5/Ogawa). **Still open:** worth the build
  weight, or is an OBJ/glTF/FBX sequence enough for animation? Defer until asked.

**Start order:** OBJ-sequence → MP4 driver first, then ufbx FBX import. Alembic deferred.

## (9) Camera archetype presets — DONE 2026-07-14

**Shipped.** `resolveCameraPreset()` (`src/ftsl.h`) + `preset <name>` in the camera-block
parser (`readFilmExposure`, applies to `camera`/`camera_path`/`camera_orbit`/`camera_curve`
alike). The preset pre-fills the `CamSpec` film size / focal length / f-stop **before** the
block's own lines, so any dial can still be overridden after `preset <name>`, and the same
preset serves a finite-lens sim (mode A/C) or a correct-FOV pinhole (R/B/U — aperture
collapses to a point). Five archetypes ship (sensor mm / focal mm / f-stop): **cinema**
(Super35 24.6×13.8, 35mm, f/2.1), **pocket** (1″ 13.2×8.8, 8.8mm, f/4), **portable**
(full-frame 36×24, 35mm, f/1.8), **vintage** (35mm film, 50mm, f/3.5), **vintage-slr**
(35mm film, 50mm, f/1.4). Aliases accepted (cine/compact/mirrorless/rangefinder/slr).
Documented in README. Original decision below.

**Request (clarified).** NOT props in the scene. Make **named camera preset objects**
users can reference (like `material { preset gold }`), one per archetype in `cameras/`,
each supplying physically-plausible optics. One preset serves both worlds: a finite-lens
simulation in **mode A/C** and a correct-FOV **pinhole** in the backward modes (R/B/U) —
same preset, aperture just collapses to a point where there's no DOF.

**Mechanism (no new format needed).** The camera grammar already exposes the "knobs":
`film { format|size }` (sensor mm), `lens <mm>` (focal length → fov), `fstop <N>`
(aperture), `focus`, `zoom`, `film { iso shutter exposure }`, `projection`. A preset is
pure shorthand — a `resolveCameraPreset(name, CamSpec&)` (mirroring the existing
`resolveMaterialPreset` / `resolveLensPreset`) that pre-fills those same `CamSpec` fields
from a table, *before* the user's own lines, so any dial can still be overridden after
`preset <name>`.

**Archetypes (specs confirmed from the reference photos in `cameras/`):**
- **cinema** — lens reads "35 T2.1", body "4K" (Blackmagic-style cine): Super35 sensor,
  35mm, ~T2.1 (f/2.1). Shallow, cinematic.
- **pocket** — "RX0818" (Sony RX0-style rugged compact): 1" sensor (13.2×8.8mm), fixed
  ~24mm-equiv wide, ~f/4 → deep DOF.
- **portable** — white mirrorless w/ bright prime: full-frame 36×24, ~35mm, f/1.8.
- **vintage** — purple folding rangefinder (FED/Zorki lineage): 35mm film 36×24, ~50mm,
  f/3.5 collapsible.
- **vintage-slr** — silver/black classic w/ big fast lens: 35mm film 36×24, ~50mm, ~f/1.4.

**Plan.** Add `resolveCameraPreset`, wire `preset <name>` into the camera-block parser,
ship the five archetypes, document in README. Self-contained; doesn't touch the meshes.

## (10) Re-render golden gyroid hero (`scenes/showcase.ftsl`) — DONE (2026-07-14)

**Request.** Re-render the non-flyby golden gyroid hero, which previously read dark at
the front because the fourth wall was blank (wall + front fill light have since been
added).

**Status.** Confirmed `scenes/showcase.ftsl` already has the closed front wall
(`front (behind camera)`) and front fill light. Re-rendering with the live window to
confirm the gold reads correctly head-on.
