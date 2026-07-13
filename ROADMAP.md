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

**So we do NOT yet have:** a stored photon map, progressive photon mapping (PPM/SPPM), VCM/UPS
merging, or external volume-asset (VDB) import. Those are the items below.

### Dependency graph

```
        ┌──────────────────────────────────────────┐
        │  (1) Photon map / view-independent cache   │   ← keystone
        └──────────────────────────────────────────┘
             │                         │
             ▼                         ▼
   (2) Progressive PM (PPM/SPPM)   (3) VCM / UPS
                                   (needs BDPT ✔ + merging from (1))

   (4) .vdb / NanoVDB volumes  ── independent, can land anytime
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

## (2) Progressive photon mapping (PPM / SPPM)

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

## (3) VCM / UPS (Vertex Connection and Merging / Unified Path Sampling)

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

## (4) `.vdb` / external volume-asset support

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

## Suggested build order

1. **(4) `.vdb`/NanoVDB** — independent, self-contained, immediately useful, and a good warm-up
   that touches the volume path without the MIS complexity.
2. **(1) Photon map / radiance cache** — the keystone; also delivers the cross-camera flythrough
   win on its own.
3. **(2) PPM/SPPM** — progressive convergence + caustics on top of (1).
4. **(3) VCM/UPS** — glue (1)'s merging to the existing BDPT connections under MIS; the capstone.

GPU note for (1)–(3): prefer a **uniform hash grid** over a kd-tree throughout — it's the
CUDA-friendly structure the shared photon/light-vertex passes all reuse.
