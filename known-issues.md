# Known Issues & Technical Debt

Running log of unsolved bugs and accumulated tech debt. Fix items here as soon
as practical; this file is the fallback for what can't be addressed immediately.

## Recently fixed

### GPU forward camera-splat out-of-bounds write (illegal memory access) — FIXED 2026-07-12

The GPU forward/light-tracing kernel (modes A/B/C, and the splat in M/S/U) could
crash with `[cuda] forward kernel failed: an illegal memory access was encountered`.
**Root cause:** `DCamera::project()` / `lensImage()` compute the splat pixel as
`px = (int)((ix*0.5+0.5)*resX)` in **FP32** (`Real`). The on-film rejection test only
guarantees `ix,iy < 1`, but the gap between the largest float below 1 and 1.0 is
~6e-8, so for a photon landing within that gap of the film edge, `(ix*0.5f+0.5f)`
rounds to exactly `1.0f` and `px` becomes `resX` (likewise `py==resY`). `filmAdd()`
indexes `py*resX+px` with no bounds check → out-of-bounds write. Data-dependent, so
it manifested only for some scenes/resolutions and always eventually with enough
photons (longer renders reliably tripped it). The CPU `Camera::project()` uses
`double` and never rounds up this way, which is why CPU renders were unaffected.
**Fix:** clamp `px∈[0,resX-1]`, `py∈[0,resY-1]` right after the cast in all three GPU
projection sites (`project()` rectilinear + fisheye/panoramic branches, `lensImage()`;
`catchPhoton()` routes through `lensImage()`). The rejection test already guarantees
the point is on-film, so clamping the boundary roundup to the last pixel is exact.
See `render_cuda.cu` ~line 555/577/624.

## Open bugs

### `light cylinder` emits no illumination (tube is visible but lights nothing) — 2026-07-11

A `light cylinder` renders as visible glowing emissive geometry (the tessellated
lateral wall shows up when a camera ray hits it directly), but it does **not
illuminate any other surface** — neither via next-event estimation nor via BSDF
bounce. Reproduced with an isolation scene (`scraps/cyl_test.ftsl`): a white
diffuse wall lit *only* by a `light cylinder` renders pure black behind the visibly-
glowing tube, on **both** `-device cpu` and `-device gpu` (identical auto-exposure
1.54e-14, i.e. zero contribution from the light). Contrast: `light sphere` and
`light area` both illuminate correctly.

- **Where:** `ftsl.h` `addLight` cylinder branch (~line 1496) calls
  `L.scene.addCylinderLight(...)`, so the light is registered for sampling. The bug
  is downstream in the light-sampling / direct-lighting path (`scene.h` /
  `render.h` / `render_cuda.cu`) — the cylinder light is likely missing from (or
  mis-weighted in) the NEE light-sampling switch, and its emissive tris are probably
  excluded from BSDF-hit emission accounting (to avoid double counting) so both
  contributions vanish.
- **Repro:** `ftrace -in scraps/cyl_test.ftsl -mode R -device cpu -spp 128 -r 200 -o png/cyl_test.png` → wall is black.
- **Proper fix:** ensure `sampleLight`/`lightPdf` (CPU and GPU) handle the cylinder
  light type and return correct radiance+pdf, and/or let BSDF rays that hit the
  cylinder's emissive tris contribute their emission with proper MIS. Then re-test
  with `scraps/cyl_test.ftsl` (wall should light up).
- **Workaround in scenes:** use `light sphere` (rings/stacks) or `light area` for
  tube-like emitters until fixed. `scenes/mirror_selfie.ftsl` uses sphere-light
  accents + colored walls for this reason.

## Tech debt

### glTF/GLB loader is a static-geometry subset — 2026-07-12
The new glTF 2.0 loader (`src/gltf.h` + `src/third_party/json.h`) covers the common
static-mesh case but deliberately omits a number of glTF features. Each is a scoped
follow-up, not a bug:
- **No textures.** Only `baseColorFactor`/`metallicFactor`/`roughnessFactor` *scalars*
  are read; `baseColorTexture`/`metallicRoughnessTexture`/`normalTexture` are ignored.
  Proper fix: decode referenced images (glTF images are PNG/JPEG — the renderer already
  vendors stb_image), register them as `Scene::textures`, and set `reflectTex`/UV set.
- **No KHR material extensions** (transmission, clearcoat, volume, ior, emissive
  strength, sheen, specular). A glass glTF loads as an opaque glossy/diffuse, not a
  dielectric. Proper fix: read `extensions.KHR_materials_transmission`/`_ior` → map to
  `MatType::Dielectric` with the given ior; other extensions as feasible.
- **No `emissiveFactor` import.** Emissive glTF materials load unlit. (Intentionally
  skipped for now: setting `emit` without registering the tris as a sampled light would
  desync NEE; doing it right means adding mesh-emitter area lights — tied to ROADMAP §5
  "emissive triangles".)
- **No skinning, morph targets, animation, or sparse accessors.** Static bind pose only.
- **Non-triangle primitives** (points/lines/strips/fans, `mode != 4`) are skipped with a
  note; only `mode 4` (TRIANGLES) is baked.
- Materials are created **per glTF material, not deduplicated across meshes/files**. (A
  `mesh` still bakes its triangles into `Scene::tris`; use `mesh_asset`/`mesh_instance`
  for shared instanced geometry — see below.)
The core path (buffers/GLB, node transforms, POSITION/NORMAL/TEXCOORD_0, indexed +
non-indexed tris, metallic-roughness → BSDF) is validated on CPU and GPU.

### Instancing memory saving is CPU-only (GPU expands instances) — 2026-07-12
`mesh_asset`/`mesh_instance` (§5c) give a true two-level BVH on the CPU: instances share
one BLAS (triangles + BVH), so N copies cost N affines. **The GPU has no two-level
traversal** — `buildUploadScene` (`render_cuda.cu`) EXPANDS every instance into
world-space triangles, appends them to the flat device tri list, and rebuilds a single
flat BVH over the whole set at upload. Images are identical to the CPU, but device memory
scales with total instanced triangles (no sharing), so a huge instanced scene that fits on
the CPU can OOM on the GPU. Proper fix: a device two-level BVH — upload per-BLAS
node/tri/primIdx pools + an instance table (toLocal affine + blasId + matOverride) and add
an instance-leaf branch to the device `traverseClosest`/`traverseAny` that transforms the
ray into BLAS space (parametric `t` is preserved, exactly as on the CPU). Deferred because
it touches the hottest device kernel; the expand-at-upload path is correct and low-risk.

### Forward modes render ~5% brighter than the backward reference (`R`) — 2026-07-12
On a pure-diffuse Cornell box (`scraps/cornell_diffuse.ftsl`) the forward splat modes
and the new photon-map mode agree with each other but sit **~5% brighter** than the
backward path tracer:
- `R` (backward): mean 61.05  →  `B` (forward splat): 63.96  →  `M` (photon map): 63.84.

Mode `M` matching mode `B` to within 0.2% is the *expected* result (both measure the
same forward light transport, just from a stored map vs. a live splat) and **confirms
the photon map is correct**. The open question is the **pre-existing forward-vs-backward
discrepancy** — `B` and `R` should converge to the same image but don't quite. Likely
suspects: a subtle difference in area-light emission normalization / solid-angle pdf
between the forward emit sampler and the backward NEE light pdf, or a `cos`/pdf factor
at the light or first diffuse bounce. Not introduced by this work; surfaced by the mode-M
validation. **Proper fix:** derive both estimators' light-vertex measure on paper for the
1-bounce diffuse case and reconcile the constant (check `emitSampler` power vs. `sampleLight`
radiance × pdf). Until then `V`'s residual bakes this ~5% in.

### No bounded / per-object participating medium (fog is global-only) — 2026-07-12 — CPU + GPU forward DONE 2026-07-12 (box/sphere/implicit bounds, density fields, multi-medium superposition, object-name bounds)
**Resolved on the CPU forward tracer.** The `medium` block now takes an optional
`bounds { min/max }` box (AABB the fog is clipped to) and an optional `density <expr>`
scalar field (same infix expression VM as isosurface `function` fields — variables
`x y z r`, constant `pi`) that multiplies `sigma_t` per point, so fog forms blobs with
soft, formula-defined boundaries. Majorant is `density_max` (explicit or auto-estimated
on a 24³ grid over `bounds`). Sampling uses unbiased **delta (Woodcock) tracking** for
scattering and **ratio tracking** for shadow transmittance; a plain homogeneous medium is
bit-identical to before (one RNG draw in the free-flight, exact `exp` transmittance).
Implemented in `scene.h` (`Medium` struct: `density`/`densityMax`/`bounded`/`bmin`/`bmax`
+ `densityAt`/`clipToBounds`/`heterogeneous`), `ftsl.h` `addMedium` (bounds/density/
density_max parsing), `render.h` (`sampleMediumCollision`/`mediumTransmittance` + connect
updates). Validated with `scraps/fogblob.ftsl` (a soft glowing sphere blob, mode B).

**GPU forward — DONE 2026-07-12.** The density field + bounds + delta/ratio tracking are
now ported to `render_cuda.cu`: `DMedium` carries `heterogeneous`/`density` (a device
`PatNode` pool + `densityN`)/`densityMax`/`bounded`/`bmin`/`bmax`; `dMedDensityAt` (postfix
VM twin of `densityAt`), `dMedClip` (twin of `clipToBounds`), `dMedSampleCollision` (delta/
Woodcock tracking twin of `sampleMediumCollision`), and `dMedTransmittance` (ratio-tracking
twin of `mediumTransmittance`) drive the forward `shadeStep` free-flight and every camera
splat (`connect`/`connectVolume`/`connectLens`/`connectLensVolume` — the two RNG-less
connects now take `rng` for ratio tracking). A homogeneous medium keeps the exact analytic
path (no extra RNG draw). Validated on an RTX 4090 mode B: `scraps/fogblob.ftsl` GPU-vs-CPU
16×16-block RMSE 1.07/255 with ~0 bias (per-pixel diff is pure MC noise from the 0.95-albedo
fog; means 38.57 vs 38.56), and a homogeneous regression (`scraps/foghom.ftsl`) block RMSE
2.45/255, bias −0.009.

**Per-object (sphere) fog bound — DONE 2026-07-12.** `bounds` now also accepts
`{ center <x y z> radius <r> }`, confining the fog to a **sphere** region — the simple
per-object case ("the whole inside of a glass sphere"): author the same center/radius as
the object. Added `MediumBound { Box, Sphere }` + `boundShape`/`bcenter`/`bradius` to the
`Medium` struct with a ray∩sphere interval in `clipToBounds` (heterogeneous density works
inside a sphere too — the majorant grid uses the sphere's AABB, filled in by the parser).
Mirrored on the GPU (`DMedium.boundShape`/`bcenter`/`bradius`, `dMedClip` sphere branch,
upload path). Validated on an RTX 4090 mode B: open glowing orb `scraps/fogorb.ftsl`
GPU-vs-CPU block RMSE 0.96/255 (bias 0.024), glass-shell `scraps/fogsphere.ftsl` block RMSE
0.57/255 (bias −0.010); box/heterogeneous path unchanged (fogblob block RMSE still 1.07).
*Limitations:* (1) **Fog inside a `dielectric` shell is not imaged directly by the
next-event modes — an accuracy (bias) issue, empirically confirmed 2026-07-12.** The
direct view of the fog is a specular↔volume (SDS-type) path: the camera sees the fog
*through* the curved glass, i.e. along a *refracted* line. The next-event/splat modes
connect a fog vertex to the camera with a **straight** ray, which (a) is occluded by the
glass surface (`occluded()` treats every surface as opaque) and (b) could not bend even if
it weren't — so the contribution is structurally **zero**, not merely noisy. This affects
both the **pinhole splat (`B`)** and the **finite-lens splat (`A`)** — both are NEE-based,
and both render the fog-through-glass as **black** (verified: `scraps/fogsphere.ftsl` mode
B whole-image mean 6.6 but the fog-sphere center box mean 0.000; mode A identical). Only the
**physically-tracing modes** — photon-catch (`C`) and BDPT (`D`) — can sample the path at
all, because a real photon scatters in the fog, **refracts** out through the glass, and
lands on a finite aperture. For `C` that path is extraordinarily improbable (a fog-scattered
photon must exit heading almost exactly at the pupil), so at practical sample counts `C` is
effectively black too (60 M photons, aperture 0.45: fog-sphere center still mean 0.000) — an
**efficiency** problem on top of the accuracy one. **BDPT `D` — RESOLVED 2026-07-12** (volumetric
BDPT, below): its camera subpath refracts through the shell (specular vertices) to a volume
in-scatter vertex, then MIS-connects to the light, so a lantern inside a fogged glass sphere
images as a bright disc — `scraps/fogsphere.ftsl` mode D fog-sphere center box mean 0.22
(saturating) vs mode B's 0.00, at the same absolute exposure.
The fog still correctly **lights the surrounding room** (indirect, via NEE off the walls),
and an **open** fog sphere (no glass shell) is directly viewable in every forward mode
(`scraps/fogorb.ftsl` mode B center box mean 135.8). A naive "let connect rays pass through
glass" hack is wrong (it draws the fog along a straight line, with no lensing, in the wrong
place) and is deliberately avoided — the correct fix is the analytic specular connection below.

**Mode-B analytic specular connection through glass SPHERES — DONE 2026-07-12.** The proper
refractive/manifold next-event estimation is now implemented for the tractable case: a
**glass sphere** in the **pinhole splat (`B`)**. For each fog in-scatter vertex (and each
diffuse surface vertex), the renderer solves — in closed form — the refracted eye ray that
leaves the vertex, bends through the sphere, and reaches the pinhole: a planar reduction of
the two-refraction manifold to a **1-D root solve**, with a ray-differential Jacobian
(`G = eps²/|ax·by − ay·bx|`) supplying the splat weight, and Fresnel-transmittance ×
Beer-Lambert interior absorption × medium transmittance along the two glass segments. The
sphere's ior is evaluated at the photon's **own wavelength**, so the refraction is dispersive
for free. Unified surface/volume vertices via a `SpecVtx`/`DSpecVtx` `term(wP)` (Lambertian
`rho/π·cosSurf` vs HG-phase·albedo). Implemented on **CPU** (`render.h`:
`connectSpecularSphere`/`connectSpecularSphereInside`, `camSpecularSplatAll`/`…VolumeAll`)
and **GPU** (`render_cuda.cu`: `dConnectSpecularSphere`/`…Inside`, `camSpecularSplatAll`/
`camSpecularSplatVolumeAll`), validated GPU-vs-CPU and vs BDPT/mode-P ground truth. So a
lantern glowing inside — and a fly-through *through* — a clear glass orb now images correctly
in mode B (see `scenes/lanterns.ftsl`). *Still out of scope for now:* the finite-lens splat
(`A`), photon-catch (`C`), and **non-spherical** dielectric shells — those still render the
direct fog-through-glass view black in the forward splat modes (use BDPT `D`, which handles
any shape). Mode A and flat-plane (window/pane) analytic connections are the next tracked
items.

**Multiple coexisting media (superposition) — DONE 2026-07-12.** `Scene::medium` is now a
vector `Scene::media` of independent, possibly overlapping media; several `medium {}` blocks
coexist (e.g. two tinted fog orbs + a global haze). The forward tracer superposes them
physically: extinction adds, so total transmittance is the **product** of the per-medium
transmittances (`Renderer::mediaTransmittance` / `dMediaTransmittance`), and the first
collision is the **earliest** of the media's independent free-flights, with the winning
medium's albedo/`g` driving the scatter (`sampleMediaCollision` / `dMediaSampleCollision` —
Poisson superposition). A single-medium scene stays bit-identical. `Scene::backwardMedium()`
returns the first medium for the homogeneous-only backward/BDPT path. Implemented in
`scene.h`, `render.h`, `ftsl.h` (`addMedium` appends), `render_cuda.cu` (`DScene.media`/
`mediaN` + a `DMedium` array). Validated on an RTX 4090 mode B: `scraps/fogmulti.ftsl`
(warm + cool disjoint orbs + global haze) GPU-vs-CPU 16×16-block RMSE 1.40/255 (bias 0.012,
means match to 0.02%); single-medium regression (`scraps/fogorb.ftsl`) block RMSE 1.07,
unchanged.

**Object-name / implicit-shape fog bounds — DONE 2026-07-12.** `bounds { object "<name>" }`
shapes the fog to a **named** scene object: a named `sphere` → its exact analytic sphere
bound; a named `isosurface` → **field membership** (a new `MediumBound::Implicit`: the fog
fills the field interior via `fieldEval < 0` — inside-sign auto-detected from the field's
value at its AABB center — carved per-point inside delta/ratio tracking over the field's
AABB, reusing the same field VM as isosurface rendering); a named `mesh` → the mesh's world
**AABB** (box approximation; true mesh containment deferred). Media are resolved in a
deferred second sweep so the object may be authored anywhere. Implemented in `scene.h`
(`Medium::boundField`/`boundFieldExpr`/`boundInsideNeg` + `insideField`/`densityAt`/
`heterogeneous`), `ftsl.h` (name registries populated by `addSphere`/`addIsosurface`/
`addMesh`; `object` branch in `addMedium`), `render_cuda.cu` (`DMedium.boundField`/
`boundFieldN`/`boundFieldExpr`/`boundInsideNeg`, `dMedDensityAt` membership carve-out,
`appendFieldProgram` bakes the medium field into the shared device field pool). Validated on
an RTX 4090 mode B: metaball-`blob`-shaped glowing fog in a glass shell (`scraps/fogimplicit.ftsl`)
GPU-vs-CPU energy identical (absorbed 0.9978) and indirect room lighting agreeing within the
(large) dim-caustic noise floor. *(Same fog-inside-glass direct-view limitation as above
applies — an implicit-shaped fog is enclosed by its own isosurface, so its direct camera view
is a refracted SDS path; it lights the room correctly.)*

**Remaining gap (BDPT fully closed):**
- **BDPT (mode D) — ALL media DONE 2026-07-12 (CPU + GPU), incl. heterogeneous.** `bdpt.h`
  and the GPU BDPT megakernel (`render_cuda.cu` `kBdpt`) handle media of every kind —
  global haze, multiple superposed media, box/sphere/object-**bounded** fog, **and
  heterogeneous `density`-field blobs** — with volume in-scatter (`VType::Medium` /
  `BV_MEDIUM`) vertices, HG-phase connections and transmittance-weighted edges. Both
  `bdptUnsupportedFeature` (CPU) and `cudaBdptSupported` (GPU) now accept any medium.
  Validation (homogeneous): global haze CPU-vs-GPU whole-image mean 0.04698 vs 0.04702
  (+0.09%); bounded fog-through-glass (`scraps/fogsphere.ftsl`) CPU-vs-GPU center 0.237 vs
  0.242. Validation (heterogeneous): `scraps/fogblob.ftsl` (soft-edged density blob) mode D
  GPU vs mode B forward reference mean 0.04213 vs 0.04247 (−0.8%), centerMean 0.30009 vs
  0.30211 (−0.7%) — within the ~6% MC noise floor, confirming unbiased. See the resolved
  entry below for why the homogeneous cancellation is *not* required for correctness.
- **Backward modes (R/V) + P camera layer still treat it as homogeneous** (on BOTH
  backends). `backward.h` (modes R/V) and the camera-side layer of the P composite still
  use the medium as a single global homogeneous haze and ignore `density`/`bounds`; on the
  GPU, `cudaBackwardSupported` rejects *any* medium so R/V fall back to the CPU tracer,
  which shares that homogeneous-only limitation. `main.cpp` `runRender` **warns** when a
  heterogeneous/bounded medium is rendered in R/V/P. Proper fix: port delta/ratio tracking
  into the backward volume march too (then mirror it on the GPU).

### Heterogeneous (density-field) media in BDPT (mode D) — DONE 2026-07-12 (CPU + GPU)
**What:** BDPT (mode D) now renders **heterogeneous** (`density`-field) media unbiasedly on
both backends, using the *same* code path as homogeneous/bounded media — no null-scattering
rewrite was needed. Both `bdptUnsupportedFeature` (CPU) and `cudaBdptSupported` (GPU) accept
any medium; the heterogeneous rejections were removed.

**Why the earlier "cancellation breaks → biased" reasoning was wrong (corrected):** the
balance-heuristic MIS weights `w_s = p̂_s / Σ_i p̂_i` are a **partition of unity for any
consistent positive pdfs** — `Σ_s w_s = 1` holds identically, regardless of what each `p̂_i`
is. The estimator `E[ Σ_s w_s · f/p_s ] = ∫ f · (Σ_s w_s) dx = ∫ f dx` is therefore
**unbiased** whenever (a) the *sampled* strategy's throughput `f/p_s` is exact, and (b) the
weights sum to 1. Omitting the heterogeneous distance-pdf / transmittance from the MIS
weights (the homogeneous bookkeeping we reuse) only makes the `p̂_i` a *different but still
consistent* set of positive numbers — it changes the **variance**, never the bias. This is
exactly what PBRT-v3 does for heterogeneous media. The homogeneous σt·exp/transmittance
cancellation is a variance nicety, **not** a correctness requirement.

**Why the sampled-strategy throughput stays exact:** subpath medium vertices are placed by
**delta (Woodcock) tracking** (`sampleMediaCollision`) with **analog throughput** (β
unchanged; RR-absorb on albedo) — the same unbiased sampler validated mode B uses.
Connection edges are weighted by **ratio-tracking transmittance** (`mediaTransmittance`),
which appears *linearly* in the connection throughput, so its unbiased estimate keeps the
connection estimate unbiased. Albedo and phase `g` are spatially constant (only density
varies), so a medium vertex's `mediumId`/`mediumG` fully determine phase + albedo and
`vertexPdf` recomputes the cosine-free phase-direction density consistently forward/reverse
regardless of heterogeneity.

**Implementation:** removed the `heterogeneous()` guards in `bdptUnsupportedFeature`
(`main.cpp`) and `cudaBdptSupported` (`render_cuda.cu`); the existing `randomWalk` /
`dRandomWalk` medium-event blocks and `connectBDPT` / `dConnectBDPT` transmittance-weighted
connections already handle spatially-varying σt (they call the same delta/ratio-tracking
helpers the forward tracer uses). No path-budget or MIS changes were required.

**Validation:** `scraps/fogblob.ftsl` (soft-edged density blob, absolute exposure): mode D
GPU vs mode B forward reference — mean 0.04213 vs 0.04247 (−0.8%), centerMean(30%) 0.30009 vs
0.30211 (−0.7%); mode D CPU vs GPU — mean 0.04204 vs 0.04213 (−0.2%), centerMean 0.29972 vs
0.30009 (−0.1%). All within the ~6–9% MC noise floor → unbiased and backend-consistent.

**Optional future variance work (not correctness):** the null-scattering path-integral
formulation (Miller/Georgiev/Jarosz 2019; UPBP, Křivánek et al. 2014) would put the omitted
heterogeneous transmittance *into* the MIS weights, reducing variance in optically-thick
heterogeneous media. Purely a variance optimization — the current estimator is already
unbiased.

### Diffuse-transmission material — CPU DONE 2026-07-12 (GPU port pending)
Added `type translucent` (alias `diffuse_transmit`): a two-sided Lambertian BSDF — the
front hemisphere scatters the `reflect` albedo, the back hemisphere scatters the `transmit`
albedo, so light diffuses THROUGH the surface (soft "waxy"/"paper"/thin-skin look). Because
both lobes are non-specular it renders/connects in **every** CPU mode: forward A/B/C
(`render.h` two-sided `camSplatAll` — flip the normal, wrong side self-rejects), backward
R/V (`backward.h` two NEE calls, one per hemisphere via a normal-flipped `Hit`), and BDPT D
(`bdpt.h` — added to `isConnectibleMat`, `bsdfF`/`bsdfPdf` two-lobe eval, scatter lobe
selection, and — critically — the connection cosine guards in `connectBDPT` now allow the
back hemisphere for two-sided materials via `isTwoSidedMat`, using `|cos|` in the geometry
term with `bsdfF>0` as the real gate; `lambda` is now threaded through
`bsdfPdf`→`vertexPdf`→`misWeight` so the wavelength-dependent lobe-selection pdf is exact).
`reflect`+`transmit` are energy-clamped so their sum ≤ 1. Validated: `scraps/translucent_panel.ftsl`
(backlit warm panel) renders consistently across modes B, R, and D.
**GPU — DONE 2026-07-12.** `render_cuda.cu` now handles `translucent`: `D_DIFFUSETRANSMIT`
(enum aligned to `MatType` with a `D_LAYERED` placeholder), a `DMaterial::transmit[SPEC_N]`
field baked on upload, the two-lobe splat/scatter in the forward `shadeStep` (megakernel +
wavefront share it) and the backward reference `bkRadiance` (GPU mode R). The restricted GPU
BDPT kernel (`kBdpt`) has no two-sided strategy, so translucent scenes fall back to the
validated CPU BDPT via `cudaBdptSupported` (same pattern as frosted glass / textures /
fluorescence). Validated on an RTX 4090: forward B and backward R GPU-vs-CPU RMSE = 3.82/255
(pure MC noise, matching means), and GPU mode D renders the panel correctly through the CPU
fallback.
**Remaining:** a true **BSSRDF / dipole / random-walk subsurface** model (for thick solid SSS
with proper mean-free-path blurring) is still not implemented — this material is a thin
diffuse-transmission approximation, not volumetric SSS.

### Mode `P` composite is not progressive; `R`/`D` have no disk resume — 2026-07-12
The progress/budget unification (`-time`/`-noise`/`-forever`/`-preview`/`-interval`) now
covers the forward camera models (`A`/`B`/`C`) *and* the spp image modes (`R` backward,
`D` BDPT) on both CPU and GPU. Two gaps remain:
- **Mode `P` (composite) is still single-shot.** `renderComposite` (`main.cpp` ~line 1246)
  couples a forward pass (`N` photons) and a backward pass (`spp`) with a **best-fit scale
  `s`** solved once over the diffuse-side pixels, then classifies pixels and blends. Making
  it progressive means chunking *both* passes, re-fitting `s` and recomputing the residual
  each chunk (pixel classification is fixed and can be cached), and reporting the blended
  frame — doable but a real design task, deferred. `-time`/etc. are currently rejected for
  mode `P` with a warning.
- **`R`/`D` accumulate chunks in memory only.** They get live progress and can stop on a
  budget, but there's no `.ftbuf` disk checkpoint, so `-resume`/`-checkpoint` stay
  forward-mode-only. A resumable spp film would need an spp-count checkpoint format
  (the forward one stores a photon count) — proper fix is a small variant of
  `writeCheckpoint`/`readCheckpoint` keyed on spp.

## Resolved

### Unified live progress across all image modes (`R`/`D` join `A`/`B`/`C`) — DONE 2026-07-12
- **What:** modes `R` (backward reference) and `D` (BDPT) previously ran as a single
  monolithic launch with **no progress output, no periodic image write, and no way to stop
  early** — a multi-hour reference render showed nothing until it finished (and a killed
  render lost everything). Now every image-forming mode shares one progress driver: a
  status line (or `-preview` ANSI thumbnail) with a `~noise%` estimate, a periodic
  crash-safe image rewrite every `-interval` seconds, and `-time`/`-noise`/`-forever`
  budgeting with clean Ctrl-C — on both CPU and GPU.
- **How:** `R`/`D` films accumulate a **SUM over samples-per-pixel** (CPU `renderBdpt` was
  changed from ÷spp to SUM to match `renderBackward`/the GPU), so they chunk exactly like
  the forward photon-count films. The GPU kernels (`kBackward`/`kBdpt`) take
  `chunkSpp`/`sppTotal`/`sampleBase` and seed the RNG on the **global sample index**
  (`gidx = pix*sppTotal + sampleBase + local`), so any chunking draws the same union of
  streams as one `sppTotal` pass — **bit-identical** to the old single-shot for a given spp.
  `gpuSppChunks` (device) and `cpuSppChunks` (host, via a `seedOffset` on the CPU renderers)
  own the chunk loop; `runSppProgressive` (`main.cpp`) is the shared reporter, reused by the
  mode-`R` and mode-`D` dispatch. A time/noise/forever budget opens the spp target to a
  capped `UNBOUNDED_SPP=1e9` (keeps `pix*sppTotal` inside int64). New: `render_progress.h`
  (`SppProgress` callback).

### Concurrent GPU renders silently wrote a black PNG (all-black, `auto-exposure=1`) — DONE 2026-07-11
- **What:** running two or more `ftrace ... -device gpu` processes at once could make
  one emit a **completely black** image logging `auto-exposure=1` (the fallback used
  when the 99th-percentile luminance comes back zero), with exit code 0 — silently.
  The symptom was non-deterministic and non-monotonic in spp (e.g. for
  `scenes/mirror_selfie.ftsl`: 512 spp OK, 2048 spp black, 4096 spp OK), depending on
  which renders happened to overlap on the GPU. Re-running the black case **alone**
  renders correctly.
- **Root cause (our bug, not a driver mystery):** `render_cuda.cu` never checked the
  return codes of its ~15 `cudaMalloc` / `cudaMemset` / download `cudaMemcpy` calls
  nor the kernel-launch status on every path. Under contention an alloc or copy fails
  (or a launch returns `unspecified launch failure`), but the code carried on and wrote
  the zero-initialized host film out as a black PNG — no error printed because the
  failing call's status was never inspected. Earlier notes claiming "CUDA still reported
  `cudaSuccess`" were wrong: the errors were there, we just weren't reading them. This
  is process-local — MMU/context isolation means it cannot corrupt another process's GPU
  state.
- **Fix (root cause):** every CUDA call in `render_cuda.cu` is now wrapped in a
  `CUDA_CHECK(...)` macro (checks the returned `cudaError_t`, and on failure prints
  `[cuda] <call> failed at <file>:<line>: <msg>` and `std::exit(EXIT_FAILURE)`), and
  every kernel launch is followed by `cudaCheckKernel(<what>)`
  (`cudaGetLastError` + `cudaDeviceSynchronize`, same loud-exit on error). This covers
  `uploadVec`, the wavefront path, and the forward/BDPT/backward render entries — so a
  failed alloc/copy/launch aborts **before** any framebuffer is downloaded or written.
- **Verified:** built and reproduced contention by running 4 concurrent
  `-device gpu -mode R` renders of `mirror_selfie.ftsl` at 2048 spp — three rendered
  correctly (valid `auto-exposure`), one hit contention and failed loudly with
  `[cuda] backward kernel failed: unspecified launch failure` + exit 1 and wrote **no**
  PNG. No silent black image.
- **Safety net removed:** the earlier `filmIsValid()` gate in `writeFilm`
  (`src/main.cpp`) — which rejected all-zero/NaN framebuffers before tone-mapping — was
  removed now that failures are caught at the source. `writeFilm` still returns a bool
  and callers still propagate a non-zero exit, but only for a genuine image-encoder
  failure. **Tradeoff:** `CUDA_CHECK` catches CUDA-reported errors, not a numerically
  produced NaN that returns `cudaSuccess`; if such a case ever appears it would tonemap
  to black again, and the fix would be a targeted NaN check, not the blanket gate.
- **Residual caveat:** contention still wastes work (one job aborts). Prefer to
  **render GPU jobs one at a time**; the difference is a contended render now fails
  loudly (non-zero exit, no PNG) instead of silently overwriting a good image with black.

### Missing/unknown light spectrum silently fell back to 6500 K white — DONE 2026-07-12
- **What:** an explicit spectrum resource that failed to load rendered the scene with
  a silent default illuminant instead of erroring. `speclib::resolveSpectrumTokens`
  returned `false` for a failed `file:`/`glass:`/`metal:`/`reflectance:`/`illuminant:`/
  `filter:` reference — but `false` also means "not my token, try the next resolver",
  so the failure cascaded to `main.cpp resolveLight`'s `return blackbody(6500.0)`. A
  typo'd path or a `-light` name with no matching preset produced a plausible-looking
  white render with exit 0 — the wrong image, no warning. (This is also why the
  measured-LED presets appeared to "work as white" on a stale binary.)
- **Root cause:** overloaded `false` return (fall-through vs. hard failure) on the
  explicit-prefix branches, plus a catch-all 6500 K fallback for unknown `-light`.
- **Fix:** explicit resource prefixes now **throw** `std::runtime_error` with a clear
  message on load failure (`spectral_library.h`); `resolveLight` throws on an
  unrecognized explicit `-light` name (the built-in `bb6500` default always resolves
  via the parametric path, so only genuine typos trip it); `main()` wraps `run()` in a
  `try/catch` that prints `error: <msg>` and exits 1. Bare, unprefixed names still
  return `false` so the resolver layering (bb<K>, gas-discharge models, illuminant
  lookup) is unchanged. Verified: valid `file:` → exit 0; missing `file:` and unknown
  `-light` → `error:` + exit 1; no-`-light` default → exit 0.

### UV coordinates (`u`,`v`) on native primitives for pattern materials — DONE 2026-07-11
- **What:** the procedural-pattern math VM now exposes the surface texture coordinates
  `u`,`v` (previously mesh-only) to expressions on **native** objects too, so a UV-space
  checker/stripe wraps *around* a sphere/box/isosurface instead of slicing through world
  space. Native `sphere {}` (equirectangular) and `quad {}` (edge-mapped) already filled
  `hit.u/v`; an `isosurface` now accepts `uv planar|spherical|cylindrical [axis=x|y|z]`
  to synthesize a wrap from its world bounds using the **same `projectUV` used for
  un-`vt`'d meshes**.
- **Implementation:** `pattern.h` — added `PatOp::VarU/VarV`, `PatCtx.u/v`, `makePatCtx`
  u/v params, `patternEval` cases, and `u`/`v` in `varOp`. `geometry.h` — hoisted
  `UvProjection`/`parseUvProjection`/`projectUV` out of `mesh.h` (both include geometry.h)
  so implicits reuse them. `implicit.h` — `Implicit.uvProj/uvAxis/uvBounds`; `intersectImplicit`
  projects UV at the hit when enabled. `scene.h` — `patCtxFromHit` threads `h.u/h.v`.
  `ftsl.h` — `addIsosurface` parses `uv <mode> [axis=]`. GPU twins in `render_cuda.cu`:
  `DImplicit.uvProj/uvAxis/uvLo/uvHi`, device `dProjectUV`, `dPatternEval`/`dPatternScalarAt`
  thread `u,v` (and the DF_EXPR field call passes 0,0). Demo: `scenes/uv_native.ftsl`.

### `camera_curve` block (spline fly-through with variable speed) — DONE 2026-07-11
- **What:** a new top-level `camera_curve "name" { point … [frames N] [density <ρ> |
  density_at <t> <ρ> …] [look tangent|look_at|curve+look_point] [closed] [exposure_lock] … }`
  expands into N CamSpec frames whose eye rides a **Catmull-Rom spline** through the
  `point` control points (interpolating — passes through each). Placement is either a
  fixed `frames` count (uniform arc length) or a **density** (cameras per unit length)
  that can vary via `density_at` keyframes — the camera's *speed*: high density = many
  closely-spaced frames = slow dwell; low density = fast. This answers the original
  "how do you specify camera speed as a separate curve" question: density ρ(t) is
  integrated over arc length to a cumulative count C, and camera i is placed by
  inverting C at the target fraction. Orientation: travel tangent (default), a fixed
  `look_at`, or a second `look curve` (its own spline sampled in step).
- **Implementation:** `ftsl.h` `catmullRomAt()` (interpolating spline eval, open clamps
  neighbours / closed wraps) + `addCameraCurve()` (dense arc-length + density sampling,
  cumulative-count inversion for placement, tangent/fixed/curve look) + dispatch entry.
  Reuses the `camera_path`/`camera_orbit` machinery (shared CamSpec, `base<NNN>` naming,
  `pathGroup`/`exposureLock`, multi-camera render loop). Validated on CPU
  (`scraps/curve_test.ftsl`, 3 frames — eye rides the spline, holds the look_at). Same
  GPU caveat as `camera_orbit`: one camera per launch, frames render sequentially (fine).

### `camera_curve` animatable orientation + lens tracks — DONE 2026-07-12
- **What:** `camera_curve` gained the two remaining animatable degrees of freedom it was
  missing — **orientation roll** and **lens properties**. `roll[_at]` banks the camera
  about its view axis (the third orientation DOF beyond eye position and look target);
  `fov_at` / `zoom_at` / `fstop_at` / `focus_at` animate vertical field of view, focal
  multiplier, f-number and focus distance. Each is a keyframe track over the normalized
  timeline `t ∈ [0,1]` (piecewise-linear, flat-clamped at the ends — same idiom as
  `density_at`), or a constant via the bare keyword. Lens *projection*/fisheye stays a
  discrete whole-flight mode (not a continuous track), documented as such.
- **Implementation:** `ftsl.h` — new `ScalarTrack` helper (sorted `{t,v}` keys +
  flat-clamped `sample()`), `rotateAboutAxis()` (Rodrigues) for the roll bank, and a
  static `deriveCameraOptics()` factored out of `readFilmExposure()` so the per-frame loop
  can re-derive focal/fov/aperture/film-distance from the sampled tracks starting from the
  authored base values (no double-apply of zoom). `addCameraCurve()` parses the tracks,
  samples them at each frame's timeline `fr`, re-derives optics when any lens track is
  active, and applies roll to `up` about the final view direction. Demoed in
  `scenes/crystalloop.ftsl` (roll banks into the oval's turns; fov widens for the crystal
  plunge). Note: `fstop`/`focus`/DoF only bite in the physical catch modes (A/C); in the
  pinhole splat mode B the aperture is virtual, so roll/fov/zoom are the visible tracks.

### `camera_orbit` block (turntable / fly-around for MP4s) — DONE 2026-07-11
- **What:** a new top-level `camera_orbit "name" { center radius [height] [axis] frames
  [start_deg] [sweep_deg] [look_at] [exposure_lock] … }` expands into N CamSpec frames
  whose eye rides a circle around `center` (the default look_at), for stitching an orbit
  MP4. A full 360° sweep samples `i/frames` (frame N == frame 0, seamless loop); a partial
  sweep spans endpoints via `i/(frames-1)`. Reuses the `camera_path` machinery (shared
  CamSpec, per-frame naming `base<NNN>`, `pathGroup`/`exposureLock`, the multi-camera
  render loop + `_<name>` file naming).
- **Implementation:** `ftsl.h` `addCameraOrbit` (basis vectors U,W ⟂ axis; eye = center +
  axis·height + (U·cosθ + W·sinθ)·radius) + dispatch entry. Demo: `scenes/showcase_orbit.ftsl`
  (orbit tuned so its circle flies straight through the glass sphere). NOTE: the forward
  splat models A/B share one photon set across all frames (see the shared multi-camera
  entry below), but `-mode R` is camera-anchored (it traces *from* each camera) so an orbit
  on `-mode R`/`-device gpu` renders frames sequentially — which is fine, the per-frame
  cost dominates.

### Isosurface → watertight mesh export (`-export-mesh`) — DONE 2026-07-13
- **What:** any scene's `isosurface`es can be polygonised to an OBJ (`-export-mesh out.obj`)
  for Unreal / Blender import instead of being rendered. `-mesh-res <N>` sets fineness (cells
  along the longest bounds axis); `-mesh-adaptive` / `-mesh-decimate <f>` run a
  curvature-adaptive quadric-error decimation that thins triangles on flat regions and keeps
  them dense where the surface curves. Reuses the renderer's `Implicit::eval`/`gradient`.
- **Implementation:** `src/isomesh.h` (`marchImplicit`, `decimateAdaptive`, `writeObj`);
  CLI + export hook in `src/main.cpp` (~line 2644). Runs on the CPU.
- **Watertightness (proper fix, not a hack):** started on marching **cubes** → left holes /
  non-manifold edges from its face-ambiguous cases. Replaced entirely with marching
  **tetrahedra** (Kuhn/Freudenthal 6-tet split, no ambiguous cases). Surfaces that reach the
  `contained_by` domain box were leaving an **open rim**; fixed by intersecting the field with
  the box SDF (`max(f, boxSDF)`) over a lattice padded 2 cells beyond the box, sealing them
  into flat-capped closed solids (cap normals from central differences of the augmented field).
  Decimation was introducing **non-manifold edges**; fixed with a **link-condition** test
  (collapse only when the endpoints' common neighbours are exactly the shared-face opposites)
  plus foldover rejection.
- **Verification:** heart (genus-0) exports at V−E+F=2, 0 boundary, 0 non-manifold — uniform
  *and* adaptive (keep 30%). Gyroid TPMS shell → Euler −34 (genus-18), csg_mech → −4 (genus-3),
  metaballs → 2, all with 0 boundary + 0 non-manifold edges (Euler correctly tracks genus).
  Round-trip: re-rendering `heart_test.obj` via `-mesh` shows a clean solid heart with correct
  outward normals.

### Arbitrary-formula isosurfaces (`function` leaf, `f(x,y,z)=0`) — DONE 2026-07-11
- **What:** an `isosurface` can now contain a `function { expr "f(x,y,z)" }` leaf that
  renders the zero set of a hand-typed equation (gyroid, Goursat, etc.), distinct from
  the built-in analytic SDF leaves. The formula is compiled by the **same shunting-yard /
  postfix VM as procedural patterns** (`compilePatternExpr`, vars `x y z` and `r=|p|`).
- **Implementation:** `implicit.h` gained a `FieldOp::Expr` leaf (indices a per-`Implicit`
  `exprNodes` PatNode pool via `exprOff/exprN`); `fieldLeafSDF`/`fieldEval`/`fieldGradient`
  thread `const PatNode* exprPool`; new helpers `fieldHasExpr` +
  `estimateFieldLipschitz` (samples `|∇f|` on a 24³ grid over the container box).
  `ftsl.h` `addFunctionLeaf` + rewritten `addIsosurface` parse `function`,
  `contained_by { min max }`, optional `max_gradient` (Lipschitz bound; auto-estimated
  ×1.3 when omitted), and optional `accuracy` (march-step floor). GPU port in
  `render_cuda.cu`: `DF_EXPR` op, `DFieldNode.exprOff/exprN`, a flat device
  `fieldExprNodes` PatNode pool (`DScene::fieldExprNodes`), and `dFieldLeafSDF`/
  `dFieldEval`/`dFieldGradient` thread the pool + call `dPatternEval` for the Expr case
  (forward-declared above the field VM).
- **Why a container box is required:** an arbitrary field is **not** a signed distance
  and has no analytic AABB, so the marcher needs (1) a region to march inside and (2) a
  Lipschitz bound `L ≥ max|∇f|` so a step of `|f|/L` never overshoots the first zero.
- **Validation:** an expression sphere (`x*x+y*y+z*z-0.04`) matches the analytic `sphere`
  leaf to **RMSE 0.37/255 (0.15 %) on the same backend** (the ~12.6 CPU↔GPU RMSE is the
  inherent FP32/RNG divergence — the analytic sphere shows the same 12.58). `scenes/
  function.ftsl` (gyroid) renders correctly on both CPU and GPU. Means match ~1 %.
- **Ray-march strategy selector — DONE 2026-07-11 (follow-up):** any `isosurface` now
  picks how the ray finds the first zero crossing via `method adaptive|sample`,
  `samples <n>`, and `refine bisect|regula_falsi`. `implicit.h` gained `MarchMethod` /
  `RootRefine` enums + `Implicit.sampleStep`; `intersectImplicit` branches the march
  (fixed `sampleStep/dlen` vs the `|f|/lipschitz` adaptive step) and the refinement
  (bisection vs Illinois-safeguarded regula-falsi, tracking both bracket endpoints).
  `ftsl.h` `addIsosurface` parses the three keys (sample step = box diagonal / samples,
  else `accuracy`, else diag/256). GPU twins in `render_cuda.cu`: `DImplicit.method/
  refine/sampleStep` + the identical branch in the device `intersectImplicit`. The
  `adaptive` method (default) provably can't skip the first crossing given a correct
  `max_gradient`; `sample` needs no Lipschitz bound but can miss features thinner than one
  step. Validated: on the clean expression sphere `sample` vs `adaptive` agree to RMSE
  0.41/255 (CPU) / **0.01/255 (GPU)** — identical geometry; regula-falsi and bisection
  land on the same root. Bad `method`/`refine` values are rejected with a clear error.

### `-o foo.png` wrote a PPM (P6), not a PNG — extension was ignored [RESOLVED 2026-07-10]
- **What was wrong:** the image writer (`writePPM` in `src/main.cpp`) always
  emitted binary PPM (P6) regardless of the output extension, so `ftrace -o
  group.png` produced a file starting with `P6\n256 256\n255` but named `.png`.
- **Why it mattered:** anything that trusts the extension mis-handled the file.
  Concretely it softlocked a Claude session: reading the mislabeled `.png` sent
  it to the vision API as `image/png`; the API rejected the PPM bytes (`Image
  format image/png not supported`), and the bad image stayed pinned in the
  conversation, so *every* subsequent request 400'd until a fresh session.
- **Fix:** vendored `stb_image_write.h` (compiled once in `stb_image_impl.cpp`)
  and added `writeImage()`, which dispatches on the output extension — `.png` ->
  PNG, `.jpg`/`.jpeg` -> JPEG (q95), everything else -> PPM (P6). The tone-map
  writer was renamed `writePPM` -> `writeFilm` since it no longer only writes PPM.
  Verified: `-o x.png` / `x.jpg` / `x.ppm` produce correct magic bytes and a
  real `.png` now loads in the vision API without error.

## Limitations (by design, tracked for future work)

### BDPT connection edges through colored glass are not absorption-weighted
- **What:** Beer-Lambert interior absorption (`Material::absorb`, colored glass)
  is threaded through all three CPU transport loops via an `interior` medium
  pointer (forward `tracePhoton`, backward `radiance`, BDPT `randomWalk`). In
  BDPT this attenuates only the **subpath walk** — the camera/light subpaths that
  are traced by ray marching. A **connection edge** (`connectBDPT`, the
  deterministic segment joining a camera vertex to a light vertex) that happens
  to cross a dielectric is treated as unoccluded transmittance = 1, so it picks
  up no absorption tint.
- **Why it matters:** BDPT (mode D) images of scenes with colored glass will be
  slightly biased along light↔eye connections that pass through the glass — the
  glass tints direct-walk contributions correctly but not the connected ones.
  Forward (A/B/C) and backward (R) modes are unaffected (they have no connection
  edges), so the primary/reference renders are correct.
- **Proper fix:** accumulate optical depth along the connection ray by
  intersecting it against dielectric boundaries (or track the medium a connection
  endpoint sits in) and multiply the connection throughput by the resulting
  `exp(-sigma_a*dist)`. Deferred until BDPT-through-glass accuracy is needed.

### VCM (mode `U`): merges use a spectral XYZ estimate; connections through glass share BDPT's absorption gap; CPU-only
- **What:** Mode `U` (VCM/UPS, `src/vcm.h`) is single-wavelength like the rest of the
  renderer. Its **vertex connections** pair a camera subpath with its **own** light
  subpath, so both share one wavelength and the connection is exact (like BDPT). Its
  **merges**, however, gather light vertices from *other* paths (each carrying its own
  sampled wavelength), so — exactly like the photon map (modes `M`/`S`) — the merge builds
  the estimate directly in XYZ using `cie(λ_photon)` and the camera BSDF evaluated at the
  photon's wavelength. This is the standard spectral-photon-mapping approximation, valid
  because this renderer's MIS pdfs are wavelength-independent (diffuse cosine / glossy lobe
  densities don't depend on λ), but it is not a spectrally-exact merge.
- **Why it matters:** For strongly dispersive caustics (wavelength-dependent focusing) the
  merged contribution is approximated in XYZ rather than resolved per-wavelength, just as in
  modes `M`/`S`. Connections remain exact, so the diffuse/glossy portion is unaffected.
- **Also:** VCM connection edges that cross colored glass inherit the same absorption gap
  documented above for BDPT (the deterministic connect segment isn't Beer-Lambert weighted).
  And mode `U` is **CPU-only** — no GPU path yet.
- **Proper fix (if needed):** per-wavelength (hero-wavelength or spectral-bin) merging, and
  optical-depth accumulation along connection rays through dielectrics. Deferred until a
  dispersive-caustic VCM render demands it.

### `.nvdb` volume import (`density vdb:<file>`): dense bake, float-only, uncompressed
- **What:** `medium { density vdb:cloud.nvdb }` imports a NanoVDB FloatGrid (`src/vdbgrid.cpp`,
  the only TU that includes the vendored `NanoVDB.h`). On load the sparse grid is **baked into a
  dense float lattice** covering its active index-space bounding box (`VdbGrid`, `src/vdbgrid.h`).
  A CPU+GPU-shared trilinear sampler reads that lattice.
- **Limitations:**
  1. **Dense memory** — RAM/VRAM scales with the index-space AABB (nx·ny·nz·4 bytes), not the
     active voxel count, so a large but mostly-empty sparse volume can blow up. A safety cap
     (512 M voxels) rejects pathological grids with a clear error rather than OOM-ing.
  2. **Float grids only** — non-float builds (Fp4/Fp8/Fp16/level-set index grids) are rejected
     with a message. Convert to a float fog volume first.
  3. **Uncompressed `.nvdb` only** — Blosc/ZIP-compressed files are rejected (we deliberately
     don't vendor zlib/blosc). Re-export uncompressed (`nanovdb_convert`, or NanoVDB's
     `writeUncompressedGrids`).
  4. **Quoted path not accepted** — the FTSL value grammar takes one bareword token, so the path
     must be unquoted: `density vdb:scraps/cloud.nvdb` (no spaces). A quoted `vdb:"..."` form
     would need a small `parseValue` change to consume a trailing String.
  5. No emission/temperature grids (fire), no motion-blur/velocity grids.
- **Proper fix (if needed):** a native NanoVDB **sparse** device accessor (sample the tree
  directly on CPU+GPU instead of baking dense) to drop the memory cost and support huge volumes;
  fp16 dense option; a second float grid for blackbody emission. Deferred until an asset needs it.

### GPU parity for §1–4 features — DONE (implicits + patterns + translucency)
- **What:** the whole §1–4 CPU feature set is now ported to the GPU forward + backward
  tracers: **implicit surfaces** (5a), **procedural patterns** (5b), and **dielectric
  translucency** (5c — frosted glass = roughness lobe on both dielectric lobes; colored
  glass = Beer–Lambert `absorb` interior tint). The only remaining fallback is **GPU BDPT**
  (mode `D`), whose MIS kernel still can't reproduce per-hit pattern BSDFs or frosted/
  colored glass, so those scenes fall back to the CPU BDPT.
- **Implicit surfaces — DONE (2026-07-11, step 5a):** `render_cuda.cu` gained device
  twins `DFieldNode`/`DImplicit`, a postfix field evaluator (`dFieldEval`/`dFieldLeafSDF`/
  `dFieldGradient`, all FP64 for sphere-trace bisection robustness), and
  `intersectImplicit` (a direct port of the CPU sphere-trace). `buildUpload` flattens
  every `Implicit`'s `FieldNode` array into one device pool and uploads a `DImplicit`
  descriptor per primitive; `closestHit`/`occluded` dispatch BVH prims with index
  `>= nTris+nSph` to `intersectImplicit` (matching the CPU prim ordering). Validated:
  `scenes/implicit.ftsl` (metaballs + drilled CSG + tilted torus) renders on the GPU
  mode-R backward megakernel with GPU-vs-CPU RMSE 9.9/255 at 512 spp — *lower* than the
  cornell baseline (12.7/255) at the same settings, i.e. pure Monte-Carlo noise, no
  implicit-specific bias; mean brightness matches to ~1%.
- **Procedural patterns — DONE (2026-07-11, step 5b):** `render_cuda.cu` gained a device
  pattern VM — `DPattern` slices into a flat `PatNode` pool (`DScene::patNodes`), plus
  `dPatHash3`/`dPatValueNoise`/`dPatternEval`/`dPatternScalarAt`, exact ports of
  `pattern.h` (POD `PatNode`/`PatOp` uploaded verbatim; the field variable `f` is 0 at
  surfaces, matching the CPU). `DMaterial` carries `roughnessPat`/`filmThicknessPat`/
  `mixWeightPat`; `dMatRoughness`/`dMatFilmThickness`/`dMixResolveChild` consult a bound
  pattern (highest priority, above textures). `buildUpload` flattens `Scene::patterns` and
  sets the per-material indices. `cudaForwardSupported()` no longer gates patterns (only
  frosted/colored glass), so the forward + backward paths render them on-device; **GPU
  BDPT still falls back** for any pattern-driven material (`cudaBdptSupported`), because
  its MIS pdf/eval kernel (`kBdpt`) uses the constant params. Validated: `scraps/patval.ftsl`
  (checker/noise `mixWeightPat` spheres + a glossy `roughnessPat` sphere) GPU-vs-CPU RMSE
  12.9/255 at 512 spp → 7.2/255 at 2048 spp (falls as 1/√spp — pure noise, no bias);
  mean brightness matches to ~1%.
- **Dielectric translucency — DONE (2026-07-11, step 5c):** the device `refractOrReflect`
  gained a frosting lobe (jitter both the reflected and refracted directions by a
  power-cosine lobe when per-hit `dMatRoughness` > 1e-3, rejecting jitters that cross to the
  wrong side); `DMaterial` gained a baked `absorb[SPEC_N]` table; and an `interior` medium
  index (the dielectric material a photon/ray is inside, -1 = vacuum) is threaded through
  both forward paths — `shadeStep` (megakernel `kTrace` local + wavefront `WFState::interior`
  SoA slot) — and the backward `bkRadiance`, applying `beta/thr *= exp(-absorb(λ)·dSeg)` over
  each in-glass segment. `cudaForwardSupported()` no longer gates frosted/colored glass, so
  `-device gpu`/`auto` renders them on the forward + backward tracers; **GPU BDPT still falls
  back** (the `frostedOrColoredGlass` gate moved into `cudaBdptSupported`, alongside the
  pattern gate). Validated: `translucency.ftsl` (colored glass) GPU-vs-CPU RMSE 16.9/255 →
  9.5/255 at 512→2048 spp (falls ~1/√spp; mean matches 1.3%→0.7%); `procedural.ftsl` (frosted
  height-banded glass + patterns) RMSE 21.7 → 13.0 at 512→2048 spp (mean matches 0.06%→0.2%);
  forward megakernel vs wavefront agree on mean to 0.15%; BDPT falls back with the correct
  message; `cornell.ftsl` (clear glass) + `implicit.ftsl` still run on GPU (no regression).
- **Status:** DONE — logged 2026-07-11; implicit surfaces (5a), procedural patterns (5b), and
  dielectric translucency (5c) all landed the same day. Full §1–4 GPU forward/backward parity
  achieved; only GPU BDPT retains feature-scoped fallbacks (patterns, frosted/colored glass).

### Multi-camera renders re-trace photons per camera (RESOLVED — shared pass for modes A/B, CPU + GPU)
- **What:** Phase 3a implements multiple named `camera` blocks: one render
  invocation emits one image per camera (`scenes/twocam.ftsl`), with `-camera
  <name>` selection and per-camera film resolution + mode. But each camera is a
  **separate forward pass** — the photon set is re-traced from scratch for every
  camera (`runRender` is called in a loop in `src/main.cpp`).
- **Why it matters:** the wishlist's framing is "many cameras at once… *same
  render for efficiency*". For N cameras this is N× the photon work instead of 1×.
- **CPU shared pass — DONE 2026-07-11.** `tracePhoton` now takes a list of
  `CamTarget{Camera,Film}` and splats each diffuse/emitter/volume vertex to *every*
  camera at once (`camSplatAll`/`camSplatVolumeAll`). Because model-B `connect()` draws
  no RNG, adding cameras never perturbs the photon's RNG stream: the single-camera
  overload is bit-identical to the old path, and an N-camera shared pass reproduces N
  independent single-camera renders exactly. `renderForwardShared()` (src/main.cpp) runs
  one CPU photon trace feeding one film per camera; the multi-camera loop groups the
  eligible cameras (plain `-n`, per-frame auto-exposure) into that single pass and renders
  the rest per-camera as before (the GPU shared pass, below, later removed the CPU-only
  restriction). **Validated:** `twocam.ftsl`
  `-device cpu` shared vs. per-`-camera` solo renders are pixel-identical (max abs diff
  0, both films). The fluoro reradiation λ' is sampled once (camera-independent), and
  mode-A aperture RNG is drawn once, so those single-camera streams are preserved too.
- **GPU shared pass — DONE 2026-07-12.** The forward device code was refactored around a
  `DCamSet` (device pointer to a `DCamera` array + per-camera film/hit buffer arrays +
  `nCam`) that unifies single- and multi-camera tracing, so the ~240-line `shadeStep`
  isn't duplicated (single-camera is just `nCam==1`, bit-identical). `genPhoton`/`shadeStep`
  splat via `splatSurfaceAll`/`splatVolumeAll`; `buildUpload` was split into a scene-only
  bake plus a per-camera `bakeCamera`, and `renderForwardSharedCuda()` (render_cuda.cu)
  bakes the scene once, bakes N cameras, allocates one film/hit buffer per camera, and
  launches a single trace. **Validated 2026-07-12:** GPU model-B shared vs. single-camera
  GPU render pixel-identical (`cmp` clean); CPU model-B shared vs. single also identical;
  the megakernel and wavefront backends both drive the shared pass.
- **Mode A shared pass — DONE 2026-07-12.** Mode A (finite-lens splat) now joins the
  shared pass on both CPU and GPU. Because `connectLens()` draws an aperture sample per
  camera, an N-camera mode-A trace perturbs the RNG stream, so it is **unbiased per camera
  but matches a standalone render in distribution, not bit-for-bit** (validated: shared vs.
  standalone auto-exposure agree to noise). The A- and B-cameras run as **separate** shared
  passes (mode A draws RNG mid-trace, mode B doesn't, so their photon paths diverge). Mode
  C (forward catch) stays inherently per-camera (a photon is consumed by one aperture), and
  the dispatch (`main.cpp`) partitions eligible cameras into A- and B-groups, sharing only
  when a group has ≥2 members.
- **Status:** DONE 2026-07-12 — CPU + GPU shared pass for both forward splat models
  (A and B), validated. Mode C and the camera-anchored modes (R/D/P/V) render per camera
  by construction (documented in README: "Other modes do NOT save time with multiple
  cameras").

### Absolute-EV film sensitivity, non-square films, shared multi-camera pass
- **What (remaining):** three camera/film pieces are still open:
  1. ~~**Absolute EV / physical sensitivity.** `iso`/`shutter`/`exposure` act as a
     *relative* exposure compensation on top of the per-image auto-exposure (the
     film's radiometric scale is arbitrary). A true absolute exposure needs absolute
     light power (watts/lumens) on emitters.~~ **DONE 2026-07-11.** A `light` block
     may author an absolute emitted flux — `power <watts>` (radiometric) or
     `lumens <lm>` (photometric, via Φ_v = 683·∫SPD·V(λ)dλ with cieY as V) — on
     area/sphere/cylinder/spot/collimated lights. The FTSL loader (`absPower()` in
     ftsl.h) scales the emitter SPD so `power = emitIntegral·geomW` equals that flux;
     because photon β and the film accumulation are linear in the SPD, the film
     becomes physically linear. When any light is absolute, `Scene::absolute` is set
     and `writeFilm` swaps the 99th-percentile auto-exposure for a FIXED sensor gain
     (`ABS_EXPOSURE_GAIN`) times the photographic compensation, so scene power flows
     through un-renormalised and iso/shutter/exposure give exact absolute stops.
     Validated on `scenes/absolute.ftsl`: `power 100`→`200` brightens the diffuse
     walls ~2× (light patch clips), the `lumens` path engages absolute mode, and
     non-absolute scenes stay bit-identical (auto-exposure, `Scene::absolute=false`).
     Env lights reject `power`/`lumens` (their phase-space weight needs scene bounds;
     use `intensity`). Not yet metrologically calibrated to cd/m² — the single
     `ABS_EXPOSURE_GAIN` sets the sensor zero-point (tuned so ~100 W in a unit box at
     the neutral triple is mid-tone); relative stops and power ratios are exact.
     **Exposure-lock DONE 2026-07-11:** a `camera_path` can
     author `exposure_lock` (or the CLI `-exposure-lock` forces it across *all*
     rendered cameras) so the auto-exposure anchor is computed once from the first
     frame and reused for the rest — no dolly/zoom flicker. Implemented by an optional
     `double* lockAnchor` threaded `runRender → writeFilm` and a per-lock-group
     `std::map<int,double>` anchor in the multi-camera loop (`CamSpec.pathGroup/
     exposureLock`, `RenderCam.expGroup`); only the final converged write sets/reuses
     the anchor, so progressive intermediate saves don't poison it. Validated on
     `scenes/dolly.ftsl`: unlocked frames swing 2.0e-14→5.2e-13 (25×, visible
     flicker), locked frames all hold the frame-0 anchor 2.02e-14. Standalone cameras
     (no lock) stay bit-identical (null anchor → per-frame auto-exposure as before).
     True *absolute* EV still needs absolute emitter power (deferred, §7).
  2. ~~**Non-square films.** `film { res W H }` only uses the first value; the
     forward/backward tracers (and CUDA) allocate a square film.~~ **DONE
     2026-07-11.** `film { res W H }` (and the CLI `-r W H`) now flow resX≠resY
     through every tracer: CPU/GPU forward (A/B/C), backward (R), BDPT (D), composite
     (P), validate (V), plus checkpoint/resume (the identity guard mixes resY, so a
     mismatched height is rejected instead of silently poisoning the image). The
     camera already carried the true horizontal fov from width
     (`tanHalfX = tanHalfY·rx/ry`); only the render-entry plumbing had collapsed to a
     single square `res`. `renderForward/Backward/Bdpt/Composite`, `runRender`,
     `readCheckpoint`, and the CUDA entry points + `kBackward`/`kBdpt` kernels all
     take resX,resY now. Validated at 320×180: mode V PASSES (bulk RMSE 3%,
     firefly-dominated top-1%), CPU vs GPU auto-exposure agree (4.68e-13 vs 4.62e-13),
     resume accumulates 1M→2M correctly, and the guard rejects a 200×120→200×140
     mismatch.
  3. ~~**Shared multi-camera mode-B pass**~~ (CPU shared pass DONE 2026-07-11 — see the
     multi-camera entry above; GPU shared pass still deferred) — one photon trace
     splatting to every camera pupil.
- **Proper fix:** (1) ~~add a per-`camera_path` exposure-lock flag~~ (DONE
  2026-07-11); ~~absolute emitter power + a sensitometric film model~~ (absolute
  power + fixed-gain exposure DONE 2026-07-11; full cd/m² sensitometry still open).
  (2) ~~thread resX/resY through `renderForward`/`renderBackward`/CUDA and
  `writePPM`~~ (DONE 2026-07-11). (3) see multi-camera (CPU shared pass done).
- **Status:** OPEN (design captured) — logged 2026-07-10; **exposure-lock done
  2026-07-11**, **non-square films done 2026-07-11**, **absolute-EV done
  2026-07-11**, **CPU shared multi-camera pass done 2026-07-11**; only the GPU shared
  pass (and optional mode-A sharing) remains.
- **Done (2026-07-10, Phase 3a):**
  - `camera_path` keyframed motion — expands at load time into a sequence of
    `CamSpec` frames with piecewise-linear `eye`/`look_at` interpolation between
    sorted `key` control points; the multi-camera loop renders the generated list
    with frame-numbered output names. Grammar is numbers-only
    (`key <t> <ex> <ey> <ez> [<lx> <ly> <lz>]`). Validated by `scenes/dolly.ftsl`.
  - Physical film `size <w> <h>` (mm) → focal length `f = filmH/(2·tan(fov_y/2))`
    (metres); `fstop N` → `apertureR = f/(2N)` at load time (overrides `aperture`),
    giving physically-meaningful DOF in modes A/C. `iso`/`shutter`/`exposure` →
    relative exposure compensation `comp = exposure·(iso/100)·shutter` applied over
    the auto-exposure anchor in `writePPM`. Validated by `scenes/expo.ftsl` (ISO 200
    is exactly 2.0× ISO 100 in linear space).

### Fisheye/panoramic lenses: GPU mode-B done; unsupported by BDPT (mode D) and by the finite-lens modes (A/C)
- **What:** `projection <name>` / `fisheye` (equidistant, equisolid, stereographic,
  orthographic) is implemented on the **CPU** for the forward light tracer (modes
  A/B/C), the backward reference (R), and validation/composite (V/P), and on the
  **GPU** for the mode-B pinhole-splat path (see below). The mode-B splat importance
  is projection-correct (the camera computes the per-pixel solid angle
  `Camera::pixelSolidAngle`, replacing the rectilinear `1/(A_pix·cos⁴)`).
- **GPU mode-B fisheye (done 2026-07-11):** the device `DCamera` (`render_cuda.cu`)
  now carries a `projection` enum plus `halfFovY`/`rEdge`, with `HD dProjRadius` /
  `dProjRadiusDeriv` helpers mirroring `Camera::projRadius`/`projRadiusDeriv`.
  `DCamera::project()` branches rectilinear vs fisheye (azimuth + normalised
  `projRadius/rEdge`, no `cz>0` reject), and `pixelSolidAngle()` returns the
  projection-general solid angle (`aNorm·sinθ·rEdge²/(r·dr)`), keeping the
  rectilinear branch bit-identical (`A_pix·cos³`). `connect`/`connectVolume` divide
  by that solid angle. The device path only **splats** (never generates camera
  rays), so no inverse map (`projRadiusInv`) is needed on-device. Fisheye B/V/P now
  run on the GPU (no CPU fallback); validated GPU-vs-CPU on `scenes/fisheye.ftsl`
  2026-07-11 — the equisolid-160° `fish` frame matches CPU at RMSE 3.0/255 (8.8%
  rel, same noise floor as the rectilinear `rect` frame) and mean brightness within
  0.25%. Two gaps remain:
  1. **Finite-lens modes (A/C) reject fisheye.** A thin-lens/aperture camera cannot
     *form* a fisheye projection analytically, so modes A and C error out for a
     non-rectilinear lens (guarded in `src/main.cpp`). A true wide-angle physical
     camera needs the mesh-lens forward-catch mode (see the mesh-lens camera entry).
  2. **BDPT (mode D) rejects fisheye.** `bdpt.h`'s `cameraWe`/`cameraPdfDir` are the
     rectilinear pinhole convention (`1/(A·cos⁴)`, `1/(A·cos³)`) and feed the MIS
     balance heuristic; a fisheye lens there would give subtly-wrong weights, so
     mode D errors out for a non-rectilinear camera rather than lie.
- **Proper fix (future):** generalise the BDPT camera importance + its
  importance-sampling pdf to the projection's Jacobian so the MIS weights stay
  consistent (the pdfDir must match the actual sampling density over the fisheye
  image). The GPU mode-B port is complete.
- **Status:** OPEN (acceptable) — CPU fisheye done + validated (`scenes/fisheye.ftsl`)
  2026-07-11; **GPU mode-B fisheye done + validated 2026-07-11**; BDPT (mode D) and
  finite-lens (A/C) support deferred (the latter belongs to the mesh-lens camera).

### Physical (realistic) lens camera — backward realistic-camera formulation [IMPLEMENTED 2026-07-11]
- **What (done):** a camera can now carry a real **lens prescription** — a stack of
  spherical/planar refracting interfaces plus an aperture stop (`src/lens.h`,
  `LensSystem`). The backward reference tracer (mode R) samples a film point and a
  point on the rear element, traces that ray *through the actual glass interfaces*
  (per-wavelength Snell refraction, so dispersion → chromatic aberration is
  automatic) out into the scene, then path-traces. Depth of field, distortion,
  spherical aberration, coma, field curvature and **vignetting** (clipped/TIR rays
  contribute nothing) all emerge from the geometry — no thin-lens or projection
  model. Survivors carry a PBRT-style radiometric weight (cos⁴θ·A_rear/Z_rear²).
  Wired via an FTSL `camera { lens { … } }` block (`readLens` in `src/ftsl.h`);
  a physical lens forces the camera to mode R (`src/main.cpp`). Autofocus shifts the
  film plane with a paraxial probe (`focusAt`). Demo: `scenes/realcam.ftsl`
  (validated: the focus-plane sphere is sharp, near/far spheres blur; the `singlet`
  preset visibly softens from spherical aberration).
- **Presets & generators (`src/lens.h`):** `makeSinglet` (biconvex, lensmaker
  R=2(n−1)f), `makeAchromat` (cemented crown+flint doublet, powers split by Abbe
  numbers to cancel first-order CA); `resolveLensPreset` names: `singlet`/`biconvex`,
  `achromat`/`doublet`, `telephoto`, `wide`. All physically derived (not fabricated
  data), so focal length + achromatisation are correct by construction; dispersion at
  render time uses the real Sellmeier glass indices. Users can also paste an arbitrary
  real prescription as repeated `surface <radius_mm> <thickness_mm> <ior> <semi_ap_mm>
  [stop]` lines (PBRT lens-file convention: +radius ⇒ centre of curvature on the scene
  side; lens works in millimetres, scene in metres).
- **Sign-convention gotcha (fixed):** the geometry stores curvature as `centre =
  vertex + radius` with +z toward the scene (identical to PBRT's
  `IntersectSphericalElement`). The lensmaker/achromat generators emit radii in the
  opposite object→image convention, so their radii are **negated** at construction
  (see comments in `makeSinglet`/`makeAchromat`). Without the negation the doublet
  *diverges* and the autofocus places the sensor on the scene side (all rays miss) —
  the first-cut symptom was a fully black image.
- **Remaining gaps (OPEN, deferred):**
  1. **Backward-only.** No forward-catch (mode C-style) or forward-splat (A/B)
     realistic-lens path yet. A physical lens always renders in mode R. **GPU: DONE**
     (Plan A, 2026-07-11) — a dedicated GPU backward megakernel (GPU mode R) runs the
     physical lens as a ray-generation front-end (`kBackward` in `render_cuda.cu`, the
     lens bit-for-bit ported to `dGenLensRay`/`dLensTrace` with per-surface sensor-side
     ior baked into a device table). `-device auto`/`gpu` selects it. v1 scope
     (`cudaBackwardSupported`): no participating media, no environment light, no
     spot/collimated emitters, no fluorescence (all fall back to the CPU backward
     tracer), and ≤ `D_MAXLENS` (16) lens surfaces; textured albedo IS supported. The
     device RNG differs from the CPU, so the image is an independent noise realization
     that agrees to within Monte-Carlo noise.
  2. **Sensor mapping.** `genLensRay` maps the sensor width across the film width and
     derives the vertical extent from the output pixel aspect. Now that the film
     pipeline carries resX≠resY (non-square films, DONE 2026-07-11), rendering the
     physical lens into a film whose aspect matches the sensor (e.g. 3:2) covers the
     sensor with square pixels and no crop; a mismatched aspect still crops as before.
  3. **No inter-element flare/ghosting** (rays refract, they don't also partially
     reflect at each interface), **no enclosure/body geometry**, and the aperture is a
     circular clear-diameter clip (no shaped-iris bokeh).
  4. ~~**Not in BDPT (mode D).**~~ **DONE 2026-07-11 (Plan B, below).** The lens now
     also rides on the BDPT camera subpath (mode D), and mode P routes to it.
- **Status:** IMPLEMENTED (backward realistic camera, 2026-07-11; GPU backward
  megakernel / Plan A, 2026-07-11; **BDPT camera-subpath lens / Plan B, CPU + GPU,
  2026-07-11**). Supersedes the analytical thin-lens for "arbitrary real camera" use;
  the remaining gaps above (inter-element flare, shaped-iris bokeh) are follow-ups.

#### Plan B — realistic lens on the camera subpath of BDPT (mode D) and composite (mode P) [DONE 2026-07-11]
- **Why it's wanted:** the physical lens currently rides on **pure mode R** (backward
  everything), so it inherits mode R's weaknesses — noisy on in-frame caustics, and no
  fluorescence. The lens only ever lives on the *camera subpath*; in principle you can
  keep forward light transport lighting the scene (caustics on surfaces) while a
  backward lens ray samples that lit scene through the glass — i.e. attach the lens to
  the camera subpath of a bidirectional/composite estimator instead of forcing pure R.
  That would recover *some* of the forward tracer's caustic efficiency while keeping the
  physical optics.
- **The catch (why it's only a partial win, and deferred):** the multi-element lens map
  has **no closed-form inverse**, and both D and P need that inverse for the parts that
  would buy the forward advantage:
  1. **BDPT (mode D).** BDPT's power comes from light→camera connection strategies. The
     **t=1 strategy** (splat a light-subpath vertex directly onto the film) requires
     projecting a world point onto the sensor *through the glass stack* — the lens
     inversion. PBRT disables the camera-connection strategies for realistic cameras for
     exactly this reason. So a realistic lens in D must run with t=1 disabled: you keep
     the *scene-side* connections (a camera-subpath vertex out in the scene connects to
     light-subpath vertices), which recovers part of the caustic efficiency, but not the
     full forward win. It's a substantial, delicate, per-wavelength change layered on a
     mode D that already lacks fog / env / spot / fluorescence support.
  2. **Composite (mode P).** Worse fit: P's forward pass **splats to a pinhole** — it
     fundamentally assumes a pinhole camera. Routing that forward pass through a physical
     lens is the ill-posed forward-through-lens problem again. So P does not cleanly
     extend to a realistic lens without solving the same inversion.
- **Bottom line:** the clean, buildable step was **Plan A** — a dedicated GPU backward
  megakernel (GPU mode R) with the lens as a ray-generation front-end (**DONE**
  2026-07-11; see gap #1 above). Plan B (D and P) is genuinely more general but only a
  *partial* caustic recovery, and can't do the light→film splat through the glass.

- **DONE 2026-07-11 (the resolution that made Plan B clean and rigorous):** the t=1
  light-image splat is simply **disabled** for a lensed camera (no closed-form lens
  inverse), and the key realisation is that this needs **no lens direction-pdf** at all:
  1. `generateCameraSubpath` (`src/bdpt.h`): when `cam.hasLens()`, the first camera ray
     is generated by `Camera::genLensRay` (film point + rear-pupil point traced through
     the real glass, identical to mode R). The camera vertex sits at the ray's
     **scene-entry point** with `beta = wLens` (the lens radiometric weight), so a pure
     eye path measures `L·wLens` — matching mode R's film `add`. The camera vertex is
     flagged **`delta = true`**.
  2. `connectBDPT` t==1 branch returns 0 for `cam.hasLens()` (the splat needs the
     sensor projection we don't have). The retained strategies are the scene-side
     connections (s≥0, t≥2: pure path trace, NEE, interior light↔eye), which keep the
     forward tracer's caustic efficiency through the physical lens.
  3. **Why the lens direction pdf is unnecessary (the rigorous part):** `misWeight`'s
     camera loop runs `i = t-1 … 1` and only *adds* a hypothetical strategy's term when
     `!eye[i].delta && !eye[i-1].delta`. The i==1 term (the t=1 splat) is gated on
     `!eye[0].delta` — which is now false — so it is **excluded from the MIS sum**, and
     since the loop never reaches i==0, `eye[1].pdfFwd` (the only place the camera
     direction density would enter) never affects any *retained* ratio. The surviving
     strategies therefore still form a **partition of unity** ⇒ the estimator is
     **unbiased** regardless of the (unused) camera-ray direction pdf. `eye[1].pdfFwd`
     is seeded with the pinhole `cameraPdfDir` purely as an inert placeholder.
  - **Mode P (composite):** its forward pass splats to a **pinhole** and can't form the
    lens image, so a lensed camera in mode P **routes to the lens-aware BDPT (mode D)**
    when the scene is within BDPT scope, else **falls back to mode R** (fog / env / spot /
    fluorescence / layered — which R supports and D doesn't). Wired in `src/main.cpp`
    (`bdptUnsupportedFeature` helper shared by the mode-D gate and the P routing).
  - **GPU: DONE 2026-07-11.** The GPU BDPT megakernel (`kBdpt`) now takes the lens on its
    camera subpath too: `dGenCameraSubpath` generates the first ray via `dGenLensRay` (the
    same device lens tracer Plan A ported for GPU mode R), sets the camera vertex `beta =
    wLens` and `delta = 1`, and `dConnectBDPT`'s t==1 branch returns 0 for a lensed camera
    — a bit-for-bit mirror of the CPU path. The DCamera lens is already uploaded by
    `buildUpload`. So `-mode D -device gpu` on a lensed scene runs entirely on-device; the
    old CPU-force guard in `src/main.cpp` is removed.
  - **Validation** (`scenes/realcam.ftsl`, achromat 50 mm f/2.8, full-frame): mode D vs
    mode R with the same lens agree on absolute radiance and the residual is **pure Monte-
    Carlo noise**. CPU-D↔GPU-D↔R all agree (median per-pixel ratios 0.987–1.010 @ 256–512
    spp). Unbiasedness: high-spp GPU-D vs R gives median **1.0003** with the ratio IQR
    narrowing [0.905,1.096] → [0.967,1.035] as spp goes 512 → 8192 (≈√16 narrowing), and
    the auto-exposures converge (GPU-D 1.07e-11 vs R 1.08e-11). No bias, CPU or GPU. Mode P
    lens routing (→ D) and out-of-scope fallback (→ R, tested with `-fog`) both verified;
    lensless mode D/P unchanged (cornell regression).
  - **Remaining follow-up:** the true t=1 splat through an approximate lens inverse
    (PBRT-style exit-pupil sampling) for the extra light-tracing strategy — optional, since
    scene-side connections already recover the main forward win.
- **Status:** DONE 2026-07-11 (CPU + GPU BDPT + composite routing).

### Texturing is base-color only, `use_mesh`/quad UVs only (Phase 3b partial)
- **What (done 2026-07-10):** a `texture "name" { file … encoding srgb|linear
  filter nearest|bilinear wrap repeat|clamp|mirror }` block loads an image into
  `Scene::textures` (`src/texture.h`); `reflect texture:<name>` on a `diffuse`
  material binds it (`Material::reflectTex`); per-vertex UVs live on `Tri`
  (`src/geometry.h`), auto-generated for quads and read from OBJ `vt` when a mesh
  sets `uv use_mesh`; each texel is Jakob-Hanika–upsampled to a reflectance
  spectrum (coefficients precomputed at load via `Texture::buildReflCoeff`, bilerped
  + sigmoid-evaluated per hit through `diffuseReflectance()`). Shared by the forward
  tracer and the backward reference. Image formats: PNG/JPG/BMP/TGA + Radiance
  `.hdr` via the vendored stb_image (`src/third_party/stb_image.h`, compiled once in
  `src/stb_image_impl.cpp`), plus built-in PPM/PFM. Validated by
  `scenes/textured.ftsl` (quad) and `scenes/uvmesh.ftsl` (mesh) — the checker maps
  with correct orientation (blue band at v≈1/top, yellow at u≈0/left) and spectral
  colour; a PNG copy of the checker renders bit-identically to the PPM (RMSE 0.0),
  confirming the stb sRGB decode + orientation.
- **Remaining [needs engine work]:**
  1. **UV projections.** ~~Only `uv use_mesh` (OBJ `vt`) and quad corners exist.~~
     **PARTLY DONE 2026-07-11:** the analytic projections `uv planar|spherical|
     cylindrical [x|y|z]` (spec §9.2) are now synthesized at load time from the
     world-space vertex AABB (`UvProjection`/`projectUV` in `src/mesh.h`, parsed in
     `src/ftsl.h`), normalised to [0,1] across the mesh so the map wraps once by
     default. Because they fill the same per-vertex `Tri.uv{0,1,2}` slots as
     `use_mesh`, both tracers **and the GPU** interpolate them with no shading change
     (validated on `torus.obj`, which carries no `vt` — the checker maps onto the
     torus via the spherical projection; `scraps/uvproc.ftsl`). **DONE 2026-07-11:**
     `triplanar` — unlike the three analytic maps it can't be baked into per-vertex
     UVs (it blends three world-axis planar samples per hit, weighted by |n|^4), so it
     lives on the bound material as `Material::triplanarScale` (world->texture repeat
     rate) and is evaluated per hit in `Texture::reflectanceTriplanar`, called from
     `diffuseReflectance()` (CPU, shared by forward/backward/BDPT) and the device twin
     `dTexReflTriplanar` from `dDiffuseRho()` (GPU) — the two agree by construction.
     Parsed from `mesh { uv triplanar [<s>|scale=<s>] }` in `src/ftsl.h`. Validated by
     `scenes/triplanar.ftsl`: CPU vs GPU exposures match to 3 digits (4.87e-13 vs
     4.88e-13) and the box-projected checker is visually identical on both backends.
     **Parser gotcha fixed:** the scale/axis argument must be a bare number or a
     `key=val` param — a bareword `scale`/axis letter starts a *new* statement and
     clobbered the mesh's own `scale` transform (this caused an all-black render while
     the torus ballooned to 4x and occluded the box; the same fix now applies to
     `uv planar axis=x`).
  2. **Non-albedo parameters.** ~~A texture can only bind to diffuse `reflect`.~~
     **DONE 2026-07-11 (roughness + film-thickness maps, both backends):** `glossy`
     takes `roughness texture:<name>` (grayscale = roughness directly) and `thinfilm`
     takes `film_thickness_map texture:<name>` (0..1 profile × nominal `film_thickness`
     nm). Bound in `src/ftsl.h` via `bindScalarTexture`; sampled per-hit by
     `materialRoughness`/`materialFilmThickness` (`src/scene.h`) → `Texture::scalarAt`
     (mean of linear RGB). All three CPU tracers (forward/backward/BDPT) and the GPU
     forward path (megakernel + wavefront, via `dMatRoughness`/`dMatFilmThickness` +
     `dTexScalarAt` over an uploaded per-texel `gray` array) use it. **MIS
     correctness:** the CPU BDPT threads the hit UV through `bsdfPdf`/`bsdfF` so the
     sampled and evaluated roughness match; the GPU BDPT does not, so `cudaBdptSupported`
     rejects roughness/thickness-map scenes → CPU BDPT fallback. Validated by
     `scenes/scalarmap.ftsl`: CPU vs GPU forward exposure 7.3e-13 vs 7.29e-13, mean
     agrees to <0.1%, signed diff ~0.04% (unbiased). **Also DONE 2026-07-11 (mix
     blend-mask):** a 2-child `mix` takes `weight_map texture:<name>` — the map value t
     at the hit is the probability of child 0 (child 1 = 1-t, no absorption), a spatial
     A/B blend. `Material::mixWeightTex` + `mixResolveChild` (scene.h), threaded through
     all three CPU tracers and the GPU forward path (`dMixResolveChild`). Mix selection
     is a stochastic RR pick that doesn't enter the BSDF pdf, so it's unbiased in every
     tracer; the GPU BDPT mix-pick still uses constant weights, so masked mixes take the
     CPU-BDPT fallback (`cudaBdptSupported`). Validated by `scenes/maskblend.ftsl`.
     **Still deferred:** a map on `ior` (spatially-varying refractive index — rare, and
     better served by measured dispersion data than a grayscale map; low priority).
  3. ~~**GPU.** Textured scenes force the CPU tracer.~~ **DONE 2026-07-11:** the
     forward CUDA path now ports textured diffuse reflectance. `buildUpload()` uploads
     each texture's per-texel Jakob-Hanika coeff table (`DTexture`, flattened `3*w*h`)
     plus per-tri UVs (`DTri.uv0/1/2`); `intersectTri`/`intersectSphere` interpolate
     the hit UV into `DHit.u/v`; `dTexReflAt()` is the device twin of
     `Texture::reflectanceAt` (wrap + nearest/bilinear + sigmoid), used via
     `dDiffuseRho()` in the `shadeStep` diffuse/fluoro branches. `cudaForwardSupported()`
     no longer rejects `reflectTex >= 0`. Validated GPU-vs-CPU on `textured.ftsl` and
     `uvmesh.ftsl`: energy matches (absorbed 0.7066 vs 0.7068) and images agree to
     within Monte-Carlo noise (RMSE ~6/255); the wavefront backend matches the
     megakernel. The BDPT kernel (mode D) still lacks a textured vertex, so
     `cudaBdptSupported()` explicitly rejects textured scenes → they use the CPU BDPT.
  4. **Indexed-spectral palettes** (§9.3). ~~An index image + name→spectrum palette —
     not implemented.~~ **DONE 2026-07-11 (CPU):** a `texture { ... palette { <idx>
     spectrum:<name> ... } }` block resolves each index to a named reflectance spectrum
     at parse time (`Texture::palette`, `src/ftsl.h addTexture`). The red channel,
     quantized to 0..255, selects an entry per texel — nearest only (indices are
     categorical, never bilerped) via `Texture::paletteReflectanceAt`. No JH upsampling
     (palette entries are arbitrary measured spectra used directly), so `buildReflCoeff`
     skips palette maps and the GPU forward path (`cudaForwardSupported`) rejects them →
     CPU fallback. Validated by `scenes/palette.ftsl` (a 4-index swatch chart). **Limit:**
     8-bit index channel → ≤256 entries; 16-bit index maps are future work.
- **Status:** OPEN (acceptable) — base-color texturing + stb image import done
  2026-07-10; GPU port done 2026-07-11; **analytic UV projections (planar/spherical/
  cylindrical) + triplanar box projection done 2026-07-11 (CPU + GPU)**; **non-albedo
  roughness + film-thickness maps + mix blend-mask done 2026-07-11 (CPU all tracers +
  GPU forward; GPU BDPT falls back to CPU)**; **indexed-spectral palettes done 2026-07-11
  (CPU)**. Only an `ior` map (item 2) remains deferred (rare; low priority).

### Light shapes: sphere + spot done, HDRI environment deferred (Phase 3c partial)
- **What (done 2026-07-10):** two new emitter shapes on the shared `Emitter`
  (`src/scene.h`), both routed through forward + backward + CUDA:
  - **Spherical area light** — `light sphere { center x y z  radius r  spd … }` —
    `shape = EmitterShape::Sphere`, `area = 4·π·r²`. Forward and backward both call
    `Emitter::samplePoint()` (uniform surface point + outward normal; quad draws are
    byte-identical so quad scenes stay bit-identical). The FTSL loader also drops an
    emissive sphere into the geometry (mirroring the area-light quad). Validated by
    `scenes/spherelight.ftsl` (mode V PASS at 60M photons / 1024 spp; CPU==GPU).
  - **Point spotlight** — `light spot { origin … dir … inner_angle d  outer_angle d
    spd … }` — `shape = EmitterShape::Spot`, a cone with a cubic-smoothstep
    penumbra. Geometric weight is the falloff-weighted solid angle `spotOmega =
    π·(2−cosᵢ−cosₒ)`, so `power = emitIntegral·spotOmega`, peak intensity/SPD = 1.
    Forward samples a direction uniformly in the outer cone and reweights beta by
    `falloff·Ω_outer/spotOmega`; backward connects straight to the light point with
    a cone-falloff weight (`I(ω)·cosθ/d²`, no area/light-cosine). No emissive
    geometry (a point). Validated by `scenes/spotlight.ftsl` (mode V PASS at 200M
    photons / 4096 spp → 3.8% RMSE; CPU==GPU energy).
  - `finalizeEmitters()` now keys the power law / combined wavelength sampler off a
    per-emitter `geomWeight()` (area·PI for surfaces, spotOmega for spots); the
    area/sphere branch keeps the exact `emitIntegral·area·PI` expression so those
    renders stay bit-identical (verified: cornell FTSL==C++==pre-3c hash).
- **Constant environment done (2026-07-10, increment 1a):** `light env { spd … }`
  registers a uniform infinite emitter (`shape = EmitterShape::Env`,
  `geomWeight = envGeom = 4·π²·R²` with `R` the scene bounding-sphere radius set in
  `Scene::build()` from the BVH root AABB). Forward emission spawns each photon from
  a disk of radius `R` perpendicular to a uniformly-sampled sphere direction (joint
  pdf `1/envGeom` → exactly analog `beta = emitIntegral·envGeom`); backward adds
  `L(λ)·invPdfλ` on ray-miss; a per-pixel background pass (`addEnvBackground`) supplies
  the directly-viewed sky in forward mode B. Validated by `scenes/envlight.ftsl`
  (mode V: forward converges to backward on a **unit** radiance scale — best-fit
  s→1).
- **GPU constant env done (2026-07-10, increment 1b):** the device forward kernel
  now emits env photons (`DEmitter::shape == 3`) from the scene bounding sphere
  (`DScene::sceneCenter`/`sceneRadius`) exactly like the CPU path, and the
  directly-viewed sky is added by the backend-agnostic `addEnvBackground()` pass in
  `main.cpp` — so `cudaForwardSupported()` no longer rejects `envIndex ≥ 0` and env
  scenes run on the GPU. Verified CPU==GPU on `envlight.ftsl` mode V (both best-fit
  s≈0.97, absorbed≈0.129, RMSE≈58% at 8M — deltas are independent RNG streams).
  - **Absolute-radiance We fix (same change):** the model-B pinhole importance was
    normalizing by the *whole* image-plane area (`imagePlaneArea()`), making the
    forward tracer measure `radiance / (resX·resY)` — an arbitrary global constant
    that modes V/P best-fit away and auto-exposure hid. This blocked compositing the
    (true-radiance) env background with the (scaled) photon surface illumination.
    Fixed by normalizing by the **per-pixel** image-plane area
    (`Camera::pixelPlaneArea() = imagePlaneArea()/(resX·resY)`) in `connect()` /
    `connectVolume()` on **both** CPU (`render.h`) and GPU (`render_cuda.cu`). Now
    forward measures absolute radiance (mode V/P best-fit s → ~1). Displayed outputs
    are unchanged (a global scale is invisible after auto-exposure; mode-P
    `fwd·invF/s` and mode-V RMSE are scale-invariant); verified cornell mode V still
    PASSes (s 5.8e-5 → 0.98) and CPU==GPU film scale holds.
  - **Forward env is high-variance (acceptable limitation):** the env photon
    emission is isotropic over 4π, so in an open scene the vast majority of photons
    escape without hitting geometry (~87% on `envlight.ftsl`). Combined with
    single-wavelength spectral spikes, forward mode-B env images are heavily
    chromatic-noisy and need large `-n` to converge (mode V RMSE falls as 1/√N with
    s≈1 — variance, not bias: 58%@8M → 27%@60M). Clean env images come from the
    **backward** reference (mode R). A future variance reduction would importance-sample
    the emission toward the actual geometry (not just the bounding sphere) and/or
    trace multiple wavelengths per photon (hero-wavelength); deferred.
  - **Mode P + env: sky background — DONE 2026-07-11.** The directly-viewed sky is
    now composited in `renderComposite()` (mode P), not just modes B/V. The pixel
    classifier became three-way — SPEC (specular-first → backward layer), SKY (camera
    ray escapes an env scene → env radiance) and DIFF (everything else → forward
    layer) — and the env radiance (`envXYZForDir`, already in the composite's
    display-radiance units) is written on SKY pixels. Critically, **SKY pixels are now
    excluded from the forward→backward scale fit**: they are measured by env radiance
    directly (forward film ≈ 0, backward film = full bright sky), so including them
    dragged the best-fit `s` toward 0 — exactly the bias mode V avoids by adding the
    sky to `fwd` before its `compareFilms` fit. Verified on `envlight.ftsl` mode P:
    excluding sky pixels restores s 0.27 → 0.957 (matching mode V's ~0.97), the sky
    renders behind the geometry, and a non-env specular scene (`group.ftsl`) is
    unaffected (s 0.965, no env line, DIFF/SPEC split unchanged).
- **Image-based HDRI environment (2026-07-10, increment 2a — DONE, CPU):** `light env
  { file "sky.hdr"  rotate deg  intensity s }` registers an equirectangular (lat-long)
  environment. `src/envmap.h` (`EnvMap`) loads the map (via the existing `Texture`
  loader — `.hdr`/`.pfm`/LDR), upsamples each texel to a physical emission spectrum
  `L(λ) = scale·S_JH(chroma)(λ)·illum(λ)` (Jakob-Hanika chroma fit × normalized 6504 K
  illuminant, PBRT RGB-illuminant convention; `scale` carries HDR brightness), and
  builds a 2D luminance CDF (`Distribution2D`: marginal rows × conditional cols,
  `sin θ`-weighted) for importance-sampled directions. Wired through `Scene`
  (`envMap` shared_ptr, `addEnvLight(map)`, direction-dependent `envRadiance(dir,λ)` /
  `envXYZForDir(dir)` / `sampleEnvDir`), `render.h` (forward emission importance-samples
  the direction and reweights the flat power by `L(dir,λ)/(4π·pdf_ω·meanSpd(λ))` — a
  factor that is exactly 1 for a constant env, so those stay bit-identical), `backward.h`
  (miss term uses the escape direction), `main.cpp` (`addEnvBackground` uses per-texel
  spectral XYZ), and `ftsl.h` (`file`/`rotate`/`intensity` parse). The emitter power +
  wavelength CDF use the map's `sin θ`-weighted mean radiance spectrum. Validated by
  `scenes/envmap.ftsl` + `scenes/sky.pfm` (mode V: best-fit s→~0.95 and climbing with
  samples — the residual is Monte-Carlo variance from the sun glow, not bias;
  forward/backward auto-exposure agree to ~3%; energy conserves). Constant env
  (`envlight.ftsl`) stays **bit-identical** (mode-V scale 0.971252, unchanged).
  - **Increment 2b — DONE (2026-07-10):** backward env **NEE** at every diffuse and
    fog-scatter vertex (`neeEnv`/`neeEnvVolume` in `backward.h`): sample `ω` from the
    map's luminance CDF, shadow-ray past the scene bounds, and **MIS-combine** (balance
    heuristic) with the BSDF-sampled continuation that reaches the sky on a ray miss.
    The miss term is added at full weight only on a camera/specular arrival and MIS-
    weighted otherwise (gated on `specularArrival`), so nothing is double-counted;
    `envMap->pdf(d)` provably equals `sample()`'s reported pdfW, so the weights sum to
    1 (unbiased — verified: `envmap.ftsl` mode-V scale stays ~0.947 at 60M/512, energy
    conserves, residual broadly distributed). All env-NEE work is gated on
    `scene.envIndex >= 0`, so non-env scenes keep a **bit-identical** RNG stream /
    backward image (cornell mode V unchanged).
  - **Increment 2c — DONE (2026-07-10):** GPU port of the lat-long sampler
    (`render_cuda.cu`). The host flattens the EnvMap into device buffers — per-texel JH
    `coeff`/`scale`, the mean `avgCoeff`/`avgScale`, and the 2D luminance CDF (marginal
    `Distribution1D` over rows + one conditional per row) — and the device gets
    `dReflAt`/`dSample1D`/`dEnvSample`/`dEnvTexel` so the `shape==3` emission branch
    importance-samples the map and reweights beta by `L(dir,λ)/(4π·pdfW·avgSpd)`. The
    reweight's shared illuminant cancels in `L/avgSpd`, so no illuminant table is
    uploaded. `cudaForwardSupported()` now returns true for image env; the constant-env
    device path is untouched (`sc.env.scale == nullptr`). Verified: GPU vs CPU forward on
    `envmap.ftsl` agree — energy conserves (sum/emitted=1.0, escaped 0.8893 vs 0.8894),
    mean RGB within ~0.5%, auto-exposure 53.5 vs 53.8; constant env + all other GPU
    scenes unchanged.
- **Deferred (still future):**
  1. **HDRI env** — image-based environment lighting (`light env { file … }`) is fully
     done: increments 2a (CPU forward + backward miss/background), 2b (backward env-NEE
     with MIS), and 2c (GPU forward port) are all complete. Original 7-step plan below,
     all steps done, kept for reference:
     **Concrete plan (each sub-step
     independently
     buildable + validatable):**
     1. *Scene bounding sphere.* Add `Vec3 sceneCenter; double sceneRadius;`
        computed in `Scene::build()` from the BVH root AABB (`center`, `0.5·diag`).
        The env disk/emission and the "to infinity" shadow-ray length key off this.
     2. *Env data + importance sampler.* Store the lat-long map as linear RGB +
        per-texel JH coeffs (reuse `Texture`). Precompute a 2D luminance CDF
        (marginal over rows, conditional over columns, each row weighted by
        `sin θ`) → `sampleEnvDir(u1,u2, pdfω)` and `envPdf(ω)`. `envRadiance(ω,λ)`
        = `reflAt(coeff(ω), λ)` scaled by an intensity factor.
     3. *Backward first (easiest to validate in mode R).* In `radiance()`, on
        `!h.valid` return `L + thr·envRadiance(ray.d,λ)·invPdfLambda` when
        `specularArrival` (direct/mirror view of the sky). Add env NEE at diffuse
        vertices: sample `ω~envPdf`, shadow-ray to `sceneRadius`, add
        `f·envRadiance(ω,λ)·cosSurf·invPdfLambda/envPdf(ω)`. Fold the env into the
        combined wavelength sampler `g(λ)` (its geomWeight ≈ `π·sceneRadius²·avgLum`).
     4. *Forward emission.* New branch in `tracePhoton`: `shape == Env` emits a
        photon FROM the sky — importance-sample `ω~envPdf`, pick a point on the disk
        of radius `sceneRadius` perpendicular to `-ω` tangent to the bounding sphere,
        fire along `-ω`, `beta = envPower · envRadiance(ω)/(avgLum·envPdf(ω))`.
        `envPower = π·sceneRadius²·∫envRadiance dω` feeds the selection CDF like any
        other emitter's `power`.
     5. *Mode-B background.* In forward mode B a camera ray isn't traced, so the sky
        isn't directly visible via `connect()`. Add a per-pixel background pass:
        for each pixel, project the pinhole ray, and if it escapes, splat
        `envRadiance(dir,λ)` (spectrally integrated) — a cheap deterministic add,
        analogous to the existing direct-emitter `connect`.
     6. *CUDA.* Upload the env RGB+coeff tables + the marginal/conditional CDFs;
        port `sampleEnvDir`/`envPdf`/`envRadiance` and the forward disk-emission
        branch. Escaped photons already terminate; only emission + (optional) the
        mode-B background pass need device code. Keep `cudaForwardSupported()`
        returning true for env scenes once ported (else fall back to CPU).
     7. *Validation.* `scenes/envlight.ftsl` (a diffuse box open to the sky):
        mode V forward-vs-backward RMSE < 5%; energy conserves; CPU==GPU. A constant
        (single-colour) env is the smallest first milestone — it exercises steps
        1/3/4/5/6 with a trivial step 2 (uniform pdf), so land that before the full
        image-based 2D CDF.
  2. **Sphere-light importance sampling.** *(DONE 2026-07-10.)* The backward
     reference's sphere NEE now does cone/solid-angle importance sampling of only
     the visible cap toward the receiver (`Emitter::sampleSphereCone`, PBRT's
     `Sphere::Sample`): sample `cosθ` uniformly in `[cosθmax, 1]` about the
     centre-to-point axis (`sinθmax = r/dc`), find the near intersection, and weight
     in solid-angle measure with `pdfW = 1/(2π(1−cosθmax))` — so no draws land on
     the far, self-occluded, back-facing hemisphere. A receiver inside the sphere
     (`dc ≤ r`) falls back to uniform `samplePoint`. Applies to both surface
     (`neeLight`) and fog-vertex (`neeVolume`) NEE. Quad lights are untouched and
     keep a bit-identical RNG stream. Validation: `spherelight.ftsl` mode V best-fit
     scale → 0.9997 (unbiased) at 80M/1024 spp, RMSE 2.5% bulk; sphere+`-fog 0.5`
     scale 0.988; cornell (quad) unaffected. Only the backward reference changed —
     the forward tracer emits sphere photons omnidirectionally as before, so the
     GPU/CUDA path is unaffected.
  3. **Spot penumbra sampling.** The forward spot samples uniformly in the outer
     cone then reweights by falloff, so photons in the dark penumbra edge carry
     small weights (mild variance). Exact CDF sampling of the smoothstep band would
     be lower-variance but needs a quartic inverse; uniform+reweight is correct.
- **Status:** OPEN (acceptable) — sphere + spot done 2026-07-10; **constant
  environment (`light env { spd … }`) done 2026-07-10 (increments 1a CPU + 1b GPU)**
  incl. the absolute-radiance We fix and on-device env emission; **image-based HDRI
  (`light env { file … }`, 2D luminance CDF + per-texel JH spectral upsampling) fully
  done 2026-07-10 (increments 2a CPU forward+backward miss/background + 2b backward
  env-NEE with MIS + 2c GPU forward port)**; **sphere-light cone importance sampling
  in the backward reference done 2026-07-10**; only spot penumbra CDF sampling (item
  3) still deferred.

### Built-in artificial-light SPDs: F-series transcribed from memory, discharge lamps are models not measurements
- **What (added 2026-07-11):** `src/lights.h` now provides spectral envelopes for
  artificial light sources, wired into `resolveLight()` / `preset:<name>`:
  - **CIE F-series fluorescents** — `fluorescentF2/F7/F11()` (`f2`/`cool-white`,
    `f7`/`daylight-fl`, `f11`/`triphosphor`), tabulated 380–780 nm at 5 nm via
    `sampledSPD()` → `tabulatedSpectrum()`.
  - **Gas-discharge lamps** — `sodiumHigh()` (`hps`/`sodium`), `sodiumLow()`
    (`lps`/`sodium-low`), `mercuryVapor()` (`mercury`/`hg`), `metalHalide()`
    (`metal-halide`/`mh`).
  - **CCT-tuned phosphor LED** — `ledCCT(kelvin)` via the `led<K>k` name (e.g.
    `led4000k`).
- **The honesty caveats (tech debt, not a bug):**
  1. ~~**The F2/F7/F11 tables were transcribed from the canonical CIE 15 illuminant
     data by hand/from memory.**~~ **VERIFIED & CORRECTED 2026-07-11; FULLY
     EXTERNALIZED 2026-07-12.** The baked tables were diffed against the authoritative
     CIE 15:2004 F-series (via colour-science), which caught a real bug:
     `fluorescentF7()`'s tail (685–780 nm) was wrong (it wiggled back up to 4.34 at
     765 nm instead of decaying smoothly). F7 was corrected; F2/F11 already matched
     exactly. **As of 2026-07-12 the baked `fluorescentF2/F7/F11()` tables are DELETED
     from `src/lights.h`** — the measured SPDs live only in
     `data/illuminant/{f2,f7,f11}.csv` and `resolveLightPreset()` resolves the
     `f2`/`f7`/`f11`/`cool-white`/`daylight-fl`/`triphosphor` names through
     `resolveTabulatedIlluminant()` (spectral_library.h) at load time. `preset:f2`
     and `spd file:data/illuminant/f2.csv` now load the *same file* — verified
     identical by `scenes/measured_spd.ftsl`, and the loader round-trips the old baked
     values exactly (e.g. F2 P(545 nm)=24.88). This sub-item is DONE.
  2. **The sodium / mercury / metal-halide entries are deliberately *illustrative*
     spectroscopic models, not per-lamp measurements** — correct line positions and
     plausible relative strengths (from spectroscopy references) over analytic
     continua, tuned to give the right visual cast. They are not a specific
     manufacturer's lamp and are not radiometrically calibrated. Same intended
     upgrade path: swap for measured SPDs when the data-file loader lands.
- **Proper fix:** the spectral asset library now exists (`data/<category>/<name>` +
  `src/spectral_library.h`, DONE 2026-07-12), and `resolveLightPreset()` already reads
  the F-series from `data/illuminant/`. To upgrade the discharge lamps to measurements:
  drop a measured lamp SPD (LSPDD / LICA-UCM, see `data/README.md`) into
  `data/illuminant/` (e.g. `hps.csv`, alias `sodium`) — it then resolves by name with
  no rebuild, and can shadow the analytic model. Only the discharge-lamp measured CSVs
  remain to be fetched; the preset-reads-CSV wiring is now generic and done.
- **Status:** OPEN (acceptable, reduced) — library + F-series data done and the
  built-in F-series presets now read the CSVs at load time; only discharge-lamp
  measurements remain (the analytic line models stay as the default until then).

### Built-in material presets: skin/soil & iridescent recipes are representative (metals + most natural curves now measured)
- **What (added 2026-07-11):** `src/materials.h` adds built-in common-material data
  and recipes, plus expanded glasses in `src/spectrum.h`:
  - **Metals** — `metal:Au|Ag|Cu|Al|Cr|brass` reflectance R(λ) (`metalGold()` etc.).
    Au/Ag/Cu/Al/Cr now computed from published measured n,k (Johnson & Christy 1972,
    Rakić 1995/1998; CC0 via refractiveindex.info) with
    `tools/ri_nk_to_reflectance.py` — see resolved item below. `brass` is still an
    alloy fit (no single canonical dataset).
  - **Glasses/crystals** — `glass:` gained `silica`/`fused-silica`/`quartz`,
    `sapphire`, `diamond`, `water`, `ice`, `acrylic`/`pmma`, `polycarbonate`/`pc`
    (Sellmeier for glass/crystal, `cauchy()` fits for water/ice/plastics), unified
    behind `resolveGlassIor()`.
  - **Natural diffuse** — `reflectance:leaf|skin|skin-dark|snow|soil|brick|concrete`.
    `leaf`/`snow`/`brick`/`concrete` are now measured USGS splib07 samples (see item 2);
    `skin`/`skin-dark`/`soil` remain representative shapes.
  - **Whole-material recipes** — `material { preset <name> }` via
    `resolveMaterialPreset()`: metals (glossy), glasses (dielectric), and iridescent
    `soap-bubble`/`oil-slick`/`anodized-ti`/`morpho`/`beetle`/`nacre`.
- **The honesty caveats (tech debt, not a bug):**
  1. *(RESOLVED 2026-07-11)* Metal reflectances were hand-transcribed and coarse;
     now regenerated from the canonical measured n,k datasets (Johnson & Christy
     1972 for Au/Ag/Cu at native sample points, Rakić 1995/1998 for Al/Cr at 20 nm)
     via `tools/ri_nk_to_reflectance.py`, which computes normal-incidence
     R=((n-1)²+k²)/((n+1)²+k²). Colours re-validated on diffuse-lit spheres
     (gold/copper/salmon/neutral-silver/neutral-chrome). Only `brass` remains an
     alloy fit — no single canonical dataset exists for it.
  2. *(PARTLY RESOLVED 2026-07-11)* Most `reflectance:` natural curves are now
     measured samples from the USGS Spectral Library v7 (splib07, public domain,
     DOI 10.5066/F7RR1WDJ), extracted with `tools/splib_to_reflectance.py`:
     `leaf` = fresh green Oak leaf (ASD, 10 nm — captures the real chlorophyll dip
     and steep red-edge), `snow` = melting snow mSnw01a, `brick` = medium-red
     building brick GDS353, `concrete` = light-grey road concrete GDS375 (all ASD).
     Still representative shapes: `skin`/`skin-dark` (human skin isn't in splib) and
     `soil` (splib's Soils chapter is mineral mixtures/sand, not a generic loam).
     Real vegetation/skin still vary enormously sample-to-sample.
  3. **The iridescent recipes (`soap-bubble`, `oil-slick`, `anodized-ti`, `morpho`,
     `beetle`, `nacre`) are physically-motivated film/stack *configurations*, not
     measured spectra** — layer indices/thicknesses tuned to give the right colour
     family, not matched to a specimen.
- **Renderer note (not a preset bug):** metal/glass presets are *specular*, so they
  show colour only through what they reflect/transmit. In a closed pinhole (mode B)
  box with nothing bright around them they read near-black — expected forward-tracer
  behaviour (same as the existing `mirror`/`glossy`/`dielectric` types); use mode A,
  an environment light, or surrounding geometry to see them. Their reflectance data
  is correct (verified by putting the same `metal:` spectra on a diffuse surface).
- **Proper fix (remaining):** three offline generators exist — `tools/csv_to_table.py`
  (generic CSV→`table`), `tools/ri_nk_to_reflectance.py` (refractiveindex.info
  n,k→reflectance), and `tools/splib_to_reflectance.py` (USGS splib07→reflectance) —
  plus, as of 2026-07-11, a *runtime* `file:<path>` loader so a reflectance CSV can be
  bound directly (`reflect file:data/reflectance/skin-light.csv`) without re-baking
  source. Metals and leaf/snow/brick/concrete are done. Remaining debt is finding
  measured samples for `skin`/`skin-dark` (a skin-optics dataset, e.g. NIST JRES
  122.026) and `soil` (a loam/dirt reflectance, e.g. ECOSTRESS/ISRIC) and overwriting
  the placeholder files in `data/reflectance/`, plus optionally validating the
  iridescent recipes against specimens.
- **DATA EXTERNALIZED 2026-07-12:** the baked `metalGold()..metalBrass()`,
  `reflectanceLeaf()..reflectanceConcrete()` tables (`src/materials.h`) and the
  `iorBK7()..iorPolycarbonate()` dispersion functions (`src/spectrum.h`) are **deleted
  from source**. They now live as data files — `data/metal/*.csv`,
  `data/reflectance/*.csv`, `data/glass/*.glass` (Sellmeier/Cauchy coefficients) — and
  are loaded by the same-named resolvers (`resolveMetalReflectance` /
  `resolveNaturalReflectance` / `resolveGlassIor`) now living in
  `src/spectral_library.h`. Every call site is unchanged (only the data source and
  includes moved); a category is a directory of files keyed by lowercased filename
  stem + `# aliases:` header, so new metals/glasses/reflectances drop in with no
  rebuild. The `sellmeier()`/`cauchy()`/`tabulatedSpectrum()` evaluators and the
  iridescent recipes stayed native (algorithms, not data). Verified: standalone loader
  test round-trips n_d (BK7 1.5168, SF10 1.7283, water 1.333) and R(λ) (Au R(700)=0.970)
  from the files, matching the old baked values.
- **RECIPES EXTERNALIZED (bundles) 2026-07-12:** the whole-material `preset` recipes
  and named light presets are now **composite asset bundle files**, not baked C++.
  `data/material/*.material` (soap-bubble, oil-slick, anodized-ti, morpho, beetle,
  nacre) and `data/light/*.light` (sun, daylight/d65, incandescent/a, led, led-warm)
  group a `type` + several spectral envelopes (`ior`/`substrate_k`/`spd`…) + intrinsic
  scalars (`film_ior`/`film_thickness`, `layer <n> <k> <nm>` rows) under one name.
  `resolveMaterialBundle` (materials.h) / `resolveLightBundle` (lights.h) interpret the
  manifest; spectrum-valued fields reuse the scene language's primitive vocabulary via
  a new shared `speclib::resolveSpectrumTokens`. So the tuned iridescent layer stacks
  are now DATA (retune/extend with no rebuild) while the interference/Abeles/Fresnel
  evaluators (render.h) and the LED/gas-discharge line models (lights.h) stay native.
  `resolveMaterialPreset` reads a bundle first, then a generic metal→glossy /
  glass→dielectric convention so bare primitive names still resolve with no file. The
  bit-for-bit faithfulness holds: `const N` == the old `iorConstant(N)`
  (`[N](double){return N;}`), so the bundles reproduce the baked recipes exactly.
- **Status:** OPEN (acceptable, much reduced) — metals + 4 natural curves are now
  measured data, and ALL spectral data AND the whole-material / named-light recipes now
  load from `data/` files rather than baked source; skin/soil and iridescent recipes
  remain representative. All presets load on CPU==GPU and render the right colours.

### Colored-LED light bundles + `filter` gel material — DONE 2026-07-12
- **Colored LEDs (data only).** A direct-emission LED die is a single narrow band, and
  the light-bundle vocabulary already had `gaussian center=… sigma=…`, so seven colored
  LEDs (`data/light/led-royal-blue`…`led-deep-red`) are pure-data `.light` bundles — no
  native code. Representative InGaN/AlInGaP peaks + FWHMs (sigma = FWHM/2.355). Measured
  die SPDs (slightly asymmetric) can drop into `illuminant/` later (pending in
  data/README). Verified `preset:led-red` renders pure red.
- **`filter` material (option A).** New `MatType::Filter`: a thin non-scattering absorber
  (colored gel / Wratten). The photon passes straight through (direction unchanged) and
  survives with probability T(λ) = `transmit`(λ), else absorbs — Russian roulette on the
  transmittance, β unchanged, unbiased. Specular straight-through so it makes no camera
  connection (like clear glass — it colors what's behind it). Threaded through EVERY
  tracer: forward CPU (`render.h`), forward GPU (`render_cuda.cu` `D_FILTER`), backward
  CPU (`backward.h`) + GPU, and BDPT CPU (`bdpt.h`) + GPU (delta vertex, throughput
  ×= T). `type filter` in FTSL (ftsl.h) reads `transmit`; `parseMatType` adds it for
  bundles. New `filter/` data category + `filter:<name>` token + `resolveFilterTransmittance`.
- **Data (RESOLVED 2026-07-12; full set digitized 2026-07-12):** `data/filter/wratten-*.csv`
  is now the **complete 84-filter Kodak Wratten set**, each **digitized from the numeric
  percent-transmittance tables** in *Kodak Wratten Filters for Scientific and Technical Use*,
  22nd ed. (pub. B-3), 400–700 nm at 10 nm, 31 samples (book dashes = negligible → 0).
  Files are named `wratten-<n>` (letter suffix lowercased) with `# aliases:` headers keeping
  the old descriptive names (red-25, deep-red-29, orange-21, yellow-12, green-58, blue-47,
  deep-blue-47b, …) resolvable; the 7 old descriptive-named CSVs were deleted. Text-layer
  pages were coordinate-extracted (word x → column, y → row); the eight image-only pages
  (28,29,31,33,35,38,41,47) were visually transcribed from high-DPI crops. Extraction/
  transcription scripts live in `scraps/` (`wratten_extract.py`, `wratten_manual.py`,
  `wratten_all.py`; gitignored). Renders confirm correct per-filter tints. Finer spacing/
  more gels can drop into `filter/` later (Rosco `.sed` / LEE / CRC), no rebuild.

### Full physical `layered` material [IMPLEMENTED 2026-07-11]
- **What:** both the FTSL `type mix` material (stochastic per-photon pick among named
  child materials, weights ≤ 1, remainder absorbs — Phase 2d, `scenes/mixmat.ftsl`,
  mode V PASS, CPU==GPU) and the richer physical `layered` material (spec §3.2) now
  ship. `layered` is a specular *coat* interface over a weighted *body*: on each hit a
  photon reflects off the coat with probability R, else it enters and one body lobe is
  chosen from a `mix`-style `layer "name" weight` list (leftover weight absorbs). Coat R
  + body weights partition the photon, so the surface is energy-consistent (validated:
  `scenes/layered.ftsl`, forward mode B and backward mode R, `absorbed+escaped=1.0`,
  residual 0).
- **Coat models:** `coat { reflectance … }` selects the interface reflectance:
  `fresnel` (plain dielectric Fresnel from the coat `ior`, rises toward grazing —
  clearcoat sheen), `thinfilm` (Airy multiple-beam reflectance from `film_ior` /
  `film_thickness` over the body index — soap-bubble iridescence), or `manual` (a flat
  `specular` fraction). The coat reflection is a glossy lobe about the mirror direction
  (`roughness` / `roughness_map`, lossless), and `film_thickness_map` gives spatially
  varying iridescence just like a `thinfilm` material.
- **Constraints of `mix`/`layered` (by design):** children/body lobes must be non-mix,
  non-layered materials (nesting rejected by the parser to keep resolution single-step
  and the CUDA CDF bounded); the CUDA path supports ≤ 8 child lobes (more → CPU
  fallback); a mix containing a fluorescent child is forward-only but as of 2026-07-11
  runs on the GPU forward path (the device fluoro port resolves the mix child before
  dispatch and the `D_FLUORESCENT` `shadeStep` branch handles it — see the
  GPU-fluorescence note below); a textured child is likewise fine.
- **Scope / fallbacks:** `layered` is CPU-only (forward + backward). GPU forward/backward
  fall back to the CPU tracer (`cudaForwardSupported` rejects any Layered material, like
  indexed palettes); BDPT (mode D) refuses a Layered scene with a clear message
  (`render it with mode B/P or mode R`) rather than dropping the surface via the
  randomWalk `default: terminate`. A per-lobe BDPT vertex strategy for `layered`
  (mirroring the forward split) is possible future work but not required for the
  reference/forward validation paths.
- **Status:** DONE — `mix` 2026-07-10; `layered` 2026-07-11 (CPU forward + backward).

### Backward reference tracer now validates fluorescence [RESOLVED 2026-07-11]
- **What (was):** `src/backward.h` had no Fluorescent case — a fluorescent material
  fell through to the Diffuse branch, so modes R (reference) and V (validate)
  silently mis-rendered `-scene fluoro`; fluoro scenes were forward-only.
- **Fix applied:** added a bispectral reradiation case to `BackwardRenderer::radiance`
  — the unbiased backward adjoint of the forward tracer's `fluoroInteract()`:
  1. **Elastic channel** — diffuse NEE (+ RR continuation) at the output wavelength
     with the small elastic base `rho(lambda)`, exactly as before.
  2. **Fluorescent DIRECT NEE** — a *second* excitation wavelength `lambdaIn` is drawn
     from the combined emission distribution (reusing `scene.emitSampler` /
     `invPdfLambda`, so multi-light SPDs weight correctly). The lights are connected at
     `lambdaIn` with a reradiation "albedo" `aEff(lambdaIn)*Q` (shared `fluoroWeights`),
     and the result is tinted by the emission colour at the OUTPUT wavelength
     `gOut = (M(lambda)/∫M) * invPdfLambda` — the `invPdfLambda` factor deconvolves the
     camera-path wavelength-sampling density so the reradiated colour follows `M(lambda)`
     and not the light SPD used to sample `lambda`.
  3. **Indirect excitation** — a single stochastic continuation splits between an elastic
     bounce at `lambda` (prob `rho`) and a wavelength-switched bounce to `lambdaIn`
     (prob `pF ~ gOut*aEff*Q`, throughput `*= wFluo/pF`), so light that bounces before
     exciting the dye (light→wall→dye→camera) is captured without double-counting the
     direct term (`specularArrival=false` suppresses the emission-on-hit term).
- **Validation (mode V, forward mode-B vs backward, `-scene fluoro`):** best-fit
  backward→forward scale = **0.996** (≈1, i.e. the two agree on ABSOLUTE scale — a wrong
  bispectral normalisation would not), residual 94% firefly-concentrated (variance not
  bias), and the bulk RMSE (ex. top-1%) scales as **1/sqrt(N)**: 2.05% at 40M/400spp →
  **1.02%** at 160M/1600spp (4× samples ⇒ exactly 2× reduction), which a transport bias
  could not produce. `-checkfluoro` (deterministic sampler/branch/Stokes-shift self-test)
  is retained as a fast complementary check.
- **Remaining (BDPT / mode D):** bidirectional bispectral fluorescence (a wavelength
  change inside a light↔camera connection needs hero-wavelength MIS, à la Mojžík et al.
  2018) is still deferred — mode D refuses fluoro with a clear message pointing to modes
  B/P (forward) or R (backward). The backward reference now covers fluorescence
  validation, so this is low priority.
- **Status:** RESOLVED 2026-07-11 for modes R/V; BDPT (mode D) bispectral vertices
  remain future work.

### RESOLVED: Backward reference tracer now validates participating media (fog)
- **What (was):** `src/backward.h` ignored `scene.medium` — its camera rays didn't
  sample volume free-flight or in-scattering, so `-fog` with modes R/V would have
  compared a volumetric forward image against a vacuum backward image (garbage
  residual). Fog was therefore forward-only and never set in refMode.
- **Fix applied:** added a homogeneous-medium path to `BackwardRenderer::radiance`
  that mirrors the forward tracer exactly:
  1. **Free-flight sampling** competes with the surface hit each bounce
     (`tMed = -ln(1-u)/sigma_t`; on `tMed < dSurf` a volume collision occurs).
  2. **`neeVolume()`** — phase-function next-event estimation at the collision
     vertex: the surface BRDF/cosine are replaced by the single-scattering albedo
     and the HG phase function `hgPhase(dot(wIn, wi), g)`, with fog transmittance
     `exp(-sigma_t*dist)` on the shadow ray (the backward mirror of the forward
     `connectVolume`). The phase angle uses reciprocal conventions to the forward
     side, both equal to the physical `dot(prop_in, prop_out)`.
  3. **Analog scatter/absorb** continuation: survive with prob = albedo, then
     `sampleHG` a new direction (throughput unchanged); otherwise absorb.
  4. **Beer-Lambert on surface NEE too:** `neeLight` now attenuates its shadow ray
     by `exp(-sigma_t*dist)` (took a new `lambda` parameter).
- **Validation (mode V, forward vs backward, identical fog):** the best-fit
  backward→forward scale agrees to ~4 sig figs across no-fog / fog-g0.3 /
  fog-rayleigh, which a transport bug could not produce. The raw-linear residual is
  firefly-dominated (top-1% pixels hold 77–95% of it) from the unbounded 1/dist^2
  light connection, so full RMSE plateaus but the **bulk RMSE (ex. top-1%) scales
  as ~1/sqrt(N)**, proving variance not bias: for fog g=0.3 alb=0.85 at 256^2, bulk
  RMSE went 7.67% (120M/800spp) -> 4.53% (480M/3200spp) [1.69x ~ ideal 2x], with
  firefly concentration held constant at ~86% and the 4x run reporting PASS.
  No-fog bulk RMSE is 1.2% (95% firefly-concentrated). The firefly-vs-bias
  diagnostic (residual concentration + bulk RMSE) was added to `compareFilms` in
  `src/main.cpp` specifically to make this distinction rigorous.
- **Status:** RESOLVED 2026-07-10. `-fog` can now be combined with modes R/V.
  `-checkfog` (deterministic transmittance / HG mean-cosine / phase-normalization
  self-test) is retained as a fast complementary check.

### Model A redefined as the finite-lens physical camera (GPU port landed)
- **What:** as of 2026-07-11 **mode A is the physical finite-lens camera**: a finite
  aperture + thin lens + film imaged by next-event estimation of the pupil
  (`Renderer::connectLens`/`camera.h::lensImage`). It replaces the old contact-sensor
  "mode A" (a flat film wall — no aperture, so it integrated the whole hemisphere per
  pixel and could not form an image; retired). Mode B is now the pinhole (`aperture→0`)
  limit; mode C is the brute-force forward-catch oracle A is validated against
  (matching framing/DOF/scale — auto-exposure within ~2.5% at equal aperture/focus).
- **GPU port (done):** the CUDA `DCamera` now has `lensImage` (thin-lens `u' = u − ρ/f`,
  shared with the mode-C `catchPhoton`) and `kTrace` runs device `connectLens`/
  `connectLensVolume` splats under `camMode 'A'` — emitter-direct, diffuse-vertex, and
  fog-in-scatter, mirroring the CPU. `renderForward` selects `camMode 'A'` on the GPU
  and `runRender` treats mode A as a GPU-forward mode. Validated vs CPU (Cornell, 192²,
  wide aperture 0.25/focus 2.2, 40M photons): energy to 4 sig figs, auto-exposure 2.99e-8
  GPU vs 2.95e-8 CPU (1.4%), image RMSE 2.6/255 (pure MC noise — the GPU is an
  independent realization); the tiny-aperture A→B pinhole limit holds on-device (sharp,
  RMSE 3.16/255 vs mode B). The old contact-sensor GPU `deposit` path was removed with
  the port. Mode A remains **rectilinear only** (a real fisheye needs a wide-angle lens
  element the single thin-lens can't form) — a fisheye+A/C camera is rejected, and a
  fisheye lens still falls back to the CPU even on `-device gpu` (see GPU-fisheye entry).

### GPU backend (`-device gpu`) covers forward camera models A/B/C
- **What:** the CUDA backend (`src/render_cuda.cu`, `renderForwardCuda`) implements
  the finite-lens next-event splat (A), the pinhole splat (B), and the finite-aperture
  thin-lens forward catch (C), selected by the `camMode` parameter. It is used for
  `-mode A/B/C` and the forward pass of `-mode V`. It also tracks per-pixel photon
  **hit counts** on-device (a `d_hits` buffer incremented in `filmAdd`, downloaded into
  `Film::hits`) — matching the CPU `Film::add`. This fixed a latent bug where the GPU
  never populated `hits`, so the progressive `~X% noise` graininess estimate (and the
  new `-noise` stop) read a constant **0%** for any `-device gpu` render. The backward
  tracer is now on-device too: **mode R has its own GPU backward megakernel** (`kBackward`,
  Plan A, 2026-07-11 — including the physical mesh-lens as a ray-gen front-end), and the
  **mode-P composite reuses it for its camera-side layer** (`renderComposite` calls
  `renderBackwardCuda` when `cudaBackwardSupported`), so both of P's layers run on the GPU
  within the backward-GPU scope. Only scenes outside that scope (fog/env/spot/collimated/
  fluorescence) — and mode V's backward reference, kept on the CPU by design as a stable
  ground truth — still use the CPU backward tracer. **Fluorescence is now ported on-device (done
  2026-07-11):** each Fluorescent material bakes its excitation spectrum
  (`DMaterial.fluoAbsorb`) and emission-SPD CDF (a flat `fluoCdfAll` slice, per-material
  `fluoCdfOffset/N/step`); the `shadeStep` `D_FLUORESCENT` branch splats the elastic
  channel at lambda and the glow channel at a Stokes-shifted lambda' (`sampleFluoEmit`)
  with albedo `aEff*fluoYield`, then stochastically continues (elastic / reemit / absorb)
  exactly like `render.h`'s `fluoroInteract`. `shadeStep`'s `lambda` became a reference
  (Stokes shift mutates it), and the wavefront `kWfShade` now writes `st.lambda[slot]`
  back on the continue branch. `cudaForwardSupported()` no longer rejects Fluorescent
  materials. Validated GPU-vs-CPU (fluoro scene, mode B and mode A): energy conserves
  (`sum/emitted=1.0`, absorbed 0.7034 vs 0.7036), mean RGB matches to ~0.1/255, image
  RMSE 3.5/255, wavefront matches the megakernel, and `-checkfluoro` PASSes. The BDPT
  kernel (mode D) has no fluorescent vertex strategy, so `cudaBdptSupported()` explicitly
  rejects fluorescence → CPU fallback (mode D already refuses fluoro scene-wide anyway).
- **Why acceptable / validated:** model B is the default and the one mode V
  validates. The kernel `kTrace` mirrors `Renderer::tracePhoton` exactly and gates the
  camera-specific work on `camMode`: emitter/diffuse/in-scatter `connectLens` runs for
  A, the pinhole `connect`/`connectVolume` for B, and `catchPhoton` (thin lens
  `u' = u - rho/f`) for C. (Mode A validation is in the redefinition entry above; the
  historical **Mode A bullet below** records the now-removed contact-sensor GPU
  `deposit` path, not the current finite-lens next-event mode A.)
  Validation vs CPU (Cornell, 128²):
  - **Mode B:** image RMSE ≈ 0.85/255 at 200M photons (pure MC noise); `-mode V
    -device gpu` PASSes vs the backward reference (bulk RMSE 4.17% ≈ CPU 4.22%);
    ~14× speedup (400M @256²: 153s CPU → 10.9s GPU on an RTX 4090).
  - **Mode A (retired contact-sensor GPU path — historical):** energy report matched
    to 4 sig figs (sensor 0.3298 vs 0.3299); image RMSE scaled as √N — 11.18/255 @40M
    → 5.11/255 @200M (5× photons, ideal 2.24×, measured 2.19×), proving variance not
    bias. This validated the old flat-film-wall mode A, which has since been replaced
    by the finite-lens next-event camera (validated separately above); the device
    `deposit` path it exercised has been removed. Retained only as a historical record.
  - **Mode C:** energy report matches to 4 sig figs; with a wide aperture (0.25,
    focus 2.2) the caught fraction matches exactly (sensor=0.0058) and per-image
    auto-exposure agrees (1.60e-8 vs 1.59e-8). Image RMSE scales as √N —
    15.70/255 @200M → 8.10/255 @800M (4× photons, ideal 2×, measured 1.94×) —
    proving variance not bias. (The CPU is deterministic across runs, so GPU is the
    only independent noise realization; the small default aperture is catch-starved
    and its tone-mapped RMSE is dominated by per-image auto-exposure.)
- **Spectral baking:** device materials/fog sample each `std::function` Spectrum into
  a fixed 96-entry table over [360,830] nm with linear interpolation (`SPEC_N=96`).
  Smooth reflectances/Sellmeier indices make this accurate to within MC noise; a
  pathologically spiky spectrum would need a finer table. CIE CMFs are ported
  analytically (no table).
- **Precision (mixed FP32/FP64, default float transport):** consumer GeForce GPUs run
  FP64 at ~1/64 the FP32 rate, so the megakernel computes all geometry/BRDF/spectral
  transport in a compile-time `Real` scalar (`float` by default) while accumulating the
  film and energy in `double` (`atomicAdd` on `double*`). Build with
  `-DFTRACE_GPU_FP32=OFF` for a full-FP64 device path (bit-closer to the CPU, far slower
  on GeForce; sensible on datacenter cards or for precision debugging). The CPU renderer
  is always `double` and remains the ground-truth. Float-safe self-intersection epsilons
  (`RAY_EPS=1e-4`, `DET_EPS=1e-6`) replace the FP64 `1e-6`/`1e-9`. **FP32 validated vs
  FP64 CPU (Cornell, RTX 4090):** energy conserves exactly (`sum/emitted=1.000000`,
  residual=0 on A/B/C/V — no self-intersection leak from the float epsilons); fractions
  converge to 4 sig figs (retired contact-sensor mode A sensor 0.3298 vs 0.3301, mode C 0.0059 vs 0.0058); mode
  V PASSES (bulk RMSE 2.89%, firefly-dominated); ~14× faster than FP64 (400M @256² in
  0.76s vs 10.9s). The DVec3 3-arg ctor deliberately keeps `double` params so host
  brace-init from `double` Scene coords is a widening (legal) conversion, never
  narrowing; spectral/CDF tables stay `double` (host-baked, tiny, cached).
- **Portable build (multi-arch) + HIP-ready:** `-DFTRACE_CUDA_ARCH=` selects the device
  arch set — `native` (default; the local GPU only, fast builds), `all-major` (a
  redistributable fat binary: one cubin per major arch + forward-compatible PTX so newer
  GPUs JIT at load), `all`, or an explicit `"75;86;89"` list. The device kernel is
  written in the portable CUDA/HIP subset (`__global__`/`__device__`, grid-stride, double
  `atomicAdd`, `<<<>>>` launches); the only vendor-specific surface — the host runtime API
  (device query, malloc/memcpy/memset/free, error strings, synchronize) — is isolated
  behind a compat block at the top of `render_cuda.cu` that maps `cuda*` → `hip*` under
  `-DFTRACE_USE_HIP`/`__HIP_PLATFORM_AMD__`. Porting to AMD ROCm is therefore a
  build-system change (compile this one file with `hipcc`), not a code rewrite. **CUDA is
  the supported GPU backend today; HIP is a near-drop-in future target (untested — no AMD
  hardware here).**
- **Proper fix — DONE (2026-07-11):** the backward tracer is now ported to CUDA — mode R
  has its own GPU backward megakernel (`kBackward`, including the physical-lens ray-gen
  front-end), and the mode-P composite reuses it for its camera-side layer. (The device
  fluorescence path and textured-albedo path are done too — see above.) Remaining CPU-only
  backward work: scenes outside the backward-GPU scope (fog/env/spot/collimated/
  fluorescence) and mode V's reference (kept on the CPU by design).
- **Status:** OPEN (acceptable) — logged 2026-07-10; A/C, mixed-precision FP32, portable
  multi-arch build, and the HIP compat layer added same day. Requires a CUDA toolkit at
  configure time; without one the project builds CPU-only and `-device gpu` warns and
  uses the CPU.

### GPU scaling path: megakernel vs. wavefront (IMPLEMENTED — both backends ship)
- **Status:** both backends now exist. The **megakernel** (`kTrace`) is the default; the
  **wavefront** (streaming) backend is opt-in via `-wavefront`. They share the exact same
  device physics — `genPhoton()` (emitter sample + direct connect) and `shadeStep()` (one
  bounce: medium/catch/material dispatch) are `__device__` functions called by both — so
  only the *scheduling* differs. Because of that shared code, adding the wavefront left the
  megakernel bit-for-bit identical (validated: cornell/materials mode B/C images and energy
  reports unchanged after the extraction refactor).
- **Context:** the **megakernel** is one `kTrace` launch where each thread runs an entire
  photon path (emit → bounce loop → connect/catch/deposit) start to finish. This is the
  right choice for *this* renderer's typical case: an RTX 4090 has huge register/occupancy
  headroom, Cornell-class scenes are shallow, and a single kernel keeps all state in
  registers with no round-trips to global memory. It hits ~500M+ photons/s in FP32.
- **The known limitation (thread divergence):** in a megakernel, threads in a warp that
  take different material branches (a dielectric refraction next to a diffuse bounce next
  to a grating), or that terminate after wildly different path lengths, **serialize** —
  the warp runs at the speed of its slowest/most-divergent lane, and finished lanes sit
  idle while others keep bouncing. The megakernel also carries the register footprint of
  *every* material's code path in *every* thread, capping occupancy. Both effects get
  worse as scenes gain more material variety and deeper paths, and they bite harder on
  smaller GPUs (fewer SMs / less latency-hiding to absorb the idle lanes).
- **What we shipped (wavefront / path-regeneration):** the tracer is split into two
  coherent stages that alternate over a **persistent pool of photon slots** (SoA state:
  ro/rd/beta/lambda/rng/bounce/alive/hit, W = min(N, 1M) slots): **extend** (`kWfExtend`,
  one `closestHit` per live slot) then **shade** (`kWfShade`, one `shadeStep` per live
  slot). A warp's threads therefore execute the same stage together instead of diverging
  on per-photon path length. When a path terminates (or hits the bounce cap), its slot
  **immediately regenerates a fresh photon** (`wfSpawn` claims the next index from an
  atomic budget counter) — path compaction by regeneration — so SIMD lanes stay full until
  all N photons are traced. The host loop (`wavefrontTrace`) reads a live-slot counter each
  pass and stops when the pool drains. **Phase 2 not yet done:** a *sort/compaction by
  material* before the shade stage would additionally kill BSDF-branch divergence (every
  thread in a warp running the same material) — that's the remaining coherence win over
  what's implemented, and the natural next step if a material-diverse scene proves
  shade-divergence-bound.
- **The cost:** the wavefront reads/writes the whole photon state to global memory every
  bounce and launches two kernels per pass, so it's *not* a win for shallow, uniform scenes
  on a big GPU — the megakernel's register-resident state wins there. Measured on an RTX
  4090, materials mode B, 400M photons: megakernel 0.79 s vs wavefront 1.60 s (~2× slower),
  exactly the expected regime. The wavefront's payoff is on divergent / deep-path scenes and
  smaller GPUs; energy conservation and image agreement (to within Monte-Carlo noise) hold
  across every scene tested (cornell, materials A/B/C, spotlight, envlight, thin-film,
  multilayer, mix, fog).
- **Re: "wavefront helps divergent scenes AND small GPUs" (todo.txt question):** it's
  *both*, and they're related. (1) *Divergent scenes* — many materials and/or highly
  variable path lengths — benefit from the per-material sort (kills branch divergence) and
  compaction (kills path-length divergence). (2) *Small GPUs* benefit because they have
  less occupancy/latency-hiding headroom to paper over idle lanes and high per-thread
  register pressure, so keeping warps coherent and full matters more there. A big GPU on a
  shallow uniform scene (our current case) is the one regime where the megakernel clearly
  wins, which is why we ship it first.
- **Decision:** the megakernel stays the default/recommended path for the scenes this
  renderer targets (shallow, uniform, big GPU); the wavefront ships as an opt-in
  (`-wavefront`) for the divergent/deep-path/small-GPU regime. Remaining optional work: the
  Phase 2 per-material shade sort (above), and a heuristic for `-device auto` to pick the
  backend by scene material variety + path depth rather than always defaulting to the
  megakernel.

## Performance

### RESOLVED: Diffuse-mesh renders were ~60× slower per photon (degenerate BVH)
- **Symptom:** The Cornell + diffuse torus (`-mesh torus.obj`) traced at
  ~34–37 µs/photon, vs ~0.55 µs/photon for the Cornell + glass sphere. 3M photons
  took ~112s. Made mesh scenes impractical.
- **Root cause (found by instrumentation):** The BVH build was leaving giant
  leaves. Added a `-bvhstats` diagnostic (nodes/leaf-tests per ray + leaf-size
  histogram) which showed the 16k-tri torus BVH had only **245 nodes / 123 leaves,
  max leaf = 9334 primitives**, and each ray did **~1091 leaf primitive tests**.
  The culprit was the SAH termination in `Bvh::buildRecursive` (`src/bvh.h`):
  `if (bestSplit < 0 || bestCost >= leafCost) makeLeaf();`. Object-SAH on a
  ring-like shape hits a top-level pathology — splitting a torus through its
  centre yields two C-shaped halves whose AABBs each nearly equal the *whole*
  box, so every candidate split has cost ≈ the leaf cost. The greedy "only split
  if it lowers SAH" test therefore gave up immediately at the top and dumped
  most of the mesh into one leaf. (Path length was a red herring: the diffuse
  Cornell walls dominate bounce count regardless of the mesh.)
- **Fix applied:**
  1. **BVH (the real fix):** use SAH only to *choose* the split plane, and always
     recurse down to `LEAF_SIZE`, falling back to a median (`nth_element`) split
     when SAH finds no usable partition. Result: 245→**10425 nodes**, max leaf
     9334→**4**, leaf-tests/ray 1091→**0.7**. Torus 3M render 112s→**1.3s**
     (0.43 µs/photon — now *faster* than the glass-sphere reference).
  2. **Front-to-back traversal ordering** in `traverseClosest` (descend nearer
     child first; cull children against `tMax` at push time). Minor on its own
     (~8%) but correct and keeps the win robust.
  3. **Russian roulette** for Diffuse/Mirror/Glossy in `Renderer::tracePhoton`
     (`src/render.h`): terminate with prob `1-reflectance`, keep `beta` unchanged
     on survival. Unbiased; caps path length (residual now 0.0000) and removed the
     now-dead `betaCutoff`. `maxBounce=32` kept as a hard safety cap.
- **Validation:** `-checkbvh` still reports 0 mismatches on cornell/materials/
  prism/torus; energy conserves exactly (`sum/emitted=1.000000`) on all scenes.
- **Status:** RESOLVED 2026-07-10. Logged & fixed same day.

## Scene interoperability / importers

- **Mitsuba XML → FTSL: DONE 2026-07-11** (`tools/mitsuba_to_ftsl.py`). Mitsuba
  0.6/2/3 is also a spectral PBR renderer, so the mapping is nearly 1:1
  (perspective/thinlens sensor → camera, diffuse/conductor/roughconductor/
  dielectric/plastic/blendbsdf → materials, area/constant/envmap/point/directional
  emitters → lights, rectangle/cube/sphere/obj shapes with full `to_world`
  transforms, `<ref>`/`<default>` resolution). Validated on a converted Cornell
  scene (glass + gold spheres, colored walls, area light) rendering correctly in
  modes B and D. **Because Blender exports to Mitsuba XML via `mitsuba-blender`,
  this is also the Blender→FTSL path.**
  - **Known approximations (flagged with `# WARN:` in output):** roughdielectric →
    smooth dielectric (no rough transmission in FTSL); plastic/roughplastic →
    glossy (diffuse+specular coat merged); bumpmap/normalmap dropped to base BSDF;
    mask opacity ignored; `.ply`/`.serialized` meshes emitted as `mesh` lines but
    ftrace's loader is OBJ-only (convert first); mesh area-emitters have no FTSL
    equivalent (emitted as lit geometry, emission dropped); mesh `to_world` with
    rotation/shear only partly expressible (translate+scale + euler).
  - **Possible follow-ups:** map `.ply` via an auto OBJ conversion; emissive-mesh
    support (needs an emissive-triangle light primitive in the core); rough
    transmission material.
- **POV-Ray: DECLINED (2026-07-11), rationale logged.** The SDL is genuinely nice
  (programmable, exact CSG, implicit `isosurface`), but a poor fit here: (1) faithful
  parsing = writing a Turing-complete interpreter (macros/loops/functions), far more
  than an XML parse, and POV-Ray has no mesh *export* to lean on; (2) we're a
  triangle/quad/sphere renderer with no analytic CSG or implicit intersection, so
  importing means **tessellating** everything (marching cubes for isosurfaces/blobs,
  mesh-booleans for CSG) — which discards POV-Ray's exact-surface advantage, its whole
  point; (3) RGB/non-spectral/non-physical-camera means re-authoring the physics anyway.
- **Alternative worth its own feature (deferred):** a **native SDF / implicit-surface
  primitive** (sphere-traced, GPU-portable) would give metaballs/isosurfaces *exactly*
  without lossy tessellation — useful independent of any importer, and the right way to
  ever support POV-Ray-style implicit geometry. Not started.
