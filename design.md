# ftrace — design & architecture

Physically-based **spectral** renderer (C++17, single exe `ftrace.exe`), Windows /
MSVC / CMake, with a CUDA backend (RTX-class, tested sm_89). Photons are traced
**forward from the lights** in the flagship modes (hence "forward raytracer"), but
backward path tracing, BDPT, photon mapping, SPPM, VCM and a z-buffer preview
rasterizer are all built in. `README.md` is the exhaustive user-facing manual;
this file records the *internal* architecture. `known-issues.md` tracks bugs/debt.

## Render modes (dispatch in `main.cpp`)

| Mode | What | Core |
|---|---|---|
| `A` | forward + finite-lens physical camera (photons hit the lens) | `render.h` |
| `B` | forward light tracing, splat through pinhole/lens to film (flagship) | `render.h` |
| `C` | forward + contact sensor | `render.h` |
| `R` | backward (unidirectional) path tracer — the reference | `backward.h` |
| `W` | deterministic Whitted/POV-Ray preview: mode `R`'s walk with every estimator replaced by a fixed quadrature (noise-free at 1 spp, biased, CPU only) | `backward.h` (`whitted`) |
| `P` | composite: forward B + backward R passes merged | `main.cpp` orchestration |
| `D` | bidirectional path tracer (BDPT, MIS) | `bdpt.h` |
| `M` | photon map (deposit pass + per-pixel density gather; optional `-pmfg` final gather) | `photonmap.h`, `photonmap_render.h` |
| `S` | SPPM (progressive photon mapping, shrinking radius) | `sppm_render.h` |
| `U` | VCM (vertex connection & merging) | `vcm.h` |
| `V` | validation: renders B and R, reports residual | `main.cpp` |
| `-raster` | z-buffer preview rasterizer + interactive fly viewer (`-explore`) | `raster.h`, `raster_cuda.cu` |

CPU is the default device; `-device gpu|auto` enables CUDA for forward A/B/C,
M-deposit/gather, R, D (untextured), and the raster preview (`-raster-gpu`).

## Module map (src/)

- **`main.cpp`** (~6200) — CLI parsing, mode dispatch, chunking/progressive loop
  (`cpuSppChunks`, `chunkFixed = !progressive && g_showWindow` — a bare fixed `-n`
  with no `-window`/budget flag runs monolithically with output only at the end),
  periodic write/`-interval`, checkpoint/resume (`.ftbuf`), multi-camera shared
  passes, exposure-lock metering pre-pass (device-aware: each mode meters through
  its own GPU entry point when `-device gpu|auto` and the mode's support predicate
  allow, CPU fallback otherwise; all-M pinhole groups meter in one batched
  `renderPhotonMapSharedCuda` pass with `MeterConverge` early-stop via `onFrame`),
  PNG/PPM output.
  **Output-directory precheck:** an `ensureOutDir` lambda runs once after the `-check*`
  self-test early-returns — the earliest point where `out` is final, since the
  bare-invocation preview path can still rewrite it to a `$TEMP` name — and creates
  `-o`'s (and `-savemap`'s) missing parent directory, or exits 2 with a named error.
  It has to be a precheck rather than a per-write fallback because every writer is
  reached only *after* the film exists: a bad directory used to be discovered at the
  first `-interval` tick and threw the whole accumulated render away. One check on `-o`
  suffices for the `.ftbuf` sidecar, the per-camera `outFor()` names and the stereo eye
  pair, which all share its directory; `-savemap` is the only independent path.
- **`src/gpda/`** — the FTSL front end. `ftsl_scene.epeg` (in
  `tools/loom/loom/grammar/`) is the **single source of truth** for FTSL's syntax;
  `loom.grammar.emit_cpp` compiles it to `ftsl_scene.gen.cpp` — a parser graph plus
  the lexer's rule table. `gpda_lexer.hpp` (hand-written, *not* generated) turns
  source text into tokens by longest match over those rules; it derives a first-byte
  set from each rule's pattern and takes a literal fast path where it can, so a token
  costs one or two `std::regex` calls rather than one per rule.
  `tokenized.{hpp,cpp}` + `pool.hpp` (verbatim vendored copies of the upstream GPDA
  parser in `D:\visual studio projects\GraphParser`) then walk the graph to a parse
  tree, which
  `ftsl_reduce.hpp` reduces to the same `std::vector<ftsl::Block>` the hand-written
  parser produced. `ftsl_frontend.hpp` is the entry point — since 0.79.0 that is just
  `ftsl_gpda::parse`. loom parses the
  *same* grammar in Python via the pinned `tools/loom/loom/grammar/_gpda.py`, so
  ftrace and loom cannot disagree about the language. The flip (0.68) was gated on
  `-validate-grammar` reporting zero structural mismatches across all 2595 `.ftsl`
  files in the tree, down to per-statement line numbers; it held there for ten
  releases, and 0.79.0 deleted the hand-written parser, the `-legacy-parser` escape
  hatch and the cross-check differ, leaving exactly one implementation of the
  language. Loading the largest scene in
  the tree (22 KB) costs ~42 ms of lex+parse. `tools/gpda_lexcheck/` is the permanent
  differential validator for the lexer's fast paths: it brute-forces that no rule's
  first-set ever excludes a byte the rule's own regex could match, and that the fast
  lexer's token stream is identical to a naive all-regex one.
- **`scene.h` / `ftsl.h`** — scene model and the FTSL semantic pass
  (cameras, camera_curve/path/orbit, materials, lights, media, implicits, meshes).
  `FTSL.md` documents the language. Everything downstream of
  `std::vector<Block>` — turning blocks into a `Scene` — lives here. Through 0.78 a
  hand-written recursive-descent tokenizer + `Parser` lived here too, kept compiled in
  behind `-legacy-parser`; 0.79.0 deleted both, so `loadSource()` now has a single
  path. Note that `applyBracketGroup` (how a `[ … ]` group becomes a swatch / array
  literal / index selector) deliberately stays at *this* level rather than in the
  front end — keeping that decision in one place outside the parser is what let the
  grammar replace the hand-written parser as a pure parse-tree exercise.
  **Unknown-key reporting** (since 0.77.0) rides on this pass. `Stmt` carries a
  `mutable bool used`, set inside `find(const Block&, const char*)` — the single choke
  point every property read (`strOf`/`vec3Of`/`dblOf`/`spectrumParam`/…) funnels
  through, so the accounting costs no per-builder changes. `mutable` is required
  because reads take a `const Block&` and "I was read" isn't part of a block's logical
  value. The ~17 sites that iterate `b.stmts` directly instead of calling `find` —
  repeated-key gathers (`point`, `density_at`, `look_point`, `roll_at`/`fov_at`,
  `layer`, `surface`, `key`), exhaustive dispatch loops (group children, isosurface
  field elements and their nested CSG recursion, record bodies, record-override
  materials), and flat-word bodies (`table`, `palette`, `data`) — mark explicitly via
  `markUsed(b, key)` / `markAllUsed(b)`. After `Builder::build` finishes,
  `collectUnusedKeys` walks the blocks and records anything still unmarked on
  `Loaded::unknownKeys`; `loadSource` prints those to stderr before each success
  return. Warnings are carried on `Loaded` rather than printed inside `build()`
  because `prefer { } else { }` trial-builds several candidate scenes and discards all
  but one — only the accepted candidate's warnings are the author's problem. The check
  warns rather than errors so a scene with a stale property still renders; the point is
  that an unread key otherwise silently does nothing, turning a typo or a drifted
  emitter into a wrong image instead of a message. This is what makes the loom
  emitter-drift audit (`scraps/emit_audit.py`, TODO J3c) mechanically possible at all.
  Lights are `Emitter`s with an `EmitterShape`
  (Quad/Sphere/Spot/Env/Cylinder/**Mesh**/**Sun**); each carries its own SPD and a `power`
  = emitIntegral·geomWeight selection weight. **Distant sun** (since 0.84.0):
  `EmitterShape::Sun` (`Scene::addSunLight`, `light sun { … }`) is an infinitely-distant
  disc whose rays arrive parallel. It reuses the spot fields with
  `spotCosInner == spotCosOuter == cos θ`, so `spotOmega` evaluates to the cone solid
  angle `Ω` and no new field is needed on host or device; `geomWeight = Ω·πR²`. The
  authored SPD is *perpendicular irradiance*, stored as radiance `E⊥/Ω` (so `angle`
  changes the penumbra, not the exposure). Forward emission samples the cone then an
  entry disc of radius `R` perpendicular to the sampled direction — joint pdf
  `1/(Ω·πR²) = 1/geomWeight`, exactly analog, so **every** photon enters the scene.
  Backward does cone NEE and adds the direct disc view on a ray miss only when
  `specularArrival` is true — a single unbiased estimator with **no MIS weight**, since
  NEE runs at precisely the material types that then clear that flag. `Scene::sunCount`
  gates all of it, so sun-free scenes are untouched. Not area-connectible, so `bdpt.h` /
  `vcm.h` reject it like Spot/Env. The Preetham sky's `sun_disk separate` option
  (`sky::SunDisk`) unbakes the solar disc from the env map and registers an
  energy-matched Sun instead — the same picture, converging ~20× faster in forward modes.
  **Mesh area lights** (since 0.41.0): a
  material with an `emit` spectrum bound to a `mesh` registers an
  `EmitterShape::Mesh` emitter (`Scene::addMeshLight`) holding a per-triangle
  cumulative-area CDF (`Emitter::meshTris`, `EmitTri`); `samplePoint` binary-searches
  the CDF to pick a triangle by area then samples it barycentrically (pdf = 1/total
  area — the same law as a quad, so NEE / forward emission / BDPT s=0 all consume it
  through the existing generic paths). `ftsl.h addMesh` auto-registers it and, with a
  mesh-block `power`/`lumens`, rescales the SPD over the mesh area (cloning the
  material + rebinding the range's triangles so a shared material is untouched).
  Emission is one-sided (front face), so `addMesh` auto-orients a *closed* emissive
  shell outward before registration: it computes the signed volume about the range's
  centroid and, if it is negative (inward winding) and large enough to be a real
  enclosed volume (thresholded vs area^1.5, so planar/open sheets are left alone),
  reverses every triangle (swap v1↔v2 + uv/shading-normal, re-`finalize()`). This
  keeps an inward-wound import (e.g. `torus.obj`) from radiating into its own hollow.
  The GPU mirrors the sampler: `DEmitter` gains a device `DEmitTri*` CDF +
  `emitterSamplePoint` shape-5 branch, uploaded per emitter.
  **Emitter-less emissive surfaces** (since 0.118.1): `emit` lives on the *material*, so
  a sphere / CSG solid / marched `isosurface` can carry it too, but none of those have
  triangles to register an emitter against. The CPU already handled them, because
  `backward.h` reads `Material::isLight` + `emitSlot(...)` directly; the GPU did not,
  because all three of its backward emission-on-hit sites keyed off `dEmitterForMat()`
  and so rendered such surfaces black. `DMaterial` now carries `matIsLight` + a baked
  `matEmit[SPEC_N]` (plus `rgbMatEmit` for the fast RGB path), and those sites fall back
  to it when `dEmitterForMat() < 0`. The *emitter* is still preferred where one exists —
  its SPD may carry a `power`/`lumens` flux normalisation the raw material spectrum does
  not, and that ordering is also what stops a mesh light double-counting. Such surfaces
  remain emission-on-hit only (no NEE, no forward emission): they glow but illuminate
  nothing, including themselves — logged as a known limitation.
- **`geometry.h` / `bvh.h`** — primitives + SAH BVH (split plane by SAH, always
  recurse to LEAF_SIZE, median fallback; front-to-back traversal, ray-slab test
  unrolled; `tEnter` pruning). Triangles use the **Woop watertight** test (JCGT 2013):
  per-ray axis permutation + shear, then three scaled barycentric edge functions
  `U`/`V`/`W`. The watertight guarantee is that two triangles sharing an edge evaluate it
  from *bitwise identical operands in opposite order*, so their edge functions are exact
  negatives and a ray dead-on the edge is claimed by exactly one (the accept test rejects
  only **mixed** signs, so a zero is accepted). **This means the three `a*b - c*d` products
  must not be FMA-contracted** — an FMA keeps one product exact and rounds only the other, so
  on an exact tie the two sharers compute `exact(pq) - rounded(qp)` and
  `exact(qp) - rounded(pq)`, which are *equal* rather than negatives; if that sign is the
  minority one **both reject and the surface cracks**. MSVC's default `/fp:precise` does not
  contract (the build sets no `/fp:` or `/arch:` flag), so the host needs nothing; the CUDA
  twin defaults to `-fmad=true` and must go through `dCrossRn`
  (`__fmul_rn`/`__fsub_rn`, plus `double` overloads for the `Real = double` build) — see
  `render_cuda.cu`. Exactly the same failure and the same fix as the rasterizer's
  `edgeRow`/`edgeAt` below. Fixed 0.116.0; measurement in `known-issues.md`.
- **`mesh.h`** (+ `gltf.h`, `fbx.h`/`fbx_load.cpp`) — OBJ (custom fast parser:
  single fread, in-place float/int scan), glTF/GLB subset, FBX geometry-only.
- **`implicit.h` / `isomesh.h`** — implicit/isosurface evaluation and marching-cubes
  tessellation. `marchImplicit` is staged **fill → discover → resolve → wind**:
  parallel lattice `val[]` fill and parallel per-vertex bisection refine + gradient
  (pure per-slot), with the order-sensitive weld-map sweep and winding pass kept
  serial — bit-identical to the old serial code by construction. Per-implicit
  marching also runs in parallel across objects.

  **Occlusion any-hit fast path (0.118.0).** `intersectImplicit` takes an
  `anyHit` flag (default false), set only by `Scene::occluded`'s traverseAny
  callback (device twin: the `occluded` loop in `render_cuda.cu`). Occlusion
  callers discard the `Hit` — they need a boolean — so once the sphere-tracer
  finds a sign crossing whose bracket already lies inside `[tmin, hit.t)` the
  ~27-eval bisection refine, 4-eval gradient and `writeHit` are all skipped:
  the bracket `[ta,tb] ⊆ [t,tn] ⊆ [t0,t1]` with `t0 = max(tmin, tEnter)` and
  `th = ½(ta+tb) ∈ [ta,tb]` guarantees any refined `th` would be accepted, so
  `tn < hit.t` alone proves occlusion. Cap hits return `true` directly and the
  cap-normal setup is skipped. The host test is exact in double; the device
  (`Real = float` build) uses a conservative widened compare
  (`(double)t >= tmin && (double)tn < hit.t`) so float-rounding corner cases
  fall through to the unchanged exact path rather than flipping a result —
  bit-identical output by construction on both backends. Mode W's 16
  shadow rays per area light per diffuse vertex made occlusion ~88% of its
  implicit-scene time; this plus the pattern-VM CSE below took the gyroid
  scene 3.68× faster on CPU, 1.45× on GPU (hash-verified).
- **`render.h`** — CPU forward tracer (modes A/B/C + photon deposit for M/S/P):
  per-photon loop, Russian roulette, sphere-scan cos/sin tables, splatting. The hero
  variant `tracePhotonHero` follows the same **max over live λ** RR rule as
  `backward.h` below (Diffuse, DiffuseTransmit's lobe pick, and the achromatic delta
  lobes Mirror/Filter/Glossy, which keep the bundle) — with one extra obligation the
  backward tracer doesn't have: the forward tracer keeps an **energy ledger**
  (`emitted = absorbed + sensor + escaped + residual`), and the survivors' reweight
  `beta[i] *= c_i/q` is *deterministic absorption*, so each reweight books
  `e.absorbed += beta[i] * (1 - c_i/q)`. Omit that and `sum/emitted` collapses
  (measured 0.66); the old ratio reweight *created* ledger energy (up to 1.007). With
  the booking the ledger closes identically, which it never did before.
  The bounce loop lives in **`tracePhotonHeroLoop`**, split out of `tracePhotonHero`
  so the opt-in `-herosplit` policy can **re-enter it recursively** — once per
  secondary — at a dispersive interface, instead of de-hero'ing. Each sub-path
  re-runs the same interaction with its own λ (its own Snell direction / grating
  order) on its own `MediumStack` copy, then continues monochromatically; the branch
  is guarded on `secAlive`, so a sub-path never re-splits and recursion is at most
  one level deep (cost linear in C, bounded stack). Weights are untouched — the C
  sub-paths keep `base/C` each and the parent zeroes them — so the ledger stays
  exact. The policy is a whole-run choice, so it rides on the `hero::gSplit` global
  that `Renderer::heroSplit` default-initialises from, rather than being threaded
  through every entry point the way per-pass `heroC` must be.
- **`backward.h`** — CPU backward reference tracer (`radianceHero`). Every Russian
  roulette in the hero path uses the **max over live λ** as its survival
  probability, with survivors reweighting `thr[i] *= c_i/q ≤ 1` — never the hero's
  own coefficient with a `c_i/c_hero` ratio, which amplifies a secondary whenever
  the hero λ is the dark one (a saturated wall spectrum makes that a 15× weight
  spike). That applies to Diffuse, DiffuseTransmit's lobe pick, and the achromatic
  delta lobes Mirror/Filter/Glossy, which for the same reason as BDPT's
  `keepBundle` do **not** de-hero (their outgoing direction ignores λ). At
  `nUp == 1` every one of these is the scalar code verbatim.

  **Split-at-dispersion (`heroSplit`)** mirrors the forward tracer's policy: the bounce
  loop is factored out as **`radianceHeroLoop`** (`radianceHero` is now a thin wrapper
  that seeds `thr[i] = 1`), so at a Dielectric / ThinFilm / Multilayer / Grating /
  HalfMirror / Fluorescent vertex each secondary λ can **re-enter the loop** with `C == 1`
  from bounce `b+1`, carrying its own Snell/grating direction, its own `MediumStack` copy
  and its own throughput. Two things make it exact rather than approximate: (1) each
  sub-path's radiance lands in the parent's **own `L[i]` slot** — `Lout` is *assigned*, so
  the parent accumulates per-λ rather than into the hero — and (2) there is **no ×C boost
  anywhere**, since the caller already divides by C. `secAlive = false` on the sub-paths
  bounds recursion to one level and cost to linear in C. The emitter-SPD cache is
  re-pointed **zero-copy**: the table is emitter-major with stride C (`spd[e*C + i]`), so
  offsetting the base pointer by `i` while *keeping* the stride makes `spd[e*C + 0]` read
  emitter `e` at λ_i. `Renderer::heroSplit` default-initialises from `hero::gSplit`
  (`-herosplit`), and `main.cpp`'s backward worker ORs in `br.whitted` — mode `W` has no
  choice (see below), mode `R` averages the collapse away so it stays opt-in.

  **A `layered` coat is a λ-dependent DECISION, not a λ-dependent direction — so it takes a
  per-λ weight and a shared coin, not a de-hero.** The coat interface changes neither the
  outgoing direction (a glossy lobe about the mirror direction) nor the wavelength; the only
  thing λ touches is the scalar reflectance `layeredCoatReflectance(…, λ)`. So
  `radianceHeroLoop` evaluates that per live λ and, when all live λ land on the same side of
  the reflect-or-enter decision, carries the whole bundle through: mode `W` multiplies each
  channel by its own `R_i` (or `1 - R_i`) and terminates only once `maxOf(thr, nUp)` falls
  under `kWhittedCutoff`, exactly like Mirror/Filter/Glossy. The stochastic path draws **one
  shared coin** and compares it per λ, which needs no reweight at all: `u` is uniform, so
  `P(u < R_i) == R_i` exactly for each λ — common random numbers, where the probability *is*
  the weight, as in the scalar twin. When the live λ *disagree* (a high-contrast iridescent
  film, or a Fresnel coat sitting right on mode `W`'s `R ≥ 0.5` threshold) the backward loop
  fans out like the dispersive case, except each sub-path re-enters at **this same bounce**
  rather than `b + 1` — legal because a bundle wider than 1 is never inside a medium (every
  dielectric entry de-heros or splits), so the loop head's Beer-Lambert step was a no-op and
  cannot be double-applied. `tracePhotonHeroLoop` gets the same shared coin but falls back to
  `deHero()` on disagreement, because a *forward* sub-path cannot re-enter its vertex: the loop
  head has already run the model-C aperture catch, and re-entering would deposit the photon
  into the film twice. Before v0.115.1 both loops de-hero'd at every coat unconditionally,
  which in mode `W` — whose λ lattice is per-*sample*, shared by every pixel — collapsed the
  whole frame onto one wavelength and rendered every coated surface saturated green.

  **Mode `W` (the `whitted` flag)** shares this whole walk and swaps only the
  *estimators*, which is why it is a flag and not a second tracer. Every stochastic
  decision on the path gets a deterministic replacement: the area-light NEE point
  becomes an N×N lattice (`lightGrid`, `-whitted-grid`); the glossy lobe keeps ONE
  direction, weighted by its reflectance, but takes it from a 2-D radical-inverse lattice on
  the power-cosine lobe (`whittedGlossyDir` → `glossyDirUV` in `render.h`) rather than the
  rng — the path is deliberately **not** forked, which would cost N^depth inside a gyroid
  labyrinth. That lattice is what makes the mode *consistent* on rough specular: collapsing
  every sample onto the mirror direction (pre-0.109.0) meant extra spp bought edge
  antialiasing and nothing else, so a satin metal never converged at any budget (measured:
  6 % better over 256× the samples, versus 19× better now). The polar coordinate is
  *complemented* rather than `rot05`-rotated because `glossyDirUV` maps `u1 == 1` to the
  mirror direction and `radicalInverse(0) == 0` in every base — so sample 0 reproduces the
  old behaviour bit-for-bit and only `spp > 1` changes. Each bounce depth draws from its own
  prime pair so two glossy vertices on a path are not driven by one sequence.
  Russian roulette on Mirror / Filter /
  Grating / specular-bundle survival becomes `whittedAttenuate` (multiply the
  throughput by the weight, stop under `kWhittedCutoff` = 1/512 — POV-Ray's
  `adc_bailout`); HalfMirror / Layered take the dominant branch weighted (Layered per-λ, see
  above, so the bundle survives a clearcoat), and a Mix
  hard-thresholds via `mixResolveDominant` (`scene.h`); **Dielectric** likewise takes the
  dominant Fresnel branch (reflect iff R ≥ 0.5) with that branch's weight folded into the
  throughput, via `refractOrReflect`'s `whittedWeight` out-param (`render.h`) — which also
  skips the frosting perturbation, the other rng draw at that interface (0.107.0; before
  that a dielectric was the one estimator left tossing a coin, and at `-spp 1` a coin flip
  per pixel is not noise but salt-and-pepper — glass rendered as a speckled blob).
  **ThinFilm / Multilayer** take the *same* `whittedWeight` contract as of 0.112.0
  (`thinFilmInterface` / `multilayerInterface`): a lossless substrate reflects iff R ≥ 0.5 with
  weight R or 1−R, while an *opaque* substrate — where transmission is absorbed, so there is
  only one surviving branch — always reflects with weight R, the reflectance becoming a
  throughput weight rather than a survival probability, exactly as Mirror/Filter do. Those two
  were missed in 0.107.0 (no `whitted` branch at all) and stayed noisy in the noise-free mode
  until an N3b CPU/GPU A/B measured them at 6.7 codes of block-luma disagreement.
  **`gratingDiffract`'s diffraction-order pick** and **`Fluorescent`'s Stokes-shift λ_in** were
  the last two, fixed in 0.113.0. Neither is a dominant-branch problem — they are discrete draws
  from a *distribution*, so the fix is N2's rather than `whittedWeight`'s: keep one analog choice
  per sample (candidate `i` with probability `w_i/Σw`, throughput unchanged — so it stays unbiased)
  but index it by `(sIdx, bounce)`. `whittedOrderU` (bases 43/47/53/59) and `whittedFluoroU`
  (61/67/71/73) in `backward.h` supply the coordinate, passed to `gratingDiffract` as an optional
  `const double* whittedU` and to `fluoInSampler.sampleAt`; every stochastic caller passes `nullptr`
  and is bit-identical. `whittedOrderU` is deliberately *not* `rot05`-rotated, because on the
  whitted path `gratingDiffract` walks its candidates in **descending efficiency**
  (`0, −1, +1, −2, +2, …`) instead of `mm = −M..+M`, so `u == 0` at sample 0 selects the specular
  order m = 0 and extra spp fan the spectrum into the higher orders — the same "sample 0 is the
  mirror direction" convention as the glossy lobe. `whittedFluoroU` *is* rotated, since no
  excitation λ is privileged that way and the median of the CDF is the better single sample.
  Env NEE is stochastic by deliberate choice on both devices.

  **All of those lattices go through a DIGIT-SCRAMBLED radical inverse** (`radicalInverseScr` /
  `goldenDigitMul` in `backward.h`, `dRadicalInverseScr` on the device), which is load-bearing
  rather than cosmetic. A plain radical inverse in base *b* returns exactly `i/b` for `i < b`, so
  its first *N* points cover only the prefix `[0, N/b)` — and every base above is *larger* than a
  typical preview's `-spp`. Unscrambled, base 13 kept a glossy lobe hugging its mirror direction
  until `-spp 13`, base 43 made a grating's higher orders arrive in a lump at `-spp 43`, and base
  61 pinned the fluorescent λ_in to the long half of the illuminant CDF so a dye absorbing only
  below 480 nm contributed **exactly nothing** until `-spp 64`, then switched on in one step. The
  fix is Faure's standard one for high-dimensional Halton: permute the digits,
  `r = Σ π(dₖ)·b^-(k+1)`. π is multiplicative, `π(d) = (d·m) mod b` with `m = round(b/φ)`, which
  needs no permutation tables (so the CUDA twin is trivially bit-identical), is a bijection for
  every prime base, and — the load-bearing property — has **π(0) = 0**, so sample 0 still maps to
  exactly 0 in every base and every "sample 0 is the canonical outcome" contract above (mirror
  direction, specular order, median λ, pixel centre) is untouched: all `-spp 1` images are
  bit-identical across the change, and only `spp > 1` moves. Measured star discrepancy of the
  first 16 points drops from 0.754 to 0.077 at base 61 and 0.651 to 0.102 at base 43
  (`scraps/n3e_lattice.py`).

  **A fluorophore's excitation λ comes from the material's OWN distribution, not the scene's.**
  `Material::fluoInSampler` (`scene.h`) is an `EmissionSampler` over the product
  `clamp01(fluoAbsorb(λ)) · g(λ)`, where `g` is the same combined illuminant `Scene::emitSampler`
  covers. It is built in `finalizeEmitters()` — after the emitter list is final, and rebuilt
  whenever that list changes (`-ignoreenv`) — and consumed by `backward.h`'s `Fluorescent` case,
  which then uses `invPdfIn = 1/pdf` from the sampler that actually made the draw rather than the
  analytic `Scene::invPdfLambda`. Two things follow. In the stochastic modes it is plain
  importance sampling: draws no longer land above a dye's absorption edge and return zero.
  In mode `W` it is a *correctness* property, because the single 1-spp coordinate is a CDF
  median: the median of the illuminant is ~575 nm (past a blue-absorbing dye's edge, so the dye
  previewed as its bare elastic lobe), whereas the median of absorption × illuminant is 422 nm
  for a `shortpass edge=480` dye under 6500 K, where `aEff` = 0.83. That is why the dye is right
  at `-spp 1` since v0.115.0. Device twin: a second CDF slice in the existing flat
  `DScene::fluoCdfAll` (`fluoInCdfOffset/N/Step`) plus `dSampleFluoInU`. Unbiasedness is asserted
  numerically, not argued: `-checkfluoro` estimates the reradiation NEE weight from both the old
  and the new sampler and requires both within 2 % of an analytic quadrature.

  The wavelength and the subpixel offset come off radical-inverse sequences instead of the rng.
  Glass is deterministic too, because mode `W` forces **`heroSplit`** on (see below): a
  dispersive vertex fans the bundle into C monochromatic sub-paths rather than de-hero'ing
  onto one λ. That is not an optimisation here but a correctness requirement — the λ
  lattice is shared by every pixel, so de-hero'ing would collapse the *whole frame* onto
  the *same* wavelength and mistint every dispersive object (measured 36.7 pp of chroma
  error on a Cornell SF10 ball; splitting gives 0.80 pp at 1 spp, beating 16 stochastic
  passes' 4.20 pp at 7.9× the speed). It implies
  `directOnly`, with `ambient` (a flat term at each diffuse vertex) as the GI
  stand-in — pre-scaled by `Scene::ambientRef()` so the CLI value is dimensionless.

  Two invariants matter here. (1) **Every pixel uses the same offsets** — that is
  what makes the mode noise-free, since neighbours then differ only by geometry, not
  by luck. (2) The sample sequences are indexed by the **absolute** sample index and
  are *progressive*, exactly like the rng stream's `seedUnit`, so the image is
  independent of the chunk split. A per-chunk `(s+½)/spp` lattice would silently
  collapse to "sample 0 forever" under `-window` (which chunks into 1-spp batches)
  and make `-spp` a no-op on the image — this was a real bug, fixed in 0.105.0.
  Mode `W` also raises the default hero bundle to `kHeroMax`, since at 1 spp the C
  wavelengths *are* the whole spectral quadrature and they share one BVH walk.

  **The deterministic one-bounce gather (`giDirs`, `-gi`)** is mode `W`'s real GI, the
  thing `ambient` only stands in for. `giGatherHero` / `giGather` trace `giDirs` rays
  from a diffuse vertex along a fixed lattice (`giDir`: an `n`-point Fibonacci spiral on
  the whole sphere, Cranley-Patterson-rotated by two radical inverses of the absolute
  sample index), keep the ~half with `cos > 0`, weight by `cos`, and normalise by the
  realised cosine sum. Each gather ray re-enters `radianceHero`/`radiance` recursively
  with `GiCtx::depth == 1`, which is the single switch behind four behaviours: no second
  gather (single bounce), `giGrid` instead of `lightGrid` for shadow rays, `giBounce`
  instead of `maxBounce` as the loop cap, and `specularArrival` starting **false** so a
  gather ray landing straight on an emitter adds nothing (the vertex's own NEE already
  counted that) while one arriving *via* a specular bounce still does. An escaped gather
  ray picks up `ambient` as the far-field term, which is what makes the two flags
  compose: in an empty scene every direction escapes and the normalised gather collapses
  exactly back to `rho * ambient`, so switching `-gi` on never steps the exposure.
  `scraps/gi_collapse.ftsl` is the regression test for that normalisation — a lone diffuse
  quad lit only by `ambient`, where `-gi 32` and `-gi 0` must be **pixel-identical**
  (verified: 0 of 25600 pixels differ, on the CPU and the GPU).
  That scene's `lumens` is load-bearing and must not be removed: it forces **absolute**
  mode, i.e. a fixed sensor gain. Until 2026-07-30 it lacked one and the test was run with
  `-exposure 1`, which made it **vacuous** — the image is a flat uniform patch, so the p99
  auto-exposure anchor divided out *any* overall scale factor and the test passed no matter
  how badly the estimator mis-scaled, which is the one failure it exists to catch. The
  header now carries a discrimination check (`-ambient 0.05` must render exactly half as
  bright as `-ambient 0.1`) so the test cannot silently go vacuous again.
  Note the two hemisphere rejections (`cos <= 0` on
  the shading normal, and on the oriented geometric normal so a smoothed normal cannot
  gather through the true back face) drop directions *without* adding them to `wSum`, which
  is what keeps that collapse exact on a smooth-shaded surface rather than darkening it.

  Three design constraints drove this rather than a POV-Ray-style irradiance cache.
  (1) **Temporal stability.** A cache's sample set depends on render order and on local
  geometry, so on animated geometry its blotches pop between frames; the lattice is a
  pure function of (index, sample index) and never of the scene, so a seamless loop
  cannot flicker. (2) **No tangent frame.** The lattice lives in world space, so there
  is no orthonormal-basis discontinuity to show as a seam, and a direction crossing the
  horizon does so at `cos == 0` — the estimate is continuous in the normal, which is
  what makes a rotating object's shading slide instead of pop. (3) **Every pixel shares
  the rotation**, preserving the mode's core noise-free invariant; raising `-spp`
  rotates the whole frame's lattice coherently, so the residual banding refines
  progressively instead of being re-rendered identically.

  **Measured outcome** (all-diffuse Cornell, `scraps/cor_gi.ftsl`, vs mode `R` at 0.8 %
  noise; see `scraps/cor_eval.py`). Whole-frame mean |luminance error| 7.01 for the best
  flat fill vs **5.40** for `-ambient 0.01 -gi 32` (23 % better), but colour bleeding 18 %
  → **80 %** of the reference (4.5× better) — the gather's value is overwhelmingly in the
  effects a constant *cannot* produce, not in the mean level, which a well-tuned constant
  already gets close to. The gather saturates near `-gi 32` (12.91/12.80/12.77/12.77 for
  32/64/128/256 with no tail), because past that the limit is the single bounce, not the
  direction count: in a closed 0.75-albedo box the interreflection series is ~4× the first
  bounce. Hence the two flags are complements rather than alternatives — `ambient` stands
  in for the *tail of the series*, and `-gi 32` plus a small tail beats `-gi 256` alone.
  Temporal stability is verified in `scraps/gi_temporal.py` (normalised second difference
  1.907 for `-gi 32` vs a 1.851 direct-only control).

  **The firefly clamp (`giClamp`, `-gi-clamp`, 0.117.0)** is the price of that shared
  lattice. The gather's dynamic range is enormous: a ray that bounced off a wall returns
  ~`rho/pi` times a small solid angle of the lamp, while one that reaches the lamp *through*
  a glass ball returns the lamp's **full** radiance — two orders of magnitude more. That
  caustic path is real and the emitter hit is its only estimator (NEE structurally cannot
  sample a lamp behind a refracting surface, and `specularArrival` is deliberately set back
  to `true` by the dielectric so the hit counts). But `giDirs` fixed directions cannot
  *resolve* where the caustic lands, and because every pixel shares those directions, "does
  direction `k` reach the lamp through the ball?" is a step function whose boundary is one
  coherent image-space **contour** — so the error surfaces as thin, blown-out, dashed curves
  rather than as the grain a stochastic renderer would show. `-gi-clamp x` caps one gather
  ray's return at `x` times `Scene::ambientRef()`. Three choices are load-bearing:
  *per wavelength, not per bundle*, because the scalar twin `giGather()` carries a single λ
  and has nothing to take a max over, so a bundle-wide rule would make the hero and single-λ
  paths disagree on the same scene (this file works hard to keep those two identical);
  *`wSum` is left untouched*, so a clamped direction keeps its weight `c`, the estimator
  still normalises by the realised cosine sum, and an unclamped gather is bit-for-bit
  unchanged (verified against the 0.116.0 baseline PNG with `cmp`); and *the clamp also caps
  the far-field `ambient` tail an escaping ray returns*, which is not a bug but does couple
  the two knobs — the gather's fill level is exactly `min(ambient, giClamp)`, pinned in
  `scraps/gi_collapse.ftsl` where `-gi 32 -gi-clamp c` is pixel-identical to
  `-gi 0 -ambient min(ambient, c)`. Hence the documented rule "keep it above `-ambient`":
  below that it darkens the whole scene instead of only capping fireflies.
  Measured on `scraps/gi_firefly.ftsl` (the absolute-mode Cornell box *with* the glass ball,
  the deliberate counterpart to all-diffuse `cor_gi.ftsl`) at `-gi 32 -spp 1 -ambient 0.05`:
  `-gi-clamp 0.1` costs **0.31 %** of frame luminance and drops the >250-code pixel count
  from 1689 to 1350, while the caustic survives as a soft highlight; `0.2` costs 0.20 %,
  `0.5` costs 0.04 %. The `-gi-clamp 0.05` row costs 0.86 % precisely because it is *at*
  `-ambient` and starts eating the fill. CPU↔GPU agreement with the clamp on is 98.7 %
  bit-identical, max block |dLuma| 0.328 code (`scraps/n3b_check.py`, 1.5-code bar).

  Two evaluation traps worth remembering, both of which produced confidently wrong numbers
  before being caught. (1) **Auto-exposure hides `-ambient` entirely**: the anchor is the
  frame's own 99th percentile, so raising the fill raises the mean and the anchor divides it
  straight back out (measured 5× anchor swing over `-ambient 0 → 0.30`). A sweep run that
  way concluded that a flat fill made the image *worse* than none at all. Any `-ambient` or
  `-gi` comparison must put the scene in absolute mode (`lumens`/`power` on a light).
  (2) **Pick a scene whose dominant error is the one being measured**: on `gold_gyroids`
  the glossy-lobe collapse dwarfs the missing diffuse GI, and on `scenes/cornell` the
  dielectric bug does — in both cases the gather looked nearly worthless.

  `DiffuseTransmit` gathers **both** lobes (the hit normal and a flipped copy), since a
  translucent surface receives from the full sphere. This also fixed a gap: in the
  non-gather path it now adds `(rhoR + rhoT) * ambient`, where previously a
  `DiffuseTransmit` vertex in mode `W` received **no** `ambient` fill at all. So a
  translucent material renders brighter under `-ambient` than it did in v0.105.0 — an
  intentional behaviour change, and the reason a scene using `translucent` will not match
  a v0.105.0 mode-`W` render pixel-for-pixel.

  `-gi` is rejected with a message outside mode `W` (`main.cpp`), since every other mode
  either has real multi-bounce GI or no diffuse transport at all. Mode `R` is untouched:
  every gather branch is gated on `whitted` and every new parameter defaults to the old
  behaviour (`GiCtx{}` → depth 0 → `lightGrid`, `maxBounce`, `specularArrival = true`).

  **Mode `W` on the device (0.110.0).** Mode `W` is not a separate GPU kernel: it rides the
  same `kBackward` megakernel as mode `R`, exactly as on the CPU, and swaps only the
  estimators. The plumbing is a `WhittedOpts` struct (`render_cuda.h`) — the twin of
  `BackwardRenderer`'s mode-`W` fields, with `ambient` already pre-scaled by
  `Scene::ambientRef()` — passed as a trailing `const WhittedOpts*` to `renderBackwardCuda`
  (`nullptr` = mode `R`, so every existing call site is unchanged). It lands in seven
  `DScene` knobs (`bkWhitted`, `bkGrid`, `bkGiDirs`, `bkGiGrid`, `bkGiBounce`,
  `bkHeroSplit`, `bkAmbient`) whose `buildUpload` defaults are "mode `W` off", which is what
  keeps every stochastic mode bit-identical. The lattice helpers are ported one-for-one
  (`dRadicalInverse2` / `dRadicalInverseB` / `dRot05` / `dWhittedSample` /
  `dWhittedLambdaU` / `dWhittedGlossyDir` / `dWhittedAttenuate` / `dGridUV`) and compute in
  `double` off integer inputs, so the *quadrature points themselves* are bit-exact against
  the CPU even though the trace around them is not. `render.h`'s `glossyDirUV` /
  `sampleGlossy` factoring is mirrored on the device so the deterministic and stochastic
  lobe samplers share one body, and `refractOrReflect` / `dDielectricStep` gained the same
  `whittedWeight` out-param contract.

  Two details of the port are load-bearing. (1) **`sIdx` is not `gidx`.** The device seeds
  its rng on `gidx = pix*sppTotal + sampleBase + local`, which deliberately *includes* the
  pixel; mode `W`'s lattices must use the pixel-**free** `sIdx = sampleBase + local`, or
  invariant (1) above — every pixel shares the offsets — is broken and the "noise-free"
  mode comes out noisy. (2) **The u1/u2 emitter-sample coordinates are now caller-supplied.**
  `bkEmitterGeom` used to draw them internally; the G×G quadrature has to feed them from the
  lattice, so they became parameters and `dEmitterNeedsUV` (twin of `emitterNeedsUV`) tells
  the caller when to draw. The draws had to move to exactly the same point in the rng
  stream — a scene that merely *added* a spot light would otherwise reshuffle every other
  emitter's stream and change an unrelated stochastic image.

  `cudaBackwardWhittedSupported()` gates the device path on top of
  `cudaBackwardSupported()`. It used to be deliberately **narrower** — a missing
  deterministic term is a visible error, not extra noise, so unsupported constructs fell
  back to the CPU mode-`W` tracer (cheap — the mode is ~1 spp) rather than degrading. As of
  **0.116.0 nothing mode-`W`-specific narrows it any more**: it forwards straight to
  `cudaBackwardSupported()`. Dispersive materials stopped gating at 0.111.0 and `giDirs > 0`
  at 0.116.0 (N3c, below). `Layered` needs no device twin at all: it already forces a CPU
  fallback device-wide via `cudaForwardSupported`. Env NEE stays **stochastic** in mode `W`
  on both CPU and GPU (an existing deliberate choice), so it needed no device change — an
  important *non*-change, since "fixing" it on one side only would have manufactured a
  CPU/GPU divergence.

  **Split-at-dispersion on the device (0.111.0).** Dispersion-dependent materials
  (Dielectric / ThinFilm / Multilayer / Grating / HalfMirror / Fluorescent) used to gate the
  whole scene back to the CPU, because `bkRadianceHero` *de-hero'd* at those vertices and
  mode `W` cannot use a de-hero: its λ lattice is per-**sample**, shared by every pixel, so
  collapsing onto the hero collapses the entire *frame* onto one wavelength and mistints every
  glass surface (36.7 pp of chroma error, measured — see N1). The fix mirrors the CPU
  refactor: `bkRadianceHero`'s body became `template<bool AllowSplit>
  bkRadianceHeroLoop(...)`, taking the whole bundle state (`ro`/`rd`/`stk`/`lam[]`/`invPdf[]`/
  `thr[]`/`C`/`secAlive`/`specularArrival`/`contBsdfPdf`/`bounce0`) so it can be **re-entered**
  mid-path, with `bkRadianceHero` left as the thin "fresh bundle at bounce 0" wrapper that
  picks the instantiation off `sc.bkHeroSplit`. `AllowSplit == true` fans each live secondary
  into its own monochromatic sub-path from bounce `b+1` — its own direction, its own
  `DMediumStack`, its own `L[i]` slot, **no ×C boost** — via `bkRadianceHeroLoop<false>`;
  `AllowSplit == false` keeps the de-hero.

  The `if constexpr (AllowSplit)` is the load-bearing part. A runtime `if (heroSplit)` would
  leave a self-recursive call in the `false` body, which on the device means unbounded stack;
  as a compile-time switch nvcc emits two bodies and the `false` one contains **no recursive
  call at all**, so the re-entry is provably one level deep, the frame is statically sized,
  and no `cudaLimitStackSize` / `-rdc` is needed. (The CPU gets the same bound from
  `secAlive`, but can afford real recursion.) `sub[]` and the copied `DMediumStack` add ~100
  bytes to the `true` instantiation's frame.

  `bkHeroSplit` is therefore the one mode-`W` `DScene` knob that is **not** mode-`W`-only:
  `-herosplit` applies to plain mode `R` as well, so `buildUpload` defaults it from
  `hero::gSplit` rather than to 0 (the same pattern as `BackwardRenderer::heroSplit` /
  `Renderer::heroSplit`). Before 0.111.0 GPU mode `R` silently ignored `-herosplit` and
  de-hero'd where the CPU split — a real CPU/GPU divergence, fixed by the same code. The GPU
  *forward* megakernel still de-heros, which the `[hero]` startup line now says explicitly.

  **The `-gi` one-bounce gather on the device (0.116.0, N3c).** This was the last mode-`W`
  CPU fallback. The host gather is a *recursion* — a diffuse vertex shoots `giDirs` gather
  rays back into the same `radiance()` — and CUDA recursion would need `-rdc` plus a
  hand-sized device stack. So the depth became a **second, independent compile-time
  parameter** alongside `AllowSplit`: `bkRadianceHeroLoop<bool AllowSplit, int GiDepth>`,
  `bkRadiance<int GiDepth>`, `bkRadianceHero<int GiDepth>`, and `bkInteract<bool
  AllowGather>`. `GiDepth == 0` is a camera path and is the *only* instantiation that
  contains a gather call at all; the rays it spawns are `GiDepth == 1`, whose body contains
  none, so the whole thing is provably finite with statically-sized frames. A split sub-path
  **inherits** its parent's `GiDepth`, so the deepest chain is `<true,0>` → gather →
  `<true,1>` → split → `<false,1>`: three nested tracer frames, no more. `bkInteract` takes
  a `bool` rather than an `int` because the hero tracer handles `Diffuse`/`DiffuseTransmit`
  inline with the whole bundle and routes only *dispersive* materials to `bkInteract`, which
  can therefore always pass `false` — two instantiations instead of three.

  Depth 1 changes four things, all mirrored from the host: the coarser `bkGiGrid` shadow
  lattice instead of `bkGrid` (its soft-shadow detail is about to be averaged over `giDirs`
  rays anyway, so `bkGrid²` there would be wasted work), the `bkGiBounce` depth cap, a
  non-specular arrival, and the escaped-ray far-field tail. The bounce cap is
  `min(bkGiBounce, bkMaxBounce)` — the `min` is load-bearing, since without it a `-gi-bounce`
  larger than `-max-bounce` would let a *gather* ray trace deeper than the camera path it
  hangs off.

  The lattice itself (`dGiDir` / `dGiPhases`) is a fixed Fibonacci spiral over the **whole**
  sphere, built in **world space**, Cranley-Patterson-rotated by scrambled radical inverses
  of the absolute sample index in bases 7 and 11 (which collide with neither the subpixel
  lattice's 2/3, the wavelength's 5, nor the glossy/discrete lattices' ≥13). World space
  rather than a tangent frame is deliberate: no orthonormal basis means no basis
  discontinuity to show up as a seam, and a direction entering or leaving the hemisphere does
  so at `cos == 0`, i.e. with **zero weight**, so the estimate is continuous in the normal and
  a rotating object's shading slides instead of popping. It is cacheless and non-adaptive, so
  the residual error is low-frequency banding rather than noise — and because every pixel
  shares the phases, raising `-spp` rotates the whole frame's lattice coherently and the
  banding averages out.

  The **normalisation invariant** is what makes the feature safe to switch on: normalising by
  the *realised* sum of retained cosines makes the estimator exact for constant incident
  radiance, so in an empty scene every gather ray escapes, each returns `ambient` via the
  escaped-ray tail, and the whole gather collapses to exactly `rho * ambient`. `-gi 0` and
  `-gi 32` are therefore **pixel-identical** on an empty scene and turning `-gi` on never
  steps the exposure. `scraps/gi_collapse.ftsl` is that test, and it holds bit-for-bit on the
  device.

  **Whole-image bit-exactness is not achievable and is not the acceptance bar.** The device
  runs `using Real = float` (`FTRACE_GPU_FP32`) with `RAY_EPS` 1e-4 against the CPU's
  `double`/1e-6, and CUDA libdevice's transcendentals differ from the MSVC CRT's in the last
  bits. The bar instead is: bit-exactness on the lattice helpers, plus image agreement to
  fp32 tolerance with **no structural difference**. Measured (`scraps/n3_gpu.ftsl`, 800×520,
  `-spp 16`, absolute exposure, `scraps/n3_check.py`): 99.39 % of channel samples identical,
  99.96 % within one 8-bit code, and of the 113 pixels over a 3-code threshold **zero** sit
  inside a ≥3 px-wide region — i.e. the residual is entirely single-pixel slivers on
  silhouettes and shadow edges, which is the signature of epsilon rather than of a porting
  bug. `n3_check.py`'s blob-vs-sliver split is exactly that test. Same scene: 12.1 s (12 CPU
  threads) → 0.3 s (4090). Cross-checks that the shared stochastic path is intact: mode `R`
  CPU↔GPU means agree to 0.04 %, modes `B`/`D` to ~0.1 %.

  `-rgb` is refused in mode `W` (`main.cpp`, with a message): the fast RGB backward is a
  separate reduced tracer with no deterministic estimator, so it would return precisely the
  noise the mode exists to remove.
- **`bdpt.h`** — BDPT with MIS; vertices stored by **index** (never `Vertex&`
  across `push_back` — a use-after-free lived here once; see known-issues).
  Hero-wavelength capable (`HeroBundle` on both subpaths, `Vertex::betaSec/nUp`,
  per-λ connection terms under one shared hero-driven MIS weight; the splat
  normalises by `1/min(nUp_light, nUp_eye)` instead of a per-subpath ×C boost).
  The per-λ scatter factor `secF[]` is **absolute** (`f_i·cos/pdf_hero`), never a
  ratio to the hero's, and the walk's early-out is a max over live λ — so a lobe
  whose hero value is 0 (a gel filter, a saturated albedo) can't drop live
  secondaries. Delta vertices de-hero *except* Mirror and Filter, which pick their
  continuation without consulting λ and so set `keepBundle`.
- **`vcm.h`**, **`sppm_render.h`**, **`photonmap.h`/`photonmap_render.h`** — U/S/M.
  PhotonMap::build precomputes per-photon CIE X/Y/Z (the 3.65× mode-M win); VCM
  caches CIE lookups; kd/grid structures for gathers.
  **`PhotonMap` is structure-of-arrays, and that is load-bearing.** Deposit positions
  live in `pos[]`, everything the gather reads after acceptance in `photons[]`, and the
  precomputed CIE triple in `cie[]` — three arrays permuted together by the counting
  sort, so index `k` names one photon in all three. The reason: a radius-`r` query scans
  the 3×3×3 cell box but keeps only the inscribed sphere, so ~85% of candidates are
  rejected on a distance test that needs the position and nothing else. Interleaved,
  that scan strided a fat record and used a fraction of each cache line it pulled; split,
  it is a dense 24 B/photon stream. This dominates precisely where mode M hurts — a dense
  map is gigabytes, so the gather is DRAM-bandwidth-bound. `Photon` therefore holds only
  `n`/`power`/`lambda`; the incident direction it used to carry was never read by any
  gather (the estimate is Lambertian) and was pure per-photon waste — the device deposit
  record `DPhoton` lost the same field for the same reason, which matters extra there
  because its buffer is sized from *free VRAM*, so fewer bytes per record is directly more
  photons the GPU can hold (44 → 32 B). The GPU's gather record `DGatherPhoton`
  (render_cuda.cu) is the same idea, and additionally folds
  `cie*power*norm/pi` into three floats. The `-savemap` cache format is `FTPMP02`
  (two blocks: positions, then payloads); `FTPMP01` files are rejected with a message
  telling the user to re-deposit.
  **The mode-M gather radius is density-adaptive** (`PhotonMap::buildAuto`, default on;
  `-nopmauto` or an explicit `-pmradius` opts out bit-identically). `build(r)` sizes the grid
  at `cellSize == r`, so a radius chosen from scene size alone freezes the grid and makes
  photons-per-cell — and gather cost — grow *linearly* with `-n`. `buildAuto` therefore bins
  once at the requested radius as a probe, asks `medianNeighborCount()` what a typical gather
  actually sees, and re-bins at `r·sqrt(k/n)` for a target `k = kAt1M·cbrt(M/1e6)`, `M` =
  stored photons. The cube root is a deliberate choice, not a free parameter: holding the
  disc population *constant* (`r ∝ M^-1/2`) would hold variance constant too, so the image
  would never converge in noise, only in bias. `r ∝ M^-1/3` gives per-query cost `M^1/3`,
  noise `M^-1/6` and bias `M^-2/3` — both error terms → 0, with a mild cost curve.
  `build` is split into `buildGrid` + `fillCie` for this, since the probe needs a second
  counting sort but only one (expensive, threaded) CIE pass.
  **`medianNeighborCount` samples by cell, not by array index** — cells are fixed by the
  bbox and cell size, i.e. by geometry, whereas the counting sort is stable and so preserves
  a within-cell order that differs between a fresh deposit and a `-loadmap` of the same map.
  Sampling by array position therefore made `-loadmap` stop reproducing its `-savemap` run.
  Within a sampled cell the representative is the lexicographically smallest position (a
  set-minimum, hence order-free); the cell *centre* would not do, because a cell that the
  surface merely clips has its centre off-surface and reports a spuriously empty
  neighbourhood.
  The GPU shared path gets the same treatment via `renderPhotonMapSharedCuda`'s `autoK`
  argument (0 = off) — it must, since that is the high-photon-count path where a
  count-independent radius collapses worst. The gather reads `pm.radius` after the build, so
  the adapted value needs no further plumbing.
- **`spectrum.h` / `spectral_library.h` / `upsample.h` / `color.h` / `hero.h`** —
  spectral core: measured SPDs/materials, RGB→spectrum upsampling, CIE tables,
  hero-wavelength sampling (`kHeroC=4`: hero λ + 3 stratified secondaries) used by
  R, A/B/C, M/S and BDPT D on CPU and the GPU forward/backward/BDPT megakernels
  (the GPU BDPT keeps its per-vertex secondaries in a parallel array and templates
  `kBdptT<NS>` on the slot count, so the scalar instantiation costs no extra local
  memory). `hero.h` is deliberately thin and dependency-free: it owns the two shared
  constants, the stratified bundle draw `sampleBundle()` (templated on the sampler so
  it serves both the forward tracer's per-emitter `Emitter::spd` and the
  backward/BDPT tracers' scene-wide `Scene::emitSampler`), the live-λ maximum
  `maxOf()`, the `gSplit` policy flag, and — most importantly — the **single
  authoritative statement of the four hero policies** (stratification; which lobes
  de-hero and why the criterion is a λ-*dependent direction* rather than mere
  delta-ness; analog RR is max-over-live-λ; per-λ factors are absolute not ratios),
  along with the two rules that are deliberately per-tracer (the forward energy
  ledger must book the RR reweight as absorption; BDPT normalises by
  `1/min(nUp_light, nUp_eye)` instead of a ×C boost). The four tracers are different
  *estimators* rather than four copies of one, so they are **not** unified behind a
  shared `HeroLambda` struct — only the genuinely identical pieces and the policy
  prose are shared, because that prose going stale in three files is what let each
  of this feature's two real bugs be fixed independently two-to-four times.
  Emitter SPD sampling is
  cached per light. Tabulated curves (FTSL `table { }` / `file:`) build a
  `Spectrum` via `tabulatedSpectrum` (piecewise-linear, default) or
  `tabulatedSpectrumMono` (opt-in `interp=cubic`: monotone Fritsch–Carlson/PCHIP —
  C¹ but shape-preserving, no overshoot, so an interpolated reflectance/absorption
  can't ring outside its neighbouring samples). Both clamp to the endpoints outside
  the sampled range (no extrapolation); `loadSpdFile` warns once if a `file:` curve
  doesn't span the 360–830 nm render range. Since `Spectrum` is
  `std::function<double(double)>` evaluated at each photon's exact λ, the interpolant
  shape shows directly (there's no pre-binning), which is why overshoot matters.
- **`upsample.h` — the RGB→spectrum upsampler family.** Five reconstructions of a
  spectrum from a colour triple, each reached by its own FTSL head keyword
  (`rgb`/`hsv`/`hsl` + suffix), all sharing one `upsample::Basis` (95 samples,
  360–830 nm at 5 nm; weights `k·D65(λ)·CMF(λ)·Δλ`, `k` normalised so unit
  reflectance integrates to `Y=1`). They differ in what they optimise, which is why
  more than one is kept: **Jakob–Hanika** (`rgb`, the default) fits a 3-parameter
  sigmoid — always in `[0,1]`, cheap, but the *shape* is whatever the sigmoid
  family allows; **JH illuminant** (`rgbillum`) is the same fit renormalised for
  emission; **Smits 1999** (`rgbsmits`) mixes seven fixed basis curves; **3-box**
  (`rgbbox`) solves a 3×3 for one flat step per band — exact round-trip but blocky;
  **Meng 2015** (`rgbmeng`) is the *smoothest* physical reflectance producing the
  colour. Only under a non-D65 illuminant (or under dispersion) does the choice show
  in the render — every upsampler round-trips its own colour under D65 by
  construction, so it is the reconstructed *shape*, not the colour, that differs
  (`scraps/meng_test.ftsl` makes this visible by lighting four identical panels with
  illuminant A). `-checkupsample` validates all five: round-trip error, `[0,1]`
  physicality, and — for Meng specifically — that its roughness is provably below
  JH's on every test colour.
  Meng is table-driven, and the table (`src/meng_table.h`, ~140 KB) is **baked by us**
  (`tools/bake_meng.py`), not transcribed from the authors' supplemental — the method
  is published, the published data carries no licence. Two deliberate departures from
  the paper make the bake much simpler *and* more accurate for our use:
  (a) the grid is barycentric **in the sRGB primary triangle** rather than over the
  spectral locus, because every colour ftrace upsamples arrives as `rgb r g b` with
  components in `[0,1]` and therefore already lies inside that triangle — the
  barycentric coordinates are just `(r·S_R, g·S_G, b·S_B)` normalised (`S_i` = column
  sums of `linSrgbToXyz`), so there is no search, no cell classification, no locus
  polygon; and (b) each vertex is solved at **`Y=1` with no upper bound**, not at some
  fraction of max brightness with `s ≤ 1`. (b) is the load-bearing one: the
  minimum-roughness solution is linear in the target XYZ only if the feasible set is a
  *cone*, and `{s ≥ 0}` is while `{0 ≤ s ≤ 1}` is not. Tabulating against an active
  upper bound silently destroys the very property being tabulated — scaling the stored
  spectrum down to a darker colour stops being optimal — which is exactly the bug that
  first made `-checkupsample` report Meng as *rougher* than JH. The renderer scales to
  the requested luminance and clamps at use time. Interpolation weights each cell
  vertex by `bary_k / T_k` (`T_k = X+Y+Z` of its spectrum) rather than by `bary_k`
  alone, because a chromaticity is an `(X+Y+Z)`-weighted mean — that is what makes the
  interpolated chromaticity *exact* rather than merely close.
- **User-supplied upsamplers (`upsample "n" { expr … }`, head `rgb:<n>`).** The sixth
  member of the family, and the only open-ended one: the scene supplies the function.
  Deliberately *not* in `upsample.h` — the five above are numerical fits, this is a
  compiled pattern-VM program, so it lives at the loader (`ftsl.h`: `upsampleBlocks_`,
  `applyUpsample`) with `pattern.h` supplying two new pieces. Three decisions carry it:
  (a) **the body's vocabulary is disjoint from the surface one, not additive**
  (`PatVarMode::Upsample`). `r` already means *radius* in a surface program and must
  mean *red* here, so an additive design would make one spelling silently mean two
  things. Instead every surface name is rejected by name with a message saying so, and
  `r`/`g`/`b`/`w` internally reuse the `VarX`/`VarY`/`VarZ`/`VarU` slots as a pure
  register assignment — invisible, because the surface spellings are unreachable in
  this mode. (b) **`spec:<name>(w)` (`PatOp::Spec`)** samples a declared `spectrum` at
  the queried wavelength, which is what makes a *measured basis* expressible
  (`r*spec:red(w) + …`) rather than only closed-form arithmetic; it resolves its index
  at compile time through a `PatSpecScope`, exactly like `tex:`/`grid:`, and is a
  compile error outside an upsample body (an ordinary pattern has a hit point but no
  wavelength — and symmetrically `tex:`/`grid:` are errors *inside* one). `PatOp::Spec`
  never reaches the device: an upsample program is consumed at load time and is never
  stored on a `Material` or in `Scene::patterns`. (c) **the result is a live closure,
  not a baked table** — a user upsampler may be a narrow emission line, and
  pre-tabulating here would band-limit it; the renderer already tabulates where it
  needs to (`double reflect[SPEC_N]`), at a resolution it chooses. The closure captures
  the compiled program and the spectrum vector by `shared_ptr` (and the sampler thunk's
  `self` is the *vector*, not the Builder), so the produced `Spectrum` outlives the
  loader; the vector is append-only so an index handed out at compile time survives
  later growth. `isColourHead` gained one shape-only arm (`isCustomColourHead`) so the
  head is accepted at both sites that share that list — value site and record-channel
  inline-colour tag. Pinned by `-checkupsample` section (h); `scenes/_upsample.ftsl`
  is the visual companion; loom's twins are `NamedSpectrum`/`Upsample` (`scene.py`),
  `UserSpec` (`grammar/spectrum.py`) and `is_colour_space` (`record.py`).
- **`camera.h` / `lens.h`** — camera models incl. finite thin-lens, fisheye/pano,
  realistic multi-element lens; `scene_film.h` film/EV/auto-exposure (p99),
  exposure-lock anchors.
- **`materials.h` / `pattern.h` / `texture.h` / `layered`** — BSDFs (diffuse,
  mirror, glossy, dielectric w/ nested IOR, diffuse-transmission, filter gels,
  fluorescence, layered), procedural patterns (POV-derived `pov_noise.h` /
  `pov_functions.h`), UV texturing. **Tangent-space normal maps** (`normal_map
  texture:<name> strength <s>`): per-triangle tangents are built in `Tri::finalize()`
  from the UV gradients (Gram-Schmidt vs the geometric normal + a stored
  `bitangentSign`); `Scene::applyNormalMap` remaps the linear-`encoding` map's texel
  to a `[-1,1]` vector and rotates it through the surface TBN frame. It is invoked at
  the single intersection choke point (`closestHit`/`closestHitLinear` on the CPU,
  `dApplyNormalMap` in the device `closestHit`) so every renderer and both devices
  perturb shading identically; tangents transform with instances (`instanceHitToWorld`,
  the device uploading a per-instance `Wm` = toWorld linear).

  **Pattern-driven reflectance (`Material::reflectPat`).** Patterns originally drove only
  *scalar* slots (`roughness`, `film_thickness_map`, `weight_map`), because `reflect` is
  spectral and a pattern is a scalar. The resolution is that a scalar in a spectral slot is
  a per-hit **multiplier**, not a replacement — which is strictly more general than the
  obvious "greyscale reflectance" reading and degenerates to it: `reflect pattern:<n>` (and
  `reflect [0 1](u)`, which desugars to it) leaves the pattern alone in the slot, so the
  loader stores a flat-1.0 base and the multiply *is* the albedo; `reflect_map pattern:<n>`
  beside an authored spectrum or a bound `reflectTex` modulates that. Colour therefore
  cannot come from a pattern by construction. The multiplier is clamped to [0,1] (no energy
  from a formula). It is applied in exactly the two shared accessors that already funnel
  every renderer's reflect read — `diffuseReflectance` and `reflectSlot` in `scene.h`, plus
  device twins `dDiffuseRho` / `dReflectSlot` — so all six tracers and both backends pick it
  up from one edit, and `renderBackwardRGBCuda`'s baked-RGB fast path opts out alongside the
  existing `reflectTex` opt-out. The families whose reflect slot does *not* go through those
  accessors (Fluorescent's `fluoroWeights`, which has no hit to evaluate at; ThinFilm;
  Dielectric) are **rejected at load** rather than silently ignored, because the flat-1.0
  base a lone `reflect pattern:` leaves behind would otherwise render as albedo 1.0 — a
  wrong image rather than a missing effect.

  **Pattern-driven transmittance (`Material::transmitPat`).** The same mechanism on the
  `transmit` slot: `transmit pattern:<n>` / `transmit [0 1](u)` alone in the slot, or
  `transmit_map pattern:<n>` modulating an authored spectrum. Reaching it needed a
  refactor first — unlike `reflect`, the transmit slot had **no shared accessor** and was
  read as a bare `m.transmit(lambda)` at 16 sites across 7 renderer headers (`render.h`,
  `backward.h`, `bdpt.h`, `vcm.h`, `photonmap_render.h`, `sppm_render.h`) plus 16 device
  sites in `render_cuda.cu`. Those now all funnel through `transmitSlot(scene, m, h,
  lambda)` in `scene.h` (device twin `dTransmitSlot`), which is the single point of truth
  for **both** readings of the slot: a `filter`'s per-wavelength gel transmittance
  T(λ), and a `translucent`'s back-hemisphere Lambertian albedo ρ_T. Callers keep their
  own `clamp01` and, for the two-lobe case, still apply the ρ_R + ρ_T ≤ 1 energy guard
  *after* the multiplier. There is no record channel and no texture on this slot, so the
  accessor has only the one base path (simpler than `diffuseReflectance`). Only those two
  families are honoured — every other type leaves `transmit` at its 0 default and never
  reads it — so a transmit pattern anywhere else is a load error, same policy as
  `reflect`. The reflect and transmit loader paths are themselves now one function
  (`patternedSpectrumParam` + `checkSlotPatSupported`, keyed on the slot name), and
  `renderBackwardRGBCuda` opts out of its baked-RGB fast path on `transmitPat` too.

  **Pattern-driven emission (`Material::emitPat` / `Emitter::emitPat`).** The third leg of
  the trio, and the strict one. Spellings match: `emit pattern:<n>` alone in the slot (base
  collapses to flat 1.0 ⇒ the pattern *is* a greyscale emission profile) or `emit_map
  pattern:<n>` modulating an authored spectrum; a `light` block spells the same slot `spd`,
  so there the pair is `spd pattern:` / `spd_map pattern:`. `finalizeEmitters` copies the
  material's `emitPat` onto the `Emitter` it registers, so the two spellings converge on one
  runtime field. Emission is stricter than reflect/transmit because it is read from **both
  sides of transport**: once when a camera path lands on the emitter (emission-on-hit, PatCtx
  built from the `Hit`) and once at the point NEE / a light subpath samples on it (PatCtx
  built from `Emitter::samplePoint`). MIS *combines* those two estimates, so if they ever
  disagreed pointwise the image would be **biased**, not merely noisy. The profile is
  therefore only legal where the sampler's (u,v) provably equals the (u,v) a hit interpolates
  — `EmitterShape::Quad` (bilinear parameters) and `EmitterShape::Mesh` (barycentric UVs on
  `EmitTri`, which gained `uv0`/`uvE1`/`uvE2`). Every other shape (sphere, cylinder, spot,
  sun, collimated, env) is **refused at load**, at two points: `addLight`'s subtype gate, and
  `checkEmitPatsSupported` after `Scene::build()` for the material route. Making the quad
  agree required fixing a latent pre-existing bug — the area light's *second* triangle
  carried default UVs disagreeing with `addQuad`'s, i.e. a diagonal seam for any UV-driven
  emission pattern **or texture** on an area light. `Emitter::samplePoint` gained optional
  `uuOut`/`vvOut`, and the read is funnelled through three accessors in `scene.h`:
  `emitSlot` (emission-on-hit), `emitterPatMulAt`, and `emitterSamplePoint` (sample + return
  the multiplier in one call). The pattern is a **pure post-multiplier on radiance / photon
  beta**: `Emitter::power`, `pdfChoice`, `pdfPos`/`pdfA`, `emissionPdfW`, `directPdfW` and
  every VCM `dVCM`/`dVC`/`dVM` are deliberately untouched, which is exactly what makes it
  unbiased by construction (no selection or positional pdf changes anywhere). The cost is
  variance on a mostly-dark profile, and the fact that `power`/`lumens` normalise the
  *unpatterned* spectrum, so a profile averaging 0.5 emits about half the requested flux —
  documented rather than auto-corrected, since folding the mean into `power` would need a
  compensating 1/mean on photon beta and on BDPT's `pdfChoice`. Verified unbiased at 160×160
  against four independent estimators (R vs D within 0.3%, with a pattern-free control
  showing the *same* residual, so it is a pre-existing R-vs-D difference; B/400M photons
  within 0.005%; U/VCM within 0.02%). **CPU-only in 0.80.0**: the device has ~20 emission
  read sites, and a partial port would be *biased* rather than visibly incomplete, so
  `cudaForwardSupported` (and `cudaBackwardRGBSupported`) reject the whole scene and the CPU
  renders it; the port is tracked in `known-issues.md`. The preview rasteriser ignores
  `emitPat`, consistent with its existing treatment of `reflectPat`/`transmitPat`.

  **The `emit` slot name is overloaded, and `fluorescent` owns it.** On every other material
  type `emit` means *self-emission* (`Material::emit` + `isLight = true`, i.e. the surface is a
  light). On a `fluorescent` it means the **reradiation profile** — the Stokes-shifted emission
  *shape*, consumed into `fluoEmit`/`fluoEmitSampler` and normalised by its own integral `Mint`
  at every use site — so a fluorescent surface is never a light. `src/ftsl.h` therefore skips
  the generic "any material may carry an `emit` spectrum" block for `MatType::Fluorescent`;
  before v0.113.1 that block ran unconditionally, so a fluorescent's reradiation band was
  *also* installed as absolute-radiance self-emission and the surface glowed ~4 orders of
  magnitude too bright on the CPU (and not at all on the GPU, which uploads no per-material
  emit spectrum) — see `known-issues.md`. `emit_map` is hard-refused on a fluorescent rather
  than silently ignored, since a reradiation profile is not a surface pattern. A surface that
  both fluoresces *and* self-emits is expressible as a `mix` of the two materials.

  **N-D authored-data tables.** A pattern formula can sample arrays of authored numbers in
  1–4 dimensions, via two sibling datatypes ported from loom's `data.py`/`interp.py`:
  `grid:<name>(c0, …)` reads a **regular lattice** (`PatGrid`, samples in C order with axis 0
  outermost, separable N-linear over 2^ndim corners, `clamp`/`wrap`/`extrapolate` outside the
  box), and `scatter:<name>(c0, …)` reads **arbitrary positions** (`PatScatter`, Shepard
  inverse-distance weighting `1/(d²)^(power/2)`, a coincident sample returned exactly).
  Both live in `pattern.h` as `__host__ __device__` samplers (`patGridSample` /
  `patScatterSample`) so there is no device re-implementation to drift. Architectural notes:
  (a) both are the only ops whose **arity is not a property of the function name** — it is the
  table's own `ndim`, resolved at tokenize time through a `PatTableScope` (a kind-dispatching
  `{Grid, Scatter}` callback the FTSL builder installs), so a wrong argument count is a
  compile error rather than a silent zero; (b) tables address their numbers by `int off` +
  `int count` into **one flat `Scene::dataPool` shared by both kinds** — never by pointer,
  since the pool grows as later tables load — which is also exactly the layout the GPU
  uploads, so a scene costs one allocation for its tables however many it declares. The
  scopes are separate namespaces (a grid `foo` is not `scatter:foo`).

  **Reaching the field formulas (0.78.0).** Unlike `tex:`, which genuinely needs a hit's
  (u,v) and so stays a compile error outside a shading context, a table sample needs only
  coordinates — so the FTSL builder passes `&tableScope_` at four further compile sites:
  the `function` field leaf, a medium's `density` and `ior` programs, and `camera_curve`
  drivers. Evaluation is the harder half: a compiled `PatOp::Grid` node carries an index
  into `Scene::grids`, so **every** evaluation site must be able to see those vectors.
  The mechanism is `PatTables` — a non-owning `{grids, scatters, dataPool}` view built by
  `Scene::patTables()` (device: `dPatEnvOf(DScene&)`) and threaded as a **parameter**
  through `Implicit::eval`/`gradient`, `intersectImplicit`, `estimateFieldLipschitz`,
  `Medium::densityAt`/`nAt`/`gradNAt`/`insideBound`, the GRIN marcher, `isomesh::marchImplicit`
  and `airtight::check`. It is deliberately *never* a member: a `Scene` is copied and moved
  (`buildCornell` returns by value), so a stored view would dangle. The two multi-medium
  wrappers (`sampleMediaCollision` / `mediaTransmittance`, and their device twins) were
  retyped to take the whole `Scene`/`DScene` rather than `media`, so no caller can *forget*
  the tables. Load-time consumers use the real tables too — the Lipschitz bound, the
  majorant-density scan and the `boundInsideNeg` sign test would otherwise all read a
  grid-driven field as identically 0.

  The evaluator's "table not found" guards **abandon the program** (`return 0`) rather than
  pushing a placeholder, because the operand count is the missing table's own `ndim`: a push
  would leave the stack unbalanced and quietly return a *coordinate* as the result. That was
  a live wrong render before 0.78.0, reachable via `medium { density pattern:<p> }`, which
  copies a table-scoped pattern's nodes into a medium evaluated without tables.

  **Pattern-VM CSE (0.118.0).** Field formulas are the sphere-tracer's inner loop —
  `patternEval` was ~73% of mode W's CPU time on the gyroid scene — and authored
  expressions repeat subtrees heavily (each `6*rot(p)+φ` appears in both a `sin` and a
  `cos`; the rotated coordinates share `(x-c)`/`(z-c)`). `patternOptimizeCSE`
  (`pattern.h`) rewrites a compiled program so each repeated subtree executes **once**:
  it rebuilds the postfix into a hash-consed DAG (key = op + raw `a` bits + child ids),
  counts uses, ranks interior nodes reached ≥2× by `(uses−1)·subtreeSize`, assigns up to
  `PAT_CSE_REGS = 32` registers, and re-emits via a memoized post-order walk using two
  new ops appended at the enum's end — `StReg` (peek top-of-stack into `reg[a]`) and
  `LdReg` (push `reg[a]`). Three invariants carry the bit-exactness argument: (1) the
  ops are **pure data movement**, and a register load reproduces exactly what
  re-executing the identical subtree yields *in the evaluator's own precision* — so the
  optimization is bit-identical in the host double evaluator **and** the device float
  evaluator (`dPatternEvalF`) even though those two disagree with each other. Constant
  folding was rejected for exactly this reason: host-double folding would change the
  float path's bits. (2) It is hooked **only at the final-consumption site**
  (`ftsl.h addFunctionLeaf`, after `compilePatternExpr`) — never inside compilation —
  because `patternSubstitute` splices programs together and two pre-optimized fragments'
  register indices would collide. (3) It is belt-and-braces conservative: programs with
  `Grid`/`Scatter` (arity lives in the table, not the node) bail, the re-emitted program
  is re-simulated for stack balance, and 6 full-variable probe points are bit-compared
  (`memcmp`) old-vs-new before the rewrite is accepted — any mismatch keeps the
  original. Since the ops are appended at the enum end, the `VarX..VarV` range tests
  (`patternHasFreeVars`, the free-variable scans) are unperturbed; all **three**
  evaluator switches (host `patternEval`, device `dPatternEval` / `dPatternEvalF`) carry
  the two cases, and the `PatNode → PatNodeF` upload conversion is memberwise so a
  register index ≤ 32 survives the float trip exactly.

  **Inline array literals** (`roughness [0 1](u)`, `weight_map [[0 0.5][0.5 1]](u,v)`) are
  the write-it-where-you-use-it spelling of the same thing, and they are implemented as
  **pure sugar**: a loader pre-pass (`Builder::desugarArrays`, run immediately before the
  grid/scatter pass) turns each literal into an anonymous `grid "__arrN"` plus a one-line
  `pattern "__arrN" { expr "grid:__arrN<call>" }` appended to the block list, and rewrites
  the value site to `pattern:__arrN`. That is why a literal works at *every* slot that
  already accepts a pattern without any of those slots changing, and it is verified by
  rendering `scenes/pattern_array.ftsl` against a hand-written `grid` + `pattern` twin
  (bit-identical). Nesting is the shape; the domain is the **unit box** per axis (an inline
  literal has no domain of its own), deliberately unlike the `grid` element's index-lattice
  default. In the grammar the syntax is a `PARENWORD` terminal (a token that is *wholly*
  parenthesised) plus one merged `selector = '[' sel_item* ']' axistuple?` production
  covering both jobs of `[ … ]` at a value site. (The hand-written tokenizer, retired in
  0.79.0, needed no change at all — a call arrived as an ordinary bareword because ftrace
  never treated `(` as a delimiter, which is exactly what keeps `0.5+0.5*sin(2*pi*u)` one
  token.) Critically, **the front end does not decide what the brackets mean**: it collects
  the raw `ftsl::BrItem` tree and hands it to `ftsl::applyBracketGroup` at the loader level.
  That split is what kept the two front ends from drifting on the one syntax that is
  genuinely ambiguous (record stop selector vs. array literal) while both existed, and it
  is why the flip needed no loader changes.

  **What the sample call's arguments *are*.** `desugarOne` splits the call into top-level
  comma-separated arguments (`splitCallArgs`, the same "a top-level `=` is unambiguous
  because the pattern language has no comparison operators" rule `parseBindArgs` uses) and
  pins one semantics: **an argument is a driver — a coordinate expression — and a literal's
  axes are anonymous and bind by position.** So `[0 1](a)` is not a special "formal"
  construct at all; it is an ordinary coordinate that happens to name `a`, the one input
  with no per-hit intrinsic, which is what makes it survive to the use site where the §7.6
  material-bundle substitution can rebind it (`ramp(a=u)`, `ramp.reflect(a=u)`, positional
  `ramp.reflect(u)`). A literal's "formals" are therefore just the driver names in its own
  tuple, which is exactly what a rebind substitutes, and why a 2-D literal transposes under
  the simultaneous `(u=v, v=u)`. The corollary is a refusal: `formal=driver` **inside** a
  literal's own call has no formal to bind and is a load error naming both working
  spellings, because honouring it would require inventing a per-material default for `a`
  that two literals in one material could contradict. loom's `values.py` refuses the
  identical spelling (`_check_args(..., literal_target=True)`) so a scene cannot emit from
  loom and then fail to load. `-checkarray` pins the identities against independently
  authored twins (with explicit non-vacuity checks — an unbound grid pool makes every
  sample 0.0, which would make every identity pass for the wrong reason);
  `scenes/_array_formal.ftsl` shows them per tile, and `tools/measure_array_formal.py`
  reduces that render to per-tile mean/gradient numbers recorded in the scene's own header.
  The render is only the qualitative confirmation — it agrees to within its noise and the
  room's lighting profile, and the twelve tiles do not sit in a uniform field, so the
  measurement corrects each tile's horizontal gradient using the four tiles whose albedo is
  constant in `u` (one per column). `-checkarray` is what pins the identities exactly.

  **Composed literals** (`[0 1]([0.2 0.8](u))`, 0.100.0) fall straight out of "an argument
  is a coordinate expression": a coordinate may itself be a sampled value, so a literal is
  legal wherever one is. The implementation is deliberately asymmetric with the value-site
  case. `Builder::buildArrayGrid` emits the anonymous `grid` for both, but a composed
  literal gets **no `pattern` wrapper** — `grid:__arrN(coords)` is already a legal
  pattern-expression term, so the composition is spelled by textual substitution into the
  outer call (`Builder::desugarNestedLiterals`, recursive, so nesting is unbounded) and the
  outer expression ends up as `grid:__arr1(grid:__arr0(u))`. Only the value site needs a
  name for a *statement* to reference, and only it pays for a second block.

  The reason this needed a grammar change is worth recording, because it is the one place
  the lexer's design leaks. A sample call is a single `PARENWORD` token, and that terminal's
  interior class excluded `[` / `]` until 0.100.0 — so an inner literal's brackets split the
  outer token and the outer literal reported the (very misleading) *unsaturated* error. The
  fix widens the interior class only; the terminal's **balance guarantee rests entirely on
  `(` and `)` staying excluded** (two paren groups on one line cannot merge, because merging
  would have to consume the intervening `)` as an interior char), so admitting brackets
  costs nothing there. Brackets inside a call are therefore *captured but not
  balance-checked by the lexer*, which is the right trade: `desugarNestedLiterals` parses
  the argument text itself (`Builder::parseArrayText`, reproducing the tokenizer's own
  splitting rule) and reports an unbalanced or call-less inner literal against the author's
  source, instead of surfacing as a token that mysteriously fails to match. It also refuses
  a literal written flush against an identifier (`f[0 1](u)`) *before* substituting, since
  the rewrite would otherwise glue the author's token to a generated name and complain about
  an "unknown identifier `fgrid`" that appears nowhere in their file. The pattern language
  has no bracket syntax of its own, which is what makes "a `[` here always opens a literal"
  safe to assume. Note this touched `ftsl_scene.epeg` only — loom's reader uses the sibling
  typed `ftsl.epeg` — so loom's 1255-test suite is the regression check that the shared
  grammar tooling still round-trips.

  **A table call is a value in its own right** (`reflect grid:ramp(u)`, 0.101.0). TODO.md's
  increment-2 `Deferred:` clause claimed `NAME axistuple` "needs no work because ftrace's
  expression evaluator already reads `name(args)` as a call" — true *inside* a pattern
  expression, but a **value site is not an expression site**, and the slot readers only ever
  recognised `pattern:<name>` there, so `reflect grid:ramp(u)` was an "unrecognized spectrum
  expression". Closing it hooks the same four chokepoints v0.89.0 used for
  `MATERIAL.slot(args)`: `bindScalarPattern` and `patternedSpectrumParam` (the two per-hit
  readers) route a `grid:` / `scatter:` head through `Builder::tableCallPattern`, which
  compiles the token with the ordinary `compilePatternExpr` and appends the result to
  `scene.patterns` — the slot ends up holding exactly the index a hand-written one-line
  `pattern { expr "grid:ramp(u)" }` would have produced, so there is no second evaluation
  path to keep in step. `dblParam` and `evalSpectrum` are the load-time-constant readers and
  **refuse**, each naming the slots that can take a per-hit value.

  Two consequences are worth recording. First, **only the scoped spelling is accepted**: a
  bare `ramp(u)` at a value site already means the §7.6 material-bundle application, so
  accepting it for tables would make a scene's meaning depend on which namespace happens to
  hold the name — `isTableCallHead` therefore tests for the `grid:` / `scatter:` prefix and
  nothing else, and a call-less `grid:ramp` is a refusal that prints the `(u)` to add rather
  than a silent constant. Second, a table call's coordinates are expressions like any other,
  so a **composed array literal** must work inside one (`grid:ramp([0.2 0.8](u))`).
  `Builder::desugarTableCall` reuses `desugarNestedLiterals` on the token before it is
  compiled, and `desugarArrays`' visit loop dispatches to it on the cheap
  `isTableCallHead && contains('[')` test so a scene with no literals pays nothing. That in
  turn needed the *second* grammar change: a **named** table's call lexes as a `WORD`, not a
  `PARENWORD`, so `WORD`'s balanced-group alternative had to be widened to
  character-for-character `PARENWORD`'s body, brackets included. Only the group *interior*
  admits them — `WORD`'s fallback class still excludes `[` / `]`, so a bracket outside parens
  is a delimiter exactly as before and `REC.chan[2]` still stops the word at the `[`.

  **Generated blocks are re-attributed.** `desugarArrays` mints `grid`/`pattern` blocks the
  author never named, so `Builder::genSite_` maps `__arrN` back to the authoring site and
  `genWho()` is used wherever those blocks can fail. A message about `pattern '__arr3'`
  would name a symbol that appears nowhere in the scene file. A composed literal registers
  its own site too, so a bad coordinate two levels down still names what the author wrote.

  The same bracket spelling is accepted for a **`grid`/`scatter` element's own `data`**, and
  there it is *not* sugar: `desugarArrays` deliberately skips those two block types, because
  a `data [ … ]` group is the element's samples rather than a value-site literal (there is no
  sample call to make — one there is an error naming the right place to put it). `addGrid`
  and `addScatter` read the `BrItem` tree directly through the shared `flattenArray`. For a
  grid the **nesting is the shape**, so `shape` need not be written at all (writing one that
  disagrees is an error printing both); a *flat* group carries no shape, so `shape 2 2` still
  folds `data [0 1 2 3]`. For a scatter, one group per sample, each `dim+1` numbers wide.
  This is loom's `data.py` constructor convenience carried over verbatim in spirit, and it is
  verified bit-identical against the `shape` + flat-`data` spelling.
- **`envmap.h` / `sky.h`** — infinite environment lighting. `EnvMap` turns an
  equirectangular linear-RGB buffer into an importance-sampled directional emitter
  (per-texel Jakob–Hanika spectral upsampling + a luminance·sinθ 2-D sampler);
  `buildFromRgb` is the shared entry so both the image loader and analytic generators
  use it. `sky.h` is the **Preetham analytic daylight sky**: the Perez five-parameter
  distribution for luminance + CIE xy, scaled by turbidity/sun-elevation zenith values,
  baked to an equirect image with a spectrally attenuated (Rayleigh + Ångström aerosol)
  5778 K solar disk, then handed to `EnvMap::buildFromRgb` — so an analytic sky reuses
  the entire env pipeline, including the GPU `DEnvMap` upload, for free. Magnitudes are
  physical (the sun is ~10⁵× the sky) then normalised to a mean sky luminance of
  `intensity`. (Efficient-directional-sun forward sampling is a logged follow-up.)
- **`medium_stack.h` / `phase.h` / `grin.h` / `rainbow.h` / `vdbgrid.*` / `vdb_openvdb.cpp`** —
  participating media (bounded, density fields, superposition), HG + water-droplet
  (rainbow) phase functions, gradient-index bending, NanoVDB (`.nvdb`) + native OpenVDB
  (`.vdb`, self-contained BLOSC/LZ4 reader) density import. Imported volumes are baked to a
  dense lattice stored as **fp16 half-floats** (`halfBitsToFloat`/`floatToHalfBits` in
  `vdbgrid.h`, mirrored by a `__device__` decoder in `render_cuda.cu`) to halve host/GPU memory.
  The GPU sampler is **natively sparse**: `VdbGrid::buildBricks` partitions the lattice into 8³
  bricks and uploads only occupied bricks + an int32 brick-index (empty brick → density 0), so
  VRAM scales with filled volume, not the bounding box — bit-for-bit identical to the dense
  sampler (the trilinear stencil is clamped before lookup). The host keeps the dense lattice.
  A multi-grid `.vdb` selects a grid **by name** (`loadVdbGrid(..., wantName)`; the OpenVDB reader
  seeks each descriptor to the previous grid's `endPos`, since descriptors interleave with bodies).
- **Volumetric blackbody emission ("fire")** — a `Medium` may carry a second `temperature` grid
  (`Medium::temperature`/`tempPeak`/`emitKelvin`/`emissionScale`; `emissive()`/`temperatureAt()`/
  `emissionAt()` in `scene.h`), turning its hot voxels into a self-illuminating isotropic volume
  emitter. `spectrum.h` supplies `blackbodyEmissionRadiance` (Planck normalised to a 6500 K/560 nm
  reference — physical T⁴ + Wien hue, tame magnitudes); temperature is peak-normalised
  (T=emitKelvin·raw/tempPeak). `Scene::finalizeEmissiveVolumes()` (called from `build()`) MC-estimates
  each grid's mean emission `meanKe` + selection `power`=4π·V·meanKe·Δλ into `Scene::emissiveVolumes`
  (+`totalEmissionPower`). Forward `tracePhoton` (`render.h`) splits birth emitter-vs-fire by power
  (`grandTotal=totalPower+totalEmissionPower`; no extra RNG when there are no emissive volumes, so
  non-fire scenes stay bit-identical); a fire photon is born uniform-in-AABB, isotropic-dir, with λ
  **importance-sampled from `blackbody(emitKelvin)`** via a per-volume `EmissionSampler lamSampler` (built
  in `finalizeEmissiveVolumes`), carrying **β=grandTotal·κ_e/(meanKe·Δλ·p(λ))** — for a voxel at emitKelvin
  β is constant across λ, collapsing the colour-magnitude speckle (uniform p=1/Δλ recovers the plain
  β=grandTotal·κ_e/meanKe). The isotropic `1/(4π)/(dist²·Ω)` splat
  (`connectEmissionVolume`/`connectEmissionLensVolume`/`camSplatEmissionAll`) reproduces the emission
  line-integral. **Forward CPU AND GPU** (A/B/C, V/P forward layers): the GPU mirror lives in
  `render_cuda.cu` — the VDB brick sampler is factored into a reusable `DVdbGrid`/`dVdbSample`, `DMedium`
  gains a `tempGrid` + emission params, `DScene` gains a `DEmissiveVolume[]` (+ per-volume Planck-λ CDF)
  and `totalEmissionPower`, and `genPhoton` has the same power-split volume-birth branch +
  `connectEmissionVolume`/`camSplatEmissionAll` device splat (validated GPU-vs-CPU on `scraps/vdb_fire.ftsl`).
  The backward reference (mode R/V) never samples the grid — it treats media as one homogeneous haze.
- **`rng.h`** — Pcg32 + `seedUnit(rng, unitIndex, salt)` splitmix64 mixing:
  **every work unit (photon or pixel-sample) seeds its own stream**, so results are
  independent of chunk splits / thread count / banding / `-resume` boundaries.
- **`render_cuda.cu`** (~7000) — the whole GPU backend: megakernel + wavefront
  forward paths, GPU R and BDPT, M deposit/gather, device twins of hero sampling.
  GPU backward (`bkRadiance`) supports **participating media** natively since
  0.23.0 (free-flight `dMediaSampleCollision` competing with the surface hit,
  volume NEE `bkNeeVolume`, Beer–Lambert `dMediaTransmittance` on NEE + throughput,
  HG scatter + albedo Russian roulette) — homogeneous *and* heterogeneous.
  Spectral **rainbow-phase** media run on the device too since 0.37.0 (M10):
  a per-medium λ×µ Airy phase table + per-λ CDF is uploaded, and the unified
  `dMedPhase`/`dMedPhaseSample` dispatch (bilinear table eval / CDF importance-sample
  vs analytic HG) replaces the raw `hgPhase` calls across forward, backward, and BDPT.
  **GRIN (gradient-index) media** run on the backward reference (mode `R`) on-device
  too since 0.38.0 (M11): `dGrinMarch` (render_cuda.cu) is the device twin of
  `grin::march`, carrying its running Eikonal state in double, and `bkRadiance` marches
  each bounce's ray before `closestHit` (gated by `sc.hasGrin`); only mode-`D` BDPT and
  the RGB fast path still route GRIN scenes to the CPU. Since
  0.24.0 it also does **fluorescence** (bispectral Stokes-shift adjoint: elastic +
  excitation-wavelength NEE, `gOut = M(lambda)/Mint * invPdf`, stochastic
  elastic/reemit/absorb continuation — baked `fluoEmitSpec`/`fluoMint` on the
  device material). Since 0.25.0 it does a **constant environment light**
  (`bkNeeEnv` surface-vertex + `bkNeeEnvVolume` fog-vertex env-NEE, uniform-sphere
  sample, balance-heuristic MIS against a tracked `contBsdfPdf`, MIS'd env-miss on
  ray escape; `envIndex` + env geomWeight added to `DScene`/`dInvPdfLambda`). Since
  0.30.0 (M1) it also handles an **IMAGE-based (lat-long HDR) env** on the GPU
  backward: `bkNeeEnv`/`bkNeeEnvVolume` branch on `sc.env.scale != nullptr` and
  importance-sample the map on-device via `dEnvSample` (the forward path's luminance
  2-D CDF), with `dEnvRadiance` (per-texel JH `coeff`·`scale`·`illum`) and `dEnvPdf`
  for MIS-consistent env-miss; the previously-canceled illuminant table is uploaded
  as `DEnvMap::illum`. Validated GPU==CPU to 0.14% in linear radiance at 8192 spp.
  The same `dEnvRadiance` also serves **mode-M (photon map) env** on the GPU (M2,
  0.31.0): the deposit already emits env photons (env's indirect bounces land in the
  map), and `dPhotonGather` adds env's direct term on gather-ray escape — so
  `cudaPhotonMapSupported` no longer rejects env scenes (validated GPU==CPU mean 0.18%,
  background sky 0.04%). Since 0.32.0 (M3) there is a resident **GPU SPPM** session
  (`SppmSession`, `cudaSppmSupported == cudaPhotonMapSupported`): per-pixel progressive
  state (`tau`/`radius`/`nAcc`/`directSum` + this pass's visible point) lives on the device
  across passes, and each pass runs `kSppmVisiblePoint` (resample the camera visible point +
  direct term via the specular walk), reuses the mode-M forward deposit (`launchForward`,
  fresh seed = cumulative emitted), builds the grid **on-device** at the largest current
  per-pixel radius (0.39.1: deposits stay resident in a grow-only device buffer; rMax via
  `transform_reduce`, `kSppmCellKey` + stable sort + `lower_bound` cell ranges,
  `kSppmGatherConvert` bakes the photon records with double-precision cie tables — per-pass
  PCIe traffic drops from the full photon slab to a few bytes; ~13× faster passes), then
  `kSppmGather` (query + Hachisuka shared-statistics radius/flux update) and
  `kSppmResolve` (`L = directSum/passes + tau/(pi R^2 Nemit)`). The SPPM photon record bakes
  `pX = cie(lambda)·power/pi` with NO area/nEmitted fold (those depend on the current
  per-pixel radius, applied at resolve) — unlike mode M, which folds them in. Validated
  GPU==CPU on a Cornell glass-sphere caustic (mean 0.2–1.2%, background wall 0.3%, per-pixel
  diff shrinking with passes). Since 0.33.0 (M4) the mode-M **`-pmfg` Jensen final gather**
  also runs on the device: `dPhotonGather` gained an `fgRays>0` branch (NEE direct term via
  `bkNeeLight` + `K` cosine-hemisphere sub-rays), each sub-ray handled by `dPhotonGatherSub`
  (device twin of `photonGatherSub`) — it follows specular surfaces, then at the first diffuse
  hit y does a radius density query folding `rho(y)·rho(vis)` per photon wavelength (spectral
  two-bounce colour bleed), plus env-on-escape / specular-arrival emitters reflected off the
  visible point. `fgRays` is threaded through `kGather`→`renderPhotonMapSharedCuda` and the
  `g_pmFinalGather==0` caller gates in `main.cpp` were dropped. Validated GPU==CPU on a Cornell
  glass-sphere+diffuse-walls box (mean 0.43%, background 0.98%, per-pixel noise √-scaling with
  spp — unbiased). Since 0.39.0 (M12) there is a resident **GPU VCM/UPS** session (mode `U`,
  `VcmSession`, `cudaVcmSupported == cudaBdptSupported && media.empty()` — surfaces-only, pinhole
  only) mirroring `vcm.h`'s `vcmPass`: each pass (1) `kVcmLight` traces one light subpath per pixel,
  storing connectible vertices into a **per-path slab** (`lvSlab[i·vcmCap+k]`, no cross-thread
  atomics) and splatting the connect-to-camera (t=1) light-image contributions (atomic into a
  per-pass double buffer); (2) compacts the slab **on-device** (0.39.1: thrust scans over the
  per-path counts → `pathBegin`/`pathEnd` + `kVcmCompactScatter`; only the 4-byte total vertex
  count crosses PCIe per pass, vs the former ~69 MB slab download) into contiguous per-path
  ranges so the same-λ vertex CONNECTION reads its PAIRED light subpath exactly
  (single-wavelength spectral BDPT); (3) sort-builds the uniform hash grid on-device
  (`kVcmCellKey` + `thrust::sequence` + stable sort + `lower_bound` — order-identical to the
  former host counting sort; cell = merge radius, reusing the M3 device-grid query layout;
  ~4× faster passes, byte-identical output); (4) `kVcmCamera` traces one camera
  subpath per pixel doing emission (s=0) / NEE (s=1) / paired-path vertex connection (c) / grid merge
  from ALL paths (d) under one **balance-heuristic** MIS (SmallVCM `dVCM`/`dVC`/`dVM` bookkeeping,
  misArrival/misScatter inlined with Mis=identity), accumulating the running per-pixel XYZ sum
  (camera radiance + the light splat); the resolve divides by the pass count. Reuses M9's device BDPT
  BSDFs (`dBsdfF`/`dBsdfPdf`/`DVertex`); `dVcmScatter` is the device twin of `scatterSample`. The
  merge builds the density estimate in XYZ (cie(λ) per merged light vertex, other paths' λ) exactly
  like modes M/S. main.cpp mode-U mirrors mode-S (auto/gpu device, radius schedule
  `r_i=R0·i^((α−1)/2)`). Validated GPU==CPU on `absolute.ftsl` (Cornell + dielectric sphere,
  fixed-gain absolute mode to bypass per-image auto-exposure) at 500 passes: mean linear-luminance
  ratio 0.9993 (−0.07%), per-channel bias ≤0.5% (R −0.43%, G −0.06%, B +0.20%), per-pixel median rel
  error 3.0% at the ~4.5% independent-MC noise floor — no systematic bias. Since 0.34.0 (M9) the **GPU BDPT** kernel (mode `D`) threads the **per-hit
  surface point** through its connection BSDF: each `DVertex` stores the interpolated texcoords
  (`u,v`) and `dVertHit` reconstructs a minimal `DHit`, so `dBsdfF`/`dBsdfPdf` and the random
  walk evaluate per-hit-driven throughput slots consistently in BOTH the sampler and the
  pdf/eval (MIS-safe: a textured albedo changes only `f`, not the cosine pdf; per-hit glossy
  roughness feeds the same `dMatRoughness` into sampler and pdf). On-device now: textured/
  patterned/record diffuse albedo & glossy reflect, per-hit glossy roughness + thin-film maps,
  mix blend masks, and Beer–Lambert **colored-glass** interior absorption (delta vertex →
  throughput only, mirroring `bdpt.h`'s `curAbsorb`). Since 0.35.0 two-sided **diffuse-transmit**
  (translucent) also renders on-device — both lobes (front-hemisphere `reflect`, back-hemisphere
  `transmit`, energy-clamped) plus the two-sided back-hemisphere connection (allow back hemisphere,
  skip the shadow-terminator softening, `|cos|` in G); `lambda` is threaded through
  `dBsdfPdf`/`dVertexPdfF`/`dMisWeight` because the lobe-selection pdf is wavelength-dependent.
  Since 0.36.0 **frosted (rough) dielectric** also renders on-device: `refractOrReflect`/
  `dDielectricStep` already jitter the chosen reflect/refract lobe by the per-hit roughness, so a
  rough dielectric is the same non-connectable **stochastic-delta** vertex on GPU as in `bdpt.h`
  (only the gate needed relaxing). With that, **all genuine per-hit-BSDF GPU-vs-CPU parity gaps in
  mode D are closed** (M9 complete): `cudaBdptSupported` carries no per-material reject. The things
  BDPT still can't render — **fluorescence**, **layered stacks**, **spot/env/collimated lights** —
  are *not* GPU gaps: `main.cpp`'s mode-D guard (`bdptUnsupportedFeature`) refuses those scenes (or
  demotes D→B with `-on-unsupported fallback`) on both backends before any BDPT dispatch, so they
  never reach the device path; only GRIN media (curved paths) keep an in-scope mode-D scene
  on the CPU (spectral rainbow-phase media now render on-device in mode D since M10/0.37.0). Validated GPU==CPU on `textured.ftsl` (mean 0.06%,
  per-pixel diff halving 8.2%→4.3% at 4× spp — unbiased), `mixmat.ftsl` (mean 0.21%),
  `scraps/dtrans.ftsl` (mean B/A=1.0009 at 512 spp, per-pixel diff halving 8.42%→4.39% at 4× spp),
  and `scraps/frosted.ftsl` (mean B/A=0.9991 at 512 spp, per-pixel diff halving 10.86%→5.73%). Since
  0.26.0 it also does
  **point-spot lights** (deterministic connect + `spotFalloff` cone weight in
  `bkNeeLight`/`bkNeeVolume`, `spotOmega` geomWeight in `dInvPdfLambda`); only
  collimated beams (not NEE-samplable) still force the CPU backward tracer.
  Since 0.27.0 there is a separate **fast RGB backward** path
  (`renderBackwardRGBCuda`/`kBackwardRGB`/`bkRadianceRGB`, selected by `-rgb` on mode R
  via `cudaBackwardRGBSupported`): an Option-B non-spectral tracer carrying a `DVec3`
  linear-RGB throughput so one intersection walk yields a full-colour sample. Materials
  bake to a linear-RGB albedo under equal-energy white and emitters/env to
  `xyzToLinearSrgb(∫CIE·spd)` (host `namespace rgbbake`, at scene build); deposit maps
  linear-RGB→XYZ via `dRgbToXyz`. Achromatic specular uses `LREP_RGB=550nm` through the
  existing `dDielectricStep`; colored glass keeps a 3-tap Beer-Lambert `rgbAbsorb`.
  RGB Russian roulette survives with `q=rgbLuma(albedo)`. Matches the spectral backward's
  absolute luminance to noise on flat-spectrum scenes; drops dispersion/thin-film/
  fluorescence (Option-B). Gate excludes media, image-env, textured/record albedo, and
  collimated/env-shape emitters (fall back to the spectral backward).
  Since 0.28.0 there are **scene-ignore speed flags** (rasterizer-style): host-side
  `Scene::applyIgnoreFlags(noMedia,noEnv,noFluoro)` (in `scene.h`, run once at load)
  strips media (`-no-media`), the environment emitter (`-no-env`; erases
  `emitters[envIndex]`, resets `envIndex`/`envMap`/`envXYZ`, `finalizeEmitters()`), or
  demotes fluorescent → diffuse (`-no-fluoro`); each prints an `[ignore] stripped:`
  summary. Depth/Whitted params are threaded via globals `g_maxBounceOverride`
  (`-max-bounce N`) and `g_directOnly` (`-direct-only`) — mirroring the `g_heroC`
  pattern — into `BackwardRenderer::maxBounce`/`directOnly`, forward `Renderer::maxBounce`,
  and (on GPU) `DScene.bkMaxBounce`/`bkDirectOnly` set on the `renderBackward*Cuda`
  wrappers. `directOnly` (terminate after the first non-specular NEE, specular chains
  still recurse) is scoped to the camera path tracers (R spectral + RGB, P's backward
  layer); forward B and the photon/BDPT modes honour only `maxBounce`.
  Mode `W` rides the same pattern with three more globals — `g_whitted` (set by the
  `-mode W` → `'R'` normalization, which must happen in **both** `cliModePrescan` and
  the main parse loop), `g_whittedGrid` (`-whitted-grid`) and `g_ambient`
  (`-ambient`, multiplied by `Scene::ambientRef()` at the call site so the CLI value
  is scene-scale-independent), plus four for the one-bounce gather — `g_gi` (`-gi` /
  `-radiosity`), `g_giGrid` (`-gi-grid`), `g_giBounce` (`-gi-bounce`) and `g_giClamp`
  (`-gi-clamp`) → `giDirs` / `giGrid` / `giBounce` / `giClamp`. `g_gi` is zeroed with a
  message outside mode `W`; `g_giClamp` only ever reads inside the gather, so it gets an
  `[ignore]` notice when set without `-gi` rather than silently doing nothing.
  `g_giClamp` shares `g_ambient`'s scaling by `Scene::ambientRef()` — deliberately, since
  the two knobs interact (see `BackwardRenderer::giClamp`) and a user reasoning about
  "one light's own radiance" should not have to switch units between them.
  `g_whitted` also forces `g_directOnly` and excludes
  the GPU backward megakernel (`gpuBackwardMode = mode == 'R' && !g_whitted`), since
  the device path keeps the stochastic estimators.
  Since 0.102.0 the **bidirectional** modes honour `maxBounce` too (they previously
  hard-coded 8 and silently dropped the flag). Their default stays 8 — the BDPT connection
  double-loop is O(depth²) and the per-thread vertex stack is thread-local memory — so for
  D/U the flag *raises* the bound as often as it caps it. On the GPU that bound is a
  template parameter, not a `#define`: `kBdptT<int NS, int MAXD>` sizes
  `DVertex eye[MAXV]/light[MAXV]` from `MAXD`, `maxV` is threaded through
  `dRandomWalk`/`dGenCameraSubpath`/`dGenLightSubpath`, and exactly two variants are
  instantiated — `BDPT_MAXDEPTH` (8, bit-for-bit the old kernel, still the default launch)
  and `BDPT_DEEPDEPTH` (64), the deep one launched only when `-max-bounce > 8`. GPU mode U
  additionally bounds its light-vertex slab (`npix * vcmCap * sizeof(DVcmLV)`) by a 768 MB
  budget and reports when `vcmCap < maxDepth`; that only drops merge candidates at the
  deepest vertices, leaving the estimator unbiased. Why it matters: a specular cavity has a
  photon mean free path of `1/(1-R)` bounces — ~33 for silver — so truncating at 8 does not
  dim such a scene, it deletes it (`scenes/mirror_sphere_interior.ftsl` goes from 97% pure
  black at depth 8 to 29% at depth 64).
  Since 0.107.0 the **`T`** key in the `-explore` fly-viewer **cycles** the still view
  `raster → mode W → path-traced → raster`, with the path-traced stage *skipped* (not
  refused) when unavailable, so on a CPU-only box `T` is a plain raster↔W toggle rather than
  a key that prints an error. `-explore -mode W` opens straight into the W stage.
  The **mode-W stage** (`pvMode == PV_WHITTED`) is the always-available one: CPU, any scene,
  full spectral walk, and noise-free, so the pose renders ONCE rather than converging. Two
  design points, both forced by the cost spread (a mode-W frame is ~0.4 s on a Cornell box
  but ~26 s on a gyroid labyrinth at 960×600, so neither a blocking render nor a fixed
  chunk size works):
   * **Progressive row bands.** `renderBackward` grew `into`/`rowBegin`/`rowEnd` so the
     thread pool can split a BAND of a caller-owned film. The viewer renders one band per
     loop iteration and retunes `wBandRows` from the previous band's measured wall time
     toward `kWBandSec` (0.10 s), clamped to `[1, VH/4]`. Input is drained between bands, so
     moving the camera just abandons the unfinished rows. Bands come off the HIGH end of the
     film because film row 0 is the image bottom (`filmToRgb8` flips), so the picture fills
     downward.
   * **A coarse full-frame pass first** (`kWCoarse` = 1/16 linear, 1/256 the pixels). It is
     nearest-neighbour upscaled into `wImg` so there is a lit image immediately, but the real
     reason it is full-frame rather than just the first band is that its p99 gives a
     globally representative **auto-exposure anchor**, locked into `traceAnchor` for the
     whole pose. Anchoring on band 0 instead would expose the frame off one strip of it and
     blow out everything after.
  Each band is tone-mapped alone, with `N = wPass + 1` (the pass count *those rows* have
  received, not the frame's) and the locked anchor, then spliced into `wImg`. `wFilm`
  accumulates, so it is cleared per pose. When a pass completes the viewer stops — a
  bundle-only scene is exact at 1 spp — *unless* `wNeedSpp`, which is set when the scalar path
  is in use at all (`heroC <= 1`, media, GRIN, lens); then it keeps adding passes to
  `kWSppCap` (16) to resolve the wavelength collapse that case still causes. The *dispersive*
  materials (Dielectric/ThinFilm/Multilayer/Grating/HalfMirror/Fluorescent) used to be on that
  list and no longer are, because `heroSplit` resolves them geometrically at 1 spp; `Layered`
  came off it in v0.115.1, once its coat reflectance became a per-λ weight instead of an
  unconditional de-hero. Because the viewer's preview IS mode W, an `-explore` run also
  honours mode-W-only settings that would otherwise be rejected: `wPreview` in `main.cpp`
  widens the hero bundle and spares `-gi` from the "needs `-mode W`" rejection.
  The older **path-traced preview** stage uses the fast RGB backward tracer:
  a resident `BackwardRGBSession` (render_cuda.cu) bakes/uploads the scene ONCE
  (`buildUploadScene`) and keeps a persistent SUM film; while the camera holds still the
  main loop calls `backwardRGBSessionAccumulate(batch)` each idle tick (advancing the RNG
  `sampleBase`, fixed large `kSppCap` so streams stay unique) and presents the converging
  image via the shared `filmToRgb8` (auto-exposure anchor locked per pose); a camera move
  shows the responsive raster and marks the session for a `setCamera()` re-aim (which
  re-bakes only the cheap POD `bakeCamera` and zeroes the film). Availability mirrors the
  batch `-rgb` scope (`cudaBackwardRGBSupported`); the scene-ignore flags carry in for free
  (scene already stripped host-side; maxBounce/directOnly passed to `...SessionBegin`).
  (`traceHeroPhoton`/`shadeStepHero`), scene upload into `__constant__`/device
  buffers. FP32 by default (`FTRACE_GPU_FP32=ON`). Implicit sphere-tracing
  (`intersectImplicit`) marches + root-refines in FP32 on pre-converted mirror
  pools (`DFieldNodeF`/`PatNodeF`, float VM twins `dFieldEvalF`/`dPatternEvalF`)
  since 0.19.14 — the committed hit is float anyway and FP64 VM ops serialize on
  consumer GPUs' 1/64-rate FP64 pipe (~2× on implicit-heavy scenes); normals
  (`dFieldGradient`) and media bound-fields stay FP64 on the original pools.
  Photon-beams (`-beams`, since 0.19.17) run on the GPU too: `DCamSet::beamGather`
  drives a `shadeStep` branch that (with an independent per-photon `DRng crng` seeded
  in `kTrace`) crosses the medium straight and has each shared camera resample its own
  single-scatter in-scatter point — decorrelated per-frame flyby noise, megakernel-only.
  The shared multi-camera pass runs through a **resident GPU session** (0.20.2):
  `sharedForwardGpuBegin/Batch/Hits0/Download/End` (render_cuda.h) bake+upload the
  scene, bake all cameras, and allocate device films/hits/energy **once**; each
  progressive batch is then a bare `launchForward` accumulating in place, and host
  films are only downloaded lazily (`syncAcc` in `runSharedGroup`) at `-interval` /
  status / final boundaries — download REPLACES host `acc[]`/`accE` with running
  totals rather than merging. `-resume` seeds the device accumulators from the
  loaded checkpoint at Begin. Between intervals a `-noise` budget polls only cam-0's
  hits plane (`sharedForwardGpuHits0`). The old wrapper paid full
  upload/alloc/download/convert/merge/free per ~2M-photon batch, which throttled the
  loop (measured: +80% photons/s on 16 cams @ 640×360, +22% on 2 cams @ 320×240);
  `renderForwardSharedCuda` survives as a one-shot wrapper over the session.
  `raster_cuda.cu` = GPU raster (own section below).
- **`livewindow.*`** — Win32 live preview (`-window`/`-keepwindow`), interactive
  fly viewer input, camera-path timeline panel.
  The **image area is presented by D3D11**, the control strip below it by GDI.
  `LivePresenter` owns a D3D11 device and a flip-model `IDXGISwapChain1` on a **child
  HWND covering the image area** — a separate HWND because a flip-model swap chain and
  overlapping GDI child controls cannot share one window; the parent therefore carries
  `WS_CLIPCHILDREN`. The child answers `WM_NCHITTEST` with `HTTRANSPARENT`, so every
  mouse message still lands on the parent's fly-camera handlers, at unchanged
  coordinates (the child sits at the parent's client origin). `update()` uploads the
  renderer's RGB8 bytes **as-is** — there is no RGB8 DXGI format, so they go up as an
  `R8_UNORM` texture three times as wide and a pixel shader deswizzles them into an
  RGBA8 image texture — and a second full-screen pass scales that into a letterboxed
  viewport with a linear sampler. This replaced a per-pixel RGB→BGRA repack plus a
  `SetStretchBltMode(HALFTONE)` + `StretchDIBits` blit that together cost **9.0 ms per
  frame** against a **1.3 ms** present now (`-raster-bench` reports the tail), and that
  sat on the render thread's critical path because `paint()` held the same mutex
  `update()` needed. The GDI path is retained in full: `FTRACE_LIVE_GDI=1` forces it,
  any D3D failure falls back to it permanently, and **`WM_PRINTCLIENT` always uses it**
  because `PrintWindow` cannot see swap-chain content — that capture pulls the current
  frame back off the GPU (`readbackBgra`) since the image no longer exists in host
  memory. `LivePresenter` serialises its own device context with a mutex, because both
  the render thread (upload + present) and the UI thread (re-present after a resize or
  expose) drive it.
  **`renderShared(w, h, fn)` is the zero-copy entry point** (0.98.0): instead of handing
  the presenter finished host bytes, the caller is handed the presenter's own D3D11 device
  and RGBA8 image texture (both as `void*`, so the header stays API-agnostic) and fills the
  texture on the GPU; `renderShared` then presents it. It is a **callback rather than an
  exposed lock/unlock pair** because the whole map/render/unmap must run under
  `LivePresenter::mtx` — the CUDA↔D3D interop the callback performs drives D3D's immediate
  context, which is not thread-safe and is shared with the UI thread's repaints — and a
  callback makes that impossible to get wrong. It returns false **without calling `fn`**
  whenever the fast path is unavailable (GDI fallback, stub build, lost device), and false
  after a failed `fn`, so the caller always has a well-defined fallback to
  `render → update()`. `ensureImg` may hand back a *different* texture after a resolution
  change, which is exactly why the CUDA side re-registers on pointer identity.
  With `-anim … -loom` the window grows a fourth
  panel row, the **loom bind row** (channel combo → slot combo → Bind/Unbind, a `chans:`
  count box, a status readout). Child HWNDs may only be created and moved on the window's
  own message-pump thread, so the row is built by marshalling `WM_MKBINDROW` (`WM_APP+3`)
  with `SendMessageW`; `hasBindRow` publishes its validity to the render thread, and — the
  subtle part — it must be stored *before* the first `layoutPanel`, since `layoutPanel`
  skips the row unless it is set. Programmatic control changes (`CB_SETCURSEL`,
  `SetWindowTextW`) raise **no** notification, so any value the code sets is also cached
  by hand; only genuine user action produces the `CBN_SELCHANGE` / `EN_CHANGE` the
  `WM_COMMAND` handler turns into `NavInput` edges.
- **`curvedrive.h`** — header-only reader/writer for loom's `CurveDrive` JSON sidecar
  (`-anim <file.json>`), on `src/third_party/json.h`. It re-checks **the same invariants
  `CurveDrive.__init__` does** (`dims >= 1`, ≥ 2 points, every point exactly `dims` wide,
  every binding channel in range, valid `mode`/`kind`) on both load *and* save, so ftrace
  can neither accept nor write a sidecar loom would reject. Saves atomically (temp file +
  `std::filesystem::rename`) with shortest-round-tripping numbers, so editing one point
  leaves every other coordinate byte-identical.
- **`loomlink.h` / `animlive.h`** — the two live loom channels. `loomlink.h` is the shared
  child-process transport (spawn `python -X utf8 -u -m loom.<module>`, newline-delimited
  JSON over stdio, `PYTHONPATH` → `<exeDir>\tools\loom`, plus JSON escaping helpers);
  `animlive.h` is the `-anim … -loom <scene.py>` session on top of it. **ftrace never
  samples the drive** — it pushes the control *points* and asks by parameter `t`, so the
  editor's preview and loom's final render are the same computation rather than two that
  can drift. The bridge is deliberately **two queues**: `frame` messages are latest-wins on
  a single slot (fast scrubbing must collapse, not backlog), while `points` / `bindings` /
  `dims` ride a FIFO that never drops and is drained before every frame — a lost control
  message would leave loom rendering against a curve the editor no longer has. Each ack
  names an emitted `.ftsl` that replaces the scene **wholesale**, because everything
  downstream (`plight`, `prims`, the GPU's baked triangles, the resident RGB-backward
  session) is derived state and a partial swap would leave halves disagreeing. Name-keyed
  incremental re-tessellation is a possible later optimization, to be *measured* first.
- **`viewer_gui.*`** — the **native loom viewer** (`-viewer <sidecar.json>`), a Dear
  ImGui / Direct3D 11 window that short-circuits the renderer in `main`. It reads loom's
  scene-introspection sidecar (`loom.viewer.ViewerModel.save_sidecar`) into flat geometry
  structs — `CurveGeom` / `FieldGeom` / `MeshGeom` / `DagGraph` — and draws N-D curve,
  field, mesh and modulator-DAG panes. Two of its panes are *not* sidecar replays: the
  **Render** pane sphere-traces the real isosurface field by parsing the sidecar's
  companion `.ftsl` with ftrace's own `ftsl::load` and calling `renderIsoPreviewCuda`, and
  the **Meshes** pane bakes procedural skins through ftrace's own pattern VM — so the
  preview and the renderer share one implementation rather than two that can drift.
  The Meshes pane is **z-buffered on the GPU** (`MeshGpu`): the sidecar's tessellation is
  uploaded once into one interleaved vertex buffer + index buffer (per-mesh
  `firstIndex/indexCount/baseVertex` ranges, so each mesh is still its own draw call with
  its own skin and tint) and drawn into an offscreen render target that carries a
  `D32_FLOAT` depth-stencil view, shown with `ImGui::Image`. This replaced a CPU
  painter's-algorithm centroid sort that could not resolve interpenetrating surfaces —
  which loom produces routinely (a swept tube threading an isosurface). Because the
  buffers are keyed on `MeshView::geomGen` (bumped only where `adoptSidecar` installs a new
  tessellation), an orbit / zoom / colour-mode change costs one 144-byte constant-buffer
  write, not a re-projection of every vertex. The union bounds used to frame the view are
  baked with the upload for the same reason. Shading stays the flat two-sided lambert
  `0.30 + 0.70*|n.z|` of the CPU path, with the face normal taken per-pixel from
  `cross(ddx(vp), ddy(vp))` — exact under this orthographic projection — and the wireframe
  is a real second depth-tested `D3D11_FILL_WIREFRAME` pass. The **curve and field panes
  still project on the CPU**; they draw lines rather than solid surfaces, so the occlusion
  bug does not bite them, but they are the same port waiting to happen.
  **Live re-derivation (§F4 item 2, `-loom <scene.py>`)** uses `LoomLink` (a child
  `python -m loom.viewer` speaking newline-delimited JSON over stdio; `PYTHONPATH` is set
  to `<exeDir>\tools\loom`, which is why only the repo-root `ftrace.exe` can find loom) and
  `LoomBridge`, a **one worker thread + one-slot pending job** queue. Both were lifted out
  of `viewer_gui.cpp` into `src/loomlink.h` when the fly editor grew its own loom session
  (§E2 slice 3b) — two live loom channels, one transport, so a protocol or lifetime fix
  lands in both. `post()` overwrites an
  unstarted job, so a drag that moves a parameter every frame costs one bake of the final
  value — latest-wins, and the UI never blocks. Each bake writes a fresh sidecar + `.ftsl`
  into a per-process `%TEMP%\ftrace_viewer_<pid>` scratch dir; the bridge tracks the
  outstanding files and deletes each as it is consumed (a superseded result's files are
  dropped unread), then sweeps and removes the whole directory in `stop()`. Startup also
  reclaims `ftrace_viewer_<pid>` dirs whose pid is no longer alive (`OpenProcess` failing
  with `ERROR_INVALID_PARAMETER`), since a crashed or killed viewer can't clean up after
  itself and only the next run ever can. Results are
  adopted on whatever frame they land, preserving the user's orbit, zoom, active tab and
  DAG layout. **Third-party note:** `src/third_party/imnodes/imnodes.cpp` carries
  `[ftrace patch]` edits for imgui #7543 — see `known-issues.md`; re-vendoring imnodes must
  re-apply them or the DAG pane crashes in `PrimReserve`.
- **`record.h` / `record_ladder.h`** — **parametric records**: a named bank of per-channel
  look-up tables over a shared scalar domain `[lo,hi]`, sampled by one per-hit driver
  scalar so a single expression sweeps a whole material at once. `record.h` holds the
  structural model (`RecChannel` → `RecStop`s with a redistributed domain position) plus
  stop compilation and sampling (`recSampleScalar` / `recSampleSpectrum`, nearest /
  linear / smooth Fritsch–Carlson). Channels are named **by destination**, so a channel
  whose name matches a material slot auto-binds and one that doesn't is still reachable
  by dot.
  `record_ladder.h` is the **delimiter precedence ladder** a channel line's stops are
  written in: whitespace binds tightest (like `×`), comma looser (like `+`), `[ ]` are
  the parentheses — so `1 1 1, 2 2 2` reads as `(1·1·1)+(2·2·2)`, two groups of three.
  Structure is recoverable from the delimiters alone; the channel's arity only
  *validates*, which is why `[1 1 1] [2 2 2]` and `1 1 1, 2 2 2` denote the same tree.
  Parens `( )` are deliberately **not** a rung — they belong to expressions and the
  named-input application surface — so a parenthesised run is an opaque atom and
  `clamp(x,0,1)` stays one leaf.
  The ladder lives in the **loader, not the grammar**, and that is the point: `,` is not
  one of the lexer's delimiters, so a comma survives lexing glued to its word (`0,`) and
  is simply re-split here, paren-aware. Only `[` / `]` genuinely delimit, so the grammar
  carries exactly one extra rule (`stop_group`) whose markers `ftsl_reduce.hpp` flattens
  back into `[`/`]` words. `Parser::parseChannelStops` (`ftsl.h`) drives it and preserves
  additivity **structurally**: a line with no colour tag and no `,`/`[`/`]` takes the
  byte-identical pre-ladder whitespace loop, so no existing record can reparse
  differently.
  The same function strips an optional **inline colour head** (`rgb`/`hsv`/`hsl` and
  every upsampler/emission variant — `isColourHead` is the one list) into
  `RecChannel::space`, then hands `{space, comps…}` to the *same* `evalSpectrum` a
  top-level `spectrum "x" = rgb …` declaration uses. Converging on that one evaluator is
  what makes inline colour nearly free: the record path inherits all 18 built-in colour
  heads *and* the open-ended `rgb:<upsampler>` one, and the Jakob–Hanika coefficient bake
  and the GPU upload never learn that records exist.
  `tools/loom/loom/ladder.py` + `record.py` are the declared Python twins; a stop-boundary
  disagreement between them would be a *silent wrong render* rather than a parse error, so
  `tools/check_record_twins.py` diffs ftrace's per-channel stop count (probed via an
  out-of-range `rec.ch[999]` selector) against loom's across the `scenes/_record_*.ftsl`
  fixtures.
- **Named-input binding (`pattern.h` + `Parser::applyMaterial` in `ftsl.h`)** — a material
  property is an expression over **named inputs**, so a material is a *bundle* of
  slot→expression bindings and is itself a **function** whose free-input set is the union
  of its properties' (`materialFreeInputs`, the twin of loom's `Material.free_inputs`).
  Applying it at a use site — `material gold(u=v,a=1)` — binds those inputs across the
  whole bundle at once.
  The mechanism is **binding by substitution**, and postfix is what makes it nearly free:
  a variable node pushes exactly one value and so does a well-formed program, so
  `patternSubstitute` is a pure **splice** that cannot disturb the surrounding stack
  discipline. The consequence is the load-bearing one — a bound material is an *ordinary*
  material, with no environment, no closure and no runtime indirection, so the CPU
  evaluator, the verbatim GPU upload, `patternHasFreeVars` and every record sampler stay
  untouched. Substitution is **simultaneous**, so `gold(u=v,v=u)` swaps rather than
  collapsing.
  `PatOp::VarA` (`a`, albedo) is the one named input with **no per-hit intrinsic**, so it
  must be resolved at LOAD time — to whatever a use site binds, else to the material's
  `albedoDefault_` (loader-side, *not* on `Material`, which is uploaded to the device).
  Because an unresolved `a` would silently evaluate to 0, **every** by-name material
  reference routes through `lookupMaterial`, which memoises the empty application; that
  is also why `resolveMixChildren` takes a material **index** rather than a `Material&`
  (a lookup can append to `Scene::mats` and reallocate). `applyCache_` keys on
  `"<matIdx>(<args>)"`, so one application shared by N objects builds ONE material, and a
  no-op application returns the original index — the additive-superset guarantee.
  `a` is scope-gated by `compilePatternExpr`'s `allowA` (appended *last* so only the four
  material-reachable sites opt in), and `VarA` is appended at the END of `PatOp` so
  `patternHasFreeVars`' `VarX..VarV` intrinsic range is unperturbed. `parseBindArgs`
  reuses the record ladder (comma == space) and finds argument boundaries from the `=`
  signs rather than the whitespace, which is sound only because the pattern language has
  **no comparison operators** — a top-level `=` can only mean a binding. Named arguments
  bind before a positional one, so a positional takes the sole *still-free* input (loom's
  `free_inputs() - set(binds)`). `-checkbind` pins the algebra (splice == textual
  inlining, simultaneity, identity, introspection).
  **Per-property access** (§3.2, `materialPropRef`) reads ONE slot off an already-declared
  material — `src.reflect`, `src.reflect(u=v)` — with the *slot keyword* as the dot-handle,
  because FTSL properties are identified solely by slot keyword and never carry a quoted
  name (§3.2's "naming is optional" arm is therefore already the ftrace status quo,
  vacuously; what was missing was the handle to read one back out). It resolves by calling
  `applyMaterial(idx, args, L)` and then reading the slot off the *result*, so a reference
  cannot diverge from applying the bundle and reading the slot — same binding rules, same
  memo, same `a` -> **source** `albedo_default` fallback. Four value-site chokepoints carry
  it, one per shape the value can take: `patternedSpectrumParam` (the only site that can
  hold BOTH the base spectrum and the slot's per-hit pattern, so a source pattern and the
  reader's own `<slot>_map` are **composed** there via `composePatterns`, appending
  `[a…, b…, Mul]` — valid postfix for the same "each program pushes one value" reason the
  splice is), `evalSpectrum` (pattern-less spectral sites), `bindScalarPattern` (reports
  "handled" only when the source actually carries a pattern, so the call sites' existing
  `bindScalarPattern -> bindScalarTexture -> dblParam` ladder routes both shapes without
  knowing the form exists), and `dblParam`. Record-driven, texture-bound, and un-appliable
  pattern slots are **refused**, never approximated, since each would hand the reader a
  number the source does not use. `recordIndex_` is checked first so `R.chan` cannot change
  meaning when a material is named `R`. Reaching those sites needs a `Loaded&` they were
  never given, hence the `loadedRef_` member — a pointer to the owner, never to an element,
  because `applyMaterial` reallocates `Scene::mats`/`patterns`. loom twin:
  `Material.prop(name, *args, **binds)` (`tools/loom/loom/scene.py`). `-checkprop` pins it
  by loading in-memory scenes and comparing each reference against a hand-written twin.
- **`render_progress.h`** — progress hook for chunked samples-per-pixel renderers (modes
  `R`, `D`): the live status line (`[live] … photons, ~N% noise`) and noise estimation for
  `-noise` budgets.

## Watertight raster coverage (shared by both backends, 0.98.2)

Both rasterizers decide pixel coverage with **canonical edge functions**, so a pixel lying
exactly on a shared triangle edge is claimed by exactly one of the two sharers — no hairline
crack, no double-cover. `makeEdge` (raster.h) / `makeEdgeD` (raster_cuda.cu) build the rule:

- **Canonical endpoint order.** Each edge's two endpoints are sorted lexicographically by
  `(sx, sy)`. Both sharers therefore construct the same `P` and the same `Q - P` bit for bit,
  whichever way round their own vertex list runs.
- **Sign folded into the deltas.** `sf = sign(area) * flip`, and the stored deltas are
  `sf * (Q - P)`. The two sharers always receive opposite `sf` — consistent winding flips
  `flip`, inconsistent winding flips `sign(area)` instead — so their edge values are exact
  negatives. This is what lets the engine keep accepting **either** winding, which it must,
  because a mesh's triangle order may disagree with its vertex normals. (A classic
  fixed-point + top-left fill rule would have required enforcing one winding.)
- **Tie rule.** An exact zero is accepted only when `sf > 0` (cached as `tie`), true for
  exactly one sharer. Non-zero values are unambiguous by construction, so no epsilon and no
  fixed-point grid are needed.
- **Anchored at `P`.** `v = dx*(py - Py) - dy*(px - Px)`, with the row term hoisted out of
  the x-loop. The expanded affine form's constant (`Px*Qy - Py*Qx`) is ~W·H even for a short
  edge; in `float` its ulp alone displaces the edge line ~1e-3 px, which does not break
  shared *edges* but perturbs the three edges meeting at a shared *vertex* independently and
  leaves an unclaimed sliver there.
- **No incremental stepping, no derived third weight.** All three weights are evaluated
  directly. Stepping would seed each sharer from its own `xlo`, destroying bitwise identity;
  `w2 = 1 - w0 - w1` rounds asymmetrically for the same reason.
- **CUDA defeats FMA contraction** with `edgeRow`/`edgeAt` (`__fmul_rn`/`__fsub_rn`). nvcc's
  default `-fmad=true` would fuse `r - dy*ax`, leaving `r`'s rounding residual instead of a
  clean zero — and it fuses *inconsistently*, since the two sharers test the edge under
  different indices (`e0` vs `e1`). MSVC's default `/fp:precise` does not contract (the build
  sets no `/fp:` flag), so raster.h needs no intrinsics.

Five call sites share the rule: `fillTriangleG` and `fillTriangleClear` (raster.h), and
`rasterRow`, `kShade`'s barycentric resolve, and `kClear` (raster_cuda.cu). It matters most
in the clear pass, which *multiplies* into `clearT`/`milkT` — a doubly-covered edge would
darken a seam twice, an uncovered one leaves a hairline of un-tinted glass. Costs ~4% on the
CPU rasterizer; GPU unchanged. History and measurements in `known-issues.md`.

## GPU raster pipeline (`raster_cuda.cu`)

Powers `-raster -device gpu` and the interactive explorer's per-frame redraws;
steady-state cost (independent of launch/scene build/upload) is measured with
`-raster-bench N`. Four device passes per frame:

- **A — project** (`kProject`): one thread per slot, register-resident 8-case
  near-plane clip. Geometry is split hot/cold: `DGeo` (36 B — screen verts, invd,
  flags' companion) is written for every surviving slot, `DAttr` (120 B — the
  attribute payload) only for clipped slots that needed new vertices. A dense int
  `flags[]` (valid / clear / clipped) drives later passes.
- **B — classify + raster** (`kClassify`, `kRasterSmall/Med/Large`): slots are
  binned by clamped bbox pixel count (≤128 small, ≤16 384 med, else large) into
  device lists; raster kernels merge into a 64-bit packed `(invd_bits<<32)|slot`
  visibility buffer via `atomicMax` (order-independent ⇒ bit-identical under any
  thread mapping). Kernels read bin counts **from device memory** (`dbinCnt`,
  5 ints: 3 counts + med/large ticket counters), so the host never reads counts
  back and the whole frame enqueues without a mid-frame WDDM flush. Work mapping
  per bin: **small** = 1:1 thread↔item under an upper-bound grid (the hardware
  block scheduler load-balances millions of variable-cost items better than any
  grid-stride loop); **med** = warp-level ticket queue (lane 0 `atomicAdd` +
  `__shfl_sync` broadcast, 32 lanes stride rows); **large** = block-level ticket
  queue (shared-mem ticket, block strides rows). Heavy variable-cost bins need
  dynamic balancing — static grid-stride created straggler warps (+0.5 ms on
  gallery); never ticket the small bin (millions of atomics would serialize).
- **C — shade** (`kShade`): one thread per pixel resolves `vis` → shaded float
  RGB. Optional see-through mode then runs a fill+clear pass (`kFillF`+`kClear`);
  its `atomicMulF` has a benign 1-px race (see known-issues).
- **D — expose + encode**: device luminance histogram rounds give an *exact* p99
  white point (readbacks only on the first frame; later frames reuse the cached
  exposure unless the histogram shifts), then `kToneMap` encodes RGB8 on device;
  one pinned-memory D2H of the final image.

The frame is **sync-free**: no `cudaDeviceSynchronize` anywhere; only real data
dependencies block (first-frame histogram readbacks, final image download). Errors
surface through the blocking copies' return codes plus one sticky
`cudaGetLastError()` sweep per frame. Per-pass profiling (`-raster-bench`'s
breakdown) records CUDA events into the stream between passes and resolves them
once after the download — zero overhead when disabled.

**Zero-copy present (CUDA ↔ D3D11 interop, 0.98.0).** In the interactive explorer the
finished frame no longer crosses the bus at all: `bindPresentTarget` registers the live
window's own image texture with `cudaGraphicsD3D11RegisterResource(…SurfaceLoadStore)`,
and `renderFrameToTarget` runs the identical pipeline but ends in `kToneMapSurf`, which
`surf2Dwrite`s each pixel straight into that texture (map → kernel → unmap, all inside
`LiveWindow::renderShared`'s device lock). That removes the D2H image copy, the H2D
re-upload, and every host touch of the pixels in between. Design points:

- **Byte-identity by construction.** Passes A–C are literally the same code (`renderCore`,
  factored out of `renderFrame`), and the tonemap's per-pixel body — including the explicit
  RN double intrinsics that stop nvcc contracting to FMA — lives in one shared
  `__device__ inline tonemapPixel()` that both `kToneMap` (writes RGB8 to a buffer) and
  `kToneMapSurf` (writes RGBA8 to the surface) call. The two paths cannot drift.
- **The download was also the frame's fence and error check**; with it gone,
  `cudaStreamSynchronize(0)` takes over both jobs, and the profiling event that bracketed
  the download now honestly reports ~0.
- **Registration is latched off after one failure** (`gfxOff`). The real-world failure is
  D3D choosing a different adapter than the CUDA device (hybrid iGPU/dGPU laptops) — a
  condition that never heals — so retrying per frame would be pure cost; the sticky CUDA
  error is cleared so later frames aren't poisoned.
- **Re-registration is keyed on texture pointer + size.** Comparing pointers is safe
  because a registered resource holds a COM reference, so a new texture can never be
  allocated on a still-registered one's address.
- **Fallbacks are explicit, not silent.** `main.cpp`'s `rasterPresent` declines the fast
  path (returning false so the caller renders to host memory and calls `update()`) when
  there is no live window or GPU scene, when the implicit-ray iso preview owns the frame,
  or when a camera-path/edit-point overlay needs to draw into host pixels. It prints a
  one-line state change the first time each way, so a silent fallback can't masquerade as
  working interop. Warm-only clock-keeping frames deliberately keep the ordinary path
  (they must render without repainting).

Measured @ 3840² (RTX 4090): host path `render 21.97 + present tail 4.17 = 26.1 ms`
vs zero-copy `9.67 ms min`; the per-pass download line drops `4.06 ms → 0.10 ms`.
(`-raster-bench` medians pin at exactly 16.67 ms / 60.0 fps on the zero-copy phase —
that's flip-model `Present(0,0)` blocking on vblank with `BufferCount 2`, i.e. the
display refresh, not the pipeline; read `min` for the true cost.)

Perf state (2026-07 campaign, opts 1–8, RTX 4090 @1600×900): cornell **1.97 ms**
(508 fps), gallery (5.08 M tris) **~4.45 ms** (~225 fps), glassgal **5.12 ms** —
~22–25× vs the 0.19.0 baseline. Passes sit near memory-bandwidth floors; the
remaining ~1.1 ms is host-side (result-vector copy, WDDM submit, bench loop).
HIP portability note: the alias block deliberately does **not** alias
`__shfl_sync`, so a HIP build fails loudly at kRasterMed instead of silently
mis-broadcasting on wave64 GPUs (see known-issues).

**GPU clock keep-warm (interactive explorer).** The explorer re-renders one frame
per camera move, then idle-sleeps — a bursty, low-duty submission pattern the
NVIDIA driver's DVFS reads as "idle", parking the card in its lowest power state
(measured RTX 4090: **P8 @ 210 MHz** vs **P0/P2 @ 2520–2775 MHz** under load, a
~13× clock drop; ~33× for a cold first frame). That made each fresh mouse-look
burst pay a cold-clock penalty until continuous motion finally ramped the clocks.
Fix (`main.cpp` explorer loop): for `kWarmGraceSec` (2.5 s) after the last real
interaction the loop holds the boost clock with discarded "warm-only" `rasterOne`
frames (never touch the window), then past the grace window falls back to the 15 ms
passive sleep and the card powers down to P8. Two thresholds tune it: a warm frame
fires only once `idleFor` passes `kWarmGapSec` (0.10 s) — i.e. a GENUINE pause — and
then runs *continuously* (no nap) so the clock actually stays up (a sparse rate-
limited trickle was measured too weak — the card sat at P8). During an active mouse-
look or timeline-scrub drag the sub-frame gaps between input events stay under the
gap, so warm frames are suppressed and every loop slot samples the next scrub
position; otherwise a warm frame landing between two events would steal that slot and
the timeline would "chunk" by several cameras per drag (0.22.0 regression, fixed
0.22.1; see known-issues). Between events inside the gap the loop naps 3 ms (prompt
drain, no busy spin). Gated on the discrete-GPU path (`gpuRaster != nullptr`); CPU
raster unaffected.

## Threading model (CPU)

Band/chunk parallelism via `std::thread` pools sized by `hardware_concurrency`;
work units pull atomically from a shared counter in chunks. Determinism comes from
per-unit RNG seeding (above) plus order-independent accumulation per band/tile;
film merges are structured so paired runs differ only by summation-order ulps at
worst (mode R) or are bit-identical (fixed splits).

## Stopping a render (`g_stopRequested`, `-stop`)

One flag drives every clean stop: `g_stopRequested` (`main.cpp`). It is raised by the
first Ctrl-C (a second restores `SIG_DFL` and force-quits), by the live window being
closed, and — since 0.99.0 — by an **external** `-stop`. Every render loop polls it at a
chunk/frame boundary and, on seeing it, writes the final image + `.ftbuf` checkpoint and
returns normally, so the process unwinds through `cudaGracefulShutdown()`.

That last exit path is the point of the whole mechanism: **force-killing ftrace while
CUDA kernels are in flight can wedge the NVIDIA driver into a TDR/bugcheck**, so nothing
in this project may be stopped with `taskkill /F`.

`-stop` supplies the trigger a detached render (no console to Ctrl-C into) previously
lacked. It is a **sentinel file** under `<temp>/ftrace/`, not a named kernel event,
because renders run in the interactive Console session while the shell signalling them
may be in a different session / window station (the same split that makes `-window`
invisible under a sandboxed shell) and `Local\` objects are per-session.

- `<pid>.run` — published by `stopChannelStart()` for the whole process lifetime
  (including the `-keepwindow` hold), holding a `scene -> output` line; removed by
  `stopChannelEnd()`. A stale one left by a hard kill is reaped by the next `-stop`,
  which probes the pid first.
- `<pid>.stop` — written by `ftrace -stop <pid>`; a 250 ms watcher thread in the target
  consumes it once, sets `g_extStopRequested` **and** `g_stopRequested`, then exits.
- `g_extStopRequested` is separate from `g_stopRequested` because the latter is cleared
  per frame by the `-serve` loop; an external stop means "shut the process down", so it
  must survive that clear — and it also breaks the `-keepwindow` hold.

`-stop all` targets every live render, a bare `-stop` lists them, and both wait (≤120 s)
for the targets to actually exit so a rebuild can be scripted immediately after.

## GPU support gates fail safe, never coerce

`cudaForwardSupported()` (`render_cuda.cu`) is the single gatekeeper — all eight GPU
gates chain to it — and the rule it enforces is that anything the device kernels cannot
do sends the scene to the CPU tracer. Host→device enum mappings must therefore be
**closed whitelists shared with the upload**, not open ternary chains with a default
arm: `deviceEmitterShapeCode()` is the model (a `switch` with no `default`, so a new
enumerator trips MSVC C4062, plus a `-1` runtime fail-safe). Silently coercing an
unrecognised value into some other device code hands a kernel malformed geometry, and
malformed geometry inside a kernel does not fail cleanly — it can fault the display
driver. See `gpu-fallbacks.md` for the per-feature fallback tables.

## Benchmarks & perf discipline

- `scraps/bench.py` — 19 standard configs (13 CPU + 6 GPU; cornell + gallery
  scenes); min-of-reps timing, sha1 of PPM outputs, `fuzzy_ppm_diff` for GPU.
- `scraps/bench_ab.py` — interleaved A/B harness (exeA/exeB alternate per config so
  machine drift cancels); used for the 2026-07 optimization campaign report
  (`scraps/bench_final_ab.json`).
- Rule: any hot-path optimization must be **bit-identical** (CPU sha1) or
  visually/fuzzy identical (GPU) vs. the pre-change exe before committing, one
  commit per optimization so any regression can be reverted alone.

## Build & release

- `build.bat` → CMake/VS2022 x64 Release into `build_cuda2/`, copies
  `ftrace.exe` to the repo root. **Warning:** freshly-configured build dirs
  currently produce a GPU-silently-dead exe (see known-issues, 2026-07-22) — build
  in the long-lived `build_cuda2`.
- `VERSION` (single `MAJOR.MINOR.PATCH` line) bumps with every observable rebuild;
  `release.bat` publishes repo-root `ftrace.exe` as GitHub release `v<VERSION>`
  (refuses on duplicate tag).
- Output conventions: renders → `ppm/`/`png/` (flyby series in `png/<set>/`),
  scratch scripts → `scraps/`. Renders always launched with `-keepwindow`
  (+ `-checkpoint`/`-interval`) and outside the Bash sandbox so the live window is
  visible.
