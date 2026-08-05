# Known Issues & Technical Debt

Running log of unsolved bugs and accumulated tech debt. Fix items here as soon
as practical; this file is the fallback for what can't be addressed immediately.

## Open issues

### BUG — DONE (2026-08-05, v0.138.0): `-mode W -heroc 1` silently rendered the de-hero collapse it was supposed to have fixed

**What.** `-heroc 1` means "hero bundle off, single wavelength" (`ftrace -h`), which in mode W
collapses the mode's *entire fixed spectral quadrature* to one λ and forces every material down
the scalar, bundle-free path. On a scene with a spectrally-varying material that is not an
approximation, it is flatly wrong. Measured on `scenes/cornell.ftsl` (SF10 sphere), chroma error
against a converged 8192-spp direct-only mode-R reference, brightness normalised out:

| render | chroma error |
|---|---|
| mode W, 1 spp, default C=8 | **0.82 pp** (correct — reproduces N1's 0.80 pp) |
| mode W, 1 spp, `-heroc 1`  | **46.85 pp** — a flat GREEN ball |

46.85 pp is the exact de-hero pathology N1 (v0.108.0) was written to kill, still reachable through
a CLI flag, and **nothing was printed** to say so.

**Why it mattered.** Mode W's whole promise is "noise-free and *correct* at 1 spp". Everywhere else
the codebase defends that promise: the interactive viewer detects `g_heroC <= 1` and keeps
accumulating passes (`wNeedSpp`, `main.cpp`), so its preview converges out of the collapse. A batch
`-mode W -spp 1` render has nothing to average over, so it just emits the wrong image. And the
trade the user thinks they are making isn't real: N5 measured C=1 vs the C=8 default at **2.7% of
frame time** on a 15-second mode-W frame (mode W is traversal-bound — the bundle rides along on
rays already being traced), so they gave up correctness for a rounding error.

**How it was found.** Sweeping `-heroc` while measuring N5's cost slope; the `-heroc 1` frame was
included as a timing point and its *image* turned out to be the interesting result.

**Fix.** `warnWhittedHeroCollapse` (`main.cpp`), called only for a real batch mode-W render at
`g_heroC <= 1` — not for the explorer's T preview, which converges anyway and would nag every run.
It names the material class responsible, states that those surfaces are *flatly wrong* rather than
merely noisy, and quotes the ~1% cost of not doing this. The scene predicate `whittedNeedsBundle`
walks only materials actually attached to geometry (an unused library material is no reason to
nag), returns immediately for ThinFilm / Grating / Multilayer / Layered / Fluorescent, and — since
`Spectrum` is a `std::function` with no inspectable curve type — decides a Dielectric by **probing
its `ior` at 400–700 nm**, so a constant-IOR dielectric (exact at C=1) stays silent. Print-only:
the mode-W image is byte-identical to 0.137.0.

**Lesson.** A knob that is legitimate in one mode can silently void another mode's headline
guarantee. `-heroc 1` is perfectly reasonable in mode R, where more samples fix it; in mode W there
are no more samples. When a mode's contract is "correct at 1 spp", every flag that can break that
contract needs either a guard or a warning — the viewer had a guard, the batch path had neither.

### BUG — DONE (2026-08-05, v0.137.0): nvcc's FMA contraction made the GPU's mode-W sample lattices disagree with the CPU's by 1 ULP

**What.** `dRadicalInverseScr` — the digit-scrambled radical inverse that places every
deterministic mode-W sample (subpixel, wavelength, glossy lobe, grating order, fluorescence
excitation, GI gather phase) — computed its digit accumulation as `r += digit * f`.
`dGoldenDigitMul` likewise computed `(unsigned)(base * 0.6180339887498949 + 0.5)`. Both are
multiply-adds, and **nvcc contracts a multiply feeding an add into a single FMA by default**
(`-fmad=true`) while MSVC does not. The two sides therefore rounded differently: **34 191 of
2 169 156 probed values (1.6%) differed, all by exactly 1 ULP, all confined to the
`radicalInverseScr` columns.**

**Why it mattered.** FMA is the *more* accurate form — it drops the intermediate rounding —
so this is not an accuracy bug. It is a **contract** bug. `render_cuda.h` permits the
stochastic modes to be "an independent noise realization that agrees to within Monte-Carlo
noise", but **mode W has no noise for a mismatch to hide behind**: every pixel shares the same
sample offsets, which is precisely what makes the mode noise-free, so a lattice difference is
a *visible deterministic* CPU/GPU difference rather than a re-rolled die. `backward.h` and
`render_cuda.cu` both carry "Host twin: … Must stay bit-identical" comments on these
functions; they simply weren't tested, so the claim had quietly become false.

**How it was found.** `ftrace -checklattice` (TODO §N item N4a), on its first run. The
per-column mismatch histogram it prints is what made the diagnosis immediate: a 1-ULP spread
confined to exactly the columns containing a multiply-add is a compiler-contraction
signature, whereas a logic error would show a large or column-wide gap.

**Fix.** Spell both multiply-adds with `__dmul_rn` / `__dadd_rn` — individually-rounded
operations the compiler is not permitted to fuse. This pins the device's evaluation order to
the host's without touching any other kernel's codegen (a global `-fmad=false` would have
changed every kernel in the build). Post-fix all 2 169 156 values are bit-identical, and the
part-(b) image A/Bs are unregressed: `scraps/n3_gpu.ftsl` 99.31 % bit-identical, and
`scraps/cor_gi.ftsl -gi 32` 98.68 %, matching the 98.7 % recorded at the 0.116.0 gate.

**Lesson worth keeping.** Any host/device pair asserted to be bit-identical must have that
asserted *by a test*, not by a comment — and the first thing to suspect in a 1-ULP host/device
gap is FMA contraction, not the algorithm. Other device code carries the same hazard where a
comment claims bit-identity: `dGiDir`'s `kGolden * j + 2*PI*p1` contracts too, but its result
goes through `cos`/`sin`, which libdevice already computes differently from the CRT, so that
one is not bit-comparable in the first place and is covered by part (b), not part (a).

### DONE (0.135.0): preview rasterizer showed six classes of material as flat colour

**What.** `ftrace scenes\gallery_rain.ftsl -explore` previewed the ten marble caps
completely untextured, and rendered 28% of the frame as one flat `rgb(0,255,89)` slab.
Both had the same shape of cause: `raster::tessellate` / the shade pass knew about a
*subset* of the material features that determine a surface's appearance, and silently
dropped the rest.

Six gaps, all now closed on **both** the CPU (`raster.h`) and GPU (`raster_cuda.cu`) paths:

1. **Per-primitive `uv` projections were ignored for marched implicits** — the reported
   bug. `Implicit::uvProj/uvAxis/uvBounds` is what gives an isosurface a UV parameterisation
   (marching cubes emits none), and `tessellate` honoured only the *material's*
   `triplanarScale`, so it set `tex = -1` on every marched implicit. Fixed by projecting each
   marched vertex with the tracer's own `projectUV`.
   *Seam hazard closed with it*: azimuthal (spherical/cylindrical) `u` wraps 1.0→0.0. The
   ray-hit path projects AT the hit and never sees the wrap; the rasterizer *interpolates*,
   so a straddling triangle would run `u` backwards across the whole texture. Triangles whose
   `u` spread exceeds 0.5 now have their low corners lifted by +1.
2. **`emit pattern:` / `emit_map pattern:` ignored** — the flat green slab. `gridground`
   is a 528 nm emitter masked by `emit_map pattern:grid_ground`; with the mask dropped the
   whole plane glowed. The fix required moving the emissive early-out to AFTER pattern
   evaluation.
3. **`reflect pattern:` / `reflect_map pattern:` ignored** — same VM, albedo slot.
4. **Palette (indexed-spectral) maps sampled their raw index map.** `Texture::sampleRgb`
   returned the *index* out of the red channel, so entry 3 of 12 shaded as near-black. Fixed
   inside `Texture` (`buildPaletteRgb` / `paletteRgbAt`) so every colour consumer benefits,
   not just the rasterizer.
5. **Normal maps unused.** Now applied through a UV-derived TBN frame.
6. **`mix` / layered materials collapsed to the parent's `reflect`.** Now resolved via
   `mixDominantChild` — the same child deterministic mode W picks, so the two agree.

**How the class of bug was closed, not just the instances.** A single `applyMat()` now
assigns every per-material field, called from all four geometry paths (world tris, spheres,
implicits, instances); the original bug was a feature wired into three of the four. The CPU
G-buffer also stopped copying material fields per pixel and stores the *source triangle
index* instead (matching what the GPU backend already did), so a future per-material feature
needs no new per-pixel channel to be dropped from.

**Not** fixed, deliberately: roughness / film-thickness maps stay ignored (a preview has no
glossy lobe for them to drive), and there is no "textured emitter" — the language has no
such slot, so previewing one would invent detail the render does not have.

**Verified.** `gallery_rain` on both backends: 6247 of 1.17 M pixels (0.54%) differ by
more than 2/255, all of them on grid-line boundaries where a hard `step` meets the CPU's
double vs the GPU's float world position. Mean absolute difference 0.12/255.

**Also verified: the shared pattern VM did not regress the path tracer.** The VM body that
`raster_cuda.cu` now calls was *lifted out of* `render_cuda.cu` into `pattern_device.cuh`, so
the tracer had to be re-checked. `scenes/_preview_pattern_tex.ftsl` (added for this) binds
three `tex:` expressions (plain,
warped-lookup, and an `emit_map` mask) to slots the preview evaluated at the time —
`weight_map` was no good for it, because the preview then resolved a `mix` to its dominant
child and never ran that pattern. (That limitation is itself now fixed; see the 0.136.0
entry below. The scene stays as-is: `reflect`/`emit_map` bindings are the tighter test of
the VM, since they reach it without going through the mix threshold.) CPU vs GPU, matched
settings:

| path | difference |
|---|---|
| `-raster` (`DPatEnvT<DTex>`) | 133 of 160 k px (0.08%) differ by >2; mean 0.09/255; 1322 vs 1323 unique colours |
| `-mode W` (`DPatEnvT<DTexture>`) | 24 of 25.6 k px (0.09%) differ by >2; mean 0.02/255; identical auto-exposure anchor `6.11e-13` |

Same residual as above — hard checker edges under `filter nearest`, float vs double.

**Note on how slow that mode-W check was:** 74.4 s on the GPU for 4 spp at 160x160 versus
**0.1 s** on the CPU, with the `[gpu-stall]` watchdog reporting 30 s per 1-spp chunk against a
0.15 s target. That is the GPU-contention entry below (pid 20264 again), not a cost of this
change — the raster path, whose kernels are tiny, rendered the same scene in 0.42 s throughout.


### DONE (0.136.0): seventh gap — a `mix` blend mask never previewed (the preview and mode W disagreed)

**What.** A two-child `mix` carrying `weight_map texture:` / `weight_map pattern:` is a
*spatial A/B mask* — wear masks, decals, painted patterns. The preview rasterizer resolved
every mix with `mixDominantChild`, which only compares the **constant** weights. A blend
mask is always declared 50/50, so the "dominant" child was always child 0 and the mask was
never evaluated: the surface previewed as one flat winner. `-mode W` calls
`mixResolveDominant` instead, which samples the mask at the hit and hard-thresholds at ½.
So `raster.h`'s own header comment — "the same child deterministic mode W picks, so the two
agree" — was false for exactly this case.

Visible on `scenes/pattern_tex.ftsl`: mode W shows three checkered walls, the preview showed
three flat ones. 19 shipped scenes use `weight_map`.

**Fix (both backends).** `PTri`'s material payload is split out into a base `PShade`;
`PreviewGeom` now carries a `PMix` side table (`weightPat`, `weightTex`, and the child-1
`PShade`) that `PTri::mix` indexes. The shade pass evaluates the mask at the shaded point
and, below ½, repoints `const PShade* sh` at the loser — so albedo, skin, normal map and
pattern drives all switch together, per pixel. The table is keyed **per material**, so three
weight-mapped mixes cost three entries no matter how many triangles carry them (inlining a
second payload in `PTri` would have cost ~+120 MB on a 2 M-triangle scene). The GPU twin
mirrors it with a `DMix` device array consumed in `kShade`.

`emissive` and `clear` deliberately still come from child 0: `g.emis` is consumed by
auto-exposure *before* shading, and `clear` steers the separate see-through pass, so neither
can vary per pixel. A child that is itself a weight-mapped mix still flattens.

**Bug found and fixed during verification.** The first cut still previewed flat, because
`STri::needUV` — the flag that decides whether the raster pass interpolates UVs at all —
listed only skins / normal maps / pattern slots. A mix mask reads (u,v) too, so the mask was
being sampled at `u=v=0`. With `needUV |= (mix >= 0)`, `pattern_tex.ftsl` reproduces mode W
exactly. (The GPU interpolates UVs unconditionally and never had the bug — a good reminder
that CPU-only optimisations are their own parity hazard.)

**Verified.** `pattern_tex.ftsl` (`weight_map pattern:`) CPU vs GPU raster: 0.076% of pixels
over 2/255, mean 0.04/255. `maskblend.ftsl` (`weight_map texture:`): 0.068%, mean 0.05/255,
and the CPU raster is visually identical to a 16-spp `-mode W` reference. Regression check on
`_preview_pattern_tex.ftsl`: 0.083%, unchanged from before this work. `gallery_rain` still
previews its marble and grid correctly. All residuals are the usual hard-threshold /
hard-checker edges under float-vs-double.


### OPEN (2026-08-04): `textures/marble_dumbbell.png` is derived from a WATERMARKED stock preview — replace before any public release

**What.** `marble texture 3.5.avif`, one of the ten source sheets, is a VectorStock **comp**: a
black footer bar carrying the agency name and stock id runs across the bottom ~9 %. That bar is
why this one file measured `p5 = 0.000` while every other sheet floors around 0.45.
`tools/make_marble_caps.py` crops it off (`crop=(0, 0, 1, 0.905)`), so the shipped texture is
clean *looking* — but it is still a derivative of an unlicensed preview image, and it is now
committed as `textures/marble_dumbbell.png` and referenced by `scenes/gallery_rain.ftsl`.

**Why it was not silently shipped.** The repo publishes GitHub releases (`release.bat` →
`inhahe/goodphotons`), so a stock comp in the asset set is a licensing problem, not a cosmetic
one. Cropping the watermark out arguably makes it worse rather than better.

**Fix.** Replace that one source with a properly-licensed or public-domain marble photograph and
re-run `python tools/make_marble_caps.py`; nothing else changes, since the tool renormalises
whatever it is given to the cap albedo. The other nine sources are of unverified but
unwatermarked provenance and should be spot-checked at the same time. The raw drops themselves
are deliberately left untracked (see `.gitignore`); only the prepared PNGs are committed.

### OPEN (2026-08-04): `scraps/_capchroma.py` scores marble VEINS as a caustic — the metric assumes a uniform cap albedo

**Symptom.** With the tabletops textured, the gyroid cap meters **coverage 4.55 %, sat 0.434,
spread 0.201, fan 0.84** in the converged frame. The untextured control of the same frame meters
**0.00 %**, and the gold gyroid standing on that cap is *opaque* — there is no caustic there to
find. The brass cap shows the same thing more mildly.

**Why.** The metric is "excess over 2× the cap's own median", which attributes every bit of
variation across the cap to *light*. That is only valid when the cap's albedo is constant, which
it was until v0.134.0. A full-contrast marble slab swings 6.7–8.2× p2→p98, so its own veins clear
the 2× bar; and because veining varies *smoothly with position*, it also scores high on `fan`,
the statistic specifically built to separate dispersed colour from position-uncorrelated speckle.
`fan 0.84` here is being scored on rock. The tell is the `noise` control band, which is computed
the same way and jumps with it: 0.028 → 0.297 on the gyroid cap, 0.028 → 0.164 on brass.

**Impact today: none, by construction.** The seven caps that got full-contrast stone are exactly
the ones with no measurable caustic (opaque metal / opaque iridescent / the Klein bottle's
optical window), and the three caps that *are* metered were deliberately given calm stone
(swing 1.29–1.37×, `k` 0.45–0.55, `s` 0.30–0.40). Their noise floors are unchanged by the
texturing — axicon 0.050 → 0.050, diamond 0.060 → 0.062, orb 0.097 → 0.099 — so the caustic
numbers remain valid. But the trap is now armed for anyone who meters a decorative cap, or who
later raises `k` on a cap that matters.

**Proper fix.** Divide the texture back out before thresholding: `_capchroma.py` should sample
the cap's albedo map through the same planar UV projection the renderer uses and meter
`irradiance = pixel / albedo(x)` rather than `pixel`. Failing that, it should at least *detect*
a textured cap (the material carries a `texture:` reference) and refuse to report, rather than
returning a confident wrong number.

### OPEN (2026-08-04): a GPU render can be starved to a standstill by ANOTHER process on the card, and there is no way to make it yield

**Symptom.** With an unrelated CUDA process (a `python` model server, pid 20264) holding ~20 GB
of a 24 GB RTX 4090 and driving it at 100 %, `ftrace` mode D became unusable: a **160x90, 1-spp**
chunk — a few milliseconds of work on an idle card — did not return in **11 minutes**. A
1280x720 render of `gallery_rain` reached 25 spp in 30.8 s and then produced nothing for 17
minutes. `cdb -p <pid> -c "~0k"` showed the main thread parked in `nvcuda64!cuCtxSynchronize` the
whole time, and the process could not be stopped with `ftrace -stop` (see why below).

**Diagnosis.** Contention, not an ftrace bug — and *not* a VRAM-capacity bug either: note that
`cudaMemGetInfo` reported plenty free throughout, because under WDDM the driver overcommits and
pages rather than failing an allocation, which makes the capacity query worthless as a detector.
What ftrace *was* guilty of is being completely silent about it: from outside the process this is
indistinguishable from a deadlock, and `nvidia-smi` reports 100 % busy either way. Confirmed
environmental by rendering the *untextured* `git show HEAD:scenes/gallery_rain.ftsl` with
identical settings — it stalled identically, clearing the scene's new marble textures.

**What was done (v0.134.0).** Not a fix — the renderer cannot preempt another process's kernels
— but the failure is no longer mute:

- `gpuSppChunks` (`render_cuda.cu`) now runs a **stall watchdog** thread. If one chunk stays in
  flight past 30 s (target: 0.15 s) it prints `[gpu-stall]` with the chunk size and elapsed time,
  states that the render is not hung but cannot write `-interval`, honour `-time` or answer
  `-stop` until the chunk returns, and points at `nvidia-smi` / `-device cpu`. Repeats each minute.
- A `[vram]` line reports free/total device memory and the megakernel's per-thread local
  reservation (`cudaMegakernelLocalBytes`, via `cudaFuncGetAttributes`) at the device gate, and
  falls back to the CPU when the card is over budget or >88 % full. **Weak detector** — it did not
  trip in the measured case, for the WDDM reason above. Kept because a trip is conclusive; a
  non-trip means nothing.
- `gpuSppChunks` gained the `FTRACE_CHUNK_SPP` / `FTRACE_CHUNK_DEBUG` levers `cpuSppChunks`
  already had. Their absence is why this needed a debugger to characterise.

**Still open / proper fix.** `-interval`, `-time` and `-stop` are all polled *between* chunks, so
one long chunk disables the whole control surface. The robust fix is to stop assuming a chunk is
short: size the first launch from a **cheap timed probe** (the kernels already take `totalSamples`
explicitly, so a fraction of one spp is launchable) instead of starting at 1 spp and hoping, and
consider a CUDA-stream + `cudaEventQuery` poll loop so the host thread stays responsive while a
chunk is in flight instead of blocking in `cudaDeviceSynchronize`. Until then the workaround is
`-device cpu` whenever the card is busy.

### FIXED (2026-08-04, v0.133.0): `-stop` cannot stop a MULTI-CAMERA render — it ends the current frame and the batch marches on to the next camera

**Symptom.** `scenes/gallery_rain.ftsl` declares 601 cameras. A render launched without
`-camera` finishes the still and then walks the whole 600-frame flyby set. Two `ftrace -stop
<pid>` calls each reported

```
[stop] asked pid 90024 to finish and exit cleanly.
[stop] still running after 120s: 90024
```

and `-stop all` did no better, while the process kept emitting frames at ~2-3/s. It had to be
killed with `taskkill /F`.

**Diagnosis.** `-stop` is a *finish the current work item and exit* request, and the work item
is one FRAME, not one invocation. In a multi-camera batch the stop flag is consumed (or
cleared) at the frame boundary and the loop advances to camera N+1, so the request can never
retire the run. The 120 s timeout then reports "still running", which reads like a hung process
and invites a force-kill — exactly the thing `-stop` exists to avoid, and dangerous mid-CUDA.

**Fix (v0.133.0).** The per-camera loop in `main.cpp` (the `restIdx` loop) now polls
`g_stopRequested` between frames, exactly as the mode-M loop above it already did, and breaks —
announcing `[stop] stopping the batch: N of M cameras not rendered.` so the truncation is visible
rather than silent. Root cause was that the flag was only checked *inside* a frame: once set,
every subsequent `runRender()` returned immediately at ~0 spp, so the batch sprayed near-black
frames at several per second and `-stop` could never retire it. **Verified:** `-stop` now returns
`[stop] done — stopped cleanly` in ~1 s (was: 120 s timeout then force-kill), with the log reading
`[stop] stopping the batch: 596 of 601 cameras not rendered`.

**Second bug, same incident — also FIXED (v0.133.0).** Those flyby frames were written **loose
next to `-o`** as `png/rain_axicon_fly000.{png,pfm,png.ftbuf}` — 148 of them before the kill —
violating the project rule that a multi-frame series lives in its own `png/<setname>/`.
`CamSpec` now carries a `pathBase` (the path's name without the frame number, empty for a
standalone `camera`), set at all three path-generation sites in `ftsl.h` (`camera_path`, `orbit`,
`camera_curve`); `outFor` in `main.cpp` uses it to write any series of >1 frame into
`<stem>_<pathBase>/`. **Verified:** flyby frames land in `png/_stoptest_fly/` while a standalone
camera's still stays beside `-o`. (The 148 stray frames were deleted.)

### OPEN (2026-08-04): mode `W` is NOT noise-free at `-spp 1` when the scene has a medium — the fog term is still a Monte-Carlo free flight

**Symptom.** `ftrace scenes/gallery_rain.ftsl -mode W -spp 1 -r 480 270` returns a black frame
covered in isolated, fully-saturated speckles instead of smooth haze. The same scene with
`-no-media` is clean and perfectly deterministic, and `scenes/cornell.ftsl -mode W -spp 1` is
clean too, so the surface half of mode W is fine — it's the volume half that is noisy.

**Why.** Mode `W` is meant to be mode `R` with every *estimator* replaced by a fixed quadrature
(4x4 shadow rays per light, 8 wavelengths per sample, a grid instead of a random lobe). The
participating-medium branch was never converted. `Backward::radiance` (`src/backward.h` ~1193)
still does the plain analog free flight

```cpp
double tMed = -std::log(1.0 - rng.uniform()) / st;
if (tMed < dSurf) { ...neeVolume(...)...; }        // then analog scatter-or-absorb
```

and `neeVolume` (~685) draws `rng.uniform()` again for the light-point / sun-cone sample. So each
pixel gets **one** random scattering depth and **one** random shadow connection: it either misses
the light entirely (black) or connects and carries the whole `1/pdf` (a blown-out speckle). It is
*deterministic* in the sense the docs promise — two runs are bit-identical (`scraps/imgdiff.py`
gives max |diff| 0.00000) because the RNG is seeded from the pixel — but it is not *noise-free*,
which is the property that makes mode W useful as a 1-spp preview. `-spp 64` converges it, at
which point mode W costs the same as mode R and its reason to exist is gone.

**Proper fix.** Quadrature-ise the volume branch the same way the surface branch already was:
march the camera segment in `N` equal-transmittance (or equal-`t`) strata instead of sampling one
free flight, accumulate `sigma_s * T * phase * NEE` at each stratum, and give `neeVolume` the same
fixed `4x4` light-sample grid `neeLight` uses (`gridUV(s, G, u1, u2)` at ~540/578) rather than
`rng.uniform()`. Both are `whitted`-gated, so mode R stays bit-identical.

### OPEN (2026-08-04): the CPU and GPU backward tracers render DIFFERENT fog — the CPU one still collapses `scene.media` to a single global haze

**Symptom.** `scenes/gallery_rain.ftsl -mode W` renders the ceiling cloud, the mesh-bound
raincloud and a full spectral **rainbow** from the `phase rainbow { droplet_um 500 }` curtain on
the **GPU**, and none of them on the **CPU** — same scene, same mode, same spp, two different
pictures:

```
ftrace scenes/gallery_rain.ftsl -mode W -spp 32 -r 480 270 -o png/gpu.png   # clouds + rainbow
ftrace scenes/gallery_rain.ftsl -mode W -spp 32 -r 480 270 -device cpu -o png/cpu.png   # neither
```

**Why.** The device backward megakernel grew full multi-medium support (`dMediaSampleCollision`
/ `dMediaTransmittance` / `bkNeeVolume` in `src/render_cuda.cu` ~1787, ~6929) — extinction adds
over `sc.media[0..mediaN)`, homogeneous media use an exact free flight and heterogeneous ones
Woodcock tracking, and the phase function is looked up per medium — but the CPU twin never did.
`src/backward.h` still calls `scene.backwardMedium()` (`src/scene.h` ~1021), which is literally
`media.front()`, and treats it as a *global homogeneous* haze with `bounds` and `density` ignored.
Its own comment claims it "mirrors backward.h radiance() so the two estimators agree", which is
no longer true. Mode `V` (which compares a forward render against the **CPU** backward reference
by design) is therefore also measuring the wrong fog on any multi-medium scene.

**Partially addressed (v0.128.1):** the `[medium] …` warning used to fire for every R/W/V/P
render on a multi/bounded/heterogeneous-media scene, including GPU ones where nothing is actually
lost. It now runs *after* the `-device` resolution in `src/main.cpp` and fires only when the
render's backward layer really lands on the CPU tracer, and it names `-device gpu` as the fix.

**Proper fix.** Port the superposition to `backward.h`: replace the single
`scene.backwardMedium()` with the same "earliest of the media's independent free-flight samples,
scatterer chosen by Poisson superposition" loop the forward tracer and the device kernel already
use, and give `neeVolume` the per-medium phase/albedo lookup. Then delete `backwardMedium()` and
the warning entirely. Until then, prefer `-device gpu` for any backward render of a scene with
more than one medium.

### BUG — DONE (2026-08-04, v0.128.0): the fp32 GPU build lost most of a DISTANT light's energy (a sun modelled as a far-away sphere rendered 2.7x too dim)

**Symptom.** `scenes/gallery_rain.ftsl` modelled the sun as a Lambertian sphere `radius 1.85`
at 400 m (subtending the sun's own 0.53°) carrying `power 2.011e9` = one solar constant. Mode D
on the **GPU** rendered it **2.7x darker than the CPU reference**; the same sphere moved to
40 m was 1.5x too dark, and by ~4 m the error vanished. `light sun` in the identical scene
matched the CPU to 0.15%, which is what made the loss look like a delta-light problem when it
was nothing of the sort — it was a plain float32 precision bug hitting *any* distant emitter.

**Reproduce** (`scraps/_sunmatch_{sun,sphere,sphere40}.ftsl`, a grey floor + one light, plus
`scraps/lumdiff.py`, which undoes the sRGB curve so the printed ratio is a real energy ratio):

```
ftrace -in scraps/_sunmatch_sphere.ftsl -camera cam -r 200 200 -time 10 -ev 0.02 -o png/a.png
ftrace -in scraps/_sunmatch_sphere.ftsl -camera cam -r 200 200 -time 25 -ev 0.02 -device cpu -o png/b.png
python scraps/lumdiff.py png/a.png png/b.png      # was 0.377, now 1.023
```

**Two independent root causes, both fixed in `src/render_cuda.cu`:**

1. **`intersectSphere` used the textbook `disc = b² − 4ac`.** Both terms are `O(dist²)` while
   their difference is `O(radius²)`, so the subtraction catastrophically cancels: a sphere `k`
   radii away carries roughly `k²` ulp of error in its hit distance. At 400 m with `r = 1.85`
   that is **±1 cm**. Replaced with the stable Ray-Tracing-Gems form — build the discriminant
   from the ray's *perpendicular* offset (`disc = a·(r² − |f⊥|²)`, every term `O(r²)`) and take
   the near root by Vieta (`c/q`). This also helps any camera ray hitting a distant sphere.
2. **Connection rays were shortened by a fixed absolute epsilon** (`dist - 2e-6`, inherited
   verbatim from the all-double CPU reference in `bdpt.h`). One ulp of a float is `dist·2⁻²³`,
   so past **~17 m** the `2e-6` rounds straight back to `dist` and the shortening vanishes —
   the shadow ray then ends exactly *on* the sampled light point and re-hits the emitter's own
   geometry, so the sample is discarded as occluded. Replaced by `connMaxT(dist, absEps)`,
   which shortens by `max(absEps, dist·CONN_REL_EPS)` with `CONN_REL_EPS = 1e-5` (fp32) / `0`
   (the fp64 build, which therefore stays bit-identical to the CPU). `absEps == 0` still means
   "don't shorten at all", which is what the distant-sun connection needs — its far end is the
   scene exit, not a sampled surface point. Applied at all nine connection sites (BDPT `s=1`
   / `t=1` / `s,t≥2`, the VCM equivalents, and the caustic chains' connections to the eye).

Either fix alone is insufficient: with only (2) the sphere was still 30% dark at 400 m.

**Validation.** `scraps/_sunmatch_*` GPU-vs-CPU energy ratio, was → now: sphere @400 m
**0.377 → 1.023**, sphere @40 m **0.662 → 1.023**, `light sun` **0.998 → 0.999** (unchanged, as
expected — a delta sun never shadow-rays *to* a surface point). `scenes/cornell.ftsl` mode D at
a matched 400 spp, GPU vs CPU: mean ratio **0.9978**, mean |diff| 0.0060, auto-exposure anchor
1.16e-13 vs 1.17e-13 — no regression at ordinary scene scale. All 16 `-check*` self-tests PASS.

**Note for anyone reading old output:** every GPU mode-D/U render of a scene with a light more
than ~20 m from the shading points was too dark before 0.128.0, the more so the further away
the light. Re-render rather than trusting the exposure of an archived frame.

### DONE (2026-08-04, v0.128.0): `scenes/gallery_rain.ftsl` now uses a real `light sun`

The scene used to model the sun as the distant sphere above, with two comments justifying it on
the grounds that GPU BDPT rejected delta lights. That stopped being true at 0.126.0/0.127.0
(the GPU BDPT and VCM kernels now do `spot`/`sun`), so the workaround and both comments were
stale. It is now `light sun { dir -0.1470 0.7071 0.6916  angle 0.53  spd blackbody 5800
intensity 9.2649e-14 }` — same direction, same 0.53° disc, same one solar constant, with
`intensity = 1000 / Σ₃₆₀^830 planck(5800 K, λ) dλ` because `light sun` takes an irradiance and
rejects absolute `power`. `dir` is spelled out because ftsl measures azimuth from **+x** while
`scraps/bowmap.py` measures it from **+z**.

The switch was not cosmetic. Independently of the fp32 bug above, the sphere needs **2.011e9 W**
to deliver 1000 W/m² across 400 m, which crushes the emitter power CDF: against the xenon lamp's
15 kW and the sky panel's 8 kW, `p(lamp) = 7.5e-6`, so the lamp — the light that keeps five
exhibits and the whole cloud off black — was sampled once in ~134,000 draws and arrived as rare
clipping fireflies rather than as illumination. Measured at 640×360 / 150 s / mode D, both
absolute-exposed: the frame is **2.1x darker** with the sphere, and the gap collapses to **1.10x**
once the lamp is deleted from both variants — i.e. the missing factor of two *is* the starved
lamp. `light sun`'s power is `irradiance·π·R_scene²` ≈ 4.8e5 W, putting the three lights at
~95/3/2 %. Dropping the sphere also shrinks the scene bounding radius from ~400 m to ~12 m.

### BUG — DONE (2026-08-04, v0.124.0): mode D could not render `spot` / `sun` lights at all — now it can (and the entry that claimed it did so *silently* was WRONG)

**Correction first.** The original version of this entry claimed a mode-D scene containing a
`sun`/`env`/`spot`/`collimated` light rendered black *with no diagnostic*, and that the
unsupported emitter *stole* its power-weighted share of the sample budget from the lights
that work. That was wrong, and reading the code rather than trusting the writeup is what
caught it. `main.cpp`'s `bdptUnsupportedFeature()` has always listed all four types and is
consulted at the mode-D/U guard *before* any BDPT dispatch, so the engine refused loudly:

```
[mode D] camera 'cam' uses spot / environment / sun / collimated lights, which that mode
can't render; use mode A/B/C/R, add a prefer{}/else{} fallback, or pass
-on-unsupported fallback|strip.
```

`-on-unsupported strip` cannot bypass it either (`stripUnsupportedFeature` only strips GRIN;
anything else falls back to mode R). So the `return 0` guards inside `bdpt.h` / `vcm.h` were
unreachable belt-and-suspenders, not a live silent-corruption path. What was real was the
plain **missing feature**: mode D simply refused those scenes.

**What was implemented.** `spot` and `sun` lights are now first-class in BDPT (mode D), CPU
backend, following PBRT's delta-light treatment:

* `bdpt.h` `isDeltaEmitter` / `isInfiniteEmitter` / `Vertex::isDeltaLight()` classify them.
* `deltaLightSubpath()` emits light subpaths: a spot fires from its point uniformly into the
  outer cone (pdf `1/(2π(1-cosOuter))`, the smoothstep penumbra carried as *throughput* so
  the mean walk weight is still the emitter's power); a sun fires from a point on a disc of
  radius `sceneRadius` one radius upstream of the scene centre (pdf `1/(πR²)`) along a
  direction drawn inside the solar cone (pdf `1/Ω`).
* `connectBDPT`'s `s == 1` NEE handles both: the spot connects deterministically to its point
  with the cone falloff, the sun samples a direction in its cone and shadow-rays to the scene
  exit (no `1/dist²`, no `cosLight` — the source is at infinity). All three emission models
  now share one λ-independent weight `Wgeom`.
* `vertexPdfLight` gained the spot-cone and the *planar* infinite-light density branches, and
  `vertexPdfLightOrigin` returns 0 for a delta light — matching PBRT, and matching the 0 the
  light subpath stores in `path[0].pdfFwd`, so both sides of every MIS ratio agree.
* `generateLightSubpath` applies PBRT's "correct subpath densities for infinite area lights"
  patch (rewriting `path[1].pdfFwd` to the planar form) — without it the forward and reverse
  densities disagree and the MIS weights are wrong.
* `misWeight` drops the `s == 0` strategy for a delta light (`light[0].isDeltaLight()`), since
  no eye ray can land on a mathematical point or on an infinitely distant disc.
* Because `s == 0` is gone, the sun's own disc — and every mirror/water glint of it — would
  otherwise vanish, and NEE cannot supply those (a specular vertex is not connectible). So
  `randomWalk` now reports where an eye ray *escaped* (`struct Escape`) and `renderRows` adds
  the sun's radiance with MIS weight exactly 1 when every eye vertex on the path was delta —
  precisely the case where no other strategy competes.

**Validated** by rendering the same camera in mode D and in mode R (the backward reference)
and comparing mean luminance (`python scraps/imgdiff.py A.png B.png`):
`scenes/_spot_cornell.ftsl` agrees to 0.2 %, `scenes/_sun_check.ftsl` to 0.01 %, and the new
mixed regression scene `scenes/_deltalight_mix.ftsl` (area quad + spot + sun + a mirror
sphere) to 0.33 % — R at 20000 spp vs D at 6000 spp, mean |diff| 0.0039. That last scene
authors **absolute** light units on purpose: a non-absolute pair is auto-exposed per image,
so comparing two auto-exposed frames by mean luminance could not have detected a global
energy error at all. Both modes put a single saturated pixel at (53,104) — the sun's disc
reflected off the mirror sphere, i.e. exactly the escaped-ray strategy — and mode D is
visibly *cleaner* than mode R (which sprays fireflies over that mirror), as expected of a
bidirectional estimator.

**What is still unsupported, deliberately** — each refused loudly at the mode guard, never
silently dropped:

* `env` lights in mode D/U. An environment is an infinite *area* light: it needs escaped-ray
  radiance from every direction plus importance-sampled lat-long emission, which the BDPT
  random walk and its MIS densities don't do. This is the genuinely large remaining piece.
* `collimated` lights in mode D/U. A delta emission *direction* from a finite surface means a
  shading point can never next-event-estimate it (the `s == 1` strategy has zero measure);
  only forward transport reaches it.
* `spot` / `sun` in the **GPU** BDPT — see the entry below.

### DEBT — DONE (2026-08-04): VCM (mode U) refused `spot` / `sun` lights that BDPT renders

**Resolved 2026-08-04 (0.125.0).** The delta-light treatment is now ported into `src/vcm.h`,
so mode U renders spot and sun lights exactly like mode D and both guards in `main.cpp`
(`vcmUnsupportedFeature()` and the mode-`U` branch of `modeFeatureUnsupported`) have dropped
their emitter check. What the port needed, in SmallVCM's compact `dVCM`/`dVC`/`dVM`
bookkeeping rather than pbrt's explicit pdfFwd/pdfRev loop:

* `traceLightSubpath` gained the two delta emission cases. Both sample uniformly in the cone
  (`pdfDirW = 1/Omega`; note `spotOmega` is the *falloff-weighted* solid angle, so the
  sampling cone `2*PI*(1-cosOuter)` is recomputed — they coincide only for a sun). A spot
  emits from `em.origin` with `pdfPos = 1`; a sun from a disc of radius `sceneRadius` one
  radius upstream of the scene centre, `pdfPos = 1/(pi R^2)`. `|cos|` at the light is exactly
  1 for both (the emission normal *is* the direction), and the spot's smoothstep penumbra is
  folded into the radiance, never into a pdf.
* **`dVC` (and hence `dVM`) start at 0 for a delta light**, and NEE forces `wLight = 0`. That
  is what *drops* the unsamplable strategies from the balance heuristic instead of
  under-weighting the ones that work. `dVCM` comes out as `Omega` for a spot and `pi R^2` for
  a sun — the same values SmallVCM's `PointLight` / `DirectionalLight` produce.
* `misArrival` gained a `foldDist2` flag, false for exactly one edge in the renderer: the
  first edge of an **infinite** light's subpath, whose density is the planar `1/(pi R^2)`
  rather than a solid-angle density (pbrt's "correct subpath sampling densities for infinite
  area lights" patch; SmallVCM's `if (pathLength > 1 || isFiniteLight)`).
* The camera NEE branch grew per-shape connection geometry: spot → `directPdfW = dist^2`,
  `emissionPdfW = 1/Omega`, `cosAtLight = 1`; sun → `directPdfW = 1/Omega`,
  `emissionPdfW = 1/(pi R^2 Omega)`, shadow ray out of the scene with **no** endpoint epsilon.
* `traceCameraSubpath` tracks `camAllDelta` and adds the **escaped-ray sun** at the
  `!h.valid` branch with MIS weight exactly 1 — the same strategy `bdpt.h` needed, and for
  the same reason (nothing else can reach a sun through a purely specular chain).

Validated against the GPU backward reference (mode R), all three regression scenes:

| scene | mode U | mean ratio | mean \|diff\| | note |
|---|---|---|---|---|
| `_sun_check.ftsl` | 400 passes, `-pmradius 0.02` | **1.0002** | 0.0038 | identical auto-exposure (4.74e-14); shadow-core box 0.0813 vs mode D's 0.0810 |
| `_spot_cornell.ftsl` | 300 passes, `-pmradius 0.003` | 1.0061 | 0.0068 | within mode U's 5.8% noise |
| `_deltalight_mix.ftsl` | 250 passes, `-pmradius 0.02` | 0.9949 | 0.0081 | **absolute** units, so this is a real energy comparison; the worst pixels are mode *R*'s fireflies |

The sun's glint on the mirror sphere is present and correctly weighted in mode U: the 7x7
window at (50,101) sums to exactly 1.0000 in R, U and D alike — one saturated pixel, not two.
Area-light scenes are **bit-identical** to before (same RNG draw order, same densities).

Note the residual gap at the *default* merge radius (`sceneRadius * 0.02`) is larger — mean
ratio 0.9834 on the mix scene — and shrinks monotonically with the radius. That is ordinary
photon-mapping radius bias, not a delta-light error: `-pmradius 0.02` (8.5x smaller) takes it
to 0.9949, and the mode-D/mode-R pair shows the same shadow-core value the small-radius mode U
converges to.

### DEBT — DONE (2026-08-04, v0.126.0): the GPU BDPT kernels now do delta lights, so mode D no longer falls back to the CPU on a spot/sun scene

`cudaBdptSupported()` (`src/render_cuda.cu`) used to reject `Spot` and `Sun` alongside `Env`
and `collimated`, routing such a scene to the CPU BDPT with a printed `[device] … ; using CPU`
notice. That was *correct* — the alternative would have been `dGenLightSubpath` /
`dConnectBDPT` returning 0 for the emitter they just spent a CDF draw selecting, which really
would drop the light and steal its share of the sample budget — but it cost the GPU speedup on
exactly the daylight scenes that want it most. The `bdpt.h` delta-light work (the entry at the
top of this file) has now been mirrored into the device kernels, one-for-one:

* `dIsDeltaEmitter` / `dIsInfiniteEmitter` / `dIsDeltaLightVertex` — device twins of
  `isDeltaEmitter` / `isInfiniteEmitter` / `Vertex::isDeltaLight()`.
* `dGenLightSubpath` grew the delta-emission branch (device twin of `deltaLightSubpath`):
  spot fires from `em.origin` uniformly into the outer cone (`pdfPos = 1`, `pdfDirW =
  1/(2π(1−cosOuter))`, smoothstep penumbra carried as throughput); sun fires from a disc
  point of radius `sceneRadius` one radius upstream of `sceneCenter` (`pdfPos = 1/(πR²)`)
  along a direction in the solar cone. `path[0].pdfFwd = 0`, `delta = 0`, `matId = -1`, and
  the infinite-light `path[1].pdfFwd` planar patch is applied after the walk.
* `dConnectBDPT`'s `s == 1` NEE now uses the same unified λ-independent `Wgeom` the CPU does
  — spot `fall/(d²·pdfChoice)`, sun `spotOmega/pdfChoice` with `occlEps = 0` so the shadow
  ray reaches the scene exit unshortened, area `cosL·A/(d²·pdfChoice)` — and sets
  `sampled.matId = -1` / `beta = Le·Wgeom` / `pdfFwd = 0` for a delta light.
* `dVertexPdfLightF` gained the spot-cone and *planar* infinite branches (and now takes the
  `DScene` so it can reach the emitter); `dVertexPdfLightOriginF` returns 0 for a delta light.
* `dMisWeight`'s `deltaPrev` at `i == 0` is the `IsDeltaLight()` test rather than a hard
  `false`, dropping the `s == 0` strategy — reading the SUBSTITUTED endpoint when `s == 1`.
* `struct DEscape` + an optional out-param on `dRandomWalk` / `dGenCameraSubpath` report where
  an eye ray escaped, and `kBdptT` adds the sun's radiance with MIS weight exactly 1 when
  every eye vertex was delta — the strategy that carries the solar disc and its mirror glints.

`cudaBdptSupported()` now rejects only `Env` / `collimated`.

**Validated** by rendering the same camera on both backends (`-device cpu` vs the default
GPU) at matched spp and comparing with `python scraps/imgdiff.py`:

| scene | mode D, spp | mean ratio | mean \|diff\| | note |
|---|---|---|---|---|
| `_spot_cornell.ftsl` | 400 | 0.9963 | 0.0036 | auto-exposure 6.42e-14 vs 6.45e-14 |
| `_sun_check.ftsl` | 400 | 0.9992 | 0.0039 | identical auto-exposure 4.75e-14 |
| `_deltalight_mix.ftsl` | 3000 | 0.9989 | 0.0034 | absolute units, fixed gain |
| `cornell.ftsl` (area-light regression) | 600 | 1.0128 | 0.0043 | auto-exposure 9.97e-14 vs 9.71e-14 |
| `cornell.ftsl` (same, 10× spp) | 6000 | 1.0043 | **0.0015** | mean \|diff\| falls 0.0043 → 0.0015 (√10× ⇒ pure MC noise, not bias); the residual mean ratio is auto-exposure jitter (a p99 statistic); worst pixels sit on the glass ball's caustic |

On the mix scene the escaped-ray solar-disc strategy is present and singly counted on the
GPU exactly as on the CPU: the 7×7 window at (50,101) — the sun's disc reflected off the
mirror sphere — sums to `0.02041 = 1/49` on both, i.e. one saturated pixel each. The GPU is
also ~14× faster there (16.5 s vs 229 s at 3000 spp), which is the point of the exercise.

Area-light scenes are unchanged apart from float association: the `s == 1` estimator now
spells `|cosSurf| · cosLight·A/(d²·pdfChoice)` where it used to spell `|cosSurf|·cosLight/d²
÷ (pdfChoice/A)` — algebraically identical, ~1 ulp apart — and every other edit is gated
behind `dIsDeltaEmitter`, which is false for them. The RNG draw order is byte-for-byte
unchanged on every path (the `u1,u2` pair is still drawn before the shape branch).

**Follow-on, now also DONE (2026-08-04, v0.127.0): the GPU VCM kernels do delta lights too.**
Relaxing `cudaBdptSupported()` above would have silently broken **GPU VCM**, because
`cudaVcmSupported()` was `cudaBdptSupported() && media.empty()` while `kVcmLightT` carried its
own `shape == 2 || 3 || 6 || collimated` reject — it would have drawn a spot/sun out of the
power CDF and then *discarded* it, losing the light AND its share of the sample budget. That
was patched over with an explicit spot/sun reject in `cudaVcmSupported()`; the reject is now
gone, because the kernels were ported. Against SmallVCM's running-partial `dVCM`/`dVC`/`dVM`
bookkeeping (not PBRT's explicit pdf arrays), the three rules from `src/vcm.h` transliterate as:

- **`kVcmLightT`** grew the per-shape emission block — cone sampling via `dSunSampleCone`
  (recomputing the sampling cone `2π(1−cosOuter)`, since `spotOmega` is the *falloff-weighted*
  solid angle), a point origin for a spot vs. a `sceneRadius` disc for a sun, `emitScale` kept
  out of every density, and `directPdfW = pdfChoice · (isInfinite ? pdfDirW : isDelta ? 1 : pdfPos)`.
  **`dVC` (and hence `dVM`) start at 0** for a delta light — the `dVC` analogue of
  `vertexPdfLightOrigin` returning 0. The resulting `dVCM` is Ω for a spot and πR² for a sun,
  exactly SmallVCM's `PointLight` / `DirectionalLight` constants.
- The **first-edge `dist²` fold is skipped** for a sun (`if (!(isInfinite && edges == 1)) dVCM *= dist*dist;`)
  — the device form of `misArrival`'s `foldDist2` parameter, false in exactly that one place.
- **`kVcmCameraT`**'s NEE branch grew the same three-way connection geometry as the CPU
  (`neeDirectPdfW` / `neeEmissionPdfW` / `occlEps`, with the sun's shadow ray running to the
  scene exit with **no** endpoint epsilon), and `wLight` is forced to **0** for a delta light.
- **`kVcmCameraT`** tracks `camAllDelta` and adds the **escaped-ray sun** at MIS weight 1 on a
  miss, gated on `sc.sunCount > 0`.
- Vertex connection (c) and vertex merging (d) needed **no** change on either backend: they
  read `dVCM`/`dVC`/`dVM` off the stored light vertices and the zeros propagate correctly.

**Validated** CPU vs GPU at matched passes:

| scene | mode U | mean ratio | mean \|diff\| | note |
|---|---|---|---|---|
| `_spot_cornell.ftsl` | 300, `-pmradius 0.003` | **0.9997** | 0.0030 | auto-exposure 7.51e-14 vs 7.48e-14; GPU 2.5 s vs CPU 427 s (**~170×**) |
| `_sun_check.ftsl` | 400, `-pmradius 0.02` | 0.9959 | 0.0048 | auto-exposure 4.76e-14 vs 4.81e-14 — the ~1% exposure jitter *is* the ratio |
| `_deltalight_mix.ftsl` | 250, `-pmradius 0.02` | **0.9996** | 0.0108 | **absolute** units / fixed gain, so this is a real energy comparison; GPU 1.0 s vs CPU 100 s |

The escaped-ray solar disc is present and singly counted on the GPU: the 4×4 window at (25,50)
on the mix scene reads `0.06250 = 1/16` on **both** backends (one saturated pixel each).

**Cross-check of the two ports against each other** (the strongest single result, because it
compares two *independently* transliterated device estimators rather than a port against its
own source): `_deltalight_mix.ftsl` at 400×300 in **absolute units / fixed gain**, GPU mode `D`
at 3000 spp (16.7 s) vs GPU mode `U` at 2000 passes (14.3 s) — mean ratio **0.9974**, mean
|diff| 0.0037, and the 7×7 solar-disc window at (50,101) reads `0.02041 = 1/49` in **both**.
The worst pixels cluster on the spot's caustic under the mirror sphere, i.e. VCM merge-radius
bias plus MC noise, exactly where the two estimators are expected to differ.

Area-light scenes are **byte-identical**: `cornell.ftsl` mode U on GPU at 150 passes hashes to
the same SHA-256 before and after the port (`015acdee6e10d5ce…`). The non-delta branch keeps
the original RNG draw order and the original expressions verbatim; every new density is behind
`dIsDeltaEmitter`, which is false for an area light.

### DONE (2026-08-03): the gallery Klein bottle is now a glassblower's bottle WITH THE INTERNALS, and it needs no mount

`scenes/gallery.ftsl`'s `klein` was `meshes/klein_hunyuan.obj`, an image-to-3D reconstruction: the
right silhouette, but a closed shell in which the neck's passage through the wall was only *implied*.
In a dielectric scene that is the whole point of the object, so it has been replaced by
`meshes/klein_bottle_full.obj` (from `d:\youtube\philosophy\3d objects\full_package_klein_bottle\`),
where the neck genuinely pierces the bulb and continues **down inside it** — a horizontal cut low in
the body shows four loops (the outer wall's two surfaces plus the descending inner tube), three at the
pass-through, and the tube alone above the shoulder.

The mesh checks out as a dielectric: 12262 v / 12264 quads, every directed edge used exactly once, so
closed, consistently wound and orientable; Euler characteristic **−2** (genus 2), which is what an
immersed Klein bottle in R³ must be; signed volume **+150.356** against **2520.73** of area, i.e. a
wall `2V/A = 0.119` raw units thick — **2.4 mm** at the shipped scale, real blown glass rather than a
slab. (`trimesh.load` reports it non-watertight with χ = 2 until `merge_vertices(merge_tex=True,
merge_norm=True)`; that is v/vt/vn vertex splitting, not a defect, and ftrace is unaffected because
`src/mesh.h` fan-triangulates off the raw `v` array.)

**The mount is gone.** `meshes/collar_klein.obj` and `tools/make_klein_collar.py` existed for exactly
one reason — the old bottle had *no* near-upright equilibrium (44 resting orientations, the most
upright leaning 73°), so only a gripping collar could hold it. The new one is a bottle with a punted
foot and simply stands: foot ring radius **66.2 mm**, COM **224.5 mm** above the foot, static tipping
angle **6.46°**. On the bare slate cap at settle_scene's own 5e-4 rolling/spinning friction, with its
own 0.03 m/s + 0.30 rad/s kick from 12 directions at 3 spawn tilts (`scraps/newklein_stand.py`): rest
drift **2.0 mm**, lean **0.00°**, and **36/36** pokes inside settle_scene's 10 mm POKE_TOL. The bake
itself reports `OK klein: margin +5.0 mm, contacts 4, on stand_klein, poke 0.2 mm`.

**A seat ring was designed and rejected on measurement**, and the reasoning is worth keeping. Take
true horizontal cross-sections of the placed piece (vertex binning is useless — the shell is hollow,
so a height band holds rings from both walls and the "max radius" alternates): the body flares
*continuously* off the foot, 66.2 mm at the base → 78.3 mm at 2.3 mm → 82.6 mm at 4 mm → 92.6 mm at
8 mm → 112.4 mm at 20 mm. So a bore has to clear the running maximum up to the ring's top, which means
the radial play it leaves at the base is exactly the flare over the ring's height — a bore loose enough
to lower the piece into is loose enough to let it slide the same distance. On top of that
`slab_sections()` quantises a static collider vertically at `STATIC_SLAB_MAX_T = 8 mm`, and 8 mm of
height on a 1.2–5 mm/mm flank is 1–4 cm of bore slop in the sim regardless of what is authored — the
same quantisation that once left an in-pedestal collar bore as a dimple with its floor 3 mm above the
real cap. A ring would have been decoration that made the sim worse.

Placement: raw mesh is Y-up with its **foot ring centred on the origin** and its base plane at
y = 0.025, extents 22.700 × 30.000 × 15.000, so `scale 0.02` → 0.454 × 0.600 × 0.300 — the same 0.60 m
height as the piece it replaces. The foot goes on the pedestal axis (5.9, 2.6) rather than the volume
centroid (which sits 41 mm toward +x), because the foot is what the eye reads as centred on the column.
`rotate 0 -12 0` turns the handle into profile for the hero camera, which sees this pedestal along
(−0.207, 0, 0.978). The bake now also seats it: `--seat heart:stand_heart,klein:stand_klein`, so the
base lands exactly `--seat-gap` (1 mm) over the cap instead of wherever the VHACD proxy's error left it.

### BUG — DONE (2026-08-03): an `expr` isosurface inside a rotated `group` was INVISIBLE to every ray-traced mode

Presented as "the morpho heart renders black in `png/heart_check.png`". It was not a material
problem at all — the heart was not being *hit*. (`-checkmultilayer` passes, and an Abeles
computation put `morpho`'s luminous normal-incidence reflectance at **0.5473** vs `oil-slick`'s
**0.3728**, i.e. it reflects *more* than a jack that renders vividly. Swapping in a plain diffuse
material left it equally invisible, which killed the material hypothesis outright.)

**Root cause.** A `function { expr ... }` field is not a distance function, so `intersectImplicit`
clips the ray to the authored `contained_by` box and sizes each step as `|f| / max_gradient` — a
bound the author only guarantees **inside that box**. `ftsl.h`'s `addIsosurface` stored only the
world AABB of the 8 transformed corners, and clipped to that. Under a rotation the AABB is
strictly larger than the box, and the field out there is far steeper. Measured for the gallery
heart (`scraps/heart_lipschitz.py`):

| region | max &#124;f&#124; | max &#124;grad f&#124; |
|---|---|---|
| authored `contained_by` box | 1688 | 18858 |
| AABB of that box after the settle rotation | 37738 | 245658 |

— **4.36× the volume**. The first step is then `37738 / 60 ≈ 629 m` across a 0.6 m object: the
sphere-trace leaps clean over it and reports a miss. Triggered the moment
`tools/settle_scene.py` baked a `group { rotate 50.6839 9.91871 -34.6649 }` rest pose onto the
piece — which is why `gallery.ftsl` showed the heart and `gallery_settled.ftsl` did not, with a
byte-identical heart block in both.

**Diagnostic signature worth remembering: the rasterizer showed it and every ray-traced mode did
not.** `isomesh.h` marching cubes samples a lattice and never sphere-traces, so it cannot
overshoot — that asymmetry localises a fault to the marcher immediately. (`-raster -raster-iso 96`
is a ~0.1 s geometry check; `-mode W -ambient 0.15` a ~10 s deterministic, noise-free ray-traced
A/B.)

**Fix (0.121.1).** `Implicit` now stores the container in its own frame (`boxOriented`, `boxInv` =
world→container-local, `boxLo`/`boxHi`) and both the CPU (`implicit.h`) and CUDA
(`render_cuda.cu`) intersectors run the slab test there; `Affine::applyDirTranspose` maps the two
face normals back to world. `estimateFieldLipschitz` likewise surveys the oriented box rather than
its inflated AABB, so a rotated piece doesn't get a needlessly large `L` that would slow every
march. `boxOriented` is set only when the local→world map is not axis-preserving, so unrotated
scenes take exactly the code they always did — verified by rendering `gallery.ftsl` (no oriented
container anywhere) with the pre- and post-fix binaries: **0 differing pixels, max channel delta
0**. No scene change was needed; `max_gradient 60` stays as authored, and bumping it would only
have papered over a general engine bug affecting every rotated expression isosurface.

**Regression test.** `-checkcontainer` (`checkContainer` in `main.cpp`) builds one sextic solid
twice — axis-aligned and rigidly rotated — under a shared `max_gradient`, and fires
correspondingly rotated rays. A rigid motion cannot change a hit distance, so any disagreement is
the clip region leaking outside the container. Confirmed to actually guard the bug: with the
oriented branch disabled it reports **76 vanished-when-rotated → FAIL**.

### TECH DEBT: `isomesh.h` caps a *rotated* container against its world AABB, not its true faces

`boxSDF` / `capBox` in `src/isomesh.h` still use `im.bounds` (the world AABB) for
`Container::Box`, so a rotated `capped` isosurface is sealed along AABB planes rather than the
authored box's real faces — the cap sits outside the intended surface on the rotated faces. Not
the overshoot bug above (marching cubes never sphere-traces, so it can't miss the object), and
currently invisible: it only matters for a surface that actually *reaches* its container, which
the gallery heart does not, and it only affects `-export-mesh` and `-raster`.

**Proper fix:** give the mesher the same treatment as `intersectImplicit` — when
`im.boxOriented`, transform the sample point by `im.boxInv` and evaluate the box SDF against
`im.boxLo`/`im.boxHi` (the local extents), scaling the result by the map's uniform scale so the
`max(f, contSDF)` blend stays in the same units as the field.

### BUG — DONE (2026-08-03): `settle_scene.py` reported a piece lying on the FLOOR as `OK` — both its acceptance tests are local

`heart` baked into `gallery_settled.ftsl` with `translate 1.19 -2.81 0.45 rotate -177.6 -45.3 56.1`
— a delta that puts its COM at y ≈ 0.35 when it was authored at y = 1.255, i.e. it had slid off
`stand_heart`'s cap, fallen ~0.9 m, and come to rest **wedged between the `stand_heart` and
`stand_dumbbell` shafts**. The bake reported:

```
  OK      heart:  margin  +10.2 mm  contacts   2  on stand_dumbbell, stand_heart  (poke  6.0 mm)
```

which reads like a success. `scraps/settled_aabb.py` independently said `heart cap top 1.000,
bot 0.213, gap -0.786` — the two tools flatly contradicted each other, and the wrong one is the
one that gates the bake.

**Root cause — not a physics bug, a *missing question*.** Both acceptance tests were LOCAL:

| test | asks | why it passed a piece on the floor |
|---|---|---|
| support margin | is the COM inside the hull of its load-bearing contacts? | it is — the wedge is a perfectly good support polygon |
| poke | does the rest survive a shove + spin? | it does — wedged between two granite shafts is *more* stable than the cap |

Neither test can see the authored scene, and to pybullet "wedged on the floor" and "sitting on
its pedestal" are both just *rest*. The verdict even *listed* `stand_heart` in `resting_on`,
because the heart grazed that shaft on the way down — so a naive "is it touching its stand?"
check would also have passed it.

**Fix.** A third acceptance test, computed against the AUTHORED pose before anything moves
(`intended_supports()` in `tools/settle_scene.py`): each settled piece's *intended* supports are
the other named objects whose plan (XZ) footprint overlaps its own and whose top is below the
piece's **mid height**. A piece then FELLs unless it is (a) resting on at least one of them
**and** (b) still above that support's top. Both halves are needed — (a) alone passes the
wedge (it touches `stand_heart`), (b) alone would be fooled by a piece that slid onto a
*neighbouring* cap at the same height.

The mid-height rule (rather than "the support's top is below the piece's underside") is what
lets a **mount** count: the retired `collar_klein`'s top was *above* the Klein bottle's lowest
point, because the bottle hung down inside its bore — yet it was the thing holding it up.

Two traps found while fixing it:

1. **Displacement is the wrong metric, even though it looks like the obvious one.** The first
   attempt gated on "did the COM move more than 60 mm from where it was authored". It is wrong
   in both directions: `heart` settling *correctly* under `--tether` moves its COM **222 mm**
   (it honestly tips from its authored tilt onto a stable lobe), while a piece can slide clean
   off a narrow cap having moved far less. What matters is *what it ends up on*, not how far it
   went.
2. **The verdict order matters.** `FELL` must be reported before `PERCHED`/`TOPPLES`, because a
   piece on the floor has *healthy* margin and poke numbers — that is the whole point of the bug.

Also fixed in the same change: the canonical bake command in `scenes/gallery.ftsl`'s header now
carries `--tether --seat heart:stand_heart` (it had neither), with a comment saying what breaks
without each. The bake is now `OK` for all five pieces on the first attempt.

### TECH DEBT (2026-08-03): `settle.drop()` falls back to the surface's TOP PLANE when the footprint "misses" the mesh

Seating the heart prints `[settle] warning: object footprint misses the surface mesh; resting on
the surface top plane instead.` even though the rotated heart's footprint is comfortably inside
`stand_heart`'s 0.50 m cap — so the raycast test in `tools/settle.py`'s `drop()` is rejecting a
footprint it should hit. The fallback happens to give the exactly right answer here (the cap top
*is* a flat plane at y = 1.000, and the seated heart lands 1.3 mm above it, fully 24 mm inside
every cap edge), which is why it has gone unnoticed. It would be silently wrong on a stand with
a domed, stepped or sloped top. Proper fix: find out why the down-rays miss — likely the
footprint sample points or the ray origin height — rather than leaning on the plane fallback.

### BUG — DONE (2026-08-03, v0.121.0): the exported Klein-bottle OBJs are a solid ball — the container sphere got welded on as a cap

`meshes/klein_a120_b060_c30_d127.obj` and `..._lite.obj` (the mesh `gallery.ftsl` /
`gallery_settled.ftsl` load as the object named `klein`) render as a **featureless sphere**.
Measured: both files are a closed shell at `r = 10` — 64.9 % of the full file's 1 096 216 verts and
65 % of the lite file's faces lie at `r > 9.3`, and the bbox is exactly ±10 on every axis. Strip
those faces (`scraps/klein_core.obj`) and a real Klein bottle is inside; see `png/klein_look.png`
(shipped mesh left = sphere, core-only right = the bottle).

**Root cause.** The export scene `scraps/klein/mesh_export.ftsl` is missing the `open` keyword —
it is the *only* scene in `scraps/klein/` that lacks it (compare `mesh_hi.ftsl` / `mesh_open.ftsl`,
which are otherwise identical in the isosurface block). The Klein field is authored **negated**
(`clamp(-(...), -10, 10)`, matching POV's sign), so `f < 0` — "solid" — is the region *outside* the
bottle. With `capped = true` (`isomesh.h`, `const bool doCap = im.capped;`) the marcher therefore
seals the entire `contained_by` sphere as a cap, and the exported solid is *the ball with a
Klein-bottle-shaped void hollowed out of it*. Reproduced at res 96: capped → 402 140 tris, bbox
±1.786 (= the container); `open` → 142 952 tris, bbox x ±1.298, y 0.263..2.300, z ±1.465 (= the
bottle), and it renders as a Klein bottle (`png/klein_open_view.png`).

**Fixed (v0.121.0).** Three parts:

1. `scraps/klein/mesh_export.ftsl` gained the missing `open`, with a comment saying why it is
   load-bearing.
2. **A permanent guard in `-export-mesh`.** `isomesh::capFraction()` (`src/isomesh.h`) classifies
   each output triangle by whether the container SDF or the field won the `max()` at its centroid,
   and `main.cpp` warns when the cap is more than half the output. Verified both ways: the capped
   Klein export reports `WARNING: 66% of these triangles are CONTAINER CAP` (matching the measured
   65 % shell), while the `open` export and all 17 gallery isosurfaces — several of which are
   legitimately capped — stay silent.
3. **The scene no longer uses the procedural export at all.** The user wanted their own
   AI-photogrammetry bottle, which was still on disk at `scraps/klein_hunyuan_clean.glb`
   (Hunyuan3D). Converted to `meshes/klein_hunyuan.obj` (317 140 v / 634 280 f, watertight,
   consistent winding, volume 0.870) so it can legitimately be a dielectric, and wired into
   `gallery.ftsl` at `scale 0.30` → 0.375 × 0.598 × 0.314, which fits `stand_klein`'s 0.52 cap.
   The stale `meshes/klein_a120_*.obj` files are no longer referenced by any scene.

The mislabelled "GYROID SPHERE in glass … triply-periodic minimal surface trimmed to a ball"
comment above the block (written when the ball silhouette was mistaken for an intentional trimmed
gyroid) is also gone.

### BUG — DONE (2026-08-03, v0.121.0): `settle_scene.py --tether` bakes poses that overhang the pedestal — the acceptance check only tests COM height

Two of the settled pieces in `scenes/gallery_settled.ftsl` are visibly off their stands. Measured
by exporting every isosurface to world-space OBJ groups (`ftrace -in scenes/gallery_settled.ftsl
-export-mesh scraps/gal_dump.obj -mesh-res 48`) and taking per-group AABBs:

| piece | settled COM (x,z) | its cap (x,z) extent | lateral drift | verdict |
|---|---|---|---|---|
| `oiljack` | 5.58, 1.87 | 5.07..5.53, 1.27..1.73 | **0.46 m** (cap half-width 0.23) | hangs off the cap's corner, bbox y 0.67..1.14 vs cap top 0.92 |
| `heart` | 4.83, 2.29 | 4.35..4.85, 1.65..2.15 | **0.45 m** (cap half-width 0.25) | overhangs toward `stand_dumbbell`, bbox y 0.98..1.31 |
| `brass_cluster` | 5.90, 2.01 | 5.73..6.27, 1.73..2.27 | 0.10 m | fine |
| `brass_dumbbell` | 4.55, 2.78 | 4.25..4.75, 2.45..2.95 | 0.05 m | on the cap, but see the next issue |

**Not** a transform-order bug: `R·c + t` reproduces every measured position to < 0.07 m (bbox-centre
vs COM residual), while `R·(c + t)` is off by 1–4 m — so FTSL's `group { translate; rotate }` and the
bake agree, and the renderer puts each piece exactly where the bake said.

**Root cause.** `--tether` (added in 45fad2d) applies a horizontal restoring spring at the COM every
step. Being a fictitious body force it does not vanish at rest, so the solver can converge on a pose
that gravity alone would not support — a piece resting on the *corner* of its cap with its COM out
over empty air. The 2026-07-19 acceptance check (`scraps/check_settle_heights.py`) then passed it,
because it only asks whether the settled COM *height* is within 0.45 m of the stand top; a piece that
drifted 0.46 m sideways but stayed at the right altitude reads as "on stand". That is why the fix
was logged as successful while the render shows two pieces hanging in the air.

**…but the tether was only hiding a deeper AUTHORING bug.** Releasing the spring made every piece
fall off, which is the correct physics: the hero CSG bodies are authored in the **unit cube**, so
their world centre is `translate + 0.5·scale`, *not* `translate`. `brass_cluster` does that
arithmetic (`5.6 + 0.4 = 6.0` = its cap centre); `oiljack` and `heart` do not — written as
`translate 5.3 … 1.5` (stand_oil's cap centre) the jack actually sat at (5.61, 1.81), hanging
0.31 m off the +x+z corner of its own pedestal. Measured authored-vs-stand centres before the fix:

| piece | authored world centre | its stand's cap centre | error |
|---|---|---|---|
| `oiljack` | 5.610, 1.810 | 5.3, 1.5 | **+0.31, +0.31** |
| `heart` | 4.766, 2.203 | 4.6, 1.9 | **+0.17, +0.30** |
| `brass_dumbbell` | 4.550, 2.750 | 4.5, 2.7 | +0.05, +0.05 |
| `brass_cluster` | 5.973, 1.953 | 6.0, 2.0 | −0.03, −0.05 (correct) |

**Fixed (v0.121.0).** In `scenes/gallery.ftsl`, all four pieces were re-centred on their caps and
dropped to 0.03 m above the cap top (verified by AABB dump: every centre now matches its stand's to
≤ 1 mm). In `tools/settle_scene.py`: the tether is ramped to zero and the pose re-settled before it
is read (`RELAX_RAMP_STEPS`), the acceptance test is now the real one — the COM must project inside
the convex hull of the **load-bearing** contact points (`convex_hull_2d` / `support_margin`) — and
the tool prints a per-piece stability table plus how far each piece moved when the spring was let go.

**Two further bugs found while validating that check**, both of which made it lie:

* `run()` exited early on its "everything has been still for 0.5 s" test — which is true the
  instant the release phase *begins*, since the pieces have just settled. So the ramp stopped at
  ~75 % stiffness and the "gravity-only" pose was still tethered. Fixed with a `min_steps` floor.
* Reading contacts after `p.performCollisionDetection()` returns **zero normal force on every
  point**: that call rebuilds the manifolds, and a freshly created contact point has no applied
  impulse yet (`normalForce` is the solver's impulse, not a geometric quantity). Pieces genuinely
  resting on their stands reported `contacts 0 … resting on nothing`. Fixed by reading the
  manifolds the last `stepSimulation()` left behind, and by disabling sleep on the dynamic bodies
  (`ACTIVATION_STATE_DISABLE_SLEEPING`) so a settled body stays in the solver.
* The load-bearing cut was an **absolute** force (`1e-3 N`). A unit-mass body at 240 Hz carries only
  `m·g·Δt = 0.041 N·s` of normal impulse in total, and a VHACD proxy resting on a concave trimesh
  splits that across one manifold per convex-child/triangle pair — so each point's share falls below
  the cut as the proxy gets *finer*, and `oiljack` again reported "contacts 0" while sitting
  squarely on its stand. The cut is now a fraction of the body's own total impulse
  (`CONTACT_FORCE_FRAC = 0.01`), which is scale-free.

**And a third, unrelated to physics:** `find_ftrace()` probed `build_cuda/bin/ftrace.exe` first, but
`build.bat` builds into `build_cuda2/` and installs to the repo root — so every settle bake had been
polygonising the scene with a **17-July binary**. It now prefers the repo-root exe and falls back to
build dirs newest-first.

### BUG — DONE (2026-08-03, v0.121.0): `brass_dumbbell` settles balanced on its ring — an unperturbed knife-edge equilibrium

`brass_dumbbell` is a wheel: `torus { rotate 0 0 90 major 0.15 minor 0.045 }` (axis along X, so the
ring lies in the YZ plane) with a sphere on each end of an X-axis bar. Scaled by 0.78 the ring's outer
radius is 0.152 and the spheres' is 0.117, so **only the ring rim can touch the cap** — the balls
never reach it. Its baked pose is `rotate 11.0397 0.346691 -0.760179`, i.e. essentially untilted, and
its world bbox is y 0.88..1.18 with the cap top at exactly 0.88: it is standing on the rim like a
bicycle wheel, which is what the user reported ("balancing evenly on the ring"). A rigid-body solver
started from a perfectly symmetric pose with no lateral perturbation has nothing to break the
symmetry, so it never topples.

**Fixed (v0.121.0)** — and it took four separate fixes, because each one exposed the next.

1. **Spawn jitter.** `--jitter` (default 2°) tilts each piece about a random horizontal axis at
   spawn, lifted by the sagitta that tilt sweeps (`rmax·(1 − cos θ)`) so it doesn't drive a corner
   into the stand.
2. **Automatic retry.** Jitter alone is not enough and the first re-bake proved it: this body is (to
   within the small bored-out bite) a *solid of revolution about X*, so a draw near the X axis maps
   the body onto itself and perturbs nothing. The measured delta came back `rotate 8.59 −0.81 0.42`
   — a rotation about X, i.e. the piece spun about its own symmetry axis and stayed on the rim. Any
   piece still unstable is now re-thrown with a fresh draw up to `SETTLE_ATTEMPTS` (4) times,
   re-using the built collision world so VHACD isn't repeated.
3. **`rollingFriction` was holding it up.** Bullet's `rollingFriction` is a resistance **arm in
   metres** — it caps the resistive torque at `mu_r · N`, so a body of radius R cannot tip past
   `asin(mu_r / R)`. The bodies were created with `rollingFriction = 0.02`, i.e. **2 cm**, on a wheel
   of world radius 0.117: tipping was capped at 9.8°, and the second re-bake's baked tilt was 5.05°
   — sitting right under the cap. So the "ring balance" was being maintained by a fictitious torque.
   Real metal-on-stone rolling resistance is a fraction of a millimetre; now `5e-4`.
4. **The support-polygon test can't see this failure**, because a wheel on its rim has its COM
   *exactly* over its contact point — it is in perfect equilibrium, just an unstable one. Added a
   **poke phase**: after the free settle each piece gets a small random shove + spin and is
   re-settled; a stable rest absorbs it, an unstable one topples. The pose that survives the poke is
   the one baked, and the report distinguishes `PERCHED` (overhanging) from `TOPPLES` (balancing).

**And the shape itself had to change.** With the sim finally honest, the piece has *no stable rest
pose at all*: ring outer radius `0.195·0.78 = 0.152` vs ball radius `0.150·0.78 = 0.117`, so the ring
hangs 35 mm below the balls and they can never reach the stand. Tipping about Z brings a ball down at
14.1°, but at that tilt **both** contacts (rim at +0.037, ball at +0.121) are on the same side of the
COM, so it keeps going — all the way to axle-vertical, balanced on one ball. No simulation can invent
a rest pose that doesn't exist. `gallery.ftsl`'s torus is now `major 0.105 minor 0.038` (outer 0.143 <
0.150), so the balls are the lowest feature and the piece rests on its two spheres like a dumbbell
should.

### LIMITATION (2026-08-02): emissive geometry with no registered emitter is invisible to NEE, so it lights nothing

A material's `emit` makes *any* surface glow — including a marched isosurface or a CSG / quadric
solid, which have no triangles to register an `Emitter` against (only the single-material mesh
path calls `Scene::addMeshLight`). Those surfaces are picked up by **emission-on-hit only**: a
camera or specular ray that lands on one sees its radiance, but NEE and light subpaths can never
sample a point on them. Consequences: an emissive isosurface casts **no** light on the room — or
on itself, since a mode-W gather ray arrives with `specularArrival = false` and so takes emission
only through NEE — and it contributes nothing at all in the forward modes A/B/C, where transport
starts *at* an emitter. `tools/loom/examples/glowing_jack.py` documents the practical fallout:
turning up the jack's emission brightens the jack and nothing else, so the room still needs its
own panel to shade the subject.

**Proper fix:** tessellate emissive implicit/CSG surfaces at load (the machinery exists in
`src/isomesh.h`) into a sampling-only proxy mesh and register it via `addMeshLight`, keeping the
marched surface for intersection. The proxy's area and per-point radiance must agree with the
marched surface closely enough that MIS stays unbiased, which is the hard part and the reason
this is deferred rather than bodged.

### TECH DEBT (2026-08-02): `fieldLeafSDF` computes `c.r = sqrt(x²+y²+z²)` unconditionally per Expr eval

`src/implicit.h` (~95–105, the `FieldNode::Expr` case): every field-formula evaluation pays a
`std::sqrt` to populate the `r` pattern variable whether or not the program reads `VarR` (the
device twins in `render_cuda.cu` do the same). Profiling mode W on the gyroid scene (which never
uses `r`) puts this at roughly 2–4% of `patternEval`-path time. **Proper fix:** a `usesR` flag
computed once per program (scan for `PatOp::VarR`), carried on `FieldNode` / `DFieldNode` /
`DFieldNodeF` through the upload conversion, and checked at the 4 eval sites before the sqrt.
Pure elision of a dead store, so bit-identical by construction. Deferred from the 0.118.0
mode-W optimization pass as below its noise floor.

### TECH DEBT (2026-07-30): loom's `Isosurface` cannot emit a CSG field tree, so every scene that wants one hand-rolls its own `Element`

Found while writing `tools/loom/examples/jumping_jack.py` (a jack of six sphere+cylinder arms
carved out of a **world-static** gyroid). FTSL has had analytic CSG inside `isosurface` since
v0.115.0 — `union` / `intersect` / `difference` / the `k`-blended smooth trio over
`sphere` / `ellipsoid` / `box` / `torus` / `cylinder` / `cone` / `plane` / `function` leaves — but
loom cannot reach any of it:

- `loom/iso.py` (~342) hard-codes exactly one `function { expr … }` plus one `contained_by`, so
  the field tree is unreachable from the authoring API.
- `Room` (~410) *does* transform a field, but by folding the affine into the **field** frame
  (`M_eff = M·Pᵀ`, `p_eff = P·p_local + T`). That is structurally the wrong mapping for the
  common case of a **moving solid sampling a stationary field**: it drags the lattice along with
  the object instead of letting the object sweep through it. The world-static idiom needs the
  animated transform on the *shape* leaves and **no** transform on the `function` leaf.

The workaround in `jumping_jack.py` therefore re-derives three non-obvious things by hand, each
of which belongs in the library:

1. **Aiming a cylinder.** An FTSL `cylinder` leaf's axis is local +y, so pointing an arm along a
   direction `d` needs the Euler triple for ftrace's own composition `R = Rz(rz)·Ry(ry)·Rx(rx)`
   (`src/mesh.h` ~119, `affineFromTRS`): `rx = asin(d.z)`, `rz = atan2(−d.x, d.y)`, `ry = 0`.
   Verified numerically in `scraps/jack_check.py` (max residual 6.5e-16), but no caller should
   have to know it.
2. **A pose-independent `contained_by`.** Required on any `function` field; must bound the solid
   over the *whole* animation, not the current frame.
3. **Gradient renormalisation.** A raw gyroid at frequency `f` has `|∇h| ≤ 2√3·f`, which would
   drag the sphere-tracer to `d/(2f)` steps. Dividing the expression by `2f` gives `|∇h| ≤ √3`,
   so one honest `max_gradient 2` covers the tree and the march runs ~`f`× faster.

**Proper fix.** `Field` leaf classes (`FSphere`, `FCylinder`, `FBox`, `FTorus`, `FExpr`), each
carrying an animatable `Transform`, plus `FUnion` / `FIntersect` / `FDifference` and the
`k`-blended smooth trio; `Isosurface` accepting a tree as its `field`; a library `aim_y(direction)
-> rotate` that matches `affineFromTRS`; a `contained_by` declarable as pose-independent; and
automatic gradient renormalisation of an `FExpr` leaf reusing the per-family bounds `nd_grad_bound`
already knows (`√2` / 1 / `2^((n−1)/2)` / 7). Documented as a deliberate gap in
`tools/loom/DESIGN.md` §7c until then.

**A fourth thing the workaround re-derives, added after the fact: picking a carving level by
volume fraction.** How holey an `intersect { solid, function }` looks is set by the volume
fraction of the field's sub-level set, and there is no safe closed form for it. Measured on the
3-D gyroid: the *band* fraction is `volfrac(|g| ≤ g₀) ≈ 0.647·g₀` but the *one-sided* fraction is
`volfrac(g ≤ −g₀) = (1 − 0.647·g₀)/2` — **half** the slope, which is an invisible factor-of-two
bug (asking for `solid = 0.26` silently delivered 0.378, making every "sparse" sweep look
half-filled), and any linear fit is additionally off by ~2× in the tail (0.15 → 0.325).
`jumping_jack.py` now inverts the distribution exactly (`_gyroid_samples` sorts a 48³ sample of
one period, cached and frequency-independent; `gyroid_quantile` / `gyroid_cdf` read it), verified
round-trip to 3 decimals. This inversion belongs next to `nd_grad_bound` in the library, per
field family.

**A fifth: separating "lacy" from "see-through", and fusing the result into one leaf.** The
volume fraction alone cannot do it — for a one-level carve the surviving envelope *is* the
volume fraction, so the two move together. Measured over 32 placements × 4000 rays
(`scraps/see_through.py`, paired), the fix is a sparse **counter-network**: union `g ≥ t`, the
gyroid's *other* labyrinth, onto `g ≤ c`. It occupies the middle of the first one's voids —
where the sight-lines are — so see-through halves (14.9% → 7.5%) for ~2 points of envelope,
where buying that out of `solid` costs ~12. Four rival mechanisms measured worse or uglier, and
notably the intuitive "rotate it in a higher dimension" (a quasiperiodic 4-D slice) is *worse*
than the plain carve, 18.4% vs 14.9%. Two traps recorded so they are not re-hit: a copy shifted
half a period in all three axes is the **identity** on a gyroid (`sin(x+π)cos(y+π) = sin(x)cos(y)`)
and an *unpaired* Monte-Carlo sweep hid that behind ~1.7 points of noise — always re-seed per
candidate. The pair must also be emitted as **one** `function` leaf: ftrace evaluates every leaf
at every march step, so a two-leaf `union` doubles the trigonometry in the hottest loop; here
`min(a,b) = ((a+b) − |a−b|)/2` with `a+b` constant collapses it to `((t−c) − |2g−c−t|)·s/2`,
exact and with the same Lipschitz bound. A library `Field` API should do this fusion (and the
volume-fraction inversion above) automatically rather than leaving it to each scene.

**Also worth recording as a hard limit, not debt:** the lacy look of the reference stills
(`png/gold_gyroids`, `png/gyroid_nd`) comes from `contained_by` **clipping** a bare `function`
sheet, so there is no envelope surface at all. That is *not expressible through CSG* (whose job
is to bound a solid), and it cannot be done per-arm even by hand, because `contained_by` takes a
single axis-aligned box or sphere — not a rotating ball-and-rod, and not one per CSG leaf. A
sparse carve is the CSG analogue; if a genuinely clipped multi-part sheet is ever wanted, FTSL
itself would need per-leaf clip regions.

### BUG — FIXED (2026-07-30, v0.117.0): `scraps/gi_collapse.ftsl`, the `-gi` normalisation regression test, was VACUOUS — auto-exposure divided out the very error it tests for

Found while validating `-gi-clamp`. The scene tests the gather's cosine-normalisation invariant:
in a scene where every gather ray escapes, `-gi 32` must be pixel-identical to `-gi 0`, or
"turning `-gi` on would step the exposure of every scene". Its documented invocation passed
`-exposure 1`, which is not an absolute exposure — it is a **compensation multiplier on top of a
p99 auto-exposure** (`main.cpp` ~2518). The scene renders one flat uniform diffuse quad, so the
p99 anchor normalises away **any** overall scale factor. The test therefore passed regardless of
how badly the estimator mis-scaled: exactly the failure mode it exists to catch. Confirmed
directly — `-ambient 0.05` and `-ambient 0.1` produced **byte-identical** PPMs.

This also invalidated a validation result recorded during N3c ("collapse invariant holds") and
briefly sent me chasing a phantom: `-gi-clamp 0.05` under `-ambient 0.1` came back identical to
`-gi 0`, contradicting a caveat I had just written into three files. The caveat was right; the
test was blind.

**Fix.** The light now carries `lumens 8000`, which puts the scene in **absolute** mode
(`ftsl.h` ~4679 — a fixed sensor gain instead of an auto-exposure anchor), and the header says
so in capitals, drops `-exposure` from the documented invocation, and carries a
**discrimination check**: `-ambient 0.05` must render exactly half as bright as `-ambient 0.1`
(centre pixel `0x2a` vs `0x3c`), so the test cannot silently go vacuous again. Re-verified with
the fix in place: the invariant genuinely holds, pixel-identical, on both the CPU and the GPU.

**Lesson worth generalising:** any regression test that compares *tone-mapped* frames is only
measuring what the tone mapper did not remove. `scraps/cor_gi.ftsl` already had this right and
says so in its header — a flat-fill sweep there made a constant ambient look like it made the
image *worse* because the anchor swung 5× across the sweep. Prefer `lumens`/`power` (absolute
mode) in any scene used for a numeric comparison, and never `-exposure` as a stand-in for it.
`scraps/gi_firefly.ftsl`, added for `-gi-clamp`, follows the same rule.

### NOT A BUG, but a sharp edge worth a knob — KNOB SHIPPED (2026-07-30, v0.117.0): mode W's `-gi` gather ALIASES a caustic into thin bright contour curves

Noticed on the v0.116.0 showcase render (`scenes/cornell.ftsl`, `-mode W -gi 64 -spp 4`, 900×900):
the floor around the SF10 glass ball, and the side walls, carry a family of thin, blown-out,
**dashed white curves** concentric with the ball. Diagnosed by bisection rather than by reading
code, then confirmed against the code:

| probe | result |
|---|---|
| `-gi 0` (flat ambient) | curves **gone** → it is the gather |
| `-gi-bounce 1` | curves **gone**, smooth soft rings remain; the caustic spot under the ball also goes |
| `-gi-bounce 2` | curves **back**, caustic spot still absent |
| `-spp 64` | curves **gone**, integrated into a smooth glow |
| CPU vs GPU, 450², identical flags | max \|dLuma\| **0.123** / \|dChroma\| 0.192 → identical estimator, *not* a porting artifact |

**Mechanism.** The path is *diffuse floor → gather ray → glass ball → lamp*. `bkRadiance` starts a
gather ray at `specularArrival = false` (correct — the vertex's own NEE already counted that
emitter), but the dielectric sets it back to `true`, so the subsequent emitter hit adds the lamp's
**full radiance** (`render_cuda.cu` ~6672 and ~6776; host twin in `backward.h`). That is the right
thing to do: NEE cannot sample a lamp that sits behind a refracting surface, so emission-on-hit is
the *only* estimator that path has. The contribution is real and the estimator is unbiased in the
lattice rotation — hence the clean convergence at `-spp 64`.

It renders as a **curve instead of noise** because mode W deliberately shares one world-space
direction lattice across every pixel (that invariant is what makes the mode noise-free). So
"does direction *k* reach the lamp through the ball?" is a step function of surface position whose
boundary is a single coherent contour in image space, where a stochastic renderer would smear the
same discontinuity into grain. The dashes are plain aliasing — the contour is sub-pixel-thin in
places and only registers where it passes near a pixel centre.

**Status: RESOLVED in v0.117.0 by an opt-in `-gi-clamp <x>`** (user asked for it). The behaviour
was never wrong — it is correct and it converges — so the clamp is **off by default and bit-for-bit
inert when off** (verified: `cmp` against the v0.116.0 baseline PNG is byte-identical). README's
"Honest limits" now names three levers, cheapest last: `-spp 64`, `-gi-bounce 1`, and
`-gi-clamp 0.1`.

`-gi-clamp x` caps the radiance **one** gather ray may return at `x` times
`Scene::ambientRef()` — the same dimensionless "multiple of one light's own radiance" units as
`-ambient`, so one number works at any scene scale. Implementation notes (full rationale on
`BackwardRenderer::giClamp`, `src/backward.h`; device twin `DScene::bkGiClamp`):

* **Per wavelength, not per bundle.** The scalar twin `giGather()` carries a single λ and has
  nothing to take a max over, so a bundle-wide rule would make the hero and single-λ paths
  disagree on the same scene. Costs a hue shift on a clamped ray, which is the point.
* **`wSum` is not clamped.** A clamped direction keeps its weight `c`, so the estimator still
  normalises by the realised sum of cosines and the collapse invariant survives.
* **It also caps the far-field `ambient` tail** an escaping gather ray returns. Not a bug, but it
  couples the two knobs: the gather's fill is exactly `min(ambient, giClamp)`. Pinned in
  `scraps/gi_collapse.ftsl` — `-gi 32 -gi-clamp c` is pixel-identical to
  `-gi 0 -ambient min(ambient, c)`. Hence "keep it above `-ambient`", or you darken the whole
  scene instead of capping fireflies. `-gi-clamp` without `-gi` prints an `[ignore]` notice.

**Measured** on `scraps/gi_firefly.ftsl` (added for this — the absolute-mode Cornell box *with*
the glass ball, the deliberate counterpart to all-diffuse `cor_gi.ftsl`), `-gi 32 -spp 1
-ambient 0.05`, 240², GPU:

| `-gi-clamp` | frame Δluma | pixels > 250 codes |
|---|---|---|
| `0` (off) | — | 1689 |
| `0.05` | −0.86 % | 1350 |
| `0.1` | **−0.31 %** | 1350 |
| `0.2` | −0.20 % | 1356 |
| `0.5` | −0.04 % | 1600 |

So the curves cost ~0.3 % of the frame's light to remove, and the caustic survives as a soft
highlight rather than a blown-out spike. (The `0.05` row is dearer only because it is *at*
`-ambient` and has started eating the fill.) CPU↔GPU with the clamp on: 98.7 % bit-identical,
max 20-px-block |dLuma| 0.328 code against a 1.5-code bar. All 15 self-tests pass.

**Rejected alternatives:** (a) suppressing emission-on-hit for gather rays entirely — kills the
artifact but silently discards a real caustic and makes `-gi` darker than correct; (b) forcing
`giBounce = 1` by default — same energy loss, and it would change existing images.

### BUG — FIXED (2026-07-30, v0.116.0): the GPU's *watertight* triangle test CRACKS, because nvcc contracts its edge functions into FMAs

Found while validating N3c: `scraps/cor_gi.ftsl` at 240×240 in mode `W` failed the CPU↔GPU
block-mean bar badly (max |dLuma| **7.668** codes, |dChroma| **3.062**, bar is 1.5) with **134
pixels** more than 8 codes off. `scraps/n3c_diffmap.py` (written for this) localised them, and
the shape was the giveaway: every bad pixel lay on `x == y` or `x + y == 239`, i.e. exactly the
frame diagonals, and every one of them was pure `(0,0,0)` on the GPU against a lit
`~(247,234,236)` wall on the CPU. Nothing to do with `-gi` at all — it reproduced at `-gi 0`.

The camera is exactly centred on a square Cornell box, so the back-wall quad's own
triangulation diagonal, plus all four ceiling/floor-to-side-wall corner seams, project **dead
through hundreds of consecutive pixel centres**. The GPU was losing the hit at every one of
them: the background leaking through a closed surface, from the intersection routine whose
entire selling point is that this cannot happen.

**Where:** `src/render_cuda.cu`, `intersectTri` — the three scaled barycentric edge functions of
the Woop watertight test (Woop/Benthin/Wald/Áfra, JCGT 2013), plus the exact-zero double
fallback right after them.

**Mechanism.** The watertight guarantee is that two triangles sharing an edge evaluate that edge
from **bitwise identical operands in opposite order**, so their edge functions come out exact
negatives; the accept test rejects only *mixed* signs, so a zero is accepted and a ray dead-on
the edge is claimed by exactly one sharer. nvcc defaults to `-fmad=true` and contracts
`a*b - c*d` into `fma(a, b, -(c*d))`, which keeps **one** product exact and rounds only the
other. On an exact tie the two sharers therefore compute `exact(pq) - rounded(qp)` and
`exact(qp) - rounded(pq)` — which are *equal*, not negatives. If that shared sign is the
minority one, **both triangles reject and the surface cracks.** The exact-zero fallback had the
same problem, and it is precisely the tie case, so it is the code that most needs to stay
antisymmetric.

This is the same failure and the same fix as the CPU/GPU rasterizer's `edgeRow`/`edgeAt`
(v0.98.2, item 6 below) — the ray-tracing analogue, missed at the time because the rasterizer
fix didn't prompt an audit of the *tracer's* edge functions.

**Fix.** A `dCrossRn(a,b,c,d)` helper built on `__fmul_rn`/`__fsub_rn` (with `double`
overloads, `__dmul_rn`/`__dsub_rn`, so the `Real = double` device build is covered too), used
for all three edge functions and for the double fallback. The host (`src/geometry.h`) needs
nothing — MSVC's default `/fp:precise` does not contract and the build sets no `/fp:` or
`/arch:` flag — but a comment now records that dependency and points at the CUDA twin, so a
future `/fp:fast` doesn't reintroduce the bug silently.

**Result.** `cor_gi -gi 0 -ambient 0.1`: dLuma **7.668 → 0.619**, dChroma **3.062 → 0.853**
(FAIL → PASS); the 134 black pixels became **25**, none of them black — they are a
red-wall-vs-white-ceiling tie-break one pixel wide, i.e. the two sharers now disagree about
*which* of them owns the edge rather than both dropping it, which is legitimate and is what the
watertight rule promises. `-gi 32` went dLuma 4.376 → **0.328**. The N3a/N3b/N3d beds and all 14
self-tests pass.

**Cost: none measurable.** A/B on `scraps/n3_gpu.ftsl` 1200×800 `-spp 256 -device gpu`
(a tracer-dominated ~5 s, vs ~1.2 s of fixed startup overhead): contracted 5.213 / 5.009 /
5.007 s, non-contracted 5.075 / 5.122 / 4.954 s — identical within run-to-run noise, even
though the change adds ~3 float ops per triangle test inside the BVH leaf loop. The leaf loop is
memory-bound, not ALU-bound.

**Follow-up worth doing:** audit the *rest* of the device tracer for other places where a
`a*b - c*d` determinant sign has to be consistent between two independent evaluations. The two
found so far (rasterizer edges, tracer edges) were both found by symptom rather than by audit.

### BUG — FIXED (2026-07-30, v0.115.0 → v0.115.1): every `layered` surface renders MONOCHROMATIC in mode `W` at 1 spp

Found while re-rendering `scenes/layered.ftsl` after the `shortpass`/`gaussian` parser fix below.
At `-spp 1` the clearcoated back wall and the iridescent sphere both come out saturated **green**;
at `-spp 64` the wall is its correct mauve (0.55·rgb(0.80,0.25,0.20) + 0.45·rgb(0.20,0.55,0.80))
and the sphere is white. Nothing to do with fluorescence — `-no-fluoro` gives a bit-identical
image, and the wall carries no fluorophore at all.

Cause, `src/backward.h` ~1393, in the hero-bundle loop:

```cpp
if (mp->type == MatType::Layered) {
    // Wavelength-dependent Fresnel coat: de-hero and run the scalar layered
    // handling on the hero channel.
    deHero(); nUp = 1;
```

The de-hero is **unconditional**. This is precisely the failure N1 exists to prevent: mode `W`'s
wavelength lattice is a function of the *sample index alone*, shared by every pixel, so collapsing
a bundle onto its hero λ at 1 spp collapses the whole frame onto the *same* λ — and the median of
a bb6500 CDF is ~550 nm, i.e. green. Glass got the fix (split the bundle at the dispersive vertex);
`layered` never did. The `-spp 64` image is right because 64 samples are 64 different wavelengths.

`README.md` currently understates this as "a *strongly* λ-dependent coat thickness can still tint
at 1 spp". It is not a tint and it is not conditional on the coat: a plain achromatic
`reflectance fresnel  ior 1.5` clearcoat mistints just as hard.

**Proper fix:** don't de-hero at all. The layered handling changes neither direction nor wavelength
when it enters the body — it only computes a scalar coat reflectance `R` and swaps `mp` to a child
material, then falls through to the ordinary per-λ switch. So evaluate `R` per-λ
(`layeredCoatReflectance` already takes λ) and apply it as a per-λ throughput weight, keeping all
`nUp` channels alive. The one genuinely chromatic case is a coat where some λ have `R ≥ 0.5` and
others don't (mode `W`'s dominant-branch rule) or where the stochastic coin would go different ways
per λ; that wants the same **split-at-dispersion** treatment `D_THINFILM`/`D_GRATING` get, not a
de-hero. Note the scalar path (~1226) is already fine — this is purely the hero loop.

#### FIXED in v0.115.1 — a per-λ coat weight, with a fan-out only when the coat is truly chromatic

Done exactly as scoped. `BackwardRenderer::radianceHeroLoop`'s `MatType::Layered` branch now:

1. evaluates `Rl[i] = layeredCoatReflectance(scene, cm, h, ray.d, lam[i])` for every live λ;
2. decides reflect-vs-body **per λ** — `Rl[i] >= 0.5` in mode `W`, `uCoat < Rl[i]` against ONE
   shared coin otherwise;
3. if all live λ agree (the overwhelmingly common case), takes that branch with the whole bundle
   intact. Mode `W` multiplies each channel by its own `Rl[i]` / `1 - Rl[i]` and stops only once
   `maxOf(thr, nUp)` falls under `kWhittedCutoff`, exactly like the `Mirror`/`Filter`/`Glossy`
   case. The stochastic path needs **no reweight at all**: `uCoat` is uniform, so
   `P(uCoat < Rl[i]) == Rl[i]` exactly for each λ — common-random-number analog splitting, where
   the probability *is* the weight, just as in the scalar twin;
4. if they disagree, fans out: each secondary re-enters `radianceHeroLoop` at **this same bounce**
   as its own monochromatic sub-path (`C=1`, `secAlive=false`) and makes its own coat decision,
   landing in its own `L[i]`. Re-entry at `b` rather than `b + 1` is safe because `nUp > 1` implies
   an empty medium stack (every dielectric entry de-heros or splits), so the Beer-Lambert step at
   the loop head was a no-op and cannot be double-applied.

`Renderer::tracePhotonHeroLoop` (`src/render.h` ~2000) got the same shared-coin treatment, since it
carried the identical unconditional `deHero()`. There the disagreement case falls back to
`deHero()` instead of fanning out — a forward sub-path *cannot* re-enter its vertex, because the
loop head has already run the model-C aperture catch and re-entering would deposit the photon into
the film twice. De-hero is still unbiased, and it is what every dispersive material there does
without `-herosplit`, so the fallback costs only variance in the rare chromatic case.

The viewer's `wNeedSpp` list (`src/main.cpp` ~8120) no longer forces 16 passes on any scene
containing a `Layered` material, so a clearcoated scene previews at 1 spp like everything else.

Measured on `scenes/layered.ftsl` at 320×240, against a converged `-spp 1024` reference
(block-mean bar: every 20 px block within 1.5 codes, luma and chroma separately):

| image | max \|dLuma\| | max \|dChroma\| |
|---|---|---|
| **old** mode `W` `-spp 1` | 72.2 | **190.5** |
| **new** mode `W` `-spp 1` | 12.8 | **11.0** |
| old mode `W` `-spp 64` | 1.33 | 2.26 |
| **new** mode `W` `-spp 64` | **0.354** | **0.225** |

So the 1-spp chroma error drops **17×** (and the frame goes from saturated green to the correct
red/green Cornell walls with a mauve back wall), while at 64 spp the new estimator is ~7× *closer*
to the reference than the old one — keeping all 8 channels alive instead of boosting one ×8 is a
straight variance win on top of the correctness fix.

Unbiasedness of the fan-out was checked against the **untouched scalar path** on a deliberately
pathological coat, `scenes/_lay_chroma.ftsl` (a thin-film Airy coat, `film_ior 3.5` over
`ior 1.5`, `film_thickness 200`, whose R oscillates between ~0.06 and ~0.61 across the visible band
and therefore straddles the `R >= 0.5` threshold several times). Bundle `-heroc 8 -spp 1024` vs
scalar `-heroc 1 -spp 2048`: max \|dLuma\| **0.111**, max \|dChroma\| **1.57** codes. The
disagreement branch was confirmed live with a temporary counter — 8192+ hits on that scene, and
32+ even on `scenes/layered.ftsl` (grazing silhouette pixels where Fresnel R crosses 0.5).

Mode `R` (stochastic) on `scenes/layered.ftsl` at `-spp 4096`, old vs new: max \|dLuma\| 0.964,
max \|dChroma\| 1.333 — both inside the bar, so the common-random-number split is unbiased too.
Forward mode `B` on the same scene, 2e9 photons old vs new: max \|dLuma\| **0.070**, max \|dChroma\|
**0.075** codes (PASS), and the energy ledger is identical to six decimals —
`absorbed=0.7105 escaped=0.2895 sum/emitted=0.999996` both ways. That ledger is the real check on
the forward edit, since it replaced the post-de-hero `e.absorbed += beta[0]` bookings with
`activeSum()`. Cost: **1.09×** (1117.3 s vs 1022.9 s back-to-back) — the per-λ
`layeredCoatReflectance` fan (up to 8 Airy evaluations instead of 1) is the whole of it.

Seven non-layered scenes (`cornell`, `multilayer`, `_fluo_cornell`, `_env_cornell`, `_rainbow_test`,
`_spot_cornell`, `_fog_cornell`) are **bit-identical** old vs new at `-mode W -spp 2`; so is
`scenes/layered.ftsl` itself at `-heroc 1` in modes `W`, `R` and forward `B` — the scalar,
bundle-free paths, which this change does not touch (the shared coin is drawn in the same order as
the old single draw, so at `nUp == 1` the new code is the old code verbatim). All 14 physics
self-tests PASS.

### BUG — FIXED (2026-07-30, v0.115.0): `gaussian`/`shortpass` SILENTLY IGNORED positional arguments

`src/ftsl.h` ~1840 parsed these two spectrum heads with

```cpp
for (size_t k = 1; k < w.size(); ++k) {
    std::string key, val;
    if (!splitEq(w[k], key, val)) continue;      // <-- positional args dropped
```

so only the `key=value` form worked. Both forms are documented and both appear in the checked-in
scenes, so `scenes/layered.ftsl`'s

```
absorb shortpass 470 0.2 1.0
emit   gaussian 600 30 1.0
```

parsed as `shortPass(0, 0, 1.0)` — a flat 0.5 absorption at every wavelength — and
`gaussianBand(0, 0, 1.0)`, which is identically **zero** (`sigma = 0` makes `t = ±inf`, so
`exp(-t²/2) = 0`). That gave the material `fluoEmitSampler.integral == 0`, i.e. `haveFluoro ==
false`: the fluorescent body of that scene's iridescent sphere had been inert since it was written,
with no diagnostic.

Fixed by accepting positional args (mixable with keyed ones, which override the slot they name),
**failing** on an unknown key instead of ignoring it, and rejecting `sigma`/`slope` ≤ 0 — which is
never a usable band and is far likelier to be a typo than an intent.

### BUG — FIXED (2026-07-30, v0.113.0 → v0.113.1): a `fluorescent` surface had WILDLY different power on the CPU and the GPU

Found while building N3d-2's A/B bed. On a scene whose *every other pixel is bit-identical*
between the devices, a fluorescent pane's radiance differed by ~5 orders of magnitude. This was
**not** a mode-`W` determinism problem (N3d-2 fixed that; see below) — it reproduced in mode `R`
at 512 spp. The cause turned out to be neither device's reradiation code but the **scene parser**;
see the FIXED write-up at the end of this entry.

**Repro as originally observed** (`scraps/fluo_min_area.ftsl` — deliberately minimal: floor + back wall + ONE dye pane
+ ONE area light, absolute exposure so neither image is auto-anchored):

```
ftrace scraps/fluo_min_area.ftsl -mode W -spp 1 -device cpu -o ppm/fluo_W2_cpu.ppm -window
ftrace scraps/fluo_min_area.ftsl -mode W -spp 1 -device gpu -o ppm/fluo_W2_gpu.ppm -window
python scraps/n3d2_probe.py ppm/fluo_W2_cpu.ppm ppm/fluo_W2_gpu.ppm 20 3
```

| 12 px patch | CPU | GPU |
|---|---|---|
| floor (96,116) | (84.84, 79.94, 80.59) | (84.84, 79.94, 80.59) — **bit-identical** |
| dye pane (100,68) | (255.00, 255.00, 0.00) — **clipped** | (9.01, 7.96, 8.06) — neutral grey |

Backing the tone map out (sRGB transfer ÷ the reported absolute gain, and re-rendering the CPU at
`-exposure 0.02` to un-clip it, where the pane reads (45, 81, 0)):

- **CPU** dye radiance ≈ **8 200 ×** the 0.5-albedo floor's, under the same light. A surface with
  `reflect 0.05` and `yield 0.9` cannot out-radiate a diffuse floor at all, let alone by ~4
  orders of magnitude — the CPU is **the wrong side**, and grossly so.
- **GPU** dye radiance ≈ **0.03 ×** the floor's, and *neutral* in hue. That is exactly what the
  `reflect 0.05` base lobe alone would give on a vertical pane under an overhead light: the
  fluorescent channel contributes essentially **nothing** on the device.

Ratio ≥ 30× on the tone-mapped codes with either light type (`fluo_min_area.ftsl` / `_spot`), and
the same 236.589-code block-luma gap appeared at the *same* block at 1 spp and at 256 spp, which is
what proved it systematic rather than noise.

**Two suspects were written here originally and BOTH were wrong** — recorded so the dead ends
aren't re-walked: (1) `bakeSpec(m.fluoEmit, d.fluoEmitSpec)` + `specLookup` diverging from the
host's continuous `m.fluoEmit(lambda)`; (2) the host's `spdCache` in
`neeLight(scene, h, rhoFluo, invPdfIn, lambdaIn, rng, spdCache)` matching at the wrong λ. Both
sides of the reradiation math are in fact identical; the bug was upstream of both.

#### FIXED (2026-07-30, v0.113.1) — the parser was making every fluorescent surface a LIGHT

Bisecting the material spec (`scraps/fluo_v_bare.ftsl` / `_absorb` / `_emit` / `_diffuse`) showed
the trigger was **the presence of an explicit `emit` statement on a `fluorescent` material**, and
not the fluoro channel at all: `yield 0.0` and `yield 0.9` behaved the same, while deleting the
`emit` line collapsed the CPU/GPU gap to zero.

Cause, in `src/ftsl.h`: the per-type parse for `fluorescent` (~3649) already consumes `emit` as the
**reradiation profile** —

```cpp
m.fluoEmit = spectrumParam(b, "emit", gaussianBand(560.0, 25.0, 1.0));
```

— but the *generic* "any material may carry an emit spectrum" block further down (~3732) then ran
for **every** type, unconditionally:

```cpp
if (find(b, "emit") || find(b, "emit_map")) {
    m.emit = patternedSpectrumParam(b, "emit", "emit_map", m.emitPat, constantSpectrum(0.0));
    m.isLight = true;
}
```

So the same `emit` statement was installed a *second* time as **self-emission**, and the dye pane
became a self-luminous absolute-radiance light of its own emission band. That explains the split
exactly: the CPU tracers honour `m.emit` on hit, so a `gaussian center=560` pane radiated ~8 200 ×
a 0.5-albedo floor (impossible at `yield <= 1`) and in exactly the observed yellow-green hue; the
GPU never uploads a per-material emit spectrum at all (see the separate entry below), so it
rendered only the elastic `reflect 0.05` base — hence "wrong in opposite directions".

The fix makes the generic block skip `Fluorescent`, and *rejects* `emit_map` there rather than
silently dropping it (a reradiation profile is not a surface pattern):

```cpp
if (m.type == MatType::Fluorescent) {
    if (find(b, "emit_map"))
        fail("a fluorescent material's 'emit' is its reradiation spectrum, not surface "
             "emission, so 'emit_map' is not supported here");
} else if (find(b, "emit") || find(b, "emit_map")) { ... }
```

**Verified** on the minimal bed after the fix — the divergence is gone completely:

| 12 px patch | before CPU | before GPU | after CPU | after GPU |
|---|---|---|---|---|
| dye pane | (255.00, 255.00, 0.00) clipped | (9.01, 7.96, 8.06) | (9.01, 7.96, 8.06) | (9.01, 7.96, 8.06) |
| floor | (84.84, 79.94, 80.59) | (84.84, 79.94, 80.59) | (84.84, 79.94, 80.59) | (84.84, 79.94, 80.59) |

`scraps/n3d2_probe.py` now reports **100.000 % bit-identical, |dLuma| = |dChroma| = 0.000** on
that scene. Note this fix does *not* silently change any correct scene: only a `fluorescent`
material with an explicit `emit` was ever affected, and it was unconditionally wrong there.
(In-tree scenes that do change, correctly: `scenes/_fluo_cornell.ftsl` and
`scenes/layered.ftsl`'s fluorescent body — their dyes no longer self-glow.)

The full N3d-2 bed — the one whose dye divergence forced the grating-only bed
`scraps/n3d2_grate.ftsl` to be split out in the first place — now A/Bs clean end-to-end:
`scraps/n3d2_gpu.ftsl` at `-mode W -spp 1` scores **99.627 % bit-identical, max |dLuma| 0.144 /
|dChroma| 0.100**, i.e. the same sliver-only agreement as every other bed. The three standing
regression beds are untouched by the fix (they carry no fluorescent material): n3 99.264 %,
n3b 99.561 %, n3d 99.697 %, and the grating bed still 99.606 % — all re-measured on the 0.113.1
binary.

### BUG — FIXED (2026-07-30, v0.113.1 → v0.114.0): a `fluorescent` material's reradiation channel contributed ~nothing in mode `W`

Surfaced immediately after the parser fix above removed the spurious self-emission that had been
masking it. On `scraps/fluo_min_area.ftsl` (`absorb shortpass edge=480 slope=0.2 amp=1`,
`emit gaussian center=560 sigma=25`, `reflect 0.05`), varying `yield` does nothing whatsoever:

```
yield 0.0 cpu  dye (  9.007,  7.958,  8.062)      yield 0.0 gpu  dye (  9.007,  7.958,  8.062)
yield 0.9 cpu  dye (  9.007,  7.958,  8.062)      yield 0.9 gpu  dye (  9.007,  7.958,  8.062)
```

— bit-identical, on **both** devices, and equal to what a plain `type diffuse reflect 0.05` pane
reads. So the whole bispectral term is worth ≈0.00–0.02 codes out of 9.

That is very unlikely to be right. A bb6500 illuminant puts roughly 30 % of its 360–830 nm power
below the 480 nm absorption edge, so at `yield 0.9` the *effective reradiation albedo* should be on
the order of 0.27 — about **5×** the elastic 0.05 lobe, i.e. the dye should dominate its own
appearance, not vanish into it.

**Ruled out:** the λ_in NEE weight. `scene.invPdfLambda(λ) = emitG / g(λ)` with
`emitG = emitSampler.integral` (`src/scene.h` ~1351), which is exactly `1/pin × emitG` for the
`pin` that `emitSampler.sample`/`sampleAt` returns — so `invPdfIn = scene.invPdfLambda(lambdaIn)`
is consistent and was never the cancelling factor. `-checkfluoro` also passes, so the reradiation
primitives (`fluoEmitSampler`, `fluoroWeights`, `fluoroInteract`) are all sound.

#### FIXED (2026-07-30, v0.114.0) — the (sIdx, bounce) lattice used bases far larger than `-spp`

Bisecting the *material* rather than the code found it: a 2×2×2 sweep of
{`absorb shortpass edge=480` | `absorb 1.0`} × {`emit gaussian 560/25` | wide} × {`yield 0.9` | `0.0`}
showed fluorescence working perfectly with a **flat** absorption (dye = (78.9, 134.0, 0.0), a
vivid green ≈33× the elastic lobe, exactly as the physics predicts) and contributing **exactly
zero** with the shortpass edge. So the excitation λ_in was simply never landing below 480 nm.

Cause: `whittedFluoroU` (`src/backward.h`) drew its coordinate from a **base-61** radical inverse.
A plain radical inverse in base *b* returns exactly `i/b` for `i < b`, so its first *N* points
cover only the prefix `[0, N/b)` — well distributed *within* that prefix and blind to the rest.
With `rot05` on top, u was pinned to `[0.5, 0.5 + spp/61)`: the **long** half of the illuminant
CDF, for every preview budget under 61 spp. Measured, the dye switched on in one step:

| `-spp` | 1 | 4 | 16 | **64** | 256 |
|---|---|---|---|---|---|
| dye, unscrambled | 10.6, 9.5, 9.6 | 10.4, 9.6, 10.1 | 10.3, 9.8, 10.2 | **53.4, 73.7, 0.0** | 53.9, 73.5, 0.0 |
| dye, scrambled | 10.6, 9.5, 9.6 | 43.4, 67.7, 0.0 | 48.6, 67.5, 0.0 | 54.0, 75.7, 0.0 | 54.9, 75.4, 0.0 |

This was **not** specific to fluorescence — it hit every lattice with a base above the sample
count: `whittedGlossyDir`'s bases 13–41 kept a rough lobe hugging its mirror direction until
`-spp 13`, `whittedOrderU`'s 43–59 made a grating's higher orders arrive in a lump at `-spp 43`,
and even the `-gi` gather's bases 7/11 confined its two coordinates to one corner.

Fix: **digit-scramble** the radical inverse (Faure's standard fix for high-dimensional Halton) —
`r = Σ π(dₖ)·b^-(k+1)` with `π(d) = (d·m) mod b`, `m = round(b/φ)`. Any bijection π leaves the
sequence a permutation of the same *b*-point grid (so the discrepancy is asymptotically
unchanged) but visits it scattered instead of monotone. The multiplicative form needs no
permutation tables, so the CUDA twin is trivially bit-identical, and crucially **π(0) = 0**, so
`radicalInverseScr(b, 0) == 0` in every base and every "sample 0 is the canonical outcome"
contract (mirror direction / specular order `m = 0` / median λ) survives. Measured star
discrepancy of the first *N* points (`scraps/n3e_lattice.py`):

| base (role) | N=4 plain → scr | N=16 plain → scr |
|---|---|---|
| 13 (glossy u1) | 0.769 → 0.269 | 0.215 → 0.130 |
| 43 (grating order) | 0.930 → 0.250 | 0.651 → 0.102 |
| 61 (fluoro λ_in) | 0.951 → 0.254 | 0.754 → 0.077 |

**Verified:** every `-spp 1` image is **bit-identical** to v0.113.1 (the π(0) = 0 anchor — checked
with `cmp` on all four A/B beds plus the fluorescence bed), and CPU↔GPU still agree at both 1 and
8 spp on all four beds (max |dLuma| ≤ 0.253 at 1 spp, ≤ 0.139 at 8 spp — every bed *tighter* at 8
spp than at 1). `png/n3e_montage.png` is the visual proof: the dye pane is black at 1/4/16 spp and
green only at 64 in the unscrambled column, green from 4 spp on in the scrambled one, with the two
columns converging (near-black diff) by 64.

### DEBT — FIXED (2026-07-30, v0.114.0 → v0.115.0): λ_in for fluorescence is drawn from the ILLUMINANT, not from the dye's absorption band

The residue of the bug above, and the reason a narrow-band dye still previewed as a bare elastic
lobe at exactly `-spp 1`. `MatType::Fluorescent` sampled its Stokes-shift excitation wavelength
from `scene.emitSampler` — the scene-wide illuminant CDF — and mode `W`'s 1-spp coordinate is that
CDF's **median** (~575 nm under bb6500). A dye absorbing only below 480 nm could therefore never be
excited by the single canonical sample, however good the lattice was.

It was also a plain variance problem in the stochastic modes: with `absorb shortpass edge=480`
under a broadband lamp most λ_in draws land where `fluoAbsorb ≈ 0` and contribute nothing, while
the few that land in the band carry a correspondingly large weight.

#### FIXED (2026-07-30, v0.115.0) — a per-material excitation CDF (absorb × illuminant)

`Material` grew an `EmissionSampler fluoInSampler` (`src/scene.h` ~205), built inside
`Scene::finalizeEmitters()` right after `emitSampler` from the product
`clamp01(fluoAbsorb(λ)) · g(λ)`, where `g(λ) = Σ_k geomWeight_k · spd_k(λ)` is the same combined
illuminant `emitSampler` uses. Built there (not at parse time) because it needs the finished
emitter list, and rebuilt on every `finalizeEmitters()` so `-ignoreenv` — which drops an emitter
and re-finalises — stays consistent.

`src/backward.h`'s `MatType::Fluorescent` now draws from it and sets `invPdfIn = 1.0 / pin`. That
second half matters on its own: the old code paired a *bin-discretised* CDF draw with the
*analytic* `scene.invPdfLambda(λ)` = `emitG / g(λ)`, which is only approximately the reciprocal of
the pdf the draw actually had. A material whose product integral is 0 (a dye this illuminant
cannot excite at all) falls back to `scene.emitSampler`, so the branch still terminates.

Device twin: `DMaterial` gained `fluoInCdfOffset` / `fluoInCdfN` / `fluoInCdfStep`, a second slice
appended to the existing flat `DScene::fluoCdfAll` buffer, plus `dSampleFluoInU()` next to
`dSampleSceneLambdaU()`. Only one call site needed changing — the second `D_FLUORESCENT` label in
`render_cuda.cu` is the split-at-dispersion dispatch, which re-enters `bkInteract`.

Unbiasedness is not a matter of taste here and is now asserted: `-checkfluoro` grew a fourth check
that estimates the reradiation NEE weight `Q·∫aEff(λ)·spd(λ)dλ` **both ways** — from the illuminant
CDF (the old sampler) and from absorb × illuminant (the new one) — and requires both within 2 % of
a fine analytic quadrature. Measured: analytic `4.7061e15`, illuminant-sampled `4.7030e15`,
product-sampled `4.7062e15`. It also reports the variance ratio and the mode-`W` median draw
(421.7 nm, `aEff` = 0.825 there, i.e. squarely inside the band). The reported variance ratio is a
best case — this synthetic integrand *is* the new sampler's target, so its estimator is constant
and only CDF discretisation is left; a real render carries the NEE geometry factor too.

The dye pane of `scraps/fluo_x_sp_narrow_09.ftsl` (`absorb shortpass edge=480`, `yield 0.9`),
12 × 20 px patch, mode `W` on the CPU:

| `-spp` | v0.113.x | v0.114.0 | v0.115.0 |
|---|---|---|---|
| 1 | 10.6 (bare elastic) | 10.6 (bare elastic) | **(47.6, 81.2, 0)** |
| 2 | 10.6 | (59.9, 93.5, 0) | (48.8, 79.6, 0) |
| 4 | 10.6 | 43.4 | (57.1, 77.7, 0) |
| 16 | 10.6 | 48.6 | (58.0, 78.7, 0) |
| 64 | (53.4, 73.7, 0) | 54.0 | (57.4, 78.4, 0) |
| 256 | — | 54.9 | (57.5, 78.5, 0) |
| 4096 | — | — | (57.5, 78.4, 0) |

`yield 0.0` vs `yield 0.9` now differ at `-spp 1` (they were bit-identical up to `-spp 64` in
v0.113.x — the original symptom). CPU↔GPU at 1 spp still PASSes the block-mean bar on every bed:
n3 max \|dLuma\| 0.253, n3b 0.144, n3d 0.144, n3d2 0.144, grate 0.144, and the four dye beds
0.044 / 0.003 / 0.065 / 0.002. All nine physics self-tests PASS.

### DEBT — OPEN (2026-07-30, v0.113.1): the GPU cannot do material emission-on-hit at all

`src/render_cuda.cu` ~608 notes plainly that *"DMaterial carries no emit spectrum"*. Only meshes
get registered as emitters (via `addMesh`), so any **non-mesh primitive** (a `quad`, `sphere`,
`box`, …) bound to a material with an `emit` spectrum is a visible light on the CPU and a black /
elastic-only surface on the GPU. This is a general CPU-vs-GPU divergence, not specific to
fluorescence — it is simply how the fluorescence bug above became visible as a *device* split.

Proper fix: bake a per-material emit spectrum into `DMaterial` (same `bakeSpec` treatment the other
spectra get) and honour it on hit in `bkInteract`, plus register emissive non-mesh primitives with
the device light list so NEE can see them too. Until then, emissive non-mesh geometry should
either be documented as CPU-only or rejected at upload time with a clear message rather than
silently rendering differently.

### DEBT — OPEN (2026-07-30, v0.113.1): `Fluorescent`'s `neeLight` calls omit the `gi` argument

`src/backward.h` ~927 and ~953 call `neeLight(scene, h, rho…, invPdf…, lambda…, rng, spdCache)`
without the trailing `gi` that the `Diffuse` (~1019) and `DiffuseTransmit` (~986/990) cases pass.
`neeLight` uses `gi` to pick the shadow-ray stratification grid, so at a `-gi` gather vertex a
fluorescent surface pays `lightGrid²` shadow rays instead of `giGrid²` — wasted work and a
different (needlessly fine) stratification than its neighbours. Fix is a one-word addition to
both call sites, but it changes the sample pattern, so it needs an A/B re-baseline of the
fluorescence beds and should ride along with the reradiation investigation above.

### BUG — FIXED (2026-07-29 → 2026-07-30, v0.111.0 → v0.113.0): mode `W` was still STOCHASTIC at grating / fluorescent vertices

*(All four materials are now fixed: thin-film and multilayer in **v0.112.0** (N3d-1) and grating
and fluorescent in **v0.113.0** (N3d-2). Both fix write-ups are below.)*

Mode `W`'s entire contract is "no rng draws — the same image at `-spp 1` every time, on any
device". Two material vertices still broke it, on **both** the CPU and the GPU:

- **`gratingDiffract`** (`src/render.h` ~2393) picked a diffraction order from the rng:
  `double xi = rng.uniform() * wsum;` over the propagating orders weighted `1/(1+|m|)`.
- **`MatType::Fluorescent`** (`backward.h` ~896) drew `scene.emitSampler.sample(rng, pin)` for
  the Stokes-shift excitation wavelength λ_in. *(Its continuation coin at ~914 was **not** a
  problem: mode `W` implies `directOnly`, which returns before reaching it.)*

**Neither is a dominant-branch problem, so the `whittedWeight` trick did not apply.** Both are
*discrete choices from a distribution*, which is exactly what N2 already solved for the glossy
lobe: keep ONE choice per sample but drive it from a low-discrepancy sequence indexed by
`(absolute sample index, bounce)` instead of the rng. `GiCtx` already carried `sIdx` and `bounce`
and was already threaded into `interactMaterial` / `bkInteract` on both devices, so the plumbing
existed.

**How the whole family was found:** building N3b's deterministic CPU/GPU A/B bed
(`scraps/n3b_gpu.ftsl`). `scraps/n3b_check.py` failed at max |dLuma| **4.078** / |dChroma|
**6.995** codes on exactly the 20 px blocks covering a `thinfilm` bubble, while every other block
in the frame passed under 0.2 codes. That was not a porting bug — the CPU render was equally noisy
there; CPU and GPU simply draw from independent rng streams, so the two were different
realizations of an estimator that should not have been stochastic at all.

#### FIXED (2026-07-30, v0.112.0): the thin-film / multilayer reflect-or-transmit coin

`src/render.h::thinFilmInterface` and `multilayerInterface` flipped a bare coin with **no
`whitted` branch at all** — the opaque metal-backed path did `if (rng.uniform() >= R) return
false;` and the lossless path `if (rng.uniform() < R) outDir = reflect(...) else refract(...)`.
Both now take the same `double* whittedWeight` out-param contract `refractOrReflect` already had:

- **Lossless substrate** (two real branches): reflect iff `R >= 0.5`, report `R` or `1-R`. TIR
  reports 1.0 (one branch only).
- **Opaque / absorbing substrate** (transmission is absorbed, so there is only one *surviving*
  branch): always reflect and report `R` — the reflectance becomes a throughput weight instead of
  a survival probability, exactly as `Mirror` and `Filter` already do in mode `W`.

Five places moved together: both functions in `src/render.h`, both device twins in
`src/render_cuda.cu`, and the `MatType::ThinFilm`/`Multilayer` + `D_THINFILM`/`D_MULTILAYER` call
sites in `src/backward.h` and `bkInteract`, which fold the weight in via
`whittedAttenuate`/`dWhittedAttenuate`. Every non-mode-W caller (forward `A`/`B`/`C`, `M`/`S`,
BDPT, VCM, the wavefront kernel) passes `nullptr` and is bit-identical.

Measured on `scraps/n3d_gpu.ftsl` — the N3b box with a lossless thin film (`bubble`), an
absorbing thin film (`beetle`), a lossless multilayer (`dichroic`) and an absorbing multilayer
(`morphoish`), so all four code paths are in one frame — at 400×260 `-spp 1`, absolute exposure,
CPU vs GPU via `scraps/n3b_check.py`:

| | bit-identical | max \|dLuma\| | max \|dChroma\| | |
|---|---|---|---|---|
| before (v0.111.0) | 91.780 % | 6.678 | 5.820 | **FAIL** |
| after (v0.112.0) | **99.697 %** | **0.144** | **0.078** | **PASS** |

That is inside the same 1.5-code-per-20 px-block bar the non-iridescent N3b bed passes at.
`scraps/n3d_montage.py` draws the before/after proof picture (`png/n3d_montage.png`): the
amplified CPU−GPU difference lights up on exactly the four iridescent spheres before the fix and
is black after it. The N3b and N3a A/B beds were re-run and are unchanged (99.561 % / 0.144 /
0.105 and 99.264 % / 0.253 / 0.547 — *corrected 2026-07-30: this line originally read
"99.394 % / 0.131 / 0.190" for the N3a bed, which does not reproduce and disagrees with the
99.264 % TODO §N3a records; `scraps/n3_gpu.ftsl` contains only diffuse/mirror/filter/glossy, so
neither N3d-1 nor N3d-2 can move it and 99.264 % has been its value since N3a*), and the
stochastic path was checked to be unbiased as well
as untouched: forward `-mode C` on this scene conserves energy (sum/emitted 1.000001) and a
mode-`R` CPU↔GPU pair converges as √spp from 32 to 1024 spp (mean \|diff\| 24.95 → 5.25,
\|dLuma\| 5.83 → 1.47, \|dChroma\| 12.49 → 2.17, all ≈ √32; frame means within 0.06 %).

#### FIXED (2026-07-30, v0.113.0): the grating diffraction-order pick and the fluorescent λ_in

Both were solved the way the entry above predicted — N2's trick, not N3d-1's. Neither pick is a
dominant *branch*; each is a **discrete draw from a distribution**, and each is already **analog**
(candidate `i` with probability `w_i/Σw`, throughput unchanged), so replacing the random `u` with a
stratified one off the `(sIdx, bounce)` lattice is a pure **variance** fix — unbiased, same
estimator, only the per-pixel luck removed.

Two new lattice helpers, `BackwardRenderer::whittedOrderU` / `whittedFluoroU` (`src/backward.h`)
with bit-identical device twins `dWhittedOrderU` / `dWhittedFluoroU` (`src/render_cuda.cu`),
carrying fresh prime bases (43/47/53/59 and 61/67/71/73) so they don't correlate with
`whittedGlossyDir`'s 13/17…37/41. They differ deliberately in one respect:

- **`whittedOrderU` is NOT rot05'd**, because `gratingDiffract`'s whitted path walks its
  candidates in **descending efficiency** (`0, −1, +1, −2, +2, …`) rather than the stochastic
  path's `mm = −M..+M`. `radicalInverseB` returns 0 at `sIdx` 0 in every base, so sample 0 lands
  on the **specular order m = 0** — the exact analogue of "1 spp collapses a glossy lobe to the
  mirror direction", with extra spp fanning the spectrum out into the higher orders. (Without the
  re-ordering, `u = 0` would have picked the most *negative* — most strongly dispersed — order as
  the `-spp 1` look.) Total mass is the same `Σw` either way, so the two traversals agree in
  distribution; the stochastic path keeps its ascending walk and stays bit-identical.
- **`whittedFluoroU` IS rot05'd**, because no excitation wavelength is privileged the way `m = 0`
  is: rotating by ½ puts sample 0 at the **median** of the excitation CDF (the most representative
  single λ_in) instead of at its short-λ extreme.

Plumbed as `gratingDiffract(..., const double* whittedU = nullptr)` (host `src/render.h`, device
`src/render_cuda.cu`) and `scene.emitSampler.sampleAt(u, pin)` / `dSampleSceneLambdaU(sc, u, pin)`
at the `Fluorescent` / `D_FLUORESCENT` case. Every non-whitted caller passes `nullptr` (host
render.h ~1330; device 4591 / 7650 / 9044) and is bit-identical.

**One port subtlety worth remembering:** the device whitted branch computes the CDF walk in
**`double`**, and recomputes each `1/(1+|m|)` from the order instead of reading the `Real wgt[]`
the stochastic path builds. The weights are small exact rationals, so accumulating them in double
in the same sequence the host uses makes the *selection* bit-identical. That matters far more here
than elsewhere in the port: an fp32 tie-break near a cumulative boundary would send the ray into a
**neighbouring diffraction order** — a structural CPU/GPU difference, not the usual silhouette
sliver.

Measured on `scraps/n3d2_grate.ftsl` — four gratings (three groove spacings + one rotated 90° to
check the dispersion axis comes off `groove_dir` consistently), a flat grating pane (a flat surface
holds one incidence angle across a wide area, so an order flip is far more visible than on a
sphere), high-contrast bars behind everything so a wrong order shows as a *displaced bar* rather
than a flat tint, plus the N3d-1/N3b carry-overs. 400×260 `-spp 1`, absolute exposure:

| test | result |
|---|---|
| `n3b_check.py` CPU vs GPU, `-spp 1` | 99.606 % bit-identical, 99.932 % within 1 code, max \|dLuma\| **0.144** / \|dChroma\| **0.103** — **PASS** |
| strict `n3_check.py`, same pair | 36 hot pixels, **0 blob interior** — passes even the N3a sliver bar |
| same pair at `-spp 64` | 99.311 % bit-identical, \|dLuma\| **0.050** / \|dChroma\| **0.093** — **PASS** (*tighter* than 1 spp, as averaging over orders should be) |
| `-spp 1` vs `-spp 64` on one device | \|dLuma\| **68.279** codes — the higher orders really do fan out; 1 spp is the undiffracted preview, not the converged image (`png/n3d2_spp_fanout.png`) |
| forward `-mode C`, 20 M photons | sum/emitted **0.999998** (CPU) / **0.999970** (GPU) — stochastic path untouched and still conserving |
| N3d / N3b / N3a regression beds | 99.697 % / 99.561 % / 99.264 % — **unchanged to the digit** |

`scraps/n3d2_montage.py` draws the before/after proof (`png/n3d2_montage.png`, from the *full*
`scraps/n3d2_gpu.ftsl` bed which keeps the dye): pre-fix, the four grating spheres and the flat
pane are rainbow salt-and-pepper and the amplified CPU−GPU panel lights up on exactly them;
post-fix they are smooth and that panel goes black over them. The fluorophore's own per-device
speckle is gone too (`png/fluo_W2_ab.png`: the dye pane is a perfectly flat region at `-spp 1` on
both devices) — but its cross-device *power* still diverges wildly, which is the **separate** open
BUG at the top of this file, and is why the cross-device A/B is run on the dye-free
`n3d2_grate.ftsl`.

### DEBT (2026-07-29, v0.107.0): mode `W` picks a dielectric's dominant branch, it does not fork

*(Supersedes the v0.105.0 "still samples dielectrics stochastically" entry and the v0.106.0
"opaque bright blob" BUG — both **FIXED**, see below.)*

`refractOrReflect` (`src/render.h`) now takes a `whittedWeight` out-param: mode `W` takes
the **dominant** Fresnel branch (reflect iff R ≥ 0.5) and folds that branch's weight into
the path throughput via `whittedAttenuate`, the same trade `MatType::HalfMirror` /
`Layered` / `Mix` already make. That removed the coin flip, which was the real bug.

What remains is the inherent limit of "dominant only": a dielectric is the one place where
forking *both* branches genuinely matters (a window shows a reflection **and** what is
behind it at once), so near the Brewster/grazing crossover one of them is visibly dropped.
The honest version forks up to a small depth budget, POV-Ray-style, pruned by
`kWhittedCutoff` — which prunes hard, since a normal-incidence reflection weighs ~0.04 and
its second bounce ~0.0016 is already under the 1/512 cutoff, so the fork tree stays shallow
in practice. That needs `radiance`/`radianceHero` to grow a recursive branch, which they
currently avoid (both are iterative single-path loops).

#### FIXED (2026-07-29, v0.107.0): the stochastic coin flip

`-mode W` replaced every other Monte-Carlo estimator on the backward walk with a fixed
quadrature, but `refractOrReflect` was left alone and still drew `rng.uniform()` to choose
reflect vs refract. So glass was the one thing in a "noise-free" mode that was noisy — and
at `-spp 1`, where the mode is meant to be used, a single coin flip per pixel is not noise
but **salt-and-pepper**: `ftrace -in scenes/cornell.ftsl -mode W -spp 1` rendered the SF10
ball as a speckled blob. Fixed by the dominant-branch selection above. (The roughness
perturbation, also stochastic, is skipped in mode `W` for the same reason — consistent with
mode `W` taking the mirror direction for glossy lobes.)

#### FIXED (2026-07-29, v0.107.0): "opaque bright blob" — the original diagnosis was wrong

The v0.106.0 BUG entry claimed the sphere had "no lens structure whatsoever" and did **not**
improve with `-spp`, "converging to the wrong answer". Both claims were artifacts of the
metric it used. It scored sphere-centre *saturation max−min* (9.8 at `-spp 1`, 4.7 at 16,
4.4 at 128) and read the *fall* as bias converging; that fall was simply the **coin-flip
noise averaging out**. Measured properly against a 2504-spp mode-`R` reference at 640×400
(`scraps/wprev_ref.png`), mode `W` at `-spp 16` is now:

| metric | ref (mode R) | W `-spp 1` | W `-spp 16` |
|---|---|---|---|
| sphere / lit-wall luminance ratio | 2.19 | 3.50 | **2.04** |
| structure inside the ball (std) | 14.2 | 12.2 | **11.5** |
| hue, sphere R−G | +8.7 | **−163.6** | +33.3 |

Structure 11.5 vs the reference's 14.2 is *not* an opaque blob — the refracted lens image is
there — and the brightness ratio is within 7% of the reference, so the "~2× too bright"
reading was the direct-only wall being dark, not the glass being bright.

The one real residue is **hue**, and it is the wavelength issue below, not the interface.

### FIXED (2026-07-29, v0.108.0): mode `W` renders dispersive materials at ONE wavelength per sample

**Fixed by split-at-dispersion in the backward tracer** (`BackwardRenderer::heroSplit`,
`radianceHeroLoop` in `src/backward.h`; mode `W` enables it unconditionally, mode `R` keeps it
opt-in via `-herosplit`). At a dispersive vertex the bundle now fans into C monochromatic
sub-paths, each refracting along its own Snell direction and accumulating into **its own**
`L[i]` slot, instead of terminating the secondaries and boosting the hero ×C.

Measured on `scenes/cornell` (glass sphere region, chroma = each channel's fraction of the
total, so brightness is normalised out and only the *tint* is compared, against a 4096-spp
`-direct-only` mode-R reference):

| estimator | chroma error | render (960×600) |
|---|---|---|
| de-hero, 1 spp (old) | **36.67 pp** — the flat-green collapse | 0.5 s |
| de-hero, 16 spp (the old `wNeedSpp` workaround) | 4.20 pp | 7.1 s |
| **split, 1 spp (new)** | **0.80 pp** | **0.9 s** |

So the fix is **5× more accurate than the 16-pass workaround at 7.9× less cost**, and the
workaround is gone (`wNeedSpp` no longer lists the dispersive materials). The split costs ~1.8×
a single de-hero pass on this scene (C× traversal *past* the glass only) and is free on a scene
with no dispersive material — `_room_of_gyroids_f12` mode W is bit-identical to v0.107.0.

Verified **unbiased**, not just prettier: de-hero and split are two estimators of one integral,
and on `scenes/absolute.ftsl` (fixed gain 6, so no per-image auto-exposure) at 2048 spp they
agree to **+0.33 / +0.28 / +0.52 %** per channel against a ~2.2 % noise floor. *Do this
comparison in absolute mode:* the same run tone-mapped with per-image auto-exposure reads a
spurious "+5.4 % bias", because the split resolves the dispersive caustic geometrically and
moves the p99 anchor by ~1.8× — the identical trap the mode-U validation notes in `TODO.md`.

Still de-heroes (so still needs multiple passes in mode `W`): the scalar, bundle-free path taken
for participating media / GRIN / `-heroc 1`. `wNeedSpp` now tests for exactly those.
(**`Layered`** used to be on this list — its coat Fresnel is a λ-dependent *decision* rather than a
λ-dependent direction, so the dispersion split did not apply. v0.115.1 solved it differently, by
applying the coat reflectance as a per-λ *weight* with the bundle intact and fanning out only when
the decision itself differs across λ; see the layered de-hero entry above.)

<details><summary>Original entry (kept for the diagnosis, which the fix is built on)</summary>

The `R−G = −163.6` above: at `-spp 1` a glass ball comes out violently green. A dielectric
(or thin film / multilayer / grating / half-mirror / fluorescence) **de-heroes** the path —
`radianceHero` terminates the secondary wavelengths and continues the hero channel alone,
because the interface refracts each wavelength in a different direction. Mode `W`'s
wavelength lattice `whittedLambdaU(sIdx)` is a function of the **sample index alone**, which
is precisely what makes the mode noise-free (every pixel agrees), so at `-spp 1` the *entire
frame* de-heroes onto *one* wavelength and every dispersive object is tinted by it. It is
under-sampling, not bias: `R−G` falls −163.6 → +33.3 by `-spp 16` (reference +8.7).

Mitigated, not fixed: the README documents `-spp 8`–`16` for glass, and the interactive
viewer's mode-`W` preview auto-detects a dispersive material in the scene and keeps adding
passes to 16 spp (`wNeedSpp` in `src/main.cpp`), stopping at 1 spp otherwise since a
bundle-only scene is already exact.

*Proper fix:* split the bundle at a de-hero vertex instead of collapsing it — run the scalar
continuation once per hero wavelength and accumulate into `L[i]`, so one sample still covers
`heroC` wavelengths through glass. Costs up to `heroC`× at dielectric vertices only.
`hero::gSplit` (`-herosplit`) already does something structurally similar on the CPU forward
tracer and is the place to look for prior art. Note the comment above the scalar λ draw in
`renderRows` still says the hero path "gets C=heroC of them per sample for free" — true only
until the path de-heroes, which is exactly this case.

*Why it matters beyond glass:* it silently invalidated a GI measurement. The first attempt
to evaluate `-gi` on `scenes/cornell` compared mode-`W` frames against a mode-`R`
reference where the mistinted sphere was the single largest error in the frame, swamping the
interreflection signal the sweep was trying to measure. `scraps/cor_gi.ftsl` (all-diffuse,
no dielectric, no glossy) exists specifically to dodge this and the glossy entry below.

</details>

### FIXED (2026-07-30, v0.109.0): mode `W` over-sharpens rough glossy metal

**Fixed by driving the lobe off a deterministic lattice** (`whittedGlossyDir` in
`src/backward.h`, feeding the new `glossyDirUV` in `src/render.h`). A mode-`W` rough-specular
vertex now takes point `sIdx` of a 2-D radical-inverse lattice on the power-cosine lobe
instead of the lobe's single mirror direction, at all four sites that used to collapse it
(`Glossy` in `interactMaterial`, `Glossy` in the hero loop's achromatic-delta case, and both
`Layered` coat branches).

The real defect was not aesthetic, it was **consistency**: every sample took the *identical*
direction, so mode `W` did not converge on the true lobe at *any* budget. Measured on
`scraps/n2_rough.ftsl` (three gold balls at roughness 0.045 / 0.15 / 0.35 reflecting
high-contrast bars; mean |err| inside each ball vs a converged direct-only mode-`R`
reference, absolute exposure so no per-image anchor can move):

| spp | rough 0.045 | rough 0.15 | rough 0.35 |
|---|---|---|---|
| 1 | 5.71 → **5.71** | 15.66 → **15.66** | 33.40 → **33.40** |
| 16 | 3.61 → **2.56** | 13.65 → **6.81** | 31.60 → **11.80** |
| 64 | 3.51 → **0.97** | 13.62 → **2.45** | 31.49 → **4.16** |
| 256 | 3.49 → **0.53** | 13.61 → **1.04** | 31.49 → **1.67** |

Read the *old* column down: 33.40 → 31.49 over 256× the budget, a 6 % improvement that is
purely edge antialiasing. The lobe error never moved. The new column converges (**19×** lower
at 256 spp on the roughest ball), and `png/n2_old_256.png` vs `png/n2_new_256.png` shows it:
three indistinguishable mirror balls become a proper satin gradient matching `png/n2_ref.png`.

Two properties are load-bearing and were verified, not assumed:

* **`-spp 1` is bit-identical to v0.108.0**, including on a scene that actually contains
  glossy material. `glossyDirUV` maps `u1 == 1` to the mirror direction and
  `radicalInverse(0) == 0` in every base, so the polar sequence is *complemented*
  (`1 - radicalInverse`) rather than `rot05`-rotated, and sample 0 lands exactly where the old
  code put it. The `u1 >= 1.0` early-out returns `mdir` verbatim so `normalize()`'s last-bit
  rescale cannot spoil that.
* **Every stochastic path is bit-identical** (`-device cpu`, modes R/B/C/M/S/D/U), since
  `sampleGlossy` was only re-expressed in terms of `glossyDirUV` with its two draws sequenced
  into locals. *Compare with `-device cpu` on both binaries* — a CUDA build silently
  auto-selects the GPU, which reads as a spurious whole-frame "regression".

Each bounce depth takes its own prime pair (13/17, 19/23, 29/31, 37/41) so two glossy
vertices on one path are not driven by the same 1-D sequence. The path is **not** forked, so
there is no N^depth blowup in a gyroid labyrinth, and cost at equal spp is unchanged.

Still outstanding: because the lobe is resolved *across* samples rather than within one,
rough metal is the one thing in mode `W` that genuinely wants `-spp` > 1 — it is no longer
*wrong* at 1 spp, just as sharp as it always was. `wNeedSpp` does not test for it (a glossy
scene previews fine; it is only a rough one that benefits), so the interactive viewer stops
at one pass on rough metal.

<details><summary>Original entry (kept for the diagnosis, which the fix is built on)</summary>

`interactMaterial` sends a mode-`W` glossy vertex along the exact mirror direction,
weighted by the lobe's reflectance. That is near-exact for the tight lobes this engine's
metals usually use (the gold in `gold_gyroids` is roughness 0.045 and reads essentially
identically to full GI) and it is why the mode converges at 1 spp — but a genuinely rough
metal (roughness ≳ 0.2) previews crisper than it renders.

*Proper fix:* a small fixed lattice of lobe directions — the same trick `-whitted-grid`
plays for area lights. N deterministic offsets around the mirror direction, weighted by
the lobe, with N scaled off the roughness so smooth metals stay at one ray. Cost is
linear in N, and only on specular chains. Do **not** fork the path (that grows as
N^depth inside a labyrinth like a gyroid); drive the single lobe direction from the
low-discrepancy sequence indexed by (absolute sample index, bounce), *without* the
`rot05` half-offset on the polar coordinate, so sample 0 is exactly today's mirror
direction and higher `-spp` progressively widens the lobe.

*Upgraded 2026-07-29, v0.106.0 — this is now measured to be the LARGER error of the two,
at least on rough gold.* The claim above that roughness 0.045 gold "reads essentially
identically to full GI" is too generous. Evaluating the new `-gi` gather on
`gold_gyroids` (420², vs a converged mode-`R` reference) showed the gather buying only a
6 % whole-frame improvement for 4.5× the cost:

| mode W variant | crevices (darkest 10 %) | whole-frame mean \|err\| |
|---|---|---|
| no ambient | 25.4 (signed −21.3) | 24.7 |
| `-ambient 0.05` | 13.8 (signed −8.2) | 13.3 |
| `-ambient 0.05 -gi 32` | 13.6 (signed −7.6) | 12.5 |
| `-ambient 0.05 -gi 64` | 13.2 (signed −7.3) | 12.1 |

The difference image (`scraps/gi_diff.png`) shows the gather *is* depositing genuine
bounce light on the floor and walls, but the dominant residual sits **on the gold lattice
itself** — i.e. on the one material whose lobe mode `W` collapses to a mirror. A single
mirror direction cannot spread light into a labyrinth of crevices, so no amount of
diffuse GI fixes that scene. Fixing this entry is therefore the higher-value work.

</details>

### BUG — DONE (2026-07-29, v0.102.1): `-exposure`/`-ev` was silently ignored by `-topng`

`ftrace -topng in.ftbuf out.png -ev 3` produced a file byte-identical to the one
without `-ev`: the flag parsed as a no-op and nothing warned. Cause: `-topng` is
dispatched from `main()` at `argv[1]` *before* the main argument loop runs (it is
deliberately a pure, scene-free utility path), and `exposureCli` is a local of that
loop — so nothing on the conversion path ever saw the flag. `convertToPng()` called
`writeFilm(out, f, N)` and took the `expComp = 0.0` default.

This mattered because re-developing a checkpoint is exactly when you want an exposure
override. A `.ftbuf` holds *linear* film, and the p99 auto-exposure anchors on the
brightest 1% — so any scene with a small, very bright source (an arc lamp, a filament)
develops with everything else crushed. `scenes/mirror_sphere_interior.ftsl` lands at a
median of 33/255 for that reason. Without the flag the only way to re-develop brighter
was to re-render, which for that scene is 30 minutes for an image already sitting on
disk.

Fixed by scanning `argv[4..]` for `-exposure`/`-ev` inside the `-topng` branch and
threading it into `convertToPng(in, out, expComp)` → `writeFilm(..., expComp)`. Verified:
`exposure=4.67e-13 (auto 1.56e-13 x 3EV-comp)`, median 33 → 60. Because a `.ppm` is
already 8-bit sRGB there is nothing to re-expose, so that input now prints an explicit
`warning: -ev ignored for a .ppm input` rather than pretending to work — and its output
is confirmed byte-identical with and without the flag.

Note the flag stays a *multiplier*, not stops (`-ev 3` is 3x, not 3 EV), consistent with
`-ev` everywhere else in the CLI. That naming is itself mildly misleading but is not
worth a breaking rename.

### BUG — DONE (2026-07-29, v0.102.0): `-max-bounce` was silently ignored by mode D (BDPT) and mode U (VCM)

`-max-bounce N` parsed fine and was honoured by the unidirectional/forward paths, but the
bidirectional modes never read it: mode D hard-coded `int maxDepth = 8;` in `main.cpp` and mode
U the same. On the GPU the bound was baked in harder still — `#define BDPT_MAXDEPTH 8` /
`BDPT_MAXV (BDPT_MAXDEPTH+3)`, used to size the *per-thread* stack arrays
`DVertex eye[BDPT_MAXV], light[BDPT_MAXV]` in `kBdptT<NS>`, with a clamp in `renderBdptCuda`.
So `ftrace -mode D -max-bounce 64` rendered at depth 8 and printed `maxDepth=8` without any
diagnostic.

The original entry judged this "a correctness/UX bug, not the cause of any particular dark
render" — that was **wrong**, and `scenes/mirror_sphere_interior.ftsl` is the counter-example.
A 97%-reflective silver cavity has a photon mean free path of ~1/(1-0.97) = 33 bounces, so
truncating at 8 does not dim the image, it deletes it. Measured on that frame at equal time,
fraction of pixels at *exactly* 0.0:

| `-max-bounce` | black pixels |
|---|---|
| 8 (the old hard-coded value) | 96.9% |
| 48 | 67.4% |
| 64 | 28.9% |

Fixed as the entry proposed: `kBdptT` is now templated on the depth bound too
(`template <int NS, int MAXD>`), `maxV` is threaded through `dRandomWalk` /
`dGenCameraSubpath` / `dGenLightSubpath` instead of reading the `#define`, and only two
variants are instantiated — `BDPT_MAXDEPTH` (8, bit-for-bit the old kernel and still the
default) and `BDPT_DEEPDEPTH` (64), the deep one launched only when `-max-bounce` asks for
more than 8. `main.cpp` mode D and mode U now read `g_maxBounceOverride`.

Two things to know about the deep variant. It is **~15x slower per sample** — the BDPT
connection double-loop is O(depth²) — which is inherent, not a regression. And GPU mode U
sizes its light-vertex slab as `npix * vcmCap * sizeof(DVcmLV)`, which at `vcmCap = maxDepth
= 64` and 1100x733 would be 6.6 GB; `vcmSessionBegin` now bounds `vcmCap` by a 768 MB slab
budget and prints when it clamps. That only limits how many light-subpath vertices are
*stored for merging* (subpaths still walk to `maxDepth`, and connections are unaffected), so
the estimator stays unbiased.

Remaining limitation: past 64 the GPU still clamps. Going deeper needs another instantiation,
and the per-thread local-memory footprint is already ~13 KB there.

### BUG — OPEN (2026-07-28): `light env { spd ... intensity N }` silently ignores `intensity`

In `ftsl.h` (~4563-4573) the env-light block parses `intensity` but only applies it on the
`file`-based (image env map) path; the analytic-SPD path calls `addEnvLight(spd, binWidth_)`
and drops the scale on the floor. An authored `intensity 40` therefore changes nothing, with no
warning. Proper fix: scale the spectrum by `intensity` (or pass it through to `addEnvLight`) on
the SPD path too, so both forms of `light env` mean the same thing.

### TECH-DEBT / DOC — OPEN (2026-07-28): a field leaf's `center` is applied *before* its `rotate`

`addFieldLeaf` composes the leaf transform as `authoredXf . TRS(center)`, i.e. `center` sits on
the *inside*. So `cylinder { center 0 -0.169 0.363   rotate 115 0 0  ... }` does **not** place a
cylinder at that point and then tilt it in place — it offsets first and the subsequent rotation
swings the whole thing onto a completely different axis. Authors have to use `translate` (read
in `fieldXf`, applied *after*) to get the intuitive "put it here, then tilt it" behaviour. This
cost real debugging time on `scenes/silver_sphere_xenon.ftsl` (see the comment above the port
cylinder there).

It is arguably not a bug — `center` is a leaf-local parameter, and reversing it now would break
existing scenes — but it is undocumented and surprising. Fix: document the composition order in
the FTSL reference next to `center` / `translate` / `rotate`, and say plainly that
`center` + `rotate` on the same leaf is almost never what the author means.

### BUG — DONE (2026-07-28, v0.99.0): the CUDA emitter upload silently coerced any unknown `EmitterShape` into a zeroed-basis Quad

`buildUploadScene` in `render_cuda.cu` mapped the host `EmitterShape` to `DEmitter::shape`
with an **open** ternary chain ending in `: 0` — "anything I don't recognise is a Quad" — and
`cudaForwardSupported()` had **no emitter-shape check at all**. So a shape added host-side
before the device kernels grew a branch for it did not fall back to the CPU and did not
error: it reached the kernel as a Quad whose `origin`/`u`/`v` basis was garbage. Malformed
geometry inside a kernel does not fail cleanly — it can fault the display driver and take the
machine down (this is the class of failure the `teardownLog` / BSOD instrumentation at
`render_cuda.cu:9440` was added for). A second, live instance of the same hazard: a `Mesh`
emitter with an **empty** triangle list got `shape = 5` with `meshTris == nullptr`, and the
device sampler's shape-5 branch clamps to index `-1` and dereferences it unconditionally.

Fixed by `deviceEmitterShapeCode()` (`render_cuda.cu`, just above `cudaForwardSupported`): one
**closed whitelist** shared by the support gate and the upload, so the two cannot drift. It is
a `switch` over every enumerator with **no `default`**, so adding an `EmitterShape` now raises
MSVC C4062 at compile time; `-1` after the switch is the runtime fail-safe. `cudaForwardSupported`
rejects any unimplemented shape (and any empty-triangle `Mesh`) so the scene renders on the CPU —
and since all eight GPU gates (`cudaBdptSupported`, `cudaBackwardSupported`, `cudaBackwardRGBSupported`,
`cudaIsoPreviewSupported`, `cudaPhotonMapSupported`, `cudaSppmSupported`, `cudaVcmSupported`)
chain to it, one guard covers every GPU entry point. The upload keeps a loud `stderr` diagnostic
+ dead-emitter fallback should the gate and the upload ever disagree.

Verified: `scraps/mesh_light.ftsl` renders identically on `-device cpu` and `-device gpu`
(absorbed 0.6591 vs 0.6588; mean |Δ| 1.9/255, pure MC noise).

### FEATURE — DONE (2026-07-28, v0.99.0): `ftrace -stop <pid>|all` — never force-kill a CUDA render again

Related to the above, and the other half of the same incident: force-killing ftrace mid-CUDA
(`taskkill /F`, e.g. because a `-keepwindow` preview was holding the exe open and blocking a
rebuild) is a well-known way to wedge the NVIDIA driver into a TDR/bugcheck. ftrace's *graceful*
stop already existed end to end — `g_stopRequested`, the SIGINT/SIGBREAK handlers, the poll sites
in every render loop, the final image + `.ftbuf` write — but it could only be triggered by Ctrl-C
from the owning console, which a detached render doesn't have.

`-stop` adds the missing external trigger: a sentinel file under `<temp>/ftrace/` (`<pid>.run`
published per live render, `<pid>.stop` to signal one), polled by a 250 ms watcher thread that
raises the *same* `g_stopRequested`. A file rather than a named event because renders run in the
interactive Console session while the shell signalling them may be in another session/window
station, and `Local\` kernel objects are per-session. `-stop` also releases a `-keepwindow` hold,
and waits (≤120 s) for the target to actually exit so a rebuild can be scripted right after.

### BUG + TECH-DEBT — DONE for the mesh pane (2026-07-28, v0.96.0): the loom viewer's 3-D panes were a CPU painter's-algorithm sort, not a z-buffer — on a D3D11 device that was already running

`viewer_gui.cpp`'s mesh pane collected every triangle across every mesh, CPU-projected it,
**sorted back-to-front by centroid depth** and emitted `AddTriangleFilled` into an ImGui
draw list. Two problems:

1. **Correctness.** A painter's sort by triangle centroid *cannot* resolve interpenetrating
   or mutually-overlapping geometry — a long triangle passing through a small one is drawn
   wholly in front of or behind it. Any loom scene with intersecting swept meshes rendered
   visibly wrong, silently. (`examples/viewer_live.py` is exactly this: an `orbit` tube
   threading a gyroid ball.)
2. **Cost.** An O(n log n) sort plus per-triangle CPU projection **every frame**, on the
   thread that also runs the UI.

The irony was that `viewer_gui.cpp` already creates a **D3D11 device + swap chain** (for
ImGui) — the hardware pipeline sitting right there, used only for 2-D.

**Fixed for the mesh pane** by `MeshGpu` (`viewer_gui.cpp`): one interleaved vertex buffer +
index buffer for the whole sidecar (per-mesh `firstIndex/indexCount/baseVertex` ranges, so
each mesh keeps its own skin and tint as its own draw call), rendered into an offscreen
RTV + **D32_FLOAT depth-stencil view** with runtime-compiled HLSL, then shown with
`ImGui::Image` — the pattern the Render pane already used. Consequences:

- Geometry uploads **once per tessellation** (`MeshView::geomGen`, bumped in `adoptSidecar`),
  so an orbit / zoom / colour-mode change is a 144-byte constant-buffer write and nothing else.
  Union bounds are baked with the upload rather than rescanned per frame.
- Flat two-sided lambert is *preserved exactly* (`0.30 + 0.70*|n.z|`), with the face normal
  recovered per-pixel from `cross(ddx(vp), ddy(vp))`; under the orthographic orbit projection
  that is exact, not an approximation.
- The wireframe is now a **real second depth-tested pass** (`D3D11_FILL_WIREFRAME`,
  LESS_EQUAL + a small negative depth bias) instead of relying on fill/wire interleaving.
- The UV checker is now evaluated **per-pixel** at the interpolated UV instead of once at the
  triangle centroid — the deliberate one behaviour change, since the point of a UV checker is
  to show distortion *within* a face.

**Still open:** the **curve and field panes** keep the same CPU `project3` + draw-list
approach. They draw lines/points rather than solid surfaces, so the occlusion bug does not
bite the same way, but the per-frame CPU projection cost is the same and they should get the
same treatment.

**Not** to be confused with the main explorer's rasterizer, which is *already* resident
("upload once, re-project per frame" — `raster_cuda.h upload()`); that one is a deliberate
CUDA compute rasterizer whose device tail is byte-identical to the CPU path, and it measures
9.12 ms/frame median at 1920×1920 on a 5.17 M-triangle scene (109.6 fps; per-pass: project
1.14, raster 2.36, shade 0.16, expose+encode 1.14, **download 1.09**). Porting *that* to a
graphics API would trade the byte-identical-backends guarantee and the non-matrix
fisheye/panoramic projections for a few ms — measure before touching it.

**The present tail, however, was worth attacking, and has been — DONE (2026-07-28,
v0.97.0).** Measuring it first was the whole point: the 1.09 ms download everyone
(including this note) had been eyeing was a *small* part of the problem. `LiveWindow`'s
host-side tail — a scalar per-pixel RGB8→BGRA repack plus a `HALFTONE` `StretchDIBits`
double-buffer — measured **10–29 ms per frame** (`scraps/tailbench.cpp`: 5.07 + 23.99 ms
at 1920×1920→1264×1264; 2.35 + 7.83 ms at 1264² 1:1; 2.94 + 12.37 ms at 1920×1080 1:1),
i.e. *2.5–3× the entire GPU render*, and it serialised with the render thread because
`paint()` held the same mutex `update()` needed. The image area is now presented by a
D3D11 flip-model swap chain on its own child HWND, with the RGB8 bytes uploaded
untouched and deswizzled in a shader (see `design.md`, `livewindow.*`). `-raster-bench`
now reports the tail directly: **9.02 ms → 1.32 ms median** at 1920² on this machine,
with byte-identical output. `FTRACE_LIVE_GDI=1` restores the old path for A/B or if a
driver misbehaves.

**And the download itself is now gone too — DONE (2026-07-28, v0.98.0).** With D3D owning
the image texture, the zero-copy finish became available and has been taken:
`raster_cuda::bindPresentTarget` registers the live window's texture with
`cudaGraphicsD3D11RegisterResource(..., cudaGraphicsRegisterFlagsSurfaceLoadStore)`, and
`renderFrameToTarget` ends in a `surf2Dwrite` tone-map kernel, so the image never crosses
the bus for the GPU-rasterized explorer. Every constraint listed above is respected:
adapter mismatch is detected at registration and **latched off** (`Scene::gfxOff`) so a
doomed register isn't retried per frame; the host-RGB `update()` call sites are untouched
and `main.cpp`'s `rasterPresent` declines the fast path (falling back to render+`update()`)
for the implicit-ray iso preview and for any frame `drawOverlay` must annotate. Byte
identity is guaranteed *by construction* rather than by testing: passes A–C are literally
the same code (`renderCore`), and the tone-map's per-pixel body — RN double intrinsics and
all — is one shared `__device__ inline` that both the buffer and surface kernels call. The
download used to double as the frame's fence and error check; `cudaStreamSynchronize(0)`
took over both. Measured @ 3840²: host `render 21.97 + tail 4.17 = 26.1 ms` vs zero-copy
**9.67 ms** (per-pass download `4.06 → 0.10 ms`). Verified against the download path with
`WM_PRINTCLIENT` captures (bit-identical), and the fallbacks were exercised individually
(forward photon mode B, `FTRACE_LIVE_GDI=1`, window resize → re-register, `+Pt` overlay).

### BUG — DONE (2026-07-28, v0.98.2): the CPU rasterizer leaks a hairline CRACK along a shared triangle edge (the GPU one doesn't)

**Symptom.** In the `-raster` preview a thin dark diagonal streak cuts across the cornell
box's ceiling/right-wall seam. Reproduce:
`ftrace scenes/cornell.ftsl -raster -r 800 600 -device cpu -o ppm/x.ppm -window`.

**It is a hole, not a shading artifact.** Every pixel on the streak is *exactly*
`(69,75,85)` — bit-for-bit the frame's clear colour, the same value the empty corners
outside the box carry. No triangle covered those pixels at all. The streak is a perfect
45° line (`x + y == 699` for every one of them, running the full image height from
`(100,599)` to `(699,0)`, 220 pixels), which is the projected shared edge between the two
triangles of a box face. So the CPU rasterizer's coverage rule is **not watertight**: on
this edge both adjacent triangles reject the pixel instead of exactly one accepting it.

**The GPU rasterizer gets it right**, which is how this was isolated: of the 210 pixels
where the two backends disagree on this frame, **157 are on that single seam**, and the
GPU fills each with the correct wall colour (e.g. `(216,195,191)` at `(476,223)` where the
CPU has clear). The remaining 53 disagreements are ordinary ±1-pixel edge-coverage
differences along wall silhouettes.

**Mechanism (confirmed empirically).** Both rasterizers use the *same* coverage rule —
normalized barycentrics with `if (w0 < 0 || w1 < 0 || w2 < 0) continue;` (raster.h:447,
raster_cuda.cu:515/753). That test is *inclusive* on all three edges, which with exact
arithmetic would double-cover a shared edge, never crack it. The crack comes from the tie
being decided by floating-point noise:

- At 800×600 the box's quad diagonals are **exactly 45°** and their line equations land on
  `x - y = 100` / `x + y = 699`. Since sample points are `px = x+0.5, py = y+0.5`, such an
  edge passes **dead-on through ~220 consecutive pixel centres** — every one an exact tie.
- The two triangles sharing that edge do *not* evaluate it identically: `w0`/`w1` are
  incrementally stepped from each triangle's own `xlo` (so different accumulation lengths
  at the same pixel), scaled by each triangle's own `1/area`, and the third weight is the
  *derived* `w2 = 1 - w0 - w1`, which rounds differently again. So the "same" edge can come
  out as a tiny negative in **both** triangles, and both reject.
- **Proof:** rendering the identical scene at 801×600 or 800×601 — which moves the edge off
  exact pixel centres — drops interior holes from **127 to 0**. Nothing else changed.
- The GPU escapes it only by *luck*: `float`'s coarser rounding happens to land these ties
  non-negative where `double` lands them negative. It is not more correct, just differently
  wrong, and another scene/resolution could crack it too.

**Fix as implemented — canonical edge functions (no fixed point, no top-left rule).**
The fixed-point/top-left plan sketched originally was dropped: it needs consistent winding,
which this codebase deliberately does *not* enforce (it accepts either sign of `area`
because a mesh's winding may disagree with its vertex normals). The shipped fix gets exact
tie *detection* out of plain floating point instead, by making the two sharers of an edge
evaluate it from **bitwise-identical operands**:

1. **Canonicalize each edge's endpoint order** lexicographically by `(sx, sy)`
   (`makeEdge` / `makeEdgeD`). Both sharers then build the same `P` and the same `Q - P`,
   bit for bit, regardless of which way round their own vertex list runs.
2. **Fold the orientation sign into the deltas**: `sf = sign(area) * flip`, and store
   `dx, dy = sf * (Q - P)`. The two sharers *always* get opposite `sf` — consistent winding
   flips `flip`, inconsistent winding flips `sign(area)` instead — so their edge values are
   exact negatives of each other. Negation is exact in IEEE and round-to-nearest is
   symmetric under it, so this costs no accuracy and no extra register.
3. **Tie rule**: accept an exact zero only when `sf > 0` (stored as `tie`). That holds for
   exactly one of the two sharers, so a pixel dead-on the edge is claimed exactly once —
   no crack, no double-cover. Non-zero values are unambiguous by construction.
4. **Anchor the evaluation at `P`**: `v = dx*(py - Py) - dy*(px - Px)`, with the `dx*(py-Py)`
   term hoisted per row. The expanded affine form's constant `Px*Qy - Py*Qx` is ~W·H in
   magnitude even for a short edge — in `float` its ulp alone displaces the edge line by
   ~1e-3 px. Shared *edges* stay watertight either way, but the three edges meeting at a
   shared *vertex* are perturbed independently, and that leaves an unclaimed sliver there.
   Anchored at `P`, every operand is a local offset. (Measured: the affine form left 1 hole
   at 3 of 6 test resolutions; the anchored form leaves none.)
5. **Drop the incremental stepping and the derived `w2 = 1 - w0 - w1`.** Each of the three
   weights is now evaluated directly. Stepping cannot be kept: each sharer would seed its
   accumulator from its own `xlo`, so the values would no longer be bitwise identical.
6. **CUDA only — defeat FMA contraction.** nvcc defaults to `-fmad=true` and contracts
   `r - dy*(px - Px)` into an FMA, which evaluates `dy*ax` exactly and subtracts it from the
   *already rounded* `r`, leaving `r`'s ±0.5-ulp rounding residual instead of a clean zero.
   Worse, it applies inconsistently: the two sharers test the same edge under *different*
   indices (`e0` vs `e1`), separate expressions the compiler contracts independently.
   Measured at the cornell box's bottom-back-right corner at 640×480, the floor's `e0` came
   out `-9.24e-07` while the right wall's `e1` was exactly `0` with `tie == false` — both
   rejected. Fixed with `edgeRow`/`edgeAt` wrapping `__fmul_rn`/`__fsub_rn`. The CPU twin
   needs no intrinsics: MSVC's default `/fp:precise` does not contract, and the build sets
   no `/fp:` flag — which is exactly why the CPU was clean here and the GPU was not.
7. Applied to **all five sites**: `fillTriangleG` and `fillTriangleClear` in raster.h, and
   `rasterRow` / `kShade`'s barycentric resolve / `kClear` in raster_cuda.cu. It matters
   most in the clear pass, which *multiplies* into `clearT`/`milkT`: a doubly-covered edge
   would darken a seam line twice, an uncovered one leaves a hairline of un-tinted glass.

Rejected alternative: widening the accept to `w > -eps` turns cracks into double-coverage
(mostly harmless here, since the z-test's strict `>` lets the first writer win). But the
epsilon is scale-dependent and it only moves the failure rather than removing it.

**Verification.** Interior holes at 800×600 went **127 → 0** on the CPU, and both backends
now report **0 interior holes at all eight tested resolutions** (800×600, 801×600, 800×601,
1024×768, 1280×720, 640×480, 1920×1080, 3840×2160). Hole detection is
`scraps/_crack.py` (a clear-coloured pixel with non-clear neighbours on left+right or
up+down).

**Cost.** ~4% on the CPU rasterizer (crystalloop, 120 frames at 1920×1080: 13.15 s → 13.70 s);
GPU unchanged within noise (0.92 s → 0.90 s). That is the price of evaluating three edge
functions per pixel instead of stepping two — it cannot be recovered without giving up the
bitwise identity the fix depends on.

**Wider implication.** The CPU and GPU rasterizers are still not *byte*-identical — CPU
evaluates in `double`, GPU in `float`, so at resolutions where an edge lands dead-on many
pixel centres the two can hand a tie pixel to different (but always to *some*) triangle.
Residual cornell-box disagreement: 188 px at 800×600, 2 px at 801×600, 6 px at 800×601,
1183 px at 3840×2160 — and **none of them involve the clear colour**, i.e. every one is one
real surface vs another, never a hole. So `-device` is now a safe switch for `-raster`
coverage, but not a bit-exact one. Output bytes changed, so golden images need regenerating.

### BUG — DONE (2026-07-28, v0.98.1): a dark fringe of speckles on the tessellated sphere's silhouette (both backends, identically)

**Symptom.** Isolated very dark pixels sit on the sphere's silhouette in the `-raster`
preview, e.g. at `(354,287)`, `(445,287)`, `(344,294)`, `(455,294)`, `(327,312)`,
`(472,312)`, `(325,315)`, `(474,315)` in the 800×600 cornell frame.

**What's known.** Unlike the crack above, these are **byte-identical in the CPU and the GPU
renders** (same positions, same `(93,83,81)`), so this is in shared geometry/shading, not in
either rasterizer's coverage rule. They are also **not** holes — `(93,83,81)` is a genuine
shaded colour, not the clear colour — and they are **mirror-symmetric in pairs about the
sphere's vertical axis** (`x=399.5`), with the half-width growing monotonically down the
arc. That symmetry rules out any race or thread-mapping effect and points at deterministic
geometry: the shading normal used at the extreme silhouette facets, where the interpolated
normal is nearly perpendicular to the view and `N·L` collapses.

It is **not** a zero-copy/interop regression: the download path and the surface path are
bit-identical here, and the artifact predates v0.98.0.

**Cause (found by probing the winning fragment).** The interpolated normal was *correct*.
The winning facet at `(327,312)` has vertex normals `(-0.747,0.643,0.170)`,
`(-0.866,0.500,0.000)`, `(-0.766,0.643,0.000)` and barycentrics `(0.073,0.436,0.491)`,
giving an interpolated normal of `(-0.808,0.581,0.012)` — outward, as it should be. What
inverted it was the **two-sided flip**, `if (dot(N3,V) < 0) N3 = -N3;`, applied *per pixel*
to the smoothly-interpolated normal. At a silhouette `dot(N,V)` legitimately grazes through
zero (here `-0.0497`) while the surface is still genuinely front-facing, so the flip fired
and produced `N = (0.812,-0.584,-0.013)`, for which `N·L <= 0` at every light. `lit`
collapsed to exactly 0 and only ambient survived (`k` 0.277 → 0.103) — a 1-px dark band
along every silhouette, showing as speckles wherever the band is isolated in both axes.

**Fix.** The two-sided decision is now made **once per triangle**, at projection time
(`projectRange` in raster.h, `kProject` in raster_cuda.cu), and the per-pixel flip is gone.
A triangle counts as back-facing only when **all three vertices** face away: a silhouette
triangle straddles the horizon (here `+0.11 / -0.068 / -0.058`) and keeps its smooth
normals, while geometry genuinely seen from behind — the cornell box's walls are wound
outward and viewed from inside, so they *rely* on the flip — has every vertex agreeing and
still flips exactly as before. Being per-triangle it is also constant across each facet, so
it cannot reintroduce an intra-triangle discontinuity.

Note what does *not* work, since both look plausible: the screen-space area sign is
useless here (the back wall's winner is `+9.3e4` and the sphere's rim winner `+7.0` — the
same sign, yet they need opposite treatment, because this scene's wall winding disagrees
with its vertex normals), and the *geometric* facet normal grazes at the rim just as the
shading normal does. The vertex normals are the only reliable signal, and unanimity is
what makes them decisive.

**Verified:** isolated dark pixels on cornell 800×600 go 8 → **0** on the GPU and the 8
shared ones vanish on the CPU (its remaining 22 are the separate fill-rule crack above).
On the CUDA side the decision rides in a new `kSlotBack` flag bit rather than being baked
into stored normals, because an unclipped slot has no `DAttr` record — `kShade` reads its
attributes bit-verbatim from the source `DPTri`.

### BUG — DONE (2026-07-28, v0.95.0): `python -m loom.anim` served an *empty* slot list — a module that is both `__main__` and importable is two different classes

**Symptom.** `ftrace -anim … -loom <scene.py>` came up with `0 bindable scene variable(s)`
on a scene that plainly declares one, so the bind row had nothing to offer and the live
channel drove nothing. The same code path exercised **in-process** (the `_run_cli` test
harness) worked perfectly, which is exactly why the test suite never caught it.

**Cause.** Running `python -m loom.anim` executes the module a second time under the name
`__main__`, *in addition to* the `loom.anim` that `import loom.anim` binds. Each execution
creates its own class objects, so the `SceneDriver` (etc.) that `__main__` constructs is
**not** the `SceneDriver` the imported half's `isinstance` checks test against. Every such
check silently fails, and the failure mode is a *quiet empty result*, not an exception.

**Fix + regression pin.** Beyond the fix in `loom/anim.py`, the bug is now pinned by tests
that spawn a **real subprocess** — `test_cli_subprocess_sees_the_scenes_slots_and_drives_them`
(`tests/test_anim_live.py`) and `test_viewer_m_entry_point_serves_the_same_session`
(`tests/test_viewer.py`). An in-process harness *cannot* reproduce this class of bug, since
it only ever has one module identity; only `subprocess.run([sys.executable, "-m", …])` does.
Both assert on the emitted artifact, not just the ack — the anim one checks the written
`.ftsl` actually contains `roughness 0.75`, so an ack that lies is still caught. Verified by
temporarily reverting the fix: the test failed with `assert {} == {'rough': 0.3 ± 3e-07}`.

**Generalisation worth remembering:** any module that is *both* a `-m` entry point and an
importable API needs at least one subprocess test. Everything else tests the wrong object.

### TOOLING — NOT A BUG (2026-07-28): `SetWindowTextW` on a control in **another process** silently does nothing

Cost roughly an hour of chasing a phantom bug in the bind row's `chans:` box: the box looked
inert from an external PowerShell driver, while every other control worked. `SetWindowTextW`
across a process boundary does not reach the control — it updates only the window text USER32
caches on our side, so the EDIT's own buffer never changes and it never raises `EN_CHANGE`.
Worse, it reads back *consistently*: the external `GetWindowTextW` returns that same cached
text, so the harness "confirms" a value the target process cannot see (its in-process
`GetWindowTextW` sends `WM_GETTEXT` and gets the real, unchanged buffer).

**Use `SendMessageW(hCtl, WM_SETTEXT, 0, L"…")`** — the message a keystroke actually produces;
it updates the control and raises `EN_CHANGE` by itself, with nothing to synthesize.
`scraps/bindtest.ps1` carries this note inline so the next harness doesn't repeat it. The
`chans:` box itself was never broken (`[anim] channels 4 → 7 → 5 → 3` all verified).

### PERF — OPEN (2026-07-27): scene loading is down 5×, but the graph walk (not the lexer) is what's left

The 0.68 front-end flip made loading measurably slower than the hand-written parser —
roughly linear in scene size, ~8.7 µs/byte at first, so the largest scene in the tree
(`scenes/gallery_settled.ftsl`, 22 KB) spent ~200 ms in the front end before rendering
started. Two rounds of work took that to **~42 ms** (lex 7.1 ms + parse 34.6 ms),
measured with `tools/gpda_lexcheck/lexcheck.exe scenes/gallery_settled.ftsl`:

- **Lexer, 47 ms → 7.1 ms.** The naive loop ran all 16 rule regexes at every position and
  `std::regex` costs ~1 µs a call. `src/gpda/gpda_lexer.hpp` now derives a first-byte set
  from each pattern and skips rules that cannot start with the current byte, and matches
  metacharacter-free patterns (9 of the 16) with a string compare. Validated by
  `tools/gpda_lexcheck/`.
- **Parser, 66.6 ms → 34.6 ms** (upstream GraphParser `f8ea9a3`, re-vendored): precomputed
  the per-node `return_links` shared_ptr instead of rebuilding it on each of ~19 RuleRef
  traversals per token; made the closure's visited set a hash table instead of a linear
  scan (8.16M key comparisons per parse → 93K); dropped the pointless `std::atomic` on the
  intrusive refcount (the pool is thread-local, so cross-thread refcounting was already
  corruption) — that alone was ~20% of parse time.

**What remains.** Measured on that scene: 424K closure steps, 61K `ParseNode`s built (each
with a heap-allocated `children` vector via `plist_to_vector` plus a rule-name
`std::string` copy), 140K `plist_push`es — and only ~2.4 of the ~30 expanded terminals per
token ever match, so most of those nodes belong to derivations that die. The next real win
is therefore not another micro-optimisation but **not building parse nodes speculatively**:
keep the children as the persistent list they already are and materialise
`ParseNode::children` once, lazily, for the tree that actually wins. That changes
`ParseNode`'s public shape, so it touches `ftsl_reduce.hpp`, both GraphParser test suites
and loom's mirror. Worth doing only if scene load shows up again — 42 ms on the single
largest scene, with the median scene well under 5 ms, is no longer near the top.

### TECH-DEBT — OPEN (2026-07-26): `GraphParser/cpp/scannerless.{hpp,cpp}` still throws bare `std::runtime_error`, not the rich `ParseError`

The tokenized engine (the one ftrace uses) throws a `ParseError` carrying line/col, the
exact accepted-continuation set, and the enclosing rule chain — that diagnostic quality is
the main reason the front-end flip was worth doing. The **scannerless** engine, which
shares the same graph-walking core, still throws a plain `std::runtime_error` with no
position and no expected set.

**Why it wasn't done with the tokenized version:** the tokenized parser gets `line`/`col`
for free (the lexer stamps every token) and its expected set is a set of *token* patterns,
which `join_expected` can name directly. Scannerless has neither: it would need
(a) a byte-offset → line/col map built once per parse from the input text, and (b) an
expected-set formatter that describes **char-level** matchers (literal strings, char
classes with ranges, `.`, unicode classes) in a form a user can read — e.g. collapsing
`[0-9]` back to its source spelling rather than dumping 10 alternatives.

**Proper fix:** hoist `ParseError` + `join_expected` + `escape_token_text` into a shared
header, add an offset→line/col helper to the scannerless `Parser`, and write a
`describe_matcher(const Node&)` for char-level terminals. Not urgent — no shipped ftrace
path uses the scannerless engine.

### TECH-DEBT — DONE (2026-07-27, 0.79.0): the hand-written `.ftsl` parser is deleted — one front end, one implementation of the language

0.68.0 flipped ftrace's front end over to the shared grammar
(`tools/loom/loom/grammar/ftsl_scene.epeg` → `src/gpda/`), gated on the corpus differ
reaching **MATCH 2595/2595** (every `.ftsl` in the tree, structurally identical down to
`Stmt::line`). The retired hand-written parser stayed compiled in behind
`-legacy-parser` / `FTRACE_LEGACY_PARSER` as a one-release escape hatch. It held for
**ten** releases (0.68 → 0.78) with no scene ever needing the fallback, so the escape
hatch had earned its retirement.

**Fixed** — everything downstream of `std::vector<Block>` is shared and untouched; only
the front half went:

- `src/ftsl.h`: deleted the `// Tokenizer` section (`enum class Tok`, `struct Token`,
  `tokenize`) and the whole 266-line `struct Parser` (`parseValue` / `parseBraceBody` /
  `parseOneTopBlock` / `parseBlockList` / `parsePrefer` / `parseTop`). `loadSource()` lost
  its `legacy_parse` lambda and the branch around it; it now just calls
  `ftsl_gpda::parse()`.
- `src/gpda/ftsl_frontend.hpp`: deleted `legacy_flag()`, `use_legacy()`,
  `validate_flag()`, `validate_enabled()` and the `validate()` template. What is left is
  `parser()`, `lexer()`, `parse()`.
- `src/gpda/ftsl_reduce.hpp`: deleted the structural differ (`struct Diff`, `diff_block`,
  `diff_value`, `diff_scene`) that drove the 0.68 corpus comparison — with one front end
  there is nothing left to diff against.
- `src/main.cpp`: the argv pre-scan for the two flags is gone (it existed only because
  the parse happens before the main CLI loop). Both flags are still **accepted** in the
  CLI loop so an existing script doesn't hit the unknown-option error, but each prints
  `ftrace: <flag> was retired in 0.79.0 — the shared grammar is the only .ftsl front end
  now; ignoring`. A flag that quietly stopped doing anything would be worse than one that
  is gone; this is why the bump is **minor**, not major.

The `BrItem` / `applyBracketGroup` comments were reworded: the "keep the bracket-group
decision at the loader level, out of the front end" rule was written for two front ends,
but it is still the right layering and is now documented as such rather than as a
drift-avoidance measure.

**Verified:** all eleven deterministic self-tests PASS (`-checkbvh`, `-checkimplicit`,
`-checklens`, `-checkfluoro`, `-checkfog`, `-checkthinfilm`, `-checkmultilayer`,
`-checkgrating`, `-checkupsample`, `-checkgrid`, `-checkscatter`); every `.ftsl` in
`scenes/` still loads; both retired flags print the notice and render normally.

### PAPERCUT — DONE (2026-07-26, 0.77.1): `-o` into a non-existent directory fails every write interval and loses the whole render

`ftrace -o png/nope/out.png …` ran the full render, printed `error: could not write …` at
each `-interval` tick, and exited having written nothing — the accumulated film was simply
lost. Hit twice (once with `png/bench/`, once with `png/parserflip/`).

**Fixed** in `src/main.cpp`, right after the `-check*` self-test early-returns (where `out`
is final — the bare-invocation preview path can still rewrite it, so the check can't move
into the argument loop itself). An `ensureOutDir` lambda resolves a path's parent and:

- parent empty or already a directory → nothing to do;
- parent exists but is *not* a directory → `ftrace: output path '…' exists but is not a
  directory` and `return 2`;
- otherwise `fs::create_directories` it, printing `[out] created output directory …` so a
  typo shows up in the log instead of silently making a stray directory; if creation fails,
  `return 2` with the `std::error_code` message.

Applied to `-o` and to `-savemap` (the one output path not derived from `-o`; a discarded
photon map costs as much as a discarded film). Everything else this run writes lives beside
`-o` — the `.ftbuf` checkpoint sidecar (`out + ".ftbuf"`), the per-camera `outFor()`
variants, the stereo eye pair — so the single `-o` check covers them all.

Verified: `-o png/dirfix/deep/nested/out.png` creates the whole tree and writes both the
PNG and its `.ftbuf`; `-o png/dirfix_file.png/out.png` (parent is a regular file) fails
before the scene renders.

### BUG + TECH-DEBT — DONE (2026-07-26, 0.78.0): `grid:` / `scatter:` now reach field / isosurface / density / ior / camera-track formulas — and the guard no longer corrupts the eval stack

Fixed in 0.78.0. Two things turned out to be tangled here, and the second was worse than
this entry claimed:

1. **The gap.** All four field-formula `compilePatternExpr` sites now pass `&tableScope_`
   (`src/ftsl.h`: `addFunctionLeaf`, medium `density`, medium `ior`, the `camera_curve`
   record driver), so a `function` leaf can be a measured height field, a medium density
   can be a sampled volume, an `ior` field can be a measured index volume, and a flyby
   track can be driven by tabulated data. `texScope_` is deliberately still withheld at
   those sites: `tex:` needs a hit's (u,v), which a field formula has no access to, so it
   remains a clean compile error.

2. **The "unreachable" stub was reachable, and it was a live wrong render — not latent
   GPU debt.** `medium { density pattern:<p> }` copies a *pattern*'s nodes (compiled WITH
   a table scope, so `PatOp::Grid` is real) into `med.density`, which `Medium::densityAt`
   evaluated through a bare `PatCtx` — `c.grids == nullptr`. The guard pushed 0 **without
   popping** its `ndim` operands, so `patternEval` returned `st[0]`: the first
   **coordinate**, not the sample. Confirmed by render: `scraps/gridmed.ftsl`
   (`grid:down(x)`, descending) came out as the *mirror image* of its analytic twin
   `scraps/gridmed_ref.ftsl` (`density "1 - x"`). Both guards (`pattern.h`, and the FP64
   `dPatternEval` in `render_cuda.cu`) now `return 0.0` for the whole program instead:
   the arity is the table's own `ndim`, which is precisely what can't be read when the
   header wasn't found, so no balanced pop is possible and a placeholder push is never
   safe.

**How it's plumbed.** Host: a new `PatTables` POD + `patBindTables` (`pattern.h`) and
`Scene::patTables()`, threaded as a `const PatTables*` parameter through `fieldLeafSDF` /
`fieldEval` / `fieldGradient` / `Implicit::eval`/`gradient` / `intersectImplicit` /
`estimateFieldLipschitz` / `marchImplicit` and `Medium::insideField`/`densityAt`/`nAt`/
`gradNAt`/`insideBound`. Never a member: it points into `Scene`'s vectors and a `Scene` is
copied and moved, so a cached copy would dangle. `Renderer::sampleMediaCollision` /
`mediaTransmittance` were retyped from `const std::vector<Medium>&` to `const Scene&` (18
call sites) so no caller can forget the tables. Device: `dPatternEvalF` gained a
`const DPatEnv& env` with real `Tex`/`Grid`/`Scatter` cases (coords promoted to double and
the result demoted, as `PatOp::PovFn` already did), threaded through `dFieldLeafSDF(F)` /
`dFieldEval(F)` / `dFieldGradient` / `dMedDensityAt` / `dMedInside` / `dMedNAt` /
`dMedGradN`; `dMediaSampleCollision` / `dMediaTransmittance` likewise now take the whole
`DScene` (24 call sites).

Verified, three ways:

* **Medium density** — `png/gridmed/grid.png` (`density pattern:rho` where `rho` is
  `grid:down(x)`, a *descending* ramp) is now **byte-identical** to `png/gridmed/ref.png`
  (analytic `1 - x`) on **both** backends (`mean|d|=0.000`, `max|d|=0`), where before the
  fix they were mirror images. The left-right-mirrored comparison is far off
  (`mean|d|=21.6`), which is exactly the corrupted render the guard used to produce.
* **Isosurface field** — the scratch pair `scraps/gridiso{,_ref}.ftsl` puts a 2×2 grid holding the bilinear
  `0.05 + 0.7*x` inside a `function { expr "y - grid:hf(x, z)" }` leaf; against its
  analytic twin it agrees to `mean|d|=0.46 / max 6` on both backends (the residual is the
  grid pool's **float** storage: `0.05f` shifts the plane ~7e-9 m, a sub-pixel silhouette
  jitter), while the mirrored comparison is `mean|d|=60`.
* **Self-test** — `ftrace -checkgrid` gained sections (f) and (g), which pin the two
  invariants directly and need no renderer: an unbound table evaluates to **0, never to a
  coordinate** (1-D and 2-D calls), and `Medium::densityAt` fed `Scene::patTables()`
  returns the sampled value while omitting the tables returns a clean 0.

The load-time majorant (`density_max` estimate), the isosurface Lipschitz probe and the
implicit-bound inside-sign detect all evaluate with real tables too — otherwise a
`grid:`-driven field would majorise/bound to 0 and vanish.

Checked-in worked example: `scenes/grid_field.ftsl` — a 5×5 lattice read as an isosurface
height field (`expr "y - grid:terrain(x, z)"`) inside a corridor filled by a 1-D
`density "grid:haze(y)"` profile, i.e. both new sites in one scene.

<details><summary>original entry</summary>

### TECH-DEBT — OPEN (2026-07-27): `grid:<name>(…)` / `scatter:<name>(…)` are *surface-pattern* samplers only — field / isosurface / density formulas can't reach them

0.71.0 added the N-D `grid` datatype and the `grid:<name>(c0, …)` sampler; 0.72.0 added its
ragged sibling `scatter` and `scatter:<name>(c0, …)` (docs: FTSL.md §6.1, examples:
`scenes/pattern_grid.ftsl` / `scenes/pattern_scatter.ftsl`, self-tests: `ftrace -checkgrid` /
`-checkscatter`). Both are wired into every *pattern* site — `pattern`, `texture { rgb … }`,
record drivers/stops, material overrides — but deliberately **not** into the four
`compilePatternExpr` call sites that compile scalar *field* formulas (`src/ftsl.h` ~2864
`function` block, ~3441 density field, ~3560 isosurface `expr`, ~4446 the `allowT` record
driver): those are compiled with no `PatTableScope`, so `grid:foo(x,y,z)` there fails at
compile time with `unknown grid` (likewise `unknown scatter`).

That is the safe behaviour, not a silent wrong answer, but it blocks the obvious use: a
**sampled density volume** or a measured height field driving an isosurface, which is
exactly what these datatypes are for. The GPU side already reflects the gap —
`dPatternEvalF` (the FP32 twin in `src/render_cuda.cu` that evaluates field formulas)
carries a shared `case PatOp::Grid: case PatOp::Scatter:` that pushes `0.0f` **without
popping its operands**, because the operand count is the table's own `ndim` and is not
knowable there. It is unreachable today; it would corrupt the eval stack the moment either
sampler became reachable from a field formula.

**Proper fix:** pass `&tableScope_` at those four sites too, thread the two tables and the
shared `dataPool` into the FP32 field-eval environment the same way `DPatEnv` threads them
into `dPatternEval` (the samplers `patGridSample` / `patScatterSample` in `pattern.h` are
already `__host__ __device__` and shared, so only the plumbing is missing), and replace the
stub case with real ones that pop `ndim` coordinates. Then drop this entry and the "surface
patterns only" caveat from FTSL.md.

</details>

### TECH-DEBT — FIXED (`reflect` 0.75.0, `transmit` 0.76.0, `emit` 0.80.0)

0.73.0 added inline array literals (`roughness [0 1](u)`, FTSL.md §6.1, example
`scenes/pattern_array.ftsl`). They desugar to `pattern:__arrN` and therefore work at exactly
the set of slots that already bind a scalar pattern — which is *narrow*: `bindScalarPattern`
is called for `roughness`, `film_thickness_map`, `weight_map` and a coat's `roughness`, and
nothing else. In particular **spectrum-valued slots** (`reflect`, `transmit`, `emit`, …)
accept a spectrum expression or a `texture:<name>`, but not a pattern — so the natural
spelling from the design note, `reflect [0 1](u)`, fails with "unrecognized spectrum
expression". That is a clear error rather than a wrong render, but it is the very example
the feature was designed around, and the same gap blocks `reflect pattern:p` for a
hand-written pattern.

**FIXED for `reflect` in 0.75.0.** A scalar pattern in a spectrum slot is a per-hit
**multiplier** on whatever the slot otherwise evaluates to (`Material::reflectPat` /
`DMaterial::reflectPat`, clamped to [0,1]). Multiply is strictly more general than the
"greyscale reflectance" reading and degenerates to it: `reflect pattern:p` leaves the
pattern *alone* in the slot, so the base spectrum becomes a flat 1.0 and the albedo is
`p(hit)` — greyscale — while `reflect rgb … ` + `reflect_map pattern:p` modulates a tint
(or a bound `texture:`). Colour therefore always comes from the spectrum/texture, never
from the scalar. Applied in the two shared accessors `diffuseReflectance` / `reflectSlot`
(`scene.h`) and their device twins `dDiffuseRho` / `dReflectSlot`, so every renderer and
both backends pick it up at once; the RGB-bake fast path opts out like it already does for
`reflectTex`. Worked example: `scenes/reflect_pattern.ftsl`.

Verified: `reflect [0 1](u)` and `reflect pattern:p` (with `expr "u"`) render
**bit-identically**; the albedo read-out tracks `u` linearly; GPU-vs-CPU disagreement on
the pattern scene (RMSE 6.83/255 at 200 spp) is *below* the no-pattern control (7.12), i.e.
pure Monte Carlo noise.

Only the families whose reflect slot goes through those two accessors honour a pattern
(diffuse, translucent, mirror, halfmirror, glossy, grating); the loader hard-*rejects* one
elsewhere rather than dropping it silently, since a lone `reflect pattern:` leaves a
flat-1.0 base that would otherwise render as albedo 1.0 — a wrong image, not a missing
effect.

**FIXED for `transmit` in 0.76.0**, but only after the refactor this entry called for.
`m.transmit(lambda)` was read as a bare spectrum lookup at **16 host call sites across 6
renderer headers** (`render.h`, `backward.h`, `bdpt.h`, `vcm.h`, `photonmap_render.h`,
`sppm_render.h`) and **16 device sites** in `render_cuda.cu`, with no shared accessor. All
32 now funnel through `transmitSlot(scene, m, h, lambda)` in `scene.h` / `dTransmitSlot` in
`render_cuda.cu` — the single point of truth for *both* readings of the slot, a `filter`'s
gel transmittance T(λ) and a `translucent`'s back-hemisphere albedo ρ_T. Callers keep their
own `clamp01` and the two-lobe callers still apply the ρ_R + ρ_T ≤ 1 energy guard *after*
the multiplier. There is no record channel and no texture on this slot, so the accessor has
only the one base path. `Material::transmitPat` / `DMaterial::transmitPat` then bind
`transmit pattern:<n>` / `transmit [0 1](u)` / `transmit_map`, honoured on `translucent` and
`filter` and rejected everywhere else (every other type leaves `transmit` at 0 and never
reads it). The reflect and transmit loader paths collapsed into one shared
`patternedSpectrumParam` + `checkSlotPatSupported`, and the RGB-bake fast path opts out on
`transmitPat` alongside `reflectPat`. Worked example: `scenes/transmit_pattern.ftsl`.

Verified: `transmit pattern:p` (flat `expr "0.4"`) and `transmit 0.4 transmit_map
pattern:p_one` render **bit-identically** to plain `transmit 0.4`; the ramp panel's measured
brightness tracks `u` and plateaus exactly where the ρ_R + ρ_T guard kicks in; CPU-vs-GPU
disagreement on the pattern scene (RMSE 6.40/255 at 300 spp) is *below* the no-pattern
control (9.88), i.e. pure Monte Carlo noise; mode D (BDPT) and mode V agree — V's
forward-vs-backward best-fit scale is 0.994. All 11 `-check*` self-tests pass and the
grammar-equivalence sweep is ok=380 / mismatch=0.

**FIXED for `emit` in 0.80.0** — the strict leg. Same two spellings (`emit pattern:<n>` /
`emit [0 1](u)` alone in the slot, `emit_map pattern:<n>` modulating an authored spectrum),
plus the `light`-block alias `spd pattern:` / `spd_map pattern:` for the same slot;
`finalizeEmitters` copies `Material::emitPat` onto the `Emitter` so both spellings converge
on one runtime field. What makes emission different from reflect/transmit is that it is read
from **both sides of transport** — emission-on-hit when a camera path lands on the emitter,
and Le at the point NEE / a light subpath samples — and MIS *combines* the two, so a
pointwise disagreement is **bias**, not noise. The profile is therefore only legal where the
sampler's (u,v) provably equals the hit's: `EmitterShape::Quad` (bilinear parameters) and
`EmitterShape::Mesh` (barycentric UVs; `EmitTri` gained `uv0`/`uvE1`/`uvE2`). Sphere,
cylinder, spot, collimated and env are **refused at load** — `addLight`'s subtype gate for
the `light` route, `checkEmitPatsSupported` after `Scene::build()` for the material route.
Reads funnel through `emitSlot` / `emitterPatMulAt` / `emitterSamplePoint` in `scene.h`.
The pattern is a pure post-multiplier on radiance / photon beta: `power`, `pdfChoice`,
`pdfPos`/`pdfA`, `emissionPdfW`, `directPdfW` and every VCM `dVCM`/`dVC`/`dVM` are untouched,
which is what makes it unbiased by construction. Caveat, documented rather than
auto-corrected: `power`/`lumens` normalise the *unpatterned* spectrum, so a profile averaging
0.5 emits about half the requested flux. Worked example: `scenes/emit_pattern.ftsl`.

Verified unbiased at 160×160 against four independent estimators: R vs D (BDPT) global ratio
1.00268 — and a **control with the pattern removed** shows the same 1.00240 with the same
corner-shaped tile profile, so that residual is a pre-existing R-vs-D estimator difference,
not the pattern; B (forward photons, 400M) vs R global 1.00005; U (VCM, 1500 spp) vs R global
1.00018. Load rejection of `light sphere { spd pattern:p }` confirmed. Fixed a latent
pre-existing bug on the way: the area light's *second* triangle carried default UVs
disagreeing with `addQuad`'s, i.e. a diagonal seam for any UV-driven emission pattern **or
texture** on an area light.

### TECH-DEBT — DONE (0.82.0): the emission pattern (`emitPat`) now runs on the CUDA backends

0.80.0 shipped `emit pattern:` / `emit_map` (and `spd` / `spd_map`) on the CPU only.
`cudaForwardSupported` returned false if any `Emitter` or `Material` had `emitPat >= 0`, and
`cudaBackwardRGBSupported` did the same, so a patterned scene silently fell back to the CPU.

That was deliberate, not an oversight: unlike `reflectPat`/`transmitPat` — which funnel
through one or two shared accessors — the device has roughly **20 emission read sites**, and
because emission is read from both sides of transport a *partially* ported pattern would
produce a **biased** image rather than a visibly missing effect. Rejecting the whole scene
was the safe state until every site could be done at once.

**What landed (0.82.0), all in `src/render_cuda.cu`.** `DEmitter::emitPat` and
`DMaterial::emitPat` upload; `DEmitTri` carries `uv0`/`uvE1`/`uvE2` and the device
`emitterSamplePoint` gained optional `uuOut`/`vvOut` (filled for Quad from the bilinear
`u1,u2` and for Mesh from the chosen `EmitTri`'s barycentric UVs), so a *sampled* point
reports the same (u,v) the *hit* path interpolates. Three accessors mirror `scene.h`:
`dEmitPatMul` (the emission-on-hit side, twin of `emitSlot`'s `slotPatMul`),
`dEmitterPatMulAt` and `dEmitterSamplePointPat` (the sampler side). Every device emission
read routes through one of them — forward `genPhoton`/`genPhotonHero`; backward
`bkEmitterGeom` (folded into the λ-independent `G`, so scalar *and* hero NEE pick it up from
one place, exactly as host `emitterGeom` folds it into `w`), `bkNeeVolume`, `bkNeeLightRGB`,
`bkRadiance`/`bkRadianceHero`/`bkRadianceRGB`; the three photon-map visible-point sites;
BDPT's `dGenLightSubpath` + `dConnectBDPT` s=1; and VCM's light subpath + s=0 emission + NEE.
`DVertex` gained a cached `Real emitPatW` (twin of `bdpt.h Vertex::emitPatW`) read by
`dVertexLe`, set at all seven construction sites, so every BDPT MIS strategy is covered from
one place. Unpatterned scenes stay bit-identical: each new factor is either guarded by
`if (epat != 1.0)` or is an exact multiply by `1.0`, and the extra UV outputs consume no RNG.

Two deviations from the plan sketched above, both deliberate. (1) The planned `dEmitSlot`
became `dEmitPatMul`: `DMaterial` carries no emit spectrum on the device (emission is read
from `sc.emitters[li].emitSpd`/`.rgbEmit`), so only the *multiplier* can be factored out.
(2) The `cudaBackwardRGBSupported` gate was dropped too, not merely left in place — an
emission pattern is an **achromatic** scalar, so it commutes with the spectral→RGB bake
(`ep·∫CIE·emitSpd == ∫CIE·ep·emitSpd`) and can be applied to the pre-baked `rgbEmit`. That is
why it differs from `reflectPat`/`transmitPat`, which genuinely are not evaluated on that
path and still reject. Deliberately *not* patterned: `dInvPdfLambda`'s
`g += gw * specLookup(e.emitSpd, lambda)` — a wavelength pdf, matching the host — and the
spot/env branches, which cannot carry a pattern (refused at load).

**Validation** (`scenes/emit_pattern.ftsl`: two patterned quad area lights + a patterned
*mesh* emitter, i.e. both UV-carrying shapes; absolute exposure so images compare directly):

- **GPU vs CPU, mode R, 2000 spp, 512²** — mean luminance ratio **0.9999**, median 1.0000,
  sRGB RMSE 2.36/255, itself below the images' own 2.24% noise floor.
- **GPU cross-estimator, 160²** — global B/R = **1.00001**, D/R = **0.99995**. U/R = 1.00679,
  but an **unpatterned control** gives U/R = **1.00706** with the same per-band profile, so
  that residual is a pre-existing VCM-vs-R estimator difference on this scene, not the
  pattern. Per luminance band, B/R and D/R are 1.0000 everywhere; the U excess sits only in
  the dim indirect bands, while the band containing the directly-visible patterned panels —
  the one place the profile is read most directly — is U/R = 1.0002.
- **GPU VCM vs CPU VCM on the patterned scene** — mean **0.9998**, median 1.0000 (the direct
  test that the device VCM pattern path matches the CPU reference).
- **GPU RGB fast path vs GPU spectral R** — mean **0.9991**, median 1.0000, confirming the
  achromatic-commutes-with-the-bake argument for dropping that gate.
- **Photon map, mode M, GPU vs CPU at 30 M photons** — mean **1.0056**, median 1.0000. The
  larger per-pixel RMSE (10.7/255) is mode M's inherent density-estimate noise — the two
  backends pick different adaptive gather radii — the same character the M2/M4 validations
  recorded, not bias.
- **Forward energy closure**: `sum/emitted = 1.000000` in mode B at 4×10⁹ photons and in
  mode M at 30 M.
- **Load rejection still enforced**: `light sphere { spd_map pattern:p }` is refused with the
  "samples positions that no surface (u,v) corresponds to" diagnostic. The host-side gate
  that makes the whole feature sound is untouched by the port.
- All 11 `-check*` self-tests PASS; all 80 `scenes/*.ftsl` load.

Related, and deliberately *still* out of scope: `raster.h` / `raster_cuda.cu` (the preview
rasteriser) ignore `emitPat`, consistent with their existing behaviour for
`reflectPat`/`transmitPat` — the preview is a shading approximation, so this is a cosmetic
mismatch, not bias.

### BUILD BUG — FIXED (2026-07-26): editing a header did not rebuild the `.cu` files, and the linker could then keep a **stale copy of the function you just changed**

**Symptom that exposed it.** A change to `PhotonMap::buildAuto` (a header-inline function in
`src/photonmap.h`) had no effect on the running binary across *four* consecutive
`build.bat` runs — including one after deleting `build_cuda2/bin/ftrace.exe` to force a
relink. A `printf` added inside the edited block never appeared. `main.obj` genuinely
contained the new code (`grep -a` found the new format string in it); the linked exe did
not. Touching `src/render_cuda.cu` by hand fixed it instantly.

**Root cause.** MSVC's `ClCompile` records every `#include` it opened (`/showIncludes` →
`.tlog`) and rebuilds a `.cpp` when any of them changes. MSBuild's **`CudaCompile` task does
not do this at all** — it compares only the `.cu`'s own timestamp. So a header-only edit
left `render_cuda.obj` / `raster_cuda.obj` stale while the build reported success.

That is much worse than "the GPU path is one build behind", because `render_cuda.cu` and
`raster_cuda.cu` include the *same* headers as the C++ TUs (`photonmap.h`, `render.h`,
`scene.h`, …). Every inline/template function in those headers is emitted as a **COMDAT in
both objects**, the linker keeps exactly one copy, and it is free to keep the stale one.
Net effect: **a header-only edit could silently produce a binary running the OLD body of a
function, pulled from a `.cu` you never touched — even on a CPU-only code path.** Any
"bit-identical before/after" validation done without touching a `.cu` was potentially
meaningless.

**Fix.** `CMakeLists.txt` now runs `cmake/touch_stale_cu.cmake` as a `PRE_BUILD` step of the
`ftrace` target: it bumps the mtime of any `src/*.cu` older than the newest `src/*.h` /
`src/*.cuh`, which is the one signal `CudaCompile` does honour. `OBJECT_DEPENDS` is **not** a
usable fix here — CMake's Visual Studio generator emits `<CudaCompile Include="..."/>` items
with no metadata at all, so the `AdditionalInputs` never reach MSBuild (verified by
inspecting the generated `ftrace.vcxproj`). The check is deliberately coarse (any header
newer than a `.cu` rebuilds both `.cu` files, ~1.5 min) and is idempotent, since the touch
sets the mtime to now. **Verified:** `touch src/photonmap.h` followed by `build.bat` now
prints `[cuda-deps] photonmap.h is newer than render_cuda.cu — touching it so nvcc rebuilds`
and recompiles both `.cu` TUs.

**Consequence for past results:** the SoA photon-map change (`fd643ce`) was re-validated
after this fix — `-nopmauto` output is bit-identical to the pre-change binary
`scraps/ftrace_base_99a898d.exe` on CPU mode M, CPU mode S, GPU mode M and GPU mode S, so
that commit's claims stand. Earlier header-only measurements that were never re-checked
should be treated with suspicion.

### TECH-DEBT — DONE (2026-07-24, v0.49.0): volumetric blackbody emission ("fire", C3) now runs on the GPU forward tracer

**Was:** the emissive-volume path (a `medium` with a `temperature vdb:` grid + `emission blackbody`) ran only
on the **CPU forward** tracer; `cudaForwardSupported()` returned false for any `emissive()` medium so
`-device gpu`/`auto` fell back to the CPU (without the gate the device `genPhoton` crashed indexing
`sc.emitters[0]` on an emissive-only scene with `nEmitters==0`).

**Fix (shipped v0.49.0):** full GPU mirror in `render_cuda.cu`. (1) The VDB brick sampler was refactored into
a reusable `DVdbGrid` + `dVdbSample()` (density path unchanged, bit-for-bit) and `DMedium` now carries a
second `tempGrid` + `emissive`/`emitKelvin`/`tempPeak`/`emissionScale`. (2) `DScene` gained a
`DEmissiveVolume[]` (`mediumIndex`/`bmin`/`bmax`/`meanKe`/`power` + the per-volume Planck-λ CDF `lamCdf`) plus
`totalEmissionPower`, uploaded from `Scene::emissiveVolumes`. (3) `dBlackbodyEmissionRadiance` /
`dMedTemperatureAt` / `dMedEmissionAt` port the host emission field. (4) `genPhoton` got a volume-birth branch
(power-weighted emitter-vs-fire split with NO extra RNG when there are no volumes → non-fire scenes
unperturbed; uniform-AABB point, λ importance-sampled from `lamCdf`, isotropic dir,
`β=grandTotal·κ_e/(meanKe·Δλ·p(λ))`) and an isotropic `connectEmissionVolume`/`connectEmissionLensVolume` +
`camSplatEmissionAll` device splat. (5) The `emissive()` clause was dropped from `cudaForwardSupported`.
Fire scenes always have media so the hero path (gated on `mediaN==0`) never runs; an emissive-only scene never
indexes `sc.emitters` because `volumeBirth` is always true when `totalPower==0`. **Verified:** `-device gpu`
on `scraps/vdb_fire.ftsl` renders without crashing and matches the CPU flame shape/colour in distribution;
the refactored density path (`scraps/vdb_cloud.ftsl`) still renders correctly with `sum/emitted=1.000000`.

### TECH-DEBT — DONE (2026-07-24): fire emission now importance-samples λ from a blackbody → collapses the magnitude speckle

**Was:** a fire photon drew its wavelength **uniformly** over the band and carried
`β = grandTotal·κ_e(x,λ)/meanKe`. Because a warm blackbody (default 1500 K) is heavily red-weighted, the
per-photon β varied wildly with the κ_e *magnitude* across λ, so a partly-converged fire showed coloured
(notably green) speckle in its hot core.

**Fix (shipped v0.48.1):** each `EmissiveVolume` now carries an `EmissionSampler lamSampler` built in
`finalizeEmissiveVolumes()` from `blackbody(emitKelvin)` (a per-nm CDF over the band). The volume-birth branch
in `render.h` draws `λ = ev.lamSampler.sample(rng, pdfLam)` and weights
`β = grandTotal·κ_e/(meanKe·Δλ·p(λ))`. For a voxel at `emitKelvin`, `κ_e ∝ Planck(emitKelvin) = p(λ)·integral`,
so β is **exactly constant** across λ — the κ_e magnitude variance is removed entirely (cooler voxels get a
mild residual, always <1). Unbiased for any `p(λ)>0`; uniform `p=1/Δλ` recovers the old estimator. Verified
side-by-side: the old uniform render (more converged) is riddled with green specks; the importance-sampled
render (less converged) is markedly cleaner and red-orange throughout.

**Residual (inherent, not a bug):** the CIE-*shape* variance remains — a rare green-λ photon is still a
full-brightness green dot because its colour is `CIE(λ)`. This is intrinsic to per-photon spectral splatting and
only shrinks with more samples (or full hero-wavelength XYZ splatting). Left as-is; converges correctly.

### TECH-DEBT — DONE (2026-07-24): analytic sky (K2) bakes the physical solar disk into the env, so sun-lit surfaces converge slowly in forward modes

**Resolved 2026-07-27 (0.84.0)** by building exactly the proposed proper fix: a first-class
**`EmitterShape::Sun`** distant directional emitter, exposed as `light sun { elevation … azimuth …
angle … spd … }` and as a new `sun_disk on|off|separate` option on the Preetham sky block
(`separate` strips the baked disk out of the map and registers an **energy-matched** `light sun`
beside the skylight dome).

Forward emission (`Scene::addSunLight` + `render.h` / `render_cuda.cu` `genPhoton`/`genPhotonHero`)
samples a travel direction in the solar cone (pdf `1/Ω`) and an entry point on a disc of radius
`R` *perpendicular to it*, pushed upstream to `sceneCenter − dir·R` (pdf `1/πR²`); with
`geomWeight = envGeom = Ω·πR²` the spawn is exactly analog and **every photon enters the scene**.
Backward NEE (`backward.h` `neeLight`, device `bkEmitterGeom`/`bkNeeLight`/`bkNeeLightHero`/
`bkNeeVolume`/`bkNeeLightRGB`) samples the cone with `1/pdfW = Ω`; the directly-viewed disc is
added on a ray miss **only under the `specularArrival` gate**, which is exactly unbiased with no
MIS weight because NEE runs precisely at the material types that then clear that flag. The hard
cone reuses `spotCosInner == spotCosOuter == cos θ` (making `spotOmega` evaluate to `Ω`), so no new
emitter field was needed on host or device. The authored `spd` is **perpendicular irradiance**, so
widening `angle` softens the penumbra without touching the exposure. Modes `D`/`U` refuse a sun
scene (not connectible in area measure), as they already do for `spot`/`env`.

Measured: forward B vs backward R **0.09%**, CPU vs GPU mode R **0.01%**, `-rgb` fast path
**0.00%**, photon-map M vs R **0.02%**, SPPM S CPU vs GPU **0.16%** (8 passes, radius still
shrinking), `angle` 0.53°→8° exposure shift **0.019%**, `sun_disk
separate` vs baked `on` **0.12%** (once the map resolves the disc at res 4096). New deterministic
self-test **`ftrace -checksun`** pins the four invariants with no scene/renderer/RNG: the spot-field
reuse really equals the cone solid angle, `L·Ω == E⊥` at every angular diameter (exposure
invariance), `sampleCone` is uniform in solid angle about both the forward and the NEE axis, and
`inCone` agrees with it at the rim (so the NEE estimator and the direct-view miss term see the
same disc and the split cannot double-count). Worst absolute error 5.3e-14.

The convergence win this entry was filed for, measured on the very scene named below
(`scraps/sky_test.ftsl`, mode B, GPU): **baked** reached only 7.2% noise after 30 s / 2×10⁹
photons and the image still had *no* warm sunlight and *no* cast shadows (the sun had barely been
hit), while **`sun_disk separate`** hit the 4% target in **5.5 s / 3.0×10⁸ photons** with the sun
fully formed — ~**20× fewer photons** for the same noise, and a qualitatively correct picture
rather than a skylight-only one. Original entry follows.

The Preetham sky (`src/sky.h`, `light env { sky preetham … }`) bakes the solar disk into the
equirectangular `EnvMap` at its **physical** magnitude (~10⁵× the mean sky luminance). Directly-viewed
sky (including the sun) is read from the env background and is noise-free, and the diffuse sky dome
converges normally, but **sun-lit diffuse surfaces are high-dynamic-range** and pick up firefly/shot
noise that needs a large photon budget to denoise in the forward modes (A/B/C) — e.g. `scraps/sky_test.ftsl`
at 400×400 did not reach 4% noise in thousands of GPU batches, while the sky itself was clean (CPU==GPU,
std 2.8). This is **the same limitation ftrace has with any sunny HDRI** (a small ultra-bright env
feature), not specific to the sky — it is consistent with the "lights like an HDRI" design, so it ships
as-is. **Proper fix:** add a first-class **distant directional sun** emitter (a new `EmitterShape`) that
emits *parallel* photons across the scene's projected cross-section in forward mode (every photon enters
the scene, none wasted) and is next-event-estimated within its angular cone in the backward/MIS paths,
with the sky dome (no baked hard disk) carrying only skylight. That separates the ~10⁵× sun from the env
importance sampler and makes daylight scenes converge like any single-light scene. Requires touching the
emitter enum + forward emission (`render.h`), backward NEE (`backward.h`), and the GPU mirror
(`render_cuda.cu` `DEmitter` + emission + NEE) — a real feature, logged here until built.

### TECH-DEBT — DONE (2026-07-23): GPU VCM (mode U) downloads the whole light-vertex slab every pass (memory + PCIe overhead scales with resolution)

**Resolved 2026-07-23 (0.39.1)** by implementing exactly proper-fix option (a): compaction and the
merge-grid build now run **entirely on-device** (thrust `exclusive_scan`/`inclusive_scan` over
`lvCount` → `pathBegin`/`pathEnd`, a scatter kernel packs the slab into a tight compact array, a
float bbox `transform_reduce` + `kVcmCellKey` + **stable** `sort_by_key` + `lower_bound` reproduce
`VcmGrid::build`'s counting sort type-for-type). Only a single 4-byte vertex count crosses PCIe per
pass (was ~69 MB down + ~27 MB up at 256², plus per-pass cudaMalloc/Free churn — now grow-only
buffers + a thrust arena allocator, zero steady-state device mallocs). The stable sort preserves the
host counting sort's exact output order and the grid geometry repeats the host float/double mixed
expressions, so the result is **byte-identical PNGs** vs the 0.39.0 baseline (validated on
`cornell.ftsl` mode U). The SPPM session got the same treatment (photon grid built on-device, cie
fold via double device twins of `color.h` — one ±1/255 pixel in 65,536 from CUDA-vs-MSVC `exp` ulp).

<details><summary>Original entry (for context)</summary>

Logged while implementing M12 (GPU VCM, `VcmSession` in render_cuda.cu). The device stores each
light subpath's connectible vertices into a **per-path slab** `lvSlab[i·vcmCap + k]` (one fixed
`vcmCap = maxDepth = 8`-vertex slice per pixel, chosen to avoid cross-thread atomics), and each
pass `vcmSessionPass` **downloads the entire slab plus the per-path counts** and compacts it on
the host into contiguous per-path ranges before rebuilding the merge grid and uploading it back —
mirroring the SPPM session's download-photons-then-host-build pattern. Costs: slab memory
`npix · vcmCap · sizeof(DVcmLV)` (~64 MB at 256², ~1 GB at 1024²) and the full slab copied
device→host every pass plus compacted array + grid host→device.

</details>

### TECH-DEBT — OPEN (2026-07-23): mode-R GPU GRIN has a small bent-region float-vs-double residual (does not converge with spp)

Logged while validating M11 (GRIN Eikonal marcher on the GPU backward path). The device
marcher `dGrinMarch` (render_cuda.cu:2040) is algorithmically byte-identical to the CPU
`grin::march` (grin.h) and accumulates its running (ro,rd) in **double** to mirror the CPU
ground truth, so both backends bend rays by the same large amount and the images are
structurally near-identical (SSIM ≈ 0.99, Pearson ≈ 0.99 on the linear-gradient validator).
**But** a small *systematic* (non-noise) rel-error survives inside the **bent lens region**:

| scene (`scraps/`) | disc rel-err | periphery rel-err | converges with spp? |
|---|---|---|---|
| `grin_lin.ftsl` (smooth linear `1.35-0.35*y`) | ~2.7% | ~0.7% | **disc does NOT** (2.90%→2.69% from 400→1600 spp); periphery halves (1.2%→0.69% = pure noise) |
| `grin_val.ftsl` (strong radial caustic) | ~17% | ~1.7% | disc stays high; caustic amplifies it |

Root cause (measured, not guessed): the device geometry/BLAS + `dMed*` queries run in
**float** (Real=float, FTRACE_GPU_FP32) while the CPU is double. Through a gradient-index
lens the ray→image map is magnified, so a sub-pixel float difference in the marched trajectory
lands the warped high-contrast checker edge a fraction of a pixel away from the CPU — averaging
to ~2.6% over the disc, scaling to ~17% in the caustic-heavy radial case. Accumulating the
running Eikonal state in double (already done) does NOT remove it — the residual enters via the
float geometry queries, and the same queries on the *straight-ray* periphery show ~zero bias
(they converge to 0.69%), so it is pure lens amplification, not a marcher-logic error. A
separate pre-existing global ~1.2× mode-R float-GPU vs double-CPU exposure difference exists
even with NO medium and is folded into this (both are the device-float regime).

Not a correctness bug (both backends bend correctly and agree structurally); it is the accepted
GPU float-precision envelope showing up amplified in the bent region. **Proper fix (if ever
warranted):** run the device geometry/BLAS intersection for GRIN-marched rays in double, or
supersample the bent region on GPU — high cost for a sub-pixel edge placement difference, so
deferred. Repro: render `scraps/grin_lin.ftsl -mode R` on `-device gpu` and `-device cpu`, then
`python scraps/grin_residual.py png/grin_lin_gpu.png png/grin_lin_cpu.png` (watch disc rel-err
vs spp).

### BUG — OPEN (2026-07-23): GPU vs CPU participating-media brightness disagree systematically across modes (phase-independent, pre-existing)

Discovered while statistically validating M10 (rainbow media on device). For a bounded
homogeneous fog scene (`scraps/rb_val_lowvar.ftsl` / `rb_val_hg.ftsl`, absolute exposure so
GPU and CPU share a FIXED gain), the GPU and CPU converge to **different overall brightness**
for the SAME mode/scene — and the gap is **phase-independent** (a plain-HG control shows the
same factor as a rainbow medium), so it is **not** an M10 rainbow bug:

| mode | GPU↔CPU B/A (CPU/GPU), rainbow | HG control | notes |
|---|---|---|---|
| B (forward photon) | ~1.25 (median 1.02) | ~1.58 (median 1.00) | firefly-heavy; **bulk medians agree**, but the image *mean* and the `[energy]` line disagree |
| D (BDPT) | **2.41 uniform** (median 2.44, p10=1.48…p90=4.27) | **2.41 uniform** | cleanest signal: well-converged (~3% noise), low firefly tail, uniform shift across *every* percentile ⇒ real systematic factor, not noise |
| R (backward ref) | ~1000× (CPU near-black) | ~1000× | mode R forces media to a single GLOBAL homogeneous haze; camera embedded in fog renders lit-fog+bow on GPU but near-black on CPU ⇒ CPU mode-R likely mishandles camera-inside-global-haze in-scatter/NEE |

Corroborating engine diagnostic: at albedo 0.5 (slab optical depth ~0.16) forward mode prints
`[energy] absorbed=0.1516` on GPU vs `absorbed=0.0008` on CPU — a ~190× gap. GPU's ~0.15 is the
physically-expected single-pass absorbed fraction; **CPU appears to barely absorb via medium
albedo**, which would also make CPU *brighter* (consistent with B/A>1 in modes B and D). So the
direction hints the **CPU** may be the wrong one (under-absorbing / an accounting-vs-transport
mismatch) — but it could equally be the GPU BDPT missing volume connection strategies (which
would make GPU dimmer). **Needs investigation to determine which backend is correct.**

Scope: affects ALL GPU participating media (HG and rainbow alike); pre-existing (M10's git diff
leaves the HG BDPT phase path bit-for-bit unchanged, yet HG still shows the 2.41×). Repro:
`ftrace scraps/rb_val_hg.ftsl -mode D -o ppm/x_gpu.ppm -device gpu -noise 3` and again with
`-device cpu`, then `python scraps/rb_robust_compare.py ppm/x_gpu.ppm 6 ppm/x_cpu.ppm 6`.
Proper fix: reconcile the two backends' medium collision/absorption + BDPT volume-connection
strategies so GPU==CPU converges to B/A≈1.0; likely a forward-medium albedo-absorption
accounting fix on CPU and/or a missing GPU-BDPT volume strategy.

### PERF — DONE (2026-07-23): interactive `-explore`/`-fly` felt intermittently slow (GPU parked in its idle power state between mouse-look bursts)

Users reported the GPU rasterizer explorer being slow right after launch on
`scenes/gallery_settled.ftsl`, then "suddenly fast" after moving around for a second or
two in the *same* window (no resize, no restart), then slow again on a fresh `-explore`
relaunch. This was NOT a rasterizer regression — `-raster-bench 200` on that scene reports
**3.91 ms/frame (255 fps)** at 1440×810 warm, so the kernels are fast (the 92% speedup a
prior session landed is real). The intermittency was **GPU DVFS**: the explorer re-renders
ONE frame per camera move, then idle-sleeps 15 ms, so the NVIDIA driver reads the bursty,
low-duty submission pattern as "idle" and parks the card in its lowest power state. Measured
live on an RTX 4090 with `nvidia-smi`: **P8 @ 210 MHz idle vs P0/P2 @ 2520–2760 MHz under
sustained load — a ~13× clock drop**, and ~33× slower for a first cold frame once
first-frame allocations are added (an early interactive frame timed at ~130 ms / 7.6 fps vs
3.9 ms warm). Only a second or two of *continuous* motion built enough sustained load to
ramp the clocks — exactly the "slow → suddenly fast → slow again after relaunch" pattern.

**Fix (main.cpp, explorer loop):** a GPU clock keep-warm grace window. For `kWarmGraceSec`
(2.5 s) after the last real interaction, the loop keeps submitting GPU render work even when
the frame hasn't changed — a "warm-only" `rasterOne` whose result is discarded and never
touches the window/overlay — and skips the idle sleep, so the driver holds the boost clock
through an active exploration session. Once the user is idle past the grace window the loop
falls back to the passive 15 ms sleep and lets the card power all the way down. Gated on the
discrete-GPU path (`gpuRaster != nullptr`); the CPU rasterizer is unaffected. Net effect:
mouse-look no longer pays the cold-clock penalty on every fresh burst.

**Follow-up (2026-07-23, 0.22.1): the first cut of the keep-warm made the timeline slider
chunk.** The initial version rendered a discarded warm frame on *every* non-changed loop
iteration during the 2.5 s grace window. During an active scrub drag, whenever `drainNav`
briefly returned no new thumb position, the loop burned a full ~4 ms `rasterOne` warm frame
while the user's thumb kept moving — so the next drain jumped several cameras ahead. Whether
a warm frame landed in that gap was timing-dependent, so the timeline "chunked" by N cameras
per drag *intermittently*, toggling within one session (reported after the keep-warm landed).
Root cause: the warm frame stole the loop slot that would otherwise have sampled the next
scrub position. Fix (final, 0.22.1): gate the warm frame on `kWarmGapSec` (0.10 s) of `idleFor`
so it fires ONLY after a genuine pause — during an active drag the sub-frame gaps between input
events stay under the gap, so warm frames are fully suppressed and every loop slot samples the
next scrub position (smooth tracking, no chunk). Once past the gap (a real pause) warm frames
run *continuously* to actually hold the boost clock; an intermediate attempt to rate-limit them
to a 0.12 s trickle was measured too weak — the card stayed at P8 (210 MHz), defeating the
keep-warm. Between events inside the gap the loop naps 3 ms (prompt drain, no busy spin).

### BUG — DONE (2026-07-22): a group-scaled collimated beam lit only half its footprint (offset by half its width from the aim point)

A group-scaled `light collimated { ... }` illuminated only one half of its `w×w`
footprint, split along a hard seam through the aim point — a straight-down (`dir 0 -1 0`)
god-ray beam over a fog box lit only `x > aim`, leaving the other half dark. Repro:
`scraps/_fogtune2.ftsl` showed a sharp vertical brightness seam at the beam's aim x that
did **not** move when the beam was widened (scale 250→700) and vanished when the beam was
tilted off-axis.

Root cause (NOT the ONB — `linalg.h onb()` is the robust Duff branchless construction and
is fine for `-y`): the emitter quad is sampled **corner-anchored** — `Emitter::samplePoint`
does `y = origin + u*u1 + v*u2` with `u1,u2 ∈ [0,1]` (`scene.h:549`), correct for ordinary
area lights where the user gives `origin + u + v` explicitly. But `ftsl.h`'s collimated
branch passed `xf.apply(origin)` as that corner, so the `w×w` footprint extended `+u,+v`
from the aim point instead of straddling it. For a bare 3 cm pencil (`w = Len(0.03)*scale`)
the ½-width offset is invisible; under `scale 250` (`w = 7.5`) the beam sat entirely on the
`+u/+v` side, so the aim point fell on the footprint edge — the seam. Tilting changed `u/v`
and smeared the offset diagonally, which is why it "fixed" it.

**Fix:** `ftsl.h` collimated branch now anchors the quad at `xf.apply(origin) − ½(U+V)`, so
`origin` (the aim point) is the **centre** of the beam footprint. Verified with
`scraps/_seamtest.ftsl` (pure `dir 0 -1 0`, `scale 300`): the footprint is now symmetric
about the aim point with no seam. Note this shifts a bare default pencil by ~1.5 cm×scale
vs. the old corner behaviour (intended — `origin` should mean the beam centre).

### TOOLING (2026-07-22): Nsight Compute (`ncu`) blocked by ERR_NVGPUCTRPERM — GPU perf counters admin-locked in the driver

`ncu` profiling of ftrace kernels fails with `ERR_NVGPUCTRPERM` (GPU performance
counters restricted to admin). Until unlocked, kernel-level profiling has to fall back
to ablation timing (code ablation + scene-copy ablation, as used for the 2026-07-22
BDPT campaign) and `cuobjdump -res-usage ftrace.exe` for register/stack pressure.

**Fixes (in order of practicality):**
1. **Just run `ncu` from an elevated (Administrator) shell** — the restriction is only
   enforced for non-admin users, so an admin PowerShell/Terminal profiles without any
   system change or reboot. Simplest for occasional profiling.
2. **Permanently allow all users via the driver registry policy** (this is the value the
   Control-Panel checkbox actually writes). In an elevated PowerShell:
   ```powershell
   New-Item -Path "HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" -Force | Out-Null
   Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" -Name "RmProfilingAdminOnly" -Type DWord -Value 0
   ```
   `RmProfilingAdminOnly = 0` = allow all users; `1` or absent = admin-only. **Reboot**,
   then `ncu` works from a normal shell.
3. NVIDIA Control Panel → Developer → Manage GPU Performance Counters → "Allow access to
   all users" is the documented GUI path, **but the Developer section is frequently
   absent** from recent drivers' classic Control Panel (verified 2026-07-22: no Developer
   node in the left tree, and there is no toggle for it under Desktop/3D-settings), so
   don't rely on this — use (1) or (2).

### DEBT (2026-07-22): gallery GPU configs are now dominated by the ~8.4 s fixed CPU-side setup (parse/tessellate/BVH/upload)

After the 0.19.14 FP32 implicit march, the gallery kernels are cheap enough that the
fixed per-run scene setup — `.ftsl` parse, mesh/implicit prep, BVH build, device upload,
shared by every mode — is the biggest remaining term: ~8.4 s of `M_gpu_gallery`'s ~9.7 s
total and ~46% of `D_gpu_gallery`'s ~18.2 s (numbers from `scraps/bench_dm_final.json`,
RTX 4090). Any further wall-clock win on these configs must come from the setup path
(profile first: parse vs tessellation vs BVH vs upload — unmeasured as of this entry).

### BUG (2026-07-22): freshly CMake-configured build dirs produce a GPU-silently-dead ftrace.exe (black renders, no CUDA error) — affects fresh clones!

Any build dir configured from scratch (observed with VS 2022 generator + CUDA 13.0,
`cmake -S . -B <fresh> -G "Visual Studio 17 2022" -A x64 -DFTRACE_GPU_FP32=ON`) yields an
exe whose GPU renders are silently dead: the GPU is detected, kernels appear to launch and
"complete" (cornell `-n 1e6 -device gpu` finishes in ~0.5 s), no CUDA error is reported,
but the tally is `absorbed=1.0000 sensor=0.0000`, auto-exposure=1, image black. Proven
**source-independent** during the 2026-07-21/22 optimization campaign: revisions 5b75bfb
and f0d2f13 both broken when built from freshly configured dirs (tried both
`FTRACE_CUDA_ARCH=native` default and `=89` → `--generate-code=arch=compute_89,code=[compute_89,sm_89]`,
full recompiles), while the **same 5b75bfb source built in the long-lived
`build_cuda2`** (incremental, originally configured weeks ago) renders correctly
(absorbed=0.6724 escaped=0.3301 on cornell). HEAD from the long-lived dir also works.
The failure signature (everything "absorbed", zero sensor hits, no error) suggests device
code that runs but sees zeroed/duplicated `__constant__` scene state — prime suspects:
RDC device-link `__constant__` symbol duplication, or a difference in the VS CUDA
integration props/targets picked up at configure time between the old and new CMake runs.
**Workaround:** build in the long-lived `build_cuda2` (for old revisions: temporary
`git checkout <rev> -- src/`, build, `git checkout HEAD -- src/`). **Must investigate** —
a fresh clone of the repo currently cannot produce a working GPU build. Next steps: diff
the generated `ftrace.vcxproj` + CMakeCache between the long-lived and a fresh dir
(beyond the arch flags already ruled out), check CUDA toolset version selection, and probe
`cudaMemcpyFromSymbol` of the scene constants at render start in a fresh-dir build.

### MINOR (2026-07-22): GPU raster see-through output isn't run-to-run bit-stable (atomicMulF product order)

The GPU rasterizer's see-through clear pass (`kClear` in src/raster_cuda.cu) accumulates
per-pixel transmittance/milk as `atomicMulF` products. Float multiplication isn't
associative, so when ≥2 clear fragments cover one pixel the low-order bits of the final
product depend on which thread's CAS lands first — a GPU-scheduling artifact. Observed
during the 2026-07-22 raster-perf campaign: `-in scenes/gallery_settled.ftsl -camera cam
-raster -see-through -raster-bench 20 -device gpu` flips between exactly two output
sha1s (`fdd7e910…`/`dfb365a7…`) across runs, stable *within* a time window (back-to-back
invocations of even *different* builds agree; runs hours apart can differ — consistent
with clock/thermal state steering the same race the same way). Affects the 0.19.0
pre-optimization binary identically, so it predates the optimization work; the opaque
paths (no `-see-through`) are fully deterministic. The kernel comment's
"order-independent" claim holds mathematically (commutative product) but not bit-exactly.
Impact: invisible (±1 ulp on a transmittance product); matters only to byte-comparison
harnesses, which must A/B ref-vs-new within one run (as scraps/rb_verify.sh does) rather
than compare hashes across sessions. Proper fix if ever needed: deterministic ordered
reduction (sort fragments per pixel by slot index, or accumulate in fixed-point).

### MINOR tech debt (2026-07-22): HIP alias block deliberately omits `__shfl_sync` (kRasterMed ticket broadcast)

`kRasterMed` in src/raster_cuda.cu broadcasts its warp ticket with
`__shfl_sync(0xffffffffu, li, 0)`. The file's HIP compatibility alias block does **not**
alias it, on purpose: a naive `__shfl` alias would compile on ROCm but silently
mis-broadcast on wave64 GPUs (64-lane wavefronts vs the 32-lane mask/stride the kernel
assumes), dropping triangles. As written, a HIP build fails loudly at the call site
instead. Proper fix when HIP is actually targeted: make the ticket queue wave-size-aware
(`warpSize`-based lane math + the matching wave-wide shuffle/ballot intrinsics). No
impact on CUDA builds; HIP remains untested/no-AMD-hardware anyway (see the HIP entry
near the end of this file).

### FIXED (2026-07-21): mode D heap-use-after-free (dangling `Vertex&` across `push_back`) + per-work-unit RNG seeding makes all CPU spp/photon modes chunk- and resume-independent

Two intertwined fixes, one commit (v0.18.2):

**(1) BDPT use-after-free (real, long-standing).** `bdpt::randomWalk` (src/bdpt.h) took
`Vertex& prev = path.back()` *before* `path.push_back(v)`; when the push reallocated,
`prev` dangled — the later `wo = normalize(prev.p - cur.p)` read freed heap (corrupting
MIS pdfs by up to ~9% rel in ~1e5 doubles of a 256² film) and `prev.pdfRev = …` **wrote**
8 bytes into the freed block (heap corruption; almost certainly the one-off hard crash
seen in a bench run — exit with no output, never reproduced). Whether the realloc fired
at a given vertex depended on the *capacity history* of the per-thread `eye`/`light`
vectors, i.e. on how `cpuSppChunks` happened to split the spp — which is wall-clock
adaptive — so paired runs differed and the corruption masqueraded as a thread race.
Fix: index (`prevSurfIdx`), matching the medium branch, which already did it right.
Verified: MSVC ASan (container annotations proven active via probe) clean on the
reproducing config; ftbufs byte-identical across three builds (ASan / plain CPU /
CUDA) under forced splits (`FTRACE_CHUNK_SPP=K` debug env, added to `cpuSppChunks`
alongside `FTRACE_CHUNK_DEBUG`); adaptive-split pairs now differ only at ≤1e-15 rel
(pure summation-order ulp, same benign class as mode R).

**(2) Per-work-unit seeding.** Every photon (A/B/C/P/M/S) and every (pixel, sample)
(R/D) now seeds its own Pcg32 via `seedUnit(rng, unitIndex, salt)` (splitmix64 mix in
src/rng.h) instead of seeding per chunk/thread — the realization is independent of
chunk splits, thread count, banding, and `-resume` boundaries. This also fixed SPPM
(mode S) re-emitting the *same* photons every pass (`tracePhotonPass` now takes
`seedBase` = cumulative emitted count), which had silently capped S convergence.
Observable consequence: CPU renders of R/P/D/M/S produce different (correct-noise)
realizations than v0.18.1; bench reference hashes rebased (`scraps/bench_cpu2.json`).

### PERF NOTE (2026-07-22): GPU photon-gather CIE side-table tried and REVERTED — 11–14% *slower*; don't retry

The mode-M CPU win (commit 9907034: precompute per-photon `cieX/Y/Z` once in
`PhotonMap::build`, 3.65×) does NOT transfer to the GPU gather (`dPhotonGather`,
src/render_cuda.cu). A device twin (`kPhotonCie` filling a 3-`Real`-per-photon
side-buffer, reads bit-identical — sha1 matched the per-visit evaluation exactly)
benched 96.7–99.1s vs 86.8s baseline on `M_gpu_cornell` (-n 3e7 -spp 64 -r 512,
RTX 4090). Reason: with FTRACE_GPU_FP32 the CMF fit is 21 `expf` → SFU ops, nearly
free, while the gather is memory-latency-bound on random photon reads — adding a
second 12-byte random-access stream per visit only added traffic. Packing the CIE
into `DPhoton` itself would grow the struct 32→44B and tax every mode's deposit
bandwidth, so that variant wasn't pursued either. Keep the CPU-side table only.

### TECH-DEBT (2026-07-20, updated 2026-07-26): hero-wavelength sampling is on ALL the CPU tracers (R + A/B/C + M/S + BDPT D + VCM U) and the whole GPU megakernel (forward A/B/C + M-deposit, backward R, BDPT D, VCM U) — the GPU wavefront backend is the last single-λ path
`radianceHero()` in `src/backward.h` gives the **backward reference tracer (`-mode R`, CPU)** and
`tracePhotonHero()` in `src/render.h` gives the **forward light tracers (`-mode A/B/C`, CPU)** and the
**CPU photon-mapping modes M (photon map) + S (SPPM)** hero-wavelength spectral sampling (hero λ + 3 stratified
secondaries, `hero.h` `kHeroC=4`). Its **device twin** (`traceHeroPhoton`/`genPhotonHero`/`shadeStepHero` in
`src/render_cuda.cu`) now gives the **GPU forward megakernel** the same thing (modes A/B/C and the mode-M photon
deposit), `bkRadianceHero()` gives the **GPU backward megakernel** mode R, and the templated `kBdptT<NS>` gives
the **GPU BDPT megakernel** mode D, and the templated `kVcmLightT<NS>`/`kVcmCameraT<NS>` gives the **GPU VCM/UPS
session** mode U (all DONE items below).
Validated: mode R on `cornell.ftsl` (chroma 0.89× overall / 0.74×
spectral-dominated, luma flat); modes A/B/C on `cornell` mode B (chroma 0.77×, luma 0.97×, energy
`sum/emitted≈1.0025`, dispersion intact); mode M on `cornell` (energy conserved exactly — auto-exposure identical,
chroma 0.87×, luma flat); GPU mode B on `cornell` (n=5e7, 300²: `-heroc 4` and `-heroc 1` both converge to
auto-exposure 1.06e-13, energy conserved exactly). The remaining §L-HERO sub-items are **not yet done**:
- **GPU forward megakernel (A/B/C + M-deposit) — DONE 2026-07-20.** `render_cuda.cu` grows a device twin of
  `tracePhotonHero`: `genPhotonHero` (stratified λ via the shared `sampleLambdaU`), `shadeStepHero` (per-λ
  deposit + camera splat via `connectHero`/`connectLensHero`/`camSpecularSplatAllHero`, hero RR with secondary
  reweight), `traceHeroPhoton` (de-hero at a dispersive interface boosts the hero ×C and falls through to the
  scalar `shadeStep`). No duplication: the nine specular lobes were extracted into a shared device
  `interactSpecular()` used by both `shadeStep` and de-hero. `kTrace` branches on a new `heroC` parameter;
  `launchForward` gates it on `mediaN==0 && !hasGrin` and **forces the megakernel** (hero is not in the wavefront
  scheduler), threaded through `renderForwardCuda`/`renderForwardSharedCuda`/`renderPhotonMapSharedCuda` fed
  `g_heroC`. `-heroc 1` is bit-identical to the classic single-λ device stream.
- **GPU wavefront (streaming) backend** (`render_cuda.cu` `wavefrontTrace`) — still 1 λ/photon; `-wavefront`
  with `-heroc>1` silently falls back to the megakernel (which does carry hero). Port the SoA pool to hold the
  `lam[]`/`beta[]` bundle if the streaming backend ever needs the chroma win on divergent scenes.
- **GPU backward megakernel (`renderBackwardCuda`) — DONE 2026-07-26.** `bkRadianceHero()` in
  `src/render_cuda.cu` is the 1:1 device twin of the CPU `radianceHero` (`bkNeeLightHero`/`bkNeeEnvHero` ↔
  `neeLightHero`/`neeEnvHero`, `bkInteract` ↔ `interactMaterial`, same de-hero material set, same gate). Four
  supporting extractions (`dSampleSceneLambdaU`, `bkEmitterGeom`→`BkNeeGeom`, `bkEnvGeom`→`BkEnvGeom`,
  `bkInteract`) are pure code moves that deliberately return geometric *pieces* rather than fused weights, so the
  scalar path's fp32 rounding is untouched. `kBackward` branches on a new `heroC`; `renderBackwardCuda` gates on
  `heroC>1 && mediaN==0 && !hasGrin && !cam.hasLens()`. Validated on `cornell` (dispersive SF10 sphere, so
  de-hero fires): `-heroc 1` byte-identical to the pre-change binary at `-spp 1`; converged `-heroc 4` vs
  `-heroc 1` agree to 0.03 % mean luminance (both auto-expose 1.04e-13); chroma noise 0.540→0.416 (0.77×) at
  C=4 with luma flat; same wall-clock; media/GRIN scenes byte-identical between `-heroc 1` and `-heroc 4`.
- **BDPT (mode D, CPU) — DONE 2026-07-26.** `src/bdpt.h` carries a `HeroBundle` (the C wavelengths + their
  `invPdfLambda`, drawn once per sample from ONE stratified base draw) along **both** subpaths: `Vertex` gained
  `betaSec[kHeroMax-1]` + `nUp`, `randomWalk` propagates the secondaries with a per-material
  `secRatio[i] = f_{i+1}/f_hero` reweight and de-heros at every delta vertex, and all four `connectBDPT`
  strategies evaluate `f`/`Le` per-λ into a `Lsec[]` out-parameter. Two design points worth remembering:
  (a) every *sampling* decision is hero-driven, so **one** `misWeight` serves the whole bundle; (b) the ×C
  de-hero boost used by the unidirectional tracers is deliberately **not** folded into the vertex throughputs —
  two independently de-hero'd subpaths would square it — so each vertex records `nUp` and the splat normalises
  once by `1/min(nUp_light, nUp_eye)`. Because Glossy is *connectible* (non-delta) in BDPT, mode D keeps the
  bundle alive across glossy bounces, which the unidirectional tracers do not. Validated: `-heroc 1` on
  `cornell` byte-identical to a pre-change rebuild; energy checked on `scenes/absolute.ftsl` (absolute mode =
  fixed sensor gain, so the noisy p99 auto-exposure can't confound it) — `-heroc 4` matches `-heroc 1` to
  **0.002 %** mean luminance at 2048 spp; chroma-noise RMS 0.1564→0.1247 (0.80×) with luma 0.2178→0.1967
  (0.90×) at 128 spp vs a 2048-spp reference; `_fog_cornell.ftsl` byte-identical across `-heroc 1`/`4`
  (media gate). Cost 1.19–1.38× wall-clock on these trivially light scenes, so still a win at equal time.
  *Methodology note for future hero work:* never compare two runs by their printed `auto-exposure` — it is a
  p99 statistic printed to 3 significant figures, worth ~±1 % on its own. Use an absolute-mode scene.
- **GPU BDPT megakernel (`kBdptT`) — DONE 2026-07-26.** 1:1 device port of the CPU `HeroBundle`/`nUp` scheme:
  `DHeroBundle` (C λ + their `invPdfLambda`), `dRandomWalk`/`dGenCameraSubpath`/`dGenLightSubpath`/`dConnectBDPT`
  all carry the bundle, one `dMisWeight` per connection, `1/min(nUp_light, nUp_eye)` normalisation at the splat.
  **Design point:** the per-vertex secondary throughputs live in a **parallel array** (`pathSec[v*secStride+i]`)
  rather than inside `DVertex`, and the kernel is templated on `int NS` (secondary slots) so the scalar
  instantiation `kBdptT<0>` sizes those arrays at 1 and pays *zero* extra local memory — `DVertex` is ~100 B ×
  2·`BDPT_MAXV` per thread (~8 KB of spilled frame already), so widening the struct would have cost the scalar
  path real occupancy. The host `renderBdptCuda(..., int heroC)` gates on
  `heroC>1 && mediaN==0 && !hasGrin && !cam.hasLens()` and picks the `kBdptT<BDPT_NSEC>` vs `kBdptT<0>`
  instantiation. Validated: `-heroc 1` byte-identical to a pre-change rebuild (`cornell` 160²/256 spp);
  energy on `scenes/absolute.ftsl` (absolute mode = fixed gain) `-heroc 4` vs `1` = **−0.008 %** at 2048 spp, and
  on a new glossy+translucent scratch scene `scraps/abs_hero_mats.ftsl` **−0.000 %** at 4096 spp; chroma noise
  0.1614→0.1270 (0.79×) / luma 0.2313→0.2087 on `absolute`, 0.1195→0.0955 (0.80×) on `cornell`, and — because
  Glossy and DiffuseTransmit are both connectible, so the bundle *never* de-heros there — **0.42× chroma AND
  0.42× luma** on `abs_hero_mats`; cost 1.17–1.18×. Media (`_fog_cornell.ftsl`) and lens (`scenes/realcam.ftsl`)
  gates byte-identical across `-heroc 1`/`4`. CPU vs GPU hero BDPT agree to **0.03 %** on `absolute` at 2048 spp.
- **Mirror and Filter no longer de-hero in BDPT (mode D) — DONE 2026-07-26.** `randomWalk`'s old rule was
  "every delta vertex de-heros", which is right for dielectric / thin-film / multilayer / grating / half-mirror
  (each picks its continuation by a wavelength-dependent process) but needlessly conservative for `Mirror` and
  `Filter`, whose outgoing direction does not depend on λ at all. Both now set a `keepBundle` flag that opts
  them out of the `if (delta) nUp = 1;` collapse and carry the secondaries on a per-λ factor instead (CPU
  `src/bdpt.h` + GPU `src/render_cuda.cu` `dRandomWalk`, kept 1:1).
  **This forced a real correctness fix, not just a policy tweak.** The per-λ factor used to be a *ratio* to
  the hero's (`secRatio[i] = f_i / f_hero`), which is undefined exactly where it matters most: a Wratten gel's
  `T(λ)` is legitimately **0** across most of the spectrum, so `T(λ_hero) == 0` while a secondary is wide
  open. The old code's `if (f_hero <= 0) terminate` then dropped the whole bundle including the live
  secondaries — measured **−4.9 %** energy on a Wratten-58 test scene. The array is now the **absolute**
  per-λ factor `secF[i] = f_i·cos/pdf_hero` (plus a `secChromatic` flag for the λ-independent cases, which
  just reuse `betaFactor`), and the early-out became a **max-over-live-λ** test. That also removes the same
  latent bias for Diffuse / Glossy / DiffuseTransmit, whose spectral albedo can hit exactly 0 at the hero λ.
  With `nUp == 1` every hero loop is empty and `mxF == betaFactor`, so it is the old scalar test verbatim.
  **Validated** on a new scratch scene `scraps/abs_hero_delta.ftsl` (absolute mode; a `metal:gold` mirror back
  wall + sphere and a `filter:wratten-58` gel pane, so nearly every path crosses one):
  energy `-heroc 4` vs `1` = **−0.006 %** at 2048 spp (was −3.978 % with the ratio formulation, and the two
  lobes isolated separately gave mirror −0.006 % / filter −4.894 %, pinning it on the filter);
  noise at 128 spp vs a 32768-spp reference — single-λ chroma 0.1801 / luma 0.3685, hero-4 *before* this
  change 0.1676 / 0.3522 (0.93× / 0.96× — the bundle died at the first mirror, so hero bought almost
  nothing), hero-4 *after* **0.0842 / 0.2086 = 0.47× chroma and 0.57× luma**;
  `-heroc 1` byte-identical to the pre-change build on both backends (`cornell` GPU 160²/256 spp and CPU
  96²/64 spp, plus `abs_hero_delta` GPU 200²/2048 spp); the earlier BDPT-hero validation scenes are
  unregressed (`scenes/absolute.ftsl` −0.008 %, `scraps/abs_hero_mats.ftsl` −0.000 %, both exactly as before);
  smoke sweep `mirror_selfie` / `group` / `material_presets` / `translucency` / `mixmat` clean.
  The backward tracer got both fixes (next bullet) and the forward tracers after it (the one after that), so
  every hero tracer now uses the max-over-live-λ rule.
- **Backward tracer (mode R): achromatic delta lobes keep the bundle, and EVERY Russian roulette is now
  max-over-live-λ — DONE 2026-07-26 (VERSION 0.63.0).** Two changes to `radianceHero` (`src/backward.h`) and
  its device twin `bkRadianceHero` (`src/render_cuda.cu`), kept 1:1:
  1. **Mirror / Filter / Glossy stop de-heroing.** Like the BDPT change above, their outgoing *direction* is
     λ-independent (a mirror reflects, a gel passes straight through, a glossy lobe is the mirror direction
     blurred by a λ-independent roughness), so only the per-λ coefficient differs. The unidirectional form is
     an *analog RR* rather than a throughput multiply, so the survival probability became `q = max_i c_i` with
     survivors reweighting by `c_i/q ≤ 1`.
  2. **Diffuse and DiffuseTransmit no longer roll the continuation coin on the hero alone.** The old
     `if (u >= rho[0]) die; thr[i] *= rho_i/rho_0` *amplifies* a secondary by up to `rho_max/rho_hero` — on
     the built-in `redWall`/`greenWall` spectra (0.05 … 0.75) that is a **15× weight spike** every diffuse
     bounce. Now `q = max_i rho_i` and `thr[i] *= rho_i/q ≤ 1`, so no λ is ever amplified. DiffuseTransmit
     picks its lobe from the per-lobe maxima `qR`/`qT` (proportionally rescaled in the rare case they sum
     past 1, which the per-λ energy guard allows).
  At `nUp == 1` both are the scalar code verbatim (`q == c[0]`, every reweight `*= 1.0`, same rng draws in the
  same order), so `-heroc 1` stays byte-identical.
  **The second half is what actually made hero pay off in mode R** — change 1 alone *regressed* luma. Noise
  RMS vs a 262144-spp `-heroc 1` reference, GPU, 256², as (luma / chroma):

  | scene | spp | single-λ | hero-4 before | +keepBundle only | **hero-4 after (both)** |
  |---|---|---|---|---|---|
  | `abs_hero_diffuse` (coloured Cornell, no delta lobes) | 4096 | 0.0604 / 0.0714 | 0.0605 / 0.0565 | — | **0.0254 / 0.0254** |
  | `abs_hero_mats` (glossy sphere + translucent leaf) | 4096 | 0.0875 / 0.1136 | 0.1566 / 0.1110 | — | **0.0398 / 0.0597** |
  | `abs_hero_delta` (gold mirror + Wratten-58 gel + coloured walls) | 1024 | 0.4197 / 0.2726 | 0.4148 / 0.2659 | 0.4287 / 0.1397 | **0.1853 / 0.0957** |
  | " | 4096 | 0.2072 / 0.1683 | 0.2021 / 0.1655 | 0.2530 / 0.1015 | **0.0940 / 0.0579** |
  | " | 16384 | 0.1122 / 0.0955 | 0.1062 / 0.0936 | 0.1305 / 0.0710 | **0.0549 / 0.0393** |

  Read the first two rows: **before this change hero was worth nothing in mode R on a coloured scene** (0.0605
  vs 0.0604 single-λ) and on the glossy/translucent scene it was actively *worse* (0.1566 vs 0.0875), because
  the ratio amplification ate the whole stratification win. After, it is ~0.42–0.52× RMS = a **4× variance
  reduction**, for 1.35× CPU / 1.36× GPU wall-clock (`abs_hero_delta` 4096 spp: 62.1→84.1 s CPU, 1.1→1.5 s
  GPU) ⇒ **≈2.9× at equal noise**. Energy is unbiased (`−0.015 %` … `+0.034 %` vs the reference).
  Diagnosis trail worth remembering: an achromatic-wall control scene (`scraps/abs_hero_delta_gray.ftsl`, same
  geometry, `whitewall 0.75` everywhere) showed change 1 alone going 0.0755→0.0406 luma with *no* regression —
  which pinned the regression on the coloured diffuse walls rather than on the delta lobes.
  Validated: `-heroc 1` byte-identical to `scraps/ftrace_base_6b8ca3f.exe` on **both** backends for
  `abs_hero_delta` and `abs_hero_mats` (256²/64 spp).
  **Methodology trap (cost an hour):** a reference rendered at 2× the test's spp *shares half its samples* with
  the test image — seeds are keyed on the absolute sample index (`seedUnit(rng, (sampleBase+s)*nPix+pixIdx)`),
  so a 32768-spp reference contains the 16384-spp render verbatim. `rms(test − ref)` is then silently
  discounted by ~0.7× **for whichever estimator produced the reference**, which flatters the baseline. Use a
  reference at ≥16× the test spp (or a different estimator).
- **Forward tracers (modes A/B/C + the M/S photon deposit): the same two fixes — DONE 2026-07-26
  (VERSION 0.64.0).** `tracePhotonHero` (`src/render.h`) and its device twin `shadeStepHero`
  (`src/render_cuda.cu`) got the identical pair of changes described in the previous bullet: Mirror / Filter /
  Glossy stop de-heroing (achromatic delta lobes keep the bundle riding), and every continuation Russian
  roulette — Diffuse, DiffuseTransmit and the new delta group — survives on `q = max_i c_i` with survivors
  reweighting by `c_i/q ≤ 1` instead of rolling the coin on the hero and reweighting by `c_i/c_hero`.
  **The forward tracers keep an energy ledger, which the previous bullet's tracer does not — and that
  ledger caught a real bug in the first attempt.** The reweight `beta[i] *= c_i/q` is *deterministic
  absorption*, so it has to be booked: without it `sum/emitted` fell to **0.6597** on
  `scraps/abs_hero_delta.ftsl`. Each reweight loop now does `e.absorbed += beta[i] * (1 - w); beta[i] *= w;`
  and the ledger closes at `1.000000` again. Booking it makes the ledger *exactly* consistent for the first
  time: absorb books `Σβᵢ` with probability `1-q`, survivors book `Σβᵢ(1-ρᵢ/q)` and carry `Σβᵢρᵢ/q`, and
  `(1-q)Σβᵢ + q·Σβᵢ = Σβᵢ` identically. The **old** ratio reweight never closed — it *created* ledger energy
  (`sum/emitted` 1.00281 `cornell`, 1.00459 `group`, 1.00172 `material_presets`, 1.00743 `mixmat`,
  1.00713 `abs_hero_diffuse`, 1.00696 `abs_hero_mats`; all now 0.999996 … 1.000003). Note this was a *ledger*
  inconsistency, not image bias — `E[ρ₀ · βᵢρᵢ/ρ₀] = βᵢρᵢ` is correct, so the old estimator was unbiased in
  expectation. What the amplification actually cost was **variance**: a very heavy-tailed weight
  distribution, which is why the old hero-4 mean luminance can still read −0.25 % off at 200 M photons.
  Noise RMS vs an 8 × 10⁹-photon `-heroc 1` reference, GPU mode B, 256², as (luma / chroma); every test at
  200 M photons except the last column, which is single-λ given the *same wall-clock* as new hero-4:

  | scene | single-λ | hero-4 before | **hero-4 after** | single-λ at equal time |
  |---|---|---|---|---|
  | `abs_hero_delta` (gold mirror + Wratten-58 gel + coloured walls) | 0.0460 / 0.0350 (2.9 s) | 0.0492 / 0.0342 (5.8 s) | **0.0289 / 0.0172** (7.3 s) | 0.0321 / 0.0228 (500 M, 6.8 s) |
  | `abs_hero_diffuse` (coloured Cornell, no delta lobes) | 0.0407 / 0.0563 (2.9 s) | 0.0514 / 0.0539 (6.1 s) | **0.0254 / 0.0216** (7.8 s) | 0.0269 / 0.0361 (540 M, 7.4 s) |
  | `abs_hero_mats` (glossy sphere + translucent leaf) | 0.0377 / 0.0587 (3.3 s) | 0.1069 / 0.0443 (7.1 s) | **0.0233 / 0.0239** (9.0 s) | 0.0247 / 0.0423 (545 M, 8.5 s) |

  As in mode R, hero-4 *before* this change was a net **loss** in the forward tracers — worse luma than
  single-λ at 2–2.5× the cost on all three scenes, catastrophically so on `abs_hero_mats` (0.1069 vs 0.0377,
  with the mean luminance still −0.251 % off at 200 M photons — heavy tails, not bias). After, it is a genuine
  win, but a **smaller one than mode R's ≈2.9× at equal noise**: forward hero shares only the *main path's* BVH walk, while the
  per-λ camera splat / photon deposit costs a full C×, so the equal-time margin is ~1.1× luma and 1.3–1.8×
  chroma on GPU (the chroma win is the point — that is what hero is for). CPU amortizes better
  (`abs_hero_mats` 20 M photons: single-λ 0.1148 / 0.0973 in 6.5 s, hero-4 0.0641 / 0.0333 in 12.3 s ⇒ 1.30×
  luma / 2.13× chroma at equal time). Mean luminance is unbiased (−0.007 % … +0.062 %, all within the tests'
  own noise). Validated: `sum/emitted = 1.000000` on both backends; `-heroc 1` byte-identical to
  `scraps/ftrace_base_c64cd9f.exe` on GPU (`abs_hero_delta` 50 M, md5 `c16c9efa…`) and CPU (3 M, md5
  `145db925…`); **mode M** photon deposit unbiased (`abs_hero_mats` 8 M photons, hero-4 mean +0.002 % vs
  `-heroc 1`, ledger 1.000000 both).
- **Photon-mapping modes M (photon map) + S (SPPM) — DONE (CPU).** `tracePhotonHero`'s map deposit now writes
  **all `nUp` live wavelengths** as per-λ photon records (`for (i<nUp) depositPhoton(h.p, ray.d, h.n, lam[i],
  beta[i]);` in `src/render.h`), and the shared `tracePhotonPass` (`src/photonmap_render.h`, used by M and S) sets
  `r.useHero` under the `kHeroC>1 && scene.media.empty() && !sceneHasGrin` gate. The gather keys off each photon's
  own λ (`photonmap_render.h:245`), so the heterogeneous-λ map gathers correctly; C records of `base/C` sum to
  `base` and `nEmitted` counts PATHS, so the estimate is energy-identical to single-λ. Cost: up to C× more stored
  photons from one shared BVH walk (the intended chroma-noise win). **Mode U (VCM/UPS)** got the same treatment
  later the same day — see the VCM bullet below.
- **VCM/UPS (mode U, CPU) — DONE 2026-07-26 (VERSION 0.69.0).** `src/vcm.h` now carries a `bdpt::HeroBundle`
  along **both** subpaths, so all four strategies (emission `s=0`, NEE `s=1`, vertex *connection*, vertex
  *merging*) evaluate per-λ. Design points worth remembering:
  * **One bundle is drawn per *path index*, not per subpath.** `vcmPass` pre-draws `bundles[nPix]` from a single
    stratified variate each and both the light-tracing and the camera-tracing worker index it by the same `i`,
    so light path *p* and camera path *p* share the same C wavelengths. That makes the **connection** strategy
    exact per-λ: `nUpConn = min(nUp_cam, nUp_lightVertex)`, sum over the shared λ, normalise by `1/nUpConn`.
  * **Merging stays keyed on the LIGHT vertex's own wavelengths.** A merge crosses paths, so there is no shared
    λ set; the estimator sums over the stored vertex's live λ and divides by its `nUp`, with the BSDF re-evaluated
    at each stored λ against the camera vertex. That is the pre-existing spectral-photon-mapping approximation
    generalised, not a new one.
  * **MIS weights stay the hero's** everywhere. Every *sampling* density in this renderer is λ-independent
    (cosine / glossy-lobe pdfs don't depend on λ; only throughput *values* do), so one `dVCM`/`dVC`/`dVM`
    bookkeeping triple serves the whole bundle — same argument as BDPT's single `misWeight`.
  * **The secondary payload is a parallel array** (`std::vector<LightVertexSec>` indexed in lockstep with
    `lightVerts`), never extra fields inside `LightVertex`. Stored light vertices are the dominant memory cost
    of a VCM pass (and of the GPU slab at ~`vcmCap·npix·128 B`), so `-heroc 1` must allocate exactly nothing
    extra — same reasoning as the GPU BDPT `pathSec[v*secStride+i]` split above.
  * `scatterSample` gained the **absolute** per-λ `secF[]` block copied verbatim from `bdpt.h::randomWalk`
    (including `keepBundle` for Mirror/Filter and the max-over-live-λ early-out), and Beer-Lambert absorption in
    the medium stack is now per-λ on both walks — that *is* the colour of coloured glass.
  Validated: `-heroc 1` **bit-identical** to the 0.68.1 binary on `cornell` mode U; `scenes/absolute.ftsl`
  (absolute mode = fixed sensor gain) C=4 vs C=1 at 1600 passes = **−0.001 %** mean, with self-noise RMS
  3.107→2.284; a new `scraps/_vcm_hero_gel.ftsl` (Wratten-58 gel pane + mirror slab — the `keepBundle` stress
  case) **+0.016 %** at 3200 passes with self-noise RMS 1.837→0.932 (**0.51×** = ~4× variance reduction).
  The GPU half followed the same day — see the next bullet.
- **VCM/UPS (mode U, GPU megakernel) — DONE 2026-07-26 (VERSION 0.70.0).** The device session lives in
  `src/render_cuda.cu` (NOT a separate `vcm_cuda.cu` — older notes said otherwise), and its two kernels became
  `kVcmLightT<NS>` / `kVcmCameraT<NS>`, templated on the SECONDARY slot count exactly like `kBdptT<NS>`. The
  `<0>` instantiation sizes every per-λ array at 1, so every hero loop compiles away to nothing, the C==1 λ draw
  is literally the old `dSampleSceneLambda(sc, rng, pdfL)` (rng stream untouched), and every hero *term* keeps
  the ORIGINAL floating-point expression order — hence `-heroc 1` is **bit-identical** (verified `cmp`-clean
  against the 0.68.1 binary on `cornell` mode U, 160², 64 spp). Design points:
  * **The secondary payload is a parallel device slab**, `lvSec[(i*vcmCap + k)*secStride + j]` with
    `secStride == C-1`, never inline in `DVcmLV`. The light-vertex slab is `npix · vcmCap · ~128 B` and is by
    far the largest allocation in a VCM session, so `-heroc 1` must allocate *zero* extra — same rule (and the
    same indexing shape) as GPU BDPT's `pathSec`. `kVcmCompactScatter` compacts the sec rows alongside the
    vertices.
  * **`DVcmSec` is only 16 B** — `{double beta; float lam;}`. CIE weights are *not* cached per secondary; they
    are recomputed as `cieX/Y/Z(row[q].lam)` at gather time, which is bit-identical to caching (the hero's own
    `DVcmLV::cx` is just `(double)cieX(lambda)`) and halves the slab.
  * **`lamBuf`/`invLamBuf` were widened to stride C** (`lamBuf[i*C + k]`) so the light kernel's bundle for path
    *i* is handed to the camera kernel for path *i* — that shared-bundle property is what makes the
    **connection** strategy exact per-λ (`nUpConn = min(nUp_cam, lv.nUp)`).
  * **Merging stays keyed on the light vertex's `lv.nUp`**, same approximation as the CPU side.
  * `dVcmScatter` gained the absolute per-λ `secF[]` / `secChromatic` / `keepBundle` block. The `<=0` terminates
    on Diffuse/Glossy/Mirror/Filter became max-over-live-λ tests, but **Grating's `r<=0` bail was kept** — it
    gates the RNG-consuming `gratingDiffract`, so removing it would desynchronise the device rng stream.
  Validated on an RTX 4090 at 200², 2048 spp against 32768-spp single-λ references (luma / chroma / wall-clock,
  C1 → C4): `absolute` 0.9366→0.8402 / 1.2373→**0.9817 (0.79×)** / 8.0→12.0 s; `abs_hero_delta`
  0.8714→0.7494 / 1.3584→**1.1138 (0.82×)** / 7.2→10.8 s; `abs_hero_diffuse` 0.8575→0.7502 /
  1.1000→**0.7961 (0.72×)** / 8.7→13.2 s; `abs_hero_mats` 0.9104→0.7612 / 1.2459→**0.9179 (0.74×)** /
  11.6→19.6 s. Bias on `scenes/absolute.ftsl` at 4096 passes, C4 vs C1: **+0.011 %**. CPU vs GPU hero VCM
  (200², 1024 passes, `-heroc 4`) agree to **0.028 %** — and the GPU is 31× faster (5.9 s vs 182.6 s).
  *Reading the bias numbers:* `abs_hero_diffuse` shows +0.396 % (C1) / +0.418 % (C4) against its reference, but
  that is the progressive-radius estimator's own convergence bias at 2048 vs 32768 passes — the C4-minus-C1
  delta is only +0.022 %.
- **Opt-in split-at-dispersion (`-herosplit`) — DONE 2026-07-26 (VERSION 0.65.0), CPU forward.** The alternative
  to the default de-hero policy: at a dispersive interface all C wavelengths **continue**, each running the same
  interaction with its **own** λ (its own Snell direction / grating order / Stokes shift), so one bundle fans out
  into C independent monochromatic sub-paths. Both estimators are unbiased; splitting resolves a prism / rainbow /
  dispersive caustic's chromatic spread *geometrically per photon* instead of stochastically across many photons.
  * **The implementation trick.** The bounce loop of `tracePhotonHero` (`src/render.h`) was extracted verbatim
    into `tracePhotonHeroLoop(..., ray, stk, lam, beta, secAlive, bounce0, ...)`, and the split branch
    **re-enters that method recursively**, once per secondary, with `secAlive = false` and `bounce0 = bounce+1`.
    That keeps all ~20 `return` sites in the loop body working unchanged — no explicit work stack, no CPS
    rewrite. Because the branch is guarded on `secAlive`, a sub-path can never re-split, so recursion is at most
    **one level deep**: cost is linear in C, not exponential, and the per-frame footprint (one `MediumStack` copy
    + two `kHeroMax` double arrays, ~600 B) is bounded. Each sub-path gets its **own copy** of the
    `MediumStack`, since that is exactly where the sub-paths diverge inside the glass.
  * **Ledger.** No ×C boost: the C sub-paths keep `base/C` each and the parent zeroes `beta[i]`, so the total
    equals what the de-hero'd hero would have carried alone and every sub-path books its own terminal fate
    (`interactPhotonSpecular` already books `e.absorbed += beta` on every `return false`).
  * **Plumbing.** `hero::gSplit` (in `src/hero.h`) is a single global policy flag set once by `main()` during
    argv parsing, and `Renderer::heroSplit` default-initialises from it. That is deliberate: `heroC` has to be
    threaded explicitly because the drivers vary it per pass (the meter pre-pass, the media/GRIN/lens gate),
    but a whole-run policy choice does not — so modes `A`/`B`/`C` **and** the `M`/`S` deposit picked it up with
    **zero** call-site churn (`photonmap_render.h`, and `sppm_render.h` through it, just construct a `Renderer`).
  * **Validated** on a new scene `scraps/abs_herosplit.ftsl` (absolute-exposure Cornell + `glass:SF10` flint
    sphere, mode B, 256², fixed gain 6, CPU), against a 200 M-photon reference:
    - flag **off** byte-identical to `scraps/ftrace_base_cc20a46.exe` (3 M photons, md5 `e2eef2cb…` both);
      `-heroc 1 -herosplit` byte-identical to plain `-heroc 1` (md5 `2b8f0fe8…` both);
    - flag **on** `sum/emitted = 1.000000` **exactly** (0.999990 off), mean luminance −0.026 % vs the
      reference (−0.054 % for off) ⇒ both unbiased;
    - **equal wall clock (180 s each):** caustic-region noise RMS luma 0.0340→**0.0303 (0.89×)**, chroma
      0.0386→**0.0271 (0.70×)**; whole frame 0.0579→0.0532 (0.92×) and 0.0417→0.0333 (0.80×).
      (`scraps/region_rms.py`, new — whole-image RMS is dominated by the flat diffuse walls and hides what
      a change did to the caustic, so it reports a box as well as the full frame.)
    - **cost 1.11×** per photon (20 M photons back-to-back: 173.3 s off, 192.3 s on). The split PNG is 6 %
      *smaller* (91 126 vs 97 040 B) — less caustic noise compresses better.
    **Methodology warning (cost this session ~40 min):** an early measurement read 1.54× (187 s vs 122 s)
    because the two runs were ~25 min apart and the machine's throughput drifted by 1.6× in between. Never
    compare wall clock across runs separated in time — run the two policies **back-to-back**, or give both
    the same `-time` budget and compare photon counts.
  * **Not covered:** the GPU forward tracer (execution divergence as the fan-out wavelengths take different
    branches, plus the emission back-pressure / fixed work-pool needed to keep the device photon buffers
    bounded), and modes `R` / `D` / `U`. Also **not** applied at `Mix` — see the next bullet. (`Layered`
    was in the same boat until v0.115.1, which keeps the bundle alive across a coat via a per-λ weight
    and a shared coin instead; see the layered de-hero entry near the top of this file.)
- **Three known approximations in the CPU hero path** (all minor, documented for when they're revisited):
  - **A zero hero throughput kills the whole bundle — FIXED everywhere (2026-07-26).** `bdpt.h`'s
    `randomWalk` (absolute `secF[]` + max-over-live-λ early-out), `backward.h`'s `radianceHero` /
    `bkRadianceHero`, and `render.h`'s `tracePhotonHero` / `shadeStepHero` all now survive on
    `q = max_i c_i` rather than on the hero's own coefficient, so a surface that is exactly black at λ₀ but
    coloured at λ₁ no longer drops the secondaries' contribution there. What remains unfixed is the deeper
    issue this was a symptom of: the λ pdf is still a scalar hero pdf rather than a per-λ spectrum, so
    hero sampling still cannot MIS across wavelengths. The proper fix is PBRT-v4's formulation — carry the
    pdf itself as a per-λ spectrum and MIS across wavelengths — which is an architecture change across
    every mode, not a local patch.
  - **Mix material stays multi-λ with a shared child selection.** Exact for constant mix weights; for *spectrally
    varying* mix weights with diffuse children it introduces a small bias (the child is picked by the hero λ's
    weight, secondaries ride along). Acceptable vs. de-heroing every Mix; revisit if a spectral-mix scene shows it.
    **`-herosplit` does not cover it.** This is a λ-dependent *decision*, not a λ-dependent *direction*, and its
    natural split point sits *before* any interaction has happened — which does not fit
    `tracePhotonHeroLoop`'s "resume from a ray" entry shape the way the dispersive case does (there each
    secondary can just re-run `interactPhotonSpecular` at the same hit with its own λ and hand back a fresh ray).
    Extending split to it needs a "resume at this hit with this material" entry point; worth doing if a
    spectral-mix scene ever shows the bias, but nothing observed yet.
    `Layered`'s coat used to be listed here for the same reason, and it was worse than a bias — it de-hero'd
    outright and mistinted the whole frame at 1 spp. v0.115.1 fixed it without needing a resume-at-this-hit
    entry point: the coat decision is made **per λ against one shared coin**, the bundle rides through with a
    per-λ weight whenever the live λ agree, and the backward loop fans out into monochromatic sub-paths
    re-entering the *same* vertex when they don't. That re-entry trick works for a coat precisely because it
    happens before any interaction — nothing has been deposited or absorbed yet, and a wide bundle is never
    inside a medium. The same approach would work for `Mix`, and is the obvious next step if it ever matters.
  - **Equal-*time* benefit is geometry-dependent.** Hero shares one BVH walk across C wavelengths, so its win grows
    with scene complexity. On trivial geometry (Cornell: a few quads + 2 spheres) traversal is nearly free and the
    4× per-λ shading makes hero ~1.6× slower per spp, so at *equal time* single-λ can edge it there. On heavy
    geometry the shared traversal amortizes and hero wins outright. This is expected hero behaviour, not a bug.

### TECH-DEBT (2026-07-20): forward mode B barely converges the water-droplet rainbow (`scraps/rainbow_test.ftsl`); mode D works well
The Airy rainbow phase model (`src/rainbow.h`) is **verified correct** — `ftrace -rainbow-selftest`
reports textbook geometry (primary 40.7°–42.5° antisolar, secondary 50.1°–53.4° with reversed
colour order, water dispersion violet n=1.343 > red n=1.330, phase normalised 2π∫p dμ=1.000). But
rendering the bounded rain-curtain scene `scraps/rainbow_test.ftsl` (thin fog slab + distant sun,
camera toward the antisolar point):
- **mode B (forward pinhole splat):** after 450 M photons / 75 s the image is essentially **black**
  (`sensor=0.0000`, ~95 % noise, auto-exposure ~1.9e-10). The single-scatter-to-camera path over the
  scene's hundreds-of-metres scale is too rare for the forward splat to accumulate the bow.
- **mode D (BDPT):** converges the **full, clearly-coloured bow** in the same 75 s (~7.5 % noise) —
  primary + secondary + Alexander's dark band all visible (`png/rainbow_rain.png`). The fogbow
  variant (`scraps/fogbow_test.ftsl`, 10 µm droplets) likewise renders correctly as a broad,
  supernumerary-dominated pale bow (`png/fogbow.png`).
So this is not a physics bug — the machine and mode D are fine. It is a **forward-mode
efficiency gap**: mode B (and likely A/C) can't practically image a thin-medium single-scatter bow
at this scale. **Proper fix (deferred):** give the forward medium-scatter path a next-event /
camera-connection term for participating media (so each scatter vertex connects to the camera with a
proper importance weight, like BDPT does) instead of relying on the plain pinhole splat, or document
that rainbow/fog-bow scenes should use mode D. Low priority — mode D covers the use case.
(Note: a compact, dense, short-range fog *does* image in mode B — see `scenes/_rainbow_test.ftsl`,
a room-scale droplet box flooded by a wide collimated beam, which shows a full centred bow in mode B
though noisily. The gap is specifically thin/large-scale atmospheric slabs.)

### ENHANCEMENT (2026-07-21): photon beams for a CHEAP, clean rainbow/volumetric FLYBY (shared deposit + per-camera gather) — **DONE (single-scatter long-beam, `-beams`, CPU + GPU)**
**DONE 2026-07-21.** Shipped as the `-beams` (alias `-photonbeams`) CLI flag on the shared forward
mode-B pass. The deposit/gather decoupling is implemented as an **unbiased single-scattering
long-beam estimator** rather than the full Jarosz beam-BVH (see "How it was actually built" below);
this is the correct trade for a crisp view-dependent bow (rainbows/fogbows/glories ARE single
scatter). Validated on `scenes/_rainbow_beams_decorr.ftsl` (two identically-placed mode-B cameras):
baseline shared pass → camA/camB **bit-identical** (frozen speckle, the flaw); `-beams` → camA/camB
**decorrelated** (93.6% of pixels differ) with a **matching mean** (123.29 vs 123.28) — i.e. same
bow, independent per-frame noise, unbiased. `[energy] sum/emitted=1.000000`. The original problem
statement and design are kept below for reference.

**Problem.** A rainbow (or any volumetric single-scatter effect: fogbow, glory, crepuscular rays)
is **view-dependent single scattering** — the phase angle θ is measured *to the eye*, so every
camera/frame sees a different bow. On a flyby this forces a per-frame cost with today's integrators:
- **Forward shared multi-camera pass** (`renderForwardShared`, `main.cpp` ~1264) traces ONE photon
  flight and splats every volume vertex to *all* cameras — 1× photon cost for the whole flyby, and
  the bow is per-camera-correct (each connection uses that camera's angle). BUT all frames ride the
  *same* photon realisation → one **frozen noise/speckle pattern** baked into every frame, which
  looks wrong in a video (hence a `camera_path` with `exposure_lock` is deliberately rendered
  UN-shared, `main.cpp` ~5579-5602).
- **Un-shared forward / mode D (BDPT) per frame** → independent per-frame noise (good video), correct
  bow, but **N× cost** (retrace the whole light transport every frame). Mode D is the quality route
  today (the entry above); it is inherently per-camera and cannot share across frames.
- **Mode M (surface photon map) is the WRONG tool** and cannot help even if volumetric scattering were
  added to it: photon mapping's payoff is caching the **view-independent** multiple-scatter / indirect
  solution, but the bow is view-dependent **single** scatter — there is nothing view-independent to
  cache about it. Mode M also currently has *no* participating-media scattering at all (only the
  nested-dielectric IOR/Beer-Lambert stack, `photonmap_render.h`), so it renders scattering fog as if
  it weren't there.

**The fix — photon beams (Jarosz et al. 2011, "progressive photon beams").** Store each photon's
*path segment through the medium* as a BEAM (origin, direction, power, per-λ), which is **view-
independent** — deposit the beam set ONCE and reuse it for the whole flyby. Then each camera
ray-marches its primary rays and gathers in-scattered radiance from the nearby beams, evaluating the
droplet phase `p(θ,λ)` (`src/rainbow.h`) toward *its own* eye. This **decouples the expensive light
deposit (shared, 1×) from the per-camera gather (independent noise, correct per-view angle)** — i.e.
the "fast AND best" combination: ~1× photon cost across the flyby, clean non-frozen per-frame noise,
correct view-dependent bow. It is essentially a noise-decorrelated version of the shared forward pass.

**Scope / cost.** A substantial new volumetric integrator: a beam data structure + acceleration
(beam BVH or the standard photon-beam grid), a ray-march-and-gather estimator on the camera side, the
spectral/hero plumbing to carry per-λ beam power, and wiring into the flyby/checkpoint machinery
(and ideally the GPU forward path). Only worth building if rainbow/fogbow/volumetric flybys become a
recurring need. **For a one-off showcase, use mode D per frame (best quality, works today).** Related:
the "forward medium-scatter next-event/camera-connection" proper-fix in the entry above is a smaller
step that would make mode B converge the bow (and ride the shared pass for 1× cost) but does NOT solve
the frozen-noise-in-video problem — only the deposit/gather decoupling of photon beams does.

**How it was actually built (2026-07-21).** Instead of a stored beam data structure + BVH, the
implementation exploits the fact that a rainbow is *single scatter* and folds deposit-and-gather into
the existing shared forward photon trace, decorrelated per camera:
- **Photon crosses the medium STRAIGHT** in `-beams` mode (`tracePhoton`, `src/render.h`): the analog
  in-medium collision sampling is skipped (`doBeamGather = beamGather && nCam>1 && !forwardCatch &&
  !media.empty()`), so the photon carries the full light path up to the medium and is attenuated by
  extinction (`beta *= mediaTransmittance(...)`, the loss booked to `e.absorbed` so `sum/emitted=1`).
  No analog scatter/redirect ⇒ no correlated realisation shared across cameras.
- **Each camera independently draws its OWN single-scatter point** along the crossing with a
  per-photon `Pcg32 crng` (seeded from the photon's RNG): `sampleMediaCollision` over `[0,dSurf]`,
  then splat via `connectVolume` (+ `camSpecularSplatVolumeAll`). This is unbiased for single scatter
  because the free-flight collision pdf's `Tr` cancels `connectVolume`'s `albedo·phase·T_cam·β`, so
  `E = I(dSurf)` = the exact single-scatter in-scatter integral, and each camera's independent RNG
  gives independent (non-frozen) noise. `connectLensVolume` used under `-lens`.
- **Deliberately omits multiple scattering** (the desaturating haze wash the baseline shows) — that is
  what makes the `-beams` bow visibly *crisper* than the shared baseline, at slightly lower total
  brightness (single-scatter only). This is the intended quality trade for a clean view-dependent bow.
- **Wiring** (`src/main.cpp`): `static bool g_beamGather`; `-beams`/`-photonbeams` flag; `beamGather`
  param threaded into `renderForwardShared` → `r.beamGather`. Rides the existing shared
  chunking/checkpoint/`-resume` machinery unchanged. Bonus: single-scatter photons terminate at the
  medium so the pass is *faster* than the full analog trace.
- **GPU port — DONE 2026-07-22.** The per-camera in-scatter resample now runs in the CUDA forward
  tracer too, so `-beams -device gpu/auto` no longer forces CPU. Wiring: `DCamSet::beamGather`;
  `shadeStep` takes an independent per-photon `DRng* crng` and, when `beamGather && nCam>1 &&
  camMode!=C && mediaN>0`, skips the analog medium redirect and instead loops the cameras — each
  draws its own free-flight collision (`dMediaSampleCollision` on `crng`) and splats via
  `connectVolume`/`connectLensVolume` + a 1-camera-slice `camSpecularSplatVolumeAll`, then the photon
  is attenuated by `dMediaTransmittance` over the whole crossing and continues straight. `crng` is
  seeded once per photon in `kTrace` from the main stream (so non-beam renders are bit-unchanged).
  Beams force the megakernel (the wavefront pool carries no per-photon beam stream), like hero.
  `renderForwardSharedCuda` gained a `beamGather` arg. Validated on `scenes/_beams_hg_decorr.ftsl`
  (two identically-placed mode-B cameras, HG fog): GPU baseline camA==camB bit-identical (RMS 0);
  GPU `-beams` camA≠camB (RMS 48.7) with matching means (0.06% apart) — same decorrelation magnitude
  as CPU (RMS 48.7); GPU-vs-CPU beam means agree to 0.02% and the energy split matches (absorbed
  0.6914 vs 0.6911). Spectral-rainbow-phase media are still CPU-tabulated (`cudaForwardSupported`
  rejects them) so a rainbow/fogbow scene still falls back to CPU — HG/Rayleigh fog runs on GPU.
- **Not done / future:** multiple-scatter beams (full Jarosz) and the stored-beam BVH (only needed if
  a use case wants view-independent beam reuse beyond the shared-pass model), and the GPU wavefront
  backend for beams (megakernel-only today). Mode M/D untouched.

### RESOLVED (2026-07-21): `scenes/gallery_settled.ftsl` OOMs — but ONLY when the 600-frame flyby is in the camera selection
**FIXED 2026-07-21.** The real root cause was narrower than the per-render-target
`Film` note below: `struct Camera` (`src/camera.h`) *embeds a full `Film film;`
member*, and `Camera::lookAt()` called `film.alloc()` on it. Every selected camera
(600 `toRender` RenderCams **plus** 600 `meterPlan` MeterCams for the exposure_lock
pre-pass) therefore carried a live 960×540 film (~16.6 MB) → ~20 GB of allocations
*before a single photon was traced*. But nothing ever reads a `Camera` instance's
embedded `xyz`/`hits` — `project()`/`genRay()`/`pixelPlaneArea()`/`pixelSolidAngle()`/
`genLensRay()` only read `film.resX`/`resY` metadata; every actual render path
(`renderForward`, `renderPhotonCamera`, `renderForwardShared`'s per-thread targets,
the backward tracer, checkpoints) owns a *separate* `Film`. Fix: drop the
`film.alloc()` in `lookAt()`, keep only `film.resX/resY`. Each camera falls from
~16 MB to a few hundred bytes; verified the flyby now plateaus at ~938 MB (was
~20 GB) and renders correct, energy-conserving frames. Bit-identical to baseline
(only an unused allocation removed). This is cleaner than the "chunk the film
allocation in batches" workaround floated below — it fixes *all* multi-camera
renders, not just this scene. `render_gallery_flyby.bat` drives the full 600-frame
mode-M flyby → `png/gallery_settled_fly/gallery_settled.mp4` (cap photons at ~2M per
the mode-M shared-build caveat elsewhere in this file).

Original report follows.

### BUG (2026-07-19; root-caused 2026-07-20): `scenes/gallery_settled.ftsl` OOMs — but ONLY when the 600-frame flyby is in the camera selection
`error: bad allocation` (exit 1) rendering `scenes/gallery_settled.ftsl`.
**Root-caused 2026-07-20 — it is NOT a generic "scene too heavy" OOM.** With 26 GB
free on a 64-bit build, ~1.5 M scene tris and a 1280×720 film cannot exhaust
memory; the `bad_alloc` is a single *pathological* allocation that scales with the
**number of selected cameras**, and the scene defines a `camera_curve "fly"` of
**600 frames**. Reproduction matrix (all on this machine, after the 2026-07-20
Klein→gyroid-lite swap):
- `-camera cam` (single still)  → **works** in every path: mode D GPU render
  (1280×720, 5.4 s), CPU `-raster -seethrough` (1440×810, 5.08 M tessellated tris),
  and the interactive `-explore -seethrough` fly viewer.
- no `-camera` (selects `cam` + all 600 `fly` frames) → `bad_alloc` right after
  `[selftest]`.
- `-camera fly` (the 600-frame flyby alone) → `bad_alloc` right after
  `[camera] path 'fly' -> 600 frames`.
So the old "OOMs at default resolution" claim was misleading: the *still camera*
renders fine at default res; only pulling the whole flyby into one selection blows
up. The allocation is proportional to the frame count (each selected camera gets
its own `Film` accumulator — see `renderForwardShared`, `main.cpp` ~1240/5590, and
the "~3 GB of films" note at ~5845), so 600 simultaneous films (or a similar
per-camera structure built during camera expansion) overflow.
**Workaround (works today):** always pass a single camera, e.g.
`ftrace -in scenes/gallery_settled.ftsl -camera cam -explore -seethrough`, or
`-camera fly` **with** a frame selector so only a handful of frames are live at
once. **Proper fix (deferred):** chunk the multi-camera / flyby film allocation so
a long `camera_curve` renders in bounded-size batches (render K frames per photon
flight, recycle the film buffers) instead of allocating one film per frame up
front — mirror the "one-frame of host RAM" strategy already noted at ~5845 for the
budgeted path. Not blocking (the still camera renders the showcase fine); the
flyby just needs the batching fix before it can render all 600 frames in one
invocation on this machine.

### RESOLVED (2026-07-19): stale `absorb 3 0.5 0.3` example in `src/ftsl.h` — untagged spectrum triples don't parse
The comment above the `dielectric` `absorb` handling (`src/ftsl.h` ~1838) read
`e.g. `absorb 3 0.5 0.3` (per-channel, upsampled)`, implying a bare/untagged numeric triple is a valid
spectrum expression. It is **not**: `evalSpectrum` only accepts a *single* number as a constant;
a 3-word run with a numeric head (`3 0.5 0.3`) fell through every branch to
`fail("unrecognized spectrum expression '3'")`. Verified empirically — a scene with `absorb 3 0.5 0.3`
(and likewise `reflect 0.8 0.7 0.2`) errored out with `[ftsl] unrecognized spectrum expression '<head>'`.
The valid spellings are a scalar (`absorb 0.5`), a tagged colour (`absorb rgb 3 0.5 0.3`), or a named/ref
spectrum. **Fixed:** (1) corrected the comment to the tagged example plus an explicit "the `rgb` tag is
required" note; (2) added a targeted parser hint — when the head is numeric and every word in the run is a
number (`w.size() >= 3`), `evalSpectrum` now fails with `… — a bare numeric triple isn't a spectrum; tag it,
e.g. \`rgb 3 0.5 0.3\`` instead of the opaque `unrecognized spectrum expression '3'`. (VERSION 0.9.2 → 0.9.3.)

### RESOLVED (2026-07-19): loom's shared-grammar `reflect`/`roughness`/`*_map` fields are now shape-validated
The grammar-backed reader (`tools/loom/loom/grammar/reader.py`) shape-checks *purely-spectral* fields
(`ior`/`transmit`/`absorb`/`substrate_k`/`emit` on materials, `spd` on lights) against the shared spectrum
grammar (`loom/grammar/spectrum.py`). The binding-union fields `reflect`, `roughness`, and any `*_map` binder
are now validated too, via new `loom/grammar/bindings.py` — a faithful mirror of ftrace's `bindReflectTexture`
/ `bindScalarTexture` / `bindScalarPattern` + `spectrumParam` / `dblParam` (`src/ftsl.h` `buildMaterial`):
`as_color_binding` (`reflect` = `texture:<name>` | spectrum — reflect binds only a UV texture, never a
`pattern:`), `as_scalar_binding` (`roughness` = `pattern:` | `texture:` | one scalar number, not a spectrum),
and `as_map_binding` (`film_thickness_map`/`weight_map` = `pattern:` | `texture:` only). Wired into
`_build_material` (`_validate_bindings`); shape-only (a bound name's scene membership stays a later,
scene-aware check). An untagged triple `reflect 0.8 0.7 0.2` is now rejected exactly as ftrace rejects it.
`tests/test_grammar_bindings.py` (10 cases) + `test_grammar_material.py` additions; loom suite 823 passed.
**Remaining scope (not this item):** the record-driven *whole-material override* block (`from R(...)` +
`slot = REC.chan`, ftrace's `isRecordOverrideBlock`) is a distinct block form loom does not emit, and the
final step is mirroring the `value`/`spectrum`/binding grammar into ftrace's C++ front-end at the J3c port.

### RESOLVED (2026-07-19): loom `Light(color=…, size=…, turbidity=…)` emitted fields ftrace's `addLight` ignored
loom's `Light` accepted free-form props and emitted them verbatim (`light { kind …  color 0.9 0.8 0.7  size 2 2 }`),
but ftrace's `addLight` reads emission from `spd` (a spectrum) plus per-subtype geometry
(`origin`/`u`/`v`/`center`/`radius`/…) — it has **no** `color`, `size`, or `turbidity` field, so those loom props
were silently dropped at render. **Fixed:** loom's `Light` now maps `color=(r,g,b)` → `spd rgb r g b` (a spectral
emission, since ftrace lights are spectral) and otherwise passes only ftrace-valid props through; the test battery
dropped its `size`/`turbidity` fixtures and uses ftrace's real light schema (`u`/`v` area geometry,
`center`/`radius` sphere light). `size`/`turbidity` were demo-only and are gone — if loom wants to author a light
it uses ftrace's language. (`test_light_color_emits_spd_rgb` locks the mapping; full loom suite green, 824 passed.)
The analytic-sky *feature* that would give `turbidity` meaning is deferred — see TODO ("Analytic physical sky").

### RESOLVED (2026-07-19): `gallery_settled.ftsl` — several objects mis-positioned (resting on the floor)
**Root cause:** the free-settle physics bake tumble, not a transform/collider/floor bug. Three of the
five settled pieces landed correctly; klein (COM y≈0.37) and heart (y≈0.14) had tipped, rolled past
the rim of their NARROW museum pedestals, and fallen to the floor. Verified by recovering each piece's
settled COM height from the baked `group` delta (`pos_sim = R·c_auth + t`; see
`scraps/check_settle_heights.py`) — confirming the transform math and colliders were correct and only
the two tippy shapes had walked off their caps. **Fix:** added a during-sim horizontal restoring
spring to `tools/settle_scene.py` (`--tether [k]`, default k=150) applied AT each body's COM every
step, pulling it back toward its authored XZ. Because it acts at the COM it exerts **zero torque**, so
a piece is still free to tip/rotate onto its cap but cannot walk off it sideways — the result is a
genuine physics rest pose that stayed home. At k=150 all five pieces settle onto their stands in one
pass (klein rescued 0.37→1.38, heart 0.14→1.12). The committed `gallery_settled.ftsl` was repaired by
swapping the five tumbled `group` deltas for the tether-settled values. (Distinct from the separate
OPEN mode-D GPU-BDPT launch-failure on the same scene, logged 2026-07-15.)

### TECH DEBT (2026-07-19): `scenes/gallery_settled.ftsl` has diverged from its authored source `scenes/gallery.ftsl`
The generated settled file was hand-edited by three later commits (4c86261 prefer{}/else{} camera
mode-D/B fallback, f0786d6 flyby `frames 600`, bc9d664 scene-level `default_mode` + fps hint) that were
**never back-ported** to the authored `scenes/gallery.ftsl` (still plain `camera mode D`, `frames 144`).
So the "generated" file is now the de-facto source of truth for those features, and re-running
`tools/settle_scene.py --scene scenes/gallery.ftsl` would silently REGRESS them. That is why the
2026-07-19 floor-bug fix above patched the five `group` deltas in place instead of regenerating. **Proper
fix:** reconcile the two — back-port the camera/frames/default_mode authoring into `gallery.ftsl` so the
settled file is fully regeneratable from source, then regenerate with `--tether`. Blocked on confirming
authoring intent (is `frames 600` or `144` canonical? is the prefer/else wrapper wanted in source?).

### TECH DEBT — DONE (2026-08-03, v0.121.0): settle sim slow — stand colliders won't decimate below ~50–115k tris
`tools/settle_scene.py` decimates static concave colliders to `STATIC_TRI_CAP` (4000) via
`trimesh.simplify_quadric_decimation`, but the marching-cubes museum-stand meshes bottom out far above
the cap, keeping per-step collision cost high. Clean closed meshes (gyroid, lamp, chrome_ring) hit 4000
fine. Worked around by validating at `--mesh-res 64`.

**Measured (2026-08-03).** A full gallery bake took **40+ minutes**; the sim ran at **60 ms/step** for
five dynamic bodies.

Three of the original hypotheses were wrong and are recorded so they aren't re-tried:

* *"The stands are non-manifold / multi-shell, so decimation gives up."* **No** — every stand mesh is
  watertight, has no duplicate or degenerate faces, and `merge_vertices` changes nothing. Cleaning the
  mesh first makes exactly zero difference.
* *"Decimation just needs more passes or more aggression."* **No** — iterating
  `simplify_quadric_decimation` to convergence takes stand_oil 207100 → 51354 (6 passes) and stand_glass
  372940 → 97046 (12 passes), while `aggression=12` is no better than the default 7. It is a genuine
  wall, and the iterated result has already lost **29% of the mesh volume**, so pushing harder would
  trade a bad collider for a wrong one.
* *"Cost is dominated by total static triangle count / the BVH is missing."* **No** — with the dynamic
  bodies teleported 60 m away the same 3.6 M-triangle static set steps in **0.01 ms**. The broadphase and
  BVH are working fine; the cost is entirely **contact-manifold generation against the thousands of
  marching-cubes slivers directly underneath a resting piece**. Reducing triangles helps only because it
  reduces triangles *in the contact patch*.

**Fixed** by `slab_hulls()` (`tools/settle_scene.py`): a static collider that will not decimate is
decomposed into 32 horizontal slabs, each replaced by the convex hull of its own vertices (slabs overlap
by one polygonisation cell so no seam gap opens). Verified on all ten gallery stands: cap-top height and
XZ extent are reproduced **exactly**, at ~4000 tris per stand instead of 50–115k. Decimation is still
preferred when it works, since it keeps the concave shape.

A single whole-mesh convex hull was rejected even though it is faster still (0.30 vs 1.5 ms/step) and
also preserves the cap top exactly: it fills in the taper between a wide base and a narrow column,
inventing a sloped shoulder a piece could come to rest on — which would be baked into the scene as a
piece floating in mid-air beside its stand. VHACD on the stands was rejected too: it moves the cap top
by 5 mm and *under*-estimates stand volume by 19%.

Also fixed alongside it:
* **Caching** (`scraps/.settle_cache/`, `--no-cache` to bypass). The `-export-mesh` polygonisation is
  keyed on (scene text, `--mesh-res`, ftrace mtime) and the VHACD proxies on the proxy mesh's content
  hash. Re-running a bake with a different `--tether`/`--jitter`/`--seed` — the normal iteration loop —
  now skips both entirely.
* **Per-phase timing** is printed (`attempt N phase M: k/8000 steps in Ts (X ms/step)`), including an
  explicit `<-- hit the cap, did not settle` marker, so a slow or non-converging bake is visible rather
  than silent.

**Result: 40+ min → 70 s**, sim at 0.5–0.67 ms/step (~100×), with all five hero pieces reporting `OK`.
`--mesh-res` is *not* the lever it appeared to be — it only helped as a side-effect of the broken
decimation, and can now stay at full resolution.

### DONE (2026-08-03): the Klein bottle mesh has no stable upright rest — it cannot stand on its pedestal
`meshes/klein_hunyuan.obj` as placed in `scenes/gallery.ftsl` is correctly positioned — bottom at
y=1.018 over stand_klein's cap top at y=1.00, footprint 0.34×0.33 m on a 0.52×0.52 m cap — but a settle
dropped it on the floor every time (the last bad bake moved it `translate 1.06 3.90 -0.005 rotate
-86.4 -51.4 -68.2`).

**Root cause, measured.** Not a placement or simulation bug. Take the convex hull of the placed mesh
and keep the faces whose supporting plane has the COM over them: those are *exactly* the orientations
in which the piece can rest on a plane. There are **44, and the most upright of them leans 73°**. The
shape has no near-upright equilibrium at all, so **nothing it merely rests on can hold it up** — only a
mount that *grips* can. (An earlier version of this entry blamed a "47×42 mm contact blob with the COM
43 mm outside it". That described a symptom of one particular pose, not the cause, and it sent the fix
down several dead ends.)

**Five mount families were built and falsified by measurement**, each for a distinct geometric reason:

| mount | why it fails |
|---|---|
| circular seat / bore rim | the section radius about the pedestal axis swings 25→130 mm, so a *circle* touches exactly 2 lobes 180° apart — a knife edge with the COM on the line between them |
| spherical dish | a sphere is the one surface on which rolling is free; it rolled off |
| sleeve / cage above the cap | the flank **narrows downward**, so leaning always *opens* clearance; the lean ran away 2→6→11→19→35→65° |
| conforming height-field cradle | the underside is a paraboloid ρ²/244 mm, and a sphere rolls freely inside its own negative, so tilt is not resisted; also numerically pathological — a mesh colliding a surface coincident with it produced 1400–1700 N of penetration recovery on a 9.81 N piece |
| discrete museum posts | a support point needs a near-horizontal surface normal, and the underside is only shallow within ρ≈70 mm — barely past the COM's own 67 mm offset |

**Fix (shipped): a shaped tapered collar**, `meshes/collar_klein.obj`, generated by
`tools/make_klein_collar.py` and referenced from `scenes/gallery.ftsl` as the `collar_klein` mesh
object. The flank widens upward, so a bore cut to the piece's *own cross-section* captures it: it
cannot sink because the taper jams, and its weight is carried all round the perimeter instead of on two
lobes. Wedging was never the bug — it is the mechanism; the bug was that a *circular* bore wedges
against only two points. The bore never re-narrows going up, so the piece still lifts straight out (not
captive, the museum rule). Result: rests at 0.77° lean, `on collar_klein`, poke drift **1.6 mm** against
settle_scene's POKE_TOL of 10 mm.

**Two traps worth remembering, both of which produced confident wrong answers:**

1. *Harness friction must match the tool.* Every early sweep hard-coded `rollingFriction=0.002` /
   `spinningFriction=0.02` against settle_scene's actual `5e-4`/`5e-4` — 4× and 40× too much. That
   alone made several seats look stable in the harness and fail in the bake. All harnesses now
   **import** `ROLLING_FRICTION` / `SPINNING_FRICTION` from `settle_scene`.
2. *A bore sampled as radius-per-azimuth is not the outline.* The first shipped collar lofted
   `r[level, azimuth]` samples. That is safe (it can never cut into the piece) but it fills in every
   radial concavity, and this section is strongly non-star-shaped about the pedestal axis: the polar
   bore came out **1.49× the true 0.5 mm offset at y=1.090**, 114 cm² of void. It rested at 20.1° lean
   and held 12/36 pokes. Cutting the bore from the outline *polygon* instead gives 36/36. Scored on
   settle_scene's own 10 mm POKE_TOL, not a looser threshold, because that is the number that prints
   TOPPLES.

### TECH DEBT — MOOT (2026-08-03): the VHACD proxy for `klein_hunyuan.obj` is a poor fit, so the collar shows a visible gap
**Closed the same day it was opened, by deleting the collar.** `klein_hunyuan.obj` is no longer in the
gallery (see "the Klein bottle is now a glassblower's bottle with the internals", above) and neither is
`collar_klein`, so there is no bore for the proxy to be cut to. The underlying observation still stands
as a general warning — *VHACD decomposes a thin curved shell badly, and anything cut to fit the proxy
will not visually fit the mesh* — so if a future piece needs a shaped mount, read this first. The
original text follows.

`settle_scene` collides a VHACD convex decomposition of each dynamic piece, and for the Klein bottle
that decomposition is bad: only **9 hulls** at 1.09× the true volume, whose bulges reach up to
**427.5 mm below** the true underside (the true surface dips below the proxy by at most 25.8 mm).

Because the sim collides the proxy while the render shows the true mesh, `tools/make_klein_collar.py`
must cut the bore to the **proxy** — the wider, containing body — or the settle would spawn the
collided body already buried in the collar. The cost is that the *rendered* bottle sits a few mm clear
of the collar bore where the proxy bulges past the true surface, so the mount reads as slightly loose
on screen even though the physics is tight.

**Proper fix:** get a better collision proxy, then re-cut the bore to it. Options, cheapest first:
raise VHACD's `resolution` / `maxConvexHulls` / lower `concavity` for this mesh (the current call in
`proxy_mesh()` / `settle_scene` uses defaults); or switch to CoACD, which handles thin curved shells
like this far better than VHACD; or, since the collar only needs the band y∈[1.030, 1.110], collide the
*true* mesh there via a per-piece override rather than the whole-body proxy. Whichever is chosen, the
check is already in the generator: it prints how deep each body penetrates the collar at the authored
pose (currently 0.00 mm for the rendered mesh, 0.99 mm for the proxy).

### FEATURE REQUEST (2026-07-19): cache ftrace's per-scene preprocessing before rasterizing
Add an option to **cache the scene-derived data ftrace computes at load** (tessellation / BVH /
material+texture bake / whatever the rasterizer consumes) to disk, keyed on the scene (hash of the
`.ftsl` + referenced assets), so re-opening the same scene in the raster preview skips the rebuild.
Motivation: interactive raster/preview startup on a heavy scene repeats expensive preprocessing every
launch. Design notes: invalidate on any input change (scene text, mesh/texture/vdb asset mtimes,
relevant CLI flags that affect the baked data); store next to the scene or in a sidecar cache dir;
make it opt-in (`-scene-cache` or similar) at first. Overlaps conceptually with the `.ftbuf`
checkpoint sidecar but is about *input* preprocessing, not *output* accumulation.

### PERF (2026-07-19): preview rasterizer much slower than an equivalent WebGL renderer
Our software/CUDA preview rasterizer is far slower than a WebGL rendition of a comparable scene.
Needs profiling before any fix. Likely factors to measure: (a) we rasterize on the CPU in double
precision by default (GPU path is opt-in via `-device gpu|auto`), where WebGL is GPU float hardware
with fixed-function raster; (b) per-frame host readback + p99 auto-exposure on the host each frame
(known-issues "Per-frame readback"); (c) we re-tessellate / rebuild scene data per launch (see the
scene-cache request above); (d) no persistent GPU-resident vertex buffers / we may re-upload. **Next
step:** profile a representative scene (CPU vs `-raster-gpu`) to find the actual bottleneck rather
than guessing; compare against what a trivial WebGL draw of the same triangle count costs. This is a
"why is it slow" investigation, not a confirmed single bug.
**Update 2026-07-22:** the optimization campaign cut CPU raster gallery startup+render
2.68× (15.24 s → 5.68 s; OBJ parser rewrite 00b0765, parallel per-implicit marching
828e7a0, parallel marchImplicit stages 1a78ef6). Still not WebGL-class — the remaining
gap is per-launch scene rebuild + the software shading passes; the profiling task above
stands.

### FEATURE REQUEST (2026-07-19): option for curve-editor curve to be occluded by geometry in front of it
The camera-path / curve overlay shown in the curve editor currently draws over everything (an
always-on-top helper). Add an **option** to instead depth-test the curve against the scene so objects
in front of it occlude it (more spatially truthful), while keeping the always-visible mode available
(often you *want* to see the whole path through geometry). Implementation: test the curve fragments'
depth against the opaque z-buffer the rasterizer already produces; expose as a toggle (CLI flag +/or
editor control). Low risk — the z-buffer is already there.

### DONE (2026-07-22): interactive hover-look spun the view off-screen on light scenes (frame-rate-locked turn rate)
The live-window fly viewer's hover-look turn was applied **per rendered frame**
(`kYaw = 0.040`, `kPitch = 0.030` rad/frame in main.cpp's interactive loop), so the turn
speed scaled with render fps. On a heavy scene that self-limited, but a light model
(e.g. `ftrace cloud1.glb`, which raster-previews at hundreds of fps) turned hundreds of
times faster than intended: since it's *hover*-look (the view keeps turning while the
cursor merely sits off-centre), just having the cursor slightly off-centre after clicking
Reset flung the model out of frame almost instantly. **Fix:** integrate the turn by the
wall-clock frame time — `kYaw`/`kPitch` are now rad/**second** (1.6 / 1.2) multiplied by
the loop's already-computed `dt` (clamped to 0.25 s), making steering frame-rate
independent. Free *translation* stays feedback-locked per frame (collision safety — can't
skip through geometry between unseen frames); rotation-in-place never moves the eye, so it
had no reason to be frame-locked. (0.19.15; README interactive-controls note updated.)

### DONE (2026-07-19): loom RBF scatter field rebuilt the interpolator every frame
`RbfScatterField` / `VecRbfScatterField` (`loom/interp.py`, `_RbfEngine`) used to rebuild the
`scipy.interpolate.RBFInterpolator` once per frame, gated only on the frame number, even when
the sample positions and values were completely static — pure waste for the very common case
of a fixed scatter field queried along a moving path over a long animation.
**Fixed** by replacing the bare per-frame gate with change detection: `_RbfEngine` now caches
the last-built interpolator together with the exact position/value arrays it was built from,
and on a frame advance re-evaluates the sample arrays and reuses the cached interpolator
verbatim (bit-for-bit identical output) unless the positions or values actually changed. A
static field now builds exactly once regardless of animation length; animated values still
rebuild per changed frame (correct, and cheap at realistic scatter sizes). Tests in
`tests/test_rbf.py` assert build-once for static fields (scalar + vector), bit-identical reuse
vs. a fresh rebuild, and correct rebuild-on-change for animated values.
Deliberately **not** done: the deeper "static positions, animated values → reuse the O(M³)
factorization, re-solve only the RHS" optimization. scipy exposes no public refactor-with-new-RHS
API, so it would require coupling to scipy's private compiled internals
(`_rbfinterp_np._build_system`) — a version-fragile dependency for a gain that is negligible at
the small scatter sizes loom fields realistically use. `neighbors=` (local k-NN RBF) still
mitigates cost for large point sets.

### MOSTLY DONE (E4, 2026-07-24): loom VDB read/write — `loom.vdbio` (was DEFERRED 2026-07-18)
**Status: the generator/reader is built (roadmap §E4); a few format variants remain open.**
Originally deferred; superseded by `tools/loom/loom/vdbio.py`, a **loom-native OpenVDB `.vdb`
encoder+decoder** (no OpenVDB/NanoVDB dependency — ftrace's reader is under our control too):
- **Write:** `write_vdb` / `write_volume` / `bake_field` serialise dense `<f4` lattices to a
  multi-grid `.vdb` ftrace ingests directly (`density vdb:<path>`). Codecs: full float32, **ZIP**
  (zlib, interchange-only — ftrace is LZ4-only), **blosc** (LZ4+byte-shuffle, read by *both* loom
  and ftrace → usable on the render path), and **half-float** (`Tree_float_5_4_3_HalfFloat`, read
  by ftrace via the type suffix).
- **Read:** `read_vdb` parses the ACTIVE_MASK/full/half/ZIP/blosc value codecs and the
  axis-aligned transform maps real DCC tools emit (ScaleTranslate/UniformScaleTranslate/
  UniformScale/Scale/Translation), validated against genuine third-party files
  (`scraps/_smoke.vdb` Houdini blosc smoke, `_fire.vdb`, `_sphere.vdb`/`_cube.vdb` level sets).
  Reader is numpy-vectorised (~20 s → 2.8 s on the four real samples).
- **NanoVDB read (DONE 2026-07-27):** `read_nvdb` ingests `.nvdb` v32.6 float `5_4_3` in both
  layouts (`FileHeader` container + raw grid buffer); `read_vdb_grids` dispatches on magic.
  Read-only — loom has no NanoVDB *writer* and no demand for one. Validated against ftrace's
  independent `src/vdbgrid.cpp` (bbox/AABB/peak match), NanoVDB's own redundant per-node
  statistics, and an end-to-end render A/B (means agree to 0.007%; per-pixel diff halves at 4×
  photons → √N noise, not bias). See TODO §E4 for the format notes.
- **Still open (E4 leftovers, not yet built):**
  1. ~~**Rotated `AffineMap`/`UnitaryMap` grids**~~ — **DONE 2026-07-26**: `read_vdb_grids` returns
     `{name: ReadGrid}` with the index array + full `VdbTransform`, so any linear map reads; plain
     `read_vdb` still refuses a rotated grid rather than hand back a wrong axis-aligned box.
  2. **Vec3 grids** (`Tree_vec3s_5_4_3`, e.g. velocity/colour fields) — reader handles scalar
     `Tree_float_5_4_3` only. **Note:** ftrace's volume path also ingests scalar float only
     (`src/vdb_openvdb.cpp` rejects non-`Tree_float_5_4_3`), and none of the sample `.vdb`s carry
     Vec3 data, so this is loom-completeness-only with no render-path consumer and no real file to
     validate against — low priority until a concrete use appears.
  3. ~~**NanoVDB `.nvdb` ingest in loom**~~ — **DONE 2026-07-27** (see above). What remains
     unbuilt is the *write* end: loom cannot author a `.nvdb`. No consumer wants it (ftrace reads
     loom's byte-verified `.vdb` directly), so this is not tracked as debt.
  4. **Sparse-storage transforms + sparse↔dense resampling** — `vdbio` reads a sparse `.vdb` into a
     *dense* numpy array; the field-domain transforms E4 envisioned (N-D rotate-and-slice, warps,
     resample between sparse and dense backings) are not implemented — only straight read/write.
- **When to revisit each:** (4) when a DCC asset arrives with a genuinely-sparse transform loom must
  keep, or a volume big enough that a dense bake is untenable; (2) when a loom or ftrace feature
  actually consumes a vector volume (and a real vec3 file exists to validate against).

### DEFERRED (2026-07-18): `PatOp::MatMulAdd` — a fused matrix·vec+offset pattern opcode (future optimization)
**Status: intentionally not built. This is an optimization of an already-working path, not a
missing capability.** Why we may want it someday, and why we don't need it now:

- **The need it *seems* to fill is already met.** loom's N-D isosurfaces (`gyroid_nd`) rotate the
  lattice in higher dimensions by an affine map `A·(x,y,z) + c` fed into the field — and that matrix
  multiply **must** live inside the isosurface function ftrace evaluates. It already does: `_arg_expr()`
  (and `_pov_nd_coords`/`_pov_affine_coords`) in `tools/loom/examples/gyroid_nd.py` emit each matrix row
  as a plain scalar expression `(coeff*((a)*x+(b)*y+(c)*z)+phase)`. ftrace's pattern parser
  (`src/pattern.h`) compiles that straight into `Const/VarX/VarY/VarZ/Mul/Add` postfix bytecode and
  evaluates it directly on both CPU and GPU (the `-raster-gpu` iso preview ray-marches D=8 tumble gyroids
  today with the matrix baked in). So there is **no field ftrace currently fails to render** for lack of
  a matrix op.
- **What MatMulAdd would actually buy.** Purely a more compact *encoding*: one fused opcode replacing the
  ~6 scalar ops per emitted matrix row (`Const, VarX, Mul, VarY, Mul, Add, …`). Same math, bit-identical
  result — fewer `PatNode`s in the compiled program and a slightly cheaper inner eval. It is **not** a new
  capability.
- **Why we skip it now (MEASURED 2026-07-27 — the original reasoning was wrong, the conclusion holds
  for default workloads).** The 2026-07-18 note claimed "the sin/cos/`PovFn` terms and the sphere-march
  dominate the field eval". **That is false.** `patternEval` is a pure interpreter costing a measured
  **~4.7 ns per `PatNode` regardless of opcode** — a `Mul` costs the same as a `sin`. So pattern cost is
  proportional to *node count*, nothing else, and the baked affine rows (85–89% of every emitted
  `gyroid_nd` field) are exactly where the nodes are. MatRow really would cut node count ~3.5–4×.
  The actual reason to skip it is **Amdahl**, not opcode mix: field eval is only a small share of any
  default workload. Measured end-to-end (probe method below):

  | workload | nodes | field eval | MatRow end-to-end |
  |---|---|---|---|
  | `gyroid_nd` default random draw (median of 8, `--dims-range 3 8`, cyclic) | 76 | ~4% | ~3% |
  | D=8, `--oscillating 6 --harmonics 2`, cyclic — `-raster-gpu` 600² | 162 | 7% | ~5% |
  | …same scene, `-raster -raster-iso 96` 600² | 162 | 12% | ~9% |
  | …same scene, `-export-mesh -mesh-res 160` | 162 | 11–14% | ~10% |
  | **D=16, `--oscillating 16 --harmonics 3 --coupling all`** — `-raster-gpu` 600² | **3917** | **62%** | **47% (1.9×)** |

  So at the sizes anyone actually renders, MatRow buys ~3–9%. It only becomes worthwhile in the
  fully-coupled high-D regime, which is reachable but not default.
- **The cost model (use this instead of re-measuring).** GPU iso preview at 600², the field is the only
  variable: **`time ≈ 3.50 s + 1.452 ms × nodes`**. Field-eval share is therefore a function of node
  count alone — 10% at ~270 nodes, 25% at ~800, 50% at ~2400. MatRow's node reduction is remarkably
  stable at **3.5–4.0×** across every configuration tested (because the affine-row fraction barely
  moves), so the end-to-end win is fully determined by node count. Node count vs the two knobs that
  drive it (`scraps/g3_sweep.py`, regenerable):

  | D | coupling | nodes | MatRow gain | | D | coupling | nodes | MatRow gain |
  |---|---|---|---|---|---|---|---|---|
  | 4 | cyclic | 90 | 3% | | 4 | all | 134 | 4% |
  | 8 | cyclic | 236 | 7% | | 8 | all | 821 | 19% |
  | 12 | cyclic | 392 | 10% | | 12 | all | 2147 | 35% |
  | 16 | cyclic | 536 | 14% | | 16 | all | 4007 | 47% |

  Default `cyclic` coupling grows node count ~linearly in D (edges ~D) and stays under 20% even at
  D=16. `--coupling all` grows it ~quadratically (edges ~D²) and crosses 20% at D=8.
- **When to revisit — now a number, not a vibe.** Build MatRow when a real workload's emitted field
  exceeds **~800 pattern nodes** (≈25% field eval, ≈1.25× win) and that workload is run often enough to
  matter — in practice that means someone actually rendering `--coupling all` at D≥8, or a long video at
  D≥12 fully coupled where 1.4–1.9×/frame is real wall-clock. Below ~250 nodes it is measurement noise;
  don't bother.
- **How the numbers were measured (reproducible).** The probes are in `scraps/` (git-ignored):
  `g3_bench.cpp`/`g3_build.bat` (per-opcode ns/node microbenchmark + a `-count` node census built
  against ftrace's own `src/pattern.h`), `g3_render.py` (end-to-end render probe), `g3_export2.py`
  (export probe), `g3_sweep.py` (node count vs D/coupling). The trick that makes them valid: change
  *only* program size while keeping output bit-identical, by rewriting the field `E` as `(E+E)/2`
  repeatedly — bit-exact in IEEE (adding a value to itself just bumps the exponent) and it adds
  *realistic* nodes rather than a branch-predictable padding tail. Every run asserts an identical
  output-PNG md5 / identical triangle count across the x1/x2/x4 variants, so any time difference is
  pattern eval and nothing else. Fit a line through (nodes, time) → slope is ns/node, intercept is
  everything that isn't field eval.
- **How to build it (the design fork), if revisited.** The pattern VM is a single-scalar-stack machine:
  every `PatNode` is a POD `{PatOp op; double a;}` that pops N and pushes **exactly one** scalar. A
  matrix·vec+offset is 12 coefficients in → a **3-vector** out, which doesn't fit that contract. Two ways:
  - **Option A — single-output "matrow" + coefficient pool (preferred).** Add a `MatRow` op computing one
    output component `m0·x + m1·y + m2·z + off`; emit 3 per point. Preserves the one-push-per-node
    invariant and the POD/GPU-uploadable node, but needs `PatNode` to gain a side-array index (the single
    `double a` can't hold 4 coeffs). Contained; the five standard PatOp touch-points (opcode enum
    `pattern.h:29`, CPU eval switch `pattern.h:112`, device eval switch `render_cuda.cu` ~2754, the
    parser, and PatNode storage) plus the coefficient pool.
  - **Option B — true multi-output (invasive, not recommended for an ergonomics gain).** Give the stack
    vector semantics so a node can push 3 values. Cleanest for "transform a point," but rewrites the VM's
    fundamental one-scalar-out contract across CPU eval, device eval, arity accounting, and every consumer.

### DEFERRED (2026-07-18, reaffirmed 2026-07-26): GPU marching cubes — export-only mesh extraction (TODO §8 G4)
**Status: intentionally not built.** Like MatMulAdd above, this is an optimization of an
already-working path, not a missing capability. It is the last open item in TODO section B
(the `gyroid_nd` `--oscillate` grammar), which is otherwise complete.

- **What it would be.** Run marching cubes on the GPU to extract an isosurface triangle mesh,
  instead of the current CPU implementation (`src/isomesh.h`).
- **Why it is NOT needed for the video path.** The reason per-frame tessellation used to matter
  was that every animation frame re-marched the field on the CPU. That bottleneck is already
  gone: **G2** added the GPU primary-ray isosurface preview (`kIsoPreview`, `-raster-gpu`), which
  sphere-traces the implicit field directly and **never tessellates at all**. `gyroid_nd` routes
  its frames through it, so the video pipeline does not call marching cubes. G4 would therefore
  accelerate only the *explicit mesh-export* path (`--export-mesh` / `.obj` output), which is an
  occasional, offline, one-shot operation rather than a per-frame cost.
- **Why we skip it now (MEASURED 2026-07-27 — and the measurement moved the target).** Timing the
  phases of a res-160 gyroid export (21.47 s total, 3.29M tris) by watching when the `.obj` file is
  created showed the march was **not** where the time went:

  | phase | before | share |
  |---|---|---|
  | startup + scene load + march + dedup | 9.70 s | 45% |
  | …of which pattern/field eval | ~2.4–3.0 s | ~13% |
  | **ASCII OBJ write** | **11.77 s** | **55%** |

  The single biggest cost in a mesh export was **writing the file**, which GPU marching cubes does
  nothing about. Even an infinitely fast, free GPU march would only have taken 21.47 s → 11.77 s
  (1.8× ceiling), and a realistic port — plus a device→host readback of 1.6M verts / 3.3M tris —
  lands well short of that.
- **So we fixed the writer instead (DONE 2026-07-27, v0.84.3).** `isomesh::writeObj` was doing one
  `std::fprintf` per line — ~6.6M calls for this mesh — where FILE locking and format-string
  reparsing dominate actual I/O (measured **22 MB/s** for a 257 MB file). Rewrote it to format into
  an 8 MB staging buffer flushed with `fwrite`, with a hand-rolled decimal conversion for the
  integer-only face lines; float fields still go through `snprintf` with the *same* conversion
  specifiers, so output is **byte-identical** (verified: same md5 as the old binary's file).
  Result: **write 11.77 s → 4.66 s (2.5×), whole export 21.47 s → 13.40 s (1.6×)** — a bigger win
  than MatRow gives on any default scene, for a fraction of the work a CUDA marcher would cost.
  Multi-group exports (`scenes/implicit.ftsl`, 3 isosurfaces) re-validated structurally.
- **Remaining headroom in the writer (not taken — diminishing returns).** The write is now 4.66 s /
  35% of export at 55 MB/s. What's left is ~3.3M `snprintf` calls for the `v`/`vn` float lines
  (~1.25 µs/line); the face lines are already hand-rolled. Squeezing further means reimplementing
  printf's `%.6g` float formatting, which risks byte-exactness for a moderate gain — deliberately
  not done. A binary/compressed mesh format would sidestep it entirely if export size ever matters.
- **When to revisit G4 itself.** Now that the writer is fixed, the march is the largest remaining
  block (~8.7 s of 13.4 s, and it scales as res³). Revisit if mesh export becomes a repeated cost —
  batch-exporting a frame sequence, or interactive export at res ≥ 384 where the CPU march visibly
  stalls — bearing in mind ~13% of that block is field eval (which MatRow, not G4, would address)
  and the readback is unavoidable.

### ~~TECH DEBT (2026-07-18): `-raster-gpu` iso preview shades flat per-material albedo (no textures)~~ — FIXED 2026-07-19 (G5)
The GPU primary-ray isosurface preview (G2, `kIsoPreview` in `src/render_cuda.cu`,
wired as `-raster-gpu`) cast one ray/pixel with the shared `closestHit` and shaded
each hit with a **flat per-material solid colour** (`raster::materialColor` baked into
`matCol[]`), unlike the *triangle* GPU rasterizer — so a textured mesh previewed as flat
colour. **FIXED 2026-07-19 (TODO G5):** ported the CPU rasterizer's textured-preview path
(`Texture::sampleRgb`/`sampleRgbTriplanar`) into `kIsoPreview`. A shared flattened
linear-RGB texel array + per-texture `DPTex` meta + per-material `matTex`/`matTri` binding
(mirroring `raster.h` buildScene's rule exactly) are uploaded alongside `matCol`; the
kernel samples the hit `(u,v)` (or world triplanar) and replaces the flat albedo for
non-emitter hits. Covers **image** skins and, because E1 procedural (formula) skins bake
to `rgb` at load, **formula** skins too — one path. Flat (no-texture) hits are unchanged
(`matTex==-1` skips the sampler). Validated: `-raster-gpu` matches CPU `-raster` within
edge-coverage tolerance on `procskin.ftsl` (formula), `textured.ftsl` (image UV) and
`triplanar.ftsl` (mean channel diff ~0.03/255, ~1033 shared box-edge px); flat
`implicit.ftsl` unchanged. The device sampler is a private twin of `raster_cuda.cu`'s
(separate translation unit).

### TECH DEBT (2026-07-18): FBX import (C8) consumes geometry only — no materials/skinning/animation
The new FBX loader (`src/fbx_load.cpp`, vendored `ufbx`) imports **baked triangle
geometry + generated-if-missing normals + the first UV set**, applying every mesh
instance's `geometry_to_world`. It does **not** yet consume:
- **FBX materials** (Phong/Lambert/PBR) — every triangle takes the mesh block's
  FTSL `material`. glTF already maps `pbrMetallicRoughness`; FBX should get a similar
  material bridge (ufbx exposes `ufbx_material` + `pbr`/`fbx` property maps).
- **Skinning / blend shapes** — `ufbx_load_opts` could `evaluate` a bind/rest pose,
  but the loader currently reads the static mesh. A future path could bake a chosen
  animation time (ufbx supports `ufbx_evaluate_scene`).
- **Animation** — no per-frame FBX animation sampling (loom emits frames instead).
- **Multiple UV sets / per-face materials / vertex colors** — only `vertex_uv` set 0
  is read; `face_material` segmentation is ignored.
These are additive follow-ups; the geometry path is validated (`scenes/cube.fbx` →
8 verts / 12 tris, `scenes/fbxcube.ftsl`). The proper fix for materials is a
`ufbx_material` → spectral-BSDF mapping mirroring `gltf.h`'s material import.

### TECH DEBT (2026-07-17): GPU preview rasterizer — feature parity with the CPU rasterizer

The GPU preview rasterizer (`src/raster_cuda.{h,cu}`, wired into `main.cpp`'s
`-raster` block via the `rasterOne` dispatcher, gated on `-device gpu|auto`) now
accelerates **all camera projections** (rectilinear + fisheye/panoramic), **opaque +
textured (skinned)** previews, and **see-through (clear-glass)** compositing on the GPU.
It only falls back to the CPU rasterizer (`raster::renderFrame`) on a device failure
(the `renderFrame` returning empty). Remaining deferred work is limited to bit-exactness
and readback (below). History:

- **Fisheye / panoramic projections (M2). — DONE (2026-07-17).** `kProject` now branches
  rectilinear (`x/z` + near-plane Sutherland-Hodgman clip → ≤2 sub-tris) vs angular
  (the same `projRadius(projection, θ)/rEdge` map as `raster::projectVtx`, behind-camera
  reject-clip → 1 sub-tri) via `DCam.projection`/`DCam.rEdge`; `kRaster`/`kShade`/
  `exposeAndEncode` were untouched (projection-agnostic as predicted). Validated GPU vs
  CPU on `scenes/fisheye.ftsl` (fish camera: mean abs diff 0.015/255, 515/2.07 M edge
  pixels — tighter than the rectilinear `rect` frame's 0.034/1200), and rectilinear
  regression unchanged.
- **See-through (`-see-through`) on GPU. — DONE (2026-07-17).** `kProject` now propagates a
  per-triangle `clear` flag; `kRaster` skips clear surfaces when `seeThrough` (so only opaque
  geometry wins the visibility buffer); a new `kClear` device pass mirrors `fillTriangleClear`
  — it rasterizes each clear sub-triangle against the opaque `zbuf` written by `kShade`, and
  for every fragment IN FRONT multiplies that pixel's `clearT` (transmittance) and `milkT`
  (1 − per-surface milk, incl. the grazing/rim term) via a CAS-based `atomicMulF` (the product
  is commutative → order-independent, so races are safe). `renderFrame` resets `clearT`/`milkT`
  to 1 with `kFillF`, runs `kClear`, downloads both, and feeds the shared `exposeAndEncode`
  (which already composites them). Validated GPU vs CPU on `scenes/cornell.ftsl -see-through`:
  mean abs diff 0.020/255, 0.034 % edge pixels — same float-vs-double edge gap as opaque.
- **Image skins (textured `reflect texture:<name>` albedo) on GPU. — DONE (2026-07-17).** Each
  `Scene::textures` entry's linear-RGB buffer is flattened into one shared device texel array
  (`DTex` metadata: w/h/filter/wrap/offset) in `upload`; `DPTri`/`DSTri`/`DVtxCS` carry `uv`,
  `tex` and `triplanarScale` (with UV lerped through the near-plane clip); `kShade` samples via
  device twins `dSampleRgb`/`dSampleRgbTri` (nearest/bilinear, v-flip, wrap modes; triplanar
  |n|^4 axis blend) exactly mirroring the host `Texture::sampleRgb`/`sampleRgbTriplanar`.
  Indexed-palette textures sample their raw index-map RGB here, identical to the CPU preview's
  `sampleRgb` path. Validated GPU vs CPU on `scenes/textured.ftsl` (UV skin: mean 0.019/255,
  0.029 % edge) and `scenes/triplanar.ftsl` (world triplanar: mean 0.018/255, 0.028 % edge).
- **Parity is visual, not bit-exact.** The device geometry/shading is single precision
  vs the CPU's double, so silhouette-edge pixels can differ by one pixel of coverage
  (measured ~0.03 % of pixels on cornell/implicit, all on color boundaries, mean abs
  diff ~0.02/255). The exposure + tone-map tail IS shared host double code
  (`raster::exposeAndEncode`), so brightness/lock never drift. Acceptable for a preview;
  a full-FP64 device path would close the edge gap at a large speed cost (not worth it).
- **Per-frame readback.** Each frame downloads the HDR accum + z + emis buffers and runs
  the p99 + encode on the host. Fine today; if it ever bottlenecks, move the tone map
  onto the device and read back only RGB8 (would then need the anchor computed on-device
  or shared explicitly for the exposure-lock case).

### TECH DEBT (2026-07-17): GPU BDPT (mode D) can't texture — falls back to CPU BDPT

The forward (A/B/C) and backward-reference (R) GPU megakernels sample image skins per
hit via `dDiffuseRho` (device twin of `diffuseReflectance`), so a textured
`reflect texture:<name>` albedo renders on the GPU for those modes. The GPU BDPT kernel
(`kBdpt` in `src/render_cuda.cu`), however, drops the surface-local `(u,v)` from its
`DVertex` and its diffuse vertices sample only the constant `reflect` spectrum — so
`cudaBdptSupported` (render_cuda.cu ~5443, `usesTexOrFluoro`: `m.reflectTex >= 0`) rejects
any textured scene and `-mode D -device gpu` falls back to the (correct) CPU BDPT. Same
for fluorescence, roughness/film-thickness/pattern-driven params and mix masks. Verified:
`scenes/textured.ftsl -mode D -device gpu` prints "BDPT-GPU-unsupported feature … using
CPU" and matches the CPU BDPT image. Proper fix: thread the hit `(u,v)` through `DVertex`
+ upload the textures to the device (as the forward path already does), then have the
BDPT `dBsdfF`/`dBsdfPdf`/`dConnect` sample `dDiffuseRho` at the vertex instead of the
constant spectrum. Non-trivial (MIS pdfs must stay consistent), low priority (CPU BDPT is
correct; textured caustic-heavy scenes are rare).

### TECH DEBT (2026-07-17): loom preview server (`-serve`) is resident-process only

The M12 preview server (ftrace `-serve` in `src/main.cpp` `runServe`, loom
`loom/preview.py` `PreviewServer`) delivers only the *resident-process* win: it
keeps the process, live window, CUDA context, and spectral tables alive across
frames, re-rendering each `.ftsl` path streamed on stdin. What it does **not** yet
do (DESIGN.md §11.9 — the real interactivity speedup):

- **Per-frame delta push.** Loom bakes a whole new scene per frame; only a handful
  of constants actually change between adjacent frames. `-serve` still re-parses the
  full ftsl and rebuilds everything each frame. Proper fix: a delta protocol
  (loom sends only changed baked constants; ftrace patches them in place).
- **Static-geometry / BVH caching.** Geometry that doesn't move between frames is
  re-tessellated and its accel structure rebuilt every frame. Needs primitive
  identity + an incremental/cached BVH so unchanged geometry is reused.
- **Preview LOD.** No reduced-fidelity fast path (e.g. coarser isosurface fineness /
  fewer samples) distinct from the final render budget.

Also: the resident live window keeps the *first* frame's resolution for the whole
session (`liveWindowUpdate` / raster window create are guarded by `!g_liveWin`), so a
preview run must hold `-r` constant. Fine for a fixed-size scrub; revisit if
per-frame resolution changes are ever needed.

### DONE (2026-07-16): raster see-through for clear objects (`-see-through`)

Opt-in preview transparency for the `-raster` viewer, so clear materials read as
see-through instead of the old solid pale ghost — *without* refraction. Implemented
entirely in `raster.h` + a CLI flag in `main.cpp`:

- **Model.** Each clear surface (dielectric / thin-film / filter / diffuse-transmit,
  via `isClearPreviewType`) between the camera and the opaque background multiplies a
  per-pixel transmittance (`clarity`, default 0.85) and accumulates a milk product, so
  N crossed surfaces give `clarity^N` dimming + growing haze (a closed ball = 2
  crossings). A grazing-angle Fresnel-ish term adds silhouette milk so edges read.
  Composited in display-linear space in the tone-map pass:
  `c = c*clearT + milkColor*(1 - milkT)`.
- **Architecture.** `PTri`/`STri` gained a `clear` flag (set in `tessellate` from the
  material type). Opaque pass skips clear tris; a second **order-independent**
  band-parallel pass (`fillTriangleClear`) accumulates transmittance/milk against the
  finished opaque z-buffer — the product is commutative so no transparent depth sort is
  needed. Auto-exposure is still computed on the opaque-only accum (glass doesn't skew
  exposure).
- **CLI.** `-see-through` / `-seethrough` / `-glass` enable it; `-glass-clarity <0..1>`
  sets the per-surface transmittance (implies the flag). Wired into the main raster
  path and the interactive fly-viewer `renderFrame` calls (metering pass stays opaque).
- **Possible follow-ups (not done):** per-channel coloured transmittance for tinted
  glass/filters (currently neutral dimming); modelling thickness so a thin edge dims
  less than a thick centre (currently every crossed triangle counts equally).

### DONE (2026-07-16): camera_curve editor — all five phases + rough edges landed

The in-viewer `camera_curve` editor (main.cpp fly-viewer, Rec / +Pt / Ins / Del / Save)
landed as **Phase 1** (core point authoring + recording + live spline overlay + Save a
`camera_curve` block with a `look curve`). All planned follow-ons are now DONE:

- **Phase 2 — speed painting. DONE (2026-07-16).** Additive wheel brush modulates
  per-control-point speed (inverse density) in Paint mode; emitted as a `density_at`
  track and retimes live playback. Paint/Flat panel controls + speed readout.
- **Phase 3 — orientation painting. DONE (2026-07-16).** Mouse-look in Paint mode steers
  the nearest control points' `fwd`, reshaping the `look curve` (WYSIWYG in the overlay
  and saved block).
- **Phase 4 — rendered-sequence source. DONE (2026-07-16).** `ftrace -review <base>`
  plays a directory of rendered frames (`<base><digits>.<ext>`; reads png/jpg/bmp/tga/ppm)
  on the live window/timeline, scrub/Play, re-times via the Paint-mode speed brush, and
  Save writes a re-paced copy into `<dir>/retimed/` + an ffmpeg hint. `reviewMode()` in
  main.cpp.
- **Phase 5 — round-trip. DONE (2026-07-16).** Opening a scene with an existing
  `camera_curve` under `-explore`/`-fly` seeds the editor's `editPts` from that curve's
  control points (eye + look direction from look curve/look_at/tangent + per-point speed
  from the `density` track). Captured at load in `ftsl.h` (`AuthoredCurve` on `Loaded`,
  filled in `addCameraCurve`), consumed in `main.cpp`'s viewer. Save re-emits a revised
  curve. The loaded flyby still plays at full fidelity until the first edit.

Rough edges — all addressed (2026-07-16):
- **(a) look-spline bowing. FIXED.** Saved `look_point`s are now placed one MEAN control-
  point spacing ahead along each point's fwd (scene-relative, clamped), instead of a fixed
  1 world-unit. A larger, consistent offset keeps `(lookSample − eyeSample)` well away from
  the two splines' interpolation noise, so the aim spline stays smooth between sparse points.
  Direction at each control point is preserved exactly (any positive distance along the same
  fwd). In `saveCurveFn` (main.cpp).
- **(b) explicit point selection. FIXED.** A `selectedPoint()` helper drives both the red
  overlay highlight and Del. Locked to the path it follows the timeline (the control point
  nearest the scrub position — scrub to select), and in free flight it's the point nearest
  the eye. Also fixed a latent bug: Ins now finds its segment via `bracket()` (normalizes by
  the ACTUAL explorePath length) instead of `pathPos / kPreviewPerSeg`, which was wrong for a
  freshly-loaded curve whose frame count isn't a multiple of the preview sampling rate.
- **(c) multi-curve round-trip. FIXED.** The viewer records the flyby's base name (frame
  "beta00" → "beta") and the editor seeds from the authored curve whose name matches, so
  `-camera <name>` selects which curve is edited — not blindly the first. In main.cpp
  (`exploreCurveName` + the Phase-5 seeding block).

### OPEN: interactive raster fly-viewer can peg all cores / grow RAM when orphaned or on a heavy scene

The interactive fly-camera viewer loop (`main.cpp`, ~line 4037) only re-rasterizes
when `changed` is true and sleeps 15 ms when `!nav.any()`, so a *normal* idle viewer
is cheap. But an **orphaned** ftrace (e.g. a `-explore` test whose parent `timeout`
sent a signal the GUI process ignored, leaving it running with no console) was observed
pegging **all** CPU cores and climbing from ~2 GB to ~8 GB working set on
`scenes/gallery_settled.ftsl` — i.e. it was re-rendering flat-out (`changed` stuck true)
with no user input. Exact trigger unconfirmed (likely a teardown-state artifact where
`clientSize`/`drainNav` return values that keep flipping `changed`), and it wasn't
reproduced because doing so re-pegs the machine. Two things to consider as the proper fix:
(1) **bound the re-render rate** in the viewer loop (pace the loop to ~60 fps regardless
of `changed`/`nav.any()`), so even a stuck-`changed` runaway or a fast light scene can't
burn 100% of every core for no visible benefit; (2) make the loop exit on the same
signals as a normal render (so `-explore`/interactive processes die cleanly on SIGTERM,
not just window-close), and audit the resize-follow (`fitRes` vs `clientSize`) for any
size oscillation that would flip `changed` every iteration. Operational note: kill
interactive/`-window` test processes **explicitly** (PowerShell `Stop-Process -Force`) —
`timeout`-wrapping a GDI window app does not reliably terminate it.

### OPEN: rainbow phase — `SpecVtx::term` (through-glass-sphere fog connection) is HG-only

The new **rainbow droplet phase** (`rainbow.h`, tabulated Airy/Mie spectral phase) is
dispatched everywhere a medium's phase is evaluated **except one spot**: the specialised
"trace a photon *through a glass sphere* and connect its interior fog to the sensor" path
in `render.h` (`SpecVtx::term`, ~line 598) still calls `hgPhase(dot(wIn, wP), g)` directly
instead of `Medium::phaseValue(...)`. That code path predates the phase abstraction and
uses a flattened per-vertex `g` (no `mediumId`/λ), so it can't see a `RainbowPhase`. Impact
is tiny — it only affects fog that sits *inside* a refracting glass sphere viewed on the
special two-refraction connection — but a rainbow medium placed there would silently fall
back to the smooth HG lobe. **Proper fix:** give `SpecVtx` the owning `mediumId` (and thread
λ into `term`) so it can call `scene.media[mediumId].phaseValue(cosθ, λ)` like every other
site. Left HG-only for now to avoid reworking that specialised connector in the same change.

### DONE (2026-07-15): rainbow (water-droplet) phase — implemented, wired, and validated end-to-end

A medium can now scatter through a physically-tabulated **Airy water-droplet phase**
(`rainbow.h`) via FTSL `phase rainbow { .. }`, instead of the smooth Henyey-Greenstein lobe.
- **Physics core** (`rainbow.h`): Airy theory of the rainbow tabulated on a (λ×μ) grid with
  per-λ CDF importance sampling; normalised so `2π∫p dμ = 1` per λ. Self-test confirms exact
  Airy values, textbook Descartes angles, and unit normalisation.
- **Data model** (`scene.h`): `Medium::rainbowPhase` (shared_ptr, null for the common HG case
  → HG media stay bit-identical); `phaseValue`/`phaseSample` dispatch to it when set (overrides
  `g`). Symmetric phase → forward pdf == reverse pdf.
- **Wiring:** phase dispatch threaded through CPU forward (`render.h`), backward (`backward.h`),
  and BDPT (`bdpt.h` `phaseF`/`mediumScatterF`, `-dot(wo,wi)` scattering cosine). GPU volume
  path is HG-only, so `cudaForwardSupported`/`cudaBdptSupported` now **refuse rainbow media**
  and let the render fall back to the CPU (rather than silently dropping the bow to HG).
- **FTSL grammar** (`ftsl.h addMedium`): `phase hg` (default) / `phase rainbow { droplet_um,
  secondary, supernumerary, strength, forward_g, secondary_ratio }`. Features on by default.
- **Parser bug found & fixed:** `phase rainbow { .. }` — the subtype bareword `rainbow` before
  `{` is consumed by `parseValue` as the nested block's **`type`**, so `val.words` was empty and
  `kind` silently defaulted to HG (no bow, no error). Fixed by reading the kind from
  `ph->val.block->type` when `val.words` is empty. This was the reason early validation renders
  showed only a smooth veil despite correct physics.
- **Validated:** `scraps/rainbow_ring.ftsl` (centred-ring geometry, mode D, CPU) analysed with
  `scraps/radial_profile.py` shows the full signature — a **primary bow at ~42°** (violet inner /
  red outer), **Alexander's dark band** (~43–50°), and a **secondary bow at ~51–53°** with
  **reversed colours** (red inner / blue outer). Peak/median luminance ratio ~3.5× and climbing
  with samples.

Remaining rainbow tech debt: the `SpecVtx::term` glass-sphere-interior connector is still
HG-only (see the OPEN note above).

### DONE (2026-07-15): gradient-index (GRIN) media — Phase 2 wired through forward (CPU+GPU) + backward

**Phase 1 (landed earlier):** a `medium { ior "<expr over x y z r>" bounds { .. } }`
defines a **gradient-index region**: rays entering its bound bend continuously via a
symplectic Eikonal march (`d/ds(n·dr/ds)=∇n`) instead of travelling straight. Data model
(`Medium::ior`/`iorStep`, `nAt`/`gradNAt`/`insideBound` in `scene.h`), ftsl parsing
(`ior` / `ior_step` in `ftsl.h addMedium`), and the CPU backward marcher were done.

**Phase 2 (this commit):** the one canonical marcher now lives in **`grin.h`**
(`grin::sceneHasGrin` + `grin::march`, extracted verbatim from the backward tracer) and is
shared by **CPU backward** (`backward.h`, mode R), **CPU forward** (`render.h tracePhoton`,
modes A/B/C) and the **GPU forward megakernel + wavefront** (`render_cuda.cu` `dGrinMarch`,
`dMedInside`/`dMedNAt`/`dMedGradN`; `DMedium.ior`/`iorN`/`iorStep` uploaded; gated by
`DScene::hasGrin`). All bend rays identically; `ior`-free scenes stay bit-identical (the
march is only entered when `sceneHasGrin`). Validated: `scraps/_grin_lens.ftsl` warps a
checker in mode R; `scraps/_grin_caustic.ftsl` (a converging GRIN sphere over a floor)
shows the same lens redistribution in CPU mode B **and** GPU mode B, and a smooth
unperturbed pool in mode R (backward's straight NEE shadow ray can't bend — see below).

**BDPT (mode D) deliberately REFUSES GRIN.** BDPT's connection geometric term, area-measure
pdf conversion and MIS weights all assume STRAIGHT connecting segments, so a bent path would
bias the estimator. `main.cpp bdptUnsupportedFeature()` returns a GRIN message (mode D errors
out with "use mode A/B/C or R"), and `cudaBdptSupported()` rejects GRIN as defense-in-depth.
GPU backward (mode R) already falls back to the CPU for *any* medium (`cudaBackwardSupported`
rejects `anyMedium()`), so GRIN mode R runs on the GRIN-aware CPU backward tracer — correct.

Remaining GRIN tech debt is tracked as its own OPEN entry below.

### OPEN: GRIN tech debt (Phase-1 semantics carried forward — not regressions)

Follow-on work for the GRIN Phase-2 wiring above. None of these are bugs in the shipped
behavior; they are known limits of the current bend-only model.

1. **Only PRIMARY rays bend; connection/NEE rays are still straight.** Each tracer bends only
   its primary ray — the backward camera ray and the forward photon path. Its *connection/NEE
   ray* is straight: mode R's shadow ray to a light can't bend through a GRIN region (so it
   misses GRIN caustics — that's why `_grin_caustic` is dark in R), and the forward
   camera-splat (modes A/B) is straight (so imaging a surface *through* a GRIN lens via splat
   doesn't warp — use mode R for "camera looks through a GRIN lens", forward for "GRIN caustic
   onto a surface viewed directly"). Mode C (forward-catch) is fully unbiased but
   sample-starved. Curved-path connections (bending the shadow/splat ray) are a future
   enhancement.
2. **Exterior IOR inside a GRIN region is hard-coded to 1.0.** At a dielectric interface
   *inside* a GRIN region the exterior IOR should be `nAt(hit)` not 1.0 (the current code
   assumes GRIN regions sit in open air). **Directly relevant to the pending xenon-lamp work:**
   its bulb is a nested dielectric whose interior gas index differs from air, so a GRIN
   gradient over that same region would need the correct surrounding index at each interface.
3. **Absorbing/scattering GRIN not integrated along the curve.** A GRIN medium that is *also*
   absorbing/scattering isn't integrated along the curved path (treated as clear — the classic
   use). Fixed-step RK1 (`iorStep`, default bound/64) suits smooth fields; steep gradients may
   want RK4 / adaptive stepping.

**GRIN in BDPT (mode D) — deferred, may implement someday.** Mode D refuses GRIN today (see
the DONE entry above: its connection G-term, area-measure pdf conversion and MIS weights all
assume straight edges). Two tiers exist if we revisit it: (1) **cheap** — let the camera/light
subpaths bend on their PRIMARY march (they already can elsewhere) and pdf-consistently *skip*
any connection whose straight edge crosses a GRIN region; stays unbiased, only under-samples
pure-GRIN-caustic paths (negligible for a weak gradient like the lamp gas), and would let mode
D literally accept a GRIN scene. (2) **research-grade** — true curved connections (solve the
two-point boundary-value problem for the connecting geodesic, generalized geometric/Jacobian
term, consistent MIS); expensive and numerically nasty near a focus, poor ROI. Keep this on
the radar; neither is built.

**Showcase (`scenes/gallery_settled.ftsl`) render-mode note.** The hero stays in **mode D**
for now (its documented command). If we want the xenon lamp to carry a GRIN gas gradient *and*
have the whole hero render, the pragmatic route is to render the showcase in **mode B**
(forward) instead — B supports GRIN and captures *all* the effects — accepting that it's
**slower to converge** than D on this mixed scene. Tracked so we remember the trade-off:
mode D now (fast, no lamp GRIN) vs. mode B later (slow, full effects incl. lamp GRIN), unless/
until tier-1 "GRIN in mode D" above is built. The showcase now encodes this trade-off directly:
its still camera is wrapped in `prefer { mode D } else { mode B }`, so mode D wins today and the
loader auto-falls back to mode B the day a mode-D-hostile feature (a GRIN lamp-gas field) is added.

**Showcase idea: a fog rainbow that MOVES with the camera (`phase rainbow`).** Now that the
water-droplet phase is wired end-to-end (FTSL `phase rainbow { .. }`, validated: primary +
secondary bows, Alexander's dark band, reversed secondary colours — see the DONE note below),
we could dress the showcase (or a dedicated flyby) with a thin rain curtain that produces a
real rainbow. The compelling part is that **the bow is centred on the antisolar point (the
anti-sun direction), not on any object** — so as a moving camera pans/dollies, the bow slides
across the frame and *follows the view* exactly the way a real rainbow "runs away" from you.
That motion is the giveaway that it's genuine scattering physics, not a painted arc. Recipe
to keep on the radar for a flyby (own subdir `png/<set>/` per the flyby rule):
- Distant sun **behind** the camera's general travel direction (parallel rays → sharp bow);
  keep the sun *outside* the fog slab so it isn't extincted before lighting the drops.
- A **bounded, thin** rain curtain in front (`sigma_t`~0.001–0.002, `albedo` ~0.99, optical
  depth ≲ 0.3) so single scattering — which carries the bow — dominates the multiply-scattered
  veil. `phase rainbow { droplet_um 500 secondary on supernumerary on }`.
- A `camera_curve` that pans the antisolar point across frame (e.g. yaw the look direction, or
  translate laterally) so the ring visibly tracks the camera. Render **mode B** (forward) or
  **mode D** (BDPT, bounded fog) on the **CPU** — the GPU volume path is HG-only and auto-falls
  back to CPU for rainbow media, so a big flyby is CPU-bound (budget accordingly).
- Validated seed scenes to crib geometry/params from: `scraps/rainbow_ring.ftsl` (centred
  ring, mode D) and `scraps/rainbow_test.ftsl` (antisolar-aimed slab).

**Minor tech debt: `prefer{}/else{}` trial builds reload meshes.** Resolving a `prefer` node
trial-builds each candidate branch to test renderability (`ftsl::load`, `tryBuild` lambda). The
common **single-node** case is optimized — the accepted trial's `Loaded` is reused as the final
scene, so meshes load exactly once (verified on `gallery_settled.ftsl`). But when a node has
several branches that get *rejected* before one is accepted, each rejected trial still re-parses
and RE-LOADS every mesh; and a **multi-node** `prefer` does one extra final rebuild on top of the
per-node trials. For a heavy scene (600k-tri OBJs) that multiplies OBJ-load time. Proper fix if it
ever bites: build the shared/non-`prefer` blocks (all the meshes) *once* and only re-resolve the
small mode-sensitive delta per branch, instead of flattening + full-building the whole block list
each trial. Negligible today (branches carry only a camera + medium), so left as-is.

### DONE (2026-07-15): `exposure_lock` selector meter pre-pass now covers every render mode

Previously the real-render `exposure_lock <selector>` meter pre-pass only metered the
forward models (A/B/C) and the backward reference (R); modes D/M/P silently fell back to
locking on whichever frame rendered first, *ignoring the selector*. Fixed: the meter
pre-pass (`meterAnchor` lambda in main.cpp, the `meterPlan` loop just after the `-raster`
block) now renders the selector-chosen viewpoint in its **own** mode — `renderBdpt` for
D, a lazily-built shared reduced photon map + `renderPhotonCamera` for M, and
`classifyComposite`+forward+backward+`compositeFromFilms` for P — and any other mode
(S/U/V/…) falls back to a **general forward mode-B light-trace** (still a correct
scene-brightness anchor, never an arbitrary frame). Because the p99 anchor is a property
of the radiance, not the integrator, every mode yields a consistent anchor. There is now
**no silent frame-0 fallback anywhere**; a bare `exposure_lock` also defaults to the path
**average** rather than the first frame. Validated on scraps/lock_test.ftsl in modes
B/M/D/P with `index`/`average` selectors (all honoured, all frames flicker-free).

### DONE (2026-07-15): A/B/C now agree in absolute brightness at equal `power` (finite-lens catch modes were near-black)

Previously, `ABS_EXPOSURE_GAIN` (main.cpp ~927, value `6.0`) was calibrated **only for
mode B** (the pinhole splat). The finite-lens catch modes **A** (`connectLens`) and **C**
(forward pupil catch) produced a film scale ~10^3–10^4× dimmer at the same gain, so an
absolute scene shot in mode A/C came out essentially **black**. Root cause: mode B's
`connect()` records **radiance** (i.e. it is implicitly divided by the pixel solid angle
Ω_pix = `pixelPlaneArea()·cosCam³`), whereas the A/C splat records **flux-per-cell**
(`× R²/dist²`, with no cell-area normalisation) — a dimensionally different quantity.

**Fix (the F = 1/A_cell factor).** Fold a single per-camera constant
`F = 1/(pixelPlaneArea()·filmDist²) = 1/A_cell` into the A/C splat, converting its
flux deposit into film irradiance on the same scale mode B uses. Derivation: the raw A/C
splat = `B·(π R²·A_pix)`; the target (camera equation) = `B·(π R²/filmDist²)`; on-axis
`A_pix = pixelPlaneArea()`, so the correcting ratio is exactly `1/A_cell`. Because it is a
per-camera constant it cancels under p99 auto-exposure (auto-exposed scenes stay
byte-identical) and only re-seats the *absolute* level. Applied at four sites, CPU + GPU:
- `render.h` `connectLens` and `connectLensVolume` (`contrib *= 1/(pixelPlaneArea()·filmDist²)`),
- `render.h` mode-C catch (`cCell` factor on the film `add`),
- `render_cuda.cu` `connectLens`, `connectLensVolume`, and CAM_C catch (same factor).

Modes A/C use plain gain 6 (`comp = 1`); mode B keeps its aperture `camEq = (π/4)/N²` fold
(the separate DONE issue below) — the two paths each carry the `1/N²` once, never doubled.

**Validated** (`scraps/abs_calib.ftsl` / `_acalib_wide.ftsl`, Cornell box + 100 W area
light), tone-mapped 8-bit whole-image mean, GPU **and** CPU:
- f/4:   A `6.79` vs B `7.16` (95%).
- f/1.4: GPU A `25.54` vs B `25.90` (**98.6%**); CPU A `25.49` vs B `25.86` (**98.6%**).
- Mode C converges to B as its (pupil-catch) noise falls — mean `16.81 → 20.70 → 21.54`
  as noise `82% → 35.8% → 31.6%` (residual gap is tone-map clamp bias on fireflies, not a
  scale error). A and C now land at the same absolute brightness as B instead of black.

NOTE: the `-raster` preview still uses its relative reference aperture (Rref=0.02) for the
aperture-brightness *ratio*; that remains correct. Known wart (minor, logged for later):
the **`-aperture` CLI override** changes A/C's physical pupil `R` but does **not** feed
mode B's `camEq` comp (which reads the scene's `fstop`/`lens`), so cross-mode comparison
via `-aperture` desyncs B — always set `fstop` in the scene for an apples-to-apples A/B/C
comparison. Not a render-correctness bug (a normal single-mode render is unaffected).

### DONE (2026-07-15): absolute EV — mode B now applies the aperture's exposure (light-gathering)

**Fixed** in `src/main.cpp` (~3380, the ftsl per-camera render setup): when a scene
is in absolute EV *and* a physical aperture was actually authored (`fstop`/`lens`,
detected by `c.lensF > 0`), a mode-B render now folds the camera-equation aperture
term `camEq = (π/4)/N²` (with `N = c.lensF/(2·c.apertureR)`) into the exposure comp.
This adds only the **radiometric** light-gathering factor — the pinhole keeps zero
DoF. Gated tightly so it only *darkens* a mode-B camera that opted into an f-number;
with no aperture authored (`c.lensF == 0`, e.g. shipped `scenes/absolute.ftsl`) the
branch is skipped and the pinhole stays the pure radiance reference (comp unchanged,
byte-identical). Modes A/C are untouched (they must NOT double-apply `1/N²` — their
gross-scale mis-seat is the separate issue above, now DONE via `F = 1/A_cell`).

**Validated** (`scraps/abs_calib.ftsl`, Cornell box + 100 W area light, mode B, GPU):
rendering the same scene at f/2 vs f/8 now separates by exactly 4 stops —
patch linear-luminance mean `2.2225e-2` (f/2) vs `1.3919e-3` (f/8), ratio **15.97×**
(≈ 16× = 4 stops); whole-image mean ratio 15.99×. The startup log shows the comp
scaling correctly: `exposure=1.18 (absolute: gain 6 x 0.196 comp)` for f/2, where
`0.196 = (π/4)/2²`. Author `fstop`/`lens` at the **camera-block** level (not inside
`film{}` — the film block only reads res/size/format/iso/shutter/exposure).

The other half of the "unification" (making A/B/C agree in *absolute* brightness at
equal power) is now also DONE — see the "A/B/C now agree in absolute brightness"
issue above (the `F = 1/A_cell` A/C re-seat). Mode B is internally correct wrt
aperture, and A/B/C now match within noise at equal `power`.

The old rationale for excluding aperture from the exposure comp ("in splat mode B
the aperture is virtual, so an f-number term would double-count / be an artifact",
CamSpec/main.cpp ~921) held **only for modes A/C**, which already carry the physical
`R²` in their splat weight (render.h `connectLens`). Mode B has **no** `R²`, so a
virtual-aperture exposure term there is clean and non-redundant — hence the fix above
adds it in B only. When the A/C gross-scale gain is eventually re-seated (issue above),
keep the two paths consistent: A/C get `1/N²` from the pupil-area `R²` splat weight
(fix the gain, don't add `camEq`), B gets it as the pure exposure factor added here —
do NOT double-apply. Ideal end state is one camera-equation absolute model (add
`cos⁴θ` natural vignetting too) that makes A, B, C agree at equal `power`.

### DONE (2026-07-22, no longer reproduces): mode D (GPU BDPT) — data-dependent "unspecified launch failure" on gallery_settled.ftsl

Rendering `scenes/gallery_settled.ftsl` in **mode D on the GPU** used to crash with
`[cuda] bdpt kernel failed: unspecified launch failure` reproducibly at **spp 14**
(~232 s in; earlier spp complete fine and write correct images) — an illegal memory
access inside the BDPT megakernel (`kBdpt`), NOT a TDR timeout (chunks ~0.15 s) and
NOT GPU contention (single process). Bounds inspection of the per-thread
`eye[]`/`light[]` subpath arrays (`BDPT_MAXV=11`), `DMediumStack` (CAP 8, push
guarded), media free-flight loops, and the `double st[64]` VM stacks found nothing;
a bounded memcheck on the still cam (`-camera cam -spp 6`) was clean.

**Resolution (v0.19.8): the crash no longer reproduces, on any of three axes tried.**
1. **Faithful seed replay** — the original failing run was `-noise 3` (progressive,
   `sppTotal = UNBOUNDED_SPP = 1e9`, so `gidx = pix*1e9 + k`); replaying exactly that
   (`-camera cam -noise 3 -device gpu`) ran clean to **45 spp**, 3× past the historical
   crash point, correct images throughout.
2. **Fixed-budget replay** (`-camera cam -spp 16`, a *different* seed schedule since
   `gidx` mixes the run's total spp) — clean.
3. **Flyby-position soak** — the "crash is flyby-viewpoint-specific" hypothesis:
   72 positions along the fly curve (scratch scene with the curve set to
   `mode D frames 72`, `-spp 4` each, 960×540) — all 72 frames clean, zero CUDA errors.

A fresh code audit of `kBdpt` / `dConnectBDPT` / `dMisWeight` / `DCamera::project` /
`selectEmitter` found no OOB (project is edge-clamped, MIS fully guarded, maxDepth
clamped to device capacity). Prime suspect for the incidental fix: the **watertight
ray-triangle intersection** rework (41ac6a4, 2026-07-18, Woop et al. JCGT 2013) —
changed hit numerics re-route the data-dependent pathological path; the
shadow-terminator and adjoint-correction changes in the same window are also
candidates. Root cause was never pinned to a line, so treat as *mitigated in
practice*, not proven-fixed.

**If it ever recurs:** bound the repro (always pin `-camera` and a finite budget) and
run the real `compute-sanitizer.exe` (in the CUDA `compute-sanitizer/` subdir — NOT
`bin/compute-sanitizer.bat`, which exits 127 from the bash tool); the build has
`-lineinfo`, so memcheck reports the exact `render_cuda.cu:<line>`:
`"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/compute-sanitizer/compute-sanitizer.exe" --tool memcheck --log-file scraps/_sanit.log build_cuda2/bin/ftrace.exe -in scenes/gallery_settled.ftsl -camera cam -mode D -device gpu -spp 45 -o png/_sanit.png`

### DONE (2026-07-15): Forward modes now smooth-shade interpolated normals — Veach adjoint correction applied

A smooth-shaded mesh (authored `vn` **or** crease-smoothed via `mesh { smooth }`) used
to render smooth in the backward reference mode R but **faceted in the forward modes**.
Root cause was the **shading-normal adjoint asymmetry** (Veach §5.3): a backward
estimator integrates the incident cosine through the *shading* normal, so interpolated
normals smooth the shading for free; a forward/particle tracer deposits irradiance per
**geometric** area and stays faceted.

**Fix shipped.** `shadingAdjointCorr(wi, wo, ns, ng)` in `geometry.h`
(`corr = |cos(wi,Ns)·cos(wo,Ng)| / |cos(wi,Ng)·cos(wo,Ns)|`, guarded denom, exactly 1
when Ns==Ng) is multiplied into the **particle throughput** at every non-specular
continuation and every camera connection of the LIGHT/importance subpath:
- `render.h` modes A/B/C — `connect`/`connectLens` splats + Diffuse/Fluorescent/
  DiffuseTransmit continuations (shared `tracePhoton` walk, so the M/S deposit inherits it).
- `bdpt.h` mode D — `randomWalk` (gated `mode == Importance`) + `connectBDPT` light-side `f`.
- `vcm.h` mode U — light-subpath continuation + light-image splat + the VC connection's
  light-vertex `fLit`. The eye/Radiance side is never corrected (it smooth-shades for free).
- GPU twins in `render_cuda.cu` — `dShadingAdjointCorr` threaded through `connect`/
  `connectLens`/`splatSurfaceAll` (forward tracer) and `dRandomWalk`(importance flag)/
  `dConnectBDPT` (GPU BDPT).

Validated smooth against mode R on `scraps/_smooth_on.ftsl` and the harsher
`scraps/_iso_sphere.ftsl` (directional-lit sphere, no indirect): modes B, D (CPU+GPU),
and U all match the mode-R gradient. Exactly 1 when Ns==Ng ⇒ every flat/analytic scene is
bit-identical (whole existing validation suite untouched).

**Surprising finding — photon-density gathers (modes M/S) did NOT need a gather-side
correction.** An initial hypothesis added a `cos_s/cos_g` gather reweight to the mode-M
photon-map and mode-S SPPM density estimates. Empirically this was WRONG: mode M/S already
smooth-shade (verified matching mode R by center-column brightness profile on both the
Cornell and isolated directional scenes, at gather radii from 0.004 to 0.017), and adding
the reweight *introduced* facet banding. That speculative correction was reverted; M/S
gathers are left uncorrected. See the tech-debt note below for the one place a gather-side
correction WAS needed (VCM's vertex-merge) and why the asymmetry isn't fully understood.

### TECH DEBT: VCM vertex-merge needs a gather-side shading-normal reweight that M/S don't — asymmetry not fully understood 2026-07-15

Direct consequence of the DONE entry above. The mode-M photon-map and mode-S SPPM
photon-density gathers **smooth-shade correctly with no gather-side correction** (verified
against mode R). But mode U's **VCM vertex-merge (VM) strategy** genuinely *facets* on the
same smooth meshes — strong full faceting on `scraps/_iso_sphere.ftsl` — even though its VM
density estimate is mathematically the same kind of radius gather. The current fix is a
**scoped, file-local `vmGatherCorr(wp, ns, ng) = |cos(wp,Ns)| / |cos(wp,Ng)|`** in `vcm.h`,
multiplied into the merge's camera-side `fCam` only (the `vmNorm` carries no geometric
cosine, so the raw VM value is as smooth as mode M — the faceting instead enters through the
per-facet MIS coupling between the VM and VC strategies).

**Why the asymmetry exists is not fully understood.** Best current hypothesis: the VM/VC
MIS weights assume normal consistency between the merged light-vertex and camera-vertex
BSDF evaluations, and the shading/geometric-normal mismatch breaks that assumption for VM in
a way the standalone M/S estimators (no competing MIS strategy) never see. The **cleaner
future fix** is to make the VM↔VC MIS-weight derivation consistent under interpolated
normals (so no ad-hoc `fCam` reweight is needed), rather than patching `fCam`. The current
`vmGatherCorr` is a no-op on flat tris / analytic spheres (`Ns==Ng` ⇒ ratio 1), so it
cannot regress any existing (non-smooth) scene. Low priority — mode U smooth-shades
correctly today; this is about *why* and a tidier derivation.

### DONE (2026-07-15): Shading-normal geometric-hemisphere clamp propagated to all connection sites (+ fixed a pre-existing GPU mode-R light leak)

The geometric-hemisphere clamp (stop a smoothed shading normal from leaking light in
through the geometric back face; `orientedGeoN()` in `geometry.h`) was previously only in
the backward reference (`backward.h` `neeLight`/`neeEnv`) and the forward tracer's camera
connection (`render.h` `connect`/`connectLens`). It is now applied at **every** NEE /
connection site: `bdpt.h` (mode D — the t==1 splat, s==1 NEE, and interior connection, each
endpoint), `vcm.h` (mode U — light-image splat, NEE, and both VC-connection endpoints), and
the GPU twins in `render_cuda.cu` (`connect`/`connectLens` for A/B/C, `dConnectBDPT` for
mode D, and `bkNeeLight` for backward). **Per-site recipe applied:** alongside the existing
`dot(Ns, wi) <= 0` shading-side test, also require `dot(ngo, wi) > 0` (ngo = geo normal
oriented to Ns) and offset the shadow/connection ray along `ngo`. In `bdpt.h`/`vcm.h` the
clamp is **guarded by `!isTwoSidedMat`** so transmissive (glass) connections through the
back hemisphere are not wrongly killed; GPU BDPT is reflect-only (v1) so the clamp is
unconditional there. No-op for flat tris / analytic spheres (`ngo == Ns`), so every
non-smooth scene is bit-identical.

**Modes M/S need nothing:** mode M's direct lighting reuses `bw.neeLight` (already clamped)
and its photon gather has no shadow ray; SPPM's visible-point walk does no NEE at all.

**Pre-existing bug fixed as a side effect.** The GPU backward NEE (`bkNeeLight`) was missing
the clamp its CPU twin (`backward.h neeLight`) already had, so **GPU mode R silently leaked
light through geometric back faces** — on `scraps/_iso_sphere.ftsl` GPU-R showed a smooth
(leaking) terminator while CPU-R showed the correct leak-free (faceted) one. Adding the
clamp to `bkNeeLight` makes CPU-R and GPU-R identical.

**Caveat — shadow terminator (NOW SOFTENED, see DONE entry below).** A *hard* geometric
clamp reveals the shadow-terminator problem: on a low-poly smooth-normal sphere under grazing
light the terminator shows the underlying facets (hard dark slivers) rather than a smooth
gradient. This was the mode-R reference's existing behavior too, so the clamp made the forward
modes *consistent with the reference*. The hard cutoff has since been replaced everywhere by
Chiang et al. 2019 softening — see the next DONE entry.

### DONE (2026-07-15): Shadow-terminator softening (Chiang et al. 2019) replaces the hard geometric clamp everywhere

The hard geometric-hemisphere cutoff from the entry above carved dark facet slivers at the
terminator of low-poly smooth-normal meshes under grazing light (the classic shadow-terminator
artifact). Replaced the hard `dot(ngo, wi) <= 0 ? reject` at **every** clamp site with a smooth
ramp: `shadowTerminatorG(wi, ns, ng)` in `geometry.h` (Chiang, Li, Burley & Hovhannisyan 2019,
"Taming the Shadow Terminator"; same cubic as Cycles' `bump_shadowing_term`).

    g = cos(Ng,wi) / (cos(Ns,wi)·cos(Ng,Ns)),  softened by  -g³ + g² + g  on (0,1)

Returns a `[0,1]` factor multiplied into the surface response (NEE/connection contrib): still
**exactly 0** when `wi` is behind the true geometry (no back-face leak — leak-free is preserved),
but ramps up smoothly off the geometric horizon instead of a step. Applied **uniformly to all
modes including the mode-R reference** so R softens too and every mode stays mutually consistent:
- `backward.h` mode R — `neeLight` (spot + sphere-cone + area/quad sites) and `neeEnv`.
- `render.h` modes A/B/C — `connect` and `connectLens` camera splats.
- `bdpt.h` mode D — t==1 splat, s==1 NEE, and both interior-connection endpoints (`!isTwoSidedMat` guarded).
- `vcm.h` mode U — light-image splat, NEE, and both VC-connection endpoints (`!isTwoSidedMat` guarded).
- GPU twins in `render_cuda.cu` — `dShadowTerminatorG` threaded through `connect`/`connectLens`,
  `dConnectBDPT` (3 subsites), and `bkNeeLight`.

**Bit-identity guard.** `shadowTerminatorG` short-circuits to a plain leak-free step (return
exactly 1.0 when in front of the geometry, 0.0 behind — identical to the old hard clamp) whenever
`dot(Ng,Ns) >= 1 - 1e-7`, i.e. the shading and geometric normals coincide (flat tris, analytic
spheres). Without this guard a re-normalized `ns` differs from `ng` in the last bit, the cubic
returns ~1 (not bit-exactly 1), and every flat/analytic scene would drift by ~1e-7. With it, the
softening engages **only** once `ns` and `ng` genuinely diverge (a real smooth/crease-smoothed
mesh), so the whole flat-scene validation suite stays bit-identical.

Validated on `scraps/_iso_sphere.ftsl` (grazing directional-lit low-poly smooth sphere): the
terminator is now a smooth gradient (no facet slivers) and mutually consistent across CPU R/D/U
and GPU R/D; flat `scenes/cornell.ftsl` renders unchanged.

**Diagnostic note — disabling the clamp entirely (the "option 2" that was considered).** A debug
toggle to *fully disable* the geometric-hemisphere clamp (reverting to a pure shading-normal test,
which leaks light through geometric back faces but never facets) was considered and **deliberately
not implemented**: softening is the correct fix, so a disable toggle is only ever a diagnostic, and
plumbing a runtime flag through the CUDA kernels (device-constant, kernel signatures, host parsing)
isn't worth the surface area for a debug-only path. If ever needed to isolate a back-face-leak vs.
terminator issue, edit `shadowTerminatorG` (and `dShadowTerminatorG`) to `return dot(ng,wi) > 0 ? 1
: 1` (always 1 → no clamp, no softening) or `return 1` unconditionally, rebuild, and compare — a
one-line local change, no scene/CLI plumbing.

### Headless-spawned `-window` render on gallery.ftsl hangs with no output — NEEDS INVESTIGATION 2026-07-14

A `ftrace -in scenes/gallery.ftsl -mode R -n 1500000 -spp 8 -window` invocation *spawned
as a background process without an interactive window station* ran for 15+ min burning
~7 CPU cores (3400+ CPU-seconds) at **1% GPU**, producing **zero stdout and no image /
checkpoint**, and never exited. Meanwhile the same scene stripped to room+lights (no
isosurfaces/fog), rendered with `-mode R -device gpu -spp 64 -preview` (NO `-window`, no
`-n`), built and rendered in seconds on the GPU concurrently. Two suspects, not yet
isolated: (a) `-window` can't create its Win32 GDI window when the process has no window
station (background/detached spawn) and the code spins instead of erroring; (b) passing
`-n <photons>` to **mode R** (which is spp-based, not photon-count) mis-budgets into a
huge CPU loop. Likely (a). **Repro to confirm:** run the heavy scene once with `-window`
+ no `-n`, and once with `-n` + no `-window`, from a detached shell. If (a): make the
`-window` init detect a missing/invalid window station and fall back to `-preview`
(or error cleanly) instead of hanging. NB: do NOT `taskkill /F` a live ftrace — the
nvlddmkm teardown BSOD (below) has fired after an abrupt kill.

### `look tangent` pitches hard at path folds (gallery fly cusps) — FIXED 2026-07-14

`camera_curve` `look tangent` aims at a point a FIXED arc-length ahead (`sTgt =
sHere + 0.045*Smax`, ftsl.h ~2726). Where the path makes a horizontal U-turn (a
"fold") while also changing height, that look-ahead reaches across the fold to a
point at a very different y, so the view pitches sharply — a visible frame-to-frame
flick. The gallery `fly` curve had two folds: the dive turnaround (frames ~117-120,
old peak pitch **+24.5°** staring UP into the y=4.48 ceiling lights — the jerk the
user reported) and the loop closure (frames ~171-175, old peak **-70°** staring DOWN
at the floor, pre-existing/unreported). **Fix (scene-level):** keep y as flat as
possible ACROSS each fold and push the height change onto the straight opening
corridor — return apex lowered from y~2.6 to ~2.2. New peaks +10.6° / -15°, worst
frame-to-frame pitch swing ~30° → 6.5° (measured with `scraps/_cam_curve.py`, a
faithful re-impl of the ftsl.h sampler). Commit d46cacb. **Engine-side fix (also
done):** the latent general issue is now fixed in the `look tangent` branch. Root
cause pinned down numerically: at a fold the look-ahead chord's HORIZONTAL reach
collapses (e.g. closure frame 171: dy only -0.46 but h=0.17), so `asin(dy/L)` blows
the pitch up to -70°. Two defences, both only touching near-fold frames (well-
conditioned frames incl. legit steep dives keep their reach → byte-identical):
(1) `min_reach <frac>` (default 0.5) floors the horizontal reach used for the pitch
at `frac * lookAheadChord`; (2) `look_smooth <sigma_frames>` (default 0/off) does a
wrap-aware Gaussian smooth of the decomposed yaw+pitch so a fold's unavoidable fast
pan (the flight genuinely reverses direction) is spread over frames instead of
snapping. Implemented as a pre-pass before the frame loop (ftsl.h ~2720). Validated
A/B on the ORIGINAL unfixed control points (`scraps/_engine_fix_test.ftsl` vs
`_engine_fix_legacy.ftsl`): legacy frame 11 rakes into a ceiling light, fixed frame
11 is level. Numerically (`scraps/_cam_smooth_test.py`) the original -69.7° rake
becomes a bounded ±30° near-level pan with `min_reach 0.5 look_smooth 2`. NOTE: the
scene-level control-point fix is still the best-LOOKING result for the gallery (it
removes the sharp reversal geometrically → jerk 6.5°); the engine fix is the general
safety net so aggressive future paths degrade to a bounded pan instead of a rake.

### Mode-M dense photon map makes per-frame gather slow — FIXED 2026-07-26 (noted 2026-07-14)

With a very dense saved map (the 60M-photon gallery map deposits ~58.3M photons), each
per-camera density-estimate gather is expensive (~90–120 s/frame at 960×540, 48 spp on a
4090), so a 180-frame flythrough runs ~3–4.5 h. The dense map already yields a smooth
density estimate, so most of the per-frame spp is spent on anti-aliasing rather than noise
reduction. **Tuning opportunity:** once the map is saved (`-savemap`), re-gather via
`-loadmap` at reduced spp (~16–20) for roughly a 2–3× speedup with near-identical quality
(the map deposit — the physically expensive part — is skipped entirely). Not a bug; a
knob worth remembering when iterating on camera angles / radius on a fixed map.

**Root cause, and it is worse than linear (measured 2026-07-26).** The gather radius is
chosen *independently of the photon count* — `main.cpp:2722` (and the twins at 6130 /
6205 / 6564) set `radius = g_pmRadiusAbs > 0 ? g_pmRadiusAbs : scene.sceneRadius *
g_pmRadiusFactor`. `PhotonMap::build(r)` then sizes the uniform grid at `cellSize = r`
(`photonmap.h:84–97`), so the grid resolution is **frozen** for a given scene+radius no
matter how many photons land in it — e.g. `scraps/abs_herosplit.ftsl` reports
`grid 59x59x59` at 500k emitted *and* at 8M emitted. Photons per cell therefore grows
**linearly** with `-n`, and every pixel's 3×3×3 neighbourhood scan grows with it.

Timed sweep (`-device cpu -mode M`, 256×256, `scraps/abs_herosplit.ftsl`, wall clock
including the deposit):

| `-n` emitted | photons stored | total |
|---|---|---|
| 1 M | ~6.7 M | 65 s |
| 2 M | 13.4 M | 139 s (2.14×) |
| 4 M | ~27 M | 369 s (2.66×) |
| 8 M | ~53 M | killed — no output after ~16 min, 5.4 GB RSS |

So it scaled ≈ `N^1.4`, not `N` — and there turned out to be **two independent causes**,
both of which are now fixed. The 8 M case looks like a hang from the outside (no progress
line during the gather); it isn't, it's just this curve. Note this is **not** a
`-heroc`/`-herosplit` regression: it was measured identically with the hero bundle off,
and hero only multiplies the *stored* count (C deposits per photon), which moves you
along the same curve faster.

**Cause 2 (the superlinear part) — FIXED 2026-07-26: the map was memory-bound and the
layout was array-of-structs.** Linear photons-per-cell explains linear growth, not `N^1.4`.
The excess came from bandwidth: the map was **104 B/photon** (an 80 B `Photon` — `pos`,
`wi`, `n`, `power`, `lambda` — plus a 24 B `cie` entry), so 27 M stored photons is 2.6 GiB
and 53 M is 5.1 GiB, matching the observed 5.4 GB RSS exactly. Far past any cache, so the
gather streamed from DRAM. Two things made that much worse than it needed to be:
* A radius-`r` query scans the 3×3×3 cell box but keeps only the inscribed sphere —
  `(4/3·π r³)/(27 r³) = 15.5%`, so **~84% of candidates are rejected on a distance test
  that reads the position and nothing else**. With `pos` embedded in the record, that scan
  strided 80 B to use 24 B of it, touching every cache line and wasting 70% of each.
  `PhotonMap` is now structure-of-arrays: `pos[]`, `photons[]` (payload) and `cie[]`,
  permuted together by the counting sort so index `k` still names one photon in all three.
  The reject scan is now a dense 24 B/photon stream — **3.3× less bandwidth** on the
  dominant path.
* `Photon::wi` (the incident direction, 24 B) was written by both deposit paths and
  **never read by any gather** — the density estimate is Lambertian, so it only needs the
  normal. Deleted. The GPU had already figured this out: its `DGatherPhoton` drops `wi`
  and folds `cie*power*norm/pi` into three floats.

Net: **104 → 80 B/photon (−23%)**, so 53 M stored photons is 3.95 GiB instead of 5.13 GiB.
The GPU *deposit* record `DPhoton` lost the same dead `wi` (44 → 32 B); that buffer's
capacity is computed from free VRAM, so it is directly ~27% more photons the device can
hold in one pass. Measured back-to-back on `scraps/abs_herosplit.ftsl` (`-device cpu
-mode M`, 256², wall clock incl. deposit) with nothing else running, and the speedup
**grows with map size** exactly as a bandwidth explanation predicts:

| `-n` emitted | old | new | speedup |
|---|---|---|---|
| 1 M | 65.9 s | 62.4 s | 1.06× |
| 2 M | 141.8 s | 113.3 s | 1.25× |
| 4 M | 351.1 s | 259.0 s | 1.36× |

Method note for anyone re-measuring: run the two binaries **strictly serially** and with
nothing else on the machine. Two concurrent ftrace runs contend for every core and the
ratio comes out meaningless — that mistake produced three mutually-contradictory 4 M
numbers (1.77×, 9.35×, 11.51×) before it was caught, so the local harness
(`scraps/pm_bench.sh`, throwaway/not checked in) now refuses to start if another ftrace is
live. Bit-identical output, verified on 5 configs (CPU mode M at `-heroc 4` and `-heroc 1`,
CPU mode S, GPU mode M, GPU mode S) plus a `-savemap` → `-loadmap` round trip. The cache format went `FTPMP01` → `FTPMP02` (two blocks: positions,
then payloads); old files are refused with a message telling the user to re-deposit and
fall back to a fresh deposit.

**Cause 1 (the linear part) — FIXED 2026-07-26: the gather radius is now density-adaptive
(`PhotonMap::buildAuto`, on by default in mode M).** The radius used to come from the scene
size alone, so `build(r)` froze the grid at `cellSize = r` and photons-per-cell — plus the
3×3×3 scan cost — grew linearly with `-n`. Measured populations at the fixed radius were
perfectly linear: 350 / 1395 / 5623 / 22433 photons per gather at 1.67 M / 6.7 M / 26.7 M /
107 M stored.

The fix bins once at the starting radius `r0`, **probes the actual density** (median photons
seen by a sample of real `queryR` calls), then re-bins at
`r1 = r0 · sqrt(k / n_probe)` where the target population is

```
k(M) = kAt1M · cbrt(M / 1e6)        (kAt1M = 200, i.e. -pmcount)
```

so `r ∝ M^(-1/3)`: per-query cost `M^(1/3)`, noise `M^(-1/6)`, bias `M^(-2/3)` — both error
terms still go to zero, unlike the tempting `r ∝ M^(-1/2)` (constant population ⇒ constant
*variance* ⇒ the image never converges in noise, only in bias). The MSE-optimal 2-D bandwidth
would be `M^(-1/6)` (cost `M^(2/3)`); the cube root is the more aggressive engineering pick.

`kAt1M = 200` is **calibrated to the old look**: at the default `-pmradiusfrac` the fixed
radius already delivered ~185–210 photons per gather at 1 M stored, so ordinary renders keep
the population (and therefore the styling) they already had — only the runaway tail is cut.
At 1.67 M / 6.7 M / 26.7 M / 107 M stored the old 311 / 1256 / 5045 / 20221 become
237 / 377 / 598 / 949.

Measured on `scraps/abs_herosplit.ftsl` (`-device cpu -mode M`, 256², wall clock incl.
deposit, runs strictly serial):

| `-n` emitted | fixed | adaptive | speedup | fixed µs/M emitted | adaptive µs/M |
|---|---|---|---|---|---|
| 500 k | 27.5 s | 13.5 s | 2.03× | 54.9 | 27.0 |
| 1 M | 52.3 s | 18.6 s | 2.82× | 52.4 | 18.6 |
| 2 M | 111.8 s | 26.5 s | 4.22× | 55.9 | 13.2 |
| 4 M | 269.1 s | 36.1 s | 7.45× | 67.3 | 9.0 |

Note the *shape*: cost per emitted photon used to **rise** with `-n` and now **falls**. The
grid also unfreezes — 72 → 113 → 179 → 282 cells per axis across 250 k → 16 M emitted, where
it used to sit at 59³ forever.

This trades blur (bias) for grain (variance), so it was validated at **matched wall time**,
not matched `-n`, against an independent converged BDPT reference (`png/pmref/ref_bdpt.png`,
scored by the throwaway `scraps/pm_quality.py`). RMSE drops ~20–24% whole-frame and ~32–41%
on a flat left-wall patch where noise dominates — and adaptive at 29.5 s beats fixed at
54.5 s. So it is a quality *improvement* even at the small end, not just a speed/quality
trade.

Requirements from the original entry, all satisfied: VERSION minor bump (0.66.0 → **0.67.0**),
README note (mode-M bullet + three CLI-table rows), and an opt-out — `-nopmauto` or an
explicit `-pmradius` (which implies it) reproduces the old output **bit-for-bit**, verified on
CPU mode M, CPU mode S, GPU mode M, GPU mode S, plus a `-savemap` → `-loadmap` round trip.
`-pmcount <k>` retunes the target. The GPU shared-map path takes the same target as an `autoK`
argument to `renderPhotonMapSharedCuda` and adapts identically.

Two implementation notes worth keeping:
* `build()` was split into `buildGrid()` + `fillCie()` so the probe can bin twice while paying
  the expensive threaded CIE precompute only once.
* The probe **must be order-independent**, or `-loadmap` stops reproducing its `-savemap` run.
  The counting sort is stable, so within-cell photon order differs between a fresh deposit and
  a reload of the same map; striding over `photons[]` by array index therefore picked different
  probe points and yielded a different radius (0.008981 vs 0.008977). It now samples by *cell*
  (geometry-determined) and takes the lexicographically smallest position in the cell as the
  representative — a set minimum, hence order-free. The cell *centre* will not do: a cell the
  surface merely clips has its centre off-surface and reports a spuriously empty neighbourhood.

Modes S/U keep their own per-pass radius reduction (`R0 * pow(it, 0.5*(alpha-1))`) and are
unaffected. `-savemap`/`-loadmap` is still the way to pay a big deposit only once.

### Shared FORWARD (A/B) multi-camera pass writes all frames only at the end — FIXED 2026-07-14

The shared photon-map path (mode M, both CPU and GPU) writes each frame to disk the moment
its gather completes (crash-safe incremental output). The shared **forward** models A/B
(`renderForwardShared` / `renderForwardSharedCuda`, dispatched from `main.cpp runSharedGroup`)
now do too: `runSharedGroup` was rewritten to chunk the N-photon pass (folding a per-chunk
`seedBase` into the RNG so successive chunks draw independent photons and seedBase==0 stays
bit-identical), accumulate into per-camera SUM films, write all current films + per-camera
`.ftbuf` checkpoints every `-interval` seconds, and support `-resume` from them — mirroring
the single-camera chunked modes. Verified GPU+CPU bit-identity vs standalone, checkpoint,
resume, time budget, and energy conservation (sum/emitted = 1.000000). See commit d43fb6b.

### System BSOD/reboot on GPU context teardown (nvlddmkm.sys driver bug) — MITIGATED 2026-07-14

Twice, the whole machine bugchecked and rebooted **a couple of seconds after an ftrace
render window closed** (once after an abrupt `taskkill /F`). Forensics (minidumps copied
from `C:\Windows\Minidump` via elevated PowerShell, analyzed with `cdb -z <dump> -c
"!analyze -v"`):

- Both dumps fault **inside `nvlddmkm.sys`** (NVIDIA kernel driver, build ~Jan 2026,
  driver 591.86) at the **same function** (+0xcff9xx), at **IRQL 2 (DISPATCH_LEVEL / DPC
  context)** — bugcheck `0xBE` (ATTEMPTED_WRITE_TO_READONLY_MEMORY) and `0xD1`
  (DRIVER_IRQL_NOT_LESS_OR_EQUAL). A DPC-level fault has **no attributable user process**;
  it is the driver's own asynchronous context-teardown DPC, running *after* our process
  has exited. `nvlddmkm` Event 13 (Xid 13, "Graphics FECS Exception") also appears in the
  System log during render sessions. This is a **driver bug**, widely reported for
  RTX 4090 + nvlddmkm across driver versions — not something ftrace causes directly.
- **ftrace's own CUDA kernels are clean.** `compute-sanitizer --tool memcheck` and
  `--tool initcheck` on a mode-M render (`scenes/gallery.ftsl -mode M -camera fly090 -n
  150000 -spp 1 -r 64 48`) both report **0 errors** — no out-of-bounds or uninitialized
  device memory access. So the crash is not an app-side OOB feeding the driver a bad
  pointer; it is purely the driver's teardown path.

**Root enabler found: TDR was disabled.** `HKLM\System\CurrentControlSet\Control\
GraphicsDrivers\TdrLevel = 0` (Timeout Detection & Recovery **off** — no `TdrDelay`
override). With TDR off, a wedged GPU op / driver fault has **no recovery path** and
escalates straight to a bugcheck instead of the driver being reset. (TDR was likely
disabled so long compute kernels wouldn't be killed by the default 2 s watchdog.)

**Mitigations:**
1. **App-side (done):** added `cudaGracefulShutdown()` (`render_cuda.cu`) — a
   `cudaDeviceSynchronize()` + `cudaDeviceReset()` called from `main()` on every exit path
   (normal or exception, guarded by `HAVE_CUDA`). This destroys the CUDA context
   **synchronously, in-process, while quiescent**, instead of leaving it for the driver to
   reap asynchronously from a DPC after `main()` returns — closing the exact window in
   which the fault fires. Build now also compiles device code with `-lineinfo` (free at
   runtime) so any future GPU fault maps to `file:line`.
2. **Operational (done):** always shut renders down **gracefully** (close the live window /
   Ctrl-C / let it finish) — **never `taskkill /F`** a live CUDA process, which yanks the
   context mid-flight and is the most reliable way to hit the teardown fault.
   - **Teardown tracer:** set `FTRACE_TEARDOWN_LOG=<path>` to have `cudaGracefulShutdown()`
     append a flushed line to `<path>` around each driver call (`cudaDeviceSynchronize` /
     `cudaDeviceReset` enter+return). Each line is written with reopen+`fflush` so a hard
     reboot mid-teardown leaves the **failing step as the last line on disk**. Reading it
     after a crash tells us whether the fault is *inside* our `cudaDeviceReset` call (last
     line = "cudaDeviceReset enter", no "returned") or purely the driver's post-exit DPC
     (last line = "cudaDeviceReset returned") — which we can't touch from user space.
3. **OS-side (APPLIED 2026-07-14, takes effect after a reboot):** re-enabled TDR with a
   generous delay so the OS *recovers* a hung GPU instead of bugchecking, while still not
   killing legitimate multi-second kernels. Set via elevated `reg add` under
   `HKLM\System\CurrentControlSet\Control\GraphicsDrivers`:
   `TdrLevel=3 (REG_DWORD)`, `TdrDelay=60` (0x3c s), `TdrDdiDelay=60`. (Was `TdrLevel=0`,
   TDR fully disabled.) Verified present in the registry; **requires a reboot** to take
   effect. Revert with `TdrLevel=0`. Also worth doing: update the NVIDIA driver (591.86 is
   months old) or DDU clean-reinstall.
4. **App-side follow-up (done 2026-07-14):** the shared **mode-M** gather
   (`main.cpp runSharedPhotonMap`) was the ONE render path that never installed a SIGINT
   handler — every other mode wraps its loop in `signal(SIGINT, onInterrupt)`, but this one
   relied on the default terminate action. A backgrounded / headless Ctrl-C therefore
   abruptly killed the live CUDA context mid-gather instead of routing through
   `g_stopRequested` + `cudaGracefulShutdown()` — the exact abrupt-teardown scenario above.
   Fixed with a `SigGuard` RAII (installs SIGINT+SIGBREAK on entry, restores on every exit
   path incl. the GPU early-return) plus stop-checks: the GPU path's `writeFrame`/`liveProg`
   callbacks already return `g_stopRequested`, and the CPU fallback gather loop now breaks on
   `g_stopRequested` between frames. So an interrupt now finishes the current frame, writes
   it, and returns for the orderly teardown.

## Recently fixed

### loom rejected its own `reflect pattern:<name>` emission — FIXED 2026-07-28 (v0.93.0)

`loom/grammar/bindings.py::as_color_binding` treated a `pattern:<name>` value as an error in
every colour slot, so loom's own reader raised `ShapeError: unrecognized spectrum expression
'pattern:p_rings'` on `.ftsl` that **loom itself emits** for a `FuncPattern`-driven material
(`scenes/reflect_pattern.ftsl`, `scenes/transmit_pattern.ftsl`). ftrace accepts it — see
`patternedSpectrumParam` in `src/ftsl.h` (~2083): a lone `pattern:` in a colour slot *is* the
albedo, over a flat 1.0 base. The validator, not the emitter, held the wrong belief, and two
tests (`test_color_binding_rejects_pattern`, `test_reflect_rejects_pattern_bind`) encoded it.

Found by round-tripping the checked-in `scenes/*.ftsl` corpus through the new reader (J3c).
**Fix:** `as_color_binding` accepts `pattern:` in any colour slot and grew a `texture=` keyword
so the *texture* restriction stays exact — only `reflect` binds an image albedo, everything else
(`transmit`, `absorb`, …) rejects `texture:` with an explanatory message. `reader.py` moved
`transmit` out of `_SPECTRAL_ONLY_FIELDS` into a new `_PATTERNED_SPECTRAL_FIELDS`. The two
tests were rewritten to assert the correct behaviour, plus `test_transmit_rejects_texture_bind`.

### `type mix` lost every layer but the last on read — FIXED 2026-07-28 (v0.93.0)

`loom/grammar/reader.py::_build_material` folded a material body into a `dict`, so the repeated
`layer "<name>" <weight>` lines of a `type mix` material collapsed to one entry and the reader
silently built a **one-layer mix** from a two-layer source. Same class of bug as a dict-folded
`camera_curve` body (repeated `point` lines). **Fix:** `_props()` returns an *ordered list* of
`(key, tokens)` and `_build_mix()` reads the repeated `layer` lines off it into a real
`MixMaterial`; the generic `Block` fallback is ordered for the same reason. Regression tests:
`tests/test_grammar_block.py::test_mix_material_reads_its_repeated_layer_lines` and
`::test_duplicate_keys_are_kept_in_order`.

### VDB sampler read the *second* voxel in the low-face half-voxel shell — FIXED 2026-07-27 (v0.84.2)

`VdbGrid::sample` (`src/vdbgrid.h`) and its device twin `dVdbSample` (`src/render_cuda.cu`)
clamped the interpolation **stencil indices** to `[0, n-1]` but left the interpolation
**fraction** alone. For a point just below index 0 — `fi ∈ (-0.5, 0)`, which the half-voxel
margin explicitly admits — `floor(fi)` is `-1`, so `i0` clamped up to `0` and `i1` became `1`,
while `tx = fi - floor(fi)` stayed in `(0.5, 1)`. The sample was therefore **dominated by the
second voxel**, and got *wronger the further outside the point was* (at `fi = -0.49` it was
99% `v[1]`, 1% `v[0]`), which is backwards. The upper shell was fine, because clamping `i1`
down to `n-1` makes both stencil taps the same voxel there — so the bug was asymmetric and
only ever hit the x/y/z **min** faces.

Effect: a half-voxel-thick shell on the three low faces of every imported `.vdb`/`.nvdb`
volume rendered with density taken from the *next* slab in. Usually invisible (a cloud's edge
voxels are near-zero), but plainly wrong wherever the boundary slab differs from its neighbour
— on `scraps/cloud.nvdb` the shell read `0.325` where the edge voxel is `0.0`.

**Fix:** clamp the sample *coordinate* to `[0, n-1]` before the floor, which is what the clamp
was always meant to mean, then derive `i0`/`i1`/`tx` from the clamped coordinate. Inside the
lattice this is **bit-identical** to the old code (and drops three `floor()` calls from the hot
path, since `(int)` truncation equals `floor` for a non-negative coordinate). Applied
identically to the CPU and CUDA samplers so the twins stay in step.

Found by porting the sampler into loom (`ReadGrid.sample`, `loom/vdbio.py`) for the new
`VolumeField` term and noticing that a **360° rotation** — which must be a no-op — moved the
baked field by up to 0.32. Regression test:
`tests/test_vdbio.py::test_sampler_edge_shell_reads_the_edge_voxel_not_the_second`, plus
`test_volume_field_placement_is_lossless`, which catches it independently; both fail against
the old stencil-only clamp.

### Scene grammar: two value-less flag keywords can't share a line — FIXED (docs+scene) 2026-07-14

`camera_curve "fly" { … closed   exposure_lock … }` silently dropped the exposure lock, so
the gallery flythrough flickered (every frame auto-exposed independently). Cause: in the
FTSL grammar (`ftsl.h parseValue`) a statement's value is *always* the next token — required
for value-bearing barewords like `material white`, `look tangent`, `caps on`. So
`closed exposure_lock` parses as key `closed` with value `"exposure_lock"`; there is no
separate `exposure_lock` statement, `find(b,"exposure_lock")` returns null, and the lock
never applies (`closed` still works by accident since any non-`off` value is truthy). The
grammar genuinely can't tell `closed exposure_lock` from `material white`, so this is a
by-design limitation, **not** a parser bug to "fix." **Resolution:** put each value-less
flag on its own line — done in `scenes/gallery.ftsl`, and the misleading one-line example in
the `camera_curve` doc comment (`ftsl.h`) is corrected with an explicit NOTE. The CLI
`-exposure-lock` (global override) was always a reliable alternative. Workaround for authors:
one flag keyword per line.

### Mode `M` photon map ported to the GPU (direct density query) — ADDED 2026-07-14

Mode `M` was **CPU-only** — a serious backend gap, since the shared photon map (build
once, gather every camera) is exactly the workload a GPU wins at (e.g. a 90-frame
flythrough). Ported the **direct density query** to CUDA (`render_cuda.cu`
`renderPhotonMapSharedCuda`), reusing the existing pieces: the forward deposit runs on
the **same** `kTrace`/`shadeStep` megakernel (added a `depositPhoton` branch at every
`Diffuse`/`DiffuseTransmit` vertex, gated by a device deposit buffer — a two-pass
count-then-fill for exact sizing), the grid build reuses the tested host
`PhotonMap::build` (download hits → build → re-upload sorted photons + `cellStart`), and
the gather is a new `kGather`/`dPhotonGather` kernel mirroring the CPU `photonGather`
(specular walk, 3×3×3 grid query, per-photon-wavelength XYZ density estimate,
cross-surface reject, emitter term, Beer–Lambert interior; no env term). Gather is
spp-chunked to stay under the Windows TDR watchdog. Gated by `cudaPhotonMapSupported`
(POD-bakeable materials, no env light) + pinhole cameras + no `-pmfg`; otherwise the CPU
path runs. Host dispatch in `main.cpp runSharedPhotonMap` honors the shared
`-exposure-lock` anchor so a flythrough doesn't flicker.

**Validation** (`scenes/_gpumtest.ftsl`, dispersive-glass Cornell box, 2 mode-M cameras):
energy conserves exactly (sum/emitted = 1.000000). CPU-vs-GPU converges *together* as
photons rise (the signature of a correct port whose only difference is the RNG noise
realisation): 4M/8spp → linear relRMSE 51%, Pearson 0.836; 40M/32spp → 28%, 0.954. After
a heavy Gaussian blur to remove Monte-Carlo noise, the two radiance fields agree to
**1.7% / 3.8% linear RMSE at Pearson 0.999** with a best-fit scale of **1.00** (no
systematic brightness bias). The residual per-pixel error is pure caustic/light-source
noise (bright in linear space, slow to converge). **`-pmfg` final gather and env-lit
scenes still fall back to the CPU** — porting the final-gather sub-ray pass is future
work. `README.md` mode-`M`/CUDA/`-device` sections updated.

### Mode `M` optional Jensen final gather (`-pmfg`) — ADDED 2026-07-14

Mode `M` was a **direct** radius density query at the visible point, which inherits the
density estimate's low-frequency blur *at that surface* (softening contact shadows / fine
detail at large gather radii). Added an optional true **final gather** (`-pmfg <K>`,
`g_pmFinalGather`): at the first diffuse hit it shoots `K` cosine-weighted hemisphere
sub-rays (`photonGatherSub` in `photonmap_render.h`), traces one bounce each, and queries
the map at *those* points — so the blur now lives one bounce away, decoupling the visible-
surface sharpness from the gather radius. **Direct** lighting at the visible point is done
with low-variance next-event estimation (`BackwardRenderer::neeLight`) instead of relying
on gather rays randomly striking the light; the gather rays therefore collect indirect
(+ env + specular-direct via a `specularSeen` gate) only, so there is no double-count. The
cosine/pdf and Lambertian `1/pi` cancel to `rho(x)`, folded per-photon at the gather hit
(spectrally, at each photon's wavelength) so two-bounce colour bleed stays correct.
`K = 0` (default) keeps the original direct query — a pure superset. Validated on the
diffuse Cornell box: direct query reproduces the prior numbers (M/R=0.989, relRMSE 4.7%,
r=0.998) and final gather matches mode `R` in energy (diffuse-mask M/R=1.010). The point
of the feature shows up at a large gather radius (`-pmradius 0.06`), in the darkest 10% of
the reference (contact shadows / corner creases): the direct query suffers the classic
photon-map **boundary/corner-darkening bias** (the `1/(pi r^2)` normalisation overshoots
where the gather disc runs off the surface or into shadow) reading **M/R=0.929**, while
final gather is essentially unbiased at **M/R=0.994** (see `scraps/_shadow_bias.py`). Cost
~`K`× per sample, so pair with fewer `-spp`. `DiffuseTransmit`/`Fluorescent` visible points
fall back to the direct query. `README.md` mode-`M` description + CLI table updated. (A
secondary-hemisphere final gather is what this file previously listed as future work.)

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

### Klein glass mesh had a non-manifold pinch vertex — FIXED 2026-07-14
The Klein mesh (`scraps/klein_hunyuan_clean.obj` and its staged copy `klein_staged.obj`, used
by `settle_test_settled.ftsl` / `klein_glass_ior152.ftsl` / `klein_glass_ior242.ftsl`) failed
the new `-check-watertight` audit. Diagnosis: in RAW OBJ indexing the mesh is a *perfect* closed
2-manifold (951420 edges, every one shared by exactly 2 faces, zero boundary, zero non-manifold).
The defect was a single **3-sheet pinch vertex** — three distinct vertices (ids 151608/151609/
153154, 7+5+5=17 incident faces) snapped by the AI mesh generator to the *same* point
(~(-0.008, 0.331, -0.453), within 9.5e-8) — which only shows as a non-manifold vertex once the
audit welds coincident vertices (weld eps = bbox_diag·1e-7). The audit reported different counts
per instance (3 at full scale, 8–9 on the smaller staged copy) because the weld eps scales with
each mesh's post-transform bbox diagonal. Not a hole and not a broad self-intersection, so
MeshFix ("could not fix everything", changed nothing) was the wrong tool. **Fixed** with
`tools/repair_mesh.py` (MeshLab engine): merge-close-vertices → repair-non-manifold-edges
(remove faces) → repair-non-manifold-vertices → close-holes, taking `klein_hunyuan_clean.obj`
and `klein_staged.obj` to `[OK] … watertight, dielectric` (317038 v / 634076 f, −102 v / −204 f).
Pre-repair copies kept locally as `*.orig.obj` (scraps/ is git-ignored, so the meshes aren't
versioned — only the repair tool is). NB: this pinch was measure-zero and its render impact was
negligible; the audit is just strict about it.

_(former `light cylinder` entry moved to Resolved — it was a misdiagnosis.)_

## Tech debt

### `-savemap` / `-loadmap` are silently ignored on the CPU (GPU-only) — 2026-07-26
The photon-map disk cache is wired into `renderPhotonMapSharedCuda` only. `main.cpp`
passes `g_pmapSave`/`g_pmapLoad` at exactly one call site (~line 6638, inside the
`#ifdef`-guarded GPU branch of the shared-camera mode-M path); the CPU shared path
immediately below it, and the single-camera mode-M path at ~2721, never look at either
variable. So `-device cpu -mode M -loadmap foo.ftpm` **silently re-traces the whole photon
pass** and `-savemap` silently writes nothing — no warning, and the only symptom is the
deposit line reporting the default `-n` instead of the file's photon count. (This bit a
validation script: a CPU `-savemap`/`-loadmap` round trip "failed" because neither run had
loaded anything.) Two things to fix: (1) at minimum warn when either flag is set on a path
that ignores it — a silently-dropped flag is the worst failure mode; (2) properly, hoist
the load/save either side of `tracePhotonPass` + `pm.build` in both CPU paths, which is
easy now that the file format (`FTPMP02`) is just PhotonMap's two arrays — the GPU code in
`render_cuda.cu` (`savePhotonMap`/`loadPhotonMap`, ~line 10408) is already host-side and
operates on a `PhotonMap&`, so it can be moved to `photonmap.h` and called from both.

### Fast RGB backward (`-rgb`) omits media + textured/record albedo — 2026-07-23
The Stage-2 fast RGB backward path (`renderBackwardRGBCuda`/`bkRadianceRGB` in
`src/render_cuda.cu`, selected by `-rgb` on mode R) is a deliberately-reduced Option-B
tracer. Its scope gate `cudaBackwardRGBSupported` currently rejects (falls back to the
spectral backward with a warning): (a) **participating media** — the RGB walk has no
volume collision / NEE / Beer-Lambert leg yet, unlike the spectral `bkRadiance`; (b)
**textured / record-driven reflectance** (`reflectTex>=0`, `recBindingFor(REC_SLOT_REFLECT)`)
— the per-material RGB albedo is a single baked triple, so spatially-varying albedo isn't
represented. Proper fixes: port `dMediaSampleCollision`/`bkNeeVolume`/`dMediaTransmittance`
into `bkRadianceRGB` carrying the RGB `beta` (bake a 3-tap RGB `sigma_a`/`sigma_s` per
medium, already have `rgbAbsorb` precedent); and sample the reflectance texture per hit
into a linear-RGB albedo (reuse the forward `specLookup`→XYZ→RGB bake path at runtime).
Also deferred: image-based env and collimated beams (shared with the spectral backward's
own deferrals, Stage 1d). None are correctness bugs — the gate routes unsupported scenes
to the exact spectral tracer — just missing fast-path coverage.

### Mesh repair exists (`tools/repair_mesh.py`); isosurface cap-at-polygonise still TODO — 2026-07-14 — MESH PART DONE
`-check-watertight` DETECTS non-airtight geometry and **`tools/repair_mesh.py`** now FIXES meshes
(MeshLab engine by default: merge-close-vertices → repair non-manifold edges/vertices → close
holes; `--engine meshfix` for Attene's MeshFix on self-intersection/hole-heavy meshes; `--place-like`
re-applies a derived copy's transform). Used it to make the Klein glass mesh airtight (see the
FIXED bug entry above).
Still open — **isosurfaces**: `-check-watertight` polygonises the field and audits that, but a
`contained_by` box that clips the surface open would need a fix at *polygonise* time — emit the
flat cap on the clip plane so the marched mesh is closed by construction (marching cubes is already
watertight otherwise). `repair_mesh.py` could also just be run on an exported isosurface OBJ, but
the cap-at-source approach is cleaner. Also optional: a `-repair-mesh` CLI wrapper if we ever want
it in-process (currently repair is a separate Python step, which is fine).

### `-export-mesh` QEM decimation is pathologically slow on huge/self-intersecting meshes — 2026-07-13
`isomesh::decimateAdaptive` (QEM edge-collapse) is fine at small/medium counts but effectively
hangs on multi-million-triangle inputs. Meshing the Klein bottle `a=1.2 b=0.6 c=3.0 d=12.7`
at `-mesh-res 224` produces ~2.19 M tris (the neck self-intersects, so there's a lot of
interior surface); `-mesh-adaptive -mesh-decimate 0.18` on it ran for minutes with flat memory
(~469 MB) and made no visible progress before it was killed. Workaround for now: march at a
lower `-mesh-res` (e.g. 128 → ~716 k tris) to get a lighter mesh directly instead of decimating
a huge one. Proper fix: profile the collapse loop — likely the priority-queue / cost-update or
the link-condition neighbour scan is super-linear on high-valence, self-touching vertices; add
a progress log and a spatial cap, or switch to a vertex-clustering pre-pass before QEM.

### POV-Ray pattern/pigment/spline internal functions not ported — 2026-07-13
`src/pov_functions.h` (generated by `tools/pov_functions_gen.py` from POV-Ray's
`source/vm/fnintern.cpp`) now ports **78 of POV-Ray's ~79 internal isosurface functions**
as exact formulas, shared by the CPU (`pattern.h`/`patternEval`) and GPU
(`render_cuda.cu`/`dPatternEval`).

**DONE — Perlin noise ported (2026-07-13):** `f_noise3d` (76), `f_noise_generator` (78),
`f_hetero_mf` (29), `f_ridge` (58), `f_ridged_mf` (59) are now supported. `src/pov_noise.h`
(generated by `tools/pov_noise_gen.py`) is an exact host+device port of POV's `Noise()`:
its three init tables (`hashTable[8192]`, gradient `RTable`, and the Perlin
`NoisePermutation`/`NoiseGradients` lattice) are re-derived by replicating POV's
deterministic 32-bit-LCG init procedures and baked in as constant data, so the CPU and GPU
evaluate byte-identical noise with no runtime init. All three generators are supported
(1=Original, 2=RangeCorrected [default], 3=Perlin). Device storage uses `__device__`
globals (via `#ifdef __CUDA_ARCH__`) so the ~130 KB of tables sidestep the 64 KB
constant-memory limit. Validated visually: `sqrt(x²+z²)-1 + 0.5*f_noise3d(3x,3y,3z)`
renders a coherent Perlin-lumped cylinder identical in character to POV's f_noise3d.

**Still not ported** (the `EXCLUDE` set in the generator): `f_pattern` (77) — plus the
S-table `f_pigment`/`f_transform`/`f_spline`. These reference a whole
`TPATTERN`/`PIGMENT`/`TRANSFORM`/`Spline` object via `private_data`; they are
function-*wrappers* around POV's texturing engine, not standalone math. Out of scope until
(if ever) that engine is ported. Parser rejects any unported name as an "unknown
identifier", so scenes fail loudly rather than silently.

### Isosurface `contained_by` is box-only — add a sphere/curved container — 2026-07-13 — DONE 2026-07-13
**DONE:** `contained_by { sphere { center <x y z>  radius r } }` is now accepted (`ftsl.h`
`addIsosurface`), storing `Container::Sphere` + world `sphereCenter`/`sphereRadius` on the
`Implicit` (box stays the default). `intersectImplicit` (`implicit.h`) and the device twin
(`render_cuda.cu`) clip the ray against the actual container (sphere → quadratic; box →
face-tracking slab) and carry the container's outward normals for cap shading. The AABB
`im.bounds` is still the BVH-leaf/broad bound (set to the sphere's AABB for sphere
containers). Validated on `f_enneper`/`f_klein_bottle` (rounded clip vs box facets) — see
`scraps/gen_container_test.py` → `png/iso_container_grid.png`.

**What:** an isosurface's `contained_by { min <x y z>  max <x y z> }` is the *only*
container shape we support — an axis-aligned box (see `ftsl.h` `addIsosurface`
~line 1545; the 8 corners are transformed to world and reduced to an AABB stored as
`im.bounds`). POV-Ray also lets the container be a `sphere` (and in fact any shape).
**Why it matters:** for a surface that reaches the container wall (any *unbounded*
surface like `f_enneper`, or a solid lump that pokes out), a box clips it along **flat
planes**, so the cut reads as hard angular facets. A **sphere** container clips along a
smooth curved boundary, so the unavoidable cut looks like a natural rounded edge instead
of a sawn plane — this is why hand-tuned POV enneper/klein renders frame cleanly and ours
show flat patches. It's container ergonomics, not a math gap: both engines must clip an
infinite surface *somewhere*; the sphere just hides the seam.
**Where / proper fix:** `ftsl.h` `addIsosurface` — accept `contained_by { sphere {
center <x y z> radius r } }` (keep `min`/`max` box as the default). Store the container
shape on the `Implicit` (currently just `im.bounds`, an AABB used to clip the ray in
`implicit.h intersectImplicit` ~line 246). The ray-clip step must then intersect the ray
with the actual container (sphere slab → quadratic) rather than the AABB, and the CUDA
mirror (`render_cuda.cu` `dIntersectImplicit`) needs the same. AABB stays as the BVH-leaf
bound regardless.

### Isosurface container has no cap/`open` control (and no proper cap at all) — 2026-07-13 — DONE 2026-07-13
**DONE:** `intersectImplicit` (CPU `implicit.h` + device `render_cuda.cu`) now caps the
container. In the **default capped** mode a ray that enters the container already inside
the solid (`f < 0` at the near clip) registers a hit on the container's near face (a NEAR
cap); a ray that reaches the container exit still inside the solid registers a hit on the
far face (a FAR cap, only when the far clip is the container itself, so bounce/transmission/
shadow rays originating inside the solid seal correctly). Both use the container's outward
normal and the isosurface material. The **`open`** keyword on the `isosurface {}` block
(`ftsl.h`, default `capped = true` for expr fields) suppresses both caps, revealing the
cut edge. Fully-bounded surfaces (`f > 0` at entry) never trigger a cap, so SDF/CSG leaves
are byte-identical. Validated: `f_enneper`/`f_klein_bottle` render as cleanly sealed solids
by default and open shells with `open`.

**What:** where an isosurface's solid interior (`f < 0`) is sliced by the container wall,
we render **neither** a clean sealed cap **nor** a clean open edge. `intersectImplicit`
(`implicit.h` ~line 245) clips the ray to the container and reports the first field
*sign change* inside it; it never treats the container faces as geometry. So a solid cut
by the box returns the next interior crossing (its back/inner wall) or passes straight
through — reading as odd flat interior patches or see-through holes.
**Background (what a "cap" is):** convention is `f < 0` = solid inside, `f > 0` = outside.
When the container plane cuts through solid material you must choose: **capped/"closed"**
(POV default) draws that slice as a flat face of the object's material, sealing the solid
flush with the wall (looks cleanly sawn off); **`open`** (POV keyword) omits the wall so
the surface just ends and you see into/through the interior. Only matters for surfaces
that actually *reach* the container (`f_enneper`, the klein bottle's outer shell); a fully
bounded surface never touches the wall so the choice is moot.
**Why it matters:** without a real capped mode, box-cut solids can't be shown as clean
solids; without an `open` option, thin-shell / hollow looks aren't authorable. Today's
behavior is effectively a broken third option.
**Where / proper fix:** in `intersectImplicit` (CPU) and `dIntersectImplicit`
(`render_cuda.cu`), detect the case where the ray enters the container already inside the
solid (`f < 0` at the near clip `t0`, or exits the far clip `t1` still `f < 0`) and, in the
**default capped** mode, register a hit on the container face itself (position = clip
point, normal = the container's inward face normal, material = the isosurface material).
Add an `open` toggle to the `isosurface {}` block (`ftsl.h`) that suppresses these caps
(current behavior). Pairs naturally with the sphere-container item above (a sphere cap is
a spherical patch with the sphere's radial normal). Validate on `f_enneper` (should read
as a cleanly-capped solid by default, an open shell with `open`).

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
- **No `emissiveFactor` import.** Emissive glTF materials load unlit. The underlying
  mechanism now exists — mesh-emitter area lights shipped in 0.41.0 (C5): an FTSL `emit`
  material bound to a `mesh` registers an `EmitterShape::Mesh` sampled light. What's
  missing is wiring glTF's `emissiveFactor`/`KHR_materials_emissive_strength` into that
  path (set `Material::emit` from the factor and call `Scene::addMeshLight` for the
  imported range). Bind an FTSL `emit` material to the mesh as a workaround.
- **No skinning, morph targets, animation, or sparse accessors.** Static bind pose only.
- **Non-triangle primitives** (points/lines/strips/fans, `mode != 4`) are skipped with a
  note; only `mode 4` (TRIANGLES) is baked.
- Materials are created **per glTF material, not deduplicated across meshes/files**. (A
  `mesh` still bakes its triangles into `Scene::tris`; use `mesh_asset`/`mesh_instance`
  for shared instanced geometry — see below.)
The core path (buffers/GLB, node transforms, POSITION/NORMAL/TEXCOORD_0, indexed +
non-indexed tris, metallic-roughness → BSDF) is validated on CPU and GPU.

### Instancing memory saving is CPU-only (GPU expands instances) — 2026-07-12 — DONE 2026-07-13
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

**RESOLVED 2026-07-13 — device two-level BVH.** `render_cuda.cu` now mirrors the CPU.
`Scene::bvh` (TLAS) is uploaded **verbatim** in all cases; its prim-index layout
`[tris | spheres | implicits | instances]` is understood by the device leaf dispatch in
both `closestHit` and `occluded` (a prim index `>= nTris+nSph+nImplicits` is an instance
leaf). New device structs `DBlas { nodeOff, triOff, primOff }` and `DInstance { Lm[9],
Lt[3] (world→local affine), Nm[9] ((toWorld)⁻ᵀ normal matrix, host-precomputed), blasId,
matOverride }`. Each `Blas` contributes its local-space tris/BVH-nodes/primIdx to three
**concatenated shared pools** (`blasTris`/`blasNodes`/`blasPrim`) uploaded ONCE, and a
`DInstance` places it via an affine — so N copies cost one `DInstance` each, not N× tris.
Device `blasClosest`/`blasOccluded` walk the shared sub-BVH in BLAS-local space (48-deep
local stack); `affPoint`/`affDir` transform the ray (dir NOT renormalized, so local `t` ==
world `t`, matching the host `Blas`); `instanceHitToWorld` maps the hit back (normal by
`Nm`, shading normal re-oriented, matOverride applied). Validated with
`scraps/instance_test.ftsl` (4 tori sharing one 16 384-tri BLAS, incl. a material override
and a mirror): GPU mode B matches CPU backward reference mode R at Pearson r=0.996 (MAE
~1/255; residual is the forward-vs-backward mirror-highlight difference). Implicit scenes
still render correctly (the leaf-dispatch bounds change is a no-op with no instances).
Device geometry memory is now flat in instance count.

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
**AABB** (box approximation; true mesh containment deferred — **since closed**, see
"True mesh containment for fog bounds" below). Media are resolved in a
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

**True mesh containment for fog bounds — DONE 2026-08-03.** Closes the gap left above.
`bounds { object "<mesh>" }` no longer degrades to the mesh's AABB: the mesh is
**solid-voxelized** at load into an occupancy lattice (`src/meshvoxel.h`, new
`MediumBound::Mesh` + `Medium::boundGrid`) and the fog fills its true interior. Method is
**signed-crossing (generalized winding)** x-scanlines rather than parity, because the
meshes people import are routinely several closed bodies or self-intersecting shells
(`cloud1.glb` is two) and parity double-toggles those into hollows; winding renders their
union. Triangles are visited once and scattered into the rows they cover, so cost is
O(tris + covered rows): 1.85 M triangles bake in ~1 s. The lattice is a plain `VdbGrid`,
so it rides the existing sparse-brick device upload and `dVdbSample` — **GPU support came
free**, no new plumbing. `voxels <n>` (default 160) sets resolution on the longest axis.

Two bugs found and fixed while validating, both worth remembering:
- **`Tri::gn` is not populated during the load.** `Tri::finalize()` runs in
  `Scene::build()`, *after* the loader's deferred medium sweep, so the voxelizer read
  `gn == (0,0,0)` for every triangle and gave every crossing the same winding sign. The
  winding then never returned to zero and each scanline filled solid from its first
  crossing to its last — i.e. the mesh's **x-convex hull**. Single convex shapes hid it
  perfectly (hull == shape); only a multi-body test exposed it. Fixed by deriving the
  facing from `det` (= `cross(v1-v0, v2-v0).x`), the same quantity, computed from the
  vertices in hand and correct at any load stage. *Anything else running inside the
  loader must not read `Tri::gn` either.*
- Majorant estimation sampled `Medium::densityAt`, which returns 0 outside a membership
  bound, so a coarse 25³ probe of a thin or low-volume-fraction mesh/implicit shape could
  majorise to ~0 and silently delete the medium. Split out `Medium::densityFieldAt` (the
  field with no membership carve) and estimate from that — membership only multiplies by
  0 or 1, so the uncarved peak is always a valid conservative majorant.

Validated against exact lattice-point counts by the permanent regression check
**`tools/check_meshvox.py`** (`python tools/check_meshvox.py [--res 200]`), which builds
each mesh, reproduces meshvox's lattice in numpy and compares: sphere 50.0 % vs 50.1 %
expected, two **overlapping** spheres 54.9 % vs 55.0 %, two **disjoint** spheres 38.3 %
vs 38.3 % — all within the meshes' own inscribed-faceting error. The *disjoint* case is
the one that catches the convex-hull regression above; a convex-only test suite cannot.
The one deviation is a surface lying exactly on a
lattice-centre plane (axis-aligned box face, 94.2 % vs 95.6 %), where membership is a
floating-point coin flip; it errs consistently toward **erosion**, the safe direction for
a fog bound, and is sub-voxel against a boundary the trilinear ramp softens anyway.
Rendered end-to-end on GPU (mode D, `scraps/_meshbound_test.ftsl` → `png/meshbound_cloud.png`).

**`mesh { shape_only yes }` — DONE 2026-08-03.** The other half of the above: a mesh whose
job is to *define* a volume should not also be drawn. Its triangles are loaded, handed to
the medium bake, then stripped from `Scene::tris` before the BVH is built, so they neither
render nor cost anything. Safe to renumber because `Scene::tris` is indexed only through
`MeshGroup::triStart/triCount` (fixed up in `stripShapeOnlyMeshes`) — mesh area lights
*copy* their triangles into `Emitter::meshTris`, and `shape_only` is refused on an emissive
material regardless. Refused without a `"name"`. `-check-watertight` reports such a mesh as
skipped rather than as a vacuously airtight 0-triangle object.

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

## Resolved

### Nested-dielectric exterior IOR hardcoded to 1.0 (glass-in-water wrong) — DONE 2026-07-14 (Level 0)
Every dielectric interface previously assumed the exterior medium was vacuum (IOR 1.0), so
glass-in-water refracted 1.0↔1.52 instead of 1.33↔1.52 and nested/overlapping solids were
wrong in **all** modes. Fixed at ROADMAP §7 **Level 0** (Schmidt & Budge 2002 priority
field): each path now carries a tiny LIFO medium stack (`src/medium_stack.h`, device
`DMediumStack` in `render_cuda.cu`); at every dielectric hit the exterior IOR comes from the
enclosing highest-priority medium, and the lower-priority surface inside an overlap is
suppressed (ray passes straight through). Wired through **all** integrators: CPU R/A/B/C/D/M/S/U
(`backward.h`, `render.h`, `bdpt.h`, `photonmap_render.h`, `sppm_render.h`, `vcm.h`) and GPU
forward/backward/BDPT/photon (`render_cuda.cu`). **Safe fallback:** the priority rule fires
only when *both* sides carry an explicit `priority` (air/empty stack always valid at 1.0), so
priority-free scenes render bit-identically. An ahead-of-time scene audit warns when two
dielectric bounding volumes overlap and either lacks a `priority` (isosurfaces compared by
their `contained_by` container bounds — conservative, never misses a real overlap). Validated:
mode-R priority vs no-priority differ across 33.6% of pixels (mean 5.5/255) — well above render
noise; GPU-priority matches CPU-priority reference to mean 1.3/255; modes C/M energy-conserving.
- **Remaining gap (pre-existing, not a regression):** device BDPT (`dRandomWalk` in
  `render_cuda.cu`) still does **not** apply Beer-Lambert interior absorption across in-glass
  segments — it never did, and Level 0 only added the priority-driven exterior-IOR resolution
  there, not absorption. GPU BDPT already falls back to CPU for frosted/colored glass anyway
  (`cudaBdptSupported`), so colored-glass absorption is exercised on the CPU path; clear-glass
  device BDPT is unaffected. Proper fix: thread `stk.topMat()` absorption into `dRandomWalk`'s
  segment loop the way the forward/backward device paths do.
- **Future (ROADMAP §7 Levels 1/2, deferred):** true physical stacking of co-located media and
  interpenetrating volumes remain opt-in tiers beyond Level 0.

### Mode `P` composite is not progressive; `R`/`D` have no disk resume — DONE 2026-07-13
Both gaps closed. `-time`/`-noise`/`-forever`/`-preview`/`-interval` and `-resume`/
`-checkpoint` now cover **all** the accumulating image modes — the forward camera models
`A`/`B`/`C`, the spp reference modes `R` (backward) / `D` (BDPT), and the composite `P`.
- **Mode `P` is now progressive** (`runCompositeProgressive`, `main.cpp` ~line 1986). The
  view-dependent first-hit pixel classification is computed **once** (`classifyComposite`)
  and reused; the driver then alternates forward (model-B, `N` photons) and backward
  (camera-side, `spp`) batches into two persistent SUM films, adapting the batch toward
  ~0.5 s so early frames appear fast. After each batch it re-fits the forward→backward
  scale `s` and re-blends (`compositeFromFilms`), writing the image + a status line every
  `-interval`. The old single-shot `renderComposite` wrapper was deleted.
- **`R`/`D` now disk-resume** through `runSppProgressive`, reusing the single-film
  `Checkpoint`/`writeCheckpoint`/`readCheckpoint` format keyed on **spp** (the mode byte is
  folded into the identity guard so an `R` checkpoint can't be loaded as `D`, verified).
- **Mode `P` gets a dual-film checkpoint** (`CompositeCheckpoint`, magic `FTPCM02`) storing
  the forward SUM + backward SUM + their counts + the forward energy tally.
- **Seed decorrelation on resume:** fresh samples are biased past the loaded ones via
  `SppProgress::sampleBase` (CPU: added to the per-chunk seed; GPU: XORed into the megakernel
  seed base) so continued samples are an independent noise realization. Validated: `R`
  58 196→116 545 spp with noise tracking 100/√spp exactly (0.41 %→0.29 %); `D`
  1016→2052 spp (3.14 %→2.21 %); `P` 36.2 M photons/4636 spp→56.5 M/7228 spp with the
  diffuse-side residual falling 0.0281→0.0226 — proving the resumed samples reduce variance
  rather than re-tracing identical paths.

### `light cylinder` "emits no illumination" — NOT A BUG (misdiagnosis) — RESOLVED 2026-07-13
The original 2026-07-11 report claimed a `light cylinder` glows but lights nothing on
both CPU and GPU, citing an "auto-exposure 1.54e-14, i.e. zero contribution." That
inference was wrong on two counts, and re-testing shows the cylinder light works
correctly on **both** backends.
- **A ~1e-14 auto-exposure is normal here, not "zero light."** This renderer uses
  physically-scaled blackbody SPDs whose absolute radiance is ~1e13 W/m²/sr, so the
  content-based auto-exposure lands around 1e-14 for *any* such scene — the stock
  Cornell box (`-scene cornell`) reports `auto-exposure=8.87e-14`.
- **The isolation scene had no explicit `power`**, so `absPower` was a no-op and the
  emitter surface kept the raw (astronomically bright) blackbody radiance. A directly-
  visible emitter that bright dominates the auto-exposure anchor and crushes the
  genuinely-lit wall to near-black in the tonemap. **A `light sphere` in the identical
  no-`power` isolation scene behaves the same way** — so it was never cylinder-specific.
- **Controlled proof.** In absolute mode (each light given an explicit `power`, so a
  fixed sensor gain is used instead of content-based auto-exposure), a cylinder and a
  sphere light of equal power illuminate the wall essentially identically: at `power 30`
  wall-region mean ≈ 1.32 (cyl) vs 1.11 (sph); at `power 4000`, 21.33 (cyl) vs 19.65
  (sph). GPU forward (mode B) matches the CPU backward (mode R): wall mean 21.60 vs
  21.33. The `neeLight`/`neeVolume` switches in `backward.h` already dispatch the
  Cylinder shape (`sampleCylinderVisible` for the un-capped front-facing arc; uniform
  `samplePoint` for capped capsules), and forward photon emission selects it via the
  power CDF — all correct.
- **Repro (now shows a properly lit wall):** `scraps/cyl_test.ftsl` was updated to give
  the tube `power 4000`; `ftrace -in scraps/cyl_test.ftsl -mode R -device cpu -spp 128
  -o png/cyl_test.png` shows the wall lit with correct falloff around the tube.
- **Lesson for the tracker:** don't read a tiny auto-exposure as "black"; verify with
  absolute-`power` lighting or by measuring HDR/PNG pixels of a receiver away from the
  directly-visible emitter.

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
- **Container-aware caps (2026-07-13):** the mesher originally *always* box-capped at
  `im.bounds` (the AABB), ignoring the isosurface's `contained_by` shape and `open` flag — a
  sphere-container isosurface would mesh with flat AABB caps that bulge toward the box corners
  instead of a clean spherical cut. `marchImplicit` now switches the cap SDF on `im.container`
  (`Container::Sphere` → `sphereSDF(center,radius)`, else box), so the mesh boundary matches
  what the ray tracer / `klein_explorer.html` show. When the isosurface is `open`
  (`im.capped == false`) the field is left un-sealed (`augEval = im.eval`), so the surface's own
  cut edge stays an open rim rather than being force-capped. `src/isomesh.h` ~line 84.
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

### `.nvdb` / `.vdb` volume import (`density vdb:<file>`): dense bake, float-only
- **What:** `medium { density vdb:cloud.nvdb }` imports a NanoVDB FloatGrid (`src/vdbgrid.cpp`,
  the only TU that includes the vendored `NanoVDB.h`) **or a native OpenVDB `.vdb`**
  (`src/vdb_openvdb.cpp`, no OpenVDB/NanoVDB dependency). `loadVdbGrid` dispatches on the file
  magic (`"VDB "` → the native reader, else the NanoVDB path). On load the sparse grid is
  **baked into a dense float lattice** covering its active index-space bounding box (`VdbGrid`,
  `src/vdbgrid.h`). A CPU+GPU-shared trilinear sampler reads that lattice.
- **Native `.vdb` reader:** parses the file container (header, grid descriptor, per-grid
  compression/metadata/transform), the `float 5_4_3` tree topology, and BLOSC+ACTIVE_MASK+
  HalfFloat leaf/tile buffers by hand. Blosc's **LZ4** codec is decoded with a vendored
  single-file LZ4 (`src/third_party/lz4.*`) plus a compact reimplementation of blosc1's
  chunk/block/byte-shuffle framing — validated **bit-for-bit against python-blosc** on the
  official OpenVDB smoke/sphere/cube samples (`scraps/_vdb_parse.py` reproduces the parse with
  read-position exactness). Render-validated: `scraps/vdb_smoke_native.ftsl` shows the smoke
  plume.
- **Limitations:**
  1. **Dense memory** — RAM/VRAM scales with the index-space AABB (nx·ny·nz·4 bytes), not the
     active voxel count, so a large but mostly-empty sparse volume can blow up. A safety cap
     (512 M voxels) rejects pathological grids with a clear error rather than OOM-ing.
  2. **Float grids only** — non-float NanoVDB builds (Fp4/Fp8/Fp16/level-set index grids) and
     non-`Tree_float_5_4_3` `.vdb` grids are rejected with a message. Convert to a float fog
     volume first.
  3. **`.nvdb`: uncompressed only** — Blosc/ZIP-compressed `.nvdb` files are rejected (we
     deliberately don't vendor zlib/blosc for NanoVDB). Re-export uncompressed. **`.vdb`: LZ4
     blosc only** — the native reader decodes blosc **LZ4** (Houdini's and OpenVDB `-l lz4`
     default) but not blosc **BloscLZ / Zlib / Zstd** or standalone **ZIP**; those are reported
     with a clear "re-export with LZ4" message. Vendoring zstd/zlib for the other codecs is a
     follow-up if an asset needs it.
  4. **Quoted path not accepted** — the FTSL value grammar takes one bareword token, so the path
     must be unquoted: `density vdb:scraps/cloud.nvdb` (no spaces). A quoted `vdb:"..."` form
     would need a small `parseValue` change to consume a trailing String.
  5. No emission/temperature grids (fire), no motion-blur/velocity grids.
- **Proper fix (if needed):** a native **sparse** device accessor (sample the tree directly on
  CPU+GPU instead of baking dense) to drop the memory cost and support huge volumes; fp16 dense
  option; a second float grid for blackbody emission; the remaining blosc codecs. Deferred until
  an asset needs it.

### GPU parity for §1–4 features — DONE (implicits + patterns + translucency)
- **What:** the whole §1–4 CPU feature set is now ported to the GPU forward + backward
  tracers: **implicit surfaces** (5a), **procedural patterns** (5b), and **dielectric
  translucency** (5c — frosted glass = roughness lobe on both dielectric lobes; colored
  glass = Beer–Lambert `absorb` interior tint). The only remaining fallback is **GPU BDPT**
  (mode `D`), whose MIS kernel still can't reproduce per-hit pattern BSDFs or frosted/
  colored glass, so those scenes fall back to the CPU BDPT.
- **Implicit surfaces — DONE (2026-07-11, step 5a):** `render_cuda.cu` gained device
  twins `DFieldNode`/`DImplicit`, a postfix field evaluator (`dFieldEval`/`dFieldLeafSDF`/
  `dFieldGradient`, originally all FP64 for sphere-trace bisection robustness — since
  0.19.14 the march/refine runs FP32 on mirrored pools, see the 2026-07-22 GPU-implicit
  entry; gradients/normals and media bound-fields are still FP64), and
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
- **Re: "wavefront helps divergent scenes AND small GPUs" (notes/todo.txt question):** it's
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

### DONE (2026-07-30, v0.116.0): mode W is the only render mode with NO GPU path
**Fully closed by TODO.md §N/N3a+N3b+N3c.** Mode W now rides the same `kBackward`
megakernel as mode R with the estimators swapped: seven `DScene` knobs (`bkWhitted`, `bkGrid`,
`bkGiDirs`, `bkGiGrid`, `bkGiBounce`, `bkHeroSplit`, `bkAmbient`) uploaded from a `WhittedOpts`
(`src/render_cuda.h`), device twins of every lattice helper (`dRadicalInverse2` /
`dRadicalInverseB` / `dRot05` / `dWhittedSample` / `dWhittedLambdaU` / `dWhittedGlossyDir` /
`dWhittedAttenuate` / `dGridUV`), the G×G light quadrature in `bkNeeLight`/`bkNeeLightHero`, and
the whitted branches of `bkInteract` / `bkRadiance` / `bkRadianceHero`. Measured on
`scraps/n3_gpu.ftsl` (800×520 `-spp 16`, absolute exposure, `scraps/n3_check.py`): **99.39 %** of
channel samples bit-identical CPU↔GPU, 99.96 % within one 8-bit code, **zero** pixels inside a
≥3 px-wide disagreeing region (all residual is single-pixel silhouette/shadow-edge slivers from
the device's fp32 `Real` + coarser `RAY_EPS`); **12.1 s → 0.3 s**, i.e. ≈40×, well above the ~8.5×
mode-R ratio predicted below.

**Dispersive materials followed in N3b (v0.111.0).** `bkRadianceHero`'s body became
`template<bool AllowSplit> bkRadianceHeroLoop(...)`, the device twin of N1's `radianceHeroLoop`:
the `<true>` instantiation fans each live secondary λ into its own monochromatic sub-path from
bounce `b+1` (own Snell direction, own copied `DMediumStack`, own `L[i]` slot, no ×C boost) and
continues the hero unboosted, while `<false>` keeps the old de-hero. `if constexpr` (not a runtime
`if`) is what makes this safe on the device: the `<false>` body contains no recursive call, so
re-entry is provably one level deep with a statically-sized frame — no `cudaLimitStackSize`, no
`-rdc`. `sceneHasDispersiveMat` is gone. Measured on `scraps/n3b_gpu.ftsl` (the N3a box plus
SF10/BK7/diamond balls and a half-mirror pane, 400×260 `-spp 1`, absolute): 99.561 % bit-identical,
max |dLuma| 0.144 / |dChroma| 0.105 codes per 20 px block (`scraps/n3b_check.py`, limit 1.5), and
0 blob interior even under the strict `n3_check.py` bar. The N3a scene re-rendered byte-for-byte
identically, proving the refactor is inert off the split path.

That commit also fixed a **latent divergence**: `buildUpload` now defaults
`sc.bkHeroSplit = hero::gSplit`, matching `BackwardRenderer::heroSplit` / `Renderer::heroSplit`.
Before, GPU mode R *silently ignored* `-herosplit` and always de-hero'd, so CPU and GPU mode R
disagreed by ~4.4 codes of block luma on a glass scene whenever the flag was passed.

**`-gi <n>` followed in N3c (v0.116.0) — the last mode-W fallback is gone.** The gather is a
recursion (a diffuse vertex shoots `giDirs` rays back into `radiance()`), which on the device
would need `-rdc` plus a hand-sized stack, so the depth became a **second, independent
compile-time parameter**: `bkRadianceHeroLoop<bool AllowSplit, int GiDepth>`,
`bkRadiance<int GiDepth>`, `bkRadianceHero<int GiDepth>`, `bkInteract<bool AllowGather>`. Only
`GiDepth == 0` contains a gather call; the rays it spawns are `GiDepth == 1`, whose body
contains none, and a split sub-path *inherits* its parent's `GiDepth`, so the deepest chain is
`<true,0>` → gather → `<true,1>` → split → `<false,1>` — three statically-sized frames, no
`-rdc`. (`bkInteract` takes a `bool` because the hero tracer handles Diffuse/DiffuseTransmit
inline and routes only dispersive materials there, so it always passes `false` — 2
instantiations, not 3.) `dGiDir` / `dGiPhases` are the world-space Fibonacci-spiral lattice,
Cranley-Patterson-rotated by scrambled radical inverses in bases 7/11. `cudaBackwardWhittedSupported()`
now narrows **nothing** beyond `cudaBackwardSupported()`.

Measured on `scraps/cor_gi.ftsl` (Cornell box, 240×240, `-spp 1`, `scraps/n3b_check.py`,
1.5-code bar): `-gi 32` CPU↔GPU **98.681 %** bit-identical, 99.914 % within one code, max
|dLuma| **0.328** / |dChroma| **0.295** codes per 20 px block → PASS; strict `n3_check.py`
gives 40 hot pixels and **0 blob interior**. The scalar `bkRadiance`/`bkGiGather` path
(`-heroc 1 -gi 32`) passes independently at dLuma 0.573 / dChroma 0.733. The normalisation
invariant holds bit-for-bit on the device: `scraps/gi_collapse.ftsl` renders **pixel-identically**
at `-gi 0` and `-gi 32`, i.e. the gather collapses to exactly `rho * ambient` in an empty scene,
so switching `-gi` on never steps the exposure. The N3a / N3b / N3d beds all re-verified
(99.447 / 99.533 / 99.611 % bit-identical, dLuma ≤ 0.153) and all 14 physics self-tests pass.

N3c also fixed three smaller latent divergences found while validating:
- The **device scalar `bkRadiance`'s gather bounce cap was missing the `min`** — it used
  `bkGiBounce` outright where the host uses `std::min(maxBounce, giBounce)`, so a `-gi-bounce`
  larger than `-max-bounce` would have let a gather ray trace *deeper* than the camera path.
- **`Fluorescent`'s NEE omitted the gi context on *both* sides** (`backward.h`
  `MatType::Fluorescent`, `render_cuda.cu` `D_FLUORESCENT`), so a fluorescent surface paid
  `lightGrid²` shadow rays inside a `-gi` gather while every Diffuse vertex paid `giGrid²` — the
  two materials disagreed on the shadow lattice for no reason. Now both pass `gi` / `gi.depth`.
- **`--help` still said mode `W` was "(CPU only)"**; it now says "(CPU or GPU)".

Also worth recording: **`-rgb` is now refused in mode W** (message in `main.cpp`) rather than
silently taken — the fast RGB kernel is a separate reduced tracer with no deterministic
estimator, so it would return exactly the noise mode W exists to remove.

<details><summary>original entry (v0.107.0)</summary>

- **Where:** `main.cpp:4292` — `const bool gpuBackwardMode = (mode == 'R' && !g_whitted);`
  is the entire gate. Mode W is mode R with `g_whitted`, so that one clause routes it to the
  CPU unconditionally. `render_cuda.cu` has zero whitted plumbing: the device `DScene` carries
  `bkDirectOnly` (mode R's `-direct-only`) but no `bkWhitted` / `bkGrid` / `bkGi` / `bkAmbient`.
  Every other mode (A/B/C forward, D BDPT, M photon map, R backward spectral + `-rgb`, S SPPM,
  U VCM, G2 iso preview) has a device backend.
- **Measured payoff (RTX 4090 vs 12 CPU threads, 480x300, `-heroc 8`, this machine):**

  | scene | mode | CPU | GPU | ratio |
  |---|---|---|---|---|
  | `cornell.ftsl` | R, 32 spp | 1.8 s | 0.2 s | ~9x |
  | `_room_of_gyroids_f12.ftsl` | R, 8 spp | 14.5 s | 1.7 s | ~8.5x |
  | `_room_of_gyroids_f12.ftsl` | **W, 1 spp** | **5.0 s** | (none) | — |

  So a device mode W projects to ~0.6 s on the gyroid room at 480x300 (~3 s at 960x600, down
  from 25.7 s) — enough to turn the viewer's banded progressive preview into a per-pose redraw.
- **Why it should beat 8.5x:** that ratio is strikingly low for a 4090 over 12 threads, which
  says the spectral backward megakernel is divergence/register-bound rather than throughput-
  bound. Mode W is *more* coherent than mode R by construction — fixed `lightGrid^2` quadrature
  instead of one random shadow ray, `whittedAttenuate`'s deterministic cutoff instead of
  Russian roulette (threads in a warp now terminate together), mirror direction instead of a
  sampled glossy lobe, dominant branch instead of a Fresnel coin flip. All four remove warp
  divergence, so mode W is closer to the GPU's happy path than anything already ported.
- **Why it is still not trivial:** `kBackward` is a hand-written *mirror* of `backward.h`, not
  shared code, so all ~30 `whitted` sites in `backward.h` need device twins. And the usual
  escape hatch does not apply — `render_cuda.h` explicitly allows the stochastic modes to be
  "an independent noise realization that agrees to within Monte-Carlo noise", but mode W has no
  noise to hide a mismatch behind. Any disagreement in the quadrature, the radical-inverse
  lattices (`whittedSample` / `whittedLambdaU`, incl. the `rot05` offsets), or the de-hero point
  is a *visible deterministic* CPU/GPU difference. The port therefore needs a bit-exact A/B
  against the CPU as its acceptance test, not a statistical one.
- **Proper fix:** add `bkWhitted` / `bkGrid` / `bkGiGrid` / `bkAmbient` to `DScene`, port the
  whitted branches into `bkInteract` + the `kBackward` light loop, drop the `&& !g_whitted` from
  the gate, and gate on a new `cudaBackwardWhittedSupported()`. Do it on the **spectral**
  `kBackward`, not `kBackwardRGB` — the spectral device scope (media, fluorescence, textured
  albedo, constant env, lens) is far wider than the RGB one, so this would also give the viewer
  a much broader-scope GPU preview than `-rgb` reaches today.
- **Spectral vs `-rgb`: do NOT assume spectral is much slower.** Measured on GPU, same scene/spp:

  | scene | spp | `-rgb` | spectral `-heroc 8` | spectral penalty |
  |---|---|---|---|---|
  | `cornell.ftsl` | 1024 | 0.7 s | 1.2 s | 1.7x |
  | `_room_of_gyroids_f12.ftsl` | 64 | 9.5 s | 12.1 s | **1.27x** |

  Not 8x — hero sampling carries all 8 wavelengths down ONE shared BVH walk, and traversal is
  the expensive part, so only the (cheap) shading arithmetic multiplies. The heavier the
  geometry the smaller the penalty.
- **Where `-rgb` would genuinely win is de-hero, not speed.** For the stochastic modes `-rgb`'s
  advantage is convergence (no chroma noise); at mode W's 1 spp there is no noise, so that
  evaporates and only the 1.27-1.7x per-sample cost is left. But spectral mode W on a glass
  scene collapses the frame onto one wavelength (the de-hero DEBT below) and needs up to 16
  passes — call it ~20x the RGB cost — whereas the RGB walk has no wavelength dimension at all
  and so is honestly 1-spp-clean on plain glass (`MatType::Dielectric` IS in the RGB scope,
  `render_cuda.cu:10976`).
- **Therefore fix the de-hero bundle split FIRST, and re-measure before writing an RGB mode W.**
  Once spectral mode W is 1-spp-clean on glass, `-rgb`'s entire remaining advantage is 1.27-1.7x
  — not worth a second hand-written megakernel that has to stay bit-exact with the CPU. The
  likely outcome is that the bundle-split fix makes an RGB mode W unnecessary.
- **Order:** (1) de-hero bundle split [CPU], (2) deterministic glossy-lobe lattice [CPU],
  (3) port spectral mode W to the device, (4) re-measure and only then judge `-rgb`. Steps 1-2
  come first because they change mode W's estimator, and porting an estimator that is about to
  change means writing the device twin twice.
- **Nothing to do on the CPU side:** all four knobs already ship there — `-mode W` (`g_whitted`)
  and `-whitted-grid N` (`g_whittedGrid`) in v0.105.0, `-gi`/`-radiosity` (`g_gi`) and
  `-ambient`/`-amb` (`g_ambient`) in v0.106.0. `bkWhitted`/`bkGrid`/`bkGiGrid`/`bkAmbient` are
  just the proposed device-`DScene` field names (matching the existing `bkDirectOnly`) that would
  carry those existing CPU features across. This is a pure port, not a feature.
</details>

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
    ftrace's loader is OBJ-only (convert first); mesh `to_world` with
    rotation/shear only partly expressible (translate+scale + euler). (Mesh
    area-emitters *do* now have an FTSL equivalent — a `mesh` bound to an `emit`
    material, since 0.41.0 — so the exporter could emit that instead of dropping the
    emission; not yet wired up.)
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
  ever support POV-Ray-style implicit geometry. **DONE** — `isosurface` (`src/implicit.h`,
  FTSL §10), on CPU and GPU, with POV's marching controls adopted name-for-name
  (`contained_by`, `max_gradient`, `accuracy`, `method sample`, `refine`), all **78** of
  POV's `functions.inc` internal functions as bit-exact ports of `fnintern.cpp`
  (`src/pov_functions.h`), and an exact port of POV's Perlin `Noise()` (`src/pov_noise.h`).
- **Does that reopen the importer? NO — re-evaluated 2026-07-30, still declined.** The SDF
  primitive answers only part of objection (2), and objections (1) and (3) are untouched.
  The decisive point: POV's CSG is **ray-interval arithmetic over closed solids** (every
  primitive must report *all* hits along a ray plus an `inside()` predicate), whereas ours
  is **min/max on distance fields**. Those are not interchangeable. Ours cannot take a
  `mesh` (there is no mesh→SDF path in the tree, and adding one is lossy), nor POV's
  `prism` / `lathe` / `sor` / `text` / `bicubic_patch`; POV's `quadric` / `quartic` /
  `poly` are expressible as `function { expr }` but are then *marched* rather than
  root-found. Conversely ours does smooth/filleted booleans, which POV's cannot. So a
  general POV scene using CSG still lands back on tessellation + mesh booleans for
  anything non-SDF — objection (2) verbatim, for that class of scene.
  **Note the reason is value and prerequisites, not effort:** the blockers are that a
  faithful importer needs a Turing-complete interpreter plus POV's `.inc` standard library
  (most real-world scenes `#include` at minimum `colors.inc`), and that POV's RGB /
  non-physical lights and cameras must be re-authored into spectral absolute radiance
  regardless — so the only thing an importer actually delivers is *geometry*, for which
  Mitsuba XML (and hence Blender) already works. What was worth taking from POV-Ray has
  been taken: its function corpus, its isosurface machinery, and its deterministic Whitted
  model (mode `W`, with `-gi` standing in for radiosity).
- **The separable question, if mesh booleans are ever wanted:** general interval CSG over
  the *native* primitives. Scope is real but bounded — `allHits()` alongside `closestHit()`,
  a BVH traversal that collects instead of pruning by `tMax`, fixed-size interval stacks on
  the device, and interaction with the dielectric medium stack. Only `sphere` / `mesh` /
  `isosurface` would qualify (`quad` and `triangle` are infinitely thin, so they are not
  solids and CSG over them is meaningless — POV carries the same caveat). Mesh `inside()`
  by parity counting *is* reliable here because the triangle test is watertight. Still not
  recommended: the payoff is essentially "drill a hole in an imported mesh", which Blender
  does better upstream of the existing import path. The cheap ~80 % alternative would be
  POV-style **`clipped_by`** (clip a primitive by a half-space/convex region and cap the
  opening) — no interval lists needed, and the isosurface container-cap code already has
  most of that machinery.

## DONE (2026-07-22): Mode-M shared photon-map deposit/build hangs on the full gallery scene (4M photons) — was the CPU meter pre-pass; meter now runs on the requested device

- **Symptom:** `renderPhotonMapSharedCuda` on `scenes/gallery_settled.ftsl` (via
  `scraps/gallery_fly.ftsl`) with `-n 4000000 -spp 6 -r 320 180` ran **67 min pegging
  ~6 CPU cores (24,255 CPU-s) with GPU util ~1-6% and never wrote the `-savemap` file**
  (i.e. the one-time deposit+build phase never completed). Working set stayed ~1 GB, so
  it is NOT a huge-photon-set RAM blowup.
- **Contrast:** the identical pipeline with `-device gpu -n 2000000 -spp 4` on the SAME
  scene completed fully (deposit+build+2 gathers+mode-D still) in ~2 min. So 4M is >30x
  slower than 2M — wildly super-linear, pointing at a pathological host-side phase
  (suspect `PhotonMap::build` counting sort or the device->host photon download/convert
  loop in render_cuda.cu ~5680-5698), not simple scaling.
- **Repro:** `ftrace -in scraps/gallery_fly.ftsl -n 4000000 -spp 6 -r 320 180 -window
  -savemap gallery/hero_map.ftpmap -o png/fly/fly.png`
- **Proper fix (TODO):** profile the deposit/build with 2M vs 4M; find why host CPU
  scales super-linearly (likely an O(n^2) or lock-contended path, or grid cellSize
  degenerating so build buckets explode). Until fixed, cap flyby photons at ~2M.
- **UPDATE 2026-07-21 — the "2M is fine" escape hatch does NOT hold at full film
  resolution.** Re-tested on the real 600-frame `fly` curve (960×540, mode M) after
  the Camera-film OOM fix: `-device gpu -n 2000000 -spp 4` (and `-spp 6`) **also
  hangs** — GPU util sits at 0–6 %, working set ~1 GB, and **zero frames** are written
  after 3–8 min (killed). The known-good "2M in ~2 min" measurement above used
  `-r 320 180`; the hang is NOT purely a photon-count effect — it recurs at 2M once
  the gather resolution is full. So the host-side build/download phase is the real
  culprit regardless of photon count, and `-device gpu` is currently unusable for
  this flyby at any practical resolution. The **CPU** shared path works but is slow
  (~16 s/frame at 400k/spp3, 960×540 → ~3–5 h for 600 frames). `render_gallery_flyby.bat`
  therefore forces `-device cpu`. Fixing the GPU host-side build is the unblock that
  would make the flyby render in minutes instead of hours.
- **CORRECTION 2026-07-21 — the "GPU hang" above was a MISDIAGNOSIS; the real cost is
  the mode-M CPU meter pre-pass, not the GPU photon-map build.** `gallery_settled.ftsl`
  uses `exposure_lock` (EXPLOCK_AVERAGE), so `meterAnchor` (main.cpp ~5484) runs a
  CPU-only metering phase BEFORE any GPU render, **regardless of `-device`**: for mode M
  it builds a ~4M-photon **CPU** photon map once (`meterN = clamp(W·H·40, 500k, 4M)` = 4M
  at 960×540) and then gathers up to `kMeterMax = 64` frames at `meterSpp = 16` on the
  CPU. That front-loaded phase is 10,000+ CPU-seconds (~28+ min) during which the GPU
  correctly sits at ~1–6 % and **zero frames are written** (meter frames aren't saved) —
  exactly the "hang" symptom I attributed to `renderPhotonMapSharedCuda`. Killing at
  3–8 min just killed it mid-meter, before the GPU render ever started. So: the GPU
  shared build is NOT known to hang at 2M/960×540; it was never reached. **The real
  proper fix is to make the mode-M meter cheap** — e.g. drop `meterN`/`meterSpp` for the
  noise-robust p99 anchor, cap `kMeterMax` lower, or GPU-accelerate `meterAnchor` — after
  which `-device gpu` should be re-tested on the full-res flyby before concluding anything
  about the GPU build path. (The `-r 320 180` "2M in ~2 min" run was fast partly because
  its meter was also tiny: `meterN` scales with `W·H`.)
- **RESOLVED 2026-07-22 — the meter pre-pass now runs on the requested device
  (main.cpp `meterGpu`), and the GPU shared build is confirmed healthy at 4M/960×540.**
  Two changes in the exposure-lock meter (`run()`, main.cpp ~5654):
  1. **Per-frame metering follows `-device`.** `meterAnchor` now dispatches each mode's
     meter render through the same GPU entry point (and the same support predicate) its
     real render uses — A/B/C via `renderForward(useGpu)`, R via `renderBackwardCuda`,
     D via `renderBdptCuda`, P both layers — falling back to the CPU renderer whenever
     the predicate says no. `-device cpu` is **bit-identical** to before (verified:
     old-vs-new exe print the same anchor to 4 sig figs AND all rendered frames sha1-match
     on the gallery m8 scratch scene at 192×108).
  2. **Mode-M groups meter in ONE batched GPU pass.** An all-M pinhole group (the flyby
     case) now meters via `renderPhotonMapSharedCuda` — one device photon map + GPU
     gathers for up to `kMeterMax` meter frames, early-stopped by the same `MeterConverge`
     test through the shared path's `onFrame` hook — instead of one CPU map + up to 64
     full-res CPU gathers. Gated exactly like `runSharedPhotonMap`'s GPU branch; any
     gate miss falls through to the per-frame CPU loop unchanged.
  **End-to-end verification** (8-frame `fly` scratch copy of `gallery_settled.ftsl`,
  960×540, `-device gpu -n 4000000 -spp 8`): meter = deposit + 8 GPU gathers in ~2–3 min
  (was 10,000+ CPU-s ≈ 28+ min), anchor 2.981e-09 (CPU meter cross-check at 192×108:
  2.967e-09, agreement ~0.007 stops); then the REAL `renderPhotonMapSharedCuda` phase —
  the part the correction above said "was never reached" — ran the **4M-photon device
  deposit + build + per-camera gathers and streamed frames to disk** with the locked
  exposure applied (auto-exposure=2.98e-09 on every frame). So the GPU shared build
  never had a hang at all; the whole symptom was the CPU meter. Remaining perf note:
  each full-res 960×540×16spp meter gather is ~15 s on the 4090 — the gather kernel is
  a hot-path optimization candidate (tracked in the 2026-07 D/M GPU optimization work).

## Mode-M shared render ignores `-window` — no live preview opens (2026-07-21)
- **Symptom:** rendering the gallery `fly` curve in mode M with `-window` never opens
  a Win32 live-preview window. Verified via `Get-Process ftrace` → `MainWindowHandle=0`
  and empty `MainWindowTitle` for the entire render, on both the GPU and CPU paths,
  even while the CPU path was actively writing frames to disk. (Single-camera forward
  and mode-R/B/C renders DO open the window normally — this is specific to the
  mode-M shared-photon-map multi-camera path, `runSharedPhotonMap` in `src/main.cpp`.)
- **Impact:** violates the project's "always show the live preview so the user can
  watch it converge" rule for exactly the long multi-frame renders where watching
  matters most. The user cannot see a flyby converging; only the on-disk PNGs reveal
  progress.
- **Proper fix (TODO):** wire the shared-photon-map gather loop into the same
  `LiveWindow` lifecycle the single-camera forward path uses — create/show the window
  before the first gather and push each freshly-gathered frame's tone-mapped buffer to
  it (mirroring the per-frame `writeFrame`), instead of only writing the PNG.

## Access-violation popup when closing the live-preview window (intermittent) — PARTIALLY ADDRESSED (2026-07-17)

- **Symptom (user report, 2026-07-17):** closing two `ftrace.exe` live-preview windows
  produced two "access violation" WER popups on exit. The renders were mode-R gyroid
  batches launched with `-window -keepwindow` (studio-env HDR variants).
- **Could NOT reproduce despite faithful attempts.** Tried under **cdb** (break on AV /
  heap-corruption `0xC0000374` / fastfail `0xC0000409`) and, to defeat any debugger
  Heisenbug, under **procdump** (`-e -ma`, full native speed) across: (a) mode-B GPU
  render closed after finishing, (b) mode-R GPU render (`scenes/implicit.ftsl`) closed
  **mid-render**, and an isolated `src/livewindow.cpp` harness (`scraps/livewin_repro.cpp`)
  in three lifecycle modes — plain batch close, destructor-closes-a-still-live-window,
  and a no-sleep `update()`/`setTitle()` "hammer" race closed externally. **Every path
  tore down cleanly (exit 0, no dump).** So the isolated live-window lifecycle is NOT
  the fault on the paths tested.
- **Prime suspect: CUDA/driver async teardown.** The binary imports `nvcuda.dll`; closing
  a `-keepwindow` window is exactly what unblocks the whole shutdown (`main()` hold loop →
  `cudaGracefulShutdown()` → CRT static dtors incl. `g_liveWin`). `main.cpp` (~5568)
  already documents a related async `nvlddmkm` DPC teardown fault they mitigate. A
  userspace AV popup would be a fault inside `nvcuda.dll`/driver teardown, which our code
  can't fully fix — but capturing a real dump would confirm the module.
- **PARTIAL FIX applied (`src/livewindow.cpp`):** hardened a genuine latent cross-thread
  hazard found by inspection — `impl_->hwnd` was never cleared after the window was
  destroyed, yet it is read from the render thread (`setTitle`/`clientSize`/`enablePanel`/
  `setPathCount`) and from `~LiveWindow` (`PostMessageW(WM_CLOSE)`). Windows **recycles
  HWND values**, so a since-reused handle belonging to another window/thread could receive
  our `WM_CLOSE`/`WM_SETTEXT`. Made `hwnd` `std::atomic<HWND>`, null it on `WM_DESTROY`,
  and made every cross-thread caller (and the dtor) `load()` once + null-check so nothing
  marshals to a stale/recycled handle after close. Verified no regression via the isolated
  harness + a real mode-R render. This removes a real defect but is **not confirmed to be
  THE crash** (couldn't reproduce the original AV).
- **Next step if it recurs:** capture a full dump of the actual fault to pin the module —
  e.g. attach `procdump -ma -e` to the live PID, or `procdump -i -ma <dir>` (admin) to
  install a system postmortem catcher — then analyze `.dmp` (`!analyze -v`) to confirm
  whether it is `nvcuda`/driver teardown vs. app code.

## FIXED (2026-07-19): `-n` ignored scientific notation (`-n 2e8` → 2 photons)
`-n <count>` was parsed with `std::atoll`, which stops at the first non-digit — so the
common shorthand `-n 2e8` / `-n 8e7` (used in README examples) silently parsed as just the
leading integer (`2`, `8`), rendering a near-black image from a handful of photons. Fixed
in `run()` (src/main.cpp): tokens containing `e`/`E`/`.` are now parsed as a double and
rounded to the nearest count; plain integers still go through `atoll` exactly. Found while
validating `-stereo`.

## OPEN (minor, 2026-07-26): native `-viewer` Meshes tab ignores a skin's `clamp`/`mirror` wrap mode
The F4 texture display (`SkinLib` in `src/viewer_gui.cpp`) uploads each skin as a plain
D3D11 texture and lets **ImGui's own DX11 backend sampler** do the filtering — and that
sampler is created once, with `D3D11_TEXTURE_ADDRESS_WRAP` on all three axes
(`imgui_impl_dx11.cpp`, `ImGui_ImplDX11_CreateDeviceObjects`). So a sidecar texture
declared `wrap="clamp"` or `wrap="mirror"` still *tiles* in the preview whenever a mesh's
UVs run outside `[0,1]`. Everything else about the skin is faithful (ftrace's own
`Texture::load` decodes images, ftrace's own pattern VM bakes formula skins), so this is
the one place the preview can disagree with a real render. The nearest/bilinear `filter`
mode *is* honoured, but only because the bake resamples through `Texture::sampleRgb` —
the GPU sampler is always linear on top of that, so a `nearest` skin gets a faint
smoothing at extreme magnification.

**Proper fix:** stop relying on the shared backend sampler for skins. Either (a) give
`SkinLib` its own `ID3D11SamplerState` per wrap/filter combination and bind it with an
`ImDrawList` callback around the mesh draw (`AddCallback` → set sampler → restore), or
(b) pre-expand the wrap mode into the *UVs* at bake time (clamp/mirror the per-vertex UVs
on the CPU before `PrimVtx`), which is cheaper but only correct when a triangle doesn't
straddle the tile boundary. (a) is the real answer.

## OPEN (cosmetic, 2026-07-28): `python -m loom.viewer` prints a runpy double-import RuntimeWarning
Starting the F4 live channel spawns `python -X utf8 -u -m loom.viewer <scene.py>`, and
Python's `runpy` prints to stderr:

```
RuntimeWarning: 'loom.viewer' found in sys.modules after import of package 'loom',
but prior to execution of 'loom.viewer'; this may result in unpredictable behaviour
```

Cause: `tools/loom/loom/__init__.py` re-exports the viewer API (`from .viewer import
ViewerModel, serve_viewer, ...`), so importing the *package* `loom` — which `-m
loom.viewer` does first — already puts `loom.viewer` in `sys.modules`; runpy then
executes the same file a second time as `__main__`. It is harmless here (the module has
no import-time side effects and the duplicate module object is never handed out), but
`LoomLink` deliberately gives the child ftrace's own stderr so a scene traceback is
readable, which means the warning lands on the user's console on every viewer launch.

**Proper fix:** give the CLI its own entry module — `tools/loom/loom/__main__.py`-style
`loom/viewer_main.py` (or a `console_scripts`-shaped `loom.viewer.__main__`) that only
does `from .viewer import serve_viewer; serve_viewer(...)`, and switch `LoomLink::start`
and the documented invocation to it. `-m loom.viewer` should keep working (deprecated),
so the ftrace side must not hard-depend on the new name until the docs are updated
together.

## FIXED (2026-08-01): loom's `skip_existing` adopted half-converged frames; `-resume` overshot spp
Two bugs in the same code path, `render_range` in `tools/loom/loom/drive.py`, both found
while re-rendering the 432-frame `jumping_jack` sequence after a crash.

**1. Existence is not completeness.** `skip_existing` tested only `png.is_file() and
png.stat().st_size > 0`. But ftrace writes the PNG at *every* `-interval` tick — that is
the whole point of the crash-safety feature — so an interrupted frame leaves a perfectly
valid PNG at whatever spp it reached. A resumed run then skipped it forever. Observed:
frame 244 of the jack sequence was silently accepted at **3 of 8 spp**, i.e. a visible
noise pop in the middle of a loop that nothing would have flagged.

**Fix:** read the `.ftbuf` checkpoint sidecar and compare its accumulated sample count
against the requested `-spp`. New helpers `checkpoint_spp()` (parses the packed
little-endian `FTBUF01\n` header — `i32 resX, i32 resY, i32 mode, i64 N` — from
`src/main.cpp:3574`) and `_target_spp()` (pulls `-spp` out of `extra_args`). A frame is
done only when `have >= want`; a partial one is *continued*, not restarted, and the
shortfall is announced rather than hidden. When the budget is not spp-based (`-noise`,
`-time`) there is no completeness test available at all, so the fallback to existence is
now logged once at the top of the run instead of silently assumed.

**2. `-spp` is ADDITIVE under `-resume`.** Resuming a 3-spp frame with the original
`-spp 8` renders it to **11**, not 8 — so the naive retry produced frames with more
samples than their neighbours, which is the same flicker in the other direction. Fix:
substitute `str(want - have_now)` into the resumed command line so every frame lands on
exactly `want`.

**And the assumption underneath it was false — see the next entry.** Continuing a partial
frame is only legitimate if `3 spp + resume 5 spp` is *bit-identical* to a plain `8 spp`.
`scraps/resume_check.py` was written to prove that by `filecmp`, and on its first
successful run it said **False**, max channel difference 163/255. That turned out to be a
genuine CUDA bug, not a loom one.

## FIXED (2026-08-01, 0.117.1): GPU `-resume` restarted the sample sequence at 0, so a resumed mode-W frame was the wrong image
Found by `scraps/resume_check.py` while validating the loom fix above: on the **GPU**,
`3 spp` + `-resume -spp 5` was NOT the same image as a plain `8 spp` — max channel
difference **163/255**. On the **CPU** the same ladder passed all three rungs, which
localised it immediately.

`cpuSppChunks` (`src/main.cpp` ~3941) biases each chunk's sample index past whatever the
checkpoint holds:

```cpp
const unsigned long long seedBias = (unsigned long long)prog->sampleBase;
...
Film f = renderOne(c, seedBias + (unsigned long long)done);
```

Its GPU counterpart `gpuSppChunks` (`src/render_cuda.cu` ~11598) did not — it called
`launch(c, done)` with `done` starting at 0 every invocation. So a resumed run re-rendered
absolute sample indices `[0, c)` on top of a film that already contained them. For the
Monte-Carlo modes that was merely disguised (the host deliberately XORs `prog->sampleBase`
into the device seed, so the repeated indices at least drew *different* streams and the
average still converged). For **mode W** it was plainly wrong: mode W is a deterministic
quadrature whose lattice is a pure function of the absolute sample index — `kBackward`
computes `sIdx = sampleBase + local` for exactly that purpose — so `3 + resume 5` averaged
samples `{0,1,2} ∪ {0,1,2,3,4}` instead of `{0..7}`. Duplicated coverage in part of the
lattice, gaps in the rest, and no amount of further resuming would fix it.

A second, subtler defect rode along: the kernel seeds on `gidx = pix * sppTotal +
sampleBase + local`, and `sppTotal` was **this invocation's** requested spp, not the final
total. Once `sampleBase` is honoured, `sampleBase + local` can exceed `sppTotal` and walk
out of its pixel's slot into the next pixel's seed range, correlating two samples that are
supposed to be independent.

**Fix** (`src/render_cuda.cu`): `gpuSppChunks` passes `prog.sampleBase + done`; all three
host drivers (`renderBackwardCuda`, `renderBdptCuda`, `renderBackwardRGBCuda`) compute
`sppBase = prog->sampleBase`, pass `sppTotal = sppBase + spp` as the kernel's stride, and
use `launch(spp, sppBase)` on the single-shot path. In `renderBackwardCuda` the
decorrelating seed XOR is now applied only when **not** in mode W: a deterministic
quadrature must reproduce the monolithic render exactly, so its seed base has to stay put,
while the Monte-Carlo modes keep the mix as insurance against a rare `gidx` collision with
a stream the checkpointed samples already drew.

Verified with `scraps/resume_check.py` (three-rung ladder: plain-vs-plain with adaptive
chunks, plain-vs-plain with `FTRACE_CHUNK_SPP=1`, then `8` vs `3 + resume 5`) — all True on
both CPU and GPU. Regression-checked with `scraps/resume_regress.py`: frames 000/100/200/243
of the loom `jumping_jack` sequence, rendered by the *previous* binary, re-render
**bit-identically** under the new one, confirming that a non-resuming render is untouched
(`sampleBase == 0` ⇒ `sppTotal == spp` and the seed is unchanged).

**Known remaining wart (by design, documented rather than fixed):** because `gidx` carries
`sppTotal`, a render resumed *twice* to different totals seeds its Monte-Carlo streams
differently than a monolithic render of the final total would. Mode W is unaffected (its
lattice never consults `gidx`), and for the stochastic modes distinct-but-valid streams are
all that is required, so this only means "a twice-resumed mode R/D image is a different
realization", not a wrong one. Removing it entirely would mean switching the device to the
CPU's sample-major index `(sampleBase + local) * npix + pix`, which is total-independent
*and* would move mode R/D closer to host/device bit-exactness (todo N4a) — but it changes
the noise realization of every existing GPU render, so it belongs with N4a, not here.

## DONE (2026-07-28, 0.92.0): the DAG panel crashed in `PrimReserve` — imgui #7543 vs. imnodes node rects

The viewer's Graph pane died intermittently with an access violation writing to `0x20`,
always on the same stack: `drawDagPanel` → `ImNodes::EndNodeEditor` → `DrawLink` →
`ImDrawList::AddBezierCubic` → `PrimReserve` → `memcpy`. Working set at the fault was
~9.9 GB.

**Root cause** — an upstream ImGui behaviour change that imnodes was never updated for.
Since 1.90.7 (imgui #7543) `EndGroup()` folds `g.LastItemData.Rect.Max` into the group's
bounding box as a workaround for `EndTable()` undershooting `CursorMaxPos`:

```cpp
ImRect group_bb(group_data.BackupCursorPos,
                ImMax(ImMax(window->DC.CursorMaxPos, g.LastItemData.Rect.Max),
                      group_data.BackupCursorPos));
```

`BeginGroup()` backs up and resets `CursorMaxPos`, but it never clears `LastItemData`.
For normal stacked layout that is harmless — the previous item is above and to the left.
imnodes is the pathological case: it hard-positions every node anywhere on the canvas via
`SetCursorPos`, so "the previous item" is *the previously drawn node*, and each node's
reported rect became the running maximum of every earlier node's right/bottom edge.
ftrace then feeds `GetNodeDimensions()` back into `measureDag` to lay the graph out, so
the extent grew geometrically frame over frame (748 → 5443 → 110498 → …) until a link was
~1e8 px long. `GetCubicBezier` scales `NumSegments` with link length at 0.1/px and never
bounds it, so that one link asked for ~80M segments → ~322M `ImDrawVert` (~6.9 GB);
`PrimReserve`'s allocation returned null and the following `memcpy` wrote through it.

**Fix** (three parts, all in `src/third_party/imnodes/imnodes.cpp`, marked
`[ftrace patch]`): a `ResetLastItemForGroup()` helper called immediately before every
`ImGui::BeginGroup()` in imnodes (`BeginNodeEditor`, `BeginNode`, `BeginNodeTitleBar`,
`BeginPinAttribute`, `BeginStaticAttribute`); an explicit `LastItemData.Rect` override in
`EndNodeTitleBar`, because `GetNodeTitleRect()` is as wide as *last frame's* node and
leaving it as the last item would latch a node to its historical maximum width so it could
never shrink again (e.g. on zoom-out); and a `ImClamp(..., 1, 4096)` on
`GetCubicBezier`'s segment count as defence in depth, so a single stray coordinate can
never again turn one link into a multi-gigabyte vertex request.

Verified: working set 9882 MB → 256 MB, extent stable at 748×198.2, and a 20-round
scripted right-drag sweep (`scraps/viewer_hammer.ps1`) ran to completion with memory
oscillating 315–821 MB and no monotone growth.

## DONE (2026-07-28, 0.92.0): the DAG panel re-packed itself when its pane was scrolled out of view

Second bug, found while validating the fix above. `drawDagPanel` adopts imnodes'
`GetNodeDimensions()` as the authoritative node size, but the Graph pane sits at the
bottom of a scrolling side column and is routinely clipped to **zero height**. ImGui then
sets `SkipItems` on the canvas window and every `ImGui::Text` inside a node returns without
measuring anything — yet imnodes still reports a rect, namely the node origin expanded by
`NodePadding`. Adopting that re-packed the whole graph at ~16×32 px per node, so the layout
was visibly wrong the moment the user scrolled the pane back into view.

**Fix:** reject the entire frame's measurements unless *every* node's reported content
width exceeds `2 * NodePadding.x * zoom`. A node always draws at least its title, so a
content width of zero means "not measured this frame", never "an empty node". The previous
frame's sizes are kept until a frame that actually drew comes along.

## FIXED (2026-08-03, 0.119.0): mode `W` renders a scene to pure black — with no diagnostic — when every light is enclosed in refractive geometry

Reported from `ftrace -explore -whitted-grid 1 -mode W -in scenes\gallery_settled.ftsl`:
the raster stage navigates fine, but the instant the camera settles and the mode-`W` lit
preview takes over, the window goes black and stays black. Reproduces headlessly:

```
ftrace -in scenes/gallery_settled.ftsl -mode W -spp 1 -whitted-grid 1 -camera cam \
       -r 320 200 -o png/gal_w.png            # -> all-zero image, "auto-exposure=1"
```

**Not a viewer bug.** `gallery_settled`'s entire illumination is one 8 mm arc,

```
light sphere { center 5.0 3.10 3.0   radius 0.008   spd blackbody 6000 }
```

sitting at the exact centre of two nested dielectric meshes (`lamp_bulb`, fused silica,
scale 0.11; `lamp_xe`, xenon gas, scale 0.1034) plus the chrome electrodes. Mode `W` does
*all* of its lighting by NEE — a shadow ray from each vertex to a point on the light — and
an occlusion test treats a dielectric as an opaque blocker, so **every** direct connection
in the scene is blocked. Mode `W` also drops stochastic diffuse indirect (it implies
`-direct-only`), so no energy reaches anything by any route, and the result is exactly
zero everywhere. `-gi` does not help: gather rays collect mode-`W` radiance, which is
itself zero. This is the classic Whitted limitation (POV-Ray behaves identically) and is
exactly why the scene's own `prefer{}` block selects mode **D** — BDPT starts paths at the
arc and refracts them *out* through the quartz.

Confirmed by deleting only the two dielectric shells (`scraps/gal_noglass.ftsl`):
auto-exposure goes `1` → `1.32e-09` and the hall lights up.

**Workaround:** `-ambient 0.2` gives a fully readable flat-lit preview
(`-no-media` on top, since mode `W` ignores the bounded cloud anyway and treats the global
haze as a single homogeneous term):

```
ftrace -explore -mode W -whitted-grid 1 -ambient 0.2 -no-media -in scenes\gallery_settled.ftsl
```

**Fixed** by detecting the *condition* rather than the symptom, so the diagnostic names the
cause instead of reporting a black frame after the fact. `Scene::emitterSeal()`
(`src/scene.h`) probes each emitter with a deterministic lattice of 512 outgoing directions
— stratified over the emitter surface via the existing `Emitter::samplePoint`, uniform over
the outgoing hemisphere (or inside the cone, for a spot) — and reports the fraction whose
first hit is a material for which `isSpecularType()` holds. That predicate is exactly right
rather than approximately right: `backward.h` calls `neeLight()` from the `Diffuse`,
`DiffuseTransmit` and `Fluorescent` cases and *nowhere else*, so a first hit on any
specular type (glossy included) is a direction whose power NEE can never collect. Hits on
the emitter's own surface yield no evidence and are excluded, so a concave mesh light
seeing itself is not mistaken for a sealed one. `warnSealedLights()` (`src/main.cpp`) runs
it once at startup whenever `g_whitted || wPreview`, so `-explore` is covered too (the
viewer's `T` preview *is* mode `W`).

The threshold is **0.95**, not 1.0, and this is the part that needed measuring rather than
guessing: the gallery's arc probes at **98.2 %**, not 100 %, because the lamp assembly has
its own socket and cord *inside* the envelope — diffuse surfaces that are genuinely lit but
illuminate nothing except themselves. A first pass thresholded at 0.995 and silently missed
the very scene it was written for. Past ~95 % the scene is at least 20× underlit against
what the author intended, so the preview is misleading whether or not it is literally zero.

Verified against all 98 scenes in `scenes/`: every one reaches the mode-`W` path, and
exactly three trip the warning — `gallery.ftsl`, `gallery_settled.ftsl` and
`mirror_sphere_interior.ftsl`, all the same sealed-lamp assembly. No false positives. The
third is an independent confirmation rather than a third instance of one mistake: its own
header comment already states that you cannot "next-event-connect a shading point to a
light through a refracting interface", and it renders at `auto-exposure=7.18e-14` — a
near-black frame with a few specular specks, exactly what the warning predicts.

Still open as a possible extension: mode `R`/`P` hit the same condition, where it shows up
as pathological convergence rather than a black frame, and would benefit from the same
warning. The probe is mode-agnostic; only the call site is gated.

## OPEN (minor, 2026-08-03): rendering a scene without `-camera` also renders its 600-frame flyby

`ftrace -in scenes/gallery_settled.ftsl -mode W -o png/gal_w.png` renders the still camera
*and* then all 600 frames of the scene's `camera_curve "fly"`, writing `png/gal_w_fly000
…599.png` (~4 min). Selecting a camera with `-camera cam` avoids it. Arguably working as
designed — "render every camera in the scene" — but it is a surprising default for a scene
that carries a long flypath, and it silently spams the output directory next to the `-o`
path. Worth at least a printed warning naming how many frames are about to be written.

**Now affects `gallery_rain` too** (2026-08-04): it gained its own `camera_curve "fly"`, so
`-parseonly` on it reports **601 cameras** and a bare render of the scene writes 600 frames
into whatever directory `-o` points at. Use `-camera cam` for the still; `-camera near=X,Y,Z`
is the convenient way to pull the single flyby frame closest to a point (it prints which
`flyNNN` it picked and how far off it was), which is how the four fly-through passes were
each validated without rendering the loop.

## OPEN (tech debt, 2026-08-04): `design.md`'s measurement rigs live in git-ignored `scraps/`

`design.md` cites `scraps/_gemsweep.py`, `scraps/_capchroma.py`, `scraps/_capcrop.py` and
`scraps/_pfm.py` as the authority for decisions that are *shipped* in `scenes/gallery_rain.ftsl`
— which glass the axicon is cut from, what drop it hangs at, how big its cap is, and (2026-08-04)
that it gets a 0.04 m girdle and no crown. Those files are **untracked**: `.gitignore` has a
blanket `/scraps/`, on the correct general principle that scraps is for throwaway scripts.

The result is that every measured table in `design.md` is unreproducible from a clean clone.
Losing the directory would not break a build, but it would strand the numbers: nobody could
re-derive them, and nobody could tell whether a later scene edit had moved them.

Proper fix: promote the four load-bearing ones to `tools/` (which is for permanent checked-in
tooling) and update the ~40 `scraps/_*.py` path references in `design.md` and the scene comments
in the same commit. They have real interdependencies (`_gemsweep` and `_capchroma` both import
`_pfm`, `_capcrop` imports `_capchroma`), so it is a move of the whole cluster or none of it.
Logged rather than done because renaming the paths that every measured table cites is a change
that wants to be its own commit, not a rider on a scene tweak.

## OPEN (metrology, 2026-08-04): `-fireflies 3` does not always clear the gem rig's peak, so `peak` alone can be nonsense

While sweeping crown angles for the axicon's girdle (`scraps/_gemsweep.py piece gcone0.08/20/0.60
0.65`, SF10, 480 px / 600 spp, `-hdr -fireflies 3`) the rig printed:

```
gcone0.08/20/0.60 drop 0.65   peak 19945.07x   coverage  0.02%   sat 0.254   spread 0.041
```

`peak` is four orders of magnitude above every neighbouring row while `coverage` says there is
essentially no caustic at all — i.e. a single monochromatic hero-wavelength spike survived the
firefly filter. That filter clamps an isolated outlier to 3× its second-brightest *neighbour*,
so it is defeated whenever a spike lands next to another bright pixel (two adjacent caustic-path
hits, or a spike on the edge of a specular highlight). It is a **rig/metric** problem, not a
renderer bug — the shipped scene never meters `peak` in isolation.

Impact today: none of the decisions in `design.md` rest on `peak` alone (`coverage`, `sat`,
`spread` and `fan` all agreed that every crown geometry kills the caustic), and the row above is
published with its `peak` struck out. But `peak` is reported as a headline number in
`_gemsweep.measure()` and will mislead the next reader.

Proper fix: report a robust peak instead of the max — e.g. the 99.9th percentile of cell
luminance, or the brightest *4× box cell* rather than the brightest pixel, both of which are
immune to a single spike by construction. `measure()` already builds the downsampled cell list,
so this is a two-line change; the reason it is logged rather than done is that changing the
statistic silently invalidates every `peak` in `design.md`, so the fix has to re-run the tables
in the same change.

## OPEN (2026-08-04): a quad's emission is invisible from the side its winding faces away from

The backward tracer only adds a surface's own emission when the camera/specular ray strikes
the FRONT face:

```cpp
// src/render_cuda.cu ~7006 (host twin: src/backward.h)
if (specularArrival && dot(rd, h.ng) < 0) { ... L += thr * specLookup(eSpd, lambda) ... }
```

A `quad`'s geometric normal is `cross(u, v)`, so

```ftsl
quad { origin -18 0 -19   u 46 0 0   v 0 0 45   material glowing }   # ng = -y, dark from above
quad { origin -18 0 -19   u 0 0 45   v 46 0 0   material glowing }   # ng = +y, glows
```

are visually identical for every *reflective* slot — `bkNeeLight`/`bkNeeEnv` flip the normal
toward the incoming light themselves, so a diffuse body shades the same either way — but only
the second one shows its `emit`. The result is a silent, total loss of the emissive component
with no diagnostic; `scenes/gallery_rain.ftsl`'s wireframe ground rendered pure black for this
reason (the stone floor it replaced had no `emit`, so the bad winding had gone unnoticed).

Repro: `scraps/_grid_test.ftsl` (flip `u`/`v` on the four tiles).

Proper fix: emission on a single-sided primitive should be authorable. Either (a) treat `quad`
emission as two-sided by default — test `dot(rd, ng) != 0` and take `|dot|` — since a quad has
no interior for the "back face is inside the solid" argument to protect, or (b) keep it
one-sided but add a `double_sided` / `emit_backface` material flag, and (c) either way, warn at
load time when a scene contains an emissive `quad` whose normal faces away from every camera.
Option (a) matches what people actually author (a glowing panel), and the one-sided rule is
only really load-bearing for closed `isosurface`/mesh solids, whose normals are already
outward.

## OPEN (2026-08-04): `type glossy` renders black in the backward modes (R / W) — no NEE

`bkInteract`'s `D_GLOSSY` case (`src/render_cuda.cu` ~6733, host twin `src/backward.h`)
reflects the ray into a lobe around the mirror direction and returns. It never calls
`bkNeeLight` / `bkNeeEnv`. So a Glossy surface in modes R and W contributes *only* whatever its
mirror ray happens to hit — there is no direct-lighting term at all. Every other reflective
type either NEEs (Diffuse, DiffuseTransmit, Fluorescent) or is genuinely specular (Mirror,
Dielectric, ThinFilm), so Glossy is the one type whose backward appearance does not match its
forward appearance.

Consequence: a rough surface under a small light and a dark surround previews as pure black,
even at high `-spp`, while modes A/B/C/D/M shade it normally. Verified with
`scraps/_grid_test.ftsl`: two tiles differing *only* in `type diffuse` vs `type glossy` render
as dark grey vs `(0,0,0)`. This cost real debugging time on `gallery_rain`, whose ground had to
be demoted to `type diffuse` to be visible in any preview.

Proper fix: give `D_GLOSSY` a direct-lighting term like the other non-specular types — NEE the
light with the Cook-Torrance/Phong lobe's BRDF value and MIS it (balance heuristic) against the
existing lobe-sampled continuation, which already carries `contBsdfPdf`. That is the same
structure `D_DIFFUSE` uses, just with a non-constant BRDF, and it fixes both the black preview
and the (currently very high) variance of a glossy surface in mode R.

## OPEN (2026-08-04): `phase rainbow` — the 2048-bin uniform-in-mu table under-resolves large droplets, and monodisperse supernumeraries read as a white arc

Two related problems with `src/rainbow.h`, both visible in `scenes/gallery_rain.ftsl` and both
reproduced independently in `scraps/bowplot.py` (a Python re-implementation of the same
construction, using `scipy.special.airy` for an exact Ai).

**1. Table resolution.** The phase function is tabulated as 95 wavelengths x 2048 bins
*uniform in mu = cos(theta)*. Near the primary bow (theta ~ 138 deg, sin theta ~ 0.66) one bin
spans `dmu / sin(theta)` ~ 0.085 deg. The principal Airy lobe's half-width is `2.338 / K` with
`K = (2/h)^(1/3) * (2*pi*a/lambda)^(2/3)`, which shrinks as `a^(2/3)`:

| droplet radius | lobe half-width | bins per lobe |
|---|---|---|
| 300 um | ~0.42 deg | ~8 |
| 500 um | ~0.30 deg | ~5 |
| 1000 um | ~0.19 deg | ~3 |

At 3 bins per lobe the bow's peak amplitude is a sampling accident of where the bins land, and
it varies per wavelength (each lambda has its own `theta_rb`), so the *colour* of the arc
aliases too. Proper fix: tabulate uniform in theta rather than in mu (the bow is a
theta-domain feature and mu wastes almost all its resolution near the poles), or keep mu but
add a locally-refined region around each bow's `theta_rb`.

**2. Monodisperse supernumeraries.** With `supernumerary on` the Airy train `Ai(z)^2` rings for
~11 deg inside the primary, while the *coloured* part of the bow — the spread of `theta_rb`
between 450 nm (138.76 deg) and 650 nm (137.65 deg) — is only ~1.2 deg wide. So the eye sees a
broad achromatic ringing band with a thin coloured fringe on each side: "a white curve with
thin bands of colour on top and bottom", which is exactly what the scene was reporting. Real
rain is polydisperse and the supernumeraries average away; only the first lobe survives.
`supernumerary off` already does the right thing, but it is opt-*out*, so the default look is
the wrong one. Proper fix: make the phase function take a size *distribution* (a
gamma/Marshall-Palmer with a width parameter) and integrate over it when building the table,
so `supernumerary on` means "narrow distribution, supernumeraries survive" rather than
"physically-impossible single droplet size".

## FIXED (2026-08-04, 0.129.0): GPU mode D dropped a *material's own* emission — every non-mesh glowing surface rendered black, while the CPU rendered it

BDPT's s=0 strategy ("the eye path lands on an emitter") asks two questions of the last eye
vertex: is it a light, and what does it emit. The device answered both by looking up a
registered `Emitter`:

```cpp
// src/render_cuda.cu — before
__device__ static inline bool dIsLightVertex(const DVertex& v) {
    return v.type == BV_LIGHT || (v.type == BV_SURFACE && v.lightIdx >= 0);
}
__device__ static double dVertexLe(...) {
    if (v.lightIdx < 0) return 0.0;
    return specLookup(sc.emitters[v.lightIdx].emitSpd, lambda) * invPdfLambda * v.emitPatW;
}
```

But **only tessellated geometry registers an Emitter.** A `quad`, an `isosurface` or a CSG
solid is marched, never tessellated, so it has an emissive *material* (`Material::isLight`,
`matIsLight` on the device) and `lightIdx == -1`. Both tests therefore failed and the surface
contributed exactly zero emission on the GPU.

The CPU has always been right: `bdpt.h`'s `Vertex::Le` tests `mat->isLight` and reads
`mat->emit(lambda)` directly, and `Vertex::isLightVertex` does the same. So mode D produced
*two different images* depending on `-device`, with no warning — the exact class of divergence
that is hardest to notice, because the GPU image is not obviously broken, just missing a
material's glow. (The backward tracer never had the bug: `bkRadiance` already fell back to
`mp->matEmit` when `dEmitterForMat` returned -1, which is why modes R/W showed the emission
that mode D did not.)

Found via `scenes/gallery_rain.ftsl`, whose green wireframe ground and stand cages are emissive
`quad`/`isosurface` geometry: mode W and CPU mode D rendered the grid, GPU mode D rendered a
featureless grey plane.

Fix: `dIsLightVertex` now also takes the `DScene` and accepts `sc.mats[v.matId].matIsLight`;
`dVertexLe` falls back to `sc.mats[v.matId].matEmit` when there is no registered emitter,
keeping the emitter's baked SPD first (it may carry a `power`/`lumens` normalisation the raw
material spectrum does not) — the same precedence `bkRadiance` uses. Bit-identical for every
scene whose emissive surfaces are meshes, since those still take the `lightIdx >= 0` branch.

The MIS densities needed no change: `dVertexPdfLightF` already routes a `BV_SURFACE` vertex
down the cosine-Lambertian branch, and `dVertexPdfLightOriginF` returns 0 for `lightIdx < 0`,
which `dMisWeight`'s remap-0 turns into 1 — the same 0 the CPU's `vertexPdfLightOrigin`
returns for a `Vertex` with a null `light`, so the two agree.

Repro (pre-fix): `ftrace -in scraps/_grid_test.ftsl -camera cam -mode D -time 20` — grid absent
on GPU, present with `-device cpu`.

---

### TECH DEBT — DONE (2026-08-04, v0.130.0): a mesh fog bound had a hard silhouette, so imported clouds read as cut-outs

`medium { bounds { object "<mesh>" } }` bakes a **binary** occupancy lattice, and the
`VdbGrid` sampler's trilinear filter ramps 0 -> 1 across exactly **one voxel**. On
`scenes/gallery_rain.ftsl`'s raincloud that voxel is 14 mm across a 3.4 m cloud, so the bound
was a hard edge in every practical sense: the fog stopped dead on the mesh surface and the
cloud rendered with the edge quality of a sticker. That is right for a body or a bottle and
wrong for every volumetric thing anyone actually imports a mesh bound for — cloud, smoke,
dust — whose real edge is a *zone* where concentration falls off over metres.

There was no way to soften it. `density` multiplies on top of membership, but it is evaluated
from world position and knows nothing about where the surface is, so it cannot taper toward an
arbitrary imported silhouette; the only lever was to hand-fit an algebraic falloff to the mesh,
which defeats the point of using a mesh.

Fixed by `feather <metres>` on a mesh bound (`meshvox::featherGrid`), which replaces the 0/1
occupancy with `smoothstep(distance-to-the-outside / feather)`.

* The distance is an **exact** Euclidean distance transform — Felzenszwalb & Huttenlocher
  2012's separable lower-envelope-of-parabolas algorithm, one O(n) sweep per axis
  (`meshvox::edt1d`). A chamfer/Manhattan approximation was rejected on purpose: its error is
  anisotropic, so it would print the lattice's own axes onto the falloff, which is precisely
  the artefact being removed.
* Two numerical traps, both handled. The "unreached" seed must be a **large finite** value
  (`nx^2 + ny^2 + nz^2 + 1`), not `1e300`: F&H intersects two parabolas by *subtracting* their
  seeds, and `1e300 - 1e300` is pure cancellation noise. And the ramp is a smoothstep, not
  linear — a linear ramp creases visibly where it reaches 1, and that crease reads as a second,
  softer silhouette just inside the first.
* Majorant-safe for free: feathering only ever *lowers* density, so `maxVal` stays 1 and delta
  tracking is unaffected. It does thin the object overall; raise `sigma_t` if the core should
  hold its previous opacity.
* `ftsl.h` takes the value in world metres, divides by the voxel edge, warns below one voxel
  (the trilinear filter already gives you that), and **rejects the key on a sphere or
  isosurface bound**, which are carved analytically and have no lattice to soften.

Repro of the old behaviour: drop `feather` from the raincloud in `scenes/gallery_rain.ftsl`.

---

### OPEN: `gallery_rain`'s mode-B fallback branch is a much worse image than its comment admits

`scenes/gallery_rain.ftsl` wraps its still camera in `prefer { mode D } else { mode B }`, so a
machine without a working CUDA BDPT falls back to the forward pinhole splat. That fallback was
authored for the ROOFED edition and has not been re-validated since the walls and ceiling came
off. It should be.

Measured (v0.130.0, `-mode B -time 150` vs the 1180-spp mode-D still): the frame mean drops
47.3 -> 25.2 and the diamond stand's cap 209 -> 152, with the gold gyroid, the glass sphere and
the chrome ring rendering solid black and heavy photon noise everywhere else. The scene header
calls this "a real downgrade... in mode B every piece of glass and the chrome ring go black",
which is true as far as it goes but understates how far from usable the result now is.

Two compounding causes, both created by removing the room:
* **57% of every photon escapes to infinity** (`escaped=0.5662`) instead of bouncing off a wall
  and getting another chance at a camera-visible surface.
* **The emission disc grew with the scene.** A `light sun` is an infinite directional source, so
  a forward tracer has to emit it over a disc covering the scene's cross-section. Opening out to
  a ~33 m ground plane took R_scene from ~12 m to ~33 m, so only ~(5/33)^2 ~ 2% of photons now
  land anywhere near the exhibits. The scene header already works this out for *light selection*
  (it is why the xenon lamp was removed) without noticing it applies to the forward modes too.

**NOT evidence, despite looking like it:** the `[energy]` line reports `sensor=0.0000` for this
scene, but that is normal and says nothing — `scenes/_fog_cornell.ftsl`, a sealed box that mode
B renders perfectly, reports `sensor=0.0000` too. A pinhole subtends ~zero solid angle, so the
splat is an importance-weighted estimator and literal sensor energy is always ~0. (This entry
originally cited that figure as proof the branch was dead; it was not, and the corrected
measurements above are the frame mean and cap level.)

Options, none yet chosen:
* point the else-branch at **mode M** (photon-map camera) or CPU mode D instead of B;
* or drop the `prefer{}/else{}` entirely and let the scene simply require D, since 0.127.0 made
  the GPU BDPT handle the delta sun and the else-branch's original reason (mode D refusing a
  `light sun`) no longer exists.

Repro: `ftrace scenes/gallery_rain.ftsl -mode B -time 150 -o png/_b.png`

### DONE: `gallery_rain`'s caustic screens cannot show a caustic — they are metered into the clip

*(Resolved 2026-08-04 over several revisions, each of which retracted part of the one before —
kept in full because the retractions are the useful part. Short version: the screens were
metered into the clip AND the measurements were taken through that same clip; the fix is
`capwhite` 0.15, a tenth exhibit that actually disperses, `-hdr` for the metering and
`-fireflies 3` to replace the outlier rejection the clamp had been doing by accident.)*

Three revisions of that scene have widened the stand caps specifically so the dispersive
caustics from the glass sphere and the diamond gyroid have somewhere to land. They still do not
read, and the cap width was never the binding constraint. Three separate causes, in order of
size:

1. **The screen is clipped.** `exposure 0.25` was chosen to put a sunlit 0.88-albedo cap just
   under white — correct metering for the brightest diffuse thing in a sunlit frame, and
   exactly wrong for a caustic screen, which has to be DARKER than the caustic. Measured on the
   1180-spp mode-D frame: the diamond's cap averages 209/255 with **24.7% of its pixels at full
   white**; the glass sphere's cap 189/255 with 10% clipped. A caustic adds light to that and
   has nowhere to go.
2. **The sphere is sitting on its own focal length.** The glass sphere is R=0.5 resting directly
   on the cap. For n~1.5 a sphere focuses at nR/(2(n-1)) = 0.75 m from its centre, i.e. 0.25 m
   *below* the cap surface, so what reaches the cap is the cone cut short — a defocused disc
   ~0.34 m across at maybe 5x the direct sun, not a caustic. And with the sun 45 deg off
   vertical the cone axis walks 0.49 m in -z over the 0.5 m drop, centring the disc at z~3.01
   against a cap whose front edge is z=2.95, so roughly a third of it falls off the front.
3. **BDPT is the slow way to get one.** It cannot aim a light subpath; it has to randomly hit
   the sphere, randomly refract and randomly land usefully. Verified with cap albedo dropped to
   0.30 (unclipped, 129/255): at 413 spp there is a faintly brighter region in about the right
   place, buried in chromatic fireflies.

**Addressed (2026-08-04)** in the scene, with one part of the recipe RETRACTED:

- (1) `capwhite` albedo 0.88 -> **0.30**. This is the change that matters; the rest only make
  the caustic bright once there is headroom for "bright" to mean anything.
- (2) The orb is **levitated 0.30 m** on three `wirecage` pins (`stand_glass_mount`) to
  `center 6.7 1.70 3.5`, an 0.80 m drop to the cap. The height was **measured, not computed**:
  the textbook ball-lens `f = nR/(2(n-1))` = 0.734 m is *paraxial*, and a full-aperture sphere
  has gross spherical aberration, so the marginal rays (which carry most of the flux, area
  going as r^2) cross far nearer the glass and the paraxial focus is the wrong target.
  `scraps/_focalsweep.py` renders the orb at a range of heights and meters peak linear
  irradiance on the cap:

  | drop | peak (linear) | |
  |---|---|---|
  | 0.50 | 0.371 | orb resting tangent on the cap — the original scene |
  | 0.65 | 0.371 | |
  | **0.80** | **0.642** | best; 73% brighter than either neighbour 0.15 m away |
  | 0.95 | 0.424 | |
  | 1.15 | 0.363 | |

  So the original scene was at 0.58x of the achievable peak — badly metered *and* badly placed.
  The trade-off at 0.80: the sight line to the disc centre passes 0.514 m from the orb centre,
  i.e. it clears the 0.5 m limb only barely, so the disc reads as breaking out from behind the
  glass rather than sitting cleanly beside it. Deliberate, for the 1.5x brightness.
- (3) The glass cap is 1.7 deep and **cantilevered 0.45 m in -z** off its column, which is
  where the disc actually lands: (6.87, 2.72), from +0.2079 x / -0.9781 z per metre of drop.
  Its stand also moved x 6.2 -> 6.7 to make room (and to fix an unrelated intersection, below).

Result at 1400 spp, mode D, profiled along the sun azimuth on the glass cap (sRGB): shadow
floor 84-100, sunlit cap 145-151, **caustic peak 176.3** — i.e. in linear light the caustic
adds ~1.6x the direct sun's own contribution. It is a real caustic, not a bright patch.

**The GYROID half of this entry stayed broken for a further revision, for a different reason,
and the fix is a topology change (2026-08-04).** The orb above is now a solved problem; the
"diamond gyroid" named in the title was not, and widening its cap again would not have helped
either. It was a gyroid **shell**, `|G| < 0.55` — which is not a pack of prisms (the theory the
scene asserted) but a labyrinth of thin *curved sheets*. A ray crosses a dozen of them and is
deviated a dozen small random ways, so the piece is a **diffuser**: what lands on the cap is a
shadow with a filigree of sub-centimetre threads, too thin to survive even a 4x box downsample.
`scraps/_gemsweep.py` floats one piece at a time over a bare cap in the scene's own sun, box-
averages 4x in linear light (mode D is one hero wavelength per sample, so raw per-pixel colour
is speckle — see the entry below), and scores coverage above 1.2x the bare level, excess-
weighted saturation, and peak:

| piece | coverage | sat | peak |
|---|---|---|---|
| crystal **orb**, drop 0.80 | **0.72%** | **0.246** | 5.99x |
| **solid** gyroid k=6, drop 0.90 | 0.39% | 0.214 | 6.12x |
| **solid** gyroid k=10, drop 0.90 | 0.37% | 0.208 | 5.64x |
| **solid** gyroid k=8, drop 0.90 | 0.31% | 0.188 | 5.60x |
| shell gyroid k=6, drop 0.90 | 0.56% | 0.172 | 5.74x |
| shell gyroid k=13, drop 0.90 | 0.24% | 0.173 | 5.76x (what was in the scene) |
| **Klein bottle**, standing | 0.12% | 0.173 | 2.01x |

Above 1.5x the bare cap, everything except the orb and the solid gyroids goes to **zero**.
Three conclusions, two of which reverse what the scene used to claim:

1. **The fix is topology, not frequency.** Dropping the `abs` takes the field to `G < 0`, one of
   the two interpenetrating **solid networks** — chunky glass with one entry and one exit, i.e.
   an actual optical body. Worth 1.5x the area and 1.2x the saturation at the same pitch. The
   scene now uses solid k=10 at 0.90 m of drop (metered: 0.50 -> 0.26%, 0.70 -> 0.28%,
   **0.90 -> 0.37%**, 1.40 -> 0.11%), on the orb's three pins, over a cap widened to 1.7 x 1.8
   and cantilevered to (3.687, 2.719) where the core lands.
2. **The plain sphere still wins, by 2x** — the exact opposite of the "the gyroid is where the
   colour is" claim the scene carried. A ball lens has one refracting surface pair and all its
   flux lands in one place.
3. **The Klein bottle cannot be fixed at all.** A 2.4 mm wall is optically a *window*, not a
   lens. Making it solid glass would give it a caustic and destroy the internal descending tube
   that is the entire point of the piece. Its cap is therefore only widened (1.2 x 1.30, still
   centred on its column) so the soft 2x patch and the piece's own glints land on white — there
   is no focus to chase and the scene now says so.

**FURTHER RETRACTION, and the actual fix for COLOUR (2026-08-04).** Conclusions 1 and 2 above
are about *brightness*, and re-metering at the honest 2x bar showed the colour question had not
actually been answered by any of it: at 2x, **both** gyroid forms, the Klein bottle and a prism
all score 0.00% coverage, and the orb — the scene's brightest caustic — is a near-white disc.
The reason `sat` never revealed this is that `sat` measures distance from white and cannot tell
a uniformly amber patch from a red-to-violet fan. `_gemsweep.py` now also reports **spread**:
per caustic cell take chromaticity (r, b) = (R, B)/(R+G+B), find the excess-weighted centroid,
and report twice the weighted RMS radius about it. White collapses to a point plus sampler
noise; a spectrum is a long streak.

**Brightness and colour are separate properties.** A caustic is bright because a surface
*converges* light and coloured because it disperses light *sideways*, and the two normally
exclude each other: a ball lens is concentric, so its dispersion is purely longitudinal (every
wavelength on the same axis a little deeper, stacking into one white disc), while a prism
disperses hard sideways but is parallel-in/parallel-out, so it never converges and its coloured
band never rises above the bare sunlight beside it. An **axicon** — a flat top over a cone —
does both: light enters the top undeviated, every point of the conical exit face is a prism at
the *same* tilt, so the deviation is constant (a line focus, hence bright) while the dispersion
is a prism's (hence coloured). Measured at 2x, each piece at its own best drop:

| piece | coverage | sat | **spread** |
|---|---|---|---|
| **axicon**, 45 deg, apex down, drop 0.35 | 0.29% | **0.320** | **0.212** |
| axicon, same, drop 0.90 | 0.31% | 0.319 | 0.183 |
| apex-**up** cone, 42 deg, drop 0.90 | 0.07% | 0.289 | 0.157 |
| round **brilliant** cut, R=0.40, drop 0.90 | 0.18% | 0.257 | 0.092 |
| oblate spheroid (astigmatic lens) | 0.19% | 0.196 | 0.051 |
| crystal **orb**, drop 0.80 | 0.16% | 0.265 | 0.048 |
| glass torus (ring lens) | 0.03% | 0.210 | 0.045 |
| solid gyroid k=10 / shell gyroid k=13 / Klein / prism | **0.00%** | — | — |

Three further conclusions:

4. **A round brilliant LOSES**, which is a surprise given the trade cut it for "fire". Built as
   an intersection of half-spaces (1 table + 16 girdle + 8 crown bezels at 34.5 deg + 8 pavilion
   mains at 40.75 deg) it scores 0.092 spread, because a 40.75 deg pavilion sits just past
   crystal's 40.2 deg critical angle and **total-internally-reflects** — it throws the fire back
   up at the viewer, not down at the table. Sweeping the pavilion to 20/25/30/35 deg does not
   recover it.
5. **The lattice cannot be rescued by reshaping its outer boundary.** Clipping the solid gyroid
   to this same axicon instead of to a ball measures 0.13% / 0.205 / 0.099 — less than half a
   plain axicon on every axis. The clip only sets the *first* surface a ray meets, and behind it
   are the same internal sheets that make a gyroid a diffuser. So the crystal gyroid stays as
   it is and the colour source is an **additional** exhibit, not a replacement.
6. **The scene now has a tenth exhibit, `crystal_axicon`** — a 45 deg cone 1.12 m across, hung
   apex down on three short pins with its point 0.07 m over the tabletop, at near-left
   (2.60, 5.45). Its cap is 1.6 x 1.1, *wide and shallow* where every other cap is square or
   deep, because an axicon's caustic reads as two rainbow cusps flanking the piece rather than a
   disc under it, and it is cantilevered only 0.12 m (a lens throws its focus ~0.98 m downwind
   per metre of drop; an axicon has no focal *point* to displace, only a focal *line* starting
   at the exit face). Drop 0.35 and 45 deg are both metered optima. **Both stated REASONS are
   retracted below** (see the two RETRACTION blocks): colour does *not* fall off with drop, and
   the failure past 45 deg is not TIR. The choices themselves stand.

   **FACETING IT DOES NOT HELP, and that is worth knowing** because from the scene's low
   camera (11 deg above the horizon) a 45 deg cone is twice as wide as it is tall and
   foreshortens into a squat glass wedge — optically right, visually mute — so a faceted
   version that reads as a cut jewel was the obvious cosmetic fix. Every facet is at the same
   45 deg tilt, so the optics "should" survive; they do not. Cutting the same solid into
   6/8/12/16 pavilion facets at drop 0.35 gives 0.12/0.11/0.12/0.14% coverage at 0.063/0.162/
   0.099/0.132 spread, against the smooth cone's 0.29% and 0.212. Sampling the ring focus at n
   discrete azimuths instead of continuously collapses the patch from 1.43 x 0.96 m to a
   0.20 x 0.03 m sliver — brighter per unit area (facet 8 reaches sat 0.367) but far too small
   to read in a wide shot. The scene keeps the smooth cone. **(Re-measured on float below: the
   spreads all moved, the verdict did not.)**

   **Siting gotcha worth remembering: in this scene the NEAR row is HIGH z, not low z.** The
   still camera stands at (5.0, 2.95, 9.35) and looks toward -z, so "front of the gallery"
   means z around 5-7. The axicon's first site was z=1.10 — which reads as "front" on the page,
   since the file lists the low-z exhibits last — and that put it 8.4 m out and directly BEHIND
   the crystal gyroid, which occluded it completely. Nothing in the scene text or the audits
   catches this; only projecting the candidate point through the camera basis does (or
   rendering it, which is how it was found). At z=5.45 the piece is 4.2 m out at 178 px/m and
   clears the gyroid on screen by ~24 px.

**THE MEASUREMENTS ABOVE WERE ALL TAKEN THROUGH A CLIPPING 8-BIT PIPE, and that is a tooling
bug big enough to have its own fix (2026-08-04).** Metering the shipped frame instead of the
isolation rig (`scraps/_capchroma.py`, which projects each cap out of the scene through the
still camera and runs the same metric) showed the axicon's in-scene caustic at spread 0.057
against the rig's 0.212 — apparently a wash-out by the scene's sky-panel fill, which the rig
does not have. It was not. **596 of that cap's 22639 pixels are exactly (255, 255, 255).** A
PNG is 8-bit sRGB with a hard clamp, a caustic is by definition the brightest thing in frame,
and at the clip point all three channels become *equal* — so the tone map deletes precisely
the two quantities this whole investigation was measuring, hue and peak-to-screen ratio.
More than half the caustic's area was clipped, so its colour metered as white no matter what
the optics did, and the ranking table above was partly a ranking of how hard each piece hit
the clamp.

Fixed properly rather than worked around: **ftrace grew `-hdr`** (v0.132.0), which writes a
32-bit float PFM beside `-o` holding the scene-linear buffer — no exposure, no gamma, no
clamp — from the same `filmToLinear()` the tone map consumes, on the periodic in-progress
writes as well as the final one. `_capchroma.py` reads either format and **prints a warning
banner when handed an 8-bit file**. Rules that follow from this:

* **Never meter a caustic (or any highlight) off a PNG.** Render with `-hdr` and measure the
  `.pfm`. Looking at the PNG is fine; measuring it is not.
* A clipped core is not only a measurement problem, it is a *rendering* one: a caustic whose
  core is blown to white looks white on screen too. A caustic screen wants its albedo set so
  the caustic **peak** lands just under clip, not so the ambient does — `capwhite` was already
  taken from 0.88 to 0.30 for this reason and it is still not far enough under the axicon.
* Any earlier conclusion in this file that rests on a *bright* caustic's colour should be
  re-checked against a `.pfm` before it is trusted.

**AND THE CLAMP WAS, BY ACCIDENT, THE RIG'S ONLY OUTLIER REJECTION (2026-08-04).** The first
float measurements came back *worse* than the clipped ones, not better: the glass orb metered
peak **1389x** / sat 0.838, and the solid gyroid k=10 — a piece that throws no caustic at all,
0.00% coverage in every table above — metered peak **1214x** / spread **0.594** and would have
"won" the whole ranking on three pixels. Mode D carries one hero wavelength per sample, so a
rare specular-caustic path deposits an enormous **monochromatic** spike; the 8-bit clamp used
to flatten those to white, which is why nobody ever saw them. The 4x box average does not help
— a box is linear and zero-mean-symmetric, so it removes chroma *speckle* but a heavy-tailed
1000x outlier survives it scaled by 1/16 and still dominates.

So `-fireflies 3` (clamp a pixel brighter than 3x its 2nd-brightest neighbour, hue preserved)
is **mandatory for any float metering of this scene**, and is now in both rigs' render command.
It is safe for real caustics precisely because a caustic is never isolated — its neighbours are
bright too. Evidence: the shipped axicon measures peak **14.68x with and without** the flag,
bit-identical, while the gyroid's 1214x vanishes.

**The corrected ranking (float + `-fireflies 3`, 600 spp, 480x480, 4x box, 2x cut)** — this
supersedes every `coverage / sat / spread` table above, all of which were computed through the
clamp. Each piece at its own best drop; `patch` is the excess-weighted 5-95% extent:

| piece | peak | coverage | sat | **spread** | noise | **fan** | patch x by z |
|---|---|---|---|---|---|---|---|
| **axicon 45 deg, drop 0.35 (shipped)** | **14.68x** | **0.30%** | 0.275 | **0.079** | 0.008 | **0.76** | **1.37 x 0.96** |
| axicon 45 deg, drop 0.80 | 24.58x | 0.29% | **0.314** | 0.083 | 0.011 | **0.85** | 1.98 x 1.34 |
| apex-**up** cone k=0.9, drop 0.90 | 7.14x | 0.07% | 0.201 | 0.018 | 0.005 | — | 1.49 x 0.35 |
| round **brilliant** R=0.40, drop 0.80 | 3.68x | 0.06% | 0.219 | 0.047 | 0.011 | — | 0.35 x 0.06 |
| oblate spheroid b=0.22, drop 0.90 | 9.09x | 0.19% | 0.272 | 0.024 | 0.004 | 0.97 | 0.09 x 0.18 |
| glass **torus**, drop 0.90 | 7.85x | 0.03% | 0.210 | 0.006 | 0.003 | — | 0.03 x 0.06 |
| crystal **orb**, drop 0.80 | 3.00x | 0.16% | 0.253 | 0.013 | 0.008 | 0.94 | 0.09 x 0.15 |
| solid gyroid k=10, drop 0.80 | 2.60x | **0.00%** | — | — | — | — | — |

The *ordering* survives — the axicon still wins decisively, on area (1.3 m of cusp against the
orb's 9 cm patch), on spread 6:1, and on peak 5-8x — so every conclusion drawn from the old
tables about *which piece to ship* stands. But no absolute number in them does: they were
suppressed by the clamp at the top end and inflated by fireflies at the tail, in **opposite
directions**, so they cannot be rescued by rescaling. Two specific reversals:

* **The brilliant's "spread win" was a clamp artifact.** It read 0.092 through the clamp,
  nominally beating the axicon's clipped 0.079; on float it is 0.047 with too few cells to
  `fan`-test at all. Conclusion 4 above (a 40.75 deg pavilion TIRs the fire back at the viewer)
  is unaffected — the brilliant still loses, just by more.
* **`spread` measured over a handful of cells is not a comparable statistic**, which is why the
  table now carries `coverage`, `patch` and `fan` beside it. The oblate spheroid's fan 0.97 and
  the orb's 0.94 are *perfectly organised* colour over a 9 cm patch: real, and negligible.

**RETRACTION: drop is not a colour parameter — it buys patch AREA (2026-08-04).** Conclusion 6
above says "colour falls off monotonically above ~0.5 m of drop". That was the clamp talking:
a bigger drop throws a brighter caustic, which clipped harder, which the PNG scored as *less*
colourful. Swept on float:

| drop | peak | coverage | sat | spread | fan | patch x by z |
|---|---|---|---|---|---|---|
| 0.35 (shipped) | 14.68x | 0.30% | 0.275 | 0.079 | 0.76 | 1.37 x 0.96 |
| 0.50 | 18.43x | 0.27% | 0.257 | 0.076 | 0.82 | 1.55 x 1.11 |
| 0.65 | 34.54x | 0.24% | 0.281 | 0.076 | 0.82 | 1.78 x 1.23 |
| 0.80 | 24.58x | 0.29% | **0.314** | **0.083** | **0.85** | 1.98 x 1.34 |
| 0.95 | 6.27x | 0.36% | 0.277 | 0.083 | 0.82 | 2.22 x 1.49 |
| 1.10 | 5.02x | 0.56% | 0.256 | 0.065 | 0.67 | 2.33 x 1.55 |

`spread` is flat at 0.065-0.083 across the whole range — drop is very nearly a **free
parameter for colour** between 0.35 and 0.95. What it actually controls is how far the cusps
walk apart: the patch grows 1.37 x 0.96 m -> 2.33 x 1.55 m, i.e. **a bigger drop needs a bigger
cap**. That is a *staging* trade, not an optical one, and it is why the shipped 0.35 stays:
0.80 is nominally best on `sat` (0.314) and `fan` (0.85), but taking it would need the axicon
cap grown from 1.6 x 1.1 to ~2.1 x 1.45, whose near edge lands at z ~ 4.41 against the diamond
cap's z = 4.40 *and* overlapping it in y (0.70-0.90 vs 0.65-0.85, since the caps were
thickened). The gain is inside the run-to-run scatter; the layout surgery is not.

**RETRACTION: the "past 45 deg the exit face TIRs" mechanism is wrong (2026-08-04).** Same
conclusion 6. Swept on float at drop 0.35 (`k` = tan of the cone's half-angle; `k=1.0` is
45 deg), with the caustic core's z offset from the piece's axis:

| k | half-angle | peak | coverage | sat | spread | fan | core z |
|---|---|---|---|---|---|---|---|
| 0.70 | 35.0 deg | 6.49x | 0.09% | 0.234 | **0.129** | — | **+0.63** |
| 0.85 | 40.4 deg | 13.16x | **0.03%** | 0.480 | 0.009 | — | -0.53 |
| **1.00** | **45.0 deg** | **14.68x** | **0.30%** | 0.275 | 0.079 | **0.76** | -0.16 |
| 1.20 | 50.2 deg | 12.94x | 0.25% | 0.219 | 0.052 | 0.93 | -0.61 |
| 1.40 | 54.5 deg | 11.16x | 0.10% | 0.193 | 0.031 | — | -0.62 |

45 deg is confirmed optimal, but not for the stated reason. The collapse is at **k=0.85
(40.4 deg), BELOW 45 deg**, and both k=1.20 and k=1.40 keep working — so this is not a
one-sided TIR cliff past 45 deg. The core also **switches sides**, +0.63 z at k=0.70 to -0.61 z
at k=1.20, crossing near the null: the ring focus is passing through infinity there (the
constant prism deviation sweeping past the drop distance), which is what empties the 2x bar,
not the critical angle. 40.4 deg landing on crystal's 40.2 deg critical angle is a coincidence
worth naming precisely so nobody re-derives the wrong mechanism from it. Shallower cones are
also **short-range only**: `vcone0.70` at drop 0.80 and 1.10, and `vcone0.60` at 0.80, all read
0.00% coverage.

**FACETING STILL LOSES — the clamped conclusion survives honest measurement (2026-08-04).**
The one previously-rejected option that would have improved the scene *visually*, re-tested on
float at drop 0.35:

| piece | peak | coverage | sat | spread | noise | fan | patch x by z |
|---|---|---|---|---|---|---|---|
| facet6 | 3.19x | 0.12% | 0.252 | 0.023 | 0.011 | — | 0.20 x 0.03 |
| facet8 | 4.52x | 0.11% | 0.394 | 0.110 | 0.010 | — | 0.20 x 0.03 |
| facet12 | 5.58x | 0.12% | 0.244 | 0.045 | 0.010 | — | 0.17 x 0.03 |
| facet16 | 5.21x | 0.14% | 0.306 | 0.049 | 0.012 | 0.17 | 0.20 x 0.03 |
| **smooth cone** | **14.68x** | **0.30%** | 0.275 | 0.079 | 0.008 | **0.76** | **1.37 x 0.96** |

The absolute numbers all moved (facet8's spread went 0.162 -> 0.110, facet6's 0.063 -> 0.023)
but the verdict does not: every facet count sits at 0.11-0.14% coverage on a ~0.20 x 0.03 m
**sliver** — 1/220th the area of the smooth cone's patch — at a third to a fifth of the peak,
and **not one has enough cells for a trustworthy `fan`** (facet16's 0.17 is the only number and
it is weak, i.e. what colour is there is barely organised). Sampling the ring focus at n
discrete azimuths instead of continuously is what collapses it, exactly as the clamped
measurement said. The scene keeps the smooth cone.

**CLOSED for the axicon: the in-scene frame now measures as a coloured caustic (2026-08-04).**
`_capchroma.py` on a 1036-spp `-hdr -fireflies 3` render of the shipped scene, per cap, at the
2x-own-median bar:

| cap | coverage | sat | spread | xspread | peak | clipped | noise floor | **fan** |
|---|---|---|---|---|---|---|---|---|
| **axicon** | **6.11%** | 0.247 | **0.185** | **0.273** | **6.8x** | 0.66% | 0.056 | **0.46** |
| glass orb | 0.56% | 0.595 | 0.237 | 0.445 | 2.05x | 0.00% | 0.100 | — |
| brass cluster | 4.76% | 0.596 | 0.022 | 0.055 | 2.31x | 0.15% | 0.169 | — |
| crystal gyroid ("diamond") | 0.65% | 0.837 | 0.053 | 0.074 | 2.47x | 0.00% | 0.053 | — |
| chrome / dumbbell / heart / Klein / solid gyroid | 0.00% | — | — | — | 1.0-1.9x | 0.00% | 0.005-0.024 | — |

The axicon throws **11x the caustic area of anything else in the scene at 3x the peak**, and
it is the only cap in the scene with enough caustic on it to ask the colour question at all
(the `—` in the fan column is "fewer than 20 cells, unmeasurable", not "not coloured").
The scene's colour problem is solved.

**Two new controls, and `spread` on its own should never have been trusted without them.**
Looking at the frame after all this, the axicon's cusps still read whitish by eye, which does
not square with spread 0.185 — so the metric got two companions, both in `_pfm.py` so the two
rigs share one implementation:

* **`noise`** — `spread` measured over a *control band* of cells at 1.0–1.2x the cap's median,
  i.e. bare sunlit cap with no caustic on it. Whatever chromatic scatter those show is the
  render's speckle floor. **This is the thing that made `spread` untrustworthy**: mode `D`
  carries one hero wavelength per sample, so an unconverged pixel is *randomly coloured*, and
  an RMS radius in chromaticity scores a loud random cloud exactly like a rainbow. A 4x box
  only divides speckle by 4. In-scene at 2114 spp the axicon's floor is 0.056 against its
  0.185, so it clears by 3.3x; but the glass orb reads spread 0.237 on a 0.100 floor from
  three cells and swung between 0.001 and 0.237 across successive writes of the same render —
  that number was always noise, and the rig (which says the orb is white, 0.013) was right.
* **`fan`** — the fraction of chromatic variance explained by a weighted least-squares
  *quadratic in position* on the cap (adjusted R^2). Real dispersion means chromaticity is a
  **function of position**; speckle is uncorrelated with position and scores ~0 however loud.
  This is the one that actually settles it. The basis has to be quadratic: a first draft fitted
  a **plane** and scored the axicon 0.09, because an axicon disperses *radially* about its own
  axis — red outside, violet inside, on both flanking cusps at once — and a plane is blind to
  that by symmetry. On the quadratic basis the axicon scores **0.46 in-scene** (control band
  0.15) and **0.76 in the sun-only rig** (control 0.00, noise floor 0.008).

The rig also re-measures the orb at **fan 0.94 on spread 0.013**, which is the pair working as
intended: a ball lens's tinted rim is perfectly *organised* colour, there is just almost none
of it, over a 0.09 x 0.15 m patch. So `fan` validates a reading and `spread` sizes it — the
axicon wins on magnitude 6:1 and the ordering is unchanged.

**THEN LOOK AT IT, WHICH THE METRICS DO NOT REPLACE (2026-08-04).** All of the above is
statistics on a cap; none of it says what the picture looks like. `scraps/_capcrop.py` crops a
cap's screen footprint out of the float buffer and prints it three ways, stacked and upscaled:
**as shipped** (linear x GAIN, sRGB — exactly the PNG), **under-exposed** (gain set so the
cap's own 2x2 peak lands just under white), and **chromaticity only** (every pixel renormalised
to equal luminance with the saturation stretched). The three rows separate three different
failures that all look like "it reads white":

* row 1 white, row 2 coloured  -> the colour is real and the **tone map** is eating it;
* row 1 and row 2 both white, row 3 a smooth gradient -> the colour is real but **weak**;
* row 3 confetti -> there is no colour, only speckle (this is `fan` made visible).

On the crystal axicon it printed the second case: the cusp is a pale white arc with a faint
warm fringe, and row 3 shows a smooth but *low-amplitude* hue gradient. So `spread 0.185 /
fan 0.46` and "it reads whitish" were both true, and the two together are the actual
diagnosis — organised colour, not enough of it.

**Two levers were ruled out by measurement before the third was tried.** (1) **Darken the
screen further.** It cannot work: chromaticity is scale-invariant, so albedo moves the caustic
and its pedestal together (see below). (2) **Cut the sky fill**, on the theory that the 11000 K
panel washes the cusps out. Also no — profiling the cap's luminance percentiles says the
pedestal is almost entirely *direct sun*, not fill:

| percentile of the axicon cap | scene-linear luminance | what it is |
|---|---|---|
| p5 | 0.0089 | the piece's own shadow, i.e. **sky fill alone** |
| p50 | 0.0893 | sunlit cap = sun + fill |
| p100 | 0.6292 | the caustic core |

The fill is **10% of the pedestal**; removing all of it would raise the caustic:screen ratio by
a tenth. And the remaining 90% is sunlight, which cannot be reduced without dimming the caustic
by the same factor, since both arrive from the same sun. The pedestal is therefore fixed, and
the only thing left to change is **how much the piece disperses**.

**THE FIX: cut the axicon from DENSE FLINT instead of crystal (2026-08-04).** What splays a
caustic across the spectrum is the Abbe number, and the scene's own material comment had the
answer written in it the whole time — `glass:SF10` (V_d 28.5) splays 1.5x as far as
`glass:crystal`/F2 (36.3). The rig gained `GEMIOR` (and `GEMRES`/`GEMSPP`, because at 480 px a
0.13%-coverage caustic is 19 cells and `fan` needs 20). **A denser glass also deviates harder,
so it moves the ring focus and the drop has to be re-swept with the material** — SF10 at
crystal's optimum drop of 0.35 lands in a null (0.08% coverage). Swept, at 45 deg, 480 px:

| drop | coverage | sat | spread | fan |
|---|---|---|---|---|
| 0.30 | 0.04% | 0.520 | 0.045 | — |
| 0.35 | 0.08% | 0.548 | 0.050 | — |
| 0.40 | 0.08% | 0.545 | 0.047 | — |
| 0.50 | 0.13% | 0.480 | 0.109 | — |
| **0.65** | 0.15% | 0.469 | 0.139 | **0.95** |
| 0.80 | 0.16% | 0.482 | 0.121 | 0.92 |

Head to head at matched settings (960 px, 1200 spp — the sweep setting is not fine enough to
adjudicate this), each glass at its own best drop:

| glass | drop | peak | coverage | sat | spread | fan | patch |
|---|---|---|---|---|---|---|---|
| crystal | 0.35 | 16.44x | **0.28%** | 0.321 | 0.152 | 0.51 | 1.37 x 0.98 |
| SF10 | 0.50 | **24.71x** | 0.13% | 0.509 | 0.162 | 0.57 | 1.27 x 1.05 |
| **SF10** | **0.65** | 21.99x | 0.15% | **0.516** | **0.187** | **0.77** | 1.41 x 1.14 |

**Note the metrics are resolution-dependent** — the same crystal configuration reads spread
0.079 / fan 0.76 at 480 px and 0.152 / 0.51 at 960 px, because finer cells resolve more
structure *and* carry more per-cell noise. Only compare rows taken at the same `GEMRES`.

Shipped: `material "flint" { type dielectric ior glass:SF10 }` on `crystal_axicon` only (the
orb and the gyroid keep crystal — their caustics are white-with-a-rim whatever the glass), the
piece raised 0.30 m to drop 0.65 (`translate 2.6 1.55 5.45`, `contained_by` y 1.25..1.85, pins
0.37 -> 0.67 m long), and the cap regrown to the new patch: 1.6 x **1.28** centred (2.54, 5.41).
`_flyplan.py`'s and `_capchroma.py`'s collider tables were moved with it — both hardcode the
axicon's apex and both would otherwise mask the wrong region. In the finished frame:

| | crystal, drop 0.35 | **SF10, drop 0.65** |
|---|---|---|
| coverage | 4.48% | 3.64% |
| sat | 0.269 | **0.444** |
| spread | 0.204 | **0.299** |
| xspread (pedestal removed) | 0.292 | **0.413** |
| peak | 6.29x | **8.05x** |
| clip at albedo 0.15 | 0.52% | **0.13%** |
| noise floor | 0.066 | 0.045 |
| **fan** | 0.37 | **0.76** |

(Both columns are the fully converged 600 s stills, `png/rain_axicon.pfm` and
`png/rain_axicon_cam.pfm`. Intermediate reads from ~380 spp on were already inside a few
percent of these, so the comparison was never spp-limited.)

1.7x the saturation, 1.5x the chromatic spread and 2.1x the organisation, for 80% of the area
— and it *clips less*, because the caustic got smaller as it got brighter. That leaves the
0.15 albedo with more headroom than it now needs (doubling it to 0.30 would clip 1.36%, against
3.06% for the crystal piece), so a brighter, more museum-white tabletop is available if the
scene ever wants one. Left alone here to avoid moving two variables at once.

**The obvious objection to that table is TAIL SELECTION** — SF10 lights 80% as much of the cap,
so maybe its `sat` is high only because the threshold kept a smaller, brighter, more selected
slice. Tested by sweeping `GEMCUT` (the multiple of the cap's own median that counts as
caustic) on both finished stills until they meet on area:

| GEMCUT | crystal: coverage / sat / spread / fan | SF10: coverage / sat / spread / fan |
|---|---|---|
| 2.0 | 4.48% / 0.269 / 0.204 / 0.37 | 3.64% / **0.444** / **0.299** / **0.76** |
| 2.5 | **3.37%** / 0.210 / 0.117 / 0.35 | 1.48% / 0.344 / 0.172 / 0.72 |
| 3.0 | 2.58% / 0.192 / 0.065 / 0.22 | 0.74% / 0.347 / 0.100 / -- |
| 3.5 | 2.06% / 0.179 / 0.057 / 0.17 | 0.47% / 0.369 / 0.101 / -- |
| 4.0 | 1.32% / 0.186 / 0.051 / 0.49 | 0.32% / 0.368 / 0.093 / -- |

**The objection is refuted, and backwards.** Squeezing crystal down to SF10's area (cut 2.5,
3.37% vs 3.64%) makes it *worse* on every axis — sat 0.269 -> 0.210, spread 0.204 -> 0.117 —
so at matched coverage SF10 wins by 2.1x on sat, 2.6x on spread and 2.2x on fan, a wider
margin than the headline table shows. SF10 at cut 2.5 beats crystal at cut 2.0 on sat and fan
while lighting a third of the area.

The mechanism is in the column shapes, and it is the whole point of the swap. **Crystal's
caustic gets WHITER toward its core** (sat 0.269 -> 0.179 from 2x to 3.5x the pedestal): its
colour lives in the low-excess *fringe*, which is exactly the part a tone map crushes and the
eye reads as dim. **SF10's does not** — sat is flat at 0.34-0.37 all the way out to 4x, i.e.
the colour survives into the bright core, where it is actually visible. That is why the piece
metered as coloured under crystal yet looked white on screen, and why it now looks amber:
raising V_d^-1 did not just add dispersion, it moved the dispersion into the bright pixels.
(`fan` reads `--` past cut 2.5 for SF10 because fewer than 20 cells survive, not because the
structure decays — see the quadratic-basis note above.)

**And `capwhite` 0.30 -> 0.15 is a TONE-MAP decision, not a measurement one — a distinction
worth writing down because it is easy to get backwards.** On the float buffer, chromaticity
and peak:median are both scale-invariant, so a diffuse screen's albedo changes `spread`, `sat`
and `coverage` by *literally nothing* (verified: identical to three decimals at both albedos,
by rescaling the same `.pfm`). What it changes is how much of the caustic the tone map deletes:
on that same buffer the axicon cap clips **3.06% of its pixels at 0.30 and 0.70% at 0.15** —
half the caustic's own area versus a tenth of it. So `_capchroma.py` now reports a `clip`
column beside the float metrics (scene-linear x `GAIN` 1.5 >= 1.0, where 1.5 = ftrace's
`ABS_EXPOSURE_GAIN` 6.0 x the film's `exposure 0.25`), because the float metrics alone cannot
tell you the picture is still throwing the caustic away.

**Tooling bug found and fixed along the way: `rotate` is not a valid key inside an FTSL
`function` block** (only `translate` is), and the loader emits a **warning, not an error**, then
carries on — so a rotated field silently renders unrotated. A prism roll sweep came back as six
identical rows before this was noticed. Fixed two ways: rotate half-space normals *algebraically*
in the generator (a half-space `n.p <= d` under `p = Rq` is `(R^T n).q <= d`, so pre-rotating the
normals is exact and needs no transform support), and gate every generated scene through
`-parseonly`, raising on any warning rather than measuring the wrong object. **Worth checking
whether other FTSL blocks accept unknown keys with only a warning** — a silent mis-parse that
still renders is the worst failure mode a scene language can have.

**RETRACTION: "render mode M" was bad advice — mode M silently drops participating media.**
`src/photonmap_render.h` contains no `scene.media` handling at all; its only `Medium` symbols
are the nested-dielectric IOR stack. Verified by running gallery_rain in mode M: the cloud,
the rain and the rainbow are simply absent and the sky is pure black. For this scene mode **D**
is the correct choice and always was — `main.cpp` (~line 4491) explicitly exempts D from the
`mediaNeedForward` warning because "it handles multiple superposed, box/sphere/object-bounded
AND heterogeneous media correctly on both devices". See the next entry.

---

### OPEN: mode `M` (and `S`) silently ignore participating media instead of refusing the scene

`src/photonmap_render.h` has no participating-media code — no free-flight sampling, no volume
photon deposits, no in-scatter on the camera walk. (`photonmap.h`'s header comment claims the
pass "deposits a record at each diffuse/**volume** vertex", which is aspirational.) A scene
with a `medium` therefore renders in mode M as if the medium were not there.

The problem is not the missing feature, it is that **nothing tells you**. ftrace already has
the machinery to say so: `unsupportedFeature()` / `vcmUnsupportedFeature()` in `main.cpp`
(~3960, ~4004) make mode `U` refuse a scene outright with "participating media (mode U is
surfaces-only)", and `prefer{}/else{}` then falls through to a mode that works. Modes `M` and
`S` have no such check, so `gallery_rain` — whose whole subject is a rain cloud and a rainbow —
rendered a black sky for six minutes with no diagnostic.

Reproduce: `ftrace scenes/gallery_rain.ftsl -mode M -n 4e8 -spp 24 -o png/x.png` and compare
with `-mode D`. Cloud, rain and bow are missing.

Proper fix: add `scene.media.empty()` to the mode `M`/`S` support predicate alongside mode U's,
so a media scene either refuses (and `prefer{}` falls through) or at minimum warns. The real
fix — a volume photon map with beam/point-query in-scatter — is a much larger job and should be
tracked separately; refusing loudly is the correct behaviour until then.

---

### DONE (2026-08-04): `gallery_rain` stand caps intersecting neighbouring stands' columns

Widening the caps for the caustic work put `stand_gyroid_cap` (2.3 x 2.3 at x=5.0, z=4.8,
y=0.52..0.55) straight through the glass stand's COLUMN (x 5.99..6.41, y 0.18..0.87). The
audit done at the time compared cap-vs-cap only and concluded the caps "stagger like shelves";
that reasoning is valid for two caps (each occupies one thin y slab) and invalid for a cap
against a column (which spans a whole y range, so plan overlap = intersection).

Two further intersections turned up that PREDATED the wireframe rework, both base slab against
base slab: `stand_gyroid` x `stand_glass` (0.30 x 0.18 x 0.20) and `stand_klein` x `stand_brass`
(0.61 x 0.22 x 0.11).

Fixed: gyroid cap 2.3 x 2.3 -> 2.3 x 2.1 (near edge 3.65 -> 3.75, clearing both the glass and
diamond columns); `stand_glass` x 6.2 -> 6.7; `stand_brass` z 2.0 -> 1.84 with the settled
`brass_cluster` shifted by the same -0.16. `scraps/_standaudit.py` now audits all 30 colliders
across the 9 stands in 3-D and comes back clean — run it after any stand edit.

The audit script itself had a bug worth remembering: it terminated each `isosurface` block on a
regex `\n\}`, but cap blocks are written `isosurface "x_cap" { material capwhite\n    box {...} }`
and close on the SAME line, so every cap silently swallowed the following stand and reported
that stand's cage boxes under the cap's name (8 stands instead of 9, and 18 bogus duplicate
pairs). It now uses a real brace matcher.

---

### PARTLY FIXED (2026-08-04, v0.131.0): a dispersive caustic in mode `D` is single-wavelength with no variance reduction — chromatic speckle is unavoidable

Mode `D` is the only mode that renders a scene containing both participating media and a
dispersive caustic (M and S drop the media, see above; U refuses media outright). But every
wavelength-sharing optimisation the renderer has is off in exactly that combination:

- **`-heroc` (hero-wavelength bundle) is disabled by the media.** It is documented as "ignored
  (still single-lambda) by ... any scene with participating media", so `gallery_rain` gets no
  bundle at all.
- **A dispersive refraction would de-hero the bundle anyway.** The dispersive event terminates
  the N-1 secondary wavelengths, so even without media the orb's own refraction kills the
  bundle at the first surface.
- **`-herosplit` — the fix for exactly that — is ignored by mode `D`.** It is implemented for
  CPU forward `A`/`B`/`C` and the `M`/`S` deposit only.
- ~~**There is no denoiser.** No `-denoise` flag exists.~~ — **FIXED, see below.**

Net effect: the caustic converges one wavelength per path, so it arrives as saturated red/green/
blue speckle that only brute-force sample count removes (variance goes as 1/spp, so halving the
speckle costs 4x the time). 1400 spp / 17 min at 1280x720 is visibly noisy in the disc.

Proper fix, in increasing order of work: (a) extend `-herosplit` to mode `D`'s subpath
construction, which is where the win is largest since D is the mode media scenes are forced
into; (b) allow `-heroc` on media scenes by carrying per-lambda transmittance through
delta/ratio tracking instead of bailing; (c) a denoiser.

**(c) DONE 2026-08-04 (v0.131.0): `-denoise` (`src/denoise.h`).** Chroma-only edge-aware
à-trous filter in `filmToRgb8`, so it affects the file and the live window identically and runs
before the p99 auto-exposure anchor is measured. On `gallery_rain` at 120 spp, against an
8000 spp reference of the same frame: chroma RMSE 23.8 → 13.8 (58 %), PSNR +1.8 dB, for ~1 % of
render time, with luma left **bit-identical**. Still a net win at 480 spp (+0.6 dB), so it is
safe to leave on. It attacks the *symptom* rather than the cause, so **(a) and (b) remain open**
— a denoiser cannot recover a colour gradient the sampler never resolved, and the fringing at a
caustic rim is exactly the chroma detail the filter has to soften.

Three bugs were found and fixed while building it, each of which passed a naive visual check
and is now pinned by `ftrace -checkdenoise`:

1. **A plain bilateral gather ate 30 % of the frame's luminance.** `out_i = Σ w_ik in_k / Σ w_ik`
   is row-stochastic but not column-stochastic, so on heavy-tailed MC noise it regresses toward
   the local *mode*. Fixed with a symmetric (geometric-mean) tolerance so `w_ik == w_ki`, plus a
   *scatter* for luma, `out_i = Σ w_ik (in_k / D_k)`, whose total is exactly `Σ in_k`.
2. **A constant image was not a fixed point** (bright frame around a flat grey field): border
   taps were being dropped, so edge rows summed to < 1 and handed out more than they held.
   Fixed with **half**-sample edge mirroring — whole-sample mirroring has fixed points at the
   two end samples, which double-counts, breaks the symmetry property 1 depends on, and leaked
   0.06 %.
3. **Averaging chroma turned speckle into coherent purple/orange blobs, and was measurably
   *further* from the truth** (PSNR −2.4 dB) even though it looked less noisy. Chroma is stored
   as a ratio to luma, and the mean of a ratio whose denominator is the noisy quantity is not
   the wanted quantity: near-black pixels have enormous ratios and dominated the average. Fixed
   by averaging numerator and denominator separately, `Σw(R−G) / ΣwY` — the luma-weighted mean
   hue.

The lesson worth keeping: **energy conservation and a flat-field test both passed while the
filter was making the image objectively worse.** The only metric that caught it was RMSE against
a converged reference (`scraps/_dneval.py`), which is now how any change here should be judged.

## FIXED (2026-08-05, loom-only): `write_obj`'s atomic replace flaked with WinError 5, and the same helper was copy-pasted four times
`tools/loom/tests/test_mcubes.py::test_isomesh_static_field_baked_once` failed
intermittently (once in three full-suite runs) with

    PermissionError: [WinError 5] Access is denied:
      '…\tmpwoxkkky5.obj.tmp' -> '…\s.obj'      (loom/sweep.py:235)

`os.replace` is atomic on Windows but **not immune to sharing**: `MoveFileEx` fails with
`ERROR_ACCESS_DENIED` (5) or `ERROR_SHARING_VIOLATION` (32) if anything holds a handle to
the source or destination at that instant — which includes the antivirus scanner and the
search indexer opportunistically opening a file loom just closed, not only a reader loom
knows about. That makes it a live-channel hazard, not just a test flake: §F4 re-emits a
scene on a worker thread while ftrace is loading the previous emission's assets out of the
same directory, which is precisely when a handle collision is most likely.

Aggravating factor: the same temp-file + `os.replace` + cleanup block existed **four
times** — `sweep.py` `write_obj`, `anim.py` `CurveDrive.save`, and two byte-identical
`_atomic_write_text` copies in `anim.py` and `viewer.py` — so a fix in one would not have
reached the others.

**Fix:** one shared `loom/atomicio.py` (`write_atomic` / `replace_atomic`) that all four
call sites now use. It retries a `PermissionError` on a doubling backoff (2 ms → 64 ms,
1 s total budget) and then re-raises, so a transient scanner handle costs milliseconds
while a genuine permission problem still fails promptly and audibly. The temp file is
still removed on any failure, so the old file survives intact. Pinned by
`tools/loom/tests/test_atomicio.py` (6 tests): the retry succeeds after two simulated
WinError 5s, a persistent one propagates with the old file unchanged, a non-sharing
`OSError` is *not* retried, and no `.tmp` litter is left in any case.

## FIXED (2026-08-05, loom-only): loom's own tests modelled a `point` light, which ftrace does not have
Four loom test modules built scenes with `Light("point", position=(3, 3, 3), name="key")`
or `Light("point", intensity=1.0)`. ftrace's `addLight` (`src/ftsl.h` ~4840) recognises
`collimated` / `sphere` / `cylinder` / `spot` / `env` / `sun`, and **everything else falls
through to the default rectangular area light** — so a `point` light renders as a large
white quad, while `position` / `name` / `intensity` are unknown keys that only *warn*.

Harmless inside those tests (none of them render), but it is exactly the kind of fixture
that gets copied: writing a validation scene from that spelling produced a render dominated
by a giant white quad before the cause was obvious. loom's `Light` is schema-free **on
purpose** (its docstring: "loom does not invent light fields"), so the fix is not
validation — it is that the fixtures now spell lights that exist:
`Light("sphere", center=…, radius=…, power=…)` in `test_image_term.py`,
`test_material_bundle.py` and `test_viewer.py`, and `Light("collimated", origin=…, dir=…)`
in `test_grammar_scene.py` (which deliberately round-trips a *non-default* subtype).
