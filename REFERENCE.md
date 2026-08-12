# ftrace — reference

The complete reference for ftrace's render modes, cameras, materials, spectra,
lights, geometry, textures, patterns, media, scene language and command line.
[`README.md`](README.md) is the short version — what ftrace is, how to build it,
and how to get a first image out of it.

Three neighbouring documents cover what this one only summarises:

| Document | Covers |
|---|---|
| [`FTSL.md`](FTSL.md) | The **authoritative** scene-language grammar: every block, every key, every value form, and exactly what the parser accepts. The [Scene language](#scene-language-ftsl) section below is a tour, not a spec. |
| [`tools/loom/README.md`](tools/loom/README.md) | Loom, the Python procedural-animation toolkit that generates scenes and drives frame sequences. |
| [`known-issues.md`](known-issues.md) | Unsolved bugs and tracked technical debt. |

---

## Contents

- [Render modes (`-mode`, or per-camera `mode`)](#render-modes--mode-or-per-camera-mode)
  - [Mode `W` — the deterministic (POV-Ray-style) preview](#mode-w--the-deterministic-pov-ray-style-preview)
  - [Speed / accuracy / ability tradeoffs](#speed--accuracy--ability-tradeoffs)
  - [Backends & performance (`-device`, `-wavefront`)](#backends--performance--device--wavefront)
- [Cameras](#cameras)
- [Materials](#materials)
  - [Hair and fur fibers (`hair`)](#hair-and-fur-fibers-hair)
    - [The medulla — what makes fur not hair](#the-medulla--what-makes-fur-not-hair)
    - [`preset` — measured species](#preset--measured-species)
    - [Dual scattering (`-dual-scatter`)](#dual-scattering--dual-scatter)
      - [The fiber-density grid (`-dual-grid`)](#the-fiber-density-grid--dual-grid)
      - [The coat as a medium (`-fur-volume`)](#the-coat-as-a-medium--fur-volume)
      - [Choosing a tier (`-fur-lod`)](#choosing-a-tier--fur-lod)
- [Spectra (SPDs, reflectances, indices)](#spectra-spds-reflectances-indices)
  - [Spectral representation vs. other renderers](#spectral-representation-vs-other-renderers)
- [Lights](#lights)
- [Geometry](#geometry)
  - [Curves and fibers (`curve`)](#curves-and-fibers-curve)
  - [Grooms (`fur`)](#grooms-fur)
  - [Implicit surfaces (`isosurface`)](#implicit-surfaces-isosurface)
- [Textures](#textures)
  - [Stochastic tiling (`tiling stochastic`)](#stochastic-tiling-tiling-stochastic)
- [Procedural patterns (math-driven materials)](#procedural-patterns-math-driven-materials)
  - [Putting them together — non-stationary noise](#putting-them-together--non-stationary-noise)
- [Participating media / fog](#participating-media--fog)
- [Scene language (FTSL)](#scene-language-ftsl)
  - [Conditional blocks (`prefer { … } else { … }`)](#conditional-blocks-prefer----else---)
  - [Camera animation (`camera_path`, `camera_orbit`)](#camera-animation-camera_path-camera_orbit)
  - [Multi-camera shared photon pass (modes `A`, `B`, and `M`)](#multi-camera-shared-photon-pass-modes-a-b-and-m)
  - [Stereoscopic 3-D (`-stereo`)](#stereoscopic-3-d--stereo)
  - [Animated geometry (OBJ sequences) → video](#animated-geometry-obj-sequences--video)
  - [The shared grammar](#the-shared-grammar)
  - [Importing Mitsuba scenes](#importing-mitsuba-scenes)
- [Command-line reference](#command-line-reference)
- [Output](#output)
  - [Denoising (`-denoise`)](#denoising--denoise)

---

## Render modes (`-mode`, or per-camera `mode`)

Modes `A`/`B`/`C`/`P` trace **identical forward physics** and differ only in how
the camera *measures* the light (splat vs. physical catch vs. composite). `R` and
`D` are separate backward / bidirectional estimators. All modes are unbiased and
converge to the same image; they trade **speed, noise character, and which light
paths they can capture at all**.

| Mode | Name | What it does | Backend |
|---|---|---|---|
| `A` | Finite-lens camera | Forward next-event splat through a finite aperture + thin lens (true depth of field, efficient) | CPU + GPU |
| `B` | Pinhole splat *(default)* | Light-tracing splat to a pinhole camera; independent photons | CPU + **GPU** |
| `C` | Finite-aperture catch | Forward photon catch through a thin lens (real depth of field) | CPU + GPU |
| `R` | Backward reference | Backward path-traced reference image; drives the physical-lens camera | CPU + **GPU** |
| `W` | Deterministic preview | Mode `R` with **every estimator replaced by a fixed quadrature** — a POV-Ray-style Whitted render that is noise-free at `-spp 1` and ~2 orders of magnitude faster than converging `R`. Trades multi-bounce GI (see `-ambient` for a flat fill, `-gi` for a real deterministic one-bounce gather) and unbiasedness for speed. Also drives the interactive viewer's live lit preview (`-explore`, `T`) | CPU + **GPU** (see below) |
| `V` | Validate | Runs `B` and `R` and reports the best-fit residual between them | CPU (+GPU forward pass) |
| `P` | Composite | Forward `B` for diffuse/caustic pixels + a backward camera ray for specular/coated surfaces | CPU + **GPU** |
| `D` | BDPT | Bidirectional path tracing with MIS over every light×camera connection | CPU + **GPU** |
| `M` | Photon map | Builds a **view-independent** photon map once, then gathers the camera image from it — a direct radius density estimate at the first diffuse hit, or a Jensen final gather one bounce away with `-pmfg <K>` (reusable across cameras) | CPU + **GPU** (direct estimate) |
| `S` | SPPM | Stochastic **progressive** photon mapping: repeated photon passes with a shrinking per-pixel radius — converges (unbiased in the limit), bounded memory, excels at caustics | CPU + **GPU** |
| `U` | VCM/UPS | Vertex **connection and merging**: BDPT vertex connections **and** SPPM photon merging combined under one MIS weight — robust across diffuse GI, glossy, and caustics in a single estimator | CPU + **GPU** |

### Mode `W` — the deterministic (POV-Ray-style) preview

Every other backward mode here is **Monte Carlo**, so its images are noisy until you
spend samples: mode `R` draws a random point on each area light, a random direction in
each glossy lobe, a random Russian-roulette coin at each specular vertex, a random
subpixel offset, and a random wavelength. That's why `-mode R` looks grainy even with
`-rgb` — `-rgb` only removes the *wavelength* dimension, not the other four.

The classic ray tracers (POV-Ray, and Whitted's original) are noise-free precisely
because they replace all of those draws with **fixed quadratures**, and `-mode W` does
the same on top of ftrace's mode-R walk:

| Mode `R` (random) | Mode `W` (fixed) |
|---|---|
| one random point per area light | an **N×N lattice** over the light (`-whitted-grid`, default 4 → 16 shadow rays) |
| random direction in the glossy lobe | the **mirror direction**, weighted by the lobe's reflectance |
| Russian roulette at specular vertices | **attenuate the throughput** instead, and stop below an `adc_bailout`-style 1/512 cutoff |
| random branch at half-mirrors / layers / mixes / thin films / multilayers | the **dominant** branch, weighted (mixes hard-threshold at ½) |
| random diffraction order at a `grating`; random Stokes-shift excitation λ at a `fluorescent` | the same weighted pick, but driven by a **low-discrepancy lattice** indexed by (sample, bounce) instead of the rng — one *fixed* choice per sample, not a coin. Sample 0 gives the grating's **specular order m = 0** and the **median** excitation λ; extra `-spp` fan the diffracted spectrum out into the higher orders |
| random wavelength per sample | a fixed lattice of **8 hero wavelengths** riding one BVH walk |
| random subpixel jitter | a progressive low-discrepancy pattern, **identical in every pixel** (sample 0 is the pixel centre) |
| stochastic diffuse indirect | dropped — implies `-direct-only`; see `-ambient` |

The last row is the point: neighbouring pixels differ only by their *geometry*, never by
their *luck*, so there is nothing to average out. On a 420² test scene, mode `R` on the
CPU needs ~625 spp (~80 min) to reach 4 % graininess; mode `W` produces a clean image in
**14.6 s at `-spp 1`** — about **300×**. In mode `W`, `-spp` stops meaning "less noise"
and starts meaning *finer antialiasing and a denser spectrum*; the picture gets sharper,
never less grainy, and the progress line reads `deterministic` instead of a noise figure.

**One exception: participating media.** The quadrature above covers the *surface* half of
mode `R`. The fog branch was never converted, so a scene with a `medium` still takes a
**random free flight** (`−ln(1−u)/σt`) and a **random** volume shadow connection at each
pixel. The result is bit-identical run to run (the rng is seeded from the pixel), but it is
*speckled*, not noise-free — at `-spp 1` each pixel either misses the light entirely or
connects and carries the whole `1/pdf` as a blown-out dot. Add `-no-media` for a genuinely
clean 1-spp preview, or spend `-spp 32..64` to converge the haze.

**And media are one of the few places the CPU and GPU backward tracers disagree.** The
**GPU** megakernel superposes the scene's **whole** `media` list — every `bounds` region,
every `density` field, every per-medium phase function, `phase rainbow` included — so
`-mode W` on the GPU renders bounded clouds and a real rainbow. The **CPU** backward
(`backward.h`) still collapses everything to the **first authored medium**, as a global
homogeneous haze with `bounds`/`density` ignored, so the same scene on `-device cpu` loses
the clouds and the bow entirely. ftrace warns (`[medium] …`) when a render's backward layer
lands on the degraded CPU path. Both the noise and the divergence are logged in
`known-issues.md`.

**`-ambient <v>` — the GI stand-in.** With the diffuse indirect bounce gone, a *closed*
room previews with black shadows, because everything not directly facing the light is lit
purely by bounce. `-ambient` adds POV-Ray's flat fill at every diffuse vertex; it is
dimensionless — a fraction of a light's own radiance — so a given value means the same
thing whatever the scene's absolute radiometric scale. That makes it *scale*-independent,
which is not the same as scene-independent: how much fill a room actually wants still
depends on how closed and how reflective it is. `0.02..0.2` is the useful band for a fairly
open room (on the gold-gyroid room, `0.05` roughly halves the error against the full-GI
reference), but a small closed bright box wants far less — nearer `0.01` (see `-gi` below).
Sweep it rather than trusting one number. It is physically a lie, and it is what makes the
mode usable indoors.

**`-gi <n>` — real one-bounce GI, and why it's safe on animation.**
`-gi <n>` replaces the flat term with an actual **deterministic single-bounce hemisphere
gather**: at every diffuse vertex it traces `n` rays along a fixed lattice and takes
whatever mode-`W` radiance each one finds. `16..64` is the useful band; cost scales
with `n`.

Measured on an all-diffuse Cornell box (`scraps/cor_gi.ftsl`, 240², **absolute exposure**,
against mode `R` converged to 0.8 % noise / 15636 spp), whole-frame mean |luminance error|
over the box interior, and a **colour-bleed** score — how much redder the floor is beside
the red wall than beside the green one, which cancels the light's own tint and the exposure
and so is the one thing a grey fill provably cannot buy:

| mode `W` variant | mean \|err\| | colour bleed |
|---|---|---|
| direct only | 33.32 | 13 % |
| best flat fill, `-ambient 0.01` | 7.01 | 18 % |
| gather only, `-gi 32` | 12.91 | 55 % |
| **`-ambient 0.01 -gi 32`** | **5.40** | **80 %** |

So the honest headline is **23 %** less luminance error than the best flat fill — not a
landslide — but **4.5× more of the colour bleeding** (80 % vs 18 %), and that gap is the
real point: the flat fill's bleed score never exceeds 43 % no matter how bright it is
driven, because a grey constant cannot carry a wall's colour. Contact darkening behaves the
same way: the gather's signed error in the darkest decile is +0.2 at `-ambient 0.005`,
i.e. essentially exact, where the flat fill has to trade the crevices against the open
faces with one knob.

Two limits worth knowing, both measured:

* **The gather saturates at about `-gi 32`.** 12.91 → 12.80 → 12.77 → 12.77 for
  `-gi 32/64/128/256`. Past that the residual is not the direction count, it is the
  *single bounce*: in a closed box of 0.75-albedo walls the interreflection series totals
  ~1/(1−0.75) = 4× the first bounce, so one bounce structurally cannot get there. This is
  why the combination wins — `-ambient 0.01 -gi 32` (5.40) beats `-gi 256` alone (12.77)
  at a fraction of the cost, because the flat tail is standing in for the *rest of the
  series* rather than for all of GI.
* **`-spp` is what removes the banding.** Residual blotchiness falls 6.57 → 2.41 → 1.65
  for `-spp 1/4/16` (and mean error 5.40 → 5.09 → 4.88). `-spp 4` buys most of it.

> **The useful `-ambient` value is scene-dependent — sweep it.** The `0.02..0.2` band
> above suits a fairly open room; on this *closed* box the optimum was **0.01**, and
> `0.05` already overshot the darkest decile by +28 luminance units. Because it is a
> fraction of a light's own radiance, a small closed bright box needs far less of it than
> the number alone suggests.

The design difference from POV-Ray matters if you are rendering a sequence. POV-Ray's
radiosity caches irradiance at an **adaptively chosen** sparse point set and interpolates.
Which points get sampled depends on render order and on the local geometry, so on animated
geometry the cache's low-frequency blotches pop in and out between frames — which is why
POV-Ray animations conventionally pre-bake one cache and reuse it, and why that only works
when nothing moves. This gather has **no cache and no adaptivity**: its direction set is a
pure function of (lattice index, sample index) and never of the scene, so two frames of a
rotating object are lit by the identical estimator and **a seamless loop cannot flicker**.
The price is that residual error appears as low-frequency *banding* rather than noise — but
banding is a smooth function of the surface normal, so it slides smoothly as geometry
turns, where cache splotches jump. Raise `-gi`, or raise `-spp` (which rotates the lattice
progressively), to push it down.

That is measured too, not just argued (`scraps/gi_temporal.py`): a box on a turntable
through a 12° arc, scored on the per-pixel discrete *second* difference in luminance — the
quantity that spikes when a value pops between frames even if the frame-to-frame difference
looks reasonable. Normalised against the first difference, `-gi 32` scores **1.907** against
a direct-only control of **1.851** — within 3 % of the smoothest thing this renderer can
produce — and at `-spp 4` it comes in *below* the control, at 1.369. (The test resolves
gross flicker, the multi-unit blotches a cache pops; 8-bit output puts a rounding floor
under the subtle end.)

`-ambient` still applies alongside `-gi`, now in its honest role: the **far-field** fill a
gather ray picks up when it escapes the geometry. In an empty scene every direction
escapes, so the gather collapses back to the flat term — switching `-gi` on never steps the
exposure. This is verified rather than asserted: on `scraps/gi_collapse.ftsl` (a lone
diffuse quad lit only by `ambient`) `-gi 32` and `-gi 0` render **pixel-identical** frames,
which is the check that the gather's cosine normalisation is right. The one legitimate
exception is a directly visible light: a gather ray that lands on an emitter contributes
nothing, because the vertex's own next-event estimation already counted that light, so the
solid angle the emitter subtends loses its share of the far-field fill. In effect the
luminaire occludes the ambient sky, which is what you want, but it does mean `-gi` and
`-ambient` are not bit-identical in an open scene with a visible lamp.

> **Watch the auto-exposure when judging `-ambient`.** By default the tone map anchors on
> the image's own 99th percentile, so raising `-ambient` raises the mean radiance and the
> anchor immediately divides it back out — the frame does not get brighter so much as
> **flatter**. Sweeping `-ambient 0 → 0.30` on a closed box moved the anchor by **5×**
> (3.57e-13 → 7.05e-14). If you are comparing fills, or matching a preview against a
> converged render, put the scene in **absolute mode** first (author `lumens`/`power` on a
> light — see *Absolute power*) so the gain is fixed and a brightness change is a real
> brightness change.

**On the GPU.** Since v0.110.0 mode `W` also runs on the **backward megakernel** (`-device
gpu`/`auto`), with the deterministic estimators ported rather than approximated — the light
grid, the throughput cutoff, the dominant branches and all three radical-inverse lattices
(subpixel / wavelength / glossy lobe) are the same quadratures the CPU uses, indexed by the
same absolute sample index. On the A/B test bed (`scraps/n3_gpu.ftsl`, 800×520, `-spp 16`,
absolute exposure) **99.4 % of channel samples are bit-identical** between `-device cpu` and
`-device gpu` and 99.96 % are within one 8-bit code; the residual is confined to
**single-pixel slivers on silhouettes and shadow edges** (zero pixels sit inside a ≥3 px-wide
disagreeing region), which is what the device's fp32 `Real` and its coarser `RAY_EPS` cost.
It is not bit-exact and cannot be — but it is not a different *image*. The same render is
**12.1 s → 0.3 s** (≈40×) on a 4090 versus 12 CPU threads.

Since v0.116.0 **the whole mode is on the device** — nothing mode-`W`-specific falls back to
the CPU any more. `-gi` (the deterministic one-bounce gather) was the last holdout and landed
in 0.116.0; the device runs the gather as a compile-time-bounded recursion, and it reproduces
the CPU image to the same fp32 tolerance as the rest of the mode (on a Cornell box at
`-gi 32`, 98.7 % of samples bit-identical, every 20 px block agreeing in luma *and* chroma to
under 0.4 of an 8-bit code). Diffuse, diffuse-transmit, mirror, glossy, filter, material
mixes, the full **split-at-dispersion** walk over glass / thin-film / multilayer / grating /
half-mirror / fluorescent (described below), `-gi`, `-ambient`,
area/sphere/cylinder/spot/sun/env lights and the physical lens all run on the GPU. (Mode `W`
can still fall back for reasons that are not mode-`W`-specific — a `layered` material, say,
gates the whole backward megakernel — and a `[device] … using CPU` line says so when it
happens.) `-rgb` is ignored in mode `W`: the fast RGB kernel is a separate reduced tracer with
no deterministic estimator, so it would hand back exactly the noise mode `W` exists to remove.

**Honest limits.** Mode `W` is a *preview*, not a reference: it is biased. **Rough glossy metal is the one
thing that wants `-spp` > 1**: at 1 spp the lobe is its single mirror direction, so a satin
metal previews crisper than it renders. It is not stuck there — the lobe direction comes off
a deterministic lattice indexed by the sample index, so extra passes resolve it (on gold at
roughness 0.35, mean error falls **19×** from 1 spp to 256 spp; before v0.109.0 it fell 6 %,
because every sample re-traced the *identical* direction and no budget could fix it). A
**`grating` wants `-spp` > 1 for the same reason**: at 1 spp it takes the specular order
`m = 0`, so it previews as a plain mirror with no rainbow, and extra passes fan the spectrum
out into the higher orders. A **`fluorescent` dye is free at 1 spp** since v0.115.0: its single
excitation wavelength is the median of that dye's *own* excitation distribution — absorption band ×
illuminant — rather than of the illuminant alone, so a dye that only absorbs blue is excited by the
one canonical sample instead of being sampled past its own absorption edge. (Under a 6500 K lamp a
`shortpass edge=480` dye lands at 422 nm, where its `aEff` is 0.83.) The same per-material
excitation CDF is a variance reduction in the stochastic modes, where draws used to be thrown at
the whole illuminant and mostly wasted; `-checkfluoro` estimates the reradiation weight both ways
and asserts they agree, so it is a pure importance-sampling change, not a re-tuning.
**A `-gi` gather over a *caustic* wants `-spp` > 1** — put a glass ball in a box and the floor
around it picks up thin, bright, dashed contour curves at 1–4 spp. They are real light: a
gather ray that refracts through the ball and lands on the lamp, which is a caustic path next-event
estimation structurally cannot sample (the lamp is behind a refracting surface), so the only
estimator for it is the emitter hit itself. They read as *curves* rather than as grain because
sharing one direction lattice across every pixel is the whole point of the mode: "does gather
direction #k reach the lamp through the ball?" flips at one coherent contour in the image instead
of dissolving into per-pixel noise. Three levers, cheapest last: they integrate away —
invisible by `-spp 64`; `-gi-bounce 1` removes them outright at 1 spp by denying a gather ray the
second bounce a caustic needs, keeping the colour bleed and losing only the caustic; or
**`-gi-clamp 0.1`** caps one gather ray's returned radiance and keeps the caustic as a *soft*
highlight for free. The clamp is the usual answer — measured on the repro scene at `-gi 32 -spp 1`
it costs **0.31 % of frame luminance** and, because it is applied per wavelength rather than per
bundle, cannot make the hero and single-λ paths disagree. Keep it above `-ambient`, though: an
escaping gather ray returns the flat `-ambient` far-field fill and the clamp caps that too, so the
gather's fill is effectively `min(-ambient, x)` and a smaller `x` simply darkens the whole scene.
A half-mirror or layered coat picks its dominant branch instead of forking, and a
pattern-driven material mix hard-thresholds instead of dithering. **Glass is free at 1 spp** — mode `W` always
**splits the hero bundle at a dispersive vertex** (see `-herosplit`), fanning it into one
monochromatic sub-path per wavelength so each λ refracts along its own direction and lands
in its own slot. It has to: the alternative policy (terminate the secondaries, boost the
hero) is fine for a stochastic mode, but mode `W`'s wavelength lattice is a function of the
sample index alone — that is what makes it noise-free — so at `-spp 1` *every pixel* would
collapse onto the *same* wavelength and the whole glass object would come out strongly
mistinted (a Cornell SF10 ball used to render flat green: **36.7 pp** of chroma error).
Splitting brings that to **0.80 pp** at 1 spp, better than 16 stochastic passes managed
(4.20 pp) and **7.9× faster**, and costs nothing on scenes without dispersive glass.
**A `layered` clearcoat is free at 1 spp too** since v0.115.1: the coat's reflectance is applied as
a per-λ *weight* with the bundle intact (it changes neither the direction nor the wavelength, so
there is nothing to collapse), and the bundle fans out into monochromatic sub-paths only where the
reflect-or-enter decision genuinely differs across λ — a high-contrast iridescent film, or a Fresnel
coat right at the dominant-branch threshold. Before that fix every coated surface de-hero'd
unconditionally and rendered *saturated green* at 1 spp; `scenes/layered.ftsl`'s chroma error against
a converged reference fell **17×** (190 → 11 codes), and keeping all eight channels alive instead of
boosting one ×8 also made the 64-spp image ~7× closer to that reference. `-gi` is one bounce only, terminated on
the `-ambient` tail — it is not a substitute for a converged render. All of these are
tracked in `known-issues.md`.

**A light sealed inside glass renders the scene black — and mode `W` now says so.** Mode
`W` lights a surface *only* by next-event estimation, and a shadow ray is blocked by any
geometry at all, dielectrics very much included (the SDS limitation: you cannot connect
through a refracting interface). So a lamp modelled the way a real one is built — an arc
sealed in a quartz envelope, a filament inside a closed reflector — can reach no vertex
anywhere in the scene, and the whole image comes out **pure black**. Nothing is wrong with
the scene; it needs a transport that can refract back *out* of the enclosure, which is why
`scenes/gallery_settled.ftsl` and `scenes/mirror_sphere_interior.ftsl` select mode `D`.
That used to fail silently — the only trace was `auto-exposure=1`, the "no signal at all to
scale" fallback, which reads like a normal number, and under `-explore` the window simply
went black the instant the camera settled and the mode-`W` stage took over. Since v0.119.0
ftrace probes every emitter at startup (a few hundred rays, free next to any render) and
reports the fraction of its outgoing directions that a specular surface blocks, naming the
blocking mesh:

```
[mode W] WARNING: light 1 of 1 is SEALED inside dielectric geometry (mesh 'lamp_xe')
                 -- 98.2% of the directions leaving it are blocked
```

The reported number is the share of the light's emitted power that no NEE connection can
ever collect. It warns past **95 %** rather than at a literal 100 % because a real lamp
assembly has hardware *inside* the envelope — the gallery's arc probes at 98.2 %, the
missing 1.8 % being its own socket and cord, which are diffuse but light nothing except
themselves. The check runs under `-explore` too, since the viewer's `T` preview *is* mode
`W`. Across all 98 scenes in `scenes/` exactly the three lamp-enclosure scenes trip it. The
workaround it suggests, `-ambient 0.15`, gives a flat-lit preview that is perfectly good
for navigating and framing.

Where extra `-spp` *does* buy convergence (the glossy lobe, the grating's orders, the dye's
excitation band), it now does so **from the second sample onward**. Each of those lattices uses its
own prime base so that two vertices on one path aren't driven by the same sequence, and those
bases (13…73) are larger than any sane preview budget — which used to mean the sequence only
explored a `spp / base` sliver of its range and the effect arrived in a lump once `-spp` passed
the base. Since v0.114.0 the sequences are **digit-scrambled** (Faure's fix for high-dimensional
Halton), so *N* samples spread over the whole range for any *N*, while sample 0 still lands on
exactly the same canonical value as before — so every `-spp 1` image is unchanged.
When you want the truth, that's what `R`/`D`/`U` are for.

```sh
# fast look preview
ftrace -in scenes/cornell.ftsl -mode W -spp 1 -ambient 0.05 -window -keepwindow -o png/preview.png
# ... with real bounce light (occlusion + colour bleeding), still deterministic
ftrace -in scenes/cornell.ftsl -mode W -spp 1 -ambient 0.05 -gi 32 -window -keepwindow -o png/preview_gi.png
```

> **Quick preview — `-raster` (not a transport mode).** To eyeball *composition*
> and *camera motion* before committing to a full render, `-raster` skips light
> transport entirely: it tessellates the whole scene once (analytic spheres →
> UV spheres, isosurfaces/CSG → marching-tetrahedra mesh, `curve` strands → a
> round-cone mesh per segment, instanced meshes baked
> to world space) and z-buffers each camera as solid, flat diffuse+headlight
> triangles — roughly **1 fps at 1280×720**. There is **no** transparency,
> reflection, refraction, shadow, caustic or GI: a dielectric shows as a solid
> ghost and a mirror as a flat tint. (Opt in to **see-through clear objects** with
> `-see-through` — see below — which drops the ghost for a dim + milky-haze pass
> that still refracts nothing.) But everything that gives a surface its look **at a
> single point** *is* shown, on both the CPU and GPU preview alike:
>
> * **Image skins** — a material whose albedo is a bound texture
>   (`reflect texture:<name>`) is sampled per pixel, so a skinned
>   globe/wallpaper/torus reads with its actual image rather than a flat colour. The
>   UVs come from the surface's per-vertex coords; from the material's world
>   **triplanar** projection for an un-UV'd mesh; or, for a marched
>   isosurface/CSG, from the primitive's own `uv planar|spherical|cylindrical`
>   projection (marching cubes produces no per-vertex UVs, so without this a skinned
>   implicit would preview bare).
> * **Palette (indexed-spectral) maps** — previewed as the colours their palette
>   spectra actually reflect, not as the raw index stored in the image.
> * **Procedural `pattern` drives** — `reflect pattern:` / `reflect_map pattern:`
>   modulate the albedo, and `emit pattern:` / `emit_map pattern:` modulate the
>   **emission**, both evaluated per pixel by the same expression VM the real
>   renderer uses. So a masked emitter (a glowing grid on a dark floor) previews as
>   the grid, not as one flat glowing slab.
> * **Normal maps** — `normal_map` perturbs the shading normal through the
>   triangle's UV-derived tangent frame, so surface relief shows.
> * **`mix` / layered materials** — previewed as their dominant child (the same
>   choice the deterministic `-mode W` viewer makes) rather than the parent's colour.
> * **`mix` blend masks** — a two-child `mix` carrying a `weight_map texture:` /
>   `weight_map pattern:` is instead resolved **per pixel**, hard-thresholded at ½
>   exactly as `-mode W` does, and the winning child's *whole* payload (albedo, skin,
>   normal map, pattern drives) is swapped in. So a wear mask, decal or painted A/B
>   blend previews as the spatial pattern it is, not as one flat winner.
>
> Roughness and film-thickness maps are deliberately *not* previewed: the preview
> has no glossy lobe for them to drive. Shading sums a diffuse term from **every**
> scene light using its real position/direction (spot cones included), so multi-
> light rooms read with their true key directions. It reuses the **same camera
> projection** as the real renderer, so the pinhole's off-axis stretch (spheres
> elongating toward the frame edge) and the fisheye/panoramic lenses reproduce
> faithfully. It **emulates the same auto-exposure as the real render**: the raw
> shaded image is anchored by a 99th-percentile tone map (lit surfaces → ~0.9,
> emitters clip to white) exactly like `filmToRgb8`, then each camera's
> **photographic exposure** (film `iso`/`shutter`/`exposure` compensation) is
> applied on top as exact stops — so an ISO 200 camera previews one stop brighter
> than ISO 100, and the composition sits at the brightness it will render at
> instead of an arbitrary fixed level. Because that p99 anchor divides out any
> uniform scale, **aperture** is (correctly) invisible in the default pipeline;
> it feeds preview brightness only where the real renderer keeps it — an
> *absolute-EV* scene (a light with `power`/`lumens`), where a wider pupil is
> genuinely brighter (∝ 1/N²) and the auto-exposure is bypassed. This holds in the
> finite-lens catch modes (A/C), where the pupil area rides in the splat weight, and
> now in the pinhole splat (**mode B**) too: when an `fstop`/`lens` is authored, an
> absolute mode-B render applies the camera-equation light-gathering term `(π/4)/N²`
> as a pure exposure factor (f/2 renders exactly four stops brighter than f/8) while
> keeping the pinhole's zero depth of field. A `camera_curve`/`camera_path` with `exposure_lock` shares
> one anchor across all its frames, so a preview flyby doesn't flicker
> frame-to-frame just as the final render won't. It honours the `-camera`
> selection and the `-window` live view, and a `camera_curve` flyby animates
> through every frame in the window. On a heavy scene the live window **pops up
> immediately** — before tessellation finishes — showing a dark placeholder and a
> **`tessellating (N/M, P%)`** progress readout in the title bar (and matching
> `[raster] tessellating implicit N/M` lines on stdout) as each isosurface/CSG
> implicit is marched, so you're never left staring at a blank screen wondering
> whether it hung. Control the
> isosurface mesh fineness with `-raster-iso <n>` (default 96 cells along the
> longest axis; `0` skips implicit surfaces). Example:
> `ftrace -in scenes/gallery_settled.ftsl -raster -window -o png/preview.png`.
>
> **Fur / curve strands are previewed under a triangle budget —
> `-raster-curve-budget <n>` (default 12000000).** Every curve segment is swept into a
> round-cone mesh so a furred subject doesn't preview bald, but a groomed pelt is
> *millions* of segments: `scenes/fur_creature.ftsl` and the creature in
> `scenes/gallery_rain.ftsl` carry ~1.79 M each, and at the full 80-triangles-per-segment
> cone that would be tens of gigabytes of preview geometry. So the sweep picks the
> coarsest tube that fits the budget — 10-sided capped → 6/4/3-sided → capless →
> a flat double-sided ribbon — and only if even the cheapest one busts the budget does it
> thin whole **strands** (kept strands stay continuous, never dashed). At the default
> those two pelts land on a 3-sided capless tube, ~6 triangles/segment, which is
> visually indistinguishable from the full cone at preview resolution. When the budget
> bites, ftrace says so:
> `[raster] 1786496 curve segments over the 12000000-triangle preview budget: 3-sided
> uncapped tube, 6 tris/segment (10718976 tris)`. Raise `-raster-curve-budget` for
> rounder strands if you have the RAM (each preview triangle is ~320 B), lower it on a
> small machine. This only affects the **preview**; the ray-traced modes intersect the
> analytic strands exactly and ignore the budget entirely.
>
> **GPU-accelerated preview — `-device gpu` (or `auto`).** When ftrace is built
> with CUDA, the preview rasterizer runs on the GPU: the tessellated world
> triangles are uploaded **once** and every camera is projected, depth-resolved
> (a 64-bit `atomicMax` visibility buffer) and shaded on the device, then the
> **same** 99th-percentile auto-exposure + sRGB tone map as the CPU path runs on
> the host — so a GPU frame matches a CPU frame (only sub-pixel float-vs-double
> differences at silhouette edges) and an exposure-locked flyby still shares one
> anchor with no flicker. It pays off on **heavy** scenes: a 4.5 M-triangle
> isosurface at 1600 px rasterizes in ~0.30 s/frame on the GPU vs ~1.6 s on the
> CPU (~5×), with the one-time tessellation unchanged; tiny scenes are launch-bound
> and roughly tie. **Scope:** the GPU path covers **all camera projections**
> (rectilinear **and** fisheye/panoramic — the device applies the same angular lens
> map the real camera uses), **opaque** geometry, **and** `-see-through` clear-glass
> compositing (a device clear-accumulation pass mirrors the CPU one). Its shading has
> **full parity with the CPU preview** — image skins, palette maps, normal maps and
> procedural `pattern` drives on albedo and emission all run on-device, the last through
> the very same expression VM the GPU path tracer uses. Only a device allocation failure
> falls back to the CPU rasterizer per camera (mixed camera lists just work), so
> `-device gpu` never fails a
> preview it can't accelerate. Example:
> `ftrace -in scenes/gallery_settled.ftsl -raster -device gpu -window -o png/preview.png`.
>
> **No-tessellation GPU isosurface preview — `-raster-gpu`.** The `-device gpu`
> preview above still *tessellates* the world (marching cubes) first, then rasterizes
> the triangles — so isosurface-heavy scenes pay a growing CPU tessellation cost every
> frame. `-raster-gpu` skips tessellation entirely: it casts **one primary ray per
> pixel** on the device and finds the nearest surface with the shared `closestHit`,
> which **sphere-traces implicit isosurfaces directly** (no mesh). It shades with the
> same preview model (per-material albedo — or a sampled **image/procedural skin**, at
> the hit UV or by world triplanar — plus ambient + weighted N·L keys +
> a headlight fill) and runs the **same** shared auto-exposure + sRGB tone map on the
> host, so the image matches `-raster` (surfaces are actually *cleaner* — no marching-
> cubes faceting) and an exposure-locked flyby still shares one anchor. It falls back
> to the CPU rasterizer automatically when the GPU can't handle the config (no CUDA
> device, `-see-through`/`-glass-clarity`, or a physical mesh-lens camera). Ideal for
> morphing-isosurface video (the `gyroid_nd` loom example routes frames through it with
> `--raster-gpu`). Example:
> `ftrace -in scenes/implicit.ftsl -raster-gpu -window -o png/preview.png`.
>
> **See-through clear objects — `-see-through`.** By default a clear material
> (dielectric / thin-film / filter / diffuse-transmit) previews as a solid pale
> ghost. Pass **`-see-through`** (aliases `-seethrough`, `-glass`) to instead render
> those surfaces as actually transparent — *without* refraction. Each clear surface
> between the camera and the opaque background **dims** what's behind it by a
> per-surface transmittance and adds a little **milky haze**, and both effects
> **accumulate with the number of clear surfaces crossed** (a closed glass ball =
> two crossings, front + back), so thicker/stacked glass reads progressively darker
> and hazier. A grazing-angle (Fresnel-like) term thickens the haze at silhouettes
> so glass edges still read. It's **order-independent** (the transmittance is a
> commutative product), so overlapping transparent objects need no depth sort and
> the pass stays nearly free. Tune the per-surface transmittance with
> **`-glass-clarity <0..1>`** (default `0.85`; higher = clearer/less dimming, and
> passing it implies `-see-through`). This is a *look* preview only — there's still
> no bending, reflection or coloured absorption. Example:
> `ftrace -in scenes/cornell.ftsl -raster -see-through -window -o png/preview.png`.
> Because rasterizing is nearly free, a preview whose size you haven't pinned with
> `-r` is **upscaled so its long edge is at least 1440 px** (aspect preserved) —
> a scene that authored a small `film { res 256 256 }` still previews big and
> readable instead of a postage stamp; already-large cameras are left as-is, and a
> real light-transport render always keeps its authored resolution.
>
> **Interactive camera (raster + live window).** When a **single** camera is
> rasterized into a `-window` (a still preview, including the double-click default —
> a `camera_curve` flyby instead animates through its frames), the window becomes an
> interactive **fly-camera**: move the mouse over the window to look around, then fly
> around and read off the numbers to author a `.ftsl` camera. There is a single unified view
> — you always **travel where you look** (or the exact opposite when reversing), so
> there is no separate "aim the target" mode and no crosshair. The world up is fixed,
> so there is no roll. (To drop straight into this viewer **seeded at the first frame
> of a multi-frame flyby** — instead of animating through every frame — add
> `-explore` / `-fly`; the flyby's frames become a **camera-path timeline** you can
> scrub, play and lock onto from the control panel below the image. See the flags table.)
>
> | input | does |
> |---|---|
> | **move the mouse over the window** | **steer** (joystick/rate look) — the cursor's offset from the window centre sets a **turn rate**: rest it near the centre (a neutral dead zone) and the view holds still so you can look at the scene; push it toward an edge and the view keeps turning that way (left/right = yaw, up/down = pitch, clamped just shy of straight up/down) for as long as you hold it there, so you can look a full circle. Where you look is where you fly. The pointer stays **visible** and free; steering only happens while the cursor is inside the window and stops the moment it leaves. |
> | **`Space`** or **`+`** (held) | **fly forward** continuously along the view direction — one fixed **step per rendered frame** (see note below) |
> | **`Shift`** or **`-`** (held) | **fly backward** — the exact opposite of where you're looking |
> | **mouse wheel** | **dolly** forward (up) / back (down) — a discrete, fully-rendered move per notch (can't overshoot into geometry). Each notch travels several fly-steps, so it's a quick reposition; the held-key travel is the fine cruise |
> | **`Ctrl` + mouse wheel** | change the **step size**: up = bigger steps, down = smaller (held-key step starts at 2 % of the scene radius, clamped to a sane band; the wheel dolly scales with it) |
> | `C` | cycle **wall collision**: `slide` → `stop` → `noclip` (see note below) |
> | `0` (or `Home`) | reset to the authored camera |
> | `P` | print a paste-ready `camera "cam" { eye … look_at … up … fov_y … }` block |
> | **resize the window** | change the preview resolution: the raster renders at the window's **actual pixel dimensions**, so the picture **fills the window (no letterbox bars)** and **shrinking it renders fewer pixels (faster on a heavy scene) while growing it renders more (crisper)**, up to the authored longest edge. The horizontal field of view widens/narrows with the window (like a game viewport — `fov_y` stays fixed, pixels stay square), so a wider window simply reveals more to the sides |
>
> **Motion is feedback-locked, not wall-clock-based.** Each held-key frame (and each
> wheel notch) moves the eye exactly one fixed `step`, and *one frame is rendered per
> move* — so travel rate automatically scales with render speed: a heavy scene dollies
> in a careful crawl, a light one moves briskly, and because every position you pass
> through is actually drawn you can **never skip through a wall into the void between two
> frames you didn't see**. Adjust the per-move distance live with `Ctrl`+wheel.
>
> **Wall collision keeps the camera out of solid geometry** (on by default). Each move is
> cast against the scene (the engine's own BVH), so you can't fly through a wall. `C`
> cycles the response: **`slide`** (the default) stops at the wall but lets the leftover
> motion slide along it, so holding forward against a wall carries you around a corner
> into open space; **`stop`** halts dead at the wall (no sideways drift); **`noclip`**
> turns collision off entirely, to place a camera *outside* the room or *inside* glass.
> Start with collision off via **`-noclip`** (`showcase_flyby.py --noclip`).
>
> **Control panel (below the image).** The live window reserves a strip under the
> preview for on-screen controls, so you don't have to remember key bindings. Two
> buttons are always present: **Clip** (cycles the same `slide` → `stop` → `noclip`
> collision modes as `C`, showing the current mode) and **Reset** (a dependable escape:
> **releases any path-lock and returns to the authored camera in free flight**, so
> mouse-look always works again afterwards — handy if you accidentally locked onto the
> path by clicking the timeline).
> When you entered via `-explore` / `-fly` on a **multi-frame flyby**, the panel also
> gains the flyby's **camera-path timeline** and its controls:
>
> | control | does |
> |---|---|
> | **timeline slider** | **scrub / jump** to any camera on the path — dragging or clicking snaps the view to that frame's exact eye, orientation, up and fov, and **locks onto the path** (pausing playback) |
> | **Play / Pause** | auto-advance along the path (engages path-lock); it **stops at the end** of the timeline |
> | **Path** (toggle) | **lock to / release** the path: while locked, forward/back travel along the timeline and the view uses each frame's authored orientation/up/fov (mouse-look and free translation are suspended); release to fly freely again from wherever you are |
> | **cams/upd** | **stride** traversal speed: cameras advanced per **rendered frame** (feedback-locked, like the fly motion) |
> | **cams/s** | **rate** traversal speed: cameras per **wall-clock second** (may skip frames on a slow render to keep real-time pace); defaults to the scene's authored fps |
> | **per upd / per sec** switch | choose which of the two speeds above is in effect (they're mutually exclusive) |
>
> While locked to the path, `Space`/`+` and `Shift`/`-` move **forward/backward along
> the timeline** (instead of through free space) at the selected speed, the mouse wheel
> **nudges one camera per notch**, and the slider tracks your position live. Toggle
> **Path** off (or press Reset) to return to free flight.
>
> **Camera-curve editor (author a flyby by flying it).** An always-present editor row
> lets you build a real [`camera_curve`](#camera-animation-camera_path-camera_orbit) right in the
> viewer — fly the shot you want, then Save it. It works from a lone camera or on top of
> an existing flyby:
>
> | control | does |
> |---|---|
> | **Rec** | start/stop **recording** your free flight; while armed it samples the pose as you move, then (on Stop) turns those samples into control points — either every sample (**raw**) or a tolerance-simplified subset |
> | **+Pt** | append the **current pose** (eye + look direction + up + fov) as a control point |
> | **Ins** | **insert** a control point at the current scrub position (splits that segment) |
> | **Del** | delete the **selected** control point — the one highlighted red in the overlay (see below) |
> | **Save** | write the authored `camera_curve { … }` block to a file next to the scene (`<scene>_curve.ftsl`, non-clobbering) **and echo it to stdout** to paste into a scene |
> | **raw** (checkbox) | keep **every** recorded sample instead of simplifying |
> | **tol** | recording **simplify tolerance** in world units (Ramer–Douglas–Peucker on the eye path; `0` = keep raw) |
>
> As you author, a **live spline overlay** is drawn on the preview: the control points as
> yellow markers — with the **selected** one highlighted **red** (the Del target) — and the
> interpolated path as a green polyline, sampled with the **same centripetal Catmull-Rom**
> math the renderer uses for `camera_curve`, so the preview is WYSIWYG. **Selecting a point:**
> when you're locked to the path, the selection follows the timeline — the control point
> nearest the current scrub position — so you just **scrub to a point to select it** (then
> Del removes it); in free flight the selection is the point nearest the eye. The saved block
> records each control point as a `point` plus a `look curve` (a second spline of `look_point`
> targets, each placed one mean control-point spacing ahead along the view ray so the aim
> spline stays smooth) so the camera's orientation is authored too, and carries the current
> `up`, `fov_y`, render `mode`, `frames`, and scene `fps`.
>
> **Painting speed and orientation (Paint mode).** Two more controls sit at the right end
> of the timeline row: a **Paint** toggle and a **Flat** button, with a live speed readout.
> With **Paint** on and the view locked to the path, *fly the timeline* (Play or the
> throttle keys) while:
>
> - **rolling the mouse wheel** to paint the **local traversal speed** at the current point
>   — an *additive brush* (wheel up = faster there, down = slower), clamped, so you can
>   play a pass, speed up the boring stretches and slow down the money shot, then play
>   again to refine. Speed is the inverse of camera density, so it's exported as a
>   `density_at` track and **both** the live playback pace **and** the rendered flyby's
>   frame spacing follow it. The readout shows the multiplier (e.g. `1.35x`); **Flat**
>   resets the whole speed track to a uniform pace.
> - **moving the mouse** to **steer the orientation** at the current point — the nearest
>   control points' look directions bend toward where you aim, reshaping the `look curve`
>   live (WYSIWYG in the overlay and in the saved block).
>
> With Paint **off**, the wheel nudges one camera per notch and mouse-look is suspended
> (the normal path-lock behaviour).
>
> **Editing an existing curve in place (round-trip).** When you open a scene that already
> contains a `camera_curve` with `-explore` / `-fly`, the editor **seeds itself from that
> curve's control points** — each `point` becomes an editor control point (with its look
> direction taken from the curve's `look curve` / `look_at` / tangent, and its local speed
> from the `density` track). The control-point markers appear in the overlay immediately,
> so you can Del/Ins/steer/re-paint speed and Save a revised curve rather than starting
> from an empty editor. The loaded flyby still plays at full fidelity until you make the
> first edit. When a scene defines **several** `camera_curve`s, the editor seeds from the
> one you're actually flying (chosen with `-camera <name>`), not blindly the first.
>
> **Editing a loom animation drive (`-anim <file.json>`).** The same editor can reshape a
> loom **`CurveDrive`** — an N-dimensional curve whose dimensions ("channels") drive
> arbitrary *scene variables*, not just the camera. Pass `-anim <sidecar.json>` (it implies
> `-explore`) and the editor's control points become the drive's points: channels 0–2 are
> the point you see and move in 3-D (for a camera drive that *is* the eye position), while
> channels 3 and up are values no viewport can show, so they ride along with each point.
> **Save** then writes the reshaped curve back to the sidecar — atomically, keeping the
> drive's name, mode, dimension count and every **channel → scene-variable binding**
> exactly as they were, so an editing pass never drops associations loom authored. Pointing
> `-anim` at a file that doesn't exist yet is how you *start* a drive: whatever control
> points the scene seeded become a new 3-channel drive that the first Save creates. The
> sidecar is plain, human-diffable JSON that loom reads and writes with `loom.anim`
> (`python -m loom.anim <scene.py> --config <sidecar.json>`), so either side can author it.
>
> **Watching the drive actually drive the scene (`-anim … -loom <scene.py>`).** Add `-loom
> <scene.py>` and the editor stops previewing just a camera path: it starts a live loom
> session on that scene and, every time you scrub, asks loom for the scene **as of that
> point on the curve**. The bound scene variables move in the viewport — a radius breathes,
> a light dims, a material shifts — even if the camera never budges. ftrace deliberately
> does *not* sample the curve itself; it sends loom the control points and asks by
> parameter, so what you see while editing is what loom will render for the video. Scrubbing
> fast is safe: only the newest position is kept (stale ones are dropped, never queued),
> while edits to the points/bindings/channel count are queued losslessly ahead of it.
>
> With the live session up, the panel grows a **loom bind row**:
>
> | Control | What it does |
> |---|---|
> | `ch N` | Which drive channel you're inspecting or editing. |
> | slot list | The scene variables loom reports as bindable (`loom.anim`'s `slots`). Pick-only — no typing — plus `(none)`. |
> | **Bind** | Binds the chosen channel to the chosen slot (or to nothing, with `(none)`). |
> | **Unbind** | Drops the chosen channel's binding. |
> | `chans:` | The drive's channel count. Growing it widens every control point; shrinking it drops the bindings on the channels that no longer exist — the same thing loom does — rather than leaving a stale association behind. |
> | status | What the current channel does *right now*, plus link health: `ch 0 → ball_r  \|  live — 4 baked, 2 ms`. |
>
> Every edit takes effect immediately in the preview, and **Save** writes the channel count
> and the bindings back to the sidecar along with the points.
>
> **Reviewing a rendered flyby (`-review <base>`).** Once a flyby has actually been
> *rendered* to a directory of images, `ftrace -review <base>` plays that sequence back
> on the same live window + timeline — so you can watch the real rendered result (not the
> raster preview), scrub/Play it, and **re-time** it. `<base>` is a filename stem with an
> optional path; frames are the files named `<base><digits>.<ext>` (ftrace appends a
> zero-padded index), so `-review png/swoop/swoop` matches `swoop000.png`, `swoop001.png`,
> … (numeric-sorted). It reads `.png` / `.jpg` / `.bmp` / `.tga` and ftrace's own `.ppm`
> output. No scene is loaded — it's a pure playback utility. With **Paint** on, the wheel
> paints local speed exactly as in the editor (an additive brush, fast regions skimmed,
> slow regions dwelt on); **Flat** resets it. **Save** writes a re-paced copy of the
> sequence into `<dir>/retimed/` (each output frame is the source frame chosen by the
> painted speed profile) and prints an `ffmpeg` line to assemble it into a video. Close
> the window to finish.
>
> The controls are deliberately **keyboard-layout-independent** (`Space`/`Shift` and
> the `+`/`-` keys land in the same place on QWERTY, Dvorak, Colemak, etc.) — there
> are no letter-key bindings to relearn. The mouse pointer stays visible the whole
> time — nothing captures or hides the cursor. Steering is **rate-based**: the cursor
> acts like a joystick whose distance from the window centre sets how fast the view
> turns (centre = a dead zone that holds still so you can see the scene; toward an edge
> = keep turning that way), and moving the pointer off the window (to the title bar,
> another app, etc.) stops the turn entirely. The turn rate is **wall-clock-based**
> (radians per second, integrated by the frame time), so steering feels the same
> whether a scene raster-previews at 20 fps or 300 — a light model won't spin off-screen
> at the slightest cursor offset. (Free *translation* stays feedback-locked per frame so
> you can't fly through geometry between two frames you never saw.) The window title
> shows the live `eye(…) dir(…)` as you move. Frames re-rasterize at
> the live window's resolution — drag a corner to make the preview smaller (and
> snappier) or larger (and sharper); the aspect ratio and the readout are
> resolution-independent, so this only trades preview sharpness for speed while you
> navigate. Close the window to finish.
>
> **Double-click / bare invocation.** Running ftrace with just a scene file and
> nothing else — `ftrace scene.ftsl` (a positional path ending in `.ftsl`,
> `.scene`, or `.fts`, as produced by a file association or drag-and-drop) —
> defaults to exactly this quick preview: it turns on `-raster`, `-window`, **and**
> `-keepwindow` automatically and shows the room in a live window that **stays open
> after the raster finishes** (so a double-click preview doesn't flash-and-vanish —
> close the window yourself to exit), writing the preview PNG to a temp file (no
> stray output in the working directory). Passing any real-render
> control (`-mode`, `-n`, `-time`, `-noise`, `-forever`, `-device`, `-camera`,
> `-view`, an explicit `-o`/`-r`, etc.) opts out of the auto-preview and renders
> normally; `-in <path>` is likewise always an explicit render, never a preview.

### Speed / accuracy / ability tradeoffs

At a glance (the prose bullets below expand every row). The first row is the
`-raster` **preview** — not a light-transport mode, but included so you can weigh
"quick look" against a real render; everything below it is an unbiased estimator
that converges to the same physical image.

| Mode | Best for | Speed | Specular-first | Depth of field | Caustics | GPU | Main limitation |
|---|---|---|---|---|---|---|---|
| **`-raster`** *(preview)* | Composition & camera-motion preview | Instant | — *(flat ghost/tint)* | ✗ | ✗ | ✓ *(`-device gpu`; else threaded CPU)* | **No light transport at all** — no shadows, reflection, refraction or GI (`-see-through` fakes clear glass) |
| `B` *(default)* | Diffuse & caustic-heavy scenes | **Fastest** | ✗ *(black)* | ✗ | ✓ | ✓ | Can't shade a directly-seen mirror/glass; no depth of field |
| `A` | Efficient depth of field / bokeh | Fast | ✗ | ✓ | ✓ | ✓ | Rectilinear only; specular-first still black |
| `C` | Ground-truth DoF oracle | Slow | ✗ | ✓ | ✓ | ✓ | Catch-starved → far noisier than `A` for the same budget |
| `R` | Quiet reference; any first hit; **fluorescence** | Medium | ✓ | ✓ *(physical lens)* | ✗ *(noisy)* | ✓ | Noisy on caustics |
| `W` *(preview)* | **Noise-free look preview** — materials, shadows, reflections, at `-spp 1`; also the interactive viewer's lit preview (`-explore`, `T`) | ~300× `R` | ✓ | ✓ | ✗ | ✗ | Biased: GI is a flat `-ambient` fill or a one-bounce `-gi` gather, rough glossy needs `-spp` to resolve its lobe; fully on the GPU |
| `V` | Correctness check (`B` vs `R` residual) | ~2× *(runs both)* | ✓ *(via `R`)* | ✓ *(via `R`)* | ~ | forward pass | Diagnostic, not a production renderer |
| `P` | Mixed diffuse + mirrors/coatings | Medium | ✓ | ✓ *(routes to `D` w/ lens)* | ✓ | ✓ | Costs more than `B`; possible seam between layers |
| `D` | Specular-first + diffuse caustics + **participating media** in one pass | Slow / sample | ✓ | ✓ *(physical lens)* | ✓ | ✓ | Highest per-sample cost; no fluorescence / env / collimated lights |
| `M` | Many cameras sharing one lighting solution (flythroughs); reusable/persistable map | Fast per frame *(after one shared pass)* | ✓ *(walks to diffuse)* | — | ✓ | ✓ *(direct query)* | Direct query blurs contact shadows (use `-pmfg`) |
| `S` | **Caustics / SDS**; progressive, bounded memory | Slow *(many passes)* | ✓ | ✓ | ✓✓ | ✓ *(resident session; pinhole only)* | Many passes to converge |
| `U` | Robust "have it all" (diffuse GI + caustics), no per-scene mode picking | Heaviest / pass | ✓ | ✓ | ✓ | ✓ *(resident session; pinhole only, no media)* | Heaviest per-pass cost |

- **`B` — pinhole splat (default, fastest).** Every photon that hits a
  camera-visible surface splats to the pinhole, so essentially no photons are
  wasted — **orders of magnitude faster** than physically catching photons through
  an aperture. GPU-accelerated. *Cost:* a pinhole has no depth of field, and it
  **cannot render specular-first pixels** (a mirror/glass surface seen directly
  splats nothing and stays black — use `P`, `D`, or `R` for those). Best default
  for diffuse and caustic-heavy scenes. In an *absolute-EV* scene an authored
  `fstop`/`lens` still sets exposure here — the pinhole has no depth of field, but it
  applies the camera-equation light-gathering term `(π/4)/N²`, so f/2 is exactly four
  stops brighter than f/8, matching a real sensor (and A/C, once their gain is fixed).
- **`A` — finite-lens camera (efficient depth of field).** A physical finite
  aperture + thin lens + film, but imaged by **next-event splatting** each photon to
  the lens pupil (like `B`'s splat, through a real aperture instead of a pinhole).
  This gives **true thin-lens depth of field and bokeh** at a fraction of `C`'s cost,
  because photons don't have to physically hit the aperture. `B` is the `aperture→0`
  pinhole limit of this camera; rectilinear only (a fisheye needs a wide-angle element
  the single thin lens can't form). Use when you want DoF without `C`'s noise.
- **`C` — finite-aperture catch (brute-force DoF oracle, slow).** Photons must
  physically pass through the aperture to be counted, giving true thin-lens depth of
  field and bokeh — but it is **catch-starved** (most photons miss the aperture), so
  it is **much noisier / slower** than `A`/`B` for the same photon budget. Mainly the
  ground-truth `A` is validated against; use directly only when you want the
  unapproximated forward-catch.
- **`R` — backward reference (unbiased, general).** Traces from the camera, so it
  renders **any** first-hit surface including specular, and is the **quiet, reliable
  reference** for camera-visible lighting. It also renders **fluorescence** — the
  backward tracer is bispectral, sampling a separate excitation wavelength and
  reradiating with the material's emission colour (so `V` can validate the forward
  fluorescent tracer). GPU-accelerated (its own backward megakernel, which also drives
  the **physical multi-element lens** camera on the GPU). It gets **noisy on caustics**
  (light focused through glass/water is hard to find backward). *GPU scope:* the
  megakernel covers area/sphere/cylinder Lambertian **and point-spot** lights, all the
  specular/textured materials, **participating media** (homogeneous + heterogeneous),
  **fluorescence** and **both constant *and* image-based (lat-long HDR) environment
  lights** (env-NEE + MIS'd env-miss, at surface *and* fog vertices — the image env is
  importance-sampled on-device from its luminance CDF; **spectral rainbow-phase media**
  and **gradient-index (GRIN) media** — the same Eikonal marcher as the CPU — also run
  on-device now); scenes using collimated beams still fall back to the CPU tracer
  automatically. Add **`-rgb`** for a **fast RGB preview** (GPU only): instead of
  sampling one wavelength per sample it carries an RGB throughput triple and does one
  intersection walk per full-colour sample, so a clean colour image converges much
  faster. Materials bake to a per-material linear-RGB albedo and emitters/env to a
  linear-RGB radiance at scene build; achromatic specular uses a 550 nm representative
  wavelength. On spectrally-flat scenes it matches the spectral backward's absolute
  luminance to noise; colour scenes carry an Option-B approximation (no dispersion /
  thin-film / spectral fluorescence). `-rgb` falls back to the spectral backward (with a
  warning) on scenes outside its scope (media, image-env, textured/record albedo, exotic
  materials).
- **`V` — validate.** Runs `B` and `R` and reports their residual; a correctness
  check, not a production renderer (roughly twice the work).
- **`P` — composite (fills in what `B` misses).** Uses fast forward `B` for
  diffuse-first pixels and caustics, and a backward camera ray for
  specular/coated surfaces that `B` leaves black — a good "best of both" for scenes
  that mix diffuse lighting with mirrors/coatings. Both layers are GPU-accelerated
  (the forward layer via the `B` megakernel, the camera-side via `R`'s backward
  megakernel) when the scene is within the backward-GPU scope; otherwise the
  camera-side layer falls back to the CPU. *Cost:* more expensive than plain `B`;
  there can be a subtle seam between the two layers. With a **physical lens** the
  pinhole-splat forward pass can't form the lens image, so `P` automatically routes
  to the lens-aware BDPT (`D`) — or, if the scene is outside BDPT scope (env /
  collimated lights, fluorescence, layered materials, GRIN media), falls back to the
  backward realistic camera (`R`).
- **`D` — BDPT (most general, slowest per sample).** One unbiased estimator that
  traces a light *and* a camera subpath and MIS-combines every connection, so it
  captures **specular-first pixels and diffuse caustics in a single pass** on the
  absolute-radiance scale (no composite seam). GPU-accelerated (its own megakernel).
  It also supports the **physical (realistic) lens on its camera subpath** — the
  camera ray is traced through the real glass while forward light transport keeps its
  caustic efficiency (the light-image splat strategy is disabled, since a multi-element
  lens has no closed-form sensor projection; runs on the CPU **and GPU**). It renders
  **participating media of every kind** — global haze, multiple superposed media,
  box/sphere/object-bounded fog, and **heterogeneous `density`-field blobs** — with volume
  in-scatter vertices, HG-phase connections and transmittance-weighted edges (subpath
  medium vertices placed by delta tracking, connections weighted by ratio-tracking
  transmittance), so fog *inside a glass shell* images correctly here (a case the
  next-event modes leave dark). Since 0.124.0 it also renders **`spot` and `sun` lights** —
  delta emitters, so the strategies that would have to sample the Dirac (an eye path
  *landing* on the light) are dropped from the MIS weight, and a `sun` additionally gets an
  escaped-ray strategy so a mirror can throw its disc back at the lens. *Cost:* highest cost
  per sample; it **does not support fluorescence, layered materials, or env & collimated
  lights** (use `B`/`P` or `R` for those). Since 0.126.0 spot/sun scenes run on the **GPU**
  too — the device kernels do delta lights, so there is no CPU fallback for them any more.
- **`M` — photon map (view-independent, reusable).** Traces a forward photon pass
  **once** and stores every diffuse deposit in a **view-independent photon map** (a
  uniform hash grid), then forms the camera image by a backward camera pass. By default
  it uses a **direct density query**: each camera ray walks through specular surfaces
  until it lands on a diffuse one, where a radius density estimate over the nearby
  photons gives the radiance directly at that surface. Optionally, `-pmfg <K>` switches
  to a **Jensen final gather**: at the first diffuse hit it shoots `K` cosine-weighted
  hemisphere sub-rays, traces one bounce each, and queries the map at *those* points —
  so the density estimate's blur lives one bounce away instead of on the visible surface
  (direct light at the visible point is recovered by gather rays that strike an emitter
  directly). Because the map is independent of the camera, it can be **built once and
  reused across every frame of a flythrough** (or every camera of a multi-camera render)
  — the cost of the photon pass amortizes over all views. *Cost:* with the direct query
  the density estimate **blurs sharp contact shadows** at large gather radii (bias
  controlled by `-pmradius` / `-pmradiusfrac`); final gather (`-pmfg`) keeps those
  contact shadows and fine detail **sharp** while still smoothing indirect light, at
  roughly `K`× the per-sample cost (so pair it with fewer `-spp`). Directly-viewed
  emitters carry a little chromatic speckle at low spp. Best when many cameras share one
  lighting solution. **GPU-accelerated** for both the direct density query *and* the
  `-pmfg` Jensen final gather: the device deposits the photon pass, hands the hits to the
  same grid builder, then gathers every camera on the GPU from the one shared map — so a
  whole flythrough builds the map once and renders each frame in device time. Environment
  lights (constant **and** image-based lat-long HDR, M2) are handled on the GPU: the
  deposit emits env photons for the indirect bounces and the device gather adds env's
  direct term on gather-ray escape. The final gather (M4) runs its NEE direct term plus
  the `K` cosine-hemisphere sub-rays' one-bounce density queries entirely on the device.
  (Unsupported-material scenes and physical-lens cameras still fall back to the CPU.) The built map can also be
  **persisted to disk** with `-savemap <f>` and reloaded with `-loadmap <f>`: because it
  is view-independent, a reloaded map re-gathers new camera angles or a new gather radius
  **without re-tracing a single photon** (the expensive forward pass is skipped entirely).
  A scene-identity guard rejects a stale map built for a different scene, falling back to
  a fresh deposit. (Matches the forward splat modes `A`/`B`/`C` — same forward physics,
  just measured from a stored map.) The gather radius is **density-adaptive by default**:
  after the deposit ftrace measures how many photons a typical gather actually sees, then
  re-bins the grid at the radius that hits a target population — a target that grows only as
  the *cube root* of the stored photon count. Without this the radius came from the scene
  size alone, so photons-per-cell (and gather time) grew linearly with `-n`, and a high-`-n`
  render could look stalled when it was only grinding through a huge map. Measured on a
  Cornell-style test scene, cost per emitted photon used to **rise** with `-n`
  (54.9 → 67.3 µs/M going from 500k to 4M) and now **falls** (27.0 → 9.0 µs/M) — a 7.5×
  wall-clock win at `-n 4000000` (269 s → 36 s). Quality improves too, because the same wall clock now buys
  far more photons: scored against a converged BDPT reference at **matched render time**,
  RMSE drops ~20–24% overall and ~32–41% on flat wall areas where noise dominates. Controls:
  `-pmcount <k>` sets the target population at 1 M stored photons (default `200`, calibrated
  so ordinary renders keep the look they already had — raise for smoother/blurrier, lower
  for sharper/grainier); `-nopmauto` restores the old fixed-radius behaviour exactly
  (bit-identical), as does passing an explicit `-pmradius`. `-savemap`/`-loadmap` (currently
  honoured only on `-device gpu`) still lets the deposit be paid just once.
- **`S` — SPPM (progressive, caustic-strong).** Stochastic progressive photon mapping
  (Hachisuka 2008/2009): instead of one fixed-radius map, it runs **repeated bounded
  photon passes** and **shrinks each pixel's gather radius** over iterations, so the
  estimate is **unbiased in the limit** with **flat memory** (the map is rebuilt small
  each pass, never grown). Each pass also re-samples the camera subpaths, which
  anti-aliases and makes it robust for depth of field / glossy. Its stand-out strength is
  **caustics and SDS paths** — light focused through glass or off metal, which the
  backward tracer (`R`) and even BDPT (`D`) resolve slowly but a photon method captures
  directly. `-n` is photons **per pass**, `-spp` is the **number of passes** (or use a
  `-time`/`-noise`/`-forever` budget); the radius-shrink rate is `-sppmalpha` (default
  `0.7`) and the initial radius reuses `-pmradius`/`-pmradiusfrac`. A single pass reduces
  exactly to mode `M`. GPU-accelerated (`-device gpu`/`auto`): a resident device SPPM
  session keeps every pixel's progressive state (flux/radius/count + this pass's visible
  point) on the GPU across passes and reuses the mode-`M` deposit + an on-device
  gather/update — same scope as the GPU photon map (pinhole cameras); otherwise it runs on
  the CPU. *Cost:* many passes to converge; the running preview starts blurry (large
  radius) and sharpens as the radius shrinks.
- **`U` — VCM/UPS (the "have it all" estimator).** Vertex Connection and Merging
  (Georgiev et al. 2012, a.k.a. Unified Path Sampling): each pass traces a **light
  subpath and a camera subpath per pixel**, and combines **every** BDPT-style vertex
  **connection** (what `D` does — great for diffuse/glossy interreflection connected
  directly to the light) **and** every SPPM-style photon **merge** (what `S` does — great
  for caustics / SDS focusing) under **one multiple-importance-sampling (balance-
  heuristic) weight**. That single weighting makes it robust across the whole gamut: it
  matches the backward tracer on diffuse GI *and* resolves caustics like a photon method,
  with no per-scene mode picking. Like SPPM it is **progressive** and **unbiased in the
  limit**, shrinking the merge radius as `r_i = R0·i^((alpha-1)/2)` across passes. `-n` is
  **ignored** (light-path count follows the film resolution); `-spp` is the **number of
  passes** (or a `-time`/`-noise`/`-forever` budget); the radius-shrink rate is
  `-vcmalpha` (default `0.75`) and the initial radius reuses `-pmradius`/`-pmradiusfrac`.
  Since 0.125.0 it renders **`spot` and `sun` lights** on the same terms mode `D` does
  (delta emitters: the unsamplable strategies are dropped from the vc/vm MIS weights, and a
  `sun` gets the escaped-ray strategy so a mirror throws its disc back at the lens) — on the
  CPU, and since 0.127.0 on the **GPU** too (the device VCM session got the same port mode `D`
  got in 0.126.0, and is ~170× faster than the CPU on a spotlit Cornell box). *Cost:* the heaviest per-pass (both a full light pass and a full camera pass,
  plus a grid build), but the most consistent quality per pass — at equal time it beats
  SPPM on caustics *and* stays as clean as BDPT on diffuse GI. (Single-wavelength note:
  connections pair a camera path with its **own** light path so they share one wavelength
  and are exact; merges gather photons from other paths, so like modes `M`/`S` they use
  the standard spectral-photon-mapping XYZ estimate.)

The **image-forming modes are all progressive** — the forward camera models
(`A`/`B`/`C`), the backward reference (`R`), the bidirectional tracer (`D`), and the
composite (`P`) each refine an image whose brightness is fixed while only graininess
falls, so they share the same live progress and budget flags (`-time` / `-noise` /
`-forever` / `-preview` / `-interval`, and periodic crash-safe writes) on **both** the CPU
and the GPU. They're all GPU-eligible too: **`A`/`B`/`C` and the forward pass of `V`** via
the forward megakernel, **`D`** via its own GPU BDPT megakernel, **`R` (including the
physical-lens camera) and `W` (the deterministic preview, with the quadratures *and* the
split-at-dispersion walk ported — only `-gi` still falls back)** via the GPU backward megakernel
— which the **`P` composite
reuses for its camera-side layer**, so both of `P`'s layers run on the GPU when the scene
is within the backward-GPU scope — and the **`M` photon map** (direct density query
*and* `-pmfg` final gather), which builds one shared map on the device and gathers every camera from it. Outside that scope `P`'s camera-side layer, and `V`'s
backward reference (kept on the CPU as a stable ground truth), remain CPU-only. The
composite `P` classifies its pixels once, then alternates forward and backward batches
into two accumulating films, re-fitting the forward→backward scale and re-blending each
interval. **Disk `-resume`/`-checkpoint` now cover `A`/`B`/`C` (photon-count checkpoint),
`R`/`D` (spp-count checkpoint), and `P` (dual forward+backward film)** — a resumed render
continues the *absolute* sample sequence past whatever the checkpoint holds, so its added
samples genuinely reduce variance; in the deterministic mode `W` the continuation is exact,
and `-spp 3` followed by `-resume -spp 5` gives bit-for-bit the pixels of a plain `-spp 8`
(note that `-spp` under `-resume` means *additional* samples, not a total). Only the
persistent-state photon modes `M`/`S`/`U` (whose per-pass state a film alone can't restore)
stay non-resumable.

### Backends & performance (`-device`, `-wavefront`)

- **`-device auto` (default, recommended).** Uses the GPU when a supported CUDA
  device is present *and* the render is one it can handle (forward modes
  `A`/`B`/`C` on a non-fluorescent scene, mode `D`'s BDPT megakernel, mode `R`'s
  backward megakernel — including the physical-lens camera — both layers of the
  mode-`P` composite, or mode `M`'s shared photon map with the direct density query
  on a pinhole scene); otherwise the CPU. Prints its choice.
- **`-device gpu` / `cpu`.** Force the backend. The GPU **falls back to the CPU**
  for the mode-`P` camera-side layer and for `R`/`D` scenes outside their GPU scope
  (spot/sun/collimated lights; GRIN media in mode `D` BDPT — mode `R` now runs fog, spectral
  **rainbow-phase** media, **GRIN gradient-index bending**, fluorescence, and constant *and*
  image-based env lights on the device), and for
  fluorescent/oversized-mix forward scenes (mode `M`'s `-pmfg` final gather now runs
  on the GPU too). Mode `D`'s GPU BDPT megakernel renders
  **all** participating media — haze, superposed, bounded, and heterogeneous
  `density`-field fog — directly on the device. Implicit surfaces / `isosurface`, **procedural patterns**, and
  **dielectric translucency** (frosting + Beer–Lambert colored-glass tint) are all
  GPU-accelerated now — the device sphere-traces the same field expressions, runs the
  same pattern VM, and threads the interior-absorption medium through both the forward
  and backward tracers. GPU **BDPT** (mode `D`) now threads the **per-hit surface point**
  (texcoords stored on each path vertex, reconstructed into a `Hit` for the connection
  BSDF) so textured/patterned/record-driven diffuse albedo & glossy reflect, per-hit
  glossy roughness + thin-film maps, mix blend masks, Beer–Lambert **colored-glass**
  interior absorption, **diffuse-transmit** (two-sided Lambertian — both lobes +
  back-hemisphere connections), and **frosted (rough) glass** (stochastic-delta lobe jitter
  by per-hit roughness) all render **on-device** with MIS-consistent densities — the GPU
  BDPT scope now matches the CPU BDPT for every *material* (no per-material fallback).
  (Fluorescence, layered stacks, and env/collimated lights aren't a GPU limitation — BDPT
  can't render them on *any* backend, so mode `D` refuses or drops to mode `B` for those
  scenes on both CPU and GPU; use mode B/P/R for them. **Spot and sun lights** render
  on-device in mode `D` since 0.126.0 and in mode `U` since 0.127.0 — the delta-emitter light
  subpath, the per-shape NEE connection geometry and the escaped-ray solar disc are all in the
  kernels, so a spot/sun scene no longer falls back in either bidirectional mode. GRIN media
  likewise keep an in-scope
  mode-`D` scene on the CPU; spectral **rainbow-phase** media now render on-device in mode `D`.) **Parametric records** (a
  material's slots driven by a per-hit driver sampling a named LUT bank — see *Parametric
  records* below) run on the **GPU forward, backward, and BDPT (`D`) tracers for both the
  reflect/albedo and roughness slots** (constant stop selectors bake into the device
  material; per-hit driven reflect uploads the record's baked LUT + driver program, and a
  driven scalar/roughness slot uploads each stop's compiled expression + driver — both
  sampled on-device by exact twins of the CPU sampler; mode `D`'s connection BSDF
  reconstructs the per-hit point to sample the driver). The fallback is automatic. `cpu`
  is fully deterministic and is used for reference/validation baselines.
- **`-wavefront` vs. the default megakernel** (GPU forward renders only). Both run
  identical, exactly energy-conserving physics. The **megakernel** runs each
  photon's whole path in one thread and is usually fastest on **shallow, uniform
  scenes on a big GPU**. The **wavefront** splits the trace into coherent
  extend/shade passes over a persistent photon pool and wins on **divergent /
  deep-path scenes and smaller GPUs**. Their RNG streams differ, so their images
  match only to within Monte-Carlo noise.

---

## Cameras

Defined with a `camera "name" { … }` block (or the built-in scene camera).

**Basics:** `eye`, `look_at`, `up`, `fov_y`, `mode`, and a `film { res N M … }`
block. Film size can be a preset **format** — `full-frame`, `aps-c`,
`micro-four-thirds`, `super35`, `medium-format`, `6x6`, `6x7`, `large-format`,
`4x5`, `8x10` — or an explicit `size W H` in millimetres.

**Camera archetype presets** (`preset <name>`): one line that fills in a
physically-plausible **sensor size + focal length + f-number** for a real camera
*type*, exactly like `material { preset gold }`. It runs *before* the block's own
knobs, so any dial (`lens`, `fstop`, `film { size }`, …) written afterward
overrides it. A single preset serves both worlds — in the finite-lens catch modes
(`A`/`C`) the sensor + focal + f-stop give real depth of field, while in the
pinhole/backward modes (`R`/`B`/`U`) the same numbers set the correct field of view
and the aperture simply collapses to a point (no DOF). Available archetypes:

| `preset` | Sensor | Focal | f-stop | Character |
|---|---|---|---|---|
| `cinema` | Super35 (24.6×13.8 mm) | 35 mm | f/2.1 | Blackmagic-style cine; shallow, filmic |
| `pocket` | 1″ (13.2×8.8 mm) | 8.8 mm | f/4 | RX0-style compact; wide, deep DOF |
| `portable` | full-frame (36×24 mm) | 35 mm | f/1.8 | mirrorless with a bright prime |
| `vintage` | 35 mm film (36×24 mm) | 50 mm | f/3.5 | folding rangefinder normal |
| `vintage-slr` | 35 mm film (36×24 mm) | 50 mm | f/1.4 | classic fast fifty |

```ftsl
camera "cine" {
    preset cinema          # Super35, 35mm, T2.1 — DOF in mode A/C, right FOV in R/B/U
    eye 0 0.7 3   look_at 0 0.5 0   up 0 1 0
    focus 3
    # fstop 4              # ← would override the preset's f/2.1 if uncommented
    film { res 512 512 }
}
```

**Projections** (`projection …`): `rectilinear` (default perspective),
`equidistant` and `equisolid` fisheye, `stereographic` ("little planet"), and
`orthographic`. These are analytic remaps available in the forward pinhole mode.

**Analytic depth of field:** `aperture`, `focus`, `lens` (focal length, mm),
`fstop`, and `zoom` give a thin-lens camera with a real focus plane and bokeh.
This is the **fast, approximate** option: an ideal paraxial thin lens with a
circular aperture, evaluated analytically, so it runs in the forward modes (`B`
splat / `C` catch) with no per-element ray tracing. It gives correct focus-plane
placement and blur size but **no optical aberrations** (no spherical/chromatic
aberration, distortion, or field curvature) and a perfectly circular bokeh.

**Analytic projections** (fisheye/panoramic/orthographic, above) are likewise a
cheap closed-form remap in the forward pinhole mode — a true wide field of view
with none of the aberration or vignetting a real objective would add.

**Physical (realistic) lens** — `lens { … }`:

A camera can carry a real **lens prescription**: a stack of spherical/planar
glass interfaces plus an aperture stop. The backward tracer samples a film point
and a point on the rear element and traces the ray *through the actual glass*
(per-wavelength Snell refraction), so **depth of field, distortion, spherical &
chromatic aberration, field curvature and vignetting all emerge from the
geometry** — no thin-lens approximation. A physical lens renders in mode `R`
(backward realistic camera) by default, or in mode `D` (BDPT with the lens on the
camera subpath) when you want forward light transport's caustic efficiency through
the glass; mode `P` routes to whichever of those fits the scene.

```ftsl
camera "real" {
    eye 0 0.55 -1.6   look_at 0 0.35 2.4   up 0 1 0
    focus 2.4
    film { res 512 512   format full-frame }
    lens {
        preset achromat   # singlet | biconvex | achromat | doublet | telephoto | wide
        focal 50          # mm
        fstop 2.8
        glass BK7
    }
}
```

- **Presets** are physically derived (lensmaker equation for the singlet, Abbe-number
  power split for the achromatic doublet), so focal length and colour correction are
  correct by construction; render-time dispersion uses real Sellmeier glass indices.
- Or paste an **arbitrary real prescription** as repeated
  `surface <radius_mm> <thickness_mm> <ior> <semi_aperture_mm> [stop]` lines
  (PBRT lens-file convention). See `scenes/realcam.ftsl` for a working demo.

**Analytic vs. simulated — the tradeoff:** the physical lens is the **accurate but
slower** option. It captures real optical behaviour the thin lens cannot
(aberrations, distortion, field curvature, natural vignetting, dispersion-driven
colour fringing, and aperture-shaped bokeh), but it traces every camera ray
through the glass stack, so it is more expensive per sample than the analytic thin
lens. In both mode `R` and mode `D` it **runs on the GPU** (the backward and BDPT
megakernels each refract the camera ray through the glass stack on the device via the
same lens tracer), so within the GPU-supported scope it is still fast. Reach for the
analytic lens/projection when you want speed and a clean ideal image, and the physical
lens when you want a specific real objective's look.

*Current limits:* the lens attaches to the **camera subpath** — mode `R` (backward),
mode `D` (BDPT, keeping forward caustics but with the light-image splat disabled), or
mode `P` (which routes to `D`/`R`). It maps the sensor across the film width, so a
film whose aspect matches the sensor (e.g. `res 360 240` for a 3:2 sensor) covers it
without cropping, while a mismatched aspect crops. It does not model inter-element
flare/ghosting or shaped-iris bokeh. On the GPU it inherits its mode's scope (no
env/collimated lights or fluorescence, and at most 16 lens surfaces); outside that scope it falls
back to the CPU automatically.

---

## Materials

Declared with `material "name" { type <type> … }`.

| Type | Description | Key parameters |
|---|---|---|
| `diffuse` | Lambertian reflector | `reflect` (spectrum or `texture:<name>`) |
| `translucent` | Two-sided Lambertian (**diffuse transmission** / thin-subsurface look) — light diffuses THROUGH the surface, so a backlit sheet glows softly. Front hemisphere scatters `reflect`, back hemisphere scatters `transmit`; non-specular, so it connects/renders in every mode (A/B/C/R/V/D/P). Both lobes and the back-hemisphere connections run **on the GPU in mode D** (BDPT, M9). Alias `diffuse_transmit` | `reflect` (spectrum, `texture:<name>` or `pattern:`/`reflect_map`), `transmit` (spectrum or `pattern:`/`transmit_map`); the two are energy-clamped so `reflect+transmit ≤ 1` |
| `dielectric` | Refractive glass with dispersion, optional **frosting**, **colored-glass tint** and **nested-dielectric priority** | `ior` (Sellmeier glass or constant); `roughness` (constant or `pattern:`/`texture:` map) frosts the reflected & transmitted lobes; `absorb` (spectrum, σₐ per metre) tints via Beer–Lambert interior absorption; `priority <N>` (integer) disambiguates overlapping dielectrics — see below |
| `mirror` | Perfect specular reflector | `reflect` |
| `halfmirror` | Lossless beamsplitter; `reflect` is the reflect probability (default 0.5 = 50/50). A spectral `reflect` gives a wavelength-dependent (dichroic) split | `reflect` |
| `filter` | Colored **gel / Wratten filter**: a thin non-scattering absorber. Light passes straight through (no reflection or refraction), surviving with probability `transmit`(λ) — the per-wavelength transmittance T(λ) ∈ [0,1] — and is absorbed otherwise. Like clear glass it isn't lit directly; you see its effect on whatever is behind it | `transmit` (spectrum: `filter:<name>`, `file:<path>`, or a primitive like `gaussian`; also `pattern:<name>` / `transmit_map` for a transmittance that varies across the gel) |
| `glossy` | Rough microfacet reflector | `reflect`, `roughness` (constant or `texture:<name>` map) |
| `thinfilm` | Single-layer interference (iridescence) | `ior`, `film_ior`, `film_thickness` (nm), `film_thickness_map texture:<name>`, `substrate_k` |
| `multilayer` | N-layer Abelès transfer-matrix stack | `ior`, `substrate_k`, repeated `layer <n> <k> <nm>` |
| `grating` | Reflective diffraction grating | `reflect`, `groove_spacing` (nm), `groove_dir`, `max_order` |
| `fluorescent` | Stokes-shifted fluorescence. **Note `emit` means something different here:** on a fluorescent it is the *reradiation* spectrum — the SHAPE of the Stokes-shifted emission band, normalised by its own integral — **not** self-emission, so a fluorescent surface is never a light. (`emit_map` is therefore rejected on a fluorescent: a reradiation profile isn't a surface pattern.) For a surface that both fluoresces and glows on its own, use a `mix` of a `fluorescent` and an emissive `diffuse` | `reflect` (elastic base lobe), `absorb` (excitation band), `emit` (reradiation band), `yield` (quantum yield ≤ 1) |
| `hair` | **Fiber BCSDF** for hair / fur strands (Marschner R + TT + TRT, Chiang importance-sampled form, plus Yan's scattering medulla) — a scattering model for a translucent dielectric *cylinder*, not a surface. Meant for `curve` / `fur` geometry. See [Hair and fur fibers](#hair-and-fur-fibers-hair) below | `preset <species>` (a measured fur — everything below defaults from it); `reflect` (the colour you want the coat to be — inverted into an absorption, **not** a Lambertian albedo), or `sigma_a` (the absorption directly, which wins if present); `eta`, `beta_m`, `beta_n` (longitudinal / azimuthal roughness), `alpha` (cuticle tilt, degrees); `medulla` (κ), `medulla_sigma_s`, `medulla_sigma_a`, `medulla_g` |
| `mix` | Stochastic blend of materials | repeated `layer <material> <weight>`; optional `weight_map texture:<name>` **or `weight_map pattern:<name>`** (2-child spatial blend mask — with a pattern this becomes a math-driven *per-point material selection*, see Procedural patterns) |
| `layered` | Physical coat over a weighted body: reflect off the coat with prob R, else enter and pick one body lobe (energy-consistent). CPU only | `coat { reflectance fresnel\|thinfilm\|manual, ior, roughness[/roughness_map], film_ior, film_thickness[/film_thickness_map], specular }` + repeated body `layer <material> <weight>` |

**Whole-material presets** (`preset <name>`) fill a complete `Material` from a name:

- **Metals** (polished glossy lobe, override with `roughness`): `gold`/`Au`,
  `silver`/`Ag`, `copper`/`Cu`, `aluminium`/`aluminum`/`Al`,
  `chromium`/`chrome`/`Cr`, `brass`.
- **Glasses** (dispersive `dielectric`): `glass` (=BK7), plus every `glass:<name>`
  below (`BK7`/`crown`, `SF10`/`flint`, `silica`, `sapphire`, `diamond`, `water`,
  `ice`, `acrylic`, `polycarbonate`).
- **Iridescent / structural colour** (thin-film or multilayer stacks): `soap-bubble`,
  `oil-slick`, `anodized-ti`/`anodized-titanium`, `morpho`, `beetle`/`jewel-beetle`,
  `nacre`/`mother-of-pearl`.

Each iridescent preset is a **bundle file** (`data/material/<name>.material`) that
groups the material's several spectral envelopes (`ior`, `substrate_k`) and its tuned
film/stack geometry (`film_thickness`/`film_ior` or `layer <n> <k> <nm>` rows) under
one name — so new structural-colour materials drop in with **no rebuild**. Metals and
glasses need no file: a bare `metal:`/`glass:` name resolves by the generic convention
above. (The interference math stays native; only the parameters are data.)

**Translucency (dielectrics).** Beyond perfectly clear glass, a `dielectric` supports
two physically-motivated translucency controls (both compose with dispersion):

- **Frosted glass** — a `roughness` (0..1) puts a microfacet lobe on *both* the
  reflected and the refracted ray, so light scatters as it passes through. It accepts a
  constant, a `texture:<name>` map, or a `pattern:<name>` (so frosting can vary over the
  surface — see `scenes/procedural.ftsl`, whose height-banded glass sphere is clear at
  the bottom and frosted at the top).
- **Colored glass** — an `absorb` spectrum (absorption coefficient σₐ per metre)
  attenuates throughput by `exp(-σₐ(λ)·d)` over each in-glass path segment
  (Beer–Lambert), so thick regions tint more than thin edges. Authored like any
  spectrum (e.g. `absorb gaussian center=470 sigma=60 amp=14` for amber). Interior
  absorption is threaded through all three CPU transport loops (forward, backward,
  BDPT); see `scenes/translucency.ftsl`. *(GPU: forward + backward `R` accelerate both
  frosting and colored-glass tint; mode-`D` BDPT now runs colored glass **and** frosted
  (rough) glass on-device too (M9).)*

**Nested dielectrics (`priority`).** When two glass/liquid solids overlap — a glass
ice cube in a whisky, a lens cemented to another, a coating flush against a body — the
exterior index at the shared boundary is ambiguous: is the ray leaving *into air* or
*into the other medium*? Give each `dielectric` an integer `priority <N>` and the
higher priority wins wherever they overlap (Schmidt & Budge 2002). The winning medium's
surface refracts; the losing (lower-priority) surface inside it is *suppressed* — the
ray passes straight through it — and the exterior IOR at each real interface is taken
from the medium actually enclosing the ray (so glass-in-water refracts 1.33↔1.52, not
1.0↔1.52). Every render mode honours it (CPU forward/backward/BDPT/VCM/photon/SPPM and
the GPU forward/backward/BDPT/photon backends).

Priorities are **opt-in and safe to omit**: a scene that never writes `priority` renders
exactly as before (each dielectric treated against air). Because that flat model is
ambiguous precisely where dielectrics overlap, ftrace runs an **ahead-of-time audit** at
load and prints `[priority] WARNING: …` for every pair of overlapping *different*
dielectrics that don't both carry a disambiguating priority (spheres, meshes, and
isosurfaces alike — isosurface overlap is detected conservatively by comparing their
`contained_by` bounds). Add distinct priorities to the flagged materials to silence it.

```
material "water" { type dielectric ior 1.33  priority 1 }
material "glass" { type dielectric ior 1.52  priority 2 }   # wins where it overlaps water
```

### Hair and fur fibers (`hair`)

Every other material here shades a **surface**: it takes a normal, projects incoming light
by `cos(n, w)`, and scatters into a hemisphere. A hair or fur strand is not a surface. It
is a translucent dielectric **cylinder** a few tens of microns across, and most of the
light that meets it goes *through* it. Shading a strand with `diffuse` therefore gets the
two things that actually make hair look like hair exactly backwards: the coat has no
forward glow when it is backlit, and its highlight sits in the wrong place.

`type hair` implements the standard fiber BCSDF for that geometry — Marschner et al. 2003
("Light Scattering from Human Hair Fibers") for the lobe decomposition, Chiang et al. 2016
("A Practical and Controllable Hair and Fur Model for Production Path Tracing") for the
importance-sampled, energy-conserving form used here:

| Lobe | Path | What you see |
|---|---|---|
| **R** (p=0) | reflects off the cuticle | the white, unsaturated primary highlight — displaced toward the root by the tilted cuticle scales |
| **TT** (p=1) | in one side, out the other | the strong **forward** lobe. This is the rim of light on backlit hair, and it carries the strand's colour (one crossing of the absorbing interior) |
| **TRT** (p=2) | in, one internal bounce, out the same side | the **secondary** highlight: offset from R, and much more saturated (two crossings) |
| residual | p ≥ 3, folded into one term | makes the lobe weights sum to exactly 1, so a non-absorbing fiber passes a white furnace test |

Parameters:

| Key | Default | Meaning |
|---|---|---|
| `reflect` | `0.3` | The colour you want the **coat** to be. This is *not* a Lambertian albedo — it is inverted (Chiang eq. 9) into the interior absorption that reproduces that colour under multiple scattering, so what you type is roughly what converges. |
| `sigma_a` | — | The absorption coefficient directly, in units of 1/(fiber radius). The physical spelling; when present it **wins** and `reflect` is not consulted. Real hair is roughly `rgb 0.42 0.63 1.19` (brown) to `rgb 3.3 5.2 7.6` (black). |
| `eta` | `1.55` | Cuticle index of refraction. 1.55 is keratin. |
| `beta_m` | `0.3` | **Longitudinal** roughness (0–1): how far the highlight smears *along* the strand. ~0.05 wet or glass fiber, ~0.3 normal hair, ~0.7 coarse animal fur. |
| `beta_n` | `0.3` | **Azimuthal** roughness (0–1): how far it smears *around* the strand. Low values give a hard, glinting TRT. |
| `alpha` | `2.0` | Cuticle scale tilt, in **degrees**. This is what separates R and TRT into two distinct bands; 2° is human hair. |

```
material "blonde" { type hair  reflect rgb 0.72 0.55 0.28  beta_m 0.30  beta_n 0.30 }
material "sleek"  { type hair  sigma_a rgb 0.42 0.63 1.19  beta_m 0.10  beta_n 0.12
                    alpha 3.0 }
fur "coat" { on "head"  material blonde  count 70000  length 0.055  radius 0.00005 }
```

#### The medulla — what makes fur not hair

A human hair is close to a solid rod. **Animal** fur is not: it has a **medulla**, a wide
scattering core running down the middle, and that core is most of the difference between
the two. Light that enters an animal fiber usually does not cross it cleanly — it hits the
core, bounces around inside, and leaves in a direction that has partly forgotten where it
came in. So a real coat's forward glow is soft and broad where a hair's TT lobe is a sharp
blade, and its secondary highlight is a wash rather than a glint.

`type hair` models this after Yan et al. 2017 ("A BSSRDF Model for Efficient Rendering of
Fur with Global Illumination"), which adds two **scattered** lobes, TT<sup>s</sup> and
TRT<sup>s</sup>, alongside the three specular ones. Its key simplification is to give the
cortex and the medulla the *same* index of refraction, so the interior ray does not bend at
the core boundary — the path topology stays exactly R / TT / TRT, and the medulla only
changes what happens *along* a chord.

| Key | Default | Meaning |
|---|---|---|
| `medulla` | `0` | **κ**, the medulla's radius as a fraction of the fiber's. `0` leaves a solid Marschner cylinder and makes the three keys below inert, so nothing changes for an existing `hair` material. Human hair is ≈ 0.36; every animal Yan measured is 0.65–0.91. |
| `medulla_sigma_s` | `0` | Scattering coefficient of the core, in 1/(fiber radius). This is the knob that turns a glint into a wash. |
| `medulla_sigma_a` | `0` | Absorption coefficient of the core. Usually small — the core mostly scatters. |
| `medulla_g` | `0` | Henyey–Greenstein anisotropy of the core, −1…1. Forward-peaked (positive) cores randomise the direction more slowly. |

Energy is still exact: the scattered lobes are built as the *difference* between the
specular chain with the medulla and the same chain with the core replaced by cortex, so all
six lobes sum to 1 in a white furnace as an algebraic identity, not a near-miss
(`-checkhair` §S1 asserts it at 1e-12, medullated and not).

> **Where the colour has to go.** `reflect` and `sigma_a` tint the **cortex** — and on a fur
> fiber there is barely any cortex left. κ = 0.87 means the core is 87 % of the radius, so a
> ray crosses a thin pigmented shell and then meets a strongly scattering, colourless core
> that sends it back out almost untinted. Tinting the cortex of a big-core species gives you
> a *pale* coat, not a coloured one. Use `medulla_sigma_a` — which is a spectrum, like every
> absorption here — to colour the core, and use `reflect` for the shell on top of it. On a
> small-κ fiber (human hair) the old intuition still holds and `reflect` is all you need.
>
> The same effect makes a medullated coat much **brighter** than the solid fiber with
> identical parameters: measured on one ball under one key light, a `preset cat` coat reads
> 82 against 34 for the same fiber with `medulla 0`. A solid cat fiber has `beta_n` 1.3°, so
> its TT lobe is a razor-thin forward spike that fires light straight through the coat into
> whatever is underneath; the core intercepts it a fraction of a radius in and scatters it
> broadly back out. `scenes/fur_species.ftsl` renders that A/B side by side.

#### `preset` — measured species

Yan et al. fitted their model to goniophotometer measurements of ten real fibers. Those
fits are shipped as named presets:

```
material "fox" { type hair  preset redfox }
```

`bobcat`, `cat`, `deer`, `dog`, `mouse`, `rabbit`, `raccoon`, `redfox`, `springbok`,
`human`. Spelling is forgiving — `red fox`, `red_fox`, `RedFox` all work.

A preset supplies **defaults only**, so any key you also write wins:
`preset redfox  beta_n 0.05` is a red fox with the glint sharpened, and
`preset rabbit  reflect rgb 0.9 0.9 0.9` is a white rabbit (an explicit `reflect` overrides
the preset's measured cortex absorption; without one, the measured value is used). A preset
sets `eta`, `alpha`, `beta_m`, `beta_n`, `sigma_a` and all four medulla keys at once — the
paper reports the roughnesses as Gaussian widths in degrees, and they are converted into
the perceptual 0–1 knobs on the way in, so the table in the source can be diffed against
the paper line by line.

Put it on [`curve`](#curves-and-fibers-curve) or [`fur`](#grooms-fur) geometry: those
intersectors report the fiber axis and the impact parameter, which is what the model needs.
On a triangle mesh it still shades (the surface tangent stands in for the axis) but the
result is not physically meaningful.

Two practical notes:

- **Light it broadly.** A fiber is thin, so a small or point-like source leaves most of
  each strand's circumference unlit and the whole coat reads as black felt. Area lights,
  and a back light to feed the TT lobe, are what make the model worth having.
- **Mode support.** Hair renders in `W`, `R`, `A`/`B`/`C`, `D` (BDPT) and `U` (VCM). Modes
  `M` (photon map) and `S` (SPPM) *scatter* through it correctly but never gather on it —
  their photon records store no incident direction, so a directional fiber lobe has nothing
  to evaluate against; a strand is treated like a glossy surface there. Hair runs **on the
  GPU** in the forward modes (`A`/`B`/`C`) and the backward tracer (`R`, `W`) — and the
  modes composed from them (`V`, `P`) — since 0.181.0: on `hair_basics`, GPU mode `R` is
  ~20× the CPU and mode `B` ~6×. Renders that still fall back to the CPU tracer:
  `-dual-scatter` (the approximation is host-side), and hair scenes in the GPU BDPT (`D`),
  photon-map (`M`/`S`) and VCM (`U`) backends, whose vertex/gather machinery would shade a
  strand as Lambertian.

See `scenes/hair_basics.ftsl`.

#### Dual scattering (`-dual-scatter`)

A pale coat is *dominated* by multiple scattering. A single blonde fiber is nearly
transparent; a head of blonde hair is bright and soft, and essentially all of that
brightness is light that has crossed dozens of strands. A path tracer gets this right by
brute force, which means a hundred-plus bounces per path, and that is what makes white fur
the slowest thing in this renderer.

`-dual-scatter` replaces those bounces with the analytic approximation of Zinke et al. 2008
("Dual Scattering Approximation for Fast Multiple Scattering in Hair"). It splits the
multiply-scattered radiance in two:

- **Global** — light reaching the shading fiber *through* the coat. A shadow ray is walked
  strand by strand (Zinke's §4.1.1 "ray shooting"), and each crossing multiplies in that
  fiber's average forward attenuation `ā_f(θ)` and adds its forward spread `β̄_f(θ)²`. The
  light then arrives attenuated **and** blurred, not from a point.
- **Local** — light that scattered *backward* out of the strands behind the shading point
  and came back. That is a closed form in `ā_b`, an infinite sum over how many times the
  light bounced back and forth, which Zinke collapses into `Ā_b`, a mean shift `Δ̄_b` and a
  width `σ̄_b`.

| Flag | Default | Meaning |
|---|---|---|
| `-dual-scatter` | off | Enable the approximation. Backward modes (`R`, `W`) only. |
| `-dual-density <d>` | `0.7` | Zinke's `d_f` = `d_b`: "how enclosed is a strand", i.e. how much of the coat's own scattering the analytic terms should account for. The paper uses 0.7 throughout and suggests 0.6–0.8. Lower = a more open, darker coat. This is the knob to reach for if the coat reads dark — see the table below. |
| `-dual-db <d>` / `-dual-df <d>` | follow `-dual-density` | Override one density factor on its own — `d_b` weights the local backscatter lobe, `d_f` the light let through the coat. Mostly a diagnostic: `-dual-df 0` leaves only the directly-lit term. |
| `-dual-max-cross <n>` | `64` | How many strands one shadow ray counts before giving up. A dense coat can exceed this; the ray is then treated as reaching the light with whatever it accumulated. |
| `-dual-grid [cells]` | off (`2097152` = 128³ when given) | Count the crossings from a **fiber-density grid** instead of walking the strands — Zinke's §4.1.2 instead of §4.1.1. Much faster on a dense coat; see below. The optional argument is a cell *budget*, split into roughly cubic cells over the fur's bounding box. |

Two things here are deliberately *not* what the paper does:

- **The six averaged curves are measured from this renderer's own BCSDF**, not from
  Marschner's three lobes. `ā_f`, `ā_b`, `ᾱ_f`, `ᾱ_b`, `β̄_f`, `β̄_b` are built per material,
  per wavelength, by importance-sampling `hair::sample()` and splitting the result by
  azimuthal half. So the table inherits the model's energy conservation exactly (`ā_f + ā_b`
  is the furnace total, asserted to 1e-4 by `-checkhair` §S11) and picks up the medulla's
  TT<sup>s</sup> / TRT<sup>s</sup> lobes for free.
- **The exact series are summed, not Zinke's eq. 16/17 fits.** Those are expansions in
  `a_b²/(1−a_f²)²`, which is not small for any plausible coat, and eq. 16 also carries a
  sign error (it prints `1 − 2u` where the sum gives `1 + 2u`; `-checkhair` §S11 demonstrates
  this by showing the printed form is first-order wrong in `u` while the corrected one is
  second-order accurate). The sums have closed forms and cost nothing at table-build time,
  so they are used directly.

**This is biased, on purpose.** The analytic terms already carry the coat's multiple
scattering, so continuing the path from a fiber would double-count it: `-dual-scatter`
therefore **terminates the path at a fiber vertex**. Everything else in the scene keeps
full path tracing — only fur becomes one-bounce. Direct lights and the environment both go
through the approximation; what the coat loses is *indirect* illumination, i.e. light that
reached it by bouncing off the rest of the scene first.

##### The fiber-density grid (`-dual-grid`)

The global term only needs to know **how many** fibers a shadow ray crossed and at what
inclinations — never *which* ones. The default walk (§4.1.1) nevertheless pays for the
identities: it must find every strand along the ray in order, which means it cannot stop at
the first blocker the way an ordinary shadow ray can, and on a dense coat that is thousands
of curve intersections per shadow ray.

`-dual-grid` builds Zinke's §4.1.2 aggregate instead. Each cell stores two things summed
over the fiber pieces inside it:

- a scalar density `c = (2/V)·Σ rᵢℓᵢ`, and
- a normalised orientation tensor `T = Σ rᵢℓᵢ t̂ᵢt̂ᵢᵀ / Σ rᵢℓᵢ`.

A fiber of radius `r`, length `ℓ` and tangent `t̂` presents cross-section `2rℓ·sinθ` to a ray
travelling along `d`, with `sin²θ = 1 − (d·t̂)²`. Pulling the square root outside the sum
(Jensen) turns the per-strand sum into those two aggregates:

> `σ_t(d) ≈ c·√(1 − dᵀTd)`

and the same quadratic form `dᵀTd` **is** `⟨sin²θ⟩` in Marschner's longitudinal frame, so one
DDA march yields both the optical depth and the inclination the averaged tables want. The
key identity is that `∫σ_t dt` along a ray *is the expected number of fiber crossings* — the
walk's `n` — obtained with no primitive tests at all.

Two consequences worth knowing:

- **The count is sampled, not rounded.** `τ = ∫σ_t dt` is a *mean*; the walk returns a random
  draw, and the shader is not linear in it. So the grid draws `N ~ Poisson(τ)` from one extra
  uniform. That keeps `E[a_f^N] = e^{τ(a_f−1)}` correct rather than `a_f^τ` (a 26% error at
  `a_f = 0.8`, `τ = 10`), keeps the spread a distribution of widths, and — most visibly — keeps
  the `N = 0` *directly lit* case reachable, so a rim strand at `τ = 0.3` is still fully lit
  70% of the time instead of always being dimmed.
- **The Jensen step is biased high by at most +3.98%**, and one-signed, so the grid never
  under-attenuates. It is *exact* wherever the fibers in a cell are locally parallel (`T` is
  then rank-1 and the root factors out) and worst at full isotropy. `-checkfurgrid` measures
  this directly against the real curve intersector.

Since the grid only replaces the *counting*, an ordinary early-outing occlusion query
(`occludedSkipHair`) still runs first for everything that is not a hair fiber, so opaque
blockers shadow the coat exactly as before.

The grid is opt-in because it does lose two things: the shading point's own texture
coordinates are used for the crossed material's tables (the walk samples each crossed fiber),
and the tangent's *sign* is unrecoverable from `t̂t̂ᵀ`, so the tables are evaluated at ±θ and
averaged. Both are small — the dual tables are near-even in θ apart from the ~3° cuticle
tilt — but they are approximations on top of an approximation.

**What it actually costs and buys.** Measured against each scene's own 200-bounce path-traced
reference, on the fur alone (a centred crop, scene-linear mean luminance — the whole frame
would be diluted by the room):

| Scene | `-dual-scatter` (walk) | `+ -dual-grid` | speed vs reference |
|---|---|---|---|
| Pale coat, lit only by an area light (`scenes/_dual_pale_lamp.ftsl`) | **0.77×** in 18.2 s (0.99× at `-dual-density 0.9`) | **0.81×** in 14.1 s | **2.2× → 2.9× faster** |
| The same coat lit only by a constant sky (`scenes/_dual_pale_sky.ftsl`) | **0.89×** in 15.5 s | **0.89×** in 14.0 s | **2.4× → 2.6× faster** |
| `scenes/fur_species.ftsl` — medullated, absorbing, in a white room | **0.67×** in 84.8 s | **0.68×** in 57.4 s | **0.7× (slower) → 1.06× faster** |

Single scattering alone — no multiple-scattering term at all — reads 0.17×, 0.80× and 0.37×
on those three, which is what the approximation is being asked to close.

Read that as: on the case it exists for — a pale coat where the brute-force walk is long —
it recovers most of the multiple scattering at a third to a half of the cost, and
`-dual-density` closes the rest (0.7 is the paper's conservative default, not a fit to your
coat). On a *dark* coat there was little multiple scattering to approximate in the first
place, and the walk was slow enough to be a net loss; the grid is what makes that case pay
(1.5× faster than the walk on `fur_species`, at essentially unchanged accuracy). And in a
bright room the dropped indirect bounce is the dominant error, not the approximation itself.
Render the reference when the coat *is* the picture; use `-dual-scatter` for
look-development, for pale fur that is scenery rather than subject, and for flyby frames —
and add `-dual-grid` whenever the coat is dense.

Grid build cost is small and one-off: 187 ms / 64 MB at 128³ over the pale coat's 900k
segments, 423 ms / 64 MB over `fur_species`'s 2.48M.

The tables are cached per (material, wavelength bin, absorption bin) and built lazily on
first use, so the cost is paid once per render regardless of how many strands there are.

##### The coat as a medium (`-fur-volume`)

`-dual-grid` keeps the strands and reads only the *shadow* off the grid. `-fur-volume` goes
the whole way: with it on, **fibers are not geometry at all**. `Scene::closestHit` skips
every `MatType::Hair` curve, and the backward tracer instead samples a free flight against
the same field's `σ_t(d)`. This is the coat's **far LOD tier** — for fur that is small on
screen, where a strand is under a pixel and there is no silhouette left to resolve.

A collision is three steps, all of which `-checkfurvol` verifies independently:

1. **Free flight**, exact rather than delta-tracked. Along a *fixed* ray the direction
   argument of `σ_t(d)` never changes, so `σ_t` is piecewise constant on the DDA's own cell
   segments and `∫σ_t dt = −log(1−u)` inverts by running subtraction inside the march. No
   majorant, no null collisions — which matters, because a coat (a thin dense skin inside a
   mostly empty box) is exactly the case delta tracking handles worst.
2. **A tangent**, drawn from the cell's reconstructed orientation distribution. The grid
   stores only the second moment `T`, so the ODF is reconstructed as the **Bingham** —
   the maximum-entropy distribution on the sphere with that moment. (Two cheaper families
   were measured first: a Watson mixture turns a girdle into two orthogonal lobes and an ACG
   smears a combed clump.) The draw is importance-sampled **by cross-section**, since a ray
   meets a perpendicular fiber more often than a parallel one.
3. **The ordinary fiber BCSDF** at a *virtual* hit. There is no surface to measure the impact
   parameter `h` from, so `h` is drawn uniformly on `[−1, 1]` — as it is for a ray crossing a
   cylinder at a uniform offset — and the normal that *would* have produced it is
   reconstructed. Everything downstream is the same `hair::` code a real strand runs.

Next-event estimation changes with it. `scene.occluded` would report the very strands this
tier is pretending not to have as blockers, so a connection from a fur collision tests only
non-hair geometry for occlusion and multiplies in the coat's `exp(−τ)` transmittance as a
continuous factor instead. Direct lights and the environment both go through it.

| Flag | Default | Meaning |
|---|---|---|
| `-fur-volume [cells]` | off (`2097152` = 128³ when given) | Render every `type hair` coat as a participating medium. Backward modes (`R`, `W`) only. Shares the density field with `-dual-grid` — given both, it is built once at the larger budget — and adds 16 B/cell for the orientation table (32 MB at 128³). Once the field is built the fibers are **deleted**, before any BVH is built over them. |
| `-fur-keep-strands` | off | Opt out of that deletion: keep the fibers loaded and in the BVH even under `-fur-volume`. For A/B-ing the two tiers in one process, or if something in a scene still needs the geometry. |

**What it costs and buys.** It is faster than the strands it replaces, and — the point of the
whole tier — its cost barely moves with fiber count. Measured on the same coat at three fiber
counts at **fixed optical density** (strand count ×k with radius ÷k, so the picture and the
number of scattering events are unchanged and only geometric complexity grows), mode `R`,
200×150, 200 spp, `-max-bounce 32`, every run landing on the same 7.07 % noise:

| strands | curve segments | strands | `-fur-volume` | speed-up |
|---|---|---|---|---|
| 90 k | 900 k | 13.2 s | **7.7 s** | 1.7× |
| 300 k | 3.0 M | 28.2 s | **7.9 s** | 3.6× |
| 900 k | 9.0 M | 60.3 s | **8.6 s** | 7.0× |
| 3 M | 30 M | *does not load* | **8.1 s** | — |
| 9 M | 90 M | *does not load* | **9.2 s** | — |

Across that **100×** range in fiber count the aggregate grows **1.19×**, and over the 10× where
the strand tier can still be measured at all it grows **4.6×**. That is the property the tier
exists for. The last two rows are the other half of it: at 3 M strands and above the strand tier
dies with `error: bad allocation` (30 M `CurveSeg`s plus a BVH over them), while the aggregate —
which deletes the fibers as soon as it has summarised them — renders in about the same time as
the smallest coat in the table.

(Before v0.179.0 the tier was a 3.8× *loss* and scaled just like the strands, because making
fibers invisible was done at the BVH leaf rather than by removing them from the tree; before
v0.180.0 it kept them in memory. See `known-issues.md` for both.)

It is also **accurate**: developed through one shared `-exposure-anchor`, the aggregate's
scene-linear mean luminance over the coat lands **0.9 %** below the strand reference (0.9989 of
it over the whole frame). It composes with fog correctly (the first collision in a union of
independent media is the minimum of their independent free flights, which is exactly how the two
are sampled), and it deliberately does **not** combine with `-dual-scatter`: dual scattering is
an analytic stand-in for the very multiple scattering this path now simulates directly.

**It saves memory too, which is what makes the last two rows possible.** Since v0.180.0 the
fibers are *deleted* once the density field and the orientation table have been built from them —
and deleted **before** the BVH is built, not after, because the BVH build over the coat *is* the
memory peak (at 9 M segments it reserves 2 N nodes × 64 B = 1.15 GB on top of a `BuildPrim` and
an `Aabb` per primitive and the 720 MB of segments themselves). Freeing afterwards would have
returned the memory without ever letting a bigger coat load.

| coat | before | after |
|---|---|---|
| 900 k strands / 9 M segments | 2221 MB peak | **875 MB** |
| 3 M strands / 30 M segments | `error: bad allocation` | **2669 MB** |
| 9 M strands / 90 M segments | `error: bad allocation` | **7796 MB** |

The deletion is skipped automatically whenever something still needs the geometry — `-fur-lod`
(its near tier traces strands), `-dual-scatter`, `-raster` / `-explore` / `-anim` / `-loom`, or a
camera whose mode is not backward `R`/`W`, in which case the run says so:

```
[fur-volume] keeping the strands: mode B traces fiber geometry directly (only backward R/W
             renders the coat as a medium)
```

`-fur-keep-strands` forces that same behaviour by hand. The image is unaffected either way: the
dropped and kept renders are byte-for-byte identical, and both are byte-for-byte identical to
what v0.179.0 produced.

Two things are genuinely lost, both from the same cause — a collision knows its cell, not a
strand. There is no `u`/`v`, so a hair material whose colour comes from a texture or pattern
reads at the default hit's coordinates (the same class of approximation as `-dual-grid`'s
textured `σ_a`); and per-strand silhouette detail is gone by construction, which is the
point of a far tier and the reason it is opt-in rather than automatic.

##### Choosing a tier (`-fur-lod`)

`-fur-volume` on its own is a *mode*: every path goes through the medium, however close the
camera is. `-fur-lod` makes it a *decision*, and the decision is about what a pixel can see.

The ruler is the width of one pixel where the coat starts, measured in fiber diameters —
`Camera::footprintPerDist(1)` times the distance at which the camera ray enters the coat's
bounding box, over the grid's length-weighted mean fiber diameter. Below `d0` diameters a
strand still has a silhouette worth tracing and the path uses the strands; above `d1` it does
not and the path uses the medium.

| Flag | Default | Meaning |
|---|---|---|
| `-fur-lod [d0[:d1]]` | off; `1:4` when given | Trace strands while a pixel is narrower than `d0` fiber diameters at the coat, the aggregate once it is wider than `d1`, stochastic crossfade between. Implies `-fur-volume`. One number sets `d0` and puts `d1` two octaves up. |

Three things about it are deliberate.

**The ruler is the pixel, not the sample.** `footprintPerDist(1)`, never
`footprintPerDist(spp)` — which is the opposite of what the `fw` shading footprint does, and
the one part of this that is easy to get backwards. `fw` band-limits a sampler that cannot
average over its own pixel, so more samples must relax it. LOD is not that: if a fiber is
thinner than a pixel, *no* number of samples will put its silhouette into the final image —
the reconstruction filter averages it away, and the aggregate is precisely that average. A
ruler that shrank with `-spp` would make a converged render pick a different tier from its own
preview, which is exactly the pop the flag exists to prevent.

**The crossfade is stochastic, and per path.** Inside the band each path flips one coin
against a smoothstep of the footprint. Not a weighted sum of two renders, because a blend
needs *both* estimators evaluated — in the band that costs more than either tier alone, and it
would still have to reconcile two incompatible visibility conventions inside one path. A coin
costs nothing, is unbiased for the same blend, and mode `R` already averages hundreds of paths
per pixel, so what the image shows is the blend and not the coin. Smoothstep rather than a
linear ramp so that the derivative vanishes at both ends and neither edge of the band is
itself an edge.

**The choice is sticky.** A path decides once, on its first segment, and a gather ray or a
`-herosplit` re-entry inherits it rather than re-rolling. A path that half-believed in the
strands would test visibility against geometry its own vertices were not built from — the
aggregate vertex skips fibers and multiplies in `exp(−τ)`, the strand vertex tests them
directly, and mixing the two inside one path double-counts the coat.

**Outside the band it costs nothing, exactly.** No coin is flipped unless the footprint is
*inside* [`d0`, `d1`], so the random stream is untouched and a render below `d0` comes out
**byte-identical** to one with no `-fur-lod` at all, while one above `d1` is byte-identical to
plain `-fur-volume`. Measured on a 90 k-strand coat at 200×150, 400 spp, sweeping the
threshold so the same geometry walks the whole band (coat-only scene-linear mean luminance;
all seven renders landed on the same auto-exposure):

| threshold | coat mean Y | vs. strands | time |
|---|---|---|---|
| *(strands, no flag)* | 0.67641 | — | 28.5 s |
| `-fur-lod 100:200` | 0.67641 | byte-identical | 33.3 s |
| `-fur-lod 40:80` | 0.67641 | byte-identical | 32.0 s |
| `-fur-lod 24:48` | 0.67641 | byte-identical | 29.4 s |
| `-fur-lod 12:24` | 0.67620 | ×0.9997 — in the band | 36.5 s |
| `-fur-lod 4:8` | 0.67353 | ×0.9957 | 90.6 s |
| *(`-fur-volume`)* | 0.67353 | ×0.9957, byte-identical to `4:8` | 90.8 s |

The two endpoints are 0.43 % apart, which is the real reason the transition does not pop: the
tiers already agree on brightness, so the fade only has to hide a change in *noise character*.

`-checkfurvol` §9 covers both halves: that the entry distance leaves exactly zero optical
depth behind it (checked against the grid march, not against itself), and that the realised
aggregate fraction tracks the smoothstep it claims — monotonically, and *exactly* 0 and 1 at
the two ends, since a coat that is one-in-a-thousand aggregate at point-blank range is a coat
with sparkling holes in it.

**Parametric records.** A **record** is a named bank of per-channel look-up tables over
a shared scalar domain `[lo,hi]`. A single per-hit **driver** scalar samples every
channel at once, and each channel whose name matches a material slot fills that slot at
the driven value — so one expression coordinates a sweep across a material's slots
(`reflect` → diffuse albedo, `roughness` → glossy roughness). Colour channels list
prefixed spectrum refs and interpolate in linear RGB (then upsample back to a
reflectance); scalar channels list pattern expressions (the same math VM as procedural
patterns). `interp nearest|linear|smooth` selects the sampling mode — `smooth` is a
monotone Fritsch–Carlson cubic (no overshoot).

```
grad = range 0-1 [
    reflect  spectrum:steel  spectrum:gold  spectrum:copper   # steel -> gold -> copper
    interp   smooth
]
sphere { center 0 0 0  radius 1  material grad(u) }                   # sweep along u
sphere { center 2 0 0  radius 1  material grad(noise(9*x,9*y,9*z)) }  # mottled by noise
```

**Inline colour channels.** A colour channel doesn't have to name pre-declared spectra:
tag the channel with any colour head — `rgb`, `hsv`, `hsl`, any upsampler/emission
variant (`rgbmeng`, `rgbsmits`, `hsvillum`, `hslline`, …), or a user upsampler
(`rgb:<name>`) — and write the triples inline.
The tag fixes arity 3, so each group is one stop, and the components go through exactly
the same evaluator as a top-level `spectrum "x" = rgb …` declaration:

```
palette = range 0-1 [
    reflect  rgb 0.9 0.1 0.1, 0.1 0.9 0.1, 0.1 0.1 0.9   # red -> green -> blue
    tint     rgbmeng 0.15 0.65 0.85                       # a lone stop needs no comma
]
```

**Stop delimiters** form a **precedence ladder**: whitespace binds tightest (like `×`),
comma looser (like `+`), and `[ ]` are the parentheses. Structure comes from the
delimiters alone — a channel's arity only validates — so these are all the same channel:

```
reflect  rgb 0.9 0.1 0.1, 0.1 0.9 0.1        # comma-separated triples
reflect  rgb [0.9 0.1 0.1] [0.1 0.9 0.1]     # bracketed groups
reflect  rgb [0.9 0.1 0.1, 0.1 0.9 0.1]      # one bracketed comma list
```

Parens `( )` are *not* a rung — they're reserved for expressions, so a parenthesised run
is opaque and `0  clamp(u,0,1)  1` stays three stops. A channel line using none of
`,` `[` `]` and no colour tag reads exactly as it always did, so the ladder is a strict
addition: no existing scene reparses differently. Position pins are the orthogonal
`p:<pos>` prefix and work in either spelling.

Bind a record to geometry with the inline `material NAME(driver)` form, where `driver`
is any pattern expression evaluated per hit (`x y z nx ny nz r u v f`, `noise(…)`, …).
A record driving the **reflect/albedo** *or* **roughness** slot runs on the **GPU**
forward, backward, **and BDPT (`D`)** tracers (the LUT/stop programs + driver upload to
the device and are sampled by device twins of the CPU sampler; mode `D`'s connection BSDF
reconstructs the per-hit point to sample the driver). Fallback is automatic. See FTSL.md
§7.5 for the full grammar.

**Materials are parameterized bundles.** A material property is an expression over
*named inputs*, so a material is itself a function whose free-input set is the union of
its properties' free inputs. Applying it at a use site binds those inputs across the
whole bundle at once:

```
pattern "rough_ua" { expr "0.5*a*(0.2+0.8*u)" }        # free inputs: a, u
material "gold" { type glossy  reflect spectrum:gold
                  roughness pattern:rough_ua  albedo_default 0.5 }

sphere { center 0 0 0  radius 1  material gold(u=v,a=1) }  # bind u<-v, a<-1
sphere { center 2 0 0  radius 1  material gold(u=v a=1) }  # identical — same ladder
sphere { center 4 0 0  radius 1  material gold(u=v) }      # partial: a -> albedo_default
sphere { center 6 0 0  radius 1  material gold(0.5,a=1) }  # positional binds the last free input
```

Bindable inputs are the surface intrinsics `x y z nx ny nz r u v f` plus **`a`** —
albedo, the one input with no per-hit intrinsic, resolved at load time either to what a
use site binds or to the material's `albedo_default` (default `1.0`). Binding is pure
substitution into the postfix program, so an applied material is an *ordinary* material:
no environment, no runtime indirection, and a material nobody applies is bit-identical
to before. `ftrace -checkbind` pins the algebra (splice == inlining, simultaneity,
identity). See FTSL.md §7.6.

You can also read **one property** off an already-declared material and use it as a value
elsewhere — `reflect src.reflect`, `reflect src.reflect(u=v)`, `roughness other.roughness`.
The handle is the **slot keyword**, since FTSL properties are written with the slot keyword
and never carry a quoted name. The argument list is the same one above, so an unbound `a`
resolves against the **source** material's `albedo_default` — the property carries the
source's notion of albedo with it. A reference carries both the base spectrum *and* the
slot's per-hit pattern, and composes (multiplies) with the reader's own `<slot>_map` rather
than clobbering it; a record-driven, texture-bound, or un-appliable pattern slot is refused
rather than approximated. `ftrace -checkprop` pins it against hand-written twins. See
FTSL.md §7.7.

---

## Spectra (SPDs, reflectances, indices)

Anywhere a spectrum is expected (`spd`, `reflect`, `ior`, …) you can write:

- **`preset:<name>`** — illuminants and light sources:
  - **Blackbody / daylight:** `bb<K>` Planckian (e.g. `bb6500`), `sun`,
    `d65`/`daylight`, `a`/`incandescent`.
  - **White LED:** `led` (neutral), `led-warm`, and `led<K>k` phosphor LED at a colour
    temperature (e.g. `led4000k`).
  - **Colored LED:** single-die narrow-band emitters `led-violet`, `led-royal-blue`,
    `led-blue`, `led-cyan`, `led-green`, `led-amber`, `led-red`, `led-deep-red`
    (measured die SPDs, Brendel 2021, CC BY-SA 4.0).
  - **Fluorescent:** `fluorescent`/`cfl` (generic compact-fluorescent model) plus the
    measured CIE F-series `f2`/`cool-white`, `f7`/`daylight-fl`, `f11`/`triphosphor`.
  - **Gas-discharge lamps:** `hps`/`sodium` (high-pressure sodium),
    `lps`/`sodium-low` (low-pressure sodium), `mercury`/`hg` (mercury vapor),
    `metal-halide`/`mh` (analytic line model), plus measured ceramic-metal-halide
    SPDs `cmh`/`cmh-3000k` (warm white) and `cmh-4200k` (cool white), digitized from
    the GE ConstantColor CMH G12 datasheet and colour-matched to its published
    chromaticity.
- **`rgb r g b`** — Jakob–Hanika sigmoid upsampling to a reflectance spectrum
  (round-trips under D65).
- **`hsv h s v`** — an HSV colour (hue `h` in `[0,1]` turns and *wraps*, so a hue
  swept over a loop cycles the whole wheel seamlessly; `s`/`v` in `[0,1]`),
  converted to RGB and then upsampled exactly like `rgb`.
- **`hsl h s l`** — an HSL colour on the same wrapping hue wheel, but `l` is
  *lightness* (`l=0.5` is the pure hue, `l→1` white, `l→0` black; matches CSS);
  `s`/`l` in `[0,1]`. Converted to RGB and upsampled exactly like `rgb`.
- **`rgbline r g b [sigma]`** (also `hsvline …`, `hslline …`) — the
  *dominant-wavelength* form: instead of a broadband reflectance, map the colour to a
  single dominant wavelength (the standard spectral-locus construction — a ray from the
  D65 white point through the colour's chromaticity to the spectral horseshoe) and emit
  a **narrow Gaussian line** there. A near-monochromatic source, so glass disperses it
  correctly. Line width defaults to the colour's saturation (a vivid colour → tight
  spike; a pale one → a broad band tending back to white) or is forced by an explicit
  `sigma` in nm. Purples/magentas (no real dominant wavelength) become a two-line
  violet+red mix. Meant for **lights** (`spd rgbline 0 0 1`); a reflectance has no
  single wavelength, but the form is accepted anywhere a spectrum is.
- **`rgbillum r g b`** (also `hsvillum …`, `hslillum …`) — the Jakob–Hanika
  *illuminant* upsample: the **emitter analogue of `rgb`**. Where `rgb` fits a bounded
  (0,1) reflectance under D65, `rgbillum` fits a smooth, full-spectrum **emission** SPD
  — modelled as `A·sigmoid(quadratic)` so the magnitude is unbounded — whose integral
  under the *bare* CIE observer reproduces the colour exactly (round-trips to <0.001 for
  every colour, including saturated primaries and white). Unlike `rgbline` this is a
  broadband source, not a monochromatic spike, so it reads as a natural coloured light
  rather than a laser line. Meant for **lights** (`spd rgbillum 1 0.6 0.2`); accepted
  anywhere a spectrum is.
- **`rgbsmits r g b`** (also `hsvsmits …`, `hslsmits …`) — the classic **Smits 1999**
  RGB→reflectance upsampler: the colour is decomposed additively over seven tabulated
  basis reflectances (white / C M Y / R G B) sampled at 10 wavelengths. A **selectable,
  lower-fidelity alternative** to the default `rgb` (Jakob–Hanika) fit — cheaper and
  historically standard, but rounds sRGB back to within only ~0.07 (vs `rgb`'s <0.001).
  Produces a valid `[0,1]` reflectance, so it's a *material* upsampler like `rgb`; offered
  for comparison / when a scene wants the Smits basis specifically.
- **`rgbbox r g b`** (also `hsvbox …`, `hslbox …`) — the simplest RGB→reflectance
  upsampler: a **calibrated 3-box** spectrum, one flat step per band (blue 400–500,
  green 500–600, red 600–700 nm) whose three heights are solved from a fixed 3×3 matrix
  so the reflectance integrates back to the requested linear-sRGB colour *exactly*
  (round-trips to <0.02 — the tightest of the reflectance upsamplers). Cheap and analytic
  but blocky (hard band edges, no smoothness), so it's the baseline against which `rgb`
  (Jakob–Hanika) and `rgbsmits` trade fidelity for smoothness. A valid `[0,1]`-clamped
  *material* reflectance like the others.
- **`rgbmeng r g b`** (also `hsvmeng …`, `hslmeng …`) — the **Meng 2015** upsampler: of
  all physical reflectances that produce the requested colour, take the **smoothest** one
  (the minimum of `Σ(s[i+1]−s[i])²`), read from a baked table over the sRGB chromaticity
  triangle. This is the **highest-fidelity** of the four: it round-trips sRGB to <0.0001
  *and* is provably smoother than the `rgb` (Jakob–Hanika) fit for the same colour, since
  smoothness is exactly what it minimises. That matters whenever a reflectance is seen
  under a strongly **non-D65** light or **dispersed** — all four upsamplers agree under
  D65 by construction, but only the reconstructed *shape* decides the colour under a
  tungsten or sodium source, and a smooth shape is what real pigments have. Slightly more
  memory than the analytic fits (a ~140 KB table) and a table lookup instead of a solve.
  A valid `[0,1]`-clamped *material* reflectance like the others.
- **`rgb:<name> r g b`** (also `hsv:<name> …`, `hsl:<name> …`) — **your own upsampler**.
  The five heads above are a closed set of built-in fits; this one names an
  `upsample "<name>" { expr "f(r, g, b, w)" }` block declared in the scene, so a scene
  that needs a mapping none of them provides supplies the function itself rather than
  picking the least-wrong built-in:

  ```
  spectrum "prim_r" = gaussian center=620 sigma=40
  spectrum "prim_g" = gaussian center=540 sigma=40
  spectrum "prim_b" = gaussian center=460 sigma=40
  upsample "basis" { expr "r*spec:prim_r(w) + g*spec:prim_g(w) + b*spec:prim_b(w)" }
  material "m" { type diffuse  reflect rgb:basis 0.25 0.55 0.85 }
  ```

  The body is a pattern-VM expression over a **disjoint** vocabulary — there is no hit
  point here, so it sees only `r`, `g`, `b` (the colour, linear sRGB), `w` (wavelength
  in nm), `pi`, the usual functions, and **`spec:<spectrum>(w)`**, which samples a
  declared `spectrum` block at the queried wavelength. That last one is what makes a
  **measured basis** expressible (weight real primary curves by the three channels)
  rather than only closed-form arithmetic; it is in scope *only* inside an `upsample`
  body, since an ordinary pattern has a hit point but no wavelength. Because `r` means
  RED here and radius in the surface vocabulary, every surface/shading variable is
  rejected **by name** with a message saying so, rather than silently reinterpreted.
  The triple is converted to linear sRGB *before* the body runs, so all three heads
  feed identical `r, g, b`. Evaluated per queried wavelength, not pre-tabulated, so a
  narrow emission line isn't quietly band-limited. Blocks resolve lazily by name
  (declaration order is irrelevant). Usable everywhere a colour head is, including a
  record channel's inline-colour tag. See `scenes/_upsample.ftsl`; pinned by
  `-checkupsample`.
- **`table { 400:0.05 450:0.12 … }`** — a measured/tabulated spectrum. Interpolated
  **piecewise-linear** by default; add an **`interp=cubic`** flag among the entries
  (`table { interp=cubic  400:0.05 … }`) for a **monotone cubic (PCHIP)** curve —
  C¹-smooth yet shape-preserving, so it never overshoots (a value stays within its
  neighbouring samples, so a reflectance/absorption can't ring negative the way a plain
  spline would). Cubic is worth it for **sparse** control points where linear kinks
  show; for dense data it matches linear. Outside the sampled range the value is held
  flat at the nearest endpoint (no extrapolation).
- **`file:<path>`** — load a measured curve (SPD, reflectance, or n(λ)) from an
  external CSV/whitespace data file (`#` comments, a header row, `wavelength_nm,value`
  rows); the runtime ingestion point for the data under `data/`. E.g.
  `spd file:data/illuminant/f2.csv` (see `scenes/measured_spd.ftsl`). Same
  interpolation controls as `table`: append **`interp=cubic`** for monotone-cubic
  (`absorb file:data/absorb/red.csv interp=cubic`), else piecewise-linear. If the file
  doesn't span the render's spectral range a one-line **coverage warning** is printed
  (the tails are held flat, not extrapolated).
- **`glass:<name>`** — dispersive index via Sellmeier: `BK7`/crown, `SF10`/flint,
  `silica`/fused-silica, `sapphire`, `diamond`, plus Cauchy fits for `water`,
  `ice`, `acrylic`/PMMA, `polycarbonate`.
- **`metal:<name>`** — measured metal reflectance: `Au`/`gold`, `Ag`/`silver`,
  `Cu`/`copper`, `Al`/`aluminium`, `Cr`/`chromium`, `brass`.
- **`reflectance:<name>`** — measured natural-material diffuse reflectances:
  `leaf`/`vegetation`, `skin`/`skin-light`, `skin-dark`, `snow`, `soil`/`dirt`,
  `brick`/`red-brick`, `concrete`.
- **`filter:<name>`** — gel/Wratten filter transmittances T(λ) (for a `filter`
  material's `transmit`): the **complete 84-filter Kodak Wratten set**, named
  `wratten-<n>` (e.g. `wratten-25`, `wratten-34a`, `wratten-47b`). Descriptive
  aliases resolve too: `red-25` (`red`), `deep-red-29`, `orange-21`, `yellow-12`,
  `green-58`, `blue-47`, `deep-blue-47b`, `magenta`, `cyan`, ….
- **`spectrum "name" { … }`** blocks to define and reuse a named SPD.

The `glass:`, `metal:`, `reflectance:`, `filter:` and `preset:` (illuminant) presets —
plus the whole-material `preset <name>` recipes and the named light presets — are a
**drop-in spectral asset library**: their data lives in external files under
`data/{glass,metal,reflectance,illuminant,filter,material,light}/`, loaded at runtime — add a
file to a category directory and it resolves by name with **no rebuild** (the
lowercased filename is the preset name; a `# aliases:` header line adds more). The
`material/` and `light/` files are *bundles* that group several envelopes plus scalars
into one named asset (a thin-film material owns an index curve, a substrate-extinction
curve and film thickness/index at once). Only the data is external; the dispersion
evaluators, interference/BSDF math and light models stay in the renderer. See
`data/README.md`.

### Spectral representation vs. other renderers

*How* a renderer carries colour along a light path decides whether it can split
dispersion correctly. There are three representations, and ftrace sits at the
physically-strictest end:

1. **RGB triple** — three channels ride every ray/photon. Cheap, but colour is
   already collapsed into R/G/B, so a dispersive interface (prism, lens, water)
   cannot fan wavelengths into different directions: no true dispersion.
2. **Co-sampled full spectrum** — one ray/photon carries *all* N spectral bins at
   once (a whole SPD per sample). Spectrally correct in energy, but because every
   wavelength rides the *same* photon it still physically cannot land in different
   places per wavelength — so **dispersive caustics through a photon map don't
   split** (they stay energy-correct but colour-averaged).
3. **One wavelength per photon** (ftrace) — each photon carries a single λ, refracts
   at *that* wavelength's index, and lands where *that* colour focuses. Dispersive
   caustics split into true spectral colour for free. The modern **hero-wavelength**
   schemes are the same idea softened: a few stratified wavelengths share one "hero"
   λ that drives the path (which is why they, too, can disperse).

**"Accurate" splits into two independent axes, and ftrace is only unique on the
second:**

- **Spectral energy accuracy** — right colour under odd illuminants, metamerism,
  saturated lights, fluorescence: everything RGB's three channels smear. *Every*
  full-spectrum renderer below is as accurate here as ftrace, and the hero-wavelength
  ones (PBRT-v4, Mitsuba 3) reach it with *less* noise by carrying four wavelengths
  per path. **We claim no edge on this axis** — and ftrace now closes the noise gap on
  the **CPU** tracers — the **backward reference tracer (`-mode R`)**, the
  **forward light tracers (`-mode A/B/C`)**, the **photon-mapping modes (`-mode M/S`)** and
  **BDPT (`-mode D`)** — plus the **GPU megakernel** (forward modes `A/B/C`, the `M`
  photon-map deposit, the backward reference `R`, and BDPT `D`)
  all use hero-wavelength sampling (a hero λ plus 3 stratified secondaries riding one
  shared BVH walk, secondaries de-hero'd at the first dispersive interface — Wilkie et
  al. 2014 / PBRT-v4 `TerminateSecondary`), so they reach a given colour-noise level in
  fewer samples (measured ~0.77× chroma-noise RMS at equal photons in the light tracers,
  ~0.87× in the photon map, ~0.77× in GPU mode `R`, ~0.80× in BDPT on a glass-sphere
  Cornell box — which de-heros at the glass — and as low as **0.38–0.49× chroma with
  0.55–0.63× luma** on saturated coloured interiors where the bundle rides the whole
  path) while dispersion stays bit-for-bit intact.
  For photon mapping each traced path deposits all its live wavelengths as per-λ photon
  records, so total stored energy is unchanged. In BDPT *both* subpaths carry the same
  bundle and every connection is evaluated per-λ with a single shared MIS weight (all the
  pdfs are hero-driven); glossy vertices are connectible there, so — unlike the
  unidirectional tracers — mode `D` keeps the bundle alive across glossy bounces, and it
  likewise rides straight through **mirrors** and **gel filters**, whose outgoing direction
  doesn't depend on λ either (worth **0.47× chroma / 0.57× luma** noise on a
  mirror-and-Wratten-gel box, where de-heroing at the first mirror had left hero buying
  almost nothing). Mode `D`
  is hero-capable on **both** backends, and the CPU and GPU BDPT agree to 0.03%.
  The **backward tracer (`-mode R`)** and the **forward tracers (`A/B/C`, and the `M`/`S`
  photon deposit)** do the same on both backends: mirrors, gel filters and glossy lobes
  keep the bundle, and — more importantly — every Russian
  roulette along the path survives on the *strongest* live wavelength rather than on the
  hero's own, so no wavelength is ever amplified by a `ρ(λᵢ)/ρ(λ_hero)` ratio. On
  saturated (strongly coloured) diffuse interiors that ratio used to cancel the whole
  benefit; mode `R` there went from "hero buys nothing in luma" to **0.42–0.52× noise
  RMS on luma *and* chroma** — a 4× variance cut for 1.35× the time, i.e. ~2.9× faster
  to a given noise level. The forward tracers gain less, because they share only the
  main path's BVH walk while the per-λ camera splat / photon deposit costs a full 4×:
  ~1.1× luma and 1.3–1.8× chroma at equal GPU time (1.3× / 2.1× on CPU). Before the
  fix hero was a net *loss* in both — up to 2.8× the luma noise of plain single-λ on a
  glossy/translucent interior.
  Hero collapses to a single continuous
  wavelength the instant dispersion matters, so it *keeps* the forward photon map's true
  caustic splitting (below) rather than trading it away. **VCM/UPS (`-mode U`)** carries the
  bundle too, on **CPU *and* GPU**: both subpaths of a path index are drawn from one bundle, so its
  vertex *connections* are exact per-λ like BDPT's, while its vertex *merges* — the one
  strategy that crosses paths — key off each stored light vertex's own wavelengths. Measured
  on a Wratten-58 gel + mirror box, **0.51× noise RMS** at equal passes, agreeing with the
  single-λ estimator to 0.02 %; on the GPU it is **0.72–0.82× chroma noise** for 1.5–1.7× the
  time across four stress scenes, and the two backends agree to 0.03 %. Still single-λ **by
  design or pending work**: the **GPU wavefront** backend (`-wavefront` — hero forces the
  megakernel). Scenes with
  participating media, a GRIN volume, or a finite-lens camera also stay single-λ everywhere.
  The bundle size is runtime-configurable with **`-heroc N`** (default 4, range 1–8);
  `-heroc 1` turns hero off, reducing every hero tracer (CPU and the GPU megakernel)
  bit-identically to the classic single-λ estimator. **Do not do that in mode `W`.** Mode
  `W` defaults to the widest bundle (`8`) because at 1 spp the `N` hero wavelengths *are*
  the whole spectral quadrature — there are no further samples to average a collapse away,
  so `-heroc 1` there does not make a dispersive surface noisier, it makes it **wrong**
  (a Cornell `glass:SF10` sphere renders a flat green ball: 46.85 pp of chroma error against
  0.82 pp at the default). It also buys almost nothing: mode `W` is traversal-bound, so its
  full 8-wavelength bundle measures at **2.7 %** of frame time over a single wavelength
  (versus 61 % in mode `R`). Passing `-heroc 1` to a batch mode-`W` render on a scene with a
  dispersive dielectric, thin film, grating, multilayer, layered coat or fluorescence now
  prints a warning naming the material.
  **`-herosplit`** changes the dispersive-event policy from terminate-secondaries to
  **split**: all `N` wavelengths carry on, each refracting along its *own* per-λ
  direction, so one bundle fans out into `N` monochromatic sub-paths through the glass.
  Both policies are unbiased and converge to the same image; splitting resolves the
  chromatic spread of a prism / rainbow / dispersive caustic *geometrically* instead of
  stochastically over many photons. On a dispersive `glass:SF10` flint-sphere Cornell box
  that is worth **0.70× the chroma noise and 0.89× the luma noise inside the caustic** (and
  0.80× / 0.92× over the whole frame) **at equal wall clock** — the extra traversal past
  the split is linear, not exponential (once monochromatic a sub-path never re-splits) and
  is paid only by the photons that actually reach the glass, so it costs just **1.11×** per
  photon there. It stays opt-in because that ratio is scene-dependent: a scene that is
  mostly glass pays much more of it. The backward tracer (`R` and `W`) honours the flag on
  **both the CPU and the GPU**; CPU forward modes `A`/`B`/`C` and the `M`/`S` photon deposit
  honour it too, while the GPU *forward* megakernel still de-heros and can adopt it later.
  **Mode `W` always splits, flag or no flag** — its λ lattice is shared by every
  pixel, so terminating the secondaries would mistint the *whole frame* rather than add
  noise. Splitting is what makes glass come out right at `-spp 1` there.
- **Dispersion — colours actually splitting** through a prism / lens / water. Only
  the single-λ (ours) and hero-wavelength (PBRT-v4, Mitsuba 3) schemes get this right;
  co-sampled spectral (PBRT-v3, Mitsuba 0.x) and every RGB pipeline cannot.

Where popular physically-based renderers fall (verified against their docs/source;
see sources below):

| Renderer (engine) | Default colour | Spectral mode | Per-path/photon carrier |
|---|---|---|---|
| **ftrace (this — forward photon)** | spectral | always | **hero wavelength, 4 λ/photon (CPU A/B/C, R, photon-map M/S, BDPT D & VCM U; GPU megakernel A/B/C, M, R, BDPT D & VCM U); 1 λ on the GPU wavefront only** — true dispersive caustics either way |
| PBRT-v3 (SPPM photon map) | RGB | compile-time (`SampledSpectrum`, ~30 bins @ 10 nm) | **co-sampled: all bins on one photon** — no split |
| Mitsuba 0.x (`ptracer`/`ppm`/`sppm`) | RGB | compile-time (`SPECTRUM_SAMPLES`, e.g. 15–30) | **co-sampled: all bins per sample** — no split |
| PBRT-v4 | spectral | always | hero wavelength, 4 λ/path (default, recompilable) |
| Mitsuba 3 (`*_spectral` variant) | RGB build variant | build variant | hero wavelength, 4 λ/ray |
| Maxwell Render | spectral | always | full-spectral transport¹ |
| Indigo Renderer | spectral | always | full-spectral transport¹ |
| LuxCoreRender | RGB (sRGB) | on-demand only (dispersion) | RGB; spectral only at a dispersive glass event |
| Blender Cycles | RGB | fork/experimental only | RGB |
| Arnold | RGB | — | RGB |

The **two forward light tracers that carry every wavelength on a single photon** are
the *spectral* builds of **PBRT-v3** (`SampledSpectrum`) and **Mitsuba 0.x**
(`SPECTRUM_SAMPLES` > 3): both are RGB by default and, even compiled spectral,
co-sample the whole SPD per photon — so neither reproduces colour-split dispersive
caustics in its photon map.

To be clear, **ftrace is not uniquely spectrally accurate** — Maxwell, Indigo and the
spectral builds of PBRT/Mitsuba integrate the true spectrum just as faithfully (and
PBRT-v4 / Mitsuba 3 do it with *less* colour noise). What is unique here is the
**pairing**: ftrace is the only renderer in this table that couples *accurate
single-wavelength photons* with a *forward photon map*, so true dispersive **caustics**
— focused, colour-split light, a rainbow thrown through a glass of water onto a table —
fall straight out of the forward pass. Pure path tracers (PBRT-v4, Mitsuba 3) disperse
a directly-seen ray correctly but struggle with *caustics* regardless of how good their
spectral model is; the fully-spectral bidirectional/MLT tracers (**Maxwell**, **Indigo**)
reach caustics by a different, costlier route; **LuxCoreRender**, **Cycles** and
**Arnold** are RGB pipelines (LuxCore invokes a single wavelength only at a dispersive
glass hit). ftrace pays for the combination with more photons for chromatic smoothness —
the price of one wavelength at a time.

¹ Maxwell and Indigo document spectral transport end-to-end, but do not publicly
specify whether a path samples one wavelength or co-samples many — so their
per-path carrier is left unqualified here.

*Sources:* [pbrt-v4 spectral representation](https://pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Representing_Spectral_Distributions),
[Wilkie et al. 2014, *Hero Wavelength Spectral Sampling*](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.12419),
[Mitsuba 3 spectral variants](https://mitsuba.readthedocs.io/en/latest/src/key_topics/variants.html),
[Indigo — spectral throughout](https://indigorenderer.com/features),
[Maxwell Render features](https://maxwellrender.com/features/),
[LuxCoreRender — spectral on demand](https://forums.luxcorerender.org/viewtopic.php?t=1728),
[Cycles — RGB (spectral is a fork)](https://devtalk.blender.org/t/thoughts-on-making-cycles-into-a-spectral-renderer/2192).

---

## Lights

`light <subtype> { … }`:

| Subtype | Description | Key parameters |
|---|---|---|
| *(default)* area | Rectangular area light | `origin`, `u`, `v`, `normal`, `spd` |
| `sphere` | Spherical area light | `center`, `radius`, `spd` |
| `cylinder` | Cylindrical tube light | `center`, `axis`, `length`, `radius`, `caps`, `spd` |
| `spot` | Cone spotlight with penumbra | `origin`, `dir`, `inner_angle`, `outer_angle`, `spd` |
| `collimated` | Parallel beam (3 cm pencil, ×enclosing group scale), centered on `origin` | `origin`, `dir`, `spd` |
| `sun` | Distant directional sun (parallel beam over the whole scene, soft-edged disc) | `elevation` + `azimuth` (or `dir`), `angle`, `spd`, `intensity` |
| `env` | Environment / IBL light | `file` (lat-long HDR) or `spd`, `rotate`, `intensity`, **or `sky`** (analytic sky, below) |

**Distant sun.** `light sun { … }` is a first-class **directional** emitter: an
infinitely-distant disc of angular diameter `angle` (degrees; default `0.53`, the real
sun) whose rays arrive parallel. Aim it with `elevation <deg>` + `azimuth <deg>`
(azimuth from +x toward +z, the same convention as the sky block) or a raw
`dir <x y z>` pointing *toward* the sun.

Its `spd` is the **perpendicular spectral irradiance** — the light falling on a surface
that faces the sun — so a grey Lambertian floor reads exactly `ρ/π · E⊥ · cos θ`, and
widening `angle` softens the shadow penumbra **without changing the exposure**
(verified: 0.53° → 8° shifts the lit-floor level by 0.02%). Because a distant light's
total flux depends on the scene's cross-section rather than the light, absolute
`power` / `lumens` is refused here; scale with `intensity <s>` instead.

Unlike every other light, the sun costs nothing in forward modes: photons are born on a
disc the size of the scene's own cross-section, aimed down the beam, so **every** photon
enters the scene instead of most missing it. Backward modes next-event-estimate it
inside its cone, and the disc itself is directly viewable (aim a camera at it). Runs on
both CPU and GPU in every mode: A/B/C/R/P/M/S all along, mode `D` (BDPT) since 0.124.0 on the
CPU and 0.126.0 on the GPU (together with `spot`), and mode `U` (VCM) since 0.125.0 on the CPU
and 0.127.0 on the GPU. No mode falls back to the CPU for a sun any more.
See `scenes/_sun_check.ftsl` and `scenes/_deltalight_mix.ftsl`, and `ftrace -checksun` for the deterministic self-test
(cone solid angle, exposure invariance, uniform-in-solid-angle cone sampling, and
NEE/direct-view rim agreement).

**Analytic physical sky.** An `env` light can synthesise a **Preetham daylight sky**
instead of loading an HDRI — write `sky preetham` (or just supply `turbidity` /
`sun_dir` / `sun_elevation`, which implies it). ftrace bakes an equirectangular sky
image from the [Preetham et al. 2002] analytic model — a blue dome that whitens toward
the horizon and the sun, plus a **spectrally attenuated solar disk** (a 5778 K blackbody
extinguished by Rayleigh + Ångström-aerosol optical depth over the sun's air mass, so a
low sun reddens into a proper orange sunset) — and feeds it through the same `EnvMap`
pipeline as an image env, so it importance-samples, upsamples to spectra, and runs on
both the CPU and GPU exactly like an HDRI. Parameters: `turbidity <t>` (≈2 clear
deep-blue … ≈10 hazy/milky, default 2.5), the sun via either `sun_dir <x y z>` or
`sun_elevation <deg>` + `sun_azimuth <deg>` (azimuth from +x toward +z), `ground_albedo
<a>` (tints the below-horizon hemisphere, default 0.3), `intensity <s>` (scales the
normalised mean sky luminance, default 1), `res <px>` (equirect width, height = res/2,
default 1024), and `rotate <deg>`. Because the solar disk is physically ~10⁵× brighter
than the sky (as in any real sunny HDRI), sun-lit diffuse surfaces are high-dynamic-range
and benefit from a generous photon budget / higher `-noise` target in forward modes; the
sky background itself is read directly and is noise-free. See `scraps/sky_test.ftsl`
(daytime) and `scraps/sky_sunset.ftsl` (low-sun reddening).

**`sun_disk on | off | separate`** controls how that solar disk is delivered. The
default `on` bakes it into the equirect map (accurate but slow to converge in forward
modes, since a photon must randomly land on a disc covering ~10⁻⁵ of the sphere).
`separate` instead strips the disk out of the map and registers an equal-energy
`light sun` alongside the skylight dome — same picture, but the sun is now
parallel-emitted and cone-NEE'd, which is dramatically faster to converge. On
`scraps/sky_test.ftsl` (mode B, GPU): **baked** reached only 7.2% noise after 30 s /
2×10⁹ photons and the image still showed *no* warm sunlight and *no* cast shadows (the
sun had barely been hit at all), while **`separate`** hit the 4% target in **5.5 s /
3.0×10⁸ photons** with the sun fully formed — about **20× fewer photons** for the same
noise, and a qualitatively correct picture instead of a skylight-only one. `off` drops
the disk entirely (skylight only). The `separate` split is energy-matched to the baked
profile (measured agreement 0.12% once the map resolves the disc).

**Absolute power.** Any non-env light may author a real physical output —
`power <watts>` (radiometric radiant flux) or `lumens <lm>` (photometric luminous
flux, via `Φᵥ = 683·∫spd·V dλ`) — instead of relying on the per-image
auto-exposure. Authoring either on *any* light puts the whole scene in **absolute
mode**: the film is physically linear, the auto-exposure is replaced by a fixed
sensor gain, and `iso`/`shutter`/`exposure` become true absolute stops (doubling
`power` is exactly one stop brighter). See `scenes/absolute.ftsl`.

**Emissive meshes (mesh area lights).** Any material may carry an **`emit <spd>`**
spectrum (e.g. `material "glow" { type diffuse reflect rgb 0.02 0.02 0.02 emit
preset:bb6500 }`). A `mesh` bound to such a material becomes a real **area light**:
every triangle it appended radiates `emit(λ)` from its front face, and the whole
triangle soup is registered as one emitter that next-event estimation, forward
photon emission and BDPT all sample uniformly by triangle area (pick a triangle from
a cumulative-area CDF, then a barycentric point; pdf = 1/total-area, the same law as
a quad light). So an arbitrary glowing shape — a torus ring, a tessellated logo, an
imported OBJ — lights the scene like a physically-sized luminaire, on both CPU and
GPU. An optional `power`/`lumens` on the mesh block rescales the SPD to that flux
over the mesh's total area (cloning the material so a shared material isn't
disturbed). Emission is **one-sided** (front face only); a closed shell whose
triangles happen to be wound *inward* (common in imported OBJs) would otherwise
radiate into its own interior and look black, so such shells are auto-oriented
outward at load — the loader flips the winding of any emissive mesh that encloses a
volume with inward orientation, leaving flat/open sheets (which enclose no volume)
exactly as authored. Emissive meshes count as lights, so a scene lit *only* by one
needs no separate `light` block. (Meshes that import their own materials — glTF/GLB —
are not auto-lit; bind an FTSL `emit` material instead.)

**Emissive non-mesh geometry (glowing solids).** `emit` is a property of the
*material*, not of the `mesh` block, so binding an emissive material to anything else —
a `sphere`, a `quad`, a CSG solid, a marched `isosurface`, a `curve` fiber — makes it glow
too, identically on CPU and GPU. The difference is that only a mesh has triangles to
register an emitter against, so these surfaces are seen by **emission-on-hit only**: a
camera ray (or a specular bounce) that lands on one picks up `emit(λ)`, but NEE and
light subpaths cannot sample points on them, which means *they do not illuminate the
rest of the scene* and contribute nothing in the forward modes `A/B/C`. Treat them as
self-luminous **appearance**, not as luminaires, and keep a real light in the scene to
do the actual lighting — see `tools/loom/examples/glowing_jack.py`, where a
gyroid-carved isosurface glows from inside its own filigree while a ceiling panel
shades it.

**Emission is one-sided — mind a `quad`'s winding.** A surface shows its `emit` only to a
ray arriving on the *front* face, and a `quad`'s front face is the side `cross(u, v)`
points at. So

```ftsl
quad { origin -18 0 -19  u 46 0 0  v 0 0 45  material glowgrid }   # normal -y: dark from above
quad { origin -18 0 -19  u 0 0 45  v 46 0 0  material glowgrid }   # normal +y: glows
```

differ only in the order of `u` and `v`, and only the second one lights up. Every
*reflective* slot looks identical either way — direct lighting flips the normal toward the
light itself — so a mis-wound emissive quad shades perfectly normally and silently loses
just its glow. If a glowing panel or floor renders as a plain lit surface, swap `u` and `v`
first.

---

## Geometry

`sphere`, `quad` (parallelogram), `triangle`, `curve` (a hair/fur/wire strand — see
**Curves and fibers** below), and `mesh` (**OBJ, glTF 2.0 / GLB,
Autodesk FBX, and `.ftmesh`** import — the loader dispatches on file extension). glTF brings
its node transform hierarchy, per-vertex normals/UVs, and `pbrMetallicRoughness`
materials (base color upsampled to a reflectance spectrum, metallic → glossy tint,
roughness → lobe width; `import_materials no` forces the FTSL `material` instead).
`skip_material <substr>[,…]` (repeatable) drops glTF primitives whose material name
matches — the way to strip the ground plane / studio backdrop that asset-store models
bundle in with the subject, since geometry can't be subtracted after it loads.
**FBX** (`.fbx`, via the vendored MIT/public-domain [`ufbx`](https://github.com/ufbx/ufbx)
library) imports baked triangle geometry — every mesh instance's faces are
triangulated and baked through ufbx's world transform, with generated-if-missing
per-vertex normals and the first UV set filling the same smooth-shading / texturing
slots the OBJ/glTF paths use; the scene is normalized to right-handed Y-up metres at
load. (FBX materials, skinning, blend shapes and animation are not yet consumed — see
known-issues.) OBJ
supports `usemtl use_names` for per-face materials and `uv use_mesh` for mesh UVs.
OBJ **vertex normals (`vn`) are read as smooth shading normals** — a hit
barycentric-interpolates them (CPU and GPU) for smooth-shaded curved meshes, with
no visible faceting; a mesh with no `vn` stays exactly flat-shaded (geometric
normal). For low-poly OBJs that ship **no** `vn`, opt into **crease-angle
auto-smoothing** with `mesh { smooth [<deg>] }` (default `40°`): the loader welds
coincident positions (so split-vertex exporters still smooth), then synthesizes a
per-corner shading normal as the **angle-weighted** average (Thürmer & Wüthrich) of
the adjacent faces whose dihedral angle is **below** the threshold — so a sphere's
gentle facets fuse into a smooth gradient while a cube's 90° edges stay crisp.
Only the shading normal is affected; the silhouette stays true to the geometry.
(Smooth shading of interpolated normals — both authored `vn` and crease-smoothing —
is faithful in **all** render modes: the backward reference `R`, and the forward
tracers `A/B/C/D/M/S/U` (CPU and GPU), which apply Veach's shading-normal adjoint
correction so the light/particle transport smooth-shades to match the reference.
Light connections are clamped to the geometric hemisphere so a smoothed normal never
leaks light through the true back face, and the terminator where light grazes off is
softened (Chiang et al. 2019) so low-poly smooth meshes show a smooth shadow gradient
instead of hard facet slivers — applied uniformly to every mode including `R`. Flat
meshes are unaffected — both the correction and the softening are exactly a no-op when
the shading and geometric normals coincide.)
**`.ftmesh`** is ftrace's own compact binary mesh: a 24-byte header followed by
little-endian `f32` positions (plus optional normals and UVs) and `u32` triangle
indices. It exists for the **loom live viewer**, which re-derives and reloads a mesh
on every frame — parsing that as text was the single largest asset cost, and a binary
blob loads ~2.4× faster overall (~6.5× on read+decode alone) at about half the file
size. It is also *more* precise than the OBJ text it replaces, which went through
`%.6g` (6 significant digits, against `f32`'s ~7.2). Write one from Python with
`loom.ftmesh.write_ftmesh(path, verts, faces)`, or have any loom scene emit them by
passing `mesh_format="ftmesh"` to `Scene.emit` (the live viewer channel already
defaults to it — and now sends those bytes straight down its pipe rather than via a
file, so the format doubles as the live wire format). `Scene.emit` also takes a
`mesh_sink=` dict, which collects the encoded meshes in memory instead of writing
them while leaving the emitted scene text byte-for-byte identical — that is the hook
the live channel uses. `mesh { smooth … }`, `uv …`, transforms and materials all behave
identically to the OBJ path — both formats share the same normal-synthesis code.
Meshes without their own `vt` coordinates can be textured via a procedural
projection — `mesh { uv planar|spherical|cylindrical [x|y|z] }` synthesizes UVs
at load time from the mesh's world-space bounding box (the optional token is the
projection/up axis, default `y`).
`group { translate … rotate … scale … shear … <children> }` composes transform
hierarchies (baked to world space at load). Children may be `sphere`, `quad`,
`triangle`, `mesh`, `mesh_instance`, `isosurface`, `curve`, `light`, or nested `group`s —
so a physically-settled rest pose (e.g. from `tools/settle_scene.py`) can wrap an
isosurface CSG/implicit just as easily as a mesh. `shear <a> <b> <c>` adds a
unit-diagonal upper-triangular skew (`x' = x + a·y + b·z`, `y' = y + c·z`) to the
group's affine, applied in the group's local frame (innermost, before scale/rotate);
it lets quad/tri/mesh geometry be sheared into parallelograms. A `sphere` under a
**non-uniform scale or shear** (which the analytic ray-sphere can't represent) is
**automatically tessellated** into a smooth-normal ellipsoid / sheared quadric mesh
at load — so squashed and skewed spheres just work; a uniform-scaled sphere keeps
the fast analytic path.

**Instancing.** `mesh_asset "name" { file … material … }` loads a mesh once into
its local space; `mesh_instance { of "name"  translate … rotate … scale …
[material …] }` places that shared geometry through a per-copy affine. Instances
share one triangle set and one bottom-level BVH — a **two-level BVH** (TLAS over
instances → shared BLAS) — so N copies cost N affines instead of N triangle sets,
and a per-instance `material` can override the asset's own materials. Works in
every render mode, and the memory sharing holds on **both** the CPU and the GPU: the
device also uses a true two-level BVH (shared per-BLAS pools + an instance table that
transforms the ray into BLAS space), so device memory scales with unique geometry, not
with the instance count. Everything is accelerated by a BVH.

**Watertight ray–triangle test.** Triangles are intersected with the Woop/Benthin/Wald/Áfra
watertight test (JCGT 2013) rather than Möller–Trumbore. A ray through a shared edge is
claimed by *exactly one* of the two triangles that meet there, so closed meshes render with
**no grazing-edge cracks** (background pixels leaking through a silhouette) and no dropped
hits. This holds on both the CPU double path and — where it matters most, since floating-point
edge signs are what used to crack — the GPU float path. (v0.116.0 fixed a real crack on the
GPU: the guarantee needs the edge functions' two products to stay *unfused*, and nvcc's
default fused multiply-add was silently breaking the antisymmetry, so a mesh edge that landed
dead on a column of pixel centres let the background through. See `known-issues.md`.)

### Curves and fibers (`curve`)

A `curve` is a **strand**: control points plus a radius, for hair, fur, grass, wire,
thread and cables. It exists because the alternative is untenable — a triangle ribbon
costs ~64 triangles per hair, so one furred animal (1–10 M hairs × 8–32 segments) is
**10⁸–10⁹ triangles**. A strand instead costs a handful of segments.

```
curve "guide_hair" {
    material gold
    basis      catmull_rom       # linear | catmull_rom | bezier | bspline
    radius     0.016             # root radius (default 0.001 = 1 mm)
    radius_tip 0.002             # tip radius (default = radius, i.e. untapered)
    segments   12                # round cones per span (default 4; linear forces 1)
    point 0.30 0.02 0.55
    point 0.36 0.24 0.48
    point 0.27 0.46 0.60  r=0.028   # `r=` overrides the taper at one point
    point 0.35 0.68 0.50
    point 0.29 0.88 0.56
}
```

**How it is traced.** ftrace **flattens the curve at load time** into a chain of
**round cones** — each the convex hull of a sphere at either end, i.e. a capsule whose
radius varies linearly (`src/curve.h`). The BVH indexes one leaf per *round cone*, not
per strand, because a whole hair's bounding box is mostly empty space. Flattening up
front rather than evaluating the basis per ray means exact leaf bounds ("bound what you
test"), no basis evaluation in the inner loop, and a POD segment record that a future
GPU port can upload directly.

Because adjacent round cones **share their end sphere**, the chain is watertight and
smooth at the joints with no mitre logic: a strand is a single closed surface however
sharply it bends. That is why the `linear` zig-zag in `scenes/curve_basics.ftsl` has
no cracks at its corners.

**Bases.** All four are affine-invariant, so a `group { translate … rotate … scale … }`
transforms the control points and the flattening is exact; the radius picks up the
group's uniform scale (a non-uniform scale warns and uses the volume-preserving
geometric mean, since a round fiber cannot become elliptical).

| `basis` | points needed | spans | behaviour |
|---|---|---|---|
| `linear` | ≥ 2 | `n−1` | the control points **are** the polyline; `segments` is forced to 1 |
| `catmull_rom` *(default)* | ≥ 2 | `n−1` | **interpolating** — the strand passes exactly through every point you author, so it is the right basis for a hand-placed guide hair. Ends are clamp-duplicated |
| `bezier` | `3k+1` | `(n−1)/3` | cubic chain, `P0 C C P1 C C P2 …` |
| `bspline` | ≥ 4 | `n−3` | **approximating** and C2 — smoother than its control polygon and does *not* pass through the points; the right basis for a groom solver's output |

Radius is interpolated **linearly** between a span's two endpoint control points, never
through the basis — a Catmull-Rom radius can overshoot, and a negative radius is not a
taper, it is a bug.

**Texture coordinates.** The hit's `u` runs `0` (root) → `1` (tip) along the strand and
`v` runs around its circumference, so a `pattern` or texture can vary along or around a
fiber (`scenes/curve_basics.ftsl` bands one strand with
`expr "0.5+0.5*sin(2*pi*9*u)"`). `tangent` is the strand's axis, Gram-Schmidt'd against
the shading normal — the frame an anisotropic or fiber BSDF wants.

**Correctness.** `ftrace -checkcurve` runs six sections: the round-cone intersector
against Inigo Quilez's exact analytic SDF (position, first-hit `t`, normal vs. the
numerical SDF gradient, `u`/`v` ranges, and AABB containment); the degenerate case where
one end sphere swallows the other, which must equal the analytic ray–sphere test
exactly; `anyHit` agreement with the full path including origins *inside* the fiber;
watertightness at chain joints; basis flattening (span counts, monotone `u`, chain
contiguity, and that interpolating bases hit their control points while `bspline` stays
off them); and the **fp32 conditioning** of the quadric at fiber scale, which the CUDA
path depends on (see below).

**Numerics.** The round-cone quadric's constant term is a difference of two large,
nearly equal products. At fiber scale — a 1 mm radius on a 1 cm segment viewed from 2 m
— those products are ~4·10⁻⁴ while their difference is ~10⁻¹⁰: **six decades**, i.e. the
entire fp32 mantissa. So before forming the quadric, both the host and the device slide
the ray origin along itself to its closest approach to the segment's root point, and undo
that shift on each accepted root. This is an exact re-parameterisation — it changes only
the *conditioning* — and it is what makes the intersector usable in the fp32 CUDA
megakernel. Without it, 12–36 % of fiber hits are lost outright and the rest land tens of
radii off (strands render as speckled holes on the GPU while looking perfect on the CPU);
with it the fp32 error stays under 0.06 radii. `-checkcurve` §6 instantiates the real
intersector at `float` and guards exactly this.

**Limits (v1).** Curves run on **both the CPU and the GPU** in the ray-traced modes as of
0.151.0 (measured 74× on a 96 000-segment fur patch), and the `-raster` / `-raster-gpu`
previews show strands too (each round cone is meshed at preview fidelity). What is still
missing is the *aggregate LOD*: every fiber is intersected individually, so a groom's cost
is linear in strand count and sub-pixel fibers are a variance sink. See `known-issues.md`,
which also covers the per-segment azimuthal `v` frame and `CurveSeg`'s memory footprint.

**Authoring one procedurally.** The bundled Loom toolkit emits `curve` blocks from an
animated spine: `Strand` / `strand()` / `hair()` (`tools/loom/loom/scene.py`). It samples
the spine per frame and picks the basis for you — `catmull_rom` for an open spine (it
interpolates, so the fiber passes through the samples), and for a *closed* spine a
**periodic** `bspline` whose control points are solved so the loop is C2 through the seam
*and* still on the spine. That is worth knowing even if you never use loom: it is the
general recipe for closing a loop with this primitive, since `catmull_rom`'s
clamp-duplicated ends cannot. See `tools/loom/DESIGN.md` §7a′ and
`tools/loom/examples/strand_loop.py`.

### Grooms (`fur`)

A `curve` is one strand, written out by hand. A coat is 10⁴–10⁶ strands, which nobody
authors as text — so a **`fur { }`** block generates them (`src/fur.h`), scattering roots
over a named surface and growing each one with a closed-form shape:

```
sphere "ball" { center 0.22 0.20 0.55  radius 0.13  material skin }

fur "ball_coat" {
    on "ball"                 # a named sphere, mesh, quad or triangle
    material brown
    count 60000               # or `density <n>` — strands per authored unit^2
    length 0.055  length_jitter 0.35
    radius 0.0006             # radius_tip defaults to 0.25*radius
    droop 0.55  jitter 0.25  seed 1
}
```

The full key list is in FTSL.md → **§8.7 `fur`**. The shaping controls are `lift` /
`jitter` (growth direction), `direction` + `comb` (combing and wind), `gravity` + `droop`
(sag), `curl` + `curl_freq` (a helix), and `clump` + `clump_size` (tufts).

**`bald` keeps features bare.** A coat is grown per body *part*, but an eye or a nose is a
separate little sphere sitting *on* that part, and the part's groom roots area-uniformly
over the ring of skin it overlaps — so every strand rooted around the eye grows straight
across the eyeball, and no `length`/`lift`/`comb` fixes it, because the problem is where
the roots are, not which way they point. `bald "eye_l" 0.001` (a named scene sphere plus an
optional margin, or an explicit `<x> <y> <z> <r>`; repeatable) culls any strand that
reaches into that volume. The test is span-wise over the *whole* strand and runs *after*
clumping, so a hair that merely arcs through under `droop`/`comb`/`clump` goes too;
survivors are untouched, so the coat outside a zone is bit-for-bit unchanged and only the
count drops. The load line reports the cull.

**It is not a new kind of geometry.** The generator emits exactly the `Curve`/`CurveSeg`
records a hand-written `curve` produces, so the BVH, the CPU tracer, the CUDA megakernel
and the raster preview need no new code — everything above about round cones, bases,
taper, watertight joints and the `u`/`v`/`tangent` frame applies to generated strands
unchanged, and a groom inherits the GPU speedup for free.

**Area-uniform roots.** Over a mesh, a triangle is picked proportional to its area and the
barycentrics are sqrt-warped, so a low-poly belly and a dense face grow the same hairs per
square centimetre and `density` is meaningful. A `sphere` target is not tessellated at all
— roots land on the analytic surface with the analytic normal, so a furred ball has no
faceting in its coat and no tessellation bias in its density.

**Deterministic and parallel.** A strand is a pure function of `(surface, parameters, seed,
index)` — no simulation, no solver, no neighbour queries — so the build is a lock-free
`parallelFor` whose result does not depend on how the work was scheduled, and the same seed
always rebuilds the same coat (which is what a checkpoint/resume, a flyby, or a CPU-vs-GPU
comparison all quietly depend on). Growth is linear in the arc parameter `t` and every bend
is quadratic in it, so the root leaves the skin along its growth direction while the tip
carries the full displacement.

**Clumping changes shape, never density.** Strands blend toward their **nearest** tuft
guide with weight `clump · t`: roots stay exactly where the area-uniform sampler put them
while tips converge. Guides are located through a CSR uniform grid rather than by hashing
the root cell — hashing produces cube-shaped tufts on a grid, nearest-guide produces
Voronoi tufts, which is what hair does.

**Correctness.** `ftrace -checkfur` runs eight sections: roots lying on the target surface
(off-plane distance, in-triangle containment, sphere radial error); area-uniformity (a 3:1
area split must produce a 3:1 strand split, the mean barycentric must be ⅓ rather than the
½ an unwarped map gives, and `density × area` must be the exact count); determinism, with
the **two** consumers of the seed checked separately — the per-strand rng with clumping off,
and the guide rng by forcing the guide count to one at full clump strength, so every tip *is*
that guide's tip; growth direction never pointing into the skin, with lengths inside the jitter window
and the shaped arc inside its analytic bound; clumping collapsing tip spacing while moving
no root; the emitted chain being well-formed (segment count, contiguity, monotone taper and
`u`, correct back-pointers); a regression section for the load-order trap below; and `bald`
zones — no surviving segment inside a zone, survivors bit-for-bit unchanged from the same
seed grown without one, and a zone covering the whole target leaving nothing. Each
section is mutation-tested — a deliberate break in `fur.h` must make the section that owns
it fail — and that mutation run has twice caught a test that could not fail. It is why the
determinism section splits the two seed paths (with clumping on, either path alone moving is
enough to pass, so a build that had stopped seeding the *strands* from `spec.seed` slipped
through), and why the `bald` section grows a second zone that **floats clear of the skin**,
containing no roots at all: the original zone sat on the surface and swallowed the roots of
everything crossing it, so culling by root alone — exactly the bug `bald` exists to prevent —
passed it.

**Load-order trap (fixed, and pinned).** The deferred `fur` sweep runs during scene
*loading*, but `Tri::finalize()` — which computes geometric normals and back-fills absent
shading normals — does not run until `Scene::build()`. So the generator sees zeroed normals
on every quad, triangle and mesh without authored `vn`. Normalizing that zero vector made
every strand NaN, and the NaN was then swallowed by the flattener's coincident-point guard,
so a groom emitted **zero strands with no error printed anywhere**. The generator now
derives the geometric normal from the vertices itself and uses shading normals only when
they are genuinely present; `-checkfur` §7 builds on deliberately un-finalized triangles to
keep it that way.

**Limits.** The generator's ceiling is deliberate: no collision, no styling curves, no
interactive brushing. It also inherits the curve limits above — most importantly the
missing aggregate LOD, which is what decides how large a groom stays affordable. A
`mesh_instance` cannot be a target (its triangles live in a BLAS, not in world space);
see `known-issues.md`.

Worked examples: `scenes/fur_basics.ftsl` isolates each parameter; `scenes/fur_creature.ftsl`
is the payoff — an animal built from overlapping analytic spheres wearing ~300 000 strands
from a dozen `fur` blocks at one shared `density`, which is also a demonstration that fur
hides the seams of the geometry underneath it.

### Implicit surfaces (`isosurface`)

Besides the explicit primitives above, geometry can be defined *implicitly* as the
zero set of a signed-distance field and rendered by **sphere-tracing** (see
`src/implicit.h`). An `isosurface { material <m>  <one field element> }` block contains
exactly one root **field element**, which is either a **leaf** primitive or a **CSG
combinator** whose children are themselves field elements:

| Leaf | Parameters |
|---|---|
| `sphere` | `center`, `radius` |
| `ellipsoid` | `center`, `radius <rx> <ry> <rz>` (a non-uniformly scaled sphere) |
| `box` | `center`, `size <x> <y> <z>`, `round` (corner-rounding radius, 0 = sharp) |
| `torus` | `major`, `minor` (ring / tube radii; axis = local +y) |
| `cylinder` | `radius`, `height` (axis = local +y) |
| `cone` | `radius` (bottom), `radius2` (top, 0 = pointed), `height` |
| `plane` | `normal`, `offset` |
| `function` | `expr "f(x,y,z)"` — arbitrary formula leaf (see below) |

| Combinator | Meaning |
|---|---|
| `union` / `intersect` / `difference` | hard boolean CSG (min / max / subtract) |
| `smooth_union` / `smooth_intersect` / `smooth_difference` | filleted boolean; blend radius `k` softens the seam |
| `blob` | alias for `smooth_union` — with `k`, children fuse like **metaballs** |

Every element (leaf *or* combinator) may carry its own `translate` / `rotate` / `scale`.
To rotate a leaf **in place**, position it with `translate` (applied outside the
rotation) rather than `center` (applied inside — it would orbit the world origin).
Non-uniform scale is supported (the field stays a valid Lipschitz-1 SDF by de-rating
the step to the smallest axis scale), so an `ellipsoid`, a squashed `box`/`torus`, etc.
all work. Surface normals come from the analytic field gradient. A worked example with
metaballs, drilled CSG, and a tilted torus is in `scenes/implicit.ftsl`. Implicit
surfaces are sphere-traced on **both the CPU and the GPU** (the device port matches the
CPU to Monte-Carlo noise).

#### Arbitrary-formula isosurfaces (`function`)

The analytic leaves above are all built-in signed-distance fields. To render the
surface of an **arbitrary equation** `f(x,y,z) = 0` — a gyroid, a Goursat/heart shape,
any hand-typed formula — use a `function` leaf:

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.45           # (optional) place / rotate / scale the field frame
        expr "sin(28*x)*cos(28*y) + sin(28*y)*cos(28*z) + sin(28*z)*cos(28*x)"
    }
    contained_by { min 0.3 0.3 0.25   max 0.7 0.7 0.65 }   # REQUIRED bound box
    max_gradient 48                       # (optional) Lipschitz bound; auto-estimated if omitted
    accuracy 1e-4                         # (optional) march-step floor, world units
}
```

The `expr` string is compiled by the **same math VM as procedural patterns** (variables
`x y z` and `r = |p|`, plus `sin cos tan exp log sqrt abs floor fract sign min max pow
atan2 clamp mix smoothstep noise`, the vector-noise components `dnoisex/y/z` /
`dturbx/y/z` for gradient-noise domain warping, cellular noise
`worley/worley2/worleyd/worleyid(x, y, z, metric)`, blue-noise placement
`bnoise/bnoise2/bnoised/bnoiseid(x, y, z, r)`, and the constant `pi`). Because an arbitrary field is
**not** a signed distance and has no analytic bound, a `function` isosurface **must**
supply a `contained_by { min <x y z>  max <x y z> }` box (the region the surface is
marched inside). Safe sphere-tracing needs a **Lipschitz bound** `L ≥ max|∇f|` so a step
of `|f|/L` never overshoots the first zero crossing; give it explicitly with
`max_gradient`, or omit it and the loader auto-estimates it by sampling `|∇f|` over the
box (padded ×1.3). `accuracy` overrides the march-step floor. A `function` leaf also
composes inside CSG combinators (`union`, `difference`, `blob`, …) like any other leaf.
The worked gyroid example is in `scenes/function.ftsl`; expression isosurfaces run on
**both the CPU and the GPU** (the device evaluates the identical formula VM — an
expression sphere matches the analytic `sphere` leaf to RMSE ≈ 0.15 % on the same
backend).

**Container shape and caps.** The container can be a box **or a sphere**, and you can
choose whether the container *seals* the solid it cuts:

```
isosurface {
    material gold
    function { expr "f_enneper(x, y, z, 1)"  scale 1.7  translate 0 1.2 0 }
    contained_by { sphere { center 0 1.2 0  radius 1.7 } }   # curved boundary
    max_gradient 20
    open                                                     # (optional) don't cap
}
```

`contained_by { sphere { center <x y z>  radius r } }` clips the ray along a **smooth
curved boundary** instead of the axis-aligned `min`/`max` box, so an *unbounded* surface
(e.g. `f_enneper`, or a solid that pokes out of the container) reads as a natural rounded
edge rather than hard box facets. The sphere `center`/`radius` are taken in the field's
frame and transformed to world (exact under uniform scale; a conservative bounding sphere
under rotation/shear). Where the container wall slices through **solid** material
(`f < 0`), it is **capped** by default — sealed with a flat/curved face of the isosurface
material (a cleanly sawn-off solid). The **`open`** keyword suppresses those caps, leaving
the surface's cut edge and a see-through opening into the interior (`open off` forces the
default). Caps only affect surfaces that actually reach the container wall; a fully
bounded surface never touches it, so the choice is moot. Both the container shape and the
cap policy run identically on CPU and GPU.

##### POV-Ray internal functions (`f_torus`, `f_heart`, …)

The formula VM also ships the **complete set of POV-Ray's built-in isosurface
functions** — the classic `functions.inc` library (`f_torus`, `f_heart`,
`f_klein_bottle`, `f_superellipsoid`, `f_dupin_cyclid`, `f_helix1`, `f_spiral`,
`f_kummer_surface_v1/v2`, `f_boy_surface`, `f_steiners_roman`, and ~60 more). They are
**exact ports of POV-Ray's own C++ source** (`source/vm/fnintern.cpp`), so a scene using
them evaluates to the same field POV-Ray computes — call them straight from any `expr`
string, no `#include` needed:

```
isosurface {
    material copper
    function {
        expr "f_torus(x, y, z, 0.28, 0.10)"    # major R = 0.28, minor r = 0.10
        translate 0.5 0.5 0.5   rotate 62 0 0
    }
    contained_by { min 0.05 0.05 0.05   max 0.95 0.95 0.95 }
    max_gradient 1.5
}
```

Exactly as in POV-Ray, the **first three arguments are the coordinates** (`x, y, z`,
which you may pre-transform) and the rest are the function's parameters — e.g.
`f_torus(x,y,z, majorR, minorR)`, `f_heart(x,y,z, strength)`,
`f_superellipsoid(x,y,z, e, n)`. Distance-like functions (`f_torus`, `f_sphere`,
`f_rounded_box`, `f_helix1`) are ~unit-Lipschitz and march robustly; the many
**polynomial** surfaces (`f_heart`, `f_klein_bottle`, `f_dupin_cyclid`, …) are not signed
distances, so give them a `max_gradient` (POV clamps most of them to ±10, so a bound in
the 5–50 range usually works).

The **noise-based** entries — `f_noise3d`, `f_noise_generator`, `f_ridge`, `f_ridged_mf`,
`f_hetero_mf` — are supported too, driven by an **exact host+device port of POV-Ray's
Perlin `Noise()`** (`src/pov_noise.h`, generated by `tools/pov_noise_gen.py`): POV's init
tables are re-derived and baked in, so a scene like
`sqrt(x*x+z*z) - 1 + 0.5*f_noise3d(3*x,3*y,3*z)` yields the same bumpy field POV produces,
identically on CPU and GPU (all three noise generators: 1=Original, 2=RangeCorrected
[default], 3=Perlin). Only `f_pattern` (which needs POV's full pattern/pigment engine)
remains unported — tracked in `known-issues.md`. The functions are generated by
`tools/pov_functions_gen.py` into `src/pov_functions.h` and evaluated by the same
host+device VM, so they render identically on CPU and GPU.

##### Ray-march strategy (`method`, `refine`)

Any `isosurface` (analytic *or* `function`) chooses how the ray finds the field's first
zero crossing:

| Key | Values | Meaning |
|---|---|---|
| `method` | `adaptive` (default) / `sample` | how the ray steps toward the surface |
| `samples` | `<n>` | *sample mode only* — number of fixed steps across the container's diagonal (default 256; or size the step with `accuracy`) |
| `refine` | `bisect` (default) / `regula_falsi` | how a bracketed sign change is refined to the root |

- **`adaptive`** steps by `max(|f|/max_gradient, accuracy)` — sphere-tracing for a true
  SDF (`max_gradient = 1`), or a Lipschitz-bounded march for a `function` field. With a
  correct `max_gradient` it **provably cannot skip** the first crossing (across one step
  `f` can change by at most the step size, so it can't dip through zero and back), and it
  slows down only near surfaces. This is the right choice almost always.
- **`sample`** ignores `|f|` and marches by a **fixed** world step (POV-Ray's sampling
  mode). It needs **no** Lipschitz bound, so it's the fallback when `max_gradient` can't be
  trusted (spiky/near-unbounded gradients where the auto-estimate is unreliable) — but a
  feature thinner than one step *between two samples* can be missed, so raise `samples`
  until the surface is clean.
- **`refine`** only changes root-polishing speed, not the result: `bisect` is
  unconditionally robust (linear); `regula_falsi` (secant with the Illinois safeguard)
  converges faster on smooth brackets. Both land on the same root to ~1e-12.

Validated: on a clean surface the `sample` and `adaptive` marchers agree to RMSE ≈ 0.4/255
(CPU) / 0.01/255 (GPU) — same geometry, both backends. `scenes/function.ftsl` uses the
adaptive default; `method sample` + `samples`/`refine` are shown in the scraps test
scenes.

##### Exporting an isosurface to a mesh (`-export-mesh`)

Any scene's isosurfaces can be **polygonised into a watertight triangle mesh** and written
as an OBJ (for import into Unreal, Blender, etc.) instead of being rendered:

```
ftrace -in scene.ftsl -export-mesh out.obj -mesh-res 192
ftrace -in scene.ftsl -export-mesh out.obj -mesh-res 256 -mesh-adaptive -mesh-decimate 0.35
```

| Flag | Meaning |
|---|---|
| `-export-mesh <file.obj>` | polygonise every `isosurface` in the scene (marching **tetrahedra**), write an OBJ, then exit (no render). Each isosurface becomes one OBJ object (`o isosurface_k`). |
| `-mesh-res <N>` | **fineness** — grid cells along the longest bounds axis (default 128). The other axes get proportional counts so cells stay ~cubic. Higher = more triangles / finer detail. |
| `-mesh-adaptive` | after marching, run a curvature-adaptive **quadric-error decimation** pass. |
| `-mesh-decimate <f>` | adaptive target: keep this fraction of triangles (default 0.5; implies `-mesh-adaptive`). |

The exporter reuses the **exact field the renderer sees** — `f(x,y,z)` for edge crossings and
`∇f` for normals — so the mesh matches the rendered surface. It uses **marching tetrahedra**
(Kuhn/Freudenthal 6-tet split of each cell) rather than marching cubes: tetrahedra have no
face-ambiguous cases, so the output is a guaranteed **watertight 2-manifold** (marching cubes
can leave holes / non-manifold edges). The field is **intersected with its `contained_by`
domain box** (a CSG `max(f, boxSDF)` over a lattice padded a couple cells beyond the box), so a
surface that reaches the boundary is sealed with a flat cap into a **closed solid** instead of
leaving an open rim. Vertices are welded by a canonical grid-edge id (adjacent cells reference
one vertex ⇒ **no cracks**), crossings are refined by bisection on the real field, per-vertex
normals come from the field gradient (box-face normals on caps), and each triangle is wound so
its geometric normal points outward.

**Watch the cap warning.** Capping is only correct when `f < 0` means *inside the shape you
want*. If the expression's sign is inverted, "solid" becomes everything **outside** the shape,
the container sits entirely within it, and the exporter faithfully returns the whole container
as a closed shell with the intended surface hollowed out invisibly inside — an export that
looks like a plain ball or box from every angle. Since v0.121.0 the exporter measures how many
output triangles lie on the cap and says so:

```
[export-mesh]   WARNING: 66% of these triangles are CONTAINER CAP, not surface.
```

If you see that, either add `open` to the `isosurface` (skip capping, keep the raw cut rim) or
negate the expression. This is not hypothetical — it is how `meshes/klein_a120_b060_c30_d127*.obj`
silently became featureless balls; see `known-issues.md`.

The **adaptive** pass collapses cheap edges first: the quadric error is near-zero on flat
regions (a vertex can slide freely) and large where the surface curves, so triangles thin out
on flat areas and stay dense on detailed ones — the requested curvature-driven tessellation. A
**link-condition** test plus foldover rejection keep the mesh a watertight 2-manifold through
the collapses. (The mesher runs on the CPU; it reads `Implicit::eval`/`gradient` from
`src/isomesh.h`.)

##### Auditing airtightness (`-check-watertight`)

Glass (`dielectric`) surfaces must be **watertight** — a closed 2-manifold where every edge
is shared by exactly two triangles. The renderer decides *entering vs exiting* from the
surface normal at each hit and carries the "which medium am I inside" state along the whole
photon path, so a **hole** (boundary edge) lets a ray reach the interior without a refraction
event and desyncs that bookkeeping, a **non-manifold** edge (3+ faces) is geometrically
ambiguous, and a **flipped** (inconsistently-wound) facet inverts the enter/exit test — any of
which bends light wrong for the rest of that path and can splash artifacts far from the object.

```
ftrace -in scene.ftsl -check-watertight      # or the -airtight alias
```

audits every named `mesh` and every `isosurface` (polygonised at `-mesh-res` first), prints a
per-object `[OK]`/`[WARN]` report with the offending edge counts, then exits without rendering.
Dielectric objects are flagged with `!` since a leak actively corrupts their refraction. The
process exit code is non-zero if any object is not airtight, so it doubles as a CI gate. (Marching
cubes output is watertight by construction; warnings there usually mean a `contained_by` box
clipped the surface open. Imported OBJ meshes are the common offender — self-intersections and
mouth openings show up as non-manifold or boundary edges.)

Note the renderer intersects an isosurface by **ray-marching the analytic field directly** at
render time — it never builds a mesh. Marching cubes runs only offline, for `-export-mesh` and
for the `-check-watertight` audit (which polygonises the field purely to reuse the same mesh
edge-checker). So the audit on an isosurface is a faithful *proxy* for the field's closedness,
not the exact geometry the renderer marches.

##### Auditing the *marched* field directly (`-check-airtight`)

Because `-check-watertight` audits a polygonised *copy* of an isosurface, it inherits marching
cubes' resolution blind spot: a leak, a thin wall, or a spike narrower than a grid cell can slip
through. `-check-airtight` instead probes the **exact zero level-set the renderer sphere-traces** —
it calls the same field marcher the camera rays use, so there is no proxy.

```
ftrace -in scene.ftsl -check-airtight              # 4000 chords/isosurface
ftrace -in scene.ftsl -check-airtight -check-airtight-rays 20000
```

The test is a Monte-Carlo **ray-parity** audit: it fires random chords that start and end
*outside* the container, so both endpoints are unambiguously outside the solid. A closed,
airtight solid crosses its boundary an **even** number of times along any such chord (every entry
is matched by an exit), so the renderer's marcher must report an even hit count. An **odd** count
means the interior connects to the exterior — a leak:

- on an **`open`** (uncapped) surface, the solid poking through a `contained_by` wall (an open
  cap) — the audit also directly samples the container boundary and reports the interior area and
  its worst `f`, and suggests `capped` / shrinking `contained_by`;
- on a **`capped`** surface, a crossing the marcher *skipped* — a wrong `max_gradient`/Lipschitz
  bound overshooting, or a feature thinner than the march step — which shows up as a real light
  leak at render time.

A dense reference sampling (finer than the march step) runs alongside; where the marcher finds
*fewer* crossings than the reference it flags **overshoot** even when parity stays even (two
missed crossings). Non-destructive, exits non-zero on any leak (so it too works as a CI gate).
An analytic isosurface, unlike a mesh, cannot self-intersect — it is a level set of a continuous
field, locally a smooth manifold at every regular point — so this audit only tests closedness,
not self-intersection.

##### Repairing a non-airtight mesh (`tools/repair_mesh.py`)

There are two philosophies for getting watertight geometry, and they are complementary, not
ranked:

- **Author it airtight** with **`manifold3d`** (Emmett Lalish's Manifold library). Its guarantee
  is a *closure property* — **manifold in ⇒ manifold out**: its boolean/offset operations, given
  valid 2-manifold inputs, are algorithmically guaranteed to produce a valid 2-manifold, so you
  never *introduce* a leak. It achieves that by **requiring clean input** — hand it a broken mesh
  and it reports a non-manifold error rather than fixing it. Reach for it when you build/combine
  geometry (CSG) and want to never produce a self-intersection or crack in the first place. It is
  *not* a repair tool.
- **Repair a broken mesh** after the fact with **`tools/repair_mesh.py`**, which wraps two
  engines (both `pip install`-able):
  - **pymeshlab** (default, `pip install pymeshlab`) — MeshLab's repair filters, run as an
    *ordered pipeline* (order matters): merge-close-vertices (welds coincident vertices so a
    pinch becomes a visible singularity) → remove-duplicate/null-faces → repair-non-manifold-edges
    (`method=0` removes the offending faces) → repair-non-manifold-vertices (splits pinched
    sheets apart) → close-holes (caps the openings that leaves). This is the **go-to engine for
    the "pinch vertex" defect** (N surface sheets snapped to one point) that AI mesh generators
    emit — a defect that is a valid 2-manifold in raw OBJ indexing but non-manifold once
    coincident vertices are welded, which is exactly the class MeshFix leaves untouched. It is
    the engine that took the Klein bottle to `[OK]`.
  - **pymeshfix** (`--engine meshfix`) — Marco Attene's MeshFix: best for genuine self-intersections
    and large holes; weaker on pure non-manifold pinches.

```
python tools/repair_mesh.py broken.obj fixed.obj            # MeshLab engine
python tools/repair_mesh.py broken.obj fixed.obj --engine meshfix
# repair a master mesh, then place the result exactly where a derived copy sat:
python tools/repair_mesh.py master.obj staged.obj --place-like staged_original.obj
```

Re-audit the output with `-check-watertight` to confirm `[OK]`. (Example: the `klein_hunyuan`
glass mesh had a single 3-sheet pinch vertex — invisible in raw OBJ indexing but a non-manifold
singularity once coincident vertices are welded; `repair_mesh.py` with the MeshLab engine removes
the pinch and closes the hole, taking it to `[OK]`.)

## Textures

`texture "name" { file <path> encoding srgb|linear filter nearest|bilinear wrap
repeat|clamp|mirror }` loads PNG / JPG / HDR / PPM / PFM images; bind one to a
diffuse albedo with `reflect texture:<name>`. Each texel is Jakob–Hanika
upsampled to a reflectance spectrum. UVs come from quad corners, OBJ `vt`
(`uv use_mesh`), a procedural `planar`/`spherical`/`cylindrical` projection, or
per-hit `triplanar` box projection for un-UV'd meshes (see Geometry). Besides
base-colour albedo, a texture can also drive a **scalar** parameter: a grayscale
**roughness map** on `glossy` (`roughness texture:<name>`) or a **film-thickness
map** on `thinfilm` (`film_thickness_map texture:<name>`, a 0..1 profile × the
nominal `film_thickness`). All of these run on both the CPU and GPU forward paths.
A texture can also be an **indexed-spectral palette** — `palette { 0 spectrum:navy
1 spectrum:crimson … }` maps red-channel indices (0..255) to named reflectance
spectra, looked up nearest (CPU only; GPU falls back). A 2-child `mix` can take a
**blend mask** (`weight_map texture:<name>`) that selects child 0 vs child 1 per hit.
A scalar map on `ior` remains future work.

Any material can also carry a **tangent-space normal map** — `normal_map
texture:<name> strength <s>` — that perturbs the shading normal per hit without adding
geometry, so a flat surface picks up per-texel highlights and self-shadowing. The map
must be declared `encoding linear` (it stores raw XYZ vectors, not sRGB colour; the
loader warns if it isn't). Per-triangle tangents are derived from the UV gradients
(with a stored handedness sign), the sampled `[0,1]` texel is remapped to a
`[-1,1]` vector, rotated through the surface TBN frame, and blended toward the
geometric normal by `strength` (1.0 = full). Tangents follow mesh instances, and the
perturbed normal is applied at the single intersection choke point so **every**
renderer — backward, forward, BDPT, VCM, SPPM, photon-map — and both the CPU and GPU
paths shade identically. See `scraps/ripple_test.ftsl` (a flat wall reading as
corrugated under grazing light).

A texture's albedo can also be **procedural in UV space**: in place of `file`, give
three quoted ftsl expressions of the surface UV — `rgb "r(u,v)" "g(u,v)" "b(u,v)"`
(the pattern infix grammar; variables `u v`, constant `pi`; each output clamped to
`[0,1]`, interpreted as linear RGB). ftrace bakes them once at load to a `res`×`res`
grid (default 512) and then treats it exactly like an image texture — the same
UV-wrap, Jakob–Hanika upsampling, triplanar, GPU and raster paths, and
`reflect texture:<name>` binding all apply unchanged. This completes the skin matrix
alongside image skins and 3-D-space procedural patterns: a **UV-space procedural**.
See `scenes/procskin.ftsl` (loom: `ProcTexture` / `func_skin`).

A texture can also be **grown rather than drawn**, by a
**Gray–Scott reaction–diffusion** simulation run once at load:

```
texture "hide" {
    reaction { preset spots      # spots | holes | maze | coral | worms | mitosis
               sim 256           # solve grid (see below — this is the density knob)
               steps 6000 }      # how long the reaction runs
    wrap repeat
}
```

Two chemicals diffuse and react — `du/dt = Du·∇²u − uv² + F(1−u)`,
`dv/dt = Dv·∇²v + uv² − (F+k)v` — and Turing's observation, which is the reason this
is here, is that the *uniform* solution of such a system can be unstable to spatial
perturbation while remaining stable in time. A featureless sheet therefore organises
itself into spots, labyrinths or dividing blobs with an intrinsic wavelength that
appears nowhere in the equations. That makes it categorically different from every
other source here: `noise`, `worley`, `gabor` and `bnoise` all place features *by
fiat*, whereas these are the **outcome of a process**, so their spacing, branch points
and defects are correlated the way a real coat pattern's are — and this is in fact the
standard model of exactly that (animal markings, coral, fingerprints, chemical Turing
patterns in a gel). It is a **bake** and not a pattern op because the value at a point
is the endpoint of a trajectory of the whole field: there is no local closed form to
evaluate per hit. Living in `texture` means the entire existing pipeline then applies
with no renderer changes at all — UV wrap, Jakob–Hanika upsampling, triplanar, GPU
upload, raster preview, `reflect texture:<name>`, and `tex:<name>(u, v)` as one term
inside a pattern formula.

Notes that matter in practice:

- **It tiles seamlessly.** The Laplacian wraps on both axes, so the solve runs on a
  torus. This cannot be retrofitted — blending the edges of a finished RD field would
  destroy precisely the long-range correlations that make it not-noise — so the
  topology is chosen up front, and the seed is periodic for the same reason.
- **`sim` is the density knob, not `res`.** A feature is a fixed number of *grid cells*
  wide, so doubling `sim` puts twice as many features across the texture; `res`
  (default `sim`) only sets the resolution the result is stored at.
- **Presets are load-bearing.** The (F, k) plane is mostly *not* interesting — outside
  a thin crescent every seed decays back to the uniform state — and the crescent is
  only about 0.01 wide in `k`, so plausible-looking hand-picked numbers usually give a
  blank texture. `feed`/`kill` are authorable for exploring it, and a solve that
  settles to a uniform state warns at load rather than silently baking a grey sheet.
- **The default diffusion is a rescale of the textbook one** (`Du` 1.0 / `Dv` 0.5 =
  0.16 / 0.08 × 2.5²). Multiplying both by s² is a pure spatial rescale of the same
  continuum problem, so published (F, k) values still mean what they say, but each
  feature becomes s× wider in cells; at the raw 0.16 a spot is only ~4 cells across
  and comes out visibly *square*, pixel-locked to the lattice.
- **Explicit Euler has a stability bound**, `dt·max(Du,Dv)·1.6 ≤ 2` (the 9-point
  stencil's Fourier symbol bottoms out at −1.6). Exceeding it does not degrade
  gracefully, it goes to NaN in a few dozen steps, so it is a **load error**, not a
  warning.
- **Not every regime converges.** Spot and blob regimes settle; the `maze` regime
  genuinely never does — its corridors keep reconnecting — so there `steps` is an
  aesthetic choice rather than a convergence criterion.

Worked example `scenes/pattern_reaction.ftsl` (a maze wall, spots gated by a noise, a
3×3 seamless tiling, pitted flooring, and a triplanar-projected leopard torus),
deterministic self-test `ftrace -checkreaction`.

### Stochastic tiling (`tiling stochastic`)

Any image texture can be tiled **without the lattice being visible**, by
**histogram-preserving blending** (Heitz & Neyret, HPG 2018):

```
texture "lichen" {
    file     scenes/lichen.ppm
    encoding srgb
    wrap     repeat
    tiling   stochastic     # default: none
    patch    1.0            # lattice cell size, in texture repeats (1 = the paper's)
    seed     3              # which realisation of the random crop offsets
}
```

The problem it solves is *not* seams — a seamlessly periodic source still fails, because
at six repeats per metre the eye locks onto the same feature marching in a grid, and no
wrap mode can fix that. The only cure is to stop showing the same crop twice.

At every shading point the operator draws **three randomly offset crops** of the source
on a triangle lattice and blends them with the barycentric weights. Done naively that
is *worse* than repeating: averaging three crops of a bimodal image yields the mean of
its two modes — a colour occurring nowhere in the source — and the texture turns to
soup. So instead each channel is **rank-transformed** at load onto `N(1/2, 1/6)`, the
three taps are blended there, the variance the average destroyed is restored by
dividing the centred blend by `sqrt(Σwᵢ²)`, and the result is inverted through a stored
1-D LUT. Every value emitted is therefore a value the source actually contains:
contrast, histogram and colour statistics survive, the lattice does not.
(`Σwᵢ²` averages `1/2` for Dirichlet(1,1,1) weights, so an unrestored blend would sit at
`0.707×` the source's standard deviation; `-checkstochtile` measures both.)

Notes that matter in practice:

- **`patch` is measured in texture repeats**, not metres or texels, so it composes with
  whatever UV scale the projection uses. `patch 1.0` is the paper's default; smaller
  values shuffle more aggressively at the cost of blurring features larger than a cell.
- **`seed` picks the realisation.** Two textures with the same file and different seeds
  decorrelate; the same seed reproduces exactly, on every backend.
- **The blend happens in linear RGB, not in spectral-coefficient space.** Jakob–Hanika
  coefficients are *not* a colour space — interpolating them channel-wise produces
  visible blue-cyan fringing — so the three taps are blended as RGB and the blended
  colour is then converted to a reflectance through one shared, texture-independent
  64³ coefficient LUT (built lazily, threaded, ~0.6 s, 3.1 MB, the first time a
  stochastic texture loads). This is also what lets the spectral CPU path, the CUDA
  path and the mode-`W` raster preview run the *identical* operator on the *identical*
  planes.
- **Cost is three taps instead of one**, plus a LUT lookup per channel; the
  rank-transform planes are built once at load and roughly double the texture's memory.
- Works with every UV source (including `triplanar`), on `reflect texture:<name>` albedo
  and inside `tex:<name>(u, v)` pattern terms, on CPU and GPU alike.

Worked A/B example `scenes/stochtile.ftsl` (one wall, two halves, one source image:
`tiling none` on the left, `tiling stochastic` on the right), deterministic self-test
`ftrace -checkstochtile`.

## Procedural patterns (math-driven materials)

A `pattern "name" { … }` block compiles a **scalar field** — a function of the hit
point evaluated per shading sample — that can drive any scalar material parameter
*procedurally*, without a texture image. The variables available to a pattern are the
world-space position `x y z`, the implicit field value `f` (the SDF value at the hit,
`~0` on an isosurface; `0` for explicit geometry), the surface normal `nx ny nz`, the
radius `r = √(x²+y²+z²)`, the **surface UV coordinates `u v`** (mesh-interpolated,
or a native-primitive wrap — see below), the **mean curvature `curv`** (see
*Curvature-driven, non-stationary patterns* below), the **enclosure `cavity`**
(see *Enclosure-driven patterns* below), and the **shading footprint `fw`** (see
`fnoise` below). Two authoring forms:

- **Free-form expression** — `expr "0.5 + 0.5*sin(40*y)"` (must be quoted). Compiled by
  a shunting-yard parser to a postfix scalar VM. Supports `+ - * / ^ %`, comparison-free
  math, `pi`, and functions `abs sqrt sin cos tan exp log floor fract sign saturate min
  max atan2 step pow clamp mix smoothstep noise gabor fnoise`. On top of the scalar `noise` there is
  **vector-valued gradient noise for domain warping**: `dnoisex/y/z(x, y, z)` are the
  three decorrelated components of POV-Ray's `DNoise` vector at a point, and
  `dturbx/y/z(x, y, z, octaves, lambda, omega)` the components of its octave sum
  `DTurbulence` (POV defaults 6, 2.0, 0.5; `octaves` clamps to [1, 10]; values signed,
  roughly `[-1, 1]`). Where scalar `noise()` can only modulate a value, these *displace
  the coordinate another pattern is sampled at* — `sin(6.2832*(3*x +
  1.1*dturbx(3*x,3*y,3*z, 6, 2, 0.5)))` is the classic POV marble; worked example
  `scenes/pattern_warp.ftsl`, deterministic self-test `ftrace -checkvnoise`.
  There is also **cellular (Worley / Voronoi) noise** — one jittered feature point
  per unit lattice cell, queried by `worley(x, y, z, metric)` (F1, the distance to
  the nearest point), `worley2` (F2), `worleyd` (F2−F1, `0` exactly on cell
  borders — crack networks), and `worleyid` (a flat per-cell random value in
  `[0,1)`), all under a runtime `metric` operand rounded and clamped to
  **0 Euclidean / 1 Manhattan / 2 Chebyshev**. Distances are raw at cell size 1
  (Euclidean F1 mean ≈ 0.65), F1/F2 are exact (adaptive ring search, not the
  common 3×3×3 approximation) and CPU/GPU bit-identical; worked example
  `scenes/pattern_worley.ftsl`, deterministic self-test `ftrace -checkworley`.
  And there is **anisotropic, band-limited Gabor noise** —
  `gabor(x, y, z, f, wx, wy, wz)`, returning `[0,1]` with mean `0.5` like `noise`.
  `f` is the frequency in **cycles per unit of the coordinates you hand it**, and
  `(wx, wy, wz)` is the **steering direction**: the field oscillates *along* that
  vector, so the visible streaks run perpendicular to it. A zero-length vector means
  **isotropic band-pass** noise. Both the direction and the frequency are ordinary
  sub-expressions, so they can vary from point to point — which is the reason the
  primitive exists. Steering a *lattice* noise means writing `noise(R(p)·p)`, whose
  Jacobian is `R + (dR/dp)·p`: the second term grows with distance from the origin, so
  the texture shears further and further out and never has quite the orientation you
  asked for (the same trap as spatially varying frequency — see *Putting them
  together* below). A Gabor kernel only sees the offset from **its own centre**, at
  most one cell, so a varying direction leaves a residual bounded by the local turning
  rate rather than by `|p|`. Being band-limited it is also the one noise here that
  minifies gracefully. Everything is analytic: the impulses are a genuinely
  homogeneous Poisson process (so the field is stationary under *arbitrary*, not just
  integer, translation), the compactly supported C² envelope makes the 3×3×3 search
  exact rather than the usual 95%-of-a-Gaussian, and the per-impulse random phase makes
  the variance independent of `f`. Worked example `scenes/pattern_gabor.ftsl`
  (isotropic speckle, brushed metal, wood rings round an off-screen trunk, flow-aligned
  fibre), deterministic self-test `ftrace -checkgabor`.
  Alongside Worley there is **blue-noise (Poisson-disk) placement** —
  `bnoise(x, y, z, r)` / `bnoise2` / `bnoised` / `bnoiseid`, the same four slots over a
  point set with a **guaranteed minimum separation** `r` (in cell units, clamped to
  `[0,1]`). This is the placement primitive for scattered features — freckles, pores,
  seeds, dimples, spatter. Worley's sites are a jittered lattice, so two of them can be
  arbitrarily close (both jitter to the shared cell wall) while elsewhere the lattice
  leaves holes; threshold F1 to draw spots and that clumping reads instantly as computer
  texture. Evenness is a property of *where the points are* and cannot be recovered
  downstream, which is why it needs its own primitive rather than a filter. `r` sweeps
  continuously: at `r = 0` nothing is excluded and the set is exactly the jittered
  lattice (so this is a strict generalisation of Worley's placement); at `r = 1` it is
  maximally blue, keeping 0.2661 points per cell. Pair `bnoiseid` with `bnoise` to give
  each spot its own radius — equal-sized dots read as polka dots, unequal ones as
  freckles. Classical dart-throwing is *sequential* and so cannot answer a per-hit query
  at all; instead acceptance here is one round of Luby's maximal-independent-set
  algorithm under a strict total order on candidates (equivalently a Matérn type-II
  thinning of a jittered lattice), which makes the separation a **theorem** and
  membership a purely **local** predicate — so an unbounded set is queryable in `O(1)`
  with no bake, no tiling and no repetition, at essentially Worley's cost (29 cells
  hashed per query against Worley's 27). Worked example
  `scenes/pattern_bluenoise.ftsl` (which puts Worley- and blue-noise-placed spots on the
  two halves of one wall at matched density), deterministic self-test
  `ftrace -checkbluenoise`.
  And there is **filtered (band-limited) fBm** — `fnoise(x, y, z, w, octaves)`, the sum
  of `octaves` octaves of the same lattice `noise` at lacunarity 2 and gain 0.5, again
  `[0,1]` with mean `0.5`, but with each octave weighted by how much of it a shading
  sample **of width `w` can actually resolve**. `w` is in the units of the coordinates
  you hand it (scale the coordinates, scale the width) and is the **diameter of the
  surface patch the sample stands for**; `w ≤ 0` means unfiltered and reproduces plain
  fBm exactly, so filtering is opt-in and free when off. This is the antialiasing
  primitive: every other noise here is evaluated at a *point*, which is a lie once the
  sample stands for an area — the finer detail does not merely vanish, it folds down
  into a moiré of the sampling lattice, and more samples do not fix it. The per-octave
  weight is the **linear-MMSE coefficient measured off this very lattice**
  (`scraps/fnoise_fit2.py`, pinned by the self-test) rather than a chosen falloff, and
  the measurement contradicts the intuitive design twice over: the optimal weight is
  still `0.95` **at** Nyquist and `0.81` where a naive cutoff would already have dropped
  the octave whole, and over-filtering is *not* the safe direction — it deletes
  low-frequency content the footprint genuinely contains and lands further from the
  truth than no filtering at all. (For the same reason `w` means a 2-D surface patch and
  not a solid ball: their weights differ by a whole power of `w` in the tail.) It is the
  **deterministic** samplers this is for — mode `W`, the raster preview, and low-spp
  backward renders; the forward photon modes already integrate each pixel's footprint
  stochastically, so there `w` buys nothing and costs detail. Deterministic self-test
  `ftrace -checkfnoise`, which measures the result against a brute-force footprint
  average rather than merely asserting that it blurs. Worked example
  `scenes/pattern_fnoise.ftsl`.
  You do not have to derive `w` yourself: the variable **`fw`** *is* it — the
  world-space diameter of the surface patch one shading sample stands for, computed by
  the renderer at the hit, so `fnoise(90*x, 90*y, 90*z, 90*fw, 3)` is the whole idiom
  (scale the width exactly as you scaled the coordinates; `fw` is always in world
  units). It is built from the solid angle one pixel subtends — via
  `Camera::pixelSolidAngle`, so fisheye and panoramic lenses need no special case —
  the hit distance, and the obliquity: the footprint on a slanted surface is an
  *ellipse* with minor axis `d` and major axis `d/|cos|`, and `fw` reports the
  geometric mean, i.e. the disc of equal area (clamped at `|cos| = 0.02`, or every
  silhouette would filter to a flat grey band). Supersampling divides it by `√spp`,
  since jittered samples already average over the footprint — so the filtering backs
  off on its own as a render converges and one scene serves both a 1-spp preview and a
  ground-truth render. **`fw` is 0 — meaning *unfiltered*, never a small blur —
  wherever the renderer cannot honestly answer**: the forward photon modes and
  stochastic mode `R` (both already area-average), secondary bounces (which would need
  ray differentials to know how far their footprint had spread), and implicit-field /
  medium formulas (evaluated at march samples, not at a surface). It is filled at
  primary hits in mode `W` and at every pixel of the raster preview, and — like `curv`
  and `cavity` — is rejected in an `emit` pattern, where it is doubly wrong because an
  emitter's radiance cannot depend on who is looking at it.
  It can also **sample a declared image
  as a term**: `tex:<name>(u, v)` returns the mean of the texel's three linear RGB
  channels (the same `Texture::scalarAt` sampler a `texture:<name>` slot binding uses,
  honouring that texture's `filter`/`wrap`), so a photo can be one *operand* of a formula
  — `expr "saturate(tex:grime(u,v) * (0.5 + 0.5*sin(28*u)) * 2)"` — instead of only being
  pasted over a slot. Its coordinates are ordinary sub-expressions, so the lookup can be
  warped. `tex:` needs a real surface, so it is a **compile error** in an isosurface
  `function { expr }`, in a medium `density`/`ior` program, and at load-time constant
  sites — never a silent zero. Worked example: `scenes/pattern_tex.ftsl`.
  It can equally **sample an N-D array of authored numbers**: a `grid "name" { shape …
  lo … hi … outside … data { … } }` block declares a **regular lattice in 1–4
  dimensions** (samples in C order, axis 0 outermost), and `grid:<name>(c0, …)` reads it
  back with separable **N-linear** interpolation — so a measured curve, a lookup table or
  a small height field becomes a term in a formula. The call's **arity is the grid's own
  dimensionality**, so a wrong argument count is a compile error rather than a silent
  zero. `lo` defaults to zeros and `hi` to the unit-spacing *index* lattice (coordinates
  are then literally indices); a single `hi` number gives an isotropic lattice. Outside
  the box, `outside` picks `clamp` (default), `wrap` (period `hi-lo`) or `extrapolate`.
  `data` can also be written **bracketed**, and then the **nesting is the shape**
  (`data [[0 1 2][3 4 5]]` is a 2×3 grid), so `shape` need not be written at all — the one
  shape you can't get wrong, because it isn't written down twice.
  Grids upload verbatim to the GPU and both backends run the *same* sampler, so a grid
  renders identically either way; `ftrace -checkgrid` is the deterministic self-test and
  `scenes/pattern_grid.ftsl` the worked example (one feature per wall strip).
  For data that never sat on a lattice there is the **ragged sibling**: a
  `scatter "name" { dim … power … data { … } }` block holds values at **arbitrary
  positions** in 1–4 dimensions (one interleaved `p0 … p_{n-1} value` run per sample),
  and `scatter:<name>(c0, …)` blends them by **Shepard inverse-distance weighting** —
  the interpolant for a handful of probe measurements, authored control points or
  samples along a path. Like `grid:`, the call's arity is the table's own `dim`. Every
  sample is reproduced *exactly* at its own position and with zero slope (the
  characteristic plateau), and `power` is the locality knob: low values blend broadly,
  high values approach a nearest-neighbour/Voronoi look. Self-test `ftrace -checkscatter`,
  worked example `scenes/pattern_scatter.ftsl`.
  Unlike `tex:`, a table sample needs no surface — only coordinates — so `grid:`/`scatter:`
  are legal in **scalar field formulas** too: an isosurface / CSG `function { expr }` leaf
  (`expr "grid:terrain(x, z) - y"` is a measured height field), a medium's `density`
  program (a measured volume without going through `vdb:`), a medium's `ior` program
  (`ior "1 + grid:n(x, y, z)"` — a measured refractive-index profile that bends rays, see
  GRIN media), and a `camera_curve` driver (`fov_from lens.fov(grid:zoom(t))`). Everything
  downstream reads the same tables: the isosurface polygoniser, the sphere-trace Lipschitz
  bound, the medium's majorant scan and the GRIN marcher. Only a **load-time constant**
  site — evaluated before the tables are visible — remains a compile error. Worked example
  `scenes/grid_field.ftsl` (a 5×5 lattice as an isosurface height field, plus a 1-D
  profile as a medium's density).
  A table sample is finally legal **at a value site directly**, with no `pattern` wrapper
  to write: `reflect grid:ramp(u)`, `roughness scatter:probe(u,v)`. The slot takes the
  sample exactly as if you had written the one-line `pattern { expr "grid:ramp(u)" }` and
  pasted its name in, so every per-hit slot that accepts `pattern:<name>` accepts a table
  call too, and the coordinates are ordinary expressions — including a composed array
  literal, `reflect grid:ramp([0.2 0.8](u))`. The **scoped** spelling is the one that
  works: a bare `ramp(u)` at a value site already means "apply the material `ramp`, with
  `u` bound to its formal" (see material bundles), so only `grid:` / `scatter:` name a
  table unambiguously. The call is required — `reflect grid:ramp` says which table but
  not *where* to read it, and is a load error that shows you the `(u)` to add — and a
  **load-time constant** slot (`film_ior`, a fixed `ior`) still refuses a per-hit value,
  naming the slots that can take one.
- **Inline array literal** — a tiny table written *where it is used*, with no block, no
  name and no `pattern` wrapper: `roughness [0.05 0.4 0.05](u)`. The `[ … ]` is the data
  and **nesting is the shape** (axis 0 outermost, C order, as in a `grid`'s `data`), so a
  shape is never spelled and cannot disagree with the data; `[[0 0.5][0.5 1]](u,v)` is
  2-D. The trailing `(…)` is the **sample call** — one coordinate per nesting level, each
  a full pattern expression (`[0 1](0.5+0.5*sin(2*pi*3*u))` sweeps the ramp back and
  forth). A literal spans the **unit interval on every axis**, which is what makes `(u)` /
  `(u,v)` the natural coordinates. The call is required: an *unsaturated* array is a load
  error, as is a ragged one or a wrong coordinate count. Each literal desugars to an
  anonymous `grid` + `pattern`, so it works in **every slot that accepts `pattern:<name>`**
  and interpolates N-linearly exactly like a grid — including the `reflect` slot, where
  `reflect [0 1](u)` is a greyscale albedo ramp (see below). Worked example
  `scenes/pattern_array.ftsl`; reach for a named `grid`/`scatter` when the data is big,
  shared, or worth naming.
  A coordinate may also be **`a`**, the one input with no per-hit meaning of its own — that
  spends nothing and leaves the axis as a **formal** for whoever *uses* the material:
  `material "ramp" { reflect [0 1](a) }` is then bound at the use site by `ramp(a=u)`, at a
  property reference by `ramp.reflect(a=u)`, or positionally by `ramp.reflect(u)`, all three
  identical to `reflect [0 1](u)`. Multi-axis rebinds are simultaneous, so a 2-D literal
  transposes under `(u=v, v=u)`. A literal's *formals* are the driver names in its own tuple;
  it has no second, private namespace of axis names, so a `formal=driver` argument **inside**
  a literal's call (`[0 1](a=u)`) is a load error that names both spellings that work.
  Because a coordinate is a full expression, a literal may also be **composed** into
  another's call — `[0 1]([0.2 0.8](u))` reads the outer table at a coordinate the inner
  one produces, to any depth, on any single axis of a multi-axis call
  (`[[0 0.3][0.6 1]]([0.5 1](u), v)`), and as a term inside coordinate arithmetic. That is
  the general remapping idiom: an inner table becomes the transfer curve applied before the
  outer lookup. Pinned by `ftrace -checkarray`; per-tile scene `scenes/_array_formal.ftsl`.
- **Named generator** — `type <gen>` plus params (mirrors material syntax):

  | Generator | Parameters | Result |
  |---|---|---|
  | `axis`    | `axis <x\|y\|z>` `[scale]` `[offset]` | a coordinate ramp |
  | `radial`  | `[center <x y z>]` `[scale]` | distance from a point |
  | `bands`   | `axis <x\|y\|z>` `[freq]` `[phase]` | `0.5+0.5·sin(2π·freq·coord+phase)` stripes |
  | `checker` | `[size]` | 0/1 world-space checkerboard |
  | `noise`   | `[freq]` | deterministic value noise in [0,1] (CPU/GPU bit-identical) |
  | `field`   | `[scale]` | the raw implicit field value `f`, scaled |

Bind a pattern anywhere a scalar `texture:<name>` map is accepted, using
`pattern:<name>` instead: **roughness** (`dielectric`/`glossy`/preset/`layered` coat),
**film thickness** (`thinfilm`, preset), and a 2-child `mix` **`weight_map`**. The
`weight_map` case is the powerful one: because a `mix` blends whole materials, a pattern
weight makes the *material itself* — colour **and** BSDF type — vary from point to
point (checkerboard of red vs green diffuse, noise-selected metal vs glass, …). See
`scenes/procedural.ftsl`. *(GPU: patterns run on the device forward and backward
paths, including a roughness pattern on a `dielectric` (frosted glass). GPU BDPT
(mode `D`) now runs pattern-driven diffuse albedo / glossy reflect & roughness, thin-film
maps, mix `weight_map` masks, colored glass, and frosted (rough) glass on-device too
(per-hit point threaded through its MIS kernel).)*

**Pattern-driven reflectance.** The `reflect` slot takes a pattern as well, and a scalar in
a *spectral* slot is a per-hit **multiplier** on whatever the slot otherwise holds. Two
spellings, one mechanism: `reflect pattern:<name>` (or `reflect [0 1](u)`) leaves the
pattern alone in the slot, so the base spectrum is a flat 1.0 and the albedo is the
pattern's own value — a **greyscale reflectance**; `reflect_map pattern:<name>` written
beside a spectrum or a `reflect texture:<name>` **modulates that** instead. So colour always
comes from the spectrum or texture and variation from the pattern — a scalar can't invent a
hue — and the multiplier is clamped to [0,1] so a formula can't manufacture energy. Works
on `diffuse`, `translucent`, `mirror`, `halfmirror`, `glossy` and `grating`; on the families
that read their reflect spectrum directly (`fluorescent`, `thinfilm`, `dielectric`) the
loader **rejects** it rather than ignore it silently. Same code path on CPU and GPU. Worked
example `scenes/reflect_pattern.ftsl`.

**Pattern-driven transmittance.** The same two spellings on the `transmit` slot —
`transmit pattern:<name>` / `transmit [0 1](u)` alone in the slot, or `transmit_map
pattern:<name>` modulating an authored spectrum. That varies a `translucent` surface's
back-hemisphere albedo across its face (still energy-guarded so reflect + transmit ≤ 1), or
a `filter`'s per-wavelength gel transmittance — a colored gel whose density is painted by a
formula. Those two families are the only ones that read the slot at all, so a transmit
pattern anywhere else is a load error, same policy as `reflect`. Worked example
`scenes/transmit_pattern.ftsl`.

**Pattern-driven emission.** The same two spellings again, on the `emit` slot —
`emit pattern:<name>` alone (the pattern *is* the greyscale emission profile) or
`emit_map pattern:<name>` beside a spectrum (the lamp keeps its colour, the pattern meters
its brightness). A `light` block spells its emission slot `spd`, so there the pair reads
`spd pattern:<name>` / `spd_map pattern:<name>`. Paint a stained-glass window, a filament's
hot spot, an LED matrix or a gobo straight onto a lamp's surface.

Emission is stricter than the other two slots for a reason: the profile is read from **both
sides of transport** — once when a camera path lands on the emitter, once at the point NEE
or a light subpath samples on it — and MIS combines the two, so if the two readings ever
disagreed the render would be **biased**, not merely noisy. It is therefore only allowed on
the two emitter shapes whose sampled (u,v) provably equals the (u,v) a hit interpolates: a
rectangular **area** light and a **mesh** emitter. A sphere / cylinder / spot / env light is
refused at load. Two more consequences worth knowing: `power`/`lumens` normalise the
*unpatterned* spectrum (the pattern is a pure post-multiplier on radiance — which is exactly
what leaves every pdf untouched — so a profile averaging 0.5 emits about half the requested
flux), and since **0.82.0** the **GPU megakernel** implements *both* of those readings —
forward `A`/`B`/`C`, backward `R` (spectral, hero, and the fast RGB path), the photon-map
deposit and gather, BDPT `D` and VCM `U` — so a patterned lamp renders on-device in every
supported mode instead of falling back to the CPU. Worked example
`scenes/emit_pattern.ftsl`.

**UV on native primitives.** The `u v` pattern variables aren't limited to meshes.
A native `sphere {}` carries built-in equirectangular (lat/long) UVs, a `quad {}`
maps its `u`/`v` edges to (0,0)→(1,1), and an `isosurface` can request a procedural
wrap with `uv planar|spherical|cylindrical [axis=x|y|z]` — the **same projection used
for un-`vt`'d meshes**, referenced to the surface's world bounds. So a checker or
stripe authored in `(u,v)` space wraps *around* the object (a globe, tiles converging
at the poles, a grid on box faces) instead of slicing through world space. See
`scenes/uv_native.ftsl`.

**Curvature-driven, non-stationary patterns.** Every noise above is **stationary**: it
is a function of position, so its statistics are the same everywhere and translating an
object slides the pattern off it — the texture describes the *space*, not the shape in
it. Real surface ageing is the opposite. Paint wears where the form is **convex** and
something rubbed it; grime settles where the form is **concave** and nothing washes it
out. That is a property of the geometry, so no amount of position-driven noise can
express it. The `curv` variable closes the gap: it is the **mean curvature**
H = (k₁+k₂)/2 at the shading point, in 1/length units, so multiplying any noise by a
curvature mask makes it *non-stationary* — it follows the shape, and keeps following it
when the object moves.

`curv` is **signed toward the side being shaded**: convex is positive, concave negative,
flat 0. Every intersector negates it when it flips the normal to face the ray, so the
same sphere reads `+1/R` from outside and `−1/R` from within. What each geometry reports:

| Geometry | `curv` |
|---|---|
| `sphere` | ±1/radius, analytic |
| `curve` / fiber | 1/(2·radius) at the hit's interpolated radius (fibers are thin, so this is a **large** number — which is exactly what lets a pattern tell fiber from body) |
| mesh **with** `vn` | per-face, from the interpolated shading-normal field |
| mesh **without** `vn` | `0` — a flat-shaded mesh has no normal field to differentiate, so it honestly reads zero rather than guessing |
| `quad` / flat facet | `0` |
| `isosurface` | `0` (would need the field's Hessian; see `known-issues.md`) |

Two things to get right. First, **the mesh must carry vertex normals** — generate one
with `tools/make_mesh.py --smooth`, or the whole material collapses to a flat colour.
(The loader warns about exactly this: if every primitive using a curv-reading material
reports 0, it says so by name at load rather than letting you render a flat image and
wonder why.) Second, **curvature is 1/length, so an instance's scale changes it**: `scale 0.5` doubles
what the shader sees. Thresholds cut against the authored mesh will be wrong once it is
placed, which is the easiest way to end up with a flat-looking result — measure the
actual range rather than guessing.

```
# crevice grime: a noise field gated by the LEAST convex part of the form
pattern "grime" { expr "smoothstep(-5.0, -1.0, -curv) * (0.35 + 0.65*noise(11*x,11*y,11*z))" }
# edge wear: the mirror image, gated by the MOST convex band
pattern "wear"  { expr "smoothstep(5.8, 6.9, curv) * step(0.5, noise(16*x,16*y,16*z))" }
```

`smoothstep` requires `lo < hi`, so a mask that should *increase* as curvature falls is
written by negating the field (`-curv`) rather than by swapping the edges. Feeding either
mask to a `mix` `weight_map` gives the usual dirt / edge-wear pass, driven by actual
differential geometry instead of a painted or baked mask. Note that a dirt mask usually
wants "least convex", not strictly "negative" — grime collects wherever the form turns
away and nothing wipes it. Same values on CPU and GPU, including through the instance
rescale. Deterministic self-test `ftrace -checkcurv`, worked example
`scenes/pattern_curvature.ftsl`.

**Enclosure-driven patterns — `cavity`.** `curv` makes a pattern follow the shape of
the surface it sits on. `cavity` answers the question `curv` structurally cannot: how
**enclosed** is this point? It fires a short hemispherical probe of radius
`cavity_radius` and returns the blocked fraction, in `[0,1]` — `0` on an open plane,
`~0.5` in a right-angled interior corner, `→1` down a crevice. It works on every
primitive, isosurfaces included.

The two are complements:

| | `curv` | `cavity` |
|---|---|---|
| Corner between two **flat** faces | `0` on both — they really are flat | `~0.5`, the dirtiest place in the room |
| Contact between two **separate** objects | invisible; curvature is a property of one surface | a ring on **both** bodies — the probe hits whatever is there |
| A lone sphere in an empty room | `1/R` everywhere | `0` everywhere — nothing to be enclosed by |
| `isosurface` | `0` (not yet derived) | works normally |

So edge wear still wants `curv`, and contact grime wants `cavity`.

```
# grime that collects wherever the form encloses, and only as much as the noise says
pattern "crevice" { expr "smoothstep(0.06, 0.45, cavity) * noise(19*x, 19*y, 19*z)" }
```

Controlled by two `scene`-block keys. **`cavity_radius <len>`** is how far the probe
reaches, and it *is* the look: enclosure has no intrinsic scale, so the same corner is
"deeply enclosed" at a 1 cm probe and "wide open" at 1 m. Left unset it derives 2% of
the scene's AABB diagonal and prints a line saying so — a starting point, not an
answer. **`cavity_samples <n>`** (default 16) is the ray count.

The direction set is a **fixed** cosine-weighted Fibonacci spiral rather than a random
draw, and that is a design decision, not an optimisation: a pattern input is read many
times per pixel by different tracers, and every read must agree or the material itself
becomes a variance source that no amount of sampling averages away cleanly.
Deterministic makes `cavity` a true function of position — noise-free, identical on CPU
and GPU, stable frame to frame in an animation. The price is **banding** into at most
`cavity_samples + 1` levels, which is why the idiom above multiplies by noise (and why
a `smoothstep`, which quantises anyway, hides it too).

`cavity` is the only pattern input that spends **rays**, so it is gated twice: a scene
that never writes it fires none at all, and within a scene that does, only materials
whose own programs read it are charged (a `mix` inherits the flag from its layers). An
**emission** pattern may not read `cavity` or `curv` — an emitter's sampled point
carries neither, so the emitted profile would disagree with the one emission-on-hit
reads and MIS would bias the image; the loader rejects it with an explanation.

Deterministic self-test `ftrace -checkcavity`, worked example
`scenes/pattern_cavity.ftsl`.

**Distance-driven patterns — `sdf`.** `curv` reads the surface a point is *on* and
`cavity` reads how enclosed it is; neither can answer *"how far is this point from
**that** object?"* — the question behind moss creeping up from the ground, frost
thickening away from a heat source, or wear radiating out from a contact. An `sdf`
element bakes the **signed distance to one named mesh** onto a 3-D lattice and publishes
it under the ordinary `grid:` namespace:

```
mesh "ring" { file "scenes/torus.obj"  material steel  scale 0.75  translate 1 0.22 0.8 }
sdf  "halo" { object "ring"  res 128  pad 0.65 }     # res 8..512 (96); pad ⇒ ¼ longest axis

pattern "bloom" { expr "1 - smoothstep(0.12, 0.30, grid:halo(x, y, z))" }
```

Negative inside, positive outside, in world units. Because it *is* a grid it needs no new
syntax and works at every site a grid already works — a `pattern`, a material slot, an
`isosurface` leaf, and a **medium `density`/`ior` program**, which is the case the other
two cannot reach at all: a volume has no normal, no UV and no hit point to hang a surface
property off, so a spatial field is the only kind of input it can take.

`pad` is the range of the effect: outside the lattice the sampler clamps, so a
distance-keyed mask stops varying there. The default suits a band hugging the object;
reaching across a room means saying so. `res` buys spatial resolution, not accuracy —
distances are measured exactly (an exact point-triangle narrow band, then Bridson
closest-triangle sweeps), and the sign comes from the same generalized-winding
voxelization a `medium`'s `bounds { object … }` uses, so a model made of several
overlapping closed bodies reads as their **union** instead of hollowing out.

Because the bake needs the geometry, an `sdf` cannot be read by anything evaluated
*during* the load — a procedural `texture { rgb "…" }`, a `camera_curve` driver. The
loader refuses those rather than returning 0, which in a distance field would mean
"exactly on the surface".

| | `curv` | `cavity` | `sdf` |
|---|---|---|---|
| Kind | local (2nd derivative) | non-local, short probe | non-local, unbounded |
| About | the surface you are on | *everything* nearby | **one named object** |
| A ring hovering clear of a floor | `0` — the floor is flat | `~0` — nothing touches | the halo you wanted |
| Readable from a `medium` | no | no | **yes** |
| Cost | free | rays, gated per material | one load-time bake |

Deterministic self-test `ftrace -checksdf` (it bakes axis-aligned boxes, which carry no
tessellation error and whose signed distance is closed-form, and checks every lattice
sample against it exactly), worked example `scenes/pattern_sdf.ftsl`. Full syntax in
FTSL.md §6.

### Putting them together — non-stationary noise

**The problem all three solve.** Every noise primitive in the language — `noise`,
`dturb*`, `worley*` — is **stationary**: its statistics are the same everywhere. Turn
one loose on a room and every square metre gets the same grain, the same contrast, the
same feature size. Real surfaces are not like that. Wear is coarse where a hand grips
and fine where nothing touches it; frost is chunky near the cold and powdery away from
it. What changes across the object is not *how much* noise there is but *what kind*.

Nothing in the noise functions offers that, and nothing needs to: **every argument of a
noise call is an ordinary expression**, so a scene-aware field can drive it. `curv`,
`cavity` and `grid:<sdf>(x, y, z)` are those fields.

**Gate versus field — the division of labour.** The two jobs are different, and the
whole idiom is keeping them apart:

* a **gate** says *where* the effect may appear at all. `curv` and `cavity` are gates:
  cheap, about the surface's own situation, and naturally read through a `smoothstep`
  into `[0,1]`.
* a **field** says *what kind* of noise appears there. An `sdf` is the natural one — it
  is smooth, unbounded, and about a **named** object, so "coarse near the bead" is
  something you can actually write.

Multiplying a mask over noise is the *weakest* thing you can do with a field, because it
only changes **visibility** — the noise underneath is identical everywhere and the eye
reads it as a stencil. What sells non-stationarity is a change of *statistics* on a
surface that is fully covered.

| Modulate | How | Reads as |
|---|---|---|
| Amplitude | `t * noise(k*p)` | a stencil — weakest |
| Grain | `mix(noise(3*p), noise(16*p), t)` | coarse ⇄ fine |
| Detail | `noise(k*p) + t*0.30*(noise(4*k*p) - 0.5)` | plain ⇄ busy |
| Character | `noise(k*p + t*A*dturb*(…))` | regular ⇄ churned |
| Cell shape | `worley1(…, t)` with `t` a **step** | round ⇄ diamond ⇄ square |

**Trap 1 — do not vary frequency by scaling the coordinate.** The obvious way to make
noise finer in one place is `noise(k(p) * p)` with `k` driven by a field. It does not
work. The local frequency of that expression is the derivative of its argument,

```
d/dp [ k(p)·p ] = k(p) + p · dk/dp
```

so the pattern's actual frequency is **not** the `k` you asked for; it depends on how far
`p` happens to be from the *origin* (move the object and the texture changes); and
because the extra term points along `∇k`, the noise is **sheared into streaks** in the
direction the field varies. Crossfade two **fixed** frequencies instead —
`mix(noise(3*p), noise(16*p), t)`. Both operands are stationary and correct, and only the
blend moves.

**Making the crossfade actually read.** Two things sink it even when the maths is right.

First, a blend of two *independent* noises has **less contrast than either of them** — the
variance of `(a+b)/2` is half the variance of `a` — so the half-way band arrives as a flat
grey smear between two textures you can no longer see. Put the contrast stretch **after**
the blend rather than on each band before it: `smoothstep(0.38, 0.62, mix(a, b, t))`
re-normalises whatever the blend hands it, so the midpoint reads as an intermediate grain
at full strength.

Second, **match the ramp to the distances that actually occur** on the surface, and keep
it short enough that both ends own real area. Measure them, don't guess: in the worked
scene the bead hovers 0.30 m above the floor, so no floor point is nearer than 0.21 m and
the far corners of the room are only ~0.9 m away — the whole floor lives inside a single
2:1 span of distance. A `smoothstep(0.35, 0.85, …)` ramp therefore left almost the entire
visible floor between `t = 0.9` and `t = 1.0`: only the fine band was ever on screen, and
the render looked perfectly stationary even though the expression was correct.
`smoothstep(0.26, 0.58, …)` gives pure fine grain inside a 0.18 m circle under the bead,
pure coarse swell outside a 0.60 m one, and the crossfade in the annulus between.

**Trap 2 — do not vary `octaves`.** `dturb*`'s octave count is truncated to an integer
(`int oct = (int)octaves`), so a field-driven count does not fade an octave in: it
**pops** on the contour where the field crosses each whole number, leaving a visible
seam. Fade the extra octave's **amplitude** instead, centred on zero so it adds detail
without shifting the mean:

```
noise(16*x, 16*y, 16*z) + t * 0.30 * (noise(48*x, 48*y, 48*z) - 0.5)
```

The same applies to Worley's `metric`, which is rounded to `0..2`: drive it with a
**step**, never a ramp — a half-way value is not a half-way metric, it is one of the two
whole ones.

**Trap 3 — repeating the field is free, so repeat it.** The expression language has no
local variables (no `let`), and a pattern cannot reference another pattern, so a field
that gates one term and steers another has to be written out in full at every site. That
looks expensive and is not: the loader runs a common-subexpression pass over every
`pattern` program and every medium `density`/`ior` program, collapsing the repeats to one
evaluation — and, for an `sdf`, **one lattice fetch** per shading point. Set
`FTRACE_CSE_DEBUG=1` to watch it happen; on the worked example below it reports

```
[cse] pattern 2: 126 -> 71 nodes, 6 -> 1 table sample(s)
```

**Worked example: `scenes/pattern_nonstationary.ftsl`.** One `sdf` around a glowing bead
drives three surfaces, each giving the same field a different job:

```
sdf "bead" { object "bead"  res 128  pad 1.0 }

# FLOOR — no gate at all: a pure grain crossfade over the whole surface, plus a
# faded third octave. Nothing is hidden; only the statistics change.
pattern "grain" {
    expr "0.10 + 0.80 * (smoothstep(0.38, 0.62,
                             mix(noise( 4.0*x,  4.0*y,  4.0*z),
                                 noise(22.0*x, 22.0*y, 22.0*z),
                                 1 - smoothstep(0.26, 0.58, grid:bead(x, y, z))))
                         + (1 - smoothstep(0.26, 0.58, grid:bead(x, y, z)))
                           * 0.30 * (noise(60*x, 60*y, 60*z) - 0.5))"
}

# PLINTH — `cavity` gates WHERE crust may form; the field sets its grain.
pattern "crust" {
    expr "smoothstep(0.04, 0.34, cavity) *
          (0.25 + 0.75 * smoothstep(0.38, 0.62,
                             mix(noise( 4.5*x,  4.5*y,  4.5*z),
                                 noise(30.0*x, 30.0*y, 30.0*z),
                                 1 - smoothstep(0.26, 0.58, grid:bead(x, y, z)))))"
}
```

The third pattern (`wear`, on a ring) is gated by `curv` and spends the field on a
**domain warp** — nearness *squared*, so the warp stays near zero until the bead is close
and then rises fast — turning a regular pattern into a churned one without changing its
amplitude at all.

## Participating media / fog

`medium { sigma_t <v> albedo <v> g <v> rayleigh <bool> }`, or from the CLI with
`-fog <sigma_t> -fogalbedo <a> -fogg <g> [-fograyleigh]`. Henyey–Greenstein phase
function by default; Rayleigh optional.

**Rainbow (water-droplet) phase.** Add `phase rainbow { .. }` to a medium and its fog
scatters through a physically-tabulated Airy water-droplet phase instead of the smooth
HG lobe, so rain/mist actually shows a **primary bow (~42°) + secondary bow (~51°)**,
wavelength dispersion (red-outer/violet-inner on the primary, reversed on the
secondary), **Alexander's dark band**, and **supernumerary arcs**. Features are on by
default; block knobs (`droplet_um`, `secondary`, `supernumerary`, `strength`,
`forward_g`, `secondary_ratio`) tune or disable them — small drops broaden toward a
white **fogbow**. Point the camera at the antisolar point with a distant sun behind it
and keep the fog thin (single-scatter regime). Evaluated on **both CPU and GPU** across
forward A/B/C, backward R, and BDPT D — the λ×µ Airy phase table + per-λ CDF is uploaded
per-medium and importance-sampled on the device (no CPU fallback). See FTSL.md §12.

**Multiple, overlapping media.** Author as many `medium` blocks as you like — they
coexist as independent regions (e.g. two differently-tinted fog orbs plus a faint
global haze). The forward tracer superposes them physically: extinction adds (total
transmittance is the *product* of the per-medium transmittances) and each scatter is
drawn from the *earliest* of the media's independent free-flights. A single-`medium`
scene is bit-identical to before.

**Bounded, heterogeneous fog (blobs).** The medium isn't limited to a single global
haze. Add `bounds { min <x y z> max <x y z> }` to confine it to an axis-aligned box —
or `bounds { center <x y z> radius <r> }` to confine it to a **sphere** region, i.e.
simple *per-object* fog like **the whole inside of a glass sphere** (author the same
center/radius as the sphere). Or shape the fog to a **named object** with
`bounds { object "<name>" }`: a named `sphere` gives its exact analytic bound, a named
`isosurface` fills the field's interior (the fog takes the metaball/SDF silhouette
exactly, carved per-point during tracking), and a named `mesh` gives **true containment**
of the imported triangle mesh — the mesh is solid-voxelized at load into an occupancy
lattice and the fog fills its actual interior, so `mesh { file "cloud1.glb" }` shapes the
fog like a cloud rather than filling a box around it. `voxels <n>` (default 160) sets the
bake resolution on the longest axis; the fill uses generalized winding rather than parity,
so a model made of several closed bodies (or with self-intersecting shells) comes out as
their union instead of hollowing where they overlap, and 1.85 M triangles bake in about a
second. The lattice is the same dense volume format an imported `.nvdb` uses, so mesh
bounds run on the **GPU** too. Because that bake is binary and the sampler's trilinear
filter only ramps across one voxel, a mesh bound has a **hard edge** by default — right for
a body or a bottle, wrong for a cloud. `feather <metres>` fixes that: it replaces the 0/1
occupancy with a smoothstep of the **exact Euclidean distance** to the outside (Felzenszwalb
& Huttenlocher's separable transform, three linear-time sweeps — isotropic, no axis-aligned
banding), so density rises from zero at the surface to full that far in and the silhouette
becomes a falloff *zone* the way real cloud edges are. Pair it with
`mesh { … shape_only yes }`, which loads a mesh purely as a shape and strips its triangles
from the scene once the bound is baked, so the shell defining the fog isn't also drawn
around it. An *open* fog sphere is directly viewable in every mode.
Fog (and any diffuse surface) seen through a **glass sphere** *is* imaged directly by the
pinhole splat `B`, via the **analytic specular connection**: for each glowing haze in-scatter
(or Lambertian surface) vertex the renderer solves the refracted eye ray that reaches the
camera through the sphere in closed form (a planar reduction to a 1-D root solve, with a
ray-differential Jacobian for the splat weight), so a lantern glowing inside a fogged glass
orb — and the fly-through *through* that orb — renders correctly rather than black. The
solve evaluates the ior at the photon's own wavelength, so the refraction is dispersive for
free. It runs on both CPU and GPU. This currently covers **glass spheres in mode `B`** (both
surface and volume vertices); the finite-lens splat `A`, photon-catch `C`, and non-spherical
dielectric shells are not yet covered by the analytic path — for those, seeing the fog
through the curved glass is a refracted (specular↔volume) path that the straight camera
connection can't bend, so that direct view renders black (the fog still correctly **lights
the surrounding room** indirectly). **BDPT `D` images fog-through-glass for any shape**: its
camera subpath refracts through the shell (specular vertices) to a volume in-scatter vertex,
then MIS-connects bidirectionally to the light. Photon-catch `C` traces the same path but
far more slowly (the fog-scattered photon must refract out and hit the pupil).
Add `density "<expr>"` (or `density pattern:<name>`) —
a scalar field over world `x y z` (the same infix expression language as isosurface
`function` fields) that scales `sigma_t` per point — for **fog blobs with soft, formula-defined
boundaries**: e.g. a smooth radial falloff `pow(saturate(1 - dist/R), 2)` renders a
glowing sphere of haze whose edge fades gradually instead of a hard surface. Sampling is
unbiased **delta (Woodcock) tracking** for scattering and **ratio tracking** for shadow
transmittance — exact, no voxelization. A majorant `density_max` is auto-estimated over
`bounds` (or set explicitly). Heterogeneous/bounded fog is honored by the **forward**
modes (A/B/C) on **both the CPU and the GPU** (the device runs the identical density VM +
delta/ratio tracking). **BDPT `D`** renders media of **every kind** — global haze,
multiple superposed media, box/sphere/object-**bounded** fog, and **heterogeneous
`density`-field blobs** — unbiased on both the CPU and the GPU: subpath medium vertices are
placed by delta tracking (analog throughput) and connection edges weighted by ratio-tracking
transmittance, exactly as the forward tracer samples them. (The MIS weights omit the
heterogeneous distance-pdf / transmittance — a variance-only simplification per PBRT-v3;
the balance heuristic is a partition of unity so the estimator stays unbiased regardless.)
The backward reference (R/V) and the P composite treat the medium as a single global homogeneous haze
and warn if you author `density`/`bounds` for them. See `FTSL.md` §12.1.

**Imported volumes (`.nvdb` / `.vdb`).** Instead of a formula, point the density field at a
real sparse volume: `density vdb:<path>` imports either a NanoVDB **FloatGrid** (`.nvdb`, the
compact GPU-friendly form) **or a native OpenVDB `.vdb`** directly — the loader dispatches on
the file magic. The native `.vdb` path is a **self-contained reader** (no OpenVDB/NanoVDB
dependency): it parses the file container, tree topology (`float 5_4_3`) and
BLOSC+ACTIVE_MASK+HalfFloat leaf buffers by hand, decoding blosc's **LZ4** codec with a
vendored single-file LZ4. (Other blosc codecs — BloscLZ/Zlib/Zstd — and ZIP are reported with
a clear "re-export with LZ4" message; validated bit-for-bit against python-blosc on the
official OpenVDB smoke/sphere/cube samples.) On load the grid is **baked into a dense lattice**
(stored as **fp16 half-floats** to halve host RAM and GPU VRAM — density fields tolerate half
precision's ~0.05% error easily) plus a world→index affine,
so the *identical* trilinear sampler runs on the CPU and the GPU and any affine map
(translation/scale/rotation) is honored. On the **GPU the lattice is uploaded as a native
sparse brick grid** (8³ bricks; only bricks with a nonzero voxel reach the device, plus a small
int32 brick-index), so **VRAM scales with occupied volume, not the bounding box** — a mostly-empty
plume can drop to a fraction of the dense footprint (a startup line reports the ratio). The sparse
sampler is bit-for-bit identical to the dense one. The grid's
world AABB auto-seeds the medium bound and its peak value the delta-tracking majorant — so
`medium { sigma_t 40  albedo 0.9  density vdb:cloud.nvdb }` is all it takes to light an
imported cloud. Values are treated as a dimensionless density multiplier on `sigma_t`, so
you still dial the optical thickness with `sigma_t`. Only **float** grids are supported; the
host bake is **dense** (host RAM ~ the grid's index-space bounding box, bounded by a safety cap)
while the **GPU sampler is natively sparse** (bricked, see above). Works in the forward modes
(A/B/C) and BDPT `D` on CPU and GPU, exactly like a
`density` formula. Generate a test asset with `scraps/make_nvdb.cpp`.

**Gradient-index (GRIN) media — bending light.** Give a
medium an `ior "<expr over x y z r>"` field (or `ior pattern:<name>`) and it becomes a
**gradient-index region**: rays that enter its `bounds{}` no longer travel straight — they
**bend continuously**, integrating the Eikonal ray equation `d/ds(n·dr/ds)=∇n` with a
small symplectic march step (`ior_step <v>`, default 1/64 of the smallest bound extent).
This makes mirages, hot-air shimmer, and **gradient lenses that focus/warp with no glass
surface at all**. E.g. `medium { bounds { center 0 0 2 radius 0.9 } ior "1.6 - 0.6*(sqrt(x*x+y*y+(z-2)*(z-2))/0.9)" }`
is a radial index ball (n=1.6 core → 1.0 rim) that visibly lenses a checkerboard behind it
(`scenes/grin_lens.ftsl`). GRIN bending runs on the **forward light tracer (modes `A`/`B`/`C`)
and the backward reference (mode `R`), on both CPU and GPU** — all share one symplectic
marcher (the GPU carries its running Eikonal state in double to match the CPU; a small
bent-region float-vs-double residual on the GPU is noted in `known-issues.md`). Only **BDPT
(mode `D`)** still refuses GRIN (its straight-line connection geometry would be biased) — use
`A`/`B`/`C` or `R` for a GRIN scene.

**Authoring media procedurally (loom).** The [loom toolkit](tools/loom/README.md) emits
these `medium {}` blocks from a `loom.Volume(...)`: `sigma_t` / `albedo` / `g` are
animatable `Signal`s, and a `density` is any loom `SpatialExpr` field (the same
`X Y Z T` DSL that drives its isosurfaces), so an animated procedural cloud/fog is emitted
as an inline `density "<expr>"`. Bound it with `box=` / `sphere=` / `obj=`, cap the majorant
with `density_max=`, or point `density="vdb:<path>"` at an existing NanoVDB grid (loom
references sparse volumes but doesn't generate them). E.g. a sphere-bounded procedural fog
blob: `Volume(sigma_t=8.0, albedo=0.9, g=0.4, density=0.6 + 0.4*sin(8*X)*sin(8*Y)*sin(8*Z),
sphere=((0.5, 0.45, 0.5), 0.32), density_max=1.2)`.

---

## Scene language (FTSL)

> **Full reference: [`FTSL.md`](FTSL.md)** — the complete grammar (every block, key,
> default, spectrum/pattern/material form, and parsing quirk). The overview below is a
> quick tour; `FTSL.md` is the authoritative spec.

An FTSL file is a list of blocks. Top-level block types: `scene` (the
`units …` / `spectral …` header), `material`, `texture`, `pattern` (procedural scalar
field), `spectrum`, `sphere`, `quad`, `triangle`, `mesh`, `curve` (a hair/fur/wire
strand), `isosurface` (implicit SDF
surface / CSG / metaballs / arbitrary `function` formulas), `light`, `group`, `medium`,
`camera`, `camera_path` (keyframed camera animation), `camera_orbit` (turntable /
fly-around: N frames on a circle around a `center`, for MP4 orbits), `camera_curve`
(spline fly-through with variable speed), and `render` (render-setting overrides). See the `scenes/` directory for worked examples
(`cornell.ftsl`, `fisheye.ftsl`, `spotlight.ftsl`, `envlight.ftsl`,
`material_presets.ftsl`, `realcam.ftsl`, `implicit.ftsl`, `function.ftsl`,
`procedural.ftsl`, `uv_native.ftsl`, `showcase_orbit.ftsl`, `translucency.ftsl`,
`gallery.ftsl` (a large room packed with varied materials around a gold gyroid),
`gallery_rain.ftsl` (the same hall with weather in it: a fog cloud whose *shape* is a
mesh, a rain curtain under it using the `rainbow` phase function, and a ceiling slot plus
a distant-sphere sun sited so a real 42° primary bow lands in frame), …).

**Scene-header defaults (`default_mode`, `fps`).** The `scene { … }` header can set
two project-wide defaults alongside `units`/`spectral`:

- **`default_mode <letter>`** — the render mode to use when *nothing else* selects one:
  no `-mode` on the CLI, and the camera/render blocks don't author their own `mode`.
  It's the lowest-priority source, so the resolution order is `-mode` (CLI) → a camera's
  own `mode` → `default_mode` → the built-in `B`. Handy when several cameras would
  otherwise all repeat the same `mode M`.
- **`fps <n>`** — the default playback rate for flyby animations, read by the assembly
  tooling (e.g. `tools/showcase_flyby.py` when `--fps` is omitted). A `camera_curve`/
  `camera_path`/`camera_orbit` block can override it with its own `fps <n>`; the tool's
  resolution order is `--fps` → the flyby's `fps` → the scene-level `fps` → `30`. `fps`
  is purely a playback hint — it doesn't change what ftrace renders.

**Render-setting block (`render { … }`).** A top-level `render` block carries defaults for
settings that otherwise come from the CLI — `photons <n>`, `mode <letter>`, `res <px>`,
`device <name>`, `out <file>`, and **`max_bounce <n>`**. The matching CLI flag always wins;
the block is for settings a scene *needs* rather than ones an operator prefers.

`max_bounce` is the clearest case. Modes `D`/`U` run **8** path edges by default (see
`-max-bounce`), which is not enough for deeply nested dielectrics: `scenes/gallery.ftsl`'s
Klein bottle is a 2.4 mm-walled glass shell with another tube *inside* it, so one line of
sight crosses about eight interfaces and the innermost tube renders as a solid **black
plug** — truncated paths, not a material bug. That scene therefore declares
`render { max_bounce 32 }` and looks right without the operator having to know. When a scene
sets it, the run prints `[scene] max bounce = N (from the scene's render block)`.

### Conditional blocks (`prefer { … } else { … }`)

Some features aren't renderable in every mode — most notably **gradient-index (GRIN)
media** and **non-rectilinear (fisheye/panoramic) cameras**, which the bidirectional
modes (`D` BDPT, `U` VCM) can't handle because their path-connection geometry assumes
straight edges. Rather than maintaining two separate scene files, wrap the
mode-sensitive blocks in a `prefer { … } else { … }` chain:

```
prefer {
    camera "cam" { … mode D … }      # fast, robust — but no GRIN
    medium  "lampgas" { … }           # (plain, non-GRIN)
} else {
    camera "cam" { … mode B … }      # slower, but renders everything
    medium  "lampgas" { … ior … }     # GRIN version
}
```

- Each **branch** is a complete set of top-level blocks (so it carries its own camera
  mode *and* its own media — the two travel together, dissolving the "which mode
  supports which medium" circularity).
- `else` chains **flat** — `prefer { A } else { B } else { C }` — and you may **not**
  nest a `prefer` inside a branch.
- At load time the resolver **trial-builds each branch in order and picks the first one
  that's renderable** under the active mode; if none qualify it falls back to the last
  branch. It prints `[prefer] branch N rejected (<reason>); trying the next` and
  `[prefer] using branch N of M` so you can see which won.
- Only **cameras** and **media** (the features with real mode gaps) participate in the
  support test; everything else always builds.

The showcase (`scenes/gallery_settled.ftsl`) uses this to render in mode D today while
keeping a mode-B "full-effects" branch (with the GRIN lamp gas) ready for the future.
See also `-on-unsupported` under the command-line reference, which controls what
happens when the *selected* mode still can't render a feature (error / fall back to
mode R / strip the feature).

### Camera animation (`camera_path`, `camera_orbit`)

Both expand into a sequence of frames sharing look_at/up/fov/mode/film/lens; a
multi-camera render writes one file per frame (`_<name>` inserted before the
extension), which ffmpeg concatenates into a video. Any flyby block may carry an
`fps <n>` playback hint (read by the assembly tooling; overrides the scene-level
`fps` default — see *Scene-header defaults* above).

**Where the frames land.** A path/orbit/curve of more than one frame is written into
its **own subdirectory**, named from the output stem and the block, rather than loose
beside `-o`: `-o png/rain.png` with a `camera_path "fly"` writes
`png/rain_fly/rain_fly000.png`, `…_fly001.png`, and so on (the directory is created if
needed). Standalone `camera` blocks in the same scene still write beside `-o` as
`png/rain_<name>.png`. This keeps a 600-frame flyby from burying the stills — and any
unrelated images — in its output directory. To render just one image from a scene that
also declares a path, pass `-camera <name>`.

- **`camera_path "name" { … key <t> <ex ey ez> [<lx ly lz>] [<fov>] … frames N }`** —
  keyframed fly-through: the eye (and optionally look_at / fov) is linearly
  interpolated across `key` frames. Optional `dolly_zoom` holds the subject's
  on-screen size (Vertigo effect); optional `exposure_lock [selector]` shares one
  auto-exposure across all frames (metered from a selectable viewpoint, default the
  path average — see below).
- **`camera_orbit "name" { center <x y z> radius <m> [height <m>] [axis x|y|z] frames N
  [start_deg <d>] [sweep_deg <d>] [look_at <x y z>] [exposure_lock [selector]] }`** — a turntable /
  fly-around whose eye rides a circle around `center` (the default look_at). The circle
  lies in the plane perpendicular to `axis` (default y); `height` offsets the eye along
  the axis. A full 360° sweep is sampled so frame N == frame 0 (seamless loop); a
  partial sweep spans its endpoints. See `scenes/showcase_orbit.ftsl` (an orbit tuned
  to fly straight *through* a glass sphere).
- **`camera_curve "name" { point <x y z> … [frames N] [density <ρ> | density_at <t> <ρ> …]
  [spline uniform|centripetal|chordal|<alpha>] [look tangent [min_reach <f>] [look_smooth <n>] |
  look_at <x y z> | look curve + look_point <x y z> …] [closed] }`** — a
  fly-through along a **Catmull-Rom spline** that passes through the `point` control
  points. `spline` selects the parameterization: `uniform` (α=0, the default — simple but
  can **overshoot** and swing wide between unevenly-spaced control points), `centripetal`
  (α=0.5 — the recommended choice; provably no cusps or self-intersections, stays tight to
  the control polygon, so an irregularly-spaced fly path reads smooth instead of lurching),
  or `chordal` (α=1.0); a bare number sets α directly. Camera placement is either a fixed
  `frames` count (uniform arc length) or a
  **density** (cameras per unit length) that can vary along the curve via `density_at`
  keyframes — this is the camera's *speed*: high density = many closely-spaced frames =
  slow dwell, low density = fast. `density_at`'s `t` is the normalized **arc-length**
  position along the curve (`t=0.5` is half-way *by distance*, **not** the middle control
  point), so a dwell stays on the beat it was measured for even when the waypoints around
  it are unevenly spaced — which on a real flight path they always are.
  Aim along the travel tangent (default), at a fixed
  `look_at`, or at a second `look curve`. The **travel tangent is fold-robust**: where the
  path makes a sharp horizontal U-turn its look-ahead chord loses horizontal reach and would
  otherwise rake the view steeply up into the ceiling / down at the floor, so `min_reach <f>`
  (default `0.5`, `0` = legacy) floors that reach for the pitch calculation and `look_smooth
  <n>` (default `0`; a Gaussian sigma in frames) temporally smooths the look direction so a
  fold reads as a bounded near-level pan instead of a flick. **Orientation and lens can also be animated**
  per frame over the normalized timeline `t ∈ [0,1]` (`t=0` first frame, `t=1` last),
  each keyframed by `<name>_at <t> <value>` (piecewise-linear and flat-clamped at the ends,
  like `density_at` — but note these tracks run on the **frame** timeline `i/N`, not on
  `density_at`'s arc length) or held constant by the bare keyword: **`roll[_at]`** banks the
  camera about its view axis (the third orientation degree of freedom), and
  **`fov_at` / `zoom_at` / `fstop_at` / `focus_at`** animate the vertical field of view,
  focal-length multiplier, f-number, and focus distance. (`fstop`/`focus` change depth of
  field only in the physical catch modes `A`/`C`; in the pinhole splat `B` the aperture is
  virtual, so there `roll`/`fov`/`zoom` are the visible ones. Lens *projection*/fisheye is
  a discrete whole-flight mode, not a continuous track — set it once with `projection`.)
  Any of these scalars can instead be driven by a **parametric record** (see *Parametric
  records* below) with **`<name>_from RECORD.channel[(driver)]`** — the channel is sampled
  over the flyby timeline, the optional driver defaults to the raw `t` and may be any
  expression in `t` (`fov_from zoom.fov(t*t)` eases in), and the record's `interp`
  (nearest/linear/smooth) shapes the curve. A record track overrides an `_at` track, which
  overrides the constant; a linear record reproduces the matching linear `_at` keyframes
  frame-for-frame. (The driver sees **only** `t` here — surface variables like `u`/`x` are
  out of scope and error.)

- **Two-axis camera orientation — `fwd_at` / `up_at` / `frame` / `fwd_frame` / `up_frame`.**
  The camera basis is set by a **forward** axis and an **up** axis (`right` is always
  derived, never authored). Each axis is read in a **reference frame** — `frame world|travel`
  sets the default for both and `fwd_frame`/`up_frame` override it per axis. `world` is the
  fixed world axes (classic behavior); `travel` is the curve's **rotation-minimizing frame**
  (RMF), a twist-free moving basis parallel-transported along the path (double-reflection
  method, *not* the flip-prone Frenet frame) so the shot **banks into turns** — and on a
  `closed` loop the residual twist is distributed so the frame closes seamlessly. Forward
  (2 DOF) comes from `fwd_at <t> <x y z>` direction keyframes, else `look_at`/`look curve`,
  else the tangent; up (1 DOF) comes from `up_at <t> <x y z>`, else `roll`/`roll_at`, else the
  reference up. A `fwd_at`/`up_at` vector is read **in its axis's frame**: under `travel` its
  components are `(right, up, forward)` in the RMF basis, under `world` a plain world
  direction. Authoring none of these keywords reproduces the legacy world-up framing exactly.

**`exposure_lock` — one shared auto-exposure across a whole path.** On any
`camera_path`/`camera_orbit`/`camera_curve`, `exposure_lock` freezes a single
auto-exposure anchor and applies it to *every* frame of that path, so a fly-through
doesn't pump brighter/darker as the framing changes (the flicker you'd get if each
frame metered itself). A **selector** chooses which viewpoint the whole path meters
from — before any frame renders, a quick reduced-sample **meter pre-pass** renders
just that viewpoint, computes its exposure, and locks the path to it (the same happens
in the `-raster` preview, so preview and final agree):

  - **`exposure_lock`** (bare, or `on`) — meter the **average** across *all* frames of the path (the **default**: a robust compromise that won't expose the whole flythrough for one possibly-atypical opening frame). Aliases `average`/`avg`/`mean`.
  - **`exposure_lock first`** — meter the **first** frame (deliberately "expose for the establishing shot").
  - **`exposure_lock index <i>`** (alias `frame`) — meter frame **`i`** (0-based; negative counts from the end, so `-1` = last frame).
  - **`exposure_lock near <x> <y> <z>`** — meter whichever frame's **eye is nearest** the world point `x y z`.
  - **`exposure_lock camera "name"`** (or just **`exposure_lock "name"`**) — meter a **separately-defined `camera "name"`** — a purpose-built metering viewpoint that need not be on the path at all.
  - **`exposure_lock off`** (alias `false`/`0`) — disable; each frame meters itself.

  The selector is **always honoured** — the meter pre-pass renders the chosen viewpoint
  in its own render mode (`A`/`B`/`C` forward, `R` backward, `D` BDPT, `M` photon map,
  `P` composite; anything else falls back to a general forward light-trace), all of which
  converge to the same scene brightness, so there is **no silent "just use frame 0"**
  fallback for any mode. Absolute-EV scenes have no auto-exposure to lock, so
  `exposure_lock` is a no-op there. The global `-exposure-lock` CLI flag instead locks
  *all* rendered cameras to one anchor (metered from the first frame), overriding
  per-path selectors.

  **Both are process-local.** A scene `exposure_lock` and `-exposure-lock` can only
  share an anchor between frames rendered by the *same* `ftrace` invocation — they hold
  the anchor in memory and it dies with the process. A sequence rendered **one frame per
  invocation** (loom's `render_range`, a batch loop, or a single frame re-rendered
  later) therefore still meters every frame independently, and can still flicker. For
  that case use **`-exposure-anchor <value|file>`**, which carries the anchor across
  processes: given a *number* it locks to it outright; given a *path*, the first run
  meters, writes its anchor there, and every later run pointed at the same file loads
  and reuses it. It implies `-exposure-lock`.

  **Why per-frame metering can jump at all.** The auto-exposure anchors on a single
  fixed-rank order statistic — the 99th percentile of per-pixel `max(r,g,b)` — and that
  statistic solves `area(L) = 1%` for the level `L`. On a scene with a bright, compact
  specular population the density up there is very thin (measured on `pastel_jack_ring`:
  ~**0.25% of frame per octave** above p95), so the inversion is ill-conditioned —
  roughly **5 octaves of level per 1% of area**. A rotating highlight changes that area
  by a few hundredths of a point, and the anchor moves a third of an octave even though
  nothing about the picture's brightness did. Measured over that flyby (432 frames,
  static camera, rotating gold ring): the worst single-frame step was **36.2% in the
  anchor** while the static background moved **2.2%**, p95 **0.7%**, the median **0.6%**
  and the whole-frame log-average **1.0%**. A shared anchor removes the jump (worst
  full-frame step after repair: **0.8%**).

  **A shared anchor is the fix, not a workaround.** Nine studies measured six replacement families — other ranks, rank-band
  blends, a `min(p99, C·p95)` clamp, power means, energy quantiles, and the Reinhard
  log-average key — against *stability* on the sequence and *fidelity* to today's exposure
  across 186 ordinary renders. None satisfies both, and not by a tuning margin: the stable
  candidates land **1.5–7.7 stops** adrift. The reason is structural — where p99
  misbehaves its value is genuinely arbitrary, so there is nothing to be faithful to. Nor
  can the bad case be *detected* and specially handled: by local density at the anchor,
  `pastel_jack_ring` is denser than a quarter of ordinary renders. A single frame contains
  no evidence distinguishing "the highlight rotated" from "the scene got brighter"; that
  information exists only across frames, which is exactly what a shared anchor uses. The
  residual caveat is narrow: a **single still** is still metered by p99 and can anchor on
  an atypical glint, with no neighbour to reveal it — use `-ev` or `-exposure-anchor
  <value>` if it does. See `known-issues.md` for the full measurements.

  **Repairing an already-rendered sequence.** If the frames were rendered with
  `-checkpoint`, no re-render is needed: each `<frame>.png.ftbuf` still holds the raw
  linear film, so `ftrace -topng <ftbuf> <png> -exposure-anchor <value>` re-develops it
  at a shared gain in milliseconds. loom's `loom.stabilize_exposure(pngs)` automates
  exactly that — it develops each checkpoint once to read back the anchor it *would*
  have chosen, takes the **median** across the sequence (not the first frame, which on
  `pastel_jack_ring` sat at the 93rd percentile — a full stop off), then re-develops
  every frame at that median. `render_range(..., stabilize=True)` (the default) runs it
  automatically after the last frame.

### Multi-camera shared photon pass (modes `A`, `B`, and `M`)

When several cameras render at once (multiple `camera` blocks, or the frames a
`camera_path`/`orbit`/`curve` expands into) in a **forward next-event** mode, the
tracer flies **one** photon set and splats every vertex to **all** cameras of that
mode at once, instead of re-flying the photons per camera. This is the "many cameras
for one photon set" win — emission, BVH traversal, and scattering are paid once. It
runs on **both the CPU and the GPU** (`-device gpu`), and applies to the two forward
splat models:

- **`B` (pinhole splat)** — `connect()` draws no random numbers, so the shared pass is
  **bit-identical** to rendering each camera on its own.
- **`A` (finite-lens camera)** — each camera samples its own aperture pupil (draws
  RNG), so the shared photon flight is **unbiased per camera** but matches a standalone
  render **in distribution**, not bit-for-bit. Rectilinear cameras only.

**Mode `M` (photon map) shares even more cheaply.** Because the photon map is
**view-independent**, a multi-camera mode-`M` render builds the map **once** and runs
each camera's backward density gather against that one shared map — the whole forward
photon flight amortizes across every frame. Unlike `A`/`B` (which reuse a photon
*flight*, so every camera inherits the *same* fixed noise), each mode-`M` camera gathers
with its **own** independent backward samples, so frames share only the underlying
radiance solution, **not** the noise. That makes `M` safe to share across
**exposure-locked** `camera_path` frames too (it isn't restricted to per-frame
auto-exposed cameras the way `A`/`B` sharing is) — the ideal mode for a flythrough of a
static scene.

The `A`, `B`, and `M` cameras form **separate** shared passes (`A` perturbs the RNG
stream during the trace, `B` doesn't, and `M` gathers backward instead of splatting).
`A`/`B` sharing applies to any per-frame-auto-exposed group; `M` sharing applies to any
group (including exposure-locked paths).

The `A`/`B` shared pass is **crash-safe and resumable** just like the single-camera
forward path: it traces the group's one photon flight in accumulation chunks (each
seeded off the cumulative photon count so successive chunks draw independent photons),
drives the live `-window`, and every chunk writes each camera's image plus a per-camera
`<out>.ftbuf` checkpoint. So `-checkpoint`, `-resume`, `-time`, `-noise`, and `-forever`
all work **while still sharing** the flight — a crash or Ctrl-C loses at most one
interval, and `-resume` reloads every camera's film and continues (the whole group
resumes together; a half-written or mismatched sidecar set falls back to a fresh start).
(`-resume`/budget flags still render per camera for mode `M`, whose per-camera gather is
independent anyway.)

**Shared vs. independent randomness across cameras (matters for video and for
side-by-side cameras).** This is the key per-mode difference in how randomness is
distributed *between* cameras. Note that **a "frame" here is simply a camera in the same
scene**: a `camera_path`/`orbit`/`curve` expands into one `camera` per frame, and they all
render together in a single scene exactly like several hand-authored `camera` blocks — so
everything below applies identically whether you wrote the cameras out by hand or generated
them as animation frames:

- **`B`** — every camera is splatted from the *same* photon set, so they share identical
  random paths: a camera's noise is **correlated** with every other camera's (and a camera
  rendered in the group is bit-identical to rendering it alone). Across an animation the
  grain drifts coherently frame-to-frame rather than reshuffling.
- **`A`** — cameras share the photon *flight* but each draws its own aperture-pupil
  samples, so each carries **independent** randomness on top of the shared paths (unbiased
  per camera; correlated only through the shared flight).
- **`M`** — cameras share the photon *map* (the radiance solution) but each runs its own
  backward density gather, so each frame's noise is **independent** — the best of both:
  the expensive forward pass is paid once, yet frames don't inherit a shared grain.
- **`C`/`R`/`D`/`P`/`V`** — each camera is traced **fully independently** with its own
  sample budget, so their randomness (and noise) is **uncorrelated** by construction.

Mode `B`'s correlation is usually invisible (and cheaper), but if you want independent,
film-grain-like noise per camera/frame, render them separately (e.g. via a budget flag,
which falls back to per-camera passes) so each draws its own photons.

> **Other modes do NOT save time with multiple cameras.** `C` (finite-aperture catch)
> consumes each photon at the first aperture it hits, so it can't share a photon set; and
> `R`, `D`, `P`, and `V` are camera-anchored estimators that trace **from** each camera —
> a multi-camera render of those modes simply renders **each camera independently**
> (re-tracing the full sample budget per camera), so it costs the same as running them
> one at a time. Only `A`, `B`, and `M` amortise the forward trace across cameras.

### Stereoscopic 3-D (`-stereo`)

`-stereo <mode>` turns any render — a still *or* every frame of a `camera_path` movie —
into **3-D stereoscopic output**. Each selected camera is rendered **twice** (a Left and a
Right eye) and the two images are fused into the `-o` file:

- **`-stereo sbs`** — side-by-side **wall-eyed** (Left\|Right), for free-viewing or a
  parallel-view stereoscope. Output is `2·resX` wide.
- **`-stereo cross`** — side-by-side **cross-eyed** (Right\|Left), for the cross-your-eyes
  free-viewing technique.
- **`-stereo anaglyph`** — **red-cyan** glasses. Uses the **Dubois** least-squares colour
  matrices (far less ghosting / retinal rivalry than a naïve channel split). Same
  resolution as a mono render.
- **`-stereo anaglyph-gm`** — **green-magenta** Dubois anaglyph.

**Off-axis rig (why it's comfortable).** The two eyes are *parallel* cameras offset along
the camera **right axis**, each with an **asymmetric (sheared) frustum** that shares a
single **convergence plane**. This is the correct method: the naïve "toe-in" (rotating the
two cameras to cross) introduces **vertical parallax** that causes eye strain, which the
off-axis shear avoids entirely. The convergence plane is the depth that appears *at the
screen* (zero parallax); objects nearer than it pop out toward you, farther objects recede
behind the screen.

**Physical geometry.** The baseline (eye separation in the scene) and convergence are
derived from the real viewing setup, so the depth reads naturally:

- **`-eye-sep <m>`** — your interocular distance (default `0.063` m).
- **`-view-dist <m>`** — how far you sit from the screen (default `0.6` m).
- **`-dpi <n|auto>`** — screen pixel density. Given a number, the screen's physical width
  is `resX·0.0254/dpi`. `auto` reads the Windows *logical* system DPI (a rough hint). If
  you omit `-dpi` (the default), the screen width is taken as the camera's horizontal field
  seen at `-view-dist` (`W = 2·view-dist·tan(½·fovX)`).
- **`-convergence <m>`** — the convergence-plane distance in **scene units** (default: the
  camera's look-at target distance).

From these the frustum shear is `S = eye-sep / screen-width` (so a point at **infinity**
lands exactly one interocular apart on screen — parallel gaze, the comfortable far limit),
and the baseline is `b = 2·convergence·tan(½·fovX)·S`. Equivalently `b/convergence =
eye-sep/screen-width`: **the camera's separation relative to its subject equals your eyes'
separation relative to the screen.** Because it's expressed as that ratio, the same flags
give sensible depth at any scene scale.

Both eyes share a single auto-exposure anchor, so Left and Right — and, for an
**exposure-locked** `camera_path`, *every frame* — tone-map identically (no L/R brightness
mismatch or stereo shimmer). Each eye rides the full render pipeline (checkpoints, budgets,
GPU, the live `-window`), so nothing else about how you render changes. The intermediate
per-eye PNGs are deleted after compositing unless you pass **`-stereo-keep-eyes`**.
Rectilinear cameras only — a fisheye/panoramic camera renders mono with a warning.

```
# red-cyan anaglyph still, physical defaults, convergence on the look-at target
ftrace -in scene.ftsl -mode B -n 2e8 -stereo anaglyph -o png/scene3d.png -keepwindow

# wall-eyed side-by-side, wider baseline via an explicit near convergence plane
ftrace -in scene.ftsl -mode B -n 2e8 -stereo sbs -convergence 1.5 -o png/scene_sbs.png -keepwindow

# a whole exposure-locked flyby in green-magenta 3-D (one composite per frame)
ftrace -in scene.ftsl -camera fly -stereo anaglyph-gm -o png/fly/fly.png
```

### Animated geometry (OBJ sequences) → video

Camera animation (above) moves the camera over **one static scene**. To animate the
*geometry* itself — a cloth/fluid sim, a growing crystal, a Blender/Houdini point-cache
baked to one OBJ per frame — use **`tools/obj_sequence_to_video.py`**, a self-contained
driver that renders each OBJ frame with ftrace and encodes the frames into an MP4 with
ffmpeg (no new renderer dependencies).

You supply a **template** scene (camera, lights, materials, render mode) with a `{obj}`
placeholder where the animated mesh goes; the driver substitutes each frame's OBJ, renders
`frame_NNNNN.png`, then ffmpeg concatenates them:

```
# make a starter template, then edit its camera/lights/materials
python tools/obj_sequence_to_video.py --write-template anim.ftsl

# render the sequence to a 24fps clip (~4s/frame, mode B, on the GPU)
python tools/obj_sequence_to_video.py "cache/*.obj" --template anim.ftsl \
    -o png/growth.mp4 --mode B --device gpu -r 960 --time 4 --fps 24
```

The template's mesh block just references the placeholder: `mesh { file "{obj}" … }`
(other tokens: `{frame}`, `{frame1}`, `{obj_stem}`). `FRAMES` is a directory of `*.obj` or
a quoted glob, naturally sorted. Per-frame budget is `--time`/`--spp`/`--noise`; other
useful flags: `--resume` (skip already-rendered frames), `--start/--end/--step` (sub-range),
`--encode-only` (re-encode existing PNGs at a new `--fps` without re-rendering),
`--no-encode`, `--keep-frames`, `--crf`/`--codec`/`--pix-fmt`, and `--dry-run`. Run with
`--help` for the full list.

### The shared grammar

FTSL's syntax is written **once**, as a formal grammar:
`tools/loom/loom/grammar/ftsl_scene.epeg`. It is compiled to a parser graph that both
loom (in Python) and ftrace (as generated C++, `src/gpda/ftsl_scene.gen.cpp`) consume,
so the language has exactly one definition and the two tools cannot drift apart.
**As of 0.68 the shared grammar is ftrace's front end** — it parses every scene you
load, and the `.epeg` file is the single source of truth for what FTSL accepts. As of
**0.79 it is the *only* front end**: the hand-written parser it replaced is gone from
the source tree entirely.

Besides removing the drift risk, the grammar parser gives much better errors. Because
it walks a graph of cursors and expands only to *terminals*, at a failure the live
cursor set is exactly the set of continuations the grammar would have accepted, and
the cursor stacks name the enclosing rules:

```
scene.ftsl:1:15: unexpected NEWLINE '\n'; expected '{'
    (in brace_body < plain_header < top_block < item)
```

where the retired hand-written parser could only manage `line 1: expected '{' after
material`.

**The transition is over.** Two flags covered it while the two parsers ran side by
side — `-legacy-parser` (parse with the old hand-written parser instead) and
`-validate-grammar` (parse *both* ways and structurally diff the block trees). The
differ is what earned the flip: the two parsers agreed on **all 2595 `.ftsl` files in
the tree**, structurally identical down to per-statement line numbers, and held there
for ten releases with `-legacy-parser` available and unused. **0.79.0 deleted the old
parser, the escape hatch and the differ.** Both flags (and the `FTRACE_LEGACY_PARSER` /
`FTRACE_VALIDATE_GRAMMAR` environment variables) are retired: the flags are still
*accepted* on the command line so an existing script doesn't break, but they print a
one-line "retired … ignoring" notice and do nothing.

### Importing Mitsuba scenes

`tools/mitsuba_to_ftsl.py` converts a Mitsuba (0.6 / 2 / 3) XML scene to FTSL:

```
python tools/mitsuba_to_ftsl.py scene.xml scene.ftsl
ftrace -in scene.ftsl -mode D -o out.png
```

Mitsuba is also a spectral, physically-based renderer, so most constructs map
almost 1:1: the `perspective`/`thinlens` sensor → an FTSL `camera` (`thinlens`
becomes mode `A` with aperture + focus), `diffuse`/`conductor`/`roughconductor`/
`dielectric`/`plastic` BSDFs → `diffuse`/`mirror`/`glossy`/`dielectric` materials,
and `area`/`constant`/`envmap` emitters → FTSL lights. `rectangle`, `cube`,
`sphere`, and `obj` shapes are supported (with full `to_world` transforms); RGB
reflectances ride FTSL's Jakob–Hanika upsampling and measured/blackbody spectra
pass through losslessly. Constructs outside FTSL's scope (rough transmission,
bump/normal maps, `.ply`/`.serialized` meshes, mesh area-emitters) degrade to a
documented approximation and are flagged with `# WARN:` comments in the output.
Since **Blender can export directly to Mitsuba XML** (via the `mitsuba-blender`
add-on), this doubles as a Blender → FTSL path.

---

## Command-line reference

**Core**

| Flag | Meaning |
|---|---|
| `-in <path>` | Load an FTSL scene file |
| `-scene <name>` | Built-in scene (e.g. `cornell`) |
| `-n <photons>` | Trace exactly this many photons/samples |
| `-r <res>` / `-r <W> <H>` | Output resolution (overrides scene default); one value = square, two = non-square film |
| `-o <path>` | Output image (`.png` / `.jpg` / `.ppm` by extension). Missing parent directories are created before the render starts (reported as `[out] created output directory …`), so a render aimed at a fresh `png/<series>/` subdir can't be traced to completion and then lost at write time. Mode `V` produces a *pair* of images (the two independent estimates it cross-checks) and writes them as `<out>_forward` / `<out>_backward` beside the given path |
| `-topng <in> <out.png> [-ev <c>] [-exposure-anchor <v\|file>]` | Convert an existing `.ppm` or `.ftbuf` to a 24-bit PNG (no rendering); `-ev` re-develops a `.ftbuf` brighter/darker, `-exposure-anchor` develops it at a **shared** gain so a whole sequence of checkpoints can be re-developed flicker-free. See **Output** |
| `-review <base>` | Play a directory of already-rendered frames (`<base><digits>.<ext>`, e.g. `png/swoop/swoop`) on the live window/timeline — scrub/Play, re-time by painting speed, and Save a re-paced copy (no rendering); see the fly-viewer section |
| `-serve` | **Resident preview server.** With `-serve -in <scene.ftsl> [flags…]`, ftrace does *not* exit after one render: it keeps the process — and with it the live window, CUDA context, and spectral/spectral-upsampling tables — resident, and re-renders whenever a new scene path arrives on **stdin** (one path per line), reusing all the other flags (`-mode`/`-n`/`-r`/`-window`/`-o`/…) with only `-in` swapped per frame. Line protocol: prints `[serve] ready` once, then `[serve] done <path>` after each frame; `quit`/`exit`/EOF ends the loop (`[serve] shutdown`). This skips the per-frame cost of process spawn + window/CUDA/table init — the dominant fixed overhead for cheap preview frames — so an external driver (e.g. loom's `PreviewServer`) can stream an animation into a single window that updates in place. Scope: resident-process reuse only; each frame is still a full independent render (no delta/geometry caching yet) and the window keeps the first frame's resolution for the session. |
| `-mode <A..D,M,S,U,P,R,V>` | Render mode (default `B`) |
| `-on-unsupported error\|fallback\|strip` | What to do when the selected mode can't render a scene feature (GRIN media, or a fisheye camera in mode `D`/`U`). `error` (default) prints a diagnostic and aborts; `fallback` renders that camera in mode `R` (backward reference) instead; `strip` removes the offending feature (e.g. drops the GRIN `ior`, turning the medium into a plain one) and renders in the requested mode anyway. Complements `prefer { … } else { … }` in the scene file, which resolves the mode/feature mismatch *before* this policy is consulted |
| `-pmradius <r>` / `-pmradiusfrac <f>` | Mode `M`/`S`/`U` photon-map/merge gather radius (initial radius for `S`/`U`): absolute world units, or a fraction of the scene radius (default `0.02`). Smaller = sharper contact shadows but noisier. In mode `M` this is the *starting* radius the density adaptation refines from — except `-pmradius`, which pins it exactly (implies `-nopmauto`) |
| `-pmcount <k>` | Mode `M` density-adaptive gather radius: target number of photons a typical gather should see, at 1 M stored photons (default `200`; the target grows as the cube root of the stored count). Implies `-pmauto`. Higher = smoother/blurrier and slower, lower = sharper/grainier and faster |
| `-pmauto` / `-nopmauto` | Turn the mode-`M` density-adaptive gather radius on (default) or off. `-nopmauto` reproduces the old fixed-radius output bit-for-bit |
| `-pmfg <K>` | Mode `M` final gather: `K` cosine-weighted hemisphere sub-rays per sample, querying the map one bounce away for sharp contact shadows / fine detail (default `0` = off, direct density query). ~`K`× per-sample cost — pair with fewer `-spp` |
| `-savemap <f>` / `-loadmap <f>` | Mode `M` (GPU) view-independent photon-map cache. `-savemap` writes the built map to `<f>` after the forward deposit; `-loadmap` reloads it and **skips the deposit**, re-gathering any camera / radius for free. A scene-identity guard falls back to a fresh deposit if the file was built for a different scene. Like `-o`, a missing parent directory for `-savemap` is created up front rather than discovered after the deposit |
| `-sppmalpha <a>` | Mode `S` radius-shrink rate (default `0.7`; smaller shrinks faster) |
| `-vcmalpha <a>` | Mode `U` (VCM) radius-shrink rate (default `0.75`; smaller shrinks faster) |
| `-heroc <N>` | Hero-wavelength bundle size on the spectral tracers — **CPU** modes `A`/`B`/`C`, `R`, photon-map `M`/`S`, BDPT `D` and VCM `U`, plus the **GPU megakernel** (forward `A`/`B`/`C`, the `M` deposit, backward `R`, BDPT `D`, and VCM `U`): each path carries `N` wavelengths (a hero + `N-1` stratified secondaries) down one shared BVH walk, cutting colour noise at a given sample count for free. In BDPT both subpaths carry the bundle and each connection is evaluated per-λ under one shared MIS weight — on **both** backends, which agree to 0.03%. VCM (`U`) does the same on **both** backends: one bundle per path index feeds both its light and camera subpath, so its *connections* are exact per-λ while its *merges* key off each stored light vertex's own wavelengths — **0.51× noise RMS** at equal passes on a gel + mirror box (CPU), **0.72–0.82× chroma noise** for 1.5–1.7× the time on the GPU, matching the single-λ estimator to 0.02 % and each other to 0.03 %. In modes `R` and `A`/`B`/`C` (and the `M`/`S` deposit) the bundle also rides through mirrors/gels/glossy lobes and every Russian roulette survives on the strongest live λ (no per-λ ratio amplification), worth ~0.42–0.52× noise RMS on coloured interiors in `R` and ~1.1× luma / 1.3–1.8× chroma at equal time in the forward modes. Default `4`; clamped to `1..8`. **Mode `W` defaults to `8` instead** — at 1 spp the bundle *is* the spectral quadrature, and it is nearly free there (measured 2.7 % of frame time versus a single wavelength, because mode `W` is traversal-bound; the same step costs 61 % in mode `R`). `-heroc 1` turns hero **off** (bit-identical to the classic single-λ estimator) — fine in the sampled modes, but in mode `W` it renders dispersive surfaces flatly **wrong** rather than merely noisy, and a batch mode-`W` render now warns and names the offending material. Ignored (still single-λ) by the GPU **wavefront** backend (`-wavefront`) and by any scene with participating media, a GRIN volume, or a finite-lens camera |
| `-herosplit` | **Split-at-dispersion** instead of the default de-hero policy. Normally a dispersive event (dielectric refraction, grating order, fluorescent Stokes shift) *terminates* the `N-1` secondary wavelengths and boosts the hero ×`N`, because they can no longer follow one shared direction. With `-herosplit` all `N` wavelengths **continue**, each running the same interaction with its own λ — refracting along its own Snell direction, diffracting into its own grating order — so the bundle fans out into `N` independent monochromatic sub-paths. Same mean (both estimators are unbiased; verified `sum/emitted = 1.000000` and converged luminance matching a 200 M-photon reference to 0.03 %), but the chromatic spread of a prism / rainbow / dispersive caustic is resolved **geometrically per photon** instead of stochastically across many. Measured on a dispersive `glass:SF10` flint-sphere Cornell box, **at equal wall clock**: **0.70× chroma / 0.89× luma noise RMS inside the caustic**, 0.80× / 0.92× over the whole frame. The extra traversal past the split is linear, not exponential (a monochromatic sub-path never re-splits) and is paid only by photons that actually reach the glass — **1.11×** per photon there — but that ratio is scene-dependent, which is why it stays opt-in. No-op with `-heroc 1`. **CPU forward modes `A`/`B`/`C` + the `M`/`S` photon deposit** only — ignored by the GPU backends, the backward tracer (`R`), BDPT (`D`) and VCM (`U`). |
| `-beams` / `-photonbeams` | **Decorrelated single-scatter volumetrics** for the shared forward mode-`B` multi-camera / flyby pass. Normally that pass splats one photon realisation to every camera, so a view-dependent single-scatter effect (rainbow / fogbow / glory) has the *same* frozen speckle in every frame. `-beams` switches to a **single-scattering long-beam** estimator: the photon crosses the medium straight (deposited once), and **each camera independently samples its own in-scatter point** toward its own eye — so all cameras share the same mean bow but get **independent per-frame noise** (≈1× photon cost across the flyby, correct per-view angle, non-frozen grain). Deliberately omits the multiple-scatter haze wash (crisper bow). Runs on **CPU and GPU** (ported to the CUDA forward tracer; spectral-rainbow-phase media stay CPU-tabulated and fall back to CPU); needs ≥2 shared cameras + a scattering `medium`. No effect otherwise. |
| `-camera <sel>` | Pick which camera(s) to render (and thus what `-window`/`-preview` shows). `<sel>` is `all`, an exact name (`hero`, `fly137`), a **path base name** (`fly` selects every frame of `camera_curve "fly"` — `fly000..fly143` — while excluding unrelated stills), an index `#N` into the declared cameras (0-based, `#-1` = last), or `near=X,Y,Z` (the camera whose eye is closest to that point). The path-base form renders one whole flyby from a scene that also declares one-off stills; the index / nearest forms aim the live view at one frame of a long `camera_curve` without hunting for its frame name. |
| `-view EX,EY,EZ/LX,LY,LZ[/FOV]` | Render a brand-new ad-hoc camera (eye → look, optional vertical FOV; `,` and `/` are interchangeable separators) instead of the scene's cameras — a quick way to preview a scene from an arbitrary angle. Works with `-in` scenes and built-in `-scene`s. |
| `-t <threads>` | CPU thread count |
| `-device auto\|cpu\|gpu` | Hardware backend |
| `-wavefront` | Streaming GPU backend instead of megakernel |

**Camera / physics overrides**

| Flag | Meaning |
|---|---|
| `-light <preset>` | Override light SPD by preset |
| `-aperture <r>` / `-focus <d>` | Thin-lens aperture radius / focus distance |
| `-mesh <path>` / `-meshscale <s>` | Load & scale an OBJ into the built-in scene |
| `-export-mesh <out.obj>` | Polygonise the scene's isosurfaces into a watertight OBJ mesh (marching tetrahedra, box-capped) and exit, instead of rendering — for Unreal / Blender import (see **Exporting an isosurface to a mesh**) |
| `-mesh-res <N>` | Mesh export fineness: grid cells along the longest bounds axis (default 128) |
| `-mesh-adaptive` / `-mesh-decimate <f>` | Curvature-adaptive QEM decimation of the exported mesh; `<f>` = triangle fraction to keep (default 0.5) |
| `-check-watertight` / `-airtight` | Audit every named `mesh` and every `isosurface` in the scene for a closed, consistently-oriented surface, print a per-object `[OK]`/`[WARN]` report, then exit (no render). Warns per object about **boundary edges** (holes / open border), **non-manifold edges** (3+ faces share an edge), and **flipped** (inconsistently-wound) facets; a dielectric object is flagged with `!` because a leak breaks its refraction / interior-medium tracking. Isosurfaces are polygonised at `-mesh-res` first. Exit code is non-zero if any object is not airtight. |
| `-check-airtight` | Audit every `isosurface` by **ray-parity on the marched field** (not a polygonised proxy): fire chords from outside the container and flag any that cross the boundary an odd number of times (a leak — an open cap on an `open` surface, or a `max_gradient`/thin-feature overshoot the marcher skips), plus a dense-reference **overshoot** check. Prints `[OK]`/`[WARN]` and exits non-zero on any leak. See **Auditing the marched field directly**. |
| `-check-airtight-rays <N>` | Chord count per isosurface for `-check-airtight` (default 4000). |
| `-parseonly` | Load the scene, print a one-line contents summary (materials / records / emitters / spheres / tris / implicits / textures / patterns / cameras), then exit without rendering. A fast syntax + semantic check — every `.ftsl` diagnostic still fires, so it's the cheap way to sweep a whole scene directory for load errors. |
| `-fog <σt>` / `-fogalbedo <a>` / `-fogg <g>` / `-fograyleigh` | Fog controls |
| `-filmthickness <nm>` / `-filmior <n>` | Thin-film iridescence demo params |
| `-diffraction <mode>` / `-nodiffraction` | Enable/disable grating & thin-film diffraction |
| `-spp <n>` | Samples per pixel for modes `R`, `D`, `M`, and `V`; **number of passes** for SPPM (`S`) and VCM (`U`) |
| `-n <photons>` (mode `S`) | Photons traced **per pass** (SPPM rebuilds a bounded map each pass). *(Mode `U` ignores `-n` — its light-path count follows the film resolution.)* |

**Scene-ignore (speed knobs)** — rasterizer-style flags that strip or cap expensive
scene features so a render (especially the backward camera modes `R`/`P`, and the fast
`-rgb` path) runs faster. The three strip flags mutate the scene once at load and print an
`[ignore] stripped: …` summary; `-max-bounce` / `-direct-only` are render parameters.

| Flag | Meaning |
|---|---|
| `-no-media` / `-nomedia` | Drop all participating-media volumes (fog / homogeneous / heterogeneous). Also un-gates the fast `-rgb` backward, which otherwise falls back to spectral on any medium. |
| `-no-env` / `-noenv` | Remove the environment light (constant or image-based): the scene renders against black, and the emitter CDF is rebuilt without it. |
| `-no-fluoro` / `-nofluoro` | Demote every fluorescent material to a plain diffuse (using its elastic reflectance albedo) — skips the wavelength-shifting re-emission. |
| `-max-bounce <N>` | Set path depth to `N` bounces (applies to forward `A`/`B`/`C`, backward `R`, the composite `P`, the photon modes, and the bidirectional `D`/`U`). Default is the tracer's own cap: **32** for the unidirectional tracers, **8** for `D`/`U`, whose connection cost grows ~depth². For `D`/`U` the flag therefore *raises* the depth as often as it caps it — a specular-only cavity (a mirror-lined sphere, a kaleidoscope, deeply nested dielectrics) truncates its recursive images to black at 8 edges and wants `-max-bounce 24`–`48` before the hall of mirrors fills in. Specular vertices are cheap there: a delta BSDF has no connection to make. A scene that always needs the deeper walk can say so itself with `render { max_bounce <n> }` (see **Render-setting block**); this flag overrides that. |
| `-direct-only` / `-directonly` | **Whitted mode:** after a non-specular vertex (diffuse / diffuse-transmit / elastic-fluorescent / fog single-scatter) does its direct-lighting NEE, stop — no diffuse indirect (no colour bleeding, black shadows). Specular chains (mirror / glass / glossy / filter) still recurse. Scoped to the **camera** path tracers (`R` spectral + `-rgb`, and `P`'s backward layer); forward `B` and the photon/BDPT modes honour `-max-bounce` but ignore this. |
| `-whitted-grid <n>` | **Mode `W` only.** Fire an `n`×`n` fixed lattice of shadow rays at every area light instead of one random point (default `4` → 16 rays). This is the single knob that decides how smooth a soft shadow is; a point/spot/collimated light is a deterministic connection already and ignores it. |
| `-ambient <v>` / `-amb <v>` | **Mode `W` only.** Flat ambient fill added at every diffuse vertex (POV-Ray's `ambient`) — the cheap stand-in for the diffuse GI mode `W` drops, without which a **closed** room previews with black shadows. **Dimensionless:** `v` is a fraction of a light's own radiance (internally scaled by `Scene::ambientRef()`), so the same value behaves the same in any scene whatever its absolute radiometric scale. Default `0`; `0.02..0.2` is the useful band. With `-gi` it keeps applying, as the **far-field** term a gather ray picks up when it escapes the geometry. |
| `-gi <n>` / `-radiosity <n>` | **Mode `W` only.** Replace the flat `-ambient` term with a real **deterministic one-bounce hemisphere gather**: `n` rays per diffuse vertex along a fixed world-space lattice, each carrying whatever mode-`W` radiance it finds. Brings back the two things a constant cannot — **contact darkening** in crevices and **colour bleeding** (a gold object actually tints the room). Default `0` (off); `16..64` is the useful band. Unlike POV-Ray's radiosity there is **no irradiance cache**, so nothing depends on render order or on which sample points the geometry happened to trigger — which is what makes it safe for a **seamless animated loop**. Residual error shows as low-frequency banding rather than noise; `-spp` rotates the lattice, so it refines progressively. |
| `-gi-grid <n>` | **Mode `W` only.** `n`×`n` shadow rays at a *gather* vertex (default `1`). Separate from `-whitted-grid` because a gather vertex's soft-shadow detail is averaged over `-gi` directions anyway, so paying the full grid there multiplies the gather's cost for almost no visible return. |
| `-gi-bounce <n>` | **Mode `W` only.** Max bounces along one gather ray (default `4`). Bounds the cost of a specular chain: gold is ~0.9 reflective, so the `adc_bailout` cutoff alone would let a single gather direction ricochet ~60 times inside a gold lattice. |
| `-gi-clamp <x>` | **Mode `W` only.** Firefly ceiling on the radiance **one** gather ray may return, as a multiple of one light's own radiance — same dimensionless units as `-ambient`, so the same number works at any scene scale. `0` (default) is off and bit-for-bit inert. Fixes the thin bright dashed curves a glass ball or mirror casts onto nearby diffuse surfaces at low `-spp`: those are gather rays reaching the lamp *through* the specular surface, carrying its full radiance, and the shared direction lattice turns the on/off boundary into an image-space contour instead of noise (see "Honest limits"). Try `0.05`–`0.2`; keep it above `-ambient`, which the clamp also caps. Clamped per wavelength, not per bundle, so the hero and single-λ paths cannot drift apart; the weight of a clamped direction is left alone, so the gather still normalises by the realised sum of cosines. |
| `-dual-scatter` | Backward modes only. Approximate a **coat's multiple scattering** analytically (Zinke et al. 2008) instead of path-tracing it. Terminates the path at a `hair` vertex, so it is biased by construction — fast look-development for fur, not a reference. Full description under [Dual scattering](#dual-scattering--dual-scatter). |
| `-dual-density <d>` | Zinke's density factor `d_f` = `d_b`, "how enclosed is a strand" (default `0.7`, sensible range 0.6–0.8). Lower reads as a more open, darker coat; raising it toward `0.9` is what brings a dense pale coat up to its own reference. |
| `-dual-db <d>` / `-dual-df <d>` | Override `d_b` (the local backscatter lobe) or `d_f` (the light let through the coat) on its own; either unset follows `-dual-density`. `-dual-df 0` leaves only the directly-lit term, which is how a brightness error gets attributed to one branch. |
| `-dual-max-cross <n>` | Strands one dual-scattering shadow ray counts before it stops (default `64`). |
| `-dual-grid [cells]` | Count dual-scattering crossings by marching a **fiber-density grid** (Zinke §4.1.2) instead of walking the strands one by one — the crossing count comes from `∫σ_t dt` with no curve intersections, and is drawn as a Poisson variate so it stays a drop-in for the walk. 1.5× faster than the walk on a dense coat, which is what turns `-dual-scatter` from a net loss into a win there. Optional argument is a cell **budget** (default `2097152` = 128³), split into roughly cubic cells over the fur's bounds; ~64 MB at the default. Needs `-dual-scatter`. See [The fiber-density grid](#the-fiber-density-grid--dual-grid). |
| `-fur-volume [cells]` | Render `type hair` coats as a **participating medium** instead of as strands — the coat's far LOD tier. Fibers leave the BVH entirely; a ray free-flights against the same grid's `σ_t(d)` (exact inverse-CDF, not delta tracking) and each collision invents one virtual fiber, drawing its tangent from the cell's reconstructed Bingham orientation distribution and shading it with the ordinary BCSDF. Cost stops scaling with fiber count; per-strand silhouette and texture coordinates are lost, so this is for fur that is small on screen. Backward modes only. Shares the field (and the `cells` budget) with `-dual-grid`, plus 16 B/cell. Once the field is built the strands are **deleted before the BVH is built over them**, so the summary replaces the thing it summarises in memory as well as in the render — a coat that used to die with `bad allocation` now loads. Suppressed automatically when something else still needs the geometry (`-fur-lod`, `-dual-scatter`, the raster paths, a forward mode). See [The coat as a medium](#the-coat-as-a-medium--fur-volume). |
| `-fur-lod [d0[:d1]]` | Turn that far tier from a mode into a **LOD decision**: trace strands while one pixel is narrower than `d0` fiber diameters where the coat begins, the aggregate once it is wider than `d1`, and cross-fade stochastically between (one coin per path against a smoothstep, so the switch dissolves into the sampling instead of drawing a line across the image). Implies `-fur-volume`. One number sets `d0` and puts `d1` two octaves up; default `1:4`. The ruler is the **pixel** footprint and does not shrink with `-spp`. See [Choosing a tier](#choosing-a-tier--fur-lod). |
| `-fur-keep-strands` | Opt out of `-fur-volume`'s deletion of the strands: the fibers stay loaded and in the BVH (still invisible to the far tier's rays, which free-flight against the grid either way). For A/B-ing the two tiers in one process, or if something in a scene still needs the curve geometry. Inert without `-fur-volume`. |

**Long-running / output** — `-time` / `-noise` / `-forever` / `-preview` / `-window` /
`-interval` apply to every image-forming mode (forward `A`/`B`/`C`, the spp modes `R`/`D`,
the composite `P`, and the photon modes `M`/`S`/`U`), on both CPU and GPU. `-resume` /
`-checkpoint` cover `A`/`B`/`C` (photon-count checkpoint), `R`/`D` (spp-count checkpoint),
and `P` (dual forward+backward film) — `M`/`S`/`U` keep persistent per-pass state a film
alone can't restore, so they are not disk-resumable.

| Flag | Meaning |
|---|---|
| `-time <s>` | Render until a wall-clock budget |
| `-noise <pct>` | Render until the noise floor drops below `pct` % |
| `-forever` | Refine indefinitely (Ctrl-C stops gracefully) |
| `-preview` | Live ANSI thumbnail while rendering |
| `-window` | Open a real OS window (Win32; no-op off Windows) showing the actual tone-mapped pixels, refreshed every `-window-interval` (default 0.2 s, independent of the `-interval` disk-write cadence — so the image builds up on screen while it renders rather than appearing only when it's finished). The image is **presented by Direct3D 11** (a flip-model swap chain; the control strip below it stays GDI), which is what keeps a fast renderer fast: the previous CPU present — a per-pixel RGB→BGRA repack plus a `HALFTONE` `StretchDIBits` — cost **9 ms/frame** at 1920², more than the render it was displaying, and charged it to the render thread; it is now **~1.3 ms**. In the GPU-rasterized interactive explorer (`-raster -explore -device gpu`) the frame skips host memory **entirely**: CUDA is handed the window's own D3D11 texture and the tone-map kernel writes the finished pixels straight into it, so there is no device→host download, no re-upload, and no host touch of the image at all (measured at 3840²: **26.1 ms → 9.7 ms** per displayed frame). ftrace prints one line saying which way it's presenting; the copy path is used automatically whenever the fast one can't be (notably when D3D picks a different adapter than the CUDA device, as on hybrid iGPU/dGPU laptops, or when a flypath overlay has to be drawn into the pixels). `FTRACE_LIVE_GDI=1` forces the old GDI path (and ftrace falls back to it automatically if D3D can't start). The window is put on screen **before the render starts** — as soon as the scene has loaded and the frame size is known — showing a near-black placeholder with the current stage in the title bar (`preparing…`, `mode W — starting…`), then the first rendered chunk replaces it. Previously it was created lazily by the first repaint, so in the ray-traced modes no window existed until the render was already over and the finished image appeared to flash up for a split second as the process exited. Full-resolution, unlike `-preview`'s terminal thumbnail; runs on its own UI thread. A plain fixed-`-n` forward render is auto-chunked so the view converges live, and closing the window stops the render (final image is still written). The title bar identifies the render as `ftrace — <scene> → <output>`, then the transport mode driving that frame (`mode B (pinhole)`, `mode D (BDPT)`, `mode M (photon map)`, …; a per-camera flight shows the mode of the frame currently on screen), then the live status (`spp` / `% noise` or photon count) as it converges, and finally the **compute backend** actually in use — `GPU (NVIDIA GeForce RTX 4090)` (the real device name, so a multi-GPU box says *which* one) or `CPU (12 threads)` — so you can tell at a glance which scene/file the window is showing, how it's being rendered, how far along it is, and what's doing the work. The backend is reported from where the device is *resolved*, not from what `-device` asked for, so a `-device gpu` that fell back (no CUDA build, no free VRAM, an unsupported feature) reads `CPU` and says so; in the interactive explorer, where the raster preview and a `mode W` refinement run on the CPU while a path-trace refinement runs on the GPU, the label follows whichever pass produced the frame on screen. The window opens at (and won't be dragged smaller than) a readable minimum so that `<scene> → <output>` title stays legible even for a small image; the picture is aspect-fit and letterboxed inside whatever size the window is. |
| `-keepwindow` / `-hold` | Like `-window`, but **don't auto-close** the live window when the render finishes — normally the window is torn down at process exit the instant the last frame completes, so a finished image only flashes on screen. With this set, ftrace keeps the final image up and blocks until you close the window yourself (handy for inspecting a quick `-raster` preview or a completed still). Implies `-window`. |
| `-interval <s>` | Periodic image write / status line / ANSI `-preview` refresh (default 15 s). This is the **crash-safety** cadence — how often the PNG and the `.ftbuf` checkpoint are rewritten — and is deliberately *not* what drives the live window (see `-window-interval`). |
| `-window-interval <s>` | How often the `-window` live view repaints (default 0.2 s), independent of `-interval`. The two used to share one timer, which meant any render finishing inside one interval never showed a single live frame — a 5 s `-mode W` frame under `-interval 8` painted once, as the process was exiting, so the finished image just flashed and vanished. They are separate now because they want opposite cadences: rewriting a PNG and a multi-megabyte checkpoint five times a second is pointless disk churn, while repainting a window every 15 s defeats the point of having one. Repaint granularity is bounded below by the renderer's own chunk size (one chunk ≈ 0.15 s of GPU work, minimum 1 spp), so on a 480² `-mode W -spp 8` frame you get one repaint per spp — the first complete image lands after ~0.6 s instead of after 5 s. Measured cost of the extra repaints there: **+3.9 %** of render time (a repaint tone-maps and presents the whole frame, ~25 ms at 480²). The floor is adaptive — never less than the larger of this value and 12× what the last repaint actually cost — so a 4K film backs itself off instead of spending all its time painting. `0` means "every chunk, subject only to that budget". `FTRACE_WINDOW_DEBUG=1` logs each repaint and its cost. |
| `-raster` | Fast solid-shaded **preview** (no light transport): z-buffer the whole scene as flat-shaded triangles, one image per selected camera. Honours `-camera` and `-window` (a `camera_curve` flyby animates in the window; a single still becomes an **interactive fly camera** — Space/`+` fly forward, Shift/`-` back, move the mouse off-centre to steer (rate/joystick look, cursor stays visible), wheel = dolly, Ctrl+wheel = step size, `C` = wall collision, `0` resets, `P` prints a paste-ready camera, plus **Clip/Reset buttons** in a panel below the image). See the preview note under **Render modes**, and `-explore` below to drop straight into this viewer at a flyby's first frame. |
| `-raster-iso <n>` | Isosurface mesh fineness for `-raster` (cells along the longest bounds axis; default 96, `0` skips implicits) |
| `-raster-curve-budget <n>` | Cap on the preview triangles spent tessellating **curve / fur strands** (default `12000000`, ~3.8 GB of preview geometry). Past it the round-cone tubes coarsen (10-sided capped → 6/4/3-sided → capless → flat ribbon), and only if the cheapest tube still busts the budget are whole **strands** thinned out. A groomed pelt is millions of segments, so without this a `-raster`/`-explore` on one would allocate tens of GB and appear to hang. Preview-only — the ray-traced modes intersect the analytic strands and ignore it. |
| `-raster-bench <n>` | Raster **frame-rate benchmark**: after the scene is built (and uploaded, on the GPU), re-render the first selected camera `n` times and report steady-state **ms/frame** (min/median/mean + fps) — the interactive explorer's per-move cost, measured independently of startup. With `-device gpu` also prints a per-pass breakdown (clearvis/project/raster/shade/clear/expose+encode/download, timed with CUDA events on the GPU timeline). Add `-window` and it also reports the **live-window present tail** — what handing each finished frame to the preview costs the render thread — because that tail used to be larger than the render itself and a backend speedup is only real if it stays small. With `-device gpu -window` it then runs a **second, zero-copy phase**: the same `n` frames rendered directly into the window's D3D11 texture, reported as one combined `render+present` figure (there is no separate tail to report — there is no handoff) plus its own per-pass breakdown, so the two presentation paths can be compared pass by pass on one run. Note that the zero-copy *median* pins at the display refresh (16.67 ms / 60.0 fps) because presenting blocks on vblank once both back buffers are queued — read **min** for the true pipeline cost. Writes the last frame to `-o` so backends/builds can be byte-compared. |
| `-see-through` / `-seethrough` / `-glass` | In `-raster`, render **clear** materials (dielectric / thin-film / filter / diffuse-transmit) as actually see-through instead of solid ghosts: each clear surface between the camera and the opaque background **dims** and **milkily hazes** what's behind it, cumulative with the number of clear surfaces crossed (no refraction, no coloured absorption). Order-independent, so overlapping glass needs no sort. See the preview note under **Render modes**. |
| `-glass-clarity <0..1>` | Per-surface transmittance for `-see-through` (default `0.85`; higher = clearer / less dimming). Passing it implies `-see-through`. |
| `-explore` / `-fly` | **Interactive fly-through** of a multi-frame flyby without rendering it. Seeds the interactive raster viewer at the **first frame** of the selected `-camera` path (e.g. `-camera fly`) and hands control to you: Space/`+` fly forward, Shift/`-` back, move the mouse off-centre to steer (rate/joystick look, cursor stays visible), wheel = dolly, Ctrl+wheel = step size, `C` = wall collision, `T` = cycle the lit preview (see below), `0` resets the view, `P` prints a paste-ready camera block, close the window to finish. The flyby's frames are kept as a **camera-path timeline** in the panel below the image: **scrub/play/pause** across them, **lock** the camera onto the path (travel forward/back along it at a **cams/update** or **cams/second** speed), or release to fly freely — see **Interactive camera** for the full panel. Implies `-raster -window -keepwindow -no-meter`. Use it to preview/author a flyby camera without watching or writing every frame. **`T` — cycle the lit preview:** the still view cycles **raster → mode `W` → path-traced → raster**. The flat raster (default) is instant and is what you navigate with; the other two render the pose you are actually standing at. Whichever is active, the instant you move the camera it drops back to the responsive raster and re-renders once you settle, so navigation stays fluid. The scene-ignore flags (`-no-media`/`-no-env`/`-no-fluoro`, `-max-bounce`, `-direct-only`) apply to both, so you can strip/cap the scene for a faster preview.<br><br>**mode `W`** is the deterministic Whitted preview (see **Render modes**) on the **CPU**, so unlike the path-traced stage it works on **any scene** and needs no GPU — full spectral walk, all materials, media, environment, and the `-gi` one-bounce gather if you asked for one. It is **noise-free**, so it does not need to converge: the pose renders **once**. Because a mode-`W` frame costs anywhere from ~0.4 s (a Cornell box) to ~26 s (a gyroid labyrinth at 960×600), it is delivered progressively — a **coarse full-frame pass lands immediately**, then full-resolution **row bands** sweep down over it, with the band height continuously retuned from the measured cost of the previous band to keep the viewer responsive on fast and slow scenes alike. The title bar shows the percentage complete. Moving the camera simply abandons the unfinished rows. If the scene contains a material that **de-heroes** the path onto one wavelength — a `layered` coat, participating media, a GRIN volume, or `-heroc 1` — the preview keeps adding passes up to 16 spp to resolve its colour; on any other scene 1 spp is already exact and it stops there. Dispersive materials (glass, thin film, multilayer, grating, half-mirror, fluorescence) used to be on that list and no longer are: mode `W` splits the bundle at a dispersive vertex, so they are colour-correct in the very first pass — see the glass note under **Render modes**. **`-explore -mode W` opens straight into this preview** instead of the raster.<br><br>**path-traced** progressively traces the pose with the fast **RGB backward** tracer (the Stage-2 `-rgb` walk) into a resident GPU session: while the camera holds still the image **converges in place** (the title shows accumulated `spp`), so it ends up more correct than mode `W` — real multi-bounce GI — but it starts noisy, needs a CUDA GPU, and only works when the scene+camera are inside the fast-RGB scope (same scope as `-rgb`). When it isn't available the cycle **skips it**, so `T` becomes a plain raster ↔ mode `W` toggle. |
| `-no-meter` / `-nometer` | Skip the **exposure-lock metering pre-pass**. Normally a locked `camera_curve`/`camera_path`/`camera_orbit` group meters (up to 64 of) its frames up front to compute one shared exposure anchor, so the flyby doesn't flicker. With this flag that pre-pass is skipped and each frame **auto-exposes on its own** — faster startup (no metering the whole path), at the cost of possible frame-to-frame brightness flicker on an animated flyby. Implied by `-explore` (the interactive viewer auto-exposes per frame, so metering a whole flyby just to fly one frame is wasted work). |
| `-noclip` / `-nocollide` | Start the interactive fly-viewer with **wall collision off** (fly through geometry) — for placing a camera *outside* the room or *inside* glass. Collision is **on by default** (you can't fly through walls); press `C` in the viewer to cycle `slide` → `stop` → `noclip` live. See the fly-camera controls under **Interactive fly camera**. |
| `-anim <file.json>` | Edit a **loom `CurveDrive` sidecar** in the interactive fly editor (implies `-explore`). The editor's control points become the drive's N-dimensional points: channels 0–2 are the point you see and move in 3-D, channels 3+ are non-spatial values carried along per point. **Save** writes the reshaped curve back to the sidecar atomically, preserving the drive's name/mode/dims and every channel → scene-variable **binding**. A sidecar that doesn't exist yet is created on the first Save (from whatever control points the scene seeded), so this is also how you start a drive. See **Editing a loom animation drive** under **Interactive fly camera**. |
| `-loom <scene.py>` | Keep a **live loom process** alongside the window so the scene can be *re-derived*, not merely re-viewed. With `-anim` it turns the fly editor into a real animation editor: scrubbing asks loom for the scene as of that point on the drive, so the **bound scene variables** move in the viewport (loom does the curve sampling, so the preview can't drift from the final render), and the panel grows a **loom bind row** for editing channel → variable bindings and the channel count live. With `-viewer` it drives the loom sidecar viewer's **Live (loom)** panel instead (one control per `build()` parameter, plus the sweep axis). See **Editing a loom animation drive** and **Live re-derivation**. |
| `-viewer <sidecar.json>` | Open the **loom native viewer** on a scene-introspection sidecar (written by `loom.viewer.ViewerModel.save_sidecar`) instead of rendering — a Dear ImGui / Direct3D 11 window with the object list, dataset table, N-D curve pane, strip charts, modulator-DAG graph, Fields / Meshes tabs and (when the sidecar carries a `source` key) a Render tab that raymarches the real field in-process. Add `-loom <scene.py>` to make it **live**. See **Native viewer** in the README. |
| `-play` | With `-viewer`: open with the clock **already playing**, so a loop can be watched — or its per-frame cost read off the `[play]` breakdown printed to stdout — without clicking into the window. Ignored (with a printed reason) when there is no live loom channel or the sidecar advertises `frames = 1`. |
| `-prebake` | With `-viewer`: walk the clock **once** on open and keep every frame's adopted state in memory, then play out of that cache on a wall clock at the panel's `fps` rather than at loom's bake rate. Costs ~0.01 ms a frame to show, so the requested rate is actually delivered, and it makes scrubbing the frame slider instant too. The cache is dropped whenever a build parameter or `frames` changes — a cache built at other values is not a cache of what you are looking at. Same thing as the panel's **prebake** button; the flag exists so a played frame rate can be measured from a script. |
| `-prebake-cap <MB>` | Memory budget for `-prebake` (default `1024`, also settable live as the panel's **cap MB**). A walk that reaches the cap stops there and says where: the **prefix** it did fill still plays from memory and the remaining frames fall back to bake-paced play, so a long clock degrades instead of failing. |
| `-resume` / `-checkpoint` | Resume from / always write a `<out>.ftbuf` checkpoint (modes `A`/`B`/`C`, `R`/`D`, and `P`) |
| `-stop [<pid>\|all]` | **Stop a running render cleanly, from another shell.** `ftrace -stop <pid>` asks that render to do exactly what Ctrl-C does — finish the current chunk, write the final image **and** `.ftbuf` checkpoint, release the CUDA context through the graceful-shutdown path — then waits (up to 120 s) for it to actually exit, so it's safe to script a rebuild right after. `-stop all` targets every running render; a bare `-stop` just **lists** them (pid + scene → output). This exists because a render launched detached has no console to Ctrl-C into, and **force-killing ftrace mid-CUDA is a known way to wedge the NVIDIA driver into a TDR/bugcheck** — so never `taskkill /F` a render, use this. It also releases a window being held open by `-keepwindow`. Implemented as a sentinel file under `<temp>/ftrace/` (a `<pid>.run` entry per live render, a `<pid>.stop` to signal it), which — unlike a named kernel event — crosses the session / window-station boundary between a detached render and the shell signalling it. Stale channel files are reaped by the next `-stop`: both a `<pid>.run` left by a hard kill and a `<pid>.stop` nobody was left to consume, in each case only once the owning pid is gone. A stop that arrives while the process is still **loading the scene** aborts the load rather than being waited out: it prints `[stop] scene load stopped before rendering — nothing was rendered or written.` and exits **1** (no scene was built, so nothing could be rendered — the non-zero exit is the correct outcome, not an error in your `.ftsl`). Since **0.182.0** it covers every long-lived ftrace process, not just renders: a `-viewer` (loom native viewer) or `-explore` GUI is listed by a bare `-stop` and shuts down cleanly when targeted, its event loop polling the same flag a render polls and then leaving through its normal teardown (D3D11, the loom python child, the window). Also since 0.182.0 the **exit code is honest**: `0` only when every target is genuinely gone (a pid that was already dead counts, and reports `nothing to stop`), and `2` with `[stop] FAILED — still running after 120s: <pids>` when one outlives the wait — previously it printed `stopped cleanly` and exited 0 regardless, so a stop that did nothing looked like a success. Nothing is ever force-killed either way. |
| `-exposure-lock` | Share one auto-exposure anchor across all rendered cameras (no `camera_path` flicker); a per-path `exposure_lock [selector]` keyword instead locks just that path, metered from a chosen viewpoint (default the path `average`; also `first`/`index i`/`near x y z`/`camera "name"`). **Process-local** — it can only share an anchor between frames rendered by *this* invocation; for a frame-per-invocation sequence use `-exposure-anchor` |
| `-exposure-anchor <v\|file>` | **Share one auto-exposure anchor across separate `ftrace` invocations** — the missing piece for a sequence whose frames are each rendered by their own process (loom's `render_range`, a batch script, a re-render of one frame). Implies `-exposure-lock`. With a **number** the anchor is used directly (no metering). With a **path**: if the file exists and holds a number that anchor is loaded and reused; otherwise this run meters normally and **writes** its resolved anchor there, so every later frame pointed at the same file develops at the identical gain. Also accepted by **`-topng`**, which is how a *finished* sequence is repaired from its `.ftbuf` checkpoints with no re-render (see **Output**). Without it, per-frame metering can jump — the p99 anchor solves `area(L) = 1%` for a level, and on a scene with a bright compact highlight population the tail density is so thin (~0.25% of frame per octave) that the inversion is ill-conditioned, so a rotating highlight swings the anchor by a third of an octave (measured on `pastel_jack_ring`: 36% single-frame anchor step while every honest brightness measure moved ≤ 2.3%). This is not fixable in the statistic — see the `exposure_lock` notes and `known-issues.md` |
| `-hdr` | Also write a **32-bit float PFM** beside `-o` (`<out>.pfm`) holding the **scene-linear** image — the exact buffer the tone map consumes, with no exposure, no gamma and **no clamp**. Written on every periodic in-progress write too, so a still-converging render can be metered. Use it whenever you intend to *measure* rather than look: an 8-bit PNG clamps at white, and a caustic is by definition the brightest thing in frame, so its core prints as `#FFFFFF` with all three channels **equal** — the tone map destroys the caustic's colour and its peak-to-screen ratio before any analysis can see them. (Values are radiance in the film's own scale; peak/median ratios and chromaticity are exposure-invariant, so two renders shot at different stops stay comparable.) PFM is a 3-line ASCII header + raw little-endian `float32` RGB triples, raster order left-to-right **bottom-to-top**. |
| `-exposure <c>` / `-ev <c>` | Override the exposure **compensation** for every rendered camera (a relative stop multiplied on top of the p99 auto-exposure; `1.0` = neutral), replacing the per-camera film `exposure`. Applies to both the real render and the `-raster` preview — handy when a scene's authored `exposure` (tuned for the physical integrator's bright highlights/caustics) blows out the flat-shaded raster. |
| `-stereo <mode>` | **3-D stereoscopic output** (stills *and* movies). Renders each camera **twice** — a Left/Right eye pair — and composites them into the `-o` image. `mode` picks the fusion: `sbs` (side-by-side **wall-eyed**, L\|R), `cross` (side-by-side **cross-eyed**, R\|L), `anaglyph` (**red-cyan** Dubois glasses, the default kind), or `anaglyph-gm` (**green-magenta** Dubois). Uses the correct **off-axis** rig — two *parallel* cameras offset along the camera right axis with **asymmetric (sheared) frusta** sharing a convergence plane, so there's **no vertical parallax** (toe-in's eye-strain cause). Both eyes share one auto-exposure anchor, so L/R — and every frame of an exposure-locked `camera_path` — tone-map identically. Rectilinear cameras only (a fisheye camera renders mono, with a warning). See **Stereoscopic 3-D** below. |
| `-eye-sep <m>` | Interocular (eye-to-eye) distance for `-stereo`, in metres. Default `0.063` (63 mm, average human). |
| `-view-dist <m>` | Viewing distance (eye-to-screen) for `-stereo`, in metres. Default `0.6`. Used to derive the screen width (screen shows the camera's horizontal field at this distance) when `-dpi` isn't given. |
| `-dpi <n\|auto>` | Screen pixel density for `-stereo`. With a number, screen width `= resX·0.0254/dpi`. `auto` reads the Windows **logical** system DPI (a rough hint — often 96; not the panel's physical pitch). Omit it (the default) to instead derive screen width from `-view-dist` × the camera FOV. |
| `-convergence <m>` | Convergence-plane distance for `-stereo`, in **scene units** — the depth that lands at the screen (zero parallax); nearer objects pop out, farther recede. Default: the camera's **look-at target** distance. |
| `-stereo-keep-eyes` | Keep the intermediate per-eye PNGs (`<out>_<cam>__eyeL/​R.png`) that `-stereo` writes before compositing. By default they're deleted once the composite is done. |

**Diagnostics / self-tests:** `-checkbvh`, `-bvhstats`, `-checkimplicit`,
`-checkcurve`, `-checkfur`, `-checkfurgrid`, `-checkfurvol`, `-checkcontainer`, `-checklens`, `-checkfluoro`, `-checkfog`,
`-checkthinfilm`,
`-checkmultilayer`, `-thinfilmswatch`, `-checkgrating`, `-checkupsample`,
`-checkgrid`, `-checkscatter`, `-checkvnoise`, `-checkworley`, `-checkgabor`,
`-checkbluenoise`, `-checkfnoise`, `-checkstochtile`, `-checkreaction`, `-checkcurv`,
`-checkcavity`, `-checksdf`, `-checksun`,
`-checkbind`, `-checkprop`, `-checkhair`,
`-checkarray`, `-checklattice`. Each runs deterministically without a scene and prints
`PASS`/`FAIL`. `-checkcurve` guards the `curve` primitive: it cross-checks the
round-cone intersector against the exact analytic SDF, the degenerate
one-sphere-swallows-the-other case against the analytic ray–sphere test, `anyHit`
against the full path (with half the origins *inside* the fiber), watertightness at
chain joints, the four bases' flattening, and the **fp32 conditioning** of the quadric at
fiber scale (the same code instantiated at `float`, as the CUDA megakernel runs it) —
see **Curves and fibers** above. `-checkfur` guards the `fur` generator on top of that,
in seven sections: roots on the surface, area-uniform root distribution, determinism
across seeds, growth never pointing into the skin, clumping that collapses tips without
moving roots, a well-formed segment chain, and a regression on the loader-ordering trap
that once made a whole groom generate zero strands silently — see **Grooms** above.
`-checkfurgrid` guards the **fiber-density grid** behind `-dual-grid`, in five sections, and
is unusual in that it validates a mean-field model against the *shipping* curve intersector
rather than against another closed form: mass conservation (`Σ rℓ` deposited equals `Σ rℓ`
built, exactly), the orientation tensor's rank-1 and isotropic limits, `σ_t` against a brute
force sum over every segment, the **expected crossing count** `∫σ_t dt` against the number of
strands 200k real rays actually hit (which is where the +3.98% Jensen bias is measured — it
comes out at +0.81% ± 0.65% where the fibers are locally parallel), and the DDA march against
a 400k-step Riemann sum through the same field. Two of its sections exist because earlier
drafts got a *wrong* answer that looked right: capsule end caps are 5% of an isolated
segment's cross-section (and correctly zero for a chained strand), and a coarse grid straddling
a density taper dilutes σ_t along exactly the rays being measured, by almost exactly enough
to cancel the Jensen bias.
`-checkfurvol` guards the **aggregate scattering model** that turns those grid cells into a
participating medium, in ten sections: the symmetric eigensolver; the startup table that
inverts the **Bingham** distribution's second moment (worst error 1.2e-3 over the whole space
of eigenvalue triples); the reconstructed orientation distribution's own second moment against
the cell's `T`, plus its *first* moment where the sign rule is exact; the round trip from an
offset `h` to a virtual fiber normal and back; the Jensen factor against its 1.0398 bound; and
— the section that decides whether the far tier is worth having — the aggregate's directional
response against **explicit fiber populations**, as L1 error over the whole outgoing sphere.
That last one is what rejected two cheaper orientation distributions: a Watson mixture turns a
girdle into two orthogonal lobes (0.43) and an ACG smears a combed clump (0.26), where the
shipping Bingham measures 0.006 / 0.080 / 0.040 / 0.023 on parallel / combed / isotropic /
girdle. Its tolerances sit just above those numbers on purpose, so a regression in the ODF
family is caught rather than tolerated. The last two sections cover the medium itself: the
16-byte-per-cell ODF cache the far tier reads instead of re-running an eigendecomposition per
collision, and the **free flight**, which is sampled by exact inverse-CDF inside the DDA
rather than by delta tracking (`σ_t` is piecewise constant along a fixed ray, so no majorant
is needed — and a coat, a thin skin of dense cells in a mostly empty box, is the case delta
tracking handles worst). That one is falsifiable and so worth having: the survival probability
over a segment must be exactly `exp(−τ)` for the same `τ` `-checkfurgrid` §4 already tied to
the number of strands real rays hit. The last two sections guard the machinery *around* the
medium rather than the medium itself: §9 the near/far transition (`-fur-lod`), and §10 the
**hair-free BVH** — the second acceleration structure a skip-hair query traverses. §10 is a
pure-optimisation check, and the two ways such a thing can go wrong are the two it tests: the
filtered tree numbers its primitives differently, so a bad remap would decode a leaf as the
wrong primitive and silently lose a wall; and it must agree with `isHairCurve` about what a
fiber *is*, since grass and wire are curves too and have to keep blocking. Both queries are
run over 20 000 rays on a scene mixing all four populations, once on the filtered tree and once
with the remap swapped out so the old leaf-rejection path runs, and any disagreement fails.
`-checkhair` guards the **fiber BCSDF** (Marschner's R / TT / TRT lobes in Chiang's
energy-conserving form, plus Yan's medulla and Zinke's dual scattering) in eleven sections. A hair BCSDF that is subtly wrong still looks
like hair, so every claim is a number rather than a picture — and wherever the physics
allows it, an *exact* number rather than a Monte-Carlo estimate. Because the lobes
separate into a longitudinal `M_p`, an azimuthal `N_p` and an attenuation `A_p`, and each
of the three is normalised on its own domain, the white-furnace test reduces to the
algebraic identity `Σ_p A_p = 1` — which telescopes exactly, and is asserted to `1e-12`
instead of the ~3 % a sphere-uniform estimator could manage against a lobe this narrow.
The rest: the two `M_p` branches agree beyond their analytic `exp(-2/v)` gap; `M_p` and
`N_p` integrate to one under deterministic quadrature; the trimmed-logistic sampler
inverts its own CDF; `sample()` returns exactly the `f` and pdf that `f()` and `pdf()`
report, and its empirical density matches; absorption darkens monotonically while the R
lobe — which never enters the fiber — survives an opaque one; the three lobes peak within
0.02° of the azimuths Snell predicts at impact parameter `h = 0.6`, and a 3° cuticle tilt
moves the R highlight by 2.97°; and `h` is recovered from the hit geometry to 5e-14.
The medulla adds two of its own: the six-lobe furnace still closes to `2e-16` over 405
(κ, σ_s, g, h, θ) combinations, and the one piece of the model that is *not* Yan's — the
analytic stand-in for their unpublished `C^M` / `C^N` tables — is pinned against a
brute-force Henyey-Greenstein random walk through the core, which it tracks to 0.003.
[Dual scattering](#dual-scattering--dual-scatter) adds a section of pure algebra: the
forward/backward split reproduces the furnace total exactly (`ā_f + ā_b = 1` to four
decimals at every inclination, medullated or not) and falls with absorption; Zinke's
triple sum over `(i, j, k)` collapses to a single sum weighted by `n(n+1)/2`, which
reproduces his own closed form to `6e-16`; and `Δ̄_b`'s two coefficients, summed in closed
form, show that eq. 16 as printed is *first-order* wrong in `u = a_b²/(1−a_f²)²` while the
same expression with one sign flipped is second-order accurate — a sign typo in the paper,
demonstrated rather than asserted.
`-checkcontainer` guards the isosurface container clip: rotating an
isosurface must not change what a ray sees, so it builds the same solid twice
(axis-aligned and rigidly rotated) and checks that correspondingly rotated rays
return identical hit distances. `-checklattice` guards **mode W**'s deterministic
sample lattices: first the structural contracts (the digit scramble is a bijection,
`radicalInverseScr(b, 0) == 0` exactly so sample 0 stays the mirror direction /
specular order / median λ, the first *b* points are a permutation of the *b*-point
grid, and 16 samples span at least half the unit interval in every base), then a
**bit-exact** CPU-vs-GPU sweep over every lattice helper. Mode W has no Monte-Carlo
noise for a device mismatch to hide behind, and these helpers are pure
integer-and-`double` arithmetic on both sides, so unlike a rendered image they must
agree to the last bit — a CUDA build compares them directly, a CPU-only build reports
that half as `SKIPPED`.

**Scene front end:** the shared grammar parses every `.ftsl`, with no flag to
configure. `-legacy-parser` and `-validate-grammar` were retired in 0.79.0; they are
still accepted but print a notice and do nothing. See **The shared grammar** above.

---

## Output

Images are written as **PNG** (24-bit RGB, 8 bits/channel — no alpha), **JPEG**
(q95), or binary **PPM (P6)**, chosen by the output file extension, tone-mapped from
the internal linear spectral film to 8-bit sRGB. Long renders can checkpoint to
`<out>.ftbuf` and resume deterministically.

**Converting existing artifacts to PNG** — `ftrace -topng <in> <out.png>` re-encodes
an artifact to a 24-bit PNG *without re-rendering*:

- a **`.ppm`** (binary P6, 8-bit) is copied to PNG losslessly;
- a **`.ftbuf`** resume-checkpoint has its raw linear film tone-mapped (with the
  default p99 auto-exposure — the sidecar doesn't store the exposure mode, so an
  absolute/lumens scene may read brighter or darker than its original `-o` image;
  re-render for an exposure-exact PNG).

A trailing **`-ev <c>`** scales that auto-exposure, so you can **re-develop a finished
render brighter or darker without paying to render it again** — useful when a scene's
p99 anchor is dominated by a small very bright source (an arc, a filament) and leaves
the rest of the frame too dark:

```
ftrace -topng out.png.ftbuf out_bright.png -ev 3
```

`-ev` applies only to a `.ftbuf` (which still holds linear film); on a `.ppm` input it
warns and is ignored, since that file is already 8-bit sRGB. *(Before 0.102.1 `-ev` was
silently dropped on this path — `-topng` runs before the main argument loop.)*

A trailing **`-exposure-anchor <value|file>`** instead replaces the per-image p99
auto-exposure with a **shared** anchor, which is how a finished *sequence* is repaired
without re-rendering a single photon: develop every frame's checkpoint at one gain and
the frame-to-frame brightness flicker of independent metering disappears.

```
ftrace -topng frame000.png.ftbuf frame000.png -exposure-anchor anchor.txt   # meters, saves
ftrace -topng frame001.png.ftbuf frame001.png -exposure-anchor anchor.txt   # loads, reuses
```

Like the render-side flag, a bare number is used as-is and a path is
load-if-present / meter-and-save otherwise. `-exposure-anchor` also applies only to a
`.ftbuf`. See **`exposure_lock`** under **Cameras** for why per-frame metering jumps and
for loom's `stabilize_exposure()`, which drives this automatically over a whole
directory of checkpoints.

### Denoising (`-denoise`)

`-denoise` runs an edge-aware à-trous (SVGF-style) filter on the linear image just before
tone-mapping. Because it sits in the one function that feeds both the file and the live
window, the preview shows exactly what you'll get.

**It filters chroma only, and leaves luma bit-identical.** That is the point rather than a
timid default. A spectral path carries a *single* wavelength, so wherever the
hero-wavelength bundle can't be used — participating media, or any dispersive refraction —
every sample deposits a fully saturated colour, and the pixel only turns white once enough
different wavelengths have averaged in. The variance is therefore concentrated in *chroma*,
while luma is already converging at the usual 1/spp. Since the eye resolves chroma detail
at roughly a third the acuity of luma (the reason every video codec subsamples it), chroma
can be blurred hard at no visible cost. Turning it on cannot lose you an edge, a wire, a
caustic rim or a speck of geometric detail, so there is no tradeoff to weigh.

Measured on `gallery_rain` at 120 spp against an 8000 spp reference of the same frame:

| | chroma RMSE | PSNR | cost |
|---|---|---|---|
| no denoise | 23.8 | 20.4 dB | — |
| `-denoise` | **13.8** (58 %) | **22.2 dB** (+1.8) | ~1 % of render time |

Use it on anything with dispersive caustics or fog. `ftrace -checkdenoise` runs the
filter's self-test (exact energy conservation, constant-image fixed point, bit-identical
luma).

| flag | meaning |
|---|---|
| `-denoise [amount]` | enable; `amount` scales the chroma tolerance (default 1) |
| `-denoise-chroma <x>` | chroma edge-stop tolerance in local sigma (default 2 — a flat knob, barely sensitive across a factor of 8) |
| `-denoise-levels <n>` | à-trous levels, support 2ⁿ wide (default 3). Swept against the reference, the optimum is a plateau at 2–3, *not* SVGF's 5: with no luma term holding the edges a very wide chroma support bleeds gold into white and caustic into floor, and by 7 levels it is a net loss |
| `-denoise-luma <x>` | **also** filter luma (default 0 = off). Measured, this makes the image *worse* — −2.4 dB at `0.45`, because the filter cannot tell a wire or a caustic rim from a noise spike. Available for stills you want *smoothed*, not *truer* |
| `-fireflies <k>` | clamp isolated outliers to `k`× the 2nd-brightest neighbour, hue preserved. Deliberately not energy-preserving. Implies `-denoise`; try 2–4 |

A `.ftsl` is a *scene*, not an image — render it with `-in scene.ftsl -o out.png`.
Three drag-and-drop Windows helpers in the repo root wrap this: **`ppm_to_png.bat`**,
**`ftbuf_to_png.bat`** (both call `-topng`), and **`ftsl_to_png.bat`** (renders the
scene). Drop a file on one, or run `ppm_to_png.bat input.ppm [output.png]`.
