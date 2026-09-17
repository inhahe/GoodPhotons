# ftrace — design & architecture

Physically-based **spectral** renderer (C++17, single exe `ftrace.exe`), Windows /
MSVC / CMake, with a CUDA backend (RTX-class, tested sm_89). Photons are traced
**forward from the lights** in the flagship modes (hence "forward raytracer"), but
backward path tracing, BDPT, photon mapping, SPPM, VCM and a z-buffer preview
rasterizer are all built in. `README.md` is the user-facing landing page (what it is,
how to build it, first renders) and `REFERENCE.md` the exhaustive user-facing manual
(modes, cameras, materials, spectra, lights, geometry, media, CLI) — they were one
3300-line file until the README was split; keep an observable change in whichever of
the two describes it. `FTSL.md` is the authoritative scene-language grammar.
This file records the *internal* architecture. `known-issues.md` tracks bugs/debt.

## Render modes (dispatch in `main.cpp`)

| Mode | What | Core |
|---|---|---|
| `A` | forward + finite-lens physical camera (photons hit the lens) | `render.h` |
| `B` | forward light tracing, splat through pinhole/lens to film (flagship) | `render.h` |
| `C` | forward + contact sensor | `render.h` |
| `R` | backward (unidirectional) path tracer — the reference | `backward.h` |
| `W` | deterministic Whitted/POV-Ray preview: mode `R`'s walk with every estimator replaced by a fixed quadrature (noise-free at 1 spp, biased; CPU + GPU since 0.110.0, fully on-device since 0.116.0) | `backward.h` (`whitted`), `render_cuda.cu` (`WhittedOpts`) |
| `P` | composite: forward B + backward R passes merged | `main.cpp` orchestration |
| `D` | bidirectional path tracer (BDPT, MIS) | `bdpt.h` |
| `M` | photon map (deposit pass + per-pixel density gather; optional `-pmfg` final gather; optional `-beams` view-independent volume cache; two-map caustic split with an aimed second emission pass; `-savemap`/`-loadmap` persist both halves). Since 0.252.0 a **single-camera** render re-draws all three maps every epoch under a fresh salt and averages the epochs' films (`-beamfreeze` to opt out), so the light-side half of its error converges with render time instead of freezing — that was M-FROZEN, whose visible symptom was coloured bars through `phase rainbow` media. 0.253.0 extended that to the **shared/GPU** route, which a plain fixed-`-spp` single-camera render takes. **Multi-camera renders still build once**, which is the cross-frame amortisation this mode exists for | `photonmap.h`, `photonmap_render.h`, `photonbeams.h`, `causticaim.h`, `photonmap_io.h` |
| `S` | SPPM (progressive photon mapping, shrinking radius) | `sppm_render.h` |
| `U` | VCM (vertex connection & merging) | `vcm.h` |
| `J` | UPBP (unifying points, beams and paths): mode `D`'s BDPT connections **and** mode `M`'s `-beams` beam×ray merges under one MIS weight — the volumetric counterpart of `U`. Beams default **on** (`-nobeams` reduces it to `D` bit-for-bit). It traces its own light subpaths for the beam map (0.216.0) and both techniques are MIS-weighted (0.218.0), with merged paths capped at `maxDepth` like the connections (0.219.0). The map sizes itself from the scene's `-beamk` knee via a discarded pilot (0.242.0), so **leave `-n` off** — passing it disables the budget. The **camera pass runs on the GPU** as of 0.244.0 (mode `D`'s megakernel with `MERGE=true`); the light/beam pass stays on the CPU on both backends. Since 0.247.0 the light side is **re-drawn every epoch and averaged** (`-beamfreeze` to opt out), so the merge half converges with render time instead of freezing on one realization — that was UPBP-THICK, and it was variance, not bias. Correct, but **not yet faster than `D`** — see UPBP-CONV. Since 0.258.0 `-jsurf` adds a **second merge kind** — mode `U`'s surface point×point merges, deposited by the same light subpaths and gathered per camera surface vertex — so connections, beam merges and vertex merges share **one** balance-heuristic denominator; opt-in and CPU-only until validated, after which mode `U` is retired (UPBP-VM) | `bdpt.h` + `surfmerge.h` + `beamgather.h` + `photonbeams.h` + `render_cuda.cu` |
| `V` | validation: renders B and R, reports residual | `main.cpp` |
| `-raster` | z-buffer preview rasterizer + interactive fly viewer (`-explore`) | `raster.h`, `raster_cuda.cu` |

CPU is the default device; `-device gpu|auto` enables CUDA for forward A/B/C,
M-deposit/gather, R, D (untextured), and the raster preview (`-raster-gpu`).

### Which mode-`M` renders actually reach the GPU (0.197.2)

Mode `M` has **two** dispatch routes, and only one of them has a device implementation — so
what looks like a `-device` question is really a routing question:

- **`groupM` → `runSharedPhotonMap`** builds one view-independent map and gathers every camera
  from it (the flythrough win). This is the route with the GPU path,
  `renderPhotonMapSharedCuda`. A *single* mode-`M` camera goes here too — the group of one is
  still the fast route.
- **anything else → `runRender`'s single-camera `mode == 'M'` branch** is unconditionally CPU.
  It never even reads `device`.

The gate is `rc.mode == 'M' && plainRender`. What legitimately belongs in `plainRender` is only
a **progressive budget**: the shared path gathers a fixed spp per frame, so `-time` / `-noise` /
`-forever` / `-preview` genuinely need the single-camera driver. Until 0.197.2 it also included
`resume || wantCheckpointFlag`, which was a pure loss — mode `M` *discards* both flags 3000
lines later (a photon map is persistent state a film-only `.ftbuf` cannot rebuild) and says so,
yet passing `-checkpoint` moved the render off the RTX 4090 and onto 12 CPU threads. A flag the
mode announces it is ignoring must not silently cost it the whole device backend, so those two
terms are gone; the shared path re-emits the "ignoring for mode M" note itself, since the
`runRender` copy is on the branch it no longer takes. The shared path needs no checkpoint
regardless — it writes each frame the instant that frame's gather completes.

**Still CPU-only underneath this:** a `-time` / `-noise` / `-forever` / `-preview` mode-`M`
render. Closing that means teaching the shared device path to gather in spp chunks under
`runSppProgressive`, the way the forward and mode-`R` GPU paths already do (`known-issues.md`).

### The SPECTRAL FOLD — carrying the achromatic fold through surfaces (0.255.0)

**The problem it solves.** A stored photon beam is a *line*: one deposit paints a whole chord,
so a beam that carries a single saturated `CIE(λ)` sample lays a **coloured streak** across the
image rather than a coloured dot. That is the "coloured bars" artifact, and it is chromatic
variance, not geometry — the `1/sinθ` Beam×Ray-1D singularity produces *dots* (a near-parallel
beam projects to a point) and is only log-divergent, so it was never the cause.

Since 0.210.0 the **achromatic fold** (`-beamachro`) removes that variance outright when it can:
if nothing on the path has depended on λ, the estimator's expected colour is the per-emitter
constant `cieMean = ∫CIE·SPD/∫SPD`, so substituting the constant for the sample is unbiased *and*
noiseless. But the claim was **all-or-nothing** — the first λ-dependent event retired it for good
— and instrumenting `gallery_rain` showed exactly where it died:

```
[achrokill] deposits: folded 2004785 | unfolded by: surface 11321, chromatic-medium
                                       scatter 1574, glass absorption 0, GRIN 0, birth 0
```

**87.8 % of unfolded deposits lost the fold to a SURFACE event** — a diffuse albedo — and 12.2 %
to the genuinely chromatic rain. Forcing the fold on for 100 % of beams (a deliberately *wrong*
diagnostic build) measured the floor: cloud chroma p99 **560 → 182**, mean **80 → 23**, with the
bars gone and only the rain's (correct) per-wavelength streaks left. So the whole artifact was
recoverable, and the surface case was the lever.

**The generalisation.** A diffuse albedo is wavelength-*dependent* but not wavelength-
*divergent*: it leaves the path's geometry identical for every λ and only rescales its weight.
So carry the rescaling instead of surrendering. Replace the fold `E_λ[CIE(λ)]` with

```
    cieFold = E_λ[ CIE(λ) · T(λ) ],   T(λ) = Π_j f_j(λ) / Π_j f_j(λ_hero)
```

where `f_j` are the path's spectral factors so far. A deposited beam contributes
`CIE(λ_h)·β·G` with `β = P_emit·Π f_j(λ_h)/Π p_j`; direction pdfs are λ-independent on such a
path, so the `p_j` factor out and the *only* λ-dependence in the estimator is `CIE·Π f_j`.
Substituting gives `E_λ[CIE(λ)·Π f_j(λ)]·G/Π p_j` — exactly the spectral integral the
monochromatic estimator only reaches in expectation. `T(λ_h) ≡ 1`, so a path with no spectral
factors folds to `cieMean` and reproduces the old behaviour **bit-for-bit**.

At a Lambertian vertex the factor is applied as an **analog roulette** (survive with probability
`ρ(λ_h)`, `β` unchanged), so conditional on survival the fold carries `T_k *= ρ(λ_k)/ρ(λ_h)`;
its expectation over the roulette is `ρ(λ_h)·ρ(λ_k)/ρ(λ_h) = ρ(λ_k)`, which is the per-λ weight
the integral wants with the survival lottery cancelled out. Unbiased.

**The quadrature** (`scene.h`, `kFoldBins = 12`). `Emitter::foldCie[k]` is the CIE response
integrated over bin *k* against the emitter's own emission pdf, and `foldLam[k]` is that bin's
SPD-weighted mean wavelength. The bins are cut at **equal SPD mass**, so each carries `1/K` of
the emission probability, the 12 reflectance evaluations land where the emitter actually emits
(a 5800 K sun spends none of them in the far red; a narrow LED spends all of them inside its
line), and — because a 1 nm grid sample is assigned *whole* to one bin, never split — the
identity `Σ_k foldCie[k] == cieMean` is **exact**. That exactness is what makes the unweighted
case bit-identical. `foldN` compacts the live bins forward so a laser line costs one evaluation,
not twelve. Built in `finalizeEmitters` beside `cieMean`, on the same grid.

**When NOT to fold, and why the test may not look at λ_hero.** Folding trades the chromatic
variance of `CIE(λ_h)` for a `1/f(λ_h)` in the weight, and for a sharply *peaked* factor that
is a loss — a saturated red wall sampled at a green hero λ produces a surviving photon carrying
an enormous `T`, i.e. a firefly. Both second moments are computable in closed form from the
quadrature already in hand, so `foldWorthIt` compares them rather than guessing a threshold:

```
    unfolded  E[X²] = E_λ[ f |CIE|² ]        ≈ K · Σ_k f_k |F_k|²
    folded    E[Y²] = E_λ[1/f] · |E_λ[CIE f]|²  ≈ (1/K Σ_k 1/f_k) · |Σ_k f_k F_k|²
```

with `F_k = foldCie[k]`. For a **constant** `f`, Cauchy–Schwarz makes `E[Y²] ≤ E[X²]`
unconditionally — a neutral surface always folds, which is the case the artifact lives in.

The verdict must be a function of the *surface and emitter only*, never of `λ_hero`. If it
depended on `λ_h` the fold/no-fold split would correlate with the wavelength and the mixture
would stop being unbiased: `P(fold)·E_λ[CIE·f] ≠ ∫_{folded λ} p·f·CIE` in general. This is also
why `DiffuseTransmit` tests **both** lobes up front and folds only if both pass — *which* lobe
the photon takes is itself a function of `ρ_R(λ_h)`, so a per-lobe verdict would be a `λ_h`-
dependent verdict. (Its energy guard `ρ_R+ρ_T ≤ 1` is re-applied **per bin**, because the guard
changes the sampling *probability* and the probability is what the analog roulette uses.)

**What still retires the fold**, and why each is right rather than merely conservative:

| Event | Why |
|---|---|
| Dispersive refraction, gratings, thin film / multilayer, fluorescence, hair | The wavelengths no longer share a *chord*. There is no `T(λ)` to carry — the fold is undefined, not just unprofitable. |
| `Layered` coat | Its Fresnel/Airy reflectance is the archetypal peaked factor (near-zero at one λ, near-one two bins away), so `1/R(λ_h)` is an outright firefly generator — and its iridescence is the *point* of the material, which a fold would smear. |
| Glass absorption `exp(-σ_a(λ)·d)` | Foldable in principle; measured **0** deposits lost to it, so it stays retired rather than earning a code path nothing exercises. |
| GRIN | The arc's *geometry* is a function of λ. |
| Scatter in a chromatic medium (`phase rainbow`) | Same: the direction sampling diverges per λ. Measured at 8.53 % of the cloud's foldable deposits after the surface case was absorbed (see the ceiling section below), and it is **correct** that it does not fold — folding the rain would kill the bow. The lever there is the `-beamspec` bundle, which since 0.257.0 **does** carry per-wavelength weights on the record (`float PhotonBeam::wS[]`) and so survives arbitrarily many λ-dependent transport steps instead of dying at the first one. Note this is *scattering* in such a medium; merely **depositing** in one is handled by the gather-time fold (0.256.0), which resolves the colour once the angle is known. |

**No longer retires (0.260.1, `FOLD-GLOSSY`):** `Glossy`. It sat in the specular group by
inheritance from the 0.210.0 all-or-nothing rule, but its lobe *geometry* is wavelength-free —
`sampleGlossy` reads the roughness and nothing else — so the chord test passes and only the
albedo varies with λ, which is exactly what `foldT[]` carries. Both forward tracers now fold
it like Diffuse (albedo on the quadrature + bundle, `foldWorthIt` guard, ratio applied after
the survival roulette), with `FK_DeclineGlossy` for the cases the guard turns down. Sized on
a fog Cornell with a coloured glossy sphere: 9.7 % of foldable deposits had been retiring
under "specular" for it.

**Cost.** `kFoldBins` reflectance evaluations per Lambertian bounce, gated on `achroPath`, which
is only ever set when the render is actually depositing beams. A photon that never reaches a
surface pays nothing: `foldT[]` is left uninitialised until the first spectral factor arrives
(`foldChroma`).

**Not yet ported to the device.** `render_cuda.cu`'s forward tracer still runs the 0.210.0
all-or-nothing rule, so a `-device gpu` mode-`M` render folds less and is noisier than the CPU
one. Both are unbiased and converge to the same image; only the variance differs.

#### What it measured, and where the ceiling is (0.255.1)

`FTRACE_FOLDDIAG=1` turns on a per-medium attribution of every beam that *could* have folded
(its medium is achromatic, so the gather-time tail is flat) but didn't, keyed by the exact
event that retired the path — `FoldKill` / `g_foldKillHist` in `render.h`, printed as a
`[folddiag]` line beside the per-medium beam stats. `FTRACE_FOLDFORCE=1` additionally makes
`foldWorthIt` answer yes unless the fold is *undefined* (a zero bin), so one render separates
"the guard declined it" from "the path genuinely diverged". `FTRACE_NOSURFFOLD=1` (0.257.0) is
the opposite lever: `foldWorthIt` always answers **no**, which retires the claim at every
surface and so restores the pre-0.255.0 (mode `M`) / pre-0.257.0 (mode `J`) rule exactly. It is
the A/B switch the surface fold is *measured* with — one binary, one seed, one fixed `-spp`,
two runs, so the only difference between the two images is the fold itself. That matters
because a `-time`-budgeted run varies ±0.03 relRMSE between repeats at a fixed seed, which is
larger than the effect being measured; and in mode `M` the light-side refresh re-draws the map
on a *wall-clock* cadence, so two `-time` runs are not paired even in principle. Pin both with
`-spp N -beamfreeze`.

On `gallery_rain` at 640×360, mode `M -beams`, seed 1:

```
medium 0: 12047 foldable deposits: folded 90.10%, chroma-medium 8.53%,
                                   specular 1.35%, decline-diffuse 0.02%
```

That is the answer to the question the fold was built to settle. **The variance guard is not
the bottleneck** — it declines 0.02 % of the opportunities, i.e. essentially none; the surface
case that used to be 87.8 % of the residual is now fully absorbed. What is left is 8.53 %
`chroma-medium` (a photon that scattered in the `phase rainbow` rain and then crossed the
cloud) and 1.35 % `specular`, and **neither can be folded, at any tuning**:

- The fold's whole premise is that every wavelength travels the *same chord*, so the only
  λ-dependence left is a scalar factor `T(λ)`. A rainbow scatter samples its outgoing
  direction *from* a λ-dependent phase function, so the wavelengths physically go different
  ways. There is no shared chord to attach `T` to.
- The obvious repair — sample the direction at λ_hero and reweight with
  `T_k *= p(ω|λ_k)/p(ω|λ_h)` — **is biased**, and for the reason the λ_hero-independence rule
  above already names. `ω` is drawn from `p(·|λ_h)`, so the `foldWorthIt` verdict computed on
  those per-bin phase values is a function of λ_hero. Writing the mixture out, the folded
  branch contributes `∫dλ_h p_e(λ_h) Σ_k F_k P_k(A_{λ_h})` and the monochromatic branch
  `Σ_k F_k P_k(A^c_{λ_k})`; those sum to `Σ_k F_k` only if
  `∫dλ_h p_e P_k(A_{λ_h}) = P_k(A_{λ_k})`, which is false in general. Removing the guard to
  restore λ_hero-independence makes it unbiased and useless: at the bow the ratio is exactly
  the peak that fireflies. Drawing `ω` from a λ-independent proposal instead *would* be both
  unbiased and guardable, but it hands the hero path the peak ratio as its own weight, which
  is a strictly worse trade.

So **90.1 % is the deposit-time fold's principled ceiling on this scene**, and the residual bars
are not a bug: they are correct single-wavelength samples of genuinely divergent chromatic
transport, whose expectation is right and whose variance falls with photon count.

Widening the same diagnostic to every medium is what found the next lever, though:

```
medium 1: 14355 foldable deposits: folded 83.89%, chroma-medium 10.87%,
                                   specular 5.23%, decline-diffuse 0.01%
```

83.89 % of the **rain's** beams are on a wavelength-independent path — they are refused not
because the path diverged but because the rain's *gather-time phase* is a rainbow, so a
deposit-time fold has nothing to write down. That is a different problem with a different
answer, and it is what the gather-time fold below solves.

Visually (`scraps/bars_compare.png`, cloud+rain crop, A = 0.254.0, C = forced 100 % fold, E =
0.255.0): the forced-fold floor is dramatically cleaner than either, which is the same fact
from the other side — the ~10 % of beams that cannot fold carry nearly all the visible colour,
because a beam is a *line* and one saturated wavelength paints its whole length. Chroma spread
about the ROI median hue (×1000, luminance-masked, `scraps/_chromabars.py`):

| | mean | p90 | p99 | max |
|---|---|---|---|---|
| 0.254.0 baseline, cloud | 80.32 | 181.37 | 559.56 | 4266.66 |
| 0.255.0 spectral fold, cloud | 73.82 | 171.80 | 545.71 | 4267.33 |
| forced 100 % fold, cloud | 22.75 | 47.81 | 181.65 | 606.97 |

### The GATHER-TIME SPECTRAL FOLD — folding a rainbow phase (0.256.0)

The deposit-time fold asks "what colour is this beam?" and writes the answer on the record.
For a medium whose **phase function** is chromatic that question has no answer at deposit
time, because the colour depends on the scattering angle and the scattering angle is not
known until a camera ray crosses the beam. So the beam is stored monochromatic and paints a
saturated streak. That is the `medium 1: folded 83.89 %` line above: five sixths of the rain's
beams are on a perfectly wavelength-independent path and are refused purely for want of an
angle.

The fix is to move the fold to the other end. A camera ray crossing a beam *does* know
`cosθ`, and for a medium with flat `σ_s`/`σ_t` (`Medium::achroSigma`) the phase is the **only**
λ-dependent factor left in the gather. So the whole spectral integral collapses to a function
of one scalar variable, tabulable once per (emitter, medium) pair at scene-build time:

```
Bow(cosθ) = ∫ spd_e(λ) · CIE(λ) · p(cosθ, λ) dλ  /  ∫ spd_e(λ) dλ
```

`Scene::BowLut` (`scene.h`) holds 8192 bins uniform in `cosθ ∈ [-1, 1]`, integrated on the
**1 nm grid** — deliberately *not* the 12-bin `Emitter::foldCie` quadrature, which exists to
carry a smooth product through a surface and would alias a sharply-λ-peaked bow into visible
colour steps. `finalizeBowLuts()` builds it only for media where `bowLutEligible` — flat
coefficients, chromatic phase — which is exactly the case the deposit-time fold must refuse.

**Splitting `Medium::achro`.** The old flag meant two things at once. It is now
`achroSigma` (are the coefficients flat?) and `achro = !rainbow() && achroSigma` (is the whole
medium λ-independent?). `mediumAchromatic()` keeps its old meaning and its old callers; the
new table keys off `achroSigma` alone.

**The record.** `PhotonBeam::achro` becomes three-valued: `0` = no fold, `1` = deposit-time
fold (use `cieA`), `2` = **gather-time** fold — the path was wavelength-independent and carried
no spectral weight (`T ≡ 1`), but the colour is still undecided. A `short emIdx` names the
emitter whose `bowLut` the gather is to evaluate; `cieA` still carries `cieMean` as the
fallback. This widened the on-disk beam a third time, so `photonmap_io.h` freezes
`PhotonBeamV6` and bumps the magic to `FTPMP07`; a v6 file widens to `emIdx = -1`, which is
exactly "this file predates the gather-time fold". (0.257.0 widened it a fourth time for the
weighted bundle — `PhotonBeamV7` is frozen and the current magic is **`FTPMP08`**; see the
mode-`J` bundle section.)

**`phaseLum`, and why the gather's arithmetic is untouched.** The table stores
`phaseLum = Bow.y / cieMean.y` and `cie = Bow / phaseLum`. The gather substitutes `phaseLum`
for the scalar `md.phaseValue(...)` it would otherwise have computed and `cie` for
`bm.cie[idx]`, so the contribution is `(Bow/phaseLum) · (rest · phaseLum) = Bow · rest` —
identically the spectral integral the monochromatic beams were converging to — while the `> 0`
phase guard and mode `J`'s MIS weight keep seeing a scalar phase value and need no changes.

**Measured** (`gallery_rain`, 640×360, mode `M -beams`, 300 s, seed 7). The plain
"spread about the median hue" metric is the wrong instrument here — a rainbow ROI is
*supposed* to be multi-hued — so `scraps/_streak.py` scores the chroma left after subtracting
a 7 px box blur, which a saturated streak has and a smooth arc does not (×1000):

| | cloud mean | cloud p99 | rain mean | rain p99 |
|---|---|---|---|---|
| `-beamachro off` (no fold at all) | 846.45 | 4146.93 | 385.12 | 1562.53 |
| both folds on (0.256.0) | 219.55 | 1261.85 | 298.12 | 1215.24 |

and at 150 s, seed 1, against the earlier generations (`png/barsdiag/`):

| | cloud mean | rain mean |
|---|---|---|
| A 0.254.0 baseline | 276.69 | 459.95 |
| E 0.255.0 deposit-time fold | 258.02 | 451.72 |
| **G 0.256.0 + gather-time fold** | **246.42** | **381.03** |
| C forced 100 % deposit-time fold | 107.18 | 414.77 |

The last row is the point: the gather-time fold beats a *hypothetical 100 %-successful*
deposit-time fold on the rain, because no deposit-time fold can resolve a rainbow phase at
all.

**Unbiasedness, measured.** The algebra above says the substitution is *exact*, and three
seeds (7, 11, 13) × 300 s of `-beamachro off` vs both-folds-on agree with that: every
per-channel ROI mean moves by less than the **seed-to-seed spread of either arm** (cloud
+3.46/+0.25/−3.77 % against a ±2.3…±5.0 % spread; rain +0.12/+1.51/+1.26 % against ±2.9…±6.3 %;
full frame +4.65/−1.53/+1.61 % against ±4.4…±17.4 %). The same table shows the variance
reduction from the other side: the cloud's blue-channel **seed spread itself** falls from
±5.03 % to ±0.95 %. Comparing single-seed *means* is the wrong instrument here, and instructively
so — the unfolded arm is heavy-tailed by construction (a saturated streak is a rare large
value), so its mean is noisy while its median sits low; folding raises the median toward the
mean without moving the mean.

**Mode `J` does NOT get it — but for a measured reason, not the theoretical one.** Both
reasons are worth stating, because the tempting one turns out to be the weaker one.

*The theoretical objection.* The fold is a *Rao–Blackwellisation*: it replaces the sample
`CIE(λ)·p(cosθ,λ)` by its conditional expectation over λ. That substitution is exact — unbiased
**and** variance-reducing — under one precondition: λ must appear **nowhere else** in the term.
Mode `M` satisfies it exactly, because its weight is the constant 1 (`BeamWeightOne`). Mode `J`
does not: the merge's MIS weight is a ratio of **path densities**, one of which carries the
phase function, so `w1` is *itself a function of λ* and

> `E[w1(λ)·CIE(λ)·p(λ)]  −  E[w1(λ)]·E[CIE(λ)·p(λ)]  =  Cov_λ(w1, CIE·p)  ≠  0`

Folding the weight as well would not remove it: every *connection* technique evaluates its
densities at the camera's hero λ, so a band-averaged merge weight would be comparing itself
against quantities no other technique sees, and the partition of unity would break.

*But that covariance is not measurable here.* Isolating it — bundle suppressed in **both** arms
with `-beamspec 1`, so the fold is the only difference — against the 6212-spp mode-`D`
reference, the folded arm comes out *slightly better* on both seeds (rain relRMSE 0.4157 vs
0.4263 at seed 7; 0.4875 vs 0.5033 at seed 11). Whatever bias it carries is smaller than the
variance it removes. So the theoretical objection, though real, is **not** what decides this.

*What decides it* sits one level up, at the deposit: `BeamBank::push` cannot store a fold and a
`-beamspec` bundle in the same record (`nSec = (lamS && nSec > 0 && !cieA)`), because one
`PhotonBeam` cannot carry both an emitter-folded colour and a set of secondary wavelengths. So
**asking for the fold silently destroys the bundle** — and in mode `J` the bundle is worth about
four times what the fold is worth. Rain ROI, relative RMSE:

| mode `J` rain | seed 7 | seed 11 |
|---|---|---|
| monochromatic (neither) | 0.4263 | 0.5033 |
| bow fold, no bundle | 0.4157 | 0.4875 |
| bundle, no fold | 0.2876 | 0.2835 |
| **shipped** (bundle + the cloud's own deposit-time fold) | **0.2587** | **0.2788** |

Mode `M` orders the middle two the other way round (rain 0.4340 folded against 0.5731 bundled,
seed 11), and that is not a contradiction: in mode `M` the beam map **is** the entire volumetric
estimate, so removing *all* chromatic variance from every crossing beats sampling four
wavelengths of it. In mode `J` the merge is one MIS-weighted technique among several, and the
weight stays λ-dependent whatever the colour does — variance a fold cannot reach, but a bundle,
each of whose members carries a whole wavelength, can.

So mode `J` declines the fold **at deposit time**, by passing `foldEmIdx = -1`, and that is what
keeps its rain beams bundled. The gather-side gate is the belt to that braces: a compile-time
trait on the weight functor, `WeightFn::kFoldGatherTime` — `true` on `BeamWeightOne`, `false` on
`BeamMergeWeight` — so that if a folded bank ever does reach mode `J` (a `-loadmap` of a map
another mode wrote) the gather stays bit-identical to pre-0.256.0 rather than quietly taking a
substitution it cannot justify. Tracked as `UPBP-BOWFOLD` in `known-issues.md`.

`gatherPhotonBeamsW` also hands `w1` a separate `phaseMis`, the monochromatic
`phaseValue(cosθ, λ_hero)`. Under the gate that is redundant — `bow` is null whenever a MIS
weight is in play — but it keeps the invariant the code depends on: a band-averaged phase
belongs in the **estimator** and nowhere else.

**Where it does not apply.** Scattering *inside* a rainbow medium still retires the
deposit-time fold, for the λ_hero-independence reason proved in the previous section — the
table folds a *gather* whose angle is a given, not a *sample* whose direction is drawn from a
λ-dependent density. It also needs `T(λ) ≡ 1`: a beam that picked up a surface albedo carries
per-bin weights the table cannot know, and falls back to the monochromatic record.

**Not on the device.** `render_cuda.cu`'s forward tracer never deposits `achro == 2` (it sets
`emIdx = -1` explicitly), and the device gather has no bow table, so `uploadBeamMapCuda`
**demotes** a CPU-traced `achro == 2` beam to `CIE(λ)` — the pre-0.256.0 record. Pairing the
band-averaged `cieMean` with a single-wavelength phase value would be neither estimator; the
demotion keeps a GPU gather of a CPU-traced map unbiased, merely as noisy as it was. Tracked
in `known-issues.md`.

### Mode `J` (UPBP) — how the two halves are wired (0.214.0, Phase 1)

Mode `J` is deliberately **not** a new renderer. It is mode `D`'s renderer given a pointer:

```cpp
struct BdptRenderer {
    ...
    const BeamMap* beams = nullptr;   // null == mode D
};
```

Everything else follows from that one field.

- **Dispatch** (`main.cpp`, immediately after mode `D`'s block). It refuses through
  `bdptUnsupportedFeature` — mode `J` inherits mode `D`'s scope *exactly*, because a scene whose
  emitters or media BDPT cannot weight is one whose merges it cannot weight either. Then it traces
  the photon pass, builds the beam map, and hands `&bmap` to `renderBdpt`.
- **The photon pass deposits beams only.** `tracePhotonPass(..., depositSurfaces=false)` leaves the
  `PhotonMap` empty: surface transport is BDPT's job here, and a surface map would be both unused
  and — at the photon counts a beam map wants — the largest allocation in the process. Nothing in
  the pass branches on `photonDeposit` except `Renderer::depositPhoton`, which is a no-op when it
  is null, so the beams a beams-only pass deposits are **bit-identical** to those a full mode-`M`
  pass would deposit at the same seed.
- **An empty map is passed as `nullptr`, not as an empty map.** So the merge path is never
  *entered*, rather than entered and skipped per-ray. That is what makes the degenerate-reduction
  test (below) a property of the code instead of of floating-point luck.
- **Flag polarity is reversed from `A`/`B`/`M`.** They need `-beams` to opt in; mode `J` needs
  `-nobeams` to opt out (`g_noBeams`). A mode-`J` render with no beam map is not a degraded UPBP,
  it is literally mode `D`, so making the interesting half opt-in behind a flag whose absence
  silently selects a different mode would be a trap.
- **Everything mode `D` had, mode `J` has**, by being added to the same lists: `useCamera`, the
  `-time`/`-noise`/`-forever` progressive set, the `-resume`/`-checkpoint` set, the fisheye
  refusal, the exposure meter's `case 'D': case 'J':`, the built-in-scene selection, the
  `camera_path` lens routing (`J` keeps its mode when given a lens, as `D` and `P` do), and the
  `bdptUnsupportedFeature` → fall back to `B` rule.

**Validation gate 1 (the degenerate reduction), passed 0.214.0.** With no beam map, mode `J` must
be mode `D` byte-for-byte — twice over, once with no media at all and once with the whole beam
prologue actually running:

| Test | Command pair | Result |
|---|---|---|
| 1a — no media | built-in cornell, `-mode D -device cpu` vs `-mode J`, 96², spp 4 | `cmp` **identical** |
| 1b — media, map built | `_fog_cornell.ftsl -mode J` (beams default on: 7 873 846 beams from 200 000 photons, BVH built) vs `-mode D -device cpu` | `cmp` **identical** |
| 1c — media, map refused | `_fog_cornell.ftsl -mode J -nobeams` vs `-mode D -device cpu` | `cmp` **identical** |

1b is the load-bearing one: it proves the photon pass and the BVH build perturb neither the BDPT
RNG stream nor anything else the connection half reads. 1c is the cheap companion that keeps the
opt-out path honest. Compare mode `D` with **`-device cpu`** —
plain `-mode D` takes the GPU BDPT megakernel (`gpuBdptMode = (mode == 'D')`) and is a different
realisation, which is not a bug and not a comparison.

The MIS derivation for the merge weights, the remaining gates, and what UPBP is and is not
predicted to buy are in `known-issues.md` under the UPBP entry.

### Mode `J` Phase 2 — the merges, still unweighted (0.215.0)

Phase 2 adds the second estimator and **deliberately does not weight it**, so that the plumbing
can be proved correct before the MIS arithmetic (Phase 3) is layered on top of it. A Phase-2
render is therefore *wrong on purpose*: both techniques contribute at weight 1, so the volume
term is counted about twice.

**The estimator was extracted, not duplicated.** `gatherPhotonBeams` lived in
`photonmap_render.h` while mode `M` was its only caller. Mode `J` lives in `bdpt.h`, which has no
business including the photon-map renderer (that would drag in `causticaim.h`, `photonmap_io.h`
and the whole deposit machinery to reach one function), and moving BDPT's merge the other way is
worse still — a UPBP merge weight is built from the *BDPT subpath densities*, so it belongs with
them. The primitive therefore moved **down** to the level both callers already share, as
**`src/beamgather.h`**, which includes exactly `render.h` (for `Renderer::mediaTransmittance`)
and `photonbeams.h` (for the map).

It gained one thing: a **per-hit weight hook**, as a *template* parameter rather than a
`std::function`. Mode `M` sums the estimator raw; mode `J` must scale each beam hit by its own
MIS weight, because the balance-heuristic ratio depends on the merge geometry (`sinθ` above all)
and so differs from hit to hit — a weight the caller cannot apply after the fact, because by then
the hits have been summed. The default functor returns `1.0`, folds away, and leaves mode `M`'s
generated code as it was; `gatherPhotonBeams` survives as that instantiation under its old name
and signature, so every existing call site is untouched. The weight is applied *before* the
`w > 0` rejection, so a technique the weight kills costs no transmittance marches.

**Camera SEGMENTS, not camera vertices.** A merge is an alternative to the camera's free-flight
*distance sample*, so it has to see the whole span that sample was drawn from — the segment from
a vertex to the **surface** that ends it, not to the medium event that happened to be sampled
inside it. `randomWalk` was throwing that span away, so it now records a `CamSeg` per bounce
(origin, direction, distance-to-surface, throughput, glass absorption, and the index of the
originating camera vertex, which Phase 3 needs for the MIS partials). `CamSeg::beta` carries
**no** free-flight factor: ftrace's media transport is analog, so a homogeneous free flight draws
from the exact transmittance pdf, `beta` is unchanged across the event, and the throughput at a
segment origin is the throughput anywhere along it. The gather then applies each beam's own
`Tr` to *its* closest-approach point.

The merge loop runs **after every connection**, deliberately. The gather draws from `rng` (ratio
tracking in a heterogeneous medium), and running it last means those draws cannot shift the
connection half of the same sample; they cannot shift the next sample either, because the stream
is re-seeded per (pixel, sample). That is what keeps gate 1 alive even though the two halves
share a generator. The gather returns XYZ built at the **beams'** wavelengths (the standard
spectral-photon-mapping estimate, and the same choice mode `U` makes for its merges);
`sg.beta` is this sample's own hero throughput, a scalar, and multiplies that triple — matching
mode `M`, whose `L += gather(...) * thr` likewise carries no camera-λ `invPdf`.

**Validation, passed 0.215.0** (`_fog_cornell.ftsl`, `-device cpu`):

| Test | What it proves | Result |
|---|---|---|
| mode `M -beams` on 0.214.0 vs 0.215.0 binaries, same seed | the extraction into `beamgather.h` changed nothing | `cmp` **identical** |
| `-mode J -nobeams` vs `-mode D -device cpu` | gate 1 still holds now that merges exist | `cmp` **identical** |
| `-mode J` vs `-mode D`, equal spp, `-hdr` | the merges land, and land *only* additively | see below |

The third is the informative one, and it needs `-hdr`: a PNG is auto-exposed, which renormalises
a global brightness change away entirely. On the scene-linear PFMs (`scraps/mergediff.py`), with
mode `J` sharing mode `D`'s stream so that `J − D` *is* the merge output and nothing else:

```
J/D mean ratio  : 2.0374
pixels touched  : 65536 / 65536 (100.0%)
pixels brighter : 65536 / 65536 (100.0%)
pixels dimmer   :     0 / 65536 (  0.0%)
```

Zero dimmer pixels is the load-bearing number: a merge can only add, so a single negative pixel
would mean the two runs' RNG streams had diverged and the connection halves were no longer
comparable. The ≈2× ratio is the expected double count.

**Cost.** The merge is currently the dominant expense — 61 s/spp against mode `D`'s 0.55 s/spp on
this scene, because a probe ray gathers ~100 beams and each surviving beam costs two
transmittance marches, now paid per camera *segment* per bounce rather than once per camera ray
as in mode `M`. Phase 3's weights will kill many hits before their marches (hence the ordering
above), but this is the number to watch.

**Phase 3a replaces the borrowed photon pass** (done in 0.216.0, next section). Phase 1 built the beam map with
`tracePhotonPass` because that was the cheapest way to get *a* map; that is a scaffold, not the
design. A balance-heuristic weight is a ratio of the densities with which competing techniques
would have produced the same path, and a beam from `Renderer::tracePhoton` carries no densities
and is not even drawn from the same distributions the connection half is weighted against — so
the ratio would be between two things that are not comparable. Mode `J` will therefore trace its
own light subpaths with the existing `randomWalk`, recording a `LightSeg` per bounce exactly as
Phase 2 records a `CamSeg`, and depositing one **long** beam per (segment, medium) crossing.
`tracePhotonPass` / `emitBeams` and their CUDA twins are then untouched, and `PhotonBeam` grows
no fields on behalf of a mode that does not use them. See `known-issues.md` for the full
argument, including why the beams must stay *long* (a short beam's stochastic length already
carries `Tr`, which `beamgather.h` would then apply a second time).

### Mode `J` Phase 3a — mode `J` traces its own light subpaths (0.216.0)

`bdpt::traceLightBeamPass` replaces `Renderer::tracePhotonPass` as mode `J`'s deposit pass. It
runs `generateLightSubpath` — the *same* function whose vertices the connection half is weighted
against — once per emitted path across a thread pool, and deposits the resulting segments through
the existing `Renderer::emitBeams`. `tracePhotonPass`, `emitBeams`, `BeamBank`, `PhotonBeam` and
their CUDA twins are all unchanged, exactly as the Phase 2 note promised: the new pass is a
*caller* of the deposit machinery, not a fork of it.

**`CamSeg` became `PathSeg`, and that is the whole trick.** The span recorder Phase 2 added to
`randomWalk` was never camera-specific — `randomWalk` is the shared walk, and a light subpath's
"vertex to the surface that ends the segment" span is *exactly* the long beam the estimator
wants. So the struct was renamed (`CamSeg`→`PathSeg`, `camVert`→`vert`), `generateLightSubpath`
and `deltaLightSubpath` gained a trailing `PathSegs*`, and the deposit pass is a walk plus a loop
over the segments it recorded. No second traversal, no duplicated media logic.

Four consequences worth knowing:

- **`-n` now counts light subpaths, not photons.** The two are the same quantity, which is why
  `nEmitted` and the `1/nEmitted` in `beamgather.h` need no change: `generateLightSubpath` starts
  from `betaWalk = Le·cos/(pdfChoice·pdfPos·pdfDir)`, which reduces to `Le·area/pdfChoice` =
  emitter power, and `Renderer::tracePhoton` is born carrying `scene.totalPower`. Same units, so
  a beam deposited by either pass means the same thing.
- **The map now carries multiple scattering**, because a light subpath keeps walking after a
  medium event and records a segment for every bounce. Phase 1's borrowed map did too, but now
  every one of those segments has a *vertex index* (`PathSeg::vert`) pointing back into a subpath
  whose densities Phase 3b can read — which is the entire reason for the change.
- **No Russian roulette on the deposit.** The banks are built with `cap = 0`, so `BeamBank` never
  self-halves and `keepProb == 1` for every beam. Mode `M`'s self-thinning leaves each beam a
  history-dependent existence probability that a merge weight cannot reconstruct at gather time,
  and a MIS weight that cannot read the density it is dividing by is not a MIS weight.
  **`-beamcount` therefore cannot be met by *thinning* in mode `J`** — but since 0.242.0 it is
  not inert: it is met by tracing **fewer subpaths**, sized by a discarded pilot. See the beam
  budget below.
- **Mode `J`'s beams get BOTH the achromatic fold and the `-beamspec` bundle (0.250.0, 0.251.0).**
  This bullet used to say neither applied, on the grounds that both "need a wavelength-independent
  `beta`" and a BDPT light subpath's is not (`Le` carries `spd(λ)·invPdfLambda` off the
  **scene-wide** emission sampler, so `spd/pdf` is a function of λ whenever the scene holds more
  than one emitter — `gallery_rain` holds five). That error cost a visibly iridescent cloud *and*
  rain curtain: mode `J` deposited one saturated wavelength per subpath, and a beam is a *line*,
  so each deposit laid a coloured streak down a whole chord. Both halves are now fixed, by two
  different arguments that are worth keeping separate.
  - The **fold** (0.250.0) only needs the ESTIMATOR'S EXPECTATION to be λ-independent, which holds
    for any sampling density `p`:
    `E_λ[β(λ)·CIE(λ)] = ∫p·(K·spd/p)·CIE = K∫spd·CIE`, and
    `E_λ[β(λ)]·cieMean = K(∫spd)·(∫spd·CIE/∫spd)` — the same number. So substituting
    `Emitter::cieMean` for the sampled `CIE(λ)` is unbiased in mode `J` exactly as in mode `M`,
    with no change to `beta` at all. That is what neutralised the cloud.
  - The **bundle** genuinely does need a pointwise λ-independent `beta`, because the record stores
    wavelengths and *no per-wavelength weights* — every member has to carry identical power. Mode
    `J` gets there (0.251.0) by **converting its subpath into mode `M`'s**, analytically, at the
    subpath's birth (`BeamSpectral` / `beginBeamSpectral`, `bdpt.h`):
    - `scale = ∫spd_em / (spdFn(hero)·invPdfLambda(hero))` multiplies `beta` so that the
      `spd_em(λ)/p_comb(λ)` it was carrying becomes the flat `∫spd_em` a mode-`M` photon carries.
    - the deposit's wavelengths are then **all** drawn stratified from that emitter's OWN SPD.
      The subpath's hero λ is **replaced, not extended**: it came from the scene-wide mixture
      density, so weighting it `1/C` beside the others would bias the estimate toward
      `∫p_comb·f` instead of `∫p_em·f`.

    The result is arithmetically the deposit mode `M` would have made, so mode `M`'s bundle *and*
    fold both become valid. **With one reference-wavelength rule (0.259.3, `UPBP-CHROMA`):** the
    replacement only covers an *unfolded* deposit, whose power `beta·scale·fW[0]` is the flux
    at member 0 and is gathered at `CIE(lam₀)`. A *folded* deposit is gathered at
    `Σ foldCie[k]·foldT[k]`, whose ratios are relative to the walk's own hero, so its power
    must be `beta·scale` at that hero — re-expressing it at member 0 left a stray
    `T(lam₀)/T(λ_walk)` whose mean over two independent wavelengths is `E[T]·E[1/T] ≥ 1`:
    nothing on a flat wall, +40 % in X off the red one. This is a strict improvement even at `-beamspec 1 -beamachro off`,
    where it reduces to "deposit at a wavelength drawn from the right density instead of the
    wrong one" and removes the `spd_em/p_comb` weight-ratio variance — measurably: it took the
    **cloud** crop's chroma noise *further* down (saturation 0.0560 → 0.0390) even though the
    cloud was already folding.
  - **Wavefront beam gather on the device (0.261.0 — `UPBP-CONV`).** The device gather was one
    thread per camera path walking the beam BVH and evaluating every surviving beam in place —
    warp-divergent by construction, which is what a 17× gap between `-beamk 32` and `-beamk 1`
    looked like. Mode `J`'s camera pass now separates the phases: `kBdptT` queues each segment
    with the weight state it already builds (`DWfSeg` = segment + `DBeamMergeW` + `DTrRay` +
    pixel/λ/seed), `kWfBeamHits` (one thread per segment) enumerates `(segment, beam, t, s)`
    candidates, `kWfBeamEval` (one thread per candidate) runs the estimator — `dBeamHitEval`,
    hoisted so the inline path calls the same function — and accumulates with atomics. The host
    runs a chunk as consecutive waves, each a contiguous range of pixels — a horizontal band. The render’s first wave is 1024 paths; every later one is the largest size that keeps the densest band it would cover at ~70 % of the hit queue, up to the segment-sized ceiling `segCap / (depth + 3)` — judged from the previous wave (bands are neighbours) and, from the second chunk on, from the previous chunk’s **per-band profile**: the hits per path and segments per path every band produced, keyed by pixel slot, stationary across chunks (same map, same camera). That size is a fixed point (a smaller wave reaches fewer bands), reached by shrinking from the last-wave size, so a band gets what its own density allows and the jumps the last wave cannot see — the chunk boundary (bottom band → top band) and the doubling at a light’s edge inside every chunk — are simply profile entries. The profile persists across the progressive loop’s epochs — each epoch re-enters `renderBdptCuda` with a fresh map, ~17 times a minute on `_fog_thick`, and the band densities are the same map to map — so every epoch’s first chunk opens with a properly sized wave instead of relearning from 1024 paths. That carry is what keeps a dense map out of the on-the-spot path: on `_fog_cornell` 128² it is the difference between **28.2 M spilled hits and none at all** (same 21 spp) — without it every epoch re-learns through the same overflowing waves — and on `_fog_thick` it is worth +0.6 %. Measured against 0.261.0’s schedule on the same binary (`FTRACE_WFLEGACY`, alternating 60 s passes so machine drift hits both): **+2.2 % paths on `_fog_thick`** (4534 against 4434 spp) with the waves 36 % larger (65 k against 48 k paths) and 0.727 against 0.753 ms per 1000 paths. (Cross-binary readings taken hours apart disagreed by up to 8 % in both directions; this machine drifts that much over an evening, so every schedule claim here is a same-binary alternating A/B.) (The intermediate designs were measured: the last wave’s size carried across chunks doubled `_fog_cornell`’s spill, 59 M → 117 M hits at 256² / 60 s, the sparse bottom band’s size landing on the dense top band; the chunk’s densest wave carried across still left 94.6 M, because the light’s edge is inside every chunk.) The alternative was measured and rejected as the default: `FTRACE_WFSTRAT=1` enumerates the chunk’s pixels in 32-pixel runs permuted by an odd golden-ratio multiplier on a power-of-two run domain (`DWfQueue::runMul`, a division-free bijection; padding runs and slots are skipped, per-pixel sample streams are unchanged), so every wave covers the whole image like a Fibonacci lattice and the size estimate becomes exact (spill 59 M → 8.4 M) — but it costs 7 % of `_fog_thick`’s paths through the same code (4430 against 4765 spp / 60 s): the segment queue is filled by grid-wide atomics, so a `kWfBeamHits` warp holds 32 segments from random paths *of the wave*, and a band’s paths are alike while a stratified wave’s are as unalike as the image allows — warp divergence in the beam-tree walk. A spilled candidate is evaluated in place and costs nothing measurable, so bands win; overflow
    degrades to the inline gather (segments) or on-the-spot evaluation (hits), never to a drop.
    The per-hit RNG stream is derived from the segment seed and beam index, so heterogeneous
    transmittance stays stochastic and unbiased. `FTRACE_NOWAVEFRONT=1` is the A/B control. The
    render reports its queue use (`[gpu] mode J wavefront gather:`) and where its seconds went —
    CUDA-event time of the walk, the beam-hit search and the hit evaluation, and the host gaps
    between waves (`[gpu] mode J wave loop:`; the wave already syncs, so the events are free).
    Measured: `_fog_thick` 128² at `-beamk 32`, 60 s: 380 → 4905 spp (**12.9×**).
  - **Weighted bundles, and one shared surface rule (0.257.0 — `UPBP-BOWFOLD`).** Both claims used
    to die at the **first surface interaction of any kind**, which is what capped mode `J` at
    36.2 % bundled / 47.0 % folded: a subpath that so much as grazed a diffuse wall deposited the
    rest of its chords monochromatically. Two things were wrong with that rule.
    - The record had **no per-member weight**, so a bundle's members had to carry *identical*
      power, so any λ-dependent factor — even a plain Lambertian albedo — genuinely did invalidate
      it. `PhotonBeam` now carries `float wS[kBeamSecMax]`, the gather multiplies member `i` by
      `b.wS[i]`, and the members may therefore diverge in power. This widened the on-disk record a
      fourth time: `photonmap_io.h` freezes `PhotonBeamV7` and the magic becomes **`FTPMP08`**. A
      v7 file widens to `wS[k] = 1.0f`, which is **exact**, not merely safe — equal weights is
      precisely what a v7 bundle meant.
    - With weights available, the retirement rule becomes the same one the fold already wanted:
      only a wavelength-**DIVERGENT** event retires the claim (the table at the top of this
      document — dispersion, gratings, thin film, fluorescence, hair, GRIN, glass absorption, a
      scatter in a chromatic medium). A merely wavelength-**DEPENDENT** event — a diffuse albedo —
      is now *absorbed* into the weights instead: `foldT[k] *= f(foldLam[k])/f(λ_h)` for the fold's
      12 quadrature bins and `beamW[i] *= f(bs.lam[i])/f(λ_h)` for the bundle's members, in one
      evaluation loop over `fldF[kFoldBins + kBeamSpecMax]`, because they are the same quantity on
      two different grids. `Diffuse` and `DiffuseTransmit` are covered; `Glossy` is deliberately
      **not** (see the follow-up note in `known-issues.md`).
    - `foldWorthIt` — the second-moment verdict that decides whether `E[1/f]` is affordable — was
      **hoisted out of `tracePhoton` into a free function in `render.h`**, because both forward
      tracers now need the identical verdict: `tracePhoton` (mode `M`) and `randomWalk` (mode `J`)
      deposit into the *same* `PhotonBeam` records read back by the *same* gather. If their rules
      could drift, two beams in one bank would disagree about what `power`, `cieA` and `wS` mean.
      `DiffuseTransmit` must test **both** lobes up front, since which lobe is taken depends on
      λ_h, and its expected-value ratio `secF/betaFactor = ρ_k(lobe)/ρ_hero(lobe)` is identical to
      the analog-roulette ratio `render.h` uses — the two tracers really do agree.
    - `beamW[0]` is what makes this exact in mode `J` specifically. Unlike `render.h`'s `specW`
      (relative to a hero that *is* a member), the bundle's hero λ_walk is **not** a member — it
      came from the scene-wide `p_comb`, and the λ-replacement to `bs.lam[0]` is inseparable from
      the `scale` factor described above. So `beamW` covers member 0 too, deposit power becomes
      `sg.beta · dSc · fW[0]`, and `wS[i-1] = fW[i]/fW[0]`. A zero `fW[0]` means `T(bs.lam[0]) == 0`
      — no flux — and the deposit is skipped.
    - **Measured** on `gallery_rain` at 640×360, mode `J`, seed 11: coverage **47.0 % → 82.4 %**
      folded (cloud) and **36.2 % → 75.1 %** bundled (rain), against mode `M`'s 90.8 % / 81.8 %.
      Coverage *by power* moved much less (91.2 → 93.6 %, 85.9 → 88.1 %) because the newly-covered
      beams are the low-energy post-bounce ones — which is exactly why the **tail** metrics move
      far more than the means. At a fixed `-spp 64` (same binary, same seed, `FTRACE_NOSURFFOLD`
      off vs on) seed 11 gives cloud chromaRMSE **2.0938 → 1.2067** (−42 %), cloud streak p99
      **2601 → 1516** (−42 %), rain relRMSE **0.3414 → 0.3107** (−9.0 %); seed 7 gives cloud streak
      p99 1438 → 1329 (−7.6 %) and rain streak mean 263.5 → 255.5 (−3.0 %). **Bias is unmoved at
      both seeds** (cloud +9.92 → +9.86 and +11.51 → +11.64; rain +2.31 → +2.35 and +4.38 → +4.33),
      which is the point: the change is *correct*, not merely quieter.
    - `FTRACE_NOSURFFOLD=1` makes `foldWorthIt` always answer NO, restoring the retire-at-any-surface
      rule exactly. It is the A/B switch the numbers above were taken with, and it exists because
      `-time`-budgeted runs vary ±0.03 relRMSE run-to-run — more than this feature's effect — so the
      measurement has to be one binary, one seed, one fixed `-spp`, two runs.
    - **Mode `M` is unchanged, and was proven so rather than assumed** — the fix restructured
      `render.h`'s claim birth and every retirement site, so `M`'s light pass had to be re-measured:
      same 11612 / 14005 chords, same 90.8 % / 81.8 % folded, same 878641 deposits, same
      25617 → 608871 split as v0.256.0. Its one behavioural delta is a new **2.2 % spectrally
      bundled** on medium 1 — the weighted bundle collecting part of the gather-time fold's
      residual — which a paired deterministic A/B shows is a strict improvement (rain chromaRMSE
      −4.9 %, streak mean −6.6 %, bias toward zero; cloud identical to four decimals). See
      `known-issues.md` for the table and for the two measurement traps it exposed: `-beamfreeze`
      silently makes a render eligible for the **device**, which implements neither fold, and mode
      `M`'s light-side refresh re-draws on a **wall-clock** cadence, so `-time` runs are never
      paired. Compare with `-device cpu -spp N -beamfreeze`.
  - The two are **alternatives per beam, not additions** — `BeamBank::push` prefers the fold when
    offered both — and `emitBeams` chooses **per medium**, so one segment crossing `gallery_rain`'s
    achromatic cloud and its `phase rainbow` rain folds the cloud's beam and bundles the rain's.
    The rain cannot fold (its scattering really is chromatic, so `mediumAchromatic` is false) but
    it can and now does bundle, which is what neutralised the curtain. Measured on `gallery_rain`
    at 640×360, 180 s, seed 1 — rain-crop saturation **0.1320 → 0.0928**, against mode `D`'s
    0.0933 and mode `M`'s 0.1085, so mode `J` went from worst to matching the path-traced
    reference and beating mode `M`. Whole-frame `Y` moved +0.2%, i.e. the λ swap introduced no
    exposure shift.
  - The λ swap is **MIS-safe**: `mergeEtaPrime` uses λ only for `md.sigmaT(λ)` and `trDet(...)`,
    i.e. only for **extinction**, which `beamSpectralOK` already guarantees is λ-free scene-wide.
    So the light-side merge partials (`accC`/`accM`, computed at `hb.lam[0]`) stay exactly right
    while the *deposit* moves to a different wavelength.
  - Nothing on the CAMERA side has to be λ-independent either, and that is worth stating because
    it looks like an exposure and is not: the merge already multiplies a camera-λ throughput by a
    beam-λ colour (the documented spectral-mismatch fudge in `mergeWeightJ`), so the two λs are
    already independent draws. The fold replaces `E[CIE(λ_b)·g(λ_b)]` with `cieMean·E[g(λ_b)]`,
    which are equal under the same per-medium flatness the fold already tests — whatever the
    camera side happens to be.
  - Implementation: `PathSeg` carries `achro` (the per-segment half of the claim), the subpath
    carries one `BeamSpectral` (the per-emitter half), and `randomWalk` runs the same
    survives-an-achromatic-scatter rule `Renderer::tracePhoton` runs (since 0.257.0: cleared by a
    wavelength-*divergent* surface event, by glass absorption, and by a scatter in a chromatic
    medium — but **not** by a diffuse bounce, which is folded into `foldT`/`beamW` instead, and
    **not** by a scatter in an achromatic medium, which is what lets it ride the cloud's ~278
    bounces). `PathSeg` carries the resulting `fCie` and `fW[]` per segment. The rule tests the
    medium the walk is scattering *in*, not the media the segment *crosses*, which is precisely
    why a subpath that scattered in the cloud can still deposit a good bundle into the rain.
    `traceLightBeamPass` hoists `beamSpecOK`/`beamAchroOK`/`beamSpecC` out of the worker
    (`beamSpectralOK` scans every medium's spectra and must not run per subpath).
    No GRIN guard is needed, unlike the photon walk: `bdptUnsupportedFeature` refuses GRIN
    scene-wide. No device twin is needed: both land in `BeamMap::cie[]` / `PhotonBeam::lamS[]` at
    map-build time, which is what the CUDA gather reads, and mode `J`'s light pass is CPU-only on
    both backends.
  - Mode `D` is **bit-identical**: `generateLightSubpath`'s `bs` parameter defaults to `nullptr`
    and only `traceLightBeamPass` passes one, so `beginBeamSpectral` — the only new RNG consumer —
    never runs outside the beam pass and the general BDPT RNG stream is untouched.
  - Coverage is reported per medium on the `photon beams: medium N:` log line, as
    `NN% folded achromatically` and/or `NN% spectrally bundled`, each by count and by **power**
    (`power × len`, since a long beam lays a long streak). A medium that structurally cannot fold
    prints `achromatic fold n/a` rather than `0.0%`, so a rainbow curtain reads as out of scope
    instead of as a failure — while still showing its bundle figure, which is what is doing the
    work there.

**Validation, passed 0.216.0** (`_fog_cornell.ftsl`, 128², `-device cpu`):

| Test | What it proves | Result |
|---|---|---|
| `-mode J -nobeams` vs `-mode D -device cpu` | gate 1 survives the new pass | `cmp` **identical** |
| `-mode J` vs `-mode D`, equal spp, `-hdr` | the own-pass merges land, and only additively | `J/D` mean **1.9348**, 16384/16384 brighter, **0 dimmer** |

The ratio landing next to Phase 2's 2.0374 — from a differently-built map, with multiple
scattering added and the spectral bundle removed — is the corroboration that matters: a π or 4π
normalisation error could not reproduce it.

**Cost, and the memory to watch.** At `-n 200000` on `_fog_cornell` the pass produced 1 202 979
chords → 7 668 297 post-split beams (38.34 beams/subpath, **761 MB**), in 11.1 s of which 10.9 s
was the BVH build. The map was sized by `-n` with **no trim** — mode `M`'s decimation is exactly
what Phase 3a had to give up — so this was the resource that bit first on a real scene. It is
now bounded by the beam budget below.

### `-sunnee` — the sun's single scatter by next-event estimation, not beams (0.312.0)

**What it fixes.** A collimated light through a thin medium is, in the beam map, a handful of
long coherent chords per pixel: every order-1 sun chord in the rain is a line the gather kernel
paints along its whole length, so the estimate's error is *structure* (streaks, shafts), not
grain, and it does not average like grain. Measured on gallery_rain's backlit rain: 5x the
photons (five shared realizations) took the error from 13.2 to 9.8 levels against a converged
mode-D reference and a fit put the floor near 8.7 -- the kernel's own bias, which no realization
count touches (known-issues, "BEAM STREAKS"). The order >= 2 population was already replaced by a
smooth cached field (VOLCACHE); what remained *as beams* was exactly the order-1 sun population.

**The estimator.** With `-sunnee`, mode `M`'s gather adds, per straight camera segment and per
medium, the single-scatter integral of every `light sun` directly:

    L_1 = integral_0^tMax  Tr_cam(t) * sigma_s(p_t) * phase(cos) * Tr_sun(p_t) * L_sun * Omega  dt

evaluated with `pbeams::kSunNeeSteps` (64) **jittered** steps over the medium's clipped span --
one random offset per segment, so it is an unbiased estimate rather than a fixed quadrature --
`Tr_cam` carried incrementally from step to step, and `Tr_sun` by ratio tracking to the scene
sphere plus a shadow ray, with the sun direction cone-sampled (pdf 1/Omega) exactly as the BDPT
volume-vertex sun connection does (`backward.h neeVolume`, `render_cuda.cu` line ~10600). An
unbounded medium is cut at the deposit's own far limit (`kBeamFarScale`), so both estimators
integrate one volume. Host: `beamgather.h sunNeeMarch`; device: `render_cuda.cu dSunNeeMarch`;
called from the two segment sites of `photonGather` / `dPhotonGather` next to the beam gather,
and not gated on the map being non-empty, because the erase below may have emptied it.

**The erase, and why it needs provenance.** `sunNeeSplit` (beamgather.h, run by `buildBeamMap`
before the VOLCACHE split) erases the chords the march now accounts for: **order 1, born on a
`Sun`, no surface interaction before the chord**. Order 1 alone is not enough -- a photon that
bounced off the ground and then crossed the rain is order 1 too, and that bounced light is
something the march cannot see -- so the deposit now records **provenance** on every chord:
`PhotonBeam::srcEm` (the emitter the photon was born on; -1 for a volumetric fire birth) and
`PhotonBeam::surf` (surface interactions so far, counted in `tracePhoton` / `shadeStep` right
before the material is handled; 255 = unknown). Both live in the record's tail padding, so the
record is still 120 bytes; the map file magic is `FTPMP09`, and an `FTPMP08` file loads at the
same stride with the two fields reset to unknown. **Unknown provenance makes `-sunnee` refuse the
map**, loudly, and clears `pbeams::gSunNee` -- erasing blind would drop bounced light, and erasing
nothing while the march runs would count the sun twice. The flag travels with the map it was
decided on: `DBeamMap::sunNee` is stamped at upload, after the split, so both backends see one
decision. Env-lit and bounced order-1 chords, and every order >= 2 chord, stay as beams.

**Colour.** The term is folded where the beams' would have been, so a rainbow is deterministic
in colour rather than chromatically grainy: a fully achromatic medium (`Medium::achro`) takes
the sun's CIE integral `Emitter::viewXYZ` outright; a bow medium with a `Scene::bowLut` for this
(emitter, medium) pair takes `Bow(cos) * integral SPD` (the table's `cie * phaseLum` times
`viewXYZ.y / cieMean.y`); anything else evaluates at the hero wavelength times `invPdfL` like
every other spectral term. Transmittances are at the hero wavelength in every branch, exactly
as the folded beams' are. `DEmitter::viewXYZ` was added for the device side.

**`-beamachro all` and the trilinear cache read (0.313.0) — the colour blotches.** With the sun
marched and order >= 2 cached, the cloud still showed soft colour blotches at 960x540. Measured:
the cloud's chroma noise was 17.6 levels of which 4.6 were per-seed and the rest **per-map** — a
fixed 3D mottling that would ride along through the flyby — from the ~3 % of chords by power
whose paths were wavelength-dependent (through the rain's bow, or glass) and are therefore
stored at ONE hero wavelength: each is a saturated streak, or a tinted cache cell. The chroma
denoiser then smooths that into 10–20 px patches (its correlation length grows from 1 px to 8+),
and because its weights follow the per-seed luma, the patches move frame to frame. Two changes:
`-beamachro all` is a post-pass in `buildBeamMap` (after the `-sunnee` erase, before the
volcache split) that folds every unfolded chord at its emitter's `cieMean` — a bundle's total
power, `achro = 1`, or `achro = 2` through the bow table where the (emitter, medium) pair has
one — accepting the covariance of path power with CIE(λ) as bias; and `VolCache::fetch` /
`dVolCacheFetch` read the fluence grid **trilinearly between cell centres** (edge-clamped)
instead of nearest-cell, which is what painted a cell's realization noise as a five-pixel square.

**The sun's disc through the cloud (0.313.1).** Correct in brightness — the mode-D reference has
it at the same value — but mode M's estimate of it swung ±25 % between seeds at 32 spp: the
disc is ~10⁴ times brighter than the cloud around it, and the cloud's transmittance along that
one ray was a single ratio-tracking sample per camera sample. `camMediaTr` / `dCamMediaTr`
(the two extinction sites of `photonGather` / `dPhotonGather`) average
`pbeams::kSunDiscTrSamples` (64) samples for a camera segment whose direction lies inside a
sun's disc, and one sample otherwise — unbiased either way, and paid only by the handful of
pixels on the disc.
That turned out to be the smaller half. With the transmittance averaged the disc still swung
3100–5300 between seeds while every other pixel stayed bit-identical: the rest was the **hero
wavelength** — each camera sample evaluates the disc's blackbody at one λ, and CIE-Y under a
random λ has ~140 % relative spread, ±25 % at 32 spp. So (0.314.0) the gather **stratifies λ
across a pixel's samples**: in `kGather` sample `s` of `sppTotal` takes stratum `s` of the
emission CDF (`dSampleSceneLambdaU`), jittered within the stratum, under a per-pixel random
rotation (a `DRng` seeded from the pixel) so neighbours do not walk the spectrum in step; the
jitter draw stands in for the sampler's own uniform, so the sample's remaining stream is
unchanged. `renderPhotonCamera` does the same per chunk (`photonGather(..., lambdaU)`,
`EmitSampler::sampleAt`), each chunk being a complete stratification on its own. This is the
1/√N → ~1/N^1.5 improvement of every bright spectral term in mode M's gather, not just the disc.

**Cost.** Per camera segment and medium: 64 shadow rays and 64 short ratio-tracking calls, plus
the incremental camera transmittance -- independent of the beam count, which is the point; the
beams it removes were the ones every probe ray had to walk. Measured (gallery_rain frame 555,
320x180, spp 32, GPU): 44.6 % of the map's chords were direct sun; erasing them took a probe
from 528 to 243 beams and the frame from 163 s to 142 s with energy parity within 1 % per ROI;
with the volcache on as well, 102 beams and 76 s. What the flag cannot remove is the physical
fan of the rain shadowing itself along the sun direction, nor the lines of the sky-lit order-1
chords that stay as beams (known-issues, "-sunnee").

### Imported glTF dielectrics keep their specular lobe (0.316.0)

**What was wrong.** `gltf.h` mapped `pbrMetallicRoughness` onto one BSDF per material: transmissive
-> `Dielectric`, `metallic >= 0.5` -> `Glossy`, **everything else -> `Diffuse`**. But glTF's model
gives a dielectric a specular lobe as well as an albedo -- F0 = `((n-1)/(n+1))^2`, 4 % at the
default ior 1.5, modulated by the same roughness map the metal branch reads -- so that last branch
threw away half of what the asset described. `gallery_rain`'s Alice is the case that surfaced it:
one Meshy-class material, `metallicFactor 1.0` / `roughnessFactor 1.0` with the real values in a
metallicRoughness map (mean metalness **0.28** -> dielectric, mean roughness **0.25** -> satin), and
she imported as chalk in every mode.

**The exit interface: a coated body's effective albedo (0.323.0).** The coat used to be modelled
on the way IN and not on the way out -- it reflected with probability R, and otherwise the ray
entered and the body shaded with no exit interface at all. So a clearcoat *diluted* colour (a white
specular laid over an unchanged body) where a real varnish *deepens* it.

What was missing is that light leaving a Lambertian body meets the coat from inside, and past the
critical angle (~41.8 deg at n = 1.5 -- most of a cosine-weighted hemisphere) it is thrown back
down, scatters again and retries. Summing that series in closed form turns the body's albedo into

    a_eff(lambda) = a(lambda) (1 - F_dr) / (1 - a(lambda) F_dr)

with `F_dr` the internal diffuse Fresnel reflectance of the interface (Egan-Hilgeman; 0.5967 at
n = 1.5), a constant of the index alone. Three properties make this the right shape here:

* **a_eff(1) = 1 exactly**, so a white body under a lossless coat conserves energy exactly;
* it is **nonlinear in a**, so a saturated body deepens -- the low channel makes more passes through
  the pigment. This is the varnish effect, and it is what the old model could not produce;
* it is a property of the **material**, not of a direction, so it needs no BSDF interface change and
  works in every mode on both backends for free.

Implementation: `Scene::finalizeLayeredCoats` mints a per-stack **copy** of each body material and
sets `Material::coatFdr` on the copy (so a body shared with an uncoated object is unaffected, and
re-running it is idempotent). The transform is applied in the albedo funnels -- `diffuseReflectance`
and `reflectSlot`, and `dReflectSlot` on the device -- *after* texture, record, pattern and
vertex-colour, so a textured body gets it as surely as a constant one. Entry transmission is already
carried by the coat/body selection probability, so there is no double count.

Measured in a white furnace (`tools/furnace_rig.py`, camera inside, six single-quad emissive mesh
blocks). The rig is validated first -- mirror 1.0004, diffuse 1.0 -> 1.0003, diffuse 0.5 -> 0.5001 --
and `F` is measured from a black-body-under-coat case (0.0401) rather than fitted:

| coated body | rendered | predicted `F + (1-F) a_eff` | err |
|---|---:|---:|---:|
| white 1.0 | 0.9998 | 1.0000 | **-0.02 %** |
| grey 0.5 | 0.3159 | 0.3159 | **-0.02 %** |

and with the coat's own lobe suppressed (`reflectance manual specular 0.0`) so only `a_eff` acts, a
deep red body deepens from **R/G 9.01 to 21.76** (2.41x) and darkens to 0.540x. A coloured body's
per-CHANNEL prediction does *not* hold to 0.02 %, and correctly so: `a_eff` is nonlinear and is
applied per WAVELENGTH, which does not commute with integrating against the CIE curves. The flat
white and grey cases are the ones where per-channel arithmetic is exact.

**Absorption inside the layer: a tinted lacquer (0.324.0).** The coat had an exit interface but
was still perfectly clear, so `layered` could not be a tinted varnish or candy paint. `coat { absorb
<spectrum> depth <metres> }` adds Beer-Lambert through the layer, and it *composes with* the series
above rather than sitting beside it: light that fails to escape crosses the absorbing layer twice
more before it gets another try. The geometry enters as two cosine-weighted mean secants, both
closed form in the index alone --

    escape cone (theta < theta_c), what gets in and out:  2 n^2 (1 - cos theta_c)  = 1.1459 at n=1.5
    beyond it   (theta > theta_c), one trapped leg:       2 / cos theta_c          = 2.6833

-- so with `T = exp(-sigma_a d s_esc)` per leg and `T_rt = exp(-2 sigma_a d s_tir)` per round trip,

    a_eff = a T^2 (1 - F_dr) / (1 - a F_dr T_rt)

which reduces EXACTLY to the clear case at sigma_a = 0. The trapped light travels more than twice as
far as the escaping light, because the light beyond the critical angle is precisely the grazing
light; that is what makes a tinted coat's internal series bite so much harder than a single crossing
suggests. Both path lengths are folded into constants at scene-build time, so shading pays two
`exp`s and only when the coat actually absorbs.

Validated in the same furnace, white body (so the rendered colour IS the coat), mode D:

| coat | rendered | expected |
|---|---:|---|
| no `absorb`, no `depth` | 0.9999 | the clear result, 0.9998 |
| `absorb 2000` with no `depth` | 0.9999 | clear -- both keywords are required |
| `depth 1e-4` with no `absorb` | 0.9999 | clear |
| `absorb 2000  depth 1e-4` | **0.3475** | **0.3476** analytic |
| `absorb rgb 400 2600 5200  depth 1e-4` | R 0.556 G 0.293 B 0.155 | an amber lacquer |

CPU and GPU agree to 0.1 % on the absorbing case.

**One funnel, and the defect that proved it was needed.** Both host albedo sites
(`diffuseReflectance`, `reflectSlot`) and both device sites (`dDiffuseRho`, `dReflectSlot`) now go
through a single `coatedAlbedoAt` / `dCoatedAlbedoAt`. 0.323.0 did *not*: it applied the transform
in `dReflectSlot` on the device but not in host `reflectSlot`, so a **glossy** body under a coat
rendered differently on the two backends for exactly one version. Unifying the funnel for A2 is what
surfaced it; the two now agree to 0.003 % on that case.

**The approximation A1 makes:** entry is exact and angular (the coat/body split already uses
R(theta_i) per hit) while the exit is directionally AVERAGED into `F_dr`. **A2 averages both legs**,
which is a smaller concession than it sounds: Snell compresses the entire incident hemisphere into
the escape cone, so the entry secant only ranges over 1.0 .. 1/cos(theta_c) = 1.342 -- the angular
part of coat absorption is at most a 34 % swing in path length, while the tint is the whole visual
point, and the strong silhouette cue of a coat (R(theta) rising to 1 at grazing) is already exact.
Still absent: Snell refraction into a directional body (A3). The EXPLICIT
multi-bounce version remains deliberately deferred: see known-issues, "the EXPLICIT multi-bounce
layered BSDF", which records the four conditions that would make it worth building and how to test
that they hold.

**Note on the rig, because it cost two wrong measurements.** A furnace built from `light area {}`
is **blind to specular** -- those emitters have no hittable surface, so a mirror in the box reads
exactly 0.0000 -- and a furnace built as one emissive mesh is silently flipped outward and reads
black. Both are recorded in known-issues; `tools/furnace_rig.py` is the version that works, and it
carries its three ground-truth controls (mirror, diffuse 1.0, diffuse 0.5) so the rig proves itself
before any result is read.

**A real coat, once the device could render one (0.317.0).** The import is `layered`: a Fresnel
interface of index `ior` over the diffuse body, sharing the roughness and normal maps. That gives
both the 4 % normal-incidence reflectance and the **angular ramp** toward grazing that the
constant-weight `mix` of 0.316.0 could not express. `-import-specular mix` still builds that mix,
because **modes `D` / `J` / `U` refuse layered materials outright** -- a mode gate in
`sceneModeUnsupported`, unrelated to the device -- and a scene whose stills are rendered in mode D
needs a form those modes accept.

**Why it could not be `layered` before.** `MatType::Layered` had an enum slot on the device
(`D_LAYERED`) and no branch, so `cudaForwardSupported`'s `unsupported()` rejected any scene
containing one and the whole render fell back to the CPU tracer -- a mode-M flyby would have left
the GPU without a word. See the port below.

**The preview.** `raster.h`'s pass 1 collapses a mix to its *dominant* child, which for
specular-over-diffuse is the body -- so the lobe was baked away before it could be drawn. Pass 1.5
takes the first glossy child's lobe with `f0` scaled by its selection weight, on top of the dominant
child's colour, and only when pass 1 found no lobe of its own. `PShade` is the host bake both
rasterizers upload from, so the CUDA preview gets it too.

### `MatType::Layered` on the GPU (0.317.0)

The coat is a set of FIELDS on the parent material, and the device has no way to shade fields --
which is why the port is not a dozen new branches but one idea: **at upload time, give every
layered material a synthetic GLOSSY child** (`uploadScene`, appended past the scene's own
materials; it inherits the parent's roughness/normal/pattern plumbing, is white because a
dielectric coat's reflection carries no tint, and drops the body lobes and any emission). A layered
material then resolves exactly like a mix: `dResolveCompound` returns the coat child with
probability R -- `dLayeredCoatReflectance`, the device twin of `render.h`'s, covering Fresnel,
thin-film and manual coats -- and otherwise picks a body lobe with `dMixResolveChild`, leftover
absorption included. The eleven sites that resolved `D_MIX` (shadeStep and its hero twin,
`bkRadiance` x2, the RGB gather, `dRandomWalk`, the two photon gathers, SPPM, and the two VCM
subpath walks) each became `dIsCompound(...)` plus a call; `dResolveCompoundDominant` mirrors
backward.h's Whitted rule (the coat only when it dominates). Nothing indexes the device material
array by host id past `scene.mats.size()`, and `dEmitterForMat` scans emitters rather than a
per-material table, so the appended entries are invisible to everything that does not ask for them.

**The CPU gather had to be fixed to match.** `photonGather` / `photonGatherSub` resolved a layered
material with `mixPickChild(...)  // approximate: gather the body lobe` -- the camera ray always
entered the stack, so mode M rendered every clearcoat matte while modes R and W showed its sheen.
Once the device did it properly the two backends disagreed outright (`scraps/lay_probe.ftsl`: a 0.6
lacquer coat peaked at 0.254 on the GPU against 0.005 on the CPU), so both now take the coat with
probability R, as the BDPT walk always has.

### Object view navigates differently from scene view (0.319.0)

`ftrace model.glb` and `-explore scene.ftsl` shared one navigation model — a fly camera — which is
right for walking a scene and wrong for inspecting a model, where the gesture you want is to turn
the thing round. They are now distinct, and the split is free of conflict because **the fly camera
never used a mouse button**: its "mouse-look" is HOVER steering (the cursor's dead-zoned offset from
centre sets a turn RATE). So a drag was unclaimed input, and in object view it means *rotate the
object* while throttle and steering keep working exactly as before.

**Rotating the object is implemented as orbiting the camera** about the scene's bounding-sphere
centre, with `fwd` re-aimed at that centre each drag so the model stays framed. The two are
indistinguishable on screen and the orbit costs nothing — no vertex rewrite, no transform node, no
BVH rebuild. Pitch is clamped shy of the poles by the same rule the fly camera's pitch uses, so the
turntable cannot tip over and invert the horizon.

**The window reports the gesture; the render loop decides what it means.** `NavInput` gains
`dragDx` / `dragDy` (pixels since the last drain, an accumulator — unlike `lookX/lookY`, which are
persistent state) and `dragging`. `livewindow.cpp` captures the mouse on button-down so a drag that
leaves the window keeps working until the button comes up, releases on `WM_CAPTURECHANGED` as well
as button-up, and forces the hover-look rate to zero while a button is held so one gesture cannot
both orbit and steer. `main.cpp` applies the orbit only when `positionalMesh` — a bare mesh was
opened directly — and ignores the drag otherwise; a scene viewer therefore behaves exactly as it
did, except that holding a button now parks the cursor without steering.

### SPECGATHER — mode M's camera walk carries its spectral factors as a spectrum (0.321.0)

Mode M's walk is monochromatic at the camera sample's wavelength; its photon map is polychromatic.
The density estimate always evaluated the gather point's BRDF per photon at `ph.lambda`, but the
reflectances the ray collected *on the way there* were folded into a scalar `thr` at the camera's
wavelength and then multiplied a photon sum spanning every wavelength. Exact for a flat spectrum,
wrong in either direction for a coloured one (measured up to 14 %).

`SpecThr` (`photonmap_render.h`) and `DSpecThr` / `dSpecFold` (`render_cuda.cu`) carry the same
product over a 24-bin grid beside the scalar; the estimate reweights each photon by
`v[bin(ph.lambda)] / cam`. The denominator `cam` is accumulated from **the very scalars that went
into `thr`**, not re-sampled from the grid — that is what makes `thr * ratio` collapse exactly to
(everything else) x v[lambda_p]. Folding a factor probes three wavelengths first and treats an
agreeing one as constant, so uncoloured materials cost three lookups rather than 24 and flat scenes
are unchanged. Cost measured at +4 % on a gallery_rain frame.

**Curves of curves (0.326.0).** `ftsl.h addCurve` is now recursive (`flattenCurveNode`): a
`curve` node's children are `point`s (a strand, unchanged) or `curve`s (inline or by name), and a
node of curves flattens to instances placed along the Catmull-Rom through its children's roots
with the shape blended point-wise on root-relative offsets — both through the camera's
`catmullRomAt`, placement through `splineArcParams` (the `camera_curve` two-pass arc-length /
density inversion, factored out; `camera_curve` itself is untouched). Siblings are resampled to a
common point count by arc length (`resampleStrandCR`); every child must yield the same strand
count. Named curves register flattened in `Builder::curveByName_` whether or not they render —
a material-less named curve is a definition — which is what `fur … guides` (0.3) reads. The
strand tessellator gained the same knot rule (`tessellateCurve(..., alpha)`, Barry-Goldman when
`alpha > 0`, bit-identical at 0) so `spline centripetal` means one thing on a flight path, a
strand and a groom. `-dumpcurves` writes the emitted polylines; `tools/curve_rig.py` checks the
rule exactly against them.

**Guided grooms (0.327.0).** `fur { guides "name" … }` resolves named curves from
`curveByName_` in the deferred fur sweep (complete by then), resamples each strand to `points`
and stores it root-relative (`FurSpec::Guide`). `furBuildStrandGuided` (fur.h) replaces the
closed-form builder when guides exist: k nearest guide roots by linear scan (guides are tens to
hundreds; a grid would cost more than it saves), inverse-square weights with an optional
Gaussian `guide_falloff`, the blended offsets laid from the strand's own root, then the
existing jitter / curl on top and clump / bald downstream unchanged. Random draws are taken in
the closed-form builder's order so a seed is the same hair either way. Roots are untouched —
guides are a shape field over area-uniform roots, never a root distribution.

**Alice's hair, end to end (0.327.0-0.328.1).** Four files make a groom for one imported doll,
and each is regenerable: `tools/alice_scalp.py` segments the hair mass out of the single fused
Meshy mesh (colour and roughness both fail to separate hair from face — measured — so it is a
nose-tip face ellipsoid, a throat box, and the largest *welded* component; Meshy duplicates every
vertex along every UV seam) and writes both the whole mass and a **scalp cap** (the mass within
16 cm of the skull centre, above the nape — a doll roots its hair on the skull, and rooting over
the whole mass reads as a fuzzy dome). `tools/alice_guides.py` traces **streamlines** down the
sculpted surface from rings of roots, lifted off the surface along the normal (a blend of
surface-hugging guides on a convex head lies *inside* it), with a side part and a short swept
fringe, and writes them as curves of curves; `--place` wraps them in a `group` for a scene that
shows her transformed. `scenes/alice_hair.ftsl` is the groom at true size (10 in); `scenes/
alice_hair_gallery.ftsl` the same groom carried into gallery_rain's placement with the fur's
metres at the hall's 3x. Both tools honour the GLB's node transform, so `tools/glb_rescale.py`
could put the doll at her real size without touching the raw vertices the segmentation was
tuned on. Fiber numbers are a rooted doll's: 15 000 strands, 0.05 mm radius, plug-sized locks.

**The media term joined it in 0.322.0** (`Renderer::mediumTransmittanceSpec`,
`dMedTransmittanceSpec`). A transmittance is a stochastic estimate, so it cannot use the ratio
trick (`E[A/B] != E[A]/E[B]`); the walk carries a per-wavelength vector instead, in three tiers —
**flat in sigma_t** (one scalar walk repeated: exact, free, and what most media are),
**homogeneous** (`exp(-sigma_t(lambda) * len)`: analytic, exact), and **heterogeneous** (ratio
tracking with a majorant bounding every wavelength and one weight update per wavelength, so the
estimates stay correlated). Measured: a coloured fog went from a 113 % channel spread against
mode D to 1.0 %, and gallery_rain is unaffected in both image and cost (96 s).

Covered: `Mirror`, `Glossy`, `Filter`, the `ThinFilm`/`Multilayer`/`Grating` arm, coloured-glass
absorption, and media transmittance. Not covered, with reasons, in known-issues: media transmittance (a stochastic estimator,
not a function), the `Hair` BCSDF (cost), and the binary stochastic choices in `HalfMirror` and the
layered coat.

### Layered materials in BDPT and VCM -- modes `D`, `J`, `U` (0.318.0)

Those modes refused layered materials outright (`bdptUnsupportedFeature` -> "layered materials").
What they needed was what the device needed: a **material** to shade with, since their walks
resolve each vertex to one and then evaluate `bsdfF` / `bsdfPdf` against it. So the coat is now a
material -- `Material::coatChild`, built once by `Scene::finalizeLayeredCoats()` (an uncoloured
glossy lobe carrying the coat's roughness and maps; idempotent, because `build()` runs more than
once on some paths) -- and the three walk sites (`bdpt.h randomWalk`, `vcm.h` x2) resolve a
layered stack exactly as they already resolve a mix: coat with probability R, else a body lobe.
The device dropped its own upload-time synthesis and points `coatChild` at the host's, so there
is now ONE definition of a coat shared by BDPT/VCM, mode M's gather and the device.

**The MIS convention is the mix's, deliberately.** A resolved vertex stores the CHILD's pdf, not
the mixture's -- the selection is the sampling. That is what `Mix` has always done here, and
matching it keeps one rule rather than two.

**Validated by identity, not by eyeball.** A manual coat at `specular 1.0` always wins its draw,
so that stack must render *exactly* like a plain glossy of the same roughness; at `specular 0.0`
it must render exactly like the bare body. Holding the sphere and its position fixed and swapping
only the material (`scraps/ident_*.ftsl`, 512 spp):

| identity | mode D | mode M |
|---|---:|---:|
| `layered(coat=1.0)` / `glossy` | **1.0001** | **1.0076** |
| `layered(coat=0.0)` / `diffuse` | **0.9999** | **1.0002** |

The first version of this test compared four spheres side by side and failed everywhere, because
a small off-centre key light does not light four positions equally -- the geometry has to be held
fixed and only the material varied. The second version compared the coat against
`reflect rgb 1 1 1` and read 1.29 in mode M, which turned out to be a property of the WHITE, not
of the coat: see the open entry on `rgb 1 1 1` vs `1.0` in known-issues.

### A directional environment for the preview's specular (0.317.0)

The split sum's environment half was `PreviewLight::ambient`, one scalar -- and a constant
environment is invisible in a reflection: a 4 % coat reflecting exactly the ambient it is already
lit by adds nothing the eye can find, which is why an imported dielectric read as chalk in
`-explore` next to the viewer it came from (those reflect an HDRI). `deriveLight` now also fills
`envUp` / `envDn` -- neutral in hue, 1.65x and 0.35x ambient, so the pair's **mean is `ambient`**
and what changes is directionality, not exposure -- and both rasterizers look the gradient up along
the reflection of the view about the shading normal. `raster_cuda.cu` carries the twin, since
`-explore` uses the GPU rasterizer when one is available.

**One more thing was needed before any of it showed on Alice.** Her coat's roughness lives in a
map, and the material's CONSTANT roughness was the glTF *factor* -- `1.0`, the Meshy house style,
with the real 0.25 in the map. Every consumer that cannot sample the texture read that constant and
drew a lobe so broad it was invisible. The importer now harvests the map's **mean** as the constant,
exactly as it already harvested mean metalness. On the preview that is the difference between 2,217
pixels changed at +5 levels and **43,677 at +109**.

**A trap worth knowing.** `-import-specular` is read in the **pre-scan** argument loop
(`main.cpp` ~18770), not the main one: the scene is loaded at ~18907, well before the main loop
runs, so a flag that changes how an asset is *imported* has no effect if it is parsed there. The
first version of this flag was, and the A/B silently compared two identical renders.

### The beam budget — sizing the map from the scene's own knee (0.242.0)

`-n` alone was a cliff, not a tuning wart: at the inherited forward-mode default of `-n 2e6`,
`_fog_cornell` at 128² deposited so many beams that a 120 s budget produced **no image at
all**. The fix is a budget, and the measurement that shaped it is the interesting part.

**The obvious rule is wrong.** A beam map looks like a cache — built once, paid back once per
camera sample — so its worthwhile size ought to scale with `res · resY · spp`, exactly as
`buildBeamMap`'s split length does. Sweeping `-n` at a fixed 120 s budget against a converged
mode-`D` reference says otherwise: 18 708 / 75 288 / 300 033 raw beams give 15 / 15 / 4 spp at
0.426 / **0.410** / 0.887 relative RMSE at 128², and 4 / 4 / 1 spp at **0.805** / 0.822 / 1.888
at 256². Four times the pixels, **same knee**. Only the build is amortised over the frame; the
gather is per-sample and grows with the map, so the two scale together and cancel.

**What the knee actually is: the `-beamk` floor releasing.** While `probeK` at the raw
`blur × mfp` radii is below the floor (32), `buildAuto` inflates the radii to hold the
*gathered* count at 32 — so extra beams cost the gather nothing and buy a tighter, less blurred
kernel. Past that point the floor lets go and every further beam is gathered and paid for. It is
a property of the **scene**, and it moves a long way: `_fog_thick` (σ_t 20, bounded, mfp 0.48 m)
is already past the floor at 16 428 beams, where `_fog_cornell` needs ~300 000 — **23× at the
same resolution**.

So mode `J` measures it per scene, in a pilot it runs and then throws away:

- **`BeamBudgetReq` → `traceLightBeamPass`.** The budget is spent by lowering `nPaths`, never by
  thinning afterwards: the map is the pass's only product, and a uniform post-hoc thin is equal
  only *in expectation* to tracing fewer subpaths anyway (keeping fraction `keep` and
  normalising by `nEmitted · keep` is what `nEmitted = nPaths · keep` already means).
- **One pilot, two measurements**: beams per subpath (converting a beam count into a subpath
  count) and `probeK0` at the real radii (locating the knee, via its near-linearity in the beam
  count — 3.98× beams gave 4.09× `probeK0` on `_fog_thick`). The budget sits **at** the knee —
  the smallest map that gets the requested kernel width without the `-beamk` floor widening it.
- **The pilot is discarded, and that is the point.** Reusing its beams would make `nPaths` a
  function of the map's own contents and introduce an O(pilot/nPaths) bias. Mode `J`'s whole
  claim is that its *absolute* radiance is right (gate 3). Run under an independent salt and
  thrown away, the pilot leaves the map unbiased for every fixed `nPaths`, hence unbiased over
  the pilot's choice of it.
- **Precedence:** explicit **`-n` disables the budget** (the more specific knob wins);
  otherwise `-beamcount` is a resource ceiling and the enforced budget is `min(ceiling, knee)`.
- **`buildBeamMap` now names the knee** when `probeK0` has passed `-beamk`, in the same voice as
  the existing `-beamsplitmax` hint — for modes `M` and `J` alike.

**Measured landing** (`scraps/_jauto.sh`, same 120 s and reference as the sweep, `-n` absent):
the two scenes' knees differ by **8.7×** (`_fog_cornell` ~113 995 beams, `_fog_thick` ~13 122),
and both runs report `knee-bound`. On `_fog_thick` the budget **beats every hand-swept point**
(rel. RMSE **1.942** vs a best of 2.741 — 29 % better) by choosing a map *smaller* than the
sweep's cheapest: 13 285 beams with `probeK` exactly at the floor. On `_fog_cornell` it comes
out above the sweep's best (0.587 at 7 spp vs 0.410 at 15), because that scene's equal-time
optimum is near 0.66× its knee. Normalised by the knee the optimum spans only **0.66×–1.0×**
across the two scenes — a 1.5× residual from a 23× raw spread, which is the case for the knee
as normaliser.

**Why it aims at the knee and not below it (0.243.0).** It first shipped aiming at *half* the
knee. That was a bias, and the equal-time comparisons above could not see it — nor could any of
the four gates, since all four pin `-n` and pinning `-n` disables the budget. Gate 3 re-run with
`-n` dropped came out **6.4 % off the absolute level**; at the knee that falls to **0.34 %**.
The cause is that **below the knee `buildAuto` widens the radii** to hold the gathered count at
the `-beamk` floor, and a wider kernel bleeds energy from bright regions into dim ones.
Undershooting is not free, though — traversal cost (not the pinned returned count) is where
per-segment time goes, so a smaller map genuinely buys spp. The trade is declined because the
sides differ in kind: noise washes out with render time, a silently widened kernel does not,
and it means `-beamblur` no longer says what the render did. The knee is the smallest map whose
radii are the requested `blur × mfp` un-inflated. Caveat: when the gather is a small share of
per-sample cost (`_slab_ss` at `-max-bounce 1`), quality keeps improving past the knee;
`-beamcount` / `-n` are the escape hatch.

The escape clamp `kBeamFarScale` was **deliberately left alone**: on `_fog_cornell` its 8 × 0.866
≈ 6.9 m is only 2.8 mfp, so a principled mfp-based clamp would make beams *longer*. Tightening
it for speed would be a silent accuracy regression buying a cost the budget removes properly.

### Gate (2) — a second MIS weight, and a flag that breaks it on purpose (0.217.0)

`misWeight` is PBRT's **relative** form: it never builds a path density, only the ratios
`p_j/p_s`, telescoped one vertex at a time. That is where all the index arithmetic lives, and
an off-by-one there yields a plausible weight rather than a crash — so it wants an audit. The
obvious one ("sum every strategy's weight, assert 1") **cannot fail**: within a call the result
is `r_c/(1+Σr)` by construction, and across calls you are summing weights of *different* paths.

`bdpt::misWeightReference` is the audit that can. It computes each strategy's density
**outright** and divides, sharing no arithmetic with the ratio loops: build the unified
light-to-camera path `x[0..n-1]`, give each vertex both `pl[i]` (density if walked from the
light) and `pc[i]` (if from the camera) — `pdfFwd`/`pdfRev` with the roles **swapped on the eye
half**, since "forward" there means camera-to-light — then
`p_j = Π_{i<j} pl[i] · Π_{i≥j} pc[i]`. The products overflow doubles in both directions over a
dozen area densities, so it sums in logs via prefix/suffix arrays normalised by `p_s`. It runs
*inside* `misWeight`, with the `ScopedAssign`s still installed, so both forms read identical
densities: the densities are not under test, the combination arithmetic is.

**`-misaudit` reports; `-misaudit-poison` is the negative control.** A cross-check that has only
ever agreed is equally consistent with "both are right" and "the check is vacuous" — a hook that
never fires, a mistyped tolerance. So the poison flag injects exactly the bug the gate exists to
catch (it omits the eye-half swap) and the run **must** disagree; if it doesn't, ftrace prints
`FAILED: the poisoned reference agreed anyway ... its clean runs prove nothing`. It ships rather
than being reverted after one use, because Phase 3b will want it again the moment merge
strategies enter the weight — which is when a vacuous audit would cost the most.

| Run (`_fog_cornell`, `-device cpu`) | Checked | Disagreed | Worst rel. diff |
|---|---|---|---|
| `-mode D -r 96 -spp 2 -misaudit` | 276 198 | **0** | 2.753e-14 (`s=9,t=1`) |
| `-mode J -r 64 -spp 1 -n 50000 -misaudit` | 62 086 | **0** | 5.177e-15 (`s=7,t=2`) |
| `-mode D -r 96 -spp 2 -misaudit-poison` | 276 198 | **213 566** | 1.000e+00 (`s=6,t=3`) |

The clean residual peaks on the *longest* subpath in each run, which is where accumulated
rounding should peak. The poisoned run checks the same 276 198 weights — evidence the hook is
unchanged and only the reference moved — and its render is `cmp`-identical to the clean one,
since the reference weight is observed and never used. Gate (2) currently certifies the
**connection half only**; that is the point of running it before Phase 3b, so that a later
disagreement can only be the new merge code.

### Mode `J` Phase 3b — the merge weights (0.218.0)

Phase 3b is UPBP proper: both techniques are now weighted, the double count is gone, and a
mode-`J` image is a *correct* estimate of the same integral mode `D` estimates.

**The one quantity everything is built from.** For a merged path, the merge technique competes
against exactly one BDPT strategy — call it **C1** — the one that would have connected the same
light vertex `y_{s-1}` to the same medium vertex `x`, with `x` as the *last camera* vertex. Their
density ratio is

```
eta_i  =  p_merge,i / p_C1  =  n_m · 2r · sin(theta_i) · p_L(x_i) / ( sigma_t(x_i) · Tr(e_i) )
```

with `e_i` the **camera-side** edge. Two things fall out of that and shape the whole
implementation. First, the *connection* edge's transmittance cancels **exactly**, so mode `D`'s
"omit `Tr` from the recorded densities" convention needs no change and BDPT-MIS-TR stays
orthogonal (see `known-issues.md`). Second, the kernel factor is **`2r`, not `K1(d_perp)`** — for
the same reason VCM uses `1/(pi r^2)` and not the kernel value: the density being compared is that
of a *hypothetical* merge, which by construction sits at `d_perp = 0`. `1/(2r)` is the mean of
the 1D Epanechnikov kernel over its support. `n_m · 2r` is factored out as the scalar `mergeKappa`.

**Only three densities are merge-specific.** The merged path reuses `randomWalk`'s recorded
`pdfFwd`/`pdfRev` for every vertex except `pdfRev(y_{s-1})`, the pair at `x`, and
`pdfRev(eye[k])` — the same three `misWeight` already patches with `ScopedAssign`. Everything
else is already right, because **the beam leaves `y_{s-1}` along exactly the direction the walk
continued on**: sliding the next vertex along that same ray changes no neighbour's density, given
phase-function symmetry and BSDF reciprocity.

**Where the weight is computed — three places, because the path is never assembled.** A beam hit
knows its beam and its camera segment but has no access to either subpath's vertex array, and
materialising one per hit would defeat the estimator. The denominator is therefore *pre-summed
from both ends* and multiplied together at the hit:

| Piece | Where | What it holds |
|---|---|---|
| `BeamMis` (per beam) | `traceLightBeamPass`, during the deposit | `sumC`, `sumM` — the light-side connection and merge sums telescoped down to `y_{s-1}`; plus `pdfDir`, `rCoef`, `gateC1`, `etaPrev`, `leadIn` |
| `segSumC` / `segSumM` (per eye vertex) | `BdptRenderer::renderRows`, once per camera subpath | the same two sums telescoped up from the camera |
| `BeamMergeWeight::operator()` | inside `gatherPhotonBeamsW`, per hit | joins them: `w_M = etaS / den` |

`BeamMis` deliberately is **not** a member of `BeamBank`: mode `M`'s photon pass shares that type
and must not pay 40 bytes a beam for something it never reads. It is a parallel array kept in
lockstep, which is safe only because mode `J`'s pass runs with `cap = 0` so `BeamBank::halve()`
never fires; `BeamMap::misIdx` maps post-split beams back to their pre-split entry, and if the two
arrays ever disagree in length the MIS data is **dropped** rather than mis-indexed — the gather
then falls back to weight 1, i.e. mode `M`'s over-bright estimator, a failure the image shows
rather than hides.

**`trDet` — deterministic transmittance, for weights only** (`beamgather.h`).
`Renderer::mediaTransmittance` is an unbiased *estimator*: two calls on one segment of a
heterogeneous medium return two different numbers. That is right inside a contribution and wrong
inside a weight — a balance heuristic sums to 1 only if each edge's transmittance is the *same*
number everywhere it appears. So the weight needs a deterministic **function**. Note what that
does not demand: accuracy. `Tr == 1` everywhere would also be unbiased, merely a poor weight. So
`trDet` is exact for a homogeneous medium (closed-form exponential) and a fixed 4-point midpoint
quadrature for a heterogeneous one.

**One flag gates both halves.** `mergeKappa` is threaded through `connectBDPT` → `misWeight` →
`misWeightReference`, so the connection weights gain their merge terms in exactly the runs where
merges are actually gathered. `mergeKappa == 0` deletes every merge term and makes both weight
functions byte-identical to mode `D`'s — which is what keeps gate 1 alive.

Three approximations are deliberate and documented under **UPBP-W** in `known-issues.md`: one
scene-wide kernel radius (`BeamMap::radRef`), a weight built from two wavelengths, and a merge
dropped when the light-side density is delta. All three are weight *quality*, not bias.

**The depth cap (0.219.0), which is not optional.** The merged path has `s = j+1` light vertices
and `t = k+2` camera ones, so its depth is `j + k + 1` — up to `2·maxDepth + 2`. The connection
loop refuses `depth > maxDepth`; until 0.219.0 the merge loop did not, so mode `J` rendered path
lengths mode `D` never builds, adding energy with no competing technique to weight it against.
That is invisible in a thin fog and worth **1.65×** at `sigma_t 20 / albedo 0.95`, and it *grew*
with `-n`, because an uncontested technique's balance weight tends to 1 as `kappa` grows. The fix
is a single gate — `BeamMis::vert` carries `j`, `BeamMergeWeight` carries `k` and `maxDepth`, and
the weight returns 0 past the cap. One gate suffices because every strategy in that denominator
describes the *same* path with the same vertex count, so they are all in or all out together;
`misWeight`'s hypothetical merge terms need no gating for the identical reason. See **UPBP-DEPTH**
in `known-issues.md` for the measurements.

**Validation, passed 0.218.0** (`_fog_cornell.ftsl`, `-device cpu`):

| Test | What it proves | Result |
|---|---|---|
| `-mode J -nobeams` vs `-mode D -device cpu`, 96², spp 2 | gate 1 survives the weights | `cmp` **identical** |
| `-mode J -r 64 -spp 1 -n 50000 -misaudit` | gate 2 now covers the **merge** terms too | 62 086 checked, **0** disagreed, worst 7.903e-15 (`s=9,t=1`) |
| `-mode J … -misaudit-poison` | the audit is not vacuous with merges present | 62 086 checked, **47 941** disagreed, worst 1.000e+00 |
| `-mode J` (43 spp, `-n 50000`) vs `-mode D` (512 spp), `-hdr` | the double count is gone | `J/D` mean **1.0006** (was **2.0374** in Phase 2) |

The last row is the one Phase 3b exists for. Phase 2's ≈2× was the two estimators each claiming
the whole volume term; a correct weighting returns it to 1, and unlike Phase 2 a *dimmer* pixel
is now expected and correct — the connections are legitimately down-weighted wherever a merge
could have produced the same path (53.3 % of pixels here). The ratio walked
`1.0611 → 1.0035 → 0.9897 → 0.9811 → 0.9985 → 1.0006` as mode `J` accumulated samples, i.e. it
oscillates about 1 rather than settling off it.

**Read that convergence correctly: `-spp` does not converge the merge half.** The beam map used to
be built **once**, before the first pixel, so every sample of every pixel merged against the *same*
50 000 light subpaths. More `-spp` therefore reduced only the camera-side variance; the light-side
sample set was frozen, so the merge estimator's residual scaled as `1/sqrt(n_m)` and the only knob
that moved it was **`-n`**.

**That stopped being true in 0.247.0, and calling it "a property, not a defect" was the mistake.**
Measured with `-seed` (see UPBP-THICK in known-issues.md), the frozen light side is not a small
residual: on `_fog_thick.ftsl` at `-n 256` it is a **31.6 % whole-frame standard deviation** across
independent realizations, and it is a *floor* — no amount of render time touches it — which the
image hides by going visually smooth at 0.6 % pixel noise while sitting up to 63 % off. A
converged-*looking* wrong answer is worse than a visibly noisy one. So mode `J` now **refreshes the
light side**: the render is split into epochs, the subpaths are re-traced under a fresh salt between
them (`RngSaltScope`, `src/rng.h`), and the epochs' films are averaged, which makes the merge noise
fall as `1/sqrt(epochs)` alongside the connection noise. Same `-n`, same 60 s: s.d.
**31.6 % → 2.4 %**, worst-case error **62.9 % → 4.0 %**, for ~10 % fewer samples. It is affordable
only because the *gather* dominates mode `J` (UPBP-CONV) — the whole light side is ~2.6 % of a
render. `-beamfreeze` restores the old single-map behaviour bit-for-bit; `-beamrefresh <frac>`
(default 0.10) sets the rebuild share of the wall clock, and the epoch length adapts to the
*measured* per-epoch overhead, so a heavy scene — where `renderBdptCuda` re-uploads the whole scene
every epoch — stretches its epochs out and degrades gracefully toward `-beamfreeze` instead of
thrashing.

**MODE `J`'S SURFACE LIGHT SIDE RUNS ON THE DEVICE, PER CHUNK (0.272.0).** `-jsurf`'s surface
photon map is traced by `kJSurfLightT` — the device twin of the deposit in `traceLightBeamPass` —
and gridded on the device with the machinery `vcmSessionPass` already used (bbox reduce, cell
key, stable sort, `lower_bound`). `buildJSurfMapDevice` is called from inside `launch()`, which
`gpuSppChunks` invokes once per chunk, and the chunk is further split four ways, so a render
draws a fresh light-side realization every sub-launch instead of once per refresh epoch.

**The reason is frequency, not speed, and that distinction decides the design.** The light pass
is ~4 % of a mode-`J` render, so moving it buys 4 % of the wall clock and nothing else. What it
buys is *realizations*: on the host a realization costs 27 ms, so mode `J` managed ~126 in 90 s
against mode `U`'s ~8500, and that gap — not the estimator — was most of `U`'s measured
advantage. On the device the rebuild is free (a media frame fits 88 of them with `host gaps
0.0 s`), so the limit moves to the chunk, and splitting the chunk moves it again. Measured:
**11.2x less whole-frame variance at equal wall clock** on a surfaces-only scene, bias unmoved.

**AND THE BEAM TREE IS BUILT ON THE DEVICE TOO (0.272.3).** `kBeamMorton` / `kLbvhInternal` /
`kLbvhLeaves` / `kLbvhRefit` are a Karras LBVH emitted into the host's own `DNode` layout —
internal `[0..n-2]`, leaves `[n-1..2n-2]` with `count == 1`, which is exactly
`BvhNode::isLeaf() == count > 0` — so `dGatherPhotonBeams` traverses either builder's output
without knowing which ran. `BeamMap::build` gained `skipBvh` and `BeamMap::worldBounds` carries
what `bvh.nodes[0].box` used to supply, since after the skip there is no root node to read.

The measurement that justifies it: a host light-side realization is **72 % BVH build** (0.22 s of
0.32 s on `_fog_thick`), so the tree — not the trace — was what capped the realization count.
4.3 ms against 220 ms makes a realization 2.7x cheaper and buys 1.85x more of them at unchanged
samples, for 0.719x the variance against a predicted 0.735x. The host tree remains the only one a
CPU gather can use, so mode `M`, `-device cpu` and `-loadmap` all keep it; `-jhostlight` restores
it for mode `J` as well.

**Two boundaries are part of the design rather than defects.** The split has an **interior
optimum** (k=16 is worse than k=4) because more realizations cut the merge half's variance while
costing camera samples that raise the connection half's — so the knob is a trade, not a dial to
turn up. And the whole thing does nothing on a scene with **media**, where beam merges dominate
and the beam map is still host-traced per epoch; the device *beam* deposit is the remaining
piece. `-jhostlight` restores the host pass.

**…and `-beamfreeze` is what makes a mode-`J` FLYBY share one light side (0.271.0).** Mode `J`
stays in the per-camera `restIdx` loop — its connection half is BDPT, whose frames are independent
by construction — so it is not promoted to a shared group the way `A`/`B`/`M` are. Instead the
three objects the light pass produces (`BeamMap`, `bdpt::SurfMap` and the `BeamBudgetInfo` the
budget pilot settled on) move into a `JLightCache` owned by that loop, which `runRender` takes by
pointer and mode `J` binds its locals to **as references**; a matching key skips
`buildLightSide(0)` outright. The key is `(scene, -n, res, resY, spp, maxDepth, threads,
diffraction, wantBeams, wantSurf, surfRadius)` — everything a second camera could differ in, since
the rest of the pass's inputs are command-line globals fixed for the process. Two properties are
worth stating because they are what make the change safe rather than merely plausible:

* **It is bit-identical, not "acceptably correlated".** The light pass seeds per *absolute subpath
  index* under a fixed estimator salt (`bdpt.h`), epoch 0's `RngSaltScope` is the identity
  (`rng.h`), and the pilot has its own fixed seed — so every camera of a frozen flyby was already
  building the *same* map and merely paying to rebuild it. Verified with `cmp` on
  `scenes/_jfly.ftsl` at two map sizes; 3 cameras went 86.2 s → 58.0 s.
* **It is gated on `-beamfreeze` for a real reason.** Without the freeze the light side is redrawn
  between epochs, and which realization a camera *ends* on depends on how many epochs its own
  budget fitted — so there is no single map to share, and hoisting one would silently hand frame 2
  whatever frame 1 happened to stop on. Sharing a *refreshing* light side means transposing the
  loops (one map, advance every camera's epoch, rebuild), which also changes each frame's
  write/checkpoint cadence; that half of PERF is still open.

**Mode `M` has the same disease, and got the same cure in 0.252.0** (M-FROZEN in known-issues.md).
A single-camera mode-`M` render builds its photon map, caustic map and beam map once and then
spends the whole budget growing camera spp, so the light-side half of its error is a floor — and
one that gets *more* visible with time, because the camera grain that was masking it falls away
and leaves the frozen pattern standing as apparent structure. On `gallery_rain` that reads as
saturated **coloured bars** through the rain and cloud: `phase rainbow` is genuinely
wavelength-dependent, so a beam that scatters in the rain must retire its spectral bundle and
deposit at one wavelength (50.5 % of that scene's rain chords, 42.3 % of its power, and there is
no way to make them white), and a beam is a **line**, so a single-λ deposit is a streak down a
whole chord rather than a speck of grain. Un-freezing is therefore the only available fix, and it
works because the streaks then land somewhere new every realization. The mode-`M` block reuses
mode `J`'s epoch loop verbatim, with two additions: its **gather radii are pinned to epoch 0's
choice** (photon mapping is biased at a finite radius, so re-adapting per epoch would make each
epoch a *different* estimator and their average a mixture rather than a variance reduction), and
refreshes are **silent** since the map's shape does not change, only its realization. It is even
cheaper here than in mode `J` — the deposit and all three builds are ~0.35 % of a 900 s render.
The **multi-camera / flyby** path is deliberately excluded: there, building the map once and
reusing it across frames *is* the feature.

**Mode `M` reaches that light side down two routes, and 0.253.0 taught the second one to refresh
as well.** Which route a mode-`M` render takes is decided by `plainRender` — no `-time`, `-noise`,
`-forever` or `-preview` — and a *plain* fixed-`-spp` render is grouped into the **shared**
photon-map path even when there is only one camera, because that path is where the GPU gather
lives. So through 0.252.0 `-mode M -beams -time 900` refreshed while `-mode M -beams -spp 400`
stayed frozen. The shared path now runs the same epoch loop for a lone camera, and a lone camera
that would *not* actually get the device gather (no CUDA, `-device cpu`, a lens camera, an
unsupported scene) is folded back to `runRender`'s branch instead, since staying would have bought
a CPU gather off a frozen map rather than the same CPU gather off a refreshed one. `-savemap` /
`-loadmap` pin a render to the shared path and switch the refresh off: they are implemented only
there, and a *loaded* map is a stored realization with nothing left to re-trace. On the device the
salt does both jobs at once — the gather seed mixes `g_rngSalt`, so one `RngSaltScope` around the
whole pass decorrelates the map and the camera stream together and no `sampleBase` analogue is
needed. Two shared structs carry the epoch discipline across the host/device line: `PmRadiiPin`
(`render_cuda.h`) pins the gather radii and the caustic `kGather` to epoch 0's choice, and
`g_gpuQuietRebuild` (`render_progress.h`) is the device twin of `buildBeamMap(..., quiet)` — mode
`J` sets it too, since `renderBdptCuda` re-uploads and re-reports the beam map every epoch.

**What it costs and what it buys here — nothing on *this* scene; 3× on a thick one at the depth it needs (see the end of this paragraph).** On `_fog_cornell` mode `J` at
43 spp measures 15.25 % noise; mode `D`'s 4.42 % at 512 spp is 15.25 % when scaled by `sqrt`. The
two are *identical*, for ~25× the time. That is the expected result on this scene and not a
failure of the weights: `_fog_cornell` is a thin fog whose scattering points the camera's
free-flight sampling reaches easily, so the connections were never starved and there is nothing
for the merges to rescue. Mode `J` also inherits the beam×ray estimator's `1/sin(theta)` tail, so
its peak pixel is ~1.9× mode `D`'s on the same scene: brighter fireflies, in exchange for reaching
paths mode `D` cannot.

  **Where it does pay, measured (0.262.0).** On `_fog_thick` (`sigma_t 20, albedo 0.95`) at `-max-bounce 32` — the depth that medium actually needs, since `D`/`J` default to 8 while the unidirectional tracers default to 32, and at 8 all three modes miss 88 % of the image — mode `J` beats mode `D` at equal time by **3× on mean relative squared error** (1.85× trimmed, 1.38× at the 99th percentile) against a 600 s mode-`R` reference, while rendering 12.9× fewer samples. That is the technique's own mechanism: long multiple-scattering paths are what connections sample badly and merges capture, and the merge's cost is the beam gather, which does not grow with depth the way the connection cost's ~depth² does. At depth 8 the same comparison has mode `J` 1.4× behind — the crossover is the depth, not the estimator. See `known-issues.md` → UPBP-CONV (4)/(5).

  **The `1/sinθ` singularity is bounded now (0.262.0, `-beamsinmin`, default 0.3).** The beam×ray kernel's Jacobian denominator is `sinθ` between the camera ray and the beam, so a near-parallel merge contributes without bound — and the MIS weight does not suppress it, because such a configuration is genuinely one the connections sample badly and the balance heuristic correctly hands the merge a large weight. It is clamped where `sinθ` is *computed* (`BeamMap::hitBeam`; `dBeamHitEval` on the device), not at the estimator, because the MIS weight reads the same value — so the merge stays one function, a technique whose kernel saturates at grazing angles, rather than an estimator weighted by a pdf it no longer has. Measured on `_fog_thick` at 180 s: no resolvable bias (−0.61 % of the image mean against the unbounded estimator's own −0.57 %), worst pixel 4350× → 1690× the reference, mean relative squared error 1.219 → 0.836, and no cost in samples. `-beamsinmin 0` restores the literal estimator; past ~0.5 the bias is real, and at 1.0 the Jacobian is gone (−12 %).

**And on a thick medium (0.219.1) — the merges work, the estimator is too expensive.** Measured on
`scenes/_fog_thick.ftsl` (`sigma_t 20 / albedo 0.95`) against a converged mode-`D` reference, mode
`J`'s **per-sample** variance is **2.4–8.8× lower** than mode `D`'s, and the margin *grows* with
`-n` — which is the signature of the merges genuinely reaching paths the connections cannot, i.e.
the feature doing its job. It still loses at equal time by ~19×, purely because a sample costs
45–250× more: the auto-tuned radius has a probe ray gather **252 beams** per camera segment, and
cost scales exactly linearly with that count. The MIS weight is *not* the expense — hoisting its
per-ray work into `TrRay` bought 1.5 % — it is the beam×ray estimator's own two transmittance
marches per surviving hit, which mode `M` pays identically. The lever that remains is therefore
the **GPU port**, where mode `M`'s CUDA beam gather is the template; see UPBP-CONV in
`known-issues.md` for the tables.

### Mode `J` on the GPU (0.244.0)

**What moved and what did not.** Mode `J` is two passes with opposite cost profiles. The **light
pass** — trace light subpaths, split each medium span into sub-beams, build the beam BVH, and
precompute each beam's MIS partials — is a *one-off build* whose cost is independent of spp, so
porting it would buy a constant. The **camera pass** is paid once per sample per pixel, and it is
already mode `D`'s BDPT megakernel plus a merge loop. So only the camera pass moved: `-device gpu`
launches `kBdptT<NS, MAXD, MERGE=true>` with the finished beam map, its BVH and its MIS partials
uploaded once, while the light pass stays on the CPU on **both** backends. The banner prints both
halves (`camera pass on GPU, light pass on 12 CPU threads`) rather than naming a single backend,
because naming one would be false about half the render.

**Why `MERGE` is a template parameter and not a runtime flag.** `mergeKappa` is zero in every mode
but `J`, and that zero already deletes every merge term arithmetically — so a runtime flag would be
*correct*. It would also cost mode `D`, the far more common mode, the registers and the local-memory
footprint of the `DPathSeg` array that only merges use. Templating it means mode `D` compiles to
exactly the kernel it compiled to before the port, which is what makes "mode `J` with no beams is
mode `D` **bit-for-bit**" true on the device and not merely true in principle.

**The camera walk has to record segments.** A connection is evaluated at a *vertex*; a merge is
integrated along a whole *ray*. `dRandomWalk` therefore fills a `DPathSeg` array — origin, direction,
`tMax` to the surface that ends the ray (past any medium collision), throughput carrying **no**
free-flight factor, and `vert = n - 1` — recorded after the hit is resolved and *before* the
free-flight draw, and consuming no RNG, so a mode-`J` walk draws the identical random stream a mode-`D`
walk does. That array is the port's whole VRAM delta: the preflight reports mode `D`'s kernel at
4.36 GB local and mode `J`'s at 4.58 GB.

**The preflight has to guess which kernel will launch.** `cudaMegakernelLocalBytes` is called before
the beam map exists, and mode `J` can launch either instantiation — a media-free scene is mode `D`
exactly, hero-wavelength bundle and all, while a scene *with* a medium has the hero bundle gated off
by the medium and takes the scalar merge kernel with the `DPathSeg` array on top. Under-reporting is
the failure that matters (it is what lets a render start and then thrash on host memory over PCIe),
so it reports the **larger** of the two.

**The bug the gates caught.** Gate 1a — mode `J` == mode `D` bit-for-bit on a media-free scene —
failed on the first build, and not for any reason involving merges: mode `J` was passing `heroC = 1`
to `renderBdptCuda` where mode `D` passes `g_heroC`, so mode `D` ran the hero-wavelength bundle and
mode `J` did not. Forcing `-mode D -device gpu -heroc 1` reproduced mode `J`'s image exactly, which
is what identified it. The fix is to pass `g_heroC`; the kernel applies the same hero gate the CPU
`BdptRenderer` does, so a scene with a medium falls back to the single-λ walk on its own and the
merge kernel is still the one that launches.

**Measured (RTX 4090 vs 12 CPU threads, `_fog_thick`, 128², equal 7-minute budgets).**

| configuration | CPU spp | GPU spp | speedup | why |
|---|---|---|---|---|
| default `-beamk 32` | 1042 | 2871 | **2.8×** | a segment gathers 32 beams; the divergent BVH traversal dominates and parallelises worst |
| `-beamk 1` | 3354 | 50767 | **15×** | a segment gathers ~1 beam; the BDPT walk dominates, which is what the GPU is good at |

So the speedup is a function of how many beams a segment collects, and the beam gather — not the
MIS weight, not the walk — is the part that resists the device, exactly as the CPU profile predicted.

**Agreement, three ways.** Bit-for-bit against mode `D` on the GPU with no media *and* with media
under `-nobeams` (gates 1a/1c); **1.0029×** against the closed-form single-scatter slab where the CPU
reads 1.0030× (gate 3); and CPU-vs-GPU whole-frame energy on `_fog_thick` agreeing to **0.1 %** at
`-beamk 1`. That last number is the one that matters for a port, and it is also what proved the
thick-medium overshoot (UPBP-THICK) is *not* a port bug: both backends overshoot mode `D` by the same
1.64×, so the device faithfully reproduces a bias the CPU already had.

### Mode `J` gate 3 — an *analytic* reference, and the truncation bug it found (0.241.0)

Gates 1, 2 and 4 all compare mode `J` against its own siblings: gate 1 says it reduces to mode
`D`, gate 2 says its MIS weights sum to one, gate 4 says it converges to mode `D`'s image. None
of them can catch an error the whole family shares, and none pins the **absolute** radiance to
physics. Gate 3 does.

**The rig.** `scenes/_slab_ss.ftsl` is a homogeneous slab (`sigma_t 2.0 albedo 1.0 g 0.0`,
bounded `-1 -1 -1` → `3 1 1`) containing one small emitting sphere and *no other geometry at
all*, so the only light transport in the frame is single scattering from a point-like source.
`tools/slab_ss_ref.py` evaluates that integral by deterministic Simpson quadrature
(`NQUAD = 4096`; its own self-test drifts 9.6e-14) and compares whole-frame **energy ratios**,
`sum(image) / sum(reference)`, banded by reference quartile.

Two things make this work as physics rather than as another renderer:

- **A uniformly-emitting Lambertian sphere is *exactly* a point source of intensity Φ/(4π)** at
  every external point, so the closed form needs no small-angle approximation. Inside a medium
  the disc-averaged transmittance multiplier is distance-independent, so it lands entirely in
  the fitted global constant and cannot fake a distance-dependent error.
- **The comparator is an energy ratio, not a median per-pixel ratio.** A median is biased *low*
  on a noisy Monte Carlo image, which mimics precisely the missing-energy bug this gate exists
  to detect.

**What it caught immediately: mode `J` was 0.0227× — 44× too dark.** The diagnosis is written up
in full under **J-SCENERADIUS** in `known-issues.md`; in short, `Scene::sceneRadius` counted only
the geometry BVH, so a scene that is mostly *fog* reported the bounding sphere of its *props*
(0.0346 m here), and `Renderer::emitBeams`' escape clamp `kBeamFarScale * sceneRadius` truncated
every photon beam at 0.277 m. The merges returned exactly zero while the MIS weights still
divided the connections down by the density those merges were supposed to supply — which is why
it read as a weight bug for so long, and why `-misaudit` passing meant nothing (it cross-checks
*connection* weights only, never the gather's merge weight).

**Two diagnostics were built to separate weight from map, and are kept** (documented in
`REFERENCE.md` under the environment-variable diagnostics):

- **`FTRACE_J_HALF=connections|merges|merges-raw`** (`bdpt.h`) renders one half of the estimator
  with its MIS weights intact. The halves sum to *exactly* the full image because the RNG is
  re-seeded per (pixel, sample), so suppressing one cannot perturb the other's stream. `merges`
  was 0; `merges-raw` — weight forced to 1 — was *also* 0, which exonerated the weight outright.
- **`FTRACE_BEAM_DIAG=1`** (`photonbeams.h`) tallies the beam×ray query: BVH candidates, each of
  `closestApproach`'s four geometric rejections, the closest approach ever seen as a multiple of
  the kernel radius, and each of the gather's own drops. It reported 1.92 M candidates and 0
  geometric hits — and grep then showed all 1.92 M came from the *beam-density probe*
  (`photonbeams.h:1000`), not from the render's `gather()` (`photonbeams.h:1151`), which saw
  nothing at all. `gBeamDiag` is a namespace-scope `inline` variable, not a function-local
  static, precisely because `closestApproach` is the hottest function in mode `J` and a
  thread-safe-init guard there would tax every render that never sets the variable. Measured
  cost of leaving it in: none above this machine's ±20 % run-to-run noise, and byte-identical
  output — the numbers are in the header comment beside the struct.

**Run mode `R` on the GPU, and only on the GPU.** The CPU backward tracer collapses every
authored medium into one global *homogeneous* haze and ignores `bounds` regions entirely (it
says so on startup) — and `_slab_ss.ftsl` is nothing *but* a bounded box medium, so under
`-device cpu` mode `R` renders an unbounded fog: a different scene, not a noisier one. It then
fits the reference at **304×** the correct level with a noise rms of **51**, which reads as a
catastrophic gate-3(b) failure and is really just the wrong scene. Modes `D` and `J` honour the
bounds on either device, so the gate is `-mode R -device gpu` against `-mode D`/`-mode J`
`-device cpu`.

**Gate 3, passing** (`_slab_ss.ftsl`, mode `R` `-device gpu`, modes `D`/`J` `-device cpu`;
quadrature converged to 1e-13, 13 250× dynamic range across the frame):

| image | scale | Q1 | Q2 | Q3 | Q4 | noise rms |
|---|---|---|---|---|---|---|
| `slab_r.pfm` (mode `R`) | 4.40908e+09 | 1.0018 | 1.0005 | 0.9998 | 1.0000 | 0.0156 |
| `slab_d.pfm` (mode `D`) | 0.9997× | 1.0038 | 1.0000 | 1.0008 | 0.9998 | 0.0505 |
| `slab_j_nb.pfm` (`J -nobeams`) | 1.0006× | 1.0027 | 1.0119 | 0.9903 | 1.0011 | 1.0476 |
| `slab_j.pfm` (mode `J`) | 1.0030× | 1.0213 | 1.0153 | 0.9926 | 1.0005 | 0.2826 |

The last row is the gate: mode `J`'s absolute radiance matches closed-form single scattering to
0.3 %, with no reference to mode `D` or mode `R` anywhere in the chain. It reproduces from a
clean rebuild — an independent re-run landed mode `R` on the identical `4.40908e+09` and mode
`J` on `1.0029×`.

**Gate 4 re-run after the fix**, to show the scenes where mode `J` was *already* right were not
disturbed by it (`-r 128 -n 50000 -max-bounce 8 -device cpu`, whole-frame energy ratio against a
converged mode `D`):

| scene | mode `J` | mode `D` | `J/D` | previously |
|---|---|---|---|---|
| `_fog_cornell.ftsl` | 9 spp | 410 spp | **1.0057** | 1.0006 (43 spp) |
| `_fog_thick.ftsl` | 35 spp | 1672 spp | **1.0152** | 0.9973 |

Both are inside their own sampling error at these spp counts (the `_fog_cornell` ratio has
historically walked `1.0611 → 1.0035 → 0.9897 → 0.9811 → 0.9985 → 1.0006` while accumulating,
and `_fog_thick` is a 17 %-noise image), and neither scene has a bounded medium, so the
`sceneRadius` change cannot reach them — which is what the numbers say.

Both fog scenes needed `-n` well below the default here: at `-n 200000` and 256² the beam map
alone is 7.7 M beams and mode `J` cannot finish a single spp inside a two-minute budget. Since
0.242.0 that is what the beam budget prevents — **leave `-n` off** and the pilot sizes the map
from the scene's `-beamk` knee. Passing `-n` explicitly still turns the budget off, which is
why these historical runs reproduce. See **J-BEAMCOST** in `known-issues.md`.

## Module map (src/)

- **`main.cpp`** (~6200) — CLI parsing (the option table is a chain of `else if`s split
  into **segments** — each ends `else handled = false;` and the next is guarded by
  `if (!handled)`, because one unbroken chain hit MSVC's `C1061: blocks nested too deeply`
  at ~128 links and the build then fails on whatever flag was added last; append new flags
  to a segment, and start another once one nears ~100 links). It is also the TU that forced
  **`/bigobj`** on the whole C++ build in 0.215.0: a COFF object may hold only 65 279 sections
  and every COMDAT — each inline function, each template instantiation, each string literal —
  costs one, so a file that transitively includes essentially the whole renderer only ever
  climbs. Mode `J`'s one extra template instantiation was the straw (`error C1128: number of
  sections exceeded object file format limit`). `/bigobj` raises the ceiling to 2³² and changes
  no generated code; it is set for all C++ TUs, not just this one, because `viewer_gui.cpp`
  includes the same headers and is next in line. Also: mode dispatch,
  chunking/progressive loop
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
  `ftsl_gpda::parse`. **`include "file.ftsl"` (0.325.0)** is the one brace-less top-level
  form: a grammar rule `include_decl = 'include' STRING` (no other brace-less header exists,
  so it could not be a block), reduced to a `Block{type="include", name=path}` that
  `loadSource` **splices out again** (`expandIncludes`) *after* parsing and *before* the
  Builder runs — so the Builder never learns the type exists, `prefer` branches get it by
  recursion, and every name crosses file boundaries as if pasted. Resolution is
  beside-the-including-file first, then `assetbytes::resolve`; cycles are caught on the
  normalised absolute path and reported as the whole chain. Adding the rule meant
  regenerating `ftsl_scene.gen.cpp` with `python -m loom.grammar.emit_cpp`. loom parses the
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
  lexer's token stream is identical to a naive all-regex one. It also pins the
  **comment-marker semantics** as literal token streams — a check nothing else covers,
  because both of the above compare the lexer against *itself* and so pass happily on a
  wrong rule table.

  **Comment markers (0.248.0): `#` and `//`.** `#` is the original; `//` was added
  because it is what an author who has written any C-family language types without
  thinking, and until then it rejected the *whole file* with a message that never said
  the word "comment" (`//` lexed as a WORD, the following word became a block name, and
  the parser demanded a `{`). One grammar terminal changed —
  `COMMENT = /(?:#|\/\/)[^\n]*/` — and rule order plus longest match does the rest:
  `// foo` is 6 chars of COMMENT against 2 of WORD so COMMENT wins, while in `a//b` no
  rule *starts* at the slashes at all, so an embedded `//` cannot cut a bareword or an
  unquoted path in half. (That is a deliberate asymmetry with `#`, which *is* excluded
  from WORD's character class and does split `a#b`.) Verified by tokenizing all 683
  `.ftsl` in the tree under the old and new tables: exactly one file differs, the
  purpose-built repro. There is no `/* … */` and there will not be — NEWLINE is a
  significant token here, so a comment able to cross a line would silently delete the
  statement separators it crossed; instead `ftsl_frontend.hpp::block_comment_hint`
  appends a named diagnostic when a failing parse has a `/*` at a token start on the
  error's line, so the twin trap also has a way out.
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
  `fwd_at`/`up_at`, `aim_at`/`aim_weight_at`, `layer`, `surface`, `key`), exhaustive
  dispatch loops (group children, isosurface
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
  **The `render { … }` block** carries the handful of settings that belong to the scene
  rather than to the run — `photons`, `mode`, `res`, `device`, `out`, and (since 0.122.0)
  `max_bounce`. `applyRender` parses them onto `Loaded` (`Loaded::maxBounce`, `-1` = not
  specified) and `main.cpp` folds each into the corresponding CLI variable *only if the
  operator left it unset, so an explicit flag always wins*. `max_bounce` exists because
  path depth is sometimes a property of the geometry and not of the operator's taste:
  mode D/U run 8 path edges by default, and a thin-walled glass shell with another tube
  inside it presents about eight dielectric interfaces along one line of sight, so at the
  default the innermost surface's paths are truncated and it renders as a solid **black
  plug** — indistinguishable, by eye, from a material or winding bug. The gallery's Klein
  bottle is exactly that shape and so declares `render { max_bounce 32 }` itself; the
  scene, not the command line, is the thing that knows. Pickup is announced
  (`[scene] max bounce = N (from the scene's render block)`) so the number never applies
  invisibly. Cost is real — measured ~35% of the sample rate on the gallery — which is
  why it stays a per-scene declaration rather than a raised global default.
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
  gates all of it, so sun-free scenes are untouched. Not area-*connectible*, but since
  0.124.0 **BDPT (mode `D`) renders a Sun and a Spot anyway** — on the CPU (`bdpt.h`) and,
  since 0.126.0, on the **GPU** too (`render_cuda.cu`: `dGenLightSubpath`'s delta branch,
  `dConnectBDPT`'s unified `Wgeom`, the delta-aware `dVertexPdfLight*`/`dMisWeight`, and
  `DEscape` + `kBdptT`'s escaped-ray solar disc). **VCM (mode `U`)** followed: CPU in 0.125.0
  (`vcm.h`), GPU in 0.127.0 (`kVcmLightT`/`kVcmCameraT`); see "Delta lights in BDPT"/"in VCM"
  below. So both bidirectional modes now render a Spot and a Sun on **both** backends, with no
  CPU fallback left; Env/collimated stay outside BDPT/VCM entirely.
  The Preetham sky's `sun_disk separate` option
  (`sky::SunDisk`) unbakes the solar disc from the env map and registers an
  energy-matched Sun instead — the same picture, converging ~20× faster in forward modes.
  **Mesh area lights** (since 0.186.0): a
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
  0.118.1 fixed only the *backward* (mode R/W) sites; **0.129.0 extends the same fallback
  to BDPT (mode `D`) on the GPU**, whose s=0 strategy asked the same question through
  `dIsLightVertex` (`v.lightIdx >= 0`) and `dVertexLe` (`sc.emitters[v.lightIdx]`) and so
  still rendered these surfaces black, while `bdpt.h`'s `Vertex::Le` / `isLightVertex` —
  which read `mat->isLight` / `mat->emit(lambda)` — rendered them correctly. Mode D was
  therefore producing two different images depending on `-device`. Both device predicates
  now consult `sc.mats[v.matId].matIsLight` / `matEmit`. The MIS densities needed no
  change: `dVertexPdfLightF` already routes a `BV_SURFACE` vertex down the cosine-
  Lambertian branch, and `dVertexPdfLightOriginF` returns the same 0 the CPU's
  `vertexPdfLightOrigin` returns for a `Vertex` with a null `light` (both remapped to 1 by
  the balance heuristic). Mesh emitters still take the `lightIdx >= 0` branch, so every
  pre-existing scene is bit-identical.
  One authoring trap survives on the *quad* side of this: emission is one-sided
  (`dot(rd, ng) < 0`), and a `quad`'s `ng` is `cross(u, v)`, so swapping `u` and `v` is
  the difference between a glowing panel and a black one — while every *reflective* slot
  looks identical either way, because NEE flips the normal toward the light itself. Open
  in `known-issues.md`.
  **0.199.3 finishes the job on the two paths 0.118.1/0.129.0 did not reach — mode `M`'s
  photon gather — and fixes a separate, larger bug the same investigation turned up: a
  surface that both REFLECTS and EMITS was shaded wrong by every renderer except mode R.**
  Three independent defects, all on the same `type diffuse` + `emit` material:
  * `bdpt.h::connectBDPT` / `dConnectBDPT` opened with PBRT's *"ignore invalid connections
    related to infinite area lights"* guard transcribed as `eye[t-1].isLightVertex()`
    instead of PBRT's `type == VertexType::Light`. PBRT rejects a fictitious infinite-light
    endpoint; ours also rejected any ordinary Surface vertex carrying an `emit` slot, so
    **every `s >= 1` strategy ending on a glowing surface was discarded — all NEE and every
    light-subpath connection onto it.** Mode D therefore rendered a 5 %-albedo emissive
    plane at the ~1 % its surviving indirect happened to look like. Now tests the type,
    which for an eye vertex is never `Light` (env escape is carried by `Escape`, not by a
    vertex), so the guard is correctly inert on the camera side.
  * `photonGather` / `photonGatherSub` and their device twins added the emission and then
    `return`ed, as if an emissive material had no BSDF. They now add it and **fall through**
    to the density estimate.
  * `dPhotonGather` / `dPhotonGatherSub` keyed emission off `dEmitterForMat() >= 0` with no
    `matIsLight`/`matEmit` fallback, so on GPU a glowing quad/isosurface/CSG solid lost its
    emission entirely — while the host twins tested `m.isLight` and lost the *body* instead.
    The two mode-M paths disagreed with each other and both disagreed with R and D.
  Mode-M emission-on-hit is now also one-sided (`dot(ray.d, h.ng) < 0`) like `Vertex::Le`
  and `bkRadiance`; it used to be two-sided, so quad winding changed the image between M
  and D. Regression evidence: `scraps/mini_grid.ftsl`'s four slot-variant tiles agree across
  D/R/M-gpu/M-cpu, and `_cornell_diffuse.ftsl` holds D vs R at 1.2/255 mean abs difference
  with matching auto-exposure. Scenes authored against the old darkening need their albedo
  re-tuned — `gallery_rain`'s ground went 5 % → 1 % in the same commit.
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
  - *NORMAL ORIENTATION — the one convention every primitive and every consumer shares
    (settled 0.185.0, PBRT's `Triangle::Intersect`).* A `Hit` carries two normals and they
    mean different things:
    - **`hit.ng`** — the **geometric** normal, faceforwarded onto the hemisphere the mesh's
      authored `vn` picked (`gn = dot(tri.gn, ns) < 0 ? -tri.gn : tri.gn`), and **never**
      flipped toward the ray. That last clause is load-bearing, not stylistic: ~30 sites
      read the *side of the surface* off it — `bool entering = dot(d, h.ng) < 0` for
      refraction and medium crossing, `dot(rd, h.ng) < 0` for one-sided emitters — and a
      pre-flipped `ng` makes both unconditionally true, i.e. every dielectric runs its IOR
      ratio one way forever. Every other primitive (sphere, quad, curve, implicit) already
      obeyed this; triangles were the exception.
    - **`hit.n`** — the **shading** normal, `±ns`, with the sign chosen from
      `flipped = !(dot(r.d, gn) < 0)` — a **geometric** test. It used to be
      `!(dot(r.d, ns) < 0)`, the interpolated normal, which is the one quantity that cannot
      answer "which side did I hit": smooth shading means `ns` keeps rotating past the
      facet, so it grazes through zero in a ~1-px band at every silhouette *while the facet
      is still front-facing*. `dot(tri.gn, ns)` instead sits near ±1 on any sane mesh, so
      the faceforward is stable; its only job is a *global* winding-vs-`vn` disagreement.
    - **`hit.curv`** is signed by the same `flipped` (a bulge seen from behind is a pit), so
      the old predicate reported a sphere's bulge as a pit around its own outline.
    - Consequences that other code may rely on: `orientedGeoN(h)` — the accessor for
      "geometric normal on the *shaded* side", and **not** dead code — provably equals
      `flipped ? -gn : gn`; and `entering == !flipped`. In the silhouette band `hit.n`
      legitimately faces *away* from the ray on a front-facing hit; that is what a shading
      normal is, and `shadowTerminatorG` (Chiang 2019) is what handles it.
    - The host (`geometry.h`) and device (`render_cuda.cu`) copies must stay identical or
      one backend shades silhouettes differently from the other. Guarded by
      **`ftrace -checktrinormal`** (`checkTriNormal()` in `main.cpp`); a flat-shaded mesh is
      bit-identical by construction, since `ns == gn` makes the faceforward a no-op and the
      old and new predicates the same test. Full history, sweep numbers and the *refractive*
      mesh bug this was hiding: `known-issues.md`.
- **`curve.h`** — the **curve / fiber** primitive (hair, fur, grass, wire, thread),
  added 0.150.0 for TODO §P1. A `curve` is control points + a radius; the loader
  **flattens it at load time** into a chain of `CurveSeg` **round cones** (the convex
  hull of a sphere at either end — a capsule with a linearly varying radius), and the
  BVH indexes one leaf per *cone*, not per strand.
  - *Why flatten, and why round cones.* A triangle ribbon costs ~64 tris/hair, so one
    furred animal is 10⁸–10⁹ triangles — that is the whole reason the primitive exists.
    Flattening up front (rather than evaluating the basis per ray) buys **exact leaf
    bounds** ("bound what you test", vs. a strand box that is mostly empty), keeps basis
    evaluation out of the inner loop that §P2's sub-pixel variance will hammer, and
    leaves a POD record the GPU port uploads directly (0.151.0 does exactly that —
    `DCurveSeg` is a narrowing memcpy of `CurveSeg`). It costs memory
    (80 B/seg) and discretised curvature. Round rather than camera-facing ribbons
    because adjacent cones **share their end sphere**, making the chain watertight and
    smooth at joints with **no mitre logic** — a strand is one closed surface however
    sharply it bends.
  - *The intersector.* A round cone's boundary is three pieces — the tangent **lateral**
    cone and a **spherical cap** at each end — discriminated by one axial coordinate
    `y = dot(ba, p−p0) − r0·(r0−r1)` against the band `(0, d2)`, `d2 = |ba|² − (r0−r1)²`
    (parameterisation from Inigo Quilez). Entry into a union is the **min over pieces of
    entry into each piece**, so the code enumerates *all* roots of *all three* quadrics,
    restricts each root by its own piece's `y` range, and takes the minimum. That is what
    makes it correct for an origin **inside** the fiber (a shadow ray must still find its
    exit) as well as outside. `d2 <= 0` means one ball swallows the other and the hull is
    just the bigger sphere — a separate branch, checked against `intersectSphere`
    bit-for-bit. `makeCurveRay` hoists the one `sqrt` per ray (cf. `TriShear`), and
    `scene.h` builds it only when `curveSegs` is non-empty so a curve-free scene pays a
    predictable branch and nothing else — **verified bit-identical** to the pre-curve
    binary (modes R and W × triangles / implicits / instances, md5-compared against a
    `git worktree` build of HEAD).
  - *Conditioning: origin recentering (0.151.0).* The quadric's constant term
    `k0 = d2·m5 − m1² + 2·m1·rr·r0 − m0·r0²` is a difference of two nearly equal large
    products. At the scale a fiber is actually authored at — `r ≈ 1 mm`, segment ≈ 1 cm,
    origin 2 m away — `d2·m5` and `m1²` are each ~4·10⁻⁴ while `k0` is ~10⁻¹⁰: **six
    decades**, the entire fp32 mantissa. So `curveSegCrossings` first slides the ray
    origin along itself to its closest approach to `p0` (`shift = dot(p0−ro, rd)`) and
    every accepted root undoes the shift. This is an exact re-parameterisation — it
    changes conditioning, nothing else — and it is the round-cone analogue of the
    perpendicular-offset trick `intersectSphere` already uses (Ray Tracing Gems ch. 7).
    Measured: without it the fp32 intersector **loses 12–36 % of fiber hits** and
    misplaces the rest by **11–42 radii** (strands would render as speckled holes on the
    GPU while looking perfect on the CPU, with no image-level test able to blame the
    intersector); with it, ≤ 0.06 radii. It also tightened the double path 16× (§1 max SDF
    residual 8.60e-13 → 5.23e-14). **Trap:** after recentering the near root is normally
    *negative* (the origin now sits beside the fiber), so `tmin`/running-closest filtering
    must happen **after** undoing the shift — filtering in shifted space discards exactly
    the root you want. Both the host lambda and the device `CURVE_OFFER` macro do the
    unshift first, and both say so in a comment.
  - *Root enumeration is scalar-templated.* `curveSegCrossings<S>(ro, rd, p0, p1, r0, r1,
    shift, offer)` takes geometry as `S[3]` arrays (not `Vec3`) precisely so a `float`
    instantiation does its dot products in float — that being the thing under test. The
    renderer instantiates it at `double`; `-checkcurve` §6 instantiates the *same code* at
    `float`. Without the templating, the self-test could only have validated algebra the
    GPU does not run.
  - *Bases.* `linear` (exact, `segments` forced to 1), uniform **Catmull-Rom** (default,
    interpolating — an authored guide hair passes through its points; ends
    clamp-duplicated), **Bezier** (cubic chain, `3k+1` points), uniform cubic
    **B-spline** (approximating + C2, the shape a groom solver emits). All four are
    affine-invariant, which is why `ftsl.h` transforms the *control points* and flattens
    afterwards — a `group { rotate … }` carries a strand exactly, for free. Radius is
    interpolated **linearly** between a span's endpoint control points, never through the
    basis: a Catmull-Rom radius can overshoot, and a negative radius is not a taper.
  - *Hit record.* `u` = strand parameter 0 (root) → 1 (tip), carried on the segment as
    `u0`/`u1` so a pattern can band a fiber lengthwise; `v` = azimuth from the segment's
    own `onb(axis)`; `tangent` = the axis Gram-Schmidt'd against the shading normal.
    The v1 azimuthal frame is **per-segment, not parallel-transported**, so `v` can step
    at a sharp joint — logged in `known-issues.md`.
  - *Verification.* `-checkcurve` (`main.cpp`) is six sections: the intersector vs. the
    exact analytic SDF (sphere-traced ground truth, with rays *aimed* at the fiber —
    uniformly random rays essentially never hit something 1 mm wide, which made the first
    draft of this section vacuous at 950 hits per 200 k rays); the degenerate containment
    case vs. `intersectSphere`; `anyHit` vs. the full path with half the origins inside;
    watertightness at chain joints; basis flattening; and **fp32 conditioning** at four
    fiber scales (error measured in *fiber radii*, since an absolute tolerance is
    meaningless across scales). **Mutation-tested**: dropping the p0-cap band restriction
    fails only §3, dropping the p1 cap fails §1 and §4, and forcing `shift = 0` fails only
    §6 (on all four rows) — so the sections are complementary, not redundant.
  - *On the GPU since 0.151.0.* `DCurveSeg` + `intersectCurveSeg` in `render_cuda.cu` are
    the device twin, and `curveSegs` became the **fifth** prim range in the flat BVH
    dispatch (`tris | spheres | implicits | curveSegs | instances`) — which shifts the
    instance index arithmetic in *both* `closestHit` and `occluded`, the classic way to
    break instancing while porting something else. `cudaForwardSupported` no longer
    rejects curve scenes; it only screens their **materials**, like every other
    primitive's. Measured 74× (512 spp, 400×300, 96 000 segments: 59.1 s → 0.8 s at
    identical 4.42 % noise). The general lesson worth carrying to the next primitive: a
    device port is **not mechanical** when the megakernel is fp32 — check the
    conditioning of any quadric *at the scale the primitive is actually authored at*,
    because a well-behaved `double` intersector can be numerically useless in `float`.
    The **raster preview** was never gated —
    `raster::tessellate` stage (2b) meshes each round cone by
    sweeping rings through the same three pieces the intersector knows (back cap →
    tangent lateral band → front cap; both tangent circles sit at polar angle `acos(a)`,
    `a = (r0−r1)/|ba|`, which is what makes one angular sweep cover all three
    continuously and the preview mesh closed). Coarse by design, ~80 tris/segment.
    *Coarse was still ~46 GB* (0.160.0): a groom is 10⁶ segments and `sizeof(PTri)` is
    320 B, so the fixed 80-tris/segment cone made `-explore` on `gallery_rain`
    (1.79 M segments) allocate tens of gigabytes and page-thrash forever behind an
    unchanging `tessellating (0/33)` placeholder. Stage (2b) now runs under a **triangle
    budget** (`raster::kDefaultCurveBudget`, 12 M; `-raster-curve-budget`): it takes the
    richest LOD on the ladder `{10,2} → {6,1} → {4,1} → {3,1} → {4,0} → {3,0} → {2,0}`
    (`{azimuth divisions, rings per cap}`; `ccap == 0` drops the sub-pixel end caps,
    `cu == 2` is a double-sided ribbon) that fits, and thins whole strands by
    `CurveSeg::curveId` only if even the cheapest tube doesn't. The lesson generalises:
    **a per-primitive preview cost that is fine for the primitive is not fine for the
    generator that authors 10⁶ of them** — every "coarse by design" constant that a groom,
    an instancer or a particle system can multiply needs a budget, not a constant.
  - *A new prim range lives in **three** places, not two* (learned 0.153.1). The BVH leaf
    and the device leaf are the obvious two, and both announce a mistake loudly — the
    scene renders wrong, or falls back. The third is `Scene::closestHitLinear`, the
    brute-force reference `-checkbvh` compares the BVH *against*, and it fails in the
    opposite direction: it stayed blind to `curveSegs` for three versions, so every strand
    the BVH correctly found scored as a mismatch and `-checkbvh` reported a permanent
    ~0.2 % FAIL on any fiber scene. A stale reference doesn't weaken a cross-check, it
    **inverts** it — the self-test generates exactly the noise a real regression would
    have to be spotted in. The same commit made the reference's `O(rays × prims)` sweep
    threaded and taught its ray budget to count non-triangle prims, because a groom is the
    first thing to put 10⁶ prims through a linear scan (`fur_basics`: 68 min → 2m38s).
  - *Verified across all ten renderable modes on both devices* (0.153.1) by rendering
    `curve_basics` against a strand-stripped twin and scoring the fraction of pixels the
    fibers change; 10–38 % everywhere, CPU↔GPU mean luminance within 0.4 %. The one
    subtlety is that **A and C must be scored in HDR**: as physically-absolute
    finite-aperture cameras their image sits at ~5e-5 scene-linear, so on the tone-mapped
    PNG every pixel quantises into the bottom 8-bit level and they report 0.00 % coverage
    *even at 2.4e9 photons* — identical to what a mode that ignored curves would print.
    Harness: `scraps/curve_mode_sweep.py`.
- **`fur.h`** — the **groom generator** (`fur { on "<object>" … }`), added 0.152.0 for
  TODO §P1 stage 2. `curve.h` gave ftrace a strand; this gives it a way to *author* a
  coat, which is a different problem — nobody types 10⁴–10⁶ hairs, so without a
  generator the primitive is limited to the handful of wires you are willing to write
  out. A pure function of `(surface, parameters, seed, index) -> strands`.
  - *It emits no new geometry type.* `generateFur` appends exactly the `Curve` /
    `CurveSeg` records `ftsl.h`'s hand-authored `curve` path produces, so the BVH, the
    CPU tracer, the CUDA megakernel and the raster preview all needed **zero** new code
    and a groom inherited the 0.151.0 GPU port for free. That constraint is what kept
    the feature to one header plus a loader arm.
  - *Roots are area-uniform.* A prefix CDF over triangle area plus the sqrt barycentric
    warp — sampling a triangle *uniformly* instead is the classic mistake and is nearly
    invisible on an even mesh, yet it thins the coat over large polygons and makes
    `density` meaningless. A `sphere` target is **not** tessellated at all: roots land on
    the analytic surface with the analytic normal, so a furred ball has no faceting in
    its root distribution.
  - *Shaping is closed-form, and that is the design.* Growth is linear in the arc
    parameter `t`; droop, comb, curl and clump are all quadratic in it, so the root
    leaves the skin along its growth direction (a strand that bends at its root reads as
    broken) while the tip carries the full displacement. Closed-form is what makes each
    strand independent of every other, hence the build is a lock-free `ft::parallelFor`
    into a preallocated slice whose result does not depend on scheduling. No simulation,
    no solver, no collision — a deliberate first tier.
  - *Variable output, handled honestly.* `tessellateCurve` may emit fewer cones than the
    upper bound (it drops coincident samples), so each strand records its own count and a
    serial compaction closes the gaps; the "nothing dropped" case (every ordinary groom)
    degenerates to a single bulk append. If any strand *was* dropped, every `curveId` is
    renumbered rather than left stale.
  - *Clumping uses a real spatial structure.* Strands blend toward their **nearest** guide
    with weight `clump·t`, so roots stay exactly where the sampler put them — clumping
    must change a coat's **shape, never its density**. Guides live in a CSR uniform grid
    with shell-only ring expansion and a "one ring past the first hit" stop rule (the
    classic off-by-one: the nearest guide can sit beyond the first non-empty ring). The
    cell count is capped at 2·10⁶ by coarsening, so a tiny `clump_size` on a large model
    cannot allocate a billion cells. Hashing the root cell would have been simpler and is
    visibly wrong — it gives cube-shaped tufts on a grid instead of Voronoi ones.
  - *`bald` zones (0.154.0) — the fix for hair growing out of an eyeball.* A coat is grown
    per body **part**, but the features that must stay bare (an eye, a nose leather) are
    separate little spheres sitting *on* that part, and the part's groom does not know they
    exist: it roots area-uniformly over the whole target, including the ring of skin the
    eye overlaps, so every strand rooted around that ring grows straight across the eye.
    `scenes/fur_creature.ftsl` had exactly this — the eyes rendered as black patches
    peppered with hair — and the scene's *existing* comment about placing the eyes proud of
    the coat did not prevent it: standing proud stops the eye being **buried**, which is a
    different failure from the eye being **crossed**. `FurSpec::bald` is a list of spheres
    no strand may enter, culled in `generateFur`. Three choices carry the feature:
      - *The whole strand is tested, not just the root* — and `furStrandHitsBald` tests
        **spans** (point-to-segment distance), not control points, because a zone smaller
        than the gap between two control points would otherwise be skewered. A hair rooted
        outside a zone that arcs through it under droop/comb is precisely the hair that
        shows on an eyeball, so a root-only cull would defeat the parameter.
      - *After clumping*, since clumping is what drags a tip sideways into a zone its own
        root pointed clear of.
      - *Culling cannot bias the coat.* Roots are still drawn area-uniformly and survivors
        are untouched, so fur outside a zone is bit-for-bit what it was; only the count
        drops. A cull reuses the existing `nseg == 0` compaction path — there is no second
        way for a strand to disappear — with a parallel `baldFlag` byte array, allocated
        only when a zone exists, so the load line can report bald culls apart from
        degenerate strands (a `bald` that quietly ate a whole groom has to be visible, not
        inferred from a low count).
    Loader side: `bald "<sphere>" [margin]` resolves through `sphereByName_`, so the bare
    patch stays welded to the feature instead of being a second copy of its coordinates
    that rots when the face moves; `bald <x> <y> <z> <r>` covers zones that aren't authored
    spheres. Both are repeatable statements, scanned like `density_at`.
  - *LOAD-ORDER TRAP (cost a debugging session; now pinned).* The deferred `fur` sweep
    runs inside the **loader**, but `Tri::finalize()` — which computes `gn` and back-fills
    absent shading normals — is called from **`Scene::build()`**, i.e. afterwards. So the
    generator sees `gn == n0 == n1 == n2 == (0,0,0)` on every quad, triangle, and mesh
    without authored `vn`. `normalize` of that zero made the whole strand NaN, and the NaN
    was then swallowed by `tessellateCurve`'s `dot(dp,dp) > 0` coincidence guard — so a
    groom emitted **zero strands and printed no error at all**. `furSampleRoot` now
    derives the geometric normal from the vertices itself and uses shading normals only
    when genuinely present. The general lesson: **anything reading `Tri` during loading
    must assume it is not finalized.**
  - *Loader wiring (`ftsl.h`).* `triRangeByName_` maps a name → `(first, count)` into
    `Scene::tris`, filled by `addQuad`/`addTriangle`/`addMesh`. The target is resolved as
    an **index range**, never a `Tri*` held across the parameter reads — `Scene::tris` is
    a vector and a stored pointer would dangle the moment anything appended. Like
    `medium`, `fur` is collected in Pass 3 and processed in a **deferred sweep** so
    `on "name"` resolves regardless of authoring order; the sweep runs *before*
    `stripShapeOnlyMeshes`, so a groom can grow on an invisible scalp. `density` is
    divided by `L_²` because it is authored per *authored* unit², not per m².
  - *Verification.* `-checkfur` (`main.cpp`) is eight sections — roots on the surface;
    area-uniformity (a 3:1 area split must give a 3:1 strand split, mean barycentric ⅓ not
    ½, `density × area` an exact count); determinism across seeds; growth never into the
    skin plus length inside the jitter window and shaped arc inside its analytic bound;
    clumping collapsing tip spacing while moving **no** root; a well-formed chain; the
    load-order regression, built on deliberately **un-finalized** triangles; and `bald`
    zones. Note §4 measures the direction the strand *leaves* at, not where it ends up: a
    shallow strand under heavy droop legitimately curves back to the ground.
    **Mutation-tested** by `tools/mutate_fur.py` — eleven deliberate breaks in `fur.h`, each
    caught by the section that owns it (`python tools/mutate_fur.py 6 9` re-runs individual
    ones; a full sweep is one rebuild per mutation). `tools/fur_noise.py` is the companion
    measurement behind TODO §P2's corrected forward-vs-backward note.
  - *And the mutation run paid for itself immediately.* `spec.seed` has **two** independent
    consumers — the per-strand rng and the clump-guide rng — and §3's original "seed 8
    differs from seed 7" test ran with clumping **on**, where either one moving alone is
    enough to make the buffers differ. Deleting `spec.seed` from the *strand* seeding was
    therefore MISSED: the groom would silently have ignored the seed entirely in any scene
    without `clump`. §3 now checks each path in isolation — the strand rng with clumping
    off (nothing else can move), and the guide rng by driving the guide count to exactly
    **one** (`clump_size` large enough that `G = round(area/πr²)` clamps to 1) at
    `clump 1.0`, so `w = clump·t` is exactly 1 at the tip and every strand's last control
    point *is* that single guide's tip — a point the strand rng cannot influence. §3 also
    asserts the tips really did collapse, so the isolation can't silently fail open.
  - *And then it paid for itself a second time, on §8 (0.154.0).* The first §8 tested `bald`
    with a 0.12 m zone sitting **on** the ball's pole — which, being wider than the coat,
    swallows the *roots* of everything that crosses it. So mutation #10 ("test only the
    root, not the whole strand") was MISSED: root-only culling produced exactly the same
    image on that zone, while being precisely the bug the parameter exists to prevent (an
    eye is peppered by hairs rooted on the skin *around* it). §8 gained a second zone that
    **floats clear of the skin** — 50 mm above the pole, 35 mm radius, so no root is within
    reach of it — where a positive cull is only possible if the span test ran. The test
    recomputes "no root is inside" rather than asserting it in a comment, so it stays
    honest if the ball or the coat length ever change. The lesson generalises: a guard whose
    fixture makes two different implementations agree is not guarding anything, and only a
    mutation run will tell you which fixture that is.
- **`hair.h`** — the **fiber BCSDF**: Marschner's (2003) R / TT / TRT lobes in Chiang et
  al.'s (2016) energy-conserving form. Header-only, `<cmath>` + `linalg.h`, no renderer
  dependencies, so it can be unit-tested with no scene (`-checkhair`, eleven sections). 0.171.0
  added the core; **0.172.0 wired it up** as `material { type hair }` — see the
  `hair_shade.h` entry below for the Scene↔BCSDF bridge and how each renderer consumes it;
  **0.173.0 added the medulla**, the scattering core that separates fur from hair; **0.174.0
  added dual scattering** (`hair::Dual`, `-dual-scatter`), which is about a *coat* rather
  than a fiber.
  - *Why Chiang's form and not Marschner's directly.* Marschner's `M_p` is a flat Gaussian
    in θ, which integrates to **more** than 1 as roughness grows — that is exactly where the
    2003 model's famous energy gain comes from, and why practical implementations bolt on an
    ad-hoc normalisation. Chiang's `M_p` is normalised on the sphere by construction (it is
    a von Mises–Fisher-like term carrying a Bessel `I0`), and his `N_p` is a **trimmed
    logistic** rather than a wrapped Gaussian. The logistic buys two things: its CDF is
    elementary, so it inverts in closed form and gives an *exact* importance sampler; and
    trimmed to exactly one revolution it stays normalised on the circle, instead of leaking
    probability past ±π the way a wrapped Gaussian does unless you sum the wrap terms.
  - *The p ≥ 3 residual is what makes the energy balance exact.* With four lobes
    (R, TT, TRT, and one folded residual) the attenuations telescope algebraically at zero
    absorption: `f + (1−f)² + (1−f)²f + (1−f)f² ≡ 1`. That is not a curiosity, it is the
    whole verification strategy — since `f = Σ_p M_p A_p N_p / |cos θ_i|` **separates**, and
    `M_p` and `N_p` are each normalised on their own domain, the white-furnace test collapses
    from a noisy sphere integral into an identity assertable at 1e-12. It also quantifies the
    thing everyone says about the three-lobe model: dropping the residual loses 0.21 % of a
    white fiber's energy.
  - *`h` stays out of the intersector.* The impact parameter ∈ [−1, 1] is recovered at
    shading time by `hFromHit(n, tangent, wo)` from the hit normal and fiber tangent. The
    curve intersector is the hottest loop in a fur render and has no business knowing a BCSDF
    exists; §9 pins the recovery at 5e-14 and the local-frame round-trip at 8e-16.
  - *The local frame is PBRT's (+x = fiber tangent), deliberately.* Adopting the published
    convention means the published test values are **checks** rather than re-derivations —
    if the frame were rotated to taste, every number in the literature would have to be
    re-derived before it could disagree with anything, which is how a subtly wrong model gets
    shipped. (σa being scalar-per-λ here, rather than PBRT's RGB triple, is strictly simpler
    and needs no reconciliation.)
  - *A real bug the tests caught, and would not have caught as a picture.* `besselI0` was
    the usual fixed **ten-term** ascending series — fine at float precision, but 0.26 % low by
    x = 10 and 4 % low by x = 12. `I0` sits inside `M_p`'s normalisation, so that error is not
    cosmetic: it is a longitudinal lobe that does not integrate to one, i.e. energy invented
    or lost, worst at grazing angles where `cos θ_i cos θ_o / v` is largest. It showed up as
    three simultaneous failures (`∫M_p cos dθ − 1 = 5.9e-3`, the pdf's sphere integral off by
    2.8e-3, and the two `M_p` branches disagreeing by 4.4 %) which all had the same cause. The
    series now iterates to double convergence and hands x > 12 to the A&S 9.7.1 asymptotic
    (12 terms, ~1e-10 relative at the crossover and better above); the three numbers became
    2.1e-9, 9.4e-9 and 5.7e-15. The general lesson: a series truncation tuned for `float` is a
    *model* error, not a rounding error, once it sits inside a normalisation constant.
  - *Four independent uniforms, not PBRT's two-plus-demux.* The demux exists to preserve a
    2-D sampler's stratification; this renderer hands out independent uniforms anyway, so
    reusing bits of `u0` to pick the lobe would only add a correlation hazard for no benefit.
    `sample()` fills `pdfOut` and `fOut` to match the direction it returns, so a caller's
    weight is just `fOut·|cos θ_i|/pdfOut` — and §5 asserts those agree with independent
    `f()` / `pdf()` calls to *zero* relative error, which is the cheapest possible guard
    against the classic sample/evaluate drift.
  - *The medulla (0.173.0), and why it grafts on instead of replacing anything.* A fur fiber
    is a hair with a wide scattering core; Yan et al. (2017) model it as a **double
    cylinder** and add two scattered lobes TT^s / TRT^s to Marschner's three. Their decisive
    simplification is to give cortex and medulla the **same IOR**, so the interior ray does
    not refract at the core boundary: the path topology stays exactly R/TT/TRT and the
    medulla only changes what happens *along* a chord. That is why the whole feature is one
    `Chord` struct out of `refractGeom` plus two extra lobes, rather than a second model.
    With `kappa = 0` every added term is provably inert and `refractGeom` takes a branch that
    reproduces the stage-1 expression to the last bit.
  - *The scattered lobes are a DIFFERENCE, which is what keeps the furnace exact.* Yan's
    eq. 20 leaves the scattered *fraction* implicit in the normalisation of their measured
    `C^N` table. Supplying it the obvious way — `1 − exp(−σ_s·chord)` times a Fresnel guess —
    does not conserve energy, because the scattered light then pays a different toll leaving
    the fiber than the specular light it was taken from. So `ApScattered` runs the stage-1
    chain **twice**: once with the real medulla, once with the core replaced by cortex. The
    gap between the two totals *is* the energy the medulla removed, by construction. Multiply
    by the core's single-scattering albedo (keep what scattered, not what was absorbed) and
    Yan's `Tb` (the trip out), and at zero absorption the six lobes sum to
    `Σ_medullated + (1 − Σ_medullated) = 1` — an algebraic identity, asserted at 1e-12 over
    405 (κ, σ_s, g, h, θ) combinations exactly as the solid fiber is.
  - *Similarity theory stands in for `C^M` / `C^N`, and is checked against a random walk.*
    Yan's angular profiles are 24×16×16×720 Monte-Carlo tables, rank-16 tensor-decomposed to
    150 KB — and never published. Re-deriving them would mean shipping either a 600 MB
    simulation or a blob nobody can check. Instead one number does the work: after reduced
    optical depth `τ' = (1−g)σ_s·chord`, the transmitted mean cosine decays as `exp(−τ')`,
    which drives both the longitudinal variance and the azimuthal logistic width, with the
    right limits at both ends (thin core → the specular lobe; thick core → azimuthally
    uniform, which is what a diffusive core should look like). Because that is a claim about
    an actual walk rather than a fit, §S10 checks it against one: free flights along the
    chord, Henyey–Greenstein deflections composed as real 3-D rotations, mean cosine measured
    over 120 000 walks. It tracks the analytic value to 0.003 — and the same section
    re-derives the medulla half-chord by **bisection** rather than
    `sqrt(κ² − sin²γ_t)`, so a swapped radius would surface as a geometry error instead of
    hiding inside a plausible-looking spread.
  - *Measured species are data, not presets-as-code.* `hair::speciesTable()` is Yan Table 4
    verbatim — ten fitted fibers, angles left in **degrees** and σ's in 1/radius exactly as
    printed, so the table can be diffed against the paper line by line. The degree→perceptual
    conversion happens on the way out through `betaMFromDegrees` / `betaNFromDegrees`, which
    invert Chiang's fitted polynomials by their leading quadratic (the β²⁰/β²² terms exist
    only to blow up in the last few percent before β = 1 and are numerically absent below
    that; the largest measurement in the table is 18.94°, where the cubic-and-up terms
    contribute < 1e-9). The FTSL `preset` keyword feeds the row in as *defaults only*, so
    every key an author also writes still wins — `preset redfox  beta_n 0.05` needs nothing
    re-typed. Note what the table says: every animal Yan measured has κ ∈ [0.65, 0.91] and
    human hair 0.36, i.e. the medulla is not a refinement of the hair model, it is the
    difference between hair and fur.
  - *Dual scattering (0.174.0) — the coat, not the fiber.* A pale coat's appearance is
    **mostly** multiple scattering: a single blonde fiber is nearly transparent, and a head
    of blonde hair is bright because light has crossed dozens of strands. Brute force means
    100+ bounces per path, which is why white fur is the slowest thing here. Zinke et al.
    (2008) split that radiance in two — a **global** term (light arriving *through* the coat,
    attenuated by `Π ā_f` and blurred by `Σ β̄_f²` along the shadow path) and a **local** term
    (light that scattered backward out of the strands behind and came back, a closed form
    `Ā_b` with mean shift `Δ̄_b` and width `σ̄_b`). `hair::Dual` is six curves over 48
    inclination bins plus those three derived ones, built once per material/λ/σ_a.
  - *The curves are MEASURED from our own BCSDF, not from Marschner's three lobes.*
    `makeDual` importance-samples `hair::sample()` on a Halton lattice and splits the samples
    by azimuthal half (`wi.y < 0` ⇒ forward). The sampler's weight `f·cosθ/pdf` has
    expectation `Σ_p A_p` — the §S1 furnace total — so `ā_f + ā_b` **is** that total, split
    two ways. The table therefore inherits energy conservation exactly rather than
    approximating it (§S11 asserts `ā_f + ā_b = 1` to four decimals at every inclination),
    and it picks up the medulla's TT^s / TRT^s lobes for free, which a Marschner-lobe
    derivation would have had to be re-done for. `ᾱ` and `β̄` fall out of the same pass as the
    first two moments of the deviation `θ_i + θ_o`.
  - *The exact series are summed; eq. 16/17 are checked, not used — and eq. 16 has a sign
    typo.* Zinke states `Δ̄_b` and `σ̄_b` as power-series fits, because the exact forms are a
    sum over `i ≥ 1` and a triple sum over `i, j, k`. But substituting `k = j+1+q` makes the
    exponent `m = 2(i+q)` independent of `j`, so `j` is just a multiplicity of `i`, and with
    `n = i+q` the triple sum collapses to `Σ_n x^n · n(n+1)/2 · X(2n)` — one loop, converging
    geometrically. `dualSeries` computes it, so there is no reason to inherit a fit's error:
    `u = a_b²/(1−a_f²)²` is *not* small for any plausible coat (`a_f = a_b = 0.5` already
    gives 0.44). Summing the coefficients closed-form also settles a discrepancy: `Δ̄_b`'s
    coefficient of `ᾱ_b` is exactly `(1+3u)/(1+u) = 1 + 2u + O(u²)`, where eq. 16 prints
    `1 − 2u`. §S11 demonstrates it rather than asserting it — it shrinks `a_b` and shows the
    printed form's error is O(u¹·⁰⁰) while the sign-corrected one is O(u¹·⁹⁸). `dualDeltaFit`
    keeps both behind a `signFix` flag purely so the test can compare them; the renderer
    itself never calls either.
  - *`fBack`'s prefactor is derived, not copied — because the normalisations differ.* The
    paper writes `2 Ā_b g / (π cos²θ_d)`, which is Marschner's convention (`S = M N / cos²θ_d`).
    `hair.h` is Chiang's (`f = Σ_p A_p M_p N_p / |cos θ_i|`, so `∫ f cos θ_i dω = Σ_p A_p`).
    Requiring the same of the backscatter lobe — unit-area Gaussian in `dev`, density `1/π`
    over the π-wide backward azimuths, total albedo `Ā_b` — gives `Ā_b g / (π cos²θ_i)`: the
    `2` goes and the cosine is the sample's. Worth well under a percent on a real coat (the
    lobe is a few per cent of the radiance), but it removes a guarded `cos²θ_d` denominator
    that was a 10⁴ firefly multiplier waiting to happen.
- **`hair_shade.h`** — the **Scene ↔ `hair.h` bridge** (0.172.0). `hairShadeAt(scene, mat,
  hit, λ, wo)` builds a `HairShade { hair::Bcsdf b; hair::Frame fr; Vec3 woLocal; double
  radius; }` from a `Material` + `Hit`: it resolves σa (either authored directly, or inverted
  out of the authored `reflect` colour through Chiang eq. 9 — which needs the final `beta_n`,
  so the inversion has to happen here at shading time and not at parse time), builds the fiber
  frame from `hit.tangent`, and recovers the impact parameter. Two exported helpers carry the
  two things every renderer then gets wrong if left to itself:
  - *`hairFCos(hs, wi)` returns `f · cos θ_long`, not `f`.* **The fiber projection factor is
    the LONGITUDINAL cosine, not `dot(n, w)`.** A round strand has no azimuthal foreshortening:
    the BCSDF is normalised against `cos θ_long = sqrt(1 − w_x²)` in the fiber frame. Using
    `dot(n, w)` instead both double-counts the tube's curvature and breaks energy conservation,
    and — the dangerous part — it *looks* almost right: a smooth angular error that reads as
    "my hair is a bit dark at grazing angles". Returning the product means no call site can
    supply its own projection.
  - *`hairExitOffset(hs, n, w)` — TT and TRT exit the FAR side of a real solid tube.* The
    near-field model puts every exit at the entry point, but the curve primitive is a genuine
    solid round cone, so a connection or continuation leaving through the far side is occluded
    by the strand's own body. Strand radii are **microns**, so the ordinary `ng * 1e-6` nudge
    lands *inside* the tube and the ray instantly reports itself blocked by the very hair it
    left — deleting exactly the TT forward glow a pale coat is mostly made of. The offset steps
    `2.5 × Hit::fiberRadius` (a new field, set by `curve.h`) on the far side and degrades to
    the ordinary epsilon on the near side. **Every shadow ray that uses it must also shorten
    its max-t by the same amount**, or it overshoots into the light it is testing.
  - *The sampler weight is exactly `T = Σ_p A_p ≤ 1`.* Since `f·cos = Σ_p A_p M_p N_p` and
    `pdf = Σ_p (A_p/T) M_p N_p`, the ratio `fv·cosLong/pdf` is a deterministic per-hit number.
    So it is simultaneously the natural Russian-roulette survival probability (β unchanged —
    the same trick `Mirror` plays with its reflectance, except here the number is the physics
    rather than an authored albedo) and the Whitted attenuation weight.
  - *No Veach shading-normal adjoint, no shadow-terminator softening.* Both are corrections
    for using an *interpolated normal* as a projection axis; a fiber does not project about its
    normal. On curve geometry `h.n == ±h.ng` so both would evaluate to 1 anyway — the explicit
    `isFiberMat` guards are what keep `hair` honest if it is ever put on a smooth-shaded
    triangle mesh.
  - *Unidirectional tracers de-hero at a strand; bidirectional ones do not.* σa is per-λ (and
    an authored `reflect` is inverted into one per-λ), so a hero packet cannot share one fiber
    interaction across C wavelengths — forward (`render.h`) and backward (`backward.h`) put
    `Hair` in their dispersive/de-hero group beside `Fluorescent`. BDPT/VCM by contrast already
    re-evaluate each secondary's `f` along the hero's sampled direction (`secF[i]`), so the
    bundle survives a strand there — which is fortunate, since a fiber is precisely where the
    spectral spread is interesting.
  - *BDPT/VCM get the projection through a PRE-DIVIDE, not a special case.* Those integrators
    speak only "f, its pdf, and a geometry term carrying cos(ns, w)", and that term is formed
    in roughly a dozen places. Rather than teach all of them about fibers, `bdpt::bsdfF`
    returns `hairFCos(wi) / |cos(ns, wi)|` for `Hair`, pre-dividing by the cosine BDPT is about
    to multiply back in — so `f·G` comes out to exactly `hairFCos × cosOther/dist²` and every
    MIS ratio, strategy weight and density conversion stays untouched and provably consistent.
    A `hairCosGuard` floor of 1e-7 keeps the division finite at a grazing endpoint.
  - *...and that pre-divide is exactly what makes `f` NON-RECIPROCAL, so a particle vertex needs
    the ADJOINT (0.276.0).* The convention divides by `cos(ns, wi)`, the **newly sampled**
    direction — self-consistent on a continuation, where the direction evaluated is the direction
    sampled, but not at a *connection*. `Glossy` is the clearest case: its lobe factor is
    symmetric under swapping `wo`/`wi` (`dot(wi, reflect(-wo,ns)) == dot(wo, reflect(-wi,ns))`,
    algebraically) while the denominator is not, so one physical path picked up
    `r·lobe/cos(camera-side)` when a light subpath built it and `r·lobe/cos(light-side)` when a
    camera subpath did, and MIS combined two estimators that disagreed by `cos(wo)/cos(wcam)`.
    Measured on mode `D` as **15 % too dim** with the camera near the normal and too bright with
    it grazing — a sign flip. `bsdfFAdjoint(m, ns, wo, wi) = bsdfF(m, ns, wi, wo)` (device twin
    `dBsdfFAdjoint`) is Veach's particle-tracing rule for the BSDF itself; `shadingAdjointCorr`
    was already the shading-normal half of it, and the two are **independent** — the latter is 1
    on flat geometry, where this is still required. **The rule for new code: every site that needs
    `shadingAdjointCorr` needs `bsdfFAdjoint` too, and no other site does.** That criterion is
    greppable, which is how the scope was bounded to 12 call lines across `bdpt.h`, `vcm.h` and
    `render_cuda.cu` rather than guessed at. Which direction is correct is settled by energy, not
    preference: under uniform illumination `∫ r·lobe/cos(w_incident) · cos(w_incident) dω = r`,
    whereas dividing by the camera-side cosine does not integrate to `r` — and a white-furnace
    test measures mode `R`, which builds paths camera-side, flat to **0.04 %**. Leave `bsdfPdf`
    alone: the lobe factor is symmetric, so the densities were always reciprocal.
  - *Modes M / S scatter but never gather.* `struct Photon { Vec3 n; float power; float
    lambda; }` carries **no incident direction**, so a directional BCSDF has nothing to
    evaluate against at a density-estimate gather. `sppm_render.h` and `photonmap_render.h`
    therefore give `Hair` an explicit *scattering* case — without it, it would fall into their
    `default:`, which treats an unknown material as a **mirror**, and would be silently wrong.
    A strand is not a visible point and not a deposit site, the same treatment those modes
    already give glossy/specular surfaces. Logged in `known-issues.md`.
  - *Dual scattering rides the NEE connection, and terminates the path (0.174.0).* The
    coat-level half of Zinke lives here: `hairDualFor()` caches a `hair::Dual` per
    `(material, λ bin, σ_a bin)` (λ to 10 nm, σ_a sqrt-spaced over 64 levels) behind a mutex,
    built *outside* the lock so racing threads each build one and the loser's copy is
    dropped; `hairDualResponse()` walks the shadow ray with `Scene::walkFibers` and
    accumulates `T_f = Π ā_f(θ)` and `σ̄_f² = Σ β̄_f(θ)²` over the strands it crosses, using
    each crossed fiber's OWN table (a two-toned coat attenuates correctly) — or, under
    `-dual-grid`, gets the same two quantities from one DDA march of the fiber-density grid
    via `hairShadowGrid()` (see `fur_grid.h` below); and
    `hairDualFCos()` evaluates Zinke's Figure-5 combination. `n = 0` crossings means directly
    lit, and the shading point gets `f_s + d_b f_back`; otherwise it gets
    `T_f d_f (f_s + d_b f_back)` evaluated at a direction drawn from the forward spread.
    Four things are worth naming:
    - *It replaces the estimator inside `emitterGeom` and `envGeom`, not the emitter loop.*
      `neeLight`'s four emitter branches (sun / cone-sampled / spot / area) all funnel through
      one `response` lambda, so dual scattering is one extra first branch there plus a
      `blocked` override — every non-dual path stays bit-identical, and there is no fourth
      copy of the emitter-sampling code to drift. `envGeom` gets the same substitution, with
      `wMis = 1`: the path ends here, so there is no BCSDF-sampled partner to weight against.
      Leaving the sky on single scattering would have lit a coat's sunlit side with the full
      model and its sky-lit side with a bare fiber.
    - *`hair::f` must be paired with ITS OWN cosine — this was a 13.4× error.* The shading
      integral is `∫ f(ω_o,ω_i) L_in(ω_i) cos θ_i dω_i`, so the projection belongs to the
      direction the light actually arrives from, which under `n > 0` is the spread draw and
      not the light. And `hair::f` is in Chiang's normalisation, ending in `/= |cos θ_i|`, so
      pairing it with its own cosine cancels exactly. The first version took `cos θ` from the
      light direction; after a dozen crossings `σ̄_f` exceeds a radian, so the Gaussian draw
      lands near the fiber axis constantly, `|cos θ_i|` hits its clamp, and `f` comes back
      amplified with no cosine to undo it. Measured on `scenes/fur_species.ftsl`: 13.4× too
      bright overall, 67× at the 99th percentile. Fixed, the same scene lands at 0.82× of its
      reference with the firefly tail gone (max 3.5 vs the reference's 10.1).
    - *The N^G table is replaced by a Monte-Carlo draw.* The paper precomputes a 2-D
      azimuthally-convolved `N^G` and widens each lobe's variance to account for light
      arriving *spread* rather than from a point. A path tracer does not need that: draw ONE
      direction from the spread distribution per connection (`hairSpreadDir` — a Gaussian of
      width `σ̄_f` in θ about the light, uniform over the forward azimuthal half) and let the
      pixel average do the integral. Unbiased with respect to the approximation, and no
      second table. It also means the spread is *spent*: `f_back`'s Gaussian is NOT widened
      by `σ̄_f²` the way eq. 10 does, because `dev` already carries that blur — Zinke widens
      analytically only because his incident direction is still the light's.
    - *It TERMINATES the path at a fiber vertex, deliberately.* The analytic terms already
      carry the coat's multiple scattering; continuing the path would double-count it. So
      `-dual-scatter` makes fur one-bounce while everything else in the scene keeps full path
      tracing. This is the one place in the renderer where a flag buys speed with bias, and
      it is documented as such — the reference is `-mode R` without it. What the coat gives
      up is *indirect* illumination: lights and the environment both go through the model,
      but light that bounced off the room first never reaches the fur. Measured on the fur
      alone against each scene's own 200-bounce reference: a pale coat lit only by an area
      light lands at 0.77× at the paper's `d = 0.7` and 0.99× at `d = 0.9` (single scattering
      alone: 0.17×) for 2.1× the speed; the same coat under a constant sky 0.92× for 2.4×;
      but `scenes/fur_species.ftsl` — an absorbing medullated coat in a white room — only
      0.57×, *and* 1.6× slower than its reference, because a dark coat's brute-force paths
      die in a couple of bounces while the fiber walk still cannot early-out. That case is
      logged in `known-issues.md`; the fix Zinke gives for it is §4.1.2's voxel density grid
      instead of BVH ray shooting.
  - *CUDA runs hair natively (0.181.0).* `hair.h` is ported wholesale to `__device__`
    doubles as `render_cuda.cu`'s `dhair` namespace (`D_HAIR = (int)MatType::Hair`, still
    appended at the **end** of the enum because the `D_*` tags are `(int)m.type`); the device
    `Hit` carries `fiberRadius` stamped by the curve intersector (the tangent already rode in
    `tan`). Forward: `interactHair` — connect-then-scatter like Fluorescent (narrow-but-finite
    lobes are camera-visible, so it splats to every mode-A/B camera via `splatSurfaceAllHair`
    before sampling), RR on the fiber throughput `fv·cosLong/pdf`, far-side TT/TRT exits
    offset by `dHairExitOffset`; the hero tracer de-heroes onto it (σ_a is per-λ). Backward:
    `bkInteractHair` + NEE with `rho = 1` where `bkEmitterGeom`/`bkNeeLight`/`bkEnvGeom`/
    `bkNeeEnv` take an optional `DHairShade*` and swap the surface cosine for the full-sphere
    `π·hairFCos` fiber response. Both bodies are `__noinline__` so the fat double-precision
    frames (DHairShade + `dhair::sample` locals) stay out of the shared tracer frames and out
    of every kernel's statically-computed MIN_STACK. One porting trap is load-bearing:
    `hair.h`'s `besselI0 ↔ logBesselI0` mutual recursion (runtime-safe, statically a cycle)
    made nvlink mark every kernel reaching them "stack size cannot be statically determined",
    which strips MIN_STACK and drops those kernels to the 1 KB `cudaLimitStackSize` default —
    every GPU render then died of stack overflow, hair or not. The device pair is therefore
    split into acyclic cores (`besselI0Series` / `logBesselI0Asym`) with branch-picking
    wrappers, bit-identical per branch. Validated on `hair_basics`: CPU/GPU channel means
    agree ≤0.5 %; mode R 19.7× faster, mode B 6.4× (RTX 4090). Still CPU-only: `-dual-scatter`
    hair (host-side approximation — the gate lives in `main.cpp`'s `backwardOnGpuOk`), and
    hair scenes in the BDPT / photon-map GPU backends (their vertex/gather machinery shades
    non-specular vertices as Lambertian; they reject via `sceneUsesHairMaterial`, and the
    VCM / SPPM backends inherit the reject by chaining those gates).
- **`Scene::walkFibers`** (`scene.h`, 0.174.0) — the shadow-ray traversal dual scattering
  needs and nothing else has: call back for every `Hair` strand crossed, and report `false`
  if anything opaque is in the way.
  - *One traversal, not one per strand — and this is the difference between the feature
    being worth having and not.* The obvious implementation is `closestHit`, step past the
    strand by `2.5 × fiberRadius` (the `hairExitOffset` clearance, so a micron-radius tube
    cannot re-report itself), repeat. That was the first version, and it made `-dual-scatter`
    **2.8× slower** than the brute-force path tracing it exists to replace: a coat crosses
    dozens of strands and each crossing paid a full root-down BVH descent. Riding
    `bvh.traverseAny` instead visits every primitive overlapping the segment exactly once
    (`primIdx` is a permutation, so nothing is reported twice) for one descent total.
  - *Why unordered traversal is legal here.* `T_f` is a product and `σ̄_f²` a sum — both
    commutative — and an opaque blocker anywhere along the segment kills the connection
    regardless of where it sits relative to the strands. So the any-hit callback returns
    "stop" only for a blocker, and the fiber crossings accumulate as a side effect. The cost
    of that trade is that nothing excludes a self-hit any more, so the caller must start the
    ray already clear of its own strand (which `hairDualResponse` does anyway).
  - *The crossing count is bounded, and overflow fails BRIGHT.* `maxCrossings`
    (`-dual-max-cross`, default 64) caps how many strands are accumulated; past it the ray
    is still treated as reaching the light with what it has. A dense coat can exceed any
    bound, and stopping *dark* there would paint a hard black silhouette exactly where the
    coat is thickest — the most visible possible failure.
- **`fur_grid.h`** (0.175.0) — the **fiber-density + orientation grid**, Zinke's §4.1.2
  aggregate, behind `-dual-grid`. It replaces `walkFibers` as the *counting* half of the
  global term, and it exists because that walk is the one thing in dual scattering that
  cannot early-out (see the `known-issues.md` entry it closes: `fur_species` 84.8 s → 57.4 s,
  from *slower* than its own path-traced reference to 1.06× faster).
  - *The two aggregates.* A fiber of radius `r`, length `ℓ`, unit tangent `t̂` presents
    cross-section `2rℓ√(1 − (d·t̂)²)` to a ray along `d`. Exact directional extinction would
    need every strand; pulling the root outside the sum (Jensen) collapses the whole cell onto
    two quantities that *aggregate*: a scalar `c = (2/V)Σ rᵢℓᵢ` and a normalised symmetric
    `T = Σ rᵢℓᵢ t̂ᵢt̂ᵢᵀ / Σ rᵢℓᵢ`, giving `σ_t(d) ≈ c√(1 − dᵀTd)`. `FurCell` is exactly 32 B
    (7 floats + a material id), so 128³ is 64 MB.
  - *The load-bearing identity: `∫σ_t dt` **is** the expected number of fiber crossings.* A ray
    of length `L` through volume `V` (cross-section `A = V/L`) hits fiber *i* with probability
    `2rᵢℓᵢ sinθᵢ / A`, so the expected count over `L` is `L·σ_t(d)` — exactly the walk's `n`,
    with no primitive tests. And `dᵀTd` is `⟨sin²θ⟩` **in Marschner's frame directly** (that
    frame's longitudinal axis *is* the tangent, which is why `hairShadowCross` computes
    `sinθ = dot(w, t)`), so one quadratic form yields both the extinction and the inclination
    the dual tables are indexed by. One DDA march, two answers.
  - *The count is SAMPLED, not rounded — this is the whole correctness argument.* `τ` is a
    *mean*; the walk returns a random *draw*, and `hairDualFCos` is not linear in it.
    Substituting the mean breaks three things at once: `a_f^τ` instead of
    `E[a_f^N] = e^{τ(a_f−1)}` (0.107 vs 0.135 at `a_f = 0.8`, `τ = 10` — 26% too dark); the
    spread becomes one fixed width instead of a distribution of widths; and the `n == 0`
    *directly lit* branch could never fire, so a rim strand at `τ = 0.3` would always be dimmed
    by a density factor applied to light that never met a fiber. `hairShadowGrid` therefore
    draws `N ~ Poisson(τ)` by CDF inversion from a single extra uniform (`HairDualCtx::u3`),
    which makes the grid an **exact drop-in** for the walk rather than merely a fast
    approximation of it. *A draft that rounded came out closer to the path-traced reference
    than the walk did — and that was a symptom, not a success.*
  - *Visibility is a separate query, and that is the other half of why it pays.*
    `Scene::occludedSkipHair()` (`scene.h`) is an ordinary occlusion test that treats `Hair`
    curve segments as invisible but still lets grass/wire curves and all opaque geometry block.
    Unlike `walkFibers` it **can** stop at the first blocker.
  - *Build (3 passes).* Radius-inflated AABB over fiber segments only, degenerate axes padded;
    roughly-cubic axes derived from a cell **budget** rather than a resolution, so one number
    bounds memory whatever the coat's aspect ratio. Then each round cone is sub-sampled at
    0.5 × the smallest cell edge and whole `r·dl` pieces are deposited into the midpoint's cell
    — which conserves `Σ rℓ` *exactly*, asserted by `-checkfurgrid` §1. Material id is
    winner-take-all by mass margin, and the side table for that is only allocated when the scene
    actually mixes fiber materials. 187 ms over 900k segments, 423 ms over 2.48M.
  - *What it deliberately loses.* (a) The Jensen step is biased **high by at most +3.98%** —
    one-signed, so a grid never *under*-attenuates — and is *exactly zero* where a cell's fibers
    are locally parallel (`T` rank-1, the root factors out), worst at full isotropy
    (`√(2/3) = 0.8165` vs the true `⟨sinθ⟩ = π/4 = 0.7854`). (b) The tangent's **sign** is not
    in `T`, since `t̂t̂ᵀ == (−t̂)(−t̂)ᵀ`, so each cell also carries the **first** moment
    `v = Σrℓt̂ / Σrℓ` as a 12:12 octahedral direction plus an 8-bit coherence, packed into the
    4 bytes `tzz` used to occupy (`tzz = 1 − txx − tyy` because `T` is unit-trace by
    construction) — a first *and* a second moment for the same 32 bytes. `-dual-grid`'s own
    lookup still marginalises over both signs (`furAvgSigned` evaluates the table at ±θ and
    averages), which is right for it because the dual tables are near-even in θ; the aggregate
    far tier (`fur_volume.h`) uses the first moment instead, and needs to — see below.
    (c) The crossed material's tables are evaluated at the **shading point's** texture
    coordinates, where the walk had each crossed fiber's own. All three are approximations on
    top of an approximation, which is why `-dual-grid` is opt-in and the walk stays the default.
  - *The walk path is byte-for-byte unchanged.* `u3` is drawn only when `dc.grid` is set, so an
    existing `-dual-scatter` render is bit-identical rather than merely equal in expectation
    (verified with `cmp`).
  - *`-checkfurgrid`* validates the model against the **shipping curve intersector**, not
    against another closed form — see `REFERENCE.md`. Two of its five sections exist because
    earlier drafts got a wrong answer that *looked* right: capsule end caps are `πr²` of a
    stadium of area `2rℓ sinθ + πr²`, i.e. 5% for an isolated test segment (and correctly zero
    for a chained strand, where every cap but the two ends is interior to the chain); and a
    coarse grid straddling a density taper dilutes `σ_t` along exactly the rays being measured,
    by almost precisely enough to cancel the Jensen bias and certify a broken model.
- **`fur_volume.h`** — the **aggregate scattering model** that turns a `FurGrid` cell into a
  participating medium, so distant fur can be *marched* instead of intersected. Validated by
  `-checkfurvol` (six sections).
  - *The model is three steps at a collision.* `σ_t(d) = c·√(1 − dᵀTd)` gives the free flight
    (∫σ_t dt is literally Zinke's **expected number of fiber crossings**, with no primitive
    tests); a tangent is drawn from a reconstructed **orientation distribution**; and the
    existing `hair::` BCSDF is evaluated at a *virtual* hit whose normal `fiberNormalFor()`
    reconstructs from that tangent and an offset `h`. Nothing about the fiber shading model
    changes — only where its inputs come from.
  - *The ODF is a **Bingham**, `p(t̂) ∝ exp(t̂ᵀBt̂)`* — the maximum-entropy distribution on the
    sphere with a given second moment, i.e. the one that assumes nothing beyond what the grid
    stored. Two cheaper families were built and measured against explicit fiber populations
    first, and both fail a case fur actually contains: a **Watson mixture** on `T`'s
    eigenvectors turns a *girdle* (strands every which way within a plane, what a coat does
    over a whorl) into two orthogonal deltas (L1 error 0.43), and the **ACG** fixes the girdle
    but its polynomial tails smear a tight combed clump over twice the sphere it should
    (0.26). Bingham is Gaussian-tailed like Watson *and* covers girdles like the ACG, and wins
    on every population measured (0.006 / 0.080 / 0.040 / 0.023). The comparison table lives
    in the header comment so the two rejected families stay rejected.
  - *`B` is obtained from `T` by a startup table.* The moment map has a closed-form azimuthal
    integral in **exponentially scaled** modified Bessels `k_n = e⁻ˣIₙ(x)` (unscaled, both the
    exponential and `I₀` overflow well before the concentration a near-parallel cell needs),
    with a `c = w·sinh(κz)` substitution that keeps ~40 quadrature nodes inside a peak of
    half-width `1/√(2b)` however sharp it gets. The inverse is a damped Newton over the
    eigenvalue triangle sheared onto the unit square, run once and interpolated. **Its
    variables are the *gap* `log(1+b₁−b₂)` and the *floor* `log(1+b₂)`, not the two b's** —
    in the obvious parameterisation the whole `τ₃ = 0` edge (every planar cell) pins both
    coordinates at their maximum with only their difference distinguishing points, and the
    Newton stalls there, silently returning the *girdle* solution for a **parallel** cell.
    That bug cost a factor of 10 in table accuracy (1.15e-2 → 1.15e-3).
  - *Sampling is Kent, Ganeiber & Mardia (2013)*: an ACG proposal `Ω = I + 2Λ/b_k` sampled in
    closed form as `normalize(Ω^(−1/2)z)`, accepted with `e^(−t̂ᵀΛt̂)(t̂ᵀΩt̂)^(3/2)/M`.
    Acceptance is 100% at isotropy, ~90% on a girdle, never below ~50%.
  - *The tangent's sign is drawn separately, and this is not optional.* The axis part of the
    ODF is antipodally symmetric by construction, so a sampled **axis** is aligned with the
    cell's first moment `v` with probability `(1+|v|)/2` — exact for a two-delta population
    and right at both endpoints. Skipping it (drawing the sign uniformly) moved the aggregate
    response **27% even on a perfectly parallel cell**, because the ~3° cuticle tilt `α` tips
    R/TRT toward the root and so breaks `t̂ → −t̂` symmetry. Keeping the axis and the sign as
    separate steps is what leaves `T` untouched by the choice, since `t̂t̂ᵀ == (−t̂)(−t̂)ᵀ`.
  - *`-checkfurvol` §6 measures **L1 over the whole outgoing sphere**, not a worst-case
    pointwise ratio.* The first draft took the worst ratio over three random `wi` and reported
    3.59 for a combed clump — an alarming number that meant nothing, picked up in a direction
    the tight true lobe cannot reach and where *both* functions are ~0. A path tracer sees the
    integral, so the error that matters is `Σ|S_agg − S_true| / Σ S_true`.
  - *The white-furnace test that is **not** there.* Averaging `f·|cos|/pdf` over BCSDF samples
    is **vacuous** in this model and the first draft shipped it: `apPdf` normalises the very
    `A_p` that `f` sums, so the ratio is identically `Σ_p A_p` — exactly 1.0 (dev 0.00e+00)
    whether or not the tangent, the normal or the frame are right. That is a useful *fact*
    (an aggregate collision's throughput multiplier is the fiber albedo with zero variance)
    but it is not a test, and a green line that cannot fail is worse than no line.
  - *`FurVolume` — the medium itself, and why the expensive half is a side table.*
    `FurODF::fromCell` is a Jacobi eigendecomposition, a table lookup and a root find: nothing
    once per cell, ruinous once per *collision*, and a path through a dense coat collides tens
    of times. So it runs once per occupied cell at startup into **16 bytes** — two 16:16
    octahedral eigenvectors (the third is their cross product; Gram-Schmidt at decode repairs
    the ~0.002° quantisation costs) and three halves (`b₀`, `b₁`, Kent's `b_k`). `b_k` is
    *refined* at decode by three Newton steps rather than trusted, because the rejection bound
    `M` is only an upper bound when `b_k` really is the root and a half-precision one is not
    (`-checkfurvol` §7 isolates that refine: `|1/M · M_exact − 1| = 5.6e-16`).
  - *Free flight is **exact**, not delta-tracked.* Along a *fixed* ray the direction argument of
    `σ_t(d)` never changes, so `σ_t` is piecewise constant on the DDA's own cell segments and
    `∫σ_t dt = −log(1−u)` inverts by running subtraction inside the same march that computes τ.
    No majorant, no null collisions, no dependence on the density ratio between the densest and
    emptiest cell — and a coat (a thin dense skin inside a mostly empty box) is precisely the
    case delta tracking handles worst. `-checkfurvol` §8 falsifies the claim the only way it
    can be falsified: survival vs `exp(−τ)` as a binomial z-score over 268 rays × 400 draws.
- **`-fur-volume` — the coat as a medium** (P2 stage 2b, `backward.h`). Where `-dual-grid`
  keeps the strands and reads only the *shadow* off the grid, this replaces the strands
  outright: `BackwardRenderer::furVol` non-null makes `Scene::closestHit` skip every
  `MatType::Hair` curve (a new `skipHair` flag mirroring `occludedSkipHair`) and the tracer
  free-flights against `σ_t(d)` instead. Both the scalar and the hero loop; the hero loop
  de-heros at a collision, exactly as it already does at a `Hair` surface, because a fiber's
  response is wavelength-dependent through `σ_a`.
  - *Composition with fog is a **minimum**, not a special case.* The fur flight is sampled
    before the fog block and shortens `dSurf`. That is exact rather than convenient: the first
    collision in a union of independent media is the minimum of their independent free flights,
    and taking the min this way also attributes the collision to the right medium.
  - *NEE needed a third visibility path.* `scene.occluded` reports the very strands the tier is
    pretending not to have as blockers — used unchanged, the coat self-shadows to black. So
    `HairShade::aggregate` selects `occludedSkipHair` (walls only) **plus**
    `FurVolume::transmittance` as a continuous `exp(−τ)` factor folded into the response, in
    both `emitterGeom` and `envGeom`. This is the same shape as the dual-scattering
    substitution: the shadow ray becomes part of the shading rather than a separate binary test.
  - *No dual-scattering branch, on purpose.* Dual scattering is an **analytic stand-in** for
    exactly the multiple scattering this path now simulates directly; running both double-counts
    it. `-fur-volume` therefore does not end the path at a fiber the way `-dual-scatter` does.
  - *What it is and is not for.* What it buys is that cost stops scaling with **fiber count**,
    and that the coat finally has an aggregate representation a footprint-based LOD decision can
    switch to (stage 2c). What it loses is everything that lived on an individual strand: no
    silhouette, and no `u`/`v` at a collision, so a textured hair `reflect` reads at the default
    `Hit`'s coordinates — the same class of approximation as `-dual-grid`'s textured `σ_a`.
  - *It needed the hair-free BVH (below) to be a win at all.* As first written the tier was a
    3.8× *loss* at 90 k strands and its cost still scaled with fiber count, because `skipHair`
    rejected fibers at the BVH leaf rather than removing them from the tree. With
    `Scene::buildNoHairBvh` it is 1.7×/3.6×/7.0× **faster** than the strands at 90 k/300 k/900 k,
    and grows only 1.19× across that whole 100× range where the strand tier grows 4.6×.
  - *It saves memory too, and that took deleting the strands* (v0.180.0). The tier used to stop
    *traversing* a coat it could not stop *storing*, so a coat too big to load stayed too big.
    Freeing `curveSegs` after the load would not have fixed that: the peak **is** the BVH build
    (nodes + `BuildPrim` + the transient box list, ~256 B/segment on top of the segments
    themselves), so the strands have to be gone *before* `Scene::build()` runs. See "Deleting the
    summarised coat" below. Measured on the same fixed-density ladder, peak working set at
    900 k strands falls 2221 MB → **875 MB** with a byte-identical PNG, and 3 M strands
    (30 M segments), which previously died with `error: bad allocation`, now loads and renders in
    18.7 s at 2669 MB; 9 M renders at 7796 MB.
- **Deleting the summarised coat** (`Scene::dropHairCurves` / `Scene::droppedBounds`, `scene.h`;
  `ftsl::Loaded::beforeBvh`, `ftsl.h`; the prescan + hook in `main.cpp`). Once the density grid
  and the ODF table exist, a plain `-fur-volume` render has a complete summary of geometry it will
  never intersect — so it throws the geometry away rather than accelerating it.
  - *It hangs off a hook because of argv ordering.* The FTSL loader builds the BVH at the end of
    `load()`, which runs **before** main's option loop, so at build time nobody knows `-fur-volume`
    was passed. `Loaded::beforeBvh` is called at the very end of the loader — after all parsing, so
    `L.cameras` / `L.mode` / `L.defaultMode` are final — but immediately before `L.scene.build()`:
    the one moment at which a summary exists and the tree does not. A matching prescan loop in
    `main.cpp` (sharing `parseFurFlag` with the real option loop, so the two cannot drift) reads the
    fur flags and the raster-ish flags out of argv early enough to install it.
  - *The gating is the hard part, not the code.* The drop is suppressed for `-fur-lod` (its near
    tier traces the strands), `-dual-scatter` (it terminates paths *at* fibers), any raster-ish
    entry point (`raster.h` iterates `curveSegs` itself — `-raster`, `-explore`, `-fly`, `-loom`,
    `-anim`, …), any mode other than backward `R`/`W` (forward `A`/`B`/`C` trace the curves
    directly and would render a bald ball; `V` renders both), and explicitly by
    `-fur-keep-strands`. When a scene's own cameras disagree with the summary the hook says so and
    keeps the strands rather than silently rendering the wrong picture.
  - *The bounding sphere had to be preserved by hand.* `build()` derives `sceneCenter` /
    `sceneRadius` from the BVH root box, and that sphere sizes environment photon emission — so
    deleting a coat would shrink the sphere to the shaved animal and change the image.
    `dropHairCurves` accumulates the deleted extent into `Scene::droppedBounds`, which `build()`
    unions back in. The geometry is still physically there; it just isn't traced. Empty (`lo > hi`)
    and inert for every scene that drops nothing.
  - *The same sphere also unions every **bounded participating medium** (0.241.0).* A fog box is
    part of the world even though it is not a BVH primitive: light has to reach it, photon beams
    have to cross it, and a sun's emission disc has to cover it. Deriving `sceneRadius` from the
    props alone made it a property of the furniture rather than of the scene, and it
    *under*-reported — the failure direction that is hardest to see. `scenes/_slab_ss.ftsl`
    (a 4 × 2 × 2 m fog box lit by one 2 cm sphere and nothing else) reported `sceneRadius`
    0.0346 m, and `Renderer::emitBeams`' escape clamp `kBeamFarScale * sceneRadius` then truncated
    every photon beam at 0.277 m — so mode J's merges returned exactly zero while its MIS weights
    still divided the connections down by the density those merges were supposed to supply, and the
    image came out 44× too dark. `build()` now unions each bounded medium's `bmin`/`bmax` (the
    region AABB for box, sphere and implicit alike) into the same box, and `emitBeams` applies the
    escape clamp **per medium and only to unbounded ones** — a bounded medium's own `clipToBounds`
    is already finite, so the clamp could only ever cut a beam short of the region it must fill,
    and clamping up front also let one unbounded haze shorten the beams of an unrelated bounded
    cloud it happened to overlap. `dEmitBeams` in `render_cuda.cu` carries the identical change.
    See known-issues.md → J-SCENERADIUS.
  - *Compaction, not a second container.* `curveSegs` is stably compacted in place with a
    `newIndex` remap, then `shrink_to_fit()` (handing the pages back is the entire point), and the
    `Curve` records' `firstSeg` / `segCount` are rewritten through the remap. `buildNoHairBvh`
    early-returns when no hair remains, so after a drop it is automatically a no-op and
    `closestHit(skipHair)` falls through to `bvh` — which is already hair-free.
  - *It opened a GPU hole that had to be closed.* Only the CPU `BackwardRenderer` honours
    `furVol`; `renderBackwardCuda` does not. `MatType::Hair` being un-bakeable was the only thing
    keeping fur scenes off the device, so deleting the hair removed that accident. `main.cpp`'s
    `backwardOnGpuOk()` now vetoes the GPU explicitly whenever a fur volume is live, at all six
    backward-GPU decision sites.
- **The hair-free BVH** (`Scene::buildNoHairBvh` / `bvhNoHair` / `noHairPrim`, `scene.h`). A
  second acceleration structure over every primitive *except* `MatType::Hair` curve segments,
  built opt-in by `main.cpp` when `-fur-volume` or `-dual-scatter` is on, and traversed by
  `closestHit(skipHair=true)` and `occludedSkipHair`.
  - *Why a whole second tree rather than the leaf test it replaces.* Rejecting a fiber when the
    traversal reaches its leaf does not skip the traversal, and — the part that actually hurt —
    because no fiber ever survives to shorten `tMax`, the descent cannot **prune**. A coat the
    strand tier exits at the first fiber was walked end to end by the aggregate tier, which is
    why the far tier was slower than the strands *and* why its cost kept scaling with fiber
    count. The any-hit case is worse still: a shadow ray that rejects every fiber it reaches can
    never early-out inside a coat.
  - *One box list, two trees.* `collectPrimBoxes(boxes, dropHair, remap)` is the single
    definition of primitive order; the filtered tree records a `noHairPrim` map from its own leaf
    index back to the **global** prim index, so both trees decode with the same arithmetic. Two
    copies of that loop would be two chances for the trees to disagree about what prim 7 is.
  - *`isHairCurve` is the one predicate.* The tree, `closestHit`'s fallback and
    `occludedSkipHair` must agree exactly on what a fiber is — grass and wire are curves too and
    must keep blocking — so all three call it. The leaf-level test survives as the fallback for
    when no filtered tree was built.
  - *Pure optimisation, verified as one.* All six benchmark renders and a `-dual-scatter` render
    are bit-identical to the pre-change binary; `-checkfurvol` §10 runs both queries over 20 000
    rays on a scene mixing hair curves, non-hair curves, triangles and a sphere, once on the
    filtered tree and once with `noHairPrim` swapped out so the old path runs, and requires zero
    mismatches.
- **`-fur-lod` — choosing a tier** (P2 stage 2c, `backward.h`). Turns the above from a mode
  into a decision. The ruler is the width of one pixel where the coat starts, in fiber
  diameters: `Camera::footprintPerDist(1)` × `FurVolume::entryDist` ÷ `FurGrid::meanRadius()×2`.
  Strands below `d0`, medium above `d1`, stochastic smoothstep crossfade between.
  - *The ruler is the **pixel**, not the sample* — `footprintPerDist(1)`, never
    `footprintPerDist(spp)`, which is the exact opposite of what `fwPerDist` does two fields
    above it in the same struct. `fw` band-limits a sampler that cannot average over its own
    pixel, so more samples must relax it. LOD is not that: a sub-pixel silhouette cannot reach
    the final image at *any* sample count, because the reconstruction filter averages it away
    and the aggregate is precisely that average. A ruler that shrank with `-spp` would make a
    converged render switch tiers relative to its own preview — the pop the flag exists to
    prevent.
  - *The crossfade is stochastic and per path*, one coin against a smoothstep, not a weighted
    sum. A blend needs both estimators evaluated (in the band, dearer than either tier alone)
    and would still have to reconcile two incompatible visibility conventions inside one path.
    The coin is unbiased for the same blend and mode R already averages hundreds of paths per
    pixel. Smoothstep, not a ramp, so neither edge of the band is itself an edge.
  - *The choice is sticky*, carried in `GiCtx::furTier` so a gather ray and a `-herosplit`
    re-entry inherit rather than re-roll. A path that half-believed in the strands would test
    visibility against geometry its own vertices were not built from, double-counting the coat.
  - *`entryDist` uses the grid's AABB, not the first fiber* — finding the first fiber means the
    BVH traversal the far tier exists to avoid, and the two differ by at most the coat's own
    depth, far inside an octave-wide transition band.
  - *Outside the band it costs nothing, exactly.* No coin is drawn unless the footprint lies
    inside [`d0`, `d1`], so the rng stream is untouched and the endpoints are **byte-identical**
    rather than merely close. Confirmed by sweeping the threshold over one fixed coat
    (200×150, 400 spp): `-fur-lod 100:200`, `40:80` and `24:48` all md5 the same as the no-flag
    strand render, `4:8` md5s the same as plain `-fur-volume`, and only `12:24` lands in the
    band (coat mean Y 0.67620 vs 0.67641 strands / 0.67353 aggregate). The endpoints are 0.43 %
    apart, which is why it does not pop — the fade only has to hide a change in noise character.
  - `-checkfurvol` §9 checks `entryDist` **against the march** (zero optical depth may lie
    behind it) and the realised aggregate fraction against the smoothstep: monotone, and
    exactly 0 and 1 at the ends — a coat one-in-a-thousand aggregate at point-blank range is a
    coat with sparkling holes in it.
- **`ndwarp.h`** (0.220.0) — `-nd`: lift the scene's authored meshes into N dimensions,
  rotate with the `n(n-1)/2` Givens plane rotations, project orthographically back to 3-D,
  and REPLACE `Scene::tris` with the result — so the warp reaches the rasterizer, every
  render mode and the exporter through one path instead of three. The design turns on one
  fact: a zero-filled lift reads only `R`'s first three columns, so the whole thing collapses
  to `p' = M p` with `M = R[0:3,0:3]` — one 3x3 contraction, identical for every vertex,
  which is why `-nd 8` on a plain lift can do nothing a `transform` block cannot. (`M` can
  still have negative determinant, so the MIRROR image is reachable, which 3-D rotation
  cannot do.) That case takes a fast path: no welding, no complex, and the authored shading
  normals carry through the inverse transpose so the model's own creases survive exactly.
  The per-dimension FILLS are what make it non-affine — `emboss` gives dimension *k* a
  per-vertex scalar (curvature via the umbrella Laplacian, radius, height, `povSolidNoise`,
  u, v; each centred to zero mean and unit peak so the model does not drift off-origin),
  and `extrude` sweeps the mesh into a real N-D prism. A prism's boundary is a 3-manifold,
  which no triangle rasterizer can draw, so what is built is its **2-skeleton** — both lids
  plus a quad per welded EDGE, the way a tesseract is drawn as its square faces — giving
  `V→2V, E→2E+V, F→2F+E` and a `projectedTriCount` that lets `-nd-budget` refuse a
  configuration before allocating it. Non-linear fills re-derive normals through
  `mesh.h`'s own `meshFinishTris`, so a warped surface creases exactly where a loaded one
  would. Warps only `Scene::meshGroups` (native primitives and instanced BLAS assets stay
  put; emissive meshes are skipped because `addMeshLight` COPIES their triangles into the
  emitter and the two would disagree), and rewrites the group ranges afterwards since an
  extrusion changes the triangle count. Exports OBJ (per-corner normals/UVs, `usemtl` per
  material) or binary `.ftmesh`. `-checknd` pins the algebra and the combinatorics.

The N-D *field* route (`-nd` over an `isosurface` whose formula reads `d4`…) has a hazard
the mesh route does not: it runs through the pattern VM, and the VM exists three times — the
host `patternEval`, the fp64 device `dPatternEval`, and `dPatternEvalF`, the FP32 twin the
sphere-trace march uses. Adding the `d4`…`d12` opcodes to two of the three left the GPU
tracer rendering the gyroid as *nothing*: no `default:` case means an unknown opcode is
skipped rather than diagnosed, the stack ends up one short, the field reads 0 everywhere and
never crosses the isolevel. An invisible surface is indistinguishable from an empty scene,
so it cost a bisect across four backends. `-checkpatops` now closes that hole permanently:
one minimal program per opcode through all three VMs, non-zero inputs (including the
extra-dimension bank, without which the very family that broke would agree with itself at
zero and pass), arity taken from `patOpStackEffect`, and a declared exemption table for the
three opcodes that genuinely do not belong in all three. Verified by deleting the FP32
`d4` case again: it reports nine mismatches, names each opcode, and identifies the fp32 VM
as the one at fault.
- **`mesh.h`** (+ `gltf.h`, `fbx.h`/`fbx_load.cpp`) — OBJ (custom fast parser:
  single fread, in-place float/int scan), glTF/GLB subset, FBX geometry-only,
  PLY and STL (0.191.0).
  **Crease-angle auto-smoothing** (`smooth 1` on a mesh with no authored `vn`) welds
  vertices by quantized position, then gives each corner an angle-weighted average
  (Thürmer & Wüthrich) of the incident face normals, skipping any face across a
  crease sharper than `creaseAngleDeg`. Three details are load-bearing for speed
  (0.147.0 — it was ~2/3 of the whole cost of loading a mesh, and the live loom
  viewer reloads a mesh every frame): the weld map is **hashed, not ordered** (welded
  ids are opaque slots and both containers assign them in first-seen order, so the
  result is unchanged); the vertex→incident-corner index is **CSR**, one flat array
  plus offsets, rather than one `std::vector` per vertex; and each corner's interior
  angle is **precomputed once** into `ang[tri*3+c]` instead of being recomputed from
  the innermost loop, which made a vertex of degree *d* pay O(*d*²) `acos` calls per
  fan. Storing the corner index (not just the triangle) in the CSR is also what makes
  the angle a lookup — and fixed a latent bug in passing: a sliver triangle whose two
  corners weld to the *same* vertex used to contribute its first corner's angle twice
  rather than each corner's own. Measured 2.6–3.8× on the smoothing pass, bit-identical
  on 188 of the 189 `.obj` files in the tree (the 189th is the sliver fix).
  **`.ftmesh` — the binary mesh handoff** (0.147.0, `loadFtmesh`). Written by
  `loom.ftmesh` (`tools/loom/loom/ftmesh.py`), read here, and dispatched on the file
  extension at both `.ftsl` sites (`mesh` and `mesh_asset`), so a scene swaps formats
  by changing one filename and nothing else. Layout — 24-byte header
  (`"FTMESH\0\0"`, u32 version = 1, u32 flags, u32 nverts, u32 ntris), then f32
  positions, then optional f32 normals, then optional f32 UVs, then u32 indices; all
  little-endian, `HAS_NORMALS = 1`, `HAS_UVS = 2`. Every section's size is implied by
  the header, so a **truncated file is detectable** rather than being read as geometry
  made of whatever followed — which matters because the live viewer channel re-emits
  while ftrace may still be opening the previous frame's file. Out-of-range indices
  roll the partial load back (`s.tris.resize(triStart)`) and return 0 with `err` set.
  Both loaders then call the same `meshFinishTris`, so **crease smoothing cannot
  diverge between the two formats by construction**; only header/layout/index handling
  is format-specific, which is exactly what `tools/loom/tests/test_ftmesh.py` and
  `meshbench --compare` check. f32 storage is a *fidelity gain*, not a loss: the OBJ
  text it replaces went through `%.6g`, i.e. 6 significant decimal digits against
  f32's ~7.2 — measured worst vertex displacement 1.7e-08 × the mesh bbox diagonal and
  worst shading-normal tilt 0.0002–0.011°. Load is 2.4× faster end-to-end and 6.5×
  on read+decode alone, at 0.52–0.56× the file size. **What it did not fix** is the
  cost of the file itself: on this machine opening any freshly-written file larger
  than ~32 KB costs a flat ~8 ms before a byte is read (Defender), which was ~90 % of
  the viewer's per-frame `assets` term. 0.148.0 removed that from the live channel
  outright by never writing the file (see `loomlink.h`/`viewer_gui.*`) and reduced it
  everywhere else by prefetching (see `assetbytes.h`) — `known-issues.md` records what
  the gate actually is, because most of what looked obvious about it was wrong.
  **PLY and STL** (0.191.0, `loadPly`/`loadPlyBytes`, `loadStl`/`loadStlBytes`) close a
  gap where the docs claimed both formats and the dispatch implemented neither, so a
  `.ply` fell through to the OBJ parser, matched nothing, and rendered as an empty grey
  scene — see `known-issues.md`. PLY parses its header into an element/property list and
  then walks the body **generically**: unknown properties are consumed by their declared
  type rather than skipped by a hardcoded stride, which is the only way a Gaussian-splat
  export with sixty per-vertex scalars can be read at all. The same body walker is
  templated over a binary `Reader` (with byte-swap for `binary_big_endian`) and an
  `AsciiReader` (a token stream over the existing `objParseDouble`), so encoding is a
  type parameter and not three copies of the format. STL discriminates binary from ascii
  arithmetically — `84 + 50·count == filesize` — rather than by sniffing the word
  `solid`, which binary exporters put in the header often enough to make that unreliable;
  it deliberately **discards the per-facet normal**, because it is only the geometric
  normal and storing it would pin every STL to flat shading, whereas dropping it lets
  `smooth` weld by position and crease-smooth normally. Both call `meshFinishTris`, so
  the same "format cannot change shading by construction" invariant that holds for
  `.ftmesh` now holds across five formats, and **`-checkmesh`** is what asserts it: one
  cube in six encodings must produce one triangle set.
  **The structural fix, though, is the zero-triangle guard.** Both `.ftsl` dispatch
  sites used to call the loader for effect and discard the count, which is what turned
  "unsupported format" into "successfully loaded nothing" — an empty scene renders fine.
  A load that yields no triangles is now a hard scene error, so the *next* unhandled
  extension is a message rather than a grey image. **`describeOpenFailure`**
  (0.191.1, `assetbytes.h`) is the other half of that: an error is only useful if it
  names the right *cause*, and "cannot open" conflates a typo, a directory passed in
  place of a file, and a genuinely unreadable file. It separates those, calls out a
  0-byte file (a half-finished write, not a permissions failure), reports when the
  *directory* is what's missing, and for a missing file scans **every directory the
  search path would have looked in** (0.192.0 — see *Where a relative asset path is
  looked for*) and suggests the nearest names by Levenshtein distance — because the
  report that started all this was a single dropped character in a 30-character
  filename, which is invisible to the person who typed it. When more than one directory
  was consulted it also names them (`[searched: …]`), so a miss tells you *where* it
  looked rather than leaving you to guess the base directory. Budgeted at ¼ of the name
  length (clamped 1–6) so it cannot suggest an unrelated file, capped at 4000 directory
  entries per root, and reached only on an already-failed path, so it is off every hot
  path by construction.
- **`assetbytes.h`** (0.148.0) — the three things a scene's asset *bytes* may need that
  aren't parsing: an **overlay**, a **warmer**, and (0.192.0) the **search path** that
  turns an authored relative path into a real one — `resolve` / `ScopedSceneDir` /
  `setExeDir`, described in full under *Where a relative asset path is looked for*.
  `Overlay` is a map from `normKey`
  (lowercased, forward-slashed — so loom's `Path.as_posix()` names match ftrace's
  lookups on Windows) to bytes; `ftsl::Builder` carries one and every mesh dispatch
  site consults it before touching the disk, which is what lets the live viewer hand
  meshes over the pipe under the very same filenames the `.ftsl` text names, so nothing
  downstream has to know which transport produced them. `Warmer` is the complement for
  assets that *are* on disk: `loadSource` scans the scene text for `file "…"` paths
  (`scanAssetPaths`) and reads-and-discards them on one background thread, capped at
  64 MB, purely to make the OS and the virus scanner do their work concurrently with
  the GPDA parse instead of serially in front of each `open()`. It is deliberately
  format-agnostic and constant-memory — it never decodes anything, so it cannot
  disagree with the real loader. Skipped when a non-empty overlay is present (the
  bytes are already here), which doubles as the A/B switch `scraps/warmbench.cpp` uses.
  Measured on cold assets: **−21.5 %** on a 24-mesh scene, **−6.2 %** on gallery's 27 MB;
  on settled assets, where it can only ever lose, it costs nothing (−0.7 to −1.9 %,
  i.e. still marginally ahead). That last column is the one that justifies it being
  unconditional.
- **`implicit.h` / `isomesh.h`** — implicit/isosurface evaluation and marching-cubes
  tessellation. `marchImplicit` is staged **fill → discover → resolve → wind**:
  parallel lattice `val[]` fill and parallel per-vertex bisection refine + gradient
  (pure per-slot), with the order-sensitive weld-map sweep and winding pass kept
  serial — bit-identical to the old serial code by construction. Per-implicit
  marching also runs in parallel across objects.

  **Oriented container box (0.121.1).** An `expr` isosurface is NOT a distance
  field, so the marcher clips the ray to the authored `contained_by` box and steps
  by `|f| / max_gradient` — a bound the author only guarantees *inside* that box.
  `Implicit` therefore stores the container in its OWN frame (`boxOriented`,
  `boxInv` = world→container-local, `boxLo`/`boxHi`) and `intersectImplicit` runs
  the slab test there. An affine map preserves the ray parameter
  (`p(t) = o + t·d` ↦ `boxInv(o) + t·boxInv_dir(d)`), so the resulting `t` values
  are directly comparable with `tmin`/`hit.t`; only the two face normals return to
  world, via `Affine::applyDirTranspose` (`boxInv`'s linear part transposed — NOT
  `applyNormal`, which would invert a second time). `ftsl.h`'s `addIsosurface`
  sets `boxOriented` only when the local→world map is not axis-preserving, so
  every unrotated scene takes the original world-AABB path and renders
  bit-identically.

  Clipping to the world AABB instead is a real invisibility bug: under a rotation
  that AABB is strictly larger than the box (for the gallery heart, **4.36× the
  volume**), the field out there is far steeper (max `|f|` 1688 → 37738), and the
  first sphere-trace step of `37738/60 ≈ 629 m` leaps clean over a 0.6 m object.
  Diagnostic signature: **the rasterizer shows it and every ray-traced mode does
  not** — marching cubes samples a lattice and never sphere-traces, so it cannot
  overshoot. Guarded by `-checkcontainer`, which builds one sextic solid twice
  (axis-aligned and rigidly rotated) under a shared `max_gradient` and fires
  correspondingly rotated rays: a rigid motion cannot change a hit distance, so
  any disagreement is the clip region leaking outside the container.

  **Cap-fraction guard on `-export-mesh` (0.121.0).** A capped isosurface marches
  `max(f, contSDF(p))`, so if the field's sign is inverted (`f < 0` *outside* the
  intended shape) the container wins everywhere and the export is the `contained_by`
  shape with the object hollowed out invisibly inside it — indistinguishable from a
  plain ball until you strip the shell. `isomesh::capFraction()` classifies each
  output triangle by which term won the `max()` at its centroid, and `main.cpp`'s
  export loop warns when the cap exceeds half the output. Diagnostic only; it never
  changes the mesh. (Cost is one field eval per triangle, once, at export time.)

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
  labyrinth. **A glossy vertex also takes the light connection here (0.275.0)**, on the same
  lattice-vs-rng principle: `neeLight`'s `whitted` branch walks a `lightGrid`² quadrature over the
  light and the lattice direction supplies the other half of the MIS weight, so mode `W`'s glossy
  estimator is mode `R`'s with quadrature substituted for sampling rather than a different one.
  Until 0.275.0 all four glossy sites (host/device × scalar/hero) guarded that connection behind
  `!whitted` and returned before it, which made a glossy surface **pure black at `-spp 1`** — the
  mode's headline configuration, since sample 0 of the lattice *is* the mirror direction. See
  `known-issues.md` → the `type glossy` backward-NEE entry. That lattice is what makes the mode *consistent* on rough specular: collapsing
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

  **Host/device bit-exactness is a TEST, not a comment (`-checklattice`, 0.137.0).** Every
  claim above ("trivially bit-identical", "must stay bit-identical") is now asserted by a
  scene-free self-test. `src/lattice_probe.h` is a dependency-free header giving `main.cpp`
  and `render_cuda.cu` **one** definition of a 33-column probe row (radicalInverse2, rot05,
  whittedSample, whittedLambdaU, whittedOrderU, whittedFluoroU, whittedGlossyUV, giPhases,
  gridUV, and `radicalInverseScr` in all 20 bases the mode uses); `cudaLatticeProbe` runs the
  device twins over an index sweep and the host compares raw bit patterns. Two helpers were
  factored out (`whittedGlossyUV` from `whittedGlossyDir`, `giPhases` from the two
  `giGather*` bodies, with `dWhittedGlossyUV` mirroring the first) precisely so the probe
  reads the renderer's own code rather than a copy of it — a copy could drift, which is the
  bug class the probe exists to catch. This is N4 part (a); part (b), image agreement to fp32
  tolerance with no *structural* difference, stays with `scraps/n3_check.py`, because whole-
  image bit-exactness is not achievable (device `Real` is fp32, `RAY_EPS` is 1e-4f vs 1e-6,
  and libdevice's transcendentals differ from the CRT's).

  The test earned its keep immediately: nvcc contracts a multiply feeding an add into an FMA
  by default and MSVC does not, so `dRadicalInverseScr`'s `r += digit * f` disagreed with the
  CPU by 1 ULP on 1.6 % of indices. FMA is the *more* accurate form, but accuracy is not the
  contract — mode `W` has no Monte-Carlo noise to absorb a difference, so the host is the
  reference and the device must reproduce its rounding. Fixed by spelling both multiply-adds
  with `__dmul_rn` / `__dadd_rn`, which the compiler may not fuse, rather than a global
  `-fmad=false` that would have perturbed every other kernel. **General rule this establishes:
  a `double` expression duplicated host-and-device is not bit-identical by construction — a
  bare `a*b + c` will diverge — so either write it with the `_rn` intrinsics or don't claim
  bit-exactness for it.** (`dGiDir` still contracts, but its output goes through `cos`/`sin`
  and was never bit-comparable; it is covered by part (b).)

  **Sealed-light detection (0.119.0).** Because mode `W` implies `-direct-only`, NEE is the
  *only* way anything is lit — and `Scene::occluded` blocks a shadow ray on any geometry,
  dielectrics included (the SDS limitation). A light sealed inside refractive or mirrored
  geometry therefore reaches no vertex at all and the mode renders pure black, previously
  with `auto-exposure=1` as the sole hint. `Scene::emitterSeal()` (`scene.h`) probes an
  emitter with a deterministic 512-direction lattice — stratified over the surface through
  the existing `Emitter::samplePoint`, uniform over the outgoing hemisphere, or inside the
  cone for a `Spot` — and returns the fraction whose first hit is a material satisfying
  `isSpecularType()`. That predicate is exact rather than heuristic: `backward.h` calls
  `neeLight()` from the `Diffuse`, `DiffuseTransmit` and `Fluorescent` cases and nowhere
  else, so the two sets are complements by construction, and a change to either must keep
  them so. Self-hits (`h.matId == e.matId`) yield no evidence and are excluded, so a
  concave mesh light is not mistaken for a sealed one; `Env`/`Sun` return 0 outright.
  `main.cpp`'s `warnSealedLights()` runs it once when `g_whitted || wPreview` — the explorer
  is included because its `T` preview *is* mode `W` — and warns past `kSealWarnFrac` = 0.95.
  The threshold is short of 1.0 on purpose: a real lamp assembly has hardware inside the
  envelope (the gallery's arc probes 98.2 %, the remainder being its own socket and cord),
  and a first pass at 0.995 missed the exact scene the check exists for. Verified across all
  98 scenes in `scenes/`: three trip it, all the same sealed-lamp assembly, no false
  positives. The probe is mode-agnostic; only the call site is gated, so `R`/`P` could
  adopt it.

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
  wavelengths *are* the whole spectral quadrature and they share one BVH walk — measured at
  **2.7 %** of frame time for the full 8 versus 1 (N5, 0.138.0), because the mode is
  traversal-bound. The converse is that `-heroc 1` here is not a speed/quality trade at all:
  it is a correctness hazard, since there are no further samples to average the collapse
  away. See the `-rgb`-in-mode-`W` note under `render_cuda.cu`.

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

  **An fp32 device cannot simply inherit the CPU reference's absolute epsilons or its
  algebra** — 0.128.0 fixed two places where it had, and both showed up as the *same*
  symptom: a **distant** light losing most of its energy in mode `D`/`U` (a 1.85 m sphere
  400 m away — a scene modelling the sun that way — rendered 2.7× too dim; 1.5× at 40 m;
  clean by ~4 m). (a) `intersectSphere` used the textbook `disc = b² − 4ac`, whose terms are
  `O(dist²)` while their difference is `O(radius²)`, so a sphere `k` radii away carries ~`k²`
  ulp of error in its hit distance (±1 cm at 400 m). It now uses the stable Ray-Tracing-Gems
  form: discriminant from the ray's *perpendicular* offset, near root by Vieta. (b) Every
  connection ray was shortened by an absolute `dist - 2e-6` copied from the all-double
  `bdpt.h`; one float ulp is `dist·2⁻²³`, so past ~17 m that rounds back to `dist`, the ray
  ends exactly *on* the sampled light point, re-hits the emitter and is thrown away as
  occluded. `connMaxT(dist, absEps)` now shortens by `max(absEps, dist·CONN_REL_EPS)`, with
  `CONN_REL_EPS = 1e-5` in fp32 and `0` in the fp64 build so the latter stays bit-identical
  to the CPU; `absEps == 0` still means "don't shorten", which is what the distant-sun
  connection (far end = the scene exit, not a surface point) requires. **The general rule:
  any epsilon compared against a distance must be relative on the device, and any quadratic
  solved there must be written in a cancellation-free form.**

  0.259.0 found that rule had only been applied to the BDPT/VCM *connection* rays. Every
  NEE shadow ray (`bkEmitterGeom`, `bkNeeVolume`, `bkNeeLightRGB`) and every forward-mode
  camera leg (`connect*`, `connectEmission*`, `connectHero`/`connectLensHero`) still used
  the absolute `dist - 2*RAY_EPS` — 2e-4, which is below half an ulp of a float past
  ~3360 scene units. The symptom was the same one again in mode `R`: a sphere light
  6325 units away rendered 23.7 % dark on the GPU, 1.6 % at 3162, clean at 1581, and the
  identical scene scaled by 0.1 was clean — a precision failure, not an algorithmic one
  (known-issues.md GPU-NEE-EPS; it had been misfiled for six weeks as a participating-media
  disagreement). All fourteen sites now go through `connMaxT(dist, 2*RAY_EPS)`, and
  `tools/check_distant_light.py` renders a far/near scene pair on both devices as a gate.
  Two gaps remain and are logged: the hair variants (`bkHairBlocked`, `connectHair`,
  `connectLensHair`) still subtract an absolute `off + RAY_EPS`; and the ray **origins**
  are all absolute — `p + n*RAY_EPS` (dead past ~1680 units), the BDPT/VCM `ng*1e-6f` and
  the caustic chains' double `1e-6` cast to float (both dead past ~17 units) — which is
  observed in mode `B` as photons re-hitting the emitter they were born on (GPU-ORIGIN-EPS).
  The origin fix is not a relative epsilon (that leaks through walls at city scale) but
  Wächter–Binder's integer ulp offset along the departure-side geometric normal.
  0.259.1 made that the rule. `dOffsetAlong(p, ng, w)` (render_cuda.cu, next to `DVec3`) is
  the ONLY way a device ray may leave a surface: it advances `p` by 256 float ulps along
  `ng` flipped to the side of `w` (Ray Tracing Gems ch. 6), with a 1/65536 absolute push
  inside |p| < 1/32 where ulps are too fine to matter — scale-correct by construction and
  bounded in absolute terms (0.25 units at |p| = 10 km, which is also the thinnest wall
  fp32 can still resolve there). It replaced ~100 `p + n*RAY_EPS` bounce and shadow
  origins, the BDPT/VCM `ng*1e-6f` connection origins, the caustic chains' double
  `1e-6` (which the cast to float had been discarding), and the photon-birth
  `origin + dir*RAY_EPS`, which was along the wrong vector as well as the wrong size:
  births now step along `emitN`. Medium vertices have nothing to clear and are not
  offset; hair fibres keep `dHairExitOffset` (2.5 radii along the exit direction). The
  fp64 device build keeps `p + n*1e-6`, bit-identical to the CPU. `tmin = RAY_EPS` on
  the next trace stays as a second guard, not the first.
  **A scale-correct origin push breaks an invariant the old code relied on**, and the first
  build with it showed how: the far-end shortening (`2*RAY_EPS`) used to be *larger* than the
  origin push (`RAY_EPS`), so a shadow ray that kept the direction and length computed
  from the unmoved point still stopped short of its target. With a 256-ulp push toward the
  target that is no longer true whenever the light is nearer than ~3x the shading point's
  coordinate magnitude: the far end overshoots the sampled emitter point by push*cos(a),
  lands inside the emitter, and the sample is discarded as occluded — a radius-6 sphere
  613 units from a quad at |p| = 5000 lost 97 % of its NEE samples in modes R and D alike
  (a spot light there: clean; mode B, whose shadow rays end at the pinhole: clean). So
  every occlusion query with a specific far-end point now goes through `occludedTo(sc, o,
  target, eps)`, which re-aims the ray *from the moved origin* (PBRT's `SpawnRayTo`) —
  the estimator's own G, pdfs and directions are untouched — and `connMaxT` gained a
  `coordMag` term (8 float ulps of the target's largest coordinate) because at |p| ~ 5e4
  the target point's own quantisation (4e-3) is as large as `dist*1e-5`. The four
  scene-exit rays (sun cone, environment) keep plain `occluded()`: nothing sits at their
  far end to overshoot into. Rule: **moving a ray's origin means re-deriving its direction
  and length from where it now starts; never keep a direction computed from a point the
  ray no longer leaves from.**

  `-rgb` is refused in mode `W` (`main.cpp`, with a message): the fast RGB backward is a
  separate reduced tracer with no deterministic estimator, so it would return precisely the
  noise the mode exists to remove.

  **And there will not be a deterministic RGB twin of it, because mode `W` is
  traversal-bound (measured, N5, 0.138.0).** The obvious follow-up — "write an RGB mode `W`
  and get `-rgb`'s 1.6–2.3× in the deterministic mode too" — does not survive measurement.
  That speedup is entirely *bundle width*, not the cost of being spectral: on the Cornell
  box a **spectral** mode-`R` render at `-heroc 1` costs exactly what `-rgb` costs
  (4.4 s / 4.4 s at 8192 spp), so the fixed overhead of SPD evaluation, upsampling and CIE
  integration measures as **zero**. And mode `W` barely pays for width, because it fires
  `4×4` shadow rays per light per hit (plus the `-gi` gather) and the extra wavelengths ride
  along on rays that are already being traced: on a 15-second frame (gyroid room, 7680×4800,
  `-gi 32`) the full 8-wide default costs **2.7 %** over a single wavelength, against **61 %**
  for mode `R`'s 4-wide bundle on Cornell. So a second hand-written RGB megakernel could win
  ~1–2 %, in exchange for a permanent — and now *tested*, via `-checklattice` — obligation to
  stay bit-exact with a CPU twin, plus `cudaBackwardRGBSupported`'s scope gate blanking the
  deterministic preview on exactly the materials it is most useful for (media, thin film,
  gratings, multilayer, layered, fluorescence, textured albedo). **The general rule: before
  porting a cost win from one mode to another, check which term it actually came from — mode
  `R` is shading-bound and mode `W` is traversal-bound, so their cost models do not
  transfer.**

  The same measurement is why mode `W` *raises* the default bundle to `kHeroMax` rather than
  economising on it, and why `-heroc 1` is a hazard there specifically: at 1 spp the bundle
  is the entire spectral quadrature, so narrowing it does not add noise that more samples
  would remove — it changes the answer. `warnWhittedHeroCollapse` guards the batch path
  (the viewer already guarded itself via `wNeedSpp`); see `known-issues.md`.
- **`lighttree.h` / `lighttree_build.h`** (0.187.0) — the **Conty–Kulla adaptive light
  BVH**, the many-lights selector. `lighttree.h` holds the node layout and `ltSample()`
  and is deliberately **dependency-free** (`<cmath>`, `<cstdint>`; `LT_FN` =
  `__host__ __device__ inline` under `__CUDACC__`) so the `.cu` includes the *same*
  traversal source the CPU compiles — one implementation of the importance heuristic,
  not two that can drift. Nodes carry the members' bounding **sphere** (`center`, `r2` —
  baked by the builder from the AABB as `0.5·(bmin+bmax)` / `0.25·|bmax−bmin|²`, the exact
  expressions the traversal used to derive per call before 0.190.4 stored them
  pre-computed; `sinθ_o` is baked the same way), a bounding cone of emission
  normals (axis, `cosθ_o`, `sinθ_o`, `cosθ_e`) and subtree flux; importance is
  `power · cos(θ − θ_o − θ_u) / d²` against a `θ_u`-widened `|cos|` at the receiver.
  `lighttree_build.h` does the build; `Scene::buildLightTree()` calls it from
  `finalizeEmitters()`, so every path that rebuilds the emitter list (`-ignoreenv`,
  `applyIgnoreFlags`, the built-in scenes) gets a matching tree rather than a stale one.
  The CLI knobs live in the header as `lt::gEnabled` / `lt::gSplit` / `lt::gSamples`
  (the `hero::gSplit` pattern — `inline` variables, legal in the `.cu` because
  `CMAKE_CUDA_STANDARD` is 17), which is what lets `render_cuda.cu` read the same policy
  `main.cpp` set without a second copy of the flags.

  **Why it exists, and what it replaced.** Backward NEE ran `for (e = 0; e < nEm; ++e)`
  and cast a shadow ray to *every* emitter at *every* non-specular vertex — an unbiased
  splitting estimator whose cost is O(N_lights) per bounce. Measured on the GPU (mode R,
  256 spp, 256², same room, same total 20 000 lm): 1 light 0.4 s → 256 lights **75.9 s**,
  at an identical 6.25 % noise. `BackwardRenderer::pickEmitters` and its device mirror
  `dPickEmitters` now replace the loop *bound* only: they hand back a set of
  (emitter, `1/pdf`) draws, so `sum_e w_e` becomes `sum_selected w_e / p(e)`. That
  framing is what keeps the change surgical — the connection code below it is untouched,
  and on the fallback path the weight is **exactly** `1.0`, so `x * 1.0 == x` leaves
  legacy images bit-identical rather than merely close.

  **The unbiasedness contract.** Every bound in the heuristic must be *conservative*
  (hence `|cos|`, and `cosθ_o = −1` for spheres/cylinders): a bound that is too tight
  gives a contributing emitter probability 0, and the image silently *darkens* instead
  of getting noisier. **Adaptive splitting** visits both children at probability 1 while
  `(radius/distance)² > gSplit`, and only chooses stochastically below that — which is
  why small-N scenes are safe: full splitting degenerates *exactly* to the old
  estimator. One light builds no tree at all (`lightTreeRoot < 0`), and mode `W` always
  takes the exact path, since its deterministic shadow grid has no variance to trade.

  **The REVERSE walk, `ltSelectPdf()` (0.270.0).** `ltSample` answers "which emitters, and with
  what probability"; the BSDF-sampling half of a glossy MIS weight needs the opposite — it has an
  emitter, discovered one bounce later by a ray hit, and needs the probability the selector
  *would* have chosen it. `ltSelectPdf` walks root→leaf and multiplies the factor each step
  contributed: **1** at a split (both children taken with certainty), the child's `ltImportance`
  share at a stochastic step. The path is **unique** in a tree, so there is nothing to integrate —
  the worry that adaptive splitting makes this a sum over the split set is why the function did
  not exist for a year, and it is unfounded. `Scene::lightTreeParent` (node → parent) and
  `Scene::lightTreeLeaf` (emitter → its leaf node) are the indices it needs, derived from the
  finished node array by **one linear pass** in `buildLightTree()`, so the recursive builder is
  untouched and every pre-existing selection pdf is unchanged.

  **What it deliberately does not reproduce, and why that is safe.** `ltSample` declines a split
  when its output or stack is full; that depends on traversal order and therefore on the rng, and
  is not recoverable a bounce later. The resolution is not to chase it but to have **both halves
  of the weight call this one function** — a weight owes *consistency*, not correctness, since MIS
  is unbiased for any weights forming a partition of unity, and the true selection pdf enters the
  **estimator** (through `EmitterDraw::weight`, which `ltSample` returns exactly) rather than the
  weight. Getting that backwards is what made the first attempt unbiased-but-noisier; see
  `known-issues.md` → GLOSSY-NEE.

  **And the walk is usually skipped (0.270.1).** `ltSample` gained a `roomLimited` out-param; when
  it comes back false its returned pdf *is* what `ltSelectPdf` would produce, so the NEE side uses
  the number it already has. `ltSelectPdf` mirrors `ltSample`'s arithmetic exactly — including
  taking the right child as `1 − pL` rather than `iR/sum`, which differ in the last bits — so
  "reuse instead of re-walk" is a **bit-identity** claim that is verified on both backends rather
  than argued. Measured: the re-walk cost 4.8 % of a frame on a 48-light scene and bought nothing.

  **The second O(N) term, which only appeared once the shadow rays were gone.** The CPU
  refilled a per-sample table of every emitter's SPD at every sampled wavelength (1024
  Planck evaluations per sample at N=256, C=4) and then summed over emitters *again* to
  derive `invPdfLambda`. Both are now over **distinct spectra**: `spectrum.h` gained
  `SharedSpectrum`/`ScaledSpectrum` so an emitter's SPD is provably `base_g(λ) · scale_e`
  (the FTSL loader memoises identical spectrum expressions, and `absPower` — where
  `lumens`/`power` normalisation happens — returns a `ScaledSpectrum` instead of a fresh
  closure, which is what made identity survive normalisation); `Scene::buildSpdBases`
  collects the distinct bases plus `spdBaseGeom[g] = Σ geomWeight_e·scale_e`; and
  `backward.h`'s `SpdCache` became an *accessor* (`at(e,i) = base[baseIdx[e]*C+i] *
  scale[e]`) rather than a materialised emitter-major table, so nothing O(N) is even
  stored per sample. The value is bit-identical because `base*scale` is literally how
  `ScaledSpectrum::operator()` computes itself; the `Σ_g base_g·spdBaseGeom[g]`
  regrouping is the one place the summation *order* changes, and it is exact whenever
  there is one emitter per base (every single-light scene). CPU ml256 at 64 spp:
  **524.7 s → 7.0 s**.

  **Verified, not assumed.** `-no-lighttree` on the new binary reproduces the pre-tree
  binary bit-for-bit across a 10-scene corpus on both devices (`scraps/regress_pair.py`,
  against a `git worktree` build of the previous commit). With the tree: ml256 image
  means 37.745 vs 37.749 (ratio 1.00012); variance is unchanged (16 spp against a
  4096-spp reference — RMSE 6.616 no-tree, 6.665 at the default 8 samples); and the case
  designed to break a biased tree — a 40 m corridor whose 256 panels span orders of
  magnitude of per-light importance (`scraps/gen_corridor.py`) — lands within 0.07 % of
  its no-tree reference while running 6.1 s → 0.6 s.

  **Not yet using it:** the forward/BDPT/VCM `selectEmitter` power CDF
  (`render_cuda.cu`), and a mesh emitter's own triangles (still area-sampled) — both
  tracked in `known-issues.md`.
- **`radcache.h`** (0.190.0; device notice 0.190.1, deterministic chunking 0.190.2,
  checkpointed table 0.190.3) — the
  **world-space diffuse radiance cache** behind `-radcache`, mode `R` on the CPU. A clipmapped hash of cells (54-bit quantised
  position, clipmap level, 54 normal buckets) each holding a 16-bin spectral mean, its
  variance and a confidence flag; `backward.h`'s `radianceHeroLoop` reads it at a diffuse
  vertex *after* that vertex's own NEE and *before* the continuation roulette, adds
  `thr·ρ·(E/π)·invPdf` and stops. That placement is what makes the partition exact: direct
  light is counted once by the reader, everything beyond the vertex once by the cache.
  (The reader marks first and reads second; with `-radcache-jitter` off the two keys are
  identical, so since 0.190.4 the call site derives key+mix once and probes via
  `lookupBundleAt` instead of letting `lookupBundle` re-derive both.)

  **Camera paths never write it**, which is the design's load-bearing asymmetry and the
  reason a miss is free. A separate update pass between chunks (`main.cpp`
  `renderBackwardBand` → `mergeMarks` / `takeWork` /
  `BackwardRenderer::updateRadCacheCells` / `apply`) shoots its own cosine rays from the
  cells camera paths *marked*, under a budget of update samples **per cache consult the
  chunk actually made** — proportional to the work the cache is being asked to do, not to
  the size of the table. So an unresolved corner renders exactly rather than answering
  with noise, and the table is immutable for the whole of any render pass (merge, update
  and apply all run after the join), which makes its state a pure function of the chunks
  rendered so far rather than of the thread interleaving.

  **Update rays must not read the cache** (`rcTrain=true` on the update path). Otherwise a
  cell stores `direct + somebody else's cached tail` — a fixed point of the cache's own
  error, amplified by `1/(1−f)` — and the collapsed sample variance corrupts the very
  confidence gate that is supposed to catch it.

  **Verification against the readers** (`-radcache-validate`, default 0.05) is what
  actually bounds the error. A random fraction of readable vertices decline to terminate,
  trace the tail to full length, and report `(offer, tail)` back to the cell; `Σtail/Σoffer`
  is an unbiased estimate of that cell's *total* systematic error **as used** — cell
  averaging, normal-cone spread and unsampled tails, weighted exactly as readers weight
  them. Well determined → adopt as `corr`; provably wrong but unpinnable → retire; not
  enough data → 1. Two details are load-bearing: the coin is flipped **before** the tail is
  known (an earlier fate-selected design measured 1.8 % dark), and `lookupBundle` returns
  the **raw** mean with `corr` handed back separately, so a validation record is an
  absolute measurement rather than one relative to a correction that is itself moving.

  **Why 54 normal buckets** (6 faces × 3×3, not 6 × 1): a 90° face bucket folds a sphere's
  normals into one average, a systematic error no spp removes. The 3×3 split keeps the face
  *centre* in its own bucket, so an exactly axis-aligned normal cannot be shattered across
  four buckets by ±1e-17 of rounding; a flat surface pays nothing either way.

  **Auto cell size is a pixel footprint, not a scene fraction** (`main.cpp`,
  `32 · 2·d·tanHalfY / resY`). What pays for a cell is the number of camera samples that
  read it, which the image sets, not the scene — so this keeps the economics
  resolution-invariant (4× the resolution gives 4× the cells *and* 4× the paths).

  Measured, and the limits, in `REFERENCE.md` → *Radiance cache*; the residual failure mode
  (concentrated specular/caustic transport, −18.76 % ± 0.31 % per raw read on a dispersive
  Cornell vs +2.38 % ± 0.07 % all-diffuse) and the fur-class scenes it does not help are in
  `known-issues.md`. **Not the same decision as mode `W`'s `-gi` gather**, which
  deliberately has no cache (see the three constraints under `backward.h`'s `-gi` note):
  that estimator must stay a pure function of `(index, sample index)` so an animated
  seamless loop cannot flicker, and `-radcache` fails exactly that test — its cells depend
  on render order and on what the update pass happened to sample, so it is opt-in and
  documented as unsuitable for animation.

  **It made the chunk split observable, which cost `-device cpu` its determinism** until
  0.190.2. Every CPU render is supposed to be a pure function of the scene and the sample
  budget: `cpuSppChunks` may split the budget however it likes because per-`(pixel, sample)`
  seeding means chunk `[base, base+c)` renders the same samples whatever `c` is. The cache
  advances *between* chunks (the update pass runs at each boundary), so the boundaries
  joined the realization — and they were being chosen by wall clock, retargeting toward
  ~0.4 s per chunk. Two identical invocations measured 40.1 % vs 41.4 % of consults
  terminated and images 0.45 % rel-RMS apart; pinning the split via the pre-existing
  `FTRACE_CHUNK_SPP` triage hook made them bit-identical, which is what pinned the cause.
  `cpuSppChunks` now switches to a schedule derived from `sppTarget` when `g_radCache` is
  set (1, 9, then an equal split into ~64 chunks); flagless renders keep the timed rule and
  are byte-unchanged. The non-obvious part is *why the deterministic schedule has that
  shape*: the chunk count IS the update-pass count, and the cache is violently sensitive to
  it — 65 chunks terminate 35.5 % of consults, 16 chunks 2.3 %, 5 chunks 0 %, with the last
  two **slower than not caching** because the update pass is paid for and no cell ever
  resolves. A flat split — the obvious way to make something reproducible — is precisely the
  wrong answer here, so the rule targets a chunk *count*, not a chunk *size*.

  **The table is checkpointed (0.190.3), sparsely and under its own guard.** It rides in the
  `.ftbuf` sidecar as a trailing section after the film blobs (`writeRadCacheSection` /
  `readRadCacheSection`, `main.cpp`), which needs no magic bump because both film readers
  consume a fixed number of fixed-size records and never demand the stream end there — the
  format is extended, not reinterpreted. Three details carry the design. **Sparse**, because
  the default table is 262 144 cells (~82 MB) and would otherwise be rewritten every
  `-interval`; occupancy is a tiny fraction of that (36 cells on a Cornell box, 8.7 k on
  `fur_creature`) since cells tile the visible *surface*. **Its own guard**
  (`RadianceCache::configGuard`), because `checkpointGuard` covers scene/mode/resolution and
  no cache parameter at all, so it would happily accept a sidecar whose cells were keyed
  under a different `-radcache-cell` or a different clipmap centre — a value measured over
  one volume of space filed under the key of another. Two guards let a changed cache setting
  discard the table while the image still resumes. **Deferred adoption**
  (`radCacheAdoptPending`), because the reader runs before the cache's configuration exists:
  the auto cell size comes from the camera and resolution, so the records are parked and
  folded in at the top of the first backward pass, through `loadCell`, which rehomes each
  cell by key and therefore tolerates a changed table capacity. Measured effect on
  `cornell.ftsl` at 200², resuming +512 spp on top of 1024: 68.4 % of consults terminated /
  65.9 M rays warm, versus 24.9 % / 70.5 M from the same film with the section stripped. The
  motivation is not only speed — a cold resume mixes unbiased (never-terminating) samples
  with biased ones in a ratio set by the interruption history, which is not a property an
  image should have. It does *not* make a resume equal a single shot, because the chunk
  schedule restarts at `done = 0` and the update passes land elsewhere.

  **It reaches exactly one code path, and 0.190.1 makes that audible.** The GPU backward
  megakernel (`bkRadianceHeroLoop`, `render_cuda.cu`) and the scalar `radiance()` fallback
  (`heroC == 1`, media, GRIN, finite lens) have no read site, so `-radcache` there was a
  *silent* no-op — and, because `-device auto` prefers the GPU for mode `R`, it was the
  **default** no-op on any machine with a GPU: flag accepted, image correct, cache unused,
  a missing status line the only evidence. `runRender` now prints
  `[radcache] IGNORED: <reason>` at the same "device is resolved" moment as the existing
  `[medium]` warning, covering the GPU case, `-whitted`, `-direct-only` and the forward
  modes. Porting the read site to the device is in `open-work.md`; the hard part there is
  not the table (flat POD array, upload-after-update / read-only-during-chunk) but
  verification's per-thread `std::vector` of records, which needs an atomic-counter slab
  drained to the host each chunk.
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

  **Degenerate-vertex guards (0.199.1).** Every direction BDPT reconstructs from two vertex
  positions — the subpath walk's `wo`, and the incoming `wo`/`woE`/`woL` of all four
  connection branches (`s==0`, `t==1`, `s==1`, interior) — now rejects a zero-length edge
  before `normalize`, which `vertexPdf` had always done and the throughput side had not. Two
  vertices land on the same point whenever a free flight is shorter than the position's ULP;
  `Pcg32::uniformOpen` removes the exact-zero flight, but `render_cuda.cu` is `Real = float`,
  so a *sub-ULP* flight still collapses there and the guard is what actually fixes it.
  Relatedly, **every `max <= 0.0` reject in both BDPT ports is now written `!(max > 0.0)`**:
  NaN compares false against everything, so the old form passed an undefined contribution
  straight to the film. The CUDA film loop had no such reject at all and now mirrors the
  CPU's. Together these closed the "mode D goes numerically undefined in thick, high-albedo
  media" entry in known-issues.

  **Delta lights in BDPT** (since 0.124.0): mode `D` renders `light spot` and `light sun`.
  Both are Dirac emitters, so the strategies that would have to *sample* the delta are
  unavailable and must be dropped from the balance heuristic — otherwise the surviving
  strategies are under-weighted and the image loses energy. The pieces, all in `bdpt.h`:
  `isDeltaEmitter`/`isInfiniteEmitter` classify an `Emitter`; `Vertex::isDeltaLight()`
  reports it at a vertex; `vertexPdfLightOrigin` returns **0** for a delta emitter (there is
  no positional density to hit), which makes `s == 0` — "the eye path lands on emissive
  geometry" — vanish on both sides of every MIS ratio (`remap0` then turns both into 1, so
  the ratios stay finite); `misWeight`'s light loop skips a term whose *previous* light
  vertex is delta, mirroring pbrt's `deltaLightvertex` hook; and `vertexPdfLight` grows two
  branches — uniform-in-cone `1/Ω · 1/d²` for a Spot, and the planar `1/(πR²)` (no `1/d²`)
  for an infinite light, whose subpath origin is a fictitious disc point.
  `deltaLightSubpath` starts the light subpath: a Spot emits from its point, a Sun from a
  disc of radius `sceneRadius` shifted `-R` along the beam, and after the walk the Sun
  rewrites `path[1].pdfFwd` to the planar density (pbrt's "correct subpath sampling
  densities for infinite area lights" patch) so forward and reverse agree.
  `connectBDPT`'s `s == 1` NEE branch folds all three emitter families into one
  λ-independent `Wgeom` (`fall/(d²·pdfChoice)` for a Spot — whose SPD is an *intensity*,
  W/sr — `Ω/pdfChoice` for a Sun, `cosL·A/(d²·pdfChoice)` for an area light), so the
  radiometric conventions of `scene.h` are honoured in one place.
  Finally a Sun needs an **escaped-ray** strategy, because a specular chain can only see it
  by leaving the scene inside the solar cone and NEE cannot connect through a delta vertex:
  `randomWalk` optionally fills an `Escape` (direction + hero/secondary throughput) and
  `BdptRenderer::renderRows` adds `beta·Le` for every Sun whose cone contains that direction
  — with **MIS weight exactly 1**, valid only because the block is gated on *every* eye
  vertex being delta, which is precisely when no other strategy can reach the sun.
  Gated by `Scene::sunCount` so sun-free scenes pay nothing.
  `scenes/_deltalight_mix.ftsl` is the regression scene (area + spot + sun + a mirror
  sphere); mode `D` vs mode `R` agrees to 0.33 % of mean luminance.

  Since **0.126.0 the same treatment runs on the GPU**, so a spot/sun scene no longer falls
  back to the CPU in mode `D`. `render_cuda.cu` mirrors every piece one-for-one:
  `dIsDeltaEmitter`/`dIsInfiniteEmitter`/`dIsDeltaLightVertex` classify a `DEmitter` (the
  device vertex has no light pointer, so `dVertexPdfLightF` took a `const DScene&` to reach
  `sc.emitters[lightIdx]`); `dVertexPdfLightOriginF` returns 0 for a delta; `dMisWeight`'s
  light loop skips the delta-previous term — with the twist that the device does *not* mutate
  the vertex arrays the way pbrt's ScopedAssignment does, so the `i == 0` test must read the
  substituted endpoint out of `*QsP` when `s == 1` (`(si == 0) ? *QsP : light[0]`);
  `dGenLightSubpath` grew the delta branch (point origin for a Spot, `sceneRadius` disc for a
  Sun, plus the same `path[1].pdfFwd` planar-density patch); `dConnectBDPT`'s `s == 1` folds
  the three families into the same `Wgeom`; and the escaped-ray sun rides home in a new
  `DEscape` (filled by `dRandomWalk` on a miss, threaded through `dGenCameraSubpath`) which
  `kBdptT` accumulates straight into `camFilm` under the all-delta gate. `cudaBdptSupported()`
  consequently rejects only `Env` / `collimated`. Validated CPU-vs-GPU at equal spp:
  `_spot_cornell` 0.9963 mean ratio, `_sun_check` 0.9992, `_deltalight_mix` 0.9989 (absolute
  units) with the mirror's solar disc identical on both backends, and `cornell.ftsl` unchanged
  apart from ~1 ulp of float association in the rewritten `s == 1` estimator. The device is
  ~14× faster on the mix scene (16.5 s vs 229 s).

  **Delta lights in VCM** (since 0.125.0): the same treatment, restated in SmallVCM's
  compact `dVCM`/`dVC`/`dVM` running-partial-MIS form (`vcm.h` does *not* keep pbrt's
  explicit pdfFwd/pdfRev arrays). The three rules that carry the whole thing:
  (1) **`dVC` — and hence `dVM` — start at 0** for a delta light, and the NEE weight forces
  `wLight = 0`; that is the `dVC`/`dVM` analogue of `vertexPdfLightOrigin` returning 0.
  (2) **`dVCM` starts at `1/pdfDirW` = Ω for a Spot and `1/pdfPos` = πR² for a Sun**, which
  falls out of `directPdf/emissionPdfW` once `directPdf` is defined per family
  (`pdfChoice·pdfPos` for an area light, `pdfChoice·1` for a delta position,
  `pdfChoice·pdfDirW` for the infinitely distant sun) — the same two constants SmallVCM's
  `PointLight`/`DirectionalLight` produce. (3) **`misArrival` skips the `dist²` factor on the
  first edge of an infinite light's subpath** (new `foldDist2` parameter, false in exactly
  that one place), which is this renderer's form of the `path[1].pdfFwd` patch. On top of
  those, `traceLightSubpath` gained the two cone-sampling emission cases (the spot's
  smoothstep penumbra scales *radiance*, never a pdf; note `spotOmega` is the falloff-weighted
  solid angle, so the sampling cone `2π(1−cosOuter)` is recomputed), the camera NEE branch
  grew per-shape connection geometry (sun: shadow ray to the scene exit with **no** endpoint
  epsilon), and `traceCameraSubpath` tracks `camAllDelta` to add the escaped-ray sun at
  weight 1. Vertex connection and vertex merging needed **no** change: they read `dVCM`/`dVC`/
  `dVM` off the stored light vertices, and the zeros propagate correctly through `misScatter`.
  Validated vs mode `R`: `_sun_check` 1.0002 mean ratio (identical auto-exposure),
  `_spot_cornell` 1.0061, `_deltalight_mix` (absolute units) 0.9949 — the residual at the
  *default* merge radius is ordinary photon-mapping radius bias and shrinks with `-pmradius`.

  Since **0.127.0 the GPU VCM session does delta lights too**, so mode `U` no longer falls back
  either. `render_cuda.cu`'s `kVcmLightT` gained the per-shape emission block (cone sampling,
  point vs. disc origin, `directPdfW = pdfChoice·(isInfinite ? pdfDirW : isDelta ? 1 : pdfPos)`,
  and `dVC = 0` for a delta), the first-edge `dist²` skip for a sun
  (`if (!(isInfinite && edges == 1)) dVCM *= dist*dist;` — the device form of `foldDist2`), while
  `kVcmCameraT` gained the three-way NEE connection geometry with `wLight = 0` for a delta plus
  `camAllDelta` and the escaped-ray solar disc at weight 1. Connection and merging needed no
  change on either backend. CPU-vs-GPU: `_spot_cornell` 0.9997 (GPU 2.5 s vs CPU 427 s, ~170×),
  `_sun_check` 0.9959 (that residual *is* the ~1% auto-exposure jitter), `_deltalight_mix`
  0.9996 in absolute units, solar disc `1/16` on both — and `cornell.ftsl` mode U is
  **byte-identical** before and after the port, since every new density sits behind
  `dIsDeltaEmitter` and the area path keeps its RNG draw order.
- **`volcache.h`** (0.299.0; deposit split 0.300.0) — **PROTOTYPE, environment-gated, off by
  default and not a supported feature.** A volumetric **fluence** cache for the `order >= 2` part of
  the beam gather: an `res^3` grid of CIE-weighted photon path-length density, built by splatting
  multiply-scattered chords and marched by the camera in their place
  (`L += sigma_s * phase * fluence * T * dx`). `FTRACE_VOLCACHE=<res>` marches it; adding
  `FTRACE_VOLCACHE_SPLIT=1` also **erases** those chords from the `BeamMap` before `BeamMap::build`,
  via `volCacheSplit` in `beamgather.h`, so the SAH split, the CIE table, the boxes and the BVH are
  all built over the order-1 remainder only. Measured 64 % faster and energy-correct to 0.2 % on
  thick isotropic media; the erase, not the query-side skip, is where the whole gain is.
  Since 0.301.0 it also serves **anisotropic** media: in-scattering is a convolution of the
  directional radiance with the phase, which is diagonal in spherical harmonics, so directional
  bins were built (`FTRACE_VOLCACHE_SH=1`, Legendre moment `g^l` per band). Measured not to
  help on the diffusely-lit `_beams_ms` enclosure -- but a collimated backlight is different: on
  gallery_rain's sun-lit cloud the scalar `l = 0` reconstruction is 7 % dark and the bins recover
  it at no measurable cost, so since 0.311.0 the bins are the **default whenever `g != 0`**
  (`FTRACE_VOLCACHE_SH=0` forces the scalar reconstruction, `=1` forces the bins).
  Since 0.315.0 it is a command-line feature: `-volcache [res|off]` (the split included) and
  `-volcache-sh auto|on|off` set `vccfg::*` in `volcache.h`, which the accessors consult before
  the `FTRACE_VOLCACHE*` environment variables the prototype was gated by (still honoured).
  Since 0.302.0 it serves **heterogeneous** media too (optical depth is integrated along the
  chord rather than assumed) and keeps **one grid per medium**, so a scene can be partially
  cached -- `gallery_rain` caches its HG cloud while its `phase rainbow` rain stays as beams.
  Since 0.303.0 it also serves **rainbow** phases, by a different formulation: a bow's colour is a
  function of scattering angle, so the CIE lives in the KERNEL (per-channel Legendre moments of
  `Scene::BowLut`, by quadrature) and the stored coefficients are scalar. Restricted to the
  gather-time-fold (`achro == 2`) beams of one emitter; everything else stays a real beam.
  The march is clipped to each grid's box, not to the camera ray, since a ray that hits nothing
  is handed `tMax = 1e30`. **Since 0.310.0 the march has a device twin** (`render_cuda.cu`
  `dVolCacheMarch`, called at the top of `dGatherPhotonBeams` so it runs even when the split left
  no beams): the host builds the grids and splits the map inside the shared-map build lambda
  (`buildBeamMap` → `volCacheSplit`), `volCachePrepareForGather` publishes the grids through
  `g_volCachesForUpload`, and `uploadBeamMapCuda` uploads them as `DVolCache` records (the SH
  coefficients as one float buffer per medium) hung off `DBeamMap`. The device march is the
  host's loop to the letter — 64 steps over the grid's span, `sigma_s(550 nm) * density * T *
  inScatter * dt`, with `dVolCacheInScatter` reproducing `VolCache::inScatter` including the
  bow-mode kernel moments. The split's refusal (`volCacheHostGather() == false`) is now only for
  gather paths that have no march at all -- mode J's UPBP-conv wavefront queue (`kWfBeamHits` /
  `kWfBeamEval` run their own per-hit estimator and never enter `dGatherPhotonBeams`); the
  shared mode-M device path sets it true. See known-issues, VOLCACHE, for the scope limits, the measurements, and for the
  cache-ownership bug that made its first measurement meaningless.

- **`surfmerge.h`** (0.258.0) — the **surface photon map mode `J` merges against**, i.e. the
  half folded in from mode `U`. Three things live here and nowhere else:
  **On by default on the CPU since 0.260.0** — all three UPBP-VM gates green (gate 3: switching the
  third technique in on `_fog_cornell` moved the image by X −0.86 % / Y −0.27 % / Z +0.12 % mean,
  medians within 0.1 %). **On the GPU too since 0.263.1** — the device carries the same three-technique
  estimator: `DSurfMap` (the host's dense lattice uploaded as it stands, so both sides bin by
  one rule), `dSurfMergeAt` gathering per camera *vertex*, and a device weight that finally
  carries BOTH merge kinds. That last part is the reason the port could not be partial: with a
  single kappa the camera-side `segSumM` could be left unscaled and multiplied once in the
  denominator, which no longer works when two kinds compete, so the device adopted the host's
  convention of scaling each kind at accumulation time. Gated bit-for-bit both ways — merges off is still
  mode `D` exactly, and the beams-only path is byte-identical to before the port — and validated
  against a 3.1 M-spp mode-`R` reference: the merge half matches the CPU's to a median ratio of
  0.9998, the full render to −0.7 / −0.4 / +0.6 % of ground truth. One trap worth keeping in mind:
  `SEGN`, the size of the camera-side MIS sum arrays, is `MERGE ? MAXV : 1`, so the merging kernel
  must be selected for EITHER merge kind — selecting it from the beam map alone left a media-free
  `-jsurf` render weighting every merge by a denominator missing its camera-side term (UPBP-VM).

  * **`SurfPhoton`** — a light-subpath vertex on a surface: position, `wo` toward the previous
    (light-side) vertex, λ, β, the precomputed CIE triple (the same trick that bought mode `M`
    3.65×), and `misIdx` into a parallel `SurfMis` array.
  * **`SurfMis`** — the light half of the merge weight, frozen at deposit time:
    `sumC` (light-side connections), `sumMb` / `sumMs` (light-side merges, one accumulator per
    merge KIND because `n_m·2r_b` and `n_m·π r_s²` cannot be folded into one κ), `pdfFwdA`,
    `rCoef` (the cosine, the inverse-square and the remapped `1/pdfFwd` of the one step the
    gather has to patch), `gateC1`, and `vert` for the depth cap. **The index shift is the
    subtle part**: `bdpt.h`'s convention is that C1 is "the merge site is the last *camera*
    vertex", and a point merge's site *is* an existing vertex rather than a new one, so C1
    splits after `y_{j-1}` and every light-side accumulator is read at **j-1** — one index
    earlier than a beam merge, which inserts a vertex and reads at **j**.
  * **`SurfMap`** — flat SoA arrays plus a counting-sort uniform grid with `long long` cell
    arithmetic that **coarsens** rather than refusing on a huge scene (mode `U`'s own dense grid
    still has the `(int)(nCells+1)` overflow this avoids). `query(p, fn)` scans the 3×3×3 cell
    box and keeps the inscribed sphere. `misOf(i)` returns **null** when the map carries no
    weight for a photon, and the gather then **skips the merge entirely** rather than weighting
    it 1 — unlike a beam map, whose weight-1 fallback degrades to mode `M`'s estimator (too
    bright, still a picture), an unweighted point merge would be added *on top of* a complete
    BDPT sum and double-count every path it touches.

  `vmGatherCorr` (the shading-vs-geometric-normal correction the density estimate needs, exactly
  1 on flat geometry) was **moved** here out of `vcm.h`, which now does `using bdpt::vmGatherCorr;`
  — mode `J`'s point merge and mode `U`'s VM are the same estimator and must not be able to drift
  apart. Modes `M`/`S` deliberately do **not** use it.

  The deposit side lives in `bdpt.h`'s `traceLightBeamPass` (same subpaths, same `nEmitted`, as
  the beams — they are two views of *one* set of light subpaths, which is why the two merge kinds
  can share a denominator at all), and the gather side in `BdptRenderer::renderRows` under
  `SurfMergeWeight`. Per **vertex**, not per segment: a point merge happens where the camera walk
  actually landed, so there is no ray to march and **no RNG draw** — which keeps "merges off ==
  mode `D` bit-for-bit" true for free.

  **The radius shrinks per light-side epoch, and that is a correctness property, not a tuning
  knob.** `main.cpp`'s `surfRadiusFor(epoch)` applies Georgiev/SmallVCM's schedule
  `r_e = r_0·(e+1)^((α−1)/2)` under the same `-vcmalpha` mode `U` uses, mode `J`'s refresh epoch
  being its analogue of a VCM iteration. A merge estimator's bias is `O(r²)`, so a fixed radius is
  a bias floor sampling cannot get under — measured on `_cornell_diffuse` (1024 spp vs an 8192 spp
  mode-`D` reference) with `r` pinned: **−0.11 %** at `r=0.005`, **−0.45 %** at `0.0173`, **−0.52 %**
  at `0.035`, monotone and heading for zero, i.e. the `O(r²)` floor rather than a defect in the
  weight. Consistency then follows from plain averaging: epoch bias `~C·r_e² ~ e^(α−1) → 0`, and the
  render reports the unweighted mean of its epochs, so the total bias is the Cesàro mean of a null
  sequence. This was the one thing mode `U` had that mode `J` lacked, and it had to be acquired
  before `U` could be retired — though the measured head-to-head has since said `U` wins its own workload by 4–6× at equal time and stays (UPBP-VM). Only one plumbing fact makes it a two-line change: `mk.surf` is
  derived from `SurfMap::radius` at gather time and the light pass's `κ_s` from the same field, so
  handing `build()` a different radius retunes the estimator *and* its MIS weight together, per
  epoch, with no second source of truth. Epoch 0 gets `r_0` exactly.

  **Sizing (`-jsurf-count`, default `4e6` ≈ 288 MB at 72 B per photon+`SurfMis`).** The surface map
  and the beam map are budgeted by the *same* pilot but with deliberately different authority. The
  beam side names a *beam* ceiling and sizes the pass, because its radius **adapts** to the budget
  and undershooting the `-beamk` knee widens the kernel — a measured bias (+6.4 % on the analytic
  slab). The surface side names a *photon* ceiling and may only clamp the subpath count
  **downward**, after the beams have chosen, because its radius follows the schedule above and a
  larger map costs nothing but memory. On a media-free scene the beam pilot measures nothing, so the
  light pass is sized at **one subpath per pixel** (mode `U`'s convention) and capped by this
  ceiling; before that existed, the inert `-n 2000000` default asked for 7.3 M photons / 500 MB and
  failed the allocation outright. The pilot's own subpaths are discarded, so its surface banks are
  **cleared** alongside the beam banks — `nEmitted` comes from the real pass, and leaving the
  pilot's photons in the map would put ~3 % more flux in it than the normalisation divides by.
- **`vcm.h`**, **`sppm_render.h`**, **`photonmap.h`/`photonmap_render.h`/`photonmap_io.h`/`causticaim.h`** — U/S/M.
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
  `cie*power*norm/pi` into three floats. The `-savemap` cache format is `FTPMP05`
  (header, surface photons, beams, then the caustic map — each version a strict superset of the
  last, so `FTPMP03` and `FTPMP02` files still load, the former with an empty caustic map and the
  latter with no beams either); `FTPMP01` files are rejected with a message telling the user to
  re-deposit. `FTPMP05` (0.202.0) differs from `FTPMP04` only in the *width* of the beam record —
  it gained the spectral bundle (below) — so the reader keeps `PhotonBeamV4` frozen in
  `photonmap_io.h` and widens a `FTPMP04` beam block in 65536-record chunks (chunked so peak
  memory is not doubled), setting `nSec = 0`. The same pass fixed a latent reader bug: a file with
  beams loaded by a caller that did not ask for them used to fall straight into reading the caustic
  header out of the middle of the beam array, and now seeks past the block in 1 GB hops (`fseek`'s
  offset is a 32-bit `long` on Windows).
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
  **The lattice is HASHED, not dense** (`pmCellHash`, 0.199.6). A dense grid indexes cells as
  `(iz·ny+iy)·nx+ix` and must therefore *allocate* `nx·ny·nz` ints — a count that grows as
  `(L/r)³` in scene size over gather radius, while the photons themselves live on surfaces and
  only occupy `(L/r)²` of it. So on any large scene the *empty* cells set the memory bill, and
  `buildAuto` had to defend itself with a guard (`kMaxCells`) that **grew `r` back** until the
  array fit. That guard was the binding constraint on every large scene, and it bound hardest
  exactly where a fine radius was most wanted: on `gallery_rain` (scene radius 32.7 m) a
  requested 0.031 m was inflated to 0.094 m, three times too coarse to resolve a caustic —
  which is what smeared the scene's caustics into a colourless grey veil. Hashing sizes the
  bucket table from the **photon count** (`pmTableSize`, 2× rounded up to a power of two, i.e.
  a flat ~8 B/photon against the map's 56), so the volume term disappears and `r` is free to
  follow the measured density; the only limit left is the deliberate two-octave clamp. Cells
  that collide into one bucket are not a correctness problem — the query already distance-tests
  every candidate — and at 2× occupancy that costs ~13 extra tests across a 27-bucket
  neighbourhood. The mix is SplitMix64's finaliser over three odd-constant products rather than
  the classic XOR-of-three-primes, because the XOR form leaves *neighbouring* cells correlated
  in the low bits and a gather visits 27 neighbouring cells at once. Host and device share the
  one function (`PM_HD`), and the device's three traversals (mode-M gather, its final-gather
  sub-ray, SPPM) now go through one `dPmNeighborhood` helper instead of three hand-copied
  transcriptions of the host binning. `cellCoord` no longer clamps to the bbox: with no array
  to run off, clamping was actively slightly wrong (a query just outside got folded onto the
  edge cell and lost the far row of its neighbourhood). `-checkpmgrid` validates the query
  against brute force on exactly the configuration the dense lattice could not represent.
  **`medianNeighborCount` samples by bucket, not by array index** — bucket membership is fixed
  by the bbox, cell size and hash, i.e. by geometry, whereas the counting sort is stable and so
  preserves a within-cell order that differs between a fresh deposit and a `-loadmap` of the
  same map. Sampling by array position therefore made `-loadmap` stop reproducing its
  `-savemap` run. Within a sampled bucket the representative is the lexicographically smallest
  position (a set-minimum, hence order-free); the cell *centre* would not do, because a cell
  that the surface merely clips has its centre off-surface and reports a spuriously empty
  neighbourhood.
  The GPU shared path gets the same treatment via `renderPhotonMapSharedCuda`'s `autoK`
  argument (0 = off) — it must, since that is the high-photon-count path where a
  count-independent radius collapses worst. The gather reads `pm.radius` after the build, so
  the adapted value needs no further plumbing.
  **TWO MAPS: the Jensen caustic split** (0.199.7; `-caustics`/`-nocaustics`, `-pmccount`, on by
  default). All of the above solves for **one** population, and a scene with caustics has two
  that differ by orders of magnitude in density — a gem's focused light lands as a thin bright
  filament, the ambient illumination as a broad wash. `buildAuto`'s answer is set by the
  majority, i.e. the wash, so the caustic is convolved with a kernel far wider than the feature
  and vanishes; forcing the single radius down instead (`-pmcount 11`) brings the caustic back
  and takes the whole image to grain with it. Mode `M` therefore **partitions its deposits**:
  a photon arriving at a diffuse vertex after ≥1 **focusing** vertex and **no scattering** one
  (the classic L·S⁺·D regular expression) goes to a second `PhotonMap`, the caustic map, built at
  its own radius; every other deposit goes to the global map. The gather is the plain **sum** of
  the two density estimates. The partition is *strict* (each deposit lands in exactly one map)
  and both maps divide by the **same `nEmitted`** — which counts paths EMITTED, not photons
  stored, and is the classic two-map bug if you use the caustic map's own count instead: a rare
  caustic would be rescaled to full light-source brightness.
  **THE GATHER FOOTPRINT (`-gatherarea`, on by default at 8 probes since v0.268.0; mode `S`
  gained it in v0.273.1; the fiber gate v0.277.0, and `-gabias` / `-gagate` / `-gaball` all
  on by default since v0.278.0).** The density estimate above divides by `pi r^2`, the area of a FULL
  disc, while two things stop the photons it sums from having come from one: the query rejects
  any photon whose normal disagrees with the hit's by more than 60 degrees, and nothing clips the
  disc to the surface, so wherever the disc overhangs a silhouette or a thin feature that part of
  it collected from geometry that is not there. Both are area mismatches and both are one-sided,
  so they add: on `gallery_rain` the error runs 0 % on flat ground, −38 % on a tabletop edge,
  −44 % on cloth and −70 % on hair, in proportion to how much of the disc is real surface.

  The fix measures the covered fraction by **ray probe** rather than analytically: `M` stratified
  points in the tangent-plane disc, each traced from `r` above the plane straight down, accepted
  on the same 60-degree test the photon query uses, and weighted by `1/cos(tilt)` because the
  probe samples PROJECTED area while the estimator needs SURFACE area. `gatherCoverage` /
  `gatherAreaScale` in `photonmap_render.h`, device twin `dGatherCoverage` in `render_cuda.cu`,
  and `sppm_render.h` applies it to each pass's flux BEFORE it enters `tau` — coverage belongs to
  the radius actually gathered at, and SPPM shrinks its radius every pass, so a single coverage
  measured at the final radius would misprice every earlier one.

  **An analytic clip was the obvious design and is the wrong one here:** clipping same-facing
  primitives to the disc is exact for triangles and impossible for fur, isosurfaces and CSG
  solids — and fur is the single largest loss on the scene that motivated this, so the exact
  method would have missed the case it was built for. One intersector call handles every
  representation, and it is the same intersector the render already trusts.

  **It was not a quality dial, and three separate defects were why (all fixed, v0.277.0-v0.278.0).**
  The estimator had four things wrong with it at once, and because they partly cancelled, fixing
  any one alone made the numbers *worse* — which is why the entry stalled for so long and why each
  of these shipped only with the others.

  * **Dense fur had the sign backwards.** A tangle *overfills* the disc, so `creature`'s coat was
    accurate uncorrected and **+48 %** corrected. Fixed by the **fiber gate** (v0.277.0): skip the
    correction wherever the gather point is on curve geometry, tested on `Hit::fiberRadius` /
    `DHit::fiberRadius`. It is a *geometric* test, not a photon statistic — reject-rate and depth
    heuristics were both tried and cannot separate fur from hair, while the fiber test does it
    100 % to 0 %. Counting the hidden layers is NOT the fix and was measured: their area is real
    and their photons are accepted, but the visible point is on the FRONT layer, so dividing its
    photons by front-plus-back area dilutes the surface being shaded.
  * **Jensen's bias made the probe count a BIAS knob, not a convergence knob.** The estimate is
    `1/c-hat` of a noisy `c-hat` and `E[1/c-hat] > 1/E[c-hat]`, so fewer probes read brighter.
    Fixed by the pseudo-count `(area+1)/(M+1)` (`-gabias`), which is exactly 1.0 at full coverage —
    so flat ground stays untouched by construction — bounded by `M+1`, and the textbook
    near-unbiased estimator of `1/p`.
  * **The flat-interior early-out asked for more evidence when given more probes.** Its threshold
    was `probe0 = M/4`, i.e. 2-of-2 flat-on probes at `M = 8` but 8-of-8 at `M = 32`, so the same
    disc passed the same test at different rates purely because the budget changed. `-gagate`
    holds it at a fixed count, and `-1` (the default) removes the early-out entirely.
  * **The probe accepted a CYLINDER while the query gathers from a BALL.** Probes start `r` above
    the tangent plane and accept `h.t <= 2r`; the numerator is `queryR(p, r)`. A point at tangent
    offset `rr` and height `dz` is at `sqrt(rr^2 + dz^2) >= rr`, so on anything non-flat the probe
    counted rim surface the query can never reach — divisor too big, estimate too dark. `-gaball`
    adds the one inequality, and on flat geometry `dz = 0` makes it algebraically inert.

  Together: mean absolute error on `gallery_rain`'s five ROIs **36.8 -> 8.0 points**, and the
  estimator is now nearly independent of its own probe count (`M = 8` vs `32` disagreement
  **3.30 -> 1.06**) rather than accurate by cancellation. All three flags restore the previous
  behaviour when set to 0, and all three read the same environment channel on host and device, so
  the backends cannot disagree about which estimator is running.

  `-gatherarea 0` still restores the pre-0.267 estimator exactly, but it is now a **speed/accuracy
  knob rather than a correctness escape hatch**: the correction costs ~1.6x the frame on a
  surfaces-only mode-`M` render, and is unmeasurable on a beam-heavy one where the volume gather
  dominates.

  The classifier is `photonVertexKind` (`render.h`) with device twin `dPhotonVertexBit`
  (`render_cuda.cu`), three-way: **FOCUS** (dielectric, mirror, thin-film, multilayer, grating,
  half-mirror, and glossy at roughness ≤ `kCausticGlossRoughness`), **SCATTER** (rough glossy,
  fluorescent, hair, an *analog* medium collision, and any diffuse vertex — which sets the flag
  for itself, since what leaves a diffuse vertex is indirect light by definition), and
  **NEUTRAL** (`filter`, a coloured absorber the photon passes straight through undeflected —
  and, for the same reason, a `-beams` straight medium crossing, so a gem's caustic seen through
  fog still counts). `kCausticGlossRoughness = 0.15` (a ~12° Phong lobe) lives in `photonmap.h`
  rather than `render.h` because `render_cuda.cu` does not include the latter, and two threshold
  constants would let a CPU and a GPU render of one scene classify differently — a discrepancy
  whose only symptom would be "the GPU looks wrong".
  On the **GPU** the classification runs in the kernel (a per-path `int pathBits`, threaded
  through `shadeStep`/`shadeStepHero`/`kTrace` and, in the wavefront backend, held per-slot in
  `WFState::pathBits` exactly like the medium stack), but the **partition happens on the host at
  download time** off an explicit `DPhoton::caustic` field (record 32 → 36 B). A second device
  atomic cursor would have needed its own capacity, overflow detection and rerun-at-a-lower-rate
  path, all kept in agreement with the first one's; a packed sign bit would have saved 12.5 % of
  a buffer several kernels write and two host loops read. `kCountCaustic` is a one-pass device
  reduction whose only job is to let the host size **both** output vectors exactly before the
  copy — sizing both to `nDep` would double the host map's peak footprint at exactly the photon
  counts where it is already the binding constraint, and growing them chunk-by-chunk would make
  `std::vector::resize` recopy a multi-GB array once per chunk.
  `buildAuto` gained an `rMax` cap whose one caller is the caustic map, passing the global map's
  radius: `k(M) = kAt1M·cbrt(M/1e6)` shrinks with the count, so a **sparse** caustic population
  makes `sqrt(k/n)` ask to *grow* the radius — on `gallery_rain` at `-n 40M`, 21 808 caustic
  photons drove 0.655 m → 1.094 m, smearing the one thing the split exists to keep sharp. Capped,
  the worst case degenerates to the unsplit answer. Mode `S` (SPPM) is deliberately untouched —
  its progressive radius already handles caustics, and its `causticDeposit` stays null.
  That was the *storage* half of Jensen's scheme only, and on its own it leaves the caustic map
  sharp but nearly empty (0.12 % of `gallery_rain` deposits are L·S⁺·D). The *sampling* half is
  `causticaim.h`, next.
  **One radius per MAP is still one radius (0.205.0, `-pmadaptive`).** The split assumed the
  caustic map holds only caustics. It does not: L·S⁺·D is satisfied by *any* specular bounce, so
  in a gallery full of glass and metal the caustic map also holds a broad, sparse, room-wide
  specular wash — a second population inside the second map, and again the majority one, so
  `buildAuto`'s median-density probe is decided by it. Measured on `gallery_rain`: the probe saw
  1–6 photons against a target of 47 and therefore asked to *grow*, `rMax` pinned it, and the
  caustic map's radius came out **exactly equal to the global map's** (0.1775 m) in every run —
  `-pmccount` inert, the split a no-op. The visible consequence was issue 6 of the mode-M
  showcase report ("I don't see any colourful caustics"): metered against a mode-D reference of
  the same camera, the axicon cap's caustic peaked at **2.9×** its own median where mode D peaks
  at **9.7×** — the energy and the hue were right, the feature was smeared flat. A `-loadmap`
  radius sweep confirmed the radius was the whole story (peak 2.93 → 3.75 → 6.41 → **8.05×** as
  the map-wide radius went 0.1775 → 0.0887 → 0.0435 → 0.0217 m) and also why shrinking it
  map-wide is not the fix: at the small end every sparse-region gather catches one or two
  *spectral* photons and paints a random saturated hue, so the wash turns to chromatic speckle.
  The fix is to solve **per query**: `PhotonMap::adaptiveRadius` / `dPmAdaptiveRadius` return the
  radius of the smallest disc around the gather point holding `kGather` same-facing photons,
  capped at the map's radius and floored at `r/256`, and the caller renormalises by `1/(π r_q² N)`
  for the radius it actually used. A filament shrinks its own kernel to its own scale; the wash
  finds fewer than `k` in the whole disc, returns the map radius unchanged, and is bit-identical
  to before. Only the caustic map sets `kGather` (to the same `k` `buildAuto` solved its radius
  for); the global map and SPPM keep the fixed-radius path exactly.
  It costs **one** extra neighbourhood walk, not a kNN heap (a per-thread heap is what a GPU
  gather cannot afford) and not an iterated count-and-shrink (a full 3×3×3 walk per iteration).
  The walk histograms each candidate's `d²/r²` into 16 geometric shells `(2^-(i+1), 2^-i]` via
  `i = -ilogb(t)-1`, so the *suffix sum* of the histogram is a whole radius/population profile
  from one pass; take the tightest shell still holding `k` and interpolate inside it under local
  uniform density, `r_q² = r_i²·k/C_i`. The normal test is applied in the counting walk too, not
  only in the caller's sum — counting a wall's photons while gathering the floor beside it would
  shrink the radius for a population the sum then rejects, and since the estimate is divided by
  `π r_q²` either way that mismatch would print a dark seam along every surface junction. On the
  GPU the per-record fold already carries the map's fixed-radius normalisation, so the caustic
  sum is rescaled by the area ratio `r²/r_q²` rather than re-folding every record.
- **`causticaim.h`** (0.203.0 CPU, 0.204.0 GPU; `-causticn`, `-causticaimk`) — the **aimed
  caustic emission pass**, Jensen's projection-map half, built as a *continuous mixture
  importance sampler* rather than a discretised spherical grid. Every primitive whose material
  `materialMayFocus` (conservative — the undecidable "roughness driven by a texture/pattern/record"
  case is *called* focusing, because a false positive costs efficiency and a false negative costs
  caustics) is clustered into `-causticaimk` bounding spheres. The union of their footprints is
  then closed form: a **cone** per sphere when the emitter is local (quad/sphere/cylinder/mesh/spot
  and a volumetric `fire` birth — the origin is fixed, so the *direction* is aimed), a **disc** per
  sphere when it is distant (sun/env — the direction is fixed by the solar cone or the sky
  importance map, so the *origin* on the upstream plane is aimed).
  **The collapse that makes the pdf free:** choosing target `j` with probability `P_j = m_j/T`
  (proportional to its own footprint measure) and sampling uniformly inside it (`q_j = 1/m_j`)
  gives `p_a(x) = Σ_j P_j q_j [x∈j] = count(x)/T` — one overlap count and one divide, whatever the
  geometry, with no normalisation integral and no grid.
  **Unbiasedness is structural, via the balance heuristic.** The aimed pass does not *replace* the
  main pass: `tracePhotonPass` runs it as a second pass of `nAimed` photons that deposit into
  `pmCaustic` only, and **both** passes' caustic deposits are scaled by the same per-photon weight
  `w = 1/(1 + (N_c/N_m)·rho)`, `rho = p_a/p_u`, computed at emission in `Renderer::applyCausticAim`
  — while the caustic map's `nEmitted` stays at the **main** pass's count. A main-pass photon
  nowhere near a gem has `rho = 0`, `w = 1`, and is untouched. So a target set that is incomplete,
  over-eager or badly clustered is an *efficiency* question and never a correctness one, and
  `-causticn 0` is bit-for-bit the pre-0.203.0 result (the aim map is never bound, so not even an
  RNG draw differs). Corollaries: an aimed proposal where `p_u = 0` (outside a spot's outer cone,
  below an area emitter's horizon, outside the upstream disc) is a zero *contribution*, not a lost
  sample; and a **collimated** emitter, which has no free variable to aim, gets `rho = 1` — the
  caustic pass emits it with the ordinary sampler and MIS degenerates to splitting the deposit
  between two identical passes.
  Two non-obvious constraints. The clustering objective is **Σr²**, not SAH: the sampler spends its
  budget uniformly over each footprint and a footprint's measure ∝ r², so Σr² *is* the expected
  fraction of aimed photons that miss. And the aimed pass must **transport by the same rules as the
  main pass** or MIS combines estimators of two different integrands — under `-beams` the main pass
  crosses media straight (an analog collision would set `sawScatter` and destroy the very
  classification the caustic map is defined by), so `Renderer::beamStraightOnly` makes the aimed
  pass cross straight too *without* storing beams, since the beam map belongs to the main pass and
  is normalised by its `nEmitted`.
  The aimed pass runs **after** the main pass rather than alongside it, so both use every core and
  so an `ftrace -stop` mid-trace simply skips it (leaving the un-aimed caustic map, which is still
  correct, just noisier). `buildCausticMap` prints a permanent **stored flux / emitted**
  diagnostic — the one quantity aiming must leave alone — which is how the weighting was verified
  (see `known-issues.md` for the tables). Measured: 74× more caustic photons on `gallery_rain` for
  a 25 % larger budget, with the global map's deposit count bit-identical.
  **The GPU twin (0.204.0)** is a field-for-field mirror, not a re-derivation: `DAimMap`/`DAimTarget`
  mirror `caim::AimMap`/`caim::Target`, `dApplyCausticAim` mirrors `Renderer::applyCausticAim`, and
  `renderPhotonMapSharedCuda` gained `aim` + `nAimed` and a second, caustic-only `depositLaunch`
  with `cs.aim.aimed` / `cs.beamStraightOnly`. The MIS weight is fixed at *birth*, so it rides in
  the per-path state — `int pathBits` became `DPathCaustic { int bits; Real w; }`, carried alike by
  the megakernel, the hero tracer and the wavefront's per-slot `st.pathC`; and because a deposit
  sets `PV_BIT_SCATTER` immediately after storing, at most **one** caustic deposit occurs per path,
  so the weight is consumed exactly once or never. The aimed launch's energy goes to a scratch
  buffer and is discarded (those photons are not additional emitted light), and `render_cuda.cu`'s
  `buildCaustic` prints the same *stored flux / emitted* line as the host, which is what makes the
  two backends directly comparable — they agree to 2e-4 on `scraps/caim_glass.ftsl`.
- **`allocreport.h`** (0.199.1) — `ftalloc::resize` / `ftalloc::reserve`, which turn a
  `std::bad_alloc` from a *command-line-sized* buffer into a message naming the buffer, the
  element count and size, the total in binary units, and the flag that shrinks it. Wrapped:
  the photon map positions/payloads (CPU pass, CUDA download, `-loadmap`), the photon-beam
  map, the split beam array, the beam CIE table, the beam BVH boxes. `main()` carries a
  `std::bad_alloc` backstop listing the four memory knobs for anything unwrapped. Motivation:
  `std::bad_alloc::what()` is the bare string `bad allocation`, so a render that died between
  two progress lines used to be diagnosable only by bisecting `-n` by hand.
  Since **0.204.1** the message also says *whose* fault it was, because naming the buffer alone
  can mislead: a 201 MiB failure in a process holding 5 GiB means the **machine** filled up
  (this actually happened — see `known-issues.md`), and "Lower `-n`" is then the wrong advice.
  `ftalloc::MemStat` + the `ftalloc::memStat` function hook (installed by `main()`, which is the
  only TU allowed to include `<windows.h>`/`<psapi.h>`) report the process's own **commit
  charge** (`PROCESS_MEMORY_COUNTERS_EX::PrivateUsage`) and the system's commit limit and free
  commit (`MEMORYSTATUSEX::ullTotalPageFile` / `ullAvailPageFile` — commit, not physical RAM, is
  what an allocation actually fails against on Windows). `memAdvice()` prints those and calls it
  **ours** if either the failed request is ≥25 % of what we already hold, or we are ≥50 % of the
  system's committed memory; failing both it says the machine ran out and that shrinking flags
  is the wrong move. The `bad_alloc` backstop passes `failedBytes < 0` (it never learns the
  size), so its verdict rests on the second test alone.

- **`photonbeams.h`** — the **view-independent volume cache** that makes mode `M` see
  participating media at all (CLI `-beams`, since 0.20.7). The surface map above is a *point*
  cache with no volume records whatsoever, so before this a fog / rain / cloud / rainbow scene
  rendered its subject as **nothing** under mode `M`; and `render.h`'s existing `-beams` is a
  camera **splat** (it needs the camera list at photon-trace time), which is structurally the
  opposite of the view-independence mode `M` exists for. Note also that `Photon` deliberately
  carries no incident direction (above) — and a rainbow is *entirely* a function of the angle
  between photon and eye, so the record that would serve it is precisely the one the surface
  map optimised away. Hence a **separate array with its own BVH**, not an extension of
  `Photon`.
  `PhotonBeam` stores a photon's straight crossing of one medium as a segment
  (`o`, `d`, `s0`, `len`, `power`, `lambda`, `absorb`, `med`, plus the `lamS`/`nSec` spectral
  bundle added in 0.202.0 — see below) — **one beam per (segment,
  medium) pair**, clipped to that medium's bound. That decomposition is *correct*, not merely
  tidy: `sampleMediaCollision` samples each medium independently and takes the minimum (a
  union of Poisson processes), so in-scatter at a point is the **sum** over the media
  containing it, and each beam must carry its own `sigma_s` and its own phase function.
  The camera estimate is **Beam × Ray with a 1D blur** (Jarosz et al. 2011):
  `L += K1(d_perp)/sin(theta) · Phi_b · Tr_beam(0→s_b) · sigma_s(x) · f_p(cos theta) ·
  Tr_cam(0→t_c) / nEmitted` — dimensionally `[W]·[1/m]·[1/m]·[1/sr]` = radiance, with **no**
  `1/(pi r²)` because the 1D Epanechnikov kernel carries its own normalisation. Parallel
  pairs (`sin theta → 0`) are a measure-zero singularity and are rejected. The gather rides
  `Bvh::traverseAny` with a leaf callback that always returns `false`, which turns the
  occlusion traversal into an **all-hits** one for free — no second traversal to keep in sync.
  **THE RADIUS IS NOT THE SURFACE MAP'S RADIUS — it is smaller by orders of magnitude**, and
  this is the trap the first working version fell into. A beam is a 1D object blurred in 1D,
  so `E[beams gathered] = (pi/2)·r·L_ray·S/V` (`S` = total stored beam length) — **linear** in
  `r` and in `S`, where the surface estimate's population goes as `r²`. At the photon map's
  radius that is ~44 000 beams per camera ray on a 1 m fog box and a 200×200 / 16 spp render
  never finishes. Solving the same expression for `r` at a target population `K`, substituting
  the convex-body mean chord `L_ray = 4V/A`, gives **`r = K·A/(2·pi·S)`** (`A` = beam-AABB
  surface area; `V` cancels).
  **BUT THAT EXPRESSION MUST NOT BE THE RULE — it was until 0.201.0, and it is a trap (0.201.0).**
  `S` grows with the photon count, so pinning `K` pins `r ∝ 1/n`: every extra photon the user
  pays for is spent **shrinking the kernel**, and the gather still averages the same `K` beams
  whatever the budget. Each beam is monochromatic (`PhotonBeam::lambda`) and one monochromatic
  sample is far outside sRGB, so averaging ~32 of them leaves full-saturation colour speckle —
  a hard variance floor **neither `-n` nor `-spp` can move**. That is exactly the reported
  "iridescent" `gallery_rain` cloud: measured, `-spp 16→64` moved the cloud's luminance sd
  0.1733→0.1674 and `-n 40M→160M` moved its chroma saturation 0.1157→0.1078 (both ≈ nothing,
  the log showing the radius shrinking 2.83e-4→1.65e-4 to hold `K`), while `-beamk 32→2048`
  moved saturation 0.1157→**0.0487**. And the bias the trade was buying **does not exist**: on
  the `_beams_ms` invariant (mode M + `-beams` must converge to the mode D reference), the
  ball/reference ratio is 1.0563 / 1.0816 / 1.0831 / 1.0884 / 1.0841 / 1.0876 for `-beamk`
  8 / 32 / 128 / 512 / 2048 / 8192 — under one point across a **1000× radius sweep**, i.e. the
  residual few percent is a constant offset, not kernel bias.
  **The radius is now PER MEDIUM and set by the medium's own measured mean free path.** Each
  stored beam *is* a sampled free-flight chord clipped to the medium's bound, so
  `mfp_m = S_m / N_m` (`BeamMap::mediumStats`, over the raw pre-split beams) measures it
  directly — no scene access, no wavelength choice, heterogeneity handled exactly, and bounded
  above by the medium's own extent. Then **`r_m = beamBlur · mfp_m`** (`-beamblur`, default
  `0.01`): a fixed fraction of the distance over which the medium's radiance field varies, so
  the blur is bounded by physics and **independent of `n`** — which is the point, because the
  gathered count now grows linearly in `n` and `-n` finally buys noise reduction. Per *medium*
  because a scene can hold a dense cloud and a thin rain curtain at once and one global radius
  is simultaneously too blurry for one and too noisy for the other. The constant is not fitted:
  `0.01` lands `gallery_rain`'s two media on 12.9 mm / 17.3 mm, just under the 18.6 mm that its
  `-beamk 2048` saturation sweep found good, and `_beams_ms` on 2.81 mm, between its bias-flat
  `-beamk 2048` (1.61 mm) and `-beamk 8192` (6.5 mm). The measured mfp is **stable to four
  digits across a 4× photon change** (1.292 m / 1.727 m at both `-n 40M` and `-n 160M`), which
  is the check that it is a property of the medium and not of the sampling. Measured result on
  the cloud: saturation **0.1157 → 0.0533** at unchanged mean luminance, i.e. the old default's
  noise was 2.2× the new one's *for the same photons* — and `-n 160M` now takes it to **0.0419**
  (−21 %, against −6.8 % under the old rule), with the residual gap to `1/sqrt(n)` being
  `-beamcount 1e6` capping the stored beams at 1.87× rather than 4×. The `_beams_ms` invariant is
  unmoved at ball/ref **1.0899** against 1.0816..1.0884 across the whole `-beamk` sweep, i.e. the
  much wider kernel introduces no new bias.
  `-beamk <K>` survives as a **floor**, not a target — `buildAuto` probes the gathered count at
  the mfp radii and scales them **up only** if a camera segment would gather fewer than `K`,
  which is what stops a very sparse map rendering as individual streaks (iterated up to 3
  rounds × 16×, because a single ratio against a probe that measured 0 is meaningless).
  `-beamareaslack <f>` (default `1.0`) is the matching **ceiling**: `areaSlackScale` solves the
  exact quadratic `A(s) = A0 + B·s + C·s²` for the largest uniform scale on the radii whose
  kernel inflation grows the total sub-beam AABB area — the gather's cost metric — by at most
  `f` over the tight `r=0` area. Bias bounded by physics, cost bounded by area, nothing pinned
  to a sample count. `-beamradius <r>` still overrides everything with one value for all media.
  The closed form survives only inside the **probe**, which brute-forces the gathered count over
  96 random chords of the beam bbox on a strided subsample (no BVH needed — none exists yet),
  because the closed form's two assumptions — isotropic relative orientation, beams spread
  evenly through the box — are both false in a sunlit rain volume where every beam is
  near-parallel to the sun, by more than an order of magnitude.
  On the **device** the per-medium radius rides in `DBeamRec::invRad` (one float per sub-beam,
  32 MB at the 8 M ceiling) rather than in a per-medium array indexed by `med`, which would be a
  dependent load in the volume gather's innermost loop. The reciprocal form also makes the
  kernel *cheaper* than the old shared-radius one: the reject test becomes `d2·invRad² >= 1`,
  which is the same `(d/r)²` the Epanechnikov kernel needs, where the old code took a `sqrt` to
  get `d` and squared it straight back.
  **A stored beam is SPECTRAL, not monochromatic (`-beamspec <n>`, default 4, 0.202.0).** The
  monochromatic record above is defensible for a surface photon and not for a beam: a photon is a
  *point* and one spectral sample there lays grain, while a beam is a *line* and one spectral
  sample lays a **saturated streak** down its whole length, which the eye reads as structure.
  `PhotonBeam` therefore carries `float lamS[kBeamSecMax]` + `int nSec` (hero in `lambda`, up to 3
  stratified secondaries), and `gatherPhotonBeams` / `dGatherPhotonBeams` fold all of them into XYZ
  **from the same chord**. Three facts make that nearly free, and each one is load-bearing:
  (1) **no per-wavelength weight is needed** — `EmissionSampler` samples `p(λ) = SPD(λ)/∫SPD`, so
  `spd(λ)/pdf(λ) = ∫SPD` independent of λ, and every wavelength in the bundle carries exactly
  `power/nLam()`; the record stores wavelengths only, and the gather multiplies `invC = 1/nLam()`
  onto the *end* of the hero weight so the `nSec = 0` case stays bit-identical to pre-0.202.0.
  **This is the one fact mode `J` had to buy rather than inherit** — its λ comes from the
  scene-wide sampler, not the chosen emitter's SPD — which it does by converting `β` and redrawing
  the deposit's wavelengths at birth (0.251.0; see the mode-`J` beam-map section above). Everything
  below applies to both modes unchanged, because after that conversion the two deposits are the
  same arithmetic.
  (2) **the gather's cost is almost entirely λ-independent** — geometry, `densityAt` (hoisted to
  one call shared by hero and secondaries) and the two ratio-tracking `mediaTransmittance` marches
  dominate and are shared; the per-λ tail is `sigma_s(λ)`, `phaseValue(cosθ, λ)` and three CIE
  lookups, measured at **~1.1×**. (3) **the existing hero machinery could not be reused**:
  `photonmap_render.h` gates it off with `heroOn = (heroC > 1) && scene.media.empty() && !hasGrin`,
  i.e. off in exactly the scenes that have beams. So the bundle is a separate, self-contained carry
  on the scalar tracer, stratified from one variate the same way hero is (`u + i/C` wrapped into
  `[0,1)` through the same CDF).
  **Unbiasedness rests on a deliberately conservative liveness rule**: the bundle survives exactly
  **one** transport iteration — birth on a plain SPD-sampled emitter, through empty space and
  achromatic media, to the first deposited chord — then collapses to `nSec = 0`. It is retired at
  GRIN scenes, glass Beer-Lambert absorption, image-env emitters and volumetric (fire) births, and
  **unconditionally at the end of every loop iteration, outside the beam block** — that placement
  is the subtle part: if the deposit is skipped (a march hit, a zero-length chord) the photon still
  goes on to interact, so a bundle left live there would be picked up by a *later* chord after a
  wavelength-dependent surface event. Scene-wide the feature is gated by `beamSpectralOK(scene)`,
  which lives in **`scene.h`** (not `photonmap_render.h`, which `render_cuda.cu` cannot include)
  next to the `Medium` it interrogates: it samples every medium's `sigmaT` at 33 wavelengths and
  refuses the bundle if any is chromatic, because the *shared* transmittance march would otherwise
  hand a secondary the hero's attenuation — a bias, not noise.
  On the **device**, `DBeamRec` quantises secondaries to `u16` as `(λ − 360)·100` over 360–830 nm
  (0.01 nm, two orders finer than the 1 nm spectral tables) because every byte of that record is
  8 MB of VRAM *and* 8 MB of innermost-loop traffic at the 8 M sub-beam ceiling; `DBeamSpec` is the
  in-flight carry threaded through `genPhoton` → `shadeStep` → `dEmitBeams`. Both splitters copy
  the whole `PhotonBeam` struct, so the bundle propagates through splitting for free.
  Measured on the isolated rain scene (`-n 60M -spp 16`, both runs producing byte-identical beam
  maps so the bundle is provably the only difference): luma texture rms 0.08828 → 0.08468 (−4.1 %),
  chroma 0.17615 → 0.16111 (−8.5 %), with the `_beams_ms` invariant unmoved (ball/ref 1.0865 vs
  1.0916, rest-of-frame 1.0200 both). A free, unbiased variance reduction — not a large one *in
  that configuration*, because its gather already averages ~1243 beams per probe ray and so already
  integrates ~1243 distinct wavelengths; the bundle earns most where the gathered count is low.
  **`-beamcount <n>` (default 1e6) is a ceiling reached by *adaptive* thinning, not by a
  predicted survival rate** — and that distinction was learned the hard way. The first
  version set a fixed `keepProb = n / nPhotons` up front, which silently assumes ~1 beam per
  photon; the real yield is a property of the scene's geometry and was wrong by two orders of
  magnitude *in both directions* (`_fog_cornell`, one unbounded medium: 2.25 M beams for a
  1 M budget, 2.3× over; `gallery_rain`, small bounded media in a big hall: **7 634** beams
  for the same budget, 131× under — visible as rain rendered in individual streaks rather
  than a volume). `BeamBank` now self-thins: each thread fills to a cap of
  `2·n/nThreads`, then **halves in place** (keep with p=½, double the survivor's power) and
  halves its own future `keepProb`, which the depositor re-reads per call. Every beam
  therefore carries the reciprocal of its own inclusion probability, so the estimator stays
  unbiased at any thinning depth; `decimateTo` applies the same trick once more at the end
  for the exact trim to `n`. The 2× headroom is what makes the banks still sum to ≥ `n`
  after one halving each. Measured effect on `gallery_rain`: **7 634 → 751 419** beams.
  What the knob buys **changed in 0.201.0** along with the radius rule. While the radius was
  sized to hold `K`, fewer beams meant a *bigger* kernel, so `-beamcount` traded **sharpness**
  against time and not noise (`1e6 → 1e5` on `_fog_cornell`: 7m29s → 43s, indistinguishable
  image, auto-exposure agreeing to 3 s.f. — itself an unbiasedness check). Now the radius is a
  physical scale that does not move with the stored count, so `-beamcount` trades **noise**
  against time and leaves sharpness alone — which is the sane meaning, and the one that makes
  it the volume analogue of `-pmcount` that it was always documented to be.
  **Beam splitting** keeps the BVH tight: a long diagonal beam is a mostly-empty AABB, so
  beams are cut into sub-segments — but a sub-segment keeps the parent's
  **true** origin `o` and records only its own `[s0, s0+len]` range, so the gather still
  measures transmittance from where the stored power actually applies and the splitter needs
  neither a `Scene` nor an RNG to re-integrate anything.
  **WHERE to cut is an area minimisation, and it has a work term (0.196.0).** The cost of the
  gather is not how many beams it collects — measured, a 16× change in `-beamk` moves the time
  3%. It is how many AABBs the ray must *enter*, which by Cauchy's formula is proportional to
  their total surface area. Splitting a beam of length `L` into pieces of length `p` gives
  `A(p) = 2L[Q·p + 4rE + 12r²/p]` (`Q = |dx||dy|+|dy||dz|+|dz||dx|`, `E = Σ|di|`), balancing the
  empty space a long diagonal box encloses against the kernel inflation paid per piece, and
  minimising gives `p* = 2r·sqrt(3/Q)`. That result is *per beam* and **O(r) with no scene-scale
  term** — which is exactly what the rule it replaced, `max(diag/64, S/3N)`, could not be: two
  scene-scale quantities that measurably did not move across a 10× beam-count sweep, making the
  gather's cost linear in stored beams. It also never splits an axis-aligned beam (`Q = 0`),
  whose box is already tight. But `p*` is the infinite-work limit and the BVH build is linear in
  sub-beam count (~1.6 µs each), amortised over every camera sharing the map, so the rule
  minimises `c_build·(S/p) + W·k·A(p)` instead and lands on
  `p_opt = sqrt(3/Q)·sqrt(4r² + κ/W)` — the area optimum with a work-dependent floor, `W` being
  the total pixel-samples the sharing cameras will gather. **A flythrough therefore splits far
  finer than a still of the same scene**, which is why `buildBeamMap` takes a `work` argument
  and the shared mode-`M` path sums it over the whole camera group. `κ ≈ 72 m²` is measured, not
  fitted: it predicts the observed single-camera optimum from two independent constants. Bounded
  by `-beamsplitmax` (memory, not quality — the optimal piece *count* grows as ~S²). Measured
  156 s → 92 s on `_fog_cornell` with bit-identical output; table in `known-issues.md`.
  **MULTIPLE SCATTERING (0.199.0).** Through 0.198.0 the beam map carried exactly one
  scattering order: the depositing photon crossed every medium straight, the extinction it
  suffered was booked as *absorbed*, and the stored beam paid back only the unscattered
  in-scatter — anything that scattered twice was deleted. In an optically thick, high-albedo
  medium that is most of the light. MEASURED on `scenes/_beams_ms.ftsl` (`sigma_t 6`,
  `albedo 0.95`, `g 0` over a 0.3 m ball, against a 20000-spp mode-D reference): the fog ball
  rendered at **0.330×** its correct brightness, i.e. two thirds of the volume was missing, and
  the photon ledger reported `absorbed=0.7542` where the truth is `0.6797`.

  The photon now runs plain **analog** transport under `-beams` and deposits **one long beam per
  chord between scattering events**, chord *k+1* carrying *k*-times-scattered flux. The chords
  tile the walk without overlap, so the gather estimates the *full* transport: the same ball
  comes out at **1.085×**, the residual being the Beam×Ray kernel's finite-radius bias (the rest
  of the frame sits at 1.019, which is the surface estimate's own radius bias).

  Three things make this a small, local change rather than a rewrite, and each is a trap if
  missed:
  - ftrace's beams are **long** beams — the segment runs to the next *surface* and the gather
    applies `Tr_beam(0→s_b)` analytically (`photonbeams.h:28`). Truncating a beam at the sampled
    collision instead would be a *short*-beam estimator and would **double-charge**
    transmittance. So the chord under MS is `dSurf`, not `dEvent`.
  - the analog `mediumEvent` in-scatter splat must be **suppressed** for any medium a beam chord
    already covered (`coveredByBeam`), or the same order is counted twice. GRIN collisions — on
    the marched curve, covered by neither a beam nor a `MedStraight` resample — still splat.
  - the beam's lead-in transmittance filter changes from `MedStraight` to `MedAll`: under single
    scatter GRIN extinction was carried *stochastically* by clipping, and under MS nothing is
    stochastic.

  Order counting: the first chord is already order 1, hence
  `beamMSAllowed(nDone) = (max == 0) || (nDone + 2) <= max`. The cap is CLI `-beams-order <n>`
  (`0` = unlimited, the default), carried to the CUDA translation unit as the header-inline
  global `pbeams::gOrderMax` for the same reason `hero::gSplit` is one — host renderer and
  device TU do not share `main.cpp`'s statics. **`-beams-order 1` takes the pre-0.199.0 code path
  verbatim and is bit-identical**, which keeps the crisper single-scatter bow available (dropping
  the desaturating multiply-scattered veil genuinely sharpens a rainbow) and doubles as a free
  regression lever. `kWfShade` (wavefront) and `traceHeroPhoton` pass no scatter counter, so they
  degrade to single scatter — neither runs beams. Cost note: MS deposits several times more beams
  per photon, so a `-n` that fit before may now exhaust the beam budget (see known-issues.md).
  **Each stored chord also RECORDS its order** in `PhotonBeam::order` (v0.273.4) — 1 = single
  scatter, 2 = it scattered once in a medium first, and so on — because the cap above is applied
  at DEPOSIT time, so without the field nothing downstream can tell an order-1 chord from an
  order-5 one. It costs no memory: it occupies the padding byte between `achro` and the
  2-byte-aligned `emIdx`, leaving `sizeof(PhotonBeam)` unchanged. `BeamBank::push` and
  `Renderer::emitBeams` take it as a REQUIRED parameter, so a new depositor cannot silently
  inherit a wrong value. **Mode `J` stores the sentinel `kBeamOrderUnknown`**: its natural
  candidate, `PathSeg::vert`, is the subpath VERTEX index and counts surface bounces, where this
  field means MEDIUM scattering order — passing it would type-check and make `order >= 2` mean two
  different things depending on mode. Anything that cannot be represented becomes the sentinel
  rather than the nearest representable order, where it would be indistinguishable from a
  measurement. The beam-map report line prints the distribution (`gallery_rain`: 59.3 % single
  scatter, 40.7 % order >= 2; `_fog_thick`: 17.2 % / 82.8 %) and stays silent when no chord tracks
  its order, so it cannot invent one for mode `J`.
  **Ported to the device in 0.197.0 — deposit *and* gather.** The scope is larger than "upload
  the BVH" because `-beams` changes the **transport**, not just the reconstruction: the
  depositing photon crosses straight, so the surface photon map and the beam map have to come
  from the *same* forward pass, and the device therefore had to learn to **deposit** beams as
  well. Both halves live in `render_cuda.cu`. The deposit is `__device__ dEmitBeams` (twin of
  `Renderer::emitBeams`) writing `DBeamDep` records through an atomic counter, gated in
  `shadeStep` by `doBeamDeposit = cs.beamCount && crng && sc.mediaN > 0`. (That gate carried a
  scene-wide `&& !sc.hasGrin` until 0.197.1; GRIN media are now refused **per medium**, inside
  `dEmitBeams` itself, because a photon bends only *inside* a gradient-index region — see the
  per-medium block below.) The
  gather is `DBeamRec` / `DBeamMap` / `dGatherPhotonBeams`, called from `dPhotonGather` **before**
  `thr` takes the segment's own attenuation — the ordering matters, because each gathered beam
  carries transmittance to *its own* closest approach, not to the segment end (the host does the
  same, `photonmap_render.h::photonGather`). Host orchestration is `struct BeamPass`
  (`render_cuda.h`), which hands `renderPhotonMapSharedCuda` the `BeamMap*`, the `-beamcount`
  target and the host `build` callback; the build deliberately runs **after** `-savemap`, since
  the file must hold raw crossings (see the persistence bullet below). Three non-obvious details:
  (a) **fold at upload** — `DBeamRec.pXYZ = cie(λ)·power/nEmitted`, mirroring `DGatherPhoton`, so
  the inner loop does no CIE evaluation and no `invN` multiply; (b) the host's `den = 1 - cosT²`
  is **unusable in float** (eps ≈ 1.2e-7 vs its `1e-9` rejection), so the device uses the
  algebraically identical, cancellation-free `den = |cross(dc, d_b)|²`; (c) the deposit's roulette
  and beam-side transmittance draw from the per-photon **side** stream `crng`, never the transport
  stream, so the device buffer-overflow rerun at a reduced `beamKeep` reproduces the surface
  deposit photon-for-photon (the host `emitBeams` uses the transport `rng` — a deliberate
  divergence, and why the two raw crossing counts differ ~12% while the images agree to 0.03%).
  The overflow rescale aims at **0.95×** the cap, not 1.0: survivors are Binomial and records past
  the cap are *dropped* (unlike photons, whose atomic still reports the true count), which would
  bias the estimate. Validated GPU-vs-CPU as mean linear-radiance ratios — 0.9996 gather-only
  (same map via `-savemap`/`-loadmap`), 0.9997 deposit-only, 0.9943 through a forced overflow
  rerun at `keep≈0.100`, 0.9945 on `_rainbow_test`'s spectral phase path. `_fog_cornell`
  128²×8spp: **3m42s → well under a minute**, now dominated by the *host* beam-BVH build.
  **There is NO backend carve-out, and the GRIN limit is per-medium (0.197.1).** The port first
  shipped with `beamsCpuOnly = wantBeams && grin::sceneHasGrin(scene)`, inherited by narrowing the
  old blanket fallback rather than deleting it. That was wrong twice over. **First, the fallback
  bought nothing**: the CPU refuses the GRIN deposit on *identical* terms (`render.h`'s
  `doBeamDeposit` carried the same `!grinAny`), so the "safe" backend rendered the same volumeless
  image, only far slower — a slower identical answer is a worse answer, not a safer one. The
  inability to store a curved photon is a property of the beam *representation*, not of the
  backend, so no backend can be the fallback for it. **Second, the gate was scene-wide.**
  `grin::sceneHasGrin` is "any enabled medium carries an `ior` field", but a photon bends only
  *inside* a GRIN region — `grin::march` jumps straight between them — so a photon's chord through
  an ordinary fog is a perfectly good straight beam even when a GRIN lens sits elsewhere in the
  room. The scene-wide gate therefore threw away every beam in the scene because of one unrelated
  object. Both fixed: the refusal moved into `emitBeams` / `dEmitBeams` as a per-medium
  `if (md.grin()) continue;` (device: `md.iorN > 0`), sitting right beside the existing
  absorbing-only skip, and the host `doBeamDeposit` / device `doBeamDeposit` gates lost their GRIN
  terms. `warnBeamsGrinMedia` replaces the fallback message: it fires only when a *scattering*
  medium is GRIN, names how many, says their volume will not appear, and states outright that
  `-device cpu` renders the same image. Regression scene `scenes/_grin_fog.ftsl` — a GRIN lens and
  an unrelated scattering haze in one room — deposits 1.9 M crossings and images the haze; before
  the fix it deposited zero.
  **A GRIN medium keeps its analog transport instead of being deleted (0.198.0).** Refusing the
  *deposit* is right; what 0.197.1 left behind was that refusing the deposit still let the photon
  be crossed **straight** and charged the medium's extinction as absorption — so a scattering GRIN
  medium was billed for out-scatter that nothing ever paid back, and behaved as a **pure
  absorber**: its volume rendered as nothing *and* surfaces lit through it came out dark. The fix
  splits the scene's media into two populations transported by different rules, and threads that
  split through the free-flight sampler and the transmittance integrator on both backends as a
  filter argument (`Renderer::MedFilter` = `MedAll`/`MedStraight`/`MedCurved`, device twin
  `DMedFilter`, both defaulting to `All` so every pre-existing call site is bit-identical):
  **non-GRIN media** (`MedStraight`) are crossed straight, deposited and gathered as beams exactly
  as before; **GRIN media** (`MedCurved`) are excluded from the crossing and keep full analog
  scattering — the same transport they get without `-beams` — sampled along the marched curve
  (see the `grin.h` bullet below). The straight crossing is then clipped at the analog collision
  (`dEvent`, not `dSurf`, in `tracePhoton` and `shadeStep`), which is also what makes the
  bookkeeping exact rather than merely plausible: because the crossing stops there, the
  probability that a stored beam covers depth *s* is already Tr_curved(0→s), so `emitBeams` /
  `dEmitBeams` must apply only `MedStraight` transmittance to the power they store, or the curved
  part would be counted twice. With no GRIN medium in the scene `MedCurved` matches nothing (no
  RNG is drawn, `dEvent == dSurf`) and `MedStraight` matches everything, so ordinary beam renders
  are unchanged draw-for-draw. Verified two ways: `scenes/_grin_scatfog.ftsl` (a GRIN lens plus a
  scattering GRIN haze) rendered GPU mode `M` at 20 M photons **with and without `-beams`** agrees
  to **energy bias −0.07 %, rel-RMS 1.09 % @8×8** (whole-frame ratio 0.9993, both halves within
  0.0008 of 1) against a ~4 % per-pixel photon-map noise floor — through 0.197.x it did not; and a
  sweep of **non-GRIN** media scenes against a 0.197.2 baseline binary came out **bit-identical**
  on every path that is deterministic at all (`R` GPU/CPU, `B` megakernel/CPU, `M` CPU `-beams`,
  and the `_g`/env variants). The four paths that differed — `B` GPU wavefront, `M` GPU `-beams`,
  `M` GPU plain, `S` GPU — differ from *themselves* by the same margin when the **baseline binary
  is run twice** (see the non-determinism entry in `known-issues.md`), so the sweep is clean.
  `warnBeamsGrinMedia` is now a *note* rather than a warning, and
  states the one thing that is still genuinely missing: a scattering GRIN medium has no volumetric
  **in-scatter** in the beam map — there is no straight chord inside a bending region to store —
  so the region's own glow is absent while everything it illuminates is correct.
  **…which is why the CPU branch had to learn the live window (0.195.1).** Back when `-beams`
  was CPU-only it *forced* `runSharedPhotonMap`'s CPU branch, and that branch called
  `renderPhotonCamera` with no progress hook — its GPU twin has passed a `SppProgress` since
  the feature landed. So the slowest, longest render in the engine was the only one with no
  live preview at all: frames were written to disk correctly while the window sat on the
  `preparing…` placeholder from start to finish, making a working render look identical to a
  hung one. The gather now routes through `cpuSppChunks` with a window-repainting
  `SppProgress`, exactly as the single-camera mode-`M` path already did, so each frame also
  *converges* on screen rather than appearing whole at the end. Bit-identical: seeding is per
  (pixel, **absolute** sample), so the chunk split cannot move the realization, and with no
  hook armed `cpuSppChunks` degenerates to the old single-shot call (verified 0 channel
  difference across all frames of a 6-frame curve, windowed vs headless). The stage is named
  throughout — `tracing photons…`, `building photon map…`, `building beam map…`,
  `frame k/N` — and the `exposure_lock` meter pre-pass, which runs *before* the group dispatch
  and so used to precede the existence of any window, now raises and titles one itself.

  **Every phase between the deposit and the first gathered pixel is now named and
  interruptible (v0.212.0).** The list above covered the deposit and the map build; on a
  `-beams` run in mode `M` the stretch after them was four further blocks that reported nothing
  and honoured no `-stop`, measured at tens of seconds each on a showcase `-n` and far more at
  flyby split budgets. They are now `building photon map`, `building caustic map` (split out —
  a stall in the caustic sort used to look like a stall in the global one), `splitting photon
  beams`, `uploading photon map`, `uploading caustic map`, `uploading photon beams`, and only
  then `gathering frame k/N`. The three uploads report real progress and rate, since each has a
  natural chunk loop to hang it on; the two builds report indeterminate, because a host counting
  sort has no cheap cursor.

  Each also carries a stop seam, and the seams are placed so an abandoned phase can never be
  half-consumed. The photon-map upload breaks on its existing 4 M-photon chunk boundary and
  still registers the partly-filled device buffer with `up.keep`, so it is freed on the way out.
  The beam upload skips the PCIe transfer *and* the `uploaded for the volume gather` log line
  when stopped, so the log never claims the device holds a beam BVH it does not; `dbm.nNodes ==
  0` is already the "no volume gather" sentinel, so nothing downstream needs a new case. The BVH
  build (`bvh.h`, shared by every BVH in the renderer) polls every 256 nodes and, when it bails,
  **turns the current range into a leaf rather than abandoning the tree half-written** — the
  result is a structurally valid BVH, merely a coarse one, so there is no window in which a
  partial tree could be traversed and answer wrongly, and callers that discard on stop and
  callers that do not are both safe. Verified on `gallery_rain` at `-beamsplitmax 80M`: a stop
  issued ~6 s into a build that takes 25.7 s uninterrupted ended it at 5.42 s, skipped both
  uploads, and wrote nothing.

  **The gather launch itself is both cancellable and observable from outside, via mapped pinned
  memory in each direction (v0.211.0, v0.213.0).** At 960x540 the gather's sub-chunk slice loop —
  which exists precisely to give `-stop` a seam — degenerates: `sliceOcc = 4 * 2048 * 128` exceeds
  the 518400-pixel frame, so the loop runs once and covers everything. There is **one kernel
  launch per spp**, and `518400 samples / 262144 threads = 1` means each thread runs exactly one
  grid-stride iteration, so a poll at the top of `kGather`'s own loop fires at the launch instant
  and never again (measured: it did not shorten a stop at all). Shrinking the slice is not the
  answer — a slice below `sliceOcc` leaves most of the persistent grid idle and cost an 11x
  throughput regression when it was tried. The fix is that **the host and the launch communicate
  through host memory the device can address directly**, because the launching thread is parked in
  `cudaDeviceSynchronize` for the whole launch and can issue no CUDA call: a `cudaHostAlloc(...,
  cudaHostAllocMapped)` word, its device address published to a `__device__` global by
  `cudaMemcpyToSymbol` before anything is in flight, and one 50 ms poller thread (RAII-joined).

  Backwards, that word is a **stop flag** the poller raises with a plain store, read where the
  time actually is — `dGatherPhotonBeams`'s beam-BVH walk every 64 nodes, and `dPhotonGather`'s
  bounce loop for scenes with no media. Stop latency across the three states: **879 s → 186 s →
  2 s**, at no measurable throughput cost (matched A/B: per-spp `3:17 / 3:24` vs `3:26 / 3:16`,
  both reaching 2 spp at 7:24, a between-arm difference smaller than the spread between two
  identical launches in one process). One correctness change is not optional: the post-launch
  `i < total` guard had to go, because a cancelled launch leaves the scratch film holding a sample
  for some pixels and not others while host-side `i` still reaches `total` — folding that partial
  chunk lays a ~1/spp brightness band across the frame, so it must route to `kFilmFold`'s existing
  discard path.

  Forwards, the same mechanism is a **progress counter**: `kGather` `atomicAdd_system`s each
  retired sample into a mapped word and the same poller reads it, which is the only way to see
  inside a launch the host cannot query. It is `#if __CUDA_ARCH__ >= 600` guarded — system scope
  is what makes it coherent with a concurrently-reading CPU thread, and it arrived with Pascal,
  while `all-major` fat binaries still include sm_50/sm_52. The reject path (`pdf <= 0`) retires
  too, since an undercount would leave the bar stalled just short of 100% at the end of a launch,
  which reads as exactly the wedged render the counter exists to rule out.

  **This progress deliberately shows no ETA, and that is why `StageProgress` has a second entry
  point.** `kGather` is wildly divergent — a ray crossing the thickest cloud gathers thousands of
  beams, one hitting the floor gathers none — so its retirement curve is steeply **convex**:
  measured on `gallery_rain`, 39% of the frame retires in the first minute, 10% in the second, 1%
  in the third. A trailing rate window fed that curve collapses to 145/s and predicts `~59:31
  left` with a minute remaining; a cumulative average under-reads instead, because extrapolating
  from the fast opening cannot see the tail. Neither estimator survives it. So `reportLive(text,
  done, total)` shares `report`'s caption and repaint/log throttles but prints percentage +
  elapsed + `in flight…` with no rate, and — the reason it is a separate function rather than a
  flag — **does not feed the rate window at all**, so the convexity cannot poison the per-launch
  `report` calls that follow. Arming is per launch (`progOffset` set to the frame's already-retired
  base immediately before, cleared to `-1` immediately after) and the clear happens **under the
  mutex the poller takes**, since a `StageProgress` mutates a shared Ticker that is not reentrant
  and the main thread is about to call it itself. Verified: the caption climbs `2% → 20% → 37% →
  47% → 50%` through one launch and on to `100%`, pre-gather phases keep uncorrupted rates, and
  two spp cost 7:03 against a 7:24 baseline.

  **Validated on `scenes/_rainbow_test.ftsl`** against the pre-existing A/B splat estimator,
  which is the only independent implementation of the same single-scatter trade. Getting the
  comparison honest took two corrections worth recording. First, `-mode M` *without* `-beams`
  renders the room correctly and the bow and the fog as literally nothing — that render is the
  before-picture for this whole bullet. Second, `-mode B -beams` at **one** camera is not a
  beams render at all: both the CPU and CUDA forward tracers gate the splat form on `nCam > 1`
  (`render_cuda.cu` ~13842), on purpose, since its entire reason to exist is amortising one
  flight over many cameras — so B with and without `-beams` came out identical to 3 s.f. and
  looked, misleadingly, like mode `M` was 20× too dark. Adding a second camera turns the gate
  on, and then the two agree: lit backdrop 182 vs 186 sRGB, bow arc 150 vs 155, floor 1.7 vs
  1.7, walls and sphere identically near-black, bow at the same angular radius with primary and
  secondary in the same colour order. The apparent darkness was real and correct — in that
  scene the sun is *horizontal*, so nothing but multiply-scattered fog light reaches the floor,
  and single scatter is exactly what both estimators drop. A third control confirmed no energy
  is lost on the surface path: with `sigma_t` cut to 0.0002 (near vacuum), mode `M` with and
  without `-beams` match to 0.1 sRGB in every region.
  **The map persists** via `photonmap_io.h` (`-savemap` / `-loadmap`), which is new in 0.195.0
  and is the piece that makes the amortisation argument complete rather than per-process: the
  file carries surface photons *and* raw beam crossings, so a volumetric flyby can pay the
  forward pass once, ever. Three decisions in that file are load-bearing. (1) It is a **header
  shared by both mode-M paths**, because the old copy was file-static inside `render_cuda.cu` —
  none of it is CUDA, so the flags were GPU-only by accident of placement, and therefore
  missing from exactly the configuration that needs them most (`-beams` forced the CPU path at
  the time, so the slowest forward pass was the one that could not be banked; `-savemap -beams`
  exited 0 and
  wrote nothing, silently). (2) It stores **no derived structure** — not the photon grid, not
  the beam BVH, and specifically **not the post-`splitLong` sub-beams**: `buildAuto` picks the
  split length from the radius it just solved, so persisting split beams would freeze the
  radius into the file and silently ignore a later `-beamk`. Storing the raw crossings keeps
  one file valid for any gather. (3) The magic went `FTPMP02` → `FTPMP03` with the beam block
  **appended**, so old caches still load and simply report zero beams; asking for `-beams`
  against such a file warns instead of rendering a volumeless image that looks like a beams
  render. Verified: save-then-load is bit-identical, `-beamk 128` on a reloaded cache re-solves
  the radius 4× larger, and the map phase drops 53.3 s → 2.6 s.
  **Beam noise reads as coloured streaks**, not grain — too few beams under a thin kernel are
  individually resolvable — so `-spp` is the wrong knob for it and `-beamcount` / `-beamk` are
  the right ones.
- **`beamgather.h`** — the Beam×Ray 1D estimator itself (Jarosz et al. 2011), split out of
  `photonmap_render.h` in 0.215.0 once it acquired a second caller. Mode `M` gathers it as its
  whole volume answer; mode `J` (UPBP, `bdpt.h`) gathers it as one of two competing techniques.
  `bdpt.h` could not reach the old home without including the entire photon-map renderer, and
  the reverse move — BDPT's merge into `photonmap_render.h` — is worse, since a UPBP merge
  weight is built from the *BDPT subpath densities* and belongs beside them. So the primitive
  sits at the level both callers already share, needing only `render.h` and `photonbeams.h`.
  The one addition over the original is `gatherPhotonBeamsW`'s **per-hit weight hook**, a
  template parameter (not a `std::function`: this is the innermost loop of the mode-`M` gather,
  and an indirect call per hit would be paid on mode `M`'s behalf for a feature it does not
  use). `BeamWeightOne` returns `1.0`, folds away, and `gatherPhotonBeams` is that
  instantiation under the historical name and signature — verified `cmp`-identical against the
  pre-split binary. Mode `J` needs a *per-hit* weight rather than a per-call one because the
  balance-heuristic ratio between "merge this beam" and "connect this vertex" depends on the
  merge geometry — `sinθ` above all — and so differs from hit to hit; by the time control
  returns to the caller the hits have already been summed.
- **`grin.h` — gradient-index media, and what the marcher owes the rest of the engine.**
  A `medium { ior "<expr>" }` is a region where rays integrate the Eikonal equation
  `d/ds(n·dr/ds) = ∇n` rather than travelling straight, via a symplectic march of step
  `ior_step`. One canonical CPU marcher (`grin.h`) and one device twin (`dGrinMarch`,
  render_cuda.cu, carrying its running Eikonal state in **double** so the two backends bend
  alike) serve every transport path; `grin::sceneHasGrin` / `DScene::hasGrin` gate the whole
  thing, so an `ior`-free scene never enters it and stays bit-identical. Marching does **not**
  consume a bounce: it is a pre-pass run before each bounce's `closestHit`, stopping when a
  surface is within one step or the ray has left every GRIN region.
  **Media along the curve are integrated by the caller, one straight sub-segment at a time
  (0.198.0).** `grin::march` became a no-op-hook wrapper over `grin::marchSegments(scene, ray,
  onSeg)`, which calls back for **every** straight piece of the walked polyline — the Eikonal
  steps *and* the straight jumps between regions — handing over that piece's origin, travel
  direction and true geometric length (`|(T/n)·ds|`, which equals `ds` only to first order).
  Returning true terminates the march at a distance along that piece, which is how a caller says
  "my medium scattered here". The device equivalent is a `DGrinMedia` struct parameter (no
  `--extended-lambda`, so no device lambdas) plus a `maxSteps` driving parameter.
  **This existed because the marched span was previously invisible to every media integrator.**
  `march` advanced the ray geometrically over the whole curved span and each tracer then sampled
  media only on the **short straight remainder** — so a medium that both scatters and carries
  `ior` lost essentially all of its scattering, in every mode, on both backends. `grin.h`
  documented that as an intentional approximation; measurement showed it was closer to deletion.
  The control is a pair of scenes differing by **one line**, a *constant* `ior "1.0"`: a
  constant index has zero gradient, so it bends nothing and only routes the medium through the
  marcher. They are checked in as `scenes/_grin_constior_a.ftsl` (no `ior`) and
  `scenes/_grin_constior_b.ftsl` (`ior "1.0"`); compare with
  `ftrace -in scenes/_grin_constior_{a,b}.ftsl -mode R -spp 2000 -o png/ci_{a,b}.png -hdr` then
  `python tools/pfmcmp.py png/ci_a.pfm png/ci_b.pfm`. At `-mode R -device gpu -spp 2000`
  (2.24 % per-render noise floor) the haze came out to a **0.885** whole-image and **0.779**
  centre-disc radiance ratio before the fix, auto-exposure moved 1.9×, and the haze visually
  vanished; after it, **energy bias +0.00 %, rel-RMS 1.86 % @8×8**, ratios 1.0000 whole /
  1.0007 centre / 1.0000 outside. "`ior "1.0"` must render identically to no `ior` at all" is
  the reference-free invariant this feature is validated against.
  **The decomposition is exact, not an approximation.** A spatial Poisson process with rate
  σ_t(x) is Markov in arc length, so first-collision sampling along a polyline decomposes
  exactly into "sample within segment 1; failing that, resample afresh in segment 2; …", and
  transmittance along a polyline is exactly the product of the per-segment transmittances.
  **Along the curve every medium goes analog, `-beams` or not.** That is not a shortcut, it is
  the thing a beam cannot be: `-beams` trades analog scattering for a straight chord the gather
  integrates, and there is no straight chord inside a bending region. Splitting the curve into
  its 10⁴–10⁵ Eikonal steps and storing a micro-beam for each would explode both the beam map
  and the gather cost to buy an answer analog sampling already gives exactly.
  **Each tracer uses the hook for what it actually needs**, which is why the hook is a callback
  rather than media code inside `grin.h` (whose only dependency is `scene.h`, while the media
  samplers are `Renderer` members): the forward tracer (`render.h::tracePhoton`, device
  `kTrace`/`kWfExtend` → `shadeStep`) samples a real collision and reports it as this bounce's
  medium event at zero travelled distance; `backward.h` draws **one** exponential flight before
  the march and lets accumulated arc length consume it (the exponential is memoryless, so this
  is identical in distribution to a per-sub-segment draw and ~10⁵× fewer RNG calls); the
  photon-map gather runs its **volume estimator per sub-segment**; the rest just accumulate arc
  length for the dielectric Beer-Lambert. In the GPU **wavefront** backend the march happens in
  `kWfExtend` but is consumed in `kWfShade`, so its outcome crosses the kernel boundary in two
  new `WFState` arrays (`gmMed`, `gmArc`) the way `hit` already did, and `kWfExtend` writes the
  advanced RNG back.
  **Modes `M`/`S` bend the camera ray too (0.198.0).** Their forward deposit had marched since
  GRIN landed, but `photonGather` / `dPhotonGather`, `photonGatherSub` / `dPhotonGatherSub` and
  `sppmVisiblePoint` / `dSppmVisiblePoint` all called `closestHit` on a straight ray — so a
  gradient lens bent the photons and not the view and **rendered dead flat**, while mode `R`
  lensed the same scene into a radial disc, and nothing warned. The surface half was a drop-in.
  The `-beams` **volume** half was not: Beam × Ray is a closest approach between two *straight*
  lines, so a curved camera ray has to be fed to it one Eikonal step at a time — which is what
  `maxSteps = 1` driving (loop until `!med->stepped`) is for on the device, and the per-sub-
  segment hook on the host. Exact, because radiance along a path is additive over its pieces and
  the running `thr` carries each piece's transmittance forward, which is precisely what
  `gatherPhotonBeams`' own per-beam `Tr_cam(0 → tCam)` assumes (it measures from the sub-segment
  start). One-step driving costs nothing extra — the marcher already redoes the
  region-membership test and `closestHit` on every iteration — and it keeps the bending maths in
  one function instead of a second copy of it. What is still missing is a scattering GRIN
  medium's own volumetric **in-scatter** in mode `M`: there are no beams inside a bending region
  to gather, and a curved beam primitive is a research problem, so the region's glow is absent
  while everything it illuminates, attenuates and scatters is now correct.
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
  **A component above 1 is FACTORED, not clamped (`upsample::gamutSplit`, 0.199.5).** All
  five model a reflectance, which is bounded by 1 by definition, so each of them used to open
  by clamping its input to `[0,1]` — and that quietly made `rgb` unusable for the spectral
  slots that are *not* reflectances. FTSL routes physically **unbounded coefficients** through
  the very same head: a dielectric's `absorb` (Beer–Lambert σₐ in 1/m), a `medium`'s
  `sigma_a`/`sigma_s`, a `hair`'s `sigma_a`, a metal's `substrate_k`, an `ior`. The clamp was
  total data loss on those: `absorb rgb 22.3 79.3 70.6` became `(1,1,1)`, i.e. a flat,
  colourless 1 /m absorption — 95 % transmittance across a 5 cm stone — so thirteen deeply
  saturated gems in `gallery_rain` rendered as *clear glass*, in every mode. `gamutSplit`
  divides the largest component out as a scalar magnitude, hands the remaining in-gamut colour
  to whichever fit the head named, and multiplies that spectrum back by the magnitude.
  Factoring rather than widening the fits is the right shape of fix because each fit is a
  *shape* solver over a bounded family and a magnitude is not a shape — dividing leaves a legal
  colour for the existing solver, and the multiply restores the level exactly, so hue and the
  relative depth between channels (the entire content of the authored number) survive. The
  magnitude is `max(r, g, b, 1)` and **the floor of 1 is load-bearing**: an in-gamut triple then
  divides and multiplies by exactly `1.0`, which is exact in IEEE, so every pre-existing scene
  upsamples bit for bit as before and only the destroyed case moves. The *device* twin
  (`stochJhCoeff`'s 64³ LUT over the unit cube) still clamps, which is harmless for its only
  inputs — image texels are in `[0,1]` by construction — but does mean a pattern-driven
  coefficient slot cannot yet carry a magnitude; see `known-issues.md`.
- **`upsample::fitMany` — the bulk path every image texture goes through (0.138.1).**
  A single Jakob–Hanika fit is ~40 Gauss–Newton iterations over the 95-sample basis:
  a few microseconds, negligible for a material, *seconds per megapixel* for a texture.
  Run serially per texel it was pure scene-load latency with nothing on screen —
  `scenes/gallery_rain.ftsl` (ten marble maps, 4.6 Mtexel total) spent **~45 s of its
  47 s startup** inside this one loop, which is what made `-explore` on it look hung.
  `fitMany` fixes it with two exact accelerations, in this order:
  (a) **deduplicate.** The fit is a pure function of the colour and an 8-bit source
  decodes through a 256-entry per-channel table, so equal texels are *bit*-equal
  `Vec3`s and a hash of their raw bit patterns collapses them with no tolerance and
  no quantisation. Real images collapse hard — the project's own procedural marble
  maps carry 96 to 80 k distinct colours over 0.05–1.05 M texels;
  (b) **thread the survivors** through `ft::parallelFor`. Chunk-stealing rather than a
  static split matters here because the per-colour cost is wildly uneven (a saturated
  colour never trips the residual bail-out and burns all 40 iterations).
  Net on that scene: **47 s → 2.9 s to parse, 4.5 s to a live `-explore` window.**
  Both steps are *optimizations, not method changes*: the coefficients are bit-identical
  to the serial loop, which `-checkupsample` check (i) asserts permanently over a
  synthetic 40 k-texel image built to exercise both the dedup hit and miss paths.
  `EnvMap::buildFromRgb` (`envmap.h`) has the same per-texel fit plus a 95-sample XYZ
  integral and is parallelised the same way — but its sin θ-weighted *mean* is left
  serial (one FMA per texel, i.e. free) so the summation order, and hence the emitter
  power and wavelength CDF derived from it, cannot drift with core count.
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
- **`bvh.h` sphere queries** — `Aabb::dist2To` (exact point-to-box squared distance) and
  `Bvh::traverseSphere(c, r, leafFn)`, an unordered no-early-exit traversal reporting every
  primitive in a node box meeting the ball. Built for M-GATHERAREA's geometric footprint: the
  gather divides by pi r^2 while collecting only the same-facing surface actually present, and
  no photon statistic can recover the difference (measured bracket on `fur_creature`'s belly:
  -33.6 % with the fiber gate, +46.8 % without). Candidates are a deliberate superset — a false
  candidate costs a test, a missed one costs correctness. `-checkspherequery` brute-force
  verifies the no-miss invariant.
- **`gafootprint.h`** — `gatherFootprintArea()`: the same-facing surface area actually inside
  a gather ball, measured geometrically. Fires `kDisc` stratified rays through the tangent
  disc and MARCHES each one, resuming past every hit until the ball's chord is exhausted, so
  every layer is counted rather than only the nearest — which is the whole difference from
  the probe in `photonmap_render.h`, and the reason that probe fails on a tangle holding
  ~600 curve segments. Marching also means there is no per-primitive-class code: triangles,
  spheres, implicits, curve segments and instances all work through the same
  `Scene::closestHit`, and a class added later needs no change here. An earlier per-class
  version had exactly two routines and silently returned ~0 area on `gallery_rain`, whose
  caps are implicit isosurfaces. Consumed by `-gafparea` (diagnostic) and `-gageom` (wires
  it into `gatherCoverageRaw` as the coverage, host only, off by default).
  **Validated**: exactly 1.0000 on a flat plane, 0.5986 on floor within 0.25 m of a wall
  against an analytic half-disc, 1.0079 on `gallery_rain`'s open ground. **Not used on
  fibers**: a coat's footprint measures ~2.9x pi r^2, so dividing by it would darken fur
  threefold — the footprint is right there and the DIVISION is the wrong estimator, because
  the density estimate assumes the ball meets one locally flat surface.
- **`roiboxes.h`** — `-roiboxes`: per-material measurement ROIs read off a pixel-centre
  primary-visibility pass, split into connected components and gated on purity/share.
  Depends on `camera.h` (it asks `genRay` which raster row is the top rather than
  assuming one), so it is included after it.
- **`filmToRgb8` auto-exposure cost** — the p99 anchor wants exactly **one** order
  statistic, so it uses `std::nth_element` (O(n), partitions in place) rather than a full
  `std::sort` of every pixel's luminance, and it builds the luminance array **only when
  that anchor is actually computed** — an absolute-EV render and a `camera_path` frame with
  a locked anchor both skip the pass and the allocation outright. `nth_element` guarantees
  the element at that index is the one a full sort would have placed there, so the anchor
  and every output pixel are **bit-for-bit identical** (verified against a pre-change
  render). This matters because `filmToRgb8` is the shared choke point for `writeFilm`
  *and* the live window, and once the window got its own repaint cadence it began running
  several times a second: at 480² the sort alone was the bulk of a ~40 ms repaint, now
  ~25 ms.
- **`-exposure-anchor <value|file>` — an exposure anchor that survives process exit.**
  The pre-existing lock machinery (`expAnchors`, `RenderCam.expGroup`, the `meterPlan`
  pre-pass, `-exposure-lock`) shares an anchor only among frames rendered by **one**
  `ftrace` invocation; it lives in a `std::map` that dies with the process. A sequence
  rendered one-frame-per-invocation (loom's `render_range`, a batch loop, a single frame
  re-rendered later) therefore meters every frame independently and can flicker. The new
  `ExposureAnchorFile` (`main.cpp`, just after `writeFilm`) closes that gap with a
  deliberately dumb medium — a text file holding one `%.17g` double. `resolve()` first
  tries to parse the argument **whole** as a finite positive double (the trailing-junk
  check is what stops a filename like `12frames.txt` reading as the number 12); failing
  that it treats it as a path, loading the anchor if the file parses and otherwise
  recording it as a `writePath`. So the *same* command line means "meter and save" on the
  first frame and "load and reuse" on every later one, which is what lets a caller loop
  without special-casing frame 0. The flag implies `-exposure-lock`, and must be resolved
  **before** the camera list is built (the lock changes how cameras are grouped).
  Write-back is RAII (`AnchorWriteback`, destructor reads `expAnchors[0]`) because the
  render dispatch has a dozen early `return`s and a save at any one of them would have
  been missed at the others. `-topng` accepts the same flag — it runs before the main
  argument loop, so it parses it itself and passes a `double*` down through
  `convertToPng` → `writeFilm`'s `lockAnchor`.
  **Why it was needed** (measured on `png/pastel_jack_ring`, 432 frames, static camera —
  only the ring rotates, so any global brightness change is by construction an artifact):
  the p99 anchor stepped **36.2%** between adjacent frames while the static background
  moved 2.2%, p95 0.68%, the median 0.59% and the whole-frame log-average 0.97%.
  The cause is **not** a bimodal histogram (an earlier revision of this bullet and of
  `known-issues.md` said so; it is wrong, and the fixes it implied are refuted). The tail
  has no gap — it *saturates* at 4.4780e+13, an emitter plateau over ranks 0.995–1.000 that
  is identical in both of the worst-pair frames. p99 sits *below* it in a sparse continuum
  with only ~**0.25% of frame per octave** above p95, so solving `area(L) = 1%` for `L` is
  ill-conditioned — ≈ **5 octaves per 1% of area** — and the 0.07-point area change the
  ring actually makes (1.1836% → 1.1128% above 4× p95) moves the level a third of an
  octave. Every fixed-rank statistic inherits this.
  **The shared anchor is therefore the fix, not a mitigation.** Nine studies measured six replacement families against stability *and*
  fidelity; none satisfies both, the stable ones landing 1.5–7.7 stops from today's
  exposure, because where p99 misbehaves its value is arbitrary and there is nothing to be
  faithful to. The bad case is also undetectable — pastel_jack_ring's local density at the
  anchor (median 1.23%) exceeds the 25th percentile of ordinary renders (0.87%) — so a
  gated regulariser is unavailable too. The needed information is temporal, not spatial.
  The narrow residual: a single **still** can still anchor on an atypical glint. See
  `known-issues.md` for all tables, plus a separate, genuinely fixable defect found the
  same way — the tone map hard-clips with no shoulder, which is *why* the anchor is forced
  to track the moving top of the histogram.
  **Repair path for already-rendered sequences** — `loom.stabilize_exposure(pngs)`
  (`tools/loom/loom/drive.py`) develops each `<frame>.png.ftbuf` once with `-topng` to
  read back the anchor that frame *would* have chosen (scraped from the
  `auto-exposure=<v>` line), takes the **median** across the sequence, and re-develops
  every frame at it. Median rather than "first frame wins" because on `pastel_jack_ring`
  frame 0's anchor sat at the **93rd percentile** — 0.502× the median, a full stop — so
  the obvious rule would have darkened the whole movie. It is a pure post-pass over
  checkpoints (milliseconds a frame, no photons re-flown), and `render_range(...,
  stabilize=True)` (the default) runs it after the last frame.
- **`-hdr` (a 32-bit float PFM beside `-o`)** — the escape hatch from the tone map, added
  because *measuring* off a PNG had quietly been wrong all along. An 8-bit sRGB image clamps
  at white, and a caustic is by definition the brightest thing in frame, so its core prints
  as `#FFFFFF` with all three channels **equal**: the clamp destroys exactly the two
  quantities a caustic study cares about, its **hue** and its **peak-to-screen ratio**.
  (Measured in `gallery_rain`: 596 of one cap's 22639 pixels were pure white — more than half
  the caustic's area — so the piece's colour metered as white no matter what the optics did,
  and a long chain of shape experiments had been ranked through that clamp.) `writeFilm` now
  calls the new `filmToLinear()` — the same denoised, scene-linear buffer `filmToRgb8`
  consumes — and dumps it as PFM, so the sidecar is an exact record of the PNG's *input*
  rather than a second reduction of the film. It is written on the periodic in-progress
  writes too, so a converging render can be metered while it runs. **Scene-linear, not
  exposed**: peak/median ratios and chromaticity are exposure-invariant, so renders shot at
  different stops stay comparable. PFM's raster order (left-to-right, **bottom-to-top**) is
  the film's own row order, so unlike the 8-bit path it needs no vertical flip.
  **Corollary nobody expected: the 8-bit clamp had been the only outlier rejection those
  measurements had.** With it gone, a spectral rig metered a piece that throws *no* caustic at
  peak 1214× on three cells — mode `D` carries one hero wavelength per sample, so a rare
  specular path deposits a huge *monochromatic* spike that a box average (linear, zero-mean-
  symmetric) cannot touch. So float metering must be paired with `-fireflies 3`, which is safe
  for genuine caustics precisely because a caustic is never isolated: the shipped axicon reads
  peak 14.68× bit-identically with and without it.
- **`denoise.h`** — `-denoise`, an edge-aware à-trous (SVGF-style) filter for Monte-Carlo
  speckle. It runs inside `filmToRgb8` on the **linear** image and *before* the p99
  auto-exposure anchor is measured (so a firefly can't set the exposure); because
  `filmToRgb8` is the single choke point for both `writeFilm` and the live window, the
  preview shows exactly what the file gets. **It filters CHROMA ONLY and leaves luma
  bit-identical** — a deliberate design choice, not a weak default: spectral paths carry
  one wavelength, so wherever the hero-wavelength bundle is unavailable (participating
  media, dispersive refraction) the variance is overwhelmingly chromatic, while luma is
  already converging at 1/spp. Measured against an 8000 spp reference of `gallery_rain`,
  filtering luma made the image *worse* (−2.4 dB; it cannot distinguish a wire or caustic
  rim from a noise spike), so `-denoise-luma` exists but defaults off. Three invariants are
  pinned by `-checkdenoise` and were each violated by a working draft: total luminance is
  conserved **exactly** (a plain bilateral gather is row- but not column-stochastic and ate
  30 % of the frame — fixed by a symmetric geometric-mean tolerance plus a luma *scatter*);
  a constant image is a fixed point (needs **half**-sample edge mirroring — whole-sample
  has fixed points at the edges, which double-counts and leaks 0.06 %); and chroma-only
  leaves per-pixel luma bit-identical (needs the luminance-preserving gamut projection in
  `fromYcc`, since clamping a negative channel to 0 *adds* light). Chroma is *stored* as a
  ratio to luma (scale-free across ~4 decades) but **averaged luma-weighted**, `Σw(R−G) /
  ΣwY` — averaging the ratios instead lets near-black pixels with wild ratios dominate,
  which turned per-pixel speckle into coherent purple/orange blobs. That weight guards only
  one tail, and **the other one bit** (v0.306.1): nothing stopped a very *bright* tap, which
  dominates numerator and denominator alike so the result collapses to its hue. The 3×3
  **median** guide — added to stop the *luma* scatter printing the kernel as a lattice — is
  what made it fatal, because a median is designed to make a lone spike look ordinary to the
  edge-stop, so the spike was accepted at full weight while the sums still used the raw `Y`.
  **Each defence disarmed the other**, and the same lattice reappeared in chroma. The weight
  is now capped at `trust` (4.0) × the tap's own median luma, which is a no-op for any
  non-outlier; swept against the 8192 spp reference it is simultaneously the chroma-error
  minimum (−8.2 %) and the PSNR maximum (+0.26 dB), so it removed a bias rather than trading
  one off. `levels` and `chroma` were swept against the reference too: the optimum is a
  plateau at **2–3 levels**, not SVGF's 5, because with no luma term holding the edges a wide
  chroma support bleeds colour across material boundaries (by 7 levels it is a net loss).
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
  renders it; the port is tracked in `known-issues.md`. The preview rasteriser DOES honour
  `emitPat` (and `reflectPat`) since 0.135.0 — see *Preview shading model* below; ignoring
  it made a masked emitter preview as one flat glowing slab.

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

  **loom emits both blocks (2026-08-05, loom-only).** The port ran one way for a while:
  ftrace had the datatypes, but loom's own `Grid`/`Scatter` could still only be sampled in
  Python (a `GridField` is a `Signal`, so it bakes to one number per frame). `grid(X, Y)` —
  a query in ftsl's *spatial* coordinates — now builds a `GridSample`/`ScatterSample` term
  (`loom/spatial.py`) that emits the table call, with the companion `grid`/`scatter` block
  collected automatically by `Scene.add` and its values baked at the emit clock. Its
  `eval_np` is a vectorised port of `patGridSample`/`patScatterSample`, so loom's raster
  preview and the render agree; a dataset's placement `Transform` folds into the emitted
  *coordinates*, since these blocks (unlike geometry) carry no transform of their own. What
  ftrace cannot express — vector-valued samples, cubic interpolation, a throwing
  out-of-domain policy, > 4 axes — raises in loom rather than emitting something that means
  something else. See `tools/loom/DESIGN.md` §6.

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
  float path's bits. (2) It is hooked **only at final-consumption sites** — never inside
  compilation — because `patternSubstitute` splices programs together and two
  pre-optimized fragments' register indices would collide. (3) It is belt-and-braces
  conservative: an op whose arity it cannot determine makes it bail, the re-emitted program
  is re-simulated for stack balance, and 6 full-variable probe points are bit-compared
  (`memcmp`) old-vs-new before the rewrite is accepted — any mismatch keeps the
  original. Since the ops are appended at the enum end, the `VarX..VarV` range tests
  (`patternHasFreeVars`, the free-variable scans) are unperturbed; all **three**
  evaluator switches (host `patternEval`, device `dPatternEval` / `dPatternEvalF`) carry
  the two cases, and the `PatNode → PatNodeF` upload conversion is memberwise so a
  register index ≤ 32 survives the float trip exactly.

  **Vector-valued POV noise (`PatOp::DNoise` / `PatOp::DTurb`, v0.158.0).** Exact ports of
  POV-Ray's `DNoise` (gradient-vector noise) and `DTurbulence` (its octave fBm) live beside
  `povNoise` in `pov_noise.h`, sharing its tables — the 534-double `g_povRTable` was in fact
  *sized for* DNoise all along (267 conceptual entries = 255 hash range + 12 for the three
  8-double-stride component records read at `mp`, `mp+8`, `mp+16`). The VM stays scalar: one
  op per **component**, with the component index (0/1/2) riding the node payload `a` exactly
  like `PovFn`'s function id, surfacing as `dnoisex/y/z(x,y,z)` and
  `dturbx/y/z(x,y,z,octaves,lambda,omega)`. `DTurb` is its own op (not authored from
  `DNoise`) because the VM has no loops, so the octave sum cannot be expressed in-language.
  Both ops are appended at the enum end (the `VarX..VarV` scans are unperturbed), registered
  in `patOpStackEffect` so CSE can rewrite through them, and keyed by `a` in the hash-consing
  so `dnoisex+dnoisey` of one point stay two nodes while a repeated identical call collapses
  to one. The fp32 device VM (`dPatternEvalF`) promotes to double internally and demotes the
  result — `PovFn`'s established contract — so all three backends run the identical lattice
  arithmetic. `-checkvnoise` pins the port in six mutation-tested sections; the load-bearing
  one is the cross-invariant that POV's generator-1 *scalar* noise is definitionally
  `DNoise[0] + 0.5` clamped, bit-exact — anchoring the new code to the already-trusted
  scalar port with zero tolerance.

  **Cellular / Worley noise (`PatOp::Worley`, v0.159.0).** Self-contained in
  `src/worley.h` (host+device, the same `__CUDACC__` guard idiom as `pov_noise.h`): one
  feature point per integer lattice cell, jittered by a murmur3-finalizer hash chain
  (full avalanche, so neighbouring cells are uncorrelated and CPU/GPU agree
  bit-for-bit). One VM op serves four surface spellings — `worley` (F1), `worley2`
  (F2), `worleyd` (F2−F1), `worleyid` (per-cell random in `[0,1)`) — with the **output
  selector riding the node payload `a`** (the `DNoise` component-index convention),
  while the **metric is a runtime operand** (rounded, clamped to 0 Euclidean /
  1 Manhattan / 2 Chebyshev) so it can itself be an expression. The search is *exact*,
  unlike the common 3×3×3 neighbourhood (whose F1 can be beaten by a cell two rings
  out): concentric Chebyshev rings with the early-out "ring r's cells all lie ≥ r−1
  away in every supported metric (each metric ≥ the Chebyshev point distance), so once
  r−1 ≥ F2 no farther ring can improve" — average cost the same 27 cells, worst case a
  ring or two more; the hard ring cap 8 is unreachable math and only guards NaN inputs.
  Wiring follows the O2 checklist verbatim: enum appended at the end, `patOpStackEffect`
  arity 4, payload keys the CSE hash-consing (`worley+worley2` of one point stay two
  nodes), fp32 device VM promotes/demotes around the double core. `-checkworley` runs
  seven mutation-tested sections; the load-bearing one compares F1/F2/**and id** against
  a fixed ±6-block (13³) brute force with no early-out and independent floor/min logic —
  provably sufficient since cells beyond ring 6 lie ≥ 6 away while F2 within the
  always-populated 3×3×3 block is ≤ 6 (its Manhattan diameter) — plus metric-ordering
  (order statistics are monotone), 1-Lipschitz continuity straddling cell walls (the
  floor-vs-truncation trap), compile/reject, CSE, and distribution sanity.

  **Mean curvature (`PatOp::VarCurv`, O3 stage 1, v0.161.0).** Every other noise in the
  VM is **stationary** — a function of position, so translating an object slides the
  pattern off it. `curv` is the first *non-stationary* driver: a per-hit property of the
  SHAPE, so `noise * mask(curv)` follows the geometry and keeps following it when the
  object moves. It is the mean curvature H = (k₁+k₂)/2 in 1/length, carried on
  `HitRecord::curv` and threaded into `PatCtx` by `patCtxFromHit`.

  Unlike the ops above it is **not a function but a leaf variable**, and it lives
  *outside* the contiguous `VarX..VarV` range that `patternHasFreeVars` scans (it is
  appended at the enum end, per the O2 checklist, so that range stays unperturbed).
  That combination is the trap: it must therefore be named **explicitly** in both
  `patternHasFreeVars` *and* `patOpStackEffect`. Omitting the latter is not the local
  no-op it looks like — an unknown arity makes `patternOptimizeCSE` bail on the **whole
  program**, so a single `curv` anywhere would have silently switched CSE off for the
  entire surrounding expression. (That bug was real and was caught only by the
  self-test's CSE section.)

  Where the number comes from, per primitive: `sphere` is analytic (±1/R); a `curve`
  round-cone is a surface of revolution, k = 1/R and ~0 along the axis, so H = 1/(2R) at
  the interpolated radius; a **mesh** is per-face in `Tri::finalize()`, H = ½·trace of
  the shading-normal differential restricted to the tangent plane. That trace must be
  taken in the **dual basis** — {e₁,e₂} is not orthonormal, so the 2×2 Gram matrix has to
  be inverted. A quad/flat facet, a flat-shaded mesh (no normal field to differentiate)
  and an isosurface all honestly report 0 rather than guess.

  Two invariants the rest of the system has to respect. **Sign** is relative to the side
  being shaded: every intersector negates `curv` when it flips the normal toward the ray,
  so a bulge is always + and a pit always −, and the same sphere reads +1/R outside and
  −1/R inside. **Scale**: H is 1/length, so the instance path rescales it (`scale 0.5`
  doubles it); a non-uniform scale is approximated by |det|^(1/3).

  `-checkcurv` is the self-test, nine sections, mutation-tested. Worth recording *why*
  section 6 (basis independence) is built on a **cylinder** and not the sphere it started
  on: a sphere is **umbilic**, dn is a multiple of the identity there, so the naive
  per-edge trace that omits the dual basis returns the correct 1/R for *every* basis. The
  omission is only observable on a non-umbilic surface — the sphere sections passed the
  mutant happily.

  One load-time guard falls out of this: an **emission** pattern may not read `curv`
  (`checkEmitPatsSupported`). Emission is read from both sides of transport and MIS
  combines them, but `Emitter::samplePoint` has no curvature to report, so the sampled
  side would read 0 against the hit side's real value — biased, not merely noisy. Same
  reasoning that already refuses emission patterns on sphere/cylinder/spot/env emitters.

  **Enclosure (`PatOp::VarCavity`, O3 stage 2, v0.162.0).** `curv` made a pattern follow
  the SHAPE of the surface it sits on; `cavity` answers the question `curv` structurally
  cannot: how ENCLOSED is this point? It fires a short hemispherical probe of radius
  `scene { cavity_radius }` and returns the blocked fraction — 0 on an open plane, ~0.5
  in a right-angled interior corner, →1 down a crevice. The two are complements, not
  alternatives, and the difference is **locality**: curvature lives in the second
  derivative of one normal field, so a right-angled corner between two FLAT faces reads
  curv = 0 on both of them even though that is exactly where dirt collects, and no
  differential property of the floor could ever produce the dark ring where a *ball*
  rests on it. Conversely `cavity` is blind to gentle convexity — a bare sphere in an
  empty room reads 0 however curved it is — so edge wear still wants `curv`.

  The direction set is a **fixed** cosine-weighted golden-angle Fibonacci hemisphere,
  not a random draw, and that is the load-bearing decision. A pattern input is read many
  times per pixel by different tracers (camera hit, light-sample hit, MIS's other
  estimator) and every read must agree, or the *material* becomes a variance source that
  MIS smears rather than averages. Deterministic makes `cavity` a true function of
  position: noise-free, bit-comparable CPU vs GPU, stable frame to frame. The price is
  **banding** into at most N+1 levels, which is why the documented idiom multiplies the
  mask by fBm — the mask says where dirt MAY settle, the noise says how much did.

  It is the only pattern input that spends **rays**, so it is gated **twice**:
  `Scene::needsCavity` (scene-wide, one integer compare that is false for essentially
  every scene) and `Material::readsCavity` (per-hit, keyed off `Hit::matId`). The second
  gate is not an optimisation but a correctness-of-cost requirement: `patCtxFromHit` runs
  for EVERY patterned material, so a scene-wide-only flag would charge N occlusion rays
  to every unrelated noise-textured surface in a scene that used `cavity` once.
  `ftsl::setupCavity` sets the flag from `materialFreeInputs` and then lifts it through
  `mixChildren` to a fixed point, so a `mix` inherits it from a layer (and from a nested
  mix's layer). The value is computed lazily on first read and cached in
  `Hit::cavity`/`cavityDone` — which is why those are the only two `DHit` fields with
  default member initialisers: no intersector writes them, and `dVertHit()` / the scratch
  hits inside `occluded()` would otherwise hand the cache garbage.

  The device probe (`dCavityAt`) **inlines** the host's exact Duff ONB rather than calling
  `d3onb` (wrong vector type) and works in `double` throughout even where `Real` is
  `float`, so the two halves stratify identically; verified numerically (deterministic
  mode-W CPU vs GPU max Δ 6/255, mode-D auto-exposure identical to all digits). The
  raster previews pass 0 — they have no BVH at all, by construction (`known-issues.md`).

  `-checkcavity` is the self-test, ten sections, mutation-tested four ways. Its anchor is
  analytic and **tight** rather than sampling-limited: a horizontal ceiling at height h
  blocks exactly the cap cosθ > h/R, whose **cosine**-weighted measure is 1 − (h/R)² (it
  would be 1 − h/R under uniform weighting, and the test asserts it is *not* that), and
  because the sample set stratifies u = (i+½)/N at bin centres with cosθ = √(1−u) the
  discrete count matches to 2e-3. Mutation testing found a real hole in it: §6 originally
  set `h.ng = h.n`, so replacing `orientedGeoN(h)` with `h.ng` SURVIVED — the fix was a
  `hitAt2(p, n, ng)` helper that holds the geometric normal fixed while varying only the
  shading normal.

  **Distance to a named object (`sdf`, O3 stage 3, v0.163.0).** `curv` reads the surface a
  point is ON and `cavity` reads how enclosed it is; neither can answer "how far is this
  point from THAT object", which is non-local in a way no per-hit property can be, and is
  about a *named* object rather than about everything. The whole feature is **a bake plus a
  header**: `sdf "halo" { object "ring" res 128 pad 0.65 }` measures the signed distance to
  one `mesh` onto a lattice and registers it as an ordinary `PatGrid`, read as
  `grid:halo(x, y, z)`. **Zero new VM opcodes, zero new device code** — which is exactly why
  it works, unchanged, at every site `grid:` already works: patterns, material slots,
  `isosurface` leaves, `camera_curve` drivers, and *medium `density`/`ior` programs*. That
  last one is the case `curv`/`cavity` structurally cannot reach: a volume has no normal, no
  UV and no hit point, so a spatial field is the only input it can take.

  The bake (`meshvox::bakeSignedDistance`) is three stages, and each one is a deliberate
  choice over a cheaper wrong alternative:
  1. **Sign** from `voxelizeSolidInto` — the signed-crossing (generalized winding) scanline
     extracted out of `voxelizeSolid` for this. A multi-body or self-intersecting model
     therefore reads as its **union** instead of hollowing out where the bodies overlap. Sign
     is the only part of an SDF a mesh can be genuinely ambiguous about, and this is the one
     place in the codebase where that has already been argued out and measured (the fog
     bound).
  2. **Exact narrow band**: every triangle visits the samples within 2 voxels of its AABB
     and records the exact point-triangle distance *and which triangle produced it*, via
     `pointTriDistSq` (Ericson §5.1.5, full Voronoi-region test — the "project and clamp the
     barycentrics" shortcut is wrong on obtuse triangles, i.e. on most imported geometry).
     Parallelised over z-slabs after bucketing triangles into the slabs they touch, so no
     two threads share a sample and no atomics are needed.
  3. **Propagation** by Bridson's closest-**triangle** sweep (SDFGen): 2 rounds × 8 octant
     sweeps, each sample adopting a neighbour's closest triangle if re-measuring against it
     wins. Distance stays exact-per-triangle everywhere; only the *search* is approximate.

  Stage 3 was originally the exact separable EDT that `featherGrid` uses, seeded with the
  narrow band's exact squared distances. That is **not a distance transform**: F&H computes
  min over q of (|p−q|² + f[q]), which is only correct for BINARY seeds — with exact seeds
  it adds *squares* where distance adds *lengths* (the two legs are collinear in the case
  that matters, so the answer wants (|p−q| + d_q)², not |p−q|² + d_q²). It read a 71.5 mm
  torus tube as 47 mm: smooth, plausible, and 34 % short. Propagating the closest
  *primitive* rather than a distance is what makes the composition legal. The trap is
  documented at the function with the measured numbers.

  **Load ordering is forced to split in two**, because the name must resolve before Pass 1c
  compiles patterns but the samples cannot exist before Pass 3 loads geometry.
  `Builder::reserveSdf` (Pass 1a′) registers an empty ndim-3 `PatGrid` under `gridIndex_`, so
  `grid:halo(x,y,z)` type-checks and any other arity is a compile error; `fillSdfs` (after
  Pass 3) resolves the `MeshGroup`, bakes, and appends to `Scene::dataPool`. The gap between
  them is real, so `rejectUnbakedSdf` guards the only two places a pattern is *evaluated
  during the load* — a procedural `texture { rgb … }` bake and a `camera_curve` driver — and
  makes them load errors. Letting them through would return `patGridSample`'s empty-table 0,
  and 0 in a distance field means "exactly on the surface": the most confidently wrong answer
  available.

  One subtlety worth the comment it carries: `SdfBake::d` is written in **PatGrid order (axis
  0 outermost)**, not the x-fastest order the voxelizer and sweeps work in, so the array can
  be appended to the pool verbatim. The transpose happens once, in stage 4. Getting it
  backwards is invisible in every aggregate statistic — min, max, histogram, "deepest
  interior" all agree — and shows up only as a field that is plausible but rotated, which is
  why `-checksdf` §7 samples through `patGridSample` and separately asserts that the
  transposed reading would fail loudly.

  `-checksdf` rests on one choice: **the test geometry is an axis-aligned box**, exactly
  representable by 12 triangles and with a closed-form signed distance, so the bake and the
  analytic answer are the same number at all ~40 000 lattice samples (checked to 2e-6, i.e.
  float storage precision) rather than merely close. A sphere would have buried every real
  defect under its own tessellation error, R·(1−cos π/N). Nine sections: the anchor, the
  propagation reach reported separately for samples > 3 voxels out, union semantics on
  overlapping boxes (a parity voxelizer hollows the overlap and dies here), a disjoint pair
  for gap-crossing propagation, lattice geometry, `pointTriDistSq` against an independently
  written exact routine on obtuse triangles, the memory order, the loader round trip through
  an in-memory OBJ via `assetbytes::Overlay`, and the six refusals.

  Known limitation, logged: *inside* an overlapping union the magnitude is the distance to
  the nearest triangle, which may be a buried one, so it is not the union's own interior
  distance. Exterior distances are provably exact (a buried triangle can never be nearer to
  an exterior point than the boundary the segment to it crosses), and the exterior is what
  the feature is for.

  **The non-stationary idiom, and the CSE it forced (O3 close-out, v0.164.0).** `curv`,
  `cavity` and `sdf` are three answers to one question, and the payoff is not any of them
  alone but what they let you do to a noise call: *every argument of one is an ordinary
  expression*, so a scene-aware field can change **what kind** of noise appears where
  rather than merely how much of it shows. The division of labour that makes this legible
  is **gate** (where the effect may appear at all — `curv`, `cavity`, cheap and local)
  versus **field** (what the noise there should look like — an `sdf`, smooth, unbounded,
  about a named object). Documented in REFERENCE.md → "Putting them together —
  non-stationary noise", cross-referenced from FTSL.md §6, worked in
  `scenes/pattern_nonstationary.ftsl` (one `sdf` around a hovering bead driving three
  surfaces with three different gates).

  Two of the three traps written up there are properties of *this implementation*, not
  folklore, and both were found by reading the code rather than by rendering:
  `povDTurbulence` truncates its octave count (`int oct = (int)octaves`) and Worley rounds
  its metric to `0..2`, so a field-driven value of either **pops at integer contours**
  instead of fading. The correct idioms are to fade the extra octave's *amplitude*
  (centred on zero, so detail arrives without shifting the mean) and to drive `metric`
  with a step. The third trap is analytic: `noise(k(p)·p)` has local frequency
  `k + p·dk/dp`, which is not the `k` requested, depends on distance from the *origin*,
  and shears the pattern into streaks along `∇k` — so crossfade two **fixed** frequencies,
  `mix(noise(3*p), noise(16*p), t)`.

  A fourth lesson came out of the demo render rather than the code, and is now in both the
  scene's header and the REFERENCE section: a correct crossfade can still be *invisible*,
  and for two independent reasons.

  * `mix` of two independent noises has **half their variance**, so the half-way band is a
    flat grey smear. The stretch belongs **after** the blend
    (`smoothstep(0.38, 0.62, mix(a, b, t))`), which re-normalises it at every `t`, not on
    each band before it, which does not.
  * The ramp has to be **matched to the distances that actually occur**. This was measured,
    not guessed, by rendering the raw `grid:bead` value as a reflectance and dividing (in
    *linear* light — the tone map otherwise flatters every ratio toward 1) by a constant-1
    render of the same frame. The answer: the bead hovers 0.30 m up, so no floor point is
    nearer than 0.21 m and the far corners are only ~0.9 m away — the entire floor lives
    inside a single 2:1 span. The 0.35..0.85 m ramp that "obviously" spanned the room in
    fact held the whole visible floor between t = 0.9 and t = 1.0, showing the fine band
    alone and looking perfectly stationary. 0.26..0.58 m reads immediately.

  Related, and already documented in the scene: `cavity_radius` larger than the feature it
  is probing (0.22 m on a 0.22 m plinth) saturates the mask to a flat 1. The general shape
  of both mistakes is the same — a field-driven mask is only as good as the match between
  its transfer curve and the range the field actually takes on the surface being shaded.

  The idiom then ran straight into a real cost. The expression language has **no local
  variables** (no `let`, and a pattern cannot name another pattern), so a field that gates
  one term and steers another must be spelled out at every site — six times in the worked
  scene's `wear` pattern. Every one of those was a separate trilinear lattice fetch,
  because CSE had never been applied to material patterns at all (it was wired only at
  `addFunctionLeaf`, for isosurface `function` exprs) and, worse, `patOpStackEffect`
  returned `false` for `Grid`/`Scatter`, which would have made the pass bail on the whole
  expression even if it had run. So documenting the idiom honestly required making it
  cheap:

  * `patOpStackEffect` and `patternOptimizeCSE` take an optional `const PatTables*`. A
    table op's arity is the **table's** `ndim`, which is nowhere in the program, so it can
    only be answered by whoever knows which tables the program will be evaluated against.
    The clamp applied there is deliberately the *evaluator's* (`[1, PAT_ND_MAX_DIM]`), not
    the header's nominal value: what has to match is the actual pop count.
  * §6's probe contexts now bind those tables. Unbound, `patternEval` abandons the whole
    program at a `Grid` and returns 0.0 — so both sides would have compared equal no
    matter how wrong the re-emission was, i.e. the safety net would have silently stopped
    catching anything on exactly the programs it was newly being asked to cover. The probe
    rows also gained `curv`/`cavity` columns for the same reason: a variable left at 0 is
    a variable the comparison is not exercising.
  * `Builder::optimizePatterns` runs **last** in `build()` — after `setupCavity`,
    `warnCurvOnFlatGeometry`, `checkEmitPatsSupported` — over every `Scene::patterns`
    program and every `Medium::density`/`ior`. CSE preserves the first occurrence of every
    op, so those analyses would still be correct if reordered, but running it last means
    no future analysis has to know that `LdReg` exists.
  * `addFunctionLeaf` now passes the tables too, so a `grid:`-sampling field formula is
    optimized as well. It stays a separate call site because the program is about to be
    appended to a *shared* node pool holding one leaf after another, and the pass requires
    a single-rooted program.

  Measured on `scenes/pattern_nonstationary.ftsl` via the new opt-in `FTRACE_CSE_DEBUG`
  (same shape as `FTRACE_CHUNK_DEBUG`; the report is the only way to see a pass that is
  bit-identical by construction): `pattern 2: 126 -> 71 nodes, 6 -> 1 table sample(s)`.
  Table samples are reported separately from nodes because a `grid:` fetch is 8 pool reads
  and a trilinear blend, so the node count alone understates it. `-checkgrid` §(h) pins
  all three behaviours — shrinks and emits `LdReg` with tables in hand, stays bit-identical
  over 64 probe points, and **declines**, leaving the program untouched, without them.

  **Anisotropic Gabor noise (`PatOp::Gabor`, O4, v0.165.0).** `src/gabor.h`,
  `gabor(x, y, z, f, wx, wy, wz)` — arity 7, the widest op in the VM. The O3 write-up had
  just finished explaining why you must *not* make a lattice noise's frequency a function
  of position (`noise(k(p)·p)` has local frequency `k + p·dk/dp`, so it shears with
  distance from an arbitrary origin). O4 asks for the same thing one axis over — noise
  *steered* along a direction field — and the obvious `noise(R(p)·p)` fails for exactly
  the same reason: the Jacobian is `R + (dR/dp)·p`, and the second term again grows with
  `|p|`. There is no way to fix that inside a lattice noise, because a lattice noise's
  orientation *is* its grid. So the honest answer to O4 is a different construction, one
  where orientation is a **kernel parameter**: a Gabor kernel only ever sees the offset
  from its own centre, which is bounded by one cell, so a varying direction leaves a
  residual bounded by the local turning rate rather than by position. That is measured,
  not asserted: `-checkgabor` §6 runs a *varying* direction field and finds the same
  local frequency at the origin and 4000 units out.

  The TODO's premise for O4 was "needs a per-hit tangent frame". It doesn't — and `Hit`
  has carried one since C6 (`tangent` + `bitangentSign`) anyway. Exposing `tx ty tz` was
  considered and rejected: it largely duplicates the already-exposed `u`/`v`, and it would
  only have fed the coordinate-rotation idiom that the paragraph above rules out. The
  useful thing to expose was the *primitive*, not the frame.

  Four design choices are worth recording because each replaces a standard approximation
  with something exact:

  * **A compactly supported C² envelope, `(1-r²)³` of radius exactly one cell**, instead
    of Lagae's Gaussian truncated at 5% of its peak. The textbook 3×3×3 search is
    therefore *approximate* — it silently drops the tails — whereas here a cell two rings
    out is >1 away in the infinity norm, hence outside every kernel it can hold, so
    3×3×3 is **mathematically exact** (`-checkgabor` §1 pins it against a ±4-block brute
    force, bit for bit, and separately counts impulses that a 3×3×3 would have missed:
    zero). It also removes the `exp` entirely and lets the support test run on the
    *squared* distance, which rejects ~85% of candidates in three multiplies.
  * **Per-cell Poisson(λ) points, uniform in the cell.** The union of independent Poisson
    processes on disjoint regions is a homogeneous Poisson process, so the impulse set is
    not merely jittered-on-a-grid — it genuinely has no grid, and the noise is stationary
    under *arbitrary* translation, not just integer ones. §7 measures that, and asserts
    the contrast against lattice value noise (whose variance differs 8× between
    on-lattice and mid-cell samples) so it cannot pass vacuously.
  * **A random phase per impulse** (Lagae & Drettakis' phase-augmented form). This is what
    makes the normalisation *analytic*: `E_φ[cos²(θ+φ)] = ½` pointwise, so Campbell's
    theorem gives `Var = λ·E[w²]·½·∫E² = 0.28566907` with **no dependence on `f`** — which
    matters here far more than in an offline tool, because `f` is a runtime operand and a
    zero-phase kernel's variance would drift as it moved.
  * **Its own cosine.** `patGaborCosTurns` reduces in *turns* (`t - floor(t+0.5)` is
    exact), folds to `[0, π/2]` and evaluates a Taylor series to `x²²`. libm's `cos` is
    not correctly rounded and CUDA's differs from the host's by up to 2 ulp, which would
    break the "a pattern evaluates bit-for-bit identically on every backend" contract that
    `patWorley`/`povDNoise` keep. §2 verifies it against libm to 8e-16 — and had to be
    *fixed* to do so: comparing against `cos(2π·t)` at 4096 turns measures libm's own
    argument error (~3e-12), 300× the discrepancy being looked for, so the reference
    reduces in turns first and large arguments are covered by an exact-periodicity check.

  Wiring is the O2 checklist verbatim (enum appended at the end, one case in each of the
  three VMs with the fp32 one promoting/demoting around the double core, `patOpStackEffect`
  arity 7, no payload so CSE shares identical calls by structure alone). Worked example
  `scenes/pattern_gabor.ftsl`: isotropic band-pass speckle, brushed metal, **wood end
  grain** (steer radially about a vertical axis and the bands close into growth rings — one
  expression, no polar remap, and the degenerate direction at the axis falls back to
  isotropic, which is what a real pith looks like), flow-aligned fibre, latitude striation.
  Two independent anisotropy knobs are worth stating outright, because authors reach for
  the wrong one: the **direction** chooses which way the field oscillates (streaks run
  *perpendicular* to it), while **scaling the input coordinates unevenly** stretches the
  kernels themselves.

  This also lands most of O8 as a side effect: the spectrum is a narrow band the author
  picks rather than everything up to the lattice Nyquist, so it is the one noise here that
  minifies gracefully.

  **Blue-noise / Poisson-disk placement (`PatOp::BlueNoise`, O5, v0.166.0).**
  `src/bluenoise.h`, `bnoise/bnoise2/bnoised/bnoiseid(x, y, z, r)` — the same four slots as
  `worley` (F1, F2, F2−F1, per-point id) over a **different point set**: one with a
  guaranteed minimum separation `r` in cell units. Worley's sites are a jittered lattice,
  one point per cell placed uniformly inside it, so two of them can be arbitrarily close
  (both jitter to the shared wall — measured closest pair 0.026 in a 70³ block) while
  elsewhere the lattice leaves holes. Threshold F1 to draw spots and that clumping is
  instantly legible as computer texture. It is also **unrecoverable downstream**: evenness
  is a property of *where the points are*, so no filter over `worley`'s output puts it
  back. Hence a new primitive rather than a mode.

  The obvious construction can't be used. Dart throwing (Bridson) is inherently
  **sequential** — a dart's acceptance depends on every dart accepted before it — so
  answering "nearest point to `p`" would mean simulating the whole plane, against a
  requirement of O(1) at an arbitrary point of an *unbounded* domain, no bake,
  bit-identical on CPU and GPU. The fix is to make acceptance **locally decidable**: every
  cell carries one candidate (jittered position + 32-bit rank), and a candidate is kept iff
  no candidate within `r` outranks it. That is one round of Luby's MIS, equivalently a
  Matérn type-II hardcore thinning whose parent is stratified rather than Poisson. Minimum
  separation is then a *theorem*, and the acceptance neighbourhood is exactly 3×3×3
  provided `r ≤ 1` (a candidate two cells out is >2−1 = 1 ≥ `r` away on one axis alone).
  Three details carry weight:

  * **`patBNPrecedes` is a strict *total* order** — rank, then `cz`, `cy`, `cx`. Totality is
    not pedantry: ranks collide with probability ~n/2³², which over an unbounded domain is
    certain *somewhere*, and under a merely partial order a colliding pair would each fail
    to outrank the other and both be kept, overlapping. The tie-break on cell coordinates
    closes the only hole in the separation proof.
  * **Stratification beats the dense-Poisson ceiling.** Matérn-II on a Poisson parent of
    intensity `L` retains `(1−e^{−LV})/V → 1/V = 3/(4π) = 0.23873` as `L→∞`. One stratified
    candidate per cell yields 0.2665, *12% above* that ceiling, because stratification
    removes the close candidate pairs that consume rank competition for nothing — so adding
    candidates per cell would make the result sparser, not denser. Along the way the
    conflict count had to be derived properly: it is **not** the ball volume, because the
    candidate's own cell holds no competitor, and subtracting the cube's self-overlap
    (per-axis triangle kernel) gives `E[N] = (3/2)πr⁴ − (8/5)r⁵ + (1/6)r⁶` for `r ≤ 1`
    (3.2791 at `r=1`, against 4.1888 from the naive argument). A first version of the
    self-test failed on the naive number; the fix was to redo the integral and confirm it
    by Monte Carlo (`scraps/bn_density.py`), not to widen the tolerance.
  * **Two early-outs keep the cost at Worley's** — 29 cells hashed per query against 27.
    A per-cell *geometric* lower bound (distance from the query to the nearest point of the
    cell) is tested before anything is hashed; and inside the acceptance test **rank is
    compared before position**, because the rank *is* the base cell hash (free) while the
    position costs three more mixes, so half the neighbours are dismissed for one mix.

  `r` is a genuine knob and a strict generalisation: `r = 0` vetoes nothing and reproduces
  the jittered lattice exactly (density 1/cell — `-checkbluenoise` §4 asserts equality),
  `r = 1` is maximally blue at 0.2661/cell with F1 in ~[0, 1.6]. Matching a `worley`
  texture's spot density therefore wants a worley cell `0.2661^(1/3) = 0.6432×` the size,
  which is what the demo's side-by-side back wall does. Device-side the query is
  `__noinline__`: inlined into `dPatternEval` it pushed the function to regcount 179 against
  the BDPT kernels' 168 and ptxas refused the build, so its frame is now paid only by
  programs that call it. `-checkbluenoise` has nine sections, each written so it cannot pass
  vacuously — pruned acceptance vs an unpruned ±3-block brute force (plus an assertion that
  both outcomes were seen thousands of times), F1/F2/id vs a ±6-block brute force at
  tolerance 0, minimum separation over every accepted pair in a 72³ block with the jittered
  lattice as control, the `r=0` identity, `E[N]` vs the closed form *and* density vs a rank
  model built from different data (a real test of rank/position independence, since both
  derive from one hash), number variance against a matched random-thinning control, the
  radial distribution function, ring depth and NaN/`r`-clamp guards, cells hashed per query,
  and the compile path. Worked example `scenes/pattern_bluenoise.ftsl`. One authoring fact
  that only rendering revealed: these are **solid** 3-D point sets, so a surface cuts the
  spheres at assorted depths and sees discs of radius `sqrt(R²−h²)` — a uniform `R` already
  gives a spread of spot sizes, with near-tangential ones very small.

  **Reaction–diffusion textures (`src/reaction.h`, O6, v0.167.0).** `texture "n" { reaction
  { … } }` solves Gray–Scott — `du/dt = Du∇²u − uv² + F(1−u)`, `dv/dt = Dv∇²v + uv² −
  (F+k)v` — once at load and stores V as a grey image. Turing's result is that the uniform
  solution of such a system can be unstable to spatial perturbation while stable in time, so
  the features are the **outcome of a process** rather than placed by fiat as `noise` /
  `worley` / `gabor` / `bnoise` all do; their spacing, branch points and defects are
  correlated the way a real coat pattern's are. It is a bake and not a `PatOp` because the
  value at a point is the endpoint of a trajectory of the whole field — no local closed form
  exists — and `texture` is exactly the shape of "an offline solve producing an image". That
  choice is what makes it free: UV wrap, Jakob–Hanika upsampling, triplanar, `reflect
  texture:`, `tex:<name>(u,v)` as a pattern term, GPU upload and the raster preview all work
  with **zero renderer changes**, and the demo exercises each. Design points, all of them
  measured rather than assumed:
  • *Periodic by construction.* The Laplacian wraps, so the solve is on a torus. Not
  retrofittable — blending the edges of a finished field destroys the long-range correlations
  that make it not-noise — so the topology, and the seed's exact `x·nb/N` block partition,
  are periodic up front.
  • *The 9-point stencil* (0.2 ortho / 0.05 diag / −1) is Pearson's, so published (F,k) maps
  transfer directly, and it is far more isotropic than the 5-point one, whose axis bias shows
  as square grain in the labyrinth regimes.
  • *The default diffusion is the classic 0.16/0.08 rescaled by s² = 2.5².* This was a real
  bug found by looking at renders, not by reasoning: a feature is a fixed number of grid
  *cells* wide (~2π√(D/F)), so at the textbook Du a spot is ~4 cells and comes out visibly
  **square**, pixel-locked to the lattice. Scaling both coefficients by s² is a pure spatial
  rescale — identical (F,k) physics, features s× wider — and s ≤ 2.79 by stability, so 2.5
  leaves 20% margin.
  • *Stability is a load error.* Explicit Euler needs `dt·max(Du,Dv)·1.6 ≤ 2` (the stencil's
  Fourier symbol bottoms out at −1.6 at a=b=π); past it the field NaNs within a few dozen
  steps, which is not a graceful degradation, so `rdStable` rejects at load with the
  arithmetic printed.
  • *The seed is the subtle part.* (u,v)=(1,0) is a fixed point **and linearly stable for
  every F,k>0** — Gray–Scott is subcritical — so infinitesimal noise gives a blank sheet; the
  perturbation must be finite-amplitude and domain-wide. Both its parameters were forced by
  measurement: a per-cell seed is smoothed away before it nucleates (five of six presets
  decayed), so blocks are ~one feature wide; and at 50% fill the ON blocks *percolate* into
  one domain-spanning region, after which patterning hinges on whether that one region
  survives — at sim=128 `spots` lost it and left a single nucleus creeping across a blank
  texture. 25% fill keeps the blocks isolated, and then every preset patterns at every `sim`
  and `seed`.
  • *Threading.* `ft::parallelFor` spawns a pool per call, so paying it 6000 times costs more
  than the arithmetic; threads start once and rendezvous at a sense-reversing `RDBarrier`,
  with double buffers indexed by step parity. Every cell reads only the previous buffer and
  there is no reduction, so the result is bit-identical for any band count (`-checkreaction`
  §5b pins it against a serial `rdLaplacian` reference at tolerance 0). Thread 0 alone polls
  `-stop`, *before* its compute phase, so the barrier publishes one shared answer.
  • *Presets are load-bearing, not a convenience*: the pattern-forming crescent is ~0.01 wide
  in k, so hand-picked numbers usually bake nothing. The six shipped were chosen by scanning
  the plane and **looking** at every tile — which is how the lattice-locking bug surfaced —
  and a seventh (`waves`) was dropped because it washed out at the rescaled diffusion.
  `-checkreaction` §7 re-derives contrast (as the *standard deviation* of the normalised
  field; its range is 1.0 by construction, so asserting on range would be vacuous) and the
  dominant wavelength from a radially averaged separable DFT, at the shipped defaults, so a
  preset name cannot rot silently. Its other sections pin the stencil against its Fourier
  symbol at every representable wavenumber, the optimised loop against the reference
  Laplacian, the uniform fixed point, torus translation invariance, seam vs interior
  gradients, `rdStable` against actual divergence, and determinism/resampler identities.
  • *Not everything converges*: spot regimes settle to ~4e-5 relative change per step by
  24000 steps, but `maze` never does (~3e-3 even then) — the labyrinth keeps reconnecting —
  so there `steps` is an aesthetic choice, and the docs say so rather than implying otherwise.

  **Band-limited fBm (`PatOp::FNoise`, O8, v0.168.0).** `fnoise(x, y, z, w, octaves)` in
  `src/pattern.h` — the same lattice `noise`, summed at lacunarity 2 / gain 0.5, with each
  octave scaled by how much of it a sample of width `w` can resolve. `w` is in the units of
  the coordinates passed in (so a footprint of `w` metres sampled at `90*x` must be handed
  in as `90*w` — the easiest mistake to make with this op), and `w <= 0` is plain
  unfiltered fBm, which keeps every existing use unchanged. Every other noise here is
  evaluated at a point, which is a lie the moment one shading sample stands for an area:
  a feature smaller than the footprint doesn't fade out, it **folds down** into a coarse
  pattern of the sampling lattice, and no amount of extra samples removes it because the
  signal was never band-limited. It matters for the **deterministic** samplers — mode W,
  the raster preview, low-spp backward. The forward photon modes spread millions of hit
  points across each pixel's footprint, so they area-average stochastically for free and
  `fnoise` would only cost them detail.

  The whole design is in the weight, and it was **measured, not chosen**:

  * *The optimal weight is the linear-MMSE coefficient*, and it has a closed form worth
    writing down: `a(s) = Cov(footprint mean, point) / Var(point) = mean_{u∈footprint} R(u)
    / R(0)`, the footprint-average of the noise's own autocorrelation, at `s = freq·w`.
    That makes it a measurable quantity rather than a taste, and it is nothing like the
    obvious schedule: the true weight is **0.949 at Nyquist** (`s = 0.5`) and still
    **0.815 at `s = 1`**, exactly where a naive cutoff drops the octave whole. The first
    implementation *was* that naive smoothstep and it failed §3/§5 of its own self-test by
    filtering **worse than not filtering** — the fix was to derive and measure `a(s)`
    (`scraps/fnoise_fit*.py`) and fit a monotone rational to it (max abs error 2.5e-4).
  * *The footprint's **dimension** changes the tail by a whole power of `s`* — a 3-D ball
    average decays like `s⁻³`, a 2-D surface patch like `s⁻²`, a 1-D segment like `s⁻¹` —
    so one curve cannot serve all three, and this is not a detail that can be papered over.
    An intermediate version fitted the **3-D cube** curve; §3 then passed and §5 still
    failed, and the cause was exactly this mismatch (§5 was measuring against a 1-D
    average). `w` is therefore defined as **the diameter of a disc lying in the surface**,
    orientation-averaged, because that is what a shading sample actually stands for; the
    self-test was rewritten to average over a randomly-oriented disc so it measures the
    contract rather than a convenient proxy.
  * *Over-filtering is **not** the safe direction* — the instinct that blur is conservative
    is wrong here, and measured to be wrong: applying the 3-D curve to a 2-D footprint
    gives only a 1.6× improvement at `s = 2` against 2.3× for the opposite mismatch,
    because it deletes low-frequency content the footprint genuinely contains and lands
    further from the truth than not filtering at all. Hence the docs tell an author in
    doubt to lean toward the *minor* axis of the elliptical footprint.
  * *Normalisation is by the **unfiltered** amplitude sum*, so a faded octave is genuinely
    gone and the field converges to 0.5 rather than being renormalised back up into
    something that still crawls.

  `-checkfnoise` is written so it cannot pass by tuning. §2 pins the closed form against
  the 22-entry measured disc table (reached two ways — `patOctaveFade(1, s)` and
  `patOctaveFade(8, s/8)` — so a mis-scaled `freq·w` cannot hide), sweeps 200001 points for
  monotonicity, and carries an explicit anti-regression that the fade has not "collapsed to
  a Nyquist cutoff". §5 is the interesting one: it originally asserted a fixed ratio and
  failed at 1.68×, and rather than loosen the number the test now asserts the *structure*
  that the sweep actually revealed — point-sampling's low-frequency error **saturates**
  under minification (0.0148 → 0.0247 → 0.0269 → 0.0262) while `fnoise`'s keeps falling
  (0.0088 → 0.0091 → 0.0052 → 0.0028), so the two separate without bound (1.68× → 9.28×).
  A saturating error curve is the signature of aliasing, and that is what is now pinned.

  One authoring fact that only rendering revealed, and that the worked example
  `scenes/pattern_fnoise.ftsl` exists to state: **a classic wide fBm barely aliases**. In a
  gain-0.5 stack the fine octaves carry almost no amplitude (the seventh is 1/64 of the
  first) and they are precisely the ones a distant sample cannot resolve, so a wide fBm
  both aliases faintly *and* filters faintly — the first demo (7 octaves at base 9) had two
  indistinguishable halves, which was the primitive being right rather than the demo being
  right. What aliases visibly is a texture whose energy sits **at** the resolution limit,
  which is also the common case in practice (grain, weave, gravel, stucco), so the demo is
  3 octaves at base 90/m on a floor running to the horizon, split down the middle. Measured
  off that render, mean |pixel − its 3×3 mean| by distance: plain stays at 9–11 from 3.8 m
  to the horizon while filtered falls 9.3 → 0.0. The flat line is the point.

  Wiring is the O2 checklist (enum at the end, one case in each of the three VMs,
  `patOpStackEffect` arity 5, no payload).

  **Stage 2 — the `fw` variable** (v0.169.0). The author should not have to derive `w`;
  the renderer knows it. `fw` is the world-space **diameter of the surface patch one
  shading sample stands for**, and it is split in two so that no two backends can disagree
  about it: `Camera::footprintPerDist(spp, rx, ry)` returns the distance-independent
  coefficient, and `patShadingFootprint(perDist, dist, cosSurf)` in `pattern.h` (shared
  `__host__ __device__`) combines it with one hit. Three decisions are load-bearing:

  - **The pixel's angular size comes from `pixelSolidAngle()`**, converted to the diameter
    of the disc subtending it (`2·√(Ω/π)`), not from `fov_y/res_y`. That makes fisheye and
    panoramic lenses fall out for free, and it is ~18% wider than the naive number for a
    square pixel (the tan expansion plus the equal-area conversion). Evaluated **on axis**:
    the off-axis variation is a cos³ effect, an order of magnitude below the obliquity
    term, and taking it on-axis keeps `fw` from depending on which pixel a surface lands in.
  - **Obliquity uses the geometric mean of the ellipse axes** (`d/√|cos|` — the disc of
    equal area), floored at `|cos| = 0.02`. Not the major axis: stage 1 measured that
    over-filtering is the worse mismatch, so ties break toward the minor. Without the floor
    every silhouette would filter to a flat grey band.
  - **0 means unknown means UNFILTERED**, never a small blur. `fw` is filled only where the
    renderer can answer honestly: primary hits in mode W (`BackwardRenderer::fwPerDist` /
    `DScene::bkFwPerDist`, stamped at `b == 0 && gi.depth == 0` in both the scalar and hero
    path loops — `b == 0`, *not* `b == bounce0`, since a heroSplit re-entry resumes deeper)
    and every pixel of both raster previews (`dRasterFw` / `raster.h`'s shade pass, which
    pass `W`/`H` as the resolution override because the preview window is not the camera's
    film). It stays 0 in the forward modes and stochastic mode R — a sampler that jitters
    over its own footprint is already area-averaging, and filtering on top would only cost
    detail — at secondary bounces (ray cones through specular bounces are the obvious
    extension, and `curv` is already sitting at every hit waiting to drive them), and in
    implicit-field/medium formulas, where a filtered SDF would round off the very detail
    the sphere-trace is looking for and make its distance bound non-conservative at grazing
    angles. Like `curv`/`cavity` it is rejected in an `emit` pattern, for the strictly
    stronger reason that it is view-dependent.

  `spp` divides it by `√spp`, and mode W latches ONE `g_fwSpp` for the whole run (the run's
  requested `-spp`, not a chunk's) so every chunk of a progressive or resumed render filters
  identically and their average stays a render of one image. `-checkfnoise` §10 pins the
  geometric-mean / cos-floor / zero-is-unfiltered conventions, the `1/√spp` and
  resolution-override scalings, and the variable's parse / free-var / VM wiring. CPU and GPU
  mode W were measured to differ no more with `fw` than without it (max 8/255 of one channel
  on 0.9% of the demo's pixels, against 9/255 on 1.1% for the same scene with `fw` removed —
  i.e. entirely the pre-existing float/double divergence). The demo scene now reads
  `fnoise(90*x, 90*y + 0.5, 90*z, 90*fw, 3)`, and its hand derivation is kept in the
  comment as the explanation of what `fw` contains.

  **Stochastic tiling (`src/stochtile.h`, O7, v0.170.0).** `texture "n" { tiling stochastic
  patch <p> seed <s> }` is Heitz & Neyret's (HPG 2018) histogram-preserving blending: at each
  shading point, three randomly offset crops on a triangle lattice, blended with the
  barycentric weights. The problem is **not** seams — the demo's source is deliberately
  seamlessly periodic and still fails, because at six repeats per metre the eye locks onto
  the same rosette marching in a grid, which no wrap mode can address. The naive fix is worse
  than the disease: averaging three crops of a bimodal image emits the mean of its two modes,
  a colour occurring nowhere in the source, and the wall goes to soup. So each channel is
  **rank-transformed** at load onto `N(1/2, 1/6)` (`gaussRgb` / `gaussGray` planes), the taps
  are blended there, the variance the average destroyed is restored by dividing the centred
  blend by `sqrt(Σwᵢ²)`, and the result is inverted through a stored 1-D LUT (`lutRgb` /
  `lutGray`). `E[Σwᵢ²] = 1/2` for Dirichlet(1,1,1) weights, so an unrestored blend sits at
  `1/√2 = 0.707×` the source's sd; `-checkstochtile` measures 0.4999 and 0.708× against those
  closed forms. Architecture points, each of which was forced by a measurement:
  • *The blend is in linear RGB, not in Jakob–Hanika coefficient space.* Blending coefficients
  per channel produced visible blue-cyan fringing, because coefficient space is not a colour
  space: `R = sigmoid(p(t))` is nonlinear in `c`, so a weighted mean of coefficient triples is
  not the weighted mean of the colours. The three taps are therefore blended as RGB and the
  blended colour converted through **one shared, texture-independent** 64³ coefficient LUT
  (`upsample::coeffLut`, `stochJhCoeff`), built lazily and threaded (~0.6 s, 3.1 MB) on first
  stochastic-texture load. This is also what makes the three backends agree: the spectral CPU
  path (`Texture::reflectanceAt`), CUDA (`dTexReflAt`) and the mode-W raster preview run the
  *identical* operator on the *identical* planes, differing only in where the planes live.
  • *The LUT's real problem was conditioning, not resolution.* A colour whose reflectance is
  pinned at 0 or 1 across the band needs `|p| → ∞`; the unbounded fitter returns `|c| = 1.6e6`
  at the white corner, and trilinear interpolation between that and a moderate neighbour is
  meaningless — worst reflectance error 0.81. The fix is **projected Gauss-Newton**:
  `fitSigmoid` takes an optional `pMax`, and inside its backtracking line search the trial
  point is pulled back into the bounded set (`upsample::clampSaturated`) *before* being
  scored, so what the search accepts is the best **bounded** colour match rather than an
  unbounded one mangled afterwards. Bounding a finished fit instead costs 0.037 of
  `|XYZ − target|` near white; projecting inside the loop lets the remaining freedom
  compensate for the clip. `clampSaturated` soft-compresses `p` through `pMax·tanh(p/pMax)`
  (identity for small `p`, and it preserves every root exactly, so the wavelengths where R
  crosses ½ do not move) and weighted-least-squares a quadratic back through the compressed
  curve over 380–730 nm with weight `max(dSigmoid(pc), 1e-3)`. Three simpler alternatives were
  implemented and rejected on measurement, and the source comment records each: uniform
  scaling of `c` (shrinks the unsaturated middle; dark saturated red `|dR| = 0.97`), a 3-node
  clamp at 360/595/830 nm (`|dR| = 1.0` near white — the fit is *unconstrained* where the CMFs
  vanish, so those nodes hold noise), and one at 400/550/700 nm (`|XYZ − target| = 0.37` near
  white, because white's fit puts its roots within a nanometre of exactly 400 and 700). A
  CMF-weighted LS also lost (worst 0.037 → 0.088).
  • *The right metric is round-trip `|XYZ − target|`, not reflectance agreement with
  `upsample::fit`.* Comparing curves is a trap for exactly the reason above: two coefficient
  triples producing the identical colour can differ by `|dR| = 1` past 700 nm where the CMFs
  are zero. `-checkstochtile` §8 therefore scores XYZ round-trip over 3000 colours (half cubed
  toward black) and carries the un-tabulated fit as a **control**, holding the LUT to "no
  worse than the fitter it stands in for" (`wL < wF + 0.025`) rather than to an unachievable
  ideal. Grid size was picked from a sweep, not taste: 48³ worst 0.036, 64³ worst 0.019, 80³
  worst 0.016 — it stops improving because the residual is no longer the table but the
  sigmoid-of-a-quadratic model's own 0.019 error at pure white, so `STOCH_JH_N = 64`,
  `STOCH_JH_PMAX = 60`.
  • *Validated in the render, not only in the unit test.* `scenes/stochtile.ftsl` rendered on
  all three backends and compared statistically: the lattice autocorrelation at exactly one
  repeat (64 px) is +0.998/+1.000/+1.000 on the plain half and **+0.017 on the stochastic
  half — the same value to three decimals on all three renderers**, which is the real
  evidence they run one operator rather than three similar ones. And the operator's own claim
  holds end to end: the stochastic half's mean and sd (0.2690 / 0.1485) match the plain half's
  (0.2679 / 0.1492), i.e. the histogram survived the blend rather than regressing to the mean.
  CPU vs CUDA spectral differ by mean |Δ| 0.0058 with means agreeing to 0.0005, so the
  difference is Monte Carlo noise between two independent runs, not a systematic split.
  • *A latent `fitSigmoid` divergence was found on the way and fixed.* Undamped Gauss-Newton
  diverged for dark saturated colours (red at Y=0.001 gave `|XYZ − target| = 1.41`, blue at
  Y=0.01 gave 1.40); the backtracking line search brought those to 0.0008 / 0.0001, and §8
  now asserts them so they cannot regress. This affected *every* Jakob–Hanika upsample in the
  renderer, not just the tiling LUT.

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
- **`majorant.h` — the per-cell control/residual majorant grid, and RESIDUAL RATIO TRACKING
  (0.200.0).** Every heterogeneous, *bounded* medium with a positive `density_max` gets a
  `MajorantGrid` baked at load (`ftsl.h`, immediately before the `Medium` is pushed): a regular
  lattice (long axis 128, capped at 4 M cells) storing two floats per cell — `ctrl`, a
  piecewise-constant **control** density, and `res`, a majorant of `|density − ctrl|` over the
  cell. `Medium::majorant` is a `shared_ptr`, uploaded to the device as `DMajorant` (two float
  arrays + the lattice transform) and walked by a 3-D DDA (`Renderer::majorantWalk` on the host,
  `DMajWalk` on the device) that hands each traversed cell's `[t0,t1]` to the estimator.
  - **What it fixes, and the counterintuitive reason a plain local majorant does NOT.** The
    obvious use of a majorant grid is to tighten `σ_max` for ratio tracking. Measured (a Python
    sim of `gallery_rain`'s cloud), that buys **exactly nothing**: ratio tracking's
    `Tr *= 1 − σ(x)/σ_max` converges *deterministically* to `exp(−τ)` as `σ_max → ∞`, and
    degenerates to a binary hit/miss as `σ_max → σ`. A tight majorant is the **bad** end of that
    range. Ratio tracking wants a LOOSE bound; its relative sd grows like `e^(τ/2)` regardless.
  - **Residual ratio tracking (Novák et al. 2014) is the actual win**, and it needs exactly the
    two numbers the grid stores: integrate the control analytically and track only the residual,
    `Tr_cell = exp(−σ_c·L) · Π(1 − (σ(x_i) − σ_c)/σ_r)` with candidates drawn at rate `σ_r`. The
    factors now sit near **1** instead of near 0, and the residual is *signed*, so a factor above
    1 is correct and is what keeps the estimator unbiased. Same sim: relative sd 13.11 → 0.385 at
    τ=8 (**1160×** variance reduction) and 41 → 0.55 at τ=16 (**5700×**) — and it is also
    *faster*, because `σ_r` is far below the global majorant so there are far fewer candidates.
  - **Collision sampling takes the opposite half.** Delta (Woodcock) tracking is unbiased for any
    upper bound, so `sampleMediumCollision` uses the per-cell **sup** `ctrl + res` — fewer null
    collisions — and skips a vacuum cell with no RNG draw at all.
  - **Conservativeness is bought three ways**, because a sampled grid that under-estimates a peak
    silently biases the render: each cell is probed on an (S+1)³ corner lattice (S=2, so
    neighbouring cells share probe planes); each cell's `[min,max]` is then **dilated over its
    3×3×3 neighbourhood**, so a feature that fell between probe points is still covered and
    "vacuum" means a three-cell-wide margin of zero; and the residual half-range carries a ×1.15
    safety factor. The probe uses `densityAt` (**with** the membership carve) — unlike the global
    `densityMax` estimate, which must use `densityFieldAt` (see `meshvoxel.h` below) because a
    coarse probe of a thin shape would otherwise majorise to ~0. That difference is safe here
    precisely because the grid is local: a cell outside the carve really is vacuum, and the
    3×3×3 dilation is what makes reading it as vacuum safe.
  - **It is a pure speed/variance change, with no new knob and no scene-file syntax.** The only
    observable is the load-time line, e.g. `gallery_rain`'s cloud:
    `[medium] majorant grid: 128x80x125 cells (0.0213 m), 72.0% vacuum, mean control 0.204, mean
    residual 0.0739 (peak 0.575) vs global majorant 1.150` — i.e. the tracking rate through that
    medium fell **15×**. Both estimators keep the global-majorant loop as a fallback for media
    with no grid (unbounded, homogeneous, or a build that was `-stop`ped), so nothing regresses.
    Energy conservation on `gallery_rain` is unchanged (`sum/emitted = 0.999964`).
- **`rainbow.h` — droplet SIZE DISTRIBUTION (0.199.0).** The Airy bow's *angle* is geometric and
  size-independent, but the fold scale `K = (2/h)^(1/3)·(2πa/λ)^(2/3)` is not, so `z` scales as
  `a^(2/3)` and averaging `Ai(z)²` over a spread of sizes smears the supernumerary train into its
  envelope while leaving `theta_rb` exactly in place. `Params::dispersion` (FTSL `dispersion` /
  `spread` / `rain_mm_h`) is the relative sd of the **scattering-weighted** size distribution, a
  gamma DSD with `K2 = 1/dispersion²`; `0.577` (`K2 = 3`, `k = 1`) is exactly Marshall-Palmer and
  is the **default**, because that is what rain is. Note MP's relative width is *rain-rate
  independent* — `Λ = 4.1·R^(-0.21)` moves the mean drop size but not the shape (verified at
  R = 1, 5, 25 mm/h in `scraps/_supernum_mp.py`).
  Two subtleties the implementation turns on:
  - **two different moments of the DSD, not one.** Which sizes a photon *samples* is weighted by
    `sigma_sca ~ a²` (shape `K2`); how bright each size's *bow* is goes as `a^(7/3)` (van de
    Hulst), so the mix that shapes the bow is shape `Kb = K2 + 1/3`. `droplet_um` is defined on
    the `a²`-weighted mean, which is why it means the same thing at every dispersion.
  - **cells are averaged, not point-sampled.** At `z = -80` the Airy phase sweeps ~1400 rad across
    a Marshall-Palmer distribution, so point-sampling 64 sizes would alias into fixed-pattern
    ringing through the bow interior. `airyAi2Mean(a,b)` returns the *exact* mean of `Ai²` over
    an interval from a tabulated tail integral `S(u) = ∫_u^∞ Ai²`, so the quadrature converges on
    the cell count of the smooth density (64) rather than of the oscillation (thousands). Cost is
    2 lookups per cell **at table-build time only** — the render sees only the finished λ×µ table,
    so the GPU needs no change at all.
  This replaced `Params::supernumerary`, a flat clamp holding the Airy *peak* across the whole bow
  interior; measured 3.8×–14.9× too bright and collapsing arc/interior contrast from 6.40× to
  1.01× (full table in `known-issues.md`). The FTSL key survives as a deprecated alias
  (`on → dispersion 0`, `off → 0.577`) with a one-time note. Same change fixed `airyAi(x)`
  returning **0** below `x = -30` (the RK4 table's floor), which hard-clipped the bow interior
  where the `Ai²` envelope is still at 35% of peak; it now falls through to the exact asymptotic.
- **`meshvoxel.h` — mesh containment for fog bounds.** `medium { bounds { object "<mesh>" } }`
  used to degrade to the mesh's AABB; `meshvox::voxelizeSolid` now **solid-voxelizes** the
  named mesh into an occupancy `VdbGrid` (`MediumBound::Mesh`, `Medium::boundGrid`,
  `Medium::insideMesh`). Deliberately routed through `VdbGrid` rather than a bespoke
  structure: that is already the dense-volume vehicle, already sampled by identical CPU/GPU
  code and already uploaded as a sparse brick lattice, so the GPU path, majorants and
  delta/ratio tracking all worked with **no new plumbing**. Note the distinction from
  `Medium::vdb`, which *replaces* the density field — `boundGrid` only decides membership,
  so an authored `density` formula still multiplies on top and shapes fog *within* the mesh
  silhouette (the same semantics an isosurface bound has). Fill is by **signed-crossing
  (generalized winding)** x-scanlines, not parity, so multi-body / self-intersecting imports
  come out as a union rather than hollowing in the overlap; triangles are scattered into the
  rows they cover, giving O(tris + covered rows).
  - Two load-order hazards this exposed, both fixed and worth knowing: **`Tri::gn` is empty
    during the load** (`Tri::finalize()` runs in `Scene::build()`, after the loader's
    deferred medium sweep), so anything running inside the loader must derive geometric
    facing from the vertices — the voxelizer uses `det = cross(v1-v0, v2-v0).x`. And
    majorant estimation must use `Medium::densityFieldAt` (the field with no membership
    carve) rather than `densityAt`, or a coarse probe of a thin shape majorises to ~0 and
    the medium silently vanishes.
  - **`feather <metres>` (0.130.0)** is the answer to the one thing that bake gets visually
    wrong. Occupancy is **binary** and the `VdbGrid` sampler's trilinear filter ramps 0→1 over
    exactly **one voxel**, so a mesh bound is effectively a hard silhouette — correct for a
    body, badly wrong for a cloud, whose real edge is a zone metres deep. `meshvox::featherGrid`
    replaces the 0/1 fill with `smoothstep(dist_to_outside / feather)`. The distance is an
    **exact** Euclidean distance transform (Felzenszwalb & Huttenlocher 2012: a 1-D
    lower-envelope-of-parabolas pass run once per axis, O(n) each, `meshvox::edt1d`), *not* a
    chamfer/Manhattan approximation — a chamfer's error is anisotropic and would print the
    lattice's own axes onto the falloff, which is exactly the artefact being removed. Two
    numerical traps, both already paid for: the "unreached" seed must be a **large finite**
    value (`nx²+ny²+nz²+1`), because F&H intersects parabolas by *subtracting* two seeds and
    `1e300 - 1e300` is pure cancellation noise; and the ramp is a smoothstep rather than a
    linear one, because a linear ramp creases visibly where it reaches 1 and that crease reads
    as a second, softer silhouette. Because it only ever *lowers* density it is majorant-safe
    for free (`maxVal` stays 1). `ftsl.h` takes the value in **world metres**, divides by the
    voxel edge, warns below one voxel, and **rejects the key on sphere/isosurface bounds**,
    which are carved analytically and have no lattice to soften.
  - **`mesh { shape_only yes }`** is the companion: the triangles are loaded for the bake and
    then removed from `Scene::tris` by `Builder::stripShapeOnlyMeshes` — run between the
    deferred medium sweep and `Scene::build()`, so they never reach the BVH. Renumbering is
    safe because `Scene::tris` is indexed only via `MeshGroup::triStart/triCount` (fixed up
    there); mesh area lights *copy* their triangles into `Emitter::meshTris`, and the key is
    refused on an emissive material anyway.
  - **`scenes/gallery_rain.ftsl`** is the shipped worked example, and it exercises the whole
    path in anger: `cloud1.glb` (1.85 M tris, two disjoint lobes — the case parity fill gets
    wrong) bakes to a 195×122×191 lattice at 30.1 % solid, uploads as 2406/6000 sparse bricks
    (5.3 MB → 2.4 MB VRAM), and all 1.85 M triangles are then stripped by `shape_only`. It is
    also the shipped example of `feather` (0.13 m ≈ 9 voxels), without which the cloud reads
    as a sticker. Under it hangs a rain curtain using the `rainbow` phase function. Several
    things about that scene are load-bearing and non-obvious, so they are written up in its
    header rather than only here:
    - **The rain shaft starts *inside* the cloud, not below it.** Both the top of its bounds
      and the horizontal extent of its density ellipse must sit within the cloud's own
      silhouette, and the density must fade to zero *before* the box's top face — otherwise
      the "shaft" shows a straight lid and two vertical shoulders in clear air.
    - **`light area` is real opaque geometry in the BVH**, so any panel can eclipse something.
      The sky panel is therefore *raised* to y=12 (a sun ray leaving the scene's lowest exit
      point clears its far edge) rather than shrunk, and the two later fill panels are placed
      by checking where a shadow ray toward the sun crosses their plane.
    - **The two fill panels exist to be reflected, not seen.** With the walls gone, a polished
      metal's rim points at an empty black sky and renders black; roughness alone cannot fix
      it, because a wide lobe around a grazing reflection still samples mostly black sky. Both
      panels are placed just outside the still camera's 41° half-field so the background stays
      pure black. The right-hand one is deliberately pushed *back* level with the camera:
      mirroring the left one's z would have put it 1.6 m from the rain and poured fill into
      the one medium that must stay dark, since a rainbow is single-scatter and only survives
      against a dark backdrop. Verified by measuring the bow region's mean level before and
      after — it moved 0.3 %.
    - **The cloud's medium is the DELTA-EDDINGTON REDUCTION of a real cloud, not eyeballed
      (retuned 0.199.0).** A real cloud is `sigma_t ~ 10`, `albedo ~ 0.999`, `g ~ 0.85` (Mie,
      ~10 µm droplets). Splitting the forward Mie spike `f = g²` off as unscattered light gives
      `sigma_t* = (1-wf)sigma_t = 2.78`, `w* = w(1-f)/(1-wf) = 0.9964`, `g* = g/(1+g) = 0.46` —
      which is the scene's *original* `3.0 / 0.5` to within rounding. So only the albedo was
      ever wrong, and badly: `0.92` made a water cloud absorb 8 % per scattering event
      (water's imaginary index at 550 nm is ~1e-9, so a droplet's albedo is 0.9999+), which is
      `0.92^10 = 0.43` over a crossing. Clouds are white *because* they scatter conservatively;
      a raincloud's dark base is optical **depth**, not absorption. Measured at 480×270 mode
      `D`, 800 spp: the fix is **+29 %** on the cloud (+2.6 % on the frame, so it is a cloud
      change, not an exposure shift). The **full** form was rendered too and rejected on
      evidence — it flattens the crown/base contrast from 1.13 to 0.99 (destroying the dark
      underside the bow is read against), is **2.5× noisier** at equal spp, and is the only one
      of the three that emits monochromatic fireflies (8.4 % of cloud pixels over 3:1 channel
      ratio, peaking at 5e29 — logged in `known-issues.md` with the mode-`D` NaNs, likely one
      bug). Full table in the scene's own `medium` comment; measured by `tools/check_cloud.py`.
      This mattered little before 0.199.0 and matters now: `-beams` was single-scatter through
      0.198.0, so mode `M` had no scattering orders for a wrong albedo to compound over.
    - The sun spent the scene's early versions as a **distant sphere** because mode `D` refused
      a scene containing a `light sun` outright; 0.124.0/0.127.0 lifted that and it is now a
      real `light sun`. The old global haze is deleted: its `bounds` box was invisible only
      because the walls hid its faces.
    - **The caustic screens are dark grey (0.15), not white — and that is the whole trick.** A
      caustic is only visible as a *ratio* to its screen, and a display shows no ratio above
      its clip point. At the original 0.88 albedo the sunlit caps sat at 189–209/255 with
      10–25 % of their pixels already at pure white, so a 5× caustic, a 50× one and the plain
      sunlight beside them all printed the same `#FFFFFF`: the caps were metered *into* the
      clip, not too dim. 0.30 was the first fix and was **still not enough** — with the axicon
      in the scene, 596 of that cap's 22639 pixels were pure white, more than half the
      caustic's own area (found only once `-hdr` existed to measure it; see below). The screen
      has to be set so the caustic **peak** lands under clip, not so the ambient does. Because
      a diffuse cap's albedo scales its radiance exactly, the choice was made arithmetically on
      the linear buffer at zero render cost, and **the metric it moves is the displayed one,
      not the metered one**: on the float sidecar, chromaticity and peak:median are both
      scale-invariant, so albedo changes `spread` / `sat` / `coverage` by literally nothing
      (measured: identical to 3 decimals at 0.15 and 0.30). What it changes is how much of the
      caustic the tone map deletes — on the same 1036 spp buffer the axicon cap clips **3.06 %
      of its pixels at 0.30 and 0.70 % at 0.15**, i.e. half the caustic's own area versus a
      tenth of it. Measured through the clamp (which is what a viewer sees) that is spread
      0.111 → 0.147, +32 %. 0.15 puts the caps 2¾ stops below clip with room for the axicon's
      ~7× cusps on top; below it the returns fall off sharply (0.09 buys a further +0.017 for
      a dingy sRGB-80 tabletop).
    - **The ten tabletops are MARBLE, and each one is renormalised to that same 0.15 before it
      is allowed near a cap.** The caps are not decoration, they are the caustic screens, so a
      texture on one is not a free cosmetic change: a caustic on a diffuse surface is
      **multiplicative** (`pixel = albedo(x) · irradiance(x)`), which makes the stone a
      multiplier on the very quantity the scene exists to measure. The raw marble photographs
      mean 0.34–0.85 *linear* (sRGB 0.62–0.93), so dropping one in as-is would raise a cap's
      albedo by up to **5.7×** — two and a half stops back *into* the clip the 0.15 was chosen
      to escape, and every caustic in the scene would print flat `#FFFFFF` again. So
      `tools/make_marble_caps.py` prepares each sheet and its one non-negotiable operation is
      to rescale the image so its **mean linear reflectance equals 0.15**. The pattern
      survives; the brightness does not. Two further knobs follow from the same
      multiplicativity and are applied *in linear light* so they compose with the rescale as
      pure multiplies: **contrast `k`** (albedo swing about the mean — a vein at half the mean
      halves the caustic that crosses it, i.e. a dark vein *erases* it) and **saturation `s`**
      (a coloured stone *tints* the caustic, attacking exactly the `sat`/`spread` the axicon
      exists to produce). Both default to 1 and are turned down only where they must be.
      Assignment is by **measured caustic strength, not taste**: the three caps that catch a
      real caustic (axicon 0.30 % / sat 0.275 — and the only piece that makes colour; crystal
      orb 0.72 % / 0.246; solid gyroid k=10 0.37 % / 0.208) get the flattest, most neutral
      sheets, further calmed to k 0.45–0.55 / s 0.30–0.40, which leaves them swinging only
      **1.29–1.37×** p2→p98. The other seven measure at or near zero — gold/brass/chrome are
      opaque, the heart and jack are opaque iridescent, and the Klein bottle's 2.4 mm wall is
      optically a window (0.12 %, nothing at the 1.5× bar) — so they are *free*, and get the
      dramatic gold-veined slabs at full contrast, deliberately putting gold marble under the
      gold gyroid and the brass cluster. Sources are also matched to each cap's **on-screen
      footprint** (gyroid 634×293 px down to oil 72×16 px), so the two ~200 px drops go to caps
      under 160 px and the 4650² and 1600×1067 sheets to the two that matter; everything is
      box-filtered to ≤1024 px because a 1.6 m cap never spans more than ~650 px. Two source
      files needed handling rather than trust: `marble texture 2.jpg` (actually a palette PNG)
      bottomed out at 0.006 linear (sRGB 19) at full contrast, reading as a *hole* in the
      tabletop, so it takes k=0.75 to lift the floor without touching its veins; and
      `marble texture 3.5.avif` is a **watermarked VectorStock preview** whose black footer bar
      is why it alone measured p5 = 0.000 — the bar is cropped off (bottom 9.5 %), but it is
      still a stock comp and is flagged here and in the scene rather than shipped silently.
      Each cap gets `uv planar axis=y` so the slab is quarried once across the whole tabletop
      rather than tiling.

      **Verified, not asserted.** The same frame was rendered twice at matched settings and
      matched convergence — marble 259 spp / 6.21 % noise against the untextured scene at
      230 spp / 6.59 % — and metered with `_capchroma.py`. On the three caps that carry a
      caustic, nothing moved:

      | cap | coverage | sat | spread | peak | clip | noise | fan |
      |---|---|---|---|---|---|---|---|
      | **axicon** untextured | 3.64 % | 0.436 | 0.271 | 7.90× | 0.15 % | 0.050 | 0.75 |
      | **axicon** marble | 3.53 % | 0.452 | 0.267 | 8.03× | 0.15 % | 0.050 | 0.74 |
      | solid gyroid (diamond cap) untextured | 4.93 % | 0.886 | 0.087 | 3.17× | 0.00 % | 0.060 | 0.50 |
      | solid gyroid (diamond cap) marble | 5.04 % | 0.904 | 0.081 | 3.28× | 0.00 % | 0.062 | 0.45 |
      | crystal orb (glass cap) untextured | 0.37 % | 0.260 | 0.008 | 2.48× | 0.00 % | 0.097 | — |
      | crystal orb (glass cap) marble | 0.37 % | 0.223 | 0.003 | 2.29× | 0.00 % | 0.099 | — |

      Every number is within run-to-run scatter; the axicon's `clip` is *identical* at 0.15 %,
      which is the specific thing the 0.15 normalisation existed to protect. (An early read at
      70 spp showed clip 0.13 % → 0.38 % and looked like a real regression — it was pure
      sample-count artifact and vanished on convergence. Do not meter this at low spp.) The
      orb's `sat`/`spread` drop is on absolute values of 0.008 → 0.003, i.e. inside the noise
      of a caustic already documented as "organised colour, and almost none of it".
    - **CAVEAT — `_capchroma.py` assumes a UNIFORM cap albedo, and a high-contrast texture
      breaks it.** The metric is "excess over 2× the cap's own median", which silently
      attributes *all* variation to light. On the full-contrast decorative slabs the stone's
      own veins clear that bar, so the meter reports a caustic that does not exist: the gyroid
      cap reads **coverage 4.55 %, sat 0.434, spread 0.201, fan 0.84** where the untextured
      control reads a flat **0.00 %** — and the gold gyroid is opaque, so there is no caustic
      on that cap at all. `fan 0.84` is being scored on *rock*, because marble veining also
      varies smoothly with position and that is precisely what `fan` detects. The `noise`
      control band gives it away (0.028 → 0.297 on that cap; brass 0.028 → 0.164), which is
      the tell to look for. This costs nothing today — the seven textured-for-drama caps are
      exactly the ones with no caustic to measure — but **only the three calm caps remain
      metrologically valid**, and their noise floors are untouched (axicon 0.050 → 0.050,
      diamond 0.060 → 0.062, orb 0.097 → 0.099), which independently confirms the `k`/`s`
      calming was sized correctly. To meter a decorative cap properly the tool would have to
      divide the texture back out before thresholding.
    - **The glass orb is levitated 0.30 m on three pins, and its cap is cantilevered.** The
      height was *measured*, not computed. The textbook ball-lens `f = nR/(2(n−1))` — 0.734 m
      from centre for BK7 at R=0.5 — is **paraxial**, and a full-aperture sphere has gross
      spherical aberration: the marginal rays carry most of the flux (area ∝ r²) and cross far
      nearer the glass, so the paraxial focus is the wrong target and its disc-area estimate of
      concentration is badly optimistic. `scraps/_focalsweep.py` renders a sweep of heights and
      meters peak linear irradiance on the cap; the answer is a sharp optimum at **0.80 m of
      drop** (0.642, vs 0.371 at 0.65 and 0.424 at 0.95 — 73 % brighter than either neighbour
      0.15 m away), i.e. *short* of the paraxial focus, not past it. Resting tangent (0.50 m,
      the original scene) was 0.58× of the achievable peak. The cost of 0.80 m is that the
      sight line to the disc centre passes 0.514 m from the orb's centre, only just clearing
      the 0.5 m limb, so the disc reads as breaking out from behind the glass. The cap is 1.7
      deep and centred 0.45 m −z of its column because that is where the disc lands, (6.87,
      2.72), from the sun's +0.2079 x / −0.9781 z per metre of drop.
    - **The dispersive caustic is chromatic speckle; `-denoise` is the cure, not more spp.**
      In mode `D` the media disable `-heroc`, the orb's own dispersive refraction would
      de-hero the bundle anyway, and `-herosplit` is implemented for CPU forward `A`/`B`/`C`
      and the `M`/`S` deposit only — so the caustic converges one wavelength per path and the
      variance lands almost entirely in *chroma*. `-denoise` targets exactly that: measured on
      this frame at 120 spp it cuts chroma RMSE against an 8000 spp reference to 58 % and
      gains 1.8 dB PSNR, for ~1 % of render time and zero loss of luma detail. Render this
      scene with it on.
    - **A gyroid SHELL is a diffuser, not a bank of prisms — and the plain sphere beats it.**
      The scene long asserted the opposite: its crystal gyroid was a shell (`|G| < 0.55`) on
      the theory that a gyroid is a pack of small prisms and prisms split light.
      `tools/_gemsweep.py` disproves it. A shell is a labyrinth of thin *curved sheets*, so a
      ray crosses a dozen of them and is deviated a dozen small random ways; what reaches the
      cap is a shadow with a filigree of sub-centimetre threads. The fix is not the lattice
      frequency but the *topology*: dropping the `abs` takes the field to `G < 0`, one of the
      two interpenetrating **solid networks** (50 % by volume), chunky glass with one entry
      and one exit — an actual optical body. That is worth 1.5× the caustic area and 1.2× the
      saturation at the same pitch. Measured over a bare cap in the scene's own sun, as
      coverage above 1.2× the bare level / excess-weighted saturation / peak:

      | piece | coverage | sat | peak |
      |---|---|---|---|
      | crystal **orb**, drop 0.80 | **0.72 %** | **0.246** | 5.99× |
      | **solid** gyroid k=6, drop 0.90 | 0.39 % | 0.214 | 6.12× |
      | **solid** gyroid k=10, drop 0.90 | 0.37 % | 0.208 | 5.64× |
      | shell gyroid k=6, drop 0.90 | 0.56 % | 0.172 | 5.74× |
      | shell gyroid k=13, drop 0.90 | 0.24 % | 0.173 | 5.76× |
      | **Klein bottle**, standing | 0.12 % | 0.173 | 2.01× |

      Above 1.5× everything except the orb and the solid gyroids goes to *zero*. The Klein
      bottle cannot be rescued at all: a 2.4 mm wall is optically a window, and making it
      solid would destroy the internal tube that is the piece's whole point.
    - **The clip shape is an optical component, and for the solid gyroid a THICK BOX beats the
      sphere on brightness while losing on rainbow.** `contained_by` seals a clip with the
      piece's own material, so the clip is not a mask — it is the piece's outer refracting
      envelope. A sphere clip therefore leaves a curved lens surface; a box clip caps the solid
      with flat *parallel* faces, and a plane-parallel plate displaces a beam but cannot
      converge one. That predicts the box should simply lose, and at one quarter thickness it
      does. It is wrong for thick boxes, because the focusing is not done by the cap at all but
      by the **depth of network behind it**. Swept over thickness at 960 px / 1200 spp, cut
      1.2× cap, spike rejection on, each at the best drop of its own sweep:

      | clip (half-extents 0.50 × hy × 0.50) | coverage | sat | spread | peak | fan |
      |---|---|---|---|---|---|
      | sphere r = 0.50, drop 0.90 | 0.37 % | 0.196 | **0.060** | 3.25× | 0.18 |
      | box 1 : ¼ : 1, drop 0.50 | **0.55 %** | 0.177 | 0.008 | 2.92× | 0.04 |
      | **box 1 : ⅝ : 1, drop 0.50** *(shipped)* | 0.19 % | **0.216** | 0.018 | **8.43×** | 0.13 |
      | box 1 : ¾ : 1, drop 0.50 | 0.14 % | 0.190 | 0.012 | 6.86× | 0.20 |

      Thickness trades **area for concentration, monotonically**. The thin slab covers more cap
      than the ball but all of it is wash — `spread` 0.008 is barely off the 0.003 speckle floor
      and `fan` 0.04 is nothing — and at k = 10 the cell is 0.63 m, so a ¼ box is 0.4 of *one*
      cell thick and renders as a perforated plate with big round through-holes that stops
      reading as a lattice. Both problems close by hy ≈ 0.25. At ⅝ the box has 2.6× the ball's
      peak and beats it on saturation in half the area, but **no box on the curve reaches the
      ball's chromatic spread**. `gallery_rain` ships the ⅝ box: the wide-rainbow role there is
      filled far better by the axicon (spread 0.211, fan 0.51) than any gyroid could, the room
      already has a *spherical* gyroid in gold, and a slab buys silhouette variety on top.
      Moving the clip also moves the `function`'s `translate` — the lattice phase is anchored at
      the piece centre, and 0.40 m is 0.64 of a period. **And changing a clip is a layout
      change**: the slab is wider on screen and, because its optimum drop is 0.50 rather than
      0.90, it sits 0.40 m lower, which more than doubled its silhouette overlap with the
      axicon standing in front of it (85 × 62 px for the ball → 114 × 112 px for the slab,
      `scraps/_proj.py`). Fixing that cost a documented +0.30 m override on `gallery_rain`'s
      otherwise formulaic radial spread plus a 0.25 m re-centring of the exhibit's cap — the
      only direction with any room, since −z runs the flyby into the piece (clearance
      0.252 → 0.037 m) and moving the axicon instead throws it off the left edge of the frame.
    - **…but thickness was the wrong variable. What actually sets a gyroid's caustic is the
      piece's VERTICAL EXTENT, and the winning body is a small UPRIGHT PLATE.** The table above
      swept one degree of freedom (how thick a 1 m-square pancake is) and read the answer as
      "thicker focuses better". Re-cutting the same solid as a plate standing on edge —
      horizontal : depth : vertical = 1 : *f* : 1 — shows what that sweep was really measuring.
      At 480 px / 600 spp, cut 1.2× cap, spike rejection on (raw-pixel *and* cell), each at the
      best drop of its own sweep:

      | body (h × d × v) | drop | coverage | sat | spread | peak |
      |---|---|---|---|---|---|
      | sphere r = 0.50 | 0.90 | 0.35 % | 0.196 | 0.014 | 3.25× |
      | pancake 1 × 1 × 0.625 *(was shipped)* | 0.50 | 0.15 % | 0.205 | 0.009 | 6.07× |
      | upright 1 : 5⁄16 : 1, **1.000 m** | 0.70 | 0.04 % | 0.200 | 0.021 | 4.60× |
      | upright 1 : 5⁄16 : 1, **0.875 m** | 0.58 | **1.12 %** | 0.197 | 0.014 | 4.15× |
      | upright 1 : 5⁄16 : 1, **0.750 m** | 0.51 | 0.88 % | 0.209 | 0.020 | 7.46× |
      | **upright 1 : 5⁄16 : 1, 0.625 m** *(shipped)* | 0.45 | 0.65 % | **0.212** | 0.019 | **13.81×** |

      Two things fall out. First, **the proportion is not what matters — the size is.** Hold the
      1 : 5⁄16 : 1 shape fixed and shrink it and `peak` climbs monotonically to 13.81×, while the
      *full-size* version of the very same shape is the worst row in the table (coverage 0.04 %,
      a twentieth of its own 0.625 m sibling). Under a near-overhead sun it is the **vertical**
      dimension that sets how much network a ray traverses, and ~0.625 m is where that path
      focuses; 1.0 m over-diffuses. Second, **at equal path length the upright plate still beats
      the pancake** — 2.3× the peak and 4.3× the coverage, on a plan aperture a *twentieth* the
      size — because a pancake presents parallel faces square to the beam, so only the network
      does any work, whereas a plate turns that network's output out through its perpendicular
      side faces. The optimum is a plateau, not a spike (*f* = 0.25…0.375 and drop 0.35…0.45 all
      give peak 10–14×), so it is robust; *f* = 0.1875 and *f* ≥ 0.5 both fall off to ~3.5–5×.
      The cost is **size**: at k = 10 the cell is 0.63 m, so the shipped plate is exactly one unit
      cell square and under a third of a cell thick — it reads as a single cell seen edge-on
      rather than as a block of lattice. The 0.750 m row is the documented fallback if that ever
      reads too small, since it still beats the pancake on peak *and* coverage.
    - **Beware ranking pieces across resolutions.** `coverage` is a cell count and is
      resolution-stable (the solid gyroid: 0.35 % at 480 px, 0.37 % at 960 px), but `spread` is
      **not** — same piece, 0.014 at 480 px against 0.060 at 960 px — because it is computed
      over whichever cells clear the threshold, and how many that is depends on resolution. The
      box/sphere ranking on `spread` *inverts* between 480 px and 960 px. Rank at one
      resolution or not at all; the proper fix (meter at a fixed world-space cell size) is
      logged in `known-issues.md`.
    - **Brightness and colour are separate properties, and the shape that gives both is a
      cone.** A caustic is *bright* because a surface **converges** light and *coloured*
      because it disperses light **sideways**, and the two normally exclude each other. A ball
      lens is concentric, so its dispersion is purely *longitudinal*: every wavelength lands on
      the same axis a little deeper and they stack into one white disc with a tinted rim. A
      prism disperses hard sideways but is parallel-in/parallel-out, so it never converges and
      its coloured band never rises above the bare sunlight beside it — it scores a flat 0 % on
      any threshold worth using, which is the answer and not a bug. An **axicon** (flat top
      over a cone) breaks the trade-off: light enters the top undeviated, and every point of
      the conical exit face is a prism at the *same* tilt, so the deviation is constant (a line
      focus — bright) while the dispersion is a prism's (coloured).
    - **`sat` cannot tell a uniform tint from a rainbow; `spread` can.** `sat` is distance from
      white, so a uniformly amber patch and a red-to-violet fan score alike. `_gemsweep.py`
      therefore also reports **spread**: per caustic cell take the chromaticity
      (r, b) = (R, B)/(R+G+B), find the excess-weighted centroid, and report twice the weighted
      RMS radius about it. White or uniformly tinted collapses to a point; a real spectrum is a
      long streak. **`spread` in turn needs two controls, and is worthless without them** —
      mode `D` carries one hero wavelength per sample, so an unconverged pixel is *randomly
      coloured* and an RMS radius scores a loud random cloud exactly like a rainbow (a 4× box
      only divides speckle by 4). So `_pfm.py` — shared by both rigs — also gives **`noise`**,
      the same statistic over a control band of cells at 1.0–1.2× the median (bare lit screen,
      no caustic, hence the render's speckle floor), and **`fan`**, the fraction of chromatic
      variance explained by a weighted least-squares *quadratic in position* (adjusted R²).
      Dispersion means chromaticity is a **function of position**; speckle is uncorrelated with
      position and scores ~0 however loud. The basis must be quadratic: a first draft fitted a
      plane and scored the axicon 0.09, because an axicon disperses *radially* about its axis —
      red outside, violet inside, on both flanking cusps at once — which a plane cannot see by
      symmetry. The shipped axicon scores **fan 0.76** in the rig (control 0.00, floor 0.008)
      and **0.46** in the finished frame. The orb scores fan 0.94 on spread 0.013, which is the
      pair working as designed: a ball lens's tinted rim is perfectly organised colour, there is
      just almost none of it. Measured at the honest 2× bar on the **float** buffer with
      `-fireflies 3` (600 spp, 480², 4× box), each piece at its own best drop — this supersedes
      an earlier table taken through the 8-bit clamp, which was clip-suppressed at the top end
      and firefly-inflated at the tail, *in opposite directions*, and so cannot be rescaled:

      | piece | peak | coverage | sat | **spread** | noise | **fan** | patch x × z |
      |---|---|---|---|---|---|---|---|
      | **axicon**, 45°, apex down, drop 0.35 | **14.68×** | **0.30 %** | 0.275 | **0.079** | 0.008 | **0.76** | **1.37 × 0.96** |
      | axicon, same, drop 0.80 | 24.58× | 0.29 % | **0.314** | 0.083 | 0.011 | **0.85** | 1.98 × 1.34 |
      | apex-**up** cone, drop 0.90 | 7.14× | 0.07 % | 0.201 | 0.018 | 0.005 | — | 1.49 × 0.35 |
      | round **brilliant** cut, R = 0.40 | 3.68× | 0.06 % | 0.219 | 0.047 | 0.011 | — | 0.35 × 0.06 |
      | oblate spheroid (astigmatic lens) | 9.09× | 0.19 % | 0.272 | 0.024 | 0.004 | 0.97 | 0.09 × 0.18 |
      | crystal **orb**, drop 0.80 | 3.00× | 0.16 % | 0.253 | 0.013 | 0.008 | 0.94 | 0.09 × 0.15 |
      | glass torus (ring lens) | 7.85× | 0.03 % | 0.210 | 0.006 | 0.003 | — | 0.03 × 0.06 |
      | **solid gyroid k=10, shell gyroid k=13, Klein bottle, prism** | 2.60× | **0.00 %** | — | — | — | — | — |

      The orb is the scene's brightest ball-lens caustic *and* one of its whitest — fan 0.94 on
      a 9 cm patch is real, organised and negligible, which is the whole reason the axicon was
      added. Three results are worth keeping. A **round brilliant loses**, because a 40.75°
      pavilion sits just past crystal's 40.2° critical angle and total-internally-reflects the
      fire back up at the viewer instead of down at the table (sweeping the pavilion to 20–35°
      does not recover it) — and its one-time "spread win" of 0.092 was purely a clamp artifact:
      on float it is 0.047 with too few cells to `fan`-test. **The lattice cannot be rescued by
      reshaping its outer boundary** — a solid gyroid clipped to this same axicon measures
      0.13 % / 0.205 / 0.099, less than half a plain axicon, because the clip only sets the
      *first* surface a ray meets and behind it are the same internal sheets that make a gyroid
      a diffuser. And **faceting the axicon still loses on honest numbers**: 6/8/12/16 pavilion
      facets read 0.11–0.14 % coverage on a ~0.20 × 0.03 m sliver (1/220th the smooth cone's
      patch) at a third of the peak, none with enough cells to `fan`-test, because sampling the
      ring focus at *n* discrete azimuths instead of continuously collapses it.
    - **An axicon's SIZE is a rainbow knob, and unusually it costs nothing in saturation.** A
      bigger aperture normally buys a brighter, *whiter* patch. An axicon's does not, because
      it has no focal point: the cusps land at a distance that scales with the piece, so
      growing it simply pushes the wavelengths apart that were previously landing on top of
      each other. Sweeping the half-height *h* (top radius 2*h*) in SF10 at drop 0.65, 480 px,
      cut 1.2× cap — one resolution, so these rank only against each other:

      | *h* | top radius | coverage | sat | **spread** | **fan** | patch x × z |
      |---|---|---|---|---|---|---|
      | 0.28 *(shipped)* | 0.56 | **0.56 %** | 0.327 | 0.107 | 0.65 | 1.43 × 0.44 |
      | 0.40 | 0.80 | 0.29 % | **0.527** | 0.128 | **0.76** | 1.93 × 0.47 |
      | 0.52 | 1.04 | 0.60 % | 0.445 | **0.172** | 0.44 | 2.48 × 1.90 |

      `spread` rises monotonically with size and `sat` rises with it too; *h* = 0.40 is the most
      **organised** colour measured anywhere in `gallery_rain` (fan 0.76), and *h* = 0.52 the
      widest rainbow, with `fan` falling to 0.44 only because the two cusps grow into each
      other. **The shipped size is the layout optimum, not the optical one** — *h* = 0.52 is a
      2.08 m piece, larger than anything else in the room, throwing a patch bigger than its own
      1.8 × 1.7 m cap. Growing an axicon means re-siting and re-capping it.
    - **The axicon is cut from DENSE FLINT, not crystal, and it is the only piece that is.**
      What splays a caustic across the spectrum is the Abbe number: `glass:SF10` (V_d 28.5)
      splays 1.5× as far as `glass:crystal`/F2 (36.3). The orb and the gyroid keep crystal —
      their caustics are white-with-a-rim whatever the glass — but the axicon exists only to
      put colour on a screen, so it gets `material "flint"`. **A denser glass also deviates
      harder, so it moves the ring focus and the drop has to be re-swept with the material**:
      SF10 at crystal's optimum drop of 0.35 lands in a null (0.08 % coverage), and its own
      optimum is 0.65. Head to head at matched rig settings (960 px, 1200 spp — the 480 px
      sweep setting is not fine enough to adjudicate this, and the metrics are
      resolution-dependent, so only compare rows taken at the same resolution):

      | glass | drop | peak | coverage | sat | spread | fan | patch |
      |---|---|---|---|---|---|---|---|
      | crystal | 0.35 | 16.44× | **0.28 %** | 0.321 | 0.152 | 0.51 | 1.37 × 0.98 |
      | SF10 | 0.50 | **24.71×** | 0.13 % | 0.509 | 0.162 | 0.57 | 1.27 × 1.05 |
      | **SF10** | **0.65** | 21.99× | 0.15 % | **0.516** | **0.187** | **0.77** | 1.41 × 1.14 |

      In the finished frame that is **sat 0.269 → 0.444, spread 0.204 → 0.299, fan 0.37 →
      0.76, peak 6.3× → 8.1×** for 80 % of the caustic area — and it *clips less* (0.13 %
      against 0.52 %), because the caustic got smaller as it got brighter. That is not a
      tail-selection artifact of the smaller patch: sweeping the caustic threshold until the
      two match on area makes crystal *worse* (at 3.37 % coverage it reads sat 0.210, spread
      0.117), so at matched coverage flint wins by 2.1× on sat rather than 1.7×. **The
      mechanism is where in the patch the colour sits.** Crystal's caustic gets whiter toward
      its core (sat 0.269 → 0.179 from 2× to 3.5× the pedestal) — its colour is in the dim
      fringe, which the tone map crushes — while SF10's sat is flat at 0.34–0.37 out to 4×,
      so the colour survives into the bright pixels. That is the whole resolution of "meters
      as coloured, looks white". Two other levers
      were ruled out by measurement first: darkening the screen cannot work (chromaticity is
      scale-invariant, so albedo moves caustic and pedestal together), and cutting the sky
      fill would buy almost nothing, because the cap's pedestal profiles as 0.0089 in the
      piece's shadow (fill alone) against 0.0893 sunlit — **the fill is 10 % of it** and the
      other 90 % is the same sun the caustic comes from.
    - **The axicon has a 4 cm GIRDLE, and that is the only gem cut it can afford.** Asked to
      shape it "more like a diamond", the answer is that a diamond cut is an *anti-caustic*
      shape — but "how much gem silhouette can it carry" is a different question from "should
      it be a brilliant", and it has its own sweep (`tools/_gemsweep.py`, piece `gcone`; SF10,
      drop 0.65, 480 px / 600 spp, and **`GEMBOX=0.7`** — see the box caveat below). Everything
      added *above* the girdle plane, leaving the 45° conical exit face untouched:

      | above the girdle plane | peak | coverage | sat | spread | patch x × z |
      |---|---|---|---|---|---|
      | nothing (a bare cone) | 35.79× | 0.12 % | **0.509** | **0.282** | 1.40 × 1.17 |
      | girdle 0.02 | 36.41× | 0.15 % | 0.480 | 0.273 | 1.46 × 1.20 |
      | **girdle 0.04 — SHIPPED** | **37.07×** | **0.17 %** | 0.463 | 0.254 | **1.49 × 1.22** |
      | girdle 0.08 | 27.35× | 0.19 % | 0.392 | 0.226 | 1.60 × **0.38** |
      | girdle 0.04, 16 vertical facets | 32.41× | 0.17 % | 0.428 | 0.242 | 1.49 × **0.38** |
      | girdle 0.04, 8 vertical facets | 13.83× | 0.15 % | 0.377 | **0.099** | 1.49 × **0.35** |
      | + crown 34.5° / 75 % table | 3.75× | **0.01 %** | 0.272 | 0.036 | — |
      | + crown 34.5° / 53 % table (a real brilliant) | 3.48× | **0.01 %** | 0.229 | 0.004 | — |
      | + crown 20° / 60 % table | — | **0.02 %** | 0.254 | 0.041 | — |

      **A straight girdle is better than free.** A ray entering the flat table crosses no
      interface until the cone, so it arrives there on the same line it always did, just
      starting 0.04 m higher; only the rim is touched. 0.04 m comes out *brighter* and 42 %
      wider-covering than the bare cone, for ~9 % of `sat`. Past that the band eats the rim:
      at 0.08 the patch collapses from two cusps (1.17 m of z) to one band (0.38 m). Sixteen
      **vertical** facets — which by construction cannot deviate the descending aperture at
      all — collapse it the same way, and eight facets are far worse again (spread 0.099, peak
      13.83×), so faceting costs monotonically with coarseness and the glinting waist is not
      worth its cost at any count. **A crown
      is fatal**, which is the round-brilliant row of the table above arrived at from the other
      direction: a crown facet at angle *c* bends a descending ray inward by
      *c* − asin(sin *c* / n) — 15.6° at *c* = 34.5° in SF10 — so the whole annulus outside the
      table meets the cone at the wrong incidence. Three crown geometries all read 0.01–0.02 %
      coverage, i.e. *no caustic*. That is the cut working as designed: a brilliant throws its
      fire **up at the viewer**, and flint makes that worse than crystal (critical angle 35.4°
      against 40.2°), so the flint recut moves *away* from a gem cut, not toward one.

      **Confirmed in the finished frame, not just in the rig**, at matched convergence
      (`-device cpu`, 800×450, control 324 spp / 5.56 % noise against girdled 305 spp /
      5.73 %), which matters because the rig floats one piece over a bare white cap while the
      real exhibit sits on textured marble under a partly-shadowed sky:

      | axicon cap | coverage | sat | spread | xspread | peak | clip | noise | fan |
      |---|---|---|---|---|---|---|---|---|
      | pre-girdle | 2.40 % | 0.324 | 0.130 | 0.201 | 4.94× | 0.13 % | 0.059 | — |
      | **0.04 girdle** | **2.77 %** | 0.335 | 0.135 | 0.205 | **5.17×** | 0.15 % | 0.062 | **0.80** |

      Every axis moves the right way and the caustic becomes *more* organised (`fan` goes from
      unmeasurable to 0.80). The rig's +42 % coverage lands as +15 % in frame — the shipped
      exhibit is shadowed and textured, so the rig overstates the gain, which is the expected
      direction and the reason both are measured. Every other cap is unchanged to three
      decimals (gyroid 2.84 %/0.444→0.445, glass 0.00 % both), i.e. no side effects.
    - **`contained_by` is a hard clip in the gem rig, and its default box was shaving the
      control.** `tools/_gemsweep.py` boxed every piece at ±0.5 m while `vcone1.0`'s girdle
      radius is 0.56, so every axicon number the rig printed before the girdle sweep was
      measured on a cone with four flats cut into its rim. Harmless for *ranking* shapes that
      all share the box, which is why the default is unchanged — but `GEMBOX=0.7` now exists,
      and any absolute axicon number must say which box it came from. (The rows above are all
      `GEMBOX=0.7`; the rows in the piece-ranking table further up are all ±0.5.)
    - **Metrics do not replace looking at it: `tools/_capcrop.py`.** It crops a cap's screen
      footprint out of the float buffer and prints it three ways — *as shipped* (exactly the
      PNG), *under-exposed* (gain set so the cap's own 2×2 peak lands just under white), and
      *chromaticity only* (renormalised to equal luminance, saturation stretched). The three
      rows separate three failures that all look alike: row 1 white but row 2 coloured means
      the **tone map** is eating real colour; both white with a smooth row 3 means the colour
      is real but **weak**; a confetti row 3 means there is no colour at all, only speckle
      (this is `fan` made visible). The crystal axicon printed the middle case — a pale arc
      with a faint warm fringe over a smooth but low-amplitude hue gradient — which is what
      sent the investigation to the glass rather than to the exposure.
    - **Drop is not a colour parameter — it buys patch AREA.** Swept on float, the axicon's
      `spread` is flat at 0.065–0.083 from drop 0.35 to 1.10 while the patch grows 1.37 × 0.96 m
      → 2.33 × 1.55 m; peak peaks near 0.65 (34.5×) and `sat`/`fan` near 0.80 (0.314 / 0.85).
      An earlier claim that "colour falls off monotonically above ~0.5 m of drop" was the clamp
      talking: a bigger drop threw a *brighter* caustic, which clipped harder, which the PNG
      scored as less colourful. So drop is a **staging** trade — a bigger drop needs a bigger
      cap — and the shipped 0.35 stays because growing the cap to ~2.1 × 1.45 would collide it
      with the diamond cap in both z (4.41 vs 4.40) and y (0.70–0.90 vs 0.65–0.85), for a gain
      inside the run-to-run scatter.
    - **45° is the right cone angle, but NOT because of TIR.** Sweeping the half-angle at drop
      0.35 (k = tan of the half-angle) gives coverage 0.09 / **0.03** / **0.30** / 0.25 / 0.10 %
      at k = 0.70 / 0.85 / 1.00 / 1.20 / 1.40 (35 / 40.4 / 45 / 50.2 / 54.5°). The collapse is
      at 40.4°, **below** 45°, and both 50.2° and 54.5° keep working — so this is not a
      one-sided cliff past 45° as was once written. Across the null the caustic core also
      **switches sides**, +0.63 z at k = 0.70 to −0.61 z at k = 1.20: the ring focus is passing
      through infinity there (the constant prism deviation sweeping past the drop distance),
      which is what empties the 2× bar. 40.4° landing on crystal's 40.2° critical angle is a
      coincidence, and worth naming as one so the wrong mechanism is not re-derived from it.
      Shallow cones are also short-range only — `vcone0.70` at drop 0.80 and 1.10 and
      `vcone0.60` at 0.80 all read 0.00 % coverage.
    - **An axicon's caustic barely walks with height, and lands beside the piece rather than
      under it.** A lens throws its focus ~0.98 m downwind per metre of drop (the sun walks
      +0.2079 x / −0.9788 z), which is why the orb's and the gyroid's caps are cantilevered
      most of a metre. An axicon has no focal *point* to displace — it has a focal *line*
      starting at the exit face — so at drop 0.65 in SF10 the core sits within 0.06 m of its
      own axis and the cap is pulled only 0.06 m in x and 0.04 m in z. What it *does* need is
      **width**: the caustic reads as two bright rainbow cusps flanking the piece left and
      right, and 90 % of its light falls within x −0.77…+0.65 m but z −0.61…+0.53 m, so its cap
      is wide and shallow (1.6 × 1.28) where every other cap is roughly square or deep. It grew
      from 1.1 to 1.28 deep with the flint swap: the extra height separates the cusps, taking
      the patch from 1.37 × 0.98 to 1.41 × 1.14. Both the extent and the core offset are
      reported by `_gemsweep.py` as excess-weighted 5–95 % quantiles, *not* a raw bounding box:
      a handful of stray sparkle cells at the frame edge trebled the raw box.
    - **The scene's colour source is a tenth exhibit, the CRYSTAL AXICON** (named for the
      gallery label; it is cut from flint, see above), added rather than swapped in because
      nothing already present could be made to do the job (the lattice cannot, whatever its
      outer shape, and the Klein bottle is a window). A 45° cone, 1.12 m across, hung **apex
      down** on three pins with its point 0.37 m over the tabletop, at near-left
      (2.60, 5.45). It is the only exhibit authored directly in **world
      coordinates with no `group { translate }` wrapper**, because unlike the other nine it has
      no earlier layout position to be displaced from. Its site is checked four ways: 0.37 m
      clear of the nearest cap (`_standaudit.py -v`) and 0.24 m clear in z of the crystal
      gyroid beside it, 0.94 m clear of the flyby (`_flyplan.py`), 29.5° off the still
      camera's axis against a 40.9° half-width, and it balances a frame whose left half was the
      original reason the scene lost its roof. **The near row is high z, not low z** — the
      camera stands at z=9.35 looking toward −z, so "front of the gallery" is z ≈ 5–7. Siting
      it at z=1.10, which reads as "front" on the page, put it 8.4 m out and directly *behind*
      the crystal gyroid, which occluded it completely; the error was invisible in the scene
      text and obvious the moment it was rendered. Project the candidate point through the
      camera basis before committing to a position.
    - **`rotate` is not a valid key inside an FTSL `function` block** (only `translate` is), and
      the loader emits a **warning, not an error**, then carries on — so a rotated field
      silently renders unrotated. A prism roll sweep came back as six identical rows before
      this was found. Two fixes: rotate half-space normals *algebraically* in the generator (a
      half-space `n·p ≤ d` under `p = Rq` is `(Rᵀn)·q ≤ d`, so pre-rotating the normals is
      exact and needs no transform support), and gate every generated scene through
      `-parseonly`, failing loudly on any warning rather than measuring the wrong object.
    - **Per-pixel chroma is unmeasurable in mode `D`, so the metric has to downsample first.**
      Mode `D` renders one hero wavelength per sample, so a dispersive caustic arrives as
      chromatic *speckle* and per-pixel saturation measures the sampler, not the optics. The
      speckle is zero-mean in chroma over a neighbourhood — the same fact `-denoise` is built
      on — so `_gemsweep.py` box-averages 4× in **linear** light before asking how saturated
      the result is. A centroid-split metric (‖centroid_R − centroid_B‖) was tried first and
      is hopelessly noise-dominated. **Coverage, not peak, is the number that separates a
      filigree from a pool:** a thin web of threads and a broad disc can have identical peak
      brightness and read nothing alike.
    - **The 1.6× spread is authored as `group { translate }` wrappers, not as rewritten
      coordinates.** Nine exhibits displaced radially about (5.00, 2.72) is ~90 numbers if
      done by hand, and every comment quoting one of them would go stale. A `group` composes
      onto `isosurface` children including all 8 `contained_by` corners, and a pure
      translation is a positive-diagonal map so `im.boxOriented` stays false and no
      oriented-box clipping penalty is paid. Each exhibit therefore gets *two* wrappers — one
      round its stand, one round its piece — carrying the same offset, and every relationship
      *within* an exhibit (cap over column, pin against sphere, settled rest pose) stays
      literally true as authored. Only clearances quoted *between* exhibits go stale, and
      every one of those gaps grew, so they read conservative rather than wrong.
    - **Each cap is as thick as its own stand's base slab (0.18–0.24 m), and the thickness was
      added DOWNWARD.** The stone-era caps were 3 cm — a sheet of paper laid over a wire frame,
      with a slab eight times thicker at the bottom of the same stand, so the two ends of one
      object disagreed about what it was made of. Matching each cap to its own slab (rather
      than picking one global number) keeps each stand internally consistent. It has to grow
      *downward* because the hero pieces were dropped onto these caps by `settle_scene.py` and
      their rest poses are literal coordinates: moving a top face by a centimetre leaves a
      piece floating or sunk, whereas the bottom face touches nothing. Side effect worth
      knowing: the gyroid plinth's middle cage tier is only 0.28 m tall, so a 0.24 m cap covers
      all but ~7 cm of it and that stand now reads as two slabs with a wire gap rather than
      three tiers.
    - **Cap-vs-cap clearance is not enough; caps must be checked against neighbouring
      COLUMNS.** A cap used to occupy one thin y slab, so caps could overlap in plan freely;
      now that they are 0.18–0.24 m deep that is no longer automatic, which is another reason
      the audit is not optional. A column spans a whole y range, so plan overlap *is*
      intersection. `scraps/_standaudit.py`
      brace-parses every cage's outer box and every cap box and checks all 42 colliders across
      the 12 stands in 3-D; run it after any stand edit (`-v` also lists every cap's world
      footprint, which is what you need in front of you before siting a new exhibit — the
      question is never "is there floor", since there are no walls, but "how wide a cap fits
      between its neighbours"). It is **spread-aware** — it
      accumulates every enclosing `group`'s translate onto each collider, because the boxes
      are authored in the pre-spread frame and reading them off the page would audit a layout
      the scene no longer has.
    - **The flyby threads five pieces, and the ring is threaded *toward* the hall.**
      `camera_curve "fly"` is a closed 45-point loop through the gold gyroid's empty-air
      channel (gallery_settled's validated points shifted the same +1.25 z the ball was), the
      FUR CREATURE's coat, the glass orb, the Klein bottle's bulb, and the chrome ring's bore.
      Three things are
      non-obvious. (a) The Klein pass height is **measured**, not guessed: the quoted bounding
      box includes the handle and the *foot ring*, not the body, sits on the mesh origin, so
      `scraps/_kleinslice.py` walks the real vertices through the mesh/settle/spread chain and
      finds the body is a clean circle of radius 0.151 at y=1.12 — the widest chord, and the
      height at which a horizontal cut shows four loops. (b) The ring's `rotate 90 0 0` puts
      its axis along z, so the pass must be nearly parallel to z, and it runs **+z** because
      rendering it −z showed the camera emerging at the back-left corner into an empty frame;
      reversed, the bore is a reveal onto the whole hall. (c) `density_at` stops are
      arc-length fractions, and this loop is wildly unevenly spaced (0.27 m between channel
      points, 1.3 m across the cruise), so naming the dwells by point index would put every
      one of them in the wrong place — `scraps/_flyplan.py` converts the beats and also tests
      the **spline** (not the chord polyline, which cannot see the curve bow outside its own
      control points on a turn) at 2 cm against every cage, cap and hero piece. Until 0.155.0
      the loader was reading those stops as control-point-index fractions anyway, behind the
      scene's back; see the `density_at` entry below. (d) **Which waypoint is written first is
      a free choice about the film, not about the path** — a `closed` centripetal Catmull-Rom
      is invariant under a cyclic rotation of its control points, so rotating the list moves
      only where frame 0 falls. gallery_rain's loop originally opened one metre off the gold
      gyroid's clip sphere pitched 52° down into it — an extreme close-up of a lattice,
      mid-dive, with nothing establishing where any of it was — so the two closure-arc points
      were moved to the front (2026-09-01). Frame 0 now sits at the loop's furthest-back,
      highest-looking-in point aimed down the centre line, within 5° of the still camera's own
      framing, and the dive starts at frame 18. **The catch, and the reason this is worth
      recording:** `density_at` stops name positions *on the curve*, so a phase rotation is
      correct only if every stop shifts by the same arc-length delta (here +0.0555) — leaving
      them put would slide each dwell 1.33 m off its beat and start the creature pass inside
      the gyroid channel's exit. `tools/flyphase.py` does both halves and, unlike
      `scraps/_flyplan.py`, parses the scene instead of keeping its own copy of the control
      points, so it cannot drift out of step with an edited curve. Verified as a pure phase
      change by sampling both curves at 4000 arc positions: **max deviation 7.3e-11 m**, with
      identical spline length (23.872145 m) and identical `Cmax`.
    - **`density_at`'s `t` is normalized ARC LENGTH, and until 0.155.0 the loader read it as a
      normalized control-point INDEX** (`src/ftsl.h`, the camera-curve sampler). Every scene,
      every comment and both docs described it as a position along the curve; the code
      evaluated `densityAt(g / nSeg)`. On an evenly-spaced curve the two agree, which is why it
      survived — on gallery_rain's loop, where one leg packs seven points into 0.27 m and
      another spends 1.3 m on two, the dwell meant for the glass orb was landing short of it.
      The fix makes the sampler **two-pass**, because arc length is not known until the curve
      has been walked and the obvious single-pass version has nothing to feed `densityAt`
      except `g/nSeg`: pass one walks the spline at `max(64, 64·nSeg)` samples accumulating
      *pure* (density-free) arc length, pass two re-integrates `densityAt(s/s_total)·ds` into
      the cumulative-count table the frame placement actually inverts. Note that this fixes
      only `density_at`; the lens/orientation tracks (`roll_at`, `fov_at`, `zoom_at`,
      `fstop_at`, `focus_at`) run on a **third** clock, the frame fraction `i/N`, and were
      always correct. `scraps/_flyplan.py` had the mirror-image bug — it measured the *chord
      polyline*, which cannot see the curve bow outside its own control points on a turn — and
      was moved onto a transcription of `catmullRomAt` at the same time, so its numbers and the
      loader's are now the same numbers.
    - **The fur creature is the eleventh exhibit, and it is the only one authored entirely in
      BAKED WORLD COORDINATES.** `fur { }` is a top-level-only block — the loader rejects it
      inside a `group` — so a placed groom cannot borrow the spread's `group { translate }`
      idiom that every other exhibit uses. `scraps/_bakecreature.py` therefore carries the
      whole body/face/fur table from `scenes/fur_creature.ftsl` and emits the placed block
      through `p_world = PIVOT_W + S·R_y(yaw)·(p_src − PIVOT_S)` with `S = 1.6`, `yaw = −64.4°`,
      pivot (6.45, 0.90, 4.40) — every sphere centre *and* every fur `direction` pre-rotated.
      Three things make this work. (a) **The scale is free.** Fur coverage goes as
      density × radius × length, so scaling `length`/`radius`/`curl`/`clump_size` by S and
      `density` by 1/S² gives a visually identical coat at S× size with the *same* strand count
      and the same memory — 1.6× costs nothing. (b) **The yaw is not a pose choice, it is what
      the flyby demanded.** `fur` combs nose-to-tail, so a pass running against the grain reads
      as a hedge; the camera's velocity has to be −F. −64.4° is the yaw at which the animal's
      long axis already lies along the existing curve point (4.78, 1.35, 5.20)→pivot, which
      normalizes to (+0.9018, −0.4320) = −F to four places, so the reroute cost two waypoints
      instead of a rebuilt leg. It also still leaves the still camera at (5.0, 2.95, 9.35) a
      48°-off-head-on three-quarter *front* view, which the two constraints otherwise pull
      apart. (c) **The pass height is solved, not chosen.** `_bakecreature.py` sweeps it and
      reports, per height, the shortest distance to any skin sphere and the total chord inside
      the coat; 1.295 buries the lens in the head and 1.34 grazes only the back, so the curve
      holds y = 1.308 dead level across four control points — 12.8 mm of clear air off the head
      (the closest this camera comes to solid geometry anywhere in the scene) while threading
      rump, barrel, chest, neck, head and tail brush for 1.16 m at 21 mm/frame. Its cap
      (1.10 × 0.90) was sized from the real footprint *including* the coat.
    - **The gem compote is the twelfth exhibit, and it is the scene's worked example of
      giving one GLB thirteen different materials.** The asset (`meshes/compote_with_gems.glb`)
      is a crystal dish holding twelve faceted stones as thirteen nodes. It **must not** be
      loaded with `import_materials yes` — originally because ftrace's glTF reader implemented
      no KHR material extension, and this file keeps all of its colour in
      `KHR_materials_volume`/`transmission`, so every one of the thirteen arrived as
      baseColor (1,1,1) / metallic 0: thirteen identical **white diffuse** surfaces, i.e. the
      asset destroyed. `src/gltf.h` reads that extension family now (0.224.0), so an import
      would at least arrive as *glass* — but not as **this** glass, and the reasons are
      permanent rather than a missing feature: KHR's attenuation is one RGB triple with **no
      dispersion**, whereas the twelve stones are authored on SF10 (Abbe ≈ 28) precisely so
      their caustics split; and glTF has no way at all to express the nested-dielectric
      `priority` chain below. The choice is now about spectra and interface bookkeeping.
      The only lever for per-node materials is **`skip_material`**, which
      drops primitives whose glTF material *name* matches a substring and works independently
      of `import_materials`; so the scene loads the file **thirteen times**, each load skipping
      the other twelve names, and binds a hand-authored dielectric to each. Two consequences
      worth knowing before copying the idiom: the patterns must be **comma-joined into one
      token** (FTSL starts a new statement at the second bareword), and thirteen loads of a
      4.5 MB GLB cost thirteen parses but only 250,744 triangles. The colours are carried
      across exactly, by **σ = −ln(attenuationColor)/attenuationDistance** (glTF attenuates as
      `color^(d/distance)`, ftrace's `absorb` is a Beer–Lambert σ_a in 1/m), then **divided by
      the display scale 3.0** so that enlarging the piece holds its authored optical depth
      instead of cubing the transmittance to near-black. All thirteen carry a **distinct**
      `priority` (bowl 1, stones 2–13): the stones outrank the dish so a facet sunk in its wall
      refracts as gem, and ranking the stones against *each other* is what silences the 22
      overlapping-bounds ties a shared priority produced (a heap of stones in a dish has
      thoroughly interleaved AABBs even though the solids are disjoint). Sited at (7.35, 6.20)
      in the front-right court — the one court with no flyby control point in it — after two
      measured rejections recorded in the scene: the middle-of-layout hole (57 × 74 px, and it
      covered a third of the dumbbell behind it) and the front-left court (214 × 226 px over
      the axicon, the one exhibit that exists to throw a coloured caustic).
    - **Mode `M` is not an option for this scene** even though it is the caustic-friendly mode
      on paper: `photonmap_render.h` has no participating-media code, so M renders the cloud,
      the rain and the bow away entirely — and, unlike mode `U`, does not refuse the scene or
      warn. Logged in `known-issues.md`. Mode `D` is what the scene's `prefer{}` asks for and
      is the one mode documented as handling superposed bounded heterogeneous media on both
      devices.
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
  The **CPU** backward reference (modes R/W/V) does not sample the *emission* grid — volumetric
  blackbody emission stays forward-only — but since 0.254.0 it samples the media themselves
  correctly, superposing the whole vector with each region's `bounds` and `density` field,
  exactly as the GPU backward megakernel does.
- **`rng.h`** — Pcg32 + `seedUnit(rng, unitIndex, salt)` splitmix64 mixing:
  **every work unit (photon or pixel-sample) seeds its own stream**, so results are
  independent of chunk splits / thread count / banding / `-resume` boundaries.
  `uniform()` is the ordinary `[0,1)` draw on the 24-bit grid `{k/2^24}`;
  **`uniformOpen()` (0.199.1)** is the `(0,1)` variant that every *exponential* draw must
  use. `uniform()` returns exactly 0 once in 2^24 draws, and the inverse-CDF free flight
  `ta - log(1-u)/sigma_t` then lands exactly on `ta` — the ray origin — putting two path
  vertices on the same point, which makes any direction reconstructed between them `0/0`.
  `uniformOpen()` returns an interior point of that same bin instead of its lower edge:
  one `next()`, unbiased, every other grid point bit-for-bit unchanged. Call sites:
  `sampleMediumCollision` (free flight + delta tracking), `mediumTransmittance` (ratio
  tracking, where a zero-length step also double-charged one point's null-collision
  factor), `backward.h`'s GRIN free flights, and the three CUDA twins. `render_cuda.cu`'s
  `DRng` carries the same pair — but note that with `using Real = float` there, ANY flight
  shorter than the position's float ULP collapses the two vertices, so the sampler fix is
  necessary and not sufficient and the consumers carry explicit guards (see `bdpt.h`).
- **`render_cuda.cu`** (~7000) — the whole GPU backend: megakernel + wavefront
  forward paths, GPU R and BDPT, M deposit/gather, device twins of hero sampling.
  GPU backward (`bkRadiance`) supports **participating media** natively since
  0.23.0 (free-flight `dMediaSampleCollision` competing with the surface hit,
  volume NEE `bkNeeVolume`, Beer–Lambert `dMediaTransmittance` on NEE + throughput,
  HG scatter + albedo Russian roulette) — homogeneous *and* heterogeneous, and over the
  **whole** `scene.media` vector by Poisson superposition (bounds regions and density
  fields honoured). This was strictly ahead of the CPU twin until **0.254.0**, which closed
  the gap: `backward.h` now calls the *same* `Renderer::sampleMediaCollision` /
  `Renderer::mediaTransmittance` the forward tracer uses (via `BackwardRenderer::mediaTr`,
  which short-circuits a vacuum scene so media-free renders stay bit-identical), instead of
  collapsing everything to `media.front()` as an unbounded homogeneous haze. `Scene::
  backwardMedium()` and the `[medium] …` warning are gone. Besides fixing modes `R`/`W`/`V`
  and `P`'s camera layer, this fixed **mode `M`'s `-pmfg` final gather**, which borrows
  `neeLight` for its direct term and so was multiplying every shadow ray in `gallery_rain`
  by the raincloud's `exp(−σ_t·d)` at 10–30 m — deleting the direct lighting outright
  (M-FGDARK, known-issues.md).
  A second CPU/GPU-shared gap: mode `W`'s quadrature covers only the *surface* estimators —
  the fog branch is still an analog free flight plus a one-sample volume NEE on both devices,
  so a medium makes mode `W` speckled (deterministic, but not noise-free) at `-spp 1`.
  Spectral **rainbow-phase** media run on the device too since 0.37.0 (M10):
  a per-medium λ×µ Airy phase table + per-λ CDF is uploaded, and the unified
  `dMedPhase`/`dMedPhaseSample` dispatch (bilinear table eval / CDF importance-sample
  vs analytic HG) replaces the raw `hgPhase` calls across forward, backward, and BDPT.
  **GRIN (gradient-index) media** run on the backward reference (mode `R`) on-device
  too since 0.38.0 (M11): `dGrinMarch` (render_cuda.cu) is the device twin of
  `grin::march`, carrying its running Eikonal state in double, and `bkRadiance` marches
  each bounce's ray before `closestHit` (gated by `sc.hasGrin`) — and since 0.198.0 that march
  also integrates the participating media along the curve, and the mode-`M`/`S` gathers march
  their camera rays too (see the `grin.h` bullet); only mode-`D` BDPT and
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
  only; it briefly carried its own spot/sun reject in 0.126.0, dropped again in 0.127.0 once the
  kernels learned delta lights) mirroring `vcm.h`'s `vcmPass`: each pass (1) `kVcmLight` traces one light subpath per pixel,
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
  BDPT still can't render — **fluorescence**, **layered stacks**, **env/collimated lights** —
  are *not* GPU gaps: `main.cpp`'s mode-D guard (`bdptUnsupportedFeature`) refuses those scenes (or
  demotes D→B with `-on-unsupported fallback`) on both backends before any BDPT dispatch, so they
  never reach the device path. **Spot/sun lights** were a genuine GPU gap from 0.124.0 to
  0.125.0; since 0.126.0 (mode `D`) and 0.127.0 (mode `U`) the device kernels do them too (see
  "Delta lights in BDPT"/"in VCM" above), so neither `cudaBdptSupported` nor `cudaVcmSupported`
  rejects such a scene any more. GRIN media (curved
  paths) likewise keep an in-scope mode-D scene
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
  `kWSppCap` to resolve the wavelength collapse that case still causes. **(0.199.2)** That
  predicate and that cap are now the *shared* `whittedDeHeroes(scene)` / `kWhittedDeHeroSpp`
  in `main.cpp`, and the cap moved 16 → **64**. Two copies of the rule is how the viewer and
  the batch path drifted apart: mode W's wavelength lattice is a function of the ABSOLUTE
  SAMPLE INDEX alone, shared by every pixel — which is what makes the mode noise-free on a
  bundle scene, and what makes a de-hero'd one come out **uniformly mistinted** instead of
  grainy, since an N-spp frame is the whole image rendered at N *shared* wavelengths. It is a
  global error, so it does not average down per pixel. Measured on `gallery_rain` (rain
  volume), frame-mean B/G against the converged 512-spp answer 0.705: `1 → 0.000` (literally
  no blue in the picture), `2 → 0.085`, `4 → 0.483`, `8 → 1.243` (magenta), `16 → 0.691`,
  `32 → 0.787`, `64 → 0.694`. 16 was still ~18% off in R/G; 64 lands within ~2%, and costs
  the viewer nothing in responsiveness because any camera movement abandons the unfinished
  refinement outright. The **batch** path has no such escape — it obeys `-spp` literally —
  so `warnWhittedDeHeroSpp` prints a `[mode W] WARNING` naming the mechanism whenever a
  de-heroing scene is rendered below the floor. This is also the explanation for a live
  `-window` on such a render appearing to *fluctuate in hue*: each repaint is a different,
  still-incomplete set of shared wavelengths. The *dispersive*
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
- **Live-window refresh cadence** — the window repaints on its **own** timer
  (`-window-interval`, default 0.2 s; `liveWindowDue()` / `liveWindowUpdate()` in
  `main.cpp`), *not* on `-interval`. Every progressive driver (`runSppProgressive` for
  R/W/D, `runCompositeProgressive` for P, the forward A/B/C loop, and the shared
  multi-camera forward loop) computes two independent flags per report — `wantSave` on
  `-interval` for the crash-safe PNG + `.ftbuf` + status line, and `wantWin` on the window
  timer — and does only the work each one needs. They were one flag until 0.149.0, and the
  consequence was that **any render finishing inside one `-interval` never displayed a live
  frame at all**: the drivers only touched the window inside their
  `done || sinceSave >= intervalSec` block, so a 5 s `-mode W` frame under `-interval 8`
  painted exactly once, on `done`, as the process was exiting — the image appeared for a
  split second and vanished.
- **The window is created UP FRONT, not on the first repaint** (`liveWindowPlaceholder()`
  in `main.cpp`). It used to be constructed lazily inside `liveWindowUpdate`, which meant
  that in the ray-traced modes it did not exist at all until the first repaint — so the
  "painted once, at the end" bug above was really "*created* once, at the end", and what
  the user actually saw was a window flashing up as the process exited. Only the
  raster/`-explore` path popped up an early placeholder. Since 0.149.1 both paths call
  `liveWindowPlaceholder(w, h, stage)`: the raster block before tessellating, and `run()`'s
  render dispatch (plus the top of `runRender` as a per-frame re-title) before the CUDA
  probe / scene bake / device upload / first chunk. It fills a near-black frame and names
  the stage in the title bar (`preparing…`, `tessellating (3/8)`, `mode W — starting…`), so
  a long silent setup phase is legible instead of looking hung. It deliberately does **not**
  stamp `g_lastWindowPaint`, so the first real frame lands the instant it exists rather than
  waiting out a window interval. Because the placeholder now creates the window, "have we
  painted yet" is tracked by its own flag `g_windowPainted` rather than by
  `g_liveWin != nullptr` — the cold-cost exclusion below keys off the first *image*, and
  conflating the two would have re-introduced the 4 s adaptive floor. Measured on
  `gallery_rain` at 480² `-mode W`: window on screen at 2.2 s (as soon as the scene has
  loaded) and live from spp 1, versus not existing until the render was over. The one
  remaining silent stretch is the scene load itself, which is before the frame size is
  known — opening a guessed-size window there would leave it the wrong shape for the whole
  render, so it isn't done.
- **The title bar names the compute backend** (`liveTitle()` / `setLiveTitle()` next to
  `g_windowMode` in `main.cpp`). Every window title is assembled in one place —
  `scene → output  —  <mode>  —  <frame>  —  <progress>  —  <backend>` — instead of each
  call site concatenating its own string, so no fixed field can be dropped by one of them.
  **Since 0.307.0 the assembler owns all four fixed fields.** Before, only
  `liveWindowUpdate()` prepended the mode; the tessellation, exposure-metering, raster and
  explore stages titled the window directly and so showed *no mode at all*, `mode W` was
  hand-written into two of its own titles, `modeLabel()` had no `W` entry, the shared
  multi-camera paths (`runSharedGroup` / `runSharedPhotonMap`) never stamped a device
  because only `runRender` did, and only the shared mode-M gather ever named the frame.
  Now `g_windowMode`, `g_windowFrame` and `g_windowBackend` are stamped by whichever loop
  knows them (the flight loop names `'fly0555' (556/1147)`, the shared gather `frame
  7/601`, a shared forward group `12 cameras (shared flight)`), and a caller can only
  supply the progress text. **The progress text is one format for every driver** too:
  each fills a `LiveStatus` (tag, elapsed, budget, photons, spp, batches, noise, target,
  reason) and `liveStatusText()` prints the fields in a fixed order with a derived
  throughput — `[time] 0:58 / 1:00  ·  12.3M / 40.0M photons  ·  8 / 32 spp  ·  245.1k/s  ·
  ~8.84% noise (target 2%)  ·  (stopping)` — leaving out only what a driver has no value
  for (a backward mode has no photons) and never anything it does know. The console lines
  keep their older per-driver wording; the window is the thing that was inconsistent.
  `backendLabel(gpu, nThreads)` renders `GPU (NVIDIA GeForce RTX 4090)` (the real
  `cudaDeviceName()`, so a multi-GPU box says *which*) or `CPU (12 threads)`, and it is
  stamped into `g_windowBackend` **where the device is actually resolved**, i.e. after
  the VRAM probe in `runRender` — not from the `-device` flag, which is a request that
  a failed probe or an unsupported feature can silently override. The raster/`-explore`
  block is the one place a single window legitimately alternates backends within a
  session (the raster preview and a `mode W` refinement run on the CPU while a `PV_PT`
  path-trace runs on the GPU), so it precomputes `rasterBackend`/`cpuBackend`/`gpuBackend`
  once and re-stamps per branch rather than reporting whichever device the frame started
  on.
- **The title bar also says when the render has *finished*** (`markLiveWindowDone()` /
  `noteFinishReason()`, same block in `main.cpp`). Before this the only end-of-render
  signal in the window was that the progress text stopped changing — indistinguishable
  from a render still inside a long chunk between repaints, and worst under
  `-keepwindow`, whose whole purpose is to outlive the render. Now `liveTitle()` prefixes
  `✔ DONE — <why>` once the render is over, e.g. `✔ DONE — noise target met  —  ftrace —
  …`. A **prefix**, because both the taskbar button and a narrow title bar truncate on
  the right. The reason is not decided by the window code: each progressive driver
  (`runSppProgressive`, `runCompositeProgressive`, the A/B/C batch loop, the shared
  multi-camera loop, and the non-chunked fixed-`-n` path) already tracks *which* budget
  tripped — it now keeps a `metTime` alongside its existing `metNoise` and calls
  `noteFinishReason("noise target met" / "time budget reached" / "photon target reached"
  / "sample target reached" / "stopped early")` on the way out. `main()` reads that
  string once, after `run()` returns, rather than letting the drivers set the title
  themselves: a multi-camera flight runs one driver **per frame**, so only the last one's
  finish is the *render's* finish. The catch block records `"stopped by an error"`, and
  `setLiveTitle()` caches the last mode/progress text (`g_windowRest`) so adding the
  prefix keeps the final `…40133 spp, ~0.50% noise` line rather than blanking it.
- **Mode `M` used to be the one mode with no progress in the title at all** (fixed 0.199.4).
  Two independent gaps, and a showcase `gallery_rain -beams` flight hit both. (a) The shared
  multi-camera GPU branch called `liveWindowUpdate(f, sppDone, exp, absolute)` with the
  optional `status` argument **omitted**, so its `SppProgress` repainted the image but left
  the caption frozen — and the branch never called `noteFinishReason()`, so the `✔ DONE`
  prefix never appeared either. Both are now supplied: `pmGatherStatus()` (next to the
  checkpoint helpers in `main.cpp`) formats the same shape every other mode reports —
  `frame k/N — [gather] 107 / 160 spp (67%), 80.0M photons, 1017.1s, ~9.67% noise` — where
  the noise term is the mode-`M` estimator `100/sqrt(mean per-pixel hits over LIT pixels)`,
  the same formula the forward driver uses, with unlit pixels excluded so a black border
  cannot read as perfect convergence. (b) More importantly, **a mode-`M` render spends most
  of its wall clock before a film exists at all** — the photon deposit and the map/beam
  builds — and `SppProgress` cannot report there because it hands back a `Film`. That phase
  is now covered by **`StageProgress`** (`render_progress.h`): a `report(text, done, total)`
  with no return value, since a stage is stopped through the ordinary `ft::stopRequested()`
  flag rather than by its reporter. `makeStageProgress()` renders it as
  `tracing photons — 1.0M / 80.0M (1%), 7s, 150.5k/s, ~525s left` on a 0.25 s window cadence
  and a 30 s stdout cadence, and `total == 0` degrades to `building photon map… 12s` for a
  phase with no meaningful measure.
  - On the **CPU**, `tracePhotonPass` is a join-and-wait, so the count has to be sampled from
    outside it: the workers publish into a relaxed `std::atomic<long long>` on the *same*
    4096-photon cadence as their existing stop poll (one `fetch_add` per 4096 path traces is
    unmeasurable), and a monitor thread — started only when a reporter was actually passed —
    polls it while the main thread blocks on the joins.
  - On the **GPU** there is no such seam, because the deposit is a single `launchForward`. So
    when a reporter is present the deposit is **split into chunks**, sized adaptively to ~1 s
    of work (1 M-photon probe, never shrinking below it, never growing more than 4× a step).
    This is safe because `kTrace`'s `N` is only a grid-stride bound, `genPhoton` gives each
    photon an absolute beta, and the estimate normalises by `pm.nEmitted` — so K launches of
    N/K emit exactly the energy one launch of N does, and `launchForward`'s
    `kseed = C + seedBase*C` makes chunk 0 reproduce the single-shot stream. Two traps this
    walked into and back out of: the split is armed **unconditionally**, *not* gated on
    `g_showWindow`, because gating it would make the same command produce a different image
    depending on whether anyone was watching; and the loop now tracks `depEmitted` and assigns
    `pm.nEmitted = depEmitted` **after** the deposit, because a `-stop` between chunks would
    otherwise leave `nEmitted = N` while having emitted fewer and darken the whole map by the
    untraced fraction. The chunk seam is also the first place a `-stop` can land during a GPU
    deposit — before this there was none.
  - The monitor thread is why `setLiveTitle()` now takes a **mutex**: it re-titles from a
    non-main thread while the main thread is inside the joins, and `g_windowRest` is a
    `std::string` that would otherwise be written from two threads at once. `SetWindowTextW`
    itself is already cross-thread safe (it marshals a `WM_SETTEXT`), but *creating* the
    window is not, so `liveWindowPlaceholder` compares against a recorded `g_mainThreadId` and
    off-thread callers only re-title.
- Repaint granularity is bounded below by the renderer's chunk size, not by this timer:
  `gpuSppChunks` / `cpuSppChunks` retarget ~0.15 s per chunk with a 1 spp floor, so a 480²
  `-mode W -spp 8` frame gets one repaint per spp and the first complete image lands after
  ~0.6 s. The floor is adaptive — `max(-window-interval, 12 × last measured repaint cost)`
  — so a 4K film, or the shared multi-camera path where a repaint also forces a full
  device→host film download (`liveWindowNotePaintCost` charges that to the same budget),
  backs itself off instead of spending the render on painting. The **first** paint is
  excluded from the estimate: it runs cold and includes one-time window creation, and
  feeding its 342 ms in set a 4 s floor that made the second repaint also the last.
  Measured on a 480² `-mode W -spp 8` frame: `-window` at all costs +0.28 s (pre-existing
  D3D11 / swap-chain init), the seven extra repaints +0.20 s = **+3.9 %**.
  `FTRACE_WINDOW_DEBUG=1` logs each repaint and its cost.
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
  That preview kernel (`kIsoPreview`) shades two-sided, and since 0.183.1 it picks the side
  from the **geometric** normal, not the shading normal: it first undoes any flip
  `intersectTri` applied by testing the *interpolated* normal (`dot(N, Ng) < 0`), then
  applies one genuine backface test (`dot(Ng, V) < 0`) to both. Testing `dot(N, V)`
  directly — what it used to do — inverts `N·L` in the ~1-px band at every silhouette where
  a smooth normal grazes through zero on a still-front-facing triangle, stippling the
  outline with light and dark specks. `raster.h` decides the same flip once per triangle at
  projection time and the path tracer re-derives it via `ngo`, for the same reason; see
  `known-issues.md` for the underlying `intersectTri` convention that all three work around.
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
  baked with the upload for the same reason. Shading is two-sided lambert
  `0.30 + 0.70*|n.z|` from a **real interpolated per-vertex normal**: the vertex format is
  position + uv + `NORMAL`, and `buildMeshPaneVerts` derives the normals on upload with the
  same crease-limited angle-weighted algorithm ftrace's loader uses (`src/mesh.h`) —
  position weld at `1e-6 x diag`, Thürmer–Wüthrich corner-angle weights, neighbours merged
  only inside the crease angle, corners split back apart where their normals disagree.
  The angle comes from the sidecar: loom ships `mesh.smooth` as the *resolved crease
  angle in degrees* (`loom.scene.smooth_crease_deg`, the same number its `mesh { smooth
  <deg> }` carries), so the preview creases exactly where the render will, and a strand
  tubed by the pane itself asks for ftrace's default 40°. **Authored `normals` in the
  sidecar win outright**, matching the way OBJ `vn` beats `smooth` in the loader — and as
  of 2026-08-12 that is the common case, because a `SweptMesh` with the default
  `smooth=True` ships its surface's *analytic* per-vertex normals (`loom.sweep.ring_normals`
  differentiates the ring lattice, which is literally a parameterisation of the swept
  surface). That matters because a crease angle is fundamentally a guess: on a smooth but
  coarsely-sampled profile — `r(a) = 1 + 0.34·cos(3a)` at 18 samples — dihedrals reach 119°
  and 18% of edges refuse to merge at 40°, so the surface renders faceted, correctly by the
  rule and wrongly by the surface. The generator knows; the triangles cannot.
  This replaced a per-pixel face normal rebuilt from
  `cross(ddx(vp), ddy(vp))`, which could only show facets *and* — because `ddx`/`ddy` are
  evaluated over 2×2 pixel quads — emitted a garbage normal wherever a quad straddled a
  triangle boundary, painting a one-pixel band of wrong shading along every edge (the
  "jagged seams" on loom's sweeps). The target is **4× MSAA** (`pickSamples` falls back to
  2×/1× if the device refuses) resolved into the single-sample texture ImGui samples, and
  the wireframe is a real second depth-tested `D3D11_FILL_WIREFRAME` pass. The **curve and field panes
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
  value — latest-wins, and the UI never blocks. **Each bake is carried entirely over the
  pipe the two processes already share — nothing per-frame touches the filesystem.** The
  sidecar rides back inside the ack as a JSON member (`stealMember` *moves* the subtree
  out, so the parse the bridge already did on its worker thread is the only one), and the
  meshes ride as raw bytes: the emit request carries `"assets":"inline"`, which makes
  loom's `EmitCtx.mesh_sink` divert the encoded mesh into a dict instead of writing it,
  and `serve_viewer` frames those payloads **after** the ack line, back-to-back with no
  delimiters, every length declared up front in the ack's `blobs:[{name,bytes}]` manifest.
  A reader must therefore drain them even if it wants none, or the stream desyncs — hence
  `LoomLink::readExact`. They arrive as an `assetbytes::Overlay` that `ftsl::loadSource`
  consults before ever calling `open()`. This removed the *whole* per-frame filesystem
  round-trip, whose dominant cost was not the I/O but Windows Defender scanning files
  ftrace's own child had written microseconds earlier: 130.3 → **102.6 ms/frame**
  (7.70 → 9.75 fps, n=104), with the `sidecar` term collapsing 22 → 2 ms and `ftsl` 31 →
  21. The scratch dir is now only a *naming scheme* for the assets keys, never created;
  `stop()` still sweeps it, and startup still reclaims `ftrace_viewer_<pid>` dirs whose
  pid is no longer alive (`OpenProcess` failing with `ERROR_INVALID_PARAMETER`), since a
  crashed or killed viewer can't clean up after itself and only the next run ever can —
  both retained because older builds did litter. Results are
  adopted on whatever frame they land, preserving the user's orbit, zoom, active tab and
  DAG layout.
  **Playback (§F8).** Two schemes, and which one runs is decided per frame by whether the
  cache holds the frame the clock wants.
  *(a) Bake-paced.* The clock advances **only at the `bridge.take(r)` site**, never on a
  timer. That is not a simplification, it is the only correct pacing given a latest-wins
  one-slot queue: a timer-driven play loop would post frames faster than they bake and
  most would be superseded before running, showing a stutter of whichever ones won the
  slot rather than the animation. Starting play must post **once** to prime the loop, or
  nothing is in flight, nothing lands, and the clock sits still.
  *(b) Prebaked (`PlayCache`, `-prebake`).* Walks the clock once — serially, one
  outstanding job at a time, for the same latest-wins reason — and keeps each frame's
  **adopted** state (`Sidecar`, curves/strips/fields/meshes, `DagGraph`, `ftsl::Loaded`).
  Adopted, not the payload: `Sidecar::adopt` and `ftsl::loadSource` both *consume* their
  input, so replaying payloads would need a deep copy of a ~900 KB `minijson` tree per
  frame *and* would still pay sidecar + ftsl adoption every time round the loop — 63 % of
  the frame. Frames are exchanged by `std::swap` under a single invariant: **the live
  locals hold frame `liveIdx`, and slot `liveIdx` is empty.** Showing frame *k* is then
  park + unpark, two O(1) swaps with no copies, and — the reason this shape was chosen
  over pointing the pane locals at the cache — it required no refactor of the function
  those locals live in. The vacated slot doubles as the buffer the next park swaps into,
  so a playback loop allocates nothing. Results are claimed by a **params fingerprint**
  (`playCacheKey`, carried on `LoomJob`/`LoomResult`), not by frame number alone, so a
  bake still in flight when a control moved cannot be filed into the new cache under an
  index it happens to share. A cache that hits its **cap** covers a prefix, and the clock
  running off the end drops back to (a) — without that fallback play deadlocks there,
  since nothing posts, so nothing lands, so the clock never moves. Measured on
  `scatter_modulated_sweep` (96 frames, 1863 MB at 19.4 MB/frame since the profile went
  30×200): a requested 24 fps is delivered at `cache 0.01 + raymarch 20` per frame, against
  3.4 fps bake-paced. Note the cache is ~3× what it was at 18×120, so the default 1024 MB
  cap no longer covers this clock — which is why, since 0.183.2, the walk **projects** the
  total after 4 frames (enough for a stable MB/frame, early enough to still be a warning)
  and prints either "fits the N MB cap" or the shortfall plus the `-prebake-cap` value that
  would cache everything. A prefix cache degrades silently otherwise: the loop runs at the
  target rate until it walks off the end of the cache and then stutters, which reads as a
  performance bug rather than as a budget that was set too low.
  Two measurement rules the feature had to fix to be believable: a cached frame **clears**
  `bake`/`sidecar`/`ftsl` (otherwise the breakdown prints work that did not happen, its
  parts summing to several times the period beside them), and the fps EMA smooths the
  **period** and inverts at the end — averaging *rates* over a pacer that alternates 2 and
  3 vblanks reports 25 fps for a true 24. The pacer likewise advances its deadline by
  exactly one period instead of resetting to `now`, because discarding the overshoot
  quantises the achievable rate to the display refresh (a 24 fps request delivered a
  rock-steady 20 at 60 Hz), while banking at most one period of debt so a hitch is
  absorbed rather than repaid as a burst.
  **Third-party note:** `src/third_party/imnodes/imnodes.cpp` carries
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

## Analytic specular camera connections (forward mode `B`, 0.160.0 / 0.188.0)

The forward tracer draws a surface by *connecting* each photon vertex to the camera. A
**specular** vertex has no such connection — a delta lobe scatters into exactly one
direction, so the density of "aims at the pinhole" is zero — and `tracePhoton` therefore
skips the camera connect at every `isSpecularType()` material. That is the SDS gap: every
directly-seen mirror/glass surface rendered **pure black** in `B` while `R` drew it.

Where the specular geometry is simple, the connection is *solved* instead of sampled. Both
CPU (`render.h`) and GPU (`render_cuda.cu`) implement the same three connectors, and the
device versions do all the precision-critical math in `double` regardless of the render
`Real`, so the two backends agree.

| Connector (CPU / GPU) | Surface | Reflection point | Geometry factor `G` |
|---|---|---|---|
| `connectSpecularSphere`(`Inside`) / `dConnectSpecularSphere` | smooth `dielectric` sphere, eye outside or inside | 1-D root scan of a planar miss function + 40-step bisection (≤4 roots = multiple refracted images) | ray differentials, `eps²/\|dA×dB\|` |
| `connectSpecularSphereMirror` / `dConnectSpecularSphereMirror` | `mirror`/`halfmirror`/roughness-0 `glossy` **sphere** | same scan, one reflection instead of two refractions | same, re-expressed as an equivalent unfolded distance `D = 1/√G` so it shares one splat with the plane |
| `connectSpecularPlane` / `dConnectSpecularPlane` | coplanar **mirror triangles** (a flat panel) | **unfolding** — no root solve at all | exactly `1/D²`, `D = \|p − eye′\|` |

**Unfolding** is what makes the flat case closed-form. Mirroring the eye across the plane,
`eye′ = eye − 2·(dot(n,eye) − d)·n`, turns the bent chain `p → R → eye` into the straight
segment `p → eye′`; `R` is where that segment crosses the plane, and the specular Jacobian
is `dΩ_eye/dA_perp = 1/D²` with `D = |p − R| + |R − eye|` — i.e. `connect()`'s own
inverse-square law measured along the folded-out path. So the mirror estimator *is*
`connect()` with a longer, bent distance and one reflectance factor.

**Per-plane, not per-triangle.** `Scene::buildMirrorPlanes` (`scene.h`, run at the tail of
`Scene::build` because it needs finalized `Tri::gn`) groups mirror-material triangles by a
quantised `(n, d)` key — normal canonically oriented, since a mirror is two-sided — into
`Scene::mirrorPlanes`, each carrying the plane, the member AABB and the total area.
That makes the per-photon-vertex cost **O(#mirror surfaces)**, not O(#mirror triangles),
and it is capped at `kMaxMirrorPlanes = 64` (largest-area planes win). The list uploads
verbatim to `DScene::mirrorPlanes` as `DMirrorPlane`. The sphere connectors get the same
treatment (0.190.4): `buildMirrorPlanes` also fills `Scene::dielSphereIdx` /
`mirrorSphereIdx` — the ascending indices of the dielectric / planar-mirror spheres — and
`camSpecularSplatAllVtxN` (both backends; the lists upload to `DScene::dielSph`/`mirrorSph`)
iterates those instead of testing every sphere's material per vertex per λ, so a scene of
many plain spheres pays nothing. Ascending order keeps the film accumulation order, and
therefore the images, bit-identical. Within each connector the tests are ordered
cheap-reject-first (0.190.4): pure projections and root solves before any BVH query, then
the any-hit `occluded()` before the closest-hit `mirrorSeenAt` — legal because every
reordered step is pure and RNG-free, so a vertex that fails either way computes the same
splat or none.

**One traversal does three jobs.** A plane does not know where its panel *ends*, so
`mirrorSeenAt` / `dMirrorSeenAt` casts one `closestHit(eye → R)` and requires
`|hit.t − dE| ≤ 1e-4·(1+dE)` **and** `isPlanarMirrorMat(mats[hit.matId])`. That
simultaneously proves (a) the authored panel really contains `R`, (b) the eye-side leg is
unoccluded, and (c) hands back a real `Hit`, so `reflectSlot` reads the mirror's
texture/pattern reflectance at the exact point. The light-side leg is a separate
`occluded()`, and `splatMirrorLegs` / `dSplatMirrorLegs` applies fog transmittance on both
legs.

**Which materials qualify** (`isPlanarMirrorMat`, `scene.h`; `dIsPlanarMirrorMat`):
`mirror`; `halfmirror` (its reflect *probability* is its specular reflectance in
expectation); `glossy` only when `roughness <= 0 && roughnessTex < 0 && roughnessPat < 0`.

**Hero bundles.** A mirror is achromatic, so ONE geometry solve serves the whole live-λ
bundle and only the reflectance varies per λ — hence `camSpecularSplatAllVtxN`
(`…AllVtx` is now a 1-λ wrapper over it) and the split of `SpecVtx::term()` into a
λ-independent `shape()` plus the per-λ weight. A dielectric sphere is dispersive and
**cannot** share geometry: each λ traces its own refracted image, which is exactly what
makes a glass-orb caustic image chromatic. Its loop nesting (λ → sphere → camera) was
preserved verbatim through the refactor, and `term()` was left byte-identical rather than
being expressed through `shape()` (`w*(cos/π) ≠ (w/π)*cos` in floating point), so every
pre-existing image is bit-for-bit unchanged (verified against a baseline binary built from
the previous commit, CPU and GPU).

**Known limits** — all deliberate, all documented in `REFERENCE.md` and tracked in
`known-issues.md`: ONE specular vertex per connection (a mirror inside a mirror is still
black); **pinhole only** (`lensMode` / `forwardCatch` return early, so `A`/`C` are
unchanged); flat mirrors must be authored as world-space triangles (an instanced/BLAS
mirror is not collected, and beyond `kMaxMirrorPlanes` the surplus planes are dropped);
only a `halfmirror`'s *reflected* leg is built, so what lies behind a beamsplitter is still
a delta transmission and stays black; rough specular is not a delta and needs a real
estimator. Use `P`/`D`/`R`/`M`/`S`/`U` for those.

**Test scenes.** `scenes/_mirror_fwd.ftsl` (flat panel), `scenes/_mirror_sphere_fwd.ftsl`
(mirror ball) and `scenes/_mirror_mats_fwd.ftsl` (one panel each of `mirror`, `halfmirror`
and roughness-0 `glossy`, black-backed so the comparison is reflection-only). All three put
the emitters *outside* the frustum, so the lit result exists only by reflection and a
regressed connector shows up as black rather than as a subtle level shift. Meter them with
`-hdr` + `scraps/cmp_pfm.py`, comparing each mirror's pixel box against a mirror-free patch
of the same frame — mode `B` and mode `R` carry a systematic ~0.6–1 % offset everywhere, so
the mirror-free patch, not 1.0, is the number the mirror box has to match.

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

## N-D field slicing (`-nd` on an isosurface, 0.230.0)

The mesh path needs `emboss`/`extrude` because a mesh is a 2-manifold in a 3-flat. A FIELD
is defined everywhere in N-space, so it needs neither: `-nd` tilts the 3-D SLICE the field
is evaluated on, and the cross-section genuinely changes. `PatSlice` (in pattern.h, so both
backends see one definition) carries `A` — the first three columns of the rotation, row k
being dimension k's direction in the slice — and an offset; `patApplySlice` rewrites a
PatCtx's x/y/z onto the slice and fills `d4..d12`, nine new VM variables.

**It rides in `PatTables`,** which already reaches every field evaluation on both backends,
rather than becoming a global or a new parameter on the intersectors. `dims == 0` is the
ordinary 3-D case and every evaluator skips the transform, so a scene without `-nd` is
untouched.

**The one condition, stated at load.** A three-input field sliced in N-D only ever sees an
affine remap of (x,y,z) — the same result as the mesh case and for the same reason — so
`anyFieldReadsExtraDims` scans the compiled programs and says plainly when nothing reads
`d4`. Without that, the honest answer ("your field has no fourth dimension to rotate into")
is indistinguishable from a broken feature.

**Three device evaluators had to learn the opcodes, and forgetting one is silent.** There
is the host `patternEval`, the device template `dPatternEval` (fp64) — and
`dPatternEvalF`, an FP32 twin in render_cuda.cu that is what the sphere-trace march
actually runs. Patching only the first two left the GPU tracer rendering the gyroid as
*nothing at all*: an unknown opcode makes the VM bail, the field reads 0 everywhere, and an
invisible surface looks like an empty scene rather than a wrong one. The bisect that found
it is worth repeating — the same scene rendered correctly by the CPU tracer and by both
rasterizers, and `scenes/function.ftsl` (no `d4`) rendered identically on both devices,
which localised it to "the d4 opcode on the device" rather than to the slice.

## N-D warp (`ndwarp.h`, `-nd`, 0.220.0)

**Where it sits.** The warp mutates `Scene::tris` immediately after the scene is built and
before anything reads geometry, rather than transforming preview triangles. That choice is
what makes one implementation serve three consumers: `raster::tessellate` sees the warped
mesh because it reads the scene; every render mode path-traces it for the same reason; and
`-nd-export` writes the very triangles that were on screen. Transforming `PreviewGeom`
instead would have been cheaper and would have covered only the preview.

**What is warped.** Every `MeshGroup` with `blasId < 0`, minus emissive ones. Native
primitives are deliberately excluded — `-nd` is for looking at a *model*, and a scene's
floor folding up with it is noise — and instanced BLAS assets are not reachable through
`Scene::tris` at all. Emissive meshes are skipped because `Scene::addMeshLight` copies
triangle data into the emitter at load time, so warping the surface without rebuilding the
light leaves the two disagreeing about where the light is. Sensors (`sensorId >= 0`) never
move. The unwarped triangles keep their relative order at the head of the rebuilt array, so
an untouched group's range only SHIFTS and can be remapped by counting retained triangles.

**Interactive cost model.** A slider drag re-warps from the pristine captured copy (angles
are absolute — composing deltas would drift), re-tessellates and re-uploads the GPU
rasterizer, but deliberately does NOT rebuild the BVH: the rasterizer has no use for one and
a 2 M-triangle extrusion would cost seconds per tick. The tree is marked stale and rebuilt
lazily, once, on the first traced ('T') frame that needs it. Emboss signals are cached per
source for the same reason — curvature costs a full adjacency walk that a drag would
otherwise repeat sixty times a second.

**The edge-on report.** An extra dimension reaches the image only through column *k* of the
rotation matrix's first three rows. Until a rotated plane contains axis *k* that column is
zero: an extrusion's side walls project to exactly zero area and are culled by the
degenerate-triangle guard, so the count falls straight back to the two lids and it looks
like the fill did nothing. It is the correct picture of a solid seen edge-on, so
`Stats::edgeOn` records it and both the CLI and the panel name the axis and suggest a plane
rather than leaving the user to wonder.

**The fill cells are the point of the panel.** Shipping the plane sliders without them
(0.220.0) made the viewer able to do nothing but rotate and squash, because a lift whose
extra coordinates are all zero is exactly one 3x3 matrix — the fills were reachable only
from the command line, so the interactive tool could not reach the interesting half of its
own feature. `ndwarp::fillChoices()` is now the single list both the combo and the CLI
resolve against, with `applyFillChoice` / `fillChoiceOf` the only conversions, so the two
front ends cannot drift. Two assists live in the render loop rather than the window,
because both need the warp's state: a fill picked at amount 0 gets a visible default, and a
fill switched on into a fully edge-on axis turns the `z`-axis plane to 30 degrees so the
change is visible instead of silently culled.

**The two shading toggles.** `Color` and `See-through` sit on the panel's first row, so
they are available in any `-explore` session rather than only under `-nd`. They differ in
where they act, and that decides what a toggle costs. See-through is read per frame by
`rasterOne` and handed straight to both `raster::renderFrame` and
`raster_cuda::renderFrame`, so flipping it costs exactly one re-render. Colour is
`raster::stripColor`, a pass over the BAKED `PreviewGeom` that replaces every albedo with
neutral clay and drops everything that could re-introduce a colour (skin, triplanar,
`reflect`/`emit` pattern drives, normal map, and both children of a per-hit mix plus its
mask). Doing it to the geometry rather than as a flag inside the shade pass is what makes
the CPU and GPU backends agree for free — neither needs to know the mode exists — and the
price is that toggling colour re-tessellates and re-uploads, the same work a scene swap
does. That trade is right for a button pressed occasionally and wrong for a slider, which
is why the N-D angles went the other way.

**Panel geometry.** The slider bank's height is a function of the window WIDTH (it wraps),
so `panelH` is re-derived from `panelBaseH` on every layout instead of being accumulated as
a one-time delta — otherwise narrowing the window would lay slider rows out below the strip
where they cannot be seen. Every builder that grows the window must also call `layoutView`,
not just `layoutPanel`: growing normally fires `WM_SIZE`, which lays out both, but the grow
is CLAMPED once the window would exceed the work area, and a clamped `SetWindowPos` changes
nothing and so raises no `WM_SIZE`. The D3D child then keeps a rect taller than the image
area and the swap chain presents the frame offset and cropped — the model slides off toward
the bottom right — until some unrelated resize happens to correct it. The N-D bank is what
exposed this (it is the tallest strip, so it is the one that gets clamped), but the bug was
latent in `buildPanel` and `buildBindRow` too and is fixed in all three. Changing the dimension box destroys and rebuilds every trackbar;
angles are carried across by PLANE IDENTITY (`planeIndex(oldN, i, j)`), so `xw` stays `xw`
rather than being silently re-indexed as the plane order lengthens.

## Per-vertex colour: store the COLOUR, resolve the spectrum per hit (0.226.0)

Four formats carry a per-vertex colour — PLY's `red/green/blue`, OBJ's extended
`v x y z r g b`, glTF's `COLOR_0`, an FBX colour layer — and every loader was dropping it.
`Tri` gained `vcol`, an INDEX into `Scene::vertColors` rather than nine inline floats,
because Tri already sits in the BVH's hot leaf data and almost no mesh has vertex colours.

**The design question is what to store, and the obvious answer is wrong.** ftrace is
spectral: the tracer reads `reflect(lambda)`, so an RGB vertex colour must become a
spectral reflectance, and `rgbToReflectanceJH` is a ~40-iteration Gauss-Newton fit — the
file's own comment records that running it per texel cost 45 s of a 47 s scene load, so
per hit is unthinkable. The natural fix is to fit once per vertex and interpolate the
COEFFICIENTS. That is wrong twice over: it interpolates the wrong quantity (the colour
varies linearly across a face; the sigmoid coefficients that reproduce it do not, and they
move fastest near black), and it pays a fit per vertex at load.

**What it does instead** is store the linear RGB, interpolate the COLOUR at the hit, and
resolve the spectrum through `upsample::coeffLut()` + `stochJhCoeff`. That table exists
already, built for the identical predicament: stochastic tiling *invents* a colour at every
hit by blending three crops and then owes the renderer a spectrum for it. Reusing it means
no load-time fitting at all, the exact quantity gets interpolated, and — since
`stochJhCoeff` is `STOCH_HD` — the device port needs no new maths.

**The two consumers want different things from the same numbers, which is why RGB is what
is stored.** The rasterizer IS an RGB pipeline: it skips the lift entirely and multiplies
the interpolated colour into the albedo. The tracer needs the spectral lift. Storing fitted
coefficients would have served neither cleanly.

**Where each backend interpolates, and why they differ.** The host rasterizer's shade pass
is DEFERRED — it sees only the G-buffer — so the interpolated colour has to be written into
a G-buffer channel. That channel is allocated only when the tessellation actually contains a
vertex-coloured triangle, so an ordinary scene pays neither the 24 B/pixel nor the per-pixel
write. The GPU rasterizer needed no buffer at all: `kShade` already recomputes this pixel's
barycentrics to recover position/normal/UV, so the colours ride along on `DPTri` for free.
The two came out bit-identical. The tracer interpolates in `intersectTri`, the only place
holding both the triangle and the barycentrics, and deposits the result on the `Hit`
alongside `u`/`v`; `diffuseReflectance` — the single resolver both the forward tracer and the
backward reference already share — is the one place that multiplies it in.

`Scene::vertColors` reaches the intersector as an explicit argument rather than a global:
there are only a handful of call sites and every one is a Scene method holding the vector.
A `Blas` has its own triangle array but needs no table of its own — `mesh_asset` loads
through the ordinary loaders into `Scene::tris` and only then copies the run out, so the
indices already point into the scene-wide table, and that table is append-only so slicing
the triangles cannot invalidate them. The TLAS traversal passes `vcolData()` down.

**The GPU tracer resolves it in a different place than the host, deliberately.** The host
does it inside `intersectTri`, which is free because Scene's intersectors are methods
already holding the table. The device intersector is a free function called from the BVH
leaf loops with no `DScene` in scope, and threading one through the hottest loop in the
renderer to serve a rare feature is the wrong trade — so it stores the index and two
barycentrics (the third is `1-b0-b1`; no more writes than the host's three floats) and
`dDiffuseRho` finishes the job, using the same `stochJhCoeff` on the same uploaded table.
Measured 10.2x over the CPU on a vertex-coloured mesh, converged images agreeing to
0.22/255 per channel.

**The instanced raster path is where this kind of change goes wrong.** `tessellate` bakes
BLAS triangles in a SECOND loop, and the first version of this work copied the colour only
in the flat-triangle loop — so instanced meshes rendered untinted in the preview while
every other backend had them. Both loops now call one `copyVcol` helper.

## glTF glass: the colour is in the extensions, not the core block (0.224.0)

`gltf.h` read only `pbrMetallicRoughness` and mapped metallic>=0.5 to Glossy and everything
else to Diffuse. For a transmissive asset that is not an approximation, it is a total loss:
a coloured gem is routinely authored as `baseColorFactor [1,1,1,1]` with the tint in
`KHR_materials_volume` and the transparency in `KHR_materials_transmission`, so a tray of
twelve differently coloured gems imported as twelve identical opaque white ones. Reported
as "Color only changes the whole tint a little, and See-through does nothing" — both
symptoms of the same omission, since with no dielectric in the scene the see-through pass
has nothing to act on.

Now imported: transmission (>=0.5 -> Dielectric, mirroring the existing 0.5 cut on
metallic, because these are single-BSDF materials), ior, volume (attenuationColor /
attenuationDistance -> `absorb`, Beer-Lambert both sides so the conversion is just
sigma = -ln(color)/distance), and dispersion (defined as 20/Abbe, so it converts to a
two-term Cauchy holding n_d and the F-to-C spread).

**Material::absorbRefDist exists because a coefficient is not a colour.** Beer-Lambert
absorption is per unit LENGTH; how dark a piece of glass looks needs a distance. The tracer
measures the real path through the solid, but the preview cannot — its clear pass is
order-independent precisely so it needs no depth sort, so it never pairs a front face with
its own back face. glTF states that distance, so recording it lets `clearTintOf` return a
real transmittance instead of only a hue: measured on the compote's gems, the tints land
within a few percent of sqrt(attenuationColor), which is the authored figure split over the
two crossings a closed solid presents. Assets that do not state a distance keep the hue-only
path.

**The clear surface's SOLID preview is tinted too.** Without see-through a clear material
previews from `reflect`, which for imported glass is routinely pure white — so the gems came
out as identical pale ghosts even once the glass colour was known. Multiplying the ghost by
the same tint is what makes the default view show twelve different gems.

## See-through glass takes its colour from the material (0.223.0)

Through 0.222.0 the clear-surface pass multiplied ONE scalar per crossing (`-glass-clarity`)
into a single float per pixel, so a red filter and a clear window previewed identically.
`PShade::clearTint` now carries a per-surface RGB transmittance and `clearT` is three floats
per pixel.

**Where the tint comes from is not one rule, because the materials do not state the same
thing.** `Filter` / `DiffuseTransmit` carry `transmit`, a dimensionless T(lambda) — that IS a
transmittance, so hue and magnitude are both real. `Dielectric` / `ThinFilm` carry `absorb`,
a Beer-Lambert coefficient per unit LENGTH, and turning that into a transmittance needs a
thickness the rasterizer does not have: the pass is order-independent precisely so it needs
no depth sort, which means it never pairs a front face with its own back face. So a
dielectric contributes HUE only (normalised to unit peak) and `-glass-clarity` keeps setting
the magnitude. Faking a thickness would have been the easy option and would have made the
preview lie about scale.

**Two subtleties that are easy to get wrong, and were.** The exponential is taken in
WAVELENGTH space and converted to RGB afterwards — `exp()` of an RGB-collapsed coefficient
is not the RGB of the exponential, and the difference is exactly the saturation of a
strongly absorbing glass. And the conversion is WHITE-BALANCED against a flat spectrum:
`spectrumToLinearRgb` of an equal-energy stimulus is ~(1.198, 0.950, 0.908), not (1,1,1), so
the first cut gave a perfectly colourless window a warm cast — visible as an 18 000-pixel
change on `cornell.ftsl`. Dividing by the flat response makes `absorb 0` come out exactly
white by construction (the same input reaches the same converter) rather than approximately.

**The host multiplies tau in float, not double.** `(float)clarity * (float)tint.x`, matching
`kClearAccum`'s float arithmetic — computing the product in double and rounding once lands a
ULP away from the device on some pixels. What remains between the backends after that is
+-1 LSB from `atomicMulF`'s nondeterministic accumulation order (float multiplication is not
associative), and, separately, a PRE-EXISTING coverage tie-break disagreement worth 120/255
on a wall seam — see RASTER-GPU-DIFF in `known-issues.md`, which this work found but did not
cause.

## Preview shading model (`raster.h` + `raster_cuda.cu`, 0.135.0; per-hit mix 0.136.0)

The preview has no light transport, so nothing that needs a *second* bounce — reflection,
refraction, shadows, GI, glossy lobes — can exist in it. But everything that determines a
surface's appearance **at a single point** can, and now does. Both backends implement the
identical set, because they share the bake (`raster::tessellate`) and mirror the shade.

| Material feature | Preview behaviour |
|---|---|
| `reflect texture:` skin | Sampled per pixel: per-vertex UV, world triplanar, or the primitive's own `uv` projection |
| Palette (indexed-spectral) map | Each palette entry pre-resolved to a linear-sRGB colour; nearest lookup |
| `reflect pattern:` / `reflect_map pattern:` | Scalar clamped to [0,1], multiplies the albedo |
| `emit pattern:` / `emit_map pattern:` | Scalar clamped to [0,1], multiplies the **emission** |
| `normal_map` | Perturbs the shading normal through the triangle's UV-derived TBN |
| `mix` / layered material | Resolved to its dominant child (`mixDominantChild`) |
| 2-child `mix` + `weight_map texture:`/`pattern:` | Resolved **per pixel**: the mask is sampled at the shaded point and hard-thresholded at ½, exactly as `mixResolveDominant`, and the winning child's whole payload is swapped in |
| `roughness_map`, `film_thickness` maps | Ignored **by design** — a preview has no glossy lobe for them to drive |
| Textured light | No such slot exists in the language; a light's look is its SPD × `emit_map pattern:` |

Design points worth keeping:

- **The emission mask is evaluated BEFORE the emissive early-out.** This is the whole
  reason the feature matters. `gallery_rain`'s ground is a 528 nm emitter masked by
  `emit_map pattern:grid_ground`; skipping the pattern made 28% of the frame one flat
  `rgb(0,255,89)` slab instead of a dark plane with thin glowing grid lines.
- **Marched implicits get their UVs from the primitive, not the mesh.** Marching cubes
  emits no per-vertex UVs, so a skinned isosurface has *no* UV source unless the
  primitive carries a `uv planar/spherical/cylindrical` projection — which `tessellate`
  now re-evaluates per marched vertex with the tracer's own `projectUV`. Missing this was
  the original bug: `gallery_rain`'s ten marble caps previewed untextured while the
  traced render showed marble.
  Azimuthal projections need **per-triangle seam repair**: the ray-hit path projects *at*
  the hit and never sees the 1.0→0.0 wrap, but the rasterizer *interpolates*, so a
  triangle straddling the seam would run `u` backwards across the entire texture. Any
  triangle whose `u` spread exceeds 0.5 has its low corners lifted by +1.
- **A new PRIMITIVE needs the same "did every consumer get it?" sweep a new material
  feature does** (0.150.0). The `applyMat` comment above exists because a per-material
  feature once got wired into three of four geometry paths; the `curve` primitive was the
  geometry-level version of the same miss. It shipped with the tracer, the BVH and a CUDA
  gate all correct, and `-raster` drawing `curve_basics` as `[raster] 12 triangles` — the
  box, none of the five strands, **silently**. Worse than the CUDA case, because CUDA at
  least has a gate to fall back through and the preview has none: nothing warns, the scene
  just looks empty. Now stage `(2b)` of `tessellate` meshes each `CurveSeg` as a round
  cone, sweeping rings through the same three pieces the intersector knows (back cap →
  tangent lateral band → front cap). Both tangent circles sit at polar angle `acos(a)`,
  `a = (r0−r1)/|ba|`, on their respective end spheres — that single fact is what lets one
  angular sweep cover all three pieces continuously, so the preview mesh is closed exactly
  like the surface it approximates, and the `|a| ≥ 1` degenerate (one ball swallows the
  other) falls out for free as a collapsed cap. `v` uses the same `onb(axis)` the analytic
  hit does, so a `u`/`v` pattern previews where it will land. ~80 tris/segment, which does
  not scale to a real groom — see `known-issues.md`.
- **A `weight_map` mix is a PER-PIXEL material swap, not a per-material choice** (0.136.0).
  `mixDominantChild` only compares the *constant* weights, so a weight-mapped mix (always
  50/50, hence always child 0) previewed as one flat winner while `-mode W` — which calls
  `mixResolveDominant` and evaluates the mask at the hit — showed the blend. That broke
  `raster.h`'s own stated invariant that "raster and mode W agree on what a mix looks
  like". The fix splits `PTri`'s material payload into a base `PShade` and adds a `PMix`
  side table (`PreviewGeom::mixes`, indexed by `PTri::mix`); the shade pass evaluates the
  mask and, below ½, repoints `const PShade* sh` at the loser's payload — so albedo, skin,
  normal map and pattern drives all switch together. Key consequences:
  - **Per MATERIAL, not per triangle.** Inlining a second payload in `PTri` would add ~58 B
    to a ~276 B struct (~+120 MB on a 2 M-triangle scene); the side table has one entry per
    weight-mapped mix however many triangles carry it.
  - **`PTri : PShade` by inheritance**, so every existing `pt.color` / `t.tex` reference in
    both backends still compiles. Safe because `PTri` is only ever default-constructed.
  - **Geometry and side table travel together** as `PreviewGeom` — a `PTri::mix` index is
    only meaningful against the `mixes` built in the same `tessellate()` call.
  - **`emissive` and `clear` deliberately come from child 0 only.** `g.emis` is written in
    the raster pass and consumed by auto-exposure *before* shading, and `clear` steers the
    separate see-through composite pass; neither can vary per pixel.
  - **`needUV` must include `mix >= 0`.** The mask is sampled at (u,v) whether it is a
    pattern or a scalar texture. Missing this was the whole visible bug on the first cut:
    `pattern_tex.ftsl` previewed with `u=v=0`, so its two `tex:`-driven walls came out flat
    and only the floor (which also reads world `x`) showed any structure. The GPU twin
    interpolates UVs unconditionally and needed no equivalent change.
  - A child that is *itself* a weight-mapped mix still flattens — no recursive per-pixel
    walk, and no scene in the library nests them.
- **One `applyMat()` assigns every per-material field**, called from all four geometry
  paths (world tris, spheres, implicits, instances). The class of bug being closed is a
  feature wired into three paths and silently dropped on the fourth.
- **The G-buffer stores a source triangle INDEX**, not a copy of each material field. The
  shade pass reads `tris[g.tri[i]]`, so every future per-material feature is free of a new
  per-pixel channel — and it matches what the GPU backend already did (`slot >> 1`).
- **Palette resolution lives in `Texture`, not in the rasterizer** (`buildPaletteRgb` /
  `paletteRgbAt`), so every consumer that wants a *colour* rather than a spectrum gets it.
  Previously `sampleRgb` on an index map returned the raw index out of the red channel —
  entry 3 of 12 shading as near-black.
- **The pattern VM is shared source, not a second implementation** (`pattern_device.cuh`).
  It was extracted verbatim out of `render_cuda.cu` and templated on the texture-record
  type, because the two backends store textures differently: the tracer uploads spectral
  `DTexture`s (Jakob-Hanika coefficients), the preview flat linear-RGB `DTex`s. The VM
  reaches a texture only through `TexT::patScalarAt(u,v)`, which each backend implements
  against its own storage (both mirroring the host's `Texture::scalarAt`). Everything else
  it touches — `PatGrid`, `PatScatter`, the flat float pool, the POV function library — was
  already shared `__host__ __device__` code in `pattern.h`.

CPU/GPU parity is verified on `gallery_rain`: of 1.17 M pixels, 6247 (0.54%) differ by
more than 2/255, and every one of them lies on a grid-line boundary — the pattern is a
hard step, the CPU evaluates it at a double-precision world position and the GPU at a
float one, so a sub-texel shift flips a pixel across the step. Mean absolute difference
over the frame is 0.12/255. The per-hit mix agrees to the same tolerance and for the same
reason (the ½ threshold is another hard step): `pattern_tex.ftsl` 0.076% of pixels over
2/255, `maskblend.ftsl` (a `weight_map texture:`) 0.068%, `_preview_pattern_tex.ftsl`
0.083% — unchanged from before the mix work, i.e. no regression. Both raster backends now
reproduce `-mode W`'s image of all three `pattern_tex` walls.

## CPU raster frame loop (`raster.h`, perf architecture, 0.136.1)

`renderFrame` is seven strictly-sequential parallel passes (project → zbuf clear →
G-buffer raster → optional see-through → shade → auto-exposure anchor → tonemap/encode).
The feature work above (textures / pattern VM / per-pixel mix / normal maps) added genuine
per-pixel cost, but profiling showed the frame was dominated by *fixed* overhead, removed
in 0.136.1 (2.4–3.5× at 1280×960; every optimization below is **byte-identical** on a
9-image corpus — the invariants that make that provable are the point of this section):

- **`RasterScratch`** (owned by the caller, one per explorer/bench session, passed down as
  an optional pointer): the G-buffer, accum, see-through buffers, per-thread projection
  parts and the `BandPool` all persist across frames. Before, every frame re-allocated and
  `assign()`-zeroed ~129 MB — pure page-fault + memset cost. Channels are now `resize()`d
  (no re-zero) and only **zbuf** is cleared, because every other channel is write-before-read:
  each is only ever read where `zbuf > 0`, and the raster pass that sets zbuf writes them
  all (auto-exposure's scan short-circuits on `zbuf[i] <= 0 || emis[i]`, and `emis` is
  written by the same raster pass). Stale-data hazard to preserve: `S.parts` buffers past
  the current thread count must be explicitly cleared or a shrink leaks old triangles into
  the concatenation.
- **`BandPool`**: a persistent worker pool (generation-counted condition-variable
  broadcast, `run(body(workerIdx))`, not nestable) replacing ~84 `std::thread` spawns per
  frame (7 passes × N threads). Bit-identity argument: the pool runs the **same partition
  formulas** (`chunk = (n+nT-1)/nT`, row bands) as the spawn path, so band ownership —
  and therefore every band-ordered write — is unchanged. `exposeAndEncodeT` takes the pool
  as an optional parameter (other caller: `render_cuda.cu`'s G2 iso preview, which passes
  none) and falls back to spawning if the pool's size mismatches `nThreads`.
- **`selectKthNonNeg`** (auto-exposure anchor): the p99-luminance anchor ran a *serial*
  `std::nth_element` over ~1.1 M doubles (~4 ms). Non-negative IEEE doubles order
  monotonically as raw bit patterns, so a parallel 16-bit-radix histogram (top 16 bits)
  locates the bucket holding rank k, a parallel scan collects just that bucket's members,
  and a tiny `nth_element` selects within it. Exactness: selection is by **value** over a
  multiset, so neither pack order nor tie order can change the result; small n / no pool
  falls back to plain `nth_element`. Requires non-negative inputs (the anchor packs
  `max(r,g,b,0.0)`).
- **Feature-cost hoists** in the shade pass: `normalize(g.wn[i])` computed once and shared
  by the pattern context and shading; the mix `weight_map` PatCtx is built lazily (texture
  masks don't need it); the normal-map tangent's raw dP/dU half (`triTangentRaw`) is baked
  per triangle at tessellation (`PTri::tanRaw`, zero when UVs degenerate ⇒ per-pixel
  fallback basis) leaving only the per-pixel Gram-Schmidt. The GPU twin deliberately keeps
  recomputing the raw tangent per pixel — its whole shade pass is 0.19 ms.
- **What was measured, not guessed** (`scenes/_raster_bench.ftsl` + its `_plain` twin,
  1280×960, 12 threads): features-heavy
  86 → ~36 ms median, all-plain 69 → ~20 ms; remaining with-anchor cost over absolute-EV
  is ~2–3 ms = the two collection scans, inherent to the percentile semantics. GPU was
  profiled first and left untouched: 2.73 ms/frame with all new feature code costing
  0.19 ms in `kShade`.

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

**Adaptive fly step (0.193.1).** The explorer's travel distance per frame used to be
`sceneR * 0.02`, with `sceneR` = half the diagonal of the AABB of *all* geometry
(`scene.h`'s `sceneRadius`). That is a bad proxy for navigation scale the moment a
scene's extent is set by something other than its subject: `gallery_rain` is eleven
~0.5 m exhibits spaced ~1.5 m apart standing on a 46×45 m floor, so `sceneR ≈ 32 m`
and one wheel notch (`kWheelDolly` = 8 steps) moved **5.2 m** — past three exhibits,
in a viewer whose job is to look at them. Fixed by sizing the step from what you are
actually looking at rather than from the bounding box: before each move `autoStep()`
casts **one ray** along the view direction and takes `2 %` of the distance to the
first hit (falling back to `sceneR` when the ray escapes), clamped to
`[sceneR*1e-4, sceneR*0.05]`. A notch is then ~16 % of the way to whatever is under
the crosshair — about six clicks to arrive, identically when crossing a hall and when
closing on a plinth. Verified at both ends of the scale: `gallery_rain` reports
`~0.301 u` (= 0.02 × the 15.04 m along the view ray to the floor) and `cornell`
`~0.0425 u` (= 0.02 × 2.13 m to its back wall) from the same code. The one-ray probe
is free next to the raster frame it precedes.
Consequence for `Ctrl`+wheel: it can no longer *set* an absolute step, because the
next frame's probe would overwrite it — it now scales a persistent `stepScale` bias
(×1.15/notch, clamped ×0.02…×50) that multiplies the probe result.
The **collision standoff** had the identical defect and the identical cause:
`kSkin = sceneR * 0.02` held the eye 0.65 m off every surface in `gallery_rain`, so a
0.5 m exhibit could not be approached at all. It is now
`clamp(sceneR * 2e-4, 1e-3, 0.05)` — sized against the raster's near plane, which is a
**fixed absolute** `zn = 1e-3` (`raster.h`) and does not scale with the scene, so a
scene-scaled standoff never had a justification. 6.5 mm in `gallery_rain` against a
1 mm near plane.

## Threading model (CPU)

Band/chunk parallelism via `std::thread` pools sized by `hardware_concurrency`;
work units pull atomically from a shared counter in chunks. Determinism comes from
per-unit RNG seeding (above) plus order-independent accumulation per band/tile;
film merges are structured so paired runs differ only by summation-order ulps at
worst (mode R) or are bit-identical (fixed splits).

**Load-time loops use `src/parallel.h` (`ft::parallelFor`), not a hand-rolled pool.**
Each renderer's pool is tuned to its traversal (raster.h keeps a persistent one,
photonmap.h bands by photon block, isomesh.h splits the lattice) and none is reachable
from the loader — so scene setup kept doing its embarrassingly parallel passes on one
core, which is how per-texel spectral upsampling grew into a 45-second startup stall
(see `upsample::fitMany`). `parallelFor(n, grain, fn)` is deliberately minimal: one
atomic cursor handing out `grain`-sized chunks (chunk-stealing, because these loops
have very uneven per-item cost), the caller acting as one of the workers, and a serial
inline path below `2*grain` so a small array spawns nothing. It is for **pure,
independent, one-shot** passes only; anything needing a reduction either keeps that
reduction serial (envmap's mean radiance) or must justify the changed summation order.

That same cursor is the program's **load-time cancellation seam** (0.138.2) — see
"Stopping a render" below.

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

`-stop all` targets every live ftrace process, a bare `-stop` lists them, and both wait
(≤120 s) for the targets to actually exit so a rebuild can be scripted immediately after.

### Stopping the loom VIEWER, and an honest exit code (0.182.0)

The channel was render-only until 0.182.0, and in the worst possible way: the `-viewer`
branch `return`s before `stopChannelStart()`, so a viewer published no `.run` file — and
the wait loop's predicate was `alive && exists(<pid>.run)`, which for an *unpublished* pid
is false on the very first poll. `-stop <viewer-pid>` therefore printed
`[stop] done — stopped cleanly.` and exited **0** while the viewer kept running. That is
worse than not supporting the viewer at all: the one command the project offers instead of
`taskkill /F` reported success for a stop it had not performed, and anything chaining off
it (`build.bat`, which can't copy `ftrace.exe` while a viewer holds it open) proceeded on a
false premise — straight back to the force-kill the channel exists to prevent. Three
changes close it:

- **The viewer registers.** `stopChannelStart(sidecar + " -> (loom viewer)")` /
  `stopChannelEnd()` now bracket `runViewerGui`, so a viewer is listed by a bare `-stop`
  and is a legal target. Nothing in the channel was ever render-specific.
- **The GUI polls.** `viewer_gui.cpp`'s `PeekMessage` loop calls `ft::stopRequested()` once
  per frame (the probe is installed at `main.cpp:18485`, *before* the viewer dispatch) and
  sets `done`, leaving through the ordinary teardown that releases D3D11, the loom python
  child and the window. `-explore` needed nothing: its loop already reads `g_stopRequested`
  directly (`main.cpp:17092`), and it registers because `stopChannelStart` precedes `run()`.
- **The wait can't lie.** `stopTargetGone()` replaces the conjunction: on Windows process
  liveness is authoritative and is the *only* thing consulted, so a live target is never
  counted as gone. (Off Windows `ftraceProcessAlive()` is a conservative "yes", so there the
  `.run` file disappearing is still the signal.) A pid that was already dead returns 0 with
  `nothing to stop`; a pid alive but unregistered stays a target and, if it outlives the
  deadline, the command prints `[stop] FAILED — still running after 120s` and exits **2**.

The directory scan also **reaps orphan `.stop` sentinels**, not just orphan `.run` files. A
sentinel is normally deleted by the target that consumes it, so one only survives when nobody
is left to consume it — the target died first, or never watched the channel (the
alive-but-unregistered path drops one deliberately). Eight had piled up over five days before
this was noticed. It is housekeeping rather than a safety net: `stopChannelStart` already
deletes any sentinel bearing its own pid before starting its watcher, so a recycled pid cannot
inherit somebody else's stop and kill itself at startup. The reaper only fires once the owner
is gone, so a sentinel still in flight to a live target is never snatched away.

### Stopping during SCENE LOAD (0.138.2)

The pid is published *before* `run()`, so a process can be signalled from the moment it
starts — but until 0.138.2 the loader polled nothing, and a stop aimed at a process that
hadn't reached its render loop was accepted and then waited out the full load. The load is
now cooperatively cancellable at two granularities:

- **Inside a long pass:** `ft::parallelFor` polls the flag at its chunk cursor, before
  claiming work, so a stop drains the cursor and each worker finishes at most one chunk.
- **Between assets:** `Builder::stopped()` polls it between top-level blocks in the
  texture / pattern / `mesh_asset` / geometry / deferred-`medium` passes, which covers the
  still-serial loaders (glTF/OBJ import, `meshvox::voxelizeSolid`, isomesh tessellation)
  without threading them.

`parallel.h` cannot name `g_stopRequested` — it is a file-static `volatile sig_atomic_t`,
because a signal handler writes it — so `main()` hands it over as a **probe**
(`ft::setStopProbe`) before any scene work. Reading it through a function pointer is free
here precisely because it is read once per *chunk*, never per item.

A cancelled `parallelFor` returns `false` and its output is **partial**, so the return is
`[[nodiscard]]` and the contract is *abandon the load*, never "carry on with half a
texture": `upsample::fitMany` → `Texture::buildReflCoeff` (which clears the partial
coefficient table) → `addTexture` → load failure, and `EnvMap::buildFromRgb` → its `err`
out-param. `main.cpp` reports it as `[stop] scene load stopped before rendering` and exits
1 rather than printing a scene-error diagnostic. `prefer { } else { }` resolution aborts
outright on a stop — treating an interrupted branch as *rejected* would otherwise make it
build the next branch and ignore the stop for another whole load.

### Stopping during a PHOTON DEPOSIT (0.197.2)

The same hole existed one stage later, and lasted longer. `tracePhotonPass`
(`photonmap_render.h`) — the forward deposit that builds the map for modes `M` and `S` —
looped `for (i = lo; i < hi; ++i) tracePhoton(...)` with **no poll of any kind**. 0.194.0 had
taught the mode-`M` camera *gather* to stop, which made the omission easy to miss: a stop
aimed at a mode-`M` render *was* honoured, just not until the entire `-n` had already been
traced. On `-n 2000000000` that was ten minutes of a process ignoring every stop request
while its photon map grew without bound (10 GB of private bytes and climbing) — the exact
scenario that pushes someone toward the `taskkill /F` this whole channel exists to replace.
An unresponsive `-stop` is a safety bug here, not an annoyance.

The worker now polls `ft::stopRequested()` **every 4096 photons**. The granularity matters in
both directions: a photon is microseconds, so a per-iteration atomic load would show up in the
hottest loop of a mode-`M`/`S` build, while 4096 of them still lands the stop in well under a
tenth of a second. On break the thread reports the count it **actually traced** rather than its
assigned share — `pm.nEmitted` normalises the density estimate, so crediting untraced photons
would darken a truncated pass rather than merely making it noisier. Measured: `-n 4000000000`
stopped after 182,222,848 photons and exited 3.5 s after the `-stop`.

## Where a relative asset path is looked for (`assetbytes::resolve`, 0.192.0)

Every file an FTSL scene names — meshes, textures, `.vdb` volumes, `file:` spectra, animation
sidecars — passes through **one** function, `assetbytes::resolve()`, on its way to an `open`.
Before 0.192.0 there was no such function and no notion of a base directory anywhere: the
authored string went straight to `fopen`, so it resolved against the **process working
directory** and a scene was loadable from exactly the one directory its paths had been written
relative to. That is not a cosmetic limitation — it is the first link of the chain documented
in the next section, where a wrong cwd became an unloadable scene, an unloadable scene became
a silently *empty* one, and an empty scene faulted the display driver.

`resolve()` tries these roots in order and returns the first that exists:

| # | root | why |
|---|---|---|
| 1 | **the path as authored** (relative to the cwd) | first *on purpose*: every invocation that worked before resolves to the identical file, so the search path can only turn failures into successes, never move a load from one file to another. |
| 2 | **the scene file's directory** | a scene's assets belong to the scene — the rule every format that references external files uses (glTF buffer URIs already worked this way). |
| 3 | **the scene's ancestors, ≤ 3 levels** | the natural layout puts scenes in a subdirectory and assets in siblings: `proj/scenes/x.ftsl` naming `textures/y.png` means `proj/textures/y.png`. This repo is laid out that way, so stopping at row 2 would not have fixed the reported bug. Bounded so a stray name can't be answered from the root of a drive. |
| 4 | **the directory of `ftrace.exe`** | engine data (`data/glass/*`, `data/metal/*`) ships beside the binary and has nothing to do with the scene. |

Absolute paths pass through untouched. A path matching **nothing** comes back *unchanged*, so
the error message quotes what the author wrote rather than the last candidate tried, and
`describeOpenFailure` then runs its did-you-mean scan under every root and lists the
directories it searched.

Two deliberate scoping decisions:

- **The scene directory is scoped, not global.** `ftsl::loadSource` (which already receives the
  scene's path) holds an RAII `assetbytes::ScopedSceneDir` for the duration of the load. A
  later load of a different scene must not inherit it, and every branch of a `prefer` block
  must see the same one. A `nameForMsgs` that isn't a real path (the loom live channel passes a
  placeholder) has no parent directory and simply leaves the search path alone.
- **Engine data does not consult the scene directory** (`speclib::categoryDir`). `speclib::index()`
  caches a name→path map per category for the process lifetime, so a scene-dependent answer
  there would be frozen at whichever scene happened to load first. It tries cwd, then the
  executable's directory, both of which are process-constant.

`assetbytes::readFile` resolves internally, so every loader reading bytes through it
(OBJ/PLY/STL/`.ftmesh`) inherits the policy without a call site. The loaders that open files
themselves call `resolve()` explicitly: `texture.h`, `gltf.h` (resolving the document *before*
deriving the `baseDir` its external URIs hang off), `fbx_load.cpp` (ufbx insists on a path),
`curvedrive.h`, `vdbgrid.cpp`, `vdb_openvdb.cpp`, and the FTSL `file:` spectrum loader. The
`Warmer` prefetch resolves on the *calling* thread, since the search path is scoped to the call
and reading it from the warmer thread would race the scope's exit. `assetbytes::Overlay` is
consulted **before** any of this and keys on the authored path, so the loom live channel is
untouched by resolution.

`-checkpaths` (`main.cpp`) is the regression guard: it builds a real `proj/scenes` +
`proj/textures` tree on disk and asserts the end-to-end load from an unrelated cwd, the
ancestor walk, absolute passthrough, unresolvable-returns-unchanged, that **the cwd still
wins**, that the scene directory is restored on scope exit, and that `data/glass` resolves
beside the executable.

## `prefer { } else { }` resolution (`ftsl.h loadSource`, hardened 0.191.2)

A `prefer` node lets one scene file name several ways to render itself and let the loader
pick — the gallery wraps its camera so the scene asks for mode `D` and settles for mode `B`
when `D` can't render something. Resolution **trial-builds** candidate scenes: `tryBuild`
flattens the block list with one branch spliced in per node, runs a fresh `Builder`, and
returns one of exactly **three** outcomes, which the rest of the algorithm must not conflate:

| `Trial` | meaning |
|---|---|
| `built == false` (`buildErr` set) | a real authoring error. The branch produced **nothing**. |
| `built`, `reason != nullptr` | a real scene that *this mode* cannot render (the caller's `SupportFn`, from main.cpp's per-mode gates). |
| `built`, `reason == nullptr` | renderable. Take it and stop. |

**The fallback exists only for the middle row.** Through 0.191.1 the single-node fast path
wrote `if (singleNode && (renderable || c == nb - 1)) accepted = trial;` — accepting the last
branch *whether or not it built*. When every branch failed (the realistic trigger: a scene run
from the wrong directory, so every cwd-relative asset missed — the path resolution described
in the previous section is what stops that happening), `loadSource` returned **true**
with a default-constructed `Loaded`: 0 cameras, 0 geometry, 0 emitters, announced as
`[ftsl] loaded scene from …`. The empty scene then reached the CUDA forward kernel and faulted
the driver. So the standing invariant is: **a `Loaded` that no build ever produced is never
handed back.** The fallback is the last branch that actually *built*; when none did, control
falls through to the final rebuild, whose failure carries the builder's own error up.

**Resolution is order-dependent, so it sweeps to a fixed point.** Nodes resolve
left-to-right, and while node *j* is being tried every *other* node sits at its current
choice. A **later** node whose branch 1 doesn't build therefore makes **every** trial of an
**earlier** node fail, for a reason belonging to neither. The old code hid that behind the
unconditional fallback; making the fallback strict exposed it as a spurious whole-scene
failure. The pass now repeats until a whole sweep changes no choice — at which point that
sweep's trials are known to have used the final context — capped at *nodes+1* sweeps against
oscillation. Per-branch diagnostics are **buffered** and only the converged sweep's are
printed, so an unconverged sweep's artifacts never reach the console. A **single** node has no
other node to be perturbed by, so it still resolves in one sweep; that is also the case the
fast path serves, keeping the resolving trial as the final scene instead of re-loading every
mesh a second time.

A stop request (`ft::stopRequested`) aborts resolution outright rather than counting as a
branch rejection — see above. `-checkprefer` (`main.cpp`) is the regression guard: seven
cases covering all three `Trial` outcomes, the single-node fast path, the multi-node rebuild
and the ordering artifact.

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

A **whole missing primitive** is the same rule at the coarsest grain. When 0.150.0 added
`curveSegs` as a fifth prim range, the megakernel's hit routine still knew only four
(`tri | sphere | implicit | instance`), so the gate sent any curve scene to the CPU
wholesale. Without that gate the device would not have failed at all — it would happily
miss every strand and render a furred subject **bald**, which is far worse than a fallback
because it looks like a plausible image. 0.151.0 removed the gate by writing the missing
device twin (`DCurveSeg` + `intersectCurveSeg`, and the fifth range in both `closestHit`
and `occluded`); what survives is the *screen*, which still checks curve **materials**
exactly like every other primitive's. That is the intended lifecycle of one of these
gates: ship it the moment the hole exists, delete it only by filling the hole.

**A kernel prologue is the last of these gates, and each sampler needs its own.** The host
uploads a *null* pointer for an empty array (`d_ems = dems.empty() ? nullptr : …`), so any
device sampler that indexes `sc.emitters[…]` without first checking `sc.nEmitters` faults the
driver rather than producing a black frame. The three forward/BDPT emitter samplers had drifted
apart on this: `dGenLightSubpath` tested `sc.nEmitters == 0 || sc.totalPower <= 0.0`,
`genPhoton` tested only `grandTotal <= 0` (which lets a scene with emissive volumes but no
emitters through, since the volume-birth coin can miss on `uniform() == 1`), and `genPhotonHero`
tested **nothing at all** — which is why an emitter-less scene faulted only in the hero kernel.
0.191.2 gave both forward samplers the missing test, returning `false` ("skip this photon", the
contract both call sites already honour). Such a guard must draw **no randomness**, or it
perturbs the RNG stream and changes every image; both are pure reads, so scenes that do have
emitters are bit-identical. The scene that exposed this could only arise from the `prefer` bug
above, but the guard is the right place for the invariant regardless: a malformed scene should
cost a black frame, never the display driver.

## Benchmarks & perf discipline

- `scraps/bench.py` — 19 standard configs (13 CPU + 6 GPU; cornell + gallery
  scenes); min-of-reps timing, sha1 of PPM outputs, `fuzzy_ppm_diff` for GPU.
- `scraps/bench_ab.py` — interleaved A/B harness (exeA/exeB alternate per config so
  machine drift cancels); used for the 2026-07 optimization campaign report
  (`scraps/bench_final_ab.json`).
- Rule: any hot-path optimization must be **bit-identical** (CPU sha1) or
  visually/fuzzy identical (GPU) vs. the pre-change exe before committing, one
  commit per optimization so any regression can be reverted alone.

## Measurement ROIs come from the renderer, not from the scene file (`roiboxes.h`, `-roiboxes`, 0.279.0)

Scoring an A/B on the whole frame averages the region under test together with everything that
cannot respond to it, so this project's standing rule is to score per-ROI. Enforcing that needs an
ROI per scene, and for a long time exactly one existed (`scraps/gallery_rain.rois`) because building
one meant hand-projecting primitive coordinates through the camera and then checking each box
against the projected footprint of every nearer object. The rule was therefore unenforceable
everywhere else, and cross-scene comparisons silently degraded to whole-frame.

`-roiboxes` removes the hand work by not reimplementing visibility at all. It fires one pixel-centre
camera ray per pixel — through the same `Scene::closestHit` and the same `RenderCam` the render will
use, which is why the hook sits *after* `toRender` is final — and records the material each pixel
actually sees. Occlusion needs no reasoning: it is whatever the intersector returned.

The remaining trap is that a bounding box is not a region. A material used in several places has a
bbox spanning all of them plus the gaps. So the per-pixel material image is flood-filled into
connected components in a single pass over the frame (components keyed by material id, 4-connected),
only the largest component per material is reported, and each box carries **purity** (pixels inside
it that really show the material) and **share** (the material's pixels inside that component).
Failing boxes are printed commented out with the reason, so the output cannot be copy-pasted into a
bad measurement.

**The y origin is derived, not asserted.** `Camera::genRay` maps `py = 0` to `sy = -1` (`-v`), so
raster row 0 is the image *bottom*, while the `.rois` format and `tools/roi_score.py` measure y
downward from the top. The first implementation emitted raster rows directly and produced boxes that
were right in x, landed on real objects, and were vertically mirrored — a failure with no visible
symptom in the numbers. `roiBoxesReport` now traces the first and last row and compares them against
the camera's own up vector to decide which end is the top, so the answer survives any future change
to `genRay`'s film mapping.

## Scene-authoring tools (`tools/`)

- **`settle.py`** — rests a *single* object on a surface by lowering it vertically.
- **`settle_scene.py`** — runs ONE pybullet sim containing many of a scene's objects
  and rewrites the `.ftsl`, wrapping each settled block in
  `group { translate … rotate … <original block> }`. Non-selected named objects
  become static concave colliders, so pieces can rest on each other. Isosurfaces are
  polygonised by shelling out to `ftrace -export-mesh` (whose OBJ groups are named
  after the FTSL block, which is how each group is matched back to its object), so
  the tool depends on a current `ftrace.exe` — it resolves the **repo-root** binary
  first, then build dirs newest-first.

  The delta baked into the group is `(pos − R·c, R)`: bodies spawn *at* their authored
  COM `c`, so the sim works in `v − c` and `pos + R·(v − c) = (pos − R·c) + R·v`.
  This matches FTSL's `group { translate t; rotate R }` = `p' = R·p + t` — verified
  numerically against the render, not assumed.

  Three mechanisms exist because a faithful free settle drops pieces onto *narrow*
  pedestals and they roll off:

  - `--tether k` — horizontal restoring spring applied **at the COM** (hence no
    torque: free to tip onto its cap, not free to walk off it), with a deadband so a
    piece inside tolerance settles naturally. It is a **fictitious body force that
    does not vanish at rest**, so it is ramped to zero and the pose re-settled before
    being read; otherwise the bake records a pose gravity alone cannot hold.
  - `--jitter deg` / `--seed` — random spawn tilt so a symmetric body can't rest in
    an unstable equilibrium (a ring balanced on its rim). A single draw is not enough
    for a solid of revolution — a tilt about its own symmetry axis perturbs nothing —
    so a perched piece is automatically re-thrown up to `SETTLE_ATTEMPTS` times,
    re-using the built collision world rather than redoing VHACD.
  - `--seat` — post-hoc geometric fallback: keep the settled orientation, restore the
    authored XZ, lower straight down.

  **Acceptance test — three stages, because equilibrium is not stability, and stability
  is not correctness:**

  1. *Support polygon.* The COM must project inside the convex hull of the
     **load-bearing** contact points (`convex_hull_2d` / `support_margin`), within a
     tolerance: smooth bodies genuinely touch at a point or along a line, so this is a
     tolerance rather than a required inset. It catches gross failures (a piece resting
     on the corner of its cap with the COM out over air). Failure prints `PERCHED`.
  2. *Poke.* A wheel balanced on its rim has its COM exactly over its contact and
     passes stage 1 perfectly. So each piece is given a small random shove + spin and
     re-settled; a stable rest absorbs it, an unstable one topples. The pose that
     **survives the poke** is the one baked. Failure prints `TOPPLES`.
  3. *Intended support* (`intended_supports()`). Stages 1 and 2 are both **local** — they
     interrogate the pose the sim produced and never look at the scene the author wrote.
     A piece that slid off its cap, fell a metre and wedged between two pedestal shafts
     passes both with *healthy* numbers, because down there it really is immovably at
     rest. `heart` shipped exactly that pose, reported `OK`. So the tool now reads the
     author's intent off the scene **before anything moves**: a piece's intended supports
     are the other named objects whose plan (XZ) footprint overlaps its own and whose top
     is below the piece's **mid height**, and it `FELL` unless it ends up resting on one
     of them *and* still above that support's top. Both halves are load-bearing: the
     wedged heart still *touched* `stand_heart` on the way down, so membership alone
     passes it. The mid-height rule (rather than "below the piece's underside") is what
     lets a **mount** count — a collar's top is above the piece's lowest point whenever the
     piece hangs down inside its bore, as the retired `collar_klein` did.

     Note this is deliberately *not* a displacement check. How far the COM moved is the
     obvious metric and the wrong one: `heart` settling correctly under `--tether` moves
     222 mm (it honestly tips from its authored tilt onto a stable lobe), while a piece
     can slide clean off a narrow cap having moved far less. `FELL` is reported *before*
     `PERCHED`/`TOPPLES` precisely because a fallen piece's other numbers look fine.

  Contacts are read from the manifolds the last `stepSimulation()` left behind:
  `performCollisionDetection()` rebuilds them, and `normalForce` is the solver's applied
  impulse, so every fresh point reads zero force. The load-bearing threshold is a
  *fraction of the body's total* normal impulse, not an absolute force — a VHACD proxy
  spreads the weight over dozens of manifolds, so an absolute cut rejects every genuine
  contact on a finely decomposed body. Dynamic bodies also disable sleeping, or a settled
  body drops out of the solver and reports no contacts.

  **Friction units matter.** Bullet's `rollingFriction` is a resistance *arm in metres*:
  it caps the resistive torque at `mu_r · N`, so a body of radius R cannot tip past
  `asin(mu_r / R)`. A plausible-looking 0.02 is 2 cm, which pins any gallery-scale piece
  upright — it held `brass_dumbbell` balanced on its rim below 9.8°. Values are now
  physical (`ROLLING_FRICTION = 5e-4`).

  **Some shapes have no stable rest pose at all**, and no amount of simulation invents
  one — the tool says `TOPPLES` on every retry and the *geometry* is what has to change.
  `brass_dumbbell` was one: its ring's outer radius exceeded the balls' radius, so the
  ring was the lowest feature, the balls could never reach the stand, and tipping ran away
  to axle-vertical. Shrinking the ring under the ball radius lets it rest on its two
  spheres, which is what a dumbbell at rest should look like.

  The **third possibility is that the model itself is wrong for the shelf**, and that is
  what the `klein` bottle turned out to be. The original image-to-3D bottle
  (`klein_hunyuan.obj`) had *no* acceptable rest pose: enumerate them — the convex hull's
  faces whose supporting plane has the COM over them *are* the poses it can rest in — and
  of the 44 the most upright leans 73°. A mesh that is art must not be altered, so the fix
  was a **mount that grips rather than supports**, a collar whose bore was cut to the
  piece's own cross-section (`tools/make_klein_collar.py`, retired 2026-08-03). The real
  fix was a better model: `meshes/klein_bottle_full.obj` is a glassblower's bottle with the
  neck genuinely continuing *inside* the bulb and a punted foot, so it stands — foot ring
  radius 66 mm under a COM 225 mm up, a 6.5° static tipping angle, 0.00° settled lean and
  0.2 mm poke drift on the bare slate cap. **No mount at all is the strongest mount**, and
  when a piece cannot stand, ask whether the geometry is the thing to replace before
  engineering around it.

  A seat ring for the new bottle was designed and rejected on measurement: the body flares
  continuously off the foot (66 mm at the base → 83 mm 4 mm up → 112 mm at 20 mm), so a bore
  loose enough to lower the piece into is loose enough to let it slide the same distance,
  and `slab_sections`' 8 mm vertical quantisation turns that slope into ±13 mm of bore slop
  in the sim regardless. A ring would have been decoration.

  `heart` is the third variety: a shape that *has* stable rests, just not the tilted one it
  is authored in. It therefore tips as it lands, and **tipping translates**. Free, it tips
  right off the cap; tethered, it stays on the stand but the tip walks it 222 mm sideways,
  leaving 148 mm hanging past the rim. The gallery bake uses `--tether` to keep it on the
  pedestal *and* `--seat heart:stand_heart` to put it back over the cap centre — on a flat
  level cap seating is a pure horizontal translation of the whole contact set, so it
  preserves the support margin exactly rather than trading stability for looks.

  Independent verification uses `-export-mesh` + per-group AABBs, which is exact and
  involves no physics at all: every settled piece's bbox must sit on its stand's cap.
  (This is `scraps/settled_aabb.py`, and it earned its keep: it read `heart bot 0.213`
  against a 1.000 cap while the bake called the same pose `OK`. When two tools disagree,
  the one that gates the output is the one to distrust.)

  **Collision-geometry reduction is what makes the tool usable.** Sim cost is set almost
  entirely by the number of *static* triangles in the contact patch under a resting piece
  — not by the total scene triangle count (with the dynamic bodies moved away, a 3.6 M-tri
  static set steps in 0.01 ms), and not by solver iterations. An un-reduced marching-cubes
  pedestal cap is thousands of slivers where two triangles would do, which cost the gallery
  bake 60 ms/step, i.e. ~40 min per run. Static colliders are therefore reduced two ways,
  in order of fidelity:

  1. **Quadric decimation** to `STATIC_TRI_CAP`, used whenever it actually reaches the cap
     (it does for clean closed shapes: the gyroids, lamps, `chrome_ring`). This keeps the
     concave shape, so it is always preferred.
  2. **`slab_sections()`** for the meshes where decimation stalls — the box-union pedestals
     bottom out at 25–47% of their input no matter how many passes or how much aggression,
     having already lost 29% of their volume. The mesh is cut into `STATIC_SLABS` horizontal
     slabs; each slab's true cross-section is taken at its mid-height (`section_multiplane`
     → `Path2D.polygons_full`, which needs `shapely`/`rtree`), simplified to
     `SECTION_SIMPLIFY` (0.5 mm — a raw marching-cubes outline carries thousands of
     near-collinear vertices and extrudes to ~83 k tris instead of ~2.4 k), and extruded
     back over the slab. Each prism is placed with the *section's own* `metadata['to_3D']`
     frame — the 2D frame's axes are not (x, z), and hardcoding a rotation silently
     transposes a stand's footprint — then dropped half a slab, because `to_3D` puts z = 0
     at the section height, i.e. the slab's middle. Cap height and XZ extent come out exact,
     volume to a few percent, at ~2 500 tris per stand.

     Sections rather than hulls, because **convexifying a slab fills any hole in it**. That
     is not academic: a mount with a bore is an annulus, and slab hulls plug it, so the
     settle would run against a solid plinth and bake a pose the real scene cannot hold.
     (Hulls also badly overstate the concave stands — `stand_gyroid` comes out at 3.73
     against a true 1.76 volume.) `slab_hulls()` survives as the fallback for when the
     shapely/section machinery is unavailable, and says so loudly when it is used.

     `STATIC_SLABS` is derived from `STATIC_SLAB_MAX_T` (a target slab *thickness*), not
     fixed: a slab is a stair-step, so any horizontal feature is only resolved if it is
     thicker than one slab. A fixed 32 slabs is 32 mm on a 1 m pedestal, which quantised an
     earlier in-pedestal collar bore into a 32 mm dimple with its floor 3 mm above the real
     cap. Even at the current 8 mm target this is the binding constraint on any *small*
     feature machined into a pedestal cap, and it is why the new Klein bottle gets no seat
     ring: a bore is only faithfully simulated if it is several slabs tall, and on a body
     that flares 1.2–5 mm of radius per mm of height, several slabs of height is centimetres
     of bore slop. A mount that must survive this path has to be a **hand-built ≤4000-tri
     mesh**, which is what `collar_klein` was — under `STATIC_TRI_CAP` a static collider is
     used verbatim, so a bore cut to a quarter of a millimetre survives into the sim instead
     of being decimated or re-sliced.

  Whole-mesh hulling and VHACD were both rejected for stands: one hull is exact at the cap
  but fills the taper between a wide base and a narrow column, inventing a shoulder a piece
  could rest on; VHACD shifts the cap top 5 mm.

  **Caching** (`scraps/.settle_cache/`, `--no-cache` to bypass) memoises the two pure,
  expensive setup steps: the `-export-mesh` polygonisation, keyed on (scene text,
  `--mesh-res`, ftrace mtime), and the VHACD dynamic proxies, keyed on the proxy mesh's
  content hash. Iterating on `--tether`/`--jitter`/`--seed` then skips both. Per-phase
  timings are printed so a slow or non-converging bake is visible rather than silent.

- **`make_klein_collar.py`** — **RETIRED 2026-08-03** (deleted with `meshes/collar_klein.obj`
  when the gallery's Klein bottle was replaced by one that stands; see the `klein` note under
  `settle_scene` above and known-issues.md). It generated the gripping collar that held the
  old `klein_hunyuan.obj` upright, and remains the worked example of *what to do when a piece
  has no acceptable rest pose* — a stack of 4 mm slabs, each an outer disc with the piece's
  *own outline at that slab's mid height* punched out of it, so the piece jams where its
  section equals a bore and carries its weight on a full perimeter of ledge whose plan shape
  keys it against yaw and sway as well as lean. Three invariants it taught, each learned by
  violating it: the bore must be the outline **polygon** and not a radius per azimuth (a
  non-star-shaped section fills its own radial concavities — 12/36 pokes held versus 36/36);
  the bore must **never re-narrow going up**, or the mount is captive and a display mount
  must not be; and the bore must be cut to the **VHACD proxy**, not the true mesh, because
  that is the body the sim collides — at the cost of a visible gap in the render. The
  general lesson outlived the tool: engineering a mount around a bad model is more expensive
  than replacing the model.

## Build & release

- **Build time (0.259.2).** Measured on the i7-8700K with ~3 cores of other load: a build
  that recompiles only `render_cuda.cu` is ~3:37 through MSBuild — CudaCompile 174 s, host
  link 30 s, device link 6 s — so the build system adds ~40 s and the rest is nvcc on one
  18k-line TU, whose optimiser (`cicc`) and assembler (`ptxas`) are single-threaded per TU.
  Three changes: (1) `--split-compile=0` partitions that TU's device code across all cores
  (ptxas 52.6 → 21.1 s, cicc 64.3 → 43.2 s, the TU 2:49 → 1:50; codegen checked by a render
  speed A/B); (2) `/MP` for the host TUs, which only matters on a full rebuild; (3) the
  version string moved into `src/version.cpp`, the only TU with `FTRACE_VERSION` defined —
  it used to be target-wide, so every `VERSION` bump rebuilt *everything*, which is what
  made the "15-minute builds" of 2026-09-06: bumps plus foreign CPU load, not the compiler.
  What is left is serial by nature: `cudafe++` (14 s), preprocessing (12 s), the host pass
  of the .cu (17 s) and the links; splitting `render_cuda.cu` into per-mode TUs would
  parallelise those too, at the cost of duplicating the shared device helpers per TU.
  One constraint `--split-compile` adds: a `__device__` function with EXTERNAL linkage (a
  plain `inline` template in a header, as `dPatternEval` was) can be partitioned away from
  a `__launch_bounds__`-capped caller, and `nvlink` then rejects the link when the callee's
  own register allocation exceeds the cap (169 vs 168 for `kBdptT`). Device helpers must be
  `static` (internal linkage) so the partitioner keeps them with their callers — already
  the house style, and now a requirement.
- `build.bat` → CMake/VS2022 x64 Release into `build_cuda2/`, copies
  `ftrace.exe` to the repo root. **Warning:** freshly-configured build dirs
  currently produce a GPU-silently-dead exe (see known-issues, 2026-07-22) — build
  in the long-lived `build_cuda2`.
  **A running render no longer blocks the build (0.141.0).** Windows locks a live
  exe against write/delete but still permits *rename*, so if the copy fails
  build.bat moves the old binary aside to `build_cuda2/ftrace.locked.<n>.exe` and
  installs the new one into the freed name; the running process keeps its mapped
  image and is entirely unaffected (verified against two concurrent renders). Stale
  parked copies are reaped at the start of the next build, once their holder has
  exited. This exists to remove the one situation that used to tempt a
  `taskkill /F` — which can wedge the NVIDIA driver into a TDR. Only if the rename
  *also* fails does build.bat error out, and it then restores the old exe so the
  root is never left without one.
- `VERSION` (single `MAJOR.MINOR.PATCH` line) bumps with every observable rebuild;
  `release.bat` publishes repo-root `ftrace.exe` as GitHub release `v<VERSION>`
  (refuses on duplicate tag). CMake also `file(READ)`s it into the
  `FTRACE_VERSION` compile definition (with `CMAKE_CONFIGURE_DEPENDS` on the file,
  so a bump re-configures), which `ftrace -version` / `-V` prints and the `-h`
  banner carries. Before 0.141.0 the binary was anonymous — two builds could only
  be told apart by hashing them — so anything that needs to know which build it is
  looking at should call `-version` rather than trusting a file date.
- Output conventions: renders → `ppm/`/`png/` (flyby series in `png/<set>/`),
  scratch scripts → `scraps/`. Renders always launched with `-keepwindow`
  (+ `-checkpoint`/`-interval`) and outside the Bash sandbox so the live window is
  visible.

## `creature/` — the simulated-animal subproject

A **second, self-contained project living in this repo**, not part of the renderer.
It builds physically-simulated animals whose motion is *learned* rather than keyframed
(articulated skeleton → muscle/tendon actuators → soft tissue → fur, driven by a neural
controller). It has its **own** `design.md`, `known-issues.md` and `todo.md` — those are
authoritative for it, and this file does not duplicate them. Layout: `ftcl/` (lexer,
parser, expression evaluator for the `.ftcl` creature-description language), `creaturelab/`
(schema, model build, MJCF emitter for MuJoCo, tuning, validation), `rigs/` (`.ftcl`
sources, e.g. `canis.ftcl`), `tools/`, `tests/` (28 pytest cases).

**It is Python with a heavy native stack** — MuJoCo 3.11 and a CUDA-13 build of Torch —
so unlike the renderer it needs a virtualenv, and `requirements.txt` pins it (including
the pytorch cu130 extra index, without which `torch==2.13.0+cu130` will not resolve). The
`.venv/` is gitignored and *not* reproducible from the repo alone; recreate it from
`requirements.txt`. `creature/.gitignore` is its own and its patterns are relative to
that directory, so it keeps ignoring `out/`, `scraps/`, `.venv/` and `*.log` correctly
now that it is nested — the root `.gitignore`'s equivalents are anchored (`/out/`,
`/scraps/`) and deliberately do *not* reach into it.

**It arrived by `git subtree add --prefix=creature`, not by copying**, so all 15 of its
original commits are in this repo's history rather than being flattened into one "import"
commit — those messages are substantive design records (the armature-measurement fix, the
capture-rig decision) and were worth keeping. The consequence to know: `git log` now
interleaves creature and renderer commits by date, so use `git log -- creature/` to see
just one side. Its checkout also normalised six files from CRLF to LF, matching this
repo's `core.autocrlf=input`; content is otherwise byte-identical.

**The naming collision to not trip over:** `creature/` (this subproject) is unrelated to
the renderer's *fur creature* — `scenes/fur_creature.ftsl`, `fur_creature_gi.png` and the
`fur { }` groom generator in `src/fur.h`. The latter is a shipped ftrace demo scene; the
former is a simulation project that does not yet feed it. Nothing in `src/` or `scenes/`
references `creature/`.
