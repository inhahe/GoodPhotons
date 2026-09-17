# TODO — in flight

Live working plan. Each item says what, why, how it gets validated, and where it stands, so an
interruption costs the work in progress and not the plan. Finished items move to `known-issues.md`
(with their measurements) and come off this list; `design.md` gets the architecture.

Ordering note: **B was done first even though A was asked for first**, because mode M is one of the
backends A has to be validated in and a spectral error there would have contaminated every number A
produces. Measure the ruler before measuring with it. B is now done; A is next.

---

## B. Mode M mis-colours coloured speculars — **DONE (v0.321.0)**

Fixed on both backends and written up in `known-issues.md` (SPECGATHER). Mode M / mode D on the
Cornell control went JH white 0.880 -> 1.006 and JH red 1.142 -> 1.007, with flat spectra unmoved;
the `_beams_ms` invariant reads 1.058 against its recorded 1.085; a gallery_rain frame costs +4 %.
`gallery_rain`'s gold gyroid and chrome ring now render their true colour in mode M.

Left open deliberately, recorded in the same entry: media transmittance (stochastic, needs the
hero-wavelength treatment rather than this one; inert for gallery_rain, whose media have flat
coefficients), the Hair BCSDF, and the binary stochastic choices in HalfMirror and the layered coat.

## C. Mode M mis-renders coloured media — **DONE (v0.322.0)**

Fixed on both backends; write-up and numbers in `known-issues.md`. A coloured fog went from a
**113 % channel spread** against mode D (blue 64 % bright, red 23 % dark) to **1.0 %**, with the flat
control unmoved at 0.3 %. Three tiers — flat (one scalar walk, free), homogeneous (analytic), and
heterogeneous (correlated ratio tracking). `gallery_rain` costs 96 s, unchanged.

Outstanding: an end-to-end number for the coloured-AND-heterogeneous combination. Those renders
exceed 20-40 minutes — but the control proves that is the scene's medium and not the spectral
vector (the same scene with a flat spectrum, taking the scalar path, is equally slow). What is
needed is a cheap scene that exercises that tier, not a redesign.

## A. The analytic coated-body model — TIR saturation, coat absorption, Snell

**Status: A1 DONE (v0.323.0), A2 DONE (v0.324.0). A3 analysed and deliberately NOT built
-- it is inseparable from the deferred explicit multi-bounce BSDF, whose trigger it has
fired with measured numbers. See below and `known-issues.md`.**

Today `MatType::Layered` models the coat **on the way in and not on the way out**: the coat reflects
with probability R and otherwise the ray enters and a body lobe shades. There is no exit interface,
so no total internal reflection, and no absorption in the layer. Everything below is multiplicative
on the body's contribution, so it preserves **one lobe per vertex** — the property that lets a
layered material render in every mode on both backends from one definition.

### A1. Exit interface + internal multiple reflection -- **DONE (v0.323.0)**

Implemented as an **effective albedo** on the body rather than as a BSDF term, because that makes it
a property of the MATERIAL: no interface change, so it works in every mode on both backends at once.
`Scene::finalizeLayeredCoats` mints a per-stack body COPY and sets `Material::coatFdr` on it; the
four albedo funnels (`diffuseReflectance`, `reflectSlot`, and `dReflectSlot` on the device) apply

    a_eff = a (1 - F_dr) / (1 - a F_dr),     F_dr = internalFresnelDiffuse(n) = 0.5967 at n = 1.5

*after* texture / record / pattern / vertex-colour, so a textured body gets it too. Entry
transmission stays where it was (the coat/body selection probability), so nothing is double counted.

Validated in a white furnace with the camera inside, `scraps/furn3_*.ftsl`, mode D on GPU, 3000 spp.
The rig proves itself first -- mirror **1.0004**, diffuse 1.0 -> **1.0003**, diffuse 0.5 -> **0.5001**
-- and `F` is measured from a black-body-under-coat case (**0.0401**), not fitted:

| coated body | rendered | predicted `F + (1-F) a_eff` | err |
|---|---:|---:|---:|
| white 1.0 | 0.9998 | 1.0000 | **-0.02 %** |
| grey 0.5 | 0.3159 | 0.3159 | **-0.02 %** |

With the coat's own lobe suppressed (`reflectance manual specular 0.0`) so only `a_eff` acts, a deep
red body deepens from **R/G 9.01 to 21.76** (2.41x) and darkens to **0.540x** -- the varnish effect
the old model could not produce. Full write-ups in `design.md` and `REFERENCE.md`.

**Two rigs had to be thrown away first, and both failures were the same mistake.** A furnace built
from `light area {}` is blind to specular (a MIRROR in it reads exactly 0.0000, because those
emitters have no hittable surface), and a furnace built as ONE emissive mesh is silently flipped to
emit outward and renders black -- a real renderer bug, now logged in `known-issues.md` with its
cause, its true rule (planarity, not closure) and its workaround. In both cases the tell was
available immediately: a control whose answer is known in advance read something impossible. Run the
controls before believing the result.

### A2. Coat absorption (tinted lacquer) -- **DONE (v0.324.0)**

New FTSL in the coat block: `absorb <spectrum>` (sigma_a in 1/m) + `depth <metres>`. Deliberately
NOT `film_thickness`, which is the nanometre wave-optics film -- eight orders of magnitude away, and
a silent unit trap if shared. Both keywords are required for absorption to do anything.

Beer-Lambert composes WITH A1's internal series instead of sitting beside it: light that fails to
escape crosses the absorbing layer twice more before its next try. The geometry is two
cosine-weighted mean secants, closed form in the index alone -- escape cone `2n^2(1 - cos tc)` =
1.1459, trapped leg `2/cos tc` = 2.6833 at n = 1.5 -- giving

    a_eff = a T^2 (1 - F_dr) / (1 - a F_dr T_rt)

which reduces exactly to A1 at sigma_a = 0. Both path lengths are folded into constants at
scene-build time, so shading pays two exps and only when the coat absorbs.

Validated in the white furnace, white body (so the rendered colour IS the coat), mode D, 3000 spp:

| coat | rendered | expected |
|---|---:|---|
| no `absorb`, no `depth` | 0.9999 | A1's clear 0.9998 |
| `absorb 2000`, no `depth` | 0.9999 | clear (both keywords required) |
| `depth 1e-4`, no `absorb` | 0.9999 | clear |
| `absorb 2000  depth 1e-4` | **0.3475** | **0.3476** analytic |
| `absorb rgb 400 2600 5200  depth 1e-4` | R 0.556 G 0.293 B 0.155 | amber lacquer |

CPU/GPU agree to 0.1 %.

**It also caught a defect I shipped in 0.323.0**: that version applied the coated-body albedo at
three of the four albedo sites, missing host `reflectSlot`, so a GLOSSY body under a coat differed
between backends for one version. All four now go through one `coatedAlbedoAt`; see known-issues.

**The approximation:** both absorption legs are directionally averaged. Snell compresses the whole
incident hemisphere into the escape cone, so the entry secant only ranges 1.0 .. 1.342 -- at most a
34 % swing in path length, against a tint that is the entire visual point, and with the coat's real
silhouette cue (R(theta) -> 1 at grazing) already exact.

### A3. Snell into the body -- **ANALYSED; NOT BUILT, and that is the finding**

A3 asked for Snell refraction into the body, plus a narrowing of the deferred-BSDF trigger from "a
directional body under the coat" to "a directional body under a ROUGH coat". **The narrowing is
retracted, the trigger has FIRED instead, and the full write-up with numbers is in
`known-issues.md`.** In short:

**1. An exact result reframes what A3 is.** Refract in, reflect off a body whose microfacet normal
IS the coat's normal, refract out: `sin t_out = n sin t_t = sin t_i`. The result is exactly the
mirror of the incoming direction -- **Snell in-and-out of a SMOOTH body is the identity.** So A3 is
not "the coat bends the light"; it is about the mismatch between the body's MICROFACET normal and
the coat's, which exists only for a ROUGH body. That is precisely the case where part of the lobe
lands past the critical angle and is trapped by TIR, which has no closed form. The refraction effect
and the TIR problem are co-extensive. Coat roughness never enters the argument, which is why
narrowing by it was wrong.

**2. The error is large and was measured** (`scraps/a3_snell.py`, brute-forcing the real layered
system against what ftrace renders today): up to **-37 % in directional albedo and -34 % in lobe
width** on a glossy body under a smooth coat. The bar the trigger set was "a few percent".

**3. Both controls pass**, so that number means what it says: with **n = 1.0** (no coat) truth and
model agree to **0.13 %**, and for a **Lambertian** body under a real coat they agree to **0.23 %**
-- which independently confirms A1's derivation, and whose 0.5-albedo prediction of 0.3159 is
exactly what ftrace *renders* in the furnace.

**4. No cheap fudge exists.** "Just scale the body's roughness" cannot work: the lobe-width error
**changes sign** with roughness (27-34 % too narrow at alpha 0.05, 7-10 % too wide at alpha 0.5),
because internal bounces spread the lobe while refraction compresses it.

**Why it is not built:** sampling could keep one lobe per vertex with a bounded internal loop, but
`f(wi,wo)` and `pdf(wo|wi)` have no closed form, and NEE needs the first at every shading point
while BDPT/VCM need both at every vertex. That is the stochastic-evaluation BSDF (Guo/Hasan/Zhao
2018) already named in the trigger -- real architecture, with an MIS decision to make up front.

This is exactly the outcome this item asked for in that case: *"If the Jacobian cannot be made
consistent within one-lobe-per-vertex, stop, and record it against the trigger rather than forcing
it -- that is what the trigger is for."*

**Next, if it is picked up:** build the brute-force reference IN THE RENDERER (mode R or A/B/C with
the coat traced explicitly, which needs no MIS pdf), in an enclosure, and confirm the Python numbers
above end to end before committing to the architecture.

## Standing constraints (so a fresh session does not have to be told)

- Bump `VERSION` in the same commit as any observable change; no rebuild → no bump. Commit freely,
  **never `git push`**.
- Every render passes `-window-min` (never bare `-window`) and launches with
  `dangerouslyDisableSandbox`. Stop a render with `ftrace -stop <pid>`, **never** `taskkill /F`.
- Scratch in `scraps/`, PNGs in `png/`; a flyby series gets its own `png/<setname>/`.
- `CLAUDE.md` is the user's and is not to be touched.
- The gallery_rain flyby is still pending and is the reason mode M matters: its command is in
  `known-issues.md` under the `-sunnee` entry.
