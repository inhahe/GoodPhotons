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

**Status: A1 DONE (v0.323.0). A2 and A3 not started.**

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

### A2. Coat absorption (tinted lacquer)

Beer-Lambert through the layer, in and out, with path length set by the refracted angles:
`exp(-sigma_a * d * (1/cos_t_i + 1/cos_t_o))`. Deepens toward the silhouette, which is the candy-paint
look. Needs FTSL: `coat { absorb <spectrum>  depth <metres> }`. **Do not reuse `film_thickness`** —
that is the nanometre wave-optics film for an iridescent coat, a different quantity; a new `depth`
keyword avoids a silent unit confusion.

### A3. Snell into the body (directional bodies)

For a **Lambertian** body this changes nothing beyond the transmission factors already in A1 — a
Lambertian scatters `albedo/pi` regardless of the direction the refracted ray arrives from. It
matters for a **directional** body: glossy or anisotropic under a coat, where refraction bends and
compresses the lobe.

**A correction to make to the trigger entry in `known-issues.md` while doing this:** that entry
lists "a directional body under the coat" as a trigger for the explicit multi-bounce BSDF. That is
too broad. Under a **smooth** coat, refraction is a deterministic bijection with an analytic
Jacobian (`dw_t/dw_o`), so the pdf transforms in closed form and MIS stays exact — a directional
body is tractable analytically. It is a **rough** coat that breaks the closed form. Narrow the
trigger to "a directional body under a ROUGH coat" once A3 is proven.

If the Jacobian cannot be made consistent within one-lobe-per-vertex, **stop, and record it against
the trigger rather than forcing it** — that is what the trigger is for.

### Validation for A (all of it)

- **White furnace, enclosed, camera INSIDE the box** (`scraps/furnace.ftsl`; an earlier version put
  the camera outside and a 1.0-albedo and a 0.5-albedo sphere rendered byte-identical — run the rig
  check every time). A white body under a lossless coat must read **exactly 1.000** against the
  walls at every roughness and every index. This is the energy-conservation test A1 lives or dies by.
- **Identity tests** (`scraps/ident_*.ftsl`, one sphere, fixed position, only the material swapped):
  a coat at `specular 0.0` must still equal the bare body; a coat at `specular 1.0` must still equal
  a plain glossy of the same roughness (both currently hold at 1.0001 / 0.9999 in mode D).
- **Cross-mode**: modes M, D, R agree on the coated sphere in the Cornell box to ~1 %.
- **Never measure reflection in an open scene.** See the retraction in `known-issues.md`.
- Docs in the same commit: `FTSL.md` (the `coat` block grammar), `REFERENCE.md`, `design.md`.

---

## Standing constraints (so a fresh session does not have to be told)

- Bump `VERSION` in the same commit as any observable change; no rebuild → no bump. Commit freely,
  **never `git push`**.
- Every render passes `-window-min` (never bare `-window`) and launches with
  `dangerouslyDisableSandbox`. Stop a render with `ftrace -stop <pid>`, **never** `taskkill /F`.
- Scratch in `scraps/`, PNGs in `png/`; a flyby series gets its own `png/<setname>/`.
- `CLAUDE.md` is the user's and is not to be touched.
- The gallery_rain flyby is still pending and is the reason mode M matters: its command is in
  `known-issues.md` under the `-sunnee` entry.
