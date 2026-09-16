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

## A. The analytic coated-body model — TIR saturation, coat absorption, Snell

**Status: not started.**

Today `MatType::Layered` models the coat **on the way in and not on the way out**: the coat reflects
with probability R and otherwise the ray enters and a body lobe shades. There is no exit interface,
so no total internal reflection, and no absorption in the layer. Everything below is multiplicative
on the body's contribution, so it preserves **one lobe per vertex** — the property that lets a
layered material render in every mode on both backends from one definition.

### A1. Exit interface + internal multiple reflection (the TIR saturation)

The visible one. Light leaving the body meets the coat from inside; past the critical angle (~41.8°
at n = 1.5, which is most of a cosine-weighted hemisphere) it is thrown back down, scatters again
and retries. Summed in closed form that is the body's reflectance times

    T(theta_i) * T(theta_o) / (1 - albedo * F_dr)

with `F_dr` the internal diffuse Fresnel reflectance (~0.596 at n = 1.5), a constant of the index
alone — so the `1/(1 - a*F_dr)` part can be **baked into the body copy at scene-build time** and
costs nothing at render time. Entry transmission is already carried by the coat/body selection
probability; **the exit factor is what is missing.** Watch for double counting between the two.

This is what makes lacquered red read deeper than bare red, and varnished wood richer than raw.

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
