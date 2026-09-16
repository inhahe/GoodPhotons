# TODO — in flight

Live working plan. Each item says what, why, how it gets validated, and where it stands, so an
interruption costs the work in progress and not the plan. Finished items move to `known-issues.md`
(with their measurements) and come off this list; `design.md` gets the architecture.

Ordering note: **B is first even though A was asked for first.** B is a spectral-handling question
in mode M, and mode M is one of the backends A has to be validated in — a 12 % error in how a
material's spectrum is handled would contaminate every number A produces. Measure the ruler before
measuring with it.

---

## B. Mode M mis-colours coloured speculars (root cause found; fix designed, NOT built)

**Status: diagnosed, written up in `known-issues.md`. The fix is the remaining work.**

Started as "a JH white renders 12 % dark in mode M"; the real fault is bigger and the first framing
was wrong. Mode M's camera walk is monochromatic at the camera's wavelength while the photon map is
polychromatic, and a specular/glossy bounce multiplies a scalar reflectance taken at the CAMERA's
wavelength onto a photon sum spanning all of them. Exact for a flat spectrum; wrong in either
direction for a coloured one — measured M/D of 0.880 (JH white), 1.142 (JH red), 1.005 (flat).
`gallery_rain`'s gold gyroid and chrome ring are exactly this case, so the pending flyby renders
their colour wrong relative to modes D and R.

Ruled out on the way, both by measurement: the hero-wavelength stratification added in 0.314.0
(the ratio is the same at spp 1, where it is disabled, as at spp 512), and clamping (the sign
flips). The diffuse path is correct and already does the right thing per photon
(`photonmap_render.h:1400`), which is the model to copy.

**The fix.** Carry `thrS[]` over the spectral grid beside the scalar `thr`; multiply each bounce's
reflectance into it per grid wavelength; weight each photon in the density estimate by
`thrS[bin(ph.lambda)] / thrS[bin(lambda_c)]`. The ratio is exactly 1 for flat spectra, so flat
scenes stay bit-identical — that is the regression test as well as the safety property.

1. `photonmap_render.h` — the CPU gather: both estimate sites, and every bounce that multiplies
   `thr` by a reflectance (Glossy, Mirror, HalfMirror, ThinFilm, Grating, the layered coat).
2. `render_cuda.cu` `dPhotonGather` — same structure, same bug, and the backend a flyby uses.
3. Validate: the four Cornell rows -> ~1.00; diffuse control unmoved; `_beams_ms` invariant
   unmoved; a flat-spectrum scene bit-identical.

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
