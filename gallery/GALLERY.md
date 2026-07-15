# Feature Gallery — expected appearance

Every image is produced by `gallery/render_gallery.sh` (run from the project root).
Each entry below says **what the render should look like** so you can verify it by eye.
Unless noted, the scene is the unit-cube Cornell box: **red left wall, green right
wall, white floor/ceiling/back, an area light in the ceiling**, viewed head-on through
the open front. "Specular = black in mode B" is the expected forward-tracing (SDS)
limitation, not a bug.

---

## 1. Render / camera-measurement modes
All trace the *same* forward physics; only the camera measurement differs.

| File | Expected appearance |
|---|---|
| `01_mode_B_pinhole.png` | The reference Cornell image: sharp, everything in focus. A **dispersive glass sphere** sits on the floor — it reads **black** (specular can't connect to a pinhole; expected) but casts a bright **caustic** on the floor beneath it. Red/green colour bleed onto the white walls near the corners. |
| `02_mode_A_lens.png` | Same box through a **physical thin-lens camera** with a wide aperture. The box centre (sphere) is sharp; the near floor edge and back wall are **softly defocused** (real depth of field). Grainier than B (lens next-event splat). |
| `03_mode_C_catch.png` | The `materials` box caught through a finite aperture (brute-force forward catch). **Dim and noisy** — mode C is catch-starved by design; use it as the DOF oracle, not a beauty shot. Framing/blur should match a wide-aperture mode A. |
| `04_mode_R_reference.png` | Clean **backward-path-traced** Cornell (all-diffuse variant: the sphere is diffuse white, no glass). This is the low-noise ground truth the forward tracer is validated against. |
| `05_mode_P_composite.png` | The `materials` box with the **specular spheres now visible**: a mirror sphere reflecting the walls, a rough glossy sphere, and a half-mirror sphere. Composite = forward diffuse + backward camera-side for specular. |
| `06_mode_D_bdpt.png` | The `materials` box rendered by **BDPT** in one pass — specular spheres visible *and* any caustics, on the absolute-radiance scale. Should look like `05` but from a single unbiased estimator (cleaner seams). |
| *(mode V)* | No image — prints `PASS`/`FAIL` and a best-fit forward-vs-backward residual to the console. Look for **PASS** and a low bulk RMSE. |

## 2. Camera variables & presets

| File | Expected appearance |
|---|---|
| `10_dof_shallow.png` | Wide aperture (0.30): **strong** background/foreground blur, only the focus plane (sphere) crisp. |
| `11_dof_deep.png` | Narrow aperture (0.05): nearly **everything sharp** — approaches the pinhole limit. Compare directly against `10` to see aperture control. |
| `12_fisheye_*.png` | The fisheye scene defines several cameras, so this writes a set: `_rect` (rectilinear control) plus `_fish`, `_zoom2`, `_vertigo0..2`. The fisheye frames bulge the scene into a **disc with curved wall lines** and black corners outside the image circle; `_rect` is the straight-line control to compare against. CPU-only. |
| `13_dolly_dolly0..4.png` | A `camera_path` move — writes **5 numbered frames**. Across frames the camera **pulls straight back**, the box shrinking while staying centred. |
| `14_expo_iso100.png` / `14_expo_iso200.png` | Exposure/ISO+shutter demo — same framing, differ only in **brightness**: ISO 200 is exactly 2× ISO 100 in linear light. (`14_expo_fstop.png` is a finite-aperture DOF frame that is catch-starved → near-black/speckly; ignore it as a beauty shot.) |
| `15_twocam_hero.png` / `15_twocam_side.png` | Multi-camera scene: **one image per named camera** — a hero view and a side view of the same box. |

## 3. Surfaces / materials

| File | Expected appearance |
|---|---|
| `20_materials_B.png` | Mirror, glossy and half-mirror spheres all appear **black** (specular can't connect to a pinhole), but the **walls are correctly lit** by light that bounced off them. Demonstrates the SDS limitation on purpose. |
| `21_glass_caustic.png` | Default Cornell with the **dispersive SF10 glass sphere** (identical to `01`): the sphere is black, but note the bright, faintly **rainbow-fringed caustic** it focuses onto the floor. Forward tracing renders caustics well — this is the showcase. |
| `22_prism.png` | A collimated white beam enters a **glass prism** (a black specular triangle in mode B). The dispersion shows as small **coloured spectral spots** scattered on the left wall and floor (blue deflected most, red least) — the caustic is subtle at this photon count, not a bold painted streak. |
| `23_iridescent.png` | A **thin-film sphere** dusted with **interference colours** (shifting hues across the curvature) — soap-bubble-like iridescence. Rendered in mode P, so it's speckly/noisy (specular seen via the camera-side path). |
| `24_material_presets.png` | A dim box with **gold, copper, silver, diamond, water, morpho, oil-slick** preset spheres — coloured metals and clear/iridescent dielectrics reflecting the leaf-green and concrete walls. Specular + a small light = fairly dark and noisy; bump `-n` to clean up. |
| `25_multilayer.png` | **Structural / multilayer** colour (morpho-blue / beetle-green style) — a saturated angle-dependent hue from a dielectric quarter-wave stack. Speckly (mode P specular). |
| `26_mixmat.png` | A **mixed material** (stochastic blend of two child materials) — reads as an intermediate of its two components. |
| `27_fluoro.png` | A **fluorescent sphere** that glows brighter/shifted (absorbs shorter wavelengths, re-emits longer) — appears to **emit** a colour under the white light. CPU-only. |
| `28_textured.png` | A **checker-textured quad**: blue band at the top (v≈1), yellow at the left (u≈0), correct orientation and spectral colour. |
| `29_uvmesh.png` | The same checker mapped onto a **mesh via its UVs**. |

## 4. Diffraction

| File | Expected appearance |
|---|---|
| `30_grating_on.png` | A collimated beam hits a **reflective diffraction grating** (black patch in the centre): **symmetric ±1-order rainbows fanned left and right** onto the side walls — the standout diffraction shot. |
| `31_grating_off.png` | Same geometry with diffraction **disabled**: the grating is now a plain **specular** mirror, so in mode B it reflects the beam back out of the pinhole's reach and the image is **essentially all black** (no orders at all). The stark contrast with `30` *is* the diffraction feature — every coloured order in `30` exists only because diffraction is on. |

## 5. Light shapes

| File | Expected appearance |
|---|---|
| `40_light_area.png` | Baseline **quad area light** (D65) — soft shadows, even fill. |
| `41_light_sphere.png` | A **spherical** emitter — a round glowing light with soft round-source shadows. |
| `42_light_spot.png` | A ceiling **spotlight** aimed down: a **bright circular pool on the floor with a soft (smoothstep) penumbra edge**, the box corners falling into shadow. |
| `43_light_env.png` | **Two spheres on a ground plane** lit by a **uniform sky** (constant environment) — even ambient fill from all directions, soft contact shadows, no single hard shadow. Backward render for a clean result. |
| `44_light_envmap.png` | Same two-sphere scene lit by an **HDRI equirectangular map** (`sky.pfm`) — a warm/directional sky tint (left sphere warmer, right neutral) instead of flat grey ambient. |
| `45_light_cylinder.png` | A **cylindrical** area emitter — an elongated soft light source. |
| `46_light_two.png` | **Two** distinct lights — overlapping soft shadows / two-toned illumination. |

## 6. Light SPD presets
Same Cornell box; only the illuminant SPD changes, so judge by the **white balance /
colour cast** of the (nominally white) walls.

| File | Expected cast |
|---|---|
| `50_spd_sun.png` | Neutral daylight, very slightly warm. |
| `50_spd_daylight_d65.png` | Neutral **D65** white (reference). |
| `50_spd_incandescent_A.png` | Strongly **warm/orange** (CIE A, ~2856 K). |
| `50_spd_led_neutral.png` | Neutral-white LED, faint phosphor tint. |
| `50_spd_led_warm.png` | **Warm** white LED. |
| `50_spd_fluorescent_cfl.png` | Slightly **greenish** fluorescent. |
| `50_spd_f2_cool_white.png` | CIE F2 cool-white — warm-ish with a green tinge. |
| `50_spd_f7_daylight_fl.png` | CIE F7 — the **coolest / most daylight-like** fluorescent. |
| `50_spd_f11_triphosphor.png` | CIE F11 triphosphor — warm-white, spiky. |
| `50_spd_sodium_hps.png` | **Amber/orange** high-pressure sodium (street-lamp). |
| `50_spd_sodium_lps.png` | Nearly **monochromatic orange** low-pressure sodium (worst colour rendering). |
| `50_spd_mercury.png` | **Cold blue-green** mercury vapour with a magenta edge. |
| `50_spd_metal_halide.png` | Cool-white metal-halide, fairly neutral. |
| `50_spd_blackbody_3200K.png` | Warm **tungsten** (3200 K). |
| `50_spd_blackbody_6500K.png` | Neutral **6500 K** blackbody. |
| `50_spd_led_4000K.png` | Neutral-warm **4000 K** phosphor LED. |

## 7. Fog / participating media

| File | Expected appearance |
|---|---|
| `60_fog_iso.png` | Isotropic **haze** filling the box — glow around the light, reduced contrast, visible light shafts, softened shadows. |
| `61_fog_forward.png` | Forward-scattering fog (g=0.7): the haze **brightens strongly around/behind the light** (light concentrates in the forward direction). |
| `62_fog_back.png` | Back-scattering fog (g=−0.5): haze brightens **toward the camera / around the source** differently — more even veiling, less forward glow. |
| `63_fog_rayleigh.png` | **Rayleigh** fog: short wavelengths scatter far more, so the direct/transmitted light **reddens** and the whole box takes on a warm **sunset/amber** cast (the blue has been scattered out of the beam). |

---

### Notes
- GPU (megakernel) is used automatically where supported. CPU-only renders (fisheye,
  textures, fluorescence, backward/composite/BDPT modes) are marked above and run
  slower.
- Add `-wavefront` to any forward GPU render to exercise the streaming backend; the
  image should match its megakernel twin to within Monte-Carlo noise.
- Photon counts are chosen to be "clean enough to judge." Bump `-n` / `-spp` for
  publication-grade smoothness.
