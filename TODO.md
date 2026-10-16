# TODO — in flight

Live working plan: only what is still to do. Each item says what, why, and where it stands, so an
interruption costs the work in progress and not the plan. **Finished work moves to
`TODO-completed.md`** (verbatim, with its reasoning); its measurements stay in `known-issues.md` and
its architecture in `design.md`.

**Read `TODO-completed.md`'s "Recently finished" list before starting anything.** On 2026-09-17 I
reported VOLCACHE's deposit split, GPU-VARIANCE and GPU-BEAM-TAIL as outstanding. **All three were
already closed** — I had carried them forward from a stale task list without re-reading their
entries. Checking status costs one `grep`; chasing finished work costs a session.

---

## 1. Alice's hair — what is left (the history is `TODO-completed.md` §0)

The groom tools are built (`include`, curves of curves, guided `fur`, the groom GUI's seven phases),
Alice2 carries her v3 hair in `gallery_rain`, and her hair renders on the GPU in mode M. Open:

- **The look is the user's call**: lock size, fringe/part, colour calibrated under the hall's light
  rather than the harness suns, `beta_m`/`beta_n`. Pending feedback.
- **Flyby cost**: ~44 h of GPU time for 1147 frames with hair (5.75 s/spp at 20 M photons). Levers,
  each to be measured paired: `segments 1 points 5` and fewer strands; an LOD with a CAMERA-SIDE
  stochastic fade (never a per-frame `count` ramp, which rebuilds the map per frame) — only once
  the strands match the molded base at flyby scale, or the fade itself is a visible colour step
  (GPU strands luminance 81 / hue 71 deg against the base's 86 / 54).
- **A quiet re-measure of the per-lobe fold's cost** (0.336.0's +20-50 % was taken on a busy machine).
- **Groom tool polish**: a live fur regenerate without a reload (split the parse half of `addFur` out
  so a scratch Builder can run it); named-sphere `bald "name"` zones drawn; the `on` mesh chosen from
  a list; per-point radius dragging; `-groom` on a bare mesh (auto-scalp); a depth pre-pass so a
  closed see-through shell does not blend twice; snapping a whole strand inside a part; moving a box
  selection together. (Its untested drags are GROOM-DRAG-UNTESTED, in §4.)

## 2. The `gallery_rain` 960x540 flyby — the deliverable, waiting on the user

1147 frames, `camera_curve "fly"`, mode M at the final recipe (`-mode M -device gpu -n 20000000
-spp 24 -r 960 540 -denoise -fireflies 3 -denoise-levels 2`); four scattered frames render clean
(`png/flyframes2/sheet.png`). Decided 2026-10-16: the flight path stays as it is, Alice keeps her
`rotate 0 20 0` turn, and she wears the sparkle dress (`import_metal mix`, 0.382.0).
**Still the user's decisions:** launching it (it owns the GPU for about two days, and another GPU
job beside a long mode-M gather is what killed an earlier run), and whether to LOD the hair (§1).
The command must now name the path (`-camera fly`, since 0.383.0 a plain render is the still).
Before launching, re-read the two run hazards in `known-issues.md`: the
machine's COMMIT limit (the GPU path now holds one frame of host film, so check `commit available`
rather than assume), and a gather that dies with `unspecified launch failure` under concurrent GPU
load — ftrace exits 1 with a `[cuda] ... kernel failed` line, so the chain must check exit codes
and re-run `-frames N end`.

## 3. The A3 LOBE-WIDTH claim is untested — the rig's two sides are not geometrically identical

**The lobe-width claim is untested — its own control fails at +10.2 % with no coat present, so those
numbers can't be cited. Measuring it needs a rig whose two sides are geometrically identical (a
direction sweep against a BSDF evaluation, not two renders of different scenes).**

The detail, from `tools/a3_reference.py`. Each A3 claim is gated by its own `coat ior 1.0` control —
with no index contrast there is no coat, so both constructions must agree. The energy claim passes
that control at **−0.2 %** (explicit 0.6210 vs analytic 0.6197) and is established. The lobe claim
fails it at **+10.2 %** (r50 0.322 vs 0.355), which is *larger than several of the differences the
sweep attributes to the coat* (−5.5 % at roughness 0.02). The cause is structural: the explicit
scene carries a second sphere that the analytic scene does not, so `r50` measures that geometric
difference as much as the layered model.

Two renders of different scenes can never settle this. The replacement rig evaluates **one**
geometry: sweep the incident direction explicitly and compare a brute-force layered simulation with
the analytic BSDF at the same (wi, wo) pairs, with no scene, no camera and no fireflies.

*(Keeping the failing control is deliberate — deleting it would restore the appearance of a
measurement. A test that says "this instrument cannot answer this question" is the useful output.)*

## 4. Everything open in `known-issues.md` that should be fixed

Listed 2026-10-16 from an entry-by-entry read of `known-issues.md` (its 2026-09-10 audit is the last
time every entry was checked against the code). **Not started: the user asked for the list first.**
Each line names its entry by ID or by a phrase of its heading, so `grep` finds it; the entry has the
detail. Mark a line `[x]` with the version when it is fixed, and mark the entry DONE in the same
commit.

### 4.1 Decisions the user owes (each blocks its fix)

- [ ] **TILTED-LIGHT-EMIT** — a tilted quad light is measured by its patch on the camera side and by
  its authored normal on the light side. Proposal: snap a tilted normal to its plane's side at load.
  Measured on `mirror_selfie` (mode R, `png/tilt_cmp/compare.png`): the snap moves 3.6 % (the 22°
  fill) and 10.1 % (the 37° wall-washes) of a light's flux from grazing directions behind the panel
  to grazing ones in front; the frame's mean moves 0.03 %, no region more than 3 %. Explained to the
  user 2026-10-16; waiting on the decision.
- [x] **`scenes/gallery_settled.ftsl` has diverged from `scenes/gallery.ftsl`** — closed by the
  user's answer (2026-10-16): the scene in use is `gallery_rain.ftsl`; no back-port.
- [x] **`textures/marble_dumbbell.png` derives from a WATERMARKED stock preview** — and so did four
  more caps: sources 1 (Vecteezy) and 2 (pngtree) carry tiled watermarks, legible in the shipped
  textures. All five re-quarried from the clean sheets 7 and 8 (2026-10-16, `make_marble_caps.py`
  now refuses the three).
- [x] **Rendering a scene without `-camera` also renders its 600-frame flyby** — left to judgement;
  stills by default, paths by name (0.383.0).

### 4.2 Wrong, missing or silent output

- [ ] **SPECGATHER-BIN-RESOLUTION** — mode M's 24-bin carrier misreads a spectral edge or kink inside
  a bin; the 555-585 nm ramp's green excess is unexplained.
- [ ] **SPPM-SPECTRAL** — mode S applies a camera path's spectral factors at the camera's wavelength
  to every photon.
- [ ] **BIRTH-EMPTY-STACK** — a light walk born inside a dielectric does not know it is inside.
- [ ] **GRIN-RADIANCE-SCALE** — a camera path ending inside a GRIN region misses the n² radiance
  scale.
- [ ] **STUBBLE-DEVICE-UNDERSIDE** — mode W on a stubble coat: device and host 9.6 % apart in its
  dark underside (to investigate).
- [ ] **GATHERAREA-TRANSPARENT's remaining limits** — an opaque, photon-holding surface hovering
  within the radius above another; Mix compounds.
- [ ] **EMISSIVE-VOLUME-SILENT, what is left** — GPU mode D falls back to the CPU on a fire; modes J
  and U refuse a fire (their light passes need fire-born subpaths, J's beams their own births); mode
  P refused until its composite is checked with a fire.
- [ ] **SPECTRAL-BUNDLE, what is left** — the other modes' media gates (the device's mode D among
  them); a fiber with a dispersive index (`hair.h` takes one `eta`).
- [ ] **ALICE2-CHEEK-DOT** — a pinprick dark dot on the Alice2 face in front of each ear.
- [ ] **GLTF-ROUGHNESS-MAP** — a glTF roughness is fed to the Phong lobe raw, so imports render
  rougher than glTF means.
- [ ] **"a quad's emission is invisible from the side its winding faces away from"** — re-check
  against AREA-LIGHT-SIDE (0.377.2), which fixed a neighbouring case.
- [ ] **"`-dual-scatter` drops the coat's indirect illumination"** (its part 2).
- [ ] **"`curv` reads 0 on an ISOSURFACE"** — a diagnostic names it now; the Hessian is the fix.
- [ ] **"`curv` reads 0 in the PREVIEW rasterizer"** — mechanical.
- [ ] **"`cavity` reads 0 in BOTH preview rasterizers"** — not mechanical.
- [ ] **"mode `W` is NOT noise-free at `-spp 1` when the scene has a medium"** — the fog term is
  still a Monte-Carlo free flight.
- [ ] **"freshly CMake-configured build dirs produce a GPU-silently-dead ftrace.exe"** — black
  renders with no CUDA error; affects every fresh clone.
- [ ] **"Headless-spawned `-window` render on gallery.ftsl hangs with no output"** — to investigate
  (compare CLAUDE.md's note on the sandbox's non-interactive window station).
- [ ] **"`scenes/gallery_settled.ftsl` OOMs — but ONLY when the 600-frame flyby is in the camera
  selection"** — re-test: the GPU path now renders a frame at a time; the CPU path may not.
- [ ] **"Forward modes render ~5% brighter than the backward reference (`R`)"** — from 2026-07-12;
  re-measure first, then reconcile the constant.
- [ ] **"rainbow phase — `SpecVtx::term` ... is HG-only"**.
- [ ] **"`phase rainbow` — the 2048-bin uniform-in-mu table under-resolves large droplets"**, and
  monodisperse supernumeraries read as a white arc.
- [ ] **"forward mode B barely converges the water-droplet rainbow"**.
- [ ] **"interactive raster fly-viewer can peg all cores / grow RAM when orphaned"**.
- [ ] **"native `-viewer` Meshes tab ignores a skin's `clamp`/`mirror` wrap mode"**.
- [ ] **"Mode M renders Alice ... far from mode D, differently on each device"** — 20 M photons
  closed most of it (the GPU still ~9 % dark): re-measure, then fix or close.
- [ ] **"Residual: hue lost at true underflow"** — see-through on a very deep stack (minor).
- [ ] **"a `curve`'s azimuthal `v` is a PER-SEGMENT frame, not parallel-transported"**.
- [ ] **"an `sdf` over a self-overlapping mesh is exact outside but not inside"**.
- [ ] **"`settle.drop()` falls back to the surface's TOP PLANE when the footprint 'misses'"** — find
  why the down-rays miss.
- [ ] **"an `include`d file's own asset paths resolve through the ROOT scene"**.
- [ ] **"`tools/_gemsweep.py` — `spread` is not resolution-stable"**.
- [ ] **"`tools/_capchroma.py` scores marble VEINS as a caustic"** — refuse a textured cap.
- [ ] **"`python -m loom.viewer` prints a runpy double-import RuntimeWarning"** (cosmetic).

### 4.3 Estimator approximations and accuracy debt

- [ ] **BDPT-MIS-TR** — mode D's MIS weights take every connection edge's transmittance as 1 in a
  medium.
- [ ] **UPBP-W** — three approximations in mode J's merge weight (variance only).
- [ ] **"BDPT connection edges through colored glass are not absorption-weighted"**, and mode U's
  connections share it (**"VCM (mode `U`): merges use a spectral XYZ estimate"**).
- [ ] **"VCM vertex-merge needs a gather-side shading-normal reweight that M/S don't"** — the
  asymmetry is not understood.
- [ ] **"`-radcache` is opt-in because it is *approximate*"** — residual on caustics/specular; not
  deterministic across frames.
- [ ] **"mode-R GPU GRIN has a small bent-region float-vs-double residual"**.
- [ ] **"GRIN tech debt (Phase-1 semantics carried forward)"** — only primary rays bend, and the
  rest of that entry's list.
- [ ] **"mode `W` picks a dielectric's dominant branch, it does not fork"**.
- [ ] **"the tone map hard-clips with no shoulder"** — a real defect the anchor studies assumed away.
- [ ] **"the sigmoid upsampling model itself is ~0.019 off at pure white"**.
- [ ] **"`fw` is 0 at every secondary bounce"** — `fnoise` does not band-limit in a reflection.
- [ ] **"a non-uniformly scaled instance's `curv` is approximated by |det|^(1/3)"**.
- [ ] **"GPU photon mapping isn't run-to-run bit-stable"** (~3 % under `-beams`), and **"GPU raster
  see-through output isn't run-to-run bit-stable"**.
- [ ] **"a `hair` coat is very noisy in the stochastic modes"** — path variance through the mass;
  the denoiser is today's answer.
- [ ] **"`type hair` cannot be gathered on in modes M / S"** — the photon record has no incident
  direction.
- [ ] **Mode D's heavy noise tail** — per-frame noise falls as ~`spp^-0.15`; a 4 %-energy,
  1669x-peaked connection residual in shared BDPT machinery. Characterised, no fix proposed yet.

### 4.4 Tech debt

- [ ] **GPU-STACK-DRIFT** — every kernel's stack reservation moves when unrelated device code changes.
- [ ] **SLAB-DDA-COPIES** — the box clip and the grid walk are written out several times.
- [ ] **GMIS-NULL-BY-HAND** — every caller keeps the glossy MIS record across a pass-through by hand.
- [ ] **GROOM-SILENT-OOM** — the groom tool can exit without a word when memory is short.
- [ ] **GROOM-DRAG-UNTESTED** — the groom tool's drags have no automated test.
- [ ] **"mode `U` (VCM) still uses a dense cell grid with the same `(int)(nCells + 1)` overflow"** —
  check what 0.381.7's cell cap left of it.
- [ ] **"The 12 render modes are 5 estimators wearing 12 letters"** — thin aliases over five entry
  points, and close the U/J gap that costs image quality.
- [ ] **"Checked-in scenes cite ~22 derivation scripts that live in gitignored `scraps/`"** — a fresh
  clone has none of them (the output cleaner keeps them locally, nothing more).
- [ ] **"a 'render every scene at a uniform budget' sweep cannot distinguish correct from broken"**
  — needs stored per-scene references.
- [ ] **"`isomesh.h` caps a *rotated* container against its world AABB"**.
- [ ] **"`fieldLeafSDF` computes `c.r` ... unconditionally"** (2-4 % of `patternEval`).
- [ ] **"loom's `Isosurface` cannot emit a CSG field tree"**.
- [ ] **"`grid:<name>(…)` / `scatter:<name>(…)` are *surface-pattern* samplers only"**.
- [ ] **"`fur` cannot grow on a `mesh_instance`"**, and a huge groom peaks at 2x its final memory.
- [ ] **"`CurveSeg` is 80 bytes"** — a real groom is ~1.3 GB of segments.
- [ ] **"the loom viewer's prebake cache costs ~6.4 MB per frame"**.
- [ ] **"Fast RGB backward (`-rgb`) omits media + textured/record albedo"**.
- [ ] **"GPU BDPT (mode D) can't texture"**.
- [ ] **"GPU preview rasterizer — feature parity with the CPU rasterizer"** — what that entry lists
  as left.
- [ ] **"loom preview server (`-serve`) is resident-process only"**.
- [ ] **"HIP alias block deliberately omits `__shfl_sync`"** (HIP is untested anyway).

### 4.5 Performance

- [ ] **FIRE-NEE-STRATA-COST** — one fire light sample costs up to 32 traversals.
- [ ] **Fire births** (EMISSIVE-VOLUME-SILENT's efficiency debt) — drawn uniformly over the box;
  drawing them from mode R's emission grid (`EmissiveVolume::cellW`) makes beta nearly constant.
- [ ] **"mode `D` has no resident multi-camera session"** — a flyby re-uploads the scene per frame.
- [ ] **"the beam-map build is host-only and single-threaded"**.
- [ ] **"a mesh-bound medium's voxelisation is 86 % of `gallery_rain`'s load"** — its triangle pass
  is serial.
- [ ] **"`-mode W` inherits mode R's `-spp 256` default"**, though deterministic at 1.
- [ ] **"a GPU render can be starved to a standstill by ANOTHER process"** (and its RECURRENCE entry,
  which has the detector) — no way to yield today.
- [ ] **"on `gallery_rain` the aimed caustic pass cannot concentrate"**.
- [ ] **"the preview re-marshals every texel of every texture, every frame"**.
- [ ] **"gallery GPU configs are now dominated by the ~8.4 s fixed CPU-side setup"**.
- [ ] **"preview rasterizer much slower than an equivalent WebGL renderer"** — profile first.
- [ ] **"`-export-mesh` QEM decimation is pathologically slow"** on huge or self-intersecting meshes.
- [ ] **"scene loading is down 5×, but the graph walk ... is what's left"** — only if load shows up
  again.

### 4.6 Missing features and scoped follow-ups

- [ ] **An emissive `quad` / isosurface / CSG solid glows but does not illuminate** (two entries:
  "KNOWN LIMITATION" and "LIMITATION (2026-08-02)") — register emitters for them.
- [ ] **"the live window cannot repaint mid-image"**.
- [ ] **"cache ftrace's per-scene preprocessing before rasterizing"** (feature request).
- [ ] **"option for curve-editor curve to be occluded by geometry"** (feature request).
- [ ] **"FBX import (C8) consumes geometry only"**.
- [ ] **"glTF/GLB loader is a static-geometry subset"** — what that entry lists as left.
- [ ] **"`.nvdb` / `.vdb` volume import: dense bake, float-only"**.
- [ ] **"Fisheye/panoramic lenses"** — unsupported by mode D and the finite-lens modes A/C.
- [ ] **"Absolute-EV film sensitivity, non-square films, shared multi-camera pass"** — its open items.
- [ ] **"Texturing is base-color only"** — only the `ior` map is left.
- [ ] **"Light shapes"** — spot penumbra CDF sampling is left.
- [ ] **"POV-Ray pattern/pigment/spline internal functions not ported"** — the remainder.
- [ ] **"the ~11 % of cloud beams that a dispersive event disqualifies"** — check first: 0.257.0's
  per-member bundle weights may be this (then it is 5.7).

### 4.7 Bookkeeping: entries whose heading still says OPEN for finished work

Verify each against the code, then mark it DONE (or retitle it) in `known-issues.md`.

- [ ] Both **`camera_curve` frame-range** entries in "Open bugs" — `-frames A B` shipped in 0.337.0.
- [ ] **"`-savemap` / `-loadmap` are silently ignored on the CPU"** — `photonmap_io.h` serves both.
- [ ] **"an emissive mesh that is not PLANAR is silently re-oriented OUTWARD"** — its own body says
  FIXED (0.338.0, `emit_orient`).
- [ ] **VOLCACHE** (2026-09-06) — the deposit split shipped in 0.300.0.
- [ ] **"a mesh area light is sampled UNIFORMLY BY AREA"** — the penalty no longer reproduces; the
  emitter-level tree shipped in 0.187.0.
- [ ] **BVH-BUILD** — its heading says not worth parallelising; its body says done in 0.292.0.
- [ ] **UPBP-CONV** (claims overturned), **FURDIM** (dissolved), and the **PLANNED UPBP** entry (first
  half done in 0.214.0 — check what is left of the second).
- [ ] **"the `gallery_rain` 600-frame mode-M flyby cannot start — the machine's COMMIT limit"** —
  re-test on the frame-at-a-time GPU path; it may be closed.

### 4.8 Not listed, on purpose

Notes and by-design entries (the GitHub `<video>` NOTE, the p99 anchor BY-DESIGN, `SetWindowTextW`,
`-gi-clamp`, the CIE side-table "don't retry", the timed-out-shell NOTE), work deliberately not
done (`PatOp::MatMulAdd`, GPU marching cubes, sweeps as an FTSL primitive), the environment (Nsight's
admin-locked counters, the one-off `build.bat` failure, the nvlddmkm teardown BSOD — a driver bug,
mitigated), and data whose entries call it acceptable (the F-series SPDs, the material presets).

## 5. The paired-timing rule into `CLAUDE.md`

"Time paired within a repetition, warm-up discarded" is a standing measurement rule that lives only
in session context and in this file's last section. It belongs in `CLAUDE.md`. (The old block —
the user's uncommitted changes there — no longer applies: `CLAUDE.md` is local-only and git-ignored.
It is still the user's file; ask first.)

---

## Standing constraints (so a fresh session does not have to be told)

- Bump `VERSION` in the same commit as any observable change; no rebuild → no bump. Commit freely,
  **never `git push`**.
- Every render passes `-window-min` (never bare `-window`) and launches with
  `dangerouslyDisableSandbox`. Stop a render with `ftrace -stop <pid>`, **never** `taskkill /F`.
  Never blanket-kill a shared runtime by image name (python/node/dotnet/java) — only the exact PID.
- Scratch in `scraps/`, PNGs in `png/`; a flyby series gets its own `png/<setname>/`. What outlives
  its use there is pruned by `tools/clean_outputs.py` after every build (REFERENCE.md, *How long
  output is kept*).
- **A rig that is cited as evidence does not belong in git-ignored `scraps/`** — promote it to
  `tools/` and rewrite the references in the same commit, or the table is not re-derivable from a
  clean clone.
- `CLAUDE.md` is the user's (and local-only): change it only when asked.

## Measurement discipline (each of these cost something to learn)

- **Run the controls first, and let them veto the result.** A rig that cannot see the effect
  produces confident numbers that are pure artefact: a furnace built from `light area {}` renders a
  MIRROR as exactly 0.0000, and I nearly reported that blindness as a 4 % energy bug in a material.
  `tools/furnace_rig.py` prints its three known answers before it prints anything else.
- **Never measure reflection in an open scene** — use an enclosure. This is what made the
  "mode M renders glossy 2.7x darker" claim wrong, and it had the same tell (a mirror reading 0).
- **Run the baseline control before attributing a difference**, especially before concluding that
  something new is slow (see `TODO-completed.md` §6).
- Score per-ROI, never whole-frame; separate variance from systematic; hold confounds fixed by
  construction; time paired within a repetition with the warm-up discarded.
- When a quantity must be applied at N sites, **N > 1 is the bug** — build the funnel. The 0.323.0
  host/device slip was a four-site change made at three sites.
