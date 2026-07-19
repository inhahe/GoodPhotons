# ftrace — a spectral forward + backward photon raytracer

`ftrace` is a physically-based, **spectral** light-transport renderer written in
C++20. Unlike an RGB path tracer, it transports light one **wavelength** at a
time, so dispersion, chromatic aberration, thin-film iridescence, fluorescence
and diffraction all fall out of the physics for free — there is no special-casing
of colour. It has both a **forward** (light-tracing / photon) core and an
independent **backward** (path-traced) reference, a CUDA GPU backend for the
forward pinhole mode, and a small scene-description language (**FTSL**).

---

## Highlights

- **Spectral transport** — single-wavelength photons over a configurable band
  (e.g. `spectral 360 830 1`); per-wavelength refraction gives dispersion and
  chromatic aberration with no extra code. Carrying **one wavelength per photon**
  — rather than a bundled RGB triple or wavelength-multiplexed packet, as most
  forward light tracers do — means dispersive **caustics** (light focused through
  a prism, a lens, or a glass of water) split into true spectral colour instead of
  smearing an averaged RGB, for more physically realistic focusing.
- **Forward *and* backward** engines that validate each other (mode `V` reports
  the residual between them).
- **Realistic cameras** — from a simple pinhole to a **physical multi-element
  lens** (real glass prescription: depth of field, vignetting, spherical &
  chromatic aberration all emergent), plus fisheye/panoramic projections.
- **Rich material set** — dielectrics with real glass dispersion, metals from
  measured data, rough microfacet, thin-film & multilayer interference,
  diffraction gratings, fluorescence, and stochastic mixes.
- **Wave-optical effects** — thin-film Airy interference, Abelès multilayer
  stacks, and reflective diffraction gratings.
- **Participating media** — one or many coexisting (superposed) fog regions with
  Henyey–Greenstein or Rayleigh scattering; box / sphere / **named-object** bounds
  (fog shaped to a sphere, isosurface field, or mesh AABB) and heterogeneous
  **density fields** — either formula-defined blobs with soft edges *or* imported
  **`.nvdb` (NanoVDB) volumes** (`density vdb:<file>`) — via unbiased delta/ratio
  tracking on the forward modes (CPU and GPU).
- **Gradient-index (GRIN) media** — a bounded region carrying an `ior "n(x,y,z)"`
  field bends rays continuously along the Eikonal ray equation (mirages, gradient
  lenses, hot-air shimmer) via a shared symplectic marcher. Works on the forward
  light tracer (modes `A`/`B`/`C`, CPU **and GPU**) and the backward reference
  (mode `R`, CPU); BDPT (`D`) refuses GRIN scenes (its straight-line connection
  geometry would be biased — use `A`/`B`/`C` or `R`).
- **CUDA GPU backend** for the forward pinhole splat (mode `B`), the backward and
  BDPT references (`R`/`D`), and the **view-independent photon map** (`M`, shared
  across a whole camera flythrough), megakernel or wavefront, with CPU fallback.
- **Whole camera flybys in one render** — some modes amortise a *single* light
  transport pass across an entire moving-camera shot. The **view-independent photon
  map** (mode `M`) is built **once** from one forward photon pass, then reused to
  gather every frame of a camera flythrough (or every camera of a multi-camera
  render), so an *N*-frame flyby costs roughly one render's worth of photons instead
  of *N* — far more efficient than re-tracing the scene per frame.
- **Interactive flypath viewer & editor** — the live `-window` viewer doubles as a
  **camera-curve editor**: author a real `camera_curve` flypath *by flying it* —
  record / insert / delete / steer control points, paint per-point speed and look
  direction, round-trip and revise an existing curve, then save a ready-to-render
  `camera_curve { … }` block. See [Camera animation](#camera-animation-camera_path-camera_orbit).
- **Long-running renders** — time / noise / forever budgets, live ANSI preview,
  and checkpoint/resume.
- **Loom animation toolkit** — a bundled Python toolkit for building scenes and
  seamless looping animations that emit `.ftsl` per frame (procedural ribbons/tubes,
  N-D-transformed isosurfaces, motion graphics, and more). See
  [`tools/loom/`](tools/loom/README.md).

---

## Building

Requires a C++20 compiler and CMake. CUDA is **optional** (auto-detected).

```sh
cmake -B build -S .
cmake --build build --config Release --target ftrace
```

The binary lands at `build/bin/ftrace` (`.exe` on Windows).

> **Windows + CUDA gotcha.** With the Visual Studio generator, CUDA auto-detection
> needs the CUDA **VS integration** (MSBuild props), not just `nvcc` on `PATH`. If
> configure prints `CUDA not found; building CPU-only`, point the toolset at the CUDA
> install directly and select the VS instance that has the integration, e.g.:
> ```sh
> cmake -B build -S . -G "Visual Studio 17 2022" -A x64 \
>   -T "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" \
>   -DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Community"
> ```
> A CUDA-linked `ftrace.exe` is ~3 MB vs ~0.8 MB for a CPU-only build — a quick size
> check tells you which you got.

Useful CMake options:

| Option | Meaning |
|---|---|
| *(CUDA auto-detected)* | If a CUDA toolkit is found, the GPU backend (`src/render_cuda.cu`) is compiled and `-device gpu` becomes available; otherwise a CPU-only binary is built. |
| `-DFTRACE_CUDA_ARCH=native\|all-major\|"75;86;89"` | GPU architectures to target (default `native`). |
| `-DFTRACE_GPU_FP32=ON\|OFF` | GPU transport in float with double film accumulation (default `ON`); `OFF` = full FP64 (slower, CPU-matching). |

---

## Quick start

```sh
# Built-in Cornell box, forward pinhole splat (mode B), 512²
ftrace -scene cornell -n 200000000 -r 512 -o cornell.png

# Render an FTSL scene file
ftrace -in scenes/cornell.ftsl -n 200000000 -o out.png

# Physical-lens camera demo (depth of field from real optics; forces mode R)
ftrace -in scenes/realcam.ftsl -n 6000000 -o realcam.png

# Render until a wall-clock budget with a live preview
ftrace -in scenes/group.ftsl -time 120 -preview -o group.png
```

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
| `V` | Validate | Runs `B` and `R` and reports the best-fit residual between them | CPU (+GPU forward pass) |
| `P` | Composite | Forward `B` for diffuse/caustic pixels + a backward camera ray for specular/coated surfaces | CPU + **GPU** |
| `D` | BDPT | Bidirectional path tracing with MIS over every light×camera connection | CPU + **GPU** |
| `M` | Photon map | Builds a **view-independent** photon map once, then gathers the camera image from it — a direct radius density estimate at the first diffuse hit, or a Jensen final gather one bounce away with `-pmfg <K>` (reusable across cameras) | CPU + **GPU** (direct estimate) |
| `S` | SPPM | Stochastic **progressive** photon mapping: repeated photon passes with a shrinking per-pixel radius — converges (unbiased in the limit), bounded memory, excels at caustics | CPU |
| `U` | VCM/UPS | Vertex **connection and merging**: BDPT vertex connections **and** SPPM photon merging combined under one MIS weight — robust across diffuse GI, glossy, and caustics in a single estimator | CPU |

> **Quick preview — `-raster` (not a transport mode).** To eyeball *composition*
> and *camera motion* before committing to a full render, `-raster` skips light
> transport entirely: it tessellates the whole scene once (analytic spheres →
> UV spheres, isosurfaces/CSG → marching-tetrahedra mesh, instanced meshes baked
> to world space) and z-buffers each camera as solid, flat diffuse+headlight
> triangles — roughly **1 fps at 1280×720**. There is **no** transparency,
> reflection, refraction, shadow, caustic or GI: a dielectric shows as a solid
> ghost and a mirror as a flat tint. (Opt in to **see-through clear objects** with
> `-see-through` — see below — which drops the ghost for a dim + milky-haze pass
> that still refracts nothing.) **Image skins** *are* shown: a material whose
> albedo is a bound texture (`reflect texture:<name>`) is previewed by
> interpolating the surface's per-vertex UVs — or, for an un-UV'd mesh/isosurface,
> the material's world **triplanar** projection — and sampling the texture's linear
> RGB per pixel, so a skinned globe/wallpaper/torus reads with its actual image
> rather than a flat colour. Shading sums a diffuse term from **every**
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
> map the real camera uses), **opaque and textured (skinned)** geometry (image skins
> — per-vertex UV **and** world-triplanar `reflect texture:<name>` albedo — are sampled
> on-device), **and** `-see-through` clear-glass compositing (a device clear-accumulation
> pass mirrors the CPU one). Only a device allocation failure falls back to the CPU
> rasterizer per camera (mixed camera lists just work), so `-device gpu` never fails a
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
> | **mouse wheel** | **dolly** one step forward (up) / back (down) per notch — a discrete, fully-rendered nudge for precise positioning (can't overshoot into geometry) |
> | **`Ctrl` + mouse wheel** | change the **step size**: up = bigger steps, down = smaller (starts at 2 % of the scene radius, clamped to a sane band) |
> | `C` | cycle **wall collision**: `slide` → `stop` → `noclip` (see note below) |
> | `0` (or `Home`) | reset to the authored camera |
> | `P` | print a paste-ready `camera "cam" { eye … look_at … up … fov_y … }` block |
> | **resize the window** | change the preview resolution: the raster renders ~one pixel per displayed pixel, so **shrinking the window renders fewer pixels (faster on a heavy scene) and growing it renders more (crisper)**, up to the authored resolution |
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
> collision modes as `C`, showing the current mode) and **Reset** (jumps back to the
> start — the authored camera in free flight, or frame 0 of the path when locked).
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
> lets you build a real [`camera_curve`](#animated-cameras--flypaths) right in the
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
> another app, etc.) stops the turn entirely. Because the turn is applied **per rendered
> frame** (like the fly motion), a heavy scene turns in careful steps you actually see
> rather than spinning past. The window title shows the live `eye(…) dir(…)` as you move. Frames re-rasterize at
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
| `V` | Correctness check (`B` vs `R` residual) | ~2× *(runs both)* | ✓ *(via `R`)* | ✓ *(via `R`)* | ~ | forward pass | Diagnostic, not a production renderer |
| `P` | Mixed diffuse + mirrors/coatings | Medium | ✓ | ✓ *(routes to `D` w/ lens)* | ✓ | ✓ | Costs more than `B`; possible seam between layers |
| `D` | Specular-first + diffuse caustics + **participating media** in one pass | Slow / sample | ✓ | ✓ *(physical lens)* | ✓ | ✓ | Highest per-sample cost; no fluorescence / spot / env lights |
| `M` | Many cameras sharing one lighting solution (flythroughs); reusable/persistable map | Fast per frame *(after one shared pass)* | ✓ *(walks to diffuse)* | — | ✓ | ✓ *(direct query)* | Direct query blurs contact shadows (use `-pmfg`) |
| `S` | **Caustics / SDS**; progressive, bounded memory | Slow *(many passes)* | ✓ | ✓ | ✓✓ | ✗ | Many passes to converge; CPU only |
| `U` | Robust "have it all" (diffuse GI + caustics), no per-scene mode picking | Heaviest / pass | ✓ | ✓ | ✓ | ✗ | Heaviest per-pass cost; CPU only |

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
  megakernel covers area/sphere/cylinder Lambertian lights and all the
  specular/textured materials; scenes using fog, environment lights, spot/collimated
  lights, or fluorescence fall back to the CPU tracer automatically.
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
  to the lens-aware BDPT (`D`) — or, if the scene is outside BDPT scope (fog / env /
  spot / fluorescence), falls back to the backward realistic camera (`R`).
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
  next-event modes leave dark). *Cost:* highest cost per sample, and it **does not support
  fluorescence or spot & env lights** (use `B`/`P` or `R` for those).
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
  lighting solution. **GPU-accelerated** for the direct density query: the device
  deposits the photon pass, hands the hits to the same grid builder, then gathers every
  camera on the GPU from the one shared map — so a whole flythrough builds the map once
  and renders each frame in device time (a `-pmfg` final gather still falls back to the
  CPU, as do env-lit or unsupported-material scenes). The built map can also be
  **persisted to disk** with `-savemap <f>` and reloaded with `-loadmap <f>`: because it
  is view-independent, a reloaded map re-gathers new camera angles or a new gather radius
  **without re-tracing a single photon** (the expensive forward pass is skipped entirely).
  A scene-identity guard rejects a stale map built for a different scene, falling back to
  a fresh deposit. (Matches the forward splat modes `A`/`B`/`C` — same forward physics,
  just measured from a stored map.)
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
  exactly to mode `M`. CPU only. *Cost:* many passes to converge; the running preview
  starts blurry (large radius) and sharpens as the radius shrinks.
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
  CPU only. *Cost:* the heaviest per-pass (both a full light pass and a full camera pass,
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
physical-lens camera)** via its own GPU backward megakernel — which the **`P` composite
reuses for its camera-side layer**, so both of `P`'s layers run on the GPU when the scene
is within the backward-GPU scope — and the **`M` photon map** (direct density query),
which builds one shared map on the device and gathers every camera from it. Outside that scope `P`'s camera-side layer, and `V`'s
backward reference (kept on the CPU as a stable ground truth), remain CPU-only. The
composite `P` classifies its pixels once, then alternates forward and backward batches
into two accumulating films, re-fitting the forward→backward scale and re-blending each
interval. **Disk `-resume`/`-checkpoint` now cover `A`/`B`/`C` (photon-count checkpoint),
`R`/`D` (spp-count checkpoint), and `P` (dual forward+backward film)** — a resumed render
draws a decorrelated sample stream so its added samples genuinely reduce variance. Only the
persistent-state photon modes `M`/`S`/`U` (whose per-pass state a film alone can't restore)
stay non-resumable.

### Backends & performance (`-device`, `-wavefront`)

- **`-device auto` (default, recommended).** Uses the GPU when a supported CUDA
  device is present *and* the render is one it can handle (forward modes
  `A`/`B`/`C` on a non-fluorescent scene, mode `D`'s BDPT megakernel, mode `R`'s
  backward megakernel — including the physical-lens camera — both layers of the
  mode-`P` composite, or mode `M`'s shared photon map with the direct density query
  on a non-env, pinhole scene); otherwise the CPU. Prints its choice.
- **`-device gpu` / `cpu`.** Force the backend. The GPU **falls back to the CPU**
  for the mode-`P` camera-side layer and for `R`/`D` scenes outside their GPU scope
  (env/spot/collimated lights, fluorescence; any fog for `R`), for a mode-`M` render
  that uses a `-pmfg` final gather or an env light, and for
  fluorescent/oversized-mix forward scenes. Mode `D`'s GPU BDPT megakernel renders
  **all** participating media — haze, superposed, bounded, and heterogeneous
  `density`-field fog — directly on the device. Implicit surfaces / `isosurface`, **procedural patterns**, and
  **dielectric translucency** (frosting + Beer–Lambert colored-glass tint) are all
  GPU-accelerated now — the device sphere-traces the same field expressions, runs the
  same pattern VM, and threads the interior-absorption medium through both the forward
  and backward tracers. GPU **BDPT** (mode `D`) still falls back for any pattern-driven
  material *or* frosted/colored glass, whose per-hit BSDF its MIS kernel can't yet
  reproduce. **Parametric records** (a material's slots driven by a per-hit driver
  sampling a named LUT bank — see *Parametric records* below) run on the **GPU forward
  and backward tracers for both the reflect/albedo and roughness slots** (constant stop
  selectors bake into the device material; per-hit driven reflect uploads the record's
  baked LUT + driver program, and a driven scalar/roughness slot uploads each stop's
  compiled expression + driver — both sampled on-device by exact twins of the CPU
  sampler). **Any** record-bound scene still falls back on GPU **BDPT** (mode `D`) —
  its MIS connection BSDF has no per-hit surface point to evaluate the driver. The
  fallback is automatic. `cpu` is fully deterministic and is used for reference/validation baselines.
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
fog/env/spot/fluorescence, and at most 16 lens surfaces); outside that scope it falls
back to the CPU automatically.

---

## Materials

Declared with `material "name" { type <type> … }`.

| Type | Description | Key parameters |
|---|---|---|
| `diffuse` | Lambertian reflector | `reflect` (spectrum or `texture:<name>`) |
| `translucent` | Two-sided Lambertian (**diffuse transmission** / thin-subsurface look) — light diffuses THROUGH the surface, so a backlit sheet glows softly. Front hemisphere scatters `reflect`, back hemisphere scatters `transmit`; non-specular, so it connects/renders in every mode (A/B/C/R/V/D/P). CPU only. Alias `diffuse_transmit` | `reflect` (spectrum or `texture:<name>`), `transmit` (spectrum); the two are energy-clamped so `reflect+transmit ≤ 1` |
| `dielectric` | Refractive glass with dispersion, optional **frosting**, **colored-glass tint** and **nested-dielectric priority** | `ior` (Sellmeier glass or constant); `roughness` (constant or `pattern:`/`texture:` map) frosts the reflected & transmitted lobes; `absorb` (spectrum, σₐ per metre) tints via Beer–Lambert interior absorption; `priority <N>` (integer) disambiguates overlapping dielectrics — see below |
| `mirror` | Perfect specular reflector | `reflect` |
| `halfmirror` | Lossless beamsplitter; `reflect` is the reflect probability (default 0.5 = 50/50). A spectral `reflect` gives a wavelength-dependent (dichroic) split | `reflect` |
| `filter` | Colored **gel / Wratten filter**: a thin non-scattering absorber. Light passes straight through (no reflection or refraction), surviving with probability `transmit`(λ) — the per-wavelength transmittance T(λ) ∈ [0,1] — and is absorbed otherwise. Like clear glass it isn't lit directly; you see its effect on whatever is behind it | `transmit` (spectrum: `filter:<name>`, `file:<path>`, or a primitive like `gaussian`) |
| `glossy` | Rough microfacet reflector | `reflect`, `roughness` (constant or `texture:<name>` map) |
| `thinfilm` | Single-layer interference (iridescence) | `ior`, `film_ior`, `film_thickness` (nm), `film_thickness_map texture:<name>`, `substrate_k` |
| `multilayer` | N-layer Abelès transfer-matrix stack | `ior`, `substrate_k`, repeated `layer <n> <k> <nm>` |
| `grating` | Reflective diffraction grating | `reflect`, `groove_spacing` (nm), `groove_dir`, `max_order` |
| `fluorescent` | Stokes-shifted fluorescence | `reflect`, `absorb`, `emit`, `yield` |
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
  frosting and colored-glass tint; mode-`D` BDPT still falls back to the CPU.)*

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

Bind a record to geometry with the inline `material NAME(driver)` form, where `driver`
is any pattern expression evaluated per hit (`x y z nx ny nz r u v f`, `noise(…)`, …).
A record driving the **reflect/albedo** *or* **roughness** slot runs on the **GPU**
forward and backward tracers (the LUT/stop programs + driver upload to the device and
are sampled by device twins of the CPU sampler); any record-bound scene still falls back
on **GPU BDPT** (mode `D`). Fallback is automatic. See FTSL.md §7.5 for the full grammar.

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
- **`table { 400:0.05 450:0.12 … }`** — a measured/tabulated spectrum
  (piecewise-linear).
- **`file:<path>`** — load a measured curve (SPD, reflectance, or n(λ)) from an
  external CSV/whitespace data file (`#` comments, a header row, `wavelength_nm,value`
  rows); the runtime ingestion point for the data under `data/`. E.g.
  `spd file:data/illuminant/f2.csv` (see `scenes/measured_spd.ftsl`).
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
  per path instead of our one. **We claim no edge on this axis.**
- **Dispersion — colours actually splitting** through a prism / lens / water. Only
  the single-λ (ours) and hero-wavelength (PBRT-v4, Mitsuba 3) schemes get this right;
  co-sampled spectral (PBRT-v3, Mitsuba 0.x) and every RGB pipeline cannot.

Where popular physically-based renderers fall (verified against their docs/source;
see sources below):

| Renderer (engine) | Default colour | Spectral mode | Per-path/photon carrier |
|---|---|---|---|
| **ftrace (this — forward photon)** | spectral | always | **1 λ per photon** — true dispersive caustics |
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
| `collimated` | Thin parallel pencil beam | `origin`, `dir`, `spd` |
| `env` | Environment / IBL light | `file` (lat-long HDR) or `spd`, `rotate`, `intensity` |

**Absolute power.** Any non-env light may author a real physical output —
`power <watts>` (radiometric radiant flux) or `lumens <lm>` (photometric luminous
flux, via `Φᵥ = 683·∫spd·V dλ`) — instead of relying on the per-image
auto-exposure. Authoring either on *any* light puts the whole scene in **absolute
mode**: the film is physically linear, the auto-exposure is replaced by a fixed
sensor gain, and `iso`/`shutter`/`exposure` become true absolute stops (doubling
`power` is exactly one stop brighter). See `scenes/absolute.ftsl`.

---

## Geometry

`sphere`, `quad` (parallelogram), `triangle`, and `mesh` (**OBJ, glTF 2.0 / GLB,
and Autodesk FBX** import — the loader dispatches on file extension). glTF brings
its node transform hierarchy, per-vertex normals/UVs, and `pbrMetallicRoughness`
materials (base color upsampled to a reflectance spectrum, metallic → glossy tint,
roughness → lobe width; `import_materials no` forces the FTSL `material` instead).
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
Meshes without their own `vt` coordinates can be textured via a procedural
projection — `mesh { uv planar|spherical|cylindrical [x|y|z] }` synthesizes UVs
at load time from the mesh's world-space bounding box (the optional token is the
projection/up axis, default `y`).
`group { translate … rotate … scale … <children> }` composes transform
hierarchies (baked to world space at load). Children may be `sphere`, `quad`,
`triangle`, `mesh`, `mesh_instance`, `isosurface`, `light`, or nested `group`s —
so a physically-settled rest pose (e.g. from `tools/settle_scene.py`) can wrap an
isosurface CSG/implicit just as easily as a mesh.

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
edge signs are what used to crack — the GPU float path.

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
atan2 clamp mix smoothstep noise`, and the constant `pi`). Because an arbitrary field is
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

A texture's albedo can also be **procedural in UV space**: in place of `file`, give
three quoted ftsl expressions of the surface UV — `rgb "r(u,v)" "g(u,v)" "b(u,v)"`
(the pattern infix grammar; variables `u v`, constant `pi`; each output clamped to
`[0,1]`, interpreted as linear RGB). ftrace bakes them once at load to a `res`×`res`
grid (default 512) and then treats it exactly like an image texture — the same
UV-wrap, Jakob–Hanika upsampling, triplanar, GPU and raster paths, and
`reflect texture:<name>` binding all apply unchanged. This completes the skin matrix
alongside image skins and 3-D-space procedural patterns: a **UV-space procedural**.
See `scenes/procskin.ftsl` (loom: `ProcTexture` / `func_skin`).

## Procedural patterns (math-driven materials)

A `pattern "name" { … }` block compiles a **scalar field** — a function of the hit
point evaluated per shading sample — that can drive any scalar material parameter
*procedurally*, without a texture image. The variables available to a pattern are the
world-space position `x y z`, the implicit field value `f` (the SDF value at the hit,
`~0` on an isosurface; `0` for explicit geometry), the surface normal `nx ny nz`, the
radius `r = √(x²+y²+z²)`, and the **surface UV coordinates `u v`** (mesh-interpolated,
or a native-primitive wrap — see below). Two authoring forms:

- **Free-form expression** — `expr "0.5 + 0.5*sin(40*y)"` (must be quoted). Compiled by
  a shunting-yard parser to a postfix scalar VM. Supports `+ - * / ^ %`, comparison-free
  math, `pi`, and functions `abs sqrt sin cos tan exp log floor fract sign saturate min
  max atan2 step pow clamp mix smoothstep noise`.
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
paths, including a roughness pattern on a `dielectric` (frosted glass). GPU BDPT is the
exception — its MIS kernel falls back to the CPU for any pattern or frosted/colored
glass.)*

**UV on native primitives.** The `u v` pattern variables aren't limited to meshes.
A native `sphere {}` carries built-in equirectangular (lat/long) UVs, a `quad {}`
maps its `u`/`v` edges to (0,0)→(1,1), and an `isosurface` can request a procedural
wrap with `uv planar|spherical|cylindrical [axis=x|y|z]` — the **same projection used
for un-`vt`'d meshes**, referenced to the surface's world bounds. So a checker or
stripe authored in `(u,v)` space wraps *around* the object (a globe, tiles converging
at the poles, a grid on box faces) instead of slicing through world space. See
`scenes/uv_native.ftsl`.

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
and keep the fog thin (single-scatter regime). Evaluated by the CPU tracers (forward
A/B/C, backward R, BDPT D); a rainbow-phase scene automatically falls back to the CPU
on the GPU backend (the device volume path is HG-only). See FTSL.md §12.

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
exactly, carved per-point during tracking), and a named `mesh` uses the mesh's world
AABB (a box approximation; true mesh containment is deferred). An *open* fog sphere is directly viewable in every mode.
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

**Imported volumes (`.nvdb`).** Instead of a formula, point the density field at a real
sparse volume: `density vdb:<path.nvdb>` imports a NanoVDB **FloatGrid** (the compact,
GPU-friendly form of an OpenVDB volume — convert a `.vdb` with OpenVDB's `nanovdb_convert`,
uncompressed). On load the grid is **baked into a dense lattice** plus a world→index affine,
so the *identical* trilinear sampler runs on the CPU and the GPU (`dMedDensityAt` reads the
uploaded lattice) and any affine map (translation/scale/rotation) is honored. The grid's
world AABB auto-seeds the medium bound and its peak value the delta-tracking majorant — so
`medium { sigma_t 40  albedo 0.9  density vdb:cloud.nvdb }` is all it takes to light an
imported cloud. Values are treated as a dimensionless density multiplier on `sigma_t`, so
you still dial the optical thickness with `sigma_t`. Only **float** grids are supported and
the bake is **dense** (memory ~ the grid's index-space bounding box), so very large sparse
volumes are bounded by a safety cap; a native sparse device sampler is a future
optimization. Works in the forward modes (A/B/C) and BDPT `D` on CPU and GPU, exactly like a
`density` formula. Generate a test asset with `scraps/make_nvdb.cpp`.

**Gradient-index (GRIN) media — bending light *(experimental, mode `R` only)*.** Give a
medium an `ior "<expr over x y z r>"` field (or `ior pattern:<name>`) and it becomes a
**gradient-index region**: rays that enter its `bounds{}` no longer travel straight — they
**bend continuously**, integrating the Eikonal ray equation `d/ds(n·dr/ds)=∇n` with a
small symplectic march step (`ior_step <v>`, default 1/64 of the smallest bound extent).
This makes mirages, hot-air shimmer, and **gradient lenses that focus/warp with no glass
surface at all**. E.g. `medium { bounds { center 0 0 2 radius 0.9 } ior "1.6 - 0.6*(sqrt(x*x+y*y+(z-2)*(z-2))/0.9)" }`
is a radial index ball (n=1.6 core → 1.0 rim) that visibly lenses a checkerboard behind it
(`scenes/grin_lens.ftsl`). **Currently only the CPU backward tracer (mode `R`) bends GRIN
rays** — the forward modes (A/B/C), BDPT `D`, and all GPU paths still trace these regions
straight (they ignore `ior`), so render a GRIN scene with `-mode R -device cpu` for now.
Wiring the Eikonal march through the other tracers/GPU is tracked in `known-issues.md`.

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
field), `spectrum`, `sphere`, `quad`, `triangle`, `mesh`, `isosurface` (implicit SDF
surface / CSG / metaballs / arbitrary `function` formulas), `light`, `group`, `medium`,
`camera`, `camera_path` (keyframed camera animation), `camera_orbit` (turntable /
fly-around: N frames on a circle around a `center`, for MP4 orbits), `camera_curve`
(spline fly-through with variable speed), and `render` (render-setting overrides). See the `scenes/` directory for worked examples
(`cornell.ftsl`, `fisheye.ftsl`, `spotlight.ftsl`, `envlight.ftsl`,
`material_presets.ftsl`, `realcam.ftsl`, `implicit.ftsl`, `function.ftsl`,
`procedural.ftsl`, `uv_native.ftsl`, `showcase_orbit.ftsl`, `translucency.ftsl`,
`gallery.ftsl` (a large room packed with varied materials around a gold gyroid), …).

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
  slow dwell, low density = fast. Aim along the travel tangent (default), at a fixed
  `look_at`, or at a second `look curve`. The **travel tangent is fold-robust**: where the
  path makes a sharp horizontal U-turn its look-ahead chord loses horizontal reach and would
  otherwise rake the view steeply up into the ceiling / down at the floor, so `min_reach <f>`
  (default `0.5`, `0` = legacy) floors that reach for the pitch calculation and `look_smooth
  <n>` (default `0`; a Gaussian sigma in frames) temporally smooths the look direction so a
  fold reads as a bounded near-level pan instead of a flick. **Orientation and lens can also be animated**
  per frame over the normalized timeline `t ∈ [0,1]` (`t=0` first frame, `t=1` last),
  each keyframed by `<name>_at <t> <value>` (piecewise-linear, flat-clamped at the ends,
  just like `density_at`) or held constant by the bare keyword: **`roll[_at]`** banks the
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
| `-o <path>` | Output image (`.png` / `.jpg` / `.ppm` by extension) |
| `-topng <in> <out.png>` | Convert an existing `.ppm` or `.ftbuf` to a 24-bit PNG (no rendering); see **Output** |
| `-review <base>` | Play a directory of already-rendered frames (`<base><digits>.<ext>`, e.g. `png/swoop/swoop`) on the live window/timeline — scrub/Play, re-time by painting speed, and Save a re-paced copy (no rendering); see the fly-viewer section |
| `-serve` | **Resident preview server.** With `-serve -in <scene.ftsl> [flags…]`, ftrace does *not* exit after one render: it keeps the process — and with it the live window, CUDA context, and spectral/spectral-upsampling tables — resident, and re-renders whenever a new scene path arrives on **stdin** (one path per line), reusing all the other flags (`-mode`/`-n`/`-r`/`-window`/`-o`/…) with only `-in` swapped per frame. Line protocol: prints `[serve] ready` once, then `[serve] done <path>` after each frame; `quit`/`exit`/EOF ends the loop (`[serve] shutdown`). This skips the per-frame cost of process spawn + window/CUDA/table init — the dominant fixed overhead for cheap preview frames — so an external driver (e.g. loom's `PreviewServer`) can stream an animation into a single window that updates in place. Scope: resident-process reuse only; each frame is still a full independent render (no delta/geometry caching yet) and the window keeps the first frame's resolution for the session. |
| `-mode <A..D,M,S,U,P,R,V>` | Render mode (default `B`) |
| `-on-unsupported error\|fallback\|strip` | What to do when the selected mode can't render a scene feature (GRIN media, or a fisheye camera in mode `D`/`U`). `error` (default) prints a diagnostic and aborts; `fallback` renders that camera in mode `R` (backward reference) instead; `strip` removes the offending feature (e.g. drops the GRIN `ior`, turning the medium into a plain one) and renders in the requested mode anyway. Complements `prefer { … } else { … }` in the scene file, which resolves the mode/feature mismatch *before* this policy is consulted |
| `-pmradius <r>` / `-pmradiusfrac <f>` | Mode `M`/`S`/`U` photon-map/merge gather radius (initial radius for `S`/`U`): absolute world units, or a fraction of the scene radius (default `0.02`). Smaller = sharper contact shadows but noisier |
| `-pmfg <K>` | Mode `M` final gather: `K` cosine-weighted hemisphere sub-rays per sample, querying the map one bounce away for sharp contact shadows / fine detail (default `0` = off, direct density query). ~`K`× per-sample cost — pair with fewer `-spp` |
| `-savemap <f>` / `-loadmap <f>` | Mode `M` (GPU) view-independent photon-map cache. `-savemap` writes the built map to `<f>` after the forward deposit; `-loadmap` reloads it and **skips the deposit**, re-gathering any camera / radius for free. A scene-identity guard falls back to a fresh deposit if the file was built for a different scene |
| `-sppmalpha <a>` | Mode `S` radius-shrink rate (default `0.7`; smaller shrinks faster) |
| `-vcmalpha <a>` | Mode `U` (VCM) radius-shrink rate (default `0.75`; smaller shrinks faster) |
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
| `-fog <σt>` / `-fogalbedo <a>` / `-fogg <g>` / `-fograyleigh` | Fog controls |
| `-filmthickness <nm>` / `-filmior <n>` | Thin-film iridescence demo params |
| `-diffraction <mode>` / `-nodiffraction` | Enable/disable grating & thin-film diffraction |
| `-spp <n>` | Samples per pixel for modes `R`, `D`, `M`, and `V`; **number of passes** for SPPM (`S`) and VCM (`U`) |
| `-n <photons>` (mode `S`) | Photons traced **per pass** (SPPM rebuilds a bounded map each pass). *(Mode `U` ignores `-n` — its light-path count follows the film resolution.)* |

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
| `-window` | Open a real OS window (Win32 GDI; no-op off Windows) showing the actual tone-mapped pixels, refreshed each `-interval` tick. Full-resolution, unlike `-preview`'s terminal thumbnail; runs on its own UI thread. A plain fixed-`-n` forward render is auto-chunked so the view converges live, and closing the window stops the render (final image is still written). The title bar identifies the render as `ftrace — <scene> → <output>`, then the transport mode driving that frame (`mode B (pinhole)`, `mode D (BDPT)`, `mode M (photon map)`, …; a per-camera flight shows the mode of the frame currently on screen), then the live status (`spp` / `% noise` or photon count) as it converges, so you can tell at a glance which scene/file the window is showing, how it's being rendered, and how far along it is. The window opens at (and won't be dragged smaller than) a readable minimum so that `<scene> → <output>` title stays legible even for a small image; the picture is aspect-fit and letterboxed inside whatever size the window is. |
| `-keepwindow` / `-hold` | Like `-window`, but **don't auto-close** the live window when the render finishes — normally the window is torn down at process exit the instant the last frame completes, so a finished image only flashes on screen. With this set, ftrace keeps the final image up and blocks until you close the window yourself (handy for inspecting a quick `-raster` preview or a completed still). Implies `-window`. |
| `-interval <s>` | Periodic image write / preview / window refresh (default 15 s) |
| `-raster` | Fast solid-shaded **preview** (no light transport): z-buffer the whole scene as flat-shaded triangles, one image per selected camera. Honours `-camera` and `-window` (a `camera_curve` flyby animates in the window; a single still becomes an **interactive fly camera** — Space/`+` fly forward, Shift/`-` back, move the mouse off-centre to steer (rate/joystick look, cursor stays visible), wheel = dolly, Ctrl+wheel = step size, `C` = wall collision, `0` resets, `P` prints a paste-ready camera, plus **Clip/Reset buttons** in a panel below the image). See the preview note under **Render modes**, and `-explore` below to drop straight into this viewer at a flyby's first frame. |
| `-raster-iso <n>` | Isosurface mesh fineness for `-raster` (cells along the longest bounds axis; default 96, `0` skips implicits) |
| `-see-through` / `-seethrough` / `-glass` | In `-raster`, render **clear** materials (dielectric / thin-film / filter / diffuse-transmit) as actually see-through instead of solid ghosts: each clear surface between the camera and the opaque background **dims** and **milkily hazes** what's behind it, cumulative with the number of clear surfaces crossed (no refraction, no coloured absorption). Order-independent, so overlapping glass needs no sort. See the preview note under **Render modes**. |
| `-glass-clarity <0..1>` | Per-surface transmittance for `-see-through` (default `0.85`; higher = clearer / less dimming). Passing it implies `-see-through`. |
| `-explore` / `-fly` | **Interactive fly-through** of a multi-frame flyby without rendering it. Seeds the interactive raster viewer at the **first frame** of the selected `-camera` path (e.g. `-camera fly`) and hands control to you: Space/`+` fly forward, Shift/`-` back, move the mouse off-centre to steer (rate/joystick look, cursor stays visible), wheel = dolly, Ctrl+wheel = step size, `C` = wall collision, `0` resets the view, `P` prints a paste-ready camera block, close the window to finish. The flyby's frames are kept as a **camera-path timeline** in the panel below the image: **scrub/play/pause** across them, **lock** the camera onto the path (travel forward/back along it at a **cams/update** or **cams/second** speed), or release to fly freely — see **Interactive camera** for the full panel. Implies `-raster -window -keepwindow -no-meter`. Use it to preview/author a flyby camera without watching or writing every frame. |
| `-no-meter` / `-nometer` | Skip the **exposure-lock metering pre-pass**. Normally a locked `camera_curve`/`camera_path`/`camera_orbit` group meters (up to 64 of) its frames up front to compute one shared exposure anchor, so the flyby doesn't flicker. With this flag that pre-pass is skipped and each frame **auto-exposes on its own** — faster startup (no metering the whole path), at the cost of possible frame-to-frame brightness flicker on an animated flyby. Implied by `-explore` (the interactive viewer auto-exposes per frame, so metering a whole flyby just to fly one frame is wasted work). |
| `-noclip` / `-nocollide` | Start the interactive fly-viewer with **wall collision off** (fly through geometry) — for placing a camera *outside* the room or *inside* glass. Collision is **on by default** (you can't fly through walls); press `C` in the viewer to cycle `slide` → `stop` → `noclip` live. See the fly-camera controls under **Interactive fly camera**. |
| `-resume` / `-checkpoint` | Resume from / always write a `<out>.ftbuf` checkpoint (modes `A`/`B`/`C`, `R`/`D`, and `P`) |
| `-exposure-lock` | Share one auto-exposure anchor across all rendered cameras (no `camera_path` flicker); a per-path `exposure_lock [selector]` keyword instead locks just that path, metered from a chosen viewpoint (default the path `average`; also `first`/`index i`/`near x y z`/`camera "name"`) |
| `-exposure <c>` / `-ev <c>` | Override the exposure **compensation** for every rendered camera (a relative stop multiplied on top of the p99 auto-exposure; `1.0` = neutral), replacing the per-camera film `exposure`. Applies to both the real render and the `-raster` preview — handy when a scene's authored `exposure` (tuned for the physical integrator's bright highlights/caustics) blows out the flat-shaded raster. |
| `-stereo <mode>` | **3-D stereoscopic output** (stills *and* movies). Renders each camera **twice** — a Left/Right eye pair — and composites them into the `-o` image. `mode` picks the fusion: `sbs` (side-by-side **wall-eyed**, L\|R), `cross` (side-by-side **cross-eyed**, R\|L), `anaglyph` (**red-cyan** Dubois glasses, the default kind), or `anaglyph-gm` (**green-magenta** Dubois). Uses the correct **off-axis** rig — two *parallel* cameras offset along the camera right axis with **asymmetric (sheared) frusta** sharing a convergence plane, so there's **no vertical parallax** (toe-in's eye-strain cause). Both eyes share one auto-exposure anchor, so L/R — and every frame of an exposure-locked `camera_path` — tone-map identically. Rectilinear cameras only (a fisheye camera renders mono, with a warning). See **Stereoscopic 3-D** below. |
| `-eye-sep <m>` | Interocular (eye-to-eye) distance for `-stereo`, in metres. Default `0.063` (63 mm, average human). |
| `-view-dist <m>` | Viewing distance (eye-to-screen) for `-stereo`, in metres. Default `0.6`. Used to derive the screen width (screen shows the camera's horizontal field at this distance) when `-dpi` isn't given. |
| `-dpi <n\|auto>` | Screen pixel density for `-stereo`. With a number, screen width `= resX·0.0254/dpi`. `auto` reads the Windows **logical** system DPI (a rough hint — often 96; not the panel's physical pitch). Omit it (the default) to instead derive screen width from `-view-dist` × the camera FOV. |
| `-convergence <m>` | Convergence-plane distance for `-stereo`, in **scene units** — the depth that lands at the screen (zero parallax); nearer objects pop out, farther recede. Default: the camera's **look-at target** distance. |
| `-stereo-keep-eyes` | Keep the intermediate per-eye PNGs (`<out>_<cam>__eyeL/​R.png`) that `-stereo` writes before compositing. By default they're deleted once the composite is done. |

**Diagnostics / self-tests:** `-checkbvh`, `-bvhstats`, `-checklens`,
`-checkfluoro`, `-checkfog`, `-checkthinfilm`, `-checkmultilayer`,
`-thinfilmswatch`, `-checkgrating`, `-checkupsample`.

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

A `.ftsl` is a *scene*, not an image — render it with `-in scene.ftsl -o out.png`.
Three drag-and-drop Windows helpers in the repo root wrap this: **`ppm_to_png.bat`**,
**`ftbuf_to_png.bat`** (both call `-topng`), and **`ftsl_to_png.bat`** (renders the
scene). Drop a file on one, or run `ppm_to_png.bat input.ppm [output.png]`.

---

## Loom — procedural animation toolkit

The repo bundles **Loom** (`tools/loom/`), a programmatic-first Python toolkit for
building 3-D scenes and **seamless looping animations** that render on ftrace. Loom
animates *continuous* things — modulator graphs, curves, fields, N-D-transformed
isosurfaces — and discretizes **last, per frame**, emitting one `.ftsl` per frame
which ftrace then renders (raster preview or full path trace) and assembles into a
GIF/MP4. It ships with ready-to-run examples (swept ribbons/tubes, gyroid and other
triply-periodic minimal-surface loops, higher-dimensional gyroid slices, function-driven
materials, 2-D motion graphics, spacetime-transform videos) and stands alone (it can
drive any renderer).

See **[`tools/loom/README.md`](tools/loom/README.md)** for the tour, and
`tools/loom/DESIGN.md` for the architecture.

---

## Known issues & roadmap

Open limitations and technical debt are tracked in `known-issues.md` — including
the physical-lens camera's remaining gaps (inter-element flare/ghosting,
shaped-iris bokeh).
