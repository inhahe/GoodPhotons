# Known Issues & Technical Debt

Running log of unsolved bugs and accumulated tech debt. Fix items here as soon
as practical; this file is the fallback for what can't be addressed immediately.

## Open issues

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
    bounded), and modes `R` / `D` / `U`. Also **not** applied at `Layered` / `Mix` — see the next bullet.
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
    `Layered` has the same shape (its coat Fresnel probability is λ-dependent, so it de-heros outright).
    **`-herosplit` does not cover either.** These are λ-dependent *decisions*, not λ-dependent *directions*, and
    their natural split point sits *before* any interaction has happened — which does not fit
    `tracePhotonHeroLoop`'s "resume from a ray" entry shape the way the dispersive case does (there each
    secondary can just re-run `interactPhotonSpecular` at the same hit with its own λ and hand back a fresh ray).
    Extending split to them needs a "resume at this hit with this material" entry point; worth doing if a
    spectral-mix or coated-dielectric scene ever shows the bias, but nothing observed yet.
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

### TECH DEBT (2026-07-19): settle sim slow — non-manifold stand colliders won't decimate below ~150–390k tris
`tools/settle_scene.py` decimates static concave colliders to `STATIC_TRI_CAP` (4000) via
`trimesh.simplify_quadric_decimation`, but the marching-cubes museum-stand meshes are multi-shell /
non-manifold (unions of boxes), so quadric edge-collapse gives up early and only reduces them ~65%
(e.g. stand meshes 500k–1M → 150k–390k tris), keeping per-step collision cost high. Clean closed meshes
(gyroid, lamp) hit 4000 fine. Worked around by validating at `--mesh-res 64`. **Proper fix candidates:**
(a) VHACD-decompose the static stands into convex compounds too (fast convex-vs-convex collision), or
(b) vertex-clustering / voxel-remesh decimation that ignores topology, or (c) approximate each stand
with authored box/cylinder primitive colliders instead of the polygonised isosurface.

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
**AABB** (box approximation; true mesh containment deferred). Media are resolved in a
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
  ever support POV-Ray-style implicit geometry. Not started.

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
