# Loom — procedural animation & geometry toolkit

Loom is a **programmatic-first** Python toolkit for building 3-D scenes and
**seamless looping animations** out of composable *modulators*, *curves / grids /
scatter data*, *sweeps* (ribbons / tubes / blobs), and *N-D-transformed
isosurfaces*. It targets the [Good Photons / `ftrace`](../../README.md) spectral
forward raytracer by **emitting one `.ftsl` scene per frame**, but is written to
stand alone (the model is plain Python, so it can drive any renderer or a preview).

Loom lives at `tools/loom/` and is self-contained: the modulation core (a small
`signals` DSL, vendored from soundshop) ships with it, so there is nothing to
install beyond Python (plus **Pillow** for GIF assembly and, optionally, **ffmpeg**
for MP4 and **NumPy** for the mesher / spacetime tools).

---

## Core ideas

1. **Programmatic-first.** The authoring surface is a Python API; the Python model
   is the single source of truth. Any front-end (a passive viewer, a future editor)
   is just a reader/writer of it.
2. **Functions/fields over time; discretize LAST, per frame.** Never transport a
   discretization (mesh, point cloud) through time — animate the continuous thing (a
   `Signal`, a field, a curve) and re-sample / re-mesh every frame. This is what keeps
   N-D-rotated isosurfaces contiguous and everything composable.
3. **One mechanism, unlimited depth.** Modulators are a DAG of pure functions;
   "modulators modulating modulators" is just more edges, not more machinery.
4. **Seamless loops are structural, not patched.** A loop is a *closed* path in some
   space (space, time, or a modulator's value). Built closed, it needs no seam fixup.
5. **Emit-`.ftsl`-first.** Prefer letting the renderer mesh / root-find isosurfaces
   from emitted `.ftsl`; an in-tool marching-cubes mesher is used only where a field
   must be baked to geometry.
6. **Everything transformable, everything modulatable.** Any element takes a
   signal-driven `Transform` — `element.transformed(translate=, rotate=, scale=, skew=)`
   (position / size / Euler rotation / x·y·z shear, each a `Signal`) — emitted as an ftsl
   `group{}` (`shear` included). A `Grid`/`Scatter` can carry the *same* transform as a
   local→world **placement**: the field inverse-maps a world query into the dataset's local
   frame, so a fixed world-space sampling curve reads *different* values as you move / resize
   / skew the data object under it (the curve is decoupled from the object).
7. **One influence model, on any axis.** `loom.axes` types a signal by its *free axes*
   (`{t}`, `{s}`, `{s,t}`, …), so broadcast on unshared axes and pointwise on shared ones are
   implicit and the only cross-axis op is an explicit `Reduce`. Influence is one edge model:
   `Target(kind, [mod(a), pin(b)], base)` with target-declared neutrals (`ADDITIVE` 0 /
   `GAIN` 1 / `BIPOLAR` ½). Drop such a node in **any** scene value-site and it's lowered
   automatically:

   ```python
   from loom import Sphere, Target, GAIN, mod, Sine, lower, CurveSample, Ax, Ramp

   # a GAIN target driving a radius: base 0.3, modulated by a lifted legacy Signal
   sphere = Sphere(center=(0, 0, 0), material="gold",
                   radius=Target(GAIN, [mod(0.6 + 0.4 * Sine())], base=0.3))

   # a spatial curve sampled along its own arclength axis, swept over the loop
   sphere.center = lower(CurveSample(path_curve, Ax("s")), dim=3, bind={"s": Ramp()})
   ```

   A value-site has only the clock axis in scope, so any *other* axis must be pinned with
   `bind={axis: coord-or-Signal}` — an unbound one is a construction-time error naming it.

---

## Layout

```
tools/loom/
├── loom/            the package
│   ├── signals/     modulation DAG (Signal graph): leaves, math ops, N-D vector signals, retime/delay/warp
│   ├── mathnd.py    N-D vectors / matrices, Givens-rotation builder, the 3-D slicer
│   ├── data.py      datasets: PointPath | TrackedPath | Grid | Scatter (N-D, DAG nodes)
│   ├── color.py     colour model: RGB + HSV + HSL (animatable, seamless hue loops)
│   ├── interp.py    interpolators (loop curve | tracked multi-curve | grid field | scatter field)
│   ├── iso.py       isosurfaces: gyroid / Schwarz-P / Schwarz-D / Neovius, in 3-D and true N-D (`SliceField`)
│   ├── pov.py       POV-Ray function library, with which are N-D-generalizable
│   ├── spatial.py   spatial expression DSL (X, Y, Z, U, V, T + math + Image/VolumeField/SigAt terms) → ftsl `expr`
│   ├── sweep.py     sweep engine (rotation-minimizing frames, ribbon/tube/skin_rings, OBJ out)
│   ├── mcubes.py    adaptive marching cubes (bake a scalar field to a mesh)
│   ├── vdbio.py     bake a field to a dense grid → OpenVDB .vdb (density/temperature); reads .vdb and NanoVDB .nvdb
│   ├── axes.py      axis-typed signals: broadcast/pin/mod composition + sample/reduce grammar + lower() onto any scene value-site
│   ├── anim.py      N-D curve → scene-variable go-between: config + JSON sidecar + value fan-out + named slots + live pipe
│   │                (`python -m loom.anim scene.py --config drive.json` serves the live loop over stdio;
│   │                 ftrace reshapes the same sidecar in its fly editor with `ftrace scene.ftsl -anim drive.json`,
│   │                 and `… -anim drive.json -loom scene.py` scrubs it LIVE — loom samples the curve and
│   │                 re-emits the scene, so the bound variables move in the viewport, bindable from a panel row)
│   ├── material.py  function-driven materials (waves/checker/rings/blobs, mixes)
│   ├── scene.py     Scene / Camera / Material / Texture (image skins) / geometry / Volume media (all animatable)
│   ├── transform.py per-object Transform (translate/rotate/scale/skew, animatable) → ftsl group{}; dataset inverse-map
│   ├── canvas.py    2-D canvas (motion graphics: markers, strokes)
│   ├── audio.py     procedural audio: one sample-buffer back-end → WAV (offline)
│   ├── xvideo.py    two-pass spacetime transforms (rotate/shear a 4-D block)
│   ├── ftsl_emit.py .ftsl emission
│   ├── block.py     layout-preserving generic element (Block/Stmt) for the baked kinds
│   ├── grammar/     the shared EPEG .ftsl grammar + the read direction:
│   │                `parse_element` / `parse_elements` rebuild loom elements from .ftsl
│   │                text, byte-identically re-emittable (layout, alignment, comments)
│   ├── drive.py     drivers: render a frame range → ftrace → GIF/MP4 assembly
│   ├── preview.py   resident preview server (keeps ftrace + GPU context warm)
│   └── viewer.py    native-viewer contract: build() loader + scene-introspection JSON sidecar (§F1),
│                    plus the resident live channel (`python -m loom.viewer <scene.py>`) that
│                    `ftrace -viewer <s.json> -loom <scene.py>` drives to re-derive geometry (§F4)
├── examples/        runnable scripts (see below)
├── tests/           pytest suite
└── DESIGN.md        the architecture / roadmap document
```

---

## Quick start

```sh
cd tools/loom

# a seamless looping swept ribbon (emits .ftsl frames, renders, assembles a GIF)
python examples/ribbon_loop.py

# a morphing higher-dimensional gyroid slice as a video
python examples/gyroid_nd.py --count 1

# just print what a spacetime-transform pass does (no render)
python examples/transform_video.py
```

Most examples take `--help`. Rendering shells out to the `ftrace` binary (found
automatically via `loom.drive.find_ftrace`); build it first (see the top-level
[README](../../README.md#building)).

### Rendering and assembling a loop

```python
from loom import render_range, assemble_gif_ffmpeg, assemble_mp4

pngs = render_range(scene, frames=150, outdir="png/myloop", name="myloop", loop=True)
assemble_gif_ffmpeg(pngs, "png/myloop/myloop.gif", fps=25.0)   # needs ffmpeg
assemble_mp4(pngs, "png/myloop/myloop.mp4", fps=25.0)          # needs ffmpeg
```

`assemble_gif` is the dependency-free fallback (Pillow only); `assemble_gif_ffmpeg`
builds a per-loop optimised palette and looks considerably better on detailed
imagery. Both default to `loop=0` — GIF for "repeat forever"; pass `loop=-1` for
play-once.

Two things to know if the loop must be **seamless**:

- **Use a frame rate that divides 100.** A GIF stores its inter-frame delay in whole
  centiseconds, so 25 fps (4 cs) is exact while 60 fps rounds 1.67 → 2 and plays back
  at 50 fps. MP4 doesn't care.
- **Render with `loop=True`.** That maps frame `k` to `t = k/frames`, so frame
  `frames` *is* frame 0 and is correctly left out — the encoders emit exactly one
  frame per input, never duplicating one at the seam.

### Reading `.ftsl` back

`loom.grammar.reader` goes the other way — text to elements — for editing an existing
scene:

```python
from loom import Cache, Clock
from loom.ftsl_emit import EmitCtx
from loom.grammar.reader import parse_document

doc = parse_document(open("scenes/gyroid.ftsl").read())
doc.blocks("isosurface")[0].set("resolution", "128")
open("scenes/gyroid.ftsl", "w").write(doc.emit(EmitCtx(clock=Clock(t=0), cache=Cache())))
```

This is **not** the inverse of `emit` — loom bakes its `Signal`s at a clock, so a `.ftsl`
file is a static snapshot and the authoring object that wrote it is unrecoverable. What it
guarantees is **round-trip fidelity**: every line you don't touch comes back byte for
byte, keeping its layout, column alignment, comments and blank lines (a `Document` also
holds the text *between* elements, which belongs to neither of them). Kinds that map onto
an authoring class without loss (`material`, `texture`, `sphere`, `light`, `camera`,
`spectrum`, records) come back as that class and re-emit in loom's canonical form; the
rest come back as a generic ordered `Block` that re-emits its source layout exactly.

---

## Examples

| Script | What it makes |
|---|---|
| `ribbon_loop.py` | a seamless looping swept **ribbon** (plus a twin tube) |
| `scribble_loop.py` | a seamless looping 3-D "scribble" curve |
| `gyroid_loop.py` | a seamless looping **gyroid** isosurface |
| `jumping_jack.py` | a **jack tumbling through a world-static gyroid**: six arms (3 gold, 3 SF10 glass) built as `intersect { union{sphere,cylinder} function{gyroid} }`, where only the arm leaves carry the pose so the lattice flows *through* the moving solid instead of riding along. The carving gyroid's level is picked by **volume fraction** (`--solid`, inverted numerically off the gyroid's own sampled distribution) rather than by a raw threshold — that is what makes the parts read as open lattice rather than dimpled balls. `--solid` alone cannot separate *lacy* from *see-through*, though, since for a one-level carve the surviving envelope **is** the volume fraction; so a sparse **counter-network** (`--counter`) taken from the gyroid's *other* labyrinth, `g >= t`, is unioned in. It threads down the middle of the first one's voids — exactly where the sight-lines run — so it halves how much of the room shows through the part for ~2 points of envelope, where buying the same reduction out of `--solid` costs ~12. Both fold into a **single** `function` leaf via `min(a,b) = ((a+b) − |a−b|)/2`, whose `a+b` is constant here, so the gyroid is still evaluated once per march step. The motion is a real jack's tumble: the ±y arm pair *is* the spin axis, it leans `--tilt` off vertical and precesses, so the two axis balls ride antiphase circles — and because that holds the bottom ball at a *constant* height, standing the jack exactly on the floor is the closed form `arm·cos(tilt) + ball` rather than a search (`rest_height`). One axis is all gold, one all glass, and the spin axis carries one of each so the jack's top and bottom stay distinguishable. Delivered as a 60 fps MP4 (432 frames, 7.2 s) plus a 20 fps GIF built from every 3rd frame, since a GIF's integer-centisecond frame delay cannot express 60. Shows the CSG field tree from a custom `Element`, plus mode W + `-gi` for a flicker-free loop |
| `pastel_jack.py` | the same jack **restyled as two matte pastel shells** (the `_room_of_gyroids_f12` palette, relative auto-exposure, no gloss/glass/dispersion), ringed by a thin **gold torus** tipped 45° off horizontal that spins about the room's vertical like a coin caught halfway through falling. The ring's size is not a free parameter: putting its centre on the jack's centre turns "low point on the floor" and "high point level with the top of the jack" into the single equation `R·sin(45°) + r = arm·cos(tilt) + ball`, and because the *tilt* is constant while only its azimuth turns, both extremes are constant in time too — the ring touches down at every instant of the loop, never hovering and never clipping through the floor. One `torus` leaf, which unlike the jack's `function` field is an exact SDF with analytic bounds and so needs no `contained_by` / `max_gradient`. The framing is re-tuned from `jumping_jack`'s (fov 38→44, eye pulled back 0.5 m and lifted) because the ring swings its near arc to 1.27 m of the camera, where the old frame cut it off for a third of every turn. `--ring-turns` sets the ring's rate in signed turns per jack cycle and `--cycles` how many jack cycles the loop holds; they are separate knobs because a seamless loop only needs their **product** to be whole, so a fractional rate (`--ring-turns 1.5`) becomes legal once the loop is long enough to close it (`--cycles 2`), and lengthening the loop scales the jack's own `spin`/`precess` with it so that only the ring slows down. Sign matters as much as magnitude: the jack's precession walks its lean around world +y once per cycle — the same axis and the same sense as the ring's azimuth — so a *positive* whole rate is an exact integer multiple of the jack's own motion and reads as geared to it rather than as independent of it; negative counter-rotates. **This is also the worked end-to-end example of the driven-channel path, and the source of the project README's demo clip.** Nothing in it is keyframed: every moving quantity is a `Signal` of the loop phase, the dependent geometry (the ring's radius, the jack's standing height) is *solved* from those signals rather than animated beside them, and the five authored quantities are published as named `Slot`s — `ring_spin`, `ring_tilt`, `jack_lean`, `jack_spin`, `jack_precess` — with a `DRIVE` proposal, so `python -m loom.anim examples/pastel_jack.py` drives it live. `jack_lean` shows why that is worth doing: one node feeds the pose, the standing height *and* the ring's sizing, so the floor-contact invariants hold at every value a curve can hand it (checked in `tests/test_example_pastel_jack.py`) |
| `pov_loop.py` | a seamless looping **POV-Ray function** isosurface |
| `gyroid_nd.py` | **higher-dimensional gyroid slices** — a randomized N-D gyroid whose hidden dimensions drift / rotate / *bloom*; gold or clear-glass; per-run output dirs (a full sub-tool, see its `--help`) |
| `gold_gyroids.py` | two gold gyroids in a closed room **rotating in 4-D and 5-D** (`SliceField`), assembled into a seamlessly looping GIF |
| `material_loop.py` | a seamless looping **function-driven material** |
| `mesh_bake.py` | **bake a scalar field to a mesh** with marching cubes |
| `open_timeline.py` | a **one-shot, non-looping** animation (distinct endpoints) |
| `canvas_loop.py` | a seamless looping 2-D **motion graphic** |
| `shared_pattern.py` | **one spatial definition, two backends** (same field as geometry *and* material) |
| `transform_video.py` | the two-pass **spacetime transform** (motion synthesized by tilting space into time) |
| `preview_server.py` | the **resident preview server** (warm ftrace process across edits) |

---

## Tests

```sh
cd tools/loom
python -m pytest tests -q
```

---

See **[`DESIGN.md`](DESIGN.md)** for the full architecture (the modulation DAG, the
N-D slicer, the sweep engine, the driver/IO layer) and the milestone roadmap.
