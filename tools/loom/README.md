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

---

## Layout

```
tools/loom/
├── loom/            the package
│   ├── signals/     modulation DAG (Signal graph): leaves, math ops, N-D vector signals
│   ├── mathnd.py    N-D vectors / matrices, Givens-rotation builder, the 3-D slicer
│   ├── data.py      datasets: PointPath | TrackedPath | Grid | Scatter (N-D, DAG nodes)
│   ├── color.py     colour model: RGB + HSV + HSL (animatable, seamless hue loops)
│   ├── interp.py    interpolators (loop curve | tracked multi-curve | grid field | scatter field)
│   ├── iso.py       isosurfaces: gyroid / Schwarz-P / Schwarz-D / Neovius + N-D slicing
│   ├── pov.py       POV-Ray function library, with which are N-D-generalizable
│   ├── spatial.py   spatial expression DSL (X, Y, Z, T + math) → ftsl `expr` strings
│   ├── sweep.py     sweep engine (rotation-minimizing frames, ribbon/tube/skin_rings, OBJ out)
│   ├── mcubes.py    adaptive marching cubes (bake a scalar field to a mesh)
│   ├── material.py  function-driven materials (waves/checker/rings/blobs, mixes)
│   ├── scene.py     Scene / Camera / Material / Texture (image skins) / geometry / Volume media (all animatable)
│   ├── transform.py per-object Transform (translate/rotate/scale/skew, animatable) → ftsl group{}; dataset inverse-map
│   ├── canvas.py    2-D canvas (motion graphics: markers, strokes)
│   ├── audio.py     procedural audio: one sample-buffer back-end → WAV (offline)
│   ├── xvideo.py    two-pass spacetime transforms (rotate/shear a 4-D block)
│   ├── ftsl_emit.py .ftsl emission
│   ├── drive.py     drivers: render a frame range → ftrace → GIF/MP4 assembly
│   └── preview.py   resident preview server (keeps ftrace + GPU context warm)
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

---

## Examples

| Script | What it makes |
|---|---|
| `ribbon_loop.py` | a seamless looping swept **ribbon** (plus a twin tube) |
| `scribble_loop.py` | a seamless looping 3-D "scribble" curve |
| `gyroid_loop.py` | a seamless looping **gyroid** isosurface |
| `pov_loop.py` | a seamless looping **POV-Ray function** isosurface |
| `gyroid_nd.py` | **higher-dimensional gyroid slices** — a randomized N-D gyroid whose hidden dimensions drift / rotate / *bloom*; gold or clear-glass; per-run output dirs (a full sub-tool, see its `--help`) |
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
