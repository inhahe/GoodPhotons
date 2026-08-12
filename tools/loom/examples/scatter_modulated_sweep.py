"""
Loom example: a **swept mesh that morphs over time**, driven by a modulator that is
itself a *curve running in time through a scatter plot*.

The whole point is that this is one unbroken chain of DAG nodes, not a special case
anywhere:

    Phase()                      the clock's normalized phase t
      -> LoopCurve(query, t)     a point travelling around a closed 2-D path
      -> plot(point)             ScatterField: Shepard interpolation of the plot
      -> MapRange(...)           the modulator, shaped into a useful range
      -> SweptMesh.scale/turns   and into the spine's own control points

So "the sweep morphs" is not an animation baked into geometry — the geometry is
re-derived every frame from a scalar that is being *read off a scatter plot by a
moving query point*.

**The travelling wave.** Each spine control point ``k`` reads the plot at query
parameter ``t + k/nodes`` rather than at ``t``.  All nodes therefore ride the *same*
curve through the *same* plot, just phase-shifted, so a peak in the plot sweeps
around the ring as a deformation wave instead of pumping every node in unison.  That
is what makes it read as a morph rather than as a throb.

**Why the profile is lobed, not circular.**  ``SweptMesh`` twists the cross-section
about the spine tangent — and twisting a *circle* about its own centre is the
identity, so a circular tube's ``turns`` parameter is invisible.  A lobed profile
makes the twist actually show.

**Which twist channel gets modulated, and why it matters here.**  ``SweptMesh`` has
two, and on a *closed* spine they are not interchangeable:

  * ``turns`` **accumulates** around the loop, and the seam joins the last ring to
    the first vertex-for-vertex — so the total rotation must be a whole number of
    profile revolutions or the ridges arrive misregistered and the closing span is
    visibly wrung.  Measured: an integer ``turns`` closes to the polyline
    discretisation floor (~0.03), while ``turns = 0.5`` leaves a **2.68**-unit gap.
    A fractional ``1/k`` turn does not rescue it on a ``k``-lobed profile either —
    the silhouette maps onto itself but the vertex *indices* do not.
  * ``twist`` is **uniform** — every ring rolls by the same angle, nothing
    accumulates, and it closes at *any* value.

So ``helix_turns`` is a static integer and the plot drives ``twist``.  Modulating
``turns`` instead is exactly the mistake this scene originally made, and the
symptom was precisely a seam whose ridges did not line up.

Everything the modulator does is also drawn, so the mechanism is visible in the
render rather than only in the code:

  * the **scatter plot** — one sphere per sample, its *height* the sample's value;
  * the **query curve** — beads on the plate, the closed path being traversed;
  * the **read head** — the bright sphere, riding at the height the field currently
    interpolates, i.e. the modulator's live value.

Open it live in the viewer (this is the interesting way — press **play**):

  python examples/scatter_modulated_sweep.py --sidecar ../../out/scatter_sweep.json
  ftrace -viewer out/scatter_sweep.json -loom tools/loom/examples/scatter_modulated_sweep.py

Run standalone:

  python examples/scatter_modulated_sweep.py            # frame-0 .ftsl to stdout
  python examples/scatter_modulated_sweep.py --sidecar [path]   # write the sidecar
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Beads, Camera, Light, LoopCurve, MapRange, Material, Phase, PointPath, Raw,
    Scatter, Scene, Sphere, SweptMesh, vec,
)

# --------------------------------------------------------------------------
# The plot being read, and the path that reads it.
#
# Positions live in the unit square; values are what the sweep is modulated by.
# Deliberately uneven — a scatter set exists precisely for data a lattice cannot
# represent, so an even sprinkle would miss the point.
# --------------------------------------------------------------------------
PLOT_SAMPLES = [
    ((0.09, 0.14), 0.10),
    ((0.34, 0.06), 0.55),
    ((0.62, 0.17), 0.95),
    ((0.88, 0.36), 0.30),
    ((0.75, 0.63), 0.80),
    ((0.93, 0.86), 0.15),
    ((0.52, 0.78), 0.68),
    ((0.24, 0.92), 0.05),
    ((0.07, 0.58), 0.88),
    ((0.41, 0.44), 0.22),
]

# The closed path the read head travels, in the plot's own coordinates. It is routed
# to pass near high and low samples alternately, so the modulator actually swings
# instead of hovering around the plot's mean.
QUERY_POINTS = [
    (0.20, 0.16), (0.60, 0.12), (0.86, 0.42),
    (0.66, 0.70), (0.30, 0.84), (0.10, 0.50),
]

PLATE = 2.6          # world size of the unit-square plot
PLATE_Y = -1.15      # the plate sits below the sweep
PLOT_H = 0.85        # world height of a value of 1.0


def _plot_xz(u: float, v: float):
    """Unit-square plot coords -> world x/z on the plate."""
    return ((u - 0.5) * PLATE, (v - 0.5) * PLATE)


def _lobed_profile(sides: int, lobes: int, bulge: float):
    """A rounded n-lobed cross-section: r(a) = 1 + bulge*cos(lobes*a).

    Non-circular on purpose — see the module docstring on why a circular profile
    makes ``turns`` invisible.
    """
    out = []
    for j in range(sides):
        a = 2.0 * math.pi * j / sides
        r = 1.0 + bulge * math.cos(lobes * a)
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


# --------------------------------------------------------------------------
# The §F1 contract: build(clock=None, **params) -> Scene, side-effect-free at
# import, a FRESH Scene per call. Every keyword becomes a viewer control row.
# --------------------------------------------------------------------------
def build(clock=None,
          *,
          nodes: int = 7,
          ring_radius: float = 1.05,
          morph_amp: float = 0.62,
          tube_radius: float = 0.20,
          radius_swing: float = 0.55,
          helix_turns: int = 3,
          roll_swing: float = 1.0,
          profile_lobes: int = 3,
          profile_bulge: float = 0.34,
          sides: int = 30,
          count: int = 200,
          show_plot: bool = True) -> Scene:
    """A lobed profile swept along a ring whose shape is read off a scatter plot.

    ``nodes`` is how many spine control points there are — and therefore how many
    phase-shifted taps into the plot drive the travelling wave. ``morph_amp`` is how
    far the plot pushes a spine node in and out; ``radius_swing`` how much it swells
    the cross-section; ``roll_swing`` how far the reading rolls the section.
    Set ``morph_amp``/``radius_swing``/``roll_swing`` to 0 to freeze that channel
    and confirm the others in isolation.

    ``helix_turns`` is **static and integral on purpose** — see the note on the two
    twist channels in the module docstring. It is the one control here that is not
    modulated, because it cannot be without tearing the seam.

    ``sides``/``count`` are set for a clean **silhouette**, not for clean shading.
    Shading stopped needing them once ``smooth=1`` began shipping the sweep's analytic
    per-vertex normals (see :func:`loom.sweep.ring_normals`) — those are seamless at
    any density. The outline is not: it is the actual polygon, so the only thing that
    rounds it is more segments. 30x200 is where the facets stop being visible in the
    loom viewer's mesh pane at full-screen size; below about 24x160 they return along
    the limbs, and the mesh is small enough (6k verts) that buying the margin is free.
    """
    nodes = max(3, int(nodes))

    # 1. the scatter plot: scalar values at arbitrary 2-D positions.
    plot = Scatter(PLOT_SAMPLES)

    # 2. the closed query path, in the plot's own coordinate space.
    query = PointPath(QUERY_POINTS, closed=True)

    # 3. the read heads: the SAME curve through the SAME plot, phase-shifted per
    #    spine node, so a peak travels around the ring instead of pulsing it whole.
    #    LoopCurve wraps its parameter into [0, 1), so `Phase() + k/nodes` needs no
    #    modulo of its own and stays seamless across the loop point.
    t = Phase()
    heads = [LoopCurve(query, t + (k / nodes)) for k in range(nodes)]
    taps = [plot(h) for h in heads]          # ScatterField Signals — the modulators

    # 4. the spine: a ring whose every node is pushed in/out by its own tap. The plot's
    #    values run 0..1, so centre the push on 0.5 to get a signed displacement.
    spine_points = []
    for k in range(nodes):
        a = 2.0 * math.pi * k / nodes
        r = ring_radius + morph_amp * (taps[k] - 0.5)
        spine_points.append(vec(r * math.cos(a),
                                0.42 * math.sin(2.0 * a),   # static out-of-plane tilt
                                r * math.sin(a)))
    spine = PointPath(spine_points, closed=True)

    # 5. the primary modulator (the un-shifted head) also drives the cross-section's
    #    size and its roll.
    primary = taps[0]
    radius = MapRange(primary, 0.0, 1.0,
                      tube_radius * (1.0 - radius_swing),
                      tube_radius * (1.0 + radius_swing), clamp=True)
    # The ANIMATED twist channel is `twist` (uniform), never `turns` (accumulating):
    # see the module docstring. Uniform roll closes at any value, so this can be
    # driven continuously by the plot without tearing the seam.
    roll = MapRange(primary, 0.0, 1.0,
                    -roll_swing * math.pi, roll_swing * math.pi)

    swept = SweptMesh(spine,
                      _lobed_profile(int(sides), int(profile_lobes), profile_bulge),
                      count=int(count), scale=radius,
                      twist=roll, turns=int(round(helix_turns)),
                      closed_spine=True, closed_profile=True,
                      material="body", smooth=1, name="morphing_sweep")

    scene = Scene(Camera(eye=(0.0, 1.55, 5.0), look_at=(0.0, -0.15, 0.0),
                         up=(0, 1, 0), fov_y=40, mode="R", res=(520, 520)))
    scene.add(
        Material("body", "diffuse", reflect=0.80),
        Material("plate", "diffuse", reflect=0.30),
        Material("sample", "diffuse", reflect=0.72),
        Material("track", "diffuse", reflect=0.45),
        Material("head", "diffuse", reflect=0.95),
        Material("wall", "diffuse", reflect=0.76),
        swept,
    )

    if show_plot:
        # --- the plot itself, drawn: sphere height == sample value ---
        for (u, v), val in PLOT_SAMPLES:
            x, z = _plot_xz(u, v)
            scene.add(Sphere((x, PLATE_Y + 0.04 + val * PLOT_H, z), 0.052, "sample"))
            # a stem down to the plate, so the height reads as a value not a position
            scene.add(Raw(
                f'sphere {{ center {x:.4g} {PLATE_Y + 0.02:.4g} {z:.4g}  '
                f'radius 0.022  material "plate" }}'))

        # --- the query curve, as beads on the plate ---
        track = PointPath([(_plot_xz(u, v)[0], PLATE_Y + 0.03, _plot_xz(u, v)[1])
                           for (u, v) in QUERY_POINTS], closed=True)
        scene.add(Beads(track, 72, 0.017, "track"))

        # --- the read head, riding at the height the field interpolates ---
        head = heads[0]
        scene.add(Sphere(
            vec((head[0] - 0.5) * PLATE,
                PLATE_Y + 0.04 + primary * PLOT_H,
                (head[1] - 0.5) * PLATE),
            0.075, "head"))

    scene.add(
        # open-fronted room so the sweep is lit rather than self-shadowed
        Raw(f'quad {{ origin -2.6 {PLATE_Y:.4g} -2.6  u 5.2 0 0  v 0 0 5.2  material "plate" }}'),
        Raw('quad { origin -2.6  2.9 -2.6  u 5.2 0 0  v 0 0 5.2  material "wall" }'),
        Raw(f'quad {{ origin -2.6 {PLATE_Y:.4g} -2.6  u 5.2 0 0  v 0 4.05 0  material "wall" }}'),
        Raw(f'quad {{ origin -2.6 {PLATE_Y:.4g} -2.6  u 0 0 5.2  v 0 4.05 0  material "wall" }}'),
        Raw(f'quad {{ origin  2.6 {PLATE_Y:.4g} -2.6  u 0 0 5.2  v 0 4.05 0  material "wall" }}'),
        Light("area", origin="-1.2 2.86 -1.2", u="2.4 0 0", v="0 0 2.4",
              normal="0 -1 0", spd="preset:bb6500"),
    )
    return scene


FRAMES = 96          # the loop length this scene is authored for


def main() -> int:
    from loom import Cache, Clock
    if "--sidecar" in sys.argv:
        from loom.viewer import ViewerModel
        i = sys.argv.index("--sidecar")
        out = (sys.argv[i + 1] if len(sys.argv) > i + 1
               else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "scatter_modulated_sweep.json"))
        # Save WITH a clock: the sidecar's `frame` block is where the viewer gets its
        # `frames` from, and a sidecar saved without one advertises frames = 1 — which
        # leaves the viewer's transport with nowhere to advance to.
        ViewerModel(build).save_sidecar(out, Clock.at_frame(0, FRAMES))
        print(out)
        return 0
    print(build().emit(Clock.at_frame(0, FRAMES), Cache()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
