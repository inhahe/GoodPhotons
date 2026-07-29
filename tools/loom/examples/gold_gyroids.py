"""Two gold gyroids in a closed room, **rotating in higher dimensions** -> a
6 s / 25 fps seamlessly looping GIF.

The animation is not a 3-D spin.  Each blob is a slice of a genuinely N-input
gyroid field (:class:`loom.SliceField`): the renderer's world is the 3-D
hyperplane ``{offset + a*e0 + b*e1 + c*e2}`` of an N-D space, and what animates is
an **N-D rotation of that hyperplane**.  Because the rotation planes touch axes
*outside* the slice, the extra coordinates feed real extra field inputs — so the
gold walls genuinely reconnect, pinch off and re-form (a topology change) instead
of merely tilting, which is all a 3-input field can ever do under an N-D rotation.

  * gyroid A: **4-D**, **cuboid** container.  Isoclinic double rotation in the
    ``(x,z)`` and ``(y,w)`` planes: ``(x,z)`` reads as a visible turn about y while
    ``(y,w)`` sweeps the slice through the 4th dimension, so it spins *and* morphs.
    Its lattice is also skewed by x/y/z factors 1 / 0.25 / 1 (the y coordinate is
    read a quarter as fast, so the cells stretch 4x vertically).
  * gyroid B: **5-D**, **spherical** container, unskewed.  Rotates in ``(x,v)`` and
    ``(z,w)`` — both planes leave the slice, so there is no rigid spin at all: the
    surface just boils and reconnects in place.

Both are `preset gold` metal.  The room is a fully closed box (floor + four walls +
ceiling) with a broad ceiling area light, so the gold picks up soft bounce light from
every side instead of the harsh single-source highlights it gets in open void.  The
camera sits *inside* the box, in front of the near wall.

Everything closes exactly at the wrap (frame 150 == frame 0), which is what makes the
GIF loop seamlessly:
  * every rotation angle is a ramp of a whole ``2*pi``, so each N-D rotation matrix
    returns to the identity it started at;
  * each lattice phase-drifts one full ``2*pi`` diagonally (``sin``/``cos`` are
    ``2*pi``-periodic, and at the wrap the rotation is the identity, so the drift's
    ``2*pi`` shifts survive the N-D mix intact);
  * the threshold breathes through one whole cycle, so the gold walls thicken and
    thin back to where they started.

Rendered with the fast RGB backward path tracer (`-mode R` + `-rgb`).

  python examples/gold_gyroids.py                  # print frame 0's .ftsl
  python examples/gold_gyroids.py --still          # one held preview frame
  python examples/gold_gyroids.py --render         # all 150 frames + the GIF
  python examples/gold_gyroids.py --render --resume  # ... continue an interrupted run

Flags: ``--cpu`` (CPU backend), ``--freq=N`` (lattice density), ``--capped`` (seal the
container cut with a flat mirrored face instead of opening into the labyrinth),
``--res=N``, ``--noise=P``, ``--t=T`` / ``--time=S`` (``--still`` only).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Sine, Mat, vec, rotations,
    Scene, Material, Camera, Isosurface, phase_drift, Light, Raw,
    SliceField, nd_grad_bound,
)

NAME = "gold_gyroids"
# 25 fps is a GIF-*exact* rate: a GIF stores its inter-frame delay in centiseconds,
# so 1/25 s == 4 cs lands on the grid and ffmpeg's `fps=` filter neither drops nor
# duplicates a frame.  (60 fps would ask for 1.67 cs, be rounded to 2, and play back
# at 50 fps — a silent 1.2x speed-up plus resampling judder.)  6 s is also long enough
# to watch the surface actually reconnect instead of flickering past.
FRAMES = 150          # 6 s at 25 fps
FPS = 25.0
RES = (420, 420)
FREQ = 17.0           # lattice frequency (rad per unit); ~5 cells across each blob
OPEN_CUT = True       # don't cap the container cut — look into the labyrinth

# --- room box (camera lives inside it) --------------------------------------
X0, X1 = -2.7, 2.7
Y0, Y1 = -1.15, 2.05
Z0, Z1 = -2.4, 5.8

# the lattice skew the brief asks for: read y a quarter as fast -> 4x taller cells
SKEW = Mat([[1.00, 0.0, 0.00],
            [0.00, 0.25, 0.00],
            [0.00, 0.0, 1.00]])

# The N-D field frames.  Slice axes are 0,1,2 = x,y,z; 3,4 = w,v are outside it.
DIM_BOX, DIM_BALL = 4, 5
# where the 3-D slice sits along the extra axes (an irrational-ish offset keeps the
# slice off the field's symmetry planes, so the morph never looks mirrored)
OFF_BOX = (0.0, 0.0, 0.0, 1.15)
OFF_BALL = (0.0, 0.0, 0.0, 0.85, 2.05)


def turn(n: int, *planes: tuple) -> Mat:
    """An n-D rotation running one whole ``2*pi`` turn in each of ``planes`` over the
    loop.  A ramp (not a sine swing) so the motion is continuous, and because every
    plane completes an integer number of turns the matrix is the identity again at
    the wrap — the seam condition for both the field and the GIF."""
    ang = phase_drift(1.0)                       # 2*pi*t
    return rotations(n, [(i, j, ang) for (i, j) in planes])


def build(clock=None, res=RES, freq: float = FREQ,
          open_cut: bool = OPEN_CUT) -> Scene:
    """The loom ``build()`` contract: a fresh Scene, no side effects at import.

    ``freq`` sets the lattice density (cells per unit); ``open_cut`` drops the caps
    ftrace otherwise seals the container cut with, so you look straight into the
    labyrinth instead of at a flat mirrored face."""
    drift = vec(phase_drift(1.0), phase_drift(1.0), phase_drift(1.0))
    breathe = Sine(cycles=1, amp=0.30, bias=0.0)

    # gyroid A — 4-D, cuboid container, skewed lattice, on the left.
    # (0,2) is a visible turn about y; (1,3) sweeps y through the 4th dimension.
    field_a = SliceField("gyroid", dim=DIM_BOX,
                         rotation=turn(DIM_BOX, (0, 2), (1, 3)), offset=OFF_BOX)
    ra = (0.88, 1.00, 0.88)
    box_gyroid = Isosurface(
        field_a, freq=freq, threshold=breathe, drift=drift,
        rotation=SKEW, placement=(-1.05, 0.12, 0.0),
        container="box", open=open_cut,
        bounds=((-ra[0], -ra[1], -ra[2]), (ra[0], ra[1], ra[2])),
        max_gradient=nd_grad_bound("gyroid", DIM_BOX, freq),
        material="gold", name="gyroid_box")

    # gyroid B — 5-D, spherical container, unskewed, on the right.
    # Both planes leave the slice, so it morphs in place with no rigid spin.
    field_b = SliceField("gyroid", dim=DIM_BALL,
                         rotation=turn(DIM_BALL, (0, 4), (2, 3)), offset=OFF_BALL)
    ball_gyroid = Isosurface(
        field_b, freq=freq, threshold=breathe, drift=drift,
        placement=(1.05, 0.12, 0.0), open=open_cut,
        container="sphere", center=(0, 0, 0), radius=0.90,
        max_gradient=nd_grad_bound("gyroid", DIM_BALL, freq),
        material="gold", name="gyroid_ball")

    scene = Scene(Camera(eye=(0.0, 0.80, 4.70), look_at=(0.0, 0.10, 0.0),
                         up=(0, 1, 0), fov_y=50, mode="R", res=res))
    scene.add(
        Raw('material "gold" { preset gold  roughness 0.045 }'),
        Material("wall",    "diffuse", reflect="rgb 0.80 0.78 0.74"),
        Material("floor",   "diffuse", reflect="rgb 0.42 0.40 0.42"),
        Material("accentL", "diffuse", reflect="rgb 0.26 0.40 0.62"),
        Material("accentR", "diffuse", reflect="rgb 0.62 0.32 0.26"),
        box_gyroid, ball_gyroid,
        # --- the closed room: floor + four walls + ceiling ------------------
        Raw(f'quad {{ origin {X0} {Y0} {Z0}  u {X1-X0} 0 0  v 0 0 {Z1-Z0}  '
            f'material "floor" }}'),                                    # floor
        Raw(f'quad {{ origin {X0} {Y1} {Z0}  u {X1-X0} 0 0  v 0 0 {Z1-Z0}  '
            f'material "wall" }}'),                                     # ceiling
        Raw(f'quad {{ origin {X0} {Y0} {Z0}  u {X1-X0} 0 0  v 0 {Y1-Y0} 0  '
            f'material "wall" }}'),                                     # back  (-z)
        Raw(f'quad {{ origin {X0} {Y0} {Z1}  u {X1-X0} 0 0  v 0 {Y1-Y0} 0  '
            f'material "wall" }}'),                                     # front (+z, behind cam)
        Raw(f'quad {{ origin {X0} {Y0} {Z0}  u 0 0 {Z1-Z0}  v 0 {Y1-Y0} 0  '
            f'material "accentL" }}'),                                  # left  (-x)
        Raw(f'quad {{ origin {X1} {Y0} {Z0}  u 0 0 {Z1-Z0}  v 0 {Y1-Y0} 0  '
            f'material "accentR" }}'),                                  # right (+x)
        # a broad ceiling panel: big and close -> soft, wrapped metal highlights
        Light("area", origin=f"-2.0 {Y1-0.03:g} -1.6", u="4.0 0 0", v="0 0 4.2",
              normal="0 -1 0", spd="preset:bb5200"),
    )
    return scene


# backwards-compatible alias (this file predates the build() viewer contract)
build_scene = build


def main() -> int:
    freq, res = FREQ, RES
    open_cut = OPEN_CUT and "--capped" not in sys.argv
    for a in sys.argv:
        if a.startswith("--freq="):
            freq = float(a[7:])
        if a.startswith("--res="):
            res = (int(a[6:]), int(a[6:]))
    scene = build(freq=freq, open_cut=open_cut, res=res)
    scene.check_cycles()

    device = "cpu" if "--cpu" in sys.argv else "gpu"

    if "--still" in sys.argv:
        from loom import render_still
        from loom.drive import default_outdir
        t, secs, hold = 0.0, 12.0, True
        for a in sys.argv:
            if a.startswith("--t="):
                t = float(a[4:])
            if a.startswith("--time="):
                secs = float(a[7:])
            if a == "--nohold":
                hold = False
        png = render_still(scene, t=t, name=NAME, outdir=default_outdir(NAME),
                           time_s=secs, interval=4.0, hold=hold,
                           extra_args=["-rgb", "-device", device])
        print(f"[still] {png}")
        return 0

    if "--render" not in sys.argv:
        from loom import Clock, Cache
        print(scene.emit(Clock.at_frame(0, FRAMES, FPS), Cache()))
        return 0

    from loom import render_range, assemble_gif_ffmpeg, assemble_mp4
    from loom.drive import default_outdir
    outdir = default_outdir(NAME)
    # A *noise* budget (not a time one) keeps every frame equally grainy, which is what
    # stops the loop from shimmering: a fixed -time would let a cheap frame come out
    # clean and an expensive one come out speckled, and the eye reads that difference
    # as flicker.  4% costs ~190 s/frame on this GPU (~8 h for the 150, measured over an
    # uncontended stretch) and survives the GIF's 256-colour quantisation; 2.5% costs ~5x
    # that for no visible gain.  `--resume` picks up from the frames already on disk.
    noise = 4.0
    for a in sys.argv:
        if a.startswith("--noise="):
            noise = float(a[8:])
    pngs = render_range(scene, FRAMES, name=NAME, outdir=outdir, fps=FPS,
                        noise=noise, interval=6.0, loop=True,
                        skip_existing="--resume" in sys.argv,
                        extra_args=["-rgb", "-device", device])
    gif = assemble_gif_ffmpeg(pngs, outdir / f"{NAME}.gif", fps=FPS)
    # ... and an MP4 beside it: a 150-frame gyroid lattice is about as GIF-hostile as
    # imagery gets (high-frequency detail defeats LZW, and 256 colours can't hold gold's
    # gradients), so the video is both smaller and truer.  The loop is identical — an
    # MP4 just has no loop flag, so looping is the player's job.
    mp4 = assemble_mp4(pngs, outdir / f"{NAME}.mp4", fps=FPS)
    print(f"[done] {len(pngs)} frames -> {gif}  +  {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
