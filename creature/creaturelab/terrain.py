"""Ground that is not a plane.

P1 trained every episode on `<geom type="plane">` at z = 0, which means the policy has
never once had a foot land lower than it expected. This module is the environment half of
P1b: a MuJoCo heightfield under the creature, regenerated per episode, with a difficulty
that a curriculum can widen.

**Why a heightfield rather than a foot-placement solver.** A physics-based policy meets
rough ground by having been trained on rough ground -- see todo.md P1b, and P10's Terrain
note for why procedural full-body IK was considered and rejected as the mechanism. Nothing
here touches the controller; the controller finds out about the terrain the same way an
animal does, through its feet.

Three decisions are worth stating because they are not obvious and they outlive P1b.

**The hfield's shape is a model constant; only its data changes.** `nrow`, `ncol` and
`size` live in the compiled `MjModel`, so they are chosen once, per body, in `TerrainSpec`.
`hfield_data` is a plain float buffer that MuJoCo's collision routines read live, so a
per-episode terrain is a `numpy` assignment into it and costs no recompile. That split is
what makes per-episode regeneration affordable at all.

**The patch is big enough that an episode cannot walk off it.** `TerrainSpec.for_body`
sizes the half-extent from the fastest command times the episode length -- for a dog at
0.8 Froude for 20 s that is ~37 m, so the patch is ~90 m across. The alternative (a small
fine patch plus an out-of-bounds truncation) shortens episodes to a few seconds at the top
of the command range, which changes the task far more than a coarser cell does. Cell size
is therefore the thing that gives: `for_body` asks for `cell` body-lengths and accepts
whatever `max_cells` allows.

**Difficulty is applied in metres, not as a fraction of the elevation scale.** The scale
has to cover the steepest slope the curriculum can ask for, which over a 90 m patch is
tens of metres; rubble is twenty centimetres. So each generator declares the peak-to-peak
height it wants *in metres* at difficulty 1, and the data written to the model is
`u * difficulty * span / elevation`. Scaling `u` by difficulty directly would make a
difficulty-1 rubble field a mountain range.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The terrain classes an episode can be drawn from. `flat` is included deliberately: it is
#: terrain of difficulty 0 and keeping it in the mix is what stops a terrain-trained policy
#: from forgetting the flat ground the evaluation guard is measured on.
KINDS = ("flat", "rolling", "rubble", "steps", "slope", "stairs")

#: Name of the hfield asset and of the floor geom that references it.
HFIELD = "terrain"
FLOOR = "floor"


@dataclass(frozen=True)
class TerrainSpec:
    """The model-constant half of a terrain: the shape of the hfield asset.

    `elevation` is the height a datum of 1.0 stands for, and `base` is how much solid
    material hangs below datum 0 (MuJoCo builds a box under the field so nothing can fall
    through the world at the edges).
    """

    n: int                  # grid is n x n samples
    extent: float           # half-width in metres, in both x and y
    elevation: float        # metres per unit of hfield data
    base: float             # metres of solid below datum 0
    amp: float              # metres, peak-to-peak of the *rough* classes at difficulty 1
    body_len: float         # the body's own length scale, metres -- sets feature sizes

    @property
    def cell(self) -> float:
        """Metres between adjacent samples."""
        return 2.0 * self.extent / (self.n - 1)

    @property
    def size_attr(self) -> str:
        return f"{self.extent:g} {self.extent:g} {self.elevation:g} {self.base:g}"

    @classmethod
    def for_body(cls, body_len: float, *, amp: float = 0.35, cell: float = 0.25,
                 margin: float = 1.25, max_speed: float = 0.8, seconds: float = 20.0,
                 max_slope_deg: float = 12.0, max_cells: int = 513,
                 gravity: float = 9.81) -> TerrainSpec:
        """Size a patch for one body, in that body's own units.

        `amp` and `cell` are in withers heights, so the same numbers mean the same terrain
        on a terrier and a wolfhound -- the same dynamic-similarity argument the reward and
        the command range are built on.

        `max_slope_deg` has two separate ceilings over it, and the binding one is not the
        obvious one. The obvious one is the termination test: on a genuine slope a healthy
        standing animal *is* tilted by the grade, and `fall_tilt = 50` would call a correct
        stance a fall, so the grade must stay well under half of it. That argument set this
        to 25 -- and 25 turned out to be unwalkable, which no amount of training fixed.

        The binding ceiling is **learnability**, and it was measured rather than reasoned
        (30M-step run, 2026-08-17, scoring the trained policy on slope alone):

            grade   5deg    11deg   16deg   20deg   25deg
            return  1123     831     199      32      14
            survive  100%     84%      -       -       0%

        Unlike the rough classes, `slope` and `stairs` apply their grade to the WHOLE patch,
        so difficulty here buys a climb the animal cannot put down for the length of an
        episode -- there is no flat stretch to recover on, the way there is between stones.
        Past ~16 degrees the policy stops tracking the command altogether (`r_speed` 0.45 and
        falling) and simply fights gravity. A curriculum whose top third of range is a task
        nothing can perform does not merely waste those episodes: it spends promotions
        climbing into them and then trains there, and "ignore the command" is a behaviour
        that leaks back to the classes that *are* walkable. 12 sits just under the measured
        knee, which keeps the whole difficulty axis useful, and is comfortably under
        `fall_tilt / 2` for free.
        """
        # Froude number -> m/s is v = F * sqrt(g * L); the farthest an episode can travel is
        # that at the top of the command range for its whole length.
        reach = margin * max_speed * math.sqrt(gravity * body_len) * seconds
        extent = max(reach, 4.0 * body_len)
        want = int(math.ceil(2.0 * extent / max(1e-6, cell * body_len))) + 1
        n = int(min(max_cells, max(33, want)))
        if n % 2 == 0:
            n -= 1                             # odd, so a sample sits exactly at the origin
        elevation = max(amp * body_len, math.tan(math.radians(max_slope_deg)) * 2.0 * extent)
        return cls(n=n, extent=extent, elevation=elevation, base=max(1.0, 0.5 * body_len),
                   amp=amp * body_len, body_len=body_len)


# --------------------------------------------------------------------------- generators
# Each returns `(u, span)`: a field normalised to [0, 1] and the peak-to-peak height in
# METRES it should have at difficulty 1. See the module docstring for why the span is not
# folded into `u`.

def _lattice(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """Value noise: a (k+1)^2 random lattice, smoothstep-interpolated up to n x n."""
    k = max(1, min(k, n - 1))
    g = rng.random((k + 1, k + 1))
    s = np.linspace(0.0, float(k), n)
    i = np.minimum(np.floor(s).astype(np.intp), k - 1)
    t = s - i
    t = t * t * (3.0 - 2.0 * t)                     # C1 at the lattice lines
    row = g[i] * (1.0 - t)[:, None] + g[i + 1] * t[:, None]        # (n, k+1)
    return row[:, i] * (1.0 - t)[None, :] + row[:, i + 1] * t[None, :]


def _fbm(rng: np.random.Generator, n: int, k: int, octaves: int = 3) -> np.ndarray:
    out = np.zeros((n, n))
    amp = 1.0
    for o in range(octaves):
        out += amp * _lattice(rng, n, k << o)
        amp *= 0.5
    return _unit(out)


def _blocks(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """Piecewise-constant plateaus: a k x k random field, nearest-neighbour upsampled.

    Sharp edges on purpose. Smooth ground is something a policy can ride over with a fixed
    gait; a ledge is what makes it have to catch itself.
    """
    k = max(1, min(k, n))
    r = int(math.ceil(n / k))
    g = rng.random((k, k))
    return np.repeat(np.repeat(g, r, axis=0), r, axis=1)[:n, :n]


def _ramp(n: int, theta: float) -> np.ndarray:
    """A unit-normalised linear ramp along `theta`, rows = y and cols = x."""
    a = np.linspace(-1.0, 1.0, n)
    return _unit(math.cos(theta) * a[None, :] + math.sin(theta) * a[:, None])


def _unit(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def generate(kind: str, rng: np.random.Generator, spec: TerrainSpec) -> tuple[np.ndarray, float]:
    """Build one terrain of the given class. Returns `(u in [0,1], span in metres)`."""
    n = spec.n
    # Feature sizes are quoted in body lengths and converted to lattice counts here, so a
    # bigger animal gets proportionally bigger rubble rather than the same absolute stones.
    def lat(feature_bodies: float) -> int:
        return max(1, int(round(2.0 * spec.extent / (feature_bodies * spec.body_len))))

    if kind == "flat":
        return np.zeros((n, n)), 0.0
    if kind == "rolling":
        return _fbm(rng, n, lat(4.0), octaves=3), spec.amp
    if kind == "rubble":
        # Stone-sized plateaus riding on a long undulation: the two together are what makes
        # footing unpredictable at stride scale *and* at stance scale.
        u = 0.65 * _blocks(rng, n, lat(0.6)) + 0.35 * _lattice(rng, n, lat(3.0))
        return _unit(u), spec.amp
    if kind == "steps":
        # Normalised, like every other class: `_blocks` hands back raw uniform samples, whose
        # extremes land wherever the draw put them, so an un-normalised field would be a
        # little shorter than the amplitude it declares *and* floating a little above z = 0.
        return _unit(_blocks(rng, n, lat(2.0))), spec.amp
    # Slope and stairs take the WHOLE elevation scale, which is exactly what that scale was
    # sized for in `for_body`. The realised grade varies with the ramp's direction -- the run
    # across the patch is longer on the diagonal than along an axis, so a 25 degree axis-
    # aligned maximum becomes ~18 degrees at 45 degrees. That spread is welcome: it is free
    # variation in the one parameter the class is about.
    if kind == "slope":
        th = rng.uniform(0.0, 2.0 * math.pi)
        # A little roughness on the slope, at a tenth of the rough classes' amplitude, so
        # "uphill" never degenerates into a smooth ramp a policy can lean into and forget.
        u = _ramp(n, th) + 0.1 * (spec.amp / spec.elevation) * _lattice(rng, n, lat(3.0))
        return _unit(u), spec.elevation
    if kind == "stairs":
        th = rng.uniform(0.0, 2.0 * math.pi)
        rise = 0.12 * spec.body_len                  # step height, in body lengths
        nstep = max(1, int(round(spec.elevation / rise)))
        return np.floor(_ramp(n, th) * nstep) / nstep, spec.elevation
    raise ValueError(f"unknown terrain kind {kind!r}; expected one of {KINDS}")


# -------------------------------------------------------------------------------- patch
class Patch:
    """One env's live terrain: a view onto its `MjModel`'s hfield data, plus the queries.

    The elevation array is *not* copied here -- `grid` is a reshaped view of the model's own
    `hfield_data`, so writing terrain and reading heights are the same buffer and cannot
    drift apart. That also means one `Patch` per `MjModel`: envs that share a model share a
    terrain, which is why `VecCreatureEnv` clones the model per env when terrain is on.
    """

    def __init__(self, spec: TerrainSpec, model) -> None:
        import mujoco

        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, HFIELD)
        if hid < 0:
            raise ValueError(f"model has no hfield named {HFIELD!r} -- was it emitted with "
                             f"terrain=?")
        nrow = int(model.hfield_nrow[hid])
        ncol = int(model.hfield_ncol[hid])
        if (nrow, ncol) != (spec.n, spec.n):
            raise ValueError(f"hfield is {nrow}x{ncol} but the spec says {spec.n}x{spec.n}")
        adr = int(model.hfield_adr[hid])
        # Rows are y and columns are x -- MuJoCo stores the field row-major starting at the
        # minimum y. `tests/test_terrain.py::test_layout_matches_mujoco` pins that down
        # against MuJoCo's own ray caster rather than against this comment.
        self.grid = model.hfield_data[adr:adr + nrow * ncol].reshape(nrow, ncol)
        self.spec = spec
        self.kind = "flat"
        self.difficulty = 0.0

    # ------------------------------------------------------------------ generation
    def regenerate(self, rng: np.random.Generator, difficulty: float,
                   kinds: tuple[str, ...] = KINDS) -> None:
        """Roll a fresh terrain into the model buffer. Called once per episode reset."""
        d = float(np.clip(difficulty, 0.0, 1.0))
        kind = kinds[int(rng.integers(len(kinds)))] if d > 0.0 else "flat"
        u, span = generate(kind, rng, self.spec)
        self.grid[:] = u * (d * span / self.spec.elevation)
        self.kind, self.difficulty = kind, d

    def flatten(self) -> None:
        self.grid[:] = 0.0
        self.kind, self.difficulty = "flat", 0.0

    # ----------------------------------------------------------------------- queries
    def height_at(self, x: float, y: float) -> float:
        """Ground height directly under `(x, y)`, bilinearly interpolated.

        Called once per env per control step (for the termination test), from inside the
        physics thread, so it is deliberately scalar Python rather than a numpy call: the
        per-env arrays cannot be batched and `numpy`'s per-call overhead is larger than this
        whole function.

        Bilinear where MuJoCo's collision triangulates each cell, so the two disagree by at
        most the sag of one cell's diagonal -- millimetres at any sane cell size, against a
        termination threshold of tens of centimetres. The exact surface is not needed here;
        what is needed is that flat ground reads exactly 0, which it does.
        """
        s = self.spec
        n = s.n
        c = 2.0 * s.extent / (n - 1)
        fx = (x + s.extent) / c
        fy = (y + s.extent) / c
        fx = 0.0 if fx < 0.0 else (n - 1.0 if fx > n - 1.0 else fx)
        fy = 0.0 if fy < 0.0 else (n - 1.0 if fy > n - 1.0 else fy)
        i = int(fx)
        j = int(fy)
        if i > n - 2:
            i = n - 2
        if j > n - 2:
            j = n - 2
        tx = fx - i
        ty = fy - j
        g = self.grid
        h0 = g[j, i] * (1.0 - tx) + g[j, i + 1] * tx
        h1 = g[j + 1, i] * (1.0 - tx) + g[j + 1, i + 1] * tx
        return float(h0 * (1.0 - ty) + h1 * ty) * s.elevation

    def max_height_in(self, x0: float, x1: float, y0: float, y1: float) -> float:
        """Highest ground anywhere in a footprint. This is what a spawn must clear.

        Placing the body relative to the ground *under its root* buries a foot in any bump
        it happens to straddle, and a creature spawned intersecting the floor gets an
        enormous contact impulse on step one -- the failure `place_on_ground` was written
        for in the first place, reintroduced by terrain.
        """
        s = self.spec
        n = s.n
        c = 2.0 * s.extent / (n - 1)

        def idx(v: float, up: bool) -> int:
            f = (v + s.extent) / c
            k = int(math.ceil(f)) if up else int(math.floor(f))
            return 0 if k < 0 else (n - 1 if k > n - 1 else k)

        i0, i1 = idx(x0, False), idx(x1, True)
        j0, j1 = idx(y0, False), idx(y1, True)
        return float(self.grid[j0:j1 + 1, i0:i1 + 1].max()) * s.elevation
