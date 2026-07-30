"""
Loom example: a **jumping jack** (jackstone) tumbling inside a closed room, its
six arms carved out of a *world-static* gyroid field.

The visual idea — and the whole reason this is a CSG scene — is that the gyroid is
**not** attached to the jack.  ``sin x cos y + sin y cos z + sin z cos x = 0`` is
evaluated in **world** space and never moves; what moves is the *arm volume* that
selects a piece of it::

    isosurface {                       # one march for all three gold arms
        intersect {
            union { sphere ×3  cylinder ×3 }     <- ONLY these carry the pose
            function { expr "gyroid(x,y,z)" }    <- NO transform: world-static
        }
    }

So as the jack spins, gold filigree flows *through* the arms: new lattice walls
appear at the leading face and dissolve at the trailing one.  Put the ``rotate``
on the ``isosurface`` (or on the ``function`` leaf) instead and the effect dies —
the pattern would just ride along rigidly.

Motion is a physical free-body tumble: spin about a body axis, whose direction
precesses about world +y on a cone.  A jack whose three rods are half glass and
half gold is *not* an inertially isotropic body, so a real one does precess.
Both rates are whole turns per loop, and the field is static, so the loop closes
the instant the **pose** repeats — no phase drift needed anywhere.

Three arms are gold (glossy ``metal:gold``) and three are SF10 glass, chosen on
opposite half-axes (``+x −y +z`` gold, ``−x +y −z`` glass) so the two halves
interlock instead of splitting the jack down the middle.

Why the intersection is cheap: a raw gyroid has ``|grad| <= 2*sqrt(3)*freq``, which
would force the sphere-tracer into steps of ``d/(2*freq)``.  Dividing the
expression by ``2*freq`` renormalises it to ``|grad| <= sqrt(3)``, so one honest
``max_gradient 2`` covers it and the march runs ~freq times faster — and the
CSG partner (a true SDF union) stays unit-Lipschitz either way.

Rendered in **mode W** (deterministic Whitted) with the ``-gi`` one-bounce gather,
which is noise-free at low ``-spp`` and — having no irradiance cache — cannot
flicker across an animation.  The light carries ``lumens``, which puts the scene
in **absolute** exposure mode: without it ftrace's p99 auto-exposure re-anchors
every frame and the GIF pumps in brightness.

Run:
  python examples/jumping_jack.py            # print frame-0 .ftsl to stdout
  python examples/jumping_jack.py --still    # one held still (pose check)
  python examples/jumping_jack.py --render   # render the looping GIF
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import Scene, Material, Camera, Light, Raw  # noqa: E402
from loom.scene import Element  # noqa: E402
from loom.ftsl_emit import EmitCtx, fmt, fmt3  # noqa: E402


# ---------------------------------------------------------------------------
# 3x3 rotation helpers (rows), matching ftrace's convention exactly
# ---------------------------------------------------------------------------
# ftrace builds a leaf/group transform as R = Rz(rz)·Ry(ry)·Rx(rx) from Euler
# angles in DEGREES, each a standard right-handed rotation (src/mesh.h,
# MeshXform::applyLinear).  We mirror that here so a pose computed in Python and
# a `rotate` emitted into the .ftsl agree.

Mat3 = Tuple[Tuple[float, float, float], ...]


def _rx(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def _ry(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _rz(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _mm(A: Mat3, B: Mat3) -> Mat3:
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _mv(M: Mat3, v: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(sum(M[i][k] * v[k] for k in range(3)) for i in range(3))


def aim_y_euler(d: Sequence[float]) -> Tuple[float, float, float]:
    """Euler angles (deg, ftrace's XYZ order) that rotate local **+y** onto unit ``d``.

    With ``ry = 0``, ``R·(0,1,0) = (−sin rz·cos rx, cos rz·cos rx, sin rx)``, so
    ``rx = asin(d.z)`` and ``rz = atan2(−d.x, d.y)``.  A cylinder is symmetric about
    its own axis, so leaving the third degree of freedom at zero costs nothing.
    """
    dz = max(-1.0, min(1.0, float(d[2])))
    rx = math.degrees(math.asin(dz))
    rz = math.degrees(math.atan2(-float(d[0]), float(d[1])))
    return (rx, 0.0, rz)


# ---------------------------------------------------------------------------
# The element
# ---------------------------------------------------------------------------

# Body-frame half-axes: (direction, "gold" | "glass").  Opposite signs alternate
# so neither material owns a whole hemisphere.
_ARMS: Tuple[Tuple[Tuple[int, int, int], str], ...] = (
    ((1, 0, 0), "gold"), ((-1, 0, 0), "glass"),
    ((0, 1, 0), "glass"), ((0, -1, 0), "gold"),
    ((0, 0, 1), "gold"), ((0, 0, -1), "glass"),
)


class Jack(Element):
    """Six ball-tipped arms on the body half-axes, tumbling, carved by a static gyroid.

    Emits **two** ``isosurface`` blocks — one per material — because an ftrace
    isosurface carries exactly one material, and folding each material's three arms
    into a single ``union`` means one sphere-trace instead of six.

    ``spin`` / ``precess`` are whole turns per loop (integers keep the loop seamless);
    ``tilt`` is the half-angle of the precession cone in degrees; ``body`` is a fixed
    body reorientation whose only job is to keep the spin axis off an arm (spinning
    exactly about a rod would leave two of the six arms stationary).

    ``gyroid_glass=False`` renders the glass arms as plain solid ball-and-rod — the
    fallback if intersecting a *dielectric* with a marched field proves too slow.
    """

    def __init__(self, *, arm: float = 1.05, ball: float = 0.34, rod: float = 0.20,
                 spin: float = 1.0, precess: float = 1.0, tilt: float = 27.0,
                 body: Sequence[float] = (34.0, 21.0, 0.0),
                 freq: float = 12.0, threshold: float = 0.0,
                 gold: str = "gold", glass: str = "glass",
                 gyroid_glass: bool = True, name: str = "jack") -> None:
        self.arm = float(arm)
        self.ball = float(ball)
        self.rod = float(rod)
        self.spin = float(spin)
        self.precess = float(precess)
        self.tilt = float(tilt)
        self.body = tuple(float(c) for c in body)
        self.freq = float(freq)
        self.threshold = float(threshold)
        self.gold = gold
        self.glass = glass
        self.gyroid_glass = bool(gyroid_glass)
        self.name = name

    # The pose is a closed form in the clock, not a Signal graph, so there is
    # nothing here for the cycle detector to walk.
    def roots(self) -> List:
        return []

    @property
    def reach(self) -> float:
        """Radius of the smallest origin-centred sphere containing the jack."""
        return self.arm + self.ball

    def pose(self, t: float) -> Mat3:
        """Body->world rotation at loop phase ``t`` in [0, 1).

        ``Ry(precession) · Rz(tilt) · Ry(spin) · B``: the last factor reorients the
        body, the spin turns it about its own axis, the tilt lays that axis over on a
        cone, and the precession walks the cone around world +y.  Every factor is the
        identity at ``t = 0`` and again at ``t = 1`` for integer turn counts.
        """
        bx, by, bz = self.body
        B = _mm(_rz(math.radians(bz)), _mm(_ry(math.radians(by)), _rx(math.radians(bx))))
        spin = _ry(2.0 * math.pi * self.spin * t)
        tilt = _rz(math.radians(self.tilt))
        prec = _ry(2.0 * math.pi * self.precess * t)
        return _mm(prec, _mm(tilt, _mm(spin, B)))

    def gyroid_expr(self) -> str:
        """The world-space gyroid, renormalised to ~unit Lipschitz (see module docstring)."""
        f = self.freq
        s = 1.0 / (2.0 * f)
        thr = self.threshold
        body = (f"sin({fmt(f)}*x)*cos({fmt(f)}*y)"
                f"+sin({fmt(f)}*y)*cos({fmt(f)}*z)"
                f"+sin({fmt(f)}*z)*cos({fmt(f)}*x)")
        if thr != 0.0:
            body = f"{body}-({fmt(thr)})"
        return f"({body})*{fmt(s)}"

    def _arm_leaves(self, M: Mat3, kind: str) -> List[str]:
        out: List[str] = []
        for d_body, k in _ARMS:
            if k != kind:
                continue
            d = _mv(M, d_body)
            tip = tuple(self.arm * c for c in d)
            mid = tuple(0.5 * self.arm * c for c in d)
            rx, ry, rz = aim_y_euler(d)
            out.append(f'            sphere {{ center {fmt3(tip)}  '
                       f'radius {fmt(self.ball)} }}')
            out.append(f'            cylinder {{ translate {fmt3(mid)}  '
                       f'rotate {fmt(rx)} {fmt(ry)} {fmt(rz)}  '
                       f'radius {fmt(self.rod)}  height {fmt(self.arm)} }}')
        return out

    def _block(self, M: Mat3, kind: str, material: str, carve: bool) -> str:
        leaves = self._arm_leaves(M, kind)
        lines = [f'{self.name}_{kind} = isosurface {{',
                 f'    material "{material}"']
        if carve:
            lines.append('    intersect {')
            lines.append('        union {')
            lines.extend('    ' + s for s in leaves)
            lines.append('        }')
            lines.append(f'        function {{ expr "{self.gyroid_expr()}" }}')
            lines.append('    }')
            # A `function` field always needs a container; an origin-centred sphere is
            # pose-independent, so it never has to be recomputed as the jack turns.
            lines.append(f'    contained_by {{ sphere {{ center 0 0 0  '
                         f'radius {fmt(self.reach)} }} }}')
            # |grad| of the renormalised gyroid is <= sqrt(3); the SDF partner is
            # unit-Lipschitz, and max(.,.) cannot exceed either bound.
            lines.append('    max_gradient 2')
        else:
            lines.append('    union {')
            lines.extend(leaves)
            lines.append('    }')
        lines.append('}')
        return "\n".join(lines)

    def emit(self, ctx: EmitCtx) -> str:
        M = self.pose(ctx.clock.t)
        return "\n".join([self._block(M, "gold", self.gold, True),
                          self._block(M, "glass", self.glass, self.gyroid_glass)])


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------

# Gyroid tuning.  `threshold` shifts the level set: the solid is `g <= threshold`, so a
# POSITIVE value grows it past the 50/50 split of `threshold = 0`.  At 0 the lattice eats
# the 0.2 m rods into loose chips; ~0.3 leaves the jack readable while the walls still
# bore right through it.  `freq = 12` puts ~5 cells across the jack — enough that the
# sweep through the static field is obvious frame to frame.
FREQ = 12.0
THRESHOLD = 0.3

# Mode-W render settings shared by the still and the sequence.  `-ev` is a
# *compensation multiplier* on the absolute sensor gain (not a stop count), and it is
# fixed here rather than auto-anchored so every frame develops identically.
#
# `-spp 8` is not about noise — mode W is deterministic at any spp — but about the
# gyroid-carved *glass*: a labyrinth of dielectric facets sends each of the 8 hero
# wavelengths somewhere different, so at 1-2 spp the crystal reads as saturated
# confetti and only settles into dispersion-coloured facets once the pixel averages
# several sub-positions.
#
# `-gi-clamp 0.15` is the firefly ceiling on one gather ray — a jack made of dielectric
# filigree is exactly the case it exists for.  It stays well above `-ambient 0.05`,
# because the clamp also caps the far-field tail an escaping gather ray returns.
WHITTED = ["-spp", "8", "-gi", "24", "-ambient", "0.05", "-gi-clamp", "0.15",
           "-whitted-grid", "3", "-ev", "11"]

# 90 frames at 25 fps = a 3.6 s loop carrying ONE turn of spin and one of precession:
# ~100 deg/s and 4 deg between frames, slow enough to actually follow a lattice wall
# entering one face of an arm and leaving the other.  (Two turns of spin — 10 deg/frame —
# was measurably too fast to read.)  25 fps divides 100, so the GIF's integer
# centisecond frame delay is exact and playback is not silently retimed.
FRAMES = 90
FPS = 25

# Closed room, camera inside it (so glass has a whole interior to refract).  It is
# deep in +z because the camera stands *inside* and still has to clear the jack's
# 1.4 m reach at a flattering focal length.
X0, X1 = -3.4, 3.4
Y0, Y1 = -2.2, 2.6
Z0, Z1 = -3.4, 6.6
WX, WY, WZ = X1 - X0, Y1 - Y0, Z1 - Z0


def _room() -> List[Raw]:
    q = lambda o, u, v, m: Raw(  # noqa: E731
        f'quad {{ origin {fmt3(o)}  u {fmt3(u)}  v {fmt3(v)}  material "{m}" }}')
    return [
        q((X0, Y0, Z0), (WX, 0, 0), (0, 0, WZ), "floor"),    # floor
        q((X0, Y1, Z0), (WX, 0, 0), (0, 0, WZ), "ceil"),     # ceiling
        q((X0, Y0, Z0), (WX, 0, 0), (0, WY, 0), "wall"),     # back
        q((X0, Y0, Z1), (WX, 0, 0), (0, WY, 0), "wall"),     # front (behind camera)
        q((X0, Y0, Z0), (0, 0, WZ), (0, WY, 0), "left"),     # left
        q((X1, Y0, Z0), (0, 0, WZ), (0, WY, 0), "right"),    # right
    ]


def build_scene(res=(480, 480), *, gyroid_glass: bool = True, spin: float = 1.0,
                freq: float = FREQ, threshold: float = THRESHOLD) -> Scene:
    scene = Scene(Camera(eye=(0.5, 0.7, 5.5), look_at=(0.0, 0.05, 0.0),
                         up=(0, 1, 0), fov_y=38, mode="W", res=res))
    scene.add(
        # `preset gold` / `preset glass:SF10` expand to exactly these (src/materials.h).
        Material("gold", "glossy", reflect="metal:gold", roughness=0.05),
        Material("glass", "dielectric", ior="glass:SF10"),
        Material("floor", "diffuse", reflect=0.30),
        Material("ceil", "diffuse", reflect=0.78),
        Material("wall", "diffuse", reflect=0.58),
        # Gentle warm/cool side walls rather than Cornell's saturated red/green: the
        # `-gi` gather really does bleed a wall's colour onto the metal, and at full
        # Cornell saturation that reads as green camouflage on the gold instead of
        # modelling it.
        Material("left", "diffuse", reflect="rgb 0.52 0.30 0.26"),
        Material("right", "diffuse", reflect="rgb 0.28 0.40 0.52"),
        Jack(gyroid_glass=gyroid_glass, spin=spin, freq=freq, threshold=threshold),
        *_room(),
        # `lumens` pins the exposure (absolute mode) so the loop cannot pump — see
        # the module docstring.
        Light("area", origin=f"-1.6 {fmt(Y1 - 0.02)} -1.6", u="3.2 0 0", v="0 0 3.2",
              normal="0 -1 0", spd="preset:bb6500", lumens=60000),
    )
    return scene


def _opt(flag: str, default: float) -> float:
    return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main() -> int:
    still = "--still" in sys.argv
    render = "--render" in sys.argv
    plain = "--plain-glass" in sys.argv
    r = int(_opt("--res", 480))
    scene = build_scene(res=(r, r), gyroid_glass=not plain,
                        freq=_opt("--freq", FREQ), threshold=_opt("--thr", THRESHOLD))

    if still:
        from loom.drive import render_still
        render_still(scene, t=0.0, name="jumping_jack", n=1,
                     interval=8.0, extra_args=WHITTED)
        return 0
    if not render:
        from loom import Clock, Cache
        print(scene.emit(Clock.at_frame(0, FRAMES), Cache()))
        return 0

    from loom import render_range
    from loom.drive import assemble_gif_ffmpeg, default_outdir
    pngs = render_range(scene, FRAMES, name="jumping_jack", fps=FPS, n=1,
                        interval=8.0, skip_existing=True, extra_args=WHITTED)
    out = default_outdir("jumping_jack")
    assemble_gif_ffmpeg(pngs, out / "jumping_jack.gif", fps=FPS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
