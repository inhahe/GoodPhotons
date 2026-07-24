"""
Loom animation go-between (roadmap §E2, slice 1) — N-D curve → scene variables.

``.ftsl`` cannot express animation, so when an interactive editor drives *arbitrary*
scene variables from a curve (not just the camera pose, which ftrace's native
``camera_curve`` expands on its own), **loom** must be the per-frame driver: it
holds the binding config, receives the editor's current sampled curve values at a
scrub position, and emits that one frame's ``.ftsl``.  This module is that
go-between's **authoritative in-memory model + on-disk projection** (E2 OPEN
Q1/Q2: the config lives in a loom struct, with a serialized sidecar for the
round-trip with the separate editor process; the go-between is loom itself).

Two channels, per the design (kept distinct):

* **(a) whole-video config** — a :class:`CurveDrive`: the curve's dimension count,
  the static starting control points, and the channel→scene-variable *bindings*
  (which sampled channel drives which variable, and how).  Authored once for the
  whole animation; persisted as a JSON **sidecar** (:meth:`CurveDrive.save` /
  :meth:`~CurveDrive.load`, atomic write) that the editor seeds from and writes
  back ("scene proposes, editor disposes" — associations round-trip).
* **(b) per-frame live values** — while the editor scrubs, it pushes the *current
  sampled channel values*; :meth:`CurveDrive.apply` fans them out to concrete
  scene-variable values.  This is a transient per-frame flow, separate from (a).

Value fan-out reuses the E5 influence model (:mod:`loom.axes`): each binding is a
pin/mod edge with a gain into a target of a declared quantity *kind* (additive /
gain / bipolar), so several channels can co-drive one variable with the
domain-correct accumulate operator.  (E5 "unifies E2/E4".)

Control-point **modulation is out** for the editor (the design: the editor already
owns the time axis via the points, so time-varying points would introduce a second
time axis) — ``points`` is a *static* starting array here.  Sampling below is a
uniform Catmull-Rom for loom-side preview/tests; during a live session ftrace's
editor is the sampling authority and supplies the values to :meth:`apply`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .axes import AConst, Binding, Target, ADDITIVE, GAIN, BIPOLAR

# Authoring modes (chosen up front; for most render modes the distinction is free).
MODE_FLYBY = "flyby"          # sampled channels collapse to camera pose at time t
MODE_ANIMATION = "animation"  # any sampled channel maps to any scene variable

_KINDS = (ADDITIVE, GAIN, BIPOLAR)
_MODES = ("pin", "mod")


@dataclass
class ChannelBinding:
    """One channel→scene-variable edge (E2 channel-a association).

    ``channel`` indexes a sampled curve dimension; ``target`` names the scene
    variable it drives.  ``mode``/``gain``/``kind`` are the E5 edge attributes:
    ``mode='pin'`` replaces (last-write-wins, ``gain`` blends), ``mode='mod'``
    accumulates toward the target's neutral for its quantity ``kind``.
    """

    channel: int
    target: str
    mode: str = "pin"
    gain: float = 1.0
    kind: str = ADDITIVE

    def __post_init__(self) -> None:
        self.channel = int(self.channel)
        self.target = str(self.target)
        if self.channel < 0:
            raise ValueError(f"ChannelBinding channel must be >= 0, got {self.channel}")
        if self.mode not in _MODES:
            raise ValueError(f"ChannelBinding mode must be one of {_MODES}, got {self.mode!r}")
        if self.kind not in _KINDS:
            raise ValueError(f"ChannelBinding kind must be one of {_KINDS}, got {self.kind!r}")
        self.gain = float(self.gain)

    def to_dict(self) -> dict:
        return {"channel": self.channel, "target": self.target,
                "mode": self.mode, "gain": self.gain, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelBinding":
        return cls(channel=d["channel"], target=d["target"],
                   mode=d.get("mode", "pin"), gain=d.get("gain", 1.0),
                   kind=d.get("kind", ADDITIVE))


# Sidecar schema version — bump on any breaking change to the on-disk shape.
SIDECAR_VERSION = 1


class CurveDrive:
    """The E2 channel-a config: an N-D curve + channel→scene-variable bindings.

    ``dims`` is the curve's dimension count; ``points`` is the static starting
    array of control points, each a ``dims``-tuple; ``bindings`` are the
    :class:`ChannelBinding` associations.  ``mode`` is :data:`MODE_ANIMATION`
    (any channel → any variable) or :data:`MODE_FLYBY` (channels collapse to the
    camera pose).  The whole thing serialises to a JSON sidecar for the editor
    round-trip.
    """

    def __init__(self, dims: int, points: Sequence[Sequence[float]],
                 bindings: Sequence[ChannelBinding] = (), *,
                 mode: str = MODE_ANIMATION, closed: bool = False,
                 name: str = "drive") -> None:
        dims = int(dims)
        if dims < 1:
            raise ValueError("CurveDrive needs dims >= 1")
        pts = [tuple(float(c) for c in p) for p in points]
        if len(pts) < 2:
            raise ValueError("CurveDrive needs >= 2 control points")
        for i, p in enumerate(pts):
            if len(p) != dims:
                raise ValueError(
                    f"control point {i} has {len(p)} coords, expected dims={dims}")
        if mode not in (MODE_FLYBY, MODE_ANIMATION):
            raise ValueError(f"mode must be {MODE_FLYBY!r} or {MODE_ANIMATION!r}")
        binds = list(bindings)
        for b in binds:
            if b.channel >= dims:
                raise ValueError(
                    f"binding channel {b.channel} >= dims {dims} (target {b.target!r})")
        self.dims = dims
        self.points = pts
        self.bindings = binds
        self.mode = mode
        self.closed = bool(closed)
        self.name = str(name)

    # ---- targets -----------------------------------------------------------
    def targets(self) -> List[str]:
        """Distinct scene-variable names driven, in first-seen order."""
        seen: List[str] = []
        for b in self.bindings:
            if b.target not in seen:
                seen.append(b.target)
        return seen

    # ---- sidecar (channel-a on-disk projection) ----------------------------
    def to_dict(self) -> dict:
        return {
            "version": SIDECAR_VERSION,
            "name": self.name,
            "mode": self.mode,
            "dims": self.dims,
            "closed": self.closed,
            "points": [list(p) for p in self.points],
            "bindings": [b.to_dict() for b in self.bindings],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CurveDrive":
        ver = d.get("version", SIDECAR_VERSION)
        if ver != SIDECAR_VERSION:
            raise ValueError(
                f"CurveDrive sidecar version {ver} != supported {SIDECAR_VERSION}")
        return cls(
            dims=d["dims"],
            points=d["points"],
            bindings=[ChannelBinding.from_dict(b) for b in d.get("bindings", [])],
            mode=d.get("mode", MODE_ANIMATION),
            closed=d.get("closed", False),
            name=d.get("name", "drive"),
        )

    def save(self, path: str) -> None:
        """Atomically write the sidecar JSON (temp file + ``os.replace``) so the
        editor never reads a half-written config."""
        text = json.dumps(self.to_dict(), indent=2)
        d = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> "CurveDrive":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    # ---- sampling (loom-side preview; ftrace is the live authority) --------
    def sample(self, t: float) -> Tuple[float, ...]:
        """Uniform Catmull-Rom sample of the control points at ``t`` in ``[0, 1]``.

        Returns a ``dims``-tuple.  For a ``closed`` curve the points wrap; for an
        open curve the endpoints are clamped (duplicated phantom points).  This
        mirrors the curve the editor draws so loom-side previews match; during a
        live session the editor supplies the sampled values directly.
        """
        pts = self.points
        n = len(pts)
        if n == 2 and not self.closed:
            u = _clamp01(t)
            return tuple(a + (b - a) * u for a, b in zip(pts[0], pts[1]))
        # segment index + local u
        if self.closed:
            seg_count = n
        else:
            seg_count = n - 1
        tt = (t % 1.0) if self.closed else _clamp01(t)
        f = tt * seg_count
        i = int(f)
        if i >= seg_count:
            i = seg_count - 1
        u = f - i

        def P(k: int) -> Sequence[float]:
            if self.closed:
                return pts[k % n]
            return pts[min(max(k, 0), n - 1)]

        p0, p1, p2, p3 = P(i - 1), P(i), P(i + 1), P(i + 2)
        return tuple(_catmull(a, b, c, d, u)
                     for a, b, c, d in zip(p0, p1, p2, p3))

    # ---- binding application (channel-b fan-out) ---------------------------
    def apply(self, values: Sequence[float],
              bases: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Fan ``values`` (one per channel, e.g. from a scrub) out to scene
        variables via the bindings, returning ``{target: resolved value}``.

        Multiple channels driving one target compose through an E5
        :class:`~loom.axes.Target` of the binding's ``kind`` (bindings applied in
        order); ``bases`` supplies an optional authored base per target (else the
        kind's neutral element).
        """
        if len(values) < self.dims:
            raise ValueError(
                f"apply needs >= {self.dims} channel values, got {len(values)}")
        bases = bases or {}
        # group bindings by target, preserving order
        grouped: Dict[str, List[ChannelBinding]] = {}
        for b in self.bindings:
            grouped.setdefault(b.target, []).append(b)
        out: Dict[str, float] = {}
        for target, binds in grouped.items():
            kind = binds[0].kind
            edges = [Binding(AConst(float(values[b.channel])), b.mode, b.gain)
                     for b in binds]
            base = bases.get(target)
            tgt = Target(kind, edges, base=None if base is None else float(base))
            out[target] = float(tgt.eval())
        return out

    def frame(self, t: float,
              bases: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Convenience: :meth:`sample` at ``t`` then :meth:`apply` — the loom-side
        equivalent of one editor scrub frame (preview/testing)."""
        return self.apply(self.sample(t), bases)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _catmull(p0: float, p1: float, p2: float, p3: float, u: float) -> float:
    """Uniform Catmull-Rom on one component (tension ½)."""
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


__all__ = [
    "CurveDrive", "ChannelBinding",
    "MODE_FLYBY", "MODE_ANIMATION", "SIDECAR_VERSION",
]
