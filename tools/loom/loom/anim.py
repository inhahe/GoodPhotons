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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .atomicio import write_atomic
from .axes import AConst, Binding, Target, ADDITIVE, GAIN, BIPOLAR
from .signals.core import Signal, Clock, Cache

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
        """Atomically write the sidecar JSON (temp file + ``os.replace``, see
        :mod:`loom.atomicio`) so the editor never reads a half-written config."""
        write_atomic(path, json.dumps(self.to_dict(), indent=2))

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


# ===========================================================================
# Slice 2 — named animatable slots + the editor↔loom live-value channel
# ===========================================================================

class Slot(Signal):
    """A **named animatable value-site**: a Signal leaf holding a mutable current
    value that a :class:`CurveDrive` / the editor sets per frame.

    Drop a ``Slot`` anywhere a scene parameter accepts a :class:`Signal`
    (a material roughness, an isosurface threshold via a signal-valued param, a
    transform field, …).  Because it *is* a Signal, the scene's ``roots()``/
    ``walk`` machinery discovers it and ``emit`` bakes its current value each
    frame — so binding by name (option **b**, loom's ``RefSignal``-style handle)
    needs no change to the emit path.  This is the one controlled escape from
    clock-purity: the value is pushed by the live channel, not computed from the
    clock, so always emit each scrub frame with a **fresh** :class:`Cache` (the
    :class:`SceneDriver` does).  ``default`` doubles as the authored base a
    ``mod`` binding accumulates on top of.
    """

    def __init__(self, name: str, default: float = 0.0) -> None:
        super().__init__()
        self.name = str(name)
        self.default = float(default)
        self.value = float(default)

    def set(self, v: float) -> None:
        self.value = float(v)

    def reset(self) -> None:
        self.value = self.default

    def _eval(self, clock: Clock, cache: Optional[Cache]) -> float:
        return self.value


def collect_slots(scene) -> Dict[str, List[Slot]]:
    """Walk every modulator in ``scene`` and group its :class:`Slot`s by name."""
    from .scene import element_roots  # lazy: scene.py is heavy / avoid import cycle
    from .signals.core import walk
    found: Dict[str, List[Slot]] = {}
    for el in scene._all_elements():
        for r in element_roots(el):
            for n in walk(r):
                if isinstance(n, Slot):
                    found.setdefault(n.name, []).append(n)
    return found


class SceneDriver:
    """Bind a :class:`CurveDrive`'s fan-out to a :class:`Scene`'s named
    :class:`Slot`s and emit one ``.ftsl`` per scrub frame.

    Each driven ``target`` sets every same-named ``Slot``; a target with no slot
    is ignored unless ``strict`` (then construction raises, so a typo'd binding
    fails loudly).  A slot's ``default`` is used as the ``mod`` base for its
    target unless overridden in ``bases``.
    """

    def __init__(self, scene, drive: CurveDrive, *,
                 bases: Optional[Dict[str, float]] = None,
                 strict: bool = False) -> None:
        self.scene = scene
        self.drive = drive
        self.slots = collect_slots(scene)
        # authored base per target = slot default, overridable by `bases`
        self.bases: Dict[str, float] = {}
        for target, slots in self.slots.items():
            self.bases[target] = slots[0].default
        if bases:
            self.bases.update(bases)
        if strict:
            missing = [t for t in drive.targets() if t not in self.slots]
            if missing:
                raise ValueError(f"CurveDrive targets have no Slot in the scene: {missing}")

    def set_values(self, values: Sequence[float]) -> Dict[str, float]:
        """Fan ``values`` out through the drive and push each into its slots.
        Returns the resolved ``{target: value}`` map."""
        resolved = self.drive.apply(values, self.bases)
        for target, v in resolved.items():
            for slot in self.slots.get(target, ()):
                slot.set(v)
        return resolved

    def emit_frame(self, values: Sequence[float], clock: Clock, *,
                   assets_dir=None, tag: str = "") -> str:
        """Push ``values`` then emit the scene at ``clock`` with a **fresh**
        cache (slot values are mutable state outside the clock)."""
        self.set_values(values)
        return self.scene.emit(clock, Cache(), assets_dir=assets_dir, tag=tag)


class LiveSession:
    """The loom side of the editor↔loom **live-value channel** (E2 channel-b).

    Processes one editor message (a ``dict``) and returns an ack ``dict``; the
    transport (a newline-delimited-JSON stdio pipe — :func:`serve_live` — or a
    socket) is separate, so the protocol is unit-testable with in-memory data.
    Messages (``cmd``):

    * ``frame`` — ``{values:[…] | t:float, frame:int, frames:int, out:"path.ftsl"}``:
      set the channel values (or ``sample`` at ``t``), emit that frame's
      ``.ftsl`` to ``out``; ack ``{ok, out, targets}``.
    * ``config`` — ack ``{ok, config}`` (the sidecar dict, to seed the editor).
    * ``slots`` — ack ``{ok, slots:{name: default}}`` — the **menu** of bindable
      scene variables the scene actually exposes.  "Scene proposes, editor
      disposes" needs both halves: ``config`` is what the scene proposes, this is
      what it *could* propose, so the editor can offer a pick-list instead of
      making the user type a target name into a GDI panel.
    * ``bindings`` — ``{bindings:[…]}``: replace the associations (editor
      disposes); ack ``{ok}``.
    * ``points`` — ``{points:[…]}``: replace the static control points; ack ``{ok}``.
    * ``dims`` — ``{dims:N}``: re-dimension the curve (the editor may change the
      dimension count too).  Points are padded with zeros / truncated and any
      binding left pointing past the new last channel is dropped; ack
      ``{ok, dims, dropped:[…]}``.
    * ``save`` — ``{path}`` (optional; defaults to the sidecar this session was
      seeded from): persist the sidecar; ack ``{ok, path}``.
    * ``quit`` — stop the serve loop; ack ``{ok, bye:true}``.
    """

    def __init__(self, driver: SceneDriver, *,
                 config_path: Optional[str] = None) -> None:
        self.driver = driver
        self.drive = driver.drive
        # Where a pathless `save` writes: the sidecar this session was seeded from.
        # The editor asks to save the config it is editing without having to know
        # (or re-send) the path loom was launched with.
        self.config_path = config_path

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        try:
            if cmd == "frame":
                return self._frame(msg)
            if cmd == "config":
                return {"ok": True, "config": self.drive.to_dict()}
            if cmd == "slots":
                return {"ok": True,
                        "slots": {name: slots[0].default
                                  for name, slots in sorted(self.driver.slots.items())}}
            if cmd == "dims":
                return self._dims(msg)
            if cmd == "bindings":
                self.drive.bindings = [ChannelBinding.from_dict(b)
                                       for b in msg["bindings"]]
                for b in self.drive.bindings:
                    if b.channel >= self.drive.dims:
                        raise ValueError(f"binding channel {b.channel} >= dims")
                return {"ok": True}
            if cmd == "points":
                pts = [tuple(float(c) for c in p) for p in msg["points"]]
                for p in pts:
                    if len(p) != self.drive.dims:
                        raise ValueError("point dim mismatch")
                if len(pts) < 2:
                    raise ValueError("need >= 2 points")
                self.drive.points = pts
                return {"ok": True}
            if cmd == "save":
                path = msg.get("path") or self.config_path
                if not path:
                    raise ValueError("save needs a `path` (no sidecar path for this session)")
                self.drive.save(path)
                return {"ok": True, "path": path}
            if cmd == "quit":
                return {"ok": True, "bye": True}
            return {"ok": False, "error": f"unknown cmd {cmd!r}"}
        except Exception as e:  # report, don't crash the pipe loop
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _dims(self, msg: dict) -> dict:
        """Re-dimension the curve in place (``dims`` command).

        Growing pads every control point with zeros — a *neutral* new channel, so
        the curve the editor is looking at does not move when a dimension is added.
        Shrinking truncates and drops the bindings that would now point past the
        last channel; they are named in the ack rather than silently discarded, so
        the editor can tell the user what its edit cost.
        """
        dims = int(msg["dims"])
        if dims < 1:
            raise ValueError("dims must be >= 1")
        old = self.drive.dims
        if dims != old:
            self.drive.points = [tuple(list(p[:dims]) + [0.0] * max(0, dims - len(p)))
                                 for p in self.drive.points]
            self.drive.dims = dims
        dropped = [b.to_dict() for b in self.drive.bindings if b.channel >= dims]
        if dropped:
            self.drive.bindings = [b for b in self.drive.bindings if b.channel < dims]
        return {"ok": True, "dims": dims, "dropped": dropped}

    def _frame(self, msg: dict) -> dict:
        if "values" in msg:
            values = [float(v) for v in msg["values"]]
        else:
            values = list(self.drive.sample(float(msg.get("t", 0.0))))
        k = int(msg.get("frame", 0))
        frames = int(msg.get("frames", 1))
        clock = Clock.at_frame(k, frames, loop=not msg.get("open", False))
        text = self.driver.emit_frame(values, clock, tag=f"{k:04d}")
        out = msg.get("out")
        if out:
            _atomic_write_text(out, text)
        resolved = self.driver.set_values(values)  # for the ack (already applied)
        return {"ok": True, "out": out, "frame": k, "targets": resolved}


def serve_live(session: LiveSession, in_stream, out_stream) -> None:
    """Run the newline-delimited-JSON message loop over the given streams
    (default the process's ``stdin``/``stdout``).  Reads one JSON object per
    line, dispatches to ``session.handle``, writes one JSON ack per line, and
    stops after a ``quit`` (or EOF).  Mirrors :class:`loom.PreviewServer`'s
    one-message-per-line stdio convention, in the editor→loom direction."""
    for line in in_stream:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            ack = {"ok": False, "error": f"bad json: {e}"}
        else:
            ack = session.handle(msg)
        out_stream.write(json.dumps(ack) + "\n")
        out_stream.flush()
        if ack.get("bye"):
            break


def _atomic_write_text(path: str, text: str) -> None:
    write_atomic(path, text)


# ===========================================================================
# CLI — the resident go-between an editor spawns  (E2 slice 3)
# ===========================================================================

def default_drive(scene, *, dims: int = 3, name: str = "drive") -> "CurveDrive":
    """A minimal starting :class:`CurveDrive` for a scene that proposes none.

    Two control points on the x axis and no bindings: enough for the editor to be
    a valid live session from the first frame, and it is immediately replaced by
    whatever the user records — the *scene's* own proposal (a module-level
    ``drive`` / ``DRIVE``, or a sidecar) always wins over this.
    """
    pts = [tuple([0.0] * dims), tuple([1.0] + [0.0] * (dims - 1))]
    return CurveDrive(dims, pts, [], name=name)


def _scene_drive(module) -> Optional["CurveDrive"]:
    """The drive a scene module *proposes*, if any: a module-level ``drive`` or
    ``DRIVE`` that is (or returns) a :class:`CurveDrive`."""
    for attr in ("drive", "DRIVE", "curve_drive"):
        obj = getattr(module, attr, None)
        if callable(obj) and not isinstance(obj, CurveDrive):
            try:
                obj = obj()
            except TypeError:
                continue
        if isinstance(obj, CurveDrive):
            return obj
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m loom.anim <scene.py> [--config sidecar.json]`` — serve one live
    session on stdin/stdout for an editor to drive (E2 channel-b), seeded from the
    sidecar (channel-a) when one is given.

    The editor (ftrace's curve editor, ``-anim``) spawns this, asks for ``config``
    and ``slots``, then pushes a ``frame`` per scrub position and gets a ``.ftsl``
    back.  Config precedence is *sidecar file* → *scene proposal* → a default, so
    an existing edit always survives a re-launch.
    """
    import argparse
    import sys

    from .viewer import build_scene, load_build

    ap = argparse.ArgumentParser(
        prog="python -m loom.anim",
        description="serve an E2 live animation session (editor -> loom -> .ftsl)")
    ap.add_argument("scene", help="loom scene file exposing build()")
    ap.add_argument("--config", default=None,
                    help="CurveDrive sidecar JSON to seed from (and save back to)")
    ap.add_argument("--func", default="build", help="contract function (default build)")
    ap.add_argument("--dims", type=int, default=3,
                    help="dimension count for a default drive (ignored when seeded)")
    ap.add_argument("--strict", action="store_true",
                    help="fail on a binding whose target has no Slot in the scene")
    args = ap.parse_args(list(argv) if argv is not None else None)

    build = load_build(args.scene, func=args.func)
    scene = build_scene(build)
    module = sys.modules.get(build.__module__)

    drive = None
    if args.config and os.path.exists(args.config):
        drive = CurveDrive.load(args.config)
    if drive is None and module is not None:
        drive = _scene_drive(module)
    if drive is None:
        drive = default_drive(scene, dims=args.dims)

    session = LiveSession(SceneDriver(scene, drive, strict=args.strict),
                          config_path=args.config)
    serve_live(session, sys.stdin, sys.stdout)
    return 0


__all__ = [
    "CurveDrive", "ChannelBinding",
    "MODE_FLYBY", "MODE_ANIMATION", "SIDECAR_VERSION",
    "Slot", "collect_slots", "SceneDriver", "LiveSession", "serve_live",
    "default_drive", "main",
]


if __name__ == "__main__":  # pragma: no cover
    # `python -m loom.anim` runs THIS FILE a second time, under the name `__main__`,
    # so `__main__.Slot` and `loom.anim.Slot` become two distinct classes. A scene
    # does `from loom.anim import Slot` and gets the canonical one; `collect_slots`
    # running out of `__main__` tests `isinstance(n, __main__.Slot)` and matches
    # nothing. Every binding then acks "ok" while changing precisely nothing — a
    # silent no-op, which is exactly how it presented (the editor's live preview
    # never moved). Delegate to the imported module so the code that actually runs
    # is the canonical one and both halves agree on class identity.
    from loom.anim import main as _main
    raise SystemExit(_main())
