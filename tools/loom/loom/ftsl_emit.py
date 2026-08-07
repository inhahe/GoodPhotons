"""
Loom -> .ftsl text emission helpers.

Small, dependency-free utilities to evaluate animatable fields at a clock and
format them as ftrace scene-language tokens.  The scene model (:mod:`loom.scene`)
uses these to turn a snapshot into a ``.ftsl`` string per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .signals.core import Signal, Clock, Cache, Number, lower_axsignal
from .signals.vector import VecSignal
from .color import Color

Animatable = Union[Signal, VecSignal, Number, Sequence[Number], str]


def site_node(x):
    """The :class:`Signal` / :class:`VecSignal` a value-site sees for ``x``.

    Passes a Signal/VecSignal straight through and **lowers** an axis-typed
    node (:mod:`loom.axes` — a ``Target``, a ``CurveSample``, …) into one, so
    every scene value-site accepts the E5 influence model uniformly.  Returns
    ``None`` for a plain number / sequence / string, which the callers format
    directly.
    """
    if isinstance(x, (Signal, VecSignal)):
        return x
    return lower_axsignal(x)


@dataclass
class EmitCtx:
    """Context for one frame's emission.

    ``assets_dir`` is where file-backed elements (e.g. a swept mesh) write their
    mesh; ``tag`` disambiguates per-frame filenames.  When ``assets_dir`` is None
    (a stdout preview), file-backed elements fall back to a temp dir.

    ``mesh_format`` picks how those meshes are serialised:

    ``"obj"`` (the default)
        Wavefront OBJ text.  Portable and human-readable, so a ``.ftsl`` you keep,
        share or open in another tool has meshes anyone can read.

    ``"ftmesh"``
        The binary :mod:`loom.ftmesh` format.  Skips a float->decimal format here
        and a decimal->float parse in ftrace, which is why the **live viewer
        channel** uses it: it re-emits every frame and nothing ever reads those
        files but ftrace, microseconds later, on the same machine.  Slightly *more*
        precise than the OBJ path, which formats at ``%.6g``.

    ``mesh_sink``, when given, means **don't write the mesh to disk** — put its bytes
    in this dict, keyed by the same path the ``.ftsl`` names, and let the caller
    deliver them.  The live viewer channel does this: measured 2026-08-06, a file
    another process wrote a millisecond ago costs ~8 ms to *open*, essentially all of
    it Windows Defender's on-access scan, so the fastest mesh file is the one that
    never exists.  The emitted ``.ftsl`` text is identical either way — the path is
    still computed and still named — which is what lets the two modes share tests.
    """

    clock: Clock
    cache: Optional[Cache] = None
    assets_dir: Optional[Path] = None
    tag: str = ""
    mesh_format: str = "obj"
    mesh_sink: Optional[dict] = None

    def asset_path(self, name: str, ext: str) -> Path:
        import tempfile
        d = self.assets_dir if self.assets_dir is not None else Path(tempfile.gettempdir())
        d = Path(d)
        # Only touch the filesystem if something is actually going to land there.
        if self.mesh_sink is None:
            d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}{self.tag}.{ext}"

    def write_mesh(self, name: str, verts, faces) -> Path:
        """Serialise one indexed mesh in this context's format; return its path.

        Every file-backed mesh element goes through here, so the format choice is
        made in exactly one place and the two element types cannot drift apart.
        With a ``mesh_sink`` the bytes go into the dict instead of to the path, but
        the path is still what the caller gets and still what the scene text names.
        """
        sink = self.mesh_sink
        if self.mesh_format == "ftmesh":
            path = self.asset_path(name, "ftmesh")
            if sink is None:
                from .ftmesh import write_ftmesh
                write_ftmesh(path, verts, faces)
            else:
                from .ftmesh import encode
                sink[path.as_posix()] = encode(verts, faces)
        elif self.mesh_format == "obj":
            path = self.asset_path(name, "obj")
            if sink is None:
                from .sweep import write_obj
                write_obj(path, verts, faces)
            else:
                from .sweep import encode_obj
                sink[path.as_posix()] = encode_obj(verts, faces).encode("utf-8")
        else:
            raise ValueError(
                f"EmitCtx.mesh_format must be 'obj' or 'ftmesh', got {self.mesh_format!r}")
        return path


def num(x: Union[Signal, Number], clock: Clock, cache: Optional[Cache] = None) -> float:
    n = site_node(x)
    if n is None:
        return float(x)
    if isinstance(n, VecSignal):
        raise ValueError(
            f"a scalar value-site got a {n.dim}-vector; pick a component "
            f"(e.g. .comp(1) on an axis node)")
    return n.at(clock, cache)


def vecn(x, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, ...]:
    n = site_node(x)
    if isinstance(n, VecSignal):
        return n.at(clock, cache)
    if isinstance(n, Signal):
        raise ValueError("a vector value-site got a scalar")
    return tuple(num(c, clock, cache) for c in x)


def vec3(x, clock: Clock, cache: Optional[Cache] = None) -> Tuple[float, float, float]:
    t = vecn(x, clock, cache)
    if len(t) < 3:
        raise ValueError("expected a >=3-D vector")
    return (t[0], t[1], t[2])


def fmt(v: float) -> str:
    return f"{v:.6g}"


def fmt3(t: Sequence[float]) -> str:
    return " ".join(fmt(float(c)) for c in t[:3])


def value_token(v, clock: Clock, cache: Optional[Cache] = None) -> str:
    """Format an arbitrary animatable/scalar/vector/string property value."""
    if isinstance(v, str):
        return v
    if isinstance(v, Color):
        return v.token(clock, cache)            # "rgb r g b" or "hsv h s v"
    n = site_node(v)                            # lowers an axis-typed node
    if isinstance(n, VecSignal):
        return fmt3(n.at(clock, cache))
    if isinstance(n, Signal):
        return fmt(n.at(clock, cache))
    if isinstance(v, (list, tuple)):
        return " ".join(fmt(num(c, clock, cache)) for c in v)
    return fmt(float(v))
