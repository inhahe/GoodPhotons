"""
Loom native-viewer support (roadmap §F, slice **F1**) — the loom↔viewer data
contract.

The §F native viewer (Dear ImGui + ImPlot + imnodes on ftrace's ``-raster-gpu``)
is a **C++** process; loom is Python.  So — exactly as the architecture locks it —
loom does not share objects in-process with the viewer: it exposes a scene through
a **``build()`` load contract** and hands the viewer a **JSON introspection
sidecar** enumerating the scene's objects (curves / SweptMeshes / isosurfaces /
scatter+grid fields), the datasets they reference, and the modulator DAG.  The
viewer reads that sidecar to populate its object list / panes, and drives frames
over the existing ``PreviewServer`` ↔ ``ftrace -serve`` stdio pipe.

This module is the loom side of F1:

* :func:`load_build` — import a loom scene file and return its ``build`` callable.
* :class:`ViewerModel` — wraps a ``build`` callable + params; re-evaluates the
  scene at any clock and produces the introspection sidecar.
* :func:`introspect` — walk a :class:`~loom.scene.Scene` into the sidecar dict.

**The ``build()`` load contract.**  A viewable loom file exposes a module-level
function::

    def build(clock=None, **params) -> Scene: ...

* It **returns a fresh** :class:`~loom.scene.Scene` each call, so the viewer can
  re-derive geometry live (scrub time, change a param, re-tessellate) — the file
  must **not** expose a module-level ``scene`` object baked at import.
* It is **side-effect-free at import**: importing the file must not render, emit,
  open windows or write files; all of that happens only when ``build`` is called
  (and even then geometry is derived, not rendered — the viewer renders).
* ``clock`` is an optional :class:`~loom.signals.core.Clock` (the viewer passes the
  scrub frame; ``None`` ⇒ a static frame-0 clock).  Extra ``**params`` are
  surfaced by the viewer as UI controls; a build that declares keyword params with
  defaults advertises them (see :meth:`ViewerModel.params`).
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional

from .signals.core import Clock, walk
from .data import PointPath, TrackedPath, Grid, Scatter
from .interp import eval_curve

SIDECAR_VERSION = 1

# how many points to sample along a curve for the viewer's display polyline
_POLYLINE_SAMPLES = 96

# dataset classes we surface as first-class inspectable data
_DATASET_KINDS = {
    PointPath: "path",
    TrackedPath: "tracked_path",
    Grid: "grid",
    Scatter: "scatter",
}


# ===========================================================================
# the build() load contract
# ===========================================================================

def load_build(path: str, *, func: str = "build") -> Callable[..., Any]:
    """Import the loom scene file at ``path`` and return its ``build`` callable.

    The module is executed (top level must be side-effect-free per the contract);
    ``func`` (default ``"build"``) must be a module-level callable.  Raises
    ``ImportError`` if the file can't be loaded and ``AttributeError``/``TypeError``
    if the contract function is missing or not callable.
    """
    path = os.path.abspath(path)
    modname = "loom_scene_" + str(abs(hash(path)))
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a Python module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(modname, None)
        raise
    fn = getattr(module, func, None)
    if fn is None:
        raise AttributeError(
            f"{path!r} has no module-level {func!r} — a viewable loom file must "
            f"define `def {func}(clock=None, **params) -> Scene`")
    if not callable(fn):
        raise TypeError(f"{func!r} in {path!r} is not callable")
    return fn


def _accepts_clock(build: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(build)
    except (ValueError, TypeError):
        return True  # builtins / C funcs: assume it takes it, let the call decide
    for p in sig.parameters.values():
        if p.name == "clock" or p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def build_scene(build: Callable[..., Any], clock: Optional[Clock] = None,
                **params: Any):
    """Call a ``build`` contract function → a fresh :class:`~loom.scene.Scene`.

    ``clock`` is passed only if the function's signature accepts it (a ``build``
    that ignores time need not declare the kwarg); extra ``params`` pass through.
    """
    kw = dict(params)
    if _accepts_clock(build):
        kw["clock"] = clock
    return build(**kw)


# ===========================================================================
# introspection → the JSON sidecar
# ===========================================================================

def _dataset_kind(obj: Any) -> Optional[str]:
    for cls, kind in _DATASET_KINDS.items():
        if isinstance(obj, cls):
            return kind
    return None


def _curve_geometry(path: Any, clock: Optional[Clock]) -> Dict[str, Any]:
    """Evaluate a :class:`PointPath`'s control points at ``clock`` and sample a
    display **polyline** along the interpolated (loop) curve — the actual N-D
    geometry the viewer's curve pane draws (F2).  Returns ``control_points``
    (one N-D coord list per control point) and ``polyline`` (``_POLYLINE_SAMPLES``
    sampled N-D points; a closed curve repeats the first point to visibly close)."""
    clk = clock if clock is not None else Clock(t=0.0, frame=0, frames=1, fps=1.0)
    cps = [list(p.at(clk)) for p in path.points]
    tup = [tuple(c) for c in cps]
    n = _POLYLINE_SAMPLES
    # eval_curve wraps u to [0,1), so sample u = k/n (never hitting 1.0) for both
    # cases; a closed curve then repeats the first point to visibly close the loop.
    poly = [list(eval_curve(tup, k / n, path.closed)) for k in range(n)]
    if path.closed:
        poly.append(list(poly[0]))
    return {"control_points": cps, "polyline": poly}


def _track_channels(tp: Any, clock: Optional[Clock]) -> List[Dict[str, Any]]:
    """Sample each of a :class:`TrackedPath`'s tacked-on tracks along the **same**
    seamless curve parameter the display polyline uses, so the viewer's strip charts
    (F3) line up sample-for-sample with the 3-D curve.  Each channel record carries
    the track ``name``, its ``dim``, whether it was authored ``scalar``, and a
    ``samples`` list (one ``dim``-vector per polyline sample — scalar tracks give
    1-vectors).  A closed path repeats the first sample to match the closed polyline."""
    clk = clock if clock is not None else Clock(t=0.0, frame=0, frames=1, fps=1.0)
    n = _POLYLINE_SAMPLES
    out: List[Dict[str, Any]] = []
    for name, pts in tp.tracks.items():
        # evaluate each control point's track value at the clock, then ride the same
        # curve interpolation the main point uses (identical u samples).
        cps = [tuple(vs.at(clk)) for vs in pts]
        samples = [list(eval_curve(cps, k / n, tp.closed)) for k in range(n)]
        if tp.closed:
            samples.append(list(samples[0]))
        out.append({"name": name,
                    "dim": len(cps[0]) if cps else 0,
                    "scalar": bool(tp.is_scalar(name)),
                    "samples": samples})
    return out


def _describe_dataset(obj: Any, kind: str, clock: Optional[Clock]) -> Dict[str, Any]:
    d: Dict[str, Any] = {"id": obj.id, "kind": kind}
    if kind == "path":
        d.update(dim=obj.dim, closed=obj.closed, count=len(obj))
        d.update(_curve_geometry(obj, clock))
    elif kind == "tracked_path":
        d.update(dim=obj.dim, closed=obj.closed, count=len(obj),
                 tracks=list(obj.tracks.keys()))
        d.update(_curve_geometry(obj.path, clock))
        d["channels"] = _track_channels(obj, clock)
    elif kind == "grid":
        d.update(ndim=obj.ndim, shape=list(obj.shape),
                 lo=list(obj.lo), hi=list(obj.hi),
                 value_dim=obj.value_dim, is_vector=obj.is_vector,
                 channels=list(obj.channels) if obj.channels else None)
    elif kind == "scatter":
        d.update(dim=obj.dim, count=len(obj),
                 value_dim=obj.value_dim, is_vector=obj.is_vector,
                 channels=list(obj.channels) if obj.channels else None)
    return d


def _element_roots(el: Any) -> List:
    from .scene import element_roots
    try:
        return element_roots(el)
    except Exception:
        # not a full Element (e.g. a Camera has roots() but no xf) — fall back
        return list(el.roots()) if hasattr(el, "roots") else []


def _datasets_in(el: Any, out: Dict[int, Any]) -> List[int]:
    """Collect dataset node ids reachable from ``el``; stash the objects in ``out``
    (id→dataset) for the global list.  Returns the ids this element references."""
    ids: List[int] = []
    for r in _element_roots(el):
        for n in walk(r):
            if _dataset_kind(n) is not None and n.id not in ids:
                ids.append(n.id)
                out.setdefault(n.id, n)
    return ids


def _describe_element(el: Any, oid: int, datasets: Dict[int, Any]) -> Dict[str, Any]:
    cls = type(el).__name__
    from .scene import Group  # lazy
    kind = {
        "Sphere": "sphere", "Beads": "beads", "SweptMesh": "swept_mesh",
        "IsoMesh": "iso_mesh", "Raw": "raw", "Group": "group", "Volume": "volume",
    }.get(cls, cls.lower())
    rec: Dict[str, Any] = {"id": oid, "kind": kind, "class": cls}
    name = getattr(el, "name", None)
    if name is not None:
        rec["name"] = name
    mat = getattr(el, "material", None)
    if isinstance(mat, str):
        rec["material"] = mat
    for attr in ("count", "res", "iso", "bounds", "closed_spine",
                 "closed_profile", "smooth"):
        v = getattr(el, attr, None)
        if isinstance(v, (int, float, bool)):
            rec[attr] = v
    if isinstance(el, Group):
        rec["children"] = [
            _describe_element(c, f"{oid}.{j}", datasets)
            for j, c in enumerate(el.children)
        ]
    else:
        ds = _datasets_in(el, datasets)
        if ds:
            rec["datasets"] = ds
    return rec


def _edge_param(node: Any, child: Any, index: int) -> str:
    """Best-effort name of the **parameter** on ``node`` that ``child`` feeds (F5's
    edge labels).  Loom nodes store their inputs as named attributes (``self.a``,
    ``self.x``, ``self.amount``, …) — sometimes inside a list (``self.components``) —
    so match the child by identity to the attribute it lives under.  Falls back to a
    positional ``in<i>`` when the input is nested out of reach (e.g. on a sub-object)."""
    try:
        items = list(vars(node).items())
    except TypeError:
        items = []
    for k, v in items:                       # a direct named input
        if not k.startswith("_") and v is child:
            return k
    for k, v in items:                       # inside a list/tuple/dict input
        if k.startswith("_"):
            continue
        if isinstance(v, (list, tuple)):
            for i, e in enumerate(v):
                if e is child:
                    return f"{k}[{i}]"
        elif isinstance(v, dict):
            for kk, e in v.items():
                if e is child:
                    return f"{k}[{kk}]"
    return f"in{index}"


def _describe_dag(scene: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Nodes (op + short label + stable id) and edges (child feeds parent) over
    every modulator reachable from the scene.  Each edge is ``{src, dst, param}``
    where ``src`` feeds ``dst`` through ``dst``'s ``param`` input (F5's link labels)."""
    from .scene import element_roots
    nodes: Dict[int, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    seen_edge = set()
    for el in scene._all_elements():
        try:
            roots = element_roots(el)
        except Exception:
            roots = list(el.roots()) if hasattr(el, "roots") else []
        for r in roots:
            for n in walk(r):
                if n.id not in nodes:
                    nodes[n.id] = {"id": n.id, "op": type(n).__name__,
                                   "label": _node_label(n)}
                for idx, c in enumerate(n.children()):
                    key = (c.id, n.id)
                    if key not in seen_edge:
                        seen_edge.add(key)
                        edges.append({"src": c.id, "dst": n.id,
                                      "param": _edge_param(n, c, idx)})
    return {"nodes": list(nodes.values()), "edges": edges}


def _node_label(n: Any) -> str:
    """A short human label for a DAG node (value for constants, name for named
    leaves, else the op name)."""
    for attr in ("name",):
        v = getattr(n, attr, None)
        if isinstance(v, str) and v:
            return v
    v = getattr(n, "value", None)
    if isinstance(v, (int, float)):
        return f"{float(v):g}"
    return type(n).__name__


def introspect(scene: Any, *, clock: Optional[Clock] = None) -> Dict[str, Any]:
    """Walk a :class:`~loom.scene.Scene` into the JSON introspection sidecar dict.

    Keys: ``version``; ``frame`` (the clock's frame/frames); ``objects`` (the
    geometry elements, Groups recursed, each linking the ``datasets`` it
    references by id); ``datasets`` (every :class:`PointPath` / :class:`TrackedPath`
    / :class:`Grid` / :class:`Scatter` reachable in the scene — paths also carry
    their evaluated ``control_points`` + a sampled display ``polyline``); ``camera`` /
    ``lights`` (minimal); and ``dag`` (the modulator graph — ``nodes`` + ``edges``).
    """
    datasets: Dict[int, Any] = {}
    objects = [_describe_element(el, i, datasets)
               for i, el in enumerate(scene.elements)]
    # datasets reachable from camera / materials / lights too (not just geometry)
    for extra in (*scene.materials, *scene.lights, scene.camera):
        _datasets_in(extra, datasets)
    ds_list = [_describe_dataset(obj, _dataset_kind(obj), clock)
               for obj in datasets.values()]
    ds_list.sort(key=lambda d: d["id"])
    frame = {"frame": clock.frame, "frames": clock.frames} if clock else \
            {"frame": 0, "frames": 1}
    return {
        "version": SIDECAR_VERSION,
        "frame": frame,
        "objects": objects,
        "datasets": ds_list,
        "camera": {"class": type(scene.camera).__name__,
                   "name": getattr(scene.camera, "name", None)},
        "lights": [{"kind": getattr(l, "kind", None),
                    "class": type(l).__name__} for l in scene.lights],
        "dag": _describe_dag(scene),
    }


# ===========================================================================
# ViewerModel — the loom-side handle the viewer talks to
# ===========================================================================

class ViewerModel:
    """A live handle over a ``build`` contract function for the §F viewer.

    Holds the ``build`` callable and the current ``params``; re-evaluates a fresh
    :class:`~loom.scene.Scene` at any clock (:meth:`scene`) and produces the
    introspection sidecar (:meth:`introspect` / :meth:`save_sidecar`).  This is the
    Python object the viewer's bridge drives; frame *rendering* still goes over the
    ``PreviewServer`` ↔ ``ftrace -serve`` pipe (loom emits the ``.ftsl``).
    """

    def __init__(self, build: Callable[..., Any], **params: Any) -> None:
        if not callable(build):
            raise TypeError("ViewerModel needs a callable build function")
        self.build = build
        self.params: Dict[str, Any] = dict(params)

    @classmethod
    def from_file(cls, path: str, *, func: str = "build", **params: Any) -> "ViewerModel":
        return cls(load_build(path, func=func), **params)

    def declared_params(self) -> Dict[str, Any]:
        """The build's keyword params with defaults (excluding ``clock``) — the UI
        controls the file advertises.  Params with no default are omitted (unknown
        value)."""
        try:
            sig = inspect.signature(self.build)
        except (ValueError, TypeError):
            return {}
        out: Dict[str, Any] = {}
        for p in sig.parameters.values():
            if p.name == "clock" or p.kind in (inspect.Parameter.VAR_KEYWORD,
                                               inspect.Parameter.VAR_POSITIONAL):
                continue
            if p.default is not inspect.Parameter.empty:
                out[p.name] = p.default
        return out

    def scene(self, clock: Optional[Clock] = None, **overrides: Any):
        p = dict(self.params)
        p.update(overrides)
        return build_scene(self.build, clock, **p)

    def introspect(self, clock: Optional[Clock] = None, **overrides: Any) -> Dict[str, Any]:
        return introspect(self.scene(clock, **overrides), clock=clock)

    def save_sidecar(self, path: str, clock: Optional[Clock] = None,
                     **overrides: Any) -> None:
        """Atomically write the introspection sidecar JSON for the viewer."""
        _atomic_write_text(path, json.dumps(self.introspect(clock, **overrides),
                                            indent=2))


def _atomic_write_text(path: str, text: str) -> None:
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


__all__ = [
    "SIDECAR_VERSION", "load_build", "build_scene", "introspect",
    "ViewerModel",
]
