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
from typing import Any, Callable, Dict, List, Optional

from .atomicio import write_atomic
from .signals.core import Clock, walk
from .data import PointPath, TrackedPath, Grid, Scatter
from .interp import eval_curve
from .axes import axis_annotation, binding_edges

# 2: DAG nodes/edges carry the E5 axis annotation (`axes`, target kind, pin/mod
#    edge mode+gain, value-site scope).  Purely additive — a version-1 reader
#    ignores the new keys.
SIDECAR_VERSION = 2

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


def _value_at(v: Any, clk: Clock) -> List[float]:
    """Evaluate a Grid/Scatter value node at ``clk`` and normalise it to a plain
    ``list[float]`` channel-vector — a scalar Signal becomes a 1-list, a VecSignal
    becomes its component list — so the sidecar's field data is uniform regardless of
    scalar-vs-vector."""
    r = v.at(clk)
    if isinstance(r, (list, tuple)):
        return [float(x) for x in r]
    return [float(r)]


def _swept_mesh_geometry(sm: Any, clock: Optional[Clock]) -> Dict[str, Any]:
    """Tessellate a :class:`~loom.scene.SweptMesh` at ``clock`` into the triangle
    mesh the viewer's 3-D pane draws (F4).  Mirrors ``SweptMesh.emit`` — sample the
    spine, sweep the profile into rings, skin them — but returns the geometry as
    plain lists instead of writing an OBJ: ``vertices`` (flat 3-vectors), ``faces``
    (0-based index triples), ``uvs`` (per-vertex ``(u, v)`` from the ring/profile
    lattice; ``u`` along the spine, ``v`` around the profile), plus ``rings``/
    ``profile_count`` so the viewer knows the lattice shape."""
    import math
    from . import sweep as _sweep
    from .signals.core import Cache
    clk = clock if clock is not None else Clock(t=0.0, frame=0, frames=1, fps=1.0)
    cache = Cache()

    def _num(v):
        return float(v.at(clk, cache)) if hasattr(v, "at") else float(v)

    n = sm.count
    pts = [sm.spine.sample(k / n, clk, cache) for k in range(n)]
    base_sc, base_tw = _num(sm.scale), _num(sm.twist)
    turns = _num(sm.turns)
    scales, twists = [], []
    for k in range(n):
        u = k / n
        mult = sm.scale_profile(u) if sm.scale_profile is not None else 1.0
        scales.append(base_sc * mult)
        twists.append(base_tw + turns * 2.0 * math.pi * u)
    rings = _sweep.sweep_rings(pts, sm.profile, scales, twists, sm.closed_spine)
    verts, faces = _sweep.skin_rings(rings, sm.closed_spine, sm.closed_profile)
    k = len(sm.profile)
    # per-vertex UVs from the (ring i, profile j) lattice; wrap divisor for closed
    ud = n if sm.closed_spine else max(1, n - 1)
    vd = k if sm.closed_profile else max(1, k - 1)
    uvs = [[i / ud, j / vd] for i in range(n) for j in range(k)]
    return {"vertices": [list(v) for v in verts],
            "faces": [list(f) for f in faces],
            "uvs": uvs, "rings": n, "profile_count": k}


def _iso_mesh_geometry(im: Any, clock: Optional[Clock]) -> Dict[str, Any]:
    """Bake an :class:`~loom.scene.IsoMesh`'s scalar field to a triangle mesh at
    ``clock`` via marching cubes (F7's MC-mesh fallback path) — the same bake
    ``IsoMesh.emit`` does, but returned as plain lists so the viewer's Meshes tab
    can draw it.  ``vertices`` are world-space 3-vectors and ``faces`` 0-based index
    triples; marching cubes has no natural UVs, so none are emitted."""
    from . import mcubes as _mc
    from .signals.core import Cache
    clk = clock if clock is not None else Clock(t=0.0, frame=0, frames=1, fps=1.0)
    verts, faces = _mc.mesh_field(
        im.field, bounds=im.bounds, res=im.res, iso=im.iso,
        clock=clk, cache=Cache(), adaptive=im.adaptive, coarse=im.coarse)
    return {"vertices": [list(v) for v in verts],
            "faces": [list(f) for f in faces], "iso": im.iso}


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
        clk = clock if clock is not None else Clock(t=0.0, frame=0, frames=1, fps=1.0)
        d.update(ndim=obj.ndim, shape=list(obj.shape),
                 lo=list(obj.lo), hi=list(obj.hi),
                 value_dim=obj.value_dim, is_vector=obj.is_vector,
                 channels=list(obj.channels) if obj.channels else None)
        # per-axis sample coordinates (the fixed lattice positions) and the flat
        # C-order value grid (one channel-vector per lattice node), evaluated at clk.
        d["axes"] = [obj.axis_coords(a) for a in range(obj.ndim)]
        d["values"] = [_value_at(v, clk) for v in obj.values]
    elif kind == "scatter":
        clk = clock if clock is not None else Clock(t=0.0, frame=0, frames=1, fps=1.0)
        d.update(dim=obj.dim, count=len(obj),
                 value_dim=obj.value_dim, is_vector=obj.is_vector,
                 channels=list(obj.channels) if obj.channels else None)
        # sample positions (domain coords) and their channel-vector values at clk.
        d["points"] = [list(p.at(clk)) for p in obj.positions]
        d["values"] = [_value_at(v, clk) for v in obj.values]
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


def _describe_element(el: Any, oid: int, datasets: Dict[int, Any],
                      clock: Optional[Clock] = None) -> Dict[str, Any]:
    cls = type(el).__name__
    from .scene import Group, SweptMesh, IsoMesh  # lazy
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
            _describe_element(c, f"{oid}.{j}", datasets, clock)
            for j, c in enumerate(el.children)
        ]
    else:
        ds = _datasets_in(el, datasets)
        if ds:
            rec["datasets"] = ds
        if isinstance(el, SweptMesh):
            try:
                rec["mesh"] = _swept_mesh_geometry(el, clock)
            except Exception as exc:      # never let one bad mesh sink the sidecar
                rec["mesh_error"] = str(exc)
        elif isinstance(el, IsoMesh):
            try:
                rec["mesh"] = _iso_mesh_geometry(el, clock)
            except Exception as exc:      # skimage missing / empty surface, etc.
                rec["mesh_error"] = str(exc)
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


_TEXTURE_REF = "texture:"


def _describe_texture(tex: Any, clock: Optional[Clock] = None) -> Dict[str, Any]:
    """A texture/skin definition for the viewer (F4 texture display): an **image**
    skin (``Texture`` → a file path the viewer loads) or a **formula/procedural**
    skin (``ProcTexture`` → the three ``r/g/b`` UV expressions the viewer bakes, the
    E1 procedural-skin path).  These are what the viewer needs to shade a mesh with
    its authored surface detail instead of the flat/checker placeholder.

    A ``ProcTexture`` channel may be a literal ftsl string *or* a live
    :class:`~loom.spatial.SpatialExpr` (that is what a material bundle's colour field
    lowers to — see :meth:`loom.scene.Material.expand`).  An expression object is
    neither JSON-serialisable nor useful to the viewer, so bake each channel to its
    ftsl source **at the clock** here, exactly as :meth:`ProcTexture.emit` does — the
    viewer then compiles the very same string ftrace would."""
    from .scene import Texture, ProcTexture  # lazy
    rec: Dict[str, Any] = {"name": getattr(tex, "name", None)}
    if isinstance(tex, ProcTexture):
        from .ftsl_emit import EmitCtx
        from .signals.core import Cache
        ctx = EmitCtx(clock=clock if clock is not None else Clock.at_frame(0, 1),
                      cache=Cache())
        r, g, b = (ProcTexture._chan_str(c, ctx) for c in tex._channels())
        rec.update(kind="formula", r=r, g=g, b=b, res=tex.res,
                   filter=tex.filter, wrap=tex.wrap)
    elif isinstance(tex, Texture):
        rec.update(kind="image", file=tex.file, encoding=tex.encoding,
                   filter=tex.filter, wrap=tex.wrap)
    else:                                    # unknown skin kind — name only
        rec["kind"] = type(tex).__name__.lower()
    return rec


def _prop_value(v: Any, clock: Optional[Clock]) -> Any:
    """JSON-friendly value of a material prop: primitives verbatim; an animated
    :class:`Signal` / :class:`VecSignal` evaluated at the clock (best-effort, so a
    weird prop never sinks the sidecar)."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    try:
        from .signals.core import Signal, Cache
        from .signals.vector import VecSignal
        from .ftsl_emit import num, vec3
        c = clock if clock is not None else Clock.at_frame(0, 1)
        if isinstance(v, VecSignal):
            return list(vec3(v, c, Cache()))
        if isinstance(v, Signal):
            return num(v, c, Cache())
    except Exception:
        pass
    return repr(v)


def _describe_material(mat: Any, clock: Optional[Clock] = None) -> Dict[str, Any]:
    """A material for the viewer: its ``type``, its props (serialised), and the
    **texture** it binds (the skin name behind any ``texture:<name>`` prop — F4's
    texture blocker, since the viewer only saw a bare material *name* before)."""
    props: Dict[str, Any] = {}
    texture = None
    for k, v in getattr(mat, "props", {}).items():
        if isinstance(v, str) and v.startswith(_TEXTURE_REF):
            texture = v[len(_TEXTURE_REF):]
        props[k] = _prop_value(v, clock)
    return {"name": getattr(mat, "name", None),
            "type": getattr(mat, "mtype", None),
            "texture": texture, "props": props}


def _describe_dag(scene: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Nodes (op + short label + stable id) and edges (child feeds parent) over
    every modulator reachable from the scene.  Each edge is ``{src, dst, param}``
    where ``src`` feeds ``dst`` through ``dst``'s ``param`` input (F5's link labels).

    **E5 axis annotation (sidecar v2).** This is the on-disk projection of
    :mod:`loom.axes` (``.ftsl`` itself carries none — it is a per-frame *bound*
    snapshot, see the note in that module).  A node additionally carries its free
    ``axes``, and a ``Target`` its declared ``target_kind``/``neutral``; an edge
    out of a ``Target`` carries the ``mode`` (``pin``/``mod``) and ``gain`` that
    a plain child list cannot express, and is named ``mod[i]``/``pin[i]`` instead
    of an anonymous ``in<i>``.  So an editor reading the sidecar can show the
    influence model, not just the call graph."""
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
                    rec = {"id": n.id, "op": type(n).__name__,
                           "label": _node_label(n)}
                    rec.update(axis_annotation(n))
                    nodes[n.id] = rec
                binds = binding_edges(n)      # {} unless n is a Target
                for idx, c in enumerate(n.children()):
                    key = (c.id, n.id)
                    if key in seen_edge:
                        continue
                    seen_edge.add(key)
                    e = {"src": c.id, "dst": n.id,
                         "param": _edge_param(n, c, idx)}
                    b = binds.get(c.id)
                    if b is not None:         # an E5 influence edge
                        e["mode"] = b["mode"]
                        e["gain"] = b["gain"]
                        e["param"] = f"{b['mode']}[{b['index']}]"
                    edges.append(e)
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
    their evaluated ``control_points`` + a sampled display ``polyline``);
    ``materials`` (each material's ``type``/``props`` and the ``texture`` skin it
    binds) + ``textures`` (image-file or formula skin definitions — so the viewer can
    shade meshes with their authored surface detail, F4); ``camera`` / ``lights``
    (minimal); and ``dag`` (the modulator graph — ``nodes`` + ``edges``).
    """
    datasets: Dict[int, Any] = {}
    objects = [_describe_element(el, i, datasets, clock)
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
        "materials": [_describe_material(m, clock) for m in scene.materials],
        "textures": [_describe_texture(t, clock) for t in scene.textures],
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

    def declared_param_types(self) -> Dict[str, str]:
        """Per-param **type tag** for :meth:`declared_params` — ``"int"``, ``"float"``,
        ``"bool"``, ``"str"`` or ``"other"``.

        JSON erases Python's int/float distinction on the way to the viewer (both are
        just numbers), but a build that takes ``rings=8`` and gets ``8.0`` back can
        fail outright (``range(8.0)``).  A viewer that knows the tag can render an
        integer drag and send an integer literal.  Note ``bool`` is checked before
        ``int`` — in Python ``True`` *is* an ``int``, and a checkbox is what the author
        meant.
        """
        out: Dict[str, str] = {}
        for k, v in self.declared_params().items():
            if isinstance(v, bool):     # before int: bool subclasses int
                out[k] = "bool"
            elif isinstance(v, int):
                out[k] = "int"
            elif isinstance(v, float):
                out[k] = "float"
            elif isinstance(v, str):
                out[k] = "str"
            else:
                out[k] = "other"
        return out

    def build_path(self) -> Optional[str]:
        """Absolute path of the file the ``build`` callable was defined in, or ``None``
        when it has no source file (a lambda in a REPL, a C function).  Recorded in the
        sidecar as ``"build"`` so a viewer handed only the sidecar can reconnect the
        **live** re-introspection channel (``python -m loom.viewer <that path>``) without
        being told the scene file a second time."""
        try:
            src = inspect.getsourcefile(self.build) or inspect.getfile(self.build)
        except (TypeError, OSError):
            return None
        return os.path.abspath(src) if src else None

    def scene(self, clock: Optional[Clock] = None, **overrides: Any):
        p = dict(self.params)
        p.update(overrides)
        return build_scene(self.build, clock, **p)

    def introspect(self, clock: Optional[Clock] = None, **overrides: Any) -> Dict[str, Any]:
        """The introspection sidecar for this model, plus a ``"build"`` key naming the
        file the ``build`` callable came from (see :meth:`build_path`) — the sidecar's
        own provenance, and what lets a viewer reopen the live channel from it."""
        data = introspect(self.scene(clock, **overrides), clock=clock)
        bp = self.build_path()
        if bp is not None:
            data["build"] = bp
        return data

    def save_sidecar(self, path: str, clock: Optional[Clock] = None,
                     *, emit_source: bool = True, **overrides: Any) -> None:
        """Atomically write the introspection sidecar JSON for the viewer.

        With ``emit_source`` (default), also emit the scene's ``.ftsl`` next to the
        sidecar and record its absolute path under the sidecar's ``"source"`` key,
        so the C++ ``-viewer`` can parse it with ftrace's own loader and raymarch
        the real field via ``-raster-gpu`` (``renderIsoPreviewCuda``) — the F7
        primary path — instead of the static marching-cubes fallback mesh.
        """
        scene = self.scene(clock, **overrides)
        data = introspect(scene, clock=clock)
        bp = self.build_path()
        if bp is not None:
            data["build"] = bp      # provenance: reopens the live channel from the sidecar
        if emit_source:
            src = self._emit_source(scene, path, clock)
            if src is not None:
                data["source"] = src
        _atomic_write_text(path, json.dumps(data, indent=2))

    @staticmethod
    def _emit_source(scene: Any, sidecar_path: str,
                     clock: Optional[Clock]) -> Optional[str]:
        """Emit ``scene``'s ``.ftsl`` beside ``sidecar_path``; return its abs path.

        Returns ``None`` (and leaves no file) if the scene can't be emitted, so a
        sidecar is still written — the viewer just falls back to its static mesh.
        """
        from .signals.core import Cache
        try:
            base = os.path.splitext(os.path.abspath(sidecar_path))[0]
            ftsl_path = base + ".ftsl"
            clk = clock if clock is not None else Clock.at_frame(0, 1)
            assets = os.path.dirname(ftsl_path)
            text = scene.emit(clk, Cache(), assets_dir=assets)
            _atomic_write_text(ftsl_path, text)
            return ftsl_path
        except Exception:
            return None


def _atomic_write_text(path: str, text: str) -> None:
    write_atomic(path, text)


# ===========================================================================
# the viewer↔loom live re-introspection channel  (§F4 / §F7)
# ===========================================================================

class ViewerSession:
    """The loom side of the viewer↔loom **live re-introspection channel** (§F4/§F7).

    The C++ ``-viewer`` GUI normally reads a **static** sidecar loom wrote once, so
    it can *display* geometry but can't **re-derive** it — rotating into a parameter
    dimension, scrubbing time, or editing a field all need geometry re-baked, which
    the frozen JSON can't do (the architecturally-blocked half of F4's off-thread
    re-tessellation and F7's live field edit).  This session holds a **resident**
    :class:`ViewerModel` and answers the viewer's requests to re-introspect the
    scene at a new clock / parameter values, handing back a fresh sidecar.

    It processes one message ``dict`` and returns an ack ``dict``; the transport
    (:func:`serve_viewer`, a newline-delimited-JSON stdio pipe) is separate, so the
    protocol is unit-testable with in-memory data — mirroring
    :class:`loom.anim.LiveSession` in the viewer→loom (re-introspection) direction.
    Messages (``cmd``):

    * ``introspect`` — ``{clock?:{frame,frames,open}, params?:{…}, out?:"path"}``:
      re-derive the scene at that clock/params and introspect it.  With ``out`` the
      sidecar JSON is written there (atomically) and the ack is ``{ok, out}``; else
      the sidecar dict rides back inline as ``{ok, sidecar}``.
    * ``emit`` — ``{clock?:{…}, params?:{…}, out?:"path", assets_dir?:"path",
      mesh_format?:"ftmesh"|"obj", assets?:"inline"}``: re-derive the scene at that
      clock/params and emit its ``.ftsl`` (the source ftrace parses to raymarch the
      real field — the F7 primary path).  With ``out`` the ``.ftsl`` is written there
      (atomically) and the ack is ``{ok, out}``; else the source text rides back
      inline as ``{ok, source}``.  This is what the viewer calls on
      rotate/scrub/param-edit to get freshly re-tessellated geometry (F4).
      ``assets: "inline"`` additionally keeps file-backed meshes out of the
      filesystem: they come back as private ``_blobs`` for :func:`serve_viewer` to
      frame onto the wire, and ``assets_dir`` becomes purely the naming scheme the
      scene text uses to refer to them.
    * ``params`` — ack ``{ok, params, types, build}``: the build's declared keyword
      controls (name→default) so the viewer can build its parameter UI, each one's
      type tag (``int``/``float``/``bool``/``str``/``other`` — JSON erases int-vs-float
      and a build taking ``rings=8`` breaks on ``8.0``), and the build file's path.
    * ``quit`` — stop the serve loop; ack ``{ok, bye:true}``.
    """

    def __init__(self, model: ViewerModel) -> None:
        self.model = model

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        try:
            if cmd == "introspect":
                return self._introspect(msg)
            if cmd == "emit":
                return self._emit(msg)
            if cmd == "params":
                return {"ok": True,
                        "params": self.model.declared_params(),
                        "types": self.model.declared_param_types(),
                        "build": self.model.build_path()}
            if cmd == "quit":
                return {"ok": True, "bye": True}
            return {"ok": False, "error": f"unknown cmd {cmd!r}"}
        except Exception as e:  # report, don't crash the pipe loop
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _clock(spec: Optional[dict]) -> Optional[Clock]:
        if not spec:
            return None
        frame = int(spec.get("frame", 0))
        frames = int(spec.get("frames", 1))
        return Clock.at_frame(frame, frames, loop=not spec.get("open", False))

    def _introspect(self, msg: dict) -> dict:
        clock = self._clock(msg.get("clock"))
        params = {str(k): v for k, v in (msg.get("params") or {}).items()}
        sidecar = self.model.introspect(clock, **params)
        out = msg.get("out")
        if out:
            _atomic_write_text(out, json.dumps(sidecar, indent=2))
            return {"ok": True, "out": out}
        return {"ok": True, "sidecar": sidecar}

    def _emit(self, msg: dict) -> dict:
        from .signals.core import Cache
        clock = self._clock(msg.get("clock"))
        params = {str(k): v for k, v in (msg.get("params") or {}).items()}
        scene = self.model.scene(clock, **params)
        clk = clock if clock is not None else Clock.at_frame(0, 1)
        out = msg.get("out")
        assets = msg.get("assets_dir") or (
            os.path.dirname(os.path.abspath(out)) if out else None)
        # Binary meshes on the live channel. Nothing ever reads these files but the
        # ftrace that asked for them, milliseconds later, on this machine — so the OBJ
        # text round trip (format here, parse there, every frame, for geometry that is
        # floats on both ends) buys nothing and costs several ms. `mesh_format`
        # defaults to "obj" everywhere else, where a portable artifact IS the point;
        # a caller can still ask for text here by sending `mesh_format: "obj"`.
        fmt = msg.get("mesh_format") or "ftmesh"
        # `assets: "inline"` — don't write the meshes at all; hand their bytes back
        # for the transport to deliver.  `assets_dir` still names them (the scene text
        # is the same either way), it just never has to exist.  Opt-in, because an
        # ack that carries megabytes only makes sense over a channel that can frame
        # them — see `serve_viewer`.
        sink = {} if msg.get("assets") == "inline" else None
        text = scene.emit(clk, Cache(), assets_dir=assets, mesh_format=fmt,
                          mesh_sink=sink)
        if out:
            _atomic_write_text(out, text)
            ack = {"ok": True, "out": out}
        else:
            ack = {"ok": True, "source": text}
        if sink:
            ack["_blobs"] = list(sink.items())
        return ack


def serve_viewer(session: ViewerSession, in_stream=None, out_stream=None,
                 bin_stream=None) -> None:
    """Run the newline-delimited-JSON message loop over the given streams (default
    the process's ``stdin``/``stdout``): read one JSON object per line, dispatch to
    ``session.handle``, write one JSON ack per line, and stop after a ``quit`` (or
    EOF).  Mirrors :func:`loom.anim.serve_live` and :class:`loom.PreviewServer`'s
    one-message-per-line stdio convention, in the viewer→loom direction.

    BINARY ATTACHMENTS.  ``session.handle`` may return a private ``_blobs`` key —
    ``[(name, bytes), …]`` — which this function turns into a public wire framing:
    the ack line gains ``"blobs":[{"name","bytes"}, …]`` and the raw payloads follow
    the newline back to back, in that order, with no delimiters (every length is in
    the manifest).  A reader that doesn't want them must still read past them or the
    stream desyncs.

    Keeping the framing *here* rather than in ``handle`` is deliberate: ``handle``
    stays a pure dict→dict function that unit tests can drive with no streams at all,
    and the one place that knows about bytes-on-a-pipe is the one place that has a
    pipe.  ``_blobs`` never reaches the wire under that name.

    ``bin_stream`` defaults to ``out_stream.buffer`` — the same file descriptor, seen
    unencoded.  The text stream is flushed before anything is written to it, so the
    two views cannot interleave out of order."""
    in_stream = sys.stdin if in_stream is None else in_stream
    out_stream = sys.stdout if out_stream is None else out_stream
    if bin_stream is None:
        bin_stream = getattr(out_stream, "buffer", None)
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
        blobs = ack.pop("_blobs", None)
        if blobs:
            if bin_stream is None:
                ack = {"ok": False,
                       "error": "blobs requested but this stream is text-only"}
                blobs = None
            else:
                ack["blobs"] = [{"name": n, "bytes": len(b)} for n, b in blobs]
        out_stream.write(json.dumps(ack) + "\n")
        out_stream.flush()
        if blobs:
            for _name, data in blobs:
                bin_stream.write(data)
            bin_stream.flush()
        if ack.get("bye"):
            break


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry: ``python -m loom.viewer <scene.py> [name=value …]`` loads the
    build and runs the resident re-introspection server (:func:`serve_viewer`) on
    stdin/stdout.  ``key=value`` args seed the model's initial params (parsed as
    JSON when possible, else kept as strings)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="loom.viewer",
        description="Resident loom re-introspection server for the C++ -viewer GUI.")
    parser.add_argument("scene", help="path to a loom scene file exposing build()")
    parser.add_argument("--func", default="build",
                        help="build callable name (default: build)")
    parser.add_argument("param", nargs="*",
                        help="initial params as name=value (value parsed as JSON)")
    ns = parser.parse_args(argv)

    params: Dict[str, Any] = {}
    for item in ns.param:
        if "=" not in item:
            parser.error(f"param {item!r} is not name=value")
        k, v = item.split("=", 1)
        try:
            params[k] = json.loads(v)
        except json.JSONDecodeError:
            params[k] = v

    model = ViewerModel.from_file(ns.scene, func=ns.func, **params)
    serve_viewer(ViewerSession(model))
    return 0


__all__ = [
    "SIDECAR_VERSION", "load_build", "build_scene", "introspect",
    "ViewerModel", "ViewerSession", "serve_viewer", "main",
]


if __name__ == "__main__":  # pragma: no cover
    # See the note in loom/anim.py: `python -m loom.viewer` re-executes this file as
    # `__main__`, giving every class defined here a second identity, while a scene's
    # `from loom.viewer import ...` gets the canonical one. Any isinstance test
    # across that boundary then silently fails. Delegate to the imported module.
    from loom.viewer import main as _main
    raise SystemExit(_main())
