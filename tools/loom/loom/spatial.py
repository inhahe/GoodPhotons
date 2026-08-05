"""
Loom spatial-expression tree (M10.5) — one pattern definition, evaluated **two
ways**: numerically over pixels (the 2-D backend) *and* emitted as an ftsl string
(the 3-D isosurface / material backend).

DESIGN.md §11.10: loom's temporal DAG is a function of *time* (the clock), cached
per frame.  A *field* is a function of *space* (x, y, z) — a different axis — so it
does **not** belong in the time-DAG (its per-frame cache would be wrong: one value
per pixel, not per frame).  Coordinates live here, in a small **spatial** algebra
whose leaves are the coordinate variables :data:`X`, :data:`Y`, :data:`Z` and the
loop phase :data:`T`, and whose *coefficients* may be temporal :class:`Signal`\\s
(baked per frame — exactly how the 3-D side already animates a static formula).

The leaf family (:class:`Surface`) also carries the *surface* inputs :data:`U`,
:data:`V` (the ftrace pattern vars ``u``/``v`` — emit-only) and the material
*albedo* placeholder :data:`A` (no ftrace variable — a pure binding slot).  A
:class:`SpatialExpr` reports its named inputs with :meth:`SpatialExpr.free_inputs`
and binds them by rewrite with :meth:`SpatialExpr.substitute` — this is the
substrate for materials-as-bundles (``gold(u=v, a=1)`` / ``(a=x*.5)`` authoring):
loom resolves every binding to a concrete field in real ftrace variables at emit,
so it never writes literal bundle syntax and stays renderable at every step.

A :class:`SpatialExpr` evaluates two ways:

- :meth:`SpatialExpr.eval_np` — numerically over numpy coordinate arrays (2-D
  raster fields), and
- :meth:`SpatialExpr.emit` — as an ftsl expression string in ``x``/``y``/``z``
  with the temporal coefficients baked to numbers (3-D isosurfaces / patterns).

Because it also exposes ``build(cx, cy, cz, ctx)`` and ``param_signals()`` it drops
straight into :class:`loom.Isosurface` and :class:`loom.FuncPattern` through their
existing duck-typed template protocol — no changes there.  Every function name
emitted (:func:`sin`, :func:`sign`, :func:`clamp`, …) is a real ftsl pattern
builtin (``src/pattern.h``), so the emitted string always parses; the numpy path
computes the *same* mathematics (``noise`` is intentionally absent — ftrace's value
noise has no bit-identical numpy twin, so it would break the "one definition, two
backends" honesty).

One leaf is not a coordinate but a *photograph*: :class:`Image` samples a decoded
image file as a scalar **term inside a formula**, so a picture can be multiplied,
thresholded, warped or blended like any other subexpression rather than only bound
wholesale to a material slot.  It honours the two-backend rule the same way as the
rest of the algebra: it emits ftrace's ``tex:<name>(u, v)`` pattern op (``PatOp::Tex``
in ``src/pattern.h``, with the GPU twin in ``render_cuda.cu``) and its
:meth:`Image.eval_np` is a faithful port of ``Texture::sampleRgb``/``scalarAt`` —
same ``-0.5`` texel offset, same ``v``-flip, same repeat/clamp/mirror wrapping, same
mean-of-RGB reduction.  Its ``u``/``v`` arguments are ordinary sub-expressions, so
the coordinates can themselves be warped or rebound.

:class:`VolumeField` is that leaf's 3-D twin: an imported ``.vdb``/``.nvdb`` grid
sampled as a scalar term, so a captured volume can be *modulated, warped, meshed
and re-baked* like any other subexpression.  It is what makes loom's
read → transform → write volume workflow (roadmap §E4) compose out of machinery
that already existed — value ops are the algebra, warping is its rebindable
``x``/``y``/``z`` children, meshing is :mod:`loom.mcubes`, resampling is
:func:`loom.vdbio.bake_field`.  Its affine *placement* helpers are lossless (they
compose onto the grid's own index→world transform rather than resampling), so
discretisation still happens exactly once, at the end.  It is the one leaf here
that is **single-backend**: ftrace's pattern VM has no volume-sampling opcode, so
:meth:`VolumeField.emit` raises rather than invent an ftsl string that would mean
something else — a ``VolumeField`` is baked to a ``.vdb`` and rendered as
``density vdb:<path>``.
"""

from __future__ import annotations

import operator
from typing import Callable, List, Sequence, Tuple, Union

from .signals.core import Signal, Number, as_signal
from .signals.retime import retimed_clock
from .ftsl_emit import fmt

try:  # numpy is only needed for the 2-D numeric path
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------

class SpatialExpr:
    """A scalar function of space (and optionally the loop phase ``T``).

    Build with the leaves :data:`X`/:data:`Y`/:data:`Z`/:data:`T`, Python numbers,
    temporal :class:`~loom.signals.core.Signal`\\s (used as animated coefficients),
    the arithmetic operators, and the module math functions.
    """

    # ---- operators (coerce the other operand into the spatial algebra) ----
    def __add__(self, o): return _Bin("+", self, _coerce(o))
    def __radd__(self, o): return _Bin("+", _coerce(o), self)
    def __sub__(self, o): return _Bin("-", self, _coerce(o))
    def __rsub__(self, o): return _Bin("-", _coerce(o), self)
    def __mul__(self, o): return _Bin("*", self, _coerce(o))
    def __rmul__(self, o): return _Bin("*", _coerce(o), self)
    def __truediv__(self, o): return _Bin("/", self, _coerce(o))
    def __rtruediv__(self, o): return _Bin("/", _coerce(o), self)
    def __neg__(self): return _Neg(self)
    def __pow__(self, o): return spow(self, o)
    def __abs__(self): return sabs(self)

    # ---- tree walk (leaves override the hooks) ----------------------------
    def children(self) -> Tuple["SpatialExpr", ...]:
        return ()

    def _time_signal(self):
        return None  # a _Sig leaf returns its wrapped Signal

    def _is_time(self) -> bool:
        return False  # T and _Sig leaves are time-dependent

    def _walk(self):
        stack: List[SpatialExpr] = [self]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children())

    # ---- named-input inspection / binding (J3b materials-as-bundles) -------
    def _input_name(self):
        return None  # a Surface leaf returns its binding name

    def _image_texture(self):
        return None  # an Image leaf returns the loom Texture it needs declared

    def _table_decl(self):
        return None  # a GridSample / ScatterSample leaf returns its N-D table block

    def table_decls(self) -> List:
        """The ``grid { … }`` / ``scatter { … }`` declarations required by the
        :class:`GridSample` and :class:`ScatterSample` leaves in this tree, deduped
        by name and in encounter order.

        The exact twin of :meth:`image_textures`, and for the same reason: a table
        leaf emits ftrace's ``grid:<name>(c0, …)`` / ``scatter:<name>(c0, …)`` call,
        which only resolves if a table of that name is declared, so
        :meth:`loom.Scene.add` collects the companion block automatically."""
        out, seen = [], set()
        for n in self._walk():
            t = n._table_decl()
            if t is not None and t.name not in seen:
                seen.add(t.name)
                out.append(t)
        return out

    def image_textures(self) -> List:
        """The :class:`loom.scene.Texture` declarations required by the
        :class:`Image` leaves in this tree, deduped by name and in encounter order.

        :meth:`loom.Scene.add` collects these automatically, so an ``Image`` term
        drops into a field without the author having to declare the texture
        separately — but the list is public so a hand-rolled emit path can too."""
        out, seen = [], set()
        for n in self._walk():
            t = n._image_texture()
            if t is not None and t.name not in seen:
                seen.add(t.name)
                out.append(t)
        return out

    def free_inputs(self, include_coords: bool = False) -> "frozenset[str]":
        """The set of named input leaves present in the tree.

        By default this is the *bindable* free-input set — the surface params
        (:data:`U`, :data:`V`) and the albedo (:data:`A`) — which is exactly what
        a material exposes for binding (``gold(u=v, a=1)``).  Pass
        ``include_coords=True`` to also include the system-provided spatial
        coordinates :data:`X`/:data:`Y`/:data:`Z`."""
        out = set()
        for n in self._walk():
            nm = n._input_name()
            if nm is None:
                continue
            if include_coords or not getattr(n, "is_coord", False):
                out.add(nm)
        return frozenset(out)

    def substitute(self, mapping) -> "SpatialExpr":
        """Return a copy of the tree with every named input leaf whose name is a
        key of ``mapping`` replaced by ``mapping[name]`` (coerced into the spatial
        algebra).  This is how a material binds its free inputs at emit —
        ``gold(u=v)`` rewrites the :data:`U` leaf to the consumer's expression,
        so loom always emits a concrete field in real ftrace variables and never
        literal bundle syntax."""
        kids = self.children()
        if not kids:
            return self
        return self._rebuild([k.substitute(mapping) for k in kids])

    def _rebuild(self, new_children) -> "SpatialExpr":
        return self  # leaves have no children; interior nodes override

    def time_signals(self) -> List[Signal]:
        """The temporal Signals embedded as coefficients (deduped) — the DAG roots
        an :class:`Isosurface`/:class:`FuncPattern` must expose for cycle/cache."""
        out: List[Signal] = []
        seen = set()
        for n in self._walk():
            s = n._time_signal()
            if s is not None and id(s) not in seen:
                seen.add(id(s))
                out.append(s)
        return out

    def uses_time(self) -> bool:
        """True if any ``T`` leaf or temporal-Signal coefficient is present — i.e.
        the field varies over the loop (else a 2-D raster can be baked once)."""
        return any(n._is_time() for n in self._walk())

    # ---- evaluation (subclasses implement) --------------------------------
    def emit(self, coords: Tuple[str, str, str], ctx) -> str:
        raise NotImplementedError

    def eval_np(self, coords, clock, cache):
        raise NotImplementedError

    # ---- duck-typed template protocol (Isosurface / FuncPattern) ----------
    def build(self, cx: str, cy: str, cz: str, ctx) -> str:
        return self.emit((cx, cy, cz), ctx)

    def param_signals(self) -> List[Signal]:
        return self.time_signals()


def _coerce(v: Union[SpatialExpr, Signal, Number]) -> SpatialExpr:
    if isinstance(v, SpatialExpr):
        return v
    if isinstance(v, Signal):
        return _Sig(v)
    if isinstance(v, (int, float)):
        return _Const(float(v))
    raise TypeError(f"cannot use {type(v).__name__} in a spatial expression")


def sexpr(v: Union[SpatialExpr, Signal, Number]) -> SpatialExpr:
    """Coerce a number / temporal Signal / SpatialExpr into the spatial algebra."""
    return _coerce(v)


# ---------------------------------------------------------------------------
# leaves
# ---------------------------------------------------------------------------

class _Const(SpatialExpr):
    def __init__(self, v: float) -> None:
        self.v = float(v)

    def emit(self, coords, ctx) -> str:
        return f"({fmt(self.v)})"

    def eval_np(self, coords, clock, cache):
        return self.v


class Surface(SpatialExpr):
    """A named input leaf of the field / material grammar (J3b).

    Six singletons live on this class:

    - :data:`X` / :data:`Y` / :data:`Z` — the spatial coordinates (axes 0/1/2).
      Evaluated *both* ways: ``eval_np`` indexes the coordinate arrays and
      ``emit`` writes the coordinate token.
    - :data:`U` / :data:`V` — the surface parameters (ftrace pattern variables
      ``u`` / ``v``).  **Emit-only**: they are real ftsl tokens so they render,
      but they have no numpy twin, so ``eval_np`` raises (the 2-D raster backend
      has no surface UV).
    - :data:`A` — the material *albedo* input.  ftrace's pattern VM has **no**
      ``a`` variable, so :data:`A` is a pure binding placeholder: it must be
      substituted away (``substitute({'a': ...})``) or defaulted by the material
      before ``emit``; emitting a bare :data:`A` raises.

    A material's free-input set (:meth:`SpatialExpr.free_inputs`) is the union of
    its properties' input leaves; binding rewrites those leaves by name with
    :meth:`SpatialExpr.substitute` — ``gold(u=v, a=1)``."""

    def __init__(self, name: str, *, axis: int = None, emit_ok: bool = True) -> None:
        self.name = name
        self.axis = axis
        self.is_coord = axis is not None
        self._emit_ok = emit_ok

    def emit(self, coords, ctx) -> str:
        if self.axis is not None:
            return f"({coords[self.axis]})"
        if not self._emit_ok:
            raise ValueError(
                f"the material input '{self.name}' has no ftrace pattern "
                f"variable; bind it (e.g. {self.name}=<expr>) or give the "
                f"material an albedo default before emitting")
        return f"({self.name})"

    def eval_np(self, coords, clock, cache):
        if self.axis is not None:
            return coords[self.axis]
        raise ValueError(
            f"the surface input '{self.name}' is emit-only (no numpy twin); it "
            f"exists on the 3-D / material backend, not the 2-D raster path")

    def _input_name(self):
        return self.name

    def substitute(self, mapping) -> "SpatialExpr":
        if self.name in mapping:
            return _coerce(mapping[self.name])
        return self


class Image(SpatialExpr):
    """An **image sampled as a term inside a formula** (J3b item 3b).

    ``Image("bark.png")`` is a scalar leaf whose value at a surface point is the
    image's grayscale level there, so a photo can be *multiplied into* a procedural
    field rather than pasted over it::

        grime = Image("grime.png")
        rough = 0.05 + 0.9 * grime * (0.5 + 0.5 * sin(30 * X))

    This is the complement of :func:`loom.scene.skin` — that binds a whole image to
    a material slot; this makes the image one operand of an expression.

    The coordinates default to the surface params :data:`U`/:data:`V` but are
    ordinary :class:`SpatialExpr`\\s, so the image can be warped
    (``Image("p.png", u=U * 2 + 0.1 * sin(10 * V))``) and stays rebindable — a
    :meth:`SpatialExpr.substitute` reaches inside them, which is how a material
    bundle's ``u=``/``v=`` binding flows into an image term.

    **Both backends.**  :meth:`emit` writes ftrace's ``tex:<name>(u, v)`` pattern-VM
    call (``PatOp::Tex``) and the required ``texture`` declaration is collected by
    :meth:`SpatialExpr.image_textures` — :meth:`loom.Scene.add` picks it up, so the
    author never declares it.  :meth:`eval_np` is an exact port of ftrace's
    ``Texture::sampleRgb`` / ``scalarAt`` (same v-flip, same wrap, same bilerp, same
    channel mean), so the 2-D raster path agrees with the render wherever the
    coordinates are computable there — i.e. whenever ``u``/``v`` are expressed in
    :data:`X`/:data:`Y`/:data:`Z`; bare :data:`U`/:data:`V` stay emit-only, as they
    are everywhere else.

    ``encoding`` defaults to ``"linear"`` (not ``"srgb"``, the default for a
    :class:`~loom.scene.Texture` *skin*) because a value used as a NUMBER wants the
    stored levels, not a de-gamma'd colour — the same advice ftrace gives for its
    scalar maps.
    """

    _cache: dict = {}          # path -> decoded float32 HxWx3 in [0,1] (LINEAR)

    def __init__(self, path, *, u=None, v=None, name: str = None,
                 encoding: str = "linear", filter: str = "bilinear",
                 wrap: str = "repeat") -> None:
        self.path = str(path).replace("\\", "/")
        self.u = _coerce(U if u is None else u)
        self.v = _coerce(V if v is None else v)
        if encoding not in ("srgb", "linear"):
            raise ValueError('image encoding must be "srgb" or "linear"')
        if filter not in ("bilinear", "nearest"):
            raise ValueError('image filter must be "bilinear" or "nearest"')
        if wrap not in ("repeat", "clamp", "mirror"):
            raise ValueError('image wrap must be "repeat", "clamp" or "mirror"')
        self.encoding = encoding
        self.filter = filter
        self.wrap = wrap
        self.name = name if name is not None else self._auto_name()

    # ---- texture declaration ---------------------------------------------
    def _auto_name(self) -> str:
        """A deterministic ftsl identifier for this image+sampler settings.

        Two ``Image`` leaves over the same file with the same sampler settings get
        the SAME name, so they share one ``texture`` block; differing settings get
        different names, so they don't silently collide."""
        import hashlib, re, os
        stem = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(self.path))[0])
        key = f"{self.path}|{self.encoding}|{self.filter}|{self.wrap}"
        return f"img_{stem}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"

    def _image_texture(self):
        from .scene import Texture   # lazy: scene.py is the higher layer
        return Texture(self.name, self.path, encoding=self.encoding,
                       filter=self.filter, wrap=self.wrap)

    # ---- tree ------------------------------------------------------------
    def children(self):
        return (self.u, self.v)

    def _rebuild(self, new_children):
        return Image(self.path, u=new_children[0], v=new_children[1],
                     name=self.name, encoding=self.encoding,
                     filter=self.filter, wrap=self.wrap)

    # ---- emit ------------------------------------------------------------
    def emit(self, coords, ctx) -> str:
        return f"tex:{self.name}({self.u.emit(coords, ctx)},{self.v.emit(coords, ctx)})"

    # ---- numpy twin (exact port of Texture::sampleRgb + scalarAt) --------
    @classmethod
    def _load(cls, path: str, encoding: str):
        key = (path, encoding)
        hit = cls._cache.get(key)
        if hit is not None:
            return hit
        from PIL import Image as _PILImage
        with _PILImage.open(path) as im:
            a = _np.asarray(im.convert("RGB"), dtype=_np.float32) / 255.0
        if encoding == "srgb":       # sRGB EOTF, matching ftrace's texture decode
            a = _np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        cls._cache[key] = a
        return a

    def _wrap_index(self, i, n):
        if self.wrap == "clamp":
            return _np.clip(i, 0, n - 1)
        if self.wrap == "mirror":
            period = 2 * n
            m = _np.mod(i, period)
            return _np.where(m < n, m, period - 1 - m)
        return _np.mod(i, n)         # repeat

    def eval_np(self, coords, clock, cache):
        if _np is None:              # pragma: no cover
            raise ImportError("Image.eval_np needs numpy")
        u = _np.asarray(self.u.eval_np(coords, clock, cache), dtype=_np.float64)
        v = _np.asarray(self.v.eval_np(coords, clock, cache), dtype=_np.float64)
        img = self._load(self.path, self.encoding)
        h, w = img.shape[0], img.shape[1]
        if self.filter == "nearest":
            x = self._wrap_index(_np.floor(u * w).astype(_np.int64), w)
            y = self._wrap_index(_np.floor((1.0 - v) * h).astype(_np.int64), h)
            c = img[y, x]
        else:
            tu = u * w - 0.5
            tv = (1.0 - v) * h - 0.5      # ftrace flips v (OBJ convention)
            flx = _np.floor(tu)
            fly = _np.floor(tv)
            fx = (tu - flx)[..., None]
            fy = (tv - fly)[..., None]
            x0 = self._wrap_index(flx.astype(_np.int64), w)
            x1 = self._wrap_index(flx.astype(_np.int64) + 1, w)
            y0 = self._wrap_index(fly.astype(_np.int64), h)
            y1 = self._wrap_index(fly.astype(_np.int64) + 1, h)
            a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
            b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
            c = a * (1 - fy) + b * fy
        return c.mean(axis=-1)            # Texture::scalarAt = mean of linear rgb


class VolumeField(SpatialExpr):
    """An **imported volume sampled as a term inside a formula** — the 3-D twin
    of :class:`Image`, and the piece that makes loom's *read → transform → write*
    volume workflow (roadmap §E4) actually compose.

    ``VolumeField("cloud.nvdb")`` is a scalar leaf whose value at a world point is the
    grid's trilinearly-interpolated density there, so a real captured volume can
    be an **operand** rather than a whole asset you can only pass through::

        cloud = VolumeField("cloud.nvdb")
        write_volume("thick.vdb", box=cloud.box, res=128,
                     density=cloud * (0.5 + 0.5 * sin(20 * Y)))

    Everything the spatial algebra already does now applies to volumes for free:

    * **value ops / modulation** — multiply, add, threshold, ``mix`` two volumes,
      drive one by a procedural field or an animated :class:`Signal` coefficient;
    * **warping** — ``x``/``y``/``z`` are ordinary sub-expressions, exactly like
      :class:`Image`'s ``u``/``v``, so ``VolumeField(p, x=X + 0.1 * sin(10 * Z))``
      bends the volume.  Note the convention every resampler uses: the
      coordinate expressions map the **destination** point back to the point
      *sampled in the source*, so a warp is authored as its inverse map;
    * **meshing** — :mod:`loom.mcubes` takes any callable field, so
      ``mcubes.iso_mesh(VolumeField("cloud.nvdb"), box, res, level=0.5)`` extracts an
      isosurface of imported data;
    * **resampling** — :func:`loom.vdbio.bake_field` / ``write_volume`` discretise
      onto any box and resolution you like, which is all "resample a grid" ever
      was.

    **Placement is lossless.**  :meth:`translated` / :meth:`scaled` /
    :meth:`rotated` / :meth:`fitted` do **not** resample: they compose a
    world-space affine onto the grid's own index→world transform
    (:meth:`~loom.vdbio.VdbTransform.premultiplied`), so moving a volume around
    costs no interpolation and no precision.  Error enters exactly once, at the
    final bake — loom's "keep everything as functions; discretize last" rule.

    **One backend, not two.**  Unlike every other leaf in this module,
    :meth:`emit` *raises*: ftrace's pattern VM has no volume-sampling opcode, so
    there is no ftsl string this could honestly become.  Rather than emit
    something that silently means something else, a ``VolumeField`` is bake-only —
    render it by writing a ``.vdb`` (``density vdb:<path>``), which is ftrace's
    actual volume path.  :meth:`eval_np` is a faithful port of ftrace's
    ``VdbGrid::sample``, so what you bake is what it renders.
    """

    _cache: dict = {}          # (path, grid) -> ReadGrid

    def __init__(self, path, *, grid: str = None, x=None, y=None, z=None,
                 outside: float = None, clamp_negative: bool = False,
                 _read=None) -> None:
        self.path = str(path).replace("\\", "/")
        self.grid = grid
        self.x = _coerce(X if x is None else x)
        self.y = _coerce(Y if y is None else y)
        self.z = _coerce(Z if z is None else z)
        #: Value beyond the lattice; ``None`` means the grid's own background.
        self.outside = None if outside is None else float(outside)
        #: Clamp samples to ``>= 0`` as ftrace does for a fog density.
        self.clamp_negative = bool(clamp_negative)
        self._read = _read     # an already-placed ReadGrid, if repositioned

    # ---- the decoded grid -------------------------------------------------
    @classmethod
    def _load(cls, path: str, grid):
        """The :class:`~loom.vdbio.ReadGrid` behind ``path``, cached by file.

        Accepts ``.vdb`` and ``.nvdb`` alike — :func:`loom.vdbio.read_vdb_grids`
        dispatches on the file's magic.
        """
        key = (path, grid)
        hit = cls._cache.get(key)
        if hit is not None:
            return hit
        from .vdbio import read_vdb_grids      # lazy: vdbio is the heavier layer
        grids = read_vdb_grids(path)
        if not grids:
            raise ValueError(f"VolumeField: {path!r} holds no grids")
        if grid is None:
            if len(grids) > 1 and "density" in grids:
                g = grids["density"]           # the conventional default name
            elif len(grids) > 1:
                raise ValueError(
                    f"VolumeField: {path!r} holds {len(grids)} grids "
                    f"({', '.join(sorted(grids))}) and none is named 'density'; "
                    "pass grid='<name>' to pick one")
            else:
                g = next(iter(grids.values()))
        else:
            if grid not in grids:
                raise ValueError(
                    f"VolumeField: {path!r} has no grid named {grid!r} "
                    f"(it has: {', '.join(sorted(grids))})")
            g = grids[grid]
        cls._cache[key] = g
        return g

    @property
    def read_grid(self):
        """The :class:`~loom.vdbio.ReadGrid` this leaf samples, including any
        placement applied by :meth:`translated` & co."""
        return self._read if self._read is not None else self._load(self.path, self.grid)

    @property
    def box(self):
        """The volume's axis-aligned world AABB — a ready-made ``box`` argument
        for :func:`~loom.vdbio.write_volume` / :mod:`loom.mcubes`.  Defined for a
        rotated grid too (it bounds the eight index-box corners)."""
        return self.read_grid.world_box

    # ---- lossless placement ----------------------------------------------
    def _placed(self, m=None, d=None) -> "VolumeField":
        g = self.read_grid
        return VolumeField(
            self.path, grid=self.grid, x=self.x, y=self.y, z=self.z,
            outside=self.outside, clamp_negative=self.clamp_negative,
            _read=g.with_transform(g.transform.premultiplied(m, d)))

    def translated(self, dx: float, dy: float = None, dz: float = None) -> "VolumeField":
        """This volume moved by ``(dx, dy, dz)`` in world space (no resampling).
        A single argument shifts all three axes."""
        if dy is None and dz is None:
            dy = dz = dx
        return self._placed(None, (dx, dy, dz))

    def scaled(self, factor, center=None) -> "VolumeField":
        """This volume scaled about ``center`` (default the world origin).
        ``factor`` is a scalar or a per-axis triple.  No resampling."""
        try:
            sx, sy, sz = (float(v) for v in factor)
        except TypeError:
            sx = sy = sz = float(factor)
        m = (sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, sz)
        c = (0.0, 0.0, 0.0) if center is None else tuple(float(v) for v in center)
        d = (c[0] - sx * c[0], c[1] - sy * c[1], c[2] - sz * c[2])
        return self._placed(m, d)

    def rotated(self, degrees: float, axis=(0.0, 1.0, 0.0), center=None) -> "VolumeField":
        """This volume rotated ``degrees`` about ``axis`` through ``center``
        (default the volume's own world centre).  No resampling — a rotation
        only moves the lattice, so not one voxel changes."""
        import math
        ax, ay, az = (float(v) for v in axis)
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n == 0.0:
            raise ValueError("VolumeField.rotated: axis must be nonzero")
        ax, ay, az = ax / n, ay / n, az / n
        th = math.radians(float(degrees))
        c, s = math.cos(th), math.sin(th)
        k = 1.0 - c
        m = (c + ax * ax * k,      ax * ay * k - az * s, ax * az * k + ay * s,
             ay * ax * k + az * s, c + ay * ay * k,      ay * az * k - ax * s,
             az * ax * k - ay * s, az * ay * k + ax * s, c + az * az * k)
        if center is None:
            b = self.box
            center = ((b[0] + b[3]) * 0.5, (b[1] + b[4]) * 0.5, (b[2] + b[5]) * 0.5)
        cx, cy, cz = (float(v) for v in center)
        d = (cx - (m[0] * cx + m[1] * cy + m[2] * cz),
             cy - (m[3] * cx + m[4] * cy + m[5] * cz),
             cz - (m[6] * cx + m[7] * cy + m[8] * cz))
        return self._placed(m, d)

    def fitted(self, box) -> "VolumeField":
        """This volume scaled and shifted so its world AABB becomes ``box`` —
        the "drop this asset into my scene's unit cube" op.  ``box`` takes the
        usual :mod:`loom.mcubes` forms (a half-size scalar, a 3-tuple, or a full
        6-tuple).  Each axis is fitted **independently**, so the volume fills
        ``box`` exactly — this is not a uniform scale to the tightest axis, and
        it will change the aspect ratio if ``box`` has a different one.  No
        resampling."""
        from .mcubes import _norm_bounds
        t = _norm_bounds(box)
        b = self.box
        sc, off = [], []
        for i in range(3):
            span = b[i + 3] - b[i]
            want = t[i + 3] - t[i]
            s = 1.0 if span == 0.0 else want / span
            sc.append(s)
            off.append(t[i] - s * b[i])
        return self._placed((sc[0], 0.0, 0.0, 0.0, sc[1], 0.0, 0.0, 0.0, sc[2]), off)

    # ---- tree -------------------------------------------------------------
    def children(self):
        return (self.x, self.y, self.z)

    def _rebuild(self, new_children):
        return VolumeField(
            self.path, grid=self.grid, x=new_children[0], y=new_children[1],
            z=new_children[2], outside=self.outside,
            clamp_negative=self.clamp_negative, _read=self._read)

    # ---- emit (there isn't one) ------------------------------------------
    def emit(self, coords, ctx) -> str:
        raise TypeError(
            f"VolumeField({self.path!r}) cannot be emitted as an ftsl expression: "
            "ftrace's pattern VM has no volume-sampling op. Bake it instead — "
            "loom.vdbio.write_volume(...)/bake_field(...) discretise the field to "
            "a .vdb that ftrace reads with `density vdb:<path>`.")

    # ---- numpy twin (port of VdbGrid::sample, src/vdbgrid.h) --------------
    def eval_np(self, coords, clock, cache):
        if _np is None:              # pragma: no cover
            raise ImportError("VolumeField.eval_np needs numpy")
        # `sample` broadcasts internally, so a constant coordinate (or a warp
        # that collapses one axis) needs no special handling here.
        return self.read_grid.sample(
            self.x.eval_np(coords, clock, cache),
            self.y.eval_np(coords, clock, cache),
            self.z.eval_np(coords, clock, cache),
            outside=self.outside, clamp_negative=self.clamp_negative)

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        g = f", grid={self.grid!r}" if self.grid else ""
        return f"VolumeField({self.path!r}{g})"


class GridSample(SpatialExpr):
    """A loom :class:`~loom.data.Grid` **sampled as a term inside a formula** — the
    spatial, *renderable* twin of :class:`~loom.interp.GridField`.

    ``grid(X, Y)`` builds one of these (``grid(u, v)`` with temporal arguments still
    builds the ``GridField`` Signal), so measured data becomes an operand of an
    ordinary field expression::

        heat = Grid([[0, 1], [1, 0]], lo=-1, hi=1)
        rough = 0.05 + 0.5 * heat(X, Z) * (0.5 + 0.5 * sin(20 * Y))

    **Both backends, as everywhere else in this module.**  :meth:`emit` writes
    ftrace's ``grid:<name>(c0, …)`` table call (``PatOp::Grid``) and the companion
    ``grid { … }`` block is collected by :meth:`SpatialExpr.table_decls` —
    :meth:`loom.Scene.add` picks it up, so the author never declares it.
    :meth:`eval_np` is a vectorised port of ftrace's ``patGridSample`` /
    ``patGridCellFrac``, which are themselves the documented twins of loom's own
    :func:`loom.interp._grid_weights` / :func:`loom.interp._cell_base_frac` — so the
    2-D raster preview, the temporal ``GridField`` and the render all agree.

    **Placement is folded into the coordinates.**  ftsl's ``grid`` block has no
    transform, so a :meth:`~loom.data._Transformable.transformed` Grid inverse-maps
    the *query* through its placement at construction
    (:meth:`loom.transform.Transform.inverse_apply_spatial`) — the same remap
    :func:`loom.interp._local_query` applies on the temporal path, so moving the data
    object changes what a fixed world point reads back, identically in both.

    **What ftrace cannot express is refused, not approximated.**  A vector-valued
    grid (``PatGrid`` stores scalar floats), ``interp="cubic"`` (``patGridSample`` is
    N-linear only), ``on_outside="raise"`` (a renderer samples millions of times a
    frame and cannot throw) and more than four axes (``PAT_ND_MAX_DIM``) all raise
    here rather than emit something that means something else.
    """

    #: ftrace's ``PatGrid`` axis cap (``PAT_ND_MAX_DIM``, ``src/pattern.h``).
    MAX_DIM = 4

    def __init__(self, grid, coords, *, name: str = None,
                 interp: str = "linear", on_outside: str = "clamp",
                 _local: bool = False) -> None:
        from .interp import _parse_grid_interp, _parse_on_outside  # lazy: cycle
        self.grid = grid
        if getattr(grid, "is_vector", False):
            raise TypeError(
                "a vector-valued Grid cannot be sampled as an ftsl term: ftrace's "
                "PatGrid stores scalar floats. Sample one channel into its own "
                "scalar Grid, or keep the vector field on the temporal path "
                "(VecGridField).")
        if grid.ndim > self.MAX_DIM:
            raise ValueError(
                f"a {grid.ndim}-D Grid exceeds ftrace's {self.MAX_DIM}-axis table "
                f"limit (PAT_ND_MAX_DIM)")
        if _parse_grid_interp(interp):
            raise ValueError(
                "interp='cubic' has no ftsl spelling: ftrace's patGridSample is "
                "separable N-linear only. Use interp='linear' to render, or stay on "
                "the temporal GridField for Catmull-Rom.")
        self.interp = "linear"
        out = _parse_on_outside(on_outside)
        if out == "raise":
            raise ValueError(
                "on_outside='raise' has no ftsl spelling: a renderer samples "
                "millions of times per frame and cannot throw. ftrace's authoring "
                "guard is the load-time domain check; choose 'clamp', 'wrap' or "
                "'extrapolate' for the render.")
        self.on_outside = out
        cs = tuple(_coerce(c) for c in coords)
        if len(cs) != grid.ndim:
            raise ValueError(
                f"grid sample needs one coordinate per axis: {grid.ndim} expected, "
                f"{len(cs)} given")
        if not _local:
            xf = getattr(grid, "xf", None)
            if xf is not None and not xf.is_identity():
                cs = tuple(xf.inverse_apply_spatial(cs))
        self.coords = cs
        self.name = name if name is not None else self._auto_name()

    # ---- grid declaration -------------------------------------------------
    def _auto_name(self) -> str:
        """A deterministic ftsl identifier for this grid + out-of-domain policy.

        Two leaves over the *same* :class:`~loom.data.Grid` with the same
        ``on_outside`` share one declaration; a different policy gets a different
        name because ``outside`` lives on ftsl's ``grid`` block, not on the sample
        call, so the two cannot share one block."""
        return f"grid_{self.grid.id}_{self.on_outside}"

    def _table_decl(self):
        from .scene import GridDecl   # lazy: scene.py is the higher layer
        return GridDecl(self.name, self.grid, outside=self.on_outside)

    # ---- tree ------------------------------------------------------------
    def children(self):
        return self.coords

    def _rebuild(self, new_children):
        # the coordinates are already in the grid's local frame (any placement was
        # folded in at construction), so rebuild without re-applying it.
        return GridSample(self.grid, tuple(new_children), name=self.name,
                          interp=self.interp, on_outside=self.on_outside,
                          _local=True)

    # ---- emit ------------------------------------------------------------
    def emit(self, coords, ctx) -> str:
        args = ",".join(c.emit(coords, ctx) for c in self.coords)
        return f"grid:{self.name}({args})"

    # ---- numpy twin (port of patGridSample / patGridCellFrac) ------------
    def _cell_base_frac_np(self, axis: int, coord):
        """Vectorised :func:`loom.interp._cell_base_frac` — returns ``(i0, frac)``."""
        g = self.grid
        n = g.shape[axis]
        lo, hi = g.lo[axis], g.hi[axis]
        z = _np.zeros_like(coord, dtype=_np.float64)
        if n <= 1 or hi == lo:
            return z.astype(_np.int64), z
        p = (coord - lo) / (hi - lo) * (n - 1)
        if self.on_outside == "wrap":
            span = float(n - 1)
            p = p - _np.floor(p / span) * span         # fold into [0, n-1)
            i = _np.floor(p).astype(_np.int64)
            seam = i >= (n - 1)                        # numerical guard at the seam
            f = _np.where(seam, 0.0, p - i)
            return _np.where(seam, 0, i), f
        i = _np.floor(p).astype(_np.int64)
        f = p - i
        low = p <= 0.0
        high = p >= (n - 1)
        if self.on_outside == "extrapolate":
            f = _np.where(low, _np.where(p < 0.0, p, 0.0), f)
            f = _np.where(high, _np.where(p > (n - 1), p - (n - 2), 1.0), f)
        else:                                          # clamp / edge-extend
            f = _np.where(low, 0.0, f)
            f = _np.where(high, 1.0, f)
        i = _np.where(low, 0, i)
        i = _np.where(high, n - 2, i)
        return i, f

    def eval_np(self, coords, clock, cache):
        if _np is None:              # pragma: no cover
            raise ImportError("GridSample.eval_np needs numpy")
        g = self.grid
        cs = [_np.asarray(c.eval_np(coords, clock, cache), dtype=_np.float64)
              for c in self.coords]
        shape = _np.broadcast(*cs).shape if len(cs) > 1 else cs[0].shape
        cs = [_np.broadcast_to(c, shape).astype(_np.float64) for c in cs]
        base, frac = [], []
        for a in range(g.ndim):
            i, f = self._cell_base_frac_np(a, cs[a])
            base.append(i)
            frac.append(f)
        vals = _np.array([v.at(clock, cache) for v in g.values], dtype=_np.float64)
        wrap = (self.on_outside == "wrap")
        acc = _np.zeros(shape, dtype=_np.float64)
        for corner in range(1 << g.ndim):
            w = _np.ones(shape, dtype=_np.float64)
            flat = _np.zeros(shape, dtype=_np.int64)
            for a in range(g.ndim):
                n = g.shape[a]
                up = (corner >> a) & 1
                w = w * (frac[a] if up else (1.0 - frac[a]))
                idx = base[a] + up
                idx = _np.mod(idx, n - 1) if (wrap and n > 1) else _np.clip(idx, 0, n - 1)
                flat = flat * n + idx          # C order: axis 0 outermost
            acc = acc + w * vals[flat]
        return acc

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return f"GridSample({self.name!r}, ndim={self.grid.ndim})"


class ScatterSample(SpatialExpr):
    """A loom :class:`~loom.data.Scatter` **sampled as a term inside a formula** — the
    ragged sibling of :class:`GridSample`, and the spatial twin of
    :class:`~loom.interp.ScatterField`.

    ``scatter(X, Y)`` builds one of these (a temporal query still builds the
    ``ScatterField`` Signal).  :meth:`emit` writes ftrace's ``scatter:<name>(c0, …)``
    table call (``PatOp::Scatter``) and the companion ``scatter { … }`` block is
    collected by :meth:`SpatialExpr.table_decls`; :meth:`eval_np` is a vectorised
    port of ``patScatterSample``, itself the documented twin of loom's
    :func:`loom.interp._shepard_weights`.

    Same refusals as :class:`GridSample` where ftrace cannot follow: vector-valued
    samples (``PatScatter`` stores scalar floats) and more than four dimensions.
    Unlike a grid there is no out-of-domain policy — Shepard weighting is defined
    everywhere — so ``power``/``eps`` are the only sampler settings, and because they
    live on the *block* they take part in the auto-name.
    """

    MAX_DIM = 4

    def __init__(self, scatter, coords, *, name: str = None,
                 power: float = 2.0, eps: float = 1e-9,
                 _local: bool = False) -> None:
        self.scatter = scatter
        if getattr(scatter, "is_vector", False):
            raise TypeError(
                "a vector-valued Scatter cannot be sampled as an ftsl term: ftrace's "
                "PatScatter stores scalar floats. Split the channels into scalar "
                "Scatters, or stay on the temporal path (VecScatterField).")
        if scatter.dim > self.MAX_DIM:
            raise ValueError(
                f"a {scatter.dim}-D Scatter exceeds ftrace's {self.MAX_DIM}-axis "
                f"table limit (PAT_ND_MAX_DIM)")
        self.power = float(power)
        self.eps = float(eps)
        if not (self.power > 0.0):
            raise ValueError("scatter power must be > 0")
        if self.eps < 0.0:
            raise ValueError("scatter eps must be >= 0")
        cs = tuple(_coerce(c) for c in coords)
        if len(cs) != scatter.dim:
            raise ValueError(
                f"scatter sample needs one coordinate per dimension: {scatter.dim} "
                f"expected, {len(cs)} given")
        if not _local:
            xf = getattr(scatter, "xf", None)
            if xf is not None and not xf.is_identity():
                cs = tuple(xf.inverse_apply_spatial(cs))
        self.coords = cs
        self.name = name if name is not None else self._auto_name()

    def _auto_name(self) -> str:
        """``scatter_<id>_<hash>`` — the hash covers ``power``/``eps``, which live on
        the emitted block rather than on the sample call, so two leaves that weight
        differently cannot share one declaration."""
        import hashlib
        key = f"{self.power!r}|{self.eps!r}"
        return f"scatter_{self.scatter.id}_{hashlib.sha1(key.encode()).hexdigest()[:6]}"

    def _table_decl(self):
        from .scene import ScatterDecl   # lazy: scene.py is the higher layer
        return ScatterDecl(self.name, self.scatter, power=self.power, eps=self.eps)

    def children(self):
        return self.coords

    def _rebuild(self, new_children):
        return ScatterSample(self.scatter, tuple(new_children), name=self.name,
                             power=self.power, eps=self.eps, _local=True)

    def emit(self, coords, ctx) -> str:
        args = ",".join(c.emit(coords, ctx) for c in self.coords)
        return f"scatter:{self.name}({args})"

    def eval_np(self, coords, clock, cache):
        if _np is None:              # pragma: no cover
            raise ImportError("ScatterSample.eval_np needs numpy")
        sc = self.scatter
        cs = [_np.asarray(c.eval_np(coords, clock, cache), dtype=_np.float64)
              for c in self.coords]
        shape = _np.broadcast(*cs).shape if len(cs) > 1 else cs[0].shape
        cs = [_np.broadcast_to(c, shape).astype(_np.float64) for c in cs]
        half = 0.5 * self.power
        num = _np.zeros(shape, dtype=_np.float64)
        den = _np.zeros(shape, dtype=_np.float64)
        hit = _np.zeros(shape, dtype=bool)
        hitval = _np.zeros(shape, dtype=_np.float64)
        for pos, val in zip(sc.positions, sc.values):
            p = pos.at(clock, cache)
            v = float(val.at(clock, cache))
            d2 = _np.zeros(shape, dtype=_np.float64)
            for a in range(sc.dim):
                d = cs[a] - float(p[a])
                d2 = d2 + d * d
            # coincidence: the FIRST sample within eps wins, exactly as the scalar
            # path (which returns immediately) and ftrace's patScatterSample do.
            co = (d2 <= self.eps) & ~hit
            hitval = _np.where(co, v, hitval)
            hit = hit | co
            safe = _np.where(d2 <= self.eps, 1.0, d2)
            w = (1.0 / safe) if self.power == 2.0 else _np.power(safe, -half)
            w = _np.where(d2 <= self.eps, 0.0, w)
            num = num + w * v
            den = den + w
        blend = _np.where(den > 0.0, num / _np.where(den > 0.0, den, 1.0), 0.0)
        return _np.where(hit, hitval, blend)

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return f"ScatterSample({self.name!r}, dim={self.scatter.dim})"


class _Time(SpatialExpr):
    """The loop phase ``t`` in [0, 1) at the current frame."""

    def emit(self, coords, ctx) -> str:
        return f"({fmt(ctx.clock.t)})"

    def eval_np(self, coords, clock, cache):
        return clock.t

    def _is_time(self) -> bool:
        return True


class _Sig(SpatialExpr):
    """A temporal Signal used as an animated coefficient (baked per frame)."""

    def __init__(self, sig: Signal) -> None:
        self.sig = sig

    def emit(self, coords, ctx) -> str:
        return f"({fmt(self.sig.at(ctx.clock, ctx.cache))})"

    def eval_np(self, coords, clock, cache):
        return self.sig.at(clock, cache)

    def _time_signal(self):
        return self.sig

    def _is_time(self) -> bool:
        return True


class SigAt(SpatialExpr):
    """A temporal :class:`~loom.signals.core.Signal` sampled at a **spatially
    varying** phase — the 4-D *time shear*.

    :class:`_Sig` (what a bare ``Signal`` coerces to) bakes one number per frame:
    the whole field shares the modulator's current value.  ``SigAt`` instead reads
    the modulator at a phase that is *itself* a field, so different points of space
    see different moments::

        from loom import SigAt, X, T, Sine
        wave = SigAt(Sine(cycles=3), T - X / 4.0)   # a wave whose phase lags with x

    This is the spatial half of :mod:`loom.signals.retime`, and it rests on the
    same fact: a ``Signal`` is a pure function of a clock, so evaluating it at
    another phase is just building that clock.  ``wrap`` is passed to
    :func:`~loom.signals.retime.retimed_clock` (default: wrap iff the clock is a
    closed loop, which keeps a sheared loop seamless).

    **Single-backend, deliberately.**  ftrace's pattern VM evaluates a formula
    per hit with no access to loom's modulator DAG, so a per-point signal read has
    no ftsl spelling; :meth:`emit` raises rather than bake a wrong constant.  Use
    it on the numeric path — :func:`loom.mesh_field`, :func:`loom.vdbio.bake_field`
    / :func:`loom.vdbio.write_volume`, the 2-D canvas — i.e. discretise the sheared
    field and hand *that* to the renderer.

    **Cost.**  The signal is evaluated once per *distinct* phase in the query, so a
    smooth shear over an ``N``-point grid costs ``N`` graph evaluations.  Pass
    ``quantize=k`` to snap the phase to ``k`` levels (``round(t*k)/k``) and pay only
    ``k`` — ``quantize=clock.frames`` reproduces "shear by whole frames" exactly.
    """

    def __init__(self, sig, when, *, wrap=None, quantize=None) -> None:
        self.sig = as_signal(sig)
        self.when = _coerce(when)
        self.wrap = None if wrap is None else bool(wrap)
        if quantize is not None:
            quantize = int(quantize)
            if quantize < 1:
                raise ValueError("SigAt quantize must be >= 1 (or None)")
        self.quantize = quantize

    def children(self):
        return (self.when,)

    def _rebuild(self, new_children):
        return SigAt(self.sig, new_children[0], wrap=self.wrap,
                     quantize=self.quantize)

    def _time_signal(self):
        return self.sig

    def _is_time(self) -> bool:
        return True

    def emit(self, coords, ctx) -> str:
        raise TypeError(
            "SigAt cannot be emitted as an ftsl expression: ftrace evaluates a "
            "pattern per hit and has no access to loom's modulator DAG, so a "
            "per-point signal read has no ftsl spelling (baking one number would "
            "silently drop the shear). Discretise it instead — loom.mesh_field(...) "
            "or loom.vdbio.bake_field(...)/write_volume(...) — and render the result.")

    def eval_np(self, coords, clock, cache):
        if _np is None:  # pragma: no cover
            raise ImportError("SigAt.eval_np needs numpy")
        ph = _np.asarray(self.when.eval_np(coords, clock, cache), dtype=float)
        if not _np.all(_np.isfinite(ph)):
            raise ValueError("SigAt: sample phase field has non-finite values")
        if self.quantize is not None:
            ph = _np.round(ph * self.quantize) / self.quantize
        uniq, inv = _np.unique(ph, return_inverse=True)
        vals = _np.empty(uniq.shape, dtype=float)
        for i, t in enumerate(uniq):
            rc = retimed_clock(clock, float(t), self.wrap)
            sub = None if cache is None else cache.scope((id(self), clock.frame, rc.t))
            vals[i] = self.sig.at(rc, sub)
        return vals[inv].reshape(ph.shape)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

_BINOPS = {"+": operator.add, "-": operator.sub,
           "*": operator.mul, "/": operator.truediv}


class _Bin(SpatialExpr):
    def __init__(self, op: str, a: SpatialExpr, b: SpatialExpr) -> None:
        self.op = op
        self.a = a
        self.b = b

    def children(self):
        return (self.a, self.b)

    def _rebuild(self, new_children):
        return _Bin(self.op, new_children[0], new_children[1])

    def emit(self, coords, ctx) -> str:
        return f"({self.a.emit(coords, ctx)}{self.op}{self.b.emit(coords, ctx)})"

    def eval_np(self, coords, clock, cache):
        return _BINOPS[self.op](self.a.eval_np(coords, clock, cache),
                                self.b.eval_np(coords, clock, cache))


class _Neg(SpatialExpr):
    def __init__(self, a: SpatialExpr) -> None:
        self.a = a

    def children(self):
        return (self.a,)

    def _rebuild(self, new_children):
        return _Neg(new_children[0])

    def emit(self, coords, ctx) -> str:
        return f"(-({self.a.emit(coords, ctx)}))"

    def eval_np(self, coords, clock, cache):
        return -self.a.eval_np(coords, clock, cache)


class _Fn(SpatialExpr):
    """An ftsl builtin call — ``name`` must exist in ``src/pattern.h``; ``npfn`` is
    its numpy twin (same argument order/semantics)."""

    def __init__(self, name: str, args: Sequence[SpatialExpr], npfn: Callable) -> None:
        self.name = name
        self.args = list(args)
        self.npfn = npfn

    def children(self):
        return tuple(self.args)

    def _rebuild(self, new_children):
        return _Fn(self.name, new_children, self.npfn)

    def emit(self, coords, ctx) -> str:
        inner = ",".join(a.emit(coords, ctx) for a in self.args)
        return f"{self.name}({inner})"

    def eval_np(self, coords, clock, cache):
        return self.npfn(*[a.eval_np(coords, clock, cache) for a in self.args])


# ---------------------------------------------------------------------------
# leaf singletons + math functions (each emits a real ftsl pattern builtin)
# ---------------------------------------------------------------------------

X = Surface("x", axis=0)
Y = Surface("y", axis=1)
Z = Surface("z", axis=2)
U = Surface("u")                    # surface param (emit-only)
V = Surface("v")                    # surface param (emit-only)
A = Surface("a", emit_ok=False)     # albedo binding placeholder (no ftrace var)
T = _Time()


def _mk(name: str, npfn: Callable) -> Callable[..., SpatialExpr]:
    def f(*args):
        return _Fn(name, [_coerce(a) for a in args], npfn)
    f.__name__ = name
    return f


def _np_step(edge, x):
    return _np.where(_np.asarray(x) >= edge, 1.0, 0.0)


def _np_clamp(x, lo, hi):
    return _np.minimum(hi, _np.maximum(lo, x))


def _np_mix(a, b, t):
    return a + (b - a) * t


def _np_smoothstep(e0, e1, x):
    t = _np.clip((_np.asarray(x) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _np_fract(x):
    return x - _np.floor(x)


# unary
sin = _mk("sin", lambda x: _np.sin(x))
cos = _mk("cos", lambda x: _np.cos(x))
tan = _mk("tan", lambda x: _np.tan(x))
sqrt = _mk("sqrt", lambda x: _np.sqrt(x))
exp = _mk("exp", lambda x: _np.exp(x))
log = _mk("log", lambda x: _np.log(x))
floor = _mk("floor", lambda x: _np.floor(x))
fract = _mk("fract", _np_fract)
sign = _mk("sign", lambda x: _np.sign(x))
saturate = _mk("saturate", lambda x: _np.clip(x, 0.0, 1.0))
sabs = _mk("abs", lambda x: _np.abs(x))       # ``abs`` shadows the builtin -> sabs
# binary
smin = _mk("min", lambda a, b: _np.minimum(a, b))
smax = _mk("max", lambda a, b: _np.maximum(a, b))
spow = _mk("pow", lambda a, b: _np.power(a, b))
atan2 = _mk("atan2", lambda a, b: _np.arctan2(a, b))
step = _mk("step", _np_step)
# ternary
clamp = _mk("clamp", _np_clamp)
mix = _mk("mix", _np_mix)
smoothstep = _mk("smoothstep", _np_smoothstep)


# ---------------------------------------------------------------------------
# preset patterns (shared 2-D-numeric / 3-D-emitted) — compose your own too
# ---------------------------------------------------------------------------

def waves(freq: Union[SpatialExpr, Signal, Number] = 1.0, axis: int = 0) -> SpatialExpr:
    """1-D sinusoid along one axis, remapped to [0, 1]."""
    c = (X, Y, Z)[axis]
    return 0.5 + 0.5 * sin(freq * c)


Coord3 = Tuple[Union[SpatialExpr, Signal, Number],
               Union[SpatialExpr, Signal, Number],
               Union[SpatialExpr, Signal, Number]]


def _offset(coord: SpatialExpr, c: Union[SpatialExpr, Signal, Number]) -> SpatialExpr:
    # ``coord - c``, but a literal 0 offset drops out — so a default-centered
    # source emits the plain ``x``/``y``/``z`` (byte-identical to the old rings)
    # and costs no per-pixel subtraction.
    if isinstance(c, (int, float)) and c == 0.0:
        return coord
    return coord - c


def _radial(freq: Union[SpatialExpr, Signal, Number],
            center: Coord3 = (0.0, 0.0, 0.0)) -> SpatialExpr:
    """Signed radial wave ``sin(freq * |p - center|)`` from a point source at
    ``center``.  ``freq`` and each ``center`` component may be a number or an
    animated :class:`~loom.signals.core.Signal` (baked per frame)."""
    cx, cy, cz = center
    dx, dy, dz = _offset(X, cx), _offset(Y, cy), _offset(Z, cz)
    return sin(freq * sqrt(dx * dx + dy * dy + dz * dz))


def rings(freq: Union[SpatialExpr, Signal, Number] = 1.0,
          center: Coord3 = (0.0, 0.0, 0.0)) -> SpatialExpr:
    """Concentric shells ``0.5 + 0.5 sin(freq |p - center|)`` in [0, 1] from a
    point source at ``center`` (default origin)."""
    return 0.5 + 0.5 * _radial(freq, center)


def interference(freq: Union[SpatialExpr, Signal, Number] = 1.0,
                 source_a: Coord3 = (-0.5, 0.0, 0.0),
                 source_b: Coord3 = (0.5, 0.0, 0.0)) -> SpatialExpr:
    """Two-source interference in [0, 1]: the superposition (sum) of two radial
    waves from ``source_a`` and ``source_b``.  Where the two path lengths differ
    by a constant the crests reinforce, tracing the classic hyperbolic two-slit
    fringes; feed a :class:`~loom.signals.core.Signal` into a source coordinate
    to move an emitter and the fringes sweep (loop-safe if it returns by whole
    cycles per loop).  This is the *spatial* counterpart of the temporal beat you
    get for free from ``Sine(cycles=a) + Sine(cycles=b)``."""
    return 0.5 + 0.25 * (_radial(freq, source_a) + _radial(freq, source_b))


def moire(freq: Union[SpatialExpr, Signal, Number] = 1.0,
          angle: Union[Signal, Number] = 0.2,
          freq2: Union[SpatialExpr, Signal, Number, None] = None) -> SpatialExpr:
    """Moiré in [0, 1]: the superposition of two line gratings, the second
    rotated by ``angle`` radians (and optionally ruled at its own ``freq2``).
    The slow beat between the two nearly-aligned rulings is the moiré envelope;
    an animated ``angle`` :class:`~loom.signals.core.Signal` rotates one grating
    and the fringes crawl."""
    f2 = freq if freq2 is None else freq2
    xr = cos(angle) * X - sin(angle) * Y      # X of the grating rotated by `angle`
    return 0.5 + 0.25 * (sin(freq * X) + sin(f2 * xr))


def checker(freq: Union[SpatialExpr, Signal, Number] = 1.0) -> SpatialExpr:
    """3-D checkerboard in [0, 1] (sign of the product of three sines)."""
    return 0.5 + 0.5 * sign(sin(freq * X) * sin(freq * Y) * sin(freq * Z))


def gyroid(freq: Union[SpatialExpr, Signal, Number] = 1.0) -> SpatialExpr:
    """Schoen gyroid field ``sin x cos y + sin y cos z + sin z cos x`` (an
    isosurface field at ``=0``, or a signed pattern)."""
    fx, fy, fz = freq * X, freq * Y, freq * Z
    return sin(fx) * cos(fy) + sin(fy) * cos(fz) + sin(fz) * cos(fx)


SPATIAL_PATTERNS = {
    "waves": waves, "rings": rings, "checker": checker, "gyroid": gyroid,
    "interference": interference, "moire": moire,
}
