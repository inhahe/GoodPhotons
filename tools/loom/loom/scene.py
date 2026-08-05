"""
Loom scene model — animatable geometry + materials + lights + camera, emitted
to ftrace's ``.ftsl`` scene language one frame at a time.

Every element field may be a plain number, a :class:`~loom.signals.core.Signal`,
or a :class:`~loom.signals.vector.VecSignal`, so the whole scene animates.  A
:class:`Scene` knows how to (a) collect every modulator root for a pre-render
:func:`~loom.signals.core.detect_signal_cycle` check and (b) emit the ``.ftsl``
text for a given :class:`~loom.signals.core.Clock`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from .signals.core import Signal, Clock, Cache, Const, detect_signal_cycle
from .signals.vector import VecSignal
from .interp import LoopCurve
from .data import PointPath
from .ftsl_emit import EmitCtx, num, vec3, fmt, fmt3, value_token, site_node
from . import sweep as _sweep


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Element:
    """Base scene element.  Emits ftsl text and exposes its modulator roots.

    An element may carry an optional :class:`~loom.transform.Transform` in ``xf``
    (position / size / rotation / skew, each animatable).  The transform is applied
    by the *container* (the :class:`Scene` or an enclosing :class:`Group`) when it
    emits the element — see :func:`emit_element` / :func:`element_roots` — so
    ``emit()`` always returns the element's *own* untransformed block and nesting a
    transformed element inside a transformed :class:`Group` composes correctly.
    """

    xf = None  # Optional[Transform]; applied by the container on emit

    def roots(self) -> List:
        """Every Signal / VecSignal stored on this element (for cycle checking).

        An axis-typed node (:mod:`loom.axes`) contributes the *same* lowered node
        that emission will evaluate (:func:`~loom.ftsl_emit.site_node` memoises
        it), so the cycle detector and the viewer's DAG panel see the whole graph.
        """
        out: List = []
        for v in vars(self).values():
            n = site_node(v)
            if n is not None:
                out.append(n)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        raise NotImplementedError

    def transformed(self, transform=None, *, translate=None, rotate=None,
                    scale=None, skew=None) -> "Element":
        """Attach a :class:`~loom.transform.Transform` (position / size / rotation /
        skew, all signal-modulatable) and return ``self`` for chaining::

            scene.add(Sphere((0, 0, 0), 1, "m").transformed(translate=(2, 0, 0),
                                                             scale=1.5))

        Pass a ready ``transform=Transform(...)`` or the individual fields.  Meant for
        geometry (spheres/meshes/sweeps/volumes); ``skew`` needs ftrace's ``shear``
        and does not apply to analytic ``sphere{}`` (which would become an ellipsoid).
        """
        from .transform import Transform
        self.xf = transform if transform is not None else Transform(
            translate=translate, rotate=rotate, scale=scale, skew=skew)
        return self


def emit_element(e: "Element", ctx: EmitCtx) -> str:
    """Emit an element's block, wrapping it in its :class:`~loom.transform.Transform`
    (an ftsl ``group { … }``) when it carries one.  Containers use this instead of
    calling ``e.emit()`` directly so per-element transforms are honoured (and nest)."""
    body = e.emit(ctx)
    xf = getattr(e, "xf", None)
    return xf.wrap(body, ctx) if xf is not None else body


def element_roots(e: "Element") -> List:
    """An element's modulator roots, including its transform's, for cycle checking."""
    out = list(e.roots())
    xf = getattr(e, "xf", None)
    if xf is not None:
        out.extend(xf.roots())
    return out


class Pattern(Element):
    """Marker base for procedural pattern blocks (see :mod:`loom.material`).

    A pattern is emitted before materials so a material may bind it via
    ``key pattern:<name>``.  The concrete :class:`~loom.material.FuncPattern`
    lives in ``material.py`` to avoid an import cycle; this base is what
    :class:`Scene` routes on."""


# ---------------------------------------------------------------------------
# Spectra and colour upsampling
# ---------------------------------------------------------------------------

class NamedSpectrum(Element):
    """A named spectral curve: ``spectrum "name" = <expr>``.

    ``expr`` is a `.ftsl` **spectrum expression** — anything
    :func:`loom.grammar.spectrum.parse_spectrum` accepts (``gaussian center=550
    sigma=30``, ``preset:d65``, ``file:curve.csv``, ``metal:gold``, a bare number, …) —
    and is validated on construction so a typo fails in Python rather than at render
    time.  Two things reference one by name: a material slot's ``spectrum:<name>`` and,
    inside an :class:`Upsample` body, ``spec:<name>(w)``.  The latter is what makes a
    *measured basis* upsampler expressible (see :class:`Upsample`).
    """

    def __init__(self, name: str, expr: str) -> None:
        from .grammar.spectrum import parse_spectrum
        self.name = name
        self.expr = str(expr).strip()
        parse_spectrum(self.expr)      # raises ShapeError on a bad expression
        # NOTE: no `interp=` / `table { … }` block form here.  Those are the two
        # spectrum spellings that are not a single expression, and a scene that needs
        # one writes it as `file:` instead (which is what a table is *for*).

    def roots(self) -> List:
        return []

    def emit(self, ctx: EmitCtx) -> str:
        return f'spectrum "{self.name}" = {self.expr}'


class Upsample(Element):
    """A user-supplied RGB→spectral upsampler: ``upsample "name" { expr "…" }`` (K1).

    ftrace has five *built-in* upsamplers, each reached by a glued colour head
    (``rgb``= Jakob-Hanika, ``rgbillum``, ``rgbsmits``, ``rgbbox``, ``rgbmeng``).  This
    is the open-ended sixth: the scene supplies the function itself, and a material
    slot reaches it by the colon-separated head ``rgb:<name> r g b`` (or
    ``hsv:<name>`` / ``hsl:<name>`` — ftrace converts to linear sRGB *before* calling
    the body, so the body always sees the same ``r, g, b``).

    The body is a pattern-VM expression whose free variables are exactly ``r``, ``g``,
    ``b`` (the colour, linear sRGB) and ``w`` (wavelength, nm), plus ``pi``, the usual
    functions and ``spec:<spectrum>(w)``.  Note this vocabulary is **disjoint** from the
    surface/shading one — there is no hit point here, and ``r`` means RED, not radius;
    ftrace rejects every surface name explicitly rather than silently reinterpreting it.

    The ``spec:`` sampler is the reason this is more than syntax sugar: it lets an
    upsampler be a **measured basis** rather than a closed form::

        scene.add(NamedSpectrum("sr", "gaussian center=620 sigma=40"),
                  NamedSpectrum("sg", "gaussian center=540 sigma=40"),
                  NamedSpectrum("sb", "gaussian center=460 sigma=40"),
                  Upsample("basis", "r*spec:sr(w) + g*spec:sg(w) + b*spec:sb(w)"),
                  Material("m", "diffuse", reflect="rgb:basis 0.3 0.6 0.1"))
    """

    def __init__(self, name: str, expr: str) -> None:
        self.name = name
        self.expr = str(expr).strip()
        if not self.expr:
            raise ValueError(f"upsample {name!r}: empty expression")

    def roots(self) -> List:
        return []

    def emit(self, ctx: EmitCtx) -> str:
        return f'upsample "{self.name}" {{ expr "{self.expr}" }}'


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

class Texture(Element):
    """An image **skin**: ``texture "name" { file "path" … }``.

    Bind it to a surface by pointing a material's colour at it — the whole point of
    a skin is a spatially-varying diffuse albedo, so
    ``Material("hide", "diffuse", reflect="texture:<name>")`` wraps the image around
    the geometry (sampled at each hit's UV).  ``encoding`` (``srgb``/``linear``),
    ``filter`` (``bilinear``/``nearest``) and ``wrap`` (``repeat``/``clamp``/
    ``mirror``) pass straight through to ftrace.  A ``Texture`` holds no modulators
    (the file is fixed), so it is emitted once, before the materials that reference
    it.  Use :func:`skin` to make the texture *and* its material in one call.
    """

    def __init__(self, name: str, file, *, encoding: str = "srgb",
                 filter: str = "bilinear", wrap: str = "repeat") -> None:
        self.name = name
        # normalise to forward slashes so the emitted path is portable
        self.file = str(file).replace("\\", "/")
        if encoding not in ("srgb", "linear"):
            raise ValueError('texture encoding must be "srgb" or "linear"')
        if filter not in ("bilinear", "nearest"):
            raise ValueError('texture filter must be "bilinear" or "nearest"')
        if wrap not in ("repeat", "clamp", "mirror"):
            raise ValueError('texture wrap must be "repeat", "clamp" or "mirror"')
        self.encoding = encoding
        self.filter = filter
        self.wrap = wrap

    def roots(self) -> List:
        return []

    def emit(self, ctx: EmitCtx) -> str:
        return (f'{self.name} = texture {{ file "{self.file}"  '
                f'encoding {self.encoding}  filter {self.filter}  '
                f'wrap {self.wrap} }}')


class GridDecl(Element):
    """A loom :class:`~loom.data.Grid` emitted as ftsl's ``grid "name" { … }`` block.

    The companion declaration a :class:`~loom.spatial.GridSample` term needs: the
    lattice (``shape``/``lo``/``hi``), its out-of-domain policy (``outside``) and the
    samples, **baked at the emit clock** — a Grid's values are ordinary Signals, so a
    modulated grid re-emits new numbers every frame while the lattice stays put
    (which is exactly the Grid contract: positions fixed, values modulable).

    Authors never write one: :meth:`Scene.add` collects it from the ``GridSample``
    leaves in any field it is given, the same way it collects an :class:`Image`
    term's :class:`Texture`.  ``outside`` belongs to the *block*, not to the sample
    call, which is why one Grid sampled under two policies emits two blocks.
    """

    def __init__(self, name: str, grid, *, outside: str = "clamp") -> None:
        self.name = name
        self.grid = grid
        if outside not in ("clamp", "wrap", "extrapolate"):
            raise ValueError('grid outside must be "clamp", "wrap" or "extrapolate"')
        self.outside = outside

    def roots(self) -> List:
        # the samples are the modulators; the lattice itself is fixed by design
        return list(self.grid.values)

    def emit(self, ctx: EmitCtx) -> str:
        g = self.grid
        clock, cache = ctx.clock, ctx.cache
        L = [f'grid "{self.name}" {{']
        L.append("    shape " + " ".join(str(int(s)) for s in g.shape))
        L.append("    lo " + " ".join(fmt(v) for v in g.lo))
        L.append("    hi " + " ".join(fmt(v) for v in g.hi))
        L.append(f"    outside {self.outside}")
        # C order, axis 0 outermost — the same order Grid._strides uses, and the
        # order ftrace's `data` list is read in. One row per fastest axis run so a
        # human can still read the block.
        row = g.shape[-1]
        L.append("    data {")
        for start in range(0, len(g.values), row):
            chunk = g.values[start:start + row]
            L.append("        " + " ".join(fmt(v.at(clock, cache)) for v in chunk))
        L.append("    }")
        L.append("}")
        return "\n".join(L)


class ScatterDecl(Element):
    """A loom :class:`~loom.data.Scatter` emitted as ftsl's ``scatter "name" { … }``
    block — the ragged sibling of :class:`GridDecl`.

    ``data`` interleaves each sample's coordinates with its value (ftrace reads it in
    strides of ``dim + 1``), which is the point of a scatter: unlike a lattice, a
    sample's position and value are not separable.  **Both** are Signals in loom, so
    both are baked at the emit clock and a moving sample set re-emits every frame.
    """

    def __init__(self, name: str, scatter, *, power: float = 2.0,
                 eps: float = 1e-9) -> None:
        self.name = name
        self.scatter = scatter
        self.power = float(power)
        self.eps = float(eps)

    def roots(self) -> List:
        return [*self.scatter.positions, *self.scatter.values]

    def emit(self, ctx: EmitCtx) -> str:
        s = self.scatter
        clock, cache = ctx.clock, ctx.cache
        L = [f'scatter "{self.name}" {{']
        L.append(f"    dim {int(s.dim)}")
        L.append(f"    power {fmt(self.power)}")
        L.append(f"    eps {fmt(self.eps)}")
        L.append("    data {")
        for pos, val in zip(s.positions, s.values):
            p = pos.at(clock, cache)
            row = " ".join(fmt(c) for c in p[:s.dim])
            L.append(f"        {row}   {fmt(val.at(clock, cache))}")
        L.append("    }")
        L.append("}")
        return "\n".join(L)


# colour-valued material slots — a field here lowers to a ``ProcTexture`` over the
# surface u/v; every other slot is a scalar knob lowering to a world-space pattern.
_COLOR_SLOTS = ("reflect", "transmit", "emit", "emission", "color", "tint")


def _spatial_fields(el):
    """Every :class:`~loom.spatial.SpatialExpr` reachable from an element's own
    attributes (one container level deep).

    Deliberately duck-typed rather than a per-class hook: a spatial field can sit on
    a :class:`~loom.material.FuncPattern`'s ``template``, on a :class:`ProcTexture`'s
    ``r``/``g``/``b``, in a :class:`Material`'s ``props`` dict, or on an
    ``Isosurface``'s template — one scan covers all of them and any future holder."""
    from .spatial import SpatialExpr
    out = []
    for v in vars(el).values():
        if isinstance(v, SpatialExpr):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out.extend(c for c in v if isinstance(c, SpatialExpr))
        elif isinstance(v, dict):
            for c in v.values():
                if isinstance(c, SpatialExpr):
                    out.append(c)
                elif isinstance(c, (list, tuple)):
                    out.extend(g for g in c if isinstance(g, SpatialExpr))
    return out


def _field_exprs(v):
    """If ``v`` is a scalar :class:`~loom.spatial.SpatialExpr` field or a tuple of
    them (mixing plain numbers), return the list of components (numbers coerced);
    otherwise ``None`` (a plain scalar/Signal/string property)."""
    from .spatial import SpatialExpr, sexpr
    if isinstance(v, SpatialExpr):
        return [v]
    if isinstance(v, (list, tuple)) and any(isinstance(c, SpatialExpr) for c in v):
        return [sexpr(c) for c in v]
    return None


class Material(Element):
    """A material.  Any property may be a plain value (number / :class:`Signal` /
    string / colour) **or** a loom :class:`~loom.spatial.SpatialExpr` *field* over
    the named surface inputs — this is what makes a material a **parameterized
    bundle** (J3b item 3 / ``ROADMAP_records.md`` §3.3).

    A bundle exposes its *free inputs* (:meth:`free_inputs` — the union of its
    properties' :data:`U`/:data:`V`/:data:`A` leaves) and is **applied** by binding
    them at the use site: ``gold(u=v, a=1)`` / ``gold(v)`` (positional, single free
    input only).  Binding is pure **substitution** on the field expressions, so the
    result is an ordinary material whose fields are concrete formulas in real ftrace
    variables — never literal bundle syntax.  A field property lowers to a renderable
    companion element when the material is added to a :class:`Scene` (:meth:`expand`):
    a colour slot → a :class:`ProcTexture` baked over ``u``/``v``; a scalar slot → a
    live :class:`~loom.material.FuncPattern` evaluated per hit (world ``x``/``y``/``z``,
    the field value, the hit normal and ``u``/``v`` are all in scope).  The albedo
    input :data:`A`, left unbound, resolves to the material's ``albedo_default``."""

    def __init__(self, name: str, mtype: str = "diffuse", *,
                 albedo_default: Number = 1.0, **props) -> None:
        self.name = name
        self.mtype = mtype
        self.props = props
        self.albedo_default = albedo_default

    def roots(self) -> List:
        out: List = []
        for v in self.props.values():
            n = site_node(v)                    # Signal / VecSignal / lowered axis node
            if n is not None:
                out.append(n)
            else:
                for e in (_field_exprs(v) or ()):
                    out.extend(e.time_signals())
        return out

    # ---- bundle: free inputs + application (binding by substitution) ------
    def free_inputs(self, include_coords: bool = False) -> "frozenset[str]":
        """The union of every field property's bindable inputs (``{u, v, a}``;
        ``include_coords=True`` also reports the spatial ``x``/``y``/``z``)."""
        acc = set()
        for v in self.props.values():
            for e in (_field_exprs(v) or ()):
                acc |= e.free_inputs(include_coords=include_coords)
        return frozenset(acc)

    def apply(self, *args, **binds) -> "Material":
        """Bind free inputs across the bundle and return a new material.  Keyword
        form ``gold(u=v, a=1)``; positional ``gold(expr)`` binds the sole free
        input (an error if there is not exactly one unbound).  Unbound inputs fall
        back to their system defaults (``u``/``v`` stay the surface params; ``a``
        resolves to ``albedo_default`` at emit).  Each RHS is any coercible value —
        a number, :class:`Signal`, or :class:`~loom.spatial.SpatialExpr`."""
        from .spatial import sexpr
        binds = dict(binds)
        if args:
            if len(args) != 1:
                raise TypeError("positional material binding takes one expression")
            free = self.free_inputs() - set(binds)
            if len(free) != 1:
                raise TypeError(
                    f"positional binding needs exactly one free input; "
                    f"'{self.name}' has {sorted(free)} — bind by name")
            binds[next(iter(free))] = args[0]
        mapping = {k: sexpr(v) for k, v in binds.items()}
        new_props = {}
        for k, v in self.props.items():
            exprs = _field_exprs(v)
            if exprs is None:
                new_props[k] = v
            elif isinstance(v, (list, tuple)):
                new_props[k] = tuple(e.substitute(mapping) for e in exprs)
            else:
                new_props[k] = exprs[0].substitute(mapping)
        return Material(self.name, self.mtype,
                        albedo_default=self.albedo_default, **new_props)

    def __call__(self, *args, **binds) -> "Material":
        return self.apply(*args, **binds)

    # ---- §3.2 per-property access ----------------------------------------
    def prop(self, name: str, *args, **binds):
        """The value of ONE property of this bundle — ``ROADMAP_records.md`` §3.2's
        per-property access, the twin of ftrace's ``MATERIAL.slot`` /
        ``MATERIAL.slot(args)`` (``Builder::materialPropRef`` in ``src/ftsl.h``).

        ``gold.prop("reflect")`` reads the slot with every named input at its
        default; ``gold.prop("reflect", u=v)`` / ``gold.prop("reflect", a=1)``
        rebinds first.  Rebinding routes through :meth:`apply`, so it is exactly the
        same substitution the bundle uses at a use site — a property reference can
        never diverge from applying the whole material and then reading the slot.

        An unbound :data:`A` resolves against **this** material's
        ``albedo_default``, not the consumer's: the property carries the source's
        notion of albedo with it, which is what makes the reference a *value* rather
        than a fragment needing the consumer's context.

        §3.2 makes the property NAME optional (the leading type/slot keyword is what
        binds a property to its slot; the name only mints an external dot-handle).
        loom keeps properties in a plain ``**props`` dict keyed by slot, so the key
        IS the handle here — the same choice ftrace makes for the same reason."""
        m = self.apply(*args, **binds) if (args or binds) else self
        if name not in m.props:
            raise KeyError(
                f"material '{self.name}' has no property '{name}' "
                f"(has {sorted(m.props)})")
        v = m.props[name]
        exprs = _field_exprs(v)
        if exprs is None:
            return v                                  # a plain value / Signal / colour
        if isinstance(v, (list, tuple)):
            return tuple(m._resolve_albedo(e) for e in exprs)
        return m._resolve_albedo(exprs[0])

    def _resolve_albedo(self, e):
        """Substitute an unbound albedo leaf :data:`A` with the material default."""
        if "a" in e.free_inputs():
            return e.substitute({"a": float(self.albedo_default)})
        return e

    def has_fields(self) -> bool:
        return any(_field_exprs(v) is not None for v in self.props.values())

    def expand(self, prefix: str) -> Tuple[List[Element], "Material"]:
        """Lower every field property to a renderable companion element and return
        ``(companions, resolved_material)`` where the material's field slots now
        reference the companions (``reflect texture:<p>_reflect`` /
        ``roughness pattern:<p>_roughness``).  Colour slots bake over surface
        ``u``/``v`` (a :class:`ProcTexture`) and may therefore use *only* ``u``/``v``
        — anything else raises.  Scalar slots stay live (a
        :class:`~loom.material.FuncPattern`) and are evaluated per hit, so they may
        use world ``x/y/z``, the field value, the hit normal and ``u``/``v`` in any
        combination."""
        from .material import FuncPattern
        comps: List[Element] = []
        new_props = {}
        for k, v in self.props.items():
            exprs = _field_exprs(v)
            if exprs is None:
                new_props[k] = v
                continue
            exprs = [self._resolve_albedo(e) for e in exprs]
            if k in _COLOR_SLOTS:
                for e in exprs:
                    bad = e.free_inputs(include_coords=True) - {"u", "v"}
                    if bad:
                        raise ValueError(
                            f"material '{self.name}' colour slot '{k}' field uses "
                            f"{sorted(bad)}; a colour skin bakes surface u/v only — "
                            f"bind them (e.g. u=..., a=...) or use a scalar slot")
                if len(exprs) == 1:
                    exprs = exprs * 3            # scalar field -> grayscale rgb
                texname = f"{prefix}_{k}"
                comps.append(ProcTexture(texname, exprs[0], exprs[1], exprs[2]))
                new_props[k] = f"texture:{texname}"
            else:
                # No coordinate restriction here, deliberately.  A scalar slot lowers
                # to a *live* pattern, and ftrace evaluates it through
                # `patCtxFromHit` (src/scene.h), which fills x/y/z, the field value,
                # the hit normal AND surface u/v — so a scalar field may draw on any
                # of them, and mix them freely.  (`scenes/uv_native.ftsl` ships
                # exactly that: `weight_map pattern:uvcheck8` over floor(u*8).)  The
                # asymmetry with a colour slot is real but runs the other way: a
                # colour skin is *baked* into an image indexed by u/v, so u/v is all
                # it can ever see.
                e = exprs[0]
                patname = f"{prefix}_{k}"
                comps.append(FuncPattern(patname, e))
                new_props[k] = f"pattern:{patname}"
        resolved = Material(self.name, self.mtype,
                            albedo_default=self.albedo_default, **new_props)
        return comps, resolved

    def emit(self, ctx: EmitCtx) -> str:
        if self.has_fields():
            raise ValueError(
                f"material '{self.name}' still carries field properties "
                f"{sorted(k for k, v in self.props.items() if _field_exprs(v))}; "
                f"add it to a Scene (which expands bundle fields into companion "
                f"pattern/texture elements) before emitting")
        parts = [f"type {self.mtype}"]
        for k, v in self.props.items():
            parts.append(f"{k} {value_token(v, ctx.clock, ctx.cache)}")
        return f'{self.name} = material {{ ' + "  ".join(parts) + " }"


class ProcTexture(Element):
    """A **procedural (function-defined) UV-space skin**: ``texture "name" { rgb
    "r(u,v)" "g(u,v)" "b(u,v)" res N … }``.

    Instead of a bitmap file, the albedo is three ftsl expressions of the surface
    UV coordinates ``u`` and ``v`` (and constants / ``pi``), using the ftsl pattern
    grammar (``sin cos sqrt min max clamp mix step smoothstep noise`` …).  ftrace
    bakes them once to a ``res``×``res`` **linear** RGB grid at load and then treats
    the result exactly like an image texture — so the same UV-wrap, Jakob-Hanika
    spectral upsampling, triplanar, GPU and raster paths apply, and a material binds
    it with the usual ``reflect texture:<name>``.  The expressions are functions of
    ``u, v`` only (the world-space pattern variables carry no value in UV space).
    Like :class:`Texture`, it holds no modulators and is emitted once.  Use
    :func:`func_skin` to make the texture *and* its material together.
    """

    def __init__(self, name: str, r, g, b, *, res: int = 512,
                 filter: str = "bilinear", wrap: str = "clamp") -> None:
        self.name = name
        # Each channel is a literal ftsl string *or* a loom SpatialExpr over the
        # surface params ``u``/``v`` (a material-bundle colour field).  A
        # SpatialExpr is baked per frame (its time coefficients fold into the
        # texture string, so an animated albedo re-bakes each frame); anything
        # else (a string / number) is coerced to a literal string once.
        def _chan(c):
            return c if hasattr(c, "emit") else str(c)
        self.r, self.g, self.b = _chan(r), _chan(g), _chan(b)
        res = int(res)
        if res < 1:
            raise ValueError("texture res must be >= 1")
        if filter not in ("bilinear", "nearest"):
            raise ValueError('texture filter must be "bilinear" or "nearest"')
        if wrap not in ("repeat", "clamp", "mirror"):
            raise ValueError('texture wrap must be "repeat", "clamp" or "mirror"')
        self.res = res
        self.filter = filter
        self.wrap = wrap

    def _channels(self):
        return (self.r, self.g, self.b)

    def roots(self) -> List:
        # a SpatialExpr channel exposes its temporal coefficients for cycle checking
        out: List = []
        for c in self._channels():
            if hasattr(c, "time_signals"):
                out.extend(c.time_signals())
        return out

    @staticmethod
    def _chan_str(c, ctx: EmitCtx) -> str:
        # a SpatialExpr colour field is a function of the surface u/v only (X/Y/Z
        # are rejected at bundle-expand time, so the coord args are never read)
        if hasattr(c, "emit"):
            return c.emit(("u", "v", "0"), ctx)
        return str(c)

    def emit(self, ctx: EmitCtx) -> str:
        r = self._chan_str(self.r, ctx)
        g = self._chan_str(self.g, ctx)
        b = self._chan_str(self.b, ctx)
        return (f'{self.name} = texture {{ rgb "{r}" "{g}" "{b}"  '
                f'res {self.res}  filter {self.filter}  wrap {self.wrap} }}')


def func_skin(name: str, r: str, g: str, b: str, *, mtype: str = "diffuse",
              res: int = 512, filter: str = "bilinear", wrap: str = "clamp",
              **props) -> Tuple["ProcTexture", "Material"]:
    """Wrap a **procedural** UV-space skin over a surface: build the
    :class:`ProcTexture` (three ``r(u,v) g(u,v) b(u,v)`` ftsl expressions) **and** a
    :class:`Material` bound to it::

        scene.add(*func_skin("stripes", "u", "v", "0.5+0.5*sin(2*pi*8*u)"),
                  Sphere((0, 0, 0), 1, "stripes"))

    The material's ``reflect`` is the baked skin (ftrace's ``reflect texture:<name>``);
    extra ``props`` (e.g. ``roughness=…``) pass through, and the texture and material
    share ``name``.
    """
    tex = ProcTexture(name, r, g, b, res=res, filter=filter, wrap=wrap)
    mat = Material(name, mtype, reflect=f"texture:{name}", **props)
    return tex, mat


def skin(name: str, image, *, mtype: str = "diffuse", encoding: str = "srgb",
         filter: str = "bilinear", wrap: str = "repeat",
         **props) -> Tuple["Texture", "Material"]:
    """Wrap an image over a surface: build the :class:`Texture` **and** a
    :class:`Material` bound to it, ready to drop into a scene::

        scene.add(*skin("hide", "textures/cow.png"), Sphere((0,0,0), 1, "hide"))

    The material's ``reflect`` is the image (a spatially-varying diffuse albedo,
    ftrace's ``reflect texture:<name>``); extra ``props`` (e.g. ``roughness=…``) pass
    through to the material, and the texture and material share ``name``.
    """
    tex = Texture(name, image, encoding=encoding, filter=filter, wrap=wrap)
    mat = Material(name, mtype, reflect=f"texture:{name}", **props)
    return tex, mat


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class Sphere(Element):
    def __init__(self, center, radius, material: str) -> None:
        self.center = VecSignal.of(center) if not isinstance(center, VecSignal) \
            else center
        self.radius = radius
        self.material = material

    def roots(self) -> List:
        out: List = [self.center]
        r = site_node(self.radius)
        if r is not None:
            out.append(r)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        c = vec3(self.center, ctx.clock, ctx.cache)
        r = num(self.radius, ctx.clock, ctx.cache)
        return f'sphere {{ center {fmt3(c)}  radius {fmt(r)}  material "{self.material}" }}'


class Beads(Element):
    """A view-independent "string of beads": ``count`` spheres sampled evenly
    along a :class:`LoopCurve` (or a :class:`PointPath`).  This is the simplest
    way to render a 3-D closed curve before the sweep engine (M4) exists."""

    def __init__(self, curve: Union[LoopCurve, PointPath], count: int,
                 radius, material: str) -> None:
        if isinstance(curve, PointPath):
            from .signals.core import Const
            curve = LoopCurve(curve, Const(0.0))
        self.curve = curve
        self.count = int(count)
        self.radius = radius
        self.material = material

    def roots(self) -> List:
        out: List = [self.curve]
        r = site_node(self.radius)
        if r is not None:
            out.append(r)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        r = num(self.radius, ctx.clock, ctx.cache)
        lines: List[str] = []
        for k in range(self.count):
            p = self.curve.sample(k / self.count, ctx.clock, ctx.cache)
            lines.append(
                f'sphere {{ center {fmt3(p)}  radius {fmt(r)}  material "{self.material}" }}')
        return "\n".join(lines)


class Raw(Element):
    """Escape hatch: emit a fixed block of ftsl text verbatim (not animated)."""

    def __init__(self, text: str) -> None:
        self.text = text

    def roots(self) -> List:
        return []

    def emit(self, ctx: EmitCtx) -> str:
        return self.text


class Group(Element):
    """Apply one :class:`~loom.transform.Transform` to several child elements at
    once, emitted as a single ftsl ``group { … <children> }``.

    Position / size / rotation / skew all animate (each field may be a
    :class:`~loom.signals.core.Signal` / :class:`~loom.signals.vector.VecSignal`).
    Children may themselves be transformed — a child's own ``xf`` composes *inside*
    this group's (nested ftsl groups), so::

        Group(Sphere(...).transformed(scale=2), Beads(...),
              translate=(0, 1, 0), rotate=(0, t*90, 0))

    rotates the whole cluster while the sphere keeps its local 2× size.  Give the
    transform via the individual fields or a ready ``transform=Transform(...)``.
    """

    def __init__(self, *children: "Element", translate=None, rotate=None,
                 scale=None, skew=None, transform=None) -> None:
        from .transform import Transform
        self.children = list(children)
        self.xf = transform if transform is not None else Transform(
            translate=translate, rotate=rotate, scale=scale, skew=skew)

    def roots(self) -> List:
        out: List = []
        for c in self.children:
            out.extend(element_roots(c))
        return out

    def emit(self, ctx: EmitCtx) -> str:
        return "\n".join(emit_element(c, ctx) for c in self.children)


class SweptMesh(Element):
    """A profile swept along a spine curve into a triangle mesh (M4 sweep engine).

    The ``spine`` (a :class:`LoopCurve` or :class:`PointPath`) is sampled at
    ``count`` points *at the current clock*, oriented with a rotation-minimizing
    frame, scaled/twisted, and skinned into an OBJ that is written per-frame via
    ``ctx.asset_path``; the emitted ftsl is a ``mesh { file ... }`` reference.

    ``scale`` and ``twist`` may be plain numbers or :class:`Signal`\\ s (animated).
    ``turns`` adds a full ``turns * 2pi`` twist distributed along the spine.
    ``scale_profile`` is an optional ``f(u)->float`` multiplier (``u in [0,1)``)
    that swells/pinches the section along the spine (used by the ``blob`` preset).
    """

    def __init__(self, spine: Union[LoopCurve, PointPath], profile: Sequence[Tuple[float, float]],
                 *, count: int = 64, scale=1.0, twist=0.0, turns=0.0,
                 closed_spine: bool = True, closed_profile: bool = True,
                 material: str = "default", smooth: int = 1, name: str = "swept",
                 scale_profile: Optional[Callable[[float], float]] = None) -> None:
        if isinstance(spine, PointPath):
            spine = LoopCurve(spine, Const(0.0))
        self.spine = spine
        self.profile = [(float(a), float(b)) for (a, b) in profile]
        self.count = int(count)
        self.scale = scale
        self.twist = twist
        self.turns = turns
        self.closed_spine = closed_spine
        self.closed_profile = closed_profile
        self.material = material
        self.smooth = int(smooth)
        self.name = name
        self.scale_profile = scale_profile

    def roots(self) -> List:
        out: List = [self.spine]
        for v in (self.scale, self.twist, self.turns):
            if isinstance(v, (Signal, VecSignal)):
                out.append(v)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        n = self.count
        pts = [self.spine.sample(k / n, ctx.clock, ctx.cache) for k in range(n)]
        base_sc = num(self.scale, ctx.clock, ctx.cache)
        base_tw = num(self.twist, ctx.clock, ctx.cache)
        turns = num(self.turns, ctx.clock, ctx.cache)
        scales: List[float] = []
        twists: List[float] = []
        for k in range(n):
            u = k / n
            mult = self.scale_profile(u) if self.scale_profile is not None else 1.0
            scales.append(base_sc * mult)
            twists.append(base_tw + turns * 2.0 * math.pi * u)
        rings = _sweep.sweep_rings(pts, self.profile, scales, twists, self.closed_spine)
        verts, faces = _sweep.skin_rings(rings, self.closed_spine, self.closed_profile)
        path = ctx.asset_path(self.name, "obj")
        _sweep.write_obj(path, verts, faces)
        return (f'mesh {{ file "{path.as_posix()}"  smooth {self.smooth}  '
                f'material "{self.material}" }}')


class IsoMesh(Element):
    """A scalar field **baked to a triangle mesh** per frame via marching cubes
    (M7), then referenced as ``mesh { file ... }``.

    ftrace root-finds isosurfaces directly, so most fields should be an
    :class:`~loom.iso.Isosurface` (emitted as a ``function { expr }`` string) —
    that is sharper and needs no baking.  Use ``IsoMesh`` when a field must become
    real geometry: a numpy-only field with no ftsl twin, a sampled volume, a mesh
    destined for another tool — or a field you want to **inspect in the native
    viewer**, whose Meshes tab draws (and, on a parameter sweep, re-bakes) exactly
    this element's triangles via :func:`loom.viewer._iso_mesh_geometry`; an
    ``Isosurface`` has no triangles for it to show.

    ``field`` is a :class:`~loom.spatial.SpatialExpr` (baked at the clock) or a
    vectorised ``f(X, Y, Z) -> ndarray``.  ``bounds``/``res``/``iso``/``adaptive``
    /``coarse`` pass straight through to :func:`loom.mcubes.mesh_field`.  The mesh
    is written per-frame via ``ctx.asset_path`` and re-baked every frame (so an
    animated field morphs); a **time-independent** field is baked once and cached.
    """

    def __init__(self, field, *, bounds=1.0, res=48, iso: float = 0.0,
                 adaptive: bool = False, coarse: int = 8,
                 material: str = "default", smooth: int = 1, name: str = "isomesh") -> None:
        self.field = field
        self.bounds = bounds
        self.res = res
        self.iso = float(iso)
        self.adaptive = bool(adaptive)
        self.coarse = int(coarse)
        self.material = material
        self.smooth = int(smooth)
        self.name = name
        self._cache_static: Optional[Tuple[list, list]] = None

    def roots(self) -> List:
        # A SpatialExpr exposes its temporal coefficients for cycle checking.
        if hasattr(self.field, "param_signals"):
            return list(self.field.param_signals())
        return []

    def _static(self) -> bool:
        return hasattr(self.field, "uses_time") and not self.field.uses_time()

    def emit(self, ctx: EmitCtx) -> str:
        from . import mcubes as _mc
        if self._static() and self._cache_static is not None:
            verts, faces = self._cache_static
        else:
            verts, faces = _mc.mesh_field(
                self.field, bounds=self.bounds, res=self.res, iso=self.iso,
                clock=ctx.clock, cache=ctx.cache,
                adaptive=self.adaptive, coarse=self.coarse)
            if self._static():
                self._cache_static = (verts, faces)
        path = ctx.asset_path(self.name, "obj")
        _sweep.write_obj(path, verts, faces)
        return (f'mesh {{ file "{path.as_posix()}"  smooth {self.smooth}  '
                f'material "{self.material}" }}')


def ribbon(spine, *, width: float = 0.3, material: str = "default", count: int = 64,
           twist=0.0, turns=0.0, closed_spine: bool = True, smooth: int = 0,
           name: str = "ribbon") -> SweptMesh:
    """A flat strip (open line profile) swept along the spine."""
    return SweptMesh(spine, _sweep.line_profile(width), count=count, scale=1.0,
                     twist=twist, turns=turns, closed_spine=closed_spine,
                     closed_profile=False, material=material, smooth=smooth, name=name)


def tube(spine, *, radius: float = 0.1, sides: int = 12, material: str = "default",
         count: int = 64, twist=0.0, turns=0.0, closed_spine: bool = True,
         smooth: int = 1, name: str = "tube") -> SweptMesh:
    """A closed circular tube swept along the spine."""
    return SweptMesh(spine, _sweep.circle_profile(sides, 1.0), count=count, scale=radius,
                     twist=twist, turns=turns, closed_spine=closed_spine,
                     closed_profile=True, material=material, smooth=smooth, name=name)


def blob(spine, *, radius: float = 0.15, sides: int = 16, bulge: float = 0.6,
         lobes: int = 2, material: str = "default", count: int = 96,
         twist=0.0, turns=0.0, closed_spine: bool = True, smooth: int = 1,
         name: str = "blob") -> SweptMesh:
    """A tube whose radius swells and pinches around the loop (``lobes`` bulges)."""
    def _prof(u: float) -> float:
        return 1.0 + bulge * math.sin(2.0 * math.pi * lobes * u)
    return SweptMesh(spine, _sweep.circle_profile(sides, 1.0), count=count, scale=radius,
                     twist=twist, turns=turns, closed_spine=closed_spine,
                     closed_profile=True, material=material, smooth=smooth, name=name,
                     scale_profile=_prof)


def fan(spine, *, width: float = 0.4, material: str = "default", count: int = 64,
        twist=0.0, turns=0.0, smooth: int = 0, name: str = "fan") -> SweptMesh:
    """An open ribbon swept along an *open* spine (fans out end to end)."""
    return SweptMesh(spine, _sweep.line_profile(width), count=count, scale=1.0,
                     twist=twist, turns=turns, closed_spine=False,
                     closed_profile=False, material=material, smooth=smooth, name=name)


# ---------------------------------------------------------------------------
# Participating media / volumes
# ---------------------------------------------------------------------------

class Volume(Element):
    """A participating-medium region emitted as ftrace's ``medium { … }`` block.

    ftrace already renders volumes richly — homogeneous fog, bounded
    heterogeneous blobs whose density is a formula, and imported NanoVDB grids.
    loom's strength is the *procedural* case: a ``density`` field is loom's bread
    and butter (it's how :class:`~loom.iso.Isosurface` works), so a ``Volume``
    lets you animate clouds/fog with the same signal machinery as everything
    else — the scattering coefficients and the density formula are all
    :class:`~loom.signals.core.Signal`-valued.

    ``sigma_t`` (extinction), ``albedo`` (single-scatter albedo) and ``g``
    (Henyey–Greenstein anisotropy) are animatable scalars; ``rayleigh`` swaps the
    HG phase for a Rayleigh one.

    ``density`` shapes a *heterogeneous* medium (``None`` = uniform):

    * a :class:`~loom.spatial.SpatialExpr` — the natural loom field, emitted as an
      inline ``density "<expr>"`` over world ``x y z r`` (animatable, seamless);
    * a ``str`` — either ``"pattern:<name>"`` (bind a named pattern) or a raw ftsl
      expression;
    * ``"vdb:<path>"`` — reference an existing NanoVDB grid (loom doesn't *generate*
      sparse voxels, but it can point at one).

    The region is bounded by exactly one of ``box=(min, max)``, ``sphere=(center,
    radius)`` or ``obj="name"`` (fill a named scene object's interior).  A
    heterogeneous medium needs a finite region for the delta-tracking majorant, so
    give a bound *or* an explicit ``density_max``; ``density_max`` overrides the
    engine's grid estimate when set.
    """

    def __init__(self, *, sigma_t=1.0, albedo: Union[Signal, float] = 0.8,
                 g: Union[Signal, float] = 0.0, rayleigh: bool = False,
                 density=None, density_max=None,
                 box: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                 sphere: Optional[Tuple[Sequence[float], float]] = None,
                 obj: Optional[str] = None, name: str = "volume") -> None:
        n_bounds = sum(x is not None for x in (box, sphere, obj))
        if n_bounds > 1:
            raise ValueError("Volume: give at most one of box=, sphere=, obj=")
        self.sigma_t = sigma_t
        self.albedo = albedo
        self.g = g
        self.rayleigh = bool(rayleigh)
        self.density = density
        self.density_max = density_max
        self.box = (tuple(float(c) for c in box[0]),
                    tuple(float(c) for c in box[1])) if box is not None else None
        self.sphere = ((tuple(float(c) for c in sphere[0]), float(sphere[1]))
                       if sphere is not None else None)
        self.obj = obj
        self.name = name

    def roots(self) -> List:
        out: List = []
        for v in (self.sigma_t, self.albedo, self.g, self.density_max):
            if isinstance(v, (Signal, VecSignal)):
                out.append(v)
        # A SpatialExpr density exposes its temporal coefficients for cycle checking.
        if hasattr(self.density, "param_signals"):
            out.extend(self.density.param_signals())
        return out

    def _density_token(self, ctx: EmitCtx) -> Optional[str]:
        d = self.density
        if d is None:
            return None
        if hasattr(d, "emit"):                       # a loom SpatialExpr field
            return 'density "' + d.emit(("x", "y", "z"), ctx) + '"'
        s = str(d)
        if s.startswith("pattern:") or s.startswith("vdb:"):
            return f"density {s}"
        return f'density "{s}"'                       # raw ftsl expression

    def emit(self, ctx: EmitCtx) -> str:
        clock, cache = ctx.clock, ctx.cache
        parts = [f"sigma_t {fmt(num(self.sigma_t, clock, cache))}",
                 f"albedo {fmt(num(self.albedo, clock, cache))}",
                 f"g {fmt(num(self.g, clock, cache))}"]
        if self.rayleigh:
            parts.append("rayleigh true")
        lines = ["medium {", "    " + "  ".join(parts)]
        if self.box is not None:
            mn, mx = self.box
            lines.append(f"    bounds {{ min {fmt3(mn)}  max {fmt3(mx)} }}")
        elif self.sphere is not None:
            c, rad = self.sphere
            lines.append(f"    bounds {{ center {fmt3(c)}  radius {fmt(rad)} }}")
        elif self.obj is not None:
            lines.append(f'    bounds {{ object "{self.obj}" }}')
        dtok = self._density_token(ctx)
        if dtok is not None:
            lines.append("    " + dtok)
        if self.density_max is not None:
            lines.append(f"    density_max {fmt(num(self.density_max, clock, cache))}")
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------

class Light(Element):
    """Generic ``light <kind> { ...props... }``.  Props are animatable or strings
    and must use ftrace's real light schema (``spd`` for emission, plus per-kind
    geometry: ``origin``/``u``/``v`` for an area light, ``center``/``radius`` for a
    sphere, etc. — see ftrace's ``addLight``).  loom does not invent light fields;
    the one convenience is ``color=(r, g, b)``, which is emitted as an ``spd rgb …``
    emission spectrum, since ftrace lights are spectral and have no ``color`` field.
    """

    def __init__(self, kind: str, **props) -> None:
        self.kind = kind
        self.props = props

    def roots(self) -> List:
        return [v for v in self.props.values() if isinstance(v, (Signal, VecSignal))]

    def emit(self, ctx: EmitCtx) -> str:
        # Unified header: anonymous light with the subtype carried as a `kind`
        # property (`light { kind point  ... }`) rather than a bareword after KIND.
        parts = [f"kind {self.kind}"]
        for k, v in self.props.items():
            tok = value_token(v, ctx.clock, ctx.cache)
            if k == "color":
                # ftrace lights carry their emission in a spectral `spd`; there is no
                # `color` field. Author an RGB colour, emit it as `spd rgb r g b` (the
                # Jakob-Hanika upsample turns the triple into an emission spectrum).
                parts.append(f"spd rgb {tok}")
            else:
                parts.append(f"{k} {tok}")
        return "light { " + "  ".join(parts) + " }"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera(Element):
    def __init__(self, eye, look_at, up=(0, 1, 0), fov_y=40.0,
                 mode: str = "R", res: Tuple[int, int] = (480, 480),
                 name: str = "cam") -> None:
        self.eye = VecSignal.of(eye) if not isinstance(eye, VecSignal) else eye
        self.look_at = VecSignal.of(look_at) if not isinstance(look_at, VecSignal) else look_at
        self.up = VecSignal.of(up) if not isinstance(up, VecSignal) else up
        self.fov_y = fov_y
        self.mode = mode
        self.res = (int(res[0]), int(res[1]))
        self.name = name

    def roots(self) -> List:
        out: List = [self.eye, self.look_at, self.up]
        if isinstance(self.fov_y, Signal):
            out.append(self.fov_y)
        return out

    def emit(self, ctx: EmitCtx) -> str:
        e = vec3(self.eye, ctx.clock, ctx.cache)
        la = vec3(self.look_at, ctx.clock, ctx.cache)
        up = vec3(self.up, ctx.clock, ctx.cache)
        fov = num(self.fov_y, ctx.clock, ctx.cache)
        return (f'{self.name} = camera {{\n'
                f'    eye {fmt3(e)}  look_at {fmt3(la)}  up {fmt3(up)}  fov_y {fmt(fov)}\n'
                f'    mode {self.mode}\n'
                f'    film {{ res {self.res[0]} {self.res[1]} }}\n'
                f'}}')


class CameraCurve(Element):
    """A genuine ftrace ``camera_curve`` flypath (milestone M13).

    The eye rides a Catmull-Rom spline through ``points`` with arc-length (or
    ``density``-shaped) speed, animatable lens/orientation tracks, and ftrace's
    two-axis orientation model.  Unlike :class:`Camera` — which loom re-bakes to a
    static ``camera`` block every frame — a ``camera_curve`` is emitted **once** and
    *ftrace itself* expands the N frames.  So pass it in place of the camera
    (``Scene(camera=CameraCurve(...))``) and render the single emitted ``.ftsl`` with
    ftrace to get the whole flyby; loom's per-frame clock does not drive it.

    Orientation mirrors ftrace's grammar 1:1 (nothing here loom can't emit):

    * **forward** (pick one, else the path tangent): ``look_at=(x,y,z)`` fixed target,
      ``look_points=[(x,y,z), …]`` a second aim spline, or ``fwd_at=[(t,x,y,z), …]``
      direction keyframes.
    * **up**: ``up_at=[(t,x,y,z), …]`` vector keyframes, or ``roll``/``roll_at`` an
      angle (degrees) about the reference up.
    * **reference frame**: ``frame`` sets the default for both axes and
      ``fwd_frame`` / ``up_frame`` override per axis — each ``"world"`` (fixed world
      axes, the classic behavior) or ``"travel"`` (the curve's rotation-minimizing
      frame, so the shot banks into turns; closed loops close seamlessly).  A
      ``fwd_at``/``up_at`` vector is read in the travel basis (x=right, y=up,
      z=forward) when its axis is ``"travel"``, else as a world direction.

    Scalar tracks (``roll_at``/``fov_at``/``zoom_at``/``fstop_at``/``focus_at``) are
    ``[(t, value), …]``; vector tracks (``fwd_at``/``up_at``) are ``[(t, x, y, z), …]``,
    with ``t`` the normalized timeline in ``[0, 1]``.
    """

    _FRAMES = ("world", "travel")

    def __init__(self, points, *, up=(0, 1, 0), fov_y=40.0, mode: str = "R",
                 res: Tuple[int, int] = (480, 480), frames: Optional[int] = None,
                 density=None, density_at=None, closed: bool = False,
                 spline: Optional[str] = None, look_at=None, look_points=None,
                 roll=None, roll_at=None, fov_at=None, zoom_at=None, fstop_at=None,
                 focus_at=None, fwd_at=None, up_at=None, frame: Optional[str] = None,
                 fwd_frame: Optional[str] = None, up_frame: Optional[str] = None,
                 min_reach=None, look_smooth=None, exposure_lock: bool = False,
                 fps=None, name: str = "curve") -> None:
        pts = [tuple(float(c) for c in p) for p in points]
        if len(pts) < 2:
            raise ValueError("CameraCurve needs >= 2 control `points`")
        if frames is None and density is None and density_at is None:
            raise ValueError("CameraCurve needs `frames=` or a `density=`/`density_at=`")
        if look_at is not None and look_points is not None:
            raise ValueError("CameraCurve: give at most one of look_at= / look_points=")
        for label, fv in (("frame", frame), ("fwd_frame", fwd_frame), ("up_frame", up_frame)):
            if fv is not None and fv not in self._FRAMES:
                raise ValueError(f'CameraCurve {label} must be "world" or "travel"')
        self.points = pts
        self.up = tuple(float(c) for c in up)
        self.fov_y = float(fov_y)
        self.mode = mode
        self.res = (int(res[0]), int(res[1]))
        self.frames = None if frames is None else int(frames)
        self.density = None if density is None else float(density)
        self.density_at = None if density_at is None else [(float(t), float(r)) for t, r in density_at]
        self.closed = bool(closed)
        self.spline = spline
        self.look_at = None if look_at is None else tuple(float(c) for c in look_at)
        self.look_points = (None if look_points is None
                            else [tuple(float(c) for c in p) for p in look_points])
        self.roll = None if roll is None else float(roll)
        self._scalar_tracks = {
            "roll_at": self._norm_scalar(roll_at), "fov_at": self._norm_scalar(fov_at),
            "zoom_at": self._norm_scalar(zoom_at), "fstop_at": self._norm_scalar(fstop_at),
            "focus_at": self._norm_scalar(focus_at),
        }
        self._vector_tracks = {
            "fwd_at": self._norm_vector(fwd_at), "up_at": self._norm_vector(up_at),
        }
        self.frame = frame
        self.fwd_frame = fwd_frame
        self.up_frame = up_frame
        self.min_reach = None if min_reach is None else float(min_reach)
        self.look_smooth = None if look_smooth is None else float(look_smooth)
        self.exposure_lock = bool(exposure_lock)
        self.fps = None if fps is None else float(fps)
        self.name = name

    @staticmethod
    def _norm_scalar(track):
        if track is None:
            return None
        return [(float(t), float(v)) for t, v in track]

    @staticmethod
    def _norm_vector(track):
        if track is None:
            return None
        out = []
        for kf in track:
            t, x, y, z = kf
            out.append((float(t), float(x), float(y), float(z)))
        return out

    def roots(self) -> List:
        return []   # a camera_curve is a static authored flight (no per-frame signals)

    def emit(self, ctx: EmitCtx) -> str:
        L = [f'{self.name} = camera_curve {{']
        for p in self.points:
            L.append(f"    point {fmt3(p)}")
        if self.look_points:
            L.append(f"    look curve")
            for p in self.look_points:
                L.append(f"    look_point {fmt3(p)}")
        elif self.look_at is not None:
            L.append(f"    look_at {fmt3(self.look_at)}")
        L.append(f"    up {fmt3(self.up)}   fov_y {fmt(self.fov_y)}   mode {self.mode}")
        if self.spline is not None:
            L.append(f"    spline {self.spline}")
        if self.frames is not None:
            L.append(f"    frames {self.frames}")
        if self.density is not None:
            L.append(f"    density {fmt(self.density)}")
        if self.density_at:
            for t, r in self.density_at:
                L.append(f"    density_at {fmt(t)} {fmt(r)}")
        if self.closed:
            L.append(f"    closed")
        # Reference-frame keywords (only when set; absence == world == legacy behavior).
        if self.frame is not None:
            L.append(f"    frame {self.frame}")
        if self.fwd_frame is not None:
            L.append(f"    fwd_frame {self.fwd_frame}")
        if self.up_frame is not None:
            L.append(f"    up_frame {self.up_frame}")
        # Orientation vector tracks.
        for key, track in self._vector_tracks.items():
            if track:
                for t, x, y, z in track:
                    L.append(f"    {key} {fmt(t)} {fmt3((x, y, z))}")
        # Scalar constant + tracks.
        if self.roll is not None:
            L.append(f"    roll {fmt(self.roll)}")
        for key, track in self._scalar_tracks.items():
            if track:
                for t, v in track:
                    L.append(f"    {key} {fmt(t)} {fmt(v)}")
        if self.min_reach is not None:
            L.append(f"    min_reach {fmt(self.min_reach)}")
        if self.look_smooth is not None:
            L.append(f"    look_smooth {fmt(self.look_smooth)}")
        if self.exposure_lock:
            L.append(f"    exposure_lock")
        if self.fps is not None:
            L.append(f"    fps {fmt(self.fps)}")
        L.append(f"    film {{ res {self.res[0]} {self.res[1]} }}")
        L.append("}")
        return "\n".join(L)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class Scene:
    def __init__(self, camera: Camera, *, units: str = "meters",
                 spectral: Tuple[float, float, float] = (360, 830, 1)) -> None:
        self.camera = camera
        self.units = units
        self.spectral = spectral
        # Spectra and upsamplers come first in the emitted text: an `upsample` body may
        # sample a `spectrum` by name, and a material head may name an upsampler. ftrace
        # resolves all three lazily so the order is not load-bearing — it just reads.
        self.spectra: List[Element] = []
        self.upsamplers: List[Element] = []
        # `grid`/`scatter` blocks load in ftrace's Pass 1a, before textures/patterns/
        # materials, so the order is not load-bearing — emitting them first just keeps
        # the text reading top-down (data, then the fields that sample it).
        self.tables: List[Element] = []
        self.textures: List[Element] = []
        self.patterns: List[Element] = []
        self.records: List[Element] = []
        self.materials: List[Material] = []
        self.elements: List[Element] = []
        self.lights: List[Light] = []

    def _add_image_textures(self, e: Element) -> None:
        """Declare the ``texture`` blocks any :class:`~loom.spatial.Image` term
        inside ``e`` needs.

        An ``Image`` leaf emits ftrace's ``tex:<name>(u, v)`` pattern-VM call, which
        only resolves if a texture of that name is declared — so collecting the
        companion declaration has to be automatic, or every image *term* would emit
        a dangling reference and fail to load.  The scan is duck-typed over the
        element's attributes (a spatial field may live on ``template``, on an
        ``r``/``g``/``b`` channel, or inside ``props``) and names are deduped, so
        several fields sampling the same file share one block.

        Each block is tagged ``_auto_image`` so that an *explicit* declaration of the
        same name added later replaces it rather than emitting a second block with a
        duplicate name (see :meth:`add`) — the author's own ``Texture`` always wins."""
        have = {getattr(t, "name", None) for t in self.textures}
        for src in _spatial_fields(e):
            for tex in src.image_textures():
                if tex.name not in have:
                    have.add(tex.name)
                    tex._auto_image = True
                    self.textures.append(tex)

    def _add_table_decls(self, e: Element) -> None:
        """Declare the ``grid`` / ``scatter`` blocks any
        :class:`~loom.spatial.GridSample` or :class:`~loom.spatial.ScatterSample`
        term inside ``e`` needs — the exact twin of :meth:`_add_image_textures`, and
        for the same reason: the emitted ``grid:<name>(…)`` call is a dangling
        reference until the block exists, so collecting it has to be automatic.

        Deduped by name, so several fields sampling one dataset with the same
        sampler settings share a single block."""
        have = {getattr(g, "name", None) for g in self.tables}
        for src in _spatial_fields(e):
            for td in src.table_decls():
                if td.name not in have:
                    have.add(td.name)
                    td._auto_table = True
                    self.tables.append(td)

    def add(self, *elems: Element) -> "Scene":
        from .record import Record as _Record  # lazy: record.py imports scene.Element
        for e in elems:
            self._add_image_textures(e)
            self._add_table_decls(e)
            if isinstance(e, (GridDecl, ScatterDecl)):
                # an explicit declaration supersedes one auto-collected from a term
                self.tables = [g for g in self.tables
                               if not (getattr(g, "_auto_table", False)
                                       and getattr(g, "name", None) == e.name)]
            if isinstance(e, (Texture, ProcTexture)):
                # An explicit declaration supersedes one auto-collected from an
                # Image term, whichever order they were added in.
                self.textures = [t for t in self.textures
                                 if not (getattr(t, "_auto_image", False)
                                         and getattr(t, "name", None) == e.name)]
            # Textures/patterns/records are emitted before the materials that bind
            # them (ftrace resolves them in an earlier pass, but keep the text tidy).
            if isinstance(e, NamedSpectrum):
                self.spectra.append(e)
            elif isinstance(e, Upsample):
                self.upsamplers.append(e)
            elif isinstance(e, (GridDecl, ScatterDecl)):
                self.tables.append(e)
            elif isinstance(e, (Texture, ProcTexture)):
                self.textures.append(e)
            elif isinstance(e, Pattern):
                self.patterns.append(e)
            elif isinstance(e, _Record):
                self.records.append(e)
            elif isinstance(e, Material):
                # A bundle material with field properties expands into companion
                # pattern/texture elements (emitted before it) plus a resolved
                # material that references them — see Material.expand.
                if getattr(e, "props", None) and e.has_fields():
                    comps, resolved = e.expand(e.name)
                    self.add(*comps)               # routes ProcTexture / Pattern
                    self.materials.append(resolved)
                else:
                    self.materials.append(e)
            elif isinstance(e, Light):
                self.lights.append(e)
            else:
                self.elements.append(e)
        return self

    def _all_elements(self) -> List[Element]:
        return [*self.spectra, *self.upsamplers, *self.tables, *self.textures, *self.patterns,
                *self.records, *self.materials, *self.elements, *self.lights,
                self.camera]

    def check_cycles(self) -> None:
        """Run the loop detector over every modulator in the scene."""
        for el in self._all_elements():
            for r in element_roots(el):
                detect_signal_cycle(r)

    def emit(self, clock: Clock, cache: Optional[Cache] = None, *,
             assets_dir: Optional["Path"] = None, tag: str = "") -> str:
        ctx = EmitCtx(clock=clock, cache=cache, assets_dir=assets_dir, tag=tag)
        lo, hi, step = self.spectral
        header = f"scene {{ units {self.units}  spectral {fmt(lo)} {fmt(hi)} {fmt(step)} }}"
        blocks = [header, ""]
        for sp in self.spectra:
            blocks.append(sp.emit(ctx))
        if self.spectra:
            blocks.append("")
        for up in self.upsamplers:
            blocks.append(up.emit(ctx))
        if self.upsamplers:
            blocks.append("")
        for td in self.tables:
            blocks.append(td.emit(ctx))
        if self.tables:
            blocks.append("")
        for tx in self.textures:
            blocks.append(tx.emit(ctx))
        if self.textures:
            blocks.append("")
        for p in self.patterns:
            blocks.append(p.emit(ctx))
        if self.patterns:
            blocks.append("")
        for rec in self.records:
            blocks.append(rec.emit(ctx))
        if self.records:
            blocks.append("")
        for m in self.materials:
            blocks.append(m.emit(ctx))
        blocks.append("")
        for e in self.elements:
            blocks.append(emit_element(e, ctx))
        blocks.append("")
        for lt in self.lights:
            blocks.append(emit_element(lt, ctx))
        blocks.append("")
        blocks.append(self.camera.emit(ctx))
        return "\n".join(blocks) + "\n"
