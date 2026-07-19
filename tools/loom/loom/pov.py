"""
Loom POV-Ray function library (M9) — the ~78 POV isosurface builtins that ftrace
imports (``f_torus``, ``f_heart``, ``f_sphere``, …), wrapped as **bakeable,
param-animatable** field / pattern templates.

ftrace evaluates these natively in its isosurface / pattern expression language
(see ``src/pov_functions.h``: a ``povFnLookup`` table maps each name to a total
**arity** = 3 coordinates + N shape parameters; e.g. ``f_torus`` has arity 5 →
2 params, ``f_sphere`` arity 4 → 1 param, the spherical helpers ``f_r`` / ``f_th``
/ ``f_ph`` / ``f_noise3d`` arity 3 → 0 params).  So loom does not reimplement the
algebra — it emits the call ``f_name(cx, cy, cz, p0, …)`` and lets the renderer
evaluate it.

A :class:`PovFn` is a template you drop into an :class:`loom.Isosurface` (as its
``field``) or a :class:`loom.FuncPattern` (as its ``template``):

    Isosurface(pov("f_torus", 0.8, 0.25), container="sphere", radius=1.5)
    FuncPattern("bump", pov("f_sphere", Sine(cycles=1, bias=1.2)))

The 3 coordinates receive loom's usual affine bake (freq / N-D rotation / drift),
and the **shape params are Signals baked to numbers per frame**, so the surface
animates.  Honesty (DESIGN.md §11.7): a *phase drift* only loops seamlessly for
*periodic* fields (the TPMS in ``iso.py``); most POV functions are non-periodic,
so seamless motion for them means a coordinate transform that *returns to itself*
over the loop — e.g. a Givens rotation through a whole ``2*pi`` — not a linear
drift ramp.
"""

from __future__ import annotations

from typing import List, Union

from .signals.core import Signal, as_signal, Number
from .ftsl_emit import fmt

# name -> total arity (3 coords + params), mirrored from src/pov_functions.h.
# The test suite re-derives this from the header and asserts it matches, so any
# drift between loom and the renderer fails loudly.
POV_FUNCS = {
    "f_algbr_cyl1": 8, "f_algbr_cyl2": 8, "f_algbr_cyl3": 8, "f_algbr_cyl4": 8,
    "f_bicorn": 5, "f_bifolia": 5, "f_blob": 8, "f_blob2": 7, "f_boy_surface": 5,
    "f_comma": 4, "f_cross_ellipsoids": 7, "f_crossed_trough": 4,
    "f_cubic_saddle": 4, "f_cushion": 4, "f_devils_curve": 4,
    "f_devils_curve_2d": 9, "f_dupin_cyclid": 9, "f_ellipsoid": 6, "f_enneper": 4,
    "f_flange_cover": 7, "f_folium_surface": 6, "f_folium_surface_2d": 9,
    "f_glob": 4, "f_heart": 4, "f_helical_torus": 13, "f_helix1": 10,
    "f_helix2": 10, "f_hex_x": 4, "f_hex_y": 4, "f_hetero_mf": 9,
    "f_hunt_surface": 4, "f_hyperbolic_torus": 6, "f_isect_ellipsoids": 7,
    "f_kampyle_of_eudoxus": 6, "f_kampyle_of_eudoxus_2d": 9, "f_klein_bottle": 4,
    "f_kummer_surface_v1": 4, "f_kummer_surface_v2": 7,
    "f_lemniscate_of_gerono": 4, "f_lemniscate_of_gerono_2d": 9, "f_mesh1": 8,
    "f_mitre": 4, "f_nodal_cubic": 4, "f_odd": 4, "f_ovals_of_cassini": 7,
    "f_paraboloid": 4, "f_parabolic_torus": 6, "f_ph": 3, "f_pillow": 4,
    "f_piriform": 4, "f_piriform_2d": 10, "f_poly4": 8, "f_polytubes": 9,
    "f_quantum": 4, "f_quartic_paraboloid": 4, "f_quartic_saddle": 4,
    "f_quartic_cylinder": 6, "f_r": 3, "f_ridge": 9, "f_ridged_mf": 9,
    "f_rounded_box": 7, "f_sphere": 4, "f_spikes": 8, "f_spikes_2d": 7,
    "f_spiral": 9, "f_steiners_roman": 4, "f_strophoid": 7, "f_strophoid_2d": 10,
    "f_superellipsoid": 5, "f_th": 3, "f_torus": 5, "f_torus2": 6,
    "f_torus_gumdrop": 4, "f_umbrella": 4, "f_witch_of_agnesi": 5,
    "f_witch_of_agnesi_2d": 9, "f_noise3d": 3, "f_noise_generator": 4,
}

# The genuinely N-D-generalizable subset (defined by symmetric sums/polynomials);
# for the rest, N-D slicing is only an affine remap of (x,y,z) (DESIGN.md §11.7).
POV_ND_GENERALIZABLE = frozenset({
    "f_sphere", "f_ellipsoid", "f_superellipsoid", "f_paraboloid",
    "f_quartic_paraboloid", "f_ovals_of_cassini", "f_isect_ellipsoids",
    "f_cross_ellipsoids", "f_poly4",
})

# ---------------------------------------------------------------------------
# Per-surface shape-parameter metadata (OSCILLATE_GRAMMAR.md §7, P3.1)
# ---------------------------------------------------------------------------
# `POV_FUNCS`/`pov_functions.h` store only the param *count* (arity - 3), never
# param names/meanings/defaults/ranges.  Those live in POV-Ray's docs, so the
# metadata below is **hand-authored** — a `{func: [ParamMeta, …]}` table where
# each entry is `(name, description, default, (lo, hi))`, `name` a valid axis
# identifier (so `--oscillate <name>` can address it once the surface library is
# wired in P3.3).  `POV_PARAMS` is completed for *every* `POV_FUNCS` entry: the
# well-documented shapes (and the N-D subset) get real metadata; the rest fall
# back to honest generic `p0..` placeholders (`_generic_params`).  A test asserts
# every function has exactly `arity - 3` params with valid, unique names — the
# same drift-guard discipline the arity table uses.

# (name, description, default, (lo, hi))
ParamMeta = tuple

_AUTHORED_PARAMS = {
    # zero-parameter spherical-coordinate / noise helpers
    "f_r": [], "f_th": [], "f_ph": [], "f_noise3d": [],
    # --- quadrics / simple solids (the N-D-generalizable core) ---------------
    "f_sphere": [("radius", "sphere radius", 1.0, (0.1, 4.0))],
    "f_ellipsoid": [("rx", "x semi-axis", 1.0, (0.1, 4.0)),
                    ("ry", "y semi-axis", 1.0, (0.1, 4.0)),
                    ("rz", "z semi-axis", 1.0, (0.1, 4.0))],
    "f_superellipsoid": [("e_we", "east-west roundness exponent", 1.0, (0.1, 4.0)),
                         ("e_ns", "north-south roundness exponent", 1.0, (0.1, 4.0))],
    "f_paraboloid": [("scale", "field scale", 1.0, (0.1, 4.0))],
    "f_quartic_paraboloid": [("scale", "field scale", 1.0, (0.1, 4.0))],
    "f_rounded_box": [("round", "edge-rounding radius", 0.1, (0.0, 1.0)),
                      ("hx", "x half-width", 1.0, (0.1, 4.0)),
                      ("hy", "y half-width", 1.0, (0.1, 4.0)),
                      ("hz", "z half-width", 1.0, (0.1, 4.0))],
    # --- tori ---------------------------------------------------------------
    "f_torus": [("major", "major (ring) radius", 1.0, (0.1, 4.0)),
                ("minor", "minor (tube) radius", 0.25, (0.02, 2.0))],
    # --- named curves / surfaces with a single field-scale parameter --------
    "f_heart": [("scale", "field scale", 1.0, (0.2, 3.0))],
    # --- noise --------------------------------------------------------------
    "f_noise_generator": [("gen", "noise generator select (0..3)", 2.0, (0.0, 3.0))],
}


def _generic_params(n: int) -> List[ParamMeta]:
    """Honest placeholder metadata for a function whose per-parameter meaning is
    not (yet) hand-documented: ``n`` neutral ``p0..p{n-1}`` axes."""
    return [(f"p{i}", f"shape parameter {i} (POV-Ray internal; see POV docs)",
             1.0, (-4.0, 4.0)) for i in range(n)]


# Completed for every POV_FUNCS entry: authored metadata where known, generic
# fallback otherwise.  Built so the count always matches arity - 3 by construction.
POV_PARAMS = {
    name: list(_AUTHORED_PARAMS.get(name, _generic_params(arity - 3)))
    for name, arity in POV_FUNCS.items()
}


def pov_params(name: str) -> List[ParamMeta]:
    """The shape-parameter metadata for POV builtin ``name`` — a list of
    ``(axis_name, description, default, (lo, hi))``, one per ``arity - 3`` param
    (empty for the 0-param helpers).  Raises for an unknown name."""
    if name not in POV_PARAMS:
        raise ValueError(f"unknown POV function {name!r} "
                         f"(see loom.POV_FUNCS for the {len(POV_FUNCS)} available)")
    return list(POV_PARAMS[name])


Param = Union[Signal, Number]


class PovFn:
    """A POV isosurface builtin ``f_name(x, y, z, ...params)`` as a bakeable template.

    ``params`` count must equal ``arity - 3`` (the 3 coordinates are supplied by the
    host :class:`Isosurface` / :class:`FuncPattern`).  Each param may be a Signal, so
    the shape animates; it is baked to a number each frame.

    Duck-typed protocol used by the hosts:
    - :meth:`build` — context-aware emission (bakes params at the clock);
    - :meth:`param_signals` — the param Signals, so the host folds them into the DAG
      (cycle check + per-frame cache).
    """

    def __init__(self, name: str, *params: Param) -> None:
        if name not in POV_FUNCS:
            raise ValueError(f"unknown POV function {name!r} "
                             f"(see loom.POV_FUNCS for the {len(POV_FUNCS)} available)")
        self.name = name
        self.arity = POV_FUNCS[name]
        self.nparams = self.arity - 3
        if len(params) != self.nparams:
            raise ValueError(
                f"{name} takes {self.nparams} param(s) (arity {self.arity} - 3 "
                f"coords), got {len(params)}")
        self._params: List[Signal] = [as_signal(p) for p in params]

    def param_signals(self) -> List[Signal]:
        return list(self._params)

    def build(self, cx: str, cy: str, cz: str, ctx) -> str:
        ps = [fmt(p.at(ctx.clock, ctx.cache)) for p in self._params]
        return f"{self.name}(" + ",".join([cx, cy, cz] + ps) + ")"


def pov(name: str, *params: Param) -> PovFn:
    """Build a :class:`PovFn` template for POV builtin ``name`` with ``params``."""
    return PovFn(name, *params)
