"""J3b item 3 tests: materials as **parameterized bundles**.

A material property may be a loom :class:`~loom.spatial.SpatialExpr` field over the
named surface inputs ``u``/``v``/``a``; the material then exposes those as *free
inputs* and is *applied* by binding them at the use site (``gold(u=v, a=1)`` /
positional ``gold(expr)``).  Binding is pure substitution, so a bound material is an
ordinary material whose fields are concrete formulas — and adding it to a
:class:`Scene` lowers each field to a renderable companion (colour slot → a
:class:`ProcTexture` baked over ``u``/``v``; scalar slot → a live
:class:`~loom.material.FuncPattern` seeing world ``x``/``y``/``z`` *and* ``u``/``v``),
so every emitted
``.ftsl`` is renderable with zero ftrace changes.  Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Sine,
    Scene, Material, Camera, Sphere, Light, ProcTexture, FuncPattern,
    U, V, A, X, Y, Z, sin,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402


def _emit(el, clock=None):
    return el.emit(EmitCtx(clock=clock or Clock(t=0.0), cache=Cache()))


def _cam():
    return Camera((0, 0, 5), (0, 0, 0))


# ---------------------------------------------------------------------------
# free inputs
# ---------------------------------------------------------------------------

def test_free_inputs_is_union_over_field_properties():
    m = Material("m", "diffuse",
                 reflect=(A * U, A * V, A), roughness=0.5 + 0.5 * sin(6.0 * X))
    # bindable inputs = u, v, a  (x is a system coordinate, not bindable)
    assert m.free_inputs() == frozenset({"u", "v", "a"})
    assert m.free_inputs(include_coords=True) == frozenset({"u", "v", "a", "x"})


def test_plain_material_has_no_free_inputs():
    m = Material("plain", "metal", roughness=0.2, reflect="spectrum:gold")
    assert m.free_inputs() == frozenset()
    assert not m.has_fields()


# ---------------------------------------------------------------------------
# application (binding by substitution)
# ---------------------------------------------------------------------------

def test_keyword_binding_substitutes_across_the_bundle():
    m = Material("gold", "diffuse", reflect=(A * U, A * V, A * 0.5))
    bound = m(u=V, a=1.0)                       # gold(u=v, a=1)
    # u -> v and a -> 1 everywhere; only v remains free
    assert bound.free_inputs() == frozenset({"v"})
    # the original bundle is untouched (functional)
    assert m.free_inputs() == frozenset({"u", "v", "a"})


def test_positional_binding_needs_a_single_free_input():
    tint = Material("tint", "diffuse", reflect=(A, A, A))   # sole free input: a
    p = tint(X * 0.5)                                        # tint(a = x*.5)
    assert p.free_inputs() == frozenset()
    assert p.free_inputs(include_coords=True) == frozenset({"x"})
    # ambiguous when more than one input is free
    two = Material("two", "diffuse", reflect=(A, U, V))
    try:
        two(X)
    except TypeError:
        pass
    else:
        raise AssertionError("positional binding must reject multiple free inputs")


def test_partial_application_leaves_the_rest_free():
    m = Material("m", "diffuse", reflect=(A * U, A * V, A))
    half = m(a=0.5)                             # bind a, leave u/v
    assert half.free_inputs() == frozenset({"u", "v"})


def test_binding_accepts_arbitrary_expressions():
    # a=<expr> where the RHS is any coercible value (u/v field, number, Signal)
    m = Material("m", "diffuse", reflect=(A, A, A))
    out = m(a=0.5 + 0.5 * sin(3.0 * U))          # gold(a = 0.5+0.5*sin(3u))
    assert out.free_inputs() == frozenset({"u"})
    comps, _ = out.expand("m")                   # a u/v skin -> still renderable
    assert "sin(" in _emit(comps[0]) and '"' in _emit(comps[0])


# ---------------------------------------------------------------------------
# expansion / emit (renderable lowering)
# ---------------------------------------------------------------------------

def test_colour_field_lowers_to_a_proctexture_over_uv():
    m = Material("gold", "diffuse", reflect=(A * U, A * V, A * 0.5), albedo_default=0.8)
    comps, resolved = m.expand("gold")
    assert len(comps) == 1 and isinstance(comps[0], ProcTexture)
    assert resolved.props["reflect"] == "texture:gold_reflect"
    txt = _emit(comps[0])
    # unbound albedo a resolved to the material default (0.8); u/v are the vars
    assert '"((0.8)*(u))"' in txt and '"((0.8)*(v))"' in txt


def test_scalar_field_lowers_to_a_worldspace_pattern():
    m = Material("m", "metal", roughness=0.5 + 0.5 * sin(6.0 * Z))
    comps, resolved = m.expand("m")
    assert len(comps) == 1 and isinstance(comps[0], FuncPattern)
    assert resolved.props["roughness"] == "pattern:m_roughness"
    assert "sin(" in _emit(comps[0])


def test_colour_slot_rejects_world_coordinates():
    m = Material("bad", "diffuse", reflect=(X, Y, Z))       # x/y/z in a colour skin
    try:
        m.expand("bad")
    except ValueError as e:
        assert "u/v only" in str(e)
    else:
        raise AssertionError("a colour field over x/y/z must be rejected")


def test_scalar_slot_accepts_surface_uv():
    """A scalar slot is a *live* pattern, so u/v are in scope alongside x/y/z.

    ftrace evaluates a `pattern:` bound to a scalar slot through `patCtxFromHit`
    (src/scene.h), which fills world position, the field value, the hit normal and
    the surface u/v — `scenes/uv_native.ftsl` ships `weight_map pattern:uvcheck8`
    over `floor(u*8)`.  Only a *colour* slot is u/v-restricted, and for the opposite
    reason: it bakes into an image that is indexed by u/v and nothing else."""
    m = Material("ok", "metal", roughness=0.5 * U + 0.5 * sin(6.0 * X))
    comps, resolved = m.expand("ok")
    assert len(comps) == 1 and isinstance(comps[0], FuncPattern)
    assert resolved.props["roughness"] == "pattern:ok_roughness"
    body = _emit(comps[0])
    assert "u" in body and "sin(" in body


def test_scene_expands_bundle_into_companions_before_the_material():
    m = Material("gold", "diffuse", reflect=(A * U, A * V, A * 0.5),
                 roughness=0.5 + 0.5 * sin(6.0 * X), albedo_default=0.8)
    sc = Scene(_cam())
    sc.add(m, Sphere((0, 0, 0), 1, "gold"),
           Light("point", position=(3, 3, 3), name="key"))
    txt = sc.emit(Clock(t=0.0), Cache())
    # companions emitted, referenced by the material, no bundle syntax leaked
    assert "gold_reflect = texture" in txt
    assert "gold_roughness = pattern" in txt
    assert "reflect texture:gold_reflect" in txt
    assert "roughness pattern:gold_roughness" in txt
    assert "(u=" not in txt and "(a=" not in txt
    # companions come before the material that binds them
    assert txt.index("gold_reflect = texture") < txt.index("gold = material")


def test_animated_colour_field_rebakes_per_frame_and_surfaces_roots():
    s = Sine(cycles=1.0, amp=0.3, bias=0.6)
    m = Material("puls", "diffuse", reflect=(s * U, s * V, A * 0.5), albedo_default=1.0)
    comps, resolved = m.expand("puls")
    tex = comps[0]
    # the animated coefficient reaches the texture's cycle-check roots
    assert any(r is s for r in tex.roots())
    a = _emit(tex, Clock.at_frame(0, 12))
    b = _emit(tex, Clock.at_frame(3, 12))
    assert a != b                                 # re-bakes as the signal moves
    assert _emit(tex, Clock.at_frame(0, 12)) == _emit(tex, Clock.at_frame(12, 12))


def test_unexpanded_bundle_emit_raises_helpfully():
    m = Material("m", "diffuse", reflect=(A, A, A))
    try:
        _emit(m)
    except ValueError as e:
        assert "add it to a Scene" in str(e)
    else:
        raise AssertionError("emitting an unexpanded field bundle must raise")


def test_bundle_roots_expose_field_time_signals():
    s = Sine(cycles=1.0)
    m = Material("m", "diffuse", reflect=(s * U, A, A))
    assert any(r is s for r in m.roots())


# ---------------------------------------------------------------------------
# per-property access (ROADMAP_records.md §3.2) — the twin of ftrace's
# `MATERIAL.slot` / `MATERIAL.slot(args)` (Builder::materialPropRef, src/ftsl.h).
# ---------------------------------------------------------------------------

def _txt(e):
    """The canonical text of a spatial expression — `emit` is loom's serializer, and
    two expressions are the same iff they emit the same .ftsl."""
    return e.emit(("x", "y", "z"), None)


def test_prop_reads_one_slot_of_the_bundle():
    m = Material("m", "diffuse", reflect=0.05 + 0.9 * U, roughness=0.35)
    assert m.prop("roughness") == 0.35                  # a plain value passes through
    # a field property comes back as the expression itself
    assert _txt(m.prop("reflect")) == _txt(0.05 + 0.9 * U)


def test_prop_rebinding_equals_applying_then_reading():
    """The whole point: a property reference can never diverge from applying the
    material and then reading the slot, because it IS that."""
    m = Material("m", "diffuse", reflect=0.05 + 0.9 * U, roughness=A)
    assert _txt(m.prop("reflect", u=V)) == _txt(m.apply(u=V).props["reflect"])
    # positional binding works through prop() too (one free input after `a` is named)
    assert _txt(m.prop("reflect", V, a=1)) == _txt(m.apply(V, a=1).props["reflect"])


def test_prop_resolves_an_unbound_albedo_against_the_SOURCE_default():
    """`a` follows the material it was read from, not the consumer — which is what
    makes a property reference a value rather than a fragment needing context."""
    m = Material("m", "diffuse", reflect=A * U, albedo_default=0.4)
    assert _txt(m.prop("reflect")) == _txt(0.4 * U)
    # Resolving `a` is exactly what prop() adds on top of apply(): apply() leaves the
    # albedo leaf symbolic, and a symbolic `a` REFUSES to emit rather than silently
    # becoming 0 -- so prop() is what makes a property reference a usable value.
    try:
        _txt(m.apply().props["reflect"])
    except Exception as e:                                    # noqa: BLE001
        assert "albedo" in str(e) or "'a'" in str(e)
    else:
        raise AssertionError("an unresolved albedo leaf should refuse to emit")
    assert _txt(m.prop("reflect", a=1)) == _txt(1 * U)


def test_prop_names_the_material_and_lists_its_slots_when_unknown():
    m = Material("m", "diffuse", reflect=0.5)
    try:
        m.prop("colour")
    except KeyError as e:
        assert "colour" in str(e) and "reflect" in str(e)
    else:
        raise AssertionError("expected a KeyError for an unknown property")


def test_prop_of_a_tuple_slot_resolves_every_component():
    m = Material("m", "diffuse", reflect=(A * U, A * V, A), albedo_default=0.25)
    got = m.prop("reflect")
    assert isinstance(got, tuple) and len(got) == 3
    assert all("0.25" in _txt(g) for g in got)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
