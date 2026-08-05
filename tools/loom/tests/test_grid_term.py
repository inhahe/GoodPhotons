"""loom's :class:`~loom.data.Grid` / :class:`~loom.data.Scatter` **sampled as a term
inside a formula** — the renderable, spatial tier of a dataset.

Until this existed a Grid could only be sampled on the *temporal* path
(:class:`~loom.interp.GridField`, a Signal baked to one number at a clock), so
measured data could drive a knob but could never vary across a surface in a render.
ftrace has had the machinery all along — ``PatOp::Grid`` / ``PatOp::Scatter`` in
``src/pattern.h``, fed by the ``grid { … }`` / ``scatter { … }`` blocks ftsl loads in
Pass 1a — so this is loom catching up to the renderer, exactly the shape of the
``Image``-as-a-term work in ``test_image_term.py``:

- **emit** — ``grid(X, Y)`` writes ftrace's ``grid:<name>(c0, …)`` call, its
  coordinates are ordinary sub-expressions (warpable, and reachable by
  ``substitute`` so a material bundle's ``u=``/``v=`` binding flows in), and the
  companion block is collected automatically by :meth:`Scene.add` — an author who
  never writes a ``GridDecl`` still gets renderable ``.ftsl``.
- **eval_np** — a vectorised port of ``patGridSample`` / ``patScatterSample``,
  checked here against *loom's own* interpolators (``Grid.sample`` /
  ``Scatter.sample``), because those are the documented twins of the ftrace
  functions.  If the two ever drift, the render and the preview disagree.
- **refusals** — a vector-valued dataset, ``interp="cubic"``, ``on_outside="raise"``
  and more than four axes have no ftsl spelling, so they raise here rather than emit
  something that quietly means something else.

Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from loom import (  # noqa: E402
    Cache, Camera, Clock, Grid, GridDecl, GridSample, Light, Material, Ramp,
    Scatter, ScatterDecl, ScatterSample, Scene, Sphere, Transform, U, V, X, Y, Z,
    sin, vec,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402
from loom.interp import GridField, ScatterField, _local_query  # noqa: E402
from loom.signals.vector import VecSignal  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _clock():
    return Clock(t=0.0, frame=0, frames=1, fps=1.0)


def _emit(expr, coords=("x", "y", "z"), clock=None):
    return expr.emit(coords, EmitCtx(clock=clock or _clock(), cache=Cache()))


def _decl(el, clock=None):
    return el.emit(EmitCtx(clock=clock or _clock(), cache=Cache()))


def _cam():
    return Camera(eye=(0, 0, 5), look_at=(0, 0, 0), res=(64, 64))


def _key():
    """A light in ftrace's real schema (there is no `point` subtype)."""
    return Light("sphere", center=(3, 4, 5), radius=0.4, power=4000)


def _g2x2():
    """A 2x2 grid over the unit square with four distinct values."""
    return Grid([[0.0, 0.5], [1.0, 0.25]], lo=0.0, hi=(1.0, 1.0))


def _g3x3():
    return Grid([[0.05, 0.20, 0.85],
                 [0.20, 0.95, 0.30],
                 [0.85, 0.30, 0.95]], lo=0.0, hi=(1.0, 1.0))


def _sc():
    return Scatter([((0.0, 0.0), 1.0),
                    ((1.0, 0.0), 2.0),
                    ((0.5, 1.0), -0.5)])


def _np_at(expr, xs, ys, clock=None):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    return expr.eval_np((xs, ys, np.zeros_like(xs)), clock or _clock(), Cache())


def _raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    raise AssertionError("expected a refusal, got none")


# ---------------------------------------------------------------------------
# emit — the table call
# ---------------------------------------------------------------------------

def test_spatial_query_builds_a_grid_sample_and_emits_the_table_call():
    g = _g2x2()
    e = g(X, Y)
    assert isinstance(e, GridSample)
    assert _emit(e) == f"grid:grid_{g.id}_clamp((x),(y))"


def test_spatial_query_builds_a_scatter_sample_and_emits_the_table_call():
    s = _sc()
    e = s(X, Y)
    assert isinstance(e, ScatterSample)
    txt = _emit(e)
    assert txt.startswith(f"scatter:scatter_{s.id}_")
    assert txt.endswith("((x),(y))")


def test_temporal_query_still_builds_the_signal_field():
    """The spatial tier is additive: a query of numbers/Signals is unchanged."""
    assert isinstance(_g2x2()(0.25, 0.75), GridField)
    assert isinstance(_sc()(0.25, 0.75), ScatterField)


def test_emitted_calls_are_real_ftrace_pattern_ops():
    """`grid:` / `scatter:` must exist in the shipped pattern VM, or the .ftsl is
    un-renderable."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "src", "pattern.h")
    if not os.path.exists(src):           # tests may be vendored away from the tree
        return
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        body = f.read()
    assert "PatOp::Grid" in body and "PatOp::Scatter" in body
    assert 'compare(0, 5, "grid:")' in body
    assert 'compare(0, 8, "scatter:")' in body


def test_grid_is_an_operand_like_any_other_subexpression():
    g = _g2x2()
    e = 0.05 + 0.9 * g(X, Y) * (0.5 + 0.5 * sin(30.0 * Z))
    txt = _emit(e)
    assert f"grid:grid_{g.id}_clamp((x),(y))" in txt and "sin(" in txt


def test_coordinates_are_expressions_so_the_lattice_can_be_warped():
    g = _g2x2()
    txt = _emit(g(U * 2.0 + 0.25, 1.0 - V))
    assert txt.startswith(f"grid:grid_{g.id}_clamp(")
    assert "(u)" in txt and "(v)" in txt and "(2)" in txt


def test_substitute_reaches_into_the_coordinate_arguments():
    g = _g2x2()
    e = g(U, V)
    assert e.free_inputs() == frozenset({"u", "v"})
    out = e.substitute({"u": X, "v": Y})
    assert isinstance(out, GridSample)
    assert out.free_inputs() == frozenset()
    assert _emit(out) == f"grid:grid_{g.id}_clamp((x),(y))"


def test_coordinate_arity_must_match_the_dataset_rank():
    _raises(lambda: _g2x2()(X), ValueError)
    _raises(lambda: _g2x2()(X, Y, Z), ValueError)
    _raises(lambda: _sc()(X), ValueError)


# ---------------------------------------------------------------------------
# the declaration block
# ---------------------------------------------------------------------------

def test_grid_block_is_exactly_ftsls_grid_statement():
    txt = _decl(GridDecl("g", _g2x2(), outside="wrap"))
    assert txt == (
        'grid "g" {\n'
        "    shape 2 2\n"
        "    lo 0 0\n"
        "    hi 1 1\n"
        "    outside wrap\n"
        "    data {\n"
        "        0 0.5\n"
        "        1 0.25\n"
        "    }\n"
        "}"
    )


def test_data_is_written_in_c_order_axis_zero_outermost():
    """ftrace's `flat = flat * n + idx` walks axis 0 outermost — the same order
    Grid._strides uses — so one row of the block is one run of the fastest axis."""
    g = Grid([[1, 2, 3], [4, 5, 6]], lo=0.0, hi=(1.0, 1.0))
    rows = [ln.strip() for ln in _decl(GridDecl("g", g)).split("\n")]
    assert "1 2 3" in rows and "4 5 6" in rows
    assert rows.index("1 2 3") < rows.index("4 5 6")


def test_absent_hi_emits_the_unit_spacing_index_lattice():
    """`hi=None` means "a query coordinate is a sample index" — the block must carry
    the resolved corner, not a blank, because ftsl's own default is only a mirror."""
    g = Grid([[1, 2, 3], [4, 5, 6]])          # shape (2, 3)
    txt = _decl(GridDecl("g", g))
    assert "    lo 0 0\n" in txt and "    hi 1 2\n" in txt


def test_scalar_hi_emits_the_isotropic_lattice_loom_resolved():
    g = Grid([[1, 2], [3, 4], [5, 6]], lo=0.0, hi=1.0)   # shape (3, 2)
    txt = _decl(GridDecl("g", g))                        # h = 1/2 -> hi = (1, 0.5)
    assert "    hi 1 0.5\n" in txt


def test_scatter_block_interleaves_position_and_value():
    txt = _decl(ScatterDecl("s", _sc()))
    assert txt == (
        'scatter "s" {\n'
        "    dim 2\n"
        "    power 2\n"
        "    eps 1e-09\n"
        "    data {\n"
        "        0 0   1\n"
        "        1 0   2\n"
        "        0.5 1   -0.5\n"
        "    }\n"
        "}"
    )


def test_block_values_are_baked_at_the_emit_clock():
    """A Grid's values are Signals, so a modulated grid re-emits new numbers every
    frame while the lattice stays put — the Grid contract, on the render path."""
    g = Grid([[Ramp(0.0, 1.0), 0.0], [0.0, 0.0]], lo=0.0, hi=(1.0, 1.0))
    d = GridDecl("g", g)
    assert "        0 0\n" in _decl(d, Clock(t=0.0, frame=0, frames=2, fps=1.0))
    assert "        1 0\n" in _decl(d, Clock(t=1.0, frame=1, frames=2, fps=1.0))


def test_scatter_positions_are_baked_too():
    s = Scatter([((Ramp(0.0, 1.0), 0.0), 5.0)])
    d = ScatterDecl("s", s)
    assert "        0 0   5\n" in _decl(d, Clock(t=0.0, frame=0, frames=2, fps=1.0))
    assert "        1 0   5\n" in _decl(d, Clock(t=1.0, frame=1, frames=2, fps=1.0))


def test_grid_decl_rejects_an_unknown_out_of_domain_policy():
    _raises(lambda: GridDecl("g", _g2x2(), outside="mirror"), ValueError)


def test_declaration_roots_expose_the_value_signals_for_cycle_checking():
    r = Ramp(0.0, 1.0)
    g = Grid([[r, 0.0], [0.0, 0.0]], lo=0.0, hi=(1.0, 1.0))
    assert r in GridDecl("g", g).roots()
    s = Scatter([((0.0, 0.0), r)])
    assert r in ScatterDecl("s", s).roots()


# ---------------------------------------------------------------------------
# auto-collection — the author never declares the block
# ---------------------------------------------------------------------------

def test_auto_name_is_shared_by_leaves_that_sample_the_same_way():
    g = _g2x2()
    assert g(X, Y).name == g(U, V).name


def test_auto_name_separates_leaves_that_cannot_share_a_block():
    """`outside` lives on the block, not the call, so two policies need two blocks;
    likewise a scatter's `power`/`eps`."""
    g = _g2x2()
    names = {g(X, Y, on_outside=p).name for p in ("clamp", "wrap", "extrapolate")}
    assert len(names) == 3
    s = _sc()
    assert s(X, Y).name != s(X, Y, power=3.0).name
    assert s(X, Y).name != s(X, Y, eps=1e-6).name


def test_table_decls_dedupes_and_keeps_encounter_order():
    g, s = _g2x2(), _sc()
    e = g(X, Y) * s(X, Y) + g(X, Y)
    decls = e.table_decls()
    assert [d.name for d in decls] == [g(X, Y).name, s(X, Y).name]
    assert isinstance(decls[0], GridDecl) and isinstance(decls[1], ScatterDecl)


def test_table_decls_finds_leaves_under_warped_coordinates():
    inner, outer = _g2x2(), _g3x3()
    e = outer(X + 0.1 * inner(X, Y), Y)
    assert sorted(d.name for d in e.table_decls()) == sorted(
        [inner(X, Y).name, outer(X, Y).name])


def test_scene_auto_declares_the_block_a_material_field_samples():
    g = _g3x3()
    sc = Scene(_cam())
    sc.add(Material("wall", "glossy", roughness=0.02 + 0.4 * g(X, Z)),
           Sphere((0, 0, 0), 1, "wall"), _key())
    txt = sc.emit(_clock(), Cache())
    assert f'grid "grid_{g.id}_clamp" {{' in txt
    assert f"grid:grid_{g.id}_clamp(" in txt
    assert "roughness pattern:wall_roughness" in txt


def test_auto_declared_block_precedes_every_use():
    """ftrace loads tables in Pass 1a, before patterns — emitting the block first
    keeps the text readable top-down and the order non-load-bearing either way."""
    g = _g3x3()
    sc = Scene(_cam())
    sc.add(Material("wall", "glossy", roughness=g(X, Z)),
           Sphere((0, 0, 0), 1, "wall"), _key())
    txt = sc.emit(_clock(), Cache())
    assert txt.index(f'grid "grid_{g.id}_clamp"') < txt.index("wall_roughness = pattern")


def test_scene_declares_a_shared_dataset_exactly_once():
    g = _g3x3()
    sc = Scene(_cam())
    sc.add(Material("a", "glossy", roughness=g(X, Y)),
           Material("b", "glossy", roughness=1.0 - g(X, Y)),
           Sphere((0, 0, 0), 1, "a"), _key())
    txt = sc.emit(_clock(), Cache())
    assert txt.count(f'grid "grid_{g.id}_clamp" {{') == 1


def test_two_policies_over_one_grid_emit_two_blocks():
    g = _g3x3()
    sc = Scene(_cam())
    sc.add(Material("a", "glossy", roughness=g(X, Y)),
           Material("b", "glossy", roughness=g(X, Y, on_outside="wrap")),
           Sphere((0, 0, 0), 1, "a"), _key())
    txt = sc.emit(_clock(), Cache())
    assert f'grid "grid_{g.id}_clamp" {{' in txt
    assert f'grid "grid_{g.id}_wrap" {{' in txt
    assert "outside clamp" in txt and "outside wrap" in txt


def test_an_explicit_declaration_of_the_same_name_wins_in_either_order():
    g = _g3x3()
    name = g(X, Y).name
    hand = GridDecl(name, Grid([[9.0, 9.0], [9.0, 9.0]], lo=0.0, hi=(1.0, 1.0)))
    mat = Material("a", "glossy", roughness=g(X, Y))
    for order in ((hand, mat), (mat, hand)):
        sc = Scene(_cam())
        sc.add(*order)
        sc.add(Sphere((0, 0, 0), 1, "a"), _key())
        txt = sc.emit(_clock(), Cache())
        assert txt.count(f'grid "{name}" {{') == 1
        assert "        9 9\n" in txt      # the hand-written block is the one kept


def test_a_colour_slot_samples_the_table_through_a_proc_texture():
    g = _g3x3()
    sc = Scene(_cam())
    sc.add(Material("skin", "diffuse", reflect=(g(U, V), 0.3, 0.25)),
           Sphere((0, 0, 0), 1, "skin"), _key())
    txt = sc.emit(_clock(), Cache())
    assert f'grid "grid_{g.id}_clamp" {{' in txt
    assert f"grid:grid_{g.id}_clamp((u),(v))" in txt


# ---------------------------------------------------------------------------
# placement — ftsl's grid block has no transform, so it folds into the query
# ---------------------------------------------------------------------------

def _world_points():
    return ([-0.4, 0.1, 0.5, 0.9, 1.7], [0.2, -0.3, 0.5, 1.1, 0.4])


def _temporal_at(g, xs, ys, **kw):
    """What a world-space *curve* reads — the temporal path, placement and all."""
    return [g.sample(x, y, **kw) for x, y in zip(xs, ys)]


def test_placement_is_folded_into_the_emitted_coordinates():
    g = _g3x3().transformed(translate=(0.3, -0.2, 0.0), scale=(2.0, 1.0, 1.0),
                            rotate=(0.0, 0.0, 30.0))
    txt = _emit(g(X, Y))
    assert txt.startswith(f"grid:grid_{g.id}_clamp(")
    assert "(x)" in txt and "(y)" in txt
    assert "0.3" in txt and "-0.2" in txt        # the placement really is in there


def test_a_placed_grid_reads_the_same_values_on_both_tiers():
    g = _g3x3().transformed(translate=(0.3, -0.2, 0.0), scale=(2.0, 1.0, 1.0),
                            rotate=(0.0, 0.0, 30.0), skew=(0.25, 0.0, 0.0))
    xs, ys = _world_points()
    got = _np_at(g(X, Y), xs, ys)
    want = _temporal_at(g, xs, ys)
    assert np.allclose(got, want, atol=1e-12)


def test_a_placed_scatter_reads_the_same_values_on_both_tiers():
    s = _sc().transformed(translate=(0.5, 0.25, 0.0), scale=(1.5, 0.75, 1.0))
    xs, ys = _world_points()
    got = _np_at(s(X, Y), xs, ys)
    want = [s.sample(x, y) for x, y in zip(xs, ys)]
    assert np.allclose(got, want, atol=1e-12)


def test_the_two_inverse_maps_are_the_same_formula():
    """`inverse_apply` (Signals) and `inverse_apply_spatial` (field expressions) share
    one body, so a placement can never mean two different things."""
    xf = Transform(translate=(0.3, -0.2, 0.0), scale=(2.0, 1.0, 1.0),
                   rotate=(0.0, 0.0, 30.0), skew=(0.25, 0.0, 0.0))
    xs, ys = _world_points()
    spatial = xf.inverse_apply_spatial([X, Y])
    for x, y in zip(xs, ys):
        temporal = xf.inverse_apply(VecSignal.of((x, y))).at(_clock())
        for a in range(2):
            got = float(np.asarray(_np_at(spatial[a], [x], [y])).ravel()[0])
            assert abs(got - float(temporal[a])) < 1e-12


def test_rebuilding_does_not_re_apply_the_placement():
    """`substitute` walks the coordinate arguments, which are ALREADY local — folding
    the transform in a second time would silently double it."""
    g = _g3x3().transformed(translate=(0.3, -0.2, 0.0), scale=(2.0, 1.0, 1.0))
    e = g(X, Y)
    assert _emit(e.substitute({})) == _emit(e)
    bound = g(U, V).substitute({"u": X, "v": Y})
    assert _emit(bound) == _emit(e)


def test_an_unplaced_dataset_emits_bare_coordinates():
    g = _g3x3()
    assert _emit(g(X, Y)) == f"grid:grid_{g.id}_clamp((x),(y))"


# ---------------------------------------------------------------------------
# eval_np — the numpy twin of patGridSample / patScatterSample
# ---------------------------------------------------------------------------

def _random_points(n, rng):
    return rng.uniform(-1.5, 2.5, n), rng.uniform(-1.5, 2.5, n)


def test_eval_np_matches_looms_own_interpolator_for_every_policy():
    g = _g3x3()
    rng = np.random.default_rng(7)
    xs, ys = _random_points(200, rng)
    for policy in ("clamp", "wrap", "extrapolate"):
        got = _np_at(g(X, Y, on_outside=policy), xs, ys)
        want = _temporal_at(g, xs, ys, on_outside=policy)
        assert np.max(np.abs(got - np.asarray(want))) < 1e-12, policy


def test_eval_np_matches_a_one_dimensional_grid():
    g = Grid([0.0, 1.0, 0.5, 0.25], lo=0.0, hi=1.0)
    xs = np.linspace(-0.2, 1.2, 17)
    got = g(X).eval_np((xs, np.zeros_like(xs), np.zeros_like(xs)), _clock(), Cache())
    want = [g.sample(float(x)) for x in xs]
    assert np.allclose(got, want, atol=1e-12)


def test_eval_np_matches_a_three_dimensional_grid():
    vals = [[[float((i * 3 + j) * 3 + k) for k in range(3)] for j in range(3)]
            for i in range(3)]
    g = Grid(vals, lo=0.0, hi=(1.0, 1.0, 1.0))
    rng = np.random.default_rng(11)
    xs, ys, zs = (rng.uniform(-0.3, 1.3, 40) for _ in range(3))
    got = g(X, Y, Z).eval_np((xs, ys, zs), _clock(), Cache())
    want = [g.sample(float(a), float(b), float(c)) for a, b, c in zip(xs, ys, zs)]
    assert np.allclose(got, want, atol=1e-12)


def test_eval_np_matches_the_shepard_field():
    s = _sc()
    rng = np.random.default_rng(3)
    xs, ys = _random_points(150, rng)
    for power in (2.0, 1.0, 3.5):
        got = _np_at(s(X, Y, power=power), xs, ys)
        want = [s.sample(float(a), float(b), power=power) for a, b in zip(xs, ys)]
        assert np.max(np.abs(got - np.asarray(want))) < 1e-12, power


def test_a_coincident_scatter_query_returns_that_sample_exactly():
    """`patScatterSample` returns a sample within `eps` immediately — an inverse-
    distance blend at zero distance would otherwise be 0/0."""
    s = _sc()
    got = _np_at(s(X, Y), [0.0, 1.0, 0.5], [0.0, 0.0, 1.0])
    assert np.allclose(got, [1.0, 2.0, -0.5], atol=0.0)


def test_eval_np_composes_with_the_rest_of_the_algebra():
    g = _g2x2()
    got = _np_at(1.0 - g(X, Y), [0.0], [0.0])
    assert abs(float(np.asarray(got).ravel()[0]) - (1.0 - g.sample(0.0, 0.0))) < 1e-12


def test_eval_np_preserves_the_coordinate_grid_shape():
    gx, gy = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 5))
    out = _g3x3()(X, Y).eval_np((gx, gy, np.zeros_like(gx)), _clock(), Cache())
    assert np.asarray(out).shape == gx.shape


def test_eval_np_broadcasts_a_scalar_coordinate():
    g = _g3x3()
    xs = np.linspace(0.0, 1.0, 6)
    out = g(X, 0.5).eval_np((xs, np.zeros_like(xs), np.zeros_like(xs)),
                            _clock(), Cache())
    want = [g.sample(float(x), 0.5) for x in xs]
    assert np.allclose(out, want, atol=1e-12)


def test_animated_values_evaluate_at_the_given_clock_on_both_paths():
    g = Grid([[Ramp(0.0, 1.0), 0.0], [0.0, 0.0]], lo=0.0, hi=(1.0, 1.0))
    late = Clock(t=1.0, frame=1, frames=2, fps=1.0)
    got = _np_at(g(X, Y), [0.0], [0.0], clock=late)
    assert abs(float(np.asarray(got).ravel()[0]) - 1.0) < 1e-12
    assert abs(float(np.asarray(_np_at(g(X, Y), [0.0], [0.0])).ravel()[0])) < 1e-12


# ---------------------------------------------------------------------------
# refusals — what ftrace cannot express is not approximated
# ---------------------------------------------------------------------------

def test_a_vector_valued_dataset_cannot_be_an_ftsl_term():
    """PatGrid / PatScatter store scalar floats."""
    g = Grid([vec(0.0, 1.0), vec(1.0, 0.0)], lo=0.0, hi=1.0)
    _raises(lambda: g(X), TypeError)
    s = Scatter([((0.0, 0.0), vec(1.0, 0.0)), ((1.0, 1.0), vec(0.0, 1.0))])
    _raises(lambda: s(X, Y), TypeError)


def test_cubic_interpolation_has_no_ftsl_spelling():
    """patGridSample is separable N-linear only."""
    _raises(lambda: _g3x3()(X, Y, interp="cubic"), ValueError)
    # ...but it is still available on the temporal path, which is not a renderer
    assert isinstance(_g3x3()(0.5, 0.5, interp="cubic"), GridField)


def test_on_outside_raise_has_no_ftsl_spelling():
    """A renderer samples millions of times per frame and cannot throw."""
    _raises(lambda: _g3x3()(X, Y, on_outside="raise"), ValueError)
    assert isinstance(_g3x3()(0.5, 0.5, on_outside="raise"), GridField)


def test_four_axes_are_allowed_and_five_are_refused():
    """PAT_ND_MAX_DIM is 4 — the boundary must be usable, not off by one."""
    g4 = Grid([float(i) for i in range(16)], shape=(2, 2, 2, 2))
    assert isinstance(g4(X, Y, Z, X), GridSample)
    g5 = Grid([float(i) for i in range(32)], shape=(2, 2, 2, 2, 2))
    _raises(lambda: g5(X, Y, Z, X, Y), ValueError)


def test_a_five_dimensional_scatter_is_refused():
    s = Scatter([((0.0,) * 5, 1.0), ((1.0,) * 5, 2.0)])
    _raises(lambda: s(X, Y, Z, X, Y), ValueError)


def test_degenerate_shepard_settings_are_rejected():
    _raises(lambda: _sc()(X, Y, power=0.0), ValueError)
    _raises(lambda: _sc()(X, Y, power=-1.0), ValueError)
    _raises(lambda: _sc()(X, Y, eps=-1.0), ValueError)


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
