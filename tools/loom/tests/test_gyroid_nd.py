"""Tests for the higher-dimensional gyroid generator (``examples/gyroid_nd.py``).

Cover the field-expression correctness (the D=3 lock reproduces the *exact* Schoen
gyroid, verified numerically), reproducibility from a seed, that every lock is
honoured (dims / oscillating count / harmonics count / forced axis on-off-harmonic),
and that bad or infeasible locks raise cleanly.  Runnable directly or under pytest.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples"))

import gyroid_nd as g  # noqa: E402


def _args(*argv):
    return g.build_parser().parse_args(list(argv))


def _locks(*specs):
    d = {}
    for s in specs:
        g.parse_axis_lock(s, d)
    return d


def _eval_expr(expr, x, y, z):
    """Evaluate an emitted ftsl field expression numerically (it is valid Python)."""
    return eval(expr, {"sin": math.sin, "cos": math.cos, "x": x, "y": y, "z": z})


def _true_gyroid(x, y, z, f=1.0):
    return (math.sin(f * x) * math.cos(f * y)
            + math.sin(f * y) * math.cos(f * z)
            + math.sin(f * z) * math.cos(f * x))


def test_classic_gyroid_is_exact():
    args = _args("--dims", "3", "--axis", "0:on:1", "--axis", "1:on:1",
                 "--axis", "2:on:1", "--phase0", "--freq", "1")
    locks = _locks("0:on:1", "1:on:1", "2:on:1")
    v = g.pick_variant(123, args, locks)
    expr = g.field_expr(v)
    # numerically identical to sin x cos y + sin y cos z + sin z cos x
    for (x, y, z) in [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]:
        assert abs(_eval_expr(expr, x, y, z) - _true_gyroid(x, y, z)) < 1e-9


def test_freq_scales_gyroid():
    args = _args("--dims", "3", "--axis", "0:on:1", "--axis", "1:on:1",
                 "--axis", "2:on:1", "--phase0", "--freq", "6")
    v = g.pick_variant(1, args, _locks("0:on:1", "1:on:1", "2:on:1"))
    expr = g.field_expr(v)
    for (x, y, z) in [(0.3, 1.1, -0.7), (0.1, 0.2, 0.3)]:
        assert abs(_eval_expr(expr, x, y, z) - _true_gyroid(x, y, z, 6.0)) < 1e-8


def test_reproducible_from_seed():
    args = _args()
    a = g.pick_variant(555, args, {})
    b = g.pick_variant(555, args, {})
    assert g.field_expr(a) == g.field_expr(b)
    assert [(d.oscillate, d.harmonic, d.direction, d.phase) for d in a.dim_list] == \
           [(d.oscillate, d.harmonic, d.direction, d.phase) for d in b.dim_list]


def test_dims_lock_respected():
    args = _args("--dims", "5")
    for s in range(20):
        assert g.pick_variant(s, args, {}).dims == 5


def test_oscillating_count_lock_respected():
    args = _args("--dims", "7", "--oscillating", "4")
    for s in range(20):
        assert len(g.pick_variant(s, args, {}).oscillating) == 4


def test_harmonics_count_lock_respected():
    args = _args("--dims", "8", "--oscillating", "6", "--harmonics", "3")
    for s in range(20):
        v = g.pick_variant(s, args, {})
        assert len(v.harmonic_dims) == 3


def test_forced_axis_off_never_oscillates():
    args = _args("--dims", "6")
    locks = _locks("2:off")
    for s in range(20):
        assert 2 not in g.pick_variant(s, args, locks).oscillating


def test_forced_axis_on_and_harmonic():
    args = _args("--dims", "6")
    locks = _locks("4:on:3")
    for s in range(20):
        v = g.pick_variant(s, args, locks)
        d4 = v.dim_list[4]
        assert d4.oscillate and d4.harmonic == 3


def test_classic_lock_has_no_harmonics_or_extra_dims():
    args = _args("--dims", "3", "--axis", "0:on:1", "--axis", "1:on:1", "--axis", "2:on:1")
    v = g.pick_variant(9, args, _locks("0:on:1", "1:on:1", "2:on:1"))
    assert v.oscillating == [0, 1, 2]
    assert v.harmonic_dims == []
    assert v.main == 0


def test_expr_is_wellformed():
    # random variants across seeds: balanced parens, no '+-'/'++' operator adjacency
    args = _args("--dims", "8")
    for s in range(30):
        expr = g.field_expr(g.pick_variant(s, args, {}))
        assert expr.count("(") == expr.count(")")
        assert "+-" not in expr and "++" not in expr and "*-" not in expr
        # every variant must have at least the two terms a gyroid needs
        assert expr.count("sin(") >= 2 and expr.count("cos(") >= 2


def test_bad_axis_spec_raises():
    for bad in ("2", "2:maybe", "x:on", "2:on:-1", "2:off:3"):
        try:
            g.parse_axis_lock(bad, {})
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(f"expected ArgumentTypeError for {bad!r}")


def test_axis_double_lock_conflict_raises():
    for specs in (("2:on", "2:off"), ("4:on:3", "4:on:5")):
        try:
            _locks(*specs)
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(f"expected conflict error for {specs}")


def test_infeasible_locks_raise():
    # oscillating count larger than available dims
    try:
        g.pick_variant(1, _args("--dims", "4", "--oscillating", "9"), {})
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit for oscillating > dims")
    # harmonics count larger than non-main oscillating dims
    try:
        g.pick_variant(1, _args("--dims", "3", "--oscillating", "3", "--harmonics", "9"), {})
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit for too many harmonics")


def test_axis_index_beyond_locked_dims_raises():
    try:
        g.pick_variant(1, _args("--dims", "3"), _locks("5:on"))
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit for axis index >= dims")


def test_static_field_expr_unchanged_at_t0():
    # field_expr default t=0 preserves the exact static (classic) field
    args = _args("--dims", "3", "--axis", "0:on:1", "--axis", "1:on:1",
                 "--axis", "2:on:1", "--phase0", "--freq", "1")
    v = g.pick_variant(1, args, _locks("0:on:1", "1:on:1", "2:on:1"))
    assert g.field_expr(v) == g.field_expr(v, 0.0)


def test_winding_main_anchored_others_drift():
    args = _args("--dims", "6")
    for s in range(20):
        v = g.pick_variant(s, args, {})
        by = {d.index: d for d in v.dim_list}
        assert by[v.main].winding == 0            # main is the still anchor
        for d in v.oscillating:                   # every other osc dim drifts
            if d != v.main:
                assert by[d].winding >= 1
        for d in v.dim_list:                      # inert dims never drift
            if not d.oscillate:
                assert d.winding == 0


def test_max_winding_lock_respected():
    args = _args("--dims", "8", "--oscillating", "6", "--max-winding", "1")
    for s in range(20):
        v = g.pick_variant(s, args, {})
        assert all(d.winding <= 1 for d in v.dim_list)


def test_video_loop_is_seamless():
    # t=0 and t=1 give a numerically identical field (whole-cycle phase advance)
    v = g.pick_variant(3, _args("--dims", "6"), {})
    e0, e1 = g.field_expr(v, 0.0), g.field_expr(v, 1.0)
    for (x, y, z) in [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]:
        assert abs(_eval_expr(e0, x, y, z) - _eval_expr(e1, x, y, z)) < 1e-6


def test_video_actually_moves():
    # mid-loop the field differs from the start (the pattern really morphs)
    v = g.pick_variant(3, _args("--dims", "6"), {})
    e0, eh = g.field_expr(v, 0.0), g.field_expr(v, 0.37)
    diff = max(abs(_eval_expr(e0, x, y, z) - _eval_expr(eh, x, y, z))
               for (x, y, z) in [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)])
    assert diff > 1e-3


def test_rotate_starts_from_current_gyroid():
    # the rotate transform at t=0 is the exact static field (same as drift at t=0)
    v = g.pick_variant(3, _args("--dims", "6"), {})
    assert g.field_expr(v, 0.0, "rotate") == g.field_expr(v, 0.0, "drift")
    assert g.field_expr(v, 0.0, "rotate") == g.field_expr(v)


def test_rotate_loop_is_seamless():
    # t=0 and t=1 evaluate identically under rotate (whole-turn wavevector rotation)
    v = g.pick_variant(7, _args("--dims", "6"), {})
    e0, e1 = g.field_expr(v, 0.0, "rotate"), g.field_expr(v, 1.0, "rotate")
    for (x, y, z) in [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]:
        assert abs(_eval_expr(e0, x, y, z) - _eval_expr(e1, x, y, z)) < 1e-6


def test_rotate_actually_moves_and_differs_from_drift():
    # mid-loop rotate morphs the field, and does so differently from drift
    v = g.pick_variant(7, _args("--dims", "6"), {})
    e0 = g.field_expr(v, 0.0, "rotate")
    er = g.field_expr(v, 0.31, "rotate")
    ed = g.field_expr(v, 0.31, "drift")
    pts = [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]
    moved = max(abs(_eval_expr(e0, *p) - _eval_expr(er, *p)) for p in pts)
    vs_drift = max(abs(_eval_expr(er, *p) - _eval_expr(ed, *p)) for p in pts)
    assert moved > 1e-3          # it really morphs
    assert vs_drift > 1e-3       # and it is a genuinely different motion than drift


def test_rotate_well_formed_across_seeds():
    args = _args("--dims", "8")
    for s in range(20):
        v = g.pick_variant(s, args, {})
        for t in (0.0, 0.13, 0.5, 0.77, 1.0):
            expr = g.field_expr(v, t, "rotate")
            assert expr.count("(") == expr.count(")")
            assert "+-" not in expr and "++" not in expr and "*-" not in expr


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
