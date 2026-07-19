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
import re
import sys
import types

import pytest

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


def test_bloom_frame0_is_exact_classic_gyroid():
    # In bloom mode, frame 0 must be *exactly* the classic showcase gyroid regardless of
    # how many higher dimensions the variant has.
    args = _args("--dims", "6", "--oscillate", "bloom", "--freq", "1")
    v = g.pick_variant(99, args, {})
    expr = g.field_expr(v, 0.0, "bloom")
    for (x, y, z) in [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]:
        assert abs(_eval_expr(expr, x, y, z) - _true_gyroid(x, y, z)) < 1e-9


def test_bloom_is_seamless():
    # t=0 and t=1 both collapse to the classic gyroid -> identical expression (seamless loop).
    args = _args("--dims", "7", "--oscillate", "bloom", "--freq", "2")
    v = g.pick_variant(7, args, {})
    assert g.field_expr(v, 0.0, "bloom") == g.field_expr(v, 1.0, "bloom")
    assert g.field_expr(v, 0.0, "bloom") == g._classic_gyroid_expr(v.freq)


def test_bloom_midpoint_is_full_field():
    # At t=0.5 the envelope is 1, so the field is the full N-D gyroid (drifting), not classic.
    args = _args("--dims", "6", "--oscillate", "bloom", "--freq", "3")
    v = g.pick_variant(11, args, {})
    assert g.field_expr(v, 0.5, "bloom") == g.field_expr(v, 0.5, "drift")
    assert g.field_expr(v, 0.5, "bloom") != g._classic_gyroid_expr(v.freq)


def test_bloom_default_freq_matches_showcase_density():
    # With no --freq, bloom defaults the density to showcase's (freq 40 at radius 0.32).
    args = _args("--dims", "5", "--oscillate", "bloom", "--radius", "0.32")
    v = g.pick_variant(3, args, {})
    assert abs(v.freq - 40.0) < 1e-6
    # and the density scales inversely with radius (same periods across the ball)
    args2 = _args("--dims", "5", "--oscillate", "bloom", "--radius", "1.3")
    assert abs(g.pick_variant(3, args2, {}).freq - (0.32 * 40.0 / 1.3)) < 1e-6


def test_bloom_default_param_is_dims():
    # With no --bloom, the bloom transform blooms 'dims' (backward-compatible behavior).
    v = g.pick_variant(5, _args("--dims", "6", "--oscillate", "bloom"), {})
    assert v.bloom_params == ("dims",)


def test_bloom_param_parsing_and_aliases():
    assert g._parse_bloom_params(None) == ("dims",)
    assert g._parse_bloom_params("") == ("dims",)
    assert g._parse_bloom_params("freq") == ("freq",)
    assert g._parse_bloom_params("complexity") == ("freq",)   # friendly alias
    assert g._parse_bloom_params("intricacy") == ("freq",)
    assert g._parse_bloom_params("dims,freq") == ("dims", "freq")
    assert g._parse_bloom_params("freq,freq") == ("freq",)     # de-duped
    with pytest.raises(SystemExit):
        g._parse_bloom_params("wobble")


def test_bloom_freq_holds_classic_and_pulses_frequency():
    # --bloom freq: the whole loop is the classic gyroid, but its frequency swells at mid-loop.
    v = g.pick_variant(5, _args("--oscillate", "freq", "--freq", "5"), {})
    assert v.bloom_params == ("freq",)
    # frame 0 is the exact classic gyroid at the base frequency
    assert g.field_expr(v, 0.0, "bloom") == g._classic_gyroid_expr(5.0)
    # mid-loop is still the classic gyroid but at 2x frequency (amp 1 -> +1*swing*1)
    assert abs(g.bloom_freq(v, 0.5) - 10.0) < 1e-9
    assert g.field_expr(v, 0.5, "bloom") == g._classic_gyroid_expr(g.bloom_freq(v, 0.5))
    # and it never unfolds higher-D structure (no 'dims' bloom)
    assert g.field_expr(v, 0.5, "bloom") != g.field_expr(v, 0.5, "drift")


def test_bloom_amp_scales_frequency_swing():
    v = g.pick_variant(5, _args("--oscillate", "0.5*freq", "--freq", "4"), {})
    # amp 0.5 -> mid-loop frequency is 4*(1 + 0.5*1) = 6
    assert abs(g.bloom_freq(v, 0.5) - 6.0) < 1e-9
    assert abs(g.bloom_freq(v, 0.0) - 4.0) < 1e-9      # ends unchanged


def test_bloom_amps_empty_defaults_to_shared_bloom_amp():
    # the legacy path leaves per-axis bloom_amps empty, so _swing_amp falls back to
    # the shared bloom_amp scalar -> byte-identical to before the per-axis refactor.
    v = g.pick_variant(5, _args("--transform", "bloom", "--bloom", "freq",
                                "--freq", "4", "--bloom-amp", "0.5"), {})
    assert v.bloom_amps == {}
    assert g._swing_amp(v, "freq") == v.bloom_amp == 0.5
    assert abs(g.bloom_freq(v, 0.5) - 6.0) < 1e-9


def test_bloom_amps_per_axis_override_beats_shared_scalar():
    # a per-axis override in bloom_amps takes precedence over the shared bloom_amp for
    # that swinger only; other swingers still use the shared scalar. This is what the
    # unified --oscillate grammar's per-item amp (e.g. 2*freq,0.5*threshold) will set.
    v = g.pick_variant(5, _args("--transform", "bloom", "--bloom", "freq,threshold",
                                "--freq", "4", "--bloom-amp", "1.0"), {})
    v.bloom_amps = {"freq": 0.5}                 # override freq only
    assert g._swing_amp(v, "freq") == 0.5        # overridden
    assert g._swing_amp(v, "threshold") == 1.0   # falls back to shared bloom_amp
    assert abs(g.bloom_freq(v, 0.5) - 6.0) < 1e-9   # 4*(1 + 0.5*1)
    # threshold still swings at the shared amp 1.0 * _BLOOM_SWING['threshold'] (0.6)
    assert abs(g.bloom_threshold(v, 0.5) - (v.threshold + 0.6)) < 1e-9


def test_bloom_freq_applies_to_full_field_when_dims_also_bloom():
    # --bloom dims,freq: the frequency pulse rides on the full N-D crossfade too.
    v = g.pick_variant(5, _args("--dims", "6", "--oscillate", "bloom,freq", "--freq", "3"), {})
    assert v.bloom_params == ("dims", "freq")
    assert g.field_expr(v, 0.0, "bloom") == g._classic_gyroid_expr(3.0)   # seamless frame 0
    # mid-loop equals the drift field evaluated at the bloomed (2x) frequency
    assert g.field_expr(v, 0.5, "bloom") == g.field_expr(v, 0.5, "drift",
                                                         freq=g.bloom_freq(v, 0.5))


def test_bloom_threshold_and_thickness_are_seamless_scalars():
    v = g.pick_variant(5, _args("--oscillate", "threshold,thickness"), {})
    assert v.bloom_params == ("threshold", "thickness")
    # both scalars equal their base at the loop ends and swing at the midpoint
    assert g.bloom_threshold(v, 0.0) == v.threshold
    assert g.bloom_threshold(v, 1.0) == v.threshold
    assert g.bloom_threshold(v, 0.5) != v.threshold
    assert g.bloom_thickness_scale(v, 0.0) == 1.0
    assert g.bloom_thickness_scale(v, 1.0) == 1.0
    assert g.bloom_thickness_scale(v, 0.5) > 1.0
    # the field itself is the frozen classic gyroid (no dims/freq bloom)
    assert g.field_expr(v, 0.3, "bloom") == g._classic_gyroid_expr(v.freq)


def test_bloom_gradient_bound_tracks_bloomed_frequency():
    # The Lipschitz bound must grow with the frequency pulse or the marcher will miss walls.
    from loom import Clock, Cache
    v = g.pick_variant(5, _args("--oscillate", "freq", "--freq", "5"), {})
    sh = max(3, sum(d.harmonic for d in v.dim_list if d.oscillate))

    def bound(t):
        body = g.build_scene(v, t=t, res=(32, 32), radius=1.3,
                             transform="bloom").emit(Clock(t=t), Cache())
        return float(re.search(r"max_gradient ([0-9.]+)", body).group(1))

    assert abs(bound(0.0) - 2.2 * 5.0 * sh) < 0.01
    assert abs(bound(0.5) - 2.2 * 10.0 * sh) < 0.01    # 2x freq -> 2x bound


def test_bloom_requires_bloom_transform():
    with pytest.raises(SystemExit):
        g.main(["--transform", "drift", "--bloom", "freq", "--no-video"])


def test_next_run_dir_increments(tmp_path):
    base = tmp_path / "gyroid_nd"
    d1 = g._next_run_dir(base)
    assert d1.name == "run001"
    d1.mkdir(parents=True)
    d2 = g._next_run_dir(base)
    assert d2.name == "run002"
    d2.mkdir()
    # a gap (deleting run001) never reuses a number: next is still one past the max
    import shutil
    shutil.rmtree(d1)
    assert g._next_run_dir(base).name == "run003"


def _scene_body(v, **kw):
    from loom import Clock, Cache
    return g.build_scene(v, res=(32, 32), radius=1.3, **kw).emit(Clock(t=0.0), Cache())


def test_material_gold_is_default_conductor():
    v = g.pick_variant(5, _args("--dims", "6"), {})
    body = _scene_body(v)                       # default material
    assert "preset gold" in body
    assert "dielectric" not in body


def test_material_glass_emits_clear_dielectric():
    v = g.pick_variant(5, _args("--dims", "6"), {})
    body = _scene_body(v, material="glass")
    assert "dielectric" in body
    assert "glass:BK7" in body
    assert "preset gold" not in body
    # the isosurface still references the shared surface material name
    assert 'material "surf"' in body


def test_material_invalid_raises():
    v = g.pick_variant(5, _args("--dims", "6"), {})
    with pytest.raises(SystemExit):
        _scene_body(v, material="wood")


def test_material_recorded_in_header():
    v = g.pick_variant(5, _args("--dims", "6"), {})
    assert "surface material      : glass" in g.header(v, 0, 1, material="glass")
    assert "surface material      : gold" in g.header(v, 0, 1, material="gold")


def test_material_cli_choice_validated():
    # argparse rejects an unknown --material choice
    with pytest.raises(SystemExit):
        _args("--material", "wood")
    assert _args("--material", "glass").material == "glass"


def _capture_render_cmd(monkeypatch, tmp_path, **kw):
    """Run _render_frame with subprocess stubbed out; return the ftrace argv it built."""
    import subprocess
    captured = {}

    class _R:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **_):
        captured["cmd"] = cmd
        return _R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    fp = tmp_path / "f.ftsl"
    fp.write_text("x")
    kw.setdefault("raster", True)
    g._render_frame("ftrace", tmp_path, fp, tmp_path / "f.png",
                    size=(16, 16), **kw)
    return captured["cmd"]


def test_raster_see_through_flag(monkeypatch, tmp_path):
    # clear material -> -see-through; gold -> not.
    cmd = _capture_render_cmd(monkeypatch, tmp_path, see_through=True)
    assert "-see-through" in cmd
    cmd = _capture_render_cmd(monkeypatch, tmp_path, see_through=False)
    assert "-see-through" not in cmd and "-glass-clarity" not in cmd


def test_raster_glass_clarity_flag(monkeypatch, tmp_path):
    # explicit clarity -> -glass-clarity <val> (which itself implies see-through in ftrace).
    cmd = _capture_render_cmd(monkeypatch, tmp_path, see_through=True, clarity=0.6)
    assert "-glass-clarity" in cmd
    assert "0.6" in cmd[cmd.index("-glass-clarity") + 1]


def test_path_traced_ignores_see_through(monkeypatch, tmp_path):
    # -see-through is a raster-only preview flag; it must never reach the path tracer.
    import subprocess
    captured = {}

    class _R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **_: (captured.__setitem__("cmd", cmd), _R())[1])
    fp = tmp_path / "f.ftsl"
    fp.write_text("x")
    g._render_frame("ftrace", tmp_path, fp, tmp_path / "f.png",
                    size=(16, 16), raster=False, noise=3.0, see_through=True, clarity=0.5)
    assert "-see-through" not in captured["cmd"]
    assert "-glass-clarity" not in captured["cmd"]


def test_path_trace_default_budget_noise4(monkeypatch, tmp_path):
    # no explicit budget -> back-compat default of -noise 4.
    cmd = _capture_render_cmd(monkeypatch, tmp_path, raster=False)
    assert "-noise" in cmd and cmd[cmd.index("-noise") + 1] == "4"
    assert "-time" not in cmd and "-spp" not in cmd


def test_path_trace_noise_budget(monkeypatch, tmp_path):
    cmd = _capture_render_cmd(monkeypatch, tmp_path, raster=False, noise=2.5)
    assert cmd[cmd.index("-noise") + 1] == "2.5"
    assert "-time" not in cmd and "-spp" not in cmd


def test_path_trace_time_budget(monkeypatch, tmp_path):
    cmd = _capture_render_cmd(monkeypatch, tmp_path, raster=False, time_budget=30.0)
    assert cmd[cmd.index("-time") + 1] == "30"
    # time given, no noise -> no default -noise added.
    assert "-noise" not in cmd and "-spp" not in cmd


def test_path_trace_spp_budget(monkeypatch, tmp_path):
    cmd = _capture_render_cmd(monkeypatch, tmp_path, raster=False, spp=64)
    assert cmd[cmd.index("-spp") + 1] == "64"
    assert "-noise" not in cmd and "-time" not in cmd


def test_path_trace_combined_budgets(monkeypatch, tmp_path):
    # all three combine; ftrace stops at whichever fires first.
    cmd = _capture_render_cmd(monkeypatch, tmp_path, raster=False,
                              noise=3.0, time_budget=20.0, spp=128)
    assert cmd[cmd.index("-noise") + 1] == "3"
    assert cmd[cmd.index("-time") + 1] == "20"
    assert cmd[cmd.index("-spp") + 1] == "128"


def test_raster_ignores_budget_flags(monkeypatch, tmp_path):
    # budgets are path-trace only; raster frames never carry -noise/-time/-spp.
    cmd = _capture_render_cmd(monkeypatch, tmp_path, raster=True,
                              noise=3.0, time_budget=20.0, spp=128)
    assert "-noise" not in cmd and "-time" not in cmd and "-spp" not in cmd


def test_render_budget_cli_parsed():
    a = _args()
    assert a.render_noise is None and a.render_time is None and a.render_spp is None
    a = _args("--render-noise", "2", "--render-time", "45", "--render-spp", "100")
    assert a.render_noise == 2.0 and a.render_time == 45.0 and a.render_spp == 100


def test_glass_clarity_cli_parsed():
    assert _args("--glass-clarity", "0.7").glass_clarity == 0.7
    assert _args().glass_clarity is None


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


def test_tumble_starts_from_current_gyroid():
    # the tumble transform at t=0 is the identity rotation -> the exact static field
    v = g.pick_variant(3, _args("--dims", "6", "--oscillate", "tumble"), {})
    assert g.field_expr(v, 0.0, "tumble") == g.field_expr(v, 0.0, "drift")
    assert g.field_expr(v, 0.0, "tumble") == g.field_expr(v)


def test_tumble_loop_is_seamless():
    # t=0 and t=1 evaluate identically (whole-turn N-D rotation -> R(0)=R(1)=I)
    v = g.pick_variant(7, _args("--dims", "6", "--oscillate", "tumble"), {})
    e0, e1 = g.field_expr(v, 0.0, "tumble"), g.field_expr(v, 1.0, "tumble")
    for (x, y, z) in [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]:
        assert abs(_eval_expr(e0, x, y, z) - _eval_expr(e1, x, y, z)) < 1e-6


def test_tumble_actually_moves_and_differs_from_rotate():
    # mid-loop tumble morphs the field, differently from both drift and rotate (it is a
    # coherent whole-slice rotation, not per-dim wavevector tilts or a slide)
    v = g.pick_variant(7, _args("--dims", "6", "--oscillate", "tumble"), {})
    e0 = g.field_expr(v, 0.0, "tumble")
    et = g.field_expr(v, 0.31, "tumble")
    ed = g.field_expr(v, 0.31, "drift")
    er = g.field_expr(v, 0.31, "rotate")
    pts = [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]
    assert max(abs(_eval_expr(e0, *p) - _eval_expr(et, *p)) for p in pts) > 1e-3  # morphs
    assert max(abs(_eval_expr(et, *p) - _eval_expr(ed, *p)) for p in pts) > 1e-3  # != drift
    assert max(abs(_eval_expr(et, *p) - _eval_expr(er, *p)) for p in pts) > 1e-3  # != rotate


def test_tumble_planes_are_disjoint():
    # each dim index appears in at most one Givens plane, so a rotated direction row mixes
    # at most two unit rows (|dir| <= sqrt(2)) — the marcher's Lipschitz inflation relies on it
    for s in range(20):
        v = g.pick_variant(s, _args("--dims", "8", "--oscillate", "tumble"), {})
        seen = []
        for (i, j, _w) in v.tumble_planes:
            assert i != j
            seen += [i, j]
        assert len(seen) == len(set(seen))          # no index reused across planes


def test_tumble_well_formed_across_seeds():
    args = _args("--dims", "8", "--oscillate", "tumble")
    for s in range(20):
        v = g.pick_variant(s, args, {})
        assert v.tumble_planes                       # planes were built for tumble
        for t in (0.0, 0.13, 0.5, 0.77, 1.0):
            expr = g.field_expr(v, t, "tumble")
            assert expr.count("(") == expr.count(")")
            assert "+-" not in expr and "++" not in expr and "*-" not in expr


def test_tumble_3d_falls_back_to_rigid_spin():
    # with no hidden dims (D=3) tumble degenerates to a plane rotation of the visible axes;
    # it must still build a plane, stay seamless, and morph (a rigid spin of the gyroid)
    v = g.pick_variant(5, _args("--dims", "3", "--oscillate", "tumble"), {})
    assert v.tumble_planes                            # a fallback (0,2) plane exists
    e0, e1 = g.field_expr(v, 0.0, "tumble"), g.field_expr(v, 1.0, "tumble")
    eh = g.field_expr(v, 0.29, "tumble")
    pts = [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]
    assert max(abs(_eval_expr(e0, *p) - _eval_expr(e1, *p)) for p in pts) < 1e-6   # seamless
    assert max(abs(_eval_expr(e0, *p) - _eval_expr(eh, *p)) for p in pts) > 1e-3   # spins


def test_tumble_not_built_for_other_transforms():
    # tumble planes are only drawn when needed, so drift/rotate/bloom variants keep an empty
    # list — and their RNG stream (hence reproducibility) is untouched by the feature
    for tr in ("drift", "rotate", "bloom"):
        v = g.pick_variant(4, _args("--dims", "6", "--oscillate", tr), {})
        assert v.tumble_planes == []


def test_tumble_slide_is_seamless():
    # slide mode rocks the slice via sin(2*pi*winding*t); sin is 0 at t=0 and t=1, so both
    # loop ends are exactly the base gyroid — still a seamless loop
    v = g.pick_variant(7, _args("--dims", "6", "--oscillate", "0.25*tumble"), {})
    assert v.tumble_mode == "slide"
    e0, e1 = g.field_expr(v, 0.0, "tumble"), g.field_expr(v, 1.0, "tumble")
    assert e0 == g.field_expr(v)                          # frame 0 = base gyroid
    for (x, y, z) in [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]:
        assert abs(_eval_expr(e0, x, y, z) - _eval_expr(e1, x, y, z)) < 1e-6


def test_tumble_slide_differs_from_rotate_midloop():
    # slide and rotate share the same planes but different angle schedules, so mid-loop they
    # produce different fields (slide rocks +/-tumble_amp; rotate spins a full turn)
    base = ["--dims", "6"]
    vr = g.pick_variant(11, _args(*base, "--oscillate", "tumble"), {})
    vs = g.pick_variant(11, _args(*base, "--oscillate", "0.25*tumble"), {})
    assert vr.tumble_planes == vs.tumble_planes          # same planes (RNG untouched by mode)
    er = g.field_expr(vr, 0.31, "tumble")
    es = g.field_expr(vs, 0.31, "tumble")
    pts = [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]
    assert max(abs(_eval_expr(er, *p) - _eval_expr(es, *p)) for p in pts) > 1e-3


def test_tumble_slide_extreme_at_midloop_changes_projected_scale():
    # slide's angle peaks at t=0.25 (sin(2*pi*winding*t) max); with a large amp the slice is
    # tilted well away from identity, so the field must differ substantially from frame 0
    v = g.pick_variant(3, _args("--dims", "6", "--oscillate", "0.25*tumble"), {})
    e0 = g.field_expr(v, 0.0, "tumble")
    # peak tilt occurs where winding*t = 0.25 for a winding-1 plane, i.e. t=0.25
    epk = g.field_expr(v, 0.25, "tumble")
    pts = [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]
    assert max(abs(_eval_expr(e0, *p) - _eval_expr(epk, *p)) for p in pts) > 1e-3


def test_tumble_lock_excludes_axes_from_planes():
    # locked axes must never appear in any tumble plane (they stay fixed while others tumble)
    for s in range(20):
        v = g.pick_variant(s, _args("--dims", "8", "--oscillate", "tumble", "--lock", "0,1"), {})
        assert v.tumble_locked == (0, 1)
        for (i, j, _w) in v.tumble_planes:
            assert i not in (0, 1) and j not in (0, 1)


def test_tumble_lock_keeps_locked_axis_direction_static():
    # a locked axis's direction row is identical at every t (it is excluded from the rotation)
    v = g.pick_variant(4, _args("--dims", "7", "--oscillate", "tumble", "--lock", "0"), {})
    d0 = g._tumbled_directions(v, 0.0)
    for t in (0.13, 0.5, 0.87):
        dt = g._tumbled_directions(v, t)
        assert d0[0] == dt[0]                            # axis 0 never moves


def test_tumble_mode_rejected_without_tumble_transform():
    # --tumble-mode / --tumble-lock only make sense for --transform tumble
    with pytest.raises(SystemExit):
        g.main(["--transform", "rotate", "--tumble-mode", "slide", "--no-video", "--count", "1"])
    with pytest.raises(SystemExit):
        g.main(["--transform", "drift", "--tumble-lock", "0", "--no-video", "--count", "1"])


# --------------------------------------------------------------------------
# --tumble-sequence (P3.5): explicit ORDERED / OVERLAPPING Givens-plane words
# --------------------------------------------------------------------------

def test_tumble_sequence_parses_ordered_word():
    # list order is the composition order; the optional 'xN' is a whole-turn count (default 1)
    assert g._parse_tumble_sequence("0-3,3-4x2,0-4", 5) == [(0, 3, 1), (3, 4, 2), (0, 4, 1)]
    # blank / whitespace tokens are tolerated
    assert g._parse_tumble_sequence(" 0-1 , , 2-3 ", 4) == [(0, 1, 1), (2, 3, 1)]
    assert g._parse_tumble_sequence("", 4) == []
    assert g._parse_tumble_sequence(None, 4) == []


def test_tumble_sequence_validation_errors():
    for bad in ("0-5",        # axis out of range for dims=5 (valid 0..4)
                "2-2",        # self-pair
                "0-3x0",      # non-positive turn count
                "0-3xz",      # non-integer turn count
                "0_3",        # malformed pair (no dash)
                "a-3"):       # non-integer axis
        with pytest.raises(SystemExit):
            g._parse_tumble_sequence(bad, 5)


def test_tumble_rownorm_factor_component_sizes():
    # the rigorous bound is sqrt(max connected-component size) of the plane graph
    mk = lambda planes: types.SimpleNamespace(tumble_planes=planes)
    assert g._tumble_rownorm_factor(mk([])) == 1.0                       # no rotation at all
    assert g._tumble_rownorm_factor(mk([(0, 1, 1)])) == pytest.approx(math.sqrt(2))
    # a disjoint word -> every plane its own size-2 component -> sqrt(2): byte-identical to the old
    # `coef *= sqrt(2)` shortcut, so the disjoint default's bound is unchanged.
    assert g._tumble_rownorm_factor(
        mk([(0, 3, 1), (1, 4, 1), (2, 5, 1)])) == pytest.approx(math.sqrt(2))
    # overlapping triangle {0,3,4} -> one size-3 component
    assert g._tumble_rownorm_factor(
        mk([(0, 3, 1), (3, 4, 1), (0, 4, 1)])) == pytest.approx(math.sqrt(3))
    # a chain 0-1-2-3 is one size-4 component -> 2
    assert g._tumble_rownorm_factor(
        mk([(0, 1, 1), (1, 2, 1), (2, 3, 1)])) == pytest.approx(2.0)
    # two components -> the larger one wins
    assert g._tumble_rownorm_factor(
        mk([(0, 1, 1), (1, 2, 1), (5, 6, 1)])) == pytest.approx(math.sqrt(3))


def test_tumble_sequence_wires_exact_planes_into_variant():
    v = g.pick_variant(7, _args("--dims", "6", "--oscillate", "tumble",
                                "--tumble-sequence", "0-3,3-4,0-4"), {})
    assert v.tumble_planes == [(0, 3, 1), (3, 4, 1), (0, 4, 1)]


def test_tumble_sequence_overrides_lock():
    # an explicit word is fully user-specified, so it wins over --tumble-lock (the locked axis
    # 0 appears in the word rather than being excluded).
    v = g.pick_variant(7, _args("--dims", "6", "--oscillate", "tumble", "--lock", "0",
                                "--tumble-sequence", "0-3,3-4,0-4"), {})
    assert v.tumble_planes == [(0, 3, 1), (3, 4, 1), (0, 4, 1)]
    assert any(0 in (i, j) for (i, j, _w) in v.tumble_planes)


def test_tumble_sequence_loop_is_seamless_and_starts_from_base():
    # whole-turn planes -> R(0)=R(1)=I regardless of order/overlap; frame 0 == the base gyroid
    v = g.pick_variant(7, _args("--dims", "6", "--oscillate", "tumble",
                                "--tumble-sequence", "0-3,3-4,0-4"), {})
    e0, e1 = g.field_expr(v, 0.0, "tumble"), g.field_expr(v, 1.0, "tumble")
    assert e0 == g.field_expr(v)                          # frame 0 = base gyroid
    for (x, y, z) in [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]:
        assert abs(_eval_expr(e0, x, y, z) - _eval_expr(e1, x, y, z)) < 1e-6


def test_tumble_sequence_overlap_is_order_dependent():
    # overlapping planes don't commute: swapping their order changes the mid-loop slice — a
    # reorientation path no disjoint (always-commuting) word can ever produce.
    va = g.pick_variant(7, _args("--dims", "5", "--oscillate", "tumble",
                                 "--tumble-sequence", "0-1,1-2"), {})
    vb = g.pick_variant(7, _args("--dims", "5", "--oscillate", "tumble",
                                 "--tumble-sequence", "1-2,0-1"), {})
    da, db = g._tumbled_directions(va, 0.3), g._tumbled_directions(vb, 0.3)
    assert max(abs(da[i][k] - db[i][k]) for i in da for k in range(3)) > 1e-3
    ea, eb = g.field_expr(va, 0.3, "tumble"), g.field_expr(vb, 0.3, "tumble")
    pts = [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]
    assert max(abs(_eval_expr(ea, *p) - _eval_expr(eb, *p)) for p in pts) > 1e-3


def test_tumble_disjoint_word_is_order_independent():
    # disjoint planes commute, so reordering them leaves the slice rotation identical — the
    # contrast that makes the overlapping case genuinely new motion.
    vc = g.pick_variant(7, _args("--dims", "6", "--oscillate", "tumble",
                                 "--tumble-sequence", "0-3,1-4"), {})
    vd = g.pick_variant(7, _args("--dims", "6", "--oscillate", "tumble",
                                 "--tumble-sequence", "1-4,0-3"), {})
    dc, dd = g._tumbled_directions(vc, 0.3), g._tumbled_directions(vd, 0.3)
    assert max(abs(dc[i][k] - dd[i][k]) for i in dc for k in range(3)) < 1e-12


def test_tumble_rownorm_bound_never_underestimates_overlapping():
    # the composed rotation of an overlapping word can grow a unit direction row past sqrt(2)
    # (so the general bound is genuinely needed), yet never past sqrt(component size).
    v = g.pick_variant(7, _args("--dims", "6", "--oscillate", "tumble",
                                "--tumble-sequence", "0-3,3-4,0-4"), {})
    factor = g._tumble_rownorm_factor(v)
    assert factor == pytest.approx(math.sqrt(3))
    peak = 0.0
    for k in range(201):
        d = g._tumbled_directions(v, k / 200.0)
        peak = max(peak, max(math.sqrt(sum(c * c for c in vec)) for vec in d.values()))
    assert peak > math.sqrt(2) + 1e-3            # exceeds the disjoint shortcut -> bound needed
    assert peak <= factor + 1e-9                 # but never the rigorous component-size bound


def test_tumble_default_word_still_reorients_and_keeps_sqrt2_bound():
    # plain --oscillate tumble (no sequence) keeps the tidy disjoint default: sqrt(2) bound, moves
    v = g.pick_variant(7, _args("--dims", "6", "--oscillate", "tumble"), {})
    assert g._tumble_rownorm_factor(v) == pytest.approx(math.sqrt(2))
    e0, eh = g.field_expr(v, 0.0, "tumble"), g.field_expr(v, 0.29, "tumble")
    pts = [(0.3, 1.1, -0.7), (2.0, -1.0, 0.5), (-1.5, 0.2, 2.2)]
    assert max(abs(_eval_expr(e0, *p) - _eval_expr(eh, *p)) for p in pts) > 1e-3


def test_nd_pov_path_honors_overlapping_sequence():
    # the P3.4 N-D POV path reads the EXACT per-frame matrix (not the static sqrt(2) shortcut),
    # so an overlapping word stays seamless and hole-safe with zero special-casing.
    v = _pv("--surface", "f_ellipsoid", "--dims", "5", "--oscillate", "tumble",
            "--tumble-sequence", "0-3,3-4,0-4")
    assert v.tumble_planes == [(0, 3, 1), (3, 4, 1), (0, 4, 1)]
    assert g._pov_use_nd(v, "tumble")
    assert g.field_expr(v, 0.0, "tumble") == g.field_expr(v, 1.0, "tumble")   # whole-turn seamless
    body = _auto_body(v, t=0.25, transform="tumble")
    gb = _grad_bound(body)
    assert gb > 0.0 and math.isfinite(gb)
    assert _emitted_clip_radius(body) > 0.0


# --------------------------------------------------------------------------
# --coupling / --pair (coupling-graph node + edge edits)
# --------------------------------------------------------------------------

def _pv(*argv, seed=1):
    """pick_variant with the same --pair wiring main() applies (parse the edge
    specs onto args.pair_on / args.pair_off, plus --axis locks)."""
    args = _args(*argv)
    on, off = set(), set()
    for s in args.pair:
        g.parse_pair_lock(s, on, off)
    args.pair_on = frozenset(on)
    args.pair_off = frozenset(off)
    args.surface = g.resolve_surface(getattr(args, "surface", "gyroid"))
    g.resolve_pov_param_locks(args)             # S4: extract/validate --lock NAME=VALUE pins
    g.resolve_oscillate(args)                   # S5: classify param swingers (and the motion grammar)
    locks = {}
    for s in args.axis:
        g.parse_axis_lock(s, locks)
    return g.pick_variant(seed, args, locks)


def test_coupling_cyclic_is_the_consecutive_ring():
    v = _pv("--dims", "6", "--oscillating", "6")
    assert g.coupling_pairs(v) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]


def test_coupling_all_is_every_unordered_pair():
    v = _pv("--dims", "6", "--oscillating", "6", "--coupling", "all")
    pairs = g.coupling_pairs(v)
    assert len(pairs) == 15                              # C(6,2)
    assert set(map(frozenset, pairs)) == {frozenset((i, j))
                                          for i in range(6) for j in range(i + 1, 6)}


def test_pair_off_deletes_one_edge():
    v = _pv("--dims", "6", "--oscillating", "6", "--coupling", "all", "--pair", "0,3:off")
    pairs = g.coupling_pairs(v)
    assert len(pairs) == 14
    assert frozenset((0, 3)) not in set(map(frozenset, pairs))


def test_pair_off_removes_term_from_field_expr():
    base = _pv("--dims", "6", "--oscillating", "6", "--coupling", "all")
    edit = _pv("--dims", "6", "--oscillating", "6", "--coupling", "all", "--pair", "0,3:off")
    # the edited field has exactly one fewer sin*cos term
    assert g.field_expr(edit).count("sin(") == g.field_expr(base).count("sin(") - 1


def test_pair_on_adds_chord_to_cyclic():
    v = _pv("--dims", "6", "--oscillating", "6", "--pair", "0,3:on")
    pairs = g.coupling_pairs(v)
    assert len(pairs) == 7                               # 6 cyclic + 1 chord
    assert (0, 3) in pairs                               # sin(lower)*cos(higher)


def test_pair_on_forces_endpoints_to_oscillate():
    # only 2 dims asked to oscillate, but an 'on' edge needs both its endpoints waving
    v = _pv("--dims", "8", "--oscillating", "2", "--pair", "5,6:on")
    assert 5 in v.oscillating and 6 in v.oscillating


def test_pair_on_of_existing_edge_is_noop():
    base = _pv("--dims", "6", "--oscillating", "6")
    same = _pv("--dims", "6", "--oscillating", "6", "--pair", "0,1:on")
    assert g.coupling_pairs(base) == g.coupling_pairs(same)


def test_pair_off_all_edges_gives_empty_field():
    # a 2-osc cyclic field is the two mirrored terms on edge {0,1}; turning it off empties it
    v = _pv("--dims", "3", "--oscillating", "2", "--axis", "0:on", "--axis", "1:on",
            "--pair", "0,1:off")
    assert g.coupling_pairs(v) == []
    assert g.field_expr(v) == "(0.0)"


def test_pair_edit_reflected_in_coupling_desc():
    v = _pv("--dims", "6", "--oscillating", "6", "--coupling", "all", "--pair", "0,3:off")
    assert "--pair edits -> 14 terms" in g.coupling_desc(v)


def test_pair_grad_bound_drops_when_edges_removed():
    # deleting an edge lowers the touched dims' degree, so the Lipschitz bound shrinks
    from loom import Clock, Cache

    def _grad(v):
        body = g.build_scene(v).emit(Clock(t=0.0), Cache())
        return float(re.search(r"max_gradient ([0-9.]+)", body).group(1))

    full = _grad(_pv("--dims", "6", "--oscillating", "6", "--coupling", "all"))
    cut = _grad(_pv("--dims", "6", "--oscillating", "6", "--coupling", "all",
                    "--pair", "0,3:off"))
    assert cut < full


def test_pair_bad_specs_raise():
    for bad in ("3:off", "3,4,5:off", "3,3:off", "3,x:off", "3,4:maybe", "-1,4:off"):
        with pytest.raises(argparse.ArgumentTypeError):
            g.parse_pair_lock(bad, set(), set())


def test_pair_on_off_conflict_raises():
    on, off = set(), set()
    g.parse_pair_lock("3,4:on", on, off)
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_pair_lock("3,4:off", on, off)
    # order-independent (comma endpoints are unordered)
    on, off = set(), set()
    g.parse_pair_lock("4,3:off", on, off)
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_pair_lock("3,4:on", on, off)


# ---------------------------------------------------------------------------
# --couple: spatial-coupling clusters (OSCILLATE_GRAMMAR.md §6)
# ---------------------------------------------------------------------------

def test_couple_disjoint_cyclic_rings():
    # two independent clusters -> a cyclic ring within each, concatenated in CLI order
    v = _pv("--dims", "6", "--couple", "0,1,2", "3,4")
    assert v.couple_clusters == (((0, 1, 2), "cyclic"), ((3, 4), "cyclic"))
    assert g.coupling_pairs(v) == [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3)]


def test_couple_per_cluster_full_scheme_tag():
    # a ':full' tag turns that cluster into a clique; the other keeps the default ring
    v = _pv("--dims", "6", "--couple", "0,1,2:full", "3,4")
    assert g.coupling_pairs(v) == [(0, 1), (0, 2), (1, 2), (3, 4), (4, 3)]


def test_couple_scheme_default_applies_to_all_clusters():
    v = _pv("--dims", "6", "--couple", "0,1,2", "3,4,5", "--couple-scheme", "full")
    assert v.couple_clusters == (((0, 1, 2), "full"), ((3, 4, 5), "full"))
    assert g.coupling_pairs(v) == [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]


def test_couple_forces_cluster_members_to_oscillate():
    # only 2 dims asked to oscillate, but every clustered dim must wave to couple
    v = _pv("--dims", "8", "--oscillating", "2", "--couple", "5,6")
    assert {5, 6} <= set(v.oscillating)


def test_couple_member_locked_off_conflicts():
    with pytest.raises(SystemExit):
        _pv("--dims", "6", "--couple", "0,1,2", "--axis", "1:off")


def test_couple_mutually_exclusive_with_coupling():
    with pytest.raises(SystemExit):
        _pv("--dims", "6", "--couple", "0,1", "--coupling", "all")


def test_couple_mutually_exclusive_with_pair():
    with pytest.raises(SystemExit):
        _pv("--dims", "6", "--couple", "0,1", "--pair", "2,3:on")


def test_couple_disjoint_clusters_enforced():
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_couple(["0,1", "1,2"])


def test_couple_bad_specs_raise():
    for bad in (["0,x"], ["0,-1"], ["0,1:maybe"], ["0,,1"], [""]):
        with pytest.raises(argparse.ArgumentTypeError):
            g.parse_couple(bad)


def test_couple_default_path_is_bit_identical():
    # no --couple -> couple_clusters empty, legacy coupling graph untouched
    v = _pv("--dims", "6", "--oscillating", "6")
    assert getattr(v, "couple_clusters", ()) == ()
    assert g.coupling_pairs(v) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]


def test_couple_reflected_in_coupling_desc():
    v = _pv("--dims", "6", "--couple", "0,1,2:full", "3,4")
    desc = g.coupling_desc(v)
    assert "couple" in desc and "{0,1,2}:clique" in desc and "{3,4}:ring" in desc


# ---------------------------------------------------------------------------
# value-lock spec grammar (V | LO-HI | A,B,C) and base polarity
# ---------------------------------------------------------------------------

def test_value_spec_parses_fixed_range_set():
    assert g._parse_value_spec("5", "--dims") == ("fixed", 5)
    assert g._parse_value_spec("4-8", "--dims") == ("range", 4, 8)
    assert g._parse_value_spec("4,6,8", "--dims") == ("set", [4, 6, 8])
    # float specs for --freq
    assert g._freq_spec("3.5") == ("fixed", 3.5)
    assert g._freq_spec("3-5")[0] == "range"
    assert g._freq_spec("3,5,7") == ("set", [3.0, 5.0, 7.0])


def test_value_spec_bad_and_out_of_bounds_raise():
    for bad in ("x", "4-", "-", "8-4", "4,x", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            g._dims_spec(bad)
    with pytest.raises(argparse.ArgumentTypeError):
        g._dims_spec("2")            # below the dims minimum (3)
    with pytest.raises(argparse.ArgumentTypeError):
        g._oscillating_spec("1")     # below the oscillating minimum (2)
    with pytest.raises(argparse.ArgumentTypeError):
        g._freq_spec("0")            # freq must be > 0


def test_dims_fixed_value_takes_no_rng_draw():
    # A fixed lock must not perturb the stream: forcing --dims 6 gives the same
    # variant as leaving it to be drawn as 6 from the range fallback for that seed.
    for seed in range(20):
        a = _args("--dims", "6")
        v = g.pick_variant(seed, a, {})
        assert v.dims == 6


def test_dims_range_lock_stays_in_span():
    for seed in range(30):
        v = g.pick_variant(seed, _args("--dims", "5-7"), {})
        assert 5 <= v.dims <= 7


def test_dims_set_lock_picks_only_listed_values():
    seen = set()
    for seed in range(40):
        seen.add(g.pick_variant(seed, _args("--dims", "4,6,8"), {}).dims)
    assert seen and seen <= {4, 6, 8}


def test_oscillating_set_lock_respected():
    for seed in range(30):
        v = g.pick_variant(seed, _args("--dims", "9", "--oscillating", "4,6"), {})
        assert len(v.oscillating) in (4, 6)


def test_harmonics_range_lock_respected():
    for seed in range(30):
        v = g.pick_variant(seed, _args("--dims", "9", "--oscillating", "7",
                                       "--harmonics", "2-4"), {})
        assert 2 <= len(v.harmonic_dims) <= 4


def test_freq_range_lock_stays_in_span():
    for seed in range(30):
        v = g.pick_variant(seed, _args("--freq", "3-5"), {})
        assert 3.0 <= v.freq <= 5.0


def test_freq_set_lock_picks_only_listed_values():
    seen = set()
    for seed in range(30):
        seen.add(round(g.pick_variant(seed, _args("--freq", "4,9"), {}).freq, 6))
    assert seen and seen <= {4.0, 9.0}


def test_oscillating_range_intersects_feasible_span():
    # asking for 5-99 oscillating dims in a D=6 field clamps to at most 6
    for seed in range(20):
        v = g.pick_variant(seed, _args("--dims", "6", "--oscillating", "5-99"), {})
        assert 5 <= len(v.oscillating) <= 6


def test_oscillating_set_with_no_feasible_value_raises():
    with pytest.raises(SystemExit):
        g.pick_variant(1, _args("--dims", "4", "--oscillating", "8,9"), {})


def test_axis_default_on_makes_every_unnamed_axis_oscillate():
    v = _pv("--dims", "6", "--axis-default", "on")
    assert sorted(v.oscillating) == [0, 1, 2, 3, 4, 5]


def test_axis_default_on_with_override_off():
    v = _pv("--dims", "5", "--axis-default", "on", "--axis", "2:off")
    assert 2 not in v.oscillating
    assert sorted(v.oscillating) == [0, 1, 3, 4]


def test_axis_default_off_with_override_on():
    v = _pv("--dims", "6", "--axis-default", "off", "--axis", "0:on", "--axis", "1:on")
    assert sorted(v.oscillating) == [0, 1]


def test_axis_default_off_alone_is_infeasible():
    with pytest.raises(SystemExit):
        _pv("--dims", "6", "--axis-default", "off")


def test_axis_default_random_is_unchanged_default():
    # the default polarity draws exactly as before (no --axis-default given)
    for seed in range(15):
        a = g.pick_variant(seed, _args("--dims", "6"), {})
        b = g.pick_variant(seed, _args("--dims", "6", "--axis-default", "random"), {})
        assert sorted(a.oscillating) == sorted(b.oscillating)


def test_coupling_none_is_empty_base_graph():
    v = _pv("--dims", "6", "--oscillating", "6", "--coupling", "none")
    assert g.coupling_pairs(v) == []
    assert g.field_expr(v) == "(0.0)"
    assert "none" in g.coupling_desc(v)


def test_coupling_none_built_up_with_pair_on():
    v = _pv("--dims", "6", "--coupling", "none", "--pair", "0,1:on", "--pair", "1,2:on")
    assert g.coupling_pairs(v) == [(0, 1), (1, 2)]


# --- surface selector (gyroid vs Schwarz P / primitive) --------------------

def test_surface_default_is_gyroid():
    assert _args().surface == "gyroid"
    assert _pv("--dims", "4").surface == "gyroid"


def test_surface_primitive_parsed():
    assert _args("--surface", "primitive").surface == "primitive"


def test_gyroid_field_is_pairwise_sincos():
    # default gyroid: one sin(u_a)*cos(u_b) per cyclic edge, no bare cos() terms.
    v = _pv("--dims", "3", "--oscillating", "3", "--phase0")
    expr = g.field_expr(v, 0.0, "drift")
    assert expr.count("sin(") == 3 and expr.count("cos(") == 3
    assert "sin(" in expr


def test_primitive_field_is_per_node_cos():
    # Schwarz P: one cos(u_d) per oscillating dim, and NO sin() at all.
    v = _pv("--surface", "primitive", "--dims", "3", "--oscillating", "3", "--phase0")
    expr = g.field_expr(v, 0.0, "drift")
    assert "sin(" not in expr                       # per-node cosines only
    assert expr.count("cos(") == 3                  # one per oscillating dim


def test_primitive_field_term_count_scales_with_oscillating():
    v = _pv("--surface", "primitive", "--dims", "6", "--oscillating", "5", "--phase0")
    expr = g.field_expr(v, 0.0, "drift")
    assert "sin(" not in expr and expr.count("cos(") == 5


def test_primitive_ignores_coupling_and_pair():
    # A per-node field must not read the coupling graph.  Mutate the SAME variant's edge
    # settings in place (so the RNG-drawn geometry is identical) and confirm the field is
    # unchanged -- i.e. coupling_pairs() plays no role for --surface primitive.
    import dataclasses
    v = _pv("--surface", "primitive", "--dims", "6", "--oscillating", "6", "--phase0")
    before = g.field_expr(v, 0.0, "drift")
    v2 = dataclasses.replace(v, coupling="all",
                             pair_on=frozenset([frozenset((0, 3))]))
    assert g.field_expr(v2, 0.0, "drift") == before
    # And a gyroid with the same edits DOES change, proving the knob is otherwise live.
    gyr = dataclasses.replace(v, surface="gyroid")
    gyr2 = dataclasses.replace(gyr, coupling="all")
    assert g.field_expr(gyr2, 0.0, "drift") != g.field_expr(gyr, 0.0, "drift")


def test_primitive_bloom_classic_is_schwarz_p():
    # bloom frame-0 subject for primitive is cos(fx)+cos(fy)+cos(fz), not the gyroid.
    v = _pv("--surface", "primitive", "--dims", "5", "--oscillate", "bloom")
    expr0 = g.field_expr(v, 0.0, "bloom")
    assert "sin(" not in expr0 and expr0.count("cos(") == 3


def test_gyroid_bloom_classic_is_gyroid():
    v = _pv("--dims", "5", "--oscillate", "bloom")
    expr0 = g.field_expr(v, 0.0, "bloom")
    assert expr0.count("sin(") == 3 and expr0.count("cos(") == 3


def test_primitive_grad_bound_is_positive_and_scales():
    # Per-node bound weighted = sum_d harmonic_d; a build should succeed and set a finite bound.
    v = _pv("--surface", "primitive", "--dims", "4", "--oscillating", "4")
    txt = _scene_body(v, transform="drift", material="gold")
    assert "max_gradient" in txt
    assert "cos(" in txt and "sin(" not in txt


def test_surface_desc_labels():
    prim = _pv("--surface", "primitive", "--dims", "4")
    gyr = _pv("--dims", "4")
    assert "Schwarz P" in g.surface_desc(prim)
    assert "gyroid" in g.surface_desc(gyr)


def test_primitive_coupling_desc_is_per_node():
    v = _pv("--surface", "primitive", "--dims", "5", "--oscillating", "4")
    assert "per-node" in g.coupling_desc(v)
    assert "4 cos" in g.coupling_desc(v)


# ---------------------------------------------------------------------------
# POV surface selection (P3.3 slice S1): --surface accepts any POV builtin and
# emits it as a solid isosurface at its default shape params.
# ---------------------------------------------------------------------------

def test_resolve_surface_passes_native_families():
    assert g.resolve_surface("gyroid") == "gyroid"
    assert g.resolve_surface("primitive") == "primitive"


def test_resolve_surface_resolves_tpms_alias():
    assert g.resolve_surface("schwarz_p") == "primitive"


def test_resolve_surface_passes_pov_builtins():
    assert g.resolve_surface("f_sphere") == "f_sphere"
    assert g.resolve_surface("f_torus") == "f_torus"


def test_resolve_surface_rejects_catalog_only_tpms():
    for name in ("schwarz_d", "neovius"):
        try:
            g.resolve_surface(name)
        except SystemExit as e:
            assert "catalog-only" in str(e)
        else:
            raise AssertionError(f"{name} must raise (catalog-only, not renderable)")


def test_resolve_surface_rejects_unknown():
    try:
        g.resolve_surface("f_not_real")
    except SystemExit as e:
        assert "unknown surface" in str(e)
    else:
        raise AssertionError("an unknown surface must raise")


def test_is_pov_surface_only_true_for_pov():
    assert g._is_pov_surface("f_sphere")
    assert g._is_pov_surface("f_torus")
    assert not g._is_pov_surface("gyroid")
    assert not g._is_pov_surface("primitive")


def test_pov_default_values_match_authored_defaults():
    from loom import pov_params
    for name in ("f_sphere", "f_torus", "f_ellipsoid", "f_r"):
        want = tuple(d for _a, _de, d, _r in pov_params(name))
        assert g.pov_default_values(name) == want
    assert g.pov_default_values("f_r") == ()        # 0-param helper


def test_pov_call_expr_shape_and_arity():
    assert g._pov_call_expr("f_sphere", (1.0,)) == "f_sphere(x,y,z,1)"
    assert g._pov_call_expr("f_torus", (0.8, 0.25)) == "f_torus(x,y,z,0.8,0.25)"
    # a param-count mismatch is a programming error
    try:
        g._pov_call_expr("f_sphere", (1.0, 2.0))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong param count must raise")


def test_pick_variant_sets_pov_values_from_surface():
    v = _pv("--surface", "f_torus")
    assert v.surface == "f_torus"
    assert v.pov_values == g.pov_default_values("f_torus")
    # a TPMS carries no pov_values
    assert _pv("--surface", "gyroid").pov_values == ()


def test_pov_field_expr_is_the_solid_call():
    v = _pv("--surface", "f_sphere")
    expr = g.field_expr(v, 0.0, "drift")
    assert expr == "f_sphere(x,y,z,1)"
    # POV fields are static in S1: no transform/bloom machinery touches them
    assert g.field_expr(v, 0.5, "bloom") == expr
    assert g.field_expr(v, 0.5, "drift+rotate+tumble") == expr


def test_pov_scene_is_solid_no_abs_shell():
    # A POV solid renders the interior {field < thr}; no abs()-shell (that would carve a
    # thin sheet) and the field call appears verbatim.
    v = _pv("--surface", "f_sphere")
    body = _scene_body(v)
    assert "f_sphere(x,y,z,1)" in body
    assert "abs(f_sphere" not in body
    assert "max_gradient" in body


def test_pov_scene_uses_per_function_grad_bound():
    # f_sphere is an exact SDF (|grad| == 1); the table bound is emitted, not a
    # frequency-scaled gyroid bound.
    import re
    v = _pv("--surface", "f_sphere")
    body = _scene_body(v)
    m = re.search(r"max_gradient\s+([0-9.]+)", body)
    assert m and abs(float(m.group(1)) - 1.0) < 1e-9


def test_pov_unanalyzable_grad_bound_falls_back_to_default():
    # f_klein_bottle is not an analyzable algebraic field (the S2 bounder declines it) and it
    # is not in the cheap table -> the conservative default is emitted.
    import re
    v = _pv("--surface", "f_klein_bottle")
    body = _scene_body(v)
    m = re.search(r"max_gradient\s+([0-9.]+)", body)
    assert m and abs(float(m.group(1)) - g._POV_GRAD_DEFAULT) < 1e-9


# ---------------------------------------------------------------------------
# POV solid orientation (sign) + natural isolevel (level): a positive-inside
# builtin (f_heart) must be negated so ftrace's {field<0} fills the interior,
# not the exterior; f_ellipsoid's surface lives at level 1, not 0.
# ---------------------------------------------------------------------------

def test_pov_solid_meta_defaults_to_honest_passthrough():
    # an un-tabulated function falls back to (+1, 0): raw {f<0}, emitted unchanged
    assert g._pov_solid_meta("f_klein_bottle") == (1.0, 0.0)


def test_pov_solid_meta_known_orientations():
    assert g._pov_solid_meta("f_sphere") == (1.0, 0.0)
    assert g._pov_solid_meta("f_torus") == (1.0, 0.0)
    assert g._pov_solid_meta("f_ellipsoid") == (1.0, 1.0)
    assert g._pov_solid_meta("f_heart") == (-1.0, 0.0)
    assert g._pov_solid_meta("f_hunt_surface") == (-1.0, 0.0)


def test_pov_sphere_sign_positive_emits_field_unchanged():
    # sign=+1, level=0 -> the field call appears verbatim, not negated
    v = _pv("--surface", "f_sphere")
    body = _scene_body(v)
    assert "f_sphere(x,y,z,1)" in body
    assert "-(f_sphere" not in body and "-(( f_sphere" not in body


def test_pov_heart_sign_negative_emits_negated_field():
    # f_heart is positive-inside -> the emitted solid field must be negated so
    # ftrace's {field<0} fills the true interior (else it renders a crater)
    v = _pv("--surface", "f_heart")
    body = _scene_body(v)
    call = g._pov_call_expr("f_heart", v.pov_values)
    assert call in body
    assert f"-({call})" in body


def test_pov_ellipsoid_emits_level_one_shift():
    # f_ellipsoid is >=0 everywhere with the surface at level 1 -> subtract 1
    v = _pv("--surface", "f_ellipsoid")
    body = _scene_body(v)
    assert "f_ellipsoid(" in body
    # (expr)-(1) with sign=+1 (no outer negation)
    assert "-(1)" in body
    assert "-(f_ellipsoid" not in body


def test_pov_threshold_shifts_isolevel_on_top_of_level():
    # a --threshold adds to the natural level: f_ellipsoid at thr=0.2 -> (expr)-(1.2)
    v = _pv("--surface", "f_ellipsoid", "--threshold", "0.2")
    body = _scene_body(v)
    assert "-(1.2)" in body


# ---------------------------------------------------------------------------
# POV tight active-band gradient bound (P3.3 slice S2, Option B): the emitted
# max_gradient comes from loom.pov_grad's rigorous per-function bounder, not a
# hand-picked default.
# ---------------------------------------------------------------------------

def _emitted_grad(body):
    import re
    m = re.search(r"max_gradient\s+([0-9.]+)", body)
    assert m, "no max_gradient in emitted scene"
    return float(m.group(1))


def test_pov_grad_bound_matches_resolver_for_the_box():
    # the emitted max_gradient is exactly what _pov_grad_bound computes for the render box
    # (radius 1.3 * 1.05), tying emission to the S2 bounder with no drift.
    v = _pv("--surface", "f_heart")
    body = _scene_body(v)
    want = g._pov_grad_bound("f_heart", v.pov_values, 1.3 * 1.05)
    # the emitted number is fmt()-rounded, so compare against the same formatting
    assert _emitted_grad(body) == float(g.fmt(want))


def test_pov_heart_grad_bound_is_tight_not_the_default():
    # f_heart's true |grad| ceiling is ~70 over this box: the bounder must emit that (well
    # above the 8.0 default, and far below a naive whole-box degree-6 blow-up).
    v = _pv("--surface", "f_heart")
    b = _emitted_grad(_scene_body(v))
    assert b > g._POV_GRAD_DEFAULT           # the old default would have punched holes
    assert 50.0 < b < 200.0


def test_pov_hunt_surface_needs_a_large_bound():
    # f_hunt_surface is degree 6 with big coefficients: its ceiling is ~1e4, so the 8.0 default
    # would be a catastrophic under-estimate.  The bounder catches this.
    v = _pv("--surface", "f_hunt_surface")
    b = _emitted_grad(_scene_body(v))
    assert b > 1000.0


def test_pov_sphere_grad_bound_is_analytic_one():
    # the SDF-like primitives get their exact analytic bound, no interval machinery.
    v = _pv("--surface", "f_sphere")
    assert abs(_emitted_grad(_scene_body(v)) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# POV container auto-sizing + --shell (P3.3 slice S3): the container fits the
# surface's own bounding box (no explicit --radius needed); an explicit --radius
# overrides / clips; genuinely-thin surfaces and --shell render hollow.
# ---------------------------------------------------------------------------

def _auto_body(v, **kw):
    """Scene body with NO explicit radius, so the POV path auto-sizes the container."""
    from loom import Clock, Cache
    return g.build_scene(v, res=(32, 32), **kw).emit(Clock(t=0.0), Cache())


def _emitted_box(body):
    import re
    m = re.search(r"contained_by\s*\{\s*min\s+\S+\s+\S+\s+\S+\s+max\s+([0-9.]+)", body)
    assert m, "no contained_by max in emitted scene"
    return float(m.group(1))


def _emitted_clip_radius(body):
    import re
    m = re.search(r"sphere\s*\{\s*center\s+0\s+0\s+0\s+radius\s+([0-9.]+)", body)
    assert m, "no clip sphere radius in emitted scene"
    return float(m.group(1))


def test_pov_container_auto_sizes_to_larger_surface():
    # f_hunt_surface's surface sits at radius ~3.67 -> at the old fixed 1.3 it rendered as a
    # clipped disk.  With no --radius the container must auto-grow to fit it.
    v = _pv("--surface", "f_hunt_surface")
    body = _auto_body(v)
    assert _emitted_clip_radius(body) > 3.0           # derived, not the 1.3 default
    assert _emitted_box(body) > 3.0


def test_pov_container_explicit_radius_overrides_autosize():
    # an explicit --radius wins over the derived bbox (here it *clips* the big hunt surface)
    v = _pv("--surface", "f_hunt_surface", "--radius", "1.5")
    body = _auto_body(v, radius=1.5)
    assert abs(_emitted_clip_radius(body) - 1.5) < 1e-9
    assert abs(_emitted_box(body) - 1.5 * 1.05) < 1e-6


def test_pov_container_resolver_explicit_wins():
    # the resolver returns the explicit arg verbatim, ignoring the bbox
    assert g._pov_container_radius("f_hunt_surface", (1.0,), 0.0, 2.0) == 2.0


def test_pov_container_resolver_fits_ellipsoid_long_axis():
    # f_ellipsoid (semi-axes 1/P): params (1,2,0.5) put the surface out to z=2 -> the container
    # must reach ~2 (padded), not the 1.3 default that would clip the long lobe.
    r = g._pov_container_radius("f_ellipsoid", (1.0, 2.0, 0.5), 1.0, None)
    assert 2.0 <= r <= 2.0 * g._POV_CONTAINER_PAD + 0.2


def test_pov_container_unbounded_falls_back_to_default():
    # f_kummer_surface_v1's surface runs off to the search boundary (unbounded sheets) -> the
    # resolver can't auto-size and returns the conservative default (user clips with --radius).
    r = g._pov_container_radius("f_kummer_surface_v1", (1.0,), 0.0, None)
    assert abs(r - g._POV_DEFAULT_RADIUS) < 1e-9


def test_pov_container_no_field_falls_back_to_default():
    # a builtin with no transcribed field (f_klein_bottle) can't be sized -> default
    r = g._pov_container_radius("f_klein_bottle", (1.0,), 0.0, None)
    assert abs(r - g._POV_DEFAULT_RADIUS) < 1e-9


def test_pov_solid_by_default_has_no_shell():
    # a solid POV shape fills its interior: no abs() around the field
    v = _pv("--surface", "f_sphere")
    body = _auto_body(v)
    assert "abs(" not in body


def test_pov_shell_flag_makes_any_shape_hollow():
    # --shell carves a thin shell (abs(sheet) - thickness) out of an otherwise-solid shape
    v = _pv("--surface", "f_sphere")
    body = _auto_body(v, shell=True)
    assert "abs(" in body
    assert f"-({g.fmt(v.thickness)})" in body


def test_pov_shell_thickness_tracks_thickness_flag():
    v = _pv("--surface", "f_sphere", "--thickness", "0.3")
    body = _auto_body(v, shell=True)
    assert "abs(" in body and "-(0.3)" in body


def test_pov_thin_surface_renders_hollow_by_default():
    # a genuinely-thin surface is shelled even without --shell
    assert g._pov_renders_thin("f_klein_bottle")
    v = _pv("--surface", "f_klein_bottle")
    body = _auto_body(v)
    assert "abs(" in body


def test_pov_renders_thin_predicate():
    assert g._pov_renders_thin("f_klein_bottle")
    assert g._pov_renders_thin("f_enneper")
    assert g._pov_renders_thin("f_something_2d")      # *_2d suffix -> thin
    assert not g._pov_renders_thin("f_sphere")
    assert not g._pov_renders_thin("f_heart")


def test_shell_flag_defaults_off_and_tpms_ignores_it():
    # --shell defaults off; and it never changes a TPMS render (which always has its own shell)
    assert _args().shell is False
    v = g.pick_variant(5, _args("--dims", "6"), {})
    solid_default = _scene_body(v)
    # TPMS already emits abs()-shells; passing shell=True doesn't alter its structure
    assert _scene_body(v, shell=True) == solid_default


# ---------------------------------------------------------------------------
# POV shape-param value pins (P3.3 slice S4): --lock NAME=VALUE fixes one of a
# POV surface's named shape params, overriding its authored default.
# ---------------------------------------------------------------------------

def _resolved_args(*argv):
    """argparse args wired exactly as main() does for the --lock pipeline: resolve the surface,
    extract/validate the POV param pins, THEN run the motion-grammar parse on what's left.  The
    ordering matters -- resolve_oscillate() rejects a leftover 'NAME=VALUE' token -- so running it
    here is what catches an ordering regression (a plain resolve_pov_param_locks call would not)."""
    args = _args(*argv)
    args.surface = g.resolve_surface(getattr(args, "surface", "gyroid"))
    g.resolve_pov_param_locks(args)
    g.resolve_oscillate(args)
    return args


def test_lock_pins_pov_param_value():
    v = _pv("--surface", "f_torus", "--lock", "minor=0.5")
    assert v.pov_values == (1.0, 0.5)                 # default minor 0.25 overridden
    # the un-pinned param keeps its default
    assert v.pov_values[0] == g.pov_default_values("f_torus")[0]


def test_lock_pins_multiple_params_space_separated():
    v = _pv("--surface", "f_ellipsoid", "--lock", "rx=2", "rz=0.5")
    assert v.pov_values == (2.0, 1.0, 0.5)


def test_lock_param_value_is_an_expression():
    v = _pv("--surface", "f_torus", "--lock", "minor=1/4")
    assert v.pov_values == (1.0, 0.25)


def test_lock_param_coexists_with_transform():
    # a POV param pin is orthogonal to the motion grammar, so it doesn't trip the
    # --transform/--oscillate mutual-exclusion
    v = _pv("--surface", "f_ellipsoid", "--transform", "tumble", "--lock", "ry=3")
    assert v.pov_values == (1.0, 3.0, 1.0)


def test_lock_unknown_param_errors():
    with pytest.raises(SystemExit):
        _pv("--surface", "f_torus", "--lock", "bogus=1")


def test_lock_param_on_non_pov_surface_errors():
    with pytest.raises(SystemExit):
        _pv("--surface", "gyroid", "--lock", "major=1")


def test_lock_malformed_pin_errors():
    with pytest.raises(SystemExit):
        _pv("--surface", "f_torus", "--lock", "minor=")      # missing value
    with pytest.raises(SystemExit):
        _pv("--surface", "f_torus", "--lock", "=2")          # missing name


def test_lock_out_of_range_value_is_honored_with_warning():
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v = _pv("--surface", "f_torus", "--lock", "minor=9")  # range [0.02, 2]
    assert v.pov_values == (1.0, 9.0)                 # honored (user intent respected)
    assert "outside" in buf.getvalue() and "range" in buf.getvalue()


def test_resolve_splits_pins_from_motion_tokens():
    # a mixed --lock keeps the dim-lock token ('1') in args.lock and pulls the NAME=VALUE pin
    # out; the split then survives resolve_oscillate (which needs a real motion -> --oscillate
    # tumble, and 1 is a dim held out of the tumble rotation).
    args = _resolved_args("--surface", "f_torus", "--oscillate", "tumble",
                          "--lock", "minor=0.5", "1")
    assert args.pov_param_locks == {"minor": 0.5}
    assert args.lock == ["1"]


def test_resolve_pins_only_lock_collapses_to_none():
    # a --lock carrying ONLY pins must not engage the motion grammar (args.lock -> None)
    args = _resolved_args("--surface", "f_ellipsoid", "--lock", "rx=2")
    assert args.pov_param_locks == {"rx": 2.0}
    assert args.lock is None


def test_pinned_param_flows_into_emitted_call():
    v = _pv("--surface", "f_torus", "--lock", "minor=0.5")
    body = _auto_body(v)
    assert "f_torus(x,y,z,1,0.5)" in body


def test_pinned_param_resizes_autosized_container():
    # a longer ellipsoid semi-axis (small P -> 1/P big) must grow the auto-sized container
    v_default = _pv("--surface", "f_ellipsoid")
    v_long = _pv("--surface", "f_ellipsoid", "--lock", "rz=0.3")   # z semi-axis ~3.3
    assert _emitted_clip_radius(_auto_body(v_long)) > _emitted_clip_radius(_auto_body(v_default)) + 1.0


def test_no_pins_leaves_pov_values_at_defaults():
    v = _pv("--surface", "f_torus")
    assert v.pov_values == g.pov_default_values("f_torus")


def test_pin_survives_the_motion_grammar_parse():
    # regression: main() strips NAME=VALUE pins in resolve_pov_param_locks() BEFORE
    # resolve_oscillate() parses the rest of --lock through the motion/axis grammar, which
    # rejects a '=' token.  A pins-only --lock must reach resolve_oscillate as None.
    args = _resolved_args("--surface", "f_torus", "--lock", "minor=0.4")
    assert args.pov_param_locks == {"minor": 0.4}
    assert args.lock is None                             # collapsed, so the parse saw no '='


# ---------------------------------------------------------------------------
# S5: POV shape params become --oscillate swinger axes (OSCILLATE_GRAMMAR.md sec 7)
# ---------------------------------------------------------------------------

def _emitted_pov_params(body, name):
    """The shape-param values in an emitted POV call ``name(x,y,z, p0, p1, ...)``."""
    import re
    m = re.search(re.escape(name) + r"\(x,y,z,([^)]*)\)", body)
    assert m, f"no {name}(...) call in emitted scene"
    return tuple(float(x) for x in m.group(1).split(","))


def test_oscillate_pov_param_is_recorded_as_a_swinger():
    v = _pv("--surface", "f_torus", "--oscillate", "minor")
    assert v.pov_swing == {"minor": 1.0}                 # bare axis => amp 1.0
    assert v.pov_values == g.pov_default_values("f_torus")   # base is untouched


def test_pov_param_swing_only_needs_no_winder_motion():
    # a param-only --oscillate carries no winder/dims motion but must not error "no motion
    # axes"; field_expr ignores transform for POV, so a benign 'drift' is named.
    args = _resolved_args("--surface", "f_torus", "--oscillate", "minor")
    assert args.pov_swing == {"minor": 1.0}
    assert "drift" in args.transform


def test_pov_param_swing_sweeps_over_the_loop():
    v = _pv("--surface", "f_torus", "--oscillate", "minor")
    p0 = _emitted_pov_params(_auto_body(v, t=0.0), "f_torus")
    pmid = _emitted_pov_params(_auto_body(v, t=0.5), "f_torus")
    assert p0[1] == pytest.approx(0.25)                  # base minor at the loop ends
    assert pmid[1] == pytest.approx(2.0)                 # amp=1 reaches the authored hi at peak
    assert p0[0] == pmid[0]                              # the un-swung major stays put


def test_pov_param_swing_loops_seamlessly():
    v = _pv("--surface", "f_torus", "--oscillate", "minor")
    p0 = _emitted_pov_params(_auto_body(v, t=0.0), "f_torus")
    p1 = _emitted_pov_params(_auto_body(v, t=1.0), "f_torus")
    assert p0 == pytest.approx(p1)                       # t=1 returns exactly to the base


def test_pov_param_swing_amplitude_scales():
    v = _pv("--surface", "f_torus", "--oscillate", "0.5*minor")
    pmid = _emitted_pov_params(_auto_body(v, t=0.5), "f_torus")
    assert pmid[1] == pytest.approx(0.25 + 0.5 * (2.0 - 0.25))   # halfway from base to hi


def test_pov_param_swing_container_recomputes_per_frame():
    v = _pv("--surface", "f_torus", "--oscillate", "minor")
    r0 = _emitted_clip_radius(_auto_body(v, t=0.0))
    rmid = _emitted_clip_radius(_auto_body(v, t=0.5))
    assert rmid > r0 + 0.5                               # the torus fattens => container grows


def test_pov_param_swing_negative_amp_sweeps_down():
    # amp<0 sweeps toward lo (a leading-dash amp is awkward on the CLI, so set it directly)
    v = _pv("--surface", "f_torus")
    v.pov_swing = {"minor": -1.0}
    pmid = _emitted_pov_params(_auto_body(v, t=0.5), "f_torus")
    lo = g.pov_params("f_torus")[1][3][0]
    assert pmid[1] == pytest.approx(lo)                  # reaches the authored lo at the peak


def test_pov_param_swing_over_driven_amp_clamps_to_range():
    v = _pv("--surface", "f_torus", "--oscillate", "2*minor")
    pmid = _emitted_pov_params(_auto_body(v, t=0.5), "f_torus")
    assert pmid[1] == pytest.approx(2.0)                 # clamped at hi despite amp=2


def test_pov_param_swing_on_non_pov_surface_errors():
    with pytest.raises(SystemExit):
        _pv("--surface", "gyroid", "--oscillate", "minor")


def test_pov_param_swing_unknown_param_hints_valid_names():
    with pytest.raises(SystemExit) as exc:
        _pv("--surface", "f_torus", "--oscillate", "bogus")
    assert "major" in str(exc.value) and "minor" in str(exc.value)


def test_no_pov_swing_leaves_values_static_across_the_loop():
    v = _pv("--surface", "f_torus")
    assert _emitted_pov_params(_auto_body(v, t=0.0), "f_torus") == \
           _emitted_pov_params(_auto_body(v, t=0.5), "f_torus")


def test_pov_param_swing_coexists_with_a_lock_pin():
    # pin major, swing minor: the pinned base holds while minor sweeps around its default
    v = _pv("--surface", "f_torus", "--lock", "major=1.6", "--oscillate", "minor")
    assert v.pov_values[0] == pytest.approx(1.6)
    assert v.pov_swing == {"minor": 1.0}
    pmid = _emitted_pov_params(_auto_body(v, t=0.5), "f_torus")
    assert pmid[0] == pytest.approx(1.6)                 # major un-swung at its pin
    assert pmid[1] == pytest.approx(2.0)                 # minor at hi


# ---------------------------------------------------------------------------
# S7: --param-default random draws UNSPECIFIED POV shape params (roadmap P3.3)
# ---------------------------------------------------------------------------

def test_param_default_default_keeps_the_authored_shape():
    # the ordinary path: every variant shares the one authored default shape
    for seed in (1, 7, 99):
        v = _pv("--surface", "f_torus", "--param-default", "default", seed=seed)
        assert v.pov_values == pytest.approx((1.0, 0.25))


def test_param_default_random_varies_across_seeds():
    seen = {_pv("--surface", "f_torus", "--param-default", "random", seed=s).pov_values
            for s in range(8)}
    assert len(seen) > 1                                  # a POV batch finally has variety


def test_param_default_random_is_reproducible_for_a_seed():
    a = _pv("--surface", "f_torus", "--param-default", "random", seed=42)
    b = _pv("--surface", "f_torus", "--param-default", "random", seed=42)
    assert a.pov_values == b.pov_values


def test_param_default_random_stays_within_each_range():
    ranges = [rng for _ax, _d, _def, rng in g.pov_params("f_torus")]
    for s in range(12):
        vals = _pv("--surface", "f_torus", "--param-default", "random", seed=s).pov_values
        for val, (lo, hi) in zip(vals, ranges):
            assert lo <= val <= hi


def test_param_default_random_respects_a_lock_pin():
    # a pinned param opts out of the draw; the rest still vary
    for s in range(6):
        v = _pv("--surface", "f_torus", "--param-default", "random",
                "--lock", "major=1.6", seed=s)
        assert v.pov_values[0] == pytest.approx(1.6)     # major held at its pin


def test_param_default_random_respects_a_swinger():
    # a swung param keeps its authored base (the swing animates around it); only
    # the un-swung, un-pinned params get the random draw
    for s in range(6):
        v = _pv("--surface", "f_torus", "--param-default", "random",
                "--oscillate", "minor", seed=s)
        assert v.pov_values[1] == pytest.approx(0.25)    # minor base un-drawn
        assert v.pov_swing == {"minor": 1.0}


def test_param_default_random_no_effect_on_a_tpms():
    # a periodic surface has no pov_values, so the flag is a no-op
    a = _pv("--dims", "6", "--param-default", "random", seed=3)
    assert a.pov_values == ()


# ---------------------------------------------------------------------------
# S6: affine N-D remap of a POV surface under an explicit slice motion (P3.3)
# ---------------------------------------------------------------------------

def _grad_bound(body):
    m = re.search(r"max_gradient\s+([0-9.]+)", body)
    assert m, "no max_gradient in emitted scene"
    return float(m.group(1))


def test_pov_static_by_default_leaves_coords_unremapped():
    # a plain POV render carries no explicit motion: the call stays the literal f(x,y,z)
    v = _pv("--surface", "f_torus")
    assert v.pov_motion is False
    assert g.field_expr(v, 0.0, "drift") == "f_torus(x,y,z,1,0.25)"
    assert g.field_expr(v, 0.3, "drift") == "f_torus(x,y,z,1,0.25)"


def test_pov_param_swing_alone_does_not_trigger_the_affine():
    # an --oscillate that names only a shape param is not a slice motion
    v = _pv("--surface", "f_torus", "--oscillate", "minor")
    assert v.pov_motion is False


def test_pov_tumble_sets_motion_and_remaps_mid_loop():
    v = _pv("--surface", "f_torus", "--dims", "5", "--oscillate", "tumble")
    assert v.pov_motion is True
    mid = g.field_expr(v, 0.25, "tumble")
    assert mid != g.field_expr(v, 0.0, "tumble")         # the slice has turned
    assert mid.startswith("f_torus(")                    # same surface
    assert mid.endswith(",1,0.25)")                      # major/minor params preserved


def test_pov_tumble_loops_seamlessly():
    v = _pv("--surface", "f_torus", "--dims", "5", "--oscillate", "tumble")
    assert g.field_expr(v, 0.0, "tumble") == g.field_expr(v, 1.0, "tumble")


def test_pov_rotate_loops_seamlessly():
    v = _pv("--surface", "f_torus", "--dims", "5", "--oscillate", "rotate")
    assert g.field_expr(v, 0.0, "rotate") == g.field_expr(v, 1.0, "rotate")


def test_pov_drift_is_deliberately_not_seamless():
    # a non-periodic POV shape linearly panned by drift does NOT return at t=1 (documented)
    v = _pv("--surface", "f_torus", "--dims", "5", "--oscillate", "drift")
    assert v.pov_motion is True
    assert g.field_expr(v, 0.0, "drift") != g.field_expr(v, 1.0, "drift")


def test_pov_affine_is_identity_at_loop_ends():
    v = _pv("--surface", "f_torus", "--dims", "5", "--oscillate", "tumble")
    for t in (0.0, 1.0):
        M, b = g._pov_affine(v, t, "tumble")
        smin, smax = g._mat3_singular_extremes(M)
        assert smin == pytest.approx(1.0, abs=1e-6)
        assert smax == pytest.approx(1.0, abs=1e-6)
        assert b == pytest.approx([0.0, 0.0, 0.0])


def test_pov_no_motion_affine_is_the_identity():
    v = _pv("--surface", "f_torus")                       # pov_motion False
    M, b = g._pov_affine(v, 0.4, "drift")
    assert M == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert b == [0.0, 0.0, 0.0]


def test_mat3_singular_extremes_identity():
    assert g._mat3_singular_extremes([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == pytest.approx((1.0, 1.0))


def test_mat3_singular_extremes_diagonal_scale():
    smin, smax = g._mat3_singular_extremes([[0.5, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
    assert smin == pytest.approx(0.5)
    assert smax == pytest.approx(2.0)


def test_mat3_singular_extremes_rotation_is_an_isometry():
    c, s = math.cos(0.7), math.sin(0.7)
    R = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    smin, smax = g._mat3_singular_extremes(R)
    assert smin == pytest.approx(1.0)
    assert smax == pytest.approx(1.0)


def test_pov_tumble_inflates_grad_bound_mid_loop():
    # the emitted field is f(M.p), so |grad| <= sigma_max(M)*|grad f|; a torus's base bound is 1
    v = _pv("--surface", "f_torus", "--dims", "5", "--oscillate", "tumble")
    base = _grad_bound(_auto_body(v, t=0.0, transform="tumble"))
    mid = _grad_bound(_auto_body(v, t=0.25, transform="tumble"))
    assert base == pytest.approx(1.0)                     # identity at the loop end
    assert mid > base                                     # sigma_max ~ sqrt(2) > 1


def test_pov_tumble_grows_container_mid_loop():
    v = _pv("--surface", "f_torus", "--dims", "5", "--oscillate", "tumble")
    base = _emitted_clip_radius(_auto_body(v, t=0.0, transform="tumble"))
    mid = _emitted_clip_radius(_auto_body(v, t=0.25, transform="tumble"))
    assert mid > base                                     # 1/sigma_min > 1 as an axis foreshortens


def test_pov_explicit_radius_is_not_expanded_by_motion():
    v = _pv("--surface", "f_torus", "--dims", "5", "--oscillate", "tumble", "--radius", "2.0")
    r0 = _emitted_clip_radius(_auto_body(v, t=0.0, transform="tumble", radius=2.0))
    rmid = _emitted_clip_radius(_auto_body(v, t=0.25, transform="tumble", radius=2.0))
    assert r0 == pytest.approx(2.0)
    assert rmid == pytest.approx(2.0)                     # explicit --radius clips, no auto-grow


def test_pov_motion_via_explicit_transform_flag():
    # the legacy --transform tumble also engages the affine (not only --oscillate)
    v = _pv("--surface", "f_torus", "--dims", "5", "--transform", "tumble")
    assert v.pov_motion is True


def test_tpms_ignores_pov_motion_flag():
    # a periodic surface animates via its own drift/tumble machinery, not the POV affine
    v = _pv("--dims", "6", "--oscillate", "tumble")
    # field is the gyroid sum, unaffected by _pov_affine (which only runs on POV surfaces)
    assert "f_torus" not in g.field_expr(v, 0.25, "tumble")


# ---------------------------------------------------------------------------
# P3.4: true-N-D forms for the POV_ND_GENERALIZABLE builtins.  Under `tumble`
# with a hidden dim (dims>3) an nd_pov surface renders as an honest D-coordinate
# field F(A.p + c) (loom.pov_nd), which reduces to the base f_* exactly at the
# loop ends but folds the extra axes in between — motion no affine remap can make.
# ---------------------------------------------------------------------------

# Reference: a direct Python port of the C bodies in src/pov_functions.h (the ground
# truth the N-D forms must reduce to at N=3).  Kept minimal — only the nine funcs.
def _pov_c_ref(name, x, y, z, P):
    import math as _m
    if name == "f_sphere":
        return -P[0] + _m.sqrt(x * x + y * y + z * z)
    if name == "f_ellipsoid":
        return _m.sqrt(x * x * P[0] ** 2 + y * y * P[1] ** 2 + z * z * P[2] ** 2)
    if name == "f_paraboloid":
        return min(10., max(P[0] * (-(x * x - y + z * z)), -10.))
    if name == "f_quartic_paraboloid":
        return min(10., max(P[0] * (-(x ** 4 + z ** 4 - y)), -10.))
    if name == "f_ovals_of_cassini":
        r2 = x * x + y * y + z * z + P[1] ** 2
        r = -(r2 * r2 - P[3] * P[1] ** 2 * (x * x + z * z) - P[2] ** 2)
        return min(10., max(P[0] * r, -10.))
    if name in ("f_isect_ellipsoids", "f_cross_ellipsoids"):
        x2, y2, z2 = x * x, y * y, z * z
        t = [_m.exp(-(x2 * P[0] + y2 * P[0] + z2) * P[1]),
             _m.exp(-(x2 * P[0] + y2 + z2 * P[0]) * P[1]),
             _m.exp(-(x2 + y2 * P[0] + z2 * P[0]) * P[1])]
        r = min(t) if name == "f_isect_ellipsoids" else max(t)
        return P[3] - r * P[2]
    if name == "f_poly4":
        y2 = y * y
        temp = P[0] + P[1] * y + P[2] * y2 + P[3] * y2 * y + P[4] * y2 * y2
        temp = max(temp, -5.)
        return -temp + _m.sqrt(x * x + z * z)
    if name == "f_superellipsoid":
        p, n = 2 / P[0], 1 / P[1]
        return 1 - ((abs(x) ** p + abs(y) ** p) ** (P[0] * n) + abs(z) ** (2 * n)) ** (P[1] * .5)
    raise AssertionError(name)


def _eval_pov_expr(expr, **vars):
    """Evaluate an emitted N-D field expression (valid Python once the math funcs are bound)."""
    import math as _m

    def _clamp(v, lo, hi):
        return min(hi, max(v, lo))
    ns = {"sqrt": _m.sqrt, "exp": _m.exp, "abs": abs, "pow": pow,
          "min": min, "max": max, "clamp": _clamp}
    ns.update(vars)
    return eval(expr, {"__builtins__": {}}, ns)


_ND_SAMPLE_VALUES = {
    "f_sphere": (1.0,), "f_ellipsoid": (1.0, 1.5, 0.7), "f_paraboloid": (1.0,),
    "f_quartic_paraboloid": (1.0,), "f_ovals_of_cassini": (1.0, 0.6, 1.1, 2.0),
    "f_isect_ellipsoids": (1.3, 1.0, 1.0, 1.0), "f_cross_ellipsoids": (1.3, 1.0, 1.0, 1.0),
    "f_poly4": (1.0, 0.3, -0.5, 0.1, 0.2), "f_superellipsoid": (2.0, 2.5),
}


@pytest.mark.parametrize("name", sorted(g.POV_ND_GENERALIZABLE))
def test_nd_field_eval_matches_c_body_at_n3(name):
    # the N-D generalization must reduce to the exact POV builtin when D=3.
    from loom import nd_field_eval
    P = _ND_SAMPLE_VALUES[name]
    import random
    rng = random.Random(hash(name) & 0xffff)
    for _ in range(400):
        x, y, z = (rng.uniform(-1.2, 1.2) for _ in range(3))
        got = nd_field_eval(name, [x, y, z], P)
        want = _pov_c_ref(name, x, y, z, P)
        assert abs(got - want) < 1e-9


@pytest.mark.parametrize("name", sorted(g.POV_ND_GENERALIZABLE))
def test_nd_field_expr_matches_eval(name):
    # the emitted ftsl string is numerically identical to the reference evaluator, at D>3 too.
    from loom import nd_field_expr, nd_field_eval
    P = _ND_SAMPLE_VALUES[name]
    import random
    rng = random.Random((hash(name) ^ 0x5555) & 0xffff)
    for D in (3, 5):
        coords = [f"c{i}" for i in range(D)]
        expr = nd_field_expr(name, coords, P)
        for _ in range(150):
            xi = [rng.uniform(-1.1, 1.1) for _ in range(D)]
            got = _eval_pov_expr(expr, **{f"c{i}": xi[i] for i in range(D)})
            assert abs(got - nd_field_eval(name, xi, P)) < 1e-9


@pytest.mark.parametrize("name", sorted(g.POV_ND_GENERALIZABLE))
def test_nd_grad_bound_never_underestimates(name):
    # a rigorous marcher needs |grad_xi F| <= bound over |xi_i| <= Xi (over-estimate is safe).
    from loom import nd_field_eval, nd_grad_bound_xi
    P = _ND_SAMPLE_VALUES[name]
    X, h = 1.0, 1e-6
    import random
    rng = random.Random((hash(name) ^ 0x1234) & 0xffff)
    for D in (3, 4, 5):
        bound = nd_grad_bound_xi(name, P, X, D)
        if bound is None:                       # f_superellipsoid opts out (non-Lipschitz corners)
            assert name == "f_superellipsoid"
            continue
        for _ in range(500):
            xi = [rng.uniform(-X, X) for _ in range(D)]
            grad = []
            for i in range(D):
                a, b = list(xi), list(xi)
                a[i] += h; b[i] -= h
                grad.append((nd_field_eval(name, a, P) - nd_field_eval(name, b, P)) / (2 * h))
            gm = math.sqrt(sum(c * c for c in grad))
            assert gm <= bound * 1.0001


def test_nd_field_reduces_to_base_at_loop_start():
    # at t=0 the N-D embedding is the identity (A_i=e_i, hidden rows 0, c=0), so the emitted
    # field equals the base f_* builtin numerically for every sample point.
    from loom import nd_field_eval
    v = _pv("--surface", "f_ellipsoid", "--dims", "6", "--oscillate", "tumble")
    assert g._pov_use_nd(v, "tumble")
    expr = g.field_expr(v, 0.0, "tumble")
    P = v.pov_values
    for (x, y, z) in [(0.3, 1.1, -0.7), (0.9, -0.4, 0.5), (-1.1, 0.2, 0.8)]:
        got = _eval_pov_expr(expr, x=x, y=y, z=z)
        assert abs(got - _pov_c_ref("f_ellipsoid", x, y, z, P)) < 1e-9


def test_nd_field_loops_seamlessly():
    # tumble is whole-turn, so t=0 and t=1 embeddings coincide -> the emitted field matches.
    v = _pv("--surface", "f_sphere", "--dims", "5", "--oscillate", "tumble")
    assert g.field_expr(v, 0.0, "tumble") == g.field_expr(v, 1.0, "tumble")


def test_nd_field_moves_mid_loop_and_is_not_an_affine_call():
    # mid-loop the field is an honest expanded N-D form (not an f_name(...) affine remap) and
    # differs from the loop-start field: motion no 3x3 remap of f(x,y,z) can produce.
    v = _pv("--surface", "f_sphere", "--dims", "5", "--oscillate", "tumble")
    start = g.field_expr(v, 0.0, "tumble")
    mid = g.field_expr(v, 0.25, "tumble")
    assert mid != start
    assert not mid.startswith("f_sphere(")      # expanded field, not a builtin call
    assert "f_sphere(" not in mid


def test_nd_no_motion_stays_clean_builtin_call():
    # dims>3 without motion keeps the exact static f(x,y,z) call (the nd path is motion-gated).
    v = _pv("--surface", "f_sphere", "--dims", "6")
    assert not g._pov_use_nd(v, "tumble")
    assert g.field_expr(v, 0.0, "tumble") == "f_sphere(x,y,z,1)"


def test_nd_path_requires_dims_gt_3():
    # at D=3 there is no hidden dim to fold, so tumble stays on the S6 affine (f_name call).
    v = _pv("--surface", "f_sphere", "--dims", "3", "--oscillate", "tumble")
    assert not g._pov_use_nd(v, "tumble")
    assert g.field_expr(v, 0.25, "tumble").startswith("f_sphere(")


def test_nd_path_requires_generalizable_surface():
    # f_torus is affine_pov: even at dims>3 under tumble it stays the S6 affine f_torus call.
    v = _pv("--surface", "f_torus", "--dims", "5", "--oscillate", "tumble")
    assert not g._pov_use_nd(v, "tumble")
    assert g.field_expr(v, 0.25, "tumble").startswith("f_torus(")


def test_nd_path_requires_tumble():
    # rotate/drift alone (no tumble) can't fold a hidden dim in, so they keep the affine path.
    v = _pv("--surface", "f_sphere", "--dims", "5", "--oscillate", "rotate")
    assert not g._pov_use_nd(v, "rotate")
    assert g.field_expr(v, 0.25, "rotate").startswith("f_sphere(")


def test_nd_scene_emits_finite_grad_bound_and_container():
    # the auto-sized N-D scene is hole-safe: a finite max_gradient and a positive clip radius.
    v = _pv("--surface", "f_ellipsoid", "--dims", "5", "--oscillate", "tumble")
    body = _auto_body(v, t=0.25, transform="tumble")
    gb = _grad_bound(body)
    assert gb > 0.0 and math.isfinite(gb)
    assert _emitted_clip_radius(body) > 0.0


def test_nd_superellipsoid_falls_back_to_default_grad_bound():
    # f_superellipsoid has no Lipschitz bound (nd_grad_bound_xi -> None); the scene still emits a
    # finite bound via the per-function fallback rather than crashing or emitting None.
    v = _pv("--surface", "f_superellipsoid", "--dims", "5", "--oscillate", "tumble")
    assert g._pov_use_nd(v, "tumble")
    body = _auto_body(v, t=0.25, transform="tumble")
    gb = _grad_bound(body)
    assert gb > 0.0 and math.isfinite(gb)


def test_matn3_singular_extremes_matches_mat3_on_square():
    # for a 3-row matrix the D×3 routine agrees with the dedicated 3×3 one.
    M = [[0.5, 0.1, 0.0], [0.2, 1.3, -0.4], [0.0, 0.3, 0.9]]
    assert g._matn3_singular_extremes(M) == pytest.approx(g._mat3_singular_extremes(M))


def test_matn3_extra_zero_rows_do_not_change_singular_values():
    # padding a 3×3 with all-zero hidden rows leaves A^T A (and thus the singular values) unchanged.
    M = [[0.5, 0.1, 0.0], [0.2, 1.3, -0.4], [0.0, 0.3, 0.9]]
    padded = M + [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert g._matn3_singular_extremes(padded) == pytest.approx(g._matn3_singular_extremes(M))


# ---------------------------------------------------------------------------
# unified --oscillate / --lock grammar parser (OSCILLATE_GRAMMAR.md, phase 1.1)
# ---------------------------------------------------------------------------

def test_oscillate_single_axis_defaults():
    grps = g.parse_oscillate(["tumble"])
    assert len(grps) == 1
    assert grps[0].items == [(1.0, "tumble")]
    assert grps[0].rate == 1.0 and grps[0].phase == 0.0


def test_oscillate_comma_is_one_composite_group():
    # comma = ONE oscillator on the shared diagonal (one degree of freedom)
    grps = g.parse_oscillate(["tumble,bloom"])
    assert len(grps) == 1
    assert grps[0].axes() == ["tumble", "bloom"]


def test_oscillate_space_is_independent_groups():
    # space = TWO independent oscillators (a torus)
    grps = g.parse_oscillate(["tumble", "bloom"])
    assert len(grps) == 2
    assert grps[0].axes() == ["tumble"]
    assert grps[1].axes() == ["bloom"]


def test_oscillate_amplitudes():
    grps = g.parse_oscillate(["2*tumble,1.5*bloom"])
    assert grps[0].items == [(2.0, "tumble"), (1.5, "bloom")]


def test_oscillate_amplitude_expression_splits_on_last_star():
    # amp may itself be an arithmetic expr; split on the LAST '*'
    (amp, axis), = g.parse_oscillate(["2*pi*tumble"])[0].items
    assert axis == "tumble"
    assert amp == pytest.approx(2 * math.pi)


def test_oscillate_rate_and_phase():
    grps = g.parse_oscillate(["bloom,tumble", "rate", "2", "phase", "pi/2"])
    assert len(grps) == 1
    assert grps[0].rate == 2.0
    assert grps[0].phase == pytest.approx(math.pi / 2)


def test_oscillate_rate_phase_either_order():
    a = g.parse_oscillate(["drift", "rate", "2", "phase", "1"])[0]
    b = g.parse_oscillate(["drift", "phase", "1", "rate", "2"])[0]
    assert a.rate == b.rate == 2.0
    assert a.phase == b.phase == 1.0


def test_oscillate_multi_group_with_clocks():
    grps = g.parse_oscillate(
        ["bloom,tumble", "rate", "2", "phase", "pi/2", "drift,3", "rate", "1"])
    assert len(grps) == 2
    assert grps[0].axes() == ["bloom", "tumble"]
    assert grps[0].rate == 2.0 and grps[0].phase == pytest.approx(math.pi / 2)
    assert grps[1].axes() == ["drift", "3"]      # '3' is a spatial dim index
    assert grps[1].rate == 1.0 and grps[1].phase == 0.0


def test_oscillate_dim_index_axis():
    grps = g.parse_oscillate(["0,1"])
    assert grps[0].axes() == ["0", "1"]


def test_oscillate_reserved_word_as_axis_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_oscillate(["tumble,phase"])


def test_oscillate_keyword_must_follow_a_group():
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_oscillate(["rate", "2"])


def test_oscillate_keyword_needs_expression():
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_oscillate(["tumble", "rate"])


def test_oscillate_duplicate_keyword_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_oscillate(["tumble", "rate", "1", "rate", "2"])


def test_oscillate_empty_item_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_oscillate(["tumble,"])        # stray comma


def test_oscillate_bad_expression_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_oscillate(["tumble", "rate", "2+"])
    with pytest.raises(argparse.ArgumentTypeError):
        g.parse_oscillate(["tumble", "rate", "foo"])       # unknown name


def test_oscillate_empty_tokens_gives_no_groups():
    assert g.parse_oscillate([]) == []
    assert g.parse_oscillate(None) == []


def test_lock_flattens_and_dedups():
    axes = g.parse_lock_axes(["tumble,bloom", "rate", "2", "tumble", "0"])
    assert axes == ["tumble", "bloom", "0"]    # order preserved, deduped


# --- P1.2: --transform -> --oscillate desugaring bridge (OSCILLATE_GRAMMAR §3) ---

def test_transform_desugar_drift_default():
    grps = g.transform_to_oscillate("drift")
    assert g.oscillate_spec(grps) == "drift"


def test_transform_desugar_rotate():
    assert g.oscillate_spec(g.transform_to_oscillate("rotate")) == "rotate"


def test_transform_desugar_tumble_rotate_mode():
    # default tumble mode 'rotate' -> plain tumble (amp 1)
    assert g.oscillate_spec(g.transform_to_oscillate("tumble")) == "tumble"


def test_transform_desugar_layered_is_one_composite_group():
    grps = g.transform_to_oscillate("drift+tumble")
    assert len(grps) == 1                       # comma-joined single composite
    assert g.oscillate_spec(grps) == "drift,tumble"


def test_transform_desugar_bloom_dims_default():
    # --transform bloom (default --bloom dims) -> just 'bloom'
    grps = g.transform_to_oscillate("bloom", bloom_params=("dims",))
    assert g.oscillate_spec(grps) == "bloom"


def test_transform_desugar_bloom_scalar_params():
    grps = g.transform_to_oscillate("bloom", bloom_params=("freq", "threshold"))
    # 'dims' absent -> no bloom envelope item; freq/threshold swingers at amp 1
    assert g.oscillate_spec(grps) == "freq,threshold"


def test_transform_desugar_bloom_dims_and_scalars():
    grps = g.transform_to_oscillate("bloom", bloom_params=("dims", "freq", "threshold"))
    assert g.oscillate_spec(grps) == "bloom,freq,threshold"


def test_transform_desugar_bloom_amp_scales_swingers():
    grps = g.transform_to_oscillate("bloom", bloom_params=("freq",), bloom_amp=1.5)
    assert g.oscillate_spec(grps) == "1.5*freq"
    # the dims-crossfade 'bloom' item is never scaled by bloom_amp
    grps2 = g.transform_to_oscillate("bloom", bloom_params=("dims", "freq"), bloom_amp=1.5)
    assert g.oscillate_spec(grps2) == "bloom,1.5*freq"


def test_transform_desugar_scalars_only_under_bloom():
    # scalar swingers do nothing without the bloom transform active
    grps = g.transform_to_oscillate("drift", bloom_params=("freq",))
    assert g.oscillate_spec(grps) == "drift"


def test_transform_desugar_tumble_slide_uses_tumble_amp():
    grps = g.transform_to_oscillate("tumble", tumble_mode="slide", tumble_amp=0.3)
    assert g.oscillate_spec(grps) == "0.3*tumble"


def test_transform_desugar_full_stack():
    grps = g.transform_to_oscillate("drift+tumble+bloom",
                                    bloom_params=("dims", "freq"), bloom_amp=2)
    assert g.oscillate_spec(grps) == "drift,tumble,bloom,2*freq"


def test_transform_desugar_empty_is_no_groups():
    assert g.transform_to_oscillate("") == []


def test_transform_desugar_roundtrips_through_parser():
    # desugared spec must parse back to an equivalent group model
    grps = g.transform_to_oscillate("drift+tumble+bloom",
                                    bloom_params=("dims", "freq"), bloom_amp=2)
    reparsed = g.parse_oscillate(g.oscillate_spec(grps).split())
    assert g.oscillate_spec(reparsed) == g.oscillate_spec(grps)


# --- P1.3: --oscillate wiring == the equivalent legacy --transform invocation ---

def _osc_equiv(seed, osc_argv, legacy_argv):
    """Assert `--oscillate <osc_argv>` yields a variant behaviorally identical to the
    legacy `--transform <legacy_argv>` at the same seed: same normalized transform,
    bloom targets, tumble config, and identical emitted field / swinger values."""
    va = g.pick_variant(seed, _args(*osc_argv), {})
    vb = g.pick_variant(seed, _args(*legacy_argv), {})
    tr = _reparse_transform(osc_argv, legacy_argv)
    assert va.dims == vb.dims
    assert va.freq == vb.freq
    assert va.bloom_params == vb.bloom_params
    assert va.tumble_mode == vb.tumble_mode
    assert va.tumble_amp == vb.tumble_amp
    assert va.tumble_locked == vb.tumble_locked
    assert va.tumble_planes == vb.tumble_planes
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert g.bloom_freq(va, t) == g.bloom_freq(vb, t)
        assert g.bloom_threshold(va, t) == g.bloom_threshold(vb, t)
        assert g.bloom_thickness_scale(va, t) == g.bloom_thickness_scale(vb, t)
        assert g.field_expr(va, t, tr) == g.field_expr(vb, t, tr)


def _reparse_transform(osc_argv, legacy_argv):
    # both invocations must resolve to the same canonical transform string
    a = _args(*legacy_argv)
    g.resolve_oscillate(a)
    return g._parse_transforms(a.transform)


def test_oscillate_drift_equals_default():
    _osc_equiv(11, ["--oscillate", "drift"], ["--transform", "drift"])


def test_oscillate_layered_equals_legacy():
    _osc_equiv(11, ["--dims", "6", "--oscillate", "drift,tumble"],
               ["--dims", "6", "--transform", "drift,tumble"])


def test_oscillate_tumble_equals_legacy():
    _osc_equiv(7, ["--dims", "6", "--oscillate", "tumble"],
               ["--dims", "6", "--transform", "tumble"])


def test_oscillate_tumble_slide_amp_equals_legacy():
    _osc_equiv(7, ["--dims", "6", "--oscillate", "0.3*tumble"],
               ["--dims", "6", "--transform", "tumble", "--tumble-mode", "slide",
                "--tumble-amp", "0.3"])


def test_oscillate_bloom_equals_legacy():
    _osc_equiv(5, ["--dims", "6", "--oscillate", "bloom"],
               ["--dims", "6", "--transform", "bloom"])


def test_oscillate_freq_swinger_equals_legacy():
    _osc_equiv(5, ["--dims", "6", "--oscillate", "freq"],
               ["--dims", "6", "--transform", "bloom", "--bloom", "freq"])


def test_oscillate_bloom_freq_equals_legacy():
    _osc_equiv(5, ["--dims", "6", "--oscillate", "bloom,freq"],
               ["--dims", "6", "--transform", "bloom", "--bloom", "dims,freq"])


def test_oscillate_freq_amp_equals_legacy_bloom_amp():
    _osc_equiv(5, ["--dims", "6", "--oscillate", "1.5*freq"],
               ["--dims", "6", "--transform", "bloom", "--bloom", "freq",
                "--bloom-amp", "1.5"])


def test_oscillate_lock_maps_to_tumble_lock():
    _osc_equiv(7, ["--dims", "6", "--oscillate", "tumble", "--lock", "0,1"],
               ["--dims", "6", "--transform", "tumble", "--tumble-lock", "0,1"])


def test_oscillate_per_axis_swinger_amps():
    # two swingers, each its own amplitude (not expressible with a single --bloom-amp)
    v = g.pick_variant(5, _args("--dims", "6", "--oscillate", "2*freq,0.5*threshold"), {})
    assert v.bloom_params == ("freq", "threshold")
    assert v.bloom_amps == {"freq": 2.0, "threshold": 0.5}
    assert g._swing_amp(v, "freq") == 2.0
    assert g._swing_amp(v, "threshold") == 0.5


def test_oscillate_conflicts_with_transform():
    with pytest.raises(SystemExit):
        g.main(["--oscillate", "drift", "--transform", "tumble", "--no-video", "--count", "1"])


def test_oscillate_conflicts_with_legacy_bloom_flag():
    with pytest.raises(SystemExit):
        g.main(["--oscillate", "freq", "--bloom", "threshold", "--no-video", "--count", "1"])


# --- P1.4: winder rate (= winding), phase, and bare dim-index axes ---

def _d(v, idx):
    return [d for d in v.dim_list if d.index == idx][0]


def test_oscillate_motion_rate_equals_max_winding():
    # `rate R` on a motion group is the ceiling of the varied winding cycle == --max-winding R
    _osc_equiv(11, ["--dims", "8", "--oscillate", "drift", "rate", "3"],
               ["--dims", "8", "--transform", "drift", "--max-winding", "3"])


def test_oscillate_tumble_rate_equals_max_winding():
    _osc_equiv(7, ["--dims", "8", "--oscillate", "tumble", "rate", "2"],
               ["--dims", "8", "--transform", "tumble", "--max-winding", "2"])


def test_oscillate_motion_rate_records_ceiling():
    a = _args("--dims", "8", "--oscillate", "drift", "rate", "5")
    g.resolve_oscillate(a)
    assert a.osc_max_winding == 5
    assert a.transform == "drift"


def test_oscillate_bare_dim_pins_exact_winding():
    # a lone dim index is forced on and pinned to an exact winding (round(amp*rate))
    v = g.pick_variant(3, _args("--dims", "6", "--oscillate", "3", "rate", "2"), {})
    d3 = _d(v, 3)
    assert d3.oscillate
    assert d3.winding == 2


def test_oscillate_bare_dim_amp_is_winding():
    # amp == rate for a winder, so 2*3 (amp) and `3 rate 2` both give dim 3 winding 2
    v = g.pick_variant(3, _args("--dims", "6", "--oscillate", "2*3"), {})
    assert _d(v, 3).winding == 2 and _d(v, 3).oscillate


def test_oscillate_bare_dim_defaults_to_drift():
    a = _args("--dims", "6", "--oscillate", "3")
    g.resolve_oscillate(a)
    assert a.transform == "drift"            # a bare dim moves via drift
    assert a.osc_dim_windings == {3: 1}      # default amp*rate == 1


def test_oscillate_bare_dim_raises_dim_floor():
    # naming dim 5 forces D >= 6, so --dims 4 is infeasible
    with pytest.raises(SystemExit):
        g.pick_variant(3, _args("--dims", "4", "--oscillate", "5"), {})


def test_oscillate_bare_dim_conflicts_with_axis_off():
    with pytest.raises(SystemExit):
        g.pick_variant(3, _args("--dims", "6", "--oscillate", "3"),
                       {3: g.AxisLock(on=False)})


def test_oscillate_phase_loops_seamlessly_and_shifts():
    v = g.pick_variant(11, _args("--dims", "6", "--oscillate", "drift", "phase", "pi/2"), {})
    assert v.osc_phase == math.pi / 2
    # seamless: the loop still returns to itself at t=1
    assert g.field_expr(v, 0.0, "drift") == g.field_expr(v, 1.0, "drift")
    # but the phase genuinely offsets the start relative to the unphased loop
    v0 = g.pick_variant(11, _args("--dims", "6", "--oscillate", "drift"), {})
    assert g.field_expr(v, 0.0, "drift") != g.field_expr(v0, 0.0, "drift")


def test_oscillate_phase_default_is_byte_identical():
    # no phase given => osc_phase 0 => identical to the plain drift loop
    v = g.pick_variant(11, _args("--dims", "6", "--oscillate", "drift"), {})
    assert v.osc_phase == 0.0
    vb = g.pick_variant(11, _args("--dims", "6", "--transform", "drift"), {})
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert g.field_expr(v, t, "drift") == g.field_expr(vb, t, "drift")


def test_oscillate_conflicting_rates_error():
    # two independent motion groups with different rates can't share the one winding clock
    with pytest.raises(SystemExit):
        g.pick_variant(3, _args("--dims", "8", "--oscillate", "drift", "rate", "2",
                                "rotate", "rate", "3"), {})


def test_oscillate_swinger_rate_accepted_and_stored():
    # P3.2b: swingers now carry their own envelope clock, uniform with winders/bloom.
    v = g.pick_variant(3, _args("--dims", "6", "--oscillate", "freq", "rate", "2"), {})
    assert v.bloom_rates.get("freq") == 2.0
    # rate 2 => two full bumps over the loop => the envelope peaks at t=0.25 and t=0.75
    lo = bloom_freq0 = g.bloom_freq(v, 0.0)
    assert g.bloom_freq(v, 0.25) > lo
    assert g.bloom_freq(v, 0.75) > lo
    # and it returns to the trough at the mid-loop (t=0.5) between the two bumps
    assert abs(g.bloom_freq(v, 0.5) - lo) < 1e-12


def test_oscillate_swinger_rate_default_is_byte_identical():
    # no rate/phase => rate 1 / phase 0 => byte-for-byte the legacy fixed sin^2 envelope
    v = g.pick_variant(7, _args("--dims", "6", "--oscillate", "freq"), {})
    assert v.bloom_rates == {} and v.bloom_phases == {}
    vb = g.pick_variant(7, _args("--dims", "6", "--transform", "bloom", "--bloom", "freq"), {})
    for t in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        assert g.bloom_freq(v, t) == g.bloom_freq(vb, t)
        assert g.field_expr(v, t, "bloom") == g.field_expr(vb, t, "bloom")


def test_oscillate_swinger_integer_rate_still_loops():
    # an integer rate keeps the loop seamless (last frame == first) for any phase
    v = g.pick_variant(5, _args("--dims", "6", "--oscillate", "freq", "rate", "3",
                                "phase", "pi/2"), {})
    # seamless up to floating-point (cos(2*pi*3 + phase) rounds slightly off cos(phase))
    assert abs(g.bloom_freq(v, 0.0) - g.bloom_freq(v, 1.0)) < 1e-9


def test_oscillate_swinger_phase_offsets_but_loops():
    # a phase offset shifts the envelope but (integer default rate) still loops seamlessly
    v = g.pick_variant(5, _args("--dims", "6", "--oscillate", "freq", "phase", "pi"), {})
    assert v.bloom_phases.get("freq") == math.pi
    assert abs(g.bloom_freq(v, 0.0) - g.bloom_freq(v, 1.0)) < 1e-9
    # phase pi flips the sin^2 bump: it now *peaks* at t=0 instead of troughing there
    v0 = g.pick_variant(5, _args("--dims", "6", "--oscillate", "freq"), {})
    assert g.bloom_freq(v, 0.0) > g.bloom_freq(v0, 0.0)


def test_oscillate_bloom_dims_rate_stored():
    # the dimensional crossfade ('bloom') is keyed 'dims' in the rate/phase tables
    v = g.pick_variant(5, _args("--dims", "6", "--oscillate", "bloom", "rate", "2"), {})
    assert v.bloom_rates.get("dims") == 2.0


def test_oscillate_swinger_noninteger_rate_warns(capsys):
    # a non-integer rate pulses faster but breaks the seamless loop: main() must warn
    g.main(["--dims", "6", "--oscillate", "freq", "rate", "2.5",
            "--no-video", "--count", "1"])
    out = capsys.readouterr().out
    assert "won't loop seamlessly" in out
    assert "freq" in out


# --------------------------------------------------------------------------
# P3.2: --list-surfaces / --surface-help discovery commands
# --------------------------------------------------------------------------

def test_surface_catalog_covers_tpms_and_every_pov():
    names = g.surface_names()
    # the 4 periodic TPMS families, in order, then every POV builtin, no duplicates
    assert names[:4] == ["gyroid", "primitive", "schwarz_d", "neovius"]
    assert set(names) == {"gyroid", "primitive", "schwarz_d", "neovius"} | set(g.POV_FUNCS)
    assert len(names) == len(set(names)) == 4 + len(g.POV_FUNCS)


def test_surface_group_classification_matches_honesty_sets():
    assert g.surface_group("gyroid") == "periodic"
    assert g.surface_group("schwarz_p") == "periodic"      # alias of primitive
    assert g.surface_group("f_sphere") == "nd_pov"         # in POV_ND_GENERALIZABLE
    assert g.surface_group("f_torus") == "affine_pov"      # ordinary POV builtin
    # every POV builtin lands in exactly one POV group by its N-D status
    for name in g.POV_FUNCS:
        grp = g.surface_group(name)
        assert grp == ("nd_pov" if name in g.POV_ND_GENERALIZABLE else "affine_pov")
    with pytest.raises(ValueError):
        g.surface_group("not_a_surface")


def test_list_surfaces_text_is_grouped_and_ascii():
    txt = g._list_surfaces_text()
    assert "periodic minimal surfaces" in txt
    assert "N-D-generalizable POV builtins" in txt
    assert "affine-only POV builtins" in txt
    assert "f_sphere" in txt and "params=1" in txt
    assert f"{len(g.surface_names())} surfaces total" in txt
    # console-safe: no non-ASCII (Windows cp1252 would mojibake em-dashes)
    txt.encode("ascii")


def test_surface_help_pov_lists_authored_params():
    txt = g._surface_help_text("f_torus")
    assert "surface: f_torus" in txt
    assert "affine-only" in txt
    assert "major" in txt and "minor" in txt
    assert "range [0.1, 4]" in txt
    txt.encode("ascii")


def test_surface_help_nd_pov_flags_generalization():
    txt = g._surface_help_text("f_sphere")
    assert "N-D-generalizable" in txt and "radius" in txt


def test_surface_help_tpms_has_no_shape_params():
    txt = g._surface_help_text("gyroid")
    assert "periodic minimal surface" in txt
    assert "freq / threshold / thickness" in txt


def test_surface_help_alias_shows_canonical_and_alias():
    txt = g._surface_help_text("schwarz_p")
    assert txt.startswith("surface: primitive")
    assert "alias: schwarz_p" in txt


def test_surface_help_zero_param_helper():
    txt = g._surface_help_text("f_r")
    assert "0-parameter helper" in txt


def test_surface_help_unknown_name_raises():
    with pytest.raises(SystemExit):
        g._surface_help_text("f_not_real")


def test_list_surfaces_cli_exits_without_generating(capsys):
    rc = g.main(["--list-surfaces"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "loom surface library" in out and "f_sphere" in out


def test_surface_help_cli_exits_without_generating(capsys):
    rc = g.main(["--surface-help", "f_torus"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "major" in out and "minor" in out


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
