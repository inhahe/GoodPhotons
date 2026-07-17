"""Datasets as DAG nodes (control points / grid / scatter modulable *and*
walked by the loop detector) + the HSV/RGB colour model (2-D, 3-D, and the
``.ftsl`` colour token).  Runnable directly or under pytest.
"""

from __future__ import annotations

import colorsys
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Sine, vec, RefSignal, detect_signal_cycle, SignalCycleError,
    walk, PointPath, TrackedPath, Grid, Scatter, LoopCurve, GridField, ScatterField,
    Color, rgb, hsv, hsv_to_rgb, rgb_to_hsv,
    hsl, hsl_to_rgb, rgb_to_hsl, skin, Texture, Scene, Camera, Sphere, Material,
)
from loom.ftsl_emit import value_token  # noqa: E402


def _clk() -> Clock:
    return Clock(t=0.0, frame=0, frames=1, fps=30.0)


# --- datasets are DAG nodes -------------------------------------------------

def test_datasets_have_ids():
    pp = PointPath([(0.0, 0.0), (1.0, 1.0), (0.5, 2.0)])
    g = Grid((2, 2), (0, 0), (1, 1), [0.0, 1.0, 2.0, 3.0])
    sc = Scatter([((0.0, 0.0), 1.0), ((1.0, 1.0), 2.0)])
    tp = TrackedPath([(0.0, 0.0), (1.0, 1.0)], tracks={"w": [1.0, 2.0]})
    ids = {pp.id, g.id, sc.id, tp.id}
    assert len(ids) == 4                       # all distinct


def test_interpolator_walk_includes_dataset():
    pp = PointPath([(0.0, 0.0), (1.0, 1.0), (0.5, 2.0)])
    curve = LoopCurve(pp, 0.0)
    assert pp.id in {n.id for n in walk(curve)}
    g = Grid((2, 2), (0, 0), (1, 1), [0.0, 1.0, 2.0, 3.0])
    assert g.id in {n.id for n in walk(GridField(g, vec(0.5, 0.5)))}
    sc = Scatter([((0.0, 0.0), 1.0), ((1.0, 1.0), 2.0)])
    assert sc.id in {n.id for n in walk(ScatterField(sc, vec(0.5, 0.5)))}


def test_detect_cycle_runs_on_dataset_directly():
    # a dataset is a node, so the loop detector accepts it directly (no crash)
    detect_signal_cycle(PointPath([(0.0, 0.0), (1.0, 1.0)]))
    detect_signal_cycle(Grid((2, 2), (0, 0), (1, 1), [0.0, 1.0, 2.0, 3.0]))
    detect_signal_cycle(Scatter([((0.0, 0.0), 1.0), ((1.0, 1.0), 2.0)]))


def test_cycle_through_control_point_is_caught():
    ref = RefSignal("x")
    pp = PointPath([vec(ref, 0.0), (1.0, 1.0)])
    curve = LoopCurve(pp, 0.0)
    ref.bind(curve.components[0])              # a control point loops back
    try:
        detect_signal_cycle(curve)
    except SignalCycleError:
        return
    raise AssertionError("expected a cycle through a curve control point")


def test_cycle_through_grid_value_is_caught():
    ref = RefSignal("y")
    g = Grid((2, 2), (0, 0), (1, 1), [ref, 1.0, 2.0, 3.0])
    gf = GridField(g, vec(0.5, 0.5))
    ref.bind(gf)                               # a grid value loops back
    try:
        detect_signal_cycle(gf)
    except SignalCycleError:
        return
    raise AssertionError("expected a cycle through a grid value")


# --- HSV / RGB colour model -------------------------------------------------

def test_hsv_to_rgb_matches_reference():
    clk, ca = _clk(), Cache()
    for h, s, v in [(0.0, 1.0, 1.0), (0.33, 0.8, 0.9), (0.5, 0.5, 0.5),
                    (0.9, 1.0, 1.0), (1.25, 0.7, 0.6)]:
        got = hsv(h, s, v).at(clk, ca)
        exp = colorsys.hsv_to_rgb(h % 1.0, s, v)
        assert all(abs(a - b) < 1e-9 for a, b in zip(got, exp)), (h, s, v, got, exp)


def test_rgb_hsv_round_trip():
    clk, ca = _clk(), Cache()
    base = (0.2, 0.7, 0.4)
    hsv_vals = rgb_to_hsv(*base).at(clk, ca)
    back = hsv_to_rgb(*hsv_vals).at(clk, ca)
    assert all(abs(a - b) < 1e-9 for a, b in zip(back, base)), (hsv_vals, back)


def test_color_is_rgb_vecsignal():
    c = hsv(0.0, 1.0, 1.0)                      # pure red
    assert c.dim == 3
    got = c.at(_clk(), Cache())
    assert abs(got[0] - 1.0) < 1e-9 and got[1] < 1e-9 and got[2] < 1e-9, got


def test_color_emits_ftsl_token_in_authored_model():
    clk, ca = _clk(), Cache()
    assert value_token(rgb(0.9, 0.4, 0.1), clk, ca) == "rgb 0.9 0.4 0.1"
    assert value_token(hsv(0.6, 0.8, 0.9), clk, ca) == "hsv 0.6 0.8 0.9"


def test_hue_wraps_seamlessly():
    # hue 0 and hue 1 are the same colour (seamless loop over the wheel)
    clk, ca = _clk(), Cache()
    a = hsv(0.0, 0.8, 0.9).at(clk, ca)
    b = hsv(1.0, 0.8, 0.9).at(clk, ca)
    assert all(abs(x - y) < 1e-9 for x, y in zip(a, b)), (a, b)


def test_animated_color_is_acyclic_and_changes():
    c = hsv(Sine(cycles=1) * 0.5 + 0.5, 0.8, 1.0)
    detect_signal_cycle(c)
    a = c.at(Clock.at_frame(0, 8), Cache())
    b = c.at(Clock.at_frame(2, 8), Cache())
    assert any(abs(x - y) > 1e-6 for x, y in zip(a, b)), (a, b)


def test_color_works_in_scene_and_canvas():
    from loom import Canvas2D, Material, Sphere, Camera, Scene
    cv = Canvas2D(8, 8, background=hsv(0.6, 0.5, 0.2))
    cv.plot(x=0.0, y=0.0, color=hsv(0.1, 0.9, 1.0), radius=2)
    cv.check_cycles()
    cv.rasterize(_clk(), Cache())              # must not raise
    sc = Scene(Camera(eye=(0, 0, 3), look_at=(0, 0, 0)))
    sc.add(Material("m", "diffuse", reflect=hsv(0.55, 0.7, 0.9)),
           Sphere((0, 0, 0), 1.0, "m"))
    sc.check_cycles()
    assert "reflect hsv 0.55 0.7 0.9" in sc.emit(_clk(), Cache())


# --- HSL / RGB colour model -------------------------------------------------

def test_hsl_to_rgb_matches_reference():
    clk, ca = _clk(), Cache()
    for h, s, l in [(0.0, 1.0, 0.5), (0.33, 0.8, 0.4), (0.5, 0.5, 0.5),
                    (0.9, 1.0, 0.5), (1.25, 0.7, 0.3), (0.6, 0.7, 0.5)]:
        got = hsl(h, s, l).at(clk, ca)
        exp = colorsys.hls_to_rgb(h % 1.0, l, s)   # colorsys order is h, l, s
        assert all(abs(a - b) < 1e-9 for a, b in zip(got, exp)), (h, s, l, got, exp)


def test_rgb_hsl_round_trip():
    clk, ca = _clk(), Cache()
    base = (0.2, 0.7, 0.4)
    hsl_vals = rgb_to_hsl(*base).at(clk, ca)
    back = hsl_to_rgb(*hsl_vals).at(clk, ca)
    assert all(abs(a - b) < 1e-9 for a, b in zip(back, base)), (hsl_vals, back)


def test_hsl_lightness_extremes():
    clk, ca = _clk(), Cache()
    white = hsl(0.3, 0.9, 1.0).at(clk, ca)         # l=1 -> white regardless of h/s
    black = hsl(0.3, 0.9, 0.0).at(clk, ca)         # l=0 -> black
    assert all(abs(x - 1.0) < 1e-9 for x in white), white
    assert all(abs(x) < 1e-9 for x in black), black


def test_hsl_emits_ftsl_token():
    clk, ca = _clk(), Cache()
    assert value_token(hsl(0.6, 0.7, 0.5), clk, ca) == "hsl 0.6 0.7 0.5"


def test_hsl_color_in_scene():
    sc = Scene(Camera(eye=(0, 0, 3), look_at=(0, 0, 0)))
    sc.add(Material("m", "diffuse", reflect=hsl(0.6, 0.7, 0.5)),
           Sphere((0, 0, 0), 1.0, "m"))
    sc.check_cycles()
    assert "reflect hsl 0.6 0.7 0.5" in sc.emit(_clk(), Cache())


# --- image skins (textures) -------------------------------------------------

def test_skin_makes_texture_and_material():
    tex, mat = skin("hide", "textures/cow.png", roughness=0.4)
    assert isinstance(tex, Texture)
    assert isinstance(mat, Material)
    assert tex.name == "hide"


def test_skin_emits_texture_block_and_binding():
    import tempfile
    from pathlib import Path
    sc = Scene(Camera(eye=(0, 0, 4), look_at=(0, 0, 0)))
    sc.add(*skin("hide", "textures/cow.png", roughness=0.4),
           Sphere((0, 0, 0), 1.0, "hide"))
    sc.check_cycles()
    with tempfile.TemporaryDirectory() as d:
        out = sc.emit(_clk(), Cache(), assets_dir=Path(d))
    assert 'texture "hide" {' in out
    assert 'file "textures/cow.png"' in out
    assert "reflect texture:hide" in out
    # texture block comes before the material that binds it
    assert out.index('texture "hide"') < out.index('material "hide"')


def test_skin_texture_options_stored():
    tex, _ = skin("t", "a.png", encoding="linear", filter="nearest", wrap="clamp")
    assert (tex.encoding, tex.filter, tex.wrap) == ("linear", "nearest", "clamp")


def test_texture_rejects_bad_options():
    for bad in [dict(encoding="bogus"), dict(filter="bogus"), dict(wrap="bogus")]:
        try:
            Texture("x", "a.png", **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def _run_all() -> int:
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
