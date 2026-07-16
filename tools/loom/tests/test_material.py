"""M6 tests: function-driven materials (animated patterns + mix blend).

Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Const, Sine, vec, rotations, phase_drift,
    Scene, Material, Camera, Raw, Light,
    FuncPattern, MixMaterial, waves, rings,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402


def _emit(el, clock):
    return el.emit(EmitCtx(clock=clock, cache=Cache()))


def test_pattern_emits_wellformed_block():
    fp = FuncPattern("rough", "waves", freq=3.0)
    txt = _emit(fp, Clock(t=0.0))
    assert txt.startswith('pattern "rough" {')
    assert "expr" in txt and "sin(" in txt
    assert txt.count("{") == txt.count("}")


def test_pattern_literal_is_static():
    fp = FuncPattern("p", "0.5+0.5*sin(10*x)")
    a = _emit(fp, Clock.at_frame(0, 24))
    b = _emit(fp, Clock.at_frame(9, 24))
    assert a == b, "a literal-string pattern must not vary over time"
    assert 'expr "0.5+0.5*sin(10*x)"' in a


def test_pattern_drift_animates_and_is_seamless():
    fp = FuncPattern("p", waves, freq=1.0, drift=vec(phase_drift(1.0), 0, 0))
    a0 = _emit(fp, Clock.at_frame(0, 24))
    amid = _emit(fp, Clock.at_frame(12, 24))
    awrap = _emit(fp, Clock.at_frame(24, 24))
    assert a0 != amid, "a drifting pattern should change across the loop"
    assert a0 == awrap, "loop wrap must reproduce frame 0 exactly (seamless)"


def test_pattern_no_plus_minus_adjacency():
    rot = rotations(3, [(0, 1, 0.6), (1, 2, -0.5)])
    fp = FuncPattern("p", rings, freq=2.0, rotation=rot)
    txt = _emit(fp, Clock(t=0.2))
    assert "+-" not in txt and "*-" not in txt, txt


def test_mix_material_blends_two_layers():
    mm = MixMaterial("skin", [("gold", 0.5), ("blue", 0.5)], weight_map="sel")
    txt = _emit(mm, Clock(t=0.0))
    assert "type mix" in txt
    assert txt.count("layer ") == 2
    assert "weight_map pattern:sel" in txt


def test_mix_weight_map_needs_two_layers():
    try:
        MixMaterial("bad", [("a", 0.3), ("b", 0.3), ("c", 0.4)], weight_map="sel")
    except ValueError:
        return
    raise AssertionError("a weight_map mix must reject != 2 layers")


def test_scene_orders_patterns_before_materials():
    s = Scene(Camera(eye=(0, 0, 3), look_at=(0, 0, 0), res=(16, 16)))
    s.add(
        FuncPattern("sel", rings, freq=2.0, drift=vec(phase_drift(1.0), 0, 0)),
        Material("gold", "diffuse", reflect=0.9),
        Material("blue", "diffuse", reflect=0.3),
        MixMaterial("skin", [("gold", 0.5), ("blue", 0.5)], weight_map="sel"),
    )
    s.check_cycles()
    txt = s.emit(Clock.at_frame(4, 24), Cache())
    ip = txt.index('pattern "sel"')
    im = txt.index('material "gold"')
    assert ip < im, "the pattern must be emitted before the material that binds it"
    assert 'material "skin"' in txt and "type mix" in txt


def test_material_binds_pattern_string_prop():
    # a plain Material can bind a pattern to a scalar knob via a string value
    m = Material("metal", "metal", roughness="pattern:rough")
    txt = _emit(m, Clock(t=0.0))
    assert "roughness pattern:rough" in txt


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
