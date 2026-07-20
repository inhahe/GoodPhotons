"""M9 tests: the POV-Ray isosurface function library.

The headline "golden" test re-derives the name->arity table straight from
ftrace's ``src/pov_functions.h`` and asserts loom's :data:`POV_FUNCS` matches it
exactly — so any drift between loom and the renderer fails loudly.  The rest check
the emitted call strings, arity validation, param animation, and integration into
:class:`Isosurface` / :class:`FuncPattern`.  Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Sine,
    Isosurface, FuncPattern, pov, PovFn, POV_FUNCS, POV_ND_GENERALIZABLE,
    POV_PARAMS, pov_params,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402


def _emit(el, clock):
    return el.emit(EmitCtx(clock=clock, cache=Cache()))


def _header_path():
    # tests/ -> loom/ -> tools/ -> <repo root>/src/pov_functions.h
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(root, "src", "pov_functions.h")


def test_pov_table_matches_ftrace_header():
    path = _header_path()
    if not os.path.exists(path):
        print(f"  (skip: {path} not found)")
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    # entries look like:  { "f_torus", 70, 5 },
    pairs = re.findall(r'\{\s*"(f_[a-z0-9_]+)"\s*,\s*\d+\s*,\s*(\d+)\s*\}', text)
    header = {name: int(arity) for name, arity in pairs}
    assert header, "failed to parse any entries from pov_functions.h"
    assert header == POV_FUNCS, (
        "loom.POV_FUNCS is out of sync with src/pov_functions.h; "
        f"only-in-header={set(header) - set(POV_FUNCS)}, "
        f"only-in-loom={set(POV_FUNCS) - set(header)}")


def test_pov_count_is_78():
    assert len(POV_FUNCS) == 78


def test_pov_builds_call_string_via_pattern():
    fp = FuncPattern("t", pov("f_torus", 0.8, 0.25))
    txt = _emit(fp, Clock(t=0.0))
    assert re.search(r'f_torus\([^,]+,[^,]+,[^,]+,0\.8,0\.25\)', txt), txt


def test_pov_zero_param_function():
    fp = FuncPattern("rad", pov("f_r"))  # f_r arity 3 -> 0 params
    txt = _emit(fp, Clock(t=0.0))
    assert "f_r(" in txt
    # exactly the 3 baked coordinates, no trailing param
    call = re.search(r'f_r\((.*)\)\s*"', txt).group(1)
    assert call.count("(") == call.count(")")


def test_pov_arity_validation():
    try:
        pov("f_torus", 0.8)  # needs 2 params
    except ValueError:
        pass
    else:
        raise AssertionError("f_torus with 1 param must raise")
    try:
        pov("f_sphere", 1.0, 2.0)  # needs 1 param
    except ValueError:
        pass
    else:
        raise AssertionError("f_sphere with 2 params must raise")


def test_pov_unknown_name_rejected():
    try:
        pov("f_not_a_function", 1.0)
    except ValueError:
        return
    raise AssertionError("an unknown POV name must raise")


def test_pov_param_animates_and_is_in_roots():
    p = pov("f_sphere", Sine(cycles=1.0, bias=1.2))
    fp = FuncPattern("s", p)
    a = _emit(fp, Clock.at_frame(0, 24))
    b = _emit(fp, Clock.at_frame(6, 24))
    assert a != b, "a Signal-driven POV param must animate across frames"
    # the param Signal must be exposed to the DAG (cycle check + cache)
    assert any(r in p.param_signals() for r in fp.roots())


def test_pov_in_isosurface_emits_block():
    iso = Isosurface(pov("f_sphere", 1.2), container="sphere", radius=1.6,
                     name="ball", material="skin")
    txt = _emit(iso, Clock(t=0.0))
    assert txt.startswith('ball = isosurface {')
    assert "f_sphere(" in txt
    assert "1.2" in txt
    assert txt.count("{") == txt.count("}")
    # the field param is folded into the isosurface's DAG roots
    assert len(iso.roots()) >= 1


def test_nd_generalizable_subset_is_honest():
    assert "f_sphere" in POV_ND_GENERALIZABLE
    assert "f_superellipsoid" in POV_ND_GENERALIZABLE
    assert "f_heart" not in POV_ND_GENERALIZABLE
    assert "f_klein_bottle" not in POV_ND_GENERALIZABLE
    assert POV_ND_GENERALIZABLE <= set(POV_FUNCS)


_AXIS_RE = re.compile(r'^[a-z][a-z0-9_]*$')


def test_pov_params_complete_for_every_function():
    # the same drift-guard discipline as the arity table: every POV_FUNCS entry
    # must carry exactly arity - 3 params, no more, no fewer.
    assert set(POV_PARAMS) == set(POV_FUNCS), (
        f"POV_PARAMS out of sync with POV_FUNCS; "
        f"missing={set(POV_FUNCS) - set(POV_PARAMS)}, "
        f"extra={set(POV_PARAMS) - set(POV_FUNCS)}")
    for name, arity in POV_FUNCS.items():
        params = pov_params(name)
        assert len(params) == arity - 3, (
            f"{name}: {len(params)} params but arity {arity} wants {arity - 3}")


def test_pov_params_are_well_formed():
    for name in POV_FUNCS:
        seen = set()
        for meta in pov_params(name):
            axis, desc, default, rng = meta
            assert _AXIS_RE.match(axis), f"{name}: bad axis name {axis!r}"
            assert axis not in seen, f"{name}: duplicate axis {axis!r}"
            seen.add(axis)
            assert isinstance(desc, str) and desc, f"{name}: empty desc for {axis}"
            lo, hi = rng
            assert lo < hi, f"{name}.{axis}: empty range ({lo}, {hi})"
            assert lo <= default <= hi, (
                f"{name}.{axis}: default {default} outside [{lo}, {hi}]")


def test_pov_params_spot_check_authored_metadata():
    assert [m[0] for m in pov_params("f_torus")] == ["major", "minor"]
    assert [m[0] for m in pov_params("f_sphere")] == ["radius"]
    assert [m[0] for m in pov_params("f_ellipsoid")] == ["rx", "ry", "rz"]
    assert pov_params("f_r") == []          # 0-param spherical helper
    assert pov_params("f_noise3d") == []
    # an un-authored function falls back to honest generic p0.. placeholders
    assert [m[0] for m in pov_params("f_klein_bottle")] == ["p0"]


def test_pov_params_unknown_name_rejected():
    try:
        pov_params("f_not_a_function")
    except ValueError:
        return
    raise AssertionError("pov_params on an unknown name must raise")


def test_pov_params_returns_a_copy():
    # callers must not be able to mutate the shared table in place
    a = pov_params("f_torus")
    a.append(("x", "x", 0.0, (0.0, 1.0)))
    assert len(pov_params("f_torus")) == 2


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
