"""M5 tests: animatable isosurface emission (gyroid + N-D slicer).

Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, Const, Sine, rotations, vec,
    Scene, Material, Camera, Isosurface, Room, gyroid_surface, phase_drift,
    affine, Affine,
    SliceField, ND_FIELDS, nd_grad_bound,
    gyroid, schwarz_p, schwarz_d, neovius,
    gyroid_n, schwarz_p_n, schwarz_d_n, neovius_n,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402


def _emit(el, clock):
    return el.emit(EmitCtx(clock=clock, cache=Cache()))


def test_isosurface_emits_wellformed_block():
    iso = gyroid_surface(freq=1.5, threshold=0.0, material="m")
    txt = _emit(iso, Clock(t=0.0))
    assert txt.startswith('gyroid = isosurface {')
    assert txt.count("{") == txt.count("}"), (txt.count("{"), txt.count("}"))
    assert 'function {' in txt and "expr" in txt
    assert "contained_by" in txt
    assert 'material "m"' in txt
    # gyroid uses sin/cos three times each
    assert txt.count("sin(") == 3 and txt.count("cos(") == 3


def test_no_plus_minus_adjacency():
    # a rotation makes coefficients negative; the emitter must not produce "+-".
    rot = rotations(3, [(0, 1, 0.7), (1, 2, -0.4)])
    iso = gyroid_surface(freq=1.0, rotation=rot, material="m")
    txt = _emit(iso, Clock(t=0.3))
    assert "+-" not in txt, txt
    assert "*-" not in txt, txt


def test_drift_animates_and_is_seamless():
    iso = gyroid_surface(freq=1.0, drift=vec(phase_drift(1.0), 0.0, 0.0),
                         material="m")
    a0 = _emit(iso, Clock.at_frame(0, 24))
    amid = _emit(iso, Clock.at_frame(12, 24))
    awrap = _emit(iso, Clock.at_frame(24, 24))
    assert a0 != amid, "drifting gyroid should change across the loop"
    assert a0 == awrap, "loop wrap must reproduce frame 0 exactly (seamless)"


def test_rotation_animates():
    rot = rotations(3, [(0, 2, Sine(cycles=1, amp=math.pi))])
    iso = gyroid_surface(freq=1.2, rotation=rot, material="m")
    a = _emit(iso, Clock(t=0.1, frame=2, frames=24))
    b = _emit(iso, Clock(t=0.3, frame=7, frames=24))
    assert a != b, "an animated rotation should tilt the field over time"


def test_sphere_container_and_open():
    iso = Isosurface("schwarz_p", freq=1.0, container="sphere",
                     center=(0, 0, 0), radius=3.0, open=True, material="m")
    txt = _emit(iso, Clock(t=0.0))
    assert "contained_by { sphere {" in txt
    assert "open on" in txt


def test_march_and_uv_controls_emit():
    # ftrace's addIsosurface reads samples/accuracy/refine/uv, but loom had no way to
    # emit any of them, so a sampled march was stuck on the 256-sample default. This
    # gap was found by the J3c emitter audit (scraps/emit_audit.py).
    iso = Isosurface("gyroid", freq=2.0, material="m", method="sample",
                     samples=300, accuracy=0.01, refine="regula_falsi",
                     uv="planar axis=y")
    txt = _emit(iso, Clock(t=0.0))
    for line in ("method sample", "samples 300", "refine regula_falsi",
                 "uv planar axis=y"):
        assert line in txt, (line, txt)
    assert "accuracy 0.01" in txt


def test_march_and_uv_controls_omitted_by_default():
    # None must mean "say nothing" so ftrace's own defaults still apply.
    txt = _emit(Isosurface("gyroid", freq=2.0, material="m"), Clock(t=0.0))
    for key in ("samples", "accuracy", "refine", "uv ", "method"):
        assert key not in txt, (key, txt)


def test_bareword_uv_axis_is_rejected():
    # FTSL.md §2's documented trap: `uv planar y` parses the axis as a stray
    # statement and is silently dropped, so the surface keeps default UVs. ftrace
    # now warns about it, but loom must not emit it in the first place.
    for bad in ("planar y", "planar axis=w", "spherical up", "cubic"):
        try:
            Isosurface("gyroid", freq=1.0, material="m", uv=bad)
        except ValueError:
            continue
        raise AssertionError(f"uv={bad!r} should have been rejected")
    for ok in ("planar", "spherical", "cylindrical axis=z"):
        Isosurface("gyroid", freq=1.0, material="m", uv=ok)


def test_bad_refine_and_method_are_rejected():
    for kw in ({"refine": "newton"}, {"method": "raymarch"}):
        try:
            Isosurface("gyroid", freq=1.0, material="m", **kw)
        except ValueError:
            continue
        raise AssertionError(f"{kw} should have been rejected")


def test_placement_zero_is_byte_identical():
    # default placement (origin) must emit exactly like the un-placed form.
    a = _emit(gyroid_surface(freq=1.0, material="m"), Clock(t=0.0))
    b = _emit(gyroid_surface(freq=1.0, placement=(0.0, 0.0, 0.0), material="m"),
              Clock(t=0.0))
    assert a == b


def test_placement_offsets_frame_and_container():
    iso = gyroid_surface(freq=1.0, placement=(2.0, 0.0, 0.0), material="m",
                         container="box",
                         bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    txt = _emit(iso, Clock(t=0.0))
    # the coordinate frame shifts: x becomes (x-(2))
    assert "(x-(2))" in txt, txt
    # and the container box moves with it: min x = -1+2 = 1, max x = 1+2 = 3
    assert "min 1 -1 -1" in txt, txt
    assert "max 3 1 1" in txt, txt


def test_sphere_placement_moves_center():
    iso = Isosurface("schwarz_p", freq=1.0, container="sphere",
                     center=(0, 0, 0), radius=3.0,
                     placement=(1.0, 2.0, -1.0), material="m")
    txt = _emit(iso, Clock(t=0.0))
    assert "center 1 2 -1" in txt, txt


def test_placement_animatable():
    # a Signal-driven placement should move the surface over time.
    iso = gyroid_surface(freq=1.0,
                         placement=vec(Sine(cycles=1, amp=3.0), 0.0, 0.0),
                         material="m")
    a = _emit(iso, Clock(t=0.1))
    b = _emit(iso, Clock(t=0.35))
    assert a != b, "an animated placement should move the surface"


def test_room_namespaces_child_names():
    room = Room("hall",
                gyroid_surface(freq=1.0, name="a", material="m"),
                Isosurface("schwarz_p", freq=1.0, name="b", material="m"))
    txt = _emit(room, Clock(t=0.0))
    assert "hall/a = isosurface" in txt, txt
    assert "hall/b = isosurface" in txt, txt
    # child names are restored after emit (no leakage)
    assert room.children[0].name == "a"
    assert room.children[1].name == "b"


def test_room_translation_moves_children():
    # a room translated by +5 in x moves an origin-placed child's box to x∈[4,6].
    room = Room("hall",
                gyroid_surface(freq=1.0, name="g", material="m",
                               bounds=((-1, -1, -1), (1, 1, 1))),
                frame=Affine.translation((5.0, 0.0, 0.0)))
    txt = _emit(room, Clock(t=0.0))
    assert "min 4 -1 -1" in txt, txt
    assert "max 6 1 1" in txt, txt
    # the coordinate frame shifts too: x -> (x-(5))
    assert "(x-(5))" in txt, txt


def test_room_child_placement_composes_with_frame():
    # child placed at (2,0,0) inside a room translated (5,0,0) -> world x=7.
    child = gyroid_surface(freq=1.0, name="g", material="m",
                           placement=(2.0, 0.0, 0.0),
                           bounds=((-1, -1, -1), (1, 1, 1)))
    room = Room("hall", child, frame=Affine.translation((5.0, 0.0, 0.0)))
    txt = _emit(room, Clock(t=0.0))
    assert "min 6 -1 -1" in txt, txt   # 7-1
    assert "max 8 1 1" in txt, txt     # 7+1
    assert "(x-(7))" in txt, txt


def test_room_frame_leaves_child_parent_unset():
    child = gyroid_surface(freq=1.0, name="g", material="m")
    room = Room("hall", child, frame=Affine.translation((3.0, 0.0, 0.0)))
    _emit(room, Clock(t=0.0))
    assert child._parent is None, "room must restore child _parent after emit"


def test_nested_rooms_stack_names_and_frames():
    inner = Room("inner",
                 gyroid_surface(freq=1.0, name="g", material="m",
                                bounds=((-1, -1, -1), (1, 1, 1))),
                 frame=Affine.translation((2.0, 0.0, 0.0)))
    outer = Room("outer", inner, frame=Affine.translation((5.0, 0.0, 0.0)))
    txt = _emit(outer, Clock(t=0.0))
    assert "outer/inner/g = isosurface" in txt, txt
    # composed translation 5+2 = 7
    assert "min 6 -1 -1" in txt, txt
    assert "max 8 1 1" in txt, txt


def test_room_animatable_frame_moves_over_time():
    room = Room("hall",
                gyroid_surface(freq=1.0, name="g", material="m"),
                frame=affine(3, [("move", vec(Sine(cycles=1, amp=4.0), 0.0, 0.0))]))
    a = _emit(room, Clock(t=0.1))
    b = _emit(room, Clock(t=0.6))
    assert a != b, "an animated room frame should move its children"


def test_room_in_scene_emits_and_checks_cycles():
    s = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0), res=(16, 16)))
    room = Room("hall",
                gyroid_surface(freq=Const(1.5), name="a", material="m"),
                gyroid_surface(freq=1.0, name="b", material="m",
                               placement=(3.0, 0.0, 0.0)),
                frame=affine(3, [("rot", 0, 2, Sine(cycles=1, amp=1.0))]))
    s.add(Material("m", "diffuse", reflect=0.7), room)
    s.check_cycles()  # must not raise
    txt = s.emit(Clock.at_frame(3, 24), Cache())
    assert "hall/a = isosurface" in txt and "hall/b = isosurface" in txt


def test_scene_check_cycles_ok():
    s = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0), res=(16, 16)))
    rot = rotations(3, [(0, 1, Sine(cycles=1, amp=1.0))])
    s.add(Material("m", "diffuse", reflect=0.7),
          gyroid_surface(freq=Const(1.5), rotation=rot,
                         drift=vec(phase_drift(2.0), 0.0, 0.0), material="m"))
    s.check_cycles()  # must not raise
    txt = s.emit(Clock.at_frame(3, 24), Cache())
    assert 'gyroid = isosurface' in txt


# ---------------------------------------------------------------------------
# N-D slice fields (a genuinely >=4-input field seen through an N-D rotation)
# ---------------------------------------------------------------------------

def _ndeval(expr: str, x: float, y: float, z: float) -> float:
    """Evaluate an emitted ftsl field expression numerically (the subset we emit —
    sin/cos, + - * and parens — is also valid Python)."""
    return eval(expr, {"__builtins__": {}},  # noqa: S307
                {"sin": math.sin, "cos": math.cos, "x": x, "y": y, "z": z})


def test_nd_templates_reduce_to_the_3d_forms():
    c = ("x", "y", "z")
    assert gyroid_n(c) == gyroid(*c)
    assert schwarz_p_n(c) == schwarz_p(*c)
    assert neovius_n(c) == neovius(*c)
    # D's terms are the same set, emitted in a different order
    assert sorted(schwarz_d_n(c).split("+")) == sorted(schwarz_d(*c).split("+"))


def test_nd_templates_scale_with_dimension():
    c4 = ("a", "b", "c", "d")
    assert gyroid_n(c4).count("sin(") == 4        # cyclic: one term per axis
    assert schwarz_p_n(c4).count("cos(") == 4
    assert schwarz_d_n(c4).count("*") == 8 * 3    # 2^(n-1) terms of n factors
    assert set(ND_FIELDS) == {"gyroid", "schwarz_p", "schwarz_d", "neovius"}


def test_slicefield_matches_direct_nd_evaluation():
    ang, w = 0.7, 1.3
    sf = SliceField("gyroid", dim=4, rotation=rotations(4, [(2, 3, ang)]),
                    offset=[0.0, 0.0, 0.0, w])
    expr = sf.build("x", "y", "z", EmitCtx(clock=Clock(t=0.0), cache=Cache()))
    px, py, pz = 0.31, -0.72, 1.11
    # reference: embed the slice point in 4-D, rotate it, evaluate the 4-D gyroid
    c, s = math.cos(ang), math.sin(ang)
    C = [px, py, c * pz - s * w, s * pz + c * w]
    ref = sum(math.sin(C[i]) * math.cos(C[(i + 1) % 4]) for i in range(4))
    # emitted coefficients carry ftsl_emit.fmt's 6 significant digits
    assert abs(_ndeval(expr, px, py, pz) - ref) < 1e-6


def test_slicefield_at_dim3_is_the_plain_field():
    sf = SliceField("gyroid", dim=3)
    expr = sf.build("x", "y", "z", EmitCtx(clock=Clock(t=0.0), cache=Cache()))
    for p in ((0.2, 0.4, 0.9), (-1.3, 2.2, 0.05)):
        assert abs(_ndeval(expr, *p) - _ndeval(gyroid("x", "y", "z"), *p)) < 1e-12


def test_slicefield_extra_axis_is_a_real_input():
    # The point of N-D: rotating in a plane that touches an axis OUTSIDE the slice
    # must change the surface, not merely tilt it.  A rotation confined to the slice
    # (0,1) at the same angle is an affine remap and preserves the value at a point
    # that the rotation fixes; the (2,3) one does not.
    ctx = EmitCtx(clock=Clock(t=0.0), cache=Cache())
    off = [0.0, 0.0, 0.0, 1.1]
    inside = SliceField("gyroid", dim=4, rotation=rotations(4, [(0, 1, 0.6)]),
                        offset=off).build("x", "y", "z", ctx)
    outside = SliceField("gyroid", dim=4, rotation=rotations(4, [(2, 3, 0.6)]),
                         offset=off).build("x", "y", "z", ctx)
    flat = SliceField("gyroid", dim=4, offset=off).build("x", "y", "z", ctx)
    p = (0.0, 0.0, 0.83)          # on the (0,1) rotation's fixed axis
    assert abs(_ndeval(inside, *p) - _ndeval(flat, *p)) < 1e-12
    assert abs(_ndeval(outside, *p) - _ndeval(flat, *p)) > 1e-3


def test_nd_grad_bound_is_conservative():
    # max_gradient must bound the real slope or the sphere-marcher tunnels.  The
    # per-component bounds are deliberately tight (gyroid's sqrt(2) is attained to
    # within 0.1%), so sample the *sliced* gradient densely in several dims.
    freq = 2.0
    for name in sorted(ND_FIELDS):
        for dim in (3, 4, 5):
            planes = [(1, dim - 1, 0.9), (0, 2, 0.4)]
            sf = SliceField(name, dim=dim, rotation=rotations(dim, planes),
                            offset=[0.0, 0.0, 0.0, 0.7, 1.3][:dim])
            expr = sf.build(*(f"({freq}*{v})" for v in "xyz"),
                            ctx=EmitCtx(clock=Clock(t=0.0), cache=Cache()))
            bound = sf.grad_bound(freq)
            assert abs(bound - nd_grad_bound(name, dim, freq)) < 1e-12
            h, worst = 1e-5, 0.0
            for k in range(200):
                p = [math.sin(k * 1.7) * 2.0, math.cos(k * 2.3) * 2.0,
                     math.sin(k * 0.9) * 2.0]
                g = []
                for ax in range(3):
                    q = list(p); q[ax] += h; hi = _ndeval(expr, *q)
                    q = list(p); q[ax] -= h; lo = _ndeval(expr, *q)
                    g.append((hi - lo) / (2 * h))
                worst = max(worst, math.sqrt(sum(c * c for c in g)))
            assert worst <= bound * (1 + 1e-6), (name, dim, worst, bound)


def test_nd_grad_bound_tracks_the_tightened_component_bounds():
    # Regression pins: these are the *proved* per-coordinate bounds (see the template
    # docstrings), not the naive term counts they replaced (2, and 2**(n-1)).
    assert abs(nd_grad_bound("gyroid", 4) - math.sqrt(2.0) * 2.0) < 1e-12
    assert abs(nd_grad_bound("schwarz_p", 4) - 2.0) < 1e-12
    assert abs(nd_grad_bound("schwarz_d", 5) - 4.0 * math.sqrt(5.0)) < 1e-12
    assert abs(nd_grad_bound("neovius", 3) - 7.0 * math.sqrt(3.0)) < 1e-12
    # freq and the pre-transform's largest singular value both scale it linearly
    assert abs(nd_grad_bound("gyroid", 4, freq=3.0, sigma=0.5)
               - 1.5 * math.sqrt(2.0) * 2.0) < 1e-12


def test_slicefield_animates_and_wraps_seamlessly():
    turn = rotations(4, [(0, 2, Sine(cycles=1, amp=math.pi)),
                         (1, 3, Sine(cycles=1, amp=math.pi))])
    iso = Isosurface(SliceField("gyroid", dim=4, rotation=turn,
                                offset=[0.0, 0.0, 0.0, 1.1]),
                     freq=1.0, drift=vec(phase_drift(1.0), 0.0, 0.0), material="m")
    a0 = _emit(iso, Clock.at_frame(0, 30))
    amid = _emit(iso, Clock.at_frame(11, 30))
    awrap = _emit(iso, Clock.at_frame(30, 30))
    assert a0 != amid, "an N-D rotation should morph the surface over the loop"
    assert a0 == awrap, "loop wrap must reproduce frame 0 exactly (seamless)"
    assert "+-" not in awrap and "*-" not in awrap


def test_slicefield_params_join_the_dag():
    turn = rotations(5, [(2, 4, Sine(cycles=1, amp=math.pi))])
    sf = SliceField("schwarz_p", dim=5, rotation=turn,
                    offset=[0.0, 0.0, 0.0, Const(0.5), 1.0])
    iso = Isosurface(sf, freq=1.0, material="m", name="nd")
    roots = iso.roots()
    assert any(r is sf.offset for r in roots), "slice offset must be a DAG root"
    assert len(roots) >= 25, "the 5x5 rotation entries must be DAG roots too"
    s = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0), res=(16, 16)))
    s.add(Material("m", "diffuse", reflect=0.7), iso)
    s.check_cycles()
    assert "nd = isosurface" in s.emit(Clock.at_frame(3, 24), Cache())


def test_slicefield_rejects_bad_shapes():
    for kw in (dict(dim=2), dict(dim=4, rotation=rotations(3, [(0, 1, 0.2)])),
               dict(dim=4, offset=[0.0, 0.0, 0.0])):
        try:
            SliceField("gyroid", **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"SliceField({kw}) should have been rejected")
    try:
        SliceField("nope", dim=4)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown N-D field name should raise")


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
