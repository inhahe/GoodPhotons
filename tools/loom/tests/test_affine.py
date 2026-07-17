"""M8 tests: affine composition (rotation/shear + translation folding).

An arbitrarily long chain of N-D Givens rotations and translations must fold
into one ``(linear, offset)`` affine that reproduces the sequential application
exactly.  Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Cache, vec, VecSignal,
    Mat, rotation, rotations, Affine, affine, Sine,
)


def _v(vs, clock=None):
    return tuple(round(x, 12) for x in vs.at(clock or Clock(t=0.0), Cache()))


def _close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_apply_is_linear_plus_offset():
    a = Affine(rotation(3, 0, 1, 0.7), [1.0, 2.0, 3.0])
    x = vec(1.0, 0.0, 0.0)
    got = _v(a.apply(x))
    # manual: R@x + t
    c, s = math.cos(0.7), math.sin(0.7)
    manual = (c * 1 + 1.0, s * 1 + 2.0, 0.0 + 3.0)
    assert _close(got, manual), (got, manual)


def test_composition_matches_sequential():
    A = Affine(rotation(3, 0, 1, 0.5), [1.0, 0.0, 0.0])
    B = Affine(rotation(3, 1, 2, -0.3), [0.0, 2.0, 0.0])
    x = vec(0.3, 0.7, -0.2)
    # (A ∘ B)(x) == A(B(x))
    composed = _v((A.compose(B)).apply(x))
    stepwise = _v(A.apply(B.apply(x)))
    assert _close(composed, stepwise), (composed, stepwise)


def test_matmul_operator_composes_and_applies():
    A = Affine.rotation(3, 0, 1, 0.4)
    B = Affine.translation([1.0, -1.0, 2.0])
    x = vec(1.0, 1.0, 1.0)
    assert _close(_v((A @ B).apply(x)), _v(A.apply(B.apply(x))))
    assert _close(_v(A @ x), _v(A.apply(x)))


def test_associativity():
    A = Affine.rotation(3, 0, 1, 0.2)
    B = Affine.translation([0.5, 0.0, -0.5])
    C = Affine.rotation(3, 1, 2, 0.9)
    x = vec(0.1, -0.4, 0.6)
    left = _v(((A.compose(B)).compose(C)).apply(x))
    right = _v((A.compose(B.compose(C))).apply(x))
    assert _close(left, right), (left, right)


def test_builder_first_listed_applied_first():
    # ops applied left-to-right: rot then move then rot
    ops = [("rot", 0, 1, 0.5), ("move", [1.0, 0.0, 0.0]), ("rot", 1, 2, 0.3)]
    folded = affine(3, ops)
    # manual sequential build (each step applied after the previous)
    R1 = Affine.rotation(3, 0, 1, 0.5)
    T = Affine.translation([1.0, 0.0, 0.0])
    R2 = Affine.rotation(3, 1, 2, 0.3)
    manual = R2.compose(T.compose(R1))
    x = vec(0.7, -0.2, 0.4)
    assert _close(_v(folded.apply(x)), _v(manual.apply(x)))


def test_rotation_only_affine_matches_rotations():
    ops = [("rot", 0, 1, 0.6), ("rot", 1, 2, -0.5)]
    folded = affine(3, ops)
    ref = rotations(3, [(0, 1, 0.6), (1, 2, -0.5)])
    x = vec(0.9, 0.1, -0.3)
    assert _close(_v(folded.apply(x)), _v(ref.apply(x)))
    # offset stays zero for a pure-rotation chain
    assert _close(_v(folded.offset), (0.0, 0.0, 0.0))


def test_translations_do_not_commute_with_rotation():
    x = vec(1.0, 0.0, 0.0)
    rot_then_move = affine(3, [("rot", 0, 1, math.pi / 2), ("move", [1.0, 0.0, 0.0])])
    move_then_rot = affine(3, [("move", [1.0, 0.0, 0.0]), ("rot", 0, 1, math.pi / 2)])
    a = _v(rot_then_move.apply(x))
    b = _v(move_then_rot.apply(x))
    assert not _close(a, b), "translation and rotation must not commute"


def test_animated_affine_evaluates_per_frame():
    # a Signal-driven angle => the folded affine animates
    a = affine(3, [("rot", 0, 1, Sine(cycles=1.0)), ("move", [1.0, 0.0, 0.0])])
    x = vec(1.0, 0.0, 0.0)
    p0 = _v(a.apply(x), Clock(t=0.0))
    p1 = _v(a.apply(x), Clock(t=0.25))
    assert not _close(p0, p1), "an animated affine must change across time"


def test_builder_rejects_bad_op():
    try:
        affine(3, [("spin", 0, 1, 0.5)])
    except ValueError:
        return
    raise AssertionError("affine() must reject an unknown op")


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
