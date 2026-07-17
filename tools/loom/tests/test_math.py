"""M2 tests: N-D matrices, Givens rotations, the slicer, and the honest
affine-vs-morph behavior of rotating a field in N-D and slicing back to 3-D.

Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Clock, Sin, Cos, vec, VecSignal,
    Mat, rotation, rotations, slice3, detect_signal_cycle,
)


def _clk(t: float = 0.0) -> Clock:
    return Clock(t=t, frame=0, frames=1, fps=30.0)


# ---- fields ---------------------------------------------------------------

def gyroid3(p: VecSignal):
    """3-input gyroid: uses the first 3 components of p."""
    x, y, z = p[0], p[1], p[2]
    return Sin(x) * Cos(y) + Sin(y) * Cos(z) + Sin(z) * Cos(x)


def gyroid4(p: VecSignal):
    """4-input gyroid: adds a genuine dependence on the 4th component."""
    x, y, z, w = p[0], p[1], p[2], p[3]
    return gyroid3(p) + Sin(w) * Cos(x) + Sin(x) * Cos(w)


# ---- tests ----------------------------------------------------------------

def test_givens_rotates_basis():
    R = rotation(2, 0, 1, math.pi / 2)
    out = R.apply(vec(1.0, 0.0)).at(_clk())
    assert abs(out[0]) < 1e-12 and abs(out[1] - 1.0) < 1e-12, out


def test_rotation_preserves_length():
    R = rotations(4, [(0, 1, 0.7), (1, 3, -1.2), (0, 2, 0.35)])
    v = vec(1.0, -2.0, 3.0, 0.5)
    before = v.length().at(_clk())
    after = R.apply(v).length().at(_clk())
    assert abs(before - after) < 1e-9, (before, after)


def test_matmul_matches_manual():
    A = Mat([[1.0, 2.0], [3.0, 4.0]])
    B = Mat([[5.0, 6.0], [7.0, 8.0]])
    C = (A @ B)
    got = [[C.rows[i][j].at(_clk()) for j in range(2)] for i in range(2)]
    assert got == [[19.0, 22.0], [43.0, 50.0]], got


def test_slice3_maps_correctly():
    q = slice3(vec(1.0, 0.0, 0.0, 0.0),
               vec(0.0, 1.0, 0.0, 0.0),
               vec(0.0, 0.0, 1.0, 0.0),
               vec(0.0, 0.0, 0.0, 1.0))
    p = q(2.0, 3.0, 4.0).at(_clk())
    assert p == (1.0, 2.0, 3.0, 4.0), p


def test_3input_gyroid_under_ND_rotation_is_affine():
    """Rotating a 3-input gyroid in 5-D and slicing back to 3-D is exactly an
    affine remap of (x,y,z): value == gyroid3(A@xyz + d)."""
    R = rotations(5, [(0, 3, 0.6), (1, 4, -0.9), (2, 3, 0.4), (0, 1, 0.2)])
    # rotated axes = first three columns of R; anchor with nonzero extra dims.
    cols = [[R.rows[r][c].at(_clk()) for r in range(5)] for c in range(5)]
    u, v, w = cols[0], cols[1], cols[2]
    O = [0.0, 0.0, 0.0, 0.7, -0.4]
    slicer = slice3(vec(*O), vec(*u), vec(*v), vec(*w))
    field = gyroid3  # sees only first 3 comps

    # explicit affine derived from the slicer's first three components
    A = [[u[k], v[k], w[k]] for k in range(3)]  # 3x3, columns u,v,w (first 3 rows)
    d = [O[0], O[1], O[2]]

    rng = random.Random(1)
    maxerr = 0.0
    for _ in range(200):
        x, y, z = (rng.uniform(-3, 3) for _ in range(3))
        sliced_val = field(slicer(x, y, z)).at(_clk())
        ax = [A[r][0] * x + A[r][1] * y + A[r][2] * z + d[r] for r in range(3)]
        affine_val = gyroid3(vec(*ax)).at(_clk())
        maxerr = max(maxerr, abs(sliced_val - affine_val))
    assert maxerr < 1e-9, f"3-input field is not affine under the slice: {maxerr}"


def test_4input_gyroid_morphs_but_3input_does_not():
    """Drifting the 4th slice coordinate morphs a genuine 4-input field, while a
    3-input field is invariant to it."""
    xyz = [(x * 0.5, y * 0.5, z * 0.5)
           for x in range(-3, 4) for y in range(-3, 4) for z in range(-3, 4)]

    def field_change(field_fn, dim, w0a, w0b):
        maxd = 0.0
        for (x, y, z) in xyz:
            sa = slice3(vec(0, 0, 0, w0a), vec(1, 0, 0, 0),
                        vec(0, 1, 0, 0), vec(0, 0, 1, 0))
            sb = slice3(vec(0, 0, 0, w0b), vec(1, 0, 0, 0),
                        vec(0, 1, 0, 0), vec(0, 0, 1, 0))
            va = field_fn(sa(x, y, z)).at(_clk())
            vb = field_fn(sb(x, y, z)).at(_clk())
            maxd = max(maxd, abs(va - vb))
        return maxd

    morph = field_change(gyroid4, 4, 0.0, 1.3)
    frozen = field_change(gyroid3, 4, 0.0, 1.3)
    assert morph > 0.3, f"4-input field should morph, got {morph}"
    assert frozen < 1e-9, f"3-input field must ignore the 4th coord, got {frozen}"


def test_slicer_graph_is_acyclic():
    R = rotations(4, [(0, 1, 0.3), (2, 3, 0.5)])
    slicer = slice3(vec(0, 0, 0, 0),
                    R.apply(vec(1, 0, 0, 0)),
                    R.apply(vec(0, 1, 0, 0)),
                    R.apply(vec(0, 0, 1, 0)))
    detect_signal_cycle(gyroid4(slicer(1.0, 2.0, 3.0)))


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
