"""The checker pattern puts a surface lying on a cell boundary in one cell.

A floor at y = 0 sits where every sine of the checker is 0: without the tie-break its value
there is 0.5, and each hit point's rounding error tips it to 0 or 1, which the GPU averages to
grey. The emitted formula is evaluated here with Python's own sin and sign.
"""
import math

import loom.material as material


def _eval(expr, x, y, z):
    env = {'sin': math.sin, 'sign': lambda v: float((v > 0) - (v < 0))}
    return eval(expr, env, {'x': x, 'y': y, 'z': z})


def _cells(expr, plane):
    return {_eval(expr, 0.4, plane + err, 1.1) for err in (0.0, 1e-12, -1e-12, 1e-7, -1e-7)}


def test_a_surface_on_a_cell_boundary_sits_in_one_cell():
    e = material.checker('x', 'y', 'z')
    for plane in (0.0, math.pi, -2.0 * math.pi):
        cells = _cells(e, plane)
        assert len(cells) == 1 and cells <= {0.0, 1.0}, (plane, cells)


def test_without_the_tie_break_the_boundary_is_ambiguous(monkeypatch):
    # The failure the tie-break exists for, so this test cannot pass vacuously.
    monkeypatch.setattr(material, 'CHECKER_TIE_BREAK', 0.0)
    assert len(_cells(material.checker('x', 'y', 'z'), 0.0)) > 1
