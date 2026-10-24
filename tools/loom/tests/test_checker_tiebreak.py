"""Both checker patterns put a surface lying on a cell boundary in one cell.

A floor at y = 0 sits where every sine of the checker is 0: without the tie-break its value
there is 0.5, and each hit point's rounding error tips it to 0 or 1, which the GPU averages to
grey. The emitted ftsl formula is evaluated here with Python's own sin and sign.
"""
import math

import loom.material as material
import loom.spatial as spatial


def _eval(expr, x, y, z):
    env = {'sin': math.sin, 'sign': lambda v: float((v > 0) - (v < 0))}
    return eval(expr, env, {'x': x, 'y': y, 'z': z})


def _cells(expr, plane, freq=1.0):
    return {_eval(expr, 0.4 / freq, (plane + err) / freq, 1.1 / freq)
            for err in (0.0, 1e-12, -1e-12, 1e-7, -1e-7)}


def _material():
    return material.checker('x', 'y', 'z')


def _spatial(freq):
    return spatial.checker(freq).emit(('x', 'y', 'z'), None)


def test_a_surface_on_a_cell_boundary_sits_in_one_cell():
    for plane in (0.0, math.pi, -2.0 * math.pi):
        for label, expr, freq in (('material', _material(), 1.0), ('spatial', _spatial(2.0), 2.0)):
            cells = _cells(expr, plane, freq)
            assert len(cells) == 1 and cells <= {0.0, 1.0}, (label, plane, cells)


def test_without_the_tie_break_the_boundary_is_ambiguous(monkeypatch):
    # The failure the tie-break exists for, so the test above cannot pass vacuously.
    monkeypatch.setattr(material, 'CHECKER_TIE_BREAK', 0.0)
    monkeypatch.setattr(spatial, 'CHECKER_TIE_BREAK', 0.0)
    assert len(_cells(_material(), 0.0)) > 1
    assert len(_cells(_spatial(2.0), 0.0, 2.0)) > 1
