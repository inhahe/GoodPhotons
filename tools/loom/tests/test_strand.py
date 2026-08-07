"""Strand tests: loom -> ftrace ``curve`` primitive (FTSL §8.6).

The load-bearing claim is that the control points loom emits, read back under the
basis loom names, reproduce the spine loom sampled -- and that a closed spine comes
back *seamless*.  So these tests re-implement ``src/curve.h``'s two span evaluators
exactly (same clamping, same basis matrices) and check the round trip, rather than
just eyeballing the emitted text.

Runnable directly or under pytest.
"""

from __future__ import annotations

import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Const, PointPath, LoopCurve, Clock, Cache, Strand, strand, hair, vec,
    closed_bspline_controls, strand_controls,
)
from loom.interp import eval_curve  # noqa: E402


# ---------------------------------------------------------------------------
# Reference: src/curve.h's flattener, in Python
# ---------------------------------------------------------------------------

def _span_points(pts, basis, sp):
    """``curve.h``'s ``spanPoints`` -- including the end clamp ``at(i) = pts[clamp(i)]``
    that is exactly why Catmull-Rom cannot close a loop."""
    n = len(pts)

    def at(i):
        return pts[0 if i < 0 else (n - 1 if i >= n else i)]

    if basis == "catmull_rom":
        return at(sp - 1), at(sp), at(sp + 1), at(sp + 2)
    if basis == "bspline":
        return at(sp), at(sp + 1), at(sp + 2), at(sp + 3)
    raise ValueError(basis)


def _eval_span(basis, P, u):
    """``curve.h``'s ``evalSpan``."""
    P0, P1, P2, P3 = P
    u2, u3 = u * u, u * u * u
    d = len(P0)
    if basis == "catmull_rom":
        return tuple(0.5 * (2 * P1[k]
                            + (P2[k] - P0[k]) * u
                            + (2 * P0[k] - 5 * P1[k] + 4 * P2[k] - P3[k]) * u2
                            + (3 * P1[k] - P0[k] - 3 * P2[k] + P3[k]) * u3)
                     for k in range(d))
    return tuple(((-P0[k] + 3 * P1[k] - 3 * P2[k] + P3[k]) * u3
                  + (3 * P0[k] - 6 * P1[k] + 3 * P2[k]) * u2
                  + (P2[k] - P0[k]) * 3 * u
                  + (P0[k] + 4 * P1[k] + P2[k])) / 6.0
                 for k in range(d))


def _span_count(basis, n):
    return n - 1 if basis == "catmull_rom" else n - 3


def _curve_point(pts, basis, sp, u):
    return _eval_span(basis, _span_points(pts, basis, sp), u)


def _curve_tangent(pts, basis, sp, u, h=1e-6):
    a = _curve_point(pts, basis, sp, u - h)
    b = _curve_point(pts, basis, sp, u + h)
    return tuple((b[k] - a[k]) / (2 * h) for k in range(len(a)))


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _wiggly_ring(n=7, r=1.0):
    """A closed, deliberately non-circular control polygon (so a shrinking or
    mis-phased reconstruction cannot pass by symmetry)."""
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * (1.0 + 0.35 * math.sin(3 * a))
        out.append(vec(rr * math.cos(a), 0.4 * math.sin(2 * a), rr * math.sin(a)))
    return out


def _spine(closed=True, n=7):
    return LoopCurve(PointPath(_wiggly_ring(n), closed=closed), Const(0.0))


def _emit(el):
    from loom.ftsl_emit import EmitCtx
    return el.emit(EmitCtx(clock=Clock(t=0.0, frame=0, frames=1, fps=1.0), cache=Cache()))


def _points_of(text):
    """Parse the ``point x y z [r=…]`` statements back out of an emitted block."""
    pts, radii = [], []
    for line in text.splitlines():
        w = line.strip().split()
        if not w or w[0] != "point":
            continue
        pts.append(tuple(float(x) for x in w[1:4]))
        r = [t for t in w[4:] if t.startswith("r=")]
        radii.append(float(r[0][2:]) if r else None)
    return pts, radii


def _key(text, name):
    for line in text.splitlines():
        w = line.strip().split()
        if w and w[0] == name:
            return w[1]
    return None


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------

def test_closed_bspline_controls_interpolate():
    """The solved control points must put span ``i``'s start exactly on sample ``i``."""
    for n in (3, 4, 5, 17, 64):
        S = [(math.cos(2 * math.pi * i / n) * (1 + 0.3 * math.sin(5 * i)),
              math.sin(3 * i) * 0.7,
              math.sin(2 * math.pi * i / n)) for i in range(n)]
        C = closed_bspline_controls(S)
        assert len(C) == n
        Q = C + C[:3]
        assert _span_count("bspline", len(Q)) == n, "wrap by 3 must give exactly n spans"
        for sp in range(n):
            got = _curve_point(Q, "bspline", sp, 0.0)
            assert _dist(got, S[sp]) < 1e-9, (n, sp, got, S[sp])


def test_closed_bspline_never_clamps():
    """Every index the flattener touches must be in range -- a clamped index is the
    seam defect this construction exists to avoid."""
    n = 9
    S = [(float(i), float(i * i % 5), 0.0) for i in range(n)]
    Q = closed_bspline_controls(S)
    Q = Q + Q[:3]
    for sp in range(n):
        for i in (sp, sp + 1, sp + 2, sp + 3):
            assert 0 <= i < len(Q), (sp, i)


def test_closed_strand_seam_is_smooth():
    """Tangent leaving the last span == tangent entering the first.  Catmull-Rom on
    the same samples must FAIL this -- otherwise the test proves nothing."""
    S = [tuple(p) for p in
         [(math.cos(a), 0.3 * math.sin(2 * a), math.sin(a))
          for a in [2 * math.pi * i / 11 for i in range(11)]]]
    n = len(S)

    basis, Q, _ = strand_controls(S, closed=True)
    assert basis == "bspline"
    t_end = _curve_tangent(Q, basis, n - 1, 1.0 - 1e-3)
    t_start = _curve_tangent(Q, basis, 0, 1e-3)
    seam = _dist(t_end, t_start)
    # an interior joint, for scale
    inner = _dist(_curve_tangent(Q, basis, 3, 1.0 - 1e-3),
                  _curve_tangent(Q, basis, 4, 1e-3))
    assert seam < inner + 1e-6, f"seam kink {seam} vs interior {inner}"

    # The control: the naive closed Catmull-Rom (points wrapped once) DOES kink,
    # because at(-1) clamps to the first point instead of the one before it.
    CR = S + [S[0]]
    m = _span_count("catmull_rom", len(CR))
    cr_seam = _dist(_curve_tangent(CR, "catmull_rom", m - 1, 1.0 - 1e-3),
                    _curve_tangent(CR, "catmull_rom", 0, 1e-3))
    assert cr_seam > 1e-2, "the Catmull-Rom control should visibly kink at the seam"


def test_closed_strand_actually_closes():
    """Position AND radius: the last span's end must be the first span's start, so the
    round-cone chain ftrace builds is one closed ring rather than a ring with a gap."""
    n = 13
    prof = lambda u: 1.0 + 0.4 * math.sin(2 * math.pi * u)      # noqa: E731
    el = Strand(_spine(True, 9), count=n, radius=0.02, radius_profile=prof, name="c")
    pts, radii = _points_of(_emit(el))
    end = _curve_point(pts, "bspline", n - 1, 1.0)
    start = _curve_point(pts, "bspline", 0, 0.0)
    assert _dist(end, start) < 1e-9, (end, start)
    # curve.h: span sp's radii are rat(sp+1) -> rat(sp+2)
    assert abs(radii[n - 1 + 2] - radii[0 + 1]) < 1e-12


def test_open_strand_interpolates_samples():
    S = [(0.0, 0.0, 0.0), (0.1, 0.5, 0.0), (-0.2, 1.0, 0.1), (0.0, 1.4, 0.3),
         (0.3, 1.7, 0.2)]
    basis, Q, rad_u = strand_controls(S, closed=False)
    assert basis == "catmull_rom"
    assert Q == S, "an interpolating basis needs no solve -- the samples ARE the points"
    assert rad_u == [0.0, 0.25, 0.5, 0.75, 1.0]
    for sp in range(_span_count(basis, len(Q))):
        assert _dist(_curve_point(Q, basis, sp, 0.0), S[sp]) < 1e-12
    assert _dist(_curve_point(Q, basis, len(Q) - 2, 1.0), S[-1]) < 1e-12


def test_strand_controls_rejects_too_few():
    for bad in ([], [(0.0, 0.0, 0.0)]):
        try:
            strand_controls(bad, closed=False)
            assert False, "expected ValueError"
        except ValueError:
            pass
    try:
        strand_controls([(0.0, 0, 0), (1.0, 0, 0)], closed=True)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# eval_curve's open-path clamp (what lets an open strand reach its tip)
# ---------------------------------------------------------------------------

def test_eval_curve_open_clamps_instead_of_wrapping():
    pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0)]
    start = eval_curve(pts, 0.0, False)
    end = eval_curve(pts, 1.0, False)
    assert _dist(end, start) > 0.5, "u=1 must be the far END, not wrapped to the start"
    # the open curve runs mid(p0,p1) -> mid(p_{n-2},p_{n-1})
    assert _dist(end, (2.5, 0.5)) < 1e-12
    assert _dist(eval_curve(pts, 2.0, False), end) < 1e-12       # clamped above
    assert _dist(eval_curve(pts, -1.0, False), start) < 1e-12    # and below
    # a CLOSED curve still wraps -- that is what a loop means
    assert _dist(eval_curve(pts, 1.0, True), eval_curve(pts, 0.0, True)) < 1e-12


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def test_emit_closed_block_shape():
    n = 24
    el = Strand(_spine(True), count=n, radius=0.05, material="fibre", name="loop")
    txt = _emit(el)
    assert txt.startswith('curve "loop" {') and txt.rstrip().endswith("}")
    assert _key(txt, "basis") == "bspline"
    assert _key(txt, "material") == '"fibre"'
    assert _key(txt, "radius") == "0.05"
    assert "radius_tip" not in txt
    pts, radii = _points_of(txt)
    assert len(pts) == n + 3, "closed = wrap by 3"
    assert all(r is not None for r in radii), "closed always pins per-point radii"
    assert all(abs(r - 0.05) < 1e-12 for r in radii)


def test_emit_closed_curve_passes_through_the_spine():
    """End to end: sample the spine, emit, read the points back, flatten them with
    curve.h's own B-spline -- the strand must land on the spine."""
    n = 32
    sp_curve = _spine(True)
    el = Strand(sp_curve, count=n, radius=0.02, name="loop")
    pts, _ = _points_of(_emit(el))
    clk, cache = Clock(t=0.0, frame=0, frames=1, fps=1.0), Cache()
    for k in range(n):
        want = sp_curve.sample(k / n, clk, cache)
        got = _curve_point(pts, "bspline", k, 0.0)
        assert _dist(got, want) < 1e-5, (k, got, want)   # %.6g text precision


def test_emit_open_uses_catmull_rom_and_the_loader_taper():
    el = hair(_spine(False, 6), radius=0.004, tip=0.0002, count=12, name="h")
    txt = _emit(el)
    assert _key(txt, "basis") == "catmull_rom"
    assert _key(txt, "radius") == "0.004"
    assert _key(txt, "radius_tip") == "0.0002"
    pts, radii = _points_of(txt)
    assert len(pts) == 12
    assert all(r is None for r in radii), \
        "a plain root->tip taper is index-linear, so let the loader do it"


def test_radius_profile_registers_to_the_geometry():
    """A per-point ``r=`` has to land on the control point whose SPAN ENDPOINT is the
    sample it describes -- B-spline span sp takes rat(sp+1)/rat(sp+2), so the profile
    is offset by one in the wrapped list.  Off-by-one here slides the fiber's bulge
    around the loop, which is invisible on a circle and obvious on anything else."""
    n = 16

    def prof(u):
        return 1.0 + 0.5 * math.sin(2 * math.pi * u)      # periodic: closes cleanly

    el = Strand(_spine(True), count=n, radius=0.03, radius_profile=prof, name="b")
    _, radii = _points_of(_emit(el))
    for sp in range(n):
        # span sp starts at sample sp (u = sp/n); its start radius is control point sp+1
        want = 0.03 * prof(sp / n)
        assert abs(radii[sp + 1] - want) < 1e-5 * max(want, 1e-9), \
            (sp, radii[sp + 1], want)


def test_open_radius_profile_is_per_point():
    def prof(u):
        return 1.0 + u

    el = Strand(_spine(False, 6), count=8, radius=0.01, radius_profile=prof,
                closed_spine=False, name="p")
    txt = _emit(el)
    assert "radius_tip" not in txt
    _, radii = _points_of(txt)
    for k in range(8):
        want = 0.01 * prof(k / 7)
        assert abs(radii[k] - want) < 1e-5 * want   # %.6g == ~6 significant digits


def test_per_point_radius_uses_key_equals_val():
    """FTSL's statement splitter starts a new statement at a bareword, so ``r 0.002``
    would silently truncate the point.  It must be ``r=0.002``."""
    el = Strand(_spine(True), count=8, radius=0.01, name="k")
    for line in _emit(el).splitlines():
        w = line.strip().split()
        if w and w[0] == "point":
            assert len(w) == 5 and w[4].startswith("r=") and "=" in w[4], line


# ---------------------------------------------------------------------------
# The structural-loop guard
# ---------------------------------------------------------------------------

def _warnings_from(el):
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        _emit(el)
    return [str(w.message) for w in got]


def test_closed_strand_warns_when_the_radius_does_not_close():
    el = Strand(_spine(True), count=12, radius=0.05, radius_tip=0.01, name="taper")
    msgs = _warnings_from(el)
    assert any("does not close" in m for m in msgs), msgs


def test_closed_strand_warns_on_a_non_periodic_profile():
    el = Strand(_spine(True), count=12, radius=0.05,
                radius_profile=lambda u: 1.0 + u, name="ramp")
    assert any("does not close" in m for m in _warnings_from(el))


def test_closed_strand_is_silent_when_it_closes():
    assert _warnings_from(Strand(_spine(True), count=12, radius=0.05, name="ok")) == []
    per = Strand(_spine(True), count=12, radius=0.05,
                 radius_profile=lambda u: 1.0 + 0.4 * math.sin(4 * math.pi * u),
                 name="per")
    assert _warnings_from(per) == []


def test_open_strand_never_warns_about_taper():
    el = Strand(_spine(False, 6), count=12, radius=0.05, radius_tip=0.001,
                closed_spine=False, name="open")
    assert _warnings_from(el) == []


def test_warns_only_once():
    el = Strand(_spine(True), count=12, radius=0.05, radius_tip=0.01, name="once")
    assert len(_warnings_from(el)) == 1
    assert _warnings_from(el) == []


def test_count_is_validated_at_construction():
    for kw in ({"count": 2, "closed_spine": True}, {"count": 1, "closed_spine": False}):
        try:
            Strand(_spine(True), **kw)
            assert False, f"expected ValueError for {kw}"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Factories / plumbing
# ---------------------------------------------------------------------------

def test_strand_factory_matches_the_class():
    a = _emit(strand(_spine(True), radius=0.03, count=10, material="m", name="s"))
    b = _emit(Strand(_spine(True), radius=0.03, count=10, material="m", name="s"))
    assert a == b


def test_pointpath_is_accepted_directly():
    el = Strand(PointPath(_wiggly_ring(7), closed=True), count=10, name="pp")
    assert isinstance(el.spine, LoopCurve)
    assert 'basis bspline' in _emit(el)


def test_roots_include_the_spine_and_animated_radii():
    from loom import LoopNoise
    r = 0.02 + LoopNoise(cells=4, seed=1, freq=1, amp=0.005)
    el = Strand(_spine(True), count=10, radius=r, name="anim")
    roots = el.roots()
    assert el.spine in roots and r in roots


def test_emitted_curve_round_trips_through_the_reader():
    """loom's own .ftsl reader must accept what it writes -- in particular the repeated
    ``point`` statements and the ``r=`` value continuation."""
    from loom.ftsl_emit import EmitCtx
    from loom.grammar.reader import parse_document
    txt = _emit(Strand(_spine(True), count=12, radius=0.02, name="rt")) + "\n"
    ctx = EmitCtx(clock=Clock(t=0.0, frame=0, frames=1, fps=1.0), cache=Cache())
    doc = parse_document(txt)
    assert len(doc.blocks("curve")) == 1
    assert doc.emit(ctx) == txt


def test_viewer_sidecar_describes_a_strand():
    from loom.viewer import _describe_element
    el = Strand(_spine(True), count=10, radius=0.02, name="v")
    rec = _describe_element(el, 0, {}, Clock(t=0.0, frame=0, frames=1, fps=1.0))
    assert rec["kind"] == "strand" and rec["count"] == 10
    g = rec["strand"]
    assert len(g["points"]) == 10 and len(g["radii"]) == 10
    assert g["closed"] is True and g["basis"] == "bspline"


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
