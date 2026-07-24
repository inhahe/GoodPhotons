"""E2 (slice 1) tests: loom.anim — the N-D curve → scene-variable go-between.

Runnable directly (``python tests/test_anim.py``) or under pytest.  Covers the
channel-a config model + JSON sidecar round-trip, Catmull-Rom sampling, and the
channel-b binding fan-out (which reuses the E5 pin/mod influence model).
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom.anim import (  # noqa: E402
    CurveDrive, ChannelBinding, MODE_FLYBY, MODE_ANIMATION, SIDECAR_VERSION,
)
from loom.axes import ADDITIVE, GAIN, BIPOLAR  # noqa: E402


def _drive():
    pts = [(0.0, 0.0, 0.0, 0.0),
           (1.0, 2.0, 0.0, 0.5),
           (2.0, 0.0, 1.0, 1.0),
           (3.0, -2.0, 0.0, 0.5)]
    binds = [ChannelBinding(0, "surface.threshold", "pin", 1.0, ADDITIVE),
             ChannelBinding(3, "material.roughness", "mod", 1.0, GAIN)]
    return CurveDrive(4, pts, binds, mode=MODE_ANIMATION, name="demo")


# ---- construction / validation --------------------------------------------

def test_construct_and_targets():
    d = _drive()
    assert d.dims == 4
    assert d.targets() == ["surface.threshold", "material.roughness"]


def test_point_dim_mismatch_rejected():
    with pytest.raises(ValueError):
        CurveDrive(3, [(0, 0, 0), (1, 1)])          # 2nd point wrong length


def test_channel_out_of_range_rejected():
    with pytest.raises(ValueError):
        CurveDrive(2, [(0, 0), (1, 1)], [ChannelBinding(5, "x")])


def test_needs_two_points():
    with pytest.raises(ValueError):
        CurveDrive(2, [(0, 0)])


def test_binding_validates_mode_and_kind():
    with pytest.raises(ValueError):
        ChannelBinding(0, "x", mode="blend")
    with pytest.raises(ValueError):
        ChannelBinding(0, "x", kind="weird")


# ---- sidecar round-trip ----------------------------------------------------

def test_dict_roundtrip_is_faithful():
    d = _drive()
    back = CurveDrive.from_dict(d.to_dict())
    assert back.dims == d.dims
    assert back.points == d.points
    assert back.mode == d.mode
    assert [b.to_dict() for b in back.bindings] == [b.to_dict() for b in d.bindings]


def test_save_load_sidecar_atomic():
    d = _drive()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "drive.json")
        d.save(p)
        # no leftover temp files from the atomic write
        assert os.listdir(td) == ["drive.json"]
        back = CurveDrive.load(p)
    assert back.to_dict() == d.to_dict()


def test_sidecar_version_mismatch_rejected():
    d = _drive().to_dict()
    d["version"] = SIDECAR_VERSION + 99
    with pytest.raises(ValueError):
        CurveDrive.from_dict(d)


# ---- sampling --------------------------------------------------------------

def test_sample_hits_control_points_at_knots():
    d = _drive()
    # uniform Catmull-Rom interpolates its control points at the segment knots
    assert _close(d.sample(0.0), (0.0, 0.0, 0.0, 0.0))
    assert _close(d.sample(1.0), (3.0, -2.0, 0.0, 0.5))
    knot = 1.0 / 3.0                                   # 4 points -> 3 open segments
    assert _close(d.sample(knot), (1.0, 2.0, 0.0, 0.5))


def test_sample_two_point_is_linear():
    d = CurveDrive(2, [(0.0, 0.0), (10.0, -4.0)])
    assert _close(d.sample(0.5), (5.0, -2.0))


def test_sample_closed_wraps():
    d = CurveDrive(1, [(0.0,), (1.0,), (2.0,), (1.0,)], closed=True)
    # closed: t=1 wraps back to t=0
    assert _close(d.sample(0.0), d.sample(1.0))


def test_sample_clamps_out_of_range():
    d = _drive()
    assert _close(d.sample(-1.0), d.sample(0.0))
    assert _close(d.sample(2.0), d.sample(1.0))


# ---- binding fan-out (channel b) ------------------------------------------

def test_apply_pin_and_mod_kinds():
    d = _drive()
    # channel 0 -> threshold (pin additive, base neutral 0 -> just the value)
    # channel 3 -> roughness (mod gain, neutral 1 * value**1)
    out = d.apply([0.7, 9.0, 9.0, 0.25])
    assert abs(out["surface.threshold"] - 0.7) < 1e-12
    assert abs(out["material.roughness"] - 0.25) < 1e-12


def test_apply_multiple_channels_one_target():
    pts = [(0.0, 0.0), (1.0, 1.0)]
    binds = [ChannelBinding(0, "elev", "mod", 1.0, ADDITIVE),
             ChannelBinding(1, "elev", "mod", 0.5, ADDITIVE)]
    d = CurveDrive(2, pts, binds)
    # additive neutral 0 + 1*2 + 0.5*4 = 4
    assert abs(d.apply([2.0, 4.0])["elev"] - 4.0) < 1e-12


def test_apply_base_offset():
    binds = [ChannelBinding(0, "y", "mod", 1.0, ADDITIVE)]
    d = CurveDrive(1, [(0.0,), (1.0,)], binds)
    assert abs(d.apply([3.0], bases={"y": 10.0})["y"] - 13.0) < 1e-12


def test_frame_samples_then_applies():
    binds = [ChannelBinding(0, "x", "pin", 1.0, ADDITIVE)]
    d = CurveDrive(2, [(0.0, 0.0), (10.0, 0.0)], binds)
    assert abs(d.frame(0.5)["x"] - 5.0) < 1e-12


def test_bipolar_channel():
    binds = [ChannelBinding(0, "b", "mod", 1.0, BIPOLAR)]
    d = CurveDrive(1, [(0.5,), (0.5,)], binds)
    assert abs(d.apply([0.7])["b"] - 0.7) < 1e-12       # half-centred passthrough


def _close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all anim tests passed")
