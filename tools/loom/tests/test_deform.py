"""Mesh deformers from loom (ftrace ROADMAP 11 stage E): what DeformedMesh emits.

Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import (  # noqa: E402
    Clock, Cache, Camera, Group, Motion, Scene, Material, Ramp, vec,
    DeformedMesh, Handle, Falloff, Rbf, Elastic, Dynamics,
)
from loom.deform import needs_elastic_bake  # noqa: E402
from loom.ftsl_emit import EmitCtx  # noqa: E402


def _emit(e, frame=3, frames=10, motion=None, assets=None):
    clock = Clock.at_frame(frame, frames, fps=24.0, loop=False)
    return e.emit(EmitCtx(clock=clock, cache=Cache(), motion=motion, assets_dir=assets))


def test_a_still_handle_writes_one_target():
    m = DeformedMesh("bar.obj", [Falloff([Handle((0, 0, 0), to=(0, 0.4, 0), radius=0.6)], profile="linear")],
                     material="m", name="bar")
    t = _emit(m)
    assert 'mesh "bar" { file "bar.obj"  material "m"' in t
    assert "deform falloff { profile linear  handle { at 0 0 0  to 0 0.4 0  radius 0.6 } }" in t


def test_a_moving_handle_gets_a_key_per_motion_key():
    lift = vec(0.0, Ramp(), 0.0)
    m = DeformedMesh("bar.obj", [Rbf([Handle((0, 0, 0), to=lift), Handle((1, 0, 0))], kernel="triharmonic")])
    t = _emit(m, motion=Motion((-0.5, 0.5), 3))
    keys = re.findall(r"key \{ to ([^}]*) \}", t)
    assert len(keys) == 3
    ys = [float(k.split()[1]) for k in keys]
    assert ys[0] < ys[1] < ys[2]                       # the ramp rises through the frame
    assert "kernel triharmonic" in t
    assert "handle { at 1 0 0 }" in t                   # a handle without `to` holds still
    # Without motion keys, one `to`: the instant's.
    t1 = _emit(m)
    assert "motion" not in t1 and re.search(r"to 0 [0-9.e-]+ 0", t1)


def test_a_dynamic_elastic_writes_its_cache_and_frame(tmp_path):
    m = DeformedMesh("bar.obj", [Elastic([Handle((0, 0, 0), to=(0, 0.3, 0), radius=0.12)],
                                         dynamic=Dynamics(mass=2, stiffness=20, damping=0.6))], name="bar")
    t = _emit(m, frame=7, assets=tmp_path)
    d = re.search(r"dynamic \{ ([^}]*) \}", t).group(1)
    assert 'cache "%s"' % (tmp_path.resolve() / "bar_%04d.ftec").as_posix() in d
    assert "frame 7" in d and "mass 2" in d and "stiffness 20" in d and "damping 0.6" in d and "fps 24" in d
    assert needs_elastic_bake([Group(m)])
    plain = DeformedMesh("bar.obj", [Elastic([Handle((0, 0, 0), to=(0, 0.3, 0), radius=0.12)])])
    assert not needs_elastic_bake([plain])


def test_refusals():
    with pytest.raises(ValueError):
        Elastic([Handle((0, 0, 0), to=(0, 1, 0))])      # an elastic handle holds vertices: needs a radius
    with pytest.raises(ValueError):
        DeformedMesh("bar.obj", [Falloff([Handle((0, 0, 0), radius=1)]),
                                 Elastic([Handle((0, 0, 0), radius=1)], dynamic=Dynamics())])
    with pytest.raises(ValueError):
        Falloff([Handle((0, 0, 0), radius=1)], profile="cubic")
    with pytest.raises(ValueError):
        Dynamics(damping=2)


def test_in_a_scene():
    s = Scene(Camera(eye=(0, 1, 4), look_at=(0, 0, 0), fov_y=40, mode="R", res=(32, 24)))
    s.add(Material("m", "diffuse", reflect=0.6),
          DeformedMesh("bar.obj", [Falloff([Handle((0, 0, 0), to=(0, 0.1, 0), radius=0.5)])], material="m"))
    t = s.emit(Clock.at_frame(0, 4, loop=False), Cache())
    assert "deform falloff" in t


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    test_a_still_handle_writes_one_target()
    test_a_moving_handle_gets_a_key_per_motion_key()
    with tempfile.TemporaryDirectory() as d:
        test_a_dynamic_elastic_writes_its_cache_and_frame(Path(d))
    test_refusals()
    test_in_a_scene()
    print("ok")
