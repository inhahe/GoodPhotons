"""``python -m loom.shot``: the timeline comes from the scene file's constants, the flags override
them, a shutter lands on the camera and brings motion keys, and everything after ``--`` reaches
every frame's ftrace. ``render_range`` is stubbed: these tests need no renderer.

Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import loom.drive  # noqa: E402
from loom import Motion  # noqa: E402
from loom.shot import _default_motion, main  # noqa: E402

SCENE = """
from loom import Scene, Camera, Material, Sphere
FRAMES = 12
FPS = 24.0
LOOP = False
SHUTTER = (-0.25, 0.25)

def build(clock=None):
    s = Scene(Camera(eye=(0, 0, 3), look_at=(0, 0, 0), res=(16, 16)))
    s.add(Material("m", "diffuse", reflect=0.5), Sphere((0, 0, 0), 0.5, "m"))
    return s
"""


@pytest.fixture
def scene_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tiny_shot.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(SCENE)
        yield p


@pytest.fixture
def calls(monkeypatch):
    got = []

    def fake(scene, frames, **kw):
        got.append((scene, frames, kw))
        return []
    monkeypatch.setattr(loom.drive, "render_range", fake)
    return got


def test_default_motion_lands_a_key_on_the_instant():
    assert _default_motion((-0.25, 0.25)) == Motion((-0.5, 0.5), 5)
    m = _default_motion((0.0, 1.0))
    assert m.interval == (-0.5, 1.0) and m.keys == 7       # 0.5 / 1.5 * 6 = 2: on a key


def test_the_file_sets_the_timeline(scene_file, calls):
    assert main([scene_file, "--", "-spp", "64"]) == 0
    scene, frames, kw = calls[0]
    assert frames == 12 and kw["fps"] == 24.0 and kw["loop"] is False
    assert scene.camera.shutter == (-0.25, 0.25)
    assert kw["motion"] == Motion((-0.5, 0.5), 5)
    assert kw["name"] == "tiny_shot" and kw["skip_existing"] is True
    assert kw["extra_args"] == ["-window-min", "-spp", "64"]
    assert kw["frame_range"] is None


def test_flags_override_the_file(scene_file, calls):
    main([scene_file, "--frames", "30", "--fps", "60", "--loop", "--shutter", "0", "0.5",
          "--motion", "0", "1", "3", "--range", "2", "5", "--name", "x", "--no-window",
          "--ftrace", "C:/f.exe", "--stop-file", "C:/s.stop", "--", "-mode", "W"])
    scene, frames, kw = calls[0]
    assert frames == 30 and kw["fps"] == 60.0 and kw["loop"] is True
    assert scene.camera.shutter == (0.0, 0.5)
    assert kw["motion"] == Motion((0.0, 1.0), 3)
    assert kw["frame_range"] == (2, 5) and kw["name"] == "x"
    assert kw["extra_args"] == ["-mode", "W"]
    assert kw["ftrace"] == "C:/f.exe" and kw["stop_file"] == "C:/s.stop"


def test_no_motion_and_no_shutter(scene_file, calls):
    main([scene_file, "--no-motion"])
    assert calls[0][2]["motion"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
