"""E2 slice 2 — named animatable slots + the editor↔loom live-value channel."""

import io
import json
import os
import tempfile

import pytest

from loom.anim import (
    ChannelBinding, CurveDrive, MODE_ANIMATION,
    Slot, collect_slots, SceneDriver, LiveSession, serve_live,
)
from loom.axes import ADDITIVE
from loom.signals.core import Clock, Cache
from loom.scene import Scene, Camera, Material, Sphere


# --------------------------------------------------------------------------
# a minimal scene with named slots
# --------------------------------------------------------------------------

def _scene():
    """Camera + a material whose roughness is a named Slot + a sphere."""
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    rough = Slot("rough", 0.3)
    mat = Material("m", "diffuse", roughness=rough)
    sph = Sphere((0, 0, 0), 1.0, "m")
    sc = Scene(cam)
    sc.add(mat, sph)
    return sc, rough


# --------------------------------------------------------------------------
# Slot
# --------------------------------------------------------------------------

def test_slot_is_signal_and_holds_value():
    s = Slot("x", 0.5)
    assert s.name == "x"
    assert s.default == 0.5
    assert s.value == 0.5
    clk = Clock.at_frame(0, 1)
    assert s.at(clk) == 0.5
    s.set(0.9)
    assert s.at(clk, Cache()) == 0.9  # fresh cache reflects new value
    s.reset()
    assert s.at(clk) == 0.5


def test_slot_stale_cache_pins_old_value():
    # A cache captures the value at first read; a Slot mutated afterwards is only
    # picked up with a fresh cache — the reason SceneDriver emits per-frame fresh.
    s = Slot("x", 0.2)
    cache = Cache()
    clk = Clock.at_frame(0, 1)
    assert s.at(clk, cache) == 0.2
    s.set(0.8)
    assert s.at(clk, cache) == 0.2   # stale (same frame, cached)
    assert s.at(clk, Cache()) == 0.8  # fresh


# --------------------------------------------------------------------------
# collect_slots
# --------------------------------------------------------------------------

def test_collect_slots_finds_by_name():
    sc, rough = _scene()
    slots = collect_slots(sc)
    assert set(slots) == {"rough"}
    assert slots["rough"] == [rough]


def test_collect_slots_groups_same_name():
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    a = Slot("k", 0.1)
    b = Slot("k", 0.1)
    sc = Scene(cam)
    sc.add(Material("m1", "diffuse", roughness=a),
           Material("m2", "diffuse", roughness=b),
           Sphere((0, 0, 0), 1.0, "m1"))
    slots = collect_slots(sc)
    assert set(slots) == {"k"}
    assert len(slots["k"]) == 2


def test_collect_slots_empty_when_no_slots():
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    sc = Scene(cam)
    sc.add(Material("m", "diffuse", roughness=0.4), Sphere((0, 0, 0), 1.0, "m"))
    assert collect_slots(sc) == {}


# --------------------------------------------------------------------------
# SceneDriver
# --------------------------------------------------------------------------

def _drive():
    return CurveDrive(
        dims=1,
        points=[(0.0,), (1.0,)],
        bindings=[ChannelBinding(0, "rough", mode="pin", gain=1.0, kind=ADDITIVE)],
        mode=MODE_ANIMATION,
    )


def test_scene_driver_bases_default_to_slot_default():
    sc, rough = _scene()
    drv = SceneDriver(sc, _drive())
    assert drv.bases["rough"] == 0.3


def test_scene_driver_bases_override():
    sc, _ = _scene()
    drv = SceneDriver(sc, _drive(), bases={"rough": 0.9})
    assert drv.bases["rough"] == 0.9


def test_scene_driver_strict_raises_on_missing_slot():
    sc, _ = _scene()
    bad = CurveDrive(dims=1, points=[(0.0,), (1.0,)],
                     bindings=[ChannelBinding(0, "nope")])
    with pytest.raises(ValueError):
        SceneDriver(sc, bad, strict=True)
    # non-strict: silently ignores
    SceneDriver(sc, bad, strict=False)


def test_scene_driver_set_values_pushes_into_slot():
    sc, rough = _scene()
    drv = SceneDriver(sc, _drive())
    resolved = drv.set_values([0.75])
    assert resolved["rough"] == pytest.approx(0.75)
    assert rough.value == pytest.approx(0.75)


def test_scene_driver_mod_accumulates_on_base():
    sc, rough = _scene()
    drive = CurveDrive(dims=1, points=[(0.0,), (1.0,)],
                       bindings=[ChannelBinding(0, "rough", mode="mod",
                                                gain=1.0, kind=ADDITIVE)])
    drv = SceneDriver(sc, drive)  # base = slot default 0.3
    resolved = drv.set_values([0.2])
    assert resolved["rough"] == pytest.approx(0.5)  # 0.3 + 0.2
    assert rough.value == pytest.approx(0.5)


def test_scene_driver_emit_frame_reflects_current_value():
    sc, rough = _scene()
    drv = SceneDriver(sc, _drive())
    clk = Clock.at_frame(0, 1)
    text = drv.emit_frame([0.6], clk)
    assert "roughness 0.6" in text
    text2 = drv.emit_frame([0.1], clk)
    assert "roughness 0.1" in text2  # fresh cache each frame


# --------------------------------------------------------------------------
# LiveSession
# --------------------------------------------------------------------------

def _session():
    sc, rough = _scene()
    drv = SceneDriver(sc, _drive())
    return LiveSession(drv), rough


def test_live_config_returns_sidecar():
    sess, _ = _session()
    ack = sess.handle({"cmd": "config"})
    assert ack["ok"]
    assert ack["config"]["dims"] == 1
    assert ack["config"]["bindings"][0]["target"] == "rough"


def test_live_frame_with_values(tmp_path):
    sess, rough = _session()
    out = str(tmp_path / "f0000.ftsl")
    ack = sess.handle({"cmd": "frame", "values": [0.55], "frame": 0,
                       "frames": 1, "out": out})
    assert ack["ok"]
    assert ack["targets"]["rough"] == pytest.approx(0.55)
    assert os.path.exists(out)
    with open(out) as f:
        assert "roughness 0.55" in f.read()


def test_live_frame_samples_at_t_when_no_values():
    sess, rough = _session()
    ack = sess.handle({"cmd": "frame", "t": 0.5, "frame": 0, "frames": 1})
    assert ack["ok"]
    # points (0,)->(1,), 2-pt linear at t=0.5 -> 0.5 channel value -> pin
    assert ack["targets"]["rough"] == pytest.approx(0.5)


def test_live_bindings_replace():
    sess, _ = _session()
    ack = sess.handle({"cmd": "bindings",
                       "bindings": [{"channel": 0, "target": "rough",
                                     "mode": "mod", "gain": 0.5, "kind": ADDITIVE}]})
    assert ack["ok"]
    assert sess.drive.bindings[0].mode == "mod"


def test_live_bindings_reject_out_of_range_channel():
    sess, _ = _session()
    ack = sess.handle({"cmd": "bindings",
                       "bindings": [{"channel": 3, "target": "rough"}]})
    assert not ack["ok"]
    assert "error" in ack


def test_live_points_replace():
    sess, _ = _session()
    ack = sess.handle({"cmd": "points", "points": [[0.0], [0.5], [1.0]]})
    assert ack["ok"]
    assert len(sess.drive.points) == 3


def test_live_points_reject_dim_mismatch():
    sess, _ = _session()
    ack = sess.handle({"cmd": "points", "points": [[0.0, 1.0], [1.0, 2.0]]})
    assert not ack["ok"]


def test_live_save(tmp_path):
    sess, _ = _session()
    path = str(tmp_path / "drive.json")
    ack = sess.handle({"cmd": "save", "path": path})
    assert ack["ok"]
    loaded = CurveDrive.load(path)
    assert loaded.dims == 1


def test_live_save_defaults_to_the_session_sidecar(tmp_path):
    # The editor asks to save "the config I'm editing" without having to know (or
    # re-send) the path loom was launched with.
    sc, _ = _scene()
    path = str(tmp_path / "seeded.json")
    sess = LiveSession(SceneDriver(sc, _drive()), config_path=path)
    ack = sess.handle({"cmd": "save"})
    assert ack["ok"] and ack["path"] == path
    assert CurveDrive.load(path).dims == 1


def test_live_save_without_a_path_anywhere_is_an_error():
    sess, _ = _session()
    ack = sess.handle({"cmd": "save"})
    assert not ack["ok"] and "path" in ack["error"]


# ---- slice 3: the editor's pick-list and dimension edits ------------------

def test_live_slots_lists_the_bindable_scene_variables():
    # "Scene proposes, editor disposes" needs the *menu* too, or the editor would
    # have to make the user type a target name into a GDI panel.
    sess, _ = _session()
    ack = sess.handle({"cmd": "slots"})
    assert ack["ok"]
    assert ack["slots"] == {"rough": pytest.approx(0.3)}


def test_live_dims_grow_pads_points_with_a_neutral_channel():
    sess, _ = _session()
    ack = sess.handle({"cmd": "dims", "dims": 3})
    assert ack["ok"] and ack["dims"] == 3 and ack["dropped"] == []
    cfg = sess.handle({"cmd": "config"})["config"]
    assert cfg["dims"] == 3
    assert cfg["points"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]   # curve unmoved
    assert cfg["bindings"][0]["target"] == "rough"               # binding kept


def test_live_dims_shrink_reports_the_bindings_it_dropped():
    sess, _ = _session()
    sess.handle({"cmd": "dims", "dims": 3})
    sess.handle({"cmd": "bindings", "bindings": [
        {"channel": 0, "target": "rough"}, {"channel": 2, "target": "rough"}]})
    ack = sess.handle({"cmd": "dims", "dims": 1})
    assert ack["ok"] and ack["dims"] == 1
    assert [b["channel"] for b in ack["dropped"]] == [2]
    cfg = sess.handle({"cmd": "config"})["config"]
    assert [b["channel"] for b in cfg["bindings"]] == [0]
    assert cfg["points"] == [[0.0], [1.0]]


def test_live_dims_rejects_zero():
    sess, _ = _session()
    assert not sess.handle({"cmd": "dims", "dims": 0})["ok"]


def test_live_quit():
    sess, _ = _session()
    ack = sess.handle({"cmd": "quit"})
    assert ack["ok"]
    assert ack["bye"] is True


def test_live_unknown_cmd():
    sess, _ = _session()
    ack = sess.handle({"cmd": "frobnicate"})
    assert not ack["ok"]
    assert "unknown" in ack["error"]


# --------------------------------------------------------------------------
# serve_live (stdio loop)
# --------------------------------------------------------------------------

def test_serve_live_processes_lines_and_stops_on_quit():
    sess, rough = _session()
    inp = io.StringIO(
        json.dumps({"cmd": "frame", "values": [0.42], "frame": 0, "frames": 1}) + "\n"
        + "\n"  # blank line skipped
        + json.dumps({"cmd": "quit"}) + "\n"
        + json.dumps({"cmd": "config"}) + "\n"  # never reached (after quit)
    )
    out = io.StringIO()
    serve_live(sess, inp, out)
    lines = [l for l in out.getvalue().splitlines() if l.strip()]
    assert len(lines) == 2  # frame ack + quit ack (config after quit not processed)
    acks = [json.loads(l) for l in lines]
    assert acks[0]["ok"] and acks[0]["targets"]["rough"] == pytest.approx(0.42)
    assert acks[1]["bye"] is True


def test_serve_live_reports_bad_json():
    sess, _ = _session()
    inp = io.StringIO("{not json}\n" + json.dumps({"cmd": "quit"}) + "\n")
    out = io.StringIO()
    serve_live(sess, inp, out)
    acks = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
    assert not acks[0]["ok"]
    assert "bad json" in acks[0]["error"]


# --------------------------------------------------------------------------
# the CLI the editor spawns (`python -m loom.anim`) — E2 slice 3
# --------------------------------------------------------------------------

_SCENE_PY = '''
from loom.anim import ChannelBinding, CurveDrive, Slot
from loom.scene import Scene, Camera, Material, Sphere

def build(clock=None):
    sc = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
    sc.add(Material("m", "diffuse", roughness=Slot("rough", 0.3)),
           Sphere((0, 0, 0), 1.0, "m"))
    return sc

def drive():
    return CurveDrive(2, [(0.0, 0.0), (1.0, 1.0)],
                      [ChannelBinding(1, "rough")], name="proposed")
'''


def _run_cli(monkeypatch, argv, msgs):
    """Run `loom.anim.main(argv)` over an in-memory stdio pipe, returning the acks."""
    import sys as _sys
    from loom import anim
    inp = io.StringIO("".join(json.dumps(m) + "\n" for m in msgs))
    out = io.StringIO()
    monkeypatch.setattr(_sys, "stdin", inp)
    monkeypatch.setattr(_sys, "stdout", out)
    rc = anim.main(argv)
    return rc, [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]


def _scene_file(tmp_path):
    p = tmp_path / "anim_scene.py"
    p.write_text(_SCENE_PY)
    return str(p)


def test_cli_serves_a_session_seeded_by_the_scene(monkeypatch, tmp_path):
    # No sidecar yet, so the *scene's* proposal wins: its module-level `drive()`.
    rc, acks = _run_cli(monkeypatch, [_scene_file(tmp_path)],
                        [{"cmd": "config"}, {"cmd": "slots"}, {"cmd": "quit"}])
    assert rc == 0
    assert acks[0]["config"]["name"] == "proposed" and acks[0]["config"]["dims"] == 2
    assert acks[1]["slots"] == {"rough": pytest.approx(0.3)}


def test_cli_sidecar_beats_the_scene_proposal(monkeypatch, tmp_path):
    # An existing edit must survive a re-launch, so a sidecar outranks the scene.
    cfg = str(tmp_path / "drive.json")
    CurveDrive(1, [(0.0,), (2.0,)], [ChannelBinding(0, "rough")],
               name="edited").save(cfg)
    rc, acks = _run_cli(monkeypatch, [_scene_file(tmp_path), "--config", cfg],
                        [{"cmd": "config"}, {"cmd": "quit"}])
    assert rc == 0 and acks[0]["config"]["name"] == "edited"


def test_cli_writes_back_to_the_sidecar_it_was_seeded_from(monkeypatch, tmp_path):
    cfg = str(tmp_path / "drive.json")
    out = str(tmp_path / "f0000.ftsl")
    rc, acks = _run_cli(monkeypatch, [_scene_file(tmp_path), "--config", cfg],
                        [{"cmd": "points", "points": [[0, 0], [1, 0.5], [2, 1]]},
                         {"cmd": "frame", "values": [0.0, 0.75], "frame": 0,
                          "frames": 1, "out": out},
                         {"cmd": "save"},
                         {"cmd": "quit"}])
    assert rc == 0 and all(a["ok"] for a in acks)
    assert acks[1]["targets"]["rough"] == pytest.approx(0.75)
    assert "roughness 0.75" in open(out).read()
    assert len(CurveDrive.load(cfg).points) == 3        # the edit round-tripped


def test_cli_subprocess_sees_the_scenes_slots_and_drives_them(tmp_path):
    """`python -m loom.anim` must behave exactly like the in-process CLI.

    Regression test for the dual-module-identity bug: running a module with `-m`
    executes the file a *second* time under the name `__main__`, so `__main__.Slot`
    and `loom.anim.Slot` become two distinct classes.  The scene imports the
    canonical one; `collect_slots`, running out of `__main__`, then matched
    nothing.  Every binding still acked `ok` (the ack reports the drive's resolved
    map, not what any slot received) while the emitted `.ftsl` silently carried
    the slot *default* — a no-op live channel.  Only a real subprocess can
    reproduce it; the in-process `_run_cli` harness has one module identity.
    """
    import subprocess
    import sys as _sys
    from loom import anim as _anim

    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(_anim.__file__)))
    scene = _scene_file(tmp_path)
    out = str(tmp_path / "f0000.ftsl")
    msgs = [{"cmd": "slots"},
            {"cmd": "frame", "values": [0.0, 0.75], "frame": 0,
             "frames": 1, "out": out},
            {"cmd": "quit"}]
    env = dict(os.environ, PYTHONPATH=pkg_parent, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([_sys.executable, "-X", "utf8", "-u", "-m", "loom.anim", scene],
                          input="".join(json.dumps(m) + "\n" for m in msgs),
                          capture_output=True, text=True, cwd=pkg_parent, env=env,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
    acks = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
    assert acks[0]["slots"] == {"rough": pytest.approx(0.3)}, \
        "the -m entry point lost the scene's slots"
    assert acks[1]["ok"] and acks[1]["targets"]["rough"] == pytest.approx(0.75)
    # The ack alone is not proof — the emitted scene must carry the DRIVEN value.
    assert "roughness 0.75" in open(out).read()


def test_cli_falls_back_to_a_default_drive(monkeypatch, tmp_path):
    p = tmp_path / "plain.py"
    p.write_text("from loom.scene import Scene, Camera\n"
                 "def build(clock=None):\n"
                 "    return Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))\n")
    rc, acks = _run_cli(monkeypatch, [str(p), "--dims", "4"],
                        [{"cmd": "config"}, {"cmd": "quit"}])
    assert rc == 0
    cfg = acks[0]["config"]
    assert cfg["dims"] == 4 and len(cfg["points"]) == 2 and cfg["bindings"] == []
