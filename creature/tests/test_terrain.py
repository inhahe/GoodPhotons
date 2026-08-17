"""Regression tests for P1b's rough ground.

The bias is the same as `test_env.py`'s: terrain is a layer where everything that can go
wrong produces a plausible number rather than an exception. A transposed heightfield still
trains, and only means the ground the policy feels is not the ground the termination test
reads. A spawn that ignores the local elevation still trains, and only means every episode
begins with a contact impulse. Envs sharing one `MjModel` still train, and only mean the
ground moves under a running episode every time some other env resets. None of that raises,
so each of them is pinned here.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mujoco = pytest.importorskip("mujoco")

from creaturelab import terrain                                       # noqa: E402
from creaturelab.emit_mjcf import geom_z_extent                       # noqa: E402
from creaturelab.env import EnvConfig, VecCreatureEnv, build_body     # noqa: E402

RIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "rigs", "canis.ftcl")


def cfg(**kw) -> EnvConfig:
    kw.setdefault("rig", RIG)
    kw.setdefault("terrain", True)
    return EnvConfig(**kw)


@pytest.fixture(scope="module")
def body():
    return build_body(cfg())


@pytest.fixture(scope="module")
def flat_body():
    return build_body(cfg(terrain=False))


# --------------------------------------------------------------------------- the spec
def test_patch_covers_an_episode(body):
    """The patch must be big enough that a full-speed episode cannot walk off it.

    If it is not, the animal reaches the edge of the hfield and falls off the world -- and
    the failure looks exactly like a policy that learned to dive, because that is what the
    logged termination says.
    """
    s = body.terrain
    reach = EnvConfig().speed_range[1] * math.sqrt(9.81 * body.withers) * EnvConfig().episode_seconds
    assert s.extent >= reach, f"patch half-extent {s.extent:.1f} m < episode reach {reach:.1f} m"
    assert s.n % 2 == 1, "an odd grid puts a sample exactly at the origin the body spawns on"


def test_spec_is_in_body_units():
    """The same config on two sizes of animal must mean the same terrain, scaled."""
    a = terrain.TerrainSpec.for_body(0.4)
    b = terrain.TerrainSpec.for_body(0.8)
    assert b.amp == pytest.approx(2 * a.amp)
    assert b.body_len == pytest.approx(2 * a.body_len)
    # Extent grows as sqrt(L) (Froude speed) x seconds, so the patch is not simply scaled --
    # what must hold is that the bigger animal gets the bigger patch.
    assert b.extent > a.extent


# ------------------------------------------------------------------------- generation
@pytest.mark.parametrize("kind", terrain.KINDS)
def test_generate_is_normalised(kind):
    s = terrain.TerrainSpec.for_body(0.55)
    rng = np.random.default_rng(0)
    u, span = terrain.generate(kind, rng, s)
    assert u.shape == (s.n, s.n)
    assert 0.0 <= u.min() and u.max() <= 1.0
    if kind == "flat":
        assert span == 0.0 and not u.any()
    else:
        assert u.max() == pytest.approx(1.0) and u.min() == pytest.approx(0.0)
    # Rough classes are quoted in withers heights; slope classes take the whole elevation
    # scale, which is what that scale was sized for. Mixing the two up is how a difficulty-1
    # rubble field becomes a mountain range -- see the module docstring.
    if kind in ("rolling", "rubble", "steps"):
        assert span == pytest.approx(s.amp)
    elif kind in ("slope", "stairs"):
        assert span == pytest.approx(s.elevation)


def test_difficulty_scales_amplitude_in_metres(body):
    m = body.model
    p = terrain.Patch(body.terrain, m)
    rng = np.random.default_rng(1)
    for d in (0.25, 0.5, 1.0):
        p.regenerate(rng, d, ("rubble",))
        peak = float(p.grid.max()) * body.terrain.elevation
        assert peak == pytest.approx(d * body.terrain.amp, rel=1e-4)


def test_difficulty_zero_is_exactly_flat(body):
    p = terrain.Patch(body.terrain, body.model)
    p.regenerate(np.random.default_rng(2), 1.0, ("rubble",))
    assert p.grid.any()
    p.regenerate(np.random.default_rng(2), 0.0)
    assert p.kind == "flat" and not p.grid.any()
    assert p.height_at(3.0, -7.0) == 0.0


# ------------------------------------------------------------------- the height query
@pytest.mark.parametrize("kind", ("rolling", "rubble", "steps", "slope", "stairs"))
def test_height_agrees_with_mujocos_own_ray(body, kind):
    """`height_at` must return the surface MuJoCo actually collides against.

    This is the test that pins the data layout: MuJoCo stores an hfield row-major from the
    minimum y, so rows are y and columns are x. Read it transposed and every query is wrong
    by the full amplitude on any field that is not symmetric -- while still being a
    perfectly plausible height. Checked against `mj_ray` rather than against a comment.

    The tolerance is a cell's worth of relief: we interpolate bilinearly where MuJoCo
    triangulates, so the two disagree by at most the sag of one cell's diagonal. That is the
    documented approximation, and the point of the test is that it is bounded by the cell and
    not by the patch.
    """
    m = body.model
    d = mujoco.MjData(m)
    p = terrain.Patch(body.terrain, m)
    rng = np.random.default_rng(5)
    p.regenerate(rng, 1.0, (kind,))
    mujoco.mj_forward(m, d)

    gid = np.zeros(1, dtype=np.int32)
    top = 10.0 + body.terrain.elevation
    err = []
    for _ in range(200):
        x, y = rng.uniform(-8.0, 8.0, 2)
        dist = mujoco.mj_ray(m, d, np.array([x, y, top]), np.array([0.0, 0.0, -1.0]),
                             None, 1, -1, gid)
        assert dist > 0, "the ray missed the terrain entirely"
        err.append(top - dist - p.height_at(x, y))
    err = np.abs(np.array(err))
    # One cell of relief: the steepest a cell can be is a full-amplitude step across it.
    cell_relief = body.terrain.elevation * float(np.abs(np.diff(p.grid, axis=1)).max())
    assert err.max() <= max(1e-3, cell_relief), (
        f"{kind}: worst height error {err.max():.4f} m exceeds one cell's relief "
        f"{cell_relief:.4f} m -- transposed grid, or the wrong extent")


def test_height_at_clamps_outside_the_patch(body):
    p = terrain.Patch(body.terrain, body.model)
    p.regenerate(np.random.default_rng(7), 1.0, ("rolling",))
    e = body.terrain.extent
    assert p.height_at(1e6, 0.0) == pytest.approx(p.height_at(e, 0.0))
    assert p.height_at(0.0, -1e6) == pytest.approx(p.height_at(0.0, -e))


def test_max_height_in_bounds_the_point_query(body):
    p = terrain.Patch(body.terrain, body.model)
    p.regenerate(np.random.default_rng(9), 1.0, ("rubble",))
    hi = p.max_height_in(-1.0, 1.0, -1.0, 1.0)
    rng = np.random.default_rng(10)
    for _ in range(100):
        x, y = rng.uniform(-1.0, 1.0, 2)
        assert p.height_at(x, y) <= hi + 1e-9


# ----------------------------------------------------------------------------- the env
def test_each_env_gets_its_own_terrain(body):
    """Envs sharing an `MjModel` share `hfield_data`, and a reset would move the ground
    under every other env mid-episode. The clone in `VecCreatureEnv` is what prevents it."""
    v = VecCreatureEnv([body] * 4, cfg(), seed=0)
    v.terrain_level = 1.0
    v.reset(seed=3)
    assert len({id(m) for m in v.models}) == 4
    grids = [p.grid for p in v.patches]
    assert len({g.ctypes.data for g in grids}) == 4, "two envs share one elevation buffer"
    # And the contents really differ -- a shared buffer that was written four times would
    # pass the pointer test above by accident if the clone were shallow.
    assert not np.array_equal(grids[0], grids[1])
    v.close()


def test_spawn_clears_the_local_ground(body):
    """No geom may start below the terrain. A buried foot is an enormous impulse on step 1."""
    v = VecCreatureEnv([body] * 6, cfg(), seed=0)
    v.terrain_level = 1.0
    v.reset(seed=4)
    for i in range(v.n):
        m, d, p = v.models[i], v.datas[i], v.patches[i]
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] == 0:
                continue
            lo = geom_z_extent(m, d, gid)[0]
            x, y = d.geom_xpos[gid][0], d.geom_xpos[gid][1]
            assert lo >= p.height_at(x, y) - 1e-6, (
                f"env {i} geom {gid} ({p.kind}) starts {p.height_at(x, y) - lo:.3f} m "
                f"below the ground")
    v.close()


def test_termination_is_relative_to_the_local_ground(body):
    """A healthy animal standing 14 m up a slope is not a fallen animal.

    World-z termination is wrong in both directions on terrain -- a dip reads as a fall and
    a rise reads as unusually tall -- and the dip is the expensive one, because it ends
    episodes the policy was doing nothing wrong in.
    """
    v = VecCreatureEnv([body] * 4, cfg(), seed=0)
    v.terrain_level = 1.0
    v.reset(seed=11)
    for _ in range(5):
        _, _, term, _, _ = v.step(np.zeros((v.n, v.act_dim)))
    assert not term.any(), "a standing animal terminated on terrain"
    v.close()

    # The point of the test: on a full-difficulty slope the animal stands metres away from
    # z = 0, so a world-z test is decided by the elevation rather than by the animal. Note
    # the env is built slope-only rather than having a patch forced by hand -- `reset` rolls
    # its own terrain, so a hand-written patch followed by a reset is simply discarded.
    v = VecCreatureEnv([body] * 4, cfg(terrain_kinds=("slope",)), seed=0)
    v.terrain_level = 1.0
    v.reset(seed=11)
    assert np.abs(v.raw.ground_z).max() > 1.0, "no env stands far enough from z = 0 to tell "\
                                               "the world-z and local-z tests apart"
    clear = v.raw.root_z - v.raw.ground_z
    # Clearance above the LOCAL ground is a body-scale number however high up the slope the
    # animal is; that is the whole difference from `root_z`, which is tens of metres here.
    assert (clear > EnvConfig().min_height * v.withers).all()
    assert (clear < 3.0 * v.withers).all()
    for _ in range(5):
        _, _, term, _, _ = v.step(np.zeros((v.n, v.act_dim)))
    assert not term.any(), "a standing animal terminated on a slope"
    v.close()


def test_flat_env_is_untouched(flat_body):
    """With terrain off nothing is cloned, nothing is queried, and `ground_z` stays zero --
    so the termination test is the pre-P1b one, bit for bit."""
    v = VecCreatureEnv([flat_body] * 3, cfg(terrain=False), seed=0)
    assert v.patches is None
    assert all(m is flat_body.model for m in v.models)
    v.reset(seed=0)
    for _ in range(5):
        v.step(np.zeros((v.n, v.act_dim)))
    assert not v.raw.ground_z.any()
    assert v.terrain_level == 0.0
    v.close()


def test_mixing_terrain_and_flat_bodies_raises(body, flat_body):
    with pytest.raises(ValueError, match="terrain"):
        VecCreatureEnv([body, flat_body], cfg(), seed=0)


# ------------------------------------------------------------------------ the curriculum
def _pass_a_window(v, n_eps: int) -> None:
    """Hand `_advance_curriculum` a window of perfectly-tracking finished episodes.

    Deliberately a fixed number of calls rather than a loop on `v._cur_eps`: that counter is
    zeroed the moment the window fills, and is not incremented at all once neither axis can
    promote, so any `while _cur_eps < window` spins forever at both ends.
    """
    idx = np.arange(v.n)
    for _ in range(-(-int(n_eps) // v.n)):   # ceil: enough finished episodes to fill a window
        v._steps[idx] = 10
        v._ep_track[idx] = 10.0          # r_speed == 1 on every step: a perfect window
        v._advance_curriculum(idx)


def test_promotion_alternates_between_speed_and_terrain(body):
    v = VecCreatureEnv([body] * 4, cfg(curriculum_window=8), seed=0)
    seen = []
    for _ in range(12):
        before = (v.speed_cap, v.terrain_level)
        _pass_a_window(v, v.cfg.curriculum_window)
        after = (v.speed_cap, v.terrain_level)
        assert after != before, "a perfect window earned no promotion"
        seen.append("speed" if after[0] != before[0] else "terrain")
    # Both axes must move. Advancing speed to its maximum first would train a flat-ground
    # gallop and then ask it to relearn on rubble.
    assert "speed" in seen and "terrain" in seen
    assert v.terrain_level > 0.0 and v.speed_cap > v.cfg.curriculum_start
    v.close()


def test_curriculum_stops_when_both_axes_are_maxed(body):
    v = VecCreatureEnv([body] * 4, cfg(curriculum_window=8), seed=0)
    for _ in range(60):
        _pass_a_window(v, v.cfg.curriculum_window)
    assert v.speed_cap == pytest.approx(v.cfg.speed_range[1])
    assert v.terrain_level == pytest.approx(1.0)
    v.close()


def test_terrain_level_survives_a_round_trip(body):
    """`terrain_level` is training state exactly as much as `speed_cap` is; a resume that
    restarts it at 0 hands a competent policy the flat task it solved millions of steps ago."""
    v = VecCreatureEnv([body] * 2, cfg(), seed=0)
    v.terrain_level = 0.7
    assert v.terrain_level == pytest.approx(0.7)
    v.terrain_level = 5.0
    assert v.terrain_level == 1.0
    v.reset(seed=0)
    assert all(p.difficulty == 1.0 for p in v.patches)
    v.close()


# --------------------------------------------------------------------------- the train CLI
# Subprocesses, and worth the ~30 s they cost, because both bugs below were live in the tree
# and neither is reachable in-process: they live in `tools/train.py`'s argument handling, and
# both produce a run that trains perfectly happily on the wrong task.
def _train(*args: str) -> subprocess.CompletedProcess:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, os.path.join(root, "tools", "train.py"),
         "--envs", "2", "--horizon", "8", "--eval-every", "10000", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONPATH=root), cwd=root, timeout=900)


def _levels(log: str) -> list[float]:
    return [json.loads(line)["terrain_level"] for line in open(log) if line.strip()]


def test_terrain_level_sets_the_starting_difficulty_of_a_TRAINING_run(tmp_path):
    """`--terrain-level 0.5` used to be an eval-only flag that silently did nothing here.

    It turned terrain *on* (so the run paid the 26%) and then trained at difficulty 0.0 --
    a flat-ground run wearing a terrain run's costume, which the progress line reported
    honestly as `terr 0.0` and nobody would think to disbelieve.
    """
    out = tmp_path / "run"
    r = _train("--terrain-level", "0.5", "--steps", "32", "--out", str(out))
    assert r.returncode == 0, r.stderr
    levels = _levels(out / "log.jsonl")
    assert levels and set(levels) == {0.5}, \
        f"the run trained at a difficulty nobody asked for: {levels}"


def test_resume_of_a_terrain_run_does_not_silently_continue_on_a_plane(tmp_path):
    """Terrain is compiled into the model, so it cannot be restored after the env is built.

    A `--resume` therefore has to learn it from the checkpoint *before* building anything.
    Before this was wired, `--resume ckpt` without re-passing `--terrain` rebuilt a flat env,
    restored the terrain_level onto it, and carried on -- the task changed mid-run and the
    only trace was the `terr` column quietly vanishing from the progress line.
    """
    out = tmp_path / "run"
    assert _train("--terrain-level", "0.4", "--steps", "16", "--out", str(out)).returncode == 0
    r = _train("--resume", str(out / "latest.pt"), "--steps", "32", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "terrain on" in r.stdout, "the resume did not notice it was resuming a terrain run"
    levels = _levels(out / "log.jsonl")
    assert len(levels) >= 2 and set(levels) == {0.4}, f"the resume lost the terrain: {levels}"


# ------------------------------------------------------------------- the held-out class
# P1b's bar is "survives on terrain drawn from a class it never saw in training", which is
# only measurable if a class can actually be held out of training and named at eval time.
# That is one flag, `--terrain-kinds`, and these pin the three ways it can quietly not hold.
def test_a_restricted_class_set_is_the_only_thing_drawn(body):
    """The env must draw from the configured set and nothing else.

    Worth pinning separately from the CLI: `regenerate` picks with `rng.integers(len(kinds))`,
    and an off-by-one or a stale default here would show up as the held-out class appearing in
    a run whose whole claim is that it never did.
    """
    v = VecCreatureEnv([body] * 4, cfg(terrain_kinds=("rolling", "slope")), seed=0)
    v.terrain_level = 1.0
    seen = set()
    for s in range(12):
        v.reset(seed=s)
        seen |= {p.kind for p in v.patches}
    assert seen <= {"rolling", "slope"}, f"drew a class it was not given: {seen}"
    assert seen == {"rolling", "slope"}, f"never drew part of its own set in 48 rolls: {seen}"
    v.close()


def test_terrain_kinds_reports_what_it_held_out(tmp_path):
    """The held-out class decides whether the run's eval number is a generalisation claim.

    So it has to be legible from the log file alone, months later, without the shell history
    that started the run -- a run trained on five classes and one trained on six are otherwise
    indistinguishable from their output.
    """
    out = tmp_path / "run"
    r = _train("--terrain-kinds", "flat,rolling,rubble,steps,slope", "--terrain-level", "0.6",
               "--steps", "16", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "held out: stairs" in r.stdout, r.stdout


def test_a_resume_keeps_the_held_out_class_held_out(tmp_path):
    """Unlike the terrain *flag*, the class set could be restored after the env is built --
    but it is just as much a task, and it is the half a held-out run cannot afford to lose.

    A run trained on five classes that resumes onto all six has been shown the very terrain
    its bar is about generalising to, and nothing in its output would say so.
    """
    out = tmp_path / "run"
    assert _train("--terrain-kinds", "flat,rolling", "--terrain-level", "0.6",
                  "--steps", "16", "--out", str(out)).returncode == 0
    r = _train("--resume", str(out / "latest.pt"), "--steps", "32", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "held out: rubble, steps, slope, stairs" in r.stdout, r.stdout


def test_a_misspelled_terrain_class_is_refused(tmp_path):
    """Nothing downstream would object: `generate` is only ever asked for names that were
    drawn from the set, so a typo silently shrinks the set instead of raising."""
    r = _train("--terrain-kinds", "rolling,stiars", "--steps", "16",
               "--out", str(tmp_path / "run"))
    assert r.returncode != 0
    assert "unknown terrain class stiars" in r.stderr, r.stderr


def test_scoring_on_terrain_without_a_difficulty_is_refused(tmp_path):
    """`--eval CKPT --terrain` is difficulty 0, which is a flat heightfield: the eval pays
    terrain's cost and prints a flat-ground table under a heading that says terrain."""
    out = tmp_path / "run"
    assert _train("--steps", "16", "--out", str(out)).returncode == 0
    r = _train("--eval", str(out / "latest.pt"), "--terrain")
    assert r.returncode == 2
    assert "explicit --terrain-level" in r.stderr, r.stderr


def test_scoring_on_a_named_class_says_which(tmp_path):
    """The measurement path for the bar: score a checkpoint on one class at one difficulty."""
    out = tmp_path / "run"
    assert _train("--steps", "16", "--out", str(out)).returncode == 0
    r = _train("--eval", str(out / "latest.pt"), "--terrain-kinds", "stairs",
               "--terrain-level", "1.0")
    assert r.returncode == 0, r.stderr
    assert "difficulty 1.00 (stairs)" in r.stdout, r.stdout
