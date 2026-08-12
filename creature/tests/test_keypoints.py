"""Tests for the 2D-detector <-> rig correspondence (`notes/keypoints.yaml`).

The file exists to make one class of bug unrepresentable: the detector emitting a
landmark under one name while the fit looks for it under another, so that the landmark
contributes nothing and the only symptom is a slightly worse fit on one limb. That bug
is silent by construction, so the tests here are almost all about things that would
otherwise never raise -- drift between the two files, a landmark that no joint can move,
two landmarks that are secretly the same point.

`test_keypoints.py` is named in keypoints.yaml's own header as the thing that enforces
the correspondence, so it has to actually load the rig rather than lint the YAML alone.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

yaml = pytest.importorskip("yaml")
mujoco = pytest.importorskip("mujoco")

from creaturelab.build import load                                  # noqa: E402
from creaturelab.emit_mjcf import to_mjcf                           # noqa: E402
from creaturelab.keypoints import (DEFAULT_PATH, KeypointError,     # noqa: E402
                                   check_against_rig, load_keypoints)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIG = os.path.join(ROOT, "rigs", "canis.ftcl")


@pytest.fixture(scope="module")
def kps():
    return load_keypoints()


@pytest.fixture(scope="module")
def creature():
    return load(RIG)


@pytest.fixture(scope="module")
def model(creature):
    return mujoco.MjModel.from_xml_string(to_mjcf(creature))


# --- the correspondence itself -----------------------------------------------------

def test_the_file_parses_and_is_the_expected_shape(kps):
    assert len(kps) == 21, "canis v1 is 21 landmarks (todo.md P5)"
    assert kps.rig == "rigs/canis.ftcl" and kps.creature == "canis"
    assert len(set(kps.labels)) == len(kps.labels)
    assert len(set(kps.sites)) == len(kps.sites)


def test_it_agrees_with_the_rig(kps, creature):
    """The check keypoints.yaml's header promises. Both directions, plus the classes."""
    problems = check_against_rig(kps, creature)
    assert not problems, "keypoints.yaml and canis.ftcl have drifted:\n  " + \
                         "\n  ".join(problems)


def test_every_landmark_is_addressable_in_mujoco(kps, model):
    """A site name that survives the rig but not the MJCF would break only at fit time."""
    for k in kps:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, k.site)
        assert sid >= 0, f"{k.label} -> site {k.site!r} is not in the compiled model"


def test_no_two_landmarks_are_the_same_point(kps, model):
    """Two sites at one position are a copy-paste in the rig, and are invisible.

    The fit would happily use both; they would agree perfectly, contribute a doubled
    weight at one place on the body, and never disagree with each other enough to
    complain. Only geometry catches this.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pos = {}
    for k in kps:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, k.site)
        pos[k.label] = np.array(data.site_xpos[sid])
    labels = list(pos)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            d = float(np.linalg.norm(pos[a] - pos[b]))
            assert d > 0.01, f"{a} and {b} are {d*1000:.1f} mm apart in the rest pose"


def test_left_and_right_labels_map_to_left_and_right_sites(kps):
    """The swapped-limb failure, caught at the one place it is still cheap to catch.

    A left/right swap in this file produces a fit that converges and has the legs
    crossed; it is invisible in any single view and survives every numerical check
    downstream. Here it is just a string comparison.
    """
    for k in kps:
        if k.label.endswith("_left"):
            assert "_l" in k.site and not k.site.endswith("_r"), \
                f"{k.label} maps to {k.site!r}"
        elif k.label.endswith("_right"):
            assert "_r" in k.site and not k.site.endswith("_l"), \
                f"{k.label} maps to {k.site!r}"


def test_the_body_is_covered_left_right_and_end_to_end(kps):
    """21 landmarks in the wrong places constrain no better than 8 in the right ones."""
    labels = set(kps.labels)
    assert {"nose", "occiput", "withers", "tail_base", "tail_tip"} <= labels
    for side in ("left", "right"):
        assert {f"shoulder_{side}", f"elbow_{side}", f"carpus_{side}",
                f"forepaw_{side}"} <= labels, f"fore limb {side} is under-observed"
        assert {f"hip_{side}", f"stifle_{side}", f"hock_{side}",
                f"hindpaw_{side}"} <= labels, f"hind limb {side} is under-observed"


# --- what the numbers in it mean ---------------------------------------------------

def test_soft_landmarks_are_both_wide_and_noisy(kps):
    """The two knobs have to move together, or one of them is doing nothing.

    `class: soft` widens the Huber because the residual is dominated by tissue sliding,
    not detector error. A soft landmark left at the default sigma would still be pulled
    hard by the quadratic core of the loss, and the wide delta would buy nothing.
    """
    soft = [k for k in kps if k.is_soft]
    assert soft, "no soft landmark at all -- tail_tip at least should be one"
    for k in soft:
        assert k.huber_px > max(x.huber_px for x in kps if not x.is_soft)
        assert k.sigma_px > kps.sigma_px_default, \
            f"{k.label} is soft but carries the default sigma"


def test_the_placeholder_sigma_is_flagged_as_a_placeholder(kps):
    """Guards the honesty flag, not the number.

    fit_selftest.py corrupts synthetic ground truth with these sigmas, so an optimistic
    value makes the self-test report a fit quality the real pipeline can never reach.
    That is only safe while the file says out loud that it is unmeasured -- so if
    someone flips `sigma_px_measured` to true, they have to come here and say why.
    """
    assert kps.sigma_px_measured is False, \
        ("sigma_px_measured is now true: replace this assertion with the measurement "
         "date and the held-out set it came from")
    assert 2.0 <= kps.sigma_px_default <= 20.0


def test_the_skeleton_overlay_is_a_connected_tree(kps):
    """Purely presentational, but a broken overlay is how labellers swap left and right."""
    labels = kps.labels
    assert len(kps.skeleton) == len(labels) - 1, "a tree over N nodes has N-1 edges"
    parent = {n: n for n in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in kps.skeleton:
        ra, rb = find(a), find(b)
        assert ra != rb, f"skeleton edge [{a}, {b}] closes a cycle"
        parent[ra] = rb
    assert len({find(n) for n in labels}) == 1, "the skeleton overlay is disconnected"


# --- the loader's own guarantees ---------------------------------------------------

def _write(tmp_path, text):
    p = tmp_path / "kp.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


HEAD = ("schema: creature/keypoints@1\nrig: rigs/canis.ftcl\ncreature: canis\n"
        "sigma_px_default: 6.0\nhuber_delta_px:\n  rigid: 2.0\n  soft: 8.0\n")


def test_two_keypoints_on_one_site_is_rejected(tmp_path):
    """A silent doubling of one landmark's weight in E_kp."""
    p = _write(tmp_path, HEAD + "keypoints:\n"
               "  - {label: a, site: nose, guide: x}\n"
               "  - {label: b, site: nose, guide: x}\n")
    with pytest.raises(KeypointError, match="already claimed"):
        load_keypoints(p)


def test_a_landmark_without_guide_text_is_rejected(tmp_path):
    """The guide IS the calibration: an unguided landmark carries a constant offset."""
    p = _write(tmp_path, HEAD + "keypoints:\n  - {label: a, site: nose}\n")
    with pytest.raises(KeypointError, match="guide"):
        load_keypoints(p)


def test_a_skeleton_edge_naming_an_unknown_landmark_is_rejected(tmp_path):
    p = _write(tmp_path, HEAD + "keypoints:\n  - {label: a, site: nose, guide: x}\n"
               "skeleton:\n  - [a, ghost]\n")
    with pytest.raises(KeypointError, match="ghost"):
        load_keypoints(p)


def test_a_future_schema_is_refused_rather_than_half_read(tmp_path):
    p = _write(tmp_path, HEAD.replace("@1", "@2")
               + "keypoints:\n  - {label: a, site: nose, guide: x}\n")
    with pytest.raises(KeypointError, match="schema"):
        load_keypoints(p)


def test_drift_from_the_rig_is_actually_detected(creature):
    """Proves the drift check has teeth, rather than trusting that it passes today.

    Without this, `test_it_agrees_with_the_rig` would still pass if
    `check_against_rig` were accidentally reduced to `return []`.
    """
    kps = load_keypoints()
    good = [s.name for s in creature.sites]

    class FakeSite:
        def __init__(self, name, kind):
            self.name, self.kind = name, kind

    class FakeCreature:
        name = "canis"

        def __init__(self, sites):
            self.sites = sites

    renamed = [FakeSite("elbow_LEFT" if n == "elbow_l" else n,
                        "soft" if n == "nose" else "rigid") for n in good]
    problems = check_against_rig(kps, FakeCreature(renamed))
    joined = "\n".join(problems)
    assert "elbow_l" in joined, "a renamed site was not reported"
    assert "elbow_LEFT" in joined, "the orphaned rig site was not reported"
    assert "nose" in joined, "a class/kind disagreement was not reported"


# --- the generated detector config -------------------------------------------------

def _project_tool():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import keypoints_project
    return keypoints_project


def test_the_generated_dlc_config_round_trips(kps, tmp_path):
    """Generate, then check: the drift detector must accept its own output.

    This is the property that lets `--check` be trusted. If the emitter and the checker
    disagreed about ordering or naming, every real config would look drifted and the
    warning would be ignored within a week.
    """
    kp = _project_tool()
    p = tmp_path / "config.yaml"
    p.write_text(kp.emit_dlc(kps, "canis", "lab"), encoding="utf-8")
    assert kp.check_dlc(kps, str(p)) == 0

    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert cfg["bodyparts"] == kps.labels
    assert [tuple(e) for e in cfg["skeleton"]] == kps.skeleton


def test_a_body_part_added_in_the_dlc_gui_is_caught(kps, tmp_path):
    """The one way drift can still be introduced, and the reason --check exists."""
    kp = _project_tool()
    p = tmp_path / "config.yaml"
    text = kp.emit_dlc(kps, "canis", "lab").replace(
        "bodyparts:\n", "bodyparts:\n- belly   # someone's good idea\n")
    p.write_text(text, encoding="utf-8")
    assert kp.check_dlc(kps, str(p)) == 1


def test_the_sleap_skeleton_indexes_the_same_nodes(kps):
    """SLEAP edges are index pairs, so an off-by-one here is a wrong overlay, not a crash."""
    kp = _project_tool()
    doc = json.loads(kp.emit_sleap(kps))
    assert doc["nodes"] == kps.labels
    for (a, b), (i, j) in zip(kps.skeleton, doc["edges"]):
        assert doc["nodes"][i] == a and doc["nodes"][j] == b
    # Every left landmark should be paired with its right, or the augmenter's mirror
    # flips will teach the detector that a mirrored dog has its legs swapped.
    lefts = [k.label for k in kps if k.label.endswith("_left")]
    assert len(doc["symmetries"]) == len(lefts) == 8


def test_the_shipped_file_is_the_one_the_tools_load():
    assert os.path.normcase(DEFAULT_PATH) == \
           os.path.normcase(os.path.join(ROOT, "notes", "keypoints.yaml"))
    assert os.path.exists(DEFAULT_PATH)
