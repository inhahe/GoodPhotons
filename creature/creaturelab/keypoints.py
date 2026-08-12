"""Load `notes/keypoints.yaml`: the 2D-detector <-> rig landmark correspondence.

This is the reader for the interface file described at the top of `notes/keypoints.yaml`
itself. Three consumers share it, and that sharing is the point:

* `tools/keypoints_project.py` generates the DLC/SLEAP project config from it, so the
  detector's body-part list cannot drift from the rig's site list.
* `tests/test_keypoints.py` checks it against the built rig on every run.
* stage C's fit reads it to know which `mj_jacSite` row a given detected keypoint
  constrains, and with what sigma and Huber delta.

If any of those three re-implemented the parse, they could disagree about defaults --
notably `sigma_px_default`, which is silently applied to the 18 keypoints that do not
override it and which therefore sets the weight of most of E_kp.

Validation is deliberately eager and total: every problem in the file is collected and
reported at once rather than raised at the first one. A correspondence file is edited by
a person adding several landmarks in one sitting, and a loader that reports one typo per
run turns that into several round trips.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

SCHEMA = "creature/keypoints@1"

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "notes", "keypoints.yaml")


class KeypointError(ValueError):
    """The correspondence file is malformed, or disagrees with the rig."""


@dataclass(frozen=True)
class Keypoint:
    """One landmark: what the annotator sees, and where it lives on the skeleton."""

    label: str                  # what the detector emits and the human clicks
    site: str                   # the rig site name it constrains
    cls: str                    # 'rigid' | 'soft'
    sigma_px: float             # detector noise, already defaulted
    huber_px: float             # Huber delta, already resolved from the class
    guide: str = ""             # labelling instructions; the calibration of the interface

    @property
    def is_soft(self) -> bool:
        return self.cls == "soft"


@dataclass(frozen=True)
class KeypointSet:
    """The whole file, parsed and defaulted."""

    rig: str
    creature: str
    keypoints: list[Keypoint]
    skeleton: list[tuple[str, str]]
    sigma_px_default: float
    sigma_px_measured: bool
    path: str = ""

    def __len__(self) -> int:
        return len(self.keypoints)

    def __iter__(self):
        return iter(self.keypoints)

    @property
    def labels(self) -> list[str]:
        return [k.label for k in self.keypoints]

    @property
    def sites(self) -> list[str]:
        return [k.site for k in self.keypoints]

    def by_label(self, label: str) -> Keypoint:
        for k in self.keypoints:
            if k.label == label:
                return k
        raise KeyError(label)

    def by_site(self, site: str) -> Keypoint:
        for k in self.keypoints:
            if k.site == site:
                return k
        raise KeyError(site)


_CLASSES = ("rigid", "soft")


def load_keypoints(path: str | None = None) -> KeypointSet:
    """Parse and validate the correspondence file. Raises `KeypointError` on any problem.

    The rig is NOT consulted here -- `check_against_rig` does that separately, because
    this loader has to work in contexts that have no MuJoCo (the DLC config generator
    runs in the detector's environment, which is a different install entirely).
    """
    import yaml

    path = path or DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise KeypointError(f"{path}: expected a YAML mapping, found "
                            f"{type(doc).__name__}")

    schema = doc.get("schema")
    if schema != SCHEMA:
        raise KeypointError(f"{path}: schema is {schema!r}, this build reads {SCHEMA!r}")

    huber = doc.get("huber_delta_px") or {}
    if not isinstance(huber, dict) or set(huber) != set(_CLASSES):
        raise KeypointError(f"{path}: huber_delta_px must give a delta for exactly "
                            f"{list(_CLASSES)}, found {sorted(huber)}")
    sigma_default = float(doc.get("sigma_px_default", 0.0))
    if not sigma_default > 0.0:
        # A zero or missing default is not a harmless omission: it would make every
        # un-overridden keypoint infinitely confident, and E_kp would ignore the rest of
        # the objective entirely.
        raise KeypointError(f"{path}: sigma_px_default must be a positive number of "
                            f"pixels, found {doc.get('sigma_px_default')!r}")

    raw = doc.get("keypoints")
    if not isinstance(raw, list) or not raw:
        raise KeypointError(f"{path}: `keypoints:` must be a non-empty list")

    problems: list[str] = []
    kps: list[Keypoint] = []
    seen_labels: set[str] = set()
    seen_sites: set[str] = set()
    for i, e in enumerate(raw):
        where = f"keypoint {i}"
        if not isinstance(e, dict):
            problems.append(f"{where}: expected a mapping, found {type(e).__name__}")
            continue
        label, site = e.get("label"), e.get("site")
        where = f"keypoint {label!r}" if label else where
        if not label or not isinstance(label, str):
            problems.append(f"{where}: missing `label`")
            continue
        if not site or not isinstance(site, str):
            problems.append(f"{where}: missing `site`")
            continue
        # Both directions of the mapping must be injective. A duplicate label makes the
        # detector's output ambiguous; a duplicate site makes one landmark count twice in
        # E_kp, which is a silent doubling of its weight.
        if label in seen_labels:
            problems.append(f"{where}: duplicate label")
        if site in seen_sites:
            problems.append(f"{where}: site {site!r} is already claimed by another "
                            f"keypoint")
        seen_labels.add(label)
        seen_sites.add(site)

        cls = e.get("class", "rigid")
        if cls not in _CLASSES:
            problems.append(f"{where}: class is {cls!r}, expected one of "
                            f"{', '.join(_CLASSES)}")
            cls = "rigid"
        sigma = e.get("sigma_px", sigma_default)
        try:
            sigma = float(sigma)
        except (TypeError, ValueError):
            problems.append(f"{where}: sigma_px is {sigma!r}, expected a number")
            sigma = sigma_default
        if not sigma > 0.0:
            problems.append(f"{where}: sigma_px must be positive, found {sigma}")
            sigma = sigma_default
        guide = " ".join(str(e.get("guide", "")).split())
        if not guide:
            # Not fatal, but worth refusing: the guide text is what stops the annotator
            # clicking the palpable prominence when the rig means the joint centre, and a
            # landmark without one carries a constant offset the anatomy fit will absorb
            # by making a bone the wrong length.
            problems.append(f"{where}: no `guide` text -- an unguided landmark is a "
                            f"systematically mis-clicked one")
        kps.append(Keypoint(label=label, site=site, cls=cls, sigma_px=sigma,
                            huber_px=float(huber[cls]), guide=guide))

    skel_raw = doc.get("skeleton") or []
    skeleton: list[tuple[str, str]] = []
    for e in skel_raw:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            problems.append(f"skeleton edge {e!r}: expected a pair of labels")
            continue
        a, b = str(e[0]), str(e[1])
        for end in (a, b):
            if end not in seen_labels:
                problems.append(f"skeleton edge [{a}, {b}]: {end!r} is not a keypoint "
                                f"label")
        skeleton.append((a, b))

    if problems:
        raise KeypointError(f"{path}: {len(problems)} problem(s):\n  "
                            + "\n  ".join(problems))

    return KeypointSet(rig=str(doc.get("rig", "")), creature=str(doc.get("creature", "")),
                       keypoints=kps, skeleton=skeleton,
                       sigma_px_default=sigma_default,
                       sigma_px_measured=bool(doc.get("sigma_px_measured", False)),
                       path=path)


def check_against_rig(kps: KeypointSet, creature) -> list[str]:
    """Return every disagreement between the correspondence and a built `Creature`.

    Empty list means they agree. Three distinct failures are checked, because they fail
    in three distinct ways downstream:

    * a keypoint naming a site the rig does not have -- the fit raises at `mj_name2id`,
      which is loud and fine;
    * a rig site no keypoint covers -- SILENT. The landmark simply never contributes,
      and the only symptom is a slightly worse fit on that limb;
    * a class disagreeing with the site's `kind` -- also silent, and it mis-weights the
      Huber: a soft landmark treated as rigid lets tissue slide drag the whole skeleton.
    """
    have = {s.name: s for s in creature.sites}
    out: list[str] = []
    for k in kps:
        site = have.get(k.site)
        if site is None:
            from ftcl.errors import did_you_mean
            out.append(f"keypoint {k.label!r} -> site {k.site!r}, which {creature.name} "
                       f"does not have" + did_you_mean(k.site, set(have)))
        elif site.kind != k.cls:
            out.append(f"keypoint {k.label!r} is class {k.cls!r} but rig site "
                       f"{k.site!r} is kind {site.kind!r}")
    for name in have:
        if name not in set(kps.sites):
            out.append(f"rig site {name!r} has no keypoint -- it would never be "
                       f"observed, and nothing would say so")
    return out
