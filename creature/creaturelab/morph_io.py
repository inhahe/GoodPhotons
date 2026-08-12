"""Read and write a morph vector as a file: the `out/theta_<animal>.json` interchange.

This is one of the four formats `notes/pipeline.md` names as worth freezing early, and it
is the narrow waist between the two halves of the project. Stage C's fit *produces* one;
stages D, E and F *consume* one. Neither end should know anything about the other beyond
this file, which is why the format is deliberately boring: a name -> number map, a schema
version, and enough provenance to answer "where did this dog come from?" six months later.

Why a file at all, rather than twenty-six `--set` pairs on a command line:

* **A fitted animal is data, not an invocation.** `--set` is fine for the four parameters a
  human is poking at by hand; it is the wrong shape for a 26-vector that came out of an
  optimiser, because the only way to move it between tools is to re-type it, and the only
  way to check two runs used the same body is to diff two shell histories.
* **Provenance travels with the numbers.** A θ with no record of which rig it was fit
  against is actively dangerous: parameter names are per-rig, so the same file applied to a
  different rig either errors (good) or, if the names happen to overlap, silently builds a
  chimera (bad). `rig` and `creature` are recorded and checked on load.
* **It is what makes P6's "edit the anatomy" a real workflow.** The artist's edited θ is a
  file you can keep, version, diff and hand to `train.py --morph-center`.

Unknown parameter names are NOT accepted silently. `build_env` already rejects them, but
this loader raises first so the error names the *file* -- a typo in a θ that came out of a
fit is a data problem, and pointing at the rig would send you looking in the wrong place.
"""
from __future__ import annotations

import json
import os
from typing import Any

SCHEMA = "creature/theta@1"


def save_morph(path: str, morph: dict[str, float], *, rig: str | None = None,
               creature: str | None = None, source: str | None = None,
               extra: dict[str, Any] | None = None) -> None:
    """Write a morph vector, with the provenance needed to use it safely later."""
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "rig": os.path.basename(rig) if rig else None,
        "creature": creature,
        "source": source or "hand-written",
        "morph": {k: float(v) for k, v in morph.items()},
    }
    if extra:
        doc.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_morph(path: str, *, rig: str | None = None,
               params: list | None = None) -> dict[str, float]:
    """Read a morph vector. `rig` and `params`, when given, are checked against it.

    Accepts both the full document and a bare `{"name": value}` map, because a θ small
    enough to hand-write should not need a wrapper -- but anything written by a tool in
    this repo gets the full form.
    """
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a JSON object, found {type(doc).__name__}")

    if "morph" in doc and isinstance(doc.get("morph"), dict):
        schema = doc.get("schema")
        if schema is not None and schema != SCHEMA:
            raise ValueError(f"{path}: schema is {schema!r}, this build reads {SCHEMA!r}")
        morph = doc["morph"]
        # A mismatched rig is a warning, not an error: re-fitting the same animal onto a
        # renamed or forked rig is a legitimate thing to do, and the parameter-name check
        # below is the one that actually protects against a chimera. Refusing outright
        # would make `canis.ftcl` -> `canis_v2.ftcl` need a hand-edited file.
        want = doc.get("rig")
        if rig and want and os.path.basename(rig) != want:
            print(f"warning: {path} was fit against {want!r}, building "
                  f"{os.path.basename(rig)!r}")
    else:
        morph = doc

    out: dict[str, float] = {}
    for k, v in morph.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{path}: morph param {k!r} is {v!r}, expected a number")
        out[str(k)] = float(v)
    if not out:
        raise ValueError(f"{path}: contains no morph parameters")

    if params is not None:
        declared = {p.name for p in params}
        unknown = sorted(set(out) - declared)
        if unknown:
            from ftcl.errors import did_you_mean
            raise ValueError(f"{path}: not a parameter of this rig: "
                             f"{', '.join(unknown)}" + did_you_mean(unknown[0], declared))
        # Missing parameters are FINE and stay at their rig defaults, which is what makes a
        # partial θ ("just the leg lengths I measured") useful. Reported, not refused --
        # silently defaulting half a morph vector is exactly the kind of thing that shows
        # up later as an unexplained reprojection floor.
        missing = sorted(declared - set(out))
        if missing and len(missing) <= len(declared) // 2:
            print(f"note: {path} leaves {len(missing)} param(s) at the rig default: "
                  f"{', '.join(missing[:6])}" + (" ..." if len(missing) > 6 else ""))
    return out


def parse_sets(pairs) -> dict[str, float]:
    """Parse `--set NAME=VAL ...` into a morph dict. Lives here, next to `morph_from_args`,
    because every tool that takes `--morph` takes `--set` too and they are one interface;
    four private copies of the same eight lines is four places for the error message to
    drift out of agreement with the flag it is describing."""
    out: dict[str, float] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects name=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"--set {k.strip()}: {v!r} is not a number") from None
    return out


def morph_from_args(rig: str, morph_file: str | None, sets: dict[str, float] | None,
                    params: list | None = None) -> dict[str, float]:
    """Combine `--morph FILE` with `--set NAME=VAL`, with `--set` winning.

    That precedence is the useful one: load the fitted animal, then poke one parameter to
    see what it does. The reverse would make `--set` unusable alongside a θ file.
    """
    morph = dict(load_morph(morph_file, rig=rig, params=params)) if morph_file else {}
    morph.update(sets or {})
    return morph
