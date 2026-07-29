"""Per-field *binding-union* validators for the material fields ftrace reads as
more than a bare spectrum (``src/ftsl.h``, ``buildMaterial`` / ``bindReflectTexture``
/ ``bindScalarTexture`` / ``bindScalarPattern``).

Three fields (and their kin) accept a **union** of forms, not a single value grammar:

* **colour bind** — ``reflect``: ``texture:<name>`` (``bindReflectTexture``),
  ``pattern:<name>`` (``patternedSpectrumParam`` — the pattern goes in the slot alone
  and the base spectrum becomes a flat 1.0) **or** a spectrum expression
  (:mod:`loom.grammar.spectrum`).  ``transmit`` is the same union minus the texture:
  ftrace binds an image albedo only on the ``reflect`` slot of a Lambertian family,
  and says so explicitly when a ``texture:`` turns up anywhere else.
* **scalar bind** — ``roughness``: ``pattern:<name>`` (``bindScalarPattern``) **or**
  ``texture:<name>`` (``bindScalarTexture``, grayscale) **or** a single scalar number
  (``dblParam``).  Not a full spectrum — ftrace reads one ``num(words[0])``.
* **scalar-map bind** — ``*_map`` (``film_thickness_map`` / ``weight_map``):
  ``pattern:<name>`` **or** ``texture:<name>`` only (no numeric fallback; the scale
  lives on the companion scalar field, e.g. ``film_thickness``).

A bound name's *existence* is a scene-level check (it needs the texture / pattern
tables, which the emitter can't see), so these validators only confirm the **token
form** — an unknown-but-well-formed name is left for the renderer to reject, exactly
as ftrace's ``bind*`` do after the shape is accepted.  This mirrors how
:func:`loom.grammar.spectrum.as_spectrum` validates shape, not scene membership.
"""

from __future__ import annotations

import re

from .values import ShapeError

# A binding target name is a plain identifier (the texture / pattern block's name).
_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")


def _ref_name(value: str, prefix: str) -> str:
    """The ``<name>`` of a ``<prefix><name>`` single-word ref, validated as an
    identifier.  Raises :class:`ShapeError` for an empty / multi-word / malformed
    name (matching ftrace, whose ``bind*`` take ``substr`` of the whole word)."""
    if len(value.split()) != 1:
        raise ShapeError(
            f"{prefix!r} binding takes a single '{prefix}<name>' word, got '{value.strip()}'")
    name = value[len(prefix):]
    if not _NAME_RE.match(name):
        raise ShapeError(f"invalid {prefix!r} binding name '{name}'")
    return name


def _is_scalar(value: str) -> bool:
    words = value.split()
    if len(words) != 1:
        return False
    try:
        float(words[0])
        return True
    except ValueError:
        return False


def as_color_binding(value: str, *, texture: bool = True):
    """Validate a colour-bindable field: a ``pattern:<name>`` drive, a
    ``texture:<name>`` image albedo (``reflect`` only — pass ``texture=False`` for
    ``transmit``), or a spectrum expression.  Returns ``("pattern"|"texture", name)``
    or the spectrum node."""
    if value.startswith("pattern:"):
        return ("pattern", _ref_name(value, "pattern:"))
    if value.startswith("texture:"):
        if not texture:
            raise ShapeError(
                f"'{value.strip()}': this slot takes a spectrum, not a texture — only "
                "the `reflect` slot of a `diffuse`/`translucent` material binds an "
                "image albedo. Use 'pattern:<name>' for a procedural drive")
        return ("texture", _ref_name(value, "texture:"))
    from .spectrum import as_spectrum   # lazy: spectrum -> values -> (reader) cycle
    return as_spectrum(value)


def as_scalar_binding(value: str):
    """Validate a scalar-bindable field (``roughness``): ``pattern:<name>`` /
    ``texture:<name>`` bind or a single scalar number.  Returns ``("pattern"|"texture",
    name)`` or ``("scalar", float)``."""
    if value.startswith("pattern:"):
        return ("pattern", _ref_name(value, "pattern:"))
    if value.startswith("texture:"):
        return ("texture", _ref_name(value, "texture:"))
    if _is_scalar(value):
        return ("scalar", float(value.split()[0]))
    raise ShapeError(
        f"'{value.strip()}' is not a scalar field value "
        "(expected a number, 'pattern:<name>' or 'texture:<name>')")


def as_map_binding(value: str):
    """Validate a scalar-map field (``*_map``): ``pattern:<name>`` or
    ``texture:<name>`` only.  Returns ``("pattern"|"texture", name)``."""
    if value.startswith("pattern:"):
        return ("pattern", _ref_name(value, "pattern:"))
    if value.startswith("texture:"):
        return ("texture", _ref_name(value, "texture:"))
    raise ShapeError(
        f"'{value.strip()}' is not a map binding "
        "(expected 'pattern:<name>' or 'texture:<name>')")
