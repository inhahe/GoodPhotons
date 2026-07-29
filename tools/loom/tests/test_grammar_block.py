"""J3c read direction: `.ftsl` text -> loom elements (`loom.block.Block`/`Document`,
`parse_element` / `parse_elements` / `parse_document`).

loom's emitters run one way — a `.ftsl` file is a *baked* snapshot of one frame, so the
reader cannot recover the authoring object that produced it (an `isosurface`'s field,
freq, rotation and drift are all flattened into a single `function { expr "…" }`
string).  The property that IS achievable, and the one these tests pin down, is
**round-trip fidelity**: `parse -> emit` reproduces the source byte for byte, including
line layout, alignment padding, comments and (at file scope, via `Document`) the text
*between* elements — so an editor that loads a scene, changes one block and writes it back
leaves every other line untouched.

Runnable directly (`python tests/test_grammar_block.py`) or under pytest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from loom import Cache, Clock  # noqa: E402
from loom.block import Block, Document, Stmt  # noqa: E402
from loom.ftsl_emit import EmitCtx  # noqa: E402
from loom.grammar.reader import parse_document, parse_element, parse_elements  # noqa: E402
from loom.material import MixMaterial  # noqa: E402
from loom.scene import Camera, Light, Material, NamedSpectrum  # noqa: E402


def rt(src: str) -> str:
    """parse -> emit, for a single element."""
    return parse_element(src).emit()


# ---- what becomes a Block, and what doesn't -------------------------------

def test_baked_kinds_become_blocks():
    b = parse_element('iso = isosurface { function { expr "x-y" }  resolution 64 }')
    assert isinstance(b, Block)
    assert (b.kind, b.name) == ("isosurface", "iso")
    assert b.get("resolution") == "64"
    assert b.find("function").get("expr") == '"x-y"'


def test_faithful_kinds_still_build_their_class():
    # A block that maps onto an authoring class without loss builds the real thing;
    # Block is the fallback for the baked kinds, not a replacement for the readers.
    assert isinstance(parse_element("m = material { type diffuse  reflect 0.75 }"), Material)
    assert isinstance(parse_element("light { kind area  spd preset:bb6500 }"), Light)
    assert isinstance(parse_element('spectrum "steel" = rgb 0.55 0.57 0.6'), NamedSpectrum)
    assert isinstance(parse_element(
        "cam = camera { eye 0 0 2  look_at 0 0 0  up 0 1 0  fov_y 40\n"
        "    mode R\n    film { res 64 64 }\n}"), Camera)


def test_mix_material_reads_its_repeated_layer_lines():
    # The one material whose body has a repeated key: folding it into a dict would
    # keep only the last `layer` and silently build a one-layer mix.
    src = ('m = material {\n    type mix\n    layer "glass" 0.5\n'
           '    layer "steel" 0.5\n    weight_map pattern:wiggle\n}')
    m = parse_element(src)
    assert isinstance(m, MixMaterial)
    assert m.layers == [("glass", 0.5), ("steel", 0.5)]
    assert m.weight_map == "wiggle"
    assert m.emit(None) == src


# ---- ordered entries, duplicate keys, valueless keywords ------------------

def test_duplicate_keys_are_kept_in_order():
    b = parse_element("c = camera_curve {\n    point 0 1 3\n    point 1 1 1\n"
                      "    point 2 1 3\n}")
    assert [s.numbers() for s in b.stmts("point")] == [[0, 1, 3], [1, 1, 1], [2, 1, 3]]


def test_valueless_keyword_is_present_but_empty():
    b = parse_element("c = camera_curve {\n    closed\n    fps 24\n}")
    assert b.has("closed") and b.get("closed") == ""     # present, no value
    assert b.get("nope") is None                          # absent
    assert rt("c = camera_curve {\n    closed\n    fps 24\n}") == \
        "c = camera_curve {\n    closed\n    fps 24\n}"


# ---- round-trip fidelity --------------------------------------------------

@pytest.mark.parametrize("src", [
    # single-line body stays on one line
    'quad { origin 0 0 0  u 1 0 0  v 0 1 0  material white }',
    # multi-line body keeps its indentation, and a nested block keeps its own line
    'medium {\n    bounds { min 0.2 0.2 0.2  max 0.8 0.8 0.8 }\n'
    '    density "0.5+0.5*sin(x*10)"\n}',
    # a quoted name on the header, and a `key=value` word with a NON-numeric value
    'upsample "basis" {\n    kind rbf\n    uv planar axis=y\n}',
    # legacy bareword subtype (`light area {`), which loom itself doesn't emit
    'light area { origin 0 1 0  u 1 0 0 }',
    # the header's first statement sharing its line, the rest indented below
    'group { translate -1.75 5.75 5.6   scale 250\n    scale 2\n}',
    # a numeric statement key (ftrace takes any token as a key)
    'palette { 0 spectrum:navy  1 spectrum:crimson }',
])
def test_round_trip_is_byte_identical(src):
    assert rt(src) == src


def test_alignment_padding_survives():
    # Hand-written scenes column-align their values and their braces; re-emitting the
    # canonical single spaces would reflow lines nobody edited.
    src = ('material "red"   { type diffuse    reflect redwall   }\n'
           'material "green" { type diffuse    reflect greenwall }')
    assert "\n".join(e.emit() for e in parse_elements(src)) == src


def test_comments_and_blank_lines_survive():
    # Comments are a lexer skip, so they exist only in the source text; losing them
    # would be the most destructive thing a round-trip could do to a real scene.
    # (no binder: ftrace's `assign_header` takes no quoted name, so `c = … "swoop"`
    # is a form ftrace itself rejects and loom must not invent.)
    src = ('camera_curve "swoop" {\n'
           '    point 1.7  0.9  2.2    # enter, high and back\n'
           '    # a whole-line note\n'
           '\n'
           '    point 1.0  0.4  0.2\n'
           '    # trailing note before the brace\n'
           '}')
    assert rt(src) == src


def test_two_space_and_three_space_gaps_are_distinguished():
    # loom's own emitters are not uniform: camera_curve writes three spaces between
    # statements sharing a line where a material writes two.
    assert rt("c = camera_curve {\n    up 0 1 0   fov_y 40   mode R\n}") == \
        "c = camera_curve {\n    up 0 1 0   fov_y 40   mode R\n}"


# ---- whole-file reading ---------------------------------------------------

_FILE = '''scene { units meters  spectral 360 830 1 }

white = material { type diffuse  reflect 0.75 }

iso = isosurface {
    material "white"
    function { expr "sin(x)*cos(y)" }
    uv planar axis=y
}

light { kind area  origin 0.35 0.99 0.35  spd preset:bb6500 }'''


def test_parse_elements_reads_a_whole_file_in_order():
    els = parse_elements(_FILE)
    assert [type(e).__name__ for e in els] == \
        ["Block", "Material", "Block", "Light"]
    assert [e.kind for e in els if isinstance(e, Block)] == ["scene", "isosurface"]


def test_parse_elements_round_trips_every_block():
    for el in parse_elements(_FILE):
        if isinstance(el, Block):
            assert el.emit() in _FILE


# ---- whole-file layout (Document) -----------------------------------------

def _ctx() -> EmitCtx:
    return EmitCtx(clock=Clock(t=0.0), cache=Cache())


_MESSY_FILE = ('# what this scene is\n'
               '\n'
               'scene { units meters }\n'
               '\n'
               '\n'
               '# --- the walls ---\n'
               'white = material { type diffuse  reflect 0.75 }\n'
               '\n'
               'iso = isosurface {\n'
               '    function { expr "sin(x)" }\n'
               '}\n'
               '# a closing note\n')


def test_document_round_trips_a_whole_file_byte_for_byte():
    # Blank lines that group elements and top-level comments belong to NEITHER
    # neighbouring element, so only a Document can carry them.
    assert parse_document(_MESSY_FILE).emit(_ctx()) == _MESSY_FILE


def test_document_gaps_hold_the_between_text():
    d = parse_document(_MESSY_FILE)
    assert len(d) == 3 and len(d.gaps) == 4
    assert d.gaps[0] == "# what this scene is\n\n"
    assert d.gaps[1] == "\n\n\n# --- the walls ---\n"    # after `scene`, before `white`
    assert d.gaps[-1] == "\n# a closing note\n"


def test_document_edit_leaves_every_other_line_alone():
    d = parse_document(_MESSY_FILE)
    d.blocks("isosurface")[0].set("resolution", "128")
    assert d.emit(_ctx()) == _MESSY_FILE.replace(
        '    function { expr "sin(x)" }\n',
        '    function { expr "sin(x)" }\n    resolution 128\n')


def test_document_append_keeps_the_trailing_text_last():
    d = parse_document('a = quad { origin 0 0 0 }\n')
    d.append(parse_element("b = quad { origin 1 0 0 }"))
    assert d.emit(_ctx()) == \
        'a = quad { origin 0 0 0 }\n\nb = quad { origin 1 0 0 }\n'


def test_document_pop_keeps_the_file_head_and_tail():
    d = parse_document(_MESSY_FILE)
    d.pop(1)                                    # drop the material
    out = d.emit(_ctx())
    assert out.startswith("# what this scene is\n\nscene {")
    assert out.endswith("}\n# a closing note\n")
    assert "material" not in out


def test_document_rejects_a_mismatched_gap_count():
    with pytest.raises(ValueError):
        Document([parse_element("q = quad { origin 0 0 0 }")], gaps=["", "", ""])


# ---- mutation -------------------------------------------------------------

def test_set_rewrites_a_value_and_drops_its_verbatim_slice():
    b = parse_element("iso = isosurface {\n    resolution    64\n    method sample\n}")
    b.set("resolution", "128")
    assert b.emit() == "iso = isosurface {\n    resolution 128\n    method sample\n}"


def test_add_and_remove():
    b = parse_element("m = mesh {\n    file \"a.obj\"\n}")
    b.add(Stmt("smooth", ["on"]))
    assert b.get("smooth") == "on"
    assert b.remove("file") == 1
    assert b.emit() == "m = mesh {\n    smooth on\n}"


def test_same_as_ignores_layout():
    a = parse_element("q = quad { origin 0 0 0  u 1 0 0 }")
    c = parse_element("q = quad {\n    origin    0 0 0\n    u 1 0 0\n}")
    assert a.same_as(c) and a.emit() != c.emit()


# ---- Element contract -----------------------------------------------------

def test_block_holds_no_signals():
    # A Block came from baked text; the base Element.roots() would just churn over
    # strings looking for signal sites that cannot be there.
    assert parse_element("q = quad { origin 0 0 0 }").roots() == []


if __name__ == "__main__":       # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
