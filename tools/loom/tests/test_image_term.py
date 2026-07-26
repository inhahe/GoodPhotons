"""J3b item 3(b) tests: :class:`loom.Image` — an image sampled as a **term inside a
formula**, the complement of binding a whole image to a material slot.

This is the one part of item 3 that was not zero-cost on the ftrace side: it needed a
new pattern-VM opcode, ``PatOp::Tex`` (``src/pattern.h``, GPU twin ``dPatternEval`` in
``src/render_cuda.cu``), reached from ftsl as ``tex:<name>(u, v)``.  So the tests here
check both halves of loom's two-backend contract:

- **emit** — the leaf writes exactly that ``tex:`` call, its ``u``/``v`` arguments are
  ordinary sub-expressions (warpable, and reachable by ``substitute`` so a material
  bundle's ``u=``/``v=`` binding flows in), and the ``texture`` declaration it needs is
  collected automatically by :meth:`Scene.add` — an author who never writes a
  ``Texture`` still gets renderable ``.ftsl``.
- **eval_np** — a faithful port of ftrace's ``Texture::sampleRgb`` / ``scalarAt``:
  the ``-0.5`` texel offset, the ``v``-flip (OBJ convention), repeat/clamp/mirror
  wrapping and the mean-of-linear-RGB reduction.  Checked against values computed by
  hand from a tiny known image, not against the implementation.

Runnable directly or under pytest.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from loom import (  # noqa: E402
    Clock, Cache, Scene, Material, Camera, Sphere, Light, Texture,
    Image, X, Y, U, V, sin,
)
from loom.ftsl_emit import EmitCtx  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_TMP = None


def _tmpdir() -> str:
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix="loom_imgterm_")
    return _TMP


def _write_png(name: str, rows) -> str:
    """Write a tiny 8-bit RGB PNG from a nested list of per-pixel gray levels."""
    from PIL import Image as _PIL
    a = np.array(rows, dtype=np.uint8)
    rgb = np.repeat(a[:, :, None], 3, axis=2)
    path = os.path.join(_tmpdir(), name)
    _PIL.fromarray(rgb, mode="RGB").save(path)
    return path


def _gray2x2() -> str:
    """A 2x2 image whose four texels are all distinct::

        row 0 (TOP):     0   255
        row 1 (BOTTOM): 128    64
    """
    return _write_png("gray2x2.png", [[0, 255], [128, 64]])


def _emit(expr, coords=("x", "y", "z"), clock=None):
    return expr.emit(coords, EmitCtx(clock=clock or Clock(t=0.0), cache=Cache()))


def _cam():
    return Camera(eye=(0, 0, 5), look_at=(0, 0, 0), res=(64, 64))


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

def test_emits_the_tex_pattern_op_over_u_and_v():
    txt = _emit(Image("a/bark.png", name="bark"))
    assert txt == "tex:bark((u),(v))"


def test_emitted_call_is_a_real_ftrace_pattern_op():
    """`tex:` must exist in the shipped pattern VM, or the .ftsl is un-renderable."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "src", "pattern.h")
    if not os.path.exists(src):           # tests may be vendored away from the tree
        return
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        body = f.read()
    assert "PatOp::Tex" in body or "Tex," in body
    assert 'compare(0, 4, "tex:")' in body


def test_image_is_an_operand_like_any_other_subexpression():
    # the whole point of the opcode: a photo multiplied into a procedural field
    e = 0.05 + 0.9 * Image("g.png", name="g") * (0.5 + 0.5 * sin(30.0 * X))
    txt = _emit(e)
    assert "tex:g((u),(v))" in txt and "sin(" in txt


def test_coordinates_are_expressions_so_the_image_can_be_warped():
    e = Image("p.png", name="p", u=U * 2.0 + 0.25, v=1.0 - V)
    txt = _emit(e)
    assert txt.startswith("tex:p(")
    assert "(u)" in txt and "(2)" in txt and "(v)" in txt


def test_substitute_reaches_into_the_coordinate_arguments():
    e = Image("p.png", name="p")
    out = e.substitute({"u": X, "v": Y})
    assert isinstance(out, Image)
    assert out.free_inputs() == frozenset()
    assert _emit(out) == "tex:p((x),(y))"


def test_free_inputs_reports_the_default_surface_coordinates():
    assert Image("p.png").free_inputs() == frozenset({"u", "v"})
    assert Image("p.png", u=X, v=Y).free_inputs() == frozenset()


# ---------------------------------------------------------------------------
# auto-naming and texture collection
# ---------------------------------------------------------------------------

def test_auto_name_is_deterministic_and_shared_across_identical_images():
    a = Image("dir/bark.png")
    b = Image("dir/bark.png")
    assert a.name == b.name
    assert a.name.startswith("img_bark_")


def test_auto_name_separates_images_that_sample_differently():
    a = Image("dir/bark.png", wrap="repeat")
    b = Image("dir/bark.png", wrap="clamp")
    c = Image("dir/bark.png", encoding="srgb")
    assert len({a.name, b.name, c.name}) == 3


def test_image_textures_dedupes_and_keeps_encounter_order():
    e = Image("a.png", name="ta") * Image("b.png", name="tb") + Image("a.png", name="ta")
    names = [t.name for t in e.image_textures()]
    assert names == ["ta", "tb"]
    assert all(isinstance(t, Texture) for t in e.image_textures())


def test_image_textures_finds_leaves_under_warped_coordinates():
    inner = Image("m.png", name="m")
    outer = Image("o.png", name="o", u=U + 0.1 * inner)
    assert sorted(t.name for t in outer.image_textures()) == ["m", "o"]


def test_declaration_carries_the_sampler_settings_through():
    t = Image("q.png", name="q", encoding="srgb", filter="nearest",
              wrap="mirror").image_textures()[0]
    txt = t.emit(EmitCtx(clock=Clock(t=0.0), cache=Cache()))
    assert "srgb" in txt and "nearest" in txt and "mirror" in txt


def test_encoding_defaults_to_linear_because_the_value_is_a_number():
    # a skin wants sRGB decode; a value used as a NUMBER wants the stored levels
    assert Image("q.png").encoding == "linear"


def test_invalid_sampler_settings_are_rejected_at_construction():
    for kw in ({"encoding": "rec709"}, {"filter": "cubic"}, {"wrap": "tile"}):
        try:
            Image("q.png", **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} must be rejected")


# ---------------------------------------------------------------------------
# Scene integration — the author never declares the texture
# ---------------------------------------------------------------------------

def test_scene_auto_declares_the_texture_a_material_field_samples():
    m = Material("wall", "glossy",
                 roughness=0.05 + 0.9 * Image("grime.png", name="grime"))
    sc = Scene(_cam())
    sc.add(m, Sphere((0, 0, 0), 1, "wall"),
           Light("point", position=(3, 3, 3), name="key"))
    txt = sc.emit(Clock(t=0.0), Cache())
    assert "grime = texture" in txt
    assert "tex:grime(" in txt
    assert "roughness pattern:wall_roughness" in txt


def test_auto_declared_texture_precedes_every_use():
    """ftrace builds textures in Pass 1b and bakes procedural ones *during* it, so a
    procedural texture may only sample images declared above it.  Emitting the
    auto-collected declaration first keeps that true for free."""
    m = Material("skinny", "diffuse",
                 reflect=(Image("p.png", name="p"), V, 0.5))
    sc = Scene(_cam())
    sc.add(m, Sphere((0, 0, 0), 1, "skinny"),
           Light("point", position=(3, 3, 3), name="key"))
    txt = sc.emit(Clock(t=0.0), Cache())
    assert txt.index("p = texture") < txt.index("skinny_reflect = texture")


def test_scene_declares_a_shared_image_exactly_once():
    sc = Scene(_cam())
    sc.add(Material("a", "glossy", roughness=Image("s.png", name="s")),
           Material("b", "glossy", roughness=1.0 - Image("s.png", name="s")),
           Sphere((0, 0, 0), 1, "a"),
           Light("point", position=(3, 3, 3), name="key"))
    txt = sc.emit(Clock(t=0.0), Cache())
    assert txt.count("s = texture") == 1


def test_an_explicit_texture_of_the_same_name_wins_in_either_order():
    """An author who *does* declare the texture must not end up with two blocks of
    the same name — regardless of whether their declaration is added before or after
    the material whose Image term would auto-collect it."""
    mat = Material("a", "glossy", roughness=Image("s.png", name="hand"))
    hand = Texture("hand", "s.png", filter="nearest")
    for order in ((hand, mat), (mat, hand)):
        sc = Scene(_cam())
        sc.add(*order)
        sc.add(Sphere((0, 0, 0), 1, "a"),
               Light("point", position=(3, 3, 3), name="key"))
        txt = sc.emit(Clock(t=0.0), Cache())
        assert txt.count("hand = texture") == 1
        assert "nearest" in txt          # the explicit declaration is the one kept


def test_image_in_a_scalar_slot_is_allowed():
    """A scalar slot lowers to a *live* pattern, which ftrace evaluates through
    `patCtxFromHit` — u/v are in scope there, unlike in a baked colour skin."""
    m = Material("w", "glossy", roughness=Image("g.png", name="g"))
    comps, resolved = m.expand("w")
    assert resolved.props["roughness"] == "pattern:w_roughness"
    assert "tex:g(" in _emit(comps[-1].template)


# ---------------------------------------------------------------------------
# eval_np — the numpy twin of Texture::sampleRgb / scalarAt
# ---------------------------------------------------------------------------

def _coords(us, vs):
    """`eval_np` indexes coords positionally by axis, so (x, y, z) arrays."""
    xs = np.asarray(us, dtype=np.float64)
    ys = np.asarray(vs, dtype=np.float64)
    return (xs, ys, np.zeros_like(xs))


def _sample(img: Image, us, vs):
    return img.eval_np(_coords(us, vs), Clock(t=0.0), Cache())


def _nearest(path, **kw):
    return Image(path, u=X, v=Y, filter="nearest", **kw)


def test_nearest_sampling_hits_the_expected_texel():
    p = _gray2x2()
    img = _nearest(p)
    # v is flipped: v=0 is the BOTTOM row (row index 1), v=1 the top (row 0)
    got = _sample(img, [0.25, 0.75, 0.25, 0.75], [0.25, 0.25, 0.75, 0.75])
    want = np.array([128, 64, 0, 255], dtype=np.float64) / 255.0
    assert np.allclose(got, want, atol=1e-6)


def test_value_is_the_mean_of_the_three_linear_channels():
    from PIL import Image as _PIL
    path = os.path.join(_tmpdir(), "rgb1x1.png")
    _PIL.fromarray(np.array([[[30, 60, 231]]], dtype=np.uint8), "RGB").save(path)
    got = _sample(_nearest(path), [0.5], [0.5])
    assert abs(float(got[0]) - (30 + 60 + 231) / 3.0 / 255.0) < 1e-6


def test_bilinear_uses_the_half_texel_offset_and_flipped_v():
    p = _gray2x2()
    img = Image(p, u=X, v=Y, filter="bilinear", wrap="clamp")
    # at the exact centre of the image the four texels weigh equally
    got = _sample(img, [0.5], [0.5])
    assert abs(float(got[0]) - (0 + 255 + 128 + 64) / 4.0 / 255.0) < 1e-6
    # at a texel centre (u=0.25, v=0.75 -> row 0, col 0) the sample IS that texel
    got = _sample(img, [0.25], [0.75])
    assert abs(float(got[0]) - 0.0) < 1e-6


def test_repeat_clamp_and_mirror_differ_outside_the_unit_square():
    p = _gray2x2()
    us, vs = [1.25], [0.25]                    # one full period to the right
    rep = _sample(_nearest(p, wrap="repeat"), us, vs)
    cla = _sample(_nearest(p, wrap="clamp"), us, vs)
    mir = _sample(_nearest(p, wrap="mirror"), us, vs)
    assert abs(float(rep[0]) - 128 / 255.0) < 1e-6   # wraps back to column 0
    assert abs(float(cla[0]) - 64 / 255.0) < 1e-6    # pinned to the last column
    assert abs(float(mir[0]) - 64 / 255.0) < 1e-6    # reflects onto column 1


def test_srgb_encoding_applies_the_eotf():
    p = _write_png("mid.png", [[128]])
    lin = _sample(_nearest(p, encoding="linear"), [0.5], [0.5])
    srg = _sample(_nearest(p, encoding="srgb"), [0.5], [0.5])
    assert abs(float(lin[0]) - 128 / 255.0) < 1e-6
    assert abs(float(srg[0]) - ((128 / 255.0 + 0.055) / 1.055) ** 2.4) < 1e-6
    assert float(srg[0]) < float(lin[0])


def test_eval_np_composes_with_the_rest_of_the_algebra():
    p = _gray2x2()
    e = 1.0 - _nearest(p)
    got = e.eval_np(_coords([0.25], [0.25]), Clock(t=0.0), Cache())
    assert abs(float(np.asarray(got).ravel()[0]) - (1.0 - 128 / 255.0)) < 1e-6


def test_eval_np_preserves_the_coordinate_grid_shape():
    p = _gray2x2()
    gx, gy = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 5))
    out = _nearest(p).eval_np((gx, gy, np.zeros_like(gx)), Clock(t=0.0), Cache())
    assert np.asarray(out).shape == gx.shape


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    try:
        for fn in fns:
            try:
                fn()
                print(f"  PASS  {fn.__name__}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAIL  {fn.__name__}: {e}")
    finally:
        if _TMP:
            shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
