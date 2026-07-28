"""§F1 — the loom↔viewer data contract: load_build + scene introspection sidecar."""

import json
import sys
import textwrap

import pytest

from loom.viewer import (
    SIDECAR_VERSION, load_build, build_scene, introspect, ViewerModel,
    ViewerSession, serve_viewer,
)
from loom.signals.core import Clock
from loom.scene import Scene, Camera, Material, Sphere, Light, tube
from loom.data import PointPath, Grid, Scatter
from loom.signals import Sine


# --------------------------------------------------------------------------
# a build() contract scene exercising every enumerated object kind
# --------------------------------------------------------------------------

def build(clock=None, *, radius=0.12):
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    spine = PointPath([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], closed=True)
    tb = tube(spine, radius=radius, material="skin", name="worm")
    tb.twist = Sine(cycles=1)          # animate → non-trivial DAG (scale stays = radius)
    grid = Grid([0.0, 1.0, 2.0, 3.0], shape=[2, 2], channels=["h"])
    scat = Scatter([((0.0, 0.0), 1.0), ((1.0, 1.0), 2.0)])
    mat = Material("skin", "diffuse", roughness=grid(0.5, 0.5))  # references Grid
    ball = Sphere((2, 0, 0), scat(0.5, 0.5), "skin")             # references Scatter
    sc = Scene(cam)
    sc.add(mat, tb, ball, Light("point", intensity=1.0))
    return sc


# --------------------------------------------------------------------------
# load_build / the contract
# --------------------------------------------------------------------------

def _write_scene_file(tmp_path, body):
    p = tmp_path / "scene.py"
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_load_build_returns_callable(tmp_path):
    path = _write_scene_file(tmp_path, """
        from loom.scene import Scene, Camera
        def build(clock=None, **params):
            return Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
    """)
    fn = load_build(path)
    sc = fn()
    assert isinstance(sc, Scene)


def test_load_build_missing_func_raises(tmp_path):
    path = _write_scene_file(tmp_path, "x = 1\n")
    with pytest.raises(AttributeError):
        load_build(path)


def test_load_build_not_callable_raises(tmp_path):
    path = _write_scene_file(tmp_path, "build = 42\n")
    with pytest.raises(TypeError):
        load_build(path)


def test_load_build_custom_func_name(tmp_path):
    path = _write_scene_file(tmp_path, """
        from loom.scene import Scene, Camera
        def make(clock=None):
            return Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
    """)
    fn = load_build(path, func="make")
    assert isinstance(fn(), Scene)


# --------------------------------------------------------------------------
# build_scene — clock passing
# --------------------------------------------------------------------------

def test_build_scene_passes_clock_when_accepted():
    seen = {}

    def b(clock=None):
        seen["clock"] = clock
        return build()  # reuse the rich scene

    clk = Clock.at_frame(3, 10)
    build_scene(b, clk)
    assert seen["clock"] is clk


def test_build_scene_omits_clock_when_not_accepted():
    def b(*, radius=0.1):
        return build(radius=radius)

    # must not raise despite passing a clock
    sc = build_scene(b, Clock.at_frame(0, 1))
    assert isinstance(sc, Scene)


def test_build_scene_forwards_params():
    def b(clock=None, *, radius=0.1):
        return build(radius=radius)

    sc = build_scene(b, None, radius=0.9)
    # the tube's SweptMesh scale is 0.9 (radius routed into tube scale)
    swept = [e for e in sc.elements if type(e).__name__ == "SweptMesh"][0]
    assert swept.scale == pytest.approx(0.9)


# --------------------------------------------------------------------------
# introspect — the sidecar
# --------------------------------------------------------------------------

def test_introspect_top_level_shape():
    sc = build()
    d = introspect(sc, clock=Clock.at_frame(2, 8))
    assert d["version"] == SIDECAR_VERSION
    assert d["frame"] == {"frame": 2, "frames": 8}
    assert {"objects", "datasets", "camera", "lights", "dag"} <= set(d)


def test_introspect_objects_kinds_and_material():
    d = introspect(build())
    kinds = {o["kind"] for o in d["objects"]}
    assert "swept_mesh" in kinds
    assert "sphere" in kinds
    swept = [o for o in d["objects"] if o["kind"] == "swept_mesh"][0]
    assert swept["name"] == "worm"
    assert swept["material"] == "skin"
    assert swept["closed_spine"] is True


def test_introspect_swept_mesh_links_its_dataset():
    d = introspect(build())
    swept = [o for o in d["objects"] if o["kind"] == "swept_mesh"][0]
    assert "datasets" in swept and swept["datasets"]
    ds_ids = {ds["id"] for ds in d["datasets"]}
    # every dataset the object references is present in the global dataset list
    assert set(swept["datasets"]) <= ds_ids


def test_introspect_swept_mesh_carries_tessellated_geometry():
    """F4: a swept_mesh object carries its tessellated triangle mesh — vertices,
    0-based face index triples, per-vertex UVs, and the ring/profile lattice shape —
    so the viewer's 3-D pane can draw the surface without re-running loom."""
    d = introspect(build())
    swept = [o for o in d["objects"] if o["kind"] == "swept_mesh"][0]
    m = swept["mesh"]
    n, k = m["rings"], m["profile_count"]
    assert n == swept["count"]
    # closed spine + closed profile tube: n rings x k profile pts
    assert len(m["vertices"]) == n * k
    assert all(len(v) == 3 for v in m["vertices"])
    assert len(m["uvs"]) == n * k
    assert all(len(uv) == 2 for uv in m["uvs"])
    # closed both ways -> 2 tris per quad over n spans x k edges
    assert len(m["faces"]) == 2 * n * k
    # faces index valid vertices (0-based)
    nv = len(m["vertices"])
    for f in m["faces"]:
        assert len(f) == 3 and all(0 <= i < nv for i in f)


def test_introspect_iso_mesh_carries_marching_cubes_geometry():
    """F7 (MC fallback): an iso_mesh object carries its marching-cubes triangle
    mesh (vertices + 0-based faces) baked at the clock, so the Meshes tab can draw
    the isosurface without a GPU raymarch."""
    from loom import X, Y, Z, IsoMesh
    from loom.scene import Scene, Camera
    cam = Camera(eye=(0, 0, 3), look_at=(0, 0, 0))
    sc = Scene(cam)
    sphere = X * X + Y * Y + Z * Z + (-(0.6 * 0.6))   # unit-ish sphere field
    sc.add(IsoMesh(sphere, bounds=1.0, res=20, iso=0.0, material="skin", name="ball"))
    d = introspect(sc)
    iso = [o for o in d["objects"] if o["kind"] == "iso_mesh"][0]
    m = iso["mesh"]
    assert len(m["vertices"]) > 100 and all(len(v) == 3 for v in m["vertices"])
    nv = len(m["vertices"])
    assert len(m["faces"]) > 100
    for f in m["faces"]:
        assert len(f) == 3 and all(0 <= i < nv for i in f)
    # baked at iso=0 -> vertices lie near radius 0.6
    import math
    rad = [math.sqrt(sum(c * c for c in v)) for v in m["vertices"]]
    assert abs(sum(rad) / len(rad) - 0.6) < 0.05


def test_introspect_open_swept_mesh_face_count():
    """An open ribbon (open spine + open profile) skins (n-1) spans x (k-1) edges."""
    from loom.scene import Scene, Camera, ribbon
    from loom.data import PointPath
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    sc = Scene(cam)
    sc.add(ribbon(PointPath([(0, 0, 0), (1, 0, 0), (2, 1, 0)], closed=False),
                  width=0.3, count=8, closed_spine=False, material="m"))
    d = introspect(sc)
    swept = [o for o in d["objects"] if o["kind"] == "swept_mesh"][0]
    m = swept["mesh"]
    n, k = m["rings"], m["profile_count"]
    assert len(m["faces"]) == 2 * (n - 1) * (k - 1)


def test_introspect_datasets_cover_path_grid_scatter():
    d = introspect(build())
    kinds = {ds["kind"] for ds in d["datasets"]}
    assert {"path", "grid", "scatter"} <= kinds
    grid = [ds for ds in d["datasets"] if ds["kind"] == "grid"][0]
    assert grid["shape"] == [2, 2]
    assert grid["channels"] == ["h"]
    path = [ds for ds in d["datasets"] if ds["kind"] == "path"][0]
    assert path["dim"] == 3 and path["closed"] is True and path["count"] == 4


def test_introspect_path_carries_geometry():
    d = introspect(build())
    path = [ds for ds in d["datasets"] if ds["kind"] == "path"][0]
    # control points: 4 points, each 3-D
    assert len(path["control_points"]) == 4
    assert all(len(p) == 3 for p in path["control_points"])
    assert path["control_points"][0] == [0.0, 0.0, 0.0]
    # polyline: sampled + closed (first point repeated at the end)
    poly = path["polyline"]
    assert len(poly) == 96 + 1
    assert poly[0] == poly[-1]
    assert all(len(p) == 3 for p in poly)


def test_introspect_open_path_polyline_not_wrapped(tmp_path):
    from loom.data import PointPath
    from loom.scene import Scene, Camera, Beads
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    sc = Scene(cam)
    sc.add(Beads(PointPath([(0, 0, 0), (1, 1, 1), (2, 0, 0)], closed=False),
                 count=3, radius=0.1, material="m"))
    d = introspect(sc)
    path = [ds for ds in d["datasets"] if ds["kind"] == "path"][0]
    assert path["closed"] is False
    assert len(path["polyline"]) == 96          # not wrapped
    assert path["polyline"][0] != path["polyline"][-1]


def test_introspect_tracked_path_channels():
    """F3: a tracked_path dataset carries its tacked-on tracks sampled along the
    same curve parameter as the display polyline (one strip-chart series each)."""
    from loom.data import TrackedPath
    from loom.scene import Beads
    tp = TrackedPath([(0, 0, 0), (1, 1, 1), (2, 0, 0), (0, 2, 0)],
                     tracks={"speed": [1.0, 2.0, 3.0, 4.0],           # scalar track
                             "aim": [(1, 0), (0, 1), (-1, 0), (0, -1)]},  # 2-D vector
                     closed=True)
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    sc = Scene(cam)
    sc.add(Beads(tp, count=4, radius=0.1, material="m"))
    d = introspect(sc)
    tpd = [ds for ds in d["datasets"] if ds["kind"] == "tracked_path"][0]
    assert set(tpd["tracks"]) == {"speed", "aim"}
    chans = {c["name"]: c for c in tpd["channels"]}
    assert chans["speed"]["scalar"] is True and chans["speed"]["dim"] == 1
    assert chans["aim"]["scalar"] is False and chans["aim"]["dim"] == 2
    # closed → samples wrap (first repeated), matching the polyline length
    assert len(chans["speed"]["samples"]) == 96 + 1
    assert chans["speed"]["samples"][0] == chans["speed"]["samples"][-1]
    assert all(len(s) == 1 for s in chans["speed"]["samples"])
    assert all(len(s) == 2 for s in chans["aim"]["samples"])


def test_introspect_grid_carries_field_geometry():
    """F6: a grid dataset carries its fixed lattice axis coordinates and the flat
    C-order value grid (one channel-vector per node), evaluated at the clock."""
    d = introspect(build())
    grid = [ds for ds in d["datasets"] if ds["kind"] == "grid"][0]
    # 2x2 lattice → 2 axes, each with 2 coords
    assert len(grid["axes"]) == 2
    assert all(len(ax) == 2 for ax in grid["axes"])
    # flat values in C order, each a 1-list (scalar field), matching [0,1,2,3]
    assert grid["values"] == [[0.0], [1.0], [2.0], [3.0]]


def test_introspect_scatter_carries_field_geometry():
    """F6: a scatter dataset carries its sample positions and channel values."""
    d = introspect(build())
    scat = [ds for ds in d["datasets"] if ds["kind"] == "scatter"][0]
    assert scat["points"] == [[0.0, 0.0], [1.0, 1.0]]
    assert scat["values"] == [[1.0], [2.0]]


def test_introspect_vector_scatter_field_geometry():
    """F6: a vector-valued scatter emits multi-component value vectors."""
    from loom.data import Scatter
    from loom.scene import Scene, Camera, Sphere
    from loom.signals import vec
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    sc = Scene(cam)
    scat = Scatter([((0.0, 0.0), vec(1.0, 0.0, 0.0)),
                    ((1.0, 0.0), vec(0.0, 1.0, 0.0))], channels=["r", "g", "b"])
    sc.add(Sphere((0, 0, 0), scat(0.5, 0.0).channel(0), "m"))
    d = introspect(sc)
    ds = [x for x in d["datasets"] if x["kind"] == "scatter"][0]
    assert ds["is_vector"] is True and ds["value_dim"] == 3
    assert ds["values"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_introspect_dag_nodes_and_edges():
    d = introspect(build())
    dag = d["dag"]
    ids = {n["id"] for n in dag["nodes"]}
    assert ids  # non-empty
    # every edge references known nodes and carries a param label (F5)
    for e in dag["edges"]:
        assert e["src"] in ids and e["dst"] in ids
        assert isinstance(e["param"], str) and e["param"]
    # the animated Sine node is present
    ops = {n["op"] for n in dag["nodes"]}
    assert "Sine" in ops


def test_introspect_dag_edge_param_labels():
    """F5: edges name the destination parameter the upstream node feeds — e.g. an
    Add's two operands are labelled `a` and `b`, not just positional indices."""
    from loom.signals import Sine
    # (sine*2) + 1 : Add(Mul(Sine, 2), 1) → the Mul feeds Add.a, and Sine feeds Mul.a
    sig = Sine(cycles=1) * 2.0 + 1.0
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    sc = Scene(cam)
    sc.add(Sphere((0, 0, 0), sig, "m"))
    dag = introspect(sc)["dag"]
    op_of = {n["id"]: n["op"] for n in dag["nodes"]}
    # collect the param label used on each (op_src -> op_dst) edge
    labelled = {(op_of[e["src"]], op_of[e["dst"]], e["param"]) for e in dag["edges"]}
    assert ("Mul", "Add", "a") in labelled
    assert ("Sine", "Mul", "a") in labelled


# ---------------------------------------------------------------------------
# §E5 — the on-disk projection of axis annotations (sidecar v2)
# ---------------------------------------------------------------------------

def _axis_scene():
    """A scene whose radius is a GAIN Target and whose centre is a curve sampled
    along its own `s` axis and swept over the loop — i.e. every E5 feature at a
    real value-site."""
    from loom import (Target, GAIN, mod, pin, Sine, lower, CurveSample, Ax,
                      Ramp, PointPath, LoopCurve)
    from loom.signals.core import Const
    curve = LoopCurve(PointPath([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                                closed=True), Const(0.0))
    sc = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
    sc.add(Sphere(lower(CurveSample(curve, Ax("s")), dim=3, bind={"s": Ramp()}),
                  Target(GAIN, [mod(0.6 + 0.4 * Sine(), 0.8), pin(0.5, 0.25)],
                         base=0.3),
                  "m"))
    return sc


def test_sidecar_is_version_2():
    from loom.viewer import SIDECAR_VERSION
    assert SIDECAR_VERSION == 2
    assert introspect(build())["version"] == 2


def test_dag_nodes_carry_their_axis_set():
    dag = introspect(_axis_scene())["dag"]
    by_op = {}
    for n in dag["nodes"]:
        by_op.setdefault(n["op"], []).append(n)
    # the curve sample genuinely depends on BOTH the arclength and the clock
    assert by_op["CurveSample"][0]["axes"] == ["s", "t"]
    # a coordinate leaf names itself; a constant is the empty (broadcast) set
    assert by_op["Ax"][0] == {**by_op["Ax"][0], "axes": ["s"], "leaf_axis": "s"}
    assert any(n["axes"] == [] for n in by_op["AConst"])


def test_dag_target_node_declares_its_quantity_kind():
    dag = introspect(_axis_scene())["dag"]
    tgt = [n for n in dag["nodes"] if n["op"] == "Target"][0]
    assert tgt["target_kind"] == "gain" and tgt["neutral"] == 1.0
    assert tgt["axes"] == ["t"]          # base ∅ ∪ driver {t}


def test_dag_influence_edges_carry_mode_and_gain():
    """The pin/mod edge attributes are invisible in a plain child list (sources
    hang off Binding records), so an editor could not tell a mod from a pin."""
    dag = introspect(_axis_scene())["dag"]
    tgt = [n for n in dag["nodes"] if n["op"] == "Target"][0]
    infl = {e["param"]: e for e in dag["edges"] if e["dst"] == tgt["id"]
            and "mode" in e}
    assert infl["mod[0]"]["mode"] == "mod" and infl["mod[0]"]["gain"] == 0.8
    assert infl["pin[1]"]["mode"] == "pin" and infl["pin[1]"]["gain"] == 0.25
    # the base is a plain input, not an influence edge
    base = [e for e in dag["edges"] if e["dst"] == tgt["id"] and e["param"] == "base"]
    assert base and "mode" not in base[0]


def test_dag_bridge_nodes_report_the_value_site_scope():
    dag = introspect(_axis_scene())["dag"]
    lv = [n for n in dag["nodes"] if n["op"] == "LowerVec"][0]
    assert lv["site"] == "vector"
    assert lv["clock_axis"] == "t"          # t comes from the clock
    assert lv["bound_axes"] == ["s"]        # s is pinned by bind=
    assert lv["source_axes"] == ["s", "t"]  # …of an {s,t} node
    assert any(n["op"] == "Lower" and n["site"] == "scalar" for n in dag["nodes"])


def test_dag_reduce_node_names_the_axis_it_consumes():
    from loom import Reduce, Ax
    from loom.axes import axis_annotation
    r = Reduce(Ax("s") * Ax("t"), "s", 8, "mean")
    rec = axis_annotation(r)
    assert rec["axes"] == ["t"] and rec["reduces"] == "s"
    assert rec["reduce_op"] == "mean" and rec["samples"] == 8


def test_axis_annotation_is_empty_for_a_legacy_signal():
    from loom.axes import axis_annotation, binding_edges
    from loom.signals import Sine
    assert axis_annotation(Sine()) == {}
    assert binding_edges(Sine()) == {}


def test_sidecar_with_axis_annotations_serialises():
    # the real regression: every projected value must be JSON-encodable
    json.dumps(introspect(_axis_scene()))


def test_introspect_datasets_sorted_and_unique():
    d = introspect(build())
    ids = [ds["id"] for ds in d["datasets"]]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_introspect_camera_and_lights():
    d = introspect(build())
    assert d["camera"]["class"] == "Camera"
    assert d["lights"] and d["lights"][0]["kind"] == "point"


# --------------------------------------------------------------------------
# ViewerModel
# --------------------------------------------------------------------------

def test_viewer_model_scene_and_introspect():
    vm = ViewerModel(build, radius=0.2)
    sc = vm.scene(Clock.at_frame(0, 1))
    assert isinstance(sc, Scene)
    d = vm.introspect(Clock.at_frame(0, 1))
    assert d["version"] == SIDECAR_VERSION


def test_viewer_model_declared_params():
    vm = ViewerModel(build)
    assert vm.declared_params() == {"radius": 0.12}


def test_viewer_model_overrides_params():
    vm = ViewerModel(build, radius=0.2)
    sc = vm.scene(None, radius=0.5)
    swept = [e for e in sc.elements if type(e).__name__ == "SweptMesh"][0]
    assert swept.scale == pytest.approx(0.5)


def test_viewer_model_from_file(tmp_path):
    path = _write_scene_file(tmp_path, """
        from loom.scene import Scene, Camera, Sphere, Material
        def build(clock=None):
            sc = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
            sc.add(Material("m", "diffuse"), Sphere((0, 0, 0), 1.0, "m"))
            return sc
    """)
    vm = ViewerModel.from_file(path)
    d = vm.introspect()
    assert any(o["kind"] == "sphere" for o in d["objects"])


def test_viewer_model_save_sidecar_roundtrips(tmp_path):
    vm = ViewerModel(build)
    out = str(tmp_path / "scene.viewer.json")
    vm.save_sidecar(out, Clock.at_frame(1, 4))
    with open(out) as f:
        d = json.load(f)
    assert d["frame"] == {"frame": 1, "frames": 4}
    assert d["version"] == SIDECAR_VERSION
    assert d["objects"]


def test_viewer_model_rejects_non_callable():
    with pytest.raises(TypeError):
        ViewerModel(42)


# --------------------------------------------------------------------------
# §F4 — materials + textures in the sidecar (the texture-display blocker)
# --------------------------------------------------------------------------

def test_introspect_emits_materials_with_texture_binding():
    from loom.scene import skin, func_skin, Sphere
    sc = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
    tex, mat = skin("hide", "textures/cow.png", roughness=0.3)
    ptex, pmat = func_skin("stripes", "u", "v", "0.5+0.5*sin(2*pi*8*u)")
    sc.add(tex, mat, ptex, pmat,
           Sphere((0, 0, 0), 1.0, "hide"), Sphere((2, 0, 0), 1.0, "stripes"))
    d = introspect(sc)
    mats = {m["name"]: m for m in d["materials"]}
    assert mats["hide"]["texture"] == "hide"       # resolved texture: binding
    assert mats["hide"]["props"]["roughness"] == 0.3
    assert mats["stripes"]["texture"] == "stripes"


def test_introspect_emits_image_and_formula_textures():
    from loom.scene import skin, func_skin
    sc = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
    sc.add(*skin("hide", "textures/cow.png"))
    sc.add(*func_skin("stripes", "u", "v", "0.5+0.5*sin(2*pi*8*u)", res=256))
    d = introspect(sc)
    tex = {t["name"]: t for t in d["textures"]}
    assert tex["hide"]["kind"] == "image"
    assert tex["hide"]["file"] == "textures/cow.png"
    assert tex["stripes"]["kind"] == "formula"
    assert tex["stripes"]["r"] == "u" and tex["stripes"]["res"] == 256


def test_formula_texture_channels_are_baked_to_ftsl_strings():
    """A material *bundle* with a spatial-field colour slot lowers to a
    ``ProcTexture`` whose r/g/b channels are live ``SpatialExpr`` objects, not
    strings (``ProcTexture._chan`` keeps anything with ``.emit``).  The sidecar must
    bake them at the clock the way ``ProcTexture.emit`` does — otherwise the whole
    JSON dump dies with "Object of type _Bin is not JSON serializable", and even a
    ``repr`` fallback would hand the viewer something it cannot compile."""
    from loom.scene import Material, Sphere
    from loom.spatial import U, V
    sc = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
    sc.add(Material("bundled", "diffuse", reflect=(0.5 + 0.5 * U, V, 0.25)),
           Sphere((0, 0, 0), 1.0, "bundled"))
    d = introspect(sc, clock=Clock.at_frame(0, 4))
    tex = [t for t in d["textures"] if t["kind"] == "formula"]
    assert tex, "bundle colour field should have produced a ProcTexture"
    for ch in ("r", "g", "b"):
        assert isinstance(tex[0][ch], str), f"channel {ch} not baked to a string"
    assert "u" in tex[0]["r"] and "v" in tex[0]["g"]
    json.dumps(d)          # the real regression: the sidecar must serialise at all


def test_introspect_material_animated_prop_evaluated_at_clock():
    from loom.scene import Material, Sphere
    from loom.signals import Sine
    sc = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
    sc.add(Material("m", "diffuse", roughness=Sine(cycles=1, amp=0.5, bias=0.5)),
           Sphere((0, 0, 0), 1.0, "m"))
    d = introspect(sc, clock=Clock.at_frame(0, 4))
    rough = {m["name"]: m for m in d["materials"]}["m"]["props"]["roughness"]
    assert isinstance(rough, (int, float))   # evaluated, not a Signal repr


def test_session_reintrospection_carries_materials_and_textures():
    ack = _session().handle({"cmd": "introspect"})
    assert "materials" in ack["sidecar"] and "textures" in ack["sidecar"]


# --------------------------------------------------------------------------
# §F4/§F7 — the viewer↔loom live re-introspection channel
# --------------------------------------------------------------------------

def _session():
    return ViewerSession(ViewerModel(build))


def test_session_introspect_inline_returns_sidecar():
    ack = _session().handle({"cmd": "introspect"})
    assert ack["ok"] is True
    assert ack["sidecar"]["version"] == SIDECAR_VERSION
    assert ack["sidecar"]["objects"]


def test_session_introspect_reflects_clock_and_params():
    ack = _session().handle({
        "cmd": "introspect",
        "clock": {"frame": 1, "frames": 4},
        "params": {"radius": 0.3},
    })
    d = ack["sidecar"]
    assert d["frame"] == {"frame": 1, "frames": 4}
    # the SweptMesh is re-derived, carrying its tessellated geometry
    worm = next(o for o in d["objects"] if o.get("name") == "worm")
    assert worm["mesh"]["vertices"]


def test_session_reintrospection_tracks_a_changing_param():
    # different params must yield different geometry — the whole point of the
    # live channel over a frozen sidecar. The SweptMesh's tube radius scales its
    # tessellated vertices, so re-introspecting at a new radius moves them.
    sess = _session()
    a = sess.handle({"cmd": "introspect", "params": {"radius": 0.12}})
    b = sess.handle({"cmd": "introspect", "params": {"radius": 0.40}})
    va = next(o for o in a["sidecar"]["objects"] if o["kind"] == "swept_mesh")["mesh"]["vertices"]
    vb = next(o for o in b["sidecar"]["objects"] if o["kind"] == "swept_mesh")["mesh"]["vertices"]
    assert va != vb


def test_session_introspect_out_writes_sidecar_file(tmp_path):
    out = str(tmp_path / "live.viewer.json")
    ack = _session().handle({"cmd": "introspect", "out": out})
    assert ack == {"ok": True, "out": out}
    with open(out) as f:
        d = json.load(f)
    assert d["version"] == SIDECAR_VERSION


def test_session_params_advertises_declared_controls():
    ack = _session().handle({"cmd": "params"})
    assert ack["ok"] is True
    assert ack["params"] == {"radius": 0.12}


# --------------------------------------------------------------------------
# F7 primary path — the .ftsl source the C++ viewer raymarches
# --------------------------------------------------------------------------

def test_save_sidecar_emits_ftsl_source(tmp_path):
    # save_sidecar drops a sibling .ftsl and records its absolute path so the
    # C++ -viewer can parse it and raymarch the real field (the F7 primary path).
    out = tmp_path / "scene.viewer.json"
    ViewerModel(build).save_sidecar(str(out))
    d = json.loads(out.read_text())
    src = d.get("source")
    assert src and src.endswith(".ftsl")
    import os
    assert os.path.isabs(src) and os.path.exists(src)
    text = open(src).read()
    assert "scene {" in text          # a real ftsl document


def _sphere_build(clock=None, *, radius=0.25):
    # a build whose param lands inline in the .ftsl (a Sphere's radius), unlike the
    # SweptMesh tube which bakes to an external mesh file
    cam = Camera(eye=(0, 0, 5), look_at=(0, 0, 0))
    mat = Material("skin", "diffuse")
    sc = Scene(cam)
    sc.add(mat, Sphere((0, 0, 0), radius, "skin"), Light("point", intensity=1.0))
    return sc


def test_save_sidecar_source_reflects_params(tmp_path):
    out = tmp_path / "scene.viewer.json"
    ViewerModel(_sphere_build, radius=0.42).save_sidecar(str(out))
    src = json.loads(out.read_text())["source"]
    # the sphere radius lands inline in the emitted source
    assert "0.42" in open(src).read()


def test_save_sidecar_can_skip_source(tmp_path):
    out = tmp_path / "scene.viewer.json"
    ViewerModel(build).save_sidecar(str(out), emit_source=False)
    d = json.loads(out.read_text())
    assert "source" not in d
    assert not (tmp_path / "scene.ftsl").exists()


def test_session_emit_inline_returns_ftsl_source():
    ack = _session().handle({"cmd": "emit"})
    assert ack["ok"] is True
    assert "scene {" in ack["source"]


def test_session_emit_out_writes_ftsl_file(tmp_path):
    out = str(tmp_path / "live.ftsl")
    ack = _session().handle({"cmd": "emit", "out": out})
    assert ack == {"ok": True, "out": out}
    assert "scene {" in open(out).read()


def test_session_emit_reflects_params():
    sess = ViewerSession(ViewerModel(_sphere_build))
    a = sess.handle({"cmd": "emit", "params": {"radius": 0.12}})["source"]
    b = sess.handle({"cmd": "emit", "params": {"radius": 0.40}})["source"]
    assert a != b


def test_session_unknown_cmd_and_errors_dont_crash():
    assert _session().handle({"cmd": "nope"})["ok"] is False
    # a bad param surfaces as an error ack, not an exception
    bad = _session().handle({"cmd": "introspect", "clock": {"frames": "abc"}})
    assert bad["ok"] is False and "error" in bad


def test_serve_viewer_stdio_loop_roundtrips():
    import io
    reqs = "\n".join([
        json.dumps({"cmd": "params"}),
        json.dumps({"cmd": "introspect", "clock": {"frame": 0, "frames": 2}}),
        json.dumps({"cmd": "quit"}),
        json.dumps({"cmd": "introspect"}),   # after quit — must be ignored
    ]) + "\n"
    out = io.StringIO()
    serve_viewer(_session(), io.StringIO(reqs), out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert len(lines) == 3                    # loop stops after quit
    assert lines[0]["params"] == {"radius": 0.12}
    assert lines[1]["sidecar"]["objects"]
    assert lines[2] == {"ok": True, "bye": True}


def test_serve_viewer_reports_bad_json_without_stopping():
    import io
    reqs = "not json\n" + json.dumps({"cmd": "quit"}) + "\n"
    out = io.StringIO()
    serve_viewer(_session(), io.StringIO(reqs), out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert lines[0]["ok"] is False and "bad json" in lines[0]["error"]
    assert lines[1]["bye"] is True


def test_viewer_main_smoke(tmp_path, capsys):
    import loom.viewer as vmod
    path = _write_scene_file(tmp_path, """
        from loom.scene import Scene, Camera, Sphere, Material
        def build(clock=None, *, r=1.0):
            sc = Scene(Camera(eye=(0, 0, 5), look_at=(0, 0, 0)))
            sc.add(Material("m", "diffuse"), Sphere((0, 0, 0), r, "m"))
            return sc
    """)
    import io
    orig = sys.stdin
    sys.stdin = io.StringIO(json.dumps({"cmd": "params"}) + "\n"
                            + json.dumps({"cmd": "quit"}) + "\n")
    try:
        rc = vmod.main([path, "r=3.0"])
    finally:
        sys.stdin = orig
    assert rc == 0
    outlines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert outlines[0]["params"] == {"r": 1.0}   # declared default, not the seed
    assert outlines[-1]["bye"] is True
