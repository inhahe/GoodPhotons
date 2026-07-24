"""§F1 — the loom↔viewer data contract: load_build + scene introspection sidecar."""

import json
import textwrap

import pytest

from loom.viewer import (
    SIDECAR_VERSION, load_build, build_scene, introspect, ViewerModel,
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
