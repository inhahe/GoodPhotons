"""``.ftmesh`` — the binary mesh handoff loom writes for ftrace's live viewer.

The reader that matters is `loadFtmesh` in ``src/mesh.h``, which these tests cannot
call.  What they CAN pin down is everything that would make that reader wrong:
the header fields, the section order and sizes, endianness, and the promise that
switching a scene's ``mesh_format`` changes only the file on disk and not the
geometry.  :func:`loom.ftmesh.read_ftmesh` exists for exactly this — a format with
no reader on the writing side is a format nobody can check.
"""

import os
import struct
import tempfile

import pytest

from loom.ftmesh import (HAS_NORMALS, HAS_UVS, MAGIC, VERSION, encode,
                         read_ftmesh, write_ftmesh)

# A tetrahedron: small enough to check by hand, non-degenerate, 4 verts / 4 faces.
VERTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
NORMALS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.5, 0.5)]
UVS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]


def test_header_is_exactly_what_the_c_reader_expects():
    blob = encode(VERTS, FACES)
    assert blob[:8] == MAGIC == b"FTMESH\0\0"
    version, flags, nv, nt = struct.unpack_from("<IIII", blob, 8)
    assert (version, flags, nv, nt) == (VERSION, 0, 4, 4)
    # 24-byte header, then f32 positions, then u32 indices -- nothing else.
    assert len(blob) == 24 + 4 * 3 * 4 + 4 * 3 * 4


def test_sections_are_sized_and_ordered_by_the_flags():
    plain = encode(VERTS, FACES)
    with_n = encode(VERTS, FACES, normals=NORMALS)
    with_uv = encode(VERTS, FACES, uvs=UVS)
    both = encode(VERTS, FACES, normals=NORMALS, uvs=UVS)

    assert struct.unpack_from("<I", plain, 12)[0] == 0
    assert struct.unpack_from("<I", with_n, 12)[0] == HAS_NORMALS
    assert struct.unpack_from("<I", with_uv, 12)[0] == HAS_UVS
    assert struct.unpack_from("<I", both, 12)[0] == HAS_NORMALS | HAS_UVS

    nv = len(VERTS)
    assert len(with_n) == len(plain) + nv * 3 * 4
    assert len(with_uv) == len(plain) + nv * 2 * 4
    assert len(both) == len(plain) + nv * 3 * 4 + nv * 2 * 4
    # Normals come before UVs, so a reader stepping the sections in flag order lands
    # in the right place.
    assert both[24:24 + nv * 12] == plain[24:24 + nv * 12]
    assert both[24 + nv * 12:24 + nv * 24] == with_n[24 + nv * 12:24 + nv * 24]


def test_is_little_endian_regardless_of_host():
    # The first position component is 0.0 and the second is 1.0; 1.0f little-endian
    # is 00 00 80 3F. Pinning a literal here is what catches a host-order regression
    # on a big-endian machine, where `array.tobytes()` would silently flip.
    blob = encode(VERTS, FACES)
    assert blob[24 + 12:24 + 16] == b"\x00\x00\x80\x3f"
    # ...and the index section is u32 little-endian the same way.
    assert struct.unpack_from("<III", blob, 24 + 4 * 3 * 4)[:3] == (0, 2, 1)


def test_round_trip_preserves_topology_and_f32_geometry():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "tet.ftmesh")
        write_ftmesh(p, VERTS, FACES, normals=NORMALS, uvs=UVS)
        v, f, n, u = read_ftmesh(p)
    assert f == FACES                       # indices are exact
    assert v == VERTS                       # these all happen to be f32-exact
    assert n == NORMALS
    assert u == [pytest.approx(t) for t in UVS]


def test_f32_storage_is_finer_than_the_obj_text_it_replaces():
    # write_obj formats "%.6g" -- 6 significant decimal digits. f32 carries ~7.2, so
    # a value that survives f32 must not be claimed to survive %.6g.
    val = 1.2345678
    v, _, _, _ = _rt([(val, 0.0, 0.0)], [(0, 0, 0)])
    f32_err = abs(v[0][0] - val)
    obj_err = abs(float("%.6g" % val) - val)
    assert f32_err < obj_err


def _rt(verts, faces):
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "m.ftmesh")
        write_ftmesh(p, verts, faces)
        return read_ftmesh(p)


def test_rejects_a_file_that_is_not_a_ftmesh():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "not.ftmesh")
        with open(p, "wb") as fh:
            fh.write(b"v 1 2 3\nf 1 1 1\n")
        with pytest.raises(ValueError, match="bad magic"):
            read_ftmesh(p)


def test_rejects_a_truncated_file():
    # The whole point of sizing the body from the header: a reader racing a
    # non-atomic writer must get an error, not geometry made of whatever came next.
    blob = encode(VERTS, FACES)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "half.ftmesh")
        with open(p, "wb") as fh:
            fh.write(blob[:-8])
        with pytest.raises(ValueError, match="truncated"):
            read_ftmesh(p)


def test_rejects_mismatched_attribute_counts():
    with pytest.raises(ValueError, match="normals"):
        encode(VERTS, FACES, normals=NORMALS[:2])
    with pytest.raises(ValueError, match="uvs"):
        encode(VERTS, FACES, uvs=UVS[:1])


def test_write_is_atomic():
    # Same guarantee write_obj has, and for the same reader: the live channel
    # re-emits while ftrace is still loading the previous frame out of that dir.
    # A temp file in the same directory + os.replace leaves no litter behind.
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "m.ftmesh")
        write_ftmesh(p, VERTS, FACES)
        write_ftmesh(p, VERTS, FACES)
        assert os.listdir(tmp) == ["m.ftmesh"]


# --- the scene-level switch --------------------------------------------------

def _swept_scene():
    import math

    import loom as L
    anchors = [L.vec(math.cos(2 * math.pi * i / 6), 0.3 * math.sin(i),
                     math.sin(2 * math.pi * i / 6)) for i in range(6)]
    spine = L.LoopCurve(L.PointPath(anchors, closed=True), L.Const(0.0))
    s = L.Scene(L.Camera(eye=(0, 1, 3), look_at=(0, 0, 0), res=(32, 32)))
    s.add(L.Material("m", "diffuse", reflect=0.7))
    s.add(L.tube(spine, radius=0.12, sides=10, count=40, material="m", name="tube"))
    return s


def test_mesh_format_switches_the_file_but_not_the_scene():
    import loom as L
    with tempfile.TemporaryDirectory() as tmp:
        s = _swept_scene()
        clock = L.Clock.at_frame(3, 24)
        obj_txt = s.emit(clock, L.Cache(), assets_dir=tmp, tag="o", mesh_format="obj")
        ftm_txt = s.emit(clock, L.Cache(), assets_dir=tmp, tag="b", mesh_format="ftmesh")
        names = sorted(os.listdir(tmp))
        assert "tubeo.obj" in names and "tubeb.ftmesh" in names
        # The emitted ftsl differs only in the referenced filename: same block, same
        # material. ftrace dispatches on the extension.
        assert obj_txt.replace("tubeo.obj", "X") == ftm_txt.replace("tubeb.ftmesh", "X")
        # A default sweep ships ANALYTIC normals, so there is deliberately no `smooth`
        # clause to crease-guess with: authored normals win over it in the loader, and
        # omitting it also skips the loader's position weld + adjacency build.
        assert 'smooth' not in ftm_txt

        # ...and the geometry inside is the same mesh, normals included.
        v, f, n, u = read_ftmesh(os.path.join(tmp, "tubeb.ftmesh"))
        with open(os.path.join(tmp, "tubeo.obj"), encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        ov = [tuple(float(x) for x in ln.split()[1:4])
              for ln in lines if ln.startswith("v ")]
        on = [tuple(float(x) for x in ln.split()[1:4])
              for ln in lines if ln.startswith("vn ")]
        # OBJ faces are `v//vn` triples when normals are present
        of = [tuple(int(x.split("//")[0]) - 1 for x in ln.split()[1:4])
              for ln in lines if ln.startswith("f ")]
        assert f == of
        assert len(v) == len(ov)
        assert n is not None and len(n) == len(v) == len(on)
        # OBJ text went through %.6g, so agreement is only to ~1e-6 relative -- which
        # is the point: the binary side is the more faithful of the two.
        for a, b in zip(v, ov):
            for ca, cb in zip(a, b):
                assert ca == pytest.approx(cb, abs=1e-5)
        for a, b in zip(n, on):
            for ca, cb in zip(a, b):
                assert ca == pytest.approx(cb, abs=1e-5)


def test_mesh_sink_diverts_the_bytes_and_leaves_the_scene_text_alone():
    """A mesh_sink must change *where* the bytes go and nothing else — the ftsl still
    names the same path, so a consumer keyed by that path (ftrace's asset overlay)
    doesn't have to know which mode produced it."""
    import loom as L
    with tempfile.TemporaryDirectory() as tmp:
        clock = L.Clock.at_frame(3, 24)
        onfile = _swept_scene().emit(clock, L.Cache(), assets_dir=tmp, tag="x",
                                     mesh_format="ftmesh")
        sink = {}
        piped = _swept_scene().emit(clock, L.Cache(), assets_dir=tmp, tag="x",
                                    mesh_format="ftmesh", mesh_sink=sink)
        assert onfile == piped
        (name, data), = sink.items()
        assert f'file "{name}"' in piped
        with open(os.path.join(tmp, "tubex.ftmesh"), "rb") as fh:
            assert data == fh.read()


def test_mesh_sink_does_not_create_the_assets_dir():
    """Skipping the write is not enough: creating the directory would put us back to
    touching the filesystem once per frame, which is the cost this avoids."""
    import loom as L
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "nope")
        sink = {}
        _swept_scene().emit(L.Clock.at_frame(0, 24), L.Cache(), assets_dir=d,
                            mesh_format="ftmesh", mesh_sink=sink)
        assert sink and not os.path.exists(d)


def test_unknown_mesh_format_is_an_error_not_a_silent_obj():
    import loom as L
    with tempfile.TemporaryDirectory() as tmp:
        s = _swept_scene()
        with pytest.raises(ValueError, match="mesh_format"):
            s.emit(L.Clock.at_frame(0, 24), L.Cache(), assets_dir=tmp,
                   mesh_format="stl")


def test_live_channel_defaults_to_binary():
    # The whole reason the format exists. ViewerSession._emit is what the C++
    # LoomBridge calls every frame; if it ever silently reverts to OBJ the viewer
    # just gets slower, with nothing failing to say so.
    import inspect

    from loom.viewer import ViewerSession
    src = inspect.getsource(ViewerSession._emit)
    assert '"ftmesh"' in src
    assert "mesh_format=fmt" in src
