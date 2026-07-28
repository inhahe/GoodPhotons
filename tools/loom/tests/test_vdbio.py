"""E4 tests: loom.vdbio — bake a field to a dense grid and write/read a ``.vdb``.

Runnable directly (``python tests/test_vdbio.py``) or under pytest.  These cover
the OpenVDB ``float 5_4_3`` / ACTIVE_MASK / full-float subset loom writes and
ftrace ingests; the round-trip is asserted with loom's own :func:`read_vdb`
(which parses back exactly that subset).
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

np = pytest.importorskip("numpy")

from loom import vdbio  # noqa: E402
from loom.spatial import X, Y, Z, exp as sexp  # noqa: E402


def _blob(nx=32, ny=32, nz=32, box=(-1, -1, -1, 1, 1, 1), thresh=0.05):
    x0, y0, z0, x1, y1, z1 = box
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    zs = np.linspace(z0, z1, nz)
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = np.exp(-6.0 * (Xg**2 + Yg**2 + Zg**2)).astype("<f4")
    vol[vol < thresh] = 0.0
    return vol


def _positive_subbox(vol):
    ii, jj, kk = np.nonzero(vol > 0)
    lo = (ii.min(), jj.min(), kk.min())
    hi = (ii.max(), jj.max(), kk.max())
    return vol[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1], lo, hi


def test_write_read_roundtrip_is_bit_exact():
    vol = _blob()
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "blob.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", vol, box)])
        back = vdbio.read_vdb(path)
    assert set(back) == {"density"}
    arr, box6 = back["density"]
    sub, lo, hi = _positive_subbox(vol)
    # ftrace's reader (and ours) drop v<=0, so only the positive sub-box survives.
    assert arr.shape == sub.shape
    assert float(np.abs(arr - sub).max()) == 0.0   # exact: full float32, no lossy step


def test_world_box_matches_linspace_positions():
    nx = 40
    vol = _blob(nx, nx, nx)
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "blob.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", vol, box)])
        arr, box6 = vdbio.read_vdb(path)["density"]
    _, lo, hi = _positive_subbox(vol)
    s = 2.0 / (nx - 1)                              # linspace step, corners incl.
    exp = (-1 + lo[0] * s, -1 + lo[1] * s, -1 + lo[2] * s,
           -1 + hi[0] * s, -1 + hi[1] * s, -1 + hi[2] * s)
    for a, b in zip(box6, exp):
        assert abs(a - b) < 1e-6


def test_multi_grid_named_selection():
    v1 = _blob(24, 24, 24)
    v2 = _blob(24, 24, 24) * 2.0
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "multi.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", v1, box),
                               vdbio.VolumeGrid("temperature", v2, box)])
        back = vdbio.read_vdb(path)
    assert set(back) == {"density", "temperature"}
    a1, _ = back["density"]
    a2, _ = back["temperature"]
    # temperature is 2x density everywhere both are positive
    m = (a1 > 0) & (a2 > 0)
    assert np.allclose(a2[m], 2.0 * a1[m], rtol=1e-5)


def test_bake_field_and_write_volume():
    dens = sexp(-4.0 * (X * X + Y * Y + Z * Z))
    temp = sexp(-8.0 * (X * X + Y * Y + Z * Z)) * 2.0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fire.vdb")
        vdbio.write_volume(path, box=1.5, res=40, density=dens, temperature=temp)
        back = vdbio.read_vdb(path)
    assert set(back) == {"density", "temperature"}
    # both grids sample the same gaussian shape; temperature peak ~2x density peak
    dpk = float(back["density"][0].max())
    tpk = float(back["temperature"][0].max())
    assert 1.8 < tpk / dpk < 2.2


def test_sparse_empty_leaves_are_dropped():
    # A single positive voxel → exactly one active leaf; file stays small and the
    # round-trip recovers that one voxel.
    vol = np.zeros((16, 16, 16), dtype="<f4")
    vol[8, 8, 8] = 3.0
    box = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "one.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", vol, box)])
        arr, _ = vdbio.read_vdb(path)["density"]
    assert arr.shape == (1, 1, 1)
    assert abs(float(arr[0, 0, 0]) - 3.0) < 1e-6


def test_duplicate_names_rejected():
    v = _blob(16, 16, 16)
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dup.vdb")
        with pytest.raises(ValueError):
            vdbio.write_vdb(path, [vdbio.VolumeGrid("density", v, box),
                                   vdbio.VolumeGrid("density", v, box)])


# ---- codec variants (E4 read side: half / ZIP) ----------------------------

def _rt(vol, box, **kw):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", vol, box)], **kw)
        size = os.path.getsize(path)
        arr, box6 = vdbio.read_vdb(path)["density"]
    return arr, box6, size


def test_default_output_unchanged_by_new_codec_params():
    # The new half/zip params default off → byte-for-byte the original file.
    vol = _blob()
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "a.vdb")
        p1 = os.path.join(d, "b.vdb")
        vdbio.write_vdb(p0, [vdbio.VolumeGrid("density", vol, box)])
        vdbio.write_vdb(p1, [vdbio.VolumeGrid("density", vol, box)],
                        half=False, zip=False)
        assert open(p0, "rb").read() == open(p1, "rb").read()


def test_zip_roundtrip_is_bit_exact_and_smaller():
    vol = _blob()
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    sub, _, _ = _positive_subbox(vol)
    plain, _, size_plain = _rt(vol, box)
    arr, _, size_zip = _rt(vol, box, zip=True)
    # ZIP is lossless: identical values to the uncompressed read…
    assert float(np.abs(arr - sub).max()) == 0.0
    assert float(np.abs(arr - plain).max()) == 0.0
    # …and the smooth blob compresses (this field is very zippable).
    assert size_zip < size_plain


def test_half_roundtrip_is_close_and_smaller():
    vol = _blob()
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    sub, _, _ = _positive_subbox(vol)
    _, _, size_plain = _rt(vol, box)
    arr, _, size_half = _rt(vol, box, half=True)
    assert arr.shape == sub.shape
    # half-float: ~3 significant digits, values in [0,1] here → abs err ~<1e-3.
    assert float(np.abs(arr - sub).max()) < 2e-3
    assert size_half < size_plain            # 16-bit voxels → smaller file


def test_half_and_zip_together():
    vol = _blob()
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    sub, _, _ = _positive_subbox(vol)
    arr, _, _ = _rt(vol, box, half=True, zip=True)
    assert arr.shape == sub.shape
    assert float(np.abs(arr - sub).max()) < 2e-3


def test_half_grid_type_carries_suffix():
    vol = _blob(16, 16, 16)
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "h.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", vol, box)], half=True)
        raw = open(path, "rb").read()
    # ftrace flags half by the grid-type suffix, not metadata.
    assert b"Tree_float_5_4_3_HalfFloat" in raw


def test_write_volume_threads_codec_flags():
    dens = sexp(-4.0 * (X * X + Y * Y + Z * Z))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fire.vdb")
        vdbio.write_volume(path, box=1.5, res=32, half=True, zip=True, density=dens)
        arr, _ = vdbio.read_vdb(path)["density"]
    assert arr.max() > 0.5           # gaussian peak survived the round-trip


def test_zip_and_blosc_mutually_exclusive():
    v = _blob(16, 16, 16)
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            vdbio.write_vdb(os.path.join(d, "x.vdb"),
                            [vdbio.VolumeGrid("density", v, box)],
                            zip=True, blosc=True)


# ---- blosc codec (the DCC-standard codec; needs the `blosc` package) --------
blosc = pytest.importorskip("blosc")


def test_blosc_roundtrip_is_bit_exact_and_smaller():
    vol = _blob()
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    sub, _, _ = _positive_subbox(vol)
    plain, _, size_plain = _rt(vol, box)
    arr, _, size_bl = _rt(vol, box, blosc=True)
    assert float(np.abs(arr - sub).max()) == 0.0      # LZ4 is lossless
    assert float(np.abs(arr - plain).max()) == 0.0
    assert size_bl < size_plain


def test_blosc_half_together_is_close():
    vol = _blob()
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    sub, _, _ = _positive_subbox(vol)
    arr, _, _ = _rt(vol, box, blosc=True, half=True)
    assert arr.shape == sub.shape
    assert float(np.abs(arr - sub).max()) < 2e-3


# ---- real third-party sample files (validate against genuine DCC output) ---
# These live in scraps/ (git-ignored) on the dev machine; skip where absent.
# tests/ → loom/ → tools/ → repo-root, then scraps/
_SCRAPS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "scraps")


@pytest.mark.parametrize("fname,grids", [
    ("_smoke.vdb", {"density"}),          # blosc-compressed (Houdini smoke)
    ("_fire.vdb", {"density", "temperature"}),  # multi-grid, unique-name suffix
    ("_sphere.vdb", {"ls_sphere"}),       # UniformScaleMap level set
    ("_cube.vdb", {"ls_cube"}),           # UniformScaleMap level set
])
def test_reads_real_sample_vdb(fname, grids):
    path = os.path.join(_SCRAPS, fname)
    if not os.path.exists(path):
        pytest.skip(f"sample {fname} not present")
    back = vdbio.read_vdb(path)
    assert set(back) == grids
    for name, (arr, box6) in back.items():
        assert arr.ndim == 3 and min(arr.shape) > 0
        assert float(arr.max()) > 0.0
        assert box6[3] > box6[0] and box6[4] > box6[1] and box6[5] > box6[2]


# ---- rotated / general AffineMap transforms (E4) --------------------------
def _rot_z(deg):
    """Index->world map: rotate `deg` about Z, scale 0.1, offset to (1,2,3)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    k = 0.1
    return vdbio.VdbTransform((k * c, -k * s, 0.0,
                               k * s,  k * c, 0.0,
                               0.0,    0.0,   k), (1.0, 2.0, 3.0))


def test_transform_diagonal_detection():
    diag = vdbio.VdbTransform.diagonal((2.0, 3.0, 4.0), (1.0, 0.0, -1.0))
    assert diag.is_diagonal
    assert diag.scale == (2.0, 3.0, 4.0)
    assert diag.apply(1, 1, 1) == (3.0, 3.0, 3.0)
    assert not _rot_z(30).is_diagonal
    # A 90-degree rotation is *not* diagonal even though cos(90)~0 -- the
    # off-diagonals carry the whole map.
    assert not _rot_z(90).is_diagonal


def test_diagonal_tolerance_ignores_float_crumbs():
    # A DCC composing a 180-degree rotation in floating point leaves ~1e-17 in
    # the off-diagonals; that must still read as axis-aligned, not rotated.
    eps = 1e-17
    t = vdbio.VdbTransform((0.5, eps, 0.0, eps, 0.5, 0.0, 0.0, 0.0, 0.5),
                           (0.0, 0.0, 0.0))
    assert t.is_diagonal


def test_voxel_size_is_rotation_invariant():
    # Column norms are the true index-step lengths; the bare diagonal is not.
    for deg in (0.0, 30.0, 90.0):
        vs = _rot_z(deg).voxel_size
        for v in vs:
            assert abs(v - 0.1) < 1e-12


def test_rotated_grid_roundtrips_through_affinemap():
    vol = _blob(16, 16, 16)
    xform = _rot_z(30)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rot.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", vol, transform=xform)])
        back = vdbio.read_vdb_grids(path)
    g = back["density"]
    sub, lo, _hi = _positive_subbox(vol)
    assert g.values.shape == sub.shape
    assert float(np.abs(g.values - sub).max()) == 0.0    # samples untouched by rotation
    assert tuple(g.index_lo) == tuple(int(v) for v in lo)
    assert not g.transform.is_diagonal
    for a, b in zip(g.transform.a, xform.a):
        assert abs(a - b) < 1e-12
    for a, b in zip(g.transform.t, xform.t):
        assert abs(a - b) < 1e-12
    # The decisive check: array offset (0,0,0) lands where the original map
    # sends its index, rotation included.
    for w, e in zip(g.world_of(0, 0, 0), xform.apply(*lo)):
        assert abs(w - e) < 1e-9


def test_rotated_grid_has_no_axis_aligned_box():
    vol = _blob(16, 16, 16)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rot.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", vol, transform=_rot_z(30))])
        # read_vdb must refuse rather than hand back a box that misplaces voxels
        with pytest.raises(ValueError, match="rotated"):
            vdbio.read_vdb(path)
        g = vdbio.read_vdb_grids(path)["density"]      # the full-fidelity read works
    assert g.values.ndim == 3


def test_unrotated_affinemap_still_yields_a_box():
    # An AffineMap that happens to be diagonal is a legal, common file; it must
    # come back through the plain read_vdb path with the right box.
    vol = _blob(16, 16, 16)
    xform = vdbio.VdbTransform.diagonal((0.25, 0.25, 0.25), (-1.0, -2.0, -3.0))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "aff.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", vol, transform=xform)])
        arr, box6 = vdbio.read_vdb(path)["density"]
    _, lo, hi = _positive_subbox(vol)
    exp = xform.apply(*lo) + xform.apply(*hi)
    for a, b in zip(box6, exp):
        assert abs(a - b) < 1e-9


def test_read_vdb_grids_matches_read_vdb_on_diagonal_files():
    # The two entry points must not drift: read_vdb is defined as read_vdb_grids
    # plus `.box`, so every diagonal file has to agree exactly.
    vol = _blob(20, 20, 20)
    box = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "blob.vdb")
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", vol, box)])
        pairs = vdbio.read_vdb(path)
        grids = vdbio.read_vdb_grids(path)
    assert set(pairs) == set(grids)
    arr, box6 = pairs["density"]
    g = grids["density"]
    assert float(np.abs(arr - g.values).max()) == 0.0
    assert box6 == g.box
    assert g.transform.is_diagonal


def test_volumegrid_requires_exactly_one_placement():
    vol = _blob(8, 8, 8)
    with pytest.raises(ValueError, match="exactly one"):
        vdbio.VolumeGrid("d", vol)
    with pytest.raises(ValueError, match="exactly one"):
        vdbio.VolumeGrid("d", vol, (0, 0, 0, 1, 1, 1),
                         transform=vdbio.VdbTransform.diagonal((1, 1, 1), (0, 0, 0)))


# ---- NanoVDB (.nvdb) reader -----------------------------------------------
# `scraps/cloud.nvdb` is a genuine third-party NanoVDB file (a 41^3 fog volume
# with a nonzero background and 10 constant tiles), so these assert the tree
# walk against metadata NanoVDB wrote *independently* of the tree: the file
# header's index bbox / voxel size / grid name, and the root node's own stored
# min/max statistics.  A wrong offset or bit order cannot survive that.
_NVDB_SAMPLE = os.path.join(_SCRAPS, "cloud.nvdb")


def _nvdb_file_meta(path):
    """The `.nvdb` FileMetaData fields, read straight out of the header."""
    import struct
    with open(path, "rb") as f:
        buf = f.read()
    grid_size, _file_size, _key, voxels = struct.unpack_from("<QQQQ", buf, 16)
    world = struct.unpack_from("<6d", buf, 16 + 40)
    index = struct.unpack_from("<6i", buf, 16 + 88)
    voxel_size = struct.unpack_from("<3d", buf, 16 + 112)
    name_size = struct.unpack_from("<I", buf, 16 + 136)[0]
    name = buf[16 + 176:16 + 176 + name_size].split(b"\0", 1)[0].decode()
    grid_at = 16 + 176 + name_size
    return dict(name=name, index=index, world=world, voxel_size=voxel_size,
                voxels=voxels, grid=buf[grid_at:grid_at + grid_size])


def test_nvdb_matches_its_own_header_metadata():
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    meta = _nvdb_file_meta(_NVDB_SAMPLE)
    grids = vdbio.read_nvdb(_NVDB_SAMPLE)
    assert set(grids) == {meta["name"]}
    g = grids[meta["name"]]
    # The dense bake must span exactly the header's active index bbox.
    assert tuple(g.index_lo) == tuple(meta["index"][:3])
    assert tuple(g.index_hi) == tuple(meta["index"][3:])
    # ...and the transform must reproduce the header's voxel size.
    for got, want in zip(g.transform.voxel_size, meta["voxel_size"]):
        assert abs(got - want) < 1e-12
    # The header's world bbox is voxel-inclusive (one voxel wider at the top
    # than the sample corners `box` reports).
    for got, want in zip(g.box[:3], meta["world"][:3]):
        assert abs(got - want) < 1e-9
    for got, want, vs in zip(g.box[3:], meta["world"][3:], meta["voxel_size"]):
        assert abs(got - (want - vs)) < 1e-9


def test_nvdb_matches_root_node_statistics():
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    import struct
    meta = _nvdb_file_meta(_NVDB_SAMPLE)
    # RootData sits at grid+672 (TreeData) + mNodeOffset[3]; its min/max are
    # computed by NanoVDB over the same voxels our walk visits.
    grid = meta["grid"]
    root = 672 + struct.unpack_from("<q", grid, 672 + 24)[0]
    background, vmin, vmax = struct.unpack_from("<3f", grid, root + 28)
    g = vdbio.read_nvdb(_NVDB_SAMPLE)[meta["name"]]
    assert abs(g.background - background) < 1e-9
    assert abs(float(g.values.min()) - vmin) < 1e-6
    assert abs(float(g.values.max()) - vmax) < 1e-6
    # A background-only bake would be flat; the tiles + leaves must have landed.
    assert float(g.values.max()) > background
    assert int((g.values != background).sum()) > meta["voxels"] // 2


def test_nvdb_every_leaf_voxel_lands_where_nanovdb_put_it():
    """Pin the leaf layout against a route that doesn't use the tree walk.

    NanoVDB writes leaves breadth-first and contiguously, so ``TreeData``'s
    first-leaf offset + leaf count reach every leaf directly — independently of
    the child-offset descent :func:`read_nvdb` uses.  Comparing the two catches
    a wrong value/mask offset, which the header-level assertions cannot.
    """
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    import struct
    grid = _nvdb_file_meta(_NVDB_SAMPLE)["grid"]
    tree = 672
    first_leaf, = struct.unpack_from("<q", grid, tree)
    nleaf, = struct.unpack_from("<I", grid, tree + 32)
    assert nleaf > 0
    g = vdbio.read_nvdb(_NVDB_SAMPLE)["cloud"]
    lo = g.index_lo

    LEAF_STRIDE = 2144
    checked = active_total = 0
    for li in range(nleaf):
        leaf = tree + first_leaf + li * LEAF_STRIDE
        bbmin = struct.unpack_from("<3i", grid, leaf)
        origin = tuple(v & ~7 for v in bbmin)             # LeafNode::origin()
        mask = np.unpackbits(np.frombuffer(grid, np.uint8, 64, leaf + 16),
                             bitorder="little").astype(bool)
        vals = np.frombuffer(grid, "<f4", 512, leaf + 96)
        lmin, lmax = struct.unpack_from("<2f", grid, leaf + 80)
        act = vals[mask]
        active_total += int(mask.sum())
        if act.size:
            # NanoVDB's own per-leaf statistics, over the same active voxels.
            assert abs(float(act.min()) - lmin) < 1e-6
            assert abs(float(act.max()) - lmax) < 1e-6
        for n in np.nonzero(mask)[0]:
            i = origin[0] + ((int(n) >> 6) & 7) - lo[0]
            j = origin[1] + ((int(n) >> 3) & 7) - lo[1]
            k = origin[2] + (int(n) & 7) - lo[2]
            assert abs(float(g.values[i, j, k]) - float(vals[n])) < 1e-6
            checked += 1
    assert checked > 10000                                 # not a vacuous pass
    # Leaves plus the 10 constant tiles must account for every active voxel.
    tiles = struct.unpack_from("<3I", grid, tree + 44)
    voxels, = struct.unpack_from("<Q", grid, tree + 56)
    assert active_total + tiles[0] * 8 ** 3 + tiles[1] * 128 ** 3 == voxels


def test_nvdb_root_tile_stride_matches_the_node_layout():
    """The root's tile table must end exactly where the first upper node begins.

    ``sizeof(RootData::Tile)`` is 32 (NanoVDB static-asserts every node type is
    a multiple of ``NANOVDB_DATA_ALIGNMENT``); this checks the constant against
    the file rather than trusting the header, which matters because a sample
    with a single root tile would otherwise never exercise the stride.
    """
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    import struct
    grid = _nvdb_file_meta(_NVDB_SAMPLE)["grid"]
    tree = 672
    _leaf, _lower, upper_off, root_off = struct.unpack_from("<4q", grid, tree)
    table_size, = struct.unpack_from("<I", grid, tree + root_off + 24)
    assert table_size >= 1
    root_bytes = vdbio._RD_SIZE + table_size * vdbio._RD_TILE_SIZE
    assert root_off + root_bytes == upper_off


def test_nvdb_reads_a_raw_grid_buffer():
    """The container-less layout ftrace also accepts (`writeUncompressedGrids`)."""
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    meta = _nvdb_file_meta(_NVDB_SAMPLE)
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "raw.nvdb")
        with open(raw, "wb") as f:
            f.write(meta["grid"])            # the grid buffer, no FileHeader
        got = vdbio.read_nvdb(raw)
    want = vdbio.read_nvdb(_NVDB_SAMPLE)
    assert set(got) == set(want)
    assert float(np.abs(got[meta["name"]].values
                        - want[meta["name"]].values).max()) == 0.0


def test_read_vdb_grids_dispatches_on_magic():
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    direct = vdbio.read_nvdb(_NVDB_SAMPLE)
    viaany = vdbio.read_vdb_grids(_NVDB_SAMPLE)
    assert set(direct) == set(viaany)
    for name in direct:
        assert float(np.abs(direct[name].values - viaany[name].values).max()) == 0.0


def test_nvdb_rejects_non_nanovdb():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "junk.nvdb")
        with open(path, "wb") as f:
            f.write(b"not a volume at all, really no" + b"\0" * 64)
        with pytest.raises(ValueError, match="not a NanoVDB"):
            vdbio.read_nvdb(path)


def test_nvdb_rejects_compressed_container():
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    with open(_NVDB_SAMPLE, "rb") as f:
        buf = bytearray(f.read())
    buf[14:16] = (1).to_bytes(2, "little")   # FileHeader.codec = ZIP
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "zip.nvdb")
        with open(path, "wb") as f:
            f.write(bytes(buf))
        with pytest.raises(ValueError, match="compressed .nvdb"):
            vdbio.read_nvdb(path)


def test_nvdb_rejects_non_float_grid():
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    meta = _nvdb_file_meta(_NVDB_SAMPLE)
    grid = bytearray(meta["grid"])
    grid[636:640] = (6).to_bytes(4, "little")    # GridType::Vec3f
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vec3.nvdb")
        with open(path, "wb") as f:
            f.write(bytes(grid))
        with pytest.raises(ValueError, match="only float grids"):
            vdbio.read_nvdb(path)


# ---------------------------------------------------------------------------
# VolumeField — an imported volume as a term in the spatial algebra (E4
# read -> transform -> write), plus the trilinear sampler it rides on.
# ---------------------------------------------------------------------------

def _tiny_grid(nx=5, ny=4, nz=3, scale=(0.5, 0.25, 2.0), offset=(-1.0, 3.0, 0.5)):
    """A small grid with distinct per-voxel values and a non-unit transform, so
    an index/axis mix-up cannot hide behind symmetry."""
    vals = np.arange(nx * ny * nz, dtype="<f4").reshape(nx, ny, nz) + 1.0
    xf = vdbio.VdbTransform.diagonal(scale, offset)
    return vdbio.ReadGrid("density", vals, (2, -3, 7), xf, background=0.0)


def test_transform_inverse_round_trips():
    for xf in (vdbio.VdbTransform.diagonal((0.5, 0.25, 2.0), (-1.0, 3.0, 0.5)),
               vdbio.VdbTransform((0.3, 0.1, -0.2, 0.05, 0.4, 0.11,
                                   -0.07, 0.2, 0.33), (1.0, -2.0, 0.25))):
        for ijk in ((0, 0, 0), (3.5, -2.25, 8.0), (-11.0, 4.0, 0.5)):
            w = xf.apply(*ijk)
            back = xf.to_index(*w)
            assert all(abs(a - b) < 1e-9 for a, b in zip(ijk, back))
    with pytest.raises(ValueError, match="singular"):
        vdbio.VdbTransform((1, 0, 0, 2, 0, 0, 3, 0, 0), (0, 0, 0)).inverse_linear


def test_premultiplied_moves_the_lattice_and_composes():
    xf = vdbio.VdbTransform.diagonal((0.5, 0.25, 2.0), (-1.0, 3.0, 0.5))
    # A pure translation shifts every sample by exactly d.
    d = (0.7, -0.2, 4.0)
    moved = xf.premultiplied(None, d)
    for ijk in ((0, 0, 0), (3, -2, 8)):
        a, b = xf.apply(*ijk), moved.apply(*ijk)
        assert all(abs((y - x) - dd) < 1e-12 for x, y, dd in zip(a, b, d))
    # Composition is left-multiplication: applying M then M == applying M*M.
    m = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)      # +90 deg about z
    twice = xf.premultiplied(m).premultiplied(m)
    for ijk in ((1, 2, 3), (-4, 0, 5)):
        x, y, z = xf.apply(*ijk)
        assert all(abs(a - b) < 1e-12
                   for a, b in zip(twice.apply(*ijk), (-x, -y, z)))


def test_sampler_reproduces_the_lattice_exactly():
    """Sampling at the voxel centres must return the stored voxels bit-for-bit —
    the property that makes an identity resample a no-op."""
    g = _tiny_grid()
    nx, ny, nz = g.shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    lo = g.index_lo
    a, t = g.transform.a, g.transform.t
    wx = a[0] * (ii + lo[0]) + t[0]
    wy = a[4] * (jj + lo[1]) + t[1]
    wz = a[8] * (kk + lo[2]) + t[2]
    assert np.allclose(g.sample(wx, wy, wz), g.values, rtol=0, atol=1e-12)


def test_sampler_interpolates_linearly_between_voxels():
    g = _tiny_grid()
    lo, a, t = g.index_lo, g.transform.a, g.transform.t
    # Halfway along x between (0,0,0) and (1,0,0).
    wx = a[0] * (lo[0] + 0.5) + t[0]
    wy = a[4] * lo[1] + t[1]
    wz = a[8] * lo[2] + t[2]
    want = 0.5 * (float(g.values[0, 0, 0]) + float(g.values[1, 0, 0]))
    assert abs(float(g.sample(wx, wy, wz)) - want) < 1e-9


def test_sampler_edge_shell_reads_the_edge_voxel_not_the_second():
    """Regression: ftrace's sampler used to clamp the stencil *indices* but not
    the coordinate, so a point just below index 0 kept a fraction near 1 and was
    dominated by the SECOND voxel — growing wronger the further out it went.
    Both ftrace (src/vdbgrid.h, render_cuda.cu) and this port now clamp the
    coordinate, so the outer half-voxel shell reads the edge voxel."""
    g = _tiny_grid()
    lo, a, t = g.index_lo, g.transform.a, g.transform.t
    x0 = a[0] * lo[0] + t[0]
    wy = a[4] * lo[1] + t[1]
    wz = a[8] * lo[2] + t[2]
    edge = float(g.values[0, 0, 0])
    assert float(g.values[1, 0, 0]) != edge      # else the test would be vacuous
    for frac in (0.0, -0.1, -0.25, -0.49):
        got = float(g.sample(x0 + frac * a[0], wy, wz))
        assert abs(got - edge) < 1e-9, f"at {frac} voxels below index 0: {got}"
    # Just past the half-voxel margin the grid stops existing.
    assert float(g.sample(x0 - 0.6 * a[0], wy, wz)) == 0.0
    # The upper shell was always right; check it stayed right.
    xn = a[0] * (lo[0] + g.shape[0] - 1) + t[0]
    top = float(g.values[-1, 0, 0])
    for frac in (0.0, 0.25, 0.49):
        assert abs(float(g.sample(xn + frac * a[0], wy, wz)) - top) < 1e-9


def test_sampler_outside_and_clamp_knobs():
    g = _tiny_grid()
    g.background = 0.75
    far = (1e6, 1e6, 1e6)
    assert float(g.sample(*far)) == pytest.approx(0.75)      # defaults to background
    assert float(g.sample(*far, outside=0.0)) == 0.0         # ftrace's convention
    neg = vdbio.ReadGrid("d", np.full((2, 2, 2), -3.0, dtype="<f4"), (0, 0, 0),
                         vdbio.VdbTransform.diagonal((1, 1, 1), (0, 0, 0)))
    assert float(neg.sample(0.5, 0.5, 0.5)) == pytest.approx(-3.0)
    assert float(neg.sample(0.5, 0.5, 0.5, clamp_negative=True)) == 0.0


def test_volume_field_identity_resample_is_a_no_op():
    """The whole read -> transform -> write path in miniature: bake an imported
    volume back onto its own lattice and get the original array."""
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    from loom.spatial import VolumeField
    v = VolumeField(_NVDB_SAMPLE)
    g = v.read_grid
    vals, box = vdbio.bake_field(v, v.box, g.shape)
    assert np.abs(vals - np.asarray(g.values, dtype=np.float64)).max() < 1e-6
    assert box == pytest.approx(v.box)


def test_volume_field_placement_is_lossless():
    """Placement composes onto the grid transform instead of resampling, so a
    move and its inverse return the original array — that is the whole point of
    deferring discretisation to the final bake."""
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    from loom.spatial import VolumeField
    v = VolumeField(_NVDB_SAMPLE)
    g = v.read_grid
    orig = np.asarray(g.values, dtype=np.float64)
    round_trips = {
        "rotate 4x90": v.rotated(90.0).rotated(90.0).rotated(90.0).rotated(90.0),
        "rotate 360": v.rotated(360.0, axis=(0.3, -0.5, 0.8)),
        "translate": v.translated(0.13, -0.02, 0.4).translated(-0.13, 0.02, -0.4),
        "scale": v.scaled(3.0).scaled(1.0 / 3.0),
        "scale/centre": v.scaled((2.0, 0.5, 4.0), center=(0.5, 0.5, 0.5))
                         .scaled((0.5, 2.0, 0.25), center=(0.5, 0.5, 0.5)),
    }
    for name, f in round_trips.items():
        vals, _ = vdbio.bake_field(f, v.box, g.shape)
        assert np.abs(vals - orig).max() < 1e-6, name
    # No voxel is touched by a placement — only the transform moves.
    assert v.rotated(37.0).read_grid.values is g.values


def test_volume_field_rotation_actually_rotates():
    """The complement of the round-trip test: a single rotation must *change*
    the baked field (else the loss-free assertions above pass vacuously)."""
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    from loom.spatial import VolumeField
    v = VolumeField(_NVDB_SAMPLE)
    base, _ = vdbio.bake_field(v, v.box, (24, 24, 24))
    turned, _ = vdbio.bake_field(v.rotated(37.0), v.box, (24, 24, 24))
    assert np.abs(turned - base).max() > 0.05
    # ...but a rotation conserves the volume's mass to within resampling error.
    assert turned.sum() == pytest.approx(base.sum(), rel=0.05)


def test_volume_field_fitted_lands_on_the_requested_box():
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    from loom.spatial import VolumeField
    v = VolumeField(_NVDB_SAMPLE)
    want = (0.0, -2.0, 1.0, 1.0, 3.0, 1.5)
    assert v.fitted(want).box == pytest.approx(want, abs=1e-9)
    assert v.fitted(2.0).box == pytest.approx((-2, -2, -2, 2, 2, 2), abs=1e-9)


def test_volume_field_composes_with_the_spatial_algebra():
    """A volume is an ordinary operand: arithmetic and warping both work, and a
    warp reaches inside the coordinate children."""
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    from loom.spatial import VolumeField
    v = VolumeField(_NVDB_SAMPLE)
    box, res = v.box, (16, 16, 16)
    base, _ = vdbio.bake_field(v, box, res)
    doubled, _ = vdbio.bake_field(v * 2.0 + 1.0, box, res)
    assert np.allclose(doubled, 2.0 * base + 1.0)
    # A coordinate substitution is a warp; shifting x by one voxel must equal
    # sampling the unshifted field one voxel over.
    dx = v.read_grid.transform.a[0]
    warped, _ = vdbio.bake_field(VolumeField(_NVDB_SAMPLE, x=X + dx), box, res)
    shifted, _ = vdbio.bake_field(v.translated(-dx, 0.0, 0.0), box, res)
    assert np.abs(warped - shifted).max() < 1e-6
    assert v.children() == (v.x, v.y, v.z)


def test_volume_field_refuses_to_emit_ftsl():
    """ftrace's pattern VM has no volume op, so there is no honest emit; the
    error must say what to do instead rather than produce a wrong string."""
    if not os.path.exists(_NVDB_SAMPLE):
        pytest.skip("sample cloud.nvdb not present")
    from loom.spatial import VolumeField
    with pytest.raises(TypeError, match="write_volume"):
        VolumeField(_NVDB_SAMPLE).emit(("x", "y", "z"), None)


def test_volume_field_grid_selection_and_errors():
    from loom.spatial import VolumeField
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "two.vdb")
        box = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        a = np.zeros((4, 4, 4), dtype="<f4"); a[1, 1, 1] = 1.0
        b = np.zeros((4, 4, 4), dtype="<f4"); b[2, 2, 2] = 5.0
        vdbio.write_vdb(path, [vdbio.VolumeGrid("density", a, box),
                               vdbio.VolumeGrid("temperature", b, box)])
        VolumeField._cache.clear()
        assert VolumeField(path).read_grid.name == "density"   # the conventional default
        assert VolumeField(path, grid="temperature").read_grid.name == "temperature"
        with pytest.raises(ValueError, match="no grid named"):
            VolumeField(path, grid="nope").read_grid
        VolumeField._cache.clear()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all vdbio tests passed")
