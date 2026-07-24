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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all vdbio tests passed")
