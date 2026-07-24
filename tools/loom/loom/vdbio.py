"""loom.vdbio — write (and re-read) dense scalar volumes as OpenVDB ``.vdb`` files.

This is loom's **volume-write** capability (roadmap §E4): bake any loom field —
an :class:`~loom.spatial.SpatialExpr`, a :class:`~loom.iso.Isosurface` density,
or a plain ``f(X, Y, Z)`` numpy callable — to a regular lattice and serialise it
to a ``.vdb`` grid that ftrace ingests directly (``density vdb:<path>`` /
``temperature vdb:<path>``).  A single file may carry several **named** grids
(e.g. ``density`` + ``temperature`` for a procedural *fire*), which ftrace
selects by name.

Format subset written
---------------------
The file is a real OpenVDB container for the standard ``float 5_4_3`` tree
(RootNode → Internal<5> → Internal<4> → Leaf<3>), matched **exactly** to
ftrace's hand-rolled reader (``src/vdb_openvdb.cpp``):

* **ACTIVE_MASK** compression — value buffers store just the active (positive)
  voxels, inactive voxels restore to the tree background (0).  Optionally the
  active bytes are additionally **ZIP** (zlib, ``zip=True`` — read by
  :func:`read_vdb` and any OpenVDB tool, *not* by ftrace, so interchange-only)
  or **blosc** (LZ4+byte-shuffle, ``blosc=True`` — the DCC-standard codec, read
  by **both** :func:`read_vdb` and ftrace, so usable on the render path; needs
  the ``blosc`` package).
* **full float** storage by default, or optional 16-bit **half-float** storage
  (``half=True``, grid type ``Tree_float_5_4_3_HalfFloat``) — half the file,
  read directly by ftrace and every OpenVDB tool, ~3 significant digits.
* a **ScaleTranslateMap** transform, so index ``i`` maps to world
  ``x0 + i·(x1-x0)/(n-1)`` — i.e. the grid samples span the box corners
  inclusively, matching loom's ``numpy.linspace`` bake and ftrace's sampler.

Only voxels with ``value > 0`` are stored (ftrace's reader drops ``v ≤ 0`` when
dense-baking anyway), so a compact field yields a compact file.  Intended for
**non-negative scalar fields** (fog density, blackbody temperature); a signed
field's negative lobe is not represented.

The companion :func:`read_vdb` parses back everything this module writes plus a
useful slice of what real DCC tools emit: ACTIVE_MASK / full-float / half / ZIP
/ **blosc** value codecs, over the diagonal transform maps (Scale, Translate and
their combinations).  It does **not** decode a rotated ``AffineMap`` (can't land
on an axis-aligned dense array) or ``.nvdb``.
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["write_vdb", "read_vdb", "bake_field", "write_volume", "VolumeGrid"]

# ---- OpenVDB file-format constants (mirror src/vdb_openvdb.cpp) -----------
_MAGIC = 0x56444220            # "VDB " in the low 32 bits of the int64 magic
_FILE_VERSION = 224            # ≥ 222 (node-mask compression) / ≥ 218 (uuid)
_LIB_MAJOR = 8
_LIB_MINOR = 1
_UUID = "00000000-0000-0000-0000-000000000000"   # 36 ASCII chars
_COMPRESS_ZIP = 0x1            # value buffers zlib-deflated (int64 length prefix)
_COMPRESS_ACTIVE_MASK = 0x2
_COMPRESS_BLOSC = 0x4         # value buffers blosc1-framed (read: needs `blosc`)
_META_NO_MASK_OR_INACTIVE = 0  # NO_MASK_OR_INACTIVE_VALS: inactive → background
_GRID_TYPE = "Tree_float_5_4_3"
_HALF_SUFFIX = "_HalfFloat"   # grid-type suffix flagging 16-bit half storage

Box = Tuple[float, float, float, float, float, float]


# ---- little helpers -------------------------------------------------------
_pairs = zip   # keep the builtin reachable where the ``zip=`` kwarg shadows it


def _w_str(buf: io.BytesIO, s: str) -> None:
    b = s.encode("ascii")
    buf.write(struct.pack("<I", len(b)))
    buf.write(b)


def _mask_bytes(nbits: int, bit_indices) -> bytes:
    m = bytearray(nbits // 8)
    for b in bit_indices:
        m[b >> 3] |= (1 << (b & 7))
    return bytes(m)


def _value_bytes(np, vals, half: bool) -> bytes:
    """Serialise a leaf's active values as raw float32 or half (uint16) bytes."""
    a = np.asarray(vals, dtype="<f4")
    if half:
        return a.astype("<f2").tobytes()
    return a.tobytes()


def _blosc_compress(raw: bytes, typesize: int) -> bytes:
    """Encode a single blosc1 chunk ftrace can read (LZ4 codec + byte shuffle).

    ftrace's built-in decoder only handles the LZ4 codec with *byte* (not bit)
    shuffle, so pin those; a blosc-LZ4 ``.vdb`` is then readable by both ftrace
    and every OpenVDB tool (unlike ZIP, which ftrace can't read)."""
    import blosc
    return blosc.compress(raw, typesize=typesize, cname="lz4", shuffle=blosc.SHUFFLE)


def _write_codec(buf: io.BytesIO, raw: bytes, compression: int,
                 typesize: int = 4) -> None:
    """Write a value buffer honouring the grid compression (OpenVDB io::writeData).

    ZIP/BLOSC wrap the bytes with an int64 length prefix: a **negative** prefix
    means the bytes are stored uncompressed (compression didn't shrink them),
    matching OpenVDB's convention and ftrace's reader.  Only ever called for a
    non-empty buffer (inactive tile arrays store zero values → no codec bytes)."""
    if compression & (_COMPRESS_BLOSC | _COMPRESS_ZIP):
        comp = (_blosc_compress(raw, typesize) if (compression & _COMPRESS_BLOSC)
                else zlib.compress(raw))
        if len(comp) < len(raw):
            buf.write(struct.pack("<q", len(comp)))
            buf.write(comp)
        else:                                   # negative → stored uncompressed
            buf.write(struct.pack("<q", -len(raw)))
            buf.write(raw)
    else:
        buf.write(raw)


# ---- tree node containers -------------------------------------------------
class _Leaf:
    __slots__ = ("origin", "offs", "vals")

    def __init__(self, origin, offs, vals):
        self.origin = origin      # (x,y,z) index, multiple of 8
        self.offs = offs          # ascending leaf offsets of active voxels
        self.vals = vals          # aligned float values


class _Node:
    """An internal node (Internal<5> or Internal<4>).  children keyed by local
    child offset; the node carries no active tiles (all data lives in leaves)."""
    __slots__ = ("origin", "children")

    def __init__(self, origin):
        self.origin = origin
        self.children: Dict[int, object] = {}


def _build_tree(vol) -> Dict[Tuple[int, int, int], _Node]:
    """Partition the positive voxels of a dense (nx,ny,nz) array into the
    OpenVDB ``5_4_3`` tree; return the top Internal<5> nodes keyed by origin."""
    import numpy as np

    ii, jj, kk = np.nonzero(vol > 0)
    if ii.size == 0:
        return {}
    vals = vol[ii, jj, kk].astype("<f4")
    ii = ii.astype(np.int64); jj = jj.astype(np.int64); kk = kk.astype(np.int64)
    leaf_o = np.stack([(ii >> 3) << 3, (jj >> 3) << 3, (kk >> 3) << 3], axis=1)
    off = ((ii & 7) << 6) | ((jj & 7) << 3) | (kk & 7)

    uniq, inv = np.unique(leaf_o, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]; off_s = off[order]; vals_s = vals[order]
    bounds = np.searchsorted(inv_s, np.arange(len(uniq) + 1))

    leaves: Dict[Tuple[int, int, int], _Leaf] = {}
    for u in range(len(uniq)):
        a, b = int(bounds[u]), int(bounds[u + 1])
        offs = off_s[a:b]; vv = vals_s[a:b]
        so = np.argsort(offs, kind="stable")
        origin = (int(uniq[u][0]), int(uniq[u][1]), int(uniq[u][2]))
        leaves[origin] = _Leaf(origin, offs[so].tolist(), vv[so].tolist())

    # group leaves → Internal<4> (128³) → Internal<5> (4096³)
    i4map: Dict[Tuple[int, int, int], _Node] = {}
    for lo, leaf in leaves.items():
        i4o = (lo[0] & ~127, lo[1] & ~127, lo[2] & ~127)
        node = i4map.get(i4o)
        if node is None:
            node = i4map[i4o] = _Node(i4o)
        i = (lo[0] - i4o[0]) >> 3
        j = (lo[1] - i4o[1]) >> 3
        k = (lo[2] - i4o[2]) >> 3
        node.children[i * 256 + j * 16 + k] = leaf

    i5map: Dict[Tuple[int, int, int], _Node] = {}
    for i4o, node in i4map.items():
        i5o = (i4o[0] & ~4095, i4o[1] & ~4095, i4o[2] & ~4095)
        top = i5map.get(i5o)
        if top is None:
            top = i5map[i5o] = _Node(i5o)
        i = (i4o[0] - i5o[0]) >> 7
        j = (i4o[1] - i5o[1]) >> 7
        k = (i4o[2] - i5o[2]) >> 7
        top.children[i * 1024 + j * 32 + k] = node
    return i5map


def _write_internal_topology(buf: io.BytesIO, node: _Node, numValues: int,
                             child_is_leaf: bool) -> None:
    # childMask, valueMask (all-zero: no active tiles), then values (metadata
    # byte + zero active values, since valueMask has no bits set).
    buf.write(_mask_bytes(numValues, node.children.keys()))
    buf.write(bytes(numValues // 8))                       # valueMask = 0
    buf.write(struct.pack("<b", _META_NO_MASK_OR_INACTIVE))  # values metadata
    # children in ascending child-offset order
    for off in sorted(node.children):
        child = node.children[off]
        if child_is_leaf:
            buf.write(_mask_bytes(512, child.offs))         # leaf valueMask
        else:
            _write_internal_topology(buf, child, 4096, True)


def _write_internal_buffers(buf: io.BytesIO, node: _Node,
                            child_is_leaf: bool, compression: int, half: bool) -> None:
    import numpy as np
    # readInternalBuffers recurses child-internals first, then child-leaves.
    if child_is_leaf:
        for off in sorted(node.children):
            leaf: _Leaf = node.children[off]
            buf.write(_mask_bytes(512, leaf.offs))          # valueMask again
            buf.write(struct.pack("<b", _META_NO_MASK_OR_INACTIVE))
            _write_codec(buf, _value_bytes(np, leaf.vals, half), compression,
                         2 if half else 4)
    else:
        for off in sorted(node.children):
            _write_internal_buffers(buf, node.children[off], True, compression, half)


def _serialize_body(vol, box: Box, compression: int = _COMPRESS_ACTIVE_MASK,
                    half: bool = False) -> bytes:
    nx, ny, nz = vol.shape
    x0, y0, z0, x1, y1, z1 = (float(v) for v in box)

    def _scale(lo, hi, n):
        if n > 1:
            return (hi - lo) / (n - 1)
        return (hi - lo) or 1.0

    sx, sy, sz = _scale(x0, x1, nx), _scale(y0, y1, ny), _scale(z0, z1, nz)

    body = io.BytesIO()
    body.write(struct.pack("<I", compression))
    body.write(struct.pack("<I", 0))                        # grid metamap: empty
    # transform: ScaleTranslateMap (world = scale·index + translation)
    _w_str(body, "ScaleTranslateMap")
    body.write(struct.pack("<3d", x0, y0, z0))              # translation
    body.write(struct.pack("<3d", sx, sy, sz))              # scale
    body.write(struct.pack("<3d", sx, sy, sz))              # voxelSize (skipped)
    body.write(struct.pack("<3d", 1.0 / sx, 1.0 / sy, 1.0 / sz))
    body.write(struct.pack("<3d", 1.0 / sx**2, 1.0 / sy**2, 1.0 / sz**2))
    body.write(struct.pack("<3d", 0.5 / sx, 0.5 / sy, 0.5 / sz))
    # tree
    body.write(struct.pack("<i", 1))                        # bufferCount
    body.write(struct.pack("<f", 0.0))                      # background

    tops = _build_tree(vol)
    body.write(struct.pack("<I", 0))                        # numTiles
    body.write(struct.pack("<I", len(tops)))                # numChildren
    for i5o in sorted(tops):
        body.write(struct.pack("<3i", *i5o))               # top-node origin
        _write_internal_topology(body, tops[i5o], 32768, False)
    for i5o in sorted(tops):
        _write_internal_buffers(body, tops[i5o], False, compression, half)
    return body.getvalue()


# ---- public writer --------------------------------------------------------
class VolumeGrid:
    """A named dense scalar grid: ``values`` is an (nx,ny,nz) array, ``box`` is
    the world-space extent ``(x0,y0,z0,x1,y1,z1)`` the samples span (corners
    inclusive)."""

    def __init__(self, name: str, values, box: Box):
        import numpy as np
        self.name = str(name)
        self.values = np.ascontiguousarray(values, dtype="<f4")
        if self.values.ndim != 3:
            raise ValueError("VolumeGrid values must be a 3-D (nx,ny,nz) array")
        b = tuple(float(v) for v in box)
        if len(b) != 6:
            raise ValueError("box must be a 6-tuple (x0,y0,z0,x1,y1,z1)")
        self.box: Box = b  # type: ignore[assignment]


def write_vdb(path: str, grids: Sequence[VolumeGrid], *,
              half: bool = False, zip: bool = False, blosc: bool = False) -> str:
    """Write one or more :class:`VolumeGrid` to ``path`` as a single ``.vdb``.

    Returns ``path``.  Grid names must be unique; ftrace selects a grid by name
    (``density vdb:<path>`` picks the ``density`` grid, etc.).

    ``half``  store voxels as 16-bit half-floats (grid type gains the
              ``_HalfFloat`` suffix).  Halves the file and is read directly by
              ftrace and any OpenVDB tool; the trade-off is ~3-decimal-digit
              precision (fine for fog density / temperature).
    ``zip``   zlib-deflate each value buffer (OpenVDB ``COMPRESS_ZIP``).  Read
              back by :func:`read_vdb` and any OpenVDB tool, but **not** by
              ftrace's built-in reader (which supports blosc-LZ4, not ZIP) — use
              it for interchange / round-tripping, not the render path.
    ``blosc`` blosc1-compress each value buffer (LZ4 codec + byte shuffle,
              OpenVDB ``COMPRESS_BLOSC``) — the DCC-standard codec, read by
              **both** :func:`read_vdb` and ftrace, so it *is* usable on the
              render path (needs the ``blosc`` package).  Mutually exclusive with
              ``zip``.

    The default (all off) is byte-for-byte the original ACTIVE_MASK / full-float
    output, so existing files and ftrace reads are unaffected."""
    grids = list(grids)
    if not grids:
        raise ValueError("write_vdb: no grids given")
    names = [g.name for g in grids]
    if len(set(names)) != len(names):
        raise ValueError(f"write_vdb: duplicate grid names {names}")
    if zip and blosc:
        raise ValueError("write_vdb: choose one of zip / blosc, not both")

    compression = (_COMPRESS_ACTIVE_MASK
                   | (_COMPRESS_ZIP if zip else 0)
                   | (_COMPRESS_BLOSC if blosc else 0))
    bodies = [_serialize_body(g.values, g.box, compression, half) for g in grids]
    gtype = _GRID_TYPE + (_HALF_SUFFIX if half else "")

    # header
    hdr = io.BytesIO()
    hdr.write(struct.pack("<q", _MAGIC))
    hdr.write(struct.pack("<I", _FILE_VERSION))
    hdr.write(struct.pack("<I", _LIB_MAJOR))
    hdr.write(struct.pack("<I", _LIB_MINOR))
    hdr.write(struct.pack("<B", 1))                         # hasGridOffsets
    hdr.write(_UUID.encode("ascii"))                        # 36 bytes
    hdr.write(struct.pack("<I", 0))                         # file metamap: empty
    hdr.write(struct.pack("<I", len(grids)))                # grid count
    header = hdr.getvalue()

    # descriptors carry absolute file offsets; lay them out sequentially.
    pos = len(header)
    descriptors: List[bytes] = []
    for g, body in _pairs(grids, bodies):
        d = io.BytesIO()
        _w_str(d, g.name)
        _w_str(d, gtype)
        _w_str(d, "")                                       # instance parent
        # placeholder offsets patched below
        desc_prefix = d.getvalue()
        desc_len = len(desc_prefix) + 24                    # + 3× int64
        grid_pos = pos + desc_len
        end_pos = grid_pos + len(body)
        d.write(struct.pack("<q", grid_pos))
        d.write(struct.pack("<q", grid_pos))                # block position
        d.write(struct.pack("<q", end_pos))
        descriptors.append(d.getvalue())
        pos = end_pos

    with open(path, "wb") as f:
        f.write(header)
        for desc, body in _pairs(descriptors, bodies):
            f.write(desc)
            f.write(body)
    return path


# ---- bake helpers (reuse loom's field-sampling machinery) -----------------
def bake_field(field, box, res, clock=None, cache=None):
    """Sample ``field`` on a regular lattice; return ``(values, box6)``.

    ``field``  a :class:`~loom.spatial.SpatialExpr` (baked at ``clock``), an
               :class:`~loom.iso.Isosurface` (its density field is used), or a
               numpy-vectorised ``f(X, Y, Z) -> ndarray``.
    ``box``    a scalar/3-tuple half-size or a 6-tuple ``(x0..z1)``.
    ``res``    an int or ``(nx, ny, nz)``.
    """
    import numpy as np
    from . import mcubes
    from .signals import Cache

    # An Isosurface carries its implicit function on ``.field``; unwrap to it so
    # the *density* (not the thresholded surface) is baked.
    fld = field
    if hasattr(field, "field") and not hasattr(field, "eval_np") \
            and not callable(field):
        fld = field.field

    box6 = mcubes._norm_bounds(box)
    nx, ny, nz = mcubes._norm_res(res)
    cache = cache if cache is not None else Cache()
    g = mcubes._as_sampler(fld, clock, cache)
    xs, ys, zs = mcubes._grid_axes(np, box6, (nx, ny, nz))
    vol = mcubes._sample_dense(np, g, xs, ys, zs)
    return np.ascontiguousarray(vol, dtype="<f4"), box6


def write_volume(path: str, *, box, res, clock=None, cache=None,
                 half: bool = False, zip: bool = False, blosc: bool = False,
                 **fields) -> str:
    """Bake one or more named fields over a shared ``box``/``res`` and write a
    multi-grid ``.vdb``.

    Example (a procedural fire ftrace renders as blackbody emission)::

        write_volume("fire.vdb", box=2.0, res=96,
                     density=smoke_density, temperature=hot_core)

    Each keyword becomes a named grid; ``density``/``temperature`` are the names
    ftrace's fire pipeline expects.
    """
    if not fields:
        raise ValueError("write_volume: give at least one named field")
    from .signals import Cache
    cache = cache if cache is not None else Cache()
    grids = []
    for name, field in fields.items():
        vol, box6 = bake_field(field, box, res, clock=clock, cache=cache)
        grids.append(VolumeGrid(name, vol, box6))
    return write_vdb(path, grids, half=half, zip=zip, blosc=blosc)


# ---- reader (round-trip / light loom-side read of THIS module's output) ---
class _Cur:
    def __init__(self, b: bytes, p: int = 0):
        self.b = b; self.p = p

    def take(self, k: int) -> bytes:
        q = self.b[self.p:self.p + k]
        if len(q) != k:
            raise ValueError("unexpected end of .vdb")
        self.p += k
        return q

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def i32(self): return struct.unpack("<i", self.take(4))[0]
    def i64(self): return struct.unpack("<q", self.take(8))[0]
    def u8(self): return struct.unpack("<B", self.take(1))[0]
    def i8(self): return struct.unpack("<b", self.take(1))[0]
    def f32(self): return struct.unpack("<f", self.take(4))[0]
    def f64(self): return struct.unpack("<d", self.take(8))[0]

    def string(self) -> str:
        n = self.u32()
        return self.take(n).decode("ascii")

    def skip_meta(self):
        self.string(); self.string()
        self.take(self.u32())


def _popcount(mask: bytes) -> int:
    return sum(bin(b).count("1") for b in mask)


def _bit_on(mask: bytes, n: int) -> bool:
    return bool((mask[n >> 3] >> (n & 7)) & 1)


def _blosc_decompress(chunk: bytes) -> bytes:
    """Decode a blosc1 chunk (the codec every mainstream .vdb writer uses).

    Delegates to python-``blosc`` (handles BloscLZ/LZ4/Zlib/Zstd + byte/bit
    shuffle — the full range, unlike ftrace's built-in LZ4-only decoder).  Kept a
    soft dependency: absent it, blosc grids raise a clear install hint."""
    try:
        import blosc
    except ImportError as e:                                # pragma: no cover
        raise NotImplementedError(
            "read_vdb: blosc-compressed .vdb needs the 'blosc' package "
            "(pip install blosc), or re-export the file uncompressed/zip/half"
        ) from e
    return blosc.decompress(chunk)


def _read_codec(c: _Cur, nbytes: int, compression: int) -> bytes:
    """Read one value buffer honouring the grid compression (OpenVDB io::readData).

    A ZIP/BLOSC buffer is prefixed by an int64: negative means the ``|prefix|``
    bytes that follow are stored uncompressed, positive is the compressed length."""
    if compression & (_COMPRESS_BLOSC | _COMPRESS_ZIP):
        ncomp = struct.unpack("<q", c.take(8))[0]
        if ncomp <= 0:
            return c.take(-ncomp)                           # stored uncompressed
        blob = c.take(ncomp)
        raw = (_blosc_decompress(blob) if (compression & _COMPRESS_BLOSC)
               else zlib.decompress(blob))
        if len(raw) != nbytes:
            raise ValueError("read_vdb: decompressed size mismatch")
        return raw
    return c.take(nbytes)


def _read_values(c: _Cur, dest_count: int, value_mask: bytes, background: float,
                 compression: int = _COMPRESS_ACTIVE_MASK, from_half: bool = False):
    import numpy as np
    metadata = c.i8()                                       # file ver ≥ 222
    inactive0 = background if metadata == 0 else -background
    if metadata in (2, 4, 5):
        inactive0 = c.f32()                                 # always full float
        if metadata == 5:
            c.f32()
    if metadata in (3, 4, 5):
        c.take(dest_count // 8)                             # selection mask
    temp_count = _popcount(value_mask) if metadata != 6 else dest_count
    if temp_count == 0:                                     # no codec bytes written
        temp = np.zeros(0, dtype=np.float64)
    elif from_half:
        raw = _read_codec(c, temp_count * 2, compression)
        temp = np.frombuffer(raw, dtype="<f2").astype(np.float64)
    else:
        raw = _read_codec(c, temp_count * 4, compression)
        temp = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if metadata == 6:
        return np.array(temp, dtype=np.float64)
    # Scatter the popcount stored values back to their active slots.  The mask is
    # LSB-first within each byte (matching ``_bit_on``); active values are stored
    # in ascending slot order, so a boolean-index assignment restores them.
    bits = np.unpackbits(np.frombuffer(value_mask, dtype=np.uint8),
                         count=dest_count, bitorder="little").astype(bool)
    dest = np.full(dest_count, inactive0, dtype=np.float64)
    dest[bits] = temp
    return dest


def read_vdb(path: str) -> Dict[str, Tuple["object", Box]]:
    """Parse a ``.vdb`` back into ``{name: (dense_array, box6)}``.

    Reads the ACTIVE_MASK, **half-float** (``_HalfFloat`` grid type), **ZIP**
    (``COMPRESS_ZIP``, zlib) and **blosc** (``COMPRESS_BLOSC``, via the ``blosc``
    package) value codecs over the diagonal transform maps (Scale / Translate /
    UniformScale and their combinations).  A blosc grid without the ``blosc``
    package, or a rotated ``AffineMap``, raises with a clear message."""
    import numpy as np
    with open(path, "rb") as f:
        buf = f.read()
    c = _Cur(buf)
    magic = c.i64()
    if (magic & 0xFFFFFFFF) != _MAGIC:
        raise ValueError("not an OpenVDB file")
    file_ver = c.u32()
    c.u32(); c.u32()                                        # library version
    c.u8()                                                  # hasGridOffsets
    if file_ver >= 218:
        c.take(36)                                          # uuid
    for _ in range(c.u32()):                                # file metamap
        c.skip_meta()
    grid_count = c.u32()

    # Leaf slot → (dx,dy,dz) decomposition, precomputed once for vectorised fill.
    _off = np.arange(512)
    _LEAF_DX = (_off >> 6) & 7
    _LEAF_DY = (_off >> 3) & 7
    _LEAF_DZ = _off & 7

    out: Dict[str, Tuple[object, Box]] = {}
    for _ in range(grid_count):
        # OpenVDB "unique names" append 0x1e + instance index to disambiguate
        # duplicates — strip it back to the authored grid name.
        name = c.string().split("\x1e", 1)[0]
        gtype = c.string()
        if file_ver >= 216:
            c.string()                                      # instance parent
        grid_pos = c.i64(); c.i64(); end_pos = c.i64()
        from_half = gtype.endswith(_HALF_SUFFIX)
        base_type = gtype[:-len(_HALF_SUFFIX)] if from_half else gtype
        if base_type != _GRID_TYPE:
            c.p = end_pos
            continue
        g = _Cur(buf, grid_pos)
        compression = g.u32()                               # compression flags
        for _ in range(g.u32()):                            # grid metamap
            g.skip_meta()
        # Transform: the diagonal (axis-aligned) maps that keep the samples on a
        # regular lattice.  A rotated AffineMap can't project onto loom's dense
        # array, so it's rejected.  (Byte layouts mirror ftrace's readTransform.)
        map_type = g.string()
        tx = ty = tz = 0.0
        sx = sy = sz = 1.0
        if map_type in ("ScaleTranslateMap", "UniformScaleTranslateMap"):
            tx, ty, tz = g.f64(), g.f64(), g.f64()
            sx, sy, sz = g.f64(), g.f64(), g.f64()
            for _ in range(12):
                g.f64()                                     # voxelSize, inverses
        elif map_type in ("UniformScaleMap", "ScaleMap"):
            sx, sy, sz = g.f64(), g.f64(), g.f64()          # scale first, no offset
            for _ in range(12):
                g.f64()
        elif map_type == "TranslationMap":
            tx, ty, tz = g.f64(), g.f64(), g.f64()          # offset only, unit scale
        else:
            raise ValueError(
                f"read_vdb: unsupported map '{map_type}' "
                "(only diagonal scale/translate maps; a rotated AffineMap "
                "can't be read onto an axis-aligned dense grid)")
        g.i32()                                             # bufferCount
        background = g.f32()
        num_tiles = g.u32()
        num_children = g.u32()
        for _ in range(num_tiles):
            g.i32(); g.i32(); g.i32(); g.f32(); g.u8()

        leaves: List[Tuple[Tuple[int, int, int], bytes, "object"]] = []

        def read_internal(gg, ox, oy, oz, log2dim, child_total, child_is_leaf):
            num_values = 1 << (3 * log2dim)
            mb = num_values // 8
            child_mask = gg.take(mb)
            value_mask = gg.take(mb)
            _read_values(gg, num_values, value_mask, background,
                         compression, from_half)
            mask = (1 << log2dim) - 1
            kids = []
            for off in range(num_values):
                if not _bit_on(child_mask, off):
                    continue
                i = (off >> (2 * log2dim)) & mask
                j = (off >> log2dim) & mask
                k = off & mask
                cx = ox + (i << child_total)
                cy = oy + (j << child_total)
                cz = oz + (k << child_total)
                if child_is_leaf:
                    lvm = gg.take(64)
                    kids.append(("leaf", (cx, cy, cz), lvm))
                else:
                    kids.append(("node",
                                 read_internal(gg, cx, cy, cz, 4, 3, True)))
            return (ox, oy, oz, kids)

        roots = []
        for _ in range(num_children):
            cx, cy, cz = g.i32(), g.i32(), g.i32()
            roots.append(read_internal(g, cx, cy, cz, 5, 7, False))

        def read_buffers(node):
            _ox, _oy, _oz, kids = node
            for kind, *rest in kids:
                if kind == "node":
                    read_buffers(rest[0])
            for kind, *rest in kids:
                if kind == "leaf":
                    origin, _vm = rest
                    vm = g.take(64)
                    vals = _read_values(g, 512, vm, background,
                                        compression, from_half)
                    leaves.append((origin, vm, vals))

        for r in roots:
            read_buffers(r)

        # Collect the active positive voxels (per-leaf, vectorised), then bound
        # the index box and scatter them into a dense array.  Leaf origins are
        # disjoint and offsets unique within a leaf, so global indices never
        # collide (matching the old last-write loop, now order-independent).
        xs_all: List = []; ys_all: List = []; zs_all: List = []; vs_all: List = []
        for origin, vm, vals in leaves:
            active = np.unpackbits(np.frombuffer(vm, dtype=np.uint8),
                                   count=512, bitorder="little").astype(bool)
            keep = active & (vals > 0.0)
            if not keep.any():
                continue
            idx = np.nonzero(keep)[0]
            xs_all.append(origin[0] + _LEAF_DX[idx])
            ys_all.append(origin[1] + _LEAF_DY[idx])
            zs_all.append(origin[2] + _LEAF_DZ[idx])
            vs_all.append(vals[idx])
        if not xs_all:
            raise ValueError("read_vdb: empty grid")
        xs = np.concatenate(xs_all); ys = np.concatenate(ys_all)
        zs = np.concatenate(zs_all); vs = np.concatenate(vs_all)
        lo = (int(xs.min()), int(ys.min()), int(zs.min()))
        hi = (int(xs.max()), int(ys.max()), int(zs.max()))
        nx = hi[0] - lo[0] + 1; ny = hi[1] - lo[1] + 1; nz = hi[2] - lo[2] + 1
        arr = np.zeros((nx, ny, nz), dtype=np.float64)
        arr[xs - lo[0], ys - lo[1], zs - lo[2]] = vs
        # world box of the active index range [lo, hi] (samples, corners incl.)
        box6 = (tx + lo[0] * sx, ty + lo[1] * sy, tz + lo[2] * sz,
                tx + hi[0] * sx, ty + hi[1] * sy, tz + hi[2] * sz)
        out[name] = (arr, box6)
        c.p = end_pos                                       # next descriptor
    return out
