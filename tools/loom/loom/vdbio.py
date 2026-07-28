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
/ **blosc** value codecs, over every **linear** transform map — the diagonal ones
(Scale, Translate and their combinations) and the general ``AffineMap`` /
``UnitaryMap``.

Reading NanoVDB
---------------
:func:`read_nvdb` additionally ingests **NanoVDB** ``.nvdb`` (v32.6, float 5_4_3)
in either accepted layout — a ``FileHeader``-prefixed multi-grid container, or a
bare raw grid buffer.  A ``.nvdb`` is not a serialised stream but a *memory
image*: a linear buffer of 32-byte-aligned PODs referring to each other by signed
byte offsets, so the reader indexes at fixed offsets and walks the tree rather
than decompressing anything.  This mirrors ftrace's own reader
(``src/vdbgrid.cpp``), the only ``.nvdb`` consumer on the render path;
:func:`read_vdb_grids` dispatches to it on magic, so a caller that just wants
"read whatever volume this is" need not care which format it was handed.  loom
has no NanoVDB *writer* — ``.nvdb`` support is read-only.

The two readers deliberately differ in what they hand back, because they answer
different questions:

* :func:`read_vdb` / :func:`read_vdb_grids` round-trip what this module *writes*,
  so they keep only the **active positive** voxels and bound the box from them.
* :func:`read_nvdb` must agree grid-for-grid with what ftrace renders, so it
  produces a **faithful dense bake** over the tree's active index bbox: inactive
  voxels take the grid's ``background`` and every non-child tile is expanded.
  That is not a nicety — ``LeafData::getValue`` ignores the value mask and
  ``InternalNode::getValue`` returns a tile's value whether or not the tile is
  active, so anything less would silently drop real data (the ``cloud.nvdb``
  sample carries 10 active lower-level tiles = 10 × 8³ voxels).

Rotated grids
-------------
A rotated ``AffineMap`` does not change the *samples* — an OpenVDB tree is always
a regular lattice in **index** space, and the map only says where that lattice
sits in the world.  So the dense array is unaffected; what a rotation breaks is
only the axis-aligned ``box6`` that :func:`read_vdb` returns, which cannot
express a tilted lattice.

Hence the two entry points:

* :func:`read_vdb` → ``{name: (array, box6)}``.  Unchanged, and still **rejects**
  a rotated map — returning an axis-aligned box for a tilted grid would silently
  misplace every voxel, which is worse than an error.
* :func:`read_vdb_grids` → ``{name: ReadGrid}``, carrying the index-space array,
  its index origin and the full :class:`VdbTransform`.  This reads *any* linear
  map, so it is the way to ingest a rotated grid (mirroring ftrace's
  ``readTransform``, which has always accepted ``AffineMap``/``UnitaryMap``).
"""

from __future__ import annotations

import io
import math
import struct
import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["write_vdb", "read_vdb", "read_vdb_grids", "read_nvdb", "bake_field",
           "write_volume", "VolumeGrid", "VdbTransform", "ReadGrid"]

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

# Off-diagonal magnitude (relative to the row scale) below which a map counts as
# axis-aligned.  Not zero, because a DCC that composes a 0°/90°/180° rotation in
# floating point writes ~1e-17 crumbs into the off-diagonals; treating those as a
# real rotation would reject grids that are exactly axis-aligned in intent.
_DIAG_TOL = 1e-12


class VdbTransform:
    """The grid's **index → world** affine: ``world = A · index + t``.

    ``a`` is the 3×3 linear part, **row-major** (``a[row * 3 + col]``), and ``t``
    is the world position of index ``(0, 0, 0)``.  Every OpenVDB linear map
    reduces to this pair, so one representation covers Scale / Translate /
    ScaleTranslate / Affine / Unitary alike — matching ftrace's ``readTransform``
    (``src/vdb_openvdb.cpp``), which fills exactly the same ``A``/``T``.

    Note OpenVDB serialises an ``AffineMap`` in **row-vector** convention
    (``world = index · M``); the column-vector ``A`` above is that matrix
    transposed, which is what the reader stores.
    """

    __slots__ = ("a", "t")

    def __init__(self, a: Sequence[float], t: Sequence[float]) -> None:
        if len(a) != 9 or len(t) != 3:
            raise ValueError("VdbTransform needs a 9-element matrix and a 3-vector")
        self.a = tuple(float(v) for v in a)
        self.t = tuple(float(v) for v in t)

    @classmethod
    def diagonal(cls, scale: Sequence[float], offset: Sequence[float]) -> "VdbTransform":
        """The axis-aligned case: per-axis ``scale`` about ``offset``."""
        sx, sy, sz = (float(v) for v in scale)
        return cls((sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, sz), offset)

    @property
    def is_diagonal(self) -> bool:
        """True when the lattice stays axis-aligned (no rotation or shear).

        Compared against each row's own scale so the test is unit-free: a grid
        measured in metres and the same grid in millimetres agree.
        """
        a = self.a
        for r in range(3):
            row = a[r * 3:r * 3 + 3]
            mag = max(abs(v) for v in row)
            if mag == 0.0:
                continue
            for cidx, v in enumerate(row):
                if cidx != r and abs(v) > _DIAG_TOL * mag:
                    return False
        return True

    @property
    def scale(self) -> Tuple[float, float, float]:
        """Per-axis voxel size — the diagonal of ``a``.

        Meaningful on its own only when :attr:`is_diagonal`; on a rotated map it
        is just the diagonal, not the lattice spacing (use :attr:`voxel_size`).
        """
        return (self.a[0], self.a[4], self.a[8])

    @property
    def voxel_size(self) -> Tuple[float, float, float]:
        """World length of one index step along each axis — the column norms of
        ``a``.  Correct under rotation, where the diagonal alone is not."""
        return tuple(math.sqrt(self.a[c] ** 2 + self.a[3 + c] ** 2 + self.a[6 + c] ** 2)
                     for c in range(3))

    def apply(self, i: float, j: float, k: float) -> Tuple[float, float, float]:
        """Map one index coordinate to world."""
        a, t = self.a, self.t
        return (a[0] * i + a[1] * j + a[2] * k + t[0],
                a[3] * i + a[4] * j + a[5] * k + t[1],
                a[6] * i + a[7] * j + a[8] * k + t[2])

    @property
    def inverse_linear(self) -> Tuple[float, ...]:
        """``A⁻¹`` as a row-major 9-tuple — the **world → index** linear map.

        This is exactly ftrace's ``VdbGrid::ainv`` (``src/vdbgrid.h``), which it
        obtains the same way: invert the 3×3 and sample at
        ``ainv · (p - t) - index_lo``.
        """
        a = self.a
        c0 = a[4] * a[8] - a[5] * a[7]
        c1 = a[5] * a[6] - a[3] * a[8]
        c2 = a[3] * a[7] - a[4] * a[6]
        det = a[0] * c0 + a[1] * c1 + a[2] * c2
        if det == 0.0 or not math.isfinite(det):
            raise ValueError("VdbTransform: singular linear part, cannot invert")
        inv = 1.0 / det
        return (c0 * inv,
                (a[2] * a[7] - a[1] * a[8]) * inv,
                (a[1] * a[5] - a[2] * a[4]) * inv,
                c1 * inv,
                (a[0] * a[8] - a[2] * a[6]) * inv,
                (a[2] * a[3] - a[0] * a[5]) * inv,
                c2 * inv,
                (a[1] * a[6] - a[0] * a[7]) * inv,
                (a[0] * a[4] - a[1] * a[3]) * inv)

    def to_index(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Map one world point to a **fractional** index coordinate (inverse of
        :meth:`apply`)."""
        m, t = self.inverse_linear, self.t
        rx, ry, rz = x - t[0], y - t[1], z - t[2]
        return (m[0] * rx + m[1] * ry + m[2] * rz,
                m[3] * rx + m[4] * ry + m[5] * rz,
                m[6] * rx + m[7] * ry + m[8] * rz)

    def premultiplied(self, m: Sequence[float] = None,
                      d: Sequence[float] = None) -> "VdbTransform":
        """Compose a **world-space** affine ``p ↦ M·p + d`` onto this transform.

        Returns ``A' = M·A``, ``t' = M·t + d`` — i.e. the same lattice *moved*
        in the world.  Nothing is resampled and no value changes: repositioning
        a grid is exact, because an OpenVDB/NanoVDB tree is a regular lattice in
        *index* space and the transform alone says where that lattice sits.
        That is why :class:`~loom.spatial.Volume`'s placement helpers compose
        here rather than warping coordinates — interpolation error is deferred
        to the single final bake, per loom's "discretize last" rule.
        """
        mm = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) if m is None \
            else tuple(float(v) for v in m)
        if len(mm) != 9:
            raise ValueError("premultiplied() needs a 9-element row-major matrix")
        dd = (0.0, 0.0, 0.0) if d is None else tuple(float(v) for v in d)
        if len(dd) != 3:
            raise ValueError("premultiplied() needs a 3-element offset")
        a = self.a
        na = tuple(sum(mm[r * 3 + q] * a[q * 3 + c] for q in range(3))
                   for r in range(3) for c in range(3))
        nt = tuple(sum(mm[r * 3 + q] * self.t[q] for q in range(3)) + dd[r]
                   for r in range(3))
        return VdbTransform(na, nt)

    def __eq__(self, other) -> bool:
        if not isinstance(other, VdbTransform):
            return NotImplemented
        return self.a == other.a and self.t == other.t

    def __hash__(self) -> int:
        return hash((self.a, self.t))

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        kind = "diagonal" if self.is_diagonal else "rotated"
        return f"VdbTransform({kind}, scale={self.voxel_size}, t={self.t})"


class ReadGrid:
    """One grid parsed out of a ``.vdb`` by :func:`read_vdb_grids`.

    ``values`` is dense in **index** space — ``values[di, dj, dk]`` is the voxel
    at index ``index_lo + (di, dj, dk)``.  Keeping it in index space is what lets
    a rotated grid come back at all: the samples are a regular lattice either
    way, and only :attr:`transform` knows where that lattice lies in the world.
    """

    __slots__ = ("name", "values", "index_lo", "transform", "background")

    def __init__(self, name: str, values, index_lo: Tuple[int, int, int],
                 transform: VdbTransform, background: float = 0.0) -> None:
        self.name = name
        self.values = values
        self.index_lo = tuple(int(v) for v in index_lo)
        self.transform = transform
        #: Value of every voxel outside the grid's active topology.  Usually 0
        #: for a fog volume (and always 0 for anything :func:`write_vdb` emits),
        #: but a level set stores its half-band width here, and a real ``.nvdb``
        #: may carry a nonzero haze — so a faithful bake needs it.
        self.background = float(background)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.values.shape)

    @property
    def index_hi(self) -> Tuple[int, int, int]:
        """Inclusive upper index corner."""
        return tuple(lo + n - 1 for lo, n in zip(self.index_lo, self.shape))

    def world_of(self, di: int, dj: int, dk: int) -> Tuple[float, float, float]:
        """World position of the sample at *array* offset ``(di, dj, dk)``."""
        lo = self.index_lo
        return self.transform.apply(lo[0] + di, lo[1] + dj, lo[2] + dk)

    @property
    def box(self) -> Box:
        """The axis-aligned world box of the sample range (corners inclusive).

        Raises for a rotated grid, where no axis-aligned box describes the
        lattice — an approximate one would silently misplace every voxel.
        """
        if not self.transform.is_diagonal:
            raise ValueError(
                f"read_vdb: grid '{self.name}' has a rotated transform, so it has no "
                "axis-aligned world box; use read_vdb_grids() and read .transform "
                "(index -> world) instead of read_vdb()")
        lo, hi = self.index_lo, self.index_hi
        x0, y0, z0 = self.transform.apply(*lo)
        x1, y1, z1 = self.transform.apply(*hi)
        return (x0, y0, z0, x1, y1, z1)

    @property
    def world_box(self) -> Box:
        """The axis-aligned world AABB of the sample range — like :attr:`box`,
        but defined for a **rotated** grid too (it AABBs the eight corners of the
        index box, exactly as ftrace's ``loadVdbGrid`` computes ``wmin``/``wmax``).

        Use this to *bound* a rotated grid (e.g. to pick a bake box); use
        :attr:`box` only when the lattice really is axis-aligned and you need the
        corner-to-corner sample range back.
        """
        lo, hi = self.index_lo, self.index_hi
        pts = [self.transform.apply(i, j, k)
               for i in (lo[0], hi[0]) for j in (lo[1], hi[1]) for k in (lo[2], hi[2])]
        return (min(p[0] for p in pts), min(p[1] for p in pts), min(p[2] for p in pts),
                max(p[0] for p in pts), max(p[1] for p in pts), max(p[2] for p in pts))

    def with_transform(self, transform: VdbTransform) -> "ReadGrid":
        """This grid re-placed under a new index→world map.

        The array is **shared, not copied** — repositioning a volume moves the
        lattice, it does not touch a single voxel.
        """
        return ReadGrid(self.name, self.values, self.index_lo, transform,
                        self.background)

    def sample(self, x, y, z, *, outside=None, clamp_negative: bool = False):
        """Trilinearly sample the grid at world points — a numpy-vectorised port
        of ftrace's ``VdbGrid::sample`` (``src/vdbgrid.h``).

        ``x``/``y``/``z`` are broadcastable arrays (or scalars) of world
        coordinates; the result has their broadcast shape.  Points are mapped to
        fractional lattice coordinates by ``A⁻¹·(p - t) - index_lo``, the
        interpolation stencil is clamped to ``[0, n-1]``, and points further than
        half a voxel outside the lattice read ``outside``.

        Two knobs cover the gap between "what this grid means" and "what ftrace
        renders", because the two genuinely differ:

        ``outside``  value beyond the lattice.  Defaults to the grid's
                     :attr:`background` — the tree's own answer for "no topology
                     here", which is what a level set needs (its band value).
                     Pass ``0.0`` for ftrace's convention, where a density grid
                     simply *does not exist* outside its baked box.
        ``clamp_negative``  clamp the result to ``≥ 0``, as ftrace does for a fog
                     density.  Off by default, since clamping would destroy the
                     inside of a signed level set.

        With ``outside=0.0, clamp_negative=True`` this reproduces ftrace's
        sampler exactly, up to ftrace storing its lattice as fp16.
        """
        import numpy as np
        bg = self.background if outside is None else float(outside)
        v = self.values
        nx, ny, nz = v.shape
        m = self.transform.inverse_linear
        t, lo = self.transform.t, self.index_lo
        rx = np.asarray(x, dtype=np.float64) - t[0]
        ry = np.asarray(y, dtype=np.float64) - t[1]
        rz = np.asarray(z, dtype=np.float64) - t[2]
        fi = m[0] * rx + m[1] * ry + m[2] * rz - lo[0]
        fj = m[3] * rx + m[4] * ry + m[5] * rz - lo[1]
        fk = m[6] * rx + m[7] * ry + m[8] * rz - lo[2]
        fi, fj, fk = np.broadcast_arrays(fi, fj, fk)
        # Half a voxel of margin, matching ftrace: the medium's own AABB already
        # clips rays, so this only guards the interpolation edge.
        inside = ((fi >= -0.5) & (fj >= -0.5) & (fk >= -0.5) &
                  (fi <= nx - 0.5) & (fj <= ny - 0.5) & (fk <= nz - 0.5))
        # Clamp the sample COORDINATE to [0, n-1] before the floor, exactly as
        # ftrace does.  Clamping only the stencil *indices* would leave the
        # fraction near 1 in the outer half-voxel shell below index 0, so the
        # sample would be dominated by the second voxel instead of the edge one.
        # (This also makes the int cast safe for a far-away or NaN point, whose
        # `inside` flag is False anyway.)
        ci = np.clip(np.nan_to_num(fi, nan=0.0), 0.0, nx - 1)
        cj = np.clip(np.nan_to_num(fj, nan=0.0), 0.0, ny - 1)
        ck = np.clip(np.nan_to_num(fk, nan=0.0), 0.0, nz - 1)
        i0 = ci.astype(np.int64); j0 = cj.astype(np.int64); k0 = ck.astype(np.int64)
        tx = ci - i0; ty = cj - j0; tz = ck - k0
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        k1 = np.minimum(k0 + 1, nz - 1)
        c00 = v[i0, j0, k0] * (1 - tx) + v[i1, j0, k0] * tx
        c10 = v[i0, j1, k0] * (1 - tx) + v[i1, j1, k0] * tx
        c01 = v[i0, j0, k1] * (1 - tx) + v[i1, j0, k1] * tx
        c11 = v[i0, j1, k1] * (1 - tx) + v[i1, j1, k1] * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        out = c0 * (1 - tz) + c1 * tz
        if clamp_negative:
            out = np.maximum(out, 0.0)
        return np.where(inside, out, bg)

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return (f"ReadGrid({self.name!r}, shape={self.shape}, "
                f"index_lo={self.index_lo}, {self.transform!r})")


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


def _serialize_body(vol, box: Optional[Box], compression: int = _COMPRESS_ACTIVE_MASK,
                    half: bool = False,
                    transform: Optional[VdbTransform] = None) -> bytes:
    nx, ny, nz = vol.shape

    body = io.BytesIO()
    body.write(struct.pack("<I", compression))
    body.write(struct.pack("<I", 0))                        # grid metamap: empty
    if transform is not None:
        # A general linear map, written as an AffineMap: a 4x4 of doubles in
        # OpenVDB's ROW-VECTOR convention (world = index . M), so M's upper-left
        # 3x3 is the transpose of our column-vector `a` and its last row is the
        # translation.  Read back by _read_map and by ftrace's readTransform.
        a, t = transform.a, transform.t
        _w_str(body, "AffineMap")
        body.write(struct.pack("<16d",
                               a[0], a[3], a[6], 0.0,
                               a[1], a[4], a[7], 0.0,
                               a[2], a[5], a[8], 0.0,
                               t[0], t[1], t[2], 1.0))
    else:
        x0, y0, z0, x1, y1, z1 = (float(v) for v in box)

        def _scale(lo, hi, n):
            if n > 1:
                return (hi - lo) / (n - 1)
            return (hi - lo) or 1.0

        sx, sy, sz = _scale(x0, x1, nx), _scale(y0, y1, ny), _scale(z0, z1, nz)
        # transform: ScaleTranslateMap (world = scale·index + translation)
        _w_str(body, "ScaleTranslateMap")
        body.write(struct.pack("<3d", x0, y0, z0))          # translation
        body.write(struct.pack("<3d", sx, sy, sz))          # scale
        body.write(struct.pack("<3d", sx, sy, sz))          # voxelSize (skipped)
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
    inclusive).

    Pass ``transform=`` (a :class:`VdbTransform`) instead of ``box`` to place the
    lattice with a general index→world affine — the way to write a **rotated**
    grid, which no axis-aligned ``box`` can express.  ``box`` is then derived for
    reference but the transform is what gets serialised.
    """

    def __init__(self, name: str, values, box: Optional[Box] = None, *,
                 transform: Optional[VdbTransform] = None):
        import numpy as np
        self.name = str(name)
        self.values = np.ascontiguousarray(values, dtype="<f4")
        if self.values.ndim != 3:
            raise ValueError("VolumeGrid values must be a 3-D (nx,ny,nz) array")
        if (box is None) == (transform is None):
            raise ValueError("VolumeGrid needs exactly one of box= or transform=")
        self.transform = transform
        if box is None:
            # Index 0..n-1 under the given map.  Only meaningful (and only
            # reported) when the map is axis-aligned; a rotated grid has no box.
            nx, ny, nz = self.values.shape
            lo = transform.apply(0, 0, 0)
            hi = transform.apply(nx - 1, ny - 1, nz - 1)
            self.box: Optional[Box] = (lo + hi) if transform.is_diagonal else None
            return
        b = tuple(float(v) for v in box)
        if len(b) != 6:
            raise ValueError("box must be a 6-tuple (x0,y0,z0,x1,y1,z1)")
        self.box = b  # type: ignore[assignment]


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
    bodies = [_serialize_body(g.values, g.box, compression, half, g.transform)
              for g in grids]
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


def _read_map(g: "_Cur") -> VdbTransform:
    """``Transform::read`` → the index→world affine.

    Byte layouts mirror ftrace's ``readTransform`` (``src/vdb_openvdb.cpp``) so
    the two readers agree grid-for-grid; every linear map OpenVDB emits reduces
    to one 3×3 + offset.
    """
    map_type = g.string()
    if map_type in ("ScaleTranslateMap", "UniformScaleTranslateMap"):
        t = (g.f64(), g.f64(), g.f64())
        s = (g.f64(), g.f64(), g.f64())
        for _ in range(12):
            g.f64()                                 # voxelSize, inverses
        return VdbTransform.diagonal(s, t)
    if map_type in ("UniformScaleMap", "ScaleMap"):
        s = (g.f64(), g.f64(), g.f64())             # scale first, no offset
        for _ in range(12):
            g.f64()
        return VdbTransform.diagonal(s, (0.0, 0.0, 0.0))
    if map_type == "TranslationMap":
        t = (g.f64(), g.f64(), g.f64())             # offset only, unit scale
        return VdbTransform.diagonal((1.0, 1.0, 1.0), t)
    if map_type in ("AffineMap", "UnitaryMap"):
        # A full 4x4 of doubles in OpenVDB's ROW-VECTOR convention (world =
        # index . M), so the column-vector linear part is the transpose of M's
        # upper-left 3x3 and the translation is its last row.
        m = [g.f64() for _ in range(16)]
        return VdbTransform((m[0], m[4], m[8],
                             m[1], m[5], m[9],
                             m[2], m[6], m[10]),
                            (m[12], m[13], m[14]))
    raise ValueError(f"read_vdb: unsupported transform map '{map_type}'")


def read_vdb(path: str) -> Dict[str, Tuple["object", Box]]:
    """Parse a ``.vdb`` back into ``{name: (dense_array, box6)}``.

    Reads the ACTIVE_MASK, **half-float** (``_HalfFloat`` grid type), **ZIP**
    (``COMPRESS_ZIP``, zlib) and **blosc** (``COMPRESS_BLOSC``, via the ``blosc``
    package) value codecs over the diagonal transform maps (Scale / Translate /
    UniformScale and their combinations).  A blosc grid without the ``blosc``
    package raises with a clear message.

    A **rotated** grid has no axis-aligned ``box6``, so it raises here too —
    use :func:`read_vdb_grids`, which returns the index→world transform instead.
    """
    return {name: (gr.values, gr.box) for name, gr in read_vdb_grids(path).items()}


def read_vdb_grids(path: str) -> Dict[str, ReadGrid]:
    """Parse a ``.vdb`` into ``{name: ReadGrid}`` — the full-fidelity read.

    Same value codecs as :func:`read_vdb`, but every **linear** transform map is
    accepted (including a rotated ``AffineMap``/``UnitaryMap``), because the
    samples come back in **index** space with the map carried alongside rather
    than being folded into an axis-aligned box.

    **Dispatches on the file magic**, exactly as ftrace's ``loadVdbGrid`` does,
    so a NanoVDB ``.nvdb`` is handed to :func:`read_nvdb` and callers need not
    care which container a volume arrived in.
    """
    import numpy as np
    with open(path, "rb") as f:
        buf = f.read()
    c = _Cur(buf)
    magic = c.i64()
    if (magic & 0xFFFFFFFF) != _MAGIC:
        if (magic & 0xFFFFFFFFFFFFFF) == (_NVDB_MAGIC_NUMBER & 0xFFFFFFFFFFFFFF):
            return read_nvdb(path)                       # "NanoVDB" + variant byte
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

    out: Dict[str, ReadGrid] = {}
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
        # Transform: any linear map.  The tree is a regular lattice in INDEX
        # space regardless, so a rotation costs the samples nothing — it only
        # means the caller must read `.transform` rather than `.box`.
        xform = _read_map(g)
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
        out[name] = ReadGrid(name, arr, lo, xform, background)
        c.p = end_pos                                       # next descriptor
    return out


# ---- NanoVDB (.nvdb) reader ----------------------------------------------
#
# Mirrors ftrace's `loadVdbGrid` (src/vdbgrid.cpp), which is the only consumer
# of `.nvdb` in this ecosystem, so the two readers agree grid-for-grid.  The
# format is not a serialised stream like `.vdb` — an `.nvdb` grid is a *memory
# image*: a linear buffer of 32-byte-aligned PODs referring to each other by
# signed byte offsets, laid out GridData / TreeData / RootData+tiles /
# InternalNode<5>… / InternalNode<4>… / LeafNode<3>….  So there is nothing to
# decompress: we index the buffer at fixed offsets (verified against the
# vendored `src/third_party/nanovdb/NanoVDB.h`, version 32.6) and walk it.
_NVDB_MAGIC_NUMBER = 0x304244566F6E614E   # "NanoVDB0" — grid *and* file (v32.6)
_NVDB_MAGIC_GRID = 0x314244566F6E614E     # "NanoVDB1" — grid only (newer)
_NVDB_MAGIC_FILE = 0x324244566F6E614E     # "NanoVDB2" — file container only
_NVDB_MAJOR = 32                          # NANOVDB_MAJOR_VERSION_NUMBER

# GridData, 672B.  Offsets from the start of the grid buffer.
_GD_VERSION = 16
_GD_GRID_COUNT = 28
_GD_GRID_SIZE = 32
_GD_NAME = 40                             # char[256], NUL-terminated
_GD_MAT_D = 384                           # double[9], index->world 3x3 (row-major)
_GD_VEC_D = 528                           # double[3], index->world translation
_GD_GRID_TYPE = 636                       # uint32 GridType
_GD_SIZE = 672

# TreeData, 64B, immediately after GridData.
_TD_NODE_OFFSET = 0                       # int64[4]: to first leaf/lower/upper/root
_TD_SIZE = 64

# RootData (float): BBox<Coord> mBBox(24) + mTableSize(4) + background/min/max/
# avg/dev(5*4), padded to the 32B alignment => 64B, then the tile table.
_RD_TABLE_SIZE = 24
_RD_BACKGROUND = 28
_RD_SIZE = 64
_RD_TILE_SIZE = 32                        # key(8) + child(8) + state(4) + value(4), 32B aligned

# InternalData<float, LOG2DIM>: mBBox(24) + mFlags(8) + mValueMask + mChildMask
# + min/max/avg/dev, then a 32B-aligned table of 8-byte {float value | int64
# child} unions.  (level, log2dim) -> (value_mask, child_mask, table).
_ND_UPPER = (32, 4128, 8256)              # LOG2DIM 5: 32768 slots, 4096B masks
_ND_LOWER = (32, 544, 1088)               # LOG2DIM 4:  4096 slots,  512B masks

# LeafData<float, 3>: mBBoxMin(12) + mBBoxDif(3) + mFlags(1) + mValueMask(64) +
# min/max/avg/dev(16), then a 32B-aligned float[512].
_LF_VALUES = 96

_NVDB_TYPE_NAMES = {
    0: "unknown", 1: "float", 2: "double", 3: "int16", 4: "int32", 5: "int64",
    6: "Vec3f", 7: "Vec3d", 8: "Mask", 9: "half", 10: "uint32", 11: "bool",
    12: "RGBA8", 13: "Fp4", 14: "Fp8", 15: "Fp16", 16: "FpN", 17: "Vec4f",
    18: "Vec4d", 19: "Index", 20: "OnIndex", 21: "IndexMask", 22: "OnIndexMask",
    23: "PointIndex", 24: "Vec3u8", 25: "Vec3u16",
}


def _nvdb_set_bits(np, buf: bytes, off: int, nbytes: int):
    """Indices of the set bits in a NanoVDB ``Mask`` — ``mWords[n>>6] & 1<<(n&63)``.

    That is plain little-endian bit order over the byte array, so one
    ``unpackbits`` recovers it (no per-word shuffling needed).
    """
    words = np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=off)
    return np.nonzero(np.unpackbits(words, bitorder="little"))[0]


def _nvdb_fill(arr, lo, origin, dim, value) -> None:
    """Paint the ``dim``-cubed index region at ``origin`` into ``arr``, clipped.

    A tile (a node slot with no child) stands for a whole constant block, and
    ``getValue`` returns that block's value for every voxel inside it — so a
    faithful dense bake has to expand it, exactly as ftrace's accessor does.
    """
    sl = []
    for ax in range(3):
        a = origin[ax] - lo[ax]
        b = a + dim
        a = max(a, 0)
        b = min(b, arr.shape[ax])
        if a >= b:
            return
        sl.append(slice(a, b))
    arr[sl[0], sl[1], sl[2]] = value


def _nvdb_read_grid(np, buf: bytes, base: int) -> ReadGrid:
    """Bake one NanoVDB grid buffer starting at byte ``base`` into a ReadGrid."""
    gtype = struct.unpack_from("<I", buf, base + _GD_GRID_TYPE)[0]
    if gtype != 1:                                       # GridType::Float
        raise ValueError(
            "read_nvdb: only float grids supported (grid is "
            f"{_NVDB_TYPE_NAMES.get(gtype, gtype)})")
    name = buf[base + _GD_NAME:base + _GD_NAME + 256].split(b"\0", 1)[0].decode("ascii")
    mat = struct.unpack_from("<9d", buf, base + _GD_MAT_D)
    vec = struct.unpack_from("<3d", buf, base + _GD_VEC_D)
    # NanoVDB's `matMult(mat, vec, ijk)` is row-major column-vector — the same
    # convention VdbTransform stores — so the 3x3 carries over unchanged.
    xform = VdbTransform(mat, vec)

    tree = base + _GD_SIZE
    node_off = struct.unpack_from("<4q", buf, tree + _TD_NODE_OFFSET)
    if node_off[3] == 0:
        raise ValueError(f"read_nvdb: grid '{name}' has no root node")
    root = tree + node_off[3]

    # The root's own bbox IS the tree's active index bbox (ftrace bakes exactly
    # this range), inclusive on both ends.
    bb = struct.unpack_from("<6i", buf, root)
    lo, hi = bb[:3], bb[3:]
    shape = tuple(hi[a] - lo[a] + 1 for a in range(3))
    if min(shape) <= 0:
        raise ValueError(f"read_nvdb: grid '{name}' has no active voxels")
    if shape[0] * shape[1] * shape[2] > 512 * 1024 * 1024:
        raise ValueError(
            f"read_nvdb: grid '{name}' too large to dense-bake "
            f"({shape[0]}x{shape[1]}x{shape[2]} voxels)")

    background = struct.unpack_from("<f", buf, root + _RD_BACKGROUND)[0]
    arr = np.full(shape, background, dtype=np.float64)

    def walk_internal(node, origin, log2dim, child_log2, offsets, child_is_leaf):
        vm_off, cm_off, tbl_off = offsets
        nslots = 1 << (3 * log2dim)
        mask_bytes = nslots // 8
        child_bits = set(int(v) for v in
                         _nvdb_set_bits(np, buf, node + cm_off, mask_bytes))
        children = np.frombuffer(buf, dtype="<i8", count=nslots, offset=node + tbl_off)
        # The union's float shares the low 4 bytes of each 8-byte slot.
        values = np.frombuffer(buf, dtype="<f4", count=2 * nslots,
                               offset=node + tbl_off)[0::2]
        lmask = (1 << log2dim) - 1
        child_dim = 1 << child_log2
        for n in range(nslots):
            i = (n >> (2 * log2dim)) & lmask
            j = (n >> log2dim) & lmask
            k = n & lmask
            corg = (origin[0] + (i << child_log2),
                    origin[1] + (j << child_log2),
                    origin[2] + (k << child_log2))
            if n in child_bits:
                child = node + int(children[n])
                if child_is_leaf:
                    # LeafData::getValue ignores the value mask, so ftrace's
                    # bake writes every stored voxel — active or not.
                    vals = np.frombuffer(buf, dtype="<f4", count=512,
                                         offset=child + _LF_VALUES)
                    # slot = (dx<<6)|(dy<<3)|dz is exactly C order for (8,8,8).
                    _nvdb_fill_block(arr, lo, corg, vals.reshape(8, 8, 8))
                else:
                    walk_internal(child, corg, 4, 3, _ND_LOWER, True)
            else:
                # A tile, active or not: `getValue` returns its value either way.
                _nvdb_fill(arr, lo, corg, child_dim, float(values[n]))

    table_size = struct.unpack_from("<I", buf, root + _RD_TABLE_SIZE)[0]
    for t in range(table_size):
        tile = root + _RD_SIZE + t * _RD_TILE_SIZE
        key, child = struct.unpack_from("<Qq", buf, tile)
        value = struct.unpack_from("<f", buf, tile + 20)[0]
        # KeyToCoord: 21 bits per axis, x high / y mid / z low, each the
        # ORIGINAL uint32 coordinate shifted down by the child's TOTAL (12).
        km = (1 << 21) - 1
        org = tuple(_as_i32(((key >> s) & km) << 12) for s in (42, 21, 0))
        if child != 0:
            walk_internal(root + child, org, 5, 7, _ND_UPPER, False)
        else:
            _nvdb_fill(arr, lo, org, 1 << 12, value)

    return ReadGrid(name, arr, lo, xform, background)


def _nvdb_fill_block(arr, lo, origin, block) -> None:
    """Paint an 8-cubed leaf block into ``arr``, clipped to its bounds."""
    src, dst = [], []
    for ax in range(3):
        a = origin[ax] - lo[ax]
        c0 = max(-a, 0)
        c1 = min(arr.shape[ax] - a, 8)
        if c0 >= c1:
            return
        src.append(slice(c0, c1))
        dst.append(slice(a + c0, a + c1))
    arr[dst[0], dst[1], dst[2]] = block[src[0], src[1], src[2]]


def _as_i32(v: int) -> int:
    """Reinterpret the low 32 bits of ``v`` as a signed int32."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def read_nvdb(path: str) -> Dict[str, ReadGrid]:
    """Parse a NanoVDB ``.nvdb`` into ``{name: ReadGrid}``.

    Accepts both layouts ftrace accepts: the **file container** (``FileHeader``
    then ``{FileMetaData, name, grid}…``, uncompressed) and a **raw grid
    buffer** (as written by ``writeUncompressedGrids`` or dumped from a
    ``GridHandle``).  Float ``5_4_3`` grids only — the one type ftrace renders.

    Unlike :func:`read_vdb`, the array is the grid's **faithful dense bake**
    over the tree's active index bounding box: inactive voxels carry the grid
    ``background`` and constant *tiles* are expanded, exactly as ftrace's
    accessor-driven bake in ``src/vdbgrid.cpp`` does.  (``read_vdb`` keeps its
    own older convention — active, positive voxels only — because it round-trips
    :func:`write_vdb`, whose background is always 0.)  ``ReadGrid.background``
    carries the background either way.
    """
    import numpy as np
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 16:
        raise ValueError(f"read_nvdb: file too small: '{path}'")
    magic = struct.unpack_from("<Q", buf, 0)[0]

    # Disambiguate the two layouts the way NanoVDB (and ftrace) does: in default
    # builds a file and a grid share the same magic, so the discriminator is
    # whether byte 16 parses as a compatible *grid* version field.
    def grid_major(off: int) -> int:
        if off + _GD_SIZE > len(buf):
            return -1
        return struct.unpack_from("<I", buf, off + _GD_VERSION)[0] >> 21

    out: Dict[str, ReadGrid] = {}
    if magic == _NVDB_MAGIC_GRID or (magic == _NVDB_MAGIC_NUMBER
                                     and grid_major(0) == _NVDB_MAJOR):
        # Raw grid buffer: grids follow one another, each self-describing.
        count = struct.unpack_from("<I", buf, _GD_GRID_COUNT)[0] or 1
        off = 0
        for _ in range(count):
            gr = _nvdb_read_grid(np, buf, off)
            out[gr.name] = gr
            off += struct.unpack_from("<Q", buf, off + _GD_GRID_SIZE)[0]
        return out

    if magic not in (_NVDB_MAGIC_NUMBER, _NVDB_MAGIC_FILE):
        raise ValueError(f"read_nvdb: not a NanoVDB file or raw grid: '{path}'")
    _ver, grid_count, codec = struct.unpack_from("<IHH", buf, 8)
    if codec != 0:
        raise ValueError(
            "read_nvdb: compressed .nvdb not supported (re-export "
            f"uncompressed): '{path}'")
    p = 16
    for _ in range(grid_count):
        grid_size, _file_size = struct.unpack_from("<QQ", buf, p)
        name_size = struct.unpack_from("<I", buf, p + 136)[0]
        p += 176 + name_size                             # FileMetaData + name
        gr = _nvdb_read_grid(np, buf, p)
        out[gr.name] = gr
        p += grid_size
    return out
