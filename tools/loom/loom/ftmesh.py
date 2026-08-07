"""``.ftmesh`` — a binary indexed triangle mesh, written for ftrace to read.

WHY THIS EXISTS
---------------
The live viewer channel (§F4) re-derives the scene every frame and hands ftrace the
swept mesh through a file.  As OBJ text that costs a float->decimal format here and
a decimal->float parse there, every frame, for a mesh that was floats in memory on
both ends.  Measured on the 2160-vertex / 4320-face sweep the viewer actually runs,
the round trip was ~3.2 ms formatting + ~1.4 ms writing on this side and ~2.4 ms
parsing on ftrace's — for geometry neither end ever wanted as text.

This format is the same geometry as raw little-endian f32.  It is deliberately *not*
an interchange format: no materials, no groups, no names, one mesh per file.  OBJ
stays the format for anything a human or another tool reads; this one is for
machine-to-machine handoff inside a single run.

FIDELITY
--------
f32 is a strict *improvement* over the text it replaces.  :func:`loom.sweep.write_obj`
formats with ``%.6g`` — 6 significant decimal digits — where f32 carries ~7.2.
Nothing gets coarser by switching.

LAYOUT
------
Little-endian throughout (writer and reader are the same machine; the magic catches
the case where they somehow are not)::

    0   char  magic[8]   b"FTMESH\\0\\0"
    8   u32   version    1
    12  u32   flags      bit0 = has normals, bit1 = has UVs
    16  u32   nverts
    20  u32   ntris
    24  f32   pos[nverts*3]
        f32   nrm[nverts*3]     if flags & 1
        f32   uv [nverts*2]     if flags & 2
        u32   idx[ntris*3]

Everything is 4-byte aligned and the total size is fully determined by the header,
so ftrace can detect a truncated file instead of reading garbage.

The reader is ``loadFtmesh`` in ``src/mesh.h``; keep the two in sync.
"""

from __future__ import annotations

import struct
import sys
from array import array
from typing import Iterable, Optional, Sequence, Tuple

from .atomicio import write_atomic_bytes

MAGIC = b"FTMESH\0\0"
VERSION = 1
HAS_NORMALS = 1
HAS_UVS = 2

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Tri = Tuple[int, int, int]

#: ``array`` typecodes are native-endian, and ``array.tobytes()`` is an order of
#: magnitude faster than ``struct.pack`` for a few thousand values, so we use it and
#: byteswap on the (currently hypothetical) big-endian host rather than giving that
#: up.  ``array('f')`` is IEEE single and ``array('I')`` is 32-bit on every platform
#: CPython supports that we care about; both are asserted at import.
_BIG_ENDIAN = sys.byteorder != "little"
assert array("f").itemsize == 4, "ftmesh needs a 4-byte 'f' typecode"
assert array("I").itemsize == 4, "ftmesh needs a 4-byte 'I' typecode"


def _packed(typecode: str, values: Iterable[float]) -> bytes:
    a = array(typecode, values)
    if _BIG_ENDIAN:
        a.byteswap()
    return a.tobytes()


def encode(verts: Sequence[Vec3], faces: Sequence[Tri],
           normals: Optional[Sequence[Vec3]] = None,
           uvs: Optional[Sequence[Vec2]] = None) -> bytes:
    """Serialise an indexed mesh to ``.ftmesh`` bytes.

    ``normals`` and ``uvs``, when given, are per-vertex and must be the same length
    as ``verts``.  Supplying normals means ftrace will *not* crease-smooth the mesh
    (authored normals always win, exactly as with OBJ ``vn``).
    """
    nv, nt = len(verts), len(faces)
    if normals is not None and len(normals) != nv:
        raise ValueError(f"ftmesh: {len(normals)} normals for {nv} vertices")
    if uvs is not None and len(uvs) != nv:
        raise ValueError(f"ftmesh: {len(uvs)} uvs for {nv} vertices")
    flags = (HAS_NORMALS if normals is not None else 0) \
          | (HAS_UVS if uvs is not None else 0)

    parts = [MAGIC, struct.pack("<IIII", VERSION, flags, nv, nt)]
    parts.append(_packed("f", (c for v in verts for c in v)))
    if normals is not None:
        parts.append(_packed("f", (c for n in normals for c in n)))
    if uvs is not None:
        parts.append(_packed("f", (c for t in uvs for c in t)))
    parts.append(_packed("I", (i for f in faces for i in f)))
    return b"".join(parts)


def write_ftmesh(path, verts: Sequence[Vec3], faces: Sequence[Tri],
                 normals: Optional[Sequence[Vec3]] = None,
                 uvs: Optional[Sequence[Vec2]] = None) -> None:
    """Write an indexed mesh to ``path`` **atomically**.

    Atomicity is not optional here for the same reason it is not optional for
    :func:`loom.sweep.write_obj`: the live viewer channel re-emits on a worker thread
    while ftrace is still loading the previous emission out of the same directory.
    See :mod:`loom.atomicio`.
    """
    write_atomic_bytes(path, encode(verts, faces, normals, uvs),
                       suffix=".ftmesh.tmp")


def read_ftmesh(path):
    """Read a ``.ftmesh`` back, as ``(verts, faces, normals, uvs)``.

    Only the tests and debugging tools need this — loom writes, ftrace reads — but a
    format with no reader on the writing side is a format nobody can check.
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 24 or data[:8] != MAGIC:
        raise ValueError(f"{path}: not a .ftmesh (bad magic)")
    version, flags, nv, nt = struct.unpack_from("<IIII", data, 8)
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, this module reads {VERSION}")
    has_n, has_uv = bool(flags & HAS_NORMALS), bool(flags & HAS_UVS)
    need = 24 + nv * 12 + (nv * 12 if has_n else 0) + (nv * 8 if has_uv else 0) + nt * 12
    if len(data) < need:
        raise ValueError(f"{path}: truncated ({len(data)} bytes, header implies {need})")

    off = 24

    def take(typecode: str, count: int):
        nonlocal off
        a = array(typecode)
        a.frombytes(data[off:off + count * a.itemsize])
        if _BIG_ENDIAN:
            a.byteswap()
        off += count * a.itemsize
        return a

    pos = take("f", nv * 3)
    verts = [tuple(pos[i * 3:i * 3 + 3]) for i in range(nv)]
    normals = None
    if has_n:
        nrm = take("f", nv * 3)
        normals = [tuple(nrm[i * 3:i * 3 + 3]) for i in range(nv)]
    uvs = None
    if has_uv:
        uv = take("f", nv * 2)
        uvs = [tuple(uv[i * 2:i * 2 + 2]) for i in range(nv)]
    idx = take("I", nt * 3)
    faces = [tuple(idx[i * 3:i * 3 + 3]) for i in range(nt)]
    return verts, faces, normals, uvs
