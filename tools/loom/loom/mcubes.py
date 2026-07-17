"""
Loom meshing — bake a scalar field to a triangle mesh (M7).

ftrace root-finds isosurfaces directly, so most fields need **no** mesh: you emit
them as an ftsl ``function { expr }`` (see :mod:`loom.iso` / :mod:`loom.spatial`).
This module exists for the minority case where a field *must* be baked to geometry
— e.g. a sampled scatter volume, a numpy-only field with no ftsl twin, or a mesh
you want to hand to another tool.  It turns a numeric field ``f(x, y, z)`` into
``(verts, faces)`` via marching cubes, and :class:`~loom.scene.IsoMesh` writes one
OBJ per frame and emits a ``mesh { file ... }`` reference.

**Meshing backend.**  We use scikit-image's Lewiner marching cubes
(``skimage.measure.marching_cubes``) — a battle-tested, crack-free extractor that
resolves the ambiguous MC cases correctly.  It is imported lazily so loom's core
has no hard dependency on it; :func:`mesh_field` raises a clear install hint if it
is missing.

**Adaptivity (honest scope).**  ``adaptive=True`` is *narrow-band sampling*
adaptivity: a coarse pass finds which coarse blocks straddle the isosurface, and
the field is evaluated on the fine grid **only** inside those blocks (plus a
one-block skirt); far blocks are filled with a large same-sign sentinel.  A single
global marching-cubes pass then runs over the whole fine array, so the result is
**crack-free** and identical to a dense fine mesh near the surface, while skipping
field evaluation on the ~O(res³) cells far from it.  This is the win DESIGN.md
§9 describes as "subdivides more where the field changes fast" — realised as
*evaluation cost*, not variable triangle density.  True variable-density output
(octree dual-contouring with QEF) remains future work; MC emits a uniform-density
surface by construction, and we don't pretend otherwise.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

Vec3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]

# A field is either a numpy-vectorised callable f(X, Y, Z) -> array, or a
# loom.spatial.SpatialExpr (evaluated via .eval_np at a clock).
FieldFn = Callable  # duck-typed; see _as_sampler


def _require_skimage():
    try:
        from skimage.measure import marching_cubes  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only when missing
        raise RuntimeError(
            "loom.mcubes needs scikit-image for marching cubes.\n"
            "Install it with:  pip install scikit-image"
        ) from exc
    return marching_cubes


def _norm_bounds(bounds) -> Tuple[float, float, float, float, float, float]:
    """Accept a scalar half-size, a 3-tuple half-size, or a full 6-tuple box."""
    if isinstance(bounds, (int, float)):
        h = float(bounds)
        return (-h, -h, -h, h, h, h)
    b = tuple(float(v) for v in bounds)
    if len(b) == 3:
        return (-b[0], -b[1], -b[2], b[0], b[1], b[2])
    if len(b) == 6:
        return b  # type: ignore[return-value]
    raise ValueError("bounds must be a scalar, a 3-tuple, or a 6-tuple (xmin..zmax)")


def _norm_res(res) -> Tuple[int, int, int]:
    if isinstance(res, int):
        return (res, res, res)
    r = tuple(int(v) for v in res)
    if len(r) != 3:
        raise ValueError("res must be an int or a 3-tuple")
    return r  # type: ignore[return-value]


def _as_sampler(field, clock, cache):
    """Return a vectorised g(X, Y, Z) -> ndarray for either a SpatialExpr or a
    plain numpy callable.  SpatialExprs are baked at ``clock`` (temporal
    coefficients become constants), so one call meshes one frame."""
    if hasattr(field, "eval_np"):  # duck-typed SpatialExpr
        import numpy as np

        def g(X, Y, Z):
            v = field.eval_np((X, Y, Z), clock, cache)
            return np.broadcast_to(v, X.shape).astype(np.float64)

        return g
    if callable(field):
        return field
    raise TypeError("field must be a SpatialExpr or a callable f(X, Y, Z)")


def _grid_axes(np, box, res):
    x0, y0, z0, x1, y1, z1 = box
    nx, ny, nz = res
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    zs = np.linspace(z0, z1, nz)
    return xs, ys, zs


def _sample_dense(np, g, xs, ys, zs):
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.ascontiguousarray(g(X, Y, Z).astype(np.float64))


def _sample_narrowband(np, g, xs, ys, zs, iso, coarse):
    """Evaluate ``g`` on the fine grid only inside coarse blocks that straddle
    ``iso`` (plus a one-block skirt); fill far blocks with a large same-sign
    sentinel.  Returns a full fine (nx,ny,nz) volume ready for one global MC."""
    nx, ny, nz = len(xs), len(ys), len(zs)
    cx = max(1, min(coarse, nx - 1))
    cy = max(1, min(coarse, ny - 1))
    cz = max(1, min(coarse, nz - 1))
    # Coarse corner indices along each axis (block boundaries land on fine nodes).
    ix = np.unique(np.linspace(0, nx - 1, cx + 1).round().astype(int))
    iy = np.unique(np.linspace(0, ny - 1, cy + 1).round().astype(int))
    iz = np.unique(np.linspace(0, nz - 1, cz + 1).round().astype(int))
    Xc, Yc, Zc = np.meshgrid(xs[ix], ys[iy], zs[iz], indexing="ij")
    coarse_val = g(Xc, Yc, Zc).astype(np.float64)
    s = coarse_val - iso
    big = float(np.nanmax(np.abs(s))) + 1.0 if s.size else 1.0

    vol = np.empty((nx, ny, nz), dtype=np.float64)
    # sentinel sign per fine node from the sign of the nearest coarse corner:
    # cheap nearest via searchsorted onto coarse boundary indices.
    def _blockmap(idx, n):
        # map each fine index 0..n-1 -> its owning coarse-cell index 0..len(idx)-2
        cell = np.searchsorted(idx, np.arange(n), side="right") - 1
        return np.clip(cell, 0, len(idx) - 2)
    bx = _blockmap(ix, nx)
    by = _blockmap(iy, ny)
    bz = _blockmap(iz, nz)

    filled = np.zeros((nx, ny, nz), dtype=bool)
    ncx, ncy, ncz = len(ix) - 1, len(iy) - 1, len(iz) - 1
    # A coarse cell straddles iso if its 8 corner signs are not all equal.
    for i in range(ncx):
        for j in range(ncy):
            for k in range(ncz):
                corners = s[i:i + 2, j:j + 2, k:k + 2]
                if corners.min() < 0.0 <= corners.max():
                    # refine this cell and a one-block skirt around it
                    i0, i1 = ix[max(i - 1, 0)], ix[min(i + 2, ncx)]
                    j0, j1 = iy[max(j - 1, 0)], iy[min(j + 2, ncy)]
                    k0, k1 = iz[max(k - 1, 0)], iz[min(k + 2, ncz)]
                    filled[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = True

    if filled.any():
        fi, fj, fk = np.where(filled)
        X = xs[fi]; Y = ys[fj]; Z = zs[fk]
        vol[filled] = g(X, Y, Z).astype(np.float64) - iso
    # far nodes: large sentinel with the sign of the owning coarse corner
    far = ~filled
    if far.any():
        sign_far = np.sign(s[bx[:, None, None], by[None, :, None], bz[None, None, :]])
        sign_far[sign_far == 0] = 1.0
        vol[far] = (big * sign_far)[far]
    return vol  # already iso-subtracted (surface at 0)


def mesh_field(
    field,
    bounds=1.0,
    res: Union[int, Tuple[int, int, int]] = 48,
    *,
    iso: float = 0.0,
    clock=None,
    cache=None,
    adaptive: bool = False,
    coarse: int = 8,
    gradient_direction: str = "descent",
) -> Tuple[List[Vec3], List[Tri]]:
    """Bake a scalar field to a triangle mesh via marching cubes.

    ``field``   a :class:`~loom.spatial.SpatialExpr` (baked at ``clock``) or a
                vectorised callable ``f(X, Y, Z) -> ndarray``.
    ``bounds``  scalar half-size, ``(hx, hy, hz)`` half-sizes, or a full
                ``(xmin, ymin, zmin, xmax, ymax, zmax)`` box.
    ``res``     grid resolution (int or per-axis 3-tuple) — the "fineness".
    ``iso``     iso-level of the surface (default ``0.0``).
    ``adaptive`` narrow-band sampling: evaluate the fine grid only near the
                surface (see module docstring); crack-free, cheaper for big ``res``.
    ``coarse``  coarse blocks per axis for the adaptive pass.

    Returns ``(verts, faces)`` in **world space** (verts as float 3-tuples, faces
    as 0-based int 3-tuples).  Faces are oriented so normals point toward *lower*
    field values (outward for a solid ``f<0`` interior) by default.
    """
    import numpy as np

    marching_cubes = _require_skimage()
    box = _norm_bounds(bounds)
    rx, ry, rz = _norm_res(res)
    if min(rx, ry, rz) < 2:
        raise ValueError("res must be >= 2 on every axis")
    g = _as_sampler(field, clock, cache)
    xs, ys, zs = _grid_axes(np, box, (rx, ry, rz))

    if adaptive:
        vol = _sample_narrowband(np, g, xs, ys, zs, iso, coarse)
        level = 0.0
    else:
        vol = _sample_dense(np, g, xs, ys, zs)
        level = iso

    vmin, vmax = float(np.nanmin(vol)), float(np.nanmax(vol))
    if not (vmin < level < vmax):
        # surface does not cross this box at this iso -> empty mesh
        return [], []

    spacing = (
        (box[3] - box[0]) / (rx - 1),
        (box[4] - box[1]) / (ry - 1),
        (box[5] - box[2]) / (rz - 1),
    )
    verts_ijk, faces, _normals, _vals = marching_cubes(
        vol, level=level, spacing=spacing, gradient_direction=gradient_direction
    )
    origin = np.array([box[0], box[1], box[2]], dtype=np.float64)
    verts_w = verts_ijk + origin  # spacing already applied -> shift to world origin
    out_v = [tuple(float(c) for c in v) for v in verts_w]
    out_f = [tuple(int(c) for c in f) for f in faces]
    return out_v, out_f
