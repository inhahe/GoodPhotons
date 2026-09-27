"""Numba geometry kit for building grooms from meshes (tools/alice2_hair.py uses it):

  TriGrid     a uniform grid over a triangle soup: exact closest point (batched, or `closest1`
              for one point inside other numba code), first-hit ray casts, ray-parity inside
  ZParity     a fast inside test for a CLOSED mesh: crossings of the +z ray, triangles binned by
              their xy footprint
  weld / vertex_normals / principal_dirs   (min / max curvature directions from a least-squares
              shape operator over a radius neighbourhood)
  trace_surface  streamlines of a per-vertex tangent field over a mesh (RK2, re-projected onto
              the surface every step, coasting across short degenerate stretches)
  Glb         a minimal GLB reader (JSON + BIN, accessors, embedded images)

Everything is in the units the mesh is authored in; nothing here is specific to one model.
"""
import numpy as np
import numba as nb


@nb.njit(cache=True, inline="always")
def _cpt(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    # Ericson, Real-Time Collision Detection 5.1.5, scalar form (no temporaries)
    abx = bx - ax; aby = by - ay; abz = bz - az
    acx = cx - ax; acy = cy - ay; acz = cz - az
    apx = px - ax; apy = py - ay; apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz; d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0: return ax, ay, az
    bpx = px - bx; bpy = py - by; bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz; d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3: return bx, by, bz
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return ax + v * abx, ay + v * aby, az + v * abz
    cpx = px - cx; cpy = py - cy; cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz; d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6: return cx, cy, cz
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return ax + w * acx, ay + w * acy, az + w * acz
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom; w = vc * denom
    return ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w


@nb.njit(cache=True)
def _build_counts(lo, inv, dims, tmin, tmax):
    nt = tmin.shape[0]
    counts = np.zeros(dims[0] * dims[1] * dims[2] + 1, np.int64)
    for t in range(nt):
        i0 = max(0, min(dims[0] - 1, int((tmin[t, 0] - lo[0]) * inv)))
        j0 = max(0, min(dims[1] - 1, int((tmin[t, 1] - lo[1]) * inv)))
        k0 = max(0, min(dims[2] - 1, int((tmin[t, 2] - lo[2]) * inv)))
        i1 = max(0, min(dims[0] - 1, int((tmax[t, 0] - lo[0]) * inv)))
        j1 = max(0, min(dims[1] - 1, int((tmax[t, 1] - lo[1]) * inv)))
        k1 = max(0, min(dims[2] - 1, int((tmax[t, 2] - lo[2]) * inv)))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    counts[(i * dims[1] + j) * dims[2] + k + 1] += 1
    return counts


@nb.njit(cache=True)
def _build_fill(lo, inv, dims, tmin, tmax, start):
    nt = tmin.shape[0]
    fill = start[:-1].copy()
    items = np.empty(start[-1], np.int64)
    for t in range(nt):
        i0 = max(0, min(dims[0] - 1, int((tmin[t, 0] - lo[0]) * inv)))
        j0 = max(0, min(dims[1] - 1, int((tmin[t, 1] - lo[1]) * inv)))
        k0 = max(0, min(dims[2] - 1, int((tmin[t, 2] - lo[2]) * inv)))
        i1 = max(0, min(dims[0] - 1, int((tmax[t, 0] - lo[0]) * inv)))
        j1 = max(0, min(dims[1] - 1, int((tmax[t, 1] - lo[1]) * inv)))
        k1 = max(0, min(dims[2] - 1, int((tmax[t, 2] - lo[2]) * inv)))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    c = (i * dims[1] + j) * dims[2] + k
                    items[fill[c]] = t
                    fill[c] += 1
    return items


@nb.njit(cache=True, parallel=True)
def _closest(q, V, F, lo, cell, dims, start, items, maxr):
    n = q.shape[0]
    out_p = np.empty((n, 3)); out_t = np.full(n, -1, np.int64); out_d = np.full(n, np.inf)
    for qi in nb.prange(n):
        p = q[qi]
        ci = int((p[0] - lo[0]) / cell); cj = int((p[1] - lo[1]) / cell); ck = int((p[2] - lo[2]) / cell)
        best = np.inf; best2 = np.inf; bt = -1; bx = 0.0; by = 0.0; bz = 0.0
        for r in range(0, maxr + 1):
            # the shell of cells at Chebyshev radius r; stop when it cannot beat the best
            if bt >= 0:
                # distance from p to the outside of the (2r-1)-cube already searched
                lo_x = lo[0] + (ci - r + 1) * cell; hi_x = lo[0] + (ci + r) * cell
                lo_y = lo[1] + (cj - r + 1) * cell; hi_y = lo[1] + (cj + r) * cell
                lo_z = lo[2] + (ck - r + 1) * cell; hi_z = lo[2] + (ck + r) * cell
                m = min(p[0] - lo_x, hi_x - p[0], p[1] - lo_y, hi_y - p[1], p[2] - lo_z, hi_z - p[2])
                if m >= best: break
            for i in range(ci - r, ci + r + 1):
                if i < 0 or i >= dims[0]: continue
                for j in range(cj - r, cj + r + 1):
                    if j < 0 or j >= dims[1]: continue
                    onface = abs(i - ci) == r or abs(j - cj) == r
                    kstep = 1 if (onface or r == 0) else 2 * r
                    for k in range(ck - r, ck + r + 1, kstep):
                        if k < 0 or k >= dims[2]: continue
                        c = (i * dims[1] + j) * dims[2] + k
                        for s in range(start[c], start[c + 1]):
                            t = items[s]
                            a = F[t, 0]; b = F[t, 1]; e = F[t, 2]
                            x, y, z = _cpt(p[0], p[1], p[2], V[a, 0], V[a, 1], V[a, 2], V[b, 0], V[b, 1], V[b, 2],
                                           V[e, 0], V[e, 1], V[e, 2])
                            d = (x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2
                            if d < best2:
                                best2 = d; best = np.sqrt(d); bt = t; bx = x; by = y; bz = z
        out_p[qi, 0] = bx; out_p[qi, 1] = by; out_p[qi, 2] = bz; out_t[qi] = bt; out_d[qi] = best
    return out_p, out_t, out_d


@nb.njit(cache=True)
def _ray_tri(o, d, a, b, c):
    e1 = b - a; e2 = c - a
    px = d[1] * e2[2] - d[2] * e2[1]; py = d[2] * e2[0] - d[0] * e2[2]; pz = d[0] * e2[1] - d[1] * e2[0]
    det = e1[0] * px + e1[1] * py + e1[2] * pz
    if abs(det) < 1e-14: return -1.0
    inv = 1.0 / det
    tx = o[0] - a[0]; ty = o[1] - a[1]; tz = o[2] - a[2]
    u = (tx * px + ty * py + tz * pz) * inv
    if u < 0.0 or u > 1.0: return -1.0
    qx = ty * e1[2] - tz * e1[1]; qy = tz * e1[0] - tx * e1[2]; qz = tx * e1[1] - ty * e1[0]
    v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
    if v < 0.0 or u + v > 1.0: return -1.0
    return (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv


@nb.njit(cache=True, parallel=True)
def _raycast(O, D, tmin, tmax, V, F, lo, cell, dims, start, items, count_all):
    """First hit (t, tri) per ray, or with count_all the number of crossings in (tmin, tmax)."""
    n = O.shape[0]
    out_t = np.full(n, np.inf); out_i = np.full(n, -1, np.int64); out_c = np.zeros(n, np.int64)
    for ri in nb.prange(n):
        o = O[ri]; d = D[ri]
        # clip to the grid box
        t0 = tmin; t1 = tmax
        ok = True
        for ax in range(3):
            glo = lo[ax]; ghi = lo[ax] + dims[ax] * cell
            if abs(d[ax]) < 1e-15:
                if o[ax] < glo or o[ax] > ghi: ok = False
            else:
                ta = (glo - o[ax]) / d[ax]; tb = (ghi - o[ax]) / d[ax]
                if ta > tb: ta, tb = tb, ta
                t0 = max(t0, ta); t1 = min(t1, tb)
        if not ok or t0 > t1: continue
        p = o + d * (t0 + 1e-12)
        ix = min(dims[0] - 1, max(0, int((p[0] - lo[0]) / cell)))
        iy = min(dims[1] - 1, max(0, int((p[1] - lo[1]) / cell)))
        iz = min(dims[2] - 1, max(0, int((p[2] - lo[2]) / cell)))
        step = np.zeros(3, np.int64); tnext = np.full(3, np.inf); tdelta = np.full(3, np.inf)
        idx = np.array([ix, iy, iz])
        for ax in range(3):
            if d[ax] > 0:
                step[ax] = 1; tnext[ax] = (lo[ax] + (idx[ax] + 1) * cell - o[ax]) / d[ax]; tdelta[ax] = cell / d[ax]
            elif d[ax] < 0:
                step[ax] = -1; tnext[ax] = (lo[ax] + idx[ax] * cell - o[ax]) / d[ax]; tdelta[ax] = -cell / d[ax]
        best = np.inf; bi = -1; cnt = 0
        while True:
            c = (idx[0] * dims[1] + idx[1]) * dims[2] + idx[2]
            tcell_exit = min(tnext[0], tnext[1], tnext[2])
            for s in range(start[c], start[c + 1]):
                t = items[s]
                th = _ray_tri(o, d, V[F[t, 0]], V[F[t, 1]], V[F[t, 2]])
                if th > tmin and th < tmax:
                    if count_all:
                        # count a crossing once: only in the cell that contains the hit point
                        hx = int((o[0] + d[0] * th - lo[0]) / cell); hy = int((o[1] + d[1] * th - lo[1]) / cell)
                        hz = int((o[2] + d[2] * th - lo[2]) / cell)
                        if hx == idx[0] and hy == idx[1] and hz == idx[2]:
                            cnt += 1
                    elif th < best:
                        best = th; bi = t
            if not count_all and best <= tcell_exit: break
            ax = 0
            if tnext[1] < tnext[ax]: ax = 1
            if tnext[2] < tnext[ax]: ax = 2
            if tnext[ax] > t1: break
            idx[ax] += step[ax]
            if idx[ax] < 0 or idx[ax] >= dims[ax]: break
            tnext[ax] += tdelta[ax]
        out_t[ri] = best; out_i[ri] = bi; out_c[ri] = cnt
    return out_t, out_i, out_c


class TriGrid:
    def __init__(self, V, F, cell=None, pad=1e-4):
        self.V = np.ascontiguousarray(V, np.float64)
        self.F = np.ascontiguousarray(F, np.int64)
        tri = self.V[self.F]
        tmin = tri.min(1) - pad; tmax = tri.max(1) + pad
        self.lo = tmin.min(0) - 1e-3
        hi = tmax.max(0) + 1e-3
        if cell is None:
            # ~2 triangles per cell on average over the surface
            ext = hi - self.lo
            cell = float(np.sqrt((ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2]) * 2 / max(1, len(F))) * 1.5)
        self.cell = cell
        self.dims = np.maximum(1, np.ceil((hi - self.lo) / cell).astype(np.int64))
        counts = _build_counts(self.lo, 1.0 / cell, self.dims, tmin, tmax)
        self.start = np.cumsum(counts)
        self.items = _build_fill(self.lo, 1.0 / cell, self.dims, tmin, tmax, self.start)
        self.maxr = int(self.dims.max())

    def closest(self, q, maxr=None):
        q = np.ascontiguousarray(np.atleast_2d(q), np.float64)
        return _closest(q, self.V, self.F, self.lo, self.cell, self.dims, self.start, self.items,
                        self.maxr if maxr is None else maxr)

    def raycast(self, O, D, tmin=1e-7, tmax=np.inf):
        O = np.ascontiguousarray(np.atleast_2d(O), np.float64); D = np.ascontiguousarray(np.atleast_2d(D), np.float64)
        t, i, _ = _raycast(O, D, tmin, tmax, self.V, self.F, self.lo, self.cell, self.dims, self.start, self.items, False)
        return t, i

    def crossings(self, O, D, tmin=0.0, tmax=np.inf):
        O = np.ascontiguousarray(np.atleast_2d(O), np.float64); D = np.ascontiguousarray(np.atleast_2d(D), np.float64)
        return _raycast(O, D, tmin, tmax, self.V, self.F, self.lo, self.cell, self.dims, self.start, self.items, True)[2]

    def inside(self, q):
        """Majority vote of ray parity along three skewed directions (closed mesh)."""
        q = np.ascontiguousarray(np.atleast_2d(q), np.float64)
        votes = np.zeros(len(q), np.int64)
        for d in ((0.5773, 0.5774, 0.5775), (-0.3, 0.9, 0.3161), (0.2, -0.3, 0.9327)):
            D = np.tile(np.array(d) / np.linalg.norm(d), (len(q), 1))
            votes += self.crossings(q, D) & 1
        return votes >= 2


def weld(P, T, tol=1e-5):
    key = np.round(P / tol).astype(np.int64)
    uk, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    return P[first], inv[T], inv


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / np.maximum(1e-20, np.linalg.norm(vn, axis=1))[:, None]


@nb.njit(cache=True, parallel=True)
def _shape_operator(V, N, nbr_start, nbr_idx):
    """Per-vertex least-squares shape operator from neighbour normal differences; returns
    (kmin_dir, kmax_dir, kmin, kmax) with |kmin| <= |kmax| (signed curvatures)."""
    n = V.shape[0]
    dmin = np.zeros((n, 3)); dmax = np.zeros((n, 3)); kmn = np.zeros(n); kmx = np.zeros(n)
    for i in nb.prange(n):
        nx = N[i, 0]; ny = N[i, 1]; nz = N[i, 2]
        if abs(nx) < 0.9:
            ax = 0.0; ay = nz; az = -ny
        else:
            ax = -nz; ay = 0.0; az = nx
        L = np.sqrt(ax * ax + ay * ay + az * az); ax /= L; ay /= L; az /= L
        bx = ny * az - nz * ay; by = nz * ax - nx * az; bz = nx * ay - ny * ax
        m00 = 1e-9; m01 = 0.0; m02 = 0.0; m11 = 1e-9; m12 = 0.0; m22 = 1e-9
        r0 = 0.0; r1 = 0.0; r2 = 0.0
        for s in range(nbr_start[i], nbr_start[i + 1]):
            j = nbr_idx[s]
            px = V[j, 0] - V[i, 0]; py = V[j, 1] - V[i, 1]; pz = V[j, 2] - V[i, 2]
            qx = N[j, 0] - nx; qy = N[j, 1] - ny; qz = N[j, 2] - nz
            du = px * ax + py * ay + pz * az; dv = px * bx + py * by + pz * bz
            nu = qx * ax + qy * ay + qz * az; nv = qx * bx + qy * by + qz * bz
            w = 1.0 / np.sqrt(du * du + dv * dv + 1e-12)
            # rows [du, dv, 0] -> nu and [0, du, dv] -> nv, unknowns (a, b, c) of S = [[a, b], [b, c]]
            m00 += w * du * du
            m01 += w * du * dv
            m11 += w * (dv * dv + du * du)
            m12 += w * du * dv
            m22 += w * dv * dv
            r0 += w * du * nu
            r1 += w * (dv * nu + du * nv)
            r2 += w * dv * nv
        # solve the symmetric 3x3 [[m00,m01,m02],[m01,m11,m12],[m02,m12,m22]] x = r by Cramer
        det = (m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02) + m02 * (m01 * m12 - m11 * m02))
        if abs(det) < 1e-30:
            continue
        a = (r0 * (m11 * m22 - m12 * m12) - m01 * (r1 * m22 - m12 * r2) + m02 * (r1 * m12 - m11 * r2)) / det
        b = (m00 * (r1 * m22 - m12 * r2) - r0 * (m01 * m22 - m12 * m02) + m02 * (m01 * r2 - r1 * m02)) / det
        c = (m00 * (m11 * r2 - r1 * m12) - m01 * (m01 * r2 - r1 * m02) + r0 * (m01 * m12 - m11 * m02)) / det
        tr = a + c; df = a - c
        disc = np.sqrt(df * df / 4.0 + b * b)
        l1 = tr / 2.0 + disc; l2 = tr / 2.0 - disc
        if abs(b) > 1e-12:
            e0 = l1 - c; e1 = b
        elif a >= c:
            e0 = 1.0; e1 = 0.0
        else:
            e0 = 0.0; e1 = 1.0
        L = np.sqrt(e0 * e0 + e1 * e1); e0 /= L; e1 /= L
        v1x = e0 * ax + e1 * bx; v1y = e0 * ay + e1 * by; v1z = e0 * az + e1 * bz
        v2x = -e1 * ax + e0 * bx; v2y = -e1 * ay + e0 * by; v2z = -e1 * az + e0 * bz
        if abs(l1) >= abs(l2):
            dmax[i, 0] = v1x; dmax[i, 1] = v1y; dmax[i, 2] = v1z
            dmin[i, 0] = v2x; dmin[i, 1] = v2y; dmin[i, 2] = v2z
            kmx[i] = l1; kmn[i] = l2
        else:
            dmax[i, 0] = v2x; dmax[i, 1] = v2y; dmax[i, 2] = v2z
            dmin[i, 0] = v1x; dmin[i, 1] = v1y; dmin[i, 2] = v1z
            kmx[i] = l2; kmn[i] = l1
    return dmin, dmax, kmn, kmx


def radius_neighbours(V, radius, max_k=64):
    from scipy.spatial import cKDTree
    kd = cKDTree(V)
    dd, ii = kd.query(V, k=max_k, distance_upper_bound=radius, workers=-1)
    ok = np.isfinite(dd) & (ii != np.arange(len(V))[:, None]) & (ii < len(V))
    cnt = ok.sum(1)
    start = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    idx = ii[ok].astype(np.int64)
    return start, idx


def principal_dirs(V, N, radius, max_k=64):
    start, idx = radius_neighbours(V, radius, max_k)
    return _shape_operator(np.ascontiguousarray(V), np.ascontiguousarray(N), start, idx)


@nb.njit(cache=True)
def _zbin_counts(lo, inv, dims, tmin, tmax):
    counts = np.zeros(dims[0] * dims[1] + 1, np.int64)
    for t in range(tmin.shape[0]):
        i0 = max(0, min(dims[0] - 1, int((tmin[t, 0] - lo[0]) * inv))); i1 = max(0, min(dims[0] - 1, int((tmax[t, 0] - lo[0]) * inv)))
        j0 = max(0, min(dims[1] - 1, int((tmin[t, 1] - lo[1]) * inv))); j1 = max(0, min(dims[1] - 1, int((tmax[t, 1] - lo[1]) * inv)))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                counts[i * dims[1] + j + 1] += 1
    return counts


@nb.njit(cache=True)
def _zbin_fill(lo, inv, dims, tmin, tmax, start):
    fill = start[:-1].copy(); items = np.empty(start[-1], np.int64)
    for t in range(tmin.shape[0]):
        i0 = max(0, min(dims[0] - 1, int((tmin[t, 0] - lo[0]) * inv))); i1 = max(0, min(dims[0] - 1, int((tmax[t, 0] - lo[0]) * inv)))
        j0 = max(0, min(dims[1] - 1, int((tmin[t, 1] - lo[1]) * inv))); j1 = max(0, min(dims[1] - 1, int((tmax[t, 1] - lo[1]) * inv)))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                c = i * dims[1] + j
                items[fill[c]] = t; fill[c] += 1
    return items


@nb.njit(cache=True, parallel=True)
def _zparity(q, V, F, lo, cell, dims, start, items):
    n = q.shape[0]
    out = np.zeros(n, np.int64)
    for k in nb.prange(n):
        x = q[k, 0]; y = q[k, 1]; z = q[k, 2]
        i = int((x - lo[0]) / cell); j = int((y - lo[1]) / cell)
        if i < 0 or j < 0 or i >= dims[0] or j >= dims[1]: continue
        c = i * dims[1] + j
        cnt = 0
        for s in range(start[c], start[c + 1]):
            t = items[s]
            a = F[t, 0]; b = F[t, 1]; e = F[t, 2]
            x0 = V[a, 0]; y0 = V[a, 1]; x1 = V[b, 0]; y1 = V[b, 1]; x2 = V[e, 0]; y2 = V[e, 1]
            d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(d) < 1e-20: continue
            l0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / d
            l1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / d
            l2 = 1.0 - l0 - l1
            if l0 < 0.0 or l1 < 0.0 or l2 < 0.0: continue
            zz = l0 * V[a, 2] + l1 * V[b, 2] + l2 * V[e, 2]
            if zz > z: cnt += 1
        out[k] = cnt
    return out


class ZParity:
    """Inside test for a closed mesh by counting crossings of the +z ray, with triangles binned
    by their xy footprint (one 2-D cell lookup per query). Jittered xy makes edge hits vanishingly rare."""
    def __init__(self, V, F, cell=0.004):
        self.V = np.ascontiguousarray(V, np.float64); self.F = np.ascontiguousarray(F, np.int64)
        tri = self.V[self.F]
        tmin = tri.min(1)[:, :2]; tmax = tri.max(1)[:, :2]
        self.lo = tmin.min(0) - 1e-3; hi = tmax.max(0) + 1e-3
        self.cell = cell
        self.dims = np.maximum(1, np.ceil((hi - self.lo) / cell).astype(np.int64))
        counts = _zbin_counts(self.lo, 1.0 / cell, self.dims, tmin, tmax)
        self.start = np.cumsum(counts)
        self.items = _zbin_fill(self.lo, 1.0 / cell, self.dims, tmin, tmax, self.start)

    def inside(self, q):
        q = np.array(np.atleast_2d(q), np.float64)
        votes = np.zeros(len(q), np.int64)
        for jit in ((1.3e-7, 2.9e-7), (-2.1e-7, 0.7e-7), (0.4e-7, -3.3e-7)):
            qq = q.copy(); qq[:, 0] += jit[0]; qq[:, 1] += jit[1]
            votes += _zparity(qq, self.V, self.F, self.lo, self.cell, self.dims, self.start, self.items) & 1
        return votes >= 2


@nb.njit(cache=True)
def closest1(px, py, pz, V, F, lo, cell, dims, start, items, maxr):
    """Single-point closest point on the grid's triangles within maxr cell shells; (x, y, z, tri, dist)."""
    ci = int((px - lo[0]) / cell); cj = int((py - lo[1]) / cell); ck = int((pz - lo[2]) / cell)
    best2 = np.inf; bt = -1; bx = 0.0; by = 0.0; bz = 0.0
    for r in range(0, maxr + 1):
        if bt >= 0:
            m = min(px - (lo[0] + (ci - r + 1) * cell), (lo[0] + (ci + r) * cell) - px,
                    py - (lo[1] + (cj - r + 1) * cell), (lo[1] + (cj + r) * cell) - py,
                    pz - (lo[2] + (ck - r + 1) * cell), (lo[2] + (ck + r) * cell) - pz)
            if m * m >= best2: break
        for i in range(ci - r, ci + r + 1):
            if i < 0 or i >= dims[0]: continue
            for j in range(cj - r, cj + r + 1):
                if j < 0 or j >= dims[1]: continue
                onface = abs(i - ci) == r or abs(j - cj) == r
                kstep = 1 if (onface or r == 0) else 2 * r
                for k in range(ck - r, ck + r + 1, kstep):
                    if k < 0 or k >= dims[2]: continue
                    c = (i * dims[1] + j) * dims[2] + k
                    for s in range(start[c], start[c + 1]):
                        t = items[s]
                        a = F[t, 0]; b = F[t, 1]; e = F[t, 2]
                        x, y, z = _cpt(px, py, pz, V[a, 0], V[a, 1], V[a, 2], V[b, 0], V[b, 1], V[b, 2],
                                       V[e, 0], V[e, 1], V[e, 2])
                        d = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
                        if d < best2:
                            best2 = d; bt = t; bx = x; by = y; bz = z
    return bx, by, bz, bt, np.sqrt(best2)


@nb.njit(cache=True)
def bary(px, py, pz, V, F, t):
    a = F[t, 0]; b = F[t, 1]; c = F[t, 2]
    v0x = V[b, 0] - V[a, 0]; v0y = V[b, 1] - V[a, 1]; v0z = V[b, 2] - V[a, 2]
    v1x = V[c, 0] - V[a, 0]; v1y = V[c, 1] - V[a, 1]; v1z = V[c, 2] - V[a, 2]
    v2x = px - V[a, 0]; v2y = py - V[a, 1]; v2z = pz - V[a, 2]
    d00 = v0x * v0x + v0y * v0y + v0z * v0z; d01 = v0x * v1x + v0y * v1y + v0z * v1z
    d11 = v1x * v1x + v1y * v1y + v1z * v1z; d20 = v2x * v0x + v2y * v0y + v2z * v0z
    d21 = v2x * v1x + v2y * v1y + v2z * v1z
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-30: return 1.0, 0.0, 0.0
    v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    # clamp into the triangle (the point is a closest point, so it is inside up to rounding)
    u = max(0.0, u); v = max(0.0, v); w = max(0.0, w); s = u + v + w
    return u / s, v / s, w / s


# ------------------------------------------------------------------ surface streamlines
@nb.njit(cache=True)
def _field(V, F, X, t, px, py, pz):
    u, v, w = bary(px, py, pz, V, F, t)
    a = F[t, 0]; b = F[t, 1]; c = F[t, 2]
    return (u * X[a, 0] + v * X[b, 0] + w * X[c, 0], u * X[a, 1] + v * X[b, 1] + w * X[c, 1],
            u * X[a, 2] + v * X[b, 2] + w * X[c, 2])


@nb.njit(cache=True, parallel=True)
def _trace_all(P0, T0, V, F, flow, lo, cell, dims, start, items, h, maxlen, maxpts):
    n = P0.shape[0]
    out = np.zeros((n, maxpts, 3)); tri = np.full((n, maxpts), -1, np.int64); cnt = np.zeros(n, np.int64)
    why = np.zeros(n, np.int64)
    for g in nb.prange(n):
        px = P0[g, 0]; py = P0[g, 1]; pz = P0[g, 2]; t = T0[g]
        out[g, 0, 0] = px; out[g, 0, 1] = py; out[g, 0, 2] = pz; tri[g, 0] = t
        k = 1; L = 0.0; pdx = 0.0; pdy = 0.0; pdz = 0.0; ymin = py; reason = 0; coast = 0
        gx = 0.0; gy = 0.0; gz = 0.0
        while k < maxpts:
            fx, fy, fz = _field(V, F, flow, t, px, py, pz)
            fl = np.sqrt(fx * fx + fy * fy + fz * fz)
            ok = fl > 0.2
            if ok:
                fx /= fl; fy /= fl; fz /= fl
                mx = px + 0.5 * h * fx; my = py + 0.5 * h * fy; mz = pz + 0.5 * h * fz
                qx, qy, qz, qt, qd = closest1(mx, my, mz, V, F, lo, cell, dims, start, items, 3)
                if qt < 0 or qd > 0.6 * h:
                    ok = False
                else:
                    gx, gy, gz = _field(V, F, flow, qt, qx, qy, qz)
                    gl = np.sqrt(gx * gx + gy * gy + gz * gz)
                    if gl < 0.2:
                        ok = False
                    else:
                        gx /= gl; gy /= gl; gz /= gl
                        if k > 1 and gx * pdx + gy * pdy + gz * pdz < 0.5:
                            ok = False
            if not ok:
                # degenerate or kinked field (a lock tip, a groove crossing): coast on the last
                # heading for a few steps rather than ending the strand
                if k < 2 or coast >= 8:
                    reason = 3; break
                coast += 1
                gx = pdx; gy = pdy; gz = pdz
            else:
                coast = 0
            nx_ = px + h * gx; ny_ = py + h * gy; nz_ = pz + h * gz
            qx, qy, qz, qt, qd = closest1(nx_, ny_, nz_, V, F, lo, cell, dims, start, items, 3)
            if qt < 0 or qd > 0.6 * h:
                reason = 2; break                 # ran off the surface's edge
            dx = qx - px; dy = qy - py; dz = qz - pz
            dl = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dl < 0.2 * h:
                reason = 4; break
            ndx = dx / dl; ndy = dy / dl; ndz = dz / dl
            if k > 1 and ndx * pdx + ndy * pdy + ndz * pdz < 0.3:
                reason = 3; break
            pdx = ndx; pdy = ndy; pdz = ndz
            px = qx; py = qy; pz = qz; t = qt
            L += dl
            out[g, k, 0] = px; out[g, k, 1] = py; out[g, k, 2] = pz; tri[g, k] = t
            k += 1
            ymin = min(ymin, py)
            if py > ymin + 0.02:
                reason = 5; break
            if L > maxlen:
                reason = 6; break
        k -= coast
        cnt[g] = k; why[g] = reason
    return out, tri, cnt, why


def trace_surface(grid, flow, P0, T0, h=0.003, maxlen=0.6, maxpts=300):
    """Streamlines of the per-vertex field `flow` over `grid`'s mesh from (P0, starting triangle T0).
    Returns (points[n, maxpts, 3], triangles[n, maxpts], count[n], stop reason[n]); reasons:
    2 ran off the mesh's edge, 3 degenerate/kinked field, 4 stalled, 5 climbing back up, 6 length."""
    return _trace_all(np.ascontiguousarray(P0, np.float64), np.ascontiguousarray(T0, np.int64), grid.V, grid.F,
                      np.ascontiguousarray(flow, np.float64), grid.lo, grid.cell, grid.dims, grid.start, grid.items,
                      h, maxlen, maxpts)


def interp_vert(X, F, tri, P, V):
    """Barycentric interpolation of a per-vertex field X at points P lying on triangles `tri`."""
    a, b, c = V[F[tri, 0]], V[F[tri, 1]], V[F[tri, 2]]
    v0 = b - a; v1 = c - a; v2 = P - a
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den; u = 1 - v - w
    if X.ndim == 1: return u * X[F[tri, 0]] + v * X[F[tri, 1]] + w * X[F[tri, 2]]
    return u[:, None] * X[F[tri, 0]] + v[:, None] * X[F[tri, 1]] + w[:, None] * X[F[tri, 2]]


def smooth_polyline(q, sigma):
    """Gaussian smoothing along arc length with the root pinned (blended in over 1.5 sigma)."""
    if len(q) < 4 or sigma <= 0: return q
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    W = np.exp(-0.5 * ((s[:, None] - s[None, :]) / sigma) ** 2)
    W /= W.sum(1)[:, None]
    out = W @ q
    a = np.clip(s / (1.5 * sigma), 0, 1)[:, None]
    return (1 - a) * q + a * out


def sample_roots(V, F, area, spacing, seed=1):
    """Poisson-disk-ish points on a mesh: dense area-uniform candidates, greedily thinned."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    n = int(area.sum() / (spacing ** 2) * 6)
    tri = rng.choice(len(F), n, p=area / area.sum())
    r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    a, b, c = V[F[tri, 0]], V[F[tri, 1]], V[F[tri, 2]]
    P = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    kd = cKDTree(P)
    alive = np.ones(n, bool); keep = []
    for i in range(n):
        if not alive[i]: continue
        keep.append(i)
        for j in kd.query_ball_point(P[i], spacing): alive[j] = False
    keep = np.array(keep)
    return P[keep], tri[keep]


# ------------------------------------------------------------------ GLB reader
import io as _io
import json as _json
import struct as _struct

_CT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8, 5122: np.int16, 5120: np.int8}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class Glb:
    """Minimal GLB reader: JSON + BIN chunk, packed accessors (no sparse) and embedded images.
    mesh(i) returns (positions, normals, uv0, triangles) of mesh i's first primitive."""
    def __init__(self, path):
        d = open(path, "rb").read()
        magic, ver, length = _struct.unpack("<4sII", d[:12])
        assert magic == b"glTF" and ver == 2, (magic, ver)
        jl, jt = _struct.unpack("<I4s", d[12:20])
        assert jt == b"JSON"
        self.json = _json.loads(d[20:20 + jl].decode("utf-8"))
        o = 20 + jl
        bl, bt = _struct.unpack("<I4s", d[o:o + 8])
        assert bt == b"BIN\x00"
        self.bin = d[o + 8:o + 8 + bl]

    def view_bytes(self, vi):
        bv = self.json["bufferViews"][vi]
        o = bv.get("byteOffset", 0)
        return self.bin[o:o + bv["byteLength"]]

    def accessor(self, ai):
        a = self.json["accessors"][ai]
        bv = self.json["bufferViews"][a["bufferView"]]
        assert "byteStride" not in bv or bv["byteStride"] == _NC[a["type"]] * np.dtype(_CT[a["componentType"]]).itemsize
        o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        n = a["count"] * _NC[a["type"]]
        arr = np.frombuffer(self.bin, dtype=_CT[a["componentType"]], count=n, offset=o)
        return arr.reshape(a["count"], _NC[a["type"]]) if _NC[a["type"]] > 1 else arr

    def image(self, ii, mode="RGB"):
        from PIL import Image
        im = self.json["images"][ii]
        return np.asarray(Image.open(_io.BytesIO(self.view_bytes(im["bufferView"]))).convert(mode))

    def mesh(self, mi):
        pr = self.json["meshes"][mi]["primitives"][0]
        at = pr["attributes"]
        return (self.accessor(at["POSITION"]).astype(np.float64), self.accessor(at["NORMAL"]).astype(np.float64),
                self.accessor(at["TEXCOORD_0"]).astype(np.float64), self.accessor(pr["indices"]).reshape(-1, 3).astype(np.int64))

    def mesh_by_material(self, name):
        for mi, m in enumerate(self.json["meshes"]):
            mat = m["primitives"][0].get("material")
            if mat is not None and self.json["materials"][mat].get("name") == name:
                return mi
        raise KeyError("no mesh with material %r" % name)

    def base_color_image(self, mi):
        mat = self.json["materials"][self.json["meshes"][mi]["primitives"][0]["material"]]
        ti = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        return self.image(self.json["textures"][ti]["source"])
