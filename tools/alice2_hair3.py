"""Real hair for the Alice2 doll, v3: a strand groom that FILLS her sculpted hair.

    D:/python314/python.exe -u tools/alice2_hair3.py [--glb <alice2 .glb>] [--out <dir>] [--count 130000]
                                                      [--spacing 0.0032] [--cache <dir>]

Default --glb is the sequinned Alice2 (alice2_hyper3d/model/base_basic_pbr_flat_sparkle.glb); --out
defaults to hair_opus5.5_v3 beside that model folder. ~8 minutes (numpy, scipy, numba, Pillow); with
--cache the volume and the flow (the slow part) are kept between runs. Writes into --out the same
kind of scene as v2 (tools/alice2_hair.py), whose sculpt analysis, scalp, headband, room, flyaways
and fibre it reuses:

  alice2_real_hair.ftsl       the scene: the doll with her sculpted hair skipped, the groom, cameras
  alice2_hair_guides.ftsl     the guides -- curves only, so the groom tool can edit and save them
  alice2_hair_groom.ftsl      three guided fur blocks, the fibre, the scalps, flyaways
  alice2_scalp_{fr,fl,bk}.obj the root surfaces;  alice2_headband.obj;  alice2_room.hdr

WHY v3. v2 designed its own hairstyle (a curtain about the body, locks, waves) and checked it against
the photos; the sculpt (root.1) only lent it a silhouette, and the two drifted apart -- from behind a
step at ear height, a wider and flatter back. v3 does the opposite: the sculpt's hair volume IS the
groom's, and the guides are traced through it.

HOW.
  * The VOLUME: the sculpt's hair shell (closed, 231k triangles) voxelized at 2.5 mm by column
    parity, minus the head and body and the bow modelled into it (its near-black texels): ~2.9 M voxels.
    Where the sculpt and the photos disagree, the photos win: the ledge it leaves over the forehead
    goes (the photos' hair sweeps straight back from a clean hairline), and so does everything over
    and in front of the EARS, which the sculpt covers and the photos show bare, the hair tucked
    behind them -- the scalp there too (no roots on the bare skin in front of an ear).
  * Where hair ENDS: the distance through the volume from the centre part, G (fast sweeping), peaks
    at the end of every lock -- 180-odd TIPS (G within 8 mm of its peak over 3 cm, G > 22 cm).
  * The FLOW: hair as an incompressible flow from the scalp to the tips -- Laplace's equation on the
    voxels, the scalp at 0, the tips at 1, nothing through the air or the rest of the body (CG, ~1000
    iterations). Its streamlines, from evenly spaced scalp roots, never cross, go to every lock in
    proportion to the hair it carries, and stack the way hair does: hair rooted higher runs outside
    hair rooted lower, because the inner streamlines are taken.
  * The GUIDES: streamlines of that flow (RK2, 2 mm steps, sliding along the walls where the sampled
    gradient leans out), stopped at a tip or where the flow stalls; one that reached a tip loses a
    random 0.8-3 cm off its end (so the lock tapers out instead of every guide converging on the
    tip -- a broom); then each leaves its root at a shallow angle instead of the flow's
    perpendicular, is smoothed, and resampled.
"""
import argparse
import os
import shutil
import sys
import time

import numba as nb
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hairgeo as hg            # noqa: E402
import alice2_hair as a2        # noqa: E402  (v2: the sculpt, the scalp, the writers)

DEFAULT_GLB = a2.DEFAULT_GLB
H_VOX = 0.0025                 # voxel size (m)
PART = (np.array([a2.XMID, 1.772, 0.120]), np.array([a2.XMID, 1.782, -0.020]))
TIP_G = 0.22                   # a tip is where G peaks, on hair longer than this (the crown's bumps are not tips)
TIP_TOL = 0.008                # ... found as G within 8 mm of its peak over 3 cm round
END_TRIM = (0.008, 0.030)      # a guide that reaches a tip loses a random 0.8-3 cm off its end: run on into
                               # the tip, a lock's guides converge on its small blob from the whole lock -- a
                               # fan, a broom-like end -- and all end on one surface; trimmed, the lock tapers
NPTS = 56                      # control points per guide
MIN_LEN = 0.08                 # shorter guides are dropped (no short hair on a doll)
EARS = {-1: (-0.013, 1.598, 1.683),   # her right ear: back edge z, lobe y, top y -- measured on the bare head
          1: (-0.016, 1.615, 1.697)}   # her left (the head is tilted: it sits higher). The Taubin displacement
                                       # v2 finds ears by reaches 2.5 cm behind and above them (skull moved too).
EAR_CUT = (0.004, 0.07, 0.003)  # the EARS are bare in the photos, the sculpt covers them: on each ear's side,
                               # nothing may grow or pass forward of the ear's back edge (less 4 mm), from
                               # 7 cm under its lobe (the jaw stays bare to the collar) to 3 mm over its top
def cache_tag():
    """What the cached volume and flow depend on (else they are rebuilt)."""
    return "ears %s cut %.3f %.3f %.3f tips %.3f" % (sorted(EARS.items()), *EAR_CUT, TIP_TOL)


def mirror_ears(g):
    """On a symmetric head (tools/alice2_symmetrize.py's), her left ear is her right one's mirror
    image: EARS[1] from EARS[-1], reflected through the face's mirror plane. The right ear's back
    edge, lobe and top are taken at the ear's own distance from the plane (its vertices there)."""
    import alice2_eyes as E
    P, N, UV, T = g.mesh(g.mesh_by_material("root.4"))
    V, _, _ = hg.weld(P, T)
    nm, dm = E.mirror_plane(V)
    zb, ylo, yhi = EARS[-1]
    ear = (V[:, 1] > ylo) & (V[:, 1] < yhi) & (V[:, 2] > zb) & (V[:, 2] < zb + 0.04) & ((V @ nm) - dm < -0.07)
    x = float(np.median(V[ear, 0]))
    q = np.array([[x, 0.5 * (ylo + yhi), zb], [x, ylo, zb], [x, yhi, zb]])
    r = E.reflect_p(q, nm, dm)
    EARS[1] = (float(r[0, 2]), float(r[1, 1]), float(r[2, 1]))
    log("ears: her left one mirrored from her right (back edge z %.3f, lobe y %.3f, top y %.3f)" % EARS[1])
PREFIX = "a3g_"
T0 = time.time()


def log(m):
    print("[alice2_hair3 %5.1fs] %s" % (time.time() - T0, m), flush=True)


# =================================================================================== 1. the volume
@nb.njit(cache=True)
def _columns(V, F, lo, h, nx, ny, nz, out):
    """Parity voxelization: per (x, y) voxel column the z of every triangle crossing its centre line,
    sorted, and the cells between crossing pairs toggled (overlapping closed parts add up mod 2)."""
    nT = F.shape[0]
    cnt = np.zeros(nx * ny, np.int64)
    for t in range(nT):                                     # pass 1: crossings per column
        a = V[F[t, 0]]; b = V[F[t, 1]]; c = V[F[t, 2]]
        x0 = min(a[0], b[0], c[0]); x1 = max(a[0], b[0], c[0])
        y0 = min(a[1], b[1], c[1]); y1 = max(a[1], b[1], c[1])
        i0 = max(0, int(np.ceil((x0 - lo[0]) / h - 0.5))); i1 = min(nx - 1, int(np.floor((x1 - lo[0]) / h - 0.5)))
        j0 = max(0, int(np.ceil((y0 - lo[1]) / h - 0.5))); j1 = min(ny - 1, int(np.floor((y1 - lo[1]) / h - 0.5)))
        for i in range(i0, i1 + 1):
            px = lo[0] + (i + 0.5) * h + 1.3e-9
            for j in range(j0, j1 + 1):
                py = lo[1] + (j + 0.5) * h + 2.9e-9
                d1 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
                d2 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])
                d3 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])
                if (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0):
                    cnt[i * ny + j] += 1
    start = np.zeros(nx * ny + 1, np.int64)
    for q in range(nx * ny):
        start[q + 1] = start[q] + cnt[q]
    zs = np.zeros(start[nx * ny])
    fill = start[:-1].copy()
    for t in range(nT):                                     # pass 2: their z
        a = V[F[t, 0]]; b = V[F[t, 1]]; c = V[F[t, 2]]
        x0 = min(a[0], b[0], c[0]); x1 = max(a[0], b[0], c[0])
        y0 = min(a[1], b[1], c[1]); y1 = max(a[1], b[1], c[1])
        i0 = max(0, int(np.ceil((x0 - lo[0]) / h - 0.5))); i1 = min(nx - 1, int(np.floor((x1 - lo[0]) / h - 0.5)))
        j0 = max(0, int(np.ceil((y0 - lo[1]) / h - 0.5))); j1 = min(ny - 1, int(np.floor((y1 - lo[1]) / h - 0.5)))
        for i in range(i0, i1 + 1):
            px = lo[0] + (i + 0.5) * h + 1.3e-9
            for j in range(j0, j1 + 1):
                py = lo[1] + (j + 0.5) * h + 2.9e-9
                d1 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
                d2 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])
                d3 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])
                if (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0):
                    s = d1 + d2 + d3
                    if abs(s) < 1e-300:
                        continue
                    zs[fill[i * ny + j]] = (d2 * a[2] + d3 * b[2] + d1 * c[2]) / s
                    fill[i * ny + j] += 1
    for i in range(nx):
        for j in range(ny):
            q = i * ny + j
            m = start[q + 1] - start[q]
            if m < 2:
                continue
            col = np.sort(zs[start[q]:start[q + 1]])
            for p in range(0, m - 1, 2):
                k0 = max(0, int(np.ceil((col[p] - lo[2]) / h - 0.5)))
                k1 = min(nz - 1, int(np.floor((col[p + 1] - lo[2]) / h - 0.5)))
                for k in range(k0, k1 + 1):
                    out[i, j, k] ^= 1


def voxelize(V, F, lo, h, dims):
    out = np.zeros(dims, np.uint8)
    _columns(np.ascontiguousarray(V), np.ascontiguousarray(F), lo, h, dims[0], dims[1], dims[2], out)
    return out.astype(bool)


@nb.njit(cache=True)
def _sweep(G, free, h, nx, ny, nz):
    """Fast sweeping (Zhao 2005) for |grad G| = 1 on the `free` voxels, Godunov upwind update."""
    INF = 1e30
    for it in range(8):
        si = 1 if (it & 1) == 0 else -1
        sj = 1 if (it & 2) == 0 else -1
        sk = 1 if (it & 4) == 0 else -1
        i = 0 if si > 0 else nx - 1
        while 0 <= i < nx:
            j = 0 if sj > 0 else ny - 1
            while 0 <= j < ny:
                k = 0 if sk > 0 else nz - 1
                while 0 <= k < nz:
                    if free[i, j, k] and G[i, j, k] > 0.0:
                        a = INF
                        if i > 0 and free[i - 1, j, k]: a = min(a, G[i - 1, j, k])
                        if i < nx - 1 and free[i + 1, j, k]: a = min(a, G[i + 1, j, k])
                        b = INF
                        if j > 0 and free[i, j - 1, k]: b = min(b, G[i, j - 1, k])
                        if j < ny - 1 and free[i, j + 1, k]: b = min(b, G[i, j + 1, k])
                        c = INF
                        if k > 0 and free[i, j, k - 1]: c = min(c, G[i, j, k - 1])
                        if k < nz - 1 and free[i, j, k + 1]: c = min(c, G[i, j, k + 1])
                        if a > b: a, b = b, a
                        if b > c: b, c = c, b
                        if a > b: a, b = b, a
                        if a < INF:
                            u = a + h
                            if u > b:
                                u = 0.5 * (a + b + np.sqrt(max(0.0, 2.0 * h * h - (a - b) ** 2)))
                                if u > c:
                                    s = a + b + c
                                    u = (s + np.sqrt(max(0.0, s * s - 3.0 * (a * a + b * b + c * c - h * h)))) / 3.0
                            if u < G[i, j, k]:
                                G[i, j, k] = u
                    k += sk
                j += sj
            i += si


def ear_zones():
    """Per ear: (side, lowest y, highest y, back edge z) of the region the hair keeps clear of."""
    behind, below, above = EAR_CUT
    zones = []
    for s, (zb, ylo, yhi) in sorted(EARS.items()):
        zones.append((float(s), ylo - below, yhi + above, zb - behind))
        log("ear, her %s: y %.3f..%.3f, back edge z %.3f -> kept clear: y %.3f..%.3f forward of z %.3f"
            % ("left" if s > 0 else "right", ylo, yhi, zb, *zones[-1][1:]))
    return zones


def in_ear_zone(P, zones):
    P = np.atleast_2d(P)
    m = np.zeros(len(P), bool)
    for s, y0, y1, zb in zones:
        m |= ((P[:, 0] - a2.XMID) * s > 0.02) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 2] > zb)
    return m


def cut_scalp(sc, zones):
    """No roots in front of the ears (the photos: bare skin from the temple down)."""
    keep = ~in_ear_zone(sc.VS[sc.FS].mean(1), zones)
    Fk = sc.FS[keep]
    u = np.unique(Fk.ravel()); rm = -np.ones(len(sc.VS), np.int64); rm[u] = np.arange(len(u))
    log("scalp: %d triangles (%.1f cm^2) in front of the ears cut" % ((~keep).sum(), 1e4 * sc.areaS[~keep].sum()))
    sc.VS, sc.FS, sc.nS, sc.lam, sc.areaS = sc.VS[u], rm[Fk], sc.nS[u], sc.lam[u], sc.areaS[keep]


def build_volume(g, sc, zones):
    parts = {}
    for name in ("root.0", "root.1", "root.3", "root.4"):
        P, N, UV, T = g.mesh(g.mesh_by_material(name))
        V, F, _ = hg.weld(P, T)
        parts[name] = (V, F)
    VH, FH = parts["root.1"]
    lo = VH.min(0) - 0.012
    dims = tuple(int(x) for x in np.ceil((VH.max(0) + 0.012 - lo) / H_VOX))
    inside_hair = voxelize(VH, FH, lo, H_VOX, dims)
    inside_body = np.zeros(dims, bool)
    for n in ("root.0", "root.3", "root.4"):
        inside_body |= voxelize(*parts[n], lo, H_VOX, dims)
    H = inside_hair & ~ndimage.binary_dilation(inside_body, iterations=1)
    # the bow modelled into the hair shell (its near-black texels above y 1.825) is not hair
    mh = g.mesh_by_material("root.1")
    P1, N1, UV1, T1 = g.mesh(mh)
    tex = g.base_color_image(mh).astype(float) / 255.0
    th, tw, _ = tex.shape
    uvc = UV1[T1].mean(1)
    lum = tex[np.clip((uvc[:, 1] * th).astype(int), 0, th - 1), np.clip((uvc[:, 0] * tw).astype(int), 0, tw - 1)] \
        @ np.array([0.2126, 0.7152, 0.0722])
    cc = P1[T1].mean(1)
    bow = cc[(lum < 0.45) & (cc[:, 1] > 1.825)]
    ylev = lo[1] + (np.arange(dims[1]) + 0.5) * H_VOX
    cand = np.nonzero(H & (ylev > 1.82)[None, :, None])
    dq, _ = cKDTree(bow).query(lo + (np.stack(cand, 1) + 0.5) * H_VOX, k=1)
    H[tuple(a[dq < 0.009] for a in cand)] = False
    # the sculpt's hair overhangs the forehead (a ledge in front of the hairline); filled, it made the
    # front-hairline strands bulge forward and turn back -- a stiff visor in the render, where the
    # photos have the hair swept straight back from a clean hairline. So in front, the hair may come
    # no further forward than the scalp's own front edge at that height and side (+4 mm).
    VS = sc.VS
    front = VS[VS[:, 2] > 0.04]
    ci = np.floor((front[:, :2] - lo[:2]) / H_VOX).astype(int)
    zf = np.full(dims[:2], -np.inf)
    np.maximum.at(zf, (ci[:, 0], ci[:, 1]), front[:, 2])
    zf = ndimage.maximum_filter(zf, size=5)                  # +-5 mm round each column
    zlev = lo[2] + (np.arange(dims[2]) + 0.5) * H_VOX
    ledge = H & np.isfinite(zf)[:, :, None] & (zlev[None, None, :] > zf[:, :, None] + 0.004) \
        & (zlev[None, None, :] > 0.04) & (ylev > 1.66)[None, :, None]
    H &= ~ledge
    # the ears: the sculpt's locks over them and in front of them go; what that leaves cut off from the
    # scalp (a lock's lower end under the jaw) goes too -- no hair could reach it
    hi = np.nonzero(H)
    ear = in_ear_zone(lo + (np.stack(hi, 1) + 0.5) * H_VOX, zones)
    H[tuple(a[ear] for a in hi)] = False
    d_body = ndimage.distance_transform_edt(~inside_body, sampling=H_VOX).astype(np.float32)
    hi = np.nonzero(H)
    dS, _ = cKDTree(sc.VS).query(lo + (np.stack(hi, 1) + 0.5) * H_VOX, k=1)
    lab, _ = ndimage.label(H)
    rooted = np.unique(lab[tuple(a[(dS < 0.006) & (d_body[hi] < 0.0065)] for a in hi)])
    orphan = H & ~np.isin(lab, rooted[rooted > 0])
    H &= ~orphan
    del lab, hi, dS
    log("volume: grid %s at %.1f mm, %d hair voxels (%.0f cm^3); taken out: the bow's %d, the forehead ledge's %d, "
        "%d over and in front of the ears, %d cut off by that"
        % (dims, H_VOX * 1e3, H.sum(), H.sum() * H_VOX ** 3 * 1e6, (dq < 0.009).sum(), ledge.sum(), ear.sum(), orphan.sum()))
    # G: the distance through the hair from the centre part
    a, b = PART
    blo = np.maximum(0, np.floor((np.minimum(a, b) - 0.03 - lo) / H_VOX).astype(int))
    bhi = np.minimum(np.array(dims), np.ceil((np.maximum(a, b) + 0.03 - lo) / H_VOX).astype(int))
    sl = tuple(slice(blo[k], bhi[k]) for k in range(3))
    C = lo + (np.indices(tuple(bhi - blo)).reshape(3, -1).T + blo + 0.5) * H_VOX
    s = np.clip(((C - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    dpart = np.linalg.norm(C - (a + s[:, None] * (b - a)), axis=1).reshape(tuple(bhi - blo))
    G = np.full(dims, 1e30)
    G[sl][H[sl] & (dpart < 0.012) & (d_body[sl] < 0.010)] = 0.0
    for _ in range(4):
        _sweep(G, H, H_VOX, *dims)
    log("G: max %.3f m through the hair from the part" % G[H & (G < 1e29)].max())
    return dict(lo=lo, H=H, d_body=d_body, G=G.astype(np.float32))


# =================================================================================== 2. the flow
def find_tips(vol):
    """The lock tips: hair voxels whose G is within TIP_TOL of its peak over 3 cm round, G > TIP_G."""
    H, G = vol["H"], vol["G"]
    Gm = np.where(H & (G < 1e29), G, -1.0).astype(np.float32)
    sink = H & (Gm >= ndimage.maximum_filter(Gm, size=25) - TIP_TOL) & (Gm > TIP_G)
    lab, ntip = ndimage.label(sink)
    return sink, lab, ntip, Gm


def solve_flow(vol, sc):
    lo, H, G, d_body = vol["lo"], vol["H"], vol["G"].astype(np.float64), vol["d_body"]
    dims = H.shape
    idx = np.nonzero(H)
    dS, _ = cKDTree(sc.VS).query(lo + (np.stack(idx, 1) + 0.5) * H_VOX, k=1)
    src = np.zeros(dims, bool)
    src[idx] = (dS < 0.006) & (d_body[idx] < 0.0065)
    sink, _, ntip, _ = find_tips(vol)
    free = H & ~src & ~sink
    nid = -np.ones(dims, np.int64); nid[free] = np.arange(free.sum())
    n = int(free.sum())
    fi = np.nonzero(free); me = nid[fi]
    rows, cols = [], []
    diag = np.zeros(n); rhs = np.zeros(n)
    for ax in range(3):
        for step in (-1, 1):
            nb_ = [fi[0].copy(), fi[1].copy(), fi[2].copy()]
            nb_[ax] = nb_[ax] + step
            ok = (nb_[ax] >= 0) & (nb_[ax] < dims[ax])
            j = tuple(q[ok] for q in nb_); m = me[ok]
            is_free = free[j]; conn = is_free | src[j] | sink[j]
            np.add.at(diag, m[conn], 1.0)
            rows.append(m[is_free]); cols.append(nid[j][is_free])
            np.add.at(rhs, m[sink[j]], 1.0)
    r_ = np.concatenate(rows)
    A = sp.csr_matrix((-np.ones(len(r_)), (r_, np.concatenate(cols))), shape=(n, n)) + sp.diags(diag + (diag == 0))
    x, info = spla.cg(A, rhs, x0=np.clip(G[fi] / 0.6, 0, 1), rtol=1e-7, maxiter=20000, M=sp.diags(1.0 / A.diagonal()))
    P = np.full(dims, np.nan, np.float32)
    P[free] = x; P[src] = 0.0; P[sink] = 1.0
    log("flow: %d scalp voxels -> %d tips; %d unknowns, cg info %d, residual %.1e"
        % (src.sum(), ntip, n, info, np.linalg.norm(A @ x - rhs) / np.linalg.norm(rhs)))
    return P


# =================================================================================== 3. the guides
@nb.njit(cache=True)
def tri1(A, lo, h, x, y, z):
    nx, ny, nz = A.shape
    fx = (x - lo[0]) / h - 0.5; fy = (y - lo[1]) / h - 0.5; fz = (z - lo[2]) / h - 0.5
    i = int(np.floor(fx)); j = int(np.floor(fy)); k = int(np.floor(fz))
    if i < 0 or j < 0 or k < 0 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return 0.0
    a = fx - i; b = fy - j; c = fz - k
    return ((1 - a) * ((1 - b) * ((1 - c) * A[i, j, k] + c * A[i, j, k + 1]) + b * ((1 - c) * A[i, j + 1, k] + c * A[i, j + 1, k + 1]))
            + a * ((1 - b) * ((1 - c) * A[i + 1, j, k] + c * A[i + 1, j, k + 1]) + b * ((1 - c) * A[i + 1, j + 1, k] + c * A[i + 1, j + 1, k + 1])))


@nb.njit(cache=True)
def grad1(A, lo, h, x, y, z):
    e = 0.5 * h
    return np.array([(tri1(A, lo, h, x + e, y, z) - tri1(A, lo, h, x - e, y, z)) / (2 * e),
                     (tri1(A, lo, h, x, y + e, z) - tri1(A, lo, h, x, y - e, z)) / (2 * e),
                     (tri1(A, lo, h, x, y, z + e) - tri1(A, lo, h, x, y, z - e)) / (2 * e)])


@nb.njit(cache=True)
def tri3(F, lo, h, x, y, z):
    """Trilinear sample of a vector grid (nx, ny, nz, 3)."""
    nx, ny, nz, _ = F.shape
    fx = (x - lo[0]) / h - 0.5; fy = (y - lo[1]) / h - 0.5; fz = (z - lo[2]) / h - 0.5
    i = int(np.floor(fx)); j = int(np.floor(fy)); k = int(np.floor(fz))
    out = np.zeros(3)
    if i < 0 or j < 0 or k < 0 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return out
    a = fx - i; b = fy - j; c = fz - k
    for d in range(3):
        out[d] = ((1 - a) * ((1 - b) * ((1 - c) * F[i, j, k, d] + c * F[i, j, k + 1, d]) + b * ((1 - c) * F[i, j + 1, k, d] + c * F[i, j + 1, k + 1, d]))
                  + a * ((1 - b) * ((1 - c) * F[i + 1, j, k, d] + c * F[i + 1, j, k + 1, d]) + b * ((1 - c) * F[i + 1, j + 1, k, d] + c * F[i + 1, j + 1, k + 1, d])))
    return out


@nb.njit(cache=True)
def _unit_grad(GP, lo, h, x, y, z):
    """The flow's direction: the potential's gradient, computed on the grid once and interpolated
    trilinearly -- continuous, where differencing the interpolated potential is piecewise and puts
    a small kink into a streamline at every voxel face."""
    g = tri3(GP, lo, h, x, y, z)
    n = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    if n < 1e-9:
        return g, 0.0
    return g / n, n


@nb.njit(cache=True, parallel=True)
def stream_all(P0, P, GP, dH, lo, h, step, maxpts, pend, wall):
    """Streamlines of grad P (GP) from P0 (RK2). Near a wall (dH < wall) the step loses any outward
    part and leans in, so a hair slides along the surface instead of ending on it. Stops at a tip (P >=
    pend), off the volume, or at maxpts. Returns points, count, reason (0 tip, 1 no flow, 4 off, 6 long)."""
    n = P0.shape[0]
    out = np.zeros((n, maxpts, 3)); cnt = np.zeros(n, np.int64); why = np.zeros(n, np.int64)
    for s in nb.prange(n):
        x = P0[s, 0]; y = P0[s, 1]; z = P0[s, 2]
        out[s, 0, 0] = x; out[s, 0, 1] = y; out[s, 0, 2] = z
        k = 1; reason = 6; outside = 0
        while k < maxpts:
            d1, g1 = _unit_grad(GP, lo, h, x, y, z)
            if g1 == 0.0:
                reason = 1; break
            d2, g2 = _unit_grad(GP, lo, h, x + 0.5 * step * d1[0], y + 0.5 * step * d1[1], z + 0.5 * step * d1[2])
            if g2 == 0.0:
                d2 = d1
            dh = tri1(dH, lo, h, x, y, z)
            if dh < wall:
                gw = grad1(dH, lo, h, x, y, z)
                gl = np.sqrt(gw[0] ** 2 + gw[1] ** 2 + gw[2] ** 2)
                if gl > 1e-9:
                    nw = gw / gl
                    a = d2[0] * nw[0] + d2[1] * nw[1] + d2[2] * nw[2]
                    lean = 0.35 * (wall - dh) / wall
                    if a < lean:
                        d2 = d2 + (lean - a) * nw
                        d2 /= np.sqrt(d2[0] ** 2 + d2[1] ** 2 + d2[2] ** 2)
            x += step * d2[0]; y += step * d2[1]; z += step * d2[2]
            out[s, k, 0] = x; out[s, k, 1] = y; out[s, k, 2] = z
            k += 1
            if tri1(dH, lo, h, x, y, z) <= 0.0:
                outside += 1
                if outside >= 4:
                    k -= 4; reason = 4; break
            else:
                outside = 0
            if tri1(P, lo, h, x, y, z) >= pend:
                reason = 0; break
        cnt[s] = max(k, 1); why[s] = reason
    return out, cnt, why


def hermite(p0, t0, p1, t1, n):
    u = np.linspace(0, 1, n)[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def build_guides(vol, P, sc, spacing, seed=1):
    lo, H = vol["lo"], vol["H"]
    dH = ndimage.distance_transform_edt(H, sampling=H_VOX).astype(np.float32)
    bad = ~H | ~np.isfinite(P)
    _, ind = ndimage.distance_transform_edt(bad, return_indices=True)
    Pf = P[tuple(ind)].astype(np.float32); del ind
    GP = np.stack(np.gradient(Pf, H_VOX), -1).astype(np.float32)
    R, rt = hg.sample_roots(sc.VS, sc.FS, sc.areaS, spacing, seed)
    lamr = hg.interp_vert(sc.lam, sc.FS, rt, R, sc.VS)
    nr = hg.interp_vert(sc.nS, sc.FS, rt, R, sc.VS); nr /= np.linalg.norm(nr, axis=1)[:, None]
    out, cnt, why = stream_all(R + nr * 0.0045, Pf, GP, dH, lo, H_VOX, 0.002, 700, 0.995, 0.003)
    trim = np.random.default_rng(seed + 11).uniform(END_TRIM[0], END_TRIM[1], len(R))
    del GP
    log("streamlines from %d roots at %.1f mm: %d to a tip, %d without flow, %d off the hair, %d too long"
        % (len(R), 1000 * spacing, (why == 0).sum(), (why == 1).sum(), (why == 4).sum(), (why == 6).sum()))
    scalp = hg.TriGrid(sc.VS, sc.FS)
    guides = []
    n_stall = 0
    for i in range(len(R)):
        q = out[i, :cnt[i]]
        if len(q) < 8:
            continue
        # where the flow stalls (a pocket it barely moves through): the hair ends there
        pv = np.array([tri1(Pf, lo, H_VOX, *p) for p in q])
        best = np.maximum.accumulate(pv)
        k_end = len(q)
        for k in range(40, len(q)):
            if best[k] - best[k - 40] < 0.003:
                k_end = k - 20; n_stall += 1; break
        q = q[:k_end]
        if why[i] == 0 and k_end == cnt[i]:
            # it ran into a tip: drop its own random last stretch (END_TRIM), where the lock's guides
            # converge on the tip from the whole lock
            sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
            q = q[sl <= sl[-1] - trim[i]]
        if len(q) < 8:
            continue
        # leave the root at a shallow angle: the flow leaves the scalp straight out; a hair lies over
        # it, so the first stretch -- until the strand is 8 mm up, or 2.5 cm along -- is replaced by a
        # curve from the root along the scalp (tilted 20 degrees up) into the streamline's heading
        _, _, ht = scalp.closest(q, maxr=12)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
        j = int(np.argmax((ht >= 0.008) | (s >= 0.025)))
        j = max(j, 3)
        dj = q[min(j + 1, len(q) - 1)] - q[j - 1]
        ndj = np.linalg.norm(dj)
        if j < len(q) - 2 and ndj > 1e-9:
            dj /= ndj
            tt = dj - (dj @ nr[i]) * nr[i]
            tt = tt / max(1e-9, np.linalg.norm(tt)) if np.linalg.norm(tt) > 1e-6 else dj
            t0 = tt + 0.36 * nr[i]; t0 /= np.linalg.norm(t0)
            L = np.linalg.norm(q[j] - R[i])
            head = hermite(R[i], t0 * L * 1.3, q[j], dj * L * 1.3, max(4, int(L / 0.002)))
            # never under the scalp
            cp, _, dd = scalp.closest(head, maxr=12)
            nn = head - cp
            below = (nn @ nr[i]) < 0.0008
            head[below] = cp[below] + nr[i] * 0.0008
            q = np.concatenate([head[:-1], q[j:]])
        else:
            q = np.concatenate([R[i:i + 1], q])
        q = hg.smooth_polyline(q, 0.007)
        guides.append((q, float(lamr[i])))
    # a doll has no short hair: a guide that ends within MIN_LEN of its root (the flow peters out in
    # the thin layer along the front hairline) would grow a fringe of short wisps there -- without
    # it, the fur rooted there copies the nearest long guide, which runs back over the head
    L = np.array([np.linalg.norm(np.diff(q, axis=0), axis=1).sum() for q, _ in guides])
    keep = L >= MIN_LEN
    guides = [gq for gq, k in zip(guides, keep) if k]
    log("guides: %d (%d ended where the flow stalls, %d shorter than %.0f cm dropped); length median %.0f mm, p10 %.0f, p90 %.0f"
        % (len(guides), n_stall, (~keep).sum(), 100 * MIN_LEN, 1e3 * np.median(L[keep]),
           1e3 * np.percentile(L[keep], 10), 1e3 * np.percentile(L[keep], 90)))
    return guides, dH


def coverage(guides, vol, dH):
    lo, H = vol["lo"], vol["H"]
    allp = np.concatenate([a2.resample(q, max(8, int(np.linalg.norm(np.diff(q, axis=0), axis=1).sum() / 0.002)))
                           for q, _ in guides])
    ijk = np.floor((allp - lo) / H_VOX).astype(int)
    ok = (ijk >= 0).all(1) & (ijk < np.array(H.shape)).all(1)
    occ = np.zeros(H.shape, bool); occ[tuple(ijk[ok].T)] = True
    dist = ndimage.distance_transform_edt(~occ, sampling=H_VOX)
    cov = {r: ((dist <= r) & H).sum() / H.sum() for r in (0.003, 0.005, 0.008)}
    inside = H[tuple(ijk[ok].T)].mean()
    log("coverage: %.0f%% / %.0f%% / %.0f%% of the hair volume within 3 / 5 / 8 mm of a guide; %.1f%% of guide points inside it"
        % (100 * cov[0.003], 100 * cov[0.005], 100 * cov[0.008], 100 * inside))
    return cov, inside


# =================================================================================== 4. the outputs
def write_guides(path, guides):
    with open(path, "w") as f:
        f.write("# Alice2's hair GUIDES (v3) -- GENERATED by tools/alice2_hair3.py: streamlines of a flow through\n"
                "# her sculpted hair's volume. Edit them freely, e.g. in the groom tool (groom.bat in the ftrace\n"
                "# repo); rerun the tool to start over. Curve DEFINITIONS (no material). Each piece of the scalp\n"
                "# grows its own fur from its own set: front-right and front-left of the centre part, and the back.\n\n")
        names = [[], [], []]
        for i, (q, l) in enumerate(guides):
            reg = int(a2.region_of(q[0])[0])
            nm = "%s%s%04d" % (PREFIX, a2.REGIONS[reg], i); names[reg].append(nm)
            f.write('curve "%s" {\n' % nm)
            for p in a2.resample(q, NPTS): f.write("    point %.5f %.5f %.5f\n" % tuple(p))
            f.write("}\n")
        for reg in range(3):
            f.write('\ncurve "alice2_guides_%s" {\n' % a2.REGIONS[reg])
            for nm in names[reg]: f.write('    curve "%s"\n' % nm)
            f.write("}\n")


def write_groom(path, guides, areas, zones, count, radius, clump, clump_size, seed=7, guides_file="alice2_hair_guides.ftsl"):
    with open(path, "w") as f:
        f.write("# Alice2's real hair (v3) -- GENERATED by tools/alice2_hair3.py. The guides are in their own file,\n"
                "# because the groom tool rewrites only files of curves; the scalps, the fibre and the fur are here.\n\n")
        f.write('include "%s"\n' % guides_file)
        f.write('\nmaterial "alice2_scalp_mat" { type diffuse reflect 0.5 }\n')
        for reg in range(3):
            f.write('mesh "alice2_scalp_%s" { file "alice2_scalp_%s.obj"  material alice2_scalp_mat  shape_only yes }\n'
                    % (a2.REGIONS[reg], a2.REGIONS[reg]))
        f.write("\n# synthetic doll fibre: eta 1.55, no cuticle tilt, no medulla; `reflect` calibrated against\n"
                "# the photos (v2, tools/alice2_hair.py)\n" + a2.HAIR_MAT + "\n")
        tot = sum(areas)
        for reg in range(3):
            f.write('\nfur "alice2_hair_%s" {\n' % a2.REGIONS[reg])
            f.write('    on "alice2_scalp_%s"   material alice2_hair\n' % a2.REGIONS[reg])
            f.write('    count %d\n' % int(round(count * areas[reg] / tot)))
            # guide_blend 1: a strand is its nearest guide's shape -- the guides are dense (a few mm
            # apart through the whole volume), so a strand stays within a few mm of one
            f.write('    guides "alice2_guides_%s"   guide_blend 1\n' % a2.REGIONS[reg])
            f.write('    points %d   segments 2   spline centripetal\n' % NPTS)
            f.write('    radius %.6f  radius_tip %.6f\n' % (radius, radius * 0.8))
            f.write('    length_jitter 0.03   jitter 0.0\n')
            f.write('    clump %.2f   clump_size %.4f\n' % (clump, clump_size))
            f.write('    root_offset 0.0005\n')
            f.write('    seed %d\n}\n' % (seed + reg))
        write_flyaways(f, guides, zones)


def write_flyaways(f, guides, zones, n=260, seed=5):
    """Stray fibres (every photo shows plenty), as v2 made them -- rooted with the hair, running with
    an outer-layer guide for a while, then lifting off in a loose arc -- except that v3's guides
    include the face-framing locks, so a fibre that would lift off IN FRONT of the face (forward of
    the ears, above the chin) is not made: from the side it read as a stray over her cheek. Nor one
    that would cross an ear."""
    rng = np.random.default_rng(seed)
    C0 = np.array([0.012, 1.66, 0.02])
    lam = np.array([l for q, l in guides])
    w = np.clip(0.6 - lam, 0.05, 1.0); w /= w.sum()
    f.write("\n# flyaways: stray fibres lifting off the surface\n")
    made = 0
    for _ in range(n * 4):
        if made >= n:
            break
        q = a2.resample(guides[rng.choice(len(guides), p=w)][0], 64)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
        L = s[-1]
        if L < 0.08:
            continue
        q = q + rng.normal(size=3) * 0.002
        s0 = rng.uniform(0.12, 0.7) * L
        A = rng.uniform(0.006, 0.030)
        cc = np.stack([np.full(len(q), C0[0]), np.minimum(q[:, 1], C0[1]), np.full(len(q), C0[2])], 1)
        out = q - cc; out /= np.maximum(1e-9, np.linalg.norm(out, axis=1))[:, None]
        u = np.clip((s - s0) / max(1e-6, L - s0), 0, 1)
        tng = np.gradient(q, axis=0); tng /= np.maximum(1e-9, np.linalg.norm(tng, axis=1))[:, None]
        side = np.cross(tng, out); side /= np.maximum(1e-9, np.linalg.norm(side, axis=1))[:, None]
        wav = rng.uniform(0.002, 0.008) * np.sin(2 * np.pi * s / rng.uniform(0.04, 0.12) + rng.uniform(0, 6.3)) * u
        p = q + out * (A * (u * u * (3 - 2 * u)) ** 1.3)[:, None] + side * wav[:, None]
        p = p[s <= L * rng.uniform(0.75, 1.05)][::3]
        if len(p) < 4:
            continue
        if ((p[:, 2] > 0.0) & (p[:, 1] > 1.48)).any():       # it would lift off in front of the face
            continue
        if in_ear_zone(p, zones).any():                       # ... or over an ear
            continue
        f.write("curve { material alice2_hair  radius 0.00014  radius_tip 0.00009  spline centripetal  segments 2\n")
        for pt in p: f.write("    point %.5f %.5f %.5f\n" % tuple(pt))
        f.write("}\n")
        made += 1


SCENE = a2.SCENE.replace("tools/alice2_hair.py", "tools/alice2_hair3.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glb", default=DEFAULT_GLB)
    ap.add_argument("--out", default=None)
    ap.add_argument("--count", type=int, default=130000)
    ap.add_argument("--radius", type=float, default=0.00024)
    ap.add_argument("--spacing", type=float, default=0.0032,
                    help="root spacing of the guides (m): the hair's cross-section grows ~3.5x from the scalp to "
                         "the lock ends, so they spread apart; 3.2 mm keeps the thin face-framing locks fed")
    ap.add_argument("--clump", type=float, default=0.0,
                    help="fur clumping; 0 -- it pulled strands across between diverging guides (strays)")
    ap.add_argument("--clump-size", type=float, default=0.008)
    ap.add_argument("--cache", default=None, help="keep the volume and the flow here between runs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mirror-ears", action="store_true",
                    help="the head is symmetric (tools/alice2_symmetrize.py): her left ear is her right one mirrored")
    a = ap.parse_args()
    glb_dir = os.path.dirname(os.path.abspath(a.glb))
    out = os.path.abspath(a.out) if a.out else os.path.join(os.path.dirname(glb_dir), "hair_opus5.5_v3")
    os.makedirs(out, exist_ok=True)
    try:
        glb_rel = os.path.relpath(os.path.abspath(a.glb), out).replace("\\", "/")
    except ValueError:
        glb_rel = os.path.abspath(a.glb).replace("\\", "/")
    g = hg.Glb(a.glb)
    if a.mirror_ears:
        mirror_ears(g)
    sc = a2.Sculpt(a.glb)
    gr = a2.Groom(sc)            # (v2's head analysis: the headband's seat)
    zones = ear_zones()
    cut_scalp(sc, zones)
    cfile = os.path.join(a.cache, "alice2_hair3_cache.npz") if a.cache else None
    z = np.load(cfile) if cfile and os.path.exists(cfile) else None
    if z is not None and "tag" in z.files and str(z["tag"]) == cache_tag():
        vol = dict(lo=z["lo"], H=z["H"], d_body=z["d_body"], G=z["G"]); P = z["P"]
        log("volume and flow from the cache " + cfile)
    else:
        vol = build_volume(g, sc, zones)
        P = solve_flow(vol, sc)
        if cfile:
            os.makedirs(a.cache, exist_ok=True)
            np.savez_compressed(cfile, P=P, tag=np.array(cache_tag()), **vol)
    guides, dH = build_guides(vol, P, sc, a.spacing, a.seed)
    coverage(guides, vol, dH)
    areas = [a2.write_scalp(sc, os.path.join(out, "alice2_scalp_%s.obj" % a2.REGIONS[r]), r) for r in range(3)]
    write_guides(os.path.join(out, "alice2_hair_guides.ftsl"), guides)
    write_groom(os.path.join(out, "alice2_hair_groom.ftsl"), guides, areas, zones, a.count, a.radius, a.clump,
                a.clump_size)
    # the band ends where it would come down over an ear (the photos: at the ear's top, tucked into
    # the hair passing over it) with only a shallow dip, so it is seen to reach the ear; it sits on the
    # fur rather than in it (the fur stands a few mm proud of the guides: at v2's 1.5 mm it covered the
    # band's edges, a thin wavy line); and it is as wide as the profile photo's (3.4 cm at this scale
    # against v2's 2.2: a third of the eye-to-ear distance)
    Lb = a2.write_headband(gr, guides, os.path.join(out, "alice2_headband.obj"), width=0.034, lift=0.005,
                           dip=0.004, stop=lambda P: in_ear_zone(P, zones))
    a2.write_room(os.path.join(out, "alice2_room.hdr"))
    with open(os.path.join(out, "alice2_real_hair.ftsl"), "w") as f:
        f.write(SCENE.replace("{GLB_REL}", glb_rel))
    np.savez_compressed(os.path.join(a.cache or out, "alice2_hair3_guides.npz"),
                        n=np.array([len(q) for q, _ in guides]), pts=np.concatenate([q for q, _ in guides]),
                        lam=np.array([l for _, l in guides]))
    log("wrote %s: %d guides, %d strands of radius %.2f mm on %.4f m^2 of scalp, headband %.2f m"
        % (out, len(guides), a.count, 1000 * a.radius, sum(areas), Lb))


if __name__ == "__main__":
    main()
