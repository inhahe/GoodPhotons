"""Alice2's eyes, repainted crisp and made symmetric, in a NEW COPY of the doll's GLB.

    D:/python314/python.exe -u tools/alice2_eyes.py [--glb <in.glb>] [--out <out.glb>] [--check] [--debug <dir>]

WHY. All seven parts of the Hyper3D model share one 2048 x 2048 colour atlas, and the face's share of
it is scattered over a dozen UV islands: an iris gets 8-30 texels across, so it renders as a blurry
dark-navy blob with a smeared highlight, ringed by a blurry brown band. And the model made its two
eyes unequal -- in shape, not only in paint: each is a sculpted eyeball sunk in a sharp-edged socket
under a lid ridge, with a little flap past its outer corner (most of the "lining" is that sculpt),
and her left one is much wider and more tilted than her right. So no one eye fits both sockets:
matching eyes need matching sockets.

HOW.
  * SYMMETRY. Her head is rolled and turned a little: the face's mirror plane is fitted (ICP between
    the face and its reflection). Her right eye's frame is her left eye's, reflected, so a point and
    its mirror image land on the same texel of their halves of the new texture.
  * ONE EYE FOR BOTH. Its outline is the model's two painted eyes averaged as shapes (their signed
    distance fields, her right one reflected onto her left). It is painted once, after the photos --
    a bright blue iris with a dark rim and fine fibres, a big black pupil, a white catch-light, the
    whites, a black lash line with a small upturned wing, a faint lower line and a crease -- and the
    same painting goes into both halves. Only the catch-light moves, so that it sits up and to the
    viewer's left in both eyes, as on the doll.
  * THE SOCKETS. Both sculpted sockets (eyeball, trench, lid ridge, crease, corner flap: wherever the
    face's relief stands out round either eye) are cut out over ONE region, the same on both sides,
    and refilled: a constrained Delaunay triangulation of the hole (its own boundary edges kept, a
    fine grid inside; nothing can fold), at the heights of a thin-plate spline through the face
    round the hole, plus the same gentle eyeball dome under the painted opening. Outside that
    region the face is the model's, vertex for vertex.
  * THE LOOK CARRIED ACROSS. The triangles round each eye get planar UVs into the new texture, as a
    second primitive of the face mesh (material "root.4.eyes", root.4's own with new maps). The
    model's colour, metal/roughness and normal map there are baked across first -- the normal map
    re-expressed in the new UVs' tangent frames, as the renderer builds them -- so the face round
    the eyes looks and shades exactly as before; the refilled skin is the skin colour round it.

--check only moves the eyes into the new texture, unchanged: that copy must render as the original.
"""
import argparse
import copy
import io
import json
import os
import struct
import sys

import numpy as np
from matplotlib.path import Path as MPath
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hairgeo as hg            # noqa: E402

DEFAULT_GLB = "D:/youtube/philosophy/3d objects/alice2/alice2_hyper3d/model/base_basic_pbr_flat_sparkle.glb"
FACE = "root.4"
XMID = 0.016                    # roughly the face's mid-plane (the fitted mirror plane replaces it)
HALF = 0.060                    # half the side of the square each eye's texture covers (m)
RES = 1024                      # texels per eye, each way: the 2 x 1 texture is at the glTF loader's
                                # 2048 cap (gltf.h), and the painting's line widths are tuned to it
EYES0 = (np.array([0.0794, 1.6481, 0.1093]), np.array([-0.0406, 1.6468, 0.1220]))   # her left, her right
PATCH0 = (0.052, 0.040)         # the first patch round an eye, to find its painted eye in (half-axes, m)
SEARCH = ((0.005, 0.0), (0.050, 0.038))   # where a socket's features may be: ellipse centre, half-axes (m)
RELIEF_SIGMA = 0.005            # the face's relief: its height minus a Gaussian smooth this wide (m);
GROOVE_MIN = 0.0006             # a socket's features: grooves sunk by more than this (m, the trench, the crease)
RIDGE_MIN = 0.0010              # ... and ridges standing by more than this (m, the eyeball, the corner flap),
FEATURE_MIN_AREA = 2e-6         # ... no smaller than this (m^2), and within this of each other (m), the
FEATURE_LINK = 0.004            #     gaps between them (the lid ridge) closed up
ZONE_GROW = 0.007               # the refilled region: the two sockets' features, grown by this, made convex
                                # (the corner flap's skirt reaches ~6 mm past its outline)
KEEP_OUT = 0.008                # the refill follows the face's own heights only this far (m) from the features
PATCH_GROW = 0.010              # the retextured region: that, grown by this (m)
CLEAN_REACH = 0.006             # the old eye paint's traces are cleaned this far (m) round the region --
                                # with its spread (3 texels + 2 mm), well inside PATCH_GROW: no seam
NMAP_TILT = 0.15                # ... and so is the old normal map's detail where it tilts more than this (sin)
DOME = 0.002                    # the eyeball's gentle dome under the painted opening (m, at the model's 1.9 m
                                # scale; the sculpted eyeballs stood ~5 mm over the refilled surface)


def log(m):
    print("[alice2_eyes] " + m, flush=True)


# ============================================================================ symmetry
def mirror_plane(V, iters=40):
    """The face's mirror plane (n, d): n.p = d. ICP between the face and its reflection."""
    n, d = np.array([1.0, 0.0, 0.0]), XMID
    kd = cKDTree(V)
    for _ in range(iters):
        R = V - 2.0 * ((V @ n) - d)[:, None] * n
        dist, j = kd.query(R)
        keep = dist < np.percentile(dist, 80)
        p, q = V[keep], V[j[keep]]
        w = q - p
        w = w[np.linalg.norm(w, axis=1) > 0.01]
        w = w / np.linalg.norm(w, axis=1)[:, None]
        w *= np.sign(w @ n)[:, None]
        n2 = w.mean(0); n2 /= np.linalg.norm(n2)
        d2 = float(np.median(0.5 * (p + q) @ n2))
        done = np.linalg.norm(n2 - n) < 1e-7 and abs(d2 - d) < 1e-8
        n, d = n2, d2
        if done:
            break
    return n, d


def reflect_p(X, nm, dm):
    return X - 2.0 * ((X @ nm) - dm)[..., None] * nm


def reflect_v(X, nm):
    return X - 2.0 * (X @ nm)[..., None] * nm


def is_iris(col):
    """The model's painted iris (colours as floats 0-1, last axis RGB): dark (its top is near-black under
    the lid) or blue -- but not the warm brown band over the lid, which is as dark."""
    luma = col @ np.array([0.2126, 0.7152, 0.0722])
    return ((luma < 0.40) & (col[..., 0] - col[..., 2] < 0.06)) | ((col[..., 2] > col[..., 0] + 0.03) & (luma < 0.62))


def is_white(col):
    """The model's painted eye white: pale and greyish."""
    return (col.min(-1) > 0.66) & (col.max(-1) - col.min(-1) < 0.24)


def painted_eye(T, UV, atlas):
    """Per face triangle: is its atlas colour the painted eye's (iris or white)?"""
    H, W, _ = atlas.shape
    uvc = UV[T].mean(1)
    col = atlas[np.clip((uvc[:, 1] * H).astype(int), 0, H - 1), np.clip((uvc[:, 0] * W).astype(int), 0, W - 1)]
    return is_iris(col) | is_white(col)


def eye_frames(P, T, UV, atlas, nm, dm):
    """Her left eye's frame -- its centre and facing averaged with her right eye's reflected -- and her
    right eye's, the exact reflection. x: across, away from the nose; y: up; n: out of the face."""
    cen = P[T].mean(1)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    area = 0.5 * np.linalg.norm(fn, axis=1)
    fn /= np.maximum(1e-12, 2 * area)[:, None]
    eye = painted_eye(T, UV, atlas)
    side = cen @ nm - dm
    cs, ns = [], []
    for s, c0 in ((1, EYES0[0]), (-1, EYES0[1])):
        m = eye & (np.sign(side) == s) & (np.linalg.norm(cen - c0, axis=1) < 0.045)
        c = (cen[m] * area[m, None]).sum(0) / area[m].sum()
        near = np.linalg.norm(cen - c, axis=1) < 0.02
        n = (fn[near] * area[near, None]).sum(0); n /= np.linalg.norm(n)
        cs.append(c); ns.append(n)
    c = 0.5 * (cs[0] + reflect_p(cs[1], nm, dm))
    n = ns[0] + reflect_v(ns[1], nm); n /= np.linalg.norm(n)
    x = nm - (nm @ n) * n; x /= np.linalg.norm(x)
    y = np.cross(n, x)
    left = dict(side=1, c=c, n=n, x=x, y=y)
    right = dict(side=-1, c=reflect_p(c, nm, dm), n=reflect_v(n, nm), x=reflect_v(x, nm), y=reflect_v(y, nm))
    log("the eyes' frame: centre %s, facing %s (the two painted eyes' own facings %.0f degrees apart, "
        "their centres %.1f mm apart once mirrored)"
        % (np.round(c, 4), np.round(n, 3), np.degrees(np.arccos(np.clip(ns[0] @ reflect_v(ns[1], nm), -1, 1))),
           1e3 * np.linalg.norm(cs[0] - reflect_p(cs[1], nm, dm))))
    return left, right


def plane_uv(Pv, f, k):
    """Planar UVs for eye k (0: her left, the texture's left half; 1: her right). Her right eye's frame
    being the left's reflection, a point and its mirror image get the same (u, v) in their halves."""
    q = Pv - f["c"]
    u = 0.5 + (q @ f["x"]) / (2 * HALF)
    v = 0.5 - (q @ f["y"]) / (2 * HALF)
    return np.stack([(u + k) / 2.0, v], 1)


def frame_coords(X, f):
    q = X - f["c"]
    return q @ f["x"], q @ f["y"], q @ f["n"]


def texel_grid(R):
    """Every texel's centre, in texels (x right, y down), and in the eye frame (a, b metres)."""
    yy, xx = np.mgrid[0:R, 0:R] + 0.5
    return xx, yy, (xx / R - 0.5) * 2 * HALF, (0.5 - yy / R) * 2 * HALF


# ============================================================================ the bakes
def raster(Tri_new, W, H):
    """Yield, per triangle, the covered pixels (rows, cols) and their barycentrics."""
    for i, tn in enumerate(Tri_new):
        x0, x1 = int(np.floor(tn[:, 0].min())), int(np.ceil(tn[:, 0].max()))
        y0, y1 = int(np.floor(tn[:, 1].min())), int(np.ceil(tn[:, 1].max()))
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W - 1), min(y1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = tn
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        l1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        l2 = 1 - l0 - l1
        ins = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if ins.any():
            yield i, (ys[ins] - 0.5).astype(int), (xs[ins] - 0.5).astype(int), np.c_[l0[ins], l1[ins], l2[ins]]


def bilinear(src, u, v):
    sh, sw = src.shape[:2]
    fx = u * sw - 0.5; fy = v * sh - 0.5
    ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int)
    ax = (fx - ix)[:, None]; ay = (fy - iy)[:, None]
    ix0 = np.clip(ix, 0, sw - 1); ix1 = np.clip(ix + 1, 0, sw - 1)
    iy0 = np.clip(iy, 0, sh - 1); iy1 = np.clip(iy + 1, 0, sh - 1)
    return ((1 - ax) * (1 - ay) * src[iy0, ix0] + ax * (1 - ay) * src[iy0, ix1]
            + (1 - ax) * ay * src[iy1, ix0] + ax * ay * src[iy1, ix1])


def tangent_frames(Pt, UVt):
    """Per triangle, as the renderer builds it (geometry.h): gn, the tangent (texture-u, Lengyel,
    orthogonalized) and the bitangent's handedness."""
    e1, e2 = Pt[:, 1] - Pt[:, 0], Pt[:, 2] - Pt[:, 0]
    gn = np.cross(e1, e2); gn /= np.maximum(1e-15, np.linalg.norm(gn, axis=1))[:, None]
    du1, dv1 = UVt[:, 1, 0] - UVt[:, 0, 0], UVt[:, 1, 1] - UVt[:, 0, 1]
    du2, dv2 = UVt[:, 2, 0] - UVt[:, 0, 0], UVt[:, 2, 1] - UVt[:, 0, 1]
    det = du1 * dv2 - du2 * dv1
    r = np.where(np.abs(det) > 1e-20, 1.0 / np.where(det == 0, 1, det), 0.0)[:, None]
    Tg = (e1 * dv2[:, None] - e2 * dv1[:, None]) * r
    Bg = (e2 * du1[:, None] - e1 * du2[:, None]) * r
    Tg = Tg - gn * (gn * Tg).sum(1)[:, None]
    Tg /= np.maximum(1e-15, np.linalg.norm(Tg, axis=1))[:, None]
    sign = np.where((np.cross(gn, Tg) * Bg).sum(1) < 0, -1.0, 1.0)
    return gn, Tg, sign


def bake(Tri_new, Tp, P, N, UV, uvn, atlas, mr, nmap, W, H):
    """Into a W x H image, through each patch triangle: the atlas colour and metal/roughness at the old
    UVs, and the old normal map re-expressed in the new UVs' tangent frames (decoded and re-encoded
    as the renderer does: n = 2c - 1, T from the UV gradient, B = cross(N, T) * sign)."""
    col = np.zeros((H, W, 3), np.float32); mro = np.zeros((H, W, 3), np.float32)
    nrm = np.zeros((H, W, 3), np.float32); cov = np.zeros((H, W), bool)
    Pt, UVo, UVn = P[Tp], UV[Tp], uvn[Tp]
    # the glTF loader takes UVs as (u, 1 - v) (gltf.h), and builds its tangent frames from those
    flipv = np.array([1.0, -1.0])
    _, To, so = tangent_frames(Pt, UVo * flipv)
    _, Tn, sn = tangent_frames(Pt, UVn * flipv)
    for i, yy, xx, l in raster(Tri_new, W, H):
        uo = l @ UVo[i]
        col[yy, xx] = bilinear(atlas, uo[:, 0], uo[:, 1])
        mro[yy, xx] = bilinear(mr, uo[:, 0], uo[:, 1])
        Nn = l @ N[Tp[i]]; Nn /= np.maximum(1e-12, np.linalg.norm(Nn, axis=1))[:, None]
        t = bilinear(nmap, uo[:, 0], uo[:, 1]) * 2 - 1
        t /= np.maximum(1e-12, np.linalg.norm(t, axis=1))[:, None]
        Ta = To[i] - Nn * (Nn @ To[i])[:, None]; Ta /= np.maximum(1e-12, np.linalg.norm(Ta, axis=1))[:, None]
        Ba = np.cross(Nn, Ta) * so[i]
        nw = Ta * t[:, :1] + Ba * t[:, 1:2] + Nn * t[:, 2:]
        Tb = Tn[i] - Nn * (Nn @ Tn[i])[:, None]; Tb /= np.maximum(1e-12, np.linalg.norm(Tb, axis=1))[:, None]
        Bb = np.cross(Nn, Tb) * sn[i]
        tn = np.c_[(nw * Tb).sum(1), (nw * Bb).sum(1), (nw * Nn).sum(1)]
        tn /= np.maximum(1e-12, np.linalg.norm(tn, axis=1))[:, None]
        nrm[yy, xx] = 0.5 * (tn + 1.0)
        cov[yy, xx] = True
    # uncovered texels take their nearest covered neighbour's value (nothing dark bleeds in)
    _, ind = ndimage.distance_transform_edt(~cov, return_indices=True)
    return col[ind[0], ind[1]], mro[ind[0], ind[1]], nrm[ind[0], ind[1]], cov


def patch_tris(P, T, f, k, R, region, taken):
    """The face triangles of eye k's patch: facing out, near its plane, wholly inside the eye's
    texture square (so no UV strays into the other eye's half), their centroids in `region` (a texel
    mask). Returns (mask over T, the eye's planar UVs per vertex)."""
    av, bv, _ = frame_coords(P, f)
    square = ((np.abs(av) < HALF) & (np.abs(bv) < HALF))[T].all(1)
    a, b, h = frame_coords(P[T].mean(1), f)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    fn /= np.maximum(1e-12, np.linalg.norm(fn, axis=1))[:, None]
    tx = np.clip(((a / (2 * HALF) + 0.5) * R).astype(int), 0, R - 1)
    ty = np.clip(((0.5 - b / (2 * HALF)) * R).astype(int), 0, R - 1)
    sel = square & (np.abs(h) < 0.03) & (fn @ f["n"] > 0.25) & region[ty, tx] & ~taken
    return sel, plane_uv(P, f, k)


def texel_tris(uvn, Tp, k, R):
    return np.stack([(uvn[Tp][..., 0] * 2 - k) * R, uvn[Tp][..., 1] * R], -1)


# ============================================================================ reading a bake
def analyse(img, cov):
    """Find, in one eye's bake (float RGB, its own front view), the eye OPENING (iris + white, holes
    filled, smoothed) and a circle fitted to the iris's edge where that edge is not the lid's."""
    S = img.shape[0]
    yy, xx = np.mgrid[0:S, 0:S]
    c = np.array([S / 2.0, S / 2.0])
    far = cov & (np.hypot(xx - c[0], yy - c[1]) > 0.33 * S)
    skin = np.median(img[far], axis=0)
    dskin = np.linalg.norm(img - skin, axis=2)
    core = np.hypot(xx - c[0], yy - c[1]) < 0.30 * S

    def biggest(m):
        lab, n = ndimage.label(m)
        if n == 0:
            raise RuntimeError("no eye found in the bake")
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        return lab == 1 + int(np.argmax(sizes))

    # the IRIS, with the highlights it encloses
    irism = biggest(ndimage.binary_opening(is_iris(img) & core, iterations=2))
    irism = ndimage.binary_fill_holes(ndimage.binary_closing(irism, iterations=4))
    # the WHITE: unlike skin, touching the iris
    white = is_white(img) & (dskin > 0.10) & core
    white &= ndimage.binary_dilation(irism, iterations=40)
    opening = ndimage.binary_fill_holes(ndimage.binary_closing(irism | white, iterations=8))
    opening = ndimage.gaussian_filter(opening.astype(np.float32), 5) > 0.5
    # the iris circle: its top is under the lid, so take its width at its widest and sit the
    # circle on its lowest point
    ys, xs = np.nonzero(irism)
    rows = np.unique(ys)
    width = np.array([np.ptp(xs[ys == y]) for y in rows])
    yw = rows[np.argmax(width)]
    xr = xs[ys == yw]
    r = 0.5 * (xr.max() - xr.min())
    cx = 0.5 * (xr.max() + xr.min())
    cy = ys.max() - r
    return dict(opening=opening, circle=(float(cx), float(cy), float(r)))


def sdf(mask):
    """Signed distance (texels): negative inside."""
    return ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)


def extent_mm(mask, R):
    ys, xs = np.nonzero(mask)
    return 1e3 * (np.ptp(xs) + 1) * 2 * HALF / R, 1e3 * (np.ptp(ys) + 1) * 2 * HALF / R


# ============================================================================ the eye's outline
def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def bump(u, a, p):
    """1 at u = a, 0 at u = -1 and 1, smooth: the lid's height along the eye."""
    w = np.where(u <= a, (u - a) / (1 + a), (u - a) / (1 - a))
    return np.clip(1 - w * w, 0, 1) ** p


ASPECT = 1.42                       # an eye's width over its height in the photos


class Almond:
    """The eye opening as two lid curves between two corners, in a frame along the eye: s along the
    corner-to-corner axis, t up (toward the brow), both in texels."""
    def __init__(self, opening, p_top=0.70, p_bot=0.95):
        ys, xs = np.nonzero(opening)
        pts = np.stack([xs, ys], 1).astype(np.float64) + 0.5
        self.m = pts.mean(0)
        w, V = np.linalg.eigh(np.cov((pts - self.m).T))
        ax = V[:, 1] if V[0, 1] >= 0 else -V[:, 1]           # the long axis, pointing +x
        self.ax, self.up = ax, np.array([ax[1], -ax[0]])      # up: toward smaller image y
        s, t = (pts - self.m) @ self.ax, (pts - self.m) @ self.up
        self.s0, self.s1 = s.min() + 2, s.max() - 2
        lo = s < self.s0 + 6; hi = s > self.s1 - 6
        self.t0, self.t1 = t[lo].mean(), t[hi].mean()       # the corners' heights
        # the lid heights over the corner-to-corner baseline, measured in bins along s
        bins = np.linspace(self.s0, self.s1, 41)
        k = np.clip(np.digitize(s, bins) - 1, 0, 39)
        base = self.base(s)
        top = np.array([(t - base)[k == i].max() if (k == i).any() else 0 for i in range(40)])
        bot = np.array([(base - t)[k == i].max() if (k == i).any() else 0 for i in range(40)])
        um = 2 * (0.5 * (bins[1:] + bins[:-1]) - self.s0) / (self.s1 - self.s0) - 1
        self.Ht, self.at = float(top.max()), float(um[np.argmax(top)])
        self.Hb, self.ab = float(bot.max()), float(um[np.argmax(bot)])
        self.at = float(np.clip(self.at, -0.5, 0.5)); self.ab = float(np.clip(self.ab, -0.5, 0.5))
        self.pt, self.pb = p_top, p_bot
        # the photos' shape: no taller than 1/ASPECT of its width
        h = self.Ht + self.Hb
        if h > (self.s1 - self.s0) / ASPECT:
            k = (self.s1 - self.s0) / ASPECT / h
            self.Ht *= k; self.Hb *= k

    def base(self, s):
        f = (s - self.s0) / (self.s1 - self.s0)
        return self.t0 + f * (self.t1 - self.t0)

    def u(self, s):
        return 2 * (s - self.s0) / (self.s1 - self.s0) - 1

    def top(self, s):
        return self.base(s) + self.Ht * bump(np.clip(self.u(s), -1, 1), self.at, self.pt)

    def bot(self, s):
        return self.base(s) - self.Hb * bump(np.clip(self.u(s), -1, 1), self.ab, self.pb)

    def st(self, x, y):
        q = np.stack([x - self.m[0], y - self.m[1]], -1)
        return q @ self.ax, q @ self.up

    def xy(self, s, t):
        return self.m + np.outer(s, self.ax) + np.outer(t, self.up)

    def inside(self, s, t):
        """Anti-aliased inside-ness of the opening (vertical distance to the nearer lid)."""
        inner = (s > self.s0) & (s < self.s1)
        d = np.minimum(self.top(s) - t, t - self.bot(s))
        return np.where(inner, smooth01(d / 1.5 + 0.5), 0.0)

    def mask(self, R):
        xx, yy, _, _ = texel_grid(R)
        s, t = self.st(xx, yy)
        return self.inside(s, t) > 0.5


# ============================================================================ the new eye
# The look, from the photos (p2 best: three-quarter, sharp): a bright blue iris -- darkest under the
# lid, lightest at the bottom, a dark navy rim -- a black pupil ~43% of it, one round white
# catch-light up and to the viewer's left of the pupil; white showing on both sides; the upper lid
# covering the top ~12% of the iris; a black lash line along the upper lid, thin at the inner
# corner and thick at the outer, ending in a short upturned wing; a faint brown line under the eye
# and a thin brown crease above it.
IRIS_TOP = (0.12, 0.31, 0.64)       # sRGB, under the lid
IRIS_BOT = (0.48, 0.74, 0.98)       # at the bottom
IRIS_RIM = (0.05, 0.10, 0.26)       # the limbal ring
PUPIL = (0.015, 0.02, 0.035)
WHITE = (0.955, 0.94, 0.925)
LASH = (0.05, 0.035, 0.035)
LOWER = (0.50, 0.30, 0.24)
CREASE = (0.64, 0.42, 0.34)
CATCH = (0.99, 0.99, 1.0)


def stroke(alpha, pts, half, feather=1.0):
    """Max-composite a variable-width stroke (centre points, half-widths) into `alpha`."""
    H, W = alpha.shape
    for (x, y), h in zip(pts, half):
        if h <= 0.05:
            continue
        x0, x1 = int(max(0, np.floor(x - h - 2))), int(min(W - 1, np.ceil(x + h + 2)))
        y0, y1 = int(max(0, np.floor(y - h - 2))), int(min(H - 1, np.ceil(y + h + 2)))
        if x1 < x0 or y1 < y0:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        d = np.hypot(xx + 0.5 - x, yy + 0.5 - y)
        a = np.clip((h - d) / feather + 0.5, 0, 1)
        alpha[y0:y1 + 1, x0:x1 + 1] = np.maximum(alpha[y0:y1 + 1, x0:x1 + 1], a)


def resample_curve(P, step=0.7):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / step))
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])], 1), si / max(s[-1], 1e-9)


def paint_eye(base, al, iris_c, catch_left, seed=1):
    """Paint the eye over `base` (float RGB, one eye's texture half; its outer corner on the right).
    `iris_c`: the iris centre (texels) whose place along the eye is kept (her gaze). `catch_left`:
    the catch-light toward the texture's left of the pupil (else its right)."""
    S = base.shape[0]
    out = base.copy()
    xx, yy, _, _ = texel_grid(S)
    s, t = al.st(xx, yy)
    Wd = al.s1 - al.s0
    # 1) the eye: white, iris, pupil, catch-light -- shaded by the upper lid
    ins = al.inside(s, t)[..., None]
    Hc = al.top(s) - al.bot(s)
    lidsh = 1 - 0.36 * np.exp(-np.clip(al.top(s) - t, 0, None) / np.maximum(0.10 * np.max(Hc), 1))
    corner = 1 - 0.14 * np.clip(np.abs(al.u(s)), 0, 1) ** 4
    white = np.array(WHITE) * (lidsh * corner)[..., None]
    sc0, _ = al.st(np.array([iris_c[0]]), np.array([iris_c[1]]))
    Hmax = float(np.max(al.top(np.linspace(al.s0, al.s1, 200)) - al.bot(np.linspace(al.s0, al.s1, 200))))
    r = 0.56 * Hmax
    lo_s, hi_s = al.s0 + r + 0.07 * Wd, al.s1 - r - 0.07 * Wd
    sc = float(np.clip(sc0[0], lo_s, hi_s)) if lo_s <= hi_s else 0.5 * (lo_s + hi_s)
    tc = float(al.bot(np.array([sc]))[0]) + 0.985 * r
    rho = np.hypot(s - sc, t - tc) / r
    g = np.clip((tc + r - t) / (2 * r), 0, 1) ** 1.2          # 0 at the iris top, 1 at its bottom
    iris = np.array(IRIS_TOP) * (1 - g)[..., None] + np.array(IRIS_BOT) * g[..., None]
    th = np.arctan2(t - tc, s - sc)
    rng = np.random.default_rng(seed)
    fib = sum(rng.uniform(0.3, 1.0) * np.sin(f * th + rng.uniform(0, 6.3)) for f in (13, 23, 37, 53, 71)) / 3.0
    iris = iris * (1 + 0.07 * fib * smooth01((rho - 0.45) / 0.15))[..., None]
    rim = smooth01((rho - 0.80) / 0.17)[..., None]
    iris = iris * (1 - rim) + np.array(IRIS_RIM) * rim
    iris = iris * (1 - 0.18 * smooth01(1 - np.abs(rho - 0.47) / 0.05))[..., None]   # a fine dark ring round the pupil
    iris = iris * lidsh[..., None]
    a_iris = smooth01((1 - rho) * r / 1.5 + 0.5)[..., None]
    a_pup = smooth01((0.43 - rho) * r / 1.5 + 0.5)[..., None]
    hx, hy, hr = sc + (-0.30 if catch_left else 0.30) * r, tc + 0.33 * r, 0.13 * r
    a_hl = smooth01((hr - np.hypot(s - hx, t - hy)) / 1.5 + 0.5)[..., None]
    eye = white * (1 - a_iris) + iris * a_iris
    eye = eye * (1 - a_pup) + np.array(PUPIL) * a_pup
    eye = eye * (1 - a_hl) + np.array(CATCH) * a_hl
    out = out * (1 - ins) + eye * ins
    # 2) the lines: lash line along the upper lid (inner -> outer, thickening) with an upturned wing,
    #    a faint line under the eye, a thin crease above
    ss = np.linspace(al.s0, al.s1, 400)                        # from the inner corner to the outer
    tau = np.linspace(0, 1, len(ss))
    w_in, w_out = 0.012 * Wd, 0.058 * Wd
    half = 0.5 * (w_in + (w_out - w_in) * tau ** 1.3) * smooth01(tau / 0.10)   # fading in at the inner corner
    pts, f = resample_curve(al.xy(ss, al.top(ss) + half - 2.0))
    lash = np.zeros((S, S), np.float32)
    stroke(lash, pts, np.interp(f, tau, half))
    # the wing: from the outer corner, out along the eye's axis and up 28 degrees, tapering
    so = al.s1
    to = float(al.top(np.array([so]))[0]) + 0.7 * half[-1]
    L = 0.24 * Wd
    kk = np.linspace(0, 1, 120)
    ang = np.radians(28) * kk ** 1.2
    ws = so + L * np.cumsum(np.cos(ang)) / len(kk)
    wt = to + L * np.cumsum(np.sin(ang)) / len(kk)
    pts, f = resample_curve(al.xy(ws, wt))
    stroke(lash, pts, half[-1] * 0.95 * (1 - f) ** 0.8)
    lashA = lash[..., None]
    out = out * (1 - lashA) + np.array(LASH) * lashA
    lower = np.zeros((S, S), np.float32)
    pts, f = resample_curve(al.xy(ss, al.bot(ss) - 2.5))
    stroke(lower, pts, 2.3 * smooth01((f - 0.05) / 0.12) * smooth01((0.95 - f) / 0.15))
    lowerA = (0.62 * lower)[..., None]
    out = out * (1 - lowerA) + np.array(LOWER) * lowerA
    crease = np.zeros((S, S), np.float32)
    pts, f = resample_curve(al.xy(ss, al.top(ss) + w_out + 0.24 * Hmax))
    stroke(crease, pts, 2.6 * smooth01((f - 0.12) / 0.15) * smooth01((1.0 - f) / 0.12))
    creaseA = (0.5 * crease)[..., None]
    out = out * (1 - creaseA) + np.array(CREASE) * creaseA
    info = dict(r=r, inside=ins[..., 0])
    return np.clip(out, 0, 1), info


def broad_blur(x, sigma, q=8):
    """A Gaussian blur this wide (texels; x: H x W x C), done at 1/q resolution -- the same field,
    without a thousand-tap kernel over every texel."""
    H, W, C = x.shape
    lo = x.reshape(H // q, q, W // q, q, C).mean((1, 3))
    lo = ndimage.gaussian_filter(lo, (sigma / q, sigma / q, 0))
    return ndimage.zoom(lo, (q, q, 1), order=1, grid_mode=True, mode="nearest")


def skin_fill(img, cov, zone, R):
    """The refilled region's colour: the skin round it (the model's skin is one flat colour, so this
    is close to its median, with any gentle gradient across the eye kept), blended in over 2 mm
    outside the region; anything off-skin within CLEAN_REACH outside it (the old eye paint's last
    traces: its brown band reached past the sculpt) cleaned to skin too. Returns (the new colour,
    the fill's weight per texel: 1 over the region)."""
    px_per_m = R / (2 * HALF)
    dist = ndimage.distance_transform_edt(~zone)
    ring = cov & (dist > 0.002 * px_per_m) & (dist < 0.012 * px_per_m)
    skin = np.median(img[ring], axis=0)
    good = cov & (dist > 0.0015 * px_per_m) & (np.linalg.norm(img - skin, axis=2) < 0.035)
    sig = 0.015 * px_per_m                                     # only the skin's broad shading, no streaks
    num = broad_blur(img * good[..., None], sig)
    den = broad_blur(good[..., None].astype(np.float32), sig)
    wl = np.clip(den / 0.05, 0, 1)
    fill = wl * num / np.maximum(den, 1e-6) + (1 - wl) * skin
    w = np.clip(1 - dist / (0.002 * px_per_m), 0, 1)
    off = (dist < CLEAN_REACH * px_per_m) & (np.linalg.norm(img - skin, axis=2) > 0.05)
    off = ndimage.binary_dilation(off, iterations=3)
    w = np.maximum(w, np.clip(ndimage.gaussian_filter(off.astype(np.float32), 3) * 2, 0, 1))[..., None]
    sd = np.linalg.norm(img[ring] - skin, axis=1)
    log("  skin round the region %s (spread %.3f), %d texels of old paint cleaned outside it"
        % (np.round(skin, 3), float(np.median(sd)), int((off & (dist > 0)).sum())))
    return img * (1 - w) + fill * w, w[..., 0]


def flatten_nmap(n_img, zone, R):
    """The old normal map's detail round the eyes was the old sculpt's and the old lid line's: over the
    region and wherever it tilts by more than NMAP_TILT within CLEAN_REACH of it, it fades to flat
    (over 2 mm) -- in each texel's own tangent frame, so it is right on the face's own triangles too."""
    px_per_m = R / (2 * HALF)
    tn = n_img * 2 - 1
    dist = ndimage.distance_transform_edt(~zone)
    strong = (np.linalg.norm(tn[..., :2], axis=2) > NMAP_TILT) & (dist < CLEAN_REACH * px_per_m)
    m = zone | ndimage.binary_dilation(strong, iterations=4)
    w = np.clip(1 - ndimage.distance_transform_edt(~m) / (0.002 * px_per_m), 0, 1)[..., None]
    tn = tn * (1 - w) + np.array([0.0, 0.0, 1.0]) * w
    tn /= np.maximum(1e-9, np.linalg.norm(tn, axis=2))[..., None]
    log("  normal map flattened over %d texels outside the region (its old detail)" % int((m & ~zone).sum()))
    return 0.5 * (tn + 1)


# ============================================================================ the sockets
def heightmap(V, Tw, f, R):
    """The face's front-most height (m, along the frame's n) over the eye's texture square, one value
    per texel (-inf where there is no face)."""
    a, b, h = frame_coords(V, f)
    ca, cb, ch = a[Tw].mean(1), b[Tw].mean(1), h[Tw].mean(1)
    sel = (np.abs(ca) < HALF + 0.01) & (np.abs(cb) < HALF + 0.01) & (np.abs(ch) < 0.04)
    px = (a / (2 * HALF) + 0.5) * R
    py = (0.5 - b / (2 * HALF)) * R
    tri = np.stack([px[Tw[sel]], py[Tw[sel]]], -1)
    hs = h[Tw[sel]]
    hz = np.full((R, R), -np.inf)
    for i, yy, xx, l in raster(tri, R, R):
        hv = l @ hs[i]
        better = hv > hz[yy, xx]
        hz[yy[better], xx[better]] = hv[better]
    return hz


def relief(hz, R):
    """The face's relief (m): its height minus a Gaussian smooth of it (RELIEF_SIGMA). Returns
    (relief, features: grooves deeper than GROOVE_MIN, ridges higher than RIDGE_MIN)."""
    sig = RELIEF_SIGMA * R / (2 * HALF)
    m = np.isfinite(hz)
    hf = np.where(m, hz, 0.0)
    sm = ndimage.gaussian_filter(hf, sig) / np.maximum(ndimage.gaussian_filter(m.astype(np.float64), sig), 1e-12)
    rel = np.where(m, hz - sm, 0.0)
    return rel, m & ((rel < -GROOVE_MIN) | (rel > RIDGE_MIN))


def socket_zone(hzs, opening, R):
    """ONE region for both eyes (texel mask, convex): the sculpted features round either eye (its
    relief's grooves and ridges, near the painted opening and within FEATURE_LINK of one another), the
    gaps between them closed and what they enclose filled, grown by ZONE_GROW. Also returns the
    features so closed and filled (the `core`)."""
    px_per_m = R / (2 * HALF)
    xx, yy, a, b = texel_grid(R)
    (ca, cb), (ra, rb) = SEARCH
    search = ((a - ca) / ra) ** 2 + ((b - cb) / rb) ** 2 < 1
    feat = np.zeros((R, R), bool)
    for hz in hzs:
        feat |= relief(hz, R)[1]
    feat &= search
    lab, n = ndimage.label(feat)
    sizes = ndimage.sum(feat, lab, np.arange(1, n + 1))
    feat &= np.isin(lab, 1 + np.nonzero(sizes >= FEATURE_MIN_AREA * px_per_m ** 2)[0])
    feat |= opening
    near = ndimage.distance_transform_edt(~feat) <= FEATURE_LINK * px_per_m
    lab, _ = ndimage.label(near)
    keep = np.unique(lab[opening]); keep = keep[keep > 0]
    near &= np.isin(lab, keep)
    core = ndimage.binary_fill_holes(ndimage.distance_transform_edt(near) > FEATURE_LINK * px_per_m)
    core |= feat & near
    grown = ndimage.distance_transform_edt(~core) <= ZONE_GROW * px_per_m
    ys, xs = np.nonzero(grown)
    pts = np.c_[xs + 0.5, ys + 0.5]
    hull = MPath(pts[ConvexHull(pts).vertices])
    zone = hull.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(R, R)
    return zone, core


def boundary_edges(Tr):
    e = np.concatenate([Tr[:, [0, 1]], Tr[:, [1, 2]], Tr[:, [2, 0]]])
    key = np.sort(e, 1)
    uk, cnt = np.unique(key, axis=0, return_counts=True)
    return e, {tuple(k) for k in uk[cnt == 1]}


def boundary_loop(Tr):
    """The longest closed boundary loop of a triangle set (vertex ids, in order), and how many
    boundary edges the set has in all (more than the loop's: it has holes)."""
    e, bset = boundary_edges(Tr)
    # keep the triangles' own orientation, so the loop runs consistently
    nxt = {}
    for u, v in e:
        if (min(u, v), max(u, v)) in bset:
            nxt.setdefault(int(u), []).append(int(v))
    loops, seen = [], set()
    for s0 in list(nxt):
        if s0 in seen:
            continue
        loop, cur = [s0], s0
        seen.add(s0)
        while True:
            cand = [w for w in nxt.get(cur, []) if w not in seen or (w == s0 and len(loop) > 2)]
            if not cand:
                break
            w = cand[0]
            if w == s0:
                break
            loop.append(w); seen.add(w); cur = w
        if s0 not in nxt.get(cur, []):
            raise RuntimeError("a boundary chain of the cut does not close (a bowtie vertex or mixed winding)")
        loops.append(loop)
    return np.array(max(loops, key=len)), len(bset)


def orient2d(p, q, r):
    """Twice the signed area of triangle p q r (> 0: counter-clockwise)."""
    return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])


def fill_polygon(P2, step, seed=7):
    """Triangulate the polygon P2 (n x 2, counter-clockwise; the hole's boundary, which must stay
    exactly as it is, edge for edge) with interior points on a hexagonal grid `step` apart.
    Delaunay of all the points; then each boundary edge Delaunay left out (where the boundary is
    concave) is recovered by flipping the edges that cross it (Sloan 1993); then the triangles
    outside are dropped. No triangle can fold. Returns (the interior points, the triangles indexing
    [boundary | interior points], counter-clockwise, how many boundary edges had to be recovered)."""
    from scipy.spatial import Delaunay
    n = len(P2)
    poly = MPath(P2)
    lo, hi = P2.min(0), P2.max(0)
    rows = []
    for j, y in enumerate(np.arange(lo[1] + 0.5 * step, hi[1], step * np.sqrt(3) / 2)):
        xs = np.arange(lo[0] + (0.5 * step if j % 2 else 0.0), hi[0], step)
        rows.append(np.c_[xs, np.full(len(xs), y)])
    G = np.concatenate(rows)
    G += np.random.default_rng(seed).uniform(-0.02, 0.02, G.shape) * step   # no exactly cocircular sets
    G = G[poly.contains_points(G)]
    # keep them off the boundary: no point within 0.6 step of any of its segments
    A, B = P2, np.roll(P2, -1, axis=0)
    d = np.full(len(G), np.inf)
    for i in range(0, n, 64):
        a_, b_ = A[i:i + 64], B[i:i + 64]
        ab = b_ - a_
        t = np.clip(((G[:, None] - a_[None]) * ab[None]).sum(2) / np.maximum((ab * ab).sum(1), 1e-30)[None], 0, 1)
        d = np.minimum(d, np.linalg.norm(G[:, None] - (a_[None] + t[..., None] * ab[None]), axis=2).min(1))
    G = G[d > 0.6 * step]
    X = np.r_[P2, G]
    tris = Delaunay(X).simplices.copy()
    neg = orient2d(X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]) < 0
    tris[neg] = tris[neg][:, ::-1]
    tris = [list(map(int, t)) for t in tris]
    emap = {}                                                   # edge (min, max) -> triangle ids

    def add(ti):
        t = tris[ti]
        for k in range(3):
            emap.setdefault((min(t[k], t[(k + 1) % 3]), max(t[k], t[(k + 1) % 3])), set()).add(ti)

    def drop(ti):
        t = tris[ti]
        for k in range(3):
            emap[(min(t[k], t[(k + 1) % 3]), max(t[k], t[(k + 1) % 3]))].discard(ti)
    for ti in range(len(tris)):
        add(ti)

    def crosses(e, p, q):
        a_, b_ = X[e[0]], X[e[1]]
        if len({e[0], e[1], p, q}) < 4:
            return False
        return (orient2d(X[p], X[q], a_) * orient2d(X[p], X[q], b_) < 0) and \
               (orient2d(a_, b_, X[p]) * orient2d(a_, b_, X[q]) < 0)
    recovered = 0
    for i in range(n):
        p, q = i, (i + 1) % n
        if emap.get((min(p, q), max(p, q))):
            continue
        queue = [e for e, ts in emap.items() if ts and crosses(e, p, q)]
        guard = 0
        while queue:
            guard += 1
            if guard > 100000:
                raise RuntimeError("boundary edge %d-%d could not be recovered" % (p, q))
            e = queue.pop(0)
            ts = list(emap.get(e, ()))
            if len(ts) != 2:
                continue
            t1, t2 = tris[ts[0]], tris[ts[1]]
            c = [v for v in t1 if v not in e][0]
            dd = [v for v in t2 if v not in e][0]
            a_, b_ = e
            # convex quad: a and b on opposite sides of c-d
            if orient2d(X[c], X[dd], X[a_]) * orient2d(X[c], X[dd], X[b_]) >= 0:
                queue.append(e)
                continue
            drop(ts[0]); drop(ts[1])
            n1, n2 = [a_, c, dd], [b_, dd, c]
            for nt in (n1, n2):
                if orient2d(X[nt[0]], X[nt[1]], X[nt[2]]) < 0:
                    nt.reverse()
            tris[ts[0]], tris[ts[1]] = n1, n2
            add(ts[0]); add(ts[1])
            ne = (min(c, dd), max(c, dd))
            if crosses(ne, p, q):
                queue.append(ne)
        recovered += 1
    T2 = np.array(tris, np.int64)
    cen = X[T2].mean(1)
    T2 = T2[poly.contains_points(cen)]
    # checks: the boundary is exactly the polygon's edges, and nothing folds
    _, bset = boundary_edges(T2)
    want = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    if bset != want:
        raise RuntimeError("refill boundary differs from the hole's: %d edges vs %d" % (len(bset), len(want)))
    if (orient2d(X[T2[:, 0]], X[T2[:, 1]], X[T2[:, 2]]) <= 0).any():
        raise RuntimeError("refill has a folded triangle")
    return G, T2, recovered


def socket_refill(V, Tw, f, zone, core, R, al, step=0.0012, ring=0.012):
    """Cut the face out of `zone` (every welded triangle wholly inside it, and anything its outer
    boundary encloses) and fill the hole (`fill_polygon`: its exact boundary, a fine grid inside).
    Heights: a thin-plate spline through the hole's boundary, the face within `ring` round it and
    the face's own vertices inside it -- of those last two, only the ones clear of the sculpted
    features (`core`) by KEEP_OUT, whose skirts reach past their outlines. Returns the cut, the loop, the new
    points (frame a, b), their triangles (indexing [loop vertex ids as -1-id | new points]), their
    spline heights, the dome's shape over them, and how far the old eyeball stood above the spline
    under the painted opening."""
    from scipy.interpolate import RBFInterpolator
    a, b, h = frame_coords(V, f)
    px = np.clip(((a / (2 * HALF) + 0.5) * R).astype(int), 0, R - 1)
    py = np.clip(((0.5 - b / (2 * HALF)) * R).astype(int), 0, R - 1)
    inside_img = (np.abs(a) < HALF) & (np.abs(b) < HALF) & (np.abs(h) < 0.03)
    inz = inside_img & zone[py, px]
    rem = inz[Tw].all(1)
    loop, _ = boundary_loop(Tw[rem])
    poly = MPath(np.c_[a[loop], b[loop]])
    rem |= poly.contains_points(np.c_[a, b][Tw].mean(1)) & inside_img[Tw].all(1)
    loop, nbe = boundary_loop(Tw[rem])
    if nbe != len(loop):
        raise RuntimeError("the cut has %d boundary edges but its outer loop only %d: refilling that loop "
                           "would leave a hole" % (nbe, len(loop)))
    if np.sum(a[loop] * np.roll(b[loop], -1) - np.roll(a[loop], -1) * b[loop]) < 0:
        loop = loop[::-1]                                    # counter-clockwise, seen from in front
    la, lb = a[loop], b[loop]
    poly = MPath(np.c_[la, lb])
    # the spline: the loop and the kept face within `ring` of it (in the eye's plane)
    kept_v = np.unique(Tw[~rem])
    dl, _ = cKDTree(np.c_[la, lb]).query(np.c_[a[kept_v], b[kept_v]])
    # (where it is plain face, clear of the features: a feature's skirt reaches past its outline)
    clear = ndimage.distance_transform_edt(~core) > KEEP_OUT * R / (2 * HALF)
    near = kept_v[(dl < ring) & inside_img[kept_v] & ~poly.contains_points(np.c_[a[kept_v], b[kept_v]])]
    near = near[clear[py[near], px[near]]]
    # ... and the cut's own vertices that are plain face, as clear of the features
    cut_v = np.setdiff1d(np.unique(Tw[rem]), loop)
    inner = cut_v[clear[py[cut_v], px[cut_v]] & inside_img[cut_v]]
    sv = np.unique(np.r_[loop, near, inner])
    tps = RBFInterpolator(np.c_[a[sv], b[sv]], h[sv], kernel="thin_plate_spline", smoothing=1e-9)
    # the new triangles: the hole's own boundary and a fine grid inside, a constrained Delaunay
    pts, T2, nrec = fill_polygon(np.c_[la, lb], step)
    nl = len(loop)
    tr = np.where(T2 < nl, -1 - loop[np.minimum(T2, nl - 1)], T2 - nl)
    hp = tps(pts)
    # the eyeball dome's shape: a gentle bump inside the painted opening, flat-edged (zero slope at the lids)
    pxp = (pts[:, 0] / (2 * HALF) + 0.5) * R; pyp = (0.5 - pts[:, 1] / (2 * HALF)) * R
    s_, t_ = al.st(pxp, pyp)
    u = np.clip(al.u(s_), -1, 1)
    hc = 0.5 * (al.top(s_) + al.bot(s_)); hh = np.maximum(0.5 * (al.top(s_) - al.bot(s_)), 1e-6)
    v = np.clip((t_ - hc) / hh, -1, 1)
    shape = smooth01(1 - u * u) * smooth01(1 - v * v)
    old_in = inz & (al.inside(*al.st(px + 0.5, py + 0.5)) > 0.5)
    excess = h[old_in] - tps(np.c_[a[old_in], b[old_in]]) if old_in.any() else np.zeros(1)
    return dict(rem=rem, loop=loop, pts=pts, tris=tr, hp=hp, shape=shape, nrec=nrec, nsv=len(sv),
                ninner=len(inner), excess75=float(np.percentile(excess, 75)))


# ============================================================================ the new GLB
class GlbOut:
    """Appends arrays and images to a copy of a GLB's JSON + BIN, then writes the result."""
    def __init__(self, g):
        self.j = copy.deepcopy(g.json)
        self.bin = bytearray(g.bin)

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin += b"\x00"
        bv = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.bin += data
        self.j["bufferViews"].append(bv)
        return len(self.j["bufferViews"]) - 1

    def accessor(self, arr, kind, target):
        arr = np.ascontiguousarray(arr)
        ctype = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125}[arr.dtype]
        acc = {"bufferView": self._view(arr.tobytes(), target), "componentType": ctype,
               "count": int(arr.shape[0]), "type": kind}
        if kind == "VEC3" and arr.dtype == np.float32:
            acc["min"] = [float(v) for v in arr.min(0)]; acc["max"] = [float(v) for v in arr.max(0)]
        if kind == "SCALAR":
            acc["min"] = [int(arr.min())]; acc["max"] = [int(arr.max())]
        self.j["accessors"].append(acc)
        return len(self.j["accessors"]) - 1

    def texture(self, rgb8, sampler=0):
        buf = io.BytesIO()
        Image.fromarray(rgb8).save(buf, format="PNG", optimize=True)
        self.j["images"].append({"bufferView": self._view(buf.getvalue()), "mimeType": "image/png"})
        self.j["textures"].append({"source": len(self.j["images"]) - 1, "sampler": sampler})
        return len(self.j["textures"]) - 1

    def write(self, path):
        while len(self.bin) % 4:
            self.bin += b"\x00"
        self.j["buffers"][0]["byteLength"] = len(self.bin)
        js = json.dumps(self.j, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, total))
            f.write(struct.pack("<I4s", len(js), b"JSON")); f.write(js)
            f.write(struct.pack("<I4s", len(self.bin), b"BIN\x00")); f.write(bytes(self.bin))


def to8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def write_obj(path, Pv, Nv, Tv):
    with open(path, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("".join("v %.6f %.6f %.6f\n" % tuple(p) for p in Pv))
        fo.write("".join("vn %.5f %.5f %.5f\n" % tuple(n) for n in Nv))
        fo.write("".join("f %d//%d %d//%d %d//%d\n" % (i, i, j, j, l, l) for i, j, l in Tv + 1))


def outline(mask):
    return mask & ~ndimage.binary_erosion(mask, iterations=2)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glb", default=DEFAULT_GLB)
    ap.add_argument("--out", default=None, help="default: <glb>_eyes.glb beside it")
    ap.add_argument("--check", action="store_true",
                    help="only move the eyes into the new texture, unchanged (it must render as the original)")
    ap.add_argument("--debug", default=None, help="write the textures, the regions and the face (OBJ) here")
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.glb)[0] + "_eyes.glb"
    R = RES
    px_per_m = R / (2 * HALF)
    mm = lambda texels: 1e3 * texels / px_per_m        # noqa: E731
    change = not a.check                             # (--check: the eyes moved across, unchanged)
    dbg = a.debug
    if dbg:
        os.makedirs(dbg, exist_ok=True)
    g = hg.Glb(a.glb)
    mi = g.mesh_by_material(FACE)
    prim = g.json["meshes"][mi]["primitives"][0]
    mat = g.json["materials"][prim["material"]]
    P, N, UV, T = g.mesh(mi)
    pbr = mat["pbrMetallicRoughness"]
    img_of = lambda ti: g.image(g.json["textures"][ti]["source"]).astype(np.float32) / 255.0   # noqa: E731
    atlas = img_of(pbr["baseColorTexture"]["index"])
    mr = img_of(pbr["metallicRoughnessTexture"]["index"])
    nmap = img_of(mat["normalTexture"]["index"])
    V, Tw, winv = hg.weld(P, T)                      # the face welded across its UV seams
    Nw = np.zeros_like(V); np.add.at(Nw, winv, N)
    Nw /= np.maximum(1e-12, np.linalg.norm(Nw, axis=1))[:, None]
    nm, dm = mirror_plane(V)
    log("the face's mirror plane: normal %s (%.1f degrees off the model's x axis)"
        % (np.round(nm, 3), np.degrees(np.arccos(abs(nm[0])))))
    frames = eye_frames(P, T, UV, atlas, nm, dm)
    names = ("left", "right")

    # 1) each eye as the model has it: a first bake round it, read for its painted opening and iris
    seen = np.zeros(len(T), bool)
    an = []
    _, _, ga, gb = texel_grid(R)
    ell0 = (ga / PATCH0[0]) ** 2 + (gb / PATCH0[1]) ** 2 < 1
    for k, f in enumerate(frames):
        sel, uvn = patch_tris(P, T, f, k, R, ell0, seen)
        seen |= sel
        c0, _, _, cov0 = bake(texel_tris(uvn, T[sel], k, R), T[sel], P, N, UV, uvn, atlas, mr, nmap, R, R)
        an.append(analyse(c0, cov0))
        w_, h_ = extent_mm(an[-1]["opening"], R)
        log("her %s eye as painted: %.1f x %.1f mm" % (names[k], w_, h_))

    # 2) the one eye: the two outlines averaged as shapes, the iris where the two are on average
    opening = 0.5 * (sdf(an[0]["opening"]) + sdf(an[1]["opening"])) < 0
    lab, n = ndimage.label(opening)
    if n > 1:
        opening = lab == 1 + int(np.argmax(ndimage.sum(opening, lab, np.arange(1, n + 1))))
    al = Almond(opening)
    iris_c = 0.5 * (np.array(an[0]["circle"][:2]) + np.array(an[1]["circle"][:2]))
    w_, h_ = extent_mm(al.mask(R), R)
    log("the eye both get: %.1f x %.1f mm, tilted %.1f degrees" % (w_, h_, np.degrees(np.arctan2(-al.ax[1], al.ax[0]))))

    # 3) the sockets: one region for both, from both sides' relief
    hz = [heightmap(V, Tw, f, R) for f in frames]
    zone, core = socket_zone(hz, al.mask(R) | an[0]["opening"] | an[1]["opening"], R)
    zw, zh = extent_mm(zone, R)
    log("the sockets' region (the same on both sides): %.1f x %.1f mm" % (zw, zh))
    fills = []
    bulge = 0.0
    if change:
        for k, f in enumerate(frames):
            fills.append(socket_refill(V, Tw, f, zone, core, R, al))
            fl = fills[-1]
            log("her %s socket: %d triangles cut out; refill %d triangles (%d boundary edges recovered); "
                "spline through %d points (%d of them the face's own, inside the cut); the old eyeball "
                "stood %.1f mm over it" % (names[k], int(fl["rem"].sum()), len(fl["tris"]), fl["nrec"],
                                           fl["nsv"], fl["ninner"], 1e3 * fl["excess75"]))
        bulge = DOME
        log("the dome under both eyes: %.1f mm" % (1e3 * bulge))

    # 4) per eye: the patch (the region grown) and its bake -- the same patch for --check, so the check
    #    covers every triangle the real run moves
    patch_region = ndimage.distance_transform_edt(~zone) <= PATCH_GROW * px_per_m
    if change and (fills[0]["rem"] & fills[1]["rem"]).any():
        raise RuntimeError("the two eyes' cuts overlap")
    taken = np.zeros(len(T), bool)
    baked = []
    for k, f in enumerate(frames):
        sel, uvn = patch_tris(P, T, f, k, R, patch_region, taken)
        rem = fills[k]["rem"] if change else np.zeros(len(T), bool)
        if (rem & taken).any():        # the refill is stitched to the whole cut's boundary
            raise RuntimeError("her %s eye's cut reaches into the other eye's patch" % names[k])
        sel |= rem
        taken |= sel
        baked.append((sel, rem, uvn) + bake(texel_tris(uvn, T[sel], k, R), T[sel], P, N, UV, uvn, atlas, mr, nmap, R, R))
    # metal/roughness: the skin's over the refilled region, the old painted eyes' under the new one --
    # each the median over both sides, so the two eyes shine alike
    mr_skin = np.median(np.concatenate([b_[4][b_[6] & ~zone & patch_region] for b_ in baked]), axis=0)
    mr_eye = np.median(np.concatenate([b_[4][an[k]["opening"]] for k, b_ in enumerate(baked)]), axis=0)

    # 5) per eye: the new look, and the eye primitive's pieces
    col = np.zeros((R, 2 * R, 3), np.float32)
    mrt = np.zeros((R, 2 * R, 3), np.float32)
    nrt = np.zeros((R, 2 * R, 3), np.float32)
    removed = np.zeros(len(T), bool)
    Pe, Ne, Ue, Ie = [], [], [], []
    nbase = 0
    eye_in = al.mask(R)
    for k, f in enumerate(frames):
        sel, rem, uvn, c_img, m_img, n_img, cov = baked[k]
        before = c_img.copy()
        if change:
            n_img = flatten_nmap(n_img, zone, R)
            c_img, wfill = skin_fill(c_img, cov, zone, R)
            m_img = m_img * (1 - wfill[..., None]) + mr_skin * wfill[..., None]
            c_img, info = paint_eye(c_img, al, iris_c, catch_left=(k == 0))
            ein = info["inside"][..., None]
            m_img = m_img * (1 - ein) + mr_eye * ein
            if k == 0:
                log("metal/roughness (B/G): skin %s, eye %s; iris %.1f mm across"
                    % (np.round(mr_skin, 3), np.round(mr_eye, 3), mm(2 * info["r"])))
        # the eye primitive, part 1: the patch's untouched triangles, on their own vertices
        Tk = T[sel & ~rem]
        vk, inv = np.unique(Tk.ravel(), return_inverse=True)
        Pe.append(P[vk]); Ne.append(N[vk]); Ue.append(uvn[vk]); Ie.append(inv.reshape(-1, 3) + nbase)
        nbase += len(vk)
        if change:
            # part 2: the refill -- the hole's boundary (welded vertices, their own normals) + new points
            fl = fills[k]
            loop = fl["loop"]
            hgt = fl["hp"] + bulge * fl["shape"]
            P3 = f["c"] + fl["pts"][:, :1] * f["x"] + fl["pts"][:, 1:2] * f["y"] + hgt[:, None] * f["n"]
            Pf = np.r_[V[loop], P3]
            lidx = {int(v): i for i, v in enumerate(loop)}
            tl = np.array([[lidx[-1 - t] if t < 0 else len(loop) + t for t in tr] for tr in fl["tris"]], np.int64)
            nf = np.cross(Pf[tl[:, 1]] - Pf[tl[:, 0]], Pf[tl[:, 2]] - Pf[tl[:, 0]])
            if (nf @ f["n"]).sum() < 0:                  # facing out of the face, like the rest
                tl = tl[:, ::-1]
            Nf = hg.vertex_normals(Pf, tl)
            Nf[:len(loop)] = Nw[loop]                    # the seam shades like the face round it
            uf = plane_uv(Pf, f, k)
            Pe.append(Pf); Ne.append(Nf); Ue.append(uf); Ie.append(tl + nbase)
            nbase += len(Pf)
            # the normal map: flat on the refill (the old map's detail was the sculpt's, now gone)
            flat = np.zeros((R, R), bool)
            for _, ry, rx, _ in raster(texel_tris(uf, tl, k, R), R, R):
                flat[ry, rx] = True
            n_img[flat] = (0.5, 0.5, 1.0)
            removed |= rem
        col[:, k * R:(k + 1) * R] = c_img
        mrt[:, k * R:(k + 1) * R] = m_img
        nrt[:, k * R:(k + 1) * R] = n_img
        log("her %s eye: %d of the face's triangles into the eye texture (%d cut out)"
            % (names[k], int(sel.sum()), int(rem.sum())))
        if dbg:
            Image.fromarray(to8(before)).save(os.path.join(dbg, "texture_%s_before.png" % names[k]))
            Image.fromarray(to8(c_img)).save(os.path.join(dbg, "texture_%s_after.png" % names[k]))
            rel, _ = relief(hz[k], R)
            ov = np.repeat(np.clip(0.5 + rel / 0.008, 0, 1)[..., None], 3, 2)
            for msk, cc in ((outline(zone), (1, 0, 0)), (outline(patch_region), (1, 0.6, 0)),
                            (outline(core), (1, 1, 0)), (outline(an[k]["opening"]), (0, 0.4, 1)),
                            (outline(eye_in), (0, 1, 0))):
                ov[msk] = cc
            Image.fromarray(to8(ov)).save(os.path.join(dbg, "regions_%s.png" % names[k]))
    if dbg:
        Image.fromarray(to8(mrt)).save(os.path.join(dbg, "metal_rough.png"))
        Image.fromarray(to8(nrt)).save(os.path.join(dbg, "normal.png"))
        Pd = np.concatenate(Pe); Nd = np.concatenate(Ne); Td = np.concatenate(Ie)
        write_obj(os.path.join(dbg, "face_new.obj"), np.r_[P, Pd], np.r_[N, Nd], np.r_[T[~taken], Td + len(P)])
        write_obj(os.path.join(dbg, "face_orig.obj"), P, N, T)

    go = GlbOut(g)
    # the face keeps every triangle but the eyes' ...
    keep = T[~taken].astype(np.uint32).ravel()
    prim0 = go.j["meshes"][mi]["primitives"][0]
    prim0["indices"] = go.accessor(keep, "SCALAR", 34963)
    # ... which become a second primitive: root.4's material with the new maps. Any other map would
    # sample the atlas at the new UVs, so none is carried over; the new textures clamp at their edges
    # (each half's border must not filter into the other half).
    m2 = copy.deepcopy(mat)
    m2["name"] = FACE + ".eyes"
    for slot in ("occlusionTexture", "emissiveTexture", "extensions"):
        if m2.pop(slot, None) is not None:
            log("%s's %s left off the eye material (it would read the atlas at the eye UVs)" % (FACE, slot))
    go.j.setdefault("samplers", []).append({"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071})
    smp = len(go.j["samplers"]) - 1
    m2["pbrMetallicRoughness"]["baseColorTexture"] = {"index": go.texture(to8(col), smp)}
    m2["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": go.texture(to8(mrt), smp)}
    m2["normalTexture"] = dict(mat["normalTexture"], index=go.texture(to8(nrt), smp))
    m2["normalTexture"].pop("texCoord", None)
    go.j["materials"].append(m2)
    prim1 = {"attributes": {"POSITION": go.accessor(np.concatenate(Pe).astype(np.float32), "VEC3", 34962),
                            "NORMAL": go.accessor(np.concatenate(Ne).astype(np.float32), "VEC3", 34962),
                            "TEXCOORD_0": go.accessor(np.concatenate(Ue).astype(np.float32), "VEC2", 34962)},
             "indices": go.accessor(np.concatenate(Ie).astype(np.uint32).ravel(), "SCALAR", 34963),
             "material": len(go.j["materials"]) - 1}
    if "mode" in prim0:
        prim1["mode"] = prim0["mode"]
    go.j["meshes"][mi]["primitives"].append(prim1)
    go.write(out)
    nI = len(np.concatenate(Ie))
    log("wrote %s: the face (%s) %d triangles; its eyes %d (%d of the face's, %d new) in material %s, a %d x %d texture%s"
        % (out, FACE, int((~taken).sum()), nI, int((taken & ~removed).sum()), nI - int((taken & ~removed).sum()),
           m2["name"], 2 * R, R, " -- unchanged (--check)" if a.check else ""))


if __name__ == "__main__":
    main()
