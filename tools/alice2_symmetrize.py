"""Alice2's head made symmetric, in a NEW COPY of the doll's GLB.

    D:/python314/python.exe -u tools/alice2_symmetrize.py [--glb <in.glb>] [--out <out.glb>]

WHY. The Hyper3D head (root.4) is skewed: mirrored through its own best-fit mirror plane, her right
cheek and jaw stand out ~4 mm beyond her left, her left temple and forehead ~4 mm beyond her right,
the chin's low point and the mouth sit off toward her right, and the ears differ. Seen from in front
the face looks lopsided.

WHICH SIDE. Her right -- checked against the photos (d:/youtube/philosophy/images/alice). Each
photo's camera was fitted (solvePnP, the phone's own focal length) to landmarks on the head made
symmetric from either side: her right side fits the profile and three-quarter photos far better
(landmark RMS 13.9 and 19.5 px, against 22.2 and 24.8 for her left), and its outline follows the
profile photo's forehead, nose, lips and chin. Except the MOUTH: her right half of it is wide --
mirrored, the mouth would be 0.61 of the distance between the eyes' centres -- so it is narrowed,
to the width the photos show (MOUTH_WIDTH).

HOW (the plane is the face's fitted mirror plane, alice2_eyes.mirror_plane):
  1. RE-CENTRE THE MIDLINE. The nose, philtrum, lips and chin drift toward her right as they go
     down, up to ~10 mm off the plane. Slice by slice, the profile's crest is found and shifted
     across onto the plane, fading out toward the cheeks (centre_midline). Mirrored off the plane,
     a feature comes out two-lobed: a groove down the nose, a double bow in the lip.
  2. AVERAGE THE TWO SIDES (her right, and her left mirrored, half each) in a band along the plane
     and in the chin. The average of a surface and its mirror image is symmetric, and meets the
     plane square-on (average_shape). Without it the skewed face meets the plane at an angle,
     leaving a crease down the middle. And the jaw is tilted, lower on her right: mirrored, her
     right side gives a cleft chin, her left a pointed one, and their average a round one.
  3. NARROW THE MOUTH. Her right half of it is squeezed toward the middle, the lips' relief carried
     over a smooth copy of the face, the paint moving with the vertices (narrow_mouth). The lips
     keep her right half's paint: closed, as in the photos (her left half's shows teeth).
  4. MIRROR. The head is cut along the plane (triangles across it clipped, new vertices on it), her
     left half dropped, and her right half mirrored into its place: positions and normals
     reflected, winding reversed, the SAME texture coordinates, so the paint is mirrored with the
     shape. The halves share their vertices on the plane, so the seam is watertight (mirror_right).
  5. NORMALS where the shape changed are the new surface's (renormal).
  6. REBAKE the paint where the sides were averaged, on the final head: colour, metal/roughness,
     and the normal map's world normal -- laid into the plane at the plane, then re-expressed in
     the final tangent frames, as the renderer reads it (rebake_averaged).
  7. FLATTEN the normal map under the painted eyebrows, which the model carves in (flatten_brows).
Everything else in the GLB is unchanged. The eyes come after: tools/alice2_eyes.py --glb <out>.
"""
import argparse
import io
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import alice2_eyes as E         # noqa: E402  (the mirror plane, the tangent frames, the GLB writer)
import hairgeo as hg            # noqa: E402

FACE = "root.4"
# Where the two sides are AVERAGED (her right side and her left side's mirror image, half each) --
# the average of a surface and its mirror image is symmetric by itself, and smooth across the plane:
MID_BAND = 0.015          # a band this wide (m) either side of the plane: mirrored as it is, the face
                          # meets the plane at an angle there (it is skewed) -- a crease down the middle
CHIN = ((0.0, -0.098), (0.070, 0.022))    # the chin: an ellipse in the plane's frame (m: centre across
                                          # and up from REF_Y, half-axes). The jaw is tilted, lower on her
                                          # right: mirrored, her right side gives a cleft chin, her left
                                          # a pointed one; the average cancels the tilt
ZONE_FADE = 1.5                           # ... its weight fades to 0 at this many times its size
# The mouth: her right half of it, narrowed toward the middle (shape and paint together):
MOUTH = ((0.0, -0.050), (0.050, 0.016))   # the ellipse round her lips (both halves' corners inside)
MOUTH_WIDTH = 0.50                        # ... squeezed until the mouth is this much of the distance
                                          # between the painted eyes' centres. Projected through cameras
                                          # fitted to the photos, 0.45 matched the blurry front photo and
                                          # was 0.81 of the sharp three-quarter one, where the model as
                                          # made (0.52) matched it
MOUTH_FADE = 1.6                          # ... the squeeze fading out by this many times its size
CENTRE_TOP, CENTRE_BOTTOM = 0.035, -0.130  # the midline features re-centred between these heights (m,
                                          # from REF_Y): from between the eyes down past the chin,
                                          # tapering to nothing over the top 1 cm and the bottom 2 cm
CENTRE_FIT = 0.022                        # ... each slice's crest fitted within this of the plane (m)
CENTRE_FLAT, CENTRE_REACH = 0.030, 0.060  # ... the shift whole within the first of the feature's own
                                          # centre (the nose and lips move as one piece), gone by the
                                          # second (m)
FRONT = (0.02, 0.05)                      # the front of the head: height out of the face (m, from the
                                          # plane's origin) where the shape steps start to apply, and
                                          # where in full -- the back of the head and the nape stay
REF_Y = 1.62                              # the frame's origin: the point of the plane at this height
SNAP = 5e-5                               # vertices this close to the plane (m) are put on it
PLANE_FLAT = 0.004                        # the normal map's normals are laid into the plane within this
BROWS = (0.060, 0.130, 0.110)             # the painted eyebrows: from/to this high over REF_Y and within
                                          # this of the plane (m). The model's normal map carves them in
                                          # (tilt ~0.22 against ~0.06 on the skin, leaning inward at their
                                          # edges); the doll's are flat paint, so it is flattened there


def log(m):
    print("[alice2_symmetrize] " + m, flush=True)


def plane_frame(nm, dm):
    """The plane's frame: its origin (at height REF_Y), `up` in the plane, `fwd` out of the face."""
    fwd = np.array([0.0, 0.0, 1.0]) - nm[2] * nm; fwd /= np.linalg.norm(fwd)
    up = np.cross(fwd, nm); up *= np.sign(up[1])
    c0 = np.array([dm * nm[0], REF_Y, 0.0]); c0 -= ((c0 @ nm) - dm) * nm
    return c0, up, fwd


def ellipse_r(X, nm, dm, c0, up, ell):
    """X's normalised radius in an ellipse of the plane's frame ((centre across, up), (half-axes))."""
    (ca, cu), (ra, ru) = ell
    return np.hypot(((X @ nm) - dm - ca) / ra, ((X - c0) @ up - cu) / ru)


def ellipse_fade(X, nm, dm, c0, up, ell, inner, outer):
    """1 within `inner` times the ellipse `ell`, fading smoothly to 0 at `outer` times it."""
    return E.smooth01((outer - ellipse_r(X, nm, dm, c0, up, ell)) / (outer - inner))


def left_weight(X, nm, dm, c0, up):
    """How much of her left side (mirrored) to use at X: half in the midline band and the chin,
    fading smoothly to nothing outside them."""
    band = E.smooth01((MID_BAND - np.abs((X @ nm) - dm)) / MID_BAND)
    chin = ellipse_fade(X, nm, dm, c0, up, CHIN, 1.0, ZONE_FADE)
    return 0.5 * np.maximum(band, chin)


def midline_offsets(V, Tw, nm, dm):
    """Where the face's midline features actually are, height by height: in each horizontal slice
    of the front of the face (2 mm apart, within CENTRE_FIT of the plane), the across position
    of the profile's crest -- a parabola fitted to the slice's heights out of the face, where it is
    clearly convex (the nose, the lips, the chin). Gaps are interpolated, the curve smoothed, and
    it is held to 0 above CENTRE_TOP (between the eyes the plane is right). Returns (heights up
    from REF_Y, offsets across), both in m."""
    c0, up, fwd = plane_frame(nm, dm)
    a, u, h = (V @ nm) - dm, (V - c0) @ up, (V - c0) @ fwd
    Nv = hg.vertex_normals(V, Tw)
    front = (Nv @ fwd) > 0.3
    us = np.arange(CENTRE_BOTTOM, CENTRE_TOP + 1e-9, 0.002)
    cs = np.full(len(us), np.nan)
    for k, uu in enumerate(us):
        m = front & (np.abs(u - uu) < 0.002) & (np.abs(a) < CENTRE_FIT)
        if m.sum() < 8:
            continue
        coef, *_ = np.linalg.lstsq(np.c_[np.ones(m.sum()), a[m], a[m] ** 2], h[m], rcond=None)
        if coef[2] < -4.0:                                   # convex enough (radius under ~12 cm)
            crest = -coef[1] / (2 * coef[2])
            if abs(crest) < 0.012:
                cs[k] = crest
    ok = np.isfinite(cs)
    cs = np.interp(us, us[ok], cs[ok]) if ok.any() else np.zeros(len(us))
    cs = ndimage.gaussian_filter1d(cs, 2.0, mode="nearest")
    cs *= E.smooth01((CENTRE_TOP - us) / 0.010)              # nothing at the top (between the eyes) ...
    cs *= E.smooth01((us - CENTRE_BOTTOM) / 0.020)           # ... nor at the bottom (down the neck)
    return us, cs


def centre_midline(P, N, T, nm, dm):
    """Shift the face's midline features (midline_offsets) across onto the plane -- the nose, the
    philtrum, the lips, the chin of this model drift toward her right as they go down -- fading
    out across by CENTRE_REACH, so the cheeks and the ears stay. Both halves alike. Without it,
    mirroring her right half (or averaging the two) of a feature that is off the plane makes it
    two-lobed: a groove down the nose, a double bow in the lip. Returns new P, N."""
    c0, up, fwd = plane_frame(nm, dm)
    V, Tw, winv, Nw = E.welded_normals(P, N, T)
    us, cs = midline_offsets(V, Tw, nm, dm)
    a, u = (V @ nm) - dm, (V - c0) @ up
    c = np.interp(u, us, cs, left=0.0, right=0.0)
    # whole round each feature's own centre (so it moves as one piece), fading out beyond it; and
    # only on the front of the head (it is closed: the nape behind the mouth must stay)
    fade = E.smooth01((CENTRE_REACH - np.abs(a - c)) / (CENTRE_REACH - CENTRE_FLAT)) * front_weight(V, c0, fwd)
    Vn = V - (c * fade)[:, None] * nm
    Nr0, Nr1 = hg.vertex_normals(V, Tw), hg.vertex_normals(Vn, Tw)
    # the normals turn as the surface did (the shift shears it a little)
    rot = Nw + (Nr1 - Nr0)
    rot /= np.linalg.norm(rot, axis=1)[:, None]
    moved = np.abs(c * fade) > 0
    P2, N2 = P.copy(), N.copy()
    mv = moved[winv]
    P2[mv] = Vn[winv[mv]]; N2[mv] = rot[winv[mv]]
    k = np.argsort(np.abs(us - (-0.015)))[0]; j = np.argsort(np.abs(us - (-0.050)))[0]
    log("midline: features re-centred onto the plane (the nose tip was %+.1f mm off it, the lips %+.1f mm, "
        "largest shift %.1f mm)" % (1e3 * cs[k], 1e3 * cs[j], 1e3 * np.abs(cs).max()))
    return P2, N2


def lips_weight(X, nm, dm, c0, up, fwd):
    """1 on the lips (the MOUTH ellipse, on the front of the head), fading out just round them."""
    return ellipse_fade(X, nm, dm, c0, up, MOUTH, 1.0, 1.25) * front_weight(X, c0, fwd)


def flatten_brows(color, nmap, UVf, Tr, Pf, nm, dm):
    """The normal map flat (the vertex normals as they are) under the painted eyebrows: the final
    head's texels whose own position is inside BROWS and whose paint is dark and warm, grown 2 texels
    and feathered -- on the head's own texels only (the atlas is shared by every part). Returns the
    normal atlas (a modified copy)."""
    c0, up, fwd = plane_frame(nm, dm)
    S = color.shape[0]
    pos = np.zeros((S, S, 3))
    face = np.zeros((S, S), bool)
    for i, ry, rx, l in E.raster(UVf[Tr] * S, S, S):
        pos[ry, rx] = l @ Pf[Tr[i]]
        face[ry, rx] = True
    u, a = (pos - c0) @ up, np.abs((pos @ nm) - dm)
    lo, hi, reach = BROWS
    zone = face & (u > lo) & (u < hi) & (a < reach) & (front_weight(pos, c0, fwd) > 0.5)
    lum = color @ np.array([0.2126, 0.7152, 0.0722])
    brow = zone & (lum < 0.45) & (color[..., 0] > color[..., 2] + 0.08)
    brow = ndimage.binary_opening(brow, iterations=1)
    w = ndimage.gaussian_filter(ndimage.binary_dilation(brow, iterations=2).astype(np.float32), 1.0)
    w = (w * face)[..., None]
    out = nmap * (1 - w) + np.array([0.5, 0.5, 1.0]) * w
    log("eyebrows: the normal map flattened under them (%d texels)" % int(brow.sum()))
    return out


def front_weight(X, c0, fwd):
    """1 on the front of the head, 0 behind it (by height out of the face, FRONT): the head is one
    closed mesh, and the nape lies behind the mouth's footprint."""
    return E.smooth01(((X - c0) @ fwd - FRONT[0]) / (FRONT[1] - FRONT[0]))


def narrow_mouth(P, N, UV, T, atlas, nm, dm):
    """Her right half's mouth squeezed toward the middle, the squeeze fading out round it, so that
    the mouth is MOUTH_WIDTH of the distance between the eyes' centres: her right corner (the painted
    lips' outermost point there) is measured, and so are the eyes (the painted eyes' area-weighted
    centres, eye_frames). The lips'
    relief is carried with them: each moved vertex keeps its height over a smooth copy of the face
    (Gaussian, 8 mm; made from her right side and its mirror image, so it meets the plane square-on,
    as the averaged face does). The paint moves with the vertices (their texture coordinates are
    kept). Returns new P, N."""
    c0, up, fwd = plane_frame(nm, dm)
    V, Tw, winv, Nw = E.welded_normals(P, N, T)
    a, u, h = (V @ nm) - dm, (V - c0) @ up, (V - c0) @ fwd
    # where the right corner and the eyes are
    H, W, _ = atlas.shape
    uvc = UV[T].mean(1)
    col = atlas[np.clip((uvc[:, 1] * H).astype(int), 0, H - 1), np.clip((uvc[:, 0] * W).astype(int), 0, W - 1)]
    lips = (col[:, 0] > col[:, 1] + 0.25) & (col[:, 0] > col[:, 2] + 0.12) & (col[:, 0] > 0.45)
    lips &= (ellipse_r(V[Tw].mean(1), nm, dm, c0, up, MOUTH) < 1.0) & (front_weight(V[Tw].mean(1), c0, fwd) > 0.5)
    corner = -float(a[Tw[lips & ((a[Tw].mean(1)) < 0)]].min())
    left, _ = E.eye_frames(P, T, UV, atlas, nm, dm)
    eyes = 2.0 * abs(float(left["c"] @ nm) - dm)
    k = MOUTH_WIDTH * eyes / 2.0 / corner
    g = ellipse_fade(V, nm, dm, c0, up, MOUTH, 1.0, MOUTH_FADE) * (a < 0) * front_weight(V, c0, fwd)
    mv = np.nonzero(g > 0)[0]
    # the face's height over the plane's frame round the mouth, made symmetric, and a smooth copy
    PX = 2.5e-4
    a0, u0, u1 = 0.12, -0.11, 0.0
    W_, H_ = int(2 * a0 / PX), int((u1 - u0) / PX)
    px, py = (a + a0) / PX, (u1 - u) / PX
    fn = np.cross(V[Tw[:, 1]] - V[Tw[:, 0]], V[Tw[:, 2]] - V[Tw[:, 0]])
    Tf = Tw[(fn @ fwd) > 0]
    hz = np.full((H_, W_), -np.inf)
    for i, yy, xx, l in E.raster(np.stack([px[Tf], py[Tf]], -1), W_, H_):
        hv = l @ h[Tf[i]]
        b = hv > hz[yy, xx]
        hz[yy[b], xx[b]] = hv[b]
    hz[:, W_ // 2:] = hz[:, :W_ // 2][:, ::-1]                # her right side and its mirror image
    m = np.isfinite(hz)
    sig = 0.008 / PX
    base = ndimage.gaussian_filter(np.where(m, hz, 0.0), sig) / np.maximum(ndimage.gaussian_filter(m * 1.0, sig), 1e-9)

    def base_at(aa, uu):
        q = np.c_[(aa + a0) / PX / W_, (u1 - uu) / PX / H_]
        return E.bilinear(base[..., None], q[:, 0], q[:, 1])[:, 0]
    an = a[mv] * (1.0 - (1.0 - k) * g[mv])
    hn = base_at(an, u[mv]) + (h[mv] - base_at(a[mv], u[mv]))
    Vn = V.copy()
    Vn[mv] = c0 + an[:, None] * nm + u[mv, None] * up + hn[:, None] * fwd
    # normals: the new surface's, blended in by the squeeze's weight
    Nr = hg.vertex_normals(Vn, Tw)
    Nn = Nw.copy()
    Nn[mv] = (1 - g[mv, None]) * Nw[mv] + g[mv, None] * Nr[mv]
    Nn[mv] /= np.maximum(1e-12, np.linalg.norm(Nn[mv], axis=1))[:, None]
    moved = np.isin(winv, mv)
    P2, N2 = P.copy(), N.copy()
    P2[moved] = Vn[winv[moved]]; N2[moved] = Nn[winv[moved]]
    log("mouth: her right corner %.1f mm out, the eyes' centres %.1f mm apart: squeezed across to %.2f, so the mouth "
        "is %.2f of their spacing (%d vertices moved, up to %.1f mm)"
        % (1e3 * corner, 1e3 * eyes, k, MOUTH_WIDTH, len(mv), 1e3 * np.linalg.norm(Vn[mv] - V[mv], axis=1).max()))
    return P2, N2


class Surface:
    """A triangle set to project points onto: closest point, its triangle and barycentrics."""
    def __init__(self, V, F):
        self.V, self.F = V, F
        self.m = trimesh.Trimesh(V, F, process=False)

    def closest(self, X):
        cp, _, tid = trimesh.proximity.closest_point(self.m, X)
        bary = trimesh.triangles.points_to_barycentric(self.V[self.F[tid]], cp)
        return cp, tid, bary


def texel_coverage(polys_uv, S):
    """Texel mask (S x S) covered by the given UV triangles (n x 3 x 2)."""
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    for t in polys_uv:
        d.polygon([tuple(p) for p in t * S], fill=255)
    return np.asarray(im) > 0


def average_shape(P, N, T, nm, dm):
    """Her right half where the sides are averaged (left_weight: the midline band, the chin): each
    vertex moved that far toward her left half's mirror image, its normal blended the same way.
    Returns new P, N."""
    c0, up, fwd = plane_frame(nm, dm)
    V, Tw, winv, Nw = E.welded_normals(P, N, T)
    sideT = (V[Tw].mean(1) @ nm) - dm
    left = Surface(V, Tw[sideT > 0])
    w = left_weight(V, nm, dm, c0, up) * ((V @ nm) - dm < 0)
    mv = np.nonzero(w > 0)[0]
    cp, tid, bary = left.closest(E.reflect_p(V[mv], nm, dm))
    Vn, Nn = V.copy(), Nw.copy()
    Vn[mv] += w[mv, None] * (E.reflect_p(cp, nm, dm) - V[mv])
    nl = np.einsum("ij,ijk->ik", bary, Nw[left.F[tid]])
    nl = E.reflect_v(nl / np.maximum(1e-12, np.linalg.norm(nl, axis=1))[:, None], nm)
    Nn[mv] = (1 - w[mv, None]) * Nw[mv] + w[mv, None] * nl
    Nn[mv] /= np.maximum(1e-12, np.linalg.norm(Nn[mv], axis=1))[:, None]
    log("averaging: %d of her right side's vertices moved toward her left side's mirror image (up to %.1f mm)"
        % (len(mv), 1e3 * np.linalg.norm(Vn[mv] - V[mv], axis=1).max()))
    Pn, Nu = P.copy(), N.copy()
    moved = np.isin(winv, mv)
    Pn[moved] = Vn[winv[moved]]; Nu[moved] = Nn[winv[moved]]
    return Pn, Nu


def rebake_averaged(g, mi, P0, N0, UV0, T0, Pf, Nf, UVf, Tr, atlas, mr, nmap, nm, dm):
    """The paint where the sides are averaged, rebaked on the FINAL head: every texel of its right
    half (`Tr`, which shares its texels with its mirror image) gets the blend of the original right
    side there and the original left side at the mirror point (except on the lips, which keep her
    right side's paint: lips_weight) -- colour, metal/roughness, and the
    normal map's world normal, laid into the plane near the plane (PLANE_FLAT) and re-expressed in
    the final triangle's tangent frame and vertex normals, as the renderer will read it.
    Returns the three atlases (modified copies)."""
    c0, up, fwd = plane_frame(nm, dm)
    S = atlas.shape[0]
    flipv = np.array([1.0, -1.0])
    V0, Tw0, winv0, Nw0 = E.welded_normals(P0, N0, T0)
    sideT = (V0[Tw0].mean(1) @ nm) - dm
    lsel = sideT > 0
    left = Surface(V0, Tw0[lsel])
    _, Tl, sl = E.tangent_frames(V0[Tw0[lsel]], UV0[T0[lsel]] * flipv)
    UVl = UV0[T0[lsel]]
    # each texel's original triangle and barycentrics (her right side and the plane's straddlers)
    osel = np.nonzero((((V0 @ nm) - dm)[Tw0] < 0).any(1))[0]
    _, To, so = E.tangent_frames(P0[T0[osel]], UV0[T0[osel]] * flipv)
    otri = np.full((S, S), -1, np.int64); obar = np.zeros((S, S, 3))
    for j, ry, rx, l in E.raster(UV0[T0[osel]] * S, S, S):
        otri[ry, rx] = j; obar[ry, rx] = l
    _, Tf, sf = E.tangent_frames(Pf[Tr], UVf[Tr] * flipv)
    fsel = np.nonzero((left_weight(Pf[Tr].reshape(-1, 3), nm, dm, c0, up).reshape(-1, 3) > 0).any(1))[0]
    atl, mrt, nmt = atlas.copy(), mr.copy(), nmap.copy()
    written = np.zeros((S, S), bool)
    for jj, ry, rx, l in E.raster(UVf[Tr[fsel]] * S, S, S):
        j = fsel[jj]
        oj = otri[ry, rx]
        ok = oj >= 0
        ry, rx, l, oj = ry[ok], rx[ok], l[ok], oj[ok]
        ob = obar[ry, rx]
        po = np.einsum("ij,ijk->ik", ob, P0[T0[osel[oj]]])
        wp = left_weight(po, nm, dm, c0, up)
        keep = wp > 0
        if not keep.any():
            continue
        ry, rx, l, oj, ob, po, wp = ry[keep], rx[keep], l[keep], oj[keep], ob[keep], po[keep], wp[keep]
        # the original right side there ...
        No = np.einsum("ij,ijk->ik", ob, N0[T0[osel[oj]]]); No /= np.maximum(1e-12, np.linalg.norm(No, axis=1))[:, None]
        uvo = np.einsum("ij,ijk->ik", ob, UV0[T0[osel[oj]]])
        n0 = E.decode_normal(E.bilinear(nmap, uvo[:, 0], uvo[:, 1]), No, To[oj], so[oj])
        # ... and the original left side at the mirror point, reflected
        _, tl_, bl = left.closest(E.reflect_p(po, nm, dm))
        uvl = np.einsum("ij,ijk->ik", bl, UVl[tl_])
        Nl = np.einsum("ij,ijk->ik", bl, Nw0[left.F[tl_]]); Nl /= np.maximum(1e-12, np.linalg.norm(Nl, axis=1))[:, None]
        n_l = E.reflect_v(E.decode_normal(E.bilinear(nmap, uvl[:, 0], uvl[:, 1]), Nl, Tl[tl_], sl[tl_]), nm)
        # the lips keep her right half's paint (closed lips, as in the photos; her left half's shows
        # teeth, which the band would blend into the middle of the mouth)
        wq = (wp * (1.0 - lips_weight(po, nm, dm, c0, up, fwd)))[:, None]
        nb = (1 - wq) * n0 + wq * n_l
        # on a symmetric face the normal lies in the plane AT the plane; the old map does not agree
        # across it (its islands meet there), so neither need the average: made to, near the plane
        pf = l @ Pf[Tr[j]]
        beta = E.smooth01((PLANE_FLAT - np.abs((pf @ nm) - dm)) / PLANE_FLAT)[:, None]
        nb = nb - beta * (nb @ nm)[:, None] * nm
        # re-expressed in the final frame, as the renderer will read it
        Nv = l @ Nf[Tr[j]]; Nv /= np.maximum(1e-12, np.linalg.norm(Nv, axis=1))[:, None]
        atl[ry, rx] = (1 - wq) * atlas[ry, rx] + wq * E.bilinear(atlas, uvl[:, 0], uvl[:, 1])
        mrt[ry, rx] = (1 - wq) * mr[ry, rx] + wq * E.bilinear(mr, uvl[:, 0], uvl[:, 1])
        nmt[ry, rx] = E.encode_normal(nb, Nv, Tf[j], sf[j])
        written[ry, rx] = True
    # the texels round what was written follow it, where nothing reads them any more -- the islands'
    # padding, and the dropped left parts of the triangles cut at the plane, which sit right against
    # the seam: bilinear filtering reads them there
    used = np.zeros((S, S), bool)
    for _, ry, rx, _ in E.raster(UVf[Tr] * S, S, S):      # the same texel rule as `written`
        used[ry, rx] = True
    for mj in range(len(g.json["meshes"])):
        if mj != mi:
            _, _, UVj, Tj = g.mesh(mj)
            used |= texel_coverage(UVj[Tj], S)
    dw, ind = ndimage.distance_transform_edt(~written, return_indices=True)
    do = ndimage.distance_transform_edt(~(used & ~written))
    pad = ~used & (dw <= 3) & (dw < do)
    for im in (atl, mrt, nmt):
        im[pad] = im[ind[0][pad], ind[1][pad]]
    log("averaging: %d texels rebaked from both sides (+%d of padding)" % (written.sum(), pad.sum()))
    return atl, mrt, nmt


def renormal(P, N, T, nm, dm):
    """Where the shape was changed (the averaged band and chin, the narrowed mouth), the vertex
    normals of the FINAL surface (area-weighted), blended in by how much it was changed there; the
    model's own normals elsewhere. (Blending the old normals instead leaves them up to ~30 degrees
    off the moved surface: faceted shading.) Symmetric in, symmetric out. Returns N."""
    c0, up, fwd = plane_frame(nm, dm)
    V, Tw, winv = hg.weld(P, T)
    Nr = hg.vertex_normals(V, Tw)[winv]
    mouth = ellipse_fade(P, nm, dm, c0, up, MOUTH, 1.0, MOUTH_FADE) * front_weight(P, c0, fwd)
    k = np.clip(np.maximum(2.0 * left_weight(P, nm, dm, c0, up), mouth), 0.0, 1.0)[:, None]
    N2 = (1 - k) * N + k * Nr
    N2 /= np.maximum(1e-12, np.linalg.norm(N2, axis=1))[:, None]
    ang = np.degrees(np.arccos(np.clip((N2 * Nr).sum(1), -1, 1)))
    log("normals: %d vertices take the new surface's (95th percentile off it now %.1f degrees)"
        % (int((k[:, 0] > 0).sum()), np.percentile(ang, 95)))
    return N2


def mirror_right(P, N, UV, T, nm, dm):
    """Cut the head along the plane, keep her right half (clipping the triangles across the plane),
    and add its mirror image as her left half. Returns P, N, UV, T of the symmetric head, and how
    many of its triangles (the first) are her right half's."""
    s = P @ nm - dm
    on = np.abs(s) < SNAP
    P = P - (s * on)[:, None] * nm
    s = np.where(on, 0.0, s)
    Pl, Nl, Ul = list(P), list(N), list(UV)
    cut = {}

    def cutv(i, j):
        k = (min(i, j), max(i, j))
        if k not in cut:
            if s[i] > 0:                     # always from her right end: a UV seam's two copies of the
                i, j = j, i                  # edge then give bit-identical cut points
            t = s[i] / (s[i] - s[j])
            n = (1 - t) * N[i] + t * N[j]
            Pl.append((1 - t) * P[i] + t * P[j]); Nl.append(n / np.linalg.norm(n)); Ul.append((1 - t) * UV[i] + t * UV[j])
            cut[k] = len(Pl) - 1
        return cut[k]
    out = []
    for tri in T:
        ss = s[tri]
        if (ss <= 0).all():
            out.append(tri.tolist())
            continue
        if (ss >= 0).all():
            continue
        # Sutherland-Hodgman against s <= 0
        poly = []
        for a in range(3):
            i, j = int(tri[a]), int(tri[(a + 1) % 3])
            if s[i] <= 0:
                poly.append(i)
            if (s[i] < 0 < s[j]) or (s[j] < 0 < s[i]):
                poly.append(cutv(i, j))
        for b in range(1, len(poly) - 1):
            out.append([poly[0], poly[b], poly[b + 1]])
    P2, N2, U2 = np.array(Pl), np.array(Nl), np.array(Ul)
    Tr = np.array(out, np.int64)
    used = np.unique(Tr)
    remap = -np.ones(len(P2), np.int64); remap[used] = np.arange(len(used))
    P2, N2, U2, Tr = P2[used], N2[used], U2[used], remap[Tr]
    # on the plane, the normal lies in it: the two halves' normals agree there
    onp = np.abs(P2 @ nm - dm) < SNAP
    N2[onp] -= (N2[onp] @ nm)[:, None] * nm
    N2[onp] /= np.linalg.norm(N2[onp], axis=1)[:, None]
    # her left half: the mirror image -- except along the plane, where it SHARES her right half's
    # vertices (the same numbers, so the seam is watertight: separate copies differ in the last bit
    # once stored as float32, and rays slip through the crack)
    n = len(P2)
    P2[onp] -= ((P2[onp] @ nm) - dm)[:, None] * nm
    mid = np.where(onp, np.arange(n), np.arange(n) + n)
    Pm, Nm = E.reflect_p(P2, nm, dm), E.reflect_v(N2, nm)
    Pa, Na, Ua = np.r_[P2, Pm], np.r_[N2, Nm], np.r_[U2, U2]
    Ta = np.r_[Tr, mid[Tr[:, ::-1]]]
    used = np.unique(Ta)
    remap = -np.ones(len(Pa), np.int64); remap[used] = np.arange(len(used))
    log("cut along the plane: her right half %d triangles (%d new vertices on the plane), mirrored; "
        "the halves share their %d vertices on the plane" % (len(Tr), len(cut), int(onp.sum())))
    return Pa[used], Na[used], Ua[used], remap[Ta], len(Tr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glb", default=E.DEFAULT_GLB)
    ap.add_argument("--out", default=None, help="default: <glb>_sym.glb beside it")
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.glb)[0] + "_sym.glb"
    g = hg.Glb(a.glb)
    mi = g.mesh_by_material(FACE)
    prims = g.json["meshes"][mi]["primitives"]
    prim = prims[0]
    if set(prim["attributes"]) - {"POSITION", "NORMAL", "TEXCOORD_0"} or "targets" in prim:
        raise RuntimeError("root.4 has attributes or morph targets this tool would drop: %s" % sorted(prim))
    if len(prims) > 1:
        raise RuntimeError("root.4 has %d primitives (an eyes copy? run this before tools/alice2_eyes.py, "
                           "on the plain model)" % len(prims))
    mat = g.json["materials"][prim["material"]]
    P, N, UV, T = g.mesh(mi)
    pbr = mat["pbrMetallicRoughness"]
    src = {k: g.json["textures"][ti]["source"] for k, ti in (("color", pbr["baseColorTexture"]["index"]),
                                                             ("mr", pbr["metallicRoughnessTexture"]["index"]),
                                                             ("normal", mat["normalTexture"]["index"]))}
    for k, v in src.items():                         # (they are rewritten as RGB)
        bv = g.json["bufferViews"][g.json["images"][v]["bufferView"]]
        o = bv.get("byteOffset", 0)
        if "A" in Image.open(io.BytesIO(bytes(g.bin[o:o + bv["byteLength"]]))).mode:
            raise RuntimeError("the %s atlas has an alpha channel, which rewriting it would drop" % k)
    img = {k: g.image(v).astype(np.float32) / 255.0 for k, v in src.items()}
    V, _, _ = hg.weld(P, T)
    nm, dm = E.mirror_plane(V)
    log("mirror plane: normal %s, offset %.4f m" % (np.round(nm, 4), dm))
    P, N = centre_midline(P, N, T, nm, dm)
    P0, N0, UV0, T0 = P, N, UV, T          # the re-centred head: what the paint is paired up on
    P, N = average_shape(P, N, T, nm, dm)
    P, N = narrow_mouth(P, N, UV, T, img["color"], nm, dm)
    P, N, UV, T, n_right = mirror_right(P, N, UV, T, nm, dm)
    N = renormal(P, N, T, nm, dm)
    ac, am, an = rebake_averaged(g, mi, P0, N0, UV0, T0, P, N, UV, T[:n_right],
                                 img["color"], img["mr"], img["normal"], nm, dm)
    an = flatten_brows(ac, an, UV, T[:n_right], P, nm, dm)
    go = E.GlbOut(g)
    prim = go.j["meshes"][mi]["primitives"][0]
    prim["attributes"] = {"POSITION": go.accessor(P.astype(np.float32), "VEC3", 34962),
                          "NORMAL": go.accessor(N.astype(np.float32), "VEC3", 34962),
                          "TEXCOORD_0": go.accessor(UV.astype(np.float32), "VEC2", 34962)}
    prim["indices"] = go.accessor(T.astype(np.uint32).ravel(), "SCALAR", 34963)
    # the shared atlases, with the face's averaged texels rebaked (no other part uses them)
    for k, im in (("color", ac), ("mr", am), ("normal", an)):
        buf = io.BytesIO()
        Image.fromarray(E.to8(im)).save(buf, format="PNG", optimize=True)
        go.j["images"][src[k]] = {"bufferView": go._view(buf.getvalue()), "mimeType": "image/png"}
    go.write(out)
    log("wrote %s: %s %d vertices, %d triangles, symmetric about its mirror plane" % (out, FACE, len(P), len(T)))


if __name__ == "__main__":
    main()
