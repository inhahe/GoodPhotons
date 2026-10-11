"""How big are Alice2's eyes and mouth, against the photos of the real doll?

    D:/python314/python.exe -u tools/alice2_photo_check.py --glb <alice2.glb> [--eye painted|tool] [--scale k]

Each photo's camera (d:/youtube/philosophy/images/alice) is fitted once, with cv2.solvePnP and the
phone's own lens (a 23 mm equivalent: 2556 px across the photo's 4000), to landmarks on the Hyper3D
head as made -- irises, mouth corners, nose tip, chin, ears -- read off orthographic views of it and
off the photos. Every version of the doll shares that head's coordinates, so one camera per photo is
a fixed ruler for all of them. Then a feature's corners on the GLB's head are projected through it
and compared with the same corners read off the photo:

  * her left eye in the three-quarter photo (p2): corners, top of the lash band, bottom of the
    white. `--eye painted` measures the eye painted on the head (the model's own, or one
    alice2_eyes.py already wrote); `--eye tool` measures the eye alice2_eyes.py WOULD paint on this
    head at `--scale`.
  * the mouth (the painted lips' extremes along the face) in the front (p1) and three-quarter (p2)
    photos. The front photo is blurry; the three-quarter one is sharp.

A ratio of two features within one photo (mouth to eye spacing, say) is not used: the three-quarter
view foreshortens them differently.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import alice2_eyes as E          # noqa: E402
import alice2_symmetrize as S    # noqa: E402
import hairgeo as hg             # noqa: E402

FOCAL = 23.0 / 36.0 * 4000       # px: the 23 mm (35 mm equivalent) lens across the 4000 px photo
W, H = 1868, 4000                # the photos, upright
# landmarks on the head as made (world m), and where the photos show them (upright pixels)
HEAD = {"eye_right": (-0.04107, 1.64312, 0.12304), "eye_left": (0.07652, 1.64761, 0.11384),
        "mouth_right": (-0.02017, 1.56416, 0.12281), "mouth_left": (0.05189, 1.56713, 0.12252),
        "nose_tip": (0.01964, 1.60040, 0.15795), "chin": (-0.00296, 1.52605, 0.11971),
        "ear_top_right": (-0.10245, 1.65576, 0.01019), "ear_low_right": (-0.08421, 1.59768, -0.00377),
        "ear_mid_right": (-0.08645, 1.62460, 0.00630), "ear_top_left": (0.12470, 1.69007, 0.00582),
        "ear_low_left": (0.11174, 1.61675, -0.00065), "ear_mid_left": (0.12459, 1.65376, -0.00031)}
PHOTOS = {
    "p1": {"eye_right": (718, 688), "eye_left": (968, 690), "mouth_right": (800, 900), "mouth_left": (905, 900),
           "nose_tip": (850, 815)},
    "p2": {"eye_left": (719, 820), "eye_right": (550, 821), "nose_tip": (593, 925), "mouth_left": (671, 996),
           "mouth_right": (587, 1004), "ear_top_left": (900, 846), "ear_low_left": (904, 907), "ear_mid_left": (909, 875)},
    "p5": {"eye_right": (1111, 796), "nose_tip": (1225, 879), "mouth_right": (1144, 975), "chin": (1170, 1021),
           "ear_top_right": (882, 825), "ear_low_right": (868, 900), "ear_mid_right": (882, 861)},
}
P2_EYE = dict(inner=(675.0, 830.0), outer=(768.0, 817.0), top=(721.0, 776.0), bottom=(717.0, 858.0))


def fit_cameras():
    """{photo: (rvec, tvec, K)} fitted to the head-as-made landmarks; prints each fit's RMS."""
    K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])
    cams = {}
    for ph, pts in PHOTOS.items():
        keys = list(pts)
        obj = np.array([HEAD[k] for k in keys], np.float64)
        img = np.array([pts[k] for k in keys], np.float64)
        _, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP)
        _, rv, tv = cv2.solvePnP(obj, img, K, None, rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        pr, _ = cv2.projectPoints(obj, rv, tv, K, None)
        rms = np.sqrt(((pr[:, 0] - img) ** 2).sum(1).mean())
        print("camera %s fitted to %d landmarks: RMS %.1f px" % (ph, len(keys), rms))
        cams[ph] = (rv, tv, K)
    return cams


def project(X, cam):
    rv, tv, K = cam
    pr, _ = cv2.projectPoints(np.atleast_2d(np.asarray(X, np.float64)), rv, tv, K, None)
    return pr[:, 0]


def load(glb):
    g = hg.Glb(glb)
    mi = g.mesh_by_material(E.FACE)
    P, N, UV, T = g.mesh(mi)
    mat = g.json["materials"][g.json["meshes"][mi]["primitives"][0]["material"]]
    img = lambda ti: g.image(g.json["textures"][ti]["source"]).astype(np.float32) / 255.0   # noqa: E731
    pbr = mat["pbrMetallicRoughness"]
    return P, N, UV, T, img(pbr["baseColorTexture"]["index"]), img(pbr["metallicRoughnessTexture"]["index"]), img(mat["normalTexture"]["index"])


def eye_marks(glb, mode, scale):
    """Her left eye's inner/outer corner, top and bottom (world m), as painted or as alice2_eyes would
    paint it at `scale`; and its size in mm."""
    P, N, UV, T, atlas, mr, nmap = load(glb)
    V, Tw, winv, Nw = E.welded_normals(P, N, T)
    nm, dm = E.mirror_plane(V)
    frames = E.eye_frames(P, T, UV, atlas, nm, dm)
    R = E.RES
    _, _, ga, gb = E.texel_grid(R)
    ell0 = (ga / E.PATCH0[0]) ** 2 + (gb / E.PATCH0[1]) ** 2 < 1
    seen = np.zeros(len(T), bool)
    ops = []
    for k, fr in enumerate(frames):
        sel, uvn = E.patch_tris(P, T, fr, k, R, ell0, seen)
        seen |= sel
        c0, _, _, cov0 = E.bake(E.texel_tris(uvn, T[sel], k, R), T[sel], P, N, UV, uvn, atlas, mr, nmap, R, R)
        ops.append(E.analyse(c0, cov0)["opening"])
    if mode == "painted":
        al = E.Almond(ops[0])
        ys, xs = np.nonzero(ops[0])
        _, t = al.st(xs + 0.5, ys + 0.5)
        top_t, bot_t = t.max(), t.min()
    else:
        op = 0.5 * (E.sdf(ops[0]) + E.sdf(ops[1])) < 0
        lab, n = ndimage.label(op)
        if n > 1:
            op = lab == 1 + int(np.argmax(ndimage.sum(op, lab, np.arange(1, n + 1))))
        al = E.Almond(op)
        if scale != 1.0:
            al.scale(scale)
        sm = np.linspace(al.s0, al.s1, 200)
        top_t, bot_t = al.top(sm).max(), al.bot(sm).min()
    fr = frames[0]
    hz = E.heightmap(V, Tw, fr, R)
    hz = np.where(np.isfinite(hz), hz, np.nanmedian(np.where(np.isfinite(hz), hz, np.nan)))

    def to3d(q):
        a = (q[0] / R - 0.5) * 2 * E.HALF; b = (0.5 - q[1] / R) * 2 * E.HALF
        return fr["c"] + a * fr["x"] + b * fr["y"] + hz[int(np.clip(q[1], 0, R - 1)), int(np.clip(q[0], 0, R - 1))] * fr["n"]
    sm = 0.5 * (al.s0 + al.s1)
    q = dict(inner=al.xy(np.array([al.s0]), al.base(np.array([al.s0])))[0],
             outer=al.xy(np.array([al.s1]), al.base(np.array([al.s1])))[0],
             top=al.xy(np.array([sm]), np.array([top_t]))[0], bottom=al.xy(np.array([sm]), np.array([bot_t]))[0])
    mm = 1e3 * 2 * E.HALF / R
    return {k: to3d(v) for k, v in q.items()}, (al.s1 - al.s0) * mm, (top_t - bot_t) * mm


def mouth_corners(glb):
    """The painted lips' extremes along the face (world m) and the mouth's width (mm)."""
    P, N, UV, T, atlas, _, _ = load(glb)
    V, _, _ = hg.weld(P, T)
    nm, dm = E.mirror_plane(V)
    c0, up, fwd = S.plane_frame(nm, dm)
    Hh, Ww, _ = atlas.shape
    uvc = UV[T].mean(1)
    col = atlas[np.clip((uvc[:, 1] * Hh).astype(int), 0, Hh - 1), np.clip((uvc[:, 0] * Ww).astype(int), 0, Ww - 1)]
    cen = P[T].mean(1)
    lips = (col[:, 0] > col[:, 1] + 0.25) & (col[:, 0] > col[:, 2] + 0.12) & (col[:, 0] > 0.45)
    lips &= (S.ellipse_r(cen, nm, dm, c0, up, S.MOUTH) < 1.0) & (S.front_weight(cen, c0, fwd) > 0.5)
    a = cen @ nm - dm
    r, l = cen[lips][np.argmin(a[lips])], cen[lips][np.argmax(a[lips])]
    return r, l, 1e3 * (l - r) @ nm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--glb", default=E.DEFAULT_GLB)
    ap.add_argument("--eye", choices=("painted", "tool"), default="painted")
    ap.add_argument("--scale", type=float, default=1.0, help="with --eye tool: alice2_eyes.py's --scale")
    a = ap.parse_args()
    cams = fit_cameras()
    pe = {k: np.array(v) for k, v in P2_EYE.items()}
    pw, ph = np.linalg.norm(pe["outer"] - pe["inner"]), np.linalg.norm(pe["top"] - pe["bottom"])
    marks, wmm, hmm = eye_marks(a.glb, a.eye, a.scale)
    pr = {k: project(v, cams["p2"])[0] for k, v in marks.items()}
    w, h = np.linalg.norm(pr["outer"] - pr["inner"]), np.linalg.norm(pr["top"] - pr["bottom"])
    print("her left eye (%s%s): %.1f x %.1f mm -> in the three-quarter photo %.0f x %.0f px, "
          "the real eye %.0f x %.0f px: %.2f x %.2f of it"
          % (a.eye, " x%.2f" % a.scale if a.eye == "tool" else "", wmm, hmm, w, h, pw, ph, w / pw, h / ph))
    r, l, mw = mouth_corners(a.glb)
    for phn in ("p1", "p2"):
        q = project(np.array([r, l]), cams[phn])
        real = np.linalg.norm(np.subtract(PHOTOS[phn]["mouth_left"], PHOTOS[phn]["mouth_right"]))
        print("the mouth (%.1f mm wide) in %s: %.0f px, the real one %.0f px: %.2f of it"
              % (mw, {"p1": "the front photo", "p2": "the three-quarter photo"}[phn], np.linalg.norm(q[1] - q[0]),
                 real, np.linalg.norm(q[1] - q[0]) / real))


if __name__ == "__main__":
    main()
