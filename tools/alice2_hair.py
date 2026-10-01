"""Real hair for the Alice2 doll: replace her sculpted hair with a strand groom, as an FTSL scene.

    python tools/alice2_hair.py [--glb <alice2 .glb>] [--out <dir>] [--count 90000] [--radius 0.00024]
                                [--spacing 0.0072] [--seed 1]

Default --glb is the sequinned Alice2 (D:/youtube/philosophy/3d objects/alice2/alice2_hyper3d/
model/base_basic_pbr_flat_sparkle.glb); --out defaults to hair_opus5.5 beside that model folder (or
the GLB's own folder when it is not in one called `model`), and the scene finds the GLB by a path
relative to --out. ~1 minute (numpy, scipy, numba, Pillow). It writes into --out:

  alice2_real_hair.ftsl        the scene: the doll with her hair primitive skipped (`skip_material
                               root.1`), the groom, the headband, a room to light it, cameras
  alice2_hair_guides.ftsl      ~900 guide curves (definitions) -- curves only, so the groom tool
                               (`ftrace -groom`, groom.bat) can edit and save them
  alice2_hair_groom.ftsl       includes the guides; three guided `fur` blocks, the hair material,
                               the scalps and ~240 stray flyaway fibres
  alice2_scalp_{fr,fl,bk}.obj  the invisible root surfaces (`shape_only`)
  alice2_headband.obj          the satin headband + bow
  alice2_room.hdr              the environment: an indoor room, as in the reference photos

The model is a sculpt: her hair is one closed shell (root.1, 231k triangles) wrapped round the
head, 6-10 cm thick over the crown with the headband and bow modelled into its surface. The groom
follows its hairstyle and silhouette, and more closely still the photos of the actual doll
(images/alice): a centre part, the front swept down over the forehead corners and back behind the
ears under the headband, shoulder-length soft waves with curled ends, locks in front of both
shoulders, pale golden synthetic fibre, stray flyaways.

HOW (each step's numbers come from the model and were checked by rendering):
  * Which sculpt triangles are OUTER surface: not inside, nor within 3 mm of, the head or body.
  * The SCALP: head triangles inside the hair shell, minus the ears, grown 6 mm past the ledge
    the sculpt left on the forehead (so the hairline covers it). A stack field lam on it -- 0 at
    the part, 1 at the lower hairline (geodesic distances) -- says which strands lie on top.
  * HEAD PHASE: each guide is traced over a Taubin-smoothed head (ears flattened) along a
    designed flow -- away from the part, down over the forehead corners, back behind the ears,
    then down -- lying on the hair stacked beneath it (height ~ CAP (phi - lam)^0.6), until the
    skull turns under.
  * HANG PHASE, a 'curtain' in cylindrical coordinates about the body: the strand falls with its
    radius eased to R_in + (1 - lam) KAPPA (R_out - R_in) -- R_out the sculpt's silhouette at
    that height and azimuth, R_in the body and head -- its azimuth drifting with the sculpt's own
    flow (the side hair goes behind the shoulders, its outer layer forward over them). The sides
    keep all of the sculpt's volume and ease out fast (the hair tucked behind the ears must stand
    far enough out to frame the face from the front); the centre back eases out slowly and keeps
    less level with the ears (a smooth crown from behind); below the shoulders the curtain gives up
    60 % of the sculpt's flare (the photos' back narrows to shoulder width).
  * THE EARS ARE BARE (both profile photos): a strand level with an ear is swept behind it --
    from 4 cm above the ear to its lobe, released over 7 cm below -- an order-preserving remap of
    its azimuth onto the 18 degrees behind the ear's back edge (a single line piled up into a ridge).
    The head phase hands a strand over to the hang where it comes level with an ear.
  * LOCKS: the guides are grouped by where they leave the head into 10-17 degree sectors; a lock's
    guides share its S-wave (22-32 cm crest to crest: a wave shows two highlight bands, so shorter
    ones read as crimping), its length, and its curl -- UNDER at the centre back, more often OUT at
    the sides and for the face-framing front layer -- and gather 55 % toward the lock's own centre
    line by the tip, so gaps open between locks. The centre back hangs 2.5 cm lower than the sides.
  * The groom is three `fur` blocks (front-right / front-left of the part, and the back), so no
    strand near the part blends a guide from across it, each strand following its ONE nearest
    guide (guide_blend 1): any blend across a lock boundary averages two locks into a strand that
    fills the gap between them.
  * The fibre colour was calibrated against the photos: rendered fur relative to a white
    reference reads (1 : 0.82 : 0.43) at `reflect rgb 0.98 0.60 0.28`, the photos' hair against
    the apron (1 : 0.82 : 0.46). A paler reflect reads greenish: the white cuticle highlight and
    the deep scattering lift green far above what the reflect colour suggests.

RENDER: mode R, many bounces -- a pale coat's colour and brightness live in long paths
(red / white ratio 0.46 -> 0.55 from 32 to 256 bounces):
    ftrace -in alice2_real_hair.ftsl -camera front -mode R -device gpu -spp 1024 -max-bounce 128 -denoise
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hairgeo as hg  # noqa: E402

DEFAULT_GLB = r"D:/youtube/philosophy/3d objects/alice2/alice2_hyper3d/model/base_basic_pbr_flat_sparkle.glb"

# ---- the model (raw GLB frame: 1.9 m tall, facing +z; head spans y 1.51..1.80) --------------------
XMID = 0.016                                     # the part's plane (nose tip x 0.0126, hairline 0.0201)
LAM_PART = (np.array([XMID, 1.772, 0.120]), np.array([XMID, 1.782, -0.020]))   # part, for the stack field
FLOW_PART = (np.array([XMID, 1.770, 0.150]), np.array([XMID, 1.782, -0.020]))  # ... for the flow (past the hairline)
EARS = (np.array([-0.108, 1.632, 0.006]), np.array([0.128, 1.672, 0.000]))
EAR_R = 0.030
AX = np.array([0.012, 0.0, 0.020])               # the body's vertical axis (x, -, z)
HEAD_C = np.array([0.012, 1.62, 0.02])           # centre the headband is seated from

# ---- the style ------------------------------------------------------------------------------------
CAP_HEAD = 0.038      # hair thickness over the scalp at the lower hairline (the photos' rounded dome, and
                      # from behind a crown wide enough to flow into the side hair without a step)
KAPPA = 0.80          # share of the sculpt's extra volume (beyond the body) the groom keeps at the nape
KAPPA_SIDE = 1.0     # ... and at the sides, all of it: the hair tucked behind the ears must stand far enough
                     # out to frame the face from the front (the photos: ~1.9x the face's half-width)
TAPER = (1.56, 1.38, 0.60)   # ... and gives up on the way down: from y 1.56 to 1.38 the curtain loses 60%
                      # of that extra volume. The sculpt flares past the shoulders; the photos' back narrows
                      # to about shoulder width.
CLEAR = 0.012         # clearance kept over the body and head
LOCK_DEG = (10.0, 17.0)      # lock widths round the body (azimuth). A lock's guides share its waves, its
                             # length and its curl, so the back reads as locks rather than one sheet.
WAVE_LEN = (0.22, 0.32)      # S-waves, crest to crest -- per lock. Loose: a wave shows TWO highlight bands
                             # (one per slope), so 10-15 cm crests read as crimping, and even 16-24 cm read
                             # tighter than the photos' two or three bends down a lock
WAVE_AMP = (0.011, 0.019)    # ... and their amplitude
CONVERGE = 0.55       # how far a lock gathers toward its own centre line by its tip: gaps open between locks
HEM_BACK = 0.025      # the centre back hangs this much lower than the sides (the photos' rounded hem)
EAR_ABOVE, EAR_RELEASE, EAR_MARGIN, EAR_SPREAD = 0.04, 0.07, 5.0, 18.0
                      # the hair passes BEHIND the ears (both profile photos show them bare): swept back
                      # from 4 cm above an ear down to its lobe, released over 7 cm below it (so the
                      # face-framing layer can still come forward under the jaw), spread over the 18
                      # degrees starting 5 degrees behind the ear's back edge
UPPER = (1.66, 1.54, 0.35)  # level with the ears the centre back keeps 65% of its extra volume, all of it
                            # by the nape: from behind, the crown then flows into the curtain instead of
                            # stepping out to it (at full volume the back stood 12 cm off the head there)
NPTS = 44             # control points per guide (waves every ~12 cm and a curl need more than 32)
HAIR_MAT = ('material "alice2_hair" { type hair  eta 1.55  alpha 0  reflect rgb 0.98 0.60 0.28  '
            'beta_m 0.11  beta_n 0.28 }')
REGIONS = ("fr", "fl", "bk")
REGION_DESC = ("front, her right of the part", "front, her left of the part", "back of the head")

T0 = time.time()


def log(msg):
    print("[alice2_hair %5.1fs] %s" % (time.time() - T0, msg), flush=True)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def graph(V, F, weighted=True):
    e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1) if weighted else np.ones(len(e))
    n = len(V)
    return sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([e[:, 0], e[:, 1]]),
                                                    np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n)).tocsr()


def smooth_field(X, A, iters, tangent_normals=None, unit=True):
    B = A.copy(); B.data[:] = 1.0
    W = sp.diags(1.0 / np.maximum(np.asarray(B.sum(1)).ravel(), 1)) @ B
    for _ in range(iters):
        X = 0.5 * X + 0.5 * (W @ X)
        if tangent_normals is not None:
            X = X - (X * tangent_normals).sum(1)[:, None] * tangent_normals
        if unit and X.ndim == 2:
            X = X / np.maximum(1e-12, np.linalg.norm(X, axis=1))[:, None]
    return X


def submesh(V, F, keep_faces, largest=True):
    Fk = F[keep_faces]
    u = np.unique(Fk.ravel()); rm = -np.ones(len(V), np.int64); rm[u] = np.arange(len(u))
    V2, F2 = V[u], rm[Fk]
    if largest:
        nc, lab = connected_components(graph(V2, F2, weighted=False), directed=False)
        big = np.argmax(np.bincount(lab))
        kf = (lab[F2] == big).all(1)
        return submesh(V2, F2, kf, largest=False)
    return V2, F2


def cyl(P):
    d = np.atleast_2d(P) - AX
    return np.hypot(d[:, 0], d[:, 2]), np.arctan2(d[:, 0], d[:, 2]), np.atleast_2d(P)[:, 1]


def rodrigues(v, k, th):
    return v * np.cos(th) + np.cross(k, v) * np.sin(th) + k * (k @ v) * (1 - np.cos(th))


# =================================================================================== A. the sculpt
class Sculpt:
    def __init__(self, glb):
        g = hg.Glb(glb)
        self.g = g
        mh, m4 = g.mesh_by_material("root.1"), g.mesh_by_material("root.4")
        P1, N1, UV1, T1 = g.mesh(mh)
        self.VH, self.FH, _ = hg.weld(P1, T1)
        cen = self.VH[self.FH].mean(1)
        P4, N4, UV4, T4 = g.mesh(m4)
        self.V4, self.F4, _ = hg.weld(P4, T4)
        cen4 = self.V4[self.F4].mean(1)
        self.body = [hg.weld(*g.mesh(g.mesh_by_material(n))[::3])[:2] for n in ("root.0", "root.3")]

        # 1) which hair triangles are the sculpt's OUTER surface (not in / within 3 mm of head or body)
        contact = np.zeros(len(cen), bool)
        for V, F in self.body + [(self.V4, self.F4)]:
            _, _, d = hg.TriGrid(V, F).closest(cen, maxr=8)
            contact |= hg.ZParity(V, F).inside(cen) | (d < 0.003)
        self.cO = cen[~contact]
        log("sculpt: %d hair triangles, %d outer" % (len(cen), (~contact).sum()))

        # 2) the sculpt's own flow on its outer surface (only its azimuthal drift is used, below)
        vn = hg.vertex_normals(self.VH, self.FH)
        dmin, dmax, kmn, kmx = hg.principal_dirs(self.VH, vn, 0.008, max_k=96)
        used = np.unique(self.FH[~contact].ravel()); rm = -np.ones(len(self.VH), np.int64); rm[used] = np.arange(len(used))
        VO, FO = self.VH[used], rm[self.FH[~contact]]
        nc, lab = connected_components(graph(VO, FO, weighted=False), directed=False)
        big = np.argmax(np.bincount(lab)); kf = (lab[FO] == big).all(1)
        u2 = np.unique(FO[kf].ravel()); rm2 = -np.ones(len(VO), np.int64); rm2[u2] = np.arange(len(u2))
        self.VO, self.FO = VO[u2], rm2[FO[kf]]
        oidx = used[u2]
        nO = hg.vertex_normals(self.VO, self.FO)
        aniso = (np.abs(kmx[oidx]) - np.abs(kmn[oidx])) / (np.abs(kmx[oidx]) + np.abs(kmn[oidx]) + 1e-9)
        prior = self._sculpt_prior(self.VO, nO)
        dm = dmin[oidx]
        sgn = np.sign((dm * prior).sum(1)); sgn[sgn == 0] = 1
        w = np.clip((aniso - 0.35) / 0.35, 0, 1) * np.clip((np.abs((dm * prior).sum(1)) - 0.25) / 0.35, 0, 1)
        w *= 1.0 - 0.85 * np.clip((self.VO[:, 1] - 1.70) / 0.06, 0, 1)       # the crown: headband / bow
        flow = w[:, None] * dm * sgn[:, None] + (1 - w[:, None]) * prior
        flow = flow - (flow * nO).sum(1)[:, None] * nO
        flow /= np.maximum(1e-9, np.linalg.norm(flow, axis=1))[:, None]
        self.flowO = smooth_field(flow, graph(self.VO, self.FO), 25, tangent_normals=nO)

        # 3) the scalp: head triangles inside the hair shell, minus the ears, grown 6 mm past the ledge
        scalp_in = hg.ZParity(self.VH, self.FH).inside(cen4)
        self.scalp_in = scalp_in
        ear = np.zeros(len(self.F4), bool)
        for e in EARS: ear |= np.linalg.norm(cen4 - e, axis=1) < EAR_R
        dd, _ = cKDTree(self.V4[np.unique(self.F4[scalp_in & ~ear].ravel())]).query(cen4, k=1)
        grow = (dd < 0.006) & ~ear & (cen4[:, 1] > 1.70)
        self.VS, self.FS = submesh(self.V4, self.F4, (scalp_in | grow) & ~ear)
        self.nS = hg.vertex_normals(self.VS, self.FS)
        self.areaS = 0.5 * np.linalg.norm(np.cross(self.VS[self.FS[:, 1]] - self.VS[self.FS[:, 0]],
                                                   self.VS[self.FS[:, 2]] - self.VS[self.FS[:, 0]]), axis=1)
        # the stack field: 0 at the part, 1 at the downstream hairline (not the forehead's)
        e = np.sort(np.concatenate([self.FS[:, [0, 1]], self.FS[:, [1, 2]], self.FS[:, [2, 0]]]), 1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        bnd = np.unique(ue[cnt == 1].ravel())
        a, b = LAM_PART
        s = np.clip(((self.VS - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
        dpart = np.linalg.norm(self.VS - (a + s[:, None] * (b - a)), axis=1)
        down = bnd[~((self.VS[bnd, 2] > 0.06) & (self.VS[bnd, 1] > 1.66))]
        AS = graph(self.VS, self.FS)
        dP = dijkstra(AS, directed=False, indices=np.nonzero(dpart < 0.006)[0], min_only=True)
        dD = dijkstra(AS, directed=False, indices=down, min_only=True)
        self.lam = dP / np.maximum(1e-9, dP + dD)
        log("scalp: %d triangles, %.4f m^2 (grown %d past the forehead ledge)"
            % (len(self.FS), self.areaS.sum(), (grow & ~scalp_in).sum()))

        # 4) the sculpt's headband and bow: its near-black texels (the band's centreline seats ours)
        tex = g.base_color_image(mh).astype(float) / 255.0
        H, W, _ = tex.shape
        c = UV1[T1].mean(1)
        lum = tex[np.clip((c[:, 1] * H).astype(int), 0, H - 1), np.clip((c[:, 0] * W).astype(int), 0, W - 1)] \
            @ np.array([0.2126, 0.7152, 0.0722])
        cc = P1[T1].mean(1)
        dark = (lum < 0.45) & (cc[:, 1] > 1.60) & (np.abs(cc[:, 0] - 0.01) < 0.2) & (cc[:, 2] > -0.06) & (cc[:, 2] < 0.14)
        self.band_pts = cc[dark & (cc[:, 1] <= 1.825)]
        log("sculpt headband: %d triangles (bow %d)" % (dark.sum(), (dark & (cc[:, 1] > 1.825)).sum()))

    @staticmethod
    def _sculpt_prior(P, N):
        a, b = LAM_PART
        s = np.clip(((P - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
        d = P - (a + s[:, None] * (b - a))
        d /= np.maximum(1e-9, np.linalg.norm(d, axis=1))[:, None]
        below = np.clip((1.74 - P[:, 1]) / 0.12, 0, 1)
        d = d * (1 - 0.6 * below[:, None]) + np.array([0, -1.0, 0]) * (0.6 * below[:, None])
        d = d - (d * N).sum(1)[:, None] * N
        return d / np.maximum(1e-9, np.linalg.norm(d, axis=1))[:, None]


# =================================================================================== B. the guides
class Groom:
    YS = np.arange(1.20, 1.95, 0.01)
    NPH = 72

    def __init__(self, sc):
        self.sc = sc
        V4, F4 = sc.V4, sc.F4
        # the head-phase surface: a Taubin-smoothed head (ears flattened, skull kept)
        W = sp.diags(1.0 / np.maximum(1, np.asarray(graph(V4, F4, False).sum(1)).ravel())) @ graph(V4, F4, False)
        X = V4.copy()
        for _ in range(60):
            for f in (0.55, -0.58):
                X = X + f * (W @ X - X)
        self.V4s = X
        self.n4s = hg.vertex_normals(X, F4)
        cen = X[F4].mean(1)
        _kd = cKDTree(sc.VS)
        dc, _ = _kd.query(V4[F4].mean(1), k=1)
        ax = np.abs(cen[:, 0] - XMID)
        # ... over the scalp, a strip of forehead under the front hairline (widest at the temples,
        # where the photos' hair sweeps down over the corners; crossed, never rooted on), and the
        # ring below the scalp behind the ears (walked to where the skull turns under)
        over = 0.004 + 0.034 * np.clip((ax - 0.03) / 0.05, 0, 1)
        overhang = (~sc.scalp_in) & (dc < over) & (cen[:, 2] > 0.03) & (cen[:, 1] > 1.715)
        ring = (cen[:, 1] > 1.52) & (self.n4s[F4].mean(1)[:, 2] < 0.35) & (cen[:, 2] < 0.0)
        self.GH = hg.TriGrid(X, F4[sc.scalp_in | overhang | ring])
        self.flowH = self.prior(X, self.n4s)
        d, i = _kd.query(V4, k=1)
        self.PHI4 = np.where(d < 0.05, sc.lam[i], 1.0)
        self._profiles()
        # the ears: what the Taubin smoothing flattened (moved > 2.5 mm) near each ear centre, as
        # (lowest y, highest y, front and back azimuth in degrees) keyed by side (sign of the azimuth).
        # They are big at this 1.9 m scale -- ~9 cm tall, ~40 degrees round -- and not level: the
        # head is tilted, her left ear sits 4 cm higher than her right.
        disp = np.linalg.norm(V4 - X, axis=1)
        self.ears = {}
        for e in EARS:
            m = (np.linalg.norm(V4 - e, axis=1) < 0.05) & (disp > 0.0025)
            _, pe, ye = cyl(V4[m])
            self.ears[1 if pe.mean() >= 0 else -1] = (ye.min(), ye.max(),
                                                      np.degrees(np.abs(pe)).min(), np.degrees(np.abs(pe)).max())
        log("ears: " + "; ".join("%s y %.3f..%.3f az %.0f..%.0f" % ("her left" if s > 0 else "her right", *v)
                                 for s, v in sorted(self.ears.items())))

    @staticmethod
    def prior(P, N):
        """The hairstyle's flow over the head: away from the part; in front, down over the forehead
        corners; at the temples back over / behind the ears; below the ears, down."""
        a, b = FLOW_PART
        s = np.clip(((P - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
        d = P - (a + s[:, None] * (b - a))
        d /= np.maximum(1e-9, np.linalg.norm(d, axis=1))[:, None]
        below = np.clip((1.72 - P[:, 1]) / 0.10, 0, 1)
        d = d * (1 - 0.7 * below[:, None]) + np.array([0, -1.0, 0]) * (0.7 * below[:, None])
        ax = np.abs(P[:, 0] - XMID)
        front_w = np.clip((P[:, 2] - 0.04) / 0.05, 0, 1) * np.clip((ax - 0.030) / 0.03, 0, 1)
        d = d + (0.65 * front_w)[:, None] * np.array([0, -1.0, 0])
        d /= np.maximum(1e-9, np.linalg.norm(d, axis=1))[:, None]
        beta = 1.8 * np.clip((ax - 0.055) / 0.04, 0, 1) * np.clip((P[:, 2] + 0.015) / 0.05, 0, 1) \
            * np.clip((1.785 - P[:, 1]) / 0.05, 0, 1)
        d = d + beta[:, None] * np.array([0, -0.15, -1.0])
        d /= np.maximum(1e-9, np.linalg.norm(d, axis=1))[:, None]
        d = d - (d * N).sum(1)[:, None] * N
        return d / np.maximum(1e-9, np.linalg.norm(d, axis=1))[:, None]

    # ---- curtain profiles
    def _grid_max(self, r, ph, y):
        G = np.full((len(self.YS), self.NPH), np.nan)
        iy = np.clip(np.round((y - self.YS[0]) / 0.01).astype(int), 0, len(self.YS) - 1)
        ip = np.round((ph + np.pi) / (2 * np.pi) * self.NPH).astype(int) % self.NPH
        np.fmax.at(G, (iy, ip), r)
        return G

    def _fill_circ(self, G):
        G = G.copy(); n = self.NPH
        for j in range(G.shape[0]):
            ok = np.isfinite(G[j])
            if ok.sum() == 0: continue
            idx = np.nonzero(ok)[0]
            G[j] = np.interp(np.arange(n), np.concatenate([idx - n, idx, idx + n]), np.tile(G[j][idx], 3))
        return G

    def _smooth_circ(self, G, sy, sp_):
        n = self.NPH
        Gp = gaussian_filter(np.concatenate([G[:, -n // 2:], G, G[:, :n // 2]], 1), (sy, sp_), mode="nearest")
        return Gp[:, n // 2:n // 2 + n]

    @staticmethod
    def _fill_rows(G):
        """Rows with no data at all (below the sculpt's hem, above its crown) take the nearest row that
        has some: a strand that hangs a little past the sculpt keeps the curtain it was on instead of
        collapsing onto the body in one step."""
        G = G.copy()
        ok = np.nonzero(np.isfinite(G).any(1))[0]
        if len(ok) == 0: return G
        for j in range(G.shape[0]):
            if not np.isfinite(G[j]).any():
                G[j] = G[ok[np.argmin(np.abs(ok - j))]]
        return G

    def _profiles(self):
        sc = self.sc
        r_, p_, y_ = cyl(sc.cO)
        self.Rout = self._smooth_circ(self._fill_rows(self._fill_circ(self._grid_max(r_, p_, y_))), 2.0, 2.0)
        rb, pb, yb = cyl(np.concatenate([sc.body[0][0], sc.body[1][0]]))
        rh, ph, yh = cyl(sc.V4)
        Rin = np.fmax(self._grid_max(rb, pb, yb), self._grid_max(rh, ph, yh))
        self.Rin = self._smooth_circ(np.nan_to_num(self._fill_circ(Rin), nan=0.03), 1.5, 1.0) + CLEAR
        n = self.NPH
        ip = np.round((p_ + np.pi) / (2 * np.pi) * n).astype(int) % n
        ye = np.full(n, np.nan)
        for k in range(n):
            m = ip == k
            if m.any(): ye[k] = np.percentile(y_[m], 1)
        ok = np.nonzero(np.isfinite(ye))[0]
        ye = np.interp(np.arange(n), np.concatenate([ok - n, ok, ok + n]), np.tile(ye[ok], 3))
        self.yend = gaussian_filter(np.concatenate([ye, ye, ye]), 1.5)[n:2 * n]
        # the sculpt's azimuthal drift (dphi per metre of descent)
        rO, pO, yO = cyl(sc.VO)
        ephi = np.stack([np.cos(pO), np.zeros_like(pO), -np.sin(pO)], 1)
        vdesc = -sc.flowO[:, 1]
        om = (sc.flowO * ephi).sum(1) / np.maximum(0.05, rO)
        ok = vdesc > 0.3
        D = np.zeros((len(self.YS), n)); Wt = np.zeros((len(self.YS), n))
        iy = np.clip(np.round((yO - self.YS[0]) / 0.01).astype(int), 0, len(self.YS) - 1)
        ipO = np.round((pO + np.pi) / (2 * np.pi) * n).astype(int) % n
        np.add.at(D, (iy[ok], ipO[ok]), (om / np.maximum(vdesc, 0.3))[ok]); np.add.at(Wt, (iy[ok], ipO[ok]), 1.0)
        self.DRIFT = np.clip(self._smooth_circ(D, 3, 3) / np.maximum(1e-6, self._smooth_circ(Wt, 3, 3)), -6, 6)

    def _lookup(self, G, y, ph):
        fy = np.clip((y - self.YS[0]) / 0.01, 0, len(self.YS) - 1.001)
        fp = (ph + np.pi) / (2 * np.pi) * self.NPH
        j = int(fy); a = fy - j
        k0 = int(np.floor(fp)) % self.NPH; k1 = (k0 + 1) % self.NPH; b = fp - np.floor(fp)
        return (1 - a) * ((1 - b) * G[j, k0] + b * G[j, k1]) + a * ((1 - b) * G[j + 1, k0] + b * G[j + 1, k1])

    @staticmethod
    def _circ(tab, ph):
        f = (ph + np.pi) / (2 * np.pi) * len(tab)
        k0 = int(np.floor(f)) % len(tab); k1 = (k0 + 1) % len(tab); b = f - np.floor(f)
        return (1 - b) * tab[k0] + b * tab[k1]

    # ---- the ears: the hair passes behind them
    def ear_weight(self, ph, y):
        """How strongly a point at (azimuth ph, height y) is held behind the ear on its side: 1 level
        with the ear, ramping in over EAR_ABOVE above it and out over EAR_RELEASE below its lobe. 0 for
        anything already behind the ear or well forward of it (the face)."""
        ear = self.ears.get(1 if ph >= 0 else -1)
        if ear is None: return 0.0, ph
        ylo, yhi, alo, ahi = ear
        back = np.radians(ahi + EAR_MARGIN)
        if abs(ph) >= back or abs(ph) < np.radians(40): return 0.0, back
        return float(smoothstep(yhi + EAR_ABOVE, yhi, y) * smoothstep(ylo - EAR_RELEASE, ylo, y)), back

    def ear_push(self, ph, y):
        """Order-preserving: azimuths from 40 degrees to EAR_SPREAD past the ear's back edge are remapped
        onto just that band behind the ear, so the hair tucked behind it spreads over 25 degrees instead
        of piling onto one line (from behind, that line read as a ridge across her back)."""
        ear = self.ears.get(1 if ph >= 0 else -1)
        if ear is None: return ph
        ylo, yhi, alo, ahi = ear
        a = abs(ph)
        back = np.radians(ahi + EAR_MARGIN); b2 = back + np.radians(EAR_SPREAD); a0 = np.radians(40)
        if a >= b2 or a < a0: return ph
        w = float(smoothstep(yhi + EAR_ABOVE, yhi, y) * smoothstep(ylo - EAR_RELEASE, ylo, y))
        if w <= 0: return ph
        f = back + (a - a0) / (b2 - a0) * (b2 - back)
        return np.sign(ph) * (a + w * (f - a))

    def in_ear_zone(self, P):
        """For clipping the head phase: the first point that comes level with an ear in front of its
        back edge hands the strand over to the hang, which takes it round behind."""
        _, pp, yy = cyl(P)
        return np.array([self.ear_weight(a, b)[0] > 0 for a, b in zip(pp, yy)])

    # ---- one strand's fall from the head, before its lock's waves and curl
    def hang_base(self, xh, lamv, lk, dy=0.003, blend=(0.10, 0.16)):
        r0, p0, y0 = [v[0] for v in cyl(xh)]
        ph, y = p0, y0
        # the side hair must not drape over the shoulder top: its outer layer comes FORWARD in front
        # of the shoulder (the photos' face-framing locks), the rest slides BACK behind it (136 degrees:
        # at 128 it still rode the shoulder, and from behind the hair flared past her arms); both turn
        # 6-22 cm below the hand-over, under the jaw -- and while level with an ear, all of it is held
        # behind the ear (ear_push, applied on top so the turn's own state is not disturbed)
        dg = abs(np.degrees(p0))
        turn = np.clip(1 - abs(dg - 100) / 28, 0, 1)
        front = turn * np.clip((0.28 - lamv) / 0.15, 0, 1)
        target = front * np.sign(p0) * np.radians(55) + (1 - front) * np.sign(p0) * np.radians(max(dg, 136))
        out = lk["out_front"] if front > 0.5 else lk["out"]
        # the SIDES flare out behind the ears fast -- from the front the photos' face is framed by
        # hair that is tucked behind the ears but stands well out to the side -- while the centre
        # back eases out slowly and keeps less volume level with the ears (a smooth crown from behind)
        sidew = 1.0 - smoothstep(140, 165, dg)
        bl = blend[0] * sidew + blend[1] * (1 - sidew)
        # the turn starts no higher than the lobe of the ear on this side: above it the hair is held
        # behind the ear, and a turn already done by then would snap forward in one step on release
        ear = self.ears.get(1 if p0 >= 0 else -1)
        y_turn = min(y0 - 0.06, ear[0]) if ear is not None else y0 - 0.06
        yend = self._circ(self.yend, p0)
        y_stop = (yend + lk["dL"] - HEM_BACK * smoothstep(115, 170, dg) + 0.03 * lamv
                  + lk["rho"] * (0.6 if out else 1.0))
        pts = [xh.copy()]
        k = 0
        while y > y_stop and k < 500:
            k += 1
            y -= dy
            dd = np.clip((y0 - y) / 0.04, 0, 1)
            ph += 0.5 * (1 - turn) * self._lookup(self.DRIFT, y, ph) * dy * dd
            if turn > 0:
                a = smoothstep(0, 1, (y_turn - y) / 0.13)
                ph = (1 - turn) * ph + turn * ((1 - a) * p0 + a * target)
            phe = self.ear_push(ph, y)
            rin = self._lookup(self.Rin, y, phe); rout = self._lookup(self.Rout, y, phe)
            if not np.isfinite(rout): rout = rin                     # below the sculpt's hem
            kap = ((KAPPA_SIDE * sidew + KAPPA * (1 - sidew)) * (1 - TAPER[2] * smoothstep(TAPER[0], TAPER[1], y))
                   * (1 - UPPER[2] * (1 - sidew) * (1 - smoothstep(UPPER[0], UPPER[1], y))))
            rt = rin + (1 - lamv) * kap * max(0.004, rout - rin)
            a = smoothstep(0, 1, (y0 - y) / bl)
            r = max((1 - a) * r0 + a * rt, rin)
            pts.append(np.array([AX[0] + r * np.sin(phe), y, AX[2] + r * np.cos(phe)]))
        return np.array(pts), out, front

    def finish(self, base, centre, lk, out, front, rng):
        """A lock's shape on one of its guides: gather toward the lock's centre line (azimuth as a
        function of height) by the tip, the lock's S-waves, then its curl."""
        pts = base
        if len(pts) > 3:
            rr, pp, yy = cyl(pts)
            frac = np.clip((yy[0] - yy) / max(1e-6, yy[0] - yy[-1]), 0, 1)
            c = CONVERGE * smoothstep(0.2, 1.0, frac)
            tgt = centre(yy)
            ok = np.isfinite(tgt)
            dphi = np.where(ok, np.angle(np.exp(1j * (np.where(ok, tgt, 0) - pp))), 0)
            pp = pp + c * dphi
            pp = np.array([self.ear_push(a, b) for a, b in zip(pp, yy)])
            pts = np.stack([AX[0] + rr * np.sin(pp), yy, AX[2] + rr * np.cos(pp)], 1)
            ss = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            er = np.stack([np.sin(pp), np.zeros_like(pp), np.cos(pp)], 1)
            ephi = np.stack([np.cos(pp), np.zeros_like(pp), -np.sin(pp)], 1)
            # exactly the lock's wave (no per-guide jitter): a strand follows ONE guide, so neighbouring
            # guides that differ show as cell boundaries -- short crossing tufts, a crimped look
            amp = lk["amp"] * np.clip(ss / 0.10, 0, 1)
            w = 2 * np.pi * ss / lk["wlen"] + lk["wph"]
            pts = pts + (amp * np.sin(w))[:, None] * ephi + (0.45 * amp * np.cos(w))[:, None] * er
            rr, pp, yy = cyl(pts)
            pp = np.array([self.ear_push(a, b) for a, b in zip(pp, yy)])
            rr = np.maximum(rr, [self._lookup(self.Rin, a_, b_) for a_, b_ in zip(yy, pp)])
            pts = np.stack([AX[0] + rr * np.sin(pp), yy, AX[2] + rr * np.cos(pp)], 1)
        if len(pts) >= 3:
            # the curl, a barrel roll UNDER (toward the body) or OUT, in the vertical plane through the
            # body axis: about the horizontal tangent, starting from a mostly-downward heading. Taking the
            # axis from the strand's own end heading let a wave that left it near horizontal lay the
            # loop flat -- tails jutting 15 cm out in front of her at the hem.
            rho = lk["rho"]
            th_max = lk["th"] + 0.8 * front
            L_c = rho * th_max
            _, pl, _ = cyl(pts[-1])
            ephi = np.array([np.cos(pl[0]), 0, -np.sin(pl[0])])
            axis = -ephi if out else ephi
            d = pts[-1] - pts[-3]; d = d - (d @ axis) * axis; d /= max(1e-9, np.linalg.norm(d))
            d = 0.35 * d + 0.65 * np.array([0.0, -1.0, 0.0]); d = d - (d @ axis) * axis; d /= np.linalg.norm(d)
            n_c = max(4, int(L_c / 0.004))
            P = pts[-1].copy(); extra = []
            for j in range(1, n_c + 1):
                P = P + rodrigues(d, axis, th_max * (j / n_c) ** 1.15) * (L_c / n_c)
                extra.append(P.copy())
            pts = np.concatenate([pts, np.array(extra)])
        return pts

    @staticmethod
    def lock_params(seed, lid, centre_deg):
        """One lock's shared shape. Its curl rolls UNDER at the centre back (the photos' hem turns in),
        more often OUT at the sides and over the shoulders, and out for the face-framing front layer."""
        g = np.random.default_rng(seed * 1000 + lid)
        dgc = abs(centre_deg)
        side = smoothstep(95, 120, dgc) * (1 - smoothstep(140, 160, dgc))
        return dict(wph=g.uniform(0, 2 * np.pi), amp=g.uniform(*WAVE_AMP), wlen=g.uniform(*WAVE_LEN),
                    dL=g.uniform(-0.03, 0.03), rho=g.uniform(0.022, 0.034), th=g.uniform(2.4, 4.0),
                    out=bool(g.random() < 0.15 + 0.25 * side), out_front=bool(g.random() < 0.75))

    def build(self, spacing, seed=1, h=0.003, sigma=0.010):
        sc = self.sc
        rng = np.random.default_rng(seed)
        R, rt = hg.sample_roots(sc.VS, sc.FS, sc.areaS, spacing, seed)
        lr = hg.interp_vert(sc.lam, sc.FS, rt, R, sc.VS)
        cp, ct, _ = self.GH.closest(R)
        out, tri, cnt, why = hg.trace_surface(self.GH, self.flowH, cp, ct, h=h, maxlen=0.6, maxpts=300)
        # 1) every guide's head phase, handed over where the skull turns under -- or earlier, where it
        #    comes level with an ear in front of the ear's back edge (the hang takes it round behind)
        heads = []
        n_ear = 0
        for i in range(len(R)):
            k = cnt[i]
            pts = out[i, :k]; tt = tri[i, :k]
            nn = hg.interp_vert(self.n4s, self.GH.F, tt, pts, self.GH.V); nn /= np.linalg.norm(nn, axis=1)[:, None]
            det = np.nonzero((nn[:, 1] < -0.15) & (pts[:, 1] < 1.70))[0]      # the skull turns under
            kd = max(det[0] + 1 if len(det) else k, 2)
            if R[i][2] > 0.05 and np.linalg.norm(pts[min(kd, k) - 1] - pts[0]) < 0.015:
                continue                                   # a front-hairline root that fell straight off
            pts, nn, tt = pts[:kd], nn[:kd], tt[:kd]
            # lying on the hair stacked beneath: every strand rooted between lam and the local stack
            # value phi runs underneath; concave so the hair gains body quickly off the part
            phi = hg.interp_vert(self.PHI4, self.GH.F, tt, pts, self.GH.V)
            height = np.maximum.accumulate(CAP_HEAD * np.clip(phi - lr[i], 0, 1) ** 0.6 + 0.0012)
            q = pts + nn * height[:, None]
            q[0] = R[i]
            ez = np.nonzero(self.in_ear_zone(q))[0]
            if len(ez) and ez[0] < len(q) - 1:
                q = q[:max(ez[0] + 1, 2)]; n_ear += 1
            if q[-1][2] > 0.045 and q[-1][1] > 1.60:
                continue                                   # it would hang in front of the face
            heads.append((q, lr[i]))
        # 2) locks: sectors of LOCK_DEG round the body, by where each guide leaves the head
        g = np.random.default_rng(seed + 101)
        edges = [0.0]
        while edges[-1] < 360.0 - LOCK_DEG[1]:
            edges.append(edges[-1] + g.uniform(*LOCK_DEG))
        edges = np.array(edges); off = g.uniform(0, 360)
        p_hand = np.degrees(np.array([cyl(q[-1])[1][0] for q, _ in heads]))
        lock_of = np.searchsorted(edges, (p_hand - off) % 360.0, side="right") - 1
        centre_deg = {lid: ((edges[lid] + (edges[lid + 1] if lid + 1 < len(edges) else 360.0)) / 2 + off + 180) % 360 - 180
                      for lid in np.unique(lock_of)}
        LK = {lid: self.lock_params(seed, int(lid), centre_deg[lid]) for lid in centre_deg}
        # 3) each guide's fall, then every lock's centre line (its mean azimuth by height)
        bases = [self.hang_base(q[-1], l, LK[lock_of[j]]) for j, (q, l) in enumerate(heads)]
        yg = np.arange(1.10, 1.90, 0.004)
        centres = {}
        for lid in LK:
            S = np.zeros(len(yg)); C = np.zeros(len(yg)); Nn = np.zeros(len(yg))
            for j in np.nonzero(lock_of == lid)[0]:
                b = bases[j][0]
                if len(b) < 3: continue
                _, pp, yy = cyl(b)
                ok = (yg <= yy[0]) & (yg >= yy[-1])
                pu = np.interp(yg[ok], yy[::-1], np.unwrap(pp)[::-1])
                S[ok] += np.sin(pu); C[ok] += np.cos(pu); Nn[ok] += 1
            ang = np.where(Nn > 0, np.arctan2(S, C), np.nan)
            v = np.nonzero(np.isfinite(ang))[0]
            if len(v) == 0:
                centres[lid] = lambda y: np.full(np.shape(y), np.nan); continue
            a_v = np.unwrap(ang[v])                        # a back lock straddles +-180 degrees
            # a smooth centre line: members start and stop at different heights, and each one joining
            # or leaving the mean would otherwise put a kink into every strand pulled toward it
            full = np.interp(np.arange(len(yg)), v, a_v)   # held constant past its first / last member
            full = gaussian_filter1d(full, 0.03 / 0.004, mode="nearest")
            full[:v[0]] = np.nan; full[v[-1] + 1:] = np.nan
            centres[lid] = (lambda a_: (lambda y: np.interp(y, yg, a_, left=np.nan, right=np.nan)))(full)
        # 4) each guide takes its lock's shape
        guides = []
        for j, (q, l) in enumerate(heads):
            base, out, front = bases[j]
            lid = lock_of[j]
            hang = self.finish(base, centres[lid], LK[lid], out, front, rng)
            full = hg.smooth_polyline(np.concatenate([q, hang[1:]]), sigma)
            guides.append((full, l))
        log("guides: %d (from %d roots at %.1f mm; %d handed over at an ear) in %d locks"
            % (len(guides), len(R), 1000 * spacing, n_ear, len(LK)))
        return guides


# =================================================================================== C. the outputs
def resample(q, n):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, q[:, k]) for k in range(3)], 1)


def region_of(P):
    """The part runs from the forehead to the crown only: in front of the crown the groom is split
    into her right / left of it; behind, the back is one piece (split there too, the halves would
    open a parting all the way down her back)."""
    P = np.atleast_2d(P)
    return np.where(P[:, 2] < FLOW_PART[1][2] - 0.005, 2, np.where(P[:, 0] < XMID, 0, 1))


def write_scalp(sc, path, reg):
    Fs = sc.FS[region_of(sc.VS[sc.FS].mean(1)) == reg]
    u = np.unique(Fs.ravel()); rm = -np.ones(len(sc.VS), np.int64); rm[u] = np.arange(len(u))
    with open(path, "w") as f:
        f.write("# Alice2's scalp (%s): the head under the sculpted hair, ears cut out -- the groom's roots.\n"
                "# GENERATED by tools/alice2_hair.py\n" % REGION_DESC[reg])
        for p in sc.VS[u]: f.write("v %.6f %.6f %.6f\n" % tuple(p))
        for n in sc.nS[u]: f.write("vn %.6f %.6f %.6f\n" % tuple(n))
        for a, b, c in rm[Fs] + 1: f.write("f %d//%d %d//%d %d//%d\n" % (a, a, b, b, c, c))
    return 0.5 * np.linalg.norm(np.cross(sc.VS[Fs[:, 1]] - sc.VS[Fs[:, 0]], sc.VS[Fs[:, 2]] - sc.VS[Fs[:, 0]]), axis=1).sum()


def write_flyaways(f, guides, n=240, seed=5):
    """Stray fibres (every photo shows plenty): rooted with the hair, they run with a guide for a
    while and then lift off the surface in a loose arc -- mostly the outer layers, so they are seen."""
    rng = np.random.default_rng(seed)
    C0 = np.array([0.012, 1.66, 0.02])
    lam = np.array([l for q, l in guides])
    w = np.clip(0.6 - lam, 0.05, 1.0); w /= w.sum()
    f.write("\n# flyaways: stray fibres lifting off the surface\n")
    for _ in range(n):
        q = resample(guides[rng.choice(len(guides), p=w)][0], 64)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
        L = s[-1]
        if L < 0.08: continue
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
        if len(p) < 4: continue
        f.write("curve { material alice2_hair  radius 0.00014  radius_tip 0.00009  spline centripetal  segments 2\n")
        for pt in p: f.write("    point %.5f %.5f %.5f\n" % tuple(pt))
        f.write("}\n")


def write_guides(path, guides):
    """The guides alone, in a file of nothing but curves -- the kind the groom tool (`ftrace -groom`,
    groom.bat) can rewrite, so they can be edited by hand and saved. The fur, the scalps and the
    materials live in the groom file, which includes this one."""
    with open(path, "w") as f:
        f.write("# Alice2's hair GUIDES -- GENERATED by tools/alice2_hair.py. Edit them freely, e.g. in the groom\n"
                "# tool (groom.bat in the ftrace repo); rerun the tool to start over. Curve DEFINITIONS (no\n"
                "# material). Each piece of the scalp grows its own fur from its own set: front-right and\n"
                "# front-left of the centre part, and the back.\n\n")
        names = [[], [], []]
        for i, (q, l) in enumerate(guides):
            reg = int(region_of(q[0])[0])
            nm = "a2g_%s%04d" % (REGIONS[reg], i); names[reg].append(nm)
            f.write('curve "%s" {\n' % nm)
            for p in resample(q, NPTS): f.write("    point %.5f %.5f %.5f\n" % tuple(p))
            f.write("}\n")
        for reg in range(3):
            f.write('\ncurve "alice2_guides_%s" {\n' % REGIONS[reg])
            for nm in names[reg]: f.write('    curve "%s"\n' % nm)
            f.write("}\n")


def write_groom(path, guides, areas, count, radius, seed=7, guides_file="alice2_hair_guides.ftsl"):
    with open(path, "w") as f:
        f.write("# Alice2's real hair -- GENERATED by tools/alice2_hair.py. The guides are in their own file,\n"
                "# because the groom tool rewrites only files of curves; the scalps, the fibre and the fur are here.\n\n")
        f.write('include "%s"\n' % guides_file)
        f.write('\nmaterial "alice2_scalp_mat" { type diffuse reflect 0.5 }\n')
        for reg in range(3):
            f.write('mesh "alice2_scalp_%s" { file "alice2_scalp_%s.obj"  material alice2_scalp_mat  shape_only yes }\n'
                    % (REGIONS[reg], REGIONS[reg]))
        f.write("\n# synthetic doll fibre: eta 1.55, no cuticle tilt, no medulla; `reflect` calibrated against\n"
                "# the photos (see tools/alice2_hair.py)\n" + HAIR_MAT + "\n")
        tot = sum(areas)
        for reg in range(3):
            f.write('\nfur "alice2_hair_%s" {\n' % REGIONS[reg])
            f.write('    on "alice2_scalp_%s"   material alice2_hair\n' % REGIONS[reg])
            f.write('    count %d\n' % int(round(count * areas[reg] / tot)))
            # guide_blend 1: a strand is its nearest guide's shape, so neighbouring locks stay apart (any
            # blend across a lock boundary averages two locks' offsets into a strand that fills the gap)
            f.write('    guides "alice2_guides_%s"   guide_blend 1\n' % REGIONS[reg])
            f.write('    points %d   segments 2   spline centripetal\n' % NPTS)
            f.write('    radius %.6f  radius_tip %.6f\n' % (radius, radius * 0.8))
            f.write('    length_jitter 0.02   jitter 0.004\n')
            f.write('    clump 0.25   clump_size 0.020\n')
            f.write('    root_offset 0.0005\n')
            f.write('    seed %d\n}\n' % (seed + reg))
        write_flyaways(f, guides)


def _rounded_rect(w, t, n=10):
    r = t / 2
    pts = []
    for cx, a0 in ((w / 2 - r, -np.pi / 2), (-(w / 2 - r), np.pi / 2)):
        for k in range(n + 1):
            a = a0 + np.pi * k / n
            pts.append((cx + r * np.cos(a), r * np.sin(a)))
    return np.array(pts)


def _sweep(path, wdir, w, t, closed=False, taper=None):
    """A rounded-rectangle ribbon swept along `path`, its width along `wdir`. (V, N, F)."""
    n = len(path)
    tang = np.gradient(path, axis=0); tang /= np.linalg.norm(tang, axis=1)[:, None]
    wd = wdir - (wdir * tang).sum(1)[:, None] * tang; wd /= np.linalg.norm(wd, axis=1)[:, None]
    nd = np.cross(tang, wd)
    prof = _rounded_rect(w, t); m = len(prof)
    V, N = [], []
    for i in range(n):
        s = 1.0 if taper is None else taper[i]
        for (u, v) in prof:
            V.append(path[i] + wd[i] * u + nd[i] * v * s)
            cx = np.clip(u, -(w / 2 - t / 2), w / 2 - t / 2)
            d = np.array([u - cx, v])
            d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array([0.0, 1.0 if v >= 0 else -1.0])
            nn = wd[i] * d[0] + nd[i] * d[1]
            N.append(nn / np.linalg.norm(nn))
    V, N = np.array(V), np.array(N)
    F = []
    for i in range(n if closed else n - 1):
        i2 = (i + 1) % n
        for k in range(m):
            k2 = (k + 1) % m
            a, b, c, d = i * m + k, i * m + k2, i2 * m + k2, i2 * m + k
            F.append((a, b, c)); F.append((a, c, d))
    if not closed:
        for i, flip in ((0, True), (n - 1, False)):
            ci = len(V)
            V = np.vstack([V, path[i]]); N = np.vstack([N, -tang[i] if flip else tang[i]])
            for k in range(m):
                k2 = (k + 1) % m
                F.append((ci, i * m + k2, i * m + k) if flip else (ci, i * m + k, i * m + k2))
    return V, N, np.array(F)


def write_headband(gr, guides, out_path, width=0.022, thick=0.0028):
    """The band follows the sculpt's band (its dark triangles, by angle about the head centre) but
    sits on the GENERATED hair: along each ray from the head centre, just over the outermost guide
    (the fur is a blend of its nearest guides, so its surface is theirs); the ends dip into the hair.
    The bow -- two loops and a knot -- is sized from the photos (the sculpt's is twice as wide) and
    sits on her right of the part, as in the photos."""
    sc = gr.sc
    B = sc.band_pts
    th = np.degrees(np.arctan2(B[:, 0] - HEAD_C[0], B[:, 1] - HEAD_C[1]))
    ang, cp = [], []
    for a in np.arange(-78, 86, 4):
        m = np.abs(th - a) < 4
        if m.sum() > 8: ang.append(a); cp.append(np.median(B[m], 0))
    ang, cp = np.array(ang, float), np.array(cp)
    A = np.arange(-78, 84.1, 1.5)
    raw = gaussian_filter1d(np.stack([np.interp(A, ang, cp[:, k]) for k in range(3)], 1), 3, axis=0, mode="nearest")
    GHs = hg.TriGrid(gr.V4s, sc.F4)
    u = raw - HEAD_C; u /= np.linalg.norm(u, axis=1)[:, None]
    t, tri = GHs.raycast(np.tile(HEAD_C, (len(u), 1)), u, tmin=0.0, tmax=0.5)
    assert np.isfinite(t).all(), "a headband ray missed the head"
    foot = HEAD_C + u * t[:, None]
    nf = hg.interp_vert(gr.n4s, sc.F4, tri, foot, gr.V4s); nf /= np.linalg.norm(nf, axis=1)[:, None]
    kdg = cKDTree(np.concatenate([resample(q, NPTS) for q, l in guides]))
    top = t.copy()
    for i in range(len(u)):
        for r_ in np.arange(t[i], t[i] + 0.06, 0.001):
            if kdg.query_ball_point(HEAD_C + u[i] * r_, 0.006, return_length=True) > 0: top[i] = r_
    top = gaussian_filter1d(top, 2, mode="nearest")
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(foot, axis=0), axis=1))])
    e = np.minimum(s, s[-1] - s)
    path = HEAD_C + u * (top + 0.0015 + thick / 2 - 0.010 * np.clip(1 - e / 0.03, 0, 1) ** 2)[:, None]
    path = gaussian_filter1d(path, 1.5, axis=0, mode="nearest")
    tang = np.gradient(path, axis=0); tang /= np.linalg.norm(tang, axis=1)[:, None]
    parts = [_sweep(path, np.cross(nf, tang), width, thick, taper=np.clip(e / 0.012, 0.35, 1.0))]
    k = int(np.argmin(np.abs(A + 17.0)))
    K = path[k] + nf[k] * (thick * 0.8)
    t0, n0 = tang[k], nf[k]
    w0 = np.cross(n0, t0); w0 /= np.linalg.norm(w0)
    L, H = 0.050, 0.026
    for side in (-1, 1):
        a = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        uu = L * (1 - np.cos(a)) / 2
        vv = (H / 2) * np.sin(a) + (uu / L) * H * 0.45 + 0.006
        P = K + np.outer(side * uu, t0) + np.outer(vv, n0) + np.outer((uu / L) ** 2 * 0.012, -w0)
        parts.append(_sweep(P, np.tile(w0, (len(P), 1)), width * 0.95, thick * 0.6, closed=True))
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    Pk = K + np.outer(0.0135 * np.cos(a), w0) + np.outer(0.0095 * np.sin(a) + 0.007, n0)
    parts.append(_sweep(Pk, np.tile(t0, (len(Pk), 1)), 0.016, thick * 0.7, closed=True))
    with open(out_path, "w") as f:
        f.write("# Alice2's headband + bow: a satin ribbon seated on the generated hair. GENERATED by tools/alice2_hair.py\n")
        base = 0
        for V, N, F in parts:
            for p in V: f.write("v %.6f %.6f %.6f\n" % tuple(p))
            for n in N: f.write("vn %.6f %.6f %.6f\n" % tuple(n))
            for a_, b_, c_ in F + base + 1: f.write("f %d//%d %d//%d %d//%d\n" % (a_, a_, b_, b_, c_, c_))
            base += len(V)
    return s[-1]


def write_room(path, w=2048, h=1024):
    """An indoor room as the environment, like the photos': warm walls, a light floor, a ceiling
    glow, a daylight window front-left of her and a dimmer one behind her right. A large soft
    environment lights a pale coat from every side at low variance, where one small bright panel
    sparkles in every fibre. (EnvMap convention, src/studio.h: row 0 straight up, phi = atan2(z, x).)"""
    th = (np.arange(h) + 0.5) / h * np.pi
    ph = ((np.arange(w) + 0.5) / w - 0.5) * 2 * np.pi
    T, P = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(T) * np.cos(P), np.cos(T), np.sin(T) * np.sin(P)], -1)
    el = np.degrees(np.arcsin(np.clip(d[..., 1], -1, 1)))
    floor = smoothstep(-4, -12, el)[..., None]
    ceil = smoothstep(40, 52, el)[..., None]
    img = ((1 - floor - ceil) * 0.50 * np.array([1.0, 0.95, 0.88]) + floor * 0.40 * np.array([1.0, 0.97, 0.91])
           + ceil * 0.60 * np.array([1.0, 0.97, 0.92]))
    zen = np.degrees(np.arccos(np.clip(d[..., 1], -1, 1)))
    img = img + 2.2 * smoothstep(28, 12, zen)[..., None] * np.array([1.0, 0.96, 0.88])

    def window(center, half_w, half_h, rad, tint):
        c = np.array(center, float); c /= np.linalg.norm(c)
        right = np.cross([0, 1.0, 0], c); right /= np.linalg.norm(right)
        upv = np.cross(c, right)
        a = np.degrees(np.arctan2(d @ right, d @ c)); b = np.degrees(np.arctan2(d @ upv, d @ c))
        m = smoothstep(half_w + 2, half_w - 2, np.abs(a)) * smoothstep(half_h + 2, half_h - 2, np.abs(b)) * (d @ c > 0)
        m = m * (1 - 0.85 * (np.abs(a) < 1.0)) * (1 - 0.85 * (np.abs(b) < 0.8))      # mullions
        return rad * m[..., None] * np.array(tint)

    img = img + window((0.62, 0.20, 0.76), 26, 22, 3.6, (0.95, 1.0, 1.08)) \
        + window((-0.70, 0.25, -0.66), 20, 18, 1.4, (1.0, 0.98, 0.95))
    img = img.astype(np.float32)
    m = img.max(-1)
    e = np.ceil(np.log2(np.maximum(m, 1e-32)))
    sc = np.where(m > 1e-32, 256.0 / np.exp2(e), 0.0)
    rgb = np.clip(np.floor(img * sc[..., None]), 0, 255).astype(np.uint8)
    ex = np.where(m > 1e-32, e + 128, 0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n# alice2 room environment, GENERATED by tools/alice2_hair.py\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(("-Y %d +X %d\n" % (h, w)).encode())
        f.write(np.concatenate([rgb, ex[..., None]], -1).tobytes())


SCENE = '''# Alice2 with REAL HAIR -- GENERATED by tools/alice2_hair.py (from the sculpted doll, whose hair
# primitive root.1 is skipped here and replaced by a strand groom). What it needs sits beside it --
#   alice2_hair_groom.ftsl (guides + fur + flyaways + material), alice2_scalp_*.obj, alice2_headband.obj,
#   alice2_room.hdr -- except the doll itself: {GLB_REL}
#
# Render (mode R; a pale coat needs many bounces -- its colour lives in long paths):
#   ftrace -in alice2_real_hair.ftsl -camera front -mode R -device gpu -spp 1024 -max-bounce 128 -denoise
#
# Cameras: front / three (3/4) / side / back / crown / full (the whole doll), and p1..p5 -- roughly the
# viewpoints of the five reference photos in images/alice.

camera "front" { eye 0.016 1.56 1.45    look_at 0.016 1.56 0.02   up 0 1 0  fov_y 30  film { res 1000 1000 } }
camera "three" { eye 0.78 1.66 1.05     look_at 0.016 1.57 0.0    up 0 1 0  fov_y 30  film { res 1000 1000 } }
camera "side"  { eye -1.45 1.60 -0.03   look_at 0.016 1.57 -0.03  up 0 1 0  fov_y 30  film { res 1000 1000 } }
camera "back"  { eye 0.016 1.60 -1.45   look_at 0.016 1.55 -0.03  up 0 1 0  fov_y 30  film { res 1000 1000 } }
camera "crown" { eye 0.10 2.05 0.75     look_at 0.016 1.74 0.04   up 0 1 0  fov_y 30  film { res 1000 1000 } }
camera "full"  { eye 0.9 1.25 3.6       look_at 0.016 0.98 0.0    up 0 1 0  fov_y 36  film { res 900 1200 } }
camera "p1" { eye 0.10 1.62 1.30    look_at 0.016 1.56 0.02  up 0 1 0  fov_y 30  film { res 900 900 } }
camera "p2" { eye 0.72 1.74 0.98    look_at 0.016 1.60 0.02  up 0 1 0  fov_y 30  film { res 900 900 } }
camera "p3" { eye 0.70 1.68 -1.00   look_at 0.016 1.56 0.00  up 0 1 0  fov_y 30  film { res 900 900 } }
camera "p4" { eye 0.05 1.62 -1.30   look_at 0.016 1.55 0.00  up 0 1 0  fov_y 30  film { res 900 900 } }
camera "p5" { eye -1.25 1.64 0.05   look_at 0.016 1.60 0.02  up 0 1 0  fov_y 28  film { res 900 900 } }

# the doll, with the sculpted hair (its own glTF primitive, material "root.1") left out
material "alice_fallback" { type diffuse reflect 0.5 }
mesh "alice2" { file "{GLB_REL}"  material alice_fallback  skip_material root.1 }

include "alice2_hair_groom.ftsl"

# the black satin headband and bow
material "ribbon_body" { type diffuse reflect 0.022 }
material "ribbon" { type layered  ior 1.5  coat { roughness 0.32 }  layer "ribbon_body" 1.0 }
mesh "alice2_headband" { file "alice2_headband.obj"  material ribbon }

# lit like the photos: indoors, a daylight window front-left of her
light env { file "alice2_room.hdr" }
material "floor" { type diffuse reflect 0.55 }
quad { corner -3 0 -3  u 6 0 0  v 0 0 6  material floor }
'''

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glb", default=DEFAULT_GLB)
    ap.add_argument("--out", default=None,
                    help="output folder (default: hair_opus5.5 beside the GLB's `model` folder, else the GLB's folder)")
    ap.add_argument("--count", type=int, default=90000, help="strands (all three fur blocks together)")
    ap.add_argument("--radius", type=float, default=0.00024, help="fibre root radius, metres")
    ap.add_argument("--spacing", type=float, default=0.0072, help="guide root spacing on the scalp, metres")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    glb_dir = os.path.dirname(os.path.abspath(a.glb))
    if a.out:
        out = os.path.abspath(a.out)
    elif os.path.basename(glb_dir).lower() == "model":
        out = os.path.join(os.path.dirname(glb_dir), "hair_opus5.5")
    else:
        out = glb_dir
    os.makedirs(out, exist_ok=True)
    try:
        glb_rel = os.path.relpath(os.path.abspath(a.glb), out).replace("\\", "/")
    except ValueError:                                   # --out on another drive: no relative path exists
        glb_rel = os.path.abspath(a.glb).replace("\\", "/")
    paths = dict(groom=os.path.join(out, "alice2_hair_groom.ftsl"), guides=os.path.join(out, "alice2_hair_guides.ftsl"),
                 headband=os.path.join(out, "alice2_headband.obj"),
                 room=os.path.join(out, "alice2_room.hdr"), scene=os.path.join(out, "alice2_real_hair.ftsl"))
    sc = Sculpt(a.glb)
    gr = Groom(sc)
    guides = gr.build(a.spacing, seed=a.seed)
    areas = [write_scalp(sc, os.path.join(out, "alice2_scalp_%s.obj" % REGIONS[r]), r) for r in range(3)]
    write_guides(paths["guides"], guides)
    write_groom(paths["groom"], guides, areas, a.count, a.radius)
    Lb = write_headband(gr, guides, paths["headband"])
    write_room(paths["room"])
    with open(paths["scene"], "w") as f:
        f.write(SCENE.replace("{GLB_REL}", glb_rel))
    log("wrote %s: %d guides, %d strands of radius %.2f mm on %.4f m^2 of scalp, headband %.2f m"
        % (out, len(guides), a.count, 1000 * a.radius, sum(areas), Lb))


if __name__ == "__main__":
    main()
