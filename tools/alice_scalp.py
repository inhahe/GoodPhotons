"""Segment the hair of meshes/alice.glb into a scalp mesh a groom can grow on.

    python tools/alice_scalp.py [--out meshes/alice_scalp.obj] [--glb meshes/alice.glb]

Alice is ONE fused mesh with ONE material (407 792 tris): the hair is sculpted into the same
shell as the head, and the two things one would reach for first both fail, measurably:

  * COLOUR cannot separate hair from face -- Meshy painted the forehead and the hair the same
    (hue 34 vs 36 deg, saturation 0.49 vs 0.52, value 0.91 vs 0.92 at 8k).
  * SURFACE STATISTICS cannot either -- the sculpted hair is SMOOTHER (5.3 deg mean dihedral)
    than the face (6.7 forehead, 15 cheeks), and the roughness map is flat (p10 0.33, p90 0.41).

What works is geometric and explicit: keep the warm-coloured triangles above the dress (the
blue / white dress and the dark bow ARE colour-separable), carve out a face ellipsoid placed off
the NOSE TIP (the most forward warm vertex on the head) and a throat box, then keep the largest
connected component -- after WELDING vertices by position, because Meshy duplicates every vertex
along every UV seam and index-connectivity shatters the hair into 53 pieces (welded: 8, with the
hair mass at ~91 600 tris, 0.775 m^2). Verified from five camera angles in png/alicehair/.

The output OBJ carries the glb's own vertex normals, so fur roots get smooth normals. Load it as
`mesh "alice_scalp" { file "meshes/alice_scalp.obj"  shape_only yes ... }` and grow `fur` on it;
the sculpted hair mass stays underneath as the volume, so no bald skull shows through.
"""
import argparse
import collections
import io
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def read_glb(path):
    d = open(path, "rb").read()
    _, _, length = struct.unpack_from("<III", d, 0)
    off, chunks = 12, []
    while off < length:
        clen, ctype = struct.unpack_from("<II", d, off); off += 8
        chunks.append((ctype, d[off:off + clen])); off += clen
    js = json.loads(chunks[0][1]); bin_ = chunks[1][1]
    acc, bv = js["accessors"], js["bufferViews"]
    CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def read(ai):
        a = acc[ai]; v = bv[a["bufferView"]]
        dt = CT[a["componentType"]]; n = NC[a["type"]]
        start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        arr = np.frombuffer(bin_, dtype=dt, count=a["count"] * n, offset=start)
        return arr.reshape(a["count"], n) if n > 1 else arr

    pr = js["meshes"][0]["primitives"][0]
    P = read(pr["attributes"]["POSITION"]).astype(np.float64)
    N = read(pr["attributes"]["NORMAL"]).astype(np.float64)
    UV = read(pr["attributes"]["TEXCOORD_0"]).astype(np.float64)
    I = read(pr["indices"]).astype(np.int64).reshape(-1, 3)
    mat = js["materials"][pr["material"]]; pbr = mat["pbrMetallicRoughness"]
    ti = pbr["baseColorTexture"]["index"]; ii = js["textures"][ti]["source"]; img = js["images"][ii]
    v = bv[img["bufferView"]]
    raw = bin_[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]
    return P, N, UV, I, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default="meshes/alice.glb")
    ap.add_argument("--out", default="meshes/alice_scalp.obj")
    args = ap.parse_args()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    P, N, UV, I, raw = read_glb(args.glb)
    tex = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    W = tex.shape[0]

    def sample(uv):
        u = np.clip(uv[:, 0] % 1, 0, .999999); v = np.clip(uv[:, 1] % 1, 0, .999999)
        return tex[(v * (W - 1)).astype(int), (u * (W - 1)).astype(int)].astype(np.float64) / 255.0

    # per-TRIANGLE colour: the median of the three vertices and the centroid, at full resolution
    # (a 1k downscale bleeds neighbouring atlas islands into every chart border)
    cv = sample(UV); cc = sample(UV[I].mean(1))
    tc = np.median(np.stack([cv[I[:, 0]], cv[I[:, 1]], cv[I[:, 2]], cc], 0), 0)
    mx, mn = tc.max(1), tc.min(1); val = mx; sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0)
    r, g, b = tc[:, 0], tc[:, 1], tc[:, 2]
    hue = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    cen = P[I].mean(1)
    warm = (hue > 15) & (hue < 65) & (sat > 0.25) & (val > 0.55)

    # the nose tip: the most forward warm vertex on the head (she faces +z)
    headv = np.zeros(len(P), bool); headv[I[warm]] = True
    cand_v = headv & (P[:, 1] > 0.68) & (P[:, 1] < 0.84) & (np.abs(P[:, 0]) < 0.06)
    nose = P[cand_v][np.argmax(P[cand_v][:, 2])]
    fc = nose + np.array([0.0, 0.015, -0.080]); fr = np.array([0.095, 0.125, 0.095])
    in_face = (((cen - fc) / fr) ** 2).sum(1) < 1.0
    throat = (np.abs(cen[:, 0]) < 0.06) & (cen[:, 1] < nose[1] - 0.09) & (cen[:, 1] > nose[1] - 0.24) & (cen[:, 2] > nose[2] - 0.10)
    cand = warm & (cen[:, 1] > 0.42) & ~in_face & ~throat
    print("nose tip %s; face ellipsoid centre %s; %d candidate tris (face carved %d, throat %d)"
          % (np.round(nose, 3), np.round(fc, 3), cand.sum(), (warm & in_face).sum(), (warm & throat).sum()))

    # weld by position, then the largest connected component
    key = np.round(P / 0.0003).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True); weld = weld.ravel()
    sel = np.flatnonzero(cand)
    parent = np.arange(weld.max() + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for t in sel:
        a, b_, c = weld[I[t]]; ra = find(a); parent[find(b_)] = ra; parent[find(c)] = ra
    roots = np.array([find(weld[I[t][0]]) for t in sel]); cnt = collections.Counter(roots.tolist())
    big = cnt.most_common(1)[0][0]; sel = sel[roots == big]
    T = P[I[sel]]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1).sum()
    q = P[np.unique(I[sel])]
    print("components %d; scalp = largest: %d tris, %.4f m^2, y %.2f..%.2f, |x| %.2f, z %.2f..%.2f"
          % (len(cnt), len(sel), area, q[:, 1].min(), q[:, 1].max(), np.abs(q[:, 0]).max(), q[:, 2].min(), q[:, 2].max()))

    used = np.unique(I[sel]); remap = -np.ones(len(P), dtype=np.int64); remap[used] = np.arange(len(used))
    L = ["# alice scalp: the hair mass of meshes/alice.glb, segmented by tools/alice_scalp.py",
         "# (warm-coloured, above the dress, outside a nose-tip face ellipsoid, largest welded component)"]
    L += ["v %.6f %.6f %.6f" % tuple(p) for p in P[used]] + ["vn %.5f %.5f %.5f" % tuple(n) for n in N[used]]
    for t in sel:
        a, b_, c = remap[I[t]] + 1; L.append("f %d//%d %d//%d %d//%d" % (a, a, b_, b_, c, c))
    io.open(args.out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("wrote %s (%d verts, %d tris)" % (args.out, len(used), len(sel)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
