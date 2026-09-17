"""Rescale a .glb uniformly by rewriting its ROOT node transforms -- the binary is untouched.

    python tools/glb_rescale.py <in.glb> <factor> [--out <out.glb>] [--height <metres>]

`--height` computes the factor from the file's current world-space height instead (the scene's
root nodes, transforms applied, tallest axis y). The original is copied to `<in>.orig.glb` once,
never overwritten, so the operation is reversible.

Why node transforms and not vertices: ftrace's glTF loader "bakes the glTF node transform
hierarchy (matrix or TRS) under the mesh block's own translate/rotate/scale", so a scale on every
root node is exactly equivalent to scaling the geometry, costs nothing, keeps accessor min/max
valid for whatever else reads the file, and leaves the buffers byte-identical.
"""
import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np


def read_glb(path):
    d = open(path, "rb").read()
    magic, ver, length = struct.unpack_from("<III", d, 0)
    if magic != 0x46546C67:
        raise SystemExit("%s: not a GLB" % path)
    off, chunks = 12, []
    while off < length:
        clen, ctype = struct.unpack_from("<II", d, off); off += 8
        chunks.append([ctype, d[off:off + clen]]); off += clen
    return chunks


def write_glb(path, chunks):
    body = b""
    for ctype, data in chunks:
        pad = (4 - len(data) % 4) % 4
        data = data + (b" " if ctype == 0x4E4F534A else b"\0") * pad
        body += struct.pack("<II", len(data), ctype) + data
    open(path, "wb").write(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)


def node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=float).reshape(4, 4).T
    T = np.eye(4); R = np.eye(4); S = np.eye(4)
    if "translation" in n: T[:3, 3] = n["translation"]
    if "rotation" in n:
        x, y, z, w = n["rotation"]
        R[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    if "scale" in n: S[:3, :3] = np.diag(n["scale"])
    return T @ R @ S


def world_bounds(js):
    acc, nodes, meshes = js["accessors"], js.get("nodes", []), js.get("meshes", [])
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)

    def walk(ni, M):
        nonlocal lo, hi
        n = nodes[ni]; M = M @ node_matrix(n)
        if "mesh" in n:
            for pr in meshes[n["mesh"]]["primitives"]:
                a = acc[pr["attributes"]["POSITION"]]
                mn, mx = a["min"], a["max"]
                for c in [(x, y, z) for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]:
                    w = (M @ np.array(list(c) + [1.0]))[:3]
                    lo = np.minimum(lo, w); hi = np.maximum(hi, w)
        for ch in n.get("children", []):
            walk(ch, M)

    roots = root_nodes(js)
    for r in roots:
        walk(r, np.eye(4))
    return lo, hi


def root_nodes(js):
    if js.get("scenes"):
        return list(js["scenes"][js.get("scene", 0)]["nodes"])
    children = {c for n in js.get("nodes", []) for c in n.get("children", [])}
    return [i for i in range(len(js.get("nodes", []))) if i not in children]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("factor", nargs="?", type=float)
    ap.add_argument("--height", type=float, help="target world height in metres (computes the factor)")
    ap.add_argument("--out")
    args = ap.parse_args()
    chunks = read_glb(args.glb)
    js = json.loads(chunks[0][1])
    lo, hi = world_bounds(js)
    h = hi[1] - lo[1]
    if args.height:
        factor = args.height / h
    elif args.factor:
        factor = args.factor
    else:
        raise SystemExit("give a factor or --height")
    S = np.diag([factor, factor, factor, 1.0])
    for ri in root_nodes(js):
        n = js["nodes"][ri]
        M = S @ node_matrix(n)
        for k in ("translation", "rotation", "scale"):
            n.pop(k, None)
        n["matrix"] = [float(v) for v in M.T.reshape(-1)]        # column-major, per the spec
    out = args.out or args.glb
    if out == args.glb:
        bak = args.glb[:-4] + ".orig.glb"
        if not os.path.exists(bak):
            shutil.copy2(args.glb, bak)
            print("backup: %s" % bak)
    chunks[0][1] = json.dumps(js, separators=(",", ":")).encode("utf-8")
    write_glb(out, chunks)
    lo2, hi2 = world_bounds(json.loads(read_glb(out)[0][1]))
    print("%s: height %.4f m -> %.4f m (x%.5f); wrote %s" % (os.path.basename(args.glb), h, hi2[1] - lo2[1], factor, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
