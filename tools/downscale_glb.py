#!/usr/bin/env python3
"""Reduce an image-to-3D GLB to the detail its source image can actually carry.

An AI image-to-3D mesh (Meshy, Hyper3D) is generated from one or a few photographs, so nothing
finer than those photographs' pixels is real: it is interpolation or invention. This decimates
the mesh, and downscales its textures, as far as that bound allows and no further:

* GEOMETRY -- quadric edge collapse (Garland-Heckbert, MeshLab's implementation through
  pymeshlab: optimal vertex placement, normal-flip guard, boundary preserved). A glTF repeats
  every vertex along every UV seam, and MeshLab would treat each seam as an open boundary,
  pinning it vertex for vertex; so the seam copies are WELDED first. The UVs live on the face
  corners and survive the weld, and with the texture kept the collapse is texture-aware.
  The face count is chosen from `--sweep`, which measures each candidate against the original
  in pixels of the source image (`--px` mesh units each, i.e. the mesh's height over the
  subject's height in the photograph, in pixels). The rule: the fewest faces at which no
  original vertex is more than half a pixel off the result -- detail finer than that cannot
  have come from the photograph. When a lone outlier vertex keeps the maximum above half a
  pixel at every count, relax it to: at most 0.001% of the original's vertices beyond 0.5 px,
  and the 99.9th percentile within 0.25 px.
* NORMALS -- recomputed on the welded result per face corner: the angle-weighted face normals
  around the vertex, over only the faces within `--crease` degrees of the corner's own face.
  A smooth surface shades smooth across a UV seam, and a hard edge stays hard.
* TEXTURES -- every map of the material is carried (base colour with its alpha, metal-roughness,
  normal, occlusion, emissive) and so are its factors. Each map is padded first: every texel
  outside the ORIGINAL mesh's UV charts takes its nearest chart texel, so the resize cannot
  bleed the atlas background into a chart's edge, even where decimation has moved the edge.
  Then it is resized with a Lanczos-3 filter so its longer side is `--tex` pixels (never
  enlarged): colour maps in LINEAR light (a fine dark/bright pattern keeps its mean
  brightness), base colour premultiplied by its alpha, normal maps renormalised.

The original file is never written; the result goes to `--out`, which may not name it.
Needs numpy, pillow, pymeshlab, pygltflib, scipy and opencv-python, plus trimesh for `--out`.

Usage::

    python tools/downscale_glb.py SRC.glb --out OUT.glb --faces N [--tex 1024] [--no-texture]
    python tools/downscale_glb.py SRC.glb --sweep 200000,100000,60000 --px 0.00475 [--no-texture]
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
import pymeshlab
from PIL import Image

# glTF texture slot -> how its texels are encoded
SLOTS = (('baseColorTexture', 'srgb'), ('metallicRoughnessTexture', 'linear'),
         ('normalTexture', 'normal'), ('occlusionTexture', 'linear'),
         ('emissiveTexture', 'srgb'))


def load(ms: pymeshlab.MeshSet, src: str):
    """Load `src` as ONE layer and weld the vertices glTF splits along UV seams. The UVs are
    MeshLab wedge (per-corner) coordinates, so they come through the weld unchanged."""
    ms.load_new_mesh(src, load_in_a_single_layer=True)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()


def decimate(ms: pymeshlab.MeshSet, faces: int, textured: bool):
    common = dict(targetfacenum=faces, qualitythr=0.5, preserveboundary=True, boundaryweight=1.0,
                  optimalplacement=True, preservenormal=True, planarquadric=False)
    if textured:
        ms.meshing_decimation_quadric_edge_collapse_with_texture(extratcoordw=1.0, **common)
    else:
        ms.meshing_decimation_quadric_edge_collapse(autoclean=True, **common)


def edge_defects(F: np.ndarray):
    """(open edges, non-manifold edges) of a triangle list over welded vertices."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum()), int((c > 2).sum())


def sweep(src: str, targets, px: float, textured: bool):
    """For each target face count, print how far the decimated surface strays from the original,
    in source-image pixels (`px` mesh units each): over the original's own vertices, the worst
    one (detail LOST), the share more than 0.5 px off and the 99.9th percentile; and the worst
    point of the result's faces off the original (detail ADDED)."""
    ms = pymeshlab.MeshSet()
    load(ms, src)                               # layer 0: the original, kept as the reference
    for tgt in targets:
        ms.set_current_mesh(0)
        ms.generate_copy_of_current_mesh()
        cid = ms.current_mesh_id()
        decimate(ms, tgt, textured)
        faces = ms.current_mesh().face_number()
        fwd = ms.get_hausdorff_distance(sampledmesh=cid, targetmesh=0, samplevert=True,
                                        sampleface=True, samplenum=1000000)
        ms.set_current_mesh(0)
        ms.compute_scalar_by_distance_from_another_mesh_per_vertex(measuremesh=0, refmesh=cid,
                                                                   signeddist=False)
        q = np.abs(ms.mesh(0).vertex_scalar_array()) / px
        print("%8d faces: lost max %.2f px, added max %.2f px; original vertices >0.5 px off: "
              "%.4f%%, p99.9 %.3f px"
              % (faces, q.max(), fwd['max'] / px, 100 * (q > 0.5).mean(), np.percentile(q, 99.9)),
              flush=True)
        ms.set_current_mesh(cid)
        ms.delete_current_mesh()


def corner_normals(V: np.ndarray, F: np.ndarray, crease_deg: float) -> np.ndarray:
    """Per-corner unit normals (F.shape + (3,)): for the corner of face f at vertex v, the sum of
    the normals of v's faces weighted by their angle at v, over only the faces whose normal is
    within `crease_deg` of f's."""
    e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    fn = np.cross(e1, e2)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    ang = np.empty(F.shape)
    for k in range(3):
        a = V[F[:, (k + 1) % 3]] - V[F[:, k]]
        b = V[F[:, (k + 2) % 3]] - V[F[:, k]]
        ang[:, k] = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum('ij,ij->i', a, b))
    cv = F.reshape(-1)                          # each corner's vertex
    order = np.argsort(cv, kind='stable')       # corners grouped by vertex
    starts = np.r_[0, np.flatnonzero(np.diff(cv[order])) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    # one row per vertex: its corners, padded with -1 to the largest valence
    table = np.full((len(starts), counts.max()), -1, dtype=np.int64)
    table[np.repeat(np.arange(len(starts)), counts),
          np.arange(len(order)) - np.repeat(starts, counts)] = order
    cosc = np.cos(np.radians(crease_deg))
    fnc = np.vstack([np.repeat(fn, 3, axis=0), np.zeros((1, 3))])   # corner -> its face normal
    angc = np.r_[ang.reshape(-1), 0.0]                              # index -1 -> the zero pad
    out = np.zeros((len(cv), 3))
    rows = max(64, 20_000_000 // table.shape[1] ** 2)   # bounds the valence^2 work per chunk
    for r0 in range(0, len(table), rows):
        t = table[r0:r0 + rows]
        n = fnc[t]                              # rows x valence x 3
        w = angc[t]
        near = np.einsum('rmi,rni->rmn', n, n) >= cosc
        s = np.einsum('rmn,rn,rni->rmi', near, w, n)
        ok = t >= 0
        out[t[ok]] = s[ok]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-300)
    return out.reshape(F.shape + (3,))


def glb_material(src: str):
    """The source's one material: ({slot: PIL image}, factors) -- refusing anything this tool
    would otherwise drop without a word."""
    import pygltflib
    g = pygltflib.GLTF2().load(src)
    used = {p.material for m in g.meshes for p in m.primitives}
    if len(used) > 1:
        sys.exit(f"{src}: {len(used)} materials -- a textured run handles one (use --no-texture)")
    mi = used.pop() if used else None
    if mi is None:
        return {}, {}
    mat = g.materials[mi]
    blob = g.binary_blob()
    images = {}
    pbr = mat.pbrMetallicRoughness
    infos = {'baseColorTexture': pbr and pbr.baseColorTexture,
             'metallicRoughnessTexture': pbr and pbr.metallicRoughnessTexture,
             'normalTexture': mat.normalTexture, 'occlusionTexture': mat.occlusionTexture,
             'emissiveTexture': mat.emissiveTexture}
    for slot, info in infos.items():
        if info is None:
            continue
        if (info.texCoord or 0) != 0 or info.extensions:
            sys.exit(f"{src}: {slot} uses TEXCOORD_{info.texCoord} or an extension, not carried")
        if getattr(info, 'scale', 1.0) not in (None, 1.0) or getattr(info, 'strength', 1.0) not in (None, 1.0):
            sys.exit(f"{src}: {slot} has a non-default scale/strength, not carried")
        im = g.images[g.textures[info.index].source]
        bv = g.bufferViews[im.bufferView]
        data = blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]
        img = Image.open(io.BytesIO(data))
        keep_alpha = slot == 'baseColorTexture' and ('A' in img.getbands() or 'transparency' in img.info)
        images[slot] = img.convert('RGBA' if keep_alpha else 'RGB')
    factors = dict(baseColorFactor=(pbr.baseColorFactor if pbr and pbr.baseColorFactor else None),
                   metallicFactor=(pbr.metallicFactor if pbr and pbr.metallicFactor is not None else 1.0),
                   roughnessFactor=(pbr.roughnessFactor if pbr and pbr.roughnessFactor is not None else 1.0),
                   emissiveFactor=mat.emissiveFactor, alphaMode=mat.alphaMode,
                   alphaCutoff=mat.alphaCutoff, doubleSided=mat.doubleSided)
    return images, factors


def chart_mask(uv_corners: np.ndarray, h: int, w: int) -> np.ndarray:
    """Texels of an h x w image covered by the UV triangles (v up, as MeshLab and trimesh give
    them). Filled one triangle at a time: a single multi-polygon fill uses the even-odd rule,
    which would cut every overlap of mirrored or stacked UV islands out of the mask."""
    import cv2
    mask = np.zeros((h, w), np.uint8)
    pts = np.empty_like(uv_corners)
    pts[..., 0] = uv_corners[..., 0] * w
    pts[..., 1] = (1.0 - uv_corners[..., 1]) * h
    pts = np.round(pts * 16).astype(np.int32)
    for tri in pts:
        cv2.fillConvexPoly(mask, tri, 1, lineType=cv2.LINE_8, shift=4)
    return mask.astype(bool)


def pad(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Every texel outside `mask` takes its nearest texel inside it."""
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return img[idx[0], idx[1]]


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def lanczos(arr: np.ndarray, size) -> np.ndarray:
    """Per-channel Lanczos-3 resize of a float HxWxC array to `size` = (w, h)."""
    out = [np.asarray(Image.fromarray(arr[..., c].astype(np.float32), mode="F")
                      .resize(size, Image.LANCZOS)) for c in range(arr.shape[2])]
    return np.stack(out, axis=-1)


def resample(img: Image.Image, kind: str, mask: np.ndarray, tex: int) -> Image.Image:
    """Pad `img` past the charts in `mask`, then Lanczos it so its longer side is `tex` (never
    enlarged), encoded as `kind` says."""
    arr = pad(np.asarray(img, dtype=np.float64) / 255.0, mask)
    h, w = arr.shape[:2]
    s = min(1.0, tex / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    if kind == 'srgb':
        rgb = srgb_to_linear(arr[..., :3])
        if arr.shape[2] == 4:                   # premultiplied, so a clear texel adds no colour
            a = arr[..., 3:]
            rgba = lanczos(np.concatenate([rgb * a, a], axis=2), size)
            a = np.clip(rgba[..., 3:], 0.0, 1.0)
            rgb = np.where(a > 1e-6, rgba[..., :3] / np.maximum(a, 1e-6), 0.0)
            res = np.concatenate([linear_to_srgb(rgb), a], axis=2)
        else:
            res = linear_to_srgb(lanczos(rgb, size))
    elif kind == 'normal':
        n = lanczos(arr * 2.0 - 1.0, size)
        n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
        res = n * 0.5 + 0.5
    else:
        res = lanczos(arr, size)
    return Image.fromarray(np.round(np.clip(res, 0.0, 1.0) * 255).astype(np.uint8))


def importer_stats(images: dict, factors: dict) -> str:
    """The two numbers ftrace's glTF importer derives from a metal-roughness map over ALL its
    texels (src/gltf.h): mean metalness, which types the material, and mean roughness, its
    constant -- so a result whose numbers moved would import as a different material."""
    mr = images.get('metallicRoughnessTexture')
    if mr is None:
        return "no metal-roughness map"
    a = np.asarray(mr, dtype=np.float64) / 255.0
    return "mean metalness %.4f, roughness %.4f" % (
        np.clip(a[..., 2] * factors['metallicFactor'], 0, 1).mean(),
        np.clip(a[..., 1] * factors['roughnessFactor'], 0, 1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("--out")
    ap.add_argument("--faces", type=int)
    ap.add_argument("--tex", type=int, default=1024, help="longest texture side (never enlarged)")
    ap.add_argument("--crease", type=float, default=60.0,
                    help="faces further apart than this (degrees) do not share normals")
    ap.add_argument("--no-texture", action="store_true", help="drop the material and its textures")
    ap.add_argument("--sweep", help="comma-separated face counts: print each one's error")
    ap.add_argument("--px", type=float, help="one source-image pixel in mesh units (for --sweep)")
    a = ap.parse_args()
    textured = not a.no_texture
    if a.sweep:
        if not a.px:
            ap.error("--sweep needs --px")
        sweep(a.src, [int(t) for t in a.sweep.split(",")], a.px, textured)
        return 0
    if not (a.out and a.faces):
        ap.error("--out and --faces are required")
    if os.path.exists(a.out) and os.path.samefile(a.out, a.src):
        ap.error("--out names the source file, which this tool never writes")

    import trimesh
    images, factors = glb_material(a.src) if textured else ({}, {})
    if textured and not images:
        ap.error("the source has no textures to keep: pass --no-texture")
    ms = pymeshlab.MeshSet()
    load(ms, a.src)
    src_defects = edge_defects(ms.current_mesh().face_matrix())
    if textured and not ms.current_mesh().has_wedge_tex_coord():
        ap.error("the source has no UVs: pass --no-texture")
    W0 = ms.current_mesh().wedge_tex_coord_matrix().reshape(-1, 3, 2) if textured else None
    decimate(ms, a.faces, textured)
    m = ms.current_mesh()
    V, F = m.vertex_matrix(), m.face_matrix()
    defects = edge_defects(F)
    if defects != src_defects:
        print("warning: open / non-manifold edges went from %d / %d to %d / %d"
              % (src_defects + defects), file=sys.stderr)
    N = corner_normals(V, F, a.crease).reshape(-1, 3)
    # glTF wants one UV and one normal per vertex: split each vertex once per distinct pair.
    cols = [F.reshape(-1, 1).astype(np.float64), N]
    if textured:
        cols.append(m.wedge_tex_coord_matrix().reshape(-1, 2))
    uniq, inv = np.unique(np.concatenate(cols, axis=1), axis=0, return_inverse=True)
    vi = uniq[:, 0].astype(np.int64)
    faces = inv.reshape(-1, 3)
    kw = dict(vertices=V[vi], faces=faces, vertex_normals=uniq[:, 1:4], process=False)
    if textured:
        masks = {}
        def mask_for(img):
            key = (img.height, img.width)
            if key not in masks:
                masks[key] = chart_mask(W0, *key)
            return masks[key]
        out = {slot: resample(images[slot], kind, mask_for(images[slot]), a.tex)
               for slot, kind in SLOTS if slot in images}
        mat = trimesh.visual.material.PBRMaterial(
            **out, **{k: v for k, v in factors.items() if v is not None})
        kw['visual'] = trimesh.visual.TextureVisuals(uv=uniq[:, 4:6], material=mat)
        print("importer stats: source %s; result %s"
              % (importer_stats(images, factors), importer_stats(out, factors)))
    mesh = trimesh.Trimesh(**kw)
    mesh.export(a.out)
    print("wrote %s: %d faces, %d vertices%s" % (
        a.out, len(mesh.faces), len(mesh.vertices),
        "".join(", %s %dx%d" % (s, im.width, im.height) for s, im in out.items()) if textured else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
