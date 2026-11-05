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
* NORMALS -- recomputed on the welded result: the mesh is cut along every edge whose dihedral
  angle exceeds `--crease` degrees, then each vertex takes the angle-weighted normal of its
  faces. A smooth surface shades smooth across a UV seam however much it curves, and a hard
  edge stays hard.
* TEXTURES -- every map of the material is carried (base colour, metal-roughness, normal,
  occlusion, emissive) with its sampler, and so are the material's factors; anything else
  (material extensions, vertex colours, a second material, ...) is refused rather than dropped.
  Each map is padded first: every texel outside the ORIGINAL mesh's UV charts takes its nearest
  chart texel, so the resize cannot bleed the atlas background into a chart's edge, even where
  decimation has moved the edge. Then it is resized with a Lanczos-3 filter so its longer side
  is `--tex` pixels (never enlarged): colour maps in LINEAR light (a fine dark/bright pattern
  keeps its mean brightness), a blended or masked base colour premultiplied by its alpha,
  normal maps renormalised.

The original file is never written; the result goes to `--out`, which may not name it.
Needs numpy, pillow, pymeshlab, pygltflib, scipy and opencv-python, plus trimesh for `--out`.

Usage::

    python tools/downscale_glb.py SRC.glb --out OUT.glb --faces N [--tex 1024] [--crease 60]
                                  [--no-texture]
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


def crease_normals(V: np.ndarray, F: np.ndarray, crease_deg: float):
    """Smooth normals that stay hard across creases: (per-corner unit normals, F.shape + (3,);
    per-corner smoothing-group ids, F.shape). Two faces that share an edge at no more than
    `crease_deg` degrees join their corners at both ends of it; a vertex's corners that end up
    connected form one group and share the angle-weighted normal of the group's faces. Grouping
    by connectivity is what makes a smooth but strongly curved vertex (a cone's tip) shade with
    ONE normal. MeshLab's own crease cut does the same job but refuses a non-manifold mesh, and
    the plain decimation can leave a few non-manifold edges; here such an edge just chains its
    faces pairwise."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nf = len(F)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    ang = np.empty(F.shape)
    for k in range(3):
        a = V[F[:, (k + 1) % 3]] - V[F[:, k]]
        b = V[F[:, (k + 2) % 3]] - V[F[:, k]]
        ang[:, k] = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum('ij,ij->i', a, b))
    # half-edge (f, k) runs from corner (f, k) to corner (f, k+1)
    nxt = np.roll(F, -1, axis=1)
    lo, hi = np.minimum(F, nxt).reshape(-1), np.maximum(F, nxt).reshape(-1)
    hf, hk = np.repeat(np.arange(nf), 3), np.tile(np.arange(3), nf)
    order = np.lexsort((hf, hi, lo))
    same = (lo[order][1:] == lo[order][:-1]) & (hi[order][1:] == hi[order][:-1])
    h1, h2 = order[:-1][same], order[1:][same]           # consecutive half-edges of one edge
    f1, f2, k1, k2 = hf[h1], hf[h2], hk[h1], hk[h2]
    smooth = np.einsum('ij,ij->i', fn[f1], fn[f2]) >= np.cos(np.radians(crease_deg))
    f1, f2, k1, k2 = f1[smooth], f2[smooth], k1[smooth], k2[smooth]
    c1a, c1b = 3 * f1 + k1, 3 * f1 + (k1 + 1) % 3
    c2a, c2b = 3 * f2 + k2, 3 * f2 + (k2 + 1) % 3
    flip = F[f1, k1] != F[f2, k2]                        # the faces run the edge oppositely
    c2a, c2b = np.where(flip, c2b, c2a), np.where(flip, c2a, c2b)
    n = 3 * nf
    graph = coo_matrix((np.ones(2 * len(c1a), np.int8), (np.r_[c1a, c1b], np.r_[c2a, c2b])),
                       shape=(n, n))
    ngroups, group = connected_components(graph, directed=False)
    w = ang.reshape(-1)[:, None] * np.repeat(fn, 3, axis=0)
    gn = np.stack([np.bincount(group, weights=w[:, i], minlength=ngroups) for i in range(3)], 1)
    length = np.linalg.norm(gn, axis=1, keepdims=True)
    if (length < 1e-300).any():
        # a group of nothing but degenerate faces: give it its vertex's plain smooth normal
        vn = np.stack([np.bincount(F.reshape(-1), weights=w[:, i], minlength=len(V))
                       for i in range(3)], 1)
        rep = np.zeros(ngroups, np.int64)
        rep[group] = F.reshape(-1)
        gn = np.where(length < 1e-300, vn[rep], gn)
        length = np.linalg.norm(gn, axis=1, keepdims=True)
    gn /= np.maximum(length, 1e-300)
    return gn[group].reshape(F.shape + (3,)), group.reshape(F.shape)


def to_8bit(img: Image.Image, keep_alpha: bool) -> Image.Image:
    """`img` as 8-bit RGB (RGBA if `keep_alpha`). A 16-bit greyscale map is rescaled, where
    Pillow's own convert would clip it to 255."""
    if img.mode.startswith('I'):
        a = np.asarray(img, dtype=np.float32)
        img = Image.fromarray(np.round(np.clip(a / 65535.0, 0, 1) * 255).astype(np.uint8), 'L')
    return img.convert('RGBA' if keep_alpha else 'RGB')


def glb_material(src: str):
    """The source's one material: ({slot: (texture image index, PIL image)}, {slot: sampler
    dict}, factors), refusing anything this tool would otherwise drop without a word."""
    import pygltflib
    g = pygltflib.GLTF2().load(src)
    prims = [p for m in g.meshes for p in m.primitives]
    if any(getattr(p.attributes, 'COLOR_0', None) is not None for p in prims):
        sys.exit(f"{src}: vertex colours (COLOR_0) are not carried; --no-texture drops them")
    used = {p.material for p in prims}
    if len(used) > 1:
        sys.exit(f"{src}: {len(used)} materials -- a textured run handles one (use --no-texture)")
    mi = used.pop() if used else None
    if mi is None:
        return {}, {}, {}
    mat = g.materials[mi]
    if mat.extensions:
        sys.exit(f"{src}: material extensions {sorted(mat.extensions)} are not carried")
    blob = g.binary_blob()
    pbr = mat.pbrMetallicRoughness
    opaque = (mat.alphaMode or 'OPAQUE') == 'OPAQUE'
    infos = {'baseColorTexture': pbr and pbr.baseColorTexture,
             'metallicRoughnessTexture': pbr and pbr.metallicRoughnessTexture,
             'normalTexture': mat.normalTexture, 'occlusionTexture': mat.occlusionTexture,
             'emissiveTexture': mat.emissiveTexture}
    images, samplers = {}, {}
    for slot, info in infos.items():
        if info is None:
            continue
        if (info.texCoord or 0) != 0 or info.extensions:
            sys.exit(f"{src}: {slot} uses TEXCOORD_{info.texCoord} or an extension, not carried")
        if getattr(info, 'scale', 1.0) not in (None, 1.0) or getattr(info, 'strength', 1.0) not in (None, 1.0):
            sys.exit(f"{src}: {slot} has a non-default scale/strength, not carried")
        tex = g.textures[info.index]
        if tex.source is None or tex.extensions:
            sys.exit(f"{src}: {slot}'s image comes through a texture extension, not carried")
        im = g.images[tex.source]
        if im.bufferView is None or blob is None:
            sys.exit(f"{src}: {slot}'s image is an external file, not carried (pack the asset as a GLB)")
        bv = g.bufferViews[im.bufferView]
        img = Image.open(io.BytesIO(blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]))
        # glTF ignores the base colour's alpha in OPAQUE mode, so it is not carried then: it may
        # be zero or unrelated data over painted texels.
        keep_alpha = (slot == 'baseColorTexture' and not opaque
                      and ('A' in img.getbands() or 'transparency' in img.info))
        images[slot] = (tex.source, to_8bit(img, keep_alpha))
        if tex.sampler is not None:
            s = g.samplers[tex.sampler]
            samplers[slot] = dict(magFilter=s.magFilter, minFilter=s.minFilter,
                                  wrapS=s.wrapS, wrapT=s.wrapT)
    factors = dict(baseColorFactor=(pbr.baseColorFactor if pbr and pbr.baseColorFactor else None),
                   metallicFactor=(pbr.metallicFactor if pbr and pbr.metallicFactor is not None else 1.0),
                   roughnessFactor=(pbr.roughnessFactor if pbr and pbr.roughnessFactor is not None else 1.0),
                   emissiveFactor=mat.emissiveFactor, alphaMode=mat.alphaMode,
                   alphaCutoff=mat.alphaCutoff, doubleSided=mat.doubleSided)
    return images, samplers, factors


def chart_mask(uv_corners: np.ndarray, h: int, w: int) -> np.ndarray:
    """Texels of an h x w image that the UV triangles (v up, as MeshLab and trimesh give them)
    touch. A texel is centred at +0.5 of its index while OpenCV puts a pixel's centre on the
    integer, hence the -0.5. Filled one triangle at a time: a single multi-polygon fill uses the
    even-odd rule, which would cut every overlap of mirrored or stacked UV islands out of the
    mask."""
    import cv2
    mask = np.zeros((h, w), np.uint8)
    pts = np.empty_like(uv_corners)
    pts[..., 0] = uv_corners[..., 0] * w - 0.5
    pts[..., 1] = (1.0 - uv_corners[..., 1]) * h - 0.5
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
    """Per-channel Lanczos-3 resize of a float32 HxWxC array to `size` = (w, h)."""
    out = [np.asarray(Image.fromarray(arr[..., c], mode="F").resize(size, Image.LANCZOS))
           for c in range(arr.shape[2])]
    return np.stack(out, axis=-1)


def resample(img: Image.Image, kind: str, mask, tex: int) -> Image.Image:
    """Pad `img` past the charts in `mask` (None: not padded), then Lanczos it so its longer side
    is `tex` (never enlarged), encoded as `kind` says. An RGBA image is resampled premultiplied,
    so a clear texel adds no colour to its neighbours."""
    arr = np.asarray(img, dtype=np.float32) / np.float32(255.0)
    if mask is not None:
        arr = pad(arr, mask)
    h, w = arr.shape[:2]
    s = min(1.0, tex / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    if kind == 'srgb':
        rgb = srgb_to_linear(arr[..., :3])
        if arr.shape[2] == 4:
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


def importer_stats(mr, factors: dict) -> str:
    """The two numbers ftrace's glTF importer derives from a metal-roughness map `mr` over ALL
    its texels (src/gltf.h): mean metalness, which types the material, and mean roughness, its
    constant, each texel's in ftrace's units as gltfRoughness converts it (r^2 / 0.646, at most
    1) -- so a result whose numbers moved would import as a different material."""
    if mr is None:
        return "no metal-roughness map"
    a = np.asarray(mr, dtype=np.float64) / 255.0
    r = np.clip(a[..., 1] * factors['roughnessFactor'], 0, 1)
    return "mean metalness %.4f, roughness %.4f" % (
        np.clip(a[..., 2] * factors['metallicFactor'], 0, 1).mean(),
        np.minimum(1.0, r * r / 0.646).mean())


def patch_glb(path: str, samplers: dict, factors: dict):
    """Write back into the exported GLB what trimesh cannot: the source's texture samplers, and
    its colour factors exactly (trimesh stores them as 8-bit colours)."""
    import pygltflib
    g = pygltflib.GLTF2().load(path)
    mat = g.materials[0]
    pbr = mat.pbrMetallicRoughness
    if factors.get('baseColorFactor') is not None:
        pbr.baseColorFactor = list(factors['baseColorFactor'])
    if factors.get('emissiveFactor') is not None:
        mat.emissiveFactor = list(factors['emissiveFactor'])
    slots = {'baseColorTexture': pbr.baseColorTexture,
             'metallicRoughnessTexture': pbr.metallicRoughnessTexture,
             'normalTexture': mat.normalTexture, 'occlusionTexture': mat.occlusionTexture,
             'emissiveTexture': mat.emissiveTexture}
    for slot, info in slots.items():
        if info is None or slot not in samplers:
            continue
        g.samplers.append(pygltflib.Sampler(**samplers[slot]))
        g.textures[info.index].sampler = len(g.samplers) - 1
    g.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("--out")
    ap.add_argument("--faces", type=int)
    ap.add_argument("--tex", type=int, default=1024, help="longest texture side (never enlarged)")
    ap.add_argument("--crease", type=float, default=60.0,
                    help="edges whose faces meet at more than this many degrees stay hard")
    ap.add_argument("--no-texture", action="store_true",
                    help="drop the material, its textures and any vertex colours")
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
    if not 0.0 <= a.crease <= 180.0:
        ap.error("--crease is an angle in degrees, 0 (flat shading) to 180 (smooth everywhere)")
    if os.path.exists(a.out) and os.path.samefile(a.out, a.src):
        ap.error("--out names the source file, which this tool never writes")

    import trimesh
    images, samplers, factors = glb_material(a.src) if textured else ({}, {}, {})
    if textured and not images:
        ap.error("the source has no textures to keep: pass --no-texture")
    ms = pymeshlab.MeshSet()
    load(ms, a.src)
    src_defects = edge_defects(ms.current_mesh().face_matrix())
    if textured and not ms.current_mesh().has_wedge_tex_coord():
        ap.error("the source has no UVs: pass --no-texture")
    W0 = ms.current_mesh().wedge_tex_coord_matrix().reshape(-1, 3, 2) if textured else None
    decimate(ms, a.faces, textured)
    defects = edge_defects(ms.current_mesh().face_matrix())
    if defects != src_defects:
        print("warning: open / non-manifold edges went from %d / %d to %d / %d"
              % (src_defects + defects), file=sys.stderr)
    m = ms.current_mesh()
    V, F = m.vertex_matrix(), m.face_matrix()
    N, group = crease_normals(V, F, a.crease)
    # glTF wants one normal and one UV per vertex: a vertex is split once per smoothing group
    # (groups never span two vertices) and per distinct UV within it.
    cols = [group.reshape(-1, 1).astype(np.float64)]
    if textured:
        cols.append(m.wedge_tex_coord_matrix().reshape(-1, 2))
    uniq, first, inv = np.unique(np.concatenate(cols, axis=1), axis=0,
                                 return_index=True, return_inverse=True)
    kw = dict(vertices=V[F.reshape(-1)[first]], faces=inv.reshape(-1, 3),
              vertex_normals=N.reshape(-1, 3)[first], process=False)
    if textured:
        tiled = W0.min() < -1e-4 or W0.max() > 1 + 1e-4
        if tiled:
            print("warning: the UVs leave [0,1], so the maps are taken as tiled and not padded",
                  file=sys.stderr)
        masks, done, out = {}, {}, {}
        for slot, kind in SLOTS:
            if slot not in images:
                continue
            src_index, img = images[slot]
            if (src_index, kind) not in done:        # an ORM-packed map is resampled once
                shape = (img.height, img.width)
                if not tiled and shape not in masks:
                    masks[shape] = chart_mask(W0, *shape)
                done[(src_index, kind)] = resample(img, kind, None if tiled else masks[shape], a.tex)
            out[slot] = done[(src_index, kind)]
        mat = trimesh.visual.material.PBRMaterial(
            **out, **{k: v for k, v in factors.items() if v is not None})
        kw['visual'] = trimesh.visual.TextureVisuals(uv=uniq[:, 1:3], material=mat)
        print("importer stats: source %s; result %s" % (
            importer_stats(images.get('metallicRoughnessTexture', (0, None))[1], factors),
            importer_stats(out.get('metallicRoughnessTexture'), factors)))
    mesh = trimesh.Trimesh(**kw)
    mesh.export(a.out)
    if textured:
        patch_glb(a.out, samplers, factors)
    print("wrote %s: %d faces, %d vertices%s" % (
        a.out, len(mesh.faces), len(mesh.vertices),
        "".join(", %s %dx%d" % (s, im.width, im.height) for s, im in out.items()) if textured else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
