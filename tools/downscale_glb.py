#!/usr/bin/env python3
"""Reduce an image-to-3D GLB to the detail its source image can actually carry.

An AI image-to-3D mesh (Meshy, Hyper3D) is generated from one or a few photographs, so nothing
finer than those photographs' pixels is real: it is interpolation or invention. This decimates
the mesh, and downscales its textures, as far as that bound allows and no further:

* GEOMETRY -- quadric edge collapse (Garland-Heckbert, MeshLab's implementation through
  pymeshlab: optimal vertex placement, normal-flip guard, boundary preserved), texture-aware
  when the mesh keeps its texture (UV seams are collapsed only where the parameterisation
  survives). Without a texture the UV-seam splits are welded first, so the seams do not pin
  the decimation. The face count is chosen from `--sweep`, which measures each candidate
  against the original in pixels of the source image (`--px` mesh units each, i.e. the mesh's
  height over the subject's height in the photograph, in pixels). The rule: the fewest faces
  at which no original vertex is more than half a pixel off the result -- detail finer than
  that cannot have come from the photograph. When a lone outlier vertex keeps the maximum
  above half a pixel at every count, relax it to: at most 0.001% of the original's vertices
  beyond 0.5 px, and the 99.9th percentile within 0.25 px.
* TEXTURES -- padded first (every texel outside the UV charts takes its nearest chart texel, so
  the downscale cannot bleed the atlas background into a chart's edge), then resized with a
  Lanczos-3 filter -- in LINEAR light for the sRGB base colour, so a fine dark/bright pattern
  keeps its mean brightness -- to `--tex` pixels square.

The original file is never written; the result goes to `--out`.

Usage::

    python tools/downscale_glb.py SRC.glb --out OUT.glb --faces N [--tex 1024] [--no-texture]
    python tools/downscale_glb.py SRC.glb --sweep 200000,100000,60000 --px 0.00475 [--no-texture]
"""
from __future__ import annotations

import argparse
import io
import sys

import numpy as np
import pymeshlab
import pygltflib
from PIL import Image


def load(ms: pymeshlab.MeshSet, src: str, textured: bool):
    ms.load_new_mesh(src)
    if not textured:
        ms.meshing_remove_duplicate_vertices()      # weld the UV-seam splits
        ms.meshing_remove_unreferenced_vertices()


def decimate(ms: pymeshlab.MeshSet, faces: int, textured: bool):
    common = dict(targetfacenum=faces, qualitythr=0.5, preserveboundary=True, boundaryweight=1.0,
                  optimalplacement=True, preservenormal=True, planarquadric=False)
    if textured:
        ms.meshing_decimation_quadric_edge_collapse_with_texture(extratcoordw=1.0, **common)
    else:
        ms.meshing_decimation_quadric_edge_collapse(autoclean=True, **common)


def sweep(src: str, targets, px: float, textured: bool):
    """For each target face count, print how far the decimated surface strays from the original,
    in source-image pixels (`px` mesh units each): the worst detail LOST (original sampled against
    the result) and ADDED (result against the original), and the distribution over the original's
    own vertices -- the share more than 0.5 px off, and the 99.9th percentile."""
    for tgt in targets:
        ms = pymeshlab.MeshSet()
        load(ms, src, textured)          # mesh 0: the original, kept as the reference
        load(ms, src, textured)          # mesh 1: the copy that is decimated
        decimate(ms, tgt, textured)
        faces = ms.current_mesh().face_number()
        fwd = ms.get_hausdorff_distance(sampledmesh=1, targetmesh=0, samplenum=1000000)
        bwd = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1, samplenum=1000000)
        ms.set_current_mesh(0)
        ms.compute_scalar_by_distance_from_another_mesh_per_vertex(measuremesh=0, refmesh=1,
                                                                   signeddist=False)
        q = np.abs(ms.mesh(0).vertex_scalar_array()) / px
        print("%8d faces: lost max %.2f px, added max %.2f px; original vertices >0.5 px off: "
              "%.4f%%, p99.9 %.3f px"
              % (faces, bwd['max'] / px, fwd['max'] / px, 100 * (q > 0.5).mean(),
                 np.percentile(q, 99.9)), flush=True)


def glb_images(src: str):
    """The source's base-colour and metallic-roughness images (PIL), or None for either."""
    g = pygltflib.GLTF2().load(src)
    blob = g.binary_blob()
    def image(tex_ref):
        if tex_ref is None:
            return None
        im = g.images[g.textures[tex_ref.index].source]
        bv = g.bufferViews[im.bufferView]
        data = blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]
        return Image.open(io.BytesIO(data)).convert("RGB")
    pbr = g.materials[0].pbrMetallicRoughness if g.materials else None
    if pbr is None:
        return None, None, (1.0, 1.0)
    return (image(pbr.baseColorTexture), image(pbr.metallicRoughnessTexture),
            (pbr.metallicFactor if pbr.metallicFactor is not None else 1.0,
             pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0))


def chart_mask(uv_corners: np.ndarray, size: int) -> np.ndarray:
    """Texels covered by the UV triangles (v up, as pymeshlab gives them) at `size` square."""
    import cv2
    mask = np.zeros((size, size), np.uint8)
    pts = np.empty_like(uv_corners)
    pts[..., 0] = uv_corners[..., 0] * size
    pts[..., 1] = (1.0 - uv_corners[..., 1]) * size
    cv2.fillPoly(mask, np.round(pts * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
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


def lanczos(arr: np.ndarray, size: int) -> np.ndarray:
    """Per-channel Lanczos-3 resize of a float HxWxC array (PIL scales the kernel to the ratio)."""
    out = [np.asarray(Image.fromarray(arr[..., c].astype(np.float32), mode="F")
                      .resize((size, size), Image.LANCZOS)) for c in range(arr.shape[2])]
    return np.stack(out, axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("--out")
    ap.add_argument("--faces", type=int)
    ap.add_argument("--tex", type=int, default=1024)
    ap.add_argument("--no-texture", action="store_true", help="drop the textures (and weld UV seams)")
    ap.add_argument("--sweep", help="comma-separated face counts: print each one's Hausdorff error")
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

    import trimesh
    ms = pymeshlab.MeshSet()
    load(ms, a.src, textured)
    decimate(ms, a.faces, textured)
    ms.compute_normal_per_vertex(weightmode="By Angle")
    m = ms.current_mesh()
    V = m.vertex_matrix()
    N = m.vertex_normal_matrix()
    F = m.face_matrix()
    if textured:
        W = m.wedge_tex_coord_matrix().reshape(-1, 3, 2)
        # glTF wants one UV per vertex: split each vertex once per distinct UV it carries.
        key = np.concatenate([F.reshape(-1, 1).astype(np.float64), W.reshape(-1, 2)], axis=1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        vi = uniq[:, 0].astype(np.int64)
        pos, nrm, uv = V[vi], N[vi], uniq[:, 1:3]
        faces = inv.reshape(-1, 3)
        base, mr, (metal, rough) = glb_images(a.src)
        def prep(img, srgb):
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = pad(arr, chart_mask(W, arr.shape[0]))
            if srgb:
                return Image.fromarray(np.round(linear_to_srgb(lanczos(srgb_to_linear(arr), a.tex)) * 255)
                                       .astype(np.uint8))
            return Image.fromarray(np.round(np.clip(lanczos(arr, a.tex), 0, 1) * 255).astype(np.uint8))
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=prep(base, True) if base is not None else None,
            metallicRoughnessTexture=prep(mr, False) if mr is not None else None,
            metallicFactor=metal, roughnessFactor=rough)
        visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        mesh = trimesh.Trimesh(vertices=pos, faces=faces, vertex_normals=nrm, visual=visual, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_normals=N, process=False)
    mesh.export(a.out)
    print("wrote %s: %d faces, %d vertices%s" % (a.out, len(mesh.faces), len(mesh.vertices),
                                                 ", textures %d^2" % a.tex if textured else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
