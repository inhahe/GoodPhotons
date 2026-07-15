#!/usr/bin/env python3
"""Repair a triangle mesh into a watertight (closed, non-self-intersecting) 2-manifold.

Glass (`dielectric`) surfaces in the renderer decide enter-vs-exit from the surface
normal at each hit and carry the "which medium am I inside" state along the whole photon
path. A hole (boundary edge), a non-manifold edge/vertex (3+ faces meeting an edge, or
several sheets pinched to one point), or a self-intersection makes that decision
ambiguous, so refraction goes wrong for the rest of that path and light lands in the wrong
place. `ftrace -in <scene> -check-watertight` DETECTS such meshes; this tool FIXES them,
then you re-audit to confirm `[OK]`.

Two engines (both pip-installable; see README's watertight section):

  meshlab (default, `pip install pymeshlab`) — MeshLab's repair filters. Best general
    cleaner: welds coincident vertices (so pinches become visible), drops duplicate/null
    faces, removes faces at non-manifold edges, splits non-manifold vertices, and fills the
    resulting holes. Handles the common AI-mesh-generator "pinch vertex" defect (N sheets
    snapped to one point) that MeshFix leaves alone.

  meshfix (`pip install pymeshfix`) — Marco Attene's MeshFix. Best for genuine
    self-intersections and large holes; it removes intersecting triangles, fills holes, and
    keeps the largest shell. Weaker on pure non-manifold pinches (may report "could not fix
    everything" and change nothing).

(Contrast the `manifold3d` library, which never *introduces* a leak but also cannot
*repair* a pre-broken mesh — use it when authoring/combining geometry, not for salvage.)

Examples:
    # Repair a broken OBJ in place-ish (write a new file), MeshLab engine.
    python repair_mesh.py scraps/klein_hunyuan_clean.obj scraps/klein_hunyuan_clean.obj

    # Repair the master, then emit a copy placed exactly where a derived (e.g. staged)
    # mesh sat, so scenes that load the derived file get the airtight geometry. The best-fit
    # affine input->ref is computed from vertex correspondence (same order/count) BEFORE
    # repair, then applied to the repaired mesh (a global transform is index-independent, so
    # it survives the repair's re-indexing).
    python repair_mesh.py scraps/klein_hunyuan_clean.obj scraps/klein_staged.obj \
        --place-like scraps/klein_staged.orig.obj
"""
import argparse
import sys

import numpy as np


def load_obj(path):
    """Read an OBJ's vertex positions and triangulated faces (0-based).

    Ignores normals/UVs; polygons are fan-triangulated; negative (relative) indices are
    resolved against the running vertex count.
    """
    verts, faces = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    vtok = tok.split("/")[0]
                    if not vtok:
                        continue
                    vi = int(vtok)
                    idx.append(vi - 1 if vi > 0 else len(verts) + vi)
                for k in range(1, len(idx) - 1):  # fan triangulate
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_obj(path, verts, faces, comment=None):
    with open(path, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        for x, y, z in verts:
            f.write(f"v {x:.8g} {y:.8g} {z:.8g}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def diag_of(verts):
    return float(np.linalg.norm(verts.max(0) - verts.min(0)))


def fit_affine(src, dst):
    """Least-squares 3x4 affine M s.t. dst ≈ [src|1] @ M.T  (rows of M are [a b c t])."""
    if src.shape != dst.shape:
        sys.exit(f"--place-like: vertex counts differ ({src.shape[0]} vs {dst.shape[0]}); "
                 "the reference must share vertex correspondence with the input (same OBJ, "
                 "just transformed — pass the ORIGINAL derived mesh, not a repaired one).")
    A = np.hstack([src, np.ones((src.shape[0], 1))])   # (n,4)
    Mt, *_ = np.linalg.lstsq(A, dst, rcond=None)        # (4,3)
    resid = float(np.linalg.norm(A @ Mt - dst, axis=1).max())
    return Mt, resid


def repair_meshlab(v, f, merge_eps, max_hole):
    import pymeshlab as ml
    ms = ml.MeshSet()
    m = ml.Mesh(v, f)
    ms.add_mesh(m, "in")
    ms.meshing_merge_close_vertices(threshold=ml.PureValue(merge_eps))
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_repair_non_manifold_edges(method=0)     # 0 = remove offending faces
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=max_hole)
    ms.meshing_remove_unreferenced_vertices()
    cm = ms.current_mesh()
    return (np.asarray(cm.vertex_matrix(), dtype=np.float64),
            np.asarray(cm.face_matrix(), dtype=np.int64))


def repair_meshfix(v, f, join_components):
    from pymeshfix import MeshFix
    mf = MeshFix(v, f)
    mf.repair(joincomp=join_components, remove_smallest_components=not join_components)
    return (np.asarray(mf.points, dtype=np.float64),
            np.asarray(mf.faces, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser(description="Repair a mesh into a watertight 2-manifold.")
    ap.add_argument("input", help="input OBJ (possibly non-airtight / self-intersecting)")
    ap.add_argument("output", help="output OBJ (watertight)")
    ap.add_argument("--engine", choices=("meshlab", "meshfix"), default="meshlab",
                    help="meshlab (default) = MeshLab repair filters; meshfix = Attene's MeshFix")
    ap.add_argument("--place-like", metavar="REF.OBJ",
                    help="after repair, apply the best-fit affine (input->REF) so the output "
                         "lands where REF sat; REF must share vertex correspondence with input")
    ap.add_argument("--merge-eps", type=float, default=None,
                    help="[meshlab] absolute weld distance for coincident vertices "
                         "(default: bbox_diagonal * 1e-6, matching the audit's weld scale)")
    ap.add_argument("--max-hole", type=int, default=200,
                    help="[meshlab] largest hole (edge count) to auto-close (default 200)")
    ap.add_argument("--join-components", action="store_true",
                    help="[meshfix] keep and join ALL components, not just the largest shell")
    args = ap.parse_args()

    v, f = load_obj(args.input)
    diag = diag_of(v)
    print(f"[repair] loaded {args.input}: {len(v)} verts, {len(f)} tris  (bbox diag {diag:.4f})")

    # Placement affine is computed from the ORIGINAL (pre-repair) vertices, while
    # correspondence with the reference still holds.
    place_Mt = None
    if args.place_like:
        rv, _ = load_obj(args.place_like)
        place_Mt, err = fit_affine(v, rv)
        print(f"[repair] placement affine from {args.place_like}: max fit residual {err:.3e}")

    try:
        if args.engine == "meshlab":
            eps = args.merge_eps if args.merge_eps is not None else diag * 1e-6
            print(f"[repair] engine=meshlab  merge_eps={eps:.3e}  max_hole={args.max_hole}")
            rv, rf = repair_meshlab(v, f, eps, args.max_hole)
        else:
            print(f"[repair] engine=meshfix  join_components={args.join_components}")
            rv, rf = repair_meshfix(v, f, args.join_components)
    except ImportError as e:
        sys.exit(f"[repair] {args.engine} engine not installed ({e}). "
                 f"Run: pip install {'pymeshlab' if args.engine == 'meshlab' else 'pymeshfix'}")

    if len(rf) == 0:
        sys.exit("[repair] engine produced an empty mesh — no closable shell (e.g. a flat "
                 "open sheet). Nothing written.")
    print(f"[repair] repaired: {len(rv)} verts, {len(rf)} tris "
          f"({len(rv) - len(v):+d} verts, {len(rf) - len(f):+d} tris vs input)")

    if place_Mt is not None:
        rv = np.hstack([rv, np.ones((rv.shape[0], 1))]) @ place_Mt

    write_obj(args.output, rv, rf,
              comment=f"watertight repair of {args.input} via {args.engine}"
                      + (f", placed like {args.place_like}" if args.place_like else ""))
    print(f"[repair] wrote {args.output}  (re-audit: ftrace -in <scene> -check-watertight)")


if __name__ == "__main__":
    main()
