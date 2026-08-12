"""
Loom sweep engine — drag a cross-section (a *profile*) along a *spine* curve,
orienting it with a stable frame, scaling and twisting it, and skinning
consecutive cross-sections into a triangle mesh.

The orientation uses a **rotation-minimizing frame** (RMF) via Wang et al.'s
double-reflection method: it carries one reference normal along the spine with no
unwanted roll and no flips at inflections (a naive Frenet frame spins wildly).
For a **closed** spine the residual twist between the last and first frame is
distributed evenly so the tube/ribbon closes seamlessly.

This module is pure geometry over plain float 3-tuples (no Signals): callers
sample the animated spine/params at a frame and hand concrete numbers here.  The
scene-level :class:`~loom.scene.SweptMesh` element wires it to emission.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .atomicio import write_atomic

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


# ---- tiny 3-vector helpers ------------------------------------------------

def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a: Vec3, eps: float = 1e-12) -> Vec3:
    m = math.sqrt(_dot(a, a))
    if m < eps:
        return (0.0, 0.0, 0.0)
    return (a[0] / m, a[1] / m, a[2] / m)


def _any_perp(t: Vec3) -> Vec3:
    """Some unit vector perpendicular to t."""
    ax = (1.0, 0.0, 0.0) if abs(t[0]) < 0.9 else (0.0, 1.0, 0.0)
    return _norm(_cross(t, ax))


# ---- tangents + rotation-minimizing frames --------------------------------

def tangents(points: Sequence[Vec3], closed: bool) -> List[Vec3]:
    n = len(points)
    T: List[Vec3] = []
    for i in range(n):
        if closed:
            nxt = points[(i + 1) % n]
            prv = points[(i - 1) % n]
            T.append(_norm(_sub(nxt, prv)))
        else:
            if i == 0:
                T.append(_norm(_sub(points[1], points[0])))
            elif i == n - 1:
                T.append(_norm(_sub(points[n - 1], points[n - 2])))
            else:
                T.append(_norm(_sub(points[i + 1], points[i - 1])))
    return T


def rmf_frames(points: Sequence[Vec3], closed: bool
               ) -> Tuple[List[Vec3], List[Vec3], List[Vec3]]:
    """Return (T, R, S): unit tangent, and two normals forming a right-handed
    rotation-minimizing frame at each spine point.

    Double-reflection method.  For a closed spine the accumulated twist is spread
    across the ring so frame[n-1] -> frame[0] closes with no seam.
    """
    n = len(points)
    T = tangents(points, closed)
    R: List[Vec3] = [(0.0, 0.0, 0.0)] * n
    S: List[Vec3] = [(0.0, 0.0, 0.0)] * n
    R[0] = _any_perp(T[0])
    S[0] = _norm(_cross(T[0], R[0]))

    for i in range(n - 1):
        p_i, p_j = points[i], points[i + 1]
        t_i, t_j = T[i], T[i + 1]
        r_i = R[i]
        # reflection 1: across the plane bisecting the segment p_i -> p_j
        v1 = _sub(p_j, p_i)
        c1 = _dot(v1, v1)
        if c1 < 1e-20:
            R[i + 1] = r_i
            S[i + 1] = _norm(_cross(t_j, r_i))
            continue
        rL = _sub(r_i, _mul(v1, 2.0 / c1 * _dot(v1, r_i)))
        tL = _sub(t_i, _mul(v1, 2.0 / c1 * _dot(v1, t_i)))
        # reflection 2: across the plane bisecting tL -> t_j
        v2 = _sub(t_j, tL)
        c2 = _dot(v2, v2)
        if c2 < 1e-20:
            r_next = rL
        else:
            r_next = _sub(rL, _mul(v2, 2.0 / c2 * _dot(v2, rL)))
        r_next = _norm(r_next)
        R[i + 1] = r_next
        S[i + 1] = _norm(_cross(t_j, r_next))

    if closed and n > 2:
        # measure the angle between the transported frame back at point 0 and the
        # actual R[0], then unwind it linearly over the ring.
        # Transport R[n-1] one more reflection step onto point 0's tangent.
        p_i, p_j = points[n - 1], points[0]
        t_i, t_j = T[n - 1], T[0]
        v1 = _sub(p_j, p_i)
        c1 = _dot(v1, v1)
        if c1 > 1e-20:
            rL = _sub(R[n - 1], _mul(v1, 2.0 / c1 * _dot(v1, R[n - 1])))
            tL = _sub(t_i, _mul(v1, 2.0 / c1 * _dot(v1, t_i)))
            v2 = _sub(t_j, tL)
            c2 = _dot(v2, v2)
            r_loop = rL if c2 < 1e-20 else _sub(rL, _mul(v2, 2.0 / c2 * _dot(v2, rL)))
            r_loop = _norm(r_loop)
            # signed angle from r_loop to R[0] about T[0]
            cosang = max(-1.0, min(1.0, _dot(r_loop, R[0])))
            sinang = _dot(_cross(r_loop, R[0]), T[0])
            deficit = math.atan2(sinang, cosang)
            for i in range(n):
                a = deficit * (i / n)
                ca, sa = math.cos(a), math.sin(a)
                r, s = R[i], S[i]
                R[i] = _norm(_add(_mul(r, ca), _mul(s, sa)))
                S[i] = _norm(_cross(T[i], R[i]))
    return T, R, S


# ---- sweeping a profile ---------------------------------------------------

def sweep_rings(points: Sequence[Vec3], profile: Sequence[Vec2],
                scales: Sequence[float], twists: Sequence[float],
                closed_spine: bool) -> List[List[Vec3]]:
    """Place the 2-D ``profile`` at every spine point in its RMF frame, scaled and
    twisted, giving one ring of world 3-D points per spine point."""
    n = len(points)
    _, R, S = rmf_frames(points, closed_spine)
    rings: List[List[Vec3]] = []
    for i in range(n):
        c = points[i]
        r, s = R[i], S[i]
        sc = scales[i]
        th = twists[i]
        ct, st = math.cos(th), math.sin(th)
        ring: List[Vec3] = []
        for (a, b) in profile:
            a2 = (a * ct - b * st) * sc
            b2 = (a * st + b * ct) * sc
            ring.append(_add(c, _add(_mul(r, a2), _mul(s, b2))))
        rings.append(ring)
    return rings


def skin_rings(rings: Sequence[Sequence[Vec3]], closed_spine: bool, closed_profile: bool
               ) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """Skin consecutive rings into a triangle mesh.  Returns (vertices, faces),
    faces as 0-based index triples."""
    n = len(rings)
    if n < 2:
        raise ValueError("need >= 2 rings to skin")
    k = len(rings[0])
    verts: List[Vec3] = []
    for ring in rings:
        verts.extend(ring)
    faces: List[Tuple[int, int, int]] = []

    def vid(i: int, j: int) -> int:
        return (i % n) * k + (j % k)

    n_spans = n if closed_spine else n - 1
    edges = k if closed_profile else k - 1
    for i in range(n_spans):
        for j in range(edges):
            a = vid(i, j)
            b = vid(i, j + 1)
            c = vid(i + 1, j + 1)
            d = vid(i + 1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))
    return verts, faces


def ring_normals(rings: Sequence[Sequence[Vec3]], closed_spine: bool,
                 closed_profile: bool) -> List[Vec3]:
    """Per-vertex surface normals for a swept lattice, in ``skin_rings`` order.

    A sweep is a parametric surface ``P(u, v)`` — ``u`` along the spine, ``v`` around
    the profile — so its normal is ``dP/du x dP/dv``, and the ring lattice IS that
    parameterisation sampled on a grid.  Central differences on the grid give both
    tangents to second order, wrapping where the spine or profile closes and falling
    back to one-sided differences where it doesn't.

    **Why this exists rather than leaning on ``mesh { smooth <deg> }``.**  The crease
    heuristic works from the triangles alone, where a coarsely-sampled SMOOTH curve
    and a genuine sharp fold are the same thing: both are a big dihedral.  So it has
    to guess, and on a profile like ``r(a) = 1 + 0.34*cos(3a)`` at 18 samples it
    guesses wrong — the valleys turn 62 deg per step, over any sane crease angle, and
    stay faceted no matter how fine the spine sampling gets.  The generator does not
    have to guess: it knows the profile is a closed smooth loop swept along a smooth
    spine, so it can hand over the real normal and let the tessellation be as coarse
    as it likes.  Only the silhouette then betrays the polygon count.

    Degenerate rows (a spine that doubles back on itself, a zero-scale ring) leave a
    near-zero cross product; those fall back to a neighbouring vertex's normal via a
    second pass, and to +Z if the whole mesh is degenerate.
    """
    n = len(rings)
    if n < 2:
        raise ValueError("need >= 2 rings for normals")
    k = len(rings[0])
    out: List[Vec3] = []

    def at(i: int, j: int) -> Vec3:
        return rings[i % n][j % k]

    for i in range(n):
        for j in range(k):
            # dP/du — along the spine
            if closed_spine or (0 < i < n - 1):
                du = _sub(at(i + 1, j), at(i - 1, j))
            elif i == 0:
                du = _sub(at(1, j), at(0, j))
            else:
                du = _sub(at(n - 1, j), at(n - 2, j))
            # dP/dv — around the profile
            if closed_profile or (0 < j < k - 1):
                dv = _sub(at(i, j + 1), at(i, j - 1))
            elif j == 0:
                dv = _sub(at(i, 1), at(i, 0))
            else:
                dv = _sub(at(i, k - 1), at(i, k - 2))
            # dv x du, not du x dv: skin_rings winds a quad (i,j) (i,j+1) (i+1,j+1),
            # whose geometric normal is cross(dv, du). Matching it keeps the authored
            # normal on the same side as the face, which two-sided shading doesn't
            # care about but a renderer's shading-vs-geometric normal check does.
            nx = dv[1] * du[2] - dv[2] * du[1]
            ny = dv[2] * du[0] - dv[0] * du[2]
            nz = dv[0] * du[1] - dv[1] * du[0]
            ln = math.sqrt(nx * nx + ny * ny + nz * nz)
            out.append((nx / ln, ny / ln, nz / ln) if ln > 1e-15 else (0.0, 0.0, 0.0))

    # Patch degenerate vertices from whichever neighbour in their ring has a normal.
    if any(v == (0.0, 0.0, 0.0) for v in out):
        good = next((v for v in out if v != (0.0, 0.0, 0.0)), (0.0, 0.0, 1.0))
        for idx, v in enumerate(out):
            if v != (0.0, 0.0, 0.0):
                good = v
            else:
                out[idx] = good
    return out


def circle_profile(sides: int, radius: float = 1.0) -> List[Vec2]:
    return [(radius * math.cos(2 * math.pi * j / sides),
             radius * math.sin(2 * math.pi * j / sides)) for j in range(sides)]


def line_profile(half_width: float = 0.5) -> List[Vec2]:
    return [(-half_width, 0.0), (half_width, 0.0)]


def encode_obj(verts: Sequence[Vec3], faces: Sequence[Tuple[int, int, int]],
               normals: Optional[Sequence[Vec3]] = None) -> str:
    """Serialise an indexed mesh as OBJ text.

    Split out from :func:`write_obj` so the same bytes can go somewhere other than a
    file — the live viewer channel hands them to ftrace down the pipe the two
    processes already share, never touching the filesystem.  Keeping one encoder
    means the piped mesh and the written one cannot drift.

    ``normals``, when given, is one per vertex and is emitted as ``vn`` with
    ``f v//vn`` faces.  Authored normals switch ftrace's loader out of crease
    smoothing entirely (``vn`` always wins over ``mesh { smooth }``), which is the
    point: see :func:`ring_normals`.
    """
    lines = [f"v {v[0]:.6g} {v[1]:.6g} {v[2]:.6g}" for v in verts]
    if normals is None:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for (a, b, c) in faces]
    else:
        if len(normals) != len(verts):
            raise ValueError(f"encode_obj: {len(normals)} normals for {len(verts)} vertices")
        lines += [f"vn {n[0]:.6g} {n[1]:.6g} {n[2]:.6g}" for n in normals]
        lines += [f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}"
                  for (a, b, c) in faces]
    return "\n".join(lines) + "\n"


def write_obj(path, verts: Sequence[Vec3], faces: Sequence[Tuple[int, int, int]],
              normals: Optional[Sequence[Vec3]] = None) -> None:
    """Write an OBJ **atomically** (temp file in the same directory + ``os.replace``).

    A plain ``open(path, "w")`` truncates first and fills in afterwards, so anything
    reading the file meanwhile sees an empty or half-written mesh whose ``f`` lines
    reference vertices that aren't there yet.  That window is not hypothetical: the
    live viewer channel (§F4) re-emits a scene on a worker thread while ftrace is
    still loading the *previous* emission's assets out of the same directory.
    Replacing the whole file in one step makes a reader see either the old mesh or
    the new one, never a splice of the two.  See :mod:`loom.atomicio`.
    """
    write_atomic(path, encode_obj(verts, faces, normals), suffix=".obj.tmp")
