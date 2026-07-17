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
from typing import List, Sequence, Tuple

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


def circle_profile(sides: int, radius: float = 1.0) -> List[Vec2]:
    return [(radius * math.cos(2 * math.pi * j / sides),
             radius * math.sin(2 * math.pi * j / sides)) for j in range(sides)]


def line_profile(half_width: float = 0.5) -> List[Vec2]:
    return [(-half_width, 0.0), (half_width, 0.0)]


def write_obj(path, verts: Sequence[Vec3], faces: Sequence[Tuple[int, int, int]]) -> None:
    lines = [f"v {v[0]:.6g} {v[1]:.6g} {v[2]:.6g}" for v in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for (a, b, c) in faces]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
