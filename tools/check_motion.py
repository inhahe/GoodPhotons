"""Regression check: motion keys and the camera shutter (ROADMAP 11 stage A).

    D:/python314/python.exe -u tools/check_motion.py [--out <dir>] [--exe ftrace.exe] [--no-gpu]

A box lit by a ceiling panel holds four moving groups over the scene's `motion_interval -0.5 0.5`:
a sphere sliding left to right, a cube turning a quarter turn about y, a nested pair -- a group
that rises and then, from the instant on, moves sideways, holding a group that turns -- and a
cube whose keys scale and shear it together. It requires:

  * Without a shutter, the moving scene renders BYTE FOR BYTE what a static scene holding each
    group at the frame's instant renders (mode R on the CPU, and on the GPU unless --no-gpu), even
    with a ball whose motion reaches 60 m off the stage, on a floor whose albedo follows `cavity`
    (whose default probe radius is a share of the scene's size): the keys change nothing but the
    bounds the BVH is built from, and the scene's own bounds stay its instant's.
  * With `shutter_open -0.5  shutter_close 0.5`, mode R renders the average over the shutter of
    the static scenes: held against the mean of 32 static renders at evenly spaced times (the
    groups' transforms there written out as plain translate/rotate/scale/shear, interpolated as
    the loader interpolates keys), over 8x8-pixel blocks of luminance, each within 5 standard
    errors or 3 %. The blur must be real: the blurred and instant frames differ in many blocks.
    The same scene authored in centimetres blurs the same.
  * An instantaneous shutter away from the instant (`shutter_open t  shutter_close t`) renders the
    static scene at t, for three t, the same block test -- with a ball added that orbits two whole
    turns on each side of the instant, which the keys must not lose.
  * A procedural pattern travels with its moving object: a striped ball held sharp late in the
    shutter equals the static ball there with its world-space stripes shifted by its travel
    (4x4 blocks, the same test).
  * Mode W blurs too, deterministically (two runs agree byte for byte, and differ from the
    instant).
  * `-checkbvh` passes on a scene of moving triangles, spheres, isosurfaces, curves and mesh
    instances -- every other ray at a random time, so the swept bounds are held to the moved
    primitives. An isosurface, a curve and an instance in a group that moves, turns and grows,
    held sharp at two times, render as the static group written out there; so does a group of
    lights that moves and turns (an area panel, a glowing ball, a spot, a tube, an emissive
    triangle), in modes R, B and D. So does a glowing sheet a keyed deformer grows as it slides,
    flipped by `emit_orient` to radiate down (its still twin the sheet scaled and moved by its own
    transform), in every mode on both devices -- also loaded after a `shape_only` mesh, whose
    stripped triangles renumber its own, and over a rough mirror, where the glossy bounces' MIS
    weighs it at its area then -- and one grown from nothing lights nothing at the interval's start.
    A panel whose key flattens space along its normal (a singular key) lights the room as before.
  * A ball whose sweep runs far past the instant's bounding sphere is lit by the sky (mode B) and
    by a low sun (mode B), and shadows the floor in that sun (mode D), as in blurred mode R. A
    shared flight resumes its own shutter's sidecar and refuses one rendered with another.
  * Blurred modes D (BDPT) and B (forward light tracing) on the CPU agree with blurred mode R.
    So do two mode B cameras with the shutter rendered beside a third without it (one flight per
    shutter, on any device), the third staying sharp; and the sky mode B sees directly behind a
    dark ball sweeping across it.
  * Blurred mode R on the GPU agrees with blurred mode R on the CPU: the box; an isosurface, a
    curve and an instance moving; a mover beyond the instant's bounds under the sky and under the
    sun; a pattern travelling with its object; moving lights of every shape; a moving camera. So
    does blurred mode D on the GPU with mode D on the CPU (the sky aside: mode D renders no
    environment light, on either device). So does blurred mode B on the GPU with mode B on the CPU
    in every one of those cases and with moving mirrors and glass (a mirror panel sliding and
    turning, a mirror tile whose halves move apart, a glass ball and a mirror ball crossing the
    room), and the box on the wavefront backend; blurred mode A on the GPU
    with mode A on the CPU (the box, the moving lights), and blurred mode C on the GPU, through a
    wide aperture, with mode A on the GPU. A blurred shared mode B flight flies on the GPU (its two
    cameras agreeing with blurred mode R, the test above, on the default device), and mode V says
    it renders the instant.
  * Modes M and J (an epoch at a time), S and U (a pass at a time) blur, agreeing with
    blurred mode R within their density estimates' bias, and follow a moving camera as mode R
    does -- M, S, U, J and P on the GPU too, where they (and B and D) blur moving lights (an aimed
    panel, a spot and a ball alone) and the deforming glowing sheet as mode R on the CPU does; so does mode P, its backward samples split by their own path's first event -- also with
    fog in front of a mirror, and on a floor of a mix of a diffuse and a glossy child; and fur in
    front of a mirror renders the same under -dual-scatter as without. Mode M under
    a -noise budget blurs too, and mode J whose surface map comes back empty blurs as mode D does.
  * The camera's own motion, in the box held still: a camera with `motion` keys and no shutter
    renders byte for byte the still camera at its keys' instant (CPU, and GPU); with the shutter,
    the mean of 32 renders from its poses through it (mode R; modes D and B agree with mode R, and
    mode R on the GPU with mode R on the CPU, modes D and B on the GPU, B on the wavefront too,
    with mode R). The middle frame of a straight, even
    `camera_path` with a shutter blurs as the mean of the eyes half a frame either side. The
    loader refuses camera keys without a `motion_interval`, a single key, and a key's `fov_y` on
    a camera whose `lens` sets its view. A key without `fov_y` keeps the camera's own (the same
    render as the key writing it); a keyed camera without a shutter, rendered in a flight beside
    one that blurs, stays sharp; a moving camera keeps `-radcache` in a still scene; keys that all
    agree are a still camera (it renders on the GPU); and `-fur-volume` steps aside for a coat
    whose strands move -- and keeps one seen only at the instant.
  * A spot on its group's pivot, turning 30 degrees across the shutter, lights the middle of its
    sweep along a wall as brightly as the rest, and off the pivot needs no more keys than the
    budget. Keys whose only mover is a light they would stretch unevenly are no motion at all.
    Lights scale with their group (a uniform scale for every kind, an uneven stretch and shear for
    a flat panel and an emissive mesh, and for a ball and a tube round at the instant, which are
    built as emissive meshes), their radiance kept and their area grown. Haze under a
    sweep far wider than the scene lights the floor as it does without the shutter (modes R, D
    and B; on the GPU modes R and B, under the sun and the sky, and D). The blurred sky mode B sees in a batch with under one sample a pixel is all there.
  * The loader refuses, with a message naming why: a `motion` block with no `motion_interval`; one
    with a single key; `translate` beside `motion`; fur on, or a medium bounded by, a moving
    object; an interval that excludes the instant; `shutter_close` before `shutter_open`. It keeps
    a sun in a moving group, and a spot its keys stretch unevenly, at the instant, saying so -- and a render
    held sharp late in the shutter shows the sun's shadows where the instant has them while its
    sibling balls (one an isosurface) have moved -- and builds
    a sphere its keys stretch as a mesh, as it does one an uneven static scale stretches.
  * Mode P on the GPU with the scene held on the device across its batches renders what it renders
    uploading it for each, and GPU mode M's chunked photon deposit is one map per seed; mode P
    under -radcache on the CPU renders one image per seed.
Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
W, H = 96, 72
SPP = 1024                   # blurred mode R, per seed
TIMES = 32                   # static instants averaged for the reference
SPP_T = 64                   # per instant, per seed

# The moving groups, each a list of keys spread evenly over the motion interval [-0.5, 0.5]; a key
# sets translate / rotate / scale / shear, and the loader interpolates each linearly between keys.
SLIDE = [dict(translate=(-0.8, 0.35, 0.3)), dict(translate=(0.8, 0.35, 0.3))]
TURN = [dict(translate=(0.9, 0.3, -0.5)), dict(translate=(0.9, 0.3, -0.5), rotate=(0, 90, 0))]
# A parent that kinks at the instant -- up, then sideways -- holding a child that turns at its own
# pace: the child's keys must still include the parent's kink and the instant.
RISE = [dict(translate=(-0.9, 0.0, -0.4)), dict(translate=(-0.9, 0.4, -0.4)),
        dict(translate=(-0.5, 0.45, -0.4))]
SPIN = [dict(translate=(0, 0.3, 0)), dict(translate=(0, 0.3, 0), rotate=(0, 0, 15))]
# Scale and shear changing together: their product is not linear, so blending two keys is not
# the transform as written in between.
STRETCH = [dict(translate=(0.15, 0.12, -0.9)),
           dict(translate=(0.15, 0.12, -0.9), scale=(1.8, 1.8, 1.8), shear=(0.8, 0, 0))]
# Extras for the instant test: a ball that waits, then flies far off the stage -- the moving
# scene's sweep reaching well past the static one's bounds -- over a floor read through `cavity`,
# whose default probe radius is sized by those bounds.
FLY = [dict(translate=(-0.5, 1.6, -1.2)), dict(translate=(-0.5, 1.6, -1.2)),
       dict(translate=(60.0, 1.6, -1.2))]
# A ball orbiting its group's axis, two whole turns each side of the instant: a stretch between
# breaks whose midpoint equals the blend of its ends, so only a split by turn can follow it (used
# by the sharp-at-a-time tests).
TWIRL = [dict(translate=(0.55, 1.25, -1.0)), dict(translate=(0.55, 1.25, -1.0), rotate=(0, 0, 1440))]
FIELDS = (("translate", (0, 0, 0)), ("rotate", (0, 0, 0)), ("scale", (1, 1, 1)), ("shear", (0, 0, 0)))
# A cube of half-size 1 about the origin, for mesh_instance.
CUBE_OBJ = NL.join(["v %d %d %d" % (x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)] +
                   ["f 1 3 4 2", "f 5 6 8 7", "f 1 2 6 5", "f 3 7 8 4", "f 1 5 7 3", "f 2 4 8 6"]) + NL
# A group that moves, turns and grows, holding an isosurface ball, a tube and a cube instance.
KINDS = [dict(translate=(-0.5, 0.5, 0.0), rotate=(0, -40, 0), scale=(0.8, 0.8, 0.8)),
         dict(translate=(0.5, 0.6, 0.0), rotate=(0, 50, 20), scale=(1.2, 1.2, 1.2))]
# A dark ball sweeping across an open sky: the forward modes add the sky they see directly by a
# camera-side pass of their own, which must see the ball where each time in the shutter has it.
SWEEP = [dict(translate=(-0.9, 0.9, 0)), dict(translate=(0.9, 0.9, 0))]


def at(keys, u):
    """The group's fields at u in [0, 1], interpolated linearly between its evenly spread keys."""
    s = min(max(u, 0.0), 1.0) * (len(keys) - 1)
    i = min(int(s), len(keys) - 2)
    f = s - i
    out = {}
    for name, dflt in FIELDS:
        a, b = keys[i].get(name, dflt), keys[i + 1].get(name, dflt)
        out[name] = tuple(x + (y - x) * f for x, y in zip(a, b))
    return out


def v3(v, k=1.0):
    return " ".join(repr(float(x) * k) for x in v)


def fields(d, k):
    """A key's statements; translate is a length (times the unit factor k), the rest are not."""
    return "  ".join("%s %s" % (n, v3(d.get(n, dflt), k if n == "translate" else 1.0))
                     for n, dflt in FIELDS if n in d or n == "translate")


def xf(keys, u, k):
    """The group transform at u, or its motion block when u is None."""
    if u is None:
        return "motion { %s }" % "  ".join("key { %s }" % fields(key, k) for key in keys)
    return fields(at(keys, u), k)


def cube(h, m, k):
    """Six quads: a cube of half-size h about the origin, in material m, lengths times k."""
    f = lambda *xs: " ".join(repr(x * h * k) for x in xs)
    return NL.join([
        "quad { origin %s  u %s  v %s  material %s }" % (f(-1, -1, 1), f(2, 0, 0), f(0, 2, 0), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(1, -1, -1), f(-2, 0, 0), f(0, 2, 0), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(-1, -1, -1), f(0, 0, 2), f(0, 2, 0), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(1, -1, 1), f(0, 0, -2), f(0, 2, 0), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(-1, 1, 1), f(2, 0, 0), f(0, 0, -2), m),
        "quad { origin %s  u %s  v %s  material %s }" % (f(-1, -1, -1), f(2, 0, 0), f(0, 0, 2), m),
    ])


def scene(u=None, shutter=None, units="meters", extras=False, twirl=False):
    """The box scene: its groups moving when u is None, else held at u. `shutter` is
    (open, close) in frames; `units` "centimeters" writes every length 100 times larger."""
    k = {"meters": 1.0, "centimeters": 100.0}[units]
    L = lambda *xs: " ".join(repr(x * k) for x in xs)
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    lines = [
        "scene { units %s  motion_interval -0.5 0.5 }" % units,
        'material "white" { type diffuse reflect 0.7 }',
        'material "red"   { type diffuse reflect rgb 0.7 0.15 0.1 }',
        'material "blue"  { type diffuse reflect rgb 0.1 0.2 0.7 }',
        'material "green" { type diffuse reflect rgb 0.15 0.6 0.2 }',
        'material "gold"  { type glossy reflect rgb 0.9 0.7 0.3  roughness 0.15 }',
        "quad { origin %s  u %s  v %s  material %s }" % (L(-2, 0, -2), L(0, 0, 4), L(4, 0, 0),
                                                          "cavfloor" if extras else "white"),
        "quad { origin %s  u %s  v %s  material white }" % (L(-2, 0, -2), L(4, 0, 0), L(0, 3, 0)),
        "light area { origin %s  u %s  v %s  normal 0 -1 0  spd preset:bb6500  power 60 }"
        % (L(-0.4, 2.9, -0.4), L(0.8, 0, 0), L(0, 0, 0.8)),
        "group { %s  sphere { center 0 0 0  radius %r  material gold } }" % (xf(SLIDE, u, k), 0.3 * k),
        "group { %s  %s }" % (xf(TURN, u, k), cube(0.25, "red", k)),
        "group { %s  group { %s  %s } }" % (xf(RISE, u, k), xf(SPIN, u, k), cube(0.18, "blue", k)),
        "group { %s  %s }" % (xf(STRETCH, u, k), cube(0.1, "green", k)),
        'camera "c" { eye %s  look_at %s  up 0 1 0  fov_y 40  %sfilm { res %d %d } }'
        % (L(0, 1.0, 4.2), L(0, 0.5, 0), cam, W, H),
    ]
    if twirl:
        lines.append("group { %s  sphere { center %s  radius %r  material red } }"
                     % (xf(TWIRL, u, k), L(0.45, 0, 0), 0.15 * k))
    if extras:
        # The floor's albedo follows `cavity`, whose default probe radius is a share of the
        # scene's size -- which a mover's sweep must not change.
        lines += ['pattern "cav" { expr "0.25 + 0.6 * cavity" }',
                  'material "cavfloor" { type diffuse reflect pattern:cav }',
                  "light env { spd preset:d65 }",
                  "group { %s  sphere { center 0 0 0  radius %r  material gold } }"
                  % (xf(FLY, u, k), 0.15 * k)]
    return NL.join(lines) + NL


def sky_scene(shutter):
    """The SWEEP ball over a floor under an environment light, nothing behind it but sky."""
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.6 }',
        'material "dark" { type diffuse reflect 0.1 }',
        "light env { spd preset:d65 }",
        "quad { origin -3 0 -3  u 0 0 6  v 6 0 0  material white }",
        "group { %s  sphere { center 0 0 0  radius 0.3  material dark } }" % xf(SWEEP, None, 1.0),
        'camera "c" { eye 0 0.8 3.5  look_at 0 0.8 0  up 0 1 0  fov_y 40  '
        'shutter_open %r  shutter_close %r  film { res %d %d } }' % (shutter + (W, H)),
    ]) + NL


def kinds_scene(cube_obj, u=None, shutter=None):
    """The KINDS group over the box's floor, moving, or held at u."""
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    body = ("isosurface { material blue  sphere { center -0.25 0 0  radius 0.15 } }  "
            "curve { material red  basis linear  radius 0.05  point 0 -0.2 0  point 0.1 0.25 0.1 }  "
            'mesh_instance { of "cube"  translate 0.3 0 0  rotate 0 30 0  scale 0.12  material green }')
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'material "red"   { type diffuse reflect rgb 0.7 0.15 0.1 }',
        'material "blue"  { type diffuse reflect rgb 0.1 0.2 0.7 }',
        'material "green" { type diffuse reflect rgb 0.15 0.6 0.2 }',
        'mesh_asset "cube" { file "%s"  material white }' % cube_obj,
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        LIGHT,
        "group { %s  %s }" % (xf(KINDS, u, 1.0), body),
        'camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 40  %sfilm { res %d %d } }'
        % (cam, W, H),
    ]) + NL


def reach_scene(light, shutter=(-0.5, 0.5)):
    """A big ball whose sweep runs far past the instant's bounding sphere (a small floor, the ball
    over it at the instant), lit by `light` alone: the sky's or the sun's light must reach it
    there, and its shadow must fall back on the floor."""
    keys = [dict(translate=(-1.8, 0.45, 0.0)), dict(translate=(1.8, 0.45, 0.0))]
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.6 }',
        'material "gold"  { type diffuse reflect rgb 0.8 0.6 0.2 }',
        light,
        "quad { origin -0.3 0 -0.3  u 0 0 0.6  v 0.6 0 0  material white }",
        "group { %s  sphere { center 0 0 0  radius 0.45  material gold } }" % xf(keys, None, 1.0),
        'camera "c" { eye 0 1.2 6  look_at 0 0.4 0  up 0 1 0  fov_y 45  '
        'shutter_open %r  shutter_close %r  film { res %d %d } }' % (shutter + (W, H)),
    ]) + NL


def specular_scene(shutter=(-0.5, 0.5)):
    """Mirrors and glass that move, for mode B's analytic camera connections, which must meet each
    where it is at the photon's time: a mirror panel on the back wall sliding and turning, a mirror
    floor tile whose two halves move apart (one plane of faces moving differently -- a mixed plane,
    met by its parts), and a glass ball and a mirror ball crossing the room."""
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.6 }',
        'material "red"   { type diffuse reflect rgb 0.7 0.15 0.1 }',
        'material "mir"   { type mirror }',
        'material "glass" { type dielectric ior 1.5 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material red }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }",
        "group { motion { key { translate -0.3 0 0  rotate 0 -8 0 } key { translate 0.3 0 0  rotate 0 8 0 } }  "
        "quad { origin -0.6 0.4 -1.95  u 1.2 0 0  v 0 1.2 0  material mir } }",
        "group { motion { key { translate 0 0 -0.2 } key { translate 0 0 0.2 } }  "
        "quad { origin -1.5 0.002 0.2  u 0 0 0.8  v 0.7 0 0  material mir } }",
        "quad { origin -0.8 0.002 0.2  u 0 0 0.8  v 0.7 0 0  material mir }",
        "group { motion { key { translate -0.6 0.45 0.3 } key { translate 0.6 0.45 0.3 } }  "
        "sphere { center 0 0 0  radius 0.3  material glass } }",
        "group { motion { key { translate 0.9 0.35 -0.9 } key { translate 0.9 0.35 0.0 } }  "
        "sphere { center 0 0 0  radius 0.25  material mir } }",
        'camera "c" { eye 0 1.2 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 45  %sfilm { res %d %d } }' % (cam, W, H),
    ]) + NL


# A group that moves and turns, holding an area panel, a glowing ball, a spot, a tube, an emissive
# triangle and an aimed panel (its normal leaning off its face).
LIGHTS = [dict(translate=(-0.5, 1.7, 0.0), rotate=(0, -30, 0)),
          dict(translate=(0.5, 1.5, 0.2), rotate=(0, 40, 10))]


def lights_scene(obj, u=None, shutter=None):
    """The LIGHTS group in the box, moving, or held at u: it lights the box alone."""
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    body = ("light area { origin -0.15 0 -0.15  u 0.3 0 0  v 0 0 0.3  normal 0 -1 0  "
            "spd preset:bb6500  power 25 }  "
            "light sphere { center 0.45 -0.1 0  radius 0.06  spd preset:bb5000  power 12 }  "
            "light spot { origin -0.45 0 0  dir 0.4 -1 0.2  inner_angle 15  outer_angle 30  "
            "spd preset:bb6500  power 12 }  "
            "light cylinder { center 0 -0.2 -0.3  axis 1 0 0  length 0.4  radius 0.02  "
            "spd preset:bb4000  power 8 }  "
            "light area { origin 0.25 0.1 0.25  u 0.2 0 0  v 0 0 0.2  normal 0.6 -1 0.3  "
            "spd preset:bb5600  power 10 }  "
            'mesh "glow_tri" { file "%s"  material glow  translate 0.1 -0.3 0.35  rotate 90 0 0  scale 1.5 }'
            % obj)
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'material "red"   { type diffuse reflect rgb 0.7 0.15 0.1 }',
        'material "glow"  { type diffuse reflect 0.0  emit blackbody 5200 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "quad { origin -2 0 -2  u 0 3 0  v 0 0 4  material red }",
        "group { %s  %s }" % (xf(LIGHTS, u, 1.0), body),
        'camera "c" { eye 0 1.0 4.2  look_at 0 0.7 0  up 0 1 0  fov_y 45  %sfilm { res %d %d } }'
        % (cam, W, H),
    ]) + NL


# A group that slides 1.6 m and turns, holding an aimed panel, a spot and a ball, the scene's only
# lights: where each is at a time is most of what the floor and the wall show.
LONE = [dict(translate=(-0.8, 1.6, 0.0), rotate=(0, -25, 0)),
        dict(translate=(0.8, 1.4, 0.3), rotate=(0, 35, 8))]


def lone_lights_scene(u=None, shutter=None):
    """An aimed panel, a spot and a ball alone in the LONE group, moving, or held at u."""
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    body = ("light area { origin -0.15 0 -0.15  u 0.3 0 0  v 0 0 0.3  normal 0.7 -1 0.2  "
            "spd preset:bb5600  power 25 }  "
            "light spot { origin 0.3 0 0  dir 0.3 -1 -0.4  inner_angle 15  outer_angle 25  "
            "spd preset:bb6500  power 15 }  "
            # A ball turning with its group: its samples must lie on its moved surface, or its own
            # surface shadows its light
            "light sphere { center -0.35 0 0.2  radius 0.06  spd preset:bb5000  power 12 }")
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "group { %s  %s }" % (xf(LONE, u, 1.0), body),
        'camera "c" { eye 0 1.0 4.2  look_at 0 0.7 0  up 0 1 0  fov_y 45  %sfilm { res %d %d } }'
        % (cam, W, H),
    ]) + NL


# Groups whose keys scale their lights: a similarity growing 0.6 to 1.5 times as it slides and
# turns, and an uneven stretch with a shear.
GROW = [dict(translate=(-0.4, 1.5, 0.0), rotate=(0, -20, 0), scale=(0.6, 0.6, 0.6)),
        dict(translate=(0.4, 1.6, 0.2), rotate=(0, 30, 10), scale=(1.5, 1.5, 1.5))]
WARP = [dict(translate=(-0.4, 1.5, 0.0)),
        dict(translate=(0.4, 1.5, 0.2), rotate=(0, 25, 0), scale=(1.6, 0.7, 1.2), shear=(0.3, 0, 0))]
# Round at the instant (the middle key), stretched and sheared either side of it.
STRETCH = [dict(translate=(-0.4, 1.5, 0.0), scale=(0.7, 1.3, 1.0)), dict(translate=(0.0, 1.55, 0.1)),
           dict(translate=(0.4, 1.6, 0.2), rotate=(0, 25, 0), scale=(1.5, 0.7, 1.2), shear=(0.3, 0, 0))]


def scaled_lights_scene(obj, kind, u=None, shutter=None):
    """Lights in a group its keys scale, moving, or held at u: `kind` "grow" (GROW, a similarity) holds
    every kind of light -- a panel, a ball, a tube, a spot, an aimed panel and an emissive triangle;
    "warp" (WARP, any affine) a panel and an emissive triangle; "stretch" (STRETCH) a panel, a ball
    and a tube, round at the instant and built as emissive meshes for the keys. Each light
    is given its radiance (no `power`, which would hold its flux at the instant's), so held still
    at a time it is the same light as the moving one there."""
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    panel = "light area { origin -0.15 0 -0.15  u 0.3 0 0  v 0 0 0.3  normal 0 -1 0  spd preset:bb6500 }"
    tri = ('mesh "glow_tri" { file "%s"  material glow  translate 0.1 -0.3 0.35  rotate 90 0 0  scale 1.5 }'
           % obj)
    if kind == "grow":
        body = "  ".join([
            panel,
            "light sphere { center 0.45 -0.1 0  radius 0.06  spd preset:bb5000 }",
            "light spot { origin -0.45 0 0  dir 0.4 -1 0.2  inner_angle 15  outer_angle 30  "
            "spd preset:bb6500  power 12 }",
            "light cylinder { center 0 -0.2 -0.3  axis 1 0 0  length 0.4  radius 0.02  spd preset:bb4000 }",
            "light area { origin 0.25 0.1 0.25  u 0.2 0 0  v 0 0 0.2  normal 0.6 -1 0.3  spd preset:bb5600 }",
            tri])
        keys = GROW
    elif kind == "stretch":
        body = "  ".join([
            panel,
            "light sphere { center 0.3 -0.1 0  radius 0.08  spd preset:bb5000 }",
            "light cylinder { center -0.2 -0.2 -0.3  axis 1 0 0.3  length 0.4  radius 0.03  caps on  "
            "spd preset:bb4000 }"])
        keys = STRETCH
    else:
        body = "  ".join([panel, tri])
        keys = WARP
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'material "glow"  { type diffuse reflect 0.0  emit blackbody 5200 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "group { %s  %s }" % (xf(keys, u, 1.0), body),
        'camera "c" { eye 0 1.0 4.2  look_at 0 0.7 0  up 0 1 0  fov_y 45  %sfilm { res %d %d } }'
        % (cam, W, H),
    ]) + NL


# A glowing sheet a keyed deformer grows 0.6 to 1.5 times as it slides 0.8 m: eight rbf handles
# at the corners of a box about it, each taken by the same scale and slide, so every corner of the
# sheet is (the affine part of an rbf, check_deform).
GLOW_S, GLOW_X = (0.6, 1.5), (-0.4, 0.4)


def glow_deform_scene(grid, u=None, shutter=None, scales=GLOW_S, glossy=False, strip=False):
    """The deforming glowing sheet (GLOW_S, GLOW_X) over a floor, radiating down, the scene's only
    light; or held still at u, the sheet scaled and moved there by its own transform. `scales`: its
    scale at the interval's ends, or at its start, the instant and its end (three: piecewise
    linear between them, so it can pass through a point at the instant); `glossy`: the floor a
    rough mirror, so the glossy bounces' MIS
    weighs its hits; `strip`: a `shape_only` mesh loaded before it, whose triangles are stripped
    from the scene, renumbering the sheet's."""
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    def at(u):   # the sheet's scale and slide at fraction u of the interval
        x = GLOW_X[0] + (GLOW_X[1] - GLOW_X[0]) * u
        if len(scales) == 2:
            return scales[0] + (scales[1] - scales[0]) * u, x
        k = 0 if u <= 0.5 else 1
        return scales[k] + (scales[k + 1] - scales[k]) * (u - 0.5 * k) / 0.5, x
    if u is None:
        def to(p, k):   # handle corner p at the k-th of three keys, the middle one the instant's
            s, x = at(k / 2.0)
            return "key { to %r %r %r }" % (s * p[0] + x, s * p[1], s * p[2])
        corners = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
        handles = "  ".join("handle { at %r %r %r  motion { %s } }" % (p + (" ".join(to(p, k) for k in range(3)),))
                            for p in corners)
        shape = "deform rbf { kernel triharmonic  %s }" % handles
    else:
        shape = "scale %r  translate %r 0 0" % at(u)
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'material "glow"  { type diffuse reflect 0.0  emit blackbody 5200 }',
        'material "gloss" { type glossy reflect 0.7  roughness 0.25 }',
        ('mesh "proxy" { file "%s"  material white  shape_only yes }' % grid) if strip else "",
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material %s }" % ("gloss" if glossy else "white"),
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        'group { translate 0 1.5 0  mesh "glow" { file "%s"  material glow  emit_orient flip  %s } }'
        % (grid, shape),
        'camera "c" { eye 0 1.0 4.2  look_at 0 0.7 0  up 0 1 0  fov_y 45  %sfilm { res %d %d } }'
        % (cam, W, H),
    ]) + NL


def still_scene(obj, u=None, shutter=None):
    """A moving group holding a ball, an isosurface ball and a sun: the two balls move, the sun
    (which cannot: it is not anywhere) stays at the instant. With `u`, the static equivalent: the
    balls held at u and the sun outside the group."""
    keys = [dict(translate=(-0.6, 0.45, 0.2)), dict(translate=(0.6, 0.45, 0.2))]
    ball = ("sphere { center 0 0 0  radius 0.25  material gold }  "
            "isosurface { material blue  sphere { center 0.3 0.15 0  radius 0.15 } }")
    still = "light sun { dir 0.4 1 0.3  angle 2  spd preset:d65 }"
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    lines = [
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'material "blue"  { type diffuse reflect rgb 0.1 0.2 0.7 }',
        'material "gold"  { type glossy reflect rgb 0.9 0.7 0.3  roughness 0.15 }',
        'material "glow"  { type diffuse reflect 0.0  emit blackbody 5200 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
        "  power 60 }",
    ]
    if u is None:
        lines.append("group { %s  %s  %s }" % (xf(keys, None, 1.0), ball, still))
    else:
        lines.append("group { %s  %s }" % (xf(keys, u, 1.0), ball))
        lines.append("group { %s  %s }" % (xf(keys, 0.5, 1.0), still))
    lines.append('camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 40  %sfilm { res %d %d } }'
                 % (cam, W, H))
    return NL.join(lines) + NL


def stripe_scene(u=None, shutter=None, shift=0.0):
    """A ball striped by a world-space pattern, sliding left to right. Moving (u None), its
    stripes must travel with it; held at u with the pattern shifted by `shift` in x, the static
    equivalent of that."""
    keys = [dict(translate=(-0.6, 0.45, 0.2)), dict(translate=(0.6, 0.45, 0.2))]
    cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    return NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.7 }',
        'pattern "stripes" { expr "0.5 + 0.4 * sin(25 * (x - %r))" }' % shift,
        'material "striped" { type diffuse reflect pattern:stripes }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
        "  power 60 }",
        "group { %s  sphere { center 0 0 0  radius 0.35  material striped } }" % xf(keys, u, 1.0),
        'camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 40  %sfilm { res %d %d } }'
        % (cam, W, H),
    ]) + NL


# The moving kinds -checkbvh fires its rays through: spheres keep still axes, the rest also turn.
BVH_KINDS = ["sphere { center 0 0 0  radius 0.03  material w }",
             "quad { origin -0.05 -0.05 0  u 0.1 0 0  v 0 0.1 0  material w }",
             "isosurface { material w  sphere { radius 0.04 } }",
             "curve { material w  basis linear  radius 0.01  point -0.05 0 0  point 0.05 0.02 0 }",
             'mesh_instance { of "cube"  scale 0.04 }']


def bvh_scene(cube_obj):
    """Many small moving primitives of every kind inside the unit cube -checkbvh fires its rays
    through."""
    rng = np.random.default_rng(7)
    out = ["scene { units meters  motion_interval -0.5 0.5 }",
           'material "w" { type diffuse reflect 0.5 }',
           'mesh_asset "cube" { file "%s"  material w }' % cube_obj]
    for k in range(60):
        a, b = rng.uniform(0.1, 0.9, 3), rng.uniform(0.1, 0.9, 3)
        r0, r1 = rng.uniform(-90, 90, 3), rng.uniform(-90, 90, 3)
        kind = k % len(BVH_KINDS)
        turn = kind != 0
        out.append("group { motion { key { translate %s  rotate %s }  key { translate %s  "
                   "rotate %s } }  %s }" % (v3(a), v3(r0 if turn else (0, 0, 0)), v3(b),
                                            v3(r1 if turn else (0, 0, 0)), BVH_KINDS[kind]))
    out.append("light area { origin 0.4 2 0.4  u 0.2 0 0  v 0 0 0.2  normal 0 -1 0  power 10 }")
    out.append('camera "c" { eye 0.5 0.5 3  look_at 0.5 0.5 0.5  up 0 1 0  fov_y 40 }')
    return NL.join(out) + NL


# (what is refused, the scene, a phrase its message must hold)
REFUSED = [
    ("motion without motion_interval",
     "group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  sphere { center 0 0 0 "
     "radius 0.1 material w } }", "needs the scene's `motion_interval", False),
    ("a single key",
     "group { motion { key { translate 0 0 0 } }  sphere { center 0 0 0  radius 0.1 material w } }",
     "at least two `key`", True),
    ("translate beside motion",
     "group { translate 1 0 0  motion { key { translate 0 0 0 } key { translate 1 0 0 } }  "
     "sphere { center 0 0 0  radius 0.1 material w } }", "beside a `motion` block", True),
    ("fur on a moving object",
     'group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  sphere "ball" { center 0 0 '
     '0  radius 0.2  material w } }' + NL + 'fur "coat" { on "ball"  material w  count 10  length 0.02 }',
     "cannot follow a moving object", True),
    ("a medium bounded by a moving object",
     'group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  sphere "ball" { center 0 0 '
     '0  radius 0.2  material w } }' + NL +
     'medium { sigma_t 1  albedo 0.9  bounds { object "ball" } }',
     "cannot follow a moving object", True),
]
# (what stays at the instant, the scene, the warning it must print) -- loaded, not refused
STILL = [
    ("a sun in a moving group",
     "group { motion { key { translate 0 0 0 } key { translate 1 0 0 } }  light sun { dir 0.3 1 0.2  "
     "angle 1  spd preset:d65 } }",
     "a sun or sky light in a moving group stays where it is"),
    ("a spot light its keys stretch unevenly",
     "group { motion { key { scale 1 } key { scale 1 } key { scale 1.5 1 1 } }  light spot { origin 0 1 0  "
     "dir 0 -1 0  spd preset:bb6500  power 5 } }", "whose keys stretch it unevenly stays where it is"),
]
LIGHT = ("light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500"
         "  power 60 }")
# A sphere uniform at the instant (the middle key) that its last key stretches: it must load, as a
# tessellated mesh -- an analytic sphere cannot become an ellipsoid.
UNEVEN = ("group { motion { key { scale 1 } key { scale 1 } key { scale 1 2 1 } }  sphere { center "
          "0 0 0  radius 0.1  material w } }")
# The same, the unevenness from a static ancestor: turning inside `scale 1 2 1` (undone below the
# turn, so the sphere is round at the instant) is no similarity.
UNEVEN_ANCESTOR = ("group { scale 1 2 1  group { motion { key { rotate 0 0 -45 } key { rotate 0 0 45 } }  "
                   "group { scale 1 0.5 1  sphere { center 0 0 0  radius 0.1  material w } } } }")


def grid_obj(path, n=21, size=2.0):
    """An n x n vertex grid in the xz plane (y = 0), `size` across, centred on the origin."""
    lines = []
    for j in range(n):
        for i in range(n):
            lines.append("v %r 0 %r" % (-size / 2 + size * i / (n - 1), -size / 2 + size * j / (n - 1)))
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i + 1
            lines.append("f %d %d %d" % (a, a + n, a + 1))
            lines.append("f %d %d %d" % (a + 1, a + n, a + n + 1))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(lines) + NL)


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr


def read_pfm(path):
    """A PFM as h x w x 3, top row first."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, 3)
    return data[::-1].astype(np.float64)


def remove(f):
    """Delete `f` if it exists, waiting out another process (a scanner, an indexer) that holds it
    open for a moment: the run's last render of that name must not be read in place of the next."""
    for _ in range(40):
        try:
            if os.path.exists(f):
                os.remove(f)
            return
        except PermissionError:
            time.sleep(0.25)
    os.remove(f)


def render(exe, out, path, tag, args, seed=1, want_gpu=False):
    """Render `path` to <out>/<tag>.png; returns (HDR image, raw PFM bytes, log)."""
    png = os.path.join(out, tag + ".png")
    pfm = png[:-4] + ".pfm"
    for f in (png, pfm):
        remove(f)
    cmd = [exe, "-in", path, "-seed", str(seed), "-hdr", "-no-meter", "-o", png] + args
    rc, log = run(cmd)
    if rc != 0 or not os.path.exists(pfm):
        print(log[-2000:])
        raise SystemExit("render failed (exit %d): %s" % (rc, " ".join(cmd)))
    if want_gpu and "-> GPU" not in log:
        raise SystemExit("did not run on the GPU: " + " ".join(cmd))
    with open(pfm, "rb") as f:
        raw = f.read()
    return read_pfm(pfm), raw, log


def lum(img):
    return img @ np.array([0.2126, 0.7152, 0.0722])


def blocks(img, b=8):
    h, w = img.shape[0] // b * b, img.shape[1] // b * b
    return img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def block_err(a, b, n=8):
    """Standard error of the blocks of the mean of two seeds' images a, b (from their difference)."""
    d = a - b
    h, w = d.shape[0] // n * n, d.shape[1] // n * n
    s = (d[:h, :w] ** 2).reshape(h // n, n, w // n, n).sum(axis=(1, 3))
    return np.sqrt(s) / (2.0 * n * n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_motion"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True

    def verdict(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print("%-58s %s%s" % (name, "ok" if good else "FAIL", ("  " + detail) if detail else ""))

    # --- the loader's refusals ---
    base = 'material "w" { type diffuse reflect 0.5 }' + NL + LIGHT
    obj = os.path.join(out, "tri.obj")
    write(obj, "v 0 0 0" + NL + "v 0.2 0 0" + NL + "v 0 0.2 0" + NL + "f 1 2 3" + NL)
    cube_obj = os.path.join(out, "cube.obj").replace(chr(92), "/")
    write(cube_obj, CUBE_OBJ)
    for name, body, phrase, interval in REFUSED:
        p = os.path.join(out, "refuse.ftsl")
        write(p, NL.join(["scene { units meters%s }" % ("  motion_interval -0.5 0.5" if interval
                                                        else ""), base,
                          body % {"obj": obj.replace(chr(92), "/")}]) + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        verdict("refuses " + name, rc != 0 and phrase in log,
                "" if rc != 0 and phrase in log else log.strip()[-300:])
    for name, body, phrase in STILL:
        p = os.path.join(out, "still.ftsl")
        write(p, NL.join(["scene { units meters  motion_interval -0.5 0.5 }", base,
                          body % {"obj": obj.replace(chr(92), "/")}]) + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        good = rc == 0 and phrase in log
        verdict("keeps %s at the instant, saying so" % name, good, "" if good else log.strip()[-300:])
    # ...and render so: held sharp late in the shutter, the ball has moved and the triangle and
    # the isosurface have not.
    objf = obj.replace(chr(92), "/")
    sharp, ref = os.path.join(out, "still_sharp.ftsl"), os.path.join(out, "still_ref.ftsl")
    write(sharp, still_scene(objf, shutter=(0.4, 0.4)))
    write(ref, still_scene(objf, u=0.9))
    ra = [lum(render(exe, out, sharp, "still_sharp_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                              "-spp", "256"], seed=s)[0]) for s in (1, 2)]
    rb = [lum(render(exe, out, ref, "still_ref_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                          "-spp", "256"], seed=s)[0]) for s in (1, 2)]
    ma, mr = blocks((ra[0] + ra[1]) / 2), blocks((rb[0] + rb[1]) / 2)
    err = np.hypot(block_err(ra[0], ra[1]), block_err(rb[0], rb[1]))
    z = np.abs(ma - mr) / np.maximum(err, 1e-30)
    rel = np.abs(ma - mr) / np.maximum(np.abs(mr), 1e-3 * np.abs(mr).max())
    bad = (z > 5.0) & (rel > 0.03)
    verdict("the kinds that cannot move stay; their siblings move", not bad.any(),
            "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))
    # Lights move with their group -- an area panel, a glowing ball, a spot, a tube, an emissive
    # triangle and an aimed panel -- in every mode that blurs: held sharp at a time, the scene is
    # the static group written out there. And an aimed panel and a spot alone, sliding far, in
    # every mode on both devices: modes U and S take the time a pass, and U's light paths and
    # connections carry each light there.
    light_modes = [("R", "cpu", ["-spp", "512"]), ("B", "cpu", ["-n", "40000000"]), ("D", "cpu", ["-spp", "256"])]
    lone_modes = light_modes + [("U", "cpu", ["-spp", "256"]), ("S", "cpu", ["-spp", "128", "-n", "200000"])]
    if not a.no_gpu:
        lone_modes += [("R", "gpu", ["-spp", "512"]), ("B", "gpu", ["-n", "40000000"]),
                       ("U", "gpu", ["-spp", "256"]), ("S", "gpu", ["-spp", "128", "-n", "200000"]),
                       # J with the 64 epochs a blur gives it, sharp or still: its surface radius
                       # shrinks with the epochs, so fewer would be another bias in its dark blocks.
                       # M at 8 M photons: at 2 M a dark block reads 5 sigma off from map to map.
                       ("J", "gpu", ["-spp", "256", "-jsurf", "-beamepochs", "64"]),
                       ("M", "gpu", ["-spp", "128", "-n", "8000000"])]
    # A glowing sheet that deforms, in every mode on both devices, D on the GPU too: its light
    # samples on its triangles as they are at the time, their area then in every density.
    gridf = os.path.join(out, "glow_grid.obj").replace(chr(92), "/")
    grid_obj(gridf, n=9, size=1.0)
    glow_modes = lone_modes + ([] if a.no_gpu else [("D", "gpu", ["-spp", "256"])])
    held = ([(t, "lights", lambda u=None, shutter=None: lights_scene(objf, u, shutter), light_modes)
             for t in (-0.3, 0.35)] + [(0.35, "three lights alone", lone_lights_scene, lone_modes)] +
            # lights their keys scale, their radiance kept and their area grown
            [(t, "lights their keys grow", lambda u=None, shutter=None: scaled_lights_scene(objf, "grow", u, shutter),
              lone_modes) for t in (-0.35, 0.4)] +
            [(0.4, "flat lights their keys stretch", lambda u=None, shutter=None: scaled_lights_scene(objf, "warp", u, shutter),
              lone_modes)] +
            # a ball and a tube its keys stretch, round at the instant: emissive meshes
            [(0.4, "a ball and a tube their keys stretch", lambda u=None, shutter=None: scaled_lights_scene(objf, "stretch", u, shutter),
              lone_modes)] +
            [(t, "a deforming glowing mesh", lambda u=None, shutter=None: glow_deform_scene(gridf, u, shutter),
              glow_modes) for t in (-0.35, 0.4)] +
            # ... loaded after a shape_only mesh, whose stripped triangles renumber the sheet's
            [(0.4, "a deforming glowing mesh after a shape_only one",
              lambda u=None, shutter=None: glow_deform_scene(gridf, u, shutter, strip=True),
              [m for m in glow_modes if (m[0], m[1]) in (("R", "cpu"), ("B", "cpu"), ("R", "gpu"), ("B", "gpu"),
                                                         ("D", "gpu"))])] +
            # ... passing through a point at the instant: its triangles, with no area then, chosen by
            # their keys' mean area
            [(0.4, "a deforming glowing mesh flat at the instant",
              lambda u=None, shutter=None: glow_deform_scene(gridf, u, shutter, scales=(0.8, 0.0, 1.2)),
              [m for m in glow_modes if (m[0], m[1]) in (("R", "cpu"), ("B", "cpu"), ("D", "cpu"), ("R", "gpu"),
                                                         ("B", "gpu"), ("D", "gpu"))])] +
            # ... over a rough mirror, grown 2.5 times since the instant: the glossy bounces' MIS
            # weighs each hit on it by its area then, as their NEE does
            [(0.4, "a deforming glowing mesh over a rough mirror",
              lambda u=None, shutter=None: glow_deform_scene(gridf, u, shutter, scales=(0.3, 2.0), glossy=True),
              [("M", "cpu", ["-spp", "128", "-n", "8000000"])] +
              [m for m in glow_modes if (m[0], m[1]) in (("S", "cpu"), ("R", "cpu"), ("M", "gpu"), ("S", "gpu"))])])
    for t, what, make, modes in held:
        sharp, ref = os.path.join(out, "lights_sharp.ftsl"), os.path.join(out, "lights_ref.ftsl")
        write(sharp, make(shutter=(t, t)))
        write(ref, make(u=t + 0.5))
        for mode, dev, margs in modes:
            args = ["-mode", mode, "-device", dev] + margs
            gpu = dev == "gpu"
            la = [lum(render(exe, out, sharp, "lights_sharp_s%d" % s, args, seed=s, want_gpu=gpu)[0])
                  for s in (1, 2)]
            lb = [lum(render(exe, out, ref, "lights_ref_s%d" % s, args, seed=s, want_gpu=gpu)[0])
                  for s in (1, 2)]
            # the merging and density modes' blocks are noisier (their tolerances elsewhere here)
            tol = {"S": 0.15, "M": 0.10, "J": 0.10, "U": 0.10}.get(mode, 0.05)
            good, detail = agree(la, lb, tol)
            verdict("%s move with their group (mode %s%s, sharp at %g)"
                    % (what, mode, " on the GPU" if gpu else "", t), good, detail)
    # A glowing sheet grown from nothing: at the interval's start it has no area, so it lights
    # nothing then (rather than the instant's sheet, or a division by its zero area).
    for mode, dev, margs in [m for m in glow_modes if m[0] in ("R", "B", "D", "U")]:
        args = ["-mode", mode, "-device", dev] + margs
        lit = {}
        for t in (-0.5, 0.4):
            p = os.path.join(out, "zero_area.ftsl")
            write(p, glow_deform_scene(gridf, shutter=(t, t), scales=(0.0, 1.5)))
            img = render(exe, out, p, "zero_area_%g" % t, args, want_gpu=dev == "gpu")[0]
            lit[t] = float(np.abs(lum(img)).mean())
        r = lit[-0.5] / max(lit[0.4], 1e-300)
        verdict("a light with no area at a time lights nothing then (mode %s%s)" % (mode, " on the GPU" if dev == "gpu" else ""),
                r < 1e-6 and np.isfinite(lit[-0.5]), "%.3g of its light at 0.4" % r)
    # A panel at its group's y = 0 whose last key flattens y (`scale 1 0 1`): a singular key that
    # leaves the panel as it is, so held sharp there it lights the room as its still twin does
    # (its area growth by the cofactor, where |det D| / |D^T n| is 0 / 0).
    panel = "light area { origin -0.3 0 -0.3  u 0.6 0 0  v 0 0 0.6  normal 0 -1 0  spd preset:bb6500 }"
    for nm, body, cam in (("sharp", "group { translate 0 1.5 0  group { motion { key { scale 1 1 1 } key { scale 1 1 1 } "
                                    "key { scale 1 0 1 } }  %s } }" % panel, "shutter_open 0.5  shutter_close 0.5  "),
                          ("ref", "group { translate 0 1.5 0  %s }" % panel, "")):
        write(os.path.join(out, "flatkey_%s.ftsl" % nm), NL.join([
            "scene { units meters  motion_interval -0.5 0.5 }", 'material "white" { type diffuse reflect 0.7 }',
            "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }", "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
            body, 'camera "c" { eye 0 1.0 4.2  look_at 0 0.7 0  up 0 1 0  fov_y 45  %sfilm { res %d %d } }' % (cam, W, H)]) + NL)
    for mode, dev, margs in [m for m in glow_modes if (m[0], m[1]) in (("R", "cpu"), ("B", "cpu"), ("R", "gpu"), ("B", "gpu"))]:
        args = ["-mode", mode, "-device", dev] + margs
        la = [lum(render(exe, out, os.path.join(out, "flatkey_sharp.ftsl"), "flatkey_sharp_s%d" % s, args, seed=s,
                         want_gpu=dev == "gpu")[0]) for s in (1, 2)]
        lb = [lum(render(exe, out, os.path.join(out, "flatkey_ref.ftsl"), "flatkey_ref_s%d" % s, args, seed=s,
                         want_gpu=dev == "gpu")[0]) for s in (1, 2)]
        good, detail = agree(la, lb)
        verdict("a panel its key flattens along its normal lights as before (mode %s%s)"
                % (mode, " on the GPU" if dev == "gpu" else ""), good, detail)
    # An isosurface, a curve and an instance move, turn and grow with their group: held sharp at
    # a time, they are the static group written out there.
    for t in (-0.35, 0.4):
        sharp, ref = os.path.join(out, "kinds_sharp.ftsl"), os.path.join(out, "kinds_ref.ftsl")
        write(sharp, kinds_scene(cube_obj, shutter=(t, t)))
        write(ref, kinds_scene(cube_obj, u=t + 0.5))
        args = ["-mode", "R", "-device", "cpu", "-spp", "256"]
        ka = [lum(render(exe, out, sharp, "kinds_sharp_s%d" % s, args, seed=s)[0]) for s in (1, 2)]
        kb = [lum(render(exe, out, ref, "kinds_ref_s%d" % s, args, seed=s)[0]) for s in (1, 2)]
        good, detail = agree(ka, kb)
        verdict("an isosurface, a curve and an instance move (sharp at %g)" % t, good, detail)
    for name, body in (("a sphere its keys stretch loads as a mesh", UNEVEN),
                       ("a sphere turning inside an uneven scale loads as a mesh", UNEVEN_ANCESTOR)):
        p = os.path.join(out, "uneven.ftsl")
        write(p, NL.join(["scene { units meters  motion_interval -0.5 0.5 }", base, body]) + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        m = re.search(r"(\d+) spheres, (\d+) tris", log)
        good = rc == 0 and m is not None and int(m.group(1)) == 0 and int(m.group(2)) > 100
        verdict(name, good, "" if good else log.strip()[-300:])
    # A ball or tube light an uneven static scale stretches: an emissive mesh, beside the base's panel.
    for name, light in (("a sphere light an uneven scale stretches loads as an emissive mesh",
                         "light sphere { center 0 1 0  radius 0.1  spd preset:bb6500  power 5 }"),
                        ("a tube light an uneven scale stretches loads as an emissive mesh",
                         "light cylinder { center 0 1 0  axis 1 0 0  length 0.4  radius 0.03  spd preset:bb6500  power 5 }")):
        p = os.path.join(out, "uneven_light.ftsl")
        write(p, NL.join(["scene { units meters }", base, "group { scale 1 2 1.5  shear 0.3 0 0  %s }" % light]) + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        m = re.search(r"(\d+) emitters, (\d+) spheres, (\d+) tris", log)
        good = rc == 0 and m is not None and int(m.group(1)) == 2 and int(m.group(2)) == 0 and int(m.group(3)) > 50
        verdict(name, good, "" if good else log.strip()[-300:])
    for name, sc, cam, phrase in (
            ("refuses an interval that excludes the instant", "motion_interval 0.2 1", "",
             "motion_interval"),
            ("refuses shutter_close before shutter_open", "motion_interval -0.5 0.5",
             "shutter_open 0.5  shutter_close -0.5  ", "comes before `shutter_open`")):
        p = os.path.join(out, "refuse.ftsl")
        write(p, NL.join(["scene { units meters  %s }" % sc, base,
                          "sphere { center 0 0 0  radius 0.1  material w }",
                          'camera "c" { eye 0 0 3  look_at 0 0 0  up 0 1 0  fov_y 40  %s}' % cam])
              + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        verdict(name, rc != 0 and phrase in log, "" if rc != 0 and phrase in log else log.strip()[-300:])

    # --- a moving coat cannot be summarised by the still density fields: -fur-volume steps aside ---
    hair = os.path.join(out, "hair.ftsl")
    write(hair, NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "h" { type hair  sigma_a 0.02  beta_m 0.35  beta_n 0.35  alpha 3.0 }',
        LIGHT,
        "group { motion { key { translate 0 0 0 } key { translate 0.3 0 0 } }  curve { material h  "
        "basis linear  radius 0.003  point 0 0.4 0  point 0.05 0.8 0 } }",
        'camera "c" { eye 0 0.6 2  look_at 0 0.6 0  up 0 1 0  fov_y 30  shutter_open -0.5  '
        'shutter_close 0.5  film { res 32 32 } }']) + NL)
    rc, log = run([exe, "-in", hair, "-mode", "R", "-device", "cpu", "-spp", "1", "-fur-volume", "-no-meter",
                   "-o", os.path.join(out, "hair.png")])
    verdict("-fur-volume steps aside for a moving coat, saying so",
            rc == 0 and "the coat moves" in log, "" if rc == 0 and "the coat moves" in log else log.strip()[-300:])

    # --- -checkbvh across the motion interval ---
    p = os.path.join(out, "bvh.ftsl")
    write(p, bvh_scene(cube_obj))
    rc, log = run([exe, "-in", p, "-checkbvh"])
    verdict("-checkbvh on moving primitives", rc == 0 and "PASS" in log,
            [l for l in log.splitlines() if "checkbvh" in l][-1:][0] if "checkbvh" in log else log[-300:])

    moving, static0 = os.path.join(out, "moving.ftsl"), os.path.join(out, "static_instant.ftsl")
    blurred = os.path.join(out, "blurred.ftsl")
    write(moving, scene())
    write(static0, scene(u=0.5))
    write(blurred, scene(shutter=(-0.5, 0.5)))

    # --- no shutter: byte for byte the static scene at the instant, even with a mover whose
    # sweep reaches far past the stage, under an environment light sized by the scene's bounds ---
    devices = [("cpu", False)] + ([] if a.no_gpu else [("gpu", True)])
    moving_x, static_x = os.path.join(out, "moving_x.ftsl"), os.path.join(out, "static_x.ftsl")
    write(moving_x, scene(extras=True))
    write(static_x, scene(u=0.5, extras=True))
    for dev, gpu in devices:
        args = ["-mode", "R", "-device", dev, "-spp", "16"]
        _, raw_m, _ = render(exe, out, moving_x, "instant_moving_" + dev, args, want_gpu=gpu)
        _, raw_s, _ = render(exe, out, static_x, "instant_static_" + dev, args, want_gpu=gpu)
        verdict("no shutter = static scene at the instant (mode R %s)" % dev.upper(), raw_m == raw_s)

    # --- blurred mode R = the average of static instants over the shutter ---
    args = ["-mode", "R", "-device", "cpu", "-spp", str(SPP)]
    bl = [render(exe, out, blurred, "blur_R_s%d" % s, args, seed=s)[0] for s in (1, 2)]
    inst = [np.zeros((H, W, 3)), np.zeros((H, W, 3))]
    for j in range(TIMES):
        u = (j + 0.5) / TIMES
        p = os.path.join(out, "static_t.ftsl")
        write(p, scene(u=u))
        for s in (1, 2):
            img = render(exe, out, p, "static_t_s%d" % s,
                         ["-mode", "R", "-device", "cpu", "-spp", str(SPP_T)], seed=100 * j + s)[0]
            inst[s - 1] += img / TIMES
    lb, li = [lum(x) for x in bl], [lum(x) for x in inst]
    mb, mi = blocks((lb[0] + lb[1]) / 2), blocks((li[0] + li[1]) / 2)
    err = np.hypot(block_err(lb[0], lb[1]), block_err(li[0], li[1]))
    z = np.abs(mb - mi) / np.maximum(err, 1e-30)
    rel = np.abs(mb - mi) / np.maximum(np.abs(mi), 1e-3 * np.abs(mi).max())
    bad = (z > 5.0) & (rel > 0.03)
    verdict("blurred mode R = mean of %d instants (8x8 blocks)" % TIMES, not bad.any(),
            "worst block z %.1f rel %.3f; %d of %d blocks out" % (z.max(), rel.max(), bad.sum(), bad.size))
    total = abs(mb.mean() / mi.mean() - 1.0)
    verdict("blurred mode R, whole-frame mean", total < 0.01, "%.4f off" % total)
    still = blocks(lum(render(exe, out, static0, "instant_ref", args)[0]))
    moved = (np.abs(mb - still) / np.maximum(still, 1e-3 * still.max()) > 0.10).sum()
    verdict("the blur is there (blocks off the instant by > 10%)", moved >= 8, "%d blocks" % moved)

    # --- the other modes that blur agree with mode R: BDPT (a time per path pair) and the
    # forward light tracer (a time per photon, its camera splat at the same time) ---
    for mode, margs in (("D", ["-spp", "256"]), ("B", ["-n", "40000000"])):
        bm = [lum(render(exe, out, blurred, "blur_%s_s%d" % (mode, s),
                         ["-mode", mode, "-device", "cpu"] + margs, seed=s)[0]) for s in (1, 2)]
        mm = blocks((bm[0] + bm[1]) / 2)
        err = np.hypot(block_err(lb[0], lb[1]), block_err(bm[0], bm[1]))
        z = np.abs(mm - mb) / np.maximum(err, 1e-30)
        rel = np.abs(mm - mb) / np.maximum(np.abs(mb), 1e-3 * np.abs(mb).max())
        bad = (z > 5.0) & (rel > 0.05)
        verdict("blurred mode %s = blurred mode R (8x8 blocks)" % mode, not bad.any(),
                "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

    # --- the modes that share light data across paths blur too, each epoch (M) or pass (S, U) at
    # one time: against blurred mode R, within their own bias (a density estimate's radius), and
    # the blur really there ---
    # (Mode M under a -noise budget, which has no total to split, ends an epoch every few samples,
    # so its shutter is sampled by many epochs too.)
    # On the GPU too: S and U a pass, and J's camera pass an epoch, at one time.
    shared_cases = [("M", "M", ["-spp", "128", "-n", "2000000"], "cpu"),
                    ("M under -noise", "M", ["-noise", "6.25", "-n", "2000000"], "cpu"),
                    ("S", "S", ["-spp", "128", "-n", "200000"], "cpu"),
                    ("U", "U", ["-spp", "256"], "cpu"), ("J", "J", ["-spp", "128", "-jsurf"], "cpu"),
                    ("P", "P", ["-spp", "256", "-n", "20000000"], "cpu")]
    if not a.no_gpu:
        shared_cases += [("M on the GPU", "M", ["-spp", "128", "-n", "2000000"], "gpu"),
                         ("M under -noise on the GPU", "M", ["-noise", "6.25", "-n", "2000000"], "gpu"),
                         ("S on the GPU", "S", ["-spp", "128", "-n", "200000"], "gpu"),
                         ("U on the GPU", "U", ["-spp", "256"], "gpu"),
                         ("J on the GPU", "J", ["-spp", "128", "-jsurf"], "gpu"),
                         ("P on the GPU", "P", ["-spp", "256", "-n", "20000000"], "gpu")]
    for label, mode, margs, dev in shared_cases:
        tag = label.replace(" ", "_").replace("-", "")
        gpu = dev == "gpu"
        bm = [lum(render(exe, out, blurred, "blur_%s_s%d" % (tag, s),
                         ["-mode", mode, "-device", dev] + margs, seed=s, want_gpu=gpu)[0]) for s in (1, 2)]
        mm = blocks((bm[0] + bm[1]) / 2)
        err = np.hypot(block_err(lb[0], lb[1]), block_err(bm[0], bm[1]))
        z = np.abs(mm - mb) / np.maximum(err, 1e-30)
        rel = np.abs(mm - mb) / np.maximum(np.abs(mb), 1e-3 * np.abs(mb).max())
        bad = (z > 5.0) & (rel > 0.10)
        inst = blocks(lum(render(exe, out, static0, "instant_%s" % tag, ["-mode", mode, "-device", dev] + margs,
                                 want_gpu=gpu)[0]))
        moved = (np.abs(mm - inst) / np.maximum(inst, 1e-3 * inst.max()) > 0.10).sum()
        verdict("blurred mode %s = blurred mode R, and blurs" % label, not bad.any() and moved >= 8,
                "worst block z %.1f rel %.3f; %d blocks out; %d blocks off the instant" %
                (z.max(), rel.max(), bad.sum(), moved))

    # --- several mode B cameras in one run: two with the shutter share a flight and both blur,
    # one without it (declared first) renders apart and stays sharp. On any device. ---
    text = open(blurred, encoding="utf-8").read()
    cam_c = [l for l in text.splitlines() if l.startswith('camera "c"')][0]
    cam_d = cam_c.replace('camera "c"', 'camera "d"')
    cam_a = re.sub(r"shutter_open \S+  shutter_close \S+  ", "", cam_c.replace('camera "c"', 'camera "a"'))
    multi = os.path.join(out, "blurred_multi.ftsl")
    write(multi, text.replace(cam_c, NL.join([cam_a, cam_c, cam_d])))
    imgs = {}
    for s in (1, 2):
        png = os.path.join(out, "multi_s%d.png" % s)
        for c in "acd":
            for ext in (".png", ".pfm"):
                f = png[:-4] + "_" + c + ext
                if os.path.exists(f):
                    os.remove(f)
        rc, log = run([exe, "-in", multi, "-seed", str(s), "-hdr", "-no-meter", "-o", png,
                       "-mode", "B", "-n", "40000000"] + (["-device", "cpu"] if a.no_gpu else []))
        if rc != 0:
            print(log[-2000:])
            raise SystemExit("render failed: " + multi)
        for c in "acd":
            imgs.setdefault(c, []).append(lum(read_pfm(png[:-4] + "_%s.pfm" % c)))
    for c in "cd":
        mm = blocks((imgs[c][0] + imgs[c][1]) / 2)
        err = np.hypot(block_err(lb[0], lb[1]), block_err(imgs[c][0], imgs[c][1]))
        z = np.abs(mm - mb) / np.maximum(err, 1e-30)
        rel = np.abs(mm - mb) / np.maximum(np.abs(mb), 1e-3 * np.abs(mb).max())
        bad = (z > 5.0) & (rel > 0.05)
        verdict("... shared mode B flight, camera %s = blurred mode R" % c, not bad.any(),
                "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))
    sharp_a = blocks((imgs["a"][0] + imgs["a"][1]) / 2)
    off = (np.abs(sharp_a - still) / np.maximum(still, 1e-3 * still.max()) > 0.10).sum()
    verdict("... and the camera without a shutter stays sharp", off < 4, "%d blocks off the instant" % off)

    # --- the sky a forward mode sees directly blurs behind a mover as mode R's does ---
    sky = os.path.join(out, "sky.ftsl")
    write(sky, sky_scene((-0.5, 0.5)))
    sr = [lum(render(exe, out, sky, "sky_R_s%d" % s, ["-mode", "R", "-device", "cpu", "-spp", "512"],
                     seed=s)[0]) for s in (1, 2)]
    sbm = [lum(render(exe, out, sky, "sky_B_s%d" % s, ["-mode", "B", "-device", "cpu", "-n", "40000000"],
                      seed=s)[0]) for s in (1, 2)]
    mr, mbs = blocks((sr[0] + sr[1]) / 2), blocks((sbm[0] + sbm[1]) / 2)
    err = np.hypot(block_err(sr[0], sr[1]), block_err(sbm[0], sbm[1]))
    z = np.abs(mbs - mr) / np.maximum(err, 1e-30)
    rel = np.abs(mbs - mr) / np.maximum(np.abs(mr), 1e-3 * np.abs(mr).max())
    bad = (z > 5.0) & (rel > 0.05)
    verdict("blurred mode B's sky behind a mover = mode R's", not bad.any(),
            "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

    # --- a mover whose sweep leaves the instant's bounding sphere is still lit by the sky in the
    # forward modes, and still shadows a sunlit floor in BDPT: their light enters through a sphere
    # that holds the sweep ---
    # (Under the sun mode R needs more samples: the ends of the sweep, nearly empty blocks, are
    # reached only by times near the ends of the shutter.)
    sun = "light sun { dir 1 0.3 0  angle 0.53  spd preset:d65 }"   # low, from +x
    for name, light, rspp, mode, margs in (
            ("the sky lights a mover outside the instant's bounds (mode B)",
             "light env { spd preset:d65 }", 512, "B", ["-n", "40000000"]),
            ("... the sun does (mode B)", sun, 2048, "B", ["-n", "40000000"]),
            ("... and its shadow falls in the sun (mode D)", sun, 2048, "D", ["-spp", "512"])):
        pr = os.path.join(out, "reach.ftsl")
        write(pr, reach_scene(light))
        rr = [lum(render(exe, out, pr, "reach_R_s%d" % s, ["-mode", "R", "-device", "cpu", "-spp",
                                                           str(rspp)], seed=s)[0]) for s in (1, 2)]
        rm = [lum(render(exe, out, pr, "reach_%s_s%d" % (mode, s), ["-mode", mode, "-device", "cpu"] + margs,
                         seed=s)[0]) for s in (1, 2)]
        good, detail = agree(rm, rr, 0.05)
        verdict(name, good, detail)

    # --- a shared flight resumes only a sidecar rendered with the same shutters ---
    pc = os.path.join(out, "resume_multi.ftsl")
    for shutter, phrase in (((-0.5, 0.5), None), ((-0.5, 0.5), True), ((-0.25, 0.25), False)):
        write(pc, scene(shutter=shutter).replace('camera "c"', 'camera "e" { eye 0.3 1.0 4.2  look_at 0 0.5 0  '
                                                 'up 0 1 0  fov_y 40  shutter_open %r  shutter_close %r  '
                                                 'film { res %d %d } }' % (shutter + (W, H)) + NL + 'camera "c"'))
        rargs = [exe, "-in", pc, "-mode", "B", "-device", "cpu", "-n", "300000", "-checkpoint", "-no-meter",
                 "-o", os.path.join(out, "resume_multi.png")]
        rc, log = run(rargs + (["-resume"] if phrase is not None else []))
        if phrase is None:
            continue
        loaded = "loaded shared model-B group" in log
        verdict("a shared flight resumes its own shutter's sidecar" if phrase else
                "... and not one rendered with another shutter", rc == 0 and loaded == phrase,
                "" if rc == 0 and loaded == phrase else log.strip()[-300:])

    # --- the same blur authored in centimetres: the keys move metres, as the vertices are ---
    blurred_cm = os.path.join(out, "blurred_cm.ftsl")
    write(blurred_cm, scene(shutter=(-0.5, 0.5), units="centimeters"))
    bc = [lum(render(exe, out, blurred_cm, "blur_cm_s%d" % s, args, seed=s)[0]) for s in (1, 2)]
    mc = blocks((bc[0] + bc[1]) / 2)
    err = np.hypot(block_err(lb[0], lb[1]), block_err(bc[0], bc[1]))
    z = np.abs(mc - mb) / np.maximum(err, 1e-30)
    rel = np.abs(mc - mb) / np.maximum(np.abs(mb), 1e-3 * np.abs(mb).max())
    bad = (z > 5.0) & (rel > 0.03)
    verdict("the blur authored in centimetres = in metres", not bad.any(),
            "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

    # --- an instantaneous shutter away from the instant: the scene exactly as written there ---
    for t in (-0.35, 0.1, 0.4):
        sharp, still_t = os.path.join(out, "sharp.ftsl"), os.path.join(out, "sharp_static.ftsl")
        write(sharp, scene(shutter=(t, t), twirl=True))
        write(still_t, scene(u=t + 0.5, twirl=True))
        sa = [lum(render(exe, out, sharp, "sharp_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                             "-spp", "256"], seed=s)[0]) for s in (1, 2)]
        sb = [lum(render(exe, out, still_t, "sharp_static_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                                     "-spp", "256"], seed=s)[0])
              for s in (1, 2)]
        ma, ms = blocks((sa[0] + sa[1]) / 2), blocks((sb[0] + sb[1]) / 2)
        err = np.hypot(block_err(sa[0], sa[1]), block_err(sb[0], sb[1]))
        z = np.abs(ma - ms) / np.maximum(err, 1e-30)
        rel = np.abs(ma - ms) / np.maximum(np.abs(ms), 1e-3 * np.abs(ms).max())
        bad = (z > 5.0) & (rel > 0.03)
        verdict("shutter %g..%g = the scene as written at %g" % (t, t, t), not bad.any(),
                "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

    # --- a pattern travels with its moving object: held sharp at u = 0.9 the ball has slid 0.48
    # m, and its stripes with it ---
    sharp, ref = os.path.join(out, "stripe_sharp.ftsl"), os.path.join(out, "stripe_ref.ftsl")
    write(sharp, stripe_scene(shutter=(0.4, 0.4)))
    write(ref, stripe_scene(u=0.9, shift=-0.6 + 1.2 * 0.9 - 0.0))
    pa = [lum(render(exe, out, sharp, "stripe_sharp_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                               "-spp", "256"], seed=s)[0]) for s in (1, 2)]
    pb = [lum(render(exe, out, ref, "stripe_ref_s%d" % s, ["-mode", "R", "-device", "cpu",
                                                           "-spp", "256"], seed=s)[0]) for s in (1, 2)]
    ma, mr = blocks((pa[0] + pa[1]) / 2, 4), blocks((pb[0] + pb[1]) / 2, 4)
    err = np.hypot(block_err(pa[0], pa[1], 4), block_err(pb[0], pb[1], 4))
    z = np.abs(ma - mr) / np.maximum(err, 1e-30)
    rel = np.abs(ma - mr) / np.maximum(np.abs(mr), 1e-3 * np.abs(mr).max())
    bad = (z > 5.0) & (rel > 0.03)
    verdict("a pattern travels with its moving object", not bad.any(),
            "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum()))

    # --- mode W: blurred, deterministically ---
    wargs = ["-mode", "W", "-device", "cpu", "-spp", "64"]
    _, w1, _ = render(exe, out, blurred, "blur_W_a", wargs)
    _, w2, _ = render(exe, out, blurred, "blur_W_b", wargs)
    _, w0, _ = render(exe, out, moving, "instant_W", wargs)
    verdict("mode W blurs, the same on every run", w1 == w2 and w1 != w0)

    # --- blurred mode R on the GPU = on the CPU ---
    if not a.no_gpu:
        gpu_cases = [("the box", blurred, 256)]
        pk = os.path.join(out, "gpu_kinds.ftsl")
        write(pk, kinds_scene(cube_obj, shutter=(-0.5, 0.5)))
        gpu_cases.append(("an isosurface, a curve and an instance", pk, 256))
        for nm, light in (("sky", "light env { spd preset:d65 }"),
                          ("sun", "light sun { dir 1 0.3 0  angle 0.53  spd preset:d65 }")):
            pr = os.path.join(out, "gpu_reach_%s.ftsl" % nm)
            write(pr, reach_scene(light))
            gpu_cases.append(("a mover past the instant's bounds, in the %s" % nm, pr, 512))
        ps = os.path.join(out, "gpu_stripe.ftsl")
        write(ps, stripe_scene(shutter=(-0.5, 0.5)))
        gpu_cases.append(("a pattern travelling with its object", ps, 256))
        pl = os.path.join(out, "gpu_lights.ftsl")
        write(pl, lights_scene(obj.replace(chr(92), "/"), shutter=(-0.5, 0.5)))
        gpu_cases.append(("moving lights of every shape", pl, 512))
        for nm, path, spp in gpu_cases:
            rargs = lambda dev: ["-mode", "R", "-device", dev, "-spp", str(spp)]
            g = [lum(render(exe, out, path, "gpu_R_%d" % s, rargs("gpu"), seed=s, want_gpu=True)[0]) for s in (1, 2)]
            c = [lum(render(exe, out, path, "cpu_R_%d" % s, rargs("cpu"), seed=s)[0]) for s in (1, 2)]
            good, detail = agree(g, c)
            verdict("blurred mode R on the GPU = on the CPU: " + nm, good, detail)
        # Mode D on the GPU: each sample's two subpaths at one time, against mode D on
        # the CPU (BDPT with a moving light, a sun reaching a mover past the instant's bounds).
        for nm, path, spp in [c_ for c_ in gpu_cases if "sky" not in c_[0]]:
            dargs = lambda dev: ["-mode", "D", "-device", dev, "-spp", str(spp)]
            g = [lum(render(exe, out, path, "gpu_D_%d" % s, dargs("gpu"), seed=s, want_gpu=True)[0]) for s in (1, 2)]
            c = [lum(render(exe, out, path, "cpu_D_%d" % s, dargs("cpu"), seed=s)[0]) for s in (1, 2)]
            good, detail = agree(g, c)
            verdict("blurred mode D on the GPU = on the CPU: " + nm, good, detail)
        # Modes A, B and C on the GPU: each photon and its splats at one time. Mode B
        # against mode B on the CPU over every GPU case (the hero megakernel), and the box on the
        # wavefront backend; mode A on the box and the moving lights; mode C, through an
        # aperture wide enough to catch photons, against mode A on the GPU.
        for nm, path, _spp in gpu_cases:
            bargs = lambda dev: ["-mode", "B", "-device", dev, "-n", "40000000"]
            g = [lum(render(exe, out, path, "gpu_B_%d" % s, bargs("gpu"), seed=s, want_gpu=True)[0]) for s in (1, 2)]
            c = [lum(render(exe, out, path, "cpu_B_%d" % s, bargs("cpu"), seed=s)[0]) for s in (1, 2)]
            good, detail = agree(g, c)
            verdict("blurred mode B on the GPU = on the CPU: " + nm, good, detail)
            if nm == "the box":
                cpu_box_b = c
        pm = os.path.join(out, "gpu_specular.ftsl")
        write(pm, specular_scene())
        bargs = lambda dev: ["-mode", "B", "-device", dev, "-n", "40000000"]
        g = [lum(render(exe, out, pm, "gpu_Bs_%d" % s, bargs("gpu"), seed=s, want_gpu=True)[0]) for s in (1, 2)]
        c = [lum(render(exe, out, pm, "cpu_Bs_%d" % s, bargs("cpu"), seed=s)[0]) for s in (1, 2)]
        good, detail = agree(g, c)
        verdict("blurred mode B on the GPU = on the CPU: moving mirrors and glass", good, detail)
        # The wavefront serves the scalar path only (the hero bundle runs the megakernel), so
        # -heroc 1, or the case would never leave the megakernel.
        g = [lum(render(exe, out, blurred, "gpu_Bwf_%d" % s, ["-mode", "B", "-device", "gpu", "-n", "40000000",
                                                              "-wavefront", "-heroc", "1"], seed=s,
                        want_gpu=True)[0]) for s in (1, 2)]
        good, detail = agree(g, cpu_box_b)
        verdict("... and on the wavefront backend: the box", good, detail)
        for nm, path, _spp in [c_ for c_ in gpu_cases if c_[0] in ("the box", "moving lights of every shape")]:
            aargs = lambda dev: ["-mode", "A", "-device", dev, "-n", "40000000"]
            g = [lum(render(exe, out, path, "gpu_A_%d" % s, aargs("gpu"), seed=s, want_gpu=True)[0]) for s in (1, 2)]
            c = [lum(render(exe, out, path, "cpu_A_%d" % s, aargs("cpu"), seed=s)[0]) for s in (1, 2)]
            good, detail = agree(g, c)
            verdict("blurred mode A on the GPU = on the CPU: " + nm, good, detail)
        wide = os.path.join(out, "gpu_wide.ftsl")
        with open(blurred, encoding="utf-8") as f:
            txt = f.read()
        write(wide, txt.replace("film {", "aperture 0.25  focus 4.4  film {"))
        g = [lum(render(exe, out, wide, "gpu_C_%d" % s, ["-mode", "C", "-device", "gpu", "-n", "2000000000"],
                        seed=s, want_gpu=True)[0]) for s in (1, 2)]
        c = [lum(render(exe, out, wide, "gpu_Aw_%d" % s, ["-mode", "A", "-device", "gpu", "-n", "40000000"],
                        seed=s, want_gpu=True)[0]) for s in (1, 2)]
        good, detail = agree(g, c, 0.05)
        verdict("blurred mode C on the GPU = mode A on the GPU (a wide aperture)", good, detail)
        # Lights moving, and a glowing mesh deforming, through an open shutter on the GPU against
        # blurred mode R on the CPU: in modes B and D a time per photon or path pair, in the modes
        # that take one a pass (U, S) or an epoch (J, M) each pass's light paths, deposits and
        # gathers at its own time.
        pl, pg = os.path.join(out, "gpu_lone.ftsl"), os.path.join(out, "gpu_glow.ftsl")
        write(pl, lone_lights_scene(shutter=(-0.5, 0.5)))
        write(pg, glow_deform_scene(gridf, shutter=(-0.5, 0.5)))
        for nm, path in (("three lights alone", pl), ("a deforming glowing mesh", pg)):
            rargs = ["-mode", "R", "-device", "cpu", "-spp", "512"]
            c = [lum(render(exe, out, path, "cpu_Rl_%d" % s, rargs, seed=s)[0]) for s in (1, 2)]
            for mode, margs, tol in (("B", ["-n", "40000000"], 0.05), ("D", ["-spp", "256"], 0.05),
                                     ("U", ["-spp", "256"], 0.10), ("S", ["-spp", "128", "-n", "200000"], 0.15),
                                     ("J", ["-spp", "256", "-jsurf", "-beamepochs", "64"], 0.10),
                                     ("M", ["-spp", "128", "-n", "8000000"], 0.10)):
                g = [lum(render(exe, out, path, "gpu_%sl_%d" % (mode, s), ["-mode", mode, "-device", "gpu"] + margs,
                                seed=s, want_gpu=True)[0]) for s in (1, 2)]
                good, detail = agree(g, c, tol)
                verdict("blurred mode %s on the GPU = mode R on the CPU: %s" % (mode, nm), good, detail)
        rc, log = run([exe, "-in", multi, "-mode", "B", "-device", "gpu", "-n", "100000",
                       "-no-meter", "-o", os.path.join(out, "gate.png")])
        verdict("-device gpu flies a blurred shared flight on the GPU",
                rc == 0 and "shared model-B pass" in log and "CPU threads" not in log and "using CPU" not in log,
                log.strip()[-300:] if rc else "")
    rc, log = run([exe, "-in", blurred, "-mode", "V", "-device", "cpu", "-spp", "1", "-n", "100000", "-no-meter",
                   "-o", os.path.join(out, "modev.png")])
    verdict("mode V under a shutter says it renders the instant",
            rc == 0 and "renders the frame's instant" in log, log.strip()[-300:] if rc else "")

    light_and_haze_tests(exe, out, verdict, hair, not a.no_gpu)
    p_split_tests(exe, out, verdict, not a.no_gpu)
    # Mode P under -radcache on the CPU, whose cache advances at each backward batch's end: a
    # fixed budget's batches follow its samples, not the clock (some 60 of them here, enough for the
    # cache to cut paths short), so one seed renders one image.
    p = os.path.join(out, "radcache_box.ftsl")
    write(p, scene(u=0.5))
    raws = [render(exe, out, p, "radcache_%d" % k, ["-mode", "P", "-device", "cpu", "-radcache", "-spp", "3000",
                                                     "-n", "600000000"])[1] for k in (1, 2)]
    verdict("mode P under -radcache on the CPU: one seed, one image", raws[0] == raws[1])
    j_empty_test(exe, out, verdict)
    camera_tests(exe, out, verdict, devices)
    lens_tests(exe, out, verdict, devices)
    if not a.no_gpu:
        meter_m_test(exe, out, verdict)
        m_epoch_tests(exe, out, verdict)
        split_tests(exe, out, verdict)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def split_tests(exe, out, verdict):
    """Renders the device splits into calls or chunks, the still box: mode P with the scene held
    on the device across its batches renders what it renders uploading it for each
    (FTRACE_NOHOLD=1), and GPU mode M's photon map, deposited in chunks the clock sizes, is one map
    per seed (each photon seeded from its absolute index)."""
    p = os.path.join(out, "split_box.ftsl")
    write(p, scene(u=0.5))
    args = ["-mode", "P", "-device", "gpu", "-spp", "64", "-n", "8000000"]
    held = render(exe, out, p, "split_held", args, want_gpu=True)[0]
    unheld = render_env(exe, out, p, "split_unheld", args, {"FTRACE_NOHOLD": "1"})
    d = np.abs(held - unheld).max() / max(np.abs(unheld).max(), 1e-300)
    verdict("mode P on the GPU, the scene held across batches = uploaded each", d < 1e-4,
            "largest change %.3g of the brightest pixel" % d)
    # the deposit cut into chunks of 1 M photons and of 3 M (FTRACE_DEPOSIT_CHUNK), not the clock's
    args = ["-mode", "M", "-device", "gpu", "-spp", "16", "-n", "12000000"]
    a, b = (render_env(exe, out, p, "split_dep_%d" % k, args, {"FTRACE_DEPOSIT_CHUNK": str(k * 1000000)})
            for k in (1, 3))
    d = np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)
    verdict("GPU mode M: one seed, one photon map, however its deposit is cut", d < 1e-4,
            "largest change %.3g of the brightest pixel" % d)


def render_env(exe, out, path, tag, args, env):
    """render() on the GPU under extra environment variables, each restored after (or removed, if
    it was not set)."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return render(exe, out, path, tag, args, want_gpu=True)[0]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def meter_m_test(exe, out, verdict):
    """A moving scene's mode-M group on the GPU: an exposure-locked camera path with a shutter over a
    white ball that slides the frame's width. Each of its frames gathers on the GPU alone, its maps
    redrawn at times of its own shutter (as the CPU renders each apart); and its batched GPU exposure
    meter meters what the CPU's meter does, a map of the frame's instant with the camera posed per
    sample. A gather that moved the scene to each sample's time in that map of the instant would
    read the ball dark and anchor the frames several times too bright."""
    path = os.path.join(out, "meter_m.ftsl")
    write(path, NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "white" { type diffuse reflect 0.9 }',
        'material "dark"  { type diffuse reflect 0.1 }',
        "quad { origin -3 0 -3  u 0 0 6  v 6 0 0  material dark }",
        "quad { origin -3 0 -2  u 6 0 0  v 0 4 0  material dark }",
        # no `power`: the frames are auto-exposed, so the meter's anchor sets their exposure
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500 }",
        "group { motion { key { translate -0.9 0.5 0.3 } key { translate 0.9 0.5 0.3 } }  "
        "sphere { center 0 0 0  radius 0.45  material white } }",
        'camera_path "p" { look_at 0 0.5 0  up 0 1 0  fov_y 40  frames 2  shutter_open -0.5  '
        "shutter_close 0.5  film { res %d %d }  exposure_lock average" % (W, H),
        "key 0 0 1.0 4.2",
        "key 1 0.1 1.0 4.2",
        "}"]) + NL)
    # A meter's anchor (a 99th percentile of 16 samples a pixel) varies some 7 % from seed to seed:
    # the mean of three on each device, against a 15 % tolerance.
    anchors, logs = {"gpu": [], "cpu": []}, {}
    for dev in ("gpu", "cpu"):
        for s in (1, 2, 3):
            cmd = [exe, "-in", path, "-mode", "M", "-device", dev, "-spp", "2", "-n", "100000",
                   "-seed", str(s), "-o", os.path.join(out, "meter_m_%s.png" % dev)]
            rc, log = run(cmd)
            found = re.findall(r"\[meter\] exposure lock: group 0 meters [^\n]*\(anchor ([0-9.eE+-]+)\)", log)
            if rc != 0 or not found:
                print(log[-2000:])
                raise SystemExit("metering failed (exit %d): %s" % (rc, " ".join(cmd)))
            if dev == "gpu" and "metering on" not in log:
                raise SystemExit("the meter did not run on the GPU: " + " ".join(cmd))
            anchors[dev].append(float(found[-1]))
            logs[dev] = log
    maps = logs["gpu"].count("shared photon map (mode M) on")
    verdict("-device gpu gathers each frame of a moving scene's mode-M path on the GPU, alone",
            maps == 2 and "using CPU" not in logs["gpu"], "%d device gathers" % maps)
    r = np.mean(anchors["gpu"]) / np.mean(anchors["cpu"])
    verdict("... and its batched GPU meter meters as the CPU's does", abs(r - 1) < 0.15,
            "anchor %.4f of the CPU's" % r)


def m_epoch_tests(exe, out, verdict):
    """The device's blurred mode-M epochs of the moving box: a 64th of -spp each, so -spp 64 is 64
    epochs, which write the image on the -interval cadence and at the end rather than after each
    (one write); and under a clock -beamfreeze, which the epochs set aside, leaves the render on the
    device."""
    path = os.path.join(out, "m_epochs.ftsl")
    write(path, scene(shutter=(-0.5, 0.5)))
    rc, log = run([exe, "-in", path, "-mode", "M", "-device", "gpu", "-spp", "64", "-n", "200000",
                   "-interval", "100000", "-no-meter", "-o", os.path.join(out, "m_epochs.png")])
    writes = len(re.findall(r"^wrote ", log, re.M))
    verdict("blurred mode M on the GPU writes on the -interval cadence, not every epoch",
            rc == 0 and "-> GPU" in log and writes == 1, "%d writes" % writes)
    rc, log = run([exe, "-in", path, "-mode", "M", "-device", "gpu", "-time", "4", "-beamfreeze", "-n", "200000",
                   "-no-meter", "-o", os.path.join(out, "m_epochs_t.png")])
    verdict("... and under a clock with -beamfreeze it stays on the GPU", rc == 0 and "-> GPU" in log,
            log.strip()[-300:] if rc else "")


def p_split_tests(exe, out, verdict, gpu=False):
    """Blurred mode P against blurred mode R where its split must file each sample by its path's
    first event: fog in front of a mirror (a sample scattered in the fog is the forward layer's,
    which splats the fog itself, so the backward layer must not count it too), and a floor of a
    mix of a diffuse and a glossy child (the glossy child is the forward layer's, the mix being
    no specular type, so neither layer may skip it). And fur of opacity below 1 in front of a
    mirror that shows the light, under -dual-scatter, whose hair case takes the hero path's
    dispersive group: a coverage pass-through there is no first event, or the mirror light seen
    through the gaps would be filed to the forward layer, which does not carry it."""
    for name, extra, floor in (
            ("fog in front of a mirror", "medium { sigma_t 0.15  albedo 0.8 }", "white"),
            ("a mix of a diffuse and a glossy child", "", "mx")):
        text = NL.join([
            "scene { units meters  motion_interval -0.5 0.5 }",
            'material "white" { type diffuse reflect 0.7 }',
            'material "gold"  { type glossy reflect rgb 0.9 0.7 0.3  roughness 0.25 }',
            'material "mx" { type mix  layer "white" 0.5  layer "gold" 0.5 }',
            'material "mir" { type mirror }',
            "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material %s }" % floor,
            "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material mir }",
            "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }",
            "group { %s  sphere { center 0 0 0  radius 0.3  material white } }" % xf(SLIDE, None, 1.0),
            extra,
            'camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 40  shutter_open -0.5  '
            'shutter_close 0.5  film { res %d %d } }' % (W, H)]) + NL
        ps = os.path.join(out, "psplit.ftsl")
        write(ps, text)
        rr = [lum(render(exe, out, ps, "psplit_R_s%d" % s, ["-mode", "R", "-device", "cpu", "-spp", "512"],
                         seed=s)[0]) for s in (1, 2)]
        rp = [lum(render(exe, out, ps, "psplit_P_s%d" % s, ["-mode", "P", "-device", "cpu", "-spp", "256",
                                                             "-n", "20000000"], seed=s)[0]) for s in (1, 2)]
        good, detail = agree(rp, rr, 0.10)
        verdict("blurred mode P = blurred mode R: " + name, good, detail)
        if gpu:
            pg = ["-mode", "P", "-device", "gpu", "-spp", "256", "-n", "20000000"]
            rg = [lum(render(exe, out, ps, "psplit_Pg_s%d" % s, pg, seed=s, want_gpu=True)[0]) for s in (1, 2)]
            good, detail = agree(rg, rr, 0.10)
            verdict("... and on the GPU", good, detail)
            if floor == "mx":
                # The device draws its streams by photon and by sample, whatever the batches,
                # so the same seed gives the same image however the clock sized them.
                again = lum(render(exe, out, ps, "psplit_Pg_again", pg, seed=1, want_gpu=True)[0])
                d = float(np.abs(again - rg[0]).max() / max(float(np.abs(rg[0]).max()), 1e-300))
                verdict("... and one seed renders one image on the GPU", d < 1e-9,
                        "largest difference %.3g of the peak" % d)
    strands = [("curve { material h  basis linear  radius 0.004  point %.3f 0.25 0.6  point %.3f 1.15 0.6 }"
                % (x, x)) for x in np.arange(-0.4, 0.401, 0.02)]
    text = NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "w" { type diffuse reflect 0.6 }',
        'material "mir" { type mirror }',
        'material "h" { type hair  sigma_a 0.6  beta_m 0.3  beta_n 0.3  alpha 2.0  opacity 0.3 }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material w }",
        "quad { origin -1 0.1 -0.5  u 2 0 0  v 0 1.4 0  material mir }",
        "light area { origin -0.3 1.2 2.4  u 0.6 0 0  v 0 0.4 0  normal 0 0 -1  spd preset:bb6500  power 80 }",
        "group { motion { key { translate 1.2 0.2 0 } key { translate 1.4 0.2 0 } }  "
        "sphere { center 0 0 0  radius 0.1  material w } }",
    ] + strands + [
        'camera "c" { eye 0 0.75 2.2  look_at 0 0.7 0  up 0 1 0  fov_y 30  shutter_open -0.5  '
        'shutter_close 0.5  film { res 64 48 } }']) + NL
    pf = os.path.join(out, "pfur.ftsl")
    write(pf, text)
    pargs = ["-mode", "P", "-device", "cpu", "-spp", "256", "-n", "4000000"]
    ref = [lum(render(exe, out, pf, "pfur_s%d" % s, pargs, seed=s)[0]) for s in (1, 2)]
    dual = [lum(render(exe, out, pf, "pfur_dual_s%d" % s, pargs + ["-dual-scatter"], seed=s)[0]) for s in (1, 2)]
    fur = (slice(10, 40), slice(16, 48))                          # over the fur, the mirror behind
    r = float(np.mean([x[fur].mean() for x in dual]) / np.mean([x[fur].mean() for x in ref]))
    verdict("blurred mode P, fur before a mirror, with -dual-scatter = without", abs(r - 1) < 0.1,
            "%.4f of without" % r)
    p_sun_test(exe, out, verdict, gpu)


def p_sun_test(exe, out, verdict, gpu):
    """Blurred mode P with a sun and no sky, its disc in view: an escape into the sun is the camera
    side's, which must carry it, the blurred forward layer drawing no sky: filed to the
    diffuse-first film, which only fits the scale, the disc would be missing and the scale fit to
    it."""
    text = NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "w" { type diffuse reflect 0.6 }',
        "light sun { dir 0 0.15 -1  angle 4  spd preset:d65 }",
        "quad { origin -6 0 -12  u 0 0 16  v 12 0 0  material w }",
        "group { %s  sphere { center 0 0 0  radius 0.3  material w } }" % xf(SLIDE, None, 1.0),
        'camera "c" { eye 0 0.6 3  look_at 0 0.7 -10  up 0 1 0  fov_y 40  shutter_open -0.5  '
        'shutter_close 0.5  film { res %d %d } }' % (W, H)]) + NL
    psun = os.path.join(out, "psun.ftsl")
    write(psun, text)
    rr = [lum(render(exe, out, psun, "psun_R_s%d" % s, ["-mode", "R", "-device", "cpu", "-spp", "256"],
                     seed=s)[0]) for s in (1, 2)]
    for dev in ("cpu", "gpu") if gpu else ("cpu",):
        rp = [lum(render(exe, out, psun, "psun_P_%s_s%d" % (dev, s),
                         ["-mode", "P", "-device", dev, "-spp", "128", "-n", "4000000"], seed=s,
                         want_gpu=dev == "gpu")[0]) for s in (1, 2)]
        good, detail = agree(rp, rr, 0.10)
        verdict("blurred mode P sees a sun with no sky (%s)" % dev.upper(), good, detail)


def j_empty_test(exe, out, verdict):
    """Blurred mode J whose surface map comes back empty -- every surface a mirror, so no light
    subpath stores a photon -- is mode D exactly, and blurs as mode D does, each sample at its own
    time (not the first epoch's alone)."""
    text = NL.join([
        "scene { units meters  motion_interval -0.5 0.5 }",
        'material "mir" { type mirror }',
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material mir }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material mir }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }",
        "group { %s  sphere { center 0 0 0  radius 0.3  material mir } }" % xf(SLIDE, None, 1.0),
        'camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 40  shutter_open -0.5  '
        'shutter_close 0.5  film { res %d %d } }' % (W, H)]) + NL
    pj = os.path.join(out, "j_empty.ftsl")
    write(pj, text)
    rr = [lum(render(exe, out, pj, "j_empty_R_s%d" % s, ["-mode", "R", "-device", "cpu", "-spp", "256"],
                     seed=s)[0]) for s in (1, 2)]
    rj = [lum(render(exe, out, pj, "j_empty_J_s%d" % s, ["-mode", "J", "-device", "cpu", "-spp", "128", "-jsurf"],
                     seed=s)[0]) for s in (1, 2)]
    good, detail = agree(rj, rr, 0.10)
    verdict("blurred mode J with an empty surface map = blurred mode R", good, detail)


def light_and_haze_tests(exe, out, verdict, hair, gpu):
    """A moving light's turn, a scene whose keys move nothing, a coat seen only at the instant,
    haze under a wide sweep (on the GPU too, when `gpu`), and the forward sky in a small batch."""
    # A spot on its group's pivot turning 30 degrees across the shutter sweeps its beam along a
    # wall: every time lights the wall alike, so the streak is as bright mid-way as near its ends.
    def spot_scene(spot_at, shutter=(-0.5, 0.5)):
        return NL.join([
            "scene { units meters  motion_interval -0.5 0.5 }",
            'material "w" { type diffuse reflect 0.7 }',
            "quad { origin 2 -1 -2  u 0 3 0  v 0 0 4  material w }",
            "group { motion { key { translate 0 1 0  rotate 0 -15 0 } key { translate 0 1 0  rotate 0 15 0 } }  "
            "light spot { origin %r 0 0  dir 1 0 0  inner_angle 3  outer_angle 5  spd preset:bb6500  power 20 } }"
            % spot_at,
            'camera "c" { eye -1 1 0  look_at 2 1 0  up 0 1 0  fov_y 50  shutter_open %r  shutter_close %r  '
            'film { res %d %d } }' % (shutter + (W, H))]) + NL
    p = os.path.join(out, "spot_turn.ftsl")
    write(p, spot_scene(0.0))
    img = lum(render(exe, out, p, "spot_turn", ["-mode", "R", "-device", "cpu", "-spp", "256"])[0])
    row = img[H // 2 - 2:H // 2 + 3].mean(axis=0)              # across the streak, along z
    lit = row > 0.2 * row.max()
    span = np.nonzero(lit)[0]
    if len(span) > 4:
        inner = row[span[0] + (span[-1] - span[0]) // 4: span[-1] - (span[-1] - span[0]) // 4 + 1]
        good = inner.min() > 0.6 * inner.max()
        detail = "the streak's middle half dips to %.2f of its peak" % (inner.min() / inner.max())
    else:
        good, detail = False, "no streak"
    verdict("a spot turning on its pivot lights its whole sweep", good, detail)
    p = os.path.join(out, "spot_off.ftsl")
    write(p, spot_scene(0.5))
    rc, log = run([exe, "-in", p, "-parseonly"])
    verdict("... and off its pivot needs no more keys than it has", rc == 0 and "keys to follow" not in log,
            "" if rc == 0 and "keys to follow" not in log else log.strip()[-300:])

    # Keys that move nothing (the only mover a spot they would stretch unevenly): no motion in the scene,
    # so mode V under a shutter has no blur to say it drops.
    p = os.path.join(out, "nomotion.ftsl")
    write(p, NL.join(["scene { units meters  motion_interval -0.5 0.5 }", 'material "w" { type diffuse reflect 0.5 }',
                      "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material w }",
                      STILL[1][1],
                      'camera "c" { eye 0 1 3  look_at 0 0.5 0  up 0 1 0  fov_y 40  shutter_open -0.5  '
                      'shutter_close 0.5  film { res 32 24 } }']) + NL)
    rc, log = run([exe, "-in", p, "-mode", "V", "-device", "cpu", "-spp", "1", "-n", "100000", "-no-meter",
                   "-o", os.path.join(out, "nomotion.png")])
    verdict("keys that move nothing are no motion", rc == 0 and "renders the frame's instant" not in log,
            log.strip()[-300:] if rc else "")

    # A moving coat seen only at the instant keeps -fur-volume.
    with open(hair, encoding="utf-8") as f:
        txt = f.read()
    p = os.path.join(out, "hair_instant.ftsl")
    write(p, txt.replace("shutter_open -0.5  shutter_close 0.5  ", ""))
    rc, log = run([exe, "-in", p, "-mode", "R", "-device", "cpu", "-spp", "1", "-fur-volume", "-no-meter",
                   "-o", os.path.join(out, "hair_instant.png")])
    verdict("-fur-volume keeps a moving coat seen at the instant", rc == 0 and "the coat moves" not in log,
            log.strip()[-300:] if rc or "the coat moves" in log else "")

    # Haze under a sweep far wider than the scene: the sun lights the floor through the scene's
    # haze alone, blurred or not -- in BDPT's sun connections and the forward photons alike, and
    # on the GPU in modes R and D (the sky too, in mode R).
    def haze_scene(shutter, light="light sun { dir 0.3 1 0.2  angle 0.53  spd preset:d65 }"):
        cam = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
        return NL.join([
            "scene { units meters  motion_interval -0.5 0.5 }",
            'material "w" { type diffuse reflect 0.6 }',
            light,
            "medium { sigma_t 0.04  albedo 0.5 }",
            "quad { origin -1 0 -1  u 0 0 2  v 2 0 0  material w }",
            "group { motion { key { translate -25 0.3 0 } key { translate 25 0.3 0 } }  "
            "sphere { center 0 0 0  radius 0.05  material w } }",
            'camera "c" { eye 0 1.6 2.2  look_at 0 0 0  up 0 1 0  fov_y 45  %sfilm { res %d %d } }'
            % (cam, W, H)]) + NL
    pb, ps = os.path.join(out, "haze_blur.ftsl"), os.path.join(out, "haze_still.ftsl")
    write(pb, haze_scene((-0.5, 0.5)))
    write(ps, haze_scene(None))
    sky_b, sky_s = os.path.join(out, "haze_sky_blur.ftsl"), os.path.join(out, "haze_sky_still.ftsl")
    write(sky_b, haze_scene((-0.5, 0.5), "light env { spd preset:d65 }"))
    write(sky_s, haze_scene(None, "light env { spd preset:d65 }"))
    cases = [("R", "cpu", pb, ps, ["-spp", "128"]), ("D", "cpu", pb, ps, ["-spp", "128"]),
             ("B", "cpu", pb, ps, ["-n", "20000000"])]
    if gpu:
        cases += [("R", "gpu", pb, ps, ["-spp", "128"]), ("R", "gpu", sky_b, sky_s, ["-spp", "128"]),
                  ("D", "gpu", pb, ps, ["-spp", "128"]), ("B", "gpu", pb, ps, ["-n", "20000000"]),
                  ("B", "gpu", sky_b, sky_s, ["-n", "20000000"])]
    for mode, dev, blur_p, still_p, margs in cases:
        args = ["-mode", mode, "-device", dev] + margs
        what = os.path.basename(blur_p)[:-10]
        hb = [lum(render(exe, out, blur_p, "%s_blur_%s_%s_s%d" % (what, mode, dev, s), args, seed=s,
                         want_gpu=dev == "gpu")[0]) for s in (1, 2)]
        hs = [lum(render(exe, out, still_p, "%s_still_%s_%s_s%d" % (what, mode, dev, s), args, seed=s,
                         want_gpu=dev == "gpu")[0]) for s in (1, 2)]
        floor = slice(H // 3, H)                                  # the floor, below the horizon of haze
        rb, rs = (hb[0] + hb[1])[floor].mean(), (hs[0] + hs[1])[floor].mean()
        verdict("a sweep past the scene leaves its haze as it is (mode %s%s%s)"
                % (mode, " on the GPU" if dev == "gpu" else "", ", the sky" if blur_p == sky_b else ""),
                abs(rb / rs - 1) < 0.03, "blurred floor %.4f of the still one's" % (rb / rs))

    # The blurred forward sky in a batch with under one sample a pixel to spare: each pixel's one
    # sample is taken with that chance and weighted up by it, so the sky is still all there.
    sky = os.path.join(out, "sky.ftsl")
    sr = [lum(render(exe, out, sky, "sky_R_s%d" % s, ["-mode", "R", "-device", "cpu", "-spp", "64"],
                     seed=s)[0]) for s in (1, 2)]
    sb = [lum(render(exe, out, sky, "sky_Bsmall_s%d" % s, ["-mode", "B", "-device", "cpu", "-n", "30000"],
                     seed=s)[0]) for s in range(1, 9)]
    top = slice(0, H // 6)                                        # sky alone
    r = np.mean([x[top].mean() for x in sb]) / np.mean([x[top].mean() for x in sr])
    verdict("a small batch's blurred sky is all there (mode B)", abs(r - 1) < 0.03, "%.4f of mode R's" % r)


def agree(a, b, rel_tol=0.03):
    """Whether two renders, each a pair of seeds' luminance, agree over 8x8 blocks (within 5
    standard errors or rel_tol), and the worst block."""
    ma, mb = blocks((a[0] + a[1]) / 2), blocks((b[0] + b[1]) / 2)
    err = np.hypot(block_err(a[0], a[1]), block_err(b[0], b[1]))
    z = np.abs(ma - mb) / np.maximum(err, 1e-30)
    rel = np.abs(ma - mb) / np.maximum(np.abs(mb), 1e-3 * np.abs(mb).max())
    bad = (z > 5.0) & (rel > rel_tol)
    return not bad.any(), "worst block z %.1f rel %.3f; %d blocks out" % (z.max(), rel.max(), bad.sum())


# A camera that moves over the motion interval: its two keys, at -0.5 and 0.5 frames.
CAM_KEYS = [dict(eye=(-0.4, 1.0, 4.2), look=(0.0, 0.5, 0.0), fov=38.0),
            dict(eye=(0.4, 1.0, 4.2), look=(0.2, 0.5, 0.0), fov=44.0)]


def cam_pose(u):
    """The keyed camera's pose at u in [0, 1] of the interval, blended as the loader blends."""
    a, b = CAM_KEYS
    lerp = lambda x, y: x + (y - x) * u
    return dict(eye=tuple(lerp(x, y) for x, y in zip(a["eye"], b["eye"])),
                look=tuple(lerp(x, y) for x, y in zip(a["look"], b["look"])), fov=lerp(a["fov"], b["fov"]))


def with_camera(text, cam):
    """`text` (a scene() file) with its camera "c" replaced by `cam`."""
    lines = [cam if l.startswith('camera "c"') else l for l in text.splitlines()]
    return NL.join(lines) + NL


def still_camera(p):
    return ('camera "c" { eye %s  look_at %s  up 0 1 0  fov_y %r  film { res %d %d } }'
            % (v3(p["eye"]), v3(p["look"]), p["fov"], W, H))


def keyed_camera(shutter=None, extra=""):
    sh = "shutter_open %r  shutter_close %r  " % shutter if shutter else ""
    keys = "  ".join("key { eye %s  look_at %s  fov_y %r }" % (v3(k["eye"]), v3(k["look"]), k["fov"])
                     for k in CAM_KEYS)
    return ('camera "c" { up 0 1 0  %s%smotion { %s }  film { res %d %d } }' % (sh, extra, keys, W, H))


def lens_tests(exe, out, verdict, devices):
    """The camera's thin lens across its shutter (MOTION-ZOOM-APERTURE), in mode A at f/1.4 on a
    60 x 45 mm film, so the box 4 m off is defocused by pixels: a zoom at a fixed f-number, whose
    aperture and film distance follow the focal length (about 5 pixels of defocus at 26 degrees
    and 1 at 54, focused at 0.9 m), and a rack of focus from 0.6 m to 6 m (3 pixels to sharp), each
    against the mean of its instants -- a still camera with that time's fov and focus, its lens
    derived by the loader. A lens held at the instant's aperture, film distance and focus would
    defocus the zoom's ends and the rack wrongly."""
    still = scene(u=0.5)
    cam = ('camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fstop 1.4  %s  '
           'film { res %d %d  size 60 45 } }')
    sh = "shutter_open -0.5  shutter_close 0.5"
    cases = [("zoom", "focus 0.9  %s  motion { key { fov_y 26 } key { fov_y 54 } }" % sh,
              lambda u: "fov_y %r  focus 0.9" % (26.0 + 28.0 * u)),
             ("rack", "fov_y 40  %s  motion { key { focus 0.6 } key { focus 6 } }" % sh,
              lambda u: "fov_y 40  focus %r" % (1.0 / (1.0 / 0.6 + (1.0 / 6.0 - 1.0 / 0.6) * u))),
             # from infinity (focus 0) in to 0.7 m: blended in diopters, the instants' focus 0.7 / u
             ("rack in", "fov_y 40  %s  motion { key { focus 0 } key { focus 0.7 } }" % sh,
              lambda u: "fov_y 40  focus %r" % (0.7 / u))]
    T, N = 16, 40000000
    for name, keyed, at in cases:
        pb = os.path.join(out, "lens_%s.ftsl" % name)
        write(pb, with_camera(still, cam % (keyed, W, H)))
        pt = os.path.join(out, "lens_%s_t.ftsl" % name)
        for dev, gpu in devices:
            args = ["-mode", "A", "-device", dev]
            blur = [lum(render(exe, out, pb, "lens_%s_%s_s%d" % (name, dev, s), args + ["-n", str(N)], seed=s,
                               want_gpu=gpu)[0]) for s in (1, 2)]
            mean = [np.zeros((H, W)), np.zeros((H, W))]
            for j in range(T):
                write(pt, with_camera(still, cam % (at((j + 0.5) / T), W, H)))
                for s in (1, 2):
                    mean[s - 1] += lum(render(exe, out, pt, "lens_%s_t_%s_s%d" % (name, dev, s),
                                              args + ["-n", str(N // T)], seed=100 * j + s,
                                              want_gpu=gpu)[0]) / T
            good, detail = agree(blur, mean, 0.05)
            verdict("a %s across the shutter blurs as the mean of its instants (mode A, %s)"
                    % ({"zoom": "zoom at f/1.4", "rack": "rack of focus",
                        "rack in": "rack of focus in from infinity"}[name], dev.upper()), good, detail)

    # --- no shutter: a camera whose keys rack its lens is the still camera at their instant ---
    pk, ps = os.path.join(out, "lens_keyed.ftsl"), os.path.join(out, "lens_still.ftsl")
    write(pk, with_camera(still, cam % ("fov_y 40  motion { key { focus 2 } key { focus 6 } }", W, H)))
    # their instant halfway between them in diopters, in the loader's own arithmetic (lerpLens)
    mid = 1.0 / (1.0 / 2.0 + (1.0 / 6.0 - 1.0 / 2.0) * 0.5)
    write(ps, with_camera(still, cam % ("fov_y 40  focus %r" % mid, W, H)))
    args = ["-mode", "A", "-device", "cpu", "-n", "2000000"]
    _, raw_k, _ = render(exe, out, pk, "lens_keyed", args)
    _, raw_s, _ = render(exe, out, ps, "lens_still", args)
    verdict("no shutter: a camera whose keys rack its focus = the still one at their instant", raw_k == raw_s)
    # keys that all agree on a lens: the camera's own is theirs
    write(pk, with_camera(still, cam % ("fov_y 40  focus 3  motion { key { focus 5 } key { focus 5 } }", W, H)))
    write(ps, with_camera(still, cam % ("fov_y 40  focus 5", W, H)))
    _, raw_k, _ = render(exe, out, pk, "lens_agree", args)
    _, raw_s, _ = render(exe, out, ps, "lens_agree_still", args)
    verdict("keys that agree on a focus give the camera theirs", raw_k == raw_s)
    # a camera_path's frame that zooms at an f-number has that focal length's lens, as a still camera
    pp = os.path.join(out, "lens_path.ftsl")
    write(pp, with_camera(still, NL.join([
        'camera_path "zp" { look_at 0 0.5 0  up 0 1 0  fov_y 40  fstop 1.4  focus 0.9  frames 3  '
        "film { res %d %d  size 60 45 }" % (W, H), "key 0 0 1.0 4.2 26", "key 1 0 1.0 4.2 54", "}"])))
    write(ps, with_camera(still, cam % ("fov_y 54  focus 0.9", W, H)))
    _, raw_k, _ = render(exe, out, pp, "lens_path", args + ["-camera", "zp2"])
    _, raw_s, _ = render(exe, out, ps, "lens_path_still", args)
    verdict("a camera path's zoomed frame has its focal length's lens", raw_k == raw_s)

    # --- a key may set only what the camera's lens lets it ---
    for what, text, says in (
            ("fstop on a camera without one", 'camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  '
             "motion { key { fstop 2 } key { fstop 4 } }  film { res %d %d } }" % (W, H), "`fstop` needs"),
            ("fstop on a lens camera with no f-number", 'camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  '
             "lens 50  motion { key { eye 0 1.0 4.2 } key { fstop 4 } }  film { res %d %d } }" % (W, H),
             "`fstop` needs"),
            ("focus on a camera with no lens", 'camera "c" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  '
             "motion { key { eye 0 1.0 4.2 } key { focus 2 } }  film { res %d %d } }" % (W, H),
             "`focus` cannot blend"),
            ("aperture on a camera an fstop sets", cam % ("motion { key { aperture 0.01 } key { aperture 0.02 } }",
                                                             W, H), "`aperture` needs")):
        p = os.path.join(out, "lens_refuse.ftsl")
        write(p, with_camera(still, text))
        rc, log = run([exe, "-in", p, "-mode", "A", "-device", "cpu", "-n", "1000", "-no-meter",
                       "-o", os.path.join(out, "lens_refuse.png")])
        verdict("refuses a key's %s" % what, rc != 0 and says in log, log.strip()[-200:] if rc == 0 else "")


def camera_tests(exe, out, verdict, devices):
    """The camera's own motion: keys on a still camera, and a camera path's frames."""
    still = scene(u=0.5)               # the box with its groups held still
    # --- no shutter: a keyed camera is the still camera at its keys' instant, byte for byte ---
    keyed, inst = os.path.join(out, "cam_keyed.ftsl"), os.path.join(out, "cam_instant.ftsl")
    write(keyed, with_camera(still, keyed_camera()))
    write(inst, with_camera(still, still_camera(cam_pose(0.5))))
    for dev, gpu in devices:
        args = ["-mode", "R", "-device", dev, "-spp", "16"]
        _, raw_k, _ = render(exe, out, keyed, "cam_keyed_" + dev, args, want_gpu=gpu)
        _, raw_i, _ = render(exe, out, inst, "cam_instant_" + dev, args, want_gpu=gpu)
        verdict("no shutter: a keyed camera = the still one at its instant (%s)" % dev.upper(),
                raw_k == raw_i)

    # --- blurred: the mean of the camera's poses through the shutter ---
    moving = os.path.join(out, "cam_moving.ftsl")
    write(moving, with_camera(still, keyed_camera((-0.5, 0.5))))
    args = ["-mode", "R", "-device", "cpu", "-spp", str(SPP)]
    blur = [lum(render(exe, out, moving, "cam_blur_R_s%d" % s, args, seed=s)[0]) for s in (1, 2)]
    mean = [np.zeros((H, W)), np.zeros((H, W))]
    p = os.path.join(out, "cam_t.ftsl")
    for j in range(TIMES):
        write(p, with_camera(still, still_camera(cam_pose((j + 0.5) / TIMES))))
        for s in (1, 2):
            mean[s - 1] += lum(render(exe, out, p, "cam_t_s%d" % s, ["-mode", "R", "-device", "cpu",
                                      "-spp", str(SPP_T)], seed=100 * j + s)[0]) / TIMES
    good, detail = agree(blur, mean)
    verdict("a moving camera blurs as the mean of its poses (mode R)", good, detail)
    for mode, margs, tol in (("D", ["-spp", "256"], 0.05), ("B", ["-n", "40000000"], 0.05),
                             ("M", ["-spp", "128", "-n", "2000000"], 0.10),
                             # SPPM's own density bias: 12.3 % in one block of this scene with the
                             # camera still (S against R, 128 passes), 10.5 % blurred
                             ("S", ["-spp", "128", "-n", "200000"], 0.15), ("U", ["-spp", "256"], 0.10),
                             ("J", ["-spp", "128", "-jsurf"], 0.10), ("P", ["-spp", "256", "-n", "20000000"], 0.05)):
        bm = [lum(render(exe, out, moving, "cam_blur_%s_s%d" % (mode, s),
                         ["-mode", mode, "-device", "cpu"] + margs, seed=s)[0]) for s in (1, 2)]
        good, detail = agree(bm, blur, tol)
        verdict("... mode %s = mode R" % mode, good, detail)
    if len(devices) > 1:
        g = [lum(render(exe, out, moving, "cam_blur_gpu_s%d" % s, ["-mode", "R", "-device", "gpu", "-spp", str(SPP)],
                        seed=s, want_gpu=True)[0]) for s in (1, 2)]
        good, detail = agree(g, blur)
        verdict("... mode R on the GPU = on the CPU", good, detail)
        g = [lum(render(exe, out, moving, "cam_blur_gpuD_s%d" % s, ["-mode", "D", "-device", "gpu", "-spp", "256"],
                        seed=s, want_gpu=True)[0]) for s in (1, 2)]
        good, detail = agree(g, blur, 0.05)
        verdict("... mode D on the GPU = mode R", good, detail)
        g = [lum(render(exe, out, moving, "cam_blur_gpuB_s%d" % s, ["-mode", "B", "-device", "gpu", "-n", "40000000"],
                        seed=s, want_gpu=True)[0]) for s in (1, 2)]
        good, detail = agree(g, blur, 0.05)
        verdict("... mode B on the GPU = mode R", good, detail)
        g = [lum(render(exe, out, moving, "cam_blur_gpuBwf_s%d" % s, ["-mode", "B", "-device", "gpu", "-n", "40000000",
                                                                    "-wavefront", "-heroc", "1"],
                        seed=s, want_gpu=True)[0]) for s in (1, 2)]
        good, detail = agree(g, blur, 0.05)
        verdict("... and on the wavefront backend", good, detail)
        for mode, margs, tol in (("M", ["-spp", "128", "-n", "2000000"], 0.10),
                                 ("S", ["-spp", "128", "-n", "200000"], 0.15), ("U", ["-spp", "256"], 0.10),
                                 ("J", ["-spp", "128", "-jsurf"], 0.10),
                                 ("P", ["-spp", "256", "-n", "20000000"], 0.05)):
            g = [lum(render(exe, out, moving, "cam_blur_gpu%s_s%d" % (mode, s),
                            ["-mode", mode, "-device", "gpu"] + margs, seed=s, want_gpu=True)[0]) for s in (1, 2)]
            good, detail = agree(g, blur, tol)
            verdict("... mode %s on the GPU = mode R" % mode, good, detail)

    # --- a camera path's frame blurs along the path: frames a frame apart on a straight line at
    # an even pace, so the middle frame's shutter sweeps its eye linearly half a frame each way ---
    path = ('camera_path "fly" { look_at 0 0.5 0  up 0 1 0  fov_y 40  frames 5  '
            'shutter_open -0.5  shutter_close 0.5  film { res %d %d }' % (W, H) + NL +
            "key 0 -0.8 1.0 4.2" + NL + "key 1 0.8 1.0 4.2" + NL + "}")
    fly = os.path.join(out, "cam_path.ftsl")
    write(fly, with_camera(still, path))
    fb = [lum(render(exe, out, fly, "cam_path_s%d" % s, ["-mode", "R", "-device", "cpu", "-spp",
                                                          str(SPP), "-camera", "fly2"], seed=s)[0])
          for s in (1, 2)]
    fm = [np.zeros((H, W)), np.zeros((H, W))]
    for j in range(TIMES):
        x = -0.2 + 0.4 * (j + 0.5) / TIMES
        write(p, with_camera(still, still_camera(dict(eye=(x, 1.0, 4.2), look=(0.0, 0.5, 0.0), fov=40.0))))
        for s in (1, 2):
            fm[s - 1] += lum(render(exe, out, p, "cam_t_s%d" % s, ["-mode", "R", "-device", "cpu",
                                    "-spp", str(SPP_T)], seed=100 * j + s)[0]) / TIMES
    good, detail = agree(fb, fm)
    verdict("a camera path's frame blurs along its path", good, detail)

    # --- a key that sets no fov_y keeps the camera's own: the same render as the key setting it ---
    keys_part = ('camera "c" { up 0 1 0  fov_y 40  shutter_open -0.5  shutter_close 0.5  motion { '
                 'key { eye %s  look_at %s  fov_y 30 }  key { eye %s  look_at %s%s } }  film { res %d %d } }')
    ka = (v3(CAM_KEYS[0]["eye"]), v3(CAM_KEYS[0]["look"]), v3(CAM_KEYS[1]["eye"]), v3(CAM_KEYS[1]["look"]))
    pa, pb = os.path.join(out, "cam_fov_a.ftsl"), os.path.join(out, "cam_fov_b.ftsl")
    write(pa, with_camera(still, keys_part % (ka + ("", W, H))))
    write(pb, with_camera(still, keys_part % (ka + ("  fov_y 40", W, H))))
    fargs = ["-mode", "R", "-device", "cpu", "-spp", "16"]
    _, raw_a, _ = render(exe, out, pa, "cam_fov_a", fargs)
    _, raw_b, _ = render(exe, out, pb, "cam_fov_b", fargs)
    verdict("a key without fov_y keeps the camera's own", raw_a == raw_b)

    # --- in a flight with a blurring camera, a keyed camera without a shutter stays sharp ---
    mixed = os.path.join(out, "cam_mixed.ftsl")
    cams = NL.join(['camera "s" { eye 0 1.0 4.2  look_at 0 0.5 0  up 0 1 0  fov_y 40  shutter_open -0.5  '
                    'shutter_close 0.5  film { res %d %d } }' % (W, H),
                    keyed_camera().replace('camera "c"', 'camera "k"'),
                    keyed_camera((-0.5, 0.5)).replace('camera "c"', 'camera "m"')])
    write(mixed, with_camera(still, cams))
    alone = os.path.join(out, "cam_alone.ftsl")
    write(alone, with_camera(still, still_camera(cam_pose(0.5))))
    km, ka2 = [], []
    for sd in (1, 2):
        png = os.path.join(out, "cam_mixed_s%d.png" % sd)
        for c in "skm":
            for ext in (".png", ".pfm"):
                f = png[:-4] + "_" + c + ext
                if os.path.exists(f):
                    os.remove(f)
        rc, log = run([exe, "-in", mixed, "-seed", str(sd), "-hdr", "-no-meter", "-o", png, "-mode", "B",
                       "-device", "cpu", "-n", "40000000"])
        if rc != 0:
            print(log[-2000:])
            raise SystemExit("render failed: " + mixed)
        km.append(lum(read_pfm(png[:-4] + "_k.pfm")))
        ka2.append(lum(render(exe, out, alone, "cam_alone_s%d" % sd, ["-mode", "B", "-device", "cpu", "-n",
                                                                      "40000000"], seed=sd)[0]))
    good, detail = agree(km, ka2, 0.05)
    verdict("... a keyed camera without a shutter stays sharp beside one that blurs", good, detail)

    # --- a camera moving through a still scene keeps -radcache, and keys that agree are no
    # motion: such a camera renders on the GPU ---
    rc, log = run([exe, "-in", moving, "-mode", "R", "-device", "cpu", "-spp", "4", "-radcache", "-no-meter",
                   "-o", os.path.join(out, "cam_radcache.png")])
    verdict("... and keeps -radcache in a still scene", rc == 0 and "[radcache] off" not in log,
            "" if rc == 0 and "[radcache] off" not in log else log.strip()[-300:])
    if len(devices) > 1:
        agree_keys = ('camera "c" { up 0 1 0  fov_y 40  shutter_open -0.5  shutter_close 0.5  motion { '
                      'key { eye 0 1 4.2  look_at 0 0.5 0 }  key { eye 0 1 4.2  look_at 0 0.5 0 } }  '
                      'film { res %d %d } }' % (W, H))
        pk = os.path.join(out, "cam_agree.ftsl")
        write(pk, with_camera(still, agree_keys))
        rc, log = run([exe, "-in", pk, "-mode", "R", "-device", "gpu", "-spp", "1", "-no-meter",
                       "-o", os.path.join(out, "cam_agree.png")])
        verdict("keys that agree are a still camera (it renders on the GPU)",
                rc == 0 and "-> GPU" in log, "" if rc == 0 and "-> GPU" in log else log.strip()[-300:])

    # --- what the loader refuses ---
    for name, sc, cam, phrase in (
            ("refuses camera keys without a motion_interval", "", keyed_camera(), "motion_interval"),
            ("refuses a camera with one key", "motion_interval -0.5 0.5",
             'camera "c" { eye 0 1 3  look_at 0 0 0  motion { key { eye 0 1 4 } } }', "two `key`"),
            ("refuses a key's fov_y on a camera its lens sets", "motion_interval -0.5 0.5",
             keyed_camera(extra="lens 50  "), "cannot zoom")):
        p = os.path.join(out, "refuse.ftsl")
        write(p, NL.join(["scene { units meters  %s }" % sc, 'material "w" { type diffuse reflect 0.5 }',
                          LIGHT, "sphere { center 0 0 0  radius 0.1  material w }", cam]) + NL)
        rc, log = run([exe, "-in", p, "-parseonly"])
        verdict(name, rc != 0 and phrase in log, "" if rc != 0 and phrase in log else log.strip()[-300:])


if __name__ == "__main__":
    sys.exit(main())
