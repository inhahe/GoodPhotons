"""`Creature` -> MJCF.

Deliberately a *thin* translation. Bones map almost one-to-one onto MuJoCo bodies, which
is the whole reason the grammar was shaped the way it was (design.md, "A skeleton is
FTSL's group tree with the collapse deferred"). Anything MJCF already expresses well --
joints, limits, spatial tendons, Hill muscles -- is emitted straight through rather than
reinvented. The only thing this layer adds is that every number here came from one named
morph parameter instead of being a literal.

Two things are computed rather than passed through, because getting them wrong produces a
model that *loads fine and then behaves subtly wrongly*:

* **Mass distribution.** When a bone declares a total mass, it is split across its geoms by
  volume. Setting the same mass on each geom instead would silently move the centre of
  mass, and a limb with the wrong inertia distribution learns a wrong gait that looks
  almost right -- the worst kind of bug to chase.
* **Spawn height** is left to `place_on_ground()`, which uses MuJoCo's own forward
  kinematics rather than a hand-rolled FK pass here. Re-deriving Euler conventions by hand
  is a classic place to introduce an off-by-a-rotation error.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .model import Bone, Creature, Geom


def _n(x: float) -> str:
    """Compact fixed formatting: MJCF is read by humans during debugging."""
    if x == 0:
        return "0"
    s = f"{x:.6g}"
    return s


def _v(vals) -> str:
    return " ".join(_n(float(x)) for x in vals)


def geom_volume(g: Geom) -> float:
    r = g.radius
    if g.kind == "sphere":
        return 4.0 / 3.0 * math.pi * r ** 3
    if g.kind == "capsule":
        L = math.dist(g.frm, g.to)
        return math.pi * r * r * L + 4.0 / 3.0 * math.pi * r ** 3
    if g.kind == "cylinder":
        return math.pi * r * r * math.dist(g.frm, g.to)
    if g.kind == "box":
        return 8.0 * g.size[0] * g.size[1] * g.size[2]
    if g.kind == "ellipsoid":
        return 4.0 / 3.0 * math.pi * g.size[0] * g.size[1] * g.size[2]
    raise ValueError(f"unknown geom kind {g.kind!r}")


def _geom_el(parent: ET.Element, g: Geom, creature: Creature,
             mass: float | None, idx: str) -> None:
    el = ET.SubElement(parent, "geom", {"name": idx, "type": g.kind})
    if g.kind in ("capsule", "cylinder"):
        el.set("fromto", _v(list(g.frm) + list(g.to)))
        el.set("size", _n(g.radius))
    elif g.kind == "sphere":
        el.set("pos", _v(g.at))
        el.set("size", _n(g.radius))
    else:                                    # box / ellipsoid
        el.set("pos", _v(g.at))
        el.set("size", _v(g.size))
    if mass is not None:
        el.set("mass", _n(mass))
    elif g.mass is not None:
        el.set("mass", _n(g.mass))
    else:
        el.set("density", _n(g.density if g.density is not None
                             else creature.defaults.geom_density))
    el.set("rgba", _v(g.rgba if g.rgba is not None else creature.defaults.rgba))
    el.set("friction", _v(creature.defaults.geom_friction))
    if not g.collide:
        # contype/conaffinity 0 = participates in inertia and rendering, never in contacts
        el.set("contype", "0")
        el.set("conaffinity", "0")


def _body_el(parent: ET.Element, bone: Bone, creature: Creature, is_root: bool) -> ET.Element:
    body = ET.SubElement(parent, "body", {"name": bone.name, "pos": _v(bone.origin)})
    if any(bone.euler):
        body.set("euler", _v(bone.euler))

    if is_root:
        ET.SubElement(body, "freejoint", {"name": "root"})

    for j in bone.joints:
        if j.kind == "free":
            ET.SubElement(body, "freejoint", {"name": j.name})
            continue
        attrs = {"name": j.name, "type": j.kind, "axis": _v(j.axis)}
        if j.at is not None:
            attrs["pos"] = _v(j.at)
        if j.range is not None:
            attrs["range"] = _v(j.range)
        # A joint that named no damping/armature gets the creature's default here rather
        # than at build time, so `None` stays readable as "unset" for the tuner.
        d = j.damping if j.damping is not None else creature.defaults.joint_damping
        a = j.armature if j.armature is not None else creature.defaults.joint_armature
        if d:
            attrs["damping"] = _n(d)
        if a:
            attrs["armature"] = _n(a)
        if j.stiffness is not None:
            attrs["stiffness"] = _n(j.stiffness)
        if j.springref is not None:
            attrs["springref"] = _n(j.springref)
        if j.frictionloss is not None:
            attrs["frictionloss"] = _n(j.frictionloss)
        if j.ref is not None:
            attrs["ref"] = _n(j.ref)
        ET.SubElement(body, "joint", attrs)

    # split a declared bone mass across geoms by volume (see module docstring)
    masses: list[float | None] = [None] * len(bone.geoms)
    if bone.mass is not None:
        vols = [geom_volume(g) for g in bone.geoms]
        total = sum(vols)
        if total <= 0:
            raise ValueError(f"bone '{bone.name}' has zero total geom volume")
        masses = [bone.mass * v / total for v in vols]

    for k, (g, m) in enumerate(zip(bone.geoms, masses)):
        _geom_el(body, g, creature, m, f"{bone.name}_g{k}" if len(bone.geoms) > 1
                 else bone.name)

    for child in creature.children_of(bone.name):
        _body_el(body, child, creature, False)
    return body


def natural_mass(creature: Creature) -> float:
    """Mass the model would have from densities/explicit masses alone."""
    total = 0.0
    for b in creature.bones:
        if b.mass is not None:
            total += b.mass
            continue
        for g in b.geoms:
            if g.mass is not None:
                total += g.mass
            else:
                d = g.density if g.density is not None else creature.defaults.geom_density
                total += geom_volume(g) * d
    return total


def apply_target_mass(creature: Creature) -> float:
    """Scale every density/mass uniformly so the model weighs `creature.target_mass`.

    Trunk capsules overlap heavily -- that is how you get a smooth body out of primitives
    -- so the volume they enclose double-counts and the "natural" mass comes out well
    above life. Rather than hand-tuning densities until the number looks right (which
    silently breaks the moment a morph param changes the overlap), the target mass is
    declared and the densities are scaled to meet it. The *distribution* still comes from
    geometry, which is what matters for limb inertia; only the absolute scale is pinned.

    Returns the scale factor applied (1.0 when no target is declared).
    """
    if creature.target_mass is None:
        return 1.0
    nat = natural_mass(creature)
    if nat <= 0:
        raise ValueError("model has zero natural mass")
    k = creature.target_mass / nat
    for b in creature.bones:
        if b.mass is not None:
            b.mass *= k
            continue
        for g in b.geoms:
            if g.mass is not None:
                g.mass *= k
            else:
                d = g.density if g.density is not None else creature.defaults.geom_density
                g.density = d * k
    return k


def to_mjcf(creature: Creature, include_ground: bool | None = None) -> str:
    apply_target_mass(creature)
    root = ET.Element("mujoco", {"model": creature.name})

    # angle="radian" because the lexer already normalised every `deg` suffix to radians;
    # autolimits so a joint with a range is limited without a separate flag.
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true",
                                     "inertiafromgeom": "auto"})
    ET.SubElement(root, "option", {
        "timestep": _n(creature.world.timestep),
        "gravity": _v(creature.world.gravity),
        "integrator": {"euler": "Euler", "rk4": "RK4", "implicit": "implicit",
                       "implicitfast": "implicitfast"}[creature.world.integrator],
    })

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker",
                                     "width": "512", "height": "512",
                                     "rgb1": ".18 .2 .22", "rgb2": ".24 .26 .28"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid",
                                      "texrepeat": "6 6", "reflectance": "0.1"})

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 0 4", "dir": "0 0 -1", "directional": "true"})
    ground = creature.world.ground if include_ground is None else include_ground
    if ground:
        ET.SubElement(world, "geom", {
            "name": "floor", "type": "plane", "size": "0 0 .05", "material": "grid",
            "friction": _v(creature.defaults.geom_friction)})

    _body_el(world, creature.bone(creature.root), creature, True)

    if creature.contact_excludes:
        contact = ET.SubElement(root, "contact")
        for a, b in creature.contact_excludes:
            ET.SubElement(contact, "exclude", {"body1": a, "body2": b})

    acts = [j for b in creature.bones for j in b.joints if j.actuated]
    if acts:
        actuator = ET.SubElement(root, "actuator")
        for j in acts:
            gear = j.gear if j.gear is not None else creature.defaults.motor_gear
            ET.SubElement(actuator, "motor", {
                "name": f"act_{j.name}", "joint": j.name,
                "gear": _n(gear), "ctrlrange": "-1 1"})

    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    body = "\n".join(ln for ln in pretty.splitlines()[1:] if ln.strip())
    header = (f"<!-- generated from {creature.name}.ftcl by creaturelab; do not edit.\n"
              f"     morph: " +
              ", ".join(f"{k}={v:.4g}" for k, v in creature.morph.items()) + " -->\n")
    return header + body + "\n"


# ------------------------------------------------------------------- MuJoCo-side helpers
def geom_z_extent(model, data, gid: int) -> tuple[float, float]:
    """Exact (min_z, max_z) of one geom in world space.

    `model.geom_rbound` is the *bounding sphere* radius, which for a limb capsule is
    half-length + radius -- several times the true vertical half-extent when the capsule
    stands upright. Using it to find the ground made the creature spawn ~3.6 cm too high,
    so the feet never touched, so it was in free fall -- and a body in free fall needs
    exactly zero internal torque, which is why the first static-load report read all
    zeros. The same overestimate inflated the reported withers height.

    So: a real support function per geom type. For each shape the vertical half-extent is
    the support of its shape along world -z, with `R` the geom's rotation matrix and
    `n = R[2, :]` the world z axis expressed in geom-local coordinates.
    """
    import mujoco
    import numpy as np

    t = model.geom_type[gid]
    pos = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
    n = R[2, :]                                  # world +z in the geom's local frame
    s = model.geom_size[gid]
    T = mujoco.mjtGeom

    if t == T.mjGEOM_SPHERE:
        h = s[0]
    elif t == T.mjGEOM_CAPSULE:
        # cylinder of half-length s[1] along local z, capped by spheres of radius s[0]
        h = s[1] * abs(n[2]) + s[0]
    elif t == T.mjGEOM_CYLINDER:
        # flat caps: the extreme point is on the rim, not the axis
        h = s[1] * abs(n[2]) + s[0] * math.sqrt(max(0.0, 1.0 - n[2] * n[2]))
    elif t == T.mjGEOM_BOX:
        h = float(np.abs(n[:3]) @ s[:3])
    elif t == T.mjGEOM_ELLIPSOID:
        h = float(np.linalg.norm(n[:3] * s[:3]))
    else:                                        # plane, mesh, hfield: no exact form here
        h = float(model.geom_rbound[gid])
    return float(pos[2] - h), float(pos[2] + h)


def place_on_ground(model, data, clearance: float = 0.01) -> float:
    """Drop the free-base creature so its lowest geom sits `clearance` above z=0.

    Uses MuJoCo's own kinematics rather than re-deriving Euler conventions here. Returns
    the z that was applied. A creature spawned intersecting the floor gets an enormous
    contact impulse on step one, which reads as "the physics is unstable" when in fact the
    only problem was the spawn pose.

    Pass `clearance=0` (or slightly negative) when you want the feet actually *loaded* --
    inverse dynamics at a pose with no active contact measures a free-fall, not a stance.
    """
    import mujoco
    import numpy as np

    mujoco.mj_forward(model, data)
    lowest = np.inf
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == 0:          # skip world geoms (the floor)
            continue
        lowest = min(lowest, geom_z_extent(model, data, gid)[0])
    if not np.isfinite(lowest):
        return float(data.qpos[2])
    dz = clearance - lowest
    data.qpos[2] += dz
    mujoco.mj_forward(model, data)
    return float(data.qpos[2])
