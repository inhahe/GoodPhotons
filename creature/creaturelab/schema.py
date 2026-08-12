"""The creature vocabulary: which blocks and properties an `.ftcl` file may contain.

This is the only place the language's surface is defined. The parser is generic, so adding
a property here is enough to make it parse, be arity-checked, and produce a typo
suggestion when misspelled.

Coordinate convention: **z-up, x-forward, y-left** -- MuJoCo's convention and comparative
biomechanics' convention, chosen so the MJCF emitter never has to rotate anything. FTSL is
y-up, so the *renderer* emitter converts; the simulator side, which runs billions of steps,
does not.
"""
from __future__ import annotations

from ftcl.parser import VARIADIC, BlockSpec, Prop

# --------------------------------------------------------------------------- geometry
GEOM = BlockSpec(
    label="word", choices=("capsule", "sphere", "box", "cylinder", "ellipsoid"),
    doc="A collision/inertial primitive attached to a bone, in bone-local coordinates.",
    props={
        # capsule / cylinder are defined by their axis segment; sphere/box by a centre
        "from": Prop(3, doc="proximal end of a capsule/cylinder axis"),
        "to": Prop(3, doc="distal end of a capsule/cylinder axis"),
        "at": Prop(3, doc="centre, for sphere/box/ellipsoid"),
        "radius": Prop(1),
        "size": Prop(3, doc="half-extents for box, radii for ellipsoid"),
        "density": Prop(1, doc="kg/m^3; ignored if the bone sets an explicit mass"),
        "mass": Prop(1, doc="explicit mass for this geom, overriding density"),
        "rgba": Prop(4),
        "collide": Prop(1, kind="word", choices=("yes", "no"),
                        doc="no = inertia and rendering only, no contacts"),
    },
)

JOINT = BlockSpec(
    label="word", choices=("hinge", "slide", "ball", "free"),
    doc="A degree of freedom between this bone and its parent.",
    props={
        "name": Prop(1, kind="str", doc="defaults to <bone>_<n>"),
        "axis": Prop(3),
        "at": Prop(3, doc="joint centre in bone-local coords; default is the bone origin"),
        "range": Prop(2, doc="limits; write them with units, e.g. -80deg 40deg"),
        "damping": Prop(1),
        "stiffness": Prop(1),
        "springref": Prop(1, doc="rest angle of `stiffness`"),
        "armature": Prop(1, doc="reflected rotor inertia; a little is a cheap stabiliser"),
        "frictionloss": Prop(1),
        "ref": Prop(1, doc="value of this joint in the model's reference pose"),
        "gear": Prop(1, doc="torque scale of this joint's motor; 0 leaves it unactuated"),
    },
)

BONE = BlockSpec(
    label="name",
    doc="A rigid segment. Bones form a tree; the one without a parent is the root.",
    props={
        "parent": Prop(1, kind="str"),
        "origin": Prop(3, required=False, doc="offset from the parent bone's origin"),
        "euler": Prop(3, doc="orientation relative to the parent frame"),
        "mass": Prop(1, doc="total mass of this bone, distributed over its geoms"),
        "rgba": Prop(4),
    },
    blocks={"joint": JOINT, "geom": GEOM},
)

SKELETON = BlockSpec(
    label="name", label_required=False,
    doc="The articulated tree.",
    props={"root": Prop(1, kind="str", doc="optional; otherwise inferred")},
    blocks={"bone": BONE},
)

# ------------------------------------------------------------------------- morphology
PARAM = BlockSpec(
    label="word",
    doc="One named morphology parameter. THE central abstraction -- see design.md.",
    props={
        "default": Prop(1, required=True),
        "range": Prop(2, doc="training/randomisation bounds; also the artist's slider"),
        "doc": Prop(1, kind="str"),
        "randomize": Prop(1, kind="word", choices=("yes", "no"),
                          doc="include in domain randomisation (default yes if range)"),
    },
)

MORPH = BlockSpec(
    label="name", label_required=False,
    doc="The morphology parameter vector: rig geometry, RL conditioning, randomisation "
        "axis and artist control, all from one declaration.",
    blocks={"param": PARAM},
)

DERIVE = BlockSpec(
    label="name", label_required=False,
    doc="Author-named intermediate values, e.g. `shank_len = femur_len * 0.95`.",
    free_props=Prop(1),
)

# ------------------------------------------------------------------------------ world
WORLD = BlockSpec(
    label="name", label_required=False,
    doc="Simulation setup that travels with the rig so training is reproducible.",
    props={
        "gravity": Prop(3),
        "timestep": Prop(1),
        "ground": Prop(1, kind="word", choices=("yes", "no")),
        "integrator": Prop(1, kind="word",
                           choices=("euler", "rk4", "implicit", "implicitfast")),
        "spawn_height": Prop(1, doc="root z at reset; default auto from the lowest geom"),
    },
)

DEFAULTS = BlockSpec(
    label="name", label_required=False,
    doc="Values applied to every joint/geom that does not set them itself.",
    props={
        "joint_damping": Prop(1, doc="absolute N*m*s/rad; body-specific, prefer posture"),
        "joint_armature": Prop(1, doc="absolute kg*m^2; body-specific, prefer the ratio"),
        "joint_armature_ratio": Prop(
            1, doc="armature as a fraction of each joint's own measured inertia"),
        "geom_density": Prop(1),
        "geom_friction": Prop(3),
        "motor_gear": Prop(1, doc="default torque scale for joint motors"),
        "rgba": Prop(4),
    },
)

POSTURE = BlockSpec(
    label="name", label_required=False,
    doc="Passive tone: how hard the *unactuated* skeleton resists gravity.",
    props={
        # These are goals, not gains. The per-joint stiffness and damping are MEASURED for
        # the body this morph vector actually produced (creaturelab/tune.py), because a
        # gain written as a literal is only ever correct for one body: randomise femur_len
        # and a hand-tuned stifle spring is instantly the wrong spring. "Sag no more than
        # 5 degrees" stays the same statement about every body in the distribution.
        "sag": Prop(1, required=True,
                    doc="how far a joint may sag under the standing load; sizes stiffness"),
        "tone_floor": Prop(1, doc="minimum tone as a fraction of a joint's worst-case "
                                  "gravitational load -- this is what gives the lateral "
                                  "joints tone, since they carry nothing standing square"),
        "buckle_margin": Prop(1, doc="how far above the measured buckling stiffness a "
                                     "joint's spring must sit; 1.0 is neutrally stable, "
                                     "so it wants headroom"),
        "damping_ratio": Prop(1, doc="damping as a fraction of critical, per joint; "
                                     "<1 lets the limb overshoot, >1 is sluggish"),
        "max_stiffness": Prop(1, doc="hard cap, in N*m/rad, so a near-singular joint "
                                     "cannot ask for a spring the integrator cannot "
                                     "follow at this timestep"),
    },
)

SENSING = BlockSpec(
    label="name", label_required=False,
    doc="The proprioceptive interface: where the nervous system reads the body from, how "
        "late the signals arrive, and how noisy they are.",
    props={
        # `hub` is here rather than inferred because a *conduction* delay is a distance to
        # somewhere, and only the rig knows where the nervous system's integration point
        # is. Inferring it from bone names would be a guess that silently reverses the
        # fore/hind delay ordering on any rig that spells its head differently.
        "hub": Prop(1, kind="str",
                    doc="bone the afferent/efferent path length is measured to, i.e. where "
                        "the CNS sits; default = the root bone"),
        "conduction": Prop(1, doc="signal conduction velocity along the body, m/s "
                                  "(large myelinated fibre ~60); delay = distance / this"),
        "central_delay": Prop(1, doc="fixed central processing latency, s, on top of "
                                     "conduction -- the part that does not scale with size"),
        # Noise is declared as a FRACTION of each channel's own scale, not in radians or
        # newtons, for the same reason posture declares an angle: a receptor's error is
        # proportional to what it measures, so a fraction stays true across the morph range
        # while an absolute figure is right for exactly one body.
        "angle_noise": Prop(1, doc="joint-angle noise, fraction of that joint's own range"),
        "rate_noise": Prop(1, doc="joint-velocity noise, fraction of the body's rate scale"),
        "force_noise": Prop(1, doc="contact-force noise, fraction of body weight"),
        "vestibular_noise": Prop(1, doc="noise on the gravity direction and body rates"),
    },
)

CREATURE = BlockSpec(
    label="name",
    props={
        "doc": Prop(1, kind="str"),
        # Total mass as a NAMED quantity. Without it, mass is an accident of how much the
        # trunk capsules happen to overlap -- and "40 kg heavier" stops being a knob.
        "mass": Prop(1, doc="target total mass; geom densities are scaled to hit it"),
    },
    blocks={"morph": MORPH, "derive": DERIVE, "skeleton": SKELETON,
            "world": WORLD, "defaults": DEFAULTS, "posture": POSTURE,
            "sensing": SENSING},
    required_blocks=("skeleton",),
)

ROOT = BlockSpec(blocks={"creature": CREATURE})

__all__ = ["ROOT", "CREATURE", "SKELETON", "BONE", "JOINT", "GEOM", "MORPH", "PARAM",
           "DERIVE", "WORLD", "DEFAULTS", "POSTURE", "SENSING", "VARIADIC"]
