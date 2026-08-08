"""The built creature: a fully numeric model, with the morph vector already applied.

Everything here is plain floats. `build.py` turns the symbolic AST plus a morph vector
into one of these, and the emitters turn it into MJCF or FTSL. Keeping this layer free of
`Expr` is what lets a training loop rebuild a randomised body thousands of times without
dragging the parser along.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Vec3 = tuple[float, float, float]


@dataclass
class Geom:
    kind: str
    frm: Vec3 | None = None
    to: Vec3 | None = None
    at: Vec3 = (0.0, 0.0, 0.0)
    radius: float = 0.02
    size: Vec3 | None = None
    density: float | None = None
    mass: float | None = None
    rgba: tuple[float, float, float, float] | None = None
    collide: bool = True


@dataclass
class Joint:
    name: str
    kind: str = "hinge"
    axis: Vec3 = (0.0, 1.0, 0.0)
    at: Vec3 | None = None
    range: tuple[float, float] | None = None       # radians / metres
    damping: float | None = None
    stiffness: float | None = None
    springref: float | None = None
    armature: float | None = None
    frictionloss: float | None = None
    ref: float | None = None
    gear: float | None = None

    @property
    def actuated(self) -> bool:
        """A free joint is the floating base, not something a muscle can pull on."""
        return self.kind in ("hinge", "slide") and (self.gear is None or self.gear > 0)


@dataclass
class Bone:
    name: str
    parent: str | None = None
    origin: Vec3 = (0.0, 0.0, 0.0)
    euler: Vec3 = (0.0, 0.0, 0.0)
    mass: float | None = None
    rgba: tuple[float, float, float, float] | None = None
    joints: list[Joint] = field(default_factory=list)
    geoms: list[Geom] = field(default_factory=list)


@dataclass
class MorphParam:
    name: str
    default: float
    lo: float | None = None
    hi: float | None = None
    doc: str = ""
    randomize: bool = True

    def clamp(self, v: float) -> float:
        if self.lo is not None:
            v = max(self.lo, v)
        if self.hi is not None:
            v = min(self.hi, v)
        return v


@dataclass
class World:
    gravity: Vec3 = (0.0, 0.0, -9.81)
    timestep: float = 0.002
    ground: bool = True
    integrator: str = "implicitfast"
    spawn_height: float | None = None


@dataclass
class Posture:
    """Goals for passive tone. `creaturelab.tune` turns these into per-joint gains.

    The indirection is the whole point. Ligament and resting-muscle tone are what stop a
    real skeleton folding up, and their *job description* -- "hold the stance against
    gravity, but give a little" -- is body-independent even though the gains that achieve
    it are not. Declaring the job and measuring the gains keeps the statement true across
    the entire morph range; declaring the gains would make it true for exactly one dog.
    """
    sag: float = 0.0873                        # 5 deg
    tone_floor: float = 0.15
    buckle_margin: float = 1.6
    damping_ratio: float = 0.9
    max_stiffness: float | None = None


@dataclass
class Sensing:
    """What the policy is allowed to know about the body, and how late and how wrong.

    This exists because "observation space" is not a training detail -- it is the contract
    that decides whether a policy transfers to a different body at all. Two rules follow
    from that and are enforced by `creaturelab.sensing` rather than left to the trainer:

    * **Every channel is body-local and normalised by the body's own scale**, so the same
      number means the same thing on a Chihuahua and a Great Dane. A world position or an
      absolute velocity in m/s does not have that property, trains faster, and produces a
      policy that cannot be morphed -- which is P4's entire premise thrown away for a
      short-term convergence win.
    * **Signals arrive late.** Conduction is finite, so a real animal acts on a body state
      that is 10-40 ms old, more in a large one. Training on instantaneous truth yields
      reflexes no animal has, and the tell is a twitchy over-corrected gait -- the motor
      analogue of a fixed-pivot knee. The delay is *derived* from the morph vector
      (path length / conduction velocity), so scaling the creature up makes it move
      heavier for free instead of needing a second hand-tuned number.

    Noise is deliberately NOT scaled by body size: it is declared as a fraction of each
    channel's own scale, and a receptor's fractional error is already size-independent.
    Scaling it again would be scaling it twice.
    """
    hub: str | None = None          # bone the conduction distance is measured to
    conduction: float = 60.0        # m/s
    central_delay: float = 0.010    # s
    angle_noise: float = 0.004      # fraction of joint range
    rate_noise: float = 0.02        # fraction of the body's rate scale
    force_noise: float = 0.03       # fraction of body weight
    vestibular_noise: float = 0.01


@dataclass
class Defaults:
    # Both of these are *escape hatches*, not the normal path: an absolute number in
    # kg*m^2 or N*m*s/rad is body-specific, so it is wrong for every morph but one. Leave
    # them None and `tune` measures each joint instead. See `joint_armature_ratio`.
    joint_damping: float | None = None
    joint_armature: float | None = None
    # Armature as a dimensionless fraction of the joint's own measured inertia -- which is
    # what a gear ratio physically is, since reflected rotor inertia is n^2*I_rotor and a
    # matched drive keeps that a roughly fixed share of the load it drives. Scale-free, so
    # it survives morphing; an absolute number does not. The default 0.01 kg*m^2 spanned
    # 0.7% to 3790% of true joint inertia across this rig's *own* joints, unmorphed.
    joint_armature_ratio: float = 0.1
    geom_density: float = 1000.0
    geom_friction: Vec3 = (0.8, 0.005, 0.0001)
    motor_gear: float = 30.0
    rgba: tuple[float, float, float, float] = (0.65, 0.6, 0.55, 1.0)


@dataclass
class Creature:
    name: str
    bones: list[Bone] = field(default_factory=list)
    root: str = ""
    params: list[MorphParam] = field(default_factory=list)
    morph: dict[str, float] = field(default_factory=dict)   # the vector actually applied
    world: World = field(default_factory=World)
    defaults: Defaults = field(default_factory=Defaults)
    posture: Posture | None = None          # None = no passive tone; a bare skeleton
    sensing: Sensing = field(default_factory=Sensing)
    target_mass: float | None = None
    # Body pairs that may never collide. Derived, not authored -- see tune.auto_exclude.
    contact_excludes: list[tuple[str, str]] = field(default_factory=list)
    doc: str = ""

    # -- lookups --------------------------------------------------------------------------
    def bone(self, name: str) -> Bone:
        for b in self.bones:
            if b.name == name:
                return b
        raise KeyError(name)

    def children_of(self, name: str) -> list[Bone]:
        return [b for b in self.bones if b.parent == name]

    @property
    def actuated_joints(self) -> list[Joint]:
        return [j for b in self.bones for j in b.joints if j.actuated]

    def morph_vector(self) -> list[float]:
        """The RL conditioning vector, in a stable order (declaration order)."""
        return [self.morph[p.name] for p in self.params]

    def morph_vector_normalized(self) -> list[float]:
        """Same, mapped to [-1, 1] over each param's declared range.

        Policies should consume this rather than raw metres: an unnormalised conditioning
        input whose components differ by three orders of magnitude trains badly, and the
        declared range is exactly the information needed to fix that.
        """
        out = []
        for p in self.params:
            v = self.morph[p.name]
            if p.lo is None or p.hi is None or p.hi <= p.lo:
                out.append(0.0)
            else:
                out.append(2.0 * (v - p.lo) / (p.hi - p.lo) - 1.0)
        return out

    def total_mass_hint(self) -> float | None:
        """Sum of explicit bone masses, or None when masses come from density."""
        masses = [b.mass for b in self.bones if b.mass is not None]
        return sum(masses) if len(masses) == len(self.bones) else None
