"""AST + morph vector -> numeric `Creature`.

This is where the parameterisation pays off: every geometric quantity in the rig is an
unevaluated expression, and `build()` evaluates all of them against one environment. Change
`femur_len` and *everything downstream* -- segment length, capsule endpoints, the mass that
follows from the volume, the joint the next bone hangs off -- moves consistently, because
they were all written in terms of it rather than being independently-typed literals.

`build()` is called once per training episode under domain randomisation, so it must stay
cheap: no parsing, no file I/O, just expression evaluation over a small tree.
"""
from __future__ import annotations

import random

from ftcl.errors import FtclError, did_you_mean
from ftcl.expr import Env, Expr
from ftcl.parser import Node, parse_file, parse_string

from .model import (Bone, Creature, Defaults, Geom, Joint, MorphParam, Posture,
                    Sensing, World)
from .schema import ROOT


# ------------------------------------------------------------------------------ helpers
def _f(node: Node, key: str, env: Env, default=None):
    if key not in node.props:
        return default
    vals = node.props[key]
    return vals[0].eval(env) if isinstance(vals[0], Expr) else vals[0]


def _vec(node: Node, key: str, env: Env, default=None):
    if key not in node.props:
        return default
    return tuple(v.eval(env) for v in node.props[key])


def _word(node: Node, key: str, default=None):
    return node.props[key][0] if key in node.props else default


def _str(node: Node, key: str, default=None):
    return node.props[key][0] if key in node.props else default


def _yes(node: Node, key: str, default: bool) -> bool:
    v = _word(node, key)
    return default if v is None else (v == "yes")


# --------------------------------------------------------------------------------- morph
def collect_params(cnode: Node, src: str | None) -> list[MorphParam]:
    """Read the `morph` block. Defaults may themselves be expressions over other params."""
    out: list[MorphParam] = []
    mnode = cnode.child("morph")
    if mnode is None:
        return out
    seen: set[str] = set()
    for p in mnode.all("param"):
        name = p.word
        if name in seen:
            raise FtclError(f"morph param '{name}' declared twice", p.loc, src)
        seen.add(name)
        rng = p.props.get("range")
        lo = hi = None
        # NOTE: the default is stored unevaluated on purpose -- it may reference another
        # param -- and is resolved by the Env below, which detects cycles.
        out.append(MorphParam(
            name=name,
            default=p.props["default"][0],          # Expr, resolved in build_env
            lo=rng[0] if rng else None,             # Expr or None
            hi=rng[1] if rng else None,
            doc=_str(p, "doc", "") or "",
            randomize=_yes(p, "randomize", rng is not None),
        ))
    return out


def build_env(cnode: Node, params: list[MorphParam], overrides: dict[str, float] | None,
              src: str | None) -> tuple[Env, dict[str, float], list[MorphParam]]:
    """Resolve the morph vector, then layer `derive` on top of it.

    Overrides are checked against the declared names so a typo in a randomisation dict is
    an error rather than a silently-ignored key that leaves the body at its default.
    """
    overrides = dict(overrides or {})
    names = {p.name for p in params}
    for k in overrides:
        if k not in names:
            raise FtclError(f"morph override '{k}' is not a declared param"
                            + did_you_mean(k, names))

    env = Env({}, src=src)
    for p in params:
        env.define(p.name, p.default)             # Expr; lazily resolved
    # Resolve declared bounds first: they may be expressions over other params, and an
    # override has to be clamped to them.
    resolved: list[MorphParam] = []
    for p in params:
        lo = p.lo.eval(env) if isinstance(p.lo, Expr) else p.lo
        hi = p.hi.eval(env) if isinstance(p.hi, Expr) else p.hi
        resolved.append(MorphParam(p.name, env.lookup(p.name), lo, hi, p.doc, p.randomize))

    morph: dict[str, float] = {}
    for p in resolved:
        v = overrides.get(p.name, p.default)
        morph[p.name] = p.clamp(float(v))

    env = Env(dict(morph), src=src)
    dnode = cnode.child("derive")
    if dnode is not None:
        for key in dnode.free_keys:
            if key in morph:
                raise FtclError(f"'{key}' is both a morph param and a derived value",
                                dnode.prop_locs[key], src)
            env.define(key, dnode.props[key][0])
    return env, morph, resolved


# ------------------------------------------------------------------------------ skeleton
def build_bone(bnode: Node, env: Env, defaults: Defaults, src: str | None) -> Bone:
    bone = Bone(
        name=bnode.name,
        parent=_str(bnode, "parent"),
        origin=_vec(bnode, "origin", env, (0.0, 0.0, 0.0)),
        euler=_vec(bnode, "euler", env, (0.0, 0.0, 0.0)),
        mass=_f(bnode, "mass", env),
        rgba=_vec(bnode, "rgba", env),
    )
    for n, jnode in enumerate(bnode.all("joint")):
        rng = _vec(jnode, "range", env)
        bone.joints.append(Joint(
            name=_str(jnode, "name") or (f"{bone.name}_{jnode.word}"
                                         if n == 0 else f"{bone.name}_{jnode.word}{n}"),
            kind=jnode.word,
            axis=_vec(jnode, "axis", env, (0.0, 1.0, 0.0)),
            at=_vec(jnode, "at", env),
            range=(rng[0], rng[1]) if rng else None,
            # Left as None when the author did not write one, rather than eagerly filled
            # from `defaults`. The emitter substitutes the default at emission time, so the
            # model is unchanged -- but None now means "nobody chose this", which is what
            # lets creaturelab.tune fill a measured value without ever overwriting a number
            # a human deliberately put in the rig.
            damping=_f(jnode, "damping", env),
            stiffness=_f(jnode, "stiffness", env),
            springref=_f(jnode, "springref", env),
            armature=_f(jnode, "armature", env),
            frictionloss=_f(jnode, "frictionloss", env),
            ref=_f(jnode, "ref", env),
            gear=_f(jnode, "gear", env),
        ))
    for gnode in bnode.all("geom"):
        g = Geom(
            kind=gnode.word,
            frm=_vec(gnode, "from", env),
            to=_vec(gnode, "to", env),
            at=_vec(gnode, "at", env, (0.0, 0.0, 0.0)),
            radius=_f(gnode, "radius", env, 0.02),
            size=_vec(gnode, "size", env),
            density=_f(gnode, "density", env),
            mass=_f(gnode, "mass", env),
            rgba=_vec(gnode, "rgba", env) or bone.rgba,
            collide=_yes(gnode, "collide", True),
        )
        if g.kind in ("capsule", "cylinder") and (g.frm is None or g.to is None):
            raise FtclError(f"{g.kind} geom in bone '{bone.name}' needs both "
                            f"'from' and 'to'", gnode.loc, src)
        if g.kind in ("box", "ellipsoid") and g.size is None:
            raise FtclError(f"{g.kind} geom in bone '{bone.name}' needs 'size'",
                            gnode.loc, src)
        bone.geoms.append(g)
    if not bone.geoms:
        raise FtclError(f"bone '{bone.name}' has no geom, so it has no mass or inertia "
                        f"-- give it one, or the simulator will produce a singular model",
                        bnode.loc, src)
    return bone


def validate_tree(creature: Creature, src: str | None, locs: dict[str, object]) -> None:
    """Check the bones really form a single rooted tree, and order them parents-first.

    MJCF nests bodies, so a child must be emitted inside its parent -- a cycle or a
    dangling parent has to be caught here, where we can name the file and line, rather
    than as an XML nesting failure or, worse, a silently disconnected limb.
    """
    names = {b.name for b in creature.bones}
    if len(names) != len(creature.bones):
        dupes = sorted({b.name for b in creature.bones
                        if [x.name for x in creature.bones].count(b.name) > 1})
        raise FtclError(f"duplicate bone name(s): {', '.join(dupes)}")

    roots = [b for b in creature.bones if not b.parent]
    for b in creature.bones:
        if b.parent and b.parent not in names:
            raise FtclError(f"bone '{b.name}' has unknown parent '{b.parent}'"
                            + did_you_mean(b.parent, names), locs.get(b.name), src)
    if not roots:
        raise FtclError("no root bone: every bone declares a parent, so the skeleton "
                        "contains a cycle")
    if len(roots) > 1:
        raise FtclError("more than one root bone (" +
                        ", ".join(b.name for b in roots) +
                        "); a creature must be a single connected tree")
    creature.root = roots[0].name

    # topological order, parents before children
    order: list[Bone] = []
    by_parent: dict[str | None, list[Bone]] = {}
    for b in creature.bones:
        by_parent.setdefault(b.parent, []).append(b)
    stack = [roots[0]]
    while stack:
        b = stack.pop(0)
        order.append(b)
        stack = by_parent.get(b.name, []) + stack
    if len(order) != len(creature.bones):
        orphans = sorted(names - {b.name for b in order})
        raise FtclError(f"bone(s) not reachable from root '{creature.root}' "
                        f"(cycle?): {', '.join(orphans)}")
    creature.bones = order


# ---------------------------------------------------------------------------------- API
def build_from_nodes(nodes: list[Node], src: str | None = None,
                     morph: dict[str, float] | None = None,
                     which: str | None = None) -> Creature:
    creatures = [n for n in nodes if n.kw == "creature"]
    if not creatures:
        raise FtclError("file contains no `creature` block")
    if which is not None:
        match = [n for n in creatures if n.name == which]
        if not match:
            raise FtclError(f"no creature named '{which}' "
                            f"(have: {', '.join(n.name for n in creatures)})")
        cnode = match[0]
    else:
        cnode = creatures[0]

    params = collect_params(cnode, src)
    env, morph_vec, resolved = build_env(cnode, params, morph, src)

    dnode = cnode.child("defaults")
    defaults = Defaults()
    if dnode is not None:
        defaults = Defaults(
            joint_damping=_f(dnode, "joint_damping", env, defaults.joint_damping),
            joint_armature=_f(dnode, "joint_armature", env, defaults.joint_armature),
            joint_armature_ratio=_f(dnode, "joint_armature_ratio", env,
                                    defaults.joint_armature_ratio),
            geom_density=_f(dnode, "geom_density", env, defaults.geom_density),
            geom_friction=_vec(dnode, "geom_friction", env, defaults.geom_friction),
            motor_gear=_f(dnode, "motor_gear", env, defaults.motor_gear),
            rgba=_vec(dnode, "rgba", env, defaults.rgba),
        )

    wnode = cnode.child("world")
    world = World()
    if wnode is not None:
        world = World(
            gravity=_vec(wnode, "gravity", env, world.gravity),
            timestep=_f(wnode, "timestep", env, world.timestep),
            ground=_yes(wnode, "ground", True),
            integrator=_word(wnode, "integrator", world.integrator),
            spawn_height=_f(wnode, "spawn_height", env),
        )

    pnode = cnode.child("posture")
    posture = None
    if pnode is not None:
        posture = Posture()
        posture = Posture(
            sag=_f(pnode, "sag", env, posture.sag),
            tone_floor=_f(pnode, "tone_floor", env, posture.tone_floor),
            buckle_margin=_f(pnode, "buckle_margin", env, posture.buckle_margin),
            damping_ratio=_f(pnode, "damping_ratio", env, posture.damping_ratio),
            max_stiffness=_f(pnode, "max_stiffness", env),
        )
        if posture.sag <= 0:
            raise FtclError("posture 'sag' must be a positive angle (it divides the "
                            "measured holding torque)", pnode.loc, src)

    xnode = cnode.child("sensing")
    sensing = Sensing()
    if xnode is not None:
        sensing = Sensing(
            hub=_str(xnode, "hub", sensing.hub),
            conduction=_f(xnode, "conduction", env, sensing.conduction),
            central_delay=_f(xnode, "central_delay", env, sensing.central_delay),
            angle_noise=_f(xnode, "angle_noise", env, sensing.angle_noise),
            rate_noise=_f(xnode, "rate_noise", env, sensing.rate_noise),
            force_noise=_f(xnode, "force_noise", env, sensing.force_noise),
            vestibular_noise=_f(xnode, "vestibular_noise", env, sensing.vestibular_noise),
        )
        if sensing.conduction <= 0:
            raise FtclError("sensing 'conduction' must be a positive velocity in m/s "
                            "(it divides the afferent path length)", xnode.loc, src)

    snode = cnode.child("skeleton")
    creature = Creature(name=cnode.name, params=resolved, morph=morph_vec,
                        world=world, defaults=defaults, posture=posture,
                        sensing=sensing, target_mass=_f(cnode, "mass", env),
                        doc=_str(cnode, "doc", "") or "")
    locs = {}
    for bnode in snode.all("bone"):
        creature.bones.append(build_bone(bnode, env, defaults, src))
        locs[bnode.name] = bnode.loc
    if not creature.bones:
        raise FtclError("skeleton has no bones", snode.loc, src)

    declared_root = _str(snode, "root")
    validate_tree(creature, src, locs)
    if sensing.hub is not None:
        names = {b.name for b in creature.bones}
        if sensing.hub not in names:
            raise FtclError(f"sensing hub '{sensing.hub}' is not a bone"
                            + did_you_mean(sensing.hub, names), xnode.loc, src)
    if declared_root and declared_root != creature.root:
        raise FtclError(f"skeleton declares root '{declared_root}' but the bone tree's "
                        f"root is '{creature.root}'", snode.loc, src)
    return creature


def load(path: str, morph: dict[str, float] | None = None,
         which: str | None = None) -> Creature:
    nodes, src = parse_file(path, ROOT)
    return build_from_nodes(nodes, src, morph, which)


def load_string(text: str, morph: dict[str, float] | None = None,
                which: str | None = None) -> Creature:
    nodes = parse_string(text, ROOT)
    return build_from_nodes(nodes, text, morph, which)


def sample_morph(params, rng: random.Random | None = None,
                 scale: float = 1.0) -> dict[str, float]:
    """Draw a morph vector for domain randomisation.

    `scale` shrinks the sampled range about each default, so a curriculum can widen the
    body distribution over training instead of starting at full spread -- policies that
    see the whole morph space from step one tend to learn a mushy average gait.
    """
    rng = rng or random.Random()
    out: dict[str, float] = {}
    for p in params:
        if not p.randomize or p.lo is None or p.hi is None:
            out[p.name] = p.default
            continue
        lo = p.default + (p.lo - p.default) * scale
        hi = p.default + (p.hi - p.default) * scale
        out[p.name] = rng.uniform(lo, hi)
    return out
