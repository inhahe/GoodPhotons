# Parametric records — design & build plan

A new FTSL data structure: a **named record over a scalar domain** whose channels
are named after real material slots, sampled by a per-hit driver scalar, with
nearest / linear / smooth interpolation, and composed into materials via ordered
last-write-wins `from` imports plus dot-style overrides.

Status: **design locked (2026-07-18), not yet implemented.** This file is the
authoritative spec until the feature lands; fold any change here first, then build.

---

## 1. Motivation

Materials today are static: one `material "steel" { … }` fixes every slot. To vary
a slot across a surface you reach for a `pattern` (a scalar field) bound to a single
property, or a `type mix` blend of two whole materials. There is no way to say
"as this scalar sweeps 0→1, walk this material through a *curated sequence* of slot
values" — e.g. a mirror whose tint and roughness ramp along a hand-picked palette.

A **parametric record** is that curve bank: a bundle of per-channel LUTs over a
shared 1-D domain. A driver scalar (any per-hit expression) indexes all channels at
once; each channel's sampled value fills the like-named material slot. It doubles as
the "1-D colormap / lookup table" idea (channels can be spectra) and as a general
named array whose dimensions are addressable by name.

Key philosophy the user set:
- **Name by destination.** Channels are named after the real slot they'll fill
  (`reflect`, `roughness`, `ior`, …). No alias table — the user names things after
  the actual keywords.
- **Everything lenient.** A channel name that matches no slot is not an error — it's
  just not auto-bound (still addressable by dot). A slot with no matching channel is
  left at default. Extra data is ignored, missing data is ignored.
- **Driver is a pure expression.** No `pattern:`/`texture:`/`vdb:` prefixes and no
  bare pattern-name references in the driver. The driver is an ordinary pattern-VM
  expression over intrinsics `x y z nx ny nz r u v f` plus functions (incl.
  `noise(x,y,z)`, POV funcs). If you want noise you write `noise(4*x,4*y,4*z)`
  explicitly — the coords are visible, never hidden.

---

## 2. Locked syntax

### 2.1 Declaration (top-level, same scope as `material` / `spectrum`)

```
funny_mirror = range 0-1 [
    reflect    spectrum:steel  spectrum:gold  spectrum:copper
    roughness  0.0  p:0.7 0.4  1.0
    mirror     spectrum:silver spectrum:chrome        # a non-slot channel: legal, just not auto-bound
    interp     smooth
]
```

- **Header:** `NAME = range LO-HI [ … ]`. `NAME` is a new top-level identifier;
  `range LO-HI` is the driver domain `[LO,HI]`. (Grammar note: reuse the existing
  `NAME = …` top-level statement path — the same one `spectrum NAME = …` uses — with
  a `range` keyword after `=` selecting the record form, and `[ … ]` as the body.)
- **Channels:** one per line, `channelname stop stop stop …`. Channel names are
  arbitrary identifiers; auto-binding later is by exact match to a real slot keyword.
- **Stops:** constants **or** expressions (v1) **or** `spectrum:<ref>` colours.
  - Uniform spacing across `[LO,HI]` is assumed.
  - A `p:<pos>` prefix pins the following stop to domain position `<pos>`.
  - Unspecified stops redistribute evenly: the first/last anchor to `LO`/`HI`, and
    each interior run of unpinned stops spreads evenly between its pinned neighbours.
  - Colours (`spectrum:` refs / rgb) interpolate in **linear RGB**, then
    Jakob–Hanika upsample the interpolated RGB back to a reflectance spectrum.
- **`interp nearest|linear|smooth`** (per record; default `linear`).
  `smooth` = monotone cubic Hermite (Fritsch–Carlson) — no overshoot.

### 2.2 Direct use (inline, in a geometry block's material field)

Extend the geometry `material <name>` field to also accept `<record>(<driver>)`:

```
sphere { center 0 0 0  radius 1  material funny_mirror(u) }
sphere { center 2 0 0  radius 1  material funny_mirror(noise(4*x,4*y,4*z)) }
```

The driver is a **pure pattern expression** (intrinsics + functions). Every channel
whose name matches a real slot fills that slot at the driven scalar; non-matching
channels and unfilled slots are ignored.

### 2.3 Named instance + overrides (a real `material "name" { … }` block)

Overrides live in a normal material block, as an **ordered statement list**:

```
material "m" {
    from funny_mirror(u)              # bulk import: fill every slot matching a channel, driver = u
    from other_rec(r)                 # a second import; on conflict the LATER from wins
    reflect   = transmit              # single slot <- another of THIS material's channels
    roughness = sin(v*3.14159)        # single slot <- an expression
    reflect   = funny_mirror.mirror   # single slot <- a named channel of any record (incl. non-slot names)
}
sphere { center 0 0 0  radius 1  material m }
```

**Semantics — ordered, last-write-wins:**
- The block is a list of statements evaluated top→bottom.
- `from R(driver)` bulk-assigns every slot whose name matches one of `R`'s channels,
  sampled with that `from`'s own driver. Multiple `from`s are allowed; a later `from`
  re-assigning a slot overrides an earlier one.
- `slot = <expr | channelref>` assigns one slot; overrides any preceding `from` or
  assignment to that slot.
- **Last write to a slot wins**, regardless of whether the writer was a `from` or an
  explicit assignment. Each `from` may carry a different driver, so different channel
  groups can be driven by different scalars.
- Right-hand side forms: an expression; a bare slot/channel name = "this material's
  <name> channel"; `RECORD.channel` = a named channel of any record; `RECORD.channel[i]`
  or `self.channel[i]` = the i-th stop selector.

---

## 3. Data model (C++)

A record compiles to a POD-friendly bank of per-channel LUTs, GPU-portable like
`pattern`/`ProcTexture`.

**General model (the spec).** A record is a driver domain `[LO,HI]` crossed with a
list of named **channels**; each channel outputs a value of arbitrary **arity**
`D≥1`, so a single record freely mixes per-row output dimensionality — an arity-1
roughness curve and an arity-3 rgb curve live side by side, each named, each
interpolated along the shared driver. A stop is therefore a `D`-tuple of component
programs, and interpolation (Nearest/Linear/Smooth) runs **per-component**. The
current `scalar` and `spectrum`/`rgb` channels are just the `D==1` and `D==3`
instances of this one thing; a `spectrum`/`rgb` channel is the `D==3` case tagged
for **colour-space** interpolation (linear-RGB lerp + Jakob–Hanika upsample) instead
of raw per-component lerp.

```
enum class RecInterp { Nearest, Linear, Smooth };
enum class RecSpace  { Linear, Colour };  // per-component lerp vs colour-correct(3)

struct RecStop {
    double pos;                            // domain position in [LO,HI] after redistribution
    // D component programs (constant or expression), one per output dimension;
    // each is evaluated per-hit, then components interpolate independently.
    std::vector<std::vector<PatNode>> comp;
    // colour fast-path: a stop authored `spectrum:<name>` / `rgb r g b` caches its
    // scene spectrum-table index (or an upsampled rgb); -1 = plain component stop.
    int    spectrum = -1;
};
struct RecChannel {
    std::string name;                      // arbitrary; matched to a slot keyword at bind time
    int         arity;                     // D: output dimensionality (1 scalar, 3 rgb/spectrum, …)
    RecSpace    space;                     // Linear (raw lerp) or Colour (D==3 colour-correct)
    std::vector<RecStop> stops;            // sorted by pos
    // precomputed monotone-cubic tangents (Fritsch–Carlson) when interp==Smooth,
    // one tangent series per component.
    std::vector<std::vector<double>> mTangent;
};
struct Record {
    std::string name;
    double lo = 0.0, hi = 1.0;
    RecInterp interp = RecInterp::Linear;
    std::vector<RecChannel> channels;
};
```

**What ftrace materializes today.** ftrace implements exactly two arities — the
scalar channel (`arity==1`, `space==Linear`) and the colour channel
(`arity==3`, `space==Colour`) — which is why the shipped C++ carries the narrower
`enum ChanKind { Scalar, Spectrum }` with a single `expr` / `int spectrum` per stop.
That is the concrete v1 realisation of the general model above, not a different
model: `Scalar` ≡ `(arity 1, Linear)`, `Spectrum` ≡ `(arity 3, Colour)`. Other
arities (a 2-vector, a 4-tuple) are valid in the spec and are where **loom** — the
authoring superset — carries the fully-general form (loom channels already hold
scalar-or-vector-of-any-dim values; see §J3 in `TODO.md`). If a real ftrace need for
a non-{1,3} arity appears, widen `ChanKind` toward the general `struct` above.

- Stored in `Scene` alongside `patterns` (a `std::vector<Record>` + name→index map).
- A **channel sample** at driver value `d`: clamp `d` to `[lo,hi]`, locate the stop
  interval, then Nearest/Linear/Smooth-interpolate **per component**. Component
  programs evaluate against the per-hit `PatCtx` first, then interpolate. A
  `Colour`-space (rgb/spectrum) channel interpolates in linear RGB then Jakob–Hanika
  upsamples the result.
- A **material driven by a record** carries: the record index, the compiled driver
  program (`std::vector<PatNode>`), and a slot→channel binding table (built by
  name-match, overridable). At shade time each bound slot = channel-sample(driver).

### 3.0 The type lattice (values · channels · records) — *target, not v1*

The whole authoring model reduces to **three value kinds and two containers**, with
one-way promotions between adjacent levels. This is the canonical vocabulary the rest
of §3 uses; the C++ `Record`/`RecChannel`/`RecStop` above are its concrete v1 slice.

**Values** (a value has no driver — it's a fixed quantity):

- **`number`** — a scalar. Its own type, *not* a degenerate spectrum: most quantities
  (roughness, IOR magnitude, blend weights, exponents, the `.5`/`a` in `.5*a`) are
  inherently scalar and would be meaningless as a colour.
- **`vector`** — a fixed-arity tuple of numbers (`1 1 1` = a position/scale/normal).
  Just numbers; **no** colour meaning on its own.
- **`spectrum`** — a colour: a curve over **wavelength λ**. `rgb .5 .5 .6`,
  `blackbody 6500`, `preset:D65`, `file:steel.csv` are all just ways of *writing one*.
  A spectrum has **no driver** — it is a value, not a mapping. (`rgb .5 .5 .6` is
  already "an array over λ"; that internal λ-axis is orthogonal to any driver axis and
  is never flattened into it.)

**Containers:**

- **`channel`** — a mapping **from a driver input to any type in the system**: a value
  (number/vector/spectrum), *or another channel*, *or a record*. A channel→channel
  (`u → (v → spectrum)`) is exactly a **multi-input** function by currying, so
  "several named inputs" (§3.2) is not a separate feature — it falls out of a channel
  containing a channel. The only fixed part is "it maps *from* a driver"; the result
  is unrestricted. (This is the same "channel" as `RecChannel` above — a record column
  *is* a driver→value mapping; the name is deliberately shared.)
- **`record`** — a bundle of **co-driven** channels: one declared `range`/driver
  sweeps every channel at once. The shared driver is the point of bundling (one `u`
  walks colour, roughness and displacement together), so a record is not "any channels
  in a bag" but "channels over a common input."

**Promotions are one-way, adjacent-level, and are what make the simple form typecheck
in the richer slot:**

- `number` → `spectrum`: a scalar in a colour slot lifts to a flat grey SPD, so
  `reflect = .5` is legal. The reverse never holds — a `spectrum` is **not** a
  `number` and can't fill a roughness slot. (A `vector` lifts to a `spectrum` only via
  an explicit colour keyword: `rgb 1 1 1` / `hsv …` — bare `1 1 1` stays a vector.)
- any value → **constant `channel`**: a value in a channel slot lifts to the
  zero-variation channel that ignores its driver (the §3.2 "bare constant is the
  degenerate curve").
- a single `channel` → **one-channel `record`**.

So every simpler thing is the degenerate/constant member of the next container up,
which is precisely why you can write the plain form and have it validate where the
richer one was expected.

**Slot-type vs value-expression are distinct.** A material property has two separable
parts: the **slot** (named by the leading type/slot keyword — `reflect`, `spectrum`,
`roughness`) declares *what type must come out*; the **value** is an **expression over
named inputs** that produces it. The value is *always* an expression — a constant
(`rgb …`), an open array (`[…]`), an applied channel (`[…](u)`), or a formula
(`a*.5`) are all one tier, never distinct "kinds" of property. Consequences:

- A bare array `[…]` is **driver-*open*, not implicitly-`u`**: its driver is unbound,
  filled by the slot's default input or by the consumer (§3.2). Sealing it as
  `[…](u)` is a *different* object (a function of `u`); leaving it bare hands the
  driver to the consumer. There is no hidden default driver.
- Because the value is just an expression over named inputs, both `spectrum = u*.5`
  and `spectrum = a*.5` are legal and mean different, sensible things (half the
  surface-`u` coord vs half the albedo). Allowing one *does* imply allowing the other
  — and that's the intended §3.2 "nothing is closed," not a problem to forbid.
- The value never needs to self-describe its destination: the **LHS slot keyword**
  names the output (`reflect = […]` vs `color = […]`). A standalone array is therefore
  **polymorphic data** — a bare list of stops — that the *assignment* pins to a slot,
  so the same array is reusable across colour/scalar/displacement slots.

### 3.1 Generalized stop grammar (arbitrary-arity, flexible delimiters) — *target, not v1*

Once a channel outputs an arbitrary-arity `D`-tuple (§3), each **stop** is itself a
`D`-tuple of components, and the channel is a *list* of stops — a small nested-array
structure. The general grammar descends that hierarchy (channel → stops → components)
with three delimiters that form a **precedence ladder**, not a free-order set:

> **whitespace binds like `×` (tightest), comma binds like `+` (looser), brackets
> `[ ]` are parentheses (explicit override).**

So `1 1 1, 2 2 2` parses exactly like `(1·1·1) + (2·2·2)` → two groups of three. At
every level, the *weakest separator present* defines that level's split, and the next
level down uses the next-weakest; brackets force an explicit level anywhere. This makes
**structure fully recoverable from the delimiters alone** — the channel's declared arity
is used only to *validate* the parsed shape (and to allow the bare-scalar `D=1` case
with no delimiters). All of these equivalences fall straight out of the ladder:

```
1 1 1                 ≡  [1 1 1]                         # 3-vector; bracketing one level is idempotent
1 1 1, 2 2 2, 3 3 3   ≡  [1 1 1] [2 2 2] [3 3 3]         # 3 stops of 3 components each
tint  [rgb 0 0 0, 0 1 0, 1 1 1]                          # tagged rgb channel, 3 stops
tint  rgb [0 0 0] [0 1 0] [1 1 1]                        # same, explicit per-stop brackets
```

The three ladder delimiters are exactly **`[ ]`, comma, and whitespace** — *parens `( )`
are not a grouping delimiter here*; they are reserved for expression grouping and the
function-application / rebinding surface of §3.2 (`prop(2)`, `gold.color(u=x)`,
`sin(v)`).

The subtlety that keeps it consistent: **whitespace separates *siblings* at the current
level** — when the siblings are scalars you get a vector; when they are already-bracketed
groups, whitespace concatenates them as a list of groups. Comma is just a *weaker*
sibling separator that opens a new outer level. An optional leading tag
(`rgb`/`spectrum`/…) fixes the channel's colour space and arity for validation; without
a tag the arity is whatever the delimiters produced.

**Position pins stay orthogonal to the ladder.** A stop may carry a leading
`POS:` prefix (`.2:0 0 0` = the group `0 0 0` pinned at driver position 0.2). `POS:` is
recognised as a prefix on a comma-group *before* the `+`/`×` split, so the colon never
competes with whitespace/comma/bracket.

**An additive superset, not a breaking change — and not yet in ftrace.** The generalized
ladder grammar is a *strict backward-compatible superset* of the current whitespace form:
a channel line is dispatched on whether it contains a top-level comma, so every existing
comma-free line keeps its exact current meaning (`reflect spectrum:steel spectrum:gold
spectrum:copper` = three colour stops, `rough 0 0 0` = three scalar stops), and only a
line that *introduces* a top-level comma opts into ladder parsing (`tint 0 0 0, 1 1 1` =
two arity-3 vector stops; a lone vector stop takes a trailing comma, `tint 0 0 0,`). No
existing record reparses differently. ftrace's *own* tokenizer (`src/ftsl.h`) still does
*not* treat `,` as a delimiter (a comma accretes into the preceding bareword) and makes
**every whitespace-word its own stop**, so today an rgb curve inside a record is written
as `reflect spectrum:steel spectrum:gold spectrum:copper` (one `:`-ref per stop), **never**
as inline `rgb r g b` triples. Teaching ftrace the superset requires only (a) a
comma-aware tokenizer pass and (b) the per-line comma dispatch above — purely additive.
Until then it lives in **loom** (the authoring superset, §J3b in `TODO.md`), which
parses/emits the unified grammar (`loom/record.py` — one `parse`/`emit` pair, comma
dispatch) and can lower a `D=3` channel down to the `spectrum:`-ref form ftrace
understands (synthesising the backing `spectrum` decls).

### 3.2 Binding, access, and override — *target, not v1*

A material property is an **expression over named inputs**, and access is *always
continuous* — there is no separate discrete stop-selector operator. (An array
`[0 0 0, 1 1 1]`, a pinned array `[0 0 0, .5:1 1 1]`, a formula `a*.5`, and a bare
constant `.5` are all just expressions that resolve to a value over the driver domain; a
constant "index" is simply a constant argument, `color(2)`, so the shipped discrete
`R.chan[i]` form is subsumed.)

A **bare constant is the degenerate curve** — a zero-variation function over the whole
domain: `reflect "reflect" = .5` ≡ `[.5]` ≡ `[.5 .5]`, so `gold.reflect(u)` is legal and
returns `.5` for every `u` (it simply ignores its driver). The consequence: **sampling
never fails on type grounds** — a consumer may always write `gold.prop(x)` without knowing
whether the author wrote a constant, an array, or a formula. "Constant vs array vs
function" is purely an *authoring* convenience, never a *usage* wall.

An **input** is one of:

- **system-provided, with a default at the shading point** — `a` (albedo), `u`/`v`
  (surface coords), the per-hit intrinsics, etc.; or
- **unbound** — no default; the consumer must supply it at access time.

**Nothing is ever closed.** Any named input can be rebound where the property is used —
`gold.reflectance(a=x)` rebinds a *system* input exactly the way `gold.color(u=x)`
rebinds an author-named one. The only difference between the property forms is **whether
the driver is bound to a defaultable input or left for the consumer**:

```
spectrum "color"   = [0 0 0, 1 1 1]      # driver UNBOUND. `gold.color` alone is underdetermined;
                                          #   consumer must drive it: gold.color(x)
spectrum "color"   = [0 0 0, 1 1 1](u)   # bound to u (defaults from geometry).
                                          #   gold.color works; gold.color(u=x) rebinds
reflect "reflect"  = .5*a                 # bound to a (defaults from albedo).
                                          #   gold.reflect works; gold.reflect(a=x) rebinds
```

The last two are the **same mechanism** — a different bound input and an array-vs-formula
body. Access is uniform:

- `gold.prop` — evaluate with every referenced input at its default *(valid only if all
  referenced inputs have defaults; a property with an unbound driver requires an
  argument)*.
- `gold.prop(x)` — bind the property's driver positionally.
- `gold.prop(name=x)` — rebind a specific named input.

**Purity note.** Writing `(u)` seals the array *inside* a function of `u`: the array is a
black box reachable only through the input, so `[…](u)` is genuinely a different object
from the bare `[…]`, not "data with a default binding." To let the consumer reach the
array you omit `(u)` (they drive it with `gold.color(x)`); to give a default *and* still
allow indexing you include `(u)` and they reach it via `gold.color(u=x)`.

**Naming is optional.** The **leading type/slot keyword identifies the property**; the
quoted name is only an external dot-handle. Write `spectrum "color" = …` to get
`gold.color` access, or just `spectrum = …` anonymously (still bound to its slot by the
keyword). You name a property only when you want to refer to it; omitting the name never
affects input binding (below), since input variables and property dot-handles are
different namespaces.

*(This whole section is the loom/J3b authoring superset. Shipped ftrace exposes the two
constant accessors `R.chan[i]` / `R(const)` from Stage 5a and drives records by the fixed
per-hit/`t` scope model; the uniform named-input rebinding surface above is the
generalized target, not v1.)*

### 3.3 Materials as parameterized bundles — *target, not v1*

A **material is a bundle of slot→expression bindings**, and it is itself a function: its
**free-input set is the union of its properties' free inputs**. Given

```
material "gold" = [ spectrum "color"    = [0 0 0, 1 1 1](u),
                    reflect  "reflect"  = .5*a ]
```

`gold` exposes the inputs `{u, a}`. **Applying** the material binds those inputs across
the whole bundle at once, at the use site:

```
material = gold(u=v, a=1)      # bind u<-v, a<-1
material = gold(u=v a=1)       # identical — the argument list uses the same ladder (comma == space)
material = gold(u=v)           # partial: a stays at its system default (albedo)
```

This is §3.2's per-property rebinding (`gold.color(u=x)`) lifted to **bundle granularity**
— several inputs bound in one call. Each `name=expr` RHS is evaluated in the *consumer's*
scope (so `u=v` feeds the consumer's surface coord `v` into gold's `u`), under the same
scope-check shipped in Stage 5a. Unbound inputs fall back to their system defaults;
inputs with no default and no binding are an error at the use site.

**Positional application** follows the per-property rule but guards the multi-input case:
`gold(v)` is allowed **only when the material has exactly one free input** (matching the
shipped `material NAME(driver)` inline form); a material with several free inputs
**requires named arguments** (`gold(u=v, a=1)`), since positional order is fragile. The
argument — positional or the RHS of a `name=…` binding — is an **arbitrary expression**
in the consumer's scope, not just a bare variable: `gold(v*2)` ≡ `gold(u=v*2)` binds the
sole free input to `v*2`, exactly as a per-property `gold.prop(v*2)` (§3.2) would.

*(Loom/J3b authoring superset — shipped ftrace has only the single-driver
`material NAME(driver)` inline form.)*

---

## 4. Build stages

1. **Tokenizer `[` `]` + declaration parse & data model.** Add `LBracket`/`RBracket`
   tokens. Parse `NAME = range LO-HI [ … ]` into `Record` (channels, stops, stop
   position redistribution, `interp`). No shading yet — just load, validate, and a
   round-trip unit test. Reject only structural errors (bad `range`, empty body);
   unknown channel names are NOT errors.
2. **Channel eval → material slots.** Implement channel-sample (Nearest/Linear/Smooth
   incl. Fritsch–Carlson tangents; scalar-expr stops; spectrum RGB-lerp + Jakob–Hanika).
   Wire a record into a material's slots by name-match. CPU only.
3. **Driver binding.** Compile the driver expression (reuse `compilePatternExpr`), add
   the inline `material NAME(driver)` form to the geometry material field. End-to-end
   CPU render of §2.2.
4. **`from` / override block.** Parse the ordered statement list in `material "m" { … }`
   (`from R(d)`, `slot = expr/channelref`, selectors `R.chan[i]`), apply last-write-wins.
5. **All-scope value sites.** Allow record-driven values anywhere a value is read
   (not just material slots) where it makes sense. Split into two sub-stages:
   - **5a — record refs as values (scope-checked).** Accept record references
     anywhere the value parser reads a spectrum or a scalar, in two *constant* forms:
     `R.chan[i]` (i-th stop selector) and `R(const)` (sample at a constant driver).
     Plus a **free-variable scope check**: every value site publishes the set of
     driver variables that are *in scope* there (a per-hit surface site publishes
     `x y z nx ny nz r u v f`; a load-time constant site — a light SPD, a camera
     scalar, a top-level spectrum — publishes the empty set). A driver expression is
     validated against that set and it is an **error** to reference an out-of-scope
     variable (e.g. `R(u)` in a light's SPD, where `u` has no meaning). This is the
     general rule that answers "can a light have a `from`?" — a light *may* carry a
     record ref, but only a constant one; a per-hit driver there is rejected because
     its variables aren't in scope, not because lights are special-cased.
   - **5b — camera-curve `t`-driver (DONE).** A second varying context besides
     per-hit surface intrinsics: a `camera_curve` / `camera_path` sweeps a flyby
     parameter `t`∈[0,1] as it expands into discrete `CamSpec` frames at load time
     (paralleling the existing `ScalarTrack` keyframe mechanism for roll/fov/zoom/
     fstop/focus). 5b publishes `t` as the in-scope driver variable for camera-scalar
     sites so a record can drive fov/roll/zoom/fstop/focus along the flyby
     (records-as-keyframe-tracks). Syntax: `<scalar>_from RECORD.channel[(driver)]`
     where `<scalar>` ∈ {`fov`,`roll`,`zoom`,`fstop`,`focus`}; the driver defaults to
     the raw timeline `t` and may be any expression in `t` (e.g. `zoom.fov(t*t)` for
     ease-in). A record track wins over an `_at` keyframe track, which wins over the
     authored base constant. **Scope model (leak-free, one flag + `patternHasFreeVars`
     reuse):** `t` is gated behind an `allowT` parameter that only camera `_from`
     sites pass to `tokenize`/`compilePatternExpr`; surface/constant sites leave it
     false so `t` hard-errors ("only in scope inside a camera_curve record track"),
     and the driver is additionally checked with `patternHasFreeVars` to reject
     surface vars ("only the flyby timeline `t` is in scope here"). The record stops
     themselves must be constant (no surface vars). Camera scalars are consumed at
     load time (baked into `CamSpec`), so this is CPU-only by construction — no GPU
     path needed (`dPatternEval` carries a `VarT`→0 case only for exhaustiveness).
     Validated frame-for-frame identical: `scenes/_cam5b_rec.ftsl` (`fov_from
     zoom.fov`, a `60→30` linear record) vs `scenes/_cam5b_trk.ftsl` (`fov_at 0 60 /
     fov_at 1 30` keyframe track) render bit-identical across all 5 flyby frames
     (0.000%, max 0.0). Everything outside these two varying contexts (per-hit
     surface; per-frame curve `t`) is load-time constant and admits only the constant
     record forms from 5a.
6. **GPU parity.** Upload the record LUTs + driver programs; device channel-sample
   mirroring the CPU path (bake like `ProcTexture`). Verify CPU/GPU bit/visual parity.
   Split into 6a/6b mirroring the 5a reflect/scalar split:
   - **6a — reflect slot (DONE).** Constant `selStop` bindings bake into the device
     material's `reflect[]`; per-hit driven bindings upload the baked JH coeff LUT
     (`DScene::recCoeff`) + driver program (`DScene::recDrivers`) and sample on-device
     via `dRecordReflect`/`dReflectSlot`/`dDiffuseRho` (twins of `recordReflectBound`).
     Forward-only: `cudaForwardSupported` accepts reflect records; `cudaBdptSupported`
     rejects *all* record bindings because the BDPT connection BSDF (`dBsdfF`) has no
     per-hit `DHit` to evaluate a driver. Validated on `scenes/_record_bind.ftsl`.
   - **6b — scalar slot (roughness) (DONE).** Scalar stops evaluate per-hit (they may
     reference hit vars), so they are NOT baked to a LUT: each stop's compiled `expr`
     program uploads via `DScene::recScalarStops` (`DRecScalarStop{pos,exprOff,exprN}`)
     + the shared `recDrivers` program pool, and `dRecSampleScalar` mirrors `recSampleScalar`
     exactly (recLocate → nearest/linear/monotone-cubic Fritsch–Carlson, evaluating each
     bounding stop at the hit first). `DMaterial.recRoughMode` (−1/0/1/2) routes
     `dMatRoughness` → `dRecordRoughness`; `cudaForwardSupported` now accepts roughness
     records. Validated on `scenes/_record_rough.ftsl` (driven `rough(u)`/`rough(noise)`)
     and `scenes/_record_override.ftsl` (mode-0 direct-expr roughness). This change also
     completed 6a's routing in the backward-reference megakernel (mirror/grating/
     halfmirror/glossy reflect reads → `dReflectSlot`). BDPT still CPU-only.

Each stage: build (`cmake --build build_cuda2 --config Release --target ftrace`, then
`cp build_cuda2/bin/ftrace.exe ftrace.exe`), add tests, validate with a windowed
render, commit at green. Update `FTSL.md` (grammar), `README.md` (feature), and
`docs/scene-language.md` as the observable surface grows.

---

## 5. Open / deferred

- Using channel names directly inside expressions (so an expression could reference
  another channel's sampled value) — desired "in the future", not v1.
- **Multi-dimensional *input* domain** (non-scalar drivers — a single array
  addressed by several named driver *axes*, not one `range` scalar) — still deferred;
  ftrace v1 keeps one `range` domain / one driver. This is the loom-side superset
  (§J3b in `TODO.md`); revisit for ftrace only if a real need appears.
  - *Note — distinct from output arity, which is NOT deferred:* per-channel output
    dimensionality is fully general in the spec (§3) — each channel outputs an
    arbitrary-arity `D`-tuple, and records already mix arities (scalar curve beside
    rgb curve). ftrace materializes `D∈{1,3}` today; loom carries all arities. The
    only thing still 1-D is the driver *input* domain above.
- **Generalized stop grammar** (arbitrary-arity `D`-tuple stops with interchangeable
  `[ ]` / `,` / whitespace delimiters down the channel → stops → components hierarchy;
  §3.1) — *the spec's target form, deferred in ftrace.* Today ftrace's tokenizer isn't
  comma-aware and every whitespace-word is its own stop, so inline `rgb r g b` triples
  inside a record aren't parseable — colour stops must be `spectrum:<name>` refs.
  Enabling the general grammar is a real tokenizer + parser change; until then it lives
  in loom (§J3b), which may author the flexible form and lower `D=3` channels to the
  `spectrum:`-ref form ftrace understands. The delimiter **precedence ladder**
  (whitespace `×` < comma `+` < brackets = parens) that makes stop structure recoverable
  from delimiters alone is specified in §3.1.
- **Uniform named-input binding / rebinding surface** (§3.2) — properties as expressions
  over named inputs (system-provided-with-default like `a`/`u`/`v`, or unbound),
  continuous-only access (`prop`, `prop(x)`, `prop(name=x)`), any input rebindable at the
  use site — *deferred, loom-side (§J3b).* Shipped ftrace exposes only the two constant
  accessors `R.chan[i]` / `R(const)` (Stage 5a) under the fixed per-hit / camera-`t`
  scope model; the generalized rebinding surface is the target, not v1.
