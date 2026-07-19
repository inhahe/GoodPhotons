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
`pattern`/`ProcTexture`:

```
enum class RecInterp { Nearest, Linear, Smooth };
enum class ChanKind  { Scalar, Spectrum };          // scalar LUT vs colour LUT

struct RecStop {
    double pos;                     // domain position in [LO,HI] after redistribution
    // scalar stop: a compiled pattern program (constant or expression)
    std::vector<PatNode> expr;      // evaluated per-hit; a constant compiles to one Const node
    // colour stop: index into the scene spectrum table (or an upsampled rgb)
    int    spectrum = -1;
};
struct RecChannel {
    std::string name;               // arbitrary; matched to a slot keyword at bind time
    ChanKind    kind;
    std::vector<RecStop> stops;     // sorted by pos
    // precomputed monotone-cubic tangents (Fritsch–Carlson) when interp==Smooth
    std::vector<double> mTangent;   // scalar channels only
};
struct Record {
    std::string name;
    double lo = 0.0, hi = 1.0;
    RecInterp interp = RecInterp::Linear;
    std::vector<RecChannel> channels;
};
```

- Stored in `Scene` alongside `patterns` (a `std::vector<Record>` + name→index map).
- A **channel sample** at driver value `d`: clamp `d` to `[lo,hi]`, locate the stop
  interval, then Nearest/Linear/Smooth-interpolate. Scalar stops evaluate their
  `expr` against the per-hit `PatCtx` first, then interpolate the resulting scalars.
  Spectrum stops interpolate in linear RGB then Jakob–Hanika upsample.
- A **material driven by a record** carries: the record index, the compiled driver
  program (`std::vector<PatNode>`), and a slot→channel binding table (built by
  name-match, overridable). At shade time each bound slot = channel-sample(driver).

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
   - **5b — camera-curve `t`-driver (optional, gated on user go-ahead).** A second
     varying context besides per-hit surface intrinsics: a `camera_curve` /
     `camera_path` sweeps a flyby parameter `t`∈[0,1] as it expands into discrete
     `CamSpec` frames at load time (paralleling the existing `ScalarTrack` keyframe
     mechanism for roll/fov/zoom/fstop/focus). 5b would publish `t` as the in-scope
     driver variable for camera-scalar sites so a record can drive fov/roll/zoom/
     fstop/focus along the flyby (records-as-keyframe-tracks). Everything outside
     these two varying contexts (per-hit surface; per-frame curve `t`) is load-time
     constant and admits only the constant record forms from 5a.
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
- Non-scalar drivers / multi-dimensional records (a single array of any
  dimensionality with several named driver axes) — the general form the user floated;
  v1 is 1-D (one `range` domain, one driver). Revisit if a real need appears.
