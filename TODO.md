# TODO — consolidated undone work

Single source of truth for everything planned-but-not-done, pulled together from the
project's scattered plan files. Check items off (`[x]`) as they land, and mirror the
status back into the originating file (`DESIGN.md`, `ROADMAP.md`, `OSCILLATE_GRAMMAR.md`,
`ROADMAP_heroroom.md`) when a whole section closes.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.
Origin tags point at the authoritative design text for each item.

---

## NEXT UP — unify element headers to `name = KIND { … }` (do before the ftrace grammar port)

- [x] **Switch the whole scene grammar + loom emitters to `name = KIND { … }` (and anonymous `KIND { … }`).**
      *(User-approved 2026-07-19; DONE.)* Replace the inconsistent `KIND "name" { … }` headers
      (`camera "cam" {…}`, `material "gold" {…}`, `texture "hide" {…}`, `light … {…}`) with a single binding
      shape: **`hero = camera { … }`**, **`gold = material { … }`**, **`hide = texture { … }`**, with the
      **anonymous** form just dropping `name =` (`camera { … }`, a nameless light, etc.). Records already use this
      shape (`NAME = range LO-HI [ … ]`), so they fold in as the `range` kind — the entire grammar collapses to
      `binding = (NAME '=')? KIND block`. Reads like assignment and makes anonymity natural.
      **Why now:** we're about to freeze the shared `ftsl.epeg` grammar into ftrace's C++ front-end (J3c option-a),
      so change loom's emitters (`scene.py` `emit`) **and** the shared grammar + reader **in lockstep**, once, so both
      sides adopt the new header together and we never ship two spellings into the C++ parser. **Scope:** (1) loom
      `Element.emit` for camera/material/texture/light (+ any other named block); (2) `ftsl.epeg` element rules +
      `reader.py` builders (name now optional, kind after `=`); (3) update `test_grammar_*` samples + any golden
      `.ftsl` fixtures; (4) decide the migration story for existing hand-authored `.ftsl` (support both during a
      transition, or a one-shot converter — TBD). Breaking change to emitted `.ftsl`, but pre-freeze is the cheapest
      moment. *(Design detail also mirrored in the J3c "PROPOSAL" bullet below; this NEXT-UP entry is the actionable
      one.)*
      **Migration story DECIDED:** ftrace's hand-written C++ parser **accepts both spellings** during the transition
      (`KIND "name" {…}` and `name = KIND {…}`); loom + the shared grammar hard-switch to the new form; old-form C++
      support is dropped once the shared grammar is ported into ftrace's front-end. This keeps the pipeline green at
      every commit.
      **Done (2026-07-19, v0.9.1):**
      *Increment 1 — ftrace C++ accept-both.* `parseOneTopBlock` now parses `NAME = KIND { … }` alongside
      `NAME = range …` and legacy `KIND "name" {…}`; the light subtype falls back to a `kind` property when no bareword
      subtype is present (`sun = light { kind sphere … }`), at both the top-level and group-nested dispatch sites.
      *Increment 2 — loom + shared grammar hard-switch to the new form.* Flipped every loom emitter to
      `NAME = KIND { … }` / anonymous `light { kind … }` (`scene.py` texture/material/proctexture/light/camera/
      camera_curve, `iso.py` isosurface, `material.py` pattern + mix material); updated the shared `ftsl.epeg`
      (`binder = NAME '='`; `binder?` prefix on material/texture/light/camera; light `kind` as a body property) and the
      `reader.py` builders (`_binder_name`, light kind from a property). Updated the emit/header assertions across the
      loom test suite. **Verified:** loom suite 734 passed; a loom-emitted new-form scene (`scraps/loom_emit_test.ftsl`)
      renders in ftrace; both `scraps/newform_test.ftsl` and legacy `scraps/oldform_test.ftsl` parse. Old-form C++
      support stays until the shared grammar is ported into ftrace's front-end.

- [~] **DECISION — color-vector / array syntax (locked in conversation 2026-07-19).** How numbers, commas,
      brackets, and colorspace keywords (`rgb`/`hsl`/`hsv`) group into colors and lists of colors. Settled model,
      to bake into the shared grammar + per-field shape validation at the same time as the header switch:
  - **Whitespace only joins scalars *inside* one vector; it never crosses an array boundary and never starts a
    new array.** So `2 0 0 3 0 0` is a **single 6-number vector** — invalid as a color (a color is exactly 3),
    and *not* two colors. There is **no** "chunk-a-bare-run-into-triples" rule.
  - **Array boundaries must be marked** — by a **comma**, by **brackets**, or by a **keyword** (`hsl`, a field
    name, …). The valid spellings of the two colors 2 0 0 and 3 0 0 are exactly: `2 0 0, 3 0 0` · `[2 0 0] [3 0 0]`
    · `[2 0 0], [3 0 0]` · `[2, 0, 0], [3, 0, 0]`.
  - **Comma vs whitespace.** At a *given* level they're interchangeable (inside one vector `2 0 0` ≡ `2, 0, 0`;
    between two *already-closed* siblings `[a] [b]` ≡ `[a], [b]`). The one thing only a comma can do (a space
    can't) is **close a bare, un-bracketed run and open a sibling**. A `]` or a keyword closes a run the same way.
  - **Comma's role is decided by its operands.** Between **lone scalars** it's a component separator → one vector
    (`1, 0, 0` = the single color (1,0,0), same as `1 0 0`). Between **space-grouped multi-number groups** it's an
    array separator → a list of vectors (`2 0 0, 3 0 0` = two colors). Corner that falls out: `1, 0, 0, 2, 0, 0`
    (all lone scalars) is a **single 6-vector → invalid**, not two colors; to get two colors the triples must be
    space-grouped or bracketed.
  - **`rgb`/`hsl`/`hsv` are inline *modal RLE-style tags*, not array-openers.** They sit *between* elements and set
    the colorspace for the *run* of colors that follows, until the next keyword. So
    `rgb 1 0 0, 2 0 0, 3 0 0, hsl 4 0 0, 5 0 0, 6 0 0` is **one flat array of 6 colors** (first 3 decoded rgb, last
    3 hsl) — depth-1. A single flat palette can therefore mix source colorspaces with no brackets.
  - **`[X] ≡ X` is a whole-value identity only.** A *lone* top-level bracket is transparent (`rgb [1 0 0]` =
    `rgb 1 0 0`; `rgb [c,c,c]` = `rgb c,c,c`). The instant `[X]` has a **sibling**, the brackets are load-bearing —
    they *are* the separate arrays and can't be dropped (`[2 0 0] [3 0 0]` ≠ `2 0 0 3 0 0`). Brackets scope their
    inner commas and close their run; that's their whole job.
  - **Only brackets nest.** `rgb [1 0 0, 2 0 0, 3 0 0], hsl [1 0 0, 2 0 0, 3 0 0]` is two sibling bracketed arrays
    → depth-2 `[[c,c,c],[c,c,c]]`, whereas the comma/keyword flat form above is depth-1.
  - **Syntax vs shape stays split.** The grammar is context-free and accepts any nested value-tree; each field then
    **shape-checks** it. Handing the depth-2 `[[c,c,c],[c,c,c]]` to a field that wants a *flat* color list is
    **well-formed syntax, wrong shape** → a shape error, not a parse error. It's field-relative: the same tree is
    valid for a field that wants a list of palettes. Implement as generic parse + per-field schema validation with
    good "expected flat list of colors, got list-of-lists" messages.
  - **ADDENDUM — axis-labelled arrays + N-D grid/scatter datatypes (design intent, 2026-07-20; user).** For the
    language overhaul, an array literal may carry a **trailing axis-label tuple** naming its domain axes:
    `[0 1](u)` (1-D over axis `u`), `[[0,1,2][3,4,5][6,7,8]](u,v)` (2-D over `u,v`), `(x, y, z)` for 3-D, etc.
    The labels name the *independent* coordinate axes the samples are indexed by, so e.g. a **material reflectance**
    can be authored as `reflect [0 1](u)` (reflectance sampled along `u`). Design points to bake in:
    * **The N-D grid and N-D scatterpoint grammars must accept the same `(x,y,z)`-style axis-label tuple** as the
      plain multi-dim array literal — one shared trailing-labels production, reused by array / grid / scatter.
    * **`[[0,1,2][3,4,5][6,7,8]]` is a *hybrid* of the grid and scatter datatypes** — lockstep (regular) in one
      dimension, dynamic (ragged / irregular) in the other. So the value tree must permit ragged inner rows and a
      per-row/per-element **domain range** on a sample (e.g. the `[6 .2:6.2 8]` row, where `.2:6.2` is a range
      element). A fully-ragged case is a **scatterpoint**, not a rigid grid → it should use loom's *scatter*
      interpolant (`loom/interp.py` scatter path), not the regular-grid curve; a fully-regular case uses the grid
      curve. The hybrid picks per-axis.
    * **`rgb`/`hsl`/`hsv` colour tags must be accepted anywhere an N-by-3 array is accepted** (not just the
      purely-spectral sites) — i.e. wherever a value is shaped as a list/array of 3-vectors, the inline modal
      colour tag applies to the run exactly as in the flat-palette rule above.
    * These land with **N-D scatterpoint + N-D grid datatypes ported into ftrace** (mirroring loom's `data.Grid` /
      scatter + `interp.py` curves) — see the loom→ftrace data-port item. Grammar first (shared `.epeg`), then the
      C++ front-end at the J3c port, then the runtime sampler.
  - **ADDENDUM — case-insensitive *keywords* (future intent, 2026-07-20; user).** The user wants FTSL keywords to
    (maybe, later) be **case-insensitive** — but **only keywords** (block kinds, property names, enum/mode values,
    spectrum/colour heads like `rgb`/`blackbody`/`gaussian`), **never custom identifiers** (record/material/light
    names, and library-ref *names* like the `Gold` in `metal:Gold`). Today everything is case-sensitive (heads are
    matched literally in `evalSpectrum`; only colour-*names*, presets, and file extensions get `tolower`). Doing it
    right is a front-end-wide audit (fold keyword tokens to lower at the lexer/dispatch layer while leaving identifier
    tokens untouched), best done as part of the J3c C++ grammar port rather than piecemeal. Not scheduled — captured
    so it isn't lost.
  - **Progress (2026-07-19): reference implementation landed in the shared grammar.** Added a context-free `value`
    rule to `ftsl.epeg` (`value = vrun (',' vrun)*`; `vnums = (NUMBER|REF)+` for a whitespace vector; brackets nest;
    `colour_tag` = `rgb`/`hsl`/`hsv`). New `loom/grammar/values.py`: canonical `Vec`/`Arr`/`Ref` tree +
    `parse_value` (normalizer resolving the comma-role rule, RLE colorspace tags, and the `[X] ≡ X` bracket identity)
    + per-field shape validators `as_scalar` / `as_vector` / `as_color` / `as_color_list` (raising `ShapeError` with
    "expected …, got …" messages, distinct from syntax `ValueError`s). `tests/test_grammar_values.py` pins every
    example in this decision (23 cases; loom suite 757 passed).
  - **Progress (2026-07-19): first real consumer wired — the colour round-trip.** `loom.color` now has
    `Color.parse` / `parse_color` / `parse_color_list` (exported from the package), the inverse of `Color.token`:
    they read a `.ftsl` colour token / flat palette back into animatable `Color` objects through
    `values.as_color` / `as_color_list`, so every locked spelling (bare/bracketed triple, comma or bracket-sibling
    lists, inline RLE `rgb`/`hsl`/`hsv` tags, the `[X] ≡ X` identity) is accepted and the shape rules are enforced
    (a 6-vector or a colour *list* handed to the single-colour reader → `ShapeError`; an unbalanced bracket →
    `ValueError`). `tests/test_color_parse.py` (19 cases) pins the round-trip and the errors; full loom suite 776
    passed.
  - **Progress (2026-07-19): fuller spectrum grammar + purely-spectral field validation wired.** Built the
    spectrum-expression layer the field-wiring needed: new `loom/grammar/spectrum.py` (`parse_spectrum` /
    `as_spectrum` → canonical `Const`/`Blackbody`/`Ior`/`WhiteWall`/`NamedWall`/`Band`/`ColorSpec`/`LibRef`/
    `RecordRef` nodes), a faithful 1:1 mirror of ftrace's `evalSpectrum` (`src/ftsl.h` ~1106) — a bare number,
    `blackbody`/`ior`/`whitewall` (+defaults), `redwall`/`greenwall`, `gaussian`/`shortpass` bands, tagged
    `rgb`/`hsv`/`hsl` colours (delegating to `as_color`), `glass:`/`metal:`/`reflectance:`/`filter:`/`preset:`/
    `file:`/`spectrum:` refs, and constant record channel refs; an *untagged* colour triple is rejected exactly as
    ftrace rejects it (`tests/test_grammar_spectrum.py`, 22 cases). **Extended the shared `.epeg` grammar** so it
    can actually express these: `pvalue` was `REF | STRING+ | NUMBER+ | NAME` and could NOT parse `reflect rgb r g b`
    or `blackbody K` (which is why loom had been emitting the ftrace-*invalid* untagged triple); it is now a
    spectrum/value **word run** `pwords = phead ptail*` (head word + trailing numbers / `key=value` band words via a
    new `KVWORD` terminal), stopping at the next NAME key. **Wired `as_spectrum` into the reader** for the
    *purely-spectral* fields (`ior`/`transmit`/`absorb`/`substrate_k`/`emit`, light `spd`) — non-destructive
    shape-checks that reject exactly what ftrace would. Verified end-to-end: a scene with `reflect rgb …`,
    `absorb shortpass edge=… slope=…`, `emit gaussian center=… sigma=…`, `spd blackbody …` parses+renders in ftrace.
    Full loom suite **808 passed**. Two source findings logged in `known-issues.md` (stale `absorb 3 0.5 0.3` comment
    in `ftsl.h`; loom light `color`/`size`/`turbidity` props ftrace ignores).
  - **Progress (2026-07-19): binding-union fields (`reflect`/`roughness`/`*_map`) now validated.** Built
    `loom/grammar/bindings.py` — per-field validators mirroring ftrace's `bindReflectTexture` /
    `bindScalarTexture` / `bindScalarPattern` + `spectrumParam` / `dblParam` (`src/ftsl.h` `buildMaterial`):
    `as_color_binding` (`reflect`: `texture:<name>` | spectrum — note reflect binds *only* a UV texture, never a
    `pattern:`), `as_scalar_binding` (`roughness`: `pattern:` | `texture:` | one scalar number — not a spectrum),
    `as_map_binding` (any `*_map`, i.e. `film_thickness_map`/`weight_map`: `pattern:` | `texture:` only). Wired into
    the reader's `_build_material` (`_validate_bindings`); shape-only (a bound name's scene membership stays a later
    check). This closes the last known gap and *corrects* a prior test that assumed `reflect` accepted an untagged
    triple — ftrace rejects it, so loom now does too. `tests/test_grammar_bindings.py` (10 cases) +
    `test_grammar_material.py` additions. Full loom suite **823 passed**.
    *Remaining:* the record-driven **whole-material override** block (`from R(...)` + `slot = REC.chan`,
    ftrace's `isRecordOverrideBlock`/`buildRecordOverrideMaterial`) — loom does not emit it, so it is out of the
    reader's scope for now; records still use their own PIN-carrying stop parsing (not routed through the shared
    value grammar). Then mirror the `value`/`spectrum`/per-field binding grammar into ftrace's C++ front-end at the
    **J3c C++ port** (where ftrace's front-end adopts the shared grammar as its single source of truth).
  - **Progress (2026-07-19): whole-scene GPDA grammar proven against the real corpus (shim prerequisite).**
    Before building the J3c C++ *validation shim* (run the shared grammar alongside ftrace's hand-written parser,
    diff Block trees, warn on mismatch, then flip to full replacement), we first had to prove the GPDA engine + a
    shared grammar can parse the **full** `.ftsl` language ftrace's parser actually accepts. Wrote
    `loom/grammar/ftsl_scene.epeg` — a *generic* whole-scene grammar (block/stmt/value, NOT the typed six-element
    `ftsl.epeg`) that faithfully mirrors ftrace's front-end: the trivial tokenizer (everything non-delimiter is one
    `Word`; `\r` skipped, `\n` significant), the `parseValue` continuation rule (first token unconditional, then
    continue only on numbers / `k=v` words / strings, stop at a plain bareword), and every `parseOneTopBlock` form
    (`prefer`/`else`, `NAME = range … [ … ]` records, `spectrum … = value`, unified `NAME = KIND [sub] {…}`, legacy
    `KIND ["name"] [sub] {…}`). **Coverage: 278/278 authored+scrap scenes parse** (100%; plus 129 sampled generated
    frames earlier = 407 files). **Structure verified** (not just acceptance): a spot-check walks the ParseNode tree
    and recovers each top-block's (type, name, subtype) + nested-block children for representative scenes covering
    *every* special construct — prefer/else (2 branches, nested cameras), record `[…]` bodies, spectrum `=`,
    record-override `slot = REC.chan` materials, unified assign-header, and nested CSG (isosurface/blob/difference,
    group). Harnesses live in `scraps/` (git-ignored): `scene_grammar_coverage.py`, `scene_grammar_structure.py`.
    *Next:* build the C++ shim — integrate the tokenized GPDA engine
    (`D:\visual studio projects\GraphParser\cpp\tokenized.{hpp,cpp}`) into ftrace, load `ftsl_scene.epeg`, produce
    `blocks_new`, structurally diff vs the hand-written parser's `blocks_old` behind a non-authoritative flag,
    iterate to silence on the corpus, then replace.
  - **Progress (2026-07-19): J3c C++ validation shim LANDED and wired into ftrace (non-authoritative).** Built the
    full C++ pipeline and vendored it into `src/gpda/`: (1) `loom/grammar/emit_cpp.py` compiles the authoritative
    `ftsl_scene.epeg` to standalone C++ that rebuilds the identical GPDA `Graph` + lexer table (`ftsl_scene.gen.cpp`
    — 193 nodes / 29 rules / 16 lex rules; no runtime file/JSON dependency); (2) the tokenized GPDA engine
    (`pool.hpp`, `tokenized.{hpp,cpp}`, copied verbatim from `GraphParser/cpp`); (3) a reusable regex longest-match
    lexer (`gpda_lexer.hpp`); (4) `ftsl_reduce.hpp` — the ParseNode→`ftsl::Block` reducer + structural differ that
    faithfully mirrors `parseValue`/`parseBraceBody` (value continuation, record-override `= REC.chan [i]`, `[i]`
    selector folding, nested-block type/name derivation, flat `words` dump, quote-stripping); (5) `ftsl_shim.hpp` —
    `ftsl_shim::validate(src, blocks, path)`, guarded by the `-validate-grammar` CLI flag / `FTRACE_VALIDATE_GRAMMAR`
    env var, off by default (zero cost). `ftsl.h` includes the shim after `Block/Stmt/Value/Parser` are defined and
    calls it in `load()`; `main.cpp` adds the flag; `CMakeLists.txt` builds the two vendored `.cpp` at `/W0`.
    **Proof:** the standalone harness (`scraps/gpda_shim/`) diffs the GPDA parse against ftrace's *actual* parser
    slice over the **entire corpus — MATCH 2338/2338, zero parse failures, zero mismatches**; ftrace built with the
    shim (CUDA Release) and ran `-validate-grammar` live on feature-rich scenes (prefer/else, spectrum,
    record-override, envmap) with **zero `[validate-grammar]` warnings**. **VERSION 0.9.1 → 0.9.2.**
    *Next:* run `-validate-grammar` broadly as scenes are authored/rendered to keep the mismatch count at zero, then
    flip ftrace's front-end over to the shared grammar as the single source of truth (full replacement). Deferred
    reconciliations (logged): record-driven whole-material override block (ftrace `isRecordOverrideBlock`, not emitted
    by loom); loom Light `color`/`size`/`turbidity` props ftrace ignores; stale `absorb 3 0.5 0.3` comment
    (`ftsl.h` ~1838, see known-issues.md).

---

## 0. Parametric records — FTSL data structure  *(design locked; full spec in `ROADMAP_records.md`)*

A named record over a scalar domain whose channels are named after real material slots,
sampled by a per-hit driver expression, with nearest/linear/smooth interpolation and
ordered last-write-wins `from` composition. **See [`ROADMAP_records.md`](ROADMAP_records.md)
for the authoritative spec and the 6-stage build plan.**

- [x] **Stage 1** — tokenizer `[` `]` + `NAME = range LO-HI [ … ]` declaration parse & data model. *(committed 0e24f07-precursor)*
- [x] **Stage 2** — channel eval (nearest/linear/smooth + expr stops + spectrum RGB-lerp→Jakob–Hanika) → slots. *(0e24f07)*
- [x] **Stage 3** — driver binding + inline `material NAME(driver)` in geometry. *(b3f42ce)*
- [x] **Stage 4** — `material "m" { from R(d) … slot=expr/channel }` ordered last-write-wins + selectors + record-aware specular reflect. *(989f21f)*
- [x] **Stage 5 — all-scope value sites** *(DONE; split in `ROADMAP_records.md §4`)*:
  - [x] **5a** — record refs as *constant* values (`R.chan[i]`, `R.chan(const)`) at any value site, + a free-variable scope check that errors on out-of-scope drivers (each site publishes its in-scope driver axes; a load-time constant site publishes ∅). **DONE** (commits abac704 spectrum sites, <this> scalar sites). Colour sites via the `evalSpectrum`/`spectrumParam` chokepoint (`spectrum = R.chan[i]`, `reflect R.chan[i]`/`R.chan(c)`); scalar material slots (roughness, film_ior, film_thickness, groove_spacing, max_order, yield) via a new `dblParam` chokepoint mirroring `spectrumParam`. Checks: scope (per-hit driver at a constant site → error), stop index range, channel-kind mismatch (colour↔scalar), and non-constant (per-hit-expression) stops rejected. Constant refs bake to ordinary Spectrum/double at load time → **fully GPU-eligible** (no record CPU-fallback). Standalone test green; validated with `scenes/_record_value.ftsl` (colour) + `scenes/_record_scalar.ftsl` (frosted-glass roughness). *Not yet routed: non-material scalar sites (camera/film/pattern scalars still call free `dblOf`) — deferred until a real need. Driving a camera scalar from a record is really 5b territory (a `t`-driver); other constant scalar sites can be routed through `dblParam` trivially if wanted.*
  - [x] **5b** — camera-curve `t`-driver: publishes flyby param `t`∈[0,1] as an in-scope axis so a record can drive fov/roll/zoom/fstop/focus along a `camera_curve`/`camera_path`. Syntax `<scalar>_from RECORD.channel[(driver-in-t)]` (driver defaults to raw `t`); a record track beats an `_at` keyframe track beats the authored constant. Scope stays leak-free: `t` is gated behind an `allowT` flag only camera `_from` sites pass to `tokenize`/`compilePatternExpr` (surface/constant sites hard-error on `t`), and the driver is additionally checked with `patternHasFreeVars` to reject surface vars; record stops must be constant. Camera scalars bake into `CamSpec` at load → CPU-only by construction (no GPU path; `dPatternEval` carries a `VarT`→0 case for exhaustiveness only). **DONE** — validated frame-for-frame bit-identical: `scenes/_cam5b_rec.ftsl` (record `fov_from zoom.fov`, 60→30 linear) vs `scenes/_cam5b_trk.ftsl` (`fov_at 0 60 / fov_at 1 30` track), all 5 flyby frames 0.000% / max 0.0.
- [x] **Stage 6 — GPU parity** *(bake like `ProcTexture`; split reflect/scalar mirroring 5a)* — **DONE** (6a reflect + 6b scalar/roughness both landed):
  - [x] **6a** — GPU parity for the **REFLECT** slot. A *constant* `selStop` reflect binding bakes its stop colour straight into the device material's `reflect[]` at upload (plain `specLookup` path, no device branch). A *per-hit driven* reflect binding uploads the channel's baked JH coeff LUT (`DScene::recCoeff`, `REC_LUT_N*3` doubles) + its driver program (`DScene::recDrivers`), and `dRecordReflect`/`dReflectSlot`/`dDiffuseRho` sample it on-device (device twin of `recordReflectBound`). Routed through every h-available reflect read in the forward megakernel + specular-walk gather (mirror/grating/halfmirror/glossy/diffuse). `cudaForwardSupported` now accepts reflect records (only scalar/roughness records still force CPU); `cudaBdptSupported` rejects **any** record binding (the BDPT connection BSDF `dBsdfF`/`dBsdfPdf` has no per-hit `DHit`, so it can't sample a driver → MIS bias). **DONE** — validated CPU↔GPU parity on `scenes/_record_bind.ftsl` (driven `grad(u)` + `grad(noise)`, mean per-channel match <0.2%, diff is pure MC noise) and a constant-`selStop` copper sphere matching a plain-copper reference on GPU. *(commit <this>)*
  - [x] **6b** — GPU parity for the **SCALAR** (roughness) slot. Scalar stops evaluate **per-hit** (they may reference hit vars), so they do NOT bake to a LUT — instead each stop's compiled `expr` program is uploaded (`DScene::recScalarStops` as `DRecScalarStop{pos,exprOff,exprN}` + programs in `recDrivers`), and `dRecSampleScalar` is an exact device twin of `recSampleScalar` (recLocate → nearest/linear/monotone-cubic Fritsch–Carlson, evaluating each bounding stop's expr at the hit first). `DMaterial.recRoughMode` (−1 none / 0 direct-expr / 1 constant selStop / 2 driven) routes `dMatRoughness` through `dRecordRoughness`. `cudaForwardSupported` now accepts roughness records too (all record forms are GPU-eligible for forward + backward-reference mode R). This commit also completes 6a's routing: the backward-reference megakernel's mirror/grating/halfmirror/glossy reflect reads now go through `dReflectSlot` (they were still calling `specLookup(mp->reflect)` directly). `cudaBdptSupported` still rejects all record bindings (BDPT connection BSDF has no per-hit `DHit`). **DONE** — validated CPU↔GPU parity on `scenes/_record_rough.ftsl` (driven `rough(u)` + `rough(noise)`, diff shrinks 4.7%→2.3% as noise falls 5%→1.8%, means match <0.1 = pure MC noise) and `scenes/_record_override.ftsl` (mode-0 direct-expr roughness + driven/selStop reflect together, 0.65% mean diff). *(commit <this>)*

---

## A. Camera-curve bridge + orientation axes  *(design locked in conversation; NOT previously written to any file — captured here so it isn't lost)*

**Context.** loom's `Camera` (`tools/loom/loom/scene.py` `Camera.emit`, ~line 396) currently
**bakes** a static per-frame `camera "name" { eye … look_at … up … fov_y … mode … film {res} }`
block — it animates by re-emitting numbers every frame and does **not** emit a real
`.ftsl camera_curve`. ftrace's `camera_curve` (`src/ftsl.h` `addCameraCurve`, ~line 2883)
today does a Catmull-Rom **position** spline + arc-length/density reparam + scalar tracks
(`roll_at`/`fov_at`/`zoom_at`/`fstop_at`/`focus_at`) + look modes (tangent default / `look_at`
/ `look_curve`+`look_point`) + world `up` + fold-robustness (`min_reach`/`look_smooth`). It does
**roll + aim**, not two free orientation axes.

**Goal.** (1) a loom `CameraCurve` element that emits a genuine `.ftsl camera_curve`; and
(2) ftrace-side **orientation axes** — a forward-direction curve and an up curve — with a
per-curve reference frame.

### The orientation model (locked)
Full 3-D camera rotation = 3 DOF. We author it as two independent axes; the third is derived:

- **forward** — 2 DOF (pointing direction). Authored **one** of three ways:
  - `fwd_at` **direction vector** (normalized), or
  - an **aim-point** (`look_at` fixed world point, or `look_curve` = a second spline of
    look-points) → `forward = normalize(target − eye)`, or
  - **omitted** → the path **tangent** (today's default, with `min_reach`/`look_smooth`).
  - *Direction-mode and aim-point-mode are two authoring conventions for the SAME forward
    axis — not redundant with each other, and neither is "up rotated 90°".*
- **up** — 1 DOF (roll about forward). Authored **one** of:
  - `up_at` **vector** curve (re-orthogonalized against forward), or
  - scalar `roll_at` angle about the frame's reference up (today's behavior), or
  - **omitted** → the frame's reference up.
- **right** — 0 DOF, **always derived**: `right = normalize(forward × up)`, then up is
  re-orthogonalized `up = right × forward`. Never authored.

### Reference frame (locked: **per orientation axis**, `travel | world`)
An orthogonal choice of what "reference up / straight ahead" *mean* before the `fwd_at`/`up_at`/
`roll_at` overrides apply:
- **`world`** — fixed world axes (a global up vector; today's behavior).
- **`travel`** — curve-relative **rotation-minimizing frame (RMF)** built by parallel transport
  (double-reflection / Bishop frame, **not** Frenet — no torsion flips). For a **closed** loop the
  RMF has holonomy (closure twist); **distribute the residual twist** evenly along the loop so the
  orientation returns to itself seamlessly (same technique the sweep engine's closed-spine frame
  uses, `DESIGN.md` §7a).

**This is an ftrace decision; loom mirrors it 1:1.** The orientation math lives in ftrace's
`camera_curve`; loom's bridge only emits `.ftsl` text and can express exactly what ftrace parses.
ftrace does **none** of this today (only tangent-look + world `up` + scalar `roll_at`), so we are
choosing ftrace's new behavior — not matching an existing one. **Decision: the frame is chosen
per orientation axis** (`fwd_at` and `up_at` each carry their own optional `frame travel|world`,
with a curve-level default), *not* one switch for the whole camera. Per-axis is strictly more
expressive — a single global frame is just "both axes set the same" — and it's the only way to
express e.g. *forward locked to a fixed world subject across the room while up rides the travel
frame so the shot still banks into turns*. It costs ftrace's parser one optional keyword per curve
instead of per block. loom exposes the same per-axis `frame` and emits it into each track.

### Tasks
- [x] **ftrace: `fwd_at` vector track** on `camera_curve` — parse + store a per-keyframe 3-vector
      forward direction; sample on the same `u` as position; normalize; fall back to tangent/aim
      when absent.  *(`Vec3Track fwdTrk`; sets `cs.look = eye + normalize(fwd)`.)*
- [x] **ftrace: `up_at` vector track** — parse + store a per-keyframe 3-vector up; re-orthogonalize
      against forward; fall back to reference up (`roll_at` still composes on top).  *(`Vec3Track upTrk`;
      camera `lookAt` re-orthogonalizes `u=cross(w,up)`, `v=cross(u,w)`.)*
- [x] **ftrace: per-axis `frame travel|world`** keyword — `fwd_at` and `up_at` each select world
      axes vs RMF reference independently (curve-level default); global frame = both set the same.
      *(`frame` default + `fwd_frame`/`up_frame` overrides.)*
- [x] **ftrace: RMF construction** (double-reflection parallel transport) + **closed-loop twist
      distribution** for seamless closed curves.  *(`needRMF` pre-pass builds `rmfTan/rmfUp/rmfRight`;
      closed loops measure the wrap holonomy and distribute `-ang*(i/N)`.)*
- [x] **ftrace: `right = forward × up` derivation** + up re-orthogonalization, roll composed on top.
      *(Only forward→`cs.look` and reference-up→`cs.up` are produced; camera derives right.)*
- [x] **ftrace: back-compat** — with no `fwd_at`/`up_at`/`frame` authored, behaves **bit-identically**
      to today (tangent look + world up + `roll_at`).  *(Verified 0.000% frame-identical.)*
- [x] **loom: `CameraCurve` scene element** — emit a real `camera_curve` from a `TrackedCurve`/points:
      position → `point`, speed/density track → `density_at`, roll track → `roll_at`, orientation
      tracks → `fwd_at`/`up_at`, per-axis `frame`. Mirrors ftrace's grammar 1:1 (no orientation
      semantics loom can't emit).  *(`loom.CameraCurve`, `Scene(camera=CameraCurve(...))`.)*
- [x] **Docs** — README (ftrace camera_curve grammar) + loom docstrings; update `DESIGN.md` with a
      milestone (M13) once landed.  *(FTSL.md §15.3 two-axis section, README camera_curve bullet,
      loom `DESIGN.md` M13.)*
- [x] **Tests** — loom `CameraCurve` emit golden; ftrace parse of `fwd_at`/`up_at`/`frame`; RMF +
      closed-loop seam; bit-compat when no axes authored.  *(`tests/test_emit.py` `test_camera_curve_*`;
      `scenes/_camA_travel.ftsl` closed travel-frame validation.)*

---

## B. `gyroid_nd.py` unified `--oscillate` grammar  *(origin: `tools/loom/examples/OSCILLATE_GRAMMAR.md` — "Nothing here is coded yet")*

Replaces `--transform`/`--bloom*`/`--tumble*`/`--coupling`/`--pair` with one `--oscillate`/`--lock`/
`--couple` axis grammar. Each phase is independently committable and keeps tests green.

### Phase 1 — the `--oscillate`/`--lock` core (§5 steps 1–5)
- [x] **P1.1 Parser + model, no behavior change.** `--oscillate`/`--lock` grammar →
      `Group{items:[(amp,axis)], rate, phase}`. Unit-test parser in isolation (grouping,
      amplitudes, rate/phase, reserved words `rate`/`phase`, error cases).
      *Done 2026-07-18:* `OscGroup` dataclass + `parse_oscillate`/`parse_lock_axes` +
      safe arithmetic evaluator (`pi`/`tau`/`e`, `+ - * / ** %`) in `gyroid_nd.py`; pure
      parser, not yet wired to behavior. 17 unit tests in `test_gyroid_nd.py` (grouping,
      amplitudes incl. `2*pi*x`, rate/phase either order, reserved-word/duplicate/empty/
      bad-expr errors, lock flatten+dedup). 285 loom tests green.
- [x] **P1.2 Desugar `--transform` → groups.** ✅ 2026-07-18. Added `transform_to_oscillate(...)`
      + `oscillate_spec(...)` in `gyroid_nd.py`: behavior-preserving bridge re-expressing today's
      `--transform`/`--bloom`/`--bloom-amp`/`--tumble-*` flags as one canonical composite
      `OscGroup` per §3 migration map. Pure model, execution path untouched; 13 new tests, 298
      loom tests green.
- [x] **P1.3 Wire swinger axes** (`freq`/`threshold`/`thickness`/`bloom`), `amp` = amplitude;
      replace `--bloom`/`--bloom-amp` (kept as aliases). ✅ 2026-07-18.
      - Per-axis `Variant.bloom_amps: Dict[str,float]` + `_swing_amp()`; the three swinger functions
        read the per-axis override, falling back to the shared `bloom_amp`.
      - `--oscillate`/`--lock` argparse flags + idempotent `resolve_oscillate(args)` that maps the
        parsed group model onto the canonical `transform`/`bloom`/`bloom_amps`/`tumble_*` fields
        (the exact inverse of `transform_to_oscillate`), so `pick_variant` needs no new path.
        `--transform` default → `None` for clean mutual-exclusion; conflict guards for `--transform`
        + the legacy satellite flags; `amp*tumble` → slide mode; `--lock <dims>` → tumble lock.
      - **Validated:** 14 new tests incl. field-expression equivalence to the legacy `--transform`
        invocations; a real `--oscillate bloom,freq` render through the full CLI→ftrace pipeline;
        and a byte-for-byte `.ftsl` diff (`--oscillate` ≡ legacy, incl. the `1.5*freq` amp case).
        314 loom tests green.
      - *Deferred to P1.4 (guarded with clear "not yet" errors):* per-group `rate`/`phase` (the
        shared clock / winding override) and bare spatial-dim-index axes.
- [x] **P1.4 Wire winder axes** (`drift`/`rotate`/`tumble`/bare dims), per-group `rate` (= winding) +
      `phase`; replace `--tumble-*` (keep aliases). ✅ 2026-07-18.
      - `resolve_oscillate` emits three winder-clock outputs the picker honors, all no-ops on the
        legacy `--transform` path (those variants stay bit-identical):
        - `args.osc_dim_windings` — a **bare dim index** is the atomic winder: forced on and pinned
          to an **exact** integer winding `round(amp*rate)` (`--oscillate 3 rate 2` → dim 3 winds
          twice; `2*3` is identical since `amp≡rate`). Raises the dim floor; off-lock conflict errors.
        - `args.osc_max_winding` — an explicit `rate` on a **motion** group is the **ceiling** of the
          RNG-varied `1..N` winding cycle (overrides `--max-winding`, keeps the distinct-rate spread
          — "how fast, at most", consistent with a lone dim's exact rate).
        - `args.osc_phase` — a constant radians offset (`2π` = one turn) on the shared winding clock
          (drift/rotate/tumble), from a group's `phase`; shifts the loop start, keeps `t=0==t=1`.
      - One shared winding clock ⇒ conflicting motion rates/phases across groups are rejected, as is
        `rate`/`phase` on a swinger (fixed `sin²(πt)` envelope — later step). Bare dims with no named
        motion default to `drift`.
      - **Validated:** 12 new tests + a real `--oscillate drift rate 3` video through the CLI→ftrace
        pipeline. 324 loom tests green.
- [x] **P1.5 Make `--oscillate` the documented surface** ✅ 2026-07-18. `--transform` +
      satellites (`--bloom`/`--bloom-amp`/`--tumble-*`) hidden from `--help` (`argparse.SUPPRESS`)
      but still fully supported; passing `--transform` prints a plain one-line deprecation note.
      Migrated the module docstring examples/prose + epilog to the grammar, and the test suite's
      incidental `--transform` setup usages to `--oscillate` (23 via
      `scraps/convert_transform_to_oscillate.py` + 2 hand edits; the `bloom_amps`-representation
      and deliberate desugaring/equivalence tests intentionally stay on `--transform`). Default
      motion (neither flag given) is still `drift`. 324 loom green.

### Phase 2 — `--couple` field-coupling command (§6)
- [x] **P2.1** `--couple CLUSTER CLUSTER…` (comma-joined dims, space-disjoint) with per-cluster
      `cyclic`/`full` scheme (global `--couple-scheme` default + optional `:full`/`:cyclic` tag).
      `parse_couple`/`resolve_couple` → `couple_clusters` + forced-on `couple_axes` (fed to
      `forced_on`/`max_forced_axis` like `--pair …:on`, no new RNG draws). `coupling_pairs()`
      refactored around a shared `_scheme_edges()` helper: cluster path emits ring/clique edges over
      oscillating members in CLI order; empty clusters fall through to the legacy `--coupling`/`--pair`
      base-graph path bit-identically. **Decided: kept `--coupling`/`--pair` on their own path (they
      resolve over the post-RNG active set; `--couple` names dims at parse time — no clean desugar), so
      `--couple` is mutually exclusive with a non-default `--coupling`/any `--pair`.** `coupling_desc`
      + primitive-surface warning updated; docstring/epilog/help + OSCILLATE_GRAMMAR.md §6 updated.
      11 new tests, 335 green.

### Phase 3 — surface library & per-surface params (§7)
- [x] **P3.1** Author per-surface param-metadata table `{func:[(name,desc,default,[lo,hi]),…]}`
      (extend `tools/pov_functions_gen.py` + hand fallback) + a test asserting every `POV_FUNCS`
      entry has exactly `arity−3` params.
      *Done 2026-07-18:* metadata lives in `loom/pov.py` (Python side, for `--surface-help`), not the
      C header (the VM only needs arity). `_AUTHORED_PARAMS` hand-authors real
      `(name,desc,default,(lo,hi))` for the well-documented / N-D-core shapes (f_sphere, f_ellipsoid,
      f_superellipsoid, f_paraboloid, f_quartic_paraboloid, f_rounded_box, f_torus, f_heart,
      f_noise_generator) + the 0-param helpers (f_r/f_th/f_ph/f_noise3d); every other `POV_FUNCS` entry
      falls back to honest generic `p0..` placeholders via `_generic_params`. `POV_PARAMS` is built to
      match `arity−3` by construction; `pov_params(name)` accessor returns a copy. Exported from
      `loom/__init__`. 6 new tests (completeness drift-guard, well-formedness: valid/unique axis names +
      default∈[lo,hi], spot-checks, unknown-name reject, copy-safety), 349 loom green.
- [x] **P3.2** `--list-surfaces` + `--surface-help NAME` discovery commands; main `--help` pointer.
      *Done 2026-07-18:* surface catalog in `gyroid_nd.py` (`_TPMS_CATALOG` + `POV_FUNCS`), grouped by
      N-D honesty class (`surface_group`: periodic / nd_pov / affine_pov). `--list-surfaces` prints all
      82 surfaces (4 periodic TPMS with `[nd] [loop]`, 9 N-D-generalizable POV, 69 affine-only POV) with
      each one's shape-param count; `--surface-help NAME` prints one surface's params via `pov_params`
      (axis name/meaning/default/range) or the shared-axis note for a param-free TPMS, resolving the
      `schwarz_p`→`primitive` alias. Both are early-exit (return 0 before any generation), ASCII-safe
      for the Windows console, and cross-referenced from `--surface` help + the epilog. 11 new tests,
      360 loom green.
- [x] **P3.2b** Generalize the swinger envelope to carry its own `rate`/`phase`, uniform with
      winders/bloom. *Done 2026-07-18:* each swinger's bloom is now
      `w(t) = 0.5·(1 − cos(2π·rate·t + phase))` (`_bloom_env_p`), keyed by `Variant.bloom_rates`
      / `bloom_phases` (`"dims"` for `bloom`, own name for freq/threshold/thickness). `rate`/`phase`
      are read from the swinger's group and no longer rejected. Default (rate 1 / phase 0) is
      byte-identical to the legacy fixed `sin²(πt)` envelope; integer rate loops seamlessly, a
      non-integer rate pulses faster but breaks the loop and `main()` warns. 6 new tests, 365 loom green.
- [x] **P3.3** (DONE 2026-07-18 — all slices S1–S7 shipped; see the per-slice notes below) Widen
      `--surface` to the full `iso.py` TPMS (`gyroid`/`schwarz_p`/`schwarz_d`/`neovius`)
      + `pov.py` `POV_FUNCS`, with the N-D (`POV_ND_GENERALIZABLE`) and seamless-motion (periodic-only
      `drift`) guards. Per-surface shape params become `--oscillate`/`--lock` axes.
      **Design locked (2026-07-18), building as a parallel POV emission path:**
      (1) *Solid vs shell* — POV shapes render **solid** (`f - threshold`, no abs); a small tagged set of
          genuinely-thin surfaces (klein_bottle, boy_surface, enneper, the `*_2d` curves, ...) render thin;
          a `--shell` flag forces any shape hollow. TPMS keep the abs()-shell.
      (2) *Gradient bound* — **per-function table** derived with SymPy + `mpmath.iv` (interval arithmetic)
          from the exact bodies in `src/pov_functions.h` (auto-generated exact POV ports); render-test for
          holes. Many are near-SDF (f_sphere/f_torus have |grad|~1); only the polynomial ones need work.
      (3) *N-D* — **affine remap** of x/y/z for all 78 now (extra dims only reorient via tumble/rotate);
          hand-written **true-N-D** forms for the 9 `POV_ND_GENERALIZABLE` deferred to **P3.4**. Named params
          (from `pov_params`) become `--oscillate`/`--lock` axes inheriting P3.2b rate/phase.
      (4) *Container* — per-function **bbox table** sizes bounded shapes; explicit `--radius` clips unbounded
          ones (paraboloid/cylinders/helices). POV coords are **not** freq-scaled (unit authored scale).
      (5) *Unspecified params* — default to a **random draw within the authored (lo,hi) range** per seed
          (consistent with unnamed dims), governed by `--param-default {default,random}` (shipped as S7;
          renamed from the provisional `--axis-default` to avoid the existing axis-polarity flag).
      Build order (small green slices): **(S1) done 2026-07-18** — `--surface` accepts any POV name
      (validated at runtime via `resolve_surface`: `schwarz_p` alias resolved, catalog-only
      `schwarz_d`/`neovius` + unknown names rejected). POV emits as a **solid** (`(f)-(threshold)`, no
      abs-shell) at dims=3 with authored default params, a per-function `max_gradient` from `_POV_GRAD_BOUND`
      (f_sphere/f_torus = 1.0, conservative `_POV_GRAD_DEFAULT` 8.0 otherwise), wired through `build_scene`
      via a shared `_assemble_iso_scene` helper. New Variant field `pov_values`; early POV branch in
      `field_expr` (all transforms are no-ops on a POV field for now); 13 new tests (378 loom green);
      smoke-rendered f_sphere + f_torus as clean solids. **S1 follow-up done 2026-07-18** — solid
      orientation + natural isolevel: most clamped builtins are `r = -(poly)` (positive-inside), so the
      naive `{f<0}` rendered their *exterior* (heart came out as a sphere with heart craters). Added
      `_POV_SOLID_META = {name:(sign,level)}`; emit `sign·(f − (level+threshold))` — positive-inside funcs
      negated (sign flip leaves `|∇f|`/`max_gradient` unchanged), non-zero-level funcs (f_ellipsoid, surface
      at level 1) shifted first; un-tabulated funcs fall back to honest `(+1,0)`. Validated: f_heart renders a
      solid valentine, f_ellipsoid a solid unit sphere (both were broken). 6 more tests (384 loom green).
      **(S2) done 2026-07-18 (Option B)** — tight active-band gradient bound in new `loom/pov_grad.py`,
      wired into `build_scene` via `_pov_grad_bound(name, values, box)`. A **correctness** fix, not just
      speed: POV algebraic builtins are `clamp(P0·r, ±10)` and a high-degree `r` has a huge gradient far
      from the surface — f_hunt_surface's true `|∇f|` ≈ 11000, so the old `8.0` default was a catastrophic
      under-estimate (marcher oversteps → holes). Bound `|∇f|` only over the un-railed **active band**
      (`|P0·r|<10`): rigorous (crossing ±10→0 takes ≥ `10/bound` of travel, so sphere-tracing never
      oversteps) and tight (skips the railed tails). Impl: vectorised adaptive interval branch-and-bound
      (numpy `_IV` intervals on the *factored* derivatives w/ exact even-power handling; certify a lower
      bound from band sample points, discard sub-boxes provably below it, octasect survivors, stop within
      tol; ×1.02 safety). Closed forms for SDF-like primitives (sphere/torus→1, ellipsoid→max|semi-axis|);
      returns None for noise/atan2/rotation → caller keeps default. Cross-checked vs dense numeric sample:
      rigorous + ≈1.05× tight; render-validated f_hunt_surface hole-free. 13 more tests (397 loom green).
      **(S3) done 2026-07-18** — POV container auto-sizing + `--shell`. Rather than a hand-authored 78-entry
      bbox table, `loom.pov_grad.surface_bbox(name, params, level)` *derives* each surface's natural extent
      by grid-sampling the transcribed field `f(x,y,z)` (new `_FIELD_BUILDERS` for the SDF/norm builtins
      f_sphere/f_torus/f_ellipsoid; the algebraic builtins reuse `P0·r`) and finding where it crosses the
      isolevel; returns `(half_extent, bounded)` (bounded=False when the surface runs to the search
      boundary — an unbounded paraboloid/cylinder/helix). `build_scene` now defaults `--radius` to None and
      calls `_pov_container_radius(name, values, level, radius_arg)`: explicit `--radius` wins (and *clips*
      unbounded shapes), else auto-size to the padded bbox (×1.08), else the 1.3 default (unbounded / no
      transcribed field). Fixes f_hunt_surface (surface at r≈3.67 — was a clipped disk at 1.3, now clip
      radius 3.96 / box 4.16) and f_ellipsoid's long lobes. `--shell` carves any POV shape hollow
      (`abs(sheet) − thickness`); a tagged thin set (`_POV_THIN_SURFACES`: klein_bottle/boy_surface/enneper/
      cross_cap/… + any `*_2d` curve) shells by default. TPMS keep their own abs-shell and the 1.3 default,
      untouched by `--shell`/None-radius. 20 new tests (417 loom green); render-validated f_hunt_surface
      shows its full surface (not a clipped disk).
      **(S4) done 2026-07-18** — POV shape params pinnable via `--lock NAME=VALUE`. Each POV surface has
      named shape params (`pov_params(name)` → `(axis, desc, default, (lo,hi))`, e.g. f_torus: `major`/
      `minor`; f_ellipsoid: `rx`/`ry`/`rz`); `--lock major=1.6` overrides that param's default. Rides on
      the existing `--lock` flag but stays unambiguous: the motion grammar never uses `=`, so any
      `NAME=VALUE` token is a param pin and everything else (commas, `tumble`, `spin`, …) keeps its
      motion meaning — `resolve_pov_param_locks(args)` splits the two, pins go to `args.pov_param_locks`,
      the rest stays on `args.lock` (collapsing to None if only pins were given). Space-separates multiple
      pins (`--lock "rx=2 rz=0.5"`); values are full `_osc_eval_num` expressions. Validation: pin on a
      non-POV surface, or an unknown param name, errors (SystemExit, lists valid names); out-of-range
      value warns but is honored. `pick_variant` applies pins onto `pov_default_values` before emit, so a
      pinned semi-axis both flows into the emitted `f_*` call *and* resizes the S3 auto-sized container.
      13 new tests (430 loom green). Also fixed an ordering bug found by render-validation: the pin
      extraction ran *after* `resolve_oscillate`, so `--lock minor=0.4` crashed (the motion grammar
      rejects `=`); moved it before, made the `_pv`/`_resolved_args` test helpers faithfully run
      `resolve_oscillate` (which had hidden the bug), +1 regression test (431 green).
      **(S5) done 2026-07-18** — POV shape params are now `--oscillate` swinger axes. With a POV `--surface`
      the grammar's axis set gains that surface's named params: `--surface f_torus --oscillate minor` sweeps
      the tube radius over the loop. Semantics are range-aware so amp is intuitive: `p(t) = clamp(base +
      amp*span*env(t), lo, hi)` with `span = (hi-base)` for amp≥0 else `(base-lo)`, and `env` the shared
      sin²(πt) bump — so `amp=1` reaches the param's authored range *edge* exactly at mid-loop (no plateau,
      seamless return to base), `amp<0` sweeps the other way, `|amp|>1` over-drives and clamps. Recorded in
      a new `Variant.pov_swing = {param: amp}` (kept apart from the gyroid dims-bloom swingers since it drives
      `pov_values` per frame, not the dims cross-fade), sharing the `_bloom_env_p` clock (`bloom_rates`/
      `bloom_phases`). `field_expr`/`build_scene` evaluate params via `_pov_values_at(v, t)`, so the S2
      gradient bound and the S3 auto-sized container **recompute per frame** from the swept values (an
      animated ellipsoid semi-axis grows its container as it lengthens; a torus's SDF bound stays 1). Grammar
      plumbing: `--surface` now resolves *before* `resolve_oscillate` so a param-name axis classifies against
      it; a param-only `--oscillate` names a benign `drift` (POV ignores transform) instead of erroring "no
      motion axes"; a bad axis on a POV surface hints the valid param names. 12 new tests (443 loom green);
      render-validated (torus `minor` sweeps 0.25→2.0→0.25, container 1.44→3.06→1.44).
      **(S7) done 2026-07-18** — `--param-default {default,random}` gives POV batches actual variety. A POV
      surface ignores the randomized dims/freq/harmonics (its shape is the `f_*` call args, not the N-D
      field), so a plain `-n N --surface f_torus` batch was N *identical* images. With `--param-default
      random`, every UNSPECIFIED shape param (not pinned by `--lock NAME=VALUE`, not animated by
      `--oscillate NAME`) is drawn uniformly in its authored `[lo,hi]` per variant seed, so each variant is a
      distinct shape; `default` (the flag's default) keeps the current single authored shape. The draw runs
      *last* in `pick_variant`'s RNG stream (after the hidden-offset / tumble draws) so it never perturbs the
      field's reproducibility, and it's a no-op on a TPMS (no `pov_values`) — a TPMS's shape already varies
      via its randomized freq/threshold. Explicit pins and swingers opt their param out of the draw, so
      `--lock major=1.6 --param-default random` fixes the major radius while the minor still varies. (Named
      `--param-default`, not `--axis-default`, to avoid colliding with the existing `--axis-default`
      on/off/random axis-polarity flag.) 7 new tests (450 loom green); smoke-validated (`-n 3 --surface
      f_torus --param-default random` → 3 distinct `f_torus(...)` calls; `default` → one shared default).
      **(S6) done 2026-07-18** — affine N-D remap: a POV surface's `(x,y,z)` now pass through a per-frame
      affine `M·p + b` before the `f_*` call, the honest realization of "an N-D slice of a 3-D POV field is
      an affine remap of x/y/z" (design confirmed by the user: *full affine* + *allow drift*). Rows 0/1/2 of
      `M` are the three visible slice axes' world directions, composed from the same motion layers as the
      periodic field but read as coordinate axes: **tumble** rotates the whole slice basis in N-D (a visible
      axis mixes with a hidden dim, tilting/foreshortening the shape out of the rendered 3-space and back —
      the marquee dims>3 effect), **rotate** turns each axis edge-on independently (its row scales by
      `cos α`, gaining a `hidden_offset·sin α` translation), **drift** pans each axis by `winding·t` world
      units. tumble/rotate return to identity at t=0,1 (seamless); drift is deliberately *non-seamless* for a
      non-periodic POV shape (the user opted in). New `_pov_affine(v,t,transform)` builds `(M,b)`;
      `_mat3_singular_extremes` (analytic 3×3-symmetric eigenvalues, pure stdlib) gives σ_min/σ_max so the
      render stays rigorous: the emitted field is `f(M·p+b)` whose gradient is `Mᵀ∇f`, so the S2 marcher
      bound is scaled by σ_max and the S3 container grows by `1/σ_min` (σ_min floored at 0.15 so a near-edge-
      on axis can't blow the container up unbounded; an explicit `--radius` clips instead of auto-growing).
      Gated on a new `Variant.pov_motion` (set only by a *real* explicit motion — a named drift/rotate/tumble
      or `--transform`; a pov_swing-only spec's benign filler `drift` and the default both leave it False), so
      a plain `--surface f_torus` stays the static `f(x,y,z)` (exact pre-S6 behavior). 16 new tests (466 loom
      green); render-validated (`--dims 5 --oscillate tumble --surface f_torus`: t=0 face-on torus, t=0.40
      tilted ring, t=0.25 near-edge-on sliver — all hole-free, no clipping, seamless at the loop ends).
      **P3.3 complete.** (Known refinement: a flat shape like the torus foreshortens hard mid-tumble, so the
      auto-container can jump several× within a few frames — pin `--radius` for a steadier camera.)
- [x] **P3.4** True-N-D forms for the 9 `POV_ND_GENERALIZABLE` funcs (hand-written symmetric N-D FTSL,
      bypassing the 3-coord `f_*` builtins; must match the `f_*` call at N=3). Makes the nd_pov/affine_pov
      split real. *Done 2026-07-18:* new module `loom/pov_nd.py` supplies, for each of the nine funcs,
      an honest `D`-coordinate field `F(ξ_0…ξ_{D-1})` (`nd_field_expr` FTSL emission + `nd_field_eval`
      numeric twin) that at `N=3` reduces **bit-for-bit** to the `f_*` builtin (verified to machine
      precision against direct ports of the C bodies in `src/pov_functions.h`), plus `nd_grad_bound_xi`,
      a rigorous conservative bound on `|∇_ξ F|` over the coord box (numerically confirmed to never
      under-estimate; returns `None` for `f_superellipsoid`'s non-Lipschitz corners → caller falls back
      to the per-function default). Integrated into `gyroid_nd.py`: `_pov_nd_embedding(v,t,transform)`
      builds the per-frame `D×3` slice Jacobian `A` (rest = `e_i` for the three visible dims, `0` for
      hidden dims; `c=0`) and folds hidden axes in via the same tumble Givens rotations as the affine
      path (plus rotate cos-scaling / drift pan); the emitted field is `F(A·p+c)`. Rigor mirrors S6 with
      the `D×3` Jacobian: `_matn3_singular_extremes(A)` (shares the new `_sym3_eig_extremes` helper with
      `_mat3_singular_extremes`) gives σ_min/σ_max from the `3×3` Gram `AᵀA`, so `|∇_p F| ≤ σ_max·|∇_ξ F|`
      (marcher bound) and the container grows by `(nat_rad+|c|)/max(0.15,σ_min)`. **Gated** on
      `_pov_use_nd`: `pov_motion ∧ tumble ∧ dims>3 ∧ surface∈POV_ND_GENERALIZABLE` — every other case
      (no motion, drift/rotate-only, `D≤3`, affine_pov) keeps the exact pre-P3.4 S6 path (byte-identical).
      38 new tests (504 loom green); render-validated (`--dims 5 --oscillate tumble --surface f_ellipsoid
      --lock rx=1.8 ry=0.6 rz=1.0`: t=0 face-on ellipsoid, t=0.25 the x-axis folded into a hidden dim —
      hole-free, seamless at the loop ends). **P3.4 complete.**
- [x] **P3.5** Ordered / overlapping N-D tumble via **`--tumble-sequence`** — DONE 2026-07-18.
      Implemented exactly the agreed "supersede, not alongside" single-path design. `--tumble-sequence
      i-j[xN],…` (`_parse_tumble_sequence`) parses an **ordered** word of `(i,j,winding)` Givens planes
      whose list order = composition order and whose pairs may **overlap** (share an axis); it overrides
      `--tumble-lock` and, when absent, plain `--oscillate tumble` keeps the tidy disjoint default.
      pick_variant (~1404) branches to the explicit word when given, else the existing disjoint draw. The
      one rigor change is the periodic-field Lipschitz bound: `coef *= sqrt(2)` → `coef *=
      _tumble_rownorm_factor(v)` = **sqrt(max connected-component size)** of the plane graph (union-find;
      Cauchy–Schwarz — a row draws amplitude only from its component). That **auto-returns sqrt(2) for any
      disjoint word** (each plane its own size-2 component, so the disjoint default's bound is *byte-
      identical* to the old shortcut — the waiver on tumble byte-identity was never even needed for the
      default) and grows only for overlapping words (`0-3,3-4,0-4` → component {0,3,4} → sqrt(3)). The POV
      affine (S6) and N-D (P3.4) paths already compute σ_max from the **exact** per-frame matrix via
      `_tumbled_directions`/direct plane iteration, so they honor overlapping words with **zero** changes.
      Deliverables all met: (a) `--tumble-sequence` flag + parser with full validation (axis range, self-
      pair, turn count); (b) single general construction (no legacy branch); (c) general row-norm bound;
      (d) 11 new tests (parse+validation, component-size bound incl. disjoint=sqrt2 / triangle=sqrt3 /
      chain=2, exact plane wiring, lock-override, seamless+starts-from-base, **overlap is order-dependent /
      disjoint is order-independent**, bound-never-underestimates on the composed rotation, default still
      reorients, and the N-D POV path honoring an overlapping word). **515 loom tests green** (was 504).
      Docs: OSCILLATE_GRAMMAR.md §7.y, `--tumble-sequence` help text. **P3.5 complete.**
  - **P3.5 design notes (historical, for reference — superseded by the DONE entry above):** Ordered / overlapping N-D tumble (design captured 2026-07-18; do *after* P3.3, it's
      orthogonal to the surface library). Today's `tumble` is confined to a set of **disjoint** Givens
      planes (`pick_variant` lines ~1328-1353) — i.e. a **maximal torus of SO(N)**, a commuting abelian
      subgroup where rotation order is a no-op *by construction*. Generalize to an **ordered word** of
      possibly-overlapping planes, where list order = composition order and non-commutativity yields
      genuinely richer reorientation paths the disjoint set can't reach. Key facts that make this cheap:
      (1) the evaluator `_tumbled_directions` **already composes planes sequentially in list order** — the
      restriction lives *only* in the construction, not the eval; (2) **seamlessness survives ordering** —
      each whole-turn factor returns to identity at t=1, so the product is identity at t=1 regardless of
      order/overlap; (3) the **only real cost is the Lipschitz bound**: disjoint planes cap `|rotated dir|
      <= sqrt(2)` (the current `coef *= sqrt(2)` shortcut, line ~2052), but overlapping planes can grow a
      row toward `sqrt(#coupled rows)`, so the general path must compute the true worst-case row norm.
      **Design decision (agreed 2026-07-18, revised): "supersede", not "alongside"** — the user chose the
      more elegant single path and explicitly **waived byte-identity of existing tumble renders** ("we'll
      just re-render them"). Scope of the waiver: *only* renders that use `tumble` — non-tumble seeds never
      build tumble planes, so they stay untouched. So replace the disjoint construction with ONE general
      path: tumble is an **ordered word** of `(axis_i, axis_j, winding)` Givens planes (list order =
      composition order; planes may overlap), evaluated by the existing sequential `_tumbled_directions`
      (already order-honoring), with a **single general bound** = the true worst-case visible-row norm of
      the composed rotation. That bound *auto-returns* `sqrt(2)` for a disjoint word and grows only for
      overlapping ones, so there is **no special-case code and no speed loss on the disjoint case** — the
      `coef *= sqrt(2)` shortcut is subsumed, not duplicated. No legacy/opt-in branch. **Open sub-question:**
      the seed-driven **default word** when the user gives no explicit one — keep it the current disjoint
      pairing (each visible axis <-> one hidden dim; clean, predictable, tight bound) re-expressed in the
      general framework, or make the default itself a richer overlapping draw. **Decided (2026-07-18): default
      stays disjoint-clean; the richer interacting motion is opt-in via an explicit ordered word.** Good UX
      + tight bound + guaranteed the slice tips out of the 3-space; the full-group richness is one explicit
      word away. Deliverables: (a) grammar for the ordered plane word via a new **`--tumble-sequence`** flag
      (provisional name; fits the `--tumble-mode`/`--tumble-lock`/`--tumble-amp` family) — an ordered,
      comma-separated list of axis pairs like `0-3,3-4,0-4` (each optionally carrying a turn count, e.g.
      `0-3x2`); **list order is significant** and pairs may overlap (that's what unlocks order-dependent
      motion). Plain `--oscillate tumble` with no `--tumble-sequence` keeps the tidy automatic default; (b)
      the single general
      construction; (c) the general row-norm bound (>= sqrt(2), only ever safer; `max_gradient` affects
      march step / hole-safety, never the converged image, so this is safe); (d) tests: an overlapping word
      produces motion a disjoint set can't, seamless-loop preservation (product = I at t=1 regardless of
      order), bound-never-under-estimates, and the default word still meaningfully reorients the slice.

### §8 — GPU isosurface rendering (kill per-frame tessellation; independent track)
- [x] **G1** `--raster-iso <n>` passthrough in `gyroid_nd._render_frame` → ftrace's existing
      `-raster-iso` (grid res, default 96). Zero engine changes; cuts CPU tessellation cost today.
      *Done 2026-07-18:* `--raster-iso N` CLI flag → `make_video` → `_render_frame` appends
      `-raster-iso N` on the raster path. Verified end-to-end (2-frame render at res 40 → coarse
      gyroid) and all 268 loom tests green.
- [x] **G2** GPU deterministic primary-ray isosurface **preview kernel** — per-pixel cast primary ray
      → existing `closestHit` (which sphere-traces implicits via `intersectImplicit`) + `dFieldGradient`
      shading, **no tessellation**. Wired as `-raster-gpu`; `gyroid_nd` frames route through it.
      *Done 2026-07-18:* kernel `kIsoPreview` lives in `render_cuda.cu` (where `closestHit`/`DScene`/
      `buildUpload` already are — a device twin of `raster::renderFrame`'s shading: flat per-material
      albedo, ambient + Σ weighted N·L keys + headlight fill), downloads linear-RGB + depth/emitter
      masks and calls the **shared** host `raster::exposeAndEncode` so output matches `-raster` and
      honours a camera_path's locked auto-exposure anchor. `-raster-gpu` (main.cpp) falls back to the
      CPU rasterizer when the GPU can't handle the config (no CUDA device, `-see-through`/clarity, or a
      physical mesh-lens camera). `gyroid_nd --raster-gpu` swaps the per-frame flag (`--raster-iso` moot
      — no marching cubes). Validated on `scenes/implicit.ftsl` (metaballs + CSG + torus render
      identically to `-raster`, cleaner surfaces) and a 3-frame gyroid video.
- [ ] **G3 (deferred — optimization only, not an enabler)** `PatOp::MatMulAdd` intrinsic (matrix·vec +
      offset). *Decided 2026-07-18: skip for now.* The N-D rotation loom bakes into each isosurface
      **already renders correctly** via existing scalar ops — `_arg_expr()` emits each matrix row as
      `(a)*x+(b)*y+(c)*z`, which ftrace compiles straight to `Const/VarX/Mul/Add` bytecode and evaluates
      directly (including on the GPU: `-raster-gpu` ray-marches D=8 tumble gyroids today). So MatMulAdd
      only *compresses* the encoding (one fused opcode vs ~6 scalar ops per row) — a compactness /
      marginal-speed win, **not** a new capability. Revisit only if per-frame pattern eval becomes a
      real bottleneck (it isn't — sin/cos/PovFn + the sphere-march dominate). See known-issues.md
      "Deferred: `PatOp::MatMulAdd`". Prefer the contained single-output "matrow" form (Option A) if so.
- [ ] **G4 (deferred, export-only)** GPU marching cubes — *only* to accelerate mesh export, not the
      video path. Build only if mesh-export throughput becomes a pain point.
- [x] **G5 — `-raster-gpu` / `kIsoPreview` textured shading** *(user-requested 2026-07-19; also logged
      in known-issues.md).* **DONE 2026-07-19.** Ported the CPU rasterizer's textured-preview path
      (`Texture::sampleRgb`/`sampleRgbTriplanar`) into `kIsoPreview`: a shared flattened linear-RGB texel
      array + per-texture `DPTex` meta + per-material `matTex`/`matTri` binding (mirroring `raster.h`
      buildScene's rule) upload alongside `matCol`; the kernel samples the hit `(u,v)` or world triplanar
      and replaces the flat albedo for non-emitter hits. One path covers **image** and (E1 formula-baked
      to `rgb`) **formula** skins; flat hits unchanged (`matTex==-1`). Validated vs CPU `-raster` on
      procskin/textured/triplanar (mean diff ~0.03/255) + flat implicit unchanged. Original scope:* Today the GPU isosurface preview shades **flat** — per-material albedo +
      ambient + weighted N·L keys + headlight fill, with **no texture lookup**. Make it sample and shade
      textures — both **image** (`Texture`/`skin`) and **formula** (`ProcTexture`/`func_skin`, the E1
      procedural-skin path) — so previewed isosurfaces/meshes carry their authored surface detail. Scope
      (from known-issues.md): **port the raster texture-sampling path into `kIsoPreview`** — the device
      already carries `DHit.u/v/p`, so the UVs and hit point needed for both image sampling and formula
      evaluation are in hand; wire the material's texture reference and evaluate it on-device in the
      shading step instead of using the flat material albedo. Prerequisite for **F4/F7** (the loom viewer
      showing SweptMesh/isosurface textures via `-raster-gpu`). Verify the textured GPU preview matches
      the CPU `-raster` textured output on `scenes/implicit.ftsl` + a func_skin scene, and keep the flat
      path bit-identical when a material has no texture.

---

## C. Renderer roadmap follow-ups  *(origin: `ROADMAP.md` — main items DONE; these remain)*

- [x] **C1 Mode M true final gather.** DONE 2026-07-14 (`-pmfg <K>` / `g_pmFinalGather`). At the first
      diffuse hit mode M now shoots `K` cosine-weighted hemisphere sub-rays (`photonGatherSub`,
      `photonmap_render.h`), traces one bounce each, and queries the map at *those* points, so the
      density-estimate blur lives one bounce away — the standard Jensen secondary-hemisphere final
      gather. Direct light uses low-variance NEE (`neeLight`); gather rays collect indirect/env only
      (no double-count). `K=0` keeps the original direct query (a pure superset). Validated on the
      diffuse Cornell box: final gather matches mode R in energy (diffuse-mask M/R=1.010) and is
      essentially unbiased at a large gather radius (M/R=0.994 vs. the direct query's 0.929
      corner-darkening). See `known-issues.md` "Mode M optional Jensen final gather". *Remaining
      GPU caveat (separate, lesser item): the shared GPU mode-M path still falls back to CPU when
      `-pmfg` is set — porting the final-gather sub-ray pass to CUDA is future work, tracked in
      known-issues.md.*
- [ ] **C2 VDB: native sparse device sampler.** Today the NanoVDB grid is baked to a **dense** float
      lattice for the device sampler; a native sparse GPU sampler is the follow-up.
- [ ] **C3 VDB: fp16 + emission/temperature grids** (fire) — currently float density grids only.
- [ ] **C4 VDB: native `.vdb` front-end** — validated against a downloaded official OpenVDB sample
      (only `.nvdb` is ingested today; `.vdb→.nvdb` is a manual step).
- [ ] **C5 Mesh: emissive triangles** (mesh area lights).
- [ ] **C6 Mesh: tangent-space normal maps.**
- [x] **C7 Mesh: watertight ray–triangle test** to kill grazing-edge cracks.  **DONE 2026-07-18**.
      Replaced Möller–Trumbore with the Woop/Benthin/Wald/Áfra watertight test (JCGT 2013) on BOTH the
      CPU double path (`src/geometry.h`) and the GPU float path (`src/render_cuda.cu`). Per-ray the test
      picks the dominant axis of the ray direction, permutes the other two (swapping them when the
      dominant component is negative to preserve winding), and precomputes shear constants (`TriShear` /
      `DTriShear`, built by `makeTriShear`); per-triangle it shears the relative vertices into the ray
      frame and forms the three scaled barycentric edge functions U,V,W. A hit needs the edge signs to
      agree (two-sided: all-nonneg OR all-nonpos), with an exact-zero fallback in higher precision so a
      grazing edge lands deterministically on exactly one of the two triangles sharing it — no cracks
      (background leaking through a closed mesh) and no dropped hits. The shear is **hoisted once per ray**
      at every BVH leaf loop (5 host call sites in `scene.h`, 4 device sites in `render_cuda.cu`) so the
      per-triangle cost is only the shear+edge math; an interface-preserving `intersectTri(ray, tri, …)`
      overload that builds the shear inline remains for one-off callers. The barycentric convention
      (U,V,W weight v0,v1,v2 ⇔ old w0,u,v) matches the retired M–T code, so UVs and interpolated shading
      normals are unchanged. **Validated:** `scenes/triplanar.ftsl` (16 384-tri closed torus, the shape
      whose silhouette used to crack at grazing angles) renders a clean continuous silhouette with no
      background leak on BOTH the GPU float path (where M–T's independent per-triangle edge signs cracked
      worst) and the CPU double path, with byte-identical energy (`absorbed=0.7794`).
- [x] **C8 FBX import via `ufbx`**  **DONE 2026-07-18**. Vendored the MIT / public-domain single-file
      `ufbx` (v0.23.0: `src/third_party/ufbx.{h,c}` + `ufbx-LICENSE`) and confined it to one TU
      (`src/fbx_load.cpp`, mirroring `vdbgrid.cpp`/`stb_image_impl.cpp`) behind a lightweight
      `src/fbx.h` declaration so the 220-KB header stays out of every other TU. `loadFbx` walks each
      mesh-instance node, triangulates faces with `ufbx_triangulate_face`, bakes world positions via
      ufbx's `geometry_to_world` (+ inverse-transpose for normals), then applies the mesh block's
      authored affine on top — filling the SAME `Tri` position/normal/UV slots the OBJ/glTF paths use,
      so smooth shading + texturing come free. Load opts normalize to right-handed **Y-up metres** and
      `generate_missing_normals`, so FBX lands in the engine's convention. Wired `.fbx` into `addMesh`
      **and** `addMeshAsset` extension dispatch (CMake gained `LANGUAGES … C` for `ufbx.c`). Validated:
      hand-authored `scenes/cube.fbx` → `loadFbx: … 8 verts, 12 tris`, `scenes/fbxcube.ftsl`
      render-checked. **Scope now:** baked triangle geometry + normals + first UV set. **Not yet
      consumed** (follow-ups, logged in known-issues): FBX materials, skinning/blend-shapes, animation,
      multiple UV sets, per-face materials.
- [ ] **C9 Alembic (`.abc`) import** — heavy SDK (Imath + HDF5/Ogawa); **deferred**, decide if an
      OBJ/glTF/FBX sequence suffices before taking the build weight.

---

## D. Hero-room showcase scene  *(origin: `ROADMAP_heroroom.md`)*

> **BLOCKED on user sign-off.** None of D2/D3 (the expensive verify renders) proceed until the
> user has personally verified — in the interactive rasterizer flyby-definition tool (the
> camera_curve editor) — that they *like the room and the flyby*. The look-dev of the room
> composition and the camera path is a human aesthetic decision, so don't burn photon-map renders
> on a room/flyby that hasn't been approved. Once the user says "I like it", D2/D3 are unblocked.

- [~] **D1 Flyby photon-map render** — GPU shared photon-map path (build once, gather all 144 frames),
      `-savemap gallery/hero_map.ftpmap`. (Was in progress.)
- [ ] **D2 Verify still** — raster + a real photon-mapped render frame; confirm all pieces read.
- [ ] **D3 Verify flyby** — render frames + assemble; confirm gyroid thread + glass pass + seamless loop.

---

## E. Feature ideas captured 2026-07-18  *(user-proposed; design-captured, not yet scheduled)*

### E1 — Procedural (function-defined) skin, UV-space  *(ftrace; small–medium, self-contained)*  **DONE 2026-07-18**
**Implemented (option b — three r/g/b sub-expressions baked as a texture).** A `texture "name"` block
may now give `rgb "r(u,v)" "g(u,v)" "b(u,v)"` (three quoted ftsl pattern expressions of the surface
`u,v`, constant `pi`) in place of `file`. ftrace compiles them with `compilePatternExpr` and bakes them
**once at load** to a `res`×`res` (default 512, 1–8192) **linear** RGB grid via `patternEval` over the
UV grid (matching `sampleRgb`'s `(1-v)` flip; each output clamped to `[0,1]`), then runs `buildReflCoeff`
— so the result flows through the *exact same* texture pipeline as an image skin (UV-wrap, Jakob-Hanika
spectral upsampling, triplanar, GPU, raster; `reflect texture:<name>` binds it unchanged) with **zero
`render_cuda.cu` changes** and no per-hit fit. This chose the bake-to-grid path over per-hit JH fit (far
too slow — 40-iter Gauss-Newton) and on-demand eval (no benefit for bounded UV). Fills the third square
of the skin matrix: image skins × 3-D-space procedural patterns × **UV-space procedurals**. `src/ftsl.h`
`addTexture` branches on the `rgb` statement; `scenes/procskin.ftsl` render-validated (red=u L→R,
green=v bottom→top, four blue `sin(2π4u)` stripes — all orientation checks pass). loom: `ProcTexture` /
`func_skin(name, r, g, b, …)` in `scene.py` (routed into the texture bucket so it emits before its
material), exported from `loom`, 5 new emit tests (550 loom green). Docs: FTSL.md §5.1,
docs/scene-language.md §9.1, README Textures.

**Idea.** Let a skin be defined by a *function* `f(u,v)` evaluated on demand instead of a pre-drawn
image, but applied through the **exact same UV-wrap machinery** an image skin uses — poll `f(u,v)` in
place of `image(u,v)` for each hit's interpolated UV. **Verdict: worth adding.** It's a genuine gap:
ftrace today has (a) UV-mapped *image* skins (`texture "name" { file … }`, sampled at each hit's UV —
`loom.Texture`/`skin`) and (b) *3-D-space* procedural patterns (`FuncPattern`/`SpatialExpr`, evaluated
at the surface point's world/object XYZ via `dPatternEval`). What's missing is the third square of the
matrix — a **UV-space procedural**: an arbitrary ftsl expression whose variables are the surface `u,v`
(and, cheaply, its derivatives / the hit's other channels), bound like a texture. The evaluator already
exists (`dPatternEval` runs arbitrary postfix bytecode on host+device); the work is (1) expose `u,v` as
pattern inputs when a pattern is bound in a *texture* slot, (2) a `texture "name" { expr … }` (or
`pattern uv:<name>`) grammar so a material's `reflect texture:<name>` resolves to the function instead
of a bitmap, (3) loom `FuncSkin`/`skin(expr=…)` emit, (4) tests + a render. Low risk, high reuse. Open
sub-q: also expose bump/normal-from-UV-gradient for free (the derivative is analytic on the bytecode).

### E2 — General N-D curve → scene-variable animation via the rasterizer curve editor  *(loom + ftrace; LARGE, design; extends §A)*
**Idea.** Generalize ftrace's existing interactive **camera_curve editor** (drop control points,
scrub/play, paint local speed, edit-in-place, save a real `camera_curve` block — `main.cpp` ~4473+)
from "edit a camera flyby" into "edit an **N-D curve through a grid/scatterplot** whose curve variables
can drive **any** scene variable," with **loom as the go-between** (`.ftsl` can't express animation, so
the animation binding must live in loom, which emits the per-frame `.ftsl`). **Verdict: worth capturing
as a design item; it's big and overlaps §A — schedule after §A lands.** The locked-in pieces of the
user's design:
- **Two authoring modes, chosen up front:** *pure flyby* vs. *true animation*. For most render modes the
  distinction "costs nothing." In **flyby** mode everything sampled — `curve(which, frame, dim)` or
  `grid/scatter(curve-coords, dim)` — collapses to just **camera position + orientation at time t**. In
  **animation** mode any sampled value can map to **any** scene variable (e.g. in the gyroid_nd
  isosurface example, any isosurface function parameter).
- **One exceedingly-simple binding API** (lives in the loom go-between, not `.ftsl`): *plug any curve
  variable into any scene variable* — camera position/orientation, or a surface param, etc.
- **The API has TWO distinct channels** (don't conflate them):
  - **(a) whole-video config** — the persistent authoring info: number of curve dimensions, the
    dimension↔scene-parameter *associations* (which sampled channel drives which variable), and the
    starting control points. This is authored **once for the entire animation**.
  - **(b) per-frame live values** — while the user scrubs/plays in the editor, the **rasterizer must be
    able to push the go-between the *current sampled curve values* at the scrub position** so it can
    generate/preview *that one frame*. This is a transient per-frame data flow, **separate from** (a):
    (a) decides *what maps to what* for the whole video, (b) supplies *the numbers right now* for one
    frame. The API must expose both.
- **The scene informs the editor**, through that same API, of: the curve's dimensionality, how many
  curves are tacked onto it, and the full array of **starting control points** to seed the editor with.
- **Scene proposes, editor disposes.** The scene sets the *initial* dimension count and the initial
  dimension↔scene-parameter associations, but the **editor may change them** — doing so just edits the
  original info stored in the animation definition (the persisted (a) config). So the associations aren't
  a one-way scene→editor push; they round-trip.
- **Modulable curve points are OUT for the editor.** The user resolved this: the rasterizer *already*
  owns the time dimension via curve points, so passing loom-modulable (time-varying) control points would
  introduce a *second* time axis — incoherent. So the editor receives a **static starting array** of
  control points; modulation of the points, if any, stays a loom-side concern that is *not* round-tripped
  through the editor.
- **Likely simplification (open q the user leaned toward "yes"):** there may be **no real distinction**
  between higher-D aspects of the curve itself (a 4-D curve) and extra dimensions "tacked on" (e.g. camera
  density), because the editor ignores every spatial dimension past the first three anyway — so the API
  and editing UX can treat them uniformly (one flat list of per-point dimensions).
- **Relation to §A:** §A already covers "loom emits a real `camera_curve` + ftrace orientation axes." E2
  is the strict generalization — same editor, same emit path, but the curve's sampled channels fan out to
  arbitrary scene variables, not only camera pose. Build §A first (it nails the camera/orientation case
  and the emit grammar), then E2 widens the binding target set and the editor's scene-driven seeding.
- **OPEN Q1 — where does the config (a) live: a loom in-memory data structure, or a separate animation
  definition file?** *Leaning: BOTH, at different layers — they aren't alternatives.* The **authoritative
  in-memory model is a loom data structure** (an `Animation`/`CurveDrive` object holding the dimension
  count, the channel→param bindings, and the control points). But because the editor is a **separate ftrace
  C++ process**, the config also needs a **serialized form** the editor can read to seed itself and write
  back when the user edits associations/dimensions/points — i.e. a small persisted **animation-definition
  sidecar** (JSON or an ftsl-adjacent block). loom owns the struct; the sidecar is its on-disk projection
  for the round-trip with the editor. (Note this sidecar is exactly "the animation info" that (a)-edits
  mutate, and it is *not* the `.ftsl` — the `.ftsl` stays per-frame and animation-free.)
- **OPEN Q2 — is the go-between loom, or a separate program?** *Leaning: loom.* loom is already the Python
  program that models a scene and emits per-frame `.ftsl`; it already has the curve system (`TrackedCurve`/
  `LoopCurve`/`Grid`/`Scatter`) and the scene-variable graph. A separate go-between would duplicate all of
  that. So the go-between = loom, exposing the two-channel API above (config in/out + per-frame live-value
  in → `.ftsl` out).
- **OPEN Q3 — transport for the two channels (editor C++ ↔ loom Python).** *Analysis (2026-07-18):* the two
  channels have different needs, so pick per channel:
  - **Config channel (a):** written rarely (once per edit), not latency-sensitive → the **serialized
    sidecar file** from Q1 is fine (use atomic write/rename to avoid half-read races).
  - **Live-value channel (b):** per-frame during scrub → **latency-sensitive, so NOT file-poll** (polling
    lag + disk I/O + half-written-read races). Ranking:
    1. **Anonymous stdin/stdout pipe (preferred to start).** *There is already working precedent:*
       `loom.PreviewServer` spawns a resident `ftrace -serve` child and streams it one `.ftsl` path per
       frame over stdin, reading status over stdout (`preview.py` `_build_cmd`/`show`). Anonymous stdio
       pipes are very cross-platform (subprocess stdin/stdout is identical on Windows/Linux/macOS) and
       **not** fragile in the parent-child model (coupled lifetime = child dies with parent, no ports, no
       firewall). Caveat: E2's live flow is *editor→loom* (push curve values) then *loom→ftrace* (`.ftsl`),
       i.e. more bidirectional than PreviewServer's one-way drive — doable over two pipes, slightly more
       plumbing. **Extend this channel first.**
    2. **TCP-loopback socket (`127.0.0.1`)** — reach for this *only if* E2's UX needs **decoupled,
       restartable** processes (editor restarts without killing loom) or a cleaner bidirectional protocol.
       Most portable socket option (identical Berkeley/Winsock API everywhere), decoupled lifetimes,
       reconnection; costs bind/listen/accept + port mgmt + occasional Windows firewall prompt; sub-ms
       loopback latency is negligible here.
    3. **Named pipe / Unix-domain socket — AVOID.** This is where the real cross-platform pain lives
       (`mkfifo` vs `\\.\pipe\…`; `AF_UNIX` patchy on Windows). No advantage over 1/2 for this use.
  - **Net:** live values over the existing **stdio-pipe** path (upgrade to TCP-loopback only if
    decoupled/restartable processes are wanted); config over the **sidecar file**. Decide the final wire
    format when E2 is scheduled.

### E3 — loom procedural audio: one buffer back-end, per-tick as a thin front-end  *(loom; medium; **DONE 2026-07-18**)*
**Idea / decision.** loom should be able to *generate audio files* procedurally. Two candidate output
models — (1) emit one sample value per time tick, vs. (2) random-access a sample array (`buf[t] += v`,
`=`, `*=`, …) and serialize at the end. **Decision (from `loomsound.txt`): build ONE back-end — the
random-access sample buffer as the single source of truth — and make "one sample per tick" a thin cursor
wrapper on top (`emit(v)` ≡ `buf[cursor++] += v`), NOT a second parallel pipeline.** Rationale: the
buffer model strictly subsumes streaming (it enables mixing multiple voices, overlap-add, reverb/delay
tails past a note's end, range fades, whole-file normalize-before-write, revision) — additive/subtractive
synthesis *is* the buffer model; per-tick streaming is just the buffer with a monotone write cursor and
no look-back/ahead. Two separate systems would duplicate dithering/clip/normalize/interleave/format-write
(divergent-code-path tech debt). Concrete shape: **core** = a per-channel float sample buffer (read/write/
accumulate at any index); **producers** write however they like (per-tick cursor *or* scatter-write
ranges); a single **`finalize()`** does gain/normalize/dither/clip → format-encode → write. "Per-tick" and
"whole-file" become two front-ends over one back-end. **The one genuine fork** that would force a separate
path is *real-time / unbounded* output (live to speakers, or an effectively-infinite stream you can't hold
in RAM) — then you must flush fixed-size blocks and can't revise the past; even then, share everything
below "how samples are produced" (mixer, format, dither, clip/normalize, writer). **DECIDED
2026-07-18 — OFFLINE ONLY: build just the buffer model** (no real-time/streaming path), for three
reasons the user gave: (1) loom is meant to generate **static products**, not do anything in real time;
(2) **Python is too slow** to synthesize audio in real time anyway; and (3) real-time **wouldn't even
work here** — the buffer model's whole point is that producers edit arbitrary past/future indices (mix,
overlap-add, tails, normalize), which is fundamentally incompatible with a commit-as-you-go stream. So:
one per-channel float **sample buffer** as the single source of truth; `emit_next(v)` is a thin
`buf[cursor++] += v` cursor helper for sequential generators; a single `finalize()` (gain/normalize/
dither/clip → encode → write). No second pipeline, no streaming fork. *Note: loom has no audio
subsystem today, so this is a new capability, not a refactor.*

**DONE 2026-07-18.** Implemented as `loom/audio.py` → `SampleBuffer` (exported from `loom`). One
per-channel `array('d')` back-end is the single source of truth. Random-access ops (`add`/`set`/`mul`/
`get`, out-of-range silently ignored), range ops (`add_range` overlap-add, `mul_range`, `fade` linear
ramp, `mix` another buffer with channel routing + offset). Per-tick front-end is the thin cursor
wrapper promised (`emit_next(v)` ≡ `buf[cursor] += v; cursor += 1`, plus `seek`/`tell`). Producers:
`render_fn(fn(i, t_seconds))` and `render_signal(loom Signal)` (audio-rate sampling via
`Clock.at_frame`, seamless-loop aware), each with add/set/mul modes + gain + start/count windows.
Analysis: `peak`/`rms`/`channel`. Single `finalize(path)` = gain → normalize → dither (TPDF, seeded,
default-on for 16-bit) → clip → PCM-encode → WAV (16/24-bit, stdlib `wave`). 30 tests in
`tests/test_audio.py` (round-trip WAV verify for 16/24-bit mono+stereo, dither determinism,
normalize, cursor≡add equivalence, seamless-loop signal render); 545 loom green. Smoke-validated a
real 1 s 220+660 Hz WAV.

### E4 — loom volume transforms: read and write as independent capabilities  *(loom; medium; design-captured 2026-07-18)*
**Idea / decision (user changed their mind 2026-07-18).** loom should be able to **transform volumes** —
both **sparse** (NanoVDB-style / scatter) and **dense** (regular lattice) grids. Originally the user was
wary of loom being able to *output a volume on its own* (i.e. author a grid from nothing and serialize
it), preferring only the coupled form "use an existing volume as a **basis**, transform it, then emit the
result." **Reversed:** forcing that coupling — requiring every volume *write* to be fed by a volume *read*
— is actually **more** machinery than leaving them orthogonal, so the two stay **independent, freely
composable capabilities**:
- **Read** a volume (sparse or dense) as an input field — sample it, feed it into the signal/field DAG,
  use it as a basis for a transform, drive geometry/materials from it, etc.
- **Write / output** a volume (sparse or dense) — serialize a field to a grid on disk — **without
  requiring** that field to have originated from a volume read. The source can be anything the DAG can
  produce (an isosurface function, a procedural field, an expression, a transformed read of *another*
  volume, …).
- Because reading and writing are decoupled, all four combinations are valid: read-only (sample a volume
  into the scene), write-only (bake a procedural/function field to a grid), read→transform→write
  (the "basis" workflow that motivated this), and neither.

**Transforms in scope:** the same field-domain operations the "keep everything as functions; discretize
last" principle already implies (see `loom.txt` claude-analysis) — N-D rotate-and-slice of the domain,
warps/remaps, per-voxel value ops, resampling between sparse↔dense, and modulation by other DAG signals.
Sparse and dense are two storage backings of the *same* logical field type, so a transform is authored
once against the field abstraction and the read/write ends pick the backing (a dense read can emit sparse
and vice-versa). **Open q (defer to scheduling):** on-disk formats for the write end (`.nvdb` to match
ftrace's ingest; dense raw/`.vdb`?), and whether sparse-write goes through an OpenVDB/NanoVDB dependency
or a loom-native sparse encoder.

### E5 — Axis-typed signals: one influence model (broadcast / pointwise / reduce) + mod·pin + sample·select grammar  *(loom; LARGE, design; unifies E2/E4 and records-5a)*
**Idea / decision (design-captured 2026-07-18, from a design bounce).** The whole "what can modulate what,
and does t-influencing-t break?" question collapses into **one** model: every value-producing node in the
loom signal DAG is **a function of a named set of axes** (its free variables) — e.g. a purely spatial
curve depends on `{s}` (arclength/param), a time-curve on `{t}`, an animated spatial curve on `{s,t}`, a
surface field on `{u,v}`, an N-D grid on `{a,b,c,…}`. "A influences B" = **evaluate A at the point where B
is being evaluated**, and the axis sets alone decide how:

- **Broadcast** on axes A lacks: A:`{t}` driving B:`{s,t}` contributes `A(t)`, identical for every `s`
  (⇒ "a time-curve shifts the whole elevation of a spatial curve over time"). Free, pure.
- **Pointwise** on axes A and B share: two things both depending on `t` combine at the *same* t. This is
  the "lockstep" constraint — but it is **not a rule to detect/enforce**; a function-of-t simply *cannot*
  see any t but the current one, so the illegal "run over the whole of B across time" op is
  **inexpressible**, not caught-after-the-fact. **⇒ Do NOT build a t-influences-t detector, and do NOT
  split signals into separate spatial-vs-temporal data types** (that duplicates every op, can't type the
  mixed `{s,t}` / `{u,v,t}` cases, and forbids the legal broadcast). The single axis-set-typed signal
  (the `Animatable<T>` DAG, refined so each node carries *which axes it depends on*) subsumes all of them;
  it's the tensor/shader-broadcast / Houdini-CHOPs model.

**The real (and only) expensive line — pointwise-at-P vs. cross-index-along-an-axis.** Output at eval
point P is **free/pure/streaming** iff it depends only on inputs *at P* (same `s`, same `t`). This
includes `t` (or a t-varying value) appearing inside *each point's own formula* — e.g.
`B.y(s) = f(s, some_curve(t))` reshapes the *whole* of B over time yet is still evaluated pointwise in `s`
and emits exactly **one whole spatial `.ftsl` per tick**; nothing is materialized (you pass a *scalar at
the current t*, not "the whole curve"). It also includes a spatial rotation `R(t)·p` (mixes x/y/z but at
fixed t, independent per point). The **only** cases that need materialization / caching are genuine
**cross-index** ops, where output-at-P reads inputs at *other* points along an axis:
- **Reduce over `s`** — arc length, centroid, an integral, "all of B's points at once as a set." Needs B
  materialized over all `s`. Must be an **explicit reduction node** (never smuggled in implicitly).
- **A transform mixing a spatial axis *with* `t`** — output frame t then reads input across a *range* of
  t′ ⇒ time-caching / two passes. **This is exactly the existing 4-D space-time "video node"** (`loom.txt`
  ~line 61). The test that separates it from the free case is one question: *does output-t read any t but
  the current one?* No ⇒ free (t-in-each-formula). Yes ⇒ it's the video node, pay the caching cost knowingly.

**Two orthogonal edge attributes.** A DAG edge carries `(combine-mode) × (broadcast, implied by axis sets)`:
- **combine-mode = `pin` | `mod`** — `pin` replaces (last-write-wins); `mod` accumulates toward the
  **target's identity element**, which depends on the target's quantity type: neutral **0** + `y += gain·x`
  for additive/unbounded quantities (position, elevation), neutral **1** + `y *= x` for gains/scales,
  **½-centered** `y = clamp((y−½)+gain·(x−½), 0,1)` for bipolar-[0,1] quantities. So "mod" is *one mode*
  at the authoring surface but resolves to the domain-correct operator; the edge carries `mode` + a
  **gain**, and the **target** declares its neutral/normalization (don't hardcode the ½/[0,1] assumption).
- Broadcast/pointwise is *not* an author choice — it falls out of the axis sets (above). mode and axes
  compose without interacting: axes decide *where* combining happens, mode decides *how* it combines there.

**One sample/select grammar everywhere (records, curves, grids, scatters).** A serial structure is
**sampled** with `(...)` (continuous, interpolated) and **indexed** with `[...]` (discrete constant
selector); `.name` picks a named component/channel. This is the *same* grammar records already set
(`R(driver)` sample, `R.chan[i]` stop-select, `R.chan` channel):
```
some_curve(t)          # sample the curve at parameter t (interpolated between control points)
some_curve(t).y        # …take its y component
some_curve.y(t)        # component-first spelling of the same
some_curve.dim[3](t)   # dim 3 as a discrete channel pick, then sampled at t
```
Deliberately **avoid `some_curve[t]`** for the temporal index — brackets already mean "pick a fixed
discrete stop" in records, so `[t]` would overload them; `(t)` reads as "sample here, interpolate," which
is the intended semantics. Because `some_curve(t)` yields a scalar/fixed-vector *at the current t*, it
broadcasts across the target's other axes ⇒ lands on the free side by construction.

**The unifying one-liner (shared with records-5a's free-variable scope check).** *Everything that produces
a value declares the axes it depends on. Composition broadcasts on unshared axes and combines (pin/mod)
pointwise on shared ones. Crossing an axis you don't own requires an explicit reduction (over `s`) or is
the cached space-time video node (over `t`).* Records-5a is the same mechanism seen at a value site: a
driver's free variables must be ⊆ the axes in scope there (`R(u)` errors in a light SPD because `u` isn't
in that site's axis set). **Open q (defer to scheduling):** the concrete `Animatable<T>` node taxonomy and
how axis-set inference/annotation is represented in the loom struct + the on-disk projection; where the
explicit reduction node and the video node sit in that taxonomy.

### E6 — Quick mesh viewer: open a bare mesh in a ready-lit scene  ✅ DONE 2026-07-21  *(ftrace; user-proposed 2026-07-19)*
**Shipped.** A bare positional mesh path — `ftrace model.glb` (also `.obj`/`.gltf`/`.fbx`/`.stl`/`.ply`) —
now wraps the mesh in a synthesized auto-lit FTSL scene (neutral clay fallback material under a soft
uniform `light env`; glTF/GLB keep their imported materials) with an **auto-framed camera** (aimed at the
mesh bounding sphere from a 3/4 front-high angle, pulled back to fit the vertical FOV) and opens the
interactive raster-preview live window — the same viewer as a double-clicked `.ftsl`. Presentation flags
(`-window`/`-o`/`-r`/`-camera`/`-view`) keep it a preview; a transport flag (`-mode`/`-n`/…) renders the
same auto-lit scene for real. A bare file-like positional that isn't a recognized scene/mesh is now a hard
error instead of silently rendering the demo. Implementation: `ftsl::loadSource(src,…)` (a string-source
refactor of `ftsl::load`, `src/ftsl.h`) + positional mesh detection & scene synthesis in `src/main.cpp`.
Deviations from the sketch below: entry point is the **bare positional** (not a `-view`/`-obj` flag, since
`-view` already takes eye/look numbers), the default is the **raster preview** (not a full light-transport
render), and lighting is a single soft **environment** (not a three-point rig) — sufficient for "just let
me look at this mesh," and a real render is one transport flag away.

**Original idea.** A command-line option that takes an `.obj` file and renders/opens it directly in a viewable
scene with **sufficient built-in lighting** — no hand-authored `.ftsl` required. Point ftrace at a bare
mesh (e.g. `ftrace -view foo.obj` / `-obj foo.obj`) and get an immediately watchable, well-lit result:
auto-frame the camera to the mesh bounds, drop in a default key/fill/ambient (or an environment light) so
the surface reads clearly, apply a neutral default material, and bring up the live `-window` preview.
Purpose: a fast "just let me look at this mesh" path for inspecting exported/staged OBJs (the settle
pipeline's `scraps/*_staged.obj`, decimation results, etc.) without writing a scene each time. **Scope
sketch:** (1) a CLI flag that loads the OBJ as the sole geometry; (2) auto-fit camera from the mesh AABB
(distance/fov to frame it, sensible default orientation); (3) a canned three-point or env-map lighting
rig + default material; (4) reuse the normal render/preview path (`-window`/`-keepwindow`, checkpoints).
**Accepts every mesh format ftrace already loads,** not just OBJ — the loader dispatches on extension in
`src/ftsl.h`, so the viewer flag should transparently open **Wavefront OBJ (`.obj`)**, **glTF (`.gltf` /
`.glb`)** (via the glTF loader, which also imports pbrMetallicRoughness materials), and **FBX (`.fbx`)**
(via the ufbx bridge). Reuse the same extension dispatch so any file the scene parser can `mesh { file … }`
can also be `-view`ed directly. (glTF/FBX carry their own materials — for those, prefer the file's baked
material over the neutral default; fall back to the default only for OBJ or when a mesh has no material.)
Open q: default material for materialless meshes (matte grey vs a light clay/AO look), and lighting preset
(studio three-point vs a built-in HDRI/env).

---

## F. loom native viewer  *(loom + ftrace; LARGE, design-captured 2026-07-19; the concrete realization of §E2)*
**Idea.** A **native viewer** for loom scenes, built on ftrace's own renderer (C++), not a WebGL/browser
app. It enumerates the objects a loom file defines, lets the user select and inspect them, and shows
N-D curves / SweptMeshes / isosurfaces / scatter+grid fields / the modulator DAG in one live UI.

**Decision — native, not WebGPU/WebGL (revisited 2026-07-19, reaffirmed).** Considered replacing our
rasterizer/preview with WebGPU (browser or wgpu-native). **Verdict: build §F native.** Decisive reasons:
(1) a browser can't call CUDA, so a web viewer must either *reimplement* the isosurface sphere-trace in
WGSL — a permanent second field evaluator, parity tax vs. the C++ render, FP32-only, recompile-on-
*structural*-edit — or degrade to pixel-streaming from the C++ renderer; both are strictly worse for the
*interactive isosurface-modification* goal, whereas native links `-raster-gpu` in-process and blits its GPU
output straight to the viewport. (2) §F's UI (sliders, scroll-locked strip charts, node DAG, click-to-
inspect viewport overlays) is exactly ImGui/ImPlot/imnodes's sweet spot, so web's richer-GUI edge doesn't
apply. (3) The Python↔viewer bridge is needed either way (loom is Python); web only *adds* a second
boundary (browser↔renderer). (4) Mature CUDA↔graphics interop exists for the native ImGui backends
(D3D11/OpenGL/Vulkan) but not for WebGPU, so keeping the trace on-GPU and zero-copy presenting it is clean
native, awkward via WebGPU. Accepted native costs: slower UI iteration than web hot-reload; ImGui/window/
interop build plumbing; a native exe not a URL. **Revisit WebGPU only if the goal changes to zero-install
browser sharing or getting off CUDA — and even then as an additive share/embed front-end, never a
replacement for the renderer or the primary editing tool.**

**Architecture (locked in conversation 2026-07-19):**
- **Native, on ftrace's renderer — not WebGL.** The user's primary interest is *fast isosurface
  modification*, and ftrace's **`-raster-gpu` / `kIsoPreview`** already sphere-traces implicit
  isosurfaces per-pixel on device with **no tessellation** and **exact field-eval parity** with the full
  render. WebGL would need a GLSL transpiler for every field, shader recompiles on structural field edits,
  and would still not beat the CUDA sphere-tracer for *re-evaluating a changed isosurface*. So: native
  viewer = ftrace's `-raster-gpu` for the 3-D pane + **Dear ImGui** (panels/sliders/input/picking) +
  **ImPlot** (scroll-locked strip charts) + **imnodes** (the modulator DAG). Disadvantages accepted:
  more C++/CUDA plumbing than a browser app, must ship/build the ImGui stack, no zero-install share link.
- **loom→viewer data flow: export/sidecar + the existing `.ftsl`/`-serve` path** (loom is Python, the
  viewer is C++, so no in-process sharing). loom writes a JSON sidecar of curves/fields/DAG for the
  viewer to introspect, and drives frames over the existing `PreviewServer`↔`ftrace -serve` stdio pipe
  (same precedent as §E2 Q3). Anything WebGL could do without a Python round-trip, the native viewer can
  do too.
- **Load contract: a `build(clock=…, **params) -> Scene` function, NOT a module-level `scene` object.**
  `build()` lets the viewer re-evaluate the scene live (re-tessellate, change a param, scrub time) with
  no import-time side effects. The viewer calls `build()` with the current clock/params to get a fresh
  `Scene` whenever it needs to re-derive geometry.

**Tasks:**
- [ ] **F1 — scene/object enumeration + `build()` load contract.** Load a loom file, call its
      `build(clock, **params)`, walk the resulting `Scene`, and present a selectable list of objects
      (curves, SweptMeshes, isosurfaces, scatter/grid fields). Selecting one drives the panes below.
      Define + document the `build()` contract (signature, that it must be side-effect-free at import,
      how params are surfaced to the UI).
- [ ] **F2 — N-D curve 3-D view.** Show an N-D curve by picking **3 of N** dims to display. **Rotating
      the displayed dims = a view-only transform** (no recompute); **rotating into other dims = recompute
      the projection.** Index markers along the curve show curve progression. **Stereoscopic viewing:**
      wall-eyed (L|R) and cross-eyed (R|L) side-by-side, plus **red-cyan anaglyph** — using the §I
      off-axis stereo machinery (shared with the renderer's still/movie stereo).
- [ ] **F3 — scroll-locked strip charts (ImPlot).** Below the 3-D pane, one strip chart **per curve
      dimension** and one **per tacked-on channel** (TrackedCurve). Shared index markers along the bottom
      cross-reference the 3-D index dot. **All charts scroll left/right together (scroll-locked, never
      individually)** to page through the whole curve. Hover/click on a chart cross-highlights the 3-D
      index dot and vice-versa.
- [ ] **F4 — SweptMesh tessellated view + textures + decoupled re-tessellation.** Tessellate the
      SweptMesh and show it in the 3-D pane with **any texture it defines (image *or* formula** — needs
      **G5**). **Rotation rule:** rotating an isometry of the 3 displayed spatial dims = **view-only**
      transform (just spin the existing mesh, no re-tessellate); rotating **into a parameter/extra
      dimension** = **re-tessellate off the UI thread** via a **latest-wins job queue** (drop stale bakes
      during a slider drag so the UI stays responsive). If loom couldn't define textures this would add
      it — but loom already has `Texture`/`skin` (image) and `ProcTexture`/`func_skin` (formula, E1 DONE),
      so this consumes them.
- [ ] **F5 — modulator-DAG panel (imnodes).** Introspect the signal DAG via loom's `walk()` and lay it
      out well. Each node shows the **op/function that modulates it** and a **stable identifier**; each
      **edge is labeled with the parameter name it feeds**, so you can tell which variable in a node's
      function refers to which upstream node.
- [ ] **F6 — scatter + grid field display & inspection.** **Scatter:** show the actual defined points
      (no volume fill) using the same 3-D-view/stereo mechanism, colored by a **channel selector
      (default)** or channels 0/1/2 → RGB; **click any point to inspect its location + all channel
      values**; glyphs later. **Grid:** show a **3-D slice** with **sliders for the extra dims**; same
      click-to-inspect. Coloring must handle **multi-valued** points (hence the channel selector, not a
      single fixed mapping).
- [ ] **F7 — isosurfaces via `-raster-gpu` raymarch (primary) + MC-mesh fallback.** Show a loom
      isosurface in the 3-D pane by **raymarching it through `-raster-gpu`** (primary path — the whole
      reason for the native viewer; lets the user *modify* the isosurface and see it re-evaluated fast
      with no re-tessellation). Keep the existing marching-cubes mesh (`mcubes.mesh_field` / `IsoMesh`)
      only as an **optional static-rotate fallback**. Textures via **G5**.

---

## H. loom multi-valued fields + interpolation + field-sampled curve  *(loom; medium–large, design-captured 2026-07-19)*
**Idea.** Make loom's `GridField` / `ScatterField` **multi-valued** and add a **curve that samples a
field** (the piece §F6's inspection and §E2's curve-drive both want). Answers the user's explicit
question — *should grid/scatter points be multi-valued?* — with **YES**.

**Decisions (locked 2026-07-19):**
- **Fields are vector-valued / named-channel**, mirroring the FTSL `record` channel model (§0). Keep
  **domain coordinates** (where a point sits) distinct from **value channels** (what it carries). Why
  multi-valued rather than "make a separate field per value": (1) **interpolation-weight correctness** —
  the weights depend only on the domain coords, so all channels share one weight computation; (2) **perf**
  — compute the neighbor/RBF weights **once**, apply to every channel (a multi-RHS solve for scatter);
  (3) **consistency** with records' channel model; (4) it **subsumes** the single-valued case (one
  channel). A single-valued field is just the 1-channel special case.
- **Grid interpolation:** **multilinear by default**, optional **tricubic / Catmull-Rom**.
- **Scatter interpolation: RBF** (radial basis functions) — works in any N-D, is **exact at the data
  points**, smooth, and needs **no meshing**. **Default kernel = thin-plate / polyharmonic spline**
  (parameter-free); options **multiquadric** / **Gaussian** (shape param ε) and **Wendland** compactly-
  supported for large point sets. Use `scipy.interpolate.RBFInterpolator` (its **multi-RHS solve** does
  one kernel factorization for *all* channels — reinforcing the multi-valued perf win). **Explicitly not
  simple linear interpolation between points.** **Caveat:** RBF **extrapolates/overshoots outside the
  convex hull** → the field must **clamp or flag** out-of-hull queries.
- **Field-sampled curve:** a loom **curve routed through a grid/scatter field**. You *use* it by **polling
  at a curve-progression index**, which returns **N spatial coordinates** *and* **`{channel: value}`** —
  the interpolated field value(s) at those coordinates. This is the object §E2's "curve variables drive
  scene variables" and §F6's inspection both build on.

**Tasks:**
- [x] **H1 — vector-valued `GridField`/`ScatterField`.** ✅ 2026-07-19. `Grid`/`Scatter` now infer a
      value model (`value_dim`/`is_vector`) and take optional `channels=` names (validated, resolved by
      `channel_index(name|idx)`). New `VecGridField`/`VecScatterField` (VecSignals) compute the shared
      **domain weights once** and blend every channel with them (weight kernels `_grid_weights` /
      `_shepard_weights` are shared with the scalar fields, so scalar == vector-channel bit-for-bit).
      Scalar `GridField`/`ScatterField` stay the 1-channel case and now reject vector datasets with a
      pointer to the Vec* class. 11 new tests; 573 loom tests green.
- [x] **H2 — grid interpolation.** ✅ 2026-07-19. `GridField`/`VecGridField` take `interp="linear"`
      (default, N-linear) or `interp="cubic"` (separable **Catmull-Rom** / tricubic). `_grid_weights`
      gained a `cubic` flag; cubic is a tensor product of per-axis `_catmull_rom_axis` contributions, so
      the vector field still computes the taps **once** and blends every channel with them. Boundary
      phantoms are **linearly extrapolated** (not edge-clamped) so cubic reproduces linear ramps exactly
      to the edge; axes with < 3 samples fall back to linear. 8 new tests (`tests/test_gridinterp.py`).
- [x] **H3 — RBF scatter interpolation.** ✅ 2026-07-19. `RbfScatterField` / `VecRbfScatterField` wrap
      `scipy.interpolate.RBFInterpolator` (lazy import; scipy is an optional dep). Default kernel =
      parameter-free **thin_plate_spline**; scipy's `multiquadric`/`gaussian`/`inverse_*`/`cubic`/`quintic`
      /`linear` also selectable (ε-kernels take `epsilon`). A vector scatter is **one** interpolator with a
      multi-column RHS → one factorization across all channels (`.channel(name|idx)` views). Convex-hull
      extrapolation is guarded by `on_outside="clamp"` (default; clips to per-channel data range),
      `"raise"`, or `"extrapolate"` — **no `"nan"` flag** because loom's Signal contract forbids non-finite
      values. Rebuilt at most once per frame (`_RbfEngine`). **Not offered:** Wendland (scipy's
      RBFInterpolator has no compact-support kernel) — use `neighbors=` for large sets instead. 8 new
      tests (`tests/test_rbf.py`, skip if scipy absent).
- [x] **H4 — field-sampled curve.** ✅ 2026-07-19. `FieldCurve(curve, field_builder, u)` routes a loom
      curve through any field. `field_builder` is a callable `q -> field node` (e.g.
      `lambda q: VecGridField(grid, q, interp="cubic")`), so it composes with H1–H3 freely. `.position`
      (the spatial coords `VecSignal`), `.value` (the sampled field), and `.channel(name|idx)` are real
      DAG nodes that can drive scene variables; `.sample(u, clock)` polls at an explicit progression index
      returning `(coords, {channel: value})` (channel keys are dataset names if present, else indices).
      Explicit polling uses a private probe field over a mutable query so it doesn't disturb the bound
      DAG. 8 new tests (`tests/test_fieldcurve.py`). **§H complete.**

---

## I. ftrace stereoscopic / anaglyph output  *(ftrace renderer; medium, design-captured 2026-07-19)*
**Idea.** 3-D stereoscopic output for **both stills and movies** — side-by-side (wall-eyed and
cross-eyed) and **red-cyan anaglyph** glasses. Also the shared machinery §F2 uses for the viewer's curve
stereo.

**Decisions (locked 2026-07-19):**
- **Off-axis (asymmetric-frustum, parallel cameras) stereo — NOT toe-in.** Toe-in (rotating the two eyes
  to converge) introduces **vertical parallax** that causes eye strain; the correct method is two
  **parallel** cameras with **asymmetric (sheared) frusta** sharing a convergence plane.
- **Anaglyph default = Dubois matrix** (least-squares optimal color mixing — far less ghosting/retinal
  rivalry than naïve channel-split). **Red-cyan default**, **green-magenta** an option.
- **Physically-correct baseline from viewing geometry.** CLI supplies **viewing distance**, **interocular
  distance** (both with sensible defaults), and **DPI** (default: attempt auto-detect); from these compute
  the correct stereo **baseline + convergence**. Reuse the **M13-derived camera right axis** as the
  interocular baseline direction.

**Tasks:**
- [x] **I1 — off-axis stereo core.** DONE. `Camera::frustumShiftX` (normalised off-axis shear) added and
      applied consistently in `project()`/`genRay()`/`lensImage()` (rectilinear only) on both the CPU and the
      GPU `DCamera` (photon splat, backward genRay, BDPT camera subpath). `-stereo` expands each rendered
      camera into a Left/Right eye pair: two PARALLEL rectilinear cameras offset ±baseline/2 along the M13
      right axis `u`, each with a sheared frustum so the convergence plane has zero parallax (no toe-in ⇒ no
      vertical parallax). The pair shares one exposure group so both eyes — and, for an exposure-locked
      camera_path, every frame — tone-map identically. Works for stills and per-frame movies, reusing the
      whole render pipeline per eye (checkpoints/budgets/GPU/live window unchanged).
- [x] **I2 — output modes.** DONE. Post-render `stereoComposite()` fuses each eye pair's PNGs into the `-o`
      image: **side-by-side wall-eyed (L\|R)** (`sbs`), **cross-eyed (R\|L)** (`cross`), and **anaglyph** via
      the **Dubois least-squares matrices** — **red-cyan** default (`anaglyph`), **green-magenta** option
      (`anaglyph-gm`). Intermediate per-eye files are deleted afterwards (kept with `-stereo-keep-eyes`).
- [x] **I3 — CLI + physical geometry.** DONE. `-stereo <mode>`, `-eye-sep <m>` (interocular, default 0.063),
      `-view-dist <m>` (default 0.6), `-dpi <n|auto>`, `-convergence <m>` (scene units; default = look-at
      target distance). Baseline/convergence are physical: screen width `W` comes from a measured `-dpi`
      (`W = resX·0.0254/dpi`) or, by default, from the viewing distance × FOV (`W = 2·d·tanHalfX`); the
      frustum shear `S = eyeSep/W` puts infinity at exactly interocular separation on screen (parallel gaze),
      and the baseline `b = 2·C·tanHalfX·S` so that `b/C = eyeSep/W` (camera-to-subject ratio = eye-to-screen
      ratio). `-dpi auto` reports the Windows *logical* DPI as a rough hint. Documented in README. **§I complete.**

---

## J. loom field/scene enhancements  *(loom; design-captured 2026-07-19 — this conversation)*

Three related decisions made while discussing "a curve through a field going out of bounds" and
"multiple changing isosurfaces in one scene". None started yet; J1 is small and unambiguous, J2 is
medium, J3 is large (and is the item the user actually cares most about — a loom twin of the FTSL
record so loom can round-trip `.ftsl` scenes).

### J1 — Grid field out-of-domain policy (small, back-compatible)
**Problem.** A `FieldCurve` routed through a `Grid` that wanders outside the lattice **silently
edge-extends** (`interp.py:_cell_base_frac` clamps `p<=0→0`, `p>=n-1→n-2`). That is inconsistent with
the RBF scatter field, which already has a first-class `on_outside` policy
(`"clamp"`/`"raise"`/`"extrapolate"`, interp.py ~592). Only the **Grid** field has this gap — analytic
fields (gyroid expr / `SpatialExpr`) are defined everywhere; Scatter/RBF already guard the hull.
Dimension mismatch is **already** a hard construction-time error (`GridField`/`VecGridField.__init__`
raise `query dim != grid ndim`, and `FieldCurve` builds the field eagerly), so no silent failure there.

**Decision.** Give `GridField`/`VecGridField` (and the shared `_cell_base_frac`/`_grid_weights`) the same
`on_outside` policy the RBF field has. **DONE 2026-07-19** (`tools/loom/loom/interp.py`; 6 new tests in
`test_gridinterp.py` + a FieldCurve dim-mismatch test; 604 loom tests green):
- [x] **`"clamp"`** — current edge-extend; **default**, so behavior stays byte-identical (verified against
      the old default path).
- [x] **`"raise"`** — error on leaving the domain (boundary inclusive), the guard for "grid view" authoring
      where the curve must stay inside the box.
- [x] **`"wrap"`** — periodic fold (period `hi-lo`; sample `n-1` aliases `0`), for both linear and
      Catmull-Rom stencils. Apt for a gyroid (2π-periodic) → seamless tour.
- [ ] optional **`"extrapolate"`** — linear extrapolation off the boundary cell. **Deferred** (not needed
      yet; the cubic phantom-point machinery already exists if we want it).
- [x] Re-raise the `FieldCurve` dimension-mismatch `ValueError` with FieldCurve context (names the curve's
      dim). *(J1 complete bar the optional extrapolate mode.)*

### J2 — Placed isosurfaces + a Room/Group element (multi-changing-isosurface pipeline)
**Problem.** `Isosurface` (`iso.py`) has **no position**: its frame is `freq*(row·xyz)+drift`, with no
`- center`. So a gyroid clipped to a container at (5,0,0) shows the *same phase* as one at the origin —
the container moves but the pattern does not follow. (A translation is *expressible* today by hand-folding
`drift' = drift − freq*(M·c)`, but the `contained_by` box/sphere won't track it — manual and error-prone.)
loom already has the affine machinery to fix this cleanly: `mathnd.Affine` (`linear @ x + offset`, both
animatable, composable) and `spatial._offset`.

**Decision (the two missing primitives):**
- [ ] **Placement on `Isosurface`** — a `center`/`Affine` that offsets **both** the coordinate frame
      (`freq*(M·(x − center))+drift`) **and** the `contained_by` box/sphere, animatable, so a blob can
      drift/tumble around the room over the loop.
- [ ] **`Room`/`Group` `Element`** — owns a child list + an animatable `Affine` frame, emits each child with
      the composed placement (`room_frame ∘ child_placement`), namespaces child names (`room/gyroidA`) so
      the emitted `isosurface "…"` names don't collide; may emit the shell (box / 6 planes) + shared
      materials/lights.
- [ ] **Driver pattern / factory** — refactor `gyroid_nd.py` to expose a factory (`make_gyroid(**params) ->
      Isosurface`); a new driver script builds a `Room`, instances the factory N times with different
      params/materials, and assigns each a placement (static, on a closed `LoopCurve`, or from a
      `VecGridField`/`Scatter` of placements).
- **Caveats to design in:** seamless loop (translations on *closed* curves, rotations by integer turns,
  drift by 2π·k); overlap (separate `contained_by` boxes keep blobs disjoint & cheap; union/blend is a
  CSG question ftrace-side); validate ftrace stays efficient with many overlapping sphere-traced
  isosurface containers.

### J3 — Port the FTSL **parametric-record** data structure into loom (large; the user's real ask)
**Clarified intent (2026-07-19).** *Not* loom's existing `Grid`/`Scatter`. The user wants a **loom twin of
the §0 FTSL record** (`ROADMAP_records.md`): one data type that **names its output channels** after real
destination slots, so a single record bundles **one interpolated curve per property** (e.g. every slot of a
material), each named for export. Structurally the record is **(driver domain) × (named-channel axis)** — a
bank of per-channel curves over a shared scalar driver; "each property is a named curve" is exactly what §0
already defines (my earlier "1-D" referred only to the *driver input* being one scalar). Stops sit at
**defined positions** (`p:`-pinned, else evenly redistributed), are **interpolated**
(`nearest|linear|smooth`), and may be **expressions**. Goal: a loom program can **read, represent, and
re-emit `.ftsl` scenes** (copy an existing `.ftsl`).

**Locked decisions (2026-07-19):**
- **ftrace's record grammar is UNCHANGED** — one scalar driver × N named channels. No N-D *input* domain in
  ftrace (the user sees no need). The "at least 2-D" the record needs = domain × named-channel, which it
  already is.
- **loom may be a superset** — loom MAY offer a genuine N-D *input* domain for its own authoring, but that
  superset stays loom-side (emits down to constructs ftrace already understands, or is loom-only). It does
  NOT push back into FTSL.
- **Round-trip = semantic re-emit**, not byte-faithful: parse `.ftsl` → loom `Element` tree → re-emit in
  loom's canonical style (equivalent scene, not identical formatting/ordering).
- **Per-channel output arity is fully general (spec updated 2026-07-19).** A record already mixes per-row
  output dimensionality — a `scalar` roughness curve (arity 1) sits beside an `rgb`/`spectrum` curve
  (arity 3) in the same record. So the record model is: *driver domain × named channels, where each channel
  outputs an arbitrary-arity `D`-tuple and interpolation runs per-component.* ftrace's `Scalar`/`Spectrum`
  kinds are just the `D=1` (Linear) and `D=3` (Colour) instances; `ROADMAP_records.md` §3 now states the
  general form and §5 clarifies this output-arity generality is **not** deferred (only the N-D *input* domain
  is). This maps cleanly onto loom, whose `Signal`/`VecSignal` + `Grid`/`Scatter` already carry
  scalar-or-vector-of-any-dim values — so the loom record twin (J3a) is arbitrary-arity by construction.

- [x] **J3a — loom record type mirroring the FTSL record.** **DONE.** `loom/record.py` — `Record`
      (`Element`) + `RecordChannel` + `RecordStop`. Named channels (scalar `D=1` numeric/expression stops
      or colour `D=3` `spectrum:`/`metal:`/`rgb:` refs; homogeneity enforced), positioned (`p:`-pinned)
      stops with even redistribution (`_redistribute` ports ftrace `redistributeStops`), per-record
      `interp nearest|linear|smooth`. **Emit** the `NAME = range LO-HI [ … ]` block (routed through `Scene`
      before materials); **`Record.parse`/`parse_all`** read one/every record block back out of `.ftsl`
      text (comment-stripped). **`Record.sample(channel, d)`** numeric sampler mirrors `recSampleScalar`
      (Fritsch–Carlson monotone cubic for `smooth`) for all-numeric scalar channels; colour/expression
      stops are re-emitted faithfully but not evaluated (needs the pattern VM → J3c). Round-trips every
      `scenes/_record_*.ftsl` fixture. 23 tests (`tests/test_record.py`); 627 loom green. (ftrace only
      materializes `D∈{1,3}`; non-{1,3} arities are the loom superset / J3b.)
- [ ] **J3b — loom N-D / generalized-grammar superset** (loom-only authoring). Four related generalizations
      beyond the current-FTSL mirror J3a implements:
      1. **Arbitrary channel arity** — a `D`-tuple-valued channel, not just scalar `D=1` / colour `D=3`.
         *(DONE — vector channels + inline `rgb`/`hsv`/`hsl` colour channels with a channel-level tag, plus
         `Record.lower_colours()`/`lower_ftsl()` lowering inline colour to synthesized `spectrum "<name>" = rgb …`
         decls + `spectrum:<name>` refs ftrace can parse.)*
      2. **Generalized stop grammar** (`ROADMAP_records.md` §3.1) — arbitrary-arity stops with a **delimiter
         precedence ladder** (whitespace binds like `×`, comma like `+`, brackets = parens), so structure is
         recoverable from the delimiters alone and the channel's arity only *validates*: `tint [rgb 0 0 0,
         0 1 0, 1 1 1]` ≡ `tint rgb [0 0 0] [0 1 0] [1 1 1]` (the three ladder delimiters are `[ ]` / `,` /
         whitespace — parens are reserved for expressions + the §3.2 application surface); position pins
         (`.2:0 0 0`) are an orthogonal `POS:` prefix. **NB: current FTSL cannot parse this** — its tokenizer isn't comma-aware and every
         whitespace-word is a separate stop, so today an rgb curve is `reflect spectrum:steel spectrum:gold …`
         (one `:`-ref per stop). loom now implements this as **one backward-compatible grammar** (`loom/record.py`,
         a single `parse`/`emit` pair): each channel line is dispatched on the presence of a top-level comma, so a
         comma-free line keeps the exact J3a whitespace meaning and only a comma line opts into the ladder
         (`tint 0 0 0, 1 1 1` = two arity-3 stops; a lone vector stop takes a trailing comma). It's an *additive
         superset*, not a breaking change — no existing record reparses differently.
      3. **Uniform named-input binding / rebinding** (`ROADMAP_records.md` §3.2) — a property is an expression
         over named inputs (system-provided-with-default like `a`/`u`/`v`, or unbound). Access is *continuous
         only* (no discrete `[i]` — a constant index is just a constant argument `prop(2)`); any input is
         rebindable at the use site (`gold.color(u=x)` ≡ `gold.reflect(a=x)`); `[…](u)` seals the array inside
         a function of `u` (reachable via `u=x`) whereas bare `[…]` leaves the driver for the consumer. loom
         authors this surface; shipped ftrace keeps the two constant accessors + fixed scope model.
         Also (`ROADMAP_records.md` §3.3): **materials are parameterized bundles** — a material's free-input
         set is the union of its properties' inputs, and applying it binds them across the bundle at the use
         site (`material = gold(u=v, a=1)` ≡ `gold(u=v a=1)`, ladder-separated; partial application falls back
         to system defaults; positional `gold(v)` only when there's a single free input). And **property names
         are optional** — the leading type/slot keyword identifies the property; the quoted name is only an
         external dot-handle (`spectrum = …` anonymous vs `spectrum "color" = …` for `gold.color`).
         **SETTLED SCOPE (2026-07-19):** item 3 lives entirely on the existing `SpatialExpr` tier
         (`loom/spatial.py`), not a new expression system — no VM, no `Clock.env`, no lowering to shipped
         ftrace. Concretely: (a) rename/re-expose `_Coord` as a public **`Surface`** leaf family with bare
         singletons `X Y Z` (existing), `U V A` (**new, emit-only** — no `eval_np` numpy twin, exactly as noise
         is already emit-only; ftrace evaluates `u/v/a` at each ray hit, which is precisely ftrace's
         "surface = function of u/v/a" mode, so this is bit-for-bit the same feature, not an approximation),
         and `T` (existing loop-phase); (b) add an **`Image("path")` leaf** — image-as-a-term-inside-a-function
         (multiply a procedural `SpatialExpr` by a sampled image, feed an image into the u/v function); emits
         ftrace's texture-sample call, numpy twin loads+bilinear-samples where coordinates permit. *(NB: the
         plain "import a jpg as a surface skin" arm already exists — `loom.scene.Texture` /`skin()`; the gap is
         only image-as-a-function-term.)*; (c) binding/rebinding **by substitution** — `gold(u=v)` replaces the
         `U` leaf with the consumer's expression at emit; (d) materials-as-bundles — a material's free-input set
         is the union of its properties' input sets. **Function-name parity is already an invariant**: every name
         `spatial.py` emits (`sin`/`sign`/`clamp`/…) must exist in `src/pattern.h` (only divergence: `abs`
         exposed as `sabs` in Python to dodge the builtin shadow, but it *emits* `abs`), so loom's function
         vocabulary is a subset of ftrace's by construction — which is why export is clean and J3c's shared
         grammar can be the single enforcement point.
         **`t` IS A FIRST-CLASS REBINDABLE INPUT (2026-07-19).** Don't treat `t` (clock time) as a magic ambient
         parameter — make it just one named input among `{t, x, y, z, u, v, a}`, rebindable by the same
         substitution as `u`/`v`/`a`. This unifies the Signal (temporal) and Surface (spatial) tiers *at the
         grammar level*: "evaluate as a temporal Signal" = bind `t`, leave nothing else free; "evaluate as a
         Surface" = bind `t` to the current frame's value, leave `x/y/z/u/v/a` free (symbolic on the emit path,
         numeric on `eval_np`). So **"a Surface is the expression with `t` frozen at the current frame"** falls
         out for free as a partial binding — no separate mechanism, no `t=0` memory (grid/scatter/RBF fields are
         already spatial samplers whose positions/values are Signals baked at the current frame). *Rationale /
         corrections that led here:* a Signal is a **pure, stateless function of a Clock** — loom can evaluate it
         at ANY `t` (it just reads whatever the clock carries); it is NOT true that "loom can't know the value at
         a different `t`." The only thing that assumes one-`t`-per-node-per-frame is the **cache** (`Cache` keys on
         `(node_id, frame)`, `signals/core.py`), and a future off-current-`t` sampler must fix the *cache key*
         (widen to the continuous sample point, or scope a nested cache in the retime node) — NOT forbid the
         capability. And "future `t` → loop" is unfounded for today's acyclic pure-function DAG (direction of `t`
         is irrelevant; a cycle needs a *recurrent/stateful* node, which loom has none of). **Caveat:** unifying
         the *grammar* (one node type, `t` an input) is clean, but the two *executors* stay distinct strategies on
         that node — scalar-per-frame (frame-keyed cache) vs numpy-array-over-space — don't pretend they're one call.
      4. **N-D *input* domain** (several named driver *axes*, not one `range` scalar).
      Each emits down to the J3a form or a documented construct (e.g. lower a `D=3` channel to `spectrum:`-refs +
      synthesised `spectrum` decls); non-lowerable forms stay loom-only representation.
- [ ] **J3c — full-scene `.ftsl` parser + emitter reconciliation.** Add `.ftsl -> loom Element tree` to
      complement the emitters so a whole scene round-trips (semantic re-emit). Audit every `Element.emit`
      against the live grammar and reconcile drift (e.g. `box { translate … size … round … }`,
      `uv planar axis=`, `type mix layer … weight_map pattern:…`, record `from`/dot-override blocks).
      **PARSER: use the user's GraphParser (GPDA) — `D:\visual studio projects\GraphParser`.** Write ONE shared
      EPEG `.ftsl` grammar (unified EPEG: regex terminals, `@skip`/`@mode`/`@longest`/`@left`/`@right`, actions).
      **DONE (foundation + record + material/texture + sphere/light blocks, 2026-07-19):** vendored the pinned
      tokenized `gpda.py` as `loom/grammar/_gpda.py` (commit 1ac4cbf, self-contained), shared grammar
      `loom/grammar/ftsl.epeg` (start=`element` = `record | material | texture | sphere | light | camera`, grows
      toward `scene`), reader `loom/grammar/reader.py`
      (`parse_record` → structural parity with the hand-written `Record.parse` oracle across every channel form;
      `parse_element` → `record`/`material`/`texture` blocks). Records prove parity vs the oracle; materials/textures
      (no `.parse` oracle) prove **emit is a fixed point** (emit → parse_element → re-emit byte-identical) + rebuilt
      kind/field round-trip, across image `Texture`, procedural `ProcTexture` (rgb-function), and scalar / vector /
      spectrum-ref / texture-ref material props over all material types (`tests/test_grammar_material.py`, 20 tests).
      Geometry `sphere` (fixed `center/radius/material` block), `light <kind> { … }` (material-shaped body), and the
      multi-line `camera "name" { … film { res W H } }` block added the same way (`tests/test_grammar_scene.py`,
      17 tests). *(Gotcha: a keyword literal that also matches the NAME regex — `light`, `material`, `eye`, `mode`,
      `film`… — tokenizes as a NAME node, so positional NAME extraction is brittle; `light`'s bareword kind is the
      **second** top-level NAME child, and `camera` names its parts via `cam_view`/`cam_mode`/`cam_film` sub-rules so
      the builder reads them by rule name instead of counting NAMEs.)*
      *(Gotcha fixed en route: GPDA collapses a single-terminal rule's value onto the rule node **and** keeps the child
      leaf, so the flat-`_terminals` walk must only take `node.value` on true leaves, else prop values duplicate.)*
      **KEY FINDING
      — the record ladder does NOT need scannerless.** The whitespace-form vs comma-form (§3.1) distinction is
      recoverable purely from explicit COMMA / NEWLINE tokens via a *"comma-form requires a comma"* ordered choice,
      with inter-token whitespace always `@skip`ped — so the **tokenized** flavour handles it (faster, and matches
      the near-BISON C++ path we want for ftrace). Scannerless would only be needed if whitespace were significant
      in one grammatical context but not another, which the record does not require. Later reuse the *same grammar*
      to upgrade ftrace's C++ parser (leaning full-replacement — the C++ tokenized GPDA is nearly BISON-speed).
      **SEQUENCING DECISION (2026-07-19) — option (a), grammar + ftrace front-end FIRST.** The moment loom starts
      emitting `gold(u=v)` / bundle-binding syntax (J3b item 3), those `.ftsl` files are un-renderable by shipped
      ftrace until the GraphParser front-end lands. To keep "everything loom emits is renderable" true at every
      commit, build the shared EPEG grammar + ftrace GraphParser front-end **before/in lockstep with** J3b item 3,
      not after. This front-loads the C++ parser work but never leaves an un-renderable emission window.
      **SCOPE CALL (2026-07-19) — stop the loom reader at "grammar proven", do NOT grind it breadth-first.** The
      load-bearing deliverable is the *grammar* (→ ftrace's C++ front-end); loom's Python reader is the proving
      ground. Once enough element shapes are covered to validate the grammar (record / material / texture / sphere /
      light / camera — DONE), stop extending the loom reader and pivot to (i) porting the grammar into ftrace and
      (ii) J3b item 3. Remaining loom-reader breadth (mesh-ref, `medium`, `pattern`, whole-`scene` wrapper + a
      `Scene` builder) is deferred — see next bullet.
- [ ] **FUTURE — loom full `.ftsl` read support** (deferred out of J3c above). Give loom a complete `.ftsl` → `Scene`
      reader (not just per-element round-trip): the whole-file `scene { … }` wrapper rule + a `Scene` builder that
      reassembles textures/patterns/records/materials/geometry/lights/camera into a live `Scene`, plus the lossy
      cases (`mesh { file … }` → a new lightweight `MeshRef` element that re-emits the same block; `medium`, `pattern`,
      `camera_curve`). **Motivating consumer: an editor/GUI** (load an existing `.ftsl`, manipulate in loom's object
      model, re-emit) and possibly the raster preview loading authored scenes. Not on any current critical path — the
      grammar's real job is ftrace's parser — so this waits until a concrete editor need exists.
- [ ] **PROPOSAL — unify element headers to `name = KIND { … }` (and anonymous `KIND { … }`).** Today elements
      spell their name inconsistently: records already use `NAME = range LO-HI [ … ]` (a `name = kind …` binding),
      but materials/textures/cameras use `KIND "name" { … }`. The cleaner, more programmatic form (per user, 2026-07-19)
      is to make *every* named element a binding — `hero = camera { … }`, `gold = material { … }`, `hide = texture
      { … }` — with the **anonymous** variety just dropping the `name =` (`camera { … }`, a nameless light, etc.).
      This unifies the whole scene grammar under one `binding = (NAME '=')? KIND block` shape (records fold in as the
      `range` kind), reads like assignment, and makes anonymity natural. Touches both loom's emitters (§ scene.py
      `emit`) and the shared grammar in lockstep; do it **before** the grammar ossifies into ftrace's C++ front-end so
      both sides adopt the new header at once. Decide alongside item 3 / the ftrace port.
- [ ] **SHIP — bundle GPDA with the ftrace release.** The GraphParser (GPDA) is becoming ftrace's scene front-end,
      so the shipped product now depends on it. Ensure `release.bat` / the release artifact carries the GPDA parser
      (vendored into ftrace's build, like loom vendored `_gpda.py`) and that a clean checkout builds/ships without an
      external GraphParser checkout. (Reminder logged 2026-07-19.)
- **Dependency note:** the FTSL record itself (§0) is fully implemented (Stages 1–6 + GPU parity DONE), so
  this is a loom-side mirror + parser effort, not blocked on ftrace.
- [ ] **FUTURE — loom retime / 4D time-shear node** (deferred; unlocked once `t` is a first-class input, J3b item 3).
  Once `t` is a passable *value* (not just the ambient clock), add a node that samples a subgraph at a
  **shifted / warped / per-point** time. Because a Signal is a pure stateless function of a Clock, sampling at
  an arbitrary `t` is well-defined and cheap (build a Clock at that `t`, evaluate). Capabilities this unlocks,
  none expressible under the current single-ambient-`t` model: **freeze** `sig(t=0)`, **echo/delay** `sig(t−dt)`,
  **time-warp** `sig(g(t))`, and the headline one — **4D time-shear**: sample a field at a *spatially varying*
  time, e.g. a wave whose phase lags with distance `field.at(t = T − X/c)`. Two things to get right when building
  it: (1) **cache** — `Cache` keys on `(node_id, frame)` and assumes one-`t`-per-node-per-frame; a retime node
  must key its child's memo on the actual (continuous) sample point, or scope a nested cache — do NOT restrict
  off-current-`t` sampling (that would defeat the feature). (2) **cycles — two distinct guards, don't conflate.**
  (2a) The **plain structural DAG cycle check already exists and is already enforced**: `detect_signal_cycle`
  (`signals/core.py`, 3-color DFS on `.id`/`.children()`) raises `SignalCycleError` before every render
  (`canvas.py`, `scene.py`), so a bad graph fails loudly instead of hanging/stack-overflowing — nothing to add,
  and it stays first-line. (It's effectively defensive today since Signals are immutable/bottom-up so a structural
  knot can't be tied through the API.) (2b) The **temporal-causality guard is the separate, deferred one**: a
  recurrent node (`v(t)=f(v(t−dt))`) is *structurally* a self-reference legitimately broken by a strict delay.
  Design so instantaneous edges stay in `.children()` (structural check owns them; a zero-delay algebraic loop =
  error, unchanged) and the recurrent/delayed edge is a **distinct edge kind** the structural check ignores and a
  new causality validator checks: "every path around a recurrence must cross ≥1 strict delay." Only ships with
  the first recurrent node.

---

## K. Light colour, RGB→spectral options, and an analytic physical sky  *(ftrace + loom; design-captured 2026-07-19 — this conversation)*

Fallout from the loom↔ftrace light-schema reconciliation (known-issues "RESOLVED: loom `Light(color=…)`…").
loom's `color=` now emits `spd rgb …` and `size`/`turbidity` were dropped from loom (a light is authored in
ftrace's own language). Two follow-ups were captured:

- [ ] **K1 — Multiple RGB→spectral upsampling methods (incl. a user-supplied mapping).** Today there is exactly
      **one** RGB→spectrum path, shared by *materials and lights alike*: `rgb r g b` → `rgbToReflectanceJH`
      (`src/upsample.h`), the Jakob-Hanika 2019 sigmoid-of-quadratic **reflectance** fit — coefficients solved by
      Gauss-Newton so the spectrum, viewed **under D65 through the CIE observer**, reproduces the target linear-sRGB
      colour. It always lands in (0,1) (a physical reflectance) and round-trips sRGB. **The gap:** using a
      *reflectance* fit for an **emitter** (`spd rgb`) is not principled — Jakob-Hanika also defines an *illuminant*
      upsample (no D65 pre-weighting, unbounded, so a bright/saturated emitter is representable), and other classic
      methods exist (Smits 1999, Meng 2015, a plain 3-lobe/box). Proposal: make the mapping *selectable* via a tag on
      the colour form — e.g. `rgb r g b` (default, reflectance) vs `rgb r g b illuminant` / a distinct `emit_rgb …`
      for the illuminant fit — and, further out, allow a **named user mapping**: a function `(r,g,b) -> spectrum`
      registered in the spectral-envelope store and referenced by name, so a scene can plug in its own upsampler.
      Materials would keep the reflectance default; lights would default to (or at least be able to opt into) the
      illuminant fit. Scope: an `upsample.h` illuminant variant + wire a tag through `evalSpectrum`'s `rgb`/`hsv`/
      `hsl` handlers; mirror in loom's spectrum grammar. Observable → README + VERSION bump when it lands.
    - [x] **Illuminant upsample landed** *(2026-07-20, v0.10.3).* The Jakob-Hanika *illuminant* variant is in:
          `rgbToIlluminantJH` (`src/upsample.h`) factors the SPD as `A·sigmoid(quadratic)` against the **bare** CIE
          observer (no D65), so the sigmoid carries chromaticity and the scalar `A = 2·max(X,Y,Z)` carries the
          (unbounded) magnitude — round-trips every colour, incl. saturated primaries and white, to <0.001 (see
          `-checkupsample`). Chosen surface: **head keywords** `rgbillum`/`hsvillum`/`hslillum` (parallel to the K3
          `…line` heads, since the parser drops trailing barewords), wired through `evalSpectrum` (`src/ftsl.h`) and
          mirrored in loom's spectrum grammar (`IllumSpec`). The Gauss-Newton solver was refactored into a shared
          `fitSigmoid()` so reflectance and illuminant fits share bit-identical arithmetic. Validated by
          `scraps/illum_test.ftsl` → `png/illum_test.png`.
    - [ ] **Still open:** other upsamplers (Smits 1999, Meng 2015, plain box/3-lobe) and a **named user mapping**
          — a `(r,g,b) -> spectrum` function registered in the spectral-envelope store and referenced by name.

- [ ] **K2 — Analytic physical sky (`turbidity`).** ftrace has **no** procedural sky: environment lighting is only
      an image-based env map (`env { file … }`) or a constant-radiance env. `turbidity` (atmospheric haze: ~2 = clear
      deep-blue sky, ~10 = milky/hazy) only means something inside a physically-based **sky+sun** model
      (Preetham 2002 or Hošek-Wilkie 2012), which *generates directional radiance from a sun position* — i.e. it is
      genuinely **about how the scene is rendered**, not merely a spectral envelope, so it warrants a real construct
      (a `sky` / `sun` light kind, or `env { kind hosek  turbidity t  sun_dir …  ground_albedo … }`), NOT just a name
      in the spectral-envelope store. Decision recorded (per user): a construct that only *names a spectral envelope*
      belongs in the generic spectral store; a construct that *drives the render* (like an analytic sky) is a
      first-class feature. This is the latter. Deferred as its own feature to greenlight on its own merits, not to be
      folded into reconciliation work. Bundles naturally with K1's illuminant upsample (the sky model wants proper
      emission spectra).

- [x] **K3 — RGB→wavelength map for lights (single dominant λ).** *(DONE 2026-07-20, v0.10.0.)* Distinct from K1's *upsampling* (RGB → a full
      spectral power distribution): this maps an (r,g,b) colour to **one dominant wavelength** — a monochromatic /
      narrow-line emission, so a coloured light behaves like a near-laser spike at λ(colour) rather than a broadband
      curve. Useful for pure spectral sources and for driving dispersion/refraction (a real λ so glass fans it out
      correctly), where a broadband upsample would wash the effect out. **The map:** convert the linear-sRGB colour to
      xy chromaticity, find the **dominant-wavelength** intersection on the spectral locus (the standard colorimetric
      construction: ray from the white point D65 through the sample's chromaticity to where it hits the horseshoe;
      purples between the line-of-purples endpoints have a *complementary* dominant λ and need a fallback — e.g. clamp
      to the nearest locus end or emit a two-line mix). Saturation → line narrowness / how peaked; value → intensity.
      Author it as an emission form on lights, e.g. `spd rgb r g b line` (or a dedicated `spd wavelength-of r g b`),
      routed through a new `upsample.h` helper `rgbToDominantWavelength(r,g,b)` returning a λ (nm) that then builds a
      narrow Gaussian / delta `Spectrum`. Lights-only (a *reflectance* has no meaningful single λ, so materials keep
      the K1 upsample). Mirror the form in loom's spectrum grammar. Observable → README + VERSION bump when it lands.
    - **Landed (2026-07-20, v0.10.1).** Added `upsample::rgbToDominantWavelength` + `rgbToLineEmission` in
      `src/upsample.h`: builds the CIE-1931 spectral-locus polygon once (400–700 nm at 1 nm, closed by the line of
      purples), casts the white→sample ray, and returns the crossing wavelength + excitation purity (or, for the
      purple edge, a violet↔red blend). `rgbToLineEmission` turns that into a `gaussianBand` whose width is
      `5 + 125·(1−purity)` nm by default or a forced `sigma`; purples become a two-lobe violet+red sum.
      **Syntax is a head keyword — `rgbline r g b [sigma]` / `hsvline …` / `hslline …`** — NOT a trailing `rgb r g b
      line` modifier: ftrace's `parseValue` (`src/ftsl.h` ~211) ends a property value at the next *bareword* (only
      numbers / `key=val` continue), so a trailing `line` word is silently dropped as a separate empty property. The
      first attempt used the trailing form and rendered plain reflectance (the `line` never engaged) — caught on
      re-verification. Wired the `…line` heads into `evalSpectrum`'s unified colour handler (`src/ftsl.h` ~1160).
      Validated A/B: `scraps/line_light_test.ftsl` (line) vs `scraps/line_light_ref.ftsl` (plain `rgb`) render
      *visibly different* illumination — the near-monochromatic lines are markedly more saturated (red +0.21 sat,
      green +0.26 sat). The blue sphere in the *combined* scene reads greenish, but that is **not a bug**: a deep-blue
      463 nm line has very low photopic luminance (round-trip Y≈1 vs green's Y≈71), so it is swamped by green spill
      from the brighter neighbour. The isolated `scraps/line_blue_only.ftsl` (single blue line, no spill) renders a
      **pure saturated blue** (mean RGB ≈ 0,0.6,160), confirming the full FTSL→spectral→display path is correct — the
      C++ CMFs/matrices are byte-identical to the `scraps/dbg_domwl.py` / `scraps/dbg_roundtrip.py` ports, which give
      blue=463 nm → display (0,0,1). Mirrored in loom's
      spectrum grammar (`tools/loom/loom/grammar/spectrum.py`: `LineSpec`, `_LINE_HEADS`). README's spectrum-forms list
      documents it. **Grammar shim verified clean:** `-validate-grammar` on a `rgbline` scene
      (`scraps/line_blue_only.ftsl`) reports no mismatch — the head-keyword form parses as an ordinary head+numbers
      value under the shared `.epeg` grammar, so no grammar change or shim-graph regen was needed.

## L. Native backward (camera-first) ray tracer mode  *(ftrace renderer; LARGE, design-captured 2026-07-20 — greenlit by user)*

> **STATUS CORRECTION (2026-07-20): the backward mode this section asks for ALREADY EXISTS as `mode R`.** When this
> section was written it assumed `src/backward.h` was only a "connect helper" and that a native camera-first
> integrator had to be built from scratch (as a new `mode E` / `src/backward_pt.h`). That premise was stale.
> `src/backward.h` (`BackwardRenderer`, driven by `-mode R`) is a **complete, native, camera-first, spectral
> backward path tracer**: it traces from the eye, reuses the finite-lens mode-A camera (`cam.hasLens()` /
> `genLensRay`, with chromatic aberration + DoF + vignetting), the BVH, the material BSDF interface, and the
> spectral XYZ film; it does NEE to lights + env, **MIS (balance heuristic)**, Russian-roulette termination, per-λ
> dispersion, participating media, and fluorescence; it is GPU-accelerated (its own backward megakernel) and
> respects every progressive control (`-window`/`-keepwindow`/`-interval`/`-checkpoint`/`-noise`/`-time`/`-resume`
> via `runSppProgressive`); and it is documented as a first-class render mode in README (§"`R` — backward reference
> (unbiased, general)"). So **L1, L3, and L4 below are already satisfied by `mode R`** — do NOT build a duplicate
> `mode E`. The **only** genuinely unbuilt part is L2's "3 stratified secondaries", i.e. **hero-wavelength spectral
> sampling**, which is re-scoped below (§L-HERO) as a cross-mode upgrade, not a new mode.

ftrace today is a **forward / light-first** engine: modes A/B/C shoot photons *from the lights* and accumulate on
the film (with a bidirectional connect in `bdpt.h`). This is ideal for caustics, participating media, and the
spectral effects the project cares about, but it converges slowly on directly-lit, low-caustic scenes where a
plain **backward / camera-first** path tracer (shoot rays *from the eye*, next-event-estimate to lights) is far
more efficient. The ask: add a native backward path-tracer mode as a first-class render mode alongside A/B/C —
**not** by exporting the scene to an external renderer. **(Done — this is `mode R`; see the status correction above.)**

### Why native, not an external renderer (answers the user's "lose spectral fidelity" question)

- **External renderers are RGB (tristimulus) at the core.** PBRT-v4 is spectral, but the common exchange path
  (glTF/USD/OBJ+MTL → Cycles, Embree-based tracers, OptiX samples, Mitsuba's RGB mode) carries **RGB material
  parameters**. Exporting ftrace's scene means collapsing every `Spectrum` (measured reflectance, Jakob-Hanika
  upsample, `rgbline` dominant-λ emitter, metal Fresnel curves, water Cauchy dispersion) down to three numbers at
  export time. Everything that depends on *the wavelength itself* is then gone:
  - **Dispersion / refraction fanning** (glass prism, the rainbow machine, water caustics) — needs per-λ IOR.
  - **Thin-film / iridescence** (soap film, oil, beetle shells) — interference is a function of λ.
  - **Jakob-Hanika round-trip & metamerism** — an RGB export can't reproduce two spectra that match under D65 but
    diverge under another illuminant.
  - **`rgbline` / narrow-line emitters and any measured SPD** — become a broadband RGB blob.
  So an external backward tracer would be *faster to bolt on* but would silently drop the exact features that make
  ftrace worth using. That is the "lose spectral fidelity" cost.

- **How WE keep spectral fidelity in a backward tracer (answers "how would we possibly").** The same
  **hero-wavelength Monte-Carlo** machinery the forward modes already use. A backward path is traced for a sampled
  wavelength λ (a "hero" λ plus optional stratified secondary λ's per path): at each bounce evaluate the material's
  reflectance/BSDF **at λ** (`mat.spdReflect(λ)`, IOR `n(λ)`, Fresnel at λ), do next-event estimation to a light
  and evaluate its emission `em.spdfn(λ)` (exactly the term `bdpt.h`/`backward.h` already computes), weight by
  `1/pdf(λ)`, and splat the resulting monochromatic radiance into the XYZ/spectral film accumulator via the CIE
  CMFs — the identical `color.h` path the forward modes use. Refraction uses the *per-λ* IOR so a single hero-λ
  path bends by the right amount and dispersion falls out for free. Nothing here is RGB; the film is spectral/XYZ
  and only tone-maps to sRGB at the end, same as A/B/C.

- **Is it significantly slower than an RGB backward tracer? No — essentially the same cost.** A backward path
  tracer's expense is ray traversal + BSDF sampling + NEE, which is *identical* whether the BSDF returns an RGB
  triple or a scalar-at-λ. Per-wavelength MC evaluates the BSDF at **one** wavelength per path (a scalar), which is
  actually *cheaper per-bounce* than an RGB tracer's 3-channel evaluation; the trade is slightly higher variance
  (colour noise) per sample because each path only carries one λ, needing modestly more samples for equally smooth
  colour. Hero-wavelength sampling (carry ~4 stratified λ's per path, MIS-combined — Wilkie et al. 2014) recovers
  most of that at ~unchanged traversal cost. Net: same order of magnitude as any spectral backward tracer, and the
  same order as an RGB one — *not* "significantly slower." The genuinely slow-to-converge cases (caustics, dense
  media) are exactly where you'd keep using the forward/bidirectional modes, so the two are complementary.

### Design sketch (to refine before coding)

- [x] **L1 — Mode selection + entry point.** *(Already satisfied by `mode R`.)* `-mode R` is parsed in `main.cpp`
      alongside A/B/C/D; `src/backward.h`'s `BackwardRenderer` is the camera-first integrator (NOT a mere connect
      helper). It reuses the finite-lens mode-A camera, the BVH/intersection, the material BSDF interface, and the
      spectral film, and respects all progressive controls (`runSppProgressive`, chunked by spp passes). No new mode
      letter / `backward_pt.h` needed.
- [x] **L3 — Dispersion validation.** *(Already satisfied by `mode R`.)* Per-λ dielectric IOR in the backward tracer
      already produces the correct spectral fan (`-scene prism`); `mode V` cross-validates backward (`R`) against
      forward (`B`), which is exactly this acceptance test.
- [x] **L4 — Docs + version.** *(Already satisfied.)* README's render-modes list documents `R` as a first-class mode
      ("`R` — backward reference (unbiased, general)"), including its GPU scope and the known caustic-noise weakness
      (kept on A/B/C), which is the known-issues note this item called for.

### L-HERO — hero-wavelength spectral sampling *(the genuine remaining work; re-scoped from L2; applies to ALL spectral modes)*

- [ ] **Hero-wavelength Monte-Carlo across every spectral render mode.** Today ftrace carries **one** wavelength per
      path/photon **everywhere** — the forward light tracers (**A/B/C**, CPU + GPU), the backward tracer (**R**, CPU +
      GPU megakernel), and BDPT (**D**) all sample a single λ and splat `cieX/Y/Z(λ)·L`. README §"spectral" explicitly
      contrasts this single-λ scheme with PBRT-v4 / Mitsuba 3's 4-λ hero-wavelength. **Upgrade every mode it applies
      to** (A, B, C, R, D — and the GPU megakernels, not just the CPU paths) to carry a **hero λ + 3 stratified
      secondary λ's** per path, evaluate the BSDF/IOR/Fresnel at all four, and **MIS-combine across wavelengths**
      (Wilkie et al. 2014 spectral MIS) so chromatic (colour) noise drops sharply at ~unchanged traversal cost. The
      one subtlety per mode: a **specular/dispersive interface** (dielectric with `n(λ)`) refracts each secondary λ by
      a different angle, so the secondaries must "de-hero" (collapse to the single hero λ, weight renormalized) past
      the first dispersive bounce — standard hero-wavelength practice; verify dispersion (`-scene prism`) is unchanged
      and single-scatter media / thin-film still integrate correctly.
    - [x] **A/B/C (forward light tracers) — CPU DONE.** `tracePhotonHero()` in `src/render.h` carries a hero λ + 3
          stratified secondaries (`hero.h`, `kHeroC=4`) along one shared BVH walk, with per-λ throughput `beta[C]`.
          Camera contribution splats all live λ each vertex via `camSplatAllHero`/`connectHero` (mode B),
          `connectLensHero` (mode A finite-lens pupil, achromatic thin-lens geometry shared across λ), the forwardCatch
          multi-λ deposit (mode C), and the dispersive glass-sphere caustic `camSpecularSplatAllHero` (per-λ root
          solve). No code duplication: `connect()` was split into a λ-independent `connectGeom()` reused by the scalar
          and hero paths, and the 9 specular lobes were extracted into a shared `interactPhotonSpecular()` driven by
          both `tracePhoton` and de-hero. De-hero at any dispersive/wavelength-switching interface (dielectric,
          thin-film, multilayer, mirror, grating, half-mirror, filter, glossy, fluorescent) boosts the hero ×C —
          PBRT-v4 `TerminateSecondary` convention, exactly energy-preserving at the switch. Gated by the driver
          (`main.cpp` `renderForward`/`renderForwardShared`) on `kHeroC>1 && scene.media.empty() && !sceneHasGrin`
          (media/GRIN stay scalar; C=1 is bit-identical to the classic tracer). Validated on `cornell` mode B at
          n=1e8, 300² vs a 1e9 GPU single-λ reference: converged image + glass-sphere dispersion intact, energy
          conserved (`sum/emitted≈1.0025`), **luma noise flat (0.97×), chroma noise down (0.77×)** at equal photons.
    - [x] **A/B/C + M-deposit (GPU megakernel) — DONE.** `render_cuda.cu` gains a device twin of
          `tracePhotonHero`: `genPhotonHero` (stratified λ via the shared `sampleLambdaU`, `beta[i]=base/C`,
          env reweight per-λ), `shadeStepHero` (per-λ deposit + camera splat via `connectHero`/`connectLensHero`/
          `camSpecularSplatAllHero`, hero Russian-roulette with secondary reweight `beta[i]*=rho[i]/rhoHero`), and
          `traceHeroPhoton` (emit → bounce; de-hero at a dispersive/wavelength-switching interface boosts the hero
          ×C and falls through to the ordinary single-λ `shadeStep`). No duplication: the nine specular lobes were
          extracted into a shared device `interactSpecular()` driven by both the scalar `shadeStep` and de-hero.
          `kTrace` branches on a new `heroC` parameter; `launchForward` gates it on `up.sc.mediaN==0 &&
          !up.sc.hasGrin` and **forces the megakernel** (hero is not in the wavefront scheduler). Threaded through
          `renderForwardCuda`, `renderForwardSharedCuda` (modes A/B/C) and `renderPhotonMapSharedCuda` (mode-M
          deposit — each diffuse bounce deposits all C live wavelengths as per-λ records, exactly like the CPU M
          path), all fed `g_heroC` from `main.cpp`. Validated on `cornell` mode B GPU at n=5e7, 300²: **energy
          conserved exactly** (`-heroc 4` and `-heroc 1` both converge to auto-exposure 1.06e-13), and `-heroc 1`
          reproduces the classic single-λ device stream bit-for-bit.
          **Still TODO:** the GPU forward wavefront path (streaming backend) — hero forces the megakernel there;
          and the GPU backward/BDPT megakernels (below).
    - [x] **R (backward) — CPU DONE.** `radianceHero()` in `src/backward.h` samples a hero λ + 3 stratified
          secondaries (`hero.h`, `kHeroC=4`), rides them along one shared BVH walk, evaluates materials/NEE per-λ
          (`neeLightHero`/`neeEnvHero`, shared `interactMaterial`/`emitterGeom`/`envGeom` helpers) and splats 4
          CMF-weighted contributions (÷C). De-hero at any dispersive/wavelength-switching interface (dielectric,
          thin-film, multilayer, grating, filter, fluorescent) boosts the hero ×C — PBRT-v4 `TerminateSecondary`
          convention. Gated on: `kHeroC>1 && no fog/GRIN/lens` (those stay scalar, C=1 is bit-identical). Validated
          on `cornell.ftsl`: converged image unchanged, glass-sphere dispersion intact, **luma noise flat (1.03×),
          chroma noise down (0.89× overall, 0.74× in spectral-dominated neutral regions)** at equal spp.
          **Still TODO:** the GPU backward megakernel (`renderBackwardCuda`) — mode R on GPU is still single-λ.
    - [x] **M (photon map) + S (SPPM) — CPU DONE.** The shared forward photon pass (`tracePhotonPass` in
          `src/photonmap_render.h`, used by both modes M and S) now sets `r.useHero` under the same gate as the
          forward tracers (`kHeroC>1 && scene.media.empty() && !sceneHasGrin`), so each traced path runs
          `tracePhotonHero` (`src/render.h`) with `nCam==0` (deposit-only, no camera splat). The key fix: the
          photon-map **deposit** now stores EVERY live wavelength as its own per-λ `Photon` record (a loop over
          `nUp`), because the stored map is the product here — a scalar single-λ deposit would discard `(C-1)/C`
          of the spectral energy. C records of `base/C` sum to `base`, and `nEmitted` still counts PATHS, so the
          density estimate is energy-identical to the single-λ deposit; the gather already keys off each photon's
          own λ, so a heterogeneous-λ map (diffuse bounces carry C wavelengths, post-de-hero specular bounces
          carry 1) gathers correctly. De-hero at any dispersive interface terminates the secondaries exactly as
          in the forward tracers. Validated on `cornell` mode M (`-n 2e6`, 200²) vs a C=1 rebuild: **energy
          conserved exactly** (auto-exposure identical, 1.11e-13, hero 9.09M vs single 2.80M photons), **chroma
          noise down 0.87×**, luma flat, against a 1.5e7 single-λ reference. (Milder chroma win than A/B/C's 0.77×
          because the gather already averages many photons.) Modes S (SPPM) inherits it via `tracePhotonPass`.
          **Still TODO:** mode U (VCM/UPS) — its BDPT-style light-subpath tracing (`src/vcm.h`) is the same
          complexity class as BDPT-D below; and all GPU photon-mapping paths.
    - [x] **Runtime `-heroc N` flag — DONE.** The bundle size is now a runtime CLI knob (`hero.h`: `kHeroC=4`
          default, new `kHeroMax=8` compile-time cap for the fixed stack arrays). `main.cpp` parses `-heroc N`
          (clamped 1..kHeroMax) into `g_heroC` and threads it to every CPU hero path: `Renderer::heroC`
          (modes A/B/C + photon-map M/S via `tracePhotonPass`), `BackwardRenderer::heroC` (mode R), and
          `sppmPass` (mode S). All `[kHeroC]` stack arrays in `render.h`/`backward.h` became `[kHeroMax]`; the
          hero gate is now `heroC>1`. Verified on `cornell` mode M: `-heroc 1` is bit-identical to a `kHeroC=1`
          rebuild (2.80M photons, auto-exposure 1.11e-13), `-heroc 4` matches the default (9.09M, 1.11e-13),
          `-heroc 2`/`8` interpolate and run clean; mode B `-heroc 1` gives `sum/emitted=1.000000`.
    - [ ] **Optional split-at-dispersion (crisp dispersive caustics)** — an *alternative* to the default de-hero
          policy, exposed as an opt-in flag (e.g. `-herosplit`, off by default). At a dispersive dielectric interface,
          instead of terminating the secondaries (`beta[0] *= C; secAlive = false`), **continue all C wavelengths**,
          each refracting along its *own* per-λ direction from that point — turning one bundle into C now-monochromatic
          sub-paths that fan out through the glass. This is the physically-crisp option for prism / caustic / rainbow
          shots where de-hero's single shared geometry blurs the chromatic spread. It is a legitimate, standard
          technique (PBRT-style spectral path splitting); the honest reason it is *not* the default is cost, not
          bias — (a) C× traversal work past the split (linear, not exponential — once monochromatic a child does not
          re-split at further dielectrics), and (b) GPU execution divergence as the fan-out wavelengths take different
          branches. Keep memory bounded on the forward photon map by **throttling emission** while split sub-paths are
          live (a fixed work-pool with emission back-pressure), so peak in-flight paths — and thus GPU photon buffers —
          stay constant regardless of split depth. Scope: forward photon map (M/S) first, since that's where crisp
          dispersive caustics matter most; the backward tracer (R) can adopt the same flag later. Validate that with
          the flag *off* the image is bit-identical to today, and *on* it converges to the same converged energy
          (de-hero is unbiased; splitting is a different, also-unbiased estimator — same mean, sharper caustics, more
          work per path). README + a `-herosplit` flag-table row; VERSION minor bump when shipped.
    - [ ] **U (VCM/UPS)** — carry the N λ along the light subpath and merge/connect per-λ (BDPT-level MIS). CPU + GPU.
    - [ ] **D (BDPT)** — carry the N λ along both subpaths; the connection term evaluates per-λ. GPU megakernel too.
    - [ ] **Shared plumbing** — a small `HeroLambda` struct (hero + 3 secondaries + per-λ pdf/MIS weights) threaded
          through the spectral evaluation sites, so the four modes share one wavelength-sampling + de-hero policy
          rather than four copies. Validate: every mode's converged image is unchanged vs the single-λ baseline (same
          tone-map) but reaches a given colour-noise level in ~fewer samples; dispersion/thin-film unaffected.
    - [ ] **Docs + version** — README §"spectral" updated (ftrace becomes hero-wavelength, 4 λ/path); VERSION minor
          bump; note any mode where the secondaries are deliberately dropped (e.g. hard-specular chains).

Sequencing note: L1/L3/L4 are done (mode R). The remaining L-HERO work is a real, cross-mode spectral-core upgrade;
start with the backward tracer (R) as the reference, then propagate the shared `HeroLambda` plumbing to A/B/C and D
(CPU first, then each GPU megakernel), keeping every mode bit-comparable to its single-λ baseline at convergence.

---

## M. GPU fallback closure — port CPU-only features/modes to the GPU  *(ftrace renderer; audit in `gpu-fallbacks.md`, greenlit by user 2026-07-23)*

Close the GPU/CPU-fallback gaps found by auditing every `cuda*Supported()` predicate.
Full rationale + per-feature classification lives in `gpu-fallbacks.md`. Ordered by the
recommended priority (quickest, highest-value wins first). Check each off as it lands and
mark the corresponding row in `gpu-fallbacks.md`.

- [x] **M1. Image-based env NEE in GPU backward (mode R).** *(done 2026-07-23)* `cudaBackwardSupported`
      rejected an image env (`scene.envMap`) → CPU. The forward path already uploads the lat-long luminance
      2-D CDF + per-texel JH coeff/scale and samples it on-device; added `dEnvRadiance`/`dEnvPdf`, uploaded
      the previously-canceled illuminant table (`DEnvMap::illum`), and wired the on-device env sampler into
      `bkNeeEnv` / `bkNeeEnvVolume` (+ MIS'd env-miss) so a lat-long env stays on the GPU. Dropped the
      `scene.envMap` reject. Also unblocks mode P camera-side. **Validated:** GPU vs CPU backward on
      `scenes/envmap.ftsl` at 8192 spp match to 0.14% in linear radiance (background sky 0.15%); the earlier
      ~5% gap was noisy p99 auto-exposure, not radiance.
- [x] **M2. Env term in the mode-M GPU gather.** *(done 2026-07-23)* `cudaPhotonMapSupported` rejected ANY
      env light because device `dPhotonGather` had no env term. The deposit already emits env photons (env's
      indirect bounces), so only the DIRECT term was missing: added the env contribution on gather-ray escape
      in `dPhotonGather` (constant via `emitSpd`, image via `dEnvRadiance`, monochromatic at the sampled
      lambda like CPU `photonGather`) and dropped the `envIndex >= 0` reject. **Validated:** GPU vs CPU
      mode-M on `scenes/envmap.ftsl` (20M photons) match in mean to 0.18%, background sky to 0.04%; residual
      per-pixel diff is mode-M's inherent density-estimate + monochromatic-background noise (identical
      character on CPU).
- [x] **M3. GPU SPPM (mode S).** *(done 2026-07-23)* Built a resident device SPPM session (`SppmSession` in
      `render_cuda.cu`): per-pixel progressive state (`tau`/`radius`/`nAcc`/`directSum` + this pass's visible
      point) stays on the device across passes. Each pass runs `kSppmVisiblePoint` (resample camera visible
      point + direct term, following the specular walk like CPU `sppmVisiblePoint`), deposits a bounded photon
      set via the SAME forward tracer as mode M (`launchForward`, fresh seed = cumulative emitted), host-builds
      the grid at the largest current per-pixel radius, then `kSppmGather` (query + Hachisuka shared-statistics
      radius/flux update `-sppmalpha`) and `kSppmResolve` (`L = directSum/passes + tau/(pi R^2 Nemit)`). SPPM
      photon record bakes `pX = cie(lambda)*power/pi` — NO area/nEmitted fold (those depend on the current
      per-pixel radius, applied at resolve), unlike mode M. Stores the PARENT matId at the visible point and
      gathers with it, matching CPU. `cudaSppmSupported == cudaPhotonMapSupported`; pinhole cameras only. Wired
      the GPU dispatch into main.cpp's mode-S block (`-device gpu/auto`, self-gated). **Validated:** GPU vs CPU
      mode-S on `scenes/cornell.ftsl` (glass sphere + caustic, 300k photons/pass): mean linear radiance 0.2–1.2%,
      background wall 0.3%, and the per-pixel diff shrinks 7.9%→5.4% as passes go 60→240 (independent-MC noise,
      not bias); images structurally identical incl. the floor caustic + refracted light. GPU ~4x the CPU
      pass rate at 256².
- [x] **M4. Mode-M final gather on GPU.** DONE 2026-07-23. Added device `dPhotonGatherSub` (device twin of
      `photonGatherSub`: follows specular surfaces, then at the first diffuse hit y does a radius density query
      folding `rho(y)*rho(vis)` per photon wavelength; env-on-escape and specular-arrival emitters reflected off
      the visible point) and a `fgRays>0` branch in `dPhotonGather` (NEE direct via `bkNeeLight` + K
      cosine-hemisphere sub-rays). Threaded `fgRays` through `kGather` → `renderPhotonMapSharedCuda` → header,
      and dropped the `g_pmFinalGather == 0` caller gates (main.cpp meter 6083 + flyby 6489) so `-pmfg` now runs
      on the GPU. Validated GPU==CPU on `scenes/cornell.ftsl` (glass sphere + diffuse walls, `-pmfg 16/24`): mean
      linear radiance 0.43%, background 0.98%, per-pixel diff √-scales 22%→11.5% at 4× spp (independent-MC noise,
      not bias). Falls back to CPU for lens cameras / unsupported scenes exactly as the direct gather does.
- [x] **M9. Per-hit BSDFs in GPU BDPT (mode D). DONE 2026-07-23** — all genuine per-hit-BSDF GPU-vs-CPU parity
      gaps in mode D are closed. THREE INCREMENTS. (1) `DVertex` now stores the
      per-hit texcoords (`u,v`) and `dVertHit` reconstructs a `DHit`, so the connection BSDF (`dBsdfF`/`dBsdfPdf`)
      and the random walk evaluate per-hit-driven throughput slots consistently in BOTH the sampler and the
      pdf/eval — MIS-safe. Ported: textured/patterned/record diffuse albedo & glossy reflect, per-hit glossy
      roughness + thin-film thickness maps, mix blend masks, and Beer-Lambert colored-glass interior absorption
      (delta vertex → throughput only, mirrors bdpt.h). Validated GPU==CPU on `textured.ftsl` (mean 0.06%,
      per-pixel diff halves 8.2%→4.3% at 4× spp = unbiased) and `mixmat.ftsl` (mean 0.21%). (2) Two-sided
      **diffuse-transmit** (translucent) now on-device — both lobes (front `reflect` / back `transmit`,
      energy-clamped) + the two-sided back-hemisphere connection strategy; `lambda` threaded through
      `dBsdfPdf`/`dVertexPdfF`/`dMisWeight` for the wavelength-dependent lobe-selection pdf; `dConnectBDPT`
      two-sided guards mirror bdpt.h. Validated GPU==CPU on `scraps/dtrans.ftsl` (mean B/A=1.0009 at 512 spp,
      per-pixel diff halves 8.42%→4.39% at 4× spp = unbiased). (3) **Frosted (rough) dielectric** now on-device
      — only the gate needed relaxing: `refractOrReflect`/`dDielectricStep` already jittered the chosen lobe by
      per-hit `dMatRoughness` (stochastic-delta, non-connectable, exactly like `bdpt.h`); the old "kernel treats
      every dielectric as smooth" note was stale. Validated GPU==CPU on `scraps/frosted.ftsl` (mean B/A=0.9991
      at 512 spp, per-pixel diff halves 10.86%→5.73% at 4× spp = unbiased). `cudaBdptSupported` relaxed accordingly.
      NOT GAPS (investigated, closed out): fluorescence and spot/env/collimated lights are unsupported by BDPT on
      *every* backend — `main.cpp bdptUnsupportedFeature()` refuses mode D (or demotes D→B with -on-unsupported
      fallback) for those scenes before dispatch, so they never reach the BDPT path (CPU or GPU). The stale,
      now-unreachable per-material rejects in `cudaBdptSupported` were removed. True fluorescence/spot/env is a
      mode A/B/C/R/P feature on both CPU and GPU — not a GPU-BDPT closure item.
- [x] **M10. Spectral rainbow-phase media on device — DONE (2026-07-23, 0.37.0).** The λ×µ phase table +
      per-λ CDF (`rainbow.h`) is uploaded per-medium; unified device dispatch `dMedPhase`/`dMedPhaseSample`
      (rainbow → bilinear `dRbEval` / binary-search CDF sample; HG → analytic lobe) replaces the raw
      `hgPhase` calls across forward (deposit walk, `connectVolume`/`connectLensVolume`, specular-sphere
      splat), backward (`bkNeeVolume` spot/area/env, `bkNeeEnvVolume`, `bkRadiance` scatter), and BDPT
      (`dPhaseF`/`dPhasePdf`/`dMediumScatterF` + random walk). Rainbow rejects relaxed in
      `cudaForwardSupported`/`cudaBackwardSupported`/`cudaBdptSupported`. **Validation:** isolated the rainbow
      phase from a pre-existing, phase-independent GPU↔CPU media brightness discrepancy (see known-issues.md)
      by comparing rainbow *and* a plain-HG control in every mode. In clean, well-converged **mode-D BDPT** the
      rainbow gives GPU↔CPU B/A=2.41 and HG gives B/A=2.41 (identical to 3 s.f.) — the rainbow adds **zero**
      bias beyond what HG already shows; forward-mode bulk median ratio is 1.02 (rainbow) / 1.00 (HG); visuals
      show correct primary+secondary bows with spectral separation. The git diff also proves the HG BDPT path
      is bit-for-bit unchanged by the refactor.
- [x] **M11. GRIN (gradient-index) media on device backward.** *(DONE 2026-07-23, 0.38.0.)* Ported the
      Eikonal marcher to the device as `dGrinMarch` (render_cuda.cu:2040) — byte-identical to `grin::march`,
      running (ro,rd) carried in double to mirror the CPU ground truth. `bkRadiance` marches each bounce's
      ray (incl. the primary camera ray) before `closestHit`, gated by `sc.hasGrin`; removed the GRIN reject
      in `cudaBackwardSupported`. BDPT (`cudaBdptSupported`) and the RGB fast path (`cudaBackwardRGBSupported`)
      still reject GRIN (straight-segment MIS / RGB-throughput). Validated GPU==CPU on `scraps/grin_lin.ftsl`
      (linear lens): SSIM 0.99, Pearson 0.99, both bend identically; a small bent-region float-vs-double
      residual (~2.7% disc linear, up to ~17% on a strong radial caustic, non-converging) is logged in
      known-issues.md as the accepted device-float envelope amplified through the lens.
- [x] **M12. GPU VCM (mode U).** *(DONE 2026-07-23, 0.39.0.)* Ported the CPU VCM/UPS (`vcm.h`) to the
      device as a resident `VcmSession` (render_cuda.cu) mirroring `vcmPass`. Each pass: (1) `kVcmLight`
      traces one light subpath per pixel, storing connectible vertices into a per-path slab (no cross-thread
      atomics) + splatting connect-to-camera (t=1) contributions; (2) the host downloads the slab + per-path
      counts and compacts into contiguous per-path ranges (so the same-λ vertex CONNECTION reads its PAIRED
      light subpath); (3) builds the uniform hash grid over the compacted vertices (counting sort, cell =
      merge radius — a byte-for-byte mirror of `VcmGrid::build`); (4) `kVcmCamera` traces one camera subpath
      per pixel doing emission (s=0) / NEE (s=1) / paired-path connection (c) / grid merge (d), accumulating
      the running per-pixel sum; resolve divides by the pass count. Reuses M9's device BDPT BSDFs
      (`dBsdfF`/`dBsdfPdf`/`DVertex`) and M3's device grid pattern; `dVcmScatter` is the device twin of
      `scatterSample`; misArrival/misScatter are inlined (Mis=identity, balance heuristic). Gate
      `cudaVcmSupported = cudaBdptSupported && media.empty()` (mode U is surfaces-only; pinhole cameras only).
      main.cpp mode-U GPU branch mirrors mode-S (auto/gpu device, radius schedule `r_i=R0·i^((α-1)/2)`).
      Validated GPU==CPU statistically on `scenes/absolute.ftsl` (Cornell + dielectric sphere, fixed-gain
      absolute mode to bypass per-image auto-exposure) at 500 passes: mean linear-luminance ratio 0.9993
      (−0.07%), per-channel bias all within ±0.5% (R −0.43%, G −0.06%, B +0.20%), per-pixel median rel error
      3.0% sitting at the ~4.5% independent-MC noise floor — no systematic bias. Slab-download memory scaling
      (~vcmCap·npix·128 B) is logged in known-issues.md.

**Descoped by user (2026-07-23) — not scheduled:** indexed-spectral palette maps on device forward,
Layered material on device, participating media in the RGB fast path, and textured/record albedo in the
RGB fast path. These stay on their current CPU/spectral fallbacks.

Left on CPU **by design** (not in this list): collimated beams (not NEE-samplable), dispersion-dependent
materials in the RGB fast path (inherently spectral), and fixed-cap overflows (oversized multilayer/mix,
>64-stop driven records, over-deep lens). See `gpu-fallbacks.md` → "Left on CPU by design".

---

## Progress log
- 2026-07-19: **J3c started (option-a) — GPDA vendored + shared grammar reads the record block.** Stood up
  `loom/grammar/`: vendored the pinned tokenized `gpda.py` as `_gpda.py` (GraphParser commit 1ac4cbf,
  self-contained — only `import re`) with a provenance header; the shared EPEG grammar `ftsl.epeg` (start=`record`,
  `#`-commented, will grow to `scene`); and `reader.py`'s `parse_record`, which walks the GPDA `ParseNode` tree
  into a `Record`. Round-trips to **structural parity with the hand-written `Record.parse` oracle** across every
  channel form (whitespace scalar / `spectrum:`-ref colour, vector + lone-vector trailing comma, inline
  `rgb`/`hsv`/`hsl`, position pins, all interp modes, both `LO-HI`/`-1-2`/`LO HI` domain spellings, compact
  single-line body); emit → parse → emit is a fixed point. **Corrected the earlier "ladder needs scannerless"
  assumption:** the tokenized parser handles the comma-vs-whitespace distinction via a "comma-form requires a
  comma" ordered choice (whitespace `@skip`ped, structure from explicit COMMA/NEWLINE) — no context-aware skip
  needed, keeping the fast tokenized path that also suits ftrace's C++ front-end. 20 new tests (`test_grammar.py`
  smoke + `test_grammar_record.py` round-trip), 697 loom green.
- 2026-07-19: **`t` unified as a first-class input + future retime/4D node scoped (design, no code).** Decided `t`
  is not a magic ambient parameter but one named input among `{t,x,y,z,u,v,a}`, rebindable by the same
  substitution as `u/v/a` — so Signal (temporal) and Surface (spatial) unify at the grammar level and "a Surface
  is the expression with `t` frozen at the current frame" is just a partial binding. Corrected three
  misconceptions in the process: a Signal is a *pure stateless function of a Clock* so loom CAN evaluate at any
  `t` (the constraint is the `(node_id, frame)` cache, not the math); "future `t` → loop" is unfounded for an
  acyclic pure-function DAG (a cycle needs a recurrent node, which loom has none of); the fix for off-current-`t`
  sampling is a wider cache key, never forbidding the capability. Logged the `t`-input note under J3b item 3 and
  a new deferred **retime / 4D time-shear** TODO (freeze/echo/time-warp + spatially-varying-time `field.at(t=T−X/c)`).
- 2026-07-19: **J3b item 3 / J3c design settled (no code yet).** (1) Item 3 lives on the existing `SpatialExpr`
  tier (`loom/spatial.py`) — no VM, no `Clock.env`, no lowering. Leaf set: public **`Surface`** family
  (`X Y Z` existing, `U V A` new **emit-only**, `T` loop-phase) + new **`Image("path")`** leaf (image as a
  function-term; the plain jpg-skin arm already exists via `loom.scene.Texture`/`skin()`). Binding is
  substitution (`gold(u=v)` swaps the `U` leaf at emit); materials-as-bundles (free-input = union). `u/v/a`
  emit-only because those coords exist only at a 3-D ray hit, not on loom's flat preview canvas — but the
  **emit path carries them straight to ftrace**, matching ftrace's "surface = function of u/v/a" mode
  bit-for-bit, so export parity is unaffected (the limitation only bites loom's own `eval_np` numpy preview).
  Function-name parity with `src/pattern.h` is already an invariant (`sabs` emits `abs`). (2) J3c parser =
  the user's GraphParser (GPDA), one shared EPEG grammar, scannerless variant for the comma-aware ladder.
  **Sequencing: option (a)** — stand up the shared grammar + ftrace GraphParser front-end *first/in lockstep*
  with J3b item 3, so loom never emits un-renderable `.ftsl`. TODO J3b item 3 + J3c updated with settled scope.
- 2026-07-19: **J3b item 1 complete — inline `rgb`/`hsv`/`hsl` colour channels + lowering to spectra.** A record
  colour channel can now be authored *inline* with a leading colour-space **tag** (`reflect  rgb 0.55 0.57 0.60,
  0.90 0.75 0.30`) instead of a chain of `spectrum:<name>` refs. `RecordChannel` gained a `space` field
  (`rgb`/`hsv`/`hsl`, else `None`); the tag fixes arity 3 so each comma-group is one colour stop and a lone
  tagged stop (`reflect  rgb .5 .5 .5`) needs no trailing comma. `parse` detects the leading tag, `emit` emits it
  back (round-trips, pins preserved), and `sample_vec` numerically samples an `rgb` channel (per-component =
  ftrace's linear-RGB colour interp) while `hsv`/`hsl` reject sampling until lowered. New `Record.lower_colours()`
  returns `(decls, lowered_record)` — synthesizing one deduped `spectrum "<name>" = rgb r g b` decl per unique
  colour (`hsv`/`hsl` converted to rgb via loom's own hue maths in `color.py`, single source of truth) and
  rewriting inline-colour channels to `spectrum:<name>` refs; `lower_ftsl()` returns the whole thing as one
  self-contained ftrace-parseable block. Scalar/vector/`spectrum:`-ref channels pass through unchanged;
  expression-valued colour stops raise (need the pattern VM, J3c). 10 new record tests (44 total), 670 loom green;
  DESIGN.md §8a + ROADMAP_records.md §3.1 updated; TODO J3b item 1 marked DONE. **Remaining J3b:** item 3
  (binding/application surface, §3.2/§3.3) + item 4 (N-D input domain).
- 2026-07-19: **Locked the type lattice — values · channels · records (§3.0).** Resolved a multi-turn design
  thread on how the pieces are typed and named. Three **value** kinds: `number` (a real type, *not* a
  degenerate spectrum — roughness/IOR/weights are inherently scalar), `vector` (bare `1 1 1`, no colour
  meaning), `spectrum` (a colour = a curve over **wavelength λ**, no driver). Two **containers**: `channel`
  (a mapping from a driver input to *any* type — value, another channel giving multi-input by currying, or a
  record; deliberately the same word as `RecChannel`) and `record` (a bundle of **co-driven** channels sharing
  one driver). **One-way promotions** make the simple form typecheck in the richer slot: `number`→`spectrum`
  (grey SPD; reverse never holds — a spectrum can't be a roughness), any value→constant `channel`, single
  `channel`→one-channel `record`; `vector`→`spectrum` only via an explicit `rgb`/`hsv` keyword. Also nailed
  **slot-type vs value-expression**: the LHS slot keyword declares the output type, the RHS is *always* an
  expression over named inputs (constant / open array / applied channel / formula are one tier); a bare `[…]`
  is **driver-*open*, not implicitly-`u`**; `spectrum = u*.5` and `spectrum = a*.5` are *both* legal (the
  intended "nothing is closed"); a standalone array is polymorphic data the assignment pins to a slot. Written
  into `ROADMAP_records.md` new §3.0 (canonical vocabulary for the rest of §3).
- 2026-07-19: **J3b item 1 (core) — arbitrary-arity vector channels in `Record`.** Generalized `RecordStop`
  to hold `.components` (a `D`-tuple; `.token` = the single component of an arity-1 stop, `.arity` /
  `.as_vector()` the vector view) — J3a scalar/colour paths unchanged (`.token` back-compat). `RecordChannel.kind`
  now returns `scalar` (arity 1) / `colour` (`:`-refs) / **`vector`** (arity `D` ≥ 2, homogeneous; ragged
  arity rejected), with a `.arity`. `Record.sample_vec(name, d)` interpolates per-component (scalar `sample`
  still returns a float and rejects vector channels). Initially the grammars were kept separate (`emit`/`parse`
  whitespace vs `emit_generalized`/`parse_generalized` ladder); this was later refolded into one grammar (see
  the 2026-07-19 refold entry below). `from_channels` accepts vector stops (lists) + `(value, pos)` pins.
  **Remaining J3b item 1:** inline-`rgb` colour channels + lowering to synthesized `spectrum` decls; then
  item 3 (binding/application surface) + item 4 (N-D input domain).
- 2026-07-19: **Refolded the record parser into ONE backward-compatible ladder grammar (J3b item 2).** Collapsed
  the two parallel APIs (`parse`/`emit` whitespace + `emit_generalized`/`parse_generalized` ladder) into a single
  `parse`/`emit` pair in `loom/record.py` that dispatches **per channel line** on the presence of a top-level
  comma (`_split_top_commas`): a comma-free line is the exact current-FTSL whitespace form (`metal steel gold
  copper` = three scalar stops), while a line with a top-level comma is the ladder form (`tint 0 0 0, 1 1 1` =
  two arity-3 vector stops). A **lone vector stop** is written/read with a trailing comma (`tint 0 0 0,`) to
  disambiguate it from N scalar stops. `emit` now picks the form per channel automatically (whitespace for
  scalar/colour, comma for vector) instead of rejecting vector channels — records with no vector channel emit
  byte-identically to before. This fixes the old `parse_generalized` "outermost axis = stops" bug (it split on
  commas first, so `steel gold copper` collapsed to one 3-vector stop instead of three scalar stops). Grammar is
  a **strict additive superset**, not a breaking change. Updated `test_record.py` (renamed the two `*_generalized`
  tests; added whitespace-vs-comma dispatch, lone-vector-stop, and stray-comma coverage — 34 record tests),
  DESIGN.md §8a, and `ROADMAP_records.md` §3.1 ("additive superset, not a breaking change"). 660 loom green.
- 2026-07-19: **Extended the binding model — materials-as-bundles + optional names (§3.3).** Two more locked
  grammar points from the user. (1) A **material is a parameterized bundle**: its free-input set is the union
  of its properties' free inputs, and applying it binds them across the whole bundle at the use site —
  `material = gold(u=v, a=1)` ≡ `gold(u=v a=1)` (arg list uses the same ladder, comma ≡ space); partial
  application leaves unbound inputs at their system defaults; positional `gold(v)` is allowed only for a
  single-free-input material (matching the shipped `material NAME(driver)` form). (2) **Property names are
  optional** — the leading type/slot keyword identifies the property; the quoted name is just an external
  dot-handle (`spectrum = …` anonymous vs `spectrum "color" = …` for `gold.color`). Written into
  `ROADMAP_records.md` new §3.3 + §3.2 naming note; folded into TODO §J3b item 3.
- 2026-07-19: **J3b started — delimiter-precedence-ladder parser (`loom/ladder.py`, item 2).** Built the pure
  parser for the generalized stop grammar locked in §3.1: `parse_ladder(str)` → nested `list`/`str` tree with
  whitespace = `×` (juxtaposition/vector), comma = `+` (outer level), brackets = parens; single-level groups
  unwrap (`[1 1 1]` ≡ `1 1 1`) and sum-of-products ≡ product-of-groups (`1 1 1, 2 2 2` ≡ `[1 1 1] [2 2 2]`).
  Parens `( )` are opaque atoms (not a delimiter), so `clamp(x,0,1)` stays one leaf. Plus `emit_ladder`
  (round-trips) and `shape` (rectangular dims, raises on ragged). 22 tests (`tests/test_ladder.py`), 649 loom
  green. Exported (`parse_ladder`/`emit_ladder`/`ladder_shape`); DESIGN.md §8a. **Fixed a spec slip**: §3.1/
  TODO had used parens for explicit grouping (`rgb (0 0 0)`) — corrected to brackets (`rgb [0 0 0]`), since
  `()` is reserved for expressions + the §3.2 application surface. Next J3b step: wire this into `Record` for
  arbitrary-arity channels (item 1).
- 2026-07-19: **Locked the generalized grammar's delimiter + binding model (§3.1/§3.2).** Design converged
  with the user on two points. (1) **Delimiter precedence ladder** replaces the earlier "interchangeable in
  any order" framing: whitespace binds like `×`, comma like `+`, brackets are parens — so `1 1 1, 2 2 2`
  parses as `(1·1·1)+(2·2·2)` and **structure is recoverable from delimiters alone**; the channel arity only
  *validates*. Position pins (`.2:0 0 0`) are an orthogonal `POS:` prefix. (2) **Uniform named-input binding**:
  a property is an expression over named inputs; system inputs (`a`,`u`,`v`,…) carry shading-point defaults;
  **nothing is closed** — any input is rebindable at the use site (`gold.reflect(a=x)` ≡ `gold.color(u=x)`),
  correcting the earlier "`.5*a` is closed" claim. Access is **continuous-only** (dropped the discrete `[i]`
  selector — a constant index is just a constant argument `prop(2)`, subsuming shipped `R.chan[i]`); `[…](u)`
  seals the array inside a function of `u` (purist reading) whereas bare `[…]` leaves the driver for the
  consumer. Written into `ROADMAP_records.md` §3.1 (ladder) + new §3.2 (binding/access/override) + §5, and
  TODO §J3b (now four generalizations). Target/loom-side; shipped ftrace keeps the two Stage-5a constant
  accessors + fixed per-hit/`t` scope model.
- 2026-07-19: **Captured the generalized stop grammar (§3.1).** User flagged that a generalized record lets
  a `D`-tuple channel (e.g. rgb) be authored with **interchangeable `[ ]` / `,` / whitespace** delimiters
  down the channel → stops → components hierarchy (`tint [rgb 0 0 0, 0 1 0, 1 1 1]`). Verified against the
  real tokenizer/parser: **current FTSL does NOT support this** — its tokenizer isn't comma-aware (a comma
  accretes into the preceding bareword) and the record parser makes every whitespace-word its own stop, so
  an rgb curve today is `spectrum:`-refs (one per stop), never inline triples. Documented the general form
  as the spec target in `ROADMAP_records.md` §3.1 + §5, folded it into TODO §J3b (item 2), and noted the
  scope boundary in `loom/record.py` (J3a mirrors current FTSL; the flexible grammar is J3b superset).
- 2026-07-19: **§J3a done.** Built `loom/record.py` — a loom twin of the FTSL parametric record
  (`Record`/`RecordChannel`/`RecordStop`). Emits the `NAME = range LO-HI [ … ]` block (wired into `Scene`
  before materials), parses one/every block back (`parse`/`parse_all`, comment-stripped), redistributes
  unpinned stops and numerically samples all-numeric scalar channels (nearest/linear/smooth = Fritsch–
  Carlson) exactly as ftrace's `recSampleScalar`. Round-trips every `scenes/_record_*.ftsl` fixture.
  Colour + expression stops re-emit faithfully but aren't evaluated (pattern VM → J3c). 23 new tests, 627
  loom green. Exported from `loom/__init__`; documented in loom `DESIGN.md` §8a. Next: J3b (N-D superset)
  or J3c (full-scene `.ftsl` parser) — or §J2 (placed isosurfaces + Room).
- 2026-07-19: **Generalized the record output-arity spec.** User observed `.ftsl` records already carry
  arbitrary per-row output dimensionality (an `rgb` channel is a series of 3-tuples living alongside a
  scalar reflectance/roughness curve). Formalized in `ROADMAP_records.md` §3: a channel outputs an
  arbitrary-arity `D`-tuple, each stop is `D` component programs, interpolation is per-component; the
  shipped `ChanKind{Scalar,Spectrum}` is the `D=1 (Linear)` / `D=3 (Colour)` realisation. §5 now separates
  the **not-deferred** output-arity generality from the still-deferred N-D *input* domain (loom-side, §J3b).
  Updated §J3 locked decisions + J3a/J3b accordingly (loom record twin is arbitrary-arity by construction).
- 2026-07-19: **§J1 done.** Grid out-of-domain policy `on_outside` = `clamp` (default, byte-identical) /
  `raise` / `wrap` (periodic, linear + cubic) added to `GridField`/`VecGridField` and the shared
  `_cell_base_frac`/`_catmull_rom_axis`/`_grid_weights`; FieldCurve now re-raises a dim-mismatch with its
  own context. `extrapolate` mode deferred. 604 loom tests green. Next candidates: §J2 (placed
  isosurfaces + Room) or §J3a (loom record type).
- 2026-07-19: **§I stereoscopic output complete** (I1/I2/I3) and committed (764e9b3), plus a pre-existing
  `-n` scientific-notation parse bug fixed (`-n 2e8` was truncating to 2 photons). Validated with a
  200M-photon red-cyan anaglyph of the Cornell box. Then **captured §J** (loom field/scene enhancements)
  from the design discussion: J1 Grid `on_outside` policy (small), J2 placed isosurfaces + Room/Group
  (medium), J3 port the FTSL parametric-record data structure into loom + make loom scene syntax
  `.ftsl`-round-trippable (large — the user's real ask). Nothing in §J started yet.
- 2026-07-18: file created; consolidated undone items from DESIGN.md, OSCILLATE_GRAMMAR.md, ROADMAP.md,
  ROADMAP_heroroom.md, and the just-designed camera-curve bridge (§A). Starting on item G1
  (`--raster-iso` passthrough — the trivial, zero-engine-change win).
- 2026-07-18: **G1 done.** `--raster-iso` flag threaded through `gyroid_nd`; verified end-to-end
  (coarse gyroid at res 40) + 268 loom tests green. Next: P1.1 (the `--oscillate` parser + model).
- 2026-07-18: **P1.1 done.** Standalone `--oscillate`/`--lock` grammar parser + `OscGroup` model +
  safe arithmetic evaluator in `gyroid_nd.py`; 17 parser unit tests; 285 loom tests green. No
  behavior wired yet (that's P1.2 — desugar `--transform` through the group model). Next: P1.2.
- 2026-07-18: **P1.2 done.** `transform_to_oscillate(...)` + `oscillate_spec(...)` desugaring bridge
  maps today's `--transform`/`--bloom`/`--bloom-amp`/`--tumble-*` to one canonical composite
  `OscGroup` (§3 migration map). Pure model — execution path untouched, all existing tests pass
  unchanged. 13 new tests; 298 loom tests green. Next: P1.3 (wire swinger axes freq/threshold/
  thickness/bloom to real behavior — the deterministic, non-RNG-sensitive half).
- 2026-07-18: **P1.3 foundation (partial).** Added per-axis `Variant.bloom_amps` + `_swing_amp()`;
  the three swinger functions read a per-axis amp override, falling back to the shared `bloom_amp`
  (empty dict ⇒ byte-identical to the legacy path). 2 new tests; 300 loom green.
- 2026-07-18: **P1.3 done.** Wired the `--oscillate`/`--lock` flags via an idempotent
  `resolve_oscillate(args)` that maps the group model onto the canonical transform/bloom/tumble
  fields — the inverse of `transform_to_oscillate`, so `pick_variant` gets no new path. `--transform`
  default → None for mutual-exclusion; conflict + "not yet wired" (rate/phase, bare dims) guards.
  Validated three ways: 14 field-expression-equivalence/guard tests, a real `--oscillate bloom,freq`
  render through the full CLI→ftrace pipeline (live preview), and a byte-identical `.ftsl` diff vs
  the legacy `--transform` form (incl. `1.5*freq`). 314 loom green. (Corrected an earlier bad call:
  the live-preview rule never blocked rendering-to-validate — CLAUDE.md reworded to say so.)
  Next: P1.4 (wire winder `rate`/`phase` + bare-dim axes — the RNG-order-sensitive winding piece).
- 2026-07-18: **P1.4 done.** Wired the winder axes. `resolve_oscillate` now emits `osc_dim_windings`
  (bare dim index → exact `round(amp*rate)` winding, forced on), `osc_max_winding` (a motion group's
  `rate` = the ceiling of the varied `1..N` cycle, per the user's option-2 call — consistent with a
  lone dim's exact rate), and `osc_phase` (a constant radians offset on the shared winding clock);
  the picker applies windings after the RNG cycle (no draw consumed) and threads phase into
  `field_expr`/`_tumbled_directions`, all no-ops on the legacy path so existing variants stay
  bit-identical. Single shared clock ⇒ conflicting motion rates/phases and swinger `rate`/`phase`
  are rejected. 12 new tests + a real `--oscillate drift rate 3` video (CLI→ftrace). 324 loom green.
  Next: P1.5 (flip default — `--oscillate` primary, `--transform` deprecation notice + docs).
- 2026-07-18: **P1.5 done.** `--oscillate` is now the single documented motion surface.
  `--transform` + its `--bloom`/`--bloom-amp`/`--tumble-*` satellites are hidden from `--help`
  (`argparse.SUPPRESS`) but stay fully supported; explicitly passing `--transform` prints a plain
  one-line deprecation note (checked before `resolve_oscillate` synthesizes it). Migrated the
  module docstring examples/prose + epilog quickstart to the grammar (`--oscillate bloom,1.5*freq`,
  `--oscillate 0.3*tumble`, `--oscillate tumble --lock 0,1`) and rewrote the test suite's incidental
  `--transform` setup usages to `--oscillate` (a one-shot `scraps/convert_transform_to_oscillate.py`
  did 23; 2 hand edits for the dynamic `tr` loop + `base` list). The two `bloom_amps`
  legacy-representation tests and the deliberate desugaring/equivalence references stay on
  `--transform` by design. Default motion (neither flag) is still `drift`. 324 loom green.
  Phase 1 complete — next: P2.1 (`--couple` cluster command).
- 2026-07-18: **P2.1 done.** `--couple CLUSTER CLUSTER…` — the spatial (field) counterpart of
  `--oscillate`: each cluster is comma-joined dims sharing sin*cos terms; spaces separate disjoint
  clusters (a dim in ≤1). Per-cluster `:full`/`:cyclic` tag over a global `--couple-scheme` default.
  `parse_couple`/`resolve_couple` build `couple_clusters` + a forced-on `couple_axes` set fed into
  `forced_on`/`max_forced_axis` exactly like a `--pair …:on` endpoint (no new RNG draws).
  `coupling_pairs()` refactored around a shared `_scheme_edges(dims, scheme)` helper — cluster path
  emits each cluster's ring/clique edges (over its oscillating members) in CLI order; empty
  `couple_clusters` falls through to the legacy `--coupling`/`--pair` base-graph path bit-identically.
  Kept `--coupling`/`--pair` on their own resolution path (they act over the post-RNG active set,
  `--couple` names explicit dims at parse time — no clean desugar), so `--couple` is mutually
  exclusive with a non-default `--coupling` / any `--pair`. `coupling_desc` summarizes clusters;
  primitive warning lists `--couple`; docstring/epilog/help + OSCILLATE_GRAMMAR.md §6 updated.
  11 new tests, 335 loom green. Next: Phase 3 (P3.1 surface library) or another TODO track.
- 2026-07-18: **P3.1 done.** Per-surface shape-param metadata table in `loom/pov.py` (Python side —
  it feeds the future `--surface-help`; the C header stays arity-only for the VM). `_AUTHORED_PARAMS`
  hand-documents real `(name, description, default, (lo, hi))` tuples for the well-understood /
  N-D-core shapes and the 0-param spherical/noise helpers; `_generic_params(n)` supplies honest
  `p0..p{n-1}` placeholders for every other `POV_FUNCS` entry. `POV_PARAMS` is built by comprehension
  so its per-function count always equals `arity−3`; `pov_params(name)` returns a defensive copy and
  raises on unknown names. Exported from `loom/__init__`. 6 new tests mirror the arity drift-guard
  discipline (set-equality with `POV_FUNCS`, exact `arity−3` count, valid+unique axis names,
  default∈[lo,hi] with lo<hi, authored spot-checks, unknown-name reject, copy-safety). 349 loom green.
  Next: P3.2 (`--list-surfaces` / `--surface-help NAME` discovery commands).
- 2026-07-18: **P3.2 done.** Surface-library discovery commands in `gyroid_nd.py`. A catalog
  (`_TPMS_CATALOG` for the 4 periodic minimal-surface families + `POV_FUNCS` for the 78 POV builtins)
  is grouped by N-D honesty class via `surface_group()` → periodic / nd_pov / affine_pov.
  `--list-surfaces` prints the whole library (82 surfaces) with per-surface shape-param counts and
  `[nd]`/`[loop]` tags; `--surface-help NAME` prints one surface's shape params (name, meaning,
  default, range from `pov_params`) or the shared-axis note for a param-free TPMS, resolving the
  `schwarz_p`→`primitive` alias and raising on unknowns. Both early-exit before any generation and
  emit ASCII-only text (Windows-console-safe). Cross-referenced from `--surface` help + epilog.
  11 new tests, 360 loom green. Next: P3.3 (widen `--surface` to the full library — the design-heavy
  step: map POV builtins into the N-D slice machinery with the N-D + seamless-motion guards).
- 2026-07-18: **P3.2b done.** Generalized the swinger envelope to carry an independent clock,
  making swingers uniform with winders/bloom (the user's insight: there was no good reason for
  freq/threshold/thickness to lack a rate/phase once seamless-looping was demoted from a hard
  requirement). Each swinger's bloom is now `w(t) = 0.5·(1 − cos(2π·rate·t + phase))` via the new
  `_bloom_env_p(v, key, t)`, reading `Variant.bloom_rates`/`bloom_phases` (keyed `"dims"` for the
  dimensional crossfade, own name for the scalar swingers). `resolve_oscillate` records the swinger
  group's `rate`/`phase` instead of rejecting them. Default rate 1 / phase 0 is byte-for-byte the
  legacy fixed `sin²(πt)` envelope (the `--transform` path leaves both tables empty, so all existing
  seeds reproduce exactly). An integer rate loops seamlessly for any phase; a non-integer rate pulses
  faster but breaks the loop, so `main()` prints a one-line "won't loop seamlessly" note. 6 new tests
  (rate stored + peaks at t=¼,¾; default byte-identity; integer-rate seamless; phase flips the bump
  but still loops; `bloom`→`dims` keying; non-integer warning via `main`), 365 loom green.
  Next: P3.3.
- 2026-07-18: **P3.4 + P3.5 done** (see §B entries) — true N-D forms for the 9 generalizable POV solids,
  then ordered/overlapping tumble via `--tumble-sequence`. 515 loom green; both render-validated.
- 2026-07-18: **Housekeeping.** Verified **C1 (mode-M final gather) was already done** (2026-07-14,
  `-pmfg`) and marked it off (GPU-port of the sub-ray pass remains a lesser follow-up). Added a
  **BLOCKED-on-user-sign-off** gate to §D (no D2/D3 verify renders until the user approves the room +
  flyby in the rasterizer camera_curve editor). Captured three user-proposed features as §E: E1
  UV-space procedural skin (ftrace, small), E2 general N-D-curve→scene-variable animation via the
  rasterizer curve editor (loom+ftrace, large, extends §A), E3 loom procedural audio (one buffer
  back-end, per-tick as a thin front-end — decided).
- 2026-07-18: **E3 done.** New `loom/audio.py` → `SampleBuffer`: one per-channel `array('d')`
  back-end as the single source of truth; random-access `add`/`set`/`mul`/`get` (out-of-range
  ignored), range ops (`add_range` overlap-add, `mul_range`, `fade`, `mix` with channel routing),
  the thin per-tick cursor (`emit_next` ≡ `buf[cursor]+=v; cursor+=1`, `seek`/`tell`), producers
  `render_fn(fn(i,t_sec))` + `render_signal(Signal)` (audio-rate, seamless-loop aware, add/set/mul
  modes), `peak`/`rms`, and one `finalize()` (gain→normalize→dither→clip→PCM→WAV, 16/24-bit via
  stdlib `wave`). Exported from `loom`. 30 new tests (WAV round-trips, dither determinism, cursor≡add,
  seamless-loop render); 545 loom green; real 220+660 Hz WAV smoke-validated. Next: E1 (UV-space
  procedural skin).
- 2026-07-18: **E1 done** (see §E1) — UV-space procedural color skin, option b (three r/g/b
  sub-expressions baked to a linear RGB grid at load, then run through the whole existing texture
  pipeline; zero GPU changes). `src/ftsl.h` `addTexture` `rgb`-branch + `compilePatternExpr`/`patternEval`
  bake; loom `ProcTexture`/`func_skin`; `scenes/procskin.ftsl` render-validated (all orientation checks
  pass). 5 new loom tests, 550 loom green. Next: G3 (PatOp::MatMulAdd matrix intrinsic).
- 2026-07-18: **C8 done.** FBX mesh import via vendored ufbx (single-file, confined to `fbx_load.cpp`);
  `mesh { file "*.fbx" }` triangulates + bakes world positions; no unit conversion (raw cm coords, size
  via `scale`). `scenes/fbxcube.ftsl` validated in raster + forward mode B. Geometry-only (no FBX
  materials/skins/anim) — logged in known-issues. Committed 3d6dd65.
- 2026-07-18: **G2 done.** `-raster-gpu`: GPU deterministic primary-ray isosurface preview (`kIsoPreview`
  in `render_cuda.cu`, reusing `closestHit`/`buildUpload` + shared `raster::exposeAndEncode`); no
  tessellation. main.cpp falls back to CPU raster on unsupported configs; `gyroid_nd --raster-gpu` routes
  video frames through it. Fixed a vertical-flip bug (dGenRay py=0 is image bottom, accum row 0 is top).
  Validated on `scenes/implicit.ftsl` + a gyroid video. Next: G3 (PatOp::MatMulAdd — needs a design call).
- 2026-07-19: **§A / M13 done + committed (2c2ae94)** — camera-curve two-axis orientation (ftrace
  `fwd_at`/`up_at`/`frame`/`fwd_frame`/`up_frame` + RMF double-reflection twist; back-compat bit-identical)
  and loom `CameraCurve` element (6 new emit tests). Docs updated (README, FTSL.md §15.3, DESIGN.md M13).
- 2026-07-19: **Design-captured a large batch of user-brainstormed features into new TODO sections**
  (no code yet — all design, awaiting scheduling/priority). **§F loom native viewer** (F1–F7): native
  on ftrace's `-raster-gpu` (not WebGL — chosen because the goal is fast isosurface *modification*, which
  the device sphere-tracer already does with no tessellation + exact field parity) + ImGui/ImPlot/imnodes;
  `build(clock, **params)->Scene` load contract (not a module `scene`); N-D curve 3-D view w/ view-vs-
  recompute rotation semantics + off-axis stereo (wall/cross-eyed + anaglyph); scroll-locked per-dim +
  tracked-channel strip charts; SweptMesh tessellated view w/ textures + decoupled latest-wins re-tess;
  modulator-DAG panel (`walk()` + edge param labels); scatter/grid inspect w/ channel-selector coloring;
  isosurfaces via `-raster-gpu` raymarch (primary) + MC-mesh fallback. Concrete realization of §E2.
  **§H loom multi-valued fields** (H1–H4): vector-valued Grid/ScatterField (record channel model — user's
  question answered YES); grid multilinear/tricubic; **RBF scatter interpolation** (thin-plate default,
  multiquadric/Gaussian/Wendland options, `scipy.RBFInterpolator` multi-RHS, convex-hull-overshoot caveat);
  field-sampled curve returning (coords, {channel:value}). **§I ftrace stereoscopic output** (I1–I3):
  off-axis (asymmetric-frustum, parallel) stereo — not toe-in; wall/cross-eyed side-by-side + **Dubois**
  anaglyph (red-cyan default, green-magenta option); `-stereo`/`-eye-sep`/`-view-dist`/`-dpi` (auto-detect)
  → physical baseline+convergence; reuses M13 right axis; stills + movies. **§B/G5** added: make
  `-raster-gpu`/`kIsoPreview` shade textures (image + formula) — port raster texture sampling into the
  device kernel (already has `DHit.u/v/p`); user-requested, also in known-issues.md; prereq for F4/F7.
- 2026-07-19: **G5 done.** `-raster-gpu`/`kIsoPreview` now shades **textures** (was flat per-material
  albedo). Ported the CPU rasterizer's textured-preview path (`Texture::sampleRgb`/`sampleRgbTriplanar`)
  into the kernel: a shared flattened linear-RGB texel array + per-texture `DPTex` meta + per-material
  `matTex`/`matTri` binding (mirroring `raster.h` buildScene's rule) upload alongside `matCol`; the kernel
  samples the hit `(u,v)` or world triplanar and replaces the flat albedo for non-emitter hits. One path
  covers **image** skins and E1 **formula** skins (they bake to `rgb` at load); flat/no-texture hits are
  byte-unchanged (`matTex==-1` skips the sampler). Device sampler is a private twin of `raster_cuda.cu`'s
  (separate TU). Validated: `-raster-gpu` matches CPU `-raster` within edge-coverage tolerance on
  procskin.ftsl (formula, red rises left→right = r=u), textured.ftsl (image checker), triplanar.ftsl
  (mean channel diff ~0.03/255, ~1033 shared box-edge px); flat implicit.ftsl unchanged; back-wall
  spatial variance confirms the skins actually sample. Resolves the known-issues tech-debt entry.
- 2026-07-19: **H1 done.** loom fields generalized to vector values. `Grid`/`Scatter` infer
  `value_dim`/`is_vector` from their stored values (all-scalar vs all-vector, mixing rejected) and take an
  optional `channels=` name list (validated for length/uniqueness, resolved by `channel_index(name|idx)`
  incl. negative indices). New `VecGridField`/`VecScatterField` (both `VecSignal`s, LoopCurve-style
  `_VecFieldComponent` views) interpolate every channel with the **same domain weights**, computed once
  per frame. The weight kernels `_grid_weights` (N-linear corners) and `_shepard_weights` (inverse-
  distance) are factored out and shared with the scalar `GridField`/`ScatterField`, so a vector field's
  `.channel(c)` is bit-for-bit the scalar field over that channel. Scalar fields stay the 1-channel case
  and now reject a vector dataset (pointing at the Vec* class). 11 new tests (`tests/test_vecfields.py`);
  573 loom tests green. Next in §H: H2 (tricubic/Catmull-Rom grid option) then H3 (RBF scatter, needs
  scipy).
- 2026-07-19: **H2 done.** grid fields gained an `interp=` kernel selector. `interp="linear"` (default)
  is unchanged N-linear; `interp="cubic"` is separable **Catmull-Rom** (tricubic in 3-D), built as a
  tensor product of per-axis `_catmull_rom_axis` contributions via a new `cubic` flag on `_grid_weights`
  — so `VecGridField` cubic still computes the geometric taps once and blends every channel. Boundary
  phantom points are **linearly extrapolated** (`p[-1]=2p0−p1`) rather than edge-clamped, which keeps
  cubic reproducing linear ramps exactly right up to the boundary (verified); thin axes (< 3 samples)
  fall back to linear. 8 new tests (`tests/test_gridinterp.py`); 581 loom tests green. Next in §H: H3
  (RBF scatter — needs scipy) then H4 (field-sampled curve).
- 2026-07-19: **H3 done.** scatter fields gained a radial-basis-function backend beside Shepard.
  `RbfScatterField` (scalar) and `VecRbfScatterField` (vector) wrap `scipy.interpolate.RBFInterpolator`
  behind a lazy import (`_require_scipy` — scipy stays an *optional* loom dep, clear error if absent). A
  per-field `_RbfEngine` rebuilds the interpolator at most once per frame (positions/values are
  animatable, so the kernel factorization is frame-dependent) and evaluates the query; a vector scatter
  is a single interpolator with a multi-column RHS, so all channels come from one factorization. Default
  kernel is the parameter-free thin_plate_spline (reproduces linear + exact at samples with smoothing=0);
  scipy's other kernels are selectable. Out-of-convex-hull queries are guarded by
  `on_outside="clamp"|"raise"|"extrapolate"` (hull via scipy Delaunay, 1-D/bbox fallbacks) — dropped the
  planned "nan" flag mode because loom's Signal.at() forbids non-finite values (proper fix: honor the
  invariant, "flag" == raise). Wendland isn't offered (not a scipy kernel); `neighbors=` covers large
  sets. 8 new tests (`tests/test_rbf.py`); 589 loom tests green. Perf caveat logged in known-issues
  (per-frame refactor can't be reused across frames via scipy's API). Next in §H: H4 (field-sampled
  curve) — closes §H.
- 2026-07-19: **H4 done — §H complete.** `FieldCurve` bundles a curve + a field so a single object gives
  both the spatial coordinates and the interpolated `{channel: value}` at a progression index. It takes a
  `PointPath` (built into a `LoopCurve` over `u`) or a ready position `VecSignal`, plus a *field builder*
  callable `q -> field` — so any H1–H3 field (scalar/vector, linear/cubic grid, Shepard/RBF scatter)
  drops in. `.position`, `.value`, and `.channel(name|idx)` are DAG nodes (drive scene variables, walked
  by cycle detection through the dataset + path); `.sample(u, clock)` polls at an explicit u via a private
  probe field over a `_MutableVec` query (uncached so the mutated query is honored), returning `(coords,
  {channel: value})` with channel names from the dataset when present. This is the object §E2 (curve
  vars → scene vars) and §F6 (viewer inspection) build on. 8 new tests (`tests/test_fieldcurve.py`); 597
  loom tests green. §H (multi-valued fields + interpolation + field-sampled curve) is now fully done.
- 2026-07-19: **§I done — ftrace stereoscopic / anaglyph output.** Off-axis (asymmetric-frustum) 3-D output
  for stills and movies. Engine: `Camera::frustumShiftX` (a normalised horizontal shear) added to camera.h
  and applied consistently in `project()`/`genRay()`/`lensImage()` (rectilinear only) plus the GPU `DCamera`
  mirror in render_cuda.cu (photon-splat project, backward `dGenRay`, BDPT camera subpath, lensImage) — 0 by
  default so every non-stereo render stays byte-identical. `-stereo <sbs|cross|anaglyph|anaglyph-gm>` expands
  each rendered camera into a Left/Right eye pair (two parallel cameras offset ±b/2 along the right axis `u`,
  each sheared so the convergence plane has zero parallax — no toe-in, no vertical parallax), sharing one
  exposure group so L/R (and every frame of an exposure-locked path) tone-map identically. Each eye rides the
  full existing pipeline (checkpoints/budgets/GPU/live window); a post-pass `stereoComposite()` fuses the two
  eye PNGs — side-by-side wall-eyed/cross-eyed or Dubois least-squares anaglyph (red-cyan default,
  green-magenta option) — into the `-o` file and deletes the intermediates (kept with `-stereo-keep-eyes`).
  Physical geometry: `-eye-sep`/`-view-dist`/`-dpi`/`-convergence` set screen width `W` (from dpi or
  view-dist×FOV), shear `S = eyeSep/W` (infinity at interocular ⇒ parallel gaze), baseline `b = 2·C·tanHalfX·S`
  (so camera:subject ratio = eye:screen ratio). README updated. Next §I candidates exhausted; roadmap: §F
  (loom viewer) or §C2–C6 (VDB/mesh) remain, both large and unstarted.
