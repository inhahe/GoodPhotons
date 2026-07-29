# TODO — consolidated undone work

Single source of truth for everything planned-but-not-done, pulled together from the
project's scattered plan files. Check items off (`[x]`) as they land, and mirror the
status back into the originating file (`DESIGN.md`, `ROADMAP.md`, `OSCILLATE_GRAMMAR.md`,
`ROADMAP_heroroom.md`) when a whole section closes.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.
Origin tags point at the authoritative design text for each item.

> **Looking for what's still left? Read [`open-work.md`](open-work.md) first.**
> This file has become mostly a *record of what shipped*: most entries are long prose blocks
> whose opening paragraph reads like a plan but whose later `**STATUS (date) … DONE**`
> sub-paragraph says it landed, so an entry must be read to the end before it can be called
> open. `open-work.md` is the actionable extract — the handful of genuinely-undone items, each
> pointing back here for the full design text. **This file stays authoritative for design and
> history; that one is authoritative for "what's next".** When an item lands or a new open item
> appears, update both.

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
    * **STATUS (2026-07-26): the N-D GRID datatype + sampler is DONE in ftrace (VERSION 0.71.0).** Increment 3's
      regular-lattice half shipped ahead of the axis-tuple *syntax* (increment 2) by taking the call form the
      lexer already supports — `grid:<name>(c0, …)`, the `tex:<name>(u,v)` precedent — so the runtime exists before
      the sugar. What landed:
      - `src/pattern.h`: `PatOp::Grid`, `PatGrid` (ndim ≤ `PAT_GRID_MAX_DIM` = 4, `shape`/`lo`/`hi`/`outside` +
        `off`/`count` into a shared pool), `PatGridOutside {Clamp, Wrap, Extrapolate}`, and the shared
        `__host__ __device__` `patGridCellFrac` / `patGridSample` (separable N-linear over 2^ndim corners, C order
        with axis 0 outermost). Samples are addressed by **offset into one flat pool**, never by pointer, so the
        pool can grow and the header uploads to the GPU verbatim.
      - `PatGridScope` + tokenizer/compiler support: `grid:` scans as ONE identifier like `tex:`, but its **arity is
        the grid's own `ndim`**, resolved at compile time — the first op whose argument count is not a property of
        the function name. Wrong arity, unknown name, an uncalled `grid:<name>` and a bare `grid` are all compile
        errors.
      - `Scene::grids` / `Scene::gridPool`, published via `bindPatScene` (which now also covers `bindPatTex`, so a
        future table can't be missed at one of the hand-built-PatCtx sites).
      - `src/ftsl.h`: the `grid "name" { shape … lo … hi … outside … data { … } }` element, loaded in a new **Pass
        1a** (before textures, so a procedural `texture { rgb "…" }` can sample one). `lo`/`hi` implement loom's
        `_resolve_lo`/`_resolve_hi` defaults exactly (absent-`hi` ⇒ unit-spacing index lattice; scalar-`hi` ⇒
        isotropic lattice from axis 0). `data { … }` uses the flat-word brace body the `palette {}` precedent
        already parses, so the **shared grammar needed no change**.
      - `src/render_cuda.cu`: `DScene.grids/gridPool` upload verbatim and `dPatternEval` gained a `PatOp::Grid` case
        that calls the *same* `patGridSample`. The old `(const DTexture* tex, int nTex)` parameter pair became a
        `DPatEnv` bundle across all 9 call sites + both forward declarations, so the next table costs one field
        instead of nine edits.
      - `ftrace -checkgrid`: deterministic self-test — exact sample recovery, C-order flattening, 1-D…4-D
        multilinear exactness (worst error 1e-8, float-pool storage), all three `outside` policies, and the
        compile/arity/scope rules. Cross-backend: `scraps/grid_test.ftsl` at 16384 spp agrees CPU↔GPU to **0.003 %**
        mean (RMS 0.99/255 — pure MC noise).
      *(Was: "still open here — the `[[…][…]](u,v)` authoring sugar (increment 2) reaching this same datatype."
      **That shipped in v0.73.0** — see the increment-2 STATUS block below. The only piece of the sample call
      still open is the keyword-rebind form `(a=u)`; it is listed in `open-work.md`.)*
      (Several names above were generalized by the scatter port immediately below — `PAT_GRID_MAX_DIM` →
      `PAT_ND_MAX_DIM`, `PatGridScope` → `PatTableScope`, `Scene::gridPool` → `Scene::dataPool`.)
    * **STATUS (2026-07-27): increment 3's RAGGED half — the N-D `scatter` datatype + sampler — is DONE
      (VERSION 0.72.0).** Ported from loom's `data.Scatter` / `interp.ScatterField`: N values at arbitrary
      positions, blended by **Shepard inverse-distance weighting** `w_i = 1/(d²)^(power/2)`, with a coincident
      sample (`d² ≤ eps`) returned *exactly* — which is both the correct limit and the singularity guard.
      Defaults `power = 2` (the cheap `1/d²` path), `eps = 1e-9`. loom's vector-valued `VecScatterField` and its
      `_local_query` transform were deliberately **not** ported (ftrace's tables are scalar and unit-agnostic).
      The port was done by **generalizing the grid machinery rather than cloning it**, on the same reasoning as
      the earlier `DPatEnv` bundling: adding the *next* datatype should cost an enumerator, not a new parameter
      on `compilePatternExpr` and edits at all its call sites. Concretely:
      - `src/pattern.h`: `PatOp::Scatter`, `PatScatter` (`ndim`/`count`/`off` into the shared pool + `power`/`eps`),
        and the shared `__host__ __device__` `patScatterSample`. Samples are **interleaved** at stride `ndim+1`
        (`p0 … p_{ndim-1}, value`) — one `data { … }` list, because a scatter's positions and values are not
        separable the way a lattice's are. `PatGridScope` became `PatTableScope` + `PatTableKind {Grid, Scatter}`
        and `Tok::gridId/gridDim` became `tableId/tableDim`, so one kind-dispatching resolver serves both
        namespaces (and they *are* separate namespaces: `grid:pts(…)` cannot see a scatter, and vice versa —
        both directions are covered by the self-tests' must-reject lists).
      - `Scene::gridPool` became **`Scene::dataPool` — ONE flat float pool shared by both datatypes**, so a scene
        costs exactly one GPU allocation for its tables however many it declares, of either kind.
      - `src/ftsl.h`: the `scatter "name" { dim … power … eps … data { … } }` element, loaded in the same Pass 1a
        as grids; again **no grammar change** was needed (verified by grep: `ftsl_scene.epeg` names neither
        `grid` nor `scatter` — `plain_header` + the flat-word brace body already cover both).
      - `src/render_cuda.cu`: `DScene`/`DPatEnv` gained the scatter table beside the grid one and share the
        renamed pool; `dPatternEval` calls the *same* `patScatterSample`.
      - `ftrace -checkscatter`: deterministic self-test built on properties that are analytically independent of
        the implementation — exact reproduction at every sample, 1-D…4-D partition of unity, midpoint == plain
        mean by symmetry at two powers, the closed-form two-sample weight `(1-q)^p/(q^p+(1-q)^p)` at four powers
        plus a "higher power sharpens" monotonicity check, and the far-field → mean limit. Worst error 1.4e-7
        (float-pool storage). Plus the compile/arity/scope/namespace rules. All 11 self-tests pass.
      - Validation: `scenes/pattern_scatter.ftsl` (five strips: the S-curve two-sample case, the same at power 8
        collapsing to a step, four irregular 1-D stops showing the plateau at each, and a 5-point 2-D field at
        powers 2 and 1). Rendered albedo — recovered by dividing out illumination with a flat-0.75 twin render —
        tracks the analytic IDW at all 81 probe points; CPU↔GPU agree to **0.18/255 mean, 0.81/255 worst** (pure
        MC noise).
    * **STATUS (2026-07-26): increment 1 of 3 DONE — the shared grammar + loom's canonical tree parse the call.**
      `tools/loom/loom/grammar/ftsl.epeg` now carries the axis tuple, and `loom/grammar/values.py` normalizes it
      (11 new cases in `tests/test_grammar_values.py`; suite 1072 → 1083):
      ```
      vpiece    = colour_tag | vsampled | vnums
      vsampled  = vbracket axistuple? | NAME axistuple
      axistuple = '(' arg (',' arg)* ')'
      arg       = NAME '=' coord | coord
      coord     = vsampled | NAME | NUMBER
      ```
      Two deliberate refinements of the sketch below: the tuple hangs off a **piece**, not off the whole `value`, so
      a call composes *inside* a bigger value (`[[0 1](u), [2 3](v)]`); and a bare `NAME` is a value **only when
      called**, so adding this cannot make a stray bareword parse as a value. Canonical tree gains `Call(target,
      args)` / `Arg(formal, driver)` (`target` = an array literal or a called NAME; `driver` = a name, a constant,
      or a nested `Call`). Argument *order* is intentionally not a grammar rule — the normalizer enforces
      positionals-before-keywords and no-duplicate-formals so the error can name the axis. `as_sampled()` is where
      the **unsaturated** error lives (a bare array reaching a field that samples).
      (Increment 3 is now complete on both halves — the regular `grid` in 0.71.0 and the ragged `scatter` in
      0.72.0; see the two STATUS blocks above.)
    * **STATUS (2026-07-27): increment 2 DONE — ftrace parses and RENDERS `[0 1](u)`. Shipped as 0.73.0.**
      Inline array literals now work at every value site that accepts a `pattern:<name>`, in **both** front ends.
      - **No lexer change was needed after all** — the "emit `(` as a delimiter only after a `]`" plan below was
        dropped, because it is context-sensitive and therefore inexpressible in the shared grammar's regex
        auto-lexer, and because it turned out to be unnecessary. Instead the grammar gained a context-*free*
        terminal `PARENWORD = /\([^ \t\r\n{}\[\]#"]*\)/` — a token that is *wholly* parenthesised. Longest-match
        keeps every expression safe: `(a+b)*c` is 7 chars of `WORD` against only 5 of `PARENWORD`, so `WORD` wins.
        ftrace's legacy tokenizer needs **zero** changes: a call arrives as an ordinary bareword that happens to
        start `(` and end `)`, which is exactly what `Parser::takeAxisTuple` tests for.
      - **One production, both meanings.** `selector = '[' sel_item* ']' axistuple?` (with `sel_item = NEWLINE |
        sub_array | sel_word`) covers *both* jobs of `[ … ]` at a value site — the record stop selector
        `REC.chan[2]` and an array literal — so exactly one production starts with `[` there and the grammar stays
        unambiguous. `NEWLINE` as a `sel_item` lets a big literal be laid out over several lines.
      - **Neither parser decides what the brackets mean.** Both collect the raw shape into the new
        `ftsl::BrItem` tree (`ftsl::ArrayLit` = items + call text + line) and hand it to the single shared
        `ftsl::applyBracketGroup`, which applies the five-rule decision (call ⇒ array; else nesting ⇒ unsaturated
        array; else a dotted/override predecessor ⇒ stop-selector fold; else all-numeric with nothing before ⇒
        unsaturated array; else drop). Keeping that decision in one place is what stops the two front ends
        drifting on the one genuinely ambiguous syntax. `diff_value` now compares `Value::array` too.
      - **The literal is pure sugar.** `Builder::desugarArrays` (a pre-pass run immediately before Pass 1a) turns
        each literal into an anonymous `grid "__arrN"` + `pattern "__arrN" { expr "grid:__arrN<call>" }` appended
        to the block list, and rewrites the site to `pattern:__arrN` — so every slot that already takes a pattern
        works with **no per-site change**. Nesting is the shape (C order, axis 0 outermost, rectangular enforced);
        the domain is the **unit box** per axis, deliberately unlike the `grid` element's index-lattice default
        (an inline literal has no domain of its own and is read at normalized `u`/`v`).
      - **Errors name what the author wrote**, never the generated `__arrN`: unsaturated (with the "no spaces
        inside the parentheses" rule spelled out), ragged, mixed numbers/groups, non-numeric entry, coordinate
        count vs nesting level, literal-follows-another-token, and >4 nested axes.
      - **Validation:** the corpus sweep is unchanged at ok=362 / mismatch=0 (both front ends still agree on all
        367 authored scenes), all 11 self-tests pass, and — the decisive one — `scenes/pattern_array.ftsl` and a
        hand-written `grid` + `pattern` twin (`scraps/array_explicit.ftsl`) render **bit-for-bit identically**,
        which proves the desugar is exactly the sugar it claims to be. A flat-0.75-twin albedo read-out
        (`scraps/array_profile.py`) additionally confirms the five strips track the analytic N-linear values.
      - **Deferred:** the keyword-rebind form `(a=u)` (formals don't exist yet — it currently lexes as a call and
        would fail in the expression compiler), and `NAME axistuple` (`ramp(u)`), which needs no work because
        ftrace's expression evaluator already reads `name(args)` as a call.
        *(Both clauses are now resolved — and the second one was wrong. See the increment-2 remainder below.)*
    * **STATUS (2026-07-28): increment-2 REMAINDER DONE — formals and the keyword rebind. Shipped as 0.91.0.**
      The deferral above was **stale in the more interesting direction**: the machinery it was waiting on ("formals
      don't exist yet") shipped as the §3.3 material bundles (v0.87.0) and §3.2 per-property access (v0.89.0), and
      it turned out that the *feature* the ADDENDUM asks for had come along with it, unnoticed and unpinned.
      `[0 1](a)` already compiled to a program with a free `a`, and `mat(a=u)` / `src.reflect(a=u)` /
      `src.reflect(u)` already rebound it. What this arm actually did was **pin the semantics, close the one
      genuinely open spelling, and stop the whole thing regressing silently**:
      - **The semantics TODO asked to pin — "binding site vs literal coordinate source" — resolves to *binding
        site*, and for a reason that needed no new code.** A material application substitutes ANY input name
        (`patternSubstitute`), so every driver name in a literal's tuple is rebindable; a literal's "formals" are
        simply the driver names it wrote, which is exactly what the ADDENDUM's `(u=a, v=x)` example rebinds.
        Naming `a` is not a special construct — it is an ordinary coordinate that happens to name the one input
        with no per-hit intrinsic, which is what lets it survive to the use site.
      - **The one open spelling is REFUSED, loudly.** `formal=driver` *inside a literal's own call* (`[0 1](a=u)`)
        has no formal to bind: an inline literal's axes are anonymous and positional, and there is no second
        namespace. Honouring it would have to invent a per-material default for `a` — which two literals in one
        material could contradict — so it is a load error naming BOTH escapes (`[0 1](u)` to spend the axis,
        `[0 1](a)` + `material mat(a=u)` to defer it). Previously it reached the pattern lexer and died as
        `unexpected character '='`. `desugarOne` now splits the call properly (`splitCallArgs`, sharing
        `parseBindArgs`'s "a top-level `=` is unambiguous" rule) and also names an empty axis by index.
      - **The unsaturated error now names the deferral route** — the ADDENDUM's "user-side, axis left open" case.
        ftrace has no value site that can hold an unsaturated array, so leaving an axis open is spelled `(a)`,
        not by omitting the call; the message says so instead of only offering `(u)`.
      - **Generated blocks are re-attributed** (`Builder::genSite_` / `genWho`): any error inside the anonymous
        `grid`/`pattern` a literal desugars to now names the author's site and the literal's call text. The
        increment-2 bullet above claimed errors "never name the generated `__arrN`", but that only held for the
        checks `desugarOne` did itself — a bad *coordinate* (`[0 1](nope)`) reported `pattern '__arr0'`.
      - **loom twin:** `values.py::_check_args` gained `literal_target`, refusing the same spelling with the same
        two escapes. Without it loom would happily normalize `[0 1](a=u)` into a `Call` and emit a scene ftrace
        rejects. Three existing tests were retargeted from `[0 1](…)` to `ramp(…)` (a NAME target, where a keyword
        rebind IS meaningful) plus one new refusal test; suite 1210 → 1211.
      - **`-checkarray`** (`checkArray()` in `main.cpp`) pins it deterministically against independently authored
        twins at five probe points chosen so `u != v` (a rebind test on the diagonal passes for the wrong reason):
        the three bind spellings ≡ the inline literal, a driver *expression* ≡ the same expression inline, the
        geometry-field use site ≡ the property-reference one, and the 2-D simultaneous swap ≡ the transposed
        literal **with a negative twin** proving the swap is not a no-op. Plus explicit **non-vacuity** checks —
        a `PatCtx` without the grid pool bound makes `PatOp::Grid` return 0.0, so every identity would otherwise
        compare 0 == 0 and pass. Plus eight refusals, each pinning its message.
      - **Validation:** all 15 self-tests PASS, all 85 scenes parse, the grammar corpus sweep is 85/85, loom is
        1211/1211, and `scenes/_array_formal.ftsl` renders the three identity rows (flat camera-facing quad tiles,
        for the `_material_bind.ftsl` reasons).
      - **Still open, discovered here** (logged in `open-work.md`): a literal cannot yet be *composed*
        (`[0 1]([0.2 0.8](u))` — the inner brackets break `PARENWORD`), and the `NAME axistuple` arm does NOT in
        fact work at a value site for a grid/scatter (`reflect grid:ramp(u)` is "unrecognized spectrum
        expression"); it works only for materials and material properties, where it is a bundle application.
        *(Both are now closed — see the composition and table-call STATUS blocks below.)*
    * **STATUS (2026-07-28): COMPOSITION DONE — `coord = … | value`, the last unimplemented arm of the grammar
      sketch above. Shipped as 0.100.0.**
      A coordinate may now itself be a sampled value, so a literal composes into another's call to any depth:
      `[0 1]([0.2 0.8](u))`, on any single axis of a multi-axis call (`[[0 0.3][0.6 1]]([0.5 1](u), v)`), and as a
      term inside coordinate arithmetic. The idiom it buys is remapping — the inner table is the transfer curve
      applied before the outer lookup.
      - **The blocker was the LEXER, not the loader.** `splitCallArgs` already tracked bracket depth, so the outer
        call's arguments would have split correctly; what failed was that `PARENWORD`'s interior class excluded
        `[` / `]`, so the inner literal's brackets split the token and the outer literal reported the (misleading)
        *unsaturated* error. Widening the class is safe because the terminal's **balance guarantee rests entirely
        on `(` and `)` staying excluded** — two paren groups on one line still cannot merge, since merging would
        have to consume the intervening `)` as an interior char. Longest-match still prefers `PARENWORD`, because
        `WORD`'s own group alternative keeps brackets out and so matches only the bare `(`.
      - **A composed literal costs ONE block, not two.** `desugarOne`'s core is now `Builder::buildArrayGrid`
        (flatten + arity/shape checks + the anonymous `grid`), and only the *value-site* form wraps it in a
        `pattern` — a composed one is spelled by substituting `grid:__arrN(coords)` into the outer call text
        (`Builder::desugarNestedLiterals`, recursive), which is already a legal pattern-expression term.
      - **The loader diagnoses what the lexer deliberately stopped checking.** Brackets inside a call are captured
        but not balance-checked, so `Builder::parseArrayText` re-parses the argument text with the tokenizer's own
        splitting rule and names an unbalanced group, a call-less inner literal, or a wrong inner arity against the
        author's source. A literal glued to an identifier (`f[0 1](u)`) is refused **before** substitution, or the
        rewrite would report an "unknown identifier `fgrid`" appearing nowhere in the file. Errors two levels deep
        still name the authoring site, never `__arrN`.
      - **`-checkarray` section (h)** pins five identities against twins whose coordinate is spelled
        *arithmetically* — non-circular, since a 2-sample grid over `lo 0 hi 1` interpolates linearly, so
        `[0.5 1](u)` **is** `0.5+0.5*u`. Includes a deliberately non-identity outer array (a 3-sample tent, whose
        composition with `[0.5 1](u)` must equal `[1 0](u)`), so no case can pass by the outer lookup being a
        no-op. `sameReflect` grew a tolerance argument: those twins agree to float precision (~1e-8), because grid
        samples are stored as float32 while an expression evaluates in double — demanding the bit-identity the
        rebind twins use would pin the storage format rather than the semantics.
      - **Validation:** all 15 self-tests PASS, all 87 scenes parse, loom is 1255/1255 (the change is confined to
        `ftsl_scene.epeg`; loom's reader uses the sibling typed `ftsl.epeg`), and `scraps/arr_compose.ftsl` renders
        **bit-for-bit identically** to its direct-literal twin, with the tent case likewise bit-identical to its
        analytic equivalent and demonstrably different from the identity case.
    * **STATUS (2026-07-28): `NAME axistuple` AT A VALUE SITE DONE — the increment-2 `Deferred:` clause's second
      arm, which was simply wrong. Shipped as 0.101.0.**
      That clause (line ~227) said `NAME axistuple` "needs no work because ftrace's expression evaluator already
      reads `name(args)` as a call". True *inside* a pattern expression — but a **value site is not an expression
      site**: the slot readers only recognised `pattern:<name>` there, so `reflect grid:ramp(u)` was an
      "unrecognized spectrum expression". Now every per-hit slot takes a table call directly.
      - **The same four chokepoints v0.89.0 used for `MATERIAL.slot(args)`.** `bindScalarPattern` and
        `patternedSpectrumParam` (the per-hit readers) send a `grid:` / `scatter:` head through
        `Builder::tableCallPattern`, which compiles it with the ordinary `compilePatternExpr` and appends to
        `scene.patterns` — so the slot holds *exactly* the index a hand-written one-line
        `pattern { expr "grid:ramp(u)" }` would have produced, and there is no second evaluation path.
        `dblParam` and `evalSpectrum` are the load-time-constant readers and **refuse**, each naming the slots
        that can hold a per-hit value.
      - **Only the scoped spelling is accepted.** A bare `ramp(u)` at a value site already means the §7.6 material
        bundle application, so accepting it for tables would make meaning depend on which namespace holds the
        name. `isTableCallHead` tests for the `grid:` / `scatter:` prefix and nothing else. A call-less
        `grid:ramp` is refused with the `(u)` to add — a table is read AT coordinates — rather than silently
        becoming a constant.
      - **Composition works inside a table call too** (`reflect grid:ramp([0.2 0.8](u))`), because a coordinate is
        an expression here as everywhere: `Builder::desugarTableCall` runs `desugarNestedLiterals` over the token
        before it compiles, dispatched from `desugarArrays`' visit loop on the cheap
        `isTableCallHead && contains('[')` test. This needed the *second* grammar change: a **named** table's call
        lexes as a `WORD`, not a `PARENWORD`, so `WORD`'s balanced-group alternative was widened to
        character-for-character `PARENWORD`'s body. Only the group *interior* admits brackets — `WORD`'s fallback
        class still excludes them, so a bracket outside parens is a delimiter as before and `REC.chan[2]` still
        stops the word at the `[`.
      - **`-checkarray` section (i)** pins `reflect grid:g(u)` ≡ `[0 1](u)` (with a non-constant check, since an
        unbound grid pool would make both 0.0), the 2-D `grid:g2(u,v)` form, `scatter:s(u)` ≡ a hand-written
        `pattern` twin, the formal route (`grid:g(a)` + `src.reflect(a=u)` ≡ `grid:g(u)`), both composition
        directions, and the SCALAR-slot binding via a `roughness` probe. Plus five refusals: two "fixed at load
        time" (`film_ior`, `ior`), "does not sample it", "unknown grid" and "expects 1 arg".
      - **Validation:** all 15 self-tests PASS, all 87 scenes parse, the grammar corpus sweep is 87/87, and loom is
        1255/1255 (again confined to `ftsl_scene.epeg`).
    * **ADDENDUM — call = sample; late-binding & rebinding of the consumed axis (design intent, user).** The
      trailing `(...)` is not just a *label* on a literal — it is the **sample call**, exactly like loom's
      `grid(x, y)`. Two authoring positions, so a material can *define* what an array consumes, or *defer* it to its
      user:
      - **Material-side (the array already names its own axis).** `reflect [0 1](u)` bakes the consumed coordinate
        into the material: the reflectance is `[0 1]` sampled at the surface `u`. The material's author has spent
        the axis; a user of the material supplies nothing.
      - **User-side (a bare array, axis left open).** A material may expose an array parameter *un-called*
        — `reflect [0 1]` — declaring "this is a 1-D array, you pick what drives it." Whoever *instantiates* the
        material then completes it by calling it: `reflect [0 1](u)` (or `(v)`, `(x)`, a record channel, …). A bare
        array with no call is an **unsaturated** value — legal to declare, an error to actually *render* until some
        site saturates it.
      - **Rebinding an already-named axis: `(a=u)`.** If the material *did* name an axis — say it declared the
        parameter with a formal axis `a` (`reflect [0 1](a)`) — a user who wants to feed a *different* driver than
        the formal name uses the **keyword form** `(a=u)` — read `formal=driver` — "bind my `u` to the array's `a`
        axis." Positional `(u)` rebinds the sole/next axis; keyword `formal=driver` targets a named one. **Keyword
        rebinds take multiple arguments**, one per axis — e.g. a 2-D array whose formals are `u,v` is rebound to the
        drivers `a,x` as `(u=a, v=x)` (and `(a=u, b=v)` likewise for formals `a,b`). Mix is allowed the usual way
        (positionals first, then keywords). This is ordinary call-site argument binding — positional or by-name —
        lifted onto the array-sample. Semantics to pin when built: whether a formal axis name is a *binding site*
        (rebindable) vs a *literal coordinate source* (fixed), and the error when a bare array reaches the renderer
        unsaturated.
    * **Grammar sketch (shared `.epeg`, one production reused by array / grid / scatter).** The axis tuple hangs off
      any value; args are positional coords or `name=coord` rebinds:
      ```
      sampled   = value axistuple?
      axistuple = '(' arg (',' arg)* ')'
      arg       = NAME '=' coord      # keyword rebind  (a=u)
                | coord                # positional      (u)
      coord     = NAME | NUMBER | value   # an axis driver: a var (u/v/x), a constant, or a nested sampled value
      ```
      Notes: `value` is the existing context-free array/vec tree (`values.py`); `axistuple` is optional, so an
      un-called array stays a plain `value` (the *unsaturated* case). The `arg (',' arg)*` repetition is what makes
      multi-axis calls work — positional `(u, v)` or keyword `(u=a, v=x)` (and mixes, positionals first). `coord`
      allowing a nested `sampled` gives composition (`n( m(u), v )`). Reuse the **same** `axistuple` on the N-D grid
      and scatter element grammars so literal-array / grid / scatter all sample identically.
    * **Carry loom's constructor conveniences into the ftsl grid/scatter datatype** (already shipped in loom
      `data.py`, port them alongside the datatype):
      - **`shape=` is redundant when the data is nested.** The nesting *is* the shape — `[[0 1 2][3 4 5]]` is a
        `(2, 3)` grid — so the grid element must infer shape from the value tree and only need an explicit shape to
        fold a *flat* list. (This is also exactly why `[[0,1,2][3,4,5][6,7,8]]` reads as a 2-D grid above.) Bare
        `[...]` always means an axis; a stored **vector** sample is a `vec(...)`/tagged colour, never a bare list.
      - **`lo`/`hi` domain shortcuts (optional + broadcastable).** The sample lattice's world box need not be spelled
        in full: `lo` omitted → all-zeros; `lo=<scalar>` → broadcast to every axis. `hi` omitted → a **unit-spacing
        index lattice** (`hi[a]=lo[a]+shape[a]-1`, so a query coordinate equals a sample index); `hi=<scalar>` (or a
        length-1 tuple) → pins **axis 0** and derives every other axis as a **uniform lattice** (one isotropic
        spacing `h=(hi-lo[0])/(shape[0]-1)`, `hi[a]=lo[a]+h·(shape[a]-1)` — the mathematically pure "single lattice
        constant" reading, keeping the interpolated field geometrically isotropic); a full `hi` tuple → the exact box
        (allows deliberately anisotropic cells). Whatever surface syntax the grid element grows for domain must offer
        these same defaults so common grids stay terse.

      **STATUS (2026-07-27): DONE — shipped in 0.74.0.**
      - The **`lo`/`hi` half was already shipped** with the datatype itself (0.71.0): `addGrid` implements exactly the
        rules above — `lo` omitted → zeros, `lo <scalar>` → broadcast; `hi` omitted → unit-spacing index lattice,
        `hi <scalar>` → one isotropic lattice constant derived off axis 0, full tuple → the exact box. Nothing to do.
      - The **nested-`data` half shipped here.** A `grid`/`scatter` element's `data` may now be written **bracketed**,
        and for a grid the **nesting is the shape**: `data [[0 1 2][3 4 5]]` is a 2×3 grid with no `shape` line at
        all. Writing a `shape` that disagrees with the nesting is an error that prints both. A *flat* bracket group
        carries no shape (it is just the bracketed spelling of the flat list), so `shape 2 2  data [0 1 2 3]` still
        folds as before — nothing in the corpus changed meaning. For a `scatter`, a bracketed `data` is one group per
        sample, each exactly `dim+1` numbers wide (coords then value), checked per group.
      - Implementation: the increment-2 machinery is reused wholesale. `Builder::desugarArrays` now **skips `grid` and
        `scatter` blocks** — their `data` is the element's own samples, not a value-site literal — and `addGrid` /
        `addScatter` read the `ftsl::BrItem` tree directly through the shared `flattenArray` (which already proves the
        tree rectangular and numeric). A sample call on an element's `data` (`data [0 1](u)`) is a dedicated error
        naming where the call *does* belong. No grammar change was needed: `data`'s value site already goes through
        the merged `selector` production.
      - Verified: all seven bracketed forms load; eight error paths produce clear messages; `scraps/array_nestdata.ftsl`
        (nested `data`) renders **bit-identically** to `scraps/array_explicit.ftsl` (`shape` + flat `data { … }`);
        corpus equivalence sweep still ok=362 / mismatch=0 / parsefail=5 (pre-existing); `-checkgrid` / `-checkscatter`
        / `-checkbvh` / `-checkimplicit` all PASS.
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
  - **DONE (2026-07-26): the flip landed — the shared grammar IS ftrace's front end.** VERSION 0.67.0 → **0.68.0**.
    - *Gate met:* the corpus differ reached **MATCH 2595/2595** (`old_fail=0 gpda_fail=0 mismatch=0`) — every `.ftsl`
      in the tree, structurally identical down to `Stmt::line` (which required threading token line/col through the
      parse tree, upstream GraphParser `7432f42`). loom's own suite: 1072/1072.
    - *Code:* `ftsl_shim.hpp` → **`src/gpda/ftsl_frontend.hpp`** (namespace `shim` → `ftsl_gpda`), and `loadSource`
      now calls `ftsl_gpda::parse()` by default. `-legacy-parser` / `FTRACE_LEGACY_PARSER` selects the retired
      hand-written parser (one-release escape hatch); `-validate-grammar` still cross-checks both.
      Both flags needed a **pre-scan** in `main.cpp` — the scene loads (~3813) *before* the argv loop (~3837), so
      `-validate-grammar` had silently never worked as a CLI flag (only via the env var).
    - *Error quality* is the visible win: `line 1, col 15: unexpected NEWLINE '\n'; expected '{'
      (in brace_body < plain_header < top_block < item)` vs the legacy `line 1: expected '{' after material`.
    - *Verified:* bit-identical renders GPDA vs `-legacy-parser` on `_ellipsoid_test.ftsl` (mode B/GPU) and
      `mirror_selfie.ftsl` (mode R/GPU), `-validate-grammar` silent, exit 0.
    - *Two lifetime bugs the flip exposed* (fixed upstream in GraphParser `284244f`, re-vendored):
      (1) `Parser::acquire_visited()` returned a reference into a `std::vector<Visited>` that a recursing predicate
      could reallocate — a use-after-free that showed up as a permanently retained cursor; elements are `unique_ptr`
      now. (2) `Visited` only cleared on *acquire*, so a Parser outliving a parse kept the pool's cursor stacks alive
      and they were destroyed after the `thread_local` pool at exit (heap corruption; ftrace segfaulted on exit).
      `parse()` now runs `reset_scratch()` on the way out, and the pool is owned through a deleter that frees it only
      when drained. Same latent pattern fixed in `scannerless.{hpp,cpp}`.
    - *Legacy parser: deleted in 0.79.0* (nothing in `src/` references it any more).
    - *Still open:* port the rich `ParseError` to `scannerless`
      (needs char-level expected sets + offset→line/col mapping).

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

> **SECTION CLOSED 2026-07-26.** Everything in B that delivers a capability is built and green
> (1046 loom tests pass): Phase 1 (P1.1–P1.5), Phase 2 (P2.1), Phase 3 (P3.1–P3.3, slices S1–S7),
> and §8 G1 / G2 / G5. The two remaining unchecked items — **G3** (`PatOp::MatMulAdd`) and **G4**
> (GPU marching cubes) — are **deliberate deferrals, not pending work**: both are optimizations of
> paths that already work correctly, and neither unlocks anything. G3 only *compresses* the pattern
> encoding (the N-D rotation matrix already compiles to scalar bytecode and ray-marches on the GPU
> today); G4 only speeds up *offline mesh export* (the video path no longer tessellates at all,
> since G2's `-raster-gpu` sphere-traces the field directly). Each has a written revisit trigger and
> a full build plan in `known-issues.md` ("Deferred: `PatOp::MatMulAdd`" and "Deferred: GPU marching
> cubes"). **Do not treat B as unfinished** — reopen only if one of those triggers actually fires.

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
      marginal-speed win, **not** a new capability.
      **MEASURED 2026-07-27.** The old stated reason ("sin/cos/PovFn + the sphere-march dominate") is
      **wrong**: `patternEval` costs ~4.7 ns/node *regardless of opcode*, so cost ∝ node count, and the
      affine rows are 85–89% of every emitted field — MatRow would genuinely cut nodes 3.5–4×. The real
      reason to defer is **Amdahl**: field eval is only 7% of a `-raster-gpu` render, 12% of `-raster`,
      11–14% of an export at the D=8 sizes actually rendered (and ~4% for a default random draw), so
      MatRow buys ~3–9% end-to-end. It *does* pay off in the fully-coupled high-D regime — measured
      **D=16 `--coupling all`: 3917 nodes, 62% field eval, 1.9× faster with MatRow**.
      **Sharpened trigger:** build it when a real workload's field exceeds **~800 pattern nodes**
      (≈25% field eval) — in practice `--coupling all` at D≥8, or a long video at D≥12 fully coupled.
      Default `cyclic` coupling stays under 20% even at D=16. Cost model: `time ≈ 3.50 s + 1.452 ms ×
      nodes` (GPU iso, 600²). See known-issues.md "Deferred: `PatOp::MatMulAdd`" for the full tables and
      the probe method. Prefer the contained single-output "matrow" form (Option A).
- [ ] **G4 (deferred, export-only — but the bottleneck it assumed turned out to be elsewhere)** GPU
      marching cubes, *only* for mesh export, not the video path. **MEASURED 2026-07-27:** a res-160
      export was **55% ASCII OBJ write** (11.77 s of 21.47 s), not marching — so even a free GPU march
      capped out at 1.8×. Fixed the writer instead (see G4b); the march is now the largest remaining
      block (~8.7 s of 13.4 s, scaling as res³). Revisit only for repeated export cost — batch-exporting
      a frame sequence, or interactive export at res ≥ 384.
- [x] **G4b — fast OBJ writer.** **DONE 2026-07-27 (v0.84.3).** `isomesh::writeObj` did one
      `std::fprintf` per line (~6.6M calls for a 3.3M-tri mesh), where FILE locking + format reparsing
      dominate I/O — measured 22 MB/s on a 257 MB file. Now formats into an 8 MB staging buffer flushed
      with `fwrite`, with hand-rolled decimal conversion for the integer-only face lines; floats keep the
      same `snprintf` conversion specifiers so output is **byte-identical** (verified by md5 against the
      previous binary). **Write 11.77 s → 4.66 s (2.5×); whole export 21.47 s → 13.40 s (1.6×).**
      Multi-group export (`scenes/implicit.ftsl`, 3 isosurfaces) re-validated.
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
- [x] **C2 VDB: native sparse device sampler.** **DONE 2026-07-24 (v0.47.0).** The GPU no longer uploads
      the dense lattice; instead the baked grid is partitioned into **8³ bricks** and only bricks holding a
      nonzero voxel reach the device, so VRAM tracks *occupied volume* rather than the bounding box.
      `VdbGrid::buildBricks(B, …)` (`vdbgrid.h`) emits a compact `brickIndex` (bx·by·bz int32 slots, −1 for an
      empty brick) plus contiguous `brickData` (B³ fp16 voxels per active brick); the uploader
      (`render_cuda.cu`) uploads those two arrays and the device sampler `dMedDensityAt` does a per-corner
      brick lookup (`brickIndex[(k≫3)·by·bx + (j≫3)·bx + (i≫3)]`, then `brickData[slot·512 + (lk·8+lj)·8+li]`).
      **Bit-for-bit identical** to the dense sampler — the trilinear stencil is clamped to [0,n-1] before any
      lookup, so per-brick padding voxels are never addressed. Validated on the smoke plume
      (`scraps/vdb_smoke_native.ftsl`): 3074/5488 bricks active (56 %), 5.3 MB → 3.0 MB VRAM (1.7×), and GPU
      (sparse) vs CPU (dense) energy balance identical (absorbed 0.0294 both). A one-line
      `[vdb] sparse device grid: …` footprint report prints once per grid. *(Host RAM still keeps the dense
      lattice for the CPU sampler; a sparse host representation is a possible future RAM win.)*
- [x] **C3 VDB: fp16 + emission/temperature grids** (fire). **DONE 2026-07-24** (fp16 + forward-CPU fire).
      - [x] **fp16 dense-grid storage — DONE 2026-07-24.** The baked dense lattice (`VdbGrid::data`)
        is now `std::vector<uint16_t>` half-floats instead of `float`, halving host RAM and GPU VRAM
        for every imported `.vdb`/`.nvdb` volume. `vdbgrid.h` gains portable IEEE-754 binary16↔binary32
        helpers (`halfBitsToFloat`/`floatToHalfBits`, round-to-nearest-even, full subnormal/inf
        handling); both bake sites (`vdb_openvdb.cpp`, `vdbgrid.cpp`) store `floatToHalfBits(v)` and
        the trilinear samplers (`VdbGrid::sample`) decode via `halfBitsToFloat`. The majorant `maxVal`
        is bumped ×1.001 so half-rounding can't push a stored value above the delta/ratio-tracking
        bound. GPU: `DMedium::vdbData` is `const uint16_t*`, uploaded as-is, with a matching
        `__device__ dHalfBitsToFloat` in the hot density sampler (portable bit math, no cuda_fp16
        dependency, HIP-safe). Validated on the native smoke plume (`scraps/vdb_smoke_native.ftsl`):
        CPU vs GPU energy balance identical (absorbed 0.0294 both) and plume-region mean colour agrees
        to ~0.05% relative — within Monte-Carlo noise — confirming both half-decoders are correct.
      - [x] **emission/temperature grids (fire) — DONE 2026-07-24 (forward CPU).** A medium may now
        carry a `temperature vdb:<path>` grid + `emission blackbody` (+ `emission_kelvin`/`emission_scale`),
        turning its hot voxels into a self-illuminating isotropic volume emitter. Multi-grid `.vdb` files
        (the official OpenVDB *fire* sample: one `density` + one `temperature` float grid) are selected
        **by grid name** — `loadOpenVDBGrid`/`loadVdbGrid`/vdbgrid.cpp take a `wantName` and the OpenVDB
        reader now seeks each descriptor to the previous grid's `endPos` (descriptors are interleaved with
        bodies, NOT contiguous — this was the bug that made the 2nd grid fail to load). `spectrum.h` adds
        `blackbodyRadiance`/`blackbodyEmissionRadiance` (Planck normalised to a 6500 K/560 nm reference,
        preserving physical T⁴ + Wien hue). `scene.h` `Medium` gains `temperature`/`tempPeak`/`emitKelvin`/
        `emissionScale` + `emissive()`/`temperatureAt()`/`emissionAt()` (peak-normalised T=emitKelvin·raw/
        tempPeak, default 1500 K); `Scene` gains `EmissiveVolume` + `finalizeEmissiveVolumes()` (a 20k-sample
        Monte-Carlo estimate of each grid's mean emission `meanKe` and selection `power`=4π·V·meanKe·Δλ, plus
        `totalEmissionPower`). Forward `tracePhoton` (render.h) picks emitter-vs-fire birth by power
        (`grandTotal=totalPower+totalEmissionPower`; short-circuits with NO extra RNG when there are no
        emissive volumes, so every non-fire scene stays bit-identical), and a fire photon is born at a
        uniform AABB point + isotropic direction with **λ importance-sampled from `blackbody(emitKelvin)`**
        (per-volume `EmissionSampler lamSampler`, built in `finalizeEmissiveVolumes`), carrying
        **β=grandTotal·κ_e(x,λ)/(meanKe·Δλ·p(λ))** — derived so the isotropic `1/(4π)/(dist²·Ω)` direct splat
        (`connectEmissionVolume`/`connectEmissionLensVolume`/`camSplatEmissionAll`) reproduces the emission
        line-integral exactly, and (for a voxel at emitKelvin) β is constant across λ so the spectral
        colour-magnitude speckle collapses (v0.48.1; uniform p=1/Δλ recovers the plain β=grandTotal·κ_e/meanKe).
        The "scene has no light" guard (ftsl.h) now accepts an emissive volume. Validated on the official
        fire sample (`scraps/vdb_fire.ftsl`, `png/vdb_fire.png`): the flame glows self-lit with the correct
        red-edge/hot-core shape and no external light. **GPU mirror DONE (v0.49.0):** `render_cuda.cu`
        factors the VDB brick sampler into a reusable `DVdbGrid`/`dVdbSample` (density path bit-for-bit),
        `DMedium` gains a `tempGrid` + emission params, `DScene` gains a `DEmissiveVolume[]` (+ per-volume
        Planck-λ CDF) + `totalEmissionPower`, and `genPhoton` has the same power-split volume-birth branch +
        `connectEmissionVolume`/`camSplatEmissionAll` device splat; `cudaForwardSupported` no longer rejects
        fire. Verified `-device gpu` matches the CPU flame in distribution and the density path still
        conserves energy (`scraps/vdb_cloud.ftsl`, sum/emitted=1.0). (Backward R/V is N/A: it treats media as
        one homogeneous haze and never samples the grid.)
- [x] **C4 VDB: native `.vdb` front-end** — DONE. `loadVdbGrid` dispatches on the file magic; a
      self-contained OpenVDB reader (`src/vdb_openvdb.cpp`, no OpenVDB/NanoVDB dep) parses the file
      container, `float 5_4_3` tree topology and BLOSC+ACTIVE_MASK+HalfFloat leaf buffers by hand,
      decoding blosc's LZ4 codec with a vendored single-file LZ4. Validated bit-for-bit against
      python-blosc on the official OpenVDB smoke/sphere/cube samples and render-validated
      (`scraps/vdb_smoke_native.ftsl`). Other blosc codecs (BloscLZ/Zlib/Zstd) + ZIP report a clear
      "re-export with LZ4" message — see known-issues.md.
- [x] **C5 Mesh: emissive triangles** (mesh area lights).  **DONE 2026-07-24**.
      Any material may carry an `emit <spd>` spectrum; a `mesh` bound to it becomes an
      `EmitterShape::Mesh` area light. `scene.h` adds `EmitTri`/`Emitter::meshTris` (a
      per-triangle cumulative-area CDF), a `samplePoint` Mesh branch (binary-search a
      triangle by area, then barycentric sample; pdf = 1/total-area), and
      `Scene::addMeshLight`. Emission-on-hit was already generic (`m.isLight`/`m.emit`);
      NEE / forward emission / BDPT s=0 all consume the new emitter through the existing
      `samplePoint` + 1/area paths — no per-renderer changes. `ftsl.h`: `buildMaterial`
      parses `emit` (sets `isLight`); `addMesh` auto-registers the emitter and, with a
      mesh-block `power`/`lumens`, rescales the SPD over the mesh area (clones the
      material + rebinds the range's triangles so a shared material is untouched); the
      "scene has no light" guard now also accepts an emissive mesh. GPU: `DEmitter`
      gains a device `DEmitTri*` CDF + `emitterSamplePoint` shape-5 branch + per-emitter
      upload (fixes a crash where a Mesh emitter fell through to the quad path with a
      null CDF). Emission is one-sided (front face only), so a CLOSED emissive mesh
      whose triangles are wound INWARD (e.g. an imported `torus.obj`, every face facing
      the interior) would radiate into its own hollow and look black; `addMesh` now
      auto-orients such shells outward at load — signed volume about the centroid < 0
      (thresholded against area^1.5 so planar/open sheets are never touched) reverses
      every triangle's winding (swap v1↔v2 + uv/normal, re-`finalize()`) before emitter
      registration, so both the per-Tri geometric normals used by emission-on-hit and
      the `addMeshLight` sampler normals point outward. Validated CPU **and** GPU: an
      emissive quad mesh reproduces an equivalent `light area` visually
      (`scraps/cmp_meshlight.ftsl` vs `cmp_arealight.ftsl`), and a 16384-tri inward-wound
      emissive torus (`scraps/mesh_light.ftsl`) now glows and lights the Cornell box on
      both backends with matching results (overall/torus/red-wall/green-wall means agree
      to <1%). glTF/GLB meshes that import their own materials are not auto-lit (bind an
      FTSL `emit` material) — noted in known-issues.
- [x] **C6 Mesh: tangent-space normal maps.**  **DONE 2026-07-24**.
      A material may bind a normal-map texture (`normal_map texture:<name> strength <s>`); at every hit
      the shading normal is perturbed by the tangent-space normal sampled from the map, rotated through
      the surface TBN frame. Per-triangle tangents are built in `Tri::finalize()` via Lengyel's UV-gradient
      method (Gram-Schmidt against the geometric normal + a stored `bitangentSign` for handedness); the
      sphere path derives a longitude tangent. The perturbation is applied at the single CPU intersection
      choke point (`Scene::applyNormalMap`, called from both `closestHit` and `closestHitLinear`) so ALL
      CPU renderers (backward/forward/bdpt/vcm/sppm/photonmap/grin) get it for free, and at the matching
      GPU choke point (`dApplyNormalMap` in `closestHit`). Tangents transform with instances
      (`instanceHitToWorld` on both paths; the device uploads a per-instance `Wm` = toWorld linear).
      Normal maps must be declared `encoding linear` (raw vector data, not sRGB colour) — the loader warns
      otherwise. The device uploads the raw RGB only for textures actually used as normal maps
      (`usedAsNormal` pass) to save memory. **Validated:** `scraps/ripple_test.ftsl` (a flat wall quad
      textured with a strong horizontal-corrugation normal map under a grazing area light) renders clear
      vertical light/dark banding; CPU and GPU outputs are numerically identical (col-profile std 6.11 vs
      6.10, row std 3.43 vs 3.43, mean 7.888 vs 7.891) — proving the two paths perturb shading identically.
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
- [~] **C9 Alembic (`.abc`) import** — **DON'T DO FOR NOW (user, 2026-07-24).** Heavy SDK (Imath +
      HDF5/Ogawa); an OBJ/glTF/FBX sequence suffices for now. Revisit only if a real `.abc` asset needs it.

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

### E2 — General N-D curve → scene-variable animation via the rasterizer curve editor  ✅ DONE 2026-07-28 (all 3 slices)  *(loom + ftrace; LARGE, design; extends §A)*
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

**SLICE 1 DONE 2026-07-24 (`loom.anim`).** The channel-a config model + its sidecar projection + the
channel-b value fan-out — the pure-Python go-between core, resolving OPEN Q1/Q2 (config lives in a loom
struct with a JSON sidecar; the go-between is loom). Built on the E5 influence model (`loom.axes`), so E2
value-routing *is* the E5 pin/mod edge model — the "E5 unifies E2/E4" tie-in made concrete. Pieces:
- **`CurveDrive(dims, points, bindings, *, mode, closed, name)`** — the authoritative in-memory config:
  dimension count, static starting control `points` (each a `dims`-tuple; point *modulation is out* per the
  design — the editor owns the time axis), and `ChannelBinding` associations. `mode ∈ {MODE_FLYBY,
  MODE_ANIMATION}`. Validates point dims and channel ranges.
- **`ChannelBinding(channel, target, mode, gain, kind)`** — one channel→scene-variable edge; `mode`
  (`pin`/`mod`), `gain`, and quantity `kind` (`ADDITIVE`/`GAIN`/`BIPOLAR`) are the E5 edge attributes.
- **Sidecar round-trip** — `to_dict`/`from_dict` + `save`/`load` (versioned JSON, **atomic** temp-file +
  `os.replace` so the editor never reads a half-written config; "scene proposes, editor disposes" —
  associations round-trip through this file).
- **`sample(t)`** — uniform Catmull-Rom over the control points (loom-side preview/tests; ftrace's editor is
  the sampling authority in a live session), interpolating knots, closed-wrap or clamped-open.
- **`apply(values, bases)`** / **`frame(t, bases)`** — fan the current sampled channel values out to
  `{target: value}` via the bindings: several channels co-driving one target compose through an E5 `Target`
  of the declared kind (domain-correct accumulate), with an optional authored `base` per target.

19 tests (`tests/test_anim.py`: construction/validation, sidecar faithfulness + atomic write + version guard,
Catmull-Rom knot/linear/closed/clamp, pin/mod/bipolar fan-out, multi-channel-one-target, base offset).
**SLICE 2 DONE 2026-07-24 (`loom.anim`).** Named animatable slots + the editor↔loom live-value channel,
resolving the slice-1 "how does a binding target name a real scene value-site" open item via **option (b):
named `RefSignal`-style slots** (no emit-path change). Pieces:
- **`Slot(name, default)`** — a `Signal` leaf holding a mutable current value. Drop it anywhere a scene
  parameter accepts a `Signal` (a material prop, a transform field, a signal-valued isosurface param, …);
  because it *is* a Signal the scene's `roots()`/`walk`/`emit` machinery discovers and bakes it per frame,
  so binding-by-name needs zero change to the emit grammar. `default` doubles as the authored `mod` base.
  The one controlled escape from clock-purity: its value is *pushed* by the live channel, not computed from
  the clock — so each scrub frame is emitted with a **fresh** `Cache` (a stale cache pins the old value).
- **`collect_slots(scene)`** — walks every modulator in the scene (`_all_elements` → `element_roots` →
  `walk`) and groups the `Slot`s by name.
- **`SceneDriver(scene, drive, *, bases, strict)`** — binds a `CurveDrive`'s fan-out to the scene's named
  slots. `set_values(values)` fans channels out (each slot's `default` is the target's `mod` base unless
  overridden) and pushes each resolved value into its same-named slots; `emit_frame(values, clock)` does
  that then emits with a fresh cache. `strict` raises on a target with no matching slot (typo'd binding
  fails loudly).
- **`LiveSession(driver)` / `serve_live(session, in, out)`** — the loom side of E2 channel-b: a
  newline-delimited-JSON stdio message loop (the `loom.PreviewServer` precedent, editor→loom direction).
  Commands: `frame` (set values or `sample` at `t`, emit that frame's `.ftsl` to `out`), `config` (return
  the sidecar dict to seed the editor), `bindings`/`points` (editor disposes — replace the associations /
  control points), `save`, `quit`. Each command is a pure `dict`→`dict` `handle()` so the protocol is
  unit-testable without a real pipe; errors are reported in the ack, never crash the loop.

23 tests (`tests/test_anim_live.py`: slot value/stale-cache semantics, discovery + same-name grouping,
driver base defaults/override/strict, set_values fan-out + mod-on-base, emit-frame fresh-cache, every
LiveSession command + bad-input acks, `serve_live` stop-on-quit + bad-json).
**SLICE 3a DONE 2026-07-28 (`ftrace -anim`, v0.94.0).** The C++ editor now *edits the drive itself* — the
sidecar round-trip half of slice 3. Pieces:
- **`src/curvedrive.h`** — a header-only reader/writer for loom's `CurveDrive` JSON sidecar, on
  `src/third_party/json.h` (minijson). It re-checks **the same invariants `CurveDrive.__init__` does**
  (`dims >= 1`, `>= 2` points, every point exactly `dims` wide, every binding channel in range, valid
  `mode`/`kind`) on **both** load and save, so ftrace can neither accept nor write a sidecar loom would
  reject. Writes atomically (temp file + `std::filesystem::rename`, the C++ twin of `mkstemp` +
  `os.replace`), and prints shortest-round-tripping numbers so an edit that moved one point leaves every
  other coordinate byte-identical.
- **`-anim <file.json>`** (implies `-explore`) — the fly editor's control points ARE the drive's N-D
  points. Channels 0–2 are what the viewport draws and the mouse moves (for a flyby drive, literally the
  camera eye); channels 3.. are values no 3-D viewport can show, so they ride along **per point** and are
  written back untouched. Point *orientation* is derived on seed from the chord to the successor (a drive
  is a curve of values and stores no orientation), then re-aimable by the existing orientation painting.
- **Save writes both** — the `camera_curve` `.ftsl` block as before, *and* the reshaped drive back to its
  sidecar. The editor owns only the point LIST; `name`, `mode`, `closed`, `dims` and every
  channel→variable **binding** are copied from whatever the sidecar last held, so an editing pass never
  drops an association loom authored ("scene proposes, editor disposes" in the ftrace direction).
- **A sidecar that doesn't exist yet is not an error** — that's how you *start* a drive from the editor:
  whatever points the scene seeded (e.g. an authored `camera_curve`) become a fresh 3-channel
  `mode flyby` drive that the first Save creates.
- **`trackInsert`/`trackErase`** — `editPts`' parallel per-point side tracks (the painted speed multiplier
  and now the extra channels) are mutated **only** through these, so a track can never drift out of
  alignment with the points it annotates. A new point inherits its unseen channels from its neighbours
  (midpoint in the middle, a copy at either end) rather than snapping them to 0.

Verified end-to-end by driving the real GDI panel buttons (`scraps/click_button.ps1` → `BM_CLICK`): a
loom-written 5-channel drive with 3 bindings seeds, `+Pt`/`Ins`/`Del`/`Save` reshape it, and loom re-loads
the ftrace-written file with the extra channels interpolated exactly as designed and every binding intact.

**SLICE 3b DONE 2026-07-28 (`-anim` + `-loom`, v0.95.0) — E2 IS NOW CLOSED END TO END.** The live
editor↔loom value channel plus the binding-editing panel row. Pieces:
- **`-anim <sidecar.json> -loom <scene.py>`** — the editor spawns
  `python -X utf8 -u -m loom.anim <scene.py> --config <sidecar>` and drives a `LiveSession` over
  newline-delimited JSON. **ftrace does not sample the curve; loom does** — the editor pushes the control
  *points* and then asks by parameter `t`, so what the viewport shows cannot drift from what loom will
  render for the video. Each ack names an emitted `.ftsl`, which replaces the scene wholesale (every bit of
  derived state — `plight`, `prims`, the GPU's baked triangles, the resident RGB-backward session — is
  dropped and rebuilt, because all of it is a function of the scene).
- **Two-queue bridge policy.** `frame` messages are **latest-wins on one slot** (scrubbing fast must not
  build a backlog of stale poses), while `points`/`bindings`/`dims` go on a **FIFO that never drops** and
  is drained before every frame — losing a control message would leave loom rendering against a curve or a
  binding set the editor no longer has.
- **The bind row** (a fourth panel row, built on demand once the live channel is up): a channel combo, a
  slot combo seeded from the `slots` command (pick-only, `(none)` first, so "this channel drives nothing"
  is expressible without reaching for Unbind), **Bind** / **Unbind**, a `chans:` grow/shrink box, and a
  status readout that answers "what does this channel do right now?" — `ch 0 → ball_r | live — 4 baked,
  2 ms`. Growing `dims` widens every point; shrinking it drops the bindings on the vanished channels,
  mirroring what loom itself does rather than leaving the sidecar holding an association loom has already
  forgotten. Save writes the edited dims *and* bindings back.

Verified end-to-end through the real GDI controls (`scraps/bindtest.ps1` → `CB_SETCURSEL` +
`CBN_SELCHANGE`, `BM_CLICK`, `WM_SETTEXT`), with a deliberately **static camera** drive
(`scraps/anim_static.json`: three identical eye points, channel 3 = 0.20/0.95/0.20 → `ball_r`) so that any
change in the image is provably the *driven variable* and not camera motion — the middle ball breathes
while the two static posts hold still. Bind/Unbind flip it live; `chans:` 4→7→5→3 reshapes the drive and
Save round-trips the result.

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

**WRITE side — DONE 2026-07-24 (`loom.vdbio`).** The open format question resolved to a **loom-native
OpenVDB `.vdb` encoder** (no NanoVDB/OpenVDB dependency on either end — I control ftrace's reader too):
uncompressed **`Tree_float_5_4_3`**, `COMPRESS_ACTIVE_MASK` only (no blosc/ZIP/half), full float32, a
`ScaleTranslateMap` transform matching numpy `linspace` endpoint-inclusive bakes. `tools/loom/loom/vdbio.py`
adds:
- **`write_vdb(path, [VolumeGrid(name, values, box), …])`** — serialise one or more dense `<f4` lattices
  to a multi-grid `.vdb` ftrace ingests directly (`density vdb:<path>` / `temperature vdb:<path>`). Positive
  voxels are vectorised into 8³ leaves under Internal<4>/<5> nodes; empty leaves are dropped (sparse).
- **`bake_field(field, box, res)`** — discretise any loom field/isosurface/callable to a dense `<f4` grid +
  world box (reuses the `mcubes` samplers), and **`write_volume(path, *, box, res, **fields)`** — bake each
  named field over a shared box/res and write them as named grids in one file (e.g. a `density`+`temperature`
  fire pair).
- **`read_vdb(path)`** — parses back the ACTIVE_MASK / full-float / **half** / **ZIP** subset (see the
  codec slice below) over a ScaleTranslate map into `{name: (dense_array, box6)}`.

Validated: loom round-trip is **bit-exact** (full float32, no lossy step); 6 tests in `tests/test_vdbio.py`
(891 loom green); and cross-validated end-to-end through ftrace — `scraps/make_loom_vdb.py` bakes a smoke
field to `scraps/loom_smoke.vdb`, and `scraps/loom_vdb.ftsl` renders it on both CPU and GPU (sparse device
path `1000/1000 bricks active`, energy `sum/emitted=1.000000`).

**READ codecs — half + ZIP + blosc + diagonal maps DONE 2026-07-24 (`loom.vdbio`).** The general read side:
loom now reads (and writes) the OpenVDB value codecs and axis-aligned transform maps real DCC tools emit —
**validated against genuine third-party files** (`scraps/_smoke.vdb` Houdini blosc smoke, `_fire.vdb`
density+temperature, `_sphere.vdb`/`_cube.vdb` UniformScaleMap level sets all read):
- **half-float** (`half=True`) — 16-bit voxels, grid type `Tree_float_5_4_3_HalfFloat`. Halves the file and
  is **read directly by ftrace** (which flags half by the type suffix). Cross-validated: ftrace ingests
  `scraps/loom_smoke_half.vdb` (`peak 0.9174` vs loom 0.9165; 1.16 MB vs 2.19 MB) and renders it.
- **ZIP** (`zip=True`) — value buffers zlib-deflated (OpenVDB int64 length prefix, negative = uncompressed).
  **Bit-exact** round-trip; read by any OpenVDB tool but *not* ftrace (LZ4-only) → interchange-only.
- **blosc** (`blosc=True`) — the DCC-standard codec. **Read** via the installed `blosc` package (handles
  BloscLZ/LZ4/Zlib/Zstd + shuffle — the full range; soft dep, clear install hint if absent). **Written** as
  LZ4 + byte-shuffle so it's read by **both** loom and ftrace → usable on the render path. Cross-validated:
  ftrace ingests `scraps/loom_smoke_blosc.vdb` (`peak 0.9173`) and reads the real Houdini `_smoke.vdb`.
- **Transform maps**: ScaleTranslate/UniformScaleTranslate/UniformScale/Scale/Translation (the diagonal,
  axis-aligned maps that keep samples on a regular lattice); a rotated `AffineMap` is rejected (can't land on
  a dense axis-aligned array). OpenVDB `0x1e` unique-name suffixes are stripped.
- The codec/map layer mirrors ftrace's `io::readData`/`readCompressedValues`/`readTransform`
  (`src/vdb_openvdb.cpp`); default output stays **byte-for-byte** the original file (test-asserted). The
  reader's per-voxel mask-expand + dense-fill loops were **vectorised** (numpy `unpackbits` + boolean scatter,
  bit-identical) — reading all four real samples dropped ~20 s → 2.8 s. 13 new tests (995 loom green).

**Rotated `AffineMap` — DONE 2026-07-26.** loom now decodes (and writes) every OpenVDB *linear* map,
closing a real asymmetry: ftrace's `readTransform` has always accepted `AffineMap`/`UnitaryMap` and
genuinely honours them (it inverts the 3×3 for world→index sampling and AABBs the index box's 8 corners),
while loom rejected them. The key realisation is that **a rotation costs the samples nothing** — an
OpenVDB tree is a regular lattice in *index* space either way, and the map only says where that lattice
sits — so the dense array is unaffected and the only casualty is the axis-aligned `box6`. That drove the
API shape: `read_vdb(path)` keeps returning `{name: (array, box6)}` and still **refuses** a rotated grid
(handing back an approximate box would silently misplace every voxel — the failure mode this project keeps
choosing against), while the new `read_vdb_grids(path)` returns `{name: ReadGrid}` with the index-space
array, its `index_lo`, and the full `VdbTransform` (`world = A·index + t`, mirroring ftrace's `A`/`T`).
`VolumeGrid(name, values, transform=…)` writes one back out as an `AffineMap`, so the path round-trips.
`is_diagonal` measures each off-diagonal against its own row's scale — unit-free, and tolerant of the
~1e-17 crumbs a DCC leaves when it composes a 90°/180° rotation in floating point. 8 new tests (1095 loom
green) plus an ftrace interop check: one asymmetric L-shaped volume written twice, diagonal vs 45°-about-Y
(`scraps/vdb_rot_make.py`), renders visibly rotated in ftrace with identical voxels.

**NanoVDB `.nvdb` ingest — DONE 2026-07-27.** `loom.vdbio.read_nvdb(path)` reads NanoVDB v32.6 float
`5_4_3` grids in **both** accepted layouts — a `FileHeader`-prefixed multi-grid container and a bare raw
grid buffer — and `read_vdb_grids` now **dispatches on magic**, so a caller that just wants "read whatever
volume this is" no longer has to know which format it holds. The item was unblocked (unlike Vec3, below)
because `scraps/cloud.nvdb` is *genuine third-party data* and `src/vdbgrid.cpp` is a completely independent
reader to check against — the same "must meet a real file" bar this module has held throughout.

The format is not a serialised stream but a **memory image**: a linear buffer of 32-byte-aligned PODs
referring to each other by *signed byte offsets*, laid out `GridData / TreeData / RootData+tiles /
Internal<5>… / Internal<4>… / Leaf<3>…`. Nothing to decompress — you index at fixed offsets and walk.
Every struct offset was byte-verified against the vendored `src/third_party/nanovdb/NanoVDB.h` *and* the
real file before a line of parser was written. Three details mattered:

- **Tiles are real data.** `LeafData::getValue(i)` returns `mValues[i]` with **no mask check**, and
  `InternalNode::getValue` returns `mTable[n].value` whenever `childMask` is off — active or not. So the
  bake must write every stored leaf voxel and expand every non-child tile, exactly as ftrace's
  accessor-driven bake does; `cloud.nvdb` has 10 active lower-level tiles (10 × 8³ voxels) that a
  mask-filtered read would silently drop. Hence the **deliberate convention split**, documented in the
  module docstring: `read_vdb` keeps only active-positive voxels (it round-trips loom's own writer),
  while `read_nvdb` is a faithful dense bake over the tree's active index bbox with inactive =
  `background`. `ReadGrid` gained a `background` slot for this (0 for fog, the half-band width for a level
  set, 0.03 in `cloud.nvdb`).
- **Bit/coord order is plain.** `Mask::isOn(n) = mWords[n>>6] & (1<<(n&63))` is little-endian over the byte
  array, so one `np.unpackbits(bitorder="little")` recovers it with no per-word shuffle; and the leaf slot
  `((i&7)<<6)|((j&7)<<3)|(k&7)` is *exactly* C order for an `(8,8,8)` reshape.
- **The transform carries over untransposed.** NanoVDB's `matMult` is row-major column-vector — the same
  convention `VdbTransform.a` stores — unlike OpenVDB's row-vector `AffineMap`, which needs the transpose.

**Validated three ways.** (1) Against ftrace's own reader: index bbox `41³ @ (30,30,30)`, world AABB
`0.3..0.71` and `peak 1.001` match ftrace's `[vdb]` line exactly. (2) Against NanoVDB's own redundantly
stored statistics: array min/max match `RootData`'s, every leaf's active min/max matches its
`mMinimum`/`mMaximum`, and `active_total + tiles[0]·8³ + tiles[1]·128³ == mVoxelCount` (33401) holds
exactly. (3) End-to-end through the renderer (`scraps/nvdb_roundtrip.py`): the `.nvdb` re-emitted as a
`.vdb` and rendered in the same scene by ftrace's *independent* OpenVDB reader agrees to **0.007%** in the
mean, and the per-pixel relative diff **halves with 4× photons** (8.65% → 4.35%, ratio 1.99) — √N, i.e.
pure independent-MC noise, not a volume difference. (The `.vdb` comes back 39³ rather than 41³ because
`write_vdb` trims the zero outer shell by design; that write-side trim shifts the AABB, which diverges the
RNG streams and is the whole source of the per-pixel noise.) 9 new tests (**1137 loom green**), and all
**11 layout-constant mutations are caught** — two (`_LF_VALUES`, `_RD_TILE_SIZE`) initially slipped through
and drove two extra tests: an *independent* breadth-first leaf traversal via `mNodeOffset[0]` + stride 2144
(not the reader's child-offset descent) with a per-voxel index check, and a root-tile-stride test asserting
the root table ends exactly where the first upper node begins.

**Volume transforms / read→transform→write — DONE 2026-07-27.** The capability E4 was actually *about* —
"use an existing volume as a **basis**, transform it, then emit the result" — now exists, and it needed
almost no new machinery, because the right move was to make a volume a **term in the spatial algebra**
rather than to invent a volume-transform API. `loom.spatial.VolumeField` is the 3-D twin of the existing
`Image` leaf: a scalar SpatialExpr whose value at a world point is an imported grid's trilinearly
interpolated density. (Named `VolumeField`, not `Volume` — `loom.scene.Volume` is already the `medium { }`
scene element, and shadowing it would have been a trap.)

Everything else then falls out of machinery that was already there:
- **value ops / modulation** — `cloud * (0.5 + 0.5 * sin(20 * Y))`, mixing two volumes, animated `Signal`
  coefficients: all just the spatial algebra.
- **warping** — `x`/`y`/`z` are ordinary sub-expressions exactly like `Image`'s `u`/`v`, so
  `VolumeField(p, x=X + 0.1 * sin(10 * Z))` bends the volume, and `substitute` reaches inside. (Standard
  resampler convention: the coordinate expressions map the *destination* point back to the *source*, so a
  warp is authored as its inverse map — documented loudly.)
- **meshing** — `mcubes` takes any callable field, so marching cubes over imported data is free.
- **resampling** — `bake_field`/`write_volume` discretise onto any box/res, which is all "resample a grid"
  ever meant. Sparse↔dense as *separate storage backings* stays unbuilt, and is now clearly a storage
  optimisation rather than a capability: nothing is missing from the user's point of view.

**Placement is lossless, and that's the design point.** `translated`/`scaled`/`rotated`/`fitted` do not
resample — they compose a world-space affine onto the grid's own index→world map via the new
`VdbTransform.premultiplied` (`A' = M·A`, `t' = M·t + d`). Moving a volume costs zero interpolation because
a VDB tree is a regular lattice in *index* space and only the transform says where it sits; error enters
exactly once, at the final bake. That is loom's "keep everything as functions; discretize last" rule
applied to volumes, and it's test-asserted: rotate 4×90°, rotate 360°, translate-and-back and
scale-and-back all reproduce the original array (max |Δ| ~1e-15), and `.rotated(37°).read_grid.values is
g.values` — not one voxel is touched. A complementary test asserts a *single* 37° rotation really does
change the field (so the round-trips can't pass vacuously) while conserving mass to 5%.

`VolumeField.emit()` deliberately **raises**: ftrace's pattern VM has no volume-sampling opcode, so there
is no ftsl string it could honestly become. It is bake-only, and the error says so and names
`write_volume`. Supporting infrastructure added alongside: `VdbTransform.inverse_linear`/`to_index`/
`premultiplied`/`__eq__`, `ReadGrid.world_box` (AABB of the eight index-box corners — defined for a rotated
grid, unlike `.box`) and `ReadGrid.with_transform` (shares the array; repositioning copies nothing).

**Found and fixed a real ftrace bug on the way (v0.84.2).** Porting ftrace's sampler into
`ReadGrid.sample` exposed that `VdbGrid::sample` (`src/vdbgrid.h`) and its CUDA twin `dVdbSample` clamped
the stencil *indices* to `[0, n-1]` but not the interpolation *fraction*, so in the half-voxel shell just
below index 0 the sample was dominated by the **second** voxel and got wronger the further out it went.
Symptom that caught it: a 360° rotation, which must be a no-op, moved the baked field by up to 0.32. Fixed
in both samplers by clamping the *coordinate* before the floor — bit-identical inside the lattice, and it
drops three `floor()` calls from the hot path. See `known-issues.md`.

19 new tests (**1150 loom green**); the sampler fix is mutation-verified (reverting to the stencil-only
clamp fails two tests).

**Still open:** **Vec3** grids; sparse *storage* (as opposed to sparse-source reads, which work) and
transforms authored directly against it. Note loom has no NanoVDB *writer* — `.nvdb` support is read-only,
and there's no demand for the write end (ftrace reads `.vdb` happily, and loom's `.vdb` writer is
byte-verified).
- **Vec3 grids are BLOCKED on validation data (assessed 2026-07-26).** None of the four real sample files in
  `scraps/` carries a `Tree_vec3s_*` grid (all four are `Tree_float_5_4_3_HalfFloat`), and there is no
  installable OpenVDB Python binding on this platform to synthesise one (`openvdb` / `pyopenvdb` /
  `openvdb-python` all fail to resolve on PyPI). Writing the reader blind against the spec would produce
  something validated only against loom's own writer — precisely the "works until it meets a real file"
  failure this module has so far avoided by testing against third-party output. Also note there is **no
  consumer**: ftrace supports scalar float grids only, so a Vec3 read would serve loom-internal use
  (velocity/advection) that isn't designed yet. Unblock by sourcing one real vec3 `.vdb` (a DCC export or an
  openvdb.org sample), then the tree walk is the existing one with a 3-wide value stride.

### E5 — Axis-typed signals: one influence model (broadcast / pointwise / reduce) + mod·pin + sample·select grammar  ✅ DONE 2026-07-26  *(loom; LARGE, design; unifies E2/E4 and records-5a)*
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

**FOUNDATION DONE 2026-07-24 (`loom.axes`).** The deferred open-q (concrete node taxonomy + axis-set
representation) is resolved with a small additive node set that sits *on top of* the existing scalar
`Signal` DAG (no churn — all 891 prior tests stay green; 912 total now). An **`AxSignal`** is a pure
function of a **point** (a `dict[str, float]` mapping axis names → coords) and carries `.axes`
(`frozenset[str]`, its free variables). The four E5 pillars are implemented and tested
(`tests/test_axes.py`, 21 tests):
- **Axis-set inference + broadcast/pointwise** — `Ax(name)` coordinate leaves, `AConst` (axes ∅),
  `AFn`/arithmetic compose with `axes = union(children)`. Evaluating at a point with *extra* axes ignores
  them (broadcast, implicit); a *missing* required axis errors. Shared axes combine pointwise; the illegal
  cross-`t` op is inexpressible (a node only ever sees the current point). No `t`-influences-`t` detector,
  no spatial/temporal type split — exactly as the design demands.
- **pin/mod edges + target-declared neutrals** — `Target(kind, [Binding(source, mode, gain)], base)` with
  `kind ∈ {ADDITIVE (neutral 0, y+=g·x), GAIN (neutral 1, y*=x**g), BIPOLAR (neutral ½, ½-centred clamped)}`;
  `mode='mod'` accumulates toward the neutral, `mode='pin'` is last-write-wins (gain blends). The target
  declares its normalization, not the edge — so "mod" is one authoring mode resolving to the domain-correct
  operator.
- **sample/select grammar** — `Sample(fn, arg)` is the continuous `curve(t)` form (binds the curve's param
  axis to `arg`, so `axes = arg.axes` and it broadcasts); `.comp(i)` picks a component (`curve(t).y`);
  `select(items, i)` is the discrete constant `R.chan[i]` selector.
- **The one cross-axis node** — `Reduce(body, axis, samples, op)` (`sum|mean|min|max|integral`) is the only
  node that reads other points along an axis; `axes = body.axes - {axis}`. Explicit by construction; a plain
  `AFn` can never smuggle in a cross-index read. (The cross-`t` "video node" already lives in `loom.xvideo`.)
- **Bridge** — `Lift(signal)` wraps any legacy `{t}`-typed `Signal` into the axis layer, so the new model
  composes with the whole existing DAG; `detect_signal_cycle`/`walk` duck-type over axial nodes too.

**FOLLOW-UP 1 DONE 2026-07-24 (`loom.axes`).** Folded the existing clock-parameterized interp curves and
records into the sample grammar so it binds *their* param axis directly rather than forcing a pre-baked bare
callable (which could not thread the clock the control points depend on). Pieces: **`CurveSample(curve, arg,
*, clock_axis='t', loop=True)`** wraps any loom curve exposing `.sample(u, clock, cache)` (`LoopCurve`,
`TrackedCurve` tracks, `FieldCurve` position) — `arg` binds the curve's param axis and the node *additionally*
depends on the clock axis, so an **animated** spatial curve types as `{s,t}` (its shape moves over time) while
a static one broadcasts trivially over `t`; **`RecordSample(record, channel, arg)`** binds a `Record`'s driver
axis (a static `{driver}` LUT — no clock, via `Record.sample_vec`); and a unified **`sample(obj, arg, …)`**
dispatcher picks `RecordSample` / `CurveSample` / `Sample` by duck-type (Record needs `channel=`). Both new
nodes thread the loom curve/record into the axis-layer `walk` (like `Lift`), so a cycle through a control point
stays catchable. 9 new tests (`tests/test_axes.py`, 30 total; static-broadcast vs genuine-`{s,t}`, component
pick, custom clock axis, record scalar/vector, dispatch, `Reduce`-over-`s` compose, walk reaches control
points). 1023 loom green.
**FOLLOW-UP 2 DONE 2026-07-26 (`loom.axes` + the coercion path).** *Routing scene value-sites through
`Target`* — the piece that turns E5 from a self-contained algebra into loom's actual authoring model.
- **`Lower` / `LowerVec` — the exact inverse of `Lift`.** Every loom scene value-site (`Sphere.radius`,
  `Isosurface.iso`, a material colour, a camera position, …) consumes a clock-parameterized
  `Signal`/`VecSignal`, so a `Target` only reaches a scene variable through a node that binds an `AxSignal`
  back down. The site's clock axis (default `'t'`) is fed `clock.t`; every **other** axis the node reads must
  be pinned via `bind={'s': <coord or Signal>}` — a constant reads *one* arclength of a spatial curve, a
  `Signal` **sweeps** along it over the loop. `lower(node)` picks the scalar vs vector form by probing at
  `t=0` (or pass `dim=`). `LowerVec` evaluates the axis graph **once** per frame and caches the whole tuple,
  while still exposing per-component `Lower` children so `walk`/`detect_signal_cycle`/`VecSignal` math see an
  ordinary vector node.
- **Records-5a's scope rule is enforced at CONSTRUCTION.** "A node's free variables ⊆ the axes in scope at
  this site" — a scene value-site has exactly one axis in scope (the clock), so an unbound axis raises
  immediately, *naming* it and suggesting `bind=`, instead of failing deep inside a render.
- **One memoised hook, not N constructor changes.** `signals.core.lower_axsignal(x)` is the single coercion
  point, consumed by `as_signal`, `VecSignal.of`, `ftsl_emit.site_node` (→ `num`/`vecn`/`value_token`) and
  `Element.roots()`. No element constructor changed and *every* value-site accepts an axis node uniformly.
  The memoisation (`x._site_node`) is **required, not cosmetic**: node identity is the per-frame `Cache` key
  **and** `roots()` must hand the cycle detector the very node emission will evaluate — lowering twice would
  silently defeat both.
- **Sugar + a latent bug.** `mod(src, gain)` / `pin(src, gain)` build `Binding`s; `Binding` now coerces its
  source through `as_ax`, which also `Lift`s a legacy `Signal` (so `mod(0.6 + 0.4*Sine())` just works); an
  unknown edge mode is rejected at construction. Found and fixed a pre-existing bug in `_accumulate`: a
  `GAIN` target with a **negative** source computes `x ** gain`, which Python evaluates to a *complex* number
  that then blew up in `float()` far from the cause — it now raises a domain error naming the fix.
25 new tests (`tests/test_axes.py`, 55 total; mod/pin sugar, binding coercion + bad mode, `as_ax` lift, the
GAIN domain error, `Lower` against the clock, the construction-time unbound-axis error, `bind=` to a constant
and to a `Signal`, `LowerVec` dim probe / scalar-node / dim-mismatch / per-frame cache / component identity,
`lower()` dispatch, curve sweep, `as_signal` + `VecSignal.of` coercion both ways, site-node memoisation, and an
end-to-end `Target`→`Sphere.radius` + swept `CurveSample`→`Sphere.center` round-trip through `emit`).
1120 loom green.
**FOLLOW-UP 3 DONE 2026-07-26 (`loom.axes` + the viewer sidecar + `src/viewer_gui.cpp`) — E5 IS COMPLETE.**
*The on-disk projection of axis annotations* — the second half of the deferred open-q. It resolves into a
**decision** and an **implementation**.
- **The decision: `.ftsl` carries no axis annotation, and shouldn't.** `.ftsl` is a **bound**, per-frame
  projection. By the time a scene emits, the clock axis has been fixed to `clock.t` and every other axis
  pinned by `bind=`, so an `{s,t}` node has already collapsed to a *number*. ftrace renders one frame and has
  no notion of an axis; annotating its language would turn `.ftsl` into an animation format and move the
  animation authority out of loom — against loom's core ideas 2 ("discretize LAST, per frame") and 5
  ("emit-`.ftsl`-first"). This is written down in `loom/axes.py` beside the code so it stops being re-litigated.
- **The implementation: the viewer introspection sidecar, v1 → v2.** That is the on-disk representation an
  *editor* reads (F1/F5), and it is where the annotation belongs. `axis_annotation(node)` and
  `binding_edges(target)` live in `loom/axes.py` — the model owns its own serialisation and `loom.viewer` just
  merges the dicts in.
  - **Nodes** gain their free `axes` (`{s,t}`; `[]` for a constant, which broadcasts everywhere), plus
    `target_kind` + `neutral` (a `Target`'s declared quantity type), `reduces` + `reduce_op` + `samples` (the
    *only* cross-axis node — worth surfacing), `component` / `channel` / `leaf_axis`, and on the two bridge
    nodes `site` (scalar/vector), `clock_axis`, `bound_axes` and `source_axes` — together the value-site's
    entire axis scope ("`t` from the clock, `{s}` pinned, reading an `{s,t}` node").
  - **Edges** out of a `Target` gain `mode` (`pin`/`mod`) and `gain`, and are named `mod[i]`/`pin[i]` instead
    of an anonymous `in<i>`. This is the piece a plain child list **cannot** express: the sources hang off
    `Binding` records, so a generic DAG walk saw only unlabelled inputs and an editor could not tell a `mod`
    from a `pin`, let alone at what gain.
  - **`src/viewer_gui.cpp`'s F5 panel renders all of it** — an `axes {s,t}` chip on the node, a one-line
    kind/scope caption (`gain target (neutral 1)`, `reduce s (mean, 8 samples)`, `t from clock, {s} pinned <-
    {s,t}`), and `mod[0] x0.8` on the input pin. Verified on a live viewer window.
  - Purely **additive**: a v1 reader ignores the new keys, and an unannotated (legacy-`Signal`) node renders
    exactly as before.
8 new tests (`tests/test_viewer.py`, 58 total). 1128 loom green.

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
- [x] **F1 — scene/object enumeration + `build()` load contract.** ✅ 2026-07-24 (**loom side**, `loom/viewer.py`).
      Load a loom file, call its `build(clock, **params)`, walk the resulting `Scene`, and produce the
      selectable-object data the viewer needs. Since the viewer is C++ and loom is Python, the loom side is
      the **load contract + a JSON introspection sidecar** (the C++ host that renders the list/panes is F2–F7).
      - **`build()` contract (documented in `viewer.py`):** `def build(clock=None, **params) -> Scene` —
        returns a **fresh** Scene each call (no module-level `scene`; **side-effect-free at import** — importing
        never renders/emits/opens windows), so the viewer re-derives geometry live. `clock` optional (viewer's
        scrub frame; `None` ⇒ frame-0). Keyword params with defaults are surfaced as UI controls.
      - **`load_build(path, func="build")`** imports a scene file → its `build`; **`ViewerModel(build, **params)`**
        / `.from_file(path)` wraps it: `.scene(clock, **overrides)` builds, `.declared_params()` returns the
        build's advertised keyword defaults, `.introspect(clock)` / `.save_sidecar(path, clock)` emit the sidecar
        (atomic write). `build_scene(build, clock, **params)` passes `clock` only when the signature accepts it.
      - **`introspect(scene)` sidecar:** `version`, `frame`; `objects` (geometry elements — Sphere/Beads/
        SweptMesh/IsoMesh/Group/Volume/Raw/Isosurface — Groups recursed, each linking the `datasets` it
        references by node id + kind-specific meta: name/material/count/res/iso/closed_*); `datasets` (every
        `PointPath`/`TrackedPath`/`Grid`/`Scatter` reachable from geometry **or** materials/lights/camera, with
        dim/shape/lo/hi/channels/tracks); minimal `camera`/`lights`; and `dag` (modulator graph — `nodes` =
        id+op+label, `edges` = child→parent; per-edge parameter labels deferred to F5). 20 tests
        (`tests/test_viewer.py`): contract loader (missing/non-callable/custom-name), clock passing/param
        forwarding, object kinds + dataset linking, path/grid/scatter coverage, DAG node/edge integrity,
        ViewerModel scene/introspect/declared-params/from-file/save-sidecar. 972 loom green.
      - **C++ host DONE (F2 complete):** the viewer is a **`-viewer <sidecar.json>`** mode of the ftrace
        binary — **Win32 + Direct3D 11 + Dear ImGui** (vendored `src/third_party/imgui/`, added to the
        `ftrace` CMake target, dispatched from `main()` → `src/viewer_gui.cpp`). It reads the sidecar via the
        vendored `minijson` parser and shows the Scene/Objects/Datasets panels + the full N-D curve pane
        (3-of-N dim picker, index markers, mono/anaglyph/wall/cross stereo). ImPlot & imnodes are vendored
        but not yet compiled (reserved for F3/F5). Remaining panes are F3–F7.
- [x] **F2 — N-D curve 3-D view.** ✅ 2026-07-24 Show an N-D curve by picking **3 of N** dims to display. **Rotating
      the displayed dims = a view-only transform** (no recompute); **rotating into other dims = recompute
      the projection.** Index markers along the curve show curve progression. **Stereoscopic viewing:**
      wall-eyed (L|R) and cross-eyed (R|L) side-by-side, plus **red-cyan anaglyph** — using the §I
      off-axis stereo machinery (shared with the renderer's still/movie stereo).
      - **Slice A (loom) DONE** — the introspection sidecar now carries real curve geometry: every
        path/tracked_path dataset gets `control_points` (the animated control points at the frame's clock)
        and a display `polyline` (96 samples via the same `eval_curve` midpoint-quadratic-Bezier the engine
        uses; a closed spine repeats its first point to wrap). Tested in `tools/loom/tests/test_viewer.py`.
      - **Slice B (C++ host) DONE** — the native viewer now exists as a **`-viewer <sidecar.json>`** mode of
        the ftrace binary (Dear ImGui + Win32 + Direct3D 11; vendored under `src/third_party/imgui/`, wired
        into the `ftrace` CMake target, dispatched from `main()` in `src/viewer_gui.cpp`). It reads the F1/F2
        sidecar with the vendored `minijson` parser and shows a Scene summary (camera/frame/lights/DAG
        counts), an **Objects** table (name/kind/material/linked-dataset-ids), a **Datasets** table
        (id/kind/detail), and a right-hand **curve pane** that draws every dataset's sampled polyline (with
        control-point markers) in a drag-to-orbit / wheel-to-zoom orthographic projection.
      - **Slice C (C++) DONE** — the curve pane is now the real N-D pane. **Dim picker:** three combos map
        any of the curve's N dims to screen X/Y/Z (loom curves are full N-D — the sidecar carries every
        coordinate, verified on a 5-D test curve); reassigning them is a pure view-only re-projection (no
        recompute), and the widest curve dimensionality in the scene drives the choices. An **index slider**
        highlights a position along the curve, with progression dots drawn every ⅛ of arclength. **Stereo:**
        a mode combo selects mono / **red-cyan anaglyph** / **wall-eyed (L|R)** / **cross-eyed (R|L)**
        side-by-side, with an eye-separation slider; each eye re-renders the same curve at a small yaw offset
        (anaglyph tints the two eyes red/cyan and overlays them; wall/cross split the canvas into two
        half-width viewports). Shares the §I off-axis stereo idea.
- [x] **F3 — scroll-locked strip charts (ImPlot).** ✅ 2026-07-24 Below the 3-D pane, one strip chart
      **per curve dimension** and one **per tacked-on channel** (TrackedCurve). Shared index markers along
      the bottom cross-reference the 3-D index dot. **All charts scroll left/right together (scroll-locked,
      never individually)** to page through the whole curve. Hover/click on a chart cross-highlights the 3-D
      index dot and vice-versa.
      - **loom side:** the sidecar's `tracked_path` datasets now carry a `channels` array — each track
        (`speed`, `aim`, …) sampled along the **same** curve parameter as the display polyline (scalar
        tracks → 1-vectors, vector tracks → N-vectors; closed paths wrap), so the strip charts line up
        sample-for-sample with the 3-D curve. Tested in `tools/loom/tests/test_viewer.py`.
      - **C++ side:** vendored **ImPlot** (`src/third_party/implot/`) is now compiled into the ftrace
        target. The `-viewer` right pane splits into the 3-D curve pane (top) and a stack of strip charts
        (bottom): one per polyline dimension **d0…dN** plus one per channel component. All charts share a
        **linked X axis** (`SetupAxisLinks`) so panning/zooming any one pages them all together, and each
        carries a **draggable yellow index line** wired bidirectionally to the 3-D pane's index dot
        (dragging a chart line moves the dot; the index slider moves every line).
- [x] **F4 — SweptMesh tessellated view + textures + decoupled re-tessellation.** ✅ fully done
      2026-07-28 (VERSION 0.92.0) with sub-item (2); ✅ core done
      2026-07-24 (VERSION 0.55.0). Slice A (loom, `b13122f`): each `swept_mesh` object record now
      carries a `mesh` key with the tessellated triangle mesh at the clock — `vertices` (flat
      3-vectors), `faces` (0-based index triples), per-vertex `uvs` (u along spine, v around profile),
      `rings`/`profile_count` — mirroring `SweptMesh.emit`'s `sweep_rings`+`skin_rings` without writing
      an OBJ. 2 new tests. Slice B (C++ viewer): a **Meshes tab** (`collectMeshes`/`drawMeshPane`,
      `MeshView`) draws the surface as a **shaded triangle mesh** in a 3-D orbit pane, with **flat
      two-sided lambert shading**, a **wireframe** overlay, and a colour
      selector (grey / per-object tint / **UV checker**). Orbiting the 3 spatial dims is the **view-only
      re-projection** the rotation rule calls for. A swept-mesh scene opens on the Meshes tab by default
      (its spine curves still populate the Curves tab). Verified via PrintWindow screenshot.
      - **(3) real z-buffer** — ✅ **DONE 2026-07-28 (VERSION 0.96.0).** The pane shipped as a CPU
        **painter's-algorithm** centroid sort into an ImGui draw list, which cannot resolve
        interpenetrating surfaces (loom's own `examples/viewer_live.py` — an `orbit` tube threading a
        gyroid ball — rendered wrong) and re-sorted every triangle on the UI thread every frame.
        `MeshGpu` now uploads the tessellation **once** into one vertex + index buffer (per-mesh
        `firstIndex/indexCount/baseVertex` ranges keep the per-mesh skin/tint draw calls) and renders
        it into an offscreen RTV with a **`D32_FLOAT` depth-stencil view** via runtime-compiled HLSL,
        shown with `ImGui::Image` — the pattern the Render pane already used. Re-upload is keyed on
        `MeshView::geomGen` (bumped in `adoptSidecar`), so an orbit is one 144-byte cbuffer write;
        union bounds are baked with the upload instead of rescanned per frame. Shading is preserved
        exactly (`0.30 + 0.70*|n.z|`, face normal from `cross(ddx,ddy)` — exact under the orthographic
        orbit), the wireframe became a real depth-tested `D3D11_FILL_WIREFRAME` pass, and the UV
        checker moved from a per-triangle centroid sample to **per-pixel** (the one deliberate
        behaviour change; a UV checker exists to show distortion *within* a face). Verified by
        PrintWindow screenshots of all four colour modes, wireframe, zoom, a window resize and a live
        `-loom` re-derive. The **curve and field panes still project on the CPU** — same port pending.
      - **(1) textures** (image *or* formula) — ✅ **DONE 2026-07-26 (VERSION 0.58.0).** **Loom half
        2026-07-24** — the sidecar emits a `materials` list (each material's `type`/`props` +
        the `texture` skin it binds, animated props evaluated at the clock) and a `textures` list
        (image `file`/encoding/filter/wrap, or formula `r`/`g`/`b`/`res` — `Texture` vs
        `ProcTexture`), so the viewer sees *which* skin a mesh wears and where to get it.
        4 new tests. A 5th test landed 2026-07-26 with the fix for a real serialisation bug: a
        material *bundle* whose colour slot is a `SpatialExpr` lowers to a `ProcTexture` whose
        channels are **live expression objects**, so the whole sidecar dump died with
        `Object of type _Bin is not JSON serializable`; `_describe_texture` now bakes each channel
        to its ftsl source at the clock via `ProcTexture._chan_str`, i.e. the viewer compiles the
        *identical* string ftrace would. **C++ half 2026-07-26** — `SkinLib`/`Skin` in
        `src/viewer_gui.cpp` decode every sidecar texture into a D3D11 SRV and the Meshes tab
        gained a 4th colour mode, **texture** (now the default), which draws each triangle with
        `PrimReserve`/`PrimVtx` at its **interpolated per-vertex UVs** (batching `PushTexture` per
        material rather than per triangle) modulated by the lambert shade. Image skins go through
        ftrace's own `Texture::load` (path tried verbatim, then relative to the sidecar dir);
        formula skins are baked on the CPU through ftrace's own pattern VM
        (`compilePatternExpr`/`patternEval`), a line-for-line mirror of `FtslLoader::addTexture`,
        so the preview can't drift from the renderer. A `PatTexScope`/`PatCtx::texFn` pair lets a
        formula sample an already-decoded image with `tex:<name>(u,v)` — same "images must be
        declared above" ordering rule as ftrace. Linear texels are sRGB-encoded on upload
        (IMMUTABLE `R8G8B8A8_UNORM`, ≤2048 px edge). Anything unusable — missing file, syntax
        error, unknown `tex:` name — degrades to grey lambert and prints a red per-skin reason
        under the pane instead of crashing or drawing black. `runViewerGui` also now calls
        `ImGui_ImplWin32_EnableDpiAwareness()` + `ScaleAllSizes(dpi)`/`FontScaleDpi` so the whole
        GUI renders at native resolution instead of being DWM-upscaled and blurry. Verified by
        PrintWindow screenshot on a purpose-built 4-tube scene covering all four paths (image
        skin, procedural formula, formula-sampling-an-image, no texture) plus a deliberately
        broken sidecar for the error path.
      - **(2) Re-tessellation when rotating *into* a parameter/extra dimension** via a latest-wins
        off-thread job queue — ✅ **DONE 2026-07-28 (VERSION 0.92.0).** **Loom half DONE 2026-07-24** — the
        viewer↔loom **live re-introspection channel** (`ViewerSession`/`serve_viewer` in
        `loom.viewer`, plus a `python -m loom.viewer <scene.py>` CLI entry): a resident loom process
        holds a `ViewerModel` and answers newline-delimited-JSON `introspect {clock,params}` requests
        with a fresh sidecar (the thing the frozen sidecar can't do), mirroring `loom.anim.serve_live`
        in the viewer→loom direction. 9 new tests (`tests/test_viewer.py`, 1004 loom green). The channel
        also gained an **`emit`** command 2026-07-24 (re-emit `.ftsl` for a clock/params) that **F7's
        in-process primary path (v0.56.0) already uses the static form of** — the viewer parses loom's
        emitted `.ftsl` and raymarches it live. **C++ half 2026-07-28** — a new `-loom <scene.py>` flag
        (paired with `-viewer`; the sidecar's own `build` provenance key is the fallback) has
        `runViewerGui` spawn `python -X utf8 -u -m loom.viewer <scene.py>` and hold the channel open:
        `LoomLink` owns the child + pipes, `LoomBridge` owns **one worker thread and a one-slot pending
        job** so `post()` overwrites anything not yet started — the latest-wins rule, which makes a
        continuous drag cost one bake of the final value instead of one per frame. A **Live (loom)**
        panel in the left column exposes link state, a clock scrub, one typed control per declared
        keyword param, an `auto` / `re-derive now` switch for slow scenes, and `posted / baked`
        counters (`posted > baked` = the collapsing working). Marking one continuous param the **sweep
        axis** makes a **right-drag on any 3-D pane rotate into that dimension** — the gesture this item
        existed for. Each bake asks for `introspect` **and** `emit`, so the returned sidecar refreshes
        Curves/Fields/**Meshes** while the returned `.ftsl` re-seeds F7's Render pane; results are
        adopted on whatever frame they land, preserving orbit, zoom, active tab and DAG layout, and a
        failed bake shows its error while the last good geometry stays on screen. Scratch sidecars/
        `.ftsl`/`.obj` go to a per-process `%TEMP%\ftrace_viewer_<pid>` dir that the bridge prunes as
        results are consumed and removes wholesale on exit. Verified end-to-end plus a 20-round
        scripted sweep with no crash and no memory growth. Two real bugs fell out of that validation
        and are fixed + written up in `known-issues.md` (the imgui #7543 / imnodes node-rect crash, and
        the DAG pane re-packing itself when clipped to zero height). Same channel unblocks F7's live
        field edit.
- [x] **F5 — modulator-DAG panel (imnodes).** ✅ 2026-07-24 Introspect the signal DAG via loom's `walk()`
      and lay it out well. Each node shows the **op/function that modulates it** and a **stable identifier**;
      each **edge is labeled with the parameter name it feeds**, so you can tell which variable in a node's
      function refers to which upstream node.
      - **loom side:** `_describe_dag` now tags every edge with a `param` label — the name of the input on
        the destination node that the upstream child feeds, derived by identity-matching the child to the
        attribute it's stored under (`a`/`b` for arithmetic operands, `cycles`/`phase`/`amp`/`bias` for a
        `Sine`, `components[i]` for a `VecSignal`; positional `in<i>` only when nested out of reach). Tested
        in `tools/loom/tests/test_viewer.py`.
      - **C++ side:** vendored **imnodes** (`src/third_party/imnodes/`) is compiled into the ftrace target.
        A **Modulator DAG** panel in the `-viewer` left column renders the graph with imnodes: each node is
        a box titled `<op> #<id>` (with its constant/leaf value shown when distinct), one **labelled input
        pin per incoming edge** (the param name), and one output pin; links connect upstream outputs into
        those labelled inputs. A longest-path layering places leaves (constants/oscillators) on the left and
        the params they drive on the right; pan/zoom to explore.
- [x] **F6 — scatter + grid field display & inspection.** ✅ 2026-07-24 Slice A (loom,
      `dea2abd`): `_describe_dataset` now emits real field geometry — scatter `points` (sample
      positions) + `values` (per-sample channel-vectors), grid `axes` (per-axis lattice coords) +
      flat C-order `values`; scalar values normalised to 1-lists so the C++ side sees uniform
      channel-vectors. 3 new tests (grid geometry, scalar scatter, vector-rgb scatter). Slice B
      (C++ viewer): a new **Fields tab** in the right pane (`drawFieldPane`/`collectFields`,
      `FieldView`) renders scatter/grid sample points in the shared 3-D orbit view (grid node
      positions reconstructed from axes+shape in C-order), **3-of-N dim pickers**, a **colour
      selector** (heatmap of a chosen channel — with a channel slider for multi-valued fields — or
      **ch0/1/2 → RGB**), **click-to-inspect** (nearest projected point within 14 px → prints its
      position + every channel value), and **per-extra-dim slice sliders** for N-D grids (dims not
      shown collapse to a chosen lattice index). Tabs auto-hide when their kind is absent so the
      present one is default-selected. VERSION → 0.54.0. Verified via PrintWindow screenshot.
      *(No glyphs yet — points only, as speced; deferred as a later polish.)*
- [x] **F7 — isosurfaces via `-raster-gpu` raymarch (primary) + MC-mesh fallback.** ✅ **primary path
      DONE 2026-07-24** (v0.56.0), ✅ MC-mesh fallback done 2026-07-24.
      - **Primary path (in-process raymarch) — DONE 2026-07-24 (v0.56.0).** Took the **in-process**
        route rather than the `-serve` subprocess: the `-viewer` **is** the ftrace binary, so it already
        contains both the `.ftsl` parser (`ftsl::load`) and `renderIsoPreviewCuda`. loom's
        `ViewerModel.save_sidecar` now emits the scene's **`.ftsl` beside the sidecar** and records its
        absolute path under a new sidecar **`"source"`** key (and a matching `ViewerSession` **`emit`**
        command re-emits `.ftsl` for a new clock/params over the live channel — for scrub/param/edit).
        `viewer_gui.cpp` parses `source` with `ftsl::load` and adds a **"Render" tab** (default-selected,
        the primary view) that calls `renderIsoPreviewCuda(scene, orbitCam, W, H, …)` in-process, blits
        the RGB frame into a **D3D11 dynamic texture** (`ImGui::…AddImage`), and drives an **orbit camera**
        (drag = yaw/pitch, wheel = dolly) around the scene bounding sphere; re-renders only on camera
        change (idle pane is free). Validated end-to-end: a gyroid `build()` → sidecar+`.ftsl` → the
        viewer's Render tab sphere-traces the real field (identical to `-raster-gpu`, **no tessellation**).
        This replaces F7's static MC-mesh with the actual field for the native viewer too. New loom tests
        (6) + the whole loom suite green; C++ compiled with the new `ftsl.h`/`render_cuda.h` includes.
        *Deferred within F7:* the `-serve` streaming path (only needed if the raymarch is ever pushed to a
        separate process). (F4's C++ texture display — image/formula → sampled D3D11 texture at mesh UVs —
        was still open when this was written; it landed 2026-07-26, see §F4.)
      - **How the primary path used to be scoped — RE-SCOPED 2026-07-24 after an architecture audit.** The
        field-raymarcher itself **already exists and already ships**: `-raster-gpu` (feature **G2**) casts
        primary rays straight at the implicit **with NO tessellation** — `renderIsoPreviewCuda` →
        `kIsoPreview` → `closestHit` → `intersectImplicit` sphere-traces the postfix field bytecode
        (`render_cuda.cu` ~L1801/2201), the *same* field VM the full path tracer uses. And it is already
        the **shared** preview renderer: the `rasterOne` lambda (`main.cpp` ~L4923) routes *every* preview
        consumer through it — single stills, flyby frames, **and the interactive `-explore`/`-fly` loop +
        the camera-curve editor inside it**. So `-explore`/`-fly`/the editor **already raymarch the field
        live, no re-tessellation**, whenever launched with `-raster-gpu`. The one consumer NOT yet using it
        is the **native ImGui `-viewer`**, which still shows F7's static **MC-mesh fallback**.
      - **So F7's remaining work is NOT a new raymarcher** (and NOT a duplicate D3D11/HLSL field VM in the
        viewer — that would fork the evaluator). It's a **bridge**: drive the existing `-raster-gpu`
        raymarcher over the **`ftrace -serve` pipe** and blit its frames into the viewer pane, reusing the
        §F4/§F7 **live re-introspection channel** (`ViewerSession`/`serve_viewer`, loom half done
        2026-07-24) + the M12 resident `-serve` server. When the user edits the field, loom re-emits the
        `.ftsl` (the `function { expr }` `Isosurface.emit` already produces) → the resident `-raster-gpu
        -serve` re-renders instantly. This is **C++ interactive-viewer work** (spawn/drive the resident
        process, present its frames, wire scrub/param/edit) — best done with the user present; the MC mesh
        is the working stand-in until then. Textures via **G5** + the F4 material/texture sidecar (done).

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
- [x] optional **`"extrapolate"`** — linear extrapolation off the boundary cell. **DONE 2026-07-24**
      (`tools/loom/loom/interp.py`): `_cell_base_frac` returns an *unclamped* fraction (`f<0` below,
      `f>1` above) on the boundary cell, so the linear stencil extends the edge cell's slope instead of
      edge-extending; cubic reuses the same unclamped fraction through the existing phantom-point machinery
      (reproduces linear ramps off-edge exactly). 3 tests in `test_gridinterp.py`.
- [x] Re-raise the `FieldCurve` dimension-mismatch `ValueError` with FieldCurve context (names the curve's
      dim). *(J1 complete bar the optional extrapolate mode.)*

### J2 — Placed isosurfaces + a Room/Group element (multi-changing-isosurface pipeline)
**Problem.** `Isosurface` (`iso.py`) has **no position**: its frame is `freq*(row·xyz)+drift`, with no
`- center`. So a gyroid clipped to a container at (5,0,0) shows the *same phase* as one at the origin —
the container moves but the pattern does not follow. (A translation is *expressible* today by hand-folding
`drift' = drift − freq*(M·c)`, but the `contained_by` box/sphere won't track it — manual and error-prone.)
loom already has the affine machinery to fix this cleanly: `mathnd.Affine` (`linear @ x + offset`, both
animatable, composable) and `spatial._offset`.

**Decision (the two missing primitives):** — **J2 DONE (2026-07-24).**
- [x] **Placement on `Isosurface`** — added an animatable `placement` `VecSignal` that offsets **both** the
      coordinate frame (`freq*(M·(x − placement))+drift`) **and** the `contained_by` box/sphere; a transient
      `_parent` `Affine` (set by an enclosing `Room`) folds a room frame in (`M_eff = M·Pᵀ`,
      `p_eff = P·p_local + T`, box → conservative world AABB). `placement=(0,0,0)` is byte-identical to the
      un-placed emission. (`iso.py`; tests in `test_iso.py`.)
- [x] **`Room`/`Group` `Element`** — `loom.Room` owns a child list + an animatable rigid `Affine` frame;
      on emit it hands each child the composed frame as its `_parent` (`room_frame ∘ child_placement`),
      namespaces child names (`hall/gyroidA`, stacking for nested rooms), and restores child state after.
      (Shared materials/lights + shell are authored in the driver, not the Room, keeping it a pure group.)
- [x] **Driver pattern / factory** — `examples/room_of_gyroids.py` exposes `make_gyroid(**params) ->
      Isosurface` and a driver that instances four different minimal surfaces on a *closed* circular orbit
      inside a slowly-tumbling `Room`. (Left the 4124-line CLI `gyroid_nd.py` untouched — it predates the
      Room API and isn't a clean factory; the new focused example is the canonical J2 driver.)
- **Caveats handled:** seamless loop verified (frame48==frame0 with closed-curve orbits + integer-turn room
  rotation + 2π drift). Overlap kept cheap via separate sphere containers; CSG union/blend remains an
  ftrace-side question (not needed here). *(Not yet stress-tested: many (dozens+) overlapping sphere-traced
  containers in one frame — validate ftrace throughput if a driver ever instances that many.)*

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
         **DONE IN FTRACE TOO — 2026-07-27 (v0.86.0).** Inline colour is no longer loom-only: `RecChannel` gained a
         `space` field, and `Parser::parseChannelStops` strips a leading colour-head tag off a channel line and
         feeds `{space, comps…}` to the **same `evalSpectrum`** a top-level `spectrum "x" = rgb …` decl uses. That
         convergence is the whole design: the record path inherits all 18 colour heads (3 spaces × `""`/`line`/
         `illum`/`smits`/`box`/`meng`) for free, and the Jakob–Hanika coefficient bake + GPU upload never learn
         that records exist. `isColourHead()` is the single list of accepted tags; loom's `_COLOUR_SPACES` mirrors
         it exactly, and `Record.lower_colours` now passes a non-plain head through **verbatim** (deduping on
         `(head, comps)`) instead of mis-converting it as if it were HSL. Lowering is still useful — it dedupes
         colours and targets a pre-0.86 binary — but is no longer *required*.
         Only *untagged* vector channels (`D∉{1,3}`) stay loom-only, and deliberately: ftrace has no destination
         slot to feed an arity-`D` value into, so it emits an error naming the colour tag as the fix.
      2. **Generalized stop grammar** (`ROADMAP_records.md` §3.1) — arbitrary-arity stops with a **delimiter
         precedence ladder** (whitespace binds like `×`, comma like `+`, brackets = parens), so structure is
         recoverable from the delimiters alone and the channel's arity only *validates*: `tint [rgb 0 0 0,
         0 1 0, 1 1 1]` ≡ `tint rgb [0 0 0] [0 1 0] [1 1 1]` (the three ladder delimiters are `[ ]` / `,` /
         whitespace — parens are reserved for expressions + the §3.2 application surface); position pins
         (`.2:0 0 0`) are an orthogonal `POS:` prefix. loom implements this as **one backward-compatible grammar**
         (`loom/record.py`, a single `parse`/`emit` pair): a line with no ladder delimiter keeps the exact J3a
         whitespace meaning and only a `,`/`[`/`]` opts into the ladder (`tint 0 0 0, 1 1 1` = two arity-3 stops;
         a lone vector stop takes a trailing comma). It's an *additive superset*, not a breaking change — no
         existing record reparses differently.
         **DONE IN FTRACE TOO — 2026-07-27 (v0.86.0).** The old "current FTSL cannot parse this — its tokenizer
         isn't comma-aware" note was the blocker, and it turned out to be **the reason the ladder was easy, not
         the reason it was hard**: because `,` is *not* one of ftrace's tokenizer delimiters, a comma survives
         lexing glued to its word (`0,`), so it can simply be re-split in the loader, paren-aware. (The error
         `bad stop expression '0,'` came from the *expression compiler*, not the parser.) Only `[` / `]` genuinely
         delimit, so the grammar change is one rule — `stop_group = '[' stop_item* ']'` in `ftsl_scene.epeg`, with
         `ftsl_reduce.hpp` flattening it back to `[`/`]` marker words. Everything else lives in
         **`src/record_ladder.h`** (`recladder::tokenize`/`usesLadder`/`parse`, a sum→product→factor
         recursive descent), the dependency-free C++ twin of `loom/ladder.py`.
         Additivity is *structural*, not hopeful: `parseChannelStops` routes a line with no tag and no ladder
         delimiter through the byte-identical old whitespace loop, and both ways the new path can fire were
         previously hard errors (a leading `rgb` was never a legal pattern variable; a `,`/`[`/`]` never lexed).
         `clamp(u,0,1) 0.5` still takes the fast path — its comma is at paren depth > 0.
         **Verified:** `-parseonly` (new flag) over the whole `scenes/` corpus, 81 scenes, 0 failures; 14 accept +
         5 reject cases; `scenes/_record_ladder.ftsl` writes the same channel four ways; and
         `tools/check_record_twins.py` probes ftrace's per-channel stop count via an out-of-range `rec.ch[999]`
         selector and diffs it against loom's — 27/27 channels agree across all six `_record_*.ftsl` fixtures.
         That checker exists because a loom↔ftrace stop-boundary disagreement is a *silent wrong render*, not a
         parse error; it already caught one real divergence (loom ended a record body at the first `]`, which the
         ladder made ambiguous — now bracket-matched).
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
         **DONE — parts (a),(c),(d) 2026-07-26.** (a) `loom/spatial.py`: the coordinate leaf `_Coord` is
         generalised to a public **`Surface`** family — `X Y Z` (axes, both backends), `U V` (surface params,
         ftrace `u`/`v`, **emit-only**), `A` (albedo placeholder — no ftrace var, so a bare `A` raises until
         substituted/defaulted); `SpatialExpr.free_inputs()` + `.substitute({name: expr})` (functional rewrite).
         (c) binding-by-substitution: `gold(u=v)` swaps the `U` leaf at emit — loom always resolves to a concrete
         field in real ftrace variables, **never** literal bundle syntax (so the J3c SEQUENCING concern is
         sidestepped: **zero** ftrace changes, renderable at every commit). (d) `loom/scene.py`:
         **materials-as-bundles** — a `Material` property may be a `SpatialExpr` field; `Material.free_inputs()`
         is the union; `mat(u=v, a=1)` / positional `mat(expr)` applies by substituting across the bundle; unbound
         `A`→`albedo_default`; `Scene.add` **expands** each field into a renderable companion (colour slot →
         `ProcTexture` over `u/v`; scalar slot → `FuncPattern` over `x/y/z`), with coordinate-family validation.
         Validated: a bundle scene emits `.ftsl` ftrace parses & renders identically to the hand-authored
         `func_skin` path. Tests: `test_spatial.py` (+9), `test_material_bundle.py` (14). **Remaining:** item 4
         (N-D input domain).
         **DONE — part (b) 2026-07-26 (v0.57.0).** The scoping below was right that it needed a renderer change, so
         the op was built first and the loom leaf second. **ftrace:** new `PatOp::Tex`, appended at the END of the
         enum so `patternHasFreeVars`'s `VarX..VarV` range is unperturbed, spelled **`tex:<name>(u, v)`** — the
         tokenizer scans `tex:foo` as one identifier and `funcOp` reports arity 2. The texture index rides in the
         existing `PatNode::a` double (the same trick `PatOp::PovFn` uses for its internal id), so the struct layout
         and the verbatim device upload are unchanged. `pattern.h` stays free of `texture.h` (which drags
         `upsample.h`/`spectrum.h`/`color.h` into every TU, nvcc's included): sampling goes through an opaque
         `PatCtx::texFn`/`texSelf` hook installed by `scene.h`'s `bindPatTex`, and `patCtxFromHit` now takes the
         `Scene`. Compile-time name resolution is **opt-in per value site** via a new `PatTexScope`, handed to
         `compilePatternExpr` only where a shading context exists (pattern blocks, record scalar stops / drivers /
         overrides, procedural texture channels) — so `tex:` in an isosurface `function { expr }`, a medium
         `density`/`ior` program or a load-time constant is a **specific compile error, never a silent 0.0**
         (9 error paths verified). GPU twin in `dPatternEval` over the existing `dTexScalarAt`, with the two new
         params threaded through all 8 call sites (no default args — it's forward-declared twice). Ordering: Pass 1b
         builds textures before patterns/records/materials, so file order is irrelevant for those; the one rule is
         that a *procedural* `texture { rgb "…" }` bakes during 1b and can only sample images declared above it.
         **loom:** `Image(path, u=…, v=…, encoding=…, filter=…, wrap=…)` in `spatial.py` — coordinates are ordinary
         sub-expressions (warpable, and `substitute` reaches into them so a bundle's `u=`/`v=` binding flows in);
         `_auto_name()` gives a deterministic `img_<stem>_<hash8>` over path+sampler so identical images share one
         declaration and different samplers don't collide; `SpatialExpr.image_textures()` + `Scene._add_image_textures`
         auto-declare the companion `texture` block (an explicit one of the same name wins in either order), without
         which every image term would emit a dangling reference; `eval_np` is a faithful port of
         `Texture::sampleRgb`/`scalarAt` (half-texel offset, v-flip, repeat/clamp/mirror, mean-of-linear-RGB).
         `encoding` defaults to `linear` — a value used as a NUMBER wants the stored levels.
         **Also fixed en route:** `Material.expand`'s scalar-slot guard rejected `{u,v}`, which was simply wrong
         about ftrace — a scalar slot is a *live* pattern evaluated through `patCtxFromHit`, which supplies u/v
         (`scenes/uv_native.ftsl` ships `weight_map pattern:uvcheck8` over `floor(u*8)`). Only the *colour* slot is
         coordinate-restricted, and for the opposite reason (it bakes into a u/v-indexed image).
         **Validated:** the loom-emitted `.ftsl` renders **bit-for-bit identical** to hand-authored
         `scraps/texop_pattern.ftsl`, which in turn is bit-for-bit identical to the shipped `texture:<name>` slot
         binding on both CPU and GPU; the composed case (`scraps/loom_image_mixed.ftsl` / `texop_mixed.ftsl`) matches
         CPU↔GPU to MC noise. Tests: `tools/loom/tests/test_image_term.py` (25), suite 1046 → 1071.
         Docs: README.md, FTSL.md §6.1 + isosurface section, loom DESIGN.md/README.md.
         **[superseded scoping, kept for the record] PART (b) IS NOT ZERO-COST — needs an ftrace pattern-VM texture-sample op (2026-07-26).** Unlike
         a/c/d, an `Image` leaf sampled *inside* a pattern/isosurface expression has **no** shipped ftrace target:
         `src/pattern.h` `funcOp` has no texture/image builtin (only abs/sqrt/sin/…/noise), so emitting a `tex(...)`
         call would be un-renderable — violating the "renderable at every commit" rule. The proper fix is a new
         `PatOp::Tex` (a texture-sample-at-(u,v) op) threaded through: pattern compile (resolve `tex:<name>` →
         texture index, needs the texture-index map in pattern scope), `PatNode` carrying the index, CPU eval
         (reuse `Texture::scalarAt`), and the GPU pattern interpreter in `render_cuda.cu` (`dPatternEval`, which
         already handles `VarU`/`VarV`; reuse the existing device sampler `dScalarAt`, render_cuda.cu:3426; upload
         the per-pattern texture-index table). Bounded but cross-cutting + GPU + **VERSION bump** — a distinct
         workstream from the pure-loom a/c/d, to be done as its own focused pass (in lockstep, it also wants the
         numpy twin: PIL load + bilinear sample where coordinates permit).
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
         **DONE IN FTRACE TOO — §3.3 materials-as-bundles arm, 2026-07-27 (v0.87.0).** Binding is **substitution**,
         and on a *postfix* program substitution is a pure **splice**: a variable node pushes exactly one value and
         so does a well-formed program, so `patternSubstitute` (in `src/pattern.h`, alongside `varName` /
         `patternUsesVar` / `patternCollectVars`) just replaces each `VarU` node with the replacement program's
         nodes. No environment, no closure, no runtime indirection — an applied material is an **ordinary**
         `Material` over ordinary patterns, so the GPU upload, the CPU evaluator, `patternHasFreeVars` and the
         record samplers are all untouched. Substitution is **simultaneous** (`gold(u=v,v=u)` swaps).
         `PatOp::VarA` is appended at the END of the enum (the `PatOp::Tex` precedent) so the `VarX..VarV`
         intrinsic range is unperturbed; `a` is the one named input with **no per-hit intrinsic**, so an unbound
         `a` must resolve at LOAD time — to what the use site binds, else to the material's `albedo_default`
         (system default 1.0). That default lives in the loader (`Parser::albedoDefault_`), *not* on `Material`,
         because `Material` is uploaded verbatim to the device. `a` is also **scope-gated**: `compilePatternExpr`
         takes a new trailing `allowA`, true at exactly the four material-reachable value sites (pattern block,
         record material driver, record `from` driver, record-override slot RHS) and false everywhere else, so
         `a` in an isosurface or a medium program is a specific compile error, never a silent 0.0.
         Application is **lazy and memoised**: *every* by-name material reference routes through
         `Parser::lookupMaterial` → `applyMaterial(idx, argText)`, which returns the ORIGINAL index for a no-op
         call and caches on `"<idx>(<args>)"`, so identical applications share one clone and a material nobody
         applies is bit-identical to before (additivity is structural, not hopeful — 83 scenes parse, 0 failures).
         `parseBindArgs` finds argument boundaries from the `=` signs plus top-level commas; that is unambiguous
         only because the pattern language has **no comparison operators**, so a top-level `=` is always a binding.
         Named args are resolved *before* positional ones and a positional binds the sole **still-free** input
         (loom `Material.apply` parity: `gold(v,a=1)` legal, `gold(v)` on a two-input bundle rejected, >1
         positional rejected). Routing plain lookups through `applyMaterial` can append to `Scene::mats`, so
         `resolveMixChildren` was refactored to take a material **index** rather than a `Material&` — a stored
         reference would have dangled on reallocation.
         **Spaces in an argument list** were initially not allowed and that was fixed in the same arm (see the
         separate v0.88.0 entry below): the shared grammar's `WORD`/`PARENWORD` now match a *balanced paren group*
         as part of the token, so `gold(0.5*u + 0.5*v)` and `gold(a=1, u=v)` lex as one word.
         **Pinned deterministically** by the new `-checkbind` self-test (`checkBind()` in `main.cpp`): splice ==
         textual inlining over 10 input types, simultaneity with negative assertions against *both* sequential
         collapse directions, identity for empty/absent binds, introspection, and `a` scope-gating.
         Regression scene `scenes/_material_bind.ftsl` — a 4×3 grid of camera-facing **quad tiles**, not spheres:
         `addQuad` gives a quad UVs spanning the parallelogram, so the material's `u` IS the tile's horizontal axis
         and `v` its vertical one, and rebinding `u=v` visibly ROTATES the gradient. (The first cut used spheres and
         was useless — curvature mixes albedo with shading falloff, so flat brightness couldn't be compared across
         surfaces at all, and a rotated gradient on a sphere is unreadable. The room is neutral grey for the same
         reason: colour bleed would contaminate the greyscale albedo being read.) It carries three **identity pairs**
         that are the actual asserts — `flat_a(a=0.5)` ≡ literal `reflect 0.5`, named `u=v` ≡ positional `(v)`, and
         comma ≡ comma-less argument lists. Measured at 120 s / 320²: normalising each tile against the backdrop
         strip below it (also albedo 0.5, which cancels the lighting profile) the pairs agree to 0.3% / 0.1% / 2.4%,
         `A1 0.610 < A2 1.103 < A3 1.522` tracks `0.15 < 0.5 < 1.0`, and the gradients come out horizontal (du=+39,
         dv=0) / vertical (du=0, dv=+37) / reversed (du=−38) / 45° diagonal (du=+15, dv=+14) as authored.
         Docs: FTSL.md §7.6, README.md, design.md.
         **Remaining in ftrace:** none — see the §3.2 follow-up below.

         **FOLLOW-UP — §3.2 per-property access, 2026-07-27 (v0.89.0).** `MATERIAL.slot` and
         `MATERIAL.slot(args)` now read ONE property off an already-declared material:
         `reflect src.reflect`, `reflect src.reflect(u=v)`, `roughness steel.roughness`, `ior glass.ior`.
         **The slot keyword IS the dot-handle.** §3.2 writes a property as `<slot keyword> ["name"] = <value>`
         and makes the quoted name *optional* — the slot keyword alone binds the property to its slot, and the
         name exists only to mint an external handle. ftrace properties are spelled with the slot keyword and
         have never carried a quoted name, so **the optional-property-name arm is vacuously already ftrace's
         status quo**: every ftrace property is anonymous, and the handle is therefore the slot keyword. That is
         why there is no separate work item for it.
         **Resolution routes through `applyMaterial`** (`materialPropRef` in `src/ftsl.h`), so a property
         reference *cannot diverge* from "apply the bundle, then read the slot" — same binding rules, same memo
         key, same `a` fallback, by construction rather than by a parallel implementation. In particular an
         unbound `a` resolves against the **SOURCE** material's `albedo_default`, not the reader's and not the
         system 1.0.
         **Four chokepoints** are hooked, because a value site in ftrace is reached four different ways:
         `evalSpectrum` (pattern-less spectral), `patternedSpectrumParam`, `dblParam` (pattern-less scalar), and
         `bindScalarPattern`. `patternedSpectrumParam` is **the only site that can carry both halves** of a
         spectral slot (base spectrum *and* companion per-hit pattern), which is why the pattern-aware path lives
         there and the other three refuse a pattern-carrying source with an explicit message rather than silently
         dropping the pattern.
         **Composition, not clobbering:** when the reader also writes its own `reflect_map`, `composePatterns`
         appends `[a…, b…, Mul]` in postfix so the two multiply. Both spellings mean "a per-hit multiplier on
         whatever the slot otherwise evaluates to", and letting one win would have made the result depend on
         statement order.
         **Three loud refusals** (the lesson from the item-3 review: a new limitation must fail, never
         approximate): a **record-driven** slot has no load-time value; a **texture-bound** slot cannot be carried
         by a property reference; and a **pattern-carrying** source at a site that cannot hold a pattern is
         rejected with "write it on the matching `_map` slot instead".
         **Records win a name clash** (`recordIndex_` is consulted first), so `R.chan` cannot change meaning in an
         existing scene just because some material is named `R` — necessary because `bindScalarPattern` reaches
         materials before records.
         **`loadedRef_`** is a pointer to the owning `Loaded`, never to an element: `applyMaterial` may append to
         `Scene::mats` and `Scene::patterns`, and both vectors reallocate.
         **loom twin:** `Material.prop(name, *args, **binds)` in `tools/loom/loom/scene.py`, keeping the
         `apply`/`free_inputs`/`albedo_default`/`prop` ↔ `applyMaterial`/`materialFreeInputs`/`albedoDefaultFor`/
         `materialPropRef` correspondence intact; 5 new tests in `tools/loom/tests/test_material_bundle.py`.
         **Pinned deterministically** by the new `-checkprop` self-test (`checkProp()` in `main.cpp`): ten groups
         comparing two independently-authored loaded scenes field-by-field — bare ref reproduces a pattern-driven
         and a constant slot, `u=v` == the hand-written twin, unbound `a` → source `albedo_default` 0.4, `a=1`
         overrides, composition with `reflect_map`, cross-slot `transmit`→`reflect`, scalar read-back, memo
         sharing — plus six `mustReject` cases and the record-name-clash case.
         Regression scene `scenes/_material_prop.ftsl` (flat camera-facing quads, same reasoning as
         `_material_bind.ftsl`). The **scalar arm is deliberately not in the image**: a glossy/thin-film tile
         under those two grazing strip lights renders essentially black, so a tile pair would assert nothing —
         `-checkprop`'s field-by-field comparison is the stronger assert. Row E alternates the two spellings
         `E1 E2 E1 E2` on purpose: an identity reads as a smooth left-to-right profile (just the lighting) and a
         difference reads as a zigzag, which needs no normalisation to see.
         Docs: FTSL.md §7.7, README.md, design.md.

         **FOLLOW-UP — spaces inside a paren group, 2026-07-27 (v0.88.0).** The item-3 arm shipped with an
         argument list that had to be a single *unspaced* token (`gold(0.5*u+0.5*v)` yes, `gold(0.5*u + 0.5*v)`
         no), because a value ends at the first plain bareword (`cont = NUMWORD | KVWORD | STRING`). That was
         defensible only as "pre-existing" — the shipped `RECORD(driver)` and array-literal sample-call forms had
         it too — and the user rightly called it flimsy, so it is gone. The fix is purely **lexical** and lives in
         the one shared grammar, `tools/loom/loom/grammar/ftsl_scene.epeg`: `WORD` and `PARENWORD` now match a
         **balanced paren group** as part of a token, with the group's interior character class allowing `' '` and
         `'\t'` while `WORD`'s own class still does not. So a space is a delimiter exactly while no parens are
         open. Fixing it in the lexer rather than in `matFieldId` was the only real option — once the tokens are
         split, the rest of the line has already been consumed as *other statements* and cannot be recovered.
         Three deliberate design points. (a) **Balanced, not greedy.** A `\([^)]*\)` that may cross spaces would
         happily swallow `(a) roughness 0.2` up to a `)` in a *different statement* — a silently wrong parse
         instead of an error. Regexes cannot count, so the nesting is written out to depth 4; that is not a real
         ceiling, because `WORD`'s fallback alternative still matches a bare paren as an ordinary char, so deeper
         nesting simply degrades to the old behaviour (outer parens consumed as chars, the group covering the
         innermost spaced run). (b) **Group alternative FIRST.** ECMAScript/Python regexes are leftmost-*first*,
         not leftmost-longest, so the engine must be told to prefer "consume the whole balanced group" over
         "consume `(` as an ordinary char and stop at the next space". (c) **Backward-compatible by construction:**
         the fallback alternative is byte-identical to the historical `WORD` class, so the group alternative can
         only ever *extend* a token across a space that sits inside parens — precisely the case that used to raise
         "missing ')'". Verified by replaying the old and new rule tables over the whole scene corpus with a
         faithful reimplementation of the longest-match/declaration-order lexer: **83 scenes, 26 805 tokens,
         exactly one differing token** — the intended one in `_material_bind.ftsl`. Regenerating
         `src/gpda/ftsl_scene.gen.cpp` touched **4 lines in 2 lexer rules and left the parse graph untouched**,
         which is the structural proof that the change is lexical only. Adversarial backtracking checked (40
         unclosed parens, 40 nested, an 840-char flat expression): max 33 µs, no blow-up. `parseBindArgs` needed no
         change, and `parenHint()` was rewritten — reaching "missing ')'" now genuinely means unbalanced parens.
         `_material_bind.ftsl` C4 is now deliberately spelled `grad_ua(u=0.5*u + 0.5*v)` so the scene regresses
         this, and still measures as the authored 45° diagonal. All 13 `-check*` self-tests pass; 83/83 scenes
         parse. Docs: FTSL.md §7.6 and the array-literal sample-call note (which carried the same stale warning).
      4. **N-D *input* domain** (several named driver *axes*, not one `range` scalar).
         **NOT SCHEDULED (user, 2026-07-25)** — items 1/2/3 are all done, so this is the only thing keeping J3b
         open, and it is deliberately left out. It is also the piece most entangled with the axis-labelled-array
         work above (§"ADDENDUM — axis-labelled arrays"): an N-D *record* input domain is the same "several named
         driver axes" idea the sample call already introduces on the value side, so doing it before that lands
         would build a second, competing spelling.
      Each emits down to the J3a form or a documented construct (e.g. lower a `D=3` channel to `spectrum:`-refs +
      synthesised `spectrum` decls); non-lowerable forms stay loom-only representation.
- [x] **J3c — full-scene `.ftsl` parser + emitter reconciliation. DONE (emitter-drift audit 2026-07-26 / read
      direction 2026-07-28, 0.93.0 — see the two DONE blocks below).** Add `.ftsl -> loom Element tree` to
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
      **EMITTER-DRIFT AUDIT — DONE (2026-07-26, 0.77.0).** The "audit every `Element.emit` against the live
      grammar" half is closed, but *not* by inspection: it was unachievable that way, because ftrace silently
      ignored unknown keys, so a drifted property renders a wrong image rather than raising. The prerequisite
      was a **diagnostic**, so that got built first — `Stmt::used` marked inside `find()` (the one choke point
      every property read funnels through), `collectUnusedKeys` at the end of `Builder::build`, reported from
      `loadSource` as an `[ftsl] warning`. See design.md's `ftsl.h` entry and FTSL.md §1.3. With that in place
      the audit is mechanical: `scraps/emit_audit.py` builds one scene per Element kind (11: sphere, beads,
      group, sweptmesh, isosurface, funcpattern, texture, proctexture, mixmaterial, volume, cameracurve) with
      its drift-prone optional fields exercised, emits it, loads it under `-zzz-stop`, and fails on any
      unknown-key warning. **Result: all 11 clean**, all 78 checked-in `scenes/*.ftsl` clean, 1083 loom tests
      green. Real drift found and fixed: loom's `Isosurface` could not emit `samples` / `accuracy` / `refine` /
      `uv`, so a sampled march was stuck on ftrace's 256-sample default (`tools/loom/loom/iso.py`; `uv` is
      validated against the bareword-axis trap rather than passed raw). Real corpus bugs found and fixed:
      `priority` authored on geometry in `scenes/_record_scalar.ftsl` (it is a **material** slot), and 6 dead
      `contained_by` lines on pure-analytic isosurfaces in the two gallery scenes (the loader only reads it for
      `function` fields — analytic CSG bounds itself, and a manual clip is spelled `intersect { box … }`).
      **READ DIRECTION — DONE (2026-07-28, 0.93.0).** `.ftsl` → loom Element tree landed, but *not* as
      emit's inverse, because it cannot be: loom's emitters **bake** Signals at a clock, so a `.ftsl` file is a
      static snapshot and an `Isosurface`'s field/freq/rotation/drift/placement/threshold are all flattened into
      one `function { expr "…" }` string. The property that IS achievable, and the one now proven, is
      **round-trip fidelity**: `parse_document(src).emit(ctx)` reproduces the source **byte for byte** — line layout,
      alignment padding, brace columns, 2-vs-3-space gaps, comments and blank lines included — which is exactly
      what an editor needs (load a scene, change one block, write it back, leave every other line untouched).
      Faithful kinds still build their real class (`material` incl. `type mix` → `MixMaterial`, `texture`,
      `proctexture`, `sphere`, `light`, `camera`, `spectrum "n" = …`, `range` records); the **baked** kinds fall
      back to a new layout-preserving generic element `loom/block.py` (`Block`/`Stmt`: ordered entries so
      duplicate keys survive — `camera_curve`'s repeated `point`, `mix`'s repeated `layer` — valueless keywords,
      nested blocks, `get`/`has`/`stmts`/`find`/`set`/`add`/`remove`/`same_as`, `roots() == []`).
      *(Mechanism: whitespace and comments are lexer `@skip`s and simply are not in the tree, so **every**
      formatting decision is recovered from source spans — `ParseNode.line/col` against a line-offset table —
      into per-entry `gap`/`own_line`, block `indent`/`brace_gap`/`pad`, a verbatim `raw` slice per statement
      dropped on `set()`, and `before`/`trail`/`tail_before` trivia lines.)*
      `parse_elements(text)` reads a whole file in order via a `start_rule = "elements"` override, so
      `parse_element` keeps rejecting text holding more than one element; `parse_document(text)` returns a
      **`Document`** = those elements *plus the literal text between them*, because a file is also the blank
      lines that group its elements and the top-level comments that head its sections, and those belong to
      *neither* neighbouring element (`Document.gaps` is one longer than `.elements`; `insert`/`append`/`pop`
      keep the file's head and tail in place). Grammar work en route: `KVWORD`
      relaxed toward ftrace's own spelling (`axis=y` / `wrap=clamp` are as real as `center=560`; parens excluded
      because loom's value grammar spells the sample call explicitly), the value grammar's `arg`/`kwarg`/`kwrest`
      rewritten so `a=u` inside a tuple still splits, `block` split into `block = nl? bcore nl?` so a nested
      block does not eat the separating newline, and `bhead` kept as ftrace's two distinct header shapes
      (`binder NAME subtype?` vs `NAME STRING? subtype?` — a bound element takes no quoted name).
      **Two real drift bugs found by the corpus round-trip and fixed** (see `known-issues.md`): the `type mix`
      dict-fold that dropped all but the last layer, and `as_color_binding` rejecting `pattern:<name>` in a
      colour slot that **loom itself emits** and ftrace accepts (`patternedSpectrumParam`, `src/ftsl.h` ~2083).
      Verification: 1243 loom tests green (26 new in `tests/test_grammar_block.py`); all **11** loom-emitted
      element kinds (`scraps/emit_audit/*.ftsl`) round-trip byte-identically; **65 of 97** corpus files parse and
      **64 of those 65** re-emit byte-identically as a *whole file*. The 65th is the one documented normalisation:
      a typed element re-emits through its own emitter, i.e. in loom's **canonical** form, so hand-alignment
      *inside* one is not preserved (`spectrum "steel"   = rgb …` loses the padding before `=`; it is a fixed
      point after the first save). Layout fidelity is a `Block` property, by construction.
      **SCOPE BOUNDARY (deliberate, do not "fix"):** the 32 non-parsing files are hand-authored *full-ftrace-
      language* constructs `ftsl.epeg` does not model — `/`-containing names (`hall/g0 = isosurface {`), §3.2
      per-property access (`arr_a.reflect(a=u)`), expression arguments (`grad_u(1-u)`), `[…]` array literals at
      block value sites. `ftsl.epeg` is loom's **typed** grammar and keeps real `NUMBER`/`REF`/`PIN`/`STRING`
      terminals so the record/material/spectrum validators can shape-check; adopting ftrace's catch-all `WORD`
      tokenizer would erase exactly those. The whole-language surface's home is `ftsl_scene.epeg` (the grammar
      compiled into ftrace). The whole-file `scene { … }` → live `Scene` builder likewise stays FUTURE — see the
      next bullet.
- [ ] **FUTURE — loom full `.ftsl` read support** (deferred out of J3c above). Give loom a complete `.ftsl` → `Scene`
      reader (not just per-element round-trip): the whole-file `scene { … }` wrapper rule + a `Scene` builder that
      reassembles textures/patterns/records/materials/geometry/lights/camera into a live `Scene`, plus the lossy
      cases (`mesh { file … }` → a new lightweight `MeshRef` element that re-emits the same block; `medium`, `pattern`,
      `camera_curve`). **Motivating consumer: an editor/GUI** (load an existing `.ftsl`, manipulate in loom's object
      model, re-emit) and possibly the raster preview loading authored scenes. Not on any current critical path — the
      grammar's real job is ftrace's parser — so this waits until a concrete editor need exists.
- [x] **PROPOSAL — unify element headers to `name = KIND { … }` — DONE 2026-07-19 (v0.9.1).** Superseded by the
      actionable **NEXT UP** entry at the top of this file, which shipped exactly this; kept for the rationale.
      Today elements
      spell their name inconsistently: records already use `NAME = range LO-HI [ … ]` (a `name = kind …` binding),
      but materials/textures/cameras use `KIND "name" { … }`. The cleaner, more programmatic form (per user, 2026-07-19)
      is to make *every* named element a binding — `hero = camera { … }`, `gold = material { … }`, `hide = texture
      { … }` — with the **anonymous** variety just dropping the `name =` (`camera { … }`, a nameless light, etc.).
      This unifies the whole scene grammar under one `binding = (NAME '=')? KIND block` shape (records fold in as the
      `range` kind), reads like assignment, and makes anonymity natural. Touches both loom's emitters (§ scene.py
      `emit`) and the shared grammar in lockstep; do it **before** the grammar ossifies into ftrace's C++ front-end so
      both sides adopt the new header at once. Decide alongside item 3 / the ftrace port.
- [x] **SHIP — bundle GPDA with the ftrace release — DONE 2026-07-26 (verified, no code change needed).** The
      GraphParser (GPDA) is ftrace's scene front-end shim, so the shipped product depends on it. Verified fully
      vendored and self-contained:
      * All 7 files in `src/gpda/` are git-tracked (`ftsl_reduce.hpp`, `ftsl_scene.gen.cpp`, `ftsl_shim.hpp`,
        `gpda_lexer.hpp`, `pool.hpp`, `tokenized.cpp`, `tokenized.hpp`) — the parser engine (`tokenized.*`,
        `pool.hpp`) copied verbatim from `GraphParser/cpp`, the grammar **pre-compiled to C++** by
        `loom.grammar.emit_cpp` into `ftsl_scene.gen.cpp`. Nothing is loaded from a `.epeg` at runtime, so the
        release artifact needs no data files.
      * `CMakeLists.txt` compiles the two vendored `.cpp` in both targets (lines 32–33 and 55) and adds
        `src/gpda` to the include dirs (line 48). Every `src/**.cpp|.cu|.c` named in `CMakeLists.txt` is
        git-tracked (checked mechanically against `git ls-files`) ⇒ a clean checkout has all sources.
      * The only include reaching outside `src/gpda` is `ftrace_parse_slice.hpp` in `ftsl_reduce.hpp`, and it is
        guarded by `#ifdef FTSL_SHIM_STANDALONE` — used solely by the throwaway `scraps/gpda_shim` de-risk
        harness, never by the in-tree build (in-tree, `ftsl.h` includes the header after the real `ftsl::Block`
        types are defined).
      * `build.bat`, `release.bat` and `CMakeLists.txt` contain **zero** references to
        `D:\visual studio projects\GraphParser`; `release.bat` ships one self-contained `ftrace.exe`.
      (Reminder logged 2026-07-19.)
- **Dependency note:** the FTSL record itself (§0) is fully implemented (Stages 1–6 + GPU parity DONE), so
  this is a loom-side mirror + parser effort, not blocked on ftrace.
- [x] **loom retime / 4D time-shear node — DONE 2026-07-27** (unblocked when J3b item 3 landed `t` as a first-class
      input, 2026-07-26). Shipped as designed below; what the build actually decided:
      * **`signals/retime.py`** (new) — `retimed_clock(clock, t, wrap)`, `Retime` (scalar), `VecRetime` (a vector
        retimed *as a whole*, so every component reads the same sample phase and shared sub-graphs are evaluated
        once per sample point; `.as_vec()` re-exposes it as a plain `VecSignal`), the `retime()` dispatcher, and
        the sugar `freeze(x, at)` / `delay(x, dt)` / `warp(x, g)`. Plus **`Phase`** in `signals/core.py` — the
        clock's own `t` *as a value*, which is what makes the family expressible at all (`delay` is literally
        `Retime(x, Phase() - dt)`).
      * **`wrap` defaults to "wrap iff `clock.loop`"** — that one default is what keeps `sig(t−dt)` seamless on a
        closed loop (it's the same loop, rotated) while leaving an open timeline honest off the end. Negative `dt`
        looks *ahead*, which is equally well-defined: the graph is a pure function of the clock, not a stream.
      * **(1) cache — took the "scope a nested cache" option**, not the "widen the key" one: `Cache.scope(key)`
        returns a nested `Cache` keyed `(node id, frame, sample phase)`, and every retime evaluates its child
        through it. Widening the global `(node_id, frame)` key would have touched every `at()` call site and put
        1150 passing tests at risk; `scope()` is purely additive and behaviour is byte-identical when no retime
        node exists. Off-current-`t` sampling is *not* restricted anywhere.
      * **(2) cycles — confirmed (2a) alone suffices and (2b) stays deferred.** Both retime edges (the retimed
        subtree **and** the phase driver) are ordinary structural edges reported by `children()`, so
        `detect_signal_cycle` keeps owning them unchanged. A retime is **not** a recurrence — it reads a pure
        function at another point, it does not read its own past — so no causality validator was needed or built.
      * **The headline 4-D shear is `spatial.SigAt`** (spatial tier, because a coordinate must be in scope):
        `SigAt(sig, when)` reads a modulator at a phase that is *itself a field*, e.g.
        `SigAt(Sine(cycles=3), T - X/c)`. This is the thing that could not be faked — a bare `Signal` used as a
        spatial term (`_Sig`) bakes **one number per frame** shared by the whole field. `emit()` deliberately
        raises (ftrace evaluates a pattern per hit with no access to loom's modulator DAG; baking one number would
        silently drop the shear), so it joins `VolumeField` as the second **single-backend** leaf in `spatial.py` —
        discretise-then-render via `mesh_field` / `bake_field` / `write_volume`. Cost is bounded and stated: one
        graph evaluation per *distinct* phase (`np.unique` + inverse), with `quantize=k` capping it at `k`.
      * Tests: **`tools/loom/tests/test_retime.py`** (27; suite 1150 → 1177). Mutation-checked three ways —
        degrading `Cache.scope` to the parent cache, dropping the driver edge from `children()`, and forcing
        `wrap=True` each fail a test. The cache tests specifically use a **sub-frame** retime/shear, where the
        retimed phase lands on the *same* `clock.frame`; a coarser delay maps to a different frame index and the
        keys never collide, so the first version of the test passed even with the bug reintroduced.
      * Docs: `tools/loom/design.md` § 4 + M10.5, `tools/loom/README.md` layout. Demo:
        `tools/loom/examples/time_shear.py` (tabulates the family; `--render2d` renders the travelling wave as a
        seamless loop — spatial frequency `k = w/c` falls out rather than being authored).
      * Still not built, on purpose: recurrent / stateful (delayed-feedback) nodes — see (2b) below.
  *Original entry (deferred; unlocked once `t` is a first-class input, J3b item 3).*
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

- [x] **K1 — Multiple RGB→spectral upsampling methods (incl. a user-supplied mapping).** **DONE 2026-07-28
      (v0.90.0)** — five built-in upsamplers *and* the user-supplied mapping; see the sub-entries below, the last
      of which closes the final clause. Today there is exactly
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
    - [x] **Smits 1999 upsampler landed** *(2026-07-24, v0.45.0).* The classic tabulated Smits RGB→reflectance
          basis is in `upsample.h` (`SmitsBasis` — seven basis spectra white/C M Y/R G B at 10 samples over
          [380,720] nm; `smitsCombine` does the additive white+secondary+primary decomposition, clamped to a
          physical `[0,1]` reflectance; `rgbToReflectanceSmits` linearly interpolates in λ). Surface: **head
          keywords** `rgbsmits`/`hsvsmits`/`hslsmits` (parallel to the `…illum`/`…line` heads), wired through
          `evalSpectrum` (`ftsl.h`) and mirrored in loom's spectrum grammar (`SmitsSpec` in
          `tools/loom/loom/grammar/spectrum.py`, with tests). Validated by `-checkupsample` (Smits round-trips
          sRGB to <0.07 max, all reflectances in [0,1]) and a render (`scraps/smits_test.ftsl`, Smits vs
          Jakob-Hanika panels side by side). A reflectance upsampler, so no GPU change (upsampling is a host-side
          bake into the spectral tables).
    - [x] **Plain calibrated 3-box upsampler landed** *(2026-07-24, v0.46.0).* The simplest reflectance
          upsampler: one flat step per band (blue 400–500, green 500–600, red 600–700 nm) whose three heights
          are solved from a fixed 3×3 matrix (`BoxBasis`/`boxBasis()` in `upsample.h` precompute `Minv`, the
          inverse of the per-band linear-sRGB response) so `rgbToReflectanceBox` integrates back to the
          requested colour *exactly* — round-trips sRGB to <0.02 max, the tightest of the reflectance
          upsamplers, though blocky (hard band edges). Surface: **head keywords** `rgbbox`/`hsvbox`/`hslbox`
          wired through `evalSpectrum` (`ftsl.h`) and mirrored in loom (`BoxSpec`, with tests). Validated by
          `-checkupsample` (box section, err <0.02, all reflectances clamped `[0,1]`). Reflectance upsampler →
          host-side bake, no GPU change.
    - [x] **Meng 2015 smoothest-spectrum upsampler landed** *(2026-07-27, v0.85.0).* Of all physical
          reflectances that produce a given colour, take the **smoothest** (minimum `Σ(s[i+1]−s[i])²`),
          tabulated over chromaticity and interpolated. Surface: **head keywords**
          `rgbmeng`/`hsvmeng`/`hslmeng` through `evalSpectrum` (`ftsl.h`), mirrored in loom (`MengSpec`,
          with tests). *This was previously logged as BLOCKED on the paper's published precomputed table
          ("external data we don't vendor") — that framing was simply wrong on both halves.* First, the
          supplemental carries **no licence and no copyright notice anywhere** (verified against the real
          137 MB zip), and this repo has no LICENSE of its own to reconcile it with — so there was never a
          licensing obstacle to begin with. Second, and more to the point, we don't *want* their table:
          the method is published in full, so we bake our own (`tools/bake_meng.py` → `src/meng_table.h`,
          order 16 / 153 vertices / 81 samples / ~138 KB) with two departures that make ours **more**
          accurate than the paper's grid for our use — a grid barycentric in the sRGB primary triangle
          (every colour we upsample already lies inside it, so no search or cell classification), and
          vertices solved at `Y=1` **unbounded above** rather than under an active `s ≤ 1` bound. The
          second is essential, not cosmetic: min-roughness is linear in target XYZ only over a *cone*, and
          `{0 ≤ s ≤ 1}` isn't one — tabulating against a live upper bound destroys the tabulated property,
          which is precisely the bug that first made `-checkupsample` call Meng *rougher* than JH.
          Validated by `-checkupsample` (section g): round-trip max error **0.00005** (white 0.01208,
          itself better than JH's 0.01647 — pure white is genuinely infeasible as a bounded reflectance
          under our blackbody-6504 "D65"), all samples in `[0,1]`, and roughness provably below JH's on
          every test colour. Render: `scraps/meng_test.ftsl` (four upsamplers, same colour, lit by
          illuminant A — a D65 wash cannot distinguish them by construction). Reflectance upsampler →
          host-side bake, no GPU change.
    - [x] **User-supplied named mapping landed** *(2026-07-28, v0.90.0)* — the last clause of the original
          proposal, and the item's close. Surface: `upsample "<name>" { expr "f(r, g, b, w)" }` declared at
          top level and named by the **colon head** `rgb:<name> r g b` (also `hsv:`/`hsl:`, converted to
          linear sRGB *before* the body runs so all three feed identical `r,g,b`). A colon rather than yet
          another glued suffix because the built-in suffixes are a closed set a reader can memorise while a
          user name is open-ended, and `:` is already this grammar's namespace marker (`spectrum:`, `metal:`,
          `tex:`, `grid:`). Wired through `evalSpectrum`'s existing colour-head chokepoint (`ftsl.h`), so the
          head works at *every* spectral site and, because `isColourHead` gained one shape-only arm
          (`isCustomColourHead`), at a record channel's inline-colour tag too — the one list stays one list.
          *This entry replaces a previous "Residual (deliberately not scheduled)" bullet that claimed the named
          user mapping was "already covered" by `spectrum "name" = <expr>` + the built-in heads. That was
          wrong: a named spectrum is a fixed curve, and a built-in head is a fixed function — neither lets a
          scene supply `(r,g,b) -> spectrum` itself, which is what the proposal asked for.*
          Three design decisions carried it, each chosen so a limitation fails loudly instead of approximating:
          **(a) the body's variable vocabulary is DISJOINT from the surface one, not additive**
          (`PatVarMode::Upsample`, `pattern.h`). `r` already means *radius* in a surface program and has to
          mean *red* here; an additive design would make one spelling silently mean two things depending on
          site. So in upsample mode every surface name (`x y z u v f nx ny nz r t`) is rejected **by name**
          with a message that says the spelling changed meaning, and `r`/`g`/`b`/`w` reuse the
          `VarX`/`VarY`/`VarZ`/`VarU` slots as a pure register assignment (invisible, since the surface
          spellings are unreachable in this mode). `pi` is deliberately left shared — it is a constant, not a
          context. **(b) `spec:<spectrum>(w)` (`PatOp::Spec` + `PatSpecScope`)** samples a declared `spectrum`
          block at the queried wavelength. This is what makes the feature more than syntax sugar: it makes a
          **measured basis** expressible (`r*spec:red(w) + g*spec:green(w) + b*spec:blue(w)`) rather than only
          closed-form arithmetic — and it is why the expression form was chosen over a basis-only design,
          which it strictly subsumes. Resolution is compile-time through a scope object exactly like
          `tex:`/`grid:`/`scatter:`, so `spec:` outside an upsample body and `tex:`/`grid:` inside one are
          both compile errors naming the scope rule (an ordinary pattern has a hit point but no wavelength;
          an upsample body the reverse). Spectra are memoised into an **append-only** vector so an index
          handed out at compile time survives later growth. `PatOp::Spec` never reaches the device: an
          upsample program is consumed at load time and is never stored on a `Material` or in
          `Scene::patterns`. **(c) the result is a live closure, not a baked table.** A user upsampler is free
          to be a narrow emission line, and pre-tabulating at the loader would quietly band-limit it; the
          renderer already tabulates where it must (`double reflect[SPEC_N]`), at a resolution it chooses. The
          closure captures the compiled program and the spectrum vector by `shared_ptr`, and the sampler
          thunk's `self` is the *vector* rather than the Builder, so the produced `Spectrum` outlives the
          loader (it ends up on a `Material` the `Scene` owns). Programs compile once per name and are shared
          by every colour that names them. Refusals are loud and specific: unknown upsampler, `upsample` with
          no `expr`, surface variable, unknown identifier (listing the vocabulary), unknown spectrum in
          `spec:`, an uncalled `spec:` reference, and both halves of the scope rule. Validated by
          `-checkupsample` **section (h)**: 14 asserts run through the real loader (constant body; r/g/b/w
          each reaching their own slot; `hsv:` ≡ the converted `rgb:`; `spec:` matching the gaussian's closed
          form at five wavelengths; a three-spectrum basis matching by hand; and the eight refusals) — every
          number computed in the test rather than hard-coded, and every sample taken *after* the `Loaded`
          scope exits, which is what pins (c). Visual companion `scenes/_upsample.ftsl` →
          `png/_upsample.png` (six spheres, one authored colour, six upsamplers, plus a record channel
          tagged `rgb:basis`). loom twins: `NamedSpectrum`/`Upsample` elements (`scene.py`, emitted ahead of
          textures/patterns/materials), `UserSpec` (`grammar/spectrum.py`) and `is_colour_space`
          (`record.py`, replacing the closed `_COLOUR_SPACES` membership test at both sites that asked) —
          `tools/loom/tests/test_upsample.py`, 13 tests. Host-side only, so no GPU change.
    - **Still deliberately out:** nothing. A "user-supplied custom *basis*" (bring your own basis curves)
          is now expressible as a `spec:`-weighted body, which is what that phrasing was reaching for.

- [x] **K2 — Analytic physical sky (`turbidity`).** **DONE 2026-07-24.** Implemented the **Preetham et al. 2002**
      analytic daylight model as an `env` sub-kind: `light env { sky preetham  turbidity t  sun_dir …  (or
      sun_elevation/sun_azimuth)  ground_albedo a  intensity s  res px  rotate d }`. `src/sky.h`
      (`generatePreethamSky`) evaluates the Perez five-parameter distribution for luminance Y and CIE xy (turbidity-
      dependent coefficients + zenith Yz/xz/yz from the solar elevation), converts xyY→XYZ→linear-sRGB per texel, and
      bakes an equirectangular sky image (row0=up, matching `EnvMap`'s convention). The **solar disk** is baked on top:
      a 5778 K blackbody attenuated by Rayleigh (∝λ⁻⁴) + Ångström-aerosol (β from turbidity) optical depth over the
      Kasten–Young air mass, integrated to XYZ and scaled to a physical clear-air disk luminance (~1.6e9 cd/m²·
      transmittance) — so a low sun **reddens into a proper orange sunset automatically**. Magnitudes are physical,
      then the whole image is normalised so the mean above-horizon sky luminance = `intensity`. Rather than a bespoke
      construct, the sky is fed through `EnvMap::buildFromRgb` (refactored out of `EnvMap::load`), so it **reuses the
      entire env pipeline**: luminance-importance sampling, per-texel Jakob–Hanika spectral upsampling, direct-view
      background, and the GPU `DEnvMap` upload — an analytic sky lights the scene exactly like an HDRI on **both CPU
      and GPU**. **Validated:** `scraps/sky_test.ftsl` (daytime — blue overhead 148/184/234, whitening toward the sun,
      cool sky-lit ground; CPU and GPU byte-agree on the sky) and `scraps/sky_sunset.ftsl` (low sun — visible solar
      disk on the horizon, strongly reddened orange sky with blue crushed by Rayleigh extinction). **Follow-up logged**
      (known-issues): the physical solar disk is ~10⁵× the sky, so sun-lit diffuse surfaces are HDR and converge slowly
      in forward modes (same as any sunny HDRI) — an efficient distant-directional-sun emitter (parallel forward
      emission + backward NEE) would fix this and is the proper enhancement.

- [x] **K2 follow-up — first-class distant directional sun (`EmitterShape::Sun`).** **DONE 2026-07-27 (0.84.0).**
      Built the enhancement K2 logged. New light subtype `light sun { elevation … azimuth … (or dir …) angle 0.53
      spd … intensity … }` registers an infinitely-distant disc emitter whose rays arrive **parallel**, plus a new
      **`sun_disk on | off | separate`** switch on the Preetham sky block: `separate` strips the baked disk out of
      the equirect map and registers an **energy-matched** `light sun` beside the skylight dome (the baked profile's
      `∫t dΩ = 0.8133·πR²` is converted to an equivalent hard cone `θ = acos(1 − Ω_eff/2π)`, and the disk spectrum —
      the air-mass-attenuated 5778 K blackbody — is scaled to the same luminance the map carried).
      **Radiometry:** the authored `spd` is *perpendicular spectral irradiance*, stored internally as radiance
      `E⊥/Ω`, so widening `angle` softens the penumbra without changing exposure. **Forward:** photons are born on a
      disc the size of the scene's cross-section, aimed down the beam (joint pdf `1/(Ω·πR²) = 1/envGeom`, exactly
      analog), so every photon enters the scene. **Backward:** cone NEE with `1/pdfW = Ω`, and the directly-viewed
      disc added on a ray miss under the `specularArrival` gate — unbiased with **no** MIS weight, because NEE runs
      at exactly the material types that then clear that flag. The hard cone reuses `spotCosInner == spotCosOuter`
      so `spotOmega` evaluates to `Ω` — **no new emitter field** on host or device. Touched: `scene.h` (`addSunLight`,
      `sunRadiance`/`sunXYZForDir`, `geomWeight`, `finalizeEmitters`), `sky.h` (`SunDisk` + extended generator),
      `ftsl.h` (subtype + `sun_disk`), `backward.h`, `photonmap_render.h`, `sppm_render.h`, `main.cpp`
      (`addEnvBackground`, mode-D/U guard), `raster.h` (preview shading), `bdpt.h`/`vcm.h` (reject), and the full
      **CUDA mirror** in `render_cuda.cu`. Modes `D`/`U` refuse a sun scene (not area-connectible), as for
      `spot`/`env`; everything else (A/B/C/R/`-rgb`/P/M/S) runs on **CPU and GPU**. **Validated** with
      `scenes/_sun_check.ftsl`: forward B vs backward R **0.09%**, CPU vs GPU mode R **0.01%**, `-rgb` **0.00%**,
      photon-map M vs R **0.02%**, `angle` 0.53°→8° exposure shift **0.019%**, `sun_disk separate` vs baked
      **0.12%**. The payoff, on `scraps/sky_test.ftsl` (mode B, GPU): **baked** reached only 7.2% noise after
      30 s / 2×10⁹ photons and *still* had no warm sunlight and no cast shadows, while **`sun_disk separate`**
      hit the 4% target in **5.5 s / 3.0×10⁸ photons** with the sun fully formed — ~**20× fewer photons** for
      the same noise, and a qualitatively correct picture instead of a skylight-only one. New deterministic
      self-test **`-checksun`** (0.84.1) pins the cone solid angle, `L·Ω == E⊥` exposure invariance,
      uniform-in-solid-angle cone sampling about both axes, and NEE/direct-view rim agreement — worst error
      5.3e-14, and it caught a real sign error on its first run. All 12 `-check*` self-tests pass.

  <details><summary>original K2 scope</summary>

- ~~**K2 — Analytic physical sky (`turbidity`).**~~ ftrace has **no** procedural sky: environment lighting is only
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

  </details>

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

- [x] **Hero-wavelength Monte-Carlo across every spectral render mode — DONE 2026-07-26 (through VERSION 0.70.0).**
      Every render *mode* now carries the bundle on both backends: CPU `A`/`B`/`C`, `R`, `M`/`S`, `D` (BDPT) and
      `U` (VCM/UPS), plus the GPU megakernel's forward `A`/`B`/`C`, the `M` deposit, backward `R`, BDPT `D` and
      VCM `U`. The only remaining single-λ code path is the GPU **wavefront scheduler** (`-wavefront`), which is
      not a mode — `-heroc > 1` simply forces the megakernel there — and is left single-λ **by design** (see the
      sub-item below; port the SoA pool only if the streaming backend ever needs the chroma win). Scenes with
      participating media, a GRIN volume, or a finite-lens camera also stay single-λ everywhere, also by design.
      Original scoping text follows.
      Today ftrace carries **one** wavelength per
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
          and the GPU BDPT megakernel (below). (The GPU *backward* megakernel landed 2026-07-26 — see the
          next entry.)
    - [x] **R (backward) — CPU DONE.** `radianceHero()` in `src/backward.h` samples a hero λ + 3 stratified
          secondaries (`hero.h`, `kHeroC=4`), rides them along one shared BVH walk, evaluates materials/NEE per-λ
          (`neeLightHero`/`neeEnvHero`, shared `interactMaterial`/`emitterGeom`/`envGeom` helpers) and splats 4
          CMF-weighted contributions (÷C). De-hero at any dispersive/wavelength-switching interface (dielectric,
          thin-film, multilayer, grating, filter, fluorescent) boosts the hero ×C — PBRT-v4 `TerminateSecondary`
          convention. Gated on: `kHeroC>1 && no fog/GRIN/lens` (those stay scalar, C=1 is bit-identical). Validated
          on `cornell.ftsl`: converged image unchanged, glass-sphere dispersion intact, **luma noise flat (1.03×),
          chroma noise down (0.89× overall, 0.74× in spectral-dominated neutral regions)** at equal spp.
    - [x] **R (backward) — GPU DONE 2026-07-26 (VERSION 0.59.0).** `bkRadianceHero()` in `src/render_cuda.cu` is
          the device twin of the CPU `radianceHero`, mirroring it 1:1 so the two can be diffed and neither can
          drift: `bkNeeLightHero`/`bkNeeEnvHero` ↔ `neeLightHero`/`neeEnvHero`, `bkInteract` ↔ `interactMaterial`,
          same de-hero material set, same gate. Four supporting refactors, each a pure code move that leaves the
          scalar path's fp32 rounding untouched (device `Real` is `float` by default — see `FTRACE_GPU_FP32`):
          (1) `dSampleSceneLambdaU(sc, u, pdf)` split out of `dSampleSceneLambda` so the bundle can push its own
          stratified `u` values (base draw + C−1 wrapped strata) through the same inverse-CDF sampler;
          (2) `bkEmitterGeom` → `BkNeeGeom` and (3) `bkEnvGeom` → `BkEnvGeom`, which return the geometric *pieces*
          (`wi`/`dist2`/`cosSurf`/`G`/`stG`/…) rather than a fused weight, so each caller re-forms the original
          float product verbatim; (4) the whole material switch hoisted verbatim out of `bkRadiance` into
          `bkInteract`, shared by the scalar path and by hero's de-hero fallthrough. `kBackward` takes a new
          `heroC` parameter and draws the bundle per sample; `renderBackwardCuda` gates on
          `heroC>1 && mediaN==0 && !hasGrin && !cam.hasLens()` (clamped to `hero::kHeroMax`) and is fed `g_heroC`
          from all four `main.cpp` call sites (progressive, chunked, and the two meter frames).
          **Validated** on `cornell.ftsl` (which carries a dispersive SF10 glass sphere, so the de-hero branch
          fires) at 300², GPU:
          • **`-heroc 1` is arithmetically bit-identical to the pre-change binary** — byte-for-byte equal at
            `-spp 1` (one `atomicAdd` per pixel, so accumulation order can't matter). At `-spp 64` exactly one
            channel of one pixel out of 270 000 differs by 1 LSB; that is `atomicAdd(double)` summation order
            shifting with the recompiled kernel's warp scheduling, not an arithmetic change (each binary is
            self-deterministic on rerun).
          • **Energy conserved:** converged (`-spp 131072`) `-heroc 4` vs `-heroc 1` agree to **0.03 %** in mean
            linear luminance, both auto-exposing to 1.04e-13 — so the ×C de-hero boost is exact.
          • **Chroma noise down, luma flat** at equal spp (64) against a 131 072-spp reference: rms chroma
            0.540 (C=1) → 0.477 (C=2) → **0.416 (C=4, −23 %)** → 0.404 (C=8); rms luma 0.829 → 0.803 (flat).
          • **Free:** 70.4 s (C=1) vs 68.9 s (C=4) for 131 072 spp — the bundle rides one BVH walk.
          • **Gate verified:** `_fog_cornell` (media) and `grin_lens` (GRIN) are byte-identical between
            `-heroc 1` and `-heroc 4`, i.e. they correctly fall back to single-λ.
          • **CPU ↔ GPU agree** to 0.29 % mean luminance at 2048 spp / 200² (noise-level).
          • Smoke-clean on `iridescent`, `multilayer`, `layered`, `_fluo_cornell`, `mixmat`, `translucency`,
            `material_presets`.
          **Still TODO:** the GPU backward **wavefront** path (`-wavefront`) is still single-λ, like the forward
          wavefront backend; hero forces the megakernel.
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
          **Still TODO:** all GPU photon-mapping paths. (Mode U's CPU half — the same complexity class as
          BDPT-D below — landed 2026-07-26, 0.69.0; see the U entry.)
    - [x] **Runtime `-heroc N` flag — DONE.** The bundle size is now a runtime CLI knob (`hero.h`: `kHeroC=4`
          default, new `kHeroMax=8` compile-time cap for the fixed stack arrays). `main.cpp` parses `-heroc N`
          (clamped 1..kHeroMax) into `g_heroC` and threads it to every CPU hero path: `Renderer::heroC`
          (modes A/B/C + photon-map M/S via `tracePhotonPass`), `BackwardRenderer::heroC` (mode R), and
          `sppmPass` (mode S). All `[kHeroC]` stack arrays in `render.h`/`backward.h` became `[kHeroMax]`; the
          hero gate is now `heroC>1`. Verified on `cornell` mode M: `-heroc 1` is bit-identical to a `kHeroC=1`
          rebuild (2.80M photons, auto-exposure 1.11e-13), `-heroc 4` matches the default (9.09M, 1.11e-13),
          `-heroc 2`/`8` interpolate and run clean; mode B `-heroc 1` gives `sum/emitted=1.000000`.
    - [x] **Optional split-at-dispersion (crisp dispersive caustics) — DONE 2026-07-26 (VERSION 0.65.0), CPU
          forward.** Shipped as `-herosplit` (off by default). Implementation:
          * `src/hero.h`: new `hero::gSplit` runtime flag. Unlike `heroC` — which the drivers vary per pass (the
            meter pre-pass, the media/GRIN/lens gate) and therefore thread explicitly through every renderer entry
            point — this is one global policy choice, so `main()` sets it once while parsing argv and
            `Renderer::heroSplit` default-initialises from it. That is why modes `A`/`B`/`C` **and** the `M`/`S`
            photon deposit all picked it up with zero call-site churn (`photonmap_render.h` and, through it,
            `sppm_render.h` both just construct a `Renderer`).
          * `src/render.h`: the bounce loop of `tracePhotonHero` was extracted verbatim into a new
            `tracePhotonHeroLoop(..., ray, stk, lam, beta, secAlive, bounce0, ...)`. That is the whole trick — the
            split branch **re-enters that method recursively**, once per secondary, so the ~20 `return` sites in
            the loop body keep working unchanged and no explicit work stack / CPS rewrite was needed. Each
            secondary runs `interactPhotonSpecular` with its **own** λ (its own Snell direction / grating order /
            Stokes shift) on a **copy** of the `MediumStack` (sub-paths diverge from that vertex), then continues
            from `bounce + 1` with `secAlive = false`. Because the branch is guarded on `secAlive`, a sub-path can
            never re-split: recursion is at most **one level deep**, so cost is linear in C, not exponential, and
            the per-frame footprint is a bounded ~600 B.
          * **Weights / ledger.** No ×C boost: the C sub-paths keep `base/C` each and the parent zeroes
            `beta[i]`, so the total is exactly what the de-hero'd hero would have carried alone and every
            sub-path books its own terminal fate. `interactPhotonSpecular` already books `e.absorbed += beta` on
            every `return false`, so nothing leaks.
          * `main.cpp`: `-herosplit` parse, a startup notice naming the supported modes (and warning that it is a
            no-op under `-heroc 1`), plus new `-heroc`/`-herosplit` lines in the usage text (`-heroc` had never
            been listed there).
          * **Validation** on a new scene `scraps/abs_herosplit.ftsl` (absolute-exposure Cornell box with a
            dispersive `glass:SF10` flint sphere, mode B, 256², fixed gain 6, CPU), vs a 200M-photon reference:
            flag **off** byte-identical to `scraps/ftrace_base_cc20a46.exe` (md5 `e2eef2cb…`) and
            `-heroc 1 -herosplit` byte-identical to plain `-heroc 1` (md5 `2b8f0fe8…`); flag **on**
            `sum/emitted = 1.000000` exactly (0.999990 off) with mean luminance −0.026 % vs the reference
            (off −0.054 %) ⇒ both unbiased. **At equal wall clock (180 s):** caustic-region noise RMS luma
            0.0340→**0.0303 (0.89×)**, chroma 0.0386→**0.0271 (0.70×)**; whole frame 0.92× / 0.80×. Cost
            **1.11×** per photon (20M back-to-back: 173.3 s off, 192.3 s on). The split PNG is 6 % *smaller*
            (91 126 vs 97 040 B). New helper `scraps/region_rms.py` reports RMS inside a box as well as over
            the whole frame — whole-image RMS is dominated by the flat diffuse walls and hides the caustic.
            **Methodology warning:** an early reading of 1.54× cost was wrong — the two runs were ~25 min apart
            and machine throughput drifted 1.6× in between. Compare timings **back-to-back** only, or give both
            policies the same `-time` budget and compare photon counts.
          * Still de-hero (not split) at `Layered` and `Mix`: those are λ-dependent *decisions* (coat Fresnel
            probability, child selection) rather than λ-dependent *directions*, and their split point sits
            before any interaction so it doesn't fit the resume-from-a-ray shape. Logged in `known-issues.md`.
          * NOT done (deliberately, both noted in the README and the startup notice): the GPU forward tracer
            (execution divergence as the fan-out wavelengths take different branches — and the emission
            back-pressure / fixed work-pool needed to keep the photon buffers bounded), and the backward tracer
            `R` / BDPT `D` / VCM `U`.
          Original note: an *alternative* to the default de-hero
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
    - [x] **U (VCM/UPS) — DONE 2026-07-26 (CPU 0.69.0, GPU 0.70.0).** Both subpaths carry the N λ and all four
          strategies (emission, NEE, connection, merging) evaluate per-λ with BDPT-level MIS. CPU *and* GPU
          (sub-items below).
        - [x] **CPU (`src/vcm.h`) — DONE 2026-07-26 (VERSION 0.69.0).** Both subpaths now carry the bundle.
              `vcmPass` draws one `bdpt::HeroBundle` per **path index** (replacing the single `lam[i]`/`invLam[i]`
              draw) so light path *p* and camera path *p* share the same C wavelengths — which is exactly what
              makes strategy (c), the paired vertex connection, EXACT per-λ, as in BDPT. `hero::sampleBundle`
              with `C == 1` is literally `emitSampler.sampleAt(rng.uniform(), pdf)`, so `-heroc 1` consumes the
              same rng draws in the same order and is bit-identical (verified against the 0.68.1 binary on
              `cornell` mode U: `cmp` clean).
              `scatterSample` gained the `bdpt.h::randomWalk` secondary block verbatim — per-λ `secF[]` for
              Diffuse/Fluorescent, Glossy, DiffuseTransmit (dividing by the HERO's lobe albedo, since the hero
              chose the lobe), Mirror and Filter (`keepBundle`: delta but λ-independent in direction), and the
              de-hero collapse `if (delta && !keepBundle) nUp = 1` for Dielectric/HalfMirror/ThinFilm/
              Multilayer/Grating. Every `<= 0` early-out became a max-over-live-λ test (hero.h policy 3), and
              Beer-Lambert absorption in the nested-dielectric stack is now per-λ on both walks (that IS the
              colour of coloured glass). All four strategies are per-λ: (a) emission, (b) NEE, (c) connection
              at `nUp = min(camera, light-vertex)`, each normalised by `1/nUp` at the accumulate — the same
              average-the-bundle rule bdpt.h uses at its splat, NOT a ×C boost, which two independently
              de-hero'd subpaths would square.
              (d) **merging** is the one strategy that crosses paths, so — exactly as the single-λ version
              already keyed on `LightVertex::lambda` — it stays keyed on the LIGHT vertex's wavelengths: the
              camera BSDF is evaluated at each of the light vertex's live λ, weighted by that λ's stored
              throughput and cached CIE, and averaged over `nUp_lightvertex`. The MIS weights everywhere stay
              the hero's, since every sampling density in this renderer is wavelength-INDEPENDENT.
              **Memory:** the secondaries live in a *parallel* `std::vector<LightVertexSec>` indexed in lockstep
              with `lightVerts`, never inside `LightVertex` — stored light vertices are the dominant cost of a
              VCM pass (and of the GPU slab), so a `-heroc 1` run allocates exactly nothing extra.
              **Validated** on `scenes/absolute.ftsl` (absolute units ⇒ physically-linear film, so two runs are
              directly comparable): C=1 vs C=4 at 1600 passes agree to a mean bias of **−0.001 %** (rms 1.36/255,
              below either estimator's own 400→1600 self-noise), while C=4's self-noise drops from rms 3.11 to
              2.28 (~1.85× variance reduction). On a purpose-built Wratten-58 gel + mirror box — the `keepBundle`
              stress case — C=1 vs C=4 at 3200 passes agree to **+0.016 %** and C=4 halves the RMS noise
              (1.84 → 0.93, ~4× variance reduction), which is the ideal C=4 win.
        - [x] **GPU (`src/render_cuda.cu` — the device session lives there, not in a `vcm_cuda.cu`) — DONE
              2026-07-26 (VERSION 0.70.0).** `kVcmLight`/`kVcmCamera` became `kVcmLightT<NS>`/`kVcmCameraT<NS>`,
              templated on the secondary slot count exactly like `kBdptT<NS>`, so the `<0>` instantiation sizes
              every per-λ array at 1, compiles all hero loops away, and keeps `-heroc 1` **bit-identical**
              (`cmp`-clean vs the 0.68.1 binary on `cornell` mode U). As predicted above, the secondary payload
              is a **parallel** slab `lvSec[(i*vcmCap+k)*secStride + j]`, `secStride == C-1`, never inline in
              `DVcmLV` — and each slot is only 16 B (`{double beta; float lam;}`) because the CIE weights are
              recomputed from `lam` at gather time (bit-identical to caching them, since the hero's own
              `DVcmLV::cx` is just `(double)cieX(lambda)`). `lamBuf`/`invLamBuf` were widened to stride C so
              light path *p* and camera path *p* share the bundle, which is what makes the connection strategy
              exact per-λ (`nUpConn = min(nUp_cam, lv.nUp)`); merging stays keyed on `lv.nUp`. `dVcmScatter`
              gained the absolute per-λ `secF[]`/`secChromatic`/`keepBundle` block, with Grating's `r<=0` bail
              deliberately kept (it gates the RNG-consuming `gratingDiffract`).
              **Validated** on an RTX 4090 at 200², 2048 spp vs 32768-spp single-λ references (luma / chroma,
              C1 → C4): `absolute` 0.94→0.84 / 1.24→**0.98 (0.79×)**; `abs_hero_delta` 0.87→0.75 /
              1.36→**1.11 (0.82×)**; `abs_hero_diffuse` 0.86→0.75 / 1.10→**0.80 (0.72×)**; `abs_hero_mats`
              0.91→0.76 / 1.25→**0.92 (0.74×)**; cost 1.50–1.69×. C4-vs-C1 bias on `absolute` at 4096 passes
              **+0.011 %**, and CPU vs GPU hero VCM agree to **0.028 %** (GPU 31× faster: 5.9 s vs 182.6 s).
    - [x] **D (BDPT) — DONE 2026-07-26 (CPU 0.60.0, GPU 0.61.0).** Both subpaths carry the N λ; the connection
          term evaluates per-λ. CPU *and* GPU megakernel (sub-items below).
        - [x] **CPU (`src/bdpt.h`) — DONE 2026-07-26 (VERSION 0.60.0).** Both subpaths now carry the bundle.
              `HeroBundle` (the C wavelengths + their `invPdfLambda`) is drawn once per sample from ONE stratified
              base draw (`u + i/C` wrapped, through `scene.emitSampler.sampleAt`) and handed to
              `generateCameraSubpath` / `generateLightSubpath` / `connectBDPT`, so both sides of every connection
              speak about the same λ's. `Vertex` gained `betaSec[kHeroMax-1]` + `nUp`; `randomWalk` propagates the
              secondaries with a per-material `secRatio[i] = f_{i+1}/f_hero` reweight (Diffuse/Fluorescent
              `rho_i/rho_hero`, Glossy `r_i/r_hero`, DiffuseTransmit `rho_i(lobe)/rho_hero(lobe)` for the lobe the
              hero's coin picked) and **de-heros at every delta vertex** (`nUp = 1`). All four `connectBDPT`
              strategies (s==0 pure eye, t==1 light-image splat, s==1 NEE, interior) evaluate `f`/`Le` per-λ and
              fill a `Lsec[]` out-parameter; every zero early-out became a max-over-live-λ test, since the hero can
              legitimately be black where a secondary is not.
              **The key derivation:** because every *sampling* decision (emitter pick, direction, NEE point, RR)
              is hero-driven, the MIS weight is identical for all λ — so `misWeight` is computed **once** and
              applied to the whole bundle. And the ×C de-hero boost the unidirectional tracers use is **NOT**
              folded into the vertex throughputs here: two independently de-hero'd subpaths would square it.
              Instead each vertex records `nUp` and the splat normalises once by
              `1/min(nUp_light, nUp_eye)` — which collapses to exactly the scalar estimator when either side
              de-hero'd, and is unbiased either way. Side benefit: **Glossy is non-delta in BDPT**, so unlike the
              unidirectional tracers (which de-hero there) mode D keeps the full bundle across glossy bounces.
              Gate mirrors `BackwardRenderer`: `heroC>1 && !scene.backwardMedium().enabled && !sceneHasGrin &&
              !cam.hasLens()`; `main.cpp` wires `br.heroC = g_heroC`. **Validated** (all CPU, on scenes whose
              dispersive SF10 sphere makes de-hero fire):
              (a) `-heroc 1` on `cornell` at 200²/8 spp is **byte-identical** to a pre-change rebuild;
              (b) **energy** — `scenes/absolute.ftsl` renders in absolute mode (fixed sensor gain, so the
              tone-map is identical between runs and the noisy p99 auto-exposure can't confound the
              comparison): at 2048 spp `-heroc 4` matches `-heroc 1` to **0.002 %** mean luminance;
              (c) **noise** — vs a 2048-spp reference, at 128 spp chroma-noise RMS falls 0.1564→0.1247
              (**0.80×**) and luma 0.2178→0.1967 (0.90×); the same measurement on `cornell` at 256 spp vs a
              4096-spp reference gives 0.1119→0.0908 (0.81×) / 0.1603→0.1445 (0.90×) — consistent;
              (d) **gate** — a media scene (`_fog_cornell.ftsl`) is byte-identical at `-heroc 1` and `-heroc 4`;
              (e) smoke sweep over `material_presets` / `translucency` / `mixmat` / `textured` / `absolute`
              runs clean. Cost 1.19–1.38× wall-clock (18.2s→21.6s at 256 spp on cornell), so still a win at
              *equal time*; these are trivially light scenes, and the shared-BVH amortisation grows with
              geometry. **Caveat:** don't compare hero runs by their printed `auto-exposure` — it is a p99
              statistic printed to 3 significant figures, so two runs of the *same* estimator can differ by
              ~1 % for reasons that have nothing to do with energy (this is why (b) uses absolute mode).
        - [x] **GPU BDPT megakernel — DONE 2026-07-26 (VERSION 0.61.0).** 1:1 device port of the CPU scheme, so
              the two can be diffed and neither drifts: `DHeroBundle` (C λ + their `invPdfLambda`, one stratified
              base draw) is threaded through `dGenCameraSubpath` / `dGenLightSubpath` / `dRandomWalk` /
              `dConnectBDPT`; same per-material `secRatio` set, same `if (delta) nUp = 1;` de-hero, one
              `dMisWeight` for the whole bundle, and the same `1/min(nUp_light, nUp_eye)` normalisation at the
              splat (not a ×C boost folded into `beta`). Same gate as the host CPU renderer:
              `heroC>1 && mediaN==0 && !hasGrin && !cam.hasLens()`, applied inside `renderBdptCuda(..., int heroC)`
              which `main.cpp` feeds `g_heroC`.
              **The GPU-specific design point:** the per-vertex secondary throughputs live in a **parallel array**
              (`pathSec[vertexIdx*secStride + i]`) instead of inside `DVertex`, and `kBdpt` became a template
              `kBdptT<int NS>` on the secondary-slot count, so the scalar instantiation `kBdptT<0>` sizes those
              arrays at 1 and costs **nothing**. This matters: `DVertex` is ~100 B and a thread frame holds
              `2*BDPT_MAXV` = 22 of them (~8 KB of spilled local state already, at `__launch_bounds__(128,3)`), so
              widening the struct would have cost the *scalar* path real occupancy. The hero instantiation adds
              ~1.2 KB. **Validated:**
              (a) `-heroc 1` on `cornell` 160²/256 spp is **byte-identical** to a pre-change rebuild;
              (b) **energy** (absolute mode = fixed sensor gain) — `scenes/absolute.ftsl` at 2048 spp,
              `-heroc 4` vs `1` = **−0.008 %** mean luminance; and on a new scratch scene
              `scraps/abs_hero_mats.ftsl` (a `glossy` sphere + a `translucent` quad, chosen to exercise the two
              per-λ reweights `absolute.ftsl` doesn't) **−0.000 %** at 4096 spp;
              (c) **noise** — `absolute` at 128 spp vs a 32768-spp reference: chroma 0.1614→0.1270 (**0.79×**),
              luma 0.2313→0.2087 (0.90×); `cornell` 200²/256 spp vs a 16384-spp reference: chroma
              0.1195→0.0955 (**0.80×**), luma 0.1792→0.1645 (0.92×); and `abs_hero_mats` — where Glossy and
              DiffuseTransmit are *both connectible*, so the bundle **never de-heros** — chroma 0.1794→0.0761
              and luma 0.1508→0.0634, i.e. **0.42× on both**, a far bigger win than the dispersive scenes;
              (d) **gates** — `_fog_cornell.ftsl` (media) and `scenes/realcam.ftsl` (finite lens) are each
              byte-identical across `-heroc 1`/`4`;
              (e) **cross-backend** — CPU vs GPU hero BDPT agree to **0.03 %** mean luminance on `absolute` at
              2048 spp;
              (f) smoke sweep `material_presets` / `translucency` / `mixmat` / `textured` at `-heroc 4` clean.
              Cost 1.17–1.18× wall-clock, so a clear win at equal time. Renders in `png/bdpt_hero_gpu/`.
        - [x] **Mirror / Filter keep the bundle — DONE 2026-07-26 (VERSION 0.62.0).** `Mirror` and `Filter`
              are delta, but neither consults λ to pick its continuation (a mirror reflects, a gel passes
              straight through), so the "every delta vertex de-heros" rule was leaving free chroma-noise on
              the table in mirror-heavy scenes. Both now set a `keepBundle` flag that skips the
              `if (delta) nUp = 1;` collapse (CPU `bdpt.h` + GPU `render_cuda.cu`, kept 1:1).
              **This exposed a genuine bias in the ratio formulation.** The per-λ factor was `secRatio[i] =
              f_i / f_hero`, which is undefined precisely where it matters: a Wratten gel's `T(λ)` is
              legitimately **0** across most of the spectrum, so `T(λ_hero) == 0` while a secondary is wide
              open — and the old `if (f_hero <= 0) terminate` then threw away the live secondaries too
              (measured **−4.9 %** energy). The array is now the **absolute** factor
              `secF[i] = f_i·cos/pdf_hero` (with a `secChromatic` flag so the λ-independent lobes just reuse
              `betaFactor`), and the early-out became a **max-over-live-λ** test — which also removes the
              same latent bias for Diffuse / Glossy / DiffuseTransmit when a spectral albedo hits exactly 0
              at the hero λ. At `nUp == 1` every hero loop is empty and `mxF == betaFactor`, i.e. the old
              scalar test verbatim. **Validated** on a new scene `scraps/abs_hero_delta.ftsl` (absolute mode,
              `metal:gold` mirror back wall + sphere, `filter:wratten-58` gel pane):
              energy `-heroc 4` vs `1` = **−0.006 %** at 2048 spp (−3.978 % with the ratio form; isolating
              the lobes gave mirror −0.006 %, filter −4.894 %, pinning it on the filter);
              noise at 128 spp vs a 32768-spp reference — single-λ chroma 0.1801 / luma 0.3685, hero-4
              *before* 0.1676 / 0.3522 (**0.93× / 0.96×** — the bundle died at the first mirror, so hero
              bought almost nothing), hero-4 *after* 0.0842 / 0.2086 = **0.47× chroma, 0.57× luma**;
              `-heroc 1` byte-identical to the pre-change build on GPU (`cornell` 160²/256 spp,
              `abs_hero_delta` 200²/2048 spp) and CPU (`cornell` 96²/64 spp); the earlier scenes unregressed
              (`absolute` −0.008 %, `abs_hero_mats` −0.000 %, both exactly as before); smoke sweep
              `mirror_selfie` / `group` / `material_presets` / `translucency` / `mixmat` clean.
              Renders in `png/hero_delta/`. Ported to the **backward** tracer in the next item and to the
              **forward** tracers in the one after that, so every hero tracer now keeps the bundle here.
        - [x] **Backward tracer (R): achromatic delta lobes + max-over-λ Russian roulette — DONE 2026-07-26
              (VERSION 0.63.0).** Two changes to `radianceHero` (`src/backward.h`) and its device twin
              `bkRadianceHero` (`src/render_cuda.cu`), kept 1:1. (1) `Mirror`/`Filter`/`Glossy` stop
              de-heroing — their outgoing *direction* is λ-independent, so the bundle rides on; since the
              unidirectional form is an *analog RR* rather than a throughput multiply, the survival
              probability became `q = max_i c_i` with survivors reweighting `thr[i] *= c_i/q ≤ 1`.
              (2) **The diffuse continuation stops rolling its coin on the hero alone.** The old
              `if (u >= rho[0]) die; thr[i] *= rho_i/rho_0` *amplifies* a secondary by up to
              `rho_max/rho_hero` — on the built-in `redWall`/`greenWall` spectra (0.05 … 0.75) a **15× weight
              spike per diffuse bounce**. Now `q = max_i rho_i`, `thr[i] *= rho_i/q ≤ 1`; DiffuseTransmit
              picks its lobe from the per-lobe maxima (rescaled if they sum past 1). At `nUp == 1` both are
              the scalar code verbatim, so `-heroc 1` stays byte-identical (verified on both backends for
              `abs_hero_delta` and `abs_hero_mats`, 256²/64 spp).
              **(2) is what made hero actually pay off in mode R — (1) alone regressed luma.** Noise RMS
              (luma / chroma) vs a 262144-spp `-heroc 1` reference, GPU 256²:
              `scraps/abs_hero_diffuse.ftsl` (coloured Cornell, no delta lobes) at 4096 spp — single-λ
              0.0604 / 0.0714, hero-4 *before* 0.0605 / 0.0565 (**hero was worth nothing in luma**), *after*
              **0.0254 / 0.0254**; `scraps/abs_hero_mats.ftsl` at 4096 spp — 0.0875 / 0.1136, before
              0.1566 / 0.1110 (**worse than single-λ**), after **0.0398 / 0.0597**;
              `scraps/abs_hero_delta.ftsl` at 16384 spp — 0.1122 / 0.0955, before 0.1062 / 0.0936,
              keepBundle-only 0.1305 / 0.0710, after **0.0549 / 0.0393**. That is ≈0.42–0.52× RMS = a **4×
              variance cut** for 1.35× CPU / 1.36× GPU wall-clock ⇒ **≈2.9× at equal noise**; energy
              unbiased (−0.015 % … +0.034 %). Diagnosis trail: an achromatic-wall control
              (`scraps/abs_hero_delta_gray.ftsl`) showed change (1) alone going 0.0755→0.0406 luma with no
              regression, pinning it on the coloured diffuse walls. Renders in `png/hero_achroma2/`,
              `png/hero_diff/`, `png/hero_mats/`, `png/hero_gray/`. Full write-up in known-issues.md,
              including the reference-correlation measurement trap.
        - [x] **Same two fixes for the FORWARD tracers — DONE 2026-07-26 (VERSION 0.64.0).**
              `tracePhotonHero` (`src/render.h`) and its device twin `shadeStepHero`
              (`src/render_cuda.cu`) got the identical pair: Mirror/Filter/Glossy stop de-heroing, and
              Diffuse / DiffuseTransmit / the new delta group all survive on `q = max_i c_i` with
              survivors reweighting `beta[i] *= c_i/q ≤ 1`.
              **The forward tracers keep an energy ledger, and it caught a bug the backward port could not
              have:** the reweight is *deterministic absorption*, so it must be booked — without
              `e.absorbed += beta[i]*(1-w)` the ledger fell to `sum/emitted = 0.6597`. With it the ledger
              closes *exactly* for the first time — the **old** ratio reweight created ledger energy on
              every scene tried (`sum/emitted` 1.00281 `cornell`, 1.00459 `group`, 1.00172
              `material_presets`, 1.00743 `mixmat`, 1.00713 `abs_hero_diffuse`, 1.00696 `abs_hero_mats`;
              all now 0.999996 … 1.000003). That was a ledger inconsistency, not image bias
              (`E[ρ₀·βᵢρᵢ/ρ₀] = βᵢρᵢ` is correct); what the amplification cost was **variance**.
              Noise RMS (luma / chroma) vs an 8e9-photon `-heroc 1` reference, GPU mode B 256², all tests
              at 200 M photons: `abs_hero_delta` — single-λ 0.0460 / 0.0350 (2.9 s), hero-4 *before*
              0.0492 / 0.0342 (5.8 s), *after* **0.0289 / 0.0172** (7.3 s), single-λ at equal time
              0.0321 / 0.0228; `abs_hero_diffuse` — 0.0407 / 0.0563, before 0.0514 / 0.0539, after
              **0.0254 / 0.0216**, equal-time single-λ 0.0269 / 0.0361; `abs_hero_mats` — 0.0377 / 0.0587,
              before **0.1069** / 0.0443 (mean luminance still −0.251 % off — heavy tails), after
              **0.0233 / 0.0239**, equal-time single-λ 0.0247 / 0.0423. So hero was a net *loss* in the
              forward tracers too, and is now a win — but a **smaller** one than mode R's ≈2.9×, because
              forward hero shares only the main path's BVH walk while the per-λ camera splat / photon
              deposit costs a full C×: ~1.1× luma and 1.3–1.8× chroma at equal GPU time (CPU amortizes
              better — `abs_hero_mats` 20 M: 0.1148 / 0.0973 in 6.5 s vs 0.0641 / 0.0333 in 12.3 s ⇒ 1.30×
              luma / 2.13× chroma). `sum/emitted = 1.000000` on both backends, `-heroc 1` byte-identical,
              mode M deposit unbiased (+0.002 %). Renders in `png/heroFwd/`, `png/heroFwdD/`,
              `png/heroFwdM/`. Full write-up in known-issues.md.
    - [x] **Shared plumbing — DONE 2026-07-26 (no VERSION bump: zero observable change).** Landed as the
          *achievable* subset of the original idea, after surveying what is actually duplicated. Two helpers plus
          one comment block went into `src/hero.h`:
          * **`hero::sampleBundle(sampler, u, C, lam, pdf)`** — POLICY (1), the stratified λ draw itself (hero
            takes `u`, secondary i takes `u + i/C` wrapped, all through the same emission CDF; returns false iff
            the *hero's* pdf is non-positive, since only it must be valid). It was three near-identical copies;
            now one. Templated on the sampler so the header stays dependency-free and both callers fit: the
            forward tracer samples one `Emitter::spd`, the backward/BDPT tracers the scene-wide
            `Scene::emitSampler` (both `EmissionSampler`).
          * **`hero::maxOf(a, n)`** — POLICIES (3)/(4), the max over live λ, used both as the analog-RR survival
            probability and as the "is the whole bundle black?" early-out. Replaced 6 hand-rolled loops
            (`render.h` ×3 including the DiffuseTransmit per-lobe pair, `backward.h` ×3).
          * **A single authoritative POLICY block** at the top of `hero.h` stating all four rules — (1)
            stratification, (2) which lobes de-hero, and that the criterion is a *λ-dependent direction* rather
            than merely being delta, (3) analog RR is max-over-live-λ never hero-only (with both failure modes:
            `c_hero == 0` kills live secondaries, and hero-only ratios amplify by up to 15× per diffuse bounce),
            (4) per-λ factors are absolute never ratios — plus the two rules that deliberately are *not* shared
            (the forward tracers must book the reweight as absorption in their energy ledger and nobody else may;
            BDPT normalises a connection by `1/min(nUp_light, nUp_eye)` at the splat instead of folding a ×C
            de-hero boost into vertex throughputs, since two independently de-hero'd subpaths would square it).
            This is the part that actually pays: the prose was duplicated near-verbatim across three files, and
            each of the two real bugs in this section's history had to be found and fixed independently in two-to-
            four places.
          **Why NOT the original "small `HeroLambda` struct threaded through every spectral evaluation site":**
          the four tracers are genuinely different *estimators*, not four copies of one. The forward tracers
          multiply throughputs and keep an energy ledger; the backward tracer runs analog RR on a radiometric
          weight and books nothing; BDPT stores an absolute per-λ factor per vertex, computes ONE MIS weight for
          the whole bundle, and must not de-hero-boost at all. Forcing them behind one struct would have produced
          a struct whose fields each tracer reinterprets — worse than the duplication it removed. The GPU twins
          in `render_cuda.cu` stay separate on purpose, as decided earlier in this section.
          **Validated as a pure refactor: 24/24 renders byte-identical** to `scraps/ftrace_base_99a898d.exe` —
          the full cross product {`abs_hero_delta`, `abs_hero_mats`, `abs_herosplit`} × {mode B, R, D} ×
          {`-heroc 1`, `-heroc 4`} on CPU (18), plus CPU mode M, mode S and `-herosplit`, plus GPU modes B/R/D at
          `-heroc 4` (6). Renders in `png/heroRefac/`.
    - [x] **Docs + version — DONE (rolling, through 0.70.0).** README's spectral bullet, the "what ftrace is
          actually good at" §, the renderer-comparison table and the `-heroc <N>` flag row all state the current
          coverage (CPU `A/B/C`, `R`, `M/S`, `D`, `U`; GPU megakernel `A/B/C`, `M`-deposit, `R`, `D`, `U`) and
          the exclusions (the GPU wavefront scheduler, and any scene with media / GRIN / a finite-lens camera). The
          deliberate secondary-drop is documented everywhere as the de-hero policy (dispersive or
          wavelength-switching interface → secondaries terminated, hero boosted ×C — Wilkie et al. 2014 /
          PBRT-v4 `TerminateSecondary`). VERSION took a minor bump per landing (…→0.59.0 for GPU mode R).
          Re-check this box's wording whenever a further mode gains hero.

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
- 2026-07-27: **Measured G3/G4 instead of guessing — overturned both deferrals' stated reasons, and the
  measurement pointed at a third thing that was actually the bottleneck (v0.84.3).** Built a probe method
  that changes *only* program size while keeping output bit-identical (rewrite the field `E` as `(E+E)/2`
  — bit-exact in IEEE, and unlike a padding tail it adds *realistic* nodes), asserting identical output
  md5 / triangle counts across x1/x2/x4 variants, so any time delta is pattern eval and nothing else.
  Findings: (1) `patternEval` costs **~4.7 ns/node regardless of opcode** — a `Mul` costs what a `sin`
  does — so G3's premise "sin/cos + the sphere-march dominate" is **false**; cost is purely node count,
  and affine rows are 85–89% of every emitted field, so MatRow really would cut nodes 3.5–4×. (2) But
  field eval is only **7% of `-raster-gpu`, 12% of `-raster`, 11–14% of an export** at the sizes actually
  rendered (~4% for a default random draw) — so G3 stays deferred on **Amdahl**, not opcode mix, with a
  new numeric trigger (**>800 nodes**, i.e. `--coupling all` at D≥8; measured D=16 fully coupled = 3917
  nodes, 62% field eval, 1.9× win). (3) G4's premise was also wrong: a res-160 export was **55% ASCII OBJ
  write**, not marching, capping even a *free* GPU march at 1.8×. So the right fix was the writer, not a
  CUDA port — `isomesh::writeObj` was doing one `fprintf` per line (~6.6M calls, 22 MB/s); it now formats
  into an 8 MB buffer flushed with `fwrite` plus hand-rolled integer conversion for face lines, keeping
  the same `snprintf` specifiers for floats so output is **byte-identical** (md5-verified against the old
  binary). **Write 11.77 s → 4.66 s (2.5×), export 21.47 s → 13.40 s (1.6×)** — a bigger real win than
  MatRow gives on any default scene, for a fraction of G4's cost.
- 2026-07-27: **loom retime + the 4-D time-shear — time is now a value you can pass, not just the frame
  you happen to be on.** (loom-only; no `ftrace.exe` change, so no `VERSION` bump.) A `Signal` was always a
  *pure function of a `Clock`*, so evaluating one at another phase was already well-defined and cheap —
  the only thing that ever assumed one-value-per-node-per-frame was the **memo**. So the whole family fell
  out of two additions: **`Phase`** (a leaf returning `clock.t` *as a value*) and **`Retime(x, when)`**
  (evaluate `x` against `retimed_clock(clock, when)`), plus `VecRetime` for retiming a vector as a whole.
  Sugar on top: `freeze` (a hold — and `at` may itself be animated, so a *scrubbable* hold is free),
  `delay` (`x(t−dt)`; wraps iff the clock is closed, so a delayed seamless loop stays seamless, and a
  negative `dt` legitimately looks *ahead*), `warp` (`x(g(t))`). The cache problem the roadmap flagged was
  solved by **`Cache.scope(key)`** — a nested cache keyed `(node id, frame, sample phase)` — rather than
  widening the global key, which would have touched every `at()` call site; sharing still works *within*
  one sample point and nothing leaks out to the frame. The cycle question resolved to "nothing to add":
  both retime edges are ordinary structural `children()`, and a retime is **not** a recurrence (it reads a
  pure function elsewhere, not its own past), so `detect_signal_cycle` still owns it and the temporal
  causality guard stays deferred until an actual stateful node exists. The payoff is the spatial-tier
  **`SigAt`**: a modulator sampled at a phase that is *itself a field* — `SigAt(Sine(cycles=3), T − X/c)`
  is a wave whose phase lags with distance, i.e. a genuine shear of the spacetime block, and the one thing
  that could not be faked by animating a coefficient (a bare `Signal` used as a spatial term bakes one
  number per frame for the *whole* field). `SigAt.emit()` deliberately raises — ftrace evaluates a pattern
  per hit with no access to loom's modulator DAG, and baking one number would silently drop the shear —
  making it the second single-backend leaf in `spatial.py` alongside `VolumeField`; the workflow is
  discretise-then-render. Cost is one graph evaluation per *distinct* phase, with `quantize=k` capping it.
  27 new tests (1150 → **1177**), mutation-checked three ways; notably the cache tests had to be rewritten
  to use a **sub-frame** retime — with a coarser delay the retimed phase maps to a *different* frame index,
  the keys never collide, and the first version of the test passed even with the bug put back.
  `tools/loom/{loom/signals/retime.py,loom/signals/core.py,loom/spatial.py,tests/test_retime.py,
  examples/time_shear.py,design.md,README.md}`.
- 2026-07-27: **0.83.0 — the viewer's Modulator DAG pane now shows the whole graph
  (adaptive height + wrapping layers + a real zoom).** Reported as "the modular DAG pane doesn't seem to
  be tall enough to show the whole thing," and it had three independent causes. (1) The pane was a
  hard-coded 360 px child; it now sizes to `min(extent.y, remaining side-column height)` so it fills
  whatever the window leaves it. (2) The layout was structurally taller than *any* pane: longest-path
  layering puts every leaf in level 0, and the stress sidecar has 60 of them — ~6600 px of column no
  matter how tall the pane is. `measureDag()` now **wraps each level into sub-columns** against the
  available height, so a wide-and-shallow graph turns into a grid instead of one endless column. The
  first frame has to *estimate* node boxes (imnodes hasn't laid them out yet), and an estimate that
  ignores `NodePadding` and the DPI-scaled font overlaps nodes — so after `EndNodeEditor()` the panel
  reads back `ImNodes::GetNodeDimensions()` for every node and, if any differs, re-wraps next frame from
  the *real* rects. That settles in two frames and is DPI-correct by construction rather than by fudge
  factor. (3) Even a full-window pane can't show a graph wider than the canvas, and **this imnodes build
  has no zoom at all** — only panning. So the panel implements one: ImGui 1.92's dynamic fonts make
  `PushFont(NULL, style.FontSizeBase * zoom)` a crisp re-raster (not a bitmap stretch), and scaling
  `ItemSpacing` + `ImNodesStyleVar_NodePadding` by the same factor scales the node boxes with it — a real
  zoom, at the cost of it being a *re-layout* rather than a transform, which is why the wrap re-measures
  on zoom. Wheel zooms (15–300%); **fit** runs an iterative solve (the extent isn't a closed-form function
  of zoom once wrapping is involved) — width-only, `sqrt`-damped, and gated on the read-back having
  settled, because comparing *height* is useless when wrapping pins `extent.y ≈ availH` by construction.
  **maximize** re-hosts the same graph in a full-viewport window (Esc/`dock` to return), auto-fits on
  entry and re-fits on resize. The child uses `NoScrollbar | NoScrollWithMouse` — `NoScrollWithMouse`
  alone still forwards the wheel to the parent, which scrolled the side column out from under the zoom.
- 2026-07-27: **0.82.0 — `emit pattern:` / `emit_map` now runs on the CUDA backends; the last 0.80.0
  tech-debt item is closed.** Emission is the one throughput slot read from **both sides of transport** —
  emission-on-hit (the s=0 / direct-hit / specular-arrival strategy) *and* the Le at a point the emitter
  *sampler* drew (NEE, light subpaths) — and MIS **combines** them, so the two must agree pointwise. That
  is exactly why 0.80.0 shipped it CPU-only behind a whole-scene reject: a *partially* ported pattern
  biases the image instead of dropping a visible effect, and the device has ~20 emission read sites rather
  than the one or two `reflectPat`/`transmitPat` funnel through. All of them landed at once.
  `DEmitter::emitPat` / `DMaterial::emitPat` upload; `DEmitTri` carries `uv0`/`uvE1`/`uvE2` and the device
  `emitterSamplePoint` gained optional `uuOut`/`vvOut` (Quad: the bilinear `u1,u2`; Mesh: the chosen
  `EmitTri`'s barycentric UVs) so a *sampled* point reports the same (u,v) a *hit* interpolates — the
  property that makes the profile legal on those two shapes and refused on sphere/cylinder/spot/env.
  Three accessors mirror `scene.h`: `dEmitPatMul` (hit side), `dEmitterPatMulAt` /
  `dEmitterSamplePointPat` (sampler side). Two structural choices kept the site count down: the sampler
  factor folds into `bkEmitterGeom`'s λ-independent `G`, so scalar **and** hero NEE pick it up from one
  place (host twin: `emitterGeom` folding into `w`); and a cached `Real emitPatW` on `DVertex` (twin of
  `bdpt.h Vertex::emitPatW`), read by `dVertexLe`, covers every BDPT MIS strategy from one place. The
  pattern remains a pure post-multiplier on carried radiance/beta — the emitter is still *selected* by its
  unpatterned `power` and the point still drawn uniformly over its area, so **no pdf anywhere changes**
  and the estimator is unbiased by construction (only variance rises). Unpatterned scenes stay
  bit-identical: every new factor is guarded by `if (epat != 1.0)` or is an exact multiply by `1.0`, and
  the extra UV outputs consume no RNG, so draw sequences are unchanged. Both `Supported` gates dropped —
  including the **RGB fast path**, which unlike `reflectPat`/`transmitPat` is safe because an *achromatic*
  scalar commutes with the spectral→RGB bake (`ep·∫CIE·emitSpd == ∫CIE·ep·emitSpd`), so it applies
  straight to the pre-baked `rgbEmit`. Two deliberate non-sites: `dInvPdfLambda` (a wavelength pdf,
  matching the host) and the spot/env branches (refused at load). **Validated** on
  `scenes/emit_pattern.ftsl` (two patterned quad area lights + a patterned *mesh* emitter, i.e. both
  UV-carrying shapes): GPU-vs-CPU mode R at 2000 spp / 512² mean ratio **0.9999** (median 1.0000, sRGB
  RMSE 2.36/255, under the images' own 2.24% noise floor); GPU cross-estimator global B/R **1.00001** and
  D/R **0.99995**; GPU-VCM-vs-CPU-VCM **0.9998**; RGB fast path vs spectral R **0.9991**; mode-M photon
  map GPU-vs-CPU **1.0056** (median 1.0000; the residual is M's own density-estimate noise, as in M2/M4);
  forward energy closure `sum/emitted = 1.000000` at 4×10⁹ photons. U/R came in at 1.00679, but an **unpatterned control**
  gives 1.00706 with the same per-band profile — a pre-existing VCM-vs-R estimator difference on this
  scene, not the pattern (the band holding the directly-visible patterned panels is U/R = 1.0002). 11/11
  `-check*` self-tests pass; all 80 `scenes/*.ftsl` load. `raster.h`/`raster_cuda.cu` still ignore
  `emitPat`, deliberately and consistently with `reflectPat`/`transmitPat` — a cosmetic preview mismatch,
  not bias.
- 2026-07-26: **0.81.0 — E5 is complete: scene value-sites route through `Target`, and the viewer shows the
  axis model.** Two follow-ups, both loom-side plus one C++ panel.
  *(1) Routing.* `Lift` took a clock-parameterized `Signal` **up** into the axis layer; nothing brought one
  back **down**, so `Target` — the pin/mod combine node — was a self-contained algebra that could not
  actually drive `Sphere.radius` or a camera position. `Lower`/`LowerVec` are the exact inverse: the site's
  clock axis is fed `clock.t`, and every *other* axis must be pinned with `bind={'s': <coord or Signal>}` (a
  constant reads one arclength of a spatial curve; a `Signal` sweeps along it over the loop). Records-5a's
  scope rule is enforced at **construction**, naming the unbound axes, instead of failing deep inside a
  render. Routing is **one memoised hook** (`signals.core.lower_axsignal`) consumed by `as_signal`,
  `VecSignal.of`, `ftsl_emit.site_node` and `Element.roots()` — no element constructor changed, and every
  value-site accepts an axis node uniformly. Memoising the lowered node is *required*: node identity is the
  per-frame `Cache` key **and** `roots()` must hand the cycle detector the very node emission will evaluate.
  Also `mod()`/`pin()` sugar, `as_ax` now lifts a legacy `Signal`, and a latent bug is fixed — a `GAIN`
  target with a negative source computed `x ** gain`, which Python returns as a **complex**, blowing up far
  from the cause; it now raises a domain error naming the fix.
  *(2) The on-disk projection.* Decided **`.ftsl` carries no axis annotation** — it is a *bound* per-frame
  snapshot (every axis already collapsed to a number), and annotating it would make ftrace's language an
  animation format. The projection belongs in the **viewer introspection sidecar** (v1 → **v2**), which is
  what an editor reads: nodes gain their free `axes` plus target-kind / reduced-axis / value-site-scope
  detail, and an edge into a `Target` gains the `mode` + `gain` a plain child list cannot express (sources
  hang off `Binding` records, so a generic walk saw only anonymous inputs). `src/viewer_gui.cpp`'s F5 panel
  renders all of it (`axes {s,t}`, `gain target (neutral 1)`, `mod[0] x0.8`), verified live. Purely additive.
  33 new tests; 1128 loom green.
- 2026-07-27: **0.80.0 — `emit pattern:` / `emit_map`: the reflect / transmit / emit trio is complete.**
  Same mechanism as 0.75.0/0.76.0 — a scalar pattern in a spectral slot is a per-hit **multiplier**
  clamped to [0,1], so `emit pattern:<n>` (and `emit [0 1](u)`) leaves the pattern alone in the slot and
  the base collapses to a flat 1.0 (a greyscale emission profile), while `emit <spectrum>` + `emit_map
  pattern:<n>` modulates that spectrum so the lamp keeps its colour and only its brightness varies. A
  `light` block spells the same slot `spd`, so the pair there is `spd pattern:` / `spd_map pattern:`;
  `finalizeEmitters` copies `Material::emitPat` onto the registered `Emitter`, so both spellings converge
  on one runtime field. **Emission is the strict leg**, and that drove every design choice: it is read
  from *both sides of transport* — emission-on-hit (PatCtx from the `Hit`) and Le at the point NEE / a
  light subpath samples (PatCtx from `Emitter::samplePoint`) — and MIS **combines** the two, so a
  pointwise disagreement is **bias**, not noise. So (a) the profile is only legal where the sampler's
  (u,v) provably equals the hit's — `EmitterShape::Quad` (bilinear parameters) and `EmitterShape::Mesh`
  (barycentric UVs; `EmitTri` gained `uv0`/`uvE1`/`uvE2`) — with sphere/cylinder/spot/collimated/env
  **refused at load** at two points (`addLight`'s subtype gate, `checkEmitPatsSupported` after
  `Scene::build()`); (b) all reads funnel through three new accessors in `scene.h` (`emitSlot`,
  `emitterPatMulAt`, `emitterSamplePoint`, the last sampling *and* returning the multiplier in one call),
  wired into all six tracers — `backward.h` (scalar + hero emission-on-hit, `emitterGeom`, `neeVolume`),
  `render.h` (both `genPhoton` variants), `photonmap_render.h`, `sppm_render.h`, `bdpt.h` (a new
  `Vertex::emitPatW` threaded through `Le`, `randomWalk`, `generateLightSubpath` and the `s == 1` branch)
  and `vcm.h`; and (c) it is a **pure post-multiplier on radiance / photon beta** — `power`, `pdfChoice`,
  `pdfPos`/`pdfA`, `emissionPdfW`, `directPdfW` and every VCM `dVCM`/`dVC`/`dVM` are deliberately
  untouched, which is precisely what makes it unbiased by construction. Fixed a latent pre-existing bug on
  the way: the area light's *second* triangle carried default UVs disagreeing with `addQuad`'s — a
  diagonal seam for any UV-driven emission pattern **or texture** on an area light. Caveat, documented
  rather than auto-corrected: `power`/`lumens` normalise the *unpatterned* spectrum, so a profile
  averaging 0.5 emits about half the requested flux (folding the mean into `power` would need a
  compensating 1/mean on photon beta *and* on BDPT's `pdfChoice`). **CPU-only**: the device has ~20
  emission read sites and a partial port would be *biased* rather than visibly incomplete, so
  `cudaForwardSupported` / `cudaBackwardRGBSupported` reject the whole scene and the three GPU-fallback
  "why" strings name the emission profile; the port is logged in `known-issues.md`. *Verified unbiased at
  160×160 against four independent estimators:* R vs D (BDPT) global ratio 1.00268 — and a **control with
  the pattern removed** reproduces the same 1.00240 with the same corner-shaped tile profile, so that
  residual is a pre-existing R-vs-D estimator difference, not the pattern; B (forward photons, 400M) vs R
  global **1.00005** (tiles 0.9990–1.0024); U (VCM, 1500 spp) vs R global **1.00018** (tiles
  0.9973–1.0028). Load rejection of `light sphere { spd pattern:p }` confirmed; all eleven deterministic
  self-tests pass and every scene in `scenes/` still loads. Worked example: `scenes/emit_pattern.ftsl`
  (a `spd_map` ring lamp, a lone-`spd pattern:` checkerboard, and an `emit_map` material on a mesh
  emitter via the new `meshes/uvquad.obj`).
- 2026-07-27: **0.79.0 — the hand-written `.ftsl` parser is deleted; the shared grammar is now the *only*
  front end.** 0.68.0 flipped ftrace over to the grammar
  (`tools/loom/loom/grammar/ftsl_scene.epeg` → `src/gpda/`) after the corpus differ hit **MATCH 2595/2595**
  — every `.ftsl` in the tree, structurally identical down to `Stmt::line`. The old recursive-descent
  parser stayed compiled in behind `-legacy-parser` as an escape hatch and went **ten releases unused**,
  so it was time. Deleted: the `// Tokenizer` section and the 266-line `struct Parser` from `src/ftsl.h`
  (`loadSource()` now has one path — `ftsl_gpda::parse()`); `legacy_flag()` / `use_legacy()` /
  `validate_flag()` / `validate_enabled()` / `validate()` from `src/gpda/ftsl_frontend.hpp`; the
  structural differ (`Diff` / `diff_block` / `diff_value` / `diff_scene`) from `src/gpda/ftsl_reduce.hpp`,
  which had nothing left to diff against; and the argv pre-scan in `main.cpp` that existed only because
  the scene is parsed before the main CLI loop runs. `-legacy-parser` / `-validate-grammar` are **retired,
  not removed** — still accepted so an existing script keeps working, but each prints
  `ftrace: <flag> was retired in 0.79.0 …; ignoring` rather than silently doing nothing, which is what
  keeps this a *minor* bump. Everything downstream of `std::vector<Block>` is shared and untouched, and
  `applyBracketGroup` stays at the loader level (that layering was the thing that made the flip a pure
  parse-tree exercise; the comments now say so as design rather than as drift-avoidance). *Verified:* all
  eleven deterministic self-tests PASS, every scene in `scenes/` still loads, both retired flags render
  normally after printing the notice.
- 2026-07-26: **0.78.0 — `grid:`/`scatter:` reach the field formulas, and the not-found guard no longer
  corrupts the eval stack.** Authored-data tables were compile-legal only in *pattern* sites; now the
  FTSL builder also passes `&tableScope_` at the `function` field leaf, a medium's `density` and `ior`
  programs, and `camera_curve` drivers — so `expr "grid:terrain(x, z) - y"` is a measured height field,
  `ior "1 + grid:n(x, y, z)"` a measured GRIN profile. (`tex:` stays a compile error there: it needs a
  hit's u,v, which a field formula has none of.) Evaluation was the real work: a compiled `PatOp::Grid`
  node holds an index into `Scene::grids`, so a non-owning `PatTables` view (`Scene::patTables()`;
  device `dPatEnvOf`) is threaded as a **parameter** through `Implicit::eval`/`gradient`,
  `intersectImplicit`, `estimateFieldLipschitz`, `Medium::densityAt`/`nAt`/`gradNAt`/`insideBound`,
  the GRIN marcher, `isomesh::marchImplicit` and `airtight::check` — never a member, since a `Scene`
  is copied and moved and a stored view would dangle. The multi-medium wrappers were retyped from
  `media` to the whole `Scene`/`DScene` (18 host + 24 device call sites) so no caller can *forget* the
  tables. The load-time consumers get them too: the Lipschitz bound, the majorant-density scan and the
  `boundInsideNeg` sign test all previously read a grid-driven field as identically 0. **This was a
  live wrong render, not latent debt:** `medium { density pattern:<p> }` copies a table-scoped
  pattern's nodes into a medium that was evaluated without tables, and the guard pushed 0 *without
  popping its operands*, so `patternEval` returned the first **coordinate** as the sample. Proven with
  a descending grid (`data { 1 0 }`) whose analytic twin is `density "1 - x"`: before the fix the two
  renders were mirror images, after it they are byte-identical (`mean|d|=0.000`) on **both** backends
  while the mirrored comparison is far off. Both guards now `return 0` — abandoning the whole program
  is the only balanced option, since the operand count is the missing table's own `ndim`. The FP32
  device VM's `Tex`/`Grid`/`Scatter` stubs were implemented at the same time (promote/demote around
  the double-only samplers, as `PovFn` already did). Pinned by two new `-checkgrid` sections — an
  unbound table evaluates to **0, never to a coordinate**, and `Medium::densityAt` fed
  `Scene::patTables()` returns the sample while omitting the tables returns a clean 0 — and by a
  matching isosurface pair (`scraps/gridiso{,_ref}.ftsl`, a 2×2 lattice holding a bilinear plane;
  agrees to the grid pool's float storage, ~7e-9 m of silhouette, on both backends). New worked
  example `scenes/grid_field.ftsl` exercises both new sites at once.
- 2026-07-26: **0.77.1 — a missing `-o` directory no longer eats the whole render.** `known-issues.md`'s
  oldest papercut (hit twice): `-o png/nope/out.png` traced every photon, printed `error: could not write …`
  at each `-interval` tick, and exited with nothing — the film was reachable only by the writers, which run
  *after* it exists. Fixed with an `ensureOutDir` precheck in `main.cpp`, placed after the `-check*`
  self-test early-returns because that's the earliest point where `out` is final (the bare-invocation
  preview path can still rewrite it to a `$TEMP` name). It creates the missing parent (announced as
  `[out] created output directory …`, so a typo shows up in the log rather than silently) and exits 2 with a
  named error if it can't, or if the parent exists but isn't a directory. Policy is *create*, not *refuse*,
  because renders are routinely aimed at a fresh `png/<series>/`. One check on `-o` covers the `.ftbuf`
  sidecar, the per-camera `outFor()` names and the stereo eye pair (all share its directory); `-savemap` is
  the only independent output path and gets the same treatment. Verified both ways: a 3-deep nested `-o`
  creates the tree and writes PNG + `.ftbuf`, and a parent that is a regular file fails before the scene
  renders.
- 2026-07-26: **E4 — loom reads and writes rotated `.vdb` transforms.** Closed a real loom↔ftrace
  asymmetry: ftrace's `readTransform` has always accepted `AffineMap`/`UnitaryMap` and honours them
  properly (inverting the 3×3 for world→index sampling, AABB'ing the index box's 8 corners), while loom's
  reader rejected them outright. The insight that shaped the API: **a rotation costs the samples nothing**
  — an OpenVDB tree is a regular lattice in *index* space regardless — so the dense array is untouched and
  only the axis-aligned `box6` is inexpressible. So `read_vdb` is unchanged and still refuses a rotated
  grid (an approximate box would silently misplace every voxel), and the new `read_vdb_grids` returns
  `ReadGrid` records carrying the index-space array + `index_lo` + a full `VdbTransform`; `VolumeGrid(…,
  transform=…)` writes one as an `AffineMap`. Refactoring the map decode into `_read_map` left all 19
  pre-existing tests passing untouched, incl. the four real third-party sample files and the
  byte-for-byte default-output assertion. 8 new tests (1095 loom green); ftrace interop verified by
  rendering one asymmetric L-shaped volume diagonal vs 45°-about-Y (`scraps/vdb_rot_make.py` →
  `png/vdbrot/{flat,rot}.png`) — identical voxels, visibly rotated. loom `DESIGN.md`'s E4 entry was also
  stale from before the codec work and is now current.
- 2026-07-26: **0.77.0 — the loader reports unknown keys, and J3c's emitter-drift audit is closed with it.**
  The audit ("check every `Element.emit` against the live grammar") could not be done by reading code: ftrace
  silently ignored any key no builder read, so drift produces a *wrong image* rather than an error. Built the
  missing diagnostic at the one choke point — `Stmt` gained a `mutable bool used` set inside
  `find(const Block&, const char*)`, which every property read (`strOf`/`vec3Of`/`dblOf`/`spectrumParam`/…)
  funnels through, so it cost zero per-builder changes. ~17 sites that iterate `b.stmts` directly (repeated-key
  gathers, exhaustive dispatch loops, flat-word `data`/`palette`/`table` bodies) mark explicitly via
  `markUsed`/`markAllUsed` — each of those was *discovered* by the corpus sweep naming its own false positive,
  316 → 89 → 12 → 0. `collectUnusedKeys` runs at the end of `Builder::build` and reports via `Loaded::unknownKeys`
  (carried, not printed inline, because `prefer { } else { }` trial-builds candidates and discards all but one).
  A **warning**, not an error: an old scene with a stale property must still render, but it must say so.
  Also made `evalSpectrum` explain a `texture:` in a non-texture slot instead of "unrecognized spectrum
  expression". Then ran the audit (`scraps/emit_audit.py`, 11 Element kinds): **all clean**, plus all 78
  checked-in scenes and 1083 loom tests. Real finds fixed: loom's `Isosurface` couldn't emit
  `samples`/`accuracy`/`refine`/`uv` (a sampled march was pinned to the 256-sample default — 4 new tests,
  including the FTSL §2 bareword-axis trap); `priority` authored on geometry in `scenes/_record_scalar.ftsl`
  (it's a material slot); 6 dead `contained_by` lines on pure-analytic gallery isosurfaces. Docs: FTSL.md §1.3
  (new; `prefer` renumbered to §1.4), README diagnostics, design.md `ftsl.h`.
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
