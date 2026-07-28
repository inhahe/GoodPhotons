# Open work — the short list

`TODO.md` is ~4000 lines and is now mostly a *record of what shipped*: entries there are
long prose blocks whose opening paragraph reads like a plan but whose later
`**STATUS (date) … DONE**` sub-paragraph says it landed. That makes "what's actually left?"
expensive to answer.

**This file is the actionable extract, as of 2026-07-28 (ftrace v0.92.0).** It carries only
work that is genuinely undone *and* not explicitly ruled out. `TODO.md` remains the
authoritative design text — every item below names its section/item ID there, and the full
rationale, prior art and scoping live in that entry, not here.

Keep the two in sync: when an item below lands, mark it DONE in **both** files (or delete it
here and record the DONE in `TODO.md`). When a new open item appears in `TODO.md`, add it here.

Excluded by explicit decision — see the bottom of this file, and don't re-litigate them
without asking.

---

## 1. Unblocked — nothing external is stopping these

### ~~K1 remainder — user-supplied named RGB→spectral mapping~~  **DONE 2026-07-28 (v0.90.0)**
*TODO.md §K, item K1 — now closed.*

Shipped as `upsample "<name>" { expr "f(r, g, b, w)" }`, named by the colon head
`rgb:<name> r g b` (also `hsv:`/`hsl:`). The body is a pattern-VM expression over a vocabulary
disjoint from the surface one (`r` is RED here, not radius — surface names are rejected by
name), plus `spec:<spectrum>(w)`, which is what makes a *measured basis* expressible rather
than only closed-form arithmetic. Pinned by `-checkupsample` section (h); scene
`scenes/_upsample.ftsl`; loom twins `NamedSpectrum`/`Upsample`/`UserSpec`/`is_colour_space`.

### ~~Array-literal formals + keyword rebind — `[0 1](a)` … `(a=u)`~~  **DONE 2026-07-28 (v0.91.0)**
*TODO.md "DECISION — color-vector / array syntax" — now closed; see the increment-2 remainder
STATUS there.*

The v0.73.0 deferral was stale in both directions. The feature itself had already arrived with
the §3.3 material bundles (v0.87.0) and §3.2 per-property access (v0.89.0) — `[0 1](a)` compiled
to a program with a free `a`, and `mat(a=u)` / `src.reflect(a=u)` / `src.reflect(u)` rebound it —
but nothing pinned it, and the one genuinely open spelling was still a raw lexer error.

Pinned semantics: a literal's axes are anonymous and bind by **position**, so its "formals" are
just the driver names in its own tuple (which is what a rebind substitutes; hence the simultaneous
2-D swap `(u=v, v=u)` transposes). `formal=driver` *inside* a literal's own call is refused with
both working spellings named, generated `__arrN` blocks are re-attributed to the authoring site,
the unsaturated message now names the `(a)` deferral route, loom's `values.py` refuses the same
spelling, and `-checkarray` + `scenes/_array_formal.ftsl` pin the identities (with explicit
non-vacuity checks).

### Composing array literals — `[0 1]([0.2 0.8](u))`  *(ftrace; small)*
*Discovered 2026-07-28 while closing the item above. TODO.md's grammar sketch has
`coord = NAME | NUMBER | value`, i.e. "a nested `sampled` gives composition (`n(m(u), v)`)".*

ftrace can't do it: `PARENWORD = /\([^ \t\r\n{}\[\]#"]*\)/` excludes `[` and `]`, so the inner
literal's brackets split the token and the outer literal reports the (misleading) *unsaturated*
error. loom's value grammar already normalizes the nested form. Workaround today is to name the
inner table (`grid` + `pattern`) and reference it from the outer call's coordinate expression.

### `NAME axistuple` at a value site — `reflect grid:ramp(u)`  *(ftrace; small)*
*Discovered 2026-07-28. Corrects a wrong claim in TODO.md's increment-2 `Deferred:` clause.*

That clause says `NAME axistuple` "needs no work because ftrace's expression evaluator already
reads `name(args)` as a call". True *inside* a pattern expression — but a **value site** is not an
expression site, and `reflect grid:ramp(u)` is "unrecognized spectrum expression". The form works
only where the name is a material or material property (`gold(u=v)`, `src.reflect(u=v)`), because
those go through `applyMaterial`. Closing the gap means hooking the same four chokepoints v0.89.0
used for `MATERIAL.slot(args)` (`evalSpectrum`, `patternedSpectrumParam`, `dblParam`,
`bindScalarPattern`); a bare `ramp(u)` must keep meaning a material application, so the scoped
`grid:` / `scatter:` spelling is the unambiguous one to accept. Workaround: declare the one-line
`pattern` wrapper by hand, which is exactly what an array literal desugars to.

---

## 2. Blocked on a user decision, not on code

### D1 / D2 / D3 — hero-room showcase renders  *(ftrace)*
*TODO.md §D.*

Gated on **your sign-off**, by design: D2/D3 don't proceed until you've personally verified — in
the interactive rasterizer flyby-definition tool (the camera_curve editor) — that you like the
room composition and the camera path. That's a human aesthetic call, and the point of the gate
is not to burn expensive photon-map renders on a room/flyby that hasn't been approved.

- **D1** *(in progress)* — flyby photon-map render: GPU shared photon-map path (build once,
  gather all 144 frames), `-savemap gallery/hero_map.ftpmap`.
- **D2** — verify still: raster + one real photon-mapped frame; confirm all pieces read.
- **D3** — verify flyby: render frames + assemble; confirm the gyroid thread, the glass pass and
  a seamless loop.

Once you say "I like it", D2/D3 unblock.

### E2 slice 3 — generalize the C++ `camera_curve` editor  *(ftrace; interactive)*
*TODO.md §E2.*

Slices 1–2 landed 2026-07-24 (config model + the live-value channel: `collect_slots`,
`SceneDriver`, `LiveSession`/`serve_live`, 23 tests). Slice 3 is the C++ half — make the editor
seed from and write back the sidecar, and target **arbitrary scene variables** rather than just
the camera curve.

`TODO.md` flags this as *"best done with the user present"* — it's an interactive UI whose feel
can't be validated headlessly. Not blocked technically; blocked on being worth doing together.

### E4 — Vec3 volume grids  *(loom)*
*TODO.md §E4, "Still open".*

Blocked on **validation data**: none of the four real OpenVDB sample files on hand carry a Vec3
grid, so there's nothing to check a decoder against. Compounding it, there's **no downstream
consumer** — ftrace supports scalar float grids only, so a Vec3 read would serve loom-internal
use alone. Everything else in E4 shipped (write side, read codecs incl. half/ZIP/blosc, rotated
`AffineMap`, NanoVDB `.nvdb` ingest, and the read→transform→write capability the item was
actually about).

Also listed there as still open: sparse *storage* (sparse-source *reads* already work).

---

## 3. Open but with no current driver

### J3c second half — `.ftsl` → loom Element tree  *(loom; large)*
*TODO.md §J3c.*

The **emitter-reconciliation** half shipped: the unused-key warning
(`Stmt::used` → `collectUnusedKeys` → an `[ftsl] warning` from `loadSource`) turned a silent
drift into a loud one, and the audit came back clean on all 11 element kinds and all 78
checked-in scenes, fixing real drift on the way (`Isosurface` couldn't emit
`samples`/`accuracy`/`refine`/`uv`; a misplaced `priority`; 6 dead `contained_by` lines).

Still open: the reader direction — parsing a whole `.ftsl` back into a loom Element tree so a
scene round-trips semantically.

### FUTURE — loom full `.ftsl` read support  *(loom; large)*
*TODO.md, the `FUTURE` bullet under §J3c.*

The bigger version of the above: a complete `.ftsl` → `Scene` reader (not just per-element
round-trip) — the whole-file `scene { … }` wrapper rule plus a `Scene` builder that reassembles
textures/patterns/records/materials/geometry/lights/camera into a live `Scene`, plus the lossy
cases (`mesh { file … }` → a lightweight `MeshRef` that re-emits the same block; `medium`,
`pattern`, `camera_curve`).

Explicitly parked: *"the grammar's real job is ftrace's parser — so this waits until a concrete
editor need exists."* The motivating consumer would be an editor/GUI (load an existing `.ftsl`,
manipulate in loom's object model, re-emit).

### ~~F4 item 2 — re-tessellation when rotating into a parameter dimension~~  **DONE 2026-07-28 (v0.92.0)**
*TODO.md §F4 item (2) — now closed, which closes §F4 entirely.*

The loom half had been done since 2026-07-24 (`ViewerSession`/`serve_viewer` + the
`python -m loom.viewer <scene.py>` CLI). The C++ half shipped as `-loom <scene.py>` alongside
`-viewer`: `LoomLink` holds the child process, `LoomBridge` is one worker thread with a
**one-slot** pending job so `post()` overwrites anything unstarted (latest-wins — a fast drag
costs one bake of the final value, and the UI never blocks). A **Live (loom)** panel exposes
the clock, one typed control per declared keyword param, `auto` / `re-derive now`, and
`posted / baked` counters; marking a continuous param the **sweep axis** makes a right-drag on
any 3-D pane rotate into that dimension. Each bake returns a sidecar *and* an `.ftsl`, so
Curves/Fields/Meshes and F7's Render pane all refresh, preserving orbit/zoom/tab/DAG layout.

Two real bugs surfaced during validation and were fixed (written up in `known-issues.md`): the
imgui #7543 `EndGroup()` change corrupting imnodes node rects into a multi-gigabyte
`PrimReserve`, and the DAG pane re-packing itself from degenerate measurements when clipped to
zero height.

---

## Excluded — explicitly ruled out, listed so they aren't rediscovered

Full reasoning in `TODO.md` at each item. Don't pick these up without asking.

| Item | Status |
|---|---|
| **C9** Alembic (`.abc`) import | **DON'T DO FOR NOW** (user, 2026-07-24). Heavy SDK; OBJ/glTF/FBX/STL/PLY suffice. |
| **J3b item 4** N-D record *input* domain | **NOT SCHEDULED** (user, 2026-07-25). Also entangled with the axis-labelled-array work — doing it first would build a competing spelling. |
| **G3** `PatOp::MatMulAdd` intrinsic | Skip for now (2026-07-18). Deferred on **Amdahl**, not opcode mix: field eval is only 7–12% of end-to-end. Revisit past ~800 pattern nodes. |
| **G4** GPU marching cubes | Deferred. Measured 2026-07-27: a res-160 export was 55% *ASCII OBJ write*, not marching — so a free GPU march capped at 1.8×. G4b fixed the writer instead (2.5×). Revisit only for repeated/batch export or res ≥ 384. |
| **F7** `-serve` streaming path | Only needed if the raymarch is ever pushed to a separate process. |
| Mode-R / hero wavefront scheduler | Hero forces the megakernel there; single-λ by design. |
| GPU SPPM / GPU photon paths | No current demand. |
