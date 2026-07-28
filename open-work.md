# Open work — the short list

`TODO.md` is ~4000 lines and is now mostly a *record of what shipped*: entries there are
long prose blocks whose opening paragraph reads like a plan but whose later
`**STATUS (date) … DONE**` sub-paragraph says it landed. That makes "what's actually left?"
expensive to answer.

**This file is the actionable extract, as of 2026-07-28 (ftrace v0.89.0).** It carries only
work that is genuinely undone *and* not explicitly ruled out. `TODO.md` remains the
authoritative design text — every item below names its section/item ID there, and the full
rationale, prior art and scoping live in that entry, not here.

Keep the two in sync: when an item below lands, mark it DONE in **both** files (or delete it
here and record the DONE in `TODO.md`). When a new open item appears in `TODO.md`, add it here.

Excluded by explicit decision — see the bottom of this file, and don't re-litigate them
without asking.

---

## 1. Unblocked — nothing external is stopping these

### K1 remainder — user-supplied named RGB→spectral mapping  *(ftrace; small–medium)*
*TODO.md §K, item K1.*

All five **built-in** upsamplers have landed: Jakob-Hanika reflectance (the original default),
JH illuminant (v0.10.3), Smits 1999 (v0.45.0), the plain calibrated 3-box (v0.46.0), and
Meng 2015 smoothest-spectrum (v0.85.0). Selecting among them from a scene works.

What's left is the last clause of the original proposal: a **named user mapping** — a function
`(r, g, b) -> spectrum` registered in the spectral-envelope store and referenced by name, so a
scene can plug in its own upsampler rather than picking from the built-in set.

Scope: a registration path in the spectral-envelope store + a name lookup wired through
`evalSpectrum`'s `rgb`/`hsv`/`hsl` handlers (the same chokepoint the existing method tag goes
through); mirror in loom's spectrum grammar. Observable → README + VERSION bump.

### Array-literal formals + keyword rebind — `[0 1](a)` … `(a=u)`  *(ftrace; small)*
*TODO.md "DECISION — color-vector / array syntax", the increment-2 `**Deferred:**` clause and
the `ADDENDUM — call = sample` bullet.*

**The stated blocker is now stale.** Increment 2 (v0.73.0) shipped the array sample call
`[0 1](u)` but deferred the keyword form with the explicit reason *"formals don't exist yet —
it currently lexes as a call and would fail in the expression compiler."* Formals and
`formal=driver` rebinding both exist as of the §3.3 material-bundle work (v0.87.0) and the §3.2
per-property work (v0.89.0), and the v0.88.0 lexer change made spaced argument lists legal. So
the machinery this item was waiting on is built; what remains is pointing it at array literals.

Two halves:
- **Declaring a formal axis on a literal** — `reflect [0 1](a)` names the axis rather than
  spending it, so a *user* of the material can bind it.
- **Rebinding it** — `(a=u)` at the use site, read `formal=driver`. Positional `(u)` rebinds the
  sole/next axis; keyword targets a named one; multiple axes take one argument each
  (`(u=a, v=x)`); mix is positionals-first as usual.

Semantics `TODO.md` says to pin when building: whether a formal axis name is a **binding site**
(rebindable) or a **literal coordinate source** (fixed), and the error text when a bare
unsaturated array reaches the renderer.

Also noted there as needing no work: `NAME axistuple` (`ramp(u)`), because ftrace's expression
evaluator already reads `name(args)` as a call.

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

### F4 item 2 — re-tessellation when rotating into a parameter dimension  *(C++ viewer)*
*TODO.md §F4, "Still open (deferred)".*

The **loom half is done** (2026-07-24): the viewer↔loom live re-introspection channel —
`ViewerSession`/`serve_viewer` in `loom.viewer` plus a `python -m loom.viewer <scene.py>` CLI —
holds a resident `ViewerModel` and answers `introspect {clock, params}` with a *fresh* sidecar,
which is exactly what a frozen sidecar can't do. It also gained an `emit` command.

Still open: the **C++ viewer half** — consuming that channel to re-tessellate when the user
rotates into a parameter/extra dimension, via a latest-wins off-thread job queue (so a fast
drag doesn't queue up a backlog of stale tessellations).

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
