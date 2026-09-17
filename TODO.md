# TODO — in flight

Live working plan. Each item says what, why, how it gets validated, and where it stands, so an
interruption costs the work in progress and not the plan. Finished items keep one line here and
their measurements in `known-issues.md`; `design.md` gets the architecture.

**Read the "Recently finished" list before starting anything.** On 2026-09-17 I reported VOLCACHE's
deposit split, GPU-VARIANCE and GPU-BEAM-TAIL as outstanding. **All three were already closed** —
I had carried them forward from a stale task list without re-reading their entries. Checking status
costs one `grep`; chasing finished work costs a session.

---

## 1. The `gallery_rain` 960x540 flyby — THE DELIVERABLE, and it has never been launched

1147 frames, `camera_curve "fly"`, mode M. Everything below it (`-sunnee`, VOLCACHE, the spectral
fixes, the beam work) exists to make this frame cost and this image quality possible. The single
frame currently renders in ~96 s.

**Three known blockers, all already logged — read them before launching, not after:**

- **The machine's COMMIT limit, not its RAM, is the ceiling** (`known-issues.md`, "the `gallery_rain`
  600-frame mode-M flyby cannot start"). The verified showcase command died in the film allocation.
  That entry has the exact command and the diagnosis; the flyby is now *1147* frames rather than
  600, so the allocation is larger, not smaller.
- **A mode-M GPU gather can die with `unspecified launch failure` under concurrent GPU load — and
  the batch carries on as if it had not** (logged 2026-09-16). On a 1147-frame run that silently
  produces a hole in the sequence. Decide how the run detects this *before* starting it.
- **There is no `-frames A B`**, so a run that dies cannot be resumed at the frame it died on
  (item 7).

**Before launching:** render a handful of scattered frames (`-camera fly0000`, `fly0400`, `fly0555`,
`fly1146`) at final settings and look at them. Cheap insurance against discovering a framing or
exposure problem 900 frames in.

## 2. glTF per-texel metalness -> a `mixWeightTex`-driven mix — the likely remaining Meshy gap

glTF's metalness is **per texel**; ftrace's material type is **per material**. Alice's
metallicRoughness map has mean metalness 0.28 but **p90 0.53** — parts of that one material are
properly metallic, and the importer, typing the whole thing by the mean, renders them as a 4 %
dielectric. A 4 % coat is genuinely subtle; a metal is not.

**The fix is already scoped:** import a metalness-mapped material as a **two-child mix driven by the
map** — `Material::mixWeightTex` exists and does exactly this (a per-hit blend mask on a 2-child
mix) — with a metal `glossy` child and the `layered` dielectric child. Not built.

This is the most likely remaining difference against the viewer the user compares to, and the user
has asked about Alice's dress looking glossy more than once. Full write-up in `known-issues.md`.

## 3. An emissive mesh that is not PLANAR is silently re-oriented outward (logged 2026-09-17)

So an emissive **enclosure** — a furnace, a cove, the inside of a softbox or a lampshade — renders
black, with no diagnostic. Found while building the coat validation rig, where it cost two wrong
measurements before it was understood.

Cause: an emissive mesh's signed volume about its centroid is measured and, if negative past
`-1e-6 * area^1.5`, every triangle's winding is reversed so emission points outward. The intent is
right (an inward-wound import like `torus.obj` would otherwise glow into its own hollow) but an
enclosure is indistinguishable from that case. **The real rule is planarity, not closure**: any
emissive mesh with triangles in more than one plane is at risk.

**Fix:** an explicit opt-out on the mesh block — `emit_orient keep` alongside the current `auto` —
**not** a cleverer heuristic, because no geometric test can tell a lampshade interior from a torus
wound the wrong way. The author knows which they meant; the loader cannot.
**Workaround meanwhile:** one `mesh` block per planar face (what `tools/furnace_rig.py` does).

## 4. Mode M's CPU and GPU gathers disagree by ~9 % on `gallery_rain`'s floor grid (2026-09-16)

Open, on the surface photon-map gather. It matters *because of item 1*: the flyby is mode M, and a
9 % backend disagreement on a large visible surface means one of the two is wrong in the frames
being shipped. Full entry in `known-issues.md`.

## 5. A3 / the explicit multi-bounce layered BSDF — trigger FIRED, design named, not built

The analytic coated body is built and validated (A1 0.323.0, A2 0.324.0). What is missing is a
**directional** body under a coat, and it is no longer a judgement call — the deferred trigger's
condition 2 has fired with numbers:

- Up to **-37 % in directional albedo and -34 % in lobe width** on a glossy body under a smooth
  coat, against the "a few percent" bar the trigger itself set. `tools/a3_snell.py`.
- Both controls pass, so that number means what it says: no coat -> 0.13 %, Lambertian body ->
  0.23 % (which independently reconfirms A1's derivation).
- **No cheap fudge exists**: the lobe-width error *changes sign* with roughness, so no roughness
  rescale fixes both ends.

**An exact result worth keeping in mind:** Snell in-and-out of a *smooth* body is the **identity**
(`sin t_out = n sin t_t = sin t_i`). So this is not "the coat bends the light" — it is only about the
body's MICROFACET normal disagreeing with the coat's, which exists only for a rough body, which is
exactly where TIR traps part of the lobe and no closed form survives. The two are inseparable.

**Why not built:** `f(wi,wo)` and `pdf(wo|wi)` have no closed form; NEE needs the first at every
shading point and BDPT/VCM need both at every vertex. That is the stochastic-evaluation BSDF
(Guo/Hasan/Zhao 2018) the trigger already names — real architecture, with an MIS decision to make up
front rather than halfway through.

**Next, if picked up:** build the brute-force reference IN THE RENDERER (mode R or A/B/C with the
coat traced explicitly, which needs no MIS pdf), in an enclosure, and confirm the Python numbers end
to end before committing to the architecture.

## 6. The heterogeneous spectral-media tier has no end-to-end number

The three-tier spectral media fix (C, v0.322.0) is validated for the flat and homogeneous tiers
(coloured fog: 113 % channel spread -> 1.0 %). The **coloured-AND-heterogeneous** combination is not
covered: those renders exceeded 20-40 minutes and were stopped.

**Do not re-derive the wrong conclusion from that.** I twice called the stochastic tier
"impractically slow" and was wrong both times — 24 bins vs 8 changed nothing, narrowing the spectral
spread changed nothing, and the control I should have run first settled it: the same scene with a
FLAT spectrum, taking the scalar fast path (i.e. pre-fix behaviour exactly), is **equally slow**.
The cost is that scene's heterogeneous medium, not the spectral vector. What is missing is a **cheap
scene** that exercises the tier, not a redesign.

## 7. No CLI flag renders a RANGE of a `camera_curve` (logged 2026-09-02, partly stale)

**Correction found 2026-09-17:** a single frame *can* be selected — the curve's frames are ordinary
named cameras, so `-camera fly0555` works. (`-frame N`, `-camera fly#555` and `-res` do not exist;
the flag is `-r W H`.) The entry should be narrowed to what is genuinely missing: a **range**
(`-frames A B`), which is what would let a stopped flyby resume at the frame it died on — see item 1.

## 8. BLOCKED: the paired-timing rule into `CLAUDE.md`

"Time paired within a repetition, warm-up discarded" is a standing measurement rule that lives only
in session context. It belongs in `CLAUDE.md` — which currently carries the user's own uncommitted
changes and **is not to be touched**. Do this only if that tree becomes clean, and ask first.

## 9. Characterised, no action decided: mode D's heavy noise tail

Mode D's per-frame noise falls as roughly `spp^-0.15`. The firefly framing was largely retired
(2026-09-12) — both modes peak at the same pixel at every seed — leaving a 4 %-energy, 1669x-peaked
**connection** residual that is shared BDPT machinery rather than anything mode-specific. Recorded
in `known-issues.md`; no fix proposed, and it is not blocking anything.

---

## Recently finished (one line each; measurements in `known-issues.md`, architecture in `design.md`)

- **A1 — the coat's exit interface (v0.323.0).** A coated body's albedo becomes
  `a(1-F_dr)/(1-a F_dr)`, so a coated colour deepens like varnish instead of being washed out.
  White body 0.9998 vs 1.0000 predicted; grey 0.5 -> 0.3159 vs 0.3159. `tools/furnace_rig.py`.
- **A2 — absorption in the coat layer (v0.324.0).** New FTSL `coat { absorb <spd> depth <m> }`;
  Beer-Lambert in, out, and on every internal round trip. Measured 0.3475 vs 0.3476 analytic.
  Also fixed a host/device disagreement shipped in 0.323.0 (glossy body under a coat, ~1.6x) by
  giving the coat exactly one funnel.
- **B — mode M mis-coloured coloured speculars (v0.321.0).**
- **C — mode M mis-rendered coloured media (v0.322.0).** 113 % channel spread -> 1.0 %.
- **VOLCACHE deposit split — BUILT AND MEASURED (v0.300.0)**, 64 % faster and energy-correct.
  *Not outstanding.*
- **GPU-VARIANCE — ROOT CAUSE FOUND AND PROVEN (v0.300.2):** the two backends average a different
  number of light-side realizations at the same spp. Not a noisier deposit. *Not outstanding.*
- **GPU-BEAM-TAIL — CLOSED (2026-09-14)** as a narrow edge case: needs `-beams` *and* `sigma_t`
  between 6 and 20, one scene in the repo. *Not outstanding.*

---

## Standing constraints (so a fresh session does not have to be told)

- Bump `VERSION` in the same commit as any observable change; no rebuild → no bump. Commit freely,
  **never `git push`**.
- Every render passes `-window-min` (never bare `-window`) and launches with
  `dangerouslyDisableSandbox`. Stop a render with `ftrace -stop <pid>`, **never** `taskkill /F`.
  Never blanket-kill a shared runtime by image name (python/node/dotnet/java) — only the exact PID.
- Scratch in `scraps/`, PNGs in `png/`; a flyby series gets its own `png/<setname>/`.
- **A rig that is cited as evidence does not belong in git-ignored `scraps/`** — promote it to
  `tools/` and rewrite the references in the same commit, or the table is not re-derivable from a
  clean clone.
- `CLAUDE.md` is the user's and is not to be touched.

## Measurement discipline (each of these cost something to learn)

- **Run the controls first, and let them veto the result.** A rig that cannot see the effect
  produces confident numbers that are pure artefact: a furnace built from `light area {}` renders a
  MIRROR as exactly 0.0000, and I nearly reported that blindness as a 4 % energy bug in a material.
  `tools/furnace_rig.py` prints its three known answers before it prints anything else.
- **Never measure reflection in an open scene** — use an enclosure. This is what made the
  "mode M renders glossy 2.7x darker" claim wrong, and it had the same tell (a mirror reading 0).
- **Run the baseline control before attributing a difference**, especially before concluding that
  something new is slow (see item 6).
- Score per-ROI, never whole-frame; separate variance from systematic; hold confounds fixed by
  construction; time paired within a repetition with the warm-up discarded.
- When a quantity must be applied at N sites, **N > 1 is the bug** — build the funnel. The 0.323.0
  host/device slip was a four-site change made at three sites.
