# Optimization log

The record of hot-path optimization passes, and the marker future passes resume from.

## How the marker works

**The commit that last touched this file is the marker.** Everything reachable from it
has been surveyed for optimization opportunities. To find what a new pass needs to look
at:

```bash
git log -1 --format=%h -- optimization-log.md     # -> marker commit M
git log --oneline M..HEAD                          # commits not yet surveyed
git diff M..HEAD --stat -- src/                    # the code delta to survey
```

Committing the pass's own record here moves the marker automatically — no hash to
hand-edit, nothing to forget. This replaces re-reading session logs or re-surveying the
whole source per pass.

## Method (same every pass)

1. **Baseline first.** Benchmark the *committed* binary before touching code: fixed
   deterministic workloads sized ~5–20 s, 5+ runs each, self-reported render-loop seconds
   (the last `[live]`/`[spp]` line) as the metric — it excludes scene build/load. Save a
   `.pfm` identity artifact per affected subsystem with the same binary
   (same-command reproducibility is bit-exact on both devices; proven on ml256 CPU+GPU).
2. **Optimize.** Only transformations that provably keep every computed value, RNG draw,
   and accumulation order identical — or whose difference is empirically checked (e.g.
   FMA contraction between host-baked and device-computed values).
3. **Verify.** Rebuild, re-render every identity artifact, `cmp` byte-for-byte against
   the baseline. All must be IDENTICAL before measuring.
4. **Measure.** Re-run the benchmark series with the new binary. On a machine with
   background load, compare **min across runs** (load noise is strictly one-sided);
   corroborate with medians when both sides ran clean.
5. **Record** the pass below and commit this file with the changes.

Benchmark harness + identity sweep for this pass: `scraps/optbench_run.sh`,
`scraps/ident_sweep.sh` (scraps/ is gitignored; the bench matrix is reproduced below so
the method survives a cleanup).

---

## Pass 2 — 2026-08-16 (v0.190.4)

**Marker at time of survey:** HEAD = `9041654` (0.190.3).
**Surveyed delta:** everything since the light-tree/SDS/radcache feature work landed —
`ff4fd93` (light tree: Conty–Kulla light BVH for backward NEE), `822f15f` (forward-mode
SDS mirror connectors), `d9e39a1`+`292b1b9`+`4b51f82`+`9041654` (radiance cache),
`a0dfb64` (emissive-shell auto-orient — load-time only, no hot path), `410462c`
(`-window-min` — UI only). Previous dedicated perf pass: `e5c88f3` (2026-08-02).

### Changes

**Light tree traversal** (`lighttree.h`, `lighttree_build.h`):
- `LightTreeNode` reshaped: `bmin[3]/bmax[3]` → `center[3]` + `r2` (bounding sphere),
  and `sinThetaO` baked next to `cosThetaO`. The traversal only ever *derived* those
  three quantities per call — `0.5*(bmin+bmax)`, `0.25*|bmax−bmin|²`,
  `ltSafeSqrt(1−cos²)` — so the builder now computes them once, with **exactly** those
  expressions (the bit-identity condition; both sides are host-compiled by MSVC, which
  does not contract FMAs at /fp:precise, and the GPU upload is a verbatim memcpy of the
  same struct — device identity confirmed empirically).
- `ltImportance` keeps the raw `dist2` from the distance test and reuses it for
  `invD = 1/sqrt(dist2+1e-300)` instead of re-summing the dot product.

**Forward SDS specular connectors** (`render.h`, `render_cuda.cu`, `scene.h`):
- `connectSpecularPlane` / `dConnectSpecularPlane`: the light-side `occluded()`
  (any-hit, early-out) now runs **before** `mirrorSeenAt` (closest-hit, full BVH
  traversal). Both must pass and neither draws RNG, so order is unobservable — but a
  vertex occluded from the mirror now skips the expensive query entirely.
- `connectSpecularSphereMirror` / device twin: per-root tests reordered
  cheapest-reject-first — `cam.project` + Ω-check, then chief-ray `reflectOffSphere` +
  shape, then eye-leg geometry, then ray differentials/Jacobian, and only then the two
  BVH queries (any-hit before closest-hit). All moved steps are pure and RNG-free.
- `camSpecularSplatAllVtxN` (both backends): `Scene::dielSphereIdx` /
  `mirrorSphereIdx` (built in `buildMirrorPlanes`, uploaded as `DScene::dielSph` /
  `mirrorSph`) replace the scan of *every* sphere's material per vertex (per λ for the
  dielectric loop). Ascending index order keeps film accumulation order — and images —
  bit-identical.

**Radiance cache reader** (`radcache.h`, `backward.h`):
- The reader marks first and reads second; with `-radcache-jitter` off (the default)
  the two keys are identical, so the call site now derives key+mix **once** and probes
  via the new `lookupBundleAt(k, mix, …)` / `findSlotMixed` instead of letting
  `lookupBundle` re-derive `levelOf` + `normalBucket` + three floors + splitmix per
  consult. Jittered lookups keep the full derivation (the key really differs there).
- Everything else in the cache was examined and left alone: `levelOf` already reads the
  exponent field via `ilogb` (no sqrt/log2), the bundle lookup was already one probe
  for all λ, the mark filter is a direct-mapped single-compare, and the update pass is
  budgeted + front-loaded by design. Nothing else profitable.

### Verification

11 identity artifacts re-rendered with the final binary and byte-compared against the
v0.190.3 baselines — **all IDENTICAL** (CPU *and* GPU): ml256 mode R 256 spp with tree
(both devices), ml256 `-no-lighttree` 16 spp (both devices), `_mirror_fwd` mode B
(CPU 2e6, GPU 2e7), `_mirror_sphere_fwd` (CPU 2e6, GPU 2e7), `_mirror_mats_fwd`
(CPU 2e6), cornell mode B 2e6 (dielectric-sphere loop), cornell mode R 1024 spp 200²
`-radcache` (cache table evolution + reader).

### Measured (self-reported render seconds; base = v0.190.3, after = v0.190.4)

| bench | workload | base min / med | after min / med | **time saved (min-based)** |
|---|---|---|---|---|
| lt_cpu | ml256 R 64 spp 256² cpu | 9.2 / 9.6 | 8.2 / 8.6 | **−10.9 %** |
| lt_gpu | ml256 R 1024 spp 256² gpu | 10.4 / 10.6 | 9.7 / 9.7 | **−6.7 %** |
| plane_cpu | `_mirror_fwd` B 5e6 cpu | 11.1 / 13.2 | 6.5 / 7.5 | **−41.4 %** |
| sphere_cpu | `_mirror_sphere_fwd` B 2e6 cpu | 11.1 / 11.8 | 8.5 / 10.1 | **−23.4 %** |
| mats_cpu | `_mirror_mats_fwd` B 4e6 cpu | 5.5 / 5.8 | 4.6 / 5.1 | **−16.4 %** |
| plane_gpu | `_mirror_fwd` B 2e8 gpu | 14.7 / 14.9 | 13.8 / 13.9 | **−6.1 %** |
| sphere_gpu | `_mirror_sphere_fwd` B 1e7 gpu | 9.7 / 10.0 | 9.7 / 9.7 | ~0 % |
| glass_cpu | cornell B 1e6 cpu (dielectric SDS) | 7.6 / 8.4 | 7.8 / 8.5 | ~0 % (guard: no regression) |
| rc_cpu | cornell R 512 spp 200² cpu `-radcache` | 13.6 / 13.6 | 13.3 / — | **−2.2 %** (guard passed) |

Notes: a background process (~1 of 12 cores, bursty) polluted the tail of the after
series and the rechecks — min-across-runs is the honest statistic (medians shown where
both sides ran clean; rc_cpu's after-median omitted as polluted, its minimum beat the
baseline minimum). glass_cpu was *expected* flat: cornell's dielectric connector was
not reordered (its λ→sphere→camera nesting is accumulation-order-load-bearing) and the
scene has only one sphere, so the index list saves nothing there — it exists as the
regression guard for the dielectric path, and it passed.

---

## Pass 1 — 2026-08-02 (v0.157.x, commit `e5c88f3`)

Predates this log. Occlusion any-hit fast path + pattern-VM CSE; mode W gyroid
3.68× CPU / 1.45× GPU. Recorded here so the pass numbering has its origin.
