# Known Issues & Technical Debt

Running log of unsolved bugs and accumulated tech debt. Fix items here as soon
as practical; this file is the fallback for what can't be addressed immediately.

## Performance

### Diffuse-mesh renders are ~60× slower per photon than analytic scenes
- **Symptom:** The Cornell + 25.6k-triangle diffuse torus (`-mesh torus.obj`)
  traces at ~33–67 µs/photon, vs ~0.55 µs/photon for the Cornell + glass sphere.
  8M photons took 4m26s; 3M photons > 3min. Makes mesh scenes impractical.
- **Where:** `Renderer::tracePhoton` (`src/render.h`) + BVH traversal
  (`src/bvh.h`) + `Scene::closestHit`/`occluded` (`src/scene.h`).
- **Correctness is fine:** `-checkbvh` reports 0 mismatches on the torus, so the
  BVH returns correct hits; the problem is purely cost.
- **Suspected causes (to confirm by measurement):**
  1. **BVH not pruning well / poor traversal.** Traversal in `traverseClosest`
     pushes both children unordered and prunes only via `tMax`; no front-to-back
     ordering, so far subtrees get visited before a near hit tightens `tMax`.
     Need to measure avg nodes/leaves visited per ray — if it's in the hundreds
     for a 25k-tri BVH, the tree or traversal is the culprit.
  2. **No Russian roulette.** Diffuse albedo 0.8 → beta needs ~62 bounces to hit
     the 1e-6 cutoff, but `maxBounce=32` caps paths. Inside the torus's concavity
     photons take many bounces, each doing 2 BVH traversals (closestHit +
     connect's occluded shadow ray). RR would terminate low-energy paths early
     and is unbiased.
  3. **Every diffuse bounce casts a shadow ray to the camera** (`connect`), so
     the traversal count per photon is ~2× bounce count.
- **Proper fix (planned, next):**
  - Add front-to-back child ordering in `traverseClosest` (descend nearer child
    first; push farther child, prune with updated `tMax`).
  - Add Russian roulette after a few bounces (terminate with prob 1-beta,
    rescale survivors by 1/p) — unbiased, caps path length naturally.
  - Instrument nodes-visited-per-ray to confirm the BVH quality before/after.
  - Re-measure µs/photon on the torus scene; target within ~5–10× of the sphere.
- **Status:** OPEN. Logged 2026-07-10.
