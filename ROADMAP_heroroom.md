# Hero Room / Showcase scene — build roadmap

The showcase **still** *and* **flyby** scene (`scenes/gallery.ftsl` → settled →
`scenes/gallery_settled.ftsl`). Check items off (`[x]`) as they land. Source of truth
for requirements: `heroroom.txt`.

## Asset audit (done first, per "tell me if you can't find any element")
- [x] `klein_staged.obj` — exists (`scraps/klein_staged.obj`).
- [x] Brass CSG cluster / algebraic heart / brass dumbbell-ring — inline defs exist in the old `gallery.ftsl`; reused.
- [x] Golden gyroid — the existing k=10 lattice def; reused.
- [x] Glass sphere — native sphere.
- [x] Mirror material — `preset chrome` (Cr).
- [x] Iridescent materials — `morpho`, `oil-slick`, `beetle`, `nacre`, `soap-bubble`, `anodized-ti`.
- [x] Arc-lamp spectrum — **no literal xenon CSV**; project's own conclusion (`data/README.md`)
      is that `blackbody 6000` (`preset:bb6000`) is a faithful visible-range xenon short-arc
      model (verified vs Osram XBO / Ushio UXL). Using `spd blackbody 6000`.
- [x] **MISSING: `gyref_gyroid_adaptive` mesh** — does not exist anywhere (only stale
      rendered PNGs in `png/gyref/`). Substituting a **native isosurface gyroid** (cleaner
      refraction than a mesh) in clear high-IOR `diamond` for real coloured caustics.
- [x] `Xenon_short_arc_1.jpg` — present, reference only (user may Meshy a 3D bulb later).

## Room
- [x] Rebuild **wider than long** (10 m wide x 7 m deep x 3.4 m tall).
- [x] Keep every object in a **central x band** (x≈3.5–6.2) so the pinhole lens doesn't
      elongate anything near the side walls.
- [x] Closed box: floor / ceiling / back(teal) + two coloured side walls (terracotta/sage)
      + front wall behind camera.

## Lighting
- [x] **One** small bulb only — no area light. A single **sphere light** r=0.05 in a socket.
- [x] Give it the **xenon-arc spectrum** (`spd blackbody 6000`).

## Objects (each on a museum stand; all pybullet-settled EXCEPT the cloud)
- [x] Gold gyroid — the existing one; **close-up, plain view**, front-centre (NOT settled: threaded).
- [x] Clear high-IOR gyroid (gyref substitute, `diamond`) — throws **coloured caustics** (NOT settled: lattice).
- [x] Glass sphere — the fly-through orb (NOT settled: analytic rest).
- [x] Klein bottle (`klein_staged.obj`) — **clear glass** (settled).
- [x] Algebraic heart — **tilted**, iridescent (`morpho`) (settled).
- [x] Brass CSG cluster (settled).
- [x] Brass dumbbell-ring (settled).
- [x] At least one **mirrored** object (chrome ring, `preset chrome`).
- [x] A **second iridescent** object — `oil-slick` thin-film (different model/colour than the morpho heart).
- [x] **Museum stands**: 9 isosurface stands, each with varied part dimensions AND material/colour
      (marble/granite/walnut/slate/sandstone/basalt/travertine/socketmat).

## Cloud
- [x] Isosurface-fog cloud (3-lobe noise medium), hanging near the ceiling in the
      **back-left** of the room.

## Flyby (30 fps, smooth, self-closing)
- [x] Camera path is a **closed loop** (centripetal Catmull-Rom, `closed`); the closure joint
      is as smooth as the rest (approach point moved outside the clip sphere).
- [x] Path **threads the gold gyroid** channel — validated by dense spline sampling (0 strut
      hits, worst |g|=0.550 > 0.45) and raster fly-through frames.
- [x] Path **passes through the glass sphere** (control point 6.20 1.40 3.50, dead-centre).
- [x] Enough frames for **30 fps, a few seconds** (144 frames, smooth tangent-look motion).
- [~] Use **photon mapping / stored map** — GPU shared photon-map path (build once, gather all
      144 frames), `-savemap gallery/hero_map.ftpmap`. (Render in progress.)

## Verification
- [ ] Still: raster + a real photon-mapped render frame — confirm all pieces read.
- [ ] Flyby: render frames + assemble; confirm the gyroid thread + glass pass + seamless loop.
