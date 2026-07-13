# Forward Raytracer — project instructions

## File / directory conventions

Keep the repo root clean. All generated and throwaway files live in dedicated
directories (already created, and git-ignored):

- **`ppm/`** — all `.ppm` render outputs and test images. Never write `.ppm` files to
  the repo root; pass `-o ppm/<name>.ppm`.
- **`png/`** — all `.png` render outputs and test images. Pass `-o png/<name>.png`.
- **`scraps/`** — all temporary / throwaway Python scripts (comparison helpers,
  one-off analysis, etc.) and any other scratch files. Don't leave temp `.py` files
  in the repo root or in `tools/` (which is for permanent, checked-in tooling).

When creating validation renders or scratch scripts, always target these directories
so the working tree stays tidy and nothing stray shows up in `git status`.

## Running renders — ALWAYS give the user a way to watch progress

Never launch a render as a fire-and-forget black box. Any render you start must
provide periodic, checkable output so both you and the user can see it converging
and confirm it will actually finish. Concretely, when you start a render:

- **Prefer the live image window: pass `-window`.** This opens the real OS
  live-preview (Win32 GDI) showing the actual tone-mapped image refreshed as it
  converges — the best way to watch. It also auto-chunks a plain fixed-`-n` render
  so periodic writes/status happen (see the gotcha below).
- **Always add periodic crash-safe output.** Use `-interval <sec>` (default 15) for
  the write/refresh cadence, and `-checkpoint` (forward modes A/B/C) so a resumable
  `.ftbuf` sidecar is written next to `-o` and progress survives a crash/Ctrl-C
  (`-resume` continues it). Without `-checkpoint` a long render that dies loses
  everything.
- **No GUI? Use `-preview`** for a live ANSI thumbnail in the terminal, or at least
  rely on the periodic status line (`[live] … photons, ~N% noise`).
- **Prefer a bounded budget over a giant `-n`.** `-time <sec>`, `-noise <pct>`
  (stop at a target graininess), or `-forever` (Ctrl-C to stop) all render
  *progressively* with periodic writes, so you get a usable image early instead of
  waiting for one huge batch. Pick a photon count that's plausibly completable and
  sanity-check the rate from the first status tick rather than guessing.
- **Run it in the background and check the periodic output**, so you can report
  progress and catch a stall or a runaway count early.

**Critical gotcha (the reason for this rule):** a plain fixed-`-n` render with
*none* of `-window` / `-time` / `-noise` / `-forever` takes the **non-chunked path**
(`chunkFixed = !progressive && g_showWindow`, main.cpp ~2125). It runs the *entire*
photon count as one monolithic batch and writes the image **only at the very end** —
no partial image, no checkpoint, no status line, no way to see progress or estimate
completion. So a bare `-n 4000000000` can run for hours producing nothing and
losing it all on a crash. Adding `-window` (or any budget flag) is what enables the
periodic-output loop.

## Docs to keep current

- **`README.md`** is user-facing — update it whenever an observable feature changes
  (render modes, camera models, materials/lights/spectra presets, CLI flags, GPU
  backend scope). Mode descriptions in particular must match the actual implementation
  (e.g. mode A is the finite-lens physical camera, not the old contact-sensor wall).
- **`known-issues.md`** tracks unsolved bugs and tech debt — log anything you can't fix
  immediately, and mark entries DONE when resolved.
