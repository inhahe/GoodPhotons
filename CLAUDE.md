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

## Docs to keep current

- **`README.md`** is user-facing — update it whenever an observable feature changes
  (render modes, camera models, materials/lights/spectra presets, CLI flags, GPU
  backend scope). Mode descriptions in particular must match the actual implementation
  (e.g. mode A is the finite-lens physical camera, not the old contact-sensor wall).
- **`known-issues.md`** tracks unsolved bugs and tech debt — log anything you can't fix
  immediately, and mark entries DONE when resolved.
