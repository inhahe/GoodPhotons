// The linkage of the host-and-device helpers the GPU units reach through shared headers (ggx.h,
// bluenoise.h, pov_noise.h, pov_functions.h, pattern.h, stochtile.h, worley.h, gabor.h,
// lighttree.h, gatherplane.h, photonmap.h, lens.h, curvature.h, cell_walk.h, rng.h). A
// register-capped GPU unit (FT_FAMILY_TU: render_cuda_bdpt.cu, render_cuda_jsurf.cu) must own every
// function it keeps out of line, or the device link may bind its calls to another unit's copy, held
// to another cap or to none (render_cuda_dev.cuh; tools/check_device_linkage.py): there they are
// `static`. Everywhere else they stay plain `inline`, one entity across the program. The key is the
// unit, the same in its host and device passes: under separate compilation a definition with
// external linkage must not be present in one pass and absent in the other, so `__CUDA_ARCH__`
// cannot choose it. A capped unit's host pass, which holds its kernels' stubs, then has its own
// `static` copies beside the shared headers' external inline host functions that call them --
// benign, the bodies being one. A new header of such helpers takes FT_UNIT_INLINE too.
#pragma once

#ifdef FT_FAMILY_TU
#define FT_UNIT_INLINE static inline
#else
#define FT_UNIT_INLINE inline
#endif
