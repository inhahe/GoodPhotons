// The linkage of the host-and-device helpers the GPU units reach through shared headers (ggx.h,
// bluenoise.h, pov_noise.h, pov_functions.h, pattern.h, stochtile.h). A register-capped GPU unit
// (FT_FAMILY_TU: render_cuda_bdpt.cu, render_cuda_jsurf.cu) must own every function it keeps out of
// line, or the device link may bind its calls to another unit's copy, held to another cap or to
// none (render_cuda_dev.cuh; tools/check_device_linkage.py): there they are `static`. Everywhere
// else they stay plain `inline`, one entity across the program, so that an external inline
// function calling one means the same function in every translation unit.
#pragma once

#ifdef FT_FAMILY_TU
#define FT_UNIT_INLINE static inline
#else
#define FT_UNIT_INLINE inline
#endif
