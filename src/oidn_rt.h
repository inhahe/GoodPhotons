// Intel Open Image Denoise, loaded at RUN TIME (`-oidn`).
//
// ftrace.exe is released as one standalone file, and OIDN is ~70 MB of DLLs (its core plus a
// device module per GPU vendor), so it is not linked. OpenImageDenoise.dll is loaded the first
// time a denoise is asked for and the handful of C entry points used are looked up by name: a
// machine without it renders exactly as before, and only `-oidn` fails there, naming where it
// looked. The search order is in oidn_rt.cpp (searchPaths).
//
// Kept in its own translation unit so <windows.h> stays out of main.cpp.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace oidnrt {

// Values are OIDN's own OIDNDeviceType / OIDNQuality, so they pass straight through.
enum class Device  { Default = 0, Cpu = 1, Cuda = 3 };
enum class Quality { Fast = 4, Balanced = 5, High = 6 };

struct Options {
    Device  device  = Device::Default;   // Default: OIDN picks (the GPU when it supports one)
    Quality quality = Quality::High;
};

// Denoise one image with OIDN's "RT" filter, in place.
//   color    W*H RGB float triples, scene-linear HDR (any scale: OIDN measures it), overwritten
//            with the result. Row order is the caller's, as long as all three images share it.
//   albedo   the guide albedo in [0,1], same layout, or nullptr for none.
//   normal   the guide shading normal in [-1,1], same layout, or nullptr. OIDN accepts a normal
//            only alongside an albedo, so a normal without an albedo is ignored.
//   cleanAux the guides are noise-free (OIDN then trusts their edges fully; set it only when
//            they are deterministic, or their noise is printed into the result).
//   auxKey   identifies this guide set (nonzero, unique to it): the guides are uploaded only
//            when it differs from the last call's, since a camera's guides do not change
//            during its render. 0 uploads them every call.
// Thread-safe: calls are serialised, and the device, filter and buffers persist between
// calls, so after the first one a frame costs only the copies and the filter itself.
// The device's selection (`opt.device`) is fixed by the first call.
// Returns false with `err` set if the library cannot be found or loaded or the filter fails;
// a failed load is remembered, so later calls fail at once with the same message.
bool denoise(float* color, const float* albedo, const float* normal, int W, int H,
             bool cleanAux, uint64_t auxKey, const Options& opt, std::string& err);

// One line for the log naming the library loaded and the device it runs on, or the reason
// it did not load; "" until the first denoise() call.
std::string describe();

// Device memory OIDN may be holding on a GPU (its requested cap plus slack): this process's,
// for a preflight that must not blame it on another program. 0 on the CPU or when not loaded.
size_t deviceMemoryAllowance();

// Release the filter, its buffers and the device. Call once, before the process's CUDA
// teardown, so OIDN's GPU context is not left for the driver to reap after exit. Returns true
// if the device was a GPU's (the caller should then tear CUDA down even if it never used it
// itself). Later denoise() calls fail.
bool shutdown();

}  // namespace oidnrt
