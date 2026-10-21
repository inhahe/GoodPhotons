// Run-time loader for Intel Open Image Denoise -- see oidn_rt.h for why it is not linked.
#include "oidn_rt.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace oidnrt {
namespace {

// The subset of OIDN 2.x's C API used here, declared rather than included: oidn.h is not in
// the repo (the downloaded release lives under the git-ignored external/), and these are the
// library's stable exported signatures. Its enums are C enums, i.e. ints at the call boundary.
typedef struct OIDNDeviceImpl* OIDNDevice;
typedef struct OIDNFilterImpl* OIDNFilter;
typedef struct OIDNBufferImpl* OIDNBuffer;
constexpr int kFormatFloat3 = 3;   // OIDN_FORMAT_FLOAT3

struct Api {
    OIDNDevice (*newDevice)(int type);
    void       (*releaseDevice)(OIDNDevice);
    void       (*commitDevice)(OIDNDevice);
    int        (*getDeviceError)(OIDNDevice, const char** msg);
    int        (*getDeviceInt)(OIDNDevice, const char* name);
    OIDNBuffer (*newBuffer)(OIDNDevice, size_t bytes);
    void       (*writeBuffer)(OIDNBuffer, size_t offset, size_t bytes, const void* src);
    void       (*readBuffer)(OIDNBuffer, size_t offset, size_t bytes, void* dst);
    void       (*releaseBuffer)(OIDNBuffer);
    OIDNFilter (*newFilter)(OIDNDevice, const char* type);
    void       (*releaseFilter)(OIDNFilter);
    void       (*setFilterImage)(OIDNFilter, const char* name, OIDNBuffer, int format,
                                 size_t w, size_t h, size_t offset, size_t pixStride,
                                 size_t rowStride);
    void       (*setFilterBool)(OIDNFilter, const char* name, bool);
    void       (*setFilterInt)(OIDNFilter, const char* name, int);
    void       (*commitFilter)(OIDNFilter);
    void       (*executeFilter)(OIDNFilter);
};

#ifdef _WIN32
constexpr const char* kLibName = "OpenImageDenoise.dll";
#else
constexpr const char* kLibName = "libOpenImageDenoise.so.2";
#endif

// OIDN asked to stay under this much device memory (its `maxMemoryMB`; it may exceed it a
// little). At 960x540 it does not need to tile to honour it.
constexpr int kMaxMemoryMB = 1024;

struct State {
    std::mutex mu;
    bool tried = false;          // a load has been attempted (once per process)
    bool loaded = false;         // ...and succeeded; shutdown() clears it
    std::string loadErr;         // why it is not loaded: the failure, or "shut down"
    std::string desc;            // describe()'s line
    Api api{};
    OIDNDevice dev = nullptr;
    int devType = 0;             // OIDNDeviceType of the device that was created
    int quality = -1;
    // The committed filter and its buffers, rebuilt only when the image shape changes.
    OIDNFilter filt = nullptr;
    OIDNBuffer bColor = nullptr, bAlb = nullptr, bNrm = nullptr;
    int W = 0, H = 0;
    bool hasAlb = false, hasNrm = false, clean = false;
    uint64_t auxKey = 0;            // the guide set now in bAlb/bNrm (denoise's auxKey)
};

// Never destroyed: shutdown() releases the device at a point the caller controls, ahead of
// the CUDA teardown. A static destructor would run during the CUDA runtime's own exit.
State& state() { static State* s = new State; return *s; }

fs::path exeDir() {
#ifdef _WIN32
    std::vector<wchar_t> buf(32768);
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
    if (n == 0 || n >= buf.size()) return {};
    return fs::path(std::wstring(buf.data(), n)).parent_path();
#else
    std::vector<char> buf(4096);
    const ssize_t n = readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    if (n <= 0) return {};
    return fs::path(std::string(buf.data(), (size_t)n)).parent_path();
#endif
}

// Where the library may be, in the order tried:
//   $FTRACE_OIDN                  the library itself, or the directory holding it;
//   <exe dir>, <exe dir>/oidn     a copy shipped beside ftrace.exe;
//   external/oidn/<release>/bin   in the exe's directory or up to three above it, highest
//                                 version in the release's name first. That is where the repo keeps the
//                                 downloaded release, and it covers both the repo-root
//                                 ftrace.exe and build*/bin/ftrace.exe.
std::vector<fs::path> searchPaths() {
    std::vector<fs::path> out;
    std::error_code ec;
    if (const char* e = std::getenv("FTRACE_OIDN"); e && *e) {
        const fs::path p(e);
        out.push_back(fs::is_directory(p, ec) ? p / kLibName : p);
    }
    const fs::path exe = exeDir();
    if (exe.empty()) return out;
    out.push_back(exe / kLibName);
    out.push_back(exe / "oidn" / kLibName);
    fs::path d = exe;
    for (int up = 0; up < 4; ++up) {
        const fs::path ext = d / "external" / "oidn";
        if (fs::is_directory(ext, ec)) {
            std::vector<fs::path> rel;
            for (const auto& ent : fs::directory_iterator(ext, ec))
                if (ent.is_directory(ec)) rel.push_back(ent.path());
            // Newest first by the numbers in the name, compared as numbers: as strings,
            // "oidn-2.3.1" would sort above "oidn-2.10.0".
            auto version = [](const fs::path& p) {
                std::vector<long> v;
                long cur = -1;
                for (const char c : p.filename().string()) {
                    if (c >= '0' && c <= '9') cur = (cur < 0 ? 0 : cur * 10) + (c - '0');
                    else if (cur >= 0) { v.push_back(cur); cur = -1; }
                }
                if (cur >= 0) v.push_back(cur);
                return v;
            };
            std::sort(rel.begin(), rel.end(), [&](const fs::path& a, const fs::path& b) {
                const auto va = version(a), vb = version(b);
                return va != vb ? va > vb : a.filename().string() > b.filename().string();
            });
            for (const fs::path& r : rel) out.push_back(r / "bin" / kLibName);
        }
        const fs::path parent = d.parent_path();
        if (parent.empty() || parent == d) break;
        d = parent;
    }
    return out;
}

const char* deviceName(int t) {
    switch (t) {
        case 1: return "CPU";
        case 2: return "SYCL";
        case 3: return "CUDA";
        case 4: return "HIP";
        case 5: return "Metal";
        default: return "unknown";
    }
}

// The device's pending error, as `err`, prefixed by what was being done. True if none.
bool noError(State& s, const char* what, std::string& err) {
    const char* msg = nullptr;
    const int code = s.api.getDeviceError(s.dev, &msg);
    if (code == 0) return true;
    err = std::string("OIDN: ") + what + " failed: " +
          (msg && *msg ? std::string(msg) : "error " + std::to_string(code));
    return false;
}

// Load the library, resolve the API and create and commit the device. True on success; on
// failure `s.loadErr` says why.
bool loadImpl(State& s, const Options& opt) {
    std::string looked;
    for (const fs::path& p : searchPaths()) {
        std::error_code ec;
        if (!fs::exists(p, ec)) { looked += "\n    " + p.string(); continue; }
#ifdef _WIN32
        // An absolute path with LOAD_WITH_ALTERED_SEARCH_PATH puts the DLL's own directory on
        // the search path for ITS imports: it loads OpenImageDenoise_core.dll, the device
        // modules and tbb12.dll from beside itself.
        HMODULE h = LoadLibraryExW(p.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!h) {
            looked += "\n    " + p.string() + "  (found, but it would not load: Windows error " +
                      std::to_string((unsigned long)GetLastError()) + ")";
            continue;
        }
        auto sym = [&](const char* n) { return (void*)GetProcAddress(h, n); };
#else
        void* h = dlopen(p.string().c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            looked += "\n    " + p.string() + "  (found, but it would not load: " + dlerror() + ")";
            continue;
        }
        auto sym = [&](const char* n) { return dlsym(h, n); };
#endif
        const char* missing = nullptr;
        auto get = [&](auto& fn, const char* n) {
            void* f = sym(n);
            if (!f && !missing) missing = n;
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(f);
        };
        Api& a = s.api;
        get(a.newDevice, "oidnNewDevice");           get(a.commitDevice, "oidnCommitDevice");
        get(a.releaseDevice, "oidnReleaseDevice");
        get(a.getDeviceError, "oidnGetDeviceError"); get(a.getDeviceInt, "oidnGetDeviceInt");
        get(a.newBuffer, "oidnNewBuffer");           get(a.writeBuffer, "oidnWriteBuffer");
        get(a.readBuffer, "oidnReadBuffer");         get(a.releaseBuffer, "oidnReleaseBuffer");
        get(a.newFilter, "oidnNewFilter");           get(a.releaseFilter, "oidnReleaseFilter");
        get(a.setFilterImage, "oidnSetFilterImage"); get(a.setFilterBool, "oidnSetFilterBool");
        get(a.setFilterInt, "oidnSetFilterInt");     get(a.commitFilter, "oidnCommitFilter");
        get(a.executeFilter, "oidnExecuteFilter");
        if (missing) {
            s.loadErr = "OIDN: " + p.string() + " is not an Open Image Denoise 2.x library (no " +
                        missing + ")";
            return false;
        }
        s.dev = a.newDevice((int)opt.device);
        if (!s.dev) {
            std::string why;
            noError(s, "creating the device", why);   // newDevice reports through a null device
            s.loadErr = why.empty() ? "OIDN: could not create a device" : why;
            return false;
        }
        a.commitDevice(s.dev);
        if (!noError(s, "starting the device", s.loadErr)) return false;
        const int ver = a.getDeviceInt(s.dev, "version");   // e.g. 20501 for 2.5.1
        s.devType = a.getDeviceInt(s.dev, "type");
        char line[512];
        std::snprintf(line, sizeof line, "Open Image Denoise %d.%d.%d on %s, from %s",
                      ver / 10000, (ver / 100) % 100, ver % 100, deviceName(s.devType),
                      p.string().c_str());
        s.desc = line;
        s.loadErr.clear();
        return true;
    }
    s.loadErr = std::string("OIDN: ") + kLibName + " not found. Looked in:" + looked +
                "\n  Download a release from https://github.com/RenderKit/oidn/releases and unpack it"
                " to external/oidn/, put its bin/ beside ftrace.exe, or point FTRACE_OIDN at it.";
    return false;
}

// Once per process. `loaded` is set only by a load that got all the way through, so a throw
// out of the middle (a path the code page cannot spell, a directory iterator failing) leaves
// "not loaded" behind, with the reason, rather than null entry points that look loaded.
bool load(State& s, const Options& opt) {
    if (s.tried) return s.loaded;
    s.tried = true;
    s.loadErr = "OIDN: loading did not complete";
    try {
        s.loaded = loadImpl(s, opt);
    } catch (const std::exception& e) {
        s.loaded = false;
        s.loadErr = std::string("OIDN: loading failed: ") + e.what();
    }
    return s.loaded;
}

void dropFilter(State& s) {
    Api& a = s.api;
    if (s.filt)   a.releaseFilter(s.filt);
    if (s.bColor) a.releaseBuffer(s.bColor);
    if (s.bAlb)   a.releaseBuffer(s.bAlb);
    if (s.bNrm)   a.releaseBuffer(s.bNrm);
    s.filt = nullptr; s.bColor = s.bAlb = s.bNrm = nullptr;
    s.W = s.H = 0;
    s.auxKey = 0;
}

}  // namespace

bool denoise(float* color, const float* albedo, const float* normal, int W, int H,
             bool cleanAux, uint64_t auxKey, const Options& opt, std::string& err) {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!load(s, opt)) { err = s.loadErr; return false; }
    if (W <= 0 || H <= 0 || !color) { err = "OIDN: empty image"; return false; }
    if (!albedo) normal = nullptr;   // the RT filter takes a normal only with an albedo
    const bool hasAlb = albedo != nullptr, hasNrm = normal != nullptr;
    const bool clean = cleanAux && hasAlb;
    const size_t bytes = (size_t)W * (size_t)H * 3 * sizeof(float);
    Api& a = s.api;
    if (!s.filt || s.W != W || s.H != H || s.hasAlb != hasAlb || s.hasNrm != hasNrm ||
        s.clean != clean || s.quality != (int)opt.quality) {
        dropFilter(s);
        s.bColor = a.newBuffer(s.dev, bytes);
        if (hasAlb) s.bAlb = a.newBuffer(s.dev, bytes);
        if (hasNrm) s.bNrm = a.newBuffer(s.dev, bytes);
        if (!noError(s, "allocating buffers", err)) { dropFilter(s); return false; }
        s.filt = a.newFilter(s.dev, "RT");
        a.setFilterImage(s.filt, "color", s.bColor, kFormatFloat3, W, H, 0, 0, 0);
        if (hasAlb) a.setFilterImage(s.filt, "albedo", s.bAlb, kFormatFloat3, W, H, 0, 0, 0);
        if (hasNrm) a.setFilterImage(s.filt, "normal", s.bNrm, kFormatFloat3, W, H, 0, 0, 0);
        a.setFilterImage(s.filt, "output", s.bColor, kFormatFloat3, W, H, 0, 0, 0);   // in place
        a.setFilterBool(s.filt, "hdr", true);
        a.setFilterBool(s.filt, "cleanAux", clean);
        a.setFilterInt(s.filt, "quality", (int)opt.quality);
        a.setFilterInt(s.filt, "maxMemoryMB", kMaxMemoryMB);
        a.commitFilter(s.filt);
        if (!noError(s, "setting up the filter", err)) { dropFilter(s); return false; }
        s.W = W; s.H = H; s.hasAlb = hasAlb; s.hasNrm = hasNrm; s.clean = clean;
        s.quality = (int)opt.quality;
    }
    a.writeBuffer(s.bColor, 0, bytes, color);
    // The guides are constant for a camera's whole render: upload them once per guide set.
    if (hasAlb && (auxKey == 0 || auxKey != s.auxKey)) {
        a.writeBuffer(s.bAlb, 0, bytes, albedo);
        if (hasNrm) a.writeBuffer(s.bNrm, 0, bytes, normal);
        s.auxKey = auxKey;
    }
    a.executeFilter(s.filt);
    if (!noError(s, "denoising", err)) return false;
    a.readBuffer(s.bColor, 0, bytes, color);
    return noError(s, "reading the result back", err);
}

std::string describe() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    return s.loaded ? s.desc : s.loadErr;
}

size_t deviceMemoryAllowance() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.loaded || s.devType == 1 /*CPU*/) return 0;
    return ((size_t)kMaxMemoryMB + 256) * 1024 * 1024;
}

bool shutdown() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.loaded) return false;
    const bool gpu = s.devType != 1;
    dropFilter(s);
    s.api.releaseDevice(s.dev);
    s.dev = nullptr;
    s.loaded = false;
    s.loadErr = "OIDN: shut down";
    return gpu;
}

}  // namespace oidnrt
