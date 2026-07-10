// Forward spectral photon tracer — Phase 0 scaffold.
// This first commit only proves the toolchain: it writes a test PPM gradient.
// The real photon loop replaces this in the next increment.

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    const int W = 256, H = 256;
    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 3;
            img[i + 0] = static_cast<uint8_t>(x);          // R ramp
            img[i + 1] = static_cast<uint8_t>(y);          // G ramp
            img[i + 2] = static_cast<uint8_t>((x + y) / 2);// B
        }
    }

    const char* out = (argc > 1) ? argv[1] : "test.ppm";
    std::ofstream f(out, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", out); return 1; }
    f << "P6\n" << W << ' ' << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
    std::printf("wrote %s (%dx%d)\n", out, W, H);
    return 0;
}
