// Single translation unit that compiles the stb_image implementation. Every other
// file uses only the small set of stbi_* functions forward-declared in texture.h,
// so the 8k-line vendored header is parsed exactly once (here) and never by nvcc.
//
// stb_image is public domain (see src/third_party/stb_image.h header). We disable
// the formats the renderer doesn't need to keep the binary lean; PNG/JPG/BMP/TGA
// and the HDR float path (Radiance .hdr) are the useful ones for textures.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_GIF
#define STBI_NO_PIC
#define STBI_NO_PNM   // we have our own PPM/PFM loaders in texture.h
#include "third_party/stb_image.h"

// stb_image_write: the renderer writes 8-bit RGB output. Honour the output
// extension (.png/.jpg) with a real encoder instead of always emitting PPM; see
// writeImage() in main.cpp. PNG uses stb's built-in zlib deflate.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
