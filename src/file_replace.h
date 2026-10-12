// file_replace.h -- write a file so that a reader never sees it half-written and a failed write
// never destroys the version it was replacing. Used for everything a render writes and rewrites:
// the image, the -hdr PFM, the checkpoints, the photon map, the exposure anchor.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <system_error>
#include <thread>

namespace filereplace {

// Write the whole of `write(os)` to `dest`, and report whether every byte reached the disk.
// `keep`: open an existing `dest` WITHOUT truncating it and cut it to the new length only after
// all of the new bytes are down, so a write that fails partway (a full disk) has not first
// emptied the file it was replacing. std::fstream opens SHARING reads and writes, which is what
// lets this rewrite a file a viewer holds open (stb's file writer and fopen_s share nothing).
template <class Write>
inline bool writeTo(const std::string& dest, Write& write, bool keep) {
    std::error_code ec;
    const bool existed = keep && std::filesystem::exists(dest, ec);
    std::fstream f;
    if (existed) f.open(dest, std::ios::in | std::ios::out | std::ios::binary);
    else         f.open(dest, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!f) return false;
    if (!write(static_cast<std::ostream&>(f))) return false;
    const std::streamoff len = f.tellp();
    f.close();
    if (f.fail() || len < 0) return false;
    if (existed) {
        std::filesystem::resize_file(dest, (std::uintmax_t)len, ec);
        if (ec) return false;
    }
    return true;
}

inline bool isReadOnly(const std::string& path) {
    std::error_code ec;
    const auto st = std::filesystem::status(path, ec);
    return !ec && std::filesystem::exists(st) &&
           (st.permissions() & std::filesystem::perms::owner_write) == std::filesystem::perms::none;
}

// Write the file `path` from `write(std::ostream&)`, which writes the whole file and returns
// false if it could not. Returns whether `path` now holds the new file.
//
// The file is written BESIDE, as `<path>.tmp`, and renamed over `path`: a reader never sees a
// half-written file, and a crash or a full disk mid-write leaves the last good one.
//   * Something can HOLD a just-written file: a backup program or virus scanner (IDrive's image
//     handler held a fresh 1 MB PNG ~2.4 s on this machine, measured), or a viewer that keeps it
//     open. The rename is then retried for up to 10 s. A viewer that shares writing cannot be
//     replaced, only rewritten, so from 2.5 s on -- past a scanner's typical hold, which the
//     rename should wait out -- a non-truncating in-place write is tried once a second.
//   * A read-only target can never be replaced or rewritten, so it is not waited on.
//   * A `.tmp` that cannot be removed (a scanner still on the last one, or read-only) is
//     stepped around under the next name rather than written through.
//   * Where nothing can be created beside the target at all (permissions, a path at MAX_PATH),
//     it is written in place, non-truncating. A `.tmp` that was created but could not be
//     completed means a full or failing disk: the target is then left alone.
// On failure the old file is as it was and no `.tmp` is left behind, except one something else
// still holds.
template <class Write>
inline bool writeReplacing(const std::string& path, Write&& write) {
    std::error_code ec;
    std::string tmp;
    for (int k = 0; k < 4 && tmp.empty(); ++k) {
        const std::string t = path + (k ? ".tmp" + std::to_string(k) : std::string(".tmp"));
        std::filesystem::remove(t, ec);                   // a stale one, left by a crash
        if (!std::filesystem::exists(t, ec)) tmp = t;
    }
    if (tmp.empty()) return writeTo(path, write, /*keep=*/true);
    if (!writeTo(tmp, write, /*keep=*/false)) {
        const bool made = std::filesystem::exists(tmp, ec);
        std::filesystem::remove(tmp, ec);
        return made ? false : writeTo(path, write, /*keep=*/true);
    }
    const bool readOnly = isReadOnly(path);
    for (int attempt = 0; attempt <= 100; ++attempt) {
        if (attempt) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::filesystem::rename(tmp, path, ec);
        if (!ec) return true;
        if (readOnly) break;
        if (attempt >= 25 && attempt % 10 == 5 && writeTo(path, write, /*keep=*/true)) {
            std::filesystem::remove(tmp, ec);
            return true;
        }
    }
    std::filesystem::remove(tmp, ec);
    return false;
}

}  // namespace filereplace
