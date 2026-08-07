// loomlink.h — the shared child-process link to a loom stdio server.
//
// loom exposes two long-lived newline-delimited-JSON servers that ftrace drives as
// a child process, and they speak the *same* transport:
//
//   * `python -m loom.viewer <scene.py>`  — §F4 live re-introspection, used by the
//     native viewer (`-viewer`) to re-derive geometry when the user moves along a
//     parameter dimension.
//   * `python -m loom.anim <scene.py> [--config sidecar.json]` — §E2 channel-b live
//     values, used by the fly editor (`-anim`) to turn a scrub position into a
//     freshly-emitted `.ftsl`.
//
// Both want the identical plumbing: inheritable pipes with our own ends marked
// non-inheritable, a PYTHONPATH-augmented environment so a working-copy loom is
// importable, one request/ack round trip per call, and a teardown that says `quit`
// before it resorts to terminating. That plumbing lived inside viewer_gui.cpp until
// the editor needed it too; it is here so the two callers cannot drift apart.
//
// What is NOT here: the worker thread and its latest-wins pending job. That policy
// differs per caller (the viewer coalesces param drags; the editor coalesces scrub
// positions and has a different notion of what "the scene changed" invalidates), so
// each owns its own bridge over this transport.
//
// Windows-only, like every other child-process/GUI piece in the tree. On other
// platforms `Link` compiles to a stub whose start() reports why.
#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include "third_party/json.h"

namespace loomlink {

// A binary attachment to one ack.
//
// The transport is newline-delimited JSON, which is exactly wrong for a megabyte of
// mesh: base64 in the ack would cost a 4/3 blowup plus an encode and a decode, and a
// side file costs an open — which on this machine means ~8 ms of Windows Defender
// scanning a file our own child wrote a millisecond ago (see `known-issues.md`). So
// the ack line may carry a manifest, `"blobs":[{"name":…,"bytes":N}, …]`, and the raw
// payloads follow the newline back to back in that order. Every length is known
// before a byte is read, so there is nothing to delimit and nothing to escape.
//
// The manifest is part of the ack, not a separate message, which makes the invariant
// checkable at one point: a reader that has parsed an ack knows exactly how many
// bytes still belong to it. `Link::call` therefore ALWAYS consumes them, even when
// the caller passes no `blobs` vector and even when the ack reports failure —
// leaving them in the pipe would desynchronise every request after it.
struct Blob {
    std::string name;   // the path the scene text uses to name this asset
    std::string bytes;
};

// minijson parses but does not serialise, and every request line is a small fixed
// shape, so callers build them by hand over this escape.
inline std::string jsonEsc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '"':  o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n";  break;
        case '\r': o += "\\r";  break;
        case '\t': o += "\\t";  break;
        default:
            if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
            else o += (char)c;
        }
    }
    return o;
}

} // namespace loomlink

#ifndef _WIN32
// -------------------------- Non-Windows stub ---------------------------------
namespace loomlink {
struct Link {
    std::string cmdline;
    bool alive() const { return false; }
    bool start(const std::string&, const std::string&,
               const std::vector<std::string>&, std::string& err) {
        err = "the loom live channel needs a Windows build (child-process pipes)";
        return false;
    }
    void stop() {}
    bool call(const std::string&, minijson::Value&, std::string& err,
              std::vector<Blob>* = nullptr) {
        err = "loom link is not running";
        return false;
    }
};
} // namespace loomlink
#else
// ============================== Win32 ========================================
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace loomlink {

inline std::wstring utf8ToWide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (n <= 0) return std::wstring(s.begin(), s.end());
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}

// Directory holding this executable (where `tools\loom` lives in a working copy).
inline std::wstring exeDirW() {
    wchar_t buf[MAX_PATH * 2];
    DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD)(sizeof buf / sizeof buf[0]));
    if (n == 0) return std::wstring();
    std::wstring p(buf, n);
    size_t cut = p.find_last_of(L"\\/");
    return cut == std::wstring::npos ? std::wstring() : p.substr(0, cut);
}

// The child's environment: ours, with `<exeDir>\tools\loom` prepended to PYTHONPATH.
// loom's package root sits there in a working copy; an installed loom is found the
// normal way and the extra entry is inert. Entries are kept in order (the `=X:=…`
// drive-cwd pseudo-variables must stay first), any inherited PYTHONPATH is folded
// into ours rather than dropped.
inline std::wstring childEnvBlock(const std::wstring& extraPyPath) {
    std::wstring out, oldPP;
    if (LPWCH env = GetEnvironmentStringsW()) {
        for (LPWCH p = env; *p; ) {
            std::wstring entry(p);
            p += entry.size() + 1;
            std::wstring head = entry.substr(0, 11);
            for (auto& c : head) c = (wchar_t)towupper(c);
            if (head == L"PYTHONPATH=") { oldPP = entry.substr(11); continue; }
            out += entry;
            out.push_back(L'\0');
        }
        FreeEnvironmentStringsW(env);
    }
    std::wstring pp = L"PYTHONPATH=" + extraPyPath;
    if (!oldPP.empty()) pp += L";" + oldPP;
    out += pp;
    out.push_back(L'\0');
    out.push_back(L'\0');
    return out;
}

// One live loom child. NOT thread-safe: a caller that runs requests off a worker
// thread must ensure only that thread touches the link once it is started.
struct Link {
    HANDLE proc = nullptr;
    HANDLE wr   = nullptr;      // -> child stdin
    HANDLE rd   = nullptr;      // <- child stdout
    std::string rx;             // bytes read past the last complete line
    std::string cmdline;        // shown in the caller's status row

    bool alive() const { return proc != nullptr; }

    // `module` is the `-m` target ("loom.viewer" / "loom.anim"); `args` are extra
    // argv entries appended after the scene file (each quoted).
    bool start(const std::string& module, const std::string& scenePy,
               const std::vector<std::string>& args, std::string& err) {
        SECURITY_ATTRIBUTES sa{ sizeof(sa), nullptr, TRUE };
        HANDLE inRd = nullptr, inWr = nullptr, outRd = nullptr, outWr = nullptr;
        if (!CreatePipe(&inRd, &inWr, &sa, 0)) { err = "CreatePipe(stdin) failed"; return false; }
        if (!CreatePipe(&outRd, &outWr, &sa, 0)) {
            CloseHandle(inRd); CloseHandle(inWr);
            err = "CreatePipe(stdout) failed"; return false;
        }
        // Our ends must NOT be inherited: if the child held the write end of its own
        // stdout, our read would never see EOF when the child dies and a crashed loom
        // would hang the worker thread forever instead of reporting.
        SetHandleInformation(inWr,  HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(outRd, HANDLE_FLAG_INHERIT, 0);

        // -u: unbuffered, so a traceback on the way down is not swallowed (the loom
        // servers flush their own acks). -X utf8: scene text and paths are UTF-8.
        cmdline = "python -X utf8 -u -m " + module + " \"" + scenePy + "\"";
        for (const std::string& a : args) {
            // A bare flag (`--config`) must not be quoted or argparse sees a value;
            // anything else is a path and must be.
            if (a.size() >= 2 && a[0] == '-' && a.find(' ') == std::string::npos)
                cmdline += " " + a;
            else
                cmdline += " \"" + a + "\"";
        }
        std::wstring wcmd = utf8ToWide(cmdline);
        std::vector<wchar_t> mutcmd(wcmd.begin(), wcmd.end());
        mutcmd.push_back(L'\0');                 // CreateProcessW may write to it

        std::wstring loomPath = exeDirW();
        if (!loomPath.empty()) loomPath += L"\\tools\\loom";
        std::wstring env = childEnvBlock(loomPath);

        STARTUPINFOW si{};
        si.cb = sizeof si;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput  = inRd;
        si.hStdOutput = outWr;
        // loom's stderr rides our console so a scene-file traceback is readable; if we
        // have no console the child simply gets none.
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        if (si.hStdError && si.hStdError != INVALID_HANDLE_VALUE)
            SetHandleInformation(si.hStdError, HANDLE_FLAG_INHERIT, TRUE);
        else
            si.hStdError = outWr;   // never leave it unset — the child would inherit ours

        PROCESS_INFORMATION pi{};
        BOOL ok = CreateProcessW(nullptr, mutcmd.data(), nullptr, nullptr, TRUE,
                                 CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
                                 (LPVOID)env.data(), nullptr, &si, &pi);
        CloseHandle(inRd);
        CloseHandle(outWr);
        if (!ok) {
            CloseHandle(inWr); CloseHandle(outRd);
            char b[64]; std::snprintf(b, sizeof b, " (error %lu)", GetLastError());
            err = "cannot start `" + cmdline + "`" + b + " - is python on PATH?";
            return false;
        }
        CloseHandle(pi.hThread);
        proc = pi.hProcess;
        wr = inWr;
        rd = outRd;
        return true;
    }

    void stop() {
        if (wr) {
            // A clean `quit` lets loom exit on its own; closing stdin is the backstop
            // (both serve loops end at EOF).
            const char* bye = "{\"cmd\":\"quit\"}\n";
            DWORD done = 0;
            WriteFile(wr, bye, (DWORD)std::strlen(bye), &done, nullptr);
            CloseHandle(wr); wr = nullptr;
        }
        if (proc) {
            if (WaitForSingleObject(proc, 3000) != WAIT_OBJECT_0)
                TerminateProcess(proc, 1);      // wedged in user code — don't leak it
            CloseHandle(proc); proc = nullptr;
        }
        if (rd) { CloseHandle(rd); rd = nullptr; }
        rx.clear();
    }

    bool readLine(std::string& line, std::string& err) {
        for (;;) {
            size_t nl = rx.find('\n');
            if (nl != std::string::npos) {
                line = rx.substr(0, nl);
                rx.erase(0, nl + 1);
                return true;
            }
            char buf[8192];
            DWORD got = 0;
            if (!ReadFile(rd, buf, (DWORD)sizeof buf, &got, nullptr) || got == 0) {
                err = "loom link: the python process closed its output"
                      " (see its traceback on stderr)";
                return false;
            }
            rx.append(buf, got);
        }
    }

    // Exactly `n` more bytes of the child's stdout, taken from what `readLine` has
    // already buffered before the pipe is touched again.
    bool readExact(size_t n, std::string& out, std::string& err) {
        out.clear();
        out.reserve(n);
        while (out.size() < n) {
            if (!rx.empty()) {
                size_t take = (std::min)(n - out.size(), rx.size());
                out.append(rx, 0, take);
                rx.erase(0, take);
                continue;
            }
            char buf[1 << 16];
            DWORD want = (DWORD)(std::min)(sizeof buf, n - out.size());
            DWORD got = 0;
            if (!ReadFile(rd, buf, want, &got, nullptr) || got == 0) {
                err = "loom link: the python process closed its output mid-attachment"
                      " (see its traceback on stderr)";
                return false;
            }
            out.append(buf, got);
        }
        return true;
    }

    // One request/ack round trip. A transport failure tears the link down (there is
    // no resynchronising a half-written pipe); a protocol-level `ok:false` leaves it
    // up, since loom reports scene errors that way and stays serving.
    //
    // `blobs`, when given, receives the ack's binary attachments (see `Blob`). They
    // are read off the pipe either way — passing null discards them rather than
    // leaving them to be mistaken for the next ack.
    bool call(const std::string& line, minijson::Value& ack, std::string& err,
              std::vector<Blob>* blobs = nullptr) {
        if (blobs) blobs->clear();
        if (!alive()) { err = "loom link is not running"; return false; }
        std::string msg = line;
        msg.push_back('\n');
        for (size_t off = 0; off < msg.size(); ) {
            DWORD done = 0;
            if (!WriteFile(wr, msg.data() + off, (DWORD)(msg.size() - off), &done, nullptr)
                || done == 0) {
                err = "loom link: write failed (the python process exited?)";
                stop();
                return false;
            }
            off += done;
        }
        std::string reply;
        if (!readLine(reply, err)) { stop(); return false; }
        std::string perr;
        if (!minijson::parse(reply, ack, perr)) {
            // The manifest we would need in order to skip past any attachment is in
            // the line we just failed to parse, so the pipe's framing is gone with it.
            err = "loom link: unparsable ack (" + perr + ")";
            stop();
            return false;
        }
        // Drain attachments BEFORE the ok/error check: they belong to this ack no
        // matter what it says, and the next request would read them as its reply.
        if (const minijson::Value* bl = ack.find("blobs")) {
            if (bl->isArray()) {
                for (const minijson::Value& e : bl->arr) {
                    long long n = (long long)e.numAt("bytes", -1.0);
                    if (n < 0) {
                        err = "loom link: blob manifest entry has no byte count";
                        stop();
                        return false;
                    }
                    std::string data;
                    if (!readExact((size_t)n, data, err)) { stop(); return false; }
                    if (blobs) {
                        const minijson::Value* nm = e.find("name");
                        Blob b;
                        b.name = nm && nm->isString() ? nm->str : std::string();
                        b.bytes = std::move(data);
                        blobs->push_back(std::move(b));
                    }
                }
            }
        }
        const minijson::Value* okv = ack.find("ok");
        if (!okv || !okv->asBool(false)) {
            const minijson::Value* e = ack.find("error");
            err = "loom: " + (e && e->isString() ? e->str : std::string("request failed"));
            return false;
        }
        return true;
    }
};

} // namespace loomlink
#endif // _WIN32
