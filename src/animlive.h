// animlive.h — the fly editor's live channel to loom (E2 "channel b").
//
// `ftrace scene.ftsl -anim drive.json -loom scene.py` turns the curve editor from a
// sidecar editor into a live one: every scrub position is pushed to a resident
// `python -m loom.anim` child, which applies the drive's channel->variable bindings
// and emits that frame's `.ftsl`, which ftrace then loads and previews. You are
// flying through the animation, not through a still.
//
// Division of labour with curvedrive.h (channel a):
//   * curvedrive.h owns the sidecar FILE — load, validate, atomic save. ftrace stays
//     the only writer of it, so there is no two-writer race with the child.
//   * this owns the live SESSION — the values, and the .ftsl that falls out of them.
//
// LOOM SAMPLES THE CURVE, NOT FTRACE. The editor pushes its control points to loom
// whenever they change and then asks for a frame by curve parameter `t`. It would be
// easy to sample the curve on this side instead (the editor already runs a
// Catmull-Rom for its own preview polyline) — and wrong: any drift between the two
// samplers would make the live preview quietly disagree with the video loom finally
// renders, which is the one thing this channel exists to prevent.
//
// TWO QUEUES, deliberately:
//   * frames are LATEST-WINS (one slot). A drag across the timeline generates far
//     more positions than loom can emit; the intermediate ones are worthless, and
//     queueing them would make the user sit through a backlog of stale frames.
//   * control messages (points / bindings / dims) are a FIFO that must NOT drop.
//     They are edits, not samples — losing one silently desynchronises loom's drive
//     from the editor's, and the next frame would be emitted against stale values.
// The worker drains all pending control messages before it starts a frame, so a
// frame never races ahead of the edit that should have shaped it.
#pragma once
#include <string>
#include <vector>
#include <utility>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include "loomlink.h"

namespace animlive {

// One frame request: a position on the drive curve plus the clock it stands for.
struct Job {
    long long seq    = 0;
    double    t      = 0.0;   // curve parameter in [0,1]
    int       frame  = 0;
    int       frames = 1;
};

struct Result {
    long long   seq = 0;
    bool        ok  = false;
    std::string err;
    std::string ftslPath;     // temp .ftsl loom wrote (caller reaps it)
    double      ms  = 0.0;    // wall time of the round trip
};

#ifndef _WIN32
// The link itself stubs out on non-Windows; so does the bridge, so `-anim -loom`
// reports cleanly instead of failing to compile.
struct Bridge {
    bool start(const std::string&, const std::string&, std::string& err) {
        err = "the loom live channel needs a Windows build";
        return false;
    }
    void stop() {}
    void post(Job) {}
    void control(const std::string&) {}
    bool take(Result&) { return false; }
    bool busy() const { return false; }
    bool linkUp() const { return false; }
    std::string deadReason() const { return "unavailable"; }
    const std::vector<std::pair<std::string, double>>& slots() const { return slots_; }
    const std::string& command() const { return cmd_; }
    void reap(const std::string&) {}
private:
    std::vector<std::pair<std::string, double>> slots_;
    std::string cmd_;
};
#else

struct Bridge {
    Bridge() = default;
    // Owns a thread, a child process and pipe handles: not copyable, and destruction
    // must stop the worker (~std::thread on a joinable thread calls std::terminate,
    // which would turn any early return in the editor into an abort).
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;
    ~Bridge() { stop(); }

    // ---- editor thread ----

    // Spawn `python -m loom.anim <scenePy> [--config <configPath>]` and do the two
    // synchronous handshakes (slots, config) before the worker owns the link.
    bool start(const std::string& scenePy, const std::string& configPath, std::string& err) {
        std::vector<std::string> args;
        if (!configPath.empty()) { args.push_back("--config"); args.push_back(configPath); }
        if (!link_.start("loom.anim", scenePy, args, err)) return false;

        // The bindable-variable menu: what the scene COULD expose, so the panel can
        // offer a pick-list instead of making the user type a target into a GDI box.
        minijson::Value ack;
        if (!link_.call("{\"cmd\":\"slots\"}", ack, err)) { link_.stop(); return false; }
        if (const minijson::Value* s = ack.find("slots"); s && s->isObject())
            for (const auto& kv : s->obj)
                slots_.push_back({kv.first, kv.second.asNumber(0.0)});

        // The drive loom proposes. ftrace's own sidecar read (curvedrive.h) is the
        // editor's source of truth for the curve; this is kept so the panel can say
        // what the scene wanted when the two disagree.
        if (link_.call("{\"cmd\":\"config\"}", ack, err)) {
            if (const minijson::Value* c = ack.find("config"))
                haveConfig_ = (c->isObject());
        }
        err.clear();

        if (!makeTempDir(err)) { link_.stop(); return false; }
        worker_ = std::thread([this] { workerMain(); });
        return true;
    }

    // Idempotent: the destructor calls it too.
    void stop() {
        if (worker_.joinable()) {
            { std::lock_guard<std::mutex> lk(m_); quit_ = true; }
            cv_.notify_one();
            worker_.join();
        }
        link_.stop();
        if (!tempDir_.empty()) {
            {
                std::lock_guard<std::mutex> lk(m_);
                for (const auto& f : temps_) DeleteFileA(f.c_str());
                temps_.clear();
            }
            // Sweep the whole scratch directory, not just the files we named: emitting
            // an .ftsl also drops the mesh assets it references (loom's `asset_path`
            // writes .obj next to `out`), and RemoveDirectory fails on a non-empty dir.
            WIN32_FIND_DATAA fd{};
            HANDLE h = FindFirstFileA((tempDir_ + "\\*").c_str(), &fd);
            if (h != INVALID_HANDLE_VALUE) {
                do {
                    if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
                    DeleteFileA((tempDir_ + "\\" + fd.cFileName).c_str());
                } while (FindNextFileA(h, &fd));
                FindClose(h);
            }
            RemoveDirectoryA(tempDir_.c_str());
            tempDir_.clear();
        }
    }

    // LATEST WINS: overwrites any frame job that has not started yet.
    void post(Job j) {
        std::lock_guard<std::mutex> lk(m_);
        j.seq = ++seq_;
        pending_ = j;
        hasPending_ = true;
        cv_.notify_one();
    }

    // A control message (raw JSON line). FIFO, never dropped — see the header note.
    void control(const std::string& line) {
        std::lock_guard<std::mutex> lk(m_);
        ctrl_.push_back(line);
        cv_.notify_one();
    }

    bool take(Result& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (!hasResult_) return false;
        out = result_;
        hasResult_ = false;
        return true;
    }

    // "a re-derivation is happening or is about to" — keeps a drag from reading as
    // idle in the gap between two jobs.
    bool busy() const {
        std::lock_guard<std::mutex> lk(m_);
        return running_ || hasPending_;
    }
    bool linkUp() const {
        std::lock_guard<std::mutex> lk(m_);
        return !dead_;
    }
    std::string deadReason() const {
        std::lock_guard<std::mutex> lk(m_);
        return deadErr_;
    }
    // The scene's bindable variables (name -> default), sorted by loom.
    const std::vector<std::pair<std::string, double>>& slots() const { return slots_; }
    bool haveConfig() const { return haveConfig_; }
    const std::string& command() const { return link_.cmdline; }

    // A scratch .ftsl the editor has finished loading.
    void reap(const std::string& path) {
        if (path.empty()) return;
        DeleteFileA(path.c_str());
        std::lock_guard<std::mutex> lk(m_);
        forgetLocked(path);
    }

private:
    void forgetLocked(const std::string& path) {
        for (size_t i = 0; i < temps_.size(); ++i)
            if (temps_[i] == path) { temps_[i] = temps_.back(); temps_.pop_back(); return; }
    }
    void dropLocked(const std::string& path) {
        if (path.empty()) return;
        DeleteFileA(path.c_str());
        forgetLocked(path);
    }

    bool makeTempDir(std::string& err) {
        char tmp[MAX_PATH];
        DWORD n = GetTempPathA(MAX_PATH, tmp);
        if (n == 0 || n > MAX_PATH) { err = "GetTempPath failed"; return false; }
        char dir[MAX_PATH + 64];
        std::snprintf(dir, sizeof dir, "%sftrace_anim_%lu", tmp, GetCurrentProcessId());
        if (!CreateDirectoryA(dir, nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
            err = "cannot create the live-channel scratch directory";
            return false;
        }
        tempDir_ = dir;
        return true;
    }

    // Worker thread only.
    std::string scratch(long long seq) {
        char b[MAX_PATH + 64];
        std::snprintf(b, sizeof b, "%s\\live_%lld.ftsl", tempDir_.c_str(), seq);
        std::string p = b;
        { std::lock_guard<std::mutex> lk(m_); temps_.push_back(p); }
        return p;
    }

    static std::string numJson(double v) {
        char b[40];
        std::snprintf(b, sizeof b, "%.17g", v);
        return b;
    }

    void workerMain() {
        for (;;) {
            Job job;
            std::vector<std::string> ctrl;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return quit_ || hasPending_ || !ctrl_.empty(); });
                if (quit_) return;
                ctrl.swap(ctrl_);          // drain the edits FIRST, always
                job = pending_;
                bool had = hasPending_;
                hasPending_ = false;
                running_ = had;
                if (!had) {
                    // Control-only wake-up: apply the edits and go back to sleep. The
                    // next frame the editor posts will already see them.
                    lk.unlock();
                    std::string cerr;
                    for (const std::string& line : ctrl) {
                        minijson::Value ack;
                        if (!link_.call(line, ack, cerr)) { markDead(cerr); return; }
                    }
                    continue;
                }
            }
            std::string err;
            bool ok = true;
            for (const std::string& line : ctrl) {
                minijson::Value ack;
                if (!link_.call(line, ack, err)) { markDead(err); return; }
            }

            LARGE_INTEGER f, t0, t1;
            QueryPerformanceFrequency(&f);
            QueryPerformanceCounter(&t0);

            Result r;
            r.seq = job.seq;
            std::string out = scratch(job.seq);
            std::string req = "{\"cmd\":\"frame\",\"t\":" + numJson(job.t)
                            + ",\"frame\":" + std::to_string(job.frame)
                            + ",\"frames\":" + std::to_string(job.frames)
                            + ",\"out\":\"" + loomlink::jsonEsc(out) + "\"}";
            minijson::Value ack;
            ok = link_.call(req, ack, err);
            QueryPerformanceCounter(&t1);
            r.ms = f.QuadPart ? 1000.0 * double(t1.QuadPart - t0.QuadPart) / double(f.QuadPart) : 0.0;
            if (ok) {
                r.ok = true;
                r.ftslPath = out;
            } else {
                r.ok = false;
                r.err = err;
                // A failed request may still have left a partial file behind; drop it
                // rather than let it sit in temps_ until the editor exits.
                std::lock_guard<std::mutex> lk(m_);
                dropLocked(out);
            }
            {
                std::lock_guard<std::mutex> lk(m_);
                // A result the editor never collected is stale by definition — the only
                // one worth keeping is the newest.
                if (hasResult_ && !result_.ftslPath.empty()) dropLocked(result_.ftslPath);
                result_ = r;
                hasResult_ = true;
                running_ = false;
            }
            // The link tears itself down on a transport failure (a protocol-level
            // `ok:false` leaves it up, since loom reports scene errors that way).
            if (!ok && !link_.alive()) { markDead(err); return; }
        }
    }

    void markDead(const std::string& why) {
        std::lock_guard<std::mutex> lk(m_);
        dead_ = true;
        deadErr_ = why;
        running_ = false;
        hasPending_ = false;
    }

    loomlink::Link link_;
    std::thread    worker_;
    mutable std::mutex m_;
    std::condition_variable cv_;

    bool quit_ = false, hasPending_ = false, running_ = false, hasResult_ = false, dead_ = false;
    long long seq_ = 0;
    Job       pending_;
    Result    result_;
    std::string deadErr_;
    std::vector<std::string> ctrl_;
    std::vector<std::string> temps_;
    std::string tempDir_;

    std::vector<std::pair<std::string, double>> slots_;
    bool haveConfig_ = false;
};
#endif // _WIN32

} // namespace animlive
