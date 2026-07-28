// pool.hpp — intrusive-refcount smart pointer + freelist pool allocator.
//
// Designed for hot-path allocation in the parser.  shared_ptr allocates a
// separate control block and does atomic bookkeeping via a vtable-less
// indirection; this is cheaper for simple RAII types like PList<T> and
// ParseNode.
//
// Usage:
//   struct Foo : gpda_pool::Refcounted<Foo> {
//       ...
//       static void deallocate(Foo* p) noexcept {
//           gpda_pool::Pool<Foo>::instance().destroy(p);
//       }
//   };
//   using FooPtr = gpda_pool::IntrusivePtr<Foo>;
//
//   FooPtr f(Pool<Foo>::instance().make(args...));
//
// One thread-local Pool per T.  Objects are returned to the pool when
// refcount drops to zero; memory stays alive for the thread's lifetime
// (typical parsers churn many short-lived objects, so the pool stays
// warm).
//
// THREADING: a pool-allocated object belongs to the thread that allocated it,
// and only that thread may hold, copy or drop IntrusivePtrs to it.  That is
// not a restriction introduced by the non-atomic refcount below — it is
// inherent to the thread-local pool: `T::deallocate` returns the object to
// *the releasing* thread's pool, so dropping the last reference on another
// thread corrupts that thread's freelist however the count is maintained.
// Several threads each running their own parser is fine; each touches only
// its own objects.  Handing a parse tree to another thread and refcounting it
// there was already unsupported.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace gpda_pool {

// Base for pool-allocated, intrusively-refcounted types.
//
// The counter is deliberately NOT std::atomic: see the THREADING note above —
// an object may only ever be refcounted by its allocating thread, so the
// interlocked read-modify-write buys nothing.  It is not free either: the
// parser copies IntrusivePtrs millions of times per parse (every cursor and
// every visited-set entry retains a stack), and a `lock xadd` per copy
// measured at ~20% of total parse time on a 22 KB input.
template <typename Derived>
struct Refcounted {
    mutable std::uint32_t _refcount{0};
};

template <typename T>
class Pool {
    struct alignas(T) Slot { char bytes[sizeof(T)]; };

    std::vector<std::unique_ptr<Slot[]>> chunks_;
    std::vector<T*> freelist_;
    std::size_t next_chunk_size_ = 64;
    std::size_t live_ = 0;          // objects handed out and not yet destroyed

    void grow() {
        auto chunk = std::make_unique<Slot[]>(next_chunk_size_);
        freelist_.reserve(freelist_.size() + next_chunk_size_);
        for (std::size_t i = 0; i < next_chunk_size_; ++i) {
            freelist_.push_back(reinterpret_cast<T*>(&chunk[i]));
        }
        chunks_.push_back(std::move(chunk));
        if (next_chunk_size_ < 8192) next_chunk_size_ *= 2;
    }

public:
    Pool() { freelist_.reserve(256); }

    // No copies — Pool owns raw memory with strict lifetime.
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    template <typename... Args>
    T* make(Args&&... args) {
        if (freelist_.empty()) grow();
        T* p = freelist_.back();
        freelist_.pop_back();
        ++live_;
        return ::new (p) T(std::forward<Args>(args)...);
    }

    void destroy(T* p) noexcept {
        p->~T();
        freelist_.push_back(p);
        --live_;
    }

    // Objects handed out and not yet returned.  Exposed so tests can assert
    // that a parse leaves nothing pinned.
    std::size_t live() const noexcept { return live_; }

    // A pool MUST outlive every object it handed out — and that is not
    // something scoping can guarantee here.  A pool-allocated object can end up
    // owned by something with static storage duration (a Parser kept as a
    // function-local static holds PList cursors; a parse tree can be cached),
    // and at process exit those are torn down in an order the pool has no say
    // over.  With a plain `thread_local Pool`, the pool is destroyed FIRST:
    // every straggler then runs its destructor on freed chunk memory and pushes
    // onto a destroyed freelist.  That is heap corruption at exit, and it is
    // exactly what surfaced when ftrace made this parser its .ftsl front end
    // (the Parser singleton's scratch outlived the pool).
    //
    // So ownership goes through a deleter that frees the pool only when nothing
    // is outstanding.  If objects are still live at thread exit we deliberately
    // leak the pool instead — a bounded, once-per-thread leak of that thread's
    // chunks — so the stragglers can still be destroyed safely afterwards.
    struct FreeOnlyIfDrained {
        void operator()(Pool* p) const noexcept {
            if (p && p->live_ == 0) delete p;
        }
    };

    static Pool& instance() noexcept {
        thread_local std::unique_ptr<Pool, FreeOnlyIfDrained> p(new Pool());
        return *p;
    }
};

// Intrusive smart pointer.  T must derive from Refcounted<T> and have
// `static void deallocate(T*)` that returns the object to its pool.
template <typename T>
class IntrusivePtr {
    T* ptr_;

    void retain() const noexcept {
        if (ptr_) ++ptr_->_refcount;
    }
    void release() noexcept {
        if (ptr_ && --ptr_->_refcount == 0)
            T::deallocate(ptr_);
        ptr_ = nullptr;
    }

public:
    IntrusivePtr() noexcept : ptr_(nullptr) {}
    IntrusivePtr(std::nullptr_t) noexcept : ptr_(nullptr) {}

    // Explicit adopt: wraps a newly-allocated T*, takes ownership (refcount
    // goes from 0 to 1).  Use this only with a freshly pool-allocated T.
    explicit IntrusivePtr(T* p) noexcept : ptr_(p) {
        if (ptr_) ++ptr_->_refcount;
    }

    IntrusivePtr(const IntrusivePtr& o) noexcept : ptr_(o.ptr_) { retain(); }

    IntrusivePtr(IntrusivePtr&& o) noexcept : ptr_(o.ptr_) {
        o.ptr_ = nullptr;
    }

    ~IntrusivePtr() { release(); }

    IntrusivePtr& operator=(const IntrusivePtr& o) noexcept {
        if (o.ptr_) ++o.ptr_->_refcount;
        release();
        ptr_ = o.ptr_;
        return *this;
    }

    IntrusivePtr& operator=(IntrusivePtr&& o) noexcept {
        if (this == &o) return *this;
        release();
        ptr_ = o.ptr_;
        o.ptr_ = nullptr;
        return *this;
    }

    T* get() const noexcept { return ptr_; }
    T* operator->() const noexcept { return ptr_; }
    T& operator*() const noexcept { return *ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    bool is_unique() const noexcept {
        return ptr_ && ptr_->_refcount == 1;
    }

    void reset() noexcept { release(); }

    friend bool operator==(const IntrusivePtr& a,
                           const IntrusivePtr& b) noexcept {
        return a.ptr_ == b.ptr_;
    }
    friend bool operator!=(const IntrusivePtr& a,
                           const IntrusivePtr& b) noexcept {
        return a.ptr_ != b.ptr_;
    }
};

}  // namespace gpda_pool
