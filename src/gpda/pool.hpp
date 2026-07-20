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
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace gpda_pool {

// Base for pool-allocated, intrusively-refcounted types.
template <typename Derived>
struct Refcounted {
    mutable std::atomic<std::uint32_t> _refcount{0};
};

template <typename T>
class Pool {
    struct alignas(T) Slot { char bytes[sizeof(T)]; };

    std::vector<std::unique_ptr<Slot[]>> chunks_;
    std::vector<T*> freelist_;
    std::size_t next_chunk_size_ = 64;

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
        return ::new (p) T(std::forward<Args>(args)...);
    }

    void destroy(T* p) noexcept {
        p->~T();
        freelist_.push_back(p);
    }

    static Pool& instance() noexcept {
        thread_local Pool p;
        return p;
    }
};

// Intrusive smart pointer.  T must derive from Refcounted<T> and have
// `static void deallocate(T*)` that returns the object to its pool.
template <typename T>
class IntrusivePtr {
    T* ptr_;

    void retain() const noexcept {
        if (ptr_) ptr_->_refcount.fetch_add(1, std::memory_order_relaxed);
    }
    void release() noexcept {
        if (ptr_ && ptr_->_refcount.fetch_sub(1,
                                              std::memory_order_acq_rel) == 1)
            T::deallocate(ptr_);
        ptr_ = nullptr;
    }

public:
    IntrusivePtr() noexcept : ptr_(nullptr) {}
    IntrusivePtr(std::nullptr_t) noexcept : ptr_(nullptr) {}

    // Explicit adopt: wraps a newly-allocated T*, takes ownership (refcount
    // goes from 0 to 1).  Use this only with a freshly pool-allocated T.
    explicit IntrusivePtr(T* p) noexcept : ptr_(p) {
        if (ptr_) ptr_->_refcount.fetch_add(1, std::memory_order_relaxed);
    }

    IntrusivePtr(const IntrusivePtr& o) noexcept : ptr_(o.ptr_) { retain(); }

    IntrusivePtr(IntrusivePtr&& o) noexcept : ptr_(o.ptr_) {
        o.ptr_ = nullptr;
    }

    ~IntrusivePtr() { release(); }

    IntrusivePtr& operator=(const IntrusivePtr& o) noexcept {
        if (o.ptr_) o.ptr_->_refcount.fetch_add(1, std::memory_order_relaxed);
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
        return ptr_ && ptr_->_refcount.load(std::memory_order_relaxed) == 1;
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
