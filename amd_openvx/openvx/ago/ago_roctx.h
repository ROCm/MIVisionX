/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef __AGO_ROCTX_H__
#define __AGO_ROCTX_H__

// Optional rocprof/ROCTX tracing support for AMD OpenVX.
// Enabled at build time with -DMIVISIONX_ENABLE_ROCPROF=ON.
// At runtime, set MIVISIONX_ROCPROF=1 to emit markers/ranges.
// When not tracing, the macros compile away or check a single bool flag.

#if defined(MIVISIONX_ENABLE_ROCPROF)

#include <roctracer/roctx.h>
#include <atomic>
#include <cstdlib>

namespace AgoRocTx {
    // Runtime guard: read MIVISIONX_ROCPROF once per process.
    // 0 = uninitialized, 1 = enabled, 2 = disabled.
    inline int getEnabledState() {
        static std::atomic<int> state{0};
        int s = state.load(std::memory_order_relaxed);
        if (s != 0) return s;
        const char* env = std::getenv("MIVISIONX_ROCPROF");
        s = (env && (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' || env[0] == 'Y')) ? 1 : 2;
        state.store(s, std::memory_order_relaxed);
        return s;
    }
    inline bool enabled() { return getEnabledState() == 1; }

    // Lightweight RAII scope guard for a range.
    // Usage: AgoRocTx::Range range("my range");
    class Range {
    public:
        explicit Range(const char* msg) : active_(enabled()) {
            if (active_) roctxRangePush(msg);
        }
        ~Range() {
            if (active_) roctxRangePop();
        }
        // Non-copyable, non-movable to keep stack balanced.
        Range(const Range&) = delete;
        Range& operator=(const Range&) = delete;
        Range(Range&&) = delete;
        Range& operator=(Range&&) = delete;
    private:
        bool active_;
    };
}

#define AGO_ROCTX_RANGE(name) AgoRocTx::Range _ago_roctx_range(name)
#define AGO_ROCTX_MARK(msg)        do { if (AgoRocTx::enabled()) roctxMark(msg); } while(0)
#define AGO_ROCTX_PUSH(msg)        do { if (AgoRocTx::enabled()) roctxRangePush(msg); } while(0)
#define AGO_ROCTX_POP()            do { if (AgoRocTx::enabled()) roctxRangePop(); } while(0)

#else

#define AGO_ROCTX_RANGE(name)      ((void)0)
#define AGO_ROCTX_MARK(msg)        ((void)0)
#define AGO_ROCTX_PUSH(msg)        ((void)0)
#define AGO_ROCTX_POP()            ((void)0)

#endif // MIVISIONX_ENABLE_ROCPROF

#endif // __AGO_ROCTX_H__
