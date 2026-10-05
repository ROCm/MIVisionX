/*
Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "ago_platform.h"
#include <roctracer/roctx.h>
#include <cstdio>

namespace AgoRocTx {
    // Runtime guard: read MIVISIONX_ROCPROF once per process. The function-local
    // static is initialized exactly once with C++11-guaranteed cross-thread
    // ordering, so no explicit atomics are needed. agoGetEnvironmentVariable is
    // used instead of std::getenv so the flag is honored on Windows, where the
    // CRT and Win32 keep separate environment blocks.
    inline bool enabled() {
        static const bool value = []() {
            char buf[16] = {0};
            if (!agoGetEnvironmentVariable("MIVISIONX_ROCPROF", buf, sizeof(buf)))
                return false;
            const char c = buf[0];
            return c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y';
        }();
        return value;
    }

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

// Helpers for generating unique identifiers per macro expansion so multiple
// ranges can coexist in the same scope without redefinition errors.
#define AGO_ROCTX_PASTE(a, b) a ## b
#define AGO_ROCTX_PASTE2(a, b) AGO_ROCTX_PASTE(a, b)
#define AGO_ROCTX_UNIQUE(prefix) AGO_ROCTX_PASTE2(prefix, __COUNTER__)

#define AGO_ROCTX_RANGE(name) AgoRocTx::Range AGO_ROCTX_UNIQUE(_ago_roctx_range_)(name)
#define AGO_ROCTX_MARK(msg)   do { if (AgoRocTx::enabled()) roctxMark(msg); } while(0)
#define AGO_ROCTX_PUSH(msg)   do { if (AgoRocTx::enabled()) roctxRangePush(msg); } while(0)
#define AGO_ROCTX_POP()       do { if (AgoRocTx::enabled()) roctxRangePop(); } while(0)

// Formatted variants that only pay for snprintf when tracing is compiled in
// and enabled at runtime. The buffer and RAII range share a unique prefix.
#define AGO_ROCTX_RANGE_FMT(...) \
    AGO_ROCTX_RANGE_FMT_IMPL(AGO_ROCTX_UNIQUE(_ago_roctx_fmt_), __VA_ARGS__)

#define AGO_ROCTX_RANGE_FMT_IMPL(prefix, ...) \
    char AGO_ROCTX_PASTE2(prefix, _buf)[256]; \
    AgoRocTx::Range AGO_ROCTX_PASTE2(prefix, _range)( \
        AgoRocTx::enabled() \
            ? (snprintf(AGO_ROCTX_PASTE2(prefix, _buf), sizeof(AGO_ROCTX_PASTE2(prefix, _buf)), __VA_ARGS__), \
               AGO_ROCTX_PASTE2(prefix, _buf)) \
            : nullptr)

#define AGO_ROCTX_MARK_FMT(...) \
    AGO_ROCTX_MARK_FMT_IMPL(AGO_ROCTX_UNIQUE(_ago_roctx_fmt_), __VA_ARGS__)

#define AGO_ROCTX_MARK_FMT_IMPL(prefix, ...) \
    do { \
        if (AgoRocTx::enabled()) { \
            char AGO_ROCTX_PASTE2(prefix, _buf)[256]; \
            snprintf(AGO_ROCTX_PASTE2(prefix, _buf), sizeof(AGO_ROCTX_PASTE2(prefix, _buf)), __VA_ARGS__); \
            roctxMark(AGO_ROCTX_PASTE2(prefix, _buf)); \
        } \
    } while(0)

#else

#define AGO_ROCTX_RANGE(name)      ((void)0)
#define AGO_ROCTX_RANGE_FMT(...)   ((void)0)
#define AGO_ROCTX_MARK(msg)        ((void)0)
#define AGO_ROCTX_MARK_FMT(...)    ((void)0)
#define AGO_ROCTX_PUSH(msg)        ((void)0)
#define AGO_ROCTX_POP()            ((void)0)

#endif // MIVISIONX_ENABLE_ROCPROF

#endif // __AGO_ROCTX_H__
