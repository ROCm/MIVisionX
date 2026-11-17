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

#ifndef __RUNVX_PLUGIN_HASH_H__
#define __RUNVX_PLUGIN_HASH_H__

#include <cstdint>
#include <cstddef>

// Cryptographic hash plugin interface for runvx.
// The core runvx binary loads the plugin at runtime to compute MD5 checksums.
// Implementations live in separate shared libraries (e.g., runvx_hash_openssl.so).

class IRunVxHashPlugin {
public:
    virtual ~IRunVxHashPlugin() = default;

    // Create a per-hash context. Returns nullptr on failure.
    virtual void * Create() = 0;

    // Destroy a context previously returned by Create().
    virtual void Destroy(void * ctx) = 0;

    // Feed data into the hash context.
    virtual bool Update(void * ctx, const uint8_t * data, size_t len) = 0;

    // Finalize and write the 16-byte digest into 'out'.
    virtual bool Final(void * ctx, uint8_t out[16]) = 0;
};

// Plugin factory exported by the shared library.
extern "C" {
    typedef IRunVxHashPlugin * (*RunVxCreateHashPluginFn)();
#define RUNVX_HASH_PLUGIN_FACTORY "RunVxCreateHashPlugin"
}

#endif /* __RUNVX_PLUGIN_HASH_H__ */
