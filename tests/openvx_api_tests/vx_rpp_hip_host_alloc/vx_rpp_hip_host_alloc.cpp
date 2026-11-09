/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

// vx_rpp's HIP nodes keep per-batch ROI scratch buffers in pinned host memory. They are sized in
// elements of `unsigned`, so the byte count passed to hipHostMalloc must include sizeof(unsigned).
// This program interposes hipHostMalloc, builds the Transpose and Normalize nodes on the GPU and
// checks that every pinned allocation they make is large enough for the values they store in it.

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <VX/vx.h>
#include <vx_ext_amd.h>
#include <vx_ext_rpp.h>
#include <cstdio>
#include <cstring>
#include <vector>

struct Alloc {
    void *ptr;
    size_t size;
};
static std::vector<Alloc> gAllocs;

// The executable's definition is found before the HIP runtime's for calls made by vx_rpp.
extern "C" hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
    using Fn = hipError_t (*)(void **, size_t, unsigned int);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipHostMalloc"));
    hipError_t status = real(ptr, size, flags);
    if (status == hipSuccess) gAllocs.push_back({*ptr, size});
    return status;
}

static const size_t N = 16, H = 4, W = 4, C = 3;
static const vx_int32 NHWC = 0;
static const vx_int32 XYWH = 1;

static vx_tensor makeTensor(vx_context ctx, vx_enum type, size_t d0, size_t d1, size_t d2, size_t d3) {
    size_t dims[4] = {d0, d1, d2, d3};
    return vxCreateTensor(ctx, 4, dims, type, 0);
}

static vx_tensor makeRoi(vx_context ctx) {
    size_t dims[2] = {N, 4};
    return vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT32, 0);
}

static vx_scalar i32(vx_context c, vx_int32 v) { return vxCreateScalar(c, VX_TYPE_INT32, &v); }

// The node allocates the buffer in bytes of unsigned values, so some allocation made while building it must
// be at least `needed` bytes (and well below RPP's own multi-megabyte scratch buffers).
static int checkAllocation(const char *what, size_t first, size_t needed) {
    bool found = false;
    for (size_t i = first; i < gAllocs.size(); i++)
        if (gAllocs[i].size >= needed && gAllocs[i].size < 4 * needed) found = true;
    printf("%s: pinned ROI buffer of at least %zu bytes: %s\n", what, needed, found ? "PASS" : "FAIL");
    return found ? 0 : 1;
}

int main() {
    int failures = 0;
    vx_context ctx = vxCreateContext();
    AgoTargetAffinityInfo aff;
    memset(&aff, 0, sizeof(aff));
    aff.device_type = AGO_TARGET_AFFINITY_GPU;
    vxSetContextAttribute(ctx, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &aff, sizeof(aff));
    if (vxLoadKernels(ctx, "vx_rpp") != VX_SUCCESS) {
        printf("vxLoadKernels(vx_rpp) failed\n");
        return 1;
    }

    // Transpose, NHWC: pSrcRoi holds N * (H, W, C ROI dims) * (begin, end) unsigned values
    {
        size_t first = gAllocs.size();
        vx_graph g = vxCreateGraph(ctx);
        vx_tensor src = makeTensor(ctx, VX_TYPE_UINT8, N, H, W, C), dst = makeTensor(ctx, VX_TYPE_UINT8, N, W, H, C);
        vx_array perm = vxCreateArray(ctx, VX_TYPE_UINT32, 3);
        vx_uint32 p[3] = {1, 0, 2};
        vxAddArrayItems(perm, 3, p, sizeof(vx_uint32));
        vx_node n = vxExtRppTranspose(g, src, makeRoi(ctx), dst, perm, i32(ctx, NHWC), i32(ctx, NHWC), i32(ctx, XYWH));
        if (vxGetStatus((vx_reference)n) != VX_SUCCESS || vxVerifyGraph(g) != VX_SUCCESS || vxProcessGraph(g) != VX_SUCCESS) {
            printf("Transpose: graph could not be built or run\n");
            return 1;
        }
        failures += checkAllocation("Transpose", first, N * 3 * 2 * sizeof(unsigned));
        vxReleaseGraph(&g);
    }

    // Normalize, NHWC float: pSrcDims holds N * (H, W, C ROI dims) * (begin, end) unsigned values
    {
        size_t first = gAllocs.size();
        vx_graph g = vxCreateGraph(ctx);
        vx_tensor src = makeTensor(ctx, VX_TYPE_FLOAT32, N, H, W, C), dst = makeTensor(ctx, VX_TYPE_FLOAT32, N, H, W, C);
        vx_array mean = vxCreateArray(ctx, VX_TYPE_FLOAT32, N), stddev = vxCreateArray(ctx, VX_TYPE_FLOAT32, N);
        std::vector<vx_float32> ones(N, 1.0f);
        vxAddArrayItems(mean, N, ones.data(), sizeof(vx_float32));
        vxAddArrayItems(stddev, N, ones.data(), sizeof(vx_float32));
        vx_uint8 compute = 3;
        vx_float32 scale = 1.0f, shift = 0.0f;
        vx_node n = vxExtRppNormalize(g, src, makeRoi(ctx), dst, makeRoi(ctx), i32(ctx, 7), mean, stddev,
                                      vxCreateScalar(ctx, VX_TYPE_UINT8, &compute), vxCreateScalar(ctx, VX_TYPE_FLOAT32, &scale),
                                      vxCreateScalar(ctx, VX_TYPE_FLOAT32, &shift), i32(ctx, NHWC), i32(ctx, XYWH));
        if (vxGetStatus((vx_reference)n) != VX_SUCCESS || vxVerifyGraph(g) != VX_SUCCESS || vxProcessGraph(g) != VX_SUCCESS) {
            printf("Normalize: graph could not be built or run\n");
            return 1;
        }
        failures += checkAllocation("Normalize", first, N * 3 * 2 * sizeof(unsigned));
        vxReleaseGraph(&g);
    }

    vxReleaseContext(&ctx);
    printf("vx_rpp HIP pinned allocation test: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
