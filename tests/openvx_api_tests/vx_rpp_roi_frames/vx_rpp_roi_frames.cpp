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

// vx_rpp video layouts (NFHWC): every node expands the N per-sample ROIs to N*F per-frame ROIs.
// The ROI tensors are graph inputs, so they must come back from vxProcessGraph unchanged and
// repeated executions of the same graph must give the same output.
// GPU only: the CPU path of these nodes is not covered here.

#include <hip/hip_runtime_api.h>
#include <VX/vx.h>
#include <vx_ext_amd.h>
#include <vx_ext_rpp.h>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

static const size_t N = 2, F = 3, H = 8, W = 16, C = 3;
static const vx_int32 NFHWC = 2, XYWH = 1;
static const int RUNS = 3;

// the tensor's own device buffer, which is what vx_rpp reads
static void *deviceBuffer(vx_tensor t) {
    void *p = nullptr;
    vxQueryTensor(t, VX_TENSOR_BUFFER_HIP, &p, sizeof(p));
    return p;
}

struct RoiTensor {
    vx_tensor tensor;
    std::vector<vx_uint32> initial;  // N*F entries; only the first N are meaningful
};

static RoiTensor makeRoiTensor(vx_context ctx, vx_uint32 a[4], vx_uint32 b[4]) {
    size_t dims[2] = {N * F, 4};  // room for the N*F expansion, so the old code does not overflow the tensor
    RoiTensor r;
    r.tensor = vxCreateTensor(ctx, 2, dims, VX_TYPE_UINT32, 0);
    r.initial.assign(4 * N * F, 0);
    memcpy(&r.initial[0], a, 16);
    memcpy(&r.initial[4], b, 16);
    return r;
}

struct Case {
    const char *name;
    // builds the node; fills rois with the ROI tensors the node takes (all must stay unchanged)
    std::function<vx_node(vx_context, vx_graph, vx_tensor, vx_tensor, std::vector<RoiTensor> &)> build;
};

static vx_scalar i32(vx_context c, vx_int32 v) { return vxCreateScalar(c, VX_TYPE_INT32, &v); }
static vx_scalar u32(vx_context c, vx_uint32 v) { return vxCreateScalar(c, VX_TYPE_UINT32, &v); }
static vx_scalar f32(vx_context c, vx_float32 v) { return vxCreateScalar(c, VX_TYPE_FLOAT32, &v); }

static vx_array u32Array(vx_context c, vx_uint32 v) {
    vx_array a = vxCreateArray(c, VX_TYPE_UINT32, N);
    std::vector<vx_uint32> vals(N, v);
    vxAddArrayItems(a, N, vals.data(), sizeof(vx_uint32));
    return a;
}

// returns 0 pass, 1 fail, 2 skipped (this RPP build cannot run the node)
static int runCase(const Case &c) {
    vx_context ctx = vxCreateContext();
    AgoTargetAffinityInfo aff;
    memset(&aff, 0, sizeof(aff));
    aff.device_type = AGO_TARGET_AFFINITY_GPU;
    vxSetContextAttribute(ctx, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &aff, sizeof(aff));
    if (vxLoadKernels(ctx, "vx_rpp") != VX_SUCCESS) {
        printf("%s: vxLoadKernels(vx_rpp) failed\n", c.name);
        vxReleaseContext(&ctx);
        return 2;
    }
    size_t dims[5] = {N, F, H, W, C};
    vx_tensor src = vxCreateTensor(ctx, 5, dims, VX_TYPE_UINT8, 0);
    vx_tensor dst = vxCreateTensor(ctx, 5, dims, VX_TYPE_UINT8, 0);
    vx_graph graph = vxCreateGraph(ctx);
    std::vector<RoiTensor> rois;
    vx_node node = c.build(ctx, graph, src, dst, rois);
    if (vxGetStatus((vx_reference)node) != VX_SUCCESS || vxVerifyGraph(graph) != VX_SUCCESS) {
        printf("%s: graph setup failed - SKIPPED\n", c.name);
        vxReleaseContext(&ctx);
        return 2;
    }
    // device buffers exist once the graph is verified
    std::vector<vx_uint8> in(N * F * H * W * C), out(in.size()), first;
    for (size_t i = 0; i < in.size(); i++) in[i] = (vx_uint8)((i / C) % W);
    (void)hipMemcpy(deviceBuffer(src), in.data(), in.size(), hipMemcpyHostToDevice);
    for (auto &r : rois) (void)hipMemcpy(deviceBuffer(r.tensor), r.initial.data(), r.initial.size() * 4, hipMemcpyHostToDevice);

    int bad = 0;
    for (int run = 1; run <= RUNS; run++) {
        if (vxProcessGraph(graph) != VX_SUCCESS) {
            printf("%s: vxProcessGraph failed - SKIPPED\n", c.name);
            vxReleaseContext(&ctx);
            return 2;
        }
        (void)hipMemcpy(out.data(), deviceBuffer(dst), out.size(), hipMemcpyDeviceToHost);
        for (size_t k = 0; k < rois.size(); k++) {
            std::vector<vx_uint32> now(rois[k].initial.size());
            (void)hipMemcpy(now.data(), deviceBuffer(rois[k].tensor), now.size() * 4, hipMemcpyDeviceToHost);
            if (now != rois[k].initial) {
                printf("%s: run %d: ROI tensor %zu was modified: slot 0 {%u,%u,%u,%u} slot 1 {%u,%u,%u,%u} slot 3 {%u,%u,%u,%u}\n",
                       c.name, run, k, now[0], now[1], now[2], now[3], now[4], now[5], now[6], now[7], now[12], now[13], now[14], now[15]);
                bad++;
            }
        }
        if (run == 1) {
            first = out;
        } else if (out != first) {
            printf("%s: run %d: output differs from run 1\n", c.name, run);
            bad++;
        }
    }
    printf("%s: %s\n", c.name, bad ? "FAIL" : "PASS");
    vxReleaseContext(&ctx);
    return bad ? 1 : 0;
}

int main() {
    vx_uint32 A[4] = {0, 0, (vx_uint32)W, (vx_uint32)H}, B[4] = {0, 0, (vx_uint32)(W / 2), (vx_uint32)H};
    vx_uint32 CA[4] = {0, 0, 8, 4}, CB[4] = {4, 2, 8, 4};

    std::vector<Case> cases;
    cases.push_back({"Flip", [&](vx_context ctx, vx_graph g, vx_tensor src, vx_tensor dst, std::vector<RoiTensor> &rois) {
        rois.push_back(makeRoiTensor(ctx, A, B));
        return vxExtRppFlip(g, src, rois[0].tensor, dst, u32Array(ctx, 1), u32Array(ctx, 0), u32Array(ctx, 0),
                            i32(ctx, NFHWC), i32(ctx, NFHWC), i32(ctx, XYWH));
    }});
    cases.push_back({"GridMask", [&](vx_context ctx, vx_graph g, vx_tensor src, vx_tensor dst, std::vector<RoiTensor> &rois) {
        rois.push_back(makeRoiTensor(ctx, A, B));
        return vxExtRppGridMask(g, src, rois[0].tensor, dst, u32(ctx, 4), f32(ctx, 0.5f), f32(ctx, 0.0f), u32(ctx, 0), u32(ctx, 0),
                                i32(ctx, NFHWC), i32(ctx, NFHWC), i32(ctx, XYWH));
    }});
    cases.push_back({"CropAndPatch", [&](vx_context ctx, vx_graph g, vx_tensor src, vx_tensor dst, std::vector<RoiTensor> &rois) {
        size_t dims[5] = {N, F, H, W, C};
        vx_tensor src2 = vxCreateTensor(ctx, 5, dims, VX_TYPE_UINT8, 0);
        rois.push_back(makeRoiTensor(ctx, CA, CB));
        rois.push_back(makeRoiTensor(ctx, CA, CB));
        rois.push_back(makeRoiTensor(ctx, CA, CB));
        return vxExtRppCropAndPatch(g, src, src2, dst, rois[0].tensor, rois[1].tensor, rois[2].tensor,
                                    i32(ctx, NFHWC), i32(ctx, NFHWC), i32(ctx, XYWH));
    }});

    int failed = 0, ran = 0;
    for (const Case &c : cases) {
        int r = runCase(c);
        if (r != 2) ran++;
        if (r == 1) failed++;
    }
    if (ran == 0) {
        printf("no case could run\n");
        return 1;
    }
    printf("vx_rpp ROI frames test: %d of %d cases ran, %d failed\n", ran, (int)cases.size(), failed);
    return failed ? 1 : 0;
}
