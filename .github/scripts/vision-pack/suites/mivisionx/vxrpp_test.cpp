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

// vx_rpp functional driver: one graph per case with the public vxExtRpp* API on
// a 2x120x160x3 U8 NHWC tensor. The graph is executed twice; the inputs, the
// first-execution output (<case>.run1.bin) and the second (<case>.bin) are
// written to <outdir> for vxrpp.py to compare.
//   usage: vxrpp_test <CPU|GPU> <outdir> <case|list>
#include <VX/vx.h>
#include <vx_ext_amd.h>
#include <vx_ext_rpp.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

static vx_context ctx;
static const size_t N = 2, H = 120, W = 160, C = 3;
static const int VX_NHWC = 0;   // vxTensorLayout (internal_rpp.h; not in the public header)
static const int ROI_XYWH = 1;  // RpptRoiType

struct Tensor {
    vx_tensor t = nullptr;
    std::vector<size_t> dims;
    size_t es = 1;
    size_t bytes() const { size_t n = es; for (auto d : dims) n *= d; return n; }
};

static void strides_for(const std::vector<size_t>& dims, size_t es, std::vector<size_t>& st) {
    st.resize(dims.size());
    st[0] = es;
    for (size_t i = 1; i < dims.size(); i++) st[i] = st[i - 1] * dims[i - 1];
}

static Tensor mk(std::vector<size_t> dims, vx_enum type, size_t es) {
    Tensor T;
    T.dims = dims;
    T.es = es;
    T.t = vxCreateTensor(ctx, dims.size(), dims.data(), type, 0);
    return T;
}

static vx_status put(Tensor& T, const void* data) {
    std::vector<size_t> start(T.dims.size(), 0), st;
    strides_for(T.dims, T.es, st);
    return vxCopyTensorPatch(T.t, T.dims.size(), start.data(), T.dims.data(), st.data(), (void*)data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
}

static vx_status get(Tensor& T, std::vector<uint8_t>& out) {
    out.assign(T.bytes(), 0);
    std::vector<size_t> start(T.dims.size(), 0), st;
    strides_for(T.dims, T.es, st);
    return vxCopyTensorPatch(T.t, T.dims.size(), start.data(), T.dims.data(), st.data(), out.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
}

static void dump(const std::string& path, const std::vector<uint8_t>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (f) { fwrite(v.data(), 1, v.size(), f); fclose(f); }
}

static vx_array aF(std::vector<float> v) { vx_array a = vxCreateArray(ctx, VX_TYPE_FLOAT32, v.size()); vxAddArrayItems(a, v.size(), v.data(), sizeof(float)); return a; }
static vx_array aU(std::vector<uint32_t> v) { vx_array a = vxCreateArray(ctx, VX_TYPE_UINT32, v.size()); vxAddArrayItems(a, v.size(), v.data(), sizeof(uint32_t)); return a; }
static vx_array aI(std::vector<int32_t> v) { vx_array a = vxCreateArray(ctx, VX_TYPE_INT32, v.size()); vxAddArrayItems(a, v.size(), v.data(), sizeof(int32_t)); return a; }
static vx_array aU8(std::vector<uint8_t> v) { vx_array a = vxCreateArray(ctx, VX_TYPE_UINT8, v.size()); vxAddArrayItems(a, v.size(), v.data(), sizeof(uint8_t)); return a; }
static vx_scalar sI(int32_t v) { return vxCreateScalar(ctx, VX_TYPE_INT32, &v); }
static vx_scalar sU(uint32_t v) { return vxCreateScalar(ctx, VX_TYPE_UINT32, &v); }
static vx_scalar sF(float v) { return vxCreateScalar(ctx, VX_TYPE_FLOAT32, &v); }

struct Env {
    vx_graph g;
    Tensor in, in2, roi, out;
    vx_scalar inL, outL, roiT;
};

typedef std::function<vx_node(Env&)> Builder;
struct Case { std::vector<size_t> outDims; Builder build; };

static std::map<std::string, Case> cases() {
    std::map<std::string, Case> m;
    std::vector<size_t> same = {N, H, W, C};
#define ADD(name, dims, body) m[name] = Case{dims, [](Env& e) -> vx_node body}
    ADD("Nop", same, { return vxExtRppNop(e.g, e.in.t, e.out.t); });
    ADD("Copy", same, { return vxExtRppCopy(e.g, e.in.t, e.out.t); });
    ADD("Brightness", same, { return vxExtRppBrightness(e.g, e.in.t, e.roi.t, e.out.t, aF({1.2f, 0.8f}), aF({10.f, -20.f}), aI({1, 1}), e.inL, e.outL, e.roiT); });
    ADD("Contrast", same, { return vxExtRppContrast(e.g, e.in.t, e.roi.t, e.out.t, aF({1.5f, 0.7f}), aF({128.f, 100.f}), e.inL, e.outL, e.roiT); });
    ADD("GammaCorrection", same, { return vxExtRppGammaCorrection(e.g, e.in.t, e.roi.t, e.out.t, aF({0.8f, 1.5f}), e.inL, e.outL, e.roiT); });
    ADD("Exposure", same, { return vxExtRppExposure(e.g, e.in.t, e.roi.t, e.out.t, aF({0.5f, 1.2f}), e.inL, e.outL, e.roiT); });
    ADD("Saturation", same, { return vxExtRppSaturation(e.g, e.in.t, e.roi.t, e.out.t, aF({1.3f, 0.5f}), e.inL, e.outL, e.roiT); });
    ADD("Hue", same, { return vxExtRppHue(e.g, e.in.t, e.roi.t, e.out.t, aF({30.f, -60.f}), e.inL, e.outL, e.roiT); });
    ADD("ColorTwist", same, { return vxExtRppColorTwist(e.g, e.in.t, e.roi.t, e.out.t, aF({1.1f, 0.9f}), aF({5.f, -5.f}), aF({20.f, -20.f}), aF({1.2f, 0.8f}), e.inL, e.outL, e.roiT); });
    ADD("ColorJitter", same, { return vxExtRppColorJitter(e.g, e.in.t, e.roi.t, e.out.t, aF({1.1f, 0.9f}), aF({1.2f, 0.8f}), aF({10.f, -10.f}), aF({1.1f, 0.9f}), e.inL, e.outL, e.roiT); });
    ADD("ColorTemperature", same, { return vxExtRppColorTemperature(e.g, e.in.t, e.roi.t, e.out.t, aI({20, -30}), e.inL, e.outL, e.roiT); });
    ADD("Flip", same, { return vxExtRppFlip(e.g, e.in.t, e.roi.t, e.out.t, aU({1, 0}), aU({0, 1}), aU({0, 0}), e.inL, e.outL, e.roiT); });
    ADD("Resize_Bilinear", (std::vector<size_t>{N, 60, 80, C}), { return vxExtRppResize(e.g, e.in.t, e.roi.t, e.out.t, aU({80, 80}), aU({60, 60}), sI(1), e.inL, e.outL, e.roiT); });
    ADD("Resize_Nearest", (std::vector<size_t>{N, 60, 80, C}), { return vxExtRppResize(e.g, e.in.t, e.roi.t, e.out.t, aU({80, 80}), aU({60, 60}), sI(0), e.inL, e.outL, e.roiT); });
    ADD("Resize_Bicubic", (std::vector<size_t>{N, 60, 80, C}), { return vxExtRppResize(e.g, e.in.t, e.roi.t, e.out.t, aU({80, 80}), aU({60, 60}), sI(2), e.inL, e.outL, e.roiT); });
    ADD("Resize_Up_Bilinear", (std::vector<size_t>{N, 180, 240, C}), { return vxExtRppResize(e.g, e.in.t, e.roi.t, e.out.t, aU({240, 240}), aU({180, 180}), sI(1), e.inL, e.outL, e.roiT); });
    ADD("Rotate", same, { return vxExtRppRotate(e.g, e.in.t, e.roi.t, e.out.t, aF({30.f, -45.f}), sI(1), e.inL, e.outL, e.roiT); });
    ADD("WarpAffine", same, { return vxExtRppWarpAffine(e.g, e.in.t, e.roi.t, e.out.t, aF({1.0f, 0.1f, 5.f, 0.1f, 1.0f, -3.f, 0.9f, -0.1f, 2.f, 0.05f, 1.1f, 4.f}), sI(1), e.inL, e.outL, e.roiT); });
    ADD("WarpPerspective", same, { return vxExtRppWarpPerspective(e.g, e.in.t, e.roi.t, e.out.t, aF({1.f, 0.05f, 2.f, 0.02f, 1.f, 3.f, 0.0001f, 0.0002f, 1.f, 0.95f, 0.f, 1.f, 0.f, 1.05f, -2.f, 0.f, 0.0001f, 1.f}), sI(1), e.inL, e.outL, e.roiT); });
    ADD("Blur", same, { return vxExtRppBlur(e.g, e.in.t, e.roi.t, e.out.t, e.inL, e.outL, e.roiT); });
    ADD("Erode3", same, { return vxExtRppErode(e.g, e.in.t, e.roi.t, e.out.t, sU(3), e.inL, e.outL, e.roiT); });
    ADD("Dilate3", same, { return vxExtRppDilate(e.g, e.in.t, e.roi.t, e.out.t, sU(3), e.inL, e.outL, e.roiT); });
    ADD("MedianFilter3", same, { return vxExtRppMedianFilter(e.g, e.in.t, e.roi.t, e.out.t, sU(3), sI(0), e.inL, e.outL, e.roiT); });
    ADD("GaussianFilter5", same, { return vxExtRppGaussianFilter(e.g, e.in.t, e.roi.t, e.out.t, aF({1.0f, 2.0f}), sU(5), sI(0), e.inL, e.outL, e.roiT); });
    ADD("ColorToGreyscale_NHWC1", (std::vector<size_t>{N, H, W, 1}), { return vxExtRppColorToGreyscale(e.g, e.in.t, e.roi.t, e.out.t, sI(0), e.inL, e.outL, e.roiT); });
    ADD("ColorToGreyscale_NCHW1", (std::vector<size_t>{N, 1, H, W}), { return vxExtRppColorToGreyscale(e.g, e.in.t, e.roi.t, e.out.t, sI(0), e.inL, sI(1), e.roiT); });
    ADD("Pixelate", same, { return vxExtRppPixelate(e.g, e.in.t, e.roi.t, e.out.t, sF(20.f), e.inL, e.outL, e.roiT); });
    ADD("Vignette", same, { return vxExtRppVignette(e.g, e.in.t, e.roi.t, e.out.t, aF({50.f, 80.f}), e.inL, e.outL, e.roiT); });
    ADD("Posterize", same, { return vxExtRppPosterize(e.g, e.in.t, e.roi.t, e.out.t, aU8({4, 2}), e.inL, e.outL, e.roiT); });
    ADD("Solarize", same, { return vxExtRppSolarize(e.g, e.in.t, e.roi.t, e.out.t, aF({0.5f, 0.3f}), e.inL, e.outL, e.roiT); });
    ADD("Threshold", same, { return vxExtRppThreshold(e.g, e.in.t, e.roi.t, e.out.t, aF({50.f, 60.f, 70.f, 100.f, 110.f, 120.f}), aF({200.f, 190.f, 180.f, 180.f, 170.f, 160.f}), e.inL, e.outL, e.roiT); });
    ADD("Blend", same, { return vxExtRppBlend(e.g, e.in.t, e.in2.t, e.roi.t, e.out.t, aF({0.3f, 0.7f}), e.inL, e.outL, e.roiT); });
    ADD("BitwiseAnd", same, { return vxExtRppBitwiseOps(e.g, e.in.t, e.in2.t, e.roi.t, e.out.t, e.inL, e.outL, e.roiT, sI(0)); });
    ADD("BitwiseOr", same, { return vxExtRppBitwiseOps(e.g, e.in.t, e.in2.t, e.roi.t, e.out.t, e.inL, e.outL, e.roiT, sI(1)); });
    ADD("BitwiseXor", same, { return vxExtRppBitwiseOps(e.g, e.in.t, e.in2.t, e.roi.t, e.out.t, e.inL, e.outL, e.roiT, sI(2)); });
    ADD("Magnitude", same, { return vxExtRppMagnitude(e.g, e.in.t, e.in2.t, e.roi.t, e.out.t, e.inL, e.outL, e.roiT); });
    ADD("Phase", same, { return vxExtRppPhase(e.g, e.in.t, e.in2.t, e.roi.t, e.out.t, e.inL, e.outL, e.roiT); });
    ADD("CropMirrorNormalize", same, { return vxExtRppCropMirrorNormalize(e.g, e.in.t, e.roi.t, e.out.t, aF({1.f, 1.f, 1.f, 0.5f, 0.5f, 0.5f}), aF({0.f, 0.f, 0.f, 10.f, 20.f, 30.f}), aU({1, 0}), e.inL, e.outL, e.roiT); });
    ADD("Crop", same, { return vxExtRppCrop(e.g, e.in.t, e.roi.t, e.out.t, e.inL, e.outL, e.roiT); });
    ADD("ChannelPermute", same, { return vxExtRppChannelPermute(e.g, e.in.t, e.out.t, aU({2, 1, 0, 1, 2, 0}), e.inL, e.outL); });
    ADD("FishEye", same, { return vxExtRppFishEye(e.g, e.in.t, e.roi.t, e.out.t, e.inL, e.outL, e.roiT); });
    ADD("LensCorrection", same, { return vxExtRppLensCorrection(e.g, e.in.t, e.roi.t, e.out.t,
        aF({150.f, 0.f, 80.f, 0.f, 150.f, 60.f, 0.f, 0.f, 1.f, 120.f, 0.f, 80.f, 0.f, 120.f, 60.f, 0.f, 0.f, 1.f}),
        aF({-0.2f, 0.05f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.1f, -0.02f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}), e.inL, e.outL, e.roiT); });
    ADD("Water", same, { return vxExtRppWater(e.g, e.in.t, e.roi.t, e.out.t, aF({2.f, 3.f}), aF({3.f, 2.f}), aF({0.05f, 0.08f}), aF({0.08f, 0.05f}), aF({0.f, 1.f}), aF({1.f, 0.f}), e.inL, e.outL, e.roiT); });
    ADD("Glitch", same, { return vxExtRppGlitch(e.g, e.in.t, e.roi.t, e.out.t, aU({5, 3}), aU({2, 4}), aU({0, 1}), aU({1, 0}), aU({3, 5}), aU({4, 2}), e.inL, e.outL, e.roiT); });
    ADD("Fog", same, { return vxExtRppFog(e.g, e.in.t, e.roi.t, e.out.t, aF({0.5f, 0.3f}), aF({0.3f, 0.5f}), e.inL, e.outL, e.roiT); });
    ADD("JpegCompressionDistortion", same, { return vxExtRppJpegCompressionDistortion(e.g, e.in.t, e.roi.t, e.out.t, aI({50, 80}), e.inL, e.outL, e.roiT); });
    ADD("Snow", same, { return vxExtRppSnow(e.g, e.in.t, e.roi.t, e.out.t, aF({2.0f, 2.5f}), aF({0.2f, 0.3f}), aI({0, 1}), e.inL, e.outL, e.roiT); });
    ADD("GridMask", same, { return vxExtRppGridMask(e.g, e.in.t, e.roi.t, e.out.t, sU(40), sF(0.6f), sF(0.5f), sU(0), sU(0), e.inL, e.outL, e.roiT); });
    // stochastic kernels: execution only
    ADD("rng_Noise", same, { return vxExtRppNoise(e.g, e.in.t, e.roi.t, e.out.t, aF({0.1f, 0.2f}), aF({0.5f, 0.5f}), aF({1.f, 1.f}), aF({0.f, 0.f}), sU(1234), e.inL, e.outL, e.roiT); });
    ADD("rng_GaussianNoise", same, { return vxExtRppGaussianNoise(e.g, e.in.t, e.roi.t, e.out.t, aF({0.f, 0.f}), aF({0.1f, 0.2f}), aI({1, 1}), sU(1234), e.inL, e.outL, e.roiT); });
    ADD("rng_ShotNoise", same, { return vxExtRppShotNoise(e.g, e.in.t, e.roi.t, e.out.t, aF({10.f, 20.f}), sU(1234), e.inL, e.outL, e.roiT); });
    ADD("rng_Jitter", same, { return vxExtRppJitter(e.g, e.in.t, e.roi.t, e.out.t, aU({3, 5}), sI(1234), e.inL, e.outL, e.roiT); });
    ADD("rng_Rain", same, { return vxExtRppRain(e.g, e.in.t, e.roi.t, e.out.t, sF(10.f), sU(1), sU(8), sF(-10.f), aF({0.4f, 0.6f}), e.inL, e.outL, e.roiT); });
    ADD("rng_Spatter", same, { return vxExtRppSpatter(e.g, e.in.t, e.roi.t, e.out.t, aU8({65, 50, 23}), e.inL, e.outL, e.roiT); });
#undef ADD
    return m;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <CPU|GPU> <outdir> <case|list>\n", argv[0]); return 2; }
    auto all = cases();
    std::string which = argv[3], dir = argv[2];
    if (which == "list") { for (auto& kv : all) printf("%s\n", kv.first.c_str()); return 0; }
    if (!all.count(which)) { fprintf(stderr, "unknown case %s\n", which.c_str()); return 2; }
    bool gpu = std::string(argv[1]) == "GPU";

    ctx = vxCreateContext();
    if (vxGetStatus((vx_reference)ctx) != VX_SUCCESS) { printf("RESULT %s FAIL context\n", which.c_str()); return 1; }
    AgoTargetAffinityInfo aff;
    memset(&aff, 0, sizeof(aff));
    aff.device_type = gpu ? AGO_TARGET_AFFINITY_GPU : AGO_TARGET_AFFINITY_CPU;
    vx_status s = vxSetContextAttribute(ctx, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &aff, sizeof(aff));
    if (s != VX_SUCCESS) { printf("RESULT %s FAIL set-affinity %d\n", which.c_str(), s); return 1; }
    s = vxLoadKernels(ctx, "vx_rpp");
    if (s != VX_SUCCESS) { printf("RESULT %s FAIL vxLoadKernels(vx_rpp) %d\n", which.c_str(), s); return 1; }

    Env e;
    e.g = vxCreateGraph(ctx);
    e.in = mk({N, H, W, C}, VX_TYPE_UINT8, 1);
    e.in2 = mk({N, H, W, C}, VX_TYPE_UINT8, 1);
    e.roi = mk({N, 4}, VX_TYPE_UINT32, 4);
    e.out = mk(all[which].outDims, VX_TYPE_UINT8, 1);
    e.inL = sI(VX_NHWC);
    e.outL = sI(VX_NHWC);
    e.roiT = sI(ROI_XYWH);

    std::mt19937 rng(42);
    std::vector<uint8_t> a(e.in.bytes()), b(e.in2.bytes());
    for (size_t y = 0; y < N * H; y++)
        for (size_t x = 0; x < W * C; x++) {
            size_t i = y * W * C + x;  // smooth gradient + noise, so filters and warps see structure
            a[i] = (uint8_t)(((y % H) * 2 + (x / C) + (rng() % 32)) & 0xFF);
            b[i] = (uint8_t)(rng() & 0xFF);
        }
    uint32_t roi[N * 4] = {0, 0, (uint32_t)W, (uint32_t)H, 0, 0, (uint32_t)W, (uint32_t)H};
    if (which == "Crop") { roi[0] = 10; roi[1] = 5; roi[2] = 100; roi[3] = 80; roi[4] = 20; roi[5] = 10; roi[6] = 100; roi[7] = 80; }
    dump(dir + "/in.bin", a);
    dump(dir + "/in2.bin", b);
    if (put(e.in, a.data()) || put(e.in2, b.data()) || put(e.roi, roi)) { printf("RESULT %s FAIL copy-in\n", which.c_str()); return 1; }
    std::vector<uint8_t> zero(e.out.bytes(), 0);
    put(e.out, zero.data());

    vx_node node = all[which].build(e);
    if (vxGetStatus((vx_reference)node) != VX_SUCCESS) { printf("RESULT %s FAIL node-create %d\n", which.c_str(), vxGetStatus((vx_reference)node)); return 1; }
    s = vxVerifyGraph(e.g);
    if (s != VX_SUCCESS) { printf("RESULT %s FAIL verify %d\n", which.c_str(), s); return 1; }
    s = vxProcessGraph(e.g);
    if (s != VX_SUCCESS) { printf("RESULT %s FAIL process %d\n", which.c_str(), s); return 1; }
    std::vector<uint8_t> r1;
    if (get(e.out, r1)) { printf("RESULT %s FAIL copy-out\n", which.c_str()); return 1; }
    dump(dir + "/" + which + ".run1.bin", r1);
    s = vxProcessGraph(e.g);
    if (s != VX_SUCCESS) { printf("RESULT %s FAIL process2 %d\n", which.c_str(), s); return 1; }
    std::vector<uint8_t> o;
    if (get(e.out, o)) { printf("RESULT %s FAIL copy-out\n", which.c_str()); return 1; }
    dump(dir + "/" + which + ".bin", o);
    AgoTargetAffinityInfo na;
    memset(&na, 0, sizeof(na));
    vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_AFFINITY, &na, sizeof(na));
    printf("RESULT %s PASS node_affinity=%s bytes=%zu\n", which.c_str(),
           na.device_type == AGO_TARGET_AFFINITY_GPU ? "GPU" : (na.device_type == AGO_TARGET_AFFINITY_CPU ? "CPU" : "?"), o.size());
    vxReleaseGraph(&e.g);
    vxReleaseContext(&ctx);
    return 0;
}
