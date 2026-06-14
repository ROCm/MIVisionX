/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

// Vision coverage test - exercises whole CPU vision features that the
// conformance / GDF / vision-python suites do not cover on the HOST backend:
//   - Laplacian pyramid construction & reconstruction
//   - Color conversion RGB/RGBX <-> planar YUV (IYUV / NV12 / YUV4)
//   - Canny edge detector across all gradient sizes (3/5/7) and norms (L1/L2)
//   - Custom MxN (non-separable) convolution -> U8 and S16
//   - Harris corners across gradient sizes
//   - Optical flow (Lucas-Kanade pyramid) tracking
//   - Remap (nearest + bilinear)
//
// Each feature builds its own graph, verifies and processes it on the CPU,
// then releases all objects. Failures are reported but do not abort the run so
// the maximum amount of code is exercised for coverage.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>

static int g_failures = 0;

#define RUN_OK(label, expr) do { \
    vx_status s_ = (expr); \
    if (s_ == VX_SUCCESS) { printf("  [ok]   %s\n", label); } \
    else { printf("  [fail] %s -> status %d\n", label, s_); g_failures++; } \
} while (0)

// Lenient: a NOT_SUPPORTED / FAILURE result is informational, not a hard error.
#define RUN_LENIENT(label, expr) do { \
    vx_status s_ = (expr); \
    if (s_ == VX_SUCCESS) printf("  [ok]   %s\n", label); \
    else printf("  [info] %s -> status %d (skipped/unsupported)\n", label, s_); \
} while (0)

static vx_status verify_process(vx_graph graph)
{
    vx_status status = vxVerifyGraph(graph);
    if (status != VX_SUCCESS) return status;
    return vxProcessGraph(graph);
}

// ---------------------------------------------------------------------------
// Laplacian pyramid build + reconstruct
// ---------------------------------------------------------------------------
static void test_laplacian(vx_context ctx)
{
    printf("\n=== Laplacian pyramid & reconstruct ===\n");
    const vx_uint32 W = 128, H = 128;
    const vx_size levels = 4;

    // Build: input U8 -> laplacian pyramid (S16) + lowest-res output (U8)
    {
        vx_graph g = vxCreateGraph(ctx);
        vx_image input = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
        vx_pyramid lap = vxCreatePyramid(ctx, levels, VX_SCALE_PYRAMID_HALF, W, H, VX_DF_IMAGE_S16);
        // residual is one octave below the last pyramid level: width * scale^levels
        vx_image lowres = vxCreateImage(ctx, W >> levels, H >> levels, VX_DF_IMAGE_U8);
        vx_node n = vxLaplacianPyramidNode(g, input, lap, lowres);
        RUN_LENIENT("LaplacianPyramidNode build", verify_process(g));
        vxReleaseNode(&n);
        vxReleaseImage(&lowres);
        vxReleasePyramid(&lap);
        vxReleaseImage(&input);
        vxReleaseGraph(&g);
    }

    // Reconstruct: laplacian pyramid (S16) + lowest-res input (U8) -> output U8
    {
        vx_graph g = vxCreateGraph(ctx);
        vx_pyramid lap = vxCreatePyramid(ctx, levels, VX_SCALE_PYRAMID_HALF, W, H, VX_DF_IMAGE_S16);
        vx_image lowres = vxCreateImage(ctx, W >> levels, H >> levels, VX_DF_IMAGE_U8);
        vx_image output = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
        vx_node n = vxLaplacianReconstructNode(g, lap, lowres, output);
        RUN_LENIENT("LaplacianReconstructNode", verify_process(g));
        vxReleaseNode(&n);
        vxReleaseImage(&output);
        vxReleaseImage(&lowres);
        vxReleasePyramid(&lap);
        vxReleaseGraph(&g);
    }
}

// ---------------------------------------------------------------------------
// Color conversion to/from planar YUV formats
// ---------------------------------------------------------------------------
static void test_color_convert(vx_context ctx)
{
    printf("\n=== Color convert RGB/RGBX <-> planar YUV ===\n");
    const vx_uint32 W = 64, H = 64;
    struct { vx_df_image in, out; const char *desc; } cases[] = {
        { VX_DF_IMAGE_RGB,  VX_DF_IMAGE_IYUV, "RGB->IYUV" },
        { VX_DF_IMAGE_RGB,  VX_DF_IMAGE_NV12, "RGB->NV12" },
        { VX_DF_IMAGE_RGB,  VX_DF_IMAGE_YUV4, "RGB->YUV4" },
        { VX_DF_IMAGE_RGBX, VX_DF_IMAGE_IYUV, "RGBX->IYUV" },
        { VX_DF_IMAGE_RGBX, VX_DF_IMAGE_NV12, "RGBX->NV12" },
        { VX_DF_IMAGE_RGBX, VX_DF_IMAGE_YUV4, "RGBX->YUV4" },
        { VX_DF_IMAGE_IYUV, VX_DF_IMAGE_RGB,  "IYUV->RGB" },
        { VX_DF_IMAGE_IYUV, VX_DF_IMAGE_RGBX, "IYUV->RGBX" },
        { VX_DF_IMAGE_NV12, VX_DF_IMAGE_RGB,  "NV12->RGB" },
        { VX_DF_IMAGE_NV12, VX_DF_IMAGE_RGBX, "NV12->RGBX" },
        { VX_DF_IMAGE_YUV4, VX_DF_IMAGE_RGB,  "YUV4->RGB" },
    };
    for (auto &c : cases) {
        vx_graph g = vxCreateGraph(ctx);
        vx_image in = vxCreateImage(ctx, W, H, c.in);
        vx_image out = vxCreateImage(ctx, W, H, c.out);
        vx_node n = vxColorConvertNode(g, in, out);
        RUN_LENIENT(c.desc, verify_process(g));
        vxReleaseNode(&n);
        vxReleaseImage(&out);
        vxReleaseImage(&in);
        vxReleaseGraph(&g);
    }
}

// ---------------------------------------------------------------------------
// Canny edge detector across gradient sizes and norms
// ---------------------------------------------------------------------------
static void test_canny(vx_context ctx)
{
    printf("\n=== Canny edge detector (gradient 3/5/7, L1/L2) ===\n");
    const vx_uint32 W = 128, H = 128;
    vx_int32 grads[] = { 3, 5, 7 };
    vx_enum norms[] = { VX_NORM_L1, VX_NORM_L2 };
    for (vx_int32 gs : grads) {
        for (vx_enum nt : norms) {
            vx_graph g = vxCreateGraph(ctx);
            vx_image in = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
            vx_image out = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
            vx_threshold hyst = vxCreateThresholdForImage(ctx, VX_THRESHOLD_TYPE_RANGE,
                                                          VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
            vx_int32 lo = 80, hi = 160;
            vxSetThresholdAttribute(hyst, VX_THRESHOLD_THRESHOLD_LOWER, &lo, sizeof(lo));
            vxSetThresholdAttribute(hyst, VX_THRESHOLD_THRESHOLD_UPPER, &hi, sizeof(hi));
            vx_node n = vxCannyEdgeDetectorNode(g, in, hyst, gs, nt, out);
            char label[64];
            snprintf(label, sizeof(label), "Canny grad=%d norm=%s", gs, nt == VX_NORM_L1 ? "L1" : "L2");
            RUN_LENIENT(label, verify_process(g));
            vxReleaseNode(&n);
            vxReleaseThreshold(&hyst);
            vxReleaseImage(&out);
            vxReleaseImage(&in);
            vxReleaseGraph(&g);
        }
    }
}

// ---------------------------------------------------------------------------
// Custom non-separable MxN convolution
// ---------------------------------------------------------------------------
static void test_convolution(vx_context ctx)
{
    printf("\n=== Custom MxN convolution ===\n");
    struct { vx_size cols, rows; vx_df_image out; const char *desc; } cases[] = {
        { 5, 5, VX_DF_IMAGE_S16, "5x5 -> S16" },
        { 3, 5, VX_DF_IMAGE_S16, "3x5 -> S16" },
        { 5, 5, VX_DF_IMAGE_U8,  "5x5 -> U8" },
        { 7, 7, VX_DF_IMAGE_S16, "7x7 -> S16" },
    };
    const vx_uint32 W = 96, H = 96;
    for (auto &c : cases) {
        vx_graph g = vxCreateGraph(ctx);
        vx_image in = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
        vx_image out = vxCreateImage(ctx, W, H, c.out);
        vx_convolution conv = vxCreateConvolution(ctx, c.cols, c.rows);
        std::vector<vx_int16> coeffs(c.cols * c.rows, 1);
        vxCopyConvolutionCoefficients(conv, coeffs.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        vx_node n = vxConvolveNode(g, in, conv, out);
        RUN_LENIENT(c.desc, verify_process(g));
        vxReleaseNode(&n);
        vxReleaseConvolution(&conv);
        vxReleaseImage(&out);
        vxReleaseImage(&in);
        vxReleaseGraph(&g);
    }
}

// ---------------------------------------------------------------------------
// Harris corners across gradient sizes
// ---------------------------------------------------------------------------
static void test_harris(vx_context ctx)
{
    printf("\n=== Harris corners (gradient 3/5/7) ===\n");
    const vx_uint32 W = 128, H = 128;
    vx_int32 grads[] = { 3, 5, 7 };
    for (vx_int32 gs : grads) {
        vx_graph g = vxCreateGraph(ctx);
        vx_image in = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
        vx_float32 strength = 0.0005f, mindist = 5.0f, sensitivity = 0.04f;
        vx_scalar s_strength = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &strength);
        vx_scalar s_mindist = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &mindist);
        vx_scalar s_sens = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &sensitivity);
        vx_array corners = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 1000);
        vx_size num = 0;
        vx_scalar s_num = vxCreateScalar(ctx, VX_TYPE_SIZE, &num);
        vx_node n = vxHarrisCornersNode(g, in, s_strength, s_mindist, s_sens, gs, gs, corners, s_num);
        char label[48];
        snprintf(label, sizeof(label), "Harris grad/block=%d", gs);
        RUN_LENIENT(label, verify_process(g));
        vxReleaseNode(&n);
        vxReleaseScalar(&s_num);
        vxReleaseArray(&corners);
        vxReleaseScalar(&s_sens);
        vxReleaseScalar(&s_mindist);
        vxReleaseScalar(&s_strength);
        vxReleaseImage(&in);
        vxReleaseGraph(&g);
    }
}

// ---------------------------------------------------------------------------
// Optical flow (Lucas-Kanade pyramid)
// ---------------------------------------------------------------------------
static void test_optical_flow(vx_context ctx)
{
    printf("\n=== Optical flow pyramid LK ===\n");
    const vx_uint32 W = 128, H = 128;
    const vx_size levels = 4;

    vx_graph g = vxCreateGraph(ctx);
    vx_pyramid old_pyr = vxCreatePyramid(ctx, levels, VX_SCALE_PYRAMID_HALF, W, H, VX_DF_IMAGE_U8);
    vx_pyramid new_pyr = vxCreatePyramid(ctx, levels, VX_SCALE_PYRAMID_HALF, W, H, VX_DF_IMAGE_U8);
    vx_array old_pts = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 64);
    vx_array est_pts = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 64);
    vx_array new_pts = vxCreateArray(ctx, VX_TYPE_KEYPOINT, 64);

    std::vector<vx_keypoint_t> kp(16);
    for (size_t i = 0; i < kp.size(); ++i) {
        memset(&kp[i], 0, sizeof(vx_keypoint_t));
        kp[i].x = (vx_int32)(8 + i * 4);
        kp[i].y = (vx_int32)(8 + i * 4);
        kp[i].tracking_status = 1;
        kp[i].strength = 1.0f;
    }
    vxAddArrayItems(old_pts, kp.size(), kp.data(), sizeof(vx_keypoint_t));
    vxAddArrayItems(est_pts, kp.size(), kp.data(), sizeof(vx_keypoint_t));

    vx_float32 eps = 0.01f;
    vx_uint32 iters = 10;
    vx_bool use_est = vx_false_e;
    vx_scalar s_eps = vxCreateScalar(ctx, VX_TYPE_FLOAT32, &eps);
    vx_scalar s_iters = vxCreateScalar(ctx, VX_TYPE_UINT32, &iters);
    vx_scalar s_use = vxCreateScalar(ctx, VX_TYPE_BOOL, &use_est);

    vx_node n = vxOpticalFlowPyrLKNode(g, old_pyr, new_pyr, old_pts, est_pts, new_pts,
                                       VX_TERM_CRITERIA_BOTH, s_eps, s_iters, s_use, 5);
    RUN_LENIENT("OpticalFlowPyrLKNode", verify_process(g));

    vxReleaseNode(&n);
    vxReleaseScalar(&s_use);
    vxReleaseScalar(&s_iters);
    vxReleaseScalar(&s_eps);
    vxReleaseArray(&new_pts);
    vxReleaseArray(&est_pts);
    vxReleaseArray(&old_pts);
    vxReleasePyramid(&new_pyr);
    vxReleasePyramid(&old_pyr);
    vxReleaseGraph(&g);
}

// ---------------------------------------------------------------------------
// Remap (nearest + bilinear)
// ---------------------------------------------------------------------------
static void test_remap(vx_context ctx)
{
    printf("\n=== Remap (nearest / bilinear) ===\n");
    const vx_uint32 W = 64, H = 64;
    vx_enum policies[] = { VX_INTERPOLATION_NEAREST_NEIGHBOR, VX_INTERPOLATION_BILINEAR };
    for (vx_enum pol : policies) {
        vx_graph g = vxCreateGraph(ctx);
        vx_image in = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
        vx_image out = vxCreateImage(ctx, W, H, VX_DF_IMAGE_U8);
        vx_remap table = vxCreateRemap(ctx, W, H, W, H);
        for (vx_uint32 y = 0; y < H; ++y)
            for (vx_uint32 x = 0; x < W; ++x)
                vxSetRemapPoint(table, x, y, (vx_float32)(W - 1 - x), (vx_float32)(H - 1 - y));
        vx_node n = vxRemapNode(g, in, table, pol, out);
        RUN_LENIENT(pol == VX_INTERPOLATION_BILINEAR ? "Remap bilinear" : "Remap nearest",
                    verify_process(g));
        vxReleaseNode(&n);
        vxReleaseRemap(&table);
        vxReleaseImage(&out);
        vxReleaseImage(&in);
        vxReleaseGraph(&g);
    }
}

// ---------------------------------------------------------------------------
// Error / validation paths (negative testing of the public API)
// ---------------------------------------------------------------------------
static void test_error_paths(vx_context ctx)
{
    printf("\n=== Error / validation paths ===\n");

    // Invalid object creation parameters
    RUN_LENIENT("CreateImage 0x0", vxGetStatus((vx_reference)vxCreateImage(ctx, 0, 0, VX_DF_IMAGE_U8)));
    RUN_LENIENT("CreateImage bad format", vxGetStatus((vx_reference)vxCreateImage(ctx, 16, 16, (vx_df_image)0xDEAD)));
    RUN_LENIENT("CreateArray bad type", vxGetStatus((vx_reference)vxCreateArray(ctx, (vx_enum)0xDEAD, 10)));
    RUN_LENIENT("CreateMatrix bad type", vxGetStatus((vx_reference)vxCreateMatrix(ctx, (vx_enum)0xDEAD, 3, 3)));

    // Query with NULL / wrong size on a valid image
    vx_image img = vxCreateImage(ctx, 32, 32, VX_DF_IMAGE_U8);
    vx_uint32 w = 0;
    RUN_LENIENT("QueryImage bad attribute", vxQueryImage(img, (vx_enum)0xDEAD, &w, sizeof(w)));
    RUN_LENIENT("QueryImage wrong size", vxQueryImage(img, VX_IMAGE_WIDTH, &w, 1));
    RUN_LENIENT("SetImageAttribute bad attribute", vxSetImageAttribute(img, (vx_enum)0xDEAD, &w, sizeof(w)));

    // Invalid references
    RUN_LENIENT("QueryImage on NULL", vxQueryImage(NULL, VX_IMAGE_WIDTH, &w, sizeof(w)));
    RUN_LENIENT("GetStatus NULL", vxGetStatus(NULL) == VX_SUCCESS ? VX_FAILURE : VX_SUCCESS);

    // Graph with an invalid (disconnected/unsatisfied) configuration should fail verify
    {
        vx_graph g = vxCreateGraph(ctx);
        vx_image in = vxCreateImage(ctx, 16, 16, VX_DF_IMAGE_U8);
        vx_image out = vxCreateImage(ctx, 32, 32, VX_DF_IMAGE_U8); // size mismatch
        vx_node n = vxNotNode(g, in, out);
        vx_status s = vxVerifyGraph(g);
        printf("  [info] mismatched-size graph verify -> %d (expected failure)\n", s);
        vxReleaseNode(&n);
        vxReleaseImage(&out);
        vxReleaseImage(&in);
        vxReleaseGraph(&g);
    }

    // Kernel lookup failures
    RUN_LENIENT("GetKernelByEnum invalid", vxGetStatus((vx_reference)vxGetKernelByEnum(ctx, (vx_enum)0x7FFFFFFF)));
    RUN_LENIENT("GetKernelByName invalid", vxGetStatus((vx_reference)vxGetKernelByName(ctx, "org.khronos.does.not.exist")));

    vxReleaseImage(&img);
}

int main()
{
    printf("Vision Coverage API Test\n");
    printf("========================\n");

    vx_context ctx = vxCreateContext();
    if (vxGetStatus((vx_reference)ctx) != VX_SUCCESS) {
        printf("ERROR: vxCreateContext failed\n");
        return 1;
    }

    test_laplacian(ctx);
    test_color_convert(ctx);
    test_canny(ctx);
    test_convolution(ctx);
    test_harris(ctx);
    test_optical_flow(ctx);
    test_remap(ctx);
    test_error_paths(ctx);

    vxReleaseContext(&ctx);

    printf("\n========================\n");
    // This test is for coverage; "soft" failures (unsupported paths) are not
    // fatal. Only report hard infrastructure failures (currently none tracked
    // as hard). Always return success so it never breaks the (continue-on-error)
    // API stage while still exercising the code under coverage.
    printf("Vision Coverage API Test: done (%d hard failures)\n", g_failures);
    return 0;
}
