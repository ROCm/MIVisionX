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

// VXU (OpenVX Utility) immediate-mode API coverage test.
// Exercises vxu functions that create internal graphs, verify, execute, and
// tear them down -- covering the full kernel init/validate/execute/shutdown
// path for each vision kernel type through immediate-mode invocation.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <VX/vxu.h>
#include <vx_ext_amd.h>

// vxuAccumulate* functions are declared in VX/vx_compatibility.h (included above)

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------
#define CHECK_STATUS(call) do { \
    vx_status s = (call); \
    if (s != VX_SUCCESS) { \
        printf("  FAIL: %s returned %d at %s:%d\n", #call, s, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

#define CHECK_NULL(obj, name) do { \
    if (!(obj)) { \
        printf("  FAIL: %s is NULL at %s:%d\n", name, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

// Image dimensions for tests
static const vx_uint32 WIDTH  = 64;
static const vx_uint32 HEIGHT = 64;

// ---------------------------------------------------------------------------
// Helper: create a U8 image filled with a uniform value
// ---------------------------------------------------------------------------
static vx_image createUniformU8(vx_context context, vx_uint32 w, vx_uint32 h, vx_uint8 value) {
    vx_pixel_value_t pv;
    memset(&pv, 0, sizeof(pv));
    pv.U8 = value;
    return vxCreateUniformImage(context, w, h, VX_DF_IMAGE_U8, &pv);
}

// ---------------------------------------------------------------------------
// Helper: create an S16 image filled with a uniform value
// ---------------------------------------------------------------------------
static vx_image createUniformS16(vx_context context, vx_uint32 w, vx_uint32 h, vx_int16 value) {
    vx_pixel_value_t pv;
    memset(&pv, 0, sizeof(pv));
    pv.S16 = value;
    return vxCreateUniformImage(context, w, h, VX_DF_IMAGE_S16, &pv);
}

// ---------------------------------------------------------------------------
// Test 1: vxuColorConvert (RGB -> NV12 and NV12 -> RGB)
// ---------------------------------------------------------------------------
static int test_vxuColorConvert(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuColorConvert ---\n");

    // RGB to NV12
    {
        vx_pixel_value_t pv;
        memset(&pv, 0, sizeof(pv));
        pv.RGBX[0] = 128; pv.RGBX[1] = 64; pv.RGBX[2] = 32;
        vx_image src = vxCreateUniformImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_RGB, &pv);
        vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_NV12);
        CHECK_NULL(src, "src RGB");
        CHECK_NULL(dst, "dst NV12");
        CHECK_STATUS(vxuColorConvert(context, src, dst));
        printf("  PASS: vxuColorConvert (RGB -> NV12)\n");
        if (src) vxReleaseImage(&src);
        if (dst) vxReleaseImage(&dst);
    }

    // NV12 to RGB
    {
        vx_image src_nv12 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_NV12);
        vx_image dst_rgb  = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_RGB);
        CHECK_NULL(src_nv12, "src NV12");
        CHECK_NULL(dst_rgb,  "dst RGB");
        CHECK_STATUS(vxuColorConvert(context, src_nv12, dst_rgb));
        printf("  PASS: vxuColorConvert (NV12 -> RGB)\n");
        if (src_nv12) vxReleaseImage(&src_nv12);
        if (dst_rgb)  vxReleaseImage(&dst_rgb);
    }

    // RGBX to NV12
    {
        vx_pixel_value_t pv;
        memset(&pv, 0, sizeof(pv));
        pv.RGBX[0] = 200; pv.RGBX[1] = 100; pv.RGBX[2] = 50; pv.RGBX[3] = 255;
        vx_image src = vxCreateUniformImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_RGBX, &pv);
        vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_IYUV);
        CHECK_NULL(src, "src RGBX");
        CHECK_NULL(dst, "dst IYUV");
        CHECK_STATUS(vxuColorConvert(context, src, dst));
        printf("  PASS: vxuColorConvert (RGBX -> IYUV)\n");
        if (src) vxReleaseImage(&src);
        if (dst) vxReleaseImage(&dst);
    }

    return errors;
}

// ---------------------------------------------------------------------------
// Test 2: vxuChannelExtract
// ---------------------------------------------------------------------------
static int test_vxuChannelExtract(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuChannelExtract ---\n");

    vx_pixel_value_t pv;
    memset(&pv, 0, sizeof(pv));
    pv.RGBX[0] = 100; pv.RGBX[1] = 150; pv.RGBX[2] = 200;
    vx_image src = vxCreateUniformImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_RGB, &pv);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src RGB");
    CHECK_NULL(dst, "dst U8");

    CHECK_STATUS(vxuChannelExtract(context, src, VX_CHANNEL_R, dst));
    printf("  PASS: vxuChannelExtract (R channel)\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 3: vxuChannelCombine
// ---------------------------------------------------------------------------
static int test_vxuChannelCombine(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuChannelCombine ---\n");

    vx_image ch0 = createUniformU8(context, WIDTH, HEIGHT, 100);
    vx_image ch1 = createUniformU8(context, WIDTH, HEIGHT, 150);
    vx_image ch2 = createUniformU8(context, WIDTH, HEIGHT, 200);
    vx_image out = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_RGB);
    CHECK_NULL(ch0, "ch0");
    CHECK_NULL(ch1, "ch1");
    CHECK_NULL(ch2, "ch2");
    CHECK_NULL(out, "out RGB");

    CHECK_STATUS(vxuChannelCombine(context, ch0, ch1, ch2, NULL, out));
    printf("  PASS: vxuChannelCombine (3 channels -> RGB)\n");

    if (ch0) vxReleaseImage(&ch0);
    if (ch1) vxReleaseImage(&ch1);
    if (ch2) vxReleaseImage(&ch2);
    if (out) vxReleaseImage(&out);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 4: vxuSobel3x3
// ---------------------------------------------------------------------------
static int test_vxuSobel3x3(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuSobel3x3 ---\n");

    vx_image src  = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dx   = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_S16);
    vx_image dy   = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_S16);
    CHECK_NULL(src, "src");
    CHECK_NULL(dx, "dx");
    CHECK_NULL(dy, "dy");

    CHECK_STATUS(vxuSobel3x3(context, src, dx, dy));
    printf("  PASS: vxuSobel3x3\n");

    if (src) vxReleaseImage(&src);
    if (dx)  vxReleaseImage(&dx);
    if (dy)  vxReleaseImage(&dy);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 5: vxuMagnitude
// ---------------------------------------------------------------------------
static int test_vxuMagnitude(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuMagnitude ---\n");

    vx_image gx  = createUniformS16(context, WIDTH, HEIGHT, 3);
    vx_image gy  = createUniformS16(context, WIDTH, HEIGHT, 4);
    vx_image mag = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_S16);
    CHECK_NULL(gx, "gx");
    CHECK_NULL(gy, "gy");
    CHECK_NULL(mag, "mag");

    CHECK_STATUS(vxuMagnitude(context, gx, gy, mag));
    printf("  PASS: vxuMagnitude\n");

    if (gx)  vxReleaseImage(&gx);
    if (gy)  vxReleaseImage(&gy);
    if (mag) vxReleaseImage(&mag);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 6: vxuPhase
// ---------------------------------------------------------------------------
static int test_vxuPhase(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuPhase ---\n");

    vx_image gx    = createUniformS16(context, WIDTH, HEIGHT, 10);
    vx_image gy    = createUniformS16(context, WIDTH, HEIGHT, 10);
    vx_image phase = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(gx, "gx");
    CHECK_NULL(gy, "gy");
    CHECK_NULL(phase, "phase");

    CHECK_STATUS(vxuPhase(context, gx, gy, phase));
    printf("  PASS: vxuPhase\n");

    if (gx)    vxReleaseImage(&gx);
    if (gy)    vxReleaseImage(&gy);
    if (phase) vxReleaseImage(&phase);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 7: vxuTableLookup
// ---------------------------------------------------------------------------
static int test_vxuTableLookup(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuTableLookup ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 100);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_lut  lut  = vxCreateLUT(context, VX_TYPE_UINT8, 256);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");
    CHECK_NULL(lut, "lut");

    // Fill LUT with identity mapping
    vx_uint8 lut_data[256];
    for (int i = 0; i < 256; i++) lut_data[i] = (vx_uint8)i;
    CHECK_STATUS(vxCopyLUT(lut, lut_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    CHECK_STATUS(vxuTableLookup(context, src, lut, dst));
    printf("  PASS: vxuTableLookup\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    if (lut) vxReleaseLUT(&lut);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 8: vxuHistogram
// ---------------------------------------------------------------------------
static int test_vxuHistogram(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuHistogram ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_distribution dist = vxCreateDistribution(context, 256, 0, 256);
    CHECK_NULL(src, "src");
    CHECK_NULL(dist, "dist");

    CHECK_STATUS(vxuHistogram(context, src, dist));
    printf("  PASS: vxuHistogram\n");

    if (src)  vxReleaseImage(&src);
    if (dist) vxReleaseDistribution(&dist);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 9: vxuEqualizeHist
// ---------------------------------------------------------------------------
static int test_vxuEqualizeHist(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuEqualizeHist ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 100);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuEqualizeHist(context, src, dst));
    printf("  PASS: vxuEqualizeHist\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 10: vxuAbsDiff
// ---------------------------------------------------------------------------
static int test_vxuAbsDiff(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuAbsDiff ---\n");

    vx_image in1 = createUniformU8(context, WIDTH, HEIGHT, 200);
    vx_image in2 = createUniformU8(context, WIDTH, HEIGHT, 50);
    vx_image out = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(in1, "in1");
    CHECK_NULL(in2, "in2");
    CHECK_NULL(out, "out");

    CHECK_STATUS(vxuAbsDiff(context, in1, in2, out));
    printf("  PASS: vxuAbsDiff\n");

    if (in1) vxReleaseImage(&in1);
    if (in2) vxReleaseImage(&in2);
    if (out) vxReleaseImage(&out);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 11: vxuMeanStdDev
// ---------------------------------------------------------------------------
static int test_vxuMeanStdDev(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuMeanStdDev ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    CHECK_NULL(src, "src");

    vx_float32 mean = 0.0f, stddev = 0.0f;
    CHECK_STATUS(vxuMeanStdDev(context, src, &mean, &stddev));
    printf("  PASS: vxuMeanStdDev (mean=%.2f, stddev=%.2f)\n", mean, stddev);

    if (src) vxReleaseImage(&src);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 12: vxuThreshold
// ---------------------------------------------------------------------------
static int test_vxuThreshold(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuThreshold ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_threshold thresh = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY,
                                                     VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");
    CHECK_NULL(thresh, "thresh");

    // Set threshold value to 100
    vx_pixel_value_t val;
    memset(&val, 0, sizeof(val));
    val.U8 = 100;
    CHECK_STATUS(vxSetThresholdAttribute(thresh, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE,
                                          &val, sizeof(vx_int32)));

    CHECK_STATUS(vxuThreshold(context, src, thresh, dst));
    printf("  PASS: vxuThreshold\n");

    if (src)    vxReleaseImage(&src);
    if (dst)    vxReleaseImage(&dst);
    if (thresh) vxReleaseThreshold(&thresh);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 13: vxuIntegralImage
// ---------------------------------------------------------------------------
static int test_vxuIntegralImage(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuIntegralImage ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 1);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U32);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuIntegralImage(context, src, dst));
    printf("  PASS: vxuIntegralImage\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 14: vxuErode3x3
// ---------------------------------------------------------------------------
static int test_vxuErode3x3(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuErode3x3 ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 200);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuErode3x3(context, src, dst));
    printf("  PASS: vxuErode3x3\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 15: vxuDilate3x3
// ---------------------------------------------------------------------------
static int test_vxuDilate3x3(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuDilate3x3 ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 50);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuDilate3x3(context, src, dst));
    printf("  PASS: vxuDilate3x3\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 16: vxuMedian3x3
// ---------------------------------------------------------------------------
static int test_vxuMedian3x3(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuMedian3x3 ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuMedian3x3(context, src, dst));
    printf("  PASS: vxuMedian3x3\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 17: vxuBox3x3
// ---------------------------------------------------------------------------
static int test_vxuBox3x3(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuBox3x3 ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuBox3x3(context, src, dst));
    printf("  PASS: vxuBox3x3\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 18: vxuGaussian3x3
// ---------------------------------------------------------------------------
static int test_vxuGaussian3x3(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuGaussian3x3 ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuGaussian3x3(context, src, dst));
    printf("  PASS: vxuGaussian3x3\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 19: vxuNonLinearFilter (PRIORITY: previously uncovered)
// ---------------------------------------------------------------------------
static int test_vxuNonLinearFilter(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuNonLinearFilter ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    // Create a 3x3 BOX pattern mask
    vx_matrix mask = vxCreateMatrixFromPattern(context, VX_PATTERN_BOX, 3, 3);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");
    CHECK_NULL(mask, "mask");

    CHECK_STATUS(vxuNonLinearFilter(context, VX_NONLINEAR_FILTER_MEDIAN, src, mask, dst));
    printf("  PASS: vxuNonLinearFilter (MEDIAN, BOX 3x3)\n");

    // Also test with MIN filter
    vx_image dst2 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(dst2, "dst2");
    CHECK_STATUS(vxuNonLinearFilter(context, VX_NONLINEAR_FILTER_MIN, src, mask, dst2));
    printf("  PASS: vxuNonLinearFilter (MIN, BOX 3x3)\n");

    // Also test with MAX filter
    vx_image dst3 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(dst3, "dst3");
    CHECK_STATUS(vxuNonLinearFilter(context, VX_NONLINEAR_FILTER_MAX, src, mask, dst3));
    printf("  PASS: vxuNonLinearFilter (MAX, BOX 3x3)\n");

    // Test with CROSS pattern
    vx_matrix mask_cross = vxCreateMatrixFromPattern(context, VX_PATTERN_CROSS, 3, 3);
    vx_image dst4 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(mask_cross, "mask_cross");
    CHECK_NULL(dst4, "dst4");
    CHECK_STATUS(vxuNonLinearFilter(context, VX_NONLINEAR_FILTER_MEDIAN, src, mask_cross, dst4));
    printf("  PASS: vxuNonLinearFilter (MEDIAN, CROSS 3x3)\n");

    // Test with DISK pattern
    vx_matrix mask_disk = vxCreateMatrixFromPattern(context, VX_PATTERN_DISK, 5, 5);
    vx_image dst5 = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(mask_disk, "mask_disk");
    CHECK_NULL(dst5, "dst5");
    CHECK_STATUS(vxuNonLinearFilter(context, VX_NONLINEAR_FILTER_MEDIAN, src, mask_disk, dst5));
    printf("  PASS: vxuNonLinearFilter (MEDIAN, DISK 5x5)\n");

    if (src)        vxReleaseImage(&src);
    if (dst)        vxReleaseImage(&dst);
    if (dst2)       vxReleaseImage(&dst2);
    if (dst3)       vxReleaseImage(&dst3);
    if (dst4)       vxReleaseImage(&dst4);
    if (dst5)       vxReleaseImage(&dst5);
    if (mask)       vxReleaseMatrix(&mask);
    if (mask_cross) vxReleaseMatrix(&mask_cross);
    if (mask_disk)  vxReleaseMatrix(&mask_disk);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 20: vxuConvolve
// ---------------------------------------------------------------------------
static int test_vxuConvolve(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuConvolve ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_S16);
    // Create a 3x3 identity convolution kernel
    vx_convolution conv = vxCreateConvolution(context, 3, 3);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");
    CHECK_NULL(conv, "conv");

    // Simple averaging kernel
    vx_int16 kernel_data[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    CHECK_STATUS(vxCopyConvolutionCoefficients(conv, kernel_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    vx_uint32 conv_scale = 9;
    CHECK_STATUS(vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, &conv_scale, sizeof(vx_uint32)));

    CHECK_STATUS(vxuConvolve(context, src, conv, dst));
    printf("  PASS: vxuConvolve\n");

    if (src)  vxReleaseImage(&src);
    if (dst)  vxReleaseImage(&dst);
    if (conv) vxReleaseConvolution(&conv);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 21: vxuGaussianPyramid
// ---------------------------------------------------------------------------
static int test_vxuGaussianPyramid(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuGaussianPyramid ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_pyramid pyr = vxCreatePyramid(context, 4, VX_SCALE_PYRAMID_HALF, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(pyr, "pyr");

    CHECK_STATUS(vxuGaussianPyramid(context, src, pyr));
    printf("  PASS: vxuGaussianPyramid\n");

    if (src) vxReleaseImage(&src);
    if (pyr) vxReleasePyramid(&pyr);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 22: vxuLaplacianPyramid (PRIORITY: previously completely uncovered)
// ---------------------------------------------------------------------------
static int test_vxuLaplacianPyramid(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuLaplacianPyramid ---\n");

    vx_uint32 pyr_w = WIDTH;
    vx_uint32 pyr_h = HEIGHT;
    vx_size levels = 4;

    vx_image src = createUniformU8(context, pyr_w, pyr_h, 128);
    // Laplacian pyramid has S16 format, levels-1 actual levels
    vx_pyramid lap_pyr = vxCreatePyramid(context, levels - 1, VX_SCALE_PYRAMID_HALF,
                                          pyr_w, pyr_h, VX_DF_IMAGE_S16);
    // The output is the lowest resolution image (U8)
    vx_uint32 last_w = pyr_w;
    vx_uint32 last_h = pyr_h;
    for (vx_size i = 0; i < levels - 1; i++) {
        last_w = (last_w + 1) / 2;
        last_h = (last_h + 1) / 2;
    }
    vx_image last_level = vxCreateImage(context, last_w, last_h, VX_DF_IMAGE_U8);

    CHECK_NULL(src, "src");
    CHECK_NULL(lap_pyr, "lap_pyr");
    CHECK_NULL(last_level, "last_level");

    CHECK_STATUS(vxuLaplacianPyramid(context, src, lap_pyr, last_level));
    printf("  PASS: vxuLaplacianPyramid\n");

    if (src)        vxReleaseImage(&src);
    if (lap_pyr)    vxReleasePyramid(&lap_pyr);
    if (last_level) vxReleaseImage(&last_level);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 23: vxuLaplacianReconstruct (PRIORITY: previously completely uncovered)
// ---------------------------------------------------------------------------
static int test_vxuLaplacianReconstruct(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuLaplacianReconstruct ---\n");

    vx_uint32 pyr_w = WIDTH;
    vx_uint32 pyr_h = HEIGHT;
    vx_size levels = 4;

    // First build a Laplacian pyramid so we have valid data to reconstruct
    vx_image src = createUniformU8(context, pyr_w, pyr_h, 100);
    vx_pyramid lap_pyr = vxCreatePyramid(context, levels - 1, VX_SCALE_PYRAMID_HALF,
                                          pyr_w, pyr_h, VX_DF_IMAGE_S16);
    vx_uint32 last_w = pyr_w;
    vx_uint32 last_h = pyr_h;
    for (vx_size i = 0; i < levels - 1; i++) {
        last_w = (last_w + 1) / 2;
        last_h = (last_h + 1) / 2;
    }
    vx_image last_level = vxCreateImage(context, last_w, last_h, VX_DF_IMAGE_U8);

    CHECK_NULL(src, "src");
    CHECK_NULL(lap_pyr, "lap_pyr");
    CHECK_NULL(last_level, "last_level");

    // Build the Laplacian pyramid first
    CHECK_STATUS(vxuLaplacianPyramid(context, src, lap_pyr, last_level));

    // Now reconstruct from it
    vx_image reconstructed = vxCreateImage(context, pyr_w, pyr_h, VX_DF_IMAGE_U8);
    CHECK_NULL(reconstructed, "reconstructed");

    CHECK_STATUS(vxuLaplacianReconstruct(context, lap_pyr, last_level, reconstructed));
    printf("  PASS: vxuLaplacianReconstruct\n");

    if (src)           vxReleaseImage(&src);
    if (lap_pyr)       vxReleasePyramid(&lap_pyr);
    if (last_level)    vxReleaseImage(&last_level);
    if (reconstructed) vxReleaseImage(&reconstructed);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 24: vxuMinMaxLoc
// ---------------------------------------------------------------------------
static int test_vxuMinMaxLoc(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuMinMaxLoc ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_scalar minVal = vxCreateScalar(context, VX_TYPE_UINT8, NULL);
    vx_scalar maxVal = vxCreateScalar(context, VX_TYPE_UINT8, NULL);
    vx_array  minLoc = vxCreateArray(context, VX_TYPE_COORDINATES2D, 1);
    vx_array  maxLoc = vxCreateArray(context, VX_TYPE_COORDINATES2D, 1);
    vx_scalar minCount = vxCreateScalar(context, VX_TYPE_SIZE, NULL);
    vx_scalar maxCount = vxCreateScalar(context, VX_TYPE_SIZE, NULL);
    CHECK_NULL(src, "src");

    CHECK_STATUS(vxuMinMaxLoc(context, src, minVal, maxVal, minLoc, maxLoc, minCount, maxCount));
    printf("  PASS: vxuMinMaxLoc\n");

    if (src)      vxReleaseImage(&src);
    if (minVal)   vxReleaseScalar(&minVal);
    if (maxVal)   vxReleaseScalar(&maxVal);
    if (minLoc)   vxReleaseArray(&minLoc);
    if (maxLoc)   vxReleaseArray(&maxLoc);
    if (minCount) vxReleaseScalar(&minCount);
    if (maxCount) vxReleaseScalar(&maxCount);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 25: vxuConvertDepth
// ---------------------------------------------------------------------------
static int test_vxuConvertDepth(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuConvertDepth ---\n");

    // U8 -> S16
    {
        vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
        vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_S16);
        CHECK_NULL(src, "src U8");
        CHECK_NULL(dst, "dst S16");

        CHECK_STATUS(vxuConvertDepth(context, src, dst, VX_CONVERT_POLICY_SATURATE, 0));
        printf("  PASS: vxuConvertDepth (U8 -> S16)\n");

        if (src) vxReleaseImage(&src);
        if (dst) vxReleaseImage(&dst);
    }

    // S16 -> U8
    {
        vx_image src = createUniformS16(context, WIDTH, HEIGHT, 200);
        vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
        CHECK_NULL(src, "src S16");
        CHECK_NULL(dst, "dst U8");

        CHECK_STATUS(vxuConvertDepth(context, src, dst, VX_CONVERT_POLICY_SATURATE, 0));
        printf("  PASS: vxuConvertDepth (S16 -> U8)\n");

        if (src) vxReleaseImage(&src);
        if (dst) vxReleaseImage(&dst);
    }

    return errors;
}

// ---------------------------------------------------------------------------
// Test 26: vxuWarpAffine
// ---------------------------------------------------------------------------
static int test_vxuWarpAffine(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuWarpAffine ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    // Identity affine matrix (2x3)
    vx_matrix mat = vxCreateMatrix(context, VX_TYPE_FLOAT32, 2, 3);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");
    CHECK_NULL(mat, "mat");

    // Identity: [[1,0,0],[0,1,0]]
    vx_float32 identity[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    CHECK_STATUS(vxCopyMatrix(mat, identity, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    CHECK_STATUS(vxuWarpAffine(context, src, mat, VX_INTERPOLATION_NEAREST_NEIGHBOR, dst));
    printf("  PASS: vxuWarpAffine\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    if (mat) vxReleaseMatrix(&mat);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 27: vxuWarpPerspective
// ---------------------------------------------------------------------------
static int test_vxuWarpPerspective(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuWarpPerspective ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    // Identity perspective matrix (3x3)
    vx_matrix mat = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");
    CHECK_NULL(mat, "mat");

    vx_float32 identity[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    CHECK_STATUS(vxCopyMatrix(mat, identity, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    CHECK_STATUS(vxuWarpPerspective(context, src, mat, VX_INTERPOLATION_NEAREST_NEIGHBOR, dst));
    printf("  PASS: vxuWarpPerspective\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    if (mat) vxReleaseMatrix(&mat);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 28: vxuHarrisCorners
// ---------------------------------------------------------------------------
static int test_vxuHarrisCorners(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuHarrisCorners ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_float32 strength_val = 0.0005f;
    vx_float32 min_dist_val = 5.0f;
    vx_float32 sens_val     = 0.04f;
    vx_scalar strength   = vxCreateScalar(context, VX_TYPE_FLOAT32, &strength_val);
    vx_scalar min_dist   = vxCreateScalar(context, VX_TYPE_FLOAT32, &min_dist_val);
    vx_scalar sensitivity = vxCreateScalar(context, VX_TYPE_FLOAT32, &sens_val);
    vx_array  corners    = vxCreateArray(context, VX_TYPE_KEYPOINT, 100);
    vx_scalar num_corners = vxCreateScalar(context, VX_TYPE_SIZE, NULL);
    CHECK_NULL(src, "src");

    CHECK_STATUS(vxuHarrisCorners(context, src, strength, min_dist, sensitivity,
                                   3, 3, corners, num_corners));
    printf("  PASS: vxuHarrisCorners\n");

    if (src)         vxReleaseImage(&src);
    if (strength)    vxReleaseScalar(&strength);
    if (min_dist)    vxReleaseScalar(&min_dist);
    if (sensitivity) vxReleaseScalar(&sensitivity);
    if (corners)     vxReleaseArray(&corners);
    if (num_corners) vxReleaseScalar(&num_corners);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 29: vxuFastCorners
// ---------------------------------------------------------------------------
static int test_vxuFastCorners(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuFastCorners ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_float32 sens_val = 50.0f;
    vx_scalar sens = vxCreateScalar(context, VX_TYPE_FLOAT32, &sens_val);
    vx_array  corners = vxCreateArray(context, VX_TYPE_KEYPOINT, 100);
    vx_scalar num_corners = vxCreateScalar(context, VX_TYPE_SIZE, NULL);
    CHECK_NULL(src, "src");

    CHECK_STATUS(vxuFastCorners(context, src, sens, vx_true_e, corners, num_corners));
    printf("  PASS: vxuFastCorners\n");

    if (src)         vxReleaseImage(&src);
    if (sens)        vxReleaseScalar(&sens);
    if (corners)     vxReleaseArray(&corners);
    if (num_corners) vxReleaseScalar(&num_corners);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 30: vxuOpticalFlowPyrLK (PRIORITY: may be uncovered in vxu path)
// ---------------------------------------------------------------------------
static int test_vxuOpticalFlowPyrLK(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuOpticalFlowPyrLK ---\n");

    vx_size levels = 4;
    // Create two gaussian pyramids from uniform images
    vx_image old_img = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image new_img = createUniformU8(context, WIDTH, HEIGHT, 130);
    CHECK_NULL(old_img, "old_img");
    CHECK_NULL(new_img, "new_img");

    vx_pyramid old_pyr = vxCreatePyramid(context, levels, VX_SCALE_PYRAMID_HALF,
                                          WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_pyramid new_pyr = vxCreatePyramid(context, levels, VX_SCALE_PYRAMID_HALF,
                                          WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(old_pyr, "old_pyr");
    CHECK_NULL(new_pyr, "new_pyr");

    // Build the pyramids
    CHECK_STATUS(vxuGaussianPyramid(context, old_img, old_pyr));
    CHECK_STATUS(vxuGaussianPyramid(context, new_img, new_pyr));

    // Create point arrays - put a single keypoint in the center
    vx_array old_points = vxCreateArray(context, VX_TYPE_KEYPOINT, 10);
    vx_array new_points_est = vxCreateArray(context, VX_TYPE_KEYPOINT, 10);
    vx_array new_points = vxCreateArray(context, VX_TYPE_KEYPOINT, 10);
    CHECK_NULL(old_points, "old_points");
    CHECK_NULL(new_points_est, "new_points_est");
    CHECK_NULL(new_points, "new_points");

    // Add a keypoint
    vx_keypoint_t kp;
    memset(&kp, 0, sizeof(kp));
    kp.x = WIDTH / 2;
    kp.y = HEIGHT / 2;
    kp.strength = 1.0f;
    kp.tracking_status = 1;
    CHECK_STATUS(vxAddArrayItems(old_points, 1, &kp, sizeof(vx_keypoint_t)));
    CHECK_STATUS(vxAddArrayItems(new_points_est, 1, &kp, sizeof(vx_keypoint_t)));

    vx_float32 eps_val = 0.01f;
    vx_scalar epsilon = vxCreateScalar(context, VX_TYPE_FLOAT32, &eps_val);
    vx_uint32 iter_val = 10;
    vx_scalar num_iters = vxCreateScalar(context, VX_TYPE_UINT32, &iter_val);
    vx_bool use_est_val = vx_true_e;
    vx_scalar use_est = vxCreateScalar(context, VX_TYPE_BOOL, &use_est_val);

    CHECK_STATUS(vxuOpticalFlowPyrLK(context, old_pyr, new_pyr,
                                      old_points, new_points_est, new_points,
                                      VX_TERM_CRITERIA_BOTH, epsilon, num_iters,
                                      use_est, 5));
    printf("  PASS: vxuOpticalFlowPyrLK\n");

    if (old_img)        vxReleaseImage(&old_img);
    if (new_img)        vxReleaseImage(&new_img);
    if (old_pyr)        vxReleasePyramid(&old_pyr);
    if (new_pyr)        vxReleasePyramid(&new_pyr);
    if (old_points)     vxReleaseArray(&old_points);
    if (new_points_est) vxReleaseArray(&new_points_est);
    if (new_points)     vxReleaseArray(&new_points);
    if (epsilon)        vxReleaseScalar(&epsilon);
    if (num_iters)      vxReleaseScalar(&num_iters);
    if (use_est)        vxReleaseScalar(&use_est);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 31: vxuRemap
// ---------------------------------------------------------------------------
static int test_vxuRemap(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuRemap ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_remap table = vxCreateRemap(context, WIDTH, HEIGHT, WIDTH, HEIGHT);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");
    CHECK_NULL(table, "table");

    // Set identity remap using vxCopyRemapPatch
    {
        vx_size stride = WIDTH * sizeof(vx_coordinates2df_t);
        vx_coordinates2df_t *coords = (vx_coordinates2df_t*)malloc(WIDTH * HEIGHT * sizeof(vx_coordinates2df_t));
        for (vx_uint32 y = 0; y < HEIGHT; y++) {
            for (vx_uint32 x = 0; x < WIDTH; x++) {
                coords[y * WIDTH + x].x = (vx_float32)x;
                coords[y * WIDTH + x].y = (vx_float32)y;
            }
        }
        vx_rectangle_t rect = {0, 0, WIDTH, HEIGHT};
        CHECK_STATUS(vxCopyRemapPatch(table, &rect, stride, coords, VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        free(coords);
    }

    CHECK_STATUS(vxuRemap(context, src, table, VX_INTERPOLATION_NEAREST_NEIGHBOR, dst));
    printf("  PASS: vxuRemap\n");

    if (src)   vxReleaseImage(&src);
    if (dst)   vxReleaseImage(&dst);
    if (table) vxReleaseRemap(&table);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 32: vxuHalfScaleGaussian
// ---------------------------------------------------------------------------
static int test_vxuHalfScaleGaussian(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuHalfScaleGaussian ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH / 2, HEIGHT / 2, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuHalfScaleGaussian(context, src, dst, 3));
    printf("  PASS: vxuHalfScaleGaussian (kernel_size=3)\n");

    // Also test with kernel_size=5
    vx_image dst5 = vxCreateImage(context, WIDTH / 2, HEIGHT / 2, VX_DF_IMAGE_U8);
    CHECK_NULL(dst5, "dst5");
    CHECK_STATUS(vxuHalfScaleGaussian(context, src, dst5, 5));
    printf("  PASS: vxuHalfScaleGaussian (kernel_size=5)\n");

    if (src)  vxReleaseImage(&src);
    if (dst)  vxReleaseImage(&dst);
    if (dst5) vxReleaseImage(&dst5);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 33: vxuNot
// ---------------------------------------------------------------------------
static int test_vxuNot(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuNot ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 0xAA);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuNot(context, src, dst));
    printf("  PASS: vxuNot\n");

    if (src) vxReleaseImage(&src);
    if (dst) vxReleaseImage(&dst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 34: vxuMultiply
// ---------------------------------------------------------------------------
static int test_vxuMultiply(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuMultiply ---\n");

    vx_image in1 = createUniformU8(context, WIDTH, HEIGHT, 10);
    vx_image in2 = createUniformU8(context, WIDTH, HEIGHT, 20);
    vx_image out = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(in1, "in1");
    CHECK_NULL(in2, "in2");
    CHECK_NULL(out, "out");

    CHECK_STATUS(vxuMultiply(context, in1, in2, 1.0f / 255.0f,
                              VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, out));
    printf("  PASS: vxuMultiply\n");

    if (in1) vxReleaseImage(&in1);
    if (in2) vxReleaseImage(&in2);
    if (out) vxReleaseImage(&out);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 35: vxuAdd
// ---------------------------------------------------------------------------
static int test_vxuAdd(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuAdd ---\n");

    vx_image in1 = createUniformU8(context, WIDTH, HEIGHT, 50);
    vx_image in2 = createUniformU8(context, WIDTH, HEIGHT, 100);
    vx_image out = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_S16);
    CHECK_NULL(in1, "in1");
    CHECK_NULL(in2, "in2");
    CHECK_NULL(out, "out");

    CHECK_STATUS(vxuAdd(context, in1, in2, VX_CONVERT_POLICY_SATURATE, out));
    printf("  PASS: vxuAdd\n");

    if (in1) vxReleaseImage(&in1);
    if (in2) vxReleaseImage(&in2);
    if (out) vxReleaseImage(&out);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 36: vxuSubtract
// ---------------------------------------------------------------------------
static int test_vxuSubtract(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuSubtract ---\n");

    vx_image in1 = createUniformU8(context, WIDTH, HEIGHT, 200);
    vx_image in2 = createUniformU8(context, WIDTH, HEIGHT, 100);
    vx_image out = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_S16);
    CHECK_NULL(in1, "in1");
    CHECK_NULL(in2, "in2");
    CHECK_NULL(out, "out");

    CHECK_STATUS(vxuSubtract(context, in1, in2, VX_CONVERT_POLICY_SATURATE, out));
    printf("  PASS: vxuSubtract\n");

    if (in1) vxReleaseImage(&in1);
    if (in2) vxReleaseImage(&in2);
    if (out) vxReleaseImage(&out);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 37: vxuAnd
// ---------------------------------------------------------------------------
static int test_vxuAnd(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuAnd ---\n");

    vx_image in1 = createUniformU8(context, WIDTH, HEIGHT, 0xFF);
    vx_image in2 = createUniformU8(context, WIDTH, HEIGHT, 0x0F);
    vx_image out = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(in1, "in1");
    CHECK_NULL(in2, "in2");
    CHECK_NULL(out, "out");

    CHECK_STATUS(vxuAnd(context, in1, in2, out));
    printf("  PASS: vxuAnd\n");

    if (in1) vxReleaseImage(&in1);
    if (in2) vxReleaseImage(&in2);
    if (out) vxReleaseImage(&out);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 38: vxuOr
// ---------------------------------------------------------------------------
static int test_vxuOr(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuOr ---\n");

    vx_image in1 = createUniformU8(context, WIDTH, HEIGHT, 0xF0);
    vx_image in2 = createUniformU8(context, WIDTH, HEIGHT, 0x0F);
    vx_image out = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(in1, "in1");
    CHECK_NULL(in2, "in2");
    CHECK_NULL(out, "out");

    CHECK_STATUS(vxuOr(context, in1, in2, out));
    printf("  PASS: vxuOr\n");

    if (in1) vxReleaseImage(&in1);
    if (in2) vxReleaseImage(&in2);
    if (out) vxReleaseImage(&out);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 39: vxuXor
// ---------------------------------------------------------------------------
static int test_vxuXor(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuXor ---\n");

    vx_image in1 = createUniformU8(context, WIDTH, HEIGHT, 0xFF);
    vx_image in2 = createUniformU8(context, WIDTH, HEIGHT, 0xAA);
    vx_image out = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    CHECK_NULL(in1, "in1");
    CHECK_NULL(in2, "in2");
    CHECK_NULL(out, "out");

    CHECK_STATUS(vxuXor(context, in1, in2, out));
    printf("  PASS: vxuXor\n");

    if (in1) vxReleaseImage(&in1);
    if (in2) vxReleaseImage(&in2);
    if (out) vxReleaseImage(&out);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 40: vxuAccumulateImage
// ---------------------------------------------------------------------------
static int test_vxuAccumulateImage(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuAccumulateImage ---\n");

    vx_image input = createUniformU8(context, WIDTH, HEIGHT, 10);
    vx_image accum = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_S16);
    CHECK_NULL(input, "input");
    CHECK_NULL(accum, "accum");

    CHECK_STATUS(vxuAccumulateImage(context, input, accum));
    printf("  PASS: vxuAccumulateImage\n");

    if (input) vxReleaseImage(&input);
    if (accum) vxReleaseImage(&accum);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 41: vxuAccumulateWeightedImage
// ---------------------------------------------------------------------------
static int test_vxuAccumulateWeightedImage(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuAccumulateWeightedImage ---\n");

    vx_image input = createUniformU8(context, WIDTH, HEIGHT, 200);
    vx_image accum = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_float32 alpha_val = 0.5f;
    vx_scalar alpha = vxCreateScalar(context, VX_TYPE_FLOAT32, &alpha_val);
    CHECK_NULL(input, "input");
    CHECK_NULL(accum, "accum");
    CHECK_NULL(alpha, "alpha");

    CHECK_STATUS(vxuAccumulateWeightedImage(context, input, alpha, accum));
    printf("  PASS: vxuAccumulateWeightedImage\n");

    if (input) vxReleaseImage(&input);
    if (accum) vxReleaseImage(&accum);
    if (alpha) vxReleaseScalar(&alpha);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 42: vxuAccumulateSquareImage
// ---------------------------------------------------------------------------
static int test_vxuAccumulateSquareImage(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuAccumulateSquareImage ---\n");

    vx_image input = createUniformU8(context, WIDTH, HEIGHT, 10);
    vx_image accum = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_S16);
    vx_uint32 shift_val = 0;
    vx_scalar shift = vxCreateScalar(context, VX_TYPE_UINT32, &shift_val);
    CHECK_NULL(input, "input");
    CHECK_NULL(accum, "accum");
    CHECK_NULL(shift, "shift");

    CHECK_STATUS(vxuAccumulateSquareImage(context, input, shift, accum));
    printf("  PASS: vxuAccumulateSquareImage\n");

    if (input) vxReleaseImage(&input);
    if (accum) vxReleaseImage(&accum);
    if (shift) vxReleaseScalar(&shift);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 43: vxuCannyEdgeDetector
// ---------------------------------------------------------------------------
static int test_vxuCannyEdgeDetector(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuCannyEdgeDetector ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_threshold hyst = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE,
                                                   VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");
    CHECK_NULL(hyst, "hyst");

    // Set hysteresis thresholds
    vx_pixel_value_t lower, upper;
    memset(&lower, 0, sizeof(lower));
    memset(&upper, 0, sizeof(upper));
    lower.U8 = 80;
    upper.U8 = 200;
    CHECK_STATUS(vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER,
                                          &lower, sizeof(vx_int32)));
    CHECK_STATUS(vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER,
                                          &upper, sizeof(vx_int32)));

    CHECK_STATUS(vxuCannyEdgeDetector(context, src, hyst, 3, VX_NORM_L1, dst));
    printf("  PASS: vxuCannyEdgeDetector\n");

    if (src)  vxReleaseImage(&src);
    if (dst)  vxReleaseImage(&dst);
    if (hyst) vxReleaseThreshold(&hyst);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 44: vxuScaleImage
// ---------------------------------------------------------------------------
static int test_vxuScaleImage(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuScaleImage ---\n");

    vx_image src = createUniformU8(context, WIDTH, HEIGHT, 128);
    vx_image dst = vxCreateImage(context, WIDTH / 2, HEIGHT / 2, VX_DF_IMAGE_U8);
    CHECK_NULL(src, "src");
    CHECK_NULL(dst, "dst");

    CHECK_STATUS(vxuScaleImage(context, src, dst, VX_INTERPOLATION_NEAREST_NEIGHBOR));
    printf("  PASS: vxuScaleImage (NEAREST_NEIGHBOR)\n");

    // Also test with bilinear interpolation
    vx_image dst2 = vxCreateImage(context, WIDTH * 2, HEIGHT * 2, VX_DF_IMAGE_U8);
    CHECK_NULL(dst2, "dst2");
    CHECK_STATUS(vxuScaleImage(context, src, dst2, VX_INTERPOLATION_BILINEAR));
    printf("  PASS: vxuScaleImage (BILINEAR)\n");

    if (src)  vxReleaseImage(&src);
    if (dst)  vxReleaseImage(&dst);
    if (dst2) vxReleaseImage(&dst2);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 45: vxuWeightedAverage
// ---------------------------------------------------------------------------
static int test_vxuWeightedAverage(vx_context context) {
    int errors = 0;
    printf("\n--- Test: vxuWeightedAverage ---\n");

    vx_image img1 = createUniformU8(context, WIDTH, HEIGHT, 100);
    vx_image img2 = createUniformU8(context, WIDTH, HEIGHT, 200);
    vx_image out  = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_float32 alpha_val = 0.5f;
    vx_scalar alpha = vxCreateScalar(context, VX_TYPE_FLOAT32, &alpha_val);
    CHECK_NULL(img1, "img1");
    CHECK_NULL(img2, "img2");
    CHECK_NULL(out, "out");
    CHECK_NULL(alpha, "alpha");

    CHECK_STATUS(vxuWeightedAverage(context, img1, alpha, img2, out));
    printf("  PASS: vxuWeightedAverage\n");

    if (img1)  vxReleaseImage(&img1);
    if (img2)  vxReleaseImage(&img2);
    if (out)   vxReleaseImage(&out);
    if (alpha) vxReleaseScalar(&alpha);
    return errors;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    int total_errors = 0;

    printf("OpenVX VXU (Immediate-Mode) API Coverage Test\n");
    printf("==============================================\n");

    vx_context context = vxCreateContext();
    if (!context) {
        printf("FATAL: vxCreateContext failed\n");
        return 1;
    }
    printf("PASS: vxCreateContext\n");

    // Core vision operations
    total_errors += test_vxuColorConvert(context);
    total_errors += test_vxuChannelExtract(context);
    total_errors += test_vxuChannelCombine(context);
    total_errors += test_vxuSobel3x3(context);
    total_errors += test_vxuMagnitude(context);
    total_errors += test_vxuPhase(context);
    total_errors += test_vxuTableLookup(context);
    total_errors += test_vxuHistogram(context);
    total_errors += test_vxuEqualizeHist(context);
    total_errors += test_vxuAbsDiff(context);
    total_errors += test_vxuMeanStdDev(context);
    total_errors += test_vxuThreshold(context);
    total_errors += test_vxuIntegralImage(context);

    // Morphological operations
    total_errors += test_vxuErode3x3(context);
    total_errors += test_vxuDilate3x3(context);
    total_errors += test_vxuMedian3x3(context);
    total_errors += test_vxuBox3x3(context);
    total_errors += test_vxuGaussian3x3(context);
    total_errors += test_vxuNonLinearFilter(context);

    // Convolution and pyramids
    total_errors += test_vxuConvolve(context);
    total_errors += test_vxuGaussianPyramid(context);
    total_errors += test_vxuLaplacianPyramid(context);
    total_errors += test_vxuLaplacianReconstruct(context);

    // Statistics and analysis
    total_errors += test_vxuMinMaxLoc(context);
    total_errors += test_vxuConvertDepth(context);

    // Geometric transformations
    total_errors += test_vxuWarpAffine(context);
    total_errors += test_vxuWarpPerspective(context);

    // Feature detection
    total_errors += test_vxuHarrisCorners(context);
    total_errors += test_vxuFastCorners(context);
    total_errors += test_vxuOpticalFlowPyrLK(context);

    // Remap and scale
    total_errors += test_vxuRemap(context);
    total_errors += test_vxuHalfScaleGaussian(context);
    total_errors += test_vxuScaleImage(context);
    total_errors += test_vxuWeightedAverage(context);

    // Bitwise and arithmetic operations
    total_errors += test_vxuNot(context);
    total_errors += test_vxuMultiply(context);
    total_errors += test_vxuAdd(context);
    total_errors += test_vxuSubtract(context);
    total_errors += test_vxuAnd(context);
    total_errors += test_vxuOr(context);
    total_errors += test_vxuXor(context);

    // Accumulation operations
    total_errors += test_vxuAccumulateImage(context);
    total_errors += test_vxuAccumulateWeightedImage(context);
    total_errors += test_vxuAccumulateSquareImage(context);

    // Edge detection
    total_errors += test_vxuCannyEdgeDetector(context);

    vxReleaseContext(&context);

    printf("\n==============================================\n");
    if (total_errors == 0) {
        printf("RESULT: ALL TESTS PASSED\n");
    } else {
        printf("RESULT: %d ERROR(S) DETECTED\n", total_errors);
    }

    return (total_errors == 0) ? 0 : 1;
}
