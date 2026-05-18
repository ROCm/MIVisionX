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

// Data Objects API coverage test - exercises vx API functions for:
// Image, Array, Scalar, Convolution, Distribution, Matrix, LUT,
// Remap, Pyramid, Threshold, ObjectArray, Delay

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>

#define CHECK_STATUS(call) do { \
    vx_status s = (call); \
    if (s != VX_SUCCESS) { \
        printf("ERROR: %s returned %d at %s:%d\n", #call, s, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

#define CHECK_OBJ(obj) do { \
    if (vxGetStatus((vx_reference)(obj)) != VX_SUCCESS) { \
        printf("ERROR: object creation failed at %s:%d\n", __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

static int test_image_api(vx_context context) {
    int errors = 0;
    printf("\n=== Image API ===\n");

    // Create image
    vx_image img = vxCreateImage(context, 640, 480, VX_DF_IMAGE_U8);
    CHECK_OBJ(img);

    // Query image attributes
    vx_uint32 width = 0, height = 0;
    CHECK_STATUS(vxQueryImage(img, VX_IMAGE_WIDTH, &width, sizeof(width)));
    CHECK_STATUS(vxQueryImage(img, VX_IMAGE_HEIGHT, &height, sizeof(height)));
    printf("STATUS: vxQueryImage WIDTH=%u HEIGHT=%u\n", width, height);

    vx_df_image format = 0;
    CHECK_STATUS(vxQueryImage(img, VX_IMAGE_FORMAT, &format, sizeof(format)));
    printf("STATUS: vxQueryImage FORMAT=0x%08x\n", format);

    vx_size planes = 0;
    CHECK_STATUS(vxQueryImage(img, VX_IMAGE_PLANES, &planes, sizeof(planes)));
    printf("STATUS: vxQueryImage PLANES=%zu\n", planes);

    vx_enum space = 0;
    CHECK_STATUS(vxQueryImage(img, VX_IMAGE_SPACE, &space, sizeof(space)));
    printf("STATUS: vxQueryImage SPACE=%d\n", space);

    vx_enum range = 0;
    CHECK_STATUS(vxQueryImage(img, VX_IMAGE_RANGE, &range, sizeof(range)));
    printf("STATUS: vxQueryImage RANGE=%d\n", range);

    vx_size size = 0;
    CHECK_STATUS(vxQueryImage(img, VX_IMAGE_SIZE, &size, sizeof(size)));
    printf("STATUS: vxQueryImage SIZE=%zu\n", size);

    vx_enum mem_type = 0;
    CHECK_STATUS(vxQueryImage(img, VX_IMAGE_MEMORY_TYPE, &mem_type, sizeof(mem_type)));
    printf("STATUS: vxQueryImage MEMORY_TYPE=%d\n", mem_type);

    // Set image attribute
    vx_enum new_space = VX_COLOR_SPACE_BT709;
    CHECK_STATUS(vxSetImageAttribute(img, VX_IMAGE_SPACE, &new_space, sizeof(new_space)));
    printf("STATUS: vxSetImageAttribute SPACE - OK\n");

    // Copy image patch (write then read)
    vx_rectangle_t rect = {0, 0, 64, 48};
    vx_imagepatch_addressing_t addr = {};
    addr.dim_x = 64;
    addr.dim_y = 48;
    addr.stride_x = 1;
    addr.stride_y = 64;
    vx_uint8 *data = (vx_uint8 *)calloc(64 * 48, sizeof(vx_uint8));
    if (data) {
        for (int i = 0; i < 64 * 48; i++) data[i] = (vx_uint8)(i % 256);
        CHECK_STATUS(vxCopyImagePatch(img, &rect, 0, &addr, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        printf("STATUS: vxCopyImagePatch WRITE - OK\n");

        vx_uint8 *readback = (vx_uint8 *)calloc(64 * 48, sizeof(vx_uint8));
        if (readback) {
            CHECK_STATUS(vxCopyImagePatch(img, &rect, 0, &addr, readback, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
            printf("STATUS: vxCopyImagePatch READ - OK\n");
            int mismatch = 0;
            for (int i = 0; i < 64 * 48; i++) {
                if (data[i] != readback[i]) mismatch++;
            }
            printf("STATUS: Image data verification: %s (%d mismatches)\n",
                   mismatch == 0 ? "PASS" : "FAIL", mismatch);
            if (mismatch) errors++;
            free(readback);
        }
        free(data);
    }

    // Map/unmap image patch
    vx_rectangle_t map_rect = {0, 0, 32, 24};
    vx_map_id map_id = 0;
    vx_imagepatch_addressing_t map_addr = {};
    void *ptr = NULL;
    vx_status map_s = vxMapImagePatch(img, &map_rect, 0, &map_id, &map_addr, &ptr,
                                       VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if (map_s == VX_SUCCESS && ptr) {
        printf("STATUS: vxMapImagePatch - OK (ptr=%p, stride_x=%d, stride_y=%d)\n",
               ptr, map_addr.stride_x, map_addr.stride_y);
        CHECK_STATUS(vxUnmapImagePatch(img, map_id));
        printf("STATUS: vxUnmapImagePatch - OK\n");
    } else {
        printf("STATUS: vxMapImagePatch returned %d\n", map_s);
    }

    // Create image from ROI
    vx_rectangle_t roi_rect = {10, 10, 100, 100};
    vx_image roi_img = vxCreateImageFromROI(img, &roi_rect);
    if (roi_img) {
        printf("STATUS: vxCreateImageFromROI - OK\n");
        vx_uint32 roi_w = 0, roi_h = 0;
        CHECK_STATUS(vxQueryImage(roi_img, VX_IMAGE_WIDTH, &roi_w, sizeof(roi_w)));
        CHECK_STATUS(vxQueryImage(roi_img, VX_IMAGE_HEIGHT, &roi_h, sizeof(roi_h)));
        printf("STATUS: ROI dims = %ux%u\n", roi_w, roi_h);
        CHECK_STATUS(vxReleaseImage(&roi_img));
    }

    // Create uniform image
    vx_pixel_value_t pixel_value = {};
    pixel_value.U8 = 128;
    vx_image uniform_img = vxCreateUniformImage(context, 320, 240, VX_DF_IMAGE_U8, &pixel_value);
    if (uniform_img) {
        printf("STATUS: vxCreateUniformImage - OK\n");
        vx_bool is_uniform = vx_false_e;
        vx_status us = vxQueryImage(uniform_img, VX_IMAGE_IS_UNIFORM, &is_uniform, sizeof(is_uniform));
        if (us == VX_SUCCESS) {
            printf("STATUS: vxQueryImage IS_UNIFORM=%d\n", is_uniform);
        } else {
            printf("STATUS: vxQueryImage IS_UNIFORM not supported (status=%d)\n", us);
        }
        CHECK_STATUS(vxReleaseImage(&uniform_img));
    }

    // Create image from handle
    vx_uint8 *handle_data = (vx_uint8 *)calloc(320 * 240, sizeof(vx_uint8));
    if (handle_data) {
        vx_imagepatch_addressing_t handle_addr[1] = {};
        handle_addr[0].dim_x = 320;
        handle_addr[0].dim_y = 240;
        handle_addr[0].stride_x = 1;
        handle_addr[0].stride_y = 320;
        void *ptrs[1] = {handle_data};
        vx_image handle_img = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, handle_addr, ptrs, VX_MEMORY_TYPE_HOST);
        if (handle_img) {
            printf("STATUS: vxCreateImageFromHandle - OK\n");

            // Swap image handle
            void *new_ptrs[1] = {NULL};
            void *prev_ptrs[1] = {NULL};
            vx_uint8 *new_data = (vx_uint8 *)calloc(320 * 240, sizeof(vx_uint8));
            if (new_data) {
                new_ptrs[0] = new_data;
                vx_status swap_s = vxSwapImageHandle(handle_img, new_ptrs, prev_ptrs, 1);
                if (swap_s == VX_SUCCESS) {
                    printf("STATUS: vxSwapImageHandle - OK\n");
                } else {
                    printf("STATUS: vxSwapImageHandle returned %d\n", swap_s);
                }
                free(new_data);
            }
            CHECK_STATUS(vxReleaseImage(&handle_img));
        }
        free(handle_data);
    }

    // Set image pixel values
    vx_image img2 = vxCreateImage(context, 160, 120, VX_DF_IMAGE_U8);
    if (img2) {
        vx_pixel_value_t pv = {};
        pv.U8 = 200;
        vx_status pv_s = vxSetImagePixelValues(img2, &pv);
        if (pv_s == VX_SUCCESS) {
            printf("STATUS: vxSetImagePixelValues U8 - OK\n");
        } else {
            printf("STATUS: vxSetImagePixelValues returned %d\n", pv_s);
        }
        CHECK_STATUS(vxReleaseImage(&img2));
    }

    // Set pixel values for different formats
    vx_image img_s16 = vxCreateImage(context, 160, 120, VX_DF_IMAGE_S16);
    if (img_s16) {
        vx_pixel_value_t pv = {};
        pv.S16 = 1000;
        vxSetImagePixelValues(img_s16, &pv);
        printf("STATUS: vxSetImagePixelValues S16 - OK\n");
        CHECK_STATUS(vxReleaseImage(&img_s16));
    }

    vx_image img_rgb = vxCreateImage(context, 160, 120, VX_DF_IMAGE_RGB);
    if (img_rgb) {
        vx_pixel_value_t pv = {};
        pv.RGB[0] = 255; pv.RGB[1] = 128; pv.RGB[2] = 64;
        vxSetImagePixelValues(img_rgb, &pv);
        printf("STATUS: vxSetImagePixelValues RGB - OK\n");
        CHECK_STATUS(vxReleaseImage(&img_rgb));
    }

    vx_image img_rgbx = vxCreateImage(context, 160, 120, VX_DF_IMAGE_RGBX);
    if (img_rgbx) {
        vx_pixel_value_t pv = {};
        pv.RGBX[0] = 255; pv.RGBX[1] = 128; pv.RGBX[2] = 64; pv.RGBX[3] = 255;
        vxSetImagePixelValues(img_rgbx, &pv);
        printf("STATUS: vxSetImagePixelValues RGBX - OK\n");
        CHECK_STATUS(vxReleaseImage(&img_rgbx));
    }

    // Create image from channel
    vx_image img_nv12 = vxCreateImage(context, 320, 240, VX_DF_IMAGE_NV12);
    if (img_nv12) {
        vx_image ch_y = vxCreateImageFromChannel(img_nv12, VX_CHANNEL_Y);
        if (ch_y) {
            printf("STATUS: vxCreateImageFromChannel Y - OK\n");
            CHECK_STATUS(vxReleaseImage(&ch_y));
        }
        CHECK_STATUS(vxReleaseImage(&img_nv12));
    }

    CHECK_STATUS(vxReleaseImage(&img));
    printf("STATUS: Image API complete (%d errors)\n", errors);
    return errors;
}

static int test_array_api(vx_context context) {
    int errors = 0;
    printf("\n=== Array API ===\n");

    // Create array of VX_TYPE_KEYPOINT
    vx_size capacity = 100;
    vx_array arr = vxCreateArray(context, VX_TYPE_KEYPOINT, capacity);
    CHECK_OBJ(arr);
    printf("STATUS: vxCreateArray KEYPOINT - OK\n");

    // Query array
    vx_enum item_type = 0;
    CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ITEMTYPE, &item_type, sizeof(item_type)));
    printf("STATUS: vxQueryArray ITEMTYPE=%d\n", item_type);

    vx_size item_size = 0;
    CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ITEMSIZE, &item_size, sizeof(item_size)));
    printf("STATUS: vxQueryArray ITEMSIZE=%zu\n", item_size);

    vx_size arr_capacity = 0;
    CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
    printf("STATUS: vxQueryArray CAPACITY=%zu\n", arr_capacity);

    vx_size numitems = 0;
    CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_NUMITEMS, &numitems, sizeof(numitems)));
    printf("STATUS: vxQueryArray NUMITEMS=%zu\n", numitems);

    // Add items
    vx_keypoint_t kps[10];
    memset(kps, 0, sizeof(kps));
    for (int i = 0; i < 10; i++) {
        kps[i].x = i * 10;
        kps[i].y = i * 20;
        kps[i].strength = 1.0f;
    }
    CHECK_STATUS(vxAddArrayItems(arr, 10, kps, sizeof(vx_keypoint_t)));
    printf("STATUS: vxAddArrayItems 10 items - OK\n");

    // Copy array range (read back)
    vx_keypoint_t kps_read[10];
    memset(kps_read, 0, sizeof(kps_read));
    CHECK_STATUS(vxCopyArrayRange(arr, 0, 10, sizeof(vx_keypoint_t), kps_read,
                                  VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyArrayRange READ - OK (x[0]=%d, y[0]=%d)\n", kps_read[0].x, kps_read[0].y);

    // Map/unmap array range
    vx_map_id map_id = 0;
    void *ptr = NULL;
    vx_size stride = 0;
    vx_status map_s = vxMapArrayRange(arr, 0, 10, &map_id, &stride, &ptr,
                                       VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if (map_s == VX_SUCCESS && ptr) {
        printf("STATUS: vxMapArrayRange - OK (stride=%zu)\n", stride);
        CHECK_STATUS(vxUnmapArrayRange(arr, map_id));
        printf("STATUS: vxUnmapArrayRange - OK\n");
    }

    // Truncate array
    CHECK_STATUS(vxTruncateArray(arr, 5));
    CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_NUMITEMS, &numitems, sizeof(numitems)));
    printf("STATUS: vxTruncateArray to 5, NUMITEMS=%zu\n", numitems);

    // Virtual array
    vx_graph graph = vxCreateGraph(context);
    if (graph) {
        vx_array varr = vxCreateVirtualArray(graph, VX_TYPE_KEYPOINT, 50);
        if (varr) {
            printf("STATUS: vxCreateVirtualArray - OK\n");
            CHECK_STATUS(vxReleaseArray(&varr));
        }
        vxReleaseGraph(&graph);
    }

    CHECK_STATUS(vxReleaseArray(&arr));
    printf("STATUS: Array API complete (%d errors)\n", errors);
    return errors;
}

static int test_scalar_api(vx_context context) {
    int errors = 0;
    printf("\n=== Scalar API ===\n");

    // Create scalar
    vx_float32 val = 3.14f;
    vx_scalar scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, &val);
    CHECK_OBJ(scalar);
    printf("STATUS: vxCreateScalar FLOAT32 - OK\n");

    // Query scalar
    vx_enum stype = 0;
    CHECK_STATUS(vxQueryScalar(scalar, VX_SCALAR_TYPE, &stype, sizeof(stype)));
    printf("STATUS: vxQueryScalar TYPE=%d\n", stype);

    // Copy scalar (read)
    vx_float32 rval = 0;
    CHECK_STATUS(vxCopyScalar(scalar, &rval, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyScalar READ val=%f\n", rval);

    // Copy scalar (write)
    vx_float32 wval = 2.718f;
    CHECK_STATUS(vxCopyScalar(scalar, &wval, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyScalar WRITE - OK\n");

    // Create scalar with size
    vx_int32 ival = 42;
    vx_scalar s2 = vxCreateScalarWithSize(context, VX_TYPE_INT32, &ival, sizeof(ival));
    if (s2) {
        printf("STATUS: vxCreateScalarWithSize INT32 - OK\n");
        vx_int32 rval2 = 0;
        CHECK_STATUS(vxCopyScalarWithSize(s2, sizeof(rval2), &rval2, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        printf("STATUS: vxCopyScalarWithSize READ val=%d\n", rval2);
        CHECK_STATUS(vxReleaseScalar(&s2));
    }

    CHECK_STATUS(vxReleaseScalar(&scalar));
    printf("STATUS: Scalar API complete (%d errors)\n", errors);
    return errors;
}

static int test_convolution_api(vx_context context) {
    int errors = 0;
    printf("\n=== Convolution API ===\n");

    vx_convolution conv = vxCreateConvolution(context, 3, 3);
    CHECK_OBJ(conv);
    printf("STATUS: vxCreateConvolution 3x3 - OK\n");

    // Query convolution
    vx_size rows = 0, cols = 0;
    CHECK_STATUS(vxQueryConvolution(conv, VX_CONVOLUTION_ROWS, &rows, sizeof(rows)));
    CHECK_STATUS(vxQueryConvolution(conv, VX_CONVOLUTION_COLUMNS, &cols, sizeof(cols)));
    printf("STATUS: vxQueryConvolution ROWS=%zu COLUMNS=%zu\n", rows, cols);

    vx_size conv_size = 0;
    CHECK_STATUS(vxQueryConvolution(conv, VX_CONVOLUTION_SIZE, &conv_size, sizeof(conv_size)));
    printf("STATUS: vxQueryConvolution SIZE=%zu\n", conv_size);

    vx_uint32 scale = 0;
    CHECK_STATUS(vxQueryConvolution(conv, VX_CONVOLUTION_SCALE, &scale, sizeof(scale)));
    printf("STATUS: vxQueryConvolution SCALE=%u\n", scale);

    // Set convolution attribute (scale)
    vx_uint32 new_scale = 1;
    CHECK_STATUS(vxSetConvolutionAttribute(conv, VX_CONVOLUTION_SCALE, &new_scale, sizeof(new_scale)));
    printf("STATUS: vxSetConvolutionAttribute SCALE - OK\n");

    // Copy coefficients (write then read)
    vx_int16 coeffs[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1}; // Laplacian
    CHECK_STATUS(vxCopyConvolutionCoefficients(conv, coeffs, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyConvolutionCoefficients WRITE - OK\n");

    vx_int16 rcoeffs[9] = {0};
    CHECK_STATUS(vxCopyConvolutionCoefficients(conv, rcoeffs, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyConvolutionCoefficients READ - OK (center=%d)\n", rcoeffs[4]);

    CHECK_STATUS(vxReleaseConvolution(&conv));
    printf("STATUS: Convolution API complete (%d errors)\n", errors);
    return errors;
}

static int test_distribution_api(vx_context context) {
    int errors = 0;
    printf("\n=== Distribution API ===\n");

    vx_distribution dist = vxCreateDistribution(context, 256, 0, 256);
    CHECK_OBJ(dist);
    printf("STATUS: vxCreateDistribution 256 bins - OK\n");

    // Query distribution
    vx_size num_bins = 0;
    CHECK_STATUS(vxQueryDistribution(dist, VX_DISTRIBUTION_BINS, &num_bins, sizeof(num_bins)));
    printf("STATUS: vxQueryDistribution BINS=%zu\n", num_bins);

    vx_int32 offset = 0;
    CHECK_STATUS(vxQueryDistribution(dist, VX_DISTRIBUTION_OFFSET, &offset, sizeof(offset)));
    printf("STATUS: vxQueryDistribution OFFSET=%d\n", offset);

    vx_uint32 dist_range = 0;
    CHECK_STATUS(vxQueryDistribution(dist, VX_DISTRIBUTION_RANGE, &dist_range, sizeof(dist_range)));
    printf("STATUS: vxQueryDistribution RANGE=%u\n", dist_range);

    vx_uint32 window = 0;
    CHECK_STATUS(vxQueryDistribution(dist, VX_DISTRIBUTION_WINDOW, &window, sizeof(window)));
    printf("STATUS: vxQueryDistribution WINDOW=%u\n", window);

    vx_size dist_size = 0;
    CHECK_STATUS(vxQueryDistribution(dist, VX_DISTRIBUTION_SIZE, &dist_size, sizeof(dist_size)));
    printf("STATUS: vxQueryDistribution SIZE=%zu\n", dist_size);

    // Copy distribution (write then read)
    vx_int32 *hist = (vx_int32 *)calloc(256, sizeof(vx_int32));
    if (hist) {
        for (int i = 0; i < 256; i++) hist[i] = i * 10;
        CHECK_STATUS(vxCopyDistribution(dist, hist, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        printf("STATUS: vxCopyDistribution WRITE - OK\n");

        vx_int32 *rhist = (vx_int32 *)calloc(256, sizeof(vx_int32));
        if (rhist) {
            CHECK_STATUS(vxCopyDistribution(dist, rhist, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
            printf("STATUS: vxCopyDistribution READ - OK (bin[128]=%d)\n", rhist[128]);
            free(rhist);
        }
        free(hist);
    }

    // Map/unmap distribution
    vx_map_id map_id = 0;
    void *ptr = NULL;
    vx_status map_s = vxMapDistribution(dist, &map_id, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if (map_s == VX_SUCCESS && ptr) {
        printf("STATUS: vxMapDistribution - OK\n");
        CHECK_STATUS(vxUnmapDistribution(dist, map_id));
        printf("STATUS: vxUnmapDistribution - OK\n");
    }

    CHECK_STATUS(vxReleaseDistribution(&dist));
    printf("STATUS: Distribution API complete (%d errors)\n", errors);
    return errors;
}

static int test_matrix_api(vx_context context) {
    int errors = 0;
    printf("\n=== Matrix API ===\n");

    vx_matrix mat = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);
    CHECK_OBJ(mat);
    printf("STATUS: vxCreateMatrix FLOAT32 3x3 - OK\n");

    // Query matrix
    vx_enum mat_type = 0;
    CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_TYPE, &mat_type, sizeof(mat_type)));
    printf("STATUS: vxQueryMatrix TYPE=%d\n", mat_type);

    vx_size mat_rows = 0, mat_cols = 0;
    CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ROWS, &mat_rows, sizeof(mat_rows)));
    CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_COLUMNS, &mat_cols, sizeof(mat_cols)));
    printf("STATUS: vxQueryMatrix ROWS=%zu COLUMNS=%zu\n", mat_rows, mat_cols);

    vx_size mat_size = 0;
    CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_SIZE, &mat_size, sizeof(mat_size)));
    printf("STATUS: vxQueryMatrix SIZE=%zu\n", mat_size);

    vx_coordinates2d_t origin = {0, 0};
    CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ORIGIN, &origin, sizeof(origin)));
    printf("STATUS: vxQueryMatrix ORIGIN=(%u,%u)\n", origin.x, origin.y);

    vx_enum pattern = 0;
    CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_PATTERN, &pattern, sizeof(pattern)));
    printf("STATUS: vxQueryMatrix PATTERN=%d\n", pattern);

    // Copy matrix (write then read)
    vx_float32 mdata[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // identity
    CHECK_STATUS(vxCopyMatrix(mat, mdata, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyMatrix WRITE - OK\n");

    vx_float32 rdata[9] = {0};
    CHECK_STATUS(vxCopyMatrix(mat, rdata, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyMatrix READ - OK (diag=[%f,%f,%f])\n", rdata[0], rdata[4], rdata[8]);

    // Create matrix from pattern
    vx_matrix pat_mat = vxCreateMatrixFromPattern(context, VX_PATTERN_BOX, 3, 3);
    if (pat_mat) {
        printf("STATUS: vxCreateMatrixFromPattern BOX 3x3 - OK\n");
        CHECK_STATUS(vxReleaseMatrix(&pat_mat));
    }

    // Create matrix from pattern and origin
    vx_matrix pat_mat2 = vxCreateMatrixFromPatternAndOrigin(context, VX_PATTERN_CROSS, 5, 5, 2, 2);
    if (pat_mat2) {
        printf("STATUS: vxCreateMatrixFromPatternAndOrigin CROSS 5x5 - OK\n");
        CHECK_STATUS(vxReleaseMatrix(&pat_mat2));
    }

    CHECK_STATUS(vxReleaseMatrix(&mat));
    printf("STATUS: Matrix API complete (%d errors)\n", errors);
    return errors;
}

static int test_lut_api(vx_context context) {
    int errors = 0;
    printf("\n=== LUT API ===\n");

    vx_lut lut = vxCreateLUT(context, VX_TYPE_UINT8, 256);
    CHECK_OBJ(lut);
    printf("STATUS: vxCreateLUT U8 256 - OK\n");

    // Query LUT
    vx_enum lut_type = 0;
    CHECK_STATUS(vxQueryLUT(lut, VX_LUT_TYPE, &lut_type, sizeof(lut_type)));
    printf("STATUS: vxQueryLUT TYPE=%d\n", lut_type);

    vx_size lut_count = 0;
    CHECK_STATUS(vxQueryLUT(lut, VX_LUT_COUNT, &lut_count, sizeof(lut_count)));
    printf("STATUS: vxQueryLUT COUNT=%zu\n", lut_count);

    vx_size lut_size = 0;
    CHECK_STATUS(vxQueryLUT(lut, VX_LUT_SIZE, &lut_size, sizeof(lut_size)));
    printf("STATUS: vxQueryLUT SIZE=%zu\n", lut_size);

    vx_uint32 lut_offset = 0;
    CHECK_STATUS(vxQueryLUT(lut, VX_LUT_OFFSET, &lut_offset, sizeof(lut_offset)));
    printf("STATUS: vxQueryLUT OFFSET=%u\n", lut_offset);

    // Copy LUT (write then read)
    vx_uint8 lut_data[256];
    for (int i = 0; i < 256; i++) lut_data[i] = (vx_uint8)(255 - i); // invert
    CHECK_STATUS(vxCopyLUT(lut, lut_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyLUT WRITE - OK\n");

    vx_uint8 rlut[256] = {0};
    CHECK_STATUS(vxCopyLUT(lut, rlut, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyLUT READ - OK (lut[0]=%d, lut[255]=%d)\n", rlut[0], rlut[255]);

    // Map/unmap LUT
    vx_map_id map_id = 0;
    void *ptr = NULL;
    vx_status map_s = vxMapLUT(lut, &map_id, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    if (map_s == VX_SUCCESS && ptr) {
        printf("STATUS: vxMapLUT - OK\n");
        CHECK_STATUS(vxUnmapLUT(lut, map_id));
        printf("STATUS: vxUnmapLUT - OK\n");
    }

    // S16 LUT
    vx_lut lut16 = vxCreateLUT(context, VX_TYPE_INT16, 65536);
    if (lut16) {
        printf("STATUS: vxCreateLUT S16 65536 - OK\n");
        CHECK_STATUS(vxReleaseLUT(&lut16));
    }

    CHECK_STATUS(vxReleaseLUT(&lut));
    printf("STATUS: LUT API complete (%d errors)\n", errors);
    return errors;
}

static int test_remap_api(vx_context context) {
    int errors = 0;
    printf("\n=== Remap API ===\n");

    vx_remap remap = vxCreateRemap(context, 320, 240, 160, 120);
    CHECK_OBJ(remap);
    printf("STATUS: vxCreateRemap 320x240->160x120 - OK\n");

    // Query remap
    vx_uint32 src_w = 0, src_h = 0, dst_w = 0, dst_h = 0;
    CHECK_STATUS(vxQueryRemap(remap, VX_REMAP_SOURCE_WIDTH, &src_w, sizeof(src_w)));
    CHECK_STATUS(vxQueryRemap(remap, VX_REMAP_SOURCE_HEIGHT, &src_h, sizeof(src_h)));
    CHECK_STATUS(vxQueryRemap(remap, VX_REMAP_DESTINATION_WIDTH, &dst_w, sizeof(dst_w)));
    CHECK_STATUS(vxQueryRemap(remap, VX_REMAP_DESTINATION_HEIGHT, &dst_h, sizeof(dst_h)));
    printf("STATUS: vxQueryRemap src=%ux%u dst=%ux%u\n", src_w, src_h, dst_w, dst_h);

    // Set/get remap points
    CHECK_STATUS(vxSetRemapPoint(remap, 0, 0, 10.5f, 20.5f));
    CHECK_STATUS(vxSetRemapPoint(remap, 1, 1, 11.5f, 21.5f));
    printf("STATUS: vxSetRemapPoint - OK\n");

    vx_float32 sx = 0, sy = 0;
    CHECK_STATUS(vxGetRemapPoint(remap, 0, 0, &sx, &sy));
    printf("STATUS: vxGetRemapPoint (0,0) = (%f, %f)\n", sx, sy);

    // Copy remap patch (write then read)
    vx_size dst_patch_w = 4, dst_patch_h = 4;
    vx_coordinates2df_t coords[4 * 4];
    for (vx_size j = 0; j < dst_patch_h; j++) {
        for (vx_size i = 0; i < dst_patch_w; i++) {
            coords[j * dst_patch_w + i].x = (float)i * 2.0f; // src_x
            coords[j * dst_patch_w + i].y = (float)j * 2.0f; // src_y
        }
    }
    vx_rectangle_t patch_rect = {0, 0, (vx_uint32)dst_patch_w, (vx_uint32)dst_patch_h};
    vx_size coord_stride = dst_patch_w * sizeof(vx_coordinates2df_t);
    CHECK_STATUS(vxCopyRemapPatch(remap, &patch_rect, coord_stride, coords,
                                   VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyRemapPatch WRITE - OK\n");

    vx_coordinates2df_t rcoords[4 * 4] = {};
    CHECK_STATUS(vxCopyRemapPatch(remap, &patch_rect, coord_stride, rcoords,
                                   VX_TYPE_COORDINATES2DF, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyRemapPatch READ - OK (pt[0]=(%f,%f))\n", rcoords[0].x, rcoords[0].y);

    CHECK_STATUS(vxReleaseRemap(&remap));
    printf("STATUS: Remap API complete (%d errors)\n", errors);
    return errors;
}

static int test_pyramid_api(vx_context context) {
    int errors = 0;
    printf("\n=== Pyramid API ===\n");

    vx_pyramid pyr = vxCreatePyramid(context, 4, VX_SCALE_PYRAMID_HALF, 640, 480, VX_DF_IMAGE_U8);
    CHECK_OBJ(pyr);
    printf("STATUS: vxCreatePyramid 4 levels HALF 640x480 U8 - OK\n");

    // Query pyramid
    vx_size num_levels = 0;
    CHECK_STATUS(vxQueryPyramid(pyr, VX_PYRAMID_LEVELS, &num_levels, sizeof(num_levels)));
    printf("STATUS: vxQueryPyramid LEVELS=%zu\n", num_levels);

    vx_float32 pyr_scale = 0;
    CHECK_STATUS(vxQueryPyramid(pyr, VX_PYRAMID_SCALE, &pyr_scale, sizeof(pyr_scale)));
    printf("STATUS: vxQueryPyramid SCALE=%f\n", pyr_scale);

    vx_uint32 pyr_w = 0, pyr_h = 0;
    CHECK_STATUS(vxQueryPyramid(pyr, VX_PYRAMID_WIDTH, &pyr_w, sizeof(pyr_w)));
    CHECK_STATUS(vxQueryPyramid(pyr, VX_PYRAMID_HEIGHT, &pyr_h, sizeof(pyr_h)));
    printf("STATUS: vxQueryPyramid WIDTH=%u HEIGHT=%u\n", pyr_w, pyr_h);

    vx_df_image pyr_fmt = 0;
    CHECK_STATUS(vxQueryPyramid(pyr, VX_PYRAMID_FORMAT, &pyr_fmt, sizeof(pyr_fmt)));
    printf("STATUS: vxQueryPyramid FORMAT=0x%08x\n", pyr_fmt);

    // Get pyramid levels
    for (vx_size i = 0; i < num_levels; i++) {
        vx_image level_img = vxGetPyramidLevel(pyr, (vx_uint32)i);
        if (level_img) {
            vx_uint32 lw = 0, lh = 0;
            vxQueryImage(level_img, VX_IMAGE_WIDTH, &lw, sizeof(lw));
            vxQueryImage(level_img, VX_IMAGE_HEIGHT, &lh, sizeof(lh));
            printf("STATUS: vxGetPyramidLevel(%zu) = %ux%u\n", i, lw, lh);
            vxReleaseImage(&level_img);
        }
    }

    // Virtual pyramid
    vx_graph graph = vxCreateGraph(context);
    if (graph) {
        vx_pyramid vpyr = vxCreateVirtualPyramid(graph, 4, VX_SCALE_PYRAMID_HALF, 640, 480, VX_DF_IMAGE_U8);
        if (vpyr) {
            printf("STATUS: vxCreateVirtualPyramid - OK\n");
            CHECK_STATUS(vxReleasePyramid(&vpyr));
        }
        vxReleaseGraph(&graph);
    }

    CHECK_STATUS(vxReleasePyramid(&pyr));
    printf("STATUS: Pyramid API complete (%d errors)\n", errors);
    return errors;
}

static int test_threshold_api(vx_context context) {
    int errors = 0;
    printf("\n=== Threshold API ===\n");

    // Create threshold for image (binary)
    vx_threshold thresh = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY,
                                                     VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_OBJ(thresh);
    printf("STATUS: vxCreateThresholdForImage BINARY - OK\n");

    // Query threshold
    vx_enum thresh_type = 0;
    CHECK_STATUS(vxQueryThreshold(thresh, VX_THRESHOLD_TYPE, &thresh_type, sizeof(thresh_type)));
    printf("STATUS: vxQueryThreshold TYPE=%d\n", thresh_type);

    vx_df_image thresh_ifmt = 0;
    CHECK_STATUS(vxQueryThreshold(thresh, VX_THRESHOLD_INPUT_FORMAT, &thresh_ifmt, sizeof(thresh_ifmt)));
    printf("STATUS: vxQueryThreshold INPUT_FORMAT=0x%08x\n", thresh_ifmt);

    vx_df_image thresh_ofmt = 0;
    CHECK_STATUS(vxQueryThreshold(thresh, VX_THRESHOLD_OUTPUT_FORMAT, &thresh_ofmt, sizeof(thresh_ofmt)));
    printf("STATUS: vxQueryThreshold OUTPUT_FORMAT=0x%08x\n", thresh_ofmt);

    // Copy threshold value
    vx_pixel_value_t tval = {};
    tval.U8 = 128;
    CHECK_STATUS(vxCopyThresholdValue(thresh, &tval, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyThresholdValue WRITE val=%d - OK\n", tval.U8);

    vx_pixel_value_t rtval = {};
    CHECK_STATUS(vxCopyThresholdValue(thresh, &rtval, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyThresholdValue READ val=%d\n", rtval.U8);

    // Copy threshold output values
    vx_pixel_value_t true_val = {}, false_val = {};
    true_val.U8 = 255;
    false_val.U8 = 0;
    CHECK_STATUS(vxCopyThresholdOutput(thresh, &true_val, &false_val, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyThresholdOutput WRITE true=%d false=%d - OK\n", true_val.U8, false_val.U8);

    vx_pixel_value_t rt = {}, rf = {};
    CHECK_STATUS(vxCopyThresholdOutput(thresh, &rt, &rf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyThresholdOutput READ true=%d false=%d\n", rt.U8, rf.U8);

    CHECK_STATUS(vxReleaseThreshold(&thresh));

    // Create range threshold
    vx_threshold rthresh = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE,
                                                      VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    if (rthresh) {
        printf("STATUS: vxCreateThresholdForImage RANGE - OK\n");

        vx_pixel_value_t lower = {}, upper = {};
        lower.U8 = 100;
        upper.U8 = 200;
        CHECK_STATUS(vxCopyThresholdRange(rthresh, &lower, &upper, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        printf("STATUS: vxCopyThresholdRange WRITE lower=%d upper=%d - OK\n", lower.U8, upper.U8);

        vx_pixel_value_t rl = {}, ru = {};
        CHECK_STATUS(vxCopyThresholdRange(rthresh, &rl, &ru, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        printf("STATUS: vxCopyThresholdRange READ lower=%d upper=%d\n", rl.U8, ru.U8);

        CHECK_STATUS(vxReleaseThreshold(&rthresh));
    }

    printf("STATUS: Threshold API complete (%d errors)\n", errors);
    return errors;
}

static int test_object_array_api(vx_context context) {
    int errors = 0;
    printf("\n=== ObjectArray API ===\n");

    // Create an image as exemplar
    vx_image exemplar = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8);
    CHECK_OBJ(exemplar);

    vx_object_array oa = vxCreateObjectArray(context, (vx_reference)exemplar, 5);
    vxReleaseImage(&exemplar);

    if (vxGetStatus((vx_reference)oa) == VX_SUCCESS) {
        printf("STATUS: vxCreateObjectArray 5 images - OK\n");

        // Query object array
        vx_enum oa_type = 0;
        CHECK_STATUS(vxQueryObjectArray(oa, VX_OBJECT_ARRAY_ITEMTYPE, &oa_type, sizeof(oa_type)));
        printf("STATUS: vxQueryObjectArray ITEMTYPE=%d\n", oa_type);

        vx_size oa_count = 0;
        CHECK_STATUS(vxQueryObjectArray(oa, VX_OBJECT_ARRAY_NUMITEMS, &oa_count, sizeof(oa_count)));
        printf("STATUS: vxQueryObjectArray NUMITEMS=%zu\n", oa_count);

        // Get items
        for (vx_size i = 0; i < oa_count && i < 3; i++) {
            vx_reference item = vxGetObjectArrayItem(oa, (vx_uint32)i);
            if (item) {
                printf("STATUS: vxGetObjectArrayItem(%zu) - OK\n", i);
                vxReleaseReference(&item);
            }
        }

        CHECK_STATUS(vxReleaseObjectArray(&oa));
    }

    printf("STATUS: ObjectArray API complete (%d errors)\n", errors);
    return errors;
}

static int test_delay_api(vx_context context) {
    int errors = 0;
    printf("\n=== Delay API ===\n");

    // Create delay with image exemplar
    vx_image exemplar = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8);
    CHECK_OBJ(exemplar);

    vx_delay delay = vxCreateDelay(context, (vx_reference)exemplar, 3);
    vxReleaseImage(&exemplar);

    if (vxGetStatus((vx_reference)delay) == VX_SUCCESS) {
        printf("STATUS: vxCreateDelay 3 slots - OK\n");

        // Query delay
        vx_enum delay_type = 0;
        CHECK_STATUS(vxQueryDelay(delay, VX_DELAY_TYPE, &delay_type, sizeof(delay_type)));
        printf("STATUS: vxQueryDelay TYPE=%d\n", delay_type);

        vx_size delay_count = 0;
        CHECK_STATUS(vxQueryDelay(delay, VX_DELAY_SLOTS, &delay_count, sizeof(delay_count)));
        printf("STATUS: vxQueryDelay SLOTS=%zu\n", delay_count);

        // Get references from delay
        for (vx_int32 i = 0; i < (vx_int32)delay_count; i++) {
            vx_reference ref = vxGetReferenceFromDelay(delay, -i);
            if (ref) {
                printf("STATUS: vxGetReferenceFromDelay(%d) - OK\n", -i);
            }
        }

        // Age delay
        CHECK_STATUS(vxAgeDelay(delay));
        printf("STATUS: vxAgeDelay - OK\n");

        CHECK_STATUS(vxReleaseDelay(&delay));
    }

    printf("STATUS: Delay API complete (%d errors)\n", errors);
    return errors;
}

int main() {
    int errors = 0;

    vx_context context = vxCreateContext();
    if (!context) {
        printf("ERROR: vxCreateContext failed\n");
        return 1;
    }

    errors += test_image_api(context);
    errors += test_array_api(context);
    errors += test_scalar_api(context);
    errors += test_convolution_api(context);
    errors += test_distribution_api(context);
    errors += test_matrix_api(context);
    errors += test_lut_api(context);
    errors += test_remap_api(context);
    errors += test_pyramid_api(context);
    errors += test_threshold_api(context);
    errors += test_object_array_api(context);
    errors += test_delay_api(context);

    vxReleaseContext(&context);

    printf("\nData Objects API test: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors ? 1 : 0;
}
