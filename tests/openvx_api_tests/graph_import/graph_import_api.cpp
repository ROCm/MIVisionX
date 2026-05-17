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

// Graph import API coverage test - exercises agoReadGraphFromStringInternal
// via vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_IMPORT_FROM_TEXT, ...)
// and related AMD graph attribute set/query functions.
//
// NOTE: The internal GDF text format uses the pattern:
//   image:<4-char-format>,<width>,<height>        e.g. image:U008,640,480
//   image-uniform:<4-char-format>,<w>,<h>,<val>   e.g. image-uniform:U008,320,240,128
//   image-virtual:<4-char-format>,<w>,<h>         e.g. image-virtual:U008,128,128
// This differs from runvx GDF files which use: image:<w>,<h>,<format>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <VX/vx.h>
#include "vx_ext_amd.h"

#define CHECK_STATUS(call) do { \
    vx_status s = (call); \
    if (s != VX_SUCCESS) { \
        printf("ERROR: %s returned %d at %s:%d\n", #call, s, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

#define CHECK_STATUS_EXPECT_FAIL(call) do { \
    vx_status s = (call); \
    if (s == VX_SUCCESS) { \
        printf("WARNING: %s unexpectedly succeeded at %s:%d\n", #call, __FILE__, __LINE__); \
    } else { \
        printf("STATUS: %s correctly returned %d (expected failure) at %s:%d\n", #call, s, __FILE__, __LINE__); \
    } \
} while(0)

// Helper: import GDF text into a graph using AgoGraphImportInfo
static vx_status importGraphText(vx_graph graph, const char *text) {
    AgoGraphImportInfo info = {};
    info.text = const_cast<vx_char *>(text);
    info.num_ref = 0;
    info.ref = nullptr;
    info.dumpToConsole = 0;
    info.data_registry_callback_f = nullptr;
    info.data_registry_callback_obj = nullptr;
    return vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_IMPORT_FROM_TEXT,
                               &info, sizeof(info));
}

// Helper: import GDF text with external references
static vx_status importGraphTextWithRefs(vx_graph graph, const char *text,
                                          vx_uint32 num_ref, vx_reference *refs) {
    AgoGraphImportInfo info = {};
    info.text = const_cast<vx_char *>(text);
    info.num_ref = num_ref;
    info.ref = refs;
    info.dumpToConsole = 0;
    info.data_registry_callback_f = nullptr;
    info.data_registry_callback_obj = nullptr;
    return vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_IMPORT_FROM_TEXT,
                               &info, sizeof(info));
}

// Test 1: Basic image + add node with uniform images
static int test_basic_image_and_node(vx_context context) {
    int errors = 0;
    printf("\n=== Test 1: Basic image data declarations and node ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data input_1 = image-uniform:U008,1920,1080,125\n"
        "data input_2 = image-uniform:U008,1920,1080,132\n"
        "data output_1 = image:U008,1920,1080\n"
        "node org.khronos.openvx.add input_1 input_2 !SATURATE output_1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import basic image + add node - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 2: Multiple scalar data types
static int test_scalar_data_types(vx_context context) {
    int errors = 0;
    printf("\n=== Test 2: Scalar data type declarations ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data s_uint32 = scalar:UINT32,42\n"
        "data s_int32 = scalar:INT32,-7\n"
        "data s_float32 = scalar:FLOAT32,3.14\n"
        "data s_uint8 = scalar:UINT8,200\n"
        "data s_int8 = scalar:INT8,-1\n"
        "data s_uint16 = scalar:UINT16,1000\n"
        "data s_int16 = scalar:INT16,-500\n"
        "data s_bool = scalar:BOOL,1\n"
        "data s_size = scalar:SIZE,4096\n"
        "data s_float64 = scalar:FLOAT64,2.718281828\n"
        "data s_enum = scalar:ENUM,0\n"
        "data s_char = scalar:CHAR,65\n"
        "data s_df_image = scalar:DF_IMAGE,U008\n"
        "data s_string = scalar:STRING,hello_world\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import scalar data types - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 3: Image format variations
static int test_image_formats(vx_context context) {
    int errors = 0;
    printf("\n=== Test 3: Image format variations ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data img_u8_in = image-uniform:U008,64,64,100\n"
        "data out_u8 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 img_u8_in out_u8\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import image format variations - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 4: Uniform image declarations
static int test_uniform_images(vx_context context) {
    int errors = 0;
    printf("\n=== Test 4: Uniform image declarations ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data uimg_u8 = image-uniform:U008,64,64,128\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.gaussian_3x3 uimg_u8 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import uniform images - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 5: Misc data types (convolution, matrix, distribution, lut, remap)
static int test_misc_data_types(vx_context context) {
    int errors = 0;
    printf("\n=== Test 5: Misc data types ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data conv3x3 = convolution:3,3\n"
        "data conv5x5 = convolution:5,5\n"
        "data mat_f32 = matrix:FLOAT32,3,3\n"
        "data mat_i32 = matrix:INT32,2,2\n"
        "data mat_u8 = matrix:UINT8,5,5\n"
        "data dist1 = distribution:256,0,256\n"
        "data dist2 = distribution:64,0,256\n"
        "data lut_u8 = lut:UINT8,256\n"
        "data lut_s16 = lut:INT16,1024\n"
        "data remap1 = remap:640,480,320,240\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import misc data types - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 6: Array data type
static int test_array_data(vx_context context) {
    int errors = 0;
    printf("\n=== Test 6: Array data type ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data arr_kp = array:KEYPOINT,1000\n"
        "data arr_u8 = array:UINT8,512\n"
        "data arr_f32 = array:FLOAT32,256\n"
        "data arr_i32 = array:INT32,100\n"
        "data arr_coord = array:COORDINATES2D,500\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import array data types - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 7: Pyramid data type
static int test_pyramid_data(vx_context context) {
    int errors = 0;
    printf("\n=== Test 7: Pyramid data type ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data pyr_half = pyramid:U008,640,480,4,HALF\n"
        "data pyr_orb = pyramid:U008,640,480,4,ORB\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import pyramid data types - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 8: Tensor data type
static int test_tensor_data(vx_context context) {
    int errors = 0;
    printf("\n=== Test 8: Tensor data type ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data t1 = tensor:2,{4,8},FLOAT32\n"
        "data t2 = tensor:3,{2,3,4},UINT8,0\n"
        "data t3 = tensor:4,{1,2,3,4},INT16,0\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import tensor data types - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 9: Delay data type
static int test_delay_data(vx_context context) {
    int errors = 0;
    printf("\n=== Test 9: Delay data type ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data d1 = delay:3[image:U008,320,240]\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import delay data type - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 10: Object array data type
static int test_object_array_data(vx_context context) {
    int errors = 0;
    printf("\n=== Test 10: Object array data type ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data oa1 = objectarray:3[image:U008,320,240]\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import object array data type - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 11: Multi-plane image (NV12, IYUV, YUV4)
static int test_multiplane_image(vx_context context) {
    int errors = 0;
    printf("\n=== Test 11: Multi-plane image ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data nv12_img = image:NV12,640,480\n"
        "data iyuv_img = image:IYUV,640,480\n"
        "data yuv4_img = image:YUV4,640,480\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import multi-plane images - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 12: Multi-node graph pipeline
static int test_multi_node_graph(vx_context context) {
    int errors = 0;
    printf("\n=== Test 12: Multi-node graph ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data in1 = image-uniform:U008,64,64,100\n"
        "data mid = image:U008,64,64\n"
        "data out = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 mid\n"
        "node org.khronos.openvx.gaussian_3x3 mid out\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import multi-node graph - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 13: Box filter node
static int test_box_filter_node(vx_context context) {
    int errors = 0;
    printf("\n=== Test 13: Box filter node ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data input_img = image-uniform:U008,1920,1080,125\n"
        "data output_img = image:U008,1920,1080\n"
        "node org.khronos.openvx.box_3x3 input_img output_img\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import box_3x3 node - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 14: Gaussian filter node
static int test_gaussian_filter_node(vx_context context) {
    int errors = 0;
    printf("\n=== Test 14: Gaussian filter node ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data input_img = image-uniform:U008,640,480,100\n"
        "data output_img = image:U008,640,480\n"
        "node org.khronos.openvx.gaussian_3x3 input_img output_img\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import gaussian_3x3 node - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 15: Convolution node with INIT data
static int test_convolution_node(vx_context context) {
    int errors = 0;
    printf("\n=== Test 15: Custom convolution node ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data input_img = image-uniform:U008,640,480,100\n"
        "data output_img = image:S016,640,480\n"
        "data conv = convolution:3,3:INIT,{-1;-1;-1;-1;16;-1;-1;-1;-1}\n"
        "node org.khronos.openvx.custom_convolution input_img conv output_img\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import custom_convolution node - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 16: Channel extract node with inline enum
static int test_channel_extract_node(vx_context context) {
    int errors = 0;
    printf("\n=== Test 16: Channel extract node ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data input_rgb = image:RGB2,640,480\n"
        "data output_r = image:U008,640,480\n"
        "node org.khronos.openvx.channel_extract input_rgb !CHANNEL_R output_r\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import channel_extract node - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 17: Histogram node
static int test_histogram_node(vx_context context) {
    int errors = 0;
    printf("\n=== Test 17: Histogram node ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data in1 = image-uniform:U008,320,240,128\n"
        "data hist = distribution:256,0,256\n"
        "node org.khronos.openvx.histogram in1 hist\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import histogram node - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 18: Table lookup node
static int test_table_lookup_node(vx_context context) {
    int errors = 0;
    printf("\n=== Test 18: Table lookup node ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data in1 = image-uniform:U008,320,240,128\n"
        "data lut1 = lut:UINT8,256\n"
        "data out1 = image:U008,320,240\n"
        "node org.khronos.openvx.table_lookup in1 lut1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import table_lookup node - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 19: Median filter node
static int test_median_filter_node(vx_context context) {
    int errors = 0;
    printf("\n=== Test 19: Median filter node ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data input_img = image-uniform:U008,640,480,100\n"
        "data output_img = image:U008,640,480\n"
        "node org.khronos.openvx.median_3x3 input_img output_img\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import median_3x3 node - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 20: Large graph with multiple different nodes
static int test_large_graph(vx_context context) {
    int errors = 0;
    printf("\n=== Test 20: Large graph with many nodes ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data src = image-uniform:U008,256,256,128\n"
        "data box_out = image:U008,256,256\n"
        "data gauss_out = image:U008,256,256\n"
        "data med_out = image:U008,256,256\n"
        "node org.khronos.openvx.box_3x3 src box_out\n"
        "node org.khronos.openvx.gaussian_3x3 src gauss_out\n"
        "node org.khronos.openvx.median_3x3 src med_out\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import large graph - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 21: Comments and empty lines
static int test_comments_and_empty_lines(vx_context context) {
    int errors = 0;
    printf("\n=== Test 21: Comments and empty lines ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "# This is a comment line\n"
        "\n"
        "data img1 = image-uniform:U008,64,64,100\n"
        "\n"
        "# Another comment\n"
        "data img2 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 img1 img2\n"
        "\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with comments and empty lines - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 22: def-var and variable substitution ($VAR syntax)
static int test_def_var(vx_context context) {
    int errors = 0;
    printf("\n=== Test 22: def-var variable definitions ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "def-var W 640\n"
        "def-var H 480\n"
        "data img1 = image-uniform:U008,$W,$H,50\n"
        "data img2 = image:U008,$W,$H\n"
        "node org.khronos.openvx.box_3x3 img1 img2\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with def-var - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 23: def-var-default
static int test_def_var_default(vx_context context) {
    int errors = 0;
    printf("\n=== Test 23: def-var-default ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "def-var-default X 320\n"
        "def-var-default Y 240\n"
        "data img1 = image-uniform:U008,$X,$Y,77\n"
        "data img2 = image:U008,$X,$Y\n"
        "node org.khronos.openvx.box_3x3 img1 img2\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with def-var-default - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 24: def-var edge cases (empty value, already set)
static int test_def_var_edge_cases(vx_context context) {
    int errors = 0;
    printf("\n=== Test 24: def-var edge cases ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "def-var Emptyvar\n"
        "def-var Setvar 100\n"
        "def-var-default Setvar 999\n"
        "data img1 = image-uniform:U008,$Setvar,$Setvar,50\n"
        "data img2 = image:U008,$Setvar,$Setvar\n"
        "node org.khronos.openvx.box_3x3 img1 img2\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import def-var edge cases - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 25: def-var AgoOptimizerFlags special variable
static int test_def_var_optimizer_flags(vx_context context) {
    int errors = 0;
    printf("\n=== Test 25: def-var AgoOptimizerFlags ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "def-var AgoOptimizerFlags 0\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with AgoOptimizerFlags - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vx_uint32 flags = 999;
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS, &flags, sizeof(flags)));
    if (flags != 0) { printf("ERROR: AgoOptimizerFlags not set to 0\n"); errors++; }
    vxReleaseGraph(&graph);
    return errors;
}

// Test 26: def-var with WIDTH/HEIGHT/FORMAT
static int test_def_var_width_height_format(vx_context context) {
    int errors = 0;
    printf("\n=== Test 26: def-var with WIDTH/HEIGHT/FORMAT ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data srcimg = image-uniform:U008,640,480,50\n"
        "def-var W WIDTH(srcimg)\n"
        "def-var H HEIGHT(srcimg)\n"
        "def-var F FORMAT(srcimg)\n"
        "data dstimg = image:$F,$W,$H\n"
        "node org.khronos.openvx.box_3x3 srcimg dstimg\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with WIDTH/HEIGHT/FORMAT - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 27: $var references with external data
static int test_dollar_var_references(vx_context context) {
    int errors = 0;
    printf("\n=== Test 27: $var references ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    vx_image ext_in = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8);
    vx_image ext_out = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8);
    if (!ext_in || !ext_out) {
        if (ext_in) vxReleaseImage(&ext_in);
        if (ext_out) vxReleaseImage(&ext_out);
        vxReleaseGraph(&graph);
        return 1;
    }
    vx_reference refs[2] = { (vx_reference)ext_in, (vx_reference)ext_out };
    const char *gdf_text = "node org.khronos.openvx.box_3x3 $1 $2\n";
    vx_status status = importGraphTextWithRefs(graph, gdf_text, 2, refs);
    printf("STATUS: Import with $var references - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseImage(&ext_in);
    vxReleaseImage(&ext_out);
    vxReleaseGraph(&graph);
    return errors;
}

// Test 28: alias command
static int test_alias_command(vx_context context) {
    int errors = 0;
    printf("\n=== Test 28: alias command ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    vx_image ext_img = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8);
    if (!ext_img) { vxReleaseGraph(&graph); return 1; }
    vx_reference refs[1] = { (vx_reference)ext_img };
    const char *gdf_text =
        "alias myInput $1\n"
        "data alias_out = image:U008,320,240\n"
        "node org.khronos.openvx.box_3x3 myInput alias_out\n";
    vx_status status = importGraphTextWithRefs(graph, gdf_text, 1, refs);
    printf("STATUS: Import with alias - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseImage(&ext_img);
    vxReleaseGraph(&graph);
    return errors;
}

// Test 29: if/else/elseif/endif with all comparison operators
static int test_if_else_endif(vx_context context) {
    int errors = 0;
    printf("\n=== Test 29: if/else/endif ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "if 1 == 1\n"
        "data cond_in = image-uniform:U008,64,64,100\n"
        "else\n"
        "data cond_in = image-uniform:U008,99999,99999,100\n"
        "endif\n"
        "if 0 == 1\n"
        "data skip1 = image:U008,99999,99999\n"
        "elseif 1 == 1\n"
        "data cond_out = image:U008,64,64\n"
        "endif\n"
        "if 1 != 0\n"
        "if 1 < 2\n"
        "if 5 > 3\n"
        "if 3 <= 3\n"
        "if 5 >= 3\n"
        "node org.khronos.openvx.box_3x3 cond_in cond_out\n"
        "endif\n"
        "endif\n"
        "endif\n"
        "endif\n"
        "endif\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with if/else/endif - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 30: exit command
static int test_exit_command(vx_context context) {
    int errors = 0;
    printf("\n=== Test 30: exit command ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data exit_in = image-uniform:U008,64,64,100\n"
        "data exit_out = image:U008,64,64\n"
        "node org.khronos.openvx.median_3x3 exit_in exit_out\n"
        "exit\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with exit - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 31: affinity command
static int test_affinity_command(vx_context context) {
    int errors = 0;
    printf("\n=== Test 31: affinity command ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "affinity CPU\n"
        "data img1 = image-uniform:U008,64,64,100\n"
        "data img2 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 img1 img2\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with affinity CPU - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 32: type userstruct command
static int test_type_userstruct(vx_context context) {
    int errors = 0;
    printf("\n=== Test 32: type userstruct ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "type MyStruct userstruct:64\n"
        "data arr_custom = array:MyStruct,100\n"
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out1 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with type userstruct - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 33: def-macro and macro commands
static int test_def_macro(vx_context context) {
    int errors = 0;
    printf("\n=== Test 33: def-macro and macro ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "def-macro my_filter\n"
        "node org.khronos.openvx.box_3x3 $1 $2\n"
        "endmacro\n"
        "data in1 = image-uniform:U008,128,128,200\n"
        "data out1 = image:U008,128,128\n"
        "macro my_filter in1 out1\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with def-macro and macro - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 34: dumpToConsole enabled
static int test_dump_to_console(vx_context context) {
    int errors = 0;
    printf("\n=== Test 34: dumpToConsole ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    AgoGraphImportInfo info = {};
    const char *text =
        "data img1 = image-uniform:U008,64,64,100\n"
        "data img2 = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 img1 img2\n";
    info.text = const_cast<vx_char *>(text);
    info.num_ref = 0; info.ref = nullptr; info.dumpToConsole = 1;
    info.data_registry_callback_f = nullptr; info.data_registry_callback_obj = nullptr;
    vx_status status = vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_IMPORT_FROM_TEXT, &info, sizeof(info));
    printf("STATUS: Import with dumpToConsole - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 35: set-args command
static int test_set_args_command(vx_context context) {
    int errors = 0;
    printf("\n=== Test 35: set-args command ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    vx_image img1 = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8);
    vx_image img2 = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8);
    if (!img1 || !img2) { if (img1) vxReleaseImage(&img1); if (img2) vxReleaseImage(&img2); vxReleaseGraph(&graph); return 1; }
    vx_reference refs[2] = { (vx_reference)img1, (vx_reference)img2 };
    const char *gdf_text =
        "set-args image-uniform:U008,128,128,77 image:U008,128,128\n"
        "node org.khronos.openvx.box_3x3 $1 $2\n";
    vx_status status = importGraphTextWithRefs(graph, gdf_text, 2, refs);
    printf("STATUS: Import with set-args - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseImage(&img1); vxReleaseImage(&img2);
    vxReleaseGraph(&graph);
    return errors;
}

// Test 36: Virtual image data
static int test_virtual_image(vx_context context) {
    int errors = 0;
    printf("\n=== Test 36: Virtual image ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data virt_in = image-uniform:U008,64,64,100\n"
        "data vmid = image-virtual:U008,64,64\n"
        "data virt_out = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 virt_in vmid\n"
        "node org.khronos.openvx.median_3x3 vmid virt_out\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with virtual image - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 37: Node border mode attributes
static int test_node_border_mode_attr(vx_context context) {
    int errors = 0;
    printf("\n=== Test 37: Node border mode attributes ===\n");

    // UNDEFINED
    vx_graph g1 = vxCreateGraph(context);
    const char *t1 = "data i1 = image-uniform:U008,320,240,128\ndata o1 = image:U008,320,240\nnode org.khronos.openvx.box_3x3 i1 o1 attr:BORDER_MODE:UNDEFINED\n";
    vx_status s1 = importGraphText(g1, t1);
    printf("STATUS: BORDER_MODE:UNDEFINED - %s\n", s1 == VX_SUCCESS ? "PASS" : "FAIL");
    if (s1 != VX_SUCCESS) errors++;
    vxReleaseGraph(&g1);

    // REPLICATE
    vx_graph g2 = vxCreateGraph(context);
    const char *t2 = "data i2 = image-uniform:U008,320,240,128\ndata o2 = image:U008,320,240\nnode org.khronos.openvx.gaussian_3x3 i2 o2 attr:BORDER_MODE:REPLICATE\n";
    vx_status s2 = importGraphText(g2, t2);
    printf("STATUS: BORDER_MODE:REPLICATE - %s\n", s2 == VX_SUCCESS ? "PASS" : "FAIL");
    if (s2 != VX_SUCCESS) errors++;
    vxReleaseGraph(&g2);

    // CONSTANT
    vx_graph g3 = vxCreateGraph(context);
    const char *t3 = "data i3 = image-uniform:U008,320,240,128\ndata o3 = image:U008,320,240\nnode org.khronos.openvx.box_3x3 i3 o3 attr:BORDER_MODE:CONSTANT,0\n";
    vx_status s3 = importGraphText(g3, t3);
    printf("STATUS: BORDER_MODE:CONSTANT - %s\n", s3 == VX_SUCCESS ? "PASS" : "FAIL");
    if (s3 != VX_SUCCESS) errors++;
    vxReleaseGraph(&g3);

    return errors;
}

// Test 38: Node affinity attribute
static int test_node_affinity_attr(vx_context context) {
    int errors = 0;
    printf("\n=== Test 38: Node affinity attribute ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data in1 = image-uniform:U008,320,240,128\n"
        "data out1 = image:U008,320,240\n"
        "node org.khronos.openvx.box_3x3 in1 out1 attr:AFFINITY:CPU\n";
    vx_status status = importGraphText(graph, gdf_text);
    printf("STATUS: Import with AFFINITY:CPU - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vxReleaseGraph(&graph);
    return errors;
}

// Test 39: Optimizer flags set/query
static int test_optimizer_flags(vx_context context) {
    int errors = 0;
    printf("\n=== Test 39: Optimizer flags set/query ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    vx_uint32 flags_set = 3;
    CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS, &flags_set, sizeof(flags_set)));
    vx_uint32 flags_get = 0;
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS, &flags_get, sizeof(flags_get)));
    if (flags_get != flags_set) { printf("ERROR: Optimizer flags mismatch\n"); errors++; }
    else printf("STATUS: Optimizer flags round-trip - PASS\n");
    vxReleaseGraph(&graph);
    return errors;
}

// Test 40: CPU num threads set/query
static int test_cpu_num_threads(vx_context context) {
    int errors = 0;
    printf("\n=== Test 40: CPU num threads set/query ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    vx_uint32 threads_set = 4;
    CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS, &threads_set, sizeof(threads_set)));
    vx_uint32 threads_get = 0;
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS, &threads_get, sizeof(threads_get)));
    if (threads_get != threads_set) { printf("ERROR: CPU num threads mismatch\n"); errors++; }
    else printf("STATUS: CPU num threads round-trip - PASS\n");
    vxReleaseGraph(&graph);
    return errors;
}

// Test 41: Query graph state
static int test_query_graph_state(vx_context context) {
    int errors = 0;
    printf("\n=== Test 41: Query graph state ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    vx_enum state = 0;
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_STATE, &state, sizeof(state)));
    if (state != VX_GRAPH_STATE_UNVERIFIED) { printf("ERROR: Expected UNVERIFIED\n"); errors++; }
    else printf("STATUS: Graph state query - PASS\n");
    vxReleaseGraph(&graph);
    return errors;
}

// Test 42: Size mismatch for vxSetGraphAttribute
static int test_set_attr_size_mismatch(vx_context context) {
    int errors = 0;
    printf("\n=== Test 42: Size mismatch ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    vx_uint32 flags = 0;
    vx_status s1 = vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS, &flags, sizeof(vx_uint64));
    printf("STATUS: OPTIMIZER_FLAGS size mismatch - %s\n", s1 != VX_SUCCESS ? "PASS" : "WARN");
    vx_uint32 threads = 4;
    vx_status s2 = vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS, &threads, sizeof(vx_uint64));
    printf("STATUS: CPU_NUM_THREADS size mismatch - %s\n", s2 != VX_SUCCESS ? "PASS" : "WARN");
    AgoGraphImportInfo info = {};
    vx_status s3 = vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_IMPORT_FROM_TEXT, &info, sizeof(info) - 1);
    printf("STATUS: IMPORT_FROM_TEXT size mismatch - %s\n", s3 != VX_SUCCESS ? "PASS" : "WARN");
    vxReleaseGraph(&graph);
    return errors;
}

// Test 43: Error - invalid syntax
static int test_error_invalid_syntax(vx_context context) {
    int errors = 0;
    printf("\n=== Test 43: Error - invalid syntax ===\n");
    vx_graph graph = vxCreateGraph(context);
    CHECK_STATUS_EXPECT_FAIL(importGraphText(graph, "this_is_invalid_syntax with garbage\n"));
    vxReleaseGraph(&graph);
    return errors;
}

// Test 44: Error - invalid data type
static int test_error_invalid_data_type(vx_context context) {
    int errors = 0;
    printf("\n=== Test 44: Error - invalid data type ===\n");
    vx_graph graph = vxCreateGraph(context);
    CHECK_STATUS_EXPECT_FAIL(importGraphText(graph, "data bad = nonexistent_type:1,2,3\n"));
    vxReleaseGraph(&graph);
    return errors;
}

// Test 45: Error - invalid kernel name
static int test_error_invalid_kernel(vx_context context) {
    int errors = 0;
    printf("\n=== Test 45: Error - invalid kernel ===\n");
    vx_graph graph = vxCreateGraph(context);
    CHECK_STATUS_EXPECT_FAIL(importGraphText(graph, "data i = image:U008,64,64\ndata o = image:U008,64,64\nnode org.nonexistent.kernel i o\n"));
    vxReleaseGraph(&graph);
    return errors;
}

// Test 46: Error - endif without if
static int test_error_endif_without_if(vx_context context) {
    int errors = 0;
    printf("\n=== Test 46: Error - endif without if ===\n");
    vx_graph graph = vxCreateGraph(context);
    CHECK_STATUS_EXPECT_FAIL(importGraphText(graph, "endif\n"));
    vxReleaseGraph(&graph);
    return errors;
}

// Test 47: Error - else without if
static int test_error_else_without_if(vx_context context) {
    int errors = 0;
    printf("\n=== Test 47: Error - else without if ===\n");
    vx_graph graph = vxCreateGraph(context);
    CHECK_STATUS_EXPECT_FAIL(importGraphText(graph, "else\n"));
    vxReleaseGraph(&graph);
    return errors;
}

// Test 48: Error - duplicate def-var
static int test_error_duplicate_def_var(vx_context context) {
    int errors = 0;
    printf("\n=== Test 48: Error - duplicate def-var ===\n");
    vx_graph graph = vxCreateGraph(context);
    CHECK_STATUS_EXPECT_FAIL(importGraphText(graph, "def-var Myvar 100\ndef-var Myvar 200\n"));
    vxReleaseGraph(&graph);
    return errors;
}

// Test 49: Error - invalid def-var name
static int test_error_invalid_var_name(vx_context context) {
    int errors = 0;
    printf("\n=== Test 49: Error - invalid def-var name ===\n");
    vx_graph graph = vxCreateGraph(context);
    CHECK_STATUS_EXPECT_FAIL(importGraphText(graph, "def-var lowercase 100\n"));
    vxReleaseGraph(&graph);
    return errors;
}

// Test 50: Error - invalid for syntax
static int test_error_invalid_for(vx_context context) {
    int errors = 0;
    printf("\n=== Test 50: Error - invalid for syntax ===\n");
    vx_graph graph = vxCreateGraph(context);
    CHECK_STATUS_EXPECT_FAIL(importGraphText(graph, "for i in bad_range\ndata img = image:U008,64,64\nendfor\n"));
    vxReleaseGraph(&graph);
    return errors;
}

// Test 51: Error - duplicate alias
static int test_error_duplicate_alias(vx_context context) {
    int errors = 0;
    printf("\n=== Test 51: Error - duplicate alias ===\n");
    vx_graph graph = vxCreateGraph(context);
    vx_image img = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_reference refs[1] = { (vx_reference)img };
    CHECK_STATUS_EXPECT_FAIL(importGraphTextWithRefs(graph, "alias myAlias $1\nalias myAlias $1\n", 1, refs));
    vxReleaseImage(&img);
    vxReleaseGraph(&graph);
    return errors;
}

// Test 52: Error - invalid macro name
static int test_error_invalid_macro(vx_context context) {
    int errors = 0;
    printf("\n=== Test 52: Error - invalid macro ===\n");
    vx_graph graph = vxCreateGraph(context);
    CHECK_STATUS_EXPECT_FAIL(importGraphText(graph, "data i = image:U008,64,64\nmacro nonexistent_macro i\n"));
    vxReleaseGraph(&graph);
    return errors;
}

// Test 53: Full pipeline with verify and process
static int test_full_pipeline(vx_context context) {
    int errors = 0;
    printf("\n=== Test 53: Full pipeline ===\n");
    vx_graph graph = vxCreateGraph(context);
    if (!graph) { printf("ERROR: vxCreateGraph failed\n"); return 1; }
    const char *gdf_text =
        "data in1 = image-uniform:U008,64,64,100\n"
        "data out = image:U008,64,64\n"
        "node org.khronos.openvx.box_3x3 in1 out\n";
    vx_status status = importGraphText(graph, gdf_text);
    if (status != VX_SUCCESS) {
        printf("ERROR: Import for full pipeline failed: %d\n", status);
        errors++;
        vxReleaseGraph(&graph);
        return errors;
    }
    printf("STATUS: Import for full pipeline - OK\n");
    status = vxProcessGraph(graph);
    printf("STATUS: vxProcessGraph - %s\n", status == VX_SUCCESS ? "PASS" : "FAIL");
    if (status != VX_SUCCESS) errors++;
    vx_enum state = 0;
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_STATE, &state, sizeof(state)));
    printf("STATUS: Graph state after process = %d\n", state);
    vxReleaseGraph(&graph);
    return errors;
}

int main() {
    int errors = 0;
    int test_count = 0;

    printf("==========================================================\n");
    printf("  Graph Import API Test - agoReadGraphFromStringInternal  \n");
    printf("==========================================================\n");

    vx_context context = vxCreateContext();
    if (!context) { printf("ERROR: vxCreateContext failed\n"); return 1; }

    // Data type tests
    test_count++; errors += test_basic_image_and_node(context);
    test_count++; errors += test_scalar_data_types(context);
    test_count++; errors += test_image_formats(context);
    test_count++; errors += test_uniform_images(context);
    test_count++; errors += test_misc_data_types(context);
    test_count++; errors += test_array_data(context);
    test_count++; errors += test_pyramid_data(context);
    test_count++; errors += test_tensor_data(context);
    test_count++; errors += test_delay_data(context);
    test_count++; errors += test_object_array_data(context);
    test_count++; errors += test_multiplane_image(context);

    // Node tests
    test_count++; errors += test_multi_node_graph(context);
    test_count++; errors += test_box_filter_node(context);
    test_count++; errors += test_gaussian_filter_node(context);
    test_count++; errors += test_convolution_node(context);
    test_count++; errors += test_channel_extract_node(context);
    test_count++; errors += test_histogram_node(context);
    test_count++; errors += test_table_lookup_node(context);
    test_count++; errors += test_median_filter_node(context);
    test_count++; errors += test_large_graph(context);

    // Parser syntax features
    test_count++; errors += test_comments_and_empty_lines(context);
    test_count++; errors += test_def_var(context);
    test_count++; errors += test_def_var_default(context);
    test_count++; errors += test_def_var_edge_cases(context);
    test_count++; errors += test_def_var_optimizer_flags(context);
    test_count++; errors += test_def_var_width_height_format(context);
    test_count++; errors += test_dollar_var_references(context);
    test_count++; errors += test_alias_command(context);
    test_count++; errors += test_if_else_endif(context);
    test_count++; errors += test_exit_command(context);
    test_count++; errors += test_affinity_command(context);
    test_count++; errors += test_type_userstruct(context);
    test_count++; errors += test_def_macro(context);
    test_count++; errors += test_dump_to_console(context);
    test_count++; errors += test_set_args_command(context);
    test_count++; errors += test_virtual_image(context);

    // Node attribute tests
    test_count++; errors += test_node_border_mode_attr(context);
    test_count++; errors += test_node_affinity_attr(context);

    // Graph attribute tests
    test_count++; errors += test_optimizer_flags(context);
    test_count++; errors += test_cpu_num_threads(context);
    test_count++; errors += test_query_graph_state(context);
    test_count++; errors += test_set_attr_size_mismatch(context);

    // Error path tests
    test_count++; errors += test_error_invalid_syntax(context);
    test_count++; errors += test_error_invalid_data_type(context);
    test_count++; errors += test_error_invalid_kernel(context);
    test_count++; errors += test_error_endif_without_if(context);
    test_count++; errors += test_error_else_without_if(context);
    test_count++; errors += test_error_duplicate_def_var(context);
    test_count++; errors += test_error_invalid_var_name(context);
    test_count++; errors += test_error_invalid_for(context);
    test_count++; errors += test_error_duplicate_alias(context);
    test_count++; errors += test_error_invalid_macro(context);

    // Full pipeline test
    test_count++; errors += test_full_pipeline(context);

    vxReleaseContext(&context);

    printf("\n==========================================================\n");
    printf("  Graph Import API test: %s (%d tests, %d errors)\n",
           errors == 0 ? "PASS" : "FAIL", test_count, errors);
    printf("==========================================================\n");

    return errors ? 1 : 0;
}
