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

// Coverage boost test - exercises many small uncovered API paths
// across context, graph, node, kernel, scalar, image, tensor,
// threshold, and image-from-channel APIs.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_khr_nn.h>
#include "vx_ext_amd.h"

#define CHECK_STATUS(call) do { \
    vx_status s = (call); \
    if (s != VX_SUCCESS) { \
        printf("  ERROR: %s returned %d at %s:%d\n", #call, s, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

#define CHECK_STATUS_LENIENT(call) do { \
    vx_status s = (call); \
    if (s == VX_SUCCESS) { \
        printf("  OK\n"); \
    } else if (s == VX_ERROR_NOT_SUPPORTED) { \
        printf("  INFO: %s returned VX_ERROR_NOT_SUPPORTED (expected on some configs)\n", #call); \
    } else { \
        printf("  ERROR: %s returned %d at %s:%d\n", #call, s, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Test 1: Context attribute queries
// ---------------------------------------------------------------------------
static int test_context_attributes(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 1: Context attribute queries ===\n");

    // Query VX_CONTEXT_UNIQUE_KERNELS to get count
    vx_uint32 num_kernels = 0;
    printf("  vxQueryContext(VX_CONTEXT_UNIQUE_KERNELS)...\n");
    CHECK_STATUS(vxQueryContext(ctx, VX_CONTEXT_UNIQUE_KERNELS, &num_kernels, sizeof(num_kernels)));
    printf("  Unique kernels: %u\n", num_kernels);

    // Query VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNEL_TABLE using the deprecated compat name
    if (num_kernels > 0) {
        vx_size table_size = num_kernels * sizeof(vx_kernel_info_t);
        vx_kernel_info_t *table = (vx_kernel_info_t *)calloc(num_kernels, sizeof(vx_kernel_info_t));
        if (table) {
            printf("  vxQueryContext(VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNEL_TABLE, buf, %zu)...\n", table_size);
            CHECK_STATUS(vxQueryContext(ctx, VX_CONTEXT_ATTRIBUTE_UNIQUE_KERNEL_TABLE, table, table_size));
            printf("  First kernel: enum=%d name=%s\n", table[0].enumeration, table[0].name);
            free(table);
        }
    }

    // Query VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY
    AgoTargetAffinityInfo affinity = {};
    printf("  vxQueryContext(VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY)...\n");
    CHECK_STATUS(vxQueryContext(ctx, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity)));
    printf("  Context affinity device_type: 0x%x\n", affinity.device_type);

    printf("Test 1: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 2: Graph attribute queries and sets
// ---------------------------------------------------------------------------
static int test_graph_attributes(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 2: Graph attribute queries and sets ===\n");

    // Create a graph with a simple Not node so we can verify/process
    vx_graph graph = vxCreateGraph(ctx);
    if (!graph) { printf("  ERROR: vxCreateGraph failed\n"); return 1; }

    vx_image in_img  = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
    vx_image out_img = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
    vx_node  node    = vxNotNode(graph, in_img, out_img);

    if (!node) {
        printf("  ERROR: vxNotNode failed\n");
        vxReleaseImage(&in_img);
        vxReleaseImage(&out_img);
        vxReleaseGraph(&graph);
        return 1;
    }

    // Verify the graph
    CHECK_STATUS(vxVerifyGraph(graph));

    // Query VX_GRAPH_ATTRIBUTE_STATUS after verify
    vx_status graph_status = VX_FAILURE;
    printf("  vxQueryGraph(VX_GRAPH_ATTRIBUTE_STATUS)...\n");
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_STATUS, &graph_status, sizeof(graph_status)));
    printf("  Graph status after verify: %d\n", graph_status);

    // Process the graph
    CHECK_STATUS(vxProcessGraph(graph));

    // Query graph status again after processing
    printf("  vxQueryGraph(VX_GRAPH_ATTRIBUTE_STATUS) after process...\n");
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_STATUS, &graph_status, sizeof(graph_status)));
    printf("  Graph status after process: %d\n", graph_status);

    // Query VX_GRAPH_ATTRIBUTE_AMD_AFFINITY
    AgoTargetAffinityInfo affinity = {};
    printf("  vxQueryGraph(VX_GRAPH_ATTRIBUTE_AMD_AFFINITY)...\n");
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity)));
    printf("  Graph affinity device_type: 0x%x\n", affinity.device_type);

    // Set VX_GRAPH_ATTRIBUTE_AMD_AFFINITY
    affinity.device_type = AGO_TARGET_AFFINITY_CPU;
    affinity.device_info = 0;
    affinity.group = 0;
    affinity.reserved = 0;
    printf("  vxSetGraphAttribute(VX_GRAPH_ATTRIBUTE_AMD_AFFINITY)...\n");
    CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity)));

    // Query VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_LAST
    AgoGraphPerfInternalInfo perf_info = {};
    printf("  vxQueryGraph(VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_LAST)...\n");
    CHECK_STATUS_LENIENT(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_LAST, &perf_info, sizeof(perf_info)));
    printf("  Perf internal last: kernel_enqueue=%llu kernel_wait=%llu\n",
           (unsigned long long)perf_info.kernel_enqueue, (unsigned long long)perf_info.kernel_wait);

    // Query VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_AVG
    AgoGraphPerfInternalInfo perf_avg = {};
    printf("  vxQueryGraph(VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_AVG)...\n");
    CHECK_STATUS_LENIENT(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_PERFORMANCE_INTERNAL_AVG, &perf_avg, sizeof(perf_avg)));

    // Query VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS
    vx_uint32 cpu_threads = 0;
    printf("  vxQueryGraph(VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS)...\n");
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS, &cpu_threads, sizeof(cpu_threads)));
    printf("  Graph cpu_num_threads: %u\n", cpu_threads);

    // Set VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS
    cpu_threads = 4;
    printf("  vxSetGraphAttribute(VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS)...\n");
    CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS, &cpu_threads, sizeof(cpu_threads)));

    // Query VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS
    vx_uint32 opt_flags = 0;
    printf("  vxQueryGraph(VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS)...\n");
    CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_OPTIMIZER_FLAGS, &opt_flags, sizeof(opt_flags)));
    printf("  Optimizer flags: 0x%x\n", opt_flags);

    vxReleaseNode(&node);
    vxReleaseImage(&in_img);
    vxReleaseImage(&out_img);
    vxReleaseGraph(&graph);

    printf("Test 2: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 3: Node attribute queries and sets
// ---------------------------------------------------------------------------
static int test_node_attributes(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 3: Node attribute queries and sets ===\n");

    vx_graph graph = vxCreateGraph(ctx);
    vx_image in_img  = vxCreateImage(ctx, 32, 32, VX_DF_IMAGE_U8);
    vx_image out_img = vxCreateImage(ctx, 32, 32, VX_DF_IMAGE_U8);
    vx_node  node    = vxNotNode(graph, in_img, out_img);

    if (!node) {
        printf("  ERROR: vxNotNode failed\n");
        vxReleaseImage(&in_img);
        vxReleaseImage(&out_img);
        vxReleaseGraph(&graph);
        return 1;
    }

    // Query VX_NODE_ATTRIBUTE_AMD_CPU_NUM_THREADS
    vx_uint32 threads = 0;
    printf("  vxQueryNode(VX_NODE_ATTRIBUTE_AMD_CPU_NUM_THREADS)...\n");
    CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_CPU_NUM_THREADS, &threads, sizeof(threads)));
    printf("  Node cpu_num_threads: %u\n", threads);

    // Query VX_NODE_ATTRIBUTE_AMD_AFFINITY
    AgoTargetAffinityInfo node_affinity = {};
    printf("  vxQueryNode(VX_NODE_ATTRIBUTE_AMD_AFFINITY)...\n");
    CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_AFFINITY, &node_affinity, sizeof(node_affinity)));
    printf("  Node affinity device_type: 0x%x\n", node_affinity.device_type);

    // Set VX_NODE_ATTRIBUTE_AMD_AFFINITY
    node_affinity.device_type = AGO_TARGET_AFFINITY_CPU;
    node_affinity.device_info = 0;
    node_affinity.group = 0;
    node_affinity.reserved = 0;
    printf("  vxSetNodeAttribute(VX_NODE_ATTRIBUTE_AMD_AFFINITY)...\n");
    CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_AMD_AFFINITY, &node_affinity, sizeof(node_affinity)));

    vxReleaseNode(&node);
    vxReleaseImage(&in_img);
    vxReleaseImage(&out_img);
    vxReleaseGraph(&graph);

    printf("Test 3: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 4: Kernel attribute queries
// ---------------------------------------------------------------------------
static int test_kernel_attributes(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 4: Kernel attribute queries ===\n");

    // Get a known kernel by enum (NOT is always available)
    vx_kernel kernel = vxGetKernelByEnum(ctx, VX_KERNEL_NOT);
    if (!kernel) {
        printf("  ERROR: vxGetKernelByEnum(VX_KERNEL_NOT) failed\n");
        return 1;
    }

    // Query VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE
    vx_size local_data_size = 0;
    printf("  vxQueryKernel(VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE)...\n");
    CHECK_STATUS(vxQueryKernel(kernel, VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE, &local_data_size, sizeof(local_data_size)));
    printf("  Kernel local data size: %zu\n", local_data_size);

    // Query an invalid/unsupported attribute to hit the default case
    vx_uint32 dummy = 0;
    printf("  vxQueryKernel(invalid attribute 0xDEAD)...\n");
    vx_status s = vxQueryKernel(kernel, (vx_enum)0xDEAD, &dummy, sizeof(dummy));
    if (s == VX_ERROR_NOT_SUPPORTED || s == VX_ERROR_INVALID_PARAMETERS) {
        printf("  OK: returned %d as expected for invalid attribute\n", s);
    } else {
        printf("  UNEXPECTED: returned %d\n", s);
        errors++;
    }

    vxReleaseKernel(&kernel);

    printf("Test 4: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 5: Scalar buffer query and various scalar types
// ---------------------------------------------------------------------------
static int test_scalar_buffer(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 5: Scalar buffer query ===\n");

    // Create FLOAT64 scalar
    vx_float64 f64_val = 3.14159265358979;
    vx_scalar scalar_f64 = vxCreateScalar(ctx, VX_TYPE_FLOAT64, &f64_val);
    if (scalar_f64) {
        vx_uint8 *buf_ptr = NULL;
        printf("  vxQueryScalar(VX_SCALAR_BUFFER) on FLOAT64...\n");
        CHECK_STATUS_LENIENT(vxQueryScalar(scalar_f64, VX_SCALAR_BUFFER, &buf_ptr, sizeof(buf_ptr)));
        if (buf_ptr) {
            printf("  Scalar buffer pointer: %p\n", (void *)buf_ptr);
        } else {
            printf("  Scalar buffer is NULL (scalar may not have allocated buffer)\n");
        }
        vxReleaseScalar(&scalar_f64);
    } else {
        printf("  ERROR: vxCreateScalar FLOAT64 failed\n");
        errors++;
    }

    // Create BOOL scalar
    vx_bool bool_val = vx_true_e;
    vx_scalar scalar_bool = vxCreateScalar(ctx, VX_TYPE_BOOL, &bool_val);
    if (scalar_bool) {
        vx_uint8 *buf_ptr = NULL;
        printf("  vxQueryScalar(VX_SCALAR_BUFFER) on BOOL...\n");
        CHECK_STATUS_LENIENT(vxQueryScalar(scalar_bool, VX_SCALAR_BUFFER, &buf_ptr, sizeof(buf_ptr)));
        vxReleaseScalar(&scalar_bool);
    } else {
        printf("  ERROR: vxCreateScalar BOOL failed\n");
        errors++;
    }

    // Create SIZE scalar
    vx_size size_val = 42;
    vx_scalar scalar_size = vxCreateScalar(ctx, VX_TYPE_SIZE, &size_val);
    if (scalar_size) {
        vx_uint8 *buf_ptr = NULL;
        printf("  vxQueryScalar(VX_SCALAR_BUFFER) on SIZE...\n");
        CHECK_STATUS_LENIENT(vxQueryScalar(scalar_size, VX_SCALAR_BUFFER, &buf_ptr, sizeof(buf_ptr)));
        vxReleaseScalar(&scalar_size);
    } else {
        printf("  ERROR: vxCreateScalar SIZE failed\n");
        errors++;
    }

    printf("Test 5: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 6: Image AMD attribute queries and multi-plane size
// ---------------------------------------------------------------------------
static int test_image_amd_attributes(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 6: Image AMD attribute queries ===\n");

    // Single-plane U8 image: query VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER
    vx_image img_u8 = vxCreateImage(ctx, 128, 128, VX_DF_IMAGE_U8);
    if (img_u8) {
        vx_uint8 *host_buf = NULL;
        printf("  vxQueryImage(VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER) on U8...\n");
        // Note: size check in implementation is sizeof(vx_uint8) which is 1
        CHECK_STATUS_LENIENT(vxQueryImage(img_u8, VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &host_buf, sizeof(vx_uint8)));
        printf("  Host buffer pointer: %p\n", (void *)host_buf);
        vxReleaseImage(&img_u8);
    }

    // Multi-plane NV12 image: query VX_IMAGE_SIZE (exercises multi-plane size computation)
    vx_image img_nv12 = vxCreateImage(ctx, 320, 240, VX_DF_IMAGE_NV12);
    if (img_nv12) {
        vx_size img_size = 0;
        printf("  vxQueryImage(VX_IMAGE_SIZE) on NV12 (multi-plane)...\n");
        CHECK_STATUS(vxQueryImage(img_nv12, VX_IMAGE_SIZE, &img_size, sizeof(img_size)));
        printf("  NV12 image size: %zu bytes\n", img_size);
        vxReleaseImage(&img_nv12);
    } else {
        printf("  ERROR: vxCreateImage NV12 failed\n");
        errors++;
    }

    // Multi-plane IYUV image: query VX_IMAGE_SIZE
    vx_image img_iyuv = vxCreateImage(ctx, 320, 240, VX_DF_IMAGE_IYUV);
    if (img_iyuv) {
        vx_size img_size = 0;
        printf("  vxQueryImage(VX_IMAGE_SIZE) on IYUV (multi-plane)...\n");
        CHECK_STATUS(vxQueryImage(img_iyuv, VX_IMAGE_SIZE, &img_size, sizeof(img_size)));
        printf("  IYUV image size: %zu bytes\n", img_size);
        vxReleaseImage(&img_iyuv);
    } else {
        printf("  ERROR: vxCreateImage IYUV failed\n");
        errors++;
    }

    printf("Test 6: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 7: Tensor AMD attribute queries
// ---------------------------------------------------------------------------
static int test_tensor_amd_attributes(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 7: Tensor AMD attribute queries ===\n");

    vx_size dims[3] = {4, 8, 16};
    vx_tensor tensor = vxCreateTensor(ctx, 3, dims, VX_TYPE_INT16, 0);
    if (!tensor) {
        printf("  ERROR: vxCreateTensor failed\n");
        return 1;
    }

    // Write some data so the buffer gets allocated
    vx_size start[3] = {0, 0, 0};
    vx_size end[3] = {4, 8, 16};
    vx_size strides[3] = {sizeof(vx_int16), 4 * sizeof(vx_int16), 4 * 8 * sizeof(vx_int16)};
    vx_int16 *data = (vx_int16 *)calloc(4 * 8 * 16, sizeof(vx_int16));
    if (data) {
        for (int i = 0; i < 4 * 8 * 16; i++) data[i] = (vx_int16)(i % 100);
        CHECK_STATUS(vxCopyTensorPatch(tensor, 3, start, end, strides, data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        free(data);
    }

    // Query VX_TENSOR_BUFFER_HOST
    vx_uint8 *host_buf = NULL;
    printf("  vxQueryTensor(VX_TENSOR_BUFFER_HOST)...\n");
    CHECK_STATUS_LENIENT(vxQueryTensor(tensor, VX_TENSOR_BUFFER_HOST, &host_buf, sizeof(host_buf)));
    printf("  Tensor host buffer pointer: %p\n", (void *)host_buf);

    // Query VX_TENSOR_MEMORY_TYPE
    vx_enum mem_type = 0;
    printf("  vxQueryTensor(VX_TENSOR_MEMORY_TYPE)...\n");
    CHECK_STATUS_LENIENT(vxQueryTensor(tensor, VX_TENSOR_MEMORY_TYPE, &mem_type, sizeof(mem_type)));
    printf("  Tensor memory type: %d\n", mem_type);

    CHECK_STATUS(vxReleaseTensor(&tensor));

    printf("Test 7: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 8: Threshold creation with various input/output formats
// ---------------------------------------------------------------------------
static int test_threshold_formats(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 8: Threshold creation with various formats ===\n");

    // vxCreateThresholdForImage with different input formats to hit switch cases
    struct {
        vx_df_image input_format;
        vx_df_image output_format;
        vx_enum     thresh_type;
        const char *desc;
    } threshold_tests[] = {
        { VX_DF_IMAGE_U8,   VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_BINARY, "U8->U8 BINARY" },
        { VX_DF_IMAGE_S16,  VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_BINARY, "S16->U8 BINARY" },
        { VX_DF_IMAGE_U16,  VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_RANGE,  "U16->U8 RANGE" },
        { VX_DF_IMAGE_S32,  VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_BINARY, "S32->U8 BINARY" },
        { VX_DF_IMAGE_U32,  VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_RANGE,  "U32->U8 RANGE" },
        { VX_DF_IMAGE_RGB,  VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_BINARY, "RGB->U8 BINARY" },
        { VX_DF_IMAGE_RGBX, VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_BINARY, "RGBX->U8 BINARY" },
        { VX_DF_IMAGE_NV12, VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_BINARY, "NV12->U8 BINARY" },
        { VX_DF_IMAGE_IYUV, VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_RANGE,  "IYUV->U8 RANGE" },
        { VX_DF_IMAGE_U1,   VX_DF_IMAGE_U8,   VX_THRESHOLD_TYPE_BINARY, "U1->U8 BINARY" },
    };

    for (size_t i = 0; i < sizeof(threshold_tests) / sizeof(threshold_tests[0]); i++) {
        printf("  vxCreateThresholdForImage(%s)...\n", threshold_tests[i].desc);
        vx_threshold thr = vxCreateThresholdForImage(ctx,
            threshold_tests[i].thresh_type,
            threshold_tests[i].input_format,
            threshold_tests[i].output_format);
        if (thr) {
            printf("    Created OK\n");
            vxReleaseThreshold(&thr);
        } else {
            printf("    INFO: returned NULL (format may not be supported)\n");
        }
    }

    // vxCreateVirtualThresholdForImage with different output formats
    // Note: the guard only allows input_format U8|S16 and output_format U8|U1,
    // but the switch covers many output_format cases. Exercise the allowed ones.
    vx_graph graph = vxCreateGraph(ctx);
    if (graph) {
        struct {
            vx_df_image input_format;
            vx_df_image output_format;
            vx_enum     thresh_type;
            const char *desc;
        } virtual_tests[] = {
            { VX_DF_IMAGE_U8,  VX_DF_IMAGE_U8,  VX_THRESHOLD_TYPE_BINARY, "virtual U8->U8 BINARY" },
            { VX_DF_IMAGE_U8,  VX_DF_IMAGE_U1,  VX_THRESHOLD_TYPE_BINARY, "virtual U8->U1 BINARY" },
            { VX_DF_IMAGE_S16, VX_DF_IMAGE_U8,  VX_THRESHOLD_TYPE_RANGE,  "virtual S16->U8 RANGE" },
            { VX_DF_IMAGE_S16, VX_DF_IMAGE_U1,  VX_THRESHOLD_TYPE_RANGE,  "virtual S16->U1 RANGE" },
        };

        for (size_t i = 0; i < sizeof(virtual_tests) / sizeof(virtual_tests[0]); i++) {
            printf("  vxCreateVirtualThresholdForImage(%s)...\n", virtual_tests[i].desc);
            vx_threshold thr = vxCreateVirtualThresholdForImage(graph,
                virtual_tests[i].thresh_type,
                virtual_tests[i].input_format,
                virtual_tests[i].output_format);
            if (thr) {
                printf("    Created OK\n");
                vxReleaseThreshold(&thr);
            } else {
                printf("    INFO: returned NULL\n");
            }
        }
        vxReleaseGraph(&graph);
    }

    printf("Test 8: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 9: Graph export to text
// ---------------------------------------------------------------------------
static int test_graph_export(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 9: Graph export to text ===\n");

    vx_graph graph = vxCreateGraph(ctx);
    if (!graph) { printf("  ERROR: vxCreateGraph failed\n"); return 1; }

    vx_image in_img  = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
    vx_image out_img = vxCreateImage(ctx, 64, 64, VX_DF_IMAGE_U8);
    vx_node  node    = vxNotNode(graph, in_img, out_img);

    if (!node) {
        printf("  ERROR: vxNotNode failed\n");
        vxReleaseImage(&in_img);
        vxReleaseImage(&out_img);
        vxReleaseGraph(&graph);
        return 1;
    }

    CHECK_STATUS(vxVerifyGraph(graph));

    // Export the graph to stdout via VX_GRAPH_ATTRIBUTE_AMD_EXPORT_TO_TEXT
    AgoGraphExportInfo export_info = {};
    strncpy(export_info.fileName, "stdout", sizeof(export_info.fileName) - 1);
    export_info.num_ref = 0;
    export_info.ref = NULL;
    strncpy(export_info.comment, "coverage_boost_test", sizeof(export_info.comment) - 1);

    printf("  vxSetGraphAttribute(VX_GRAPH_ATTRIBUTE_AMD_EXPORT_TO_TEXT) to stdout...\n");
    CHECK_STATUS_LENIENT(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_EXPORT_TO_TEXT, &export_info, sizeof(export_info)));

    vxReleaseNode(&node);
    vxReleaseImage(&in_img);
    vxReleaseImage(&out_img);
    vxReleaseGraph(&graph);

    printf("Test 9: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 10: Image from channel for NV12 and IYUV
// ---------------------------------------------------------------------------
static int test_image_from_channel(vx_context ctx)
{
    int errors = 0;
    printf("\n=== Test 10: Image from channel (NV12/IYUV) ===\n");

    // NV12 image: Y, U, V channels
    vx_image img_nv12 = vxCreateImage(ctx, 320, 240, VX_DF_IMAGE_NV12);
    if (img_nv12) {
        // VX_CHANNEL_Y
        vx_image ch_y = vxCreateImageFromChannel(img_nv12, VX_CHANNEL_Y);
        if (ch_y) {
            vx_uint32 w = 0, h = 0;
            vxQueryImage(ch_y, VX_IMAGE_WIDTH, &w, sizeof(w));
            vxQueryImage(ch_y, VX_IMAGE_HEIGHT, &h, sizeof(h));
            printf("  NV12 channel Y: %ux%u\n", w, h);
            vxReleaseImage(&ch_y);
        } else {
            printf("  ERROR: vxCreateImageFromChannel(NV12, Y) returned NULL\n");
            errors++;
        }

        // VX_CHANNEL_U (uncovered path for NV12)
        vx_image ch_u = vxCreateImageFromChannel(img_nv12, VX_CHANNEL_U);
        if (ch_u) {
            vx_uint32 w = 0, h = 0;
            vxQueryImage(ch_u, VX_IMAGE_WIDTH, &w, sizeof(w));
            vxQueryImage(ch_u, VX_IMAGE_HEIGHT, &h, sizeof(h));
            printf("  NV12 channel U: %ux%u\n", w, h);
            vxReleaseImage(&ch_u);
        } else {
            printf("  ERROR: vxCreateImageFromChannel(NV12, U) returned NULL\n");
            errors++;
        }

        // VX_CHANNEL_V (uncovered path for NV12 - maps to same child[1])
        vx_image ch_v = vxCreateImageFromChannel(img_nv12, VX_CHANNEL_V);
        if (ch_v) {
            vx_uint32 w = 0, h = 0;
            vxQueryImage(ch_v, VX_IMAGE_WIDTH, &w, sizeof(w));
            vxQueryImage(ch_v, VX_IMAGE_HEIGHT, &h, sizeof(h));
            printf("  NV12 channel V: %ux%u\n", w, h);
            vxReleaseImage(&ch_v);
        } else {
            printf("  ERROR: vxCreateImageFromChannel(NV12, V) returned NULL\n");
            errors++;
        }

        vxReleaseImage(&img_nv12);
    } else {
        printf("  ERROR: vxCreateImage NV12 failed\n");
        errors++;
    }

    // IYUV image: Y, U, V channels
    vx_image img_iyuv = vxCreateImage(ctx, 320, 240, VX_DF_IMAGE_IYUV);
    if (img_iyuv) {
        vx_image ch_y = vxCreateImageFromChannel(img_iyuv, VX_CHANNEL_Y);
        if (ch_y) {
            vx_uint32 w = 0, h = 0;
            vxQueryImage(ch_y, VX_IMAGE_WIDTH, &w, sizeof(w));
            vxQueryImage(ch_y, VX_IMAGE_HEIGHT, &h, sizeof(h));
            printf("  IYUV channel Y: %ux%u\n", w, h);
            vxReleaseImage(&ch_y);
        } else {
            printf("  ERROR: vxCreateImageFromChannel(IYUV, Y) returned NULL\n");
            errors++;
        }

        vx_image ch_u = vxCreateImageFromChannel(img_iyuv, VX_CHANNEL_U);
        if (ch_u) {
            vx_uint32 w = 0, h = 0;
            vxQueryImage(ch_u, VX_IMAGE_WIDTH, &w, sizeof(w));
            vxQueryImage(ch_u, VX_IMAGE_HEIGHT, &h, sizeof(h));
            printf("  IYUV channel U: %ux%u\n", w, h);
            vxReleaseImage(&ch_u);
        } else {
            printf("  ERROR: vxCreateImageFromChannel(IYUV, U) returned NULL\n");
            errors++;
        }

        vx_image ch_v = vxCreateImageFromChannel(img_iyuv, VX_CHANNEL_V);
        if (ch_v) {
            vx_uint32 w = 0, h = 0;
            vxQueryImage(ch_v, VX_IMAGE_WIDTH, &w, sizeof(w));
            vxQueryImage(ch_v, VX_IMAGE_HEIGHT, &h, sizeof(h));
            printf("  IYUV channel V: %ux%u\n", w, h);
            vxReleaseImage(&ch_v);
        } else {
            printf("  ERROR: vxCreateImageFromChannel(IYUV, V) returned NULL\n");
            errors++;
        }

        vxReleaseImage(&img_iyuv);
    } else {
        printf("  ERROR: vxCreateImage IYUV failed\n");
        errors++;
    }

    // NV21 image: test U channel extraction
    vx_image img_nv21 = vxCreateImage(ctx, 320, 240, VX_DF_IMAGE_NV21);
    if (img_nv21) {
        vx_image ch_u = vxCreateImageFromChannel(img_nv21, VX_CHANNEL_U);
        if (ch_u) {
            vx_uint32 w = 0, h = 0;
            vxQueryImage(ch_u, VX_IMAGE_WIDTH, &w, sizeof(w));
            vxQueryImage(ch_u, VX_IMAGE_HEIGHT, &h, sizeof(h));
            printf("  NV21 channel U: %ux%u\n", w, h);
            vxReleaseImage(&ch_u);
        } else {
            printf("  ERROR: vxCreateImageFromChannel(NV21, U) returned NULL\n");
            errors++;
        }

        vx_image ch_y = vxCreateImageFromChannel(img_nv21, VX_CHANNEL_Y);
        if (ch_y) {
            vx_uint32 w = 0, h = 0;
            vxQueryImage(ch_y, VX_IMAGE_WIDTH, &w, sizeof(w));
            vxQueryImage(ch_y, VX_IMAGE_HEIGHT, &h, sizeof(h));
            printf("  NV21 channel Y: %ux%u\n", w, h);
            vxReleaseImage(&ch_y);
        } else {
            printf("  ERROR: vxCreateImageFromChannel(NV21, Y) returned NULL\n");
            errors++;
        }

        vxReleaseImage(&img_nv21);
    } else {
        printf("  ERROR: vxCreateImage NV21 failed\n");
        errors++;
    }

    // YUV4 image: test all channel extractions
    vx_image img_yuv4 = vxCreateImage(ctx, 320, 240, VX_DF_IMAGE_YUV4);
    if (img_yuv4) {
        vx_image ch_y = vxCreateImageFromChannel(img_yuv4, VX_CHANNEL_Y);
        vx_image ch_u = vxCreateImageFromChannel(img_yuv4, VX_CHANNEL_U);
        vx_image ch_v = vxCreateImageFromChannel(img_yuv4, VX_CHANNEL_V);

        if (ch_y && ch_u && ch_v) {
            printf("  YUV4 all channels extracted OK\n");
        } else {
            printf("  ERROR: YUV4 channel extraction failed (Y=%p U=%p V=%p)\n",
                   (void *)ch_y, (void *)ch_u, (void *)ch_v);
            errors++;
        }

        if (ch_y) vxReleaseImage(&ch_y);
        if (ch_u) vxReleaseImage(&ch_u);
        if (ch_v) vxReleaseImage(&ch_v);
        vxReleaseImage(&img_yuv4);
    }

    printf("Test 10: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    printf("Coverage Boost API Test\n");
    printf("=======================\n");

    vx_context ctx = vxCreateContext();
    if (!ctx) {
        printf("ERROR: vxCreateContext failed\n");
        return 1;
    }

    int total_errors = 0;
    total_errors += test_context_attributes(ctx);
    total_errors += test_graph_attributes(ctx);
    total_errors += test_node_attributes(ctx);
    total_errors += test_kernel_attributes(ctx);
    total_errors += test_scalar_buffer(ctx);
    total_errors += test_image_amd_attributes(ctx);
    total_errors += test_tensor_amd_attributes(ctx);
    total_errors += test_threshold_formats(ctx);
    total_errors += test_graph_export(ctx);
    total_errors += test_image_from_channel(ctx);

    vxReleaseContext(&ctx);

    printf("\n=======================\n");
    printf("Coverage Boost API Test: %s (%d total errors)\n",
           total_errors == 0 ? "PASS" : "FAIL", total_errors);

    return total_errors ? 1 : 0;
}
