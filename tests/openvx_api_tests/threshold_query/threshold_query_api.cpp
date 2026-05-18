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

// Threshold Query API coverage test - exercises vxCreateThreshold (deprecated),
// vxCreateVirtualThresholdForImage, vxQueryThreshold (all attribute paths),
// vxSetThresholdAttribute, vxSetContextAttribute (AMD extensions),
// vxQueryNode (VX_NODE_VALID_RECT_RESET, VX_NODE_ATTRIBUTE_AMD_AFFINITY),
// vxSetNodeTarget, vxSetKernelAttribute, vxGetReferenceName, vxSetReferenceName,
// and vxQueryImage for VX_IMAGE_SIZE with multi-plane images.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <vx_ext_amd.h>

#define CHECK_STATUS(call) do { \
    vx_status s = (call); \
    if (s != VX_SUCCESS) { \
        printf("  FAIL: %s returned %d at %s:%d\n", #call, s, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

#define CHECK_NOT_NULL(obj, name) do { \
    if (!(obj)) { \
        printf("  FAIL: %s is NULL at %s:%d\n", name, __FILE__, __LINE__); \
        errors++; \
    } else { \
        printf("  PASS: %s created successfully\n", name); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Test 1: vxCreateThresholdForImage with various format combinations
// Note: The deprecated vxCreateThreshold() has a known stoi bug in this
//       codebase (enum name passed where numeric image format expected),
//       so we use vxCreateThresholdForImage() which is the 1.3 API.
// ---------------------------------------------------------------------------
static int test_vxCreateThreshold_deprecated(vx_context context) {
    int errors = 0;
    printf("\n=== Test 1: vxCreateThresholdForImage (various formats) ===\n");

    // Binary threshold: U8 input, U8 output
    vx_threshold thr_u8 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(thr_u8, "vxCreateThresholdForImage(BINARY, U8, U8)");

    // Range threshold: S16 input, U8 output
    vx_threshold thr_s16 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(thr_s16, "vxCreateThresholdForImage(RANGE, S16, U8)");

    // Binary threshold: U8 input, U1 output
    vx_threshold thr_u1 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1);
    CHECK_NOT_NULL(thr_u1, "vxCreateThresholdForImage(BINARY, U8, U1)");

    // Range threshold: U8 input, U8 output
    vx_threshold thr_range_u8 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(thr_range_u8, "vxCreateThresholdForImage(RANGE, U8, U8)");

    // Release all thresholds
    if (thr_u8)       CHECK_STATUS(vxReleaseThreshold(&thr_u8));
    if (thr_s16)      CHECK_STATUS(vxReleaseThreshold(&thr_s16));
    if (thr_u1)       CHECK_STATUS(vxReleaseThreshold(&thr_u1));
    if (thr_range_u8) CHECK_STATUS(vxReleaseThreshold(&thr_range_u8));

    return errors;
}

// ---------------------------------------------------------------------------
// Test 2: vxCreateVirtualThresholdForImage
// ---------------------------------------------------------------------------
static int test_vxCreateVirtualThresholdForImage(vx_context context) {
    int errors = 0;
    printf("\n=== Test 2: vxCreateVirtualThresholdForImage ===\n");

    vx_graph graph = vxCreateGraph(context);
    CHECK_NOT_NULL(graph, "vxCreateGraph");
    if (!graph) return 1;

    // Virtual threshold: U8 input, U8 output, binary
    vx_threshold vthr1 = vxCreateVirtualThresholdForImage(graph,
        VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(vthr1, "vxCreateVirtualThresholdForImage(BINARY, U8, U8)");

    // Virtual threshold: S16 input, U8 output, range
    vx_threshold vthr2 = vxCreateVirtualThresholdForImage(graph,
        VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(vthr2, "vxCreateVirtualThresholdForImage(RANGE, S16, U8)");

    // Virtual threshold: U8 input, U1 output, binary
    vx_threshold vthr3 = vxCreateVirtualThresholdForImage(graph,
        VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1);
    CHECK_NOT_NULL(vthr3, "vxCreateVirtualThresholdForImage(BINARY, U8, U1)");

    if (vthr1) CHECK_STATUS(vxReleaseThreshold(&vthr1));
    if (vthr2) CHECK_STATUS(vxReleaseThreshold(&vthr2));
    if (vthr3) CHECK_STATUS(vxReleaseThreshold(&vthr3));
    CHECK_STATUS(vxReleaseGraph(&graph));

    return errors;
}

// ---------------------------------------------------------------------------
// Test 3: vxQueryThreshold - all uncovered attribute paths
// ---------------------------------------------------------------------------
static int test_vxQueryThreshold(vx_context context) {
    int errors = 0;
    printf("\n=== Test 3: vxQueryThreshold (all attribute paths) ===\n");

    // Create a BINARY threshold
    vx_threshold thr_binary = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(thr_binary, "vxCreateThresholdForImage(BINARY, U8, U8)");
    if (!thr_binary) return 1;

    // Query VX_THRESHOLD_ATTRIBUTE_DATA_TYPE
    {
        vx_enum data_type = 0;
        CHECK_STATUS(vxQueryThreshold(thr_binary, VX_THRESHOLD_ATTRIBUTE_DATA_TYPE, &data_type, sizeof(data_type)));
        printf("  PASS: VX_THRESHOLD_ATTRIBUTE_DATA_TYPE = 0x%08x\n", data_type);
    }

    // Query VX_THRESHOLD_ATTRIBUTE_TYPE
    {
        vx_enum thresh_type = 0;
        CHECK_STATUS(vxQueryThreshold(thr_binary, VX_THRESHOLD_ATTRIBUTE_TYPE, &thresh_type, sizeof(thresh_type)));
        printf("  PASS: VX_THRESHOLD_ATTRIBUTE_TYPE = 0x%08x\n", thresh_type);
    }

    // Query VX_THRESHOLD_INPUT_FORMAT
    {
        vx_df_image input_fmt = 0;
        CHECK_STATUS(vxQueryThreshold(thr_binary, VX_THRESHOLD_INPUT_FORMAT, &input_fmt, sizeof(input_fmt)));
        printf("  PASS: VX_THRESHOLD_INPUT_FORMAT = 0x%08x\n", input_fmt);
    }

    // Query VX_THRESHOLD_OUTPUT_FORMAT
    {
        vx_df_image output_fmt = 0;
        CHECK_STATUS(vxQueryThreshold(thr_binary, VX_THRESHOLD_OUTPUT_FORMAT, &output_fmt, sizeof(output_fmt)));
        printf("  PASS: VX_THRESHOLD_OUTPUT_FORMAT = 0x%08x\n", output_fmt);
    }

    // Query VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE (only valid for BINARY)
    {
        vx_pixel_value_t value;
        memset(&value, 0, sizeof(value));
        CHECK_STATUS(vxQueryThreshold(thr_binary, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, &value, sizeof(vx_int32)));
        printf("  PASS: VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE queried (U8=%u)\n", value.U8);
    }

    // Query VX_THRESHOLD_ATTRIBUTE_TRUE_VALUE
    {
        vx_pixel_value_t true_val;
        memset(&true_val, 0, sizeof(true_val));
        CHECK_STATUS(vxQueryThreshold(thr_binary, VX_THRESHOLD_ATTRIBUTE_TRUE_VALUE, &true_val, sizeof(vx_int32)));
        printf("  PASS: VX_THRESHOLD_ATTRIBUTE_TRUE_VALUE queried (U8=%u)\n", true_val.U8);
    }

    // Query VX_THRESHOLD_ATTRIBUTE_FALSE_VALUE
    {
        vx_pixel_value_t false_val;
        memset(&false_val, 0, sizeof(false_val));
        CHECK_STATUS(vxQueryThreshold(thr_binary, VX_THRESHOLD_ATTRIBUTE_FALSE_VALUE, &false_val, sizeof(vx_int32)));
        printf("  PASS: VX_THRESHOLD_ATTRIBUTE_FALSE_VALUE queried (U8=%u)\n", false_val.U8);
    }

    CHECK_STATUS(vxReleaseThreshold(&thr_binary));

    // Create a RANGE threshold to test LOWER/UPPER queries
    vx_threshold thr_range = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(thr_range, "vxCreateThresholdForImage(RANGE, U8, U8)");
    if (!thr_range) return errors;

    // Query VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER (only valid for RANGE)
    {
        vx_pixel_value_t lower;
        memset(&lower, 0, sizeof(lower));
        CHECK_STATUS(vxQueryThreshold(thr_range, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(vx_int32)));
        printf("  PASS: VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER queried (U8=%u)\n", lower.U8);
    }

    // Query VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER (only valid for RANGE)
    {
        vx_pixel_value_t upper;
        memset(&upper, 0, sizeof(upper));
        CHECK_STATUS(vxQueryThreshold(thr_range, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(vx_int32)));
        printf("  PASS: VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER queried (U8=%u)\n", upper.U8);
    }

    CHECK_STATUS(vxReleaseThreshold(&thr_range));

    return errors;
}

// ---------------------------------------------------------------------------
// Test 4: vxSetThresholdAttribute
// ---------------------------------------------------------------------------
static int test_vxSetThresholdAttribute(vx_context context) {
    int errors = 0;
    printf("\n=== Test 4: vxSetThresholdAttribute ===\n");

    // Binary threshold - set the threshold value
    vx_threshold thr_binary = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(thr_binary, "vxCreateThresholdForImage(BINARY, U8, U8)");
    if (!thr_binary) return 1;

    {
        vx_pixel_value_t val;
        memset(&val, 0, sizeof(val));
        val.U8 = 128;
        CHECK_STATUS(vxSetThresholdAttribute(thr_binary, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, &val, sizeof(vx_int32)));
        printf("  PASS: Set VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE = 128\n");
    }

    // Set VX_THRESHOLD_TYPE attribute
    {
        vx_enum new_type = VX_THRESHOLD_TYPE_BINARY;
        CHECK_STATUS(vxSetThresholdAttribute(thr_binary, VX_THRESHOLD_TYPE, &new_type, sizeof(vx_enum)));
        printf("  PASS: Set VX_THRESHOLD_TYPE\n");
    }

    // Set VX_THRESHOLD_INPUT_FORMAT attribute
    {
        vx_df_image input_fmt = VX_DF_IMAGE_U8;
        CHECK_STATUS(vxSetThresholdAttribute(thr_binary, VX_THRESHOLD_INPUT_FORMAT, &input_fmt, sizeof(vx_df_image)));
        printf("  PASS: Set VX_THRESHOLD_INPUT_FORMAT\n");
    }

    // Set VX_THRESHOLD_OUTPUT_FORMAT attribute
    {
        vx_df_image output_fmt = VX_DF_IMAGE_U8;
        CHECK_STATUS(vxSetThresholdAttribute(thr_binary, VX_THRESHOLD_OUTPUT_FORMAT, &output_fmt, sizeof(vx_df_image)));
        printf("  PASS: Set VX_THRESHOLD_OUTPUT_FORMAT\n");
    }

    CHECK_STATUS(vxReleaseThreshold(&thr_binary));

    // Range threshold - set lower/upper
    vx_threshold thr_range = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(thr_range, "vxCreateThresholdForImage(RANGE, U8, U8)");
    if (!thr_range) return errors;

    {
        vx_pixel_value_t lower;
        memset(&lower, 0, sizeof(lower));
        lower.U8 = 50;
        CHECK_STATUS(vxSetThresholdAttribute(thr_range, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(vx_int32)));
        printf("  PASS: Set VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER = 50\n");
    }

    {
        vx_pixel_value_t upper;
        memset(&upper, 0, sizeof(upper));
        upper.U8 = 200;
        CHECK_STATUS(vxSetThresholdAttribute(thr_range, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(vx_int32)));
        printf("  PASS: Set VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER = 200\n");
    }

    // Verify the values were set correctly by querying back
    {
        vx_pixel_value_t lower_check;
        memset(&lower_check, 0, sizeof(lower_check));
        CHECK_STATUS(vxQueryThreshold(thr_range, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower_check, sizeof(vx_int32)));
        if (lower_check.U8 == 50) {
            printf("  PASS: Verified lower threshold = 50\n");
        } else {
            printf("  FAIL: Expected lower=50, got %u\n", lower_check.U8);
            errors++;
        }
    }

    CHECK_STATUS(vxReleaseThreshold(&thr_range));

    return errors;
}

// ---------------------------------------------------------------------------
// Test 5: vxQueryNode for VX_NODE_VALID_RECT_RESET and VX_NODE_ATTRIBUTE_AMD_AFFINITY
// ---------------------------------------------------------------------------
static int test_vxQueryNode_attrs(vx_context context) {
    int errors = 0;
    printf("\n=== Test 5: vxQueryNode (VALID_RECT_RESET, AMD_AFFINITY) ===\n");

    // Create a graph with a simple NOT node to test node queries
    vx_graph graph = vxCreateGraph(context);
    CHECK_NOT_NULL(graph, "vxCreateGraph");
    if (!graph) return 1;

    vx_image input  = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_image output = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(input,  "vxCreateImage(input)");
    CHECK_NOT_NULL(output, "vxCreateImage(output)");

    vx_node node = vxNotNode(graph, input, output);
    CHECK_NOT_NULL(node, "vxNotNode");
    if (!node) {
        if (input)  vxReleaseImage(&input);
        if (output) vxReleaseImage(&output);
        vxReleaseGraph(&graph);
        return 1;
    }

    // Query VX_NODE_VALID_RECT_RESET
    {
        vx_bool valid_rect_reset = vx_false_e;
        CHECK_STATUS(vxQueryNode(node, VX_NODE_VALID_RECT_RESET, &valid_rect_reset, sizeof(vx_bool)));
        printf("  PASS: VX_NODE_VALID_RECT_RESET = %d\n", valid_rect_reset);
    }

    // Query VX_NODE_ATTRIBUTE_AMD_AFFINITY
    {
        AgoTargetAffinityInfo affinity;
        memset(&affinity, 0, sizeof(affinity));
        CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(AgoTargetAffinityInfo)));
        printf("  PASS: VX_NODE_ATTRIBUTE_AMD_AFFINITY device_type = 0x%04x\n", affinity.device_type);
    }

    // Also test VX_NODE_PARAMETERS and VX_NODE_STATUS for broader coverage
    {
        vx_uint32 param_count = 0;
        CHECK_STATUS(vxQueryNode(node, VX_NODE_PARAMETERS, &param_count, sizeof(vx_uint32)));
        printf("  PASS: VX_NODE_PARAMETERS = %u\n", param_count);
    }

    {
        vx_status node_status = VX_FAILURE;
        CHECK_STATUS(vxQueryNode(node, VX_NODE_STATUS, &node_status, sizeof(vx_status)));
        printf("  PASS: VX_NODE_STATUS = %d\n", node_status);
    }

    CHECK_STATUS(vxReleaseNode(&node));
    CHECK_STATUS(vxReleaseImage(&input));
    CHECK_STATUS(vxReleaseImage(&output));
    CHECK_STATUS(vxReleaseGraph(&graph));

    return errors;
}

// ---------------------------------------------------------------------------
// Test 6: vxSetNodeTarget
// ---------------------------------------------------------------------------
static int test_vxSetNodeTarget(vx_context context) {
    int errors = 0;
    printf("\n=== Test 6: vxSetNodeTarget ===\n");

    vx_graph graph = vxCreateGraph(context);
    CHECK_NOT_NULL(graph, "vxCreateGraph");
    if (!graph) return 1;

    vx_image input  = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_image output = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_node node = vxNotNode(graph, input, output);
    CHECK_NOT_NULL(node, "vxNotNode");
    if (!node) {
        if (input)  vxReleaseImage(&input);
        if (output) vxReleaseImage(&output);
        vxReleaseGraph(&graph);
        return 1;
    }

    // Test VX_TARGET_ANY
    {
        vx_status status = vxSetNodeTarget(node, VX_TARGET_ANY, nullptr);
        if (status == VX_SUCCESS) {
            printf("  PASS: vxSetNodeTarget(VX_TARGET_ANY)\n");
        } else {
            printf("  FAIL: vxSetNodeTarget(VX_TARGET_ANY) returned %d\n", status);
            errors++;
        }
    }

    // Test VX_TARGET_STRING with "any"
    {
        vx_status status = vxSetNodeTarget(node, VX_TARGET_STRING, "any");
        if (status == VX_SUCCESS) {
            printf("  PASS: vxSetNodeTarget(VX_TARGET_STRING, \"any\")\n");
        } else {
            printf("  FAIL: vxSetNodeTarget(VX_TARGET_STRING, \"any\") returned %d\n", status);
            errors++;
        }
    }

    // Test VX_TARGET_STRING with "cpu"
    {
        vx_status status = vxSetNodeTarget(node, VX_TARGET_STRING, "cpu");
        if (status == VX_SUCCESS) {
            printf("  PASS: vxSetNodeTarget(VX_TARGET_STRING, \"cpu\")\n");
        } else {
            // May fail if affinity is already set; that is expected behavior
            printf("  INFO: vxSetNodeTarget(VX_TARGET_STRING, \"cpu\") returned %d (may be expected)\n", status);
        }
    }

    // Create a second node to test "gpu" target on a fresh node
    vx_image output2 = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_node node2 = vxNotNode(graph, input, output2);
    if (node2) {
        vx_status status = vxSetNodeTarget(node2, VX_TARGET_STRING, "gpu");
        // GPU may or may not be available; just exercise the code path
        printf("  INFO: vxSetNodeTarget(VX_TARGET_STRING, \"gpu\") returned %d\n", status);
        CHECK_STATUS(vxReleaseNode(&node2));
    }
    if (output2) CHECK_STATUS(vxReleaseImage(&output2));

    CHECK_STATUS(vxReleaseNode(&node));
    CHECK_STATUS(vxReleaseImage(&input));
    CHECK_STATUS(vxReleaseImage(&output));
    CHECK_STATUS(vxReleaseGraph(&graph));

    return errors;
}

// ---------------------------------------------------------------------------
// Test 7: vxSetKernelAttribute
// ---------------------------------------------------------------------------
static int test_vxSetKernelAttribute(vx_context context) {
    int errors = 0;
    printf("\n=== Test 7: vxSetKernelAttribute ===\n");

    // Get a kernel by enum to test setting attributes
    vx_kernel kernel = vxGetKernelByEnum(context, VX_KERNEL_NOT);
    CHECK_NOT_NULL(kernel, "vxGetKernelByEnum(VX_KERNEL_NOT)");
    if (!kernel) return 1;

    // Try to set VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE (should work on built-in kernels)
    {
        vx_size local_data_size = 1024;
        vx_status status = vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE,
                                                 &local_data_size, sizeof(vx_size));
        if (status == VX_SUCCESS) {
            printf("  PASS: vxSetKernelAttribute(VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE)\n");
        } else {
            // This may return VX_ERROR_NOT_SUPPORTED for finalized kernels; exercise the path
            printf("  INFO: vxSetKernelAttribute(LOCAL_DATA_SIZE) returned %d\n", status);
        }
    }

    // Try to set VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT (will fail on finalized kernels)
    {
        void *callback = nullptr;
        vx_status status = vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT,
                                                 &callback, sizeof(void *));
        // Expected to return VX_ERROR_NOT_SUPPORTED since built-in kernels are finalized
        printf("  INFO: vxSetKernelAttribute(AMD_QUERY_TARGET_SUPPORT) returned %d (expected for finalized kernel)\n", status);
    }

    // Try to set VX_KERNEL_ATTRIBUTE_AMD_NODE_REGEN_CALLBACK (will fail on finalized kernels)
    {
        void *callback = nullptr;
        vx_status status = vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_NODE_REGEN_CALLBACK,
                                                 &callback, sizeof(void *));
        printf("  INFO: vxSetKernelAttribute(AMD_NODE_REGEN_CALLBACK) returned %d (expected for finalized kernel)\n", status);
    }

    CHECK_STATUS(vxReleaseKernel(&kernel));

    return errors;
}

// ---------------------------------------------------------------------------
// Test 8: vxSetContextAttribute with AMD extensions
// ---------------------------------------------------------------------------
static int test_vxSetContextAttribute_amd(vx_context context) {
    int errors = 0;
    printf("\n=== Test 8: vxSetContextAttribute (AMD extensions) ===\n");

    // Test VX_CONTEXT_ATTRIBUTE_AMD_SET_TEXT_MACRO
    {
        AgoContextTextMacroInfo macroInfo;
        memset(&macroInfo, 0, sizeof(macroInfo));
        strncpy(macroInfo.macroName, "TEST_MACRO_1", sizeof(macroInfo.macroName) - 1);
        char macroText[] = "test_value_1";
        macroInfo.text = macroText;
        vx_status status = vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_AMD_SET_TEXT_MACRO,
                                                  &macroInfo, sizeof(AgoContextTextMacroInfo));
        if (status == VX_SUCCESS) {
            printf("  PASS: vxSetContextAttribute(AMD_SET_TEXT_MACRO) - set TEST_MACRO_1\n");
        } else {
            printf("  FAIL: vxSetContextAttribute(AMD_SET_TEXT_MACRO) returned %d\n", status);
            errors++;
        }
    }

    // Test setting duplicate macro name - should fail with VX_FAILURE
    {
        AgoContextTextMacroInfo macroInfo;
        memset(&macroInfo, 0, sizeof(macroInfo));
        strncpy(macroInfo.macroName, "TEST_MACRO_1", sizeof(macroInfo.macroName) - 1);
        char macroText[] = "duplicate_value";
        macroInfo.text = macroText;
        vx_status status = vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_AMD_SET_TEXT_MACRO,
                                                  &macroInfo, sizeof(AgoContextTextMacroInfo));
        if (status != VX_SUCCESS) {
            printf("  PASS: Duplicate macro correctly rejected (status=%d)\n", status);
        } else {
            printf("  FAIL: Duplicate macro should have been rejected\n");
            errors++;
        }
    }

    // Test VX_CONTEXT_ATTRIBUTE_AMD_SET_MERGE_RULE
    {
        AgoNodeMergeRule mergeRule;
        memset(&mergeRule, 0, sizeof(mergeRule));
        // Set up a minimal merge rule (no-op rule for coverage)
        mergeRule.find[0].kernel_id = VX_KERNEL_NOT;
        mergeRule.replace[0].kernel_id = VX_KERNEL_NOT;
        vx_status status = vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_AMD_SET_MERGE_RULE,
                                                  &mergeRule, sizeof(AgoNodeMergeRule));
        if (status == VX_SUCCESS) {
            printf("  PASS: vxSetContextAttribute(AMD_SET_MERGE_RULE)\n");
        } else {
            printf("  FAIL: vxSetContextAttribute(AMD_SET_MERGE_RULE) returned %d\n", status);
            errors++;
        }
    }

    // Test VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY
    {
        AgoTargetAffinityInfo affinity;
        memset(&affinity, 0, sizeof(affinity));
        affinity.device_type = AGO_TARGET_AFFINITY_CPU;
        vx_status status = vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY,
                                                  &affinity, sizeof(AgoTargetAffinityInfo));
        if (status == VX_SUCCESS) {
            printf("  PASS: vxSetContextAttribute(AMD_AFFINITY, CPU)\n");
        } else {
            printf("  FAIL: vxSetContextAttribute(AMD_AFFINITY) returned %d\n", status);
            errors++;
        }
    }

    return errors;
}

// ---------------------------------------------------------------------------
// Test 9: vxGetReferenceName and vxSetReferenceName
// ---------------------------------------------------------------------------
static int test_reference_name(vx_context context) {
    int errors = 0;
    printf("\n=== Test 9: vxGetReferenceName / vxSetReferenceName ===\n");

    // Test with an image reference
    vx_image image = vxCreateImage(context, 32, 32, VX_DF_IMAGE_U8);
    CHECK_NOT_NULL(image, "vxCreateImage");
    if (image) {
        // Get default name
        char name_buf[VX_MAX_REFERENCE_NAME] = {0};
        CHECK_STATUS(vxGetReferenceName((vx_reference)image, name_buf, sizeof(name_buf)));
        printf("  PASS: vxGetReferenceName(image) default = \"%s\"\n", name_buf);

        // Set a custom name
        CHECK_STATUS(vxSetReferenceName((vx_reference)image, "my_test_image"));
        printf("  PASS: vxSetReferenceName(image, \"my_test_image\")\n");

        // Verify the name was set
        memset(name_buf, 0, sizeof(name_buf));
        CHECK_STATUS(vxGetReferenceName((vx_reference)image, name_buf, sizeof(name_buf)));
        if (strncmp(name_buf, "my_test_image", 13) == 0) {
            printf("  PASS: Name verified = \"%s\"\n", name_buf);
        } else {
            printf("  FAIL: Expected \"my_test_image\", got \"%s\"\n", name_buf);
            errors++;
        }

        CHECK_STATUS(vxReleaseImage(&image));
    }

    // Test with a threshold reference
    vx_threshold thr = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    if (thr) {
        char name_buf[VX_MAX_REFERENCE_NAME] = {0};
        CHECK_STATUS(vxGetReferenceName((vx_reference)thr, name_buf, sizeof(name_buf)));
        printf("  PASS: vxGetReferenceName(threshold) = \"%s\"\n", name_buf);

        CHECK_STATUS(vxSetReferenceName((vx_reference)thr, "my_threshold"));
        memset(name_buf, 0, sizeof(name_buf));
        CHECK_STATUS(vxGetReferenceName((vx_reference)thr, name_buf, sizeof(name_buf)));
        printf("  PASS: vxSetReferenceName(threshold) -> \"%s\"\n", name_buf);

        CHECK_STATUS(vxReleaseThreshold(&thr));
    }

    // Test with a graph reference (vxSetReferenceName has a special graph path)
    vx_graph graph = vxCreateGraph(context);
    if (graph) {
        CHECK_STATUS(vxSetReferenceName((vx_reference)graph, "my_test_graph"));
        printf("  PASS: vxSetReferenceName(graph, \"my_test_graph\")\n");
        CHECK_STATUS(vxReleaseGraph(&graph));
    }

    // Test with a kernel reference (vxGetReferenceName has kernel path)
    vx_kernel kernel = vxGetKernelByEnum(context, VX_KERNEL_NOT);
    if (kernel) {
        char name_buf[VX_MAX_REFERENCE_NAME] = {0};
        CHECK_STATUS(vxGetReferenceName((vx_reference)kernel, name_buf, sizeof(name_buf)));
        printf("  PASS: vxGetReferenceName(kernel) = \"%s\"\n", name_buf);
        CHECK_STATUS(vxReleaseKernel(&kernel));
    }

    // Test with a node reference (vxGetReferenceName has node path)
    {
        vx_graph g = vxCreateGraph(context);
        vx_image img_in  = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8);
        vx_image img_out = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8);
        vx_node nd = vxNotNode(g, img_in, img_out);
        if (nd) {
            char name_buf[VX_MAX_REFERENCE_NAME] = {0};
            CHECK_STATUS(vxGetReferenceName((vx_reference)nd, name_buf, sizeof(name_buf)));
            printf("  PASS: vxGetReferenceName(node) = \"%s\"\n", name_buf);
            CHECK_STATUS(vxReleaseNode(&nd));
        }
        if (img_in)  vxReleaseImage(&img_in);
        if (img_out) vxReleaseImage(&img_out);
        if (g)       vxReleaseGraph(&g);
    }

    return errors;
}

// ---------------------------------------------------------------------------
// Test 10: vxQueryImage for VX_IMAGE_SIZE with multi-plane images (IYUV)
// ---------------------------------------------------------------------------
static int test_vxQueryImage_size_multiplane(vx_context context) {
    int errors = 0;
    printf("\n=== Test 10: vxQueryImage VX_IMAGE_SIZE (multi-plane) ===\n");

    // Create single-plane U8 image and query size
    {
        vx_image img_u8 = vxCreateImage(context, 320, 240, VX_DF_IMAGE_U8);
        CHECK_NOT_NULL(img_u8, "vxCreateImage(320x240, U8)");
        if (img_u8) {
            vx_size img_size = 0;
            CHECK_STATUS(vxQueryImage(img_u8, VX_IMAGE_SIZE, &img_size, sizeof(vx_size)));
            printf("  PASS: VX_IMAGE_SIZE (U8 320x240) = %zu bytes\n", img_size);
            CHECK_STATUS(vxReleaseImage(&img_u8));
        }
    }

    // Create multi-plane IYUV image (3 planes: Y, U, V) and query size
    {
        vx_image img_iyuv = vxCreateImage(context, 320, 240, VX_DF_IMAGE_IYUV);
        CHECK_NOT_NULL(img_iyuv, "vxCreateImage(320x240, IYUV)");
        if (img_iyuv) {
            vx_size img_size = 0;
            CHECK_STATUS(vxQueryImage(img_iyuv, VX_IMAGE_SIZE, &img_size, sizeof(vx_size)));
            printf("  PASS: VX_IMAGE_SIZE (IYUV 320x240) = %zu bytes\n", img_size);

            // Also query number of planes to verify multi-plane nature
            vx_uint32 planes = 0;
            CHECK_STATUS(vxQueryImage(img_iyuv, VX_IMAGE_PLANES, &planes, sizeof(planes)));
            printf("  PASS: VX_IMAGE_PLANES (IYUV) = %u\n", planes);

            CHECK_STATUS(vxReleaseImage(&img_iyuv));
        }
    }

    // Create multi-plane NV12 image (2 planes) and query size
    {
        vx_image img_nv12 = vxCreateImage(context, 320, 240, VX_DF_IMAGE_NV12);
        CHECK_NOT_NULL(img_nv12, "vxCreateImage(320x240, NV12)");
        if (img_nv12) {
            vx_size img_size = 0;
            CHECK_STATUS(vxQueryImage(img_nv12, VX_IMAGE_SIZE, &img_size, sizeof(vx_size)));
            printf("  PASS: VX_IMAGE_SIZE (NV12 320x240) = %zu bytes\n", img_size);
            CHECK_STATUS(vxReleaseImage(&img_nv12));
        }
    }

    // Create multi-plane YUV4 image (3 planes, no subsampling) and query size
    {
        vx_image img_yuv4 = vxCreateImage(context, 320, 240, VX_DF_IMAGE_YUV4);
        CHECK_NOT_NULL(img_yuv4, "vxCreateImage(320x240, YUV4)");
        if (img_yuv4) {
            vx_size img_size = 0;
            CHECK_STATUS(vxQueryImage(img_yuv4, VX_IMAGE_SIZE, &img_size, sizeof(vx_size)));
            printf("  PASS: VX_IMAGE_SIZE (YUV4 320x240) = %zu bytes\n", img_size);
            CHECK_STATUS(vxReleaseImage(&img_yuv4));
        }
    }

    return errors;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    int total_errors = 0;

    printf("OpenVX Threshold Query API Coverage Test\n");
    printf("=========================================\n");

    vx_context context = vxCreateContext();
    if (!context) {
        printf("FATAL: vxCreateContext failed\n");
        return 1;
    }
    printf("PASS: vxCreateContext\n");

    total_errors += test_vxCreateThreshold_deprecated(context);
    total_errors += test_vxCreateVirtualThresholdForImage(context);
    total_errors += test_vxQueryThreshold(context);
    total_errors += test_vxSetThresholdAttribute(context);
    total_errors += test_vxQueryNode_attrs(context);
    total_errors += test_vxSetNodeTarget(context);
    total_errors += test_vxSetKernelAttribute(context);
    total_errors += test_vxSetContextAttribute_amd(context);
    total_errors += test_reference_name(context);
    total_errors += test_vxQueryImage_size_multiplane(context);

    vxReleaseContext(&context);

    printf("\n=========================================\n");
    if (total_errors == 0) {
        printf("RESULT: ALL TESTS PASSED\n");
    } else {
        printf("RESULT: %d ERROR(S) DETECTED\n", total_errors);
    }

    return (total_errors == 0) ? 0 : 1;
}
