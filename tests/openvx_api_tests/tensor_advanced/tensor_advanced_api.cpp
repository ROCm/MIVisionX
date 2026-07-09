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

// Tensor Advanced API coverage test - exercises:
//   vxCreateTensorFromHandle, vxSwapTensorHandle, vxAliasTensor,
//   vxIsTensorAliased, vxAddKernel, vxGetModuleHandle, vxSetModuleHandle,
//   vxGetModuleInternalData, vxSetModuleInternalData, vxCopyTensorPatch
//   (strided/sub-region paths)

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <VX/vx.h>
#include <VX/vx_khr_nn.h>
#include <VX/vx_compatibility.h>
#include "vx_ext_amd.h"

#define CHECK_STATUS(call) do { \
    vx_status s = (call); \
    if (s != VX_SUCCESS) { \
        printf("ERROR: %s returned %d at %s:%d\n", #call, s, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

#define CHECK_NULL(ptr, name) do { \
    if (!(ptr)) { \
        printf("ERROR: %s returned NULL at %s:%d\n", name, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

// ------------------------------------------------------------------
// Dummy callbacks for vxAddKernel (old-style kernel registration)
// ------------------------------------------------------------------
static vx_status VX_CALLBACK dummy_kernel_func(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    (void)node; (void)parameters; (void)num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK dummy_input_validate(vx_node node, vx_uint32 index)
{
    (void)node; (void)index;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK dummy_output_validate(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    (void)node; (void)index; (void)meta;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK dummy_initialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    (void)node; (void)parameters; (void)num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK dummy_deinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    (void)node; (void)parameters; (void)num;
    return VX_SUCCESS;
}

// ------------------------------------------------------------------
// Test 1: vxCreateTensorFromHandle (HOST path)
// ------------------------------------------------------------------
static int test_CreateTensorFromHandle(vx_context context)
{
    int errors = 0;
    printf("\n=== Test: vxCreateTensorFromHandle (HOST) ===\n");

    const vx_size num_dims = 2;
    vx_size dims[2] = {4, 8};
    vx_size total = 4 * 8;

    // Allocate host buffer and fill with known data
    float *host_buf = (float *)calloc(total, sizeof(float));
    if (!host_buf) {
        printf("ERROR: failed to allocate host buffer\n");
        return 1;
    }
    for (vx_size i = 0; i < total; i++) {
        host_buf[i] = (float)(i * 1.5f);
    }

    // Compute expected strides: stride[0] = sizeof(float), stride[1] = dims[0] * sizeof(float)
    vx_size strides[2] = {sizeof(float), dims[0] * sizeof(float)};

    vx_tensor tensor = vxCreateTensorFromHandle(context, num_dims, dims,
                                                 VX_TYPE_FLOAT32, 0,
                                                 strides, host_buf,
                                                 VX_MEMORY_TYPE_HOST);
    CHECK_NULL(tensor, "vxCreateTensorFromHandle");
    if (!tensor) {
        free(host_buf);
        return errors;
    }
    printf("STATUS: vxCreateTensorFromHandle - OK\n");

    // Query the memory type attribute to confirm it is HOST
    vx_enum mem_type = VX_MEMORY_TYPE_NONE;
    CHECK_STATUS(vxQueryTensor(tensor, VX_TENSOR_MEMORY_TYPE, &mem_type, sizeof(mem_type)));
    if (mem_type == VX_MEMORY_TYPE_HOST) {
        printf("STATUS: VX_TENSOR_MEMORY_TYPE = VX_MEMORY_TYPE_HOST - OK\n");
    } else {
        printf("ERROR: VX_TENSOR_MEMORY_TYPE expected HOST, got %d\n", mem_type);
        errors++;
    }

    // Read data back via vxCopyTensorPatch and verify
    vx_size start[2] = {0, 0};
    vx_size end[2] = {4, 8};
    float *readback = (float *)calloc(total, sizeof(float));
    if (readback) {
        CHECK_STATUS(vxCopyTensorPatch(tensor, num_dims, start, end, strides,
                                        readback, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        int mismatch = 0;
        for (vx_size i = 0; i < total; i++) {
            if (readback[i] != host_buf[i]) mismatch++;
        }
        printf("STATUS: Data verification after CreateFromHandle: %s (%d mismatches)\n",
               mismatch == 0 ? "PASS" : "FAIL", mismatch);
        if (mismatch) errors++;
        free(readback);
    }

    CHECK_STATUS(vxReleaseTensor(&tensor));
    free(host_buf);
    return errors;
}

// ------------------------------------------------------------------
// Test 2: vxSwapTensorHandle
// ------------------------------------------------------------------
static int test_SwapTensorHandle(vx_context context)
{
    int errors = 0;
    printf("\n=== Test: vxSwapTensorHandle ===\n");

    const vx_size num_dims = 2;
    vx_size dims[2] = {3, 5};
    vx_size total = 3 * 5;
    vx_size strides[2] = {sizeof(float), dims[0] * sizeof(float)};

    // First buffer - fill with 1.0
    float *buf1 = (float *)calloc(total, sizeof(float));
    for (vx_size i = 0; i < total; i++) buf1[i] = 1.0f;

    // Second buffer - fill with 2.0
    float *buf2 = (float *)calloc(total, sizeof(float));
    for (vx_size i = 0; i < total; i++) buf2[i] = 2.0f;

    vx_tensor tensor = vxCreateTensorFromHandle(context, num_dims, dims,
                                                 VX_TYPE_FLOAT32, 0,
                                                 strides, buf1,
                                                 VX_MEMORY_TYPE_HOST);
    CHECK_NULL(tensor, "vxCreateTensorFromHandle for swap test");
    if (!tensor) {
        free(buf1);
        free(buf2);
        return errors;
    }

    // Swap handle to buf2, retrieve prev pointer
    void *prev_ptr = NULL;
    CHECK_STATUS(vxSwapTensorHandle(tensor, buf2, &prev_ptr));
    printf("STATUS: vxSwapTensorHandle - OK\n");

    // Verify the previous pointer matches buf1
    if (prev_ptr == buf1) {
        printf("STATUS: Previous handle matches original buffer - PASS\n");
    } else {
        printf("ERROR: Previous handle mismatch: expected %p, got %p\n",
               (void *)buf1, prev_ptr);
        errors++;
    }

    // Read data from tensor - should now see buf2 data (2.0)
    vx_size start[2] = {0, 0};
    vx_size end[2] = {3, 5};
    float *readback = (float *)calloc(total, sizeof(float));
    if (readback) {
        CHECK_STATUS(vxCopyTensorPatch(tensor, num_dims, start, end, strides,
                                        readback, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        int mismatch = 0;
        for (vx_size i = 0; i < total; i++) {
            if (readback[i] != 2.0f) mismatch++;
        }
        printf("STATUS: Data verification after swap: %s (%d mismatches)\n",
               mismatch == 0 ? "PASS" : "FAIL", mismatch);
        if (mismatch) errors++;
        free(readback);
    }

    // Also test swap with prev_ptr = NULL (should not crash)
    CHECK_STATUS(vxSwapTensorHandle(tensor, buf1, NULL));
    printf("STATUS: vxSwapTensorHandle with NULL prev_ptr - OK\n");

    CHECK_STATUS(vxReleaseTensor(&tensor));
    free(buf1);
    free(buf2);
    return errors;
}

// ------------------------------------------------------------------
// Test 3 & 4: vxAliasTensor and vxIsTensorAliased
// These require virtual tensors (isVirtual must be true)
// ------------------------------------------------------------------
static int test_AliasTensor(vx_context context)
{
    int errors = 0;
    printf("\n=== Test: vxAliasTensor / vxIsTensorAliased ===\n");

    // Create a graph to hold virtual tensors
    vx_graph graph = vxCreateGraph(context);
    CHECK_NULL(graph, "vxCreateGraph");
    if (!graph) return 1;

    // Create two virtual tensors
    vx_size dims_master[2] = {100, 200};
    vx_size dims_alias[2] = {50, 100};
    vx_tensor master = vxCreateVirtualTensor(graph, 2, dims_master, VX_TYPE_FLOAT32, 0);
    vx_tensor alias  = vxCreateVirtualTensor(graph, 2, dims_alias,  VX_TYPE_FLOAT32, 0);
    CHECK_NULL(master, "vxCreateVirtualTensor (master)");
    CHECK_NULL(alias,  "vxCreateVirtualTensor (alias)");

    if (master && alias) {
        // Alias 'alias' to 'master' at offset 0
        vx_size offset = 0;
        vx_status alias_status = vxAliasTensor(master, offset, alias);
        if (alias_status == VX_SUCCESS) {
            printf("STATUS: vxAliasTensor - OK\n");

            // Check that they are aliased
            vx_bool is_aliased = vxIsTensorAliased(master, offset, alias);
            if (is_aliased == vx_true_e) {
                printf("STATUS: vxIsTensorAliased (correct offset) = true - PASS\n");
            } else {
                printf("ERROR: vxIsTensorAliased expected true, got false\n");
                errors++;
            }

            // Check with wrong offset - should return false
            vx_bool is_aliased_wrong = vxIsTensorAliased(master, 999, alias);
            if (is_aliased_wrong == vx_false_e) {
                printf("STATUS: vxIsTensorAliased (wrong offset) = false - PASS\n");
            } else {
                printf("ERROR: vxIsTensorAliased with wrong offset expected false, got true\n");
                errors++;
            }
        } else {
            printf("STATUS: vxAliasTensor returned %d (requires virtual tensors)\n", alias_status);
        }
    }

    // Also test with a non-zero offset
    vx_tensor alias2 = vxCreateVirtualTensor(graph, 2, dims_alias, VX_TYPE_FLOAT32, 0);
    if (master && alias2) {
        vx_size offset2 = 128;
        vx_status alias_status2 = vxAliasTensor(master, offset2, alias2);
        if (alias_status2 == VX_SUCCESS) {
            printf("STATUS: vxAliasTensor (offset=%zu) - OK\n", offset2);
            vx_bool is_aliased2 = vxIsTensorAliased(master, offset2, alias2);
            if (is_aliased2 == vx_true_e) {
                printf("STATUS: vxIsTensorAliased (offset=%zu) = true - PASS\n", offset2);
            } else {
                printf("ERROR: vxIsTensorAliased with offset %zu expected true\n", offset2);
                errors++;
            }
        }
        vxReleaseTensor(&alias2);
    }

    if (alias) vxReleaseTensor(&alias);
    if (master) vxReleaseTensor(&master);
    vxReleaseGraph(&graph);
    return errors;
}

// ------------------------------------------------------------------
// Test 5: vxAddKernel (old-style with separate input/output validators)
// ------------------------------------------------------------------
static int test_AddKernel(vx_context context)
{
    int errors = 0;
    printf("\n=== Test: vxAddKernel (old-style) ===\n");

    // Use a vendor-specific enumeration to avoid clashing with built-in kernels
    vx_enum kernel_enum = VX_KERNEL_BASE(VX_ID_USER, 0) + 100;
    const char *kernel_name = "org.test.addkernel_old_style";

    vx_kernel kernel = vxAddKernel(context,
                                    kernel_name,
                                    kernel_enum,
                                    dummy_kernel_func,
                                    2,  // numParams
                                    dummy_input_validate,
                                    dummy_output_validate,
                                    dummy_initialize,
                                    dummy_deinitialize);
    CHECK_NULL(kernel, "vxAddKernel");
    if (kernel) {
        printf("STATUS: vxAddKernel (old-style) - OK\n");

        // Query kernel name to verify
        char name_buf[VX_MAX_KERNEL_NAME] = {0};
        vx_status qs = vxQueryKernel(kernel, VX_KERNEL_NAME, name_buf, sizeof(name_buf));
        if (qs == VX_SUCCESS && strcmp(name_buf, kernel_name) == 0) {
            printf("STATUS: Kernel name verified: %s - PASS\n", name_buf);
        } else {
            printf("ERROR: Kernel name mismatch or query failed (status=%d, name='%s')\n", qs, name_buf);
            errors++;
        }

        // Query kernel enumeration
        vx_enum queried_enum = 0;
        CHECK_STATUS(vxQueryKernel(kernel, VX_KERNEL_ENUM, &queried_enum, sizeof(queried_enum)));
        if (queried_enum == kernel_enum) {
            printf("STATUS: Kernel enum verified: 0x%08x - PASS\n", queried_enum);
        } else {
            printf("ERROR: Kernel enum mismatch: expected 0x%08x, got 0x%08x\n", kernel_enum, queried_enum);
            errors++;
        }

        // Set parameter directions to finalize the kernel
        CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        CHECK_STATUS(vxFinalizeKernel(kernel));
        printf("STATUS: Kernel finalized - OK\n");

        // Test with init and deinit set to NULL
        vx_enum kernel_enum2 = VX_KERNEL_BASE(VX_ID_USER, 0) + 101;
        vx_kernel kernel2 = vxAddKernel(context,
                                         "org.test.addkernel_no_init",
                                         kernel_enum2,
                                         dummy_kernel_func,
                                         1,
                                         dummy_input_validate,
                                         dummy_output_validate,
                                         NULL,   // no init
                                         NULL);  // no deinit
        if (kernel2) {
            printf("STATUS: vxAddKernel (NULL init/deinit) - OK\n");
            CHECK_STATUS(vxAddParameterToKernel(kernel2, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
            CHECK_STATUS(vxFinalizeKernel(kernel2));
            vxReleaseKernel(&kernel2);
        } else {
            printf("STATUS: vxAddKernel (NULL init/deinit) returned NULL\n");
        }

        vxReleaseKernel(&kernel);
    }
    return errors;
}

// ------------------------------------------------------------------
// Test 6 & 7: vxSetModuleHandle / vxGetModuleHandle
// These require a valid node, so we create a simple user kernel + graph + node.
// ------------------------------------------------------------------
static vx_status VX_CALLBACK module_test_kernel_func(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    (void)node; (void)parameters; (void)num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK module_test_validate(vx_node node, const vx_reference *parameters,
                                                    vx_uint32 num, vx_meta_format metas[])
{
    (void)node; (void)parameters; (void)num; (void)metas;
    return VX_SUCCESS;
}

static int test_ModuleHandle(vx_context context)
{
    int errors = 0;
    printf("\n=== Test: vxSetModuleHandle / vxGetModuleHandle ===\n");

    // Register a user kernel so we can create a node
    vx_enum kernel_enum = VX_KERNEL_BASE(VX_ID_USER, 0) + 200;
    vx_kernel kernel = vxAddUserKernel(context,
                                        "org.test.module_handle_kernel",
                                        kernel_enum,
                                        module_test_kernel_func,
                                        1,
                                        module_test_validate,
                                        NULL, NULL);
    CHECK_NULL(kernel, "vxAddUserKernel for module handle test");
    if (!kernel) return 1;

    CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    CHECK_STATUS(vxFinalizeKernel(kernel));

    vx_graph graph = vxCreateGraph(context);
    CHECK_NULL(graph, "vxCreateGraph for module handle test");
    if (!graph) {
        vxReleaseKernel(&kernel);
        return errors + 1;
    }

    vx_node node = vxCreateGenericNode(graph, kernel);
    CHECK_NULL(node, "vxCreateGenericNode for module handle test");
    if (!node) {
        vxReleaseGraph(&graph);
        vxReleaseKernel(&kernel);
        return errors + 1;
    }

    // Set a module handle
    const char *module_name = "test_module";
    int module_data = 42;
    CHECK_STATUS(vxSetModuleHandle(node, module_name, &module_data));
    printf("STATUS: vxSetModuleHandle - OK\n");

    // Get the module handle
    void *retrieved_ptr = NULL;
    CHECK_STATUS(vxGetModuleHandle(node, module_name, &retrieved_ptr));
    if (retrieved_ptr == &module_data) {
        printf("STATUS: vxGetModuleHandle returned correct pointer - PASS\n");
    } else {
        printf("ERROR: vxGetModuleHandle returned %p, expected %p\n",
               retrieved_ptr, (void *)&module_data);
        errors++;
    }

    // Get handle for a module that was never set - should return NULL
    void *unknown_ptr = (void *)0xDEAD;
    CHECK_STATUS(vxGetModuleHandle(node, "nonexistent_module", &unknown_ptr));
    if (unknown_ptr == NULL) {
        printf("STATUS: vxGetModuleHandle for unknown module returned NULL - PASS\n");
    } else {
        printf("STATUS: vxGetModuleHandle for unknown module returned %p (implementation-specific)\n",
               unknown_ptr);
    }

    vxReleaseNode(&node);
    vxReleaseGraph(&graph);
    vxReleaseKernel(&kernel);
    return errors;
}

// ------------------------------------------------------------------
// Test 8 & 9: vxSetModuleInternalData / vxGetModuleInternalData
// These operate on modules loaded via vxLoadKernels. We test the API
// paths -- they may return VX_ERROR_INVALID_REFERENCE if no module
// is loaded, but we still exercise the function entry points.
// ------------------------------------------------------------------
static int test_ModuleInternalData(vx_context context)
{
    int errors = 0;
    printf("\n=== Test: vxSetModuleInternalData / vxGetModuleInternalData ===\n");

    // Use a dummy module name - these functions iterate through loaded modules
    // to find a matching name, so they will exercise the loop even if no match.
    const char *module_name = "test_internal_module";
    int internal_data = 99;
    vx_size data_size = sizeof(internal_data);

    // Set module internal data -- will iterate modules list
    vx_status set_status = vxSetModuleInternalData(context, module_name, &internal_data, data_size);
    printf("STATUS: vxSetModuleInternalData returned %d (expected: iteration over modules)\n", set_status);

    // Get module internal data
    void *retrieved_ptr = NULL;
    vx_size retrieved_size = 0;
    vx_status get_status = vxGetModuleInternalData(context, module_name, &retrieved_ptr, &retrieved_size);
    printf("STATUS: vxGetModuleInternalData returned %d (expected: iteration over modules)\n", get_status);

    // These functions only succeed when a module with matching name has been loaded.
    // The code path is exercised regardless of success/failure.
    printf("STATUS: Module internal data API paths exercised - OK\n");
    return errors;
}

// ------------------------------------------------------------------
// Test 10: vxCopyTensorPatch with different stride combinations
// to hit the non-singleCopy (strided) branches
// ------------------------------------------------------------------
static int test_CopyTensorPatchStrided(vx_context context)
{
    int errors = 0;
    printf("\n=== Test: vxCopyTensorPatch (strided / sub-region paths) ===\n");

    // Create a 4D tensor [2, 3, 4, 2] of FLOAT32
    // This ensures the 4-nested-loop path is exercised in vxCopyTensorPatch
    vx_size dims[4] = {2, 3, 4, 2};
    vx_size total = 2 * 3 * 4 * 2;
    vx_tensor tensor = vxCreateTensor(context, 4, dims, VX_TYPE_FLOAT32, 0);
    CHECK_NULL(tensor, "vxCreateTensor 4D");
    if (!tensor) return 1;

    // Standard (compact) strides for a 4D tensor
    vx_size compact_strides[4] = {
        sizeof(float),
        2 * sizeof(float),
        2 * 3 * sizeof(float),
        2 * 3 * 4 * sizeof(float)
    };

    // Write data with compact strides (singleCopy path)
    float *write_data = (float *)calloc(total, sizeof(float));
    for (vx_size i = 0; i < total; i++) write_data[i] = (float)(i + 1);
    vx_size start_full[4] = {0, 0, 0, 0};
    vx_size end_full[4] = {2, 3, 4, 2};
    CHECK_STATUS(vxCopyTensorPatch(tensor, 4, start_full, end_full, compact_strides,
                                    write_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyTensorPatch WRITE (compact) - OK\n");

    // Read back using compact strides and verify (singleCopy path)
    float *read_compact = (float *)calloc(total, sizeof(float));
    CHECK_STATUS(vxCopyTensorPatch(tensor, 4, start_full, end_full, compact_strides,
                                    read_compact, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    {
        int mismatch = 0;
        for (vx_size i = 0; i < total; i++) {
            if (read_compact[i] != write_data[i]) mismatch++;
        }
        printf("STATUS: Compact read verification: %s (%d mismatches)\n",
               mismatch == 0 ? "PASS" : "FAIL", mismatch);
        if (mismatch) errors++;
    }
    free(read_compact);

    // --- Sub-region read (non-singleCopy path because start != 0) ---
    // Read a sub-region: [0:2, 1:3, 0:4, 0:2]
    // NOTE: vxCopyTensorPatch uses absolute indices for user buffer offsets,
    // so the user buffer must use the same strides and be large enough to hold
    // data at those absolute positions.
    vx_size sub_start[4] = {0, 1, 0, 0};
    vx_size sub_end[4] = {2, 3, 4, 2};
    // Use compact strides (same as tensor) so the user buffer layout matches
    // the absolute index computation in vxCopyTensorPatch
    float *sub_read = (float *)calloc(total, sizeof(float));
    memset(sub_read, 0, total * sizeof(float));
    CHECK_STATUS(vxCopyTensorPatch(tensor, 4, sub_start, sub_end, compact_strides,
                                    sub_read, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyTensorPatch READ (sub-region) - OK\n");
    // Verify sub-region data: only elements within [sub_start, sub_end) should be copied
    {
        int mismatch = 0;
        for (vx_size d3 = sub_start[3]; d3 < sub_end[3]; d3++) {
            for (vx_size d2 = sub_start[2]; d2 < sub_end[2]; d2++) {
                for (vx_size d1 = sub_start[1]; d1 < sub_end[1]; d1++) {
                    for (vx_size d0 = sub_start[0]; d0 < sub_end[0]; d0++) {
                        vx_size idx = d0 + d1 * dims[0] +
                                      d2 * dims[0] * dims[1] +
                                      d3 * dims[0] * dims[1] * dims[2];
                        if (sub_read[idx] != write_data[idx]) mismatch++;
                    }
                }
            }
        }
        printf("STATUS: Sub-region data verification: %s (%d mismatches)\n",
               mismatch == 0 ? "PASS" : "FAIL", mismatch);
        if (mismatch) errors++;
    }
    free(sub_read);

    // --- Strided write (non-singleCopy path due to larger user strides) ---
    // Use padded strides in the user buffer: stride[1] is larger than compact
    vx_size padded_strides[4] = {
        sizeof(float),
        4 * sizeof(float),    // padded from 2*sizeof(float) to 4*sizeof(float)
        4 * 3 * sizeof(float),
        4 * 3 * 4 * sizeof(float)
    };
    vx_size padded_total = 4 * 3 * 4 * 2;  // total elements in padded buffer
    float *padded_write = (float *)calloc(padded_total, sizeof(float));
    // Fill only the valid positions in the padded buffer
    for (vx_size d3 = 0; d3 < dims[3]; d3++) {
        for (vx_size d2 = 0; d2 < dims[2]; d2++) {
            for (vx_size d1 = 0; d1 < dims[1]; d1++) {
                for (vx_size d0 = 0; d0 < dims[0]; d0++) {
                    vx_size padded_idx = d0 + d1 * 4 + d2 * 4 * 3 + d3 * 4 * 3 * 4;
                    padded_write[padded_idx] = (float)(100 + d0 + d1 * 10 + d2 * 100 + d3 * 1000);
                }
            }
        }
    }
    CHECK_STATUS(vxCopyTensorPatch(tensor, 4, start_full, end_full, padded_strides,
                                    padded_write, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyTensorPatch WRITE (padded strides) - OK\n");

    // Read back with compact strides and verify the padded write
    float *verify_read = (float *)calloc(total, sizeof(float));
    CHECK_STATUS(vxCopyTensorPatch(tensor, 4, start_full, end_full, compact_strides,
                                    verify_read, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    {
        int mismatch = 0;
        for (vx_size d3 = 0; d3 < dims[3]; d3++) {
            for (vx_size d2 = 0; d2 < dims[2]; d2++) {
                for (vx_size d1 = 0; d1 < dims[1]; d1++) {
                    for (vx_size d0 = 0; d0 < dims[0]; d0++) {
                        float expected = (float)(100 + d0 + d1 * 10 + d2 * 100 + d3 * 1000);
                        vx_size compact_idx = d0 + d1 * dims[0] + d2 * dims[0] * dims[1] +
                                              d3 * dims[0] * dims[1] * dims[2];
                        if (verify_read[compact_idx] != expected) mismatch++;
                    }
                }
            }
        }
        printf("STATUS: Padded write verification: %s (%d mismatches)\n",
               mismatch == 0 ? "PASS" : "FAIL", mismatch);
        if (mismatch) errors++;
    }
    free(verify_read);

    // --- Strided read with padded user strides (non-singleCopy READ path) ---
    float *padded_read = (float *)calloc(padded_total, sizeof(float));
    memset(padded_read, 0, padded_total * sizeof(float));
    CHECK_STATUS(vxCopyTensorPatch(tensor, 4, start_full, end_full, padded_strides,
                                    padded_read, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    printf("STATUS: vxCopyTensorPatch READ (padded strides) - OK\n");
    {
        int mismatch = 0;
        for (vx_size d3 = 0; d3 < dims[3]; d3++) {
            for (vx_size d2 = 0; d2 < dims[2]; d2++) {
                for (vx_size d1 = 0; d1 < dims[1]; d1++) {
                    for (vx_size d0 = 0; d0 < dims[0]; d0++) {
                        float expected = (float)(100 + d0 + d1 * 10 + d2 * 100 + d3 * 1000);
                        vx_size padded_idx = d0 + d1 * 4 + d2 * 4 * 3 + d3 * 4 * 3 * 4;
                        if (padded_read[padded_idx] != expected) mismatch++;
                    }
                }
            }
        }
        printf("STATUS: Padded read verification: %s (%d mismatches)\n",
               mismatch == 0 ? "PASS" : "FAIL", mismatch);
        if (mismatch) errors++;
    }
    free(padded_read);

    free(padded_write);
    free(write_data);
    CHECK_STATUS(vxReleaseTensor(&tensor));
    return errors;
}

// ------------------------------------------------------------------
// Test 11: vxQueryMetaFormatAttribute(VX_TENSOR_DIMS) round-trip
// Exercises the meta-format query path from inside an output validator:
// the validator sets the output tensor meta, then reads VX_TENSOR_NUMBER_OF_DIMS
// and VX_TENSOR_DIMS back and confirms they match what was set. This covers the
// tensor branch of vxQueryMetaFormatAttribute (type-checked before the union read).
// ------------------------------------------------------------------
static int g_meta_query_errors = 0;
static bool g_meta_query_ran = false;

static vx_status VX_CALLBACK meta_query_kernel_func(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    (void)node; (void)parameters; (void)num;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK meta_query_validate(vx_node node, const vx_reference *parameters,
                                                 vx_uint32 num, vx_meta_format metas[])
{
    (void)node; (void)parameters; (void)num;
    g_meta_query_ran = true;

    const vx_size num_dims = 3;
    vx_size dims[3] = {6, 4, 2};
    vx_enum data_type = VX_TYPE_UINT8;

    // Describe the output tensor meta (index 1).
    if (vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)) != VX_SUCCESS ||
        vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, dims, sizeof(dims)) != VX_SUCCESS ||
        vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type)) != VX_SUCCESS) {
        printf("ERROR: meta_query_validate failed to set tensor meta\n");
        g_meta_query_errors++;
        return VX_FAILURE;
    }

    // Query number-of-dims back.
    vx_size got_num_dims = 0;
    if (vxQueryMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &got_num_dims, sizeof(got_num_dims)) != VX_SUCCESS ||
        got_num_dims != num_dims) {
        printf("ERROR: VX_TENSOR_NUMBER_OF_DIMS query mismatch (got " VX_FMT_SIZE ")\n", got_num_dims);
        g_meta_query_errors++;
    }

    // Query dims back and confirm each value.
    vx_size got_dims[3] = {0, 0, 0};
    if (vxQueryMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, got_dims, sizeof(got_dims)) != VX_SUCCESS) {
        printf("ERROR: VX_TENSOR_DIMS query failed\n");
        g_meta_query_errors++;
    } else if (got_dims[0] != dims[0] || got_dims[1] != dims[1] || got_dims[2] != dims[2]) {
        printf("ERROR: VX_TENSOR_DIMS mismatch (got [" VX_FMT_SIZE "," VX_FMT_SIZE "," VX_FMT_SIZE "])\n",
               got_dims[0], got_dims[1], got_dims[2]);
        g_meta_query_errors++;
    }

    // Undersized query buffer must be rejected (must not partial-copy).
    vx_size tiny[1] = {0};
    if (vxQueryMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, tiny, sizeof(tiny)) == VX_SUCCESS) {
        printf("ERROR: VX_TENSOR_DIMS accepted an undersized buffer\n");
        g_meta_query_errors++;
    }

    // Querying an attribute for a different object type (VX_IMAGE_WIDTH on a tensor meta)
    // must report VX_ERROR_INVALID_TYPE, not VX_ERROR_INVALID_PARAMETERS.
    vx_uint32 dummy_w = 0;
    vx_status type_status = vxQueryMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &dummy_w, sizeof(dummy_w));
    if (type_status != VX_ERROR_INVALID_TYPE) {
        printf("ERROR: cross-type meta query returned %d, expected VX_ERROR_INVALID_TYPE\n", type_status);
        g_meta_query_errors++;
    }
    return VX_SUCCESS;
}

static int test_QueryMetaFormatTensorDims(vx_context context)
{
    int errors = 0;
    printf("\n=== Test: vxQueryMetaFormatAttribute (VX_TENSOR_DIMS) ===\n");

    vx_enum kernel_enum = VX_KERNEL_BASE(VX_ID_USER, 0) + 300;
    vx_kernel kernel = vxAddUserKernel(context,
                                       "org.test.meta_query_kernel",
                                       kernel_enum,
                                       meta_query_kernel_func,
                                       2,
                                       meta_query_validate,
                                       NULL, NULL);
    CHECK_NULL(kernel, "vxAddUserKernel for meta query test");
    if (!kernel) return 1;

    CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    CHECK_STATUS(vxFinalizeKernel(kernel));

    vx_graph graph = vxCreateGraph(context);
    CHECK_NULL(graph, "vxCreateGraph for meta query test");
    if (!graph) { vxReleaseKernel(&kernel); return errors + 1; }

    vx_size in_dims[3]  = {6, 4, 2};
    vx_size out_dims[3] = {6, 4, 2};
    vx_tensor in_tensor  = vxCreateTensor(context, 3, in_dims,  VX_TYPE_UINT8, 0);
    vx_tensor out_tensor = vxCreateTensor(context, 3, out_dims, VX_TYPE_UINT8, 0);
    CHECK_NULL(in_tensor, "input tensor");
    CHECK_NULL(out_tensor, "output tensor");

    vx_node node = vxCreateGenericNode(graph, kernel);
    CHECK_NULL(node, "vxCreateGenericNode for meta query test");
    if (node) {
        CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)in_tensor));
        CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)out_tensor));

        // Verifying the graph invokes the output validator, which runs the meta queries.
        vx_status vstatus = vxVerifyGraph(graph);
        if (vstatus != VX_SUCCESS) {
            printf("ERROR: vxVerifyGraph returned %d\n", vstatus);
            errors++;
        }
        if (!g_meta_query_ran) {
            printf("ERROR: output validator did not run - meta query path not exercised\n");
            errors++;
        } else if (g_meta_query_errors == 0) {
            printf("STATUS: VX_TENSOR_DIMS meta-format round-trip - PASS\n");
        }
        errors += g_meta_query_errors;
        vxReleaseNode(&node);
    }

    if (in_tensor)  vxReleaseTensor(&in_tensor);
    if (out_tensor) vxReleaseTensor(&out_tensor);
    vxReleaseGraph(&graph);
    vxReleaseKernel(&kernel);
    return errors;
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
int main()
{
    int errors = 0;
    printf("Tensor Advanced API Test\n");
    printf("========================\n");

    vx_context context = vxCreateContext();
    if (!context) {
        printf("ERROR: vxCreateContext failed\n");
        return 1;
    }

    errors += test_CreateTensorFromHandle(context);
    errors += test_SwapTensorHandle(context);
    errors += test_AliasTensor(context);
    errors += test_AddKernel(context);
    errors += test_ModuleHandle(context);
    errors += test_ModuleInternalData(context);
    errors += test_CopyTensorPatchStrided(context);
    errors += test_QueryMetaFormatTensorDims(context);

    vxReleaseContext(&context);

    printf("\n========================\n");
    printf("Tensor Advanced API test: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors ? 1 : 0;
}
