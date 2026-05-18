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

// User Kernel API coverage test - exercises vxGetKernelByName,
// vxGetKernelByEnum, vxQueryKernel, vxAddUserKernel,
// vxAddParameterToKernel, vxFinalizeKernel, vxRemoveKernel,
// vxSetKernelAttribute, vxReleaseKernel,
// vxQueryNode, vxSetNodeAttribute, vxReleaseNode,
// vxQueryParameter, vxReleaseParameter, vxGetParameterByIndex,
// vxQueryContext, vxSetContextAttribute, vxQueryReference,
// vxSetReferenceName, vxQueryGraph, vxSetGraphAttribute

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

// Simple user kernel: copies input image to output
static vx_status VX_CALLBACK user_kernel_func(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    (void)node;
    if (num < 2) return VX_FAILURE;
    vx_image input = (vx_image)parameters[0];
    vx_image output = (vx_image)parameters[1];

    vx_uint32 width = 0, height = 0;
    vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width));
    vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height));

    vx_rectangle_t rect = {0, 0, width, height};
    vx_imagepatch_addressing_t addr_in = {}, addr_out = {};
    void *ptr_in = NULL, *ptr_out = NULL;
    vx_map_id map_in = 0, map_out = 0;

    vxMapImagePatch(input, &rect, 0, &map_in, &addr_in, &ptr_in, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    vxMapImagePatch(output, &rect, 0, &map_out, &addr_out, &ptr_out, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

    if (ptr_in && ptr_out) {
        for (vx_uint32 y = 0; y < height; y++) {
            vx_uint8 *src = (vx_uint8 *)ptr_in + y * addr_in.stride_y;
            vx_uint8 *dst = (vx_uint8 *)ptr_out + y * addr_out.stride_y;
            memcpy(dst, src, width);
        }
    }

    vxUnmapImagePatch(input, map_in);
    vxUnmapImagePatch(output, map_out);
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK user_kernel_validator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    (void)node;
    if (num < 2) return VX_FAILURE;
    vx_uint32 width = 0, height = 0;
    vx_df_image format = 0;
    vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width));
    vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height));
    vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format));
    vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &width, sizeof(width));
    vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &height, sizeof(height));
    vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &format, sizeof(format));
    return VX_SUCCESS;
}

static int test_kernel_api(vx_context context) {
    int errors = 0;
    printf("\n=== Kernel API ===\n");

    // Get kernel by name
    vx_kernel k = vxGetKernelByName(context, "org.khronos.openvx.add");
    if (k) {
        printf("STATUS: vxGetKernelByName 'org.khronos.openvx.add' - OK\n");

        // Query kernel attributes
        vx_enum kenum = 0;
        CHECK_STATUS(vxQueryKernel(k, VX_KERNEL_ENUM, &kenum, sizeof(kenum)));
        printf("STATUS: vxQueryKernel ENUM=%d\n", kenum);

        vx_uint32 num_params = 0;
        CHECK_STATUS(vxQueryKernel(k, VX_KERNEL_PARAMETERS, &num_params, sizeof(num_params)));
        printf("STATUS: vxQueryKernel PARAMETERS=%u\n", num_params);

        char kname[256] = {0};
        CHECK_STATUS(vxQueryKernel(k, VX_KERNEL_NAME, kname, sizeof(kname)));
        printf("STATUS: vxQueryKernel NAME=%s\n", kname);

        CHECK_STATUS(vxReleaseKernel(&k));
    }

    // Get kernel by enum
    vx_kernel k2 = vxGetKernelByEnum(context, VX_KERNEL_ADD);
    if (k2) {
        printf("STATUS: vxGetKernelByEnum VX_KERNEL_ADD - OK\n");
        CHECK_STATUS(vxReleaseKernel(&k2));
    }

    // Add user kernel
    vx_enum user_kernel_id = 0;
    CHECK_STATUS(vxAllocateUserKernelId(context, &user_kernel_id));
    vx_kernel uk = vxAddUserKernel(context, "test.user_copy", user_kernel_id,
                                    user_kernel_func, 2, user_kernel_validator,
                                    NULL, NULL);
    if (uk) {
        printf("STATUS: vxAddUserKernel 'test.user_copy' - OK\n");

        CHECK_STATUS(vxAddParameterToKernel(uk, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        CHECK_STATUS(vxAddParameterToKernel(uk, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        printf("STATUS: vxAddParameterToKernel - OK\n");

        CHECK_STATUS(vxFinalizeKernel(uk));
        printf("STATUS: vxFinalizeKernel - OK\n");

        // Use the user kernel in a graph
        vx_graph graph = vxCreateGraph(context);
        if (graph) {
            vx_image in_img = vxCreateImage(context, 64, 48, VX_DF_IMAGE_U8);
            vx_image out_img = vxCreateImage(context, 64, 48, VX_DF_IMAGE_U8);

            vx_node node = vxCreateGenericNode(graph, uk);
            if (node) {
                CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)in_img));
                CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)out_img));

                // Query node
                vx_status node_status = 0;
                CHECK_STATUS(vxQueryNode(node, VX_NODE_STATUS, &node_status, sizeof(node_status)));
                printf("STATUS: vxQueryNode STATUS=%d\n", node_status);

                vx_perf_t perf = {};
                CHECK_STATUS(vxQueryNode(node, VX_NODE_PERFORMANCE, &perf, sizeof(perf)));
                printf("STATUS: vxQueryNode PERFORMANCE - OK\n");

                vx_border_t border = {};
                CHECK_STATUS(vxQueryNode(node, VX_NODE_BORDER, &border, sizeof(border)));
                printf("STATUS: vxQueryNode BORDER - OK\n");

                // Set node attribute (border)
                vx_border_t new_border = {};
                new_border.mode = VX_BORDER_REPLICATE;
                CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &new_border, sizeof(new_border)));
                printf("STATUS: vxSetNodeAttribute BORDER - OK\n");

                // Get parameter by index
                vx_parameter param = vxGetParameterByIndex(node, 0);
                if (param) {
                    vx_enum dir = 0;
                    CHECK_STATUS(vxQueryParameter(param, VX_PARAMETER_DIRECTION, &dir, sizeof(dir)));
                    printf("STATUS: vxQueryParameter DIRECTION=%d\n", dir);

                    vx_enum ptype = 0;
                    CHECK_STATUS(vxQueryParameter(param, VX_PARAMETER_TYPE, &ptype, sizeof(ptype)));
                    printf("STATUS: vxQueryParameter TYPE=%d\n", ptype);

                    vx_enum state = 0;
                    CHECK_STATUS(vxQueryParameter(param, VX_PARAMETER_STATE, &state, sizeof(state)));
                    printf("STATUS: vxQueryParameter STATE=%d\n", state);

                    vx_reference pref = NULL;
                    CHECK_STATUS(vxQueryParameter(param, VX_PARAMETER_REF, &pref, sizeof(pref)));
                    printf("STATUS: vxQueryParameter REF=%p\n", pref);

                    CHECK_STATUS(vxReleaseParameter(&param));
                }

                // Verify and execute graph
                vx_status verify_s = vxVerifyGraph(graph);
                if (verify_s == VX_SUCCESS) {
                    printf("STATUS: vxVerifyGraph - OK\n");
                    CHECK_STATUS(vxProcessGraph(graph));
                    printf("STATUS: vxProcessGraph with user kernel - OK\n");
                }

                CHECK_STATUS(vxReleaseNode(&node));
            }

            vxReleaseImage(&in_img);
            vxReleaseImage(&out_img);
            vxReleaseGraph(&graph);
        }

        CHECK_STATUS(vxRemoveKernel(uk));
        printf("STATUS: vxRemoveKernel - OK\n");
    }

    printf("STATUS: Kernel API complete (%d errors)\n", errors);
    return errors;
}

static int test_context_api(vx_context context) {
    int errors = 0;
    printf("\n=== Context API ===\n");

    // Query context
    vx_uint16 vendor_id = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_VENDOR_ID, &vendor_id, sizeof(vendor_id)));
    printf("STATUS: vxQueryContext VENDOR_ID=%u\n", vendor_id);

    vx_uint16 version = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_VERSION, &version, sizeof(version)));
    printf("STATUS: vxQueryContext VERSION=%u\n", version);

    vx_uint32 num_kernels = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_UNIQUE_KERNELS, &num_kernels, sizeof(num_kernels)));
    printf("STATUS: vxQueryContext UNIQUE_KERNELS=%u\n", num_kernels);

    vx_uint32 num_modules = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_MODULES, &num_modules, sizeof(num_modules)));
    printf("STATUS: vxQueryContext MODULES=%u\n", num_modules);

    vx_uint32 num_refs = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_REFERENCES, &num_refs, sizeof(num_refs)));
    printf("STATUS: vxQueryContext REFERENCES=%u\n", num_refs);

    char impl_name[256] = {0};
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_IMPLEMENTATION, impl_name, sizeof(impl_name)));
    printf("STATUS: vxQueryContext IMPLEMENTATION=%s\n", impl_name);

    vx_size extensions_size = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_EXTENSIONS_SIZE, &extensions_size, sizeof(extensions_size)));
    printf("STATUS: vxQueryContext EXTENSIONS_SIZE=%zu\n", extensions_size);

    if (extensions_size > 0 && extensions_size < 4096) {
        char *extensions = (char *)calloc(extensions_size + 1, 1);
        if (extensions) {
            vxQueryContext(context, VX_CONTEXT_EXTENSIONS, extensions, extensions_size);
            printf("STATUS: vxQueryContext EXTENSIONS=%s\n", extensions);
            free(extensions);
        }
    }

    vx_size conv_max = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_CONVOLUTION_MAX_DIMENSION, &conv_max, sizeof(conv_max)));
    printf("STATUS: vxQueryContext CONVOLUTION_MAX_DIMENSION=%zu\n", conv_max);

    vx_size opt_width = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_OPTICAL_FLOW_MAX_WINDOW_DIMENSION, &opt_width, sizeof(opt_width)));
    printf("STATUS: vxQueryContext OPTICAL_FLOW_MAX_WINDOW_DIMENSION=%zu\n", opt_width);

    vx_border_t border = {};
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_IMMEDIATE_BORDER, &border, sizeof(border)));
    printf("STATUS: vxQueryContext IMMEDIATE_BORDER mode=%d\n", border.mode);

    vx_enum target = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_IMMEDIATE_BORDER_POLICY, &target, sizeof(target)));
    printf("STATUS: vxQueryContext IMMEDIATE_BORDER_POLICY=%d\n", target);

    vx_size nonlinear_max = 0;
    CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_NONLINEAR_MAX_DIMENSION, &nonlinear_max, sizeof(nonlinear_max)));
    printf("STATUS: vxQueryContext NONLINEAR_MAX_DIMENSION=%zu\n", nonlinear_max);

    // Set context attribute (immediate border)
    vx_border_t new_border = {};
    new_border.mode = VX_BORDER_REPLICATE;
    CHECK_STATUS(vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &new_border, sizeof(new_border)));
    printf("STATUS: vxSetContextAttribute IMMEDIATE_BORDER - OK\n");

    printf("STATUS: Context API complete (%d errors)\n", errors);
    return errors;
}

static int test_graph_api(vx_context context) {
    int errors = 0;
    printf("\n=== Graph API ===\n");

    vx_graph graph = vxCreateGraph(context);
    if (graph) {
        // Query graph
        vx_uint32 num_nodes = 0;
        CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_NUMNODES, &num_nodes, sizeof(num_nodes)));
        printf("STATUS: vxQueryGraph NUMNODES=%u\n", num_nodes);

        vx_status gstatus = 0;
        CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_STATE, &gstatus, sizeof(gstatus)));
        printf("STATUS: vxQueryGraph STATE=%d\n", gstatus);

        vx_perf_t perf = {};
        CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_PERFORMANCE, &perf, sizeof(perf)));
        printf("STATUS: vxQueryGraph PERFORMANCE - OK\n");

        vx_uint32 num_params = 0;
        CHECK_STATUS(vxQueryGraph(graph, VX_GRAPH_NUMPARAMETERS, &num_params, sizeof(num_params)));
        printf("STATUS: vxQueryGraph NUMPARAMETERS=%u\n", num_params);

        // Set reference name
        CHECK_STATUS(vxSetReferenceName((vx_reference)graph, "test_graph"));
        printf("STATUS: vxSetReferenceName - OK\n");

        // Query reference
        vx_enum ref_type = 0;
        CHECK_STATUS(vxQueryReference((vx_reference)graph, VX_REFERENCE_TYPE, &ref_type, sizeof(ref_type)));
        printf("STATUS: vxQueryReference TYPE=%d\n", ref_type);

        vx_uint32 ref_count = 0;
        CHECK_STATUS(vxQueryReference((vx_reference)graph, VX_REFERENCE_COUNT, &ref_count, sizeof(ref_count)));
        printf("STATUS: vxQueryReference COUNT=%u\n", ref_count);

        char ref_name[256] = {0};
        CHECK_STATUS(vxQueryReference((vx_reference)graph, VX_REFERENCE_NAME, ref_name, sizeof(ref_name)));
        printf("STATUS: vxQueryReference NAME=%s\n", ref_name);

        vxReleaseGraph(&graph);
    }

    printf("STATUS: Graph API complete (%d errors)\n", errors);
    return errors;
}

int main() {
    int errors = 0;

    vx_context context = vxCreateContext();
    if (!context) {
        printf("ERROR: vxCreateContext failed\n");
        return 1;
    }

    errors += test_kernel_api(context);
    errors += test_context_api(context);
    errors += test_graph_api(context);

    vxReleaseContext(&context);

    printf("\nUser Kernel & Context API test: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors ? 1 : 0;
}
