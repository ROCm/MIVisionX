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

// Requirement-tag coverage test for the OpenVX 1.3.2 Vision profile gaps
// reported in ROCm/MIVisionX#1762. Each case names the Khronos REQ tag from
// the CTS ReqTags suite that it guards, so a regression here points straight
// at the conformance test it would break.

#include <cstdio>
#include <cstring>
#include <VX/vx.h>

static int errors = 0;

static void check(const char *req, const char *what, bool ok, const char *detail)
{
    if (ok) printf("  PASS  %-9s %-52s %s\n", req, what, detail ? detail : "");
    else  { printf("  FAIL  %-9s %-52s %s\n", req, what, detail ? detail : ""); errors++; }
}

// REQ-0834/0840/0858/0887/0916: VX_IMAGE_PLANES is a mandatory image attribute
// and must be readable into either a vx_size or a vx_uint32 container.
static void test_image_planes(vx_context context)
{
    struct { const char *req; vx_df_image format; vx_size planes; } cases[] = {
        { "REQ-0834", VX_DF_IMAGE_U8,   1 },
        { "REQ-0840", VX_DF_IMAGE_RGB,  1 },
        { "REQ-0858", VX_DF_IMAGE_NV12, 2 },
        { "REQ-0916", VX_DF_IMAGE_IYUV, 3 },
    };
    char detail[128];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        vx_image image = vxCreateImage(context, 16, 16, cases[i].format);
        vx_size as_size = 0;
        vx_uint32 as_uint32 = 0;
        vx_status s1 = vxQueryImage(image, VX_IMAGE_PLANES, &as_size, sizeof(as_size));
        vx_status s2 = vxQueryImage(image, VX_IMAGE_PLANES, &as_uint32, sizeof(as_uint32));
        snprintf(detail, sizeof(detail), "vx_size=%zu vx_uint32=%u", (size_t)as_size, as_uint32);
        check(cases[i].req, "vxQueryImage VX_IMAGE_PLANES",
              s1 == VX_SUCCESS && as_size == cases[i].planes &&
              s2 == VX_SUCCESS && as_uint32 == (vx_uint32)cases[i].planes, detail);
        vxReleaseImage(&image);
    }
}

// REQ-0697: VX_PYRAMID_LEVELS is a mandatory pyramid attribute.
static void test_pyramid_levels(vx_context context)
{
    char detail[128];
    vx_pyramid pyramid = vxCreatePyramid(context, 4, VX_SCALE_PYRAMID_HALF, 64, 64, VX_DF_IMAGE_U8);
    vx_size as_size = 0;
    vx_uint32 as_uint32 = 0;
    vx_status s1 = vxQueryPyramid(pyramid, VX_PYRAMID_LEVELS, &as_size, sizeof(as_size));
    vx_status s2 = vxQueryPyramid(pyramid, VX_PYRAMID_LEVELS, &as_uint32, sizeof(as_uint32));
    snprintf(detail, sizeof(detail), "vx_size=%zu vx_uint32=%u", (size_t)as_size, as_uint32);
    check("REQ-0697", "vxQueryPyramid VX_PYRAMID_LEVELS",
          s1 == VX_SUCCESS && as_size == 4 && s2 == VX_SUCCESS && as_uint32 == 4, detail);
    vxReleasePyramid(&pyramid);
}

// REQ-0642/0758/1467: VX_REFERENCE_NAME is readable through either a caller
// supplied buffer or a returned pointer, and nameable on every named object.
static void check_reference_name(const char *req, const char *what, vx_reference ref)
{
    char detail[192];
    vx_status sset = vxSetReferenceName(ref, "reqtag_name");
    vx_char buffer[128];
    memset(buffer, 0, sizeof(buffer));
    vx_status sbuf = vxQueryReference(ref, VX_REFERENCE_NAME, buffer, sizeof(buffer));
    vx_char *returned = NULL;
    vx_status sptr = vxQueryReference(ref, VX_REFERENCE_NAME, &returned, sizeof(returned));
    snprintf(detail, sizeof(detail), "buffer='%s' pointer='%s'",
             buffer, returned ? returned : "(null)");
    check(req, what,
          sset == VX_SUCCESS && sbuf == VX_SUCCESS && !strcmp(buffer, "reqtag_name") &&
          sptr == VX_SUCCESS && returned && !strcmp(returned, "reqtag_name"), detail);
}

static void test_reference_names(vx_context context)
{
    vx_graph graph = vxCreateGraph(context);
    check_reference_name("REQ-0642", "reference name on vx_graph", (vx_reference)graph);
    vxReleaseGraph(&graph);

    vx_array array = vxCreateArray(context, VX_TYPE_KEYPOINT, 16);
    check_reference_name("REQ-0758", "reference name on vx_array", (vx_reference)array);
    vxReleaseArray(&array);

    vx_image exemplar = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8);
    vx_object_array objects = vxCreateObjectArray(context, (vx_reference)exemplar, 4);
    check_reference_name("REQ-1467", "reference name on vx_object_array", (vx_reference)objects);
    vxReleaseObjectArray(&objects);
    vxReleaseImage(&exemplar);
}

// REQ-0579: the context counts itself, so it always holds at least one reference.
static void test_context_references(vx_context context)
{
    char detail[64];
    vx_uint32 references = 0;
    vx_status status = vxQueryContext(context, VX_CONTEXT_REFERENCES, &references, sizeof(references));
    snprintf(detail, sizeof(detail), "references=%u", references);
    check("REQ-0579", "vxQueryContext VX_CONTEXT_REFERENCES >= 1",
          status == VX_SUCCESS && references > 0, detail);
}

// REQ-0650/0690: a graph with no nodes is a legal graph and verifies.
static void test_empty_graph(vx_context context)
{
    vx_graph graph = vxCreateGraph(context);
    check("REQ-0650", "vxVerifyGraph on a graph with zero nodes",
          vxVerifyGraph(graph) == VX_SUCCESS, NULL);
    vxReleaseGraph(&graph);

    vx_graph graph2 = vxCreateGraph(context);
    vx_image input = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_image output = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_node node = vxBox3x3Node(graph2, input, output);
    vx_status sremove = vxRemoveNode(&node);
    check("REQ-0690", "vxVerifyGraph after every node is removed",
          sremove == VX_SUCCESS && vxVerifyGraph(graph2) == VX_SUCCESS, NULL);
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    vxReleaseGraph(&graph2);
}

// REQ-0669/0670/0671/0672/0680/0693/0696/0699: replication state is reportable.
static void test_node_replication(vx_context context)
{
    char detail[160];

    vx_graph graph = vxCreateGraph(context);
    vx_image input = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_image output = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_node node = vxBox3x3Node(graph, input, output);

    vx_bool is_replicated = vx_true_e;
    vx_status sq = vxQueryNode(node, VX_NODE_IS_REPLICATED, &is_replicated, sizeof(is_replicated));
    check("REQ-0680", "VX_NODE_IS_REPLICATED is false on a fresh node",
          sq == VX_SUCCESS && is_replicated == vx_false_e, NULL);

    vx_bool flags[2] = { vx_true_e, vx_true_e };
    vx_status sf = vxQueryNode(node, VX_NODE_REPLICATE_FLAGS, flags, sizeof(flags));
    check("REQ-0671", "VX_NODE_REPLICATE_FLAGS is clear on a fresh node",
          sf == VX_SUCCESS && flags[0] == vx_false_e && flags[1] == vx_false_e, NULL);

    vxReleaseNode(&node);
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    vxReleaseGraph(&graph);

    vx_graph rgraph = vxCreateGraph(context);
    vx_pyramid pin = vxCreatePyramid(context, 4, VX_SCALE_PYRAMID_HALF, 64, 64, VX_DF_IMAGE_U8);
    vx_pyramid pout = vxCreatePyramid(context, 4, VX_SCALE_PYRAMID_HALF, 64, 64, VX_DF_IMAGE_U8);
    vx_image level0in = vxGetPyramidLevel(pin, 0);
    vx_image level0out = vxGetPyramidLevel(pout, 0);
    vx_node rnode = vxBox3x3Node(rgraph, level0in, level0out);
    vx_bool replicate[2] = { vx_true_e, vx_true_e };
    vx_status srep = vxReplicateNode(rgraph, rnode, replicate, 2);

    vx_bool ris = vx_false_e;
    vx_status sris = vxQueryNode(rnode, VX_NODE_IS_REPLICATED, &ris, sizeof(ris));
    vx_bool rflags[2] = { vx_false_e, vx_false_e };
    vx_status srflags = vxQueryNode(rnode, VX_NODE_REPLICATE_FLAGS, rflags, sizeof(rflags));
    snprintf(detail, sizeof(detail), "is_replicated=%d flags=[%d,%d]",
             (int)ris, (int)rflags[0], (int)rflags[1]);
    check("REQ-0696", "replication state after vxReplicateNode",
          srep == VX_SUCCESS && sris == VX_SUCCESS && ris == vx_true_e &&
          srflags == VX_SUCCESS && rflags[0] == vx_true_e && rflags[1] == vx_true_e, detail);

    vxReleaseNode(&rnode);
    vxReleaseImage(&level0in);
    vxReleaseImage(&level0out);
    vxReleasePyramid(&pin);
    vxReleasePyramid(&pout);
    vxReleaseGraph(&rgraph);
}

// REQ-0333: the mask of a non-linear filter is only inspectable once written,
// and an unwritten mask used to crash graph verification.
static void test_non_linear_filter(vx_context context)
{
    vx_graph graph = vxCreateGraph(context);
    vx_image input = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_image output = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_matrix mask = vxCreateMatrix(context, VX_TYPE_UINT8, 3, 3);
    vx_node node = vxNonLinearFilterNode(graph, VX_NONLINEAR_FILTER_MEDIAN, input, mask, output);
    check("REQ-0333", "vxNonLinearFilterNode verifies with an unwritten mask",
          vxGetStatus((vx_reference)node) == VX_SUCCESS && vxVerifyGraph(graph) == VX_SUCCESS, NULL);
    if (node) vxReleaseNode(&node);
    vxReleaseMatrix(&mask);
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    vxReleaseGraph(&graph);
}

// REQ-0286/0287: a remap node is creatable, verifiable and reports as a node.
static void test_remap(vx_context context)
{
    char detail[64];
    vx_graph graph = vxCreateGraph(context);
    vx_image input = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_image output = vxCreateImage(context, 32, 32, VX_DF_IMAGE_U8);
    vx_remap map = vxCreateRemap(context, 64, 64, 32, 32);
    vx_node node = vxRemapNode(graph, input, map, VX_INTERPOLATION_NEAREST_NEIGHBOR, output);
    check("REQ-0286", "vxRemapNode is creatable and graph verifiable",
          vxGetStatus((vx_reference)node) == VX_SUCCESS && vxVerifyGraph(graph) == VX_SUCCESS, NULL);

    vx_enum type = 0;
    vx_status status = vxQueryReference((vx_reference)node, VX_REFERENCE_TYPE, &type, sizeof(type));
    snprintf(detail, sizeof(detail), "type=0x%x", (unsigned)type);
    check("REQ-0287", "remap node reference type is VX_TYPE_NODE",
          status == VX_SUCCESS && type == VX_TYPE_NODE, detail);

    if (node) vxReleaseNode(&node);
    vxReleaseRemap(&map);
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    vxReleaseGraph(&graph);
}

// REQ-0236: the reconstructed output format is independent of the lowest
// resolution input format, so an S16 pyramid can drive a U8 output.
static void test_laplacian_reconstruct(vx_context context)
{
    vx_graph graph = vxCreateGraph(context);
    vx_pyramid laplacian = vxCreatePyramid(context, 3, VX_SCALE_PYRAMID_HALF, 64, 64, VX_DF_IMAGE_S16);
    vx_image lowest = vxCreateImage(context, 8, 8, VX_DF_IMAGE_S16);
    vx_image output = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
    vx_node node = vxLaplacianReconstructNode(graph, laplacian, lowest, output);
    check("REQ-0236", "laplacian reconstruct S16 pyramid to U8 output",
          vxGetStatus((vx_reference)node) == VX_SUCCESS && vxVerifyGraph(graph) == VX_SUCCESS, NULL);
    if (node) vxReleaseNode(&node);
    vxReleasePyramid(&laplacian);
    vxReleaseImage(&lowest);
    vxReleaseImage(&output);
    vxReleaseGraph(&graph);
}

// REQ-0499/0502/0504/0507: the spec calls for a 2x3 float32 affine matrix without
// fixing which dimension is which, so either orientation has to be accepted.
static void test_warp_affine_matrix(vx_context context)
{
    struct { const char *req; vx_size columns; vx_size rows; } cases[] = {
        { "REQ-0500", 2, 3 },
        { "REQ-0499", 3, 2 },
    };
    char detail[64];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        vx_graph graph = vxCreateGraph(context);
        vx_image input = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
        vx_image output = vxCreateImage(context, 64, 64, VX_DF_IMAGE_U8);
        vx_matrix matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, cases[i].columns, cases[i].rows);
        vx_node node = vxWarpAffineNode(graph, input, matrix, VX_INTERPOLATION_NEAREST_NEIGHBOR, output);
        snprintf(detail, sizeof(detail), "%zux%zu matrix",
                 (size_t)cases[i].columns, (size_t)cases[i].rows);
        check(cases[i].req, "vxWarpAffineNode accepts either matrix orientation",
              vxGetStatus((vx_reference)node) == VX_SUCCESS && vxVerifyGraph(graph) == VX_SUCCESS, detail);
        if (node) vxReleaseNode(&node);
        vxReleaseMatrix(&matrix);
        vxReleaseImage(&input);
        vxReleaseImage(&output);
        vxReleaseGraph(&graph);
    }
}

// REQ-0726: adding past the capacity of an array reports VX_FAILURE and leaves
// the item count untouched.
static void test_array_full(vx_context context)
{
    vx_array array = vxCreateArray(context, VX_TYPE_KEYPOINT, 2);
    vx_keypoint_t items[2];
    memset(items, 0, sizeof(items));
    vxAddArrayItems(array, 2, items, sizeof(items[0]));

    vx_keypoint_t extra;
    memset(&extra, 0, sizeof(extra));
    vx_status status = vxAddArrayItems(array, 1, &extra, sizeof(extra));
    vx_size numitems = 0;
    vxQueryArray(array, VX_ARRAY_NUMITEMS, &numitems, sizeof(numitems));
    check("REQ-0726", "vxAddArrayItems on a full array returns VX_FAILURE",
          status == VX_FAILURE && numitems == 2, NULL);
    vxReleaseArray(&array);
}

static vx_status VX_CALLBACK noop_kernel(vx_node, const vx_reference *, vx_uint32)
{
    return VX_SUCCESS;
}

// REQ-1896: a user kernel declares its parameter count up front, so an index
// beyond it must be rejected.
static void test_add_parameter_bounds(vx_context context)
{
    vx_enum kernel_id = 0;
    vxAllocateUserKernelId(context, &kernel_id);
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.test.reqtags_bounds",
                                       kernel_id, noop_kernel, 2, NULL, NULL, NULL);
    vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
    vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
    vx_status status = vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
    check("REQ-1896", "vxAddParameterToKernel rejects an out of range index",
          status != VX_SUCCESS, NULL);
    vxFinalizeKernel(kernel);
    vxRemoveKernel(kernel);
}

// REQ-1760/1981: VX_PARAMETER_META_FORMAT returns the meta format of the parameter,
// and each query hands out a reference the application has to release.
static void test_parameter_meta_format(vx_context context)
{
    vx_kernel kernel = vxGetKernelByEnum(context, VX_KERNEL_BOX_3x3);
    vx_parameter parameter = vxGetKernelParameterByIndex(kernel, 0);
    vx_meta_format meta = 0;
    vx_status squery = vxQueryParameter(parameter, VX_PARAMETER_META_FORMAT, &meta, sizeof(meta));
    vx_reference meta_ref = (vx_reference)meta;
    vx_status srelease = vxReleaseReference(&meta_ref);
    check("REQ-1760", "vxQueryParameter VX_PARAMETER_META_FORMAT",
          squery == VX_SUCCESS && meta != 0 && srelease == VX_SUCCESS, NULL);
    vxReleaseParameter(&parameter);
    vxReleaseKernel(&kernel);
}

int main()
{
    vx_context context = vxCreateContext();
    if (vxGetStatus((vx_reference)context) != VX_SUCCESS) {
        printf("ERROR: vxCreateContext failed\n");
        return 1;
    }

    printf("=== OpenVX 1.3.2 requirement tag coverage ===\n");
    test_context_references(context);
    test_image_planes(context);
    test_pyramid_levels(context);
    test_reference_names(context);
    test_empty_graph(context);
    test_node_replication(context);
    test_non_linear_filter(context);
    test_remap(context);
    test_laplacian_reconstruct(context);
    test_warp_affine_matrix(context);
    test_array_full(context);
    test_add_parameter_bounds(context);
    test_parameter_meta_format(context);

    printf("\n%s: %d failure(s)\n", errors ? "FAILED" : "PASSED", errors);
    vxReleaseContext(&context);
    return errors ? 1 : 0;
}
