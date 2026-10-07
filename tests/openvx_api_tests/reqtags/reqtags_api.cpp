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

// REQ-0695: a node replicated over virtual pyramids verifies and runs. Nothing in
// the graph writes the input pyramid, which is allowed even though its contents
// are undefined, so verification must not reject it.
static void test_replication_over_virtual_pyramids(vx_context context)
{
    char detail[96];
    vx_graph graph = vxCreateGraph(context);
    vx_pyramid pin = vxCreateVirtualPyramid(graph, 3, VX_SCALE_PYRAMID_HALF, 64, 64, VX_DF_IMAGE_U8);
    vx_pyramid pout = vxCreateVirtualPyramid(graph, 3, VX_SCALE_PYRAMID_HALF, 64, 64, VX_DF_IMAGE_U8);
    vx_image level0in = vxGetPyramidLevel(pin, 0);
    vx_image level0out = vxGetPyramidLevel(pout, 0);
    vx_node node = vxGaussian3x3Node(graph, level0in, level0out);
    vx_bool replicate[2] = { vx_true_e, vx_true_e };
    vx_status srep = vxReplicateNode(graph, node, replicate, 2);
    vx_status sverify = vxVerifyGraph(graph);
    vx_status sprocess = vxProcessGraph(graph);
    snprintf(detail, sizeof(detail), "verify=%d process=%d", (int)sverify, (int)sprocess);
    check("REQ-0695", "replicated node over virtual pyramids verifies",
          srep == VX_SUCCESS && sverify == VX_SUCCESS && sprocess == VX_SUCCESS, detail);

    vxReleaseNode(&node);
    vxReleaseImage(&level0in);
    vxReleaseImage(&level0out);
    vxReleasePyramid(&pin);
    vxReleasePyramid(&pout);
    vxReleaseGraph(&graph);
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

// REQ-1269: vxCopyRemapPatch addresses the user buffer relative to the patch origin,
// so a write then read of the same patch round trips.
static void test_remap_patch_roundtrip(vx_context context)
{
    char detail[96];
    vx_remap map = vxCreateRemap(context, 64, 64, 32, 32);
    vx_rectangle_t rect = { 5, 7, 6, 8 };
    vx_coordinates2df_t written;
    written.x = 12.5f;
    written.y = 34.75f;
    vx_status swrite = vxCopyRemapPatch(map, &rect, sizeof(vx_coordinates2df_t), &written,
                                        VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vx_coordinates2df_t read;
    read.x = 0.0f;
    read.y = 0.0f;
    vx_status sread = vxCopyRemapPatch(map, &rect, sizeof(vx_coordinates2df_t), &read,
                                       VX_TYPE_COORDINATES2DF, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    snprintf(detail, sizeof(detail), "wrote (%g,%g) read (%g,%g)",
             written.x, written.y, read.x, read.y);
    check("REQ-1269", "vxCopyRemapPatch write then read round trips",
          swrite == VX_SUCCESS && sread == VX_SUCCESS &&
          read.x == written.x && read.y == written.y, detail);
    vxReleaseRemap(&map);
}

// vxMapRemapPatch must reject a patch outside the destination dimensions rather than
// hand back a pointer past the table, and an unmap of a sub-patch must publish only the
// elements it covers.
static void test_remap_map_patch(vx_context context)
{
    vx_remap map = vxCreateRemap(context, 64, 64, 32, 32);
    vx_map_id map_id = 0;
    vx_size stride_y = 0;
    void *ptr = NULL;

    vx_rectangle_t outside = { 0, 32, 1, 33 };
    vx_status sbad = vxMapRemapPatch(map, &outside, &map_id, &stride_y, &ptr,
                                     VX_TYPE_COORDINATES2DF, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    check("REQ-1269", "vxMapRemapPatch rejects a patch outside the remap",
          sbad != VX_SUCCESS, NULL);

    vx_rectangle_t rect = { 3, 4, 5, 6 };
    vx_status smap = vxMapRemapPatch(map, &rect, &map_id, &stride_y, &ptr,
                                     VX_TYPE_COORDINATES2DF, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST);
    bool ok = (smap == VX_SUCCESS && ptr != NULL && stride_y >= 2 * sizeof(vx_coordinates2df_t));
    if (ok) {
        // an out-of-source coordinate (-1 at 3 fractional bits) must not be converted blindly
        vx_coordinates2df_t *first = (vx_coordinates2df_t *)ptr;
        first[0].x = -1.0f;
        first[0].y = 20.5f;
        vx_coordinates2df_t *second = (vx_coordinates2df_t *)((vx_uint8 *)ptr + stride_y) + 1;
        second->x = 10.25f;
        second->y = 11.5f;
        ok = (vxUnmapRemapPatch(map, map_id) == VX_SUCCESS);
    }
    vx_rectangle_t one = { 4, 5, 5, 6 };
    vx_coordinates2df_t got;
    got.x = 0.0f; got.y = 0.0f;
    vx_status sget = vxCopyRemapPatch(map, &one, sizeof(got), &got, VX_TYPE_COORDINATES2DF,
                                      VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    check("REQ-1269", "vxMapRemapPatch sub-patch write is visible after unmap",
          ok && sget == VX_SUCCESS && got.x == 10.25f && got.y == 11.5f, NULL);
    vxReleaseRemap(&map);
}

// A node parameter owns its lazily created meta format. Querying the meta format of a kernel
// parameter before nodes are created must not make the nodes share (and double free) it.
static void test_node_parameter_meta_format_ownership(vx_context context)
{
    vx_kernel kernel = vxGetKernelByEnum(context, VX_KERNEL_BOX_3x3);
    vx_parameter kparam = vxGetKernelParameterByIndex(kernel, 0);
    vx_meta_format kmeta = 0;
    vxQueryParameter(kparam, VX_PARAMETER_META_FORMAT, &kmeta, sizeof(kmeta));
    vx_reference kmeta_ref = (vx_reference)kmeta;
    vxReleaseReference(&kmeta_ref);

    vx_graph graph = vxCreateGraph(context);
    vx_image input = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8);
    vx_image out0 = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8);
    vx_image out1 = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8);
    vx_node n0 = vxBox3x3Node(graph, input, out0);
    vx_node n1 = vxBox3x3Node(graph, input, out1);
    bool ok = true;
    vx_node nodes[2] = { n0, n1 };
    for (int i = 0; i < 2; i++) {
        vx_parameter nparam = vxGetParameterByIndex(nodes[i], 0);
        vx_meta_format nmeta = 0;
        vx_status s = vxQueryParameter(nparam, VX_PARAMETER_META_FORMAT, &nmeta, sizeof(nmeta));
        vx_reference nmeta_ref = (vx_reference)nmeta;
        ok = ok && s == VX_SUCCESS && nmeta != 0 && nmeta != kmeta;
        vxReleaseReference(&nmeta_ref);
        vxReleaseParameter(&nparam);
    }
    vxReleaseNode(&n0);
    vxReleaseNode(&n1);
    vxReleaseGraph(&graph); // would double free a meta format shared with the kernel
    vxReleaseImage(&input);
    vxReleaseImage(&out0);
    vxReleaseImage(&out1);
    vxReleaseParameter(&kparam);
    vxReleaseKernel(&kernel);
    check("REQ-1760", "node parameters own independent meta formats", ok, NULL);
}

// REQ-0938: a VX_DF_IMAGE_U1 patch that does not end on a byte boundary only writes the pixels
// inside the patch, and reads back exactly those pixels.
static void test_u1_partial_byte_patch(vx_context context)
{
    char detail[96];
    const vx_uint32 width = 16, height = 4;
    vx_image image = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
    vx_rectangle_t full = { 0, 0, width, height };
    vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
    addr.dim_x = width; addr.dim_y = height;
    addr.stride_x = 0; addr.stride_x_bits = 1; addr.stride_y = 2;

    vx_uint8 zeros[8] = { 0 };
    vx_status s0 = vxCopyImagePatch(image, &full, 0, &addr, zeros, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    // 12 pixels wide: one whole byte plus the low four bits of the next one
    vx_rectangle_t part = { 0, 0, 12, height };
    vx_imagepatch_addressing_t paddr = addr;
    paddr.dim_x = 12;
    vx_uint8 ones[8] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
    vx_status s1 = vxCopyImagePatch(image, &part, 0, &paddr, ones, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    vx_uint8 out[8];
    memset(out, 0xaa, sizeof(out));
    vx_status s2 = vxCopyImagePatch(image, &full, 0, &addr, out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    bool ok = (s0 == VX_SUCCESS && s1 == VX_SUCCESS && s2 == VX_SUCCESS);
    int bad = 0;
    for (vx_uint32 y = 0; y < height; y++) {
        for (vx_uint32 x = 0; x < width; x++) {
            int bit = (out[y * 2 + (x >> 3)] >> (x & 7)) & 1;
            if (bit != (x < 12 ? 1 : 0)) bad++;
        }
    }
    snprintf(detail, sizeof(detail), "%d pixel(s) wrong", bad);
    check("REQ-0938", "U1 patch ending inside a byte writes only its pixels", ok && bad == 0, detail);

    // a read of a partial patch leaves the bits of the user's byte beyond the patch alone
    vx_rectangle_t rpart = { 0, 0, 12, 1 };
    vx_imagepatch_addressing_t raddr = addr;
    raddr.dim_x = 12; raddr.dim_y = 1;
    vx_uint8 rout[2] = { 0x00, 0xa0 };
    vx_status s3 = vxCopyImagePatch(image, &rpart, 0, &raddr, rout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    check("REQ-0938", "U1 patch read keeps the user's bits outside the patch",
          s3 == VX_SUCCESS && rout[0] == 0xff && rout[1] == 0xaf, NULL);
    vxReleaseImage(&image);
}

// REQ-1463: a virtual object array, and the items taken from it, cannot be accessed from outside
// the graph. The array still has to be usable as a replicated node's data inside it.
static void test_virtual_object_array(vx_context context)
{
    char detail[96];
    vx_graph graph = vxCreateGraph(context);
    vx_image exemplar = vxCreateImage(context, 32, 32, VX_DF_IMAGE_U8);
    vx_object_array varr = vxCreateVirtualObjectArray(graph, (vx_reference)exemplar, 2);
    vx_image item = (vx_image)vxGetObjectArrayItem(varr, 0);

    vx_uint8 pixels[32 * 32];
    memset(pixels, 0, sizeof(pixels));
    vx_rectangle_t rect = { 0, 0, 32, 32 };
    vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
    addr.dim_x = 32; addr.dim_y = 32; addr.stride_x = 1; addr.stride_y = 32;
    vx_status scopy = vxCopyImagePatch(item, &rect, 0, &addr, pixels, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    check("REQ-1463", "virtual object array item rejects vxCopyImagePatch",
          varr != 0 && item != 0 && scopy != VX_SUCCESS, NULL);

    // input array -> replicated Box3x3 -> virtual array -> replicated Box3x3 -> output array
    vx_object_array in = vxCreateObjectArray(context, (vx_reference)exemplar, 2);
    vx_object_array out = vxCreateObjectArray(context, (vx_reference)exemplar, 2);
    vx_image in0 = (vx_image)vxGetObjectArrayItem(in, 0);
    vx_image out0 = (vx_image)vxGetObjectArrayItem(out, 0);
    vx_image mid0 = (vx_image)vxGetObjectArrayItem(varr, 0);
    vx_node n1 = vxBox3x3Node(graph, in0, mid0);
    vx_node n2 = vxBox3x3Node(graph, mid0, out0);
    vx_bool replicate[2] = { vx_true_e, vx_true_e };
    vx_status sr1 = vxReplicateNode(graph, n1, replicate, 2);
    vx_status sr2 = vxReplicateNode(graph, n2, replicate, 2);
    vx_status sverify = vxVerifyGraph(graph);
    vx_status sprocess = vxProcessGraph(graph);
    snprintf(detail, sizeof(detail), "verify=%d process=%d", (int)sverify, (int)sprocess);
    check("REQ-1463", "replicated nodes chained through a virtual object array run",
          sr1 == VX_SUCCESS && sr2 == VX_SUCCESS && sverify == VX_SUCCESS && sprocess == VX_SUCCESS, detail);

    vxReleaseNode(&n1);
    vxReleaseNode(&n2);
    vxReleaseImage(&in0);
    vxReleaseImage(&out0);
    vxReleaseImage(&mid0);
    vxReleaseImage(&item);
    vxReleaseObjectArray(&in);
    vxReleaseObjectArray(&out);
    vxReleaseObjectArray(&varr);
    vxReleaseImage(&exemplar);
    vxReleaseGraph(&graph);
}

// REQ-1378: VX_THRESHOLD_TYPE is read-only.
static void test_threshold_type_read_only(vx_context context)
{
    vx_threshold thr = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_enum type = VX_THRESHOLD_TYPE_RANGE;
    vx_status sset = vxSetThresholdAttribute(thr, VX_THRESHOLD_TYPE, &type, sizeof(type));
    vx_enum current = 0;
    vx_status sget = vxQueryThreshold(thr, VX_THRESHOLD_TYPE, &current, sizeof(current));
    check("REQ-1378", "vxSetThresholdAttribute rejects VX_THRESHOLD_TYPE",
          sset != VX_SUCCESS && sget == VX_SUCCESS && current == VX_THRESHOLD_TYPE_BINARY, NULL);
    vxReleaseThreshold(&thr);
}

// REQ-0938: the bit offsets of the pixels are preserved on both sides of a U1 copy, so a patch
// that starts inside a byte is read from / written to the same bit positions of the user's buffer.
static void test_u1_unaligned_patch(vx_context context)
{
    const vx_uint32 width = 16, height = 2;
    vx_image image = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
    vx_rectangle_t full = { 0, 0, width, height };
    vx_imagepatch_addressing_t faddr = VX_IMAGEPATCH_ADDR_INIT;
    faddr.dim_x = width; faddr.dim_y = height;
    faddr.stride_x = 0; faddr.stride_x_bits = 1; faddr.stride_y = 2;
    vx_uint8 zeros[4] = { 0 };
    vx_status s0 = vxCopyImagePatch(image, &full, 0, &faddr, zeros, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    // pixels 5..10 of row 0 cover the top three bits of byte 0 and the low three bits of byte 1
    vx_rectangle_t rect = { 5, 0, 11, 1 };
    vx_imagepatch_addressing_t addr = faddr;
    addr.dim_x = 6; addr.dim_y = 1;
    vx_uint8 user[2] = { 0xff, 0xff }; // bits outside the patch are not written to the image
    vx_status s1 = vxCopyImagePatch(image, &rect, 0, &addr, user, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vx_uint8 whole[4] = { 0 };
    vx_status s2 = vxCopyImagePatch(image, &full, 0, &faddr, whole, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    check("REQ-0938", "U1 patch starting inside a byte is written at its bit offsets",
          s0 == VX_SUCCESS && s1 == VX_SUCCESS && s2 == VX_SUCCESS &&
          whole[0] == 0xe0 && whole[1] == 0x07 && whole[2] == 0 && whole[3] == 0, NULL);

    // a one pixel patch at x = 5 reads back to bit 5 of the user's byte, other bits untouched
    vx_rectangle_t one = { 5, 0, 6, 1 };
    vx_imagepatch_addressing_t oaddr = faddr;
    oaddr.dim_x = 1; oaddr.dim_y = 1;
    vx_uint8 out = 0x01;
    vx_status s3 = vxCopyImagePatch(image, &one, 0, &oaddr, &out, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    check("REQ-0938", "U1 patch starting inside a byte is read to its bit offsets",
          s3 == VX_SUCCESS && out == 0x21, NULL);

    // mapping widens dim_x down to the byte boundary, and the addressing helpers agree with it
    vx_map_id map_id = 0;
    vx_imagepatch_addressing_t maddr = VX_IMAGEPATCH_ADDR_INIT;
    void *base = NULL;
    vx_status s4 = vxMapImagePatch(image, &rect, 0, &map_id, &maddr, &base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
    bool ok = (s4 == VX_SUCCESS && base != NULL && maddr.dim_x == 6 + 5);
    if (ok) {
        // the patch starts at bit 5 of the first byte, so pixel x = 8 of the widened layout is
        // in the second byte of the row
        ok = (vxFormatImagePatchAddress2d(base, 8, 0, &maddr) == (vx_uint8 *)base + 1);
        vxUnmapImagePatch(image, map_id);
    }
    check("REQ-0938", "U1 map widens dim_x and addresses by bit offset", ok, NULL);
    vxReleaseImage(&image);
}

// REQ-1760: a meta format queried from a parameter whose declared type is the generic
// VX_TYPE_REFERENCE has to be releasable like any other.
static void test_reference_parameter_meta_format(vx_context context)
{
    vx_enum kernel_id = 0;
    vxAllocateUserKernelId(context, &kernel_id);
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.test.reqtags_refmeta",
                                       kernel_id, noop_kernel, 1, NULL, NULL, NULL);
    vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_REFERENCE, VX_PARAMETER_STATE_REQUIRED);
    vxFinalizeKernel(kernel);

    vx_parameter parameter = vxGetKernelParameterByIndex(kernel, 0);
    vx_meta_format meta = 0;
    vx_status squery = vxQueryParameter(parameter, VX_PARAMETER_META_FORMAT, &meta, sizeof(meta));
    vx_reference meta_ref = (vx_reference)meta;
    vx_status srelease = vxReleaseReference(&meta_ref);
    check("REQ-1760", "VX_TYPE_REFERENCE parameter meta format is releasable",
          squery == VX_SUCCESS && meta != 0 && srelease == VX_SUCCESS && meta_ref == 0, NULL);
    vxReleaseParameter(&parameter);
    vxRemoveKernel(kernel);
}

// vxCopyRemapPatch's stride is in bytes and may be padded to something that is not a multiple
// of the element size.
static void test_remap_padded_stride(vx_context context)
{
    vx_remap map = vxCreateRemap(context, 64, 64, 32, 32);
    // a one column, two row patch whose rows are 12 bytes apart
    vx_rectangle_t rect = { 3, 4, 4, 6 };
    vx_uint8 buffer[24];
    memset(buffer, 0, sizeof(buffer));
    vx_coordinates2df_t row0 = { 10.5f, 20.25f }, row1 = { 30.75f, 40.5f };
    memcpy(buffer, &row0, sizeof(row0));
    memcpy(buffer + 12, &row1, sizeof(row1));
    vx_status swrite = vxCopyRemapPatch(map, &rect, 12, buffer, VX_TYPE_COORDINATES2DF,
                                        VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    vx_uint8 back[24];
    memset(back, 0, sizeof(back));
    vx_status sread = vxCopyRemapPatch(map, &rect, 12, back, VX_TYPE_COORDINATES2DF,
                                       VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    bool ok = (swrite == VX_SUCCESS && sread == VX_SUCCESS &&
               memcmp(back, &row0, sizeof(row0)) == 0 && memcmp(back + 12, &row1, sizeof(row1)) == 0);

    // the elements really landed on the right rows of the remap
    vx_rectangle_t second = { 3, 5, 4, 6 };
    vx_coordinates2df_t got = { 0.0f, 0.0f };
    vx_status sget = vxCopyRemapPatch(map, &second, sizeof(got), &got, VX_TYPE_COORDINATES2DF,
                                      VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    check("REQ-1269", "vxCopyRemapPatch honours a byte stride that is not a multiple of 8",
          ok && sget == VX_SUCCESS && got.x == row1.x && got.y == row1.y, NULL);
    vxReleaseRemap(&map);
}

// The true/false output values of a threshold are only honoured on the CPU, so changing them on
// a threshold used by an already verified graph has to be picked up by the next execution.
static void test_threshold_output_after_verify(vx_context context)
{
    char detail[96];
    vx_graph graph = vxCreateGraph(context);
    vx_image input = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8);
    vx_image output = vxCreateImage(context, 16, 16, VX_DF_IMAGE_U8);
    vx_threshold thr = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_pixel_value_t value;
    memset(&value, 0, sizeof(value));
    value.U8 = 100;
    vxCopyThresholdValue(thr, &value, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    vx_uint8 pixels[16 * 16];
    for (int i = 0; i < 16 * 16; i++) pixels[i] = (vx_uint8)((i & 1) ? 200 : 50);
    vx_rectangle_t rect = { 0, 0, 16, 16 };
    vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
    addr.dim_x = 16; addr.dim_y = 16; addr.stride_x = 1; addr.stride_y = 16;
    vxCopyImagePatch(input, &rect, 0, &addr, pixels, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    vx_node node = vxThresholdNode(graph, input, thr, output);
    vx_status sverify = vxVerifyGraph(graph);
    vx_status sprocess = vxProcessGraph(graph);

    // now ask for different true/false values and run the same graph again
    vx_pixel_value_t tv, fv;
    memset(&tv, 0, sizeof(tv)); memset(&fv, 0, sizeof(fv));
    tv.U8 = 7; fv.U8 = 3;
    vx_status sset = vxCopyThresholdOutput(thr, &tv, &fv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vx_status sprocess2 = vxProcessGraph(graph);

    vx_uint8 result[16 * 16];
    memset(result, 0, sizeof(result));
    vx_status sread = vxCopyImagePatch(output, &rect, 0, &addr, result, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    int bad = 0;
    for (int i = 0; i < 16 * 16; i++) {
        if (result[i] != ((i & 1) ? 7 : 3)) bad++;
    }
    snprintf(detail, sizeof(detail), "%d pixel(s) wrong", bad);
    check("REQ-0490", "threshold true/false change after verify takes effect",
          sverify == VX_SUCCESS && sprocess == VX_SUCCESS && sset == VX_SUCCESS &&
          sprocess2 == VX_SUCCESS && sread == VX_SUCCESS && bad == 0, detail);

    vxReleaseNode(&node);
    vxReleaseThreshold(&thr);
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    vxReleaseGraph(&graph);
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
    test_replication_over_virtual_pyramids(context);
    test_non_linear_filter(context);
    test_remap(context);
    test_laplacian_reconstruct(context);
    test_warp_affine_matrix(context);
    test_array_full(context);
    test_add_parameter_bounds(context);
    test_parameter_meta_format(context);
    test_remap_patch_roundtrip(context);
    test_remap_map_patch(context);
    test_node_parameter_meta_format_ownership(context);
    test_u1_partial_byte_patch(context);
    test_virtual_object_array(context);
    test_threshold_type_read_only(context);
    test_u1_unaligned_patch(context);
    test_reference_parameter_meta_format(context);
    test_remap_padded_stride(context);
    test_threshold_output_after_verify(context);

    printf("\n%s: %d failure(s)\n", errors ? "FAILED" : "PASSED", errors);
    vxReleaseContext(&context);
    return errors ? 1 : 0;
}
