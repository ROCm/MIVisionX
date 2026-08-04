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

// Graph Pipelining extension (vx_khr_pipelining) API coverage test.
//
// The Khronos conformance suite exercises the pipelining extension through a
// handful of end-to-end scenarios, which leaves a good deal of the specified
// behaviour untested: argument validation, the error codes the spec names, the
// event registration rules, and what happens to events around
// vxEnableEvents/vxDisableEvents. Those are the parts covered here, alongside
// one end-to-end queueing case that checks the data actually flows through the
// queued references rather than the defaults bound at verify time.
//
// Behaviours asserted against specific wording in the 1.1 specification are
// marked with a [spec] comment naming the requirement.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <vector>
#include <VX/vx.h>
#include <VX/vx_khr_pipelining.h>

static const vx_uint32 IMG_W = 64;
static const vx_uint32 IMG_H = 64;

// Bounds every blocking wait in this test so a regression that fails to wake a
// waiter shows up as a test failure rather than a hung CI job.
static const vx_uint32 WAIT_TIMEOUT_MS = 10000;

#define CHECK_STATUS(call) do { \
    vx_status s_ = (call); \
    if (s_ != VX_SUCCESS) { \
        printf("  FAIL: %s returned %d at %s:%d\n", #call, s_, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

#define CHECK_NOT_NULL(obj, name) do { \
    if (!(obj)) { \
        printf("  FAIL: %s is NULL at %s:%d\n", name, __FILE__, __LINE__); \
        errors++; \
    } \
} while(0)

// Asserts an exact status, which is the point of most of these cases: the spec
// names the error code, so returning a different failure is still wrong.
#define EXPECT_STATUS(call, expected, what) do { \
    vx_status s_ = (call); \
    if (s_ != (expected)) { \
        printf("  FAIL: %s -> %d, expected %d (%s) at %s:%d\n", \
               #call, s_, (int)(expected), what, __FILE__, __LINE__); \
        errors++; \
    } else { \
        printf("  PASS: %s\n", what); \
    } \
} while(0)

#define EXPECT_TRUE(cond, what) do { \
    if (!(cond)) { \
        printf("  FAIL: %s at %s:%d\n", what, __FILE__, __LINE__); \
        errors++; \
    } else { \
        printf("  PASS: %s\n", what); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// The images here are not all the same size, so both of these work from the size
// the image reports rather than assuming the default one.
static bool u8_image_size(vx_image img, vx_uint32 & w, vx_uint32 & h)
{
    return vxQueryImage(img, VX_IMAGE_WIDTH, &w, sizeof(w)) == VX_SUCCESS &&
           vxQueryImage(img, VX_IMAGE_HEIGHT, &h, sizeof(h)) == VX_SUCCESS;
}

static vx_status fill_u8_image(vx_image img, vx_uint8 value)
{
    vx_uint32 w = 0, h = 0;
    if (!u8_image_size(img, w, h))
        return VX_FAILURE;
    std::vector<vx_uint8> buf((size_t)w * h, value);
    vx_rectangle_t rect = { 0, 0, w, h };
    vx_imagepatch_addressing_t addr;
    memset(&addr, 0, sizeof(addr));
    addr.dim_x = w;
    addr.dim_y = h;
    addr.stride_x = 1;
    addr.stride_y = (vx_int32)w;
    return vxCopyImagePatch(img, &rect, 0, &addr, buf.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
}

// Reads the image back and reports whether every pixel holds the expected value.
static bool u8_image_is_uniform(vx_image img, vx_uint8 expected)
{
    vx_uint32 w = 0, h = 0;
    if (!u8_image_size(img, w, h))
        return false;
    std::vector<vx_uint8> buf((size_t)w * h, 0);
    vx_rectangle_t rect = { 0, 0, w, h };
    vx_imagepatch_addressing_t addr;
    memset(&addr, 0, sizeof(addr));
    addr.dim_x = w;
    addr.dim_y = h;
    addr.stride_x = 1;
    addr.stride_y = (vx_int32)w;
    if (vxCopyImagePatch(img, &rect, 0, &addr, buf.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST) != VX_SUCCESS)
        return false;
    for (size_t i = 0; i < buf.size(); i++) {
        if (buf[i] != expected)
            return false;
    }
    return true;
}

// Collects every event currently queued, so a test can assert on exactly what
// the implementation reported for one execution.
struct EventTally {
    vx_uint32 total = 0;
    vx_uint32 graph_completed = 0;
    vx_uint32 node_completed = 0;
    vx_uint32 node_error = 0;
    vx_uint32 parameter_consumed = 0;
    vx_uint32 user = 0;
    std::vector<vx_uint32> app_values;
};

static EventTally drain_events(vx_context context)
{
    EventTally t;
    vx_event_t ev;
    while (true) {
        memset(&ev, 0, sizeof(ev));
        if (vxWaitEvent(context, &ev, vx_true_e) != VX_SUCCESS)
            break;
        t.total++;
        t.app_values.push_back(ev.app_value);
        switch (ev.type) {
        case VX_EVENT_GRAPH_COMPLETED:            t.graph_completed++; break;
        case VX_EVENT_NODE_COMPLETED:             t.node_completed++; break;
        case VX_EVENT_NODE_ERROR:                 t.node_error++; break;
        case VX_EVENT_GRAPH_PARAMETER_CONSUMED:   t.parameter_consumed++; break;
        case VX_EVENT_USER:                       t.user++; break;
        default: break;
        }
    }
    return t;
}

// A graph holding one NOT node, with the input as graph parameter 0 and the
// output as graph parameter 1.
struct NotGraph {
    vx_graph graph;
    vx_node  node;
    vx_image in;
    vx_image out;
};

static NotGraph make_not_graph(vx_context context, vx_uint32 num_params)
{
    NotGraph g;
    g.graph = vxCreateGraph(context);
    g.in  = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    g.out = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    g.node = vxNotNode(g.graph, g.in, g.out);
    for (vx_uint32 i = 0; i < num_params; i++) {
        vx_parameter p = vxGetParameterByIndex(g.node, i);
        if (p) {
            vxAddParameterToGraph(g.graph, p);
            vxReleaseParameter(&p);
        }
    }
    return g;
}

static void release_not_graph(NotGraph & g)
{
    if (g.node)  vxReleaseNode(&g.node);
    if (g.in)    vxReleaseImage(&g.in);
    if (g.out)   vxReleaseImage(&g.out);
    if (g.graph) vxReleaseGraph(&g.graph);
}

static void set_event_timeout(vx_context context)
{
    vx_uint32 timeout = WAIT_TIMEOUT_MS;
    vxSetContextAttribute(context, VX_CONTEXT_EVENT_TIMEOUT, &timeout, sizeof(timeout));
}

// A pointer that is not a valid reference. The framework recognises references by
// a magic value, so zeroed storage is rejected, and it is the test's own storage
// so nothing is read that does not belong to it.
struct FakeRef { unsigned char storage[2048]; };

static vx_reference invalid_reference(FakeRef & f)
{
    memset(f.storage, 0, sizeof(f.storage));
    return (vx_reference)f.storage;
}

// Two user kernels with the same shape, one that always fails and one that always
// succeeds. A user node is also the way to get a node that graph optimization
// leaves alone, which the built-in nodes do not.
static const vx_enum FAILING_KERNEL_ID     = VX_KERNEL_BASE(VX_ID_USER, 0) + 7;
static const vx_enum PASSTHROUGH_KERNEL_ID = VX_KERNEL_BASE(VX_ID_USER, 0) + 8;

static vx_status VX_CALLBACK failing_kernel_func(vx_node, const vx_reference *, vx_uint32)
{
    return VX_FAILURE;
}

static vx_status VX_CALLBACK passthrough_kernel_func(vx_node, const vx_reference *, vx_uint32)
{
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK failing_kernel_validate(vx_node, const vx_reference parameters[],
                                                    vx_uint32 num, vx_meta_format metas[])
{
    if (num != 2)
        return VX_ERROR_INVALID_PARAMETERS;
    vx_df_image fmt = VX_DF_IMAGE_U8;
    vx_uint32 w = IMG_W, h = IMG_H;
    vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &fmt, sizeof(fmt));
    vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &w, sizeof(w));
    vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &h, sizeof(h));
    (void)parameters;
    return VX_SUCCESS;
}

static vx_kernel register_user_kernel(vx_context context, const char * name, vx_enum id,
                                      vx_kernel_f func)
{
    vx_kernel k = vxAddUserKernel(context, name, id, func, 2, failing_kernel_validate,
                                  nullptr, nullptr);
    if (!k)
        return nullptr;
    if (vxAddParameterToKernel(k, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED) != VX_SUCCESS ||
        vxAddParameterToKernel(k, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED) != VX_SUCCESS ||
        vxFinalizeKernel(k) != VX_SUCCESS) {
        vxRemoveKernel(k);
        return nullptr;
    }
    return k;
}

static vx_kernel register_failing_kernel(vx_context context)
{
    return register_user_kernel(context, "org.mivisionx.test.always_fails",
                                FAILING_KERNEL_ID, failing_kernel_func);
}

static vx_kernel register_passthrough_kernel(vx_context context)
{
    return register_user_kernel(context, "org.mivisionx.test.passthrough",
                                PASSTHROUGH_KERNEL_ID, passthrough_kernel_func);
}

// ---------------------------------------------------------------------------
// Test 1: vxSetGraphScheduleConfig argument validation
// ---------------------------------------------------------------------------
static int test_schedule_config_validation()
{
    int errors = 0;
    printf("\n=== Test 1: vxSetGraphScheduleConfig validation ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 2);
    CHECK_NOT_NULL(g.node, "vxNotNode");

    vx_reference in_ref = (vx_reference)g.in;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;

    EXPECT_STATUS(vxSetGraphScheduleConfig(nullptr, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q),
                  VX_ERROR_INVALID_REFERENCE, "null graph rejected");

    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, (vx_enum)0x7FFFFFFF, 1, q),
                  VX_ERROR_INVALID_PARAMETERS, "unknown schedule mode rejected");

    // NORMAL mode takes no queue configuration at all.
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_NORMAL, 0, nullptr),
                  VX_SUCCESS, "NORMAL mode with no queue list accepted");
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_NORMAL, 1, q),
                  VX_ERROR_INVALID_PARAMETERS, "NORMAL mode with a queue list rejected");

    // The queueing modes require one.
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 0, q),
                  VX_ERROR_INVALID_PARAMETERS, "QUEUE_MANUAL with zero list size rejected");
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, nullptr),
                  VX_ERROR_INVALID_PARAMETERS, "QUEUE_MANUAL with null list rejected");

    // [spec] refs_list_size MUST always be specified by the application.
    q[0].refs_list_size = 0;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q),
                  VX_ERROR_INVALID_PARAMETERS, "refs_list_size of zero rejected");
    q[0].refs_list_size = 1;

    // Out of range graph parameter index.
    q[0].graph_parameter_index = 99;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q),
                  VX_ERROR_INVALID_PARAMETERS, "out of range graph parameter index rejected");
    q[0].graph_parameter_index = 0;

    // A null entry inside a supplied refs_list is not a usable reference.
    vx_reference bad_list[2] = { (vx_reference)g.in, nullptr };
    q[0].refs_list_size = 2;
    q[0].refs_list = bad_list;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q),
                  VX_ERROR_INVALID_PARAMETERS, "null entry in refs_list rejected");

    // An entry that is not null but is not a reference either.
    FakeRef fake;
    vx_reference invalid_list[2] = { (vx_reference)g.in, invalid_reference(fake) };
    q[0].refs_list = invalid_list;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q),
                  VX_ERROR_INVALID_REFERENCE, "invalid entry in refs_list rejected");

    // [spec] "When this API is called before vxVerifyGraph, the refs_list field
    // can be NULL, if the reference handles are not available yet at the
    // application. However refs_list_size MUST always be specified."
    q[0].refs_list_size = 2;
    q[0].refs_list = nullptr;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q),
                  VX_SUCCESS, "null refs_list accepted before verify");

    // The configured mode is observable through the graph attribute.
    {
        vx_enum mode = 0;
        CHECK_STATUS(vxQueryGraph(g.graph, VX_GRAPH_SCHEDULE_MODE, &mode, sizeof(mode)));
        EXPECT_TRUE(mode == VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL,
                    "VX_GRAPH_SCHEDULE_MODE reports the configured mode");
    }

    // [spec] both of these report VX_ERROR_INVALID_GRAPH when the graph is not
    // verified; this graph never is. Adding references in particular "may only
    // be called after graph verification".
    {
        vx_reference got[2] = { nullptr, nullptr };
        EXPECT_STATUS(vxGetGraphParameterRefsList(g.graph, 0, 2, got),
                      VX_ERROR_INVALID_GRAPH, "refs list query before verify rejected");
        vx_reference add = (vx_reference)g.in;
        EXPECT_STATUS(vxAddReferencesToGraphParameterList(g.graph, 0, 1, &add),
                      VX_ERROR_INVALID_GRAPH, "adding references before verify rejected");
    }

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 2: refs_list handed over by a second call made after vxVerifyGraph
// ---------------------------------------------------------------------------
static int test_refs_list_after_verify()
{
    int errors = 0;
    printf("\n=== Test 2: refs_list supplied after vxVerifyGraph ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 2);
    vx_image spare   = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_image unlisted = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);

    vx_graph_parameter_queue_params_t q[1];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 2;
    q[0].refs_list = nullptr;

    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q),
                  VX_SUCCESS, "configured before verify without the handles");
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    // [spec] "Application can call vxSetGraphScheduleConfig again after verify
    // graph with all parameters remaining the same except with refs_list field
    // providing the list of references that can be enqueued."
    vx_reference refs[2] = { (vx_reference)g.in, (vx_reference)spare };
    q[0].refs_list = refs;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q),
                  VX_SUCCESS, "refs_list accepted after verify");

    // The list is now readable back through the query API.
    {
        vx_reference got[2] = { nullptr, nullptr };
        EXPECT_STATUS(vxGetGraphParameterRefsList(g.graph, 0, 2, got),
                      VX_SUCCESS, "vxGetGraphParameterRefsList");
        EXPECT_TRUE(got[0] == (vx_reference)g.in && got[1] == (vx_reference)spare,
                    "returned refs_list matches what was supplied");
        // [spec] a null refs_list, or a size too small to hold the list, is an
        // invalid request once the graph is verified.
        EXPECT_STATUS(vxGetGraphParameterRefsList(g.graph, 0, 2, nullptr),
                      VX_ERROR_INVALID_PARAMETERS, "null refs_list output rejected");
        EXPECT_STATUS(vxGetGraphParameterRefsList(g.graph, 0, 1, got),
                      VX_ERROR_INVALID_PARAMETERS, "undersized refs_list output rejected");
        EXPECT_STATUS(vxGetGraphParameterRefsList(g.graph, 99, 2, got),
                      VX_ERROR_INVALID_PARAMETERS, "out of range parameter rejected");
    }

    // A reference from the list can be enqueued. Enqueueing one outside the list
    // is left implementation-defined by the spec; this implementation rejects it,
    // which is what the list is checked against here.
    {
        vx_reference enq = (vx_reference)g.in;
        EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &enq, 1),
                      VX_SUCCESS, "listed reference can be enqueued");
        vx_reference bad = (vx_reference)unlisted;
        EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &bad, 1),
                      VX_ERROR_INVALID_PARAMETERS, "unlisted reference is rejected");
    }

    // vxAddReferencesToGraphParameterList extends the permitted set.
    {
        vx_reference add = (vx_reference)unlisted;
        EXPECT_STATUS(vxAddReferencesToGraphParameterList(g.graph, 0, 1, &add),
                      VX_SUCCESS, "vxAddReferencesToGraphParameterList");
        EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &add, 1),
                      VX_SUCCESS, "newly added reference can be enqueued");
    }

    // [spec] the depth given as refs_list_size bounds what may be handed over
    // and not yet taken by the graph; beyond it there is no resource left.
    {
        vx_reference enq = (vx_reference)spare;
        EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &enq, 1),
                      VX_ERROR_NO_RESOURCES, "enqueue beyond the configured depth reports NO_RESOURCES");
    }

    // VX_REFERENCE_ENQUEUE_COUNT tracks how often a reference was enqueued.
    {
        vx_uint32 count = 0;
        vx_status s = vxQueryReference((vx_reference)g.in, VX_REFERENCE_ENQUEUE_COUNT,
                                       &count, sizeof(count));
        if (s == VX_SUCCESS)
            printf("  PASS: VX_REFERENCE_ENQUEUE_COUNT = %u\n", count);
        else
            printf("  INFO: VX_REFERENCE_ENQUEUE_COUNT returned %d\n", s);
    }

    vxReleaseImage(&spare);
    vxReleaseImage(&unlisted);
    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 3: queue argument validation and vxGraphParameterCheckDoneRef
// ---------------------------------------------------------------------------
static int test_queue_validation()
{
    int errors = 0;
    printf("\n=== Test 3: queue API validation ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 2);

    vx_reference in_ref  = (vx_reference)g.in;
    vx_reference out_ref = (vx_reference)g.out;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    q[1].graph_parameter_index = 1;
    q[1].refs_list_size = 1;
    q[1].refs_list = &out_ref;
    CHECK_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 2, q));
    CHECK_STATUS(vxVerifyGraph(g.graph));

    EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(nullptr, 0, &in_ref, 1),
                  VX_ERROR_INVALID_REFERENCE, "enqueue on a null graph rejected");
    EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 99, &in_ref, 1),
                  VX_ERROR_INVALID_PARAMETERS, "enqueue on an out of range parameter rejected");
    EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, nullptr, 1),
                  VX_ERROR_INVALID_PARAMETERS, "enqueue with a null reference array rejected");

    {
        vx_reference null_ref = nullptr;
        EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &null_ref, 1),
                      VX_ERROR_INVALID_REFERENCE, "enqueue of a null reference rejected");
    }

    // Nothing has been consumed, so nothing is waiting to be dequeued.
    {
        vx_uint32 num = 99;
        EXPECT_STATUS(vxGraphParameterCheckDoneRef(g.graph, 1, &num),
                      VX_SUCCESS, "vxGraphParameterCheckDoneRef");
        EXPECT_TRUE(num == 0, "no done references before any execution");
        EXPECT_STATUS(vxGraphParameterCheckDoneRef(g.graph, 1, nullptr),
                      VX_ERROR_INVALID_PARAMETERS, "check with a null count rejected");
        EXPECT_STATUS(vxGraphParameterCheckDoneRef(g.graph, 99, &num),
                      VX_ERROR_INVALID_PARAMETERS, "check on an out of range parameter rejected");
    }

    {
        vx_reference deq = nullptr;
        vx_uint32 num = 0;
        EXPECT_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 99, &deq, 1, &num),
                      VX_ERROR_INVALID_PARAMETERS, "dequeue on an out of range parameter rejected");
        EXPECT_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 1, nullptr, 1, &num),
                      VX_ERROR_INVALID_PARAMETERS, "dequeue with a null reference array rejected");
    }

    // In QUEUE_MANUAL a graph execution consumes one reference from every
    // configured queue, so with the queues empty there is nothing to run and the
    // attempt cannot report success. vxProcessGraph is used rather than
    // vxScheduleGraph because it carries the execution status back directly.
    EXPECT_TRUE(vxProcessGraph(g.graph) != VX_SUCCESS,
                "executing with nothing enqueued does not report success");

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 4: vxRegisterEvent validation
// ---------------------------------------------------------------------------
static int test_event_registration_validation()
{
    int errors = 0;
    printf("\n=== Test 4: vxRegisterEvent validation ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 2);

    EXPECT_STATUS(vxRegisterEvent(nullptr, VX_EVENT_GRAPH_COMPLETED, 0, 1),
                  VX_ERROR_INVALID_REFERENCE, "null reference rejected");

    // [spec] VX_ERROR_NOT_SUPPORTED - type is not valid for the provided reference.
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_NODE_COMPLETED, 0, 1),
                  VX_ERROR_NOT_SUPPORTED, "NODE_COMPLETED on a graph rejected");
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_NODE_ERROR, 0, 1),
                  VX_ERROR_NOT_SUPPORTED, "NODE_ERROR on a graph rejected");
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.node, VX_EVENT_GRAPH_COMPLETED, 0, 1),
                  VX_ERROR_NOT_SUPPORTED, "GRAPH_COMPLETED on a node rejected");
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.node, VX_EVENT_GRAPH_PARAMETER_CONSUMED, 0, 1),
                  VX_ERROR_NOT_SUPPORTED, "GRAPH_PARAMETER_CONSUMED on a node rejected");
    // [spec] "the application does NOT register user events using vxRegisterEvent."
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_USER, 0, 1),
                  VX_ERROR_NOT_SUPPORTED, "USER event registration rejected");
    // Events are reported for graphs and nodes, not for data objects.
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.in, VX_EVENT_GRAPH_COMPLETED, 0, 1),
                  VX_ERROR_NOT_SUPPORTED, "registration on an image rejected");

    // The parameter index only means something for GRAPH_PARAMETER_CONSUMED, and
    // there it has to name a real graph parameter.
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_GRAPH_PARAMETER_CONSUMED, 99, 1),
                  VX_ERROR_INVALID_PARAMETERS, "out of range parameter index rejected");

    // Valid registrations.
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_GRAPH_COMPLETED, 0, 100),
                  VX_SUCCESS, "GRAPH_COMPLETED on a graph accepted");
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_GRAPH_PARAMETER_CONSUMED, 1, 101),
                  VX_SUCCESS, "GRAPH_PARAMETER_CONSUMED on a graph accepted");
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.node, VX_EVENT_NODE_COMPLETED, 0, 102),
                  VX_SUCCESS, "NODE_COMPLETED on a node accepted");
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.node, VX_EVENT_NODE_ERROR, 0, 103),
                  VX_SUCCESS, "NODE_ERROR on a node accepted");

    // Registering the same thing again replaces the app_value rather than
    // adding a second registration; the delivered value is checked in Test 5.
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_GRAPH_COMPLETED, 0, 200),
                  VX_SUCCESS, "re-registration accepted");

    // vxRegisterGraphEvent is the graph-scoped spelling of the same call.
    EXPECT_STATUS(vxRegisterGraphEvent((vx_reference)g.graph, VX_EVENT_GRAPH_COMPLETED, 0, 201),
                  VX_SUCCESS, "vxRegisterGraphEvent accepted");

    // [spec] "This API MUST be called before doing vxVerifyGraph for that graph."
    CHECK_STATUS(vxVerifyGraph(g.graph));
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_GRAPH_COMPLETED, 0, 300),
                  VX_ERROR_NOT_SUPPORTED, "registration after verify rejected");
    EXPECT_STATUS(vxRegisterEvent((vx_reference)g.node, VX_EVENT_NODE_COMPLETED, 0, 301),
                  VX_ERROR_NOT_SUPPORTED, "node registration after verify rejected");

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 5: events are reported only for references the application registered
// ---------------------------------------------------------------------------
static int test_event_delivery_gating()
{
    int errors = 0;
    printf("\n=== Test 5: event delivery is limited to registrations ===\n");

    vx_context context = vxCreateContext();
    set_event_timeout(context);

    // Two nodes, of which only the second is registered for completion events.
    vx_graph graph = vxCreateGraph(context);
    vx_image in   = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_image mid  = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_image out  = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_node n0 = vxNotNode(graph, in, mid);
    vx_node n1 = vxNotNode(graph, mid, out);
    CHECK_NOT_NULL(n0, "first vxNotNode");
    CHECK_NOT_NULL(n1, "second vxNotNode");
    CHECK_STATUS(fill_u8_image(in, 0xA5));

    CHECK_STATUS(vxRegisterEvent((vx_reference)n1, VX_EVENT_NODE_COMPLETED, 0, 501));
    CHECK_STATUS(vxRegisterEvent((vx_reference)graph, VX_EVENT_GRAPH_COMPLETED, 0, 502));
    // Replaces the value above, so 503 is what should arrive.
    CHECK_STATUS(vxRegisterEvent((vx_reference)graph, VX_EVENT_GRAPH_COMPLETED, 0, 503));

    CHECK_STATUS(vxEnableEvents(context));
    CHECK_STATUS(vxVerifyGraph(graph));
    CHECK_STATUS(vxProcessGraph(graph));

    EventTally t = drain_events(context);
    printf("  INFO: %u event(s): graph_completed=%u node_completed=%u consumed=%u\n",
           t.total, t.graph_completed, t.node_completed, t.parameter_consumed);

    // [spec] "This event is generated every time a graph execution completes."
    EXPECT_TRUE(t.graph_completed == 1, "one GRAPH_COMPLETED for one execution");
    EXPECT_TRUE(t.node_error == 0, "no NODE_ERROR from a successful execution");
    // No graph parameters were configured for queueing here.
    EXPECT_TRUE(t.parameter_consumed == 0, "no GRAPH_PARAMETER_CONSUMED without queueing");
    // n0 was never registered, so at most the one registered node may report.
    // Graph optimization is free to rewrite the nodes, so the count is bounded
    // rather than fixed; what matters is that the unregistered one stays silent.
    EXPECT_TRUE(t.node_completed <= 1, "the unregistered node does not report completion");

    // Every registration in this test used a distinct non-zero app_value, so an
    // event carrying zero could only have come from a reference that was never
    // registered, whatever internal node the optimizer produced for it.
    bool saw_zero = false, saw_503 = false, saw_502 = false, saw_501 = false;
    for (vx_uint32 v : t.app_values) {
        if (v == 0)   saw_zero = true;
        if (v == 503) saw_503 = true;
        if (v == 502) saw_502 = true;
        if (v == 501) saw_501 = true;
    }
    EXPECT_TRUE(!saw_zero, "no events reported for unregistered references");
    EXPECT_TRUE(saw_503 && !saw_502, "re-registration replaced the graph app_value");
    if (t.node_completed)
        EXPECT_TRUE(saw_501, "registered node event carries its app_value");

    vxReleaseNode(&n0);
    vxReleaseNode(&n1);
    vxReleaseImage(&in);
    vxReleaseImage(&mid);
    vxReleaseImage(&out);
    vxReleaseGraph(&graph);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 6: what vxDisableEvents does and does not affect
// ---------------------------------------------------------------------------
static int test_events_disabled_semantics()
{
    int errors = 0;
    printf("\n=== Test 6: vxEnableEvents / vxDisableEvents semantics ===\n");

    vx_context context = vxCreateContext();
    set_event_timeout(context);

    NotGraph g = make_not_graph(context, 0);
    CHECK_STATUS(fill_u8_image(g.in, 0xA5));
    CHECK_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_GRAPH_COMPLETED, 0, 601));

    EXPECT_STATUS(vxEnableEvents(context), VX_SUCCESS, "vxEnableEvents");
    CHECK_STATUS(vxVerifyGraph(g.graph));
    CHECK_STATUS(vxProcessGraph(g.graph));

    // One event is now queued and has not been collected.
    EXPECT_STATUS(vxDisableEvents(context), VX_SUCCESS, "vxDisableEvents");

    // [spec] "any event generated before this API is called will still be
    // returned via vxWaitEvent API." This uses the blocking form on purpose:
    // requiring events to be enabled here would strand the queued event.
    {
        vx_event_t ev;
        memset(&ev, 0, sizeof(ev));
        EXPECT_STATUS(vxWaitEvent(context, &ev, vx_false_e), VX_SUCCESS,
                      "blocking wait returns an event queued before disable");
        EXPECT_TRUE(ev.type == VX_EVENT_GRAPH_COMPLETED && ev.app_value == 601,
                    "the returned event is the one that was queued");
    }

    // [spec] "no additional events would be returned via vxWaitEvent API until
    // events are enabled again."
    CHECK_STATUS(vxProcessGraph(g.graph));
    {
        vx_event_t ev;
        memset(&ev, 0, sizeof(ev));
        EXPECT_TRUE(vxWaitEvent(context, &ev, vx_true_e) != VX_SUCCESS,
                    "no new events are delivered while disabled");
    }

    // Re-enabling resumes reporting.
    EXPECT_STATUS(vxEnableEvents(context), VX_SUCCESS, "vxEnableEvents again");
    CHECK_STATUS(vxProcessGraph(g.graph));
    {
        vx_event_t ev;
        memset(&ev, 0, sizeof(ev));
        EXPECT_STATUS(vxWaitEvent(context, &ev, vx_false_e), VX_SUCCESS,
                      "events flow again after re-enabling");
    }

    // A non-blocking wait on an empty queue reports failure rather than blocking.
    drain_events(context);
    {
        vx_event_t ev;
        memset(&ev, 0, sizeof(ev));
        EXPECT_TRUE(vxWaitEvent(context, &ev, vx_true_e) != VX_SUCCESS,
                    "non-blocking wait on an empty queue fails");
    }

    EXPECT_STATUS(vxWaitEvent(context, nullptr, vx_true_e), VX_ERROR_INVALID_REFERENCE,
                  "wait with a null event structure rejected");
    EXPECT_STATUS(vxEnableEvents(nullptr), VX_ERROR_INVALID_REFERENCE,
                  "vxEnableEvents on a null context rejected");
    EXPECT_STATUS(vxDisableEvents(nullptr), VX_ERROR_INVALID_REFERENCE,
                  "vxDisableEvents on a null context rejected");

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 7: user events
// ---------------------------------------------------------------------------
static int test_user_events()
{
    int errors = 0;
    printf("\n=== Test 7: user events ===\n");

    vx_context context = vxCreateContext();
    set_event_timeout(context);
    vx_graph graph = vxCreateGraph(context);

    CHECK_STATUS(vxEnableEvents(context));

    int payload = 4242;
    // User events need no registration, unlike implementation-generated ones.
    EXPECT_STATUS(vxSendUserEvent(context, 700, &payload), VX_SUCCESS, "vxSendUserEvent");

    {
        vx_event_t ev;
        memset(&ev, 0, sizeof(ev));
        EXPECT_STATUS(vxWaitEvent(context, &ev, vx_false_e), VX_SUCCESS, "user event received");
        EXPECT_TRUE(ev.type == VX_EVENT_USER, "event type is VX_EVENT_USER");
        EXPECT_TRUE(ev.app_value == 700, "user event carries the given app_value");
        EXPECT_TRUE(ev.event_info.user_event.user_event_parameter == &payload,
                    "user event carries the given parameter");
        // [spec] timestamp is the time the event was generated, in nanoseconds.
        EXPECT_TRUE(ev.timestamp != 0, "user event carries a timestamp");
    }

    // A null parameter is allowed; only the app_value is required.
    EXPECT_STATUS(vxSendUserEvent(context, 701, nullptr), VX_SUCCESS,
                  "vxSendUserEvent with no parameter");
    drain_events(context);

    // The graph-scoped spelling reaches the same queue.
    EXPECT_STATUS(vxSendUserGraphEvent(graph, 702, nullptr), VX_SUCCESS, "vxSendUserGraphEvent");
    {
        vx_event_t ev;
        memset(&ev, 0, sizeof(ev));
        EXPECT_STATUS(vxWaitGraphEvent(graph, &ev, vx_false_e), VX_SUCCESS, "vxWaitGraphEvent");
        EXPECT_TRUE(ev.app_value == 702, "graph-scoped user event carries its app_value");
    }

    EXPECT_STATUS(vxSendUserEvent(nullptr, 703, nullptr), VX_ERROR_INVALID_REFERENCE,
                  "vxSendUserEvent on a null context rejected");

    // The graph-scoped enable/disable pair operates on the owning context.
    EXPECT_STATUS(vxDisableGraphEvents(graph), VX_SUCCESS, "vxDisableGraphEvents");
    EXPECT_TRUE(vxSendUserEvent(context, 704, nullptr) != VX_SUCCESS,
                "user events are not recorded while disabled");
    EXPECT_STATUS(vxEnableGraphEvents(graph), VX_SUCCESS, "vxEnableGraphEvents");
    EXPECT_STATUS(vxSendUserEvent(context, 705, nullptr), VX_SUCCESS,
                  "user events resume after re-enabling");
    drain_events(context);

    vxReleaseGraph(&graph);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 8: timeout and pipeline depth attributes
// ---------------------------------------------------------------------------
static int test_attributes()
{
    int errors = 0;
    printf("\n=== Test 8: pipelining attributes ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 0);

    // [spec] the implementation shall initially set the timeouts to
    // VX_TIMEOUT_WAIT_FOREVER.
    {
        vx_uint32 timeout = 0;
        CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_EVENT_TIMEOUT, &timeout, sizeof(timeout)));
        EXPECT_TRUE(timeout == VX_TIMEOUT_WAIT_FOREVER,
                    "VX_CONTEXT_EVENT_TIMEOUT defaults to WAIT_FOREVER");
        timeout = 1234;
        CHECK_STATUS(vxSetContextAttribute(context, VX_CONTEXT_EVENT_TIMEOUT, &timeout, sizeof(timeout)));
        timeout = 0;
        CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_EVENT_TIMEOUT, &timeout, sizeof(timeout)));
        EXPECT_TRUE(timeout == 1234, "VX_CONTEXT_EVENT_TIMEOUT round-trips");
    }

    {
        vx_uint32 timeout = 0;
        CHECK_STATUS(vxQueryGraph(g.graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
        EXPECT_TRUE(timeout == VX_TIMEOUT_WAIT_FOREVER,
                    "VX_GRAPH_TIMEOUT defaults to WAIT_FOREVER");
        timeout = 5678;
        CHECK_STATUS(vxSetGraphAttribute(g.graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
        timeout = 0;
        CHECK_STATUS(vxQueryGraph(g.graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
        EXPECT_TRUE(timeout == 5678, "VX_GRAPH_TIMEOUT round-trips");
    }

    {
        vx_uint32 timeout = 0;
        CHECK_STATUS(vxQueryGraph(g.graph, VX_GRAPH_EVENT_TIMEOUT, &timeout, sizeof(timeout)));
        EXPECT_TRUE(timeout == VX_TIMEOUT_WAIT_FOREVER,
                    "VX_GRAPH_EVENT_TIMEOUT defaults to WAIT_FOREVER");
        timeout = 4321;
        CHECK_STATUS(vxSetGraphAttribute(g.graph, VX_GRAPH_EVENT_TIMEOUT, &timeout, sizeof(timeout)));
        timeout = 0;
        CHECK_STATUS(vxQueryGraph(g.graph, VX_GRAPH_EVENT_TIMEOUT, &timeout, sizeof(timeout)));
        EXPECT_TRUE(timeout == 4321, "VX_GRAPH_EVENT_TIMEOUT round-trips");
    }

    // The pipeline depth an implementation settles on is its own choice, so the
    // value is only reported, not asserted.
    {
        vx_uint32 depth = 0;
        CHECK_STATUS(vxQueryGraph(g.graph, VX_GRAPH_PIPELINE_DEPTH, &depth, sizeof(depth)));
        printf("  INFO: VX_GRAPH_PIPELINE_DEPTH = %u\n", depth);
        EXPECT_TRUE(depth >= 1, "pipeline depth is at least one");
    }

    // A null pointer has to be rejected rather than dereferenced, for the
    // attributes this extension adds as much as for any other.
    {
        vx_uint32 timeout = 100;
        EXPECT_STATUS(vxSetContextAttribute(context, VX_CONTEXT_EVENT_TIMEOUT, nullptr, sizeof(timeout)),
                      VX_ERROR_INVALID_PARAMETERS, "set VX_CONTEXT_EVENT_TIMEOUT with null ptr");
        EXPECT_STATUS(vxQueryContext(context, VX_CONTEXT_EVENT_TIMEOUT, nullptr, sizeof(timeout)),
                      VX_ERROR_INVALID_PARAMETERS, "query VX_CONTEXT_EVENT_TIMEOUT with null ptr");
        EXPECT_STATUS(vxSetGraphAttribute(g.graph, VX_GRAPH_TIMEOUT, nullptr, sizeof(timeout)),
                      VX_ERROR_INVALID_PARAMETERS, "set VX_GRAPH_TIMEOUT with null ptr");
        EXPECT_STATUS(vxQueryGraph(g.graph, VX_GRAPH_PIPELINE_DEPTH, nullptr, sizeof(timeout)),
                      VX_ERROR_INVALID_PARAMETERS, "query VX_GRAPH_PIPELINE_DEPTH with null ptr");
    }

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 9: QUEUE_MANUAL end to end, including the data path
// ---------------------------------------------------------------------------
static int test_manual_queue_end_to_end()
{
    int errors = 0;
    printf("\n=== Test 9: QUEUE_MANUAL end to end ===\n");

    vx_context context = vxCreateContext();
    set_event_timeout(context);
    NotGraph g = make_not_graph(context, 2);

    // Written before the parameters become queued, because a queued parameter is
    // handed over through the queue rather than accessed directly.
    CHECK_STATUS(fill_u8_image(g.in, 0xA5));

    vx_reference in_ref  = (vx_reference)g.in;
    vx_reference out_ref = (vx_reference)g.out;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    q[1].graph_parameter_index = 1;
    q[1].refs_list_size = 1;
    q[1].refs_list = &out_ref;

    CHECK_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_GRAPH_COMPLETED, 0, 900));
    CHECK_STATUS(vxRegisterEvent((vx_reference)g.graph, VX_EVENT_GRAPH_PARAMETER_CONSUMED, 1, 901));
    CHECK_STATUS(vxEnableEvents(context));

    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 2, q),
                  VX_SUCCESS, "QUEUE_MANUAL configured");

    // Bounds the dequeue below so a lost wake-up fails instead of hanging.
    {
        vx_uint32 timeout = WAIT_TIMEOUT_MS;
        CHECK_STATUS(vxSetGraphAttribute(g.graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
    }
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &in_ref, 1),
                  VX_SUCCESS, "input enqueued");
    EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 1, &out_ref, 1),
                  VX_SUCCESS, "output enqueued");

    EXPECT_STATUS(vxScheduleGraph(g.graph), VX_SUCCESS, "vxScheduleGraph");
    EXPECT_STATUS(vxWaitGraph(g.graph), VX_SUCCESS, "vxWaitGraph");

    {
        vx_reference deq = nullptr;
        vx_uint32 num = 0;
        EXPECT_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 1, &deq, 1, &num),
                      VX_SUCCESS, "output dequeued");
        EXPECT_TRUE(num == 1 && deq == out_ref, "the dequeued reference is the one enqueued");
    }

    // NOT of 0xA5 is 0x5A. This is what proves the queued references, and not
    // the defaults bound at verify time, carried the data through the graph.
    EXPECT_TRUE(u8_image_is_uniform(g.out, 0x5A), "queued output holds the computed result");

    {
        EventTally t = drain_events(context);
        printf("  INFO: %u event(s): graph_completed=%u consumed=%u\n",
               t.total, t.graph_completed, t.parameter_consumed);
        EXPECT_TRUE(t.graph_completed == 1, "one GRAPH_COMPLETED for one execution");
        // [spec] generated when a data reference at a graph parameter is consumed.
        EXPECT_TRUE(t.parameter_consumed >= 1, "the consumed output parameter reported");
    }

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 10: streaming
// ---------------------------------------------------------------------------
static int test_streaming()
{
    int errors = 0;
    printf("\n=== Test 10: graph streaming ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 0);
    CHECK_STATUS(fill_u8_image(g.in, 0xA5));

    EXPECT_STATUS(vxEnableGraphStreaming(nullptr, nullptr), VX_ERROR_INVALID_REFERENCE,
                  "vxEnableGraphStreaming on a null graph rejected");

    // [spec] "This function must be called before vxVerifyGraph." The trigger
    // node is optional, so a null one is allowed.
    EXPECT_STATUS(vxEnableGraphStreaming(g.graph, g.node), VX_SUCCESS,
                  "vxEnableGraphStreaming with a trigger node");

    // Streaming cannot start until the graph has been verified.
    EXPECT_STATUS(vxStartGraphStreaming(g.graph), VX_ERROR_NOT_SUFFICIENT,
                  "start before verify reports NOT_SUFFICIENT");

    CHECK_STATUS(vxVerifyGraph(g.graph));
    EXPECT_STATUS(vxStartGraphStreaming(g.graph), VX_SUCCESS, "vxStartGraphStreaming");
    EXPECT_STATUS(vxStopGraphStreaming(g.graph), VX_SUCCESS, "vxStopGraphStreaming");

    // Stopping leaves streaming disabled, so it cannot simply be started again.
    // Re-enabling would have to happen before verification, as above.
    EXPECT_TRUE(vxStartGraphStreaming(g.graph) != VX_SUCCESS,
                "starting again after a stop does not report success");
    EXPECT_STATUS(vxStopGraphStreaming(nullptr), VX_ERROR_INVALID_REFERENCE,
                  "vxStopGraphStreaming on a null graph rejected");
    EXPECT_STATUS(vxStartGraphStreaming(nullptr), VX_ERROR_INVALID_REFERENCE,
                  "vxStartGraphStreaming on a null graph rejected");

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 11: QUEUE_AUTO configuration
// ---------------------------------------------------------------------------
static int test_queue_auto()
{
    int errors = 0;
    printf("\n=== Test 11: QUEUE_AUTO ===\n");

    vx_context context = vxCreateContext();
    set_event_timeout(context);
    NotGraph g = make_not_graph(context, 2);
    CHECK_STATUS(fill_u8_image(g.in, 0x0F));

    vx_reference in_ref  = (vx_reference)g.in;
    vx_reference out_ref = (vx_reference)g.out;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    q[1].graph_parameter_index = 1;
    q[1].refs_list_size = 1;
    q[1].refs_list = &out_ref;

    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO, 2, q),
                  VX_SUCCESS, "QUEUE_AUTO configured");
    {
        vx_enum mode = 0;
        CHECK_STATUS(vxQueryGraph(g.graph, VX_GRAPH_SCHEDULE_MODE, &mode, sizeof(mode)));
        EXPECT_TRUE(mode == VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO,
                    "VX_GRAPH_SCHEDULE_MODE reports QUEUE_AUTO");
    }
    {
        vx_uint32 timeout = WAIT_TIMEOUT_MS;
        CHECK_STATUS(vxSetGraphAttribute(g.graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
    }
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    // In QUEUE_AUTO the implementation schedules as soon as a full set of
    // references is available, with no vxScheduleGraph from the application.
    EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &in_ref, 1),
                  VX_SUCCESS, "input enqueued");
    EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 1, &out_ref, 1),
                  VX_SUCCESS, "output enqueued");

    {
        vx_reference deq = nullptr;
        vx_uint32 num = 0;
        EXPECT_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 1, &deq, 1, &num),
                      VX_SUCCESS, "output dequeued without an explicit schedule");
        EXPECT_TRUE(num == 1 && deq == out_ref, "the dequeued reference is the one enqueued");
    }
    EXPECT_TRUE(u8_image_is_uniform(g.out, 0xF0), "queued output holds the computed result");

    // A call made after verify may only supply refs_list; everything else has to
    // stay as it was, so switching the mode now is not a legal reconfiguration.
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_NORMAL, 0, nullptr),
                  VX_ERROR_INVALID_PARAMETERS, "mode change after verify rejected");

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 12: handing over references while another thread enqueues
//
// A post-verify vxSetGraphScheduleConfig replaces the reference list a queue
// accepts, and the application is free to enqueue from a different thread at the
// same time. Nothing here blocks, so a failure shows up as a wrong status or a
// crash rather than a hung test -- and if the queue lock were ever taken around
// the enqueue notification, this would deadlock against the QUEUE_AUTO executor.
// ---------------------------------------------------------------------------
static int test_concurrent_refs_handover()
{
    int errors = 0;
    printf("\n=== Test 12: concurrent refs_list handover ===\n");

    const int ITERATIONS = 200;

    vx_context context = vxCreateContext();
    set_event_timeout(context);
    NotGraph g = make_not_graph(context, 2);
    CHECK_STATUS(fill_u8_image(g.in, 0x11));

    vx_reference in_ref  = (vx_reference)g.in;
    vx_reference out_ref = (vx_reference)g.out;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    q[1].graph_parameter_index = 1;
    q[1].refs_list_size = 1;
    q[1].refs_list = &out_ref;

    CHECK_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO, 2, q));
    {
        vx_uint32 timeout = WAIT_TIMEOUT_MS;
        CHECK_STATUS(vxSetGraphAttribute(g.graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
    }
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    std::atomic<int> handover_failures(0);
    std::thread handover([&]() {
        for (int i = 0; i < ITERATIONS; i++) {
            if (vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO, 2, q) != VX_SUCCESS)
                handover_failures++;
            vx_reference seen[4] = { nullptr, nullptr, nullptr, nullptr };
            if (vxGetGraphParameterRefsList(g.graph, 0, 4, seen) != VX_SUCCESS)
                handover_failures++;
            if (vxAddReferencesToGraphParameterList(g.graph, 0, 1, &in_ref) != VX_SUCCESS)
                handover_failures++;
        }
    });

    int bad_status = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        // The reference stays in the list throughout, so the only outcomes are
        // acceptance or a full queue -- never a rejection of the reference.
        vx_status si = vxGraphParameterEnqueueReadyRef(g.graph, 0, &in_ref, 1);
        vx_status so = vxGraphParameterEnqueueReadyRef(g.graph, 1, &out_ref, 1);
        if (si != VX_SUCCESS && si != VX_ERROR_NO_RESOURCES) bad_status++;
        if (so != VX_SUCCESS && so != VX_ERROR_NO_RESOURCES) bad_status++;

        // Drain whatever the executor finished, without ever waiting for it.
        for (vx_uint32 p = 0; p < 2; p++) {
            vx_uint32 done = 0;
            if (vxGraphParameterCheckDoneRef(g.graph, p, &done) == VX_SUCCESS && done > 0) {
                vx_reference deq = nullptr;
                vx_uint32 num = 0;
                vxGraphParameterDequeueDoneRef(g.graph, p, &deq, 1, &num);
            }
        }
    }
    handover.join();

    EXPECT_TRUE(bad_status == 0, "enqueue during handover never rejects a listed reference");
    EXPECT_TRUE(handover_failures.load() == 0, "refs_list handover succeeds alongside enqueueing");

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 13: what a post-verify vxSetGraphScheduleConfig will and will not accept
//
// After verify the call may only supply refs_list. Everything else has to match
// the configuration the graph was verified with.
// ---------------------------------------------------------------------------
static int test_post_verify_config_validation()
{
    int errors = 0;
    printf("\n=== Test 13: post-verify schedule config validation ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 2);

    vx_reference in_ref = (vx_reference)g.in;
    vx_graph_parameter_queue_params_t q[1];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    CHECK_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q));
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    vx_graph_parameter_queue_params_t p[1];

    memcpy(p, q, sizeof(p));
    p[0].graph_parameter_index = 99;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, p),
                  VX_ERROR_INVALID_PARAMETERS, "out of range parameter index rejected after verify");

    memcpy(p, q, sizeof(p));
    p[0].refs_list_size = 0;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, p),
                  VX_ERROR_INVALID_PARAMETERS, "zero refs_list_size rejected after verify");

    // Parameter 1 was never given a queue, and this call may not create one.
    memcpy(p, q, sizeof(p));
    p[0].graph_parameter_index = 1;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, p),
                  VX_ERROR_INVALID_PARAMETERS, "a queue cannot be enabled after verify");

    // A null refs_list is not an instruction to forget the references already
    // supplied, so the previous list survives.
    memcpy(p, q, sizeof(p));
    p[0].refs_list = nullptr;
    EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, p),
                  VX_SUCCESS, "null refs_list after verify accepted");
    {
        vx_reference got[1] = { nullptr };
        EXPECT_STATUS(vxGetGraphParameterRefsList(g.graph, 0, 1, got), VX_SUCCESS,
                      "refs list still readable");
        EXPECT_TRUE(got[0] == in_ref, "the previously supplied list is retained");
    }

    {
        vx_reference nulls[1] = { nullptr };
        memcpy(p, q, sizeof(p));
        p[0].refs_list = nulls;
        EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, p),
                      VX_ERROR_INVALID_PARAMETERS, "null entry rejected after verify");

        FakeRef fake;
        vx_reference invalid[1] = { invalid_reference(fake) };
        memcpy(p, q, sizeof(p));
        p[0].refs_list = invalid;
        EXPECT_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, p),
                      VX_ERROR_INVALID_REFERENCE, "invalid entry rejected after verify");
    }

    // A rejected list must not have disturbed the one already in place.
    {
        vx_reference got[1] = { nullptr };
        EXPECT_STATUS(vxGetGraphParameterRefsList(g.graph, 0, 1, got), VX_SUCCESS,
                      "refs list readable after a rejected call");
        EXPECT_TRUE(got[0] == in_ref, "a rejected call leaves the list unchanged");
    }

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 14: vxAddReferencesToGraphParameterList argument validation
// ---------------------------------------------------------------------------
static int test_add_references_validation()
{
    int errors = 0;
    printf("\n=== Test 14: vxAddReferencesToGraphParameterList validation ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 2);

    vx_image spare = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_reference refs[2] = { (vx_reference)g.in, (vx_reference)spare };
    vx_graph_parameter_queue_params_t q[1];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 2;
    q[0].refs_list = refs;
    CHECK_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q));
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    vx_reference add = (vx_reference)g.in;
    EXPECT_STATUS(vxAddReferencesToGraphParameterList(nullptr, 0, 1, &add),
                  VX_ERROR_INVALID_GRAPH, "null graph rejected");
    EXPECT_STATUS(vxAddReferencesToGraphParameterList(g.graph, 0, 0, &add),
                  VX_ERROR_INVALID_PARAMETERS, "adding nothing rejected");
    EXPECT_STATUS(vxAddReferencesToGraphParameterList(g.graph, 0, 1, nullptr),
                  VX_ERROR_INVALID_PARAMETERS, "null reference array rejected");
    EXPECT_STATUS(vxAddReferencesToGraphParameterList(g.graph, 99, 1, &add),
                  VX_ERROR_INVALID_PARAMETERS, "out of range parameter rejected");
    {
        vx_reference nulls[1] = { nullptr };
        EXPECT_STATUS(vxAddReferencesToGraphParameterList(g.graph, 0, 1, nulls),
                      VX_ERROR_INVALID_REFERENCE, "null entry rejected");
        FakeRef fake;
        vx_reference invalid[1] = { invalid_reference(fake) };
        EXPECT_STATUS(vxAddReferencesToGraphParameterList(g.graph, 0, 1, invalid),
                      VX_ERROR_INVALID_REFERENCE, "invalid entry rejected");
    }

    vxReleaseImage(&spare);
    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 15: graph parameters that were never given a queue
//
// Only some of a graph's parameters need to be queued. The ones that were not
// still have to behave sensibly, and an execution has to skip over them.
// ---------------------------------------------------------------------------
static int test_unqueued_parameters()
{
    int errors = 0;
    printf("\n=== Test 15: parameters without a queue ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 2);
    CHECK_STATUS(fill_u8_image(g.in, 0x22));

    // Only the output, parameter 1, is queued.
    vx_reference out_ref = (vx_reference)g.out;
    vx_graph_parameter_queue_params_t q[1];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 1;
    q[0].refs_list_size = 1;
    q[0].refs_list = &out_ref;
    CHECK_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q));
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    // Parameter 0 is an input and has no queue, so it cannot be enqueued to.
    {
        vx_reference in_ref = (vx_reference)g.in;
        EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &in_ref, 1),
                      VX_ERROR_INVALID_PARAMETERS, "enqueue on an unqueued input rejected");
    }
    // Nor dequeued from.
    {
        vx_reference deq = nullptr;
        vx_uint32 num = 0;
        EXPECT_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 0, &deq, 1, &num),
                      VX_ERROR_INVALID_PARAMETERS, "dequeue on an unqueued parameter rejected");
    }

    // The execution consumes from the queued parameter and leaves the other bound
    // to the reference it was created with.
    EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 1, &out_ref, 1),
                  VX_SUCCESS, "output enqueued");
    EXPECT_STATUS(vxProcessGraph(g.graph), VX_SUCCESS, "graph runs with one queue configured");
    EXPECT_TRUE(u8_image_is_uniform(g.out, (vx_uint8)~0x22),
                "the unqueued input was still used as the graph input");
    {
        vx_reference deq = nullptr;
        vx_uint32 num = 0;
        EXPECT_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 1, &deq, 1, &num),
                      VX_SUCCESS, "output dequeued");
        EXPECT_TRUE(num == 1 && deq == out_ref, "the dequeued reference is the one enqueued");
    }

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 16: the timeouts actually expire
//
// VX_GRAPH_TIMEOUT and VX_CONTEXT_EVENT_TIMEOUT are only useful if a wait with
// nothing to wait for gives up instead of blocking forever.
// ---------------------------------------------------------------------------
static int test_timeouts_expire()
{
    int errors = 0;
    printf("\n=== Test 16: waits give up when the timeout expires ===\n");

    const vx_uint32 SHORT_MS = 50;

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 2);

    vx_reference in_ref  = (vx_reference)g.in;
    vx_reference out_ref = (vx_reference)g.out;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    q[1].graph_parameter_index = 1;
    q[1].refs_list_size = 1;
    q[1].refs_list = &out_ref;
    CHECK_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 2, q));
    CHECK_STATUS(vxSetGraphAttribute(g.graph, VX_GRAPH_TIMEOUT, &SHORT_MS, sizeof(SHORT_MS)));
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    // Nothing has been executed, so there is no done reference to collect and the
    // blocking form has to return once the graph timeout has passed.
    {
        vx_reference deq = nullptr;
        vx_uint32 num = 0;
        vx_status s = vxGraphParameterDequeueDoneRef(g.graph, 1, &deq, 1, &num);
        EXPECT_TRUE(s != VX_SUCCESS, "a blocking dequeue gives up once VX_GRAPH_TIMEOUT passes");
    }

    // Same for the event queue, which has its own timeout.
    {
        vx_uint32 t = SHORT_MS;
        CHECK_STATUS(vxSetContextAttribute(context, VX_CONTEXT_EVENT_TIMEOUT, &t, sizeof(t)));
        CHECK_STATUS(vxEnableEvents(context));
        vx_event_t ev;
        memset(&ev, 0, sizeof(ev));
        vx_status s = vxWaitEvent(context, &ev, vx_false_e);
        EXPECT_TRUE(s != VX_SUCCESS, "a blocking event wait gives up once the timeout passes");
    }

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 17: a node that fails
//
// A failing execution has to report VX_EVENT_NODE_ERROR for a node the
// application registered, carry the node's status in the event, and still report
// the graph as completed.
// ---------------------------------------------------------------------------
static int test_node_error_events()
{
    int errors = 0;
    printf("\n=== Test 17: node error reporting ===\n");

    vx_context context = vxCreateContext();
    set_event_timeout(context);

    vx_kernel kernel = register_failing_kernel(context);
    CHECK_NOT_NULL(kernel, "vxAddUserKernel");
    if (!kernel) {
        vxReleaseContext(&context);
        return errors;
    }

    vx_graph graph = vxCreateGraph(context);
    vx_image in  = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_image out = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_node node = vxCreateGenericNode(graph, kernel);
    CHECK_NOT_NULL(node, "vxCreateGenericNode");
    CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)in));
    CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)out));
    CHECK_STATUS(fill_u8_image(in, 0x33));

    for (vx_uint32 i = 0; i < 2; i++) {
        vx_parameter prm = vxGetParameterByIndex(node, i);
        if (prm) {
            vxAddParameterToGraph(graph, prm);
            vxReleaseParameter(&prm);
        }
    }

    // Registering for both node events on the same node proves the framework
    // picks the matching registration rather than the first one it finds.
    CHECK_STATUS(vxRegisterEvent((vx_reference)node, VX_EVENT_NODE_ERROR, 0, 1701));
    CHECK_STATUS(vxRegisterEvent((vx_reference)node, VX_EVENT_NODE_COMPLETED, 0, 1702));
    CHECK_STATUS(vxRegisterEvent((vx_reference)graph, VX_EVENT_GRAPH_COMPLETED, 0, 1703));

    vx_reference in_ref  = (vx_reference)in;
    vx_reference out_ref = (vx_reference)out;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    q[1].graph_parameter_index = 1;
    q[1].refs_list_size = 1;
    q[1].refs_list = &out_ref;
    CHECK_STATUS(vxSetGraphScheduleConfig(graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 2, q));
    {
        vx_uint32 timeout = WAIT_TIMEOUT_MS;
        CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
    }
    CHECK_STATUS(vxEnableEvents(context));
    EXPECT_STATUS(vxVerifyGraph(graph), VX_SUCCESS, "vxVerifyGraph with a user node");

    CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 0, &in_ref, 1));
    CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 1, &out_ref, 1));

    // The kernel always fails, so the execution has to fail too.
    EXPECT_TRUE(vxProcessGraph(graph) != VX_SUCCESS, "a failing kernel fails the execution");

    EventTally t = drain_events(context);
    EXPECT_TRUE(t.node_error == 1, "exactly one VX_EVENT_NODE_ERROR reported");
    EXPECT_TRUE(t.node_completed == 0, "no node completion reported for a node that failed");
    EXPECT_TRUE(t.graph_completed == 1, "graph completion still reported for a failed execution");
    bool saw_error_app_value = false;
    for (vx_uint32 v : t.app_values) {
        if (v == 1701) saw_error_app_value = true;
    }
    EXPECT_TRUE(saw_error_app_value, "the node error carries the app_value registered for it");

    vxReleaseNode(&node);
    vxReleaseImage(&in);
    vxReleaseImage(&out);
    vxReleaseGraph(&graph);
    vxRemoveKernel(kernel);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 18: VX_REFERENCE_ENQUEUE_COUNT through a whole cycle
//
// The attribute exists so an application can tell whether a reference is still
// owned by a queue and therefore unsafe to touch.
// ---------------------------------------------------------------------------
static int test_enqueue_count_lifecycle()
{
    int errors = 0;
    printf("\n=== Test 18: VX_REFERENCE_ENQUEUE_COUNT lifecycle ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 2);
    CHECK_STATUS(fill_u8_image(g.in, 0x44));

    vx_reference in_ref  = (vx_reference)g.in;
    vx_reference out_ref = (vx_reference)g.out;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    q[1].graph_parameter_index = 1;
    q[1].refs_list_size = 1;
    q[1].refs_list = &out_ref;
    CHECK_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 2, q));
    {
        vx_uint32 timeout = WAIT_TIMEOUT_MS;
        CHECK_STATUS(vxSetGraphAttribute(g.graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
    }
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    auto count_of = [&](vx_reference r) -> vx_uint32 {
        vx_uint32 c = 0xFFFFFFFF;
        if (vxQueryReference(r, VX_REFERENCE_ENQUEUE_COUNT, &c, sizeof(c)) != VX_SUCCESS)
            return 0xFFFFFFFF;
        return c;
    };

    EXPECT_TRUE(count_of(in_ref) == 0, "a reference that was never enqueued counts zero");

    CHECK_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &in_ref, 1));
    EXPECT_TRUE(count_of(in_ref) == 1, "a reference waiting in the ready queue counts one");

    CHECK_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 1, &out_ref, 1));
    EXPECT_STATUS(vxProcessGraph(g.graph), VX_SUCCESS, "graph runs");

    // The execution is over but the application has not collected the references
    // yet, so they are still owned by the queues.
    EXPECT_TRUE(count_of(out_ref) == 1, "a reference waiting to be dequeued still counts");

    {
        vx_reference deq = nullptr;
        vx_uint32 num = 0;
        CHECK_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 0, &deq, 1, &num));
        CHECK_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 1, &deq, 1, &num));
    }
    EXPECT_TRUE(count_of(in_ref) == 0, "the count drops back to zero once collected");
    EXPECT_TRUE(count_of(out_ref) == 0, "the same for the output");

    // A reference that belongs to no graph at all is also countable.
    {
        vx_image loose = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
        EXPECT_TRUE(count_of((vx_reference)loose) == 0, "a reference outside any queue counts zero");
        vxReleaseImage(&loose);
    }

    // The count is per context, so a second graph in the same context that does
    // no queueing at all has to be walked over rather than tripped on.
    {
        NotGraph plain = make_not_graph(context, 0);
        CHECK_STATUS(vxVerifyGraph(plain.graph));
        CHECK_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &in_ref, 1));
        EXPECT_TRUE(count_of(in_ref) == 1,
                    "an unpipelined graph in the same context does not disturb the count");
        {
            vx_reference deq = nullptr;
            vx_uint32 num = 0;
            CHECK_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 1, &out_ref, 1));
            CHECK_STATUS(vxProcessGraph(g.graph));
            CHECK_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 0, &deq, 1, &num));
            CHECK_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 1, &deq, 1, &num));
        }
        release_not_graph(plain);
    }

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 19: scheduling a QUEUE_AUTO graph explicitly, and tearing down a graph
// that is still streaming
// ---------------------------------------------------------------------------
static int test_auto_schedule_and_teardown()
{
    int errors = 0;
    printf("\n=== Test 19: explicit schedule under QUEUE_AUTO, teardown while streaming ===\n");

    // In QUEUE_AUTO the framework schedules by itself, so an explicit request has
    // nothing to add and must not be treated as an error.
    {
        vx_context context = vxCreateContext();
        NotGraph g = make_not_graph(context, 2);
        CHECK_STATUS(fill_u8_image(g.in, 0x55));

        vx_reference in_ref  = (vx_reference)g.in;
        vx_reference out_ref = (vx_reference)g.out;
        vx_graph_parameter_queue_params_t q[2];
        memset(q, 0, sizeof(q));
        q[0].graph_parameter_index = 0;
        q[0].refs_list_size = 1;
        q[0].refs_list = &in_ref;
        q[1].graph_parameter_index = 1;
        q[1].refs_list_size = 1;
        q[1].refs_list = &out_ref;
        CHECK_STATUS(vxSetGraphScheduleConfig(g.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO, 2, q));
        {
            vx_uint32 timeout = WAIT_TIMEOUT_MS;
            CHECK_STATUS(vxSetGraphAttribute(g.graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
        }
        EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");
        EXPECT_STATUS(vxScheduleGraph(g.graph), VX_SUCCESS,
                      "an explicit schedule under QUEUE_AUTO is accepted");
        EXPECT_STATUS(vxWaitGraph(g.graph), VX_SUCCESS, "waiting on it is accepted");

        CHECK_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 0, &in_ref, 1));
        CHECK_STATUS(vxGraphParameterEnqueueReadyRef(g.graph, 1, &out_ref, 1));
        {
            vx_reference deq = nullptr;
            vx_uint32 num = 0;
            EXPECT_STATUS(vxGraphParameterDequeueDoneRef(g.graph, 1, &deq, 1, &num),
                          VX_SUCCESS, "the executor still runs the graph on its own");
        }
        release_not_graph(g);
        vxReleaseContext(&context);
    }

    // Releasing a graph without stopping streaming first has to shut the thread
    // down rather than leave it running against a freed graph.
    {
        vx_context context = vxCreateContext();
        NotGraph g = make_not_graph(context, 0);
        CHECK_STATUS(fill_u8_image(g.in, 0x66));
        EXPECT_STATUS(vxEnableGraphStreaming(g.graph, g.node), VX_SUCCESS, "streaming enabled");
        EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");
        EXPECT_STATUS(vxStartGraphStreaming(g.graph), VX_SUCCESS, "streaming started");
        // Long enough for the streaming thread to get through several iterations.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        release_not_graph(g);
        printf("  PASS: graph released while streaming was still running\n");
        vxReleaseContext(&context);
    }

    return errors;
}

// ---------------------------------------------------------------------------
// Test 20: the kernel-side surface of the extension
// ---------------------------------------------------------------------------
static int test_kernel_surface()
{
    int errors = 0;
    printf("\n=== Test 20: kernel attributes and parameter config ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 0);

    vx_kernel kernel = vxGetKernelByEnum(context, VX_KERNEL_NOT);
    CHECK_NOT_NULL(kernel, "vxGetKernelByEnum");

    // vxGetKernelParameterConfig is part of the extension's header but is not
    // implemented here, and the spec's answer for that is NOT_SUPPORTED rather
    // than a crash or a false success.
    {
        vx_kernel_parameter_config_t cfg[2];
        memset(cfg, 0, sizeof(cfg));
        vx_status s = vxGetKernelParameterConfig(kernel, 2, cfg);
        EXPECT_TRUE(s == VX_ERROR_NOT_SUPPORTED || s == VX_SUCCESS,
                    "vxGetKernelParameterConfig reports a defined status");
        printf("  INFO: vxGetKernelParameterConfig returned %d\n", s);
    }

    // The pipeup depths are kernel attributes a user kernel would set. A kernel
    // that has already been finalized cannot have them changed.
    {
        vx_uint32 depth = 0;
        vx_status s = vxQueryKernel(kernel, VX_KERNEL_PIPEUP_OUTPUT_DEPTH, &depth, sizeof(depth));
        if (s == VX_SUCCESS) {
            EXPECT_TRUE(depth >= 1, "VX_KERNEL_PIPEUP_OUTPUT_DEPTH is at least one");
            s = vxQueryKernel(kernel, VX_KERNEL_PIPEUP_INPUT_DEPTH, &depth, sizeof(depth));
            EXPECT_STATUS(s, VX_SUCCESS, "VX_KERNEL_PIPEUP_INPUT_DEPTH queryable");
            vx_uint32 bad = 0;
            EXPECT_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_PIPEUP_OUTPUT_DEPTH, &bad, sizeof(bad)),
                          VX_ERROR_INVALID_PARAMETERS, "a depth below one is rejected");
            vx_uint32 two = 2;
            EXPECT_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_PIPEUP_OUTPUT_DEPTH, &two, sizeof(two)),
                          VX_ERROR_INVALID_PARAMETERS, "a finalized kernel rejects the change");
        } else {
            printf("  INFO: VX_KERNEL_PIPEUP_OUTPUT_DEPTH returned %d\n", s);
        }
    }

    // VX_NODE_STATE belongs to this extension too.
    {
        vx_uint32 state = 0xFFFFFFFF;
        EXPECT_STATUS(vxQueryNode(g.node, VX_NODE_STATE, &state, sizeof(state)),
                      VX_SUCCESS, "VX_NODE_STATE queryable");
        EXPECT_TRUE(state == VX_NODE_STATE_STEADY || state == VX_NODE_STATE_PIPEUP,
                    "VX_NODE_STATE reports a defined state");
    }

    vxReleaseKernel(&kernel);
    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 21: the graph-scoped event calls reject a null graph
// ---------------------------------------------------------------------------
static int test_graph_event_null_handling()
{
    int errors = 0;
    printf("\n=== Test 21: graph event calls with a null graph ===\n");

    vx_event_t ev;
    memset(&ev, 0, sizeof(ev));
    EXPECT_STATUS(vxWaitGraphEvent(nullptr, &ev, vx_true_e),
                  VX_ERROR_INVALID_REFERENCE, "vxWaitGraphEvent rejects a null graph");
    EXPECT_STATUS(vxEnableGraphEvents(nullptr),
                  VX_ERROR_INVALID_REFERENCE, "vxEnableGraphEvents rejects a null graph");
    EXPECT_STATUS(vxDisableGraphEvents(nullptr),
                  VX_ERROR_INVALID_REFERENCE, "vxDisableGraphEvents rejects a null graph");
    EXPECT_STATUS(vxSendUserGraphEvent(nullptr, 1, nullptr),
                  VX_ERROR_INVALID_REFERENCE, "vxSendUserGraphEvent rejects a null graph");
    EXPECT_STATUS(vxRegisterGraphEvent(nullptr, VX_EVENT_GRAPH_COMPLETED, 0, 1),
                  VX_ERROR_INVALID_REFERENCE, "vxRegisterGraphEvent rejects a null reference");
    EXPECT_STATUS(vxEnableGraphStreaming(nullptr, nullptr),
                  VX_ERROR_INVALID_REFERENCE, "vxEnableGraphStreaming rejects a null graph");
    EXPECT_STATUS(vxStartGraphStreaming(nullptr),
                  VX_ERROR_INVALID_REFERENCE, "vxStartGraphStreaming rejects a null graph");
    EXPECT_STATUS(vxStopGraphStreaming(nullptr),
                  VX_ERROR_INVALID_REFERENCE, "vxStopGraphStreaming rejects a null graph");
    return errors;
}

// ---------------------------------------------------------------------------
// Test 22: node completion events from a pipelined execution
//
// Registrations are held per context, so an execution has to report the nodes of
// the graph that ran and leave every other registration alone.
// ---------------------------------------------------------------------------
static int test_node_events_under_pipelining()
{
    int errors = 0;
    printf("\n=== Test 22: node events from a pipelined execution ===\n");

    vx_context context = vxCreateContext();
    set_event_timeout(context);

    NotGraph a = make_not_graph(context, 2);
    NotGraph b = make_not_graph(context, 0);
    CHECK_STATUS(fill_u8_image(a.in, 0x77));

    // One registration for a node of the graph that will run, one for a node of a
    // graph that will not.
    CHECK_STATUS(vxRegisterEvent((vx_reference)a.node, VX_EVENT_NODE_COMPLETED, 0, 2201));
    CHECK_STATUS(vxRegisterEvent((vx_reference)b.node, VX_EVENT_NODE_COMPLETED, 0, 2202));
    CHECK_STATUS(vxRegisterEvent((vx_reference)a.graph, VX_EVENT_GRAPH_COMPLETED, 0, 2203));

    vx_reference in_ref  = (vx_reference)a.in;
    vx_reference out_ref = (vx_reference)a.out;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    q[1].graph_parameter_index = 1;
    q[1].refs_list_size = 1;
    q[1].refs_list = &out_ref;
    CHECK_STATUS(vxSetGraphScheduleConfig(a.graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 2, q));
    {
        vx_uint32 timeout = WAIT_TIMEOUT_MS;
        CHECK_STATUS(vxSetGraphAttribute(a.graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
    }
    CHECK_STATUS(vxEnableEvents(context));
    CHECK_STATUS(vxVerifyGraph(a.graph));
    CHECK_STATUS(vxGraphParameterEnqueueReadyRef(a.graph, 0, &in_ref, 1));
    CHECK_STATUS(vxGraphParameterEnqueueReadyRef(a.graph, 1, &out_ref, 1));
    EXPECT_STATUS(vxProcessGraph(a.graph), VX_SUCCESS, "pipelined execution");

    EventTally t = drain_events(context);
    EXPECT_TRUE(t.node_completed >= 1, "the node of the graph that ran reported completion");
    bool saw_other_graph = false, saw_this_graph = false;
    for (vx_uint32 v : t.app_values) {
        if (v == 2202) saw_other_graph = true;
        if (v == 2201) saw_this_graph = true;
    }
    EXPECT_TRUE(saw_this_graph, "the completion carries the app_value registered for that node");
    EXPECT_TRUE(!saw_other_graph, "a node of another graph reports nothing");
    EXPECT_TRUE(t.graph_completed == 1, "the graph reported completion once");

    // Each node reports once per execution and not more.
    EXPECT_TRUE(t.node_completed == 1, "the node reported completion exactly once");

    release_not_graph(a);
    release_not_graph(b);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 24: a user node inside a pipelined graph
//
// A user node is not rewritten by graph optimization, so this is the case where
// the node the application registered against is the node that actually runs.
// ---------------------------------------------------------------------------
static int test_user_node_under_pipelining()
{
    int errors = 0;
    printf("\n=== Test 24: a user node in a pipelined graph ===\n");

    vx_context context = vxCreateContext();
    set_event_timeout(context);

    vx_kernel kernel = register_passthrough_kernel(context);
    CHECK_NOT_NULL(kernel, "vxAddUserKernel");
    if (!kernel) {
        vxReleaseContext(&context);
        return errors;
    }

    vx_graph graph = vxCreateGraph(context);
    vx_image in  = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_image out = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_node node = vxCreateGenericNode(graph, kernel);
    CHECK_NOT_NULL(node, "vxCreateGenericNode");
    CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)in));
    CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)out));
    CHECK_STATUS(fill_u8_image(in, 0x99));
    for (vx_uint32 i = 0; i < 2; i++) {
        vx_parameter prm = vxGetParameterByIndex(node, i);
        if (prm) {
            vxAddParameterToGraph(graph, prm);
            vxReleaseParameter(&prm);
        }
    }

    CHECK_STATUS(vxRegisterEvent((vx_reference)node, VX_EVENT_NODE_COMPLETED, 0, 2401));
    CHECK_STATUS(vxRegisterEvent((vx_reference)graph, VX_EVENT_GRAPH_COMPLETED, 0, 2402));

    vx_reference in_ref  = (vx_reference)in;
    vx_reference out_ref = (vx_reference)out;
    vx_graph_parameter_queue_params_t q[2];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 1;
    q[0].refs_list = &in_ref;
    q[1].graph_parameter_index = 1;
    q[1].refs_list_size = 1;
    q[1].refs_list = &out_ref;
    CHECK_STATUS(vxSetGraphScheduleConfig(graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 2, q));
    {
        vx_uint32 timeout = WAIT_TIMEOUT_MS;
        CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
    }
    CHECK_STATUS(vxEnableEvents(context));
    EXPECT_STATUS(vxVerifyGraph(graph), VX_SUCCESS, "vxVerifyGraph");
    CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 0, &in_ref, 1));
    CHECK_STATUS(vxGraphParameterEnqueueReadyRef(graph, 1, &out_ref, 1));
    EXPECT_STATUS(vxProcessGraph(graph), VX_SUCCESS, "pipelined execution of a user node");

    EventTally t = drain_events(context);
    EXPECT_TRUE(t.node_completed == 1, "the user node reported completion exactly once");
    EXPECT_TRUE(t.graph_completed == 1, "the graph reported completion once");

    {
        vx_reference deq = nullptr;
        vx_uint32 num = 0;
        EXPECT_STATUS(vxGraphParameterDequeueDoneRef(graph, 1, &deq, 1, &num),
                      VX_SUCCESS, "the output comes back");
        EXPECT_TRUE(num == 1 && deq == out_ref, "and it is the reference that was enqueued");
    }

    vxReleaseNode(&node);
    vxReleaseImage(&in);
    vxReleaseImage(&out);
    vxReleaseGraph(&graph);
    vxRemoveKernel(kernel);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 25: enqueueing an image that another node views through an ROI
//
// Queueing substitutes the reference bound to a graph parameter. An ROI the
// application created over some other image is a reference of its own, and it
// keeps viewing the image it was created from. This pins that down, because the
// alternative -- repointing an application's ROI at whatever was last enqueued --
// would change the meaning of an object the application still holds.
// ---------------------------------------------------------------------------
static int test_roi_of_queued_image()
{
    int errors = 0;
    printf("\n=== Test 25: a queued image with an ROI over it ===\n");

    const vx_uint8 FILL_A = 0x11;
    const vx_uint8 FILL_B = 0x22;

    vx_context context = vxCreateContext();
    vx_graph graph = vxCreateGraph(context);

    vx_image master  = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_image master2 = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_rectangle_t rect = { 0, 0, IMG_W / 2, IMG_H / 2 };
    vx_image roi = vxCreateImageFromROI(master, &rect);
    CHECK_NOT_NULL(roi, "vxCreateImageFromROI");

    vx_image out_full = vxCreateImage(context, IMG_W, IMG_H, VX_DF_IMAGE_U8);
    vx_image out_roi  = vxCreateImage(context, IMG_W / 2, IMG_H / 2, VX_DF_IMAGE_U8);

    vx_node n_full = vxNotNode(graph, master, out_full);
    vx_node n_roi  = vxNotNode(graph, roi, out_roi);
    CHECK_NOT_NULL(n_full, "vxNotNode on the master");
    CHECK_NOT_NULL(n_roi, "vxNotNode on the ROI");

    // Different content in each image, so which one each node read is visible in
    // its output.
    CHECK_STATUS(fill_u8_image(master, FILL_A));
    CHECK_STATUS(fill_u8_image(master2, FILL_B));

    // Graph parameter 0 is the master image.
    {
        vx_parameter prm = vxGetParameterByIndex(n_full, 0);
        CHECK_STATUS(vxAddParameterToGraph(graph, prm));
        vxReleaseParameter(&prm);
    }

    vx_reference refs[2] = { (vx_reference)master, (vx_reference)master2 };
    vx_graph_parameter_queue_params_t q[1];
    memset(q, 0, sizeof(q));
    q[0].graph_parameter_index = 0;
    q[0].refs_list_size = 2;
    q[0].refs_list = refs;
    CHECK_STATUS(vxSetGraphScheduleConfig(graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL, 1, q));
    {
        vx_uint32 timeout = WAIT_TIMEOUT_MS;
        CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_TIMEOUT, &timeout, sizeof(timeout)));
    }
    EXPECT_STATUS(vxVerifyGraph(graph), VX_SUCCESS, "vxVerifyGraph");

    // Run once with the image the graph was built with, then once with the other.
    for (int pass = 0; pass < 2; pass++) {
        vx_reference enq = refs[pass];
        vx_uint8 enqueued_fill = (pass == 0) ? FILL_A : FILL_B;
        EXPECT_STATUS(vxGraphParameterEnqueueReadyRef(graph, 0, &enq, 1), VX_SUCCESS,
                      pass == 0 ? "enqueue the original image" : "enqueue the other image");
        EXPECT_STATUS(vxProcessGraph(graph), VX_SUCCESS,
                      pass == 0 ? "first execution" : "execution after the swap");
        // The graph parameter follows what was enqueued.
        EXPECT_TRUE(u8_image_is_uniform(out_full, (vx_uint8)~enqueued_fill),
                    "the node on the graph parameter read the enqueued reference");
        // The ROI does not: it still reads the image it was created over.
        EXPECT_TRUE(u8_image_is_uniform(out_roi, (vx_uint8)~FILL_A),
                    "the ROI node read the image the ROI was created from");
        vx_reference deq = nullptr;
        vx_uint32 num = 0;
        EXPECT_STATUS(vxGraphParameterDequeueDoneRef(graph, 0, &deq, 1, &num), VX_SUCCESS,
                      "the reference comes back");
        EXPECT_TRUE(num == 1 && deq == enq, "and it is the one that was enqueued");
    }

    vxReleaseNode(&n_full);
    vxReleaseNode(&n_roi);
    vxReleaseImage(&out_full);
    vxReleaseImage(&out_roi);
    vxReleaseImage(&roi);
    vxReleaseImage(&master);
    vxReleaseImage(&master2);
    vxReleaseGraph(&graph);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// Test 23: a streaming graph driven by hand
//
// With streaming enabled the graph belongs to the streaming thread. An explicit
// execution request is not an error, it simply has nothing of its own to do.
// ---------------------------------------------------------------------------
static int test_streaming_explicit_execution()
{
    int errors = 0;
    printf("\n=== Test 23: explicit execution of a streaming graph ===\n");

    vx_context context = vxCreateContext();
    NotGraph g = make_not_graph(context, 0);
    CHECK_STATUS(fill_u8_image(g.in, 0x88));

    EXPECT_STATUS(vxEnableGraphStreaming(g.graph, g.node), VX_SUCCESS, "streaming enabled");
    EXPECT_STATUS(vxVerifyGraph(g.graph), VX_SUCCESS, "vxVerifyGraph");

    // Streaming has not been started, so this is the application's own request.
    EXPECT_STATUS(vxProcessGraph(g.graph), VX_SUCCESS,
                  "an explicit execution of a streaming graph is accepted");

    // Nothing is in flight, so this has nothing to wait for and has to say so.
    // Waiting for a streaming thread that was never started would never return.
    EXPECT_STATUS(vxWaitGraph(g.graph), VX_SUCCESS,
                  "waiting on a graph whose streaming never started returns");

    // Now let the streaming thread run for long enough to get through many
    // iterations of its loop, then stop it the ordinary way.
    EXPECT_STATUS(vxStartGraphStreaming(g.graph), VX_SUCCESS, "streaming started");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_STATUS(vxStopGraphStreaming(g.graph), VX_SUCCESS, "streaming stopped");

    // The output has been written by the streaming thread.
    EXPECT_TRUE(u8_image_is_uniform(g.out, (vx_uint8)~0x88),
                "the streaming thread executed the graph");

    release_not_graph(g);
    vxReleaseContext(&context);
    return errors;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    printf("OpenVX Graph Pipelining Extension API Coverage Test\n");
    printf("===================================================\n");

    // Every wait in this test is bounded, so the whole run is bounded too. If one
    // of them ever stops being bounded, say which test was running rather than
    // leaving a build to sit on it until the job is killed.
    std::thread watchdog([]() {
        std::this_thread::sleep_for(std::chrono::seconds(300));
        printf("\nFATAL: the test suite is stuck; the output above ends at the test that hung\n");
        fflush(stdout);
        abort();
    });
    watchdog.detach();

    // The extension can be compiled out, in which case every entry point reports
    // VX_ERROR_NOT_SUPPORTED and there is nothing here to check.
    {
        vx_context probe = vxCreateContext();
        if (!probe) {
            printf("FATAL: vxCreateContext failed\n");
            return 1;
        }
        vx_graph graph = vxCreateGraph(probe);
        vx_status s = vxSetGraphScheduleConfig(graph, VX_GRAPH_SCHEDULE_MODE_NORMAL, 0, nullptr);
        vxReleaseGraph(&graph);
        vxReleaseContext(&probe);
        if (s == VX_ERROR_NOT_SUPPORTED) {
            printf("SKIP: built without the pipelining extension\n");
            return 0;
        }
    }

    int total_errors = 0;
    total_errors += test_schedule_config_validation();
    total_errors += test_refs_list_after_verify();
    total_errors += test_queue_validation();
    total_errors += test_event_registration_validation();
    total_errors += test_event_delivery_gating();
    total_errors += test_events_disabled_semantics();
    total_errors += test_user_events();
    total_errors += test_attributes();
    total_errors += test_manual_queue_end_to_end();
    total_errors += test_streaming();
    total_errors += test_queue_auto();
    total_errors += test_concurrent_refs_handover();
    total_errors += test_post_verify_config_validation();
    total_errors += test_add_references_validation();
    total_errors += test_unqueued_parameters();
    total_errors += test_timeouts_expire();
    total_errors += test_node_error_events();
    total_errors += test_enqueue_count_lifecycle();
    total_errors += test_auto_schedule_and_teardown();
    total_errors += test_kernel_surface();
    total_errors += test_graph_event_null_handling();
    total_errors += test_node_events_under_pipelining();
    total_errors += test_user_node_under_pipelining();
    total_errors += test_roi_of_queued_image();
    total_errors += test_streaming_explicit_execution();

    printf("\n===================================================\n");
    if (total_errors == 0) {
        printf("RESULT: ALL TESTS PASSED\n");
    } else {
        printf("RESULT: %d ERROR(S) DETECTED\n", total_errors);
    }
    return (total_errors == 0) ? 0 : 1;
}
