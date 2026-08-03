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

static vx_status fill_u8_image(vx_image img, vx_uint8 value)
{
    std::vector<vx_uint8> buf((size_t)IMG_W * IMG_H, value);
    vx_rectangle_t rect = { 0, 0, IMG_W, IMG_H };
    vx_imagepatch_addressing_t addr;
    memset(&addr, 0, sizeof(addr));
    addr.dim_x = IMG_W;
    addr.dim_y = IMG_H;
    addr.stride_x = 1;
    addr.stride_y = (vx_int32)IMG_W;
    return vxCopyImagePatch(img, &rect, 0, &addr, buf.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
}

// Reads the image back and reports whether every pixel holds the expected value.
static bool u8_image_is_uniform(vx_image img, vx_uint8 expected)
{
    std::vector<vx_uint8> buf((size_t)IMG_W * IMG_H, 0);
    vx_rectangle_t rect = { 0, 0, IMG_W, IMG_H };
    vx_imagepatch_addressing_t addr;
    memset(&addr, 0, sizeof(addr));
    addr.dim_x = IMG_W;
    addr.dim_y = IMG_H;
    addr.stride_x = 1;
    addr.stride_y = (vx_int32)IMG_W;
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
// main
// ---------------------------------------------------------------------------
int main()
{
    printf("OpenVX Graph Pipelining Extension API Coverage Test\n");
    printf("===================================================\n");

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

    printf("\n===================================================\n");
    if (total_errors == 0) {
        printf("RESULT: ALL TESTS PASSED\n");
    } else {
        printf("RESULT: %d ERROR(S) DETECTED\n", total_errors);
    }
    return (total_errors == 0) ? 0 : 1;
}
