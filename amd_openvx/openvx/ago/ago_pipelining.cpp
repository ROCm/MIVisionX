/*
Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "ago_internal.h"
#include <chrono>

//
// OpenVX Pipelining Extension - AGO internal helpers
//
// This file implements the core state management and execution helpers for
// the Khronos OpenVX pipelining/streaming/event-queue extension
// (vx_khr_pipelining.h).
//

#if OPENVX_USE_PIPELINING

AgoGraphPipeliningState * agoGetGraphPipeliningState(AgoGraph * graph)
{
    if (!graph)
        return nullptr;
    if (!graph->pipelining) {
        graph->pipelining = new AgoGraphPipeliningState();
    }
    return graph->pipelining;
}

AgoContextEventSystem * agoGetContextEventSystem(AgoContext * context)
{
    if (!context)
        return nullptr;
    if (!context->events) {
        context->events = new AgoContextEventSystem();
    }
    return context->events;
}

static void agoStopGraphPipeliningExecutor(AgoGraph * graph)
{
    AgoGraphPipeliningState * pipe = graph ? graph->pipelining : nullptr;
    if (!pipe)
        return;

    pipe->executor_stop.store(true);
    {
        std::lock_guard<std::mutex> lock(pipe->enqueue_mtx);
        pipe->enqueue_cv.notify_all();
    }
    if (pipe->executor_thread.joinable()) {
        pipe->executor_thread.join();
    }

    pipe->streaming_stop.store(true);
    if (pipe->streaming_thread.joinable()) {
        pipe->streaming_thread.join();
    }
}

void agoStopGraphPipelining(AgoGraph * graph)
{
    if (!graph)
        return;
    agoStopGraphPipeliningExecutor(graph);
}

static vx_uint64 agoCurrentTimestampNs()
{
    auto now = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return (vx_uint64)ns;
}

bool agoGraphHasNodeEventRegistrations(AgoGraph * graph)
{
    if (!graph || !graph->ref.context)
        return false;
    AgoContextEventSystem * evsys = agoGetContextEventSystem(graph->ref.context);
    if (!evsys)
        return false;
    std::lock_guard<std::mutex> lock(evsys->registrations_mtx);
    for (const auto& reg : evsys->registrations) {
        if (reg.event_type == VX_EVENT_NODE_COMPLETED || reg.event_type == VX_EVENT_NODE_ERROR) {
            AgoReference * r = (AgoReference *)reg.ref;
            if (r && r->type == VX_TYPE_NODE && r->scope == (vx_reference)graph)
                return true;
        }
    }
    return false;
}

static bool agoIsPipeliningGraph(AgoGraph * graph)
{
    AgoGraphPipeliningState * pipe = graph ? graph->pipelining : nullptr;
    if (!pipe)
        return false;
    return pipe->schedule_mode != VX_GRAPH_SCHEDULE_MODE_NORMAL || pipe->streaming_enabled;
}

static AgoGraphParameterQueue * agoGetGraphParameterQueue(AgoGraphPipeliningState * pipe, vx_uint32 index)
{
    if (!pipe || index >= pipe->param_queues.size())
        return nullptr;
    return pipe->param_queues[index].get();
}

//

// The framework only reports events for references the application asked about
// through vxRegisterEvent, so the lookup has to say whether a registration
// exists at all -- not just hand back an app_value that defaults to zero.
static bool agoFindEventRegistration(AgoContext * context, vx_reference ref, vx_enum event_type,
                                     vx_uint32 graph_parameter_index, vx_uint32 * app_value)
{
    AgoContextEventSystem * evsys = agoGetContextEventSystem(context);
    if (!evsys)
        return false;
    std::lock_guard<std::mutex> lock(evsys->registrations_mtx);
    for (const auto& reg : evsys->registrations) {
        if (reg.ref == ref && reg.event_type == event_type &&
            (event_type != VX_EVENT_GRAPH_PARAMETER_CONSUMED || reg.graph_parameter_index == graph_parameter_index)) {
            if (app_value)
                *app_value = reg.app_value;
            return true;
        }
    }
    return false;
}

vx_uint64 agoEventTimestampNs()
{
    return agoCurrentTimestampNs();
}

void agoRemoveEventRegistrations(AgoContext * context, vx_reference ref)
{
    AgoContextEventSystem * evsys = agoGetContextEventSystem(context);
    if (!evsys)
        return;
    std::lock_guard<std::mutex> lock(evsys->registrations_mtx);
    auto& regs = evsys->registrations;
    regs.erase(std::remove_if(regs.begin(), regs.end(),
        [ref](const AgoEventRegistration& reg) { return reg.ref == ref; }), regs.end());
}

vx_uint32 agoGetReferenceEnqueueCount(AgoContext * context, AgoReference * ref)
{
    if (!context || !ref)
        return 0;
    CAgoLock lock(context->cs);
    vx_uint32 count = 0;
    for (AgoGraph * graph = context->graphList.head; graph; graph = graph->next) {
        AgoGraphPipeliningState * pipe = graph->pipelining;
        if (!pipe)
            continue;
        for (auto& qptr : pipe->param_queues) {
            AgoGraphParameterQueue * q = qptr.get();
            if (!q || !q->enabled)
                continue;
            std::lock_guard<std::mutex> qlock(q->mtx);
            for (AgoData * d : q->ready_refs)
                if ((AgoReference *)d == ref) ++count;
            for (AgoData * d : q->consumed_refs)
                if ((AgoReference *)d == ref) ++count;
            for (AgoData * d : q->done_refs)
                if ((AgoReference *)d == ref) ++count;
        }
    }
    return count;
}

// Event helpers
//

static void agoInternalPushEvent(AgoContext * context, const AgoEvent& evt)
{
    AgoContextEventSystem * evsys = agoGetContextEventSystem(context);
    if (!evsys || !evsys->enabled)
        return;
    {
        std::lock_guard<std::mutex> lock(evsys->events_mtx);
        evsys->events.push_back(evt);
    }
    evsys->events_cv.notify_one();
}

void agoPushEvent(AgoContext * context, const AgoEvent& evt)
{
    agoInternalPushEvent(context, evt);
}

void agoNotifyGraphCompleted(AgoGraph * graph)
{
    if (!graph || !graph->ref.context)
        return;
    AgoContextEventSystem * evsys = agoGetContextEventSystem(graph->ref.context);
    if (!evsys || !evsys->enabled)
        return;
    vx_uint32 app_value = 0;
    if (!agoFindEventRegistration(graph->ref.context, (vx_reference)graph, VX_EVENT_GRAPH_COMPLETED, 0, &app_value))
        return;
    AgoEvent evt;
    evt.event_type = VX_EVENT_GRAPH_COMPLETED;
    evt.timestamp = agoCurrentTimestampNs();
    evt.app_value = app_value;
    evt.graph = graph;
    evt.node = nullptr;
    evt.graph_parameter_index = 0;
    evt.status = VX_SUCCESS;
    evt.user_parameter = nullptr;
    agoInternalPushEvent(graph->ref.context, evt);
}

void agoNotifyNodeCompleted(AgoGraph * graph, AgoNode * node)
{
    if (!graph || !node || !graph->ref.context)
        return;
    AgoContextEventSystem * evsys = agoGetContextEventSystem(graph->ref.context);
    if (!evsys || !evsys->enabled)
        return;
    vx_uint32 app_value = 0;
    if (!agoFindEventRegistration(graph->ref.context, (vx_reference)node, VX_EVENT_NODE_COMPLETED, 0, &app_value))
        return;
    AgoEvent evt;
    evt.event_type = VX_EVENT_NODE_COMPLETED;
    evt.timestamp = agoCurrentTimestampNs();
    evt.app_value = app_value;
    evt.graph = graph;
    evt.node = node;
    evt.graph_parameter_index = 0;
    evt.status = VX_SUCCESS;
    evt.user_parameter = nullptr;
    agoInternalPushEvent(graph->ref.context, evt);
}

void agoNotifyNodeError(AgoGraph * graph, AgoNode * node, vx_status status)
{
    if (!graph || !node || !graph->ref.context)
        return;
    AgoContextEventSystem * evsys = agoGetContextEventSystem(graph->ref.context);
    if (!evsys || !evsys->enabled)
        return;
    vx_uint32 app_value = 0;
    if (!agoFindEventRegistration(graph->ref.context, (vx_reference)node, VX_EVENT_NODE_ERROR, 0, &app_value))
        return;
    AgoEvent evt;
    evt.event_type = VX_EVENT_NODE_ERROR;
    evt.timestamp = agoCurrentTimestampNs();
    evt.app_value = app_value;
    evt.graph = graph;
    evt.node = node;
    evt.graph_parameter_index = 0;
    evt.status = status;
    evt.user_parameter = nullptr;
    agoInternalPushEvent(graph->ref.context, evt);
}

// Nodes that survived graph optimization report completion from the executor
// itself. This covers the ones that did not: optimization can rewrite the
// user-visible nodes into internal ones, and the application still registered
// against the originals. Nodes still present in nodeList are skipped here
// because they have already reported, otherwise they would report twice.
static bool agoIsNodeInGraph(AgoGraph * graph, AgoNode * node)
{
    for (AgoNode * n = graph->nodeList.head; n; n = n->next) {
        if (n == node)
            return true;
    }
    return false;
}

static void agoEmitRegisteredNodeEvents(AgoGraph * graph, vx_enum event_type, vx_status err_status)
{
    if (!graph || !graph->ref.context)
        return;
    AgoContextEventSystem * evsys = agoGetContextEventSystem(graph->ref.context);
    if (!evsys || !evsys->enabled)
        return;
    std::lock_guard<std::mutex> lock(evsys->registrations_mtx);
    for (const auto& reg : evsys->registrations) {
        if (reg.event_type != event_type)
            continue;
        AgoReference * r = (AgoReference *)reg.ref;
        if (!r || r->type != VX_TYPE_NODE || r->scope != (vx_reference)graph)
            continue;
        if (agoIsNodeInGraph(graph, (AgoNode *)r))
            continue;
        AgoEvent evt;
        evt.event_type = event_type;
        evt.timestamp = agoCurrentTimestampNs();
        evt.app_value = reg.app_value;
        evt.graph = graph;
        evt.node = (AgoNode *)r;
        evt.graph_parameter_index = 0;
        evt.status = err_status;
        evt.user_parameter = nullptr;
        agoInternalPushEvent(graph->ref.context, evt);
    }
}

void agoNotifyGraphParameterConsumed(AgoGraph * graph, vx_uint32 graph_parameter_index)
{
    if (!graph || !graph->ref.context)
        return;
    AgoContextEventSystem * evsys = agoGetContextEventSystem(graph->ref.context);
    if (!evsys || !evsys->enabled)
        return;
    vx_uint32 app_value = 0;
    if (!agoFindEventRegistration(graph->ref.context, (vx_reference)graph, VX_EVENT_GRAPH_PARAMETER_CONSUMED,
                                  graph_parameter_index, &app_value))
        return;
    AgoEvent evt;
    evt.event_type = VX_EVENT_GRAPH_PARAMETER_CONSUMED;
    evt.timestamp = agoCurrentTimestampNs();
    evt.app_value = app_value;
    evt.graph = graph;
    evt.node = nullptr;
    evt.graph_parameter_index = graph_parameter_index;
    evt.status = VX_SUCCESS;
    evt.user_parameter = nullptr;
    agoInternalPushEvent(graph->ref.context, evt);
}

//
// Reference substitution for pipelined execution.
// After graph optimization the graph parameter may be attached to a wrapper
// node, while the actual work happens in internally created/rewired nodes.
// Rather than swapping only the graph-parameter node's paramList entries, we
// replace every occurrence of the default bound data object in the entire
// graph with the queued reference, execute, then swap back.
//

struct AgoParamBinding {
    AgoData * original;
    AgoData * queued;
};

static void agoSwapDataRefInGraph(AgoGraph * graph, AgoData * dataFind, AgoData * dataReplace)
{
    if (dataFind == dataReplace)
        return;
    // Replace in all node parameter lists.
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        for (vx_uint32 i = 0; i < node->paramCount; i++) {
            if (node->paramList[i] == dataFind) {
                node->paramList[i] = dataReplace;
            }
        }
    }
    // Replace in supernode data lists (GPU path).
#if (ENABLE_OPENCL||ENABLE_HIP)
    for (AgoSuperNode * super = graph->supernodeList; super; super = super->next) {
        for (size_t i = 0; i < super->dataList.size(); i++) {
            if (super->dataList[i] == dataFind) {
                super->dataList[i] = dataReplace;
            }
        }
        for (size_t i = 0; i < super->dataListForAgeDelay.size(); i++) {
            if (super->dataListForAgeDelay[i] == dataFind) {
                super->dataListForAgeDelay[i] = dataReplace;
            }
        }
    }
#endif
    // Replace ROI master links.
    for (AgoData * adata = graph->dataList.head; adata; adata = adata->next) {
        if (adata->ref.type == VX_TYPE_IMAGE && adata->u.img.isROI && adata->u.img.roiMasterImage == dataFind) {
            adata->u.img.roiMasterImage = dataReplace;
        }
    }
}

// Swap a graph parameter binding, expanding object-array/pyramid siblings when
// the queued reference belongs to a replicated object array/pyramid.
static void agoApplyDataRefSwapWithSiblings(AgoGraph * graph, AgoData * original, AgoData * queued)
{
    if (original == queued)
        return;
    AgoData * origParent = original ? original->parent : nullptr;
    AgoData * queuedParent = queued ? queued->parent : nullptr;
    if (origParent && queuedParent && origParent != queuedParent &&
        origParent->numChildren > 1 &&
        (origParent->ref.type == VX_TYPE_OBJECT_ARRAY || origParent->ref.type == VX_TYPE_PYRAMID) &&
        origParent->ref.type == queuedParent->ref.type &&
        origParent->numChildren == queuedParent->numChildren) {
        for (vx_uint32 i = 0; i < (vx_uint32)origParent->numChildren; i++) {
            agoSwapDataRefInGraph(graph, origParent->children[i], queuedParent->children[i]);
        }
    } else {
        agoSwapDataRefInGraph(graph, original, queued);
    }
}

// Graph verification reserves a device buffer for every data object that was
// bound to a GPU node parameter at verification time. Queued references come
// from the application afterwards, so once they are substituted into the graph
// they still have none and the node would launch against a null device
// pointer. Reserve buffers for the newly bound references using the same rules
// verification applies, after all substitutions are in place.
static int agoAllocGpuBuffersForQueuedRefs(AgoGraph * graph)
{
#if (ENABLE_OPENCL||ENABLE_HIP)
    for (AgoNode * node = graph->nodeList.head; node; node = node->next) {
        if (node->attr_affinity.device_type != AGO_KERNEL_FLAG_DEVICE_GPU &&
            !node->akernel->opencl_buffer_access_enable)
            continue;
        for (vx_uint32 i = 0; i < node->paramCount; i++) {
            AgoData * data = node->paramList[i];
            if (!data || data->isVirtual)
                continue;
#if ENABLE_OPENCL
            if (data->opencl_buffer)
                continue;
#else
            if (data->hip_memory)
                continue;
#endif
            if (agoIsPartOfDelay(data)) {
                int siblingTrace[AGO_MAX_DEPTH_FROM_DELAY_OBJECT], siblingTraceCount = 0;
                data = agoGetSiblingTraceToDelayForUpdate(data, siblingTrace, siblingTraceCount);
                if (!data)
                    return VX_FAILURE;
            }
#if ENABLE_OPENCL
            if (agoGpuOclAllocBuffer(data) < 0)
#else
            if (agoGpuHipAllocBuffer(data) < 0)
#endif
            {
                agoAddLogEntry(&graph->ref, VX_FAILURE,
                    "ERROR: agoAllocGpuBuffersForQueuedRefs: GPU buffer allocation failed for node %s arg#%d\n",
                    node->akernel->name, i);
                return VX_FAILURE;
            }
        }
    }
#else
    (void)graph;
#endif
    return VX_SUCCESS;
}

static std::vector<AgoParamBinding> agoCollectGraphParameterBindings(AgoGraph * graph)
{
    std::vector<AgoParamBinding> bindings;
    bindings.resize(graph->parameters.size());
    for (vx_uint32 i = 0; i < (vx_uint32)graph->parameters.size(); i++) {
        vx_parameter param = graph->parameters[i];
        if (!param || param->scope->type != VX_TYPE_NODE) {
            bindings[i] = { nullptr, nullptr };
            continue;
        }
        AgoNode * node = (AgoNode *)param->scope;
        if (!node) {
            bindings[i] = { nullptr, nullptr };
            continue;
        }
        AgoData * original = (param->index < node->paramCount) ? node->paramList[param->index] : nullptr;
        bindings[i] = { original, nullptr };
    }
    return bindings;
}

static void agoApplyQueuedRefsToBindings(AgoGraph * graph,
                                          AgoGraphPipeliningState * pipe,
                                          std::vector<AgoParamBinding>& bindings,
                                          std::vector<AgoData *>& consumed_refs)
{
    consumed_refs.assign(bindings.size(), nullptr);
    for (size_t i = 0; i < bindings.size(); i++) {
        AgoGraphParameterQueue * q = agoGetGraphParameterQueue(pipe, (vx_uint32)i);
        if (!q)
            continue;

        AgoData * ref = nullptr;
        {
            std::lock_guard<std::mutex> lock(q->mtx);
            if (!q->ready_refs.empty()) {
                ref = q->ready_refs.front();
                q->ready_refs.pop_front();
                q->consumed_refs.push_back(ref);
            }
        }
        if (!ref)
            continue;

        consumed_refs[i] = ref;
        agoRetainData(graph, ref, false);
        bindings[i].queued = ref;
        if (bindings[i].original) {
            agoApplyDataRefSwapWithSiblings(graph, bindings[i].original, ref);
        }
    }
}

static void agoRestoreBindings(AgoGraph * graph, std::vector<AgoParamBinding>& bindings)
{
    for (auto& b : bindings) {
        if (b.original && b.queued) {
            agoApplyDataRefSwapWithSiblings(graph, b.queued, b.original);
        }
    }
}

static void agoMoveConsumedRefsToDone(AgoGraph * graph)
{
    AgoGraphPipeliningState * pipe = graph->pipelining;
    if (!pipe)
        return;
    for (auto& q : pipe->param_queues) {
        vx_uint32 moved = 0;
        {
            std::lock_guard<std::mutex> lock(q->mtx);
            while (!q->consumed_refs.empty()) {
                q->done_refs.push_back(q->consumed_refs.front());
                q->consumed_refs.pop_front();
                moved++;
            }
        }
        // One event per reference actually consumed by this execution. Keying off
        // done_refs instead would keep reporting for as long as the application
        // leaves anything undequeued, including for parameters that consumed
        // nothing this time round.
        for (vx_uint32 i = 0; i < moved; i++) {
            agoNotifyGraphParameterConsumed(graph, q->index);
        }
    }
    for (auto& q : pipe->param_queues) {
        q->done_cv.notify_all();
    }
}

//
// Single pipelined execution instance (pipeline depth = 1 serialized path).
//
int agoExecutePipelinedGraphOnce(AgoGraph * graph)
{
    AgoGraphPipeliningState * pipe = graph->pipelining;
    if (!pipe)
        return VX_FAILURE;

    // Collect default data references bound to each graph parameter.
    std::vector<AgoParamBinding> bindings = agoCollectGraphParameterBindings(graph);
    // Pop one ref from each configured queue and substitute into the graph.
    std::vector<AgoData *> consumed_refs;
    agoApplyQueuedRefsToBindings(graph, pipe, bindings, consumed_refs);

    // Back the newly substituted references with device memory, then execute
    // the graph synchronously using the normal path.
    int status = agoAllocGpuBuffersForQueuedRefs(graph);
    if (status == VX_SUCCESS) {
        status = agoExecuteGraph(graph);
    }

    // Restore original bindings so the next execution sees the static defaults.
    agoRestoreBindings(graph, bindings);

    // Release references retained for this execution.
    for (AgoData * ref : consumed_refs) {
        if (ref) {
            agoReleaseData(ref, false);
        }
    }

    // Move consumed refs to done queues and wake waiters.
    agoMoveConsumedRefsToDone(graph);

    // Emit node completion events for all user-registered nodes. This covers
    // the case where graph optimization rewrote the user-visible nodes.
    if (status == VX_SUCCESS) {
        agoEmitRegisteredNodeEvents(graph, VX_EVENT_NODE_COMPLETED, VX_SUCCESS);
    }

    // Emit graph completion event.
    if (status == VX_SUCCESS) {
        agoNotifyGraphCompleted(graph);
    }

    return status;
}

//
// QUEUE_MANUAL: drain all ready queues, executing one graph instance per
// complete set of ready refs.
//
int agoExecuteGraphQueueManual(AgoGraph * graph)
{
    AgoGraphPipeliningState * pipe = graph->pipelining;
    if (!pipe)
        return VX_FAILURE;

    int overall_status = VX_SUCCESS;
    vx_uint32 batches = 0;
    for (;;) {
        // Check if every enabled queue has at least one ready ref.
        bool all_ready = true;
        for (auto& q : pipe->param_queues) {
            if (!q->enabled)
                continue;
            std::lock_guard<std::mutex> lock(q->mtx);
            if (q->ready_refs.empty()) {
                all_ready = false;
                break;
            }
        }

        if (!all_ready)
            break;

        int status = agoExecutePipelinedGraphOnce(graph);
        batches++;
        if (status != VX_SUCCESS) {
            overall_status = status;
            break;
        }
    }
    // In this mode references for every graph parameter have to be enqueued
    // before the graph is scheduled; if nothing could run, say so instead of
    // reporting a successful schedule that did no work.
    if (overall_status == VX_SUCCESS && batches == 0) {
        agoAddLogEntry(&graph->ref, VX_FAILURE,
            "ERROR: vxScheduleGraph: QUEUE_MANUAL requires a ready reference at every enqueued graph parameter\n");
        return VX_FAILURE;
    }
    return overall_status;
}

//
// Background executor loop for QUEUE_AUTO.
//
static void agoGraphQueueAutoExecutorLoop(AgoGraph * graph)
{
    AgoGraphPipeliningState * pipe = graph->pipelining;
    if (!pipe)
        return;

    auto anyReady = [&pipe]() -> bool {
        for (auto& q : pipe->param_queues) {
            if (!q->enabled)
                continue;
            std::lock_guard<std::mutex> qlock(q->mtx);
            if (!q->ready_refs.empty())
                return true;
        }
        return false;
    };

    while (!pipe->executor_stop.load()) {
        {
            CAgoLock lock(graph->cs);
            if (pipe->schedule_mode != VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO)
                break;

            // Wait until all enabled queues have at least one ready ref.
            bool all_ready = true;
            for (auto& q : pipe->param_queues) {
                if (!q->enabled)
                    continue;
                std::lock_guard<std::mutex> qlock(q->mtx);
                if (q->ready_refs.empty()) {
                    all_ready = false;
                    break;
                }
            }
            if (all_ready) {
                agoExecutePipelinedGraphOnce(graph);
                continue;
            }
        }
        // Nothing to do: block until a ref is enqueued or we're asked to stop.
        std::unique_lock<std::mutex> elock(pipe->enqueue_mtx);
        pipe->enqueue_cv.wait_for(elock, std::chrono::milliseconds(1), [&pipe, &anyReady]() {
            return pipe->executor_stop.load() || anyReady();
        });
    }
}

//
// Streaming executor loop.
//
static void agoGraphStreamingExecutorLoop(AgoGraph * graph)
{
    AgoGraphPipeliningState * pipe = graph->pipelining;
    if (!pipe)
        return;

    while (!pipe->streaming_stop.load()) {
        {
            CAgoLock lock(graph->cs);
            if (!pipe->streaming_enabled)
                break;
            agoExecutePipelinedGraphOnce(graph);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

//
// Public internal entry point used by agoProcessGraph/agoScheduleGraph when
// the graph is in a pipelining schedule mode.
//
int agoExecuteGraphPipelined(AgoGraph * graph)
{
    if (!agoIsValidGraph(graph))
        return VX_ERROR_INVALID_REFERENCE;

    AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
    if (!pipe)
        return VX_FAILURE;

    if (pipe->schedule_mode == VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL) {
        return agoExecuteGraphQueueManual(graph);
    }

    if (pipe->schedule_mode == VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO) {
        // QUEUE_AUTO runs via the background executor; synchronous entry has
        // nothing to do because the executor wakes whenever refs are enqueued.
        return VX_SUCCESS;
    }

    // Streaming mode handled by streaming thread, not via this entry.
    return VX_SUCCESS;
}

//
// Start the QUEUE_AUTO background executor if not already running.
//
void agoStartGraphPipeliningAutoExecutor(AgoGraph * graph)
{
    AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
    if (!pipe)
        return;
    if (!pipe->executor_thread.joinable()) {
        pipe->executor_stop.store(false);
        pipe->executor_thread = std::thread([graph]() {
            agoGraphQueueAutoExecutorLoop(graph);
        });
    }
}

//
// Start the streaming thread.
//
void agoStartGraphStreamingThread(AgoGraph * graph)
{
    AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
    if (!pipe)
        return;
    if (!pipe->streaming_thread.joinable()) {
        pipe->streaming_stop.store(false);
        pipe->streaming_thread = std::thread([graph]() {
            agoGraphStreamingExecutorLoop(graph);
        });
    }
}
#else
// Stubs when the pipelining/streaming/event extension is disabled.
AgoGraphPipeliningState * agoGetGraphPipeliningState(AgoGraph *) { return nullptr; }
AgoContextEventSystem * agoGetContextEventSystem(AgoContext *) { return nullptr; }
void agoStopGraphPipelining(AgoGraph *) {}
bool agoGraphHasNodeEventRegistrations(AgoGraph *) { return false; }
void agoPushEvent(AgoContext *, const AgoEvent&) {}
void agoNotifyGraphCompleted(AgoGraph *) {}
void agoNotifyNodeCompleted(AgoGraph *, AgoNode *) {}
void agoNotifyNodeError(AgoGraph *, AgoNode *, vx_status) {}
void agoNotifyGraphParameterConsumed(AgoGraph *, vx_uint32) {}
int agoExecuteGraphPipelined(AgoGraph *) { return VX_ERROR_NOT_SUPPORTED; }
int agoExecutePipelinedGraphOnce(AgoGraph *) { return VX_ERROR_NOT_SUPPORTED; }
void agoStartGraphPipeliningAutoExecutor(AgoGraph *) {}
void agoStartGraphStreamingThread(AgoGraph *) {}
vx_uint32 agoGetReferenceEnqueueCount(AgoContext *, AgoReference *) { return 0; }
void agoRemoveEventRegistrations(AgoContext *, vx_reference) {}
vx_uint64 agoEventTimestampNs() { return 0; }
#endif
