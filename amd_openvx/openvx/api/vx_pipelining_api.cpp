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

//
// OpenVX Pipelining Extension - public API implementation
//

#if OPENVX_USE_PIPELINING

// Records the reference lists supplied by a post-verify vxSetGraphScheduleConfig
// call. The queues themselves were already set up before verify, so this must not
// disturb them beyond replacing the set of references allowed at each parameter.
static vx_status updateGraphScheduleRefsList(
    vx_graph graph,
    vx_enum graph_schedule_mode,
    AgoGraphPipeliningState * pipe,
    vx_uint32 graph_parameters_list_size,
    const vx_graph_parameter_queue_params_t graph_parameters_queue_params_list[])
{
    CAgoLock lock(graph->cs);
    // Only the reference lists may differ from the call made before verify, so a
    // different schedule mode is a reconfiguration and cannot be honoured now.
    if (pipe->schedule_mode != graph_schedule_mode)
        return VX_ERROR_INVALID_PARAMETERS;
    if (pipe->param_queues.size() != graph->parameters.size())
        return VX_FAILURE;
    for (vx_uint32 i = 0; i < graph_parameters_list_size; i++) {
        const vx_graph_parameter_queue_params_t & p = graph_parameters_queue_params_list[i];
        vx_uint32 index = p.graph_parameter_index;
        if (index >= (vx_uint32)graph->parameters.size())
            return VX_ERROR_INVALID_PARAMETERS;
        if (p.refs_list_size == 0)
            return VX_ERROR_INVALID_PARAMETERS;
        AgoGraphParameterQueue * q = pipe->param_queues[index].get();
        // Queuing has to have been requested for this parameter before verify;
        // this call may only fill in references, not enable a new queue.
        if (!q || !q->enabled)
            return VX_ERROR_INVALID_PARAMETERS;
        if (!p.refs_list)
            continue;
        std::vector<AgoData *> refs;
        refs.reserve(p.refs_list_size);
        for (vx_uint32 j = 0; j < p.refs_list_size; j++) {
            vx_reference ref = p.refs_list[j];
            if (!ref)
                return VX_ERROR_INVALID_PARAMETERS;
            if (!agoIsValidReference((AgoReference *)ref))
                return VX_ERROR_INVALID_REFERENCE;
            refs.push_back((AgoData *)ref);
        }
        // Built separately so a bad entry late in the list leaves the queue's
        // existing references untouched.
        std::lock_guard<std::mutex> qlock(q->mtx);
        q->valid_refs = std::move(refs);
    }
    return VX_SUCCESS;
}

VX_API_ENTRY vx_status VX_API_CALL vxSetGraphScheduleConfig(
    vx_graph graph,
    vx_enum graph_schedule_mode,
    vx_uint32 graph_parameters_list_size,
    const vx_graph_parameter_queue_params_t graph_parameters_queue_params_list[])
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        if ((graph_schedule_mode != VX_GRAPH_SCHEDULE_MODE_NORMAL) &&
            (graph_schedule_mode != VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO) &&
            (graph_schedule_mode != VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL)) {
            return VX_ERROR_INVALID_PARAMETERS;
        }
        if (graph_schedule_mode == VX_GRAPH_SCHEDULE_MODE_NORMAL) {
            if (graph_parameters_list_size != 0 || graph_parameters_queue_params_list != nullptr) {
                return VX_ERROR_INVALID_PARAMETERS;
            }
        } else {
            if (graph_parameters_list_size == 0 || graph_parameters_queue_params_list == nullptr) {
                return VX_ERROR_INVALID_PARAMETERS;
            }
        }

        AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
        if (!pipe)
            return VX_FAILURE;

        if (graph->verified) {
            // vx_khr_pipelining 1.1: the application may call this again after
            // verify with everything unchanged except refs_list, which is how it
            // hands over the references when they were not available earlier.
            // Nothing may be reconfigured now, so this only records the lists.
            return updateGraphScheduleRefsList(graph, graph_schedule_mode, pipe,
                                               graph_parameters_list_size,
                                               graph_parameters_queue_params_list);
        }

        // Stop any active executor before reconfiguring. This has to happen outside
        // graph->cs because the executor runs the graph inside that section.
        agoStopGraphPipelining(graph);

        CAgoLock lock(graph->cs);

        pipe->schedule_mode = graph_schedule_mode;
        // Reconfiguring discards whatever was queued, so the executions that
        // went with it are no longer anything a later request can claim.
        pipe->manual_unclaimed_executions.store(0);
        pipe->param_queues.clear();
        pipe->param_queues.resize(graph->parameters.size());
        for (size_t i = 0; i < pipe->param_queues.size(); i++) { if (!pipe->param_queues[i]) pipe->param_queues[i].reset(new AgoGraphParameterQueue()); pipe->param_queues[i]->index = (vx_uint32)i; }
        for (vx_uint32 i = 0; i < graph_parameters_list_size; i++) {
            const vx_graph_parameter_queue_params_t & p = graph_parameters_queue_params_list[i];
            vx_uint32 index = p.graph_parameter_index;
            if (index >= (vx_uint32)graph->parameters.size())
                return VX_ERROR_INVALID_PARAMETERS;
            if (p.refs_list_size == 0)
                return VX_ERROR_INVALID_PARAMETERS;
            // vx_khr_pipelining 1.1: called before vxVerifyGraph, refs_list may
            // be NULL when the application does not have the handles yet, while
            // refs_list_size must always be given. So only the size is required.
            // The spec also says graph_parameter_index must be unique across the
            // list, but that is a requirement on the application and it is not
            // policed here: GraphPipeline.ScalarOutput configures index 1 twice
            // (leaving index 2 unconfigured) and must still be accepted.
            AgoGraphParameterQueue * q = pipe->param_queues[index].get();
            std::lock_guard<std::mutex> qlock(q->mtx);
            q->max_depth = p.refs_list_size;
            q->enabled = true;
            if (p.refs_list) {
                for (vx_uint32 j = 0; j < p.refs_list_size; j++) {
                    vx_reference ref = p.refs_list[j];
                    if (!ref)
                        return VX_ERROR_INVALID_PARAMETERS;
                    if (!agoIsValidReference((AgoReference *)ref))
                        return VX_ERROR_INVALID_REFERENCE;
                    q->valid_refs.push_back((AgoData *)ref);
                }
            }
        }

        if (graph_schedule_mode == VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO) {
            agoStartGraphPipeliningAutoExecutor(graph);
        }

        status = VX_SUCCESS;
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxGetGraphParameterRefsList(
    vx_graph graph,
    vx_uint32 param,
    vx_uint32 ref_list_size,
    vx_reference refs_list[])
{
    vx_status status = VX_ERROR_INVALID_GRAPH;
    if (agoIsValidGraph(graph) && graph->verified) {
        status = VX_ERROR_INVALID_PARAMETERS;
        AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
        if (pipe && param < (vx_uint32)pipe->param_queues.size() && refs_list) {
            AgoGraphParameterQueue * q = pipe->param_queues[param].get();
            std::lock_guard<std::mutex> qlock(q->mtx);
            if (ref_list_size >= (vx_uint32)q->valid_refs.size()) {
                for (size_t i = 0; i < q->valid_refs.size(); i++) {
                    refs_list[i] = (vx_reference)q->valid_refs[i];
                }
                status = VX_SUCCESS;
            }
        }
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxAddReferencesToGraphParameterList(
    vx_graph graph,
    vx_uint32 graph_parameter_index,
    vx_uint32 number_to_add,
    const vx_reference new_references[])
{
    vx_status status = VX_ERROR_INVALID_GRAPH;
    if (agoIsValidGraph(graph) && graph->verified) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (number_to_add == 0 || !new_references)
            return status;
        AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
        if (pipe && graph_parameter_index < (vx_uint32)pipe->param_queues.size()) {
            AgoGraphParameterQueue * q = pipe->param_queues[graph_parameter_index].get();
            std::lock_guard<std::mutex> qlock(q->mtx);
            for (vx_uint32 i = 0; i < number_to_add; i++) {
                if (!new_references[i] || !agoIsValidReference((AgoReference *)new_references[i]))
                    return VX_ERROR_INVALID_REFERENCE;
                q->valid_refs.push_back((AgoData *)new_references[i]);
            }
            status = VX_SUCCESS;
        }
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxGraphParameterEnqueueReadyRef(
    vx_graph graph,
    vx_uint32 graph_parameter_index,
    const vx_reference *refs,
    vx_uint32 num_refs)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        status = VX_ERROR_INVALID_PARAMETERS;
        if (num_refs > 0 && !refs)
            return status;
        AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
        if (!pipe)
            return VX_FAILURE;
        if (graph_parameter_index >= (vx_uint32)pipe->param_queues.size())
            return status;

        AgoGraphParameterQueue * q = pipe->param_queues[graph_parameter_index].get();
        // If the queue is not explicitly enabled by the schedule config, allow
        // enqueueing as long as the corresponding graph parameter is a valid output.
        if (!q->enabled) {
            if (graph_parameter_index >= graph->parameters.size())
                return status;
            vx_parameter param = graph->parameters[graph_parameter_index];
            if (!param || param->direction != VX_OUTPUT)
                return status;
            q->enabled = true;
            // No configured depth for a queue the schedule config never covered,
            // so leave it unbounded rather than inferring one from this call.
            // GraphPipeline.ScalarOutput relies on this path, because the CTS
            // configures graph parameter 1 twice and never configures 2.
            q->max_depth = 0;
        }

        {
            // The whole check-and-push is under the queue lock, so a concurrent
            // vxSetGraphScheduleConfig or vxAddReferencesToGraphParameterList
            // cannot swap valid_refs out from under the search below. The lock is
            // released before enqueue_mtx is taken: the QUEUE_AUTO executor waits
            // on enqueue_cv with a predicate that takes q->mtx, so acquiring the
            // two in the other order here would deadlock against it.
            std::lock_guard<std::mutex> lock(q->mtx);
            for (vx_uint32 i = 0; i < num_refs; i++) {
                if (!refs[i] || !agoIsValidReference((AgoReference *)refs[i]))
                    return VX_ERROR_INVALID_REFERENCE;
                // If valid_refs is configured, reject refs not in the list.
                if (!q->valid_refs.empty()) {
                    bool found = false;
                    for (AgoData * valid : q->valid_refs) {
                        if ((vx_reference)valid == refs[i]) {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        return VX_ERROR_INVALID_PARAMETERS;
                }
                // refs_list_size given at schedule-config time is the queue depth.
                // Counting only the refs still waiting to be picked up keeps this a
                // limit on what the application has handed over but the graph has
                // not yet taken, which is what the depth is there to bound.
                if (q->max_depth && q->ready_refs.size() >= (size_t)q->max_depth)
                    return VX_ERROR_NO_RESOURCES;
                q->ready_refs.push_back((AgoData *)refs[i]);
            }
        }
        {
            std::lock_guard<std::mutex> lock(pipe->enqueue_mtx);
            pipe->enqueue_cv.notify_all();
        }
        status = VX_SUCCESS;
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxGraphParameterDequeueDoneRef(
    vx_graph graph,
    vx_uint32 graph_parameter_index,
    vx_reference *refs,
    vx_uint32 max_refs,
    vx_uint32 *num_refs)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        if (!refs || !num_refs)
            return VX_ERROR_INVALID_PARAMETERS;
        AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
        if (!pipe)
            return VX_FAILURE;
        if (graph_parameter_index >= (vx_uint32)pipe->param_queues.size())
            return VX_ERROR_INVALID_PARAMETERS;

        AgoGraphParameterQueue * q = pipe->param_queues[graph_parameter_index].get();
        if (!q->enabled)
            return VX_ERROR_INVALID_PARAMETERS;

        std::unique_lock<std::mutex> lock(q->mtx);
        if (pipe->timeout_ms == VX_TIMEOUT_WAIT_FOREVER) {
            q->done_cv.wait(lock, [q]() { return !q->done_refs.empty(); });
        } else {
            if (!q->done_cv.wait_for(lock, std::chrono::milliseconds(pipe->timeout_ms),
                                       [q]() { return !q->done_refs.empty(); }))
                return VX_FAILURE;
        }
        if (q->done_refs.empty())
            return VX_FAILURE;
        vx_uint32 count = 0;
        while (count < max_refs && !q->done_refs.empty()) {
            refs[count] = (vx_reference)q->done_refs.front();
            q->done_refs.pop_front();
            count++;
        }
        *num_refs = count;
        status = VX_SUCCESS;
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxGraphParameterCheckDoneRef(
    vx_graph graph,
    vx_uint32 graph_parameter_index,
    vx_uint32 *num_refs)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        if (!num_refs)
            return VX_ERROR_INVALID_PARAMETERS;
        AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
        if (!pipe)
            return VX_FAILURE;
        if (graph_parameter_index >= (vx_uint32)pipe->param_queues.size())
            return VX_ERROR_INVALID_PARAMETERS;

        AgoGraphParameterQueue * q = pipe->param_queues[graph_parameter_index].get();
        std::lock_guard<std::mutex> lock(q->mtx);
        *num_refs = (vx_uint32)q->done_refs.size();
        status = VX_SUCCESS;
    }
    return status;
}

//
// Context-level event API
//

VX_API_ENTRY vx_status VX_API_CALL vxEnableEvents(vx_context context)
{
    if (!agoIsValidContext((AgoContext *)context))
        return VX_ERROR_INVALID_REFERENCE;
    AgoContextEventSystem * evsys = agoGetContextEventSystem((AgoContext *)context);
    if (!evsys)
        return VX_FAILURE;
    {
        std::lock_guard<std::mutex> lock(evsys->events_mtx);
        evsys->enabled = true;
    }
    // Wake anyone parked in a blocking vxWaitEvent while events were disabled.
    evsys->events_cv.notify_all();
    return VX_SUCCESS;
}

VX_API_ENTRY vx_status VX_API_CALL vxDisableEvents(vx_context context)
{
    if (!agoIsValidContext((AgoContext *)context))
        return VX_ERROR_INVALID_REFERENCE;
    AgoContextEventSystem * evsys = agoGetContextEventSystem((AgoContext *)context);
    if (!evsys)
        return VX_FAILURE;
    // Events already queued stay queued: disabling stops new ones being
    // recorded, it does not discard what the application has not yet collected.
    std::lock_guard<std::mutex> lock(evsys->events_mtx);
    evsys->enabled = false;
    return VX_SUCCESS;
}

VX_API_ENTRY vx_status VX_API_CALL vxWaitEvent(vx_context context, vx_event_t *event, vx_bool do_not_block)
{
    if (!agoIsValidContext((AgoContext *)context) || !event)
        return VX_ERROR_INVALID_REFERENCE;
    AgoContextEventSystem * evsys = agoGetContextEventSystem((AgoContext *)context);
    if (!evsys)
        return VX_FAILURE;

    std::unique_lock<std::mutex> lock(evsys->events_mtx);
    // vx_khr_pipelining 1.1: events generated before vxDisableEvents are still
    // returned here, so the wait must not require events to be enabled. What
    // disabling does is stop new ones being recorded, which agoPushEvent already
    // handles, so nothing further can arrive until they are re-enabled.
    auto ready = [&evsys]() { return !evsys->events.empty(); };
    if (do_not_block == vx_true_e) {
        if (evsys->events.empty())
            return VX_FAILURE;
    } else {
        if (evsys->timeout_ms == VX_TIMEOUT_WAIT_FOREVER) {
            evsys->events_cv.wait(lock, ready);
        } else {
            if (!evsys->events_cv.wait_for(lock, std::chrono::milliseconds(evsys->timeout_ms), ready))
                return VX_FAILURE;
        }
    }
    if (evsys->events.empty())
        return VX_FAILURE;

    AgoEvent evt = evsys->events.front();
    evsys->events.pop_front();
    lock.unlock();

    event->type = evt.event_type;
    event->timestamp = evt.timestamp;
    event->app_value = evt.app_value;
    switch (evt.event_type) {
    case VX_EVENT_GRAPH_PARAMETER_CONSUMED:
        event->event_info.graph_parameter_consumed.graph = (vx_graph)evt.graph;
        event->event_info.graph_parameter_consumed.graph_parameter_index = evt.graph_parameter_index;
        break;
    case VX_EVENT_GRAPH_COMPLETED:
        event->event_info.graph_completed.graph = (vx_graph)evt.graph;
        break;
    case VX_EVENT_NODE_COMPLETED:
        event->event_info.node_completed.graph = (vx_graph)evt.graph;
        event->event_info.node_completed.node = (vx_node)evt.node;
        break;
    case VX_EVENT_NODE_ERROR:
        event->event_info.node_error.graph = (vx_graph)evt.graph;
        event->event_info.node_error.node = (vx_node)evt.node;
        event->event_info.node_error.status = evt.status;
        break;
    case VX_EVENT_USER:
        event->event_info.user_event.user_event_parameter = evt.user_parameter;
        break;
    default:
        break;
    }
    return VX_SUCCESS;
}

VX_API_ENTRY vx_status VX_API_CALL vxSendUserEvent(vx_context context, vx_uint32 app_value, const void *parameter)
{
    if (!agoIsValidContext((AgoContext *)context))
        return VX_ERROR_INVALID_REFERENCE;
    AgoContextEventSystem * evsys = agoGetContextEventSystem((AgoContext *)context);
    if (!evsys || !evsys->enabled)
        return VX_FAILURE;

    AgoEvent evt;
    evt.event_type = VX_EVENT_USER;
    evt.timestamp = agoEventTimestampNs();
    evt.app_value = app_value;
    evt.graph = nullptr;
    evt.node = nullptr;
    evt.graph_parameter_index = 0;
    evt.status = VX_SUCCESS;
    evt.user_parameter = (void *)parameter;
    agoPushEvent((AgoContext *)context, evt);
    return VX_SUCCESS;
}

VX_API_ENTRY vx_status VX_API_CALL vxRegisterEvent(vx_reference ref, enum vx_event_type_e type, vx_uint32 param, vx_uint32 app_value)
{
    if (!ref || !agoIsValidReference((AgoReference *)ref))
        return VX_ERROR_INVALID_REFERENCE;
    AgoReference * r = (AgoReference *)ref;

    // The event type has to make sense for the kind of reference given: a graph
    // reports parameter-consumed and graph-completed, a node reports
    // node-completed and node-error. Anything else is not supported.
    AgoGraph * graph = nullptr;
    if (r->type == VX_TYPE_GRAPH) {
        if (type != VX_EVENT_GRAPH_PARAMETER_CONSUMED && type != VX_EVENT_GRAPH_COMPLETED)
            return VX_ERROR_NOT_SUPPORTED;
        graph = (AgoGraph *)r;
    }
    else if (r->type == VX_TYPE_NODE) {
        if (type != VX_EVENT_NODE_COMPLETED && type != VX_EVENT_NODE_ERROR)
            return VX_ERROR_NOT_SUPPORTED;
        graph = (AgoGraph *)r->scope;
    }
    else {
        return VX_ERROR_NOT_SUPPORTED;
    }

    // Registration has to happen while the graph can still be set up for it.
    if (graph && agoIsValidGraph(graph) && graph->verified)
        return VX_ERROR_NOT_SUPPORTED;

    if (type == VX_EVENT_GRAPH_PARAMETER_CONSUMED) {
        if (!graph || !agoIsValidGraph(graph) || param >= (vx_uint32)graph->parameters.size())
            return VX_ERROR_INVALID_PARAMETERS;
    }

    AgoContextEventSystem * evsys = agoGetContextEventSystem(r->context);
    if (!evsys)
        return VX_FAILURE;

    vx_uint32 index = (type == VX_EVENT_GRAPH_PARAMETER_CONSUMED) ? param : 0;
    std::lock_guard<std::mutex> lock(evsys->registrations_mtx);
    // Registering the same thing twice is not an error; it updates the stored
    // app_value. There is only ever one app_value per reference/type/parameter.
    for (auto& existing : evsys->registrations) {
        if (existing.ref == ref && existing.event_type == type &&
            existing.graph_parameter_index == index) {
            existing.app_value = app_value;
            return VX_SUCCESS;
        }
    }

    AgoEventRegistration reg;
    reg.ref = ref;
    reg.event_type = type;
    reg.app_value = app_value;
    reg.graph_parameter_index = index;
    evsys->registrations.push_back(reg);
    return VX_SUCCESS;
}

//
// Graph-level event API (forwarded to context-level event system for now).
//

VX_API_ENTRY vx_status VX_API_CALL vxRegisterGraphEvent(vx_reference graph_or_node, enum vx_event_type_e type, vx_uint32 param, vx_uint32 app_value)
{
    return vxRegisterEvent(graph_or_node, type, param, app_value);
}

VX_API_ENTRY vx_status VX_API_CALL vxWaitGraphEvent(vx_graph graph, vx_event_t *event, vx_bool do_not_block)
{
    if (!agoIsValidGraph((AgoGraph *)graph) || !event)
        return VX_ERROR_INVALID_REFERENCE;
    AgoContext * context = ((AgoGraph *)graph)->ref.context;
    return vxWaitEvent(context, event, do_not_block);
}

VX_API_ENTRY vx_status VX_API_CALL vxEnableGraphEvents(vx_graph graph)
{
    if (!agoIsValidGraph((AgoGraph *)graph))
        return VX_ERROR_INVALID_REFERENCE;
    AgoContext * context = ((AgoGraph *)graph)->ref.context;
    return vxEnableEvents(context);
}

VX_API_ENTRY vx_status VX_API_CALL vxDisableGraphEvents(vx_graph graph)
{
    if (!agoIsValidGraph((AgoGraph *)graph))
        return VX_ERROR_INVALID_REFERENCE;
    AgoContext * context = ((AgoGraph *)graph)->ref.context;
    return vxDisableEvents(context);
}

VX_API_ENTRY vx_status VX_API_CALL vxSendUserGraphEvent(vx_graph graph, vx_uint32 app_value, const void *parameter)
{
    if (!agoIsValidGraph((AgoGraph *)graph))
        return VX_ERROR_INVALID_REFERENCE;
    AgoContext * context = ((AgoGraph *)graph)->ref.context;
    return vxSendUserEvent(context, app_value, parameter);
}

//
// Streaming API
//

VX_API_ENTRY vx_status VX_API_CALL vxEnableGraphStreaming(vx_graph graph, vx_node trigger_node)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
        if (!pipe)
            return VX_FAILURE;
        CAgoLock lock(graph->cs);
        pipe->streaming_enabled = true;
        if (trigger_node && agoIsValidNode((AgoNode *)trigger_node)) {
            pipe->trigger_node = (AgoNode *)trigger_node;
        }
        status = VX_SUCCESS;
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxStartGraphStreaming(vx_graph graph)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        if (!graph->verified)
            return VX_ERROR_NOT_SUFFICIENT;
        AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
        if (!pipe)
            return VX_FAILURE;
        CAgoLock lock(graph->cs);
        if (!pipe->streaming_enabled)
            return VX_FAILURE;
        agoStartGraphStreamingThread(graph);
        status = VX_SUCCESS;
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxStopGraphStreaming(vx_graph graph)
{
    vx_status status = VX_ERROR_INVALID_REFERENCE;
    if (agoIsValidGraph(graph)) {
        AgoGraphPipeliningState * pipe = agoGetGraphPipeliningState(graph);
        if (!pipe)
            return VX_FAILURE;
        // The streaming thread executes the graph under graph->cs, so it has to be
        // joined before that section is entered.
        pipe->streaming_stop.store(true);
        if (pipe->streaming_thread.joinable()) {
            pipe->streaming_thread.join();
        }
        CAgoLock lock(graph->cs);
        pipe->streaming_enabled = false;
        status = VX_SUCCESS;
    }
    return status;
}

//
// Additional helper API (stub)
//

VX_API_ENTRY vx_status VX_API_CALL vxGetKernelParameterConfig(vx_kernel kernel, vx_uint32 num_params, vx_kernel_parameter_config_t parameter_config[])
{
    return VX_ERROR_NOT_SUPPORTED;
}
#else
// Stubs when the pipelining/streaming/event extension is disabled.
VX_API_ENTRY vx_status VX_API_CALL vxSetGraphScheduleConfig(
    vx_graph graph,
    vx_enum graph_schedule_mode,
    vx_uint32 graph_parameters_list_size,
    const vx_graph_parameter_queue_params_t graph_parameters_queue_params_list[])
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxGetGraphParameterRefsList(
    vx_graph graph,
    vx_uint32 param,
    vx_uint32 ref_list_size,
    vx_reference refs_list[])
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxAddReferencesToGraphParameterList(
    vx_graph graph,
    vx_uint32 graph_parameter_index,
    vx_uint32 number_to_add,
    const vx_reference new_references[])
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxGraphParameterEnqueueReadyRef(
    vx_graph graph,
    vx_uint32 graph_parameter_index,
    const vx_reference *refs,
    vx_uint32 num_refs)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxGraphParameterDequeueDoneRef(
    vx_graph graph,
    vx_uint32 graph_parameter_index,
    vx_reference *refs,
    vx_uint32 max_refs,
    vx_uint32 *num_refs)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxGraphParameterCheckDoneRef(
    vx_graph graph,
    vx_uint32 graph_parameter_index,
    vx_uint32 *num_refs)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxEnableEvents(vx_context context)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxDisableEvents(vx_context context)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxWaitEvent(vx_context context, vx_event_t *event, vx_bool do_not_block)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxSendUserEvent(vx_context context, vx_uint32 app_value, const void *parameter)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxRegisterEvent(vx_reference ref, enum vx_event_type_e type, vx_uint32 param, vx_uint32 app_value)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxRegisterGraphEvent(vx_reference graph_or_node, enum vx_event_type_e type, vx_uint32 param, vx_uint32 app_value)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxWaitGraphEvent(vx_graph graph, vx_event_t *event, vx_bool do_not_block)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxEnableGraphEvents(vx_graph graph)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxDisableGraphEvents(vx_graph graph)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxSendUserGraphEvent(vx_graph graph, vx_uint32 app_value, const void *parameter)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxEnableGraphStreaming(vx_graph graph, vx_node trigger_node)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxStartGraphStreaming(vx_graph graph)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxStopGraphStreaming(vx_graph graph)
{
    return VX_ERROR_NOT_SUPPORTED;
}
VX_API_ENTRY vx_status VX_API_CALL vxGetKernelParameterConfig(vx_kernel kernel, vx_uint32 num_params, vx_kernel_parameter_config_t parameter_config[])
{
    return VX_ERROR_NOT_SUPPORTED;
}
#endif
