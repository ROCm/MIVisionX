/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_sequence_rearrange.h"
#include "exception.h"


SequenceRearrangeNode::SequenceRearrangeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs)
{
}

void SequenceRearrangeNode::create_node()
{
    if(_node)
        return;

    vx_status status;
    _sequence_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _new_sequence_length);
    status = vxAddArrayItems(_sequence_array, _new_sequence_length, _new_order.data(), sizeof(vx_uint32));
    if(status != VX_SUCCESS)
        THROW("Adding array items failed: "+ TOSTR(status))
    _node = vxExtrppNode_SequenceRearrange(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _sequence_array, _new_sequence_length, _sequence_length, _sequence_count);
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the sequence rearrange (vxExtrppNode_SequenceRearrange) node failed: "+ TOSTR(status))
}

void SequenceRearrangeNode::init(unsigned int* new_order, unsigned int new_sequence_length, unsigned int sequence_length, unsigned int sequence_count)
{
    _new_sequence_length = new_sequence_length;
    _sequence_length = sequence_length;
    _sequence_count = sequence_count;
    _new_order.resize(_new_sequence_length);
    std::copy(new_order, new_order + _new_sequence_length, _new_order.begin());
}

void SequenceRearrangeNode::update_node()
{
}
