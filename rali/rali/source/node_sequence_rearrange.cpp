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

    // _affine.resize(6 * _batch_size);

    // uint batch_size = _batch_size;
    // for (uint i=0; i < batch_size; i++ )
    // {
    //      _affine[i*6 + 0] = _x0.renew();
    //      _affine[i*6 + 1] = _y0.renew();
    //      _affine[i*6 + 2] = _x1.renew();
    //      _affine[i*6 + 3] = _y1.renew();
    //      _affine[i*6 + 4] = _o0.renew();
    //      _affine[i*6 + 5] = _o1.renew();

    // }
    // _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    // _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    // std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    // std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());
    // // width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    // // height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    // if(width_status != 0 || height_status != 0)
    //     THROW(" vxAddArrayItems failed in the rotate (vxExtrppNode_SequenceRearrangePD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    vx_status status;
    _sequence_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _sequence_length);
    status = vxAddArrayItems(_sequence_array,_sequence_length, _new_order.data(), sizeof(vx_uint32));
    _node = vxExtrppNode_SequenceRearrange(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _sequence_array, _sequence_length);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the sequence rearrange (vxExtrppNode_SequenceRearrange) node failed: "+ TOSTR(status))
}

void SequenceRearrangeNode::init(unsigned int* new_order, unsigned int sequence_length)
{
    _sequence_length = sequence_length;
    _new_order.resize(_sequence_length);
    std::copy(new_order, new_order + _sequence_length, _new_order.begin());
    std::cerr<<"\n new sequence order of sequence length"<<_sequence_length;
    for(int i =0 ; i < _sequence_length; i++)
    {
        std::cerr<<"\n  "<<new_order[i];
    }

}

void SequenceRearrangeNode::update_node()
{
}
