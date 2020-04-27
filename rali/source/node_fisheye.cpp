#include <vx_ext_rpp.h>
#include "node_fisheye.h"
#include "exception.h"

FisheyeNode::FisheyeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs)
{
}

void FisheyeNode::create_node()
{
    if(_node)
        return;
    
    _node = vxExtrppNode_FisheyebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the fish eye (vxExtrppNode_FisheyebatchPD) node failed: "+ TOSTR(status))


}

void FisheyeNode::update_node()
{
}
