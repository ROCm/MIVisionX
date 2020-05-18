#include <vx_ext_rpp.h>
#include <node_pixelate.h>
#include <graph.h>
#include "exception.h"

PixelateNode::PixelateNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs)
{
}

void PixelateNode::create_node()
{
    if(_node)
        return;

    _node = vxExtrppNode_PixelatebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the pixelate (vxExtrppNode_Pixelate) node failed: "+ TOSTR(status))

}

void PixelateNode::update_node()
{
}

