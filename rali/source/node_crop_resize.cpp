#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_crop_resize.h"
#include "exception.h"

CropResizeNode::CropResizeNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _dest_width(_outputs[0]->info().width()),
        _dest_height(_outputs[0]->info().height_batch())
{
    _crop_param = std::make_shared<RandomCropResizeParam>(inputs[0]->info().width() , inputs[0]->info().height_single());
}

void CropResizeNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    if(_dest_width == 0 || _dest_height == 0)
        THROW("Uninitialized destination dimension")


    _node = vxExtrppNode_ResizeCrop(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _dest_width, _dest_height,
                                    _crop_param->x1, _crop_param->y1, _crop_param->x2, _crop_param->y2);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_ResizeCrop) failed: "+TOSTR(status))

    _crop_param->create_scalars(_node);
}

void CropResizeNode::update_parameters()
{
    _crop_param->update();
}
void CropResizeNode::init(float area, float x_center_drift, float y_center_drift)
{
    _crop_param->set_area_coeff(ParameterFactory::instance()->create_single_value_param(area));
    _crop_param->set_x_drift(ParameterFactory::instance()->create_single_value_param(x_center_drift));
    _crop_param->set_y_drift(ParameterFactory::instance()->create_single_value_param(y_center_drift));
}


void CropResizeNode::init(FloatParam* area, FloatParam *x_center_drift, FloatParam *y_center_drift)
{
    _crop_param->set_area_coeff(core(area));
    _crop_param->set_x_drift(core(x_center_drift));
    _crop_param->set_y_drift(core(y_center_drift));
}



