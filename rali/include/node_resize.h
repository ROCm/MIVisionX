#pragma once
#include "node.h"

class ResizeNode : public Node
{
public:

    void create(std::shared_ptr<Graph> graph) override ;
    ResizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ResizeNode() = delete;
    void update_parameters() override;
private:
    vx_array src_width_array,src_height_array, dst_width_array , dst_height_array ;
    std::vector<vx_uint32> src_width, src_height,dst_width ,dst_height  ;
    void update_dimensions();

};
