#pragma once
#include "node.h"

class ResizeNode : public Node
{
public:
    ResizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ResizeNode() = delete;
protected:
    void create_node() override;
    void update_node() override;
private:
    vx_array  _dst_roi_width , _dst_roi_height ;
};
