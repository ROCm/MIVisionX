#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"


class JitterNode : public Node
{
public:
    JitterNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    JitterNode() = delete;
    void create(std::shared_ptr<Graph> graph) override ;
    void init(int kernel_size);
    void init(IntParam *kernel_size);
    void update_parameters() override;

private:
    ParameterVX<int> _kernel_size;
    constexpr static int   KERNEL_SIZE [2] =  {2, 5};

    std::vector<vx_uint32> _width, _height;
    vx_array _width_array ,_height_array;
    void update_dimensions();
};
