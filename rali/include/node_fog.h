#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
class FogNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    FogNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    FogNode() = delete;
    void init(float fog_param);
    void init(FloatParam *fog_param);
    void update_parameters() override;

private:
    ParameterVX<float> _fog_param;
    constexpr static float FOG_VALUE_RANGE [2] = {0.2, 0.8};

    std::vector<vx_uint32> _width, _height;
    vx_array _width_array ,_height_array;
    void update_dimensions();
};


