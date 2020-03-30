#pragma once

#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class RainNode : public Node
{
public:
    RainNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    RainNode() = delete;
    void init(float rain_value, int rain_width, int rain_height, float rain_transparency);
    void init(FloatParam *rain_value, IntParam *rain_width, IntParam *rain_height, FloatParam *rain_transparency); 
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _rain_value;
    ParameterVX<int> _rain_width;
    ParameterVX<int> _rain_height;
    ParameterVX<float> _rain_transparency;
    constexpr static float RAIN_VALUE_RANGE [2] = {0.15, 0.95};
    constexpr static int RAIN_WIDTH_RANGE [2] = {1, 2};
    constexpr static int RAIN_HEIGHT_RANGE [2] = {15, 17};
    constexpr static float RAIN_TRANSPARENCY_RANGE [2] = {0.2, 0.3};
};