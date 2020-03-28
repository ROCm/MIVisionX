#pragma once

#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class RainNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    RainNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    RainNode() = delete;
    void init(float rain_value, int rain_width, int rain_height, float rain_transparency);
    void init(FloatParam *rain_value, IntParam *rain_width, IntParam *rain_height, FloatParam *rain_transparency); 
    void update_parameters() override;

private:
    ParameterVX<float> _rain_value;
    ParameterVX<int> _rain_width;
    ParameterVX<int> _rain_height;
    ParameterVX<float> _rain_transparency;
    std::vector<vx_uint32> _width, _height;
    vx_array _width_array ,_height_array;
    constexpr static float RAIN_VALUE_RANGE [2] = {0.15, 0.95};
    constexpr static int RAIN_WIDTH_RANGE [2] = {1, 2};
    constexpr static int RAIN_HEIGHT_RANGE [2] = {15, 17};
    constexpr static float RAIN_TRANSPARENCY_RANGE [2] = {0.2, 0.3};
    void update_dimensions();

    // constexpr static size_t RAIN_WIDTH = 1;
    // constexpr static size_t RAIN_HEIGHT = 15;
    // constexpr static float RAIN_TRANSPARENCY = 0.2;
};