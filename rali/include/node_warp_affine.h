#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class WarpAffineNode : public Node
{
public:
    WarpAffineNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    WarpAffineNode() = delete;
    void init(float x0, float x1, float y0, float y1, float o0, float o1);
    void init(FloatParam* x0, FloatParam* x1, FloatParam* y0, FloatParam* y1, FloatParam* o0, FloatParam* o1);
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _x0;
    ParameterVX<float> _x1;
    ParameterVX<float> _y0;
    ParameterVX<float> _y1;
    ParameterVX<float> _o0;
    ParameterVX<float> _o1;

    std::vector<float> _affine;
    vx_array _dst_roi_width,_dst_roi_height;
    vx_array _affine_array;
    constexpr static float COEFFICIENT_RANGE_0 [2] = {-0.35, 0.35};
    constexpr static float COEFFICIENT_RANGE_1 [2] = {0.65, 1.35};
    constexpr static float COEFFICIENT_RANGE_OFFSET [2] = {-10.0, 10.0};
    void update_affine_array();
};
