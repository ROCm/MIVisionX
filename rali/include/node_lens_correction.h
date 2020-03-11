#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"



class LensCorrectionNode : public Node
{
public:
    LensCorrectionNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    LensCorrectionNode() = delete;
    void create(std::shared_ptr<Graph> graph) override ;
    void init(float min, float max);
    void init(FloatParam *min, FloatParam *max);
    void update_parameters() override;

private:
    ParameterVX<float> _strength;
    ParameterVX<float> _zoom;
    constexpr static float STRENGTH_RANGE [2] = {0.05, 3.0};
    constexpr static float   ZOOM_RANGE [2] = {1.0, 1.3};
    constexpr static unsigned STRENGTH_OVX_PARAM_IDX = 2;
    constexpr static unsigned ZOOM_OVX_PARAM_IDX = 3;
};
