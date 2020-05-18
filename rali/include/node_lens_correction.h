#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"



class LensCorrectionNode : public Node
{
public:
    LensCorrectionNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    LensCorrectionNode() = delete;
    void init(float strength, float zoom);
    void init(FloatParam *strength, FloatParam *zoom);
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _strength;
    ParameterVX<float> _zoom;
    constexpr static float STRENGTH_RANGE [2] = {0.05, 3.0};
    constexpr static float   ZOOM_RANGE [2] = {1.0, 1.3};
};
