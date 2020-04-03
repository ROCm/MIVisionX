#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class BlendNode : public Node
{
public:
    explicit BlendNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    BlendNode() = delete;

    void init(float ratio);
    void init(FloatParam* ratio);

protected:
    void update_node() override;
    void create_node() override;
private:
    ParameterVX<float> _ratio;
    constexpr static float RATIO_RANGE [2] = {0.1, 0.9};
};