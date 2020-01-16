#pragma once
#include <list>
#include "node.h"
#include "parameter_vx.h"
#include "graph.h"

class RaliContrastNode : public Node
{
public:
    RaliContrastNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    RaliContrastNode() = delete;
    void create(std::shared_ptr<Graph> graph) override ;
    void init(int min, int max);
    void init(IntParam *min, IntParam * max);
    void update_parameters() override;

private:
    ParameterVX<int> _min;
    ParameterVX<int> _max;
    constexpr static int   CONTRAST_MIN_RANGE [2] = {0, 60};
    constexpr static int   CONTRAST_MAX_RANGE [2] = {90, 250};
    constexpr static unsigned CONTRAST_MIN_OVX_PARAM_IDX = 2;
    constexpr static unsigned CONTRAST_MAX_OVX_PARAM_IDX = 3;
};