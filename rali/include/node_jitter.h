#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"


class JitterNode : public Node
{
public:
    JitterNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    JitterNode() = delete;
    void init(int kernel_size);
    void init(IntParam *kernel_size);
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<int> _kernel_size;
    constexpr static int   KERNEL_SIZE [2] =  {2, 5};
};
