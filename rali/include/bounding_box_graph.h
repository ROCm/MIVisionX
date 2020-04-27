#pragma once
#include "meta_data_graph.h"
class BoundingBoxGraph : public MetaDataGraph
{
public:
    void process() override {};
    void build() override {};
    MetaDataBatch * cropMirrorNormalize(MetaDataBatch *input, int w, int h, int, int x, int y, IntParam* mirror) override { return input; };
    MetaDataBatch * crop_resize(MetaDataBatch *input, unsigned dest_width, unsigned dest_height,
                                bool is_output,
                                FloatParam* area,
                                FloatParam* x_center_drift,
                                FloatParam* y_center_drift ) override  { return input; };

    MetaDataBatch * resize(MetaDataBatch * input, unsigned dest_width, unsigned dest_height) override  { return input; };
private:
};

