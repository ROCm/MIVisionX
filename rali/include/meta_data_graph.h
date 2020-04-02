#pragma once
#include "meta_data.h"
#include "parameter_factory.h"

class MetaDataGraph
{
public:
    virtual ~MetaDataGraph()= default;
    virtual void process() = 0;
    virtual void build() = 0;
    virtual MetaDataBatch * cropMirrorNormalize(MetaDataBatch *input, int w, int h, int, int x, int y, IntParam* mirror) = 0;
    virtual MetaDataBatch * crop_resize(MetaDataBatch *input, unsigned dest_width, unsigned dest_height,
                                        bool is_output,
                                        FloatParam* area,
                                        FloatParam* x_center_drift,
                                        FloatParam* y_center_drift )  = 0;

    virtual MetaDataBatch* resize(MetaDataBatch* input, unsigned dest_width, unsigned dest_height) = 0;
};

