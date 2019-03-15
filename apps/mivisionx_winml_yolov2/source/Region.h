#pragma once

#include "Common.h"



struct Region
{
    bool initialized;
    int totalLength;
    int totalObjects;
    std::vector<float> output;
    std::vector<ibox> boxes;
    std::vector<indexsort> s;

    Region();

    void Initialize(int c, int h, int w, int size);


    void GetDetections(float* data, int c, int h, int w,
                       int classes, int imgw, int imgh,
                       float thresh, float nms,
                       int blockwd,
                       std::vector<DetectedObject> &objects);
};

