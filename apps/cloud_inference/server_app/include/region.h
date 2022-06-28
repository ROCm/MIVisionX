#ifndef YOLOREGION_H
#define YOLOREGION_H

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

struct box
{
    float x, y, w, h;
};

struct rect
{
    float left, top, right, bottom;
};

typedef struct _ObjectBB
{
    float x, y, w, h;
    float confidence;
    int   label;
} ObjectBB;


class CYoloRegion
{
public:
    CYoloRegion();
    ~CYoloRegion();

    void Initialize(int c, int h, int w, int size);
    int GetObjectDetections(float* in_data, const float *biases, int c, int h, int w,
                               int classes, int imgw, int imgh,
                               float thresh, float nms_thresh,
                               int blockwd,
                               std::vector<ObjectBB> &objects);
private:
    int Nb;              // number of bounding boxes
    bool initialized;
    int  frameNum;
    unsigned int outputSize;
    int totalObjectsPerClass;
    float *output;
    std::vector<box> boxes;

    // private member functions
    void Reshape(float *input, float *output, int n, int size);
    float Sigmoid(float x);
    void SoftmaxRegion(float *input, int classes, float *output);
    int argmax(float *a, int n);
    float box_iou(box a, box b);              // intersection over union
};

#endif // YOLOREGION_H

