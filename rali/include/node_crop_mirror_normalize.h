#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "parameter_crop_resize.h"

//class RandomCropResizeParam;

/*class CropMirrorNormalizeNode : public Node
{
public:
    explicit CropMirrorNormalizeNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs, const size_t batch_size);
    CropMirrorNormalizeNode() = delete;
    void create(std::shared_ptr<Graph> graph) override;
    void init(float area, float aspect_ratio, float x_center_drift, float y_center_drift,float mean, float sdev, int mirror);
    void init(FloatParam *area, FloatParam *aspect_ratio, FloatParam *x_center_drift, FloatParam *y_center_drift,
                                                FloatParam* mean, FloatParam* sdev, IntParam* mirror);
    void update_parameters() override;
private:

    size_t _dest_width;
    size_t _dest_height;
    std::shared_ptr<RandomCropResizeParam> _crop_param;

    ParameterVX<float> _mean;
    constexpr static float MEAN_RANGE [2] = {122.0, 130.0};//{0.0, 255.0};

    ParameterVX<float> _sdev;
    constexpr static float SDEV_RANGE [2] = {0.5, 3.5};//{1.0, 125.0};
    
    ParameterVX<int> _mirror;
    constexpr static int   MIRROR_RANGE [2] =  {0, 1};

    vx_array src_width_array,src_height_array ;
    std::vector<vx_uint32> src_width ,src_height; 
    void update_dimensions();
};*/

class CropMirrorNormalizeNode : public Node 
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    CropMirrorNormalizeNode(const std::vector<Image *> &inputs,
                            const std::vector<Image *> &outputs);
    CropMirrorNormalizeNode() = delete;
    void init(int crop_h, int crop_w, float start_x, float start_y, float mean, float std_dev, IntParam *mirror);
    void update_parameters() override;
private:
    std::vector<vx_uint32>  _src_width, _src_height;
    vx_array _src_width_array, _src_height_array;
    std::vector<vx_uint32> _x1, _x2, _y1, _y2;
    std::vector<vx_float32> _mean_vx, _std_dev_vx;
    vx_array _mean_array, _std_dev_array;
    vx_array _x1_array, _x2_array, _y1_array, _y2_array;
    void update_dimensions();
    int _crop_h;
    int _crop_w;
    int _crop_d;
    float _start_x;
    float _start_y;
    float _mean; // vector of means in future 
    float _std_dev; // vector of std_devs in future
    ParameterVX<int> _mirror; // Should come from int random number generator with values 1 or 0 - Coin Flip
    constexpr static int   MIRROR_RANGE [2] =  {0, 1};
};