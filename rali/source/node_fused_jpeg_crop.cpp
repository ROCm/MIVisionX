#include "node_fused_jpeg_crop.h"
#include "exception.h"

FusedJpegCropNode::FusedJpegCropNode(Image *output, DeviceResources device_resources):
        Node({}, {output})
{
    _loader_module = std::make_shared<ImageLoaderSharded>(device_resources);
}

void FusedJpegCropNode::init(unsigned internal_shard_count, const std::string &source_path, StorageType storage_type,
                           DecoderType decoder_type, bool shuffle, bool loop, size_t load_batch_count, RaliMemType mem_type,
                           FloatParam *area_factor, FloatParam *aspect_ratio, FloatParam *x_drift, FloatParam *y_drift)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for FusedJpegCropNode, cannot initialize")
    if(internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the FusedJpegCropNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, shuffle, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_batch_count(load_batch_count);
    auto decoder_cfg = DecoderConfig(decoder_type);

    std::vector<Parameter<float>*> crop_param;
    _area_factor = ParameterFactory::instance()->create_uniform_float_rand_param(AREA_FACTOR_RANGE[0], AREA_FACTOR_RANGE[1])->core;
    _aspect_ratio = ParameterFactory::instance()->create_uniform_float_rand_param(ASPECT_RATIO_RANGE[0], ASPECT_RATIO_RANGE[1])->core;
    _x_drift = ParameterFactory::instance()->create_uniform_float_rand_param(X_DRIFT_RANGE[0], X_DRIFT_RANGE[1])->core;
    _y_drift = ParameterFactory::instance()->create_uniform_float_rand_param(Y_DRIFT_RANGE[0], Y_DRIFT_RANGE[1])->core;
    crop_param.push_back(_area_factor);
    crop_param.push_back(_aspect_ratio);
    crop_param.push_back(_x_drift);
    crop_param.push_back(_y_drift);
    decoder_cfg.set_crop_param(crop_param);
    _loader_module->initialize(reader_cfg, decoder_cfg,
             mem_type,
             _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> FusedJpegCropNode::get_loader_module()
{
    if(!_loader_module)
        WRN("FusedJpegCropNode's loader module is null, not initialized")
    return _loader_module;
}

FusedJpegCropNode::~FusedJpegCropNode()
{
    _loader_module = nullptr;
}