#include "node_cifar10_loader.h"
#include "exception.h"


Cifar10LoaderNode::Cifar10LoaderNode(Image *output, DeviceResources device_resources):
        Node({}, {output})
{
    _loader_module = std::make_shared<CIFAR10DataLoader>(device_resources);
}

void Cifar10LoaderNode::init(const std::string &source_path, StorageType storage_type,
                           bool loop, size_t load_batch_count, RaliMemType mem_type)
{
    if(!_loader_module)
        THROW("ERROR: loader module is not set for Cifar10LoaderNode, cannot initialize")
    _loader_module->set_output_image(_outputs[0]);
    // Set reader and decoder config accordingly for the Cifar10LoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, loop);
    reader_cfg.set_batch_count(load_batch_count);
    // DecoderConfig will be ignored in loader. Just passing it for api match
    _loader_module->initialize(reader_cfg, DecoderConfig(DecoderType::TURBO_JPEG),
             mem_type, _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> Cifar10LoaderNode::get_loader_module()
{
    if(!_loader_module)
        WRN("Cifar10LoaderNode's loader module is null, not initialized")
    return _loader_module;
}

Cifar10LoaderNode::~Cifar10LoaderNode()
{
    _loader_module = nullptr;
}