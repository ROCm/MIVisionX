#pragma once
#include "node.h"
#include "image_loader_sharded.h"
#include "graph.h"

class ImageLoaderNode: public Node
{
public:
    /// \param device_resources shard count from user

    /// internal_shard_count number of loader/decoders are created and each shard is loaded and decoded using separate and independent resources increasing the parallelism and performance.
    ImageLoaderNode(Image *output, DeviceResources device_resources);
    ~ImageLoaderNode() override;
    ImageLoaderNode() = delete;
    ///
    /// \param internal_shard_count Defines the amount of parallelism user wants for the load and decode process to be handled internally.
    /// \param source_path Defines the path that includes the image dataset
    /// \param load_batch_count Defines the quantum count of the images to be loaded. It's usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images in multiples of the load_batch_count,
    /// for example if there are 10 images in the dataset and load_batch_count is 3, the loader repeats 2 images as if there are 12 images available.
    void init(unsigned internal_shard_count, const std::string &source_path, StorageType storage_type,
              DecoderType decoder_type, bool loop, size_t load_batch_count, RaliMemType mem_type);

    std::shared_ptr<LoaderModule> get_loader_module();
protected:
    void create_node() override {};
    void update_node() override {};
private:
    std::shared_ptr<ImageLoaderSharded> _loader_module = nullptr;
};