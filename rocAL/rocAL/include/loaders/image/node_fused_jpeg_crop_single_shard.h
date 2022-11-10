#pragma once
#include "node.h"
#include "image_loader_sharded.h"
#include "graph.h"
#include "parameter_factory.h"

class FusedJpegCropSingleShardNode: public Node
{
public:
#if ENABLE_HIP
    FusedJpegCropSingleShardNode(Image *output, DeviceResourcesHip device_resources);
#else
    FusedJpegCropSingleShardNode(Image *output, DeviceResources device_resources);
#endif
    ~FusedJpegCropSingleShardNode() override;

    /// \param user_shard_count shard count from user
    /// \param  user_shard_id shard id from user
    /// \param source_path Defines the path that includes the image dataset
    /// \param load_batch_count Defines the quantum count of the images to be loaded. It's usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images in multiples of the load_batch_count,
    /// for example if there are 10 images in the dataset and load_batch_count is 3, the loader repeats 2 images as if there are 12 images available.
    void init(unsigned shard_id, unsigned shard_count, const std::string &source_path, const std::string &json_path, StorageType storage_type,
              DecoderType decoder_type, bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type, std::shared_ptr<MetaDataReader> meta_data_reader,
              unsigned num_attempts, std::vector<double> &random_area, std::vector<double> &random_aspect_ratio);

    std::shared_ptr<LoaderModule> get_loader_module();
protected:
    void create_node() override {};
    void update_node() override {};
private:
    std::shared_ptr<ImageLoader> _loader_module = nullptr;
    std::vector<double> _random_area, _random_aspect_ratio;
    unsigned num_attempts;
};
