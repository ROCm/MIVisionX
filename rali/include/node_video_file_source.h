#pragma once
#include "node.h"
#include "graph.h"

enum class DecodeMode {
    USE_HW = 0,
    USE_SW = 1
};

#ifdef RALI_VIDEO
class VideoFileNode: public Node
{
public:
    VideoFileNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs, const size_t batch_size);
    ~VideoFileNode() override
    {
    }
    VideoFileNode() = delete;
    void init(const std::string &source_path, DecodeMode decoder_mode, bool loop);
    void start_loading() override {};
protected:
    void create_node() override;
    void update_node() override {};
private:
    const static unsigned MAXIMUM_VIDEO_CONCURRENT_DECODE = 4;
    DecodeMode _decode_mode  = DecodeMode::USE_HW;
    unsigned _video_stream_count;
    std::vector<std::string> _path_to_videos;
    unsigned _batch_size;
    std::unique_ptr<Image> _interm_output = nullptr;
    std::string _source_path;
    vx_node _copy_node;
    bool _loop;
};
#endif