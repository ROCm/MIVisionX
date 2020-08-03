/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

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