#pragma once
#include <memory>
#include <map>
#include "turbo_jpeg_decoder.h"
#include "reader_factory.h"
#include "timing_debug.h"
#include "loader_module.h"
enum class ImageSourceEvaluatorStatus
{
    OK = 0,
    UNSUPPORTED_DECODER_TYPE, 
    UNSUPPORTED_STORAGE_TYPE,
};
enum class MaxSizeEvaluationPolicy
{
    MAXIMUM_FOUND_SIZE,
    MOST_FREQUENT_SIZE
};

class ImageSourceEvaluator
{
public:
    ImageSourceEvaluatorStatus create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg);
    void find_max_dimension();
    void set_size_evaluation_policy(MaxSizeEvaluationPolicy arg);
    size_t max_width();
    size_t max_height();

private:
    class FindMaxSize
    {
    public:
        void set_policy(MaxSizeEvaluationPolicy arg) { _policy = arg; }
        void process_sample(unsigned val);
        unsigned get_max() { return _max; };
    private:
        MaxSizeEvaluationPolicy _policy = MaxSizeEvaluationPolicy::MOST_FREQUENT_SIZE;
        std::map<unsigned,unsigned> _hist;
        unsigned _max = 0;
        unsigned _max_count = 0;
    }; 
    FindMaxSize _width_max; 
    FindMaxSize _height_max;
    std::shared_ptr<Decoder> _decoder;
    std::shared_ptr<Reader> _reader;
    std::vector<unsigned char> _header_buff;
    static const size_t COMPRESSED_SIZE = 1024 * 1024; // 1 MB
};

