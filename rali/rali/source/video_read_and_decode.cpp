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


#include <iterator>
#include <cstring>
#include "video_decoder_factory.h"
#include "video_read_and_decode.h"

namespace filesys = boost::filesystem;

std::tuple<VideoDecoder::ColorFormat, unsigned >
video_interpret_color_format(RaliColorFormat color_format )
{
    switch (color_format) {
        case RaliColorFormat::RGB24:
            return  std::make_tuple(VideoDecoder::ColorFormat::RGB, 3);

        case RaliColorFormat::BGR24:
            return  std::make_tuple(VideoDecoder::ColorFormat::BGR, 3);

        case RaliColorFormat::U8:
            return  std::make_tuple(VideoDecoder::ColorFormat::GRAY, 1);

        default:
            throw std::invalid_argument("Invalid color format\n");
    }
}

Timing
VideoReadAndDecode::timing()
{
    Timing t;
    t.image_decode_time = _decode_time.get_timing();
    t.image_read_time = _file_load_time.get_timing();
    t.shuffle_time = _reader->get_shuffle_time();
    return t;
}

VideoReadAndDecode::VideoReadAndDecode():
    _file_load_time("FileLoadTime", DBG_TIMING ),
    _decode_time("DecodeTime", DBG_TIMING)
{
}

VideoReadAndDecode::~VideoReadAndDecode()
{
    _reader = nullptr;
    _video_decoder.clear();
}

void
VideoReadAndDecode::create(ReaderConfig reader_config, VideoDecoderConfig decoder_config, int batch_size)
{

    std::cerr<<"\n ********************** VideoReadAndDecode::create ***************************** ";
    _sequence_length = reader_config.get_sequence_length();
    _video_count = reader_config.get_video_count();
    _batch_size = batch_size;
    std::cerr<<"\n _sequence_length ::"<<_sequence_length;
    std::cerr<<"\n _video_count:: "<<_video_count;
    std::cerr<<"\n batchsize :: "<<_batch_size;

    _compressed_buff.resize(1);  
    _video_decoder.resize(1);


    _video_names.resize(_sequence_length);
    _actual_decoded_width.resize(_sequence_length);
    _actual_decoded_height.resize(_sequence_length);
    _original_height.resize(_sequence_length);
    _original_width.resize(_sequence_length);

    _video_frame_count.resize(_video_count);
    // std::cerr << "\n Frame count :: " <<  reader_config.get_frame_count();
    std::vector<size_t> frames_count_list;
    frames_count_list.resize(_video_count);
    frames_count_list = reader_config.get_frame_count();
 
    _video_decoder_config = decoder_config;

    // get the width and height for every video _actual_decoded & original
    // fill the _video_frame_start_idx & _video_idx  based on sequence length and frame count
    // shuffle both _video_frame_start_idx & _video_idx ( can do this later)
    //for sample test
    //_video_frame_count[3] = {30, 25, 54};

    for(int i = 0; i < _video_count; i++)
    {
        _video_frame_count[i] = frames_count_list[i];
        int count_sequence = 0;
        std::cerr << "\n Frames per video : " << _video_frame_count[i];
        int loop_index;
        if(_video_frame_count[i] % _sequence_length == 0)
            loop_index = (_video_frame_count[i] / _sequence_length) - 1;
        else
            loop_index = _video_frame_count[i] / _sequence_length;
        for(int j = 0; j <= loop_index; j++)
        {
            _video_frame_start_idx.push_back(count_sequence);
            _video_idx.push_back(i);
            count_sequence = count_sequence + _sequence_length;
            // std::map<int, int>
        }
    }

    _index_start_frame = 0;
    /*
    std::cerr << "\nSize : " << _video_frame_start_idx.size();
    for(int i = 0; i < _video_frame_start_idx.size(); i++)
    {
        std::cerr << "\n Video start_idx : " << _video_frame_start_idx[i];
        std::cerr << "\t Video idx : " << _video_idx[i] << "\n";
    }
    */

    _compressed_buff[0].resize(MAX_COMPRESSED_SIZE); // If we don't need MAX_COMPRESSED_SIZE we can remove this & resize in load module
    _video_decoder[0] = create_video_decoder(decoder_config);
    
    _reader = create_reader(reader_config);
    std::cerr<<"\n=== The reader config is created  ====\n";
}

void
VideoReadAndDecode::reset()
{
    // TODO: Reload images from the folder if needed
    _reader->reset();
}

size_t
VideoReadAndDecode::count()
{
    return _reader->count();
}


VideoLoaderModuleStatus
VideoReadAndDecode::load(unsigned char* buff,
                         std::vector<std::string>& names,
                         const size_t max_decoded_width,
                         const size_t max_decoded_height,
                         std::vector<uint32_t> &roi_width,
                         std::vector<uint32_t> &roi_height,
                         std::vector<uint32_t> &actual_width,
                         std::vector<uint32_t> &actual_height,
                         RaliColorFormat output_color_format,
                         bool decoder_keep_original )
{
    if(max_decoded_width == 0 || max_decoded_height == 0 )
        THROW("Zero image dimension is not valid")
    if(!buff)
        THROW("Null pointer passed as output buffer")
    if(_reader->count() < _batch_size)
        return VideoLoaderModuleStatus::NO_MORE_DATA_TO_READ;
    // load images/frames from the disk and push them as a large image onto the buff
    unsigned file_counter = 0;
    const auto ret = video_interpret_color_format(output_color_format);
    const VideoDecoder::ColorFormat decoder_color_format = std::get<0>(ret);
    const unsigned output_planes = std::get<1>(ret);
    const bool keep_original = decoder_keep_original; // check and remove

    // Decode with the height and size equal to a single image
    // File read is done serially since I/O parallelization does not work very well.
    _file_load_time.start();// Debug timing

    size_t fsize = 1280 * 720 * 3;
    if (fsize == 0) {
        WRN("Opened file " + _reader->id() + " of size 0");
    }

    _actual_read_size = _reader->read(_compressed_buff[file_counter].data(), fsize);
    for(int s = 0; s < _sequence_length; s++)
    {
        _video_names[s] = _reader->id();
    }

    _reader->close();
    _compressed_image_size = fsize;

    _file_load_time.end();// Debug timing
    const size_t image_size = max_decoded_width * max_decoded_height * output_planes * sizeof(unsigned char);

    _decompressed_buff_ptrs = buff;

    _decode_time.start();// Debug timing
#pragma omp parallel for num_threads(_batch_size)  // default(none) TBD: option disabled in Ubuntu 20.04
    
    for(size_t i= 0; i < 1; i++)
    {

        int original_width, original_height;

        for(int s = 0; s < _sequence_length; s++)
        {
            _actual_decoded_width[s] = max_decoded_width;
            _actual_decoded_height[s] = max_decoded_height;
            _original_height[s] = original_height;
            _original_width[s]  = original_width;
        }

        index = (_index_start_frame >= _video_idx.size()) ? _index_start_frame - _video_idx.size() : _index_start_frame;
        if(_video_decoder[i]->Decode(_decompressed_buff_ptrs, (_video_names[_video_idx[index]]).c_str(), _video_frame_start_idx[index], _sequence_length) != VideoDecoder::Status::OK)
        {
            continue;
        }
    }

    for(size_t i = 0; i < _sequence_length ; i++)
    {
        names[i] = _video_names[i];
        roi_width[i] = _actual_decoded_width[i];
        roi_height[i] = _actual_decoded_height[i];
        actual_width[i] = _original_width[i];
        actual_height[i] = _original_height[i];
    }

    ++ _index_start_frame; 
    _decode_time.end();// Debug timing

    return VideoLoaderModuleStatus::OK;
}
