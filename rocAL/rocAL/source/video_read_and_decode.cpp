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

#include "video_decoder_factory.h"
#include "video_read_and_decode.h"

namespace filesys = boost::filesystem;

#ifdef RALI_VIDEO
std::tuple<VideoDecoder::ColorFormat, unsigned, AVPixelFormat>
video_interpret_color_format(RaliColorFormat color_format)
{
    switch (color_format)
    {
        case RaliColorFormat::RGB24:
            return std::make_tuple(VideoDecoder::ColorFormat::RGB, 3, AV_PIX_FMT_RGB24);

        case RaliColorFormat::BGR24:
            return std::make_tuple(VideoDecoder::ColorFormat::BGR, 3, AV_PIX_FMT_BGR24);

        case RaliColorFormat::U8:
            return std::make_tuple(VideoDecoder::ColorFormat::GRAY, 1, AV_PIX_FMT_GRAY8);

        default:
            throw std::invalid_argument("Invalid color format\n");
    }
}

Timing
VideoReadAndDecode::timing()
{
    Timing t;
    t.video_decode_time = _decode_time.get_timing();
    t.video_read_time = _file_load_time.get_timing();
    t.shuffle_time = _video_reader->get_shuffle_time();
    return t;
}

VideoReadAndDecode::VideoReadAndDecode() : _file_load_time("FileLoadTime", DBG_TIMING),
                                           _decode_time("DecodeTime", DBG_TIMING)
{
}

VideoReadAndDecode::~VideoReadAndDecode()
{
    _video_reader = nullptr;
    _video_decoder.clear();
}

void VideoReadAndDecode::create(VideoReaderConfig reader_config, VideoDecoderConfig decoder_config, int batch_size)
{
    _sequence_length = reader_config.get_sequence_length();
    _stride = reader_config.get_frame_stride();
    _video_prop = reader_config.get_video_properties();
    _video_count = _video_prop.videos_count;
    _frame_rate = _video_prop.frame_rate;
    _batch_size = batch_size;
    set_video_process_count(_video_count);
    _video_decoder.resize(_video_process_count);
    _video_names = _video_prop.video_file_names;
    _sequence_count = _batch_size / _sequence_length;
    _decompressed_buff_ptrs.resize(_sequence_count);
    _actual_decoded_width.resize(_sequence_count);
    _actual_decoded_height.resize(_sequence_count);
    _video_decoder_config = decoder_config;

    // Initialize the ffmpeg context once for the video files.
    size_t i = 0;
    for (; i < _video_process_count; i++)
    {
        _video_decoder[i] = create_video_decoder(decoder_config);
        std::vector<std::string> substrings;
        char delim = '#';
        substring_extraction(_video_names[i], delim, substrings);
        video_map video_instance;
        video_instance._video_map_idx = atoi(substrings[0].c_str());
        video_instance._is_decoder_instance = true;
        if (_video_decoder[i]->Initialize(substrings[1].c_str()) != VideoDecoder::Status::OK)
            video_instance._is_decoder_instance = false;
        _video_file_name_map.insert(std::pair<std::string, video_map>(_video_names[i], video_instance));
    }
    if (_video_process_count != _video_count)
    {
        while (i < _video_count)
        {
            std::vector<std::string> substrings;
            char delim = '#';
            substring_extraction(_video_names[i], delim, substrings);
            video_map video_instance;
            video_instance._video_map_idx = atoi(substrings[0].c_str());
            video_instance._is_decoder_instance = false;
            _video_file_name_map.insert(std::pair<std::string, video_map>(_video_names[i], video_instance));
            i++;
        }
    }
    _video_reader = create_video_reader(reader_config);
}

void VideoReadAndDecode::reset()
{
    _video_reader->reset();
}

size_t
VideoReadAndDecode::count()
{
    return _video_reader->count_items();
}

float VideoReadAndDecode::convert_framenum_to_timestamp(size_t frame_number)
{
    float timestamp;
    timestamp = (float)frame_number / _frame_rate;
    return timestamp;
}

void VideoReadAndDecode::decode_sequence(size_t sequence_index)
{
    if (_video_decoder[_sequence_video_idx[sequence_index]]->Decode(_decompressed_buff_ptrs[sequence_index], _sequence_start_frame_num[sequence_index], _sequence_length, _stride,
                                                                    _max_decoded_width, _max_decoded_height, _max_decoded_stride, _out_pix_fmt) == VideoDecoder::Status::OK)
    {
        _actual_decoded_width[sequence_index] = _max_decoded_width;
        _actual_decoded_height[sequence_index] = _max_decoded_height;
    }
}

VideoLoaderModuleStatus
VideoReadAndDecode::load(unsigned char *buff,
                         std::vector<std::string> &names,
                         const size_t max_decoded_width,
                         const size_t max_decoded_height,
                         std::vector<uint32_t> &roi_width,
                         std::vector<uint32_t> &roi_height,
                         std::vector<uint32_t> &actual_width,
                         std::vector<uint32_t> &actual_height,
                         std::vector<std::vector<size_t>> &sequence_start_framenum_vec,
                         std::vector<std::vector<std::vector<float>>> &sequence_frame_timestamps_vec,
                         RaliColorFormat output_color_format)
{
    if (max_decoded_width == 0 || max_decoded_height == 0)
        THROW("Zero image dimension is not valid")
    if (!buff)
        THROW("Null pointer passed as output buffer")
    if (_video_reader->count_items() < _sequence_count)
        return VideoLoaderModuleStatus::NO_MORE_DATA_TO_READ;
    std::vector<size_t> sequence_start_framenum;
    std::vector<std::vector<float>> sequence_frame_timestamps;
    sequence_start_framenum.resize(_sequence_count);
    sequence_frame_timestamps.resize(_sequence_count);
    for (size_t it = 0; it < (_sequence_count); it++)
        sequence_frame_timestamps[it].resize(_sequence_length);
    const auto ret = video_interpret_color_format(output_color_format);
    const unsigned output_planes = std::get<1>(ret);
    _out_pix_fmt = std::get<2>(ret);
    const size_t image_size = max_decoded_width * max_decoded_height * output_planes * sizeof(unsigned char);
    _max_decoded_width = max_decoded_width;
    _max_decoded_height = max_decoded_height;
    _max_decoded_stride = max_decoded_width * output_planes;

    _file_load_time.start(); // Debug timing

    std::vector<size_t> sequential_decode_sequences;
    std::vector<size_t> parallel_decode_sequences;
    std::vector<int> video_index;
    size_t parallel_sequence_count = 0;
    _sequence_start_frame_num.resize(_sequence_count);
    _sequence_video_path.resize(_sequence_count);
    for (size_t i = 0; i < _sequence_count; i++)
    {
        auto sequence_info = _video_reader->get_sequence_info();
        _sequence_start_frame_num[i] = sequence_info.start_frame_number;
        _sequence_video_path[i] = sequence_info.video_file_name;
        _decompressed_buff_ptrs[i] = buff + (i * image_size * _sequence_length);

        // Check if the video file is already initialized otherwise use an existing decoder instance to initialize the video 
        // std::cerr << "\nThe source video is " << _sequence_video_path[i] << " MAP : "<<_video_file_name_map.find(_sequence_video_path[i])->second._video_map_idx << "\tThe start index is : " << _sequence_start_frame_num[i] << "\n";
        std::map<std::string, video_map>::iterator itr = _video_file_name_map.find(_sequence_video_path[i]);
        if (itr->second._is_decoder_instance == false)
        {
            std::map<std::string, video_map>::iterator temp_itr;
            for (temp_itr = _video_file_name_map.begin(); temp_itr != _video_file_name_map.end(); ++temp_itr)
            {
                if (temp_itr->second._is_decoder_instance == true)
                {
                    int video_idx = temp_itr->second._video_map_idx;
                    if ((std::count(_sequence_video_idx.begin(), _sequence_video_idx.end(), video_idx) >= 1) && (_sequence_video_idx.size() != 0))
                        continue;
                    std::vector<std::string> substrings;
                    char delim = '#';
                    substring_extraction(itr->first, delim, substrings);
                    if (_video_decoder[video_idx]->Initialize(substrings[1].c_str()) == VideoDecoder::Status::OK)
                    {
                        itr->second._video_map_idx = video_idx;
                        itr->second._is_decoder_instance = true;
                    }
                    temp_itr->second._is_decoder_instance = false;
                    break;
                }
            }
        }
        if (itr->second._is_decoder_instance == false)
            continue;
        _sequence_video_idx.push_back(itr->second._video_map_idx);

        // Check if the sequences are from same or different video file 
        video_index.push_back(_video_file_name_map[_sequence_video_path[i]]._video_map_idx);
        if (std::count(video_index.begin(), video_index.end(), video_index[i]) > 1)
        {
            sequential_decode_sequences.push_back(i);
        }
        else
        {
            parallel_decode_sequences.push_back(i);
            parallel_sequence_count++;
        }
    }
    if (parallel_sequence_count == 1)
    {
        sequential_decode_sequences.push_back(parallel_decode_sequences.back());
        parallel_decode_sequences.clear();
    }

    _file_load_time.end(); // Debug timing

    _decode_time.start(); // Debug timing

    for (size_t i = 0; i < sequential_decode_sequences.size(); i++)
        decode_sequence(sequential_decode_sequences[i]);

    // Initialize threads to decode
    std::vector<std::thread> decode_threads;
    for (size_t i = 0; i < parallel_decode_sequences.size(); i++)
        decode_threads.push_back(std::thread(&VideoReadAndDecode::decode_sequence, this, parallel_decode_sequences[i]));

    for (auto &th : decode_threads)
        th.join();

    _decode_time.end(); // Debug timing

    for (size_t i = 0; i < _sequence_count; i++)
    {
        std::vector<std::string> substrings1, substrings2;
        char delim = '/';
        substring_extraction(_sequence_video_path[i], delim, substrings1);
        std::string file_name = substrings1[substrings1.size() - 1];
        delim = '#';
        substring_extraction(_sequence_video_path[i], delim, substrings2);
        std::string video_idx = substrings2[0];
        sequence_start_framenum[i] = _sequence_start_frame_num[i];
        for (size_t s = 0; s < _sequence_length; s++)
        {
            sequence_frame_timestamps[i][s] = convert_framenum_to_timestamp(_sequence_start_frame_num[i] + (s * _stride));
            roi_width[(i * _sequence_length) + s] = _actual_decoded_width[i];
            roi_height[(i * _sequence_length) + s] = _actual_decoded_height[i];
        }
        names[i] = video_idx + "#" + file_name + "_" + std::to_string(_sequence_start_frame_num[i]);
    }
    sequence_start_framenum_vec.insert(sequence_start_framenum_vec.begin(), sequence_start_framenum);
    sequence_frame_timestamps_vec.insert(sequence_frame_timestamps_vec.begin(), sequence_frame_timestamps);
    _sequence_start_frame_num.clear();
    _sequence_video_path.clear();
    _sequence_video_idx.clear();
    return VideoLoaderModuleStatus::OK;
}
#endif
