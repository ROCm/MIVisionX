/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <tuple>
#include <assert.h>
#include <boost/filesystem.hpp>
#ifdef ROCAL_VIDEO
#include "node_video_loader.h"
#include "node_video_loader_single_shard.h"
#endif
#include "rocal_api.h"
#include "commons.h"
#include "context.h"
#include "node_image_loader.h"
#include "node_image_loader_single_shard.h"
#include "node_cifar10_loader.h"
#include "image_source_evaluator.h"
#include "node_fisheye.h"
#include "node_copy.h"
#include "node_fused_jpeg_crop.h"
#include "node_fused_jpeg_crop_single_shard.h"
#include "node_resize.h"
#include "meta_node_resize.h"

namespace filesys = boost::filesystem;

std::tuple<unsigned, unsigned>
evaluate_image_data_set(RocalImageSizeEvaluationPolicy decode_size_policy, StorageType storage_type,
                        DecoderType decoder_type, const std::string &source_path, const std::string &json_path)
{
    auto translate_image_size_policy = [](RocalImageSizeEvaluationPolicy decode_size_policy)
    {
        switch(decode_size_policy)
        {
            case ROCAL_USE_MAX_SIZE:
            case ROCAL_USE_MAX_SIZE_RESTRICTED:
                return MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
            case ROCAL_USE_MOST_FREQUENT_SIZE:
                return MaxSizeEvaluationPolicy::MOST_FREQUENT_SIZE;
            default:
                return MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
        }
    };

    ImageSourceEvaluator source_evaluator;
    source_evaluator.set_size_evaluation_policy(translate_image_size_policy(decode_size_policy));
    if(source_evaluator.create(ReaderConfig(storage_type, source_path, json_path), DecoderConfig(decoder_type)) != ImageSourceEvaluatorStatus::OK)
        THROW("Initializing file source input evaluator failed ")
    auto max_width = source_evaluator.max_width();
    auto max_height = source_evaluator.max_height();
    if(max_width == 0 ||max_height  == 0)
        THROW("Cannot find size of the images or images cannot be accessed")

    LOG("Maximum input image dimension [ "+ TOSTR(max_width) + " x " + TOSTR(max_height)+" ] for images in "+source_path)
    return std::make_tuple(max_width, max_height);
};

auto convert_color_format = [](RocalImageColor color_format)
{
    switch(color_format){
        case ROCAL_COLOR_RGB24:
            return std::make_tuple(RocalColorFormat::RGB24, 3);

        case ROCAL_COLOR_BGR24:
            return std::make_tuple(RocalColorFormat::BGR24, 3);

        case ROCAL_COLOR_U8:
            return std::make_tuple(RocalColorFormat::U8, 1);

        case ROCAL_COLOR_RGB_PLANAR:
            return std::make_tuple(RocalColorFormat::RGB_PLANAR, 3);

        default:
            THROW("Unsupported Image type" + TOSTR(color_format))
    }
};

auto convert_decoder_mode= [](RocalDecodeDevice decode_mode)
{
    switch(decode_mode){
        case ROCAL_HW_DECODE:
            return DecodeMode::HW_VAAPI;

        case ROCAL_SW_DECODE:
            return DecodeMode::CPU;
        default:

            THROW("Unsupported decoder mode" + TOSTR(decode_mode))
    }
};

RocalImage  ROCAL_API_CALL
rocalJpegFileSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalDecoderType dec_type)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG; // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JEPG) decType = DecoderType::HW_JPEG_DEC;

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);



        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "",
                                                                                        StorageType::FILE_SYSTEM,
                                                                                        decType,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader(),
                                                                                        decoder_keep_original
                                                                                        );
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegFileSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalDecoderType dec_type)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG; // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JEPG) decType = DecoderType::HW_JPEG_DEC;

        if(internal_shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");

        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);



        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                          source_path, "",
                                                                          std::map<std::string, std::string>(),
                                                                          StorageType::FILE_SYSTEM,
                                                                          decType,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          context->master_graph->meta_data_reader(),
                                                                          decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalSequenceReader(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        unsigned sequence_length,
        bool is_output,
        bool shuffle,
        bool loop,
        unsigned step,
        unsigned stride)
{
    Image* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try
    {
        if(sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")
        // Set sequence batch size and batch ratio in master graph as it varies according to sequence length
        context->master_graph->set_sequence_reader_output();
        context->master_graph->set_sequence_batch_size(sequence_length);
        context->master_graph->set_sequence_batch_ratio();
        bool decoder_keep_original = true;

        // This has been introduced to support variable width and height video frames in future.
        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MAX_SIZE_RESTRICTED;

        if(internal_shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        // Set default step and stride values if 0 is passed
        step = (step == 0)? 1 : step;
        stride = (stride == 0)? 1 : stride;

        // FILE_SYSTEM is used here only to evaluate the width and height of the frames.
        auto [width, height] = evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                            source_path, "",
                                                                            std::map<std::string, std::string>(),
                                                                            StorageType::SEQUENCE_FILE_SYSTEM,
                                                                            DecoderType::TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->master_graph->sequence_batch_size(),
                                                                            context->master_graph->mem_type(),
                                                                            context->master_graph->meta_data_reader(),
                                                                            decoder_keep_original, "",
                                                                            sequence_length,
                                                                            step, stride);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalSequenceReaderSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        unsigned sequence_length,
        bool is_output,
        bool shuffle,
        bool loop,
        unsigned step,
        unsigned stride)
{
    Image* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try
    {
        if(sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")
        // Set sequence batch size and batch ratio in master graph as it varies according to sequence length
        context->master_graph->set_sequence_reader_output();
        context->master_graph->set_sequence_batch_size(sequence_length);
        context->master_graph->set_sequence_batch_ratio();
        bool decoder_keep_original = true;

        // This has been introduced to support variable width and height video frames in future.
        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MAX_SIZE_RESTRICTED;

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        // Set default step and stride values if 0 is passed
        step = (step == 0)? 1 : step;
        stride = (stride == 0)? 1 : stride;

        // FILE_SYSTEM is used here only to evaluate the width and height of the frames.
        auto [width, height] = evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "",
                                                                                        StorageType::SEQUENCE_FILE_SYSTEM,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->master_graph->sequence_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader(),
                                                                                        decoder_keep_original,
                                                                                        std::map<std::string, std::string>(),
                                                                                        sequence_length,
                                                                                        step, stride);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegCaffe2LMDBRecordSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(internal_shard_count < 1 )
            THROW("internal shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);


        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "",
                                                                             std::map<std::string, std::string>(),
                                                                             StorageType::CAFFE2_LMDB_RECORD,
                                                                             DecoderType::TURBO_JPEG,
                                                                             shuffle,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(),
                                                                             context->master_graph->meta_data_reader(),
                                                                             decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegCaffe2LMDBRecordSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);


        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "",
                                                                                        StorageType::CAFFE2_LMDB_RECORD,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader(),
                                                                                        decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegCaffeLMDBRecordSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(internal_shard_count < 1 )
            THROW("internal shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);


        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "",
                                                                             std::map<std::string, std::string>(),
                                                                             StorageType::CAFFE_LMDB_RECORD,
                                                                             DecoderType::TURBO_JPEG,
                                                                             shuffle,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(),
                                                                             context->master_graph->meta_data_reader(),
                                                                             decoder_keep_original);

        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegCaffeLMDBRecordSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);


        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "",
                                                                                        StorageType::CAFFE_LMDB_RECORD,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader(),
                                                                                        decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}


RocalImage  ROCAL_API_CALL
rocalJpegCaffeLMDBRecordSourcePartialSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        std::vector<double>& area_factor,
        std::vector<double>& aspect_ratio,
        unsigned num_attempts,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalFloatParam p_x_drift_factor,
        RocalFloatParam p_y_drift_factor )
{
    Image* output = nullptr;
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        // bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::FUSED_TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                            source_path, "",
                                                                            StorageType::CAFFE_LMDB_RECORD,
                                                                            DecoderType::FUSED_TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type(),
                                                                            context->master_graph->meta_data_reader(),
                                                                            num_attempts, area_factor, aspect_ratio, x_drift_factor, y_drift_factor);


        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegCaffe2LMDBRecordSourcePartialSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        std::vector<double>& area_factor,
        std::vector<double>& aspect_ratio,
        unsigned num_attempts,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalFloatParam p_x_drift_factor,
        RocalFloatParam p_y_drift_factor )
{
    Image* output = nullptr;
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        // bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::FUSED_TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                            source_path, "",
                                                                            StorageType::CAFFE2_LMDB_RECORD,
                                                                            DecoderType::FUSED_TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type(),
                                                                            context->master_graph->meta_data_reader(),
                                                                            num_attempts, area_factor, aspect_ratio, x_drift_factor, y_drift_factor);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalMXNetRecordSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        std::vector<double>& area_factor,
        std::vector<double>& aspect_ratio,
        unsigned num_attempts,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalFloatParam p_x_drift_factor,
        RocalFloatParam p_y_drift_factor )
{
    Image* output = nullptr;
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        // bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::FUSED_TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                            source_path, "",
                                                                            StorageType::CAFFE_LMDB_RECORD,
                                                                            DecoderType::FUSED_TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type(),
                                                                            context->master_graph->meta_data_reader(),
                                                                            num_attempts, area_factor, aspect_ratio, x_drift_factor, y_drift_factor);


        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalMXNetRecordSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(internal_shard_count < 1 )
            THROW("internal shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::MXNET_RECORDIO, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);


        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "",
                                                                             std::map<std::string, std::string>(),
                                                                             StorageType::MXNET_RECORDIO,
                                                                             DecoderType::TURBO_JPEG,
                                                                             shuffle,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(),
                                                                             context->master_graph->meta_data_reader(),
                                                                             decoder_keep_original);

        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalMXNetRecordSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::MXNET_RECORDIO, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);


        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "",
                                                                                        StorageType::MXNET_RECORDIO,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader(),
                                                                                        decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegCOCOFileSource(
        RocalContext p_context,
        const char* source_path,
	    const char* json_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(internal_shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, json_path);

        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);
        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                            source_path, json_path,
                                                                            std::map<std::string, std::string>(),
                                                                            StorageType::COCO_FILE_SYSTEM,
                                                                            DecoderType::TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type(),
                                                                            context->master_graph->meta_data_reader(),
                                                                            decoder_keep_original);

        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegCOCOFileSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
	const char* json_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::TURBO_JPEG,
                                                       source_path, json_path);
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);
        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, json_path,
                                                                                        StorageType::COCO_FILE_SYSTEM,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader(),
                                                                                        decoder_keep_original);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalFusedJpegCrop(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalFloatParam p_area_factor,
        RocalFloatParam p_aspect_ratio,
        RocalFloatParam p_x_drift_factor,
        RocalFloatParam p_y_drift_factor
        )
{
    Image* output = nullptr;
    auto area_factor  = static_cast<FloatParam*>(p_area_factor);
    auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if(internal_shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, "");

        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);
        context->master_graph->add_node<FusedJpegCropNode>({}, {output})->init(internal_shard_count,
                                                                          source_path, "",
                                                                          StorageType::FILE_SYSTEM,
                                                                          DecoderType::FUSED_TURBO_JPEG,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          context->master_graph->meta_data_reader(),
                                                                          area_factor, aspect_ratio, x_drift_factor, y_drift_factor);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegCOCOFileSourcePartial(
        RocalContext p_context,
        const char* source_path,
        const char* json_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalFloatParam p_area_factor,
        RocalFloatParam p_aspect_ratio,
        RocalFloatParam p_x_drift_factor,
        RocalFloatParam p_y_drift_factor )
{
    Image* output = nullptr;
    auto area_factor  = static_cast<FloatParam*>(p_area_factor);
    auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        //bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(internal_shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, json_path);

        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<FusedJpegCropNode>({}, {output})->init(internal_shard_count,
                                                                            source_path, json_path,
                                                                            StorageType::COCO_FILE_SYSTEM,
                                                                            DecoderType::FUSED_TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type(),
                                                                            context->master_graph->meta_data_reader(),
                                                                            area_factor, aspect_ratio, x_drift_factor, y_drift_factor);

        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}


RocalImage  ROCAL_API_CALL
rocalJpegCOCOFileSourcePartialSingleShard(
        RocalContext p_context,
        const char* source_path,
        const char* json_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        std::vector<double>& area_factor,
        std::vector<double>& aspect_ratio,
        unsigned num_attempts,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalFloatParam p_x_drift_factor,
        RocalFloatParam p_y_drift_factor )
{
    Image* output = nullptr;
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        //bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, json_path);

        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                            source_path, json_path,
                                                                            StorageType::COCO_FILE_SYSTEM,
                                                                            DecoderType::FUSED_TURBO_JPEG,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type(),
                                                                            context->master_graph->meta_data_reader(),
                                                                            num_attempts, area_factor, aspect_ratio, x_drift_factor, y_drift_factor);

        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegTFRecordSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned internal_shard_count,
        bool is_output,
        const char* user_key_for_encoded,
        const char* user_key_for_filename,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        std::string user_key_for_encoded_str(user_key_for_encoded);
        std::string user_key_for_filename_str(user_key_for_filename);

        std::map<std::string, std::string> feature_key_map = {
            {"image/encoded",user_key_for_encoded_str},
            {"image/filename",user_key_for_filename_str},
        };


        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if(internal_shard_count < 1 )
            THROW("internal shard count should be bigger than 0")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::TF_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "",
                                                                             feature_key_map,
                                                                             StorageType::TF_RECORD,
                                                                             DecoderType::TURBO_JPEG,
                                                                             shuffle,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(),
                                                                             context->master_graph->meta_data_reader());
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalJpegTFRecordSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::TF_RECORD, DecoderType::TURBO_JPEG,
                                                       source_path, "");
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);
        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "",
                                                                                        StorageType::TF_RECORD,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader());
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalRawTFRecordSource(
        RocalContext p_context,
        const char* source_path,
        const char* user_key_for_encoded_str,
        const char* user_key_for_filename_str,
        RocalImageColor rocal_color_format,
        bool is_output,
        bool shuffle,
        bool loop,
        unsigned out_width,
        unsigned out_height,
        const char* record_name_prefix)
{
    Image* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    try
    {
        unsigned internal_shard_count = 1;
        std::map<std::string, std::string> feature_key_map = {
                {"image/encoded",user_key_for_encoded_str},
                {"image/filename",user_key_for_filename_str},
        };

        if(out_width == 0 || out_height == 0)
        {
            THROW("Invalid output width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height))
        }

        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);
        auto info = ImageInfo(out_width, out_height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path, "",
                                                                             feature_key_map,
                                                                             StorageType::TF_RECORD,
                                                                             DecoderType::SKIP_DECODE,
                                                                             shuffle,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(),
                                                                             context->master_graph->meta_data_reader(),
                                                                             false, record_name_prefix);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalRawTFRecordSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool shuffle,
        bool loop,
        unsigned out_width,
        unsigned out_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if((out_width == 0 || out_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height))
        }

        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);
        INFO("Internal buffer size width = "+ TOSTR(out_width)+ " height = "+ TOSTR(out_height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(out_width, out_height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path, "",
                                                                                        StorageType::TF_RECORD,
                                                                                        DecoderType::SKIP_DECODE,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type(),
                                                                                        context->master_graph->meta_data_reader());
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalFusedJpegCropSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        std::vector<double>& area_factor,
        std::vector<double>& aspect_ratio,
        unsigned num_attempts,
        bool shuffle,
        bool loop,
        RocalImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height,
        RocalFloatParam p_x_drift_factor,
        RocalFloatParam p_y_drift_factor
        )
{
    Image* output = nullptr;
    auto x_drift_factor = static_cast<FloatParam*>(p_x_drift_factor);
    auto y_drift_factor = static_cast<FloatParam*>(p_y_drift_factor);
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, "");

        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);
        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                          source_path, "",
                                                                          StorageType::FILE_SYSTEM,
                                                                          DecoderType::FUSED_TURBO_JPEG,
                                                                          shuffle,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type(),
                                                                          context->master_graph->meta_data_reader(),
                                                                          num_attempts, area_factor, aspect_ratio, x_drift_factor, y_drift_factor);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalImage  ROCAL_API_CALL
rocalVideoFileSource(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        RocalDecodeDevice rocal_decode_device,
        unsigned internal_shard_count,
        unsigned sequence_length,
        bool shuffle,
        bool is_output,
        bool loop,
        unsigned step,
        unsigned stride,
        bool file_list_frame_num)
{
    Image* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try
    {
#ifdef ROCAL_VIDEO
        if(sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")
        // Set video loader flag in master_graph
        context->master_graph->set_video_loader_flag();

        // Set default step and stride values if 0 is passed
        step = (step == 0)? sequence_length : step;
        stride = (stride == 0)? 1 : stride;

        VideoProperties video_prop;
        VideoDecoderType decoder_type;
        find_video_properties(video_prop, source_path, file_list_frame_num);
        if(rocal_decode_device == RocalDecodeDevice::ROCAL_HW_DECODE)
            decoder_type = VideoDecoderType::FFMPEG_HARDWARE_DECODE;
        else
            decoder_type = VideoDecoderType::FFMPEG_SOFTWARE_DECODE;
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);
        auto decoder_mode = convert_decoder_mode(rocal_decode_device);
        auto info = ImageInfo(video_prop.width, video_prop.height,
                              context->internal_batch_size() * sequence_length,
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );

        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<VideoLoaderNode>({}, {output})->init(internal_shard_count,
                                                                            source_path,
                                                                            VideoStorageType::VIDEO_FILE_SYSTEM,
                                                                            decoder_type,
                                                                            decoder_mode,
                                                                            sequence_length,
                                                                            step,
                                                                            stride,
                                                                            video_prop,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;

}

RocalImage  ROCAL_API_CALL
rocalVideoFileSourceSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        RocalDecodeDevice rocal_decode_device,
        unsigned shard_id,
        unsigned shard_count,
        unsigned sequence_length,
        bool shuffle,
        bool is_output,
        bool loop,
        unsigned step,
        unsigned stride,
        bool file_list_frame_num)
{
    Image* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try
    {
#ifdef ROCAL_VIDEO
        if(sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")
        // Set video loader flag in master_graph
        context->master_graph->set_video_loader_flag();

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        // Set default step and stride values if 0 is passed
        step = (step == 0)? sequence_length : step;
        stride = (stride == 0)? 1 : stride;

        VideoProperties video_prop;
        VideoDecoderType decoder_type;
        find_video_properties(video_prop, source_path, file_list_frame_num);
        if(rocal_decode_device == RocalDecodeDevice::ROCAL_HW_DECODE)
            decoder_type = VideoDecoderType::FFMPEG_HARDWARE_DECODE;
        else
            decoder_type = VideoDecoderType::FFMPEG_SOFTWARE_DECODE;
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);
        auto decoder_mode = convert_decoder_mode(rocal_decode_device);
        auto info = ImageInfo(video_prop.width, video_prop.height,
                              context->internal_batch_size() * sequence_length,
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );

        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<VideoLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path,
                                                                                        VideoStorageType::VIDEO_FILE_SYSTEM,
                                                                                        decoder_type,
                                                                                        decoder_mode,
                                                                                        sequence_length,
                                                                                        step,
                                                                                        stride,
                                                                                        video_prop,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;

}

RocalImage  ROCAL_API_CALL
rocalVideoFileResize(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        RocalDecodeDevice rocal_decode_device,
        unsigned internal_shard_count,
        unsigned sequence_length,
        unsigned dest_width,
        unsigned dest_height,
        bool shuffle,
        bool is_output,
        bool loop,
        unsigned step,
        unsigned stride,
        bool file_list_frame_num)
{
    Image* resize_output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return resize_output;
    }

    auto context = static_cast<Context*>(p_context);
    try
    {
#ifdef ROCAL_VIDEO
        if(sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")
        // Set video loader flag in master_graph
        context->master_graph->set_video_loader_flag();

        if(dest_width == 0 || dest_height == 0)
            THROW("Invalid dest_width/dest_height values passed as input")

        // Set default step and stride values if 0 is passed
        step = (step == 0)? sequence_length : step;
        stride = (stride == 0)? 1 : stride;

        VideoProperties video_prop;
        VideoDecoderType decoder_type;
        find_video_properties(video_prop, source_path, file_list_frame_num);
        if(rocal_decode_device == RocalDecodeDevice::ROCAL_HW_DECODE)
            decoder_type = VideoDecoderType::FFMPEG_HARDWARE_DECODE;
        else
            decoder_type = VideoDecoderType::FFMPEG_SOFTWARE_DECODE;
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);
        auto decoder_mode = convert_decoder_mode(rocal_decode_device);
        auto info = ImageInfo(video_prop.width, video_prop.height,
                              context->internal_batch_size() * sequence_length,
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );

        Image* output = context->master_graph->create_loader_output_image(info);
        context->master_graph->add_node<VideoLoaderNode>({}, {output})->init(internal_shard_count,
                                                                            source_path,
                                                                            VideoStorageType::VIDEO_FILE_SYSTEM,
                                                                            decoder_type,
                                                                            decoder_mode,
                                                                            sequence_length,
                                                                            step,
                                                                            stride,
                                                                            video_prop,
                                                                            shuffle,
                                                                            loop,
                                                                            context->user_batch_size(),
                                                                            context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if(dest_width != video_prop.width && dest_height != video_prop.height)
        {
            // For the resize node, user can create an image with a different width and height
            ImageInfo output_info = info;
            output_info.width(dest_width);
            output_info.height(dest_height);

            resize_output = context->master_graph->create_image(output_info, false);

            // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
            resize_output->reset_image_roi();

            std::shared_ptr<ResizeNode> resize_node =  context->master_graph->add_node<ResizeNode>({output}, {resize_output});
            if (context->master_graph->meta_data_graph())
                context->master_graph->meta_add_node<ResizeMetaNode,ResizeNode>(resize_node);

            if(is_output)
            {
                auto actual_output = context->master_graph->create_image(output_info, is_output);
                context->master_graph->add_node<CopyNode>({resize_output}, {actual_output});
            }
        }
        else{
            if(is_output)
            {
                auto actual_output = context->master_graph->create_image(info, is_output);
                context->master_graph->add_node<CopyNode>({output}, {actual_output});
            }
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return resize_output;
}

RocalImage  ROCAL_API_CALL
rocalVideoFileResizeSingleShard(
        RocalContext p_context,
        const char* source_path,
        RocalImageColor rocal_color_format,
        RocalDecodeDevice rocal_decode_device,
        unsigned shard_id,
        unsigned shard_count,
        unsigned sequence_length,
        unsigned dest_width,
        unsigned dest_height,
        bool shuffle,
        bool is_output,
        bool loop,
        unsigned step,
        unsigned stride,
        bool file_list_frame_num)
{
    Image* resize_output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return resize_output;
    }

    auto context = static_cast<Context*>(p_context);
    try
    {
#ifdef ROCAL_VIDEO
        if(sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")
        // Set video loader flag in master_graph
        context->master_graph->set_video_loader_flag();

        if(shard_count < 1 )
            THROW("Shard count should be bigger than 0")

        if(shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if(dest_width == 0 || dest_height == 0)
            THROW("Invalid dest_width/dest_height values passed as input")

        // Set default step and stride values if 0 is passed
        step = (step == 0)? sequence_length : step;
        stride = (stride == 0)? 1 : stride;

        VideoProperties video_prop;
        VideoDecoderType decoder_type;
        find_video_properties(video_prop, source_path, file_list_frame_num);
        if(rocal_decode_device == RocalDecodeDevice::ROCAL_HW_DECODE)
            decoder_type = VideoDecoderType::FFMPEG_HARDWARE_DECODE;
        else
            decoder_type = VideoDecoderType::FFMPEG_SOFTWARE_DECODE;
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);
        auto decoder_mode = convert_decoder_mode(rocal_decode_device);
        auto info = ImageInfo(video_prop.width, video_prop.height,
                              context->internal_batch_size() * sequence_length,
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );

        Image* output = context->master_graph->create_loader_output_image(info);
        context->master_graph->add_node<VideoLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path,
                                                                                        VideoStorageType::VIDEO_FILE_SYSTEM,
                                                                                        decoder_type,
                                                                                        decoder_mode,
                                                                                        sequence_length,
                                                                                        step,
                                                                                        stride,
                                                                                        video_prop,
                                                                                        shuffle,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if(dest_width != video_prop.width && dest_height != video_prop.height)
        {
            // For the resize node, user can create an image with a different width and height
            ImageInfo output_info = info;
            output_info.width(dest_width);
            output_info.height(dest_height);

            resize_output = context->master_graph->create_image(output_info, false);
            // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
            resize_output->reset_image_roi();

            std::shared_ptr<ResizeNode> resize_node =  context->master_graph->add_node<ResizeNode>({output}, {resize_output});
            if (context->master_graph->meta_data_graph())
                context->master_graph->meta_add_node<ResizeMetaNode,ResizeNode>(resize_node);

            if(is_output)
            {
                auto actual_output = context->master_graph->create_image(output_info, is_output);
                context->master_graph->add_node<CopyNode>({resize_output}, {actual_output});
            }
        }
        else{
            if(is_output)
            {
                auto actual_output = context->master_graph->create_image(info, is_output);
                context->master_graph->add_node<CopyNode>({output}, {actual_output});
            }
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return resize_output;
}

// loader for CFAR10 raw data: Can be used for other raw data loaders as well
RocalImage  ROCAL_API_CALL
rocalRawCIFAR10Source(
                 RocalContext p_context,
                 const char* source_path,
                 RocalImageColor rocal_color_format,
                 bool is_output ,
                 unsigned out_width,
                 unsigned out_height,
                 const char* filename_prefix,
                 bool loop )
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {

        if(out_width == 0 || out_height == 0)
        {
            THROW("Invalid video input width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height));
        }

        auto [width, height] = std::make_tuple(out_width, out_height);
        auto [color_format, num_of_planes] = convert_color_format(rocal_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<Cifar10LoaderNode>({}, {output})->init(source_path, "",
                                                                             StorageType::UNCOMPRESSED_BINARY_DATA,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type(), filename_prefix);
        context->master_graph->set_loop(loop);

        if(is_output)
        {
            auto actual_output = context->master_graph->create_image(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}


RocalStatus ROCAL_API_CALL
rocalResetLoaders(RocalContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        context->master_graph->reset();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}
