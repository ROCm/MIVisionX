#include <tuple>
#include "rali_api.h"
#include "commons.h"
#include "context.h"
#include "node_image_loader.h"
#include "node_image_loader_single_shard.h"
#include "node_video_file_source.h"
#include "image_source_evaluator.h"
#include "node_fisheye.h"
#include "node_copy.h"
std::tuple<unsigned, unsigned>
evaluate_image_data_set(RaliImageSizeEvaluationPolicy decode_size_policy, StorageType storage_type,
                        DecoderType decoder_type, const std::string &source_path)
{
    auto translate_image_size_policy = [](RaliImageSizeEvaluationPolicy decode_size_policy)
    {
        switch(decode_size_policy)
        {
            case RALI_USE_MAX_SIZE:
                return MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
            case RALI_USE_MOST_FREQUENT_SIZE:
                return MaxSizeEvaluationPolicy::MOST_FREQUENT_SIZE;
            default:
                return MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
        }
    };

    ImageSourceEvaluator source_evaluator;
    source_evaluator.set_size_evaluation_policy(translate_image_size_policy(decode_size_policy));
    if(source_evaluator.create(ReaderConfig(storage_type, source_path), DecoderConfig(decoder_type)) != ImageSourceEvaluatorStatus::OK)
        THROW("Initializing file source input evaluator failed ")
    auto max_width = source_evaluator.max_width();
    auto max_height = source_evaluator.max_height();
    if(max_width == 0 ||max_height  == 0)
        THROW("Cannot find size of the images or images cannot be accessed")

    LOG("Maximum input image dimension [ "+ TOSTR(max_width) + " x " + TOSTR(max_height)+" ] for images in "+source_path)
    return std::make_tuple(max_width, max_height);
};

auto convert_color_format = [](RaliImageColor color_format)
{
    switch(color_format){
        case RALI_COLOR_RGB24:
            return std::make_tuple(RaliColorFormat::RGB24, 3);

        case RALI_COLOR_BGR24:
            return std::make_tuple(RaliColorFormat::BGR24, 3);

        case RALI_COLOR_U8:
            return std::make_tuple(RaliColorFormat::U8, 1);

        default:
            THROW("Unsupported Image type" + TOSTR(color_format))
    }
};

auto convert_decoder_mode= [](RaliDecodeDevice decode_mode)
{
    switch(decode_mode){
        case RALI_HW_DECODE:
            return DecodeMode::USE_HW;

        case RALI_SW_DECODE:
            return DecodeMode::USE_SW;
        default:

            THROW("Unsupported decoder mode" + TOSTR(decode_mode))
    }
};

RaliImage  RALI_API_CALL
raliJpegTFRecordSource(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);

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
                                                       source_path);
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                             source_path,
                                                                             StorageType::TF_RECORD,
                                                                             DecoderType::TURBO_JPEG,
                                                                             loop,
                                                                             context->user_batch_size(),
                                                                             context->master_graph->mem_type());
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

RaliImage  RALI_API_CALL
raliJpegTFRecordSourceSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);

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
                                                       source_path);
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path,
                                                                                        StorageType::TF_RECORD,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type());
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

RaliImage  RALI_API_CALL
raliJpegFileSourceSingleShard(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned shard_id,
        unsigned shard_count,
        bool is_output,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);

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
                                                       source_path);
        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count,
                                                                                        source_path,
                                                                                        StorageType::FILE_SYSTEM,
                                                                                        DecoderType::TURBO_JPEG,
                                                                                        loop,
                                                                                        context->user_batch_size(),
                                                                                        context->master_graph->mem_type());
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

RaliImage  RALI_API_CALL
raliJpegFileSource(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path);

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                          source_path,
                                                                          StorageType::FILE_SYSTEM,
                                                                          DecoderType::TURBO_JPEG,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type());
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

RaliImage  RALI_API_CALL
raliJpegFileSourceCrop(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned internal_shard_count,
        bool is_output,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);

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
                               evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path);

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        INFO("Internal buffer size width = "+ TOSTR(width)+ " height = "+ TOSTR(height) + " depth = "+ TOSTR(num_of_planes))

        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );
        output = context->master_graph->create_loader_output_image(info);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count,
                                                                          source_path,
                                                                          StorageType::FILE_SYSTEM,
                                                                          DecoderType::TURBO_JPEG,
                                                                          loop,
                                                                          context->user_batch_size(),
                                                                          context->master_graph->mem_type());
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

RaliImage  RALI_API_CALL
raliVideoFileSource(
        RaliContext p_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        RaliDecodeDevice rali_decode_device,
        bool is_output,
        unsigned width,
        unsigned height,
        bool loop)
{

    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try
    {
#ifdef RALI_VIDEO
        if(width == 0 || height == 0)
        {
            THROW("Invalid video input width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(width) + " x " + TOSTR(height));
        }

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);
        auto decoder_mode = convert_decoder_mode(rali_decode_device);
        auto info = ImageInfo(width, height,
                              context->internal_batch_size(),
                              num_of_planes,
                              context->master_graph->mem_type(),
                              color_format );

        output = context->master_graph->create_image(info, is_output);

        context->master_graph->add_node<VideoFileNode>({}, {output}, context->batch_size)->init( source_path,decoder_mode, loop);
        context->master_graph->set_loop(loop);
#else
        THROW("Video decoder is not enabled since amd media decoder is not present")
#endif
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;

}

RaliStatus RALI_API_CALL
raliResetLoaders(RaliContext p_context)
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
        return RALI_RUNTIME_ERROR;
    }
    return RALI_OK;
}
