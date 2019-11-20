#include <tuple>
#include "rali_api.h"
#include "commons.h"
#include "context.h"
#include "node_jpeg_file_source.h"
#include "image_source_evaluator.h"
#include "node_fisheye.h"
#include "node_copy.h"
std::tuple<unsigned, unsigned>
find_max_image_size (RaliImageSizeEvaluationPolicy decode_size_policy, const std::string& source_path)
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
    if(source_evaluator.create(ReaderConfig(StorageType::FILE_SYSTEM, source_path), DecoderConfig(DecoderType::TURBO_JPEG)) != ImageSourceEvaluatorStatus::OK)
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

RaliImage  RALI_API_CALL
raliJpegFileSource(
        RaliContext rali_context,
        const char* source_path,
        RaliImageColor rali_color_format,
        unsigned num_threads,
        bool is_output,
        bool loop,
        RaliImageSizeEvaluationPolicy decode_size_policy,
        unsigned max_width,
        unsigned max_height)
{
    RaliImage output = nullptr;
    try
    {
        bool use_input_dimension = (decode_size_policy == RALI_USE_USER_GIVEN_SIZE);

        if(use_input_dimension && (max_width == 0 || max_height == 0))
        {
            THROW("Invalid input max width and height");
        }
        else
        {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension? std::make_tuple(max_width, max_height):
                               find_max_image_size(decode_size_policy, source_path);

        auto [color_format, num_of_planes] = convert_color_format(rali_color_format);

        auto info = ImageInfo(width, height,
                              rali_context->batch_size,
                              num_of_planes,
                              rali_context->master_graph->mem_type(),
                              color_format );

        output = rali_context->master_graph->create_loader_output_image(info);

        rali_context->master_graph->add_node<JpegFileNode>({}, {output})->init(num_threads, source_path, loop);

        if(is_output)
        {
            auto actual_output = rali_context->master_graph->create_image(info, is_output);
            rali_context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    }
    catch(const std::exception& e)
    {
        rali_context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}


RaliStatus RALI_API_CALL
raliResetLoaders(RaliContext rali_context)
{
    try
    {
        //rali_context->master_graph->reset_loaders();
    }
    catch(const std::exception& e)
    {
        rali_context->capture_error(e.what());
        ERR(e.what())
        return RALI_RUNTIME_ERROR;
    }
    return RALI_OK;
}
