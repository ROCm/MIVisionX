#include "commons.h"
#include "context.h"
#include "rali_api.h"
size_t RALI_API_CALL raliGetImageWidth(RaliImage p_image)
{
    auto image = static_cast<Image*>(p_image);
    return image->info().width();
}
size_t RALI_API_CALL raliGetImageHeight(RaliImage p_image)
{
    auto image = static_cast<Image*>(p_image);
    return image->info().height_batch();
}

size_t RALI_API_CALL raliGetImagePlanes(RaliImage p_image)
{
    auto image = static_cast<Image*>(p_image);
    return image->info().color_plane_count();
}

int RALI_API_CALL raliGetOutputWidth(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->output_width();
}

int RALI_API_CALL raliGetOutputHeight(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->output_height();
}

int RALI_API_CALL raliGetOutputColorFormat(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    auto translate_color_format = [](RaliColorFormat color_format)
    {
        switch(color_format){
            case RaliColorFormat::RGB24:
                return 0;
            case RaliColorFormat::BGR24:
                return 1;
            case RaliColorFormat::U8:
                return 2;
            case RaliColorFormat::RGB_PLANAR:
                return 3;
            default:
                THROW("Unsupported Image type" + TOSTR(color_format))
        }
    };

    return translate_color_format(context->master_graph->output_color_format());
}
size_t RALI_API_CALL raliGetAugmentationBranchCount(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    return context->master_graph->augmentation_branch_count();
}

size_t  RALI_API_CALL
raliGetRemainingImages(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    size_t count = 0;
    try
    {
        count = context->master_graph->remaining_images_count();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return count;
}

RaliStatus RALI_API_CALL raliGetStatus(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    if(!context)
        return RALI_CONTEXT_INVALID;

    if(context->no_error())
        return RALI_OK;

    return RALI_RUNTIME_ERROR;
}

const char* RALI_API_CALL raliGetErrorMessage(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    return context->error_msg();
}
TimingInfo
RALI_API_CALL raliGetTimingInfo(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    auto info = context->timing();
    return {info.image_read_time, info.image_decode_time, info.image_process_time, info.copy_to_output};
}

size_t RALI_API_CALL raliIsEmpty(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    size_t ret = 0;
    try
    {
        ret = context->master_graph->empty();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return ret;
}
