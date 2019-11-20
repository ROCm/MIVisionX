#include "commons.h"
#include "context.h"
#include "rali_api.h"
size_t RALI_API_CALL raliGetImageWidth(RaliImage image)
{
    return image->info().width();
}
size_t RALI_API_CALL raliGetImageHeight(RaliImage image)
{
    return image->info().height_batch();
}

size_t RALI_API_CALL raliGetImagePlanes(RaliImage image)
{
    return image->info().color_plane_count();
}

void RALI_API_CALL raliGetImageName(RaliImage image, char* buf, unsigned image_idx)
{

    auto ret = image->get_name();
    memcpy((void*)buf, ret[image_idx].c_str(), ret[image_idx].size());
}

unsigned RALI_API_CALL raliGetImageNameLen(RaliImage image,  unsigned image_idx)
{
    auto ret = image->get_name();
    return ret[image_idx].size();
}
int RALI_API_CALL raliGetOutputWidth(RaliContext rali_context)
{
    return rali_context->master_graph->output_width();
}

int RALI_API_CALL raliGetOutputHeight(RaliContext rali_context)
{
    return rali_context->master_graph->output_height();
}

int RALI_API_CALL raliGetOutputColorFormat(RaliContext rali_context)
{
    auto translate_color_format = [](RaliColorFormat color_format)
    {
        switch(color_format){
            case RaliColorFormat::RGB24:
                return 0;
            case RaliColorFormat::BGR24:
                return 1;
            case RaliColorFormat::U8:
                return 2;
            default:
                THROW("Unsupported Image type" + TOSTR(color_format))
        }
    };

    return translate_color_format(rali_context->master_graph->output_color_format());
}
size_t RALI_API_CALL raliGetOutputImageCount(RaliContext rali_context)
{
    return rali_context->master_graph->output_image_count();
}

size_t  RALI_API_CALL
raliGetRemainingImages(RaliContext rali_context)
{
    size_t count = 0;
    try
    {
        count = rali_context->master_graph->remaining_images_count();
    }
    catch(const std::exception& e)
    {
        rali_context->capture_error(e.what());
        ERR(e.what());
    }
    return count;
}

RaliStatus RALI_API_CALL raliGetStatus(RaliContext rali_context)
{
    if(!rali_context)
        return RALI_CONTEXT_INVALID;

    if(rali_context->no_error())
        return RALI_OK;

    return RALI_RUNTIME_ERROR;
}

const char* RALI_API_CALL raliGetErrorMessage(RaliContext rali_context)
{
    return rali_context->error_msg();
}
TimingInfo RALI_API_CALL raliGetTimingInfo(RaliContext rali_context)
{
    auto info = rali_context->timing();
    if(info.size() < 4)
        return {0,0,0,0};

    return {info[0], info[1], info[2], info[3]};
}




