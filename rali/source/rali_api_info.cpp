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

RaliMetaData
RALI_API_CALL raliCreateLabelReader(RaliContext p_context, const char* source_path) {
    auto context = static_cast<Context*>(p_context);
    if (!context)
        THROW("Invalid rali context passed to raliMetaDataReader")

    return context->master_graph->create_file_system_label_reader(source_path);

}
void
RALI_API_CALL raliGetImageName(RaliContext p_context,  char* buf, unsigned image_idx)
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(image_idx >= meta_data.first.size())
        THROW("Image idx is out of batch size range")
    memcpy(buf, meta_data.first[image_idx].c_str(), meta_data.first[image_idx].size());
}

unsigned
RALI_API_CALL raliGetImageNameLen(RaliContext p_context, unsigned image_idx)
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(image_idx >= meta_data.first.size())
        THROW("Image idx is out of batch size range")
    return meta_data.first[image_idx].size();
}

void
RALI_API_CALL raliGetImageLabels(RaliContext p_context, int* buf)
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second) {
        WRN("No label has been loaded for this output image")
        return;
    }
    size_t meta_data_batch_size = meta_data.second->get_label_batch().size();
    if(context->user_batch_size() != meta_data_batch_size)
        THROW("meta data batch size is wrong " + TOSTR(meta_data_batch_size) + " != "+ TOSTR(context->user_batch_size() ))
    memcpy(buf, meta_data.second->get_label_batch().data(),  sizeof(int)*meta_data_batch_size);
}

unsigned
RALI_API_CALL raliGetBoundingBoxCount(RaliContext p_context, unsigned image_idx )
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return 0;
    }
    return meta_data.second->get_bb_labels_batch()[image_idx].size();
}

void
RALI_API_CALL raliGetBoundingBoxLabel(RaliContext p_context, int* buf, unsigned image_idx )
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();
    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    unsigned bb_count = meta_data.second->get_bb_labels_batch()[image_idx].size();
    memcpy(buf, meta_data.second->get_bb_labels_batch()[image_idx].data(),  sizeof(int)*bb_count);
}

void
RALI_API_CALL raliGetBoundingBoxCords(RaliContext p_context, int* buf, unsigned image_idx )
{
    auto context = static_cast<Context*>(p_context);
    auto meta_data = context->master_graph->meta_data();

    if(!meta_data.second)
    {
        WRN("No label has been loaded for this output image")
        return;
    }
    auto ptr = buf;
    memcpy(ptr,meta_data.second->get_bb_cords_batch()[image_idx].data(), meta_data.second->get_bb_cords_batch()[image_idx].size() * sizeof(BoundingBoxCord));
    ptr += sizeof(BoundingBoxCord)*sizeof(int);
}


