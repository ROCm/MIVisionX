#ifndef MIVISIONX_RALI_API_META_DATA_H
#define MIVISIONX_RALI_API_META_DATA_H
#include "rali_api_types.h"
///
/// \param rali_context
/// \param source_path path to the folder that contains the dataset or metadata file
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateLabelReader(RaliContext rali_context, const char* source_path);

///
/// \param rali_context
/// \param source_path path to the file that contains the metadata file
/// \return RaliMetaData object, can be used to inquire about the rali's output (processed) tensors
extern "C" RaliMetaData RALI_API_CALL raliCreateTextFileBasedLabelReader(RaliContext rali_context, const char* source_path);
///
/// \param rali_context
/// \param buf user buffer provided to be filled with output image name
/// \param image_idx the imageIdx in the output batch
extern "C" void RALI_API_CALL raliGetImageName(RaliContext rali_context,  char* buf, unsigned image_idx);

///
/// \param rali_context
/// \param image_idx the imageIdx in the output batch
/// \return The length of the name of the image associated with image_idx in the output batch
extern "C" unsigned RALI_API_CALL raliGetImageNameLen(RaliContext rali_context,  unsigned image_idx);

/// \param meta_data RaliMetaData object that contains info about the images and labels
/// \param buf user's buffer that will be filled with labels. Its needs to be at least of size batch_size.
extern "C" void RALI_API_CALL raliGetImageLabels(RaliContext rali_context, int* buf);

///
/// \param rali_context
/// \param image_idx the imageIdx in the output batch
/// \return The size of the buffer needs to be provided by user to get bounding box info associated with image_idx in the output batch.
extern "C" unsigned RALI_API_CALL raliGetBoundingBoxCount(RaliContext rali_context, unsigned image_idx );

///
/// \param rali_context
/// \param image_idx the imageIdx in the output batch
/// \param buf The user's buffer that will be filled with bounding box info. It needs to be of size bounding box len returned by a call to the raliGetBoundingBoxCount
extern "C" void RALI_API_CALL raliGetBoundingBoxLabel(RaliContext rali_context, int* buf, unsigned image_idx );
extern "C" void RALI_API_CALL raliGetBoundingBoxCords(RaliContext rali_context, int* buf, unsigned image_idx );

#endif //MIVISIONX_RALI_API_META_DATA_H
