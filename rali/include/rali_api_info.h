#ifndef MIVISIONX_RALI_API_INFO_H
#define MIVISIONX_RALI_API_INFO_H
#include "rali_api_types.h"
///
/// \param image
/// \return
extern "C" int RALI_API_CALL raliGetOutputWidth(RaliContext rali_context);

///
/// \param image
/// \return
extern "C" int RALI_API_CALL raliGetOutputHeight(RaliContext rali_context);

///
/// \param rali_context
/// \return
extern "C" int RALI_API_CALL raliGetOutputColorFormat(RaliContext rali_context);

///
/// \param rali_context
/// \return
extern "C"  size_t  RALI_API_CALL raliGetRemainingImages(RaliContext rali_context);

/// Returned value valid only after raliVerify is called
/// \param image
/// \return Width of the graph output image
extern "C" size_t RALI_API_CALL raliGetImageWidth(RaliImage image);

/// Returned value valid only after raliVerify is called
/// \param image
/// \return Height of the pipeline output image, includes all images in the batch
extern "C" size_t RALI_API_CALL raliGetImageHeight(RaliImage image);


/// Returned value valid only after raliVerify is called
/// \param image
/// \return Color format of the pipeline output image,
extern "C" size_t RALI_API_CALL raliGetImagePlanes(RaliImage image);

/// Returned value valid only after raliRun is called
/// \param image
/// \param image_idx Index for the image in the batch
/// \return pointer to the start of buffer storing the image name currently loaded into the batch,
extern "C" void RALI_API_CALL raliGetImageName(RaliImage image, char* buf, unsigned image_idx);

extern "C" unsigned RALI_API_CALL raliGetImageNameLen(RaliImage image,  unsigned image_idx);

///
/// \param rali_context
/// \return
extern "C" size_t RALI_API_CALL raliGetOutputImageCount(RaliContext rali_context);

///
/// \param rali_context
/// \return
extern "C" RaliStatus RALI_API_CALL raliGetStatus(RaliContext rali_context);

///
/// \param rali_context
/// \return
extern "C" const char* RALI_API_CALL raliGetErrorMessage(RaliContext rali_context);

///
/// \param rali_context
/// \return
extern "C" TimingInfo RALI_API_CALL raliGetTimingInfo(RaliContext rali_context);

#endif //MIVISIONX_RALI_API_INFO_H
