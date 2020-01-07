#ifndef MIVISIONX_RALI_API_DATA_LOADERS_H
#define MIVISIONX_RALI_API_DATA_LOADERS_H
#include "rali_api_types.h"


/// Creates a JPEG image reader and decoder as a source. It allocates the resources and objects required to read and decode Jpeg images stored on a file systems.
// If images are not Jpeg compressed they will be ignored.
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to. some conversions are possible, some not. For instance
/// \param num_threads
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return
extern "C"  RaliImage  RALI_API_CALL raliJpegFileSource(RaliContext context,
                                                        const char* source_path,
                                                        RaliImageColor color_format,
                                                        unsigned num_threads,
                                                        bool is_output ,
                                                        bool loop = false,
                                                        RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                        unsigned max_width = 0, unsigned max_height = 0);


///
/// \param rali_context
/// \return
extern "C"  RaliStatus  RALI_API_CALL raliResetLoaders(RaliContext context);

#endif //MIVISIONX_RALI_API_DATA_LOADERS_H
