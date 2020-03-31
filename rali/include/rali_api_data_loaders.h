#ifndef MIVISIONX_RALI_API_DATA_LOADERS_H
#define MIVISIONX_RALI_API_DATA_LOADERS_H
#include "rali_api_types.h"


/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliJpegFileSource(RaliContext context,
                                                        const char* source_path,
                                                        RaliImageColor color_format,
                                                        unsigned internal_shard_count,
                                                        bool is_output ,
                                                        bool loop = false,
                                                        RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                        unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return
extern "C"  RaliImage  RALI_API_CALL raliJpegFileSourceSingleShard(RaliContext context,
                                                                   const char* source_path,
                                                                   RaliImageColor color_format,
                                                                   unsigned shard_id,
                                                                   unsigned shard_count,
                                                                   bool is_output ,
                                                                   bool loop = false,
                                                                   RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                                   unsigned max_width = 0, unsigned max_height = 0);


/// Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode H.264 videos stored on the file systems.
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk, multiple sources can be separated using the ":" delimiter
/// \param rali_color_format The color format the images will be decoded to.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param width The  width of the decoded frames, larger or smaller will be resized
/// \param height The height of the decoded frames, larger or smaller will be resized
/// \return
extern "C"  RaliImage  RALI_API_CALL raliVideoFileSource(RaliContext context,
                                                        const char* source_path,
                                                        RaliImageColor color_format,
                                                        RaliDecodeDevice rali_decode_device,
                                                        bool is_output ,
                                                        unsigned width , unsigned height, bool loop = false );
///
/// \param rali_context
/// \return
extern "C"  RaliStatus  RALI_API_CALL raliResetLoaders(RaliContext context);

#endif //MIVISIONX_RALI_API_DATA_LOADERS_H
