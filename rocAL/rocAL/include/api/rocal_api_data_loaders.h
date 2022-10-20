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

#ifndef MIVISIONX_ROCAL_API_DATA_LOADERS_H
#define MIVISIONX_ROCAL_API_DATA_LOADERS_H
#include "rocal_api_types.h"


/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegFileSource(RocalContext context,
                                                        const char* source_path,
                                                        RocalImageColor rocal_color_format,
                                                        unsigned internal_shard_count,
                                                        bool is_output,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                        unsigned max_width = 0, unsigned max_height = 0, RocalDecoderType rocal_decoder_type=RocalDecoderType::ROCAL_DECODER_TJPEG);

/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegFileSourceSingleShard(RocalContext context,
                                                                   const char* source_path,
                                                                   RocalImageColor rocal_color_format,
                                                                   unsigned shard_id,
                                                                   unsigned shard_count,
                                                                   bool is_output ,
                                                                   bool shuffle = false,
                                                                   bool loop = false,
                                                                   RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                   unsigned max_width = 0, unsigned max_height = 0, RocalDecoderType rocal_decoder_type=RocalDecoderType::ROCAL_DECODER_TJPEG);

/// Creates JPEG image reader and decoder. Reads [Frames] sequences from a directory representing a collection of streams.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images in a sequence will be decoded to.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances.
/// \param sequence_length: The number of frames in a sequence.
/// \param is_output Determines if the user wants the loaded sequences to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the sequences or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param step: Frame interval between each sequence.
/// \param stride: Frame interval between frames in a sequence.
/// \return Reference to the output image.
extern "C"  RocalImage  ROCAL_API_CALL rocalSequenceReader(RocalContext context,
                                                        const char* source_path,
                                                        RocalImageColor rocal_color_format,
                                                        unsigned internal_shard_count,
                                                        unsigned sequence_length,
                                                        bool is_output,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        unsigned step = 0,
                                                        unsigned stride = 0);

/// Creates JPEG image reader and decoder. Reads [Frames] sequences from a directory representing a collection of streams. It accepts external sharding information to load a singe shard only.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images in a sequence will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param sequence_length: The number of frames in a sequence.
/// \param is_output Determines if the user wants the loaded sequences to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param step: Frame interval between each sequence.
/// \param stride: Frame interval between frames in a sequence.
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalSequenceReaderSingleShard(RocalContext context,
                                                                   const char* source_path,
                                                                   RocalImageColor rocal_color_format,
                                                                   unsigned shard_id,
                                                                   unsigned shard_count,
                                                                   unsigned sequence_length,
                                                                   bool is_output,
                                                                   bool shuffle = false,
                                                                   bool loop = false,
                                                                   unsigned step = 0,
                                                                   unsigned stride = 0);

/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param rocal_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param json_path Path to the COCO Json File
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCOCOFileSource(RocalContext context,
                                                        const char* source_path,
							                            const char* json_path,
                                                        RocalImageColor color_format,
                                                        unsigned internal_shard_count,
                                                        bool is_output,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                        unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param rocal_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param json_path Path to the COCO Json File
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \param area_factor Determines how much area to be cropped. Ranges from from 0.08 - 1.
/// \param aspect_ratio Determines the aspect ration of crop. Ranges from 0.75 to 1.33.
/// \param y_drift_factor - Determines from top left corder to height (crop_height), where to start cropping other wise try for a central crop or take image dims. Ranges from 0 to 1.
/// \param x_drift_factor - Determines from top left corder to width (crop_width), where to start cropping other wise try for a central crop or take image dims. Ranges from 0 to 1.
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCOCOFileSourcePartial(RocalContext p_context,
                                                            const char* source_path,
                                                            const char* json_path,
                                                            RocalImageColor rocal_color_format,
                                                            unsigned internal_shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED,
                                                            unsigned max_width = 0, unsigned max_height = 0,
                                                            RocalFloatParam area_factor = NULL, RocalFloatParam aspect_ratio = NULL,
                                                            RocalFloatParam y_drift_factor = NULL, RocalFloatParam x_drift_factor = NULL );

/// Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param rocal_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param json_path Path to the COCO Json File
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \param area_factor Determines how much area to be cropped. Ranges from from 0.08 - 1.
/// \param aspect_ratio Determines the aspect ration of crop. Ranges from 0.75 to 1.33.
/// \param y_drift_factor - Determines from top left corder to height (crop_height), where to start cropping other wise try for a central crop or take image dims. Ranges from 0 to 1.
/// \param x_drift_factor - Determines from top left corder to width (crop_width), where to start cropping other wise try for a central crop or take image dims. Ranges from 0 to 1.
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCOCOFileSourcePartialSingleShard(RocalContext p_context,
                                                            const char* source_path,
                                                            const char* json_path,
                                                            RocalImageColor rocal_color_format,
                                                            unsigned shard_id,
                                                            unsigned shard_count,
                                                            bool is_output,
                                                            std::vector<double>& area_factor, 
                                                            std::vector<double>& aspect_ratio, 
                                                            unsigned num_attempts,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED,
                                                            unsigned max_width = 0, unsigned max_height = 0,
                                                            RocalFloatParam y_drift_factor = NULL, RocalFloatParam x_drift_factor = NULL );

/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param rocal_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param json_path Path to the COCO Json File
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCOCOFileSourceSingleShard(RocalContext context,
                                                                   const char* source_path,
                                                                   const char* json_path,
                                                                   RocalImageColor color_format,
                                                                   unsigned shard_id,
                                                                   unsigned shard_count,
                                                                   bool is_output ,
                                                                   bool shuffle = false,
                                                                   bool loop = false,
                                                                   RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                   unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for Caffe LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCaffeLMDBRecordSource(RocalContext context,
                                                            const char* source_path,
                                                            RocalImageColor rocal_color_format,
                                                            unsigned internal_shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for Caffe LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe2 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
/// \param rocal_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCaffeLMDBRecordSourceSingleShard(RocalContext p_context,
                                                            const char* source_path,
                                                            RocalImageColor rocal_color_format,
                                                            unsigned shard_id,
                                                            unsigned shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for Caffe2 LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe2 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCaffe2LMDBRecordSource(RocalContext context,
                                                            const char* source_path,
                                                            RocalImageColor rocal_color_format,
                                                            unsigned internal_shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for Caffe2 LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored on the Caffe2 LMDB Records. It accepts external sharding information to load a singe shard. only
/// \param p_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCaffe2LMDBRecordSourceSingleShard(RocalContext p_context,
                                                                        const char* source_path,
                                                                        RocalImageColor rocal_color_format,
                                                                        unsigned shard_id,
                                                                        unsigned shard_count,
                                                                        bool is_output,
                                                                        bool shuffle = false,
                                                                        bool loop = false,
                                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                        unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for MXNet records. It allocates the resources and objects required to read and decode Jpeg images stored in MXNet Records. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalMXNetRecordSource(RocalContext context,
                                                            const char* source_path,
                                                            RocalImageColor rocal_color_format,
                                                            unsigned internal_shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for MXNet records. It allocates the resources and objects required to read and decode Jpeg images stored on the MXNet records. It accepts external sharding information to load a singe shard. only
/// \param p_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalMXNetRecordSourceSingleShard(RocalContext p_context,
                                                                        const char* source_path,
                                                                        RocalImageColor rocal_color_format,
                                                                        unsigned shard_id,
                                                                        unsigned shard_count,
                                                                        bool is_output,
                                                                        bool shuffle = false,
                                                                        bool loop = false,
                                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                        unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored and Crops t
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param num_threads Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \param area_factor Determines how much area to be cropped. Ranges from from 0.08 - 1.
/// \param aspect_ratio Determines the aspect ration of crop. Ranges from 0.75 to 1.33.
/// \param y_drift_factor - Determines from top left corder to height (crop_height), where to start cropping other wise try for a central crop or take image dims. Ranges from 0 to 1.
/// \param x_drift_factor - Determines from top left corder to width (crop_width), where to start cropping other wise try for a central crop or take image dims. Ranges from 0 to 1.
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalFusedJpegCrop(RocalContext context,
                                                        const char* source_path,
                                                        RocalImageColor rocal_color_format,
                                                        unsigned num_threads,
                                                        bool is_output ,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED,
                                                        unsigned max_width = 0, unsigned max_height = 0,
                                                        RocalFloatParam area_factor = NULL, RocalFloatParam aspect_ratio = NULL,
                                                        RocalFloatParam y_drift_factor = NULL, RocalFloatParam x_drift_factor = NULL);

/// Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalFusedJpegCropSingleShard(RocalContext context,
                                                        const char* source_path,
                                                        RocalImageColor color_format,
                                                        unsigned shard_id,
                                                        unsigned shard_count,
                                                        bool is_output ,
                                                        std::vector<double>& area_factor, 
                                                        std::vector<double>& aspect_ratio, 
                                                        unsigned num_attempts,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED,
                                                        unsigned max_width = 0, unsigned max_height = 0,
                                                        RocalFloatParam y_drift_factor = NULL, RocalFloatParam x_drift_factor = NULL);

/// Creates TensorFlow records JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location of the TF records on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegTFRecordSource(RocalContext context,
                                                            const char* source_path,
                                                            RocalImageColor rocal_color_format,
                                                            unsigned internal_shard_count,
                                                            bool is_output,
                                                            const char* user_key_for_encoded,
                                                            const char* user_key_for_filename,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0);
/// Creates TensorFlow records JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location of the TF records on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegTFRecordSourceSingleShard(RocalContext context,
                                                                        const char* source_path,
                                                                        RocalImageColor rocal_color_format,
                                                                        unsigned shard_id,
                                                                        unsigned shard_count,
                                                                        bool is_output,
                                                                        bool shuffle = false,
                                                                        bool loop = false,
                                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                        unsigned max_width = 0, unsigned max_height = 0);
/// Creates Raw image loader. It allocates the resources and objects required to load images stored on the file systems.
/// \param rocal_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle: to shuffle dataset
/// \param loop: repeat data loading
/// \param out_width The output_width of raw image
/// \param out_height The output height of raw image
/// \return

extern "C"  RocalImage  ROCAL_API_CALL rocalRawTFRecordSource(RocalContext p_context,
                                                           const char* source_path,
                                                           const char* user_key_for_raw,
                                                           const char* user_key_for_filename,
                                                           RocalImageColor rocal_color_format,
                                                           bool is_output,
                                                           bool shuffle = false,
                                                           bool loop = false,
                                                           unsigned out_width=0, unsigned out_height=0,
                                                           const char* record_name_prefix = "");

/// Creates Raw image loader. It allocates the resources and objects required to load images stored on the file systems.
/// \param rocal_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param shuffle: to shuffle dataset
/// \param loop: repeat data loading
/// \param out_width The output_width of raw image
/// \param out_height The output height of raw image
/// \param record_name_prefix : if nonempty reader will only read records with certain prefix
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalRawTFRecordSourceSingleShard(RocalContext p_context,
                                                                      const char* source_path,
                                                                      RocalImageColor rocal_color_format,
                                                                      unsigned shard_id,
                                                                      unsigned shard_count,
                                                                      bool is_output,
                                                                      bool shuffle = false,
                                                                      bool loop = false,
                                                                      unsigned out_width=0, unsigned out_height=0,
                                                                      const char* record_name_prefix = "");

/// Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode mp4 videos stored on the file systems.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk.
/// source_path can be a video file, folder containing videos or a text file
/// \param color_format The color format the frames will be decoded to.
/// \param rocal_decode_device Enables software or hardware decoding. Currently only software decoding is supported.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances.
/// \param sequence_length: The number of frames in a sequence.
/// \param shuffle: to shuffle sequences.
/// \param is_output Determines if the user wants the loaded sequence of frames to be part of the output or not.
/// \param loop: repeat data loading.
/// \param step: Frame interval between each sequence.
/// \param stride: Frame interval between frames in a sequence.
/// \param file_list_frame_num: Determines if the user wants to read frame number or timestamps if a text file is passed in the source_path.
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalVideoFileSource(RocalContext context,
                                                        const char* source_path,
                                                        RocalImageColor color_format,
                                                        RocalDecodeDevice rocal_decode_device,
                                                        unsigned internal_shard_count,
                                                        unsigned sequence_length,
                                                        bool is_output = false,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        unsigned step = 0,
                                                        unsigned stride = 0,
                                                        bool file_list_frame_num = true
                                                        );

/// Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode mp4 videos stored on the file systems. It accepts external sharding information to load a singe shard only.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk.
/// source_path can be a video file, folder containing videos or a text file
/// \param color_format The color format the frames will be decoded to.
/// \param rocal_decode_device Enables software or hardware decoding. Currently only software decoding is supported.
/// \param shard_id Shard id for this loader.
/// \param shard_count Total shard count.
/// \param sequence_length: The number of frames in a sequence.
/// \param shuffle: to shuffle sequences.
/// \param is_output Determines if the user wants the loaded sequence of frames to be part of the output or not.
/// \param loop: repeat data loading.
/// \param step: Frame interval between each sequence.
/// \param stride: Frame interval between frames in a sequence.
/// \param file_list_frame_num: Determines if the user wants to read frame number or timestamps if a text file is passed in the source_path.
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalVideoFileSourceSingleShard(RocalContext context,
                                                                    const char* source_path,
                                                                    RocalImageColor color_format,
                                                                    RocalDecodeDevice rocal_decode_device,
                                                                    unsigned shard_id,
                                                                    unsigned shard_count,
                                                                    unsigned sequence_length,
                                                                    bool shuffle = false,
                                                                    bool is_output = false,
                                                                    bool loop = false,
                                                                    unsigned step = 0,
                                                                    unsigned stride = 0,
                                                                    bool file_list_frame_num = true
                                                                    );

/// Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode mp4 videos stored on the file systems. Resizes the decoded frames to the dest width and height.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk.
/// source_path can be a video file, folder containing videos or a text file
/// \param color_format The color format the frames will be decoded to.
/// \param rocal_decode_device Enables software or hardware decoding. Currently only software decoding is supported.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances.
/// \param sequence_length: The number of frames in a sequence.
/// \param dest_width The output width of frames.
/// \param dest_height The output height of frames.
/// \param shuffle: to shuffle sequences.
/// \param is_output Determines if the user wants the loaded sequence of frames to be part of the output or not.
/// \param loop: repeat data loading.
/// \param step: Frame interval between each sequence.
/// \param stride: Frame interval between frames in a sequence.
/// \param file_list_frame_num: Determines if the user wants to read frame number or timestamps if a text file is passed in the source_path.
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalVideoFileResize(RocalContext context,
                                                        const char* source_path,
                                                        RocalImageColor color_format,
                                                        RocalDecodeDevice rocal_decode_device,
                                                        unsigned internal_shard_count,
                                                        unsigned sequence_length,
                                                        unsigned dest_width,
                                                        unsigned dest_height,
                                                        bool shuffle = false,
                                                        bool is_output = false,
                                                        bool loop = false,
                                                        unsigned step = 0,
                                                        unsigned stride = 0,
                                                        bool file_list_frame_num = true
                                                        );

/// Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode mp4 videos stored on the file systems. Resizes the decoded frames to the dest width and height. It accepts external sharding information to load a singe shard only.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk.
/// source_path can be a video file, folder containing videos or a text file
/// \param color_format The color format the frames will be decoded to.
/// \param rocal_decode_device Enables software or hardware decoding. Currently only software decoding is supported.
/// \param shard_id Shard id for this loader.
/// \param shard_count Total shard count.
/// \param sequence_length: The number of frames in a sequence.
/// \param dest_width The output width of frames.
/// \param dest_height The output height of frames.
/// \param shuffle: to shuffle sequences.
/// \param is_output Determines if the user wants the loaded sequence of frames to be part of the output or not.
/// \param loop: repeat data loading.
/// \param step: Frame interval between each sequence.
/// \param stride: Frame interval between frames in a sequence.
/// \param file_list_frame_num: Determines if the user wants to read frame number or timestamps if a text file is passed in the source_path.
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalVideoFileResizeSingleShard(RocalContext context,
                                                        const char* source_path,
                                                        RocalImageColor color_format,
                                                        RocalDecodeDevice rocal_decode_device,
                                                        unsigned shard_id,
                                                        unsigned shard_count,
                                                        unsigned sequence_length,
                                                        unsigned dest_width,
                                                        unsigned dest_height,
                                                        bool shuffle = false,
                                                        bool is_output = false,
                                                        bool loop = false,
                                                        unsigned step = 0,
                                                        unsigned stride = 0,
                                                        bool file_list_frame_num = true
                                                        );

/// Creates CIFAR10 raw data reader and loader. It allocates the resources and objects required to read raw data stored on the file systems.
/// \param context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param out_width ; output width
/// \param out_height ; output_height
/// \param filename_prefix ; if set loader will only load files with the given prefix name
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalRawCIFAR10Source(RocalContext context,
                                                        const char* source_path,
                                                        RocalImageColor color_format,
                                                        bool is_output ,
                                                        unsigned out_width, unsigned out_height, const char* filename_prefix = "",
                                                        bool loop = false);

///
/// \param context
/// \return
extern "C"  RocalStatus  ROCAL_API_CALL rocalResetLoaders(RocalContext context);

/// Creates JPEG image reader and partial decoder for Caffe LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe2 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
/// \param rocal_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCaffeLMDBRecordSourcePartialSingleShard(RocalContext p_context,
                                                            const char* source_path,
                                                            RocalImageColor rocal_color_format,
                                                            unsigned shard_id,
                                                            unsigned shard_count,
                                                            bool is_output,
                                                            std::vector<double>& area_factor, 
                                                            std::vector<double>& aspect_ratio, 
                                                            unsigned num_attempts,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0,
                                                            RocalFloatParam y_drift_factor = NULL, RocalFloatParam x_drift_factor = NULL );

/// Creates JPEG image reader and partial decoder for Caffe2 LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe22 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
/// \param rocal_context Rocal context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rocal_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RocalImage  ROCAL_API_CALL rocalJpegCaffe2LMDBRecordSourcePartialSingleShard(RocalContext p_context,
                                                            const char* source_path,
                                                            RocalImageColor rocal_color_format,
                                                            unsigned shard_id,
                                                            unsigned shard_count,
                                                            bool is_output,
                                                            std::vector<double>& area_factor, 
                                                            std::vector<double>& aspect_ratio, 
                                                            unsigned num_attempts,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED,
                                                            unsigned max_width = 0, unsigned max_height = 0,
                                                            RocalFloatParam y_drift_factor = NULL, RocalFloatParam x_drift_factor = NULL );

#endif //MIVISIONX_ROCAL_API_DATA_LOADERS_H

