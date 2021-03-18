/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef MIVISIONX_RALI_API_DATA_LOADERS_H
#define MIVISIONX_RALI_API_DATA_LOADERS_H
#include "rali_api_types.h"


/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliJpegFileSource(RaliContext context,
                                                        const char* source_path,
                                                        RaliImageColor rali_color_format,
                                                        unsigned internal_shard_count,
                                                        bool is_output,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                        unsigned max_width = 0, unsigned max_height = 0, RaliDecoderType rali_decoder_type=RaliDecoderType::RALI_DECODER_TJPEG);

/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliJpegFileSourceSingleShard(RaliContext context,
                                                                   const char* source_path,
                                                                   RaliImageColor rali_color_format,
                                                                   unsigned shard_id,
                                                                   unsigned shard_count,
                                                                   bool is_output ,
                                                                   bool shuffle = false,
                                                                   bool loop = false,
                                                                   RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                                   unsigned max_width = 0, unsigned max_height = 0);


/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param json_path Path to the COCO Json File
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliJpegCOCOFileSource(RaliContext context,
                                                        const char* source_path,
							                            const char* json_path,
                                                        RaliImageColor color_format,
                                                        unsigned internal_shard_count,
                                                        bool is_output,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                        unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param json_path Path to the COCO Json File
/// \param rali_color_format The color format the images will be decoded to.
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
extern "C"  RaliImage  RALI_API_CALL raliJpegCOCOFileSourcePartial(RaliContext p_context,
                                                            const char* source_path,
                                                            const char* json_path,
                                                            RaliImageColor rali_color_format,
                                                            unsigned internal_shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MAX_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0,
                                                            RaliFloatParam area_factor = NULL, RaliFloatParam aspect_ratio = NULL,
                                                            RaliFloatParam y_drift_factor = NULL, RaliFloatParam x_drift_factor = NULL );

/// Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param json_path Path to the COCO Json File
/// \param rali_color_format The color format the images will be decoded to.
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
extern "C"  RaliImage  RALI_API_CALL raliJpegCOCOFileSourcePartialSingleShard(RaliContext p_context,
                                                            const char* source_path,
                                                            const char* json_path,
                                                            RaliImageColor rali_color_format,
                                                            unsigned shard_id,
                                                            unsigned shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MAX_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0,
                                                            RaliFloatParam area_factor = NULL, RaliFloatParam aspect_ratio = NULL,
                                                            RaliFloatParam y_drift_factor = NULL, RaliFloatParam x_drift_factor = NULL );

/// Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param json_path Path to the COCO Json File
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return
extern "C"  RaliImage  RALI_API_CALL raliJpegCOCOFileSourceSingleShard(RaliContext context,
                                                                   const char* source_path,
								   const char* json_path,
                                                                   RaliImageColor color_format,
                                                                   unsigned shard_id,
                                                                   unsigned shard_count,
                                                                   bool is_output ,
                                                                   bool shuffle = false,
                                                                   bool loop = false,
                                                                   RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                                   unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for Caffe LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliJpegCaffeLMDBRecordSource(RaliContext context,
                                                            const char* source_path,
                                                            RaliImageColor rali_color_format,
                                                            unsigned internal_shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for Caffe LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe2 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliJpegCaffeLMDBRecordSourceSingleShard(RaliContext p_context,
                                                            const char* source_path,
                                                            RaliImageColor rali_color_format,
                                                            unsigned shard_id,
                                                            unsigned shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for Caffe2 LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe2 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliJpegCaffe2LMDBRecordSource(RaliContext context,
                                                            const char* source_path,
                                                            RaliImageColor rali_color_format,
                                                            unsigned internal_shard_count,
                                                            bool is_output,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and decoder for Caffe2 LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored on the Caffe2 LMDB Records. It accepts external sharding information to load a singe shard. only
/// \param p_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliJpegCaffe2LMDBRecordSourceSingleShard(RaliContext p_context,
                                                                        const char* source_path,
                                                                        RaliImageColor rali_color_format,
                                                                        unsigned shard_id,
                                                                        unsigned shard_count,
                                                                        bool is_output,
                                                                        bool shuffle = false,
                                                                        bool loop = false,
                                                                        RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                                        unsigned max_width = 0, unsigned max_height = 0);

/// Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored and Crops t
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
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
extern "C"  RaliImage  RALI_API_CALL raliFusedJpegCrop(RaliContext context,
                                                        const char* source_path,
                                                        RaliImageColor rali_color_format,
                                                        unsigned num_threads,
                                                        bool is_output ,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MAX_SIZE,
                                                        unsigned max_width = 0, unsigned max_height = 0, 
                                                        RaliFloatParam area_factor = NULL, RaliFloatParam aspect_ratio = NULL,
                                                        RaliFloatParam y_drift_factor = NULL, RaliFloatParam x_drift_factor = NULL);

/// Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return
extern "C"  RaliImage  RALI_API_CALL raliFusedJpegCropSingleShard(RaliContext context,
                                                        const char* source_path,
                                                        RaliImageColor color_format,
                                                        unsigned shard_id,
                                                        unsigned shard_count,
                                                        bool is_output ,
                                                        bool shuffle = false,
                                                        bool loop = false,
                                                        RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MAX_SIZE,
                                                        unsigned max_width = 0, unsigned max_height = 0,
                                                        RaliFloatParam area_factor = NULL, RaliFloatParam aspect_ratio = NULL,
                                                        RaliFloatParam y_drift_factor = NULL, RaliFloatParam x_drift_factor = NULL);

/// Creates TensorFlow records JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
/// If images are not Jpeg compressed they will be ignored.
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location of the TF records on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliJpegTFRecordSource(RaliContext context,
                                                            const char* source_path,
                                                            RaliImageColor rali_color_format,
                                                            unsigned internal_shard_count,
                                                            bool is_output,
                                                            const char* user_key_for_encoded,
                                                            const char* user_key_for_filename,
                                                            bool shuffle = false,
                                                            bool loop = false,
                                                            RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                            unsigned max_width = 0, unsigned max_height = 0);
/// Creates TensorFlow records JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location of the TF records on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle Determines if the user wants to shuffle the dataset or not.
/// \param loop Determines if the user wants to indefinitely loops through images or not.
/// \param decode_size_policy
/// \param max_width The maximum width of the decoded images, larger or smaller will be resized to closest
/// \param max_height The maximum height of the decoded images, larger or smaller will be resized to closest
/// \return
extern "C"  RaliImage  RALI_API_CALL raliJpegTFRecordSourceSingleShard(RaliContext context,
                                                                        const char* source_path,
                                                                        RaliImageColor rali_color_format,
                                                                        unsigned shard_id,
                                                                        unsigned shard_count,
                                                                        bool is_output,
                                                                        bool shuffle = false,
                                                                        bool loop = false,
                                                                        RaliImageSizeEvaluationPolicy decode_size_policy = RALI_USE_MOST_FREQUENT_SIZE,
                                                                        unsigned max_width = 0, unsigned max_height = 0);
/// Creates Raw image loader. It allocates the resources and objects required to load images stored on the file systems.
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param shuffle: to shuffle dataset
/// \param loop: repeat data loading
/// \param out_width The output_width of raw image
/// \param out_height The output height of raw image
/// \return

extern "C"  RaliImage  RALI_API_CALL raliRawTFRecordSource(RaliContext p_context,
                                                           const char* source_path,
                                                           const char* user_key_for_raw,
                                                           const char* user_key_for_filename,
                                                           RaliImageColor rali_color_format,
                                                           bool is_output,
                                                           bool shuffle = false,
                                                           bool loop = false,
                                                           unsigned out_width=0, unsigned out_height=0,
                                                           const char* record_name_prefix = "");

/// Creates Raw image loader. It allocates the resources and objects required to load images stored on the file systems.
/// \param rali_context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param shard_id Shard id for this loader
/// \param shard_count Total shard count
/// \param shuffle: to shuffle dataset
/// \param loop: repeat data loading
/// \param out_width The output_width of raw image
/// \param out_height The output height of raw image
/// \param record_name_prefix : if nonempty reader will only read records with certain prefix
/// \return
extern "C"  RaliImage  RALI_API_CALL raliRawTFRecordSourceSingleShard(RaliContext p_context,
                                                                      const char* source_path,
                                                                      RaliImageColor rali_color_format,
                                                                      unsigned shard_id,
                                                                      unsigned shard_count,
                                                                      bool is_output,
                                                                      bool shuffle = false,
                                                                      bool loop = false,
                                                                      unsigned out_width=0, unsigned out_height=0,
                                                                      const char* record_name_prefix = "");
/// Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode H.264 videos stored on the file systems.
/// \param context Rali context
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
/// Creates CIFAR10 raw data reader and loader. It allocates the resources and objects required to read raw data stored on the file systems.
/// \param context Rali context
/// \param source_path A NULL terminated char string pointing to the location on the disk
/// \param rali_color_format The color format the images will be decoded to.
/// \param is_output Determines if the user wants the loaded images to be part of the output or not.
/// \param out_width ; output width
/// \param out_height ; output_height
/// \param filename_prefix ; if set loader will only load files with the given prefix name
/// \return Reference to the output image
extern "C"  RaliImage  RALI_API_CALL raliRawCIFAR10Source(RaliContext context,
                                                        const char* source_path,
                                                        RaliImageColor color_format,
                                                        bool is_output ,
                                                        unsigned out_width, unsigned out_height, const char* filename_prefix = "",
                                                        bool loop = false);

///
/// \param context
/// \return
extern "C"  RaliStatus  RALI_API_CALL raliResetLoaders(RaliContext context);

#endif //MIVISIONX_RALI_API_DATA_LOADERS_H
