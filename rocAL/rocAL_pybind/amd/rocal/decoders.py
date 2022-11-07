# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import amd.rocal.types as types
import rocal_pybind as b
from amd.rocal.pipeline import Pipeline


def image(*inputs, user_feature_key_map=None, path='', file_root='', annotations_file='', shard_id=0, num_shards=1,
    random_shuffle=False, affine=True, bytes_per_sample_hint=0, cache_batch_copy=True, cache_debug=False, cache_size=0,
    cache_threshold=0,cache_type='', device_memory_padding=16777216, host_memory_padding=8388608, hybrid_huffman_threshold=1000000,
    output_type=types.RGB,preserve=False, seed=1, split_stages=False, use_chunk_allocator=False, use_fast_idct=False, device=None):
    b.setSeed(seed)
    reader = Pipeline._current_pipeline._reader
    if(reader == 'COCOReader'):
        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0,
            "max_height": 0}
        decoded_image = b.COCO_ImageDecoderShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    elif (reader == "TFRecordReaderClassification" or reader == "TFRecordReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "num_shards": num_shards,
            'is_output': False,
            "user_key_for_encoded": user_feature_key_map["image/encoded"],
            "user_key_for_filename": user_feature_key_map["image/filename"],
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE_ORIG,
            "max_width": 2000,
            "max_height": 2000}
        decoded_image = b.TF_ImageDecoder(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    elif (reader == "Caffe2Reader" or reader == "Caffe2ReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0,
            "max_height": 0}
        decoded_image = b.Caffe2_ImageDecoderShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    elif reader == "CaffeReader" or reader == "CaffeReaderDetection":
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 0,
            "max_height": 0}
        decoded_image = b.Caffe_ImageDecoderShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    else:
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE_ORIG,
            "max_width": 2000,
            "max_height": 2000,
            "dec_type": types.DECODER_TJPEG}
        decoded_image = b.ImageDecoderShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    return (decoded_image)


def image_raw(*inputs, user_feature_key_map=None, path='', file_root='', annotations_file='', shard_id=0, num_shards=1,
    random_shuffle=False, affine=True, bytes_per_sample_hint=0, cache_batch_copy=True, cache_debug=False, cache_size=0,
    cache_threshold=0,cache_type='', device_memory_padding=16777216, host_memory_padding=8388608, hybrid_huffman_threshold=1000000,
    output_type=types.RGB,preserve=False, seed=1, split_stages=False, use_chunk_allocator=False, use_fast_idct=False, device=None):
    
    reader = Pipeline._current_pipeline._reader
    b.setSeed(seed)

    if (reader == "TFRecordReaderClassification" or reader == "TFRecordReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "user_key_for_encoded": user_feature_key_map["image/encoded"],
            "user_key_for_filename": user_feature_key_map["image/filename"],
            "color_format": output_type,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "out_width": 2000,
            "out_height": 2000}
        decoded_image = b.TF_ImageDecoderRaw(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
        return (decoded_image)


def image_random_crop(*inputs, user_feature_key_map=None, path='', file_root='', annotations_file='', num_shards=1, shard_id=0,
    random_shuffle=False, affine=True, bytes_per_sample_hint=0, device_memory_padding=16777216, host_memory_padding=8388608,
    hybrid_huffman_threshold=1000000, num_attempts=10, output_type=types.RGB, preserve=False,
    random_area=[0.08, 1.0], random_aspect_ratio=[0.8, 1.25], seed=1, split_stages=False, use_chunk_allocator=False, use_fast_idct=False, device=None):

    reader = Pipeline._current_pipeline._reader
    # Internally calls the C++ Partial decoder's
    b.setSeed(seed)
    if(reader == 'COCOReader'):

        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "shard_count": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 1200,
            "max_height": 1200,
            "x_drift_factor": None,
            "y_drift_factor": None}
        crop_output_image = b.COCO_ImageDecoderSliceShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif (reader == "CaffeReader" or reader == "CaffeReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 1200,
            "max_height": 1200,
            "x_drift_factor": None,
            "y_drift_factor": None}
        crop_output_image = b.Caffe_ImageDecoderPartialShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif (reader == "Caffe2Reader" or reader == "Caffe2ReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 1200,
            "max_height": 1200,
            "x_drift_factor": None,
            "y_drift_factor": None}
        crop_output_image = b.Caffe2_ImageDecoderPartialShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE_ORIG,
            "max_width": 2000,
            "max_height": 2000,
            "x_drift_factor": None,
            "y_drift_factor": None}
        crop_output_image = b.FusedDecoderCropShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    return (crop_output_image)


def image_slice(*inputs, file_root='', path='', annotations_file='', shard_id=0, num_shards=1,
    random_shuffle=False, affine=True, axes=None,axis_names="WH", bytes_per_sample_hint=0,
    device_memory_padding=16777216,device_memory_padding_jpeg2k=0, host_memory_padding=8388608,
    random_aspect_ratio=[0.8, 1.25], random_area=[0.08, 1.0], num_attempts=100, host_memory_padding_jpeg2k=0,
    hybrid_huffman_threshold=1000000, memory_stats=False, normalized_anchor=True, normalized_shape=True,
    output_type=types.RGB, preserve=False, seed=1, split_stages=False, use_chunk_allocator=False, use_fast_idct=False, device=None):

    reader = Pipeline._current_pipeline._reader
    b.setSeed(seed)
    #Reader -> Randon BBox Crop -> ImageDecoderSlice
    #TODO:To pass the crop co-ordinates from random_bbox_crop to image_slice in tensor branch integration, for now calling partial decoder to match SSD training outer API's .
    if(reader == 'COCOReader'):

        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "shard_count": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 1200,
            "max_height": 1200,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.COCO_ImageDecoderSliceShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif (reader == "CaffeReader" or reader == "CaffeReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 1200,
            "max_height": 1200,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.Caffe_ImageDecoderPartialShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif (reader == "Caffe2Reader" or reader == "Caffe2ReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 1200,
            "max_height": 1200,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.Caffe2_ImageDecoderPartialShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE_ORIG,
            "max_width": 2000,
            "max_height": 2000,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.FusedDecoderCropShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (image_decoder_slice)
