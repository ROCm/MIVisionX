import amd.rocal.types as types
import rocal_pybind as b
from amd.rocal.pipeline import Pipeline

def image(*inputs, user_feature_key_map = None, path='', file_root ='', annotations_file= '', shard_id = 0, num_shards = 1, random_shuffle = False, affine=True, bytes_per_sample_hint=0, cache_batch_copy= True, cache_debug = False, cache_size = 0, cache_threshold = 0,
                 cache_type='', device_memory_padding=16777216, host_memory_padding=8388608, hybrid_huffman_threshold= 1000000, output_type = types.RGB,
                 preserve=False, seed=1, split_stages=False, use_chunk_allocator= False, use_fast_idct = False, device = None):
    b.setSeed(seed)
    reader = Pipeline._current_pipeline._reader
    if( reader == 'COCOReader'):
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
            "max_height":0}
        decoded_image = b.COCO_ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

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
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 2000,
            "max_height": 2000}
        decoded_image   = b.TF_ImageDecoder(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

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
            "max_height":0}
        decoded_image = b.Caffe2_ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

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
            "max_height":0}
        decoded_image = b.Caffe_ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    else:
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 2000,
            "max_height":2000,
            "dec_type":types.DECODER_TJPEG}
        decoded_image = b.ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    return (decoded_image)


def image_raw(*inputs, user_feature_key_map = None, path='', file_root ='', annotations_file= '', shard_id = 0, num_shards = 1, random_shuffle = False, affine=True, bytes_per_sample_hint=0, cache_batch_copy= True, cache_debug = False, cache_size = 0, cache_threshold = 0,
                 cache_type='', device_memory_padding=16777216, host_memory_padding=8388608, hybrid_huffman_threshold= 1000000, output_type = types.RGB,
                 preserve=False, seed=1, split_stages=False, use_chunk_allocator= False, use_fast_idct = False, device = None):
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
        decoded_image   = b.TF_ImageDecoderRaw(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
        # decoded_image = b.TF_ImageDecoderRaw(handle, input_image, self._user_feature_key_map["image/encoded"], self._user_feature_key_map["image/filename"], , is_output, shuffle, False, decode_width, decode_height)
        return (decoded_image)


def image_random_crop(*inputs,user_feature_key_map=None ,path = '', file_root= '', annotations_file='', num_shards = 1, shard_id = 0, random_shuffle = False, affine=True, bytes_per_sample_hint=0, device_memory_padding= 16777216, host_memory_padding = 8388608, hybrid_huffman_threshold = 1000000,
                 num_attempts=10, output_type=types.RGB, preserve=False, random_area = None, random_aspect_ratio = None,
                 seed=1, split_stages=False, use_chunk_allocator=False, use_fast_idct= False, device = None):

    reader = Pipeline._current_pipeline._reader
    b.setSeed(seed)
    #Creating 2 Nodes here (Image Decoder + Random Crop Node)
    #Node 1 Image Decoder
    if( reader == 'COCOReader'):
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
            "max_height":0}
        image_decoder_output_image = b.COCO_ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
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
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 2000,
            "max_height": 2000}
        image_decoder_output_image   = b.TF_ImageDecoder(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

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
            "max_height":0}
        image_decoder_output_image = b.Caffe2_ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    elif (reader == "CaffeReader" or reader == "CaffeReaderDetection"):
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
            "max_height":0}
        image_decoder_output_image = b.Caffe_ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    else:
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 2000,
            "max_height":2000}
        image_decoder_output_image = b.ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))


    #Node 2 Random Crop
    kwargs_pybind_2 = {
        "input_image0": image_decoder_output_image,
        'is_output': False,
        "crop_width": None,
        "crop_height": None,
        "crop_depth": None,
        "crop_pox_x": None,
        "crop_pos_y": None,
        "crop_pox_z": None
    }
    crop_output_image = b.Crop(Pipeline._current_pipeline._handle ,*(kwargs_pybind_2.values()))

    return (crop_output_image)


def image_slice(*inputs,file_root='',path='',annotations_file='',shard_id = 0, num_shards = 1, random_shuffle = False, affine = True, axes = None, axis_names = "WH",bytes_per_sample_hint = 0, device_memory_padding = 16777216,
                device_memory_padding_jpeg2k = 0, host_memory_padding = 8388608,
                host_memory_padding_jpeg2k = 0, hybrid_huffman_threshold = 1000000,
                 memory_stats = False, normalized_anchor = True, normalized_shape = True, output_type = types.RGB,
                preserve = False, seed = 1, split_stages = False, use_chunk_allocator = False, use_fast_idct = False,device = None):


    reader = Pipeline._current_pipeline._reader
    b.setSeed(seed)
    #Reader -> Randon BBox Crop -> ImageDecoderSlice
    if( reader == 'COCOReader'):

        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "shard_count": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 1200, #TODO: what happens when we give user given size = multiplier * max_decoded_width
            "max_height":1200, #TODO: what happens when we give user given size = multiplier * max_decoded_width
            "area_factor": None,
            "aspect_ratio": None,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.COCO_ImageDecoderSliceShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    elif (reader == "CaffeReader" or reader == "CaffeReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.MAX_SIZE,
            "max_width": 1200,
            "max_height":1200,
            "area_factor": None,
            "aspect_ratio": None,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.Caffe_ImageDecoderPartialShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
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
            "max_width": 1200,
            "max_height":1200,
            "area_factor": None,
            "aspect_ratio": None,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.Caffe2_ImageDecoderPartialShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    else :
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 3000,
            "max_height":3000,
            "area_factor": None,
            "aspect_ratio": None,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.FusedDecoderCropShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (image_decoder_slice)
