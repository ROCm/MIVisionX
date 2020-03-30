import rali_pybind as b
import amd.rali.types as types

class Node:
    def __init__(self):
        self.data = None
        self.prev = None
        self.next = None

class FileReader(Node):
    """
    file_root (str) – Path to a directory containing data files. FileReader supports flat directory structure. file_root directory should contain directories with images in them. To obtain labels FileReader sorts directories in file_root in alphabetical order and takes an index in this order as a class label.

    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    file_list (str, optional, default = '') – Path to the file with a list of pairs file label (leave empty to traverse the file_root directory to obtain files and labels)

    initial_fill (int, optional, default = 1024) – Size of the buffer used for shuffling. If random_shuffle is off then this parameter is ignored.

    lazy_init (bool, optional, default = False) – If set to true, Loader will parse and prepare the dataset metadata only during the first Run instead of in the constructor.

    num_shards (int, optional, default = 1) – Partition the data into this many parts (used for multiGPU training).

    pad_last_batch (bool, optional, default = False) – If set to true, the Loader will pad the last batch with the last image when the batch size is not aligned with the shard size.

    prefetch_queue_depth (int, optional, default = 1) – Specifies the number of batches prefetched by the internal Loader. To be increased when pipeline processing is CPU stage-bound, trading memory consumption for better interleaving with the Loader thread.

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    random_shuffle (bool, optional, default = False) – Whether to randomly shuffle data. Prefetch buffer of initial_fill size is used to sequentially read data and then randomly sample it to form a batch.

    read_ahead (bool, optional, default = False) – Whether accessed data should be read ahead. In case of big files like LMDB, RecordIO or TFRecord it will slow down first access but will decrease the time of all following accesses.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    shard_id (int, optional, default = 0) – Id of the part to read.

    shuffle_after_epoch (bool, optional, default = False) – If true, reader shuffles whole dataset after each epoch. It is exclusive with stick_to_shard and random_shuffle.

    skip_cached_images (bool, optional, default = False) – If set to true, loading data will be skipped when the sample is present in the decoder cache. In such case the output of the loader will be empty

    stick_to_shard (bool, optional, default = False) – Whether reader should stick to given data shard instead of going through the whole dataset. When decoder caching is used, it reduces significantly the amount of data to be cached, but could affect accuracy in some cases

    tensor_init_bytes (int, optional, default = 1048576) – Hint for how much memory to allocate per image.
    """

    def __init__(self, file_root, bytes_per_sample_hint = 0, file_list = '', initial_fill = '', lazy_init = '', num_shards = 1,
                pad_last_batch = False, prefetch_queue_depth = 1, preserve = False, random_shuffle = False,read_ahead = False,
                seed = -1, shard_id = 0, shuffle_after_epoch = False, skip_cached_images = False, stick_to_shard = False, tensor_init_bytes = 1048576, device = None):
        super(FileReader, self).__init__()
        self._file_root = file_root
        self._bytes_per_sample_hint = bytes_per_sample_hint
        self._file_list = file_list
        self._initial_fill = initial_fill
        self._lazy_init = lazy_init
        self._num_shards = num_shards
        self._pad_last_batch = pad_last_batch
        self._prefetch_queue_depth = prefetch_queue_depth
        self._preserve = preserve
        self._random_shuffle = random_shuffle
        self._read_ahead = read_ahead
        self._seed = seed
        self._shard_id = shard_id
        self._shuffle_after_epoch = shuffle_after_epoch
        self._skip_cached_images = skip_cached_images
        self._stick_to_shard = stick_to_shard
        self._tensor_init_bytes = tensor_init_bytes
        self._labels = []
        self.output = Node()
    
    def __call__(self,name = ""):
        self.data = "FileReader"
        self.prev = None
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        self.output.data = self._file_root
        return self.output,self._labels

    def rali_c_func_call(self,handle):
        b.labelReader(handle,self._file_root)
        return self._file_root
    
    
    
    

class ImageDecoder(Node):
    """
        affine (bool, optional, default = True) – `mixed` backend only If internal threads should be affined to CPU cores

        bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

        cache_batch_copy (bool, optional, default = True) – `mixed` backend only If true, multiple images from cache are copied with a single batched copy kernel call; otherwise, each image is copied using cudaMemcpy unless order in the batch is the same as in the cache

        cache_debug (bool, optional, default = False) – `mixed` backend only Print debug information about decoder cache.

        cache_size (int, optional, default = 0) – `mixed` backend only Total size of the decoder cache in megabytes. When provided, decoded images bigger than cache_threshold will be cached in GPU memory.

        cache_threshold (int, optional, default = 0) – `mixed` backend only Size threshold (in bytes) for images (after decoding) to be cached.

        cache_type (str, optional, default = '') – `mixed` backend only Choose cache type: threshold: Caches every image with size bigger than cache_threshold until cache is full. Warm up time for threshold policy is 1 epoch. largest: Store largest images that can fit the cache. Warm up time for largest policy is 2 epochs To take advantage of caching, it is recommended to use the option stick_to_shard=True with the reader operators, to limit the amount of unique images seen by the decoder in a multi node environment

        device_memory_padding (int, optional, default = 16777216) – `mixed` backend only Padding for nvJPEG’s device memory allocations in bytes. This parameter helps to avoid reallocation in nvJPEG whenever a bigger image is encountered and internal buffer needs to be reallocated to decode it.

        host_memory_padding (int, optional, default = 8388608) – `mixed` backend only Padding for nvJPEG’s host memory allocations in bytes. This parameter helps to avoid reallocation in nvJPEG whenever a bigger image is encountered and internal buffer needs to be reallocated to decode it.

        hybrid_huffman_threshold (int, optional, default = 1000000) – `mixed` backend only Images with number of pixels (height * width) above this threshold will use the nvJPEG hybrid Huffman decoder. Images below will use the nvJPEG full host huffman decoder. N.B.: Hybrid Huffman decoder still uses mostly the CPU.

        output_type (int, optional, default = 0) – The color space of output image.

        preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

        seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

        split_stages (bool, optional, default = False) – `mixed` backend only Split into separated CPU stage and GPU stage operators

        use_chunk_allocator (bool, optional, default = False) – Experimental, `mixed` backend only Use chunk pinned memory allocator, allocating chunk of size batch_size*prefetch_queue_depth during the construction and suballocate them in runtime. Ignored when split_stages is false.

        use_fast_idct (bool, optional, default = False) – Enables fast IDCT in CPU based decompressor when GPU implementation cannot handle given image. According to libjpeg-turbo documentation, decompression performance is improved by 4-14% with very little loss in quality.
    """
    def __init__(self, affine = True, bytes_per_sample_hint = 0, cache_batch_copy = True, cache_debug = False, cache_size = 0, cache_threshold = 0,
                cache_type = '', device_memory_padding = 16777216, host_memory_padding = 8388608, hybrid_huffman_threshold = 1000000, output_type = 0,
                preserve = False, seed = -1, split_stages = False, use_chunk_allocator = False, use_fast_idct = False, device = None):
        super(ImageDecoder, self).__init__()
        self._affine = affine
        self._bytes_per_sample_hint = bytes_per_sample_hint
        self._cache_batch_copy = cache_batch_copy
        self._cache_debug = cache_debug
        self._cache_size = cache_size
        self._cache_threshold = cache_threshold
        self._cache_type = cache_type
        self._device_memory_padding = device_memory_padding
        self._host_memory_padding = host_memory_padding
        self._hybrid_huffman_threshold = hybrid_huffman_threshold
        self._output_type = output_type
        self._preserve = preserve
        self._seed = seed
        self._split_stages = split_stages
        self._use_chunk_allocator = use_chunk_allocator
        self._use_fast_idct = use_fast_idct
        self.output = Node()


    def __call__(self,input, num_threads=1):
        input.next = self
        self.data = "ImageDecoder"
        self.prev = input
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        return self.output

    def rali_c_func_call(self, handle, input_image, is_output):
        num_threads = 1
        if decode_width != None and decode_height != None:
            multiplier = 4
            output_image = b.ImageDecoder(handle, input_image, types.RGB, num_threads, False, False, types.USER_GIVEN_SIZE, multiplier*decode_width, multiplier*decode_height)
        else:
            output_image = b.ImageDecoder(handle, input_image, types.RGB, num_threads, is_output, False)
        return output_image


class ImageDecoderRandomCrop(Node):
    """
        affine (bool, optional, default = True) – `mixed` backend only If internal threads should be affined to CPU cores

        bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

        device_memory_padding (int, optional, default = 16777216) – `mixed` backend only Padding for nvJPEG’s device memory allocations in bytes. This parameter helps to avoid reallocation in nvJPEG whenever a bigger image is encountered and internal buffer needs to be reallocated to decode it.

        host_memory_padding (int, optional, default = 8388608) – `mixed` backend only Padding for nvJPEG’s host memory allocations in bytes. This parameter helps to avoid reallocation in nvJPEG whenever a bigger image is encountered and internal buffer needs to be reallocated to decode it.

        hybrid_huffman_threshold (int, optional, default = 1000000) – `mixed` backend only Images with number of pixels (height * width) above this threshold will use the nvJPEG hybrid Huffman decoder. Images below will use the nvJPEG full host huffman decoder. N.B.: Hybrid Huffman decoder still uses mostly the CPU.

        num_attempts (int, optional, default = 10) – Maximum number of attempts used to choose random area and aspect ratio.

        output_type (int, optional, default = 0) – The color space of output image.

        preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

        random_area (float or list of float, optional, default = [0.08, 1.0]) – Range from which to choose random area factor A. The cropped image’s area will be equal to A * original image’s area.

        random_aspect_ratio (float or list of float, optional, default = [0.75, 1.333333]) – Range from which to choose random aspect ratio (width/height).

        seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

        split_stages (bool, optional, default = False) – `mixed` backend only Split into separated CPU stage and GPU stage operators

        use_chunk_allocator (bool, optional, default = False) – Experimental, `mixed` backend only Use chunk pinned memory allocator, allocating chunk of size batch_size*prefetch_queue_depth during the construction and suballocate them in runtime. Ignored when split_stages is false.

        use_fast_idct (bool, optional, default = False) – Enables fast IDCT in CPU based decompressor when GPU implementation cannot handle given image. According to libjpeg-turbo documentation, decompression performance is improved by 4-14% with very little loss in quality.
    """
    def __init__(self, affine = True, bytes_per_sample_hint = 0, device_memory_padding = 16777216, host_memory_padding = 8388608, hybrid_huffman_threshold = 1000000, 
                num_attempts = 10, output_type = 0,preserve = False, random_area = [0.04, 0.8], random_aspect_ratio = [0.75, 1.333333],
                seed = 1, split_stages = False, use_chunk_allocator = False, use_fast_idct = False, device = None):
        super(ImageDecoderRandomCrop, self).__init__()
        self._affine = affine
        self._bytes_per_sample_hint = bytes_per_sample_hint
        self._device_memory_padding = device_memory_padding
        self._host_memory_padding = host_memory_padding
        self._hybrid_huffman_threshold = hybrid_huffman_threshold
        self._num_attempts = num_attempts
        self._output_type = output_type
        self._preserve = preserve
        self._random_area = random_area
        self._random_aspect_ratio = random_aspect_ratio
        self._seed = seed
        self._split_stages = split_stages
        self._use_chunk_allocator = use_chunk_allocator
        self._use_fast_idct = use_fast_idct
        self.output = Node()

    def __call__(self, input):
        input.next = self
        self.data = "ImageDecoderRandomCrop"
        self.prev = input
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        return self.output

    def rali_c_func_call(self, handle, input_image, decode_width, decode_height, is_output):
        b.setSeed(self._seed)
        num_threads = 1
        if decode_width != None and decode_height != None:
            multiplier = 4
            output_image = b.ImageDecoder(handle, input_image, types.RGB, num_threads, False, False, types.USER_GIVEN_SIZE, multiplier*decode_width, multiplier*decode_height)
        else:
            output_image = b.ImageDecoder(handle, input_image, types.RGB, num_threads, is_output, False)
        self.area = b.CreateFloatUniformRand(self._random_area[0],self._random_area[1])
        self.aspect_ratio = b.CreateFloatUniformRand(self._random_aspect_ratio[0],self._random_aspect_ratio[1])
        self.x_drift =  b.CreateFloatParameter(0)
        self.y_drift =  b.CreateFloatParameter(0)
        output_image = b.Crop(handle, output_image, is_output, None, None, None, None, None, None)
        # output_image = b.CropResize(handle, output_image, 500, 500, is_output, self.area, self.aspect_ratio, self.x_drift, self.y_drift)
        # output_image = b.CropResize(handle, output_image, 0, 0, is_output, None, None, None, None)
        return output_image

class ColorTwist(Node):
    """
        brightness (float, optional, default = 1.0) –

        Brightness change factor. Values >= 0 are accepted. For example:

            0 - black image,

            1 - no change

            2 - increase brightness twice

        bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

        contrast (float, optional, default = 1.0) –

        Contrast change factor. Values >= 0 are accepted. For example:

            0 - gray image,

            1 - no change

            2 - increase contrast twice

        hue (float, optional, default = 0.0) – Hue change, in degrees.

        image_type (int, optional, default = 0) – The color space of input and output image

        preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

        saturation (float, optional, default = 1.0) –

        Saturation change factor. Values >= 0 are supported. For example:

            0 - completely desaturated image

            1 - no change to image’s saturation

        seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)
    """
    def __init__(self, brightness = 1.0, bytes_per_sample_hint = 0, contrast = 1.0, hue = 0.0, image_type = 0, 
                preserve = False, saturation = 1.0,seed = -1, device = None):
        super(ColorTwist, self).__init__()
        self._brightness = brightness
        self._bytes_per_sample_hint = bytes_per_sample_hint
        self._contrast = contrast
        self._hue = hue
        self._image_type = image_type
        self._preserve = preserve
        self._saturation = saturation
        self._seed = seed
        self.output = Node()

    def __call__(self,input):
        input.next = self
        self.data = "ColorTwist"
        self.prev = input
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        return self.output

    def rali_c_func_call(self, handle, input_image, is_output):
        # b.setSeed(self._seed)
        # output_image = b.ColorTwist(handle, input_image, is_output,None,None,None,None)
        output_image = b.ColorTwistFixed(handle, input_image, self._brightness,self._contrast,self._hue,self._saturation,is_output)
        return output_image


class Resize(Node):
    """
        bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

        image_type (int, optional, default = 0) – The color space of input and output image.

        interp_type (int, optional, default = 1) – Type of interpolation used. Use min_filter and mag_filter to specify different filtering for downscaling and upscaling.

        mag_filter (int, optional, default = 1) – Filter used when scaling up

        max_size (float or list of float, optional, default = [0.0, 0.0]) –

        Maximum size of the longer dimension when resizing with resize_shorter. When set with resize_shorter, the shortest dimension will be resized to resize_shorter iff the longest dimension is smaller or equal to max_size. If not, the shortest dimension is resized to satisfy the constraint longest_dim == max_size. Can be also an array of size 2, where the two elements are maximum size per dimension (H, W).

        Example:

        Original image = 400x1200.

        Resized with:

            resize_shorter = 200 (max_size not set) => 200x600

            resize_shorter = 200, max_size =  400 => 132x400

            resize_shorter = 200, max_size = 1000 => 200x600

        min_filter (int, optional, default = 1) – Filter used when scaling down

        minibatch_size (int, optional, default = 32) – Maximum number of images processed in a single kernel call

        preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

        resize_longer (float, optional, default = 0.0) – The length of the longer dimension of the resized image. This option is mutually exclusive with resize_shorter,`resize_x` and resize_y. The op will keep the aspect ratio of the original image.

        resize_shorter (float, optional, default = 0.0) – The length of the shorter dimension of the resized image. This option is mutually exclusive with resize_longer, resize_x and resize_y. The op will keep the aspect ratio of the original image. The longer dimension can be bounded by setting the max_size argument. See max_size argument doc for more info.

        resize_x (float, optional, default = 0.0) – The length of the X dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_y is left at 0, then the op will keep the aspect ratio of the original image.

        resize_y (float, optional, default = 0.0) – The length of the Y dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_x is left at 0, then the op will keep the aspect ratio of the original image.

        save_attrs (bool, optional, default = False) – Save reshape attributes for testing.

        seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

        temp_buffer_hint (int, optional, default = 0) – Initial size, in bytes, of a temporary buffer for resampling. Ingored for CPU variant.
    """
    def __init__(self, bytes_per_sample_hint = 0, image_type = 0, interp_type = 1, mag_filter = 1, max_size = [0.0, 0.0], min_filter = 1,
                minibatch_size = 32, preserve = False, resize_longer = 0.0, resize_shorter = 0.0, resize_x = 0.0, resize_y = 0.0,
                save_attrs = False,seed = 1, temp_buffer_hint = 0, device = None):
        super(Resize, self).__init__()
        self._bytes_per_sample_hint = bytes_per_sample_hint
        self._image_type = image_type
        self._interp_type = interp_type
        self._mag_filter = mag_filter
        self._max_size = max_size
        self._min_filter = min_filter
        self._minibatch_size = minibatch_size
        self._preserve = preserve
        self._resize_longer = resize_longer
        self._resize_shorter = resize_shorter
        self._resize_x = resize_x
        self._resize_y = resize_y
        self._save_attrs = save_attrs
        self._seed = seed
        self._temp_buffer_hint = temp_buffer_hint
        self.output = Node()
    
    def __call__(self,input):
        input.next = self
        self.data = "Resize"
        self.prev = input
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        return self.output
    
    def rali_c_func_call(self, handle, input_image, is_output):
        output_image = b.Resize(handle, input_image, self._resize_x, self._resize_y, is_output)
        return output_image

class  CropMirrorNormalize(Node):
    """
        bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

        crop (float or list of float, optional, default = [0.0, 0.0]) – Shape of the cropped image, specified as a list of value (e.g. (crop_H, crop_W) for 2D crop, (crop_D, crop_H, crop_W) for volumetric crop). Providing crop argument is incompatible with providing separate arguments crop_d, crop_h and crop_w.

        crop_d (float, optional, default = 0.0) – Volumetric inputs only cropping window depth (in pixels). If provided, crop_h and crop_w should be provided as well. Providing crop_w, crop_h, crop_d is incompatible with providing fixed crop window dimensions (argument crop).

        crop_h (float, optional, default = 0.0) – Cropping window height (in pixels). If provided, crop_w should be provided as well. Providing crop_w, crop_h is incompatible with providing fixed crop window dimensions (argument crop).

        crop_pos_x (float, optional, default = 0.5) – Normalized (0.0 - 1.0) horizontal position of the cropping window (upper left corner). Actual position is calculated as crop_x = crop_x_norm * (W - crop_W), where crop_x_norm is the normalized position, W is the width of the image and crop_W is the width of the cropping window.

        crop_pos_y (float, optional, default = 0.5) – Normalized (0.0 - 1.0) vertical position of the cropping window (upper left corner). Actual position is calculated as crop_y = crop_y_norm * (H - crop_H), where crop_y_norm is the normalized position, H is the height of the image and crop_H is the height of the cropping window.

        crop_pos_z (float, optional, default = 0.5) – Volumetric inputs only Normalized (0.0 - 1.0) normal position of the cropping window (front plane). Actual position is calculated as crop_z = crop_z_norm * (D - crop_d), where crop_z_norm is the normalized position, D is the depth of the image and crop_d is the depth of the cropping window.

        crop_w (float, optional, default = 0.0) – Cropping window width (in pixels). If provided, crop_h should be provided as well. Providing crop_w, crop_h is incompatible with providing fixed crop window dimensions (argument crop).

        image_type (int, optional, default = 0) – The color space of input and output image

        mean (float or list of float, optional, default = [0.0]) – Mean pixel values for image normalization.

        mirror (int, optional, default = 0) – Mask for horizontal flip. - 0 - do not perform horizontal flip for this image - 1 - perform horizontal flip for this image.

        output_dtype (int, optional, default = 9) – Output data type. Supported types: FLOAT and FLOAT16

        output_layout (str, optional, default = 'CHW') – Output tensor data layout

        pad_output (bool, optional, default = False) – Whether to pad the output to number of channels being a power of 2.

        preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

        seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

        std (float or list of float, optional, default = [1.0]) – Standard deviation values for image normalization.
    """
    def __init__(self, bytes_per_sample_hint = 0, crop = [0.0, 0.0], crop_d = 0, crop_h = 0, crop_pos_x = 0.5, crop_pos_y = 0.5, crop_pos_z = 0.5,
                crop_w = 0 , image_type = 0, mean = [0.0], mirror = 0, output_dtype = types.FLOAT, output_layout = types.NCHW, pad_output = False, 
                preserve = False, seed = 1, std = [1.0], device = None):
        super(CropMirrorNormalize, self).__init__()
        self._bytes_per_sample_hint = bytes_per_sample_hint
        self._crop = crop
        if(len(crop) == 2):
            self._crop_d = crop_d
            self._crop_h = crop[0]
            self._crop_w = crop[1]
        elif(len(crop) == 3):
            self._crop_d = crop[0]
            self._crop_h = crop[1]
            self._crop_w = crop[2]
        else:
            self._crop_d = crop_d
            self._crop_h = crop_h
            self._crop_w = crop_w
        self._crop_pos_x = crop_pos_x
        self._crop_pos_y = crop_pos_y
        self._crop_pos_z = crop_pos_z
        self._image_type = image_type
        self._mean = mean 
        self._mirror = mirror
        self._output_dtype = output_dtype
        self._output_layout = output_layout
        self._pad_output = pad_output
        self._preserve = preserve
        self._seed = seed
        self._std = std
        self.output = Node()
        self._temp = None

    def __call__(self, input, mirror = None):
        input.next = self
        self.data = "CropMirrorNormalize"
        self.prev = input
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        self._temp = mirror
        return self.output
    
    def rali_c_func_call(self, handle, input_image, is_output):
        b.setSeed(self._seed)
        output_image = []
        # output_image = b.CropMirrorNormalize(handle, input_image, 500, 500, is_output,None, None, None, None, None, None, None)
        if self._temp is not None:
            mirror = self._temp.rali_c_func_call(handle)
            output_image = b.CropMirrorNormalize(handle, input_image, self._crop_d, self._crop_h, self._crop_w, 1,
                                            1, 1, self._mean, self._std, is_output, mirror)
        else:
            if(self._mirror == 0):
                mirror = b.CreateIntParameter(0)
            else:
                mirror = b.CreateIntParameter(1)
            output_image = b.CropMirrorNormalize(handle, input_image, self._crop_d, self._crop_h, self._crop_w, self._crop_pos_x,
                                            self._crop_pos_y, self._crop_pos_z, self._mean, self._std, is_output, mirror)
        return output_image


class Crop(Node):
    

    """
bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

crop (float or list of float, optional, default = [0.0, 0.0]) – Shape of the cropped image, specified as a list of value (e.g. (crop_H, crop_W) for 2D crop, (crop_D, crop_H, crop_W) for volumetric crop). Providing crop argument is incompatible with providing separate arguments crop_d, crop_h and crop_w.

crop_d (float, optional, default = 0.0) – Volumetric inputs only cropping window depth (in pixels). If provided, crop_h and crop_w should be provided as well. Providing crop_w, crop_h, crop_d is incompatible with providing fixed crop window dimensions (argument crop).

crop_h (float, optional, default = 0.0) – Cropping window height (in pixels). If provided, crop_w should be provided as well. Providing crop_w, crop_h is incompatible with providing fixed crop window dimensions (argument crop).

crop_pos_x (float, optional, default = 0.5) – Normalized (0.0 - 1.0) horizontal position of the cropping window (upper left corner). Actual position is calculated as crop_x = crop_x_norm * (W - crop_W), where crop_x_norm is the normalized position, W is the width of the image and crop_W is the width of the cropping window.

crop_pos_y (float, optional, default = 0.5) – Normalized (0.0 - 1.0) vertical position of the cropping window (upper left corner). Actual position is calculated as crop_y = crop_y_norm * (H - crop_H), where crop_y_norm is the normalized position, H is the height of the image and crop_H is the height of the cropping window.

crop_pos_z (float, optional, default = 0.5) – Volumetric inputs only Normalized (0.0 - 1.0) normal position of the cropping window (front plane). Actual position is calculated as crop_z = crop_z_norm * (D - crop_d), where crop_z_norm is the normalized position, D is the depth of the image and crop_d is the depth of the cropping window.

crop_w (float, optional, default = 0.0) – Cropping window width (in pixels). If provided, crop_h should be provided as well. Providing crop_w, crop_h is incompatible with providing fixed crop window dimensions (argument crop).

image_type (int, optional, default = 0) – The color space of input and output image

output_dtype (int, optional, default = -1) – Output data type. By default same data type as the input will be used. Supported types: FLOAT, FLOAT16, and UINT8

preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)


    """
    def __init__(self, bytes_per_sample_hint = 0, crop = [0.0, 0.0], crop_d = 0, crop_h = 0, crop_pos_x = 0.5, crop_pos_y = 0.5, crop_pos_z = 0.5,
                crop_w = 0 , image_type = 0, output_dtype = types.FLOAT, preserve = False, seed = 1, device = None):
        super(Crop, self).__init__()       
        self._bytes_per_sample_hint = bytes_per_sample_hint
        self._crop = crop
        if(len(crop) == 2):
            self._crop_h = crop[0]
            self._crop_w = crop[1]
            self._crop_d = crop_d
        elif(len(crop) == 3):
            self._crop_d = crop[0]
            self._crop_h = crop[1]
            self._crop_w = crop[2]
        else:
            self._crop_d = crop_d
            self._crop_h = crop_h
            self._crop_w = crop_w
        self._crop_pos_x = crop_pos_x
        self._crop_pos_y = crop_pos_y
        self._crop_pos_z = crop_pos_z
        self._image_type = image_type
        self._output_dtype = output_dtype
        self._preserve = preserve
        self._seed = seed
        self.output = Node()
        self._temp = None

                
    def __call__(self, input, is_output = False):
        input.next = self
        self.data = "Crop"
        self.prev = input
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        return self.output
    
    def rali_c_func_call(self, handle, input_image, is_output):
        b.setSeed(self._seed)
        output_image = []
        crop_h = b.CreateFloatParameter(self._crop_h)
        crop_w = b.CreateFloatParameter(self._crop_w)
        crop_d = b.CreateFloatParameter(self._crop_d)
        crop_pos_x = b.CreateFloatParameter(self._crop_pos_x)
        crop_pos_y = b.CreateFloatParameter(self._crop_pos_y)
        crop_pos_z = b.CreateFloatParameter(self._crop_pos_z)
        # output_image = b.Crop(handle, input_image, is_output, crop_w, crop_h, crop_d, crop_pos_x, crop_pos_y, crop_pos_z)
        output_image = b.Crop(handle, output_image, is_output, None, None, None, None, None, None)
        return output_image 


class CoinFlip():
    def __init__(self,probability=0.5, device = None):
        self._probablility=probability
        self.output=Node()

    def __call__(self):
        return self

    def rali_c_func_call(self, handle):
        self._values = [0,1]
        self._frequencies = [1-self._probablility, self._probablility]
        output_arr = b.CreateIntRand(self._values, self._frequencies)
        return output_arr

class GammaCorrection(Node):
    _gamma = 0

    def __init__(self,gamma, device = None):
        super(GammaCorrection, self).__init__()
        self._gamma = gamma
        self.output = Node()
    
    def __call__(self, input):
        input.next = self
        self.data = "GammaCorrection"
        self.prev = input
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        return self.output

    def rali_c_func_call(self, handle, input_image, is_output):
        output_image = b.GammaCorrection(handle, input_image, is_output, None)
        return output_image

       
class Snow(Node):
    _snow = 0

    def __init__(self,snow, device = None):
        super(Snow, self).__init__()
        self._snow = snow
        self.output = Node()
    
    def __call__(self, input):
        input.next = self
        self.data = "Snow"
        self.prev = input
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        return self.output

    def rali_c_func_call(self, handle, input_image, is_output):
        output_image = b.Snow(handle, input_image, is_output, None)
        return output_image

class Rain(Node):
    _rain = 0

    def __init__(self,rain, device = None):
        super(Rain, self).__init__()
        self._rain = rain
        self.output = Node()
    
    def __call__(self,input):
        input.next = self
        self.data = "Rain"
        self.prev = input
        self.next = self.output
        self.output.prev = self
        self.output.next = None
        return self.output

    def rali_c_func_call(self, handle, input_image, is_output):
        output_image = b.Rain(handle, input_image, is_output,None,None,None,None)
        return output_image
