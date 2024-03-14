# Copyright (c) 2018 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from amd.rocal.plugin.generic import ROCAL_iterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import datetime
import time

def HybridTrainPipe(batch_size, num_threads, device_id, data_dir, rocal_cpu = True, prefetch_queue_depth = 2):
    world_size = 1
    local_rank = 0
    resize_width = 300
    resize_height = 300
    rocal_device = 'cpu' if rocal_cpu else 'gpu'
    decoder_device = 'cpu'  # hardcoding decoder_device to cpu until VCN can decode all JPEGs
    
    # Create Pipeline instance
    pipe = Pipeline(batch_size = batch_size, num_threads = num_threads, device_id = device_id, rocal_cpu = rocal_cpu)
    with pipe:
        jpegs, _ = fn.readers.file(file_root = data_dir, shard_id = local_rank, num_shards = world_size, random_shuffle = True)
        images = fn.decoders.image(jpegs, file_root = data_dir, device = decoder_device, output_type = types.RGB, shard_id = local_rank, num_shards = world_size, random_shuffle = True)
        images = fn.resize(images, device = rocal_device, resize_x = resize_width, resize_y = resize_height)
        output = fn.rain(images, rain = 0.5)
        pipe.set_outputs(output)
        
    return pipe

def main():
	if  len(sys.argv) < 5:
		print ('Please pass image_folder cpu/gpu batch_size prefetch_queue_depth')
		exit(0)
	_image_path = sys.argv[1]
	if(sys.argv[2] == "cpu"):
		_rocal_cpu = True
	else:
		_rocal_cpu = False
	_batch_size = int(sys.argv[3])
	_prefetch_queue_depth = int(sys.argv[4])
	_num_threads = 5
	_device_id = 0
	pipe = HybridTrainPipe(batch_size = _batch_size, num_threads = _num_threads, device_id = _device_id,  data_dir = _image_path, rocal_cpu =_rocal_cpu, prefetch_queue_depth = _prefetch_queue_depth)
	pipe.build()
	imageIterator = ROCAL_iterator(pipe)
	start = datetime.datetime.now()
	for _ in range(0, 10):
		for _, (image_batch, image_tensor) in enumerate(imageIterator, 0):
			time.sleep(1)
		imageIterator.reset()
	end = datetime.datetime.now()
	print("Time taken (averaged over 10 runs) ",int((end - start).total_seconds() * 1000),"milli seconds")

if __name__ == '__main__':
    main()
