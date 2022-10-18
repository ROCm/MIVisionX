
from tkinter import W
from amd.rocal.pipeline import Pipeline
from amd.rocal.pipeline import pipeline_def
from amd.rocal.plugin.pytorch import ROCALGenericImageIterator
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cv2
#%matplotlib inline

seed = 1549361629
image_dir = "/home/rajy/work/MIVisionX-data/rocal_data/images_jpg/labels_folder"
batch_size = 4
gpu_id = 0

def show_images(image_batch, device):
    columns = 4
    rows = (batch_size + 1) // (columns)
    #fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        print('\n Display image: ', j)
        plt.subplot(gs[j])
        img = image_batch[j]
        plt.axis("off")
        if device == "cpu":
            plt.imshow(img) #image = img.detach().numpy()
        else:
            plt.imshow(img.cpu())
    plt.show()


def show_pipeline_output(pipe, device):
    pipe.build()
    data_loader = ROCALClassificationIterator(pipe, device)
    images = next(iter(data_loader))
    show_images(images[0], device)

rocal_device = "cpu"
@pipeline_def(seed=seed)
def image_decoder_pipeline(device="cpu"):
    jpegs, labels = fn.readers.file(file_root=image_dir, shard_id=0, num_shards=1, random_shuffle=False)
    images = fn.decoders.image(jpegs, file_root=image_dir, device=rocal_device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    return fn.resize(images, device=rocal_device, resize_x=300, resize_y=300)

pipe = image_decoder_pipeline(batch_size=batch_size, num_threads=1, device_id=gpu_id, rocal_cpu=True, tensor_layout=types.NHWC, reverse_channels=True, multiplier = [0.00392,0.00392,0.00392])
show_pipeline_output(pipe, device=rocal_device)