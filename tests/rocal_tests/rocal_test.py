from amd.rocal.plugin.generic import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import os

def draw_patches(img, device):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    img = (img).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_image.png", img)

def main():
    # Create Pipeline instance
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=0, seed=1, rocal_cpu=True, tensor_layout=types.NHWC, tensor_dtype=types.FLOAT16)
    # Set Params
    rocal_device = 'cpu'
    #hardcoding decoder_device to cpu until VCN can decode all JPEGs
    decoder_device = 'cpu'
    pwd = os.getcwd()
    data_path = pwd + '/images/'
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_path, shard_id=0, num_shards=1, random_shuffle=True)
        images = fn.decoders.image(jpegs, file_root=data_path, device=decoder_device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=True)
        images = fn.resize(images, device=rocal_device, resize_x=300, resize_y=300)

        output = fn.resize(images, device=rocal_device, resize_x=300, resize_y=300,
                               scaling_mode=types.SCALING_MODE_NOT_SMALLER, interpolation_type=types.TRIANGULAR_INTERPOLATION)
        pipe.set_outputs(output)

    # build the pipeline
    pipe.build()
    # Dataloader
    data_loader = ROCALClassificationIterator(pipe,device='cpu',device_id=0)

    for i, it in enumerate(data_loader, 0):
        print("**************", i, "*******************")
        print("**************starts*******************")
        print("\nImages:\n", it[0])
        print("\nLABELS:\n", it[1])
        print("**************ends*******************")
        print("**************", i, "*******************")
        for img in it[0]:
            draw_patches(img, rocal_device)
    data_loader.reset()

    print("###############################################                             SUCCESS                             ###############################################")

if __name__ == '__main__':
    main()