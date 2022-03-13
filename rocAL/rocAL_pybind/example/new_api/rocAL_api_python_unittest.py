from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from amd.rocal.plugin.pytorch import RALIClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
from parse_config import parse_args
import os

def draw_patches(img, idx):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + str(idx)+"_"+"train"+".png", image)


def main():
    args = parse_args()
    # Args
    data_path = args.image_dataset_path
    augmentation_num = args.augmentation_number
    rali_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    num_threads = args.num_threads
    random_seed = args.seed
    local_rank =  args.local_rank
    world_size =  args.world_size
    display = True if args.display else False
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/"
        os.makedirs(path)
    except OSError as error:
        print(error)
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=args.local_rank, seed=random_seed, rali_cpu=rali_cpu)
    # Set Params
    output_set = 0
    rali_device = 'cpu' if rali_cpu else 'gpu'
    decoder_device = 'cpu' if rali_cpu else 'gpu'
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        images = fn.decoders.image(jpegs, file_root=data_path, device=decoder_device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=True)
        images = fn.resize(images, device=rali_device, resize_x=300, resize_y=300)

        if augmentation_num == 0 or augmentation_num == 1:
            output = fn.resize(images, resize_x=300, resize_y=300)
        elif augmentation_num == 2:
            output = fn.rotate(images)
        elif augmentation_num == 3:
            output = fn.brightness(images)
        elif augmentation_num == 4:
            output = fn.gamma_correction(images)
        elif augmentation_num == 5:
            output = fn.contrast(images)
        elif augmentation_num == 6:
            output = fn.flip(images)
        elif augmentation_num == 7:
            output = fn.blur(images)
        elif augmentation_num == 8:
            images_hue = fn.hue(images)
            images_rotate = fn.rotate(images)
            output = fn.blend(images_hue, images_rotate)
        elif augmentation_num == 9:
            output = fn.warp_affine(images)
        elif augmentation_num == 10:
            output = fn.fish_eye(images)
        elif augmentation_num == 11:
            output = fn.vignette(images)
        elif augmentation_num == 12:
            output = fn.jitter(images)
        elif augmentation_num == 13:
            output = fn.snp_noise(images)
        elif augmentation_num == 14:
            output = fn.snow(images)
        elif augmentation_num == 15:
            output = fn.rain(images)
        elif augmentation_num == 16:
            output = fn.fog(images)
        elif augmentation_num == 17:
            output = fn.pixelate(images)
        elif augmentation_num == 18:
            output = fn.exposure(images)
        elif augmentation_num == 19:
            output = fn.hue(images)
        elif augmentation_num == 20:
            output = fn.saturation(images)
        elif augmentation_num == 21:
            output = fn.color_twist(images)
        elif augmentation_num == 22:
            output = fn.crop_mirror_normalize(images, device="cpu",
                                              output_dtype=types.FLOAT,
                                              output_layout=types.NCHW,
                                              crop=(300, 300),
                                              image_type=types.RGB,
                                              mean=[0, 0, 0],
                                              std=[1, 1, 1])
        elif augmentation_num == 23:
            output = fn.nop(images)
        elif augmentation_num == 24:
            output = fn.centre_crop(images)
        elif augmentation_num == 25:
            output = fn.color_temp(images)
        elif augmentation_num == 26:
            output = fn.copy(images)
        elif augmentation_num == 27:
            output1 = fn.rotate(images)
            output2 = fn.fish_eye(output1)
            output3 = fn.fog(output2)
            pipe.set_outputs(output1, output2, output3)
            output_set = 1
        elif augmentation_num == 28:
            output1 = fn.resize(images, resize_x=300, resize_y=300)
            output2 = fn.brightness(output1)
            output3 = fn.jitter(output2)
            pipe.set_outputs(output1, output2, output3)
            output_set = 1
        elif augmentation_num == 29:
            output1 = fn.vignette(images)
            output2 = fn.blur(output1)
            pipe.set_outputs(output1, output2)
            output_set = 1

        if output_set==0:
                pipe.set_outputs(output)
    # build the pipeline
    pipe.build()
    # Dataloader
    data_loader = RALIClassificationIterator(pipe)
    cnt = 0

    import timeit
    start = timeit.default_timer()

    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::", epoch)
        for i, it in enumerate(data_loader, 0):
            print("**************", i, "*******************")
            print("**************starts*******************")
            print("\nImages:\n", it[0])
            print("\nLABELS:\n", it[1])
            for img in it[0]:
                cnt = cnt+1
                if display:
                    draw_patches(img, cnt)
            print("**************ends*******************")
            print("**************", i, "*******************")
        data_loader.reset()

    #Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)
    print('Number of times loop iterates is:', cnt)

if __name__ == '__main__':
    main()
