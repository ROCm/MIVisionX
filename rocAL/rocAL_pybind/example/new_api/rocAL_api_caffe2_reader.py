import sys
from amd.rocal.plugin.pytorch import RALIClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.ops as ops
import amd.rocal.types as types
import amd.rocal.fn as fn
import os
import random


def draw_patches(img, idx):
    #image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/" + str(idx)+"_"+"train"+".png", image)


def main():
    if len(sys.argv) < 4:
        print('Please pass image_folder cpu/gpu batch_size classification/detection')
        exit(0)
    if(sys.argv[2] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    bs = int(sys.argv[3])
    if(sys.argv[4] == "detection"):
        _rali_bbox = True
    else:
        _rali_bbox = False

    nt = 1
    di = 0
    crop_size = 224
    image_path = sys.argv[1]
    rali_device = 'cpu' if _rali_cpu else 'gpu'
    decoder_device = 'cpu' if _rali_cpu else 'gpu'
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    num_classes = len(next(os.walk(image_path))[1])
    print("num_classes:: ", num_classes)

    pipe = Pipeline(batch_size=bs, num_threads=nt, device_id=di,
                    seed=random_seed, rali_cpu=_rali_cpu)
    # pipe = HybridTrainPipe(batch_size=bs, num_threads=nt, device_id=di, data_dir=image_path, crop=crop_size, rali_cpu=_rali_cpu, rali_type=_rali_type)
    with pipe:  # TODO: Need to add oneHotLabels, CMN, CoinFlip
        if _rali_bbox:
            jpegs, labels, bboxes = fn.readers.caffe2(
                path=image_path, bbox=_rali_bbox, random_shuffle=True)
        else:
            jpegs, labels = fn.readers.caffe2(
                path=image_path, bbox=_rali_bbox, random_shuffle=True)
        images = fn.decoders.image(jpegs, output_type=types.RGB, path=image_path, random_shuffle=True)
        images = fn.resize(images, resize_x=crop_size,
                           resize_y=crop_size, device=rali_device)
        pipe.set_outputs(images)
    pipe.build()
    data_loader = RALIClassificationIterator(pipe , display=True)

    # Training loop
    cnt = 0
    for epoch in range(1):  # loop over the dataset multiple times
        print("epoch:: ", epoch)
        if not _rali_bbox:
            for i, (image_batch, labels) in enumerate(data_loader, 0):  # Classification
                sys.stdout.write("\r Mini-batch " + str(i))
                print("Images", image_batch)
                print("Labels", labels)
                for element in list(range(bs)):
                    cnt = cnt + 1
                    draw_patches(image_batch[element],cnt)
            data_loader.reset()
        else:
            for i, (image_batch, bboxes, labels) in enumerate(data_loader, 0):  # Detection
                sys.stdout.write("\r Mini-batch " + str(i))
                print("Images", image_batch)
                print("Bboxes", bboxes)
                print("Labels", labels)
                for element in list(range(bs)):
                    cnt = cnt + 1
                    draw_patches(image_batch[element],cnt)
            data_loader.reset()
    # print('Finished Training')
    # print('Finished !!')


if __name__ == '__main__':
    main()
