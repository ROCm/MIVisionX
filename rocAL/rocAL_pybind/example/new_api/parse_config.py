from argparse import ArgumentParser
import random

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")

    common_group = parser.add_argument_group('common', 'common-related options')
     # Data-related
    common_group.add_argument('--image-dataset-path', '-d', type=str,
                        help='image folder files')
    common_group.add_argument('--batch-size', '-b', type=int, default=10,
                        help='number of examples for each iteration')
    common_group.add_argument('--display', default=False, action="store_true",
                        help='--display:to display output from the pipeline')
    common_group.add_argument('--print_tensor', default=False, action="store_true",
                        help='--print_tensor: to print tensor output from the pipeline')
    common_group.add_argument('--classification', default=False, action="store_true",
                        help='--classification: to use for classification')
    common_group.add_argument('--rocal-gpu', default=False, action="store_true",
                        help='--use_gpu to use gpu')
    common_group.add_argument('--NHWC', default=False, action='store_true',
                        help='run input pipeline NHWC format')
    common_group.add_argument('--fp16', default=False, action='store_true',
                        help='run input pipeline fp16 format')
    common_group.add_argument('--local-rank', type=int, default=0,
                        help='number of examples for each iteration')
    common_group.add_argument('--world-size', '-w', type=int, default=1,
                        help='number of examples for each iteration')
    common_group.add_argument('--num-threads', '-nt', type=int, default=1,
                        help='number of examples for each iteration')
    common_group.add_argument('--num-epochs', '-e', type=int, default=1,
                        help='number of epochs to run')
    common_group.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed')


    # rocAL_api_python_unittest.py related options
    python_unit_test = parser.add_argument_group('python-unittest', 'python-unittest-related options')
    python_unit_test.add_argument('--augmentation-name', '-aug_name', type=str, default="resize",
                        help='refer python unit test for all augmentation names ')
    # rocAL_api_coco_pipeline.py related options
    coco_pipeline = parser.add_argument_group('coco-pipeline', 'coco-pipeline-related options')
    coco_pipeline.add_argument('--json-path', '-json-path', type=str,
                        help='coco dataset json path')
    # rocAL_api_caffe_reader.py related options
    caffe_pipeline = parser.add_argument_group('caffe-pipeline', 'caffe-pipeline-related options')
    caffe_pipeline.add_argument('--detection', '-detection', type=str,
                        help='detection')
    # rocAL_api_video_pipeline.py related options
    video_pipeline = parser.add_argument_group('video-pipeline', 'video-pipeline-related options')
    video_pipeline.add_argument('--video-path', '-video-path', type=str,
                        help='video path')
    video_pipeline.add_argument('--sequence-length', '-sequence-length', type=int,
                        help='video path')

    return parser.parse_args()

