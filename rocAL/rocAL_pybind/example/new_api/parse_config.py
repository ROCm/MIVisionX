from argparse import ArgumentParser
import random

# adds mutually exclusive "--name" and "--no-name" command line arguments, with
# the result stored in a variable named "name" (with any dashes in "name"
# replaced by underscores)
# inspired by https://stackoverflow.com/a/31347222/2209313
def add_bool_arg(group, name, default=False, help=''):
    subgroup = group.add_mutually_exclusive_group(required=False)
    name_with_underscore = name.replace('-', '_').replace(' ', '_')

    truehelp = help
    falsehelp = help
    if help != '':
        falsehelp = 'do not ' + falsehelp
    if default is True:
        if truehelp != '':
            truehelp = truehelp + ' '
        truehelp = truehelp + '(default)'
    else:
        if falsehelp != '':
            falsehelp = falsehelp + ' '
        falsehelp = falsehelp + '(default)'

    subgroup.add_argument('--' + name, dest=name_with_underscore, action='store_true', help=truehelp)
    subgroup.add_argument('--no-' + name, dest=name_with_underscore, action='store_false', help=falsehelp)
    group.set_defaults(**{name_with_underscore:default})

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")

    common_group = parser.add_argument_group('common', 'common-related options')
     # Data-related
    common_group.add_argument('--image-dataset-path', '-d', type=str,
                        help='image folder files')
    common_group.add_argument('--batch-size', '-b', type=int, default=10,
                        help='number of examples for each iteration')
    add_bool_arg(common_group, 'display', default=True)
    add_bool_arg(common_group, 'print_tensor', default=True)
    add_bool_arg(common_group, 'classification', default=True)  # use --classification for Classification / --no-classification for Detection
    add_bool_arg(common_group,'rocal-gpu', default=True)

    common_group.add_argument('--NHWC', action='store_true',
                        help='run input pipeline NHWC format')
    common_group.add_argument('--fp16', action='store_true',
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

