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
    common_group.add_argument('--display', action="store_true",
                        help='--display:to display output from the pipeline')
    common_group.add_argument('--no-display', dest='display', action="store_false",
                        help='--no-display:to not display output from the pipeline')
    parser.set_defaults(display=True)   #case when none of the above is specified

    common_group.add_argument('--print_tensor', action="store_true",
                        help='--print_tensor: to print tensor output from the pipeline')
    common_group.add_argument('--no-print_tensor', dest='print_tensor', action="store_false",
                        help='--no-print_tensor: to not print tensor output from the pipeline')
    parser.set_defaults(print_tensor=False) #case when none of the above is specified

    common_group.add_argument('--classification', action="store_true",
                        help='--classification: to use for classification')
    common_group.add_argument('--no-classification', dest='classification', action="store_false",
                        help='--no-classification: to use for detection pipeline')
    parser.set_defaults(classification=True) #case when none of the above is specified

    common_group.add_argument('--rocal-gpu', default=False, action="store_true",
                        help='--use_gpu to use gpu')
    common_group.add_argument('--no-rocal-gpu', dest='rocal-gpu', action="store_false",
                        help='--no-rocal-gpu to use cpu backend')

    common_group.add_argument('--NHWC', action='store_true',
                        help='run input pipeline NHWC format')
    common_group.add_argument('--no-NHWC', dest='NHWC', action='store_false',
                        help='run input pipeline NCHW format')
    parser.set_defaults(NHWC=True) #case when none of the above is specified

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

