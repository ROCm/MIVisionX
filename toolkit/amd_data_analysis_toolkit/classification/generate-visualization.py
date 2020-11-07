# Copyright (c) 2018 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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

from adat_classification import *


__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018-2019, AMD Dataset Analysis Tool"
__credits__ = ["Mike Schmit"]
__license__ = "MIT"
__version__ = "0.9.5"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "Kiriti.NageshGowda@amd.com"
__status__ = "Alpha"


def main():
    # AMD Data Analysis Toolkit - Classification
    logger.debug("AMD Data Analysis Toolkit - Classification")
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_results',  type=str, required=False,
                        help='input inference results CSV file          [required] (File Format:ImgFileName, GroundTruth, L1, L2, L3, L4, L5, P1, P2, P3, P4, P5)')
    parser.add_argument('--image_dir',          type=str, required=False,
                        help='input image directory used in inference   [optional]')
    parser.add_argument('--label',              type=str, required=False,
                        help='input labels text file                    [required]')
    parser.add_argument('--hierarchy',          type=str, default='',
                        help='input AMD proprietary hierarchical file   [optional]')
    parser.add_argument('--model_name',         type=str, default='',
                        help='input inferece model name                 [optional]')
    parser.add_argument('--output_dir',         type=str, required=False,
                        help='output dir to store ADAT results          [required]')
    parser.add_argument('--output_name',        type=str, required=False,
                        help='output ADAT file name                     [required]')
    parser.add_argument('--file',               type=str, required=False,
                        help='Config File                               [optional]')
    args = parser.parse_args()
    if args.file:
        if not os.path.exists(args.file):
            logger.error("ERROR: Cannot find the config file")
            exit()
        argsDict = readConfig(args.file, args.__dict__)
    else:
        argsDict = args.__dict__
    generateAnalysisOutput(argsDict)


if __name__ == '__main__':
    main()
