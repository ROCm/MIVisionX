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

import datetime
import csv
import sys
import os
import argparse
from distutils.dir_util import copy_tree
import logging
import numpy as np
import json

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2018-2019, AMD Dataset Analysis Tool"
__credits__ = ["Mike Schmit"]
__license__ = "MIT"
__version__ = "0.9.5"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "Kiriti.NageshGowda@amd.com"
__status__ = "Alpha"


def readLabelFile(labelFile):

    with open(labelFile, 'r') as labels:
        labelLines = [l.strip().split(' ', 1)[1] for l in labels]
    return labelLines, len(labelLines)


def readHierarchyFile(hierarchyFile):
    if hierarchyFile != '':
        hierarchyElements = 0
        with open(hierarchyFile) as hierarchy:
            hierarchyCSV = csv.reader(hierarchy)
            hierarchyDataBase = [r for r in hierarchyCSV]
            hierarchyElements = len(hierarchyDataBase)
        return hierarchyDataBase, hierarchyElements


def readInferenceResultFile(inputCSVFile):
    # read results.csv
    numElements = 0
    with open(inputCSVFile) as resultFile:
        resultCSV = csv.reader(resultFile)
        next(resultCSV)  # skip header
        resultDataBase = [r for r in resultCSV]
        numElements = len(resultDataBase)
    return resultDataBase, numElements


def copyHtmlAssets(outputDirectory, fileName):
    # create toolkit with icons and images
    toolKit_Dir = os.path.join(outputDirectory, fileName + '-toolKit')
    toolKit_dir = os.path.expanduser(toolKit_Dir)
    if not os.path.exists(toolKit_dir):
        os.makedirs(toolKit_dir)
    # copy scripts and icons
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fromDirectory = os.path.join(dir_path, 'icons')
    toDirectory = os.path.join(toolKit_dir, 'icons')
    copy_tree(fromDirectory, toDirectory)

    new_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fromDirectory = os.path.join(new_path, 'utils')
    toDirectory = os.path.join(toolKit_dir, 'utils')
    copy_tree(fromDirectory, toDirectory)

    resultsDirectory = os.path.join(toolKit_dir, 'results')
    if not os.path.exists(resultsDirectory):
        os.makedirs(resultsDirectory)

    return toolKit_dir, resultsDirectory


def copyImages(imagesSource, toolkit_dir):
    dest = os.path.join(toolkit_dir, 'images')
    copy_tree(imagesSource, dest)
    return dest


def generateTop1Result(resultsDirectory, resultDataBase, labelLines):
    logger.debug("Generating top 1 result file")
    outputCsvFile = os.path.join(resultsDirectory, 'results.csv')
    outputJsonFile = os.path.join(resultsDirectory, 'results.json')
    result = {}

    headerText = [
        'Image', 'Ground Truth', 'Top 1' 'Label', 'Match', 'Top 1 Confidence', 'Ground Truth Text', 'Top 1 Label Text']
    fieldNames = ['image', 'gt', 'top1', 'match',
                  'top1_prob', 'gt_text', 'top1_text']

    with open(outputCsvFile, 'w') as csvFile:
        dictWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
        dictWriter.writer.writerow(headerText)

        for imgResult in resultDataBase:
            result = {}
            result['image'] = imgResult[0]
            result['gt'] = int(imgResult[1])
            result['top1'] = int(imgResult[2])
            result['match'] = 'no'
            if result['gt'] == result['top1']:
                result['match'] = 'yes'

            result['gt_text'] = labelLines[result['gt']]
            result['top1_text'] = labelLines[result['top1']]
            result['top1_prob'] = imgResult[7]
            dictWriter.writerow(result)

    logger.debug("Written file %s", outputCsvFile)


def generateComprehensiveResults(resultsDirectory, resultDataBase, labelLines, imageDir):

    useAbsolutePath = False
    # 0 to 4 are top1 to top5 data
    topk = 5
    logger.debug("Comprehenive csv generation")
    labelSummary = np.zeros(shape=(len(labelLines), topk+2), dtype=np.int)

    topCounts = [0] * topk
    topTotProb = [0] * topk
    totalMismatch = 0
    totalFailProb = 0
    totalNoGroundTruth = 0
    allImageResults = []

    with open(os.path.join(resultsDirectory, 'resultsComprehensive.csv'), 'w') as csvFile:
        writer = csv.writer(csvFile)

        for img in resultDataBase:
            imgName = img[0]
            gt = int(img[1])

            labels = [int(x) for x in img[2:2+topk]]
            labelTexts = [labelLines[l] for l in labels]
            probs = [float(x) for x in img[2+topk: 2+topk+topk]]
            if gt >= 0:
                gtLabelText = labelLines[gt]
                match = 0
                for i in range(topk):
                    if gt == labels[i]:
                        match = i+1
                        topCounts[i] += 1
                        topTotProb[i] += probs[i]
                        labelSummary[gt][i+1] += 1
                        labelSummary[gt][0] += 1
                        break
                if match == 0:
                    totalMismatch += 1
                    totalFailProb += probs[0]
                    labelSummary[gt][0] += 1

                if gt != labels[0]:
                    labelSummary[labels[0]][topk+1] += 1
            else:
                match = -1
                gtLabelText = 'Unknown'
                totalNoGroundTruth += 1

            writer.writerow([img[0], *labels, gt, match, *
                             labelTexts, gtLabelText, *probs])

            imgRes = {}

            imagePath = 'images/' if imageDir else 'file://'
            imgRes['imageName'] = img[0]
            imgRes['filePath'] = imagePath+img[0]
            imgRes['labels'] = labels
            imgRes['gt'] = gt
            imgRes['match'] = match
            imgRes['labelTexts'] = labelTexts
            imgRes['gtText'] = gtLabelText
            imgRes['probs'] = probs

            allImageResults.append(imgRes)

            # imgRes['filename'] = img[0]
            # if useAbsolutePath == True:
            #     imgRes'filename'

    with open(os.path.join(resultsDirectory, 'imageSummary.js'), 'w') as imageSummaryJson:
        imageSummaryJson.write('var imageSummary = ' +
                               json.dumps(allImageResults))
    logger.debug("Comprehenive csv generation completed")

    logger.debug("Label summary generation started")
    labelSummaryComplete = []

    with open(os.path.join(resultsDirectory, 'labelSummary.csv'), 'w') as labelSummaryCsv:
        writer = csv.writer(labelSummaryCsv)
        for i, label in enumerate(labelLines):
            writer.writerow([i, *labelSummary[i], label])

            if sum(labelSummary[i]) > 0:
                labelDict = {}
                labelDict['id'] = int(i)
                labelDict['label'] = label
                labelDict['totalImages'] = int(labelSummary[i][0])
                labelDict['match1'] = int(labelSummary[i][1])
                labelDict['match2'] = int(labelSummary[i][2])
                labelDict['match3'] = int(labelSummary[i][3])
                labelDict['match4'] = int(labelSummary[i][4])
                labelDict['match5'] = int(labelSummary[i][5])
                labelDict['misclassifiedTop1'] = int(labelSummary[i][6])
                labelDict['matchedTop1Per'] = float(float(
                    labelSummary[i][0]) / labelSummary[i][0]) * 100.0 if labelSummary[i][0] > 0 else 0
                labelDict['matchedTop5Per'] = float(float(
                    sum(labelSummary[i][1:6]))/labelSummary[i][0]) * 100.0 if labelSummary[i][0] > 0 else 0
                labelDict['check'] = False

                labelSummaryComplete.append(labelDict)

    with open(os.path.join(resultsDirectory, 'labelSummary.js'), 'w') as labelSummaryJson:
        labelSummaryJson.write('var labelSummary = ' +
                               json.dumps(labelSummaryComplete))

    logger.debug("Label summary generation completed")

    stats = {}
    netSummaryImages = len(resultDataBase) - totalNoGroundTruth
    passProb = sum(topTotProb)
    passCount = sum(topCounts)
    avgPassProb = float(passProb/passCount) if passCount > 0 else 0.
    accuracyPer = (float(passCount) / netSummaryImages) * 100.
    mismatchPer = (float(totalMismatch)/netSummaryImages) * 100.
    avgMismatchProb = 0.0
    if totalMismatch > 0:
        avgMismatchProb = float(totalFailProb)/totalMismatch

    stats['totalImages'] = len(resultDataBase)
    stats['totalMismatch'] = totalMismatch
    stats['passProb'] = passProb
    stats['passCount'] = passCount
    stats['avgPassProb'] = avgPassProb
    stats['accuracyPer'] = accuracyPer
    stats['mismatchPer'] = mismatchPer
    stats['avgMismatchProb'] = avgMismatchProb
    stats['totalFailProb'] = totalFailProb
    stats['netSummaryImages'] = netSummaryImages
    stats['totalNoGroundTruth'] = totalNoGroundTruth

    topKStats = []
    for i in range(topk):
        stat = {}
        stat['matches'] = topCounts[i]
        stat['accuracyPer'] = (float(topCounts[i]) / netSummaryImages) * 100.
        stat['avgPassProb'] = 0
        if topCounts[i] > 0:
            stat['avgPassProb'] = topTotProb[i] / topCounts[i]
        topKStats.append(stat)

    with open(os.path.join(resultsDirectory, 'results_1.js'), 'w') as resJson:
        currentDateString = (
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        jsonString = json.dumps({'stats': stats, 'topCounts': topCounts,
                                 'topKStats': topKStats,
                                 'summaryGenerationDate': currentDateString})
        jsScript = "var data = " + jsonString
        resJson.write(jsScript)

    return stats, topCounts, topKStats


def generateCompareResultSummary(toolKit_dir, modelName, dataFolder, stats):
    # Compare result summary
    summaryFileName = ''
    folderName = os.path.expanduser("~/.adatCompare")
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    modelFolderName = os.path.join(folderName, modelName)

    if not os.path.exists(modelFolderName):
        os.makedirs(modelFolderName)

    summaryFileName = os.path.join(folderName, 'modelRunHistoryList.json')
    stats['modelName'] = modelName
    stats['genDate'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists(summaryFileName):
        with open(summaryFileName, 'r') as f:
            data = json.load(f)
            data.append(stats)
    else:
        data = [stats]

    with open(summaryFileName, 'w') as f:
        json.dump(data, f)

    with open(os.path.join(toolKit_dir, 'results', 'resultHistory.js'), 'w') as f:
        f.write("var modelHistories="+json.dumps(data))

    # header = None
    # if os.path.exists(summaryFileName):
    #     header = ["Model Name", "Image DataBase",
    #               "Number Of Images", "Match, MisMatch"]

    # with open(summaryFileName, 'a') as outFile:
    #     csvwriter = csv.writer(outFile)
    #     if(header):
    #         csvwriter.writerow(header)
    #     csvwriter.writerow([modelName, dataFolder, stats['totalImages'],
    #                         stats['passCount'], stats['totalMismatch']])

    # with open(SummaryFileName) as savedResultFile:
    #     savedResultFileCSV = csv.reader(savedResultFile)
    #     next(savedResultFileCSV, None)  # skip header
    #     savedResultDataBase = [r for r in savedResultFileCSV]
    #     savedResultElements = len(savedResultDataBase)


def main():
    # AMD Data Analysis Toolkit - Classification
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_results',  type=str, required=True,
                        help='input inference results CSV file          [required] (File Format:ImgFileName, GroundTruth, L1, L2, L3, L4, L5, P1, P2, P3, P4, P5)')
    parser.add_argument('--image_dir',          type=str, required=False,
                        help='input image directory used in inference   [required]')
    parser.add_argument('--label',              type=str, required=True,
                        help='input labels text file                    [required]')
    parser.add_argument('--hierarchy',          type=str, default='',
                        help='input AMD proprietary hierarchical file   [optional]')
    parser.add_argument('--model_name',         type=str, default='',
                        help='input inferece model name                 [optional]')
    parser.add_argument('--output_dir',         type=str, required=True,
                        help='output dir to store ADAT results          [required]')
    parser.add_argument('--output_name',        type=str, required=True,
                        help='output ADAT file name                     [required]')
    args = parser.parse_args()

    modelName = args.model_name or 'Generic Model'
    inputCSVFile = args.inference_results
    inputImageDirectory = args.image_dir
    labelFile = args.label
    hierarchyFile = args.hierarchy
    modelName = args.model_name
    outputDirectory = args.output_dir
    fileName = args.output_name

    if inputImageDirectory:
        if not os.path.exists(args.image_dir):
            print("ERROR: Invalid Input Image Directory")
            exit()

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # read inference results file
    resultDataBase, numElements = readInferenceResultFile(inputCSVFile)
    # read label file
    labelLines, labelElements = readLabelFile(labelFile)

    if args.hierarchy:
        hierarchyDataBase, hierarchyElements = readHierarchyFile(
            hierarchyFile)
        if hierarchyElements != labelElements:
            print("ERROR Invalid Hierarchy file / label File")
            exit()

    toolkit_dir, resultsDirectory = copyHtmlAssets(outputDirectory, fileName)
    imageDir = None

    if inputImageDirectory:
        imageDir = copyImages(inputImageDirectory, toolkit_dir)

    generateTop1Result(resultsDirectory, resultDataBase, labelLines)
    stats, topCounts, topKStats = generateComprehensiveResults(
        resultsDirectory, resultDataBase, labelLines, imageDir)

    generateCompareResultSummary(toolkit_dir, modelName, 'images', stats)


if __name__ == '__main__':
    main()
