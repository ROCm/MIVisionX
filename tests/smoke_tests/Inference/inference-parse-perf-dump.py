import os
import getopt
import sys
import subprocess
from subprocess import call
import re

if(len(sys.argv) < 2):
	print("Usage: inference-parse-perf-dump.py [dump file]\n")
	exit()

dumpFile = sys.argv[1]
dumpFileNameList = dumpFile.split("-")
precision = dumpFileNameList[1]
print(dumpFile)
fp = open(dumpFile, 'r')
line = fp.readline()

foundNetworkName = 0
foundTensorSize = 0
tensorSize = 0
prevNetworkName = ""

lineCount = 0
while line:
	if lineCount == 0:
		today = line
	if (re.search(r"reading IR model from ", line)) and (foundNetworkName == 0):
		lineList = line.split(" ")
		if lineList[4] != prevNetworkName:
			print(lineList[4] + "," + precision + "," + today)
		prevNetworkName = lineList[4]
		foundNetworkName = 1

	if (re.search(r"OK: initialized tensor 'data' from ", line)) and (foundNetworkName == 1) and (foundTensorSize == 0):
		lineList = line.split(" ")
		tensorPathOne = lineList[5].split("/")
		tensorPathTwo = tensorPathOne[2].split("-")
		tensorSize = tensorPathTwo[1]
		foundTensorSize = 1
	if (re.search(r"(average over 100 iterations)", line)) and (foundNetworkName == 1) and (foundTensorSize == 1):
		lineList = line.split(" ")
		totalTime = lineList[3]
		timePerImage = float(totalTime) / float(tensorSize)
		print(str(tensorSize) + "," + str(timePerImage))
		foundNetworkName = 0
		foundTensorSize = 0
	line = fp.readline()
	lineCount = lineCount + 1
