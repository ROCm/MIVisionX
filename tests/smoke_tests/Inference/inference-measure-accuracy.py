import os
import getopt
import sys
import subprocess
from subprocess import call
import datetime

#Input: inference-test.py test results directory, [network]-[precision]-[date]
#		example: vgg16-fp32-2019-6-11 or vgg16-fp16-2019-6-11
#Output: Summary of argmax for each tensor.
#Process: 1. Traverse into results directory, run ./find_argmax_fp32 or ./find_argmax_fp16 on tensor with batchsize = 1
#		  2. Traverse into each directory containing collection of split tensors for batch sizes greater than 1

now = datetime.datetime.now()
today = str(now.year) + '-' + str(now.month) + '-' + str(now.day)

opts, args = getopt.getopt(sys.argv[1:], 'd:p:')

FP16 = 0
runMode = 0
pythonMode = 0

for opt, arg in opts:
    if opt =='-d':
    	resultsDir = arg
    elif opt == '-p':
    	pythonMode = int(arg)

#Parse results directory name to figure out the network and precision
i = 2
resultsDirList = resultsDir.split('-')
network = resultsDirList[0]
precision = resultsDirList[1]
outFileName = network + '-' + precision
if resultsDirList[2] == 'python':
	pythonMode = 1
	i = i + 1
	outFileName = outFileName + '-python'
elif resultsDirList[2] == 'fuseOn':
	runMode = 1
	i = i + 1
	outFileName = outFileName + '-fuseOn'
	#originalDate = resultsDirList[4] + '-' + resultsDirList[5] + '-' + resultsDirList[6]
	#outFileName = network + '-fuseOn'
if resultsDirList[3] == 'fuseOn':
	runMode = 1
	i = i + 1

originalDate = resultsDirList[i] + '-' + resultsDirList[i+1] + '-' + resultsDirList[i+2]
outFileName = outFileName + '-' + originalDate + '-accuracy.csv'


if precision == 'fp16':
	FP16 = 1

fp_result = open(outFileName, "w+")

#Get correct argmax utility

if FP16 == 0:
	argmaxExe = './find_argmax_of_tensor_FP32'
	tensorBatch1 = network + '-1'
	if pythonMode == 1:
		tensorBatch1 = tensorBatch1 + '-python'
	tensorBatch1 = tensorBatch1 + '.fp'
elif FP16 == 1:
	argmaxExe = './find_argmax_of_tensor_FP16'  
	tensorBatch1 = network + '-1-fp16'
	if pythonMode == 1:
		tensorBatch1 = tensorBatch1 + '-python'
	tensorBatch1 = tensorBatch1 + '.fp'

if FP16 == 0:
	out_to_print = network + ',fp32,1'
else:
	out_to_print = network + ',fp16,1'
fp_result.write(out_to_print)
fp_result.write('\n')

cmd = argmaxExe + ' ' + resultsDir + '/' + tensorBatch1
out = subprocess.check_output(cmd, shell=True)
out_to_print = "0," + out
fp_result.write(out_to_print)

#Batch sizes 2 - 128 need further directory traversing and loops for each separate tensor
batchSizeList = [64]
for batchSize in batchSizeList:
	if FP16 == 0:
		tensorsDir = network + '-' + str(batchSize)
	elif FP16 == 1:
		tensorsDir = network + '-' + str(batchSize) + '-fp16' 
	currentDir = resultsDir + '/' + tensorsDir
	i = 0
	currentTest = network + ',' + precision + ',' + str(batchSize)
	fp_result.write(currentTest)
	fp_result.write('\n')
	while i < batchSize:
		if FP16 == 1:
			if pythonMode == 0:
				currentTensor = network + '-' + str(batchSize) + '-fp16-' + str(i) + '.fp'
			elif pythonMode == 1:
				currentTensor = network + '-' + str(batchSize) + '-python-fp16-' + str(i) + '.fp'
		else:
			if pythonMode == 0:
				currentTensor = network + '-' + str(batchSize) + '-' + str(i) + '.fp'
			elif pythonMode == 1:
				currentTensor = network + '-' + str(batchSize) + '-python-' + str(i) + '.fp'
		
		cmd = argmaxExe + ' ' + currentDir + '/' + currentTensor
		out = subprocess.check_output(cmd, shell=True)
		out_to_print = str(i) + ',' + out
		fp_result.write(out_to_print)
		i = i + 1

fp_result.close()

cmd = 'mv ' + outFileName + ' ' + resultsDir
if os.system(cmd) != 0:
	print 'ERROR: Could not move ' + outFileName + ' into ' + resultsDir + '\n'
