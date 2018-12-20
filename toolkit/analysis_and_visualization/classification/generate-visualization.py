__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2018, AMD Dataset Analysis Tool"
__credits__     = ["Mike Schmit"]
__license__     = "MIT"
__version__     = "0.9.0"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "Kiriti.NageshGowda@amd.com"
__status__      = "Alpha"

import os
import getopt
import sys
import random
import collections
import csv
import numpy
import datetime

opts, args = getopt.getopt(sys.argv[1:], 'i:d:l:h:o:f:m:')

inputCSVFile = '';
inputImageDirectory = '';
labelFile = '';
hierarchyFile = '';
outputDirectory = '';
fileName = '';
modelName = '';

for opt, arg in opts:
    if opt == '-i':
        inputCSVFile = arg;
    elif opt == '-d':
        inputImageDirectory = arg;
    elif opt == '-l':
        labelFile = arg;
    elif opt == '-h':
        hierarchyFile = arg;
    elif opt == '-o':
        outputDirectory = arg;
    elif opt == '-f':
        fileName = arg;
    elif opt == '-m':
        modelName = arg;

# report error
if inputCSVFile == '' or inputImageDirectory == '' or labelFile == '' or outputDirectory == '' or fileName == '':
    print('Invalid command line arguments.\n'
        '\t\t\t\t-i [input Result CSV File - required](File Format:ImgFileName, GroundTruth, L1, L2, L3, L4, L5, P1, P2, P3, P4, P5)[L:Label P:Probability]\n'\
        '\t\t\t\t-d [input Image Directory - required]\n'\
        '\t\t\t\t-l [input Label File      - required]\n'\
        '\t\t\t\t-h [input Hierarchy File  - optional]\n'\
        '\t\t\t\t-m [input NN model name   - optional]\n'\
        '\t\t\t\t-o [output Directory - required]\n'\
        '\t\t\t\t-f [output file name - required]\n')

    exit();

if not os.path.exists(inputImageDirectory):
    print "ERROR Invalid Input Image Directory";
    exit();

if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory);

if modelName == '':
    modelName = 'Generic Model';

# read results.csv
numElements = 0;
with open(inputCSVFile) as resultFile:
    resultCSV = csv.reader(resultFile)
    next(resultCSV) # skip header
    resultDataBase = [r for r in resultCSV]
    numElements = len(resultDataBase)

# read labels.txt
labelElements = 0
with open(labelFile) as labels:
    LabelLines = labels.readlines()
    labelElements = len(LabelLines)

# read hieararchy.csv
hierarchySection = 0
if hierarchyFile != '':
    hierarchySection = 1
    hierarchyElements = 0
    with open(hierarchyFile) as hierarchy:
        hierarchyCSV = csv.reader(hierarchy)
        hierarchyDataBase = [r for r in hierarchyCSV]
        hierarchyElements = len(hierarchyDataBase)

    if hierarchyElements != labelElements:
        print "ERROR Invalid Hierarchy file / label File";
        exit();

# create toolkit with icons and images
toolKit_Dir = outputDirectory +'/'+ fileName + '-toolKit'
toolKit_dir = os.path.expanduser(toolKit_Dir)
if not os.path.exists(toolKit_dir):
    os.makedirs(toolKit_dir);
# copy images and icons
from distutils.dir_util import copy_tree
fromDirectory = inputImageDirectory;
toDirectory = toolKit_dir+'/images';
copy_tree(fromDirectory, toDirectory)

dir_path = os.path.dirname(os.path.realpath(__file__))
fromDirectory = dir_path+'/icons';
toDirectory = toolKit_dir+'/icons';
copy_tree(fromDirectory, toDirectory)

fromDirectory = dir_path+'/utils';
toDirectory = toolKit_dir+'/utils';
copy_tree(fromDirectory, toDirectory)

dataFolder = 'images';
resultsDirectory = toolKit_dir+'/results';
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory);

# generate detailed results.csv
print "results.csv generation .."
orig_stdout = sys.stdout
sys.stdout = open(resultsDirectory+'/results.csv','w')
print 'Image,Ground Truth, Top 1 Label, Match, Top 1 Confidence, Ground Truth Text, Top 1 Label Text'
for x in range(numElements):
    gt = int(resultDataBase[x][1]);
    lt = int(resultDataBase[x][2]);
    matched = 'no';
    if gt == lt:
         matched = 'yes';

    print ''+(resultDataBase[x][0])+','+(resultDataBase[x][1])+','+(resultDataBase[x][2])+','+(matched)+','\
            +(resultDataBase[x][7])+',"'+(LabelLines[int(resultDataBase[x][1])].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[int(resultDataBase[x][2].rstrip('\n'))].split(' ', 1)[1].rstrip('\n'))+'"'

sys.stdout = orig_stdout
print "results.csv generated"

# generate results summary.csv
top1TotProb = top2TotProb = top3TotProb = top4TotProb = top5TotProb = totalFailProb = 0;
top1Count = top2Count = top3Count = top4Count = top5Count = 0;
totalNoGroundTruth = totalMismatch = 0;


topKPassFail = []
topKHierarchyPassFail = []
for i in xrange(100):
    topKPassFail.append([])
    topKHierarchyPassFail.append([])
    for j in xrange(2):
        topKPassFail[i].append(0)
    for k in xrange(12):
        topKHierarchyPassFail[i].append(0)


topLabelMatch = []
for i in xrange(1000):
    topLabelMatch.append([])
    for j in xrange(7):
        topLabelMatch[i].append(0)

# Generate Comphrehensive Results
print "resultsComphrehensive.csv generation .."
orig_stdout = sys.stdout
sys.stdout = open(resultsDirectory+'/resultsComphrehensive.csv','w')
print(  'FileName,outputLabel-1,outputLabel-2,outputLabel-3,outputLabel-4,outputLabel-5,'\
        'groundTruthLabel,Matched,outputLabelText-1,outputLabelText-2,outputLabelText-3,outputLabelText-4,outputLabelText-5,'\
        'groundTruthLabelText,Prob-1,Prob-2,Prob-3,Prob-4,Prob-5' );
imageDataSize = numElements;
for x in range(numElements):
    truth = int(resultDataBase[x][1]);
    if truth >= 0:
        match = 0;
        label_1 = int(resultDataBase[x][2]);
        label_2 = int(resultDataBase[x][3]);
        label_3 = int(resultDataBase[x][4]);
        label_4 = int(resultDataBase[x][5]);
        label_5 = int(resultDataBase[x][6]);
        prob_1 = float(resultDataBase[x][7]);
        prob_2 = float(resultDataBase[x][8]);
        prob_3 = float(resultDataBase[x][9]);
        prob_4 = float(resultDataBase[x][10]);
        prob_5 = float(resultDataBase[x][11]);

        if(truth == label_1):
            match = 1; 
            top1Count+= 1;
            top1TotProb += prob_1;
            topLabelMatch[truth][0]+= 1;
            topLabelMatch[truth][1]+= 1;

        elif(truth == label_2):
            match = 2; 
            top2Count+= 1;
            top2TotProb += prob_2;
            topLabelMatch[truth][0]+= 1;
            topLabelMatch[truth][2]+= 1;

        elif(truth == label_3):
            match = 3; 
            top3Count+= 1; 
            top3TotProb += prob_3;
            topLabelMatch[truth][0]+= 1;
            topLabelMatch[truth][3]+= 1;

        elif(truth == label_4):
            match = 4;
            top4Count+= 1;
            top4TotProb += prob_4;
            topLabelMatch[truth][0]+= 1;
            topLabelMatch[truth][4]+= 1;

        elif(truth == label_5):
            match = 5; 
            top5Count+= 1;
            top5TotProb += prob_5;
            topLabelMatch[truth][0]+= 1;
            topLabelMatch[truth][5]+= 1;

        else:
            totalMismatch+= 1;
            totalFailProb += prob_1;
            topLabelMatch[truth][0]+= 1;


        if(truth != label_1):
            topLabelMatch[label_1][6]+= 1;

        print(resultDataBase[x][0]+','+str(label_1)+','+str(label_2)+','+str(label_3)+','+str(label_4)+','+str(label_5)+','+str(truth)+','+str(match)+',"'\
            +(LabelLines[label_1].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[label_2].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[label_3].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[label_4].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[label_5].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[truth].split(' ', 1)[1].rstrip('\n'))+'",'\
            +str(prob_1)+','+str(prob_2)+','+str(prob_3)+','+str(prob_4)+','+str(prob_5));

        # Calculate hierarchy graph
        if hierarchyFile != '':
            if(truth == label_1): 
                count = 0;
                f = 0
                while f < 1:
                    if((prob_1 < (f + 0.01)) and prob_1 > f):
                        topKPassFail[count][0]+= 1;

                        topKHierarchyPassFail[count][0]+= 1;
                        topKHierarchyPassFail[count][2]+= 1;
                        topKHierarchyPassFail[count][4]+= 1;
                        topKHierarchyPassFail[count][6]+= 1;
                        topKHierarchyPassFail[count][8]+= 1;
                        topKHierarchyPassFail[count][10]+= 1;

                    count+= 1;
                    f+= 0.01;
            else:
                count = 0;
                f = 0
                while f < 1:
                    if((prob_1 < (f + 0.01)) and prob_1 > f):
                        topKPassFail[count][1]+= 1;

                        truthHierarchy = hierarchyDataBase[truth];
                        resultHierarchy = hierarchyDataBase[label_1];
                        token_result = '';
                        token_truth = '';
                        previousTruth = 0;
                        catCount = 0;
                        while catCount < 6:
                            token_truth = truthHierarchy[catCount];
                            token_result = resultHierarchy[catCount];
                            if((token_truth != '') and (token_truth == token_result)):
                                topKHierarchyPassFail[count][catCount*2]+= 1;
                                previousTruth = 1;
                            elif( (previousTruth == 1) and (token_truth == '' and token_result == '')):
                                topKHierarchyPassFail[count][catCount*2]+= 1;
                            else:
                                topKHierarchyPassFail[count][catCount*2 + 1]+= 1;
                                previousTruth = 0;
                            catCount+= 1;
                    count+= 1;
                    f+= 0.01;

    else: 
        match = -1;
        label_1 = int(resultDataBase[x][2]);
        label_2 = int(resultDataBase[x][3]);
        label_3 = int(resultDataBase[x][4]);
        label_4 = int(resultDataBase[x][5]);
        label_5 = int(resultDataBase[x][6]);
        prob_1 = float(resultDataBase[x][7]);
        prob_2 = float(resultDataBase[x][8]);
        prob_3 = float(resultDataBase[x][9]);
        prob_4 = float(resultDataBase[x][10]);
        prob_5 = float(resultDataBase[x][11]);
        print(resultDataBase[x][0]+','+str(label_1)+','+str(label_2)+','+str(label_3)+','+str(label_4)+','+str(label_5)+','+str(truth)+','+str(match)+',"'\
            +(LabelLines[label_1].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[label_2].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[label_3].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[label_4].split(' ', 1)[1].rstrip('\n'))+'","'\
            +(LabelLines[label_5].split(' ', 1)[1].rstrip('\n'))+'",Unknown,'\
            +str(prob_1)+','+str(prob_2)+','+str(prob_3)+','+str(prob_4)+','+str(prob_5));
                    
        totalNoGroundTruth+= 1;

sys.stdout = orig_stdout
print "resultsComphrehensive.csv generated"

# Generate Comphrehensive Results
print "resultsSummary.txt generation .."
orig_stdout = sys.stdout
sys.stdout = open(resultsDirectory+'/resultsSummary.txt','w')
print("\n\n ***************** INFERENCE SUMMARY ***************** \n");
import numpy as np
netSummaryImages =  imageDataSize - totalNoGroundTruth;
passProb = top1TotProb+top2TotProb+top3TotProb+top4TotProb+top5TotProb;
passCount = top1Count+top2Count+top3Count+top4Count+top5Count;
avgPassProb = float(passProb/passCount);

print('Images with Ground Truth -- '+str(netSummaryImages));
print('Images without Ground Truth -- '+str(totalNoGroundTruth));
print('Total image set for Inference -- '+str(imageDataSize));
print("");
print('Total Top K match -- '+str(passCount));
accuracyPer = float(passCount);
accuracyPer = (accuracyPer / netSummaryImages) * 100;
print('Inference Accuracy on Top K -- '+str(np.around(accuracyPer,decimals=2))+' %');
print('Average Pass Probability for Top K -- '+str(np.around(avgPassProb,decimals=4)));
print("");    
print('Total mismatch -- '+str(totalMismatch));
accuracyPer = float(totalMismatch);
accuracyPer = (accuracyPer/netSummaryImages) * 100;
print('Inference mismatch Percentage -- '+str(np.around(accuracyPer,decimals=2))+' %');
print('Average mismatch Probability for Top 1 -- '+str(np.around(totalFailProb/totalMismatch,decimals=4)));

print("\n*****Top1*****");
print('Top1 matches -- '+str(top1Count));  
accuracyPer = float(top1Count);
accuracyPer = (accuracyPer/netSummaryImages) * 100;
print('Top1 match Percentage -- '+str(np.around(accuracyPer,decimals=2))+' %');          
if top1Count > 0:
    print('Avg Top1 pass prob -- '+str(np.around(top1TotProb/top1Count,decimals=4)));
else:
    print('Avg Top1 pass prob -- 0 %');
         
print("\n*****Top2*****");   
print('Top2 matches -- '+str(top2Count));
accuracyPer = float(top2Count);
accuracyPer = (accuracyPer/netSummaryImages) * 100;
print('Top2 match Percentage -- '+str(np.around(accuracyPer,decimals=2))+' %');
if top2Count > 0:
    print('Avg Top2 pass prob -- '+str(np.around(top2TotProb/top2Count,decimals=4)));
else:
    print('Avg Top2 pass prob -- 0 %');

print("\n*****Top3*****");
print('Top3 matches -- '+str(top3Count));
accuracyPer = float(top3Count);
accuracyPer = (accuracyPer/netSummaryImages) * 100;
print('Top3 match Percentage -- '+str(np.around(accuracyPer,decimals=2))+' %');
if top3Count > 0:
    print('Avg Top3 pass prob -- '+str(np.around(top3TotProb/top3Count,decimals=4)));
else:
    print('Avg Top3 pass prob -- 0 %');

print("\n*****Top4*****");
print('Top4 matches -- '+str(top4Count));
accuracyPer = float(top4Count);
accuracyPer = (accuracyPer/netSummaryImages) * 100;
print('Top4 match Percentage -- '+str(np.around(accuracyPer,decimals=2))+' %');
if top4Count > 0:
    print('Avg Top4 pass prob -- '+str(np.around(top4TotProb/top4Count,decimals=4)));
else:
    print('Avg Top4 pass prob -- 0 %');

print("\n*****Top5*****");
print('Top5 matches -- '+str(top5Count));
accuracyPer = float(top5Count);
accuracyPer = (accuracyPer/netSummaryImages) * 100;
print('Top5 match Percentage -- '+str(np.around(accuracyPer,decimals=2))+' %');
if top5Count > 0:
    print('Avg Top5 pass prob -- '+str(np.around(top5TotProb/top5Count,decimals=4)));
else:
    print('Avg Top5 pass prob -- 0 %');
print("\n");
sys.stdout = orig_stdout
print "resultsSummary.txt generated"

# Hierarchy Summary
print "hierarchySummary.csv generation .."
orig_stdout = sys.stdout
sys.stdout = open(resultsDirectory+'/hierarchySummary.csv','w')
print("Probability,Pass,Fail,cat-1 pass,cat-1 fail,cat-2 pass, cat-2 fail,"
      "cat-3 pass,cat-3 fail,cat-4 pass,cat-4 fail,cat-5 pass,cat-5 fail,cat-6 pass,cat-6 fail");
i = 99;
f=0.99;
while i >= 0:
    print("%.2f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" %(f,topKPassFail[i][0],topKPassFail[i][1],
        topKHierarchyPassFail[i][0],topKHierarchyPassFail[i][1],topKHierarchyPassFail[i][2],
        topKHierarchyPassFail[i][3],topKHierarchyPassFail[i][4],topKHierarchyPassFail[i][5],
        topKHierarchyPassFail[i][6],topKHierarchyPassFail[i][7],topKHierarchyPassFail[i][8],
        topKHierarchyPassFail[i][9],topKHierarchyPassFail[i][10],topKHierarchyPassFail[i][11]));
    f = f - 0.01;
    i = i - 1;

sys.stdout = orig_stdout;
print "hierarchySummary.csv generated .."


# Label Summary
print "labelSummary.csv generation .."
orig_stdout = sys.stdout
sys.stdout = open(resultsDirectory+'/labelSummary.csv','w')
print("Label,Images in DataBase, Matched with Top1, Matched with Top2, Matched with Top3, Matched with Top4, Matched with Top5,Top1 Label Match, Label Description");
for i in xrange(1000):
    print("%d,%d,%d,%d,%d,%d,%d,%d,\"%s\""%(i,topLabelMatch[i][0],topLabelMatch[i][1],topLabelMatch[i][2],
    topLabelMatch[i][3],topLabelMatch[i][4],topLabelMatch[i][5],
    topLabelMatch[i][6],(LabelLines[i].split(' ', 1)[1].rstrip('\n'))));
sys.stdout = orig_stdout
print "labelSummary.csv generated"

# generate detailed results.csv
print "index.html generation .."
#orig_stdout = sys.stdout
sys.stdout = open(toolKit_dir+'/index.html','w')

print ("<!DOCTYPE HTML PUBLIC \" -//W3C//DTD HTML 4.0 Transitional//EN\">");
print ("\n<html>");
print ("<head>");
print ("\n\t<meta http-equiv=\"content-type\" content=\"text/html; charset=utf-8\"/>");
print ("\t<title>AMD Dataset Analysis Tool</title>");
print ("\t<link rel=\"icon\" href=\"icons/vega_icon_150.png\"/>");

# page style
print ("\n\t<style type=\"text/css\">");
print ("\t");
print ("\tbody,div,table,thead,tbody,tfoot,tr,th,td,p { font-family:\"Liberation Sans\"; font-size:x-small }");
print ("\ta.comment-indicator:hover + comment { background:#ffd; position:absolute; display:block; border:1px solid black; padding:0.5em;  }");
print ("\ta.comment-indicator { background:red; display:inline-block; border:1px solid black; width:0.5em; height:0.5em;  }");
print ("\tcomment { display:none;  } tr:nth-of-type(odd) { background-color:#f2f2f2;}");
print ("\t");
print ("\t#myImg { border-radius: 5px; cursor: pointer; transition: 0.3s; }");
print ("\t#myImg:hover { opacity: 0.7; }");
print ("\t.modal{ display: none; position: fixed; z-index: 8; padding-top: 100px; left: 0; top: 0;width: 100%;");
print ("\t       height: 100%; overflow: auto; background-color: rgb(0,0,0); background-color: rgba(0,0,0,0.9); }");
print ("\t.modal-content { margin: auto; display: block; width: 80%; max-width: 500px; }");
print ("\t#caption { margin: auto; display: block; width: 80%; max-width: 700px; text-align: center; color: white;font-size: 18px; padding: 10px 0; height: 150px;}");
print ("\t.modal-content, #caption {  -webkit-animation-name: zoom;  -webkit-animation-duration: 0.6s;");
print ("\t                           animation-name: zoom; animation-duration: 0.6s; }");
print ("\t@-webkit-keyframes zoom {  from { -webkit-transform:scale(0) }  to { -webkit-transform:scale(1) }}");
print ("\t@keyframes zoom {    from {transform:scale(0)}     to {transform:scale(1) }}");
print ("\t.close { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; transition: 0.3s; }");
print ("\t.close:hover,.close:focus { color: #bbb; text-decoration: none; cursor: pointer; }");
print ("\t@media only screen and (max-width: 400px){ .modal-content {     width: 100%; } }");
print ("\t");
print ("\tbody { font-family: \"Lato\", sans-serif;}");
print ("\t.sidenav { height: 100%; width: 0; position: fixed; z-index: 7; top: 0; left: 0; background-color: #111;");
print ("\t\t overflow-x: hidden;    transition: 0.5s; padding-top: 60px;}");
print ("\t.sidenav a { padding: 8px 8px 8px 32px; text-decoration: none; font-size: 25px; color: #818181; display: block; transition: 0.3s;}");
print ("\t.sidenav a:hover { color: #f1f1f1;}");
print ("\t.sidenav .closebtn {  position: absolute; top: 0; right: 25px; font-size: 36px; margin-left: 50px;}");
print ("\t#main {  transition: margin-left .5s;  padding: 16px; }");
print ("\t@media screen and (max-height: 450px) { .sidenav {padding-top: 15px;} .sidenav a {font-size: 18px;} }");
print ("\t");
print ("\tbody {margin:0;}");
print ("\t.navbar {  overflow: hidden;  background-color: #333;  position: fixed; z-index: 6;  top: 0;  width: 100%;}");
print ("\t.navbar a {  float: left;  display: block;  color: #f2f2f2;  text-align: center;  padding: 14px 16px;  text-decoration: none;  font-size: 17px; }");
print ("\t.navbar a:hover {  background: #ddd;  color: black;}");
print ("\t.main {  padding: 16px;  margin-top: 30px; }");
print ("\t");
print ("\tselect {-webkit-appearance: none; -moz-appearance: none; text-indent: 0px; text-overflow: ''; color:maroon; }");
print ("\t");
print ("\t.tooltip { position: relative; display: inline-block;}");
print ("\t.tooltip .tooltiptext { visibility: hidden; width: 150px; background-color: black; color: gold;");
print ("\t\ttext-align: center;  border-radius: 6px;  padding: 5px; position: absolute; z-index: 3;}");
print ("\t.tooltip:hover .tooltiptext { visibility: visible;}");
print ("\t");
print ("\t.footer { position: fixed; left: 0;    bottom: 0;  width: 100%;    background-color: #333;  color: white;  text-align: center;}");
print ("\t");
print ("\t</style>");
print ("\n</head>");
print ("\n\n<body>");
print ("\t");
print ("\t<div id=\"myModal\" class=\"modal\"> <span class=\"close\">&times;</span>  <img class=\"modal-content\" id=\"img01\">  <div id=\"caption\"></div> </div>");
print ("\t");

# table content order
print ("\t<div id=\"mySidenav\" class=\"sidenav\">");
print ("\t<a href=\"javascript:void(0)\" class=\"closebtn\" onclick=\"closeNav()\">&times;</a>");
print ("\t<A HREF=\"#table0\"><font size=\"5\">Summary</font></A><br>");
print ("\t<A HREF=\"#table1\"><font size=\"5\">Graphs</font></A><br>");
print ("\t<A HREF=\"#table2\"><font size=\"5\">Hierarchy</font></A><br>");
print ("\t<A HREF=\"#table3\"><font size=\"5\">Labels</font></A><br>");
print ("\t<A HREF=\"#table4\"><font size=\"5\">Image Results</font></A><br>");
print ("\t<A HREF=\"#table5\"><font size=\"5\">Compare</font></A><br>");
print ("\t<A HREF=\"#table6\"><font size=\"5\">Model Score</font></A><br>");
print ("\t<A HREF=\"#table7\"><font size=\"5\">Help</font></A><br>");
print ("\t</div>");
print ("\t");

# scripts
print ("\t<script>");
print ("\t\tfunction openNav() {");
print ("\t\t\tdocument.getElementById(\"mySidenav\").style.width = \"250px\";");
print ("\t\t\tdocument.getElementById(\"main\").style.marginLeft = \"250px\";}");
print ("\t\tfunction closeNav() {");
print ("\t\t\tdocument.getElementById(\"mySidenav\").style.width = \"0\";");
print ("\t\t\tdocument.getElementById(\"main\").style.marginLeft= \"0\";}");
print ("\t\tfunction myreload() { location.reload();}");
print ("\t");
print ("\t\tfunction sortTable(coloum,descending) {");
print ("\t\tvar table, rows, switching, i, x, y, shouldSwitch;");
print ("\t\ttable = document.getElementById(id=\"resultsTable\"); switching = true;");
print ("\t\twhile (switching) {  switching = false; rows = table.getElementsByTagName(\"TR\");");
print ("\t\t\tfor (i = 1; i < (rows.length - 1); i+= 1) { shouldSwitch = false;");
print ("\t\t\t\tx = rows[i].getElementsByTagName(\"TD\")[coloum];");
print ("\t\t\t\ty = rows[i + 1].getElementsByTagName(\"TD\")[coloum];");
print ("\t\t\t\tif(descending){if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {");
print ("\t\t\t\t\tshouldSwitch= true;    break;}}");
print ("\t\t\t\telse{if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {");
print ("\t\t\t\t\tshouldSwitch= true;    break;}}}");
print ("\t\t\t\tif (shouldSwitch) {  rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);");
print ("\t\t\t\t\tswitching = true;}}}");
print ("\t");
print ("\t");
print ("\t\tfunction sortLabelsTable(coloum,descending) {");
print ("\t\tvar table, rows, switching, i, x, y, shouldSwitch;");
print ("\t\ttable = document.getElementById(id=\"labelsTable\"); switching = true;");
print ("\t\twhile (switching) {  switching = false; rows = table.getElementsByTagName(\"TR\");");
print ("\t\t\tfor (i = 1; i < (rows.length - 1); i+= 1) { shouldSwitch = false;");
print ("\t\t\t\tx = rows[i].getElementsByTagName(\"TD\")[coloum];");
print ("\t\t\t\ty = rows[i + 1].getElementsByTagName(\"TD\")[coloum];");
print ("\t\t\t\tif(descending){if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {");
print ("\t\t\t\t\tshouldSwitch= true;    break;}}");
print ("\t\t\t\telse{if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {");
print ("\t\t\t\t\tshouldSwitch= true;    break;}}}");
print ("\t\t\t\tif (shouldSwitch) {  rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);");
print ("\t\t\t\t\tswitching = true;}}}");
print ("\t");
print ("\t</script>");
print ("\t<script src=\"utils/sorttable.js\"></script>");
print ("\t");
print ("\t<script>");
print ("\t\tfunction filterResultTable(rowNum, DataVar) {");
print ("\t\tvar input, filter, table, tr, td, i;");
print ("\t\tinput = document.getElementById(DataVar);");
print ("\t\tfilter = input.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttd = tr[i].getElementsByTagName(\"td\")[rowNum];");
print ("\t\tif (td) { if (td.innerHTML.toUpperCase().indexOf(filter) > -1) {tr[i].style.display = \"\"; }");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("\t</script>");
print ("\t");
print ("\t");
print ("\t<script>");
print ("\t\tfunction filterLabelTable(rowNum, DataVar) {");
print ("\t\tvar input, filter, table, tr, td, i;");
print ("\t\tinput = document.getElementById(DataVar);");
print ("\t\tfilter = input.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"labelsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttd = tr[i].getElementsByTagName(\"td\")[rowNum];");
print ("\t\tif (td) { if (td.innerHTML.toUpperCase().indexOf(filter) > -1) {tr[i].style.display = \"\"; }");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("\t</script>");
print ("\t");
print ("");
print ("\t<script>");
print ("\t\tfunction clearLabelFilter() {");
print ("\t\tdocument.getElementById('Label ID').value = ''");
print ("\t\tdocument.getElementById('Label Description').value = ''");
print ("\t\tdocument.getElementById('Images in DataBase').value = ''");
print ("\t\tdocument.getElementById('Matched Top1 %').value = ''");
print ("\t\tdocument.getElementById('Matched Top5 %').value = ''");
print ("\t\tdocument.getElementById('Matched 1st').value = ''");
print ("\t\tdocument.getElementById('Matched 2nd').value = ''");
print ("\t\tdocument.getElementById('Matched 3th').value = ''");
print ("\t\tdocument.getElementById('Matched 4th').value = ''");
print ("\t\tdocument.getElementById('Matched 5th').value = ''");
print ("\t\tdocument.getElementById('Misclassified Top1 Label').value = ''");
print ("\t\tfilterLabelTable(0,'Label ID') }");
print ("\t</script>");
print ("");
print ("");
print ("\t<script>");
print ("\t\tfunction clearResultFilter() {");
print ("\t\tdocument.getElementById('GroundTruthText').value = ''");
print ("\t\tdocument.getElementById('GroundTruthID').value = ''");
print ("\t\tdocument.getElementById('Matched').value = ''");
print ("\t\tdocument.getElementById('Top1').value = ''");
print ("\t\tdocument.getElementById('Top1Prob').value = ''");
print ("\t\tdocument.getElementById('Text1').value = ''");
print ("\t\tdocument.getElementById('Top2').value = ''");
print ("\t\tdocument.getElementById('Top2Prob').value = ''");
print ("\t\tdocument.getElementById('Top3').value = ''");
print ("\t\tdocument.getElementById('Top3Prob').value = ''");
print ("\t\tdocument.getElementById('Top4').value = ''");
print ("\t\tdocument.getElementById('Top4Prob').value = ''");
print ("\t\tdocument.getElementById('Top5').value = ''");
print ("\t\tdocument.getElementById('Top5Prob').value = ''");
print ("\t\tfilterResultTable(2,'GroundTruthText') }");
print ("\t</script>");
print ("");
print ("\t<script>");
print ("\t\tfunction findGroundTruthLabel(label,labelID) {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('GroundTruthText').value = label;");
print ("\t\tdocument.getElementById('GroundTruthID').value = labelID;");
print ("\t\tandResultFilter();");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction findMisclassifiedGroundTruthLabel(label) {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('Text1').value = label;");
print ("\t\tfilterResultTable(10,'Text1');");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction highlightRow(obj){");
print ("\t\tvar tr=obj; while (tr.tagName.toUpperCase()!='TR'&&tr.parentNode){  tr=tr.parentNode;}");
print ("\t\tif (!tr.col){tr.col=tr.bgColor; } if (obj.checked){  tr.bgColor='#d5f5e3';}");
print ("\t\telse {  tr.bgColor=tr.col;}}");
print ("");
print ("\t\tfunction goToImageResults() { window.location.href = '#table4';}");
print ("");
print ("\t\tfunction findImagesWithNoGroundTruthLabel() {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('GroundTruthID').value = '-1';");
print ("\t\tfilterResultTable(3,'GroundTruthID');");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction findImageMisMatch() {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('Matched').value = '0';");
print ("\t\tfilterResultTable(9,'Matched');");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction findTopKMatch() {");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('Matched').value = '0';");
print ("\t\tnotResultFilter();");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction filterResultTableInverse(rowNum, DataVar) {");
print ("\t\tvar input, filter, table, tr, td, i;");
print ("\t\tinput = document.getElementById(DataVar);");
print ("\t\tfilter = input.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttd = tr[i].getElementsByTagName(\"td\")[rowNum];");
print ("\t\tif (td) { if (td.innerHTML.toUpperCase().indexOf(filter) <= -1) {tr[i].style.display = \"\"; }");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("\t\tfunction findImagesWithGroundTruthLabel(){");
print ("\t\tclearResultFilter();");
print ("\t\tdocument.getElementById('Matched').value = '-1';");
print ("\t\tfilterResultTableInverse(9, 'Matched')");
print ("\t\twindow.location.href = '#table4';");
print ("\t\t}");
print ("");
print ("\t\tfunction notResultFilter( ) {");
print ("\t\tvar input, filter, table, tr, td, i, rowNum, count;");
print ("\t\tcount=0;");
print ("\t\tif(document.getElementById('GroundTruthText').value != ''){");
print ("\t\tinput = document.getElementById('GroundTruthText');  rowNum = 2;count+= 1;}");
print ("\t\tif(document.getElementById('GroundTruthID').value != ''){");
print ("\t\tinput = document.getElementById('GroundTruthID'); rowNum = 3;count+= 1;}");
print ("\t\tif(document.getElementById('Matched').value != ''){");
print ("\t\tinput = document.getElementById('Matched');  rowNum = 9;count+= 1;}");
print ("\t\tif(document.getElementById('Top1').value != ''){");
print ("\t\tinput = document.getElementById('Top1'); rowNum = 4;count+= 1; }");
print ("\t\tif(document.getElementById('Top1Prob').value != ''){");
print ("\t\tinput = document.getElementById('Top1Prob');rowNum = 15;count+= 1;}");
print ("\t\tif(document.getElementById('Text1').value != ''){");
print ("\t\tinput = document.getElementById('Text1');rowNum = 10;count+= 1;}");
print ("\t\tif(document.getElementById('Top2').value != ''){");
print ("\t\tinput = document.getElementById('Top2');rowNum = 5;count+= 1;}");
print ("\t\tif(document.getElementById('Top2Prob').value != ''){");
print ("\t\tinput = document.getElementById('Top2Prob');rowNum = 16;count+= 1;}");
print ("\t\tif(document.getElementById('Top3').value != ''){");
print ("\t\tinput = document.getElementById('Top3');rowNum = 6;count+= 1;}");
print ("\t\tif(document.getElementById('Top3Prob').value != ''){");
print ("\t\tinput = document.getElementById('Top3Prob');rowNum = 17;count+= 1;}");
print ("\t\tif(document.getElementById('Top4').value != ''){");
print ("\t\tinput = document.getElementById('Top4');rowNum = 7;count+= 1;}");
print ("\t\tif(document.getElementById('Top4Prob').value != ''){");
print ("\t\tinput = document.getElementById('Top4Prob');rowNum = 18;count+= 1;}");
print ("\t\tif(document.getElementById('Top5').value != ''){");
print ("\t\tinput = document.getElementById('Top5');rowNum = 8;count+= 1;}");
print ("\t\tif(document.getElementById('Top5Prob').value != ''){");
print ("\t\tinput = document.getElementById('Top5Prob');rowNum = 19;count+= 1;}");
print ("\t\tif(count == 0){alert(\"Not Filter ERROR: No filter variable entered\");}");
print ("\t\telse if(count > 1){");
print ("\t\talert(\"Not Filter ERROR: Only one variable filtering supported. Use Clear Filter and enter one filter variable\");}");
print ("\t\tfilter = input.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttd = tr[i].getElementsByTagName(\"td\")[rowNum];");
print ("\t\tif (td) { if (td.innerHTML.toUpperCase().indexOf(filter) <= -1) {tr[i].style.display = \"\"; }");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("");
print ("\t\tfunction andResultFilter( ) {");
print ("\t\tvar inputOne, inputTwo, filterOne, filterTwo, table, tr, tdOne, tdTwo, i, rowNumOne, rowNumTwo,count;");
print ("\t\tcount=0;");
print ("\t\trowNumOne=0;");
print ("\t\tif(document.getElementById('GroundTruthText').value != ''){");
print ("\t\tinputOne = document.getElementById('GroundTruthText');   rowNumOne = 2;count+= 1;}");
print ("\t\telse if(document.getElementById('GroundTruthID').value != ''){");
print ("\t\tinputOne = document.getElementById('GroundTruthID'); rowNumOne = 3;count+= 1;}");
print ("\t\telse if(document.getElementById('Matched').value != ''){");
print ("\t\tinputOne = document.getElementById('Matched');   rowNumOne = 9;count+= 1;}");
print ("\t\telse if(document.getElementById('Top1').value != ''){");
print ("\t\tinputOne = document.getElementById('Top1'); rowNumOne = 4;count+= 1; }");
print ("\t\telse if(document.getElementById('Top1Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top1Prob');rowNumOne = 15;count+= 1;}");
print ("\t\telse if(document.getElementById('Text1').value != ''){");
print ("\t\tinputOne = document.getElementById('Text1');rowNumOne = 10;count+= 1;}");
print ("\t\telse if(document.getElementById('Top2').value != ''){");
print ("\t\tinputOne = document.getElementById('Top2');rowNumOne = 5;count+= 1;}");
print ("\t\telse if(document.getElementById('Top2Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top2Prob');rowNumOne = 16;count+= 1;}");
print ("\t\telse if(document.getElementById('Top3').value != ''){");
print ("\t\tinputOne = document.getElementById('Top3');rowNumOne = 6;count+= 1;}");
print ("\t\telse if(document.getElementById('Top3Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top3Prob');rowNumOne = 17;count+= 1;}");
print ("\t\telse if(document.getElementById('Top4').value != ''){");
print ("\t\tinputOne = document.getElementById('Top4');rowNumOne = 7;count+= 1;}");
print ("\t\telse if(document.getElementById('Top4Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top4Prob');rowNumOne = 18;count+= 1;}");
print ("\t\telse if(document.getElementById('Top5').value != ''){");
print ("\t\tinputOne = document.getElementById('Top5');rowNumOne = 8;count+= 1;}");
print ("\t\telse if(document.getElementById('Top5Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top5Prob');rowNumOne = 19;count+= 1;}");
print ("\t\tif(document.getElementById('GroundTruthText').value != '' && rowNumOne  != 2){");
print ("\t\tinputTwo = document.getElementById('GroundTruthText');   rowNumTwo = 2;count+= 1;}");
print ("\t\telse if(document.getElementById('GroundTruthID').value != '' && rowNumOne  != 3){");
print ("\t\tinputTwo = document.getElementById('GroundTruthID'); rowNumTwo = 3;count+= 1;}");
print ("\t\telse if(document.getElementById('Matched').value != '' && rowNumOne  != 9){");
print ("\t\tinputTwo = document.getElementById('Matched');   rowNumTwo = 9;count+= 1;}");
print ("\t\telse if(document.getElementById('Top1').value != '' && rowNumOne  != 4){");
print ("\t\tinputTwo = document.getElementById('Top1'); rowNumTwo = 4;count+= 1; }");
print ("\t\telse if(document.getElementById('Top1Prob').value != '' && rowNumOne  != 215){");
print ("\t\tinputTwo = document.getElementById('Top1Prob');rowNumTwo = 15;count+= 1;}");
print ("\t\telse if(document.getElementById('Text1').value != '' && rowNumOne  != 10){");
print ("\t\tinputTwo = document.getElementById('Text1');rowNumTwo = 10;count+= 1;}");
print ("\t\telse if(document.getElementById('Top2').value != '' && rowNumOne  != 5){");
print ("\t\tinputTwo = document.getElementById('Top2');rowNumTwo = 5;count+= 1;}");
print ("\t\telse if(document.getElementById('Top2Prob').value != '' && rowNumOne  != 16){");
print ("\t\tinputTwo = document.getElementById('Top2Prob');rowNumTwo = 16;count+= 1;}");
print ("\t\telse if(document.getElementById('Top3').value != '' && rowNumOne  != 6){");
print ("\t\tinputTwo = document.getElementById('Top3');rowNumTwo = 6;count+= 1;}");
print ("\t\telse if(document.getElementById('Top3Prob').value != '' && rowNumOne  != 17){");
print ("\t\tinputTwo = document.getElementById('Top3Prob');rowNumTwo = 17;count+= 1;}");
print ("\t\telse if(document.getElementById('Top4').value != '' && rowNumOne  != 7){");
print ("\t\tinputTwo = document.getElementById('Top4');rowNumTwo = 7;count+= 1;}");
print ("\t\telse if(document.getElementById('Top4Prob').value != '' && rowNumOne  != 18){");
print ("\t\tinputTwo = document.getElementById('Top4Prob');rowNumTwo = 18;count+= 1;}");
print ("\t\telse if(document.getElementById('Top5').value != '' && rowNumOne  != 8){");
print ("\t\tinputTwo = document.getElementById('Top5');rowNumTwo = 8;count+= 1;}");
print ("\t\telse if(document.getElementById('Top5Prob').value != '' && rowNumOne  != 19){");
print ("\t\tinputTwo = document.getElementById('Top5Prob');rowNumTwo = 19;count+= 1;}");
print ("\t\tif(count == 0){alert(\"AND Filter ERROR: No filter variable entered\");}");
print ("\t\telse if(count == 1){alert(\"AND Filter ERROR: Enter two variables\");}");
print ("\t\telse if(count > 2){");
print ("\t\talert(\"AND Filter ERROR: Only two variable filtering supported. Use Clear Filter and enter two filter variable\");}");
print ("\t\tfilterOne = inputOne.value.toUpperCase();");
print ("\t\tfilterTwo = inputTwo.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttdOne = tr[i].getElementsByTagName(\"td\")[rowNumOne];");
print ("\t\ttdTwo = tr[i].getElementsByTagName(\"td\")[rowNumTwo];");
print ("\t\tif (tdOne && tdTwo) { ");
print ("\t\tif (tdOne.innerHTML.toUpperCase().indexOf(filterOne) > -1 && tdTwo.innerHTML.toUpperCase().indexOf(filterTwo) > -1) ");
print ("\t\t{tr[i].style.display = \"\";}");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("");
print ("\t\tfunction orResultFilter( ) {");
print ("\t\tvar inputOne, inputTwo, filterOne, filterTwo, table, tr, tdOne, tdTwo, i, rowNumOne, rowNumTwo, count;");
print ("\t\tcount=0;");
print ("\t\trowNumOne=0;");
print ("\t\tif(document.getElementById('GroundTruthText').value != ''){");
print ("\t\tinputOne = document.getElementById('GroundTruthText');   rowNumOne = 2;count+= 1;}");
print ("\t\telse if(document.getElementById('GroundTruthID').value != ''){");
print ("\t\tinputOne = document.getElementById('GroundTruthID'); rowNumOne = 3;count+= 1;}");
print ("\t\telse if(document.getElementById('Matched').value != ''){");
print ("\t\tinputOne = document.getElementById('Matched');   rowNumOne = 9;count+= 1;}");
print ("\t\telse if(document.getElementById('Top1').value != ''){");
print ("\t\tinputOne = document.getElementById('Top1'); rowNumOne = 4;count+= 1; }");
print ("\t\telse if(document.getElementById('Top1Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top1Prob');rowNumOne = 15;count+= 1;}");
print ("\t\telse if(document.getElementById('Text1').value != ''){");
print ("\t\tinputOne = document.getElementById('Text1');rowNumOne = 10;count+= 1;}");
print ("\t\telse if(document.getElementById('Top2').value != ''){");
print ("\t\tinputOne = document.getElementById('Top2');rowNumOne = 5;count+= 1;}");
print ("\t\telse if(document.getElementById('Top2Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top2Prob');rowNumOne = 16;count+= 1;}");
print ("\t\telse if(document.getElementById('Top3').value != ''){");
print ("\t\tinputOne = document.getElementById('Top3');rowNumOne = 6;count+= 1;}");
print ("\t\telse if(document.getElementById('Top3Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top3Prob');rowNumOne = 17;count+= 1;}");
print ("\t\telse if(document.getElementById('Top4').value != ''){");
print ("\t\tinputOne = document.getElementById('Top4');rowNumOne = 7;count+= 1;}");
print ("\t\telse if(document.getElementById('Top4Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top4Prob');rowNumOne = 18;count+= 1;}");
print ("\t\telse if(document.getElementById('Top5').value != ''){");
print ("\t\tinputOne = document.getElementById('Top5');rowNumOne = 8;count+= 1;}");
print ("\t\telse if(document.getElementById('Top5Prob').value != ''){");
print ("\t\tinputOne = document.getElementById('Top5Prob');rowNumOne = 19;count+= 1;}");
print ("\t\tif(document.getElementById('GroundTruthText').value != '' && rowNumOne  != 2){");
print ("\t\tinputTwo = document.getElementById('GroundTruthText');   rowNumTwo = 2;count+= 1;}");
print ("\t\telse if(document.getElementById('GroundTruthID').value != '' && rowNumOne  != 3){");
print ("\t\tinputTwo = document.getElementById('GroundTruthID'); rowNumTwo = 3;count+= 1;}");
print ("\t\telse if(document.getElementById('Matched').value != '' && rowNumOne  != 9){");
print ("\t\tinputTwo = document.getElementById('Matched');   rowNumTwo = 9;count+= 1;}");
print ("\t\telse if(document.getElementById('Top1').value != '' && rowNumOne  != 4){");
print ("\t\tinputTwo = document.getElementById('Top1'); rowNumTwo = 4;count+= 1; }");
print ("\t\telse if(document.getElementById('Top1Prob').value != '' && rowNumOne  != 215){");
print ("\t\tinputTwo = document.getElementById('Top1Prob');rowNumTwo = 15;count+= 1;}");
print ("\t\telse if(document.getElementById('Text1').value != '' && rowNumOne  != 10){");
print ("\t\tinputTwo = document.getElementById('Text1');rowNumTwo = 10;count+= 1;}");
print ("\t\telse if(document.getElementById('Top2').value != '' && rowNumOne  != 5){");
print ("\t\tinputTwo = document.getElementById('Top2');rowNumTwo = 5;count+= 1;}");
print ("\t\telse if(document.getElementById('Top2Prob').value != '' && rowNumOne  != 16){");
print ("\t\tinputTwo = document.getElementById('Top2Prob');rowNumTwo = 16;count+= 1;}");
print ("\t\telse if(document.getElementById('Top3').value != '' && rowNumOne  != 6){");
print ("\t\tinputTwo = document.getElementById('Top3');rowNumTwo = 6;count+= 1;}");
print ("\t\telse if(document.getElementById('Top3Prob').value != '' && rowNumOne  != 17){");
print ("\t\tinputTwo = document.getElementById('Top3Prob');rowNumTwo = 17;count+= 1;}");
print ("\t\telse if(document.getElementById('Top4').value != '' && rowNumOne  != 7){");
print ("\t\tinputTwo = document.getElementById('Top4');rowNumTwo = 7;count+= 1;}");
print ("\t\telse if(document.getElementById('Top4Prob').value != '' && rowNumOne  != 18){");
print ("\t\tinputTwo = document.getElementById('Top4Prob');rowNumTwo = 18;count+= 1;}");
print ("\t\telse if(document.getElementById('Top5').value != '' && rowNumOne  != 8){");
print ("\t\tinputTwo = document.getElementById('Top5');rowNumTwo = 8;count+= 1;}");
print ("\t\telse if(document.getElementById('Top5Prob').value != '' && rowNumOne  != 19){");
print ("\t\tinputTwo = document.getElementById('Top5Prob');rowNumTwo = 19;count+= 1;}");
print ("\t\tif(count == 0){alert(\"OR Filter ERROR: No filter variable entered\");}");
print ("\t\telse if(count == 1){alert(\"OR Filter ERROR: Enter two variables\");}");
print ("\t\telse if(count > 2){");
print ("\t\talert(\"OR Filter ERROR: Only two variable filtering supported. Use Clear Filter and enter two filter variable\");}");
print ("\t\tfilterOne = inputOne.value.toUpperCase();");
print ("\t\tfilterTwo = inputTwo.value.toUpperCase();");
print ("\t\ttable = document.getElementById(\"resultsTable\");");
print ("\t\ttr = table.getElementsByTagName(\"tr\");");
print ("\t\tfor (i = 1; i < tr.length; i+= 1) {");
print ("\t\ttdOne = tr[i].getElementsByTagName(\"td\")[rowNumOne];");
print ("\t\ttdTwo = tr[i].getElementsByTagName(\"td\")[rowNumTwo];");
print ("\t\tif (tdOne && tdTwo) { ");
print ("\t\tif (tdOne.innerHTML.toUpperCase().indexOf(filterOne) > -1 || tdTwo.innerHTML.toUpperCase().indexOf(filterTwo) > -1) ");
print ("\t\t{tr[i].style.display = \"\";}");
print ("\t\telse { tr[i].style.display = \"none\";}}}}");
print ("");
print ("\t</script>");
print ("");

#TBD: Graph CODE

#Top view header
print ("\t<div class=\"navbar\">");
print ("\t<a href=\"#\">");
print ("\t<div id=\"main\">");
print ("\t<span style=\"font-size:30px;cursor:pointer\" onclick=\"openNav()\">&#9776; Views</span>");
print ("\t</div></a>");
print ("\t<a href=\"https://www.amd.com/en\" target=\"_blank\">");
print ("\t<img \" src=\"icons/small_amd_logo.png\" alt=\"AMD\" /></a>");
print ("\t<a href=\"https://gpuopen.com/\" target=\"_blank\">");
print ("\t<img \" src=\"icons/small_radeon_logo.png\" alt=\"GPUopen\" /></a>");
print ("\t<a href=\"https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules#amd-openvx-modules-amdovx-modules\" target=\"_blank\">");
print ("\t<img \" src=\"icons/small_github_logo.png\" alt=\"AMD GitHub\" /></a>");
print ("\t<img \" src=\"icons/ADAT_500x100.png\" alt=\"AMD Inference ToolKit\" hspace=\"100\" height=\"90\"/> ");
print ("\t</div>");
print ("\t");

# graph script
print("\t<script type=\"text/javascript\" src=\"https://www.gstatic.com/charts/loader.js\"></script>");
print("\t<script type=\"text/javascript\">");
print("\t");
#overall summary
print("\tgoogle.charts.load('current', {'packages':['bar']});");
print("\tgoogle.charts.setOnLoadCallback(drawChart);");
print("\tfunction drawChart(){");
print("\tvar data = google.visualization.arrayToDataTable([");
print("\t['  '     ,  'Match'  , 'Mismatch', 'No Label' ],");
print("\t['Summary',   %d     , %d        , %d         ]"%(passCount,totalMismatch,totalNoGroundTruth));
print("\t]);");
print("\tvar options = { title: 'Overall Result Summary', vAxis: { title: 'Images' }, width: 800, height: 400 };");
print("\tvar chart = new google.charts.Bar(document.getElementById('Model_Stats'));");
print("\tchart.draw(data, google.charts.Bar.convertOptions(options));}");
print("\t");

#TopK pass fail summary
topKValue = 5;
print("\tgoogle.charts.load('current', {'packages':['corechart']});");
print("\tgoogle.charts.setOnLoadCallback(drawTopKResultChart);");
print("\tfunction drawTopKResultChart() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('string', 'Top K');");
print("\tdata.addColumn('number', 'Matchs');");
print("\tdata.addRows([");
print("\t[ 'Matched Top%d Choice', %d  ],"%(topKValue,passCount));
print("\t[ 'MisMatched', %d  ]]);"%(totalMismatch));
print("\tvar options = { title:'Image Match/Mismatch Summary', width:750, height:400 };");
print("\tvar chart = new google.visualization.PieChart(document.getElementById('topK_result_chart_div'));");
print("\tchart.draw(data, options);}");
print("\t");

#topK summary
print("\tgoogle.charts.load('current', {'packages':['corechart']});");
print("\tgoogle.charts.setOnLoadCallback(drawResultChart);");
print("\tfunction drawResultChart() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('string', 'Top K');");
print("\tdata.addColumn('number', 'Matchs');");
print("\tdata.addRows([");
print("\t[ 'Matched 1st Choice', %d  ],"%(top1Count));
print("\t[ 'Matched 2nd Choice', %d  ],"%(top2Count));
print("\t[ 'Matched 3rd Choice', %d  ],"%(top3Count));
print("\t[ 'Matched 4th Choice', %d  ],"%(top4Count));
print("\t[ 'Matched 5th Choice', %d  ]]);"%(top5Count));
print("\tvar options = { title:'Image Matches', width:750, height:400 };");
print("\tvar chart = new google.visualization.PieChart(document.getElementById('result_chart_div'));");
print("\tchart.draw(data, options);}");
print("\t");
#Cummulative Success/Failure
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(drawPassFailGraph);");
print("\tfunction drawPassFailGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Match');");
print("\tdata.addColumn('number', 'Mismatch');");
print("\tdata.addRows([");
print("\t[1, 0, 0],");
fVal=0.99;
sumPass = 0;
sumFail = 0;
i = 99;
while i >= 0:
    sumPass = float(sumPass + topKPassFail[i][0]);
    sumFail = float(sumFail + topKPassFail[i][1]);
    if(i == 0):
        print("\t[%.2f,   %.4f,    %.4f]"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    else:
        print("\t[%.2f,   %.4f,    %.4f],"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    fVal=fVal-0.01;
    i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Cummulative Success/Failure', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Percentage of Dataset'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('pass_fail_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

#Cummulative L1 Success/Failure
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(drawL1PassFailGraph);");
print("\tfunction drawL1PassFailGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'L1 Match');");
print("\tdata.addColumn('number', 'L1 Mismatch');");
print("\tdata.addRows([");
print("\t[1, 0, 0],");
fVal=0.99;
sumPass = 0;
sumFail = 0;
i = 99;
while i >= 0:
    sumPass = float(sumPass + topKHierarchyPassFail[i][0]);
    sumFail = float(sumFail + topKHierarchyPassFail[i][1]);
    if(i == 0):
        print("\t[%.2f,   %.4f,    %.4f]"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    else:
        print("\t[%.2f,   %.4f,    %.4f],"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    fVal=fVal-0.01;
    i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Cummulative L1 Success/Failure', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Percentage of Dataset'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('L1_pass_fail_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

#Cummulative L2 Success/Failure
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(drawL2PassFailGraph);");
print("\tfunction drawL2PassFailGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'L2 Match');");
print("\tdata.addColumn('number', 'L2 Mismatch');");
print("\tdata.addRows([");
print("\t[1, 0, 0],");
fVal=0.99;
sumPass = 0;
sumFail = 0;
i = 99;
while i >= 0:
    sumPass = float(sumPass + topKHierarchyPassFail[i][2]);
    sumFail = float(sumFail + topKHierarchyPassFail[i][3]);
    if(i == 0):
        print("\t[%.2f,   %.4f,    %.4f]"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    else:
        print("\t[%.2f,   %.4f,    %.4f],"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    fVal=fVal-0.01;
    i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Cummulative L2 Success/Failure', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Percentage of Dataset'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('L2_pass_fail_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

#Cummulative L3 Success/Failure
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(drawL3PassFailGraph);");
print("\tfunction drawL3PassFailGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'L3 Match');");
print("\tdata.addColumn('number', 'L3 Mismatch');");
print("\tdata.addRows([");
print("\t[1, 0, 0],");
fVal=0.99;
sumPass = 0;
sumFail = 0;
i = 99;
while i >= 0:
    sumPass = float(sumPass + topKHierarchyPassFail[i][4]);
    sumFail = float(sumFail + topKHierarchyPassFail[i][5]);
    if(i == 0):
        print("\t[%.2f,   %.4f,    %.4f]"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    else:
        print("\t[%.2f,   %.4f,    %.4f],"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    fVal=fVal-0.01;
    i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Cummulative L3 Success/Failure', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Percentage of Dataset'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('L3_pass_fail_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

#Cummulative L4 Success/Failure
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(drawL4PassFailGraph);");
print("\tfunction drawL4PassFailGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'L4 Match');");
print("\tdata.addColumn('number', 'L4 Mismatch');");
print("\tdata.addRows([");
print("\t[1, 0, 0],");
fVal=0.99;
sumPass = 0;
sumFail = 0;
i = 99;
while i >= 0:
    sumPass = float(sumPass + topKHierarchyPassFail[i][6]);
    sumFail = float(sumFail + topKHierarchyPassFail[i][7]);
    if(i == 0):
        print("\t[%.2f,   %.4f,    %.4f]"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    else:
        print("\t[%.2f,   %.4f,    %.4f],"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    fVal=fVal-0.01;
    i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Cummulative L4 Success/Failure', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Percentage of Dataset'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('L4_pass_fail_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

#Cummulative L5 Success/Failure
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(drawL5PassFailGraph);");
print("\tfunction drawL5PassFailGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'L5 Match');");
print("\tdata.addColumn('number', 'L5 Mismatch');");
print("\tdata.addRows([");
print("\t[1, 0, 0],");
fVal=0.99;
sumPass = 0;
sumFail = 0;
i = 99;
while i >= 0:
    sumPass = float(sumPass + topKHierarchyPassFail[i][8]);
    sumFail = float(sumFail + topKHierarchyPassFail[i][9]);
    if(i == 0):
        print("\t[%.2f,   %.4f,    %.4f]"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    else:
        print("\t[%.2f,   %.4f,    %.4f],"%(fVal,(sumPass/netSummaryImages),(sumFail/netSummaryImages)));
    fVal=fVal-0.01;
    i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Cummulative L5 Success/Failure', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Percentage of Dataset'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('L5_pass_fail_chart'));");
print("\tchart.draw(data, options);}");
print("\t");

#Cummulative Hierarchy Success/Failure
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(drawHierarchyPassFailGraph);");
print("\tfunction drawHierarchyPassFailGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'L1 Match');");
print("\tdata.addColumn('number', 'L1 Mismatch');");
print("\tdata.addColumn('number', 'L2 Match');");
print("\tdata.addColumn('number', 'L2 Mismatch');");
print("\tdata.addColumn('number', 'L3 Match');");
print("\tdata.addColumn('number', 'L3 Mismatch');");
print("\tdata.addColumn('number', 'L4 Match');");
print("\tdata.addColumn('number', 'L4 Mismatch');");
print("\tdata.addColumn('number', 'L5 Match');");
print("\tdata.addColumn('number', 'L5 Mismatch');");
print("\tdata.addColumn('number', 'L6 Match');");
print("\tdata.addColumn('number', 'L6 Mismatch');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],");
fVal=0.99;
l1Pass = 0;
l1Fail = 0;
l2Pass = 0;
l2Fail = 0;
l3Pass = 0;
l3Fail = 0;
l4Pass = 0;
l4Fail = 0;
l5Pass = 0;
l5Fail = 0;
l6Pass = 0;
l6Fail = 0;
i = 99;
while i >= 0:
    l1Pass = float(l1Pass + topKHierarchyPassFail[i][0]);
    l1Fail = float(l1Fail + topKHierarchyPassFail[i][1]);
    l2Pass = float(l2Pass + topKHierarchyPassFail[i][2]);
    l2Fail = float(l2Fail + topKHierarchyPassFail[i][3]);
    l3Pass = float(l3Pass + topKHierarchyPassFail[i][4]);
    l3Fail = float(l3Fail + topKHierarchyPassFail[i][5]);
    l4Pass = float(l4Pass + topKHierarchyPassFail[i][6]);
    l4Fail = float(l4Fail + topKHierarchyPassFail[i][7]);
    l5Pass = float(l5Pass + topKHierarchyPassFail[i][8]);
    l5Fail = float(l5Fail + topKHierarchyPassFail[i][9]);
    l6Pass = float(l6Pass + topKHierarchyPassFail[i][10]);
    l6Fail = float(l6Fail + topKHierarchyPassFail[i][11]);
    if(i == 0):
        print("\t[%.2f,   %.4f,    %.4f,   %.4f,    %.4f,   %.4f,    %.4f,   %.4f,    %.4f,   %.4f,    %.4f,   %.4f,    %.4f]"%(fVal,
            (l1Pass/netSummaryImages),(l1Fail/netSummaryImages),
            (l2Pass/netSummaryImages),(l2Fail/netSummaryImages),
            (l3Pass/netSummaryImages),(l3Fail/netSummaryImages),
            (l4Pass/netSummaryImages),(l4Fail/netSummaryImages),
            (l5Pass/netSummaryImages),(l5Fail/netSummaryImages),
            (l6Pass/netSummaryImages),(l6Fail/netSummaryImages)
            ));
    else:
        print("\t[%.2f,   %.4f,    %.4f,   %.4f,    %.4f,   %.4f,    %.4f,   %.4f,    %.4f,   %.4f,    %.4f,   %.4f,    %.4f],"%(fVal,
            (l1Pass/netSummaryImages),(l1Fail/netSummaryImages),
            (l2Pass/netSummaryImages),(l2Fail/netSummaryImages),
            (l3Pass/netSummaryImages),(l3Fail/netSummaryImages),
            (l4Pass/netSummaryImages),(l4Fail/netSummaryImages),
            (l5Pass/netSummaryImages),(l5Fail/netSummaryImages),
            (l6Pass/netSummaryImages),(l6Fail/netSummaryImages)
            ));
    fVal=fVal-0.01;
    i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Cummulative Hierarchy Levels Success/Failure', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Percentage of Dataset'}, series: { 0.01: {curveType: 'function'} }, width:1400, height:800 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('Hierarchy_pass_fail_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
print("\t</script>");

#Overall Summary
print("\t<!-- Overall Summary -->");
print("\t<A NAME=\"table0\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>%s Overall Summary</em></font></h1></A>" %(modelName));
print("\t<table align=\"center\">");
print("\t<col width=\"265\">");
print("\t<col width=\"50\">");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Images <b>With Ground Truth</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\" onclick=\"findImagesWithGroundTruthLabel()\"><b>%d</b></font></td>"%(netSummaryImages));
print("\t</tr>");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Images <b>Without Ground Truth</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\" onclick=\"findImagesWithNoGroundTruthLabel()\"><b>%d</b></font></td>"%(totalNoGroundTruth));
print("\t</tr>");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\"><b>Total Images</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\" onclick=\"clearResultFilter();goToImageResults();\"><b>%d</b></font></td>"%(imageDataSize));
print("\t</tr>");
print("\t</table>\n<br><br><br>");
print("\t<table align=\"center\">\n \t<col width=\"300\">\n \t<col width=\"100\">\n \t<col width=\"350\">\n \t<col width=\"100\">\n<tr>");
print("\t<td><font color=\"black\" size=\"4\">Total <b>Top %d Match</b></font></td>\n"%(topKValue));
print("\t <td align=\"center\"><font color=\"black\" size=\"4\" onclick=\"findTopKMatch()\"><b>%d</b></font></td>"%(passCount));
print("\t<td><font color=\"black\" size=\"4\">Total <b>Mismatch</b></font></td>");
print("\t <td align=\"center\"><font color=\"black\" size=\"4\" onclick=\"findImageMisMatch()\"><b>%d</b></font></td>"%(totalMismatch));
print("\t</tr>\n<tr>");
print("\t<td><font color=\"black\" size=\"4\"><b>Accuracy on Top %d</b></font></td>"%(topKValue));

accuracyPer = float(passCount);
accuracyPer = (accuracyPer / netSummaryImages)*100;
print("\t <td align=\"center\"><font color=\"black\" size=\"4\"><b>%.2f %%</b></font></td>"%(accuracyPer));
print("\t<td><font color=\"black\" size=\"4\"><b>Mismatch Percentage</b></font></td>");
accuracyPer = float(totalMismatch);
accuracyPer = (accuracyPer/netSummaryImages)*100;
print("\t <td align=\"center\"><font color=\"black\" size=\"4\"><b>%.2f %%</b></font></td>"%(accuracyPer));
print("\t</tr>\n<tr>");
print("\t<td><font color=\"black\" size=\"4\">Average Pass Confidence for Top %d</font></td>"%(topKValue));
print("\t <td align=\"center\"><font color=\"black\" size=\"4\"><b>%.2f %%</b></font></td>"%((avgPassProb*100)));
print("\t<td><font color=\"black\" size=\"4\">Average mismatch Confidence for Top 1</font></td>");
print("\t <td align=\"center\"><font color=\"black\" size=\"4\"><b>%.2f %%</b></font></td>"%(((totalFailProb/totalMismatch)*100)));
print("\t</tr>\n</table>\n<br><br><br>");
#topK result
print("\t<table align=\"center\" style=\"width: 40%\">");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>1st Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>2nd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>3rd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>4th Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>5th Match</b></font></td>");
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top2Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top3Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top4Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top5Count));
print("\t\t</tr>");
print("\t<tr>");
accuracyPer = float(top1Count);
accuracyPer = (accuracyPer/netSummaryImages)*100;
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(accuracyPer));
accuracyPer = float(top2Count);
accuracyPer = (accuracyPer/netSummaryImages)*100;
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(accuracyPer));
accuracyPer = float(top3Count);
accuracyPer = (accuracyPer/netSummaryImages)*100;
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(accuracyPer));
accuracyPer = float(top4Count);
accuracyPer = (accuracyPer/netSummaryImages)*100;
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(accuracyPer));
accuracyPer = float(top5Count);
accuracyPer = (accuracyPer/netSummaryImages)*100;
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(accuracyPer));
print("\t\t</tr>");
print("</table>");

#summary date and time
print("\t<h1 align=\"center\"><font color=\"DodgerBlue\" size=\"4\"><br><em>Summary Generated On: </font><font color=\"black\" size=\"4\"> %s</font></em></h1>"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')));

#Graph
print("\t<!-- Graph Summary -->");
print("<A NAME=\"table1\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Graphs</em></font></h1></A>");
print("\t<center><div id=\"Model_Stats\" style=\"border: 1px solid #ccc\"></div></center>");
print("\t<table align=\"center\" style=\"width: 90%\">");
print("\t<tr>");
print("\t <td><center><div id=\"result_chart_div\" style=\"border: 0px solid #ccc\"></div></center></td>");
print("\t <td><center><div id=\"topK_result_chart_div\" style=\"border: 0px solid #ccc\"></div></center></td>");
print("\t</tr>");
print("\t<tr>");
print("\t <td><center><div id=\"pass_fail_chart\" style=\"border: 0px solid #ccc\" ></div></center> </td>");
print("\t <td><center><div id=\"L1_pass_fail_chart\" style=\"border: 0px solid #ccc\" ></div></center> </td>");
print("\t</tr>");
print("\t<tr>");
print("\t <td><center><div id=\"L2_pass_fail_chart\" style=\"border: 0px solid #ccc\" ></div></center> </td>");
print("\t <td><center><div id=\"L3_pass_fail_chart\" style=\"border: 0px solid #ccc\" ></div></center> </td>");
print("\t</tr>");
print("\t<tr>");
print("\t <td><center><div id=\"L4_pass_fail_chart\" style=\"border: 0px solid #ccc\" ></div></center> </td>");
print("\t <td><center><div id=\"L5_pass_fail_chart\" style=\"border: 0px solid #ccc\" ></div></center> </td>");
print("\t</tr>");
print("\t</table>");
print("\t");
print("\t <td><center><div id=\"Hierarchy_pass_fail_chart\" style=\"border: 0px solid #ccc\" ></div></center> </td>");
print("\t");

#hierarchy
print("\t<!-- hierarchy Summary -->");
print("<A NAME=\"table2\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Hierarchy Summary (by Confidence level)</em></font></h1></A>");
print("\t<table align=\"center\" style=\"width: 80%\">");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Confidence</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Pass</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Fail</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 1 Pass</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 1 Fail</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 2 Pass</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 2 Fail</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 3 Pass</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 3 Fail</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 4 Pass</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 4 Fail</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 5 Pass</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 5 Fail</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 6 Pass</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Category 6 Fail</b></font></td>");
print("\t\t</tr>");

f=0.99;
i = 99;
while i >= 0:
    print("\t\t<tr>");
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%.2f</b></font></td>"%(f));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKPassFail[i][0]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKPassFail[i][1]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][0]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][1]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][2]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][3]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][4]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][5]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][6]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][7]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][8]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][9]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][10]));
    print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topKHierarchyPassFail[i][11]));
    print("\t\t</tr>");
    f=f-0.01;
    i=i-1;

print("</table>");


#label
print("\t<!-- Label Summary -->");
print("<A NAME=\"table3\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Label Summary (stats per image class)</em></font></h1></A>");
print("\t\t<table id=\"filterLabelTable\" align=\"center\" cellspacing=\"2\" border=\"0\" style=\"width: 70%\">");
print("\t\t<tr>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Label ID\" onkeyup=\"filterLabelTable(0,id)\" placeholder=\"Label ID\" title=\"Label ID\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Label Description\" onkeyup=\"filterLabelTable(1,id)\" placeholder=\"Label Description\" title=\"Label Description\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Images in DataBase\" onkeyup=\"filterLabelTable(2,id)\" placeholder=\"Images in DataBase\" title=\"Images in DataBase\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Matched Top1 %\" onkeyup=\"filterLabelTable(3,id)\" placeholder=\"Matched Top1 %\" title=\"Matched Top1 %\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Matched Top5 %\" onkeyup=\"filterLabelTable(4,id)\" placeholder=\"Matched Top5 %\" title=\"Matched Top5 %\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Matched 1st\" onkeyup=\"filterLabelTable(5,id)\" placeholder=\"Matched 1st\" title=\"Matched 1st\"></td>");
print("\t\t</tr>\n\t\t<tr>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Matched 2nd\" onkeyup=\"filterLabelTable(6,id)\" placeholder=\"Matched 2nd\" title=\"Matched 2nd\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Matched 3th\" onkeyup=\"filterLabelTable(7,id)\" placeholder=\"Matched 3th\" title=\"Matched 3th\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Matched 4th\" onkeyup=\"filterLabelTable(8,id)\" placeholder=\"Matched 4th\" title=\"Matched 4th\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Matched 5th\" onkeyup=\"filterLabelTable(9,id)\" placeholder=\"Matched 5th\" title=\"Matched 5th\"></td>");
print("\t\t<td><input type=\"text\" size=\"14\" id=\"Misclassified Top1 Label\" onkeyup=\"filterLabelTable(10,id)\"placeholder=\"Misclassified Top1 Label\" title=\"Misclassified Top1 Label\"></td>");
print("\t\t<td align=\"center\"><button style=\"background-color:yellow;\" onclick=\"clearLabelFilter()\">Clear Filter</button></td>");
print("\t\t</tr>");
print("\t\t</table>");
print("\t\t<br>");
print("\t\t");
print("\t<table class=\"sortable\" id=\"labelsTable\" align=\"center\">");
print("\t<col width=\"80\">");
print("\t<col width=\"200\">");
print("\t<col width=\"100\">");
print("\t<col width=\"100\">");
print("\t<col width=\"100\">");
print("\t<col width=\"100\">");
print("\t<col width=\"100\">");
print("\t<col width=\"100\">");
print("\t<col width=\"100\">");
print("\t<col width=\"100\">");
print("\t<col width=\"150\">");
print("\t<col width=\"60\">");
print("\t<tr>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Label ID</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Label Description</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Images in DataBase</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Matched Top1 %</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Matched Top5 %</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Matched 1st</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Matched 2nd</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Matched 3rd</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Matched 4th</b></font></td>");
print("\t<td align=\"center\"><font color=\"maroon\" size=\"3\"><b>Matched 5th</b></font></td>");
print("\t<td align=\"center\"><font color=\"blue\" size=\"3\"><b>Misclassified Top1 Label</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>Check</b></font></td>");
print("\t\t</tr>");
totalLabelsFound = 0;
totalImagesWithLabelFound = 0;
totalLabelsUnfounded = 0;
totalImagesWithLabelNotFound = 0;
totalLabelsNeverfound = 0;
totalImagesWithFalseLabelFound = 0;
i = 0;       
while i < 1000:
    if(topLabelMatch[i][0] or topLabelMatch[i][6]):
        labelTxt = (LabelLines[i].split(' ', 1)[1].rstrip('\n'));
        labelTxt = labelTxt.lower();
        print("\t<tr>");
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\" onclick=\"findGroundTruthLabel('%s',%d)\"><b>%d</b></font></td>"%(labelTxt,i,i));
        print("\t\t<td align=\"left\" onclick=\"findGroundTruthLabel('%s',%d)\"><b>%s</b></td>"%(labelTxt,i,labelTxt)); 
        if(topLabelMatch[i][0]):
            top1 = 0;
            top5 = 0;
            top1 = float(topLabelMatch[i][1]);
            top1 = ((top1/topLabelMatch[i][0])*100);
            top5 = float(topLabelMatch[i][1]+topLabelMatch[i][2]+topLabelMatch[i][3]+topLabelMatch[i][4]+topLabelMatch[i][5]);
            top5 = (float(top5/topLabelMatch[i][0])*100);
            imagesFound = (topLabelMatch[i][1]+topLabelMatch[i][2]+topLabelMatch[i][3]+topLabelMatch[i][4]+topLabelMatch[i][5]);
            totalImagesWithLabelFound += imagesFound;
            totalImagesWithLabelNotFound += topLabelMatch[i][0] - imagesFound;
            if(top5 == 100.00):
                print("\t\t<td align=\"center\"><font color=\"green\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][0]));
            else:
                print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][0]));
            print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%.2f</b></font></td>"%(top1));  
            print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%.2f</b></font></td>"%(top5));
            totalLabelsFound+=1;
            if(top5 == 0.00):
                totalLabelsNeverfound+=1;
        else:
            print("\t\t<td align=\"center\"><font color=\"red\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][0]));                
            print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>0.00</b></font></td>");
            print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>0.00</b></font></td>");
            totalLabelsUnfounded+=1;
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][1]));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][2]));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][3]));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][4]));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][5]));
        if(topLabelMatch[i][0] and topLabelMatch[i][6] ):
            print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][6]));
        else:
            if(topLabelMatch[i][0]):
                print("\t\t<td align=\"center\"><font color=\"green\" size=\"2\"><b>%d</b></font></td>"%(topLabelMatch[i][6]));
            else:
                totalImagesWithFalseLabelFound += topLabelMatch[i][6];
                print("\t\t<td align=\"center\"><font color=\"red\" size=\"2\" onclick=\"findMisclassifiedGroundTruthLabel('%s')\"><b>%d</b></font></td>"%(labelTxt,topLabelMatch[i][6]));
        print("\t\t<td align=\"center\"><input id=\"id_%d\" name=\"id[%d]\" type=\"checkbox\" value=\"%d\" onClick=\"highlightRow(this);\"></td>"%(i,i,i));
        print("\t\t</tr>");
    i = i + 1;

print("</table>");
print("<h1 align=\"center\"><font color=\"DodgerBlue\" size=\"4\"><br><em>Label Summary</em></font></h1>");
print("\t<table align=\"center\">");
print("\t<col width=\"350\">");
print("\t<col width=\"50\">");
print("\t<col width=\"150\">");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Labels in Ground Truth <b>found</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%((totalLabelsFound-totalLabelsNeverfound)));
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b> images</font></td>"%(totalImagesWithLabelFound));
print("\t</tr>");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Labels in Ground Truth <b>not found</b></font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalLabelsNeverfound));
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b> images</font></td>"%(totalImagesWithLabelNotFound));
print("\t</tr>");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\"><b>Total</b> Labels in Ground Truth</font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalLabelsFound));
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b> images</font></td>"%((totalImagesWithLabelFound+totalImagesWithLabelNotFound)));
print("\t</tr>");
print("</table>");
print("\t<br><br><table align=\"center\">");
print("\t<col width=\"400\">");
print("\t<col width=\"50\">");
print("\t<col width=\"150\">");
print("\t<tr>");
print("\t<td><font color=\"black\" size=\"4\">Labels <b>not in Ground Truth</b> found in 1st Match</font></td>");
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b></font></td>"%(totalLabelsUnfounded));
print("\t<td align=\"center\"><font color=\"black\" size=\"4\"><b>%d</b> images</font></td>"%(totalImagesWithFalseLabelFound));
print("\t</tr>");
print("</table>");


#Image result
print("\t<!-- Image Summary -->");
print("<A NAME=\"table4\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Image Results</em></font></h1></A>");
print("\t\t<table id=\"filterTable\" align=\"center\" cellspacing=\"2\" border=\"0\" style=\"width: 60%\">");
print("\t\t<tr>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"GroundTruthText\" onkeyup=\"filterResultTable(2,id)\" placeholder=\"Ground Truth Text\" title=\"Ground Truth Text\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"GroundTruthID\" onkeyup=\"filterResultTable(3,id)\" placeholder=\"Ground Truth ID\" title=\"Ground Truth ID\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" maxlength=\"2\" id=\"Matched\" onkeyup=\"filterResultTable(9,id)\" placeholder=\"Matched\" title=\"Type in a name\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top1\" onkeyup=\"filterResultTable(4,id)\" placeholder=\"1st Match\" title=\"1st Match\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top1Prob\" onkeyup=\"filterResultTable(15,id)\" placeholder=\"1st Match Conf\" title=\"1st Match Prob\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Text1\" onkeyup=\"filterResultTable(10,id)\" placeholder=\"Text 1\" title=\"Text1\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top2\" onkeyup=\"filterResultTable(5,id)\" placeholder=\"2nd Match\" title=\"2nd Match\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top2Prob\" onkeyup=\"filterResultTable(16,id)\" placeholder=\"2nd Match Conf\" title=\"2nd Match Prob\"></td>");
print("\t\t</tr>\n\t\t<tr>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top3\" onkeyup=\"filterResultTable(6,id)\" placeholder=\"3rd Match\" title=\"3rd Match\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top3Prob\" onkeyup=\"filterResultTable(17,id)\" placeholder=\"3rd Match Conf\" title=\"3rd Match Prob\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top4\" onkeyup=\"filterResultTable(7,id)\" placeholder=\"4th Match\" title=\"4th Match\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top4Prob\" onkeyup=\"filterResultTable(18,id)\" placeholder=\"4th Match Conf\" title=\"4th Match Prob\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top5\" onkeyup=\"filterResultTable(8,id)\" placeholder=\"5th Match\" title=\"5th Match\"></td>");
print("\t\t<td><input type=\"text\" size=\"10\" id=\"Top5Prob\" onkeyup=\"filterResultTable(19,id)\" placeholder=\"5th Match Conf\" title=\"5th Match Prob\"></td>");
print("\t\t<td></td>");
print("\t\t<td align=\"center\"><button style=\"background-color:yellow;\" onclick=\"clearResultFilter()\">Clear Filter</button></td>");
print("\t\t</tr>");
print("\t\t<tr>");
print("\t\t<td align=\"center\"><button style=\"background-color:salmon;\" onclick=\"notResultFilter()\">Not Filter</button></td>");
print("\t\t<td align=\"center\"><button style=\"background-color:salmon;\" onclick=\"andResultFilter()\">AND Filter</button></td>");
print("\t\t<td align=\"center\"><button style=\"background-color:salmon;\" onclick=\"orResultFilter()\">OR Filter</button></td>");
print("\t\t</tr>");
print("\t\t</table>");
print("\t\t<br>");
print("\t\t");
print("<table id=\"resultsTable\" class=\"sortable\" align=\"center\" cellspacing=\"2\" border=\"0\" style=\"width: 98%\">");
print("\t<tr>");
print("\t\t<td height=\"17\" align=\"center\"><font color=\"Maroon\" size=\"2\" ><b>Image</b></font></td>");
print("\t\t<td height=\"17\" align=\"center\"><font color=\"Maroon\" size=\"2\"><b>FileName</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Ground Truth Text</b></font></td>");
print("\t\t<td align=\"center\"><b><div class=\"tooltip\"><font color=\"Maroon\" size=\"2\">Ground Truth</font><span class=\"tooltiptext\">Input Image Label. Click on the Text to Sort</span></div></b></td>");
print("\t\t<td align=\"center\"><b><div class=\"tooltip\"><font color=\"Maroon\" size=\"2\">1st</font><span class=\"tooltiptext\">Result With Highest Confidence. Click on the Text to Sort</span></div></b></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>2nd</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>3rd</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>4th</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>5th</b></font></td>");
print("\t\t<td align=\"center\"><b><div class=\"tooltip\"><font color=\"Maroon\" size=\"2\">Matched</font><span class=\"tooltiptext\">TopK Result Matched with Ground Truth. Click on the Text to Sort</span></div></b></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Text-1</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Text-2</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Text-3</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Text-4</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Text-5</b></font></td>");
print("\t\t<td align=\"center\"><b><div class=\"tooltip\"><font color=\"Maroon\" size=\"2\">Conf-1</font><span class=\"tooltiptext\">Confidence of the Top Match. Click on the Text to Sort</span></div></b></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Conf-2</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Conf-3</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Conf-4</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"2\"><b>Conf-5</b></font></td>");
print("\t\t</tr>");
for i in range(numElements):
    print("\t\t<tr>");
    truth = int(resultDataBase[i][1]);
    labelTxt_1 = '';
    labelTxt_2 = '';
    labelTxt_3 = '';
    labelTxt_4 = '';
    labelTxt_5 = '';
    truthLabel = '';
    match = 0;
    label_1 = int(resultDataBase[i][2]);
    label_2 = int(resultDataBase[i][3]);
    label_3 = int(resultDataBase[i][4]);
    label_4 = int(resultDataBase[i][5]);
    label_5 = int(resultDataBase[i][6]);
    prob_1 = float(resultDataBase[i][7]);
    prob_2 = float(resultDataBase[i][8]);
    prob_3 = float(resultDataBase[i][9]);
    prob_4 = float(resultDataBase[i][10]);
    prob_5 = float(resultDataBase[i][11]);
    labelTxt_1 = (LabelLines[label_1].split(' ', 1)[1].rstrip('\n').lower());
    labelTxt_2 = (LabelLines[label_2].split(' ', 1)[1].rstrip('\n').lower());
    labelTxt_3 = (LabelLines[label_3].split(' ', 1)[1].rstrip('\n').lower());
    labelTxt_4 = (LabelLines[label_4].split(' ', 1)[1].rstrip('\n').lower());
    labelTxt_5 = (LabelLines[label_5].split(' ', 1)[1].rstrip('\n').lower());
    if(truth >= 0):
        if(truth == label_1): 
            match = 1;
        elif(truth == label_2):
            match = 2;
        elif(truth == label_3):
            match = 3;
        elif(truth == label_4):
            match = 4
        elif(truth == label_5):
            match = 5;
        truthLabel = (LabelLines[truth].split(' ', 1)[1].rstrip('\n').lower());
        print("\t\t<td height=\"17\" align=\"center\"><img id=\"myImg%d\" src=\"%s/%s\"alt=\"<b>GROUND TRUTH:</b> %s<br><b>CLASSIFIED AS:</b> %s\"width=\"30\" height=\"30\"></td>"
            %(i,dataFolder,resultDataBase[i][0],truthLabel,labelTxt_1));
        print("\t\t<td height=\"17\" align=\"center\"><a href=\"%s/%s\" target=\"_blank\">%s</a></td>"
            %(dataFolder,resultDataBase[i][0],resultDataBase[i][0]));
        print("\t\t<td align=\"left\">%s</td>"%(truthLabel));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(truth));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_1));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_2));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_3));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_4));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_5));
        if(match):
            print("\t\t<td align=\"center\"><font color=\"green\" size=\"2\"><b>%d</b></font></td>"%(match));
        else:
            print("\t\t<td align=\"center\"><font color=\"red\" size=\"2\"><b>%d</b></font></td>"%(match));
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_1));
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_2));
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_3));
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_4));
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_5));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_1));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_2));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_3));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_4));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_5));
    else:
        print("\t\t<td height=\"17\" align=\"center\"><img id=\"myImg%d\" src=\"%s/%s\"alt=\"<b>GROUND TRUTH:</b> %s<br><b>CLASSIFIED AS:</b> %s\"width=\"30\" height=\"30\"></td>"
            %(i,dataFolder,resultDataBase[i][0],truthLabel,labelTxt_1));
        print("\t\t<td height=\"17\" align=\"center\"><a href=\"%s/%s\" target=\"_blank\">%s</a></td>"
            %(dataFolder,resultDataBase[i][0],resultDataBase[i][0]));
        print("\t\t<td align=\"left\"><b>unknown</b></td>");
        print("\t\t<td align=\"center\">-1</td>");
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_1));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_2));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_3));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_4));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%d</font></td>"%(label_5));
        print("\t\t<td align=\"center\"><font color=\"blue\"><b>-1</b></font></td>");
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_1));
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_2));
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_3));
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_4));
        print("\t\t<td align=\"left\">%s</td>"%(labelTxt_5));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_1));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_2));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_3));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_4));
        print("\t\t<td align=\"center\"><font color=\"black\" size=\"2\">%.4f</font></td>"%(prob_5));
        
    print("\t\t</tr>");
    print("\t\t");
    print("\t\t<script>");
    print("\t\tvar modal = document.getElementById('myModal');");
    print("\t\tvar img1 = document.getElementById('myImg%d');"%(i));
                
    print("\t\tvar modalImg = document.getElementById(\"img01\");");
    print("\t\tvar captionText = document.getElementById(\"caption\");");
    print("\t\timg1.onclick = function(){ modal.style.display = \"block\"; modalImg.src = this.src; captionText.innerHTML = this.alt; }");
    print("\t\tvar span = document.getElementsByClassName(\"modal\")[0];");
    print("\t\tspan.onclick = function() { modal.style.display = \"none\"; }");
    print("\t\t</script>");
    print("\t\t");

print("</table>");


# Compare result summary
SummaryFileName = '';
FolderName = os.path.expanduser("~/.adatCompare")
if not os.path.exists(FolderName):
    os.makedirs(FolderName);

ModelFolderName = FolderName; ModelFolderName +="/"; ModelFolderName += modelName;
if not os.path.exists(ModelFolderName):
    os.makedirs(ModelFolderName);

SummaryFileName = FolderName; SummaryFileName += "/modelRunHistoryList.csv";


# write summary details into csv
if os.path.exists(SummaryFileName):
    sys.stdout = open(SummaryFileName,'a')
    print("%s, %s, %d, %d, %d"%(modelName,dataFolder,numElements,passCount,totalMismatch));
else:
    sys.stdout = open(SummaryFileName,'w')
    print("Model Name, Image DataBase, Number Of Images, Match, MisMatch");
    print("%s, %s, %d, %d, %d"%(modelName,dataFolder,numElements,passCount,totalMismatch));

sys.stdout.close()

# write into HTML
savedResultElements = 0;
with open(SummaryFileName) as savedResultFile:
    savedResultFileCSV = csv.reader(savedResultFile)
    next(savedResultFileCSV, None) # skip header
    savedResultDataBase = [r for r in savedResultFileCSV]
    savedResultElements = len(savedResultDataBase)
sys.stdout = orig_stdout
print "results saved in backup drive"

sys.stdout = open(toolKit_dir+'/index.html','a')
print("\t<!-- Compare ResultSummary -->");
print("<A NAME=\"table5\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Compare Results Summary</em></font></h1></A>");
print("\t<!-- Compare Graph Script -->");
print("\t<script type=\"text/javascript\" src=\"https://www.gstatic.com/charts/loader.js\"></script>");
print("\t<script type=\"text/javascript\">");
print("\t");

lineNumber = 0;
while lineNumber < savedResultElements:
    print("\tgoogle.charts.load('current', {'packages':['bar']});");
    print("\tgoogle.charts.setOnLoadCallback(drawChart_%d);"%(lineNumber));
    print("\tfunction drawChart_%d(){"%(lineNumber));
    print("\tvar data = google.visualization.arrayToDataTable([");
    print("\t['  '     ,  'Match'  , 'Mismatch'],");
    print("\t['Summary',   %d     , %d        ]"%(int(savedResultDataBase[lineNumber][3]),int(savedResultDataBase[lineNumber][4])));                        
    print("\t]);");
    print("\tvar options = { title: '%s Overall Result Summary', vAxis: { title: 'Images' }, width: 400, height: 400 };"%(savedResultDataBase[lineNumber][0]));
    print("\tvar chart = new google.charts.Bar(document.getElementById('Model_Stats_%d'));"%(lineNumber));
    print("\tchart.draw(data, google.charts.Bar.convertOptions(options));}");
    print("\t");

    lineNumber += 1;

print("\t");
# draw combined graph
print("\tgoogle.charts.load('current', {'packages':['bar']});");
print("\tgoogle.charts.setOnLoadCallback(drawChart_master);");
print("\tfunction drawChart_master(){");
print("\tvar data = google.visualization.arrayToDataTable([");
print("\t['Model'   ,'Match',   'Mismatch'],");
for i in range(savedResultElements):
    if(i != (lineNumber-1)):
        print("\t['%s'   ,%d ,   %d],"%(savedResultDataBase[i][0],int(savedResultDataBase[i][3]),int(savedResultDataBase[i][4])));
    else:
        print("\t['%s'   ,%d ,   %d]"%(savedResultDataBase[i][0],int(savedResultDataBase[i][3]),int(savedResultDataBase[i][4])));

print("\t]);");
print("\tvar options = { title: 'Overall Result Summary', vAxis: { title: 'Images' }, width: 800, height: 600, bar: { groupWidth: '30%' }, isStacked: true , series: { 0:{color:'green'},1:{color:'Indianred'} }};");
print("\tvar chart = new google.charts.Bar(document.getElementById('Model_Stats_master'));");
print("\tchart.draw(data, google.charts.Bar.convertOptions(options));}");
print("\t");
print("\t</script>");

# draw graph
print("\t");
print("\t");
print("\t<center><div id=\"Model_Stats_master\" style=\"border: 1px solid #ccc\"></div></center>");
print("\t");
print("\t<table align=\"center\" style=\"width: 90%\">");
print("\t<tr>");
i = 1;
while i <= savedResultElements:
    print("\t");
    print("\t<td><center><div id=\"Model_Stats_%d\" style=\"border: 1px solid #ccc\"></div></center></td>"%(i));
    if( i % 3 == 0):
        print("\t</tr>");
        print("\t<tr>");
    i += 1;

print("\t</tr>");
print("\t</table>");


#Model Score
def calculateHierarchyPenalty(truth, result, hierarchyDataBase):
  penaltyValue = 0;
  penaltyMultiplier = 0;
  truthHierarchy = hierarchyDataBase[truth];
  resultHierarchy = hierarchyDataBase[result];
  token_result = '';
  token_truth = '';
  previousTruth = 0;
  catCount = 0;

  while catCount < 6:
    token_truth = truthHierarchy[catCount];
    token_result = resultHierarchy[catCount];
    if((token_truth != '') and (token_truth == token_result)):
      previousTruth = 1;
    elif( (previousTruth == 1) and (token_truth == '' and token_result == '')):
      previousTruth = 1;
    else:
      previousTruth = 0;
      penaltyMultiplier += 1;
    catCount += 1;

  penaltyMultiplier = float(penaltyMultiplier - 1);
  penaltyValue = (0.2 * penaltyMultiplier);
  return penaltyValue;

print("\t<!-- Model Score -->");
hierarchyPenalty = [];
top5PassFail = [];
for i in xrange(100):
    hierarchyPenalty.append([])
    top5PassFail.append([])
    for j in xrange(5):
        hierarchyPenalty[i].append(0)
    for j in xrange(10):
        top5PassFail[i].append(0)

for i in xrange(100):
  top5PassFail[i][0] = top5PassFail[i][1] = 0;
  top5PassFail[i][2] = top5PassFail[i][3] = 0;
  top5PassFail[i][4] = top5PassFail[i][5] = 0;
  top5PassFail[i][6] = top5PassFail[i][7] = 0;
  top5PassFail[i][8] = top5PassFail[i][9] = 0;
  hierarchyPenalty[i][0] = hierarchyPenalty[i][1] = 0;
  hierarchyPenalty[i][2] = hierarchyPenalty[i][3] = 0;
  hierarchyPenalty[i][4] = 0;

for x in range(numElements):
  truth = int(resultDataBase[x][1]);
  if truth >= 0:
    match = 0;
    label_1 = int(resultDataBase[x][2]);
    label_2 = int(resultDataBase[x][3]);
    label_3 = int(resultDataBase[x][4]);
    label_4 = int(resultDataBase[x][5]);
    label_5 = int(resultDataBase[x][6]);
    prob_1 = float(resultDataBase[x][7]);
    prob_2 = float(resultDataBase[x][8]);
    prob_3 = float(resultDataBase[x][9]);
    prob_4 = float(resultDataBase[x][10]);
    prob_5 = float(resultDataBase[x][11]);

    if(truth == label_1):
      count = 0;
      f = 0;
      while f < 1:
        if((prob_1 <= (f + 0.01)) and prob_1 > f):
          top5PassFail[count][0]+=1;
        count+=1;
        f+=0.01
        
    elif(truth == label_2):
      count = 0;
      f = 0;
      while f < 1:
        if((prob_1 <= (f + 0.01)) and prob_1 > f):
          top5PassFail[count][1]+=1;
          hierarchyPenalty[count][0] += calculateHierarchyPenalty(truth,label_1,hierarchyDataBase);
        if((prob_2 <= (f + 0.01)) and prob_2 > f):
          top5PassFail[count][2]+=1;
        count+=1;
        f+=0.01;
    elif(truth == label_3):
      count = 0;
      f = 0;
      while f < 1:
        if((prob_1 <= (f + 0.01)) and prob_1 > f):
          top5PassFail[count][1]+=1;
          hierarchyPenalty[count][0] += calculateHierarchyPenalty(truth,label_1,hierarchyDataBase);
        if((prob_2 <= (f + 0.01)) and prob_2 > f):
          top5PassFail[count][3]+=1;
          hierarchyPenalty[count][1] += calculateHierarchyPenalty(truth,label_2,hierarchyDataBase);
        if((prob_3 <= (f + 0.01)) and prob_3 > f):
          top5PassFail[count][4]+=1;
        count+=1;
        f+=0.01;
    elif(truth == label_4):
      count = 0;
      f = 0;
      while f < 1:
        if((prob_1 <= (f + 0.01)) and prob_1 > f):
          top5PassFail[count][1]+=1;
          hierarchyPenalty[count][0] += calculateHierarchyPenalty(truth,label_1,hierarchyDataBase);
        if((prob_2 <= (f + 0.01)) and prob_2 > f):
          top5PassFail[count][3]+=1;
          hierarchyPenalty[count][1] += calculateHierarchyPenalty(truth,label_2,hierarchyDataBase);
        if((prob_3 <= (f + 0.01)) and prob_3 > f):
          top5PassFail[count][5]+=1;
          hierarchyPenalty[count][2] += calculateHierarchyPenalty(truth,label_3,hierarchyDataBase);
        if((prob_4 <= (f + 0.01)) and prob_4 > f):
          top5PassFail[count][6]+=1;
        count+=1;
        f+=0.01;
    elif(truth == label_5):
      count = 0;
      f = 0;
      while f < 1:
        if((prob_1 <= (f + 0.01)) and prob_1 > f):
          top5PassFail[count][1]+=1;
          hierarchyPenalty[count][0] += calculateHierarchyPenalty(truth,label_1,hierarchyDataBase);
        if((prob_2 <= (f + 0.01)) and prob_2 > f):
          top5PassFail[count][3]+=1;
          hierarchyPenalty[count][1] += calculateHierarchyPenalty(truth,label_2,hierarchyDataBase);
        if((prob_3 <= (f + 0.01)) and prob_3 > f):
          top5PassFail[count][5]+=1;
          hierarchyPenalty[count][2] += calculateHierarchyPenalty(truth,label_3,hierarchyDataBase);
        if((prob_4 <= (f + 0.01)) and prob_4 > f):
          top5PassFail[count][7]+=1;
          hierarchyPenalty[count][3] += calculateHierarchyPenalty(truth,label_4,hierarchyDataBase);
        if((prob_5 <= (f + 0.01)) and prob_5 > f):
          top5PassFail[count][8]+=1;
        count+=1;
        f+=0.01;
    else:
      count = 0;
      f = 0;
      while f < 1:
        if((prob_1 <= (f + 0.01)) and prob_1 > f):
          top5PassFail[count][1]+=1;
          hierarchyPenalty[count][0] += calculateHierarchyPenalty(truth,label_1,hierarchyDataBase);
        if((prob_2 <= (f + 0.01)) and prob_2 > f):
          top5PassFail[count][3]+=1;
          hierarchyPenalty[count][1] += calculateHierarchyPenalty(truth,label_2,hierarchyDataBase);
        if((prob_3 <= (f + 0.01)) and prob_3 > f):
          top5PassFail[count][5]+=1;
          hierarchyPenalty[count][2] += calculateHierarchyPenalty(truth,label_3,hierarchyDataBase);
        if((prob_4 <= (f + 0.01)) and prob_4 > f):
          top5PassFail[count][7]+=1;
          hierarchyPenalty[count][3] += calculateHierarchyPenalty(truth,label_4,hierarchyDataBase);
        if((prob_5 <= (f + 0.01)) and prob_5 > f):
          top5PassFail[count][9]+=1;
          hierarchyPenalty[count][4] += calculateHierarchyPenalty(truth,label_5,hierarchyDataBase);
        count+=1;
        f+=0.01;

print("<A NAME=\"table6\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>Model Score</em></font></h1></A>");
print("\t");
Top1PassScore = 0;
Top1FailScore = 0;
Top2PassScore = 0;
Top2FailScore = 0;
Top3PassScore = 0;
Top3FailScore = 0;
Top4PassScore = 0;
Top4FailScore = 0;
Top5PassScore = 0;
Top5FailScore = 0;
Top1HierarchyPenalty = 0;
Top2HierarchyPenalty = 0;
Top3HierarchyPenalty = 0;
Top4HierarchyPenalty = 0;
Top5HierarchyPenalty = 0;
confID=0.99;
i = 99;
while confID >= 0:
  Top1PassScore += confID * top5PassFail[i][0];
  Top1FailScore += confID * top5PassFail[i][1];
  Top2PassScore += confID * top5PassFail[i][2];
  Top2FailScore += confID * top5PassFail[i][3];
  Top3PassScore += confID * top5PassFail[i][4];
  Top3FailScore += confID * top5PassFail[i][5];
  Top4PassScore += confID * top5PassFail[i][6];
  Top4FailScore += confID * top5PassFail[i][7];
  Top5PassScore += confID * top5PassFail[i][8];
  Top5FailScore += confID * top5PassFail[i][9];

  Top1HierarchyPenalty += hierarchyPenalty[i][0];
  Top2HierarchyPenalty += hierarchyPenalty[i][1];
  Top3HierarchyPenalty += hierarchyPenalty[i][2];
  Top4HierarchyPenalty += hierarchyPenalty[i][3];
  Top5HierarchyPenalty += hierarchyPenalty[i][4];

  confID = confID - 0.01;
  i = i - 1;

# standard score result
Top1Score = float(top1Count);
ModelScoreTop1 = (Top1Score/netSummaryImages)*100;
Top2Score = float(top1Count + top2Count);
ModelScoreTop2 = (Top2Score/netSummaryImages)*100;
Top3Score = float(top1Count + top2Count + top3Count);
ModelScoreTop3 = (Top3Score/netSummaryImages)*100;
Top4Score = float(top1Count + top2Count + top3Count + top4Count);
ModelScoreTop4 = (Top4Score/netSummaryImages)*100;
Top5Score = float(top1Count + top2Count + top3Count + top4Count + top5Count);
ModelScoreTop5 = (Top5Score/netSummaryImages)*100;

print("<br><h1 align=\"center\"><font color=\"DarkSalmon\" size=\"4\">Standard Scoring</font></h1></A>");
print("\t<table align=\"center\" style=\"width: 40%\">");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>1st Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>2nd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>3rd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>4th Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>5th Match</b></font></td>");
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count + top4Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count + top4Count + top5Count));
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop1));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop2));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop3));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop4));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop5));
print("\t\t</tr>");
print("</table>");
print("\t");

#Method 1 result
Top1Score = float(Top1PassScore);
ModelScoreTop1 = (Top1Score/netSummaryImages)*100;
Top2Score = float((Top1PassScore + Top2PassScore));
ModelScoreTop2 = (Top2Score/netSummaryImages)*100;
Top3Score = float((Top1PassScore + Top2PassScore + Top3PassScore));
ModelScoreTop3 = (Top3Score/netSummaryImages)*100;
Top4Score = float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore));
ModelScoreTop4 = (Top4Score/netSummaryImages)*100;
Top5Score = float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore + Top5PassScore));
ModelScoreTop5 = (Top5Score/netSummaryImages)*100;

print("<br><h1 align=\"center\"><font color=\"DarkSalmon\" size=\"4\">Method 1 Scoring - Confidence Aware</font></h1></A>");
print("\t<table align=\"center\" style=\"width: 40%\">");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>1st Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>2nd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>3rd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>4th Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>5th Match</b></font></td>");
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count + top4Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count + top4Count + top5Count));
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop1));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop2));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop3));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop4));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop5));
print("\t\t</tr>");
print("</table>");
print("\t");

#Method 2 result
Top1Score = float(Top1PassScore - Top1FailScore);
ModelScoreTop1 = (Top1Score/netSummaryImages)*100;
Top2Score = float((Top1PassScore + Top2PassScore) - Top2FailScore);
ModelScoreTop2 = (Top2Score/netSummaryImages)*100;
Top3Score = float((Top1PassScore + Top2PassScore + Top3PassScore) - Top3FailScore);
ModelScoreTop3 = (Top3Score/netSummaryImages)*100;
Top4Score = float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore) - Top4FailScore);
ModelScoreTop4 = (Top4Score/netSummaryImages)*100;
Top5Score = float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore + Top5PassScore) - Top5FailScore);
ModelScoreTop5 = (Top5Score/netSummaryImages)*100;

print("<br><h1 align=\"center\"><font color=\"DarkSalmon\" size=\"4\">Method 2 Scoring - Error Aware</font></h1></A>");
print("\t<table align=\"center\" style=\"width: 40%\">");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>1st Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>2nd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>3rd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>4th Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>5th Match</b></font></td>");
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count + top4Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count + top4Count + top5Count));
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop1));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop2));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop3));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop4));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop5));
print("\t\t</tr>");
print("</table>");
print("\t");

#Method 3 result
Top1Score = float(Top1PassScore - (Top1FailScore + Top1HierarchyPenalty));
ModelScoreTop1 = (Top1Score/netSummaryImages)*100;
Top2Score = float((Top1PassScore + Top2PassScore) - (Top2FailScore + Top2HierarchyPenalty));
ModelScoreTop2 = (Top2Score/netSummaryImages)*100;
Top3Score = float((Top1PassScore + Top2PassScore + Top3PassScore) - (Top3FailScore + Top3HierarchyPenalty));
ModelScoreTop3 = (Top3Score/netSummaryImages)*100;
Top4Score = float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore) - (Top4FailScore + Top4HierarchyPenalty));
ModelScoreTop4 = (Top4Score/netSummaryImages)*100;
Top5Score = float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore + Top5PassScore) - (Top5FailScore + Top5HierarchyPenalty));
ModelScoreTop5 = (Top5Score/netSummaryImages)*100;

print("<br><h1 align=\"center\"><font color=\"DarkSalmon\" size=\"4\">Method 3 Scoring - Hierarchy Aware</font></h1></A>");
print("\t<table align=\"center\" style=\"width: 40%\">");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>1st Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>2nd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>3rd Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>4th Match</b></font></td>");
print("\t\t<td align=\"center\"><font color=\"Maroon\" size=\"3\"><b>5th Match</b></font></td>");
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count + top4Count));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%d</b></font></td>"%(top1Count + top2Count + top3Count + top4Count + top5Count));
print("\t\t</tr>");
print("\t<tr>");
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop1));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop2));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop3));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop4));
print("\t\t<td align=\"center\"><font color=\"black\" size=\"3\"><b>%.2f %%</b></font></td>"%(ModelScoreTop5));
print("\t\t</tr>");
print("</table>");
print("\t");

top5ModelScore = [];
for i in xrange(100):
  top5ModelScore.append([])
  for j in xrange(20):
    top5ModelScore[i].append(0)

for i in xrange(100):
  top5ModelScore[i][0] = 0;
  top5ModelScore[i][1] = 0; 
  top5ModelScore[i][2] = 0; 
  top5ModelScore[i][3] = 0; 
  top5ModelScore[i][4] = 0;
  top5ModelScore[i][5] = 0; 
  top5ModelScore[i][6] = 0; 
  top5ModelScore[i][7] = 0; 
  top5ModelScore[i][8] = 0; 
  top5ModelScore[i][9] = 0;
  top5ModelScore[i][10] = 0; 
  top5ModelScore[i][11] = 0; 
  top5ModelScore[i][12] = 0; 
  top5ModelScore[i][13] = 0; 
  top5ModelScore[i][10] = 0; 
  top5ModelScore[i][14] = 0;
  top5ModelScore[i][15] = 0; 
  top5ModelScore[i][16] = 0; 
  top5ModelScore[i][17] = 0; 
  top5ModelScore[i][18] = 0; 
  top5ModelScore[i][10] = 0; 
  top5ModelScore[i][19] = 0;


standardPassTop1 = 0;
standardPassTop2 = 0;
standardPassTop3 = 0;
standardPassTop4 = 0;
standardPassTop5 = 0;

Top1PassScore = 0; 
Top1FailScore = 0; 
Top2PassScore = 0; 
Top2FailScore = 0; 
Top3PassScore = 0;
Top3FailScore = 0; 
Top4PassScore = 0; 
Top4FailScore = 0; 
Top5PassScore = 0; 
Top5FailScore = 0;
Top1HierarchyPenalty = 0;
Top2HierarchyPenalty = 0;
Top3HierarchyPenalty = 0;
Top4HierarchyPenalty = 0;
Top5HierarchyPenalty = 0;

confID = 0.99;
i = 99;
while i >= 0:
  Top1PassScore += confID * topKPassFail[i][0];
  Top1FailScore += confID * topKPassFail[i][1];
  Top2PassScore += confID * top5PassFail[i][2];
  Top2FailScore += confID * top5PassFail[i][3];
  Top3PassScore += confID * top5PassFail[i][4];
  Top3FailScore += confID * top5PassFail[i][5];
  Top4PassScore += confID * top5PassFail[i][6];
  Top4FailScore += confID * top5PassFail[i][7];
  Top5PassScore += confID * top5PassFail[i][8];
  Top5FailScore += confID * top5PassFail[i][9];

  Top1HierarchyPenalty += hierarchyPenalty[i][0];
  Top2HierarchyPenalty += hierarchyPenalty[i][1];
  Top3HierarchyPenalty += hierarchyPenalty[i][2];
  Top4HierarchyPenalty += hierarchyPenalty[i][3];
  Top5HierarchyPenalty += hierarchyPenalty[i][4];


  # method 1
  top5ModelScore[i][1] = (float(Top1PassScore)/netSummaryImages)*100;
  top5ModelScore[i][3] = (float(Top1PassScore + Top2PassScore)/netSummaryImages)*100;
  top5ModelScore[i][5] = (float(Top1PassScore + Top2PassScore + Top3PassScore)/netSummaryImages)*100;
  top5ModelScore[i][7] = (float(Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore)/netSummaryImages)*100;
  top5ModelScore[i][9] = (float(Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore + Top5PassScore)/netSummaryImages)*100;

  # method 2
  top5ModelScore[i][0] = (float(Top1PassScore - Top1FailScore)/netSummaryImages)*100;
  top5ModelScore[i][4] = (float((Top1PassScore + Top2PassScore + Top3PassScore) - Top3FailScore)/netSummaryImages)*100;
  top5ModelScore[i][6] = (float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore) - Top4FailScore)/netSummaryImages)*100;
  top5ModelScore[i][8] = (float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore + Top5PassScore) - Top5FailScore)/netSummaryImages)*100;

  # method 3
  top5ModelScore[i][15] = (float(Top1PassScore - (Top1FailScore + Top1HierarchyPenalty))/netSummaryImages)*100;
  top5ModelScore[i][16] = (float((Top1PassScore + Top2PassScore) - (Top2FailScore + Top2HierarchyPenalty))/netSummaryImages)*100;
  top5ModelScore[i][17] = (float((Top1PassScore + Top2PassScore + Top3PassScore) - (Top3FailScore + Top3HierarchyPenalty))/netSummaryImages)*100;
  top5ModelScore[i][18] = (float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore) - (Top4FailScore + Top4HierarchyPenalty))/netSummaryImages)*100;
  top5ModelScore[i][19] = (float((Top1PassScore + Top2PassScore + Top3PassScore + Top4PassScore + Top5PassScore) - (Top5FailScore + Top5HierarchyPenalty))/netSummaryImages)*100;

  # standard method
  standardPassTop1 += float(topKPassFail[i][0]);
  standardPassTop2 += float(topKPassFail[i][0] + top5PassFail[i][2]);
  standardPassTop3 += float(topKPassFail[i][0] + top5PassFail[i][2] + top5PassFail[i][4]);
  standardPassTop4 += float(topKPassFail[i][0] + top5PassFail[i][2] + top5PassFail[i][4] + top5PassFail[i][6]);
  standardPassTop5 += float(topKPassFail[i][0] + top5PassFail[i][2] + top5PassFail[i][4] + top5PassFail[i][6] + top5PassFail[i][8]);
  top5ModelScore[i][10] = (standardPassTop1/netSummaryImages)*100;
  top5ModelScore[i][11] = (standardPassTop2/netSummaryImages)*100;
  top5ModelScore[i][12] = (standardPassTop3/netSummaryImages)*100;
  top5ModelScore[i][13] = (standardPassTop4/netSummaryImages)*100;
  top5ModelScore[i][14] = (standardPassTop5/netSummaryImages)*100;

  confID = confID - 0.01;
  i = i - 1;

print("\t<script type=\"text/javascript\">");
print("\t");
# Top 1 Score thresholds
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Top1ScoreGraph);");
print("\tfunction Top1ScoreGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Standard');");
print("\tdata.addColumn('number', 'Method 1');");
print("\tdata.addColumn('number', 'Method 2');");
print("\tdata.addColumn('number', 'Method 3');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f,  %.4f,   %.4f,  %.4f,    %.4f]"%(fVal,top5ModelScore[i][10],top5ModelScore[i][1],top5ModelScore[i][0],top5ModelScore[i][15]));
  else:
    print("\t[%.2f,  %.4f,   %.4f,   %.4f,    %.4f],"%(fVal,top5ModelScore[i][10],top5ModelScore[i][1],top5ModelScore[i][0],top5ModelScore[i][15]));

  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Model Score Top 1', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('Top1_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
# Top 2 Score thresholds
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Top2ScoreGraph);");
print("\tfunction Top2ScoreGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Standard');");
print("\tdata.addColumn('number', 'Method 1');");
print("\tdata.addColumn('number', 'Method 2');");
print("\tdata.addColumn('number', 'Method 3');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f,  %.4f,   %.4f,   %.4f,    %.4f]"%(fVal,top5ModelScore[i][11],top5ModelScore[i][3],top5ModelScore[i][2],top5ModelScore[i][16]));
  else:
    print("\t[%.2f,  %.4f,   %.4f,   %.4f,    %.4f],"%(fVal,top5ModelScore[i][11],top5ModelScore[i][3],top5ModelScore[i][2],top5ModelScore[i][16]));
  
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Model Score Top 2', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('Top2_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
# Top 3 Score thresholds
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Top3ScoreGraph);");
print("\tfunction Top3ScoreGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Standard');");
print("\tdata.addColumn('number', 'Method 1');");
print("\tdata.addColumn('number', 'Method 2');");
print("\tdata.addColumn('number', 'Method 3');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f,    %.4f, %.4f, %.4f,    %.4f]"%(fVal,top5ModelScore[i][12],top5ModelScore[i][5],top5ModelScore[i][4],top5ModelScore[i][17]));
  else:
    print("\t[%.2f,    %.4f, %.4f, %.4f,    %.4f],"%(fVal,top5ModelScore[i][12],top5ModelScore[i][5],top5ModelScore[i][4],top5ModelScore[i][17]));
  
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Model Score Top 3', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('Top3_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
# Top 4 Score thresholds
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Top4ScoreGraph);");
print("\tfunction Top4ScoreGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Standard');");
print("\tdata.addColumn('number', 'Method 1');");
print("\tdata.addColumn('number', 'Method 2');");
print("\tdata.addColumn('number', 'Method 3');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f, %.4f,   %.4f,   %.4f,    %.4f]"%(fVal,top5ModelScore[i][13],top5ModelScore[i][7],top5ModelScore[i][6],top5ModelScore[i][18]));
  else:
    print("\t[%.2f, %.4f,   %.4f,   %.4f,    %.4f],"%(fVal,top5ModelScore[i][13],top5ModelScore[i][7],top5ModelScore[i][6],top5ModelScore[i][18]));
        
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Model Score Top 4', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('Top4_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
# Top 5 Score thresholds
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Top5ScoreGraph);");
print("\tfunction Top5ScoreGraph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Standard');");
print("\tdata.addColumn('number', 'Method 1');");
print("\tdata.addColumn('number', 'Method 2');");
print("\tdata.addColumn('number', 'Method 3');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f, %.4f,   %.4f,   %.4f,    %.4f]"%(fVal,top5ModelScore[i][14],top5ModelScore[i][9],top5ModelScore[i][8],top5ModelScore[i][19]));
  else:
    print("\t[%.2f, %.4f,   %.4f,   %.4f,    %.4f],"%(fVal,top5ModelScore[i][14],top5ModelScore[i][9],top5ModelScore[i][8],top5ModelScore[i][19]));
        
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Model Score Top 5', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('Top5_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
# Standard Score Model
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(StandardTop5Graph);");
print("\tfunction StandardTop5Graph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Top 1');");
print("\tdata.addColumn('number', 'Top 2');");
print("\tdata.addColumn('number', 'Top 3');");
print("\tdata.addColumn('number', 'Top 4');");
print("\tdata.addColumn('number', 'Top 5');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f, %.4f,    %.4f,   %.4f,   %.4f,    %.4f]"%(fVal,top5ModelScore[i][10],top5ModelScore[i][11],top5ModelScore[i][12],top5ModelScore[i][13],top5ModelScore[i][14]));
  else:
    print("\t[%.2f, %.4f,    %.4f,   %.4f,   %.4f,    %.4f],"%(fVal,top5ModelScore[i][10],top5ModelScore[i][11],top5ModelScore[i][12],top5ModelScore[i][13],top5ModelScore[i][14]));
        
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Standard Scoring Method', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('standard_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
# method 1 Score Model
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Method1Top5Graph);");
print("\tfunction Method1Top5Graph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Top 1');");
print("\tdata.addColumn('number', 'Top 2');");
print("\tdata.addColumn('number', 'Top 3');");
print("\tdata.addColumn('number', 'Top 4');");
print("\tdata.addColumn('number', 'Top 5');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f, %.4f,    %.4f,   %.4f,   %.4f,    %.4f]"%(fVal,top5ModelScore[i][1],top5ModelScore[i][3],top5ModelScore[i][5],top5ModelScore[i][7],top5ModelScore[i][9]));
  else:
    print("\t[%.2f, %.4f,    %.4f,   %.4f,   %.4f,    %.4f],"%(fVal,top5ModelScore[i][1],top5ModelScore[i][3],top5ModelScore[i][5],top5ModelScore[i][7],top5ModelScore[i][9]));
        
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Method 1 Scoring', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('method_1_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
# method 2 Score Model
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Method2Top5Graph);");
print("\tfunction Method2Top5Graph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Top 1');");
print("\tdata.addColumn('number', 'Top 2');");
print("\tdata.addColumn('number', 'Top 3');");
print("\tdata.addColumn('number', 'Top 4');");
print("\tdata.addColumn('number', 'Top 5');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f, %.4f,    %.4f,   %.4f,   %.4f,    %.4f]"%(fVal,top5ModelScore[i][0],top5ModelScore[i][2],top5ModelScore[i][4],top5ModelScore[i][6],top5ModelScore[i][8]));
  else:
    print("\t[%.2f, %.4f,    %.4f,   %.4f,   %.4f,    %.4f],"%(fVal,top5ModelScore[i][0],top5ModelScore[i][2],top5ModelScore[i][4],top5ModelScore[i][6],top5ModelScore[i][8]));
        
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Method 2 Scoring', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('method_2_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
# method 3 Score Model
print("\tgoogle.charts.load('current', {packages: ['corechart', 'line']});");
print("\tgoogle.charts.setOnLoadCallback(Method3Top5Graph);");
print("\tfunction Method3Top5Graph() {");
print("\tvar data = new google.visualization.DataTable();");
print("\tdata.addColumn('number', 'X');");
print("\tdata.addColumn('number', 'Top 1');");
print("\tdata.addColumn('number', 'Top 2');");
print("\tdata.addColumn('number', 'Top 3');");
print("\tdata.addColumn('number', 'Top 4');");
print("\tdata.addColumn('number', 'Top 5');");
print("\tdata.addRows([");
print("\t[1, 0, 0, 0, 0, 0],");
fVal=0.99;
i = 99;
while i >= 0:
  if(i == 0):
    print("\t[%.2f, %.4f,    %.4f,   %.4f,   %.4f,    %.4f]"%(fVal,top5ModelScore[i][15],top5ModelScore[i][16],top5ModelScore[i][17],top5ModelScore[i][18],top5ModelScore[i][19]));
  else:
    print("\t[%.2f, %.4f,    %.4f,   %.4f,   %.4f,    %.4f],"%(fVal,top5ModelScore[i][15],top5ModelScore[i][16],top5ModelScore[i][17],top5ModelScore[i][18],top5ModelScore[i][19]));
        
  fVal=fVal-0.01;
  i = i - 1;
print("\t]);");
print("\tvar options = {  title:'Method 3 Scoring', hAxis: { title: 'Confidence', direction: '-1' }, vAxis: {title: 'Score Percentage'}, series: { 0.01: {curveType: 'function'} }, width:750, height:400 };");
print("\tvar chart = new google.visualization.LineChart(document.getElementById('method_3_model_score_chart'));");
print("\tchart.draw(data, options);}");
print("\t");
print("\t");
print("\t</script>");
print("\t");
print("\t");
print("\t<table align=\"center\" style=\"width: 90%\">");
print("\t<tr>");
print("\t <td><center><div id=\"Top1_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t <td><center><div id=\"Top2_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t</tr>");
print("\t<tr>");
print("\t <td><center><div id=\"Top3_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t <td><center><div id=\"Top4_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t</tr>");
print("\t<tr>");
print("\t <td><center><div id=\"Top5_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t <td><center><div id=\"standard_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t</tr>");
print("\t<tr>");
print("\t <td><center><div id=\"method_1_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t <td><center><div id=\"method_2_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t</tr>");
print("\t<tr>");
print("\t <td><center><div id=\"method_3_model_score_chart\" style=\"border: 0px solid #ccc\" ></div></center></td>");
print("\t <td><center></center></td>");
print("\t</tr>");
print("\t</table>");
print("\t");

# HELP
print ("\t<!-- HELP -->");
print ("<A NAME=\"table7\"><h1 align=\"center\"><font color=\"DodgerBlue\" size=\"6\"><br><br><br><em>HELP</em></font></h1></A>");
print ("\t");
print ("\t<table align=\"center\" style=\"width: 50%\">");
print ("\t<tr><td>");
print ("\t<h1 align=\"center\">AMD Neural Net ToolKit</h1>");
print ("\t</td></tr><tr><td>");
print ("\t<p>AMD Neural Net ToolKit is a comprehensive set of help tools for neural net creation, development, training and");
print ("\tdeployment. The ToolKit provides you with help tools to design, develop, quantize, prune, retrain, and infer your neural");
print ("\tnetwork work in any framework. The ToolKit is designed help you deploy your work to any AMD or 3rd party hardware, from ");
print ("\tembedded to servers.</p>");
print ("\t<p>AMD Neural Net ToolKit provides you with tools for accomplishing your tasks throughout the whole neural net life-cycle,");
print ("\tfrom creating a model to deploying them for your target platforms.</p>");
print ("\t<h2 >List of Features Available in this release</h2>");
print ("\t<ul>");
print ("\t<li>Overall Summary</li>");
print ("\t<li>Graphs</li>");
print ("\t<li>Hierarchy</li>");
print ("\t<li>Labels</li>");
print ("\t<li>Image Results</li>");
print ("\t<li>Compare</li>");
print ("\t<li>Help</li>");
print ("\t</ul>");
print ("\t<h3 >Overall Summary</h3>");
print ("\t<p>This section summarizes the results for the current session, with information on the dataset and the model.");
print ("\tThe section classifies the dataset into images with or without ground truth and only considers the images with ground truth ");
print ("\tfor analysis to avoid skewing the results.</p>");
print ("\t<p>The summary calculates all the metrics to evaluate the current run session, helps evaluate the quality of the data set,");
print ("\taccuracy of the current version of the model and links all the high level result to individual images to help the user to ");
print ("\tquickly analyze and detect if there are any problems.</p>");
print ("\t<p>The summary also timestamps the results to avoid confusion with different iterations.</p>");
print ("\t<h3>Graphs</h3>");
print ("\t<p>The graph section allows the user to visualize the dataset and model accurately. The graphs can help detect any");
print ("\tanomalies with the data or the model from a higher level. The graphs can be saved or shared with others.</p>");
print ("\t<h3 >Hierarchy</h3>");
print ("\t<p>This section has AMD proprietary hierarchical result analysis. Please contact us to get more information.</p>");
print ("\t<h3 >Labels</h3>");
print ("\t<p>Label section is the summary of all the classes the model has been trained to detect. The Label Summary presents the");
print ("\thighlights of all the classes of images available in the database. The summary reports if the classes are found or not ");
print ("\tfound.</p>");
print ("\t<p>Click on any of the label description and zoom into all the images from that class in the database.</p>");
print ("\t<h3 >Image Results</h3>");
print ("\t<p>The Image results has all the low level information about each of the individual images in the database. It reports on ");
print ("\tthe results obtained for the image in the session and allows quick view of the image.</p>");
print ("\t<h3 >Compare</h3>");
print ("\t<p>This section compares the results of a database or the model between different sessions. If the database was tested with");
print ("\tdifferent models, this section reports and compares results among them.</p>");
print ("\t</td></tr>");
print ("\t</table>");
print ("\t<br><br><br>");
#TBD: symbol
print ("\t\t<div class=\"footer\"> <p>2018 Advanced Micro Devices, Inc</p></div>");
print ("\t");
print ("\n</body>");
print ("\n</html>");

sys.stdout = orig_stdout
print "index.html generated"
exit (0)
