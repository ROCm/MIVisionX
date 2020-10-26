#!/bin/bash

rm -rvf build*
mkdir build
cd build
cmake ..
make

width=5
height=5





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all HOST Test Cases..."
echo "##########################################################################################"
passed_host=0
failed_host=0
na_host=0
for ((case=1;case<100;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running case $case...\n"
echo "--------------------------------"
printf "\n./hipvx_sample $case $width $height 0\n"
./hipvx_sample $case $width $height 0
out=$?
if [[ "$out" -eq 255 ]]
then
    let "failed_host += 1"
elif [[ "$out" -eq 0 ]]
then
    let "na_host += 1"
elif [[ "$out" -eq 1 ]]
then
    let "passed_host += 1"
else
    let "na_host += 1"
fi
done





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all HIP Test Cases..."
echo "##########################################################################################"
passed_hip=0
failed_hip=0
na_hip=0
for ((case=1;case<100;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running case $case...\n"
echo "--------------------------------"
printf "\n./hipvx_sample $case $width $height 0\n"
./hipvx_sample $case $width $height 1
out=$?
if [[ "$out" -eq 255 ]]
then
    let "failed_hip += 1"
elif [[ "$out" -eq 0 ]]
then
    let "na_hip += 1"
elif [[ "$out" -eq 1 ]]
then
    let "passed_hip += 1"
else
    let "na_hip += 1"
fi
done





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "RESULT::"
echo "##########################################################################################"
total_tested_host=$((passed_host + failed_host + na_host))
printf "\nHOST cases passed = $passed_host"
printf "\nHOST cases failed = $failed_host"
printf "\nHOST cases not applicable = $na_host"
printf "\nHOST total cases tested = $total_tested_host"
printf "\n\n"
total_tested_hip=$((passed_hip + failed_hip + na_hip))
printf "\nHIP cases passed = $passed_hip"
printf "\nHIP cases failed = $failed_hip"
printf "\nHIP cases not applicable = $na_hip"
printf "\nHIP total cases tested = $total_tested_hip"
printf "\n"