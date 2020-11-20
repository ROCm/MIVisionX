#!/bin/bash

rm -rvf build*
mkdir build
cd build
cmake ..
make

width=16
height=16





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all HOST Test Cases..."
echo "##########################################################################################"
passed_host=0
failed_host=0
na_host=0
passed_list_host=""
failed_list_host=""
na_list_host=""
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
    failed_list_host="$failed_list_host $case"
elif [[ "$out" -eq 0 ]]
then
    let "na_host += 1"
    na_list_host="$na_list_host $case"
elif [[ "$out" -eq 1 ]]
then
    let "passed_host += 1"
    passed_list_host="$passed_list_host $case"
else
    let "na_host += 1"
    na_list_host="$na_list_host $case"
fi
done





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all HIP Test Cases..."
echo "##########################################################################################"
passed_hip=0
failed_hip=0
na_hip=0
passed_list_hip=""
failed_list_hip=""
na_list_hip=""
for ((case=1;case<100;case++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Running case $case...\n"
echo "--------------------------------"
printf "\n./hipvx_sample $case $width $height 1\n"
./hipvx_sample $case $width $height 1
out=$?
if [[ "$out" -eq 255 ]]
then
    let "failed_hip += 1"
    failed_list_hip="$failed_list_hip $case"
elif [[ "$out" -eq 0 ]]
then
    let "na_hip += 1"
    na_list_hip="$na_list_hip $case"
elif [[ "$out" -eq 1 ]]
then
    let "passed_hip += 1"
    passed_list_hip="$passed_list_hip $case"
else
    let "na_hip += 1"
    na_list_hip="$na_list_hip $case"
fi
done





printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "RESULT::"
echo "##########################################################################################"
total_tested_host=$((passed_host + failed_host + na_host))
printf "\nHOST number of cases passed = $passed_host"
printf "\nHOST number of cases failed = $failed_host"
printf "\nHOST number of cases not applicable = $na_host"
printf "\nHOST total number of cases tested = $total_tested_host"
printf "\n\n"
total_tested_hip=$((passed_hip + failed_hip + na_hip))
printf "\nHIP number of cases passed = $passed_hip"
printf "\nHIP number of cases failed = $failed_hip"
printf "\nHIP number of cases not applicable = $na_hip"
printf "\nHIP total number of cases tested = $total_tested_hip"
printf "\n\n______________________________________________\n\n"
printf "\nHOST list of cases passed = $passed_list_host"
printf "\nHOST list of cases failed = $failed_list_host"
printf "\nHOST list of cases not applicable = $na_list_host"
printf "\n\n"
printf "\nHIP list of cases passed = $passed_list_hip"
printf "\nHIP list of cases failed = $failed_list_hip"
printf "\nHIP list of cases not applicable = $na_list_hip"
printf "\n"