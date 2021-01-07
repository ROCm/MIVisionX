#!/bin/bash

rm -rvf rocprof_profiling_outputs
rm -rvf rocprof_profiling_temp
mkdir rocprof_profiling_outputs
mkdir rocprof_profiling_temp

rm -rvf build*
mkdir build
cd build
cmake ..
make
cd ..

width=16
height=16

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Profiling all HIP Test Cases..."
echo "##########################################################################################"
passed_hip=0
failed_hip=0
na_hip=0
passed_list_hip=""
failed_list_hip=""
na_list_hip=""
for ((case_num=1;case_num<244;case_num++))
do
printf "\n\n\n\n"
echo "--------------------------------"
printf "Profiling case $case_num...\n"
echo "--------------------------------"
printf "\n./hipvx_sample $case_num $width $height 0\n"
mkdir rocprof_profiling_outputs/case_$case_num
mkdir rocprof_profiling_temp/case_$case_num
rocprof -i "/media/abishek/rocprofiler/test/tool/input.xml" -o "rocprof_profiling_outputs/case_$case_num/output_case_$case_num.csv" -d "rocprof_profiling_temp/case_$case_num" -t "rocprof_profiling_temp/case_$case_num" --timestamp on --basenames on --stats --verbose ./build/hipvx_sample $case_num $width $height 1
out=$?
if [[ "$out" -eq 255 ]]
then
    let "failed_hip += 1"
    failed_list_hip="$failed_list_hip $case_num"
elif [[ "$out" -eq 0 ]]
then
    let "na_hip += 1"
    na_list_hip="$na_list_hip $case_num"
elif [[ "$out" -eq 1 ]]
then
    let "passed_hip += 1"
    passed_list_hip="$passed_list_hip $case_num"
else
    let "na_hip += 1"
    na_list_hip="$na_list_hip $case_num"
fi
done

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "HIP PROFILING RESULT::"
echo "##########################################################################################"
total_tested_hip=$((passed_hip + failed_hip + na_hip))
total_needed_to_pass_hip="$(($total_tested_hip-$na_hip))"
printf "\nHIP number of cases passed = $passed_hip"
printf "\nHIP number of cases needed to pass = $total_needed_to_pass_hip"
printf "\nHIP number of cases failed = $failed_hip"
printf "\nHIP number of cases not applicable = $na_hip"
printf "\nHIP total number of cases tested = $total_tested_hip"
printf "\n\n______________________________________________\n\n"
printf "\nHIP list of cases passed = $passed_list_hip"
printf "\nHIP list of cases failed = $failed_list_hip"
printf "\nHIP list of cases not applicable = $na_list_hip"
printf "\n"