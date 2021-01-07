# Set results directory as hip_samples/hipvx_unittests/rocprof_profiling_outputs
RESULTS_DIR = "rocprof_profiling_outputs"

import os

def main():
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE = RESULTS_DIR + "/consolidated_results.stats.csv"
    new_file = open(CONSOLIDATED_FILE,'w')
    new_file.write('"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')
    for case_num in range(1,245,1):
        CASE_RESULTS_DIR = RESULTS_DIR + "/case_" + str(case_num)
        print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)
        CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case_" + str(case_num) + ".stats.csv"
        print("CASE_FILE_PATH = " + CASE_FILE_PATH)
        try:
            case_file = open(CASE_FILE_PATH,'r')
            for line in case_file:
                if line.startswith('"Hip'):
                    new_file.write(line)
            case_file.close()
        except:
            continue

    new_file.close()
    os.system('chown $USER:$USER rocprof_profiling_outputs/consolidated_results.stats.csv')
    # os.system('chown $USER:$USER rocprof_profiling_outputs/consolidated_results.stats.csv')


main()