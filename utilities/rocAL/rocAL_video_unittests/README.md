# rocAL Video Unit Tests
This application can be used to verify the video functionality of the API offered by rocAL.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library (Part of the MIVisionX toolkit)
* [OpenCV 4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* [FFmpeg 4.4.2](https://git.ffmpeg.org/gitweb/ffmpeg.git/blob/refs/heads/release/4.4:/RELEASE_NOTES)
* Radeon Performance Primitives (RPP)

### Running the application
Executing the below command will build and run the application for the specific test case.

The other arguments can be modified in the test script.
  ````
./testScript.sh <input> <test_case>
  ````

## Test Cases
The following test cases are supported in this unittest:
1. Video Reader
2. Video Reader Resize (reader followed by resize augmentation)
3. Sequence Reader

NOTE:

* Inputs for cases 1 and 2 - Video file / folder containing videos
* Input for case 3 - Folder containing sequence of images [sample folder](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX-data/tree/main/rocal_data/video_and_sequence_samples/sequence)
