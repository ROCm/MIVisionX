# rocAL Video Unit Tests
This application can be used to verify the functionality of the video API offered by rocAL.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, version - `18.04` / `20.04`
* rocAL library (Part of the MIVisionX toolkit)
* [OpenCV 4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* [FFmpeg n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Radeon Performance Primitives (RPP)

### Running the application
Executing the below command will build and run the application for the specified test case.

The following test cases are supported in this unittest:
1. Video Reader
2. Video Reader Resize (reader followed by resize augmentation)
3. Sequence Reader

The other arguments can be modified in the test script.
  ````
./testScript.sh <input> <test-case>
  ````
The outputs will be dumped inside the build/output_frames folder

NOTE:

* Inputs for cases 1 and 2 - Video file / folder containing videos
* Input for case 3 - Folder containing sequence of images [sample folder](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX-data/tree/main/rocal_data/video_and_sequence_samples/sequence)
