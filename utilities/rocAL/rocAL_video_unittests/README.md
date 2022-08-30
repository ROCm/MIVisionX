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
  ````
./testScript.sh <input video file/folder> <test_case>
  ````
