# AMD OpenCV Extension

The AMD OpenCV (vx_opencv) is an OpenVX module that implements a mechanism to access OpenCV functionality as OpenVX kernels. These kernels can be accessed from within OpenVX framework using OpenVX API call [vxLoadKernels](https://www.khronos.org/registry/vx/specs/1.0.1/html/da/d83/group__group__user__kernels.html#gae00b6343fbb0126e3bf0f587b09393a3)(context, "vx_opencv").

## List of OpenCV-interop kernels

The following is a list of OpenCV functions that have been included in the vx_opencv module.

    bilateralFilter             org.opencv.bilateralfilter
    blur                        org.opencv.blur
    boxfilter                   org.opencv.boxfilter
    buildPyramid                org.opencv.buildpyramid
    Dilate                      org.opencv.dilate 
    Erode                       org.opencv.erode 
    filter2D                    org.opencv.filter2d
    GaussianBlur                org.opencv.gaussianblur
    MedianBlur                  org.opencv.medianblur
    morphologyEx                org.opencv.morphologyex
    Laplacian                   org.opencv.laplacian
    pyrDown                     org.opencv.pyrdown
    pyrUp                       org.opencv.pyrup
    sepFilter2D                 org.opencv.sepfilter2d
    Sobel                       org.opencv.sobel
    Scharr                      org.opencv.scharr
    FAST                        org.opencv.fast
    MSER                        org.opencv.mser_detect 
    ORB                         org.opencv.orb_detect
    ORB_Compute                 org.opencv.orb_compute   
    BRISK                       org.opencv.brisk_detect
    BRISK_Compute               org.opencv.brisk_compute 
    SimpleBlobDetector          org.opencv.simple_blob_detect                   
    SimpleBlobDetector_Init     org.opencv.simple_blob_detect_initialize 
    SIFT_Detect                 org.opencv.sift_detect 
    SIFT_Compute                org.opencv.sift_compute                         
    SURF_Detect                 org.opencv.surf_detect
    SURF_Compute                org.opencv.surf_compute
    STAR_FEATURE_Detector       org.opencv.star_detect  
    Canny                       org.opencv.canny  
    GoodFeature_Detector        org.opencv.good_features_to_track
    buildOpticalFlowPyramid     org.opencv.buildopticalflowpyramid
    DistanceTransform           org.opencv.distancetransform                                           
    Convert_Scale_Abs           org.opencv.convertscaleabs                      
    addWeighted                 org.opencv.addweighted                          
    Transpose                   org.opencv.transpose                            
    Resize                      org.opencv.resize
    AdaptiveThreshold           org.opencv.adaptivethreshold                                                          
    Threshold                   org.opencv.threshold  
    cvtcolor                    org.opencv.cvtcolor                          
    Flip                        org.opencv.flip 
    fastNlMeansDenoising        org.opencv.fastnlmeansdenoising
    fastNlMeansDenoisingColored org.opencv.fastnlmeansdenoisingcolored 
    AbsDiff                     org.opencv.absdiff                              
    Compare                     org.opencv.compare
    bitwise_and                 org.opencv.bitwise_and
    bitwise_not                 org.opencv.bitwise_not
    bitwise_or                  org.opencv.bitwise_or
    bitwise_xor                 org.opencv.bitwise_xor
    Add                         org.opencv.add 
    Subtract                    org.opencv.subtract
    Multiply                    org.opencv.multiply    
    Divide                      org.opencv.divide  
    WarpAffine                  org.opencv.warpaffine 
    WarpPerspective             org.opencv.warpperspective  

**NOTE** - For list of OpenVX API calls for OpenCV-interop refer include/[vx_ext_opencv.h](include/vx_ext_opencv.h)

## Build Instructions

### Pre-requisites

* AMD OpenVX library
* CMake 3.0 or later
* OpenCV [3.4](https://github.com/opencv/opencv/releases/tag/3.4.0) `with`/`without` **contrib**
    - OpenCV_DIR environment variable should point to OpenCV/build folder

### Build using `Visual Studio 2017` on 64-bit Windows 10

* Use amd_openvx_extensions/amd_opencv/amd_opencv.sln to build for x64 platform

### Build using CMake on Linux

* Use CMake to configure and generate Makefile
