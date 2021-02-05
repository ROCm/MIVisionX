# OpenVX Vision Function Tests

## Script to run vision tests

```
python runVisionTests.py --help
```


usage: 

```
runVisionTests.py [--runvx_directory RUNVX_DIRECTORY] -- required
                  [--hardware_mode HARDWARE_MODE]
                  [--list_tests LIST_TESTS]
                  [--test_filter TEST_FILTER]
                  [--num_frames NUM_FRAMES]
                  [--functionality FUNCTIONALITY]

Arguments:
  -h, --help        show this help message and exit
  --runvx_directory RunVX Executable Directory - required
  --hardware_mode   OpenVX Vision Function Target - optional (default:CPU [options:CPU/GPU])
  --list_tests      List Vision Performance Tests - optional (default:no [options:no/yes])
  --test_filter     Vision Performance Test Filter - optional (default:0 [range:1 - N])
  --num_frames      Run Test for X number of frames - optional (default:1000 [range:1 - N])
  --functionality   Vision Functionality Tests Enabled - optional (default:yes [options:no/yes])
```

## Vision Functionality Test GDFs

```
01_absDiff
02_accumulate
03_accumulateSquared
04_accumulateWeighted
05_add
06_and
07_box
08_canny
09_channelCombine
10_channelExtract
11_colorConvert
12_convertDepth
13_convolve
14_dilate
15_equalizeHistogram
16_erode
17_fastCorners
18_gaussian
19_harrisCorners
20_halfScaleGaussian
21_histogram
22_integralImage
23_magnitude
24_meanStdDev
25_median
26_minMaxLoc
27_multiply
28_not
29_opticalFlowLK
30_or
31_phase
32_gaussianPyramid
33_remap
34_scaleImage
35_sobel
36_subtract
37_tableLookup
38_threshold
39_warpAffine
40_warpPerspective
41_xor
```

## Vision Performance Tests

```
 Test ID - Test Name                     

   1     - absdiff-1080p-U8              

   2     - accumulate-1080p-U8           

   3     - accumulate_square-1080p-U8    

   4     - accumulate_weighted-1080p-U8  

   5     - add-1080p-U8                  

   6     - and-1080p-U8                  

   7     - box_3x3-1080p-U8              

   8     - canny_edge_detector-1080p-U8  

   9     - channel_combine-1080p-RGBA      

   10    - channel_extract-1080p-U8      

   11    - color_convert-1080p-RGB       

   12    - convertdepth-1080p-S016       

   13    - custom_convolution-1080p-S016 

   14    - dilate_3x3-1080p-U8           

   15    - equalize_histogram-1080p-U8   

   16    - erode_3x3-1080p-U8            

   17    - fast_corners-1080p-U8         

   18    - gaussian_3x3-1080p-U8         

   19    - gaussian_pyramid-1080p-U8     

   20    - halfscale_gaussian-1080p-U8   

   21    - harris_corners-1080p-U8       

   22    - histogram-1080p-U8            

   23    - integral_image-1080p-U8       

   24    - magnitude-1080p-S16           

   25    - mean_stddev-1080p-U8          

   26    - median_3x3-1080p-U8           

   27    - minmaxloc-1080p-U8            

   28    - multiply-1080p-U8             

   29    - not-1080p-U8                  

   30    - optical_flow_pyr_lk-1080p-U8  

   31    - or-1080p-U8                   

   32    - phase-1080p-S16               

   33    - remap-1080p-U008              

   34    - scale_image-1080p-U8          

   35    - sobel_3x3-1080p-U8            

   36    - subtract-1080p-U8             

   37    - table_lookup-1080p-U8         

   38    - threshold-1080p-U8            

   39    - warp_affine-1080p-U8          

   40    - warp_perspective-1080p-U8     

   41    - xor-1080p-U8 
```
