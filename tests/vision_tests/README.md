# OpenVX Vision Function Tests

OpenVX core tests using runVX

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
                  [--backend_type BACKEND_TYPE]
                  [--profiling PROFILING]

Arguments:
  -h, --help        show this help message and exit
  --runvx_directory RunVX Executable Directory - required
  --hardware_mode   OpenVX Vision Function Target - optional (default:CPU [options:CPU/GPU])
  --list_tests      List Vision Performance Tests - optional (default:no [options:no/yes])
  --test_filter     Vision Performance Test Filter - optional (default:0 [range:1 - N])
  --num_frames      Run Test for X number of frames - optional (default:1000 [range:1 - N])
  --backend_type    OpenVX Backend type - optional (default:CPU [options:CPU/HIP/OCL])
  --profiling       Enable GPU profiling with ROCm profiler (rocprof) - optional (default:no [options:yes/no])
```

## Vision Performance Tests

```
Test ID - Test Name                     

   1     - AbsDiff_U8_U8U8               

   2     - AbsDiff_S16_S16S16_Sat        

   3     - Add_U8_U8U8_Wrap              

   4     - Add_U8_U8U8_Sat               

   5     - Add_S16_U8U8_Wrap             

   6     - Add_S16_S16U8_Wrap            

   7     - Add_S16_S16U8_Sat             

   8     - Add_S16_S16S16_Wrap           

   9     - Add_S16_S16S16_Sat            

   10    - Sub_U8_U8U8_Wrap              

   11    - Sub_U8_U8U8_Sat               

   12    - Sub_S16_U8U8_Wrap             

   13    - Sub_S16_S16U8_Wrap            

   14    - Sub_S16_S16U8_Sat             

   15    - Sub_S16_U8S16_Wrap            

   16    - Sub_S16_U8S16_Sat             

   17    - Sub_S16_S16S16_Wrap           

   18    - Sub_S16_S16S16_Sat            

   19    - Mul_U8_U8U8_Wrap_Trunc        

   20    - Mul_U8_U8U8_Wrap_Round        

   21    - Mul_U8_U8U8_Sat_Trunc         

   22    - Mul_U8_U8U8_Sat_Round         

   23    - Mul_S16_U8U8_Wrap_Trunc       

   24    - Mul_S16_U8U8_Wrap_Round       

   25    - Mul_S16_U8U8_Sat_Trunc        

   26    - Mul_S16_U8U8_Sat_Round        

   27    - Mul_S16_S16U8_Wrap_Trunc      

   28    - Mul_S16_S16U8_Wrap_Round      

   29    - Mul_S16_S16U8_Sat_Trunc       

   30    - Mul_S16_S16U8_Sat_Round       

   31    - Mul_S16_S16S16_Wrap_Trunc     

   32    - Mul_S16_S16S16_Wrap_Round     

   33    - Mul_S16_S16S16_Sat_Trunc      

   34    - Mul_S16_S16S16_Sat_Round      

   35    - Magnitude_S16_S16S16          

   36    - Phase_U8_S16S16               

   37    - WeightedAverage_U8_U8U8       

   38    - And_U8_U8U8                   

   39    - Or_U8_U8U8                    

   40    - Xor_U8_U8U8                   

   41    - Not_U8_U8                     

   42    - Lut_U8_U8                     

   43    - ColorDepth_U8_S16_Wrap        

   44    - ColorDepth_U8_S16_Sat         

   45    - ColorDepth_S16_U8             

   46    - ChannelExtract_U8_U16_Pos0    

   47    - ChannelExtract_U8_U16_Pos1    

   48    - ChannelExtract_U8_U24_Pos0    

   49    - ChannelExtract_U8_U24_Pos1    

   50    - ChannelExtract_U8_U24_Pos2    

   51    - ChannelExtract_U8_U32_Pos0_UYVY

   52    - ChannelExtract_U8_U32_Pos1_YUYV

   53    - ChannelExtract_U8_U32_Pos2_UYVY

   54    - ChannelExtract_U8_U32_Pos3_YUYV

   55    - ChannelExtract_U8_U32_Pos0_RGBX

   56    - ChannelExtract_U8_U32_Pos1_RGBX

   57    - ChannelExtract_U8_U32_Pos2_RGBX

   58    - ChannelExtract_U8_U32_Pos3_RGBX

   59    - ChannelExtract_U8U8U8_U24     

   60    - ChannelExtract_U8U8U8_U32     

   61    - ChannelExtract_U8U8U8U8_U32   

   62    - ChannelCombine_U32_U8U8U8U8_RGBX

   63    - ColorConvert_RGB_RGBX         

   64    - ColorConvert_RGB_UYVY         

   65    - ColorConvert_RGB_YUYV         

   66    - ColorConvert_RGB_IYUV         

   67    - ColorConvert_RGB_NV12         

   68    - ColorConvert_RGB_NV21         

   69    - ColorConvert_RGBX_RGB         

   70    - ColorConvert_RGBX_UYVY        

   71    - ColorConvert_RGBX_YUYV        

   72    - ColorConvert_RGBX_IYUV        

   73    - ColorConvert_RGBX_NV12        

   74    - ColorConvert_RGBX_NV21        

   75    - ColorConvert_IYUV_RGB         

   76    - ColorConvert_IYUV_RGBX        

   77    - FormatConvert_IYUV_UYVY       

   78    - FormatConvert_IYUV_YUYV       

   79    - ColorConvert_NV12_RGB         

   80    - ColorConvert_NV12_RGBX        

   81    - FormatConvert_NV12_UYVY       

   82    - FormatConvert_NV12_YUYV       

   83    - ColorConvert_YUV4_RGB         

   84    - ColorConvert_YUV4_RGBX        

   85    - FormatConvert_IUV_UV12        

   86    - FormatConvert_UV12_IUV        

   87    - FormatConvert_UV_UV12         

   88    - ScaleUp2x2_U8_U8              

   89    - Box_U8_U8_3x3                 

   90    - Dilate_U8_U8_3x3              

   91    - Erode_U8_U8_3x3               

   92    - Median_U8_U8_3x3              

   93    - Gaussian_U8_U8_3x3            

   94    - ScaleGaussianHalf_U8_U8_3x3   

   95    - ScaleGaussianHalf_U8_U8_5x5   

   96    - Convolve_U8_U8_3x3            

   97    - Convolve_S16_U8_3x3           

   98    - Sobel_S16S16_U8_3x3_GXY       

   99    - Sobel_S16_U8_3x3_GX           

   100   - Threshold_U8_U8_Binary        

   101   - Threshold_U8_U8_Range         

   102   - Threshold_U8_S16_Binary       

   103   - Threshold_U8_S16_Range        

   104   - ScaleImage_U8_U8_Nearest      

   105   - ScaleImage_U8_U8_Bilinear     

   106   - ScaleImage_U8_U8_Bilinear_Replicate

   107   - ScaleImage_U8_U8_Bilinear_Constant

   108   - ScaleImage_U8_U8_Area         

   109   - WarpAffine_U8_U8_Nearest      

   110   - WarpAffine_U8_U8_Nearest_Constant

   111   - WarpAffine_U8_U8_Bilinear     

   112   - WarpAffine_U8_U8_Bilinear_Constant

   113   - FastCorners_XY_U8_NoSupression

   114   - FastCorners_XY_U8_Supression  

   115   - Canny_3x3_L1Norm              

   116   - Canny_3x3_L2Norm   
```
