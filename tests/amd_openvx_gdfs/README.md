# OpenVX Vision Function Tests Using GDFs

OpenVX core tests using 300+ GDFs with runVX

## Script to run vision gdf tests

```
python3 runOpenVX.py --help
```


usage:

```
runOpenVX.py  --runvx_directory RUNVX_DIRECTORY
              --list_tests LIST_TESTS
              --num_frames NUM_FRAMES
              --backend_type BACKEND_TYPE
              --hidden HIDDEN

Arguments:
  -h, --help        show this help message and exit
  --runvx_directory RunVX Executable Directory - optional (default:/opt/rocm/bin)
  --list_tests      List Vision Performance Tests - optional (default:no [options:no/yes])
  --num_frames      Run Test for X number of frames - optional (default:1000 [range:1 - N])
  --backend_type    OpenVX Backend type - optional (default:CPU [options:CPU/HIP/OCL])
  --hidden          Hidden Vision Tests - optional (default:no [options:no/yes])
```

## Vision GDF Tests

### Arithmetic Node Tests

```
 Test ID - GDF Name                      
   1     - AbsDiff_S16_S16S16_Sat.gdf    
   2     - AbsDiff_U8_U8U8.gdf           
   3     - Add_S16_S16S16_Sat.gdf        
   4     - Add_S16_S16S16_Wrap.gdf       
   5     - Add_S16_S16U8_Sat.gdf         
   6     - Add_S16_S16U8_Wrap.gdf        
   7     - Add_S16_U8U8_Wrap.gdf         
   8     - Add_U8_U8U8_Sat.gdf           
   9     - Add_U8_U8U8_Wrap.gdf          
   10    - Magnitude_S16_S16S16.gdf      
   11    - Mul_S16_S16S16_Sat_Round.gdf  
   12    - Mul_S16_S16S16_Sat_Trunc.gdf  
   13    - Mul_S16_S16S16_Wrap_Round.gdf 
   14    - Mul_S16_S16S16_Wrap_Trunc.gdf 
   15    - Mul_S16_S16U8_Sat_Round.gdf   
   16    - Mul_S16_S16U8_Sat_Trunc.gdf   
   17    - Mul_S16_S16U8_Wrap_Round.gdf  
   18    - Mul_S16_S16U8_Wrap_Trunc.gdf  
   19    - Mul_S16_U8U8_Sat_Round.gdf    
   20    - Mul_S16_U8U8_Sat_Trunc.gdf    
   21    - Mul_S16_U8U8_Wrap_Round.gdf   
   22    - Mul_S16_U8U8_Wrap_Trunc.gdf   
   23    - Mul_U8_U8U8_Sat_Round.gdf     
   24    - Mul_U8_U8U8_Sat_Trunc.gdf     
   25    - Mul_U8_U8U8_Wrap_Round.gdf    
   26    - Mul_U8_U8U8_Wrap_Trunc.gdf    
   27    - Phase_U8_S16S16.gdf           
   28    - Sub_S16_S16S16_Sat.gdf        
   29    - Sub_S16_S16S16_Wrap.gdf       
   30    - Sub_S16_S16U8_Sat.gdf         
   31    - Sub_S16_S16U8_Wrap.gdf        
   32    - Sub_S16_U8S16_Sat.gdf         
   33    - Sub_S16_U8S16_Wrap.gdf        
   34    - Sub_S16_U8U8_Wrap.gdf         
   35    - Sub_U8_U8U8_Sat.gdf           
   36    - Sub_U8_U8U8_Wrap.gdf          
   37    - WeightedAverage_U8_U8U8.gdf   
```

### Color Node Tests

```
 Test ID - GDF Name                      
   1     - ChannelCombine_U16_U8U8.gdf   
   2     - ChannelCombine_U24_U8U8U8_RGB.gdf
   3     - ChannelCombine_U32_U8U8U8U8_RGBX.gdf
   4     - ChannelCombine_U32_U8U8U8_UYVY.gdf
   5     - ChannelCombine_U32_U8U8U8_YUYV.gdf
   6     - ChannelCopy_U1_U1.gdf         
   7     - ChannelCopy_U1_U8.gdf         
   8     - ChannelCopy_U8_U1.gdf         
   9     - ChannelCopy_U8_U8.gdf         
   10    - ChannelExtract_U8U8U8U8_U32.gdf
   11    - ChannelExtract_U8U8U8_U24.gdf 
   12    - ChannelExtract_U8U8U8_U32.gdf 
   13    - ChannelExtract_U8_U16_Pos0.gdf
   14    - ChannelExtract_U8_U16_Pos1.gdf
   15    - ChannelExtract_U8_U24_Pos0.gdf
   16    - ChannelExtract_U8_U24_Pos1.gdf
   17    - ChannelExtract_U8_U24_Pos2.gdf
   18    - ChannelExtract_U8_U32_Pos0.gdf
   19    - ChannelExtract_U8_U32_Pos1.gdf
   20    - ChannelExtract_U8_U32_Pos2.gdf
   21    - ChannelExtract_U8_U32_Pos3.gdf
   22    - ColorConvert_IYUV_RGB.gdf     
   23    - ColorConvert_IYUV_RGBX.gdf    
   24    - ColorConvert_NV12_RGB.gdf     
   25    - ColorConvert_NV12_RGBX.gdf    
   26    - ColorConvert_RGBX_IYUV.gdf    
   27    - ColorConvert_RGBX_NV12.gdf    
   28    - ColorConvert_RGBX_NV21.gdf    
   29    - ColorConvert_RGBX_RGB.gdf     
   30    - ColorConvert_RGBX_UYVY.gdf    
   31    - ColorConvert_RGBX_YUYV.gdf    
   32    - ColorConvert_RGB_IYUV.gdf     
   33    - ColorConvert_RGB_NV12.gdf     
   34    - ColorConvert_RGB_NV21.gdf     
   35    - ColorConvert_RGB_RGBX.gdf     
   36    - ColorConvert_RGB_UYVY.gdf     
   37    - ColorConvert_RGB_YUYV.gdf     
   38    - ColorConvert_YUV4_RGB.gdf     
   39    - ColorConvert_YUV4_RGBX.gdf    
   40    - ColorConvert_alt.gdf          
   41    - ColorDepth_S16_U8.gdf         
   42    - ColorDepth_U8_S16_Sat.gdf     
   43    - ColorDepth_U8_S16_Wrap.gdf    
   44    - FormatConvert_IUV_UV12.gdf    
   45    - FormatConvert_IYUV_UYVY.gdf   
   46    - FormatConvert_IYUV_YUYV.gdf   
   47    - FormatConvert_NV12_UYVY.gdf   
   48    - FormatConvert_NV12_YUYV.gdf   
   49    - FormatConvert_UV12_IUV.gdf    
   50    - FormatConvert_UV_UV12.gdf     
   51    - Lut_U8_U8.gdf                 
   52    - ScaleUp2x2_U8_U8.gdf          
```

### Filter Node Tests

```
 Test ID - GDF Name                      
   1     - Box_U8_U8_3x3.gdf             
   2     - Convolve_S16_U8_3x3.gdf       
   3     - Convolve_S16_U8_9x3.gdf       
   4     - Convolve_S16_U8_9x9.gdf       
   5     - Convolve_U8_U8_3x3.gdf        
   6     - Convolve_U8_U8_9x3.gdf        
   7     - Convolve_U8_U8_9x9.gdf        
   8     - Dilate_U1_U1_3x3.gdf          
   9     - Dilate_U1_U8_3x3.gdf          
   10    - Dilate_U8_U1_3x3.gdf          
   11    - Dilate_U8_U8_3x3.gdf          
   12    - Erode_U1_U1_3x3.gdf           
   13    - Erode_U1_U8_3x3.gdf           
   14    - Erode_U8_U1_3x3.gdf           
   15    - Erode_U8_U8_3x3.gdf           
   16    - Gaussian_U8_U8_3x3.gdf        
   17    - Median_U8_U8_3x3.gdf          
   18    - ScaleGaussianHalf_U8_U8_3x3.gdf
   19    - ScaleGaussianHalf_U8_U8_5x5.gdf
   20    - Sobel_S16S16_U8_3x3_GXY.gdf   
   21    - Sobel_S16_U8_3x3_GX.gdf       
   22    - Sobel_S16_U8_3x3_GY.gdf       
```

### Geometric Node Tests

```
 Test ID - GDF Name                      
   1     - Remap_U8_U8_Bilinear.gdf      
   2     - Remap_U8_U8_Bilinear_Constant.gdf
   3     - Remap_U8_U8_Nearest.gdf       
   4     - Remap_U8_U8_Nearest_Constant.gdf
   5     - ScaleImage_U8_U8_Area.gdf     
   6     - ScaleImage_U8_U8_Area_align.gdf
   7     - ScaleImage_U8_U8_Area_sad.gdf 
   8     - ScaleImage_U8_U8_Bilinear.gdf 
   9     - ScaleImage_U8_U8_Bilinear_Constant.gdf
   10    - ScaleImage_U8_U8_Bilinear_Replicate.gdf
   11    - ScaleImage_U8_U8_Nearest.gdf  
   12    - WarpAffine_U8_U8_Bilinear.gdf 
   13    - WarpAffine_U8_U8_Bilinear_Constant.gdf
   14    - WarpAffine_U8_U8_Nearest.gdf  
   15    - WarpAffine_U8_U8_Nearest_Constant.gdf
   16    - WarpPerspective_U8_U8_Bilinear.gdf
   17    - WarpPerspective_U8_U8_Bilinear_Constant.gdf
   18    - WarpPerspective_U8_U8_Nearest.gdf
   19    - WarpPerspective_U8_U8_Nearest_Constant.gdf
```

### Logical Node Tests

```
 Test ID - GDF Name                      
   1     - And_U1_U1U1.gdf               
   2     - And_U1_U1U8.gdf               
   3     - And_U1_U8U1.gdf               
   4     - And_U1_U8U8.gdf               
   5     - And_U8_U1U1.gdf               
   6     - And_U8_U1U8.gdf               
   7     - And_U8_U8U1.gdf               
   8     - And_U8_U8U8.gdf               
   9     - And_alt.gdf                   
   10    - And_alt_unaligned.gdf         
   11    - And_alt_variant.gdf           
   12    - Nand_U1_U1U1.gdf              
   13    - Nand_U1_U1U8.gdf              
   14    - Nand_U1_U8U1.gdf              
   15    - Nand_U1_U8U8.gdf              
   16    - Nand_U8_U1U1.gdf              
   17    - Nand_U8_U1U8.gdf              
   18    - Nand_U8_U8U1.gdf              
   19    - Nand_U8_U8U8.gdf              
   20    - Nor_U1_U1U1.gdf               
   21    - Nor_U1_U1U8.gdf               
   22    - Nor_U1_U8U1.gdf               
   23    - Nor_U1_U8U8.gdf               
   24    - Nor_U8_U1U1.gdf               
   25    - Nor_U8_U1U8.gdf               
   26    - Nor_U8_U8U1.gdf               
   27    - Nor_U8_U8U8.gdf               
   28    - Not_U1_U1.gdf                 
   29    - Not_U1_U8.gdf                 
   30    - Not_U8_U1.gdf                 
   31    - Not_U8_U8.gdf                 
   32    - Or_U1_U1U1.gdf                
   33    - Or_U1_U1U8.gdf                
   34    - Or_U1_U8U1.gdf                
   35    - Or_U1_U8U8.gdf                
   36    - Or_U8_U1U1.gdf                
   37    - Or_U8_U1U8.gdf                
   38    - Or_U8_U8U1.gdf                
   39    - Or_U8_U8U8.gdf                
   40    - Or_alt.gdf                    
   41    - Xnor_U1_U1U1.gdf              
   42    - Xnor_U1_U1U8.gdf              
   43    - Xnor_U1_U8U1.gdf              
   44    - Xnor_U1_U8U8.gdf              
   45    - Xnor_U8_U1U1.gdf              
   46    - Xnor_U8_U1U8.gdf              
   47    - Xnor_U8_U8U1.gdf              
   48    - Xnor_U8_U8U8.gdf              
   49    - Xor_U1_U1U1.gdf               
   50    - Xor_U1_U1U8.gdf               
   51    - Xor_U1_U8U1.gdf               
   52    - Xor_U1_U8U8.gdf               
   53    - Xor_U8_U1U1.gdf               
   54    - Xor_U8_U1U8.gdf               
   55    - Xor_U8_U8U1.gdf               
   56    - Xor_U8_U8U8.gdf               
   57    - Xor_alt.gdf                   
```

### Statistical Node Tests

```
 Test ID - GDF Name                      
   1     - Threshold_U1_U8_Binary.gdf    
   2     - Threshold_U1_U8_Range.gdf     
   3     - Threshold_U8_S16_Binary.gdf   
   4     - Threshold_U8_S16_Range.gdf    
   5     - Threshold_U8_U8_Binary.gdf    
   6     - Threshold_U8_U8_Range.gdf  
```

### Vision Node Tests

```
 Test ID - GDF Name                      
   1     - Canny_3x3_L1NORM.gdf          
   2     - Canny_3x3_L2NORM.gdf          
   3     - Canny_5x5_L1NORM.gdf          
   4     - Canny_5x5_L2NORM.gdf          
   5     - Canny_7x7_L1NORM.gdf          
   6     - Canny_7x7_L2NORM.gdf          
   7     - FastCorners_XY_U8_NoSupression.gdf
   8     - FastCorners_XY_U8_Supression.gdf
   9     - Harris_3x3.gdf                
   10    - Harris_5x5.gdf                
   11    - Harris_7x7.gdf                
```

### Vision Profile Node Tests

```
 Test ID - GDF Name                      
   1     - 01_absDiff.gdf                
   2     - 02_accumulate.gdf             
   3     - 03_accumulateSquared.gdf      
   4     - 04_accumulateWeighted.gdf     
   5     - 05_add.gdf                    
   6     - 06_and.gdf                    
   7     - 07_box.gdf                    
   8     - 08_canny.gdf                  
   9     - 09_channelCombine.gdf         
   10    - 10_channelExtract.gdf         
   11    - 11_colorConvert.gdf           
   12    - 12_convertDepth.gdf           
   13    - 13_convolve.gdf               
   14    - 14_dilate.gdf                 
   15    - 15_equalizeHistogram.gdf      
   16    - 16_erode.gdf                  
   17    - 17_fastCorners.gdf            
   18    - 18_gaussian.gdf               
   19    - 19_harrisCorners.gdf          
   20    - 20_halfScaleGaussian.gdf      
   21    - 21_histogram.gdf              
   22    - 22_integralImage.gdf          
   23    - 23_magnitude.gdf              
   24    - 24_meanStdDev.gdf             
   25    - 25_median.gdf                 
   26    - 26_minMaxLoc.gdf              
   27    - 27_multiply.gdf               
   28    - 28_not.gdf                    
   29    - 29_opticalFlowLK.gdf          
   30    - 30_or.gdf                     
   31    - 31_phase.gdf                  
   32    - 32_gaussianPyramid.gdf        
   33    - 33_remap.gdf                  
   34    - 34_scaleImage.gdf             
   35    - 35_sobel.gdf                  
   36    - 36_subtract.gdf               
   37    - 37_tableLookup.gdf            
   38    - 38_threshold.gdf              
   39    - 39_warpAffine.gdf             
   40    - 40_warpPerspective.gdf        
   41    - 41_xor.gdf                    
   42    - 42_scale_nv12_to_rgb.gdf      
   43    - 43_feature_tracker.gdf
```

### CPU Node Tests

```
 Test ID - GDF Name                      
   1     - ColorConvert_U_RGB.gdf        
   2     - ColorConvert_U_RGBX.gdf       
   3     - ColorConvert_V_RGB.gdf        
   4     - ColorConvert_V_RGBX.gdf       
   5     - ColorConvert_Y_RGBX.gdf       
   6     - HarrisMergeSortAndPick_XY_HVC.gdf
   7     - MeanStdDev_DATA_U1.gdf        
   8     - MinMaxLoc_DATA_S16DATA_Loc_Max_Count_Max.gdf
   9     - MinMaxLoc_DATA_S16DATA_Loc_Max_Count_MinMax.gdf
   10    - MinMaxLoc_DATA_S16DATA_Loc_Min_Count_Min.gdf
   11    - MinMaxLoc_DATA_S16DATA_Loc_Min_Count_MinMax.gdf
   12    - MinMaxLoc_DATA_S16DATA_Loc_None_Count_Max.gdf
   13    - MinMaxLoc_DATA_S16DATA_Loc_None_Count_Min.gdf
   14    - MinMaxLoc_DATA_S16DATA_Loc_None_Count_MinMax.gdf
   15    - MinMaxLoc_DATA_U8DATA_Loc_Max_Count_Max.gdf
   16    - MinMaxLoc_DATA_U8DATA_Loc_Max_Count_MinMax.gdf
   17    - MinMaxLoc_DATA_U8DATA_Loc_Min_Count_MinMax.gdf
   18    - MinMaxLoc_DATA_U8DATA_Loc_None_Count_Max.gdf
   19    - MinMaxLoc_DATA_U8DATA_Loc_None_Count_Min.gdf
   20    - Set00Node.gdf                 
   21    - SetFFNode.gdf                 
   22    - SobelMagnitudePhase_S16U8_U8_3x3.gdf
   23    - SobelMagnitude_S16_U8_3x3.gdf 
   24    - SobelPhase_U8_U8_3x3.gdf      
   25    - ThresholdNot_U1_U8_Range.gdf  
   26    - ThresholdNot_U8_S16_Binary.gdf
   27    - ThresholdNot_U8_S16_Range.gdf 
   28    - ThresholdNot_U8_U8_Binary.gdf 
   29    - ThresholdNot_U8_U8_Range.gdf  
```

### Hidden Node Tests

```
 Test ID - GDF Name                      
   1     - CannyEdgeTrace_U8_U8.gdf      
   2     - CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM.gdf
   3     - CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM.gdf
   4     - CannySobelSuppThreshold_U8XY_U8_5X5_L1NORM.gdf
   5     - CannySobelSuppThreshold_U8XY_U8_5X5_L2NORM.gdf
   6     - CannySobelSuppThreshold_U8XY_U8_7X7_L1NORM.gdf
   7     - CannySobelSuppThreshold_U8XY_U8_7X7_L2NORM.gdf
   8     - ColorConvert_IUV_RGB.gdf      
   9     - ColorConvert_IUV_RGBX.gdf     
   10    - ColorConvert_IU_RGB.gdf       
   11    - ColorConvert_IU_RGBX.gdf      
   12    - ColorConvert_IV_RGB.gdf       
   13    - ColorConvert_IV_RGBX.gdf      
   14    - ColorConvert_UV12_RGB.gdf     
   15    - ColorConvert_UV12_RGBX.gdf    
   16    - CopyNode.gdf                  
   17    - FastCornerMerge_XY_XY.gdf     
   18    - GPU_FAIL_Convolve_S16_U8_3x9.gdf
   19    - GPU_FAIL_Convolve_S16_U8_5x5.gdf
   20    - GPU_FAIL_Convolve_S16_U8_7x7.gdf
   21    - GPU_FAIL_Convolve_S16_U8_odd.gdf
   22    - GPU_FAIL_Convolve_U8_U8_3x9.gdf
   23    - GPU_FAIL_Convolve_U8_U8_5x5.gdf
   24    - GPU_FAIL_Convolve_U8_U8_7x7.gdf
   25    - GPU_FAIL_Convolve_U8_U8_odd.gdf
   26    - GeometricScale.gdf            
   27    - HistogramMerge_DATA_DATA.gdf  
   28    - LinearFilter_ANY_2_ANY.gdf    
   29    - LinearFilter_ANY_ANY.gdf      
   30    - Mul_U24_U24U8_Sat_Round.gdf   
   31    - SelectNode.gdf     
```