#!/bin/sh
runvx -affinity:CPU file channelextract.gdf image:1280,720,IYUV:READ,stm_1280x720.yuv image:1280,720,U008:WRITE,channelextract_CPU_1280x720.rgb
runvx -affinity:GPU file channelextract.gdf image:1280,720,IYUV:READ,stm_1280x720.yuv image:1280,720,U008:WRITE,channelextract_GPU_1280x720.rgb
