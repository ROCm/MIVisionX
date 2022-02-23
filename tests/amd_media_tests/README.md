# AMD Media Tests

Change the line in the encoder.gdf to contain full path:<br>
data yuvimg  = image:1280,720,NV12:read,FULL-PATH/MIVisionX/tests/amd_media_tests/test.yuv

<br>
Run runvx -affinity:GPU file encoder.gdf<br>
<br>
Sample Expected output:<br>
INFO: writing 1280x720 4.00mbps 30.00fps gopsize=15 bframes=0 video into output.264<br>
csv,HEADER ,STATUS, COUNT,cur-ms,avg-ms,min-ms,clenqueue-ms,clwait-ms,clwrite-ms,clread-ms<br>
csv,OVERALL,  PASS,     1,      ,  6.26,  6.26,  0.00,  0.00,  0.00,  0.00 (median 6.256)<br>
