# AMD Media Tests

data yuvimg  = image:1280,720,NV12:read,test.yuv<br>

<br>
Run runvx -affinity:GPU file encoder.gdf<br>
<br>
Sample Expected output:<br>
INFO: writing 1280x720 4.00mbps 30.00fps gopsize=15 bframes=0 video into output.264<br>
csv,HEADER ,STATUS, COUNT,cur-ms,avg-ms,min-ms,clenqueue-ms,clwait-ms,clwrite-ms,clread-ms<br>
csv,OVERALL,  PASS,     1,      ,  6.26,  6.26,  0.00,  0.00,  0.00,  0.00 (median 6.256)<br>


<br>

Make sure to edit decoder.gdf and add the full path up to /MIVisionX/tests/amd_media_tests/test.mp4 <br>
Run runvx -affinity:GPU -frames:0,eof file decoder.gdf <br>
<br>
Sample expected output:<br>
INFO: reading 1920x1080 into slice#0 from /home/svcbuild/mivisionx-paveltc-tot/MIVisionX/tests/amd_media_tests/test.mp4<br>
csv,HEADER ,STATUS, COUNT,cur-ms,avg-ms,min-ms,clenqueue-ms,clwait-ms,clwrite-ms,clread-ms<br>
csv,OVERALL,  PASS,     0,      ,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00 (median 0.000)<br>
> total elapsed time:   0.00 sec<br>
