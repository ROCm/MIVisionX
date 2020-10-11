<p align="center"><img width="60%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/MIVisionX.png" /></p>


[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/GPUOpen-ProfessionalCompute-Libraries/MIVisionX?style=for-the-badge)](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases)

```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```

- ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `#f03c15`
- ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `#c5f015`
- ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) `#1589F0`

| Test Level | MIVisionX Dependencies | Modules | Libraries and Executables |
|------------|------------------------|---------|---------------------------|
| `Level_1` |cmake <br> gcc <br> g++|amd_openvx|![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `libopenvx.so` <br> ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `libvxu.so`|
| `Level_2` |ROCm OpenCL (rocm-dev/rocm-dkms) <br> +Level 1|amd_openvx <br> amd_openvx_extensions <br> utilities| ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) libopenvx.so <br> ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) libvx_loomsl.so <br> ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) libvxu.so <br> ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) loom_shell <br>![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) runcl <br> ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) runvx|
| `Level_3` |OpenCV <br>  FFMPEG <br> +Level 2|amd_openvx <br> amd_openvx_extensions <br> utilities|libopenvx.so <br>  libvx_amd_media.so <br>  libvx_loomsl.so <br>  libvx_opencv.so <br>  libvxu.so <br> loom_shell <br>  mv_compile <br>  runcl <br>  runvx|
| `Level_4` |MIOpenGEMM <br> MIOpen <br> +Level 3|amd_openvx <br>  amd_openvx_extensions <br> apps <br> utilities|libopenvx.so <br>  libvx_amd_media.so <br>  libvx_loomsl.so <br>  libvx_nn.so <br>  libvx_opencv.so <br>  libvxu.so <br> caffe2openvx <br>  inference_server_app <br>  loom_shell <br>  mv_compile <br>  runcl <br>  runvx|
| `Level_5` |AMD_RPP <br> RALI deps <br> +Level 4|amd_openvx <br> amd_openvx_extensions <br> apps <br> rali <br> utilities|libopenvx.so <br>  libvx_amd_media.so  <br> libvx_nn.so <br> libvx_rpp.so <br> librali.so   <br>  libvx_loomsl.so <br> libvx_opencv.so <br>  libvxu.so <br> rali_pybind.cpython-36m-x86_64-linux-gnu.so <br> caffe2openvx <br>  inference_server_app <br>  loom_shell <br>  mv_compile <br>  runcl <br>  runvx |
