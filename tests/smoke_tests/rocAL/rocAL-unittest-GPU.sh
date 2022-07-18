export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/rpp/lib
rm -rf rocAL-GPU-RESULTS
mkdir rocAL-GPU-RESULTS

../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-GPU-RESULTS/1-rocAL-GPU-Rotate.png 224 224 2 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-GPU-RESULTS/2-rocAL-GPU-Brightness.png 224 224 3 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-GPU-RESULTS/3-rocAL-GPU-Flip.png 224 224 6 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-GPU-RESULTS/4-rocAL-GPU-Blur.png 224 224 7 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-GPU-RESULTS/5-rocAL-GPU-SnPNoise.png 224 224 13 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-GPU-RESULTS/6-rocAL-GPU-Snow.png 224 224 14 1 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-GPU-RESULTS/7-rocAL-GPU-Pixelate.png 224 224 19 1 1
