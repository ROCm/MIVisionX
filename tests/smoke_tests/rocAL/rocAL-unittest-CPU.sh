export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/rpp/lib
rm -rf rocAL-CPU-RESULTS
mkdir rocAL-CPU-RESULTS

../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-CPU-RESULTS/1-rocAL-GPU-Rotate.png 224 224 2 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-CPU-RESULTS/2-rocAL-GPU-Brightness.png 224 224 3 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-CPU-RESULTS/3-rocAL-GPU-Flip.png 224 224 6 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-CPU-RESULTS/4-rocAL-GPU-Blur.png 224 224 7 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-CPU-RESULTS/5-rocAL-GPU-SaltAndPepperNoise.png 224 224 13 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-CPU-RESULTS/6-rocAL-GPU-Snow.png 224 224 14 0 1
../../../utilities/rocAL/rocAL_unittests/build/rocAL_unittests 0 1 image_224x224 rocAL-CPU-RESULTS/7-rocAL-GPU-Pixelate.png 224 224 19 0 1
