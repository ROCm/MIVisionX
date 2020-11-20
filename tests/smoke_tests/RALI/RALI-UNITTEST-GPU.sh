export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib:/opt/rocm/rpp/lib
rm -rf RALI-GPU-RESULTS
mkdir RALI-GPU-RESULTS

../../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-GPU-RESULTS/1-RALI-GPU-Rotate.png 224 224 2 1 1
../../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-GPU-RESULTS/2-RALI-GPU-Brightness.png 224 224 3 1 1
../../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-GPU-RESULTS/3-RALI-GPU-Flip.png 224 224 6 1 1
../../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-GPU-RESULTS/4-RALI-GPU-Blur.png 224 224 7 1 1
../../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-GPU-RESULTS/5-RALI-GPU-SnPNoise.png 224 224 13 1 1
../../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-GPU-RESULTS/6-RALI-GPU-Snow.png 224 224 14 1 1
../../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-GPU-RESULTS/7-RALI-GPU-Pixelate.png 224 224 19 1 1
