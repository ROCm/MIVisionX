export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib:/opt/rocm/rpp/lib
rm -rf RALI-CPU-RESULTS
mkdir RALI-CPU-RESULTS

../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-CPU-RESULTS/1-Rotate.png 224 224 2 0 1
../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-CPU-RESULTS/2-Brightness.png 224 224 3 0 1
../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-CPU-RESULTS/3-Flip.png 224 224 6 0 1
../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-CPU-RESULTS/4-Blur.png 224 224 7 0 1
../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-CPU-RESULTS/5-SaltAndPepperNoise.png 224 224 13 0 1
../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-CPU-RESULTS/6-Snow.png 224 224 14 0 1
../../utilities/rali/rali_unittests/build/rali_unittests image_224x224 RALI-CPU-RESULTS/7-Pixelate.png 224 224 19 0 1
