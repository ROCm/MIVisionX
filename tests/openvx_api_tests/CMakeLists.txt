################################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
cmake_minimum_required(VERSION 3.5)

# Install data indepedent tests
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../openvx_api_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/tests COMPONENT test)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../library_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/tests COMPONENT test)

# default run
# canny
add_test(
  NAME
    openvx_canny
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/canny"
                              "${CMAKE_CURRENT_BINARY_DIR}/canny"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_canny"
)

# channel extract
add_test(
  NAME
    openvx_channel_extract
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/channel_extract"
                              "${CMAKE_CURRENT_BINARY_DIR}/channel_extract"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_channel_extract"
)

# color convert
add_test(
  NAME
    openvx_color_convert
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/color_convert"
                              "${CMAKE_CURRENT_BINARY_DIR}/color_convert"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "openvx_color_convert"
)

# CPU Tests
add_test(NAME openvx_canny_CPU 
              COMMAND openvx_canny 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/canny)
set_property(TEST openvx_canny_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")
add_test(NAME openvx_channel_extract_CPU 
              COMMAND openvx_channel_extract 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/channel_extract)
set_property(TEST openvx_channel_extract_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")
add_test(NAME openvx_color_convert_CPU 
              COMMAND openvx_color_convert 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/color_convert)
set_property(TEST openvx_color_convert_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=CPU")

set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 QUIET)
find_package(OpenCV QUIET)

# OpenVX Tests
if(Python3_FOUND)
  add_test(NAME openvx_tests_runVisionPython_CPU 
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/vision_tests/runVisionTests.py
            --runvx_directory ${CMAKE_BINARY_DIR}/bin --num_frames 10
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif(Python3_FOUND)

# OpenCV Tests
if(OpenCV_FOUND AND Python3_FOUND)
  add_test(NAME vx_opencv_tests_runOpenCVPython 
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/amd_opencv_tests/runOpenCVTests.py
            --runvx_directory ${CMAKE_BINARY_DIR}/bin
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif(OpenCV_FOUND AND Python3_FOUND)

# GPU Tests
if(GPU_SUPPORT)
  # API tests
  add_test(NAME openvx_canny_GPU 
              COMMAND openvx_canny 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/canny)
  set_property(TEST openvx_canny_GPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")
  add_test(NAME openvx_channel_extract_GPU 
              COMMAND openvx_channel_extract 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/channel_extract)
  set_property(TEST openvx_channel_extract_CPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")
  add_test(NAME openvx_color_convert_GPU 
              COMMAND openvx_color_convert 
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/color_convert)
  set_property(TEST openvx_color_convert_GPU PROPERTY ENVIRONMENT "AGO_DEFAULT_TARGET=GPU")

  # OpenVX Tests
  if(Python3_FOUND)
    add_test(NAME openvx_tests_runVisionPython_GPU 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/vision_tests/runVisionTests.py
          --runvx_directory ${CMAKE_BINARY_DIR}/bin --num_frames 10 --hardware_mode GPU --backend_type ${BACKEND}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif(Python3_FOUND)

  # Neural Network tests
  if(NEURAL_NET AND Python3_FOUND)
    # caffe2nnir2openvx No Fuse flow
    add_test(NAME caffe2nnir2openvx_no_fuse
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 1
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # caffe2nnir2openvx Fuse flow
    add_test(NAME caffe2nnir2openvx_fuse 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 2 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # caffe2nnir2openvx FP16 flow
    add_test(NAME caffe2nnir2openvx_fp16
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 3 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # onnx2nnir2openvx No Fuse flow
    add_test(NAME onnx2nnir2openvxx_no_fuse 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 4 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})    
    # onnx2nnir2openvx Fuse flow
    add_test(NAME onnx2nnir2openvxx_fuse 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 5 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # onnx2nnir2openvx FP16 flow
    add_test(NAME onnx2nnir2openvxx_fp16 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 6 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # nnef2nnir2openvx No Fuse flow
    add_test(NAME nnef2nnir2openvxx_no_fuse 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 7 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})    
    # nnef2nnir2openvx Fuse flow
    add_test(NAME nnef2nnir2openvxx_fuse 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 8 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # nnef2nnir2openvx FP16 flow
    add_test(NAME nnef2nnir2openvxx_fp16 
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/neural_network_tests/runNeuralNetworkTests.py
          --profiler_mode 9 --reinstall off
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../neural_network_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/tests COMPONENT test)
  endif(NEURAL_NET AND Python3_FOUND)

endif(GPU_SUPPORT)

install(FILES ${CMAKE_BINARY_DIR}/CTestTestfile.cmake DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/tests COMPONENT test)
