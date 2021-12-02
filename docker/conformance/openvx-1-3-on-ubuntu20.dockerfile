FROM ubuntu:20.04

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN apt-get update -y
# install mivisionx base dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git
# install ROCm for mivisionx OpenCL & HIP
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install libnuma-dev wget sudo gnupg2 kmod python3-dev &&  \
        wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - && \
        echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list && \
        sudo apt-get update -y && \
        sudo apt-get -y install rocm-dev

WORKDIR /workspace

ENV OPENVX_INC=/workspace/MIVisionX/amd_openvx/openvx
ENV OPENVX_DIR_OPENCL=/workspace/build-opencl
ENV OPENVX_DIR_HIP=/workspace/build-hip
ENV VX_TEST_DATA_PATH=/workspace/conformance_tests/OpenVX-cts/test_data/

# install MIVisionX OpenCL
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && \
        python MIVisionX/docker/conformance/system_info.py && \
        mkdir build-opencl && cd build-opencl && cmake ../MIVisionX && make -j8
RUN mkdir conformance_tests && cd conformance_tests && git clone -b openvx_1.3 https://github.com/KhronosGroup/OpenVX-cts.git && \
        mkdir build-cts-opencl && cd build-cts-opencl && \
        cmake -DOPENVX_INCLUDES=$OPENVX_INC/include -DOPENVX_LIBRARIES=$OPENVX_DIR_OPENCL/lib/libopenvx.so\;$OPENVX_DIR_OPENCL/lib/libvxu.so\;pthread\;dl\;m\;rt -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts && \
        cmake --build .
RUN cd conformance_tests/build-cts-opencl && AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-OCL-centos7.md && \
        AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-GPU-CTS-OCL-centos7.md && \
        mv *.md /workspace/
# install MIVisionX HIP
RUN mkdir build-hip && cd build-hip && cmake ../MIVisionX -DBACKEND=HIP && make -j8
RUN cd conformance_tests && mkdir build-cts-hip && cd build-cts-hip && \
        cmake -DOPENVX_INCLUDES=$OPENVX_INC/include -DOPENVX_LIBRARIES=$OPENVX_DIR_HIP/lib/libopenvx.so\;$OPENVX_DIR_HIP/lib/libvxu.so\;/opt/rocm/hip/lib/libamdhip64.so\;pthread\;dl\;m\;rt -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts && \
        cmake --build .
RUN cd conformance_tests/build-cts-hip && AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-HIP-centos7.md && \
        AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-GPU-CTS-HIP-centos7.md && \
        mv *.md /workspace/