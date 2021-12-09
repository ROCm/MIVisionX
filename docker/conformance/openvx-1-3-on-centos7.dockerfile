FROM centos:centos7

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

# install mivisionx base dependencies
RUN yum -y update --nogpgcheck && yum -y install --nogpgcheck http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm && \
        yum -y install --nogpgcheck gcc gcc-c++ kernel-devel make cmake3 git && yum-config-manager --enable rhel-server-rhscl-7-rpms && \
        yum -y install --nogpgcheck centos-release-scl && yum -y install --nogpgcheck devtoolset-7
# Enable Developer Toolset 7
SHELL [ "/usr/bin/scl", "enable", "devtoolset-7" ]
# install ROCm for mivisionx OpenCL & HIP
RUN echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/yum/rpm\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" > \
        /etc/yum.repos.d/rocm.repo && yum -y install --nogpgcheck rocm-dev

WORKDIR /workspace

ENV OPENVX_INC=/workspace/MIVisionX/amd_openvx/openvx
ENV OPENVX_DIR_OPENCL=/workspace/build-opencl
ENV OPENVX_DIR_HIP=/workspace/build-hip
ENV VX_TEST_DATA_PATH=/workspace/conformance_tests/OpenVX-cts/test_data/

# install MIVisionX OpenCL
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && \
        python MIVisionX/docker/conformance/system_info.py && \
        mkdir build-opencl && cd build-opencl && cmake3 ../MIVisionX && make -j8
RUN mkdir conformance_tests && cd conformance_tests && git clone -b openvx_1.3 https://github.com/KhronosGroup/OpenVX-cts.git && \
        mkdir build-cts-opencl && cd build-cts-opencl && \
        cmake3 -DOPENVX_INCLUDES=$OPENVX_INC/include -DOPENVX_LIBRARIES=$OPENVX_DIR_OPENCL/lib/libopenvx.so\;$OPENVX_DIR_OPENCL/lib/libvxu.so\;pthread\;dl\;m\;rt -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts && \
        cmake3 --build .
RUN cd conformance_tests/build-cts-opencl && AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-OCL-centos7.md && \
        AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-GPU-CTS-OCL-centos7.md && \
        mv *.md /workspace/
# install MIVisionX HIP
RUN mkdir build-hip && cd build-hip && cmake3 ../MIVisionX -DBACKEND=HIP && make -j8
RUN cd conformance_tests && mkdir build-cts-hip && cd build-cts-hip && \
        cmake3 -DOPENVX_INCLUDES=$OPENVX_INC/include -DOPENVX_LIBRARIES=$OPENVX_DIR_HIP/lib/libopenvx.so\;$OPENVX_DIR_HIP/lib/libvxu.so\;/opt/rocm/hip/lib/libamdhip64.so\;pthread\;dl\;m\;rt -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts && \
        cmake3 --build .
RUN cd conformance_tests/build-cts-hip && AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-HIP-centos7.md && \
        AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-GPU-CTS-HIP-centos7.md && \
        mv *.md /workspace/