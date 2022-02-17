FROM centos:centos8

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

# install mivisionx base dependencies - Level 1
RUN yum -y update --nogpgcheck && yum -y install --nogpgcheck gcc gcc-c++ kernel-devel make cmake git
# install ROCm for mivisionx OpenCL dependency - Level 2
RUN echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/centos8/rpm/\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" > \
        /etc/yum.repos.d/rocm.repo && yum -y update --nogpgcheck && yum -y install --nogpgcheck rocm-dev
# install OpenCV & FFMPEG - Level 3
RUN yum -y groupinstall 'Development Tools' --nogpgcheck && yum -y install --nogpgcheck gtk2-devel libjpeg-devel libpng-devel libtiff-devel libavc1394 wget unzip && \
        mkdir opencv && cd opencv && wget https://github.com/opencv/opencv/archive/4.5.5.zip && unzip 4.5.5.zip && \
        mkdir build && cd build && \
        cmake -DWITH_OPENCL=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_VA_INTEL=OFF -DWITH_OPENCL_SVM=OFF ../opencv-4.5.5 && \
        make -j8 && make install
RUN yum -y install --nogpgcheck https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && \
        yum -y install --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm && \
        yum -y install --nogpgcheck http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/SDL2-2.0.10-2.el8.x86_64.rpm && \
        yum -y install --nogpgcheck ffmpeg ffmpeg-devel
# install MIVisionX neural net dependency - Level 4
RUN yum -y install --nogpgcheck libsqlite3x-devel bzip2-devel openssl-devel python3-devel autoconf automake libtool curl make gcc-c++ unzip && \
        mkdir neuralNet && cd neuralNet && wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && cp half-files/include/half.hpp /usr/local/include/ && \
        git clone -b rocm-4.2.0 https://github.com/RadeonOpenCompute/rocm-cmake.git && cd rocm-cmake && mkdir build && cd build && \
        cmake ../ && make -j8 && make install && cd ../../ && \
        git clone -b v3.12.0 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && make install
RUN wget https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/archive/1.1.5.zip && unzip 1.1.5.zip && \
        cd MIOpenGEMM-1.1.5 && mkdir build && cd build && cmake ../ && make -j8 && make install && cd ../../ && \
        wget https://github.com/ROCmSoftwarePlatform/MIOpen/archive/2.14.0.zip && unzip 2.14.0.zip && \
        cd MIOpen-2.14.0 && cmake -P install_deps.cmake --minimum && mkdir build && cd build && \
        cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_USE_MIOPENGEMM=On ../ && \
        make -j8 && make MIOpenDriver && make install && cd ../../ && \
        git clone -b v3.12.0 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && make install

WORKDIR /workspace

# install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && mkdir build && cd build && \
        cmake ../MIVisionX && make -j8 && make install