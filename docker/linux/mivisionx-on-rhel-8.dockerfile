FROM compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-rel-5.4:104-rhel-8.x-stg1

ENV MIVISIONX_DEPS_ROOT=/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN sudo yum --nobest update -y

# install base dependencies
RUN sudo yum -y install gcc gcc-c++ cmake pkg-config git kernel-devel

# install OpenCV
RUN sudo yum -y install opencv opencv-devel

# install neural net dependencies
RUN sudo yum -y install rocblas rocblas-devel miopen-hip miopen-hip-devel migraphx migraphx-devel

# install rocAL dependencies
RUN sudo yum -y install make unzip libomp-devel wget clang
RUN mkdir rocAL_deps && cd rocAL_deps && wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && sudo mkdir -p /usr/local/include/half && sudo cp half-files/include/half.hpp /usr/local/include/half && cd
RUN sudo yum -y install autoconf automake bzip2-devel openssl-devel python3-devel gflags-devel glog-devel lmdb-devel nasm yasm jsoncpp-devel && \
        git clone -b 2.0.6.2 https://github.com/rrawther/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ../ && make -j4 && sudo make install && cd
RUN yum -y install sqlite-devel libtool && yum -y groupinstall 'Development Tools' && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.bz2 && tar xjvf boost_1_80_0.tar.bz2 && \
    cd boost_1_80_0 && ./bootstrap.sh --prefix=/usr/local --with-python=python3 && \
    ./b2 stage -j16 threading=multi link=shared cxxflags="-std=c++11" && \
    sudo ./b2 install threading=multi link=shared --with-system --with-filesystem && \
    ./b2 stage -j16 threading=multi link=static cxxflags="-std=c++11 -fpic" cflags="-fpic" && \
    sudo ./b2 install threading=multi link=static --with-system --with-filesystem
RUN git clone -b v3.21.9 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && sudo make install && sudo ldconfig && cd
RUN git clone -b 1.0.0 https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && cd rpp && mkdir build && cd build && \
        cmake -DBACKEND=HIP ../ && make -j4 && sudo make install && cd

ENV MIVISIONX_WORKSPACE=/workspace
WORKDIR $MIVISIONX_WORKSPACE

# Install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && \
        mkdir build && cd build && cmake -DBACKEND=HIP ../MIVisionX && make -j8 && make install