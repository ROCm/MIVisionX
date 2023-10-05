FROM compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-rel-5.4:104-sles-stg1

ENV MIVISIONX_DEPS_ROOT=/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN sudo zypper update -y

# install base dependencies
RUN sudo zypper -n install gcc gcc-c++ cmake pkg-config git kernel-devel

# install rocAL dependencies
RUN sudo zypper -n install make unzip libomp-devel wget
RUN mkdir rocAL_deps && cd rocAL_deps && wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip && \
        unzip half-1.12.0.zip -d half-files && sudo mkdir -p /usr/local/include/half && sudo cp half-files/include/half.hpp /usr/local/include/half && cd
RUN zypper -n install autoconf automake bzip2 openssl-devel python3-devel gflags-devel glog-devel lmdb-devel nasm yasm jsoncpp-devel && \
        git clone -b 2.0.6.2 https://github.com/rrawther/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ../ && make -j4 && sudo make install && cd
RUN zypper -n install sqlite-devel libtool && zypper -n install -t pattern devel_basis
RUN git clone -b v3.21.9 https://github.com/protocolbuffers/protobuf.git && cd protobuf && git submodule update --init --recursive && \
        ./autogen.sh && ./configure && make -j8 && make check -j8 && sudo make install && sudo ldconfig && cd
RUN git clone -b 1.2.0 https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp.git && cd rpp && mkdir build && cd build && \
        cmake -DBACKEND=HIP ../ && make -j4 && sudo make install && cd

ENV MIVISIONX_WORKSPACE=/workspace
WORKDIR $MIVISIONX_WORKSPACE

# Install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && \
        mkdir build && cd build && cmake -DBACKEND=HIP ../MIVisionX && make -j8 && make install