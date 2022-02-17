FROM centos:centos7

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

# install mivisionx base dependencies - Level 1
RUN yum -y update --nogpgcheck && yum -y install --nogpgcheck http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm && \
        yum -y install --nogpgcheck gcc gcc-c++ kernel-devel make cmake3 git && yum-config-manager --enable rhel-server-rhscl-7-rpms && \
        yum -y install --nogpgcheck centos-release-scl && yum -y install --nogpgcheck devtoolset-7
# Enable Developer Toolset 7
SHELL [ "/usr/bin/scl", "enable", "devtoolset-7" ]
# install ROCm for mivisionx OpenCL dependency - Level 2
RUN echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/yum/rpm\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" > \
        /etc/yum.repos.d/rocm.repo && yum -y install --nogpgcheck rocm-dev
# install OpenCV & FFMPEG - Level 3
RUN yum -y groupinstall 'Development Tools' --nogpgcheck && yum -y install --nogpgcheck gtk2-devel libjpeg-devel libpng-devel libtiff-devel libavc1394 wget unzip && \
        mkdir opencv && cd opencv && wget https://github.com/opencv/opencv/archive/4.5.5.zip && unzip 4.5.5.zip && \
        mkdir build && cd build && \
        cmake3 -DWITH_OPENCL=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_VA_INTEL=OFF -DWITH_OPENCL_SVM=OFF ../opencv-4.5.5 && \
        make -j8 && make install
RUN yum -y install --nogpgcheck autoconf automake bzip2 bzip2-devel cmake freetype-devel gcc gcc-c++ git libtool make pkgconfig zlib-devel && \
        yum -y install --nogpgcheck nasm && yum -y --enablerepo=extras install --nogpgcheck epel-release && yum -y install --nogpgcheck yasm && \
        yum -y install --nogpgcheck libx264-devel libx265-devel && \
        yum -y install --nogpgcheck https://forensics.cert.org/cert-forensics-tools-release-el7.rpm && yum -y --enablerepo=forensics install --nogpgcheck fdk-aac && \
        yum -y install --nogpgcheck libass-devel && \
        export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/" && \
        wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.0.4.zip && unzip n4.0.4.zip && cd FFmpeg-n4.0.4/ && \
        ./configure --enable-shared --disable-static --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libass --enable-gpl --enable-nonfree && \
        make -j8 && make install

WORKDIR /workspace

# install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && mkdir build && cd build && \
        cmake3 ../MIVisionX && make -j8 && make install