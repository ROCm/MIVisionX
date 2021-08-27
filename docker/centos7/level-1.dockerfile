FROM centos:centos7

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

# install mivisionx base dependencies - Level 1
RUN yum -y update --nogpgcheck && yum -y install --nogpgcheck http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm && \
        yum -y install --nogpgcheck gcc gcc-c++ kernel-devel make cmake3 git && yum-config-manager --enable rhel-server-rhscl-7-rpms && \
        yum -y install --nogpgcheck centos-release-scl && yum -y install --nogpgcheck devtoolset-7
# Enable Developer Toolset 7
SHELL [ "/usr/bin/scl", "enable", "devtoolset-7" ]

WORKDIR /workspace

# install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && mkdir build && cd build && \
        cmake3 ../MIVisionX && make -j8 && make install