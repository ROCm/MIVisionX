FROM centos:centos7

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

# install mivisionx base dependencies - Level 1
RUN yum -y update --nogpgcheck && yum -y install --nogpgcheck http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm && \
        yum -y install --nogpgcheck gcc gcc-c++ kernel-devel make && yum -y install --nogpgcheck cmake3 && yum -y install --nogpgcheck git
# install ROCm for mivisionx OpenCL dependency - Level 2
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms && yum -y install --nogpgcheck centos-release-scl && yum -y install --nogpgcheck devtoolset-7 && \
        echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/yum/rpm\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" > \
        /etc/yum.repos.d/rocm.repo && yum -y install --nogpgcheck rocm-dev
# Enable Developer Toolset 7
SHELL [ "/usr/bin/scl", "enable", "devtoolset-7" ]

WORKDIR /workspace