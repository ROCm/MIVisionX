FROM centos:centos7

# install mivisionx base dependencies - Level 1
RUN yum -y update && yum -y install http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm && \
        yum -y install gcc gcc-c++ kernel-devel make && yum -y install cmake3 && yum -y install git
# install ROCm for mivisionx OpenCL dependency - Level 2
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms && yum -y install centos-release-scl && yum -y install devtoolset-7 && \
        echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/yum/rpm\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" > \
        /etc/yum.repos.d/rocm.repo && yum -y install rocm-dev
# Enable Developer Toolset 7
SHELL [ "/usr/bin/scl", "enable", "devtoolset-7" ]

WORKDIR /workspace