FROM centos:centos8

# install mivisionx base dependencies - Level 1
RUN yum -y update && yum -y install gcc gcc-c++ kernel-devel make cmake git
# install ROCm for mivisionx OpenCL dependency - Level 2
RUN echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/yum/rpm\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" > \
        /etc/yum.repos.d/rocm.repo && yum -y install rocm-dev