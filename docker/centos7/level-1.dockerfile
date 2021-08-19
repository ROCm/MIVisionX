FROM centos:centos7

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

# install mivisionx base dependencies - Level 1
RUN yum -y update --nogpgcheck && yum -y install --nogpgcheck http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm && \
        yum -y install --nogpgcheck gcc gcc-c++ kernel-devel make && yum -y install --nogpgcheck cmake3 && yum -y install --nogpgcheck git

WORKDIR /workspace