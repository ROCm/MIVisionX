FROM centos:centos7

# install mivisionx base dependencies - Level 1
RUN yum -y update && yum -y install http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm && \
        yum -y install gcc gcc-c++ kernel-devel make && yum -y install cmake3 && yum -y install git