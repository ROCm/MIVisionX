FROM centos:centos8

# install mivisionx base dependencies - Level 1
RUN yum -y update && yum -y install gcc gcc-c++ kernel-devel make cmake git

WORKDIR /workspace