FROM ubuntu:18.04

RUN apt-get update -y
# install mivisionx base dependencies - Level 1
RUN apt-get -y install gcc g++ cmake git