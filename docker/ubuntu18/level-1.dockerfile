FROM ubuntu:18.04

RUN apt-get update -y && apt-get upgrade -y && apt-get dist-upgrade -y
RUN apt-get -y install gcc g++ cmake pkg-config git