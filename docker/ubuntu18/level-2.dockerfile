FROM ubuntu:18.04

RUN apt-get update -y && apt-get upgrade -y && apt-get dist-upgrade -y
RUN apt-get -y install gcc g++ cmake pkg-config git
RUN apt-get -y install libnuma-dev wget sudo gnupg2 kmod &&  \
        wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - && \
        echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list && \
        sudo apt-get update -y && \
        sudo apt-get -y install rocm-dev