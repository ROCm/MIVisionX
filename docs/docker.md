## Docker

MIVisionX provides developers with docker images for **Ubuntu** `20.04` and **CentOS** `7` / `8`. Using docker images developers can quickly prototype and build applications without having to be locked into a single system setup or lose valuable time figuring out the dependencies of the underlying software.

Docker files to build MIVisionX containers are [available](docker/README.md)

### MIVisionX Docker

* [Ubuntu 20.04](https://hub.docker.com/r/mivisionx/ubuntu-20.04)
* [CentOS 7](https://hub.docker.com/r/mivisionx/centos-7)
* [CentOS 8](https://hub.docker.com/r/mivisionx/centos-8)

### Docker Workflow Sample on Ubuntu `20.04`

#### Prerequisites

* Ubuntu `20.04`/`22.04`
* [rocm supported hardware](https://docs.amd.com)

#### Workflow

* Step 1 - *Install rocm-dkms*

```
sudo apt update -y
sudo apt dist-upgrade -y
sudo apt install libnuma-dev wget
sudo reboot
```

```
wget https://repo.radeon.com/amdgpu-install/21.50/ubuntu/focal/amdgpu-install_21.50.50000-1_all.deb
sudo apt-get install -y ./amdgpu-install_21.50.50000-1_all.deb
sudo apt-get update -y
sudo amdgpu-install -y --usecase=rocm
sudo reboot
```

* Step 2 - *Setup Docker*

```
sudo apt-get install curl
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install -y docker-ce
sudo systemctl status docker
```

* Step 3 - *Get Docker Image*

```
sudo docker pull mivisionx/ubuntu-20.04
```

* Step 4 - *Run the docker image*

```
sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/ubuntu-20.04:latest
```
  **Note:**
  * Map host directory on the docker image

    + map the localhost directory to be accessed on the docker image.
    + use `-v` option with docker run command: `-v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH}`
    + usage:
    ```
    sudo docker run -it -v /home/:/root/hostDrive/ --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host mivisionx/ubuntu-20.04:latest
    ```

  * Display option with docker
    + Using host display
    ```
    xhost +local:root
    sudo docker run -it --device=/dev/kfd --device=/dev/dri --cap-add=SYS_RAWIO --device=/dev/mem --group-add video --network host --env DISPLAY=unix$DISPLAY --privileged --volume $XAUTH:/root/.Xauthority --volume /tmp/.X11-unix/:/tmp/.X11-unix mivisionx/ubuntu-20.04:latest
    ```

    + Test display with MIVisionX sample
    ```
    export PATH=$PATH:/opt/rocm/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
    runvx /opt/rocm/share/mivisionx/samples/gdf/canny.gdf
    ```
