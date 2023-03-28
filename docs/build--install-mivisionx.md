## Build & Install MIVisionX

### Windows

#### Using .msi packages

* [MIVisionX-installer.msi](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases): MIVisionX
* [MIVisionX_WinML-installer.msi](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/releases): MIVisionX for WinML

#### Using `Visual Studio`

* Install [Windows Prerequisites](#windows)
* Use `MIVisionX.sln` to build for x64 platform

  **NOTE:** `vx_nn` is not supported on `Windows` in this release

### macOS

macOS [build instructions](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/wiki/macOS#macos-build-instructions)

### Linux

#### Using `apt-get` / `yum`

* [ROCm supported hardware](https://docs.amd.com)
* Install [ROCm](https://docs.amd.com)
* On `Ubuntu`
  ```
  sudo apt-get install mivisionx
  ```
* On `CentOS`
  ```
  sudo yum install mivisionx
  ```

  **Note:**
  * `vx_winml` is not supported on `Linux`
  * source code will not available with `apt-get` / `yum` install
  * the installer will copy
    + Executables into `/opt/rocm/bin`
    + Libraries into `/opt/rocm/lib`
    + OpenVX and module header files into `/opt/rocm/include/mivisionx`
    + Model compiler, & toolkit folders into `/opt/rocm/libexec/mivisionx`
    + Apps, & samples folder into `/opt/rocm/share/mivisionx`
    + Docs folder into `/opt/rocm/share/doc/mivisionx`
  * Package (.deb & .rpm) install requires `OpenCV v3+` to execute `AMD OpenCV extensions`

#### Using `MIVisionX-setup.py`

* Install [ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
* Use the below commands to set up and build MIVisionX

  ```
  git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git
  cd MIVisionX
  ```

  **Note:** MIVisionX has support for two GPU backends: **OPENCL** and **HIP**:

  + Instructions for building MIVisionX with the **HIP** GPU backend (i.e., default GPU backend):

    + run the setup script to install all the dependencies required by the **HIP** GPU backend:
    ```
    python MIVisionX-setup.py
    ```

    + run the below commands to build MIVisionX with the **HIP** GPU backend:
    ```
    mkdir build-hip
    cd build-hip
    cmake ../
    make -j8
    sudo make install
    ```

  + Instructions for building MIVisionX with **OPENCL** GPU backend:

    + run the setup script to install all the dependencies required by the **OPENCL** GPU backend:
    ```
    python MIVisionX-setup.py --reinstall ON --backend OCL
    ```

    + run the below commands to build MIVisionX with the **OPENCL** GPU backend:
    ```
    mkdir build-ocl
    cd build-ocl
    cmake -DBACKEND=OPENCL ../
    make -j8
    sudo make install
    ```

  **Note:**
  + MIVisionX cannot be installed for both GPU backends in the same default folder (i.e., /opt/rocm/)
  if an app interested in installing MIVisionX with both GPU backends, then add **-DCMAKE_INSTALL_PREFIX** in the cmake
  commands to install MIVisionX with OPENCL and HIP backends into two separate custom folders.
  + vx_winml is not supported on Linux
