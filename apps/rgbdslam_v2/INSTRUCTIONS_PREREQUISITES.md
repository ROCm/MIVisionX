# Follow these instructions to install prerequisites and set up the environment for RGBDSLAM_v2:

The following instructions have been taken from the wiki page of the [original implementation of RGBDSLAM_v2](https://github.com/felixendres/rgbdslam_v2/wiki/Instructions-for-Compiling-Rgbdslam-(V2)-on-a-Fresh-Ubuntu-16.04-Install-(Ros-Kinetic)-in-Virtualbox)

## 1. If not using a fresh Ubuntu install, check to make sure you don't already have conflicting version of g2o installed

```bash
dpkg -l | grep [Gg]2[Oo]
ls /usr/local/lib
ls /usr/lib | grep [Gg]2[Oo]
```
Also check /opt/ros/kinetic/lib.

## 2. Install packages that g2o fork will need to work properly

```bash
sudo apt-get install libsuitesparse-dev
```
This will prevent errors mentioning "cholmod" or variations thereof. <br />
**Note:** g2o will compile without doing this, but you'll have the above error later.

## 3. Download and extract eigen 3.2.10 header files -- eigen consists only of header files, no binary libs to compile or link against

```bash
mkdir ~/rgbdslam_deps
cd ~/rgbdslam_deps
wget http://bitbucket.org/eigen/eigen/get/3.2.10.tar.bz2
mkdir eigen
tar -xvjf 3.2.10.tar.bz2 -C eigen --strip-components 1
```
**Note:** No need to make or make install. The header files will be used in place.

## 4. Download g2o fork

```bash
cd ~/rgbdslam_deps
git clone https://github.com/felixendres/g2o.git
cd ~/rgbdslam_deps/g2o
mkdir ~/rgbdslam_deps/g2o/build
cd ~/rgbdslam_deps/g2o/build
```

## 5. Configure g2o fork to use eigen 3.2.10 header files instead of system header files

```bash
vi ~/rgbdslam_deps/g2o/CMakeLists.txt
```
change line 251 from:
```cmake
SET(G2O_EIGEN3_INCLUDE ${EIGEN3_INCLUDE_DIR} CACHE PATH "Directory of Eigen3")
```
to:
```cmake
SET(G2O_EIGEN3_INCLUDE "$ENV{HOME}/rgbdslam_deps/eigen" CACHE PATH "Directory of Eigen3")
```
**Note:** Check to make sure the system eigen3 headers are in /usr/include/eigen3.

## 6. Build & install g2o fork -- this will fail with errors if you're using the system eigen headers.

```bash
cd ~/rgbdslam_deps/g2o/build
cmake ../
make
sudo make install
```
This installs to  /usr/local/lib

## 7. Download PCL 1.8 (instead of using system PCL 1.7)

```bash
cd ~/rgbdslam_deps
wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.0.tar.gz
tar -xvzf pcl-1.8.0.tar.gz
```
## 8. Configure PCL to compile with C++ 2011 support

We need to compile pcl with C++ 2011 support because we're going to be compiling rdbgslam with C++ 2011 support and if we don't, rgbdslam will segfault on startup. 
The PCL 1.7 library that comes installed with Ubuntu 16.04 is not compiled with C++ 2011 support.
```bash
cd ~/rgbdslam_deps/pcl-pcl-1.8.0
vi CMakeLists.txt
```
Add the following to line #146 of CMakeLists.txt (right after endif()):
```cmake
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
```
Save the file and exit (:wq).

## 9. Compile and install PCL

```bash
mkdir build
cd build
cmake ../
```
To ensure that C++ 2011 support is being compiled into PCL, run:
```bash
make VERBOSE=1
```
Copy some compiler output to a text file and search it to see if it contains "-stdc++11".
<br/>
If you can't find -stdc++11 in the output, then you probably inserted it into the wrong place in the CMakeLists.txt. 
If this happens, press ctrl-c and restart the build with just 'make' to clean up the output.
```bash
sudo make install
```
