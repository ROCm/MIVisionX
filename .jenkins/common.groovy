// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'

    String osInfo = ''
    String update = ''
    String installPackage = ''
    String cmake = ''

    if (platform.jenkinsLabel.contains('centos')) {
        osInfo = 'cat /etc/os-release && uname -r'
        update = 'sudo yum -y update'
        if (platform.jenkinsLabel.contains('centos7')) {
            update = 'echo scl enable devtoolset-7 bash | sudo tee /etc/profile.d/ree.sh && sudo chmod +x /etc/profile.d/ree.sh && . /etc/profile && scl enable devtoolset-7 bash && sudo yum -y update'
        }
        installPackage = 'python MIVisionX-setup.py --reinstall yes --installer yum --ffmpeg yes'
        cmake = 'cmake3'
    }
    else if (platform.jenkinsLabel.contains('sles')) {
        osInfo = 'cat /etc/os-release && uname -r'
        update = 'sudo zypper --non-interactive ref && sudo zypper --non-interactive update && sudo zypper --non-interactive refresh'
        installPackage = 'python MIVisionX-setup.py --reinstall yes --installer "zypper --non-interactive" --ffmpeg yes'
        cmake = 'cmake'
    }
    else {
        osInfo = 'cat /etc/lsb-release && uname -r'
        update = 'sudo apt -y update && sudo apt -y install python'
        installPackage = 'python MIVisionX-setup.py --reinstall yes --ffmpeg yes'
        cmake = 'cmake'
    }

    def command = """#!/usr/bin/env bash
                set -x
                ${osInfo}
                ${update}
                echo Install MIVisionX Prerequisites
                cd ${project.paths.project_build_prefix}
                ${installPackage}
                echo Build MIVisionX - ${buildTypeDir}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${cmake} ${buildTypeArg} ../..
                make -j\$(nproc)
                sudo make install
                sudo make package
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project) {

    String conformaceCPU = 'echo OpenVX 1.3 Conformance - CPU - NOT TESTED ON THIS PLATFORM'
    String conformaceGPU = 'echo OpenVX 1.3 Conformance - GPU - NOT TESTED ON THIS PLATFORM'
    String moveFiles = ''

    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('ubuntu')) {
        conformaceCPU = 'AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-Conformance-log.md'
        conformaceGPU = 'AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance --filter=-HarrisCorners.*:-*.ReplicateNode:-*.ImageContainmentRelationship:-*.OnRandomAndNatural:-*.vxWeightedAverage:-vxCanny.*:-*.MapRandomRemap:*.* | tee OpenVX-GPU-Conformance-log.md'
        moveFiles = 'mv *.md ../../'
    }

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                python ../../tests/vision_tests/runVisionTests.py --runvx_directory ./bin --hardware_mode CPU --num_frames 100
                python ../../tests/vision_tests/runVisionTests.py --runvx_directory ./bin --hardware_mode GPU --num_frames 100 --backend_type OCL
                python ../../tests/neural_network_tests/runNeuralNetworkTests.py
                export OPENVX_DIR=\$(pwd)/.
                export OPENVX_INC=\$(pwd)/../../amd_openvx/openvx
                mkdir conformance_tests
                cd conformance_tests
                git clone -b openvx_1.3 https://github.com/KhronosGroup/OpenVX-cts.git
                export VX_TEST_DATA_PATH=\$(pwd)/OpenVX-cts/test_data/
                mkdir build-cts
                cd build-cts
                cmake -DOPENVX_INCLUDES=\$OPENVX_INC/include -DOPENVX_LIBRARIES=\$OPENVX_DIR/lib/libopenvx.so\\;\$OPENVX_DIR/lib/libvxu.so\\;pthread\\;dl\\;m\\;rt -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts
                cmake --build .
                echo MIVisionX OpenVX 1.3 Conformance - CPU
                ${conformaceCPU}
                echo MIVisionX OpenVX 1.3 Conformance - GPU - OpenCL
                ${conformaceGPU}
                ${moveFiles}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/*.md""")
}

def runPackageCommand(platform, project) {
    def packageHelper = platform.makePackage(platform.jenkinsLabel, "${project.paths.project_build_prefix}/build/release")

    String packageType = ''
    String packageInfo = ''

    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles')) {
        packageType = 'rpm'
        packageInfo = 'rpm -qlp'
    }
    else {
        packageType = 'deb'
        packageInfo = 'dpkg -c'
    }

    def command = """#!/usr/bin/env bash
                set -x
                export HOME=/home/jenkins
                echo Make MIVisionX Package
                cd ${project.paths.project_build_prefix}/build/release
                sudo make package
                mkdir -p package
                mv *.${packageType} package/
                ${packageInfo} package/*.${packageType}
                python ../../tests/library_tests/runLibraryTests.py
                mv *.md package/
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, packageHelper[1])
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.md""")
}

return this
