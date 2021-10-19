// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-D CMAKE_BUILD_TYPE=Debug' : '-D CMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'

    String osInfo = ''
    String update = ''
    String installPackageDeps = ''
    String cmake = 'cmake'
    String codeCovFlags = ''
    String installPrefixHIP = '-D CMAKE_INSTALL_PREFIX=/opt/rocm/mivisionx/hip'
    String installPrefixOCL = ''

    if (platform.jenkinsLabel.contains('centos')) {
        osInfo = 'cat /etc/os-release && uname -r'
        update = 'sudo yum -y --nogpgcheck update && sudo yum -y --nogpgcheck install lcov zip'
        installPackageDeps = 'python MIVisionX-setup.py --reinstall yes --ffmpeg yes'
        if (platform.jenkinsLabel.contains('centos7')) {
            update = 'echo scl enable devtoolset-7 bash | sudo tee /etc/profile.d/ree.sh && sudo chmod +x /etc/profile.d/ree.sh && . /etc/profile && scl enable devtoolset-7 bash && sudo yum -y --nogpgcheck install lcov zip && sudo yum -y --nogpgcheck update'
            cmake = 'cmake3'
            codeCovFlags = '-D CMAKE_CXX_FLAGS="-fprofile-arcs -ftest-coverage"'
        }
        else {
            installPackageDeps = 'python MIVisionX-setup.py --reinstall yes --ffmpeg yes --backend HIP'
            installPrefixOCL = '-D CMAKE_INSTALL_PREFIX=/opt/rocm/mivisionx/OCL'
            installPrefixHIP = ''
        }
    }
    else if (platform.jenkinsLabel.contains('sles')) {
        osInfo = 'cat /etc/os-release && uname -r'
        update = 'sudo zypper -n --no-gpg-checks install lcov zip && sudo zypper -n --no-gpg-checks update'
        installPackageDeps = 'python MIVisionX-setup.py --reinstall yes --ffmpeg yes'
    }
    else if (platform.jenkinsLabel.contains('ubuntu')) {
        osInfo = 'cat /etc/lsb-release && uname -r'
        update = 'sudo apt-get -y --allow-unauthenticated update && sudo apt-get -y --allow-unauthenticated install lcov zip'
        installPackageDeps = 'python MIVisionX-setup.py --reinstall yes --ffmpeg yes'
        if (platform.jenkinsLabel.contains('ubuntu18')) {
            codeCovFlags = '-D CMAKE_CXX_FLAGS="-fprofile-arcs -ftest-coverage"'
        }
        else {
           installPackageDeps = 'python MIVisionX-setup.py --reinstall yes --ffmpeg yes --backend HIP'
           installPrefixOCL = '-D CMAKE_INSTALL_PREFIX=/opt/rocm/mivisionx/OCL'
           installPrefixHIP = ''
        }
    }

    def command = """#!/usr/bin/env bash
                set -x
                ${osInfo}
                ${update}
                echo Install MIVisionX Prerequisites
                cd ${project.paths.project_build_prefix}
                ${installPackageDeps}
                echo Build MIVisionX OpenCL - ${buildTypeDir}
                mkdir -p build/${buildTypeDir}-opencl && cd build/${buildTypeDir}-opencl
                ${cmake} ${buildTypeArg} ${codeCovFlags} ${installPrefixOCL} -D BACKEND=OCL ../..
                make -j\$(nproc)
                sudo make package
                sudo make install
                cd ../
                echo Build MIVisionX HIP - ${buildTypeDir}
                mkdir -p ${buildTypeDir}-hip && cd ${buildTypeDir}-hip
                ${cmake} ${buildTypeArg} ${codeCovFlags} ${installPrefixHIP} -D BACKEND=HIP ../..
                make -j\$(nproc)
                sudo make package
                sudo make install
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project) {

    String conformaceCPU_OCL = 'echo OpenVX 1.3 Conformance - CPU with OCL Backend Build - NOT TESTED ON THIS PLATFORM'
    String conformaceCPU_HIP = 'echo OpenVX 1.3 Conformance - CPU with HIP Backend Build - NOT TESTED ON THIS PLATFORM'
    String conformaceOpenCL = 'echo OpenVX 1.3 Conformance - GPU OpenCL - NOT TESTED ON THIS PLATFORM'
    String conformaceHIP = 'echo OpenVX 1.3 Conformance - GPU HIP - NOT TESTED ON THIS PLATFORM'
    String moveFiles = ''
    String platformOS = ''
    String captureCodeCovOCL = ''
    String captureCodeCovHIP = ''
    String codeCovExcludeOCL = ''
    String codeCovExcludeHIP = ''
    String codeCovListOCL = ''
    String codeCovListHIP = ''
    String codeCovPackageOCL = 'echo Code Coverage - NOT Supported ON THIS PLATFORM'
    String codeCovPackageHIP = 'echo Code Coverage - NOT Supported ON THIS PLATFORM'
    String nnTestsHIP = 'echo NN TESTS -  Backend set to OCL'
    String nnTestsOCL = 'sudo python ../../tests/neural_network_tests/runNeuralNetworkTests.py --backend_type OCL'

    if (platform.jenkinsLabel.contains('centos7')) {
        platformOS = 'centos7'
    }
    else if (platform.jenkinsLabel.contains('centos8')) {
        platformOS = 'centos8'
    }
    else if (platform.jenkinsLabel.contains('ubuntu18')) {
        platformOS = 'ubuntu18'
    }
    else if (platform.jenkinsLabel.contains('ubuntu20')) {
        platformOS = 'ubuntu20'
    }
    else if (platform.jenkinsLabel.contains('sles')) {
        platformOS = 'sles'
    }

    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('ubuntu')) {
        conformaceCPU_OCL = "AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-OCL-${platformOS}.md"
        conformaceOpenCL = "AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance  | tee OpenVX-GPU-CTS-OCL-${platformOS}.md"
        conformaceCPU_HIP = "AGO_DEFAULT_TARGET=CPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-CPU-CTS-HIP-${platformOS}.md"
        conformaceHIP = "AGO_DEFAULT_TARGET=GPU LD_LIBRARY_PATH=./lib ./bin/vx_test_conformance | tee OpenVX-GPU-CTS-HIP-${platformOS}.md"
        moveFiles = 'mv *.md ../../'
        if (platform.jenkinsLabel.contains('centos7') || platform.jenkinsLabel.contains('ubuntu18')) {
            captureCodeCovOCL = "lcov --directory . --capture --output-file ocl-coverage-${platformOS}.info"
            captureCodeCovHIP = "lcov --directory . --capture --output-file hip-coverage-${platformOS}.info"
            codeCovExcludeOCL = "lcov --remove ocl-coverage-${platformOS}.info '/usr/*' '*/runvx/*' --output-file ocl-coverage-${platformOS}.info"
            codeCovExcludeHIP = "lcov --remove hip-coverage-${platformOS}.info '/usr/*' '*/runvx/*' --output-file hip-coverage-${platformOS}.info"
            codeCovListOCL = "lcov --list ocl-coverage-${platformOS}.info | tee ocl-coverage-info-${platformOS}.md"
            codeCovListHIP = "lcov --list hip-coverage-${platformOS}.info | tee hip-coverage-info-${platformOS}.md"
            codeCovPackageOCL = "genhtml ocl-coverage-${platformOS}.info --output-directory coverage-${platformOS} && zip -r ocl-coverage-info-${platformOS}.zip coverage-${platformOS}"
            codeCovPackageHIP = "genhtml hip-coverage-${platformOS}.info --output-directory coverage-${platformOS} && zip -r hip-coverage-info-${platformOS}.zip coverage-${platformOS}"
        }
        else {
            nnTestsHIP = 'sudo python ../../tests/neural_network_tests/runNeuralNetworkTests.py --backend_type HIP'
            nnTestsOCL = 'echo NN TESTS -  Backend set to HIP'
        }
    }

    def command = """#!/usr/bin/env bash
                set -x
                echo MIVisionX - with OpenCL Tests
                cd ${project.paths.project_build_prefix}/build/release-opencl
                python ../../tests/vision_tests/runVisionTests.py --runvx_directory ./bin --hardware_mode CPU --num_frames 100
                python ../../tests/vision_tests/runVisionTests.py --runvx_directory ./bin --hardware_mode GPU --num_frames 100 --backend_type OCL
                ${nnTestsOCL}
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
                echo MIVisionX OpenVX 1.3 Conformance - CPU - OCL Backend Build
                ${conformaceCPU_OCL}
                echo MIVisionX OpenVX 1.3 Conformance - GPU - OpenCL
                ${conformaceOpenCL}
                ${moveFiles}
                echo MIVisionX OCL Backend - Code Coverage Info
                cd ../../
                ${captureCodeCovOCL}
                ${codeCovExcludeOCL}
                ${codeCovListOCL}
                ${codeCovPackageOCL}
                echo MIVisionX - with HIP Tests
                cd ../release-hip
                python ../../tests/vision_tests/runVisionTests.py --runvx_directory ./bin --hardware_mode CPU --num_frames 100
                python ../../tests/vision_tests/runVisionTests.py --runvx_directory ./bin --hardware_mode GPU --num_frames 100 --backend_type HIP
                ${nnTestsHIP}
                export OPENVX_DIR=\$(pwd)/.
                export OPENVX_INC=\$(pwd)/../../amd_openvx/openvx
                mkdir conformance_tests
                cd conformance_tests
                git clone -b openvx_1.3 https://github.com/KhronosGroup/OpenVX-cts.git
                export VX_TEST_DATA_PATH=\$(pwd)/OpenVX-cts/test_data/
                mkdir build-cts
                cd build-cts
                cmake -DOPENVX_INCLUDES=\$OPENVX_INC/include -DOPENVX_LIBRARIES=\$OPENVX_DIR/lib/libopenvx.so\\;\$OPENVX_DIR/lib/libopenvx_hip.so\\;/opt/rocm/hip/lib/libamdhip64.so\\;pthread\\;dl\\;m\\;rt -DOPENVX_CONFORMANCE_VISION=ON ../OpenVX-cts
                cmake --build .
                echo MIVisionX OpenVX 1.3 Conformance - CPU - HIP Backend Build
                ${conformaceCPU_HIP}
                echo MIVisionX OpenVX 1.3 Conformance - GPU - HIP
                ${conformaceHIP}
                ${moveFiles}
                echo MIVisionX HIP Backend - Code Coverage Info
                cd ../../
                ${captureCodeCovHIP}
                ${codeCovExcludeHIP}
                ${codeCovListHIP}
                ${codeCovPackageHIP}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release-opencl/*.md""")
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release-hip/*.md""")
    if (platform.jenkinsLabel.contains('centos7') || platform.jenkinsLabel.contains('ubuntu18')) {
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release-opencl/*.zip""")
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release-hip/*.zip""")
    }
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
                echo Make MIVisionX Package - with OpenCL support
                cd ${project.paths.project_build_prefix}/build/release-opencl
                sudo make package
                mkdir -p package
                mv *.${packageType} package/
                python ../../tests/library_tests/runLibraryTests.py --install_directory ./ --backend_type OCL
                mv *.md package/
                ${packageInfo} package/*.${packageType}
                echo Make MIVisionX Package - with HIP support
                cd ../release-hip
                sudo make package
                (for file in *.${packageType}; do mv "\$file" "HIP-\$file"; done;)
                mkdir -p package
                mv *.${packageType} package/
                python ../../tests/library_tests/runLibraryTests.py --install_directory ./ --backend_type HIP
                mv *.md package/
                ${packageInfo} package/*.${packageType}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release-opencl/package/*.md""")
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release-opencl/package/*.${packageType}""")
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release-hip/package/*.md""")
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release-hip/package/*.${packageType}""")
}

return this
