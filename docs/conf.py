# MIT License

# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from rocm_docs import ROCmDocs


external_projects_current_project = "mivisionx"

breathe_projects_source = {
    "amd_openvx": (
        "../amd_openvx/openvx/include",
        [
            "vx_ext_amd.h",
        ],
    ),
    "khronos_openvx": (
        "../amd_openvx/openvx/include/VX",
        [
            "vx_api.h",
            "vx_compatibility.h",
            "vx.h",
            "vx_import.h",
            "vx_kernels.h",
            "vx_khr_buffer_aliasing.h",
            "vx_khr_class.h",
            "vx_khr_icd.h",
            "vx_khr_import_kernel.h",
            "vx_khr_ix.h",
            "vx_khr_nn.h",
            "vx_khr_opencl_interop.h",
            "vx_khr_pipelining.h",
            "vx_khr_user_data_object.h",
            "vx_khr_xml.h",
            "vx_nodes.h",
            "vx_types.h",
            "vxu.h",
            "vx_vendors.h",
        ],
    ),
    "amd_custom": (
        "../amd_openvx_extensions/amd_custom/include",
        ["vx_amd_custom.h"],
    ),
    "amd_media": (
        "../amd_openvx_extensions/amd_media/include",
        ["vx_amd_media.h"],
    ),
    "amd_migraphx": (
        "../amd_openvx_extensions/amd_migraphx/include",
        ["vx_amd_migraphx.h"],
    ),
    "amd_nn": (
        "../amd_openvx_extensions/amd_nn/include",
        ["vx_amd_nn.h"],
    ),
    "amd_opencv": (
        "../amd_openvx_extensions/amd_opencv/include",
        [
            "internal_opencvTunnel.h",
            "internal_publishKernels.h",
            "vx_opencv.h",
            "vx_ext_opencv.h",
        ],
    ),
    "amd_rpp": (
        "../amd_openvx_extensions/amd_rpp/include",
        [
            "internal_publishKernels.h",
            "internal_rpp.h",
            "kernels_rpp.h",
            "vx_ext_rpp.h",
        ],
    ),
    "amd_winml": (
        "../amd_openvx_extensions/amd_winml/include",
        [
            "internal_publishKernels.h",
            "internal_winmlTunnel.h",
            "vx_ext_winml.h",
            "vx_winml.h",
        ],
    ),
    "bubble_pop": (
        "../apps/bubble_pop/include",
        [
            "internal_opencvTunnel.h",
            "internal_publishKernels.h",
            "vx_ext_pop.h",
            "vx_pop.h",
        ],
    ),
    "cloud_inference": (
        "../apps/cloud_inference/server_app/include",
        [
            "arguments.h",
            "common.h",
            "compiler.h",
            "configure.h",
            "infcom.h",
            "inference.h",
            "netutil.h",
            "profiler.h",
            "region.h",
            "server.h",
            "shadow.h",
        ],
    ),
    "dg_test": (
        "../apps/dg_test/include",
        [
            "annmodule.h",
            "cvui.h",
            "DGtest.h",
            "UserInterface.h",
        ],
    ),
    "mivisionx_openvx_classifier": (
        "../apps/mivisionx_openvx_classifier/include",
        [
            "caffeModels.h",
            "cvui.h",
        ],
    ),
    "model_compiler_samples": (
        "../samples/model_compiler_samples/include",
        [
            "classification.h",
            "common.h",
            "cvui.h",
            "detection.h",
            "segmentation.h",
        ],
    ),
}

os.system('find ../ -name "*.md" > "docfiles.txt"')
doc_files = open("docfiles.txt", "r")
lines = doc_files.readlines()
for file_path in lines:
    file_dir, _ = os.path.split(file_path)
    print(f"mkdir -p {file_dir[1:]}")
    os.system(f"mkdir -p {file_dir[1:]}")
    print(f"cp {file_path[:-1]} {file_path[1:]}")
    os.system(f"cp {file_path[:-1]} {file_path[1:]}")

docs_core = ROCmDocs("MIVisionX Documentation")
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.setup()
docs_core.myst_heading_anchors = 6

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
