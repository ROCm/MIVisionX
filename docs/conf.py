# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from rocm_docs import ROCmDocs

os.system('find ../ -name "*.md" > "docfiles.txt"')
doc_files = open('docfiles.txt', 'r')
lines = doc_files.readlines()
for file_path in lines:
    file_dir, _ = os.path.split(file_path)
    print(f"mkdir -p {file_dir[1:]}")
    os.system(f"mkdir -p {file_dir[1:]}")
    print(f"cp {file_path[:-1]} {file_path[1:]}")
    os.system(f"sudo cp {file_path[:-1]} {file_path[1:]}")

docs_core = ROCmDocs("MIVisionX Documentation")
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
