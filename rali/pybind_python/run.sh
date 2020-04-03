rm -r /usr/local/lib/python3.6/dist-packages/amd_rali-1.1.0-py3.6-linux-x86_64.egg
rm -r ./amd_rali.egg-info/
rm -r ./build
rm -r ./dist
python3.6 setup.py build
python3.6 setup.py install
