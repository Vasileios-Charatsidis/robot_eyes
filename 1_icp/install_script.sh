#!/bin/bash

## This setup script will install all necessary software for our homework.
## It has been designed with a Ubuntu install in mind.
##
## The script will install the following software:
##   - Cython, from the default apt repository
##   - pcl, from a personal package archive
##   - python-pcl, from source
##   - FLANN with Python bindings, from source
##
## The necessary downloads will be placed in a subdirectory downloads/.

if [ ! -d downloads ]; then
  mkdir downloads/
fi
cd downloads/

# Install Cython, needed for python-pcl
sudo apt-get install cython

# PointCloud Library (binaries for Linux found at http://www.pointclouds.org/downloads/linux.html)
sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
sudo apt-get update
sudo apt-get install libpcl-all

# python-pcl from https://github.com/strawlab/python-pcl/
git clone git@github.com:strawlab/python-pcl.git
cd python-pcl
sudo python setup.py install
cd ..

# Latest version of FLANN with Python bindings, building from sources:
wget www.cs.ubc.ca/research/flann/uploads/FLANN/flann-1.8.4-src.zip
unzip flann-1.8.4-src.zip
mkdir flann-1.8.4-src/build/
cd flann-1.8.4-src/build/
cmake ..
make -j`nproc`
cp ../src/python/pyflann/ -r ./src/python/
cd src/python/
sudo python setup.py install
cd ../../../../
