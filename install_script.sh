#!/bin/bash

## This setup script will install all necessary software for our homework.
## It has been designed with a Ubuntu install in mind.
##
## The script will install the following software:
##   - Cython, Boost.Python, NumPy, Matplotlib, from the default apt repository
##   - OpenCV 2.4.9 with Python bindings, from source
##   - pcl, from a personal package archive
##   - python-pcl, from source
##   - FLANN with Python bindings, from source
##   - pyvlfeat, from source
##
## The necessary downloads will be placed in a subdirectory downloads/.

if [ ! -d downloads ]; then
  mkdir downloads/
fi
cd downloads/

# Install packages from default repos:
# - Cython, needed for python-pcl
# - Boost.Python, NumPy, Matplotlib, needed for pyvlfeat
sudo apt-get -y install cython libboost-python-dev python-{numpy,matplotlib}

# OpenCV 2.4.9 (source downloaded from official GitHub repository)
wget https://github.com/Itseez/opencv/archive/2.4.9.tar.gz
tar xzf 2.4.9.tar.gz
mkdir opencv-2.4.9/build/
cd opencv-2.4.9/build/
cmake ..
make -j`nproc`
sudo make install
cd ../.. # Back to downloads/

# PointCloud Library (binaries for Linux found at http://www.pointclouds.org/downloads/linux.html)
sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
sudo apt-get update
sudo apt-get -y install libpcl-all

# python-pcl from https://github.com/strawlab/python-pcl/
git clone git@github.com:strawlab/python-pcl.git
cd python-pcl
sudo python setup.py install
cd .. # Back to downloads/

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
cd ../../../ # Back to downloads/

# Latest version of pyvlfeat
wget https://pypi.python.org/packages/source/p/pyvlfeat/pyvlfeat-0.1.1a3.tar.gz
tar xzf pyvlfeat-0.1.1a3.tar.gz
cd pyvlfeat-0.1.1a3/
# Recent Ubuntu version have a different -lboost_python to link to
sed -i "s/-lboost_python-mt-py26/-lboost_python-py27/" setup.py 
sudo python setup.py install
cd ../ # Back to downloads/
