//////////// FILES INCLUDED///////////////////////



================= Prerequisites =================
PointCloud Library (binaries for linux found at http://www.pointclouds.org/downloads/linux.html). As root, do:
    
    add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
    apt-get update
    apt-get install libpcl-all


python-pcl from https://github.com/strawlab/python-pcl/:

    git clone git@github.com:strawlab/python-pcl.git
    cd python-pcl

as root, do:

    python setup.py install 

     
FLANN (with python bindings), building the latest from source:

    curl www.cs.ubc.ca/research/flann/uploads/FLANN/flann-1.8.4-src.zip
    unzip flann-1.8.4-src.zip
    cd flann-1.8.4-src
    mkdir build
    cd build
    cmake ..
    make 
    cp ../src/python/pyflann/ -r ./src/python
    cd src/python
    
as root, do:

    python setup.py install


