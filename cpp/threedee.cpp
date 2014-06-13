#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/surface/poisson.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Use: " << argv[0] << " input.pcd output.vtk" << std::endl;
        return 0;
    }

    pcl::PCLPointCloud2::Ptr pcd_data(new pcl::PCLPointCloud2);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>());
    pcl::Poisson<pcl::PointNormal> model;
    pcl::PolygonMesh polygons;

    // Read data from files
    pcl::io::loadPCDFile(argv[1], *pcd_data);
    pcl::fromPCLPointCloud2(*pcd_data, *cloud);

    // Decide on model parameters
    model.setDepth(12);
    model.setSolverDivide(8);
    model.setIsoDivide(8);
    model.setPointWeight(4.0f);
    model.setScale(1.5);
    model.setInputCloud(cloud);

    model.performReconstruction(polygons);
    pcl::io::saveVTKFile(argv[2], polygons);

    return EXIT_SUCCESS;
}
