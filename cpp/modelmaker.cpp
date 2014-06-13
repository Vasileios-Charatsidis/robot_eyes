#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/surface/poisson.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Use: " << argv[0] << " input.pcd [output.vtk]" << std::endl;
        return 0;
    }
    std::string input_pcd = argv[1];
    std::string output_vtk = "/tmp/kokkooijmanwiggers.vtk";
    if (argc >= 3) {
      output_vtk = argv[2];
    }

    pcl::PCLPointCloud2::Ptr pcd_data(new pcl::PCLPointCloud2);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>());
    pcl::Poisson<pcl::PointNormal> model;
    pcl::PolygonMesh polygons;

    // Read data from files
    pcl::io::loadPCDFile(input_pcd, *pcd_data);
    pcl::fromPCLPointCloud2(*pcd_data, *cloud);

    // Decide on model parameters
    model.setDepth(12);
    model.setSolverDivide(8);
    model.setIsoDivide(8);
    model.setPointWeight(4.0f);
    model.setScale(1.5);
    model.setInputCloud(cloud);

    model.performReconstruction(polygons);
    //pcl::io::saveVTKFile(output_vtk, polygons);
    pcl::io::saveVTKFile(output_vtk, polygons);

    FILE * f = popen(("pcl_viewer " + output_vtk).c_str(), "r");
    if (f == 0) {
        fprintf(stderr, "Could not execute\n");
        return EXIT_FAILURE;
    }
    pclose(f);

    return EXIT_SUCCESS;
}
