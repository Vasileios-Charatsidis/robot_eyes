import pcl

def readpcd(name):
    p = pcl.PointCloud()
    p.from_file(name)
