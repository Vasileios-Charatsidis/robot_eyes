from pyflann import FLANN
import numpy

class NN:
    def __init__(self, n_trees=3, default_checks=30):
        self.__n_trees = n_trees
        self.__default_checks = default_checks
        self.__FLANNs = []
        self.__clouds = []

    def add(self, points):
        self.__clouds.append(points)
        flann = FLANN()
        flann.build_index(points, algorithm='kdtree', trees=self.__n_trees)
        self.__FLANNs.append(flann)

    def match(self, points, checks=default_checks):
        results = [None] * points.shape[0]
        dists = numpy.repeat(numpy.infty, points.shape[0])
        for j, flann in enumerate(self.__FLANNs):
            re, di = flann.nn_index(points, num_neighbors=1, checks=checks)
            for i, (r, d) in enumerate(zip(re, di)):
                if d < dists[i]:
                    dists[i] = d
                    results[i] = (j, r)
        return results, dists

    def get(self, indices):
        result = []
        for cloud, index in indices:
            result.append(self.__clouds[cloud][index])
        return numpy.array(result)


if __name__ == "__main__":
    points_a = numpy.array([[1, 2], [3, 4]], dtype=numpy.float64)
    points_b = numpy.array([[2, 2], [3, 3]], dtype=numpy.float64)
    points_c = numpy.array([[0, 0], [1, 1]], dtype=numpy.float64)
    points_d = numpy.array([[2, 1], [3, 3], [0, 0]], dtype=numpy.float64)
    knn = KNN()
    knn.add(points_a)
    knn.add(points_b)
    knn.add(points_c)
    r, _ = knn.match(points_d)
    print r
    print knn.get(r)
