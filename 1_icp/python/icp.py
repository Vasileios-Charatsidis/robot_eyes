import math
import numpy as np
from pyflann import FLANN
import logging


def icp(source, target, D):
    '''
    Perform ICP for two arrays containing points. Note that these
    arrays must be row-major!
    '''
    N = source.size
    flann = FLANN()

    # init R as identity, t as zero
    R = np.eye(D, dtype='float64')
    t = np.zeros((1, D), dtype='float64')

    centroid_target = np.mean(target, axis=0)
    centroid_source = np.mean(source, axis=0)

    logging.debug("ORIGINAL SOURCE {}".format(centroid_source))
    logging.debug("ORIGINAL_TARGET {}".format(centroid_target))

    # TODO somehow build index beforehand?
    rms = 1
    rms_new = 0

    while rms != rms_new:
        rms = rms_new
        logging.debug("RMS: {}".format(rms))
        # Rotate and translate the source
        transformed_source = np.dot(R, source.T).T + t

        centroid_transformed_source = np.mean(transformed_source, axis=0)
        logging.debug("SOURCE: {}".format(centroid_transformed_source))
        # Use flann to find nearest neighbours. Note that argument order means
        # 'for each transformed_source find the corresponding target'
        results, dists = \
            flann.nn(target, transformed_source, num_neighbours=1,
                     algorithm='kdtree',
                     trees=10, checks=120)
        # Compute new RMS
        #for p1, r, d in zip(transformed_source, results, dists):
        #    print "{} close to {}, dist {}".format(p1, target[r], d)
        rms_new = math.sqrt(sum(dists) / float(N))

        # Use array slicing to get the correct targets
        selected_target = target[results, :]
        centroid_selected_target = np.mean(selected_target, axis=0)
        print "TARGET:", centroid_target

        # Compute covariance, perform SVD
        correlation = np.dot(
            (transformed_source - centroid_transformed_source).T,
            (selected_target - centroid_selected_target))
        u, s, v = np.linalg.svd(correlation)
        R = np.dot(v, u.T)

        # check for righthandedness of the system
        d = np.linalg.det(u) * np.linalg.det(v)
        if d < 0:
            u[:, -1] = -u[:, 1]
        R = np.dot(u, v.T)
        t = np.dot(R, -centroid_source) + centroid_selected_target

        logging.debug("Rotation\n{} \nTranslation\n{}".format(R, t.T))

        raw_input()
    return rms


def rotation_matrix(axis, theta):
    '''
    Given an axis and an angle, create a rotation matrix
    that can be used to rotate vectors around that axis by
    that angle.

    Taken from :
    http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    '''
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2)
    b, c, d = -axis * math.sin(theta / 2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


if __name__ == "__main__":
    R = rotation_matrix(np.array([1, 1, 1]), theta=0.1)
    print R
    pcd1 = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0.5, 0.5, 0]], dtype=float)
    pcd2 = np.dot(R, pcd1) + np.array([[3, 1, 2]])
    icp(pcd1, pcd2, D=3)
