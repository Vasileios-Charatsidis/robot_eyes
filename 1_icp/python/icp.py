import math
import numpy as np
from pyflann import FLANN
import sys


def homogenize_transformation(R, t):
    '''
    Create a homogeneous transformation matrix given a rotation and
    translation.
    '''
    T = np.hstack((R, t.T))
    T = np.vstack((T, np.array([0 for i in xrange(len(R))] + [1])))
    return T


def dehomogenize_transformation(T):
    '''
    Derive R and t from a homogeneous transformation matrix.
    '''
    size = T.shape[0] - 1
    R = T[:size, :size]
    t = T[:size, size]
    return R, t


def icp(source, target, D, debug=0, epsilon=0.00001):
    '''
    Perform ICP for two arrays containing points. Note that these
    arrays must be row-major!

    NOTE:
    This function returns the rotation matrix, translation for a
    transition FROM target TO source. This approach was chosen because
    of computational efficiency: it is now possible to index the source
    points beforehand, and query the index for matches from the target.

    In other words: Each new point gets matched to an old point. This is
    quite intuitive, as the set of source points may be (much) larger.
    '''
    flann = FLANN()

    # source_h = np.hstack((source, np.ones((source.shape[0], 1))))
    # unused: target_h = np.hstack((target, np.ones((target.shape[0], 1)))).T

    # init R as identity, t as zero
    R = np.eye(D, dtype='float64')
    t = np.zeros((1, D), dtype='float64')
    T = homogenize_transformation(R, t)
    transformed_target = target

    # centroid_target = np.mean(target, axis=0)
    # centroid_source = np.mean(source, axis=0)

    # Build index beforehand for faster querying
    flann.build_index(source, algorithm='kdtree', trees=10)

    # Initialize rms to bs values
    rms = 2
    rms_new = 1

    while True:
        # Update root mean squared error
        rms = rms_new

        # Rotate and translate the target using homogeneous coordinates
        # unused: transformed_target = np.dot(T, target_h).T[:, :D]
        transformed_target = np.dot(R, transformed_target.T).T + t
        centroid_transformed_target = np.mean(transformed_target, axis=0)

        # Use flann to find nearest neighbours. Note that because of index it
        # means 'for each transformed_target find the corresponding source'
        results, dists = flann.nn_index(transformed_target, num_neighbors=1,
                                        checks=120)

        # Compute new RMS
        rms_new = math.sqrt(sum(dists) / float(len(dists)))

        # Give feedback if necessary
        if debug > 0:
            sys.stdout.write("\rRMS: {}".format(rms_new))
            sys.stdout.flush()
            if debug > 1:
                sys.stdout.write("\nTransformation:\n{}\n".format(T))
                sys.stdout.flush()
        assert rms > rms_new, "RMS was not minimized?"

        # Check threshold
        if rms - rms_new < epsilon:
            break

        # Use array slicing to get the correct targets
        selected_source = source[results, :]
        centroid_selected_source = np.mean(selected_source, axis=0)

        # Compute covariance, perform SVD using Kabsch algorithm
        correlation = np.dot(
            (transformed_target - centroid_transformed_target).T,
            (selected_source - centroid_selected_source))
        u, s, v = np.linalg.svd(correlation)

        # u . S . v = correlation =
        # V . S . W.T

        # TODO needed?
        # ensure righthandedness coordinate system and calculate R
        d = np.linalg.det(np.dot(v, u.T))
        sign_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
        R = np.dot(np.dot(v.T, sign_matrix), u.T)
        t[0, :] = np.dot(R, -centroid_transformed_target) + \
            centroid_selected_source

        # Combine transformations so far with new found R and t
        # Note: Latest transformation should be on the inside of the equation
        T = np.dot(T, homogenize_transformation(R, t))

        if debug > 2:
            try:
                if raw_input() == "q":
                    sys.exit(0)
            except EOFError:
                print("")
                sys.exit(0)
            except KeyboardInterrupt:
                print("")
                sys.exit(0)

    # Unpack the built transformation matrix
    R, t = dehomogenize_transformation(T)
    return R, t, rms_new


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
    icp(pcd1, pcd2, D=3, debug=3)
