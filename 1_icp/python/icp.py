import math
import numpy as np
from pyflann import FLANN
import sys


def icp(source, target, D, debug=0, epsilon=0.001,
        return_transformed_target=False):
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
    N = source.size
    flann = FLANN()

    # init R as identity, t as zero
    R = np.eye(D, dtype='float64')
    t = np.zeros((1, D), dtype='float64')

    centroid_target = np.mean(target, axis=0)
    # centroid_source = np.mean(source, axis=0)

    # Build index beforehand for faster querying
    flann.build_index(source, algorithm='kdtree', trees=10)

    # Initialize rms to bs values
    rms = 1000
    rms_new = 1

    # TODO max_iterations?
    while True:
        rms = rms_new
        if debug > 0:
            sys.stdout.write("\rRMS: {}".format(rms))
            sys.stdout.flush()
            if debug > 1:
                sys.stdout.write("\nRotation:\n{}\n".format(R))
                sys.stdout.flush()

        # Rotate and translate the target
        transformed_target = np.dot(R, target.T).T + t
        centroid_transformed_target = np.mean(transformed_target, axis=0)

        # Use flann to find nearest neighbours. Note that because of index it
        # means 'for each transformed_target find the corresponding source'
        results, dists = flann.nn_index(transformed_target, num_neighbours=1,
                                        checks=120)

        # Compute new RMS
        rms_new = math.sqrt(sum(dists) / float(N))
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

        # ensure righthandedness coordinate system and calculate R
        d = np.linalg.det(np.dot(v, u.T))
        sign_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
        R = np.dot(np.dot(v.T, sign_matrix), u.T)

        t[0, :] = np.dot(R, -centroid_target) + centroid_selected_source

        if debug > 2:
            try:
                l = raw_input()
                if l == "q":
                    sys.exit(0)
            except EOFError:
                print("")
                sys.exit(0)
            except KeyboardInterrupt:
                print("")
                sys.exit(0)
    if debug > 0:
        print ''

    # We've already computed the
    if return_transformed_target:
        return R, t, rms, return_transformed_target
    return R, t, rms



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
