import math
import numpy as np
from nn import NN
import sys
import utils


def iter_pcds(file_names, subsample_size, max_scenes):
    """
    return an iterable of all pcd files, from which we
    take a sample.
    """
    for file_id, file_name in enumerate(file_names):
        # max number of scenes reached
        if file_id == max_scenes:
            break
        all_points = utils.readpcd(file_name)
        sample = utils.subsample(all_points, subsample_size) if \
            subsample_size < 1 else all_points
        yield file_id, sample, all_points


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


def compute_rms(source, target, nn=None):
    '''
    Make a single call to FLANN rms.

    If a flannobject with prebuilt index is given, use that,
    otherwise, do a full search.
    '''
    if nn:
        results, dists = nn.match(target)
    else:
        nn = NN()
        nn.add(source)
        results, dists = nn.match(target)
    return math.sqrt(sum(dists) / float(len(dists)))


def merge(pcd_files, args):
    '''
    Estimate rotation translation for every consecutive frame. Use
    the estimates to create one set of points, by transforming each
    newfound set towards the last coordinate set.

    Note that the error for each estimation will then also propagate.

    Instead of estimating transformation matrix from every frame
    to its consecutive frame, we merge the scenes during the process,
    and find matches for the entire merged set.

    This will take much longer than the 'regular' process, since the
    target set will grow every iteration.
    '''
    f1 = utils.readpcd(pcd_files[0])
    # Initialize merged as the points in the first frame
    merged = f1    # Is by reference, but f1 is never altered so that's okay

    # Homogeneous transformation matrix
    T_c = homogenize_transformation(np.eye(3), np.zeros((1, 3)))

    method = args.merge_method
    all_rms = []

    for file_id, f2, f2_all in iter_pcds(pcd_files[1:],
                                         subsample_size=args.subsample,
                                         max_scenes=args.max):
        if args.verbosity > 0:
            print "Estimating R, t from {} to {}".format(file_id + args.jump,
                                                         file_id)

        # Transform f2 by all previous transformations
        R_c, t_c = dehomogenize_transformation(T_c)
        f2 = np.dot(R_c, f2.T).T + t_c

        # f1 and f2 are now numpy arrays waiting to be used
        if method == 'merge_after':
            R, t, T, rms_subsample, nn_idx = \
                icp(f1, f2, verbosity=args.verbosity)
        elif method == 'merge_during':
            R, t, T, rms_subsample, nn_idx = \
                icp(merged, f2, verbosity=args.verbosity)

        # Transform f2 to merged given R and t
        transformed_f2 = np.dot(R, f2_all.T).T + t

        # Compute rms for this scene transitions, for the whole set
        if args.subsample < 1:
            rms = compute_rms(merged, transformed_f2, nn_idx)
        else:
            rms = rms_subsample
        all_rms.append(rms)

        if args.verbosity:
            print "\rRMS for the whole scene:", rms

        # Add the transformed set of points to the total set
        merged = np.vstack((merged, transformed_f2))

        # Calc cumulative transformation matrix for f2 -> 1 movement
        T_c = np.dot(T, T_c)

        # Move to the next scene (not necessary for merge_during)
        f1 = f2

    print "Total error", sum(all_rms)

    return merged, all_rms


def icp(source, target, D=3, verbosity=0, epsilon=0.000001):
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
    nn = NN()

    # init R as identity, t as zero
    R = np.eye(D, dtype='float64')
    t = np.zeros((1, D), dtype='float64')
    T = homogenize_transformation(R, t)
    transformed_target = target

    # Build index beforehand for faster querying
    nn.add(source)

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
        results, dists = nn.match(transformed_target)

        # Compute new RMS
        rms_new = math.sqrt(sum(dists) / float(len(dists)))

        # Give feedback if necessary
        if verbosity > 0:
            sys.stdout.write("\rRMS: {}".format(rms_new))
            sys.stdout.flush()

        # We find this case some times, but are not sure if it should be
        # possible. Is it possible for the RMS of a (sub)set of points to
        # increase?
        # assert rms > rms_new, "RMS was not minimized?"

        # Check threshold
        if rms - rms_new < epsilon:
            break

        # Use array slicing to get the correct targets
        selected_source = nn.get(results)
        centroid_selected_source = np.mean(selected_source, axis=0)

        # Compute covariance, perform SVD using Kabsch algorithm
        correlation = np.dot(
            (transformed_target - centroid_transformed_target).T,
            (selected_source - centroid_selected_source))
        u, s, v = np.linalg.svd(correlation)

        # u . S . v = correlation =
        # V . S . W.T

        # Ensure righthandedness coordinate system and calculate R
        d = np.linalg.det(np.dot(v, u.T))
        sign_matrix = np.eye(D)
        sign_matrix[D-1, D-1] = d
        R = np.dot(np.dot(v.T, sign_matrix), u.T)
        t[0, :] = np.dot(R, -centroid_transformed_target) + \
            centroid_selected_source

        # Combine transformations so far with new found R and t
        # Note: Latest transformation should be on inside (r) of the equation
        T = np.dot(T, homogenize_transformation(R, t))

        if verbosity > 2:
            try:
                if raw_input("Enter 'q' to quit, or anything else to" +
                             "continue") == "q":
                    sys.exit(0)
            except EOFError:
                print("")
                sys.exit(0)
            except KeyboardInterrupt:
                print("")
                sys.exit(0)

    # Unpack the built transformation matrix
    R, t = dehomogenize_transformation(T)
    return R, t, T, rms_new, nn
