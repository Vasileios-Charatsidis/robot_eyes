#!/usr/bin/env python2

import argparse
import icp
import os
import pcl
import numpy as np
import subprocess
import time
import math
from pyflann import FLANN


def readpcd(name):
    p = pcl.PointCloud()
    p.from_file(name)
    return filter_vecs(np.array(p.to_array(), dtype='float64'))


def filter_vecs(vectors, distance=1):
    """Remove all points that are more distant than the given distance."""
    return np.array([v for v in vectors if not v[-1] > distance])


def writepcd(name, array):
    p = pcl.PointCloud()
    p.from_array(np.array(array, dtype='float32'))
    p.to_file(name)


def main(input_dir, method, maximum, subsample_size, debug):
    '''
    Responsible for collecting pcd files from the given directory,
    and calling ICP according to the selected method.
    '''
    pcd_files = sorted(list(os.path.join(input_dir, f)
                            for f in os.listdir(input_dir)
                            if f.endswith('.pcd') and not
                            f.endswith('normal.pcd')))
    if debug > 0:
        print "Using method '{}' for merging.".format(method)
        print "Subsampling {}% of data".format(subsample_size*100)
        now = time.time()

    merged = merge(pcd_files, method, maximum, subsample_size, debug)
    if debug > 0:
        print "Parsed {} files in {} seconds.".format(maximum,
                                                      time.time() - now)
    name = "merged.pcd"
    writepcd(name, merged)
    return name


def compute_rms(source, target):
    '''
    Make a single call to FLANN rms.
    '''
    flann = FLANN()
    results, dists = flann.nn(source, target, algorithm='kdtree', trees=10,
                              checks=120, num_neighbors=1)
    return math.sqrt(sum(dists) / float(len(dists)))


def iter_pcds(file_names, subsample_size, max_scenes):
    for file_id, file_name in enumerate(file_names):
        # max number of scenes reached
        if file_id == max_scenes:
            break

        all = readpcd(file_name)
        sample = subsample(all, subsample_size) if \
            subsample_size < 1 else all

        yield file_id, sample, all


def merge(pcd_files, method, max_scenes, subsample_size, debug):
    '''
    Estimate rotation translation for every consecutive frame. Use
    the estimates to create one set of points, by transforming each
    newfound set towards the last coordinate set.

    Note that the error for each estimation will then also propagate.

    Instead of estimating transformation matrix from every frame
    to its consecutive frame, we merge the scenes during the process,
    and find matches for the entire merged set.

    This will take much longer than the 'regular' process, since the
    target set will grow every iteration. Should we check the target
    for correspondences in the source instead?
    '''

    f1 = readpcd(pcd_files[0])
    # Initialize merged as the points in the first frame
    merged = f1    # Is by reference, but f1 is never altered so that's okay

    for file_id, f2, f2_all in iter_pcds(pcd_files[1:], subsample_size, max_scenes):
        if debug > 0:
            print "Estimating R, t from {} to {}".format(file_id, file_id + 1)

        # f1 and f2 are now numpy arrays waiting to be used
        if method == 'merge_after':
            R, t, rms_subsample = icp.icp(f1, f2, D=3, debug=debug)
        elif method == 'merge_during':
            R, t, rms_subsample = icp.icp(merged, f2, D=3, debug=debug)

        # Transform f2 to merged given R and t
        transformed_f2 = np.dot(R, f2_all.T).T + t
        # Compute rms for this scene transitions, for the whole set
        if subsample_size < 1:
            rms = compute_rms(merged, transformed_f2)
        else:
            rms = rms_subsample

        if debug > 0:
            print "\rRMS for the whole scene:", rms

        # Add the transformed set of points to the total set
        merged = np.vstack((merged, transformed_f2))

        # Move to the next scene (not necessary for merge_during)
        f1 = f2
    return merged


def subsample(vectors, proportion=.15):
    '''
    Return a view of a matrix representing a random subsample of input
    vectors. Note that the original matrix is shuffled.
    '''
    l = len(vectors)
    assert l > 2
    np.random.shuffle(vectors)
    num = max(2, int(proportion * l))  # We can only match at least 2 points
    return vectors[:num]


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Implementation of ICP.",
        epilog="...")
    # Location args
    arg_parser.add_argument('data_dir', help='Data directory')
    arg_parser.add_argument('merge_method',
                            choices=('merge_after', 'merge_during'),
                            help="Choose whether merges take place " +
                            "after or during estimation")
    # Optional args
    arg_parser.add_argument('-max', '--maximum', type=int, default=2,
                            help="Maximum number of scenes to read")
    arg_parser.add_argument('-s', '--subsample', type=float, default=1.,
                            help="The proportion of points to sample")
    arg_parser.add_argument('-d', '--debug', type=int,
                            default=0, help="Set debug level, " +
                            "0: silent, 1: verbose, 2: very verbose")
    arg_parser.add_argument('-nv', '--no-visualization', action='store_true',
                            help="Don't display the visualization")
    args = arg_parser.parse_args()

    name = main(input_dir=args.data_dir,
                method=args.merge_method,
                maximum=args.maximum,
                subsample_size=args.subsample,
                debug=args.debug,
                )
    if not args.no_visualization:
        print "Opening pclviewer to display results..."
        subprocess.Popen(["pcl_viewer", name],
                         stdout=open(os.devnull, 'w')).wait()
