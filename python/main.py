#!/usr/bin/env python2

import argparse
import icp
import os
import pcl
import numpy as np
import subprocess
import time


def readpcd(name):
    p = pcl.PointCloud()
    p.from_file(name)
    return filter_vecs(np.array(p.to_array(), dtype='float64'))


def writepcd(name, array):
    p = pcl.PointCloud()
    p.from_array(np.array(array, dtype='float32'))
    p.to_file(name)


def filter_vecs(vectors, distance=1):
    """Remove all points that are more distant than the given distance."""
    return np.array([v for v in vectors if not v[-1] > distance])


def iter_pcds(file_names, subsample_size, max_scenes):
    for file_id, file_name in enumerate(file_names):
        # max number of scenes reached
        if file_id == max_scenes:
            break

        all = readpcd(file_name)
        sample = subsample(all, subsample_size) if subsample_size < 1 else all
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

    for file_id, f2, f2_all in iter_pcds(pcd_files[1:], subsample_size,
                                         max_scenes):
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
            rms = icp.compute_rms(merged, transformed_f2)
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


def icp_main(args):
    '''
    Responsible for collecting pcd files from the given directory,
    and calling ICP according to the selected method.
    '''
    pcd_files = sorted(list(os.path.join(args.data_dir, f)
                            for f in os.listdir(args.data_dir)
                            if f.endswith('.pcd') and not
                            f.endswith('normal.pcd')))
    if args.verbosity > 0:
        print "Using method '{}' for merging.".format(args.merge_method)
        print "Subsampling {}% of data".format(args.subsample*100)
        now = time.time()

    merged = merge(pcd_files, args.merge_method, args.max, args.subsample,
                   debug=args.verbosity)
    if args.verbosity > 0:
        print "Parsed {} files in {} seconds.".format(args.max,
                                                      time.time() - now)
    # Currently hardcoded, TODO make argument?
    name = "merged.pcd"
    writepcd(name, merged)

    if not args.no_visualization:
        print "Opening pclviewer to display results..."
        subprocess.Popen(["pcl_viewer", name],
                         stdout=open(os.devnull, 'w')).wait()

    if not args.keep_file:
        subprocess.call(["rm", name])


def epi_main(args):
    '''
    Call necessary functions for fundamental matrix estimation
    given the correct arguments.
    '''
    pass


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Implementation of ICP.",
        epilog="...")
    subparsers = arg_parser.add_subparsers(help='Method to execute:')

    # All methods require a data folder containing pcd/img files
    arg_parser.add_argument('-v', '--verbosity', type=int,
                            default=0, help="Set verbosity level, " +
                            "0: silent, 1: verbose, 2: very verbose")

    # Subparser that handles ICP arguments
    icp_parser = subparsers.add_parser('icp', help='Iterative closest point' +
                                       ' (assignment 1)')
    icp_parser.set_defaults(func=icp_main)
    # Location args
    icp_parser.add_argument('data_dir',
                            help='Data directory containing pcd files')
    icp_parser.add_argument('merge_method', default='merge_during', nargs='?',
                            choices=('merge_after', 'merge_during'),
                            help="Choose whether merges take place " +
                            "after or during estimation")
    # Optional args
    icp_parser.add_argument('-m', '--max', type=int, default=2,
                            help="Maximum number of scenes to read")
    icp_parser.add_argument('-s', '--subsample', type=float, default=1.,
                            help="The proportion of points to sample")
    icp_parser.add_argument('-n', '--no-visualization', action='store_true',
                            help="Don't display resulting pointcloud")
    icp_parser.add_argument('-k', '--keep-file',
                            help="Save the point cloud file")

    # Subparser that handles Epipolar geometry args
    epi_parser = subparsers.add_parser('epi', help='2. Epipolar geometry' +
                                       ' and fundamental matrix estimation' +
                                       ' (Assignment 2)')
    epi_parser.set_defaults(func=epi_main)
    epi_parser.add_argument('data_dir',
                            help='Data directory containing images')
    epi_parser.add_argument('-m', '--max', type=int, default=2,
                            help="Maximum number of images to read")

    args = arg_parser.parse_args()
    args.func(args)
