#!/usr/bin/env python2

import argparse
import os
import subprocess
import time

import eightpoint as epi
import icp
import sfm
import utils

from pretty_plotter import plotter


def icp_main(args):
    '''
    Responsible for collecting pcd files from the given directory,
    and calling ICP according to the selected method.
    '''
    pcd_files = sorted(list(os.path.join(args.data_dir, f)
                            for f in os.listdir(args.data_dir)
                            if f.endswith('.pcd') and not
                            f.endswith('normal.pcd')))

    plot_dir = args.plot_dir
    if None:
        plotter.disable()
    else:
        plotter.enable()
        plotter.output_dir(plot_dir)

    if args.verbosity > 0:
        print "Performing Iterative closest point!"
        print "Using method '{}' for merging.".format(args.merge_method)
        print "Subsampling {}% of data".format(args.subsample*100)
        now = time.time()

    merged = icp.merge(pcd_files, args.merge_method, args.max, args.subsample,
                       debug=args.verbosity)
    if args.verbosity > 0:
        print "Parsed {} files in {} seconds.".format(args.max,
                                                      time.time() - now)

    name = args.output_file
    utils.writepcd(name, merged)

    if not args.no_visualization:
        print "Opening pclviewer to display results..."
        subprocess.Popen(["pcl_viewer", name],
                         stdout=open(os.devnull, 'w')).wait()

    if not args.output_file:
        subprocess.call(["rm", name])


def epi_main(args):
    '''
    Call necessary functions for fundamental matrix estimation
    given the correct arguments.
    '''
    data_set = args.data_dir.strip('/').split('/')[-1]
    img_files = sorted(list(os.path.join(args.data_dir, f)
                            for f in os.listdir(args.data_dir)
                            if f.endswith('.png')))[::args.jump]

    if args.max:
        num_files = len(img_files)
        if args.max < num_files:
            img_files = img_files[:args.max+1]
        elif args.verbosity > 0:
            print "Warning: {} only has {} files.".format(data_set, num_files)

    if args.verbosity > 0:
        print "Estimating fundamental matrix for dataset {}!".format(data_set)
        print "Using {} eightpoint algorithm".format("standard" if
                                                     not args.normalized
                                                     else "normalized")
        if args.ransac_iterations:
            print ".. with {} RANSAC iterations".format(args.ransac_iterations)
        now = time.time()

    epi.eightpoint(img_files, args.normalized, args.ransac_iterations,
                   data_set, args.output_file, threshold=args.threshold,
                   verbose=args.verbosity)
    # TODO epi.chaining ?

    if args.verbosity > 0:
        print "Parsed {} files in {} seconds.".format(len(img_files),
                                                      time.time() - now)


def sfm_main(args):
    '''
    Call necessary functions to perform (affine) structure from motion.
    '''
    # TODO load pkl files containing inlier featurepoints in images !
    try:
        import cPickle as pickle
        pointviewmat = pickle.load(open(args.points, 'rb'))
    except:
        print "Could not load file '{}'".format(args.points)
        return

    if args.verbosity > 0:
        print "Applying affine sfm to find 3d model, given a set of" +\
            "features per image '{}'".format(args.points)

    sfm.structure_from_motion(pointviewmat, args.verbosity)


if __name__ == "__main__":
    plotter.disable()  # Will be enabled by icp_main or epi_main, if needed

    arg_parser = argparse.ArgumentParser(
        description="Implementation of ICP.",
        epilog="...")
    subparsers = arg_parser.add_subparsers(help='Method to execute:')

    # All methods require a data dir containing pcd/img files
    arg_parser.add_argument('-v', '--verbosity', action='count',
                            help="Set verbosity level, " +
                            "default: silent, -v: verbose, -vv: very verbose")

    # Subparser that handles ICP arguments
    icp_parser = subparsers.add_parser('icp', help='Iterative closest point' +
                                       ' (assignment 1)')
    icp_parser.set_defaults(func=icp_main)
    # Positional args
    icp_parser.add_argument('data_dir',
                            help='Data directory containing pcd files')
    icp_parser.add_argument('merge_method', default='merge_during', nargs='?',
                            choices=('merge_after', 'merge_during'),
                            help="Choose whether merges take place " +
                            "after or during estimation")
    # Optional args
    icp_parser.add_argument('-m', '--max', type=int, default=0,
                            help="Maximum number of scenes to read")
    icp_parser.add_argument('-s', '--subsample', type=float, default=1.,
                            help="The proportion of points to sample")
    icp_parser.add_argument('-n', '--no-visualization', action='store_true',
                            help="Don't display resulting pointcloud")
    icp_parser.add_argument('-o', '--output-file', default="merged.pcd",
                            help="Save the point cloud file")
    icp_parser.add_argument('-p', '--plot-dir', default=None,
                            help="Directory to store all plots in")

    # Subparser that handles Epipolar geometry args
    epi_parser = subparsers.add_parser('epi', help='Epipolar geometry' +
                                       ' and fundamental matrix estimation' +
                                       ' (Assignment 2)')
    epi_parser.set_defaults(func=epi_main)
    epi_parser.add_argument('data_dir',
                            help='Data directory containing images')
    epi_parser.add_argument('-n', '--normalized', action='store_true',
                            help="Use normalized eightpoint")
    epi_parser.add_argument('-r', '--ransac-iterations', type=int, default=0,
                            help="Number of RANSAC iterations (default: " +
                            "do not use RANSAC")
    epi_parser.add_argument('-m', '--max', type=int, default=0,
                            help="Maximum number of images to read")
    epi_parser.add_argument('-t', '--threshold', type=float, default=1e-3,
                            help="Threshold for ransac")
    epi_parser.add_argument('-o', '--output-file', default="pointviewmat.pkl",
                            help="Save the matches")
    epi_parser.add_argument('-j', '--jump', type=int, default=1,
                            help="Don't use every image, but only every j'th")

    # Subparser that handles Structuer from motion args
    sfm_parser = subparsers.add_parser('sfm', help="Structure from motion" +
                                       "assignment 3")
    sfm_parser.set_defaults(func=sfm_main)
    # Positional args
    sfm_parser.add_argument('points',
                            help='Pkl file containing found features.')
    # Optional args
    sfm_parser.add_argument('-o', '--output-file', default="",
                            help="Save the point cloud file")

    args = arg_parser.parse_args()
    args.func(args)
