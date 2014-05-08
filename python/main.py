#!/usr/bin/env python2

import argparse
import icp
import eightpoint as epi
import os
import subprocess
import time


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
    icp.writepcd(name, merged)

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
    img_files = sorted(list(os.path.join(args.data_dir, f)
                            for f in os.listdir(args.data_dir)
                            if f.endswith('.png')))
    if args.verbosity > 0:
        print "Estimating fundamental matrix!"
        print "Using {} eightpoint algorithm".format("standard" if
                                                     not args.normalized
                                                     else "normalized")
        if args.ransac_iterations:
            print "Using {} RANSAC iterations".format(args.ransac_iterations)
        now = time.time()

    epi.eightpoint(img_files, args.normalized, args.ransac_iterations,
                   args.verbosity)
    # TODO epi.chaining ?

    if args.verbosity > 0:
        print "Parsed {} files in {} seconds.".format(args.max,
                                                      time.time() - now)


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
    icp_parser.add_argument('-o', '--output-file', default= "merged.pcd",
                            help="Save the point cloud file")

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
    epi_parser.add_argument('-m', '--max', type=int, default=2,
                            help="Maximum number of images to read")

    args = arg_parser.parse_args()
    args.func(args)
