#!/usr/bin/env python2

import argparse
import cPickle as pickle
import os
import time
import numpy as np

import eightpoint as epi
import icp
import sfm
import utils


def benchmark_icp(data_dir, max_num=20, outputname='pklfiles/benchmark.pkl',
                  subsamples=[0.1, 0.2, 0.3]):
    arg_parser = setup_argparser()
    all_rms = {}
    for subsample in subsamples:
        for merge_method in ("merge_after", "merge_during"):
            print "\n\nMethod {}, subsample {}".format(merge_method, subsample)
            args = arg_parser.parse_args(['-v', 'icp', data_dir,
                                          merge_method,
                                          '-m', str(max_num),
                                          '-s', str(subsample),
                                          '-n',
                                          '-o', 'pcdfiles/experiment'])
            all_rms[merge_method, subsample] = args.func(args)
    pickle.dump(all_rms, open(outputname, 'wb'))


def icp_main(args):
    '''
    Responsible for collecting pcd files from the given directory,
    and calling ICP according to the selected method.
    '''
    pcd_files = sorted(list(os.path.join(args.data_dir, f)
                            for f in os.listdir(args.data_dir)
                            if f.endswith('.pcd') and not
                            f.endswith('normal.pcd')))[::args.jump]

    if args.verbosity > 0:
        print "Performing Iterative closest point!"
        print "Using method '{}' for merging.".format(args.merge_method)
        print "Subsampling {}% of data".format(args.subsample*100)

    now = time.time()
    merged, all_rms = icp.merge(pcd_files, args)
    time_taken = time.time() - now

    if args.verbosity > 0:
        print "Parsed {} files in {} seconds.".format(args.max, time_taken)
    # Write and/or show output
    if args.output_file:
        output_name = "{}_m{max}_method-{method}_s{subsample}_j{jump}.pcd".\
            format(args.output_file, max=args.max, method=args.merge_method,
                   subsample=min(1.0, max(0.0, args.subsample)),
                   jump=args.jump)
        print "Saved pcd file as '{}'".format(output_name)
        utils.writepcd(output_name, merged)

        if not args.no_visualization:
            print "Opening pclviewer to display results..."
            utils.showpcd(output_name)
    return all_rms, time_taken


def epi_main(args):
    '''
    Call necessary functions for fundamental matrix estimation
    given the correct arguments.
    '''
    data_set = args.data_dir.strip('/').split('/')[-1]
    img_files = sorted(list(os.path.join(args.data_dir, f)
                            for f in os.listdir(args.data_dir)
                            if f.endswith('.png')))[::args.jump]

    num_files = min(args.max, len(img_files))
    img_files = img_files[:num_files]

    if args.verbosity > 0:
        print "Estimating fundamental matrix for dataset {}!".format(data_set)
        print "Using {} eightpoint algorithm".format("standard" if
                                                     not args.normalized
                                                     else "normalized")
        if args.ransac_iterations:
            print ".. with {} RANSAC iterations".format(args.ransac_iterations)
        now = time.time()

    pv_mat = epi.eightpoint(data_set, img_files, args)

    if args.verbosity > 0:
        print "Parsed {} files in {} seconds.".format(len(img_files),
                                                      time.time() - now)

    # By default, save the output using params to name it
    output_file = "{name}{normalized}{ransac_iter}{jump}{max}".format(
        name=args.output_file if args.output_file else "epi",
        normalized="_n" if args.normalized else "",
        ransac_iter="_r{}".format(args.ransac_iterations)
                    if args.ransac_iterations else "",
        jump="_j{}".format(args.jump) if args.jump else "",
        max="_m{}".format(num_files))
    output_file += ".pkl"

    print "Saved points in '{}'".format(output_file) + \
        ", which has size {}".format(pv_mat.shape)

    # True if not np.array([0, 0])
    if args.verbosity > 1:
        view = np.array(np.sum(pv_mat, axis=2) ==
                        np.zeros(pv_mat.shape[:2]),
                        dtype=int)
        # Add a small number to enforce image != 0
        utils.resize_and_display("Pointviewmat", view + 1e-10, 3.0, 3.0)
    pickle.dump(pv_mat, open(output_file, 'wb'))


def sfm_main(args):
    '''
    Call necessary functions to perform (affine) structure from motion.
    '''
    # TODO load pkl files containing inlier featurepoints in images !
    if args.points.endswith('.pkl'):
        try:
            pointviewmat = pickle.load(open(args.points, 'rb'))
        except:
            print "Could not load file '{}'".format(args.points)
            return
    elif args.points.endswith('.txt'):
        pointviewmat = utils.load_pointview_from_txt(args.points)
        try:
            pointviewmat = utils.load_pointview_from_txt(args.points)
        except:
            print "Could not load file '{}'".format(args.points)
            return

    if args.verbosity > 0:
        print "Applying affine sfm to find 3d model, given a set of" +\
            "features per image '{}'".format(args.points)

    sfm.structure_from_motion(pointviewmat, args)


def setup_argparser():
    # plotter.disable()  # Will be enabled by icp_main or epi_main, if needed

    arg_parser = argparse.ArgumentParser(
        description="Implementation of ICP, eightpoint, SfM.",
        epilog="...")
    subparsers = arg_parser.add_subparsers(help='Method to execute:')

    # All methods require a verbosity level, by default 'not verbose'
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
                            "after or during estimation. merge_during is" +
                            "default.")
    # Optional args
    icp_parser.add_argument('-s', '--subsample', type=float, default=1.,
                            help="The proportion of points to sample")
    icp_parser.add_argument('-o', '--output-file', default="icp_merged",
                            help="Name used to save the resulting point cloud")
    icp_parser.add_argument('-m', '--max', type=int, default=1e5,
                            help="Maximum number of scenes to read")
    icp_parser.add_argument('-j', '--jump', type=int, default=1,
                            help="Don't use every image, but only every j'th")
    icp_parser.add_argument('-nv', '--no-visualization', action='store_true',
                            help="Don't display resulting pointcloud")

    # Subparser that handles Epipolar geometry args
    epi_parser = subparsers.add_parser('epi', help='Epipolar geometry' +
                                       ' and fundamental matrix estimation' +
                                       ' (Assignment 2)')
    epi_parser.set_defaults(func=epi_main)
    # Positional args
    epi_parser.add_argument('data_dir',
                            help='Data directory containing images')
    # Optional args
    epi_parser.add_argument('-n', '--normalized', action='store_true',
                            help="Use normalized eightpoint")
    epi_parser.add_argument('-r', '--ransac-iterations', type=int,
                            help="Number of RANSAC iterations (default: " +
                            "do not use RANSAC")
    epi_parser.add_argument('-t', '--threshold', type=float, default=1e-3,
                            help="Threshold for ransac")
    epi_parser.add_argument('-o', '--output-file',
                            help="Name used to save the matches")
    epi_parser.add_argument('-m', '--max', type=int, default=0,
                            help="Maximum number of images to read")
    epi_parser.add_argument('-j', '--jump', type=int, default=1,
                            help="Don't use every image, but only every j'th")

    # Subparser that handles Structure from motion args
    sfm_parser = subparsers.add_parser('sfm', help="Structure from motion" +
                                       "assignment 3")
    sfm_parser.set_defaults(func=sfm_main)
    # Positional args
    sfm_parser.add_argument('points',
                            help='Pkl file containing a pointviewmatrix.')
    # Optional args
    sfm_parser.add_argument('-o', '--output-file', default="",
                            help="Name used to save the point cloud file")
    sfm_parser.add_argument('-nv', '--no-visualization', action='store_true',
                            help="Don't display resulting pointcloud")
    return arg_parser

if __name__ == "__main__":
    arg_parser = setup_argparser()
    args = arg_parser.parse_args()
    args.func(args)
