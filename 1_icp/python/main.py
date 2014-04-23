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
    return np.array(p.to_array(), dtype='float64')


def writepcd(name, array):
    p = pcl.PointCloud()
    p.from_array(np.array(array, dtype='float32'))
    p.to_file(name)


def main(input_dir, method, maximum, debug):
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
        now = time.time()

    merged = eval("{}(pcd_files, maximum, debug)".format(method))
    if debug > 0:
        print "Parsed {} files in {} seconds.".format(maximum,
                                                      time.time() - now)
    writepcd("merged.pcd", merged)


def merge_after(pcd_files, max_scenes, debug):
    '''
    Estimate rotation translation for every consecutive frame. Use
    the estimates to create one set of points, by transforming each
    newfound set towards the last coordinate set.

    Note that the error for each estimation will then also propagate.
    '''
    f1 = readpcd(pcd_files[0])

    merged = np.zeros((0, 3))

    for file_id, pcd_file in enumerate(pcd_files[1:]):
        if debug > 0:
            print "Estimating R, t from {} to {}".format(
                pcd_files[file_id], pcd_file)

        # max number of scenes reached
        if file_id == max_scenes:
            break
        f2 = readpcd(pcd_file)

        # f1 and f2 are now numpy arrays waiting to be used
        R, t, rms, transformed_f2 = icp.icp(f1, f2, D=3, debug=debug,
                                            return_transformed_target=True)

        # Add the transformed set of points to the total set
        merged = np.vstack((merged, transformed_f1))

        # Move to the next scene
        f1 = f2
    return merged


def merge_during(pcd_files, max_scenes, debug):
    '''
    Instead of estimating transformation matrix from every frame
    to its consecutive frame, we merge the scenes during the process,
    and find matches for the entire merged set.

    This will take much longer than the 'regular' process, since the
    target set will grow every iteration. Should we check the target
    for correspondences in the source instead?
    '''
    merged = readpcd(pcd_files[0])

    for file_id, pcd_file in enumerate(pcd_files[1:]):
        if debug > 0:
            print "Estimating R, t from scenes up to {}, to {}".\
                format(pcd_file[file_id], pcd_file)

        # max number of scenes reached
        if file_id == max_scenes:
            break

        f2 = readpcd(pcd_file)
        # f1 and f2 are now numpy arrays waiting to be used
        R, t, rms = icp.icp(merged, f2, D=3, debug=1)

        # Transform f1 given R and t
        transformed_merged = np.dot(R, merged.T).T + t
        # Add the transformed set of points to the total set
        merged = np.vstack((transformed_merged, f2))
    return merged


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Implementation of ICP.",
        epilog="...")
    # Location args
    arg_parser.add_argument('directory', help='Input directory')
    arg_parser.add_argument('merge_method',
                            choices=('merge_after', 'merge_during'),
                            help="Choose whether merges take place " +\
                            "after or during estimation")
    # Optional args
    arg_parser.add_argument('-max', '--maximum', type=int, default=2,
                            help="Maximum number of scenes to read")
    arg_parser.add_argument('-d', '--debug', type=int,
                            default=0, help="Set debug level, " +
                            "0: silent, 1: verbose, 2: very verbose")
    arg_parser.add_argument('-nv', '--no-visualization', action='store_true',
                            help="Don't display the visualization")
    args = arg_parser.parse_args()

    main(input_dir=args.directory,
         method=args.merge_method,
         maximum=args.maximum,
         debug=args.debug,
         )
    name = "merged.pcd"
    if not args.no_visualization:
        subprocess.Popen(["pcl_viewer", name]).wait()
