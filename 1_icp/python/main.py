import logging
import argparse
import icp
import os
import pcl
import numpy as np


def readpcd(name):
    p = pcl.PointCloud()
    p.from_file(name)
    return np.array(p.to_array(), dtype='float64')


def writepcd(name, array):
    p = pcl.PointCloud()
    p.from_array(np.array(array, dtype='float32'))
    p.to_file(name)


def main(input_dir, maximum, debug):
    '''
    Responsible for reading pcd files from the given directory,
    and calling ICP accordingly.
    '''
    pcd_files = sorted(list(os.path.join(input_dir, f)
                            for f in os.listdir(input_dir)
                            if f.endswith('.pcd') and not
                            f.endswith('normal.pcd')))
    f1 = readpcd(pcd_files[0])

    # To keep track of all rotation/translation/rms
    merged = np.zeros((0, 3))

    for file_id, pcd_file in enumerate(pcd_files[1:]):
        # max number of scenes reached
        if debug > 0:
            print "Estimating R, t from {} to {}".format(
                pcd_file[file_id], pcd_file)

        if file_id == maximum:
            break

        f2 = readpcd(pcd_file)
        # f1 and f2 are now numpy arrays waiting to be used
        R, t, rms = icp.icp(f1, f2, D=3, debug=1)

        # Transform stuff
        transformed_f1 = np.dot(R, f1.T).T + t
        merged = np.vstack((merged, transformed_f1))

        # Move to the next scene
        f1 = f2

    writepcd("merged.pcd", merged)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Implementation of ICP.",
        epilog="...")
    arg_parser.add_argument('directory', help='Input directory')
    arg_parser.add_argument('-max', '--maximum', type=int, default=2,
                            help="Maximum number of files read")
    arg_parser.add_argument('-d', '--debug', type=int,
                            default=0, help="Verbosity")
    args = arg_parser.parse_args()

    #logger = logging.getLogger('ICP_logger')
    #logging.basicConfig(level=(logging.DEBUG if args.verbose
    #                           else logging.WARNING),
    #                    format='%(levelname)s\t %(message)s')
    main(input_dir=args.directory,
         maximum=args.maximum,
         debug=args.debug)
