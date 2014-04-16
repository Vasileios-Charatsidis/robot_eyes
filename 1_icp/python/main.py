import argparse
import icp
import os
from utils import pairwise

def readpcd(name):
    p = pcl.PointCloud()
    p.from_file(name)


def main(input_dir, verbose):
    '''
    Responsible for reading pcd files from the given directory,
    and calling ICP accordingly.
    '''
    pcd_files = sorted(list(f for f in os.listdir(input_dir)
                            if f.endswith('.pcd') and not
                            f.endswith('normal.pcd')))
    f1 = readpcd(pcd_files[0])
    for pcd_file in pcd_files[1:]:
        f2 = readpcd(pcd_file)

        # f1 and f2 are now numpy arrays waiting to be used



        f1 = f2


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser(
        description="Implementation of ICP.",
        epilog="...")
    arg_parser.add_argument('directory', help='Input directory')
    arg_parser.add_argument('-max', '--maximum', type=int, default=2,
                            help="Maximum number of files read")
    arg_parser.add_argument('-v', '--verbose', action='store_const',
                            const=True, default=False, help="Verbosity")
    args = arg_parser.parse_args()
    main(input_dir=args.directory,
         verbose=args.verbose)
