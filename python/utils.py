import pcl
import os
import numpy as np
from itertools import tee, izip
from subprocess import Popen
import cv2


def resize_and_display(window_name, image, resize_x, resize_y):
    '''
    Just as the name says: Resize and display an image using cv2.
    '''
    view = cv2.resize(image, (0, 0), fx=resize_x, fy=resize_y)
    cv2.imshow(window_name, view)
    cv2.waitKey()


def readpcd(name):
    p = pcl.PointCloud()
    p.from_file(name)
    return filter_vecs(np.array(p.to_array(), dtype='float64'))


def writepcd(name, array):
    p = pcl.PointCloud()
    p.from_array(np.array(array, dtype='float32'))
    p.to_file(name)


def showpcd(name, pixelsize=20, color=(255, 255, 200)):
    Popen(["pcl_viewer",
           '-ps', '{}'.format(pixelsize),
           '-pc', '{},{},{}'.format(*color),
           name],
          stdout=open(os.devnull, 'w')).wait()


def load_pointview_from_txt(filename):
    '''Pointviewmatrix is 2m x n'''
    M = sum(1 for _ in open(filename, 'r').readlines())
    N = len(open(filename, 'r').readline().split())

    pointviewmat = np.zeros((M, N))

    for m, line in enumerate(open(filename, 'r').readlines()):
        numbers = np.array([float(num) for num in line.split()])
        pointviewmat[m, :] = numbers

    return pointviewmat


def filter_vecs(vectors, distance=1):
    """Remove all points that are more distant than the given distance."""
    return np.array([v for v in vectors if not v[-1] > distance])


def subsample(vectors, proportion=.15, num=None):
    '''
    Return a view of a matrix representing a random subsample of input
    vectors. Note that the original matrix is shuffled.
    '''
    np.random.shuffle(vectors)
    if not num:
        l = len(vectors)
        assert l > 2
        # We can only match at least 2 points
        num = max(2, int(proportion * l))
    return vectors[:num]


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


# nearpd from
# http://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-
# positive-semi-definite-matrix
def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T


def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return W05.I * _getAplus(W05 * A * W05) * W05.I


def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)


def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n)
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk
