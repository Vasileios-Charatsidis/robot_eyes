from itertools import izip
import numpy as np
from itertools import tee


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
