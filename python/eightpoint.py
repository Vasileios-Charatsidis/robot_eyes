import numpy as np
import cv





# 1. Detect interest points in both images
# 2. Characterize local appearance of the regions around interest points
# 3. Get a set of supposed matches
# 4. Estimate fundamental matrix

def fundamental(matches1, matches2):
    #A = np.array([[match[0][0] * match[1][0],
    #               match[0][0] * match[1][1],
    #               match[0][0],
    #               match[0][1] * match[1][0],
    #               match[0][1],
    #               match[1][0],
    #               match[1][1],
    #               1] for match in matches])

    A = np.tile(matches1, (1, 3)) * np.repeat(matches2, 3, 1)

    U, D, V = np.linalg.svd(A)

    Uf = U[:,0:3]
    Df = np.diag(D[0:3])
    Vf = V[0:3]

    # TODO make singular. It says we have to do this in the assingment, but can
    # not find the reference mentioned.

    F = Uf * Df * Vf

    Uf, Df, Vf = np.linalg.svd(F)

    # Set smallest singular value to zero
    Df[3, 3] = 0

    # Recompute F
    F = Uf * Df * Vf

    return F
