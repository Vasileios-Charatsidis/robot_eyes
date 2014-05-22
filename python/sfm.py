import numpy as np
pulp_present = False
try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpDot
    pulp_present = True
except:
    print "Pulp not available, can not remove affine ambiguity."


def structure_from_motion(pointviewmat, args):
    '''
    Create a 3D model based on a set of matching 2D points!
    '''

    # Remove columns of points that do not persist throughout the pointviewmat
    m, n, _ = pointviewmat.shape
    persisting_points = [pt_idx for pt_idx in xrange(n) if
                         np.array([0, 0]) not in pointviewmat[:, pt_idx, :]]
    print "Of {} points".format(n),
    pointviewmat = pointviewmat[:, persisting_points, :]
    m, n, _ = pointviewmat.shape
    print "{} points persist throughout the sequence".format(n)

    # Subtract the mean for each image at the same time
    pointviewmat -= np.mean(pointviewmat, axis=1)

    # Our pointviewmat is m x n x 2, we'll make it 2m x n
    pointviewmat = \
        np.vstack(tuple([pointviewmat[m_, :, :].T for m_ in xrange(m)]))

    U, W, V = np.linalg.svd(pointviewmat)
    # Enforce rank 3
    U = U[:, :3]
    W = W[:3, :3]
    V = V[:, :3]

    # Create motion and structure matrices from svd
    M = np.dot(U, W)
    S = V

    # Remove affine ambiguity, if possible
    if not pulp_present:
        return M, S

    affine_problem = LpProblem("Find L that removes ambiguity", LpMinimize)
    L = {(i, j): LpVariable("L" + str(i) + str(j))
         for j in xrange(3) for i in xrange(3)}
    d = LpVariable("d")
    affine_problem += d
    # For easy access later
    L_row = {}
    L_col = {}
    for i in xrange(3):
        L_row[i] = [L[i, j] for j in xrange(3)]
        L_col[i] = [L[j, i] for j in xrange(3)]

    for img_idx in xrange(m):
        # Select the first and second row of affine transformation matrix m
        a1 = list(M[img_idx * 2, :])
        a2 = list(M[img_idx * 2 + 1, :])

        affine_problem += \
            lpDot([lpDot(a1, L_col[i]) for i in xrange(3)], a1) == d
        affine_problem += \
            lpDot([lpDot(a2, L_col[i]) for i in xrange(3)], a2) == d
        affine_problem += \
            lpDot([lpDot(a1, L_col[i]) for i in xrange(3)], a2) == 0

    affine_problem.solve()
    assert affine_problem.status == 1, "LP failed"
    L_mat = np.zeros((3, 3))
    for i in xrange(3):
        for j in xrange(3):
            L_mat[i, j] = L[i, j].varValue

    # Perform cholesky decomposition
