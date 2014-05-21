def structure_from_motion(points2d, points3d, args):
    '''
    Create a 3D model based on a set of points2d and the corresponding 3D points!
    '''
    p1 = points2d[0]
    normalized_p1 = p1 - np.mean(p1)

    for p2 in points2d[1:]:
        normalized_p2 = p2 - np.mean(p2)

        D = np.vstack((p1, p2))
        U, W, V = np.linalg.svd(D)
        # Enforce rank 3
        U = U[:, :3]
        W = W[:3, :3]
        V = V[:, :3]

        # Create motion and structure matrices from svd
        M = np.dot(U, W)
        S = V


        normalied_p1 = normalized_p2

