import numpy as np
import cv2


def read_and_crop(img_name, min_height, max_height, min_width, max_width,
                  grayscale=True):
    """"""
    img = cv2.imread(img_name)
    # TODO enable 1D image croppin
    img = img[min_height:max_height, min_width:max_width, :]
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def filter_matches(kp1, kp2, matches, ratio=0.75):
    """
    Filter matches based on the notion that at the second closest match
    should be at least at <ratio> distance from the best match.
    """
    return [(kp1[m[0].queryIdx], kp2[m[0].trainIdx])
            for m in matches if len(m) >= 2
            and m[0].distance < ratio * m[1].distance]


def eightpoint(img_files, normalized, ransac_iterations=None,
               verbosity=0):
    """
    Perform the eightpoint algorithm for a given set of images.
    Normalized is a boolean indicating whether we should use the
    normalized eightpoint algorithm or not. If ransac_iterations
    is >0 , we use ransac to find the best fundamental matrix.
    """
    # initialize sift detector
    sift = cv2.SIFT()
    # initialize params for FLANN
    index_params = {'algorithm': 0,    # FLANN_INDEX_KDTREE,
                    'trees': 5}
    search_params = {'checks': 50}

    # For bear, crop image to 200:1400, 600:1800
    bear_crop = [200, 1400, 600, 1800]
    img1 = read_and_crop(img_files[0], *bear_crop, grayscale=True)
    # For house, don't crop image? TODO

    # Compute keypoints
    kp1, des1 = sift.detectAndCompute(img1, None)

    for img2_name in img_files[1:]:
        # Read the next file
        img2 = read_and_crop(img2_name, *bear_crop, grayscale=True)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Use flann to find best matches
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # Get set of tuples of keypoints that match
        # TODO make ratio arg
        matches = filter_matches(kp1, kp2, matches, ratio=0.5)
        if verbosity > 1:
            drawmatches(img1, img2, matches, verbosity)

        # Use some metric to reduce bad matches

        # Update
        img1, kp1, des1 = img2, kp2, des2


def drawmatches(img1, img2, matches, verbosity=0):
    """
    Since drawMatches is not yet included in opencv 2.4.9, we
    added a simple function that visualises matches in different
    colors, based on (mostly copied from):

    http://stackoverflow.com/questions/11114349/how-to-visualize-
        descriptor-matching-using-opencv-module-in-python

    It accepts two grayscale images and keypoints, plus an array
    of matches. It will only consider the first match.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if verbosity > 1:
        print "Img1: height {}, width {}".format(h1, w1)
        print "Img2: height {}, width {}".format(h2, w2)

    # Create storage for eventual matches
    view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]

    for kp1, kp2 in matches:
        # draw the keypoints
        # print m.queryIdx, m.trainIdx, m.distance
        color = tuple([np.random.randint(0, 255)
                       for _ in xrange(3)])
        cv2.line(view,
                 (int(kp1.pt[0]),
                  int(kp1.pt[1])),
                 (int(kp2.pt[0] + w1),
                  int(kp2.pt[1])),
                 color)
    # Resize for easy display
    view = cv2.resize(view, (0, 0), fx=0.25, fy=0.25)
    cv2.imshow("Matches", view)
    cv2.waitKey()


# 1. Detect interest points in both images
# 2. Characterize local appearance of the regions around interest points
# 3. Get a set of supposed matches
# 4. Estimate fundamental matrix

def fundamental(matches1, matches2):
    """
    Compute the fundamental matrix given two sets of matches.
    """
    # Compute svd of A
    A = np.tile(matches1, (1, 3)) * np.repeat(matches2, 3, 1)

    U, D, V = np.linalg.svd(A)
    # Take columns/rows of interest
    Uf = U[:, 0:3]
    Df = np.diag(D[0:3])
    Vf = V[0:3]

    # TODO find out why this is often nonsingular, and what the impact is.
    # It says we have to do this in the assignment, but can not find the
    # reference mentioned.
    F = np.dot(Uf, np.dot(Df, Vf))
    Uf, Df, Vf = np.linalg.svd(F)

    # Set smallest singular value to zero
    Df[3, 3] = 0
    # Recompute F
    F = Uf * Df * Vf

    return F
