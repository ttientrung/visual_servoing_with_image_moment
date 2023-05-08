# This file is used by calibration.py.
import numpy as np
import cv2
import glob
import pickle


def run(pattern='circle', patternsize=(13, 9), d=20):
    # termination criteria                           max iteration--⬎     ⬐-- epsilon
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((patternsize[0]*patternsize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternsize[0],
                           0:patternsize[1]].T.reshape(-1, 2)*d

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('imgs/img*.jpg')

    nums = []
    for i in range(len(images)):
        fname = images[i]
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if pattern == 'chessboard':
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, patternsize, None)
        else:
            # Find circle grid centers
            ret, corners = cv2.findCirclesGrid(gray, patternsize)

        # If found, add object points, image points (after refining them)
        if ret:
            print(fname)
            nums.append(i)
            objpoints.append(objp)

            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, patternsize, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print('---------------')
    print('Camera matrix')
    print(mtx)
    # Save camera matrix to text file
    with open('cam_matrix/intrinsic2.txt', 'w') as f:
        f.write('{}'.format(mtx))
    with open('cam_matrix/extrinsic2.txt', 'w') as f:
        f.write('{}\n----------------\n{}'.format(rvecs, tvecs))

    # Save camera matrix to binary file
    with open('cam_matrix/cam_matrix2.dat', 'wb') as f:
        pickle.dump([nums, mtx, rvecs, tvecs], f)
    return nums, mtx, rvecs, tvecs


if __name__ == '__main__':
    run()
