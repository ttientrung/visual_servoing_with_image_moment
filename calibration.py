import pickle
import get_imgs
import calib
import cv2
import numpy as np
from utils.transformation import pose2tr, compose
np.set_printoptions(precision=2, suppress=True)

with open('cam_matrix/axis_pose.dat', 'rb') as f:
    P1, P2, P3 = pickle.load(f)

# print('Getting images...')
# pose = get_imgs.run(device=2, focus=50)     # if on Windows, set device = 1
# with open('cam_matrix/pose.dat', 'rb') as f:
#     pose = pickle.load(f)
with open('cam_matrix/pose.dat', 'rb') as f:
    pose = pickle.load(f)
    print(pose)

print('------------------')
print('Run calibration...')
nums, mtx, rvecs, tvecs = calib.run(pattern='circle', patternsize=(10, 7))

T0_tool = []
for i in nums:
    T0_tool.append(pose2tr(pose[i], 'deg'))

Tcam_pattern = []
for i in range(len(rvecs)):
    R, _ = cv2.Rodrigues(rvecs[i])
    t = tvecs[i]
    Tcam_pattern.append(compose(R, t))

P1 = np.array([P1[:3]]).T
P2 = np.array([P2[:3]]).T
P3 = np.array([P3[:3]]).T
Zmean = (P1[2, 0]+P2[2, 0]+P3[2, 0])/3
P1[2, 0] = Zmean
P2[2, 0] = Zmean
P3[2, 0] = Zmean
Ox = P2-P1
Oy = P3-P1
Oz = np.cross(Ox, Oy, axis=0)
Ox = Ox/np.linalg.norm(Ox)
Oy = Oy/np.linalg.norm(Oy)
Oz = Oz/np.linalg.norm(Oz)
R = np.hstack((Ox, Oy, Oz))
t = P1
T0_pattern = compose(R, t)
print(T0_pattern)

T0_cam = []
for i in range(len(Tcam_pattern)):
    T0_cam.append(T0_pattern.dot(np.linalg.inv(Tcam_pattern[i])))

T0_tool = np.vstack(T0_tool)
T0_cam = np.vstack(T0_cam)
Ttool_cam = np.linalg.pinv(T0_tool).dot(T0_cam)
with open('cam_matrix/Ttool_cam3.dat', 'wb') as f:
    pickle.dump(Ttool_cam, f)
print('------------')
print('Ttool_cam')
print(Ttool_cam)
