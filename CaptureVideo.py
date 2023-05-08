import cv2
import pickle
from utils.communicate import Robot
arm = Robot(connection=True)
arm.tohome()

cap = cv2.VideoCapture(0)
cv2.namedWindow("Image Cap")
img_counter = 1
pose = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Image Cap", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k == ord('c'):
        # SPACE pressed
        img_name = "C://Users//Lenovo//Downloads//pc (1)//pc//imgsss//img_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("Picture {} had written!".format(img_counter))
        X, Y, Z, roll, pitch, yaw = arm.getpose()
        pose.append((X, Y, Z, roll, pitch, yaw))
        img_counter += 1

cap.release()
cv2.destroyAllWindows()


print('----------')
print('Saving pose...')
with open('cam_matrix/pose2.dat', 'wb') as f:
    pickle.dump(pose, f)
print('Finish.')

