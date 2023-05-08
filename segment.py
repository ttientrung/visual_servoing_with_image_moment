import os
import cv2

os.system('v4l2-ctl -d 2 --set-ctrl focus_auto=0')
os.system('v4l2-ctl -d 2 --set-ctrl focus_absolute=51')

thresh0 = 73
def update_thresh(value):
    global thresh
    thresh = value

cap = cv2.VideoCapture(2)
cv2.namedWindow('Segment')
cv2.createTrackbar('Threshold', 'Segment', thresh0, 255, update_thresh)
update_thresh(thresh0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, seg = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Segment', seg)
    k = cv2.waitKey(1)
    if k==32:
        break

cap.release()
cv2.destroyAllWindows()