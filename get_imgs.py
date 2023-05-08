# This file is used by calibration.py.
import os
import socket
import cv2
import pickle


def run(device=2, focus=50):
    # This block is used to set camera focus on linux, if on Windows, comment it out.
    os.system(f'v4l2-ctl -d {device} --set-ctrl focus_auto=0')
    os.system(f'v4l2-ctl -d {device} --set-ctrl focus_absolute={focus}')

    cap = cv2.VideoCapture(device)
    ret, frame = cap.read()
    if not ret:
        raise Exception("Cannot connect to camera device {}.".format(device))

    host = '192.168.1.16'
    port = 3333
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen()
    num = 0
    pose = []
    close = False
    conn, addr = s.accept()
    while not close:
        # Receive data from robot
        data = conn.recv(1024)
        if not data:
            break
        # Read camera and save image
        for i in range(10):
            ret, frame = cap.read()
        print('Saving img_{:02}.jpg ...'.format(num))
        cv2.imwrite('imgs/img_{:02}.jpg'.format(num), frame)
        # Get pose
        X, Y, Z, roll, pitch, yaw = data.decode().split(' ')
        X = float(X)
        Y = float(Y)
        Z = float(Z)
        roll = float(roll)
        pitch = float(pitch)
        yaw = float(yaw)
        pose.append((X, Y, Z, roll, pitch, yaw))
        print(pose[-1])
        conn.sendall(b'D')
        num += 1

    cap.release()
    cv2.destroyAllWindows()

    # Close connection
    conn.close()
    s.close()

    # Save poses
    print('----------')
    print('Saving pose...')
    with open('cam_matrix/pose.dat', 'wb') as f:
        pickle.dump(pose, f)
    print('Finish.')
    return pose


if __name__ == '__main__':
    run()
