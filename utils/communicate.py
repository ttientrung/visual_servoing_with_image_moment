import socket
import numpy as np
import pickle
from utils.transformation import joint2tr, tr2pose, pose2tr
np.set_printoptions(precision=2, suppress=True)


class Robot(object):
    def __init__(self, connection=True):
        with open('cam_matrix/Ttool_cam.dat', 'rb') as f:
            self.Ttool_cam = pickle.load(f)

        if connection:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.addr = (('192.168.1.1', 48952))
            self.s.connect(self.addr)
            print('Connected to address {} port {}'.format(*self.addr))
    
    @staticmethod
    def safecheck(arg1, arg2, arg3, arg4, arg5, arg6, checktype='x'):
        if checktype == 'j':
            T = joint2tr(arg1, arg2, arg3, arg4, arg5, arg6, 'deg')
            X, Y, Z, roll, pitch, yaw = tr2pose(T, 'deg')
        else:
            X, Y, Z = arg1, arg2, arg3
        d = np.sqrt(X**2 + Y**2)
        safe = (d > 200) and (d < 600) and (Z > -10) and (Z < 500)
        return safe

    def getpose(self):
        M = "P"
        M = bytes(M,'utf-8') 
        self.s.sendall(M)
        data = self.s.recv(1024)
        pose = data.decode().split(',')
        X = float(pose[0])
        Y = float(pose[1])
        Z = float(pose[2])
        r = float(pose[3])
        p = float(pose[4])
        y = float(pose[5])
        return X, Y, Z, r, p, y
        # return 0,0,0,0,0,0

    def getjoint(self):
        M = "J"
        M = bytes(M,'utf-8') 
        self.s.sendall(M)
        data = self.s.recv(1024)
        pose = data.decode().split(',')
        J1 = float(pose[0])
        J2 = float(pose[1])
        J3 = float(pose[2])
        J4 = float(pose[3])
        J5 = float(pose[4])
        J6 = float(pose[5])
        return J1, J2, J3, J4, J5, J6
        # return 0,0,0,0,0,0

    def movex(self, X, Y, Z, roll, pitch, yaw, mode, verbal = True):
        safe = self.safecheck(X, Y, Z, roll, pitch, yaw, 'x')
        M = bytes("A",'utf-8')
        signal = "busy"
        if safe:
            if verbal:
                print("Moving to ({:8.2f},{:8.2f},{:8.2f},{:8.2f},{:8.2f},{:8.2f})...".format(X,Y,Z,roll,pitch,yaw))
            while True:
                self.s.sendall(M)
                signal = self.s.recv(3)
                if signal == b'REA':
                    break
            x = "{0:8.2f}".format(X)
            y = "{0:8.2f}".format(Y)
            z = "{0:8.2f}".format(Z)
            r = "{0:8.2f}".format(roll)
            p = "{0:8.2f}".format(pitch)
            ya = "{0:8.2f}".format(yaw)
            m = "{:0>3d}".format(mode)
            data =  x + y + z + r + p + ya + m
            data = bytes(data,'utf-8')
            self.s.sendall(data)
            while True:
                signal = self.s.recv(3) 
                if signal == b'FIN':
                    break
        else:
            print("Not a safe position")
    
    def movej(self, J1, J2, J3, J4, J5, J6, mode, unit = 'deg', verbal = True):
        if unit == 'rad':
            pi = np.pi
            J1 = J1/pi*180
            J2 = J2/pi*180
            J3 = J3/pi*180
            J4 = J4/pi*180
            J5 = J5/pi*180
            J6 = J6/pi*180
        safe = self.safecheck(J1, J2, J3, J4, J5, J6, 'j')
        M = bytes("A",'utf-8')
        signal = "busy"
        if safe:
            if verbal:
                print("Moving to ({:8.2f},{:8.2f},{:8.2f},{:8.2f},{:8.2f},{:8.2f})...".format(J1, J2, J3, J4, J5, J6))
            while True:
                self.s.sendall(M)
                signal = self.s.recv(3)
                if signal == b'REA':
                    break
            J1 = "{0:8.2f}".format(J1)
            J2 = "{0:8.2f}".format(J2)
            J3 = "{0:8.2f}".format(J3)
            J4 = "{0:8.2f}".format(J4)
            J5 = "{0:8.2f}".format(J5)
            J6 = "{0:8.2f}".format(J6)
            m = "{:0>3d}".format(mode)
            data =  J1 + J2 + J3 + J4 + J5 + J6 + m
            data = bytes(data,'utf-8')
            self.s.sendall(data)
            while True:
                signal = self.s.recv(3) 
                if signal == b'FIN':
                    break   
        else:
            print("Not a safe position")
    
    def shiftx(self, X = 0, Y = 0, Z = 0, roll = 0, pitch = 0, yaw = 0, verbal = True):
        if ((abs(X) > 200) or (abs(Y) > 200) or (abs(Z) > 200) or (abs(roll) > 30) or (abs(pitch) > 30) or (abs(yaw) > 30)):
            print("Shift amount too large.")
        else:
            X0, Y0, Z0, r0, p0, y0 = self.getpose()
            self.movex(X0 + X, Y0+ Y, Z0 + Z, r0 + roll, p0 + pitch, y0 + yaw, 1, verbal)

    def shiftJ(self, J1 = 0, J2 = 0, J3 = 0, J4 = 0, J5 = 0, J6 = 0, verbal = True):
        if ((abs(J1) > 30) or (abs(J2) > 30) or (abs(J3) > 30) or (abs(J4) > 30) or (abs(J5) > 30) or (abs(J6) > 30)):
            print("Shift amount too large.")
        else:
            J10, J20, J30, J40, J50, J60 = self.getjoint()
            self.movej(J10 + J1, J20+ J2, J30 + J3, J40 + J4, J50 + J5, J60 + J6, 5, 'deg', verbal)

    def shift_cam(self, X = 0, Y = 0, Z = 0, roll = 0, pitch = 0, yaw = 0, verbal = True):
        Ttool_cam = self.Ttool_cam
        X0, Y0, Z0, r0, p0, y0 = self.getpose()
        T0_tool = pose2tr((X0, Y0, Z0, r0, p0, y0), 'deg')
        Tcam_new = pose2tr((X, Y, Z, roll, pitch, yaw), 'deg')
        T0_new = T0_tool.dot(Ttool_cam).dot(Tcam_new).dot(np.linalg.inv(Ttool_cam))
        Xnew, Ynew, Znew, rnew, pnew, ynew = tr2pose(T0_new, 'deg')
        self.movex(Xnew, Ynew, Znew, rnew, pnew, ynew, 1, verbal)

    def tohome(self, verbal = False):
        print('Move to home.')
        # self.movex(99, -369, 177, -94, 0.35, 180, 1, verbal)    
        self.movej(0, 90, 0, 0, -90, 0, 5, 'deg', verbal)
        print('At home.')
    
    def towork(self, verbal = False):
        print('Move to work.')
        self.movex(308, -360, 161, -94, 0, 180, 1, verbal)
        # self.movej(0, 90, 0, 0, -90, 0, 5, 'deg', verbal)
        print('At work.')
    
    def Air(self, mode):
        M = bytes("A",'utf-8')
        signal = "busy"
        while True:
            self.s.sendall(M)
            signal = self.s.recv(3)
            if signal == b'REA':
                break
        x = "{0:8.2f}".format(0)
        y = "{0:8.2f}".format(0)
        z = "{0:8.2f}".format(0)
        r = "{0:8.2f}".format(0)
        p = "{0:8.2f}".format(0)
        ya = "{0:8.2f}".format(0)
        m = "{:0>3d}".format(mode)
        data =  x + y + z + r + p + ya + m
        data = bytes(data,'utf-8')
        self.s.sendall(data)
        M = bytes("A",'utf-8')
        signal = "busy"
        while True:
            signal = self.s.recv(3) 
            if signal == b'FIN':
                break
    
    def stop(self):
        print('Closing...')
        self.s.close()
        print('Done.')

# arm = Robot(connection=True)
# arm.getpose()
# arm.getjoint()
# arm.movex(99, -369, 177, -94, 0.35, 180, 1, True)
# arm.movej(-75, 75, -36, 0, -40, 18, 5, 'deg', True)
# arm.shift_cam(10, 10, 50, 10, 0, 0, True)
# arm.tohome(True)
# arm.towork(True)
# arm.Air(10)
# arm.Air(9)
# arm.stop()

