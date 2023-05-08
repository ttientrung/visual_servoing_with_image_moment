import subprocess
import sys
import pickle
from PyQt5.QtWidgets import QApplication, QWidget
from utils.main_window import Ui_Form
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QEvent
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import utils.image_moments as moments
from utils.transformation import get_Zd, tr2pose, pose2tr, rotz, transl
from utils.communicate import Robot
from time import sleep
import datetime

# Camera id, if on Windows, set it to 1
device = 0

# This block is used to set camera focus on linux, if on Windows, comment it out.
# focus = 50
# retcode = subprocess.call(
#     ["v4l2-ctl", "-d", str(device), "--set-ctrl", "focus_auto=0"])
# retcode = subprocess.call(
#     ["v4l2-ctl", "-d", str(device), "--set-ctrl", f"focus_absolute={focus}"])
# if retcode != 0:
#     print("Cannot set device's focus.")

# Init robot connection
arm = Robot(connection=False)


def limit(x, low, high):
    if low > high:
        raise Exception("Lower limit must not be higher than upper limit.")
    if x < low:
        return low
    elif x > high:
        return high
    else:
        return x


class CameraThread(QThread):
    changePixmap = pyqtSignal(QPixmap)
    changeError = pyqtSignal(tuple)

    def run(self):
        cap = cv2.VideoCapture(device)
        ret, frame = cap.read()
        h, w, _ = frame.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        while True:
            ret, frame = cap.read()
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            if not ret:
                raise Exception("Lost camera connection.")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if self.parent().comboBox.currentIndex() == 0:
                mask_blue = cv2.inRange(hsv, np.array([90, 90, 100]), np.array([105, 255, 255]))
                mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
                mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
                mask_obj = mask_blue
                color = [255, 184, 99]
            elif self.parent().comboBox.currentIndex() == 1:
                mask_green = cv2.inRange(hsv, np.array([70, 70, 60]), np.array([95, 255, 255]))
                mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
                mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
                mask_obj = mask_green
                color = [0, 255, 0]
            else:
                mask_obj = None
            self.binary_img = mask_obj
            try:
                self.s = moments.features(self.binary_img)
                Zd = self.parent().Zd
                sd = self.parent().sd
                self.error = moments.get_error(self.s, sd, Zd)
                self.changeError.emit(self.error)
            except Exception as e:
                print(e)
            if self.parent().mode == 0:
                rgbImage = cv2.line(
                    frame, (w//2, 0), (w//2, h), (0, 255, 255), 1)
                rgbImage = cv2.line(rgbImage, (0, h//2),
                                    (w, h//2), (0, 255, 255), 1)
                rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)
                rgbpixmap = QPixmap.fromImage(
                    QImage(rgbImage.data, w, h, QImage.Format_RGB888))
                self.changePixmap.emit(rgbpixmap.scaledToHeight(300))
            elif self.parent().mode == 1:
                binpixmap = QPixmap.fromImage(
                    QImage(mask_obj.data, w, h, QImage.Format_Grayscale8))
                self.changePixmap.emit(binpixmap.scaledToHeight(300))
            elif self.parent().mode == 2:
                comp_img = cv2.drawContours(
                    frame, self.parent().contours, -1, color, 3)
                comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
                comp_pixmap = QPixmap.fromImage(
                    QImage(comp_img.data, w, h, QImage.Format_RGB888))
                self.changePixmap.emit(comp_pixmap.scaledToHeight(300))


class MoveThread(QThread):
    donesignal = pyqtSignal(bool)

    def __init__(self, parent=None, X=0, Y=0, Z=0, roll=0, pitch=0, yaw=0, other=None):
        super().__init__(parent)
        self.X = X
        self.Y = Y
        self.Z = Z
        self.r = roll
        self.p = pitch
        self.y = yaw
        self.other = other

    def run(self):
        if self.other is None:
            arm.shift_cam(self.X, self.Y, self.Z, self.r, self.p, self.y)
        elif self.other == 'home':
            arm.tohome()
        elif self.other == 'work':
            arm.towork()
        elif self.other == 'pick':
            pose = arm.getpose()
            Ttool = pose2tr(pose, 'deg')
            Tcam = Ttool.dot(arm.Ttool_cam)
            Ttool_new = Tcam.dot(rotz(-90, 'deg'))
            Tpick = Ttool_new.dot(transl(0, 0, self.parent().Zd*1000))
            print(Tpick)
            X, Y, Z, r, p, y = tr2pose(Tpick, 'deg')
            arm.movex(X, Y, Z, r, p, y, 1)
            # arm.shiftx(Z=-40)
            arm.Air(10)
            arm.movex(*pose, 1)
        elif self.other == 'place':
            pose = arm.getpose()
            Ttool = pose2tr(pose, 'deg')
            Tcam = Ttool.dot(arm.Ttool_cam)
            Ttool_new = Tcam.dot(rotz(-90, 'deg'))
            Tplace = Ttool_new.dot(transl(0, 0, self.parent().Zd*1000))
            X, Y, Z, r, p, y = tr2pose(Tplace, 'deg')

            arm.movex(X, Y, Z+40, r, p, y, 1)
            
            arm.shiftx(Z=-40)
            arm.Air(9)
            arm.movex(*pose, 1)
        elif self.other == 'placeposition':
            arm.movex(-10, -360, 161, -94, 0, 180, 1)
        self.donesignal.emit(True)


class TraceThread(QThread):
    donesignal = pyqtSignal(bool)

    def __init__(self, parent, full_dof=True):
        super().__init__(parent)
        self.full_dof = full_dof

    def run(self):
        self.parent().stop = False
        self.parent().record = True
        vcam_data = []
        while True:
            if self.parent().stop:
                break
            self.parent().joints_data.append((datetime.datetime.now(), arm.getjoint()))
            dan, dxn, dyn, ds4, ds5, dtheta = self.parent().error
            if self.full_dof:
                finish = (abs(dxn) < 0.5) and (
                    abs(dyn) < 0.5) and (abs(dan) < 1e-3) and ((dtheta*180/np.pi) < 1) and (abs(ds4) < 5e-6) and (abs(ds5) < 1e-8)
            else:
                finish = (abs(dxn) < 0.5) and (abs(dyn) < 0.5) and (
                    abs(dan) < 5e-4) and ((dtheta*180/np.pi) < 0.5)
            if finish:
                break
            vx = limit(dxn, -200, 200)
            vy = limit(dyn, -200, 200)
            vz = limit(1000*dan, -200, 200)
            wz = limit(dtheta*180/np.pi, -90, 90)
            if self.full_dof:
                L = self.parent().L
                Ls45 = L[3:5, 3:5]
                tmp = -np.linalg.inv(Ls45).dot(np.array([[ds4, ds5]]).T)
                wx = tmp[0, 0]
                wy = tmp[1, 0]
                wx = limit(wx*1e4, -2, 2)
                wy = limit(wy*1e4, -2, 2)
            else:
                wx = 0
                wy = 0
            vcam_data.append(
                (datetime.datetime.now(), (vx, vy, vz, wx, wy, wz)))
            arm.shift_cam(X=vx, Y=vy, Z=vz, roll=wz, pitch=wy, yaw=wx)
        self.donesignal.emit(True)
        self.parent().record = False
        now = datetime.datetime.now()
        tail = f'{now.year}{now.month:02}{now.day:02}_{now.hour:02}{now.minute:02}{now.second:02}'
        with open(f'record/error_{tail}.dat', 'wb') as f:
            pickle.dump(self.parent().error_data, f)
        with open(f'record/joints_{tail}.dat', 'wb') as f:
            pickle.dump(self.parent().joints_data, f)
        with open(f'record/vcam_{tail}.dat', 'wb') as f:
            pickle.dump(vcam_data, f)
        self.parent().error_data = []
        self.parent().joints_data = []


class Window(Ui_Form, QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Image moments based Visual Servoing")

        self.stop = False
        self.mode = 0
        self.shift_delta = self.shift_delta_spinbox.value()
        self.angle_delta = self.angle_delta_spinbox.value()
        self.full_dof = True
        self.record = False
        self.error_data = []
        self.joints_data = []

        try:
            with open('sample/sample_features_blue.dat', 'rb') as f:
                self.sd_blue, self.L_blue, self.Zd_blue = pickle.load(f)
            with open('sample/sample_contours_blue.dat', 'rb') as f:
                self.contours_blue, self.hierarchy_blue = pickle.load(f)
            with open('sample/sample_features_green.dat', 'rb') as f:
                self.sd_green, self.L_green, self.Zd_green = pickle.load(f)
            with open('sample/sample_contours_green.dat', 'rb') as f:
                self.contours_green, self.hierarchy_green = pickle.load(f)
            self.sd = self.sd_blue
            self.L = self.L_blue
            self.Zd = self.Zd_blue
            self.contours = self.contours_blue
            self.hierarchy = self.hierarchy_blue
        except Exception as e:
            print(e)

        # region Binding button
        self.home_button.clicked.connect(self.home_button_clicked)
        self.work_button.clicked.connect(self.work_button_clicked)
        self.set_button.clicked.connect(self.set_button_clicked)
        self.trace_button.clicked.connect(self.trace_button_clicked)
        self.move_up_button.clicked.connect(self.move_up_button_clicked)
        self.move_down_button.clicked.connect(self.move_down_button_clicked)
        self.move_left_button.clicked.connect(self.move_left_button_clicked)
        self.move_right_button.clicked.connect(self.move_right_button_clicked)
        self.rotate_left_button.clicked.connect(
            self.rotate_left_button_clicked)
        self.rotate_right_button.clicked.connect(
            self.rotate_right_button_clicked)
        self.tilt_x_plus_button.clicked.connect(
            self.tilt_x_plus_button_clicked)
        self.tilt_x_minus_button.clicked.connect(
            self.tilt_x_minus_button_clicked)
        self.tilt_y_plus_button.clicked.connect(
            self.tilt_y_plus_button_clicked)
        self.tilt_y_minus_button.clicked.connect(
            self.tilt_y_minus_button_clicked)
        self.zoom_in_button.clicked.connect(self.zoom_in_button_clicked)
        self.zoom_out_button.clicked.connect(self.zoom_out_button_clicked)
        self.radioButton.toggled.connect(self.update_mode)
        self.radioButton_2.toggled.connect(self.update_mode)
        self.radioButton_3.toggled.connect(self.update_mode)
        self.shift_delta_spinbox.valueChanged.connect(self.update_shift_delta)
        self.angle_delta_spinbox.valueChanged.connect(self.update_angle_delta)
        self.get_pose_button.clicked.connect(self.get_pose_button_clicked)
        self.get_error_button.clicked.connect(self.get_error_button_clicked)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        self.pick_button.clicked.connect(self.pick_button_clicked)
        self.place_button.clicked.connect(self.place_button_clicked)
        self.release_button.clicked.connect(self.release_button_clicked)
        self.comboBox.currentIndexChanged.connect(self.comboBox_changed)
        self.place_position_button.clicked.connect(
            self.place_position_button_clicked)
        # endregion

        # Set up camera view
        self.camera = CameraThread(self)
        self.camera.changePixmap.connect(self.show_image)
        self.camera.changeError.connect(self.update_error)
        self.camera.start()

        # Display
        self.show()

    # region binding function
    @pyqtSlot()
    def get_pose_button_clicked(self):
        X, Y, Z, r, p, y = arm.getpose()
        self.X_label.setText(f'{X:0.2f}')
        self.Y_label.setText(f'{Y:0.2f}')
        self.Z_label.setText(f'{Z:0.2f}')
        self.roll_label.setText(f'{r:0.2f}')
        self.pitch_label.setText(f'{p:0.2f}')
        self.yaw_label.setText(f'{y:0.2f}')

    @pyqtSlot()
    def get_error_button_clicked(self):
        dan, dxn, dyn, ds4, ds5, dtheta = self.error
        self.an_error_label.setText(f'{dan:0.4f}')
        self.xn_error_label.setText(f'{dxn:0.4f}')
        self.yn_error_label.setText(f'{dyn:0.4f}')
        self.s4_error_label.setText(f'{ds4:0.2e}')
        self.s5_error_label.setText(f'{ds5:0.2e}')
        self.theta_error_label.setText(f'{dtheta*180/np.pi:0.2f}')

    @pyqtSlot()
    def home_button_clicked(self):
        movethread = MoveThread(self, other='home')
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def work_button_clicked(self):
        movethread = MoveThread(self, other='work')
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def place_position_button_clicked(self):
        movethread = MoveThread(self, other='placeposition')
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def set_button_clicked(self):
        self.sample = self.camera.binary_img
        pose = arm.getpose()
        Ttool_cam = arm.Ttool_cam
        Zd = get_Zd(pose, Ttool_cam)
        M = cv2.moments(self.sample)
        s = moments.features(self.sample)
        L = moments.L_matrix(M, s)
        self.contours, self.hierarchy = cv2.findContours(
            self.sample, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.sd = s
        self.L = L
        self.Zd = Zd
        print(f'Zd = {Zd}')
        print(f's = {s}')
        print(f'L = {L}')
        if self.comboBox.currentIndex() == 0:
            cv2.imwrite('sample/sample_blue.jpg', self.sample)
            with open('sample/sample_pose_blue.dat', 'wb') as f:
                pickle.dump(pose, f)
            with open('sample/sample_features_blue.dat', 'wb') as f:
                pickle.dump([s, L, Zd], f)
            with open('sample/sample_contours_blue.dat', 'wb') as f:
                pickle.dump([self.contours, self.hierarchy], f)
            self.sd_blue = self.sd
            self.L_blue = self.L
            self.Zd_blue = self.Zd
            self.contours_blue = self.contours
            self.hierarchy_blue = self.hierarchy
        elif self.comboBox.currentIndex() == 1:
            cv2.imwrite('sample/sample_green.jpg', self.sample)
            with open('sample/sample_pose_green.dat', 'wb') as f:
                pickle.dump(pose, f)
            with open('sample/sample_features_green.dat', 'wb') as f:
                pickle.dump([s, L, Zd], f)
            with open('sample/sample_contours_green.dat', 'wb') as f:
                pickle.dump([self.contours, self.hierarchy], f)
            self.sd_green = self.sd
            self.L_green = self.L
            self.Zd_green = self.Zd
            self.contours_green = self.contours
            self.hierarchy_green = self.hierarchy

    @pyqtSlot()
    def trace_button_clicked(self):
        self.enable_buttons(False)
        tracethread = TraceThread(self, full_dof=self.full_dof)
        tracethread.donesignal.connect(self.enable_buttons)
        tracethread.start()

    @pyqtSlot()
    def stop_button_clicked(self):
        self.stop = True

    @pyqtSlot()
    def pick_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, other='pick')
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def place_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, other='place')
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def release_button_clicked(self):
        arm.release()

    @pyqtSlot(bool)
    def enable_buttons(self, show):
        self.home_button.setEnabled(show)
        self.work_button.setEnabled(show)
        self.set_button.setEnabled(show)
        self.trace_button.setEnabled(show)
        self.zoom_in_button.setEnabled(show)
        self.zoom_out_button.setEnabled(show)
        self.move_down_button.setEnabled(show)
        self.move_up_button.setEnabled(show)
        self.move_left_button.setEnabled(show)
        self.move_right_button.setEnabled(show)
        self.rotate_left_button.setEnabled(show)
        self.rotate_right_button.setEnabled(show)
        self.tilt_x_minus_button.setEnabled(show)
        self.tilt_x_plus_button.setEnabled(show)
        self.tilt_y_minus_button.setEnabled(show)
        self.tilt_y_plus_button.setEnabled(show)
        self.get_pose_button.setEnabled(show)
        self.place_button.setEnabled(show)
        self.pick_button.setEnabled(show)
        self.release_button.setEnabled(show)
        self.place_position_button.setEnabled(show)

    @pyqtSlot()
    def move_up_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, Y=-self.shift_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def move_down_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, Y=self.shift_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def move_left_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, X=-self.shift_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def move_right_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, X=self.shift_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def tilt_x_minus_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, yaw=-self.angle_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def tilt_x_plus_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, yaw=self.angle_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def tilt_y_minus_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, pitch=-self.angle_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def tilt_y_plus_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, pitch=self.angle_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def zoom_in_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, Z=self.shift_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def zoom_out_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, Z=-self.shift_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def rotate_left_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, roll=-self.angle_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot()
    def rotate_right_button_clicked(self):
        self.enable_buttons(False)
        movethread = MoveThread(self, roll=self.angle_delta)
        movethread.donesignal.connect(self.enable_buttons)
        movethread.start()

    @pyqtSlot(QPixmap)
    def show_image(self, pixmap):
        self.camera_view.setPixmap(pixmap)

    @pyqtSlot()
    def update_mode(self):
        if self.radioButton.isChecked():
            self.mode = 0
        elif self.radioButton_2.isChecked():
            self.mode = 1
        elif self.radioButton_3.isChecked():
            self.mode = 2

    @pyqtSlot()
    def update_shift_delta(self):
        self.shift_delta = self.shift_delta_spinbox.value()

    @pyqtSlot()
    def update_angle_delta(self):
        self.angle_delta = self.angle_delta_spinbox.value()

    @pyqtSlot(QEvent)
    def closeEvent(self, event):
        try:
            arm.stop()
        except AttributeError:
            pass
        event.accept()

    @pyqtSlot(tuple)
    def update_error(self, error):
        self.error = error
        if self.record:
            self.error_data.append((datetime.datetime.now(), error))

    @pyqtSlot(int)
    def comboBox_changed(self, value):
        if value == 0:
            self.contours = self.contours_blue
            self.hierarchy = self.hierarchy_blue
            self.L = self.L_blue
            self.sd = self.sd_blue
            self.Zd = self.Zd_blue
            self.full_dof = True
        elif value == 1:
            self.contours = self.contours_green
            self.hierarchy = self.hierarchy_blue
            self.L = self.L_green
            self.sd = self.sd_green
            self.Zd = self.Zd_green
            self.full_dof = False
    # endregion


if __name__ == "__main__":
    ap = QApplication(sys.argv)
    window = Window()
    sys.exit(ap.exec_())
