import numpy as np
import sympy as sym


def checkSym(*variables):
    """
    Check if there's symbol in variables, if there is, use cos and sin funtion
    from module sympy instead of numpy and replace array with Matrix from sympy.
    """
    contain_symbol = False
    for x in variables:
        if isinstance(x, (sym.Basic, sym.matrices.MatrixBase)):
            contain_symbol = True
            break
    return contain_symbol


def transl(x, y, z):
    """Return transformation matrix of translating by x, y, z"""
    if checkSym(x, y, z):
        array = sym.Matrix
    else:
        array = np.array
    return array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])


def rotx(theta, unit='rad'):
    """Return transformation matrix of rotating by theta (rad/degree) around
    X axis"""
    if checkSym(theta):
        cos = sym.cos
        sin = sym.sin
        array = sym.Matrix
        pi = sym.pi
    else:
        cos = np.cos
        sin = np.sin
        array = np.array
        pi = np.pi
    if unit == 'deg':
        theta = theta/180*pi
    return array([[1, 0, 0, 0],
                  [0, cos(theta), -sin(theta), 0],
                  [0, sin(theta), cos(theta), 0],
                  [0, 0, 0, 1]])


def roty(theta, unit='rad'):
    """Return transformation matrix of rotating by theta (rad/degree) around
    Y axis"""
    if checkSym(theta):
        cos = sym.cos
        sin = sym.sin
        array = sym.Matrix
        pi = sym.pi
    else:
        cos = np.cos
        sin = np.sin
        array = np.array
        pi = np.pi
    if unit == 'deg':
        theta = theta/180*pi
    return array([[cos(theta), 0, sin(theta), 0],
                  [0, 1, 0, 0],
                  [-sin(theta), 0, cos(theta), 0],
                  [0, 0, 0, 1]])


def rotz(theta, unit='rad'):
    """Return transformation matrix of rotating by theta (rad/degree) around
    Z axis"""
    if checkSym(theta):
        cos = sym.cos
        sin = sym.sin
        array = sym.Matrix
        pi = sym.pi
    else:
        cos = np.cos
        sin = np.sin
        array = np.array
        pi = np.pi
    if unit == 'deg':
        theta = theta/180*pi
    return array([[cos(theta), -sin(theta), 0, 0],
                  [sin(theta), cos(theta), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])


def rpy2tr(r, p, y, unit='rad'):
    """
    Return transformation matrix from roll, pitch, yaw angles:
        roll is rotation angle around Z axis
        pitch is rotation angle around Y axis
        yaw is rotation angle around X axis
    Rotation order is XYZ.
    """
    Rz = rotz(r, unit)
    Ry = roty(p, unit)
    Rx = rotx(y, unit)
    if checkSym(r, p, y):
        R = Rz*Ry*Rx
    else:
        R = Rz.dot(Ry).dot(Rx)
    return R


def tr2rpy(T, unit='rad'):
    """
    Return roll, pitch, yaw angles from transformation matrix:
        roll is rotation angle around Z axis
        pitch is rotation angle around Y axis
        yaw is rotation angle around X axis
    Rotation order is XYZ.
    """
    if checkSym(T):
        arcsin = sym.asin
        arctan2 = sym.atan2
        pi = sym.pi
    else:
        arcsin = np.arcsin
        arctan2 = np.arctan2
        pi = np.pi
    p = -arcsin(T[2, 0])
    r = arctan2(T[1, 0], T[0, 0])
    y = arctan2(T[2, 1], T[2, 2])
    if unit == 'deg':
        r = r*180/pi
        p = p*180/pi
        y = y*180/pi
    return r, p, y


def pose2tr(pose, angle_unit='rad'):
    """Return transformation matrix from the pose of robot."""
    X, Y, Z, r, p, y = pose
    T = rpy2tr(r, p, y, angle_unit)
    T[0, 3] = X
    T[1, 3] = Y
    T[2, 3] = Z
    return T


def tr2pose(T, angle_unit='rad'):
    """Return the pose of robot from transformation matrix."""
    X = T[0, 3]
    Y = T[1, 3]
    Z = T[2, 3]
    r, p, y = tr2rpy(T, angle_unit)
    return X, Y, Z, r, p, y


def skew(w):
    try:
        wx, = w[0]
        wy, = w[1]
        wz, = w[2]
    except Exception:
        wx = w[0]
        wy = w[1]
        wz = w[2]
    if checkSym(w):
        array = sym.Matrix
    else:
        array = np.array
    return array([[0, -wz, wy],
                  [wz, 0, -wx],
                  [-wy, wx, 0]])


def decompose(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def compose(R, t):
    if checkSym(R, t):
        zeros = sym.zeros(4)
    else:
        zeros = np.zeros((4, 4))
    T = zeros
    T[:3, :3] = R
    T[:3, 3] = t.reshape(1, 3)
    T[3, 3] = 1
    return T


def B_matrix(r, p, y):
    """Return the relation matrix of angle rate and roll, pitch, yaw rate:
    [wx]       [dr/dt]
    |wy| = B . |dp/dt|
    [wz]       [dy/dt]
    """
    if checkSym(r, p, y):
        array = sym.Matrix
        sin = sym.sin
        cos = sym.cos
    else:
        array = np.array
        sin = np.sin
        cos = np.cos
    return array([[0, -sin(r), cos(r)*cos(p)],
                  [0, cos(r), sin(r)*cos(p)],
                  1, 0, -sin(p)])


def get_Zd(pose, Ttool_cam):
    Ttool = pose2tr(pose, 'deg')
    Tcam = Ttool.dot(Ttool_cam)
    Zcam = Tcam[2, 3]
    ncam = Tcam[:3, 2]
    vZ = np.array([0, 0, -Zcam])
    Zd = (Zcam**2)/np.dot(vZ, ncam)
    return Zd/1000


# Robot Nachi-MZ07
pi = np.pi


def set_tool(Z=183):
    global T_tool
    T_tool = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, Z],
                       [0, 0, 0, 1]])


alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = pi/2, 0, pi/2, -pi/2, pi/2, 0
a1, a2, a3, a4, a5, a6 = 50, 330, 45, 0, 0, 0
d1, d2, d3, d4, d5, d6 = 345, 0, 0, 340, 0, 73
set_tool()


def Tij(theta, d, a, alpha):
    return rotz(theta).dot(transl(0, 0, d)).dot(transl(a, 0, 0)).dot(rotx(alpha))


def joint2tr(J1, J2, J3, J4, J5, J6, unit='deg'):
    if unit == 'deg':
        J1 = J1*pi/180
        J2 = J2*pi/180
        J3 = J3*pi/180
        J4 = J4*pi/180
        J5 = J5*pi/180
        J6 = J6*pi/180
    T01 = Tij(J1, d1, a1, alpha1)
    T12 = Tij(J2, d2, a2, alpha2)
    T23 = Tij(J3, d3, a3, alpha3)
    T34 = Tij(J4, d4, a4, alpha4)
    T45 = Tij(J5, d5, a5, alpha5)
    T56 = Tij(J6, d6, a6, alpha6)
    T67 = T_tool
    T = T01.dot(T12).dot(T23).dot(T34).dot(T45).dot(T56).dot(T67)
    return T
