import cv2
import numpy as np


def get_theta(img):
    m = cv2.moments(img)
    h, w = img.shape
    xg = m['m10']/m['m00']
    yg = m['m01']/m['m00']
    x0, y0 = w/2, h/2
    T = np.array([[1, 0, x0-xg], [0, 1, y0-yg]])
    timg = cv2.warpAffine(img, T, (w, h))
    theta = 1/2*np.arctan2(2*m['mu11'], m['mu20']-m['mu02'])
    R = cv2.getRotationMatrix2D((x0, y0), theta*180/np.pi, 1)
    rimg = cv2.warpAffine(timg, R, (w, h))
    m2 = cv2.moments(rimg)
    if m2['mu30'] > 0:
        theta = (theta+np.pi) % (2*np.pi)
    elif m2['mu30'] == 0:
        if m2['mu03'] > 0:
            theta = (theta+np.pi) % (2*np.pi)
    return theta


def features(image):
    M = cv2.moments(image)
    hu = cv2.HuMoments(M)
    a = M['m00']
    xg = M['m10']/a
    yg = M['m01']/a
    s4, = hu[0]
    s5, = hu[1]
    theta = get_theta(image)
    return a, xg, yg, s4, s5, theta


def get_error(s, sd, Zd):
    an = Zd*np.sqrt(np.abs(sd[0]/s[0]))
    xn = s[1]*an
    yn = s[2]*an
    dan = an - Zd
    dxn = xn - sd[1]*Zd
    dyn = yn - sd[2]*Zd
    ds4 = s[3]-sd[3]
    ds5 = s[4]-sd[4]
    dtheta = s[5] - sd[5]
    if dtheta > np.pi:
        dtheta = dtheta - 2*np.pi
    elif dtheta < -np.pi:
        dtheta = dtheta + 2*np.pi
    return dan, dxn, dyn, ds4, ds5, dtheta


def Lu(M, p, q, Zd):
    M.update({'mu10': 0, 'mu01': 0, f'mu{p}-1': 0,
              f'mu-1{q}': 0, f'mu{p+1}-1': 0, f'mu{-1}{q+1}': 0})
    xg = M['m10']/M['m00']
    yg = M['m01']/M['m00']
    n11 = M['mu11']/M['m00']
    n20 = M['mu20']/M['m00']
    n02 = M['mu02']/M['m00']
    uvz = (p+q+2)*1/Zd*M[f'mu{p}{q}']
    uwx = (p+q+3)*M[f'mu{p}{q+1}']+p*xg*M[f'mu{p-1}{q+1}']+(p+2*q+3) * \
        yg*M[f'mu{p}{q}']-4*p*n11*M[f'mu{p-1}{q}']-4*q*n02*M[f'mu{p}{q-1}']
    uwy = -(p+q+3)*M[f'mu{p+1}{q}']-(2*p+q+3)*xg*M[f'mu{p}{q}']-q*yg * \
        M[f'mu{p+1}{q-1}']+4*p*n20*M[f'mu{p-1}{q}']+4*q*n11*M[f'mu{p}{q-1}']
    uwz = p*M[f'mu{p-1}{q+1}']-q*M[f'mu{p+1}{q-1}']
    L = np.array([0, 0, uvz, uwx, uwy, uwz])
    return L


def Lnu(M, p, q, Zd):
    m00 = M['m00']
    xg = M['m10']/m00
    yg = M['m01']/m00
    upq = M[f'mu{p}{q}']
    Lm00 = np.array([0, 0, m00*2/Zd, 3*m00*yg, -3*m00*xg, 0])
    L = Lu(M, p, q, Zd)/np.power(m00, (p+q)/2+1) - \
        ((p+q)/2+1)*upq/np.power(m00, (p+q)/2+2)*Lm00
    return L


def L_matrix(M, s):
    xn = s[0]
    yn = s[1]
    an = s[2]
    xg = M['m10']/M['m00']
    yg = M['m01']/M['m00']
    n11 = M['mu11']/M['m00']
    n20 = M['mu20']/M['m00']
    n02 = M['mu02']/M['m00']
    eps11 = n11-xg*yg/2
    eps12 = n20-xg*xg/2
    eps21 = n02-yg*yg/2
    Lxn = np.array([-1, 0, 0, an*eps11, -an*(1+eps12), yn])
    Lyn = np.array([0, -1, 0, an*(1+eps21), -an*eps11, -xn])
    Lan = np.array([0, 0, -1, -3*yn/2, 3*xn/2, 0])
    d = (M['mu20']-M['mu02'])**2+4*M['mu11']**2
    beta = 5
    gamma = 1
    thwx = (beta*(M['mu12']*(M['mu20']-M['mu02'])+M['mu11']*(M['mu03']-M['mu21']))+gamma*xg *
            (M['mu02']*(M['mu20']-M['mu02'])-2*M['mu11']**2)+gamma*yg*M['mu11']*(M['mu20']+M['mu02']))/d
    thwy = (beta*(M['mu21']*(M['mu02']-M['mu20'])+M['mu11']*(M['mu30']-M['mu12']))+gamma*xg *
            M['mu11']*(M['mu20']+M['mu02'])+gamma*yg*(M['mu20']*(M['mu02']-M['mu20'])-2*M['mu11']**2))/d
    Lth = np.array([0, 0, 0, thwx, thwy, -1])
    # Ls4 = Lnu(M, 2, 0, Zd)+Lnu(M, 0, 2, Zd)
    # Ls5 = 2*(M['nu20']-M['nu02'])*(Lnu(M, 2, 0, Zd) -
    #                                Lnu(M, 0, 2, Zd))+8*M['nu11']*Lnu(M, 1, 1, Zd)
    # Ls4[:3] = 0
    # Ls4[5] = 0
    # Ls5[:3] = 0
    # Ls5[5] = 0
    s4wx = (2*xg*M['mu11']+yg*(M['mu02']-M['mu20']) +
            5*(M['mu03']+M['mu21']))/M['m00']**2
    s4wy = (xg*(M['mu02']-M['mu20'])-2*yg*M['mu11'] -
            5*(M['mu30']+M['mu12']))/M['m00']**2
    s5wx = (8*M['nu11']*(xg*M['mu02']+5*M['mu12'])-2*(M['nu02']-M['nu20']) *
            (2*xg*M['mu11']-yg*M['mu02']-yg*M['mu20']-5*M['mu03']+5*M['mu21']))/M['m00']**2
    s5wy = (-8*M['nu11']*(yg*M['mu20']+5*M['mu21'])+2*(M['nu02']-M['nu20']) *
            (xg*M['mu02']+xg*M['mu20']-2*yg*M['mu11']-5*M['mu12']+5*M['mu30']))/M['m00']**2
    Ls4 = np.array([0, 0, 0, s4wx, s4wy, 0])
    Ls5 = np.array([0, 0, 0, s5wx, s5wy, 0])

    # Ls4 = np.array([0, 0, 0, 0, 0, 0])
    # Ls5 = np.array([0, 0, 0, 0, 0, 0])
    L = np.vstack((Lxn, Lyn, Lan, Ls4, Ls5, Lth))
    return L


def get_vcam(e, L):
    if isinstance(e, list) or isinstance(e, tuple):
        e = np.array([e]).T
    vcam = np.linalg.inv(L).dot(e)
    vx, vy, vz, wx, wy, wz = vcam[:, 0]
    return vx, vy, vz, wx, wy, wz
