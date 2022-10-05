# This is copied from camera.py, updated for python 3
# https://github.com/smidm/camera.py/blob/master/camera.py

import numpy as np
import math
import yaml
from scipy.special import cbrt
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar
from warnings import warn

try:
    import cv2
except ImportError:
    warn('OpenCV not found, OpenCV camera model will be not available.')

# Bibliography:
# [1] Sara R. Matousek M. 3D Computer Vision. January 7, 2014.
#     Online: http://cmp.felk.cvut.cz/cmp/courses/TDV/2013W/lectures/tdv-2013-all.pdf


def p2e(projective):
    """
    Convert 2d or 3d projective to euclidean coordinates.
    :param projective: projective coordinate(s)
    :type projective: numpy.ndarray, shape=(3 or 4, n)
    :return: euclidean coordinate(s)
    :rtype: numpy.ndarray, shape=(2 or 3, n)
    """
    assert(type(projective) == np.ndarray)
    assert((projective.shape[0] == 4) | (projective.shape[0] == 3))
    return (projective / projective[-1, :])[0:-1, :]


def e2p(euclidean):
    """
    Convert 2d or 3d euclidean to projective coordinates.
    :param euclidean: projective coordinate(s)
    :type euclidean: numpy.ndarray, shape=(2 or 3, n)
    :return: projective coordinate(s)
    :rtype: numpy.ndarray, shape=(3 or 4, n)
    """
    assert(type(euclidean) == np.ndarray)
    assert((euclidean.shape[0] == 3) | (euclidean.shape[0] == 2))
    return np.vstack((euclidean, np.ones((1, euclidean.shape[1]))))


def column(vector):
    """
    Return column vector.
    :param vector: np.ndarray
    :return: column vector
    :rtype: np.ndarray, shape=(n, 1)
    """
    return vector.reshape((-1, 1))

# line routines, slope-intercept form y = m * x + c


def fit_line(xy):
    """
    Fit line to points.
    :param xy: point coordinates
    :type xy: np.ndarray, shape=(2, n)
    :return: line parameters [m, c]
    :rtype mc: array like
    """
    assert xy.shape[0] == 2
    x = column(xy[0, :])
    y = column(xy[1, :])
    a = np.hstack((x, np.ones((xy.shape[1], 1))))
    return np.linalg.lstsq(a, y)[0]


def line_point_distance(xy, mc):
    """
    Distance from point(s) to line.
    :param xy: point coordinates
    :type xy: np.ndarray, shape=(2, n)
    :param mc: line parameters [m, c]
    :type mc: array like
    :return: distance(s)
    :rtype: np.ndarray, shape=(n,)
    """
    m = mc[0]   # slope
    c = mc[1]   # intercept
    return (xy[0, :] * m - xy[1, :] + c) / (m ** 2 + 1)


def nearest_point_on_line(xy, mc):
    """
    Nearest point(s) to line.
    :param xy: point coordinates
    :type xy: np.ndarray, shape=(2, n)
    :param mc: line parameters [m, c]
    :type mc: array like
    :return: point(s) on line
    :rtype: np.ndarray, shape=(2, n)
    """
    m = mc[0]   # slope
    c = mc[1]   # intercept
    x = (xy[0, :] + xy[1, :] * m - c * m) / (m ** 2 + 1)
    y = m * x + c
    return np.array([x, y])

class Camera:
    """
    Projective camera model
        - camera intrinsic and extrinsic parameters handling
        - various lens distortion models
        - model persistence
        - projection of camera coordinates to an image
        - conversion of image coordinates on a plane to camera coordinates
        - visibility handling
    """

    def __init__(self, id=None):
        """
        :param id: camera identification number
        :type id: unknown or int
        """
        self.K = np.eye(3)  # camera intrinsic parameters
        self.Kundistortion = np.array([])  # could be altered based on K using set_undistorted_view(alpha)
        #  to get undistorted image with all / corner pixels visible
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.kappa = np.zeros((2,))
        self.id = id
        self.size_px = np.zeros((2,))
        # self.size_px_view = np.zeros((2,))  #

        self.bouguet_kc = np.zeros((5,))
        self.kannala_p = np.zeros((6,))
        self.kannala_thetamax = None
        self.division_lambda = 0.
        self.division_z_n = -1
        self.tsai_f = -1
        self.tsai_kappa = -1
        self.tsai_ncx = -1
        self.tsai_nfx = -1
        self.tsai_dx = -1
        self.tsai_dy = -1
        self.opencv_dist_coeff = None
        self.calibration_type = 'standard'  # other possible values: bouguet, kannala, division, opencv
        self.update_P()

    def save(self, filename):
        """
        Save camera model to a YAML file.
        """
        data = {'id': self.id,
                'K': self.K.tolist(),
                'R': self.R.tolist(),
                't': self.t.tolist(),
                'size_px': self.size_px.tolist(),
                'calibration_type': self.calibration_type
                }
        if self.Kundistortion.size != 0:
            data['Kundistortion'] = self.Kundistortion.tolist()
        if self.calibration_type == 'bouguet':
            data['bouguet_kc'] = self.bouguet_kc.tolist()
        elif self.calibration_type == 'kannala':
            data['kannala_p'] = self.kannala_p.tolist()
            data['kannala_thetamax'] = self.kannala_thetamax
        elif self.calibration_type == 'tsai':
            data_tsai = {'tsai_f': self.tsai_f,
                         'tsai_kappa': self.tsai_kappa,
                         'tsai_nfx': self.tsai_nfx,
                         'tsai_dx': self.tsai_dx,
                         'tsai_dy': self.tsai_dy,
                         'tsai_ncx': self.tsai_ncx,
                         }
            data.update(data_tsai)
        elif self.calibration_type == 'division':
            data['division_lambda'] = self.division_lambda
            data['division_z_n'] = self.division_z_n
        elif self.calibration_type == 'opencv' or self.calibration_type == 'opencv_fisheye':
            data['opencv_dist_coeff'] = self.opencv_dist_coeff.tolist()
        else:
            data['kappa'] = self.kappa.tolist()
        yaml.dump(data, open(filename, 'w'))

    def load(self, filename):
        """
        Load camera model from a YAML file.
        Example::
            calibration_type: standard
            K:
            - [1225.2, -7.502186291576686e-14, 480.0]
            - [0.0, 1225.2, 384.0]
            - [0.0, 0.0, 1.0]
            R:
            - [-0.9316877145365, -0.3608289515885, 0.002545329627547]
            - [-0.1725273110187, 0.4247524018287, -0.8888909933995]
            - [0.3296724908378, -0.8263880720441, -0.4579894432589]
            id: 0
            kappa: [0.0, 0.0]
            size_px: [960, 768]
            t:
            - [-1.365061486465]
            - [3.431608806127]
            - [17.74182159488]
        """
        data = yaml.load(open(filename))
        if 'id' in data:
            self.id = data['id']
        if 'K' in data:
            self.K = np.array(data['K']).reshape((3, 3))
        if 'R' in data:
            self.R = np.array(data['R']).reshape((3, 3))
        if 't' in data:
            self.t = np.array(data['t']).reshape((3, 1))
        if 'size_px' in data:
            self.size_px = np.array(data['size_px']).reshape((2,))
        if 'calibration_type' in data:
            self.calibration_type = data['calibration_type']
        if 'Kundistortion' in data:
            self.Kundistortion = np.array(data['Kundistortion'])
        else:
            self.Kundistortion = self.K
        if self.calibration_type == 'bouguet':
            self.bouguet_kc = np.array(data['bouguet_kc']).reshape((5,))
        elif self.calibration_type == 'kannala':
            self.kannala_p = np.array(data['kannala_p']).reshape((6,))
            self.kannala_thetamax = data['kannala_thetamax']  # not used now
            # Focal length actually used is from kannala_p. Why then K is stored? Works for me like this.
            self.K[0, 0] = self.kannala_p[2]
            self.K[1, 1] = self.kannala_p[3]
            # principal point in K and kannala_p[4:] should be consistent
            assert self.K[0, 2] == self.kannala_p[4]
            assert self.K[1, 2] == self.kannala_p[5]
        elif self.calibration_type == 'tsai':
            self.tsai_f = data['tsai_f']
            self.tsai_kappa = data['tsai_kappa']
            self.tsai_ncx = data['tsai_ncx']
            self.tsai_nfx = data['tsai_nfx']
            self.tsai_dx = data['tsai_dx']
            self.tsai_dy = data['tsai_dy']
        elif self.calibration_type == 'division':
            self.division_lambda = data['division_lambda']
            self.division_z_n = data['division_z_n']
        elif self.calibration_type == 'opencv' or self.calibration_type == 'opencv_fisheye':
            self.opencv_dist_coeff = np.array(data['opencv_dist_coeff'])
        elif self.calibration_type == 'standard':
            self.kappa = np.array(data['kappa']).reshape((2,))
        if 'id' not in data and \
                        'K' not in data and \
                        'R' not in data and \
                        't' not in data and \
                        'size_px' not in data and \
                        'calibration_type' not in data and \
                        'Kundistortion' not in data:
            warn('Nothing loaded from %s, check the contents.' % filename)
        self.update_P()

    def update_P(self):
        """
        Update camera P matrix from K, R and t.
        """
        self.P = self.K.dot(np.hstack((self.R, self.t)))

    def set_K(self, K):
        """
        Set K and update P.
        :param K: intrinsic camera parameters
        :type K: numpy.ndarray, shape=(3, 3)
        """
        self.K = K
        self.update_P()

    def set_K_elements(self, u0_px, v0_px, f=1, theta_rad=math.pi/2, a=1):
        """
        Update pinhole camera intrinsic parameters and updates P matrix.
        :param u0_px: principal point x position (pixels)
        :type u0_px: double
        :param v0_px: principal point y position (pixels)
        :type v0_px: double
        :param f: focal length
        :type f: double
        :param theta_rad: digitization raster skew (radians)
        :type theta_rad: double
        :param a: pixel aspect ratio
        :type a: double
        """
        self.K = np.array([[f, -f * 1 / math.tan(theta_rad), u0_px],
                      [0, f / (a * math.sin(theta_rad)), v0_px],
                      [0, 0, 1]])
        self.update_P()

    def set_R(self, R):
        """
        Set camera extrinsic parameters and updates P.
        :param R: camera extrinsic parameters matrix
        :type R: numpy.ndarray, shape=(3, 3)
        """
        self.R = R
        self.update_P()

    def set_R_euler_angles(self, angles):
        """
        Set rotation matrix according to euler angles and updates P.
        :param angles: 3 euler angles in radians,
        :type angles: double sequence, len=3
        """
        rx = angles[0]
        ry = angles[1]
        rz = angles[2]
        from numpy import sin
        from numpy import cos
        self.R = np.array([[cos(ry) * cos(rz),
                            cos(rz) * sin(rx) * sin(ry) - cos(rx) * sin(rz),
                            sin(rx) * sin(rz) + cos(rx) * cos(rz) * sin(ry)],
                           [cos(ry) * sin(rz),
                            sin(rx) * sin(ry) * sin(rz) + cos(rx) * cos(rz),
                            cos(rx) * sin(ry) * sin(rz) - cos(rz) * sin(rx)],
                           [-sin(ry),
                            cos(ry) * sin(rx),
                            cos(rx) * cos(ry)]
                           ])
        self.update_P()

    def set_t(self, t):
        """
        Set camera translation and updates P.
        :param t: camera translation vector
        :type t: numpy.ndarray, shape=(3, 1)
        """
        self.t = t
        self.update_P()

    def get_K_0(self):
        """
        Return ideal calibration matrix (only focal length present).
        :return: ideal calibration matrix
        :rtype: np.ndarray, shape=(3, 3)
        """
        K_0 = np.eye(3)
        K_0[0, 0] = self.get_focal_length()
        K_0[1, 1] = self.get_focal_length()
        return K_0

    def get_A(self, K=None):
        """
        Return part of K matrix that applies center, skew and aspect ratio to ideal image coordinates.
        :rtype: np.ndarray, shape=(3, 3)
        """
        if K is None:
            K = self.K
        A = K.copy()
        A[0, 0] /= self.get_focal_length()
        A[0, 1] /= self.get_focal_length()
        A[1, 1] /= self.get_focal_length()
        return A

    def get_z0_homography(self, K=None):
        """
        Return homography from world plane at z = 0 to image plane.
        :return: 2d plane homography
        :rtype: np.ndarray, shape=(3, 3)
        """
        if K is None:
            K = self.K
        return K.dot(np.hstack((self.R, self.t)))[:, [0, 1, 3]]

    def undistort_image(self, img, Kundistortion=None):
        """
        Transform grayscale image such that radial distortion is removed.
        :param img: input image
        :type img: np.ndarray, shape=(n, m) or (n, m, 3)
        :param Kundistortion: camera matrix for undistorted view, None for self.K
        :type Kundistortion: array-like, shape=(3, 3)
        :return: transformed image
        :rtype: np.ndarray, shape=(n, m) or (n, m, 3)
        """
        if Kundistortion is None:
            Kundistortion = self.K
        if self.calibration_type == 'opencv':
            return cv2.undistort(img, self.K, self.opencv_dist_coeff, newCameraMatrix=Kundistortion)
        elif self.calibration_type == 'opencv_fisheye':
                return cv2.fisheye.undistortImage(img, self.K, self.opencv_dist_coeff, Knew=Kundistortion)
        else:
            xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
            img_coords = np.array([xx.ravel(), yy.ravel()])
            y_l = self.undistort(img_coords, Kundistortion)
            if img.ndim == 2:
                return griddata(y_l.T, img.ravel(), (xx, yy), fill_value=0, method='linear')
            else:
                channels = [griddata(y_l.T, img[:, :, i].ravel(), (xx, yy), fill_value=0, method='linear')
                            for i in range(img.shape[2])]
                return np.dstack(channels)

    def undistort(self, distorted_image_coords, Kundistortion=None):
        """
        Remove distortion from image coordinates.
        :param distorted_image_coords: real image coordinates
        :type distorted_image_coords: numpy.ndarray, shape=(2, n)
        :param Kundistortion: camera matrix for undistorted view, None for self.K
        :type Kundistortion: array-like, shape=(3, 3)
        :return: linear image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert distorted_image_coords.shape[0] == 2
        assert distorted_image_coords.ndim == 2
        if Kundistortion is None:
            Kundistortion = self.K
        if self.calibration_type == 'division':
            A = self.get_A(Kundistortion)
            Ainv = np.linalg.inv(A)
            undistorted_image_coords = p2e(A.dot(e2p(self._undistort_division(p2e(Ainv.dot(e2p(distorted_image_coords)))))))
        elif self.calibration_type == 'opencv':
            undistorted_image_coords = cv2.undistortPoints(distorted_image_coords.T.reshape((1, -1, 2)),
                                                           self.K, self.opencv_dist_coeff,
                                                           P=Kundistortion).reshape(-1, 2).T
        elif self.calibration_type == 'opencv_fisheye':
            undistorted_image_coords = cv2.fisheye.undistortPoints(distorted_image_coords.T.reshape((1, -1, 2)),
                                                                   self.K, self.opencv_dist_coeff,
                                                                   P=Kundistortion).reshape(-1, 2).T
        else:
            warn('undistortion not implemented')
            undistorted_image_coords = distorted_image_coords
        assert undistorted_image_coords.shape[0] == 2
        assert undistorted_image_coords.ndim == 2
        return undistorted_image_coords

    def distort(self, undistorted_image_coords, Kundistortion=None):
        """
        Apply distortion to ideal image coordinates.
        :param undistorted_image_coords: ideal image coordinates
        :type undistorted_image_coords: numpy.ndarray, shape=(2, n)
        :param Kundistortion: camera matrix for undistorted coordinates, None for self.K
        :type Kundistortion: array-like, shape=(3, 3)
        :return: distorted image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert undistorted_image_coords.shape[0] == 2
        assert undistorted_image_coords.ndim == 2
        if Kundistortion is None:
            Kundistortion = self.K
        if self.calibration_type == 'division':
            A = self.get_A(Kundistortion)
            Ainv = np.linalg.inv(A)
            distorted_image_coords = p2e(A.dot(e2p(self._distort_division(p2e(Ainv.dot(e2p(undistorted_image_coords)))))))
        elif self.calibration_type == 'opencv':
            undistorted_image_coords_norm = (undistorted_image_coords - column(Kundistortion[0:2, 2])) / \
                                            column(Kundistortion.diagonal()[0:2])
            undistorted_image_coords_3d = np.vstack((undistorted_image_coords_norm,
                                                     np.zeros((1, undistorted_image_coords.shape[1]))))
            distorted_image_coords, _ = cv2.projectPoints(undistorted_image_coords_3d.T, (0, 0, 0), (0, 0, 0),
                                                          self.K, self.opencv_dist_coeff)
            distorted_image_coords = distorted_image_coords.reshape(-1, 2).T
        elif self.calibration_type == 'opencv_fisheye':
            # if self.Kundistortion is not np.array([]):
            #     # remove Kview transformation
            #     undistorted_image_coords = p2e(np.matmul(np.linalg.inv(self.Kundistortion),
            #                                              e2p(undistorted_image_coords)))
            # TODO check correctness
            undistorted_image_coords = p2e(np.matmul(np.linalg.inv(Kundistortion),
                                                     e2p(undistorted_image_coords)))
            distorted_image_coords = cv2.fisheye.distortPoints(undistorted_image_coords.T.reshape((1, -1, 2)),
                                                               self.K, self.opencv_dist_coeff).reshape(-1, 2).T
        else:
            assert False  # not implemented
        assert distorted_image_coords.shape[0] == 2
        assert distorted_image_coords.ndim == 2
        return distorted_image_coords

    def _distort_bouguet(self, undistorted_centered_image_coord):
        """
        Distort centered image coordinate following Bouquet model.
        see http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
        :param undistorted_centered_image_coord: linear centered image coordinate(s)
        :type undistorted_centered_image_coord: numpy.ndarray, shape=(2, n)
        :return: distorted coordinate(s)
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert undistorted_centered_image_coord.shape[0] == 2
        kc = self.bouguet_kc
        x = undistorted_centered_image_coord[0, :]
        y = undistorted_centered_image_coord[1, :]
        r_squared = x ** 2 + y ** 2

        # tangential distortion vector
        dx = np.array([2 * kc[2] * x * y + kc[3] * (r_squared + 2 * x ** 2),
                       kc[2] * (r_squared + 2 * y ** 2) + 2 * kc[3] * x * y])
        distorted = (1 + kc[0] * r_squared + kc[1] * r_squared ** 2 + kc[4] * r_squared ** 3) * \
            undistorted_centered_image_coord + dx
        return distorted

    def _distort_kannala(self, camera_coords):
        """
        Distort image coordinate following Kannala model (M6 version only)
        See http://www.ee.oulu.fi/~jkannala/calibration/calibration_v23.tar.gz :genericproj.m
        Juho Kannala, Janne Heikkila and Sami S. Brandt. Geometric camera calibration. Wiley Encyclopedia of Computer Science and Engineering, 2008, page 9.
        :param camera_coords: 3d points in camera coordinates
        :type camera_coords: numpy.ndarray, shape=(3, n)
        :return: distorted metric image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert camera_coords.shape[0] == 3
        x = camera_coords[0, :]
        y = camera_coords[1, :]
        z = camera_coords[2, :]
        k1 = self.kannala_p[0]
        k2 = self.kannala_p[1]

        # angle between ray and optical axis
        theta = np.arccos(z / np.linalg.norm(camera_coords, axis=0))

        # radial projection (Kannala 2008, eq. 17)
        r = k1 * theta + k2 * theta ** 3

        hypotenuse = np.linalg.norm(camera_coords[0:2, :], axis=0)
        hypotenuse[hypotenuse == 0] = 1  # avoid dividing by zero
        image_x = r * x / hypotenuse
        image_y = r * y / hypotenuse
        return np.vstack((image_x, image_y))

    def _undistort_tsai(self, distorted_metric_image_coord):
        """
        Undistort centered image coordinate following Tsai model.
        :param distorted_metric_image_coord: distorted METRIC image coordinates
            (metric image coordiante = image_xy * f / z)
        :type distorted_metric_image_coord: numpy.ndarray, shape=(2, n)
        :return: linear image coordinate(s)
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert distorted_metric_image_coord.shape[0] == 2
        # see http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DIAS1/
        x = distorted_metric_image_coord[0, :]
        y = distorted_metric_image_coord[1, :]
        r_squared = x ** 2 + y ** 2

        undistorted = (1 + self.tsai_kappa * r_squared) * distorted_metric_image_coord
        return undistorted

    def _distort_tsai(self, metric_image_coord):
        """
        Distort centered metric image coordinates following Tsai model.
        See: Devernay, Frederic, and Olivier Faugeras. "Straight lines have to be straight."
        Machine vision and applications 13.1 (2001): 14-24. Section 2.1.
        (only for illustration, the formulas didn't work for me)
        http://www.cvg.rdg.ac.uk/PETS2009/sample.zip :CameraModel.cpp:CameraModel::undistortedToDistortedSensorCoord
        Analytical inverse of the undistort_tsai() function.
        :param metric_image_coord: centered metric image coordinates
            (metric image coordinate = image_xy * f / z)
        :type metric_image_coord: numpy.ndarray, shape=(2, n)
        :return: distorted centered metric image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert metric_image_coord.shape[0] == 2
        x = metric_image_coord[0, :]  # vector
        y = metric_image_coord[1, :]  # vector
        r_u = np.sqrt(x ** 2 + y ** 2)  # vector
        c = 1.0 / self.tsai_kappa  # scalar
        d = -c * r_u  # vector

        # solve polynomial of 3rd degree for r_distorted using Cardan method:
        # https://proofwiki.org/wiki/Cardano%27s_Formula
        # r_distorted ** 3 + c * r_distorted + d = 0
        q = c / 3.  # scalar
        r = -d / 2.  # vector
        delta = q ** 3 + r ** 2  # polynomial discriminant, vector

        positive_mask = delta >= 0
        r_distorted = np.zeros((metric_image_coord.shape[1]))

        # discriminant > 0
        s = cbrt(r[positive_mask] + np.sqrt(delta[positive_mask]))
        t = cbrt(r[positive_mask] - np.sqrt(delta[positive_mask]))
        r_distorted[positive_mask] = s + t

        # discriminant < 0
        delta_sqrt = np.sqrt(-delta[~positive_mask])
        s = cbrt(np.sqrt(r[~positive_mask] ** 2 + delta_sqrt ** 2))
        # s = cbrt(np.sqrt(r[~positive_mask] ** 2 + (-delta[~positive_mask]) ** 2))
        t = 1. / 3 * np.arctan2(delta_sqrt, r[~positive_mask])
        r_distorted[~positive_mask] = -s * np.cos(t) + s * np.sqrt(3) * np.sin(t)

        return metric_image_coord * r_distorted / r_u
        
    def _undistort_division(self, z_r):
        """
        Undistort centered image coordinate(s) following the division model.
        :param z_r: radially distorted centered image coordinate(s)
        :type z_r: numpy.ndarray, shape(2, n)
        
        :return: linear image coordinate(s)
        :rtype: numpy.ndarray, shape(2, n)        
        """
        assert (-1 < self.division_lambda < 1)
        return (1 - self.division_lambda) / \
               (1 - self.division_lambda * np.sum(z_r ** 2, axis=0) / self.division_z_n ** 2) * z_r

    def _distort_division(self, z_l):
        """
        Distort centered image coordinate(s) following the division model.
        :param z_l: linear centered image coordinate(s)
        :type z_l: numpy.ndarray, shape(2, n)
        :return: radially distorted image coordinate(s)
        :rtype: numpy.ndarray, shape(2, n)
        """
        z_hat = 2 * z_l / (1 - self.division_lambda)
        return z_hat / (1 + np.sqrt(1 + self.division_lambda * np.sum(z_hat ** 2, axis=0) /
                                    np.sum(self.division_z_n ** 2, axis=0)))
    
    def get_focal_length(self):
        """
        Get camera focal length.
        :return: focal length
        :rtype: double
        """
        return self.K[0, 0]

    def get_principal_point_px(self):
        """
        Get camera principal point.
        :return: x and y pixel coordinates
        :rtype: numpy.ndarray, shape=(1, 2)
        """
        return self.K[0:2, 2].reshape((1, 2))

    def is_visible(self, xy_px):
        """
        Check visibility of image points.
        :param xy_px: image point(s)
        :type xy_px: np.ndarray, shape=(2, n)
        :return: visibility of image points
        :rtype: numpy.ndarray, shape=(1, n), dtype=bool
        """
        assert xy_px.shape[0] == 2
        return (xy_px[0, :] >= 0) & (xy_px[1, :] >= 0) & \
               (xy_px[0, :] < self.size_px[0]) & \
               (xy_px[1, :] < self.size_px[1])

    def is_visible_world(self, world):
        """
        Check visibility of world points.
        :param world: world points
        :type world: numpy.ndarray, shape=(3, n)
        :return: visibility of world points
        :rtype: numpy.ndarray, shape=(1, n), dtype=bool
        """
        assert world.shape[0] == 3
        xy_px = p2e(self.world_to_image(world))
        return self.is_visible(xy_px)

    def get_camera_center(self):
        """
        Returns camera center in the world coordinates.
        :return: camera center in projective coordinates
        :rtype: np.ndarray, shape=(4, 1)
        """
        return self._null(self.P)

    def world_to_image(self, world):
        """
        Project world coordinates to image coordinates.
        :param world: world points in 3d projective or euclidean coordinates
        :type world: numpy.ndarray, shape=(3 or 4, n)
        :return: projective image coordinates
        :rtype: numpy.ndarray, shape=(3, n)
        """
        assert(type(world) == np.ndarray)
        if self.calibration_type == 'opencv' or self.calibration_type == 'opencv_fisheye':
            if world.shape[0] == 4:
                world = p2e(world)
            if self.calibration_type == 'opencv':
                distorted_image_coords = cv2.projectPoints(world.T, self.R, self.t,
                                                           self.K, self.opencv_dist_coeff)[0].reshape(-1, 2).T
            else:
                distorted_image_coords = cv2.fisheye.projectPoints(
                        world.T.reshape((1, -1, 3)), cv2.Rodrigues(self.R)[0],
                        self.t, self.K, self.opencv_dist_coeff)[0].reshape(-1, 2).T
            return e2p(distorted_image_coords)
        if world.shape[0] == 3:
            world = e2p(world)
        camera_coords = np.hstack((self.R, self.t)).dot(world)
        if self.calibration_type == 'bouguet':
            xy = camera_coords[0:2, :]
            z = camera_coords[2, :]
            image_coords_metric = xy / z
            image_coords_distorted_metric = self._distort_bouguet(image_coords_metric)
            return self.K.dot(e2p(image_coords_distorted_metric))
        elif self.calibration_type == 'tsai':
            xy = camera_coords[0:2, :]
            z = camera_coords[2, :]
            image_coords_metric = xy * self.tsai_f / z
            image_coords_distorted_metric = self._distort_tsai(image_coords_metric)
            return self.K.dot(e2p(image_coords_distorted_metric))
        elif self.calibration_type == 'kannala':
            image_coords_distorted_metric = self._distort_kannala(camera_coords)
            return self.K.dot(e2p(image_coords_distorted_metric))
        elif self.calibration_type == 'division':
            # see [1, page 54]
            return self.get_A().dot(e2p(self._distort_division(p2e(self.get_k0().dot(camera_coords)))))
        else:
            xy = camera_coords[0:2, :]
            z = camera_coords[2, :]
            image_coords_distorted_metric = xy / z
            return self.K.dot(e2p(image_coords_distorted_metric))

    def image_to_world(self, image_px, z):
        """
        Project image points with defined world z to world coordinates.
        :param image_px: image points
        :type image_px: numpy.ndarray, shape=(2 or 3, n)
        :param z: world z coordinate of the projected image points
        :type z: float
        :return: n projective world coordinates
        :rtype: numpy.ndarray, shape=(3, n)
        """
        if image_px.shape[0] == 3:
            image_px = p2e(image_px)
        image_undistorted = self.undistort(image_px)
        tmpP = np.hstack((self.P[:, [0, 1]], self.P[:, 2, np.newaxis] * z + self.P[:, 3, np.newaxis]))
        world_xy = p2e(np.linalg.inv(tmpP).dot(e2p(image_undistorted)))
        return np.vstack((world_xy, z * np.ones(image_px.shape[1])))

    def get_view_matrix(self, alpha):
        """
        Returns camera matrix for handling image and coordinates distortion and undistortion. Based on alpha,
        up to all pixels of the distorted image can be visible in the undistorted image.
        :param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1
                      (when all the source image pixels are retained in the undistorted image). For convenience for -1
                      returns custom camera matrix self.Kundistortion and None returns self.K.
        :type alpha: float or None
        :return: camera matrix for a view defined by alpha
        :rtype: array, shape=(3, 3)
        """
        if alpha == -1:
            Kundistortion = self.Kundistortion
        elif alpha is None:
            Kundistortion = self.K
        elif self.calibration_type == 'opencv':
            Kundistortion, _ = cv2.getOptimalNewCameraMatrix(self.K, self.opencv_dist_coeff, tuple(self.size_px), alpha)
        elif self.calibration_type == 'opencv_fisheye':
            Kundistortion = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.opencv_dist_coeff,
                                                                                   tuple(self.size_px), self.R,
                                                                                   balance=alpha)
        else:
            # TODO
            assert False, 'not implemented'
        return Kundistortion

    def plot_world_points(self, points, plot_style, label=None,
                          solve_visibility=True):
        """
        Plot world points to a matplotlib figure.
        :param points: world points (projective or euclidean)
        :type points: numpy.ndarray, shape=(3 or 4, n) or list of lists
        :param plot_style: matplotlib point and line style code, e.g. 'ro'
        :type plot_style: str
        :param label: label plotted under points mean
        :type label: str
        :param solve_visibility: if true then plot only if all points are visible
        :type solve_visibility: bool
        """
        object_label_y_shift = +25
        import matplotlib.pyplot as plt

        if type(points) == list:
            points = np.array(points)
        points = np.atleast_2d(points)
        image_points_px = p2e(self.world_to_image(points))
        if not solve_visibility or np.all(self.is_visible(image_points_px)):
            plt.plot(image_points_px[0, :],
                     image_points_px[1, :], plot_style)
            if label:
                    max_y = max(image_points_px[1, :])
                    mean_x = image_points_px[0, :].mean()
                    plt.text(mean_x, max_y + object_label_y_shift, label)

    def _null(self, A, eps=1e-15):
        """
        Matrix null space.
        For matrix null space holds: A * null(A) = zeros
        source: http://mail.scipy.org/pipermail/scipy-user/2005-June/004650.html
        :param A: input matrix
        :type A: numpy.ndarray, shape=(m, n)
        :param eps: values lower than eps are considered zero
        :type eps: double
        :return: null space of the matrix A
        :rtype: numpy.ndarray, shape=(n, 1)
        """
        u, s, vh = np.linalg.svd(A)
        n = A.shape[1]   # the number of columns of A
        if len(s) < n:
            expanded_s = np.zeros(n, dtype=s.dtype)
            expanded_s[:len(s)] = s
            s = expanded_s
        null_mask = (s <= eps)
        null_space = np.compress(null_mask, vh, axis=0)
        return np.transpose(null_space)


def nview_linear_triangulation(cameras, correspondences,weights = None):
    """
    Computes ONE world coordinate from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param correspondences: image coordinates correspondences in n views
    :type correspondences: numpy.ndarray, shape=(2, n)
    :return: world coordinate
    :rtype: numpy.ndarray, shape=(3, 1)
    """
    assert(len(cameras) >= 2)
    assert(type(cameras) == list)
    assert(correspondences.shape == (2, len(cameras)))

    def _construct_D_block(P, uv,w=1):
        """
        Constructs 2 rows block of matrix D.
        See [1, p. 88, The Triangulation Problem]
        :param P: camera matrix
        :type P: numpy.ndarray, shape=(3, 4)
        :param uv: image point coordinates (xy)
        :type uv: numpy.ndarray, shape=(2,)
        :return: block of matrix D
        :rtype: numpy.ndarray, shape=(2, 4)
        """

        return w*np.vstack((uv[0] * P[2, :] - P[0, :],
                          uv[1] * P[2, :] - P[1, :]))
    
    # testing weighted least squares
    if weights is None:
        w = np.ones(len(cameras))
        weights = [1 for i in range(len(cameras))]
    else:
        w = [np.nan_to_num(wi,nan=0.5) for wi in weights] # turns nan confidences into 0.5
    
    
    D = np.zeros((len(cameras) * 2, 4))
    for cam_idx, cam, uv in zip(range(len(cameras)), cameras, correspondences.T):
        D[cam_idx * 2:cam_idx * 2 + 2, :] = _construct_D_block(cam.P, uv,w=w[cam_idx])
    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    pt3d = p2e(u[:, -1, np.newaxis])
    weightArray = np.asarray(weights)
    if np.count_nonzero(weights)<2:
        # return 0s if there aren't at least 2 cameras with confidence
        pt3d = np.zeros_like(pt3d)
        conf = 0 
    else:
        # if all nan slice (all cameras were splined)
        if all(np.isnan(weightArray[weightArray!=0])):
            conf=.5 # nans get 0.5 confidence
        else:
            conf = np.nanmean(weightArray[weightArray!=0])

    return pt3d,conf


def nview_linear_triangulations(cameras, image_points,weights=None):
    """
    Computes world coordinates from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param image_points: image coordinates of m correspondences in n views
    :type image_points: sequence of m numpy.ndarray, shape=(2, n)
    :return: m world coordinates
    :rtype: numpy.ndarray, shape=(3, m)
    :weights: numpy.ndarray, shape(nMkrs,nCams)
    """
    assert(type(cameras) == list)
    assert(type(image_points) == list)
    assert(len(cameras) == image_points[0].shape[1])
    assert(image_points[0].shape[0] == 2)

    world = np.zeros((3, len(image_points)))
    confidence = np.zeros((1,len(image_points)))
    for i, correspondence in enumerate(image_points):
        if weights is not None:
            w = [w[i] for w in weights]
        else:
            w = None
        pt3d, conf = nview_linear_triangulation(cameras, correspondence,weights=w)
        world[:, i] = np.ndarray.flatten(pt3d)
        confidence[0,i] = conf
    return world,confidence


def calibrate_division_model(line_coordinates, y0, z_n, focal_length=1):
    """
    Calibrate division model by making lines straight.
    :param line_coordinates: coordinates of points on lines
    :type line_coordinates: np.ndarray, shape=(nlines, npoints_per_line, 2)
    :param y0: radial distortion center xy coordinates
    :type y0: array-like, len=2
    :param z_n: distance to boundary (pincushion: image width / 2, barrel: image diagonal / 2)
    :type z_n: float
    :param focal_length: focal length of the camera (optional)
    :type focal_length: float
    :return: Camera object with calibrated division model parameter lambda
    :rtype: Camera
    """

    def lines_fit_error(p, line_coordinates, cam):
        if not (-1 < p < 1):
            return np.inf
        assert line_coordinates.ndim == 3
        cam.division_lambda = p
        error = 0.
        for line in range(line_coordinates.shape[0]):
            xy = cam.undistort(line_coordinates[line].T)
            mc = fit_line(xy)
            d = line_point_distance(xy, mc)
            nearest_xy = nearest_point_on_line(xy, mc)
            line_length_sq = np.sum((nearest_xy[:, 0] - nearest_xy[:, -1]) ** 2)
            error += np.sum(d ** 2) / line_length_sq / line_coordinates.shape[1]
    #        plt.plot(x, mc[0] * x + mc[1], 'y')
    #        plt.plot(nx, ny, 'y+')
    #        plt.plot(x, y, 'r+')
    #        plt.show()
        return error

    c = Camera()
    c.set_K_elements(u0_px=y0[0], v0_px=y0[1], f=focal_length)
    c.calibration_type = 'division'
    c.division_z_n = z_n
    res = minimize_scalar(lambda p: lines_fit_error(p, line_coordinates, c))
    c.division_lambda = float(res.x)
    return c