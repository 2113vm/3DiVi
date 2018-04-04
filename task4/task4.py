import os
from glob import glob
from typing import List
import random

import numpy as np
import cv2
from scipy import optimize as opt
import matplotlib.pyplot as plt


data_dir = 'data/'
image_names = [data_dir + 'marker' + str(num) + '.jpg' for num in range(21)]
images = [cv2.imread(image_name) for image_name in image_names]

# camera parameters
cx = 320.0
cy = 240.0
fx = 575.7
fy = 575.7

matrix_transform = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# marker size in mm (internal 4 single-colored rectangles)
markerWidth = 52.5
markerHeight = 31.0

# roi of marker
rx = 247
ry = 187
rw = 63
rh = 53

# marker coords in start
P = np.array([[-52.5, 31.0, 100],
              [0, 31.0, 100],
              [52.5, 31.0, 100],
              [-52.5, 0, 100],
              [0, 0, 100],
              [52.5, 0, 100],
              [-52.5, -31.0, 100],
              [0, -31.0, 100],
              [52.5, -31.0, 100]])

class NLOptimization:

    def __init__(self, images: List,
                 start_rvectvec: np.array,
                 start_roi: List,
                 objectPoint: np.array,
                 matrix_transform: np.array):
        self.images = images
        self.lst_rvectvec = [start_rvectvec]
        self.start_roi = [start_roi]
        self.objectPoint = objectPoint
        self.corners = []
        self.matrix_transform = matrix_transform

    def set_match(self, pxs: np.array):
        """
        Функция задает порядок точек.
        :param pxs: 2D-array, 9x2 shape
        :return: Возвращает точки в нужном порядке
        """
        pxs_list = pxs.tolist()
        assert len(pxs_list) == 9, len(pxs_list)
        pxs_list = sorted(pxs_list, key=lambda px: px[0])
        index = np.arange(0, 9).reshape((3, 3)).transpose().reshape(9)
        return np.array([pxs_list[x] for x in index[::-1]])

    def erros_px(self, matrix_rt: np.array):
        """
        Calculate error for R_i and T_i without qrt
        :param pxs2: 8x2 shape matrix. Pxs on image.
        :param pxs3: 8x3 shape matrix. Pxs in real world.
        :param matrix_rt: 1x6 shape matrix. Rotate and Transfer.
        :return: 1x18 shape matrix errors
        """

        def operator(matrix: np.array):
            """
            Operator
            :param matrix: (a, b, c)
            :return: (a/c, b/c)
            """
            if (matrix[:, 2] == 0).any():
                raise ZeroDivisionError('matrix[2] == 0')

            # matrix[:, 0] = matrix[:, 0] / matrix[:, 2]
            # matrix[:, 1] = matrix[:, 1] / matrix[:, 2]
            matrix = matrix / matrix[:, 2].reshape((9, 1))
            return matrix[:, :2]

        def camera_transform(pxs: np.array):
            return pxs.dot(matrix_transform.transpose())

        matrix_r, matrix_t = cv2.Rodrigues(matrix_rt[:3])[0].transpose(), matrix_rt[3:]
        cm = camera_transform(self.objectPoint @ matrix_r + matrix_t)
        op = operator(cm)
        er = op - self.corners[-1]

        return er.reshape((1, 18))[0]

    def transform(self, matrix_rt: np.array):
        """
        Calculate error for R_i and T_i without qrt
        :param matrix_rt: 1x6 shape matrix. Rotate and Transfer.
        :return: pxs2: 8x2 shape matrix .Project pxs3 on image with help matrix_rt
        """

        def operator(matrix: np.array):
            """
            Operator
            :param matrix: (a, b, c)
            :return: (a/c, b/c)
            """
            if (matrix[:, 2] == 0).any():
                raise ZeroDivisionError('matrix[2] == 0')

            matrix[:, 0] = matrix[:, 0] / matrix[:, 2]
            matrix[:, 1] = matrix[:, 1] / matrix[:, 2]
            return matrix[:, :2]

        def camera_transform(pxs: np.array):
            return pxs.dot(self.matrix_transform.transpose())

        matrix_r, matrix_t = cv2.Rodrigues(matrix_rt[:3])[0].transpose(), matrix_rt[3:]
        cm = camera_transform(self.objectPoint @ matrix_r + matrix_t)
        op = operator(cm)
        return op

    def get_corners(self, roi):
        """
        Function return coords of good features on image
        :param image:
        :return: matrix 9x2 shape (now returned 9x1x2)
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image=gray,
                                          maxCorners=20,
                                          qualityLevel=0.01,
                                          minDistance=6)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        new_corners = cv2.cornerSubPix(image=gray,
                                       corners=np.array(corners),
                                       winSize=(5, 5),
                                       zeroZone=(-1, -1),
                                       criteria=criteria)
        new_corners = new_corners[:9]
        size = new_corners.size
        return new_corners.reshape((int(size / 2), 2, 1))[:, :, 0]

    def get_roi(self, image: np.array, coords: List):
        x, y, w, h = coords
        return image[y:y+h, x:x+w]

    def jac(self, x):
        """
        Function return Jacobian
        :param x: vector shape 1x6. rvec + tvec
        :return: Jacobian matrix shape 18x6
        """
        return cv2.projectPoints(objectPoints=self.objectPoint,
                                 rvec=x[:3],
                                 tvec=x[3:],
                                 cameraMatrix=self.matrix_transform,
                                 distCoeffs=None)[1][:, :6]

    @staticmethod
    def draw_pxs(image, pxs, color):
        for px in pxs:
            x, y = int(px[0]), int(px[1])
            cv2.circle(image, (x, y), 2, color)

    @staticmethod
    def draw_roi(image, coords, color):
        pt1 = coords[0], coords[1]
        pt2 = coords[0] + coords[2], coords[1] + coords[3]
        cv2.rectangle(image, pt1=pt1, pt2=pt2, color=color)

    @staticmethod
    def draw_cube(img, corners):
        top = min(corners, key=lambda x: x[1])
        bot = max(corners, key=lambda x: x[1])
        left = min(corners, key=lambda x: x[0])
        right = max(corners, key=lambda x: x[0])
        pxs = np.array([left, top, right, bot])
        pxs = np.int0(pxs)
        cv2.drawContours(img, [pxs], 0, (0, 0, 255), 2)
        # imgpts = np.int32(imgpts).reshape(-1, 2)
        # # draw ground floor in green
        # img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
        # # draw pillars in blue color
        # for i, j in zip(range(4), range(4, 8)):
        #     img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
        # # draw top layer in red color
        # img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
        return img

    def add_roi(self, corners: np.array, c=50):
        # x_min = int(corners[:, 0].min() - 2)
        # x_max = int(corners[:, 0].max() + 2)
        # y_min = int(corners[:, 1].min() - 2)
        # y_max = int(corners[:, 1].max() + 2)
        mean = np.mean(corners, axis=1)
        mx, my = int(mean[0]), int(mean[1])
        x_min, y_min, x_max, y_max = mx-c, my-c, mx+c, my+c
        self.start_roi.append([x_min, y_min, x_max-x_min, y_max-y_min])

    def task4(self):
        for image in self.images:
            print('---')
            roi = self.get_roi(image=image, coords=self.start_roi[-1])
            corners = self.get_corners(roi)
            delta_x, delta_y = np.array(self.start_roi[-1][:2])
            # x and y or y and x?
            corners = corners + np.array([delta_x, delta_y])
            contour_area = cv2.contourArea(corners.astype(np.float32))
            print(contour_area)
            # self.draw_roi(image, self.start_roi[-1], 255)
            # self.draw_cube(image, corners)
            self.corners.append(self.set_match(corners))
            optim = opt.least_squares(fun=self.erros_px,
                                      x0=self.lst_rvectvec[-1],
                                      jac=self.jac,
                                      method='lm')
            self.lst_rvectvec.append(optim.x)
            self.add_roi(self.transform(optim.x), 80)
            self.corners.append(self.transform(optim.x))
            self.draw_pxs(image, pxs=self.transform(optim.x), color=255)
            plt.imshow(image)
            plt.show()


nlo = NLOptimization(images=images,
                     start_rvectvec=np.array([1, 1, -0.33, 20, 30, 1000], dtype=np.float32),
                     start_roi=[rx, ry, rw, rh],
                     matrix_transform=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                     objectPoint=P)

nlo.task4()
