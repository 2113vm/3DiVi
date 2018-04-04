import os
import time
import random
import math
from timeit import default_timer as timer
from math import cos, sin, exp

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def camera_transform(image):
    """
    Return transform image for camera
    """
    img = np.zeros((image.shape[0], image.shape[1], 3))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            img[y][x] = np.array([(x - 320) / 575.5 * image[y, x],
                                  (240 - y) / 575.5 * image[y, x],
                                  image[y, x]])
    return img


def get_mb_floor_list(image):
    """
    image[y][x] - it is array of shape 3, where array[0] - real x,
                                                array[1] - real y,
                                                array[2] - real h
    """
    mb_floor_list = dict()

    for x in range(image.shape[1]):
        # image[:, x] - it is one line
        mb_floor_list[x] = list()
        h_ = 0
        for num, h in enumerate(image[::-1, x]):
            if h_ < h[2]:
                h_ = h[2]
                helper = list(h)
                helper.append(image.shape[0]-num)
                mb_floor_list[x].append(helper)

    return mb_floor_list


def find_two_px_max_line(pxs: list, num_iter: int, max_d: float):

    def get_random_px():
        return random.choice(pxs)

    def get_two_random_px(pxs):
        n1, n2 = random.randint(a=0, b=len(pxs)-1), random.randint(a=0, b=len(pxs)-1)
        while n1 == n2:
            n1, n2 = random.randint(a=0, b=len(pxs)-1), random.randint(a=0, b=len(pxs)-1)
        return pxs[n1], pxs[n2]

    def get_d(points: np.array, px: np.array):
        points, px = np.array(points), np.array(px[1:3])
        points = points[:, 1:3]
        if np.equal(points[0], px).all() or np.equal(points[0], px).all():
            return 0
        else:
            m = points[1] - points[0]
            n = np.array([m[1], -m[0]])
            c = -(n @ points[1])
            return abs(n @ px + c) / (n[0] ** 2 + n[1] ** 2) ** 0.5

    def get_counter_line(points, num_iter):
        iter_ = 0
        counter = 0
        points_set = list()
        while iter_ < num_iter:
            point = get_random_px()
            if max_d > get_d(points=points, px=point):
                counter += 1
                points_set.append(point)
            iter_ += 1
        return points_set, counter

    iter_ = 0
    max_count = 0
    max_points = None
    max_points_set = None
    while iter_ < num_iter:
        points = get_two_random_px(pxs=pxs)
        points_set, count = get_counter_line(points=points, num_iter=len(pxs)//10)
        if count > max_count:
            max_count = count
            max_points = points
            max_points_set = points_set
        iter_ += 1
    return max_points, max_points_set


def find_floor_2d(image):
    floor = list()
    img = camera_transform(image)
    mb_floor_list = get_mb_floor_list(img)
    keys = mb_floor_list.keys()
    for key in keys:
        pxs = mb_floor_list[key]
        if len(pxs) > 1:
            _, max_points_set = find_two_px_max_line(pxs=pxs, num_iter=100, max_d=100)
            floor.append([key, max_points_set])
    return floor


def draw_floor(image, floor):
    img = image.copy()
    for line in floor:
        x, y = line[0], line[1][3]
        img[y][x] = 0
    return img


image_names = os.listdir('data/')

for image_name in image_names:
    image = cv2.imread('data/' + image_name, cv2.IMREAD_UNCHANGED)
    floor = find_floor_2d(image)
    img = draw_floor(image=image, floor=floor)
    plt.imshow(img)
    plt.show()
