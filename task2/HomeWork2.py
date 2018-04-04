# coding: utf-8

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from math import cos, sin, exp
import time
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from timeit import default_timer as timer


# get_ipython().magic('matplotlib inline')


def get_tree_random_p(lst):
    n1, n2, n3 = 0, 0, 0

    while n1 == n2 or n1 == n3 or n2 == n3:
        n1 = random.randint(a=0, b=lst.shape[0] - 1)
        n2 = random.randint(a=0, b=lst.shape[0] - 1)
        n3 = random.randint(a=0, b=lst.shape[0] - 1)

    return lst[n1], lst[n2], lst[n3]


def get_random_px(lst):
    n1 = random.randint(a=0, b=lst.shape[0] - 1)

    return lst[n1]


def get_d(tuple_, px):
    px1, px2, px3 = tuple_

    DET = np.array([[px[0] - px1[0], px[1] - px1[1], px[2] - px1[2]],
                    [px2[0] - px1[0], px2[1] - px1[1], px2[2] - px1[2]],
                    [px3[0] - px1[0], px3[1] - px1[1], px3[2] - px1[2]]])

    norm_x = np.linalg.det(np.array([[px2[1] - px1[1], px2[2] - px1[2]],
                                     [px3[1] - px1[1], px3[2] - px1[2]]]))
    norm_y = np.linalg.det(np.array([[px2[0] - px1[2], px2[0] - px1[2]],
                                     [px3[0] - px1[2], px3[0] - px1[2]]]))
    norm_z = np.linalg.det(np.array([[px2[0] - px1[0], px2[1] - px1[1]],
                                     [px3[0] - px1[0], px3[1] - px1[1]]]))
    vector_norm = np.array([norm_x, norm_y, norm_z])

    d = abs(np.linalg.det(DET)) / (norm_x ** 2 + norm_y ** 2 + norm_z ** 2) ** 0.5

    return d


def get_d_(tuple_, px):
    px1, px2, px3 = tuple_

    DET = np.array([[px[0] - px1[0], px[1] - px1[1], px[2] - px1[2]],
                    [px2[0] - px1[0], px2[1] - px1[1], px2[2] - px1[2]],
                    [px3[0] - px1[0], px3[1] - px1[1], px3[2] - px1[2]]])

    norm_x = np.linalg.det(np.array([[px2[1] - px1[1], px2[2] - px1[2]],
                                     [px3[1] - px1[1], px3[2] - px1[2]]]))
    norm_y = np.linalg.det(np.array([[px2[0] - px1[2], px2[0] - px1[2]],
                                     [px3[0] - px1[2], px3[0] - px1[2]]]))
    norm_z = np.linalg.det(np.array([[px2[0] - px1[0], px2[1] - px1[1]],
                                     [px3[0] - px1[0], px3[1] - px1[1]]]))
    vector_norm = np.array([norm_x, norm_y, norm_z])

    d = np.linalg.det(DET) / (norm_x ** 2 + norm_y ** 2 + norm_z ** 2) ** 0.5

    return d


def get_subset(lst, k_max):
    set_px = list()
    k = 0

    while k < k_max:
        px = get_random_px(lst)
        set_px.append(px)
        k += 1

    return np.array(set_px)


def camera_transform(image):
    """
    Return transform image for camera
    :param image:
    :return:
    """
    img = np.zeros((image.shape[0], image.shape[1], 3))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            img[y][x] = (x - 320) / 575.5 * image[y, x], (240 - y) / 575.5 * image[y, x], image[
                y, x]
    return img


def find_floor3d(image):

    img = camera_transform(image=image)

    lst = list()

    for x in range(img.shape[1]):
        z = np.array([z for z in reversed(image[:, x])])
        lst_ = []
        max_z = 0
        for i in range(image.shape[0]):
            if img[-i, x, 1] < 0:
                if max_z < z[i]:
                    max_z = z[i]
                    lst_.append([img[-i, x, 0], img[-i, x, 1], img[-i, x, 2], int(-i), int(x),
                                 int(image[-i, x])])

        if lst_:
            lst.extend(lst_)

    lst = np.array(lst)

    sub_set_train = get_subset(lst, 100)
    sub_set_test = get_subset(lst, 100)

    l = 0
    l_max = 100
    number_max = 0
    argmax = 0

    res = []

    while l < l_max:
        tree_px = get_tree_random_p(sub_set_train)
        number = 0
        for px in sub_set_test:
            px = get_random_px(lst=sub_set_test)
            if get_d(tuple_=tree_px, px=px) < 300:
                number += 1
        if number_max < number and number != 100:
            number_max = number
            argmax = l
        res.append([number, tree_px[0], tree_px[1], tree_px[2]])
        l += 1

    return res[argmax][1:], lst


def get_mask(image, plane, lst):
    img_ = np.zeros((image.shape[0], image.shape[1]))

    for l in lst:
        if get_d(plane, l) < 300:
            img_[int(l[3])][int(l[4])] = int(l[5])

    return img_


# In[3]:

img_names = os.listdir('data/task2/')

result = []

for image_name in img_names:
    image = cv2.imread('data/task2/' + image_name, cv2.IMREAD_UNCHANGED)
    print('Image ' + image_name)
    millisStart = timer()
    plane, lst = find_floor3d(image)
    millisEnd = timer()
    result.append([image, plane, lst])
    elapsedTime = round(1000 * (millisEnd - millisStart))
    print('Время работы: ', elapsedTime)  # время в миллисекундах

# In[4]:

image, plane, lst = result[0]

mask = get_mask(image, plane, lst)

plt.imshow(np.hstack((image, mask)))

# In[5]:

image, plane, lst = result[1]

mask = get_mask(image, plane, lst)

plt.imshow(np.hstack((image, mask)))

# In[6]:

image, plane, lst = result[2]

mask = get_mask(image, plane, lst)

plt.imshow(np.hstack((image, mask)))

# In[7]:

image, plane, lst = result[3]

mask = get_mask(image, plane, lst)

plt.imshow(np.hstack((image, mask)))

# In[8]:

image, plane, lst = result[4]

mask = get_mask(image, plane, lst)

plt.imshow(np.hstack((image, mask)))

# In[9]:

image, plane, lst = result[5]

mask = get_mask(image, plane, lst)

plt.imshow(np.hstack((image, mask)))

# In[10]:

image, plane, lst = result[6]

mask = get_mask(image, plane, lst)

plt.imshow(np.hstack((image, mask)))

# In[11]:

# fig = plt.figure(figsize=(10,70))

# for i in range(0,len(result)):

#     ax = fig.add_subplot(7, 1, i+1, projection='3d')
#     x = np.outer(np.linspace(0, result[i][0].shape[0], result[i][0].shape[0]), np.ones(result[i][0].shape[0]))
#     y = x.copy().T
#     z1 = result[i][0][:result[i][0].shape[0],:result[i][0].shape[0]]
#     z2 = get_mask(result[i][0], result[i][2], result[i][3])[:result[i][0].shape[0],:result[i][0].shape[0]]
#     surf1 = ax.scatter(x, y, z1, marker='.', s=1)
#     surf2 = ax.scatter(x, y, z2, marker='.', s=1)


# In[ ]:
