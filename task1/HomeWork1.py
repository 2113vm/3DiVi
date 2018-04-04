
# coding: utf-8

# In[1]:

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from math import cos, sin, exp
import time

get_ipython().magic('matplotlib inline')


# In[2]:

image_names = os.listdir('data/task1/')
data = []
for img in image_names:
    print(img)
    image = cv2.imread('data/task1/' + img)
    image = image[:, :, 0]
    data.append(np.array(np.nonzero(image)).transpose().astype(np.uint16))



# In[3]:

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
k=0

for cols in axes:
    for row in cols:
        row.scatter(data[k][:,1], data[k][:,0], marker='.', s=1)
        k+=1


# # Решение 1 #

# In[4]:

E = 0.001
s = 10

def get_curve(data, E=0.001, s=200): 
    
    mean, px = cv2.PCACompute(data=data, mean=np.array([data.mean(axis=0)]))
    new_data = cv2.PCAProject(data, mean, px)
    new_data = np.array(sorted(new_data, key=lambda x: x[0]))

    def dist(t1, t2, sigma):
        return exp(-(t2-t1)**2/(2*sigma))
    
    left_index = 0
    right_index = 0
    all_px = new_data.shape[0]
    curve = []
    is_ = 0
    min_, max_ = int(min(new_data[:,0]))-1, int(max(new_data[:,0]))+1

    for index in range(min_, max_, 10):
        is_ += 1
        while (right_index < all_px) and (dist(index, new_data[right_index][0], s) > E):
            right_index += 1
        while (left_index < is_) and (dist(index, new_data[left_index][0], s) < E):
            left_index += 1

        up = np.sum(np.array([x*dist(x[0], index, s) for x in new_data[left_index:right_index]]), axis=0)
        down = np.sum(np.array([dist(x[0], index, s) for x in new_data[left_index:right_index]]), axis=0)
        res = up/down
        curve.append(res)
        index += 1
        
    return cv2.PCABackProject(np.array(curve), mean, px)


curve1 = []
for dt in data:
    curve1.append(get_curve(dt))


# In[5]:

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
k=0

for cols in axes:
    for row in cols:
        row.scatter(data[k][:,1], data[k][:,0], marker='.', s=1)
        row.plot(curve1[k][:,1], curve1[k][:,0], 'r-')
        k+=1


# In[6]:

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
k=0

for cols in axes:
    for row in cols:
        row.plot(curve1[k][:,1], curve1[k][:,0])
        k+=1


# # Решение 2 #

# In[7]:

# оно долгое, около 20-30 сек на кривую. Решение получается менее гладкое
def get_curve(data, E=0.001, s=500): 
    
    mean, px = cv2.PCACompute(data=data, mean=np.array([data.mean(axis=0)]))
    new_data = cv2.PCAProject(data, mean, px)
    new_data = np.array(sorted(new_data, key=lambda x: x[0]))
    
    def dist(t1, t2, sigma):
        return exp(-(t2[0]-t1[0])**2/(2*sigma))

    left_index = 0
    right_index = 0
    index = 0
    all_px = new_data.shape[0]
    curve = []

    while index < all_px:
        is_ = False
        while (right_index < all_px) and (dist(new_data[index], new_data[right_index], s) > E):
            right_index += 1
        while (left_index < index) and (dist(new_data[index], new_data[left_index], s) < E):
            left_index += 1

        up = np.sum(np.array([x*dist(x, new_data[index], s) for x in new_data[left_index:right_index]]), axis=0)
        down = np.sum(np.array([dist(x, new_data[index], s) for x in new_data[left_index:right_index]]), axis=0)
        res = up/down
        curve.append(res)
        index += 1
        
    return cv2.PCABackProject(np.array(curve), mean, px)


curve2 = []
for dt in data:
    curve2.append(get_curve(dt))


# In[8]:

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
k=0

for cols in axes:
    for row in cols:
        row.scatter(data[k][:,1], data[k][:,0], marker='.', s=1)
        row.plot(curve2[k][:,1], curve2[k][:,0], 'r--')
        k+=1


# In[9]:

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
k=0

for cols in axes:
    for row in cols:
        row.plot(curve2[k][:,1], curve2[k][:,0])
        k+=1


# In[ ]:



