
# coding: utf-8

# In[1]:

import os
import matplotlib.pyplot as plt

import cv2
from skimage.io import imread, imshow
import numpy as np

get_ipython().magic('matplotlib inline')


# In[2]:

train_dir = 'data/task6/train/'
test_dir = 'data/task6/test/'


# In[3]:

train_image = cv2.cvtColor(cv2.imread(train_dir + 'train.jpg'), cv2.COLOR_BGR2HSV)[:,:,0]
train_mask = cv2.imread(train_dir + 'train_mask.png')


# In[4]:

h = cv2.calcHist(images=[train_image], 
                 channels=[0], 
                 mask=train_mask[:,:,0], 
                 histSize=[360], 
                 ranges=[0, 360])


# In[ ]:

h = cv2.normalize(h, h.shape)


# In[ ]:

test_images = os.listdir(test_dir)

tests = []

for test_name in test_images:
    test_img = cv2.imread(test_dir + test_name)
    img = test_img.copy()
    result = cv2.ximgproc.createSuperpixelSLIC(test_img)
    result.iterate()
    labels = result.getLabels()
    for label in range(result.getNumberOfSuperpixels()):
        c_bool = (labels == label).astype(np.int64)
        c_bool_3 = np.tile(c_bool, 3).reshape((test_img.shape))
        test_img_ = c_bool_3 * test_img
        test_img[labels == label] = test_img_.sum() / np.sum(c_bool)
    print(3)
    test_h = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)[:,:,0]
    test_h = cv2.calcBackProject(images=[test_h], 
                                 channels=[0], 
                                 hist=h, 
                                 ranges=[0, 360], 
                                 scale=3)
    img[test_h == 0] = 0
    tests.append(img)
    print('123')


# In[ ]:

# test_h = cv2.calcBackProject(images=[test_h], 
#                              channels=[0], 
#                              hist=h, 
#                              ranges=[0, 360], 
#                              scale=3)


# In[ ]:

result = cv2.ximgproc.createSuperpixelSLIC(test)
result.iterate()
labels = result.getLabels()


# In[ ]:

for label in range(result.getNumberOfSuperpixels()):
    for channel in range(3):
        test[:,:,channel][labels == label] = np.mean((labels == label).astype(np.int64) * test[:,:,channel]) * test.shape[0]*test.shape[1] / np.sum((labels == label).astype(np.int64))


# In[ ]:

plt.imshow(test)


# In[ ]:

test_h = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)[:,:,0]


# In[ ]:

plt.imshow(test_h)


# In[ ]:

test_h = cv2.calcBackProject(images=[test_h], 
                             channels=[0], 
                             hist=h, 
                             ranges=[0, 360], 
                             scale=3)


# In[ ]:

plt.imshow(test_h)


# In[ ]:

test[test_h == 0] = 0


# In[ ]:

plt.imshow(test)


# In[ ]:

np.mean((labels == label).astype(np.int64) * test[:,:,channel]) * test.shape[0]*test.shape[1] / np.sum((labels == label).astype(np.int64))


# In[ ]:

test_img[:,:,channel][labels == label] = np.mean((labels == label).astype(np.int64) * test_img[:,:,channel]) * test_img.shape[0]*test_img.shape[1] / np.sum((labels == label).astype(np.int64))


# In[ ]:

test_img = cv2.imread(test_dir + test_images[2])


# In[ ]:

plt.imshow(test_img)


# In[ ]:

result = cv2.ximgproc.createSuperpixelSLIC(test_img)
result.iterate()
labels = result.getLabels()


# In[ ]:

test_img[labels == 1] = np.mean((labels == 1).astype(np.int64)*test_img)


# In[ ]:

c = (labels == 0).astype(np.int64)


# In[ ]:

c = (labels == 0).astype(np.int64)
c = np.tile(c, 3).reshape((test_img.shape))
c * test_img


# In[ ]:

c * test_img


# In[ ]:



