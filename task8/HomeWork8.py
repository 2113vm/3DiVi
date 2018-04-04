
# coding: utf-8

# In[1]:

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[8]:

images_name = os.listdir('data/task8/database/')


# In[10]:

features_database = {}

for image_name in images_name:
    img = cv2.imread('data/task8/database/' + image_name)
    sift = cv2.xfeatures2d.SIFT_create(100)
    features_database[image_name] = sift.detect(img)


# In[11]:

features_database


# In[ ]:



