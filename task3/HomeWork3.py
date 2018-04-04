
# coding: utf-8

# In[2]:

from skimage.io import imread, imshow
from skimage.color import rgb2gray as gray
from os import listdir
from scipy import signal
import numpy as np
import time as t
import cv2
get_ipython().magic('matplotlib inline')


image_dir = 'data/task3/'
matrix = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
matrix_complex = np.array(np.complex64(matrix) + (np.complex64(matrix) * 1.j).transpose())
value = 0.2 # задает порог равный 20% от максимального значения
# Для проверки результатов из за большого количества времени поиска рекомендуется использовать 2ой способ вычисления карты


# # 1 способ#

# In[3]:

def get_lam_1(image):

    b = t.time()
    y_shape, x_shape = image.shape
    map_lambda = np.zeros((y_shape-2, x_shape-2))
    for x in range(1, x_shape-1):
        for y in range(1, y_shape-1):
            window = image[y-1:y+2,x-1:x+2]
            map_lambda[y-1,x-1] = np.min(get_eig(A(grad(window=window, matrix=matrix))))
    print('Time work for image: ' + str(t.time()-b))
        
    return map_lambda


# # 2ой способ #

# In[4]:

def get_lam_2(image):
    
    b = t.time()
    y_shape, x_shape = image.shape
    map_lambda = np.zeros((y_shape, x_shape))
    grad = signal.convolve2d(image * (1 + 1.j), matrix_complex, boundary='symm', mode='same')
    A = np.array([[np.real(grad) ** 2, np.real(grad) * np.imag(grad)],
                  [np.real(grad) * np.imag(grad), np.imag(grad)**2]])
    print('Time work of conv: ' + str(t.time() - b))
    for x in range(x_shape):
        for y in range(y_shape):
            map_lambda[y,x] = np.min(np.linalg.eig(A[:,:,y,x])[0])

    print('Time work for image: ' + str(t.time()-b))
    
    return np.int64(map_lambda > (map_lambda.max() * value)) * map_lambda


# # 3ий способ #

# In[5]:

def get_lam_3(image):

    b = t.time()
    y_shape, x_shape = image.shape
    map_lambda = np.zeros((y_shape, x_shape))
    grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    for x in range(x_shape):
        for y in range(y_shape):
            I_x = grad_x[y,x]
            I_y = grad_y[y,x]
            map_lambda[y,x] = np.min(np.linalg.eig(np.array([[I_x**2, I_x*I_y],[I_x*I_y, I_y**2]]))[0])

    print('Time work for image: ' + str(t.time()-b))
    return np.int64(map_lambda > (map_lambda.max() * value)) * map_lambda
        


# # 4ый способ #

# In[6]:

def get_lam_4(image):

    b = t.time()
    y_shape, x_shape = image.shape
    map_lambda = np.zeros((y_shape, x_shape))
    grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    grad = grad_x + 1.j * grad_y
    map_lambda = lam_f(grad)
    print('Time work for image: ' + str(t.time()-b))
    return np.int64(map_lambda > (map_lambda.max() * value)) * map_lambda
    


# # 5ый способ #

# In[7]:

def get_lam_5(image, value):
    
    b = t.time()
    y_shape, x_shape = image.shape
    grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    I_x = grad_x.reshape((y_shape*x_shape, 1))
    I_y = grad_y.reshape((y_shape*x_shape, 1))
    A = np.array([[I_x**2, I_x*I_y],[I_y*I_x, I_y**2]])
    A = np.transpose(A, (3,2,0,1))
    A = np.linalg.eigh(A)[0]
    map_lambda = (np.min(A, axis=2)).reshape((y_shape,x_shape))
    return np.int64(map_lambda > (map_lambda.max() * value)) * map_lambda
    
#     print('Time work for image: ' + str(t.time()-b))
#     return np.int64(map_lambda > (map_lambda.max() * value)) * map_lambda


# In[8]:

def grad(window, matrix):
    return (matrix * window).sum(), (matrix.transpose() * window).sum()
    
def A(I):
    I_x, I_y = I
    return np.array([[I_x**2, I_x*I_y],[I_x*I_y, I_y**2]])

def get_eig(A):
    print(np.linalg.eig(A))
    return np.linalg.eig(A)[0]

def NMS(image, window_size):
    y_shape, x_shape = image.shape
    y_shape_w, x_shape_w = int(window_size[1] / 2), int(window_size[0] / 2)
    
    
    for x in range(x_shape_w, x_shape - x_shape_w):
        for y in range(y_shape_w, y_shape - y_shape_w):
            max_ = np.max(image[y-y_shape_w:y+y_shape_w+1,x-x_shape_w:x+x_shape_w+1])
            if image[y,x] == max_:
                image[y-y_shape_w:y+y_shape_w+1,x-x_shape_w:x+x_shape_w+1] = 0
                image[y,x] = max_
            elif image[y,x] < max_:
                image[y,x] = 0
    return np.array(np.nonzero(image)).T

def nice_point(image, value, window_size):
    return NMS(get_lam_5(image, value), window_size)

def lam(I):
    I_x, I_y = np.real(I), np.imag(I)
    return np.min(np.linalg.eig(np.array([[I_x**2, I_x*I_y],[I_x*I_y, I_y**2]]))[0])

lam_f = np.vectorize(lam)


# In[17]:

images_name = listdir(image_dir)

images = []

for name in images_name:
    image = imread(image_dir + name)
    img = gray(image)
    res = nice_point(img, 0.01, (30,30))
    for px in res:
        center = [px[1], px[0]]
        cv2.circle(image, center=tuple(center), radius=3, color=(255,0,0), thickness=1)
    images.append(image)


# In[18]:

imshow(images[0])


# In[19]:

imshow(images[1])


# In[20]:

imshow(images[2])


# In[21]:

imshow(images[3])


# In[22]:

imshow(images[4])


# In[23]:

imshow(images[5])


# In[ ]:




# In[ ]:



