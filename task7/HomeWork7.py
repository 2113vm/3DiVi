
# coding: utf-8

# In[1]:

import os

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

car_dir = 'data/task7/cars/'
moto_dir = 'data/task7/moto/'
car_image = os.listdir(car_dir)
moto_image = os.listdir(moto_dir)
print(len(car_image), len(moto_image))

winSize = (256, 192) 
blockSize = (16,16) 
blockStride = (8,8) 
cellSize = (8,8) 
nbins = 9 


# In[2]:

hogObj = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
cars = []
motos = []

for image_name in car_image:
    img = cv2.imread(car_dir + image_name)
    hogVec = hogObj.compute(img)
    cars.append(hogVec)
    
for image_name in moto_image:
    img = cv2.imread(moto_dir + image_name)
    hogVec = hogObj.compute(img)
    motos.append(hogVec)

    
cars = np.array(cars)
cars = np.hstack((cars, np.ones((cars.shape[0], 1, 1))))
motos = np.array(motos)
motos = np.hstack((motos, np.zeros((motos.shape[0], 1, 1))))


# In[3]:

knn = KNeighborsClassifier()
svm = SVC(kernel='linear')
rbf = SVC(kernel='rbf')


# In[4]:

data = np.vstack((cars, motos))
data = data.reshape((data.shape[0], data.shape[1]))
Y = data[:,-1]
X = data[:,:-1]


# In[5]:

list_test_size = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
models = [knn, svm, rbf]


# In[6]:

result = []


for model in models:
    model_res = []
    for test_size in list_test_size:
        for i in range(100):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=i)
            model.fit(X_train, Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test, pred)
            model_res.append(acc)
            print(model.__class__.__name__, i, 'score = ', acc)
    result.append(model_res)


# In[7]:

len(result[0])


# In[28]:

res = {'KNN': {'0.9': result[0][:100], 
               '0.8': result[0][100:200], 
               '0.7': result[0][200:300], 
               '0.6': result[0][300:400], 
               '0.5': result[0][400:500], 
               '0.4': result[0][500:600], 
               '0.3': result[0][600:700], 
               '0.2': result[0][700:800], 
               '0.1': result[0][800:900]},
       'SVM': {'0.9': result[1][:100], 
               '0.8': result[1][100:200], 
               '0.7': result[1][200:300], 
               '0.6': result[1][300:400], 
               '0.5': result[1][400:500], 
               '0.4': result[1][500:600], 
               '0.3': result[1][600:700], 
               '0.2': result[1][700:800], 
               '0.1': result[1][800:900]}, 
       'RBF': {'0.9': result[2][:100], 
               '0.8': result[2][100:200], 
               '0.7': result[2][200:300], 
               '0.6': result[2][300:400], 
               '0.5': result[2][400:500], 
               '0.4': result[2][500:600], 
               '0.3': result[2][600:700], 
               '0.2': result[2][700:800], 
               '0.1': result[2][800:900]}}


# In[30]:

plt.boxplot([res['KNN']['0.9'],
             res['KNN']['0.8'],
             res['KNN']['0.7'], 
             res['KNN']['0.6'], 
             res['KNN']['0.5'], 
             res['KNN']['0.4'], 
             res['KNN']['0.3'], 
             res['KNN']['0.2'], 
             res['KNN']['0.1']]);


# In[31]:

plt.boxplot([res['SVM']['0.9'],
             res['SVM']['0.8'],
             res['SVM']['0.7'], 
             res['SVM']['0.6'], 
             res['SVM']['0.5'], 
             res['SVM']['0.4'], 
             res['SVM']['0.3'], 
             res['SVM']['0.2'], 
             res['SVM']['0.1']]);


# In[32]:

plt.boxplot([res['RBF']['0.9'],
             res['RBF']['0.8'],
             res['RBF']['0.7'], 
             res['RBF']['0.6'], 
             res['RBF']['0.5'], 
             res['RBF']['0.4'], 
             res['RBF']['0.3'], 
             res['RBF']['0.2'], 
             res['RBF']['0.1']]);


# In[ ]:



