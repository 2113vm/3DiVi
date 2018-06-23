import matplotlib.pyplot as plt
import cv2

from task9.task9 import DSU

data_path = 'data/'
filename = '2.jpg'

image = cv2.cvtColor(cv2.imread(data_path + filename), cv2.COLOR_BGR2GRAY)
dsu = DSU(image=image)
dct = dsu.task9(k=80)
img = dsu.draw()
plt.imshow(img)
plt.show()
plt.imshow(image)
plt.show()
