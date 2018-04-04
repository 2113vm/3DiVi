import cv2
import numpy as np

image_dir = 'data/'

images = [cv2.cvtColor(cv2.imread(image_dir + str(i) + '.png'),
                       cv2.COLOR_BGR2GRAY) for i in range(40)]

start = np.array([[5, 15],
                  [15, 15],
                  [25, 15],
                  [35, 15],
                  [45, 15],
                  [45, 25],
                  [45, 35]])


def w(delta, s):
    """
    Функция возвращает расстояние между двумя точками
    :param delta: вектор разности между двумя векторами
    :param s: параметр отвечающий за распределение расстояний
    :return: возвращает экспоненту в степени евклидово расстояние деленное на константу
    """
    return np.exp(-1. * np.sum(delta ** 2, axis=0) ** 2 / (2 * s ** 2))


def mean_shift(image, coords):
    """
    Функция, возвращающая координаты интересующих точек на изображении
    :param image: изображении, на котором точки находятся в другой месте
    :param coords: координаты точек на предыдущем изображении
    :return: координаты на новом изображении
    """
    
    pass
