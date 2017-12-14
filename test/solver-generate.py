import numpy as np
import random
from math import cos, sin, pi, floor, ceil, tan
import cv2


def ipart(x):
    if x > 0:
        i = floor(x)
    else:
        i = ceil(x)
    return i


def _round(x):
    return ipart(x + 0.5)


# fractional part
def fpart(x):
    return x - ipart(x)


def rfpart(x):
    return 1 - fpart(x)


def sqrt(x):
    return x ** 0.5


def main(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    x = np.zeros(floor(2 * sqrt(dx ** 2 + dy ** 2)))
    y = np.zeros(len(x))
    c = np.zeros(len(x))

    swapped = False

    if abs(dx) < abs(dy):
        y1, x1 = x1, y1
        y2, x2 = x2, y2
        dy, dx = dx, dy
        swapped = True

    if x2 < x1:
        x2, x1 = x1, x2
        y2, y1 = y1, y2

    gradient = dy / dx

    ##########################################################

    xend = _round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = rfpart(x1 + 0.5)
    xpxl1 = xend  # % this will be used in the main loop
    ypxl1 = ipart(yend)
    x[0] = xpxl1
    y[0] = ypxl1
    c[0] = rfpart(yend) * xgap
    x[1] = xpxl1
    y[1] = ypxl1 + 1
    c[1] = fpart(yend) * xgap
    intery = yend + gradient  # % first y-intersection for the main loop

    xend = _round(x2)
    yend = y2 + gradient * (xend - x2)
    xgap = fpart(x2 + 0.5)
    xpxl2 = xend  # this will be used in the main loop
    ypxl2 = ipart(yend)
    x[2] = xpxl2
    y[2] = ypxl2
    c[2] = rfpart(yend) * xgap
    x[3] = xpxl2
    y[3] = ypxl2 + 1
    c[3] = fpart(yend) * xgap

    k = 4

    for i in range(xpxl1 + 1, xpxl2 - 1):
        x[k] = i
        y[k] = ipart(intery)
        c[k] = rfpart(intery)
        k = k + 1
        x[k] = i
        y[k] = ipart(intery) + 1
        c[k] = fpart(intery)
        intery = intery + gradient
        k = k + 1

    # truncate the vectors to proper sizes
    x = x[1:k - 1]
    y = y[1:k - 1]
    c = c[1:k - 1]

    if swapped:
        y, x = x, y

    return x, y, c


def distance_l(x1, y1, x2, y2, x3, y3, x4, y4):
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = x1 * (y1 - y2) + y1 * (x2 - x1)
    A2 = y4 - y3
    B2 = x4 - x3
    C2 = x3 * (y3 - y4) + y3 * (x4 - x3)
    return abs(C2 - C1) * 1. / (A1 ** 2 + B1 ** 2) ** 0.5


def P(p_value):
    porog = np.random.uniform(0, 100)
    return porog < p_value * 100


def noisy(image, p_value):
    def P(p_value):
        porog = np.random.uniform(0, 100)
        return porog < p_value * 100

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if P(p_value):
                image[y][x] = np.random.uniform(0, 255)

    return image


def create_cube(a):
    return np.array([[0, 0, 0],  # 0
                     [0, 0, a],  # 1
                     [0, a, a],  # 2
                     [0, a, 0],  # 3
                     [a, a, 0],  # 4
                     [a, 0, 0],  # 5
                     [a, 0, a],  # 6
                     [a, a, a]])  # 7


def is_good(CUBE):
    def distance_V(CUBE):
        for i in range(len(CUBE)):
            for g in range(i + 1, len(CUBE)):
                if ((CUBE[i] - CUBE[g]) ** 2).sum() < 10000:
                    return False
        return True

    up = np.int64(CUBE <= 500).sum()
    down = np.int64(CUBE >= 0).sum()
    is_dist_v = distance_V(CUBE)
    return up == down == 16 and is_dist_v


while (True):

    image = np.zeros((500, 500))

    a = random.uniform(a=150, b=300)
    cube = create_cube(a)

    x, y = random.uniform(a=0, b=500), random.uniform(a=0, b=500)

    alpha = random.uniform(a=0, b=2 * pi)
    beta = random.uniform(a=0, b=2 * pi)
    gamma = random.uniform(a=0, b=2 * pi)

    X_rotate = np.array([[1, 0, 0],
                         [0, cos(alpha), -sin(alpha)],
                         [0, sin(alpha), cos(alpha)]])

    Y_rotate = np.array([[cos(beta), 0, -sin(beta)],
                         [0, 1, 0],
                         [sin(beta), 0, cos(beta)]])

    Z_rotate = np.array([[cos(gamma), -sin(gamma), 0],
                         [sin(gamma), cos(gamma), 0],
                         [0, 0, 1]])

    ROTATE = X_rotate.dot(Y_rotate).dot(Z_rotate)

    cube_rotate = cube.dot(ROTATE)

    cube_x_y = cube_rotate[:, :2] + np.array([x, y])

    CUBE = np.int64(cube_x_y)

    list_l = [int(distance_l(CUBE[0][0], CUBE[0][1], CUBE[1][0], CUBE[1][1], CUBE[4][0], CUBE[4][1], CUBE[7][0],
                             CUBE[7][1]) < 50),
              int(distance_l(CUBE[0][0], CUBE[0][1], CUBE[1][0], CUBE[1][1], CUBE[3][0], CUBE[3][1], CUBE[2][0],
                             CUBE[2][1]) < 50),
              int(distance_l(CUBE[0][0], CUBE[0][1], CUBE[1][0], CUBE[1][1], CUBE[5][0], CUBE[5][1], CUBE[6][0],
                             CUBE[6][1]) < 50),
              int(distance_l(CUBE[4][0], CUBE[4][1], CUBE[7][0], CUBE[7][1], CUBE[3][0], CUBE[3][1], CUBE[2][0],
                             CUBE[2][1]) < 50),
              int(distance_l(CUBE[4][0], CUBE[4][1], CUBE[7][0], CUBE[7][1], CUBE[5][0], CUBE[5][1], CUBE[6][0],
                             CUBE[6][1]) < 50),
              int(distance_l(CUBE[3][0], CUBE[3][1], CUBE[2][0], CUBE[2][1], CUBE[5][0], CUBE[5][1], CUBE[6][0],
                             CUBE[6][1]) < 50),
              int(distance_l(CUBE[1][0], CUBE[1][1], CUBE[2][0], CUBE[2][1], CUBE[6][0], CUBE[6][1], CUBE[7][0],
                             CUBE[7][1]) < 50),
              int(distance_l(CUBE[1][0], CUBE[1][1], CUBE[2][0], CUBE[2][1], CUBE[0][0], CUBE[0][1], CUBE[3][0],
                             CUBE[3][1]) < 50),
              int(distance_l(CUBE[1][0], CUBE[1][1], CUBE[2][0], CUBE[2][1], CUBE[5][0], CUBE[5][1], CUBE[4][0],
                             CUBE[4][1]) < 50),
              int(distance_l(CUBE[6][0], CUBE[6][1], CUBE[7][0], CUBE[7][1], CUBE[0][0], CUBE[0][1], CUBE[3][0],
                             CUBE[3][1]) < 50),
              int(distance_l(CUBE[6][0], CUBE[6][1], CUBE[7][0], CUBE[7][1], CUBE[5][0], CUBE[5][1], CUBE[4][0],
                             CUBE[4][1]) < 50),
              int(distance_l(CUBE[0][0], CUBE[0][1], CUBE[3][0], CUBE[3][1], CUBE[5][0], CUBE[5][1], CUBE[4][0],
                             CUBE[4][1]) < 50),
              int(distance_l(CUBE[6][0], CUBE[6][1], CUBE[1][0], CUBE[1][1], CUBE[2][0], CUBE[2][1], CUBE[7][0],
                             CUBE[7][1]) < 50),
              int(distance_l(CUBE[6][0], CUBE[6][1], CUBE[1][0], CUBE[1][1], CUBE[3][0], CUBE[3][1], CUBE[4][0],
                             CUBE[4][1]) < 50),
              int(distance_l(CUBE[0][0], CUBE[0][1], CUBE[5][0], CUBE[5][1], CUBE[1][0], CUBE[1][1], CUBE[6][0],
                             CUBE[6][1]) < 50),
              int(distance_l(CUBE[3][0], CUBE[3][1], CUBE[4][0], CUBE[4][1], CUBE[2][0], CUBE[2][1], CUBE[7][0],
                             CUBE[7][1]) < 50),
              int(distance_l(CUBE[2][0], CUBE[2][1], CUBE[7][0], CUBE[7][1], CUBE[0][0], CUBE[0][1], CUBE[5][0],
                             CUBE[5][1]) < 50),
              int(distance_l(CUBE[3][0], CUBE[3][1], CUBE[4][0], CUBE[4][1], CUBE[0][0], CUBE[0][1], CUBE[5][0],
                             CUBE[5][1]) < 50)]

    if sum(list_l) == 0 and is_good(CUBE):
        break


x1, y1, c1 = main(CUBE[0][0], CUBE[0][1], CUBE[1][0], CUBE[1][1])
x2, y2, c2 = main(CUBE[1][0], CUBE[1][1], CUBE[2][0], CUBE[2][1])
x3, y3, c3 = main(CUBE[2][0], CUBE[2][1], CUBE[3][0], CUBE[3][1])
x4, y4, c4 = main(CUBE[3][0], CUBE[3][1], CUBE[4][0], CUBE[4][1])
x5, y5, c5 = main(CUBE[4][0], CUBE[4][1], CUBE[5][0], CUBE[5][1])
x6, y6, c6 = main(CUBE[5][0], CUBE[5][1], CUBE[6][0], CUBE[6][1])
x7, y7, c7 = main(CUBE[6][0], CUBE[6][1], CUBE[7][0], CUBE[7][1])
x8, y8, c8 = main(CUBE[6][0], CUBE[6][1], CUBE[1][0], CUBE[1][1])
x9, y9, c9 = main(CUBE[0][0], CUBE[0][1], CUBE[3][0], CUBE[3][1])
x10, y10, c10 = main(CUBE[0][0], CUBE[0][1], CUBE[5][0], CUBE[5][1])
x11, y11, c11 = main(CUBE[2][0], CUBE[2][1], CUBE[7][0], CUBE[7][1])
x12, y12, c12 = main(CUBE[4][0], CUBE[4][1], CUBE[7][0], CUBE[7][1])

for x1_, y1_, c1_ in zip(x1, y1, c1):
    image[int(y1_)][int(x1_)] = c1_ * 255

for x2_, y2_, c2_ in zip(x2, y2, c2):
    image[int(y2_)][int(x2_)] = c2_ * 255

for x3_, y3_, c3_ in zip(x3, y3, c3):
    image[int(y3_)][int(x3_)] = c3_ * 255

for x4_, y4_, c4_ in zip(x4, y4, c4):
    image[int(y4_)][int(x4_)] = c4_ * 255

for x5_, y5_, c5_ in zip(x5, y5, c5):
    image[int(y5_)][int(x5_)] = c5_ * 255

for x6_, y6_, c6_ in zip(x6, y6, c6):
    image[int(y6_)][int(x6_)] = c6_ * 255

for x7_, y7_, c7_ in zip(x7, y7, c7):
    image[int(y7_)][int(x7_)] = c7_ * 255

for x8_, y8_, c8_ in zip(x8, y8, c8):
    image[int(y8_)][int(x8_)] = c8_ * 255

for x9_, y9_, c9_ in zip(x9, y9, c9):
    image[int(y9_)][int(x9_)] = c9_ * 255

for x10_, y10_, c10_ in zip(x10, y10, c10):
    image[int(y10_)][int(x10_)] = c10_ * 255

for x11_, y11_, c11_ in zip(x11, y11, c11):
    image[int(y11_)][int(x11_)] = c11_ * 255

for x12_, y12_, c12_ in zip(x12, y12, c12):
    image[int(y12_)][int(x12_)] = c12_ * 255

image = noisy(image, 0.1)


cv2.imwrite('image.pgm', image)
