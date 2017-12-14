import sys


if sys.argv[1] == 'solver-generate':

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

    image = noisy(image, float(sys.argv[2]))

    cv2.imwrite('image.pgm', image)


if sys.argv[1] == '-restore':

    import numpy as np
    import random
    from math import cos, sin, pi, floor, ceil, tan
    import cv2

    image = cv2.imread(sys.argv[2], -1)

    for i in range(250):
        for j in range(250):
            helper = np.array([image[2 * j][2 * i], image[2 * j + 1][2 * i], image[2 * j + 1][2 * i + 1],
                               image[2 * j][2 * i + 1]]).mean()
            if helper > 100 and helper < 200:
                image[2 * j][2 * i] = helper
                image[2 * j + 1][2 * i] = helper
                image[2 * j + 1][2 * i + 1] = helper
                image[2 * j][2 * i + 1] = helper

            if helper >= 200:
                image[2 * j][2 * i] = 255
                image[2 * j + 1][2 * i] = 255
                image[2 * j + 1][2 * i + 1] = 255
                image[2 * j][2 * i + 1] = 255

            if helper <= 100:
                image[2 * j][2 * i] = 0
                image[2 * j + 1][2 * i] = 0
                image[2 * j + 1][2 * i + 1] = 0
                image[2 * j][2 * i + 1] = 0

    for i in range(495):
        for j in range(495):
            s = 0
            for k in range(5):
                for l in range(5):
                    s += image[i + k][j + l]
            if s < 400:
                for k in range(2):
                    for l in range(2):
                        image[i + k + 1][j + l + 1] = 0

    x_y = list()

    for i in range(500):
        for j in range(500):
            if image[j][i] > 120:
                d = list()
                for psi in range(-90, 85):
                    d.append(round(i * cos(psi * pi / 180.) + j * sin(psi * pi / 180.)))
                x_y.append([[psi * pi / 180. for psi in range(-90, 85)], d])

    x_y = np.array(x_y)

    max_d = 0
    min_d = 0

    for line in x_y:
        if max_d < max(line[1]):
            max_d = max(line[1])
        if min_d > min(line[1]):
            min_d = min(line[1])

    plane = np.zeros((int(round(abs(min_d) + max_d)) + 1, 175))

    for line in x_y:
        k = 0
        for psi, d in zip(line[0], line[1]):
            plane[int(round(d + abs(min_d)))][k] += 10
            k += 1

    plane_max_psi = plane.max(axis=0)
    list_V_psi = list()

    for j in range(3):
        max_ = max(plane_max_psi)
        for i in range(175):
            if plane_max_psi[i] == max_:
                q_, p_ = i - 20, i + 20
                if q_ < 0:
                    q_ = 0
                if p_ > 175:
                    p_ = 175
                plane_max_psi[q_:p_] = 0
                list_V_psi.append(i)

    dct = dict()
    dct[list_V_psi[0]] = []
    dct[list_V_psi[1]] = []
    dct[list_V_psi[2]] = []
    plane_max_d = plane.copy()
    list_V_d = list()

    print(list_V_psi)

    for key in list_V_psi[:3]:
        for j in range(4):
            max_ = max(plane_max_d[:, int(key)])
            for i in range((int(round(abs(min_d) + max_d)) + 1)):
                if plane_max_d[i, int(key)] == max_:
                    q_, p_ = i - 50, i + 50
                    if q_ < min_d:
                        q_ = 0
                    if p_ > int(round(abs(min_d) + max_d)) + 1:
                        p_ = int(round(abs(min_d) + max_d)) + 1
                    plane_max_d[q_:p_, key] = 0
                    dct[key].append(i - abs(min_d))


    dct_ = dict()

    for key in dct.keys():
        dct_[(float(key) - 90) * pi / 180] = dct[key]


    list_keys = list(dct_.keys())

    uniq = list()

    for key in dct_.keys():
        for d in dct_[key]:
            uniq.append([d, float(key)])

    uniq = np.array(uniq)

    per = list()

    for i_line in range(uniq.shape[0]):
        for j_line in range(i_line + 1, uniq.shape[0]):
            for k_line in range(j_line + 1, uniq.shape[0]):
                # print(uniq[i_line], uniq[j_line], uniq[k_line])
                d1, psi1, d2, psi2, d3, psi3 = uniq[i_line][0], uniq[i_line][1], uniq[j_line][0], uniq[j_line][1], \
                                               uniq[k_line][0], uniq[k_line][1]

                if psi1 != psi2 and psi2 != psi3 and psi1 != psi3:
                    x1 = (d1 * sin(psi2) - d2 * sin(psi1)) / sin(psi2 - psi1)
                    y1 = d1 / sin(psi1) - x1 / tan(psi1)
                    x2 = (d2 * sin(psi3) - d3 * sin(psi2)) / sin(psi3 - psi2)
                    y2 = d2 / sin(psi2) - x2 / tan(psi2)
                    x3 = (d1 * sin(psi3) - d3 * sin(psi1)) / sin(psi3 - psi1)
                    y3 = d3 / sin(psi3) - x3 / tan(psi3)
                    per.append([x1, y1, x2, y2, x3, y3])

    list_V = list()

    for qq in per:
        s_ = sum([abs(qq[0] - qq[2]), abs(qq[2] - qq[4]), abs(qq[1] - qq[3]), abs(qq[3] - qq[5])])
        if s_ < 50:
            list_V.append([(qq[0] + qq[2] + qq[4]) / 3, (qq[1] + qq[3] + qq[5]) / 3])

    list_uniq_V = list()

    for v_1 in list_V:
        if list_uniq_V:
            is_ = 0
            for v_2 in list_uniq_V:
                if np.sum(abs(v_2[0] - v_1[0]) + abs(v_2[1] - v_1[1])) < 50:
                    is_ += 1
            if is_ == 0:
                list_uniq_V.append(v_1)
        if not list_uniq_V:
            list_uniq_V.append(v_1)

    if len(list_uniq_V) == 8:
        with open('output.txt', 'w') as f:
            for v in list_uniq_V:
                f.write(str(v[0]) + ' ' + str(v[1]))
                f.write('\n')

    if not list_uniq_V or len(list_uniq_V) < 8:
        list_V = [[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]
        with open('output.txt', 'w') as f:
            for v in list_V:
                f.write(str(v[0]) + ' ' + str(v[0]))
                f.write('\n')

    print(list_uniq_V)
