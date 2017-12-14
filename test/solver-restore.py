import numpy as np
import random
from math import cos, sin, pi, floor, ceil, tan
import cv2


image = cv2.imread('image.pgm', -1)

for i in range(250):
    for j in range(250):
        helper = np.array(
            [image[2 * j][2 * i], image[2 * j + 1][2 * i], image[2 * j + 1][2 * i + 1], image[2 * j][2 * i + 1]]).mean()
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
        if image[j][i] > 100:
            d = list()
            for psi in range(-90, 90):
                d.append(round(i * cos(psi * pi / 180.) + j * sin(psi * pi / 180.)))
            x_y.append([[psi * pi / 180. for psi in range(-90, 90)], d])

x_y = np.array(x_y)


max_d = 0
min_d = 0

for line in x_y:
    if max_d < max(line[1]):
        max_d = max(line[1])
    if min_d > min(line[1]):
        min_d = min(line[1])


plane = np.zeros((int(round(abs(min_d)+max_d))+1, 180))

for line in x_y:
    k = 0
    for psi, d in zip(line[0], line[1]):
        plane[int(round(d+abs(min_d)))][k] += 1
        k += 1


pts = list()

for i in range(300):
    res = [(int(x-abs(min_d)), (y-90)*pi/180) for x in range(int(abs(min_d)+max_d)+1) for y in range(180) if plane[x][y] == (300-i)]
    if res:
        pts.extend(res)
    if len(pts) > 100:
        break


array_pts = np.array(pts)

min_psi = 999
max_psi = -999

for pt in array_pts[:, 1]:
    if pt < min_psi:
        min_psi = pt
    if pt > max_psi:
        max_psi = pt

mean_psi = (min_psi + max_psi) / 2


####

list_max_psi = list()
list_mean_psi = list()
list_min_psi = list()

for pt in pts:
    if pt[1] > max_psi - 0.3:
        list_max_psi.append(pt[1])
    if pt[1] < min_psi + 0.3:
        list_min_psi.append(pt[1])
    if pt[1] > min_psi + 0.3 and pt[1] < max_psi - 0.3:
        list_mean_psi.append(pt[1])

min_psi = np.mean(np.array(list_min_psi))
mean_psi = np.mean(np.array(list_mean_psi))
max_psi = np.mean(np.array(list_max_psi))

####

new_pts = list()

for pt in pts:
    if pt[1] < (min_psi + mean_psi) / 2:
        q = tuple([pt[0], min_psi])
    if pt[1] < (max_psi + mean_psi) / 2 and pt[1] > (min_psi + mean_psi) / 2:
        q = tuple([pt[0], mean_psi])
    if pt[1] > (max_psi + mean_psi) / 2:
        q = tuple([pt[0], max_psi])
    new_pts.append(q)

array_new_pts = np.array(new_pts)

dct_psi1 = dict()
dct_psi2 = dict()
dct_psi3 = dict()

for line in array_new_pts:
    if line[1] == min_psi:
        is_ = False
        for key in dct_psi1.keys():
            if float(key) - 25 < line[0] and float(key) + 25 > line[0]:
                lst = dct_psi1[key]
                lst.append(line[0])
                is_ = True
        if not is_:
            dct_psi1[str(line[0])] = list()
    if line[1] == mean_psi:
        is_ = False
        for key in dct_psi2.keys():
            if float(key) - 25 < line[0] and float(key) + 25 > line[0]:
                lst = dct_psi2[key]
                lst.append(line[0])
                is_ = True
        if not is_:
            dct_psi2[str(line[0])] = list()
    if line[1] == max_psi:
        is_ = False
        for key in dct_psi3.keys():
            if float(key) - 25 < line[0] and float(key) + 25 > line[0]:
                lst = dct_psi3[key]
                lst.append(line[0])
                is_ = True
        if not is_:
            dct_psi3[str(line[0])] = list()

d_psi1 = list()
d_psi2 = list()
d_psi3 = list()

for key in dct_psi1.keys():
    lst = dct_psi1[key]
    lst.append(float(key))
    d_psi1.append(np.mean(np.array(lst)))

for key in dct_psi2.keys():
    lst = dct_psi2[key]
    lst.append(float(key))
    d_psi2.append(np.mean(np.array(lst)))

for key in dct_psi3.keys():
    lst = dct_psi3[key]
    lst.append(float(key))
    d_psi3.append(np.mean(np.array(lst)))

dct = {min_psi: d_psi1, mean_psi: d_psi2, max_psi: d_psi3}

new_new_pts = list()

for pt in new_pts:
    if pt[1] == min_psi:
        for d in dct[min_psi]:
            if d - 25 <= pt[0] <= d + 25:
                q = tuple([d, min_psi])

    if pt[1] == mean_psi:
        for d in dct[mean_psi]:
            if d - 25 <= pt[0] <= d + 25:
                q = tuple([d, mean_psi])

    if pt[1] == max_psi:
        for d in dct[max_psi]:
            if d - 25 <= pt[0] <= d + 25:
                q = tuple([d, max_psi])

    new_new_pts.append(q)

new_new_pts = np.array(new_new_pts)

uniq = list()

for s in new_new_pts:
    if uniq:
        is_ = 0
        for u in uniq:
            if s[0] == u[0] and s[1] == u[1]:
                is_ += 1
        if is_ == 0:
            uniq.append(s)

    if not uniq:
        uniq.append(s)

uniq = np.array(uniq)

per = list()

for i_line in range(uniq.shape[0]):
    for j_line in range(i_line + 1, uniq.shape[0]):
        for k_line in range(j_line + 1, uniq.shape[0]):
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
    if sum([abs(qq[0]-qq[2]), abs(qq[2]-qq[4]), abs(qq[1]-qq[3]), abs(qq[3]-qq[5])]) < 20:
        list_V.append([(qq[0]+qq[2]+qq[4])/3, (qq[1]+qq[3]+qq[5])/3])


if len(list_V) == 8:
    with open('output.txt', 'w') as f:
        for v in list_V:
            f.write(str(v[0]) + ' ' + str(v[1]))
            f.write('\n')
