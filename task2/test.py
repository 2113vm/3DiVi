import numpy as np


def get_d(points, px):
    if np.equal(points[0], px).all() or np.equal(points[1], px).all():
        return 0
    else:
        A_B = (np.array(points[0]) - np.array(points[1]))[2], \
              -(np.array(points[0]) - np.array(points[1]))[1]
        return abs(A_B[0] * px[1] + A_B[1] * px[2] + (-(A_B[0] * px[1] + A_B[1] * px[2]))) / \
                  (A_B[0] ** 2 + A_B[1] ** 2) ** 0.5


points = np.array([[0, 1], [1, 2]])
px = [5,5]


def get_d_(points: np.array, px: np.array):
    if np.equal(points[0], px).all() or np.equal(points[0], px).all():
        return 0
    else:
        m = points[1] - points[0]
        n = np.array([m[1], -m[0]])
        c = -(n @ points[1])
        return abs(n @ px + c) / (n[0]**2 + n[1]**2)**0.5

print(get_d_(points, px))