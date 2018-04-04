import numpy as np
import cv2
import matplotlib.pyplot as plt


image_dir = 'data/'

images = [cv2.cvtColor(cv2.imread(image_dir + str(i) + '.png'),
                       cv2.COLOR_BGR2GRAY) for i in range(40)]

coords = [np.array([[5, 15],
                    [15, 15],
                    [25, 15],
                    [35, 15],
                    [45, 15],
                    [45, 25],
                    [45, 35]])]


def mean_shift(image: np.array, coords: np.array, win_size: int, e: float, s: float) -> tuple:

    new_coords = []
    for coord in coords:
        y, x = coord

        def w(x, s):
            return np.exp(-1. * np.sum(x**2, axis=0)**2 / (2 * s**2))

        def step(arr, y, x, s):
            cen_y, cen_x = int(arr.shape[0] / 2), int(arr.shape[1] / 2)
            crop = np.nonzero(arr)[:2] - np.array([[cen_y], [cen_x]])
            up = np.sum(w(crop, s) * crop * arr[np.nonzero(arr)], axis=1)
            down = np.sum(w(crop, s) * arr[np.nonzero(arr)])
            return np.array([y, x]) + up / down

        while True:
            center_y, center_x = int(round(y)), int(round(x))
            new_y, new_x = step(image[center_y-win_size:center_y+win_size+1, center_x-win_size:center_x+win_size+1], y, x, s)
            if (y-new_y)**2 + (x-new_x)**2 < e**2:
                break
            y, x = new_y, new_x
        new_coords.append([new_y, new_x])
    return np.array(new_coords)


def get_rt(start_coords, end_coords):
    p, q = start_coords - start_coords.mean(axis=0), end_coords - end_coords.mean(axis=0)
    S = p.transpose().dot(np.eye(p.shape[0]).dot(q))
    U, E, V = np.linalg.svd(S)
    det = np.linalg.det(V.transpose().dot(U.transpose()))
    eye = np.eye(2)
    eye[1, 1] = det
    r = V.transpose().dot(eye).dot(U.transpose())
    t = end_coords.mean(axis=0) - r.dot(start_coords.mean(axis=0))
    return r, t


def set_match(o_coords, n_coords):
    match = []
    for coord in o_coords:
        match.append(n_coords[np.argmin(np.sum((n_coords-coord)**2, axis=1))])
    err = np.sum((np.array(match) - o_coords)**2)
    print(err)
    return err


for image in images:
    coords.append(mean_shift(image, coords[-1], 5, 0.4, 10))


for i, coord in enumerate(coords[1:]):
    old_coords = coords[0].copy()
    new_coords = coord.copy()
    err = set_match(old_coords, new_coords)

    while err > 2:
        print('===')
        r, t = get_rt(old_coords, new_coords)
        old_coords = old_coords @ r.transpose() + t
        err = set_match(old_coords, new_coords)

    for old_coord in old_coords:
        x, y = int(round(old_coord[0])), int(round(old_coord[1]))
        cv2.circle(img=images[i], center=(y, x), radius=5, color=255)

    plt.imshow(images[i])
    plt.show()
