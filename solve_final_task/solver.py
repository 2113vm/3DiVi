import os

import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


class Decstra:
    def __init__(self, list_p, mean, shape):
        self.distance = {}
        self.set_p = {tuple(x) for x in list_p}
        self.mean = tuple(mean)
        self.helper = set()
        self.helper.add(self.mean)
        self.neighbor = {}
        self.mask = np.zeros(shape)
        self.fifo = [self.mean]
        for n in self.set_p:
            self.neighbor[n] = self.get_neighbors(n)
            self.distance[n] = -1
        self.distance[self.mean] = 0

    def set_distances(self, p_neighbor):
        if p_neighbor not in self.helper:
            self.helper.add(p_neighbor)
            neighbors = self.neighbor[p_neighbor]
            dists = [self.distance[n] for n in neighbors if self.distance[n] != -1]
            self.distance[p_neighbor] = min(dists) + 1
            return True
        else:
            return False

    def decstra(self, neighbors):
        is_set_distance = [self.set_distances(n) for n in neighbors]
        neighbors_ = list(filter(lambda x: is_set_distance[neighbors.index(x)], neighbors))
        self.fifo.extend(neighbors_)

    def s(self):
        while self.fifo:
            self.decstra(self.neighbor[self.fifo.pop(0)])

    def get_neighbors(self, p):
        """
        Возвращает соседей точки p
        :param p:
        :return:
        """
        y, x = p
        tuple_list = [(y, x - 1), (y, x + 1), (y + 1, x), (y - 1, x)]
        return list(filter(lambda tuple_: tuple_ in self.set_p, tuple_list))


class Solver:
    def __init__(self):
        pass

    def transform(self, coords, h):
        """
        Transform coords (y, x, z) to (y_real, x_real, z_real)
        :param coords:
        :return: real coords
        """
        return [(coords[1] - 320) / 575.5 * h, (240 - coords[0]) / 575.5 * h, h]

    def process(self, data):

        depth, mask, rgb = data

        human = np.nonzero(mask)
        human = np.array(list(human)).transpose()
        mean = np.int64(human.mean(axis=0)).tolist()
        body = human[human[:, 0] < mean[0]]
        mean_ = np.int64(body.mean(axis=0)).tolist()

        d = Decstra(body.tolist(), mean_, mask.shape)
        d.s()
        dct = d.distance

        items = sorted(dct.items(), key=lambda x: x[1], reverse=True)[:2000]
        hands = list(filter(lambda x:
                            x[1] > 160 and not (mean_[1] - 50 < x[0][1] < mean_[1] + 50
                                                and mean_[0] - 50 < x[0][0] < mean_[0] + 50),
                            items))

        hands = np.array(np.array(hands)[:, 0].tolist())
        mean_hands = hands.mean(axis=0)
        left = np.int64(hands[hands[:, 1] > mean_hands[1]].mean(axis=0)).tolist()
        right = np.int64(hands[hands[:, 1] < mean_hands[1]].mean(axis=0)).tolist()

        # (y, x) index into mask of left, right hands and head

        head = [100,
                np.int64(mean[1] + (100 - mean[0])*(mean_[1] - mean[1])/(mean_[0] - mean[0]))]

        head = list(filter(lambda x:
                           head[0] - 15 < x[0] < head[0] + 15 and
                           head[1] - 25 < x[1] < head[1] + 25, body))
        head = np.int64(np.array(head).mean(axis=0))

        h_left = depth[left[0], left[1]]
        h_right = depth[right[0], right[1]]
        h_head = depth[head[0], head[1]]

        head_ = self.transform(head, h_head)
        head_[2] = head_[2] + 100
        wristL = self.transform(left, h_left)
        wristR = self.transform(right, h_right)

        result = np.array([head_, wristL, wristR], dtype=np.float32)

        test = np.zeros(mask.shape)
        for h in hands.tolist():
            test[h[0], h[1]] = 255

        test1 = np.zeros(mask.shape)
        for b in body:
            test1[b[0], b[1]] = 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask, '5', tuple(left[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(mask, '3', tuple(head[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(mask, '4', tuple(right[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(mask, '1', tuple(mean[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(mask, '2', tuple(mean_[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)

        cv2.circle(mask, tuple(left[::-1]), 2, 125, 3)
        cv2.circle(mask, tuple(head[::-1]), 2, 125, 3)
        cv2.circle(mask, tuple(right[::-1]), 2, 125, 3)
        cv2.circle(mask, tuple(mean[::-1]), 2, 125, 3)
        cv2.circle(mask, tuple(mean_[::-1]), 2, 125, 3)

        cv2.putText(test, '5', tuple(left[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(test, '3', tuple(head[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(test, '4', tuple(right[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(test, '1', tuple(mean[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(test, '2', tuple(mean_[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)

        cv2.circle(test, tuple(left[::-1]), 2, 125, 3)
        cv2.circle(test, tuple(head[::-1]), 2, 125, 3)
        cv2.circle(test, tuple(right[::-1]), 2, 125, 3)
        cv2.circle(test, tuple(mean[::-1]), 2, 125, 3)
        cv2.circle(test, tuple(mean_[::-1]), 2, 125, 3)

        cv2.putText(test1, '5', tuple(left[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(test1, '3', tuple(head[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(test1, '4', tuple(right[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(test1, '1', tuple(mean[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)
        cv2.putText(test1, '2', tuple(mean_[::-1]), font, 0.7, (125,), 2, cv2.LINE_AA)

        cv2.circle(test1, tuple(left[::-1]), 2, 125, 3)
        cv2.circle(test1, tuple(head[::-1]), 2, 125, 3)
        cv2.circle(test1, tuple(right[::-1]), 2, 125, 3)
        cv2.circle(test1, tuple(mean[::-1]), 2, 125, 3)
        cv2.circle(test1, tuple(mean_[::-1]), 2, 125, 3)
        plt.imshow(mask)
        plt.show()

        cv2.imwrite('1.jpg', mask)
        cv2.imwrite('2.jpg', test)
        cv2.imwrite('3.jpg', test1)
        raise Exception

        assert result.size == 9
        assert result.dtype == np.float32
        return result


if __name__ == '__main__':
    number_test = '2'

    path_test = 'data/test' + number_test
    path_depth = path_test + '/Depth/'
    path_mask = path_test + '/Mask/'
    path_rgb = path_test + '/RGB/'
    data = pd.read_csv(path_test + '/data.csv')

    list_depth = sorted(os.listdir(path_depth))
    list_mask = sorted(os.listdir(path_mask))
    list_rgb = sorted(os.listdir(path_rgb))

    assert len(list_depth) == len(list_mask) == len(list_rgb)
    max_iter = len(list_depth)

    solv = Solver()

    for i in range(max_iter):
        images = cv2.imread(path_depth + list_depth[i], cv2.IMREAD_UNCHANGED), \
                 cv2.imread(path_mask + list_mask[i], cv2.IMREAD_GRAYSCALE), \
                 cv2.imread(path_rgb + list_rgb[i], cv2.IMREAD_COLOR)
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images[1])
        plt.show()
        plt.imshow(images[2])
        plt.show()

        assert images[0] is not None or images[1] is not None or images[2] is not None

        start_time = time.time()
        result = solv.process(images)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(result, data[['head.x', 'head.y', 'head.z',
                            'wrist.L.x', 'wrist.L.y', 'wrist.L.z',
                            'wrist.R.x', 'wrist.R.y', 'wrist.R.z']].loc[0])
