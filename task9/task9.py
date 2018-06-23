import numpy as np


class DSU:
    """
    parent - it is array, where saved link on object
    In start, object it is pixels of image, then segment of pixels.

    """
    def __init__(self, image: np.array):
        self.image = image
        self.parent = {}
        self.MST = {}
        self.power_set = {}
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                index = (i, j)
                self.power_set[self._get_index_parent(index=index)] = 1
                self.MST[self._get_index_parent(index=index)] = 0
                self.make_set(self._get_index_parent(index=index))

    def _get_index_image(self, index):
        return index // self.image.shape[1], index % self.image.shape[1]

    def _get_index_parent(self, index):
        return index[0] * self.image.shape[1] + index[1]

    def _get_neighbors(self, index):
        y, x = self._get_index_image(index=index)
        index_list = [[y, x + 1], [y, x - 1], [y + 1, x], [y - 1, x]]
        abcs = list(filter(lambda elem: False if elem[0] < 0 or
                                                 elem[0] >= self.image.shape[0] or
                                                 elem[1] < 0 or
                                                 elem[1] >= self.image.shape[1]
                                              else True, index_list))
        return [self._get_index_parent(abc) for abc in abcs]

    def _w(self, index1, index2):
        y1, x1 = self._get_index_image(index1)
        y2, x2 = self._get_index_image(index2)
        return np.sum((self.image[y1][x1] -
                       self.image[y2][x2]) ** 2)

    def make_set(self, x):
        self.parent[x] = x

    def union_sets(self, a, b):
        a = self.parent[a]
        b = self.parent[b]
        if a != b:
            self.parent[b] = a

    def find_set(self, x):
        if x == self.parent[x]:
            return x
        self.parent[x] = self.find_set(self.parent[x])
        return self.parent[x]

    def task9(self, k: float):
        # идти по сегментам
        for C in self.parent:
            # нахожу соседей
            neighborts = self._get_neighbors(C)
            # сортирую соседей по пиксельной близости
            sorted(neighborts, key=lambda x: self._w(C, x))
            # иду по соседям, начиная с ближайшего
            for neighbort in neighborts:
                # если у соседа тот же предок что и у текущего элемента, то ничего не делаю
                parent_C = self.find_set(C)
                parent_neighbort = self.find_set(neighbort)
                if parent_C == parent_neighbort:
                    continue
                else:
                    # иначе, нужно решить, сливать сегменты или нет
                    # если,
                    if self._w(C, neighbort) <= min([self.MST[parent_C] +
                                                     k * 1. / self.power_set[parent_C],
                                                     self.MST[parent_neighbort] +
                                                     k * 1. / self.power_set[parent_neighbort]]):
                        self.union_sets(parent_C, parent_neighbort)
                        new_leader = self.find_set(parent_C)
                        power_set_C = self.power_set[parent_C]
                        power_set_neighbort = self.power_set[parent_neighbort]
                        self.power_set[new_leader] = 0
                        self.power_set[new_leader] = power_set_C + power_set_neighbort
                        self.MST[new_leader] = max([self._w(C, neighbort),
                                                    self.MST[parent_C],
                                                    self.MST[parent_neighbort]])
                    else:
                        continue
        return self.parent

    def draw(self):
        img = np.zeros(self.image.shape)
        val = self.parent.values()
        val_uniq = np.unique(list(val))
        print(val_uniq.shape)
        for x in range(self.image.shape[1]):
            for y in range(self.image.shape[0]):
                index_par = self.find_set(self._get_index_parent((y, x)))
                img[y][x] = np.array([(index_par % (255*255)) % 255,
                                      index_par % 255,
                                      (index_par // 255) % 255])
        return img
