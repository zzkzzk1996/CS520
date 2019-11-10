#!/usr/bin/env python
# encoding: utf-8
# @project : Project3
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: Map.py
# @time: 2019-11-08 15:45:40

import numpy as np
import matplotlib.pyplot as plt


# class Node:
#     def __init__(self, i=0, j=0):
#         self.i = i
#         self.j = j
#         self.loc = (self.i, self.j)


class Map:
    def __init__(self, dim=10):
        self.mappings = [0.1, 0.3, 0.7, 0.9]
        self.dim = dim
        self.origin_map = self.mapper()

    def mapper(self):
        origin_map = self.generate_map()
        return origin_map

    def generate_map(self):
        new_arr = np.random.choice((0, 1, 2, 3), p=[0.2, 0.3, 0.3, 0.2], size=(self.dim, self.dim))
        map = new_arr.reshape(self.dim, self.dim)
        print(map)
        return map

    def printer(self, map=None):
        map = self.origin_map.copy() if map is None else map.copy()
        plt.figure(figsize=(5, 5))
        plt.pcolor(map[::-1], edgecolors='black', cmap='BuGn', linewidths=2)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
