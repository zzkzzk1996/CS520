#!/usr/bin/env python
# encoding: utf-8
# @project : Assignment3
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: Stationery.py
# @time: 2019-11-08 16:42:12
import numpy as np


class Explorer:
    def __init__(self, origin_map):
        self.origin_map = origin_map.origin_map  # origin map which contains 4 different terrain type from 0 to 3
        self.row, self.col = np.shape(self.origin_map)
        self.mappings = origin_map.mappings  # sets for different p with different terrain type

        self.rand = np.random.randint(0, self.row * self.col)  # rand seed
        self.target = (self.rand // self.col, self.rand % self.col)  # random target generated by rand seed

        self.p_map_s1 = np.ones((self.row, self.col)) / (
                self.row * self.col)  # possibility map for search under stationery one
        self.p_map_s2 = np.ones((self.row, self.col)) / (
                self.row * self.col)  # possibility map for search under stationery two
        self.p_map_s3 = np.ones((self.row, self.col)) / (
                self.row * self.col)  # possibility map for search under stationery three

        self.search_one_count = 0  # count for search under stationery one
        self.search_one_action = 0  # actions for search under stationery one
        self.search_two_count = 0  # count for search under stationery two
        self.search_two_action = 0  # actions for search under stationery two
        self.search_three_count = 0  # count for search under stationery three
        self.search_three_action = 0  # actions for search under stationery three

        # print("Target is" + str(self.target))

    def search_one(self):  # search under stationery one
        while True:
            self.search_one_count += 1
            grid = self.grid_choice1()
            if self.check(grid):
                break
            else:
                observation = 1 - self.p_map_s1[grid] + self.p_map_s1[grid] * self.mappings[self.origin_map[grid]]
                self.p_map_s1[grid] *= self.mappings[self.origin_map[grid]]
            self.p_map_s1 /= observation
        # print("Search times: " + str(self.search_one_count1))
        return self.search_one_count

    def search_two(self):  # search under stationery two
        while True:
            self.search_two_count += 1
            grid = self.grid_choice2()
            if self.check(grid):
                break
            else:
                observation = 1 - self.p_map_s2[grid] + self.p_map_s2[grid] * self.mappings[self.origin_map[grid]]
                self.p_map_s2[grid] *= self.mappings[self.origin_map[grid]]
            self.p_map_s2 /= observation
        # print("Search times: " + str(self.search_one_count2))
        return self.search_two_count

    def check(self, grid):  # check one grid failure or success
        if self.target != grid:
            return False
        else:
            random = np.random.random()  # generate a random p from 0 to 1
            return random >= self.mappings[self.origin_map[grid]]  # check return a failure or success

    def grid_choice1(self):
        candidates = np.argwhere(self.p_map_s1 == np.max(self.p_map_s1))  # find all with max possibility ones
        return tuple(candidates[np.random.choice(len(candidates))])  # randomly choose one and return

    def grid_choice2(self):
        arr4 = np.argwhere((self.origin_map == 3) & (self.p_map_s2 == np.max(self.p_map_s2)))
        arr3 = np.argwhere((self.origin_map == 2) & (self.p_map_s2 == np.max(self.p_map_s2)))
        arr2 = np.argwhere((self.origin_map == 1) & (self.p_map_s2 == np.max(self.p_map_s2)))
        arr1 = np.argwhere((self.origin_map == 0) & (self.p_map_s2 == np.max(self.p_map_s2)))
        if arr1.size > 0:
            candidates = arr1
        elif arr2.size > 0:
            candidates = arr2
        elif arr3.size > 0:
            candidates = arr3
        else:
            candidates = arr4
        return tuple(candidates[np.random.choice(len(candidates))])

    # ================================================================================================================ #

    def get_distance(self, p, q):  # get distance between two grids
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    def search_one_with_actions(self):  # stationery rule one search with actions
        while True:
            self.search_one_count += 1
            grid = self.grid_choice1()
            self.search_one_action += self.get_distance(grid, grid if self.search_one_count == 1 else pre_grid)
            if self.check(grid):
                break
            else:
                observation = 1 - self.p_map_s1[grid] + self.p_map_s1[grid] * self.mappings[self.origin_map[grid]]
                self.p_map_s1[grid] *= self.mappings[self.origin_map[grid]]
            self.p_map_s1 /= observation
            pre_grid = grid
        # print("Search times: " + str(self.search_one_count1))
        return self.search_one_action + self.search_one_count, self.search_one_count

    def search_two_with_actions(self):  # stationery rule two search with actions
        while True:
            self.search_two_count += 1
            grid = self.grid_choice2()
            self.search_two_action += self.get_distance(grid, grid if self.search_two_count == 1 else pre_grid)
            if self.check(grid):
                break
            else:
                observation = 1 - self.p_map_s2[grid] + self.p_map_s2[grid] * self.mappings[self.origin_map[grid]]
                self.p_map_s2[grid] *= self.mappings[self.origin_map[grid]]
            self.p_map_s2 /= observation
            pre_grid = grid
        # print("Search times: " + str(self.search_one_count2))
        return self.search_two_action + self.search_two_count, self.search_two_count

    def search_three_with_actions(self):
        grid = self.grid_choice2()
        pre_grid = grid
        while True:
            self.search_three_count += 1
            grid = self.grid_choice3(pre_grid)
            self.search_three_action += self.get_distance(grid, pre_grid)
            # print(self.search_three_action)
            if self.check(grid):
                break
            else:
                observation = 1 - self.p_map_s3[grid] + self.p_map_s3[grid] * self.mappings[self.origin_map[grid]]
                self.p_map_s3[grid] *= self.mappings[self.origin_map[grid]]
            self.p_map_s3 /= observation
            pre_grid = grid
        # print("Search times: " + str(self.search_one_count2))
        return self.search_three_action + self.search_three_count, self.search_three_count

    def grid_choice3(self, pre_grid):
        arr4 = np.argwhere((self.origin_map == 3) & (self.p_map_s3 == np.max(self.p_map_s3)))
        arr3 = np.argwhere((self.origin_map == 2) & (self.p_map_s3 == np.max(self.p_map_s3)))
        arr2 = np.argwhere((self.origin_map == 1) & (self.p_map_s3 == np.max(self.p_map_s3)))
        arr1 = np.argwhere((self.origin_map == 0) & (self.p_map_s3 == np.max(self.p_map_s3)))
        if arr1.size > 0:
            candidates = arr1
        elif arr2.size > 0:
            candidates = arr2
        elif arr3.size > 0:
            candidates = arr3
        else:
            candidates = arr4
        min_grids = np.argwhere(
            abs(candidates - pre_grid).sum(axis=1) - np.min(
                abs(candidates - pre_grid).sum(axis=1)) < 10).flatten().tolist()
        return tuple(candidates[np.random.choice(min_grids)])
