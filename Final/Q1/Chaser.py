#!/usr/bin/env python
# encoding: utf-8
# @project : Final
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: Chaser.py
# @time: 2019-12-17 10:39:50

import numpy as np
import matplotlib.pyplot as plt
import time
import random


class Dot:
    def __init__(self, pos):
        self.pos = pos
        self.i = pos // 8
        self.j = pos % 8


class Chase:
    def __init__(self, pos1=None, pos2=None, pos3=None):
        dots = np.random.choice(64, 3, replace=False)
        self.sheep = Dot(dots[0])
        self.dog_1 = Dot(dots[1])
        self.dog_2 = Dot(dots[2])
        if pos1 is not None:
            self.sheep.pos = pos1
        if pos2 is not None:
            self.dog_1.pos = pos2
        if pos3 is not None:
            self.dog_2.pos = pos3
        self.count = 0
        self.flag1 = False
        self.flag2 = False

    def check_neighbor(self, num):
        direct_list = [-8, -1, 1, 8]
        direct_list_temp = [-8, -1, 1, 8]
        for direct in direct_list_temp:
            if not self.check_move(num, direct):
                direct_list.remove(direct)
        return direct_list
        # if self.sheep.pos - 8 == self.dog_1.pos or self.sheep.pos - 8 == self.dog_2.pos or self.sheep.pos - 8 < 0:
        #     direct_list.remove(-8)
        # if self.sheep.pos + 8 == self.dog_1.pos or self.sheep.pos + 8 == self.dog_2.pos or self.sheep.pos + 8 > 63:
        #     direct_list.remove(8)
        # if self.sheep.pos - 1 == self.dog_1.pos or self.sheep.pos - 1 == self.dog_2.pos or (
        #         self.sheep.pos - 1) % 8 == 7:
        #     direct_list.remove(-1)
        # if self.sheep.pos + 1 == self.dog_1.pos or self.sheep.pos + 1 == self.dog_2.pos or (
        #         self.sheep.pos + 1) % 8 == 0:
        #     direct_list.remove(1)

    # num is for determine which one you want to check, val is actually the direction
    # return false for this direction is invalid
    def check_move(self, num, direct):
        pos_list = [self.sheep.pos, self.dog_1.pos, self.dog_2.pos]
        new_pos = pos_list.pop(num) + direct
        if new_pos == pos_list[0] or new_pos == pos_list[1] or not self.check_valid(num, direct): return False
        return True

    def check_valid(self, num, direct):
        pos_list = [self.sheep.pos, self.dog_1.pos, self.dog_2.pos]
        if direct == -8 and pos_list[num] - 8 < 0: return False
        if direct == 8 and pos_list[num] + 8 > 63: return False
        if direct == -1 and (pos_list[num] - 1) % 8 == 7: return False
        if direct == 1 and (pos_list[num] + 1) % 8 == 0: return False
        return True

    def sheep_move(self):
        move_list = self.check_neighbor(0)
        # if self.flag1 and self.flag2:
        #     return True
        if not move_list:
            return self.sheep.pos != 0
        self.sheep.pos += np.random.choice(move_list)
        return True

    # def sheep_check(self):
    #     move_list = self.check_neighbor(0)
    #     # if self.flag1 and self.flag2:
    #     #     return True
    #     if not move_list:
    #         if self.sheep.pos == 63 or self.sheep.pos == 57:
    #             self.flip()
    #             return True
    #     return False
    #
    # def flip(self):
    #     direct1 = self.check_neighbor(1)
    #     direct2 = self.check_neighbor(2)
    #     if not direct1 or not direct2: return
    #     self.dog_move(1, direct1)
    #     self.dog_move(2, direct2)
    #     self.count += 1

    def dog_move(self, num, direct):
        # print("dog1 : " + str(self.dog_1.pos))
        # print("dog2 : " + str(self.dog_2.pos))
        # print ("direct : " + str(direct))
        if self.check_move(num, direct):
            if num == 1:
                self.dog_1.pos += direct
            elif num == 2:
                self.dog_2.pos += direct

    # def calculate_path(self, pos, des):
    #     if abs((pos // 8) - (des // 8)) == 0:
    #         return abs(pos - des)
    #     elif abs((pos // 8) - (des // 8)) > 0:
    #         return abs((pos // 8) - (des // 8)) + abs((pos % 8) - (des % 8))

    def set_destination(self):
        # move_list = self.check_neighbor()
        # if self.sheep.pos + 8 == self.dog_2.pos:
        #     self.flag2 = True
        # elif self.sheep.pos + 1 == self.dog_1.pos:
        #     self.flag1 = True
        #
        # if self.flag1 or self.flag2:
        #     return int(self.sheep.pos + 1), int(self.sheep.pos + 8)
        # if 8 in move_list and 1 in move_list:
        #     return int(self.sheep.pos + 8), int(self.sheep.pos + 1)
        # print(self.sheep.pos)
        return int(self.sheep.pos + 8), int(self.sheep.pos + 1)

    def move_direct(self, num, pos, des):
        temp_list = [-1, -8]
        if pos > des:
            if pos // 8 == des // 8 and self.check_move(num, -1):
                return -1
            elif pos // 8 > des // 8:
                if pos % 8 < des % 8 and self.check_move(num, 1):
                    return 1
                elif pos % 8 >= des % 8 and self.check_move(num, -8):
                    if pos % 8 == des % 8:
                        return -8
                    if self.check_move(num, -1):
                        return random.choice(temp_list)

        elif pos < des:
            if pos // 8 == des // 8:
                if num == 1 and self.check_move(num, 1):
                    return 1
                elif num == 2 and self.check_move(num, 8):
                    return 8
            elif pos // 8 < des // 8 and self.check_move(num, 8):
                if pos % 8 < des % 8 and self.check_move(num, 1):
                    return 1
                elif pos % 8 >= des % 8 and self.check_move(num, 8):
                    # if self.check_move(num, 1):
                    #     return 1
                    return 8
                elif num == 1 and pos % 8 == des % 8 and self.check_move(num, 1):
                    return 1
        return 0

    def chase(self):
        while self.sheep_move():
            des1, des2 = self.set_destination()
            direct1 = self.move_direct(1, self.dog_1.pos, des1)
            direct2 = self.move_direct(2, self.dog_2.pos, des2)
            if self.count > 1000:
                if self.sheep.pos == 63:
                    self.plot()
                if self.sheep.pos == 57:
                    self.plot()
                return -1
            # if self.sheep_check(): continue
            self.dog_move(1, direct1)
            self.dog_move(2, direct2)
            self.count += 1
            # print(self.count)
            # self.plot()
            # time.sleep(0.2)
        return int(self.count)

    def plot(self):
        tmp_arr = np.zeros((8, 8))
        tmp_arr[0][0] = -1
        tmp_arr[int(self.sheep.pos // 8)][int(self.sheep.pos % 8)] = 3
        tmp_arr[int(self.dog_1.pos // 8)][int(self.dog_1.pos % 8)] = 2
        tmp_arr[int(self.dog_2.pos // 8)][int(self.dog_2.pos % 8)] = 1
        plt.figure(figsize=(5, 5))
        plt.pcolor(tmp_arr[::-1], edgecolors='black', cmap='BuGn', linewidths=2)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
        time.sleep(0.2)
        plt.close('all')


def test():
    # chase = Chase(pos1=32, pos2=51, pos3=7)
    chase = Chase()
    res = chase.chase()
    # print(res)
    return res


# file = "/Users/zekunzhang/PycharmProjects/Final/result.txt"
if __name__ == '__main__':
    # test()
    times = 10
    result, fail = 0, 0
    for i in range(times):
        temp = test()
        if temp != -1:
            result += temp
        else:
            fail += 1
        with open('/Users/zekunzhang/PycharmProjects/Final/result.txt', 'w') as f:
            f.write(str(result) + "," + str(result // (i + 1)) + "," + str(fail))
        print("Progress:{}%".format((i * 100 / times)), flush=True)
