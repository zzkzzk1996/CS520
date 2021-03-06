#!/usr/bin/env python
# encoding: utf-8
# @project : CS520
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: minemap.py
# @time: 2019-10-21 16:34:21

import numpy as np
import matplotlib.pyplot as plt

dir_arr = ([0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1])


class MineMap:
    """
    class for generating mine map
    """

    def __init__(self, x=10, y=10, p=0.15):
        self.x = x
        self.y = y
        self.mine_number = p * x * y if 1 > p > 0 else p
        self.generate_board()

    def generate_map(self):
        """
        :return: the mine map
        generate mine map with given possibility of mine or given number of mines
        """
        x, y, mine_number = self.x, self.y, int(self.mine_number)
        tmp_arr = np.zeros(x * y)
        mines = np.random.choice(range(x * y), size=mine_number, replace=False)

        for mine in mines:
            tmp_arr[mine] = -1

        map = tmp_arr.reshape(x, y)
        return map

    def generate_tips(self, board):
        """
        :param board: the mine map
        generate the mine tips for player
        :return the tipped mine map
        """
        x, y = board.shape[0], board.shape[1]
        bigboard = np.zeros((x + 2, y + 2))  # generate a bigger map to avoid complex logical condition determination
        bigboard[1:x + 1, 1:y + 1] = board
        mine_positions = np.argwhere(bigboard == -1)
        # mark the mine's adjacent position to 1, 2, 3 or so
        for i, j in mine_positions:
            mine = (i, j)
            for dir in dir_arr:
                m = mine[0] + dir[0]
                n = mine[1] + dir[1]
                if bigboard[m, n] == -1:
                    continue
                else:
                    bigboard[m, n] += 1

        board = bigboard[1:x + 1, 1:y + 1]
        return board

    def generate_board(self):
        board = self.generate_map()
        board = self.generate_tips(board)
        self.board = board

    def drawboard(self, originalboard=None):
        if originalboard is None:
            board = self.board.copy()
        else:
            board = originalboard.copy()
        board[board == -1] = -5
        board[board == -2] = -12
        if (board != -2).all():
            board[board == -1] = -2
            # board[board>0]=11
            board[board == 0] = 0

        plt.figure(figsize=(5, 5))
        plt.pcolor(-board[::-1], edgecolors='black', cmap='bwr', linewidths=2)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()