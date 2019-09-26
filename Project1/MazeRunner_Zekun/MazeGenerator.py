#!/usr/bin/env python
# encoding: utf-8
# @project : CS520
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: MazeGenerator.py
# @time: 2019-09-20 15:01:43

import numpy as np


class Maze:

    def __init__(self, dim, p):
        self.dim, self.p = dim, p
        self.maze = np.zeros((int(dim) + 2, int(dim) + 2), dtype=int)  # Initialize a new array as maze

    # Getter function
    def get_dim(self):
        return self.dim

    def maze_generator(self):
        dim, p = self.dim, self.p
        # add a barrier for the maze
        self.maze[0, :] = self.maze[-1, :] = self.maze[:, 0] = self.maze[:, -1] = 2
        # fill the maze with '0's and '1's by the probability p
        for row in range(1, dim + 1):
            for col in range(1, dim + 1):
                if not ((row == 1 and col == 1) or (row == dim and col == dim)):
                    self.maze[row][col] = 1 if np.random.choice(2, 1, p=[1 - p, p]) == [1] else 0
        # print(self.maze)
        return self.maze


'''
maze = Maze(10, 0.5).maze_generator()
'''
