#!/usr/bin/env python
# encoding: utf-8
# @project : CS520
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: HardMaze.py
# @time: 2019-09-24 11:06:47

from Project1.MazeRunner.MyPriorityQueue import HillClimbPQ
from Project1.MazeRunner.MazeGenerator import Maze
import numpy as np


class HardMaze:

    def __init__(self, maze, algorithm, heuristic, p):
        self.queue = []
        self.pq = HillClimbPQ()
        self.maze = maze
        self.algorithm = algorithm
        self.heuristic = heuristic
        self.p = p
        self.length = 0
        self.harder = False

    def get_hard_maze(self):
        self.queue.append(self.maze)
        if is_valid(self.maze):
            self.length = self.algorithm(self.maze.copy(), self.algorithm)
            self.harder = True
            self.pq.push(priority=self.length, maze=len(self.queue) - 1, length=self.length)
        while self.harder:
            self.hill_climb()
        else:
            print("Input can't be solved")
        print(self.maze)

    def hill_climb(self):
        temp_maze = self.flip(self.maze.copy())
        if is_valid(temp_maze):
            length = self.algorithm(temp_maze,
                                    self.algorithm)  # algorithm(flip(maze, p), heuristic) if heuristic is not None else
            if length > self.length:
                self.maze = temp_maze
                self.queue.append(self.maze)
                self.pq.push(priority=length, maze=len(self.queue) - 1, length=length)

            if self.pq.qsize() != 0:
                (priority, _, old_maze, old_length) = self.pq.pop()
                self.maze = self.queue[old_maze]
                # self.length = old_length
            else:
                self.harder = False

    def flip(self, maze):
        dim = len(maze)
        amount = dim * dim
        row, col = amount // dim, amount % dim
        if (0 < row < dim) and (0 < col < dim):
            if not ((row == 1 and col == 1) or (row == dim - 1 and col == dim - 1)):
                maze[row][col] = ~maze[row][col] if np.random.choice(2, 1, p=[1 - self.p, self.p]) == [1] else \
                    maze[row][col]
        return maze


max_fringe = 0


def depth_first_search(maze, algorithm):
    start, goal = [1, 1], [len(maze) - 2, len(maze) - 2]
    stack = [start]
    directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
    while len(stack) != 0:
        cur = stack[-1]
        global max_fringe
        max_fringe = max(max_fringe, len(stack))
        i, j = cur[0], cur[1]
        if cur == goal:
            if algorithm is None:
                return np.count_nonzero(maze == -1)
            else:
                return algorithm
        for direct in directions:
            if maze[i + direct[0]][j + direct[1]] == 0:
                stack.append([i + direct[0], j + direct[1]])
                maze[i + direct[0]][j + direct[1]] = -1
                break
        else:
            stack.pop()
            maze[i][j] = -1
    return False


def get_max_fringe():
    return max_fringe


def is_valid(maze):
    m = maze.copy()
    return False if depth_first_search(m, None) is False else True


if __name__ == '__main__':
    maze = Maze(10, 0.1).maze_generator()
    HardMaze(maze, depth_first_search, get_max_fringe(), 0.5)
