#!/usr/bin/env python
# encoding: utf-8
# @project : CS520
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: SearchAlgorithm.py
# @time: 2019-09-24 07:12:26

import numpy as np
from Project1.MazeRunner.MyPriorityQueue import MyPriorityQueue

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


def breadth_first_search(maze):
    start, goal = [1, 1], [len(maze) - 2, len(maze) - 2]
    queue = [start]
    directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
    while len(queue) != 0:
        cur = queue.pop(0)
        i, j = cur[0], cur[1]
        if cur == goal:
            return np.count_nonzero(maze == -1)
        for direct in directions:
            if maze[i + direct[0]][j + direct[1]] == 0:
                queue.append([i + direct[0], j + direct[1]])
                maze[i + direct[0]][j + direct[1]] = -1
    return False


def manhattan_distance(start, end):
    return abs(end[0] - start[0]) + abs(end[1] - start[1])


def euclidean_distance(start, end):
    return ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5


def a_star(maze, heuristic):
    start, goal = [1, 1], [len(maze) - 2, len(maze) - 2]
    directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
    pq = MyPriorityQueue()
    pq.push(priority=heuristic(start, goal), cur=start, path=0, route=None)
    while pq.qsize() != 0:
        (Priority, _, cur, path, route) = pq.pop()
        i, j = cur[0], cur[1]
        if cur == goal:
            return np.count_nonzero(maze == -1)
        for direct in directions:
            if maze[i + direct[0]][j + direct[1]] == 0:
                next = [i + direct[0], j + direct[1]]
                pq.push(priority=heuristic(next, goal) + path, cur=next, path=path + 1, route=None)
                maze[i + direct[0]][j + direct[1]] = -1
    return False


def get_max_fringe():
    return max_fringe
