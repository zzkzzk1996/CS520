#!/usr/bin/env python
# encoding: utf-8
# @project : CS520
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: FireMaze.py
# @time: 2019-09-25 16:23:48

from Project1.MazeRunner_Zekun.MyPriorityQueue import MyPriorityQueue
from Project1.MazeRunner_Zekun.MazeGenerator import Maze


def a_star(maze, heuristic):
    start, goal, route = [1, 1], [len(maze) - 2, len(maze) - 2], str([1, 1])
    directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
    pq = MyPriorityQueue()
    pq.push(priority=heuristic(start, goal), cur=start, path=0, route=route)
    while pq.qsize() != 0:
        (Priority, _, cur, path, route) = pq.pop()
        i, j = cur[0], cur[1]
        if cur == goal:
            return route
        for direct in directions:
            if maze[i + direct[0]][j + direct[1]] == 0:
                next, new_route = [i + direct[0], j + direct[1]], route + " -> " + str([i + direct[0], j + direct[1]])
                pq.push(priority=heuristic(next, goal) + path, cur=next, path=path + 1, route=new_route)
                maze[i + direct[0]][j + direct[1]] = -1
    return False


def manhattan_distance(start, end):
    return abs(end[0] - start[0]) + abs(end[1] - start[1])


def euclidean_distance(start, end):
    return ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5


if __name__ == '__main__':
    maze = Maze(10, 0.1).maze_generator()
    print(maze)
    res = a_star(maze, heuristic=euclidean_distance)
    print(res)
