#!/usr/bin/env python
# encoding: utf-8
# @project : CS520
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: MazeRunner.py
# @time: 2019-09-20 15:02:17

from Project1.MazeRunner.MazeGenerator import Maze
from Project1.MazeRunner.SearchAlgorithm import manhattan_distance, euclidean_distance
import SearchAlgorithm

if __name__ == '__main__':
    maze = Maze(10, 0.5).maze_generator()
    # res = SearchAlgorithm.depth_first_search(maze)
    # res = SearchAlgorithm.breadth_first_search(maze)
    res = SearchAlgorithm.a_star(maze, heuristic=euclidean_distance)
    print(res)
    print(maze)
