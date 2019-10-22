#!/usr/bin/env python
# encoding: utf-8
# @project : CS520
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: MazeRunner_Zekun.py
# @time: 2019-09-20 15:02:17

from Project1.MazeRunner_Zekun.MazeGenerator import Maze
from Project1.MazeRunner_Zekun.SearchAlgorithm import manhattan_distance, euclidean_distance, depth_first_search, \
    get_max_fringe
from Project1.MazeRunner_Zekun.HardMaze import HardMaze
import SearchAlgorithm
import matplotlib.pyplot as plt


def draw_maze(maze):
    if maze is None:
        return
    else:
        plt.figure(figsize=(5, 5))
        plt.pcolor(maze[::-1], edgecolors='black', cmap='Blues', linewidths=1)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    maze = Maze(10, 0.5 ).maze_generator()
    draw_maze(maze)

    # res = SearchAlgorithm.depth_first_search(maze)
    # res = SearchAlgorithm.breadth_first_search(maze)
    # res = SearchAlgorithm.a_star(maze, heuristic=euclidean_distance)
    # res = SearchAlgorithm.a_star(maze, heuristic=manhattan_distance)
    # print(res)
    # print(maze)
    # HardMaze(maze, depth_first_search, get_max_fringe, 0.01).get_hard_maze()
