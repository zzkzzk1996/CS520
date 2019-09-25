#!/usr/bin/env python
# encoding: utf-8
# @project : CS520
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: HardMazeNoClass.py
# @time: 2019-09-25 16:42:25


from Project1.MazeRunner.MyPriorityQueue import HillClimbPQ, MyPriorityQueue
from Project1.MazeRunner.MazeGenerator import Maze


def get_hard_maze(maze, algorithm):
    harder = True
    hardness = algorithm(maze)
    print(maze)
    while harder:
        harder, maze, new_hardness = hill_climb(maze, algorithm)
    print(maze)
    print(str(hardness) + " -> " + str(new_hardness))


def hill_climb(maze, algorithm):
    pq = HillClimbPQ()
    length = algorithm(maze.copy())
    dim = len(maze)
    for row in range(1, dim - 1):
        for col in range(1, dim - 1):
            if not ((row == 1 and col == 1) or (row == dim - 1 and col == dim - 1)):
                temp_maze = flip(maze.copy(), row, col)
                if is_valid(temp_maze, algorithm):
                    new_length = algorithm(temp_maze.copy())
                    if length < new_length:
                        length = new_length
                        pq.push(priority=new_length, maze=temp_maze)

    if pq.qsize() == 0:
        return False, maze, length
    else:
        while pq.qsize() != 0:
            (length, _, maze) = pq.pop()
            # print(length)
        return True, maze, length


def flip(maze, row, col):
    dim = len(maze)
    if not ((row == 1 and col == 1) or (row == dim - 1 and col == dim - 1)):
        maze[row][col] = 0 if maze[row][col] == 1 else 1
    return maze


def depth_first_search(maze):
    m = maze.copy()
    start, goal = [1, 1], [len(maze) - 2, len(maze) - 2]
    stack = [start]
    directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
    max_fringe = 0
    while len(stack) != 0:
        cur = stack[-1]
        max_fringe = max(max_fringe, len(stack))
        i, j = cur[0], cur[1]
        if cur == goal:
            return max_fringe
        for direct in directions:
            if m[i + direct[0]][j + direct[1]] == 0:
                stack.append([i + direct[0], j + direct[1]])
                m[i + direct[0]][j + direct[1]] = -1
                break
        else:
            stack.pop()
            m[i][j] = -1
    return False


def is_valid(maze, algorithm):
    return False if algorithm(maze) is False else True


def manhattan_distance(start, end):
    return abs(end[0] - start[0]) + abs(end[1] - start[1])


def a_star(maze):
    m = maze.copy()
    start, goal = [1, 1], [len(maze) - 2, len(maze) - 2]
    directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
    pq = MyPriorityQueue()
    pq.push(priority=manhattan_distance(start, goal), cur=start, path=0, route=None)
    while pq.qsize() != 0:
        (Priority, _, cur, path, route) = pq.pop()
        i, j = cur[0], cur[1]
        if cur == goal:
            return path
        for direct in directions:
            if m[i + direct[0]][j + direct[1]] == 0:
                next = [i + direct[0], j + direct[1]]
                pq.push(priority=manhattan_distance(next, goal) + path, cur=next, path=path + 1, route=None)
                m[i + direct[0]][j + direct[1]] = -1
    return False


if __name__ == '__main__':
    # for a star
    maze = Maze(10, 0.3).maze_generator()
    while not is_valid(maze, a_star):
        maze = Maze(10, 0.3).maze_generator()
    get_hard_maze(maze, a_star)
