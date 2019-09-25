from util import node
from copy import deepcopy
import numpy as np
from random import randint
from Maze import Maze_generater
from new_search import dfs


# draw = Maze_generater()


def update_fringe(maze, fringe_now, p):
    # fringe_next = list()
    dir_array = ([0, 1], [1, 0], [0, -1], [-1, 0])
    fringe_next = list()
    for fire_spot in fringe_now:
        counter = 0
        # print(fire_spot.loc)
        for dir in dir_array:
            ember_spot = node(fire_spot.i + dir[0], fire_spot.j + dir[1])
            if maze[ember_spot.loc] != 2 and maze[ember_spot.loc] != 4 and maze[ember_spot.loc] != 5:
                # print(0)
                set_fire(maze, ember_spot, p)
            if maze[ember_spot.loc] == 4:
                # print(-1)
                # 4 means this spot will catch on fire this time slot
                fringe_next.append(ember_spot)
            # print(counter)
            if maze[ember_spot.loc] == 5 or maze[ember_spot.loc] == 4 or maze[ember_spot.loc] == 2:
                counter = counter + 1
                if counter == 4 and (maze[fire_spot.loc] == 5 or maze[fire_spot.loc] == 4):
                    fringe_now.remove(fire_spot)
    for ember_spot in fringe_next:
        maze[ember_spot.loc] = 5
        fringe_now.append(ember_spot)
    return fringe_now


def set_fire(maze, ember_spot, p):
    dir_array = ([0, 1], [1, 0], [0, -1], [-1, 0])
    counter = 0
    for dir in dir_array:
        spot = node(ember_spot.i + dir[0], ember_spot.j + dir[1])
        if maze[spot.loc] == 5:
            counter = counter + 1
    fire_p = 1 - pow(1 - p, counter)
    fire_p = fire_p * 100
    rand = randint(0, 100)
    if rand < fire_p:
        maze[ember_spot.loc] = 4


def walk_on_fire(maze, p):
    start_node = node(1, 1)
    fire_spot = node(1, len(maze) - 2)
    goal_node = node(len(maze) - 2, len(maze) - 2)
    mc = deepcopy(maze)
    mc[start_node.loc] = 3
    mc[fire_spot.loc] = 5
    dir_array = ([0, 1], [1, 0], [0, -1], [-1, 0])
    """
    I choose loop to simulate the time process, one loop is one time unite
    use dfs to find the way, and bfs to simulate fire express
    """
    stack = list()
    fringe_now = list()
    stack.append(start_node)
    fringe_now.append(fire_spot)
    while len(stack) != 0:
        per_cur = stack[-1]
        if per_cur.loc == goal_node.loc:
            # return np.count_nonzero(mc == 3), mc
            return stack, mc
        # print(1)
        update_fringe(mc, fringe_now, p)
        if mc[per_cur.loc] == 5 or mc[len(mc) - 1, len(mc) - 1] == 5:
            return [], mc
        for dir in dir_array:
            per_next = node(per_cur.i + dir[0], per_cur.j + dir[1])
            if mc[per_next.loc] == 0:
                stack.append(per_next)
                mc[per_next.loc] = 3
                break
        else:
            stack.pop()
            if mc[per_cur.loc] < 2:
                mc[per_cur.loc] = 3
        # draw.draw_maze(mc)

    return [], mc


def simple_walk(maze, p):
    stack = list()
    while len(stack) == 0:
        stack, mc = dfs(maze)
    # the path it attempted to go
    path = np.where(mc == 3)
    path_len = len(path[0])
    path = get_path(path)
    fringe = list()
    fire_spot = node(1, len(maze) - 2)
    if mc[fire_spot.loc] == 3:
        return [], mc
    mc[fire_spot.loc] = 5
    fringe.append(fire_spot)
    while path_len > 0:
        path_len = path_len - 1
        fringe = update_fringe(mc, fringe, p)
        for fire_spot in fringe:
            if fire_spot.loc in path:
                return [], mc
    return path, mc


def get_path(path):
    p = list()
    for i, j in zip(path[0], path[1]):
        point = (i, j)
        p.append(point)
    return p
