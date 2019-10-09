import copy
import numpy as np
from Part4.util import node

"""
The bfs and dfs implementation file.
"""
def dfs(maze):
    stack = list()
    mc = copy.deepcopy(maze)
    dir_array = ([0, 1], [1, 0], [0, -1], [-1, 0])
    start_node = node(1, 1)
    mc[1, 1] = 3
    goal_node = node(len(maze) - 2, len(maze) - 2)
    stack.append(start_node)
    while len(stack) != 0:
        cur = stack[-1]
        if cur.loc == goal_node.loc:
            return stack, mc
        for dir in dir_array:
            next_node = node(cur.i + dir[0], cur.j + dir[1])

            if mc[next_node.loc] == 0:
                stack.append(next_node)
                mc[next_node.loc] = 3
                break
        else:
            stack.pop()
            mc[cur.loc] = 3
    return [], mc


def bfs(maze):
    queue = list()
    mc = copy.deepcopy(maze)
    dir_array = ([0, 1], [1, 0], [0, -1], [-1, 0])
    start_node = node(1, 1)
    mc[1, 1] = 3
    goal_node = node(len(maze) - 2, len(maze) - 2)
    queue.append(start_node)
    while len(queue) != 0:
        cur = queue.pop(-1)
        if cur.loc == goal_node.loc:
            return np.count_nonzero(mc == 3), mc
        for dir in dir_array:
            next_node = node(cur.i + dir[0], cur.j + dir[1])

            if mc[next_node.loc] == 0:
                queue.append(next_node)
                mc[next_node.loc] = 3
    return [], mc


