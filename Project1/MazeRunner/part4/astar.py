from part4.util import MyPriorityQueue
from part4.Maze import Maze_generater
from copy import deepcopy


def a_star(maze, heuristic):
    start, goal, route = [1, 1], [len(maze) - 2, len(maze) - 2], [(1, 1)]
    # start, goal, route = [1, 1], [len(maze) - 2, len(maze) - 2], str([1, 1])
    directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
    pq = MyPriorityQueue()
    pq.push(priority=heuristic(start, goal), cur=start, path=0, route=route)
    mc = deepcopy(maze)
    while pq.qsize() != 0:
        (Priority, _, cur, path, route) = pq.pop()
        i, j = cur[0], cur[1]
        if cur == goal:
            return route
        for direct in directions:
            if mc[i + direct[0]][j + direct[1]] == 0:
                next = [i + direct[0], j + direct[1]]
                # new_route = route + "->" + str([i + direct[0], j + direct[1]])
                new_route = deepcopy(route)
                new_route.append((i + direct[0], j + direct[1]))
                # route.append((i + direct[0], j + direct[1]))
                pq.push(priority=heuristic(next, goal) + path, cur=next, path=path + 1, route=new_route)
                mc[i + direct[0]][j + direct[1]] = -1
    return False


def manhattan_distance(start, end):
    return abs(end[0] - start[0]) + abs(end[1] - start[1])


def euclidean_distance(start, end):
    return ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5


if __name__ == '__main__':
    maze = Maze_generater().generate_maze(10, 0.1)
    print(maze)
    res = a_star(maze, heuristic=manhattan_distance)
    # res = str.split(res, "->")
    # p = list()
    # for point in res:
    #     point = point[1:-1]
    #     point = str.split(point, ", ")
    #     i = int(point[1])
    #     j = int(point[4])
    #     p.append((i, j))
    # print(maze)
    print(res)
    # print(p)
