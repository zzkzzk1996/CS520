from part4.util import node
from copy import deepcopy
from random import randint
from part4.util import MyPriorityQueue
from part4.astar import a_star, manhattan_distance


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
        # if mc[per_cur.loc] == 5 or mc[len(mc) - 1, len(mc) - 1] == 5:
        #     return [], mc
        if mc[len(mc) - 2, len(mc) - 2] == 5:
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


def astar_walk_on_fire(maze, heuristic, p):
    dim = len(maze) - 2
    start, goal, route = [1, 1], [len(maze) - 2, len(maze) - 2], [(1, 1)]
    # start, goal, route = [1, 1], [len(maze) - 2, len(maze) - 2], str([1, 1])
    directions = [[1, 0], [0, 1], [0, -1], [-1, 0]]
    pq = MyPriorityQueue()
    pq.push(priority=heuristic(start, goal), cur=start, path=0, route=route)
    mc = deepcopy(maze)
    fringe = list()
    fire_spot = node(1, len(maze) - 2)
    mc[fire_spot.loc] = 5
    fringe.append(fire_spot)
    while pq.qsize() != 0:
        (Priority, _, cur, path, route) = pq.pop()
        i, j = cur[0], cur[1]
        if cur == goal:
            return route, mc
        fringe, mc = update_map(mc, fringe, p)
        for direct in directions:
            if mc[i + direct[0]][j + direct[1]] == 0:
                next = [i + direct[0], j + direct[1]]
                danger = 0

                if ((next[1] < dim and mc[next[0], next[1] + 1] == 5) or (
                        next[0] < 10 and mc[next[0] + 1, next[1]] == 5)):
                    danger = 0
                elif ((next[1] < dim - 1 and mc[next[0], next[1] + 2] == 5) or (
                        next[0] < 9 and mc[next[0] + 2, next[1]] == 5)):
                    danger = 2
                elif ((next[1] < dim - 2 and mc[next[0], next[1] + 3] == 5) or (
                        next[0] < 8 and mc[next[0] + 3, next[1]] == 5)):
                    danger = 1
                else:
                    danger = -1

                # new_route = route + "->" + str([i + direct[0], j + direct[1]])
                new_route = deepcopy(route)
                new_route.append((i + direct[0], j + direct[1]))
                # route.append((i + direct[0], j + direct[1]))
                pq.push(priority=heuristic(next, goal) + path + danger, cur=next, path=path + 1, route=new_route)
                mc[i + direct[0]][j + direct[1]] = -1
    return [], mc


def simple_walk(maze, p):
    path = a_star(maze, heuristic=manhattan_distance)
    # path = get_path(route)
    # the path it attempted to go
    # path = get_path(path)
    fringe = list()
    mc = deepcopy(maze)
    fire_spot = node(1, len(maze) - 2)

    mc[fire_spot.loc] = 5
    fringe.append(fire_spot)
    for cur_node in path:
        if mc[cur_node[0], cur_node[1]] == 5:
            return [], mc
        mc[cur_node[0], cur_node[1]] = 3
        fringe, mc = update_map(mc, fringe, p)
        # draw.draw_maze(mc)
        # if cur_node in fringe:
        #     return [], mc
    return path, mc
    # while path_len > 0:
    #     path_len = path_len - 1
    #     fringe = update_fringe(mc, fringe, p)
    #     for fire_spot in fringe:
    #         if fire_spot.loc in path:
    #             return [], mc


def update_map(maze, fringe_now, p):
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
    return fringe_now, maze
