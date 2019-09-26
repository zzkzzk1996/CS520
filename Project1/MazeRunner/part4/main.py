from part4 import Maze
from part4.new_search import dfs
from part4.Fire_search import walk_on_fire, simple_walk, astar_walk_on_fire
from part4.astar import manhattan_distance
from matplotlib import pyplot as plt
"""
This is the main function I use
if you want to reproduce what I've attached on report, plz run the file.
it will show you the performance of three algorithm
"""
p_list = list()
generate_maze = Maze.Maze_generater()
cur_1 = list()
cur_2 = list()
cur_3 = list()
solve_hard = 0
solve_simple = 0
over_all = 0
stack = []
solve_a = 0
possi = float(0)
while possi <= 1.0:
    p_list.append(possi)

    solve_hard = 0
    solve_simple = 0
    over_all = 0
    solve_a = 0
    while over_all < 10:
        while len(stack) == 0:
            maze = generate_maze.generate_maze(dim=20, p=0.5)
            stack, _ = dfs(maze)
        # p2, mcd = dfs(maze)
        over_all = over_all + 1
        p, mcw = walk_on_fire(maze, possi)

        p2, mca = astar_walk_on_fire(maze, manhattan_distance, possi)
        # generate_maze.draw_maze(mcw)
        p1, mcs = simple_walk(maze, possi)
        # generate_maze.draw_maze(mcs)
        if len(p) > 0:
            solve_hard = solve_hard + 1
        if len(p1) > 0:
            solve_simple = solve_simple + 1
        if len(p2) > 0:
            solve_a = solve_a + 1

    print('<===================================>')
    print('case = ', over_all)
    print('success rate of hard = ', solve_hard / over_all)
    print('success rate of simple = ', solve_simple / over_all)
    print('success rate of a_star = ', solve_a / over_all)
    print(possi)
    possi = possi + 0.1
    cur_1.append(solve_hard / over_all)
    cur_2.append(solve_simple / over_all)
    cur_3.append(solve_a / over_all)
    # generate_maze.draw_maze(mcs)
    # bp = 1
    # generate_maze.draw_maze(mcw)
    # bp = len(p)
    # print(bp)

# generate_maze.draw_maze(maze)
plt.title("dfs")
plt.plot(p_list, cur_1)
plt.show()
plt.title("baseline")
plt.plot(p_list, cur_2)
plt.show()
plt.title("a_star")
plt.plot(p_list, cur_3)
plt.show()
