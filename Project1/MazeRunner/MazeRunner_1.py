import Maze
from new_search import bfs, dfs
from Fire_search import walk_on_fire, simple_walk, astar_walk_on_fire
from astar import manhattan_distance

generate_maze = Maze.Maze_generater()

solve_hard = 0
solve_simple = 0
over_all = 0
stack = []

while len(stack) == 0:
    maze = generate_maze.generate_maze(dim=128, p=0.3)
    stack, _ = dfs(maze)

while over_all < 1000:

    # p2, mcd = dfs(maze)
    over_all = over_all + 1
    # p, mcw = walk_on_fire(maze, 0.7)
    p, mcw = astar_walk_on_fire(maze, manhattan_distance, 0.7)
    # generate_maze.draw_maze(mcw)
    p1, mcs = simple_walk(maze, 0.7)
    # generate_maze.draw_maze(mcs)
    if len(p) > 0:
        solve_hard = solve_hard + 1
    if len(p1) > 0:
        solve_simple = solve_simple + 1

    print('<===================================>')
    print('case = ', over_all)
    print('success rate of hard = ', solve_hard / over_all)
    print('success rate of simple = ', solve_simple / over_all)
    # generate_maze.draw_maze(mcs)
    # bp = 1
    # generate_maze.draw_maze(mcw)
    # bp = len(p)
    # print(bp)

# generate_maze.draw_maze(mcd)
print(maze)
