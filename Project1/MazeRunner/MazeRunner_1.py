import Maze
import Search
import copy
import time

maze_generater = Maze.Maze_generater()
i = 0
B = 0
D = 0
AE = 0

# print(time_start)


while B == 0:
    # print("<", 30*"=", ">", file=a.log)
    # i = i + 1
    maze = maze_generater(500, 0.35)
    maze_copy = copy.deepcopy(maze)
    BP = Search.BFS(maze_copy)
    B = len(BP)
    maze_copy = copy.deepcopy(maze)
    DP = Search.DFS(maze_copy)
    D = len(DP)
    maze_copy = copy.deepcopy(maze)
    time_start_1 = time.time()
    AEP = Search.A_star_Euc(maze_copy)
    AE = len(AEP)
    time_AE = time.time() - time_start_1
    maze_copy = copy.deepcopy(maze)
    time_start_2 = time.time()
    AMP = Search.A_star_Man(maze_copy)
    AM = len(AMP)
    time_AM = time.time() - time_start_2
    if len(DP) > 0:
        print(D)
        print(B)
        print(AE)
        print(AM)
        print("AE: ", time_AE)
        print("AM: ", time_AM)
    # print(i)
# print(maze)
# print(D, '/t', DP)
# print(B, '/t', BP)
# print(AE, '/t', AEP)


# while True:
#
#     maze = maze_generater(1000, 0.35)
#     maze_copy = copy.deepcopy(maze)
#     BP = Search.BFS(maze_copy)
#     B = len(BP)
#     maze_copy = copy.deepcopy(maze)
#     DP = Search.DFS(maze_copy)
#     D = len(DP)
#     for i in range(B - 1):
#         j = i + 1
#         d = abs(BP[j][0] - BP[i][0]) + abs(BP[j][1] - BP[i][1])
#         if d > 1:
#             print(maze)
#             print(BP)
#             print(BP[i], BP[j])
#             break
#         else:
#             print('right')
