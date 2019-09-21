import Maze
import Search
import copy
import time

maze_generater = Maze.Maze_generater()
B = 0
D = 0
AE = 0
time_start = time.time()
print(time_start)

while B == 0:
    maze = maze_generater(1000, 0.35)
    maze_copy = copy.deepcopy(maze)
    BP = Search.BFS(maze_copy)
    B = len(BP)
    maze_copy = copy.deepcopy(maze)
    DP = Search.DFS(maze_copy)
    D = len(DP)
    maze_copy = copy.deepcopy(maze)
    AEP = Search.A_star_Euc(maze_copy)
    AE = len(AEP)
# print(maze)
# print(D, '/t', DP)
# print(B, '/t', BP)
# print(AE, '/t', AEP)
print(D)
print(B)
print(AE)
print(time.time() - time_start)

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
