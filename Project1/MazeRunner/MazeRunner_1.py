import Maze
import Search
import copy

maze_generater = Maze.Maze_generater()
maze = maze_generater(5, 0.4)
print(maze)
maze_copy = copy.deepcopy(maze)
print(Search.DFS(maze_copy))
print("hello")
maze_copy = copy.deepcopy(maze)
print(Search.BFS(maze_copy))
