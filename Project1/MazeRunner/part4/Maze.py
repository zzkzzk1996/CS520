import random
import numpy as np
import matplotlib.pyplot as plt
import pylab


class Maze_generater:
    def __init__(self):
        self.maze = np.zeros((int(10 + 2), int(10 + 2)), dtype=int)

    def generate_maze(self, dim, p):
        maze = np.zeros((int(dim + 2), int(dim + 2)), dtype=int)
        maze[0, :] = maze[:, 0] = maze[-1, :] = maze[:, -1] = 2
        p = p * 100
        a, b = 1, 1
        for i in range(dim * dim):
            rand = random.randint(0, 100)
            if rand <= p:
                a = i // dim + 1
                b = i % dim + 1
                maze[a, b] = 2
            else:
                maze[a, b] = 0
        maze[1, 1] = maze[dim, dim] = 0
        # maze = [[0] * dim for i in range(dim)]
        # p = p * 100
        # for i in range(dim):
        #     for j in range(dim):
        #             continue
        #         if i == j and i == 0:
        #         elif i == j and i == dim - 1:
        #             continue
        #         else:
        #             flag = random.randint(0, 100)
        #             if flag <= p:
        #                 maze[i][j] = 1
        self.maze = maze
        return maze

    def draw_maze(self, maze=None):
        if maze is not None:
            drawMap = maze
        else:
            drawMap = self.maze
        plt.figure()
        plt.pcolor(drawMap[::-1], edgecolors='black', cmap='Blues', linewidths=2)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
        pylab.show()
