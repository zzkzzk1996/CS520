import random


class Maze_generater:

    def __call__(self, dim, p):
        maze = [[0] * dim for i in range(dim)]
        p = p * 100
        for i in range(dim):
            for j in range(dim):
                if i == j and i == 0:
                    continue
                elif i == j and i == dim - 1:
                    continue
                else:
                    flag = random.randint(0, 100)
                    if flag <= p:
                        maze[i][j] = 1
        return maze
