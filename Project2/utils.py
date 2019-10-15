import numpy as np
# import random

import matplotlib.pyplot as plt

dir_arr = ([0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1])


class MineMap:
    """
    class for generating mine map
    """

    def __init__(self, x=10, y=10, p=0.2):
        self.x = 10
        self.y = 10
        self.mine_number = p * x * y if 1 > p > 0 else p
        self.generate_board()

    def generate_map(self):
        """
        :return: the mine map
        generate mine map with given possibility of mine or given number of mines
        """
        x, y, mine_number = self.x, self.y, int(self.mine_number)
        tmp_arr = np.zeros(x * y)
        mines = np.random.choice(range(x * y), size=mine_number, replace=False)

        for mine in mines:
            tmp_arr[mine] = -1

        map = tmp_arr.reshape(x, y)
        return map

    def generate_tips(self, board):
        """
        :param board: the mine map
        generate the mine tips for player
        :return the tipped mine map
        """
        x, y = board.shape[0], board.shape[1]
        bigboard = np.zeros((x + 2, y + 2))  # generate a bigger map to avoid complex logical condition determination
        bigboard[1:x + 1, 1:y + 1] = board
        mine_positions = np.argwhere(bigboard == -1)
        # mark the mine's adjacent position to 1, 2, 3 or so
        for i, j in mine_positions:
            mine = (i, j)
            for dir in dir_arr:
                m = mine[0] + dir[0]
                n = mine[1] + dir[1]
                if bigboard[m, n] == -1:
                    continue
                else:
                    bigboard[m, n] += 1

        board = bigboard[1:x + 1, 1:y + 1]
        return board

    def generate_board(self):
        board = self.generate_map()
        board = self.generate_tips(board)
        self.board = board

    def drawboard(self, originalboard=None):
        if originalboard is None:
            board = self.board.copy()
        else:
            board = originalboard.copy()
        board[board == -1] = -5
        board[board == -2] = -12
        if (board != -2).all():
            board[board == -1] = -2
            # board[board>0]=11
            board[board == 0] = 0

        plt.figure(figsize=(5, 5))
        plt.pcolor(-board[::-1], edgecolors='black', cmap='bwr', linewidths=2)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()


# ms = MineMap(p=0.14)
# print(ms.board)
# ms.drawboard()
class Sweeper:
    """
    this class is to generate the sweeper's map and the search algorithm
    """

    def __init__(self, board=None, mine_number=0):
        self.row, self.col = np.shape(board)
        self.mine_map = board
        self.sweeper_map = np.full((self.row, self.col), -1, dtype=int)
        self.mine_number = mine_number
        self.start_point = [0, 0]

    def sweep_safe(self):
        for i in range(self.start_point[0], self.row):
            for j in range(self.col):
                current_value = self.mine_map[i][j]
                self.get_neighbor(current_value=current_value, current_location=(i, j))

    def get_neighbor(self, current_value, current_location=(0, 0)):
        i, j = current_location[0], current_location[1]
        if current_value == 0:
            for dir in dir_arr:
                point = [i + dir[0] if 0 <= i + dir[0] < self.row else -1,
                         j + dir[1] if 0 <= j + dir[1] < self.col else -1]
                if point[0] == -1 or point[1] == -1:
                    continue
                self.sweeper_map[point[0]][point[1]] = self.mine_map[point[0]][point[1]] if self.sweeper_map[point[0]][
                                                                                                point[1]] == -1 else \
                    self.sweeper_map[point[0]][point[1]]
            return 0
        elif current_value > 0:
            num_mines, num_unknown = 0, 0
            for dir in dir_arr:
                point = [i + dir[0] if 0 <= i + dir[0] < self.row else -1,
                         j + dir[1] if 0 <= j + dir[1] < self.col else -1]
                if point[0] == -1 or point[1] == -1:
                    continue
                if self.sweeper_map[point[0]][point[1]] < -1:
                    num_mines += 1
                elif self.sweeper_map[point[0]][point[1]] == -1:
                    num_unknown += 1
            if current_value == num_unknown + num_mines:
                current_value -= num_mines
                if current_value == 0:
                    self.get_neighbor(current_value=current_value, current_location=(i, j))
                else:
                    return current_value
            else:
                return -1

    def sweep_mine(self):
        check = False
        for i in range(self.start_point[0], self.row):
            for j in range(self.col):
                current_value = self.sweeper_map[i][j]
                if current_value > 0:
                    if self.get_neighbor(current_value=current_value, current_location=(i, j)) != -1:
                        if self.get_neighbor(current_value=current_value, current_location=(i, j)) == current_value:
                            self.set_mines(current_location=(i, j))
                            check = True
        return check

    def set_mines(self, current_location=(0, 0)):
        i, j = current_location[0], current_location[1]
        for dir in dir_arr:
            point = [i + dir[0] if 0 <= i + dir[0] < self.row else -1,
                     j + dir[1] if 0 <= j + dir[1] < self.col else -1]
            if point[0] == -1 or point[1] == -1:
                continue
            if self.sweeper_map[point[0]][point[1]] == -1:
                self.sweeper_map[point[0]][point[1]], self.mine_number = -2, self.mine_number - 1

    def flip(self):
        if self.mine_number == 0:
            return False
        for i in range(self.row):
            for j in range(self.col):
                if self.sweeper_map[i][j] == -1:
                    if self.mine_map[i][j] == -1:
                        self.sweeper_map[i][j], self.mine_number = -3, self.mine_number - 1
                    else:
                        self.sweeper_map[i][j] = self.mine_map[i][j]
                    return True

    def draw_board(self):
        sweeper_map = self.sweeper_map.copy()
        sweeper_map[sweeper_map == -2], sweeper_map[sweeper_map == -3] = -12, -12
        if (sweeper_map != -2).all():
            sweeper_map[sweeper_map == 0] = 0

        plt.figure(figsize=(5, 5))
        plt.pcolor(-sweeper_map[::-1], edgecolors='black', cmap='bwr', linewidths=2)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
