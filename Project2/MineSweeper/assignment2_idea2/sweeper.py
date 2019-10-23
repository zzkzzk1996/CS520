import numpy as np
import matplotlib.pyplot as plt

dir_arr = ([0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1])


class Sweeper:
    """
    this class is to generate the sweeper's map and the search algorithm
    """

    def __init__(self, board=None, mine_number=0):
        self.temp = 0
        self.row, self.col = np.shape(board)
        self.mine_map = board
        self.sweeper_map = np.full((self.row, self.col), -1, dtype=int)
        self.mine_number = mine_number
        self.start_point = [0, 0]
        self.possibility_map = np.full((self.row, self.col), 0, dtype=int)

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
                        # check_map = self.sweeper_map
                        if self.get_neighbor(current_value=current_value, current_location=(i, j)) == current_value:
                            self.set_mines(current_location=(i, j))
                            # check = self.map_check(check_map)
                            check = True
        return check

    def map_check(self, map):
        for i in range(self.row):
            for j in range(self.col):
                if map[i][j] != self.sweeper_map[i][j]:
                    return True
        return False

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
                        self.temp += 1
                    else:
                        self.sweeper_map[i][j] = self.mine_map[i][j]
                    return True

    def flip_by_possibility(self):
        if self.mine_number == 0:
            return False
        for i in range(self.row):
            for j in range(self.col):
                self.possibility_map[i][j] = 0
        pos_i, pos_j, max_p = 0, 0, 1
        for i in range(self.row):
            for j in range(self.col):
                self.add_possibility((i, j))
                if self.possibility_map[i][j] < max_p:
                    max_p, pos_i, pos_j = self.possibility_map[i][j], i, j
        if pos_i == 0 and pos_j == 0:
            return self.flip()
        else:
            if self.mine_map[pos_i][pos_j] == -1:
                self.sweeper_map[pos_i][pos_j], self.mine_number = -3, self.mine_number - 1
                self.temp += 1
            else:
                self.sweeper_map[pos_i][pos_j] = self.mine_map[pos_i][pos_j]
            return True

    def add_possibility(self, current_location=(0, 0)):
        i, j = current_location[0], current_location[1]
        unknown, mines = 0, self.sweeper_map[i][j]
        if mines < 1: return  # if the current position is not a tip return
        for dir in dir_arr:
            point = [i + dir[0] if 0 <= i + dir[0] < self.row else -1,
                     j + dir[1] if 0 <= j + dir[1] < self.col else -1]
            if point[0] == -1 or point[1] == -1:
                continue
            if self.sweeper_map[point[0]][point[1]] == -1:
                unknown += 1
            elif self.sweeper_map[point[0]][point[1]] < -1:
                mines -= 1
        if unknown == 0: return  # if there's no mine or all mines have been spotted
        if mines == unknown:
            self.set_mines((i, j))
        else:
            for dir in dir_arr:
                point = [i + dir[0] if 0 <= i + dir[0] < self.row else -1,
                         j + dir[1] if 0 <= j + dir[1] < self.col else -1]
                if point[0] == -1 or point[1] == -1:
                    continue
                self.possibility_map[point[0]][point[1]] += 1

    def calculate_possibility(self, current_location=(0, 0)):
        i, j, unknown, total = current_location[0], current_location[1], 0, 0
        possibility = 100 if i == 0 and j == 0 else 0
        for dir in dir_arr:
            point = [i + dir[0] if 0 <= i + dir[0] < self.row else -1,
                     j + dir[1] if 0 <= j + dir[1] < self.col else -1]

            if point[0] == -1 or point[1] == -1:
                continue
            total += 1
            if self.sweeper_map[point[0]][point[1]] == -1:
                unknown += 1
            elif self.sweeper_map[point[0]][point[1]] < -1:
                total -= 1

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
