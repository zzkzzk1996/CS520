import utils
import numpy as np

# map_generator = utils.MineMap()
# board = map_generator.board
# mines = map_generator.get_mines()
# print(board)

if __name__ == '__main__':
    map_generator = utils.MineMap(x=20, y=20)
    board = map_generator.board
    # board = np.array([[-1, 1, 1, -1, -1, 2, 1, 1, 1, 1, ],
    #                   [1, 2, 2, 3, 3, -1, 1, 2, -1, 2, ],
    #                   [1, 2, -1, 1, 1, 1, 2, 3, -1, 2, ],
    #                   [-1, 2, 1, 1, 0, 0, 1, -1, 2, 1, ],
    #                   [1, 1, 1, 2, 2, 1, 1, 1, 1, 0, ],
    #                   [0, 0, 2, -1, -1, 1, 0, 0, 0, 0, ],
    #                   [0, 0, 2, -1, 3, 2, 1, 2, 1, 1, ],
    #                   [0, 0, 2, 2, 2, 1, -1, 3, -1, 1, ],
    #                   [1, 1, 1, -1, 3, 3, 3, -1, 3, 2, ],
    #                   [-1, 1, 1, 2, -1, -1, 2, 1, 2, -1, ]])
    # mine_number = 0
    # row, column = np.shape(board)
    # for i in range(row):
    #     for j in range(column):
    #         if board[i][j] == -1:
    #             mine_number += 1

    sweeper = utils.Sweeper(board, map_generator.mine_number)
    while sweeper.flip():
        sweeper.sweep_safe()
        while sweeper.sweep_mine(): continue

    sweeper.draw_board()
    map_generator = utils.MineMap()
    map_generator.drawboard(board)
    print(sweeper.sweeper_map)
    print(board)
