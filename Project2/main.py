import sweeper
import minemap
import numpy as np
import time

# map_generator = utils.MineMap()
# board = map_generator.board
# mines = map_generator.get_mines()
# print(board)

if __name__ == '__main__':
    avg, avg_improve, time_con, time_con_improve = 0, 0, 0, 0
    for i in range(100):
        map_generator = minemap.MineMap(x=16, y=16)
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
        mine = map_generator.mine_number
        sweep = sweeper.Sweeper(board, map_generator.mine_number)
        start = time.time()
        while sweep.flip():
            sweep.sweep_safe()
            while sweep.sweep_mine(): continue
        end = time.time()
        time_con += end - start
        t = sweep.temp / mine
        avg += t

        sweep_improve = sweeper.Sweeper(board, map_generator.mine_number)
        start_improve = time.time()
        while sweep_improve.flip_by_possibility():
            sweep_improve.sweep_safe()
            while sweep_improve.sweep_mine(): continue
        end_improve = time.time()
        time_con_improve += end_improve - start_improve
        t_improve = sweep_improve.temp / mine
        avg_improve += t_improve
        # sweeper.draw_board()
        # map_generator = minemap.MineMap()
        # map_generator.drawboard(board)
        # print(sweeper.sweeper_map)
        # print(board)
    avg /= 100
    avg_improve /= 100
    print(str(avg) + " -> " + str(time_con))
    print(str(avg_improve) + " -> " + str(time_con_improve))

<<<<<<< HEAD
    # sweeper.draw_board()
    # map_generator = utils.MineMap()
    # map_generator.drawboard(board)
    # print(sweeper.sweeper_map)
    # print(board)

    num = utils.exploded_mine(sweeper.sweeper_map)
    mine = map_generator.mine_number

    print(num, mine)


=======
    # map_generator = minemap.MineMap(x=16, y=16)
    # board = map_generator.board
    # # board = np.array([[-1, 1, 1, -1, -1, 2, 1, 1, 1, 1, ],
    # #                   [1, 2, 2, 3, 3, -1, 1, 2, -1, 2, ],
    # #                   [1, 2, -1, 1, 1, 1, 2, 3, -1, 2, ],
    # #                   [-1, 2, 1, 1, 0, 0, 1, -1, 2, 1, ],
    # #                   [1, 1, 1, 2, 2, 1, 1, 1, 1, 0, ],
    # #                   [0, 0, 2, -1, -1, 1, 0, 0, 0, 0, ],
    # #                   [0, 0, 2, -1, 3, 2, 1, 2, 1, 1, ],
    # #                   [0, 0, 2, 2, 2, 1, -1, 3, -1, 1, ],
    # #                   [1, 1, 1, -1, 3, 3, 3, -1, 3, 2, ],
    # #                   [-1, 1, 1, 2, -1, -1, 2, 1, 2, -1, ]])
    # # mine_number = 0
    # # row, column = np.shape(board)
    # # for i in range(row):
    # #     for j in range(column):
    # #         if board[i][j] == -1:
    # #             mine_number += 1
    # mine = map_generator.mine_number
    # sweep = sweeper.Sweeper(board, map_generator.mine_number)
    # while sweep.flip_by_possibility():
    #     sweep.sweep_safe()
    #     while sweep.sweep_mine(): continue
    # sweep.draw_board()
    # map_generator = minemap.MineMap()
    # map_generator.drawboard(board)
    # print(sweep.sweeper_map)
    # print(board)
>>>>>>> 6be4fd197bec23a058ff2f8919ef82401b2be790
