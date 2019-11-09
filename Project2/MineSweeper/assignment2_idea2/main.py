import sweeper
import minemap
import numpy as np
import time

# map_generator = utils.MineMap()
# board = map_generator.board
# mines = map_generator.get_mines()
# print(board)

if __name__ == '__main__':
<<<<<<< HEAD:Project2/main.py
    # avg, avg_improve, time_con, time_con_improve = 0, 0, 0, 0
    # for i in range(100):
    #     map_generator = minemap.MineMap(x=16, y=16, p=64)
    #     board = map_generator.board
    #     mine = map_generator.mine_number
    #     sweep = sweeper.Sweeper(board, map_generator.mine_number)
    #     start = time.time()
    #     while sweep.flip():
    #         sweep.sweep_safe()
    #         while sweep.sweep_mine(): continue
    #     end = time.time()
    #     time_con += end - start
    #     t = sweep.temp / mine
    #     avg += t
    #
    #     sweep_improve = sweeper.Sweeper(board, map_generator.mine_number)
    #     start_improve = time.time()
    #     while sweep_improve.flip_by_possibility():
    #         sweep_improve.sweep_safe()
    #         while sweep_improve.sweep_mine(): continue
    #     end_improve = time.time()
    #     time_con_improve += end_improve - start_improve
    #     t_improve = sweep_improve.temp / mine
    #     avg_improve += t_improve
    #     # sweeper.draw_board()
    #     # map_generator = minemap.MineMap()
    #     # map_generator.drawboard(board)
    #     # print(sweeper.sweeper_map)
    #     # print(board)
    #     print("Progress:{}%".format(((i + 1) * 100 / 100)), flush=True)
    # avg /= 100
    # avg_improve /= 100
    # print(str(avg) + " -> " + str(time_con))
    # print(str(avg_improve) + " -> " + str(time_con_improve))
=======
    avg, avg_improve, time_con, time_con_improve = 0, 0, 0, 0
    for i in range(100):
        map_generator = minemap.MineMap(x=16, y=16, p=38)
        board = map_generator.board
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
>>>>>>> 3150100984e694563b3afd862cb6f463d3993fb5:Project2/MineSweeper/assignment2_idea2/main.py

    map_generator = minemap.MineMap(x=25, y=25)
    board = map_generator.board
    mine = map_generator.mine_number
    sweep = sweeper.Sweeper(board, map_generator.mine_number)
    while sweep.flip_by_possibility():
        sweep.sweep_safe()
        while sweep.sweep_mine(): continue
    sweep.draw_board()
    map_generator = minemap.MineMap()
    map_generator.drawboard(board)
    print(sweep.sweeper_map)
    print(board)

    # sweep.draw_board()
    # map_generator = minemap.MineMap()
    # map_generator.drawboard(board)
    # print(sweep.sweeper_map)
    # print(board)
