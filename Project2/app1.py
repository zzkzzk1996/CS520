# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 18:14
# @Author  : Ziqi Wang
# @FileName: app1.py
# @Email: zw280@scarletmail.rutgers.edu
from app import app

import multiprocessing

if __name__ == '__main__':
    p1 = multiprocessing.Process(target=app, args=(16, 16, 259))
    p2 = multiprocessing.Process(target=app, args=(25, 25, 324))
    p1.start()
    p2.start()
    # p1.join()
    # p2.join()