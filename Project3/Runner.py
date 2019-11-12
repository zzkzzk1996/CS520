#!/usr/bin/env python
# encoding: utf-8
# @project : Assignment3
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: Runner.py
# @time: 2019-11-08 16:16:35

import Map
import Stationery
import Moving
import csv

path = "/Users/zekunzhang/2019 Fall/CS520/Assignment3"
filename1 = "/Rule_1.csv"
filename2 = "/Rule_2.csv"


def write_csv(info, filename):
    csvFile = open(path + filename, "a")
    writer = csv.writer(csvFile)
    writer.writerow(info)
    csvFile.close()


def test(times):
    res1, res2 = 0, 0
    for i in range(times):
        map = Map.Map(50)
        se = Stationery.Explorer(origin_map=map)
        res1 += se.search_one()
        res2 += se.search_two()
    # print("Average Search Times: " + str(res / times))
    return res1, res2


if __name__ == '__main__':
    # for i in range(1, 1000):
    #     res1, res2 = test(i)
    #     info1 = [i, round(res1 / i)]
    #     info2 = [i, round(res2 / i)]
    #     write_csv(info1, filename1)
    #     write_csv(info2, filename2)
    #     print("Progress:{}%".format((i * 100 / 1000)), flush=True)

    # map = Map.Map(10)
    # se = Stationery.Explorer(origin_map=map)
    # res = se.search_three_with_actions()
    # print(res)
    # compare = se.search_two_with_actions()
    # print(compare)

    map = Map.Map(10)
    me = Moving.MovingExplorer(origin_map=map)
    res = me.search()
    print(res)
