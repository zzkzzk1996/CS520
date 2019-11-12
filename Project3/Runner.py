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
filename1 = "/Rule_Action50_1.csv"
filename2 = "/Rule_Action50_2.csv"
filename3 = "/Rule_Action50_3.csv"

filename = "/Moving_Rule_Action10_3.csv"


def write_csv(info, filename):
    csvFile = open(path + filename, "a")
    writer = csv.writer(csvFile)
    writer.writerow(info)
    csvFile.close()


def test(times):
    action1, search1, action2, search2, action3, search3 = 0, 0, 0, 0, 0, 0
    for i in range(times):
        map = Map.Map(50)
        se = Stationery.Explorer(origin_map=map)

        a1, s1 = se.search_one_with_actions()
        a2, s2 = se.search_two_with_actions()
        a3, s3 = se.search_three_with_actions()

        action1 += a1
        action2 += a2
        action3 += a3
        search1 += s1
        search2 += s2
        search3 += s3
    # print("Average Search Times: " + str(res / times))
    return action1, action2, action3, search1, search2, search3


def test_moving(times):
    action, search = 0, 0
    for i in range(times):
        map = Map.Map(50)
        me = Moving.MovingExplorer(origin_map=map)
        a, s = me.search()
        action += a
        search += s
    # print("Average Search Times: " + str(res / times))
    return action, search


if __name__ == '__main__':
    # for i in range(1, 200):
    #     action1, action2, action3, search1, search2, search3 = test(i)
    #     info1 = [i, round(search1 / i), round(action1 / i)]
    #     info2 = [i, round(search2 / i), round(action2 / i)]
    #     info3 = [i, round(search3 / i), round(action3 / i)]
    #     write_csv(info1, filename1)
    #     write_csv(info2, filename2)
    #     write_csv(info3, filename3)
    #     print("Progress:{}%".format((i * 100 / 200)), flush=True)

    # map = Map.Map(10)
    # se = Stationery.Explorer(origin_map=map)
    # res = se.search_three_with_actions()
    # print(res)
    # compare = se.search_two_with_actions()
    # print(compare)

    # map = Map.Map(50)
    # me = Moving.MovingExplorer(origin_map=map)
    # res = me.search()
    # print(res)

    for i in range(1, 200):
        action, search = test_moving(i)
        info = [i, round(search / i), round(action / i)]
        write_csv(info, filename)
        print("Progress:{}%".format((i * 100 / 200)), flush=True)
