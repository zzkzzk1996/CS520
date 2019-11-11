#!/usr/bin/env python
# encoding: utf-8
# @project : Project3
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: Runner.py
# @time: 2019-11-08 16:16:35

import Map
import Stationery
if __name__ == '__main__':
    map = Map.Map(10)
    map.printer()
    run = Stationery.Explorer(map)
    run.search_one()