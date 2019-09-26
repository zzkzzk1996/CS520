#!/usr/bin/env python
# encoding: utf-8
# @project : CS520
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: MyPriorityQueue.py
# @time: 2019-09-24 10:35:10

import heapq


class MyPriorityQueue:

    def __init__(self):
        self._index = 0
        self.queue = []

    def push(self, priority, cur, path, route):
        heapq.heappush(self.queue, (priority, self._index, cur, path, route))
        self._index -= 1

    def pop(self):
        return heapq.heappop(self.queue)

    def qsize(self):
        return len(self.queue)


# Override pq for hill climb
class HillClimbPQ(MyPriorityQueue):
    def push(self, priority, maze, length=None, route=None):
        heapq.heappush(self.queue, (priority, self._index, maze))
        self._index -= 1

    def pop(self):
        return heapq.heappop(self.queue)
