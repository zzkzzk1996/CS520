import heapq


class node():
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.loc = (self.i, self.j)


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
    def push(self, priority, maze, length, route=None):
        heapq.heappush(self.queue, (priority, self._index, maze, length))
        self._index += 1

    def pop(self):
        return heapq.heappop(self.queue)[-1]
