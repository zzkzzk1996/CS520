#!/usr/bin/env python
# encoding: utf-8
# @project : Final
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: KNN.py
# @time: 2019-12-21 08:46:03

import numpy as np

# pathA = 'ClassA.txt'
# pathB = 'ClassB.txt'
# pathM = 'Mystery.txt'

pathA = '/Users/zekunzhang/PycharmProjects/Final/Q2/ClassA.txt'
pathB = '/Users/zekunzhang/PycharmProjects/Final/Q2/ClassB.txt'
pathM = '/Users/zekunzhang/PycharmProjects/Final/Q2/Mystery.txt'


def read_file(path):
    class_list = []
    with open(path) as file:
        for line in file:
            newline = line.rstrip('\n').split('\t')
            if newline != ['']:
                class_list.append(newline)
    return np.array(class_list).astype('int').reshape((5, 5, 5))


cA = read_file(pathA)
cB = read_file(pathB)
cM = read_file(pathM)

train_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def KNN(data, test, k):
    resize_test = []
    for i in range(len(data)):
        resize_test.append(test)
    diffs = resize_test - data
    distances = np.square(diffs)
    distances = np.sqrt(np.sum(np.sum(distances, axis=2), axis=1))
    sorted_index = np.argsort(distances)
    top_k = sorted_index[:k]
    print(top_k)
    b_count = 0
    for j in top_k:
        b_count += train_labels[j]
    a_count = k - b_count
    if a_count == b_count:
        print('Just the same')
    else:
        # print('Prediction:', 'A' if a_count > b_count else 'B', 'Count: A:', a_count, 'B:', b_count)
        print ('A' if a_count > b_count else 'B')

for i in range(10):
    for matrix in cM:
        data = np.append(cA, cB, axis=0)
        KNN(data, matrix, i)
    print("----------")
