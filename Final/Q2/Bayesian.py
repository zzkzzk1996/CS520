#!/usr/bin/env python
# encoding: utf-8
# @project : Final
# @author: Zekun Zhang
# @contact: zekunzhang.1996@gmail.com
# @file: Bayesian.py
# @time: 2019-12-21 10:08:13

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


def one_point(point):
    isA = pA1[tuple(point)]
    isB = pB1[tuple(point)]
    return isA, isB


def one_image(image):
    point_list = np.argwhere(image == 1)
    result = 0
    A_p = []
    B_p = []
    for point in point_list:
        tempA, tempB = one_point(point)
        A_p.append(tempA)
        B_p.append(tempB)
    finalA = 1
    finalB = 1
    for i in range(len(point_list)):
        finalA *= A_p[i]
        finalB *= B_p[i]
    print('likelihood of A:', finalA, 'likelihood of B:', finalB)
    if finalA > finalB:
        return 'Class A'
    else:
        return 'Class B'


cA = read_file(pathA)
cB = read_file(pathB)
cM = read_file(pathM)

# the possibility of every points be 1(on every matrix)
#  + 1 mean we add some bias on this dataset
pA1 = (cA.sum(axis=0) + 1) / 6
pB1 = (cB.sum(axis=0) + 1) / 6

for ii in cM:
    print('prediction:', one_image(ii))
