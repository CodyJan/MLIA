#-*- coding:utf8 -*-
from numpy import *


'''
梯度法
'''


def LoadDataset():
    data = [];
    label = [];

    fr = open('testSet.txt')
    for line in fr.readlines():
        arr = line.strip().split()
        data.append([1.0, float(arr[0]), float(arr[1])])
        label.append(int(arr[2]))
    return data, label

def Sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def GradAscend(data, label):
    dmat = mat(data)
    lmat = mat(label).transpose()

    rows,cols = shape(dmat);
    alpha = 0.001
    iter_max = 500
    weights = ones((cols, 1))

    for k in range(iter_max):
        h = Sigmoid(dmat*weights)
        err = lmat - h
        weights = weights + alpha * dmat.transpose()*err;
    return weights

data, label = LoadDataset()
print 'weights is: ', GradAscend(data, label)