#-*- coding:utf8 -*-
from math import log
import operator
import treePlotter


def CalcEntropy(data):
    nums = len(data)
    labdict = {}
    for feat in data:
        lab = feat[-1]
        if lab not in labdict.keys():
            labdict[lab] = 0
        labdict[lab] += 1

    entropy = 0.0
    for key in labdict:
        prob = float(labdict[key]) / nums
        entropy -= prob * log(prob, 2)

    return entropy

def CreateData():
    data = [[1,1,'yes'], [1,1,'yes'], [1,0,'no'],[0,1,'no'],[0,1,'no']]
    #data = [[1,1,'maybe'], [1,1,'yes'], [1,0,'no'], [0,1,'no'], [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return data, labels

def SplitData(data, axis, value):
    ' 从data每项等于value的子项,并剔除该子项,相当于挑出所有第axis列等于value的行,再去掉第axis列,形成矩阵 '

    retdata = []
    for feat in data:
        if feat[axis] == value:
            reducedfeat = feat[:axis]
            reducedfeat.extend(feat[axis+1:])
            retdata.append(reducedfeat)
    return retdata

def ChooseFeatureToSplit(data):
    ' 分别计算每个特征的信息增益,选择对应最大信息增益的特征作为分裂点 '

    num_sample = len(data)
    num_feats = len(data[0]) - 1
    base_entropy = CalcEntropy(data)
    best_info_gain = 0.0
    best_feat = -1

    for i in range(num_feats):
        # 返回data每项的第i项,相当于取出每行的第i列
        feat_list = [example[i] for example in data]

        # 该列共有几种取值
        unique_vals = set(feat_list)
        new_entropy = 0.0

        # 将所有取值的熵,按出现几率加权在一起
        for val in unique_vals:
            subdata = SplitData(data, i, val)
            prob = len(subdata) / float(num_sample)
            new_entropy += prob * CalcEntropy(subdata)

        # 从而获得该特征的信息增益
        info_gain = base_entropy - new_entropy

        # 记录最大信息增益及其对应的特征,列索引
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = i
    return best_feat


def MajorityCnt(class_list):
    ' 返回某数据中,出现频率最高项的次数 '

    class_dict = {}
    for vote in class_list:
        if vote not in class_dict.keys():
            class_dict[vote] = 0
        class_dict[vote] += 1
    sorted_class_dict = sorted(class_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_dict[0][0]

def CreateTree(data, labs):
    ' 递归地创建决策树 '

    # 取出类别形成list,相当于取出最后一列
    class_list = [exmaple[-1] for exmaple in data]

    # 只有一种分类的情况下,停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 处理完所有特征,仍然不能将其划分成仅包含唯一类别的分组,所以挑选出频率最高的分类返回
    if len(data[0]) == 1:
        return MajorityCnt(class_list)

    # 寻找最大信息增益的特征
    best_feat_idx = ChooseFeatureToSplit(data)
    best_feat_lab = labs[best_feat_idx]

    mytree = {best_feat_lab:{}}
    del(labs[best_feat_idx])

    # 最大信息增益列
    feat_val = [example[best_feat_idx] for example in data]
    unique_val = set(feat_val)

    for val in unique_val:
        sublab = labs[:]
        mytree[best_feat_lab][val] = CreateTree(SplitData(data, best_feat_idx, val), sublab)
    return mytree


def classify(intree, feats, tests):
    first = intree.keys()[0]
    second = intree[first]
    featidx = feats.index(first)

    for key in second.keys():
        if tests[featidx] == key:
            if type(second[key]).__name__ == 'dict':
                classlab = classify(second[key], feats, tests)
            else:
                classlab = second[key]
    return classlab


data, labs = CreateData()
mytree = treePlotter.GrabTree('store.txt')
print classify(mytree, labs, [1,0])
print classify(mytree, labs, [1,1])
