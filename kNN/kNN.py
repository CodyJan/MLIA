#-*- coding:utf8 -*-

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

'''
kNN算法的特点是,不用训练,是直接与样本计算距离,根据距离最近的k个样本的分类,来决定最后的分类
缺点是,当样本较大,计算量就较大,
另外会受样本分布的影响
'''

def CreateDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0.0,0.0], [0.0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels

def Classify0(inx, dataset, labels, k):
    rows = dataset.shape[0]
    diffmat = tile(inx, (rows,1)) - dataset
    distances = ((diffmat**2).sum(axis=1))**0.5
    sortedindex = distances.argsort()
    classcount = {}
    for i in range(k):
        votelabel = labels[sortedindex[i]]
        classcount[votelabel] = classcount.get(votelabel, 0) + 1
    sortedclasscount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]


def AutoNorm(data):
    '将每项特征规范化到0~1之间'

    minval = data.min(0)
    maxval = data.max(0)
    range = maxval - minval
    normdata = zeros(shape(data))
    m = data.shape[0]
    normdata = data - tile(minval, (m,1))
    normdata = normdata / tile(range, (m,1))
    return normdata, range, minval

def File2Matrix(filename):
    'read matrix from a file'

    fp = open(filename)
    alllines = fp.readlines()
    numsline = len(alllines)
    retmat = zeros((numsline,3))
    labels = []
    index = 0

    str2labs = {"didntLike":0, "smallDoses":1, "largeDoses":2}

    for line in alllines:
        # 以回车符分段
        line = line.strip()

        # 以制表符分成list
        listline = line.split('\t')

        # 前三个作为特征矩阵
        retmat[index,:] = listline[0:3]

        # 最后一个作为类别标签
        labels.append(str2labs[listline[-1]])

        index += 1
    return retmat,labels


def DatingClassTest():
    ' 部分作为交叉验证,部分作为kNN训练样本 '

    ratio = 0.2
    feats, labs = File2Matrix('datingTestSet.txt')
    normfeats, ranges, minvals = AutoNorm(feats)
    m = normfeats.shape[0]
    numTestVecs = int(m*ratio)
    errorcnt = 0.0
    for i in range(numTestVecs):
        result = Classify0(normfeats[i,:], normfeats[numTestVecs:m, :], labs[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (result, labs[i])
        if result != labs[i]:
            errorcnt += 1
    print 'the total error rate is: %f' %(errorcnt/float(numTestVecs))

#grp, labs = CreateDataSet()
#print 'class is : ', Classify0([0,0], grp, labs, 3)


#DatingClassTest()
#feats, labs = File2Matrix('datingTestSet.txt')
#normfeats, ranges, minvals = AutoNorm(feats)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(feats[:,1], feats[:,2])
#ax.scatter(normfeats[:,0], normfeats[:,1], c=15.0*array(labs))

#plt.show()
#print 'end'




# 下面是手写数字识别算法
def Img2Vec(filename):
    ' 单个手写是32x32矩阵,将其装成1x1024向量 '

    vec = zeros((1, 1024))
    fp = open(filename)
    for i in range(32):
        line = fp.readline()
        for j in range(32):
            vec[0, 32*i+j] = int(line[j])
    return vec

def HandwritingClassTest():

    # 读取训练样本
    hwlabs = []
    trainFileList = os.listdir('trainingDigits')
    m = len(trainFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        filename = trainFileList[i]
        filestr = filename.split('.')[0]
        classname = int(filestr.split('_')[0])
        hwlabs.append(classname)
        trainingMat[i,:] = Img2Vec('trainingDigits/%s' % filename)

    # 交叉验证
    testFileList = os.listdir('testDigits')
    errorcnt = 0
    mTest = len(testFileList)
    for i in range(mTest):
        filename = testFileList[i]
        filestr = filename.split('.')[0]
        classname = int(filestr.split('_')[0])
        vectest = Img2Vec('testDigits/%s' % filename)
        result = Classify0(vectest, trainingMat, hwlabs, 3)

        print 'the classifier came back with: %d, the real answer is: %d' %(result, classname)

        if (result != classname):
            errorcnt += 1.0
    print '\nthe total number of errors is: %d' % errorcnt
    print '\nthe total error rate is: %f' %(errorcnt/(float(mTest)))

HandwritingClassTest()



