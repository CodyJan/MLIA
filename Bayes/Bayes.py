#-*- coding:utf8 -*-
from numpy import *
import re

def LoadDataSet():
    strlist = ['my dog has flea problems help please',
               'maybe not take him to dog park stupid',
               'my dalmation is so cute I love him',
               'stop posting stupid worthless garbage',
               'mr licks ate my steak how to stop him',
               'quit buying worthless dog food stupid']

    postingList = []
    i = 0
    for str in strlist:
        postingList.append( str.split() )
        i += 1

    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def CreateVocabList(data):
    ' 去掉data里面所有的重复单词,形成一个列表 '

    vocabSet = set([])
    for doc in data:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

def SetOfWords2Vec(vocabList, inputSet):
    ' vocabList是词汇表,根据inputSet,形成一个词汇是否出现的列表 '

    retVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            retVec[vocabList.index(word)] = 1
        else:
            print 'the word: %s is not in my vocabulary!' % word
    return retVec

def TrainNB(trainMatrix, trainClass):
    ' 各个特征的条件概率,前提是所有特征项是彼此独立的 '

    # 样本数量
    numTrain = len(trainMatrix)

    # 词汇总数量
    numWords = len(trainMatrix[0])

    # 样本分类1的概率
    pAbusive = sum(trainClass) / float(numTrain)

    # 分类1的样本词汇索引相加得到词频表->p1Num, 词汇出现次数总和->p1Denom
    # 分类0的样本词汇索引相加得到词频表->p0Num, 词汇出现次数总和->p0Denom
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrain):
        if trainClass[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 条件概率: P(Wi | C=1) 和 P(Wi | C=0)
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def ClassifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    ' 根据log后的条件概率P(Wi | Ci),得到P(Ci | Wi),其中为简化计算和提高精度,使用了log变化 '

    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def TextParse(text):
    tokens = re.split(r'\W*', text)
    return [tok.lower() for tok in tokens if len(tok)>2]




'''
1. 先是一个较简单的实例,使用朴素贝叶斯,训练词汇是否出现和分类标签,假设各词汇之间相互独立...

postlist, classlist = LoadDataSet()
vocabList = CreateVocabList(postlist)

# 根据词汇表,将样本形成一个词汇表索引矩,表示该词汇是否出现
trainMat = []
for post in postlist:
    trainMat.append(SetOfWords2Vec(vocabList, post))

# 训练
p0v, p1v, pab = TrainNB(trainMat, classlist)

test = ['love', 'my', 'dalmation']
thisdoc = array(SetOfWords2Vec(vocabList, test))
print test, 'classified as: ', ClassifyNB(thisdoc, p0v, p1v, pab)
test = ['stupid', 'garbage']
thisdoc = array(SetOfWords2Vec(vocabList, test))
print test, 'classified as: ', ClassifyNB(thisdoc, p0v, p1v, pab)

'''





'''
2. 读取文本并切分文本
'''

# 读取文本, 全单词->fullText, 分类->classList, 文本列表->docList
docList, classList, fullText = [], [], []
for i in range(1,26):
    wordList = TextParse(open('email/spam/%d.txt' % i).read())
    docList.append(wordList)
    fullText.extend(wordList)
    classList.append(1)

    wordList = TextParse(open('email/ham/%d.txt' % i).read())
    docList.append(wordList)
    fullText.extend(wordList)
    classList.append(0)
# 词汇表
vocabList = CreateVocabList(docList)

trainingSet = range(50)
testSet = []
for i in range(10):
    randIndex = int(random.uniform(0, len(trainingSet)))
    testSet.append(trainingSet[randIndex])
    del(trainingSet[randIndex])
trainMat, trainClasses = [], []

for docIndex in trainingSet:
    trainMat.append(SetOfWords2Vec(vocabList, docList[docIndex]))
    trainClasses.append(classList[docIndex])
p0V,p1V,pSpam = TrainNB(array(trainMat), array(trainClasses))
errorCount = 0

for docIndex in testSet:
    wordVector = SetOfWords2Vec(vocabList, docList[docIndex])
    if ClassifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
        errorCount += 1
print 'the error rate is: ', float(errorCount) / len(testSet)
