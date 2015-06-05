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
    vocabSet = set([])
    for doc in data:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

def SetOfWords2Vec(vocabList, inputSet):
    retVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            retVec[vocabList.index(word)] = 1
        else:
            print 'the word: %s is not in my vocabulary!' % word
    return retVec

def TrainNB(trainMatrix, trainClass):
    numTrain = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainClass) / float(numTrain)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrain):
        if trainClass[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


def ClassifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


mySent = ' This book is the best book on Python or M.L. I have ever laid eyes upon'
regEx = re.compile('\\W*')
tokens = regEx.split(mySent)



postlist, classlist = LoadDataSet()
vocabList = CreateVocabList(postlist)
trainMat = []
for post in postlist:
    trainMat.append(SetOfWords2Vec(vocabList, post))
p0v, p1v, pab = TrainNB(trainMat, classlist)

test = ['love', 'my', 'dalmation']
thisdoc = array(SetOfWords2Vec(vocabList, test))
print test, 'classified as: ', ClassifyNB(thisdoc, p0v, p1v, pab)
test = ['stupid', 'garbage']
thisdoc = array(SetOfWords2Vec(vocabList, test))
print test, 'classified as: ', ClassifyNB(thisdoc, p0v, p1v, pab)

#print SetOfWords2Vec(vocabList, postlist[3])

