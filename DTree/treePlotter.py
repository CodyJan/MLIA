#-*- coding:utf8 -*-
import matplotlib.pyplot as plt


decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def PlotNode(nodeTxt, centerPt, parentPt, nodeType):
    CreatePlot.ax.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def CreatePlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    CreatePlot.ax = plt.subplot(111, frameon=False)
    PlotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    PlotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def GetNumLeafs(myTree):
    numLeafs = 0
    first = myTree.keys()[0]
    second = myTree[first]
    for key in second.keys():
        if type(second[key]).__name__=='dict':
            numLeafs += GetNumLeafs(second[key])
        else:
            numLeafs += 1
    return numLeafs

def GetTreeDepth(myTree):
    maxdepth = 0
    first = myTree.keys()[0]
    second = myTree[first]
    for key in second.keys():
        if type(second[key]).__name__ == 'dict':
            thisdepth = 1 + GetTreeDepth(second[key])
        else:
            thisdepth = 1
        if thisdepth > maxdepth:
            maxdepth = thisdepth
    return maxdepth

def RetrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}, 3: 'maybe'}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def PlotMidText(cntrPt, parentPt, txtString):
    xmid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    ymid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    CreatePlot.ax.text(xmid, ymid, txtString)

def PlotTree(mytree, parentPt, nodeTxt):
    numleafs = GetNumLeafs(mytree)
    depth = GetTreeDepth(mytree)
    first = mytree.keys()[0]
    cntrPt = (PlotTree.xOff + (1.0 + float(numleafs))/2.0/PlotTree.totalW, PlotTree.yOff)

    PlotMidText(cntrPt, parentPt, nodeTxt)
    PlotNode(first, cntrPt, parentPt, decisionNode)
    second = mytree[first]
    PlotTree.yOff = PlotTree.yOff - 1.0/PlotTree.totalD

    for key in second.keys():
        if type(second[key]).__name__ == 'dict':
            PlotTree(second[key], cntrPt, str(key))
        else:
            PlotTree.xOff = PlotTree.xOff + 1.0/PlotTree.totalW
            PlotNode(second[key], (PlotTree.xOff, PlotTree.yOff), cntrPt, leafNode)
            PlotMidText((PlotTree.xOff, PlotTree.yOff), cntrPt, str(key))
    PlotTree.yOff = PlotTree.yOff + 1.0/PlotTree.totalD

def CreatePlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    CreatePlot.ax = plt.subplot(111, frameon=False, **axprops)
    PlotTree.totalW = float(GetNumLeafs(inTree))
    PlotTree.totalD = float(GetTreeDepth(inTree))
    PlotTree.xOff = -0.5/PlotTree.totalW
    PlotTree.yOff = 1.0
    PlotTree(inTree, (0.5,1.0), '')
    plt.show()


def StoreTree(intree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(intree, fw)
    fw.close()

def GrabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)



tree = RetrieveTree(2)
CreatePlot(tree)
print 'max leaf is: ', GetNumLeafs(tree)
print 'max depth is: ', GetTreeDepth(tree)
