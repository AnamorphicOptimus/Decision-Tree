#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How to plot out a tree
"""

import matplotlib.pyplot as plt

decisionNode=dict(boxstyle="sawtooth",fc="0.8")
# Define the descriptive attributes of the leaf nodes of the decision tree
leafNode=dict(boxstyle="round4",fc="0.8")
# Define the arrow properties of the decision tree
arrow_args=dict(arrowstyle="<-")

# Use text annotations to draw tree nodes
# nodeTxt is the text to be displayed, centerPt is the center point of the text, the point where the arrow is located, and parentPt is the point pointing to the text
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
                            xytext=centerPt,textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

# create the plot
def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot().ax1=plt.subplot(111,frameon=False)
    plotNode('Decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('Leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
    
# Get the number of leaf nodes
def getNumLeafs(myTree):
    # Define the number of leaf nodes
    numLeaf=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeaf+=getNumLeafs(secondDict[key])
        else:
            numLeaf+=1
    return numLeaf

# Get the depth of tree
def getTreeDepth(myTree):
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

# Pre-store tree information
def retrieveTree(i):
    listOfTree=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
        {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTree[i]

# Draw middle text
def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

# Draw a decision tree
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD

def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

