#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest V1 implemented in Python
"""
from collections import Counter
from random import seed, randrange, random
import math
import numpy as np
from ID3 import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine    

    
class Tree:
    def __init__(self, \
                 feaValue = None, \
                 trueBranch = None, \
                 falseBranch = None, \
                 results = None, \
                 feaIndex = -1, \
                 summary = None, \
                 data = None,\
                 feaList = None):
        self.feaValue = feaValue
        self.trueBranch = trueBranch# Left subtree
        self.falseBranch = falseBranch# Right subtree
        self.results = results
        self.feaIndex = feaIndex
        self.summary = summary
        self.data = data 
        self.feaList = feaList  

             
def randomForest(dataSet,n_estimators,min_sample_split,ratio,n_features = None):
    """
    :param dataSet: training data
    :param min_sample_split: Minimal branch sample tree
    :param n_features: Number of candidate features per split
    
    """
    forest = []
    
    if n_features==None:
        n_features = round(math.sqrt(len(dataSet[0])-1))
    
    for i in range(n_estimators):   
        idx, datas = get_bootstrap_data(dataSet,ratio,n_features)
        tree = createTree(datas,min_sample_split)
        tree.feaList= idx
        forest.append(tree) 
        
    return forest
   
def giniCnt(dataSet):
    '''
    Calculate the Gini coefficient of the data set
    
    '''
    lenDataset = len(dataSet)#数据集样本总数
    labelList = [data[-1] for data in dataSet]
    labelCounts = dict(Counter(labelList))#获取类别标签个数字典
    prob = [(float(v)/lenDataset) for v in labelCounts.values()]
    Gini = 1-sum([p*p for p in prob])
    
    return Gini


def vote(dataSet):
    '''
    Vote to choose which category to choose
    
    '''
    labelList = [data[-1] for data in dataSet]
    labelCounts = dict(Counter(labelList))
    result = sorted(labelCounts.items(),key=lambda item:item[1],reverse=True)
    
    return result[0][0]
    

def split_data(dataSet,axis,value):
    """Cart: divided into two data sets"""
    left = []
    right = []
    
    if isinstance(dataSet[0][axis],float):
        for data in dataSet:
            if data[axis] <= value:
                left.append(data)
            else:
                right.append(data)
    else:
        for data in dataSet:
            if data[axis] == value:
                left.append(data)
            else:
                right.append(data)
    
    return left, right

def get_bootstrap_data(dataSet,ratio,n_features):
    """
    Random sample
    :param ratio: Proportion of training set samples drawn
    :param n_features: Number of randomly selected features
    :param return: Random sample set
    
    """
    n_sample = round(len(dataSet)*ratio)
    n_col = len(dataSet[0])-1
    
    sample = []
    while len(sample) < n_sample:
        index = randrange(len(dataSet))
        sample.append(dataSet[index])
    
    idx = np.random.choice(n_col,n_features)
    sample_f = np.array(dataSet)[:,idx].tolist()
    for i in range(n_sample):
        sample_f[i].append(sample[i][-1])
        
    return idx, sample_f
    
  

def createTree(dataSet,min_sample_split):
    """Build a tree recursively"""

    feaLen = len(dataSet[0])-1
    dataLen = len(dataSet)
    dataGini = giniCnt(dataSet)
    bestGain = 0.0
    bestFea = None
    bestValue = None
    
    treeSummary = {'impurity': '%.3f' % dataGini, 'samples': '%d' % dataLen}
    if len(dataSet)<min_sample_split:
        return Tree(results = vote(dataSet), summary = treeSummary, data = dataSet)

    # Select the best feature among the candidate features
    for i in range(feaLen):
        feaSet = set([data[i] for data in dataSet])
        for feaType in feaSet:
            left, right = split_data(dataSet,i,feaType)
            prob = len(left)/dataLen
            gain = dataGini-prob*giniCnt(left)-(1-prob)*giniCnt(right)
            if gain > bestGain:
                bestGain = gain
                bestFea = i
                bestValue = feaType
                bestData = (left, right)
    
    
    if bestGain > 0:
        
        trueBranch = createTree(bestData[0],min_sample_split)
        falseBranch = createTree(bestData[1],min_sample_split)
        return Tree(feaValue = bestValue, \
                    trueBranch = trueBranch,\
                    falseBranch = falseBranch,\
                    feaIndex = bestFea,\
                    summary = treeSummary)
    else:
        return Tree(results = vote(dataSet), summary = treeSummary, data = dataSet)
 
    
def classify_split(data,feaList):
    
    idx = [np.array(feaList)]
    sample_f = np.array(data)[idx].tolist()
    sample_f.append(data[-1])
    
    return sample_f
    
    
    
def classify(tree,data):
    """Judging the classification result according to the tree"""
    
    if tree.results != None:
        return tree.results
        
    else:
        branch = None
        value = data[tree.feaIndex]
        if isinstance(value,float):
            if value <= tree.feaValue:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        else:
            if value == tree.feaValue:
                branch = tree.trueBranch
            else:
                 branch = tree.falseBranch
        
        return classify(branch,data)
  

def rf_classify(forest,data):
    """Random forest voting classification"""
    
    results = []
    
    for i in range(len(forest)):
        
        feaList = forest[i].feaList
        classify_split(data,feaList)
        r = classify(forest[i],data)
        results.append([r])
    
    result = vote(results)
    return result

  
def rf_score(rf,testdata):
    acc = 0
    for data in testdata: 
        if data[-1] == rf_classify(rf,data):
            acc+=1
    accr = acc/len(testdata)   
    print("Test Acc: ",accr)    
    return accr  
    
    
        
if __name__ == "__main__":
        
    wine = load_wine()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
    dataSet = [data.tolist() for data in wine.data]
    for i in range(len(dataSet)):
        dataSet[i].append(wine.target[i])

    rf = randomForest(dataSet, n_estimators=10, ratio = 1, min_sample_split=5)
    
    dataSet_t = [data.tolist() for data in Xtest]
    for i in range(len(dataSet_t)):
        dataSet_t[i].append(Ytest[i])

    rf_score(rf,dataSet_t)
    