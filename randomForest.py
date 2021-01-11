#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest implemented in Python
"""
from collections import Counter
from random import seed, randrange, random
import math
from sklearn.datasets import load_wine
from ID3 import train_test_split_data

            
class Tree:
    def __init__(self, \
                 feaValue = None, \
                 trueBranch = None, \
                 falseBranch = None, \
                 results = None, \
                 feaIndex = -1, \
                 summary = None, \
                 data = None):
        self.feaValue = feaValue
        self.trueBranch = trueBranch# Left subtree
        self.falseBranch = falseBranch# Right subtree
        self.results = results
        self.feaIndex = feaIndex
        self.summary = summary
        self.data = data

                
def randomForest(dataSet,n_estimators,min_sample_split,n_features = None):
    """
    :param dataSet: training data
    :param min_sample_split: Minimal branch sample tree
    :param n_features: Number of candidate features per split
    
    """
    forest = []
    
    if n_features==None:
        n_features = round(math.sqrt(len(dataSet[0])-1))
    
    for i in range(n_estimators):
        
        datas = get_bootstrap_data(dataSet,n_features)
        tree = createTree(datas,min_sample_split,n_features)
        forest.append(tree)
    
    return forest
    
        
def giniCnt(dataSet):
    '''
    Calculate the Gini coefficient of the data set
    
    '''
    lenDataset = len(dataSet)
    labelList = [data[-1] for data in dataSet]
    labelCounts = dict(Counter(labelList))
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

def get_bootstrap_data(dataSet,ratio):
    """
    Random sample
    :param ratio: Proportion of training set samples drawn
    :param n_features: Number of randomly selected features
    :param return: Random sample set
    
    """
    n_sample = round(len(dataSet)*ratio)
    
    sample = []
    while len(sample) < n_sample:
        index = randrange(len(dataSet))
        sample.append(dataSet[index])
        
    return sample
    
  

def createTree(dataSet,min_sample_split,n_features):
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
    
    # Randomly select candidate features
    feaList = []
    while len(feaList)<feaLen:
        index = randrange(feaLen)
        if index not in feaList:
            feaList.append(index)
    
    # Select the best feature among the candidate features
    for i in feaList:
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
        
        trueBranch = createTree(bestData[0],min_sample_split,n_features)
        falseBranch = createTree(bestData[1],min_sample_split,n_features)
        return Tree(feaValue = bestValue, \
                    trueBranch = trueBranch,\
                    falseBranch = falseBranch,\
                    feaIndex = bestFea,\
                    summary = treeSummary)
    else:
        return Tree(results = vote(dataSet), summary = treeSummary, data = dataSet)
 
       
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
    print("Test acc: ",accr)    
    return accr  
    
def rf_score_t(rf,testdata):
    acc = 0
    for data in testdata: 
        if data[-1] == rf_classify(rf,data):
            acc+=1
    accr = acc/len(testdata)   
    print("Train Acc: ",accr)    
    return accr    
        
if __name__ == "__main__":
    
    # Get wine dataset
    import time
    time_start = time.time()
    wine = load_wine()
    dataSet = [data.tolist() for data in wine.data]
    for i in range(len(dataSet)):
        dataSet[i].append(wine.target[i])
    train_dataSet, test_dataSet = train_test_split_data(dataSet,0.7)  

    # Train our forest 
    rf = randomForest(train_dataSet,n_estimators=100,min_sample_split=5)
    rf_score_t(rf,train_dataSet)
    rf_score(rf,test_dataSet)

    time_end = time.time()  
    time_c= time_end - time_start   
    print('time cost', time_c, 's')


