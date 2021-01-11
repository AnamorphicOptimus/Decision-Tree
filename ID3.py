#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ID3 Implemented in Python
"""
from collections import Counter
from math import log as log
import copy
import random
from sklearn.datasets import load_wine 


def createDataset(filename,floatList):
        
    f = open(filename)
    dataSet = []
    for line in f.readlines():
        curLine = line.strip().split("\t")
        dataSet.append(curLine)

    dataSet = [data[0:] for data in dataSet]
    feature = dataSet[0][:-1]
    del(dataSet[0])
    for data in dataSet:
        for i in floatList:
            data[i] = float(data[i])    
    f.close()
    
    return feature, dataSet

def train_test_split_data(dataSet,prob):
    """split train and test"""
    n_train = int(len(dataSet)*prob)
    random.seed(1234)
    train_dataSet = random.sample(dataSet,n_train)
    for data in train_dataSet:
        dataSet.remove(data)
    test_dataSet = dataSet
    
    return train_dataSet, test_dataSet

def calEnt(dataSet):
    '''
    Compute Information entropy
    '''
    lenDataset = len(dataSet)
    labelList = [data[-1] for data in dataSet]
    labelCounts = Counter(labelList)
    prob = [(float(v)/lenDataset) for v in labelCounts.values()]
    Ent = sum([-p*log(p,2) for p in prob])
    
    return Ent


def getSubDataset(dataSet,axis,featureType):
    '''
    to calculate the information entropy of the sub-data set according to the label category
    :param axis: the data set divided by the first few features
    :param featureType: feature category label
    :param return: quite a dataset classified as featureType according to a certain feature
    '''
    subdata = []
    for data in dataSet:
        if data[axis]==featureType:
            reducedFeaData = data[:axis]
            reducedFeaData.extend(data[axis+1:])
            subdata.append(reducedFeaData)
            
    return subdata

def getContinuousSubDataset(dataSet,axis,value,direction):
    '''
    Divide the data set (continuous value); 
    continuous attributes can be used as the partition attributes of its descendant nodes;
    :param value: split point
    :param direction: <= value: 0;> value: 1;
        
    '''
    subdata = []
    if direction==0:
        
        for data in dataSet:
            if data[axis] <= value:
                subdata.append(data)
    else:
        for data in dataSet:
            if data[axis] > value:
                subdata.append(data)
    
    return subdata

def set_to_list(sets):
    r = []
    for data in sets:
        r.append(data)
    
    return r             

def chooseFeature(dataSet):
    '''
    Select the feature with the largest information gain (ID3) as the split attribute of the current node;
    To distinguish between discrete value and continuous value features;
    dataSet is the data set on the current node;
    
    '''
    feaCnt = len(dataSet[0])-1
    initEntropy = calEnt(dataSet)# current information entropy
    bestGain = 0.0
    bestFeature = -1
    labelProperty = None
    bestValue = None
    
    for i in range(feaCnt):
        featureType = set([data[i] for data in dataSet])# [1,0]or['Yes','No']
        featureType = set_to_list(featureType)      
        # If feature is continuous
        if type(featureType[0]).__name__=='float' or type(featureType[0]).__name__=='int':
            # there is n-1 spliting point
            sortFeatList = sorted(featureType)
            if len(sortFeatList) == 1:
                continue
            splitList = []
            for j in range(len(sortFeatList)-1):
                splitList.append((sortFeatList[j]+sortFeatList[j+1])/2.0)
            
            splitLen = len(splitList)
            infoGainList = []
            for t in range(splitLen):
                feaEntropy = 0.0
                value = float(splitList[t])
                subDataset0 = getContinuousSubDataset(dataSet,i,value,0)
                subDataset1 = getContinuousSubDataset(dataSet,i,value,1)
                prob0 = len(subDataset0)/float(len(dataSet))
                Ent0 = calEnt(subDataset0)
                feaEntropy += Ent0*prob0
                prob1 = len(subDataset1)/float(len(dataSet))
                Ent1 = calEnt(subDataset1)
                feaEntropy += Ent1*prob1
                infoGainList.append(initEntropy-feaEntropy)
                
            infoGainMax = max(infoGainList)              
            MaxId = infoGainList.index(max(infoGainList))    
            if infoGainMax > bestGain:
                bestGain = infoGainMax
                bestFeature = i
                bestValue = splitList[MaxId]
                labelProperty = 1
            
        #If feature is dispersed
        else:
            feaEntropy = 0.0
            for value in featureType:
                subDataset = getSubDataset(dataSet,i,value)
                subEnt = calEnt(subDataset)
                subProb = len(subDataset)/len(dataSet)
                feaEntropy += subEnt*subProb
                
                infoGain = initEntropy-feaEntropy
                if infoGain > bestGain:
                    bestGain = infoGain
                    bestFeature = i
                    labelProperty = 0
        
    return bestValue, labelProperty, bestFeature


def vote(dataSet):
    '''
    Vote to choose which category to choose
    
    '''
    labelList = [data[-1] for data in dataSet]
    labelCounts = Counter(labelList)
    result = sorted(labelCounts.items(),key=lambda item:item[1],reverse=True)
    
    return result[0][0]


def createTree(dataSet,feature,labelProperty=None):
    '''
    Building our tree recursively
    :param feature: Existing divisible properties("nosurfacing","flitters"])
    :param labelProperty: 0=Dispersed; 1=Continuous;
    :param return: {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    
    '''
    labelList = [data[-1] for data in dataSet]
    # Stop splitting condition 1: All classes in the data set are the same
    # return the category label of the node
    if len(set(labelList))==1:
        return labelList[0]
    
    # Stop splitting condition 2: No feature can be split; 
    # return to the category label with the highest frequency
    if len(dataSet[0])==1:
        return vote(dataSet)
    # Stop splitting condition 2: There is no feature to split or the number of samples in the node is less than the predetermined threshold; 
    # return the category label with the highest frequency
    if len(dataSet[0])==1 or len(dataSet)< 5 :
        return vote(dataSet)
        
    # Continue to split; choose the split feature with the best effect;
    bestValue, labelProperty, bestFeat = chooseFeature(dataSet)#最好特征axis

    if labelProperty == 0:
        
        bestFeatLabel = feature[bestFeat]
        # store the data in the tree
        tree = {bestFeatLabel:{}}
        del(feature[bestFeat])
        featValuesSet = set([example[bestFeat] for example in dataSet])   
        for value in featValuesSet:
            sublabels = feature[:]
            tree[bestFeatLabel][value] = \
                createTree(getSubDataset(dataSet,bestFeat,value),sublabels)
    else:
        bestFeatLabel = feature[bestFeat]+'<='+str(bestValue)
        tree = {bestFeatLabel:{}}
        sublabels = feature[:]
        valueLeft = "是"
        valueRight = "否"
        tree[bestFeatLabel][valueLeft] = \
            createTree(getContinuousSubDataset(dataSet,bestFeat,bestValue,0),sublabels)
        tree[bestFeatLabel][valueRight] = \
            createTree(getContinuousSubDataset(dataSet,bestFeat,bestValue,1),sublabels)
            
    return tree

def dict_to_list(dic):
    a = []
    for key in dic.keys():
        a.append(key)
    return a
        
def classify(tree,feature,testX):
    '''
    Classify recursively
    :param tree: trained tree model
    :param testdata: data to test
    :param feature: List of original feature names, e.g. ["nosurfacing","flitters"]
    
    '''
    firstfeat = dict_to_list(tree)[0]
    strfeat = copy.deepcopy(firstfeat)
    secondDict = tree[firstfeat]
    strIndex = str(firstfeat).find("<")
    if strIndex > -1: 
        firstfeat = str(firstfeat)[:strIndex]

    featureIndex = feature.index(firstfeat)
    
    for key in secondDict.keys():
        if strIndex > -1:
            # entering into the sub tree “是”
            if testX[featureIndex] <= float(str(strfeat)[strIndex+2:]):
                 if type(secondDict['是']).__name__ == 'dict':
                     classLabel = classify(secondDict["是"],feature,testX)
                 else:
                     classLabel = secondDict["是"]
            else:# entering into the sub tree “否”
                if type(secondDict['否']).__name__ == 'dict':
                     classLabel = classify(secondDict["否"],feature,testX)
                else:
                     classLabel = secondDict["否"]
                
        else:
            if testX[featureIndex]==key:
                
                if type(secondDict[key]).__name__=='dict':
                    classLabel = classify(secondDict[key],feature,testX)
                else:
                    classLabel = secondDict[key]
    return classLabel

def getNumLeaf(tree):
    '''Get the number of leaf nodes'''
    numLeaf=0
    firstfeat = dict_to_list(tree)[0]
    secondDict = tree[firstfeat]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeaf += getNumLeaf(secondDict[key])
        else:
            numLeaf += 1
    return numLeaf

def getTreeDepth(tree):
    '''Get the tree depth'''
    
    maxDepth=0
    firstfeat = dict_to_list(tree)[0]
    secondDict = tree[firstfeat]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
    if thisDepth>maxDepth:
        maxDepth=thisDepth
    return maxDepth

def testingErr(tree,feature,dataSet):
    """test error"""
    X = [data[:-1] for data in dataSet]
    Y = [data[-1] for data in dataSet]
    n = len(Y)
    labelResult = []

    for item in X:
        labelResult.append(classify(tree,feature,item))
    
    errorCnt = 0
    for i in range(n):
        if labelResult[i]!=Y[i]:
            errorCnt += 1
    
    return errorCnt/n
   
def testingMajor(major,data_test):
    """Test the accuracy of predicting all the categories with the highest frequency"""
    error=0.0
    for i in range(len(data_test)):
        if major!=data_test[i][-1]:
            error+=1
     
    return float(error)/len(data_test)
    
def treeScore(tree,feature,dataSet):
    '''
    :param feature:["no surfacing","flippers"]
    '''

    accuracy = 1-testingErr(tree,feature,dataSet)  
    TreeSize = getNumLeaf(tree)
    TreeDepth = getTreeDepth(tree)
    print("accuracy:",accuracy)
    print("TreeSize:",TreeSize)
    print("TreeDepth:",TreeDepth)

def postPruningTree(tree,dataSet,data_Test,feature):
    """Error-based pruning"""
    firstfeat = list(tree.keys())[0]
    strfeat = copy.deepcopy(firstfeat)
    secondDict = tree[firstfeat]
    strIndex = str(firstfeat).find("<")
    if strIndex > -1:
        firstfeat = str(firstfeat)[:strIndex]
        firstvalue = float(str(strfeat)[strIndex+2:])
    featureIndex = feature.index(firstfeat)
    temp_feature = copy.deepcopy(feature)
    if strIndex == -1:
        del(feature[featureIndex])
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            if strIndex == -1:
                tree[firstfeat][key] = \
                    postPruningTree(secondDict[key],\
                                    getSubDataset(dataSet,featureIndex,key),\
                                    getSubDataset(data_Test,featureIndex,key),
                                    copy.deepcopy(feature))

            else:
                direction = 0 if key == "是" else 1
                tree[strfeat][key] = \
                    postPruningTree(secondDict[key],\
                                    getContinuousSubDataset(dataSet,featureIndex,firstvalue,direction),\
                                    getContinuousSubDataset(data_Test,featureIndex,firstvalue,direction),\
                                    copy.deepcopy(feature))
    if testingErr(tree,temp_feature,dataSet) <= testingMajor(vote(dataSet),data_Test):
        return tree
    
    return vote(dataSet)
        
                    
if __name__ == "__main__":
    
    # Get data 
    import time
    time_start = time.time()
    wine = load_wine()
    dataSet = [data.tolist() for data in wine.data]
    for i in range(len(dataSet)):
        dataSet[i].append(wine.target[i])
    feature = wine.feature_names

    # Split train and test dataset
    train_dataSet, test_dataSet = train_test_split_data(dataSet,0.7)
    feature_0 = copy.deepcopy(feature)
    feature_1 = copy.deepcopy(feature)

    # Building a ID3 tree
    tree = createTree(train_dataSet,feature)
    newTree = postPruningTree(tree,train_dataSet,test_dataSet,feature_0)
    treeScore(newTree,feature_1,train_dataSet)
    treeScore(newTree,feature_1,test_dataSet)
    time_end = time.time()  
    time_c= time_end - time_start   
    print('time cost', time_c, 's')
   




