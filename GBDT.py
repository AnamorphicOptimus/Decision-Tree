#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GBDT Implemented in Python
"""
from math import exp
import numpy as np
from sklearn.datasets import load_wine
from ID3 import  train_test_split_data

class iter_GBDT:
    def __init__(self,\
                 origin_data = None,\
                 label_set = None,\
                 dataSet = None,\
                 iterIndex = None,\
                 labelIndex = None,\
                 K = None,\
                 max_depth = None,\
                 min_sample = None,\
                 ex_residual = None,\
                 residual = None,\
                 prob = None,\
                 tree = None,\
                 lr = None,\
                 ex_F = None,\
                 new_F = None):
        self.origin_data = origin_data
        self.label_set = label_set
        self.iterIndex = iterIndex
        self.labelIndex = labelIndex
        self.dataSet = origin_data[labelIndex]
        self.K = K
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.ex_residual = ex_residual
        self.tree = createTree(self.dataSet,self.ex_residual,self.K,0,self.max_depth,self.min_sample)
        
        tmp_residual = []

        for data in self.dataSet:
            tmp_residual.append(self.tree.predict(data))        
        self.residual = tmp_residual 
        
        self.lr = lr
        self.ex_F = ex_F
        self.new_F = update_F(self.ex_F,self.residual,self.lr)
      

def GBDT_fit(dataSet,n_iter,lr,max_depth = None,min_sample = None):
    
    # Initialize p,y,r
    n_label, p_0, label_set= compute_p_0(dataSet)
    y_0 = one_hot(dataSet)
    r_0 = (np.array(y_0)-np.array(p_0)).tolist()
    data_label = get_label_data_list(dataSet)
    GBDT_model = {}
    for n_round in range(n_iter):
        GBDT_model[roundName(n_round)] = {}
        print("n_round",n_round)
        if n_round == 0: 
            for labelIndex in range(n_label):
                GBDT_model[roundName(n_round)][labelName(labelIndex)] = \
                    iter_GBDT(origin_data = data_label,\
                              label_set = label_set,\
                              iterIndex = n_round,\
                              labelIndex = labelIndex,\
                              lr=lr,\
                              K = n_label,\
                              ex_residual = r_0[labelIndex],\
                              max_depth = max_depth,\
                              min_sample = min_sample,\
                              ex_F = [0]*len(dataSet))
        else:
            ex_iter_labelIndex_F = [] 
            for j in range(n_label):
                F = GBDT_model[roundName(n_round-1)][labelName(j)].new_F
                ex_iter_labelIndex_F.append(F)
                
            for labelIndex in range(n_label):
                ex_r = get_ex_residual(data_label,labelIndex,ex_iter_labelIndex_F)
                GBDT_model[roundName(n_round)][labelName(labelIndex)] = \
                     iter_GBDT(origin_data = data_label,\
                               label_set = label_set,\
                               iterIndex = n_round,\
                               labelIndex = labelIndex,\
                               K = n_label,\
                               ex_residual = ex_r,\
                               max_depth = max_depth,\
                               min_sample = min_sample,\
                               ex_F = ex_iter_labelIndex_F[labelIndex]
                               )
    return GBDT_model

def GBDT_Predict(model,data):
    K=model["round_0"]["label_0"].K
    lr=model["round_0"]["label_0"].lr
    label_set = list(model["round_0"]["label_0"].label_set)
    prob=[]
    for labelIndex in range(K):
        F_0=0.0
        for rd in model:
            F_0 += lr*model[rd][labelName(labelIndex)].tree.predict(data)
        prob.append(F_0)
    labelMaxIndex = prob.index(max(prob))
    results = label_set[labelMaxIndex]
    return prob, results

def GBDT_score(model,test_dataSet):
    acc=0
    for data in test_dataSet:
        if data[-1] == GBDT_Predict(model,data)[1]:
            acc+=1
    accr=acc/len(test_dataSet)
    print("Test Acc:",accr)
    return accr

class Tree:
    def __init__(self,\
                 split_feature = None,\
                 trueBranch = None,\
                 falseBranch = None,\
                 predict_value = None,\
                 conditionValue = None,\
                 data = None,\
                 ex_residual = None):
        self.split_feature = split_feature
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.predict_value = predict_value
        self.conditionValue = conditionValue
        self.data = data 
        self.ex_residual = ex_residual 
    
    def predict(self,data):
        """a slide data to predict"""
        if self.predict_value!=None:
            return self.predict_value
        else:
            value = self.conditionValue
            if data[self.split_feature]<=value:
                return self.trueBranch.predict(data)
            else:
                return self.falseBranch.predict(data)
        

def createTree(dataSet,ex_residual,K,depth,max_depth=None,min_sample=None):
    """Build a tree"""

    # Stop splitting condition 1: Decision tree depth
    # Stop splitting condition 2: Decision leaf node sample number
    if max_depth==None:
        max_depth = 5
    if min_sample==None:
        min_sample = -1
    if depth > max_depth or len(dataSet) < min_sample:
        print("directly return")
        return Tree(data = dataSet,predict_value = compute_leaf_value(ex_residual,K))
    
    feaLen = len(dataSet[0])-1
    MSE_data = compute_MSE(ex_residual)
    MSE_0 = -1

    for feaIndex in range(feaLen):
        feaList = [data[feaIndex] for data in dataSet]
        feaSet = set(feaList)
        for fea in feaSet:
            trueSet, falseSet, trueRe, falseRe = split_data(dataSet,ex_residual,feaIndex,fea)
            MSE_true = compute_MSE(trueRe)
            MSE_false = compute_MSE(falseRe)
            MSE_total =MSE_true+MSE_false
            if MSE_0 < 0 or MSE_total < MSE_0:
                MSE_0 = MSE_total
                best_data = (trueSet,falseSet)
                best_residual = (trueRe,falseRe)
                best_feaIndex = feaIndex
                best_fea = fea
    if MSE_0 < MSE_data:
        trueBranch = createTree(best_data[0],best_residual[0],K,depth+1,max_depth,min_sample)
        falseBranch = createTree(best_data[1],best_residual[1],K,depth+1,max_depth,min_sample)
        return Tree(split_feature=best_feaIndex,trueBranch=trueBranch,falseBranch=falseBranch,\
                    conditionValue=best_fea,data=dataSet)
    else:
        return Tree(predict_value = compute_leaf_value(ex_residual,K))

def get_ex_residual(data_label,labelIndex,F):
    y = [data[-1] for data in data_label[labelIndex]]
    prob = []
    
    for i in range(len(data_label[0])):
        p = exp(F[labelIndex][i])/sum([exp(f[i]) for f in F])
        prob.append(p)
    
    return (np.array(y)-np.array(prob)).tolist()
    
def get_label_data_list(dataSet):
    """newdataSet[i] is a dataSet formed by 0-1 classification based on labelIndex=i"""
    newdataSet = []
    tmp_label = one_hot(dataSet)
    labelSet = set([data[-1] for data in dataSet])
    for labelIndex in range(len(labelSet)):
        temp = tmp_label[labelIndex]
        label_data_temp = []
        for i in range(len(dataSet)):
            t = dataSet[i][:-1]
            t.append(temp[i])
            label_data_temp.append(t)
        newdataSet.append(label_data_temp)
    return newdataSet
        
def compute_MSE(ex_residual):
    """Calculate MSE, when the loss function is the mean square error, the mean is the best predicted value"""
    try:
        MSE = sum((np.array(ex_residual)-np.mean(np.array(ex_residual)))**2)/len(ex_residual)
    except ZeroDivisionError:
        MSE = 0
    return MSE
    
def split_data(dataSet,ex_residual,feaIndex,value):
    trueSet = []
    trueRe = []
    falseSet = []
    falseRe = []
    for i in range(len(dataSet)):
        if dataSet[i][feaIndex] <= value:
            trueSet.append(dataSet[i])
            trueRe.append(ex_residual[i])
        else: 
            falseSet.append(dataSet[i])
            falseRe.append(ex_residual[i])
    return trueSet, falseSet, trueRe, falseRe
  
def compute_leaf_value(ex_residual,K):
    """Leaf node linear search fitting approximation"""
    ex_residual = np.array(ex_residual)
    sum1 = sum(ex_residual)
    sum2 = sum(abs(ex_residual)*(1.0-abs(ex_residual)))

    return ((K-1)/K)*(sum1/sum2)  
    
def labelName(labelIndex):
    return "label_"+str(labelIndex)
            
def roundName(n_round):
    return "round_"+str(n_round)

def one_hot(dataSet):
    """Multi-classification requires one-hot encoding of labels"""    
    label_set = set([data[-1] for data in dataSet])
    one_hot = [] 
    for label in label_set:
        label_tmp = []
        for data in dataSet:
            if data[-1]==label:
                label_tmp.append(1)
            else:
                label_tmp.append(0)
        one_hot.append(label_tmp)
    return one_hot
                   
def compute_p_0(dataSet):
    """
    Initialize p k
    :return: [[p001,p002...],[p011,p012...],[p021,p022...]]
    """
    F_0 = 0.0
    p_0 = []
    label_set = set([data[-1] for data in dataSet])
    n_label = len(label_set)
    p = exp(F_0)/(n_label*exp(F_0))
    for i in range(len(label_set)):
        p_0.append([p]*len(dataSet))
    return n_label, p_0, label_set  
      
def update_F(ex_F,residual,lr = None):
    """update F"""
    if lr==None:
        lr=1
    return (np.array(ex_F)+lr*np.array(residual)).tolist()

def data_init():
    """
    return:
    [[14.23,1.71,2.43,15.6,127.0,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065.0,0],
     [13.2,1.78,2.14,11.2,100.0,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050.0,0]]...
        
    """
    wine = load_wine()
    dataSet = [data.tolist() for data in wine.data]
    for i in range(len(dataSet)):
        dataSet[i].append(wine.target[i])
    feature = wine.feature_names
    
    return feature, dataSet

if __name__ == "__main__":
    
    import time
    time_start = time.time()
    feature,dataSet = data_init()
    train_dataSet, test_dataSet = train_test_split_data(dataSet,0.7)
    print("Finish data_init!")

    # Build GBDT model
    m = GBDT_fit(train_dataSet,n_iter=70,lr=0.2)
    GBDT_score(m,train_dataSet)
    GBDT_score(m,test_dataSet)

    time_end = time.time()  
    time_c= time_end - time_start   
    print('time cost', time_c, 's')
    


