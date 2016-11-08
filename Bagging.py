# -*- coding: utf-8 -*-
"""
Created on Tue May 03 21:32:43 2016

@author: zhenlanwang
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

class Bagging_KClass(BaseEstimator, ClassifierMixin):
        
    def __init__(self,BaseEst,M_est,BasePara,subSample,subFeature):

        self.BaseEst=BaseEst
        self.M_est=M_est
        self.estimator_=[]
        self.BasePara=BasePara
        self.subSample=subSample
        self.subFeature=subFeature
        self.subFeatureIndex=[]

        
    def fit(self,X,y):
        
        n , d = X.shape
        K = len(np.unique(y))
        self.K = K
        yp = np.zeros((n,K))
        counter = np.zeros(n)
        score = []        
        
        for i in range(self.M_est):
            subSample_index = np.random.rand(n)<self.subSample
            self.subFeatureIndex.append(np.random.rand(d)<self.subFeature)
            self.estimator_.append(self.BaseEst(**self.BasePara)\
                                   .fit(X[subSample_index][:,self.subFeatureIndex[-1]],y[subSample_index]))   
                                   
            counter[~subSample_index] += 1        
            nonZeroIndex = counter!=0             
            yp[~subSample_index] += self.estimator_[-1].predict_proba(X[~subSample_index][:,self.subFeatureIndex[-1]])
            score.append(log_loss(y[nonZeroIndex],yp[nonZeroIndex]/counter[nonZeroIndex]))
        
        plt.plot(score)
        return np.min(score)
        
        
    def predict_proba(self,X):
        n = X.shape[0]
        yp = np.zeros((n,self.K))
        
        for i in range(self.M_est):
            yp+=self.estimator_[i].predict_proba(X[:,self.subFeatureIndex[i]])
        
        return yp/self.M_est
        
    def predict(self,X):
        return np.argmax(self.predict_proba(X),1)
            
            
def Bagging_RandomSearch(X,y,Ntry,FixBasePara,FixBagPara,RandomPara):
    # for LR baselearner only
    
    result = np.zeros((Ntry,4))
    
    for i in range(Ntry):
        BagPara={}
        result[i,0] = np.random.uniform(*RandomPara[0])
        BagPara['subSample'] = result[i,0]
        result[i,1] = np.random.uniform(*RandomPara[1])
        BagPara['subFeature'] = result[i,1]
        
        BasePara={}
        result[i,2] = np.random.choice(RandomPara[2])
        BasePara.update(FixBasePara)
        BagPara['BasePara'] = BasePara
        
        BagPara.update(FixBagPara)

        model = Bagging_KClass(**BagPara)
        result[i,3] = model.fit(X,y)
        
    return pd.DataFrame(result,columns=['subSample','subFeature','C','acc'])
            
            
            
#Search = Bagging_RandomSearch(X3,y,20,\
#          {'solver':'sag','max_iter':5000},\
#          {'BaseEst':LogisticRegression,'M_est':60},\
#          [(0.1,0.6),(0.06,0.5),(1,0.1,0.001,0.0001,0.00001)])            
            
            
            