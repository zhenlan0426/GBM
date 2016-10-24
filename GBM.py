# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import random as rd
from sklearn.tree import DecisionTreeRegressor      


class Bagging_GBM(BaseEstimator):
    
    def __init__(self,BaseEst,BasePara,M_bag,subsample):
        self.BaseEst=BaseEst
        self.BasePara=BasePara
        self.M_bag=M_bag
        self.subsample=subsample
        # self.subfeature=subfeature
        self.estimator_=[]
        self.dataIndex=[]
        # self.featureIndex=[]
        
           
    def fit(self,X,y):    
        n,p=X.shape
        n_sample=int(n*self.subsample)
        # n_feature=int(p*self.subfeature)
        for i in range(self.M_bag):
            self.dataIndex.append(rd.sample(xrange(n),n_sample))
            # self.featureIndex.append(rd.sample(xrange(p),n_feature))
            self.estimator_.append(self.BaseEst(**self.BasePara).fit(X[self.dataIndex[i],:],y[self.dataIndex[i]]))
            # fit(X[self.dataIndex[i],:][:,self.featureIndex[i]],y[self.dataIndex[i]]))
        return self
    

    def predict(self,X):
        yhat=0
        for model in self.estimator_:
            yhat+=model.predict(X)
        return yhat/self.M_bag
            
            
    def OOB_error(self,X,y,scorer):
        error=0
        n=y.shape[0]
        count_out=0
        for i in range(n):
            yhat=0
            count=0
            for j in range(self.M_bag):
                if i in self.dataIndex[j]:
                    continue
                else:
                    yhat+=self.estimator_[j].predict(X[i,:])
                    count+=1
            if count==0:
                continue
            else:
                error+=scorer(y[i],yhat/count)
                count_out+=1
        return error/count_out
        
            


class GBM(BaseEstimator, ClassifierMixin):
    def __init__(self,BaseEst,M_est,learnRate,alpha,BasePara,subFold):
        self.BaseEst=BaseEst
        self.M_est=M_est
        self.learnRate=learnRate
        self.estimator_=[]
        self.alpha=alpha
        self.BasePara=BasePara
        self.subFold=subFold
        

    # plain fitting     
    def fit(self,X,y):
        n=y.shape[0]
        self.init_=np.median(y,0)
        yhat = self.init_*np.ones_like(y)
        kf = KFold(n, n_folds=self.subFold)
        for i in range(self.M_est):
            #pdb.set_trace()
            index=np.random.permutation(n) # shuffle index for subsampling
            X,y,yhat = X[index,:],y[index],yhat[index]
            for _,test in kf:
                error=y[test]-yhat[test]
                a=np.percentile(error,self.alpha)
                target=np.where(np.abs(error)<=a,error,a*np.sign(error))
                self.estimator_.append(self.BaseEst(**self.BasePara).fit(X[test],target))
                yhat+=self.learnRate*self.estimator_[-1].predict(X)
        return self  
        
    ## use OOB error to early stop ##    
    def fit1(self,X,y,max_check,ratio_check):
        n=y.shape[0]
        self.init_=np.median(y)
        yhat = self.init_*np.ones(n)
        kf = KFold(n, n_folds=self.subFold)
        n_check=0
        for i in range(self.M_est):
            #pdb.set_trace()
            index=np.random.permutation(n) # shuffle index for subsampling
            X,y,yhat = X[index,:],y[index],yhat[index]
            for train,test in kf:
                error=y[test]-yhat[test]
                a=np.percentile(error,self.alpha)
                target=np.where(np.abs(error)<=a,error,a*np.sign(error))
                self.estimator_.append(self.BaseEst(**self.BasePara).fit(X[test],target))
                temp=yhat+self.learnRate*self.estimator_[-1].predict(X)
                # use OOB error to early stop 
                if np.sum((y[train]-temp[train])**2)/np.sum((y[train]-yhat[train])**2)>ratio_check:
                    n_check+=1
                    if n_check>max_check:
                        return self
                else:
                    n_check=0
                yhat=temp
                        
        return self
        
    ## use OOB error to update learning rate ##    
    ## learning rate is incorporated by changing tree_.value ##
    def fit2(self,X,y):
        n=y.shape[0]
        self.init_=np.median(y)
        yhat = self.init_*np.ones(n)
        kf = KFold(n, n_folds=self.subFold)
        for i in range(self.M_est):
            #pdb.set_trace()
            index=np.random.permutation(n) # shuffle index for subsampling
            X,y,yhat = X[index,:],y[index],yhat[index]
            for train,test in kf:
                error_tot=y-yhat
                error=error_tot[test]
                a=np.percentile(error,self.alpha)
                target=np.where(np.abs(error)<=a,error,a*np.sign(error))
                self.estimator_.append(self.BaseEst(**self.BasePara).fit(X[test],target))
                temp=self.estimator_[-1].predict(X)
                # beta=np.sum(error_tot[train]*temp[train])/np.sum(temp[train]**2) # refit beta with OOB sample 
                beta=np.sum(error_tot*temp)/np.sum(temp**2) # refit beta with whole sample
                # print beta
                for i,value in enumerate(self.estimator_[-1].tree_.value):                    
                    self.estimator_[-1].tree_.value[i]=value*beta
                yhat+=temp*self.learnRate*beta
                        
        return self      
          
    def predict(self,X):
        yhat=self.init_
        for clf in self.estimator_:
            yhat= yhat + self.learnRate*clf.predict(X)
        return yhat
    
    def plot(self,X,y,scorer):
        yhat=self.init_
        score=[]
        for clf in self.estimator_:
            yhat = yhat + self.learnRate*clf.predict(X)
            score.append(scorer(y,yhat))
        plt.plot(score)
            
            
            
    

def L1_score(y_test, yhat):
    medianY=np.median(y_test,0)
    return 1-np.sum(np.abs(y_test-yhat))/np.sum(np.abs(y_test-medianY))
def L2_score(y_test, yhat):
    meanY=np.mean(y_test,0)
    return 1-np.sum((y_test-yhat)**2)/np.sum((y_test-meanY)**2)
        
      
#model1=GBM(DecisionTreeRegressor,1000,0.1,90,{'max_depth':4,'splitter':'random','max_features':0.8},2)            
#model1.fit(X,y)
#model1=GBM(DecisionTreeRegressor,1000,.3,90,{'max_leaf_nodes':8},2)  
#model1.fit2(X,y)


#yhat1=model1.predict(X_val)
#plt.scatter(yhat1,y_val)
#L2_score(y_val,yhat1)
#model1.plot(X_val,y_val,L2_score)


#BasePara={'BaseEst':DecisionTreeRegressor,'M_est':1000,'learnRate':0.1,'alpha':95,\
#'BasePara':{'max_depth':4,'splitter':'random','max_features':0.8},'subFold':2}
#model_agg=Bagging_GBM(GBM,BasePara,3,0.9)
#model_agg.fit(X,y)






