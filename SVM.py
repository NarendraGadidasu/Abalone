
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from helpers_SVM import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

class primalSVM_RBF(BaseEstimator, ClassifierMixin):
    '''http://scikit-learn.org/stable/developers/contributing.html'''
    
    def __init__(self, alpha=1e-9,gamma_frac=0.1,n_iter=2000):
         self.alpha = alpha
         self.gamma_frac = gamma_frac
         self.n_iter = n_iter
         
    def fit(self, X, y):
         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         
         # Get the kernel matrix
         dist = euclidean_distances(X,squared=True)
         median = np.median(dist) 
         del dist
         gamma = median
         gamma *= self.gamma_frac
         self.gamma = 1/gamma
         kernels = rbf_kernel(X,None,self.gamma )
         
         self.X_ = X
         self.classes_ = unique_labels(y)
         self.kernels_ = kernels
         self.y_ = y
         self.clf = SGDClassifier(loss='hinge',penalty='l2',alpha=self.alpha,
                                  l1_ratio=0,fit_intercept=True,verbose=False,
                                  average=False,learning_rate='optimal',
                                  class_weight='balanced',n_iter=self.n_iter,
                                  random_state=55)         
         self.clf.fit(self.kernels_,self.y_)
         
         # Return the classifier
         return self

    def predict(self, X):
         # Check is fit had been called
         check_is_fitted(self, ['X_', 'y_','clf','kernels_'])
         # Input validation
         X = check_array(X)
         new_kernels = rbf_kernel(X,self.X_,self.gamma )
         pred = self.clf.predict(new_kernels)
         return pred
    





ab = pd.read_hdf('datasets.hdf','ab')        
abX = ab.drop('rings',1).copy().values
abY = ab['rings'].copy().values

ab_trgX, ab_tstX, ab_trgY, ab_tstY = ms.train_test_split(abX, abY, test_size=0.3, random_state=0,stratify=abY)

N_ab = ab_trgX.shape[0]

alphas = [10**-x for x in np.arange(1,9.01,1/2)]


#Linear SVM

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

params_ab = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_ab)/.8)+1]}

params_ab = {'SVM__alpha':[ 0.001], 'SVM__n_iter': [428]}


ab_clf = basicResults(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,params_ab,'SVM_Lin','ab')        

y_score = ab_clf.decision_function(ab_tstX)

fpr, tpr, thresholds = roc_curve(ab_tstY, y_score)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(fpr, tpr)

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.title('ROC_Curve(Abalone)')

plt.savefig('./output/SVM_Lin_ROC_Curve.png')

plt.clf()

cm = pd.DataFrame(confusion_matrix(ab_tstY, ab_clf.predict(ab_tstX)))

cm.to_csv('./output/SVM_Lin_Confusion_matrix.csv')


ab_final_params =ab_clf.best_params_
ab_OF_params = ab_final_params.copy()
ab_OF_params['SVM__alpha'] = 1e-16





pipeA.set_params(**ab_final_params)
makeTimingCurve(abX,abY,pipeA,'SVM_Lin','ab')

pipeA.set_params(**ab_final_params)
iterationLC(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_Lin','ab')                

pipeA.set_params(**ab_OF_params)
iterationLC(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,{'SVM__n_iter':np.arange(1,200,5)},'SVM_LinOF','ab')



#RBF SVM
gamma_fracsA = np.arange(0.2,2.1,0.2)

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('SVM',primalSVM_RBF())])


params_ab = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_ab)/.8)+1],'SVM__gamma_frac':gamma_fracsA}

                                                  
ab_clf = basicResults(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,params_ab,'SVM_RBF','ab')      

ab_final_params =ab_clf.best_params_
ab_OF_params = ab_final_params.copy()
ab_OF_params['SVM__alpha'] = 1e-16

pipeA.set_params(**ab_final_params)
makeTimingCurve(abX,abY,pipeA,'SVM_RBF','ab')

pipeA.set_params(**ab_final_params)
iterationLC(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','ab')                

pipeA.set_params(**ab_OF_params)
iterationLC(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','ab')                