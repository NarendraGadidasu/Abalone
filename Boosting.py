# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""


import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helpers import dtclf_pruned
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



ab = pd.read_hdf('datasets.hdf','ab')        
abX = ab.drop('rings',1).copy().values
abY = ab['rings'].copy().values

alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]


ab_trgX, ab_tstX, ab_trgY, ab_tstY = ms.train_test_split(abX, abY, test_size=0.3, random_state=0,stratify=abY)

ab_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                

paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
          'Boost__base_estimator__alpha':alphas}

ab_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=ab_base,random_state=55)
OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)


pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('Boost',ab_booster)])


ab_clf = basicResults(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,paramsA,'Boost','ab')        

ab_final_params = ab_clf.best_params_
OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}

pipeA.set_params(**ab_final_params)

makeTimingCurve(abX,abY,pipeA,'Boost','ab')
pipeA.set_params(**ab_final_params)
iterationLC(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','ab')                

             
