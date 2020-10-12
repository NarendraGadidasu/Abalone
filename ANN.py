# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:37:46 2019

@author: A103932
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

ab = pd.read_hdf('datasets.hdf','ab')        
abX = ab.drop('rings',1).copy().values
abY = ab['rings'].copy().values

ab_trgX, ab_tstX, ab_trgY, ab_tstY = ms.train_test_split(abX, abY, test_size=0.3, random_state=0,stratify=abY)

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

d = abX.shape[1]
hiddens_ab = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]

params_ab = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_ab}

ab_clf = basicResults(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,params_ab,'ANN','ab')        



ab_final_params =ab_clf.best_params_
ab_OF_params =ab_final_params.copy()
ab_OF_params['MLP__alpha'] = 0

pipeA.set_params(**ab_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(abX,abY,pipeA,'ANN','ab')

pipeA.set_params(**ab_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','ab')                

pipeA.set_params(**ab_OF_params)
pipeA.set_params(**{'MLP__early_stopping':False})               
iterationLC(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','ab')                

