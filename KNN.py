# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:42:58 2017

@author: JTay
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

ab = pd.read_hdf('datasets.hdf','ab')        
abX = ab.drop('rings',1).copy().values
abY = ab['rings'].copy().values

ab_trgX, ab_tstX, ab_trgY, ab_tstY = ms.train_test_split(abX, abY, test_size=0.3, random_state=0,stratify=abY)

d = abX.shape[1]
hiddens_ab = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(1,9.01,1/2)]  

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('KNN',knnC())])  

params_ab= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}

ab_clf = basicResults(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,params_ab,'KNN','ab')        

ab_final_params=ab_clf.best_params_

pipeA.set_params(**ab_final_params)
makeTimingCurve(abX,abY,pipeA,'KNN','ab')