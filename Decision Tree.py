# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:55:32 2017

Script for full tests, decision tree (pruned)

"""

import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('./output/DT_{}_nodecounts.csv'.format(dataset))
    
    return



    

# Load Data       
ab = pd.read_hdf('datasets.hdf','ab')        
abX = ab.drop('rings',1).copy().values
abY = ab['rings'].copy().values

ab_trgX, ab_tstX, ab_trgY, ab_tstY = ms.train_test_split(abX, abY, test_size=0.3, random_state=0,stratify=abY)     

# Search for good alphas
alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]

pipeA = Pipeline([('Scale',StandardScaler()),                 
                 ('DT',dtclf_pruned(random_state=55))])


params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}

ab_clf = basicResults(pipeA,ab_trgX,ab_trgY,ab_tstX,ab_tstY,params,'DT','ab')        

#ab_final_params = {'class_weight': 'balanced', 'alpha': 0.0031622776601683794, 'criterion': 'entropy'}

ab_final_params = ab_clf.best_params_

pipeA.set_params(**ab_final_params)
makeTimingCurve(abX,abY,pipeA,'DT','ab')

DTpruningVSnodes(pipeA,alphas,ab_trgX,ab_trgY,'ab')