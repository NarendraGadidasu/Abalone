# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:37:46 2019

@author: A103932
"""

import pandas as pd
import numpy as np

# Preprocess with adult dataset
ab = pd.read_csv('abalone.data',header=None)
ab.columns = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings']
    
ab = pd.get_dummies(ab)
ab = ab.rename(columns=lambda x: x.replace('-','_'))

ab.loc[ab['rings']<=15, 'rings'] = 0
ab.loc[ab['rings']>15, 'rings'] = 1

ab.to_hdf('datasets.hdf','ab',complib='blosc',complevel=9)