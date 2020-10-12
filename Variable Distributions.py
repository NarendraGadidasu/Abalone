# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 23:58:01 2019

@author: A103932
"""

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

ab = pd.read_hdf('datasets.hdf','ab')  

#'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
#       'viscera_weight', 'shell_weight', 'rings', 'sex_F', 'sex_I', 'sex_M'

ab['age_classes'] = ab['rings']

fig = plt.figure()

ax1 = fig.add_subplot(141)
sns.boxplot(ab['age_classes'], ab['length'])

ax2 = fig.add_subplot(142)
sns.boxplot(ab['age_classes'], ab['diameter'])

ax3 = fig.add_subplot(143)
sns.boxplot(ab['age_classes'], ab['whole_weight'])

ax4 = fig.add_subplot(144)
sns.boxplot(ab['age_classes'], ab['shell_weight'])

plt.savefig('./output/distribution.png')

