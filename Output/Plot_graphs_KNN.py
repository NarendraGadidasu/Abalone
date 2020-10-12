# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:14:12 2019

@author: A103932
"""

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

algo = 'KNN'

data = 'ab'

res = pd.read_csv('{}_{}_reg.csv'.format(algo, data))

#model complexity curve

res = res[['mean_test_score','mean_train_score', 

           #change the parameter below
         'param_KNN__metric', 'param_KNN__n_neighbors','param_KNN__weights']]

#chnage the conditions below
res1 = res.loc[res['param_KNN__weights'] == 'distance', :]

res1 = res1.loc[res1['param_KNN__metric'] == 'euclidean', :]

res1 = res1.drop(columns = ['param_KNN__weights'])

res1.sort_values(by = 'param_KNN__n_neighbors', inplace=True)

res1 = res1.rename(columns = {'mean_train_score':'train_balanced_accuracy', 'mean_test_score':'CV_balanced_accuracy'})

fig, ax1 = plt.subplots()

lns1 = ax1.plot(res1['param_KNN__n_neighbors'], res1['CV_balanced_accuracy'], label = 'CV_balanced_accuracy', c = 'orange')
lns2 = ax1.plot(res1['param_KNN__n_neighbors'], res1['train_balanced_accuracy'], label = 'train_balanced_accuracy', c = 'b')

ax1.set_ylabel('accuracy', color = 'black')

plt.xlabel('n_neighbors')

plt.title('Eucledian weighted distance (Abalone)')

plt.legend()

plt.savefig('{}_Acc_wrt_n_neighbors_eucledian.png'.format(algo))

plt.clf()

#learning curve wrt samples

lc_train = pd.read_csv('{}_{}_LC_train.csv'.format(algo,data))

lc_test = pd.read_csv('{}_{}_LC_test.csv'.format(algo,data))

lc_train.rename(columns = {'Unnamed: 0':'num_of_samples'}, inplace = True)

lc_test.rename(columns = {'Unnamed: 0':'num_of_samples'}, inplace = True)

lc_train['train_balanced_accuracy'] = (lc_train['0']+lc_train['1']+lc_train['2']+lc_train['3']+lc_train['4'])/5

lc_test['CV_balanced_accuracy'] = (lc_test['0']+lc_test['1']+lc_test['2']+lc_test['3']+lc_test['4'])/5

lc_train.drop(columns = ['0','1','2','3','4'], inplace = True)

lc_test.drop(columns = ['0','1','2','3','4'], inplace = True)

plt.figure()

fig, ax = plt.subplots()

ax.plot(lc_train['num_of_samples'], lc_train['train_balanced_accuracy'], label = 'train_balanced_accuracy')

ax.plot(lc_test['num_of_samples'], lc_test['CV_balanced_accuracy'], label = 'CV_balanced_accuracy')

plt.legend()

plt.title('Learning curve wrt Number of training samples (Abalone)')

plt.xlabel('Number of training samples')

plt.savefig('{}_LC_wrt_Samples.png'.format(algo))

plt.clf()

#timing curve wrt samples

lc_time = pd.read_csv('{}_{}_timing.csv'.format(algo,data))

lc_time.rename(columns = {'Unnamed: 0':'fraction_of_data_to_score', 'test':'score_time', 'train':'fit_time'}, inplace = True)

lc_time['fraction_of_data_to_fit'] = 1-lc_time['fraction_of_data_to_score']

plt.figure()

fig, ax1 = plt.subplots()

lns1 = ax1.plot(lc_time['fraction_of_data_to_score'], lc_time['score_time'], label = 'score_time', c='orange')

ax1.set_ylabel('score_time', color = 'orange')

plt.xlabel('fraction_of_data_to_fit/score')

lc_time.sort_values(by = 'fraction_of_data_to_fit', inplace=True)

ax2 = ax1.twinx()

lns2 = ax2.plot(lc_time['fraction_of_data_to_fit'], lc_time['fit_time'], label = 'fit_time', c='blue')

ax2.set_ylabel('fit_time', color = 'blue')

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

plt.title('fit_or_score_time_wrt_fraction_of_data (Abalone)')

plt.savefig('{}_fit_or_score_time_wrt_fraction_of_data.png'.format(algo))

plt.clf()



