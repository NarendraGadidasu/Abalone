# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:14:12 2019

@author: A103932
"""

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

algo = 'Boost'

data = 'ab'

res = pd.read_csv('{}_{}_reg.csv'.format(algo, data))

#model complexity curve

res = res[['mean_test_score','mean_train_score', 

           #change the parameter below
         'param_Boost__base_estimator__alpha', 'param_Boost__n_estimators']]

#chnage the conditions below
res1 = res.loc[res['param_Boost__n_estimators'] == 1, :]

res1 = res1.drop(columns = ['param_Boost__n_estimators'])

res1.sort_values(by = 'param_Boost__base_estimator__alpha', inplace=True)

res1 = res1.rename(columns = {'mean_train_score':'train_balanced_accuracy', 'mean_test_score':'CV_balanced_accuracy'})

fig, ax1 = plt.subplots()

lns1 = ax1.plot(res1.iloc[2:, 2], res1.iloc[2:,0], label = 'CV_balanced_accuracy', c = 'r')
lns2 = ax1.plot(res1.iloc[2:, 2], res1.iloc[2:,1], label = 'train_balanced_accuracy', c = 'b')

#lns1 = ax1.plot(res1['param_Boost__base_estimator__alpha'], res1['CV_balanced_accuracy'], label = 'CV_balanced_accuracy', c = 'r')
#lns2 = ax1.plot(res1['param_Boost__base_estimator__alpha'], res1['train_balanced_accuracy'], label = 'train_balanced_accuracy', c = 'b')

ax1.set_ylabel('accuracy', color = 'black')

#ax1.set_ylim(0.84, 1)

plt.xlabel('base_estimator__alpha')

plt.legend()

plt.title('Number of Iterations : 1 (Abalone)')

plt.savefig('{}_Acc_wrt_Alpha.png'.format(algo))

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

#Learning Curve wrt Iterations

lc_iter = pd.read_csv('ITER_base_{}_{}.csv'.format(algo,data))

lc_iter = lc_iter[['mean_test_score','mean_train_score', 'param_Boost__n_estimators']]

lc_iter = lc_iter.rename(columns = {'mean_train_score':'train_balanced_accuracy', 'mean_test_score':'CV_balanced_accuracy'})

lc_iter.plot('param_Boost__n_estimators', ['CV_balanced_accuracy', 'train_balanced_accuracy'])

plt.xlabel('Number of Estimators')

plt.title('Learning curve wrt Number of Estimators (Abalone)')

plt.savefig('{}_LC_wrt_Iterations.png'.format(algo))

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




