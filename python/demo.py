# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:10:56 2019

@author: Administrator
"""

import import_data
import functions
import numpy as np
import pandas as pd
from scipy import signal
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

# =============================================================================
# Import data
# =============================================================================
no2, nox, pm10, meteo, time = import_data.import_data()

# =============================================================================
# Pre-processing
# =============================================================================
no2 = np.array(no2).reshape(5, -1)

pt_no2_0 = PowerTransformer()
pt_no2_0.fit(no2[0, :].reshape(-1, 1))
pt_no2_1 = PowerTransformer()
pt_no2_1.fit(no2[1, :].reshape(-1, 1))
pt_no2_2 = PowerTransformer()
pt_no2_2.fit(no2[2, :].reshape(-1, 1))
pt_no2_3 = PowerTransformer()
pt_no2_3.fit(no2[3, :].reshape(-1, 1))
pt_no2_4 = PowerTransformer()
pt_no2_4.fit(no2[4, :].reshape(-1, 1))

no2_transformed = np.zeros(np.shape(no2))
no2_transformed[0, :] = pt_no2_0.transform(no2[0, :].reshape(-1, 1)).reshape(-1)
no2_transformed[1, :] = pt_no2_1.transform(no2[1, :].reshape(-1, 1)).reshape(-1)
no2_transformed[2, :] = pt_no2_2.transform(no2[2, :].reshape(-1, 1)).reshape(-1)
no2_transformed[3, :] = pt_no2_3.transform(no2[3, :].reshape(-1, 1)).reshape(-1)
no2_transformed[4, :] = pt_no2_4.transform(no2[4, :].reshape(-1, 1)).reshape(-1)

# =============================================================================
# Configuration
# =============================================================================
time_test = np.arange('2014-03-12', '2014-03-13', dtype='datetime64[h]')
time_train = np.arange('2006-01-01', '2014-03-12', dtype='datetime64[h]')

index_test = np.arange(np.where(time == time_test[0])[0], np.where(time == time_test[-1])[0]+1)
index_train = np.arange(np.where(time == time_train[0])[0], np.where(time == time_train[-1])[0]+1)

data_test = no2_transformed[:, index_test]
data_train = no2_transformed[:, index_train]

# =============================================================================
# The model
# =============================================================================
acf = np.zeros((np.shape(data_train)[0], 2*np.shape(data_train)[1]-1))
for i in range(5):
    acf[i, :] = (signal.correlate(data_train[i]-np.mean(data_train[i]), data_train[i]-np.mean(data_train[i])))/data_train[i].shape[0]
    
index_busday = np.is_busday(time_train.astype('datetime64[D]'))
data_train_busday = data_train[:, index_busday].reshape(5, -1, 24) 
data_train_holiday = data_train[:, np.logical_not(index_busday)].reshape(5, -1, 24) 

mean_busday = np.mean(data_train_busday, axis=1)
mean_holiday = np.mean(data_train_holiday, axis=1)
    
if np.is_busday(time_test[0].astype('datetime64[D]')):
    mu_test_prior = mean_busday
else:
    mu_test_prior = mean_holiday
    
cov_test_prior = np.zeros((5, 24, 24))
for i in range(5):
    cov_test_prior[i, :, :] = functions.covmat(acf[i, :], time_test)


#std_busday = np.std(data_train_busday, axis=1)
#std_holiday = np.std(data_train_holiday, axis=1)
#
#low_busday = mean_busday - 2*std_busday
#high_busday = mean_busday + 2*std_busday
#    
##plt.figure(figsize=(20, 4))    
#fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24,4), sharey=True)
#i = 0
#for ax in axes.flat:
#    print(i)
#    im = ax.imshow(COV[i, :, :], vmin=0, vmax=1e5, cmap = 'gray')
#    i = i + 1
#    ax.set_xlabel(r'Time (hour)')
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[0, 1e5], pad = 0.01)
#cbar.ax.set_yticklabels([r'0', r'1e5'])
#axes[0].set_ylabel(r'Time (hour)')
#plt.show()
#
#mean_busday_inv = np.zeros((5, 24))
#mean_busday_inv[0, :] = pt_no2_0.inverse_transform(mean_busday[0, :].reshape(-1, 1)).reshape(-1)
#mean_busday_inv[1, :] = pt_no2_0.inverse_transform(mean_busday[1, :].reshape(-1, 1)).reshape(-1)
#mean_busday_inv[2, :] = pt_no2_0.inverse_transform(mean_busday[2, :].reshape(-1, 1)).reshape(-1)
#mean_busday_inv[3, :] = pt_no2_0.inverse_transform(mean_busday[3, :].reshape(-1, 1)).reshape(-1)
#mean_busday_inv[4, :] = pt_no2_0.inverse_transform(mean_busday[4, :].reshape(-1, 1)).reshape(-1)
#
#low_busday_inv = np.zeros((5, 24))
#low_busday_inv[0, :] = pt_no2_0.inverse_transform(low_busday[0, :].reshape(-1, 1)).reshape(-1)
#low_busday_inv[1, :] = pt_no2_0.inverse_transform(low_busday[1, :].reshape(-1, 1)).reshape(-1)
#low_busday_inv[2, :] = pt_no2_0.inverse_transform(low_busday[2, :].reshape(-1, 1)).reshape(-1)
#low_busday_inv[3, :] = pt_no2_0.inverse_transform(low_busday[3, :].reshape(-1, 1)).reshape(-1)
#low_busday_inv[4, :] = pt_no2_0.inverse_transform(low_busday[4, :].reshape(-1, 1)).reshape(-1)
#
#high_busday_inv = np.zeros((5, 24))
#high_busday_inv[0, :] = pt_no2_0.inverse_transform(high_busday[0, :].reshape(-1, 1)).reshape(-1)
#high_busday_inv[1, :] = pt_no2_0.inverse_transform(high_busday[1, :].reshape(-1, 1)).reshape(-1)
#high_busday_inv[2, :] = pt_no2_0.inverse_transform(high_busday[2, :].reshape(-1, 1)).reshape(-1)
#high_busday_inv[3, :] = pt_no2_0.inverse_transform(high_busday[3, :].reshape(-1, 1)).reshape(-1)
#high_busday_inv[4, :] = pt_no2_0.inverse_transform(high_busday[4, :].reshape(-1, 1)).reshape(-1)
#    
#fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
#for i in range(5):
#    axs[i].plot(mean_busday_inv[i, :], 'k') 
#    axs[i].plot(no2[i, index_test], 'k:')
#    axs[i].fill_between(np.arange(0, 24), low_busday_inv[i, :], high_busday_inv[i, :], color='lightgray')
#    axs[i].set_xlim([0, 23])
#    axs[i].set_xlabel(r'Time (hour)')
#    axs[i].set_ylim([0, 200])
#axs[0].set_ylabel(r'NO$_2$ ($\mu g/m^3$)')

# =============================================================================
# Iterative updating
# =============================================================================
MAX_ITERATION = 32
BATCH_SIZE = 24*7*4    # 1 weeks
TOTAL_SIZE = data_train.shape[1]

this_mu_test_prior = mu_test_prior
this_cov_test_prior = cov_test_prior

this_cov_train = np.empty((5, BATCH_SIZE, BATCH_SIZE))
this_cov_train_test = np.empty((5, BATCH_SIZE, 24))
this_cov_test_train = np.empty((5, 24, BATCH_SIZE))

this_mu_train_given_test = np.empty((5, BATCH_SIZE))
this_cov_train_given_test = np.empty((5, BATCH_SIZE, BATCH_SIZE))

this_H = np.empty((5, BATCH_SIZE, 24))
this_G = np.empty((5, BATCH_SIZE, BATCH_SIZE))

this_residual = np.empty((5, BATCH_SIZE))
this_mu_test_posterior = np.empty((5, 24))
this_cov_test_posterior = np.empty((5, 24, 24))

for iteration in range(MAX_ITERATION):
    this_train_index = np.arange(TOTAL_SIZE-(iteration+1)*BATCH_SIZE, TOTAL_SIZE-iteration*BATCH_SIZE)
    this_time_train = time_train[this_train_index]
    this_data_train = data_train[:, this_train_index]
    
    this_mu_train = functions.meanvec(mean_busday, mean_holiday, this_time_train)

    
    for i in range(5):
        this_cov_train[i, :, :] = functions.covmat(acf[i, :], this_time_train)
        this_cov_train_test[i, :, :] = functions.xcovmat(acf[i, :], this_time_train, time_test)
        this_cov_test_train[i, :, :] = np.transpose(this_cov_train_test[i, :, :])
        
#    Conditional distribution for the k-th segment of data
        this_mu_train_given_test[i, :] = this_mu_train[i, :] + np.dot(np.dot(this_cov_train_test[i, :, :], np.linalg.inv(cov_test_prior[i, :, :])), (this_mu_test_prior[i, :] - mu_test_prior[i, :]))
        this_cov_train_given_test[i, :, :] = this_cov_train[i, :, :] - np.dot(np.dot(this_cov_train_test[i, :, :], np.linalg.inv(cov_test_prior[i, :, :])), this_cov_test_train[i, :, :])
        
        this_H[i, :, :] = np.dot(this_cov_train_test[i, :, :], np.linalg.inv(cov_test_prior[i, :, :]))
        this_G[i, :, :] = this_cov_train_given_test[i, :, :] + np.dot(np.dot(this_H[i, :, :], this_cov_test_prior[i, :, :]), np.transpose(this_H[i, :, :]))
        
        this_residual[i, :] = this_data_train[i, :] - this_mu_train[i, :] - np.dot(this_H[i, :, :], (this_mu_test_prior[i, :] - mu_test_prior[i, :]))
        this_mu_test_posterior[i, :] = this_mu_test_prior[i, :] + np.dot(np.dot(np.dot(this_cov_test_prior[i, :,:], np.transpose(this_H[i, :, :])), np.linalg.inv(this_G[i, :, :])), this_residual[i, :])
        this_cov_test_posterior[i, :] = this_cov_test_prior[i, :, :] - np.dot(np.dot(np.dot(np.dot(this_cov_test_prior[i, :,:], np.transpose(this_H[i, :, :])), np.linalg.inv(this_G[i, :, :])), this_H[i, :, :]), this_cov_test_prior[i, :,:])

    this_mu_test_prior = this_mu_test_posterior
    this_cov_test_prior = this_cov_test_posterior
        
mu_test_posterior_inv = np.empty((5, 24)) 
mu_test_posterior_inv[0, :] = pt_no2_0.inverse_transform(this_mu_test_posterior[0, :].reshape(-1, 1)).reshape(-1)
mu_test_posterior_inv[1, :] = pt_no2_0.inverse_transform(this_mu_test_posterior[1, :].reshape(-1, 1)).reshape(-1)
mu_test_posterior_inv[2, :] = pt_no2_0.inverse_transform(this_mu_test_posterior[2, :].reshape(-1, 1)).reshape(-1)
mu_test_posterior_inv[3, :] = pt_no2_0.inverse_transform(this_mu_test_posterior[3, :].reshape(-1, 1)).reshape(-1)
mu_test_posterior_inv[4, :] = pt_no2_0.inverse_transform(this_mu_test_posterior[4, :].reshape(-1, 1)).reshape(-1)

std_test_posterior = np.empty((5, 24)) 
for i in range(5):
    std_test_posterior[i, :] = np.sqrt(np.diag(this_cov_test_posterior[i, :, :]))
    
low_test_posterior = this_mu_test_prior - 2*std_test_posterior
high_test_posterior = this_mu_test_prior + 2*std_test_posterior

low_test_posterior_inv = np.empty((5, 24)) 
low_test_posterior_inv[0, :] = pt_no2_0.inverse_transform(low_test_posterior[0, :].reshape(-1, 1)).reshape(-1)
low_test_posterior_inv[1, :] = pt_no2_0.inverse_transform(low_test_posterior[1, :].reshape(-1, 1)).reshape(-1)
low_test_posterior_inv[2, :] = pt_no2_0.inverse_transform(low_test_posterior[2, :].reshape(-1, 1)).reshape(-1)
low_test_posterior_inv[3, :] = pt_no2_0.inverse_transform(low_test_posterior[3, :].reshape(-1, 1)).reshape(-1)
low_test_posterior_inv[4, :] = pt_no2_0.inverse_transform(low_test_posterior[4, :].reshape(-1, 1)).reshape(-1)

high_test_posterior_inv = np.empty((5, 24)) 
high_test_posterior_inv[0, :] = pt_no2_0.inverse_transform(high_test_posterior[0, :].reshape(-1, 1)).reshape(-1)
high_test_posterior_inv[1, :] = pt_no2_0.inverse_transform(high_test_posterior[1, :].reshape(-1, 1)).reshape(-1)
high_test_posterior_inv[2, :] = pt_no2_0.inverse_transform(high_test_posterior[2, :].reshape(-1, 1)).reshape(-1)
high_test_posterior_inv[3, :] = pt_no2_0.inverse_transform(high_test_posterior[3, :].reshape(-1, 1)).reshape(-1)
high_test_posterior_inv[4, :] = pt_no2_0.inverse_transform(high_test_posterior[4, :].reshape(-1, 1)).reshape(-1)


fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for i in range(5):
    axs[i].plot(mu_test_posterior_inv[i, :], 'k') 
    axs[i].plot(no2[i, index_test], 'k:')
    axs[i].fill_between(np.arange(0, 24), low_test_posterior_inv[i, :], high_test_posterior_inv[i, :], color='lightgray')
    axs[i].set_xlim([0, 23])
    axs[i].set_xlabel(r'Time (hour)')
    axs[i].set_ylim([0, 200])
axs[0].set_ylabel(r'NO$_2$ ($\mu g/m^3$)')
 
#fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24,4), sharey=True)
#i = 0
#for ax in axes.flat:
#    print(i)
#    im = ax.imshow(this_cov_test_posterior[i, :, :], cmap = 'gray')
#    i = i + 1
#    ax.set_xlabel(r'Time (hour)')
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad = 0.01)
##cbar.ax.set_yticklabels([r'0', r'1e5'])
#axes[0].set_ylabel(r'Time (hour)')
#plt.show()