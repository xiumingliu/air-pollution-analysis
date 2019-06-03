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
meteo = np.array(meteo).reshape(5, -1)

# Replace zeros with average
no2 = functions.replace_zero_with_average(no2)

# Log-normal transform
no2_transformed = functions.log_normal(no2)

# =============================================================================
# Configuration
# =============================================================================
time_test = np.arange('2014-03-12', '2014-03-13', dtype='datetime64[h]')
time_train = np.arange('2006-01-01', '2014-03-12', dtype='datetime64[h]')

index_test = np.arange(np.where(time == time_test[0])[0], np.where(time == time_test[-1])[0]+1)
index_train = np.arange(np.where(time == time_train[0])[0], np.where(time == time_train[-1])[0]+1)

data_test = no2_transformed[:, index_test]
data_train = no2_transformed[:, index_train]

index_busday = np.is_busday(time_train.astype('datetime64[D]'))
data_train_busday = data_train[:, index_busday].reshape(5, -1, 24) 
data_train_holiday = data_train[:, np.logical_not(index_busday)].reshape(5, -1, 24) 

# =============================================================================
# The model
# =============================================================================   
acf = functions.acf(data_train)

mean_busday = np.mean(data_train_busday, axis=1)
mean_holiday = np.mean(data_train_holiday, axis=1)

# =============================================================================
# The prior of testing data
# =============================================================================
if np.is_busday(time_test[0].astype('datetime64[D]')):
    mu_test_prior = mean_busday
else:
    mu_test_prior = mean_holiday
    
cov_test_prior = np.zeros((5, 24, 24))
for i in range(5):
    cov_test_prior[i, :, :] = functions.covmat(acf[i, :], time_test)

# =============================================================================
# Iterative updating
# =============================================================================
MAX_ITERATION = 8
BATCH_SIZE = 24*7*32  # weeks
TOTAL_SIZE = data_train.shape[1]

this_mu_test_prior = mu_test_prior
this_cov_test_prior = cov_test_prior

this_mu_train = np.empty((5, BATCH_SIZE))
this_cov_train = np.empty((5, BATCH_SIZE, BATCH_SIZE))
this_cov_train_test = np.empty((5, BATCH_SIZE, 24))
this_cov_test_train = np.empty((5, 24, BATCH_SIZE))

this_mu_train_given_test = np.empty((5, BATCH_SIZE))
this_cov_train_given_test = np.empty((5, BATCH_SIZE, BATCH_SIZE))

this_H = np.empty((5, BATCH_SIZE, 24))
this_G = np.empty((5, BATCH_SIZE, BATCH_SIZE))

this_residual = np.empty((5, BATCH_SIZE))
this_coefficient = np.empty((5, 24, BATCH_SIZE))
this_mu_test_posterior = np.empty((5, 24))
this_cov_test_posterior = np.empty((5, 24, 24))

for iteration in range(MAX_ITERATION):
    this_train_index = np.arange(TOTAL_SIZE-(iteration+1)*BATCH_SIZE, TOTAL_SIZE-iteration*BATCH_SIZE)
    this_time_train = time_train[this_train_index]
    this_data_train = data_train[:, this_train_index]
    
    # Marginal distribution of the k-th segment of data
    for i in range(5):
        this_mu_train[i, :] = functions.meanvec(mean_busday[i, :], mean_holiday[i, :], this_time_train)

    # Conditional distribution of the k-th segment of data
    for i in range(5):
        this_cov_train[i, :, :] = functions.covmat(acf[i, :], this_time_train)
        this_cov_train_test[i, :, :] = functions.xcovmat(acf[i, :], this_time_train, time_test)
        this_cov_test_train[i, :, :] = np.transpose(this_cov_train_test[i, :, :])
        
        this_H[i, :, :] = np.dot(this_cov_train_test[i, :, :], np.linalg.inv(cov_test_prior[i, :, :]))
        
        this_mu_train_given_test[i, :] = this_mu_train[i, :] + np.dot(this_H[i, :, :], (this_mu_test_prior[i, :] - mu_test_prior[i, :]))
        this_cov_train_given_test[i, :, :] = this_cov_train[i, :, :] - np.dot(this_H[i, :, :], this_cov_test_train[i, :, :])
        
    # Update the posterior
        this_G[i, :, :] = this_cov_train_given_test[i, :, :] + np.dot(np.dot(this_H[i, :, :], this_cov_test_prior[i, :, :]), np.transpose(this_H[i, :, :]))
        
        this_residual[i, :] = this_data_train[i, :] - this_mu_train[i, :] - np.dot(this_H[i, :, :], (this_mu_test_prior[i, :] - mu_test_prior[i, :]))
        this_coefficient[i, :, :] = np.dot(np.dot(this_cov_test_prior[i, :, :], np.transpose(this_H[i, :, :])), np.linalg.inv(this_G[i, :, :]))
        this_mu_test_posterior[i, :] = this_mu_test_prior[i, :] + np.dot(this_coefficient[i, :, :], this_residual[i, :])
        this_cov_test_posterior[i, :] = this_cov_test_prior[i, :, :] - np.dot(np.dot(this_coefficient[i, :, :], this_H[i, :, :]), this_cov_test_prior[i, :,:])
        
    # Update the prior for next interation
    this_mu_test_prior = this_mu_test_posterior
    this_cov_test_prior = this_cov_test_posterior
        
# Inverse transformed of the log-normal data     
mu_test_posterior_inv, cov_test_posterior_inv = functions.log_normal_inverse(this_mu_test_posterior, this_cov_test_posterior)
    
samples_mean, samples_median, samples_percentile_high, samples_percentile_low = functions.sample_log_normal(this_mu_test_posterior, this_cov_test_posterior, 10000, [95, 5])    
fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for i in range(5):
#    axs[i].plot(samples_median[i, :], 'k') 
    axs[i].plot(samples_mean[i, :], 'k--') 
    axs[i].plot(no2[i, index_test], 'k:')
    axs[i].fill_between(np.arange(0, 24), samples_percentile_low[i, :], samples_percentile_high[i, :], color='lightgray')
    axs[i].set_xlim([0, 23])
    axs[i].set_xlabel(r'Time (hour)')
    axs[i].set_ylim([0, .8*np.max(no2)])
axs[0].set_ylabel(r'NO$_2$ ($\mu g/m^3$)')

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24,4), sharey=True)
i = 0
for ax in axes.flat:
    print(i)
    im = ax.imshow(cov_test_posterior_inv[i, :, :], cmap = 'binary', vmin = 0, vmax = np.max(cov_test_posterior_inv))
    i = i + 1
    ax.set_xlabel(r'Time (hour)')
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad = 0.01)
axes[0].set_ylabel(r'Time (hour)')
plt.show()