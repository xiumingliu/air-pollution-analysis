# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:10:56 2019

@author: Administrator
"""

import import_data
import functions
import numpy as np
import time

from keras.models import load_model

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
no2, nox, pm10, meteo, date = import_data.import_data()

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
test_day_start = np.datetime64('2013-01-01', dtype='datetime64[D]')
test_day_end = np.datetime64('2013-01-02', dtype='datetime64[D]')

error = np.empty(((test_day_end-test_day_start).astype('int'), 5, 24))
estmated_var = np.empty(((test_day_end-test_day_start).astype('int'), 5, 24))


time_train = np.arange('2006-01-01', '2013-01-01', dtype='datetime64[h]')
index_train = np.arange(np.where(date == time_train[0])[0], np.where(date == time_train[-1])[0]+1)
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

NN_model_busday = load_model("bestmodel_no2_5l_busday.h5")
NN_model_holiday = load_model("bestmodel_no2_5l_holiday.h5") 

for day in np.arange(test_day_start, test_day_end, dtype='datetime64[D]'):
    
    t = time.time()
    
    #time_test = np.arange('2014-03-12', '2014-03-13', dtype='datetime64[h]')
    #time_train = np.arange('2006-01-01', '2014-03-12', dtype='datetime64[h]')
    time_test = np.arange(day, day+np.timedelta64(1, dtype='datetime64[D]'), dtype='datetime64[h]')

    print(day)

    index_test = np.arange(np.where(date == time_test[0])[0], np.where(date == time_test[-1])[0]+1)

    data_test = no2_transformed[:, index_test]
    
    
    # =============================================================================
    # Prediction made by the NN
    # =============================================================================
#    meteo_test = meteo[:, index_test[0]-24:index_test[0]]
#    feature_test = np.empty((6, 24))
#    feature_test[0:4, :] = meteo_test[[0, 1, 2, 4], :]
#    feature_test[4, :] = np.sin(np.deg2rad(meteo_test[3, :]))
#    feature_test[5, :] = np.cos(np.deg2rad(meteo_test[3, :]))
#    
#    if np.is_busday(time_test[0].astype('datetime64[D]')):
#        mu_test_prior = NN_model_busday.predict(np.concatenate(feature_test).reshape(1, 144))
#    else:
#        mu_test_prior = NN_model_holiday.predict(np.concatenate(feature_test).reshape(1, 144))  
#    mu_test_prior = mu_test_prior.reshape(5, 24)
    
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
        
#    fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
#    for i in range(5):
#        axs[i].plot(np.exp(mu_test_prior[i, :]), 'k') 
#        axs[i].plot(no2[i, index_test], 'k:')
#        axs[i].set_xlim([0, 23])
#        axs[i].set_xlabel(r'Time (hour)')
#        axs[i].set_ylim([0, .8*np.max(no2)])
#    axs[0].set_ylabel(r'NO$_2$ ($\mu g/m^3$)')
#    
#    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24,4), sharey=True)
#    i = 0
#    for ax in axes.flat:
#        print(i)
#        im = ax.imshow(cov_test_prior[i, :, :], cmap = 'binary', vmin = 0, vmax = np.max(cov_test_prior))
#        i = i + 1
#        ax.set_xlabel(r'Time (hour)')
#    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad = 0.01)
#    axes[0].set_ylabel(r'Time (hour)')
#    plt.show()
    
    # =============================================================================
    # SGP updating
    # =============================================================================

    BATCH_SIZE = 24*7*40  # weeks
    TOTAL_SIZE = data_train.shape[1]
    IND_SIZE = 24*7*1
    
    this_mu_test_prior = mu_test_prior
    this_cov_test_prior = cov_test_prior
    
    this_mu_train = np.empty((5, BATCH_SIZE))
    this_cov_train = np.empty((5, BATCH_SIZE, BATCH_SIZE))
    this_cov_train_test = np.empty((5, BATCH_SIZE, 24))
    this_cov_test_train = np.empty((5, 24, BATCH_SIZE))
    this_cov_test_ind = np.empty((5, 24, IND_SIZE))
    this_cov_ind_test = np.empty((5, IND_SIZE, 24))
    this_cov_train_ind =  np.empty((5, BATCH_SIZE, IND_SIZE))
    this_cov_ind_train =  np.empty((5, IND_SIZE, BATCH_SIZE))
    
    this_mu_train_given_test = np.empty((5, BATCH_SIZE))
    this_cov_train_given_test = np.empty((5, BATCH_SIZE, BATCH_SIZE))
    
    this_mu_train_given_ind = np.empty((5, BATCH_SIZE))
    this_cov_train_given_ind = np.empty((5, BATCH_SIZE, BATCH_SIZE))
    
    this_H = np.empty((5, BATCH_SIZE, IND_SIZE))
    
    this_mu_test_posterior = np.empty((5, 24))
    this_cov_test_posterior = np.empty((5, 24, 24))
    this_P = np.empty((5, IND_SIZE, IND_SIZE))
    
    this_train_index = np.arange(TOTAL_SIZE-BATCH_SIZE, TOTAL_SIZE)
    this_time_train = time_train[this_train_index]
    this_data_train = data_train[:, this_train_index]
    
#    this_ind_index = np.random.choice(this_train_index, IND_SIZE)
    this_ind_index = np.arange(TOTAL_SIZE-IND_SIZE, TOTAL_SIZE)
    this_time_ind = time_train[this_ind_index]
    this_data_ind = data_train[:, this_ind_index]
    
    this_mu_ind_prior = np.empty((5, IND_SIZE))
    for i in range(5):
        for j in range(IND_SIZE):
            if np.is_busday(this_mu_ind_prior[i, j].astype('datetime64[D]')):
                this_mu_ind_prior[i, j] = mean_busday[i, (this_time_ind[0] - this_time_ind[0].astype('datetime64[D]')).astype('int')]
            else:
                this_mu_ind_prior[i, j] = mean_holiday[i, (this_time_ind[0] - this_time_ind[0].astype('datetime64[D]')).astype('int')]
    this_cov_ind_prior = np.empty((5, IND_SIZE, IND_SIZE))
    for i in range(5):
        this_cov_ind_prior[i, :, :] = functions.covmat(acf[i, :], this_time_ind)
    
    
    # Marginal distribution of the k-th segment of data
    for i in range(5):
        this_mu_train[i, :] = functions.meanvec(mean_busday[i, :], mean_holiday[i, :], this_time_train)

    # Conditional distribution of the k-th segment of data
    for i in range(5):
        this_cov_train[i, :, :] = functions.covmat(acf[i, :], this_time_train)
        this_cov_train_test[i, :, :] = functions.xcovmat(acf[i, :], this_time_train, time_test)
        this_cov_test_train[i, :, :] = np.transpose(this_cov_train_test[i, :, :])
        
        this_cov_test_ind[i, :, :] = functions.xcovmat(acf[i, :], time_test, this_time_ind)
        this_cov_ind_test[i, :, :] = np.transpose(this_cov_test_ind[i, :, :])
        
        this_cov_train_ind[i, :, :] = functions.xcovmat(acf[i, :], this_time_train, this_time_ind)
        this_cov_ind_train[i, :, :] = np.transpose(this_cov_train_ind[i, :, :])
        
        this_H[i, :, :] = np.dot(this_cov_train_ind[i, :, :], np.linalg.inv(this_cov_ind_prior[i, :, :]))
        
        this_mu_train_given_ind[i, :] = this_mu_train[i, :] + np.dot(this_H[i, :, :], (this_data_ind[i, :] - this_mu_ind_prior[i, :]))
        this_cov_train_given_ind[i, :, :] = np.diag(np.diag(this_cov_train[i, :, :] - np.dot(this_H[i, :, :], this_cov_ind_train[i, :, :]))) + 1e-5*np.identity(BATCH_SIZE)
        
        this_P[i, :, :] = np.linalg.inv(this_cov_ind_prior[i, :, :] + np.dot(np.dot(this_cov_ind_train[i, :, :], np.linalg.inv(this_cov_train_given_ind[i, :, :])), this_cov_train_ind[i, :, :]))
        
        this_mu_test_posterior[i, :] = this_mu_test_prior[i, :] + np.dot(np.dot(np.dot(np.dot(this_cov_test_ind[i, :, :], this_P[i, :, :]), this_cov_ind_train[i, :, :]), np.linalg.inv(this_cov_train_given_ind[i, :, :])), (this_data_train[i, :] - this_mu_train[i, :]))
        this_cov_test_posterior[i, :] = this_cov_test_prior[i, :, :] - np.dot(np.dot(this_cov_test_ind[i, :, :], np.linalg.inv(this_cov_ind_prior[i, :, :])), this_cov_ind_test[i, :, :]) + np.dot(np.dot(this_cov_test_ind[i, :, :], this_P[i, :, :]), this_cov_ind_test[i, :, :])
    
            
    # Inverse transformed of the log-normal data     
    mu_test_posterior_inv, cov_test_posterior_inv = functions.log_normal_inverse(this_mu_test_posterior, this_cov_test_posterior)
    
    samples_mean, samples_median, samples_percentile_high, samples_percentile_low, samples_var = functions.sample_log_normal(this_mu_test_posterior, this_cov_test_posterior, 10000, [95, 5])    
    error[(day-test_day_start).astype('int'), :, :] = no2[:, index_test] - samples_mean
    estmated_var[(day-test_day_start).astype('int'), :, :] = samples_var
    
    elapsed = time.time() - t
    print(elapsed)
    
    
#    fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
#    for i in range(5):
#        axs[i].plot(samples_mean[i, :], 'k') 
#        axs[i].plot(no2[i, index_test], 'k:')
#        axs[i].fill_between(np.arange(0, 24), samples_percentile_low[i, :], samples_percentile_high[i, :], color='lightgray')
#        axs[i].set_xlim([0, 23])
#        axs[i].set_xlabel(r'Time (hour)')
#        axs[i].set_ylim([0, .8*np.max(no2)])
#    axs[0].set_ylabel(r'NO$_2$ ($\mu g/m^3$)')
#    
#    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24,4), sharey=True)
#    i = 0
#    for ax in axes.flat:
#        print(i)
#        im = ax.imshow(cov_test_posterior_inv[i, :, :], cmap = 'binary', vmin = 0, vmax = np.max(cov_test_posterior_inv))
#        i = i + 1
#        ax.set_xlabel(r'Time (hour)')
#    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad = 0.01)
#    axes[0].set_ylabel(r'Time (hour)')
#    plt.show()
    

error_p95 = np.percentile(np.abs(error), 95, axis = 0)
error_p5 = np.percentile(np.abs(error), 5, axis = 0)
average_error = np.mean(np.abs(error), axis = 0)

fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for i in range(5):
    axs[i].plot(average_error[i, :], 'k')
    axs[i].fill_between(np.arange(0, 24), error_p5[i, :], error_p95[i, :], color='lightgray')
    axs[i].set_xlim([0, 23])
    axs[i].set_xlabel(r'Time (hour)')
    axs[i].set_ylim([0, 100])
axs[0].set_ylabel(r'NO$_{2}$ ($\mu g/m^3$)')
plt.tight_layout()
plt.savefig("no2_experiement_ind1_bs10_withoutNN_sgp.pdf", format='pdf')

np.save('no2_experiement_ind1_bs10_withoutNN_sgp', error)
