# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:29:52 2019

@author: Administrator
"""

import numpy as np
from scipy import signal

def replace_zero_with_average(data):
    for i in range(data.shape[0]):
        data[i, data[i, :]<=0] = np.average(data[i, :])
    
    return data

def log_normal(data):
    data_transfromed = np.empty(np.shape(data))
    for i in range(data.shape[0]):
        data_transfromed[i, :] = np.log(data[i, :])
    
    return data_transfromed

def log_normal_inverse(data_mu, data_cov): 
    data_mu_inv = np.empty(np.shape(data_mu)) 
    data_cov_inv = np.empty(np.shape(data_cov)) 
    
    for i in range(data_mu.shape[0]):
        data_mu_inv[i, :] = np.exp(data_mu[i, :]+0.5*np.diag(data_cov[i, :, :])).reshape(-1)
        
        for row in range(data_cov.shape[1]):
            for col in range(data_cov.shape[2]):
                data_cov_inv[i, row, col] = np.exp(data_mu[i, row] + data_mu[i, col] + 0.5*(data_cov[i, row, row] + data_cov[i, col, col]))*(np.exp(data_cov[i, row, col])-1)
    
    return data_mu_inv, data_cov_inv

def sample_log_normal(mu, cov, sample_size, q):
    samples_mu = np.empty(np.shape(mu))
    samples_median = np.empty(np.shape(mu))
    samples_percentile1 = np.empty(np.shape(mu))
    samples_percentile2 = np.empty(np.shape(mu))
    samples_var = np.empty(np.shape(mu))
    for i in range(mu.shape[0]):
        this_samples = np.exp(np.random.multivariate_normal(mu[i, :], cov[i, :, :], size=sample_size))
        samples_mu[i, :] = np.mean(this_samples, axis = 0)
        samples_median[i, :] = np.median(this_samples, axis = 0)
        samples_percentile1[i, :] = np.percentile(this_samples, q[0], axis=0)
        samples_percentile2[i, :] = np.percentile(this_samples, q[1], axis=0)
        samples_var[i, :] = np.var(this_samples, axis = 0)
        
    return samples_mu, samples_median, samples_percentile1, samples_percentile2, samples_var

def acf(data_train):
    acf = np.zeros((np.shape(data_train)[0], 2*np.shape(data_train)[1]-1))
    for i in range(data_train.shape[0]):
        acf[i, :] = (signal.correlate(data_train[i]-np.mean(data_train[i]), data_train[i]-np.mean(data_train[i])))/data_train[i].shape[0]

    return acf

def timelag(t1, t2): 
#    t1 and t2 are consecutive time sequences in hours
    lag_start = (t1[-1] - t2[0])/np.timedelta64(1, 'h')
    lag_stop = (t1[0] - t2[-1])/np.timedelta64(1, 'h')
    if lag_start > lag_stop:
        lag = np.arange(lag_start, lag_stop-1, -1)
    else:
        lag = np.arange(lag_start, lag_stop+1, 1)   
        
    return lag

#def meanvec(mean_busday, mean_holiday, t):
#
#    num_days = int(t.shape[0]/24)
#    
#    mu = np.empty((5, t.shape[0]))
#    for d in range(num_days):
#        if np.is_busday(t[d*24].astype('datetime64[D]')):
#            mu[:, d*24:(d+1)*24] = mean_busday
#        else:
#            mu[:, d*24:(d+1)*24] = mean_holiday
#    
#    return mu
    
def meanvec(mean_busday, mean_holiday, t):

    num_days = int(t.shape[0]/24)
    
    mu = np.empty((1, t.shape[0]))
    for d in range(num_days):
        if np.is_busday(t[d*24].astype('datetime64[D]')):
            mu[0, d*24:(d+1)*24] = mean_busday
        else:
            mu[0, d*24:(d+1)*24] = mean_holiday
    
    return mu

def covmat(acf, t1): 
    L = np.size(acf)
    l = np.size(t1)
    Sigma = np.zeros((l, l))
    
    lag = timelag(t1, t1)
    lag_max = lag[0]
    
    index_lag_zero = int((L-1)/2)
    index_lag_max = int((L-1)/2+lag_max)
    
    for row in range(l):
        Sigma[row, :] = acf[index_lag_zero-row:index_lag_max-row+1].reshape(-1)
    
    return Sigma
    
def xcovmat(xcf, t1, t2): 
    L = np.size(xcf)
    l1 = np.size(t1)
    l2 = np.size(t2)
    Sigma = np.zeros((l1, l2))
    
    lag = timelag(t1, t2)
    
    index_lag_zero = int((L-1)/2)
    index_lag_start = index_lag_zero + int(lag[0])
    index_lag_end = index_lag_zero + int(lag[-1])
    
    if index_lag_start > index_lag_end:
        this_xcf = xcf[index_lag_end:index_lag_start+1]
        this_xcf = np.flip(this_xcf)
    else:
        this_xcf = xcf[index_lag_start:index_lag_end+1]
        
    for row in range(l1):
        Sigma[row, :] = this_xcf[l1-row-1:l1+l2-row-1].reshape(-1)
    
    return Sigma  