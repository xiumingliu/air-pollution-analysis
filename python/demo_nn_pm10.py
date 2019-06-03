# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 08:58:33 2019

@author: AdminisStrator
"""

import import_data
import functions
import numpy as np
import pandas as pd
from scipy import signal

from keras.models import Sequential
from keras.layers import Dense
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
no2, nox, pm10, meteo, time = import_data.import_data()

# =============================================================================
# Pre-processing
# =============================================================================
pm10 = np.array(pm10).reshape(5, -1)
meteo = np.array(meteo).reshape(5, -1)

# Replace zeros with average
pm10 = functions.replace_zero_with_average(pm10)

# Log-normal transform
pm10_transformed = functions.log_normal(pm10)

# =============================================================================
# Configuration
# =============================================================================
time_test_output = np.arange('2014-03-12', '2014-03-13', dtype='datetime64[h]')
time_test_input = np.arange('2014-03-11', '2014-03-12', dtype='datetime64[h]')

time_train_output = np.arange('2006-01-02', '2013-01-01', dtype='datetime64[h]')
time_train_input = np.arange('2006-01-01', '2012-12-31', dtype='datetime64[h]')

index_test_output = np.arange(np.where(time == time_test_output[0])[0], np.where(time == time_test_output[-1])[0]+1)
index_test_input = np.arange(np.where(time == time_test_input[0])[0], np.where(time == time_test_input[-1])[0]+1)

index_train_output = np.arange(np.where(time == time_train_output[0])[0], np.where(time == time_train_output[-1])[0]+1)
index_train_input = np.arange(np.where(time == time_train_input[0])[0], np.where(time == time_train_input[-1])[0]+1)

data_test = pm10_transformed[:, index_test_output]
data_train = pm10_transformed[:, index_train_output]

meteo_test = meteo[:, index_test_input]
meteo_train = meteo[:, index_train_input]

index_busday_test = np.is_busday(time_test_output.astype('datetime64[D]'))
index_busday_train = np.is_busday(time_train_output.astype('datetime64[D]'))

feature_test = np.empty((7, np.size(index_test_input)))
feature_train = np.empty((7, np.size(index_train_input)))

feature_test[0:4, :] = meteo_test[[0, 1, 2, 4], :]
feature_test[4, :] = np.sin(np.deg2rad(meteo_test[3, :]))
feature_test[5, :] = np.cos(np.deg2rad(meteo_test[3, :]))
feature_test[6, :] = index_busday_test

feature_train[0:4, :] = meteo_train[[0, 1, 2, 4], :]
feature_train[4, :] = np.sin(np.deg2rad(meteo_train[3, :]))
feature_train[5, :] = np.cos(np.deg2rad(meteo_train[3, :]))
feature_train[6, :] = index_busday_train

# =============================================================================
# NN
# =============================================================================

feature_train = np.transpose(feature_train.reshape(7, -1, 24), axes=(1, 0, 2))
data_train = np.transpose(data_train.reshape(5, -1, 24), axes=(1, 0, 2))

x_train = np.empty((feature_train.shape[0], 7*24))
y_train = np.empty((data_train.shape[0], 5*24))
for i in range(feature_train.shape[0]):
    x_train[i, :] = np.concatenate(feature_train[i, :, :])
    y_train[i, :] = np.concatenate(data_train[i, :, :])
    
x_train = x_train[:, 0:145]

x_test = np.concatenate(feature_test)
y_test = np.concatenate(data_test)
    
x_test = x_test[0:145]

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))

# The Output Layer :
NN_model.add(Dense(120, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mse', optimizer='adam')

NN_model.fit(x_train, y_train, epochs=500, batch_size=10, validation_split = 0.2)

NN_model.save("model_pm10_10l.h5")

#NN_model = load_model("model_pm10_5l.h5")

NN_model.summary()

y_predict = NN_model.predict(x_test.reshape(1, 145))
y_predict = y_predict.reshape((5, 24))

fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for i in range(5):
    axs[i].plot(np.exp(y_predict[i, :]), 'k--') 
    axs[i].plot(pm10[i, index_test_output], 'k:')
    axs[i].set_xlim([0, 23])
    axs[i].set_xlabel(r'Time (hour)')
    axs[i].set_ylim([0, .8*np.max(pm10)])
axs[0].set_ylabel(r'PM$_{10}$ ($\mu g/m^3$)')
