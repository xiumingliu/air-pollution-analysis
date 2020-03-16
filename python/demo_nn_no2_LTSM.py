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
from keras.layers import Dense, LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

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

# Replace zeros with average
no2 = functions.replace_zero_with_average(no2)

# Log-normal transform
no2_transformed = functions.log_normal(no2)

# =============================================================================
# Configuration
# =============================================================================
time_test = np.arange('2014-03-11', '2014-03-12', dtype='datetime64[h]')

time_train = np.arange('2006-01-01', '2013-01-01', dtype='datetime64[h]')

index_test = np.arange(np.where(time == time_test[0])[0], np.where(time == time_test[-1])[0]+1)
index_train = np.arange(np.where(time == time_train[0])[0], np.where(time == time_train[-1])[0]+1)

data_test = no2_transformed[:, index_test]
data_train = no2_transformed[:, index_train]

station = 5

## =============================================================================
## NN for 24h air quality prediction
## =============================================================================

x_train = np.empty((2500, 24*7, 5))
y_train = np.empty((2500, 24, 5))

x_test = np.transpose(no2_transformed[:, index_test[0]-7*24-1:index_test[0]-1])
y_test = no2_transformed[station-1, index_test]

for i in range(2500):
    x_train[i, :, :] = np.transpose(data_train[:, (i)*24:((i)*24+24*7)])
    y_train[i, :, :] = np.transpose(data_train[:, (i)*24+24*7:((i)*24+24*7+24)])

model_lstm = Sequential()

# The Input Layer :
model_lstm.add(LSTM(24, input_shape=(x_train.shape[1], x_train.shape[2])))
model_lstm.add(Dense(24))
model_lstm.compile(loss='mae', optimizer='adam')
model_lstm.summary()

history = model_lstm.fit(x_train, y_train[:, :, station-1], epochs=50, batch_size=72)


model_lstm.save("model_no2_lstm_station5.h5")

#model_lstm = load_model("model_no2_lstm.h5.h5")

y_hat_test = model_lstm.predict(x_test.reshape((1, 168, 5)))

plt.figure()
plt.plot(no2[station-1, index_test], 'k--')
plt.plot(np.exp(y_hat_test)[0], 'k:')
