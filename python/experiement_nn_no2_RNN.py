# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 08:58:33 2019

@author: AdminisStrator
"""

import import_data
import functions
import numpy as np
#import pandas as pd
#from scipy import signal

#from keras.models import Sequential
#from keras.layers import Dense, LSTM
from keras.models import load_model
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint

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
# NN models
# =============================================================================
#model_rnn = list([])
model_rnn1 = load_model("model_no2_rnn_station1.h5")
model_rnn2 = load_model("model_no2_rnn_station2.h5")
model_rnn3 = load_model("model_no2_rnn_station3.h5")
model_rnn4 = load_model("model_no2_rnn_station4.h5")
model_rnn5 = load_model("model_no2_rnn_station5.h5")

# =============================================================================
# Pre-processing
# =============================================================================
no2 = np.array(no2).reshape(5, -1)

# Replace zeros with average
no2 = functions.replace_zero_with_average(no2)

# Log-normal transform
no2_transformed = functions.log_normal(no2)

# =============================================================================
# Start test
# =============================================================================
test_day_start = np.datetime64('2013-01-01', dtype='datetime64[D]')
test_day_end = np.datetime64('2016-01-01', dtype='datetime64[D]')

error = np.empty(((test_day_end-test_day_start).astype('int'), 5, 24))
for day in np.arange(test_day_start, test_day_end, dtype='datetime64[D]'):
    
    time_test = np.arange(day, day+np.timedelta64(1, dtype='datetime64[D]'), dtype='datetime64[h]')

    print(day)

    index_test = np.arange(np.where(date == time_test[0])[0], np.where(date == time_test[-1])[0]+1)

    data_test = no2_transformed[:, index_test]
    
    x_test = np.transpose(no2_transformed[:, index_test[0]-7*24-1:index_test[0]-1])
    y_test = no2_transformed[:, index_test]
    
    y_hat_test = np.empty((5, 24))
    for station in range(5):
        if station == 0:    
            y_hat_test[station, :] = model_rnn1.predict(x_test.reshape((1, 168, 5)))
        elif station == 1: 
            y_hat_test[station, :] = model_rnn2.predict(x_test.reshape((1, 168, 5)))
        elif station == 2: 
            y_hat_test[station, :] = model_rnn3.predict(x_test.reshape((1, 168, 5)))
        elif station == 3: 
            y_hat_test[station, :] = model_rnn4.predict(x_test.reshape((1, 168, 5)))
        elif station == 4: 
            y_hat_test[station, :] = model_rnn5.predict(x_test.reshape((1, 168, 5)))
            
    error[(day-test_day_start).astype('int'), :, :] = no2[:, index_test] - np.exp(y_hat_test)
    
np.save('no2_experiement_rnn', error)


mae = np.empty(5)
for station in range(5):
    mae[station] = np.mean(np.abs(error[:, station, :]))