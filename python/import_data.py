# -*- coding: utf-8 -*-
"""
Created on Wed May  8 07:56:10 2019

@author: Administrator
"""

import scipy.io
import numpy as np

def import_data():
    data = scipy.io.loadmat('..\data\data_new.mat')
    
    no2 = []
    no2.append(data['no2_essingeleden_060101_151231'])
    no2.append(data['no2_hornsgatan_060101_151231'])
    no2.append(data['no2_norrlandsgatan_060101_151231'])
    no2.append(data['no2_sveavagen_060101_151231'])
    no2.append(data['no2_torkel_060101_151231'])
    
    nox = []
    nox.append(data['nox_essingeleden_060101_151231'])
    nox.append(data['nox_hornsgatan_060101_151231'])
    nox.append(data['nox_norrlandsgatan_060101_151231'])
    nox.append(data['nox_sveavagen_060101_151231'])
    nox.append(data['nox_torkel_060101_151231'])
    
    pm10 = []
    pm10.append(data['pm10_essingeleden_060101_151231'])
    pm10.append(data['pm10_hornsgatan_060101_151231'])
    pm10.append(data['pm10_norrlandsgatan_060101_151231'])
    pm10.append(data['pm10_sveavagen_060101_151231'])
    pm10.append(data['pm10_torkel_060101_151231'])
    
    meteo = []
    meteo.append(data['radiation_norr_060101_151231'])
    meteo.append(data['rh_norr_060101_151231'])
    meteo.append(data['temperature_norr_060101_151231'])
    meteo.append(data['winddirection_norr_060101_151231'])
    meteo.append(data['windspeed_norr_060101_151231'])
    
    time = np.arange('2006-01-01', '2016-01-01', dtype='datetime64[h]')
    
    return no2, nox, pm10, meteo, time