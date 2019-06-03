# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:57:44 2019

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

no2, nox, pm10, meteo, time = import_data.import_data()

# =============================================================================
# Data
# =============================================================================
plt.figure(figsize=(10,4))
plt.plot(time[np.logical_and(time >= np.datetime64('2010-04-01'), time <= np.datetime64('2010-04-30'))], 
              no2[0][[np.logical_and(time >= np.datetime64('2010-04-01'), time <= np.datetime64('2010-04-30'))]],
              'k')
plt.xticks(ticks=np.arange('2010-04-01', '2010-04-30', np.timedelta64(7, 'D'), dtype='datetime64[D]'), 
           labels=np.arange('2010-04-01', '2010-04-30', np.timedelta64(7, 'D'), dtype='datetime64[D]'))
plt.xlim([np.datetime64('2010-04-01'), np.datetime64('2010-04-30')])
plt.ylim([0,100])
plt.xlabel(r'Time')
plt.ylabel(r'NO$_2$ at Station 1 ($\mu g/m^3$)')
plt.gca().yaxis.grid(True)
plt.tight_layout()
plt.savefig("data_no2.pdf", format='pdf')

plt.figure(figsize=(10,4))
plt.plot(time[np.logical_and(time >= np.datetime64('2010-04-01'), time <= np.datetime64('2010-04-30'))], 
              pm10[0][[np.logical_and(time >= np.datetime64('2010-04-01'), time <= np.datetime64('2010-04-30'))]],
              'k')
plt.xticks(ticks=np.arange('2010-04-01', '2010-04-30', np.timedelta64(7, 'D'), dtype='datetime64[D]'), 
           labels=np.arange('2010-04-01', '2010-04-30', np.timedelta64(7, 'D'), dtype='datetime64[D]'))
plt.xlim([np.datetime64('2010-04-01'), np.datetime64('2010-04-30')])
plt.ylim([0,300])
plt.xlabel(r'Time')
plt.ylabel(r'PM$_{10}$ at Station 1 ($\mu g/m^3$)')
plt.gca().yaxis.grid(True)
plt.tight_layout()
plt.savefig("data_pm10.pdf", format='pdf')

plt.figure(figsize=(5,4))
plt.hist(no2[0], bins=50, normed=True, color='gray')
plt.xlim([0, 200])
plt.ylim([0, 0.05])
plt.xlabel(r'Original NO$_2$ data ($\mu g/m^3$)')
plt.ylabel(r'Normalized frequency')
plt.tight_layout()
plt.savefig("data_no2_hist.pdf", format='pdf')

plt.figure(figsize=(5,4))
plt.hist(pm10[0], bins=50, normed=True, color='gray')
plt.xlim([0, 300])
plt.ylim([0, 0.04])
plt.xlabel(r'Original PM$_{10}$ data ($\mu g/m^3$)')
plt.ylabel(r'Normalized frequency')
plt.tight_layout()
plt.savefig("data_pm10_hist.pdf", format='pdf')

pt_no2 = PowerTransformer()
pt_no2.fit(no2[0])

plt.figure(figsize=(5,4))
plt.hist(pt_no2.transform(no2[0][no2[0] != 0].reshape(-1, 1)), bins=50, normed=True, color='gray')
plt.xlim([-5, 5])
plt.ylim([0, 0.5])
plt.xlabel(r'Power transformed NO$_2$ data')
plt.ylabel(r'Normalized frequency')
plt.tight_layout()
plt.savefig("data_no2_hist_trans.pdf", format='pdf')

pt_pm10 = PowerTransformer()
pt_pm10.fit(pm10[0])

plt.figure(figsize=(5,4))
plt.hist(pt_pm10.transform(pm10[0][pm10[0] != 0].reshape(-1, 1)), bins=50, normed=True, color='gray')
plt.xlim([-5, 5])
plt.ylim([0, 0.6])
plt.xlabel(r'Power transformed PM$_{10}$ data')
plt.ylabel(r'Normalized frequency')
plt.tight_layout()
plt.savefig("data_pm10_hist_trans.pdf", format='pdf')

# =============================================================================
# ACF
# =============================================================================
acf_no2 = signal.correlate(no2[0], no2[0])

maxlag = 24*7*4
lag = np.arange(-maxlag, maxlag)
plt.figure(figsize=(5,4))
plt.plot(lag, acf_no2[np.size(no2[0])-maxlag:np.size(no2[0])+maxlag], 'k')
plt.xlim([-maxlag, maxlag])
plt.xticks(ticks=np.arange(-maxlag, maxlag+1, 24*7), labels=(np.arange(-maxlag, maxlag+1, 24*7)/(24*7)).astype('int'))
plt.xlabel(r'Lag (weeks)')
plt.ylabel(r'ACF of NO$_2$ at Station 1')
plt.tight_layout()
plt.savefig("acf_no2.pdf", format='pdf')

acf_pm10 = signal.correlate(pm10[0], pm10[0])

maxlag = 24*7*4
lag = np.arange(-maxlag, maxlag)
plt.figure(figsize=(5,4))
plt.plot(lag, acf_pm10[np.size(pm10[0])-maxlag:np.size(pm10[0])+maxlag], 'k')
plt.xlim([-maxlag, maxlag])
plt.xticks(ticks=np.arange(-maxlag, maxlag+1, 24*7), labels=(np.arange(-maxlag, maxlag+1, 24*7)/(24*7)).astype('int'))
plt.xlabel(r'Lag (weeks)')
plt.ylabel(r'ACF of PM$_{10}$ at Station 1')
plt.tight_layout()
plt.savefig("acf_pm10.pdf", format='pdf')

# =============================================================================
# PSD
# =============================================================================
fs = 1/3600
f, psd_no2 = signal.periodogram(no2[0], fs, axis=0)

plt.figure(figsize=(5,4))
plt.plot(f, psd_no2, 'k')
plt.xlim(0, 1/(3600*6))
plt.ylim(0, 3e10)
plt.xticks([fs/(24*7), fs/(24), fs/(12), fs/(8)], labels=['weekly', 'daily', '12-hours', '8-hours'])
plt.xlabel(r'Periodicity ($\frac{1}{f}$)')
plt.ylabel(r'PSD of NO$_2$ at Station 1')
plt.tight_layout()
plt.savefig("psd_no2.pdf", format='pdf')

f, psd_pm10 = signal.periodogram(pm10[0], fs, axis=0)

plt.figure(figsize=(5,4))
plt.plot(f, psd_pm10, 'k')
plt.xlim(0, 1/(3600*6))
plt.ylim(0, 3e10)
plt.xticks([fs/(24*7), fs/(24), fs/(12), fs/(8)], labels=['weekly', 'daily', '12-hours', '8-hours'])
plt.xlabel(r'Periodicity ($\frac{1}{f}$)')
plt.ylabel(r'PSD of PM$_{10}$ at Station 1')
plt.tight_layout()
plt.axes([.65, .6, .2, .2])
plt.plot(f, psd_pm10, 'k')
plt.xlim(0, 1/(3600*24*7*25))
plt.xticks([fs/(24*365)], labels=['annually'])
plt.savefig("psd_pm10.pdf", format='pdf')

# =============================================================================
# Covariance matrix
# =============================================================================
Sigma_no2 = functions.covmat(acf_no2, time[0:24*7])

plt.figure(figsize=(5,4))
plt.imshow(Sigma_no2, cmap='gray')
plt.colorbar()
plt.xticks(np.arange(0, 24*7, 24))
plt.yticks(np.arange(0, 24*7, 24))
plt.xlabel(r'Time (hours)')
plt.ylabel(r'Time (hours)')

Sigma_pm10 = functions.covmat(acf_pm10, time[0:24*7])

plt.figure(figsize=(5,4))
plt.imshow(Sigma_pm10, cmap='gray')
plt.colorbar()
plt.xticks(np.arange(0, 24*7, 24))
plt.yticks(np.arange(0, 24*7, 24))
plt.xlabel(r'Time (hours)')
plt.ylabel(r'Time (hours)')

xcf_no2 = signal.correlate(no2[0], no2[2])
Sigma_no2 = functions.xcovmat(xcf_no2, time[0:24], time[0:12])

plt.figure(figsize=(5,4))
plt.imshow(Sigma_no2, cmap='gray')
plt.colorbar()
plt.xticks(np.arange(0, 24, 24))
plt.yticks(np.arange(0, 24, 24))
plt.xlabel(r'Time (hours)')
plt.ylabel(r'Time (hours)')

#xcf_no2_pm10_11 = signal.correlate(no2[0], pm10[0])
#Sigma_no2_pm10_11 = functions.xcovmat(xcf_no2_pm10_11, time[0:24*7], time[0:24*7])
#
#plt.figure(figsize=(5,4))
#plt.imshow(Sigma_no2_pm10_11, cmap='gray')
#plt.colorbar()
#plt.xticks(np.arange(0, 24*7, 24))
#plt.yticks(np.arange(0, 24*7, 24))
#plt.xlabel(r'Time (hours)')
#plt.ylabel(r'Time (hours)')
#
#xcf_no2_temp = signal.correlate(no2[0], meteo[2])
#Sigma_no2_temp = functions.xcovmat(xcf_no2_temp, time[0:24*7], time[0:24*7])
#
#plt.figure(figsize=(5,4))
#plt.imshow(Sigma_no2_temp, cmap='gray')
#plt.colorbar()
#plt.xticks(np.arange(0, 24*7, 24))
#plt.yticks(np.arange(0, 24*7, 24))
#plt.xlabel(r'Time (hours)')
#plt.ylabel(r'Time (hours)')

# =============================================================================
# Meteorology data
# =============================================================================
hour = 11
time_date = time.astype('datetime64[D]')
time_busyday = np.is_busday(time_date)

this_no2 = no2[0][time_busyday]
this_no2 = this_no2[hour:-1:24]
index = this_no2 != 0
this_no2 = this_no2[index]
this_no2 = pt_no2.transform(this_no2.reshape(-1, 1))

this_temp = meteo[2][time_busyday]
this_temp = this_temp[hour:-1:24]
this_temp = this_temp[index]

plt.figure(figsize=(4,4))
plt.hexbin(this_no2, this_temp.reshape(-1, 1), gridsize=(10,10), cmap='gray')
plt.xlim(-2, 3)
plt.ylim(-15, 30)
plt.xlabel(r'NO$_2$ (power transformed)')
plt.ylabel(r'Temperature')
plt.tight_layout()
plt.savefig("no2vstemp.pdf", format='pdf')

plt.figure(figsize=(5,4))
fig, axes = plt.subplots(4, 1, sharex=True)
axes[0].hist(this_no2[this_temp >= 20], bins=10, density=True, color='gray')
axes[0].set_ylim([0, 1])
axes[0].set_yticks([])
axes[0].set_ylabel(r"$p(x\ |\ y \geq 20)$", rotation=0, labelpad=65)

axes[1].hist(this_no2[np.logical_and((this_temp < 20), (this_temp >= 10))], bins=10, density=True, color='gray')
axes[1].set_ylim([0, 1])
axes[1].set_yticks([])
axes[1].set_ylabel(r"$p(x\ |\ y \in [10, 20))$", rotation=0, labelpad=65)

axes[2].hist(this_no2[np.logical_and((this_temp < 10), (this_temp >= 0))], bins=10, density=True, color='gray')
axes[2].set_ylim([0, 1])
axes[2].set_yticks([])
axes[2].set_ylabel(r"$p(x\ |\ y \in [0, 10))$", rotation=0, labelpad=65)

axes[3].hist(this_no2[this_temp < 0], bins=10, density=True, color='gray')
axes[3].set_ylim([0, 1])
axes[3].set_yticks([])
axes[3].set_ylabel(r"$p(x\ |\ y \leq 0)$", rotation=0, labelpad=65)

plt.xlim(-2, 3)
plt.xlabel(r'NO$_2$ (power transformed)')
plt.tight_layout()
plt.savefig("no2giventemp.pdf", format='pdf')

# =============================================================================
# Pearson correlation
# =============================================================================
hour = 11
time_date = time.astype('datetime64[D]')
time_busyday = np.is_busday(time_date)

this_no2 = no2[0][time_busyday]
this_no2 = this_no2[hour:-1:24]

this_pm10 = pm10[0][time_busyday]
this_pm10 = this_pm10[hour:-1:24]

index = np.logical_and(this_no2 != 0, this_pm10 != 0)

this_no2 = this_no2[index]
this_no2 = pt_no2.transform(this_no2.reshape(-1, 1))
this_no2 = this_no2.reshape(-1,)

this_pm10 = this_pm10[index]
this_pm10 = pt_pm10.transform(this_pm10.reshape(-1, 1))
this_pm10 = this_pm10.reshape(-1,)

this_rad = meteo[0][time_busyday]
this_rad = this_rad[hour:-1:24]
this_rad = this_rad[index]

this_rh = meteo[1][time_busyday]
this_rh = this_rh[hour:-1:24]
this_rh = this_rh[index]

this_temp = meteo[2][time_busyday]
this_temp = this_temp[hour:-1:24]
this_temp = this_temp[index]

this_windd = meteo[3][time_busyday]
this_windd = this_windd[hour:-1:24]
this_windd = np.deg2rad(this_windd[index])

this_winds = meteo[4][time_busyday]
this_winds = this_winds[hour:-1:24]
this_winds = this_winds[index]

this_data = np.stack((this_no2, this_pm10, this_rad, this_rh, this_temp, this_winds))

this_corr = np.corrcoef(this_data)

r_cs = np.corrcoef(np.cos(this_windd), np.sin(this_windd))[0, 1]
for i in range(6):
    r_yc = np.corrcoef(np.cos(this_windd), this_data[i, :])[0, 1]
    r_ys = np.corrcoef(np.sin(this_windd), this_data[i, :])[0, 1]
    r = np.sqrt((np.square(r_yc) + np.square(r_ys) - 2*r_yc*r_ys*r_cs)/(1 - np.square(r_cs)))
    print(r)


