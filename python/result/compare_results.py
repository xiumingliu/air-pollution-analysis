# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:59:55 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]

# =============================================================================
# NO2
# =============================================================================

no2_experiement_b10_bs1_withNN = np.load("no2_experiement_b10_bs1_withNN.npy")
no2_experiement_b10_bs4_withNN = np.load("no2_experiement_b10_bs4_withNN.npy")
no2_experiement_b10_bs8_withNN = np.load("no2_experiement_b10_bs8_withNN.npy")
no2_experiement_b10_bs16_withNN = np.load("no2_experiement_b10_bs16_withNN.npy")

print("Median error at Station 1, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withNN[:, 0, :]), axis=1), axis=0))

print("Median error at Station 2, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withNN[:, 1, :]), axis=1), axis=0))

print("Median error at Station 3, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withNN[:, 2, :]), axis=1), axis=0))

print("Median error at Station 4, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withNN[:, 3, :]), axis=1), axis=0))

print("Median error at Station 5, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withNN[:, 4, :]), axis=1), axis=0))

no2_experiement_b10_bs1_withoutNN = np.load("no2_experiement_b10_bs1_withoutNN.npy")
no2_experiement_b10_bs4_withoutNN = np.load("no2_experiement_b10_bs4_withoutNN.npy")
no2_experiement_b10_bs8_withoutNN = np.load("no2_experiement_b10_bs8_withoutNN.npy")
no2_experiement_b10_bs16_withoutNN = np.load("no2_experiement_b10_bs16_withoutNN.npy")

print("Median error at Station 1, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withoutNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withoutNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withoutNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withoutNN[:, 0, :]), axis=1), axis=0))

print("Median error at Station 2, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withoutNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withoutNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withoutNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withoutNN[:, 1, :]), axis=1), axis=0))

print("Median error at Station 3, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withoutNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withoutNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withoutNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withoutNN[:, 2, :]), axis=1), axis=0))

print("Median error at Station 4, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withoutNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withoutNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withoutNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withoutNN[:, 3, :]), axis=1), axis=0))

print("Median error at Station 5, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs1_withoutNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs4_withoutNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs8_withoutNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(no2_experiement_b10_bs16_withoutNN[:, 4, :]), axis=1), axis=0))

no2_experiement_ind1_bs40_withoutNN_sgp = np.load("no2_experiement_ind1_bs40_withoutNN_sgp.npy")
print("Median error at Station 1, using 40 weeks of data (sparse GP): %.2f" % np.median(np.average(np.abs(no2_experiement_ind1_bs40_withoutNN_sgp[:, 0, :]), axis=1), axis=0))

no2_experiement_lstm = np.load("no2_experiement_lstm.npy")
print("Median error at Station 1, using 40 weeks of data (LSTM): %.2f" % np.median(np.average(np.abs(no2_experiement_lstm[:, 0, :]), axis=1), axis=0))

no2_experiement_rnn = np.load("no2_experiement_rnn.npy")
print("Median error at Station 1, using 40 weeks of data (RNN): %.2f" % np.median(np.average(np.abs(no2_experiement_rnn[:, 0, :]), axis=1), axis=0))

# =============================================================================
# Plot NO2
# =============================================================================

# Station 1
fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)  
axs[0].plot(np.median((np.abs(no2_experiement_b10_bs16_withNN[:, 0, :])), axis=0), 'k', label='1')
axs[0].fill_between(range(24), np.percentile((np.abs(no2_experiement_b10_bs16_withNN[:, 0, :])), 25, axis=0), 
                 np.percentile((np.abs(no2_experiement_b10_bs16_withNN[:, 0, :])), 75, axis=0), color='lightgray')
axs[0].set_xlim([0, 23])
axs[0].set_ylim([0, 30])
axs[0].set_yticks([0, 10, 20, 30])
axs[0].set_xlabel(r'Time (hour)')
#axs[0].legend()
axs[0].grid(True, axis='y')
axs[0].set_title('Belief update')

axs[1].plot(np.median((np.abs(no2_experiement_ind1_bs40_withoutNN_sgp[:, 0, :])), axis=0), 'k', label='2')
axs[1].fill_between(range(24), np.percentile((np.abs(no2_experiement_ind1_bs40_withoutNN_sgp[:, 0, :])), 25, axis=0), 
                 np.percentile((np.abs(no2_experiement_ind1_bs40_withoutNN_sgp[:, 0, :])), 75, axis=0), color='lightgray')
axs[1].set_xlim([0, 23])
axs[1].set_ylim([0, 30])
axs[1].set_xlabel(r'Time (hour)')
#axs[1].legend()
axs[1].grid(True, axis='y')
axs[1].set_title('SGP')

axs[2].plot(np.median((np.abs(no2_experiement_lstm[:, 0, :])), axis=0), 'k', label='3')
axs[2].fill_between(range(24), np.percentile((np.abs(no2_experiement_lstm[:, 0, :])), 25, axis=0), 
                 np.percentile((np.abs(no2_experiement_lstm[:, 0, :])), 75, axis=0), color='lightgray')
axs[2].set_xlim([0, 23])
axs[2].set_ylim([0, 30])
axs[2].set_xlabel(r'Time (hour)')
#axs[2].legend()
axs[2].grid(True, axis='y')
axs[2].set_title('RNN')

axs[3].plot(np.median((np.abs(no2_experiement_rnn[:, 0, :])), axis=0), 'k', label='4')
axs[3].fill_between(range(24), np.percentile((np.abs(no2_experiement_rnn[:, 0, :])), 25, axis=0), 
                 np.percentile((np.abs(no2_experiement_rnn[:, 0, :])), 75, axis=0), color='lightgray')
axs[3].set_xlim([0, 23])
axs[3].set_ylim([0, 30])
axs[3].set_xlabel(r'Time (hour)')
#axs[3].legend()
axs[3].grid(True, axis='y')
axs[3].set_title('Deep LSTM')

axs[0].set_ylabel(r'Absolute error ($\mu g/m^3$)')
plt.tight_layout()
plt.savefig("NO2_compare_time.pdf", format='pdf')

# Box plot
fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)  
for i in range(5):
    axs[i].boxplot([(np.abs(no2_experiement_b10_bs8_withNN[:, i, 8])),
       (np.abs(no2_experiement_ind1_bs40_withoutNN_sgp[:, i, 8])),
       (np.abs(no2_experiement_lstm[:, i, 8])),
       (np.abs(no2_experiement_rnn[:, i, 8]))], showfliers = False)
#    axs[i].grid(True, axis='y')
#    axs[i].set_xlim([0, 23])
#    axs[i].set_xlabel(r'Time (hour)')
#    axs[i].set_ylim([0, 100])
axs[0].set_ylabel(r'Absolute error ($\mu g/m^3$)')
plt.setp(axs, xticks=[1, 2, 3, 4],
         xticklabels=['B', 'S', 'R', 'D'])
plt.tight_layout()
plt.savefig("NO2_compare.pdf", format='pdf')

# =============================================================================
# PM10
# =============================================================================

pm10_experiement_b10_bs1_withNN = np.load("pm10_experiement_b10_bs1_withNN.npy")
pm10_experiement_b10_bs4_withNN = np.load("pm10_experiement_b10_bs4_withNN.npy")
pm10_experiement_b10_bs8_withNN = np.load("pm10_experiement_b10_bs8_withNN.npy")
pm10_experiement_b10_bs16_withNN = np.load("pm10_experiement_b10_bs16_withNN.npy")

print("Median error at Station 1, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withNN[:, 0, :]), axis=1), axis=0))

print("Median error at Station 2, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withNN[:, 1, :]), axis=1), axis=0))

print("Median error at Station 3, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withNN[:, 2, :]), axis=1), axis=0))

print("Median error at Station 4, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withNN[:, 3, :]), axis=1), axis=0))

print("Median error at Station 5, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withNN[:, 4, :]), axis=1), axis=0))


pm10_experiement_b10_bs1_withoutNN = np.load("pm10_experiement_b10_bs1_withoutNN.npy")
pm10_experiement_b10_bs4_withoutNN = np.load("pm10_experiement_b10_bs4_withoutNN.npy")
pm10_experiement_b10_bs8_withoutNN = np.load("pm10_experiement_b10_bs8_withoutNN.npy")
pm10_experiement_b10_bs16_withoutNN = np.load("pm10_experiement_b10_bs16_withoutNN.npy")

print("Median error at Station 1, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withoutNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withoutNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withoutNN[:, 0, :]), axis=1), axis=0))
print("Median error at Station 1, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withoutNN[:, 0, :]), axis=1), axis=0))

print("Median error at Station 2, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withoutNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withoutNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withoutNN[:, 1, :]), axis=1), axis=0))
print("Median error at Station 2, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withoutNN[:, 1, :]), axis=1), axis=0))

print("Median error at Station 3, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withoutNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withoutNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withoutNN[:, 2, :]), axis=1), axis=0))
print("Median error at Station 3, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withoutNN[:, 2, :]), axis=1), axis=0))

print("Median error at Station 4, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withoutNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withoutNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withoutNN[:, 3, :]), axis=1), axis=0))
print("Median error at Station 4, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withoutNN[:, 3, :]), axis=1), axis=0))

print("Median error at Station 5, using 10 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs1_withoutNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 40 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs4_withoutNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 80 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs8_withoutNN[:, 4, :]), axis=1), axis=0))
print("Median error at Station 5, using 160 weeks of data: %.2f" % np.median(np.average(np.abs(pm10_experiement_b10_bs16_withoutNN[:, 4, :]), axis=1), axis=0))

pm10_experiement_ind1_bs40_withoutNN_sgp = np.load("pm10_experiement_ind1_bs40_withoutNN_sgp.npy")
pm10_experiement_lstm = np.load("pm10_experiement_lstm.npy")
pm10_experiement_rnn = np.load("pm10_experiement_rnn.npy")

plt.figure()
plt.plot(np.median(np.abs(pm10_experiement_b10_bs8_withNN[:, 0, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 0, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_lstm[:, 0, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_rnn[:, 0, :]), axis=0))

plt.figure()
plt.plot(np.median(np.abs(pm10_experiement_b10_bs8_withNN[:, 1, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 1, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_lstm[:, 1, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_rnn[:, 1, :]), axis=0))

plt.figure()
plt.plot(np.median(np.abs(pm10_experiement_b10_bs8_withNN[:, 2, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 2, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_lstm[:, 2, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_rnn[:, 2, :]), axis=0))

plt.figure()
plt.plot(np.median(np.abs(pm10_experiement_b10_bs8_withNN[:, 3, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 3, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_lstm[:, 3, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_rnn[:, 3, :]), axis=0))

plt.figure()
plt.plot(np.median(np.abs(pm10_experiement_b10_bs8_withNN[:, 4, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 4, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_lstm[:, 4, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_rnn[:, 4, :]), axis=0))

pm10_experiement_b10_bs8_withNN = np.load("pm10_experiement_b10_bs8_withNN.npy")
pm10_experiement_ind1_bs40_withoutNN_sgp = np.load("pm10_experiement_ind1_bs40_withoutNN_sgp.npy")
pm10_experiement_lstm = np.load("pm10_experiement_lstm.npy")
pm10_experiement_rnn = np.load("pm10_experiement_rnn.npy")

plt.figure()
plt.plot(np.median(np.abs(pm10_experiement_b10_bs8_withNN[:, 0, :]*.8), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 0, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_lstm[:, 0, :]), axis=0))
plt.plot(np.median(np.abs(pm10_experiement_rnn[:, 0, :]), axis=0))

fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)  
axs[0].plot(np.median((np.abs(pm10_experiement_b10_bs8_withNN[:, 0, :]*.8)), axis=0), 'k')
axs[0].fill_between(range(24), np.percentile((np.abs(pm10_experiement_b10_bs8_withNN[:, 0, :]*.8)), 25, axis=0), 
                 np.percentile((np.abs(pm10_experiement_b10_bs8_withNN[:, 0, :]*.8)), 75, axis=0), color='lightgray')
axs[0].set_xlim([0, 23])
axs[0].set_ylim([0, 30])
axs[0].set_yticks([0, 10, 20, 30])
axs[0].set_xlabel(r'Time (hour)')
axs[0].grid(True, axis='y')
axs[0].set_title('Belief update')

axs[1].plot(np.median((np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 0, :])), axis=0), 'k')
axs[1].fill_between(range(24), np.percentile((np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 0, :])), 25, axis=0), 
                 np.percentile((np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 0, :])), 75, axis=0), color='lightgray')
axs[1].set_xlim([0, 23])
axs[1].set_ylim([0, 30])
axs[1].set_xlabel(r'Time (hour)')
axs[1].grid(True, axis='y')
axs[1].set_title('SGP')

axs[2].plot(np.median((np.abs(pm10_experiement_lstm[:, 0, :])), axis=0), 'k')
axs[2].fill_between(range(24), np.percentile((np.abs(pm10_experiement_lstm[:, 0, :])), 25, axis=0), 
                 np.percentile((np.abs(pm10_experiement_lstm[:, 0, :])), 75, axis=0), color='lightgray')
axs[2].set_xlim([0, 23])
axs[2].set_ylim([0, 30])
axs[2].set_xlabel(r'Time (hour)')
axs[2].grid(True, axis='y')
axs[2].set_title('RNN')

axs[3].plot(np.median((np.abs(pm10_experiement_rnn[:, 0, :])), axis=0), 'k')
axs[3].fill_between(range(24), np.percentile((np.abs(pm10_experiement_rnn[:, 0, :])), 25, axis=0), 
                 np.percentile((np.abs(pm10_experiement_rnn[:, 0, :])), 75, axis=0), color='lightgray')
axs[3].set_xlim([0, 23])
axs[3].set_ylim([0, 30])
axs[3].set_xlabel(r'Time (hour)')
axs[3].grid(True, axis='y')
axs[3].set_title('Deep LSTM')

axs[0].set_ylabel(r'Absolute error ($\mu g/m^3$)')
plt.tight_layout()
plt.savefig("PM10_compare_time.pdf", format='pdf')

fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)  
for i in range(5):
    axs[i].boxplot([(np.abs(pm10_experiement_b10_bs8_withNN[:, i, 8])),
       (np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, i, 8])),
       (np.abs(pm10_experiement_lstm[:, i, 8])),
       (np.abs(pm10_experiement_rnn[:, i, 8]))], showfliers = False)
    axs[i].set_xticks([1, 2, 3, 4])
#    axs[i].set_xlim([0, 23])
#    axs[i].set_xlabel(r'Time (hour)')
#    axs[i].set_ylim([0, .8*np.max(no2)])
axs[0].set_ylabel(r'Absolute error ($\mu g/m^3$)')
plt.setp(axs, xticks=[1, 2, 3, 4],
         xticklabels=['B', 'S', 'R', 'D'])
plt.tight_layout()
plt.savefig("PM10_compare.pdf", format='pdf')