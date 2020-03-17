# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:06:27 2019

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

data_no2 = [np.average(np.abs(no2_experiement_ind1_bs40_withoutNN_sgp[:, 1, :]), axis=(1)), np.average(np.abs(no2_experiement_b10_bs4_withoutNN[:, 1, :]), axis=(1))]

plt.figure(figsize=(5, 4))
plt.boxplot(data_no2, showfliers=False)
plt.xticks([1, 2], [r'Sparsification', r'Recursive update'])
plt.ylim([0, 50])
plt.ylabel(r'Absolute error ($\mu g/m^3$)')
plt.grid()
plt.tight_layout()
plt.savefig("no2_experiement_comparison.pdf", format='pdf')

data_pm10 = [np.average(np.abs(pm10_experiement_ind1_bs40_withoutNN_sgp[:, 1, :]), axis=(1)), np.average(np.abs(pm10_experiement_b10_bs4_withoutNN[:, 1, :]), axis=(1))]

plt.figure(figsize=(5, 4))
plt.boxplot(data_pm10, showfliers=False)
plt.xticks([1, 2], [r'Sparsification', r'Recursive update'])
plt.ylim([0, 50])
plt.ylabel(r'Absolute error ($\mu g/m^3$)')
plt.grid()
plt.tight_layout()
plt.savefig("pm10_experiement_comparison.pdf", format='pdf')

dataSize = np.array([8, 16, 24, 32, 40, 48, 56])
SGPSoD = np.array([11.8, 60, 186.2, 399.2, 760.2, 1439.8, 2487.8])
SGPFIC = np.array([29.3, 54.5, 94.0, 144.8, 207.5, 456.2, 683.7])
RGP = np.array([8.8, 17.7, 26.3, 35.1, 43.9, 52.8, 61.8])


plt.figure(figsize=(10, 4))
plt.plot(dataSize, SGPSoD, 'ko-', label=r'Exact likelihood')
plt.plot(dataSize, SGPFIC, 'ko:', label=r'Sparsification')
plt.plot(dataSize, RGP, 'ko--', label=r'Recursive update')
plt.xlim([0, 60])
plt.xticks(dataSize)
plt.ylim([0, 3000])
plt.xlabel(r'Utilized data size (weeks)')
plt.ylabel(r'Runtime (seconds)')
plt.gca().yaxis.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("runtime.pdf", format='pdf')

np.median(np.mean(np.abs(no2_experiement_b10_bs1_withoutNN), axis = 2), axis = 0)
