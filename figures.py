# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:14:16 2021

@author: janousu
"""

import sys
sys.path.append(r'C:\SpaFHy_v1_Pallas')

import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm

import pickle
from netCDF4 import Dataset #, date2num

from datetime import date


saveplots = True
today = date.today()

# import model 
# import spafhy

eps = np.finfo(float).eps

# change working dir
os.chdir(r'C:\SpaFHy_v1_Pallas\FigsC3')

# results file
ncf_file = r'C:\SpaFHy_v1_Pallas\results\C3.nc'

# pickled model object
pk_file = r'C:\SpaFHy_v1_Pallas\results\C3model.pk'

""" load pre-computed results for C3 """
# spa instance
with open(pk_file, 'rb') as ff:
    spa, Qmeas, FORC = pickle.load(ff)

# get time-index when results start
ix = 1 + np.where(FORC.index == spa.pgen['spinup_end'])[0][0]
tvec = FORC.index # date vector


gis = spa.GisData
twi = gis['twi']
LAIc = gis['LAI_conif']
LAId = gis['LAI_decid']
LAIs = gis['LAI_shrub']
LAIg = gis['LAI_grass']
soil = gis['soilclass']

# soil type indexes
peat_ix = np.where(soil == 4)
med_ix = np.where(soil == 2)
coarse_ix = np.where(soil == 1)

# indices for high and low twi
htwi_ix = np.where(twi > 12)
ltwi_ix = np.where(twi < 7)

# open link to results in netcdf:
dat = Dataset(ncf_file, 'r')

# results per sub-model
cres = dat['cpy']   # canopy -submodel
bres = dat['bu']    # bucket -submodel
tres = dat['top']   # topmodel - submodel

#%%

# sar soil moisture plots
import pandas as pd

# define a big catchment mask
from spafhy_io import read_AsciiGrid

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
kenttarova_loc = list([int(kenttarova_loc[0]), int(kenttarova_loc[1])])

# reading sar data
sar_path = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\processed\16m_nc_spafhy_pallas\SAR_PALLAS_2019_mask2_16m_direct_catchment_mean4.nc'
sar = Dataset(sar_path, 'r')

sar_wliq = sar['soilmoisture']*gis['cmask']/100
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']
spa_S = tres['S']

dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 
dates_spa = tvec

#spa dates to match sar dates
date_in_spa = []
for i in range(len(dates_sar)):
    ix = np.where(dates_spa == dates_sar[i])[0][0]
    date_in_spa.append(ix)
spa_wliq = spa_wliq[date_in_spa,:,:]
spa_wliq_top = spa_wliq_top[date_in_spa,:,:]

# driest and wettest days
spasum = np.nansum(spa_wliq, axis=(1,2))

# index in sar data
day_low = int(np.where(spasum == np.nanmin(spasum))[0])
day_hi = int(np.where(spasum == np.nanmax(spasum))[0])
#sar_low = 43
#sar_hi = 20
# day in sar data
low_date = dates_sar[day_low].strftime("%Y-%m-%d")
hi_date = dates_sar[day_hi].strftime("%Y-%m-%d")

# cropping for plots
xcrop = np.arange(20,170)
ycrop = np.arange(20,250)
sar_wliq = sar_wliq[:,ycrop,:]
sar_wliq = sar_wliq[:,:,xcrop]
spa_wliq = spa_wliq[:,ycrop,:]
spa_wliq = spa_wliq[:,:,xcrop]
spa_wliq_top = spa_wliq_top[:,ycrop,:]
spa_wliq_top = spa_wliq_top[:,:,xcrop]

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16,12));
ax1 = axs[0][0]
ax2 = axs[0][1]
ax3 = axs[1][0]
ax4 = axs[1][1]
ax5 = axs[0][2]
ax6 = axs[1][2]

#fig.suptitle('Volumetric water content', fontsize=15)

im1 = ax1.imshow(sar_wliq[day_hi,:,:], cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax1.title.set_text('SAR')
#ax1.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im2 = ax2.imshow(spa_wliq[day_hi, :,:], cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax2.title.set_text('SPAFHY rootzone')
ax2.text(10, -15, f'Wet day : {hi_date}', fontsize=15)
#ax2.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im3 = ax3.imshow(sar_wliq[day_low, :,:], cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax3.title.set_text('SAR')

im4 = ax4.imshow(spa_wliq[day_low, :,:], cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax4.title.set_text('SPAFHY rootzone')
ax4.text(10, -15, f'Dry day : {low_date}', fontsize=15)


im5 = ax5.imshow(spa_wliq_top[day_hi, :, :], cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax5.title.set_text('SPAFHY topsoil')

im6 = ax6.imshow(spa_wliq_top[day_low, :,:], cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax6.title.set_text('SPAFHY topsoil')

ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")
ax5.axis("off")
ax6.axis("off")

plt.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
bar1 = fig.colorbar(im1, cax=cbar_ax)


#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.8, 0.2, 0.02, 0.6])
#bar2 = fig.colorbar(im6, cax=cbar_ax)

#ax5.plot([1.1, 1.1], [1.1, -1.2], color='black', lw=1, transform=ax2.transAxes, clip_on=False)


if saveplots == True:
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_{today}.pdf')
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_{today}.png')
    
    
#%%

# normalized plots

# mean of each pixel
spamean = np.nanmean(spa_wliq, axis=0)
spatopmean = np.nanmean(spa_wliq_top, axis=0)
sarmean = np.nanmean(sar_wliq, axis=0)

# median day of total sums
spasum = np.nansum(spa_wliq, axis=(1,2))
spamedian = spa_wliq[np.where(np.sort(np.nansum(spa_wliq, axis=(1,2)))[(int(len(spasum)/2))] == spasum)[0][0],:,:]
sarsum = np.nansum(sar_wliq, axis=(1,2))
sarmedian = sar_wliq[np.where(np.sort(np.nansum(sar_wliq, axis=(1,2)))[(int(len(sarsum)/2))] == sarsum)[0][0],:,:]
spatopsum = np.nansum(spa_wliq_top, axis=(1,2))
spatopmedian = spa_wliq[np.where(np.sort(np.nansum(spa_wliq_top, axis=(1,2)))[(int(len(spatopsum)/2))] == spatopsum)[0][0],:,:]

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16,12));
ax1 = axs[0][0]
ax2 = axs[0][1]
ax3 = axs[1][0]
ax4 = axs[1][1]
ax5 = axs[0][2]
ax6 = axs[1][2]

#fig.suptitle('Volumetric water content', fontsize=15)

im1 = ax1.imshow(sar_wliq[day_hi,:,:]/sarmean, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax1.title.set_text('SAR')
#ax1.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im2 = ax2.imshow(spa_wliq[day_hi, :,:]/spamean, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax2.title.set_text('SPAFHY rootzone')
ax2.text(10, -15, f'Wet day : {hi_date}', fontsize=15)
#ax2.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im3 = ax3.imshow(sar_wliq[day_low, :,:]/sarmean, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax3.title.set_text('SAR')

im4 = ax4.imshow(spa_wliq[day_low, :,:]/spamean, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax4.title.set_text('SPAFHY rootzone')
ax4.text(10, -15, f'Dry day : {low_date}', fontsize=15)


im5 = ax5.imshow(spa_wliq_top[day_hi, :, :]/spatopmean, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax5.title.set_text('SPAFHY topsoil')

im6 = ax6.imshow(spa_wliq_top[day_low, :,:]/spatopmean, cmap='coolwarm_r', vmin=0.0, vmax=2.0, aspect='equal')
ax6.title.set_text('SPAFHY topsoil')

ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")
ax5.axis("off")
ax6.axis("off")

plt.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
bar1 = fig.colorbar(im1, cax=cbar_ax)


#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.8, 0.2, 0.02, 0.6])
#bar2 = fig.colorbar(im6, cax=cbar_ax)

#ax5.plot([1.1, 1.1], [1.1, -1.2], color='black', lw=1, transform=ax2.transAxes, clip_on=False)


if saveplots == True:
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_norm_{today}.pdf')
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_norm_{today}.png')


#%%

# point examples from mineral and openmire

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
k_loc = list([int(kenttarova_loc[0]), int(kenttarova_loc[1])])
l_loc = [60, 55]
sar_wliq = sar['soilmoisture']*gis['cmask']/100
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']
spa_S = tres['S']

dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 
dates_spa = tvec

#spa dates to match sar dates
date_in_spa = []
for i in range(len(dates_sar)):
    ix = np.where(dates_spa == dates_sar[i])[0][0]
    date_in_spa.append(ix)
spa_wliq = spa_wliq[date_in_spa,:,:]
spa_wliq_top = spa_wliq_top[date_in_spa,:,:]

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,8));
ax1 = axs[0]
ax2 = axs[1]

#fig.suptitle('Volumetric water content', fontsize=15)

im1 = ax1.plot(sar_wliq[:,k_loc[0],k_loc[1]])
ax1.plot(spa_wliq[:,k_loc[0],k_loc[1]])
ax1.plot(spa_wliq_top[:,k_loc[0],k_loc[1]])
ax1.title.set_text('Mineral')
ax1.legend(['SAR', 'SpaFHy rootzone', 'SpaFHy top'], ncol = 3)


im2 = ax2.plot(sar_wliq[:,l_loc[0],l_loc[1]])
ax2.plot(spa_wliq[:,l_loc[0],l_loc[1]])
ax2.plot(spa_wliq_top[:,l_loc[0],l_loc[1]])
ax2.title.set_text('Open mire')
ax2.legend(['SAR', 'SpaFHy rootzone', 'SpaFHy top'], ncol = 3)


#%%

# Q-Q plots

import numpy as np 
import pylab 
import scipy.stats as stats
import pandas as pd

from spafhy_io import read_AsciiGrid

sar_wliq = sar['soilmoisture']*gis['cmask']/100
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']
spa_S = tres['S']

dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 
dates_spa = tvec

#spa dates to match sar dates
date_in_spa = []
for i in range(len(dates_sar)):
    ix = np.where(dates_spa == dates_sar[i])[0][0]
    date_in_spa.append(ix)
spa_wliq = spa_wliq[date_in_spa,:,:]
spa_wliq_top = spa_wliq_top[date_in_spa,:,:]

sar_flat_dry = sar_wliq[day_low,:,:].flatten()
#sar_flat[np.where(sar_flat <= 0)] = np.nan
spa_flat_dry = spa_wliq[day_low,:,:].flatten()
spa_top_flat_dry = spa_wliq_top[day_low,:,:].flatten()

flat_pd = pd.DataFrame()
flat_pd['sar_dry'] = sar_flat_dry
flat_pd['spa_dry'] = spa_flat_dry
flat_pd['spa_top_dry'] = spa_top_flat_dry

sar_flat_wet = sar_wliq[day_hi,:,:].flatten()
#sar_flat[np.where(sar_flat <= 0)] = np.nan
spa_flat_wet = spa_wliq[day_hi,:,:].flatten()
spa_top_flat_wet = spa_wliq_top[day_hi,:,:].flatten()

inb = 35
sar_flat_inb = sar_wliq[inb,:,:].flatten()
spa_flat_inb = spa_wliq[inb,:,:].flatten()
spa_top_flat_inb = spa_wliq_top[inb,:,:].flatten()

flat_pd['sar_wet'] = sar_flat_wet
flat_pd['spa_wet'] = spa_flat_wet
flat_pd['spa_top_wet'] = spa_top_flat_wet

flat_pd['sar_inb'] = sar_flat_inb
flat_pd['spa_inb'] = spa_flat_inb
flat_pd['spa_top_inb'] = spa_top_flat_inb

flat_pd = flat_pd.loc[np.isfinite(flat_pd['sar_dry']) & np.isfinite(flat_pd['spa_dry']) & np.isfinite(flat_pd['spa_top_dry'])]
#flat_pd = flat_pd.loc[(flat_pd['sar'] > 0) & (flat_pd['sar'] < 1)]

#g = sns.scatterplot(flat_pd['sar'], flat_pd['spa'], alpha=0.0001, s=2)
#g.set(ylim=(-0.1, 1.0))
#g.set(xlim=(-0.1, 1.0))


# Plotting
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12,8));
ax1 = axs[0][0]
ax2 = axs[0][1]
ax3 = axs[1][0]
ax4 = axs[1][1]
ax5 = axs[0][2]
ax6 = axs[1][2]

x1 = sns.regplot(ax=ax1, x=flat_pd['sar_dry'], y=flat_pd['spa_dry'], scatter_kws={'s':1, 'alpha':1.0}, line_kws={"color": "red"})
ax1.set(ylim=(0, 1))
ax1.set(xlim=(0, 1))
#ax1.set_title('Dry day')

x2 = sns.regplot(ax=ax2, x=flat_pd['sar_wet'], y=flat_pd['spa_wet'], scatter_kws={'s':1, 'alpha':1.0}, line_kws={"color": "red"})
ax2.set(ylim=(0, 1))
ax2.set(xlim=(0, 1))
#ax2.set_title('Wet day')

x3 = sns.regplot(ax=ax3, x=flat_pd['sar_dry'], y=flat_pd['spa_top_dry'], scatter_kws={'s':1, 'alpha':1.0}, line_kws={"color": "red"})
ax3.set(ylim=(0, 1))
ax3.set(xlim=(0, 1))
#ax3.set_title('Dry day')

x4 = sns.regplot(ax=ax4, x=flat_pd['sar_wet'], y=flat_pd['spa_top_wet'], scatter_kws={'s':1, 'alpha':1.0}, line_kws={"color": "red"})
ax4.set(ylim=(0, 1))
ax4.set(xlim=(0, 1))
#ax4.set_title('Wet day')

x5 = sns.regplot(ax=ax5, x=flat_pd['sar_inb'], y=flat_pd['spa_inb'], scatter_kws={'s':1, 'alpha':1.0}, line_kws={"color": "red"})
ax5.set(ylim=(0, 1))
ax5.set(xlim=(0, 1))
#ax4.set_title('Wet day')

x6 = sns.regplot(ax=ax6, x=flat_pd['sar_inb'], y=flat_pd['spa_top_inb'], scatter_kws={'s':1, 'alpha':1.0}, line_kws={"color": "red"})
ax6.set(ylim=(0, 1))
ax6.set(xlim=(0, 1))
#ax4.set_title('Wet day')

fig.suptitle('SpaFHy v1')

if saveplots == True:
    plt.savefig(f'sar_spa_qq_wetdry_{today}.pdf')
    plt.savefig(f'sar_spa_qq_wetdry_{today}.png')


#%%

# QQ plots of normalized

sar_flat = (sar_wliq[day_hi,:,:]/sarmean).flatten()
#sar_flat[np.where(sar_flat <= 0)] = np.nan
spa_flat = (spa_wliq[day_hi,:,:]/spamean).flatten()
spa_top_flat = (spa_wliq_top[day_hi,:,:]/spatopmean).flatten()

flat_pd = pd.DataFrame()
flat_pd['sar'] = sar_flat
flat_pd['spa'] = spa_flat
flat_pd['spa_top'] = spa_top_flat
flat_pd = flat_pd.loc[np.isfinite(flat_pd['sar']) & np.isfinite(flat_pd['spa']) & np.isfinite(flat_pd['spa_top'])]


# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,8));
ax1 = axs[0]
ax2 = axs[1]


x1 = sns.regplot(ax=ax1, x=flat_pd['sar'], y=flat_pd['spa'], scatter_kws={'s':0.1, 'alpha':1.0}, line_kws={"color": "red"})
ax1.set(ylim=(0, 2))
ax1.set(xlim=(0, 2))

x2 = sns.regplot(ax=ax2, x=flat_pd['sar'], y=flat_pd['spa_top'], scatter_kws={'s':0.1, 'alpha':1.0}, line_kws={"color": "red"})
ax2.set(ylim=(0, 2))
ax2.set(xlim=(0, 2))

if saveplots == True:
    plt.savefig(f'sar_spa_qq_wetdry_norm_{today}.pdf')


#%% 

np.nanmean(sar_wliq[day_hi,:,:])

sar_flat = (sar_wliq[day_hi,:,:]/np.nanmean(sar_wliq[day_hi,:,:])).flatten()
#sar_flat[np.where(sar_flat <= 0)] = np.nan
spa_flat = (spa_wliq[day_hi,:,:]/np.nanmean(spa_wliq[day_hi,:,:])).flatten()
spa_top_flat = (spa_wliq_top[day_hi,:,:]/np.nanmean(spa_wliq_top[day_hi,:,:])).flatten()

flat_pd = pd.DataFrame()
flat_pd['sar'] = sar_flat
flat_pd['spa'] = spa_flat
flat_pd['spa_top'] = spa_top_flat
flat_pd = flat_pd.loc[np.isfinite(flat_pd['sar']) & np.isfinite(flat_pd['spa']) & np.isfinite(flat_pd['spa_top'])]

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,8));
ax1 = axs[0]
ax2 = axs[1]

x1 = sns.regplot(ax=ax1, x=flat_pd['sar'], y=flat_pd['spa'], scatter_kws={'s':0.1, 'alpha':1.0}, line_kws={"color": "red"})
ax1.set(ylim=(0, 2))
ax1.set(xlim=(0, 2))

x2 = sns.regplot(ax=ax2, x=flat_pd['sar'], y=flat_pd['spa_top'], scatter_kws={'s':0.1, 'alpha':1.0}, line_kws={"color": "red"})
ax2.set(ylim=(0, 2))
ax2.set(xlim=(0, 2))






