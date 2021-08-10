# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:58:31 2021

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
from spafhy_io import read_AsciiGrid

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

cmask = gis['cmask']

#%%

# sar soil moisture plots
import pandas as pd

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
kenttarova_loc = list([int(kenttarova_loc[0]), int(kenttarova_loc[1])])

# reading sar data
sar_path = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\processed\16m_nc_spafhy_pallas\SAR_PALLAS_2019_mask2_16m_direct_catchment_mean4.nc'
sar = Dataset(sar_path, 'r')

sar_wliq = sar['soilmoisture']*cmask/100
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']

dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 
dates_spa = pd.to_datetime(tvec[1:], format='%Y%m%d') 

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

fig.suptitle('SpaFHy v1')

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.8, 0.2, 0.02, 0.6])
#bar2 = fig.colorbar(im6, cax=cbar_ax)

#ax5.plot([1.1, 1.1], [1.1, -1.2], color='black', lw=1, transform=ax2.transAxes, clip_on=False)


if saveplots == True:
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_{today}.pdf')
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_{today}.png')
    
    
    
    
#%% 

# normalized by mean plots

# mean of each pixel
#spamean = np.nanmean(spa_wliq, axis=0)
#spatopmean = np.nanmean(spa_wliq_top, axis=0)
#sarmean = np.nanmean(sar_wliq, axis=0)

# mean of wet and dry days
spamean_wet = np.nanmean(spa_wliq[day_hi,:,:])
spatopmean_wet = np.nanmean(spa_wliq_top[day_hi,:,:])
sarmean_wet = np.nanmean(sar_wliq[day_hi,:,:])
spamean_dry = np.nanmean(spa_wliq[day_low,:,:])
spatopmean_dry = np.nanmean(spa_wliq_top[day_low,:,:])
sarmean_dry = np.nanmean(sar_wliq[day_low,:,:])

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

im1 = ax1.imshow(sar_wliq[day_hi,:,:]/sarmean_wet, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax1.title.set_text('SAR/SARwet_mean')
#ax1.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im2 = ax2.imshow(spa_wliq[day_hi, :,:]/spamean_wet, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax2.title.set_text('SPAFHYROOT/SPAFHYROOTwet_mean')
ax2.text(10, -15, f'Wet day : {hi_date}', fontsize=15)
#ax2.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im3 = ax3.imshow(sar_wliq[day_low, :,:]/sarmean_dry, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax3.title.set_text('SAR/SARdry_mean')

im4 = ax4.imshow(spa_wliq[day_low, :,:]/spamean_dry, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax4.title.set_text('SPAFHYROOT/SPAFHYROOTdry_mean')
ax4.text(10, -15, f'Dry day : {low_date}', fontsize=15)


im5 = ax5.imshow(spa_wliq_top[day_hi, :, :]/spatopmean_wet, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax5.title.set_text('SPAFHYTOP/SPAFHYTOPwet_mean')

im6 = ax6.imshow(spa_wliq_top[day_low, :,:]/spatopmean_dry, cmap='coolwarm_r', vmin=0.0, vmax=2.0, aspect='equal')
ax6.title.set_text('SPAFHYTOP/SPAFHYTOPdry_mean')

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

#ax1.text(10, -15, 'norm by mean of the day', fontsize=10)

fig.suptitle('SpaFHy v1')

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.8, 0.2, 0.02, 0.6])
#bar2 = fig.colorbar(im6, cax=cbar_ax)

#ax5.plot([1.1, 1.1], [1.1, -1.2], color='black', lw=1, transform=ax2.transAxes, clip_on=False)


if saveplots == True:
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_normmean_{today}.pdf')
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_normmean_{today}.png')

#%%

# normalized by median day plots

# mean of each pixel
#spamean = np.nanmean(spa_wliq, axis=0)
#spatopmean = np.nanmean(spa_wliq_top, axis=0)
#sarmean = np.nanmean(sar_wliq, axis=0)

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

im1 = ax1.imshow(sar_wliq[day_hi,:,:]/sarmedian, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax1.title.set_text('SAR/SAR_median')
#ax1.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im2 = ax2.imshow(spa_wliq[day_hi, :,:]/spamedian, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax2.title.set_text('SPAFHYROOT/SPAFHYROOT_median')
ax2.text(10, -15, f'Wet day : {hi_date}', fontsize=15)
#ax2.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im3 = ax3.imshow(sar_wliq[day_low, :,:]/sarmedian, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax3.title.set_text('SAR/SAR_median')

im4 = ax4.imshow(spa_wliq[day_low, :,:]/spamedian, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax4.title.set_text('SPAFHYROOT/SPAFHYROOT_median')
ax4.text(10, -15, f'Dry day : {low_date}', fontsize=15)


im5 = ax5.imshow(spa_wliq_top[day_hi, :, :]/spatopmedian, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax5.title.set_text('SPAFHYTOP/SPAFHYTOP_median')

im6 = ax6.imshow(spa_wliq_top[day_low, :,:]/spatopmedian, cmap='coolwarm_r', vmin=0.0, vmax=2.0, aspect='equal')
ax6.title.set_text('SPAFHYTOP/SPAFHYTOP_median')

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

#ax1.text(10, -15, 'norm by mean of the day', fontsize=10)
fig.suptitle('SpaFHy v1')

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.8, 0.2, 0.02, 0.6])
#bar2 = fig.colorbar(im6, cax=cbar_ax)

#ax5.plot([1.1, 1.1], [1.1, -1.2], color='black', lw=1, transform=ax2.transAxes, clip_on=False)


if saveplots == True:
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_normmedian_{today}.pdf')
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_normmedian_{today}.png')

#%%

# normalized by pixel mean

# mean of each pixel
spamean = np.nanmean(spa_wliq, axis=0)
spatopmean = np.nanmean(spa_wliq_top, axis=0)
sarmean = np.nanmean(sar_wliq, axis=0)

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
ax1.title.set_text('SAR/SARpixel_mean')
#ax1.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im2 = ax2.imshow(spa_wliq[day_hi, :,:]/spamean, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax2.title.set_text('SPAFHYROOT/SPAFHYROOTpixel_mean')
ax2.text(10, -15, f'Wet day : {hi_date}', fontsize=15)
#ax2.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im3 = ax3.imshow(sar_wliq[day_low, :,:]/sarmean, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax3.title.set_text('SAR/SARpixel_mean')

im4 = ax4.imshow(spa_wliq[day_low, :,:]/spamean, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax4.title.set_text('SPAFHYROOT/SPAFHYROOTpixel_mean')
ax4.text(10, -15, f'Dry day : {low_date}', fontsize=15)


im5 = ax5.imshow(spa_wliq_top[day_hi, :, :]/spatopmean, vmin=0.0, vmax=2.0, cmap='coolwarm_r', aspect='equal')
ax5.title.set_text('SPAFHYTOP/SPAFHYTOPpixel_mean')

im6 = ax6.imshow(spa_wliq_top[day_low, :,:]/spatopmean, cmap='coolwarm_r', vmin=0.0, vmax=2.0, aspect='equal')
ax6.title.set_text('SPAFHYTOP/SPAFHYTOPpixel_mean')

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

#ax1.text(10, -15, 'norm by mean of the day', fontsize=10)
fig.suptitle('SpaFHy v1')

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.8, 0.2, 0.02, 0.6])
#bar2 = fig.colorbar(im6, cax=cbar_ax)

#ax5.plot([1.1, 1.1], [1.1, -1.2], color='black', lw=1, transform=ax2.transAxes, clip_on=False)


if saveplots == True:
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_normpixel_{today}.pdf')
    plt.savefig(f'SAR_vs.SPAFHY_soilmoist_normpixel_{today}.png')
    
    
#%%

# point examples from mineral and openmire

# soilscouts at Kenttarova
folder = r'C:\SpaFHy_v1_Pallas\data\obs'
soil_file = 'soilscouts_s3_s5_s18.csv'
fp = os.path.join(folder, soil_file)
soilscout = pd.read_csv(fp, sep=';', date_parser=['time'])
soilscout['time'] = pd.to_datetime(soilscout['time'])

# ec observation data
ec_fp = r'C:\SpaFHy_v1_Pallas\data\obs\ec_soilmoist.csv'
ecmoist = pd.read_csv(ec_fp, sep=';', date_parser=['time'])
ecmoist['time'] = pd.to_datetime(ecmoist['time'])

soilm = soilscout.merge(ecmoist)

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
k_loc = list([int(kenttarova_loc[0]), int(kenttarova_loc[1])])
l_loc = [60, 60]
sar_wliq = sar['soilmoisture']*cmask/100
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']

dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 


#spa dates to match sar dates
date_in_spa = []
date_in_soilm = []
for i in range(len(dates_sar)):
    ix = np.where(dates_spa == dates_sar[i])[0][0]
    date_in_spa.append(ix)
    yx = np.where(soilm['time'] == dates_sar[i])[0][0]
    date_in_soilm.append(yx)
   
    
spa_wliq = spa_wliq[date_in_spa,:,:]
spa_wliq_top = spa_wliq_top[date_in_spa,:,:]
soilm = soilm.loc[date_in_soilm]
soilm = soilm.reset_index()

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,8));
ax1 = axs[0]
ax2 = axs[1]

#fig.suptitle('Volumetric water content', fontsize=15)

im1 = ax1.plot(sar_wliq[:,k_loc[0],k_loc[1]])
ax1.plot(spa_wliq[:,k_loc[0],k_loc[1]])
ax1.plot(spa_wliq_top[:,k_loc[0],k_loc[1]])
ax1.plot(soilm['s3'])
ax1.plot(soilm['s5'])
ax1.plot(soilm['s18'])
ax1.title.set_text('Mineral')
ax1.legend(['SAR', 'SpaFHy rootzone', 'SpaFHy top', 's3 = -0.05m', 's5 = -0.60', 's18 = -0.3'], ncol = 6)


im2 = ax2.plot(sar_wliq[:,l_loc[0],l_loc[1]])
ax2.plot(spa_wliq[:,l_loc[0],l_loc[1]])
ax2.plot(spa_wliq_top[:,l_loc[0],l_loc[1]])
ax2.title.set_text('Open mire')
ax2.legend(['SAR', 'SpaFHy rootzone', 'SpaFHy top'], ncol = 3)

#%%

# point examples from mineral and openmire without SAR

# soilscouts at Kenttarova
folder = r'C:\SpaFHy_v1_Pallas\data\obs'
soil_file = 'soilscouts_s3_s5_s18.csv'
fp = os.path.join(folder, soil_file)
soilscout = pd.read_csv(fp, sep=';', date_parser=['time'])
soilscout['time'] = pd.to_datetime(soilscout['time'])

# ec observation data
ec_fp = r'C:\SpaFHy_v1_Pallas\data\obs\ec_soilmoist.csv'
ecmoist = pd.read_csv(ec_fp, sep=';', date_parser=['time'])
ecmoist['time'] = pd.to_datetime(ecmoist['time'])

soilm = soilscout.merge(ecmoist)

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
k_loc = list([int(kenttarova_loc[0]), int(kenttarova_loc[1])])
l_loc = [60, 60]
sar_wliq = sar['soilmoisture']*cmask/100
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']


dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 

'''
#spa dates to match sar dates
date_in_spa = []
date_in_soilm = []
for i in range(len(dates_sar)):
    ix = np.where(dates_spa == dates_sar[i])[0][0]
    date_in_spa.append(ix)
    yx = np.where(soilm['time'] == dates_sar[i])[0][0]
    date_in_soilm.append(yx)
'''
spa_wliq_df = pd.DataFrame()
spa_wliq_df['spa_k'] = spa_wliq[:,k_loc[0],k_loc[1]]
spa_wliq_df['spa_l'] = spa_wliq[:,l_loc[0],l_loc[1]]
spa_wliq_df['spatop_k'] = spa_wliq_top[:,k_loc[0],k_loc[1]]
spa_wliq_df['spatop_l'] = spa_wliq_top[:,l_loc[0],l_loc[1]]
spa_wliq_df['time'] = dates_spa

soilm = soilm.merge(spa_wliq_df)
soilm.index = soilm['time']
soilm = soilm[['s3', 's5', 's18', 'SH-5A', 'SH-5B', 'SH-20A', 'SH-20B', 'spa_k', 'spa_l', 'spatop_k', 'spatop_l']]

soilm = soilm.loc[(soilm.index > '2018-04-01') & (soilm.index < '2019-12-01')]


# Plotting
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,8));
ax1 = axs[0]
ax2 = axs[1]

#fig.suptitle('Volumetric water content', fontsize=15)

im1 = ax1.plot(soilm['spa_k'])
ax1.plot(soilm['spatop_k'])
ax1.plot(soilm['s3'], alpha=0.4)
#ax1.plot(soilm['s5'], alpha=0.4)
ax1.plot(soilm['s18'], alpha=0.4)
#ax1.plot(soilm['SH-5A'], alpha=0.4)
ax1.plot(soilm['SH-5B'], alpha=0.4)
#ax1.plot(soilm['SH-20A'], alpha=0.4)
ax1.plot(soilm['SH-20B'], alpha=0.4)
ax1.title.set_text('Mineral')
ax1.legend(['spa root', 'spa top', 's3 = -0.05', 's18 = -0.3', 'SH-5A', 'SH-5B', 'SH-20A', 'SH-20B'], ncol = 8)
ax1.set_ylim(0,0.6)

im2 = ax2.plot(soilm['spa_l'])
ax2.plot(soilm['spatop_l'])
ax2.title.set_text('Mire')
ax2.legend(['spa root', 'spa top'], ncol = 2)

fig.suptitle('SpaFHy v1')


if saveplots == True:
    plt.savefig(f'pointplots_soilmoist_{today}.pdf')
    plt.savefig(f'pointplots_soilmoist_{today}.png')
    
    
#%%

# Q-Q plots of dry, wet and inbetween day

import numpy as np 
import pandas as pd


norm = False

sar_wliq = sar['soilmoisture']*cmask/100
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']


dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 


#spa dates to match sar dates
date_in_spa = []
for i in range(len(dates_sar)):
    ix = np.where(dates_spa == dates_sar[i])[0][0]
    date_in_spa.append(ix)
spa_wliq = spa_wliq[date_in_spa,:,:]
spa_wliq_top = spa_wliq_top[date_in_spa,:,:]

if norm == True:
    spa_wliq = spa_wliq/(np.nanmean(spa_wliq))
    spa_wliq_top = spa_wliq_top/(np.nanmean(spa_wliq_top))
    sar_wliq = sar_wliq/(np.nanmean(sar_wliq))

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

inb = int((day_hi+day_low)/2)
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

x1 = sns.regplot(ax=ax1, x=flat_pd['sar_dry'], y=flat_pd['spa_dry'], scatter_kws={'s':10, 'alpha':0.1}, line_kws={"color": "red"})
if norm == False:
    ax1.set(ylim=(0, 1))
    ax1.set(xlim=(0, 1))
else:
    ax1.set(ylim=(0, 2.5))
    ax1.set(xlim=(0, 2.5))    
#ax1.set_title('Dry day')

x2 = sns.regplot(ax=ax2, x=flat_pd['sar_wet'], y=flat_pd['spa_wet'], scatter_kws={'s':10, 'alpha':0.1}, line_kws={"color": "red"})
if norm == False:
    ax2.set(ylim=(0, 1))
    ax2.set(xlim=(0, 1))
else:
    ax2.set(ylim=(0, 2.5))
    ax2.set(xlim=(0, 2.5)) 
#ax2.set_title('Wet day')

x3 = sns.regplot(ax=ax3, x=flat_pd['sar_dry'], y=flat_pd['spa_top_dry'], scatter_kws={'s':10, 'alpha':0.1}, line_kws={"color": "red"})
if norm == False:
    ax3.set(ylim=(0, 1))
    ax3.set(xlim=(0, 1))
else:
    ax3.set(ylim=(0, 2.5))
    ax3.set(xlim=(0, 2.5))     
#ax3.set_title('Dry day')

x4 = sns.regplot(ax=ax4, x=flat_pd['sar_wet'], y=flat_pd['spa_top_wet'], scatter_kws={'s':10, 'alpha':0.1}, line_kws={"color": "red"})
if norm == False:
    ax4.set(ylim=(0, 1))
    ax4.set(xlim=(0, 1))
else:
    ax4.set(ylim=(0, 2.5))
    ax4.set(xlim=(0, 2.5))     
#ax4.set_title('Wet day')

x5 = sns.regplot(ax=ax5, x=flat_pd['sar_inb'], y=flat_pd['spa_inb'], scatter_kws={'s':10, 'alpha':0.1}, line_kws={"color": "red"})
if norm == False:
    ax5.set(ylim=(0, 1))
    ax5.set(xlim=(0, 1))
else:
    ax5.set(ylim=(0, 2.5))
    ax5.set(xlim=(0, 2.5))     
#ax4.set_title('Wet day')

x6 = sns.regplot(ax=ax6, x=flat_pd['sar_inb'], y=flat_pd['spa_top_inb'], scatter_kws={'s':10, 'alpha':0.1}, line_kws={"color": "red"})
if norm == False:
    ax6.set(ylim=(0, 1))
    ax6.set(xlim=(0, 1))
else:
    ax6.set(ylim=(0, 2.5))
    ax6.set(xlim=(0, 2.5))     
#ax4.set_title('Wet day')

if norm == True:
    fig.suptitle('SpaFHy v1, norm by total means of each')
else:
    fig.suptitle('SpaFHy v1')

if norm == False:
    if saveplots == True:
        plt.savefig(f'sar_spa_qq_wetdry_{today}.pdf')
        plt.savefig(f'sar_spa_qq_wetdry_{today}.png')
else: 
    if saveplots == True:
        plt.savefig(f'sar_spa_qq_wetdry_norm_{today}.pdf')
        plt.savefig(f'sar_spa_qq_wetdry_norm_{today}.png')
        
#%%


# QQ plots of the whole season

norm = False

sar_wliq = sar['soilmoisture']*cmask/100
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']


dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 

#spa dates to match sar dates
date_in_spa = []
for i in range(len(dates_sar)):
    ix = np.where(dates_spa == dates_sar[i])[0][0]
    date_in_spa.append(ix)
spa_wliq = spa_wliq[date_in_spa,:,:]
spa_wliq_top = spa_wliq_top[date_in_spa,:,:]

sar_flat = sar_wliq[:,:,:].flatten()
spa_flat = spa_wliq[:,:,:].flatten()
spa_top_flat = spa_wliq_top[:,:,:].flatten()

flat_pd = pd.DataFrame()
flat_pd['sar'] = sar_flat
flat_pd['spa'] = spa_flat
flat_pd['spa_top'] = spa_top_flat
flat_pd = flat_pd.loc[np.isfinite(flat_pd['sar']) & np.isfinite(flat_pd['spa']) & np.isfinite(flat_pd['spa_top'])]


# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4));
ax1 = axs[0]
ax2 = axs[1]

if norm == False:
    x1 = sns.regplot(ax=ax1, x=flat_pd['sar'], y=flat_pd['spa'], scatter_kws={'s':10, 'alpha':0.005}, line_kws={"color": "red"})
    ax1.set(ylim=(0, 1))
    ax1.set(xlim=(0, 1))
    #ax1.set_title('Dry day')

    x2 = sns.regplot(ax=ax2, x=flat_pd['sar'], y=flat_pd['spa_top'], scatter_kws={'s':10, 'alpha':0.005}, line_kws={"color": "red"})
    ax2.set(ylim=(0, 1))
    ax2.set(xlim=(0, 1))
    #ax2.set_title('Wet day')
    
    fig.suptitle('SpaFHy v1')
    
    if saveplots == True:
        plt.savefig(f'sar_spa_qq_{today}.pdf')
        plt.savefig(f'sar_spa_qq_{today}.png')
    
else:
    x1 = sns.regplot(ax=ax1, x=flat_pd['sar']/(flat_pd['sar'].mean()), y=flat_pd['spa']/(flat_pd['spa'].mean()), scatter_kws={'s':10, 'alpha':0.005}, line_kws={"color": "red"})
    ax1.set(ylim=(0, 2.5))
    ax1.set(xlim=(0, 2.5))
    #ax1.set_title('Dry day')

    x2 = sns.regplot(ax=ax2, x=flat_pd['sar']/(flat_pd['sar'].mean()), y=flat_pd['spa_top']/(flat_pd['spa_top'].mean()), scatter_kws={'s':10, 'alpha':0.005}, line_kws={"color": "red"})
    ax2.set(ylim=(0, 2.5))
    ax2.set(xlim=(0, 2.5))
    #ax2.set_title('Wet day')  
    
    fig.suptitle('SpaFHy v1D, norm by total means of each')

    
    if saveplots == True:
        plt.savefig(f'sar_spa_qq_norm_{today}.pdf')
        plt.savefig(f'sar_spa_qq_norm_{today}.png')
        
#%%

# QQ plots of spa and sar

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4));
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

x1 = sns.scatterplot(ax=ax1, x=flat_pd['sar'], y=flat_pd['sar'], alpha=0.003)
ax1.set(ylim=(0, 1))
ax1.set(xlim=(0, 1))
#ax1.set_title('Dry day')

x2 = sns.scatterplot(ax=ax2, x=flat_pd['spa'], y=flat_pd['spa'], alpha=0.003)
ax2.set(ylim=(0, 1))
ax2.set(xlim=(0, 1))
#ax2.set_title('Wet day')

x3 = sns.scatterplot(ax=ax3, x=flat_pd['spa_top'], y=flat_pd['spa_top'], alpha=0.003)
ax3.set(ylim=(0, 1))
ax3.set(xlim=(0, 1))
#ax2.set_title('Wet day')
    
fig.suptitle('SpaFHy v1')
    
if saveplots == True:
        #plt.savefig(f'qq_spa_spatop_sar_{today}.pdf')
        plt.savefig(f'qq_spa_spatop_sar_{today}.png')

#%%
