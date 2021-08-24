# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:21:26 2017

@author: slauniai

MAKES FIGURES 6-8 IN THE MANUSCRIPT
CHANGE PATHS TO READ .nc FILE AND .pk -FILE.

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
import pandas as pd
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

#%%

# gis data plots

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(12,4));
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
ax4 = axs[3]
ax5 = axs[4]
ax6 = axs[5]

#fig.suptitle('Volumetric water content', fontsize=15)

im1 = ax1.imshow(gis['dem'], cmap='coolwarm_r', aspect='equal')
ax1.title.set_text('DEM')
#ax1.text(175, 0, f'Wet day : {hi_day}', fontsize=12)
im2 = ax2.imshow(gis['soilclass'])
ax2.title.set_text('soil')

im3 = ax3.imshow(gis['LAI_conif'], cmap='terrain')
ax3.title.set_text('LAI conif')

im4 = ax4.imshow(gis['LAI_decid'], cmap='terrain')
ax4.title.set_text('LAI decid')

im5 = ax5.imshow(gis['hc'], cmap='gray')
ax5.title.set_text('canopy height')

im6 = ax6.imshow(gis['cf'], cmap='summer')
ax6.title.set_text('canopy fraction')

ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")
ax5.axis("off")
ax6.axis("off")


#%% FIG. 9 long-term ET/P and components

# NOTE [ix:] ensures we neglect spinup period!

P = np.sum(FORC['Prec'][ix:])*spa.dt  # total precip

# ET components, from spinup end to end of simulation

TR = np.array(cres['Transpi'][ix:, :, :])  # transpi
EF = np.array(cres['Efloor'][ix:, :, :])  # floor evap
IE = np.array(cres['Evap'][ix:, :, :])# interception evap
ET = TR + EF + IE

#--------

fig = plt.figure()
fig.set_size_inches(10.0, 18.0)

# maps to lhs of the figure

plt.subplot(521)
sns.heatmap(LAIc + LAId + LAIs + LAIg, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); 
plt.title('LAI (m$^2$m$^{-2}$)')

plt.subplot(523)
sns.heatmap(np.sum(ET, axis=0)/P, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); 
plt.title('$\overline{ET}$/$\overline{P}$ (-)')

plt.subplot(525)
sns.heatmap(np.sum(IE, axis=0)/P, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); 
plt.title('$\overline{E}$/$\overline{P}$ (-)')

plt.subplot(527)
sns.heatmap(np.sum(TR, axis=0)/P, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); 
plt.title('$\overline{T_r}$/$\overline{P}$ (-)')

plt.subplot(529)
sns.heatmap(np.sum(EF, axis=0)/P, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); 
plt.title('$\overline{E_f}$/$\overline{P}$ (-)')

# add LAI -relations to rhs of figure

x = LAId + LAIc + LAIs + LAIg

plt.subplot(5,2,4)
y = np.sum(ET, axis=0) / P
rr = soil.copy()
y = np.ravel(y)
rr[rr == 4] = 3 # peat
cm = plt.get_cmap('coolwarm_r', 3)

norm = mplcolors.BoundaryNorm(np.arange(1, 5) - 0.5, 4)
plt.scatter(x, y, c=rr, cmap=cm, norm=norm, s=10, alpha=0.5)
cb1= plt.colorbar(ticks=np.arange(0, 4))
cb1.ax.set_yticklabels(['coarse','med','peat']) 

ax = plt.gca()
ax.grid(linestyle='--', alpha=0.5)
plt.xlim([0, 5])
plt.ylim([0.2, 0.7])
#plt.ylabel('$\overline{ET}$/$\overline{P}$ (-)', labelpad=-1);

# interception
plt.subplot(5,2,6)
rr = LAId / x
y = np.sum(IE, axis=0) / P
sc = plt.scatter(x, y, c=rr, vmin=0, vmax=0.6, s=10, alpha=0.5, cmap='coolwarm')
cb = plt.colorbar(sc, ticks = [0.0, 0.20, 0.40, 0.60])
cb.set_label('LAI$_d$ / LAI', rotation=90, fontsize=9)
plt.xlim([0, 5])
plt.ylim([0, 0.3])
#plt.ylabel('$\overline{E}$/$\overline{P}$ (-)', labelpad=-1);
ax = plt.gca()
ax.grid(linestyle='--', alpha=0.5)


# transpiration / P
plt.subplot(528)
y = np.sum(TR, axis=0) / P
rr = LAId / x # deciduous fraction
sc = plt.scatter(x, y, c=rr, vmin=0, vmax=0.6, s=10, alpha=0.5, cmap='coolwarm')
cb = plt.colorbar(sc, ticks = [0.0, 0.20, 0.40, 0.60])
cb.set_label('LAI$_d$ / LAI', rotation=90, fontsize=9)
plt.xlim([0, 9])
ax = plt.gca()
ax.grid(linestyle='--', alpha=0.5)
plt.xlim([0, 5])
plt.ylim([0, 0.4])
#plt.ylabel('$\overline{T_r}$/$\overline{P}$ (-)', labelpad=-2);

# floor evaporation / P
plt.subplot(5,2,10)
rr = twi
y = np.sum(EF, axis=0) / P

sc = plt.scatter(x, y, c=rr, s=10, alpha=0.5, cmap='coolwarm_r')
cb = plt.colorbar(sc)
cb.set_label('TWI', rotation=90, fontsize=9)
plt.xlim([0, 5])
plt.ylim([0, 0.3])
#plt.ylabel('$\overline{E_f}$/$\overline{P}$ (-)', labelpad=-2)
plt.xlabel('LAI (m$^2$m$^{-2}$)')
ax = plt.gca()
ax.grid(linestyle='--', alpha=0.5)
plt.xlim([0, 5])
plt.ylim([0, 0.3])

if saveplots == True:
    plt.savefig(f'ET_variability_{today}.pdf')

#%% plot ET partitioning as function of LAI

t_et = np.ravel(np.sum(ET, axis=0))
t_tr = np.ravel(np.sum(TR, axis=0))
t_ef = np.ravel(np.sum(EF, axis=0))
t_e = np.ravel(np.sum(IE, axis=0))

fig = plt.figure()
fig.set_size_inches(7, 6)

x = np.ravel(LAId + LAIc + LAIg + LAIs)
cm = plt.get_cmap('Blues')

# number of years
nyrs = len(np.unique(Qmeas[ix:].index.year))

plt.subplot(211) # mean annual ET and its components
plt.plot(x, t_et/nyrs, 'o', color='k', markersize=2, alpha=0.5, label='ET')
plt.plot(x, t_e/nyrs, 'o', color=cm(0.4), markersize=2, alpha=0.4, label='E')
plt.plot(x, t_tr/nyrs, 'o', color=cm(0.7), markersize=2, alpha=0.7, label='T$_r$')
plt.plot(x, t_ef/nyrs, 'o', color=cm(1.0), markersize=2, alpha=0.8, label='E$_f$')
plt.ylabel('[mm / year]')
plt.legend(fontsize=10, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
plt.xlim([0.5, 5.5])


plt.subplot(212)
plt.plot(x, t_e/t_et, 'o', color=cm(0.4), markersize=4, alpha=0.4, label='E')
plt.plot(x, t_tr/t_et, 'o', color=cm(0.7), markersize=4, alpha=0.7, label='T$_r$')
plt.plot(x, t_ef/t_et, 'o', color=cm(1.0), markersize=4, alpha=0.8, label='E$_f$')
plt.legend(fontsize=10, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
plt.ylabel('contribution to ET (-)', fontsize=10)
plt.xlabel('LAI (m$^2$m$^{-2}$)', fontsize=10, labelpad=-3)
ax = plt.gca()
ax.grid(linestyle='--', alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlim([0.5, 5.5])
plt.ylim([0, 1.0])


if saveplots == True:
    plt.savefig(f'ET_partitioning_{today}.pdf')

#%% FIG. 10: maximum snow water equivalent SWE

# snow water equivalent, seek maximum timing
SWE = np.array(cres['SWE'][ix:, :, :]) # SWE
a = np.nansum(SWE, axis=1)
a = np.nansum(a, axis=1)
swe_max_ix = int(np.where(a == np.nanmax(a))[0][0])
del a

#----
fig = plt.figure()
fig.set_size_inches(8.0, 2.5)

plt.subplot(121)
y = SWE[swe_max_ix,:,:]
sns.heatmap(y, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False)
plt.title('max SWE (mm)', fontsize=10)

laiw = LAId * 0.1 + LAIc  # wintertime lai
f = y > 0
yy = y[f]
laiw = laiw[f]
plt.subplot(122)
cm = plt.get_cmap('coolwarm')
plt.plot(laiw, yy/max(yy), 'o', color=cm(0.1), markersize=5, alpha=0.2)
plt.title('relative SWE (-)', fontsize=10)
plt.xlabel('winter LAI  (m$^2$m$^{-2}$)', fontsize=10, labelpad=-30)
plt.xlim([0, 5])
plt.ylim([0.5, 1.02])

ax = plt.gca()
ax.yaxis.set_label_position("right")
#ax.yaxis.tick_right()
ax.set_xticks([0,1,2,3,4,5])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_alpha(0.5)
ax.grid(linestyle='--', alpha=0.5)

if saveplots == True:
    plt.savefig(f'swe_variability_{today}.pdf')


#%% Fig 6: streamflow
import pandas as pd


SWE_m = pd.read_csv('C:\SpaFHy_v1_Pallas\data\obs\SWE_survey_2018-02-22_2020-06-04.txt', 
                    skiprows=5, sep=';', parse_dates = ['date'], encoding='iso-8859-1')

start = '2014-01-01'
end = '2019-12-31'

f0 = int(np.where(tvec == start)[0])
f1 = int(np.where(tvec == end)[0])

Qt = pd.DataFrame()
Qt['Q'] = 1e3*np.array(tres['Qt'])[f0:f1] # modeled streamflow
Qt.index = pd.to_datetime(tvec[f0:f1])

Qm = Qmeas[(Qmeas.index >= start) & (Qmeas.index < end) ]


metrics = False

if metrics == True:
    mse = round((((Qt[~np.isnan(Qm)] - Qm[~np.isnan(Qm)])**2) / len(Qm[~np.isnan(Qm)])).sum(), 2)
    me = round((Qm[~np.isnan(Qm)] - Qt[~np.isnan(Qm)]).sum() / len(Qm[~np.isnan(Qm)]), 2)
    nse = round(1 - ((Qm[~np.isnan(Qm)] - Qt[~np.isnan(Qm)])**2).sum() / (((Qm[~np.isnan(Qm)] - Qm[~np.isnan(Qm)].mean())**2).sum()), 2)


f0 = int(np.where(tvec == start)[0])
f1 = int(np.where(tvec == end)[0])

SWE_m = SWE_m.loc[(SWE_m['date'] >= start) & (SWE_m['date'] <= end)]

tvec0 = tvec[f0:f1]
Prec = FORC['Prec'].iloc[f0:f1] # precipitation
Wliq = np.array(bres['Wliq'])[f0:f1,:,:] # root zone moisture m3m-3

f, g = np.where(spa.GisData['cmask'] == 1)
swe = np.array(cres['SWE'])
swe = swe[f0:f1, f,g]
a = np.nansum(swe, axis=1)

swe = a / len(f)
max_swe = max(swe)

# rootzone water storage; seek catchment maximum and minimum timing
a = np.nansum(Wliq, axis=1)
a = np.nansum(a, axis=1)

ix_slow = int(np.where(a == np.nanmin(a))[0])
ix_shi = int(np.where(a == np.nanmax(a))[0])

fig1, ax = plt.subplots()
fig1.set_size_inches(6.5, 4.5)

#ax.plot(tvec0, Qm, 'k.', tvec0, Qt, 'r-', linewidth=1.0, markersize=4)
ax.plot(Qm, 'k.', linewidth=1.0, markersize=4)
ax.plot(Qt, 'r-', linewidth=1.0, markersize=4)
#
ax.plot(tvec0[ix_slow], -0.5, marker='o', mec='k', mfc='g', alpha=0.7, ms=8.0)
ax.plot(tvec0[ix_shi], -0.5, marker='o', mec='k', mfc='b', alpha=0.7, ms=8.0)
ax.set_ylim(-1, 15)
ax.legend(['measured', 'modeled'], bbox_to_anchor =(0.76, 1.15), ncol = 2) 
if metrics == True:
    ax.text(pd.to_datetime('2013-12-20'),13.5, f'NSE = {nse} ')
    ax.text(pd.to_datetime('2013-12-20'),12.5, f'ME = {me} ')
    ax.text(pd.to_datetime('2018-12-20'),13.5, f'MSE = {mse} ')

#ax.set_xlim(['2012-01-01', '2014-01-01'])
ax.set_ylabel(r'$\langle Q_f \rangle$ (mm d$^{-1}$)')
ax2=ax.twinx()   
ax2.bar(tvec0, 3600*24.0*Prec, color='k', width=1)
ax2.plot(tvec0, swe / 10.0, 'b-', linewidth=1.5)
ax2.plot(SWE_m['date'], SWE_m['SWE']/10, 'g.', linewidth=1.0, markersize=4)
# ax2.plot(tvec0, 20*swe, 'r--', linewidth=1)
ax2.set_ylabel(r'P (mm d$^{-1}$) & 0.1 x $\langle SWE \rangle$ (mm)'); plt.ylim([0, 100])
ax2.invert_yaxis()
#plt.xticks(rotation=30)        
for tick in ax.get_xticklabels():
        tick.set_rotation(30)

if saveplots == True:
    plt.savefig(f'discharge_timeseries_{today}.pdf')

#%% Fig 8: snapshots of soil moisture 

# select hydrologically contrasting years 2012 and 2013 for example
f0 = int(np.where(tvec == '2014-01-01')[0])
f1 = int(np.where(tvec == '2019-10-01')[0])

tvec0 = tvec[f0:f1]
Qt = 1e3*np.array(tres['Qt'])[f0:f1] # modeled streamflow mm/d
Wliq = np.array(bres['Wliq'])[f0:f1,:,:] # root zone moisture m3m-3
S = np.array(tres['S'])[f0:f1]  # saturation deficit 
del f0, f1 


# local saturation deficits
s_hi = 1e3*spa.top.local_s(S[ix_shi])
s_hi[s_hi<0] = 0.0
s_low = 1e3*spa.top.local_s(S[ix_slow])
s_low[s_low<0] = 0.0

# convert 1-D array back to 2-D grid
s_hi = spa._to_grid(s_hi)
s_low = spa._to_grid(s_low)

fig = plt.figure()
fig.set_size_inches(6.0, 4.0)

plt.subplot(221)
sns.heatmap(Wliq[ix_slow,:,:], cmap='RdBu',cbar=True, vmin=0.1, vmax=0.9, 
            xticklabels=False, yticklabels=False);
tt = tvec0[ix_slow]
tt = tt.strftime('%Y-%m-%d')
plt.title('dry: ' + tt, fontsize=10)
plt.xlabel('$\\theta$ (m$^3$m$^{-3}$)', fontsize=10)

plt.subplot(222)
sns.heatmap(Wliq[ix_shi,:,:], cmap='RdBu',cbar=True, vmin=0.1, vmax=0.9, xticklabels=False, yticklabels=False);
tt = tvec0[ix_shi]
tt = tt.strftime('%Y-%m-%d')
plt.title('wet: ' + tt, fontsize=10)
plt.xlabel('$\\theta$ (m$^3$m$^{-3}$)', fontsize=10)

plt.subplot(223)
sns.heatmap(s_low, cmap='RdBu_r',cbar=True, vmin=0.0, vmax=180, xticklabels=False, yticklabels=False);
plt.xlabel('S (mm)', fontsize=10)

plt.subplot(224)
sns.heatmap(s_hi, cmap='RdBu_r',cbar=True, vmin=0.0, vmax=180, xticklabels=False, yticklabels=False);
plt.xlabel('S (mm)', fontsize=10)

if saveplots == True:
    plt.savefig(f'soil_moisture_{today}.pdf')

# soil moisture as function of LAI in dry period
plt.figure()
x = LAIc + LAId
y = Wliq[ix_slow, :, :]
plt.plot(x, y, 'o');
plt.xlabel('LAI [m2 m-2]')
plt.ylabel('soil moisture [m3 m-3]')


#%% snapshots of soil moisture from a specific day

x0 = int(np.where(tvec == '2019-08-05')[0])
Wliq = np.array(bres['Wliq'])[x0,:,:] # root zone moisture m3m-3
S = np.array(tres['S'])[x0]  # saturation deficit 

fig = plt.figure()
fig.set_size_inches(9.0, 6.0)

plt.subplot(221)
plt.imshow(Wliq[:,:], cmap='coolwarm', vmin=0.0, vmax=1.0, aspect='equal')
#sns.heatmap(Wliq[:,:], cmap='twilight',cbar=True, vmin=0.1, vmax=0.9, 
#            xticklabels=False, yticklabels=False);
tt = tvec[x0]
tt = tt.strftime('%Y-%m-%d')
plt.title(tt, fontsize=10)
plt.xlabel('$\\theta$ (m$^3$m$^{-3}$)', fontsize=10)
 
#plt.savefig(f'discharge_timeseries_{today}.pdf')


#%% Extract data for analysis of soil moisture variability
f0 = int(np.where(tvec == '2014-01-01')[0])
f1 = int(np.where(tvec == '2018-12-31')[0])

tvec0 = tvec[f0:f1]
mo = np.array(tvec0.month)
yr = np.array(tvec0.year)
yr[yr == 2012] = 0
yr[yr == 2013] = 1
doy0 = np.array(tvec0.dayofyear)

f, g = np.where(spa.GisData['cmask'] == 1)

Wliq = np.array(bres['Wliq'])[f0:f1,f,g]
twi0 = np.array(spa.GisData['twi'][f,g])
soil0 = np.array(spa.GisData['soilclass'][f,g])
LAI = LAIc + LAId
lai0 = LAI[f,g]


#%%

# Kenttärova line plots spafhy vs. sensors

import pandas as pd
from spafhy_io import read_AsciiGrid

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
kenttarova_loc = list([int(kenttarova_loc[0]), int(kenttarova_loc[1])])

#sve_gtk_pintamaa, _, _, _, _ = read_AsciiGrid(r'C:\SpaFHy_v1_Pallas\data\C16\soilclass.dat')
# kenttarova soilclass = 2
soil2 = np.where((gis['soilclass'] == 2) & (gis['cmask'] == 1))

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

#skip warm up
tm = range(365,(len(dat['time'][:])))

ecmoist = ecmoist.loc[(ecmoist['time'] >= tvec[tm][0]) & (ecmoist['time'] <= tvec[tm][-1])]
soilscout = soilscout.loc[(soilscout['time'] >= tvec[tm][0]) & (soilscout['time'] <= tvec[tm][-1])]

# SpaFHy rootzone and top soil moisture
Wliq_root = np.array(bres['Wliq'][tm])
Wliq_top = np.array(bres['Wliq_top'][tm])

# Both at kenttärova and mean in soilclass 2
Wliq_root_kr = Wliq_root[:, kenttarova_loc[0], kenttarova_loc[1]]
Wliq_top_kr = Wliq_top[:, kenttarova_loc[0], kenttarova_loc[1]]

# Wliq means within the soilclass 2
Wliq_root_mean = Wliq_root[:,soil2[0],soil2[1]]
Wliq_root_mean = np.nanmean(Wliq_root_mean, axis=(1))
Wliq_top_mean = Wliq_top[:,soil2[0],soil2[1]]
Wliq_top_mean = np.nanmean(Wliq_top_mean, axis=(1))

# new dataframe for spafhy data
Wliq_final = pd.DataFrame()
Wliq_final['time'] = tvec[tm]
Wliq_final['root_kr'] = Wliq_root_kr
Wliq_final['root_mean'] = Wliq_root_mean
Wliq_final['top_kr'] = Wliq_top_kr
Wliq_final['top_mean'] = Wliq_top_mean
    
# indexes as dates
soilscout.index = soilscout['time']
ecmoist.index = ecmoist['time']
Wliq_final.index = Wliq_final['time']

Prec = FORC['Prec'][tm[0]:len(tm)+tm[0]]

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,4));
ax1 = axs[0]
ax2 = axs[1]


l1 = ax1.plot(ecmoist['SH-5A'], marker='', color='black', linewidth=1, alpha=0.4)
l2 = ax1.plot(ecmoist['SH-5B'], marker='', color='black', linewidth=1, alpha=0.6)
l3 = ax1.plot(soilscout['s3'], marker='', color='black', linewidth=1, alpha=0.2) #-0.05
#ax.fill_between(l1, l2, color='grey', alpha='0.5')
l3 = ax1.plot(Wliq_final['top_kr'], marker='', color='red', linewidth=1, alpha=0.7)
ax1.set_ylim(0,0.70)
ax1.set_ylabel('Volumetric water content')
ax1.legend(['5A = -0.05', '5B = -0.05', 's3 = -0.05', 'Wliq_top'], bbox_to_anchor =(0.95, 1.15), ncol = 4)

l4 = ax2.plot(ecmoist['SH-20A'], marker='', color='black', linewidth=1, alpha=0.4)
l5 = ax2.plot(ecmoist['SH-20B'], marker='', color='black', linewidth=1, alpha=0.6)
l6 = ax2.plot(soilscout['s18'], marker='', color='black', linewidth=1, alpha=0.2)
#ax.plot(soilscout['s18'], marker='', color='blue', linewidth=1)
l7 = ax2.plot(Wliq_final['root_kr'], marker='', color='red', linewidth=1, alpha=0.7)
ax2.set_ylim(0,0.70)
ax2.set_ylabel('Volumetric water content')
ax2.legend(['20A = -0.20', '20B = -0.20', 's18 = -0.20', 'Wliq_root'], bbox_to_anchor =(0.95, 1.15), ncol = 4)

for tick in ax1.get_xticklabels():
        tick.set_rotation(30)
for tick in ax2.get_xticklabels():
        tick.set_rotation(30)
        
ax1=ax1.twinx()   
ax1.bar(Prec.index, 3600*24.0*Prec, color='blue', width=1)
ax1.set_ylabel(r'P (mm d$^{-1}$)')
ax1.set_ylim([0, 70])
ax1.invert_yaxis()

ax2=ax2.twinx()   
ax2.bar(Prec.index, 3600*24.0*Prec, color='blue', width=1)
ax2.set_ylabel(r'P (mm d$^{-1}$)')
ax2.set_ylim([0, 70])
ax2.invert_yaxis()

if saveplots == True:
    plt.savefig(f'soilmoist_kenttarova_{today}.pdf')
    plt.savefig(f'soilmoist_kenttarova_{today}.png')
    
    
#%%
'''
# For specific time period
start = '2018-05-01'
end = '2019-09-30'

Wliq_final_19 = Wliq_final.loc[(Wliq_final['time'] >= start) & (Wliq_final['time'] <= end)]
ecmoist_19 = ecmoist.loc[(ecmoist['time'] >= start) & (ecmoist['time'] <= end)]
soilscout_19 = soilscout.loc[(soilscout['time'] >= start) & (soilscout['time'] <= end)]
Prec_19 = Prec[start:end]


# What drains the kenttärova soil?
tvec_ind = np.where(tvec == start)[0][0]
tvec_ind2 = np.where(tvec == end)[0][0]+1
tra = np.array(cres['Transpi'][tvec_ind:tvec_ind2,kenttarova_loc[0], kenttarova_loc[1]])
dra = np.array(bres['Drain'][tvec_ind:tvec_ind2,kenttarova_loc[0], kenttarova_loc[1]])
efl = np.array(cres['Efloor'][tvec_ind:tvec_ind2,kenttarova_loc[0], kenttarova_loc[1]])
rg = FORC[(np.where(FORC.index == start)[0][0]):(np.where(FORC.index == end)[0][0]+1)]['Rg']

fluxes = pd.DataFrame()
fluxes['time'] = tvec[tvec_ind:tvec_ind2]
fluxes['Tr'] = tra
fluxes['Dr'] = dra * 1000
fluxes['Ef'] = efl * 5
fluxes['Rg'] = rg / 10
fluxes.index = fluxes['time']

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,4));
ax1 = axs[0]
ax2 = axs[1]


l1 = ax1.plot(ecmoist_19['SH-5A'], marker='', color='grey', linewidth=1)
l2 = ax1.plot(ecmoist_19['SH-5B'], marker='', color='grey', linewidth=1)
l3 = ax1.plot(soilscout_19['s3'], marker='', color='grey', linewidth=1) #-0.05
#ax.fill_between(l1, l2, color='grey', alpha='0.5')
l3 = ax1.plot(Wliq_final_19['top_kr'], marker='', color='red', linewidth=1)
ax1.set_ylim(0,0.60)
ax1.set_ylabel('Volumetric water content')
ax1.legend(['5A = -0.05', '5B = -0.05', 's3 = -0.05', 'Wliq_top'], bbox_to_anchor =(0.95, 1.15), ncol = 4)

l4 = ax2.plot(ecmoist_19['SH-20A'], marker='', color='grey', linewidth=1)
l5 = ax2.plot(ecmoist_19['SH-20B'], marker='', color='grey', linewidth=1)
l6 = ax2.plot(soilscout_19['s18'], marker='', color='grey', linewidth=1)
#ax.plot(soilscout['s18'], marker='', color='blue', linewidth=1)
l7 = ax2.plot(Wliq_final_19['root_kr'], marker='', color='red', linewidth=1)
ax2.set_ylim(0,0.60)
ax2.set_ylabel('Volumetric water content')
ax2.legend(['20A = -0.20', '20B = -0.20', 's18 = -0.20', 'Wliq_root'], bbox_to_anchor =(0.95, 1.15), ncol = 4)

for tick in ax1.get_xticklabels():
        tick.set_rotation(30)
for tick in ax2.get_xticklabels():
        tick.set_rotation(30)
        
ax1=ax1.twinx()   
ax1.bar(Prec_19.index, 3600*24.0*Prec_19, color='blue', width=1)
ax1.plot(fluxes['Ef'], color='green', linewidth=1, alpha=0.4)
ax1.set_ylabel(r'P (mm d$^{-1}$) & 5 * Ef (mm)')
ax1.set_ylim([0, 50])
ax1.invert_yaxis()

ax2=ax2.twinx()   
ax2.bar(Prec_19.index, 3600*24.0*Prec_19, color='blue', width=1)
ax2.bar(fluxes.index, fluxes['Tr']*5, color='green', width=1, alpha=0.4)
ax2.set_ylabel(r'P (mm d$^{-1}$) & 5 * T (mm)')
ax2.set_ylim([0, 50])
ax2.invert_yaxis()

if saveplots == True:
    plt.savefig(f'soilmoist_kenttarova_specifictime_{today}.pdf')

#%% 

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,4));

axs.plot(fluxes['Rg'], color='orange', linewidth=1)
axs.plot(fluxes['Tr'], color='green', linewidth=1)
axs.plot(Wliq_final.loc[start:end, 'root_kr'] * 100, color='blue', linewidth=1)
axs.plot(fluxes['Dr'], color='grey')
axs.legend(['Rg/10 (W/m2)', 'Tr (mm)', 'Wliq (%)', 'Dr (mm)'])
axs.set_title('FC = 0.33, WP = 0.13')

if saveplots == True:
    plt.savefig(f'ET_RG_WLIQ{today}.pdf')
'''
#%%

# Water balance check

import pandas as pd

# Year: P - ET - Q = dS (dS = 0)

WB = pd.DataFrame()
Prec = FORC['Prec']

# from mm/s to mm/d
yPrec = Prec.resample('Y').sum() * 3600 * 24 # sum of mm/d
#yPrec = yPrec[1:6:,]

yQ = Qmeas.resample('Y').sum() # sum of mm/d
#yQ = yQ[1:6:,]

WB['P'] = yPrec[1:7]
WB['Q'] = yQ
WB['ET'] = None
WB.index = yQ.index.year[1:]

years = WB.index


for i in range(len(years)):
    year = years[i]
    ixstart = np.where(FORC.index.year == years[i])[0][0]
    ixend = np.where(FORC.index.year == years[i])[0][0]+365
    ET = np.array(cres['ET'][ixstart:ixend, :, :])  # transpi
    t_et = np.ravel(np.sum(ET, axis=0))
    WB.loc[years[i], 'ET'] = np.nanmean(t_et)
    
WB['P-Q-ET'] = WB['P'] - WB['Q'] - WB['ET']


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
sar_path = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\processed\16m_nc_spafhy_pallas\SAR_PALLAS_2019_mask2_16m_direct_catchment.nc'
sar = Dataset(sar_path, 'r')

sar_wliq = sar['soilmoisture']*gis['cmask']/100
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']
spa_S = tres['S']

dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 
dates_spa = tvec

# driest and wettest days
sarsum = np.nansum(sar_wliq, axis=1)
sarsum = np.nansum(sarsum, axis=1)

# index in sar data
sar_low = int(np.where(sarsum == np.nanmin(sarsum))[0])
sar_hi = int(np.where(sarsum == np.nanmax(sarsum))[0])
sar_low = 43
sar_hi = 20
# day in sar data
low_day = dates_sar[sar_low].strftime("%Y-%m-%d")
hi_day = dates_sar[sar_hi].strftime("%Y-%m-%d")

# index in spa data
spa_low = np.where(tvec == low_day)[0][0]
spa_hi = np.where(tvec == hi_day)[0][0]

# sar dry and wet
sar_kosteus_hi = sar_wliq[sar_hi,:,:]
sar_kosteus_low = sar_wliq[sar_low, :,:]
sar_kosteus_hi[sar_kosteus_hi == 0] = np.nan
sar_kosteus_low[sar_kosteus_low == 0] = np.nan

# spa dry and wet
spa_kosteus_hi = spa_wliq[spa_hi, :,:]
spa_kosteus_low = spa_wliq[spa_low, :,:]
spa_kosteus_hitop = spa_wliq_top[spa_hi, :, :]
spa_kosteus_lowtop  = spa_wliq_top[spa_low, :,:]

spa_S_hi = 1e3*spa.top.local_s(spa_S[spa_hi-365])
spa_S_hi[spa_S_hi<0] = 0.0
spa_S_hi = spa._to_grid(spa_S_hi)
spa_S_low = 1e3*spa.top.local_s(spa_S[spa_low-365])
spa_S_low[spa_S_low<0] = 0.0
spa_S_low = spa._to_grid(spa_S_low)

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16,12));
ax1 = axs[0][0]
ax2 = axs[0][1]
ax3 = axs[1][0]
ax4 = axs[1][1]
ax5 = axs[0][2]
ax6 = axs[1][2]

#fig.suptitle('Volumetric water content', fontsize=15)

im1 = ax1.imshow(sar_kosteus_hi, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax1.title.set_text('SAR')
ax1.text(175, 0, f'Wet day : {hi_day}', fontsize=12)
#ax1.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im2 = ax2.imshow(spa_kosteus_hi, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax2.title.set_text('SPAFHY rootzone')
#ax2.plot(kenttarova_loc[0], kenttarova_loc[1], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)

im3 = ax3.imshow(sar_kosteus_low, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax3.title.set_text('SAR')
ax3.text(175, 0, f'Dry day : {low_day}', fontsize=12)

im4 = ax4.imshow(spa_kosteus_low, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax4.title.set_text('SPAFHY rootzone')

im5 = ax5.imshow(spa_kosteus_hitop, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax5.title.set_text('SPAFHY topsoil')

im6 = ax6.imshow(spa_kosteus_lowtop, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax6.title.set_text('SPAFHY topsoil')

ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")
ax5.axis("off")
ax6.axis("off")

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

# normalized based on maximums

spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']

# max saturation
sar_wmax = sar_kosteus_hi
spa_wliq_max = spa_kosteus_hi
spa_wliq_top_max = spa_kosteus_hitop

'''
# spa pikselikohtaiset maksimit vuodelta 2019
ind1 = np.where(tvec == dates_sar[0])[0][0]
ind2 = np.where(tvec == dates_sar[-1])[0][0]
spa_wliq = spa_wliq[ind1:ind2]
spa_wliq_max = spa_wliq[:,:, 0:spalonmax]
spa_wliq_max = np.array(np.nanmax(spa_wliq_max, axis=0))
spa_wliq_top_max = spa_wliq_top[ind1:ind2]
spa_wliq_top_max = spa_wliq_top[:, :, 0:spalonmax]
spa_wliq_top_max = np.array(np.nanmax(spa_wliq_top_max, axis=0))
#plt.imshow(np.array(spa_wliq_max));plt.colorbar()
#plt.imshow(np.array(spa_wliq_top_max));plt.colorbar()
'''

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,12));
ax1 = axs[0][0]
ax2 = axs[0][1]
ax3 = axs[1][0]
ax4 = axs[1][1]
ax5 = axs[0][2]
ax6 = axs[1][2]

#fig.suptitle('Volumetric water content', fontsize=15)

im1 = ax1.imshow(sar_kosteus_hi / sar_wmax, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax1.title.set_text('SAR/SARmax')
ax1.text(300, -30, f'Wet day : {hi_day}', fontsize=12)

im2 = ax2.imshow(spa_kosteus_hi / spa_wliq_max, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax2.title.set_text('SPAFHY rootzone / rootzone max per pixel')

im3 = ax3.imshow(sar_kosteus_low / sar_wmax, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax3.title.set_text('SAR/SARmax')
ax3.text(300, -30, f'Dry day : {low_day}', fontsize=12)

im4 = ax4.imshow(spa_kosteus_low / spa_wliq_max, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax4.title.set_text('SPAFHY rootzone / rootzone max per pixel')

im5 = ax5.imshow(spa_kosteus_hitop / spa_wliq_top_max, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax5.title.set_text('SPAFHY topsoil / topsoil max per pixel')

im6 = ax6.imshow(spa_kosteus_lowtop / spa_wliq_top_max, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax6.title.set_text('SPAFHY topsoil / topsoil max per pixel')

ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")
ax5.axis("off")
ax6.axis("off")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
bar1 = fig.colorbar(im1, cax=cbar_ax)

if saveplots == True:
    plt.savefig(f'SAR_vs.SPAFHY_saturationratio_{today}.png')

#%%

# Q-Q plots

import numpy as np 
import pylab 
import scipy.stats as stats
import pandas as pd

from spafhy_io import read_AsciiGrid
asc = read_AsciiGrid('C:\PALLAS_RAW_DATA\Lompolonjanka\cmask_iso_final2.asc', setnans=True)
asc = asc[0]

# reading sar data
sar_path = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\processed\SAR_PALLAS_2019_mask2.nc'
sar = Dataset(sar_path, 'r')
sar_wliq = sar['soilmoisture']
sar_wliq = sar_wliq * asc
spa_wliq_top = bres['Wliq_top']
spa_wliq = bres['Wliq']

ind = np.where(np.isin(tvec, dates_sar))
ind = list(np.array(ind)[0][:])

sar_wliq = sar_wliq[:, sarlatmax:sarlatmin+1, sarlonmin:sarlonmax+1]
spa_wliq = spa_wliq[ind, :, 0:spalonmax+1]
spa_wliq_top = spa_wliq_top[ind, :, 0:spalonmax+1]

sar_flat = sar_wliq.flatten()
#sar_flat[np.where(sar_flat <= 0)] = np.nan
spa_flat = spa_wliq.flatten()
spa_top_flat = spa_wliq_top.flatten()

flat_pd = pd.DataFrame()
flat_pd['sar'] = sar_flat/100
flat_pd['spa'] = spa_flat
flat_pd['spa_top'] = spa_top_flat
flat_pd = flat_pd.loc[np.isfinite(flat_pd['sar']) & np.isfinite(flat_pd['spa']) & np.isfinite(flat_pd['spa_top'])]
flat_pd = flat_pd.loc[(flat_pd['spa'] > 0) & (flat_pd['spa'] < 1)]

#g = sns.scatterplot(flat_pd['sar'], flat_pd['spa'], alpha=0.0001, s=2)
#g.set(ylim=(-0.1, 1.0))
#g.set(xlim=(-0.1, 1.0))

x = sns.regplot(x=flat_pd['sar'], y=flat_pd['spa'], scatter_kws={'s':1, 'alpha':0.01}, line_kws={"color": "red"})
x.set(ylim=(0, 1))
x.set(xlim=(0, 1))

if saveplots == True:
    plt.savefig(f'sar_spa_qq_{today}.pdf')
    
#%%

# kenttärova point plots
# peat point plots
import numpy as np 
import pylab 
import scipy.stats as stats
import pandas as pd

from spafhy_io import read_AsciiGrid

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
kenttarova_loc = list([int(kenttarova_loc[1]), int(kenttarova_loc[0])])
asc = read_AsciiGrid('C:\PALLAS_RAW_DATA\Lompolonjanka\cmask_iso_final2.asc', setnans=True)
asc = asc[0]

# ec observation data
ec_fp = r'C:\SpaFHy_v1_Pallas\data\obs\ec_soilmoist.csv'
ecmoist = pd.read_csv(ec_fp, sep=';', date_parser=['time'])
ecmoist['time'] = pd.to_datetime(ecmoist['time'])
ecmoist = ecmoist[(ecmoist['time'] >= dates_sar[0]) & (ecmoist['time'] <= dates_sar[-1])]

# soilscouts at Kenttarova
folder = r'C:\SpaFHy_v1_Pallas\data\obs'
soil_file = 'soilscouts_s3_s5_s18.csv'
fp = os.path.join(folder, soil_file)
soilscout = pd.read_csv(fp, sep=';', date_parser=['time'])
soilscout['time'] = pd.to_datetime(soilscout['time'])
soilscout = soilscout[(soilscout['time'] >= dates_sar[0]) & (soilscout['time'] <= dates_sar[-1])]
soilscout.index = soilscout['time']


# reading sar data
sar_path = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\processed\SAR_PALLAS_2019_mask2.nc'
sar = Dataset(sar_path, 'r')
sar_wliq = sar['soilmoisture']
sar_wliq = sar_wliq * asc
spa_wliq_top = bres['Wliq_top']
spa_wliq = bres['Wliq']

ind = np.where(np.isin(tvec, dates_sar))
ind = list(np.array(ind)[0][:])


Prec = FORC['Prec'][ind]
sar_wliq = sar_wliq[:, sarlatmax:sarlatmin+1, sarlonmin:sarlonmax+1]
spa_wliq = spa_wliq[ind, :, 0:spalonmax+1]
spa_wliq_top = spa_wliq_top[ind, :, 0:spalonmax+1]

kr_point = pd.DataFrame()
kr_point['time'] = dates_sar
kr_point['sar_wliq_kr'] = sar_wliq[:,kenttarova_loc[0], kenttarova_loc[1]]/100
kr_point['spa_wliq_kr'] = spa_wliq[:,kenttarova_loc[0], kenttarova_loc[1]]
kr_point['spa_wliq_top_kr'] = spa_wliq_top[:,kenttarova_loc[0], kenttarova_loc[1]]
kr_point['prec'] = Prec
kr_point['sar_wliq_lv'] = sar_wliq[:,45,55]/100
kr_point['spa_wliq_lv'] = spa_wliq[:,45,55]
kr_point.index = kr_point['time']


kr_point['sar_wliq_lv'] = np.nanmean(sar_wliq[:,(45-1, 45, 45+1, 45, 45, 45),
                    (55, 55, 55, 55-1, 55, 55+1)], axis=1)/100

kr_point['sar_wliq_kr'] = np.nanmean(sar_wliq[:,(kenttarova_loc[0]-1, kenttarova_loc[0], kenttarova_loc[0]+1, kenttarova_loc[0], kenttarova_loc[0], kenttarova_loc[0]),
                    (kenttarova_loc[1], kenttarova_loc[1], kenttarova_loc[1], kenttarova_loc[1]-1, kenttarova_loc[1], kenttarova_loc[1]+1)], axis=1)/100


# Plotting
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7,5));
ax1 = axs[0]
ax2 = axs[1]


l1 = ax1.plot(kr_point['sar_wliq_kr'], marker='', color='black', linewidth=1, alpha=0.9)
#l2 = ax1.plot(kr_point['spa_wliq_top_kr'], marker='', color='red', linewidth=1, alpha=0.3)
#ax.fill_between(l1, l2, color='grey', alpha='0.5')
l3 = ax1.plot(kr_point['spa_wliq_kr'], marker='', color='red', linewidth=1, alpha=1.0)
l6 = ax1.plot(soilscout['s3'],marker='', color='blue', linewidth=1, alpha=1.0 )
ax1.text(kr_point['time'][-12], 0.9, 'Kenttärova')
ax1.legend(['SAR ~-5cm', 'SpaFHy rootzone -40cm', 'Moisture sensor ~-5cm'], bbox_to_anchor =(1.0, 1.3), ncol = 3)
l4 = ax1.plot(kr_point['time'][sar_low], kr_point['sar_wliq_kr'][sar_low], marker='o', mec='r', mfc='k', alpha=0.8, ms=6.0)
l4 = ax1.plot(kr_point['time'][sar_hi], kr_point['sar_wliq_kr'][sar_hi], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)
ax1.set_ylim(0,1.0)
ax1.set_ylabel('Volumetric water content')
ax1.xaxis.set_visible(False)
#ax1.legend(['SAR', 'SpaFHy rootzone'], bbox_to_anchor =(0.75, 1.15), ncol = 4)

l4 = ax2.plot(kr_point['sar_wliq_lv'], marker='', color='black', linewidth=1, alpha=0.9)
l7 = ax2.plot(kr_point['spa_wliq_lv'], marker='', color='red', linewidth=1, alpha=1.0)
l4 = ax2.plot(kr_point['time'][sar_low], kr_point['sar_wliq_lv'][sar_low], marker='o', mec='r', mfc='k', alpha=0.8, ms=6.0)
l4 = ax2.plot(kr_point['time'][sar_hi], kr_point['sar_wliq_lv'][sar_hi], marker='o', mec='b', mfc='k', alpha=0.8, ms=6.0)
ax2.text(kr_point['time'][-12], 0.9, 'Lompolonjänkä')
ax2.set_ylim(0,1.0)
ax2.set_ylabel('Volumetric water content')

#for tick in ax1.get_xticklabels():
#        tick.set_rotation(30)
for tick in ax2.get_xticklabels():
        tick.set_rotation(30)
        '''
ax1=ax1.twinx()   
ax1.bar(Prec.index, 3600*24.0*Prec, color='blue', width=1)
ax1.set_ylabel(r'P (mm d$^{-1}$)')
ax1.set_ylim([0, 70])
ax1.invert_yaxis()

ax2=ax2.twinx()   
ax2.bar(Prec.index, 3600*24.0*Prec, color='blue', width=1)
ax2.set_ylabel(r'P (mm d$^{-1}$)')
ax2.set_ylim([0, 70])
ax2.invert_yaxis()
'''
if saveplots == True:
    plt.savefig(f'soilmoist_kenttarova_spa_sar_{today}.pdf')
    plt.savefig(f'soilmoist_kenttarova_spa_sar_{today}.png')

#%%

# S1 kaistat
# soilscouts at Kenttarova
folder = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\muut'
S1_file = 'S1_ajat_kaistat.txt'
fp = os.path.join(folder, S1_file)
S1 = pd.read_csv(fp, sep=',', delim_whitespace=True, date_parser=['Date'])
S1 = S1[['Date', 'Time', '(Julian']]
S1 = S1.rename(columns={'(Julian': 'SwathPass', 'Time': 'Julian day'})

S1['SwathPass'] == '*m'

#%%

# define a big catchment mask
from spafhy_io import read_AsciiGrid
asc = read_AsciiGrid('C:\SpaFHy_v1_Pallas\data\C16\maintype.dat', setnans=True)

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6));
ax1 = axs[0]
ax2 = axs[1]

im1 = ax2.imshow(asc[0], cmap='plasma')
ax1.title.set_text('SPAFHY topsoil moisture on wet day')
im2 = ax1.imshow(spa_kosteus_hitop, cmap='coolwarm_r', vmin=0.0, vmax=1.0, aspect='equal')
ax2.title.set_text('Site main class 1-4')
ax2.axis('off')

cbar = fig.colorbar(im1, ticks=[1, 2, 3, 4], orientation='horizontal')
cbar.ax.set_xticklabels(['mineral soil', 'fen', 'peatland', 'open mire'])  # horizontal colorbar

#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
#bar1 = fig.colorbar(im2, cax=cbar_ax)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.05, 0.15, 0.015, 0.7])
bar2 = fig.colorbar(im2, cax=cbar_ax)

if saveplots == True:
    plt.savefig(f'SPATOPSOIL_KASVUPAIKKA_{today}.png')
    
    
    
#%%

from spafhy_io import read_AsciiGrid

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
kenttarova_loc = list([int(kenttarova_loc[0]), int(kenttarova_loc[1])])

tm = range(365,(len(dat['time'][:])))

TR = np.array(cres['Transpi'][tm,kenttarova_loc[0], kenttarova_loc[1]])  # transpi
EF = np.array(cres['Efloor'][tm,kenttarova_loc[0], kenttarova_loc[1]])  # floor evap
IE = np.array(cres['Evap'][tm,kenttarova_loc[0], kenttarova_loc[1]])# interception evap
ET = TR + EF + IE

ET = np.array(cres['ET'][tm,kenttarova_loc[0], kenttarova_loc[1]])
    

#%%

# sar vs spa soil moisture timelapse
import pandas as pd
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec


# define a big catchment mask
from spafhy_io import read_AsciiGrid
asc = read_AsciiGrid('C:\PALLAS_RAW_DATA\Lompolonjanka\cmask_iso_final2.asc', setnans=True)
asc = asc[0]

# reading sar data
sar_path = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\processed\SAR_PALLAS_2019_mask2_resampled16b.nc'
#sar_path = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\processed\SAR_PALLAS_2019_mask2.nc'
sar = Dataset(sar_path, 'r')
sar_wliq = sar['soilmoisture']
spa_wliq = bres['Wliq']
spa_wliq_top = bres['Wliq_top']
spa_S = tres['S']
# dates
dates_sar = sar['time'][:]
dates_sar = pd.to_datetime(dates_sar, format='%Y%m%d') 
dates_spa = tvec
days = dates_sar.strftime("%Y-%m-%d")

# date indexes of spa
ind_day = np.where(np.isin(dates_spa, dates_sar))
ind_day = list(np.array(ind_day)[0][:])
# lat lon indexes
sarlat = list(np.array(sar['lat'][:]))
sarlon = list(np.array(sar['lon'][:]))
spalat = list(np.array(dat['lat'][:]))
spalon = list(np.array(dat['lon'][:]))

# find the mutual in sar
sarlatmin = sarlat.index(min(spalat))
sarlatmax = sarlat.index(max(spalat))
sarlonmin = sarlon.index(min(spalon))
sarlonmax = sarlon.index(max(sarlon))

# find the mutual in spa
spalonmax = spalon.index(max(sarlon))

# mutual lat lons and days
sar_kosteus = sar_wliq * asc
sar_kosteus = sar_kosteus[:, sarlatmax:sarlatmin+1, sarlonmin:sarlonmax+1]
sar_kosteus = sar_kosteus / 100
sar_kosteus[sar_kosteus == 0] = np.nan
spa_kosteus = spa_wliq[ind_day, :, 0:spalonmax+1]
spa_kosteus_top = spa_wliq_top[ind_day, :, 0:spalonmax+1]

# data for discharge and precipiatio 
start = days[0]
end = days[-1]
f0 = int(np.where(tvec == start)[0])
f1 = int(np.where(tvec == end)[0])
tvec0 = tvec[f0:f1]
tvec0_ind = np.where(np.isin(tvec0, dates_sar))
tvec0_ind = list(np.array(tvec0_ind)[0][:])

Qt = pd.DataFrame()
Qt['Q'] = 1e3*np.array(tres['Qt'])[f0:f1] # modeled streamflow
Qt.index = pd.to_datetime(tvec[f0:f1])
Qm = Qmeas[(Qmeas.index >= start) & (Qmeas.index < end) ]
Prec = FORC['Prec'].iloc[f0:f1] # precipitation

f, g = np.where(spa.GisData['cmask'] == 1)

# data for soil moisture and precipitation
# these are from previous code snipped where kenttärova point plots are
ecmoist = ecmoist.loc[(ecmoist['time'] >= start) & (ecmoist['time'] <= end)]
soilscout = soilscout.loc[(soilscout['time'] >= start) & (soilscout['time'] <= end)]
Wliq_final = Wliq_final.loc[(Wliq_final['time'] >= start) & (Wliq_final['time'] <= end)]


### TIMELAPSE PLOT
fps = 2
nSeconds = 30
snapshots1 = sar_kosteus
snapshots2 = spa_kosteus
snapshots3 = spa_kosteus_top
'''
# First set up the figure, the axis, and the plot element we want to animate
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,12));
ax1 = axs[0][0]
ax2 = axs[0][1]
ax3 = axs[0][2]
ax4 = axs[1][0]
ax5 = axs[1][1]
ax6 = axs[1][2]
'''

fig = plt.figure(figsize=(18,10))
gs = fig.add_gridspec(3, 6)
ax1 = fig.add_subplot(gs[0:2, 0:2])
#f3_ax1.set_title('gs[1, :-1]')
ax2 = fig.add_subplot(gs[0:2, 2:4])
#f3_ax2.set_title('gs[1:, -1]')
ax3 = fig.add_subplot(gs[0:2, 4:6])
#f3_ax3.set_title('gs[-1, 0]')
ax5 = fig.add_subplot(gs[2, 0:6])
#f3_ax4.set_title('gs[1, :]')


a1 = snapshots1[0]
a2 = snapshots2[0]
a3 = snapshots3[0]

im1 = ax1.imshow(a1, cmap='coolwarm_r', interpolation='none', aspect='auto', vmin=0, vmax=1)
ax1.title.set_text('SAR')
ttl = ax1.text(260, -20, f'Soil volumetric water content {days[0]}', fontsize=12)
im2 = ax2.imshow(a2, cmap='coolwarm_r', interpolation='none', aspect='auto', vmin=0, vmax=1)
ax2.title.set_text('SPAFHY ROOT')
im3 = ax3.imshow(a3, cmap='coolwarm_r', interpolation='none', aspect='auto', vmin=0, vmax=1)
ax3.title.set_text('SPAFHY TOP')

#im4 = ax5.plot(Qm, 'k-', linewidth=1.0, markersize=4)
#im4 = ax5.plot(Qt, 'r-', linewidth=1.0, markersize=4)
im4 = ax5.plot(soilscout['s18'], 'k-', linewidth=1.0, markersize=4)
im4 = ax5.plot(Wliq_final['root_kr'], 'r-', linewidth=1.0, markersize=4)

line, = ax5.plot(tvec0[tvec0_ind[0]], 0.15, marker='o', mec='k', mfc='g', alpha=0.7, ms=8.0)
ax5.title.set_text('Measured and modeled soil moisture at Kenttärova')

ax5.set_ylim(0.1, 0.6)
ax5.legend(['measured', 'modeled'], bbox_to_anchor =(0.11, 0.97), ncol = 1) 
ax5.set_ylabel(r'$\langle Q_f \rangle$ (mm d$^{-1}$)')
ax5.set_ylabel(r'Volumetric water content')
ax7=ax5.twinx()   
ax7.bar(tvec0, 3600*24.0*Prec, color='b', width=0.5)
ax7.set_ylabel(r'P (mm d$^{-1}$)'); plt.ylim([0, 60])
ax7.invert_yaxis()
for tick in ax5.get_xticklabels():
        tick.set_rotation(30)

ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.45, 0.015, 0.4])
bar1 = fig.colorbar(im1, cax=cbar_ax)

def animate_func(i):
    im1.set_array(snapshots1[i])
    im2.set_array(snapshots2[i])
    im3.set_array(snapshots3[i])
    ttl.set_text(f'Soil volumetric water content {days[i]}')
    #ttl.set_text(days[i])
    day = tvec0[tvec0_ind[i]]
    line.set_data(day, 0.15)
    return [im1]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

anim.save('sar_spafhy_tl.gif', fps=fps)
#anim.save('sar_spafhy_tl.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])


#%%

import pandas as pd
from spafhy_io import read_AsciiGrid

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
kenttarova_loc = list([int(kenttarova_loc[0]), int(kenttarova_loc[1])])

#sve_gtk_pintamaa, _, _, _, _ = read_AsciiGrid(r'C:\SpaFHy_v1_Pallas\data\C16\soilclass.dat')
# kenttarova soilclass = 2
#soil2 = np.where((gis['soilclass'] == 2) & (gis['cmask'] == 1))

#skip warm up
tm = range(365,(len(dat['time'][:])))

# SpaFHy rootzone and top soil moisture
Wliq_root = np.array(bres['Wliq'][tm])
Wliq_top = np.array(bres['Wliq_top'][tm])

# Both at kenttärova and mean in soilclass 2
Wliq_root_kr = Wliq_root[:, kenttarova_loc[0], kenttarova_loc[1]]
Wliq_top_kr = Wliq_top[:, kenttarova_loc[0], kenttarova_loc[1]]

# Wliq means within the soilclass 2
Wliq_root_mean = Wliq_root[:,soil2[0],soil2[1]]
Wliq_root_mean = np.nanmean(Wliq_root_mean, axis=(1))
Wliq_top_mean = Wliq_top[:,soil2[0],soil2[1]]
Wliq_top_mean = np.nanmean(Wliq_top_mean, axis=(1))

# new dataframe for spafhy data
Wliq_final = pd.DataFrame()
Wliq_final['time'] = tvec[tm]
Wliq_final['root_kr'] = Wliq_root_kr
Wliq_final['root_mean'] = Wliq_root_mean
Wliq_final['top_kr'] = Wliq_top_kr
Wliq_final['top_mean'] = Wliq_top_mean
    
# indexes as dates
soilscout.index = soilscout['time']
ecmoist.index = ecmoist['time']
Wliq_final.index = Wliq_final['time']

Prec = FORC['Prec'][tm[0]:len(tm)+tm[0]]

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,4));
ax1 = axs[0]
ax2 = axs[1]


l1 = ax1.plot(ecmoist['SH-5A'], marker='', color='black', linewidth=1, alpha=0.4)
l2 = ax1.plot(ecmoist['SH-5B'], marker='', color='black', linewidth=1, alpha=0.6)
l3 = ax1.plot(soilscout['s3'], marker='', color='black', linewidth=1, alpha=0.2) #-0.05
#ax.fill_between(l1, l2, color='grey', alpha='0.5')
l3 = ax1.plot(Wliq_final['top_kr'], marker='', color='red', linewidth=1, alpha=0.7)
ax1.set_ylim(0,0.70)
ax1.set_ylabel('Volumetric water content')
ax1.legend(['5A = -0.05', '5B = -0.05', 's3 = -0.05', 'Wliq_top'], bbox_to_anchor =(0.95, 1.15), ncol = 4)

l4 = ax2.plot(ecmoist['SH-20A'], marker='', color='black', linewidth=1, alpha=0.4)
l5 = ax2.plot(ecmoist['SH-20B'], marker='', color='black', linewidth=1, alpha=0.6)
l6 = ax2.plot(soilscout['s18'], marker='', color='black', linewidth=1, alpha=0.2)
#ax.plot(soilscout['s18'], marker='', color='blue', linewidth=1)
l7 = ax2.plot(Wliq_final['root_kr'], marker='', color='red', linewidth=1, alpha=0.7)
ax2.set_ylim(0,0.70)
ax2.set_ylabel('Volumetric water content')
ax2.legend(['20A = -0.20', '20B = -0.20', 's18 = -0.20', 'Wliq_root'], bbox_to_anchor =(0.95, 1.15), ncol = 4)

for tick in ax1.get_xticklabels():
        tick.set_rotation(30)
for tick in ax2.get_xticklabels():
        tick.set_rotation(30)
        
ax1=ax1.twinx()   
ax1.bar(Prec.index, 3600*24.0*Prec, color='blue', width=1)
ax1.set_ylabel(r'P (mm d$^{-1}$)')
ax1.set_ylim([0, 70])
ax1.invert_yaxis()

ax2=ax2.twinx()   
ax2.bar(Prec.index, 3600*24.0*Prec, color='blue', width=1)
ax2.set_ylabel(r'P (mm d$^{-1}$)')
ax2.set_ylim([0, 70])
ax2.invert_yaxis()

if saveplots == True:
    plt.savefig(f'soilmoist_kenttarova_{today}.pdf')
    plt.savefig(f'soilmoist_kenttarova_{today}.png')

#%%

# ET simulated vs. measured

# cell locations of kenttarova
kenttarova, _, _, _, _ = read_AsciiGrid(r'C:\PALLAS_RAW_DATA\Lompolonjanka\16b\sve_kenttarova_soilmoist.asc')
kenttarova_loc = np.where(kenttarova == 0)
kenttarova_loc = list([int(kenttarova_loc[0]), int(kenttarova_loc[1])])

TR = np.array(cres['Transpi'][ix:, kenttarova_loc[0], kenttarova_loc[1]])  # transpi
EF = np.array(cres['Efloor'][ix:, kenttarova_loc[0], kenttarova_loc[1]])  # floor evap
IE = np.array(cres['Evap'][ix:, kenttarova_loc[0], kenttarova_loc[1]])# interception evap
ET = TR + EF + IE

P = np.array(FORC['Prec'][ix:-1])
ax = []
for i in range(len(P)):
    if P[i] == 0:
        if P[i-1] == 0:
            ax.append(i) 

dates = tvec[ix:-1]

# ET at Kenttarova
folder = r'C:\SpaFHy_v1_Pallas\data\obs'
file = 'ec_kr_et.csv'
fp = os.path.join(folder, file)
ec_kr = pd.read_csv(fp, sep=';', date_parser=['time'])
#ec_kr['time'] = pd.to_datetime(soilscout['time'])
#ec_kr = ec_kr[(ec_kr['time'] >= dates[0]) & (ec_kr['time'] <= dates[-1])]
ec_kr.index = ec_kr['time']
ec_kr = ec_kr[['ET-2', 'ET-3']]


ET_kr = pd.DataFrame()
ET_kr['ET_sim'] = ET
ET_kr['time'] = dates
ET_kr.index = ET_kr['time']
ET_kr = ET_kr[['ET_sim']]
ET_kr['P'] = P
ET_kr = ET_kr.iloc[ax]


ec_kr = ET_kr.merge(ec_kr, left_index=True, right_index=True)
ec_kr['ET-2'][ec_kr['ET-2'] < 0] = np.nan
ec_kr['ET-3'][ec_kr['ET-3'] < 0] = np.nan
ec_kr = ec_kr[(~np.isnan(ec_kr['ET-2']))]
ec_kr = ec_kr[(~np.isnan(ec_kr['ET-3']))]

## Plotting
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,8));
ax1 = axs[0]
ax2 = axs[1]

l1 = ax1.plot(ec_kr['ET-2'], marker='', color='blue', linewidth=1, alpha=0.6)
l2 = ax1.plot(ec_kr['ET_sim'], marker='', color='red', linewidth=1, alpha=0.3)
#ax.fill_between(l1, l2, color='grey', alpha='0.5')
#ax1.set_ylim(0,0.70)
#ax1.set_ylabel('Volumetric water content')
ax1.legend(['ET-2', 'ET_sim'], loc='upper right', ncol = 2)

l3 = ax2.plot(ec_kr['ET-3'], marker='', color='blue', linewidth=1, alpha=0.6)
l4 = ax2.plot(ec_kr['ET_sim'], marker='', color='red', linewidth=1, alpha=0.4)
#ax.fill_between(l1, l2, color='grey', alpha='0.5')
#ax1.set_ylim(0,0.70)
#ax1.set_ylabel('Volumetric water content')
ax2.legend(['ET-3', 'ET_sim'], loc='upper right', ncol = 2)


#%%

# ET scatterplot simulated vs. measured



