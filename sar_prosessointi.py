# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:58:23 2021

@author: janousu
"""

from netCDF4 import Dataset #, date2num
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
import pyproj
import datetime
import os
import glob
import scipy

'''
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import glob
import os
from pyproj import CRS
'''

fp = r'C:\PALLAS_RAW_DATA\SAR_maankosteus'
sc = 'kosteuskartta*.nc'

q = os.path.join(fp, sc)
files = glob.glob(q)

# irregular grid
koord = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\Pallas_S1_lat_lon.nc'
koord = Dataset(koord, 'r')
latitude = koord['latitude'][:]
latitude = np.array(latitude)
longitude = koord['longitude'][:]
longitude = np.array(longitude)

# regular grid
lats = koord['latitude'][:,0]
lats = np.array(lats)
lons = koord['longitude'][0,:]
lons = np.array(lons)





projIn = pyproj.Proj(init='epsg:4326')
projOut = pyproj.Proj(init='epsg:3067')

# transform from lon/lat
y, x = pyproj.transform(projIn, projOut, latitude, longitude)

#%%

# dates for the netcdf file (morning)
dates = [0] * len(files)
dates_int = [0] * len(files)
for i in range(len(files)):
    dates[i] = files[i][50:59]
    dates[i] = datetime.datetime.strptime(dates[i], '%d%b%Y')
    dates_int[i] = int(dates[i].strftime('%Y%m%d'))
    

'''
timefile = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\S1_ajat_kaistat.txt'
time_data = pd.read_csv(timefile, delim_whitespace=True, parse_dates=['Date'])
time_data = time_data[['Date', '(Julian']]
time_data = time_data.loc[(time_data['(Julian'] == 'IW1m') | (time_data['(Julian'] == 'IW2m') | (time_data['(Julian'] == 'IW3m')]
dates = list(time_data['Date'])
'''

#%%

# create dataset & dimensions
fname = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\test6.nc'
ncf = Dataset(fname, 'w')
ncf.description = 'SAR soil moisture Pallas'
ncf.history = 'created ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

time = ncf.createDimension('time', None)
lon = ncf.createDimension('lon', len(y))
lat = ncf.createDimension('lat', len(x))

# create variables into base and groups 'forc','eval','cpy','bu','top'
# call as createVariable(varname,type,(dimensions))
times = ncf.createVariable('time', 'f8', ('time',))
times.units = "days since 0001-01-01 00:00:00.0"

lats = ncf.createVariable('lat', 'f4', ('lat',))
lats.units = 'ETRS-TM35FIN'
lons = ncf.createVariable('lon', 'f4', ('lon',))
lons.units = 'ETRS-TM35FIN'

# Soil moisture
VWC = ncf.createVariable('value', 'f4', ('time', 'lat', 'lon',))
VWC.units = 'gridcell volumetric water content %'


#%%

# soil  moist data files



    


test_file = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\kosteuskartta__01Aug2019.nc'
data = Dataset(test_file, 'r')
kosteus = data['kosteus'][:]

maamaski = r'C:\PALLAS_RAW_DATA\SAR_maankosteus\maamaski.nc'
maamaski = Dataset(maamaski, 'r')




'''
# writing the new netcdf file with all the info
sar = Dataset(r'C:\PALLAS_RAW_DATA\SAR_maankosteus\sar.nc','w', format='NETCDF4') #'w' stands for write
tempgrp = sar.createGroup('Temp_data')

tempgrp.createDimension('lon', len(longitude))
tempgrp.createDimension('lat', len(latitude))
tempgrp.createDimension('moisture', len(latitude))
tempgrp.createDimension('time', len(dates))


q = os.path.join()



# select the files
q = os.path.join(dirpath, search_criteria)

# glob function can be used to list files from a directory with specific criteria
dem_fps = glob.glob(q)
    
# List for the source files
src_files_to_mosaic = []

# Iterate over raster files and add them to source -list in 'read mode'
for fp in dem_fps:
    src = rasterio.open(fp)
    src_files_to_mosaic.append(src)
'''