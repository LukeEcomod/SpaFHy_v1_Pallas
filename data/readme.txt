GIS DATA
C2 = 2M RESOLUTION PALLAS
C3 = ORIGINAL 16M RESOLUTION
C16 = UPDATED 16M RESOLUTION

RUNOFF DATA
Runoffs_SVEcatchments_mmd = ORIGINAL
Runoffs1d_SVEcatchments_mmd = UPDATED
Runoffs1h_SVEcatchments_mmd = UPDATED

Weather_C3 = ORIGINAL
Weather_C3_Kenttarova =
- FMI hourly observations from kenttarova resampled to daily resolution (meteohourlytodaily.py)
- Kenttärova radiation data (2017-2019 orig FMI, from Kashif) resampled to daily and merged to previous data (radiationdata.py)
- The missing radiation data 2013-2017 from LUKE grid weather data added (meteo_grid_fmi_merge.py)