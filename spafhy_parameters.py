# -*- coding: utf-8 -*-
"""
DEFAULT PARAMETERS OF SPAFHY FOR A SINGLE CATCHMENT AND POINT-SCALE SIMULATIONS

Created on Mon Jun 25 18:34:12 2018

@author: slauniai

Last edit: 11.5.2020 / SL: canopygrid can now have multiple vegetation types.
Phenology is common to all, LAI-cycle common to all deciduous
"""


def parameters():
    
    pgen = {'catchment_id': '1',
            'gis_folder': r'C:\SpaFHy_v1_Pallas\data\C16',
            'forcing_file': r'C:\SpaFHy_v1_Pallas\data\Kenttarova_forcing.csv',
            'runoff_file': r'C:\SpaFHy_v1_Pallas\data\obs\Runoffs1d_SVEcatchments_mmd.csv', #
            'ncf_file': r'C3.nc',
            'results_folder': r'C:\SpaFHy_v1_Pallas\Results',
            'start_date': '2016-01-01',
            'end_date': '2019-10-01',
            'spinup_end': '2016-05-01',
            'dt': 86400.0,
            'spatial_cpy': True,
            'spatial_soil': True     
           }
    
    # canopygrid
    pcpy = {'loc': {'lat': 67.995, 'lon': 24.224},
            'flow' : { # flow field
                     'zmeas': 2.0,
                     'zground': 0.5,
                     'zo_ground': 0.01
                     },
            'interc': { # interception
                        'wmax': 1.5, # storage capacity for rain (mm/LAI)
                        'wmaxsnow': 4.5, # storage capacity for snow (mm/LAI),
                        },
            'snow': {
                    # degree-day snow model / energy balance
                    'kmelt': 2.8934e-05, # melt coefficient in open (mm/s)
                    'kfreeze': 5.79e-6, # freezing coefficient (mm/s)
                    'r': 0.05, # maximum fraction of liquid in snow (-)
                    'albpow': 1.0, # parameter lowering albedo for aging snow 
                    'albground': 0.21, # albedo for snowfree ground
                    'cAtten': 1.0, # attenuation coeffient for canopy longwave radiation (Montehit and Unsworth 2013 fig 8.2)
                    'RDthres': 0.001, # amount of precipitation [m] above which a day is considered cloudy (affects long wave inputs)
                    'Tmin': 1.0, # threshold below which all precipitation is snow
                    'Tmax': 0.0, # threshold above which all precipitation is water (note in between will be divided to rain and snow)
                    },
                    
            # canopy conductance
            'physpara': {                          
                        'kp': 0.6, # canopy light attenuation parameter (-)
                        'rw': 0.20, # critical value for REW (-),
                        'rwmin': 0.02, # minimum relative conductance (-)
                        # soil evaporation
                        'gsoil': 1e-2 # soil surface conductance if soil is fully wet (m/s)
                        },
                                                    
            'spec_para': {
                        'conif': {'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                    'g1': 2.1, # stomatal parameter
                                    'q50': 50.0, # light response parameter (Wm-2)
                                    'lai_cycle': False,
                                     },
                        'decid': {'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                     'g1': 3.5, # stomatal parameter
                                     'q50': 50.0, # light response parameter (Wm-2)
                                     'lai_cycle': True,
                                     },                                 
                        'shrub':    {'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                     'g1': 3.0, # stomatal parameter
                                     'q50': 50.0, # light response parameter (Wm-2)
                                     'lai_cycle': False,
                                     },
                        'grass':    {'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                     'g1': 5.0, # stomatal parameter
                                     'q50': 50.0, # light response parameter (Wm-2)
                                     'lai_cycle': True,
                                     },
                        },
            'phenopara': {
                           # phenology
                           'smax': 18.5, # degC
                           'tau': 13.0, # days
                           'xo': -4.0, # degC
                           'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
                           
                           # annual cycle of leaf-area in deciduous trees
                           'lai_decid_min': 0.1, # minimum relative LAI (-)
                           'ddo': 45.0, # degree-days for bud-burst (5degC threshold)
                           'ddur': 23.0, # duration of leaf development (days)
                           'sdl': 9.0, # daylength for senescence start (h)
                           'sdur': 30.0, # duration of leaf senescence (days),
                         },                                                   

            'state': {# LAI is annual maximum LAI and for gridded simulations are input from GisData!
                      # keys must be 'LAI_ + key in spec_para
                      'LAI_conif': 1.0,
                      'LAI_decid': 1.0,
                      'LAI_shrub': 0.1, 
                      'LAI_grass': 0.2,
                      'hc': 16.0, # canopy height (m)
                      'cf': 0.6, # canopy closure fraction (-)
                       #initial state of canopy storage [mm] and snow water equivalent [mm]
                       'w': 0.0, # canopy storage mm
                       'swe': 0.0, # snow water equivalent mm
                       'Wice': 0.0, # ice in snowpack
                       'Wliq': 0.0, # liquid water in snowpack
                       'd_nosnow': 1.0, # days since snowfall
                       'd_snow': 0.0, # days with snow on the ground
                       'Tsnow': -4.0, # snow temperature 
                       'alb': 0, # initial albedo
                       'emAir': 0 # intiail air emissivity
                       }
            }

        
    # BUCKET
    pbu = {'depth': 0.4,  # root zone depth (m)
           # following soil properties are used if spatial_soil = False
           'poros': 0.43, # porosity (-)
           'fc': 0.33, # field capacity (-)
           'wp': 0.13,	 # wilting point (-)
           'ksat': 2.0e-6, # saturated hydraulic conductivity
           'beta': 4.7,
           #organic (moss) layer
           'org_depth': 0.05, # depth of organic top layer (m)
           'org_poros': 0.9, # porosity (-)
           'org_fc': 0.30, # field capacity (-)
           'org_rw': 0.15, # critical vol. moisture content (-) for decreasing phase in Ef
           'maxpond': 0.0, # max ponding allowed (m)
           #initial states: rootzone and toplayer soil saturation ratio [-] and pond storage [m]
           'rootzone_sat': 0.6, # root zone saturation ratio (-)
           'org_sat': 1.0, # organic top layer saturation ratio (-)
           'pond_sto': 0.0 # pond storage
           }
    
    # TOPMODEL
    ptop = {'dt': 86400.0, # timestep (s)
            'm': 0.025, # scaling depth (m)
            'ko': 0.001, # transmissivity parameter (ms-1)
            'twi_cutoff': 99.5,  # cutoff of cumulative twi distribution (%)
            'so': 0.05 # initial saturation deficit (m)
           }
    
    return pgen, pcpy, pbu, ptop

def soil_properties():
    """
    Defines 5 soil types: Fine, Medium and Coarse textured + organic Peat
    and Humus.
    Currently SpaFHy needs following parameters: soil_id, poros, dc, wp, wr,
    n, alpha, Ksat, beta
    """
    psoil = {
            'CoarseTextured':
                 {'airentry': 20.8,
                  'alpha': 0.024,
                  'beta': 3.1,
                  'fc': 0.21,
                  'ksat': 1E-04,
                  'n': 1.2,
                  'poros': 0.41,
                  'soil_id': 1.0,
                  'wp': 0.10,
                  'wr': 0.05,
                 },
             'MediumTextured': 
                 {'airentry': 20.8,
                  'alpha': 0.024,
                  'beta': 4.7,
                  'fc': 0.33,
                  'ksat': 1E-05,
                  'n': 1.2,
                  'poros': 0.43,
                  'soil_id': 2.0,
                  'wp': 0.13,
                  'wr': 0.05,
                 },
             'FineTextured': 
                 {'airentry': 34.2,
                  'alpha': 0.018, # van genuchten parameter
                  'beta': 7.9,
                  'fc': 0.34,
                  'ksat': 1E-06, # saturated hydraulic conductivity
                  'n': 1.16, # van genuchten parameter
                  'poros': 0.5, # porosity (-)
                  'soil_id': 3.0,
                  'wp': 0.25, # wilting point (-)
                  'wr': 0.07,
                 },
             'Peat':
                 {'airentry': 29.2,
                  'alpha': 0.08, # Menbery et al. 2021
                  'beta': 6.0,
                  'fc': 0.53, # Menbery et al. 2021
                  'ksat': 6e-05, # Menbery et al. 2021
                  'n': 1.75, # Menbery et al. 2021
                  'poros': 0.93, # Menbery et al. 2021
                  'soil_id': 4.0,
                  'wp': 0.36, # Menbery et al. 2021
                  'wr': 0.0,
                 },
              'Humus':
                 {'airentry': 29.2,
                  'alpha': 0.123,
                  'beta': 6.0,
                  'fc': 0.35,
                  'ksat': 8e-06,
                  'n': 1.28,
                  'poros': 0.85,
                  'soil_id': 5.0,
                  'wp': 0.15,
                  'wr': 0.01,
                 },
            }
    return psoil

def topsoil():
    """
    Properties of typical topsoils
    Following main site type (1-4) classification
    """
    topsoil = {
        'mineral':{
            'topsoil_id': 1,
            'org_depth': 0.05,
            'org_poros': 0.9,
            'org_fc': 0.33,
            'org_rw': 0.15
            },
        'fen':{
            'topsoil_id': 2,
            'org_depth': 0.05,
            'org_poros': 0.9,
            'org_fc': 0.514,
            'org_rw': 0.15
            },
        'peatland':{
            'topsoil_id': 3,
            'org_depth': 0.05,
            'org_poros': 0.9,
            'org_fc': 0.514,
            'org_rw': 0.15
            },
        'openmire':{
            'topsoil_id': 4,
            'org_depth': 0.05,
            'org_poros': 0.9,
            'org_fc': 0.514,
            'org_rw': 0.15
            }
        }
    return topsoil


def parameters_FIHy():
    # parameter file for running SpaFHy_point at FIHy-site
    
    pgen = {'catchment_id': 'FIHy',
            'gis_folder': None,
            'forcing_file': r'c:/Repositories/SpaFHy_v1_Pallas/Data/HydeDaily2000-2010.txt',
            'runoff_file': None,
            'output_file': r'c:/Repositories/SpaFHy_v1_Pallas/Results/FIHy_test',
            'start_date': '2013-01-01',
            'end_date': '2016-12-31',
            'spinup_end': '2013-12-31',
            'dt': 86400.0,
            'spatial_cpy': False,
            'spatial_soil': False     
           }
    
    # canopygrid
    pcpy = {'loc': {'lat': 61.4, 'lon': 23.7},
            'flow' : { # flow field
                     'zmeas': 2.0,
                     'zground': 0.5,
                     'zo_ground': 0.01
                     },
            'interc': { # interception
                        'wmax': 1.5, # storage capacity for rain (mm/LAI)
                        'wmaxsnow': 4.5, # storage capacity for snow (mm/LAI),
                        },
            'snow': {
                    # degree-day snow model
                    'kmelt': 2.8934e-05, # melt coefficient in open (mm/s)
                    'kfreeze': 5.79e-6, # freezing coefficient (mm/s)
                    'r': 0.05 # maximum fraction of liquid in snow (-)
                    },
                    
            # canopy conductance
            'physpara': {                          
                        'kp': 0.6, # canopy light attenuation parameter (-)
                        'rw': 0.20, # critical value for REW (-),
                        'rwmin': 0.02, # minimum relative conductance (-)
                        # soil evaporation
                        'gsoil': 1e-2 # soil surface conductance if soil is fully wet (m/s)
                        },
                                                    
            'spec_para': {
                        'conif': {'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                    'g1': 2.1, # stomatal parameter
                                    'q50': 50.0, # light response parameter (Wm-2)
                                    'lai_cycle': False,
                                     },
                        'decid': {'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                                     'g1': 3.5, # stomatal parameter
                                     'q50': 50.0, # light response parameter (Wm-2)
                                     'lai_cycle': True,
                                     },                                 
                        },
            'phenopara': {
                           # phenology
                           'smax': 18.5, # degC
                           'tau': 13.0, # days
                           'xo': -4.0, # degC
                           'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
                           
                           # annual cycle of leaf-area in deciduous trees
                           'lai_decid_min': 0.1, # minimum relative LAI (-)
                           'ddo': 45.0, # degree-days for bud-burst (5degC threshold)
                           'ddur': 23.0, # duration of leaf development (days)
                           'sdl': 9.0, # daylength for senescence start (h)
                           'sdur': 30.0, # duration of leaf senescence (days),
                         },                                                   

            'state': {# LAI is annual maximum LAI and for gridded simulations are input from GisData!
                      # keys must be 'LAI_ + key in spec_para
                      'LAI_conif': 3.5,
                      'LAI_decid': 0.5,
                      'hc': 16.0, # canopy height (m)
                      'cf': 0.6, # canopy closure fraction (-)

                       #initial state of canopy storage [mm] and snow water equivalent [mm]
                       'w': 0.0, # canopy storage mm
                       'swe': 0.0, # snow water equivalent mm
                       }
            }
        
    # BUCKET
    pbu = {'depth': 0.4,  # root zone depth (m)
           # following soil properties are used if spatial_soil = False
           'poros': 0.43, # porosity (-)
           'fc': 0.33, # field capacity (-)
           'wp': 0.13,	 # wilting point (-)
           'ksat': 2.0e-6, 
           'beta': 4.7,
           #organic (moss) layer
           'org_depth': 0.04, # depth of organic top layer (m)
           'org_poros': 0.9, # porosity (-)
           'org_fc': 0.3, # field capacity (-)
           'org_rw': 0.24, # critical vol. moisture content (-) for decreasing phase in Ef
           'maxpond': 0.0, # max ponding allowed (m)
           #initial states: rootzone and toplayer soil saturation ratio [-] and pond storage [m]
           'rootzone_sat': 0.6, # root zone saturation ratio (-)
           'org_sat': 1.0, # organic top layer saturation ratio (-)
           'pond_sto': 0.0, # pond storage
           'soilcode': -1 # site-specific values
           }
    
    return pgen, pcpy, pbu
