# -*- coding: utf-8 -*-
"""
Created on Sun May  8 14:49:24 2022

@author: peria
"""
#%% SET CORRECT WORKING DIRECTORY


import os
os.chdir(r'C:\Users\peria\Desktop\DATA SCIENCE\TIME SERIES\Data Science For Supply Chain Forecasting')


#%% # IMPORTAZIONE LIBRERIE GENERALI


import pandas as pd
pd.options.display.max_rows = 100
pd.options.display.max_columns = 5

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
sns.set_style('whitegrid')

from datetime import timedelta

import math

#import warnings
#warnings.filterwarnings('ignore')


#%% MOVING AVERAGE

#VERSIONE_1

def Moving_Average(d, extra_periods, n):
    
    #INIZIALIZZO
    k = len(d)
    d = np.append(d, [np.nan]*extra_periods) # domanda
    f = np.full(k + extra_periods, np.nan) # forecast
    
    #RIEMPIO
    for t in range(n, k+1): # n incluso - k+1 escluso
        f[t] = np.mean(d[t-n : t])
    
    f[t+1 : ] = f[t] # Riempio da k+1 in fondo
    
    #Creo df
    df = pd.DataFrame({'Demand': d,
                       'Forecast':f, 
                       'Error': d-f})
    
    return(df)


#------------------------------------------------------------------------------


#VERSIONE_2

def Moving_Average_variant(d, extra_periods, n):
    
    #INIZIALIZZO
    k = len(d)
    d = np.append(d, [np.nan]*extra_periods) # domanda
    f = np.full(k + extra_periods, np.nan) # forecast
    
    #RIEMPIO
    for t in range(n, k): # n incluso - k escluso
        f[t] = np.mean(d[t-n : t])
    
    f[t+1 : ] = np.mean(d[t+1-n : t+1]) # Riempio da k in fondo
    
    #Creo df
    df = pd.DataFrame({'Demand': d,
                       'Forecast':f, 
                       'Error': d-f})
    
    return(df)



#%% SIMPLE EXPONENTIAL SMOOTHING

# VERSIONE_1

def Simple_Exp_Smooth(d, extra_periods = 4, alpha = 0.3, init = 'simple'):
    
    
    #INIZIALIZZO
    k = len(d)
    d = np.append(d, [np.nan]*extra_periods) # domanda
    f = np.full(k + extra_periods, np.nan) # forecast
    
    
    # SCELTA MODALITA' di INIZIALIZZAZIONE
    if (init == 'simple'):
        f[0] = d[0]
    elif(init == 'avg'):
        n = math.floor(1/alpha)
        f[0] = d[0:n].mean()
    else:
        print('Scegli una modalità di inizializzazione consona')
        return(None)
    
    
    # RIEMPIO
    for t in range(1, k+1):
        f[t] = alpha*d[t-1] + (1-alpha)*f[t-1]
    
    f[t+1:] = f[t] # Riempio da k+1 in fondo. Alla fine del for t vale k
    
    
    #Creo df
    df = pd.DataFrame({'Demand': d,
                       'Forecast':f, 
                       'Error': d-f})
    
    
    return(df)


#------------------------------------------------------------------------------

#VARIANTE

def Simple_Exp_Smooth_Variant(d, extra_periods, alpha):
    
    #INIZIALIZZO
    k = len(d)
    d = np.append(d, [np.nan]*extra_periods) # domanda
    f = np.full(k + extra_periods, np.nan) # forecast
    
    #Inizializzo la prima forecast
    f[1] = d[0]
    
    for t in range(2, k+1): # n incluso - k escluso
        f[t] = alpha*d[t-1] + (1-alpha)*f[t-1]
    
    for t in range(k+1, k+extra_periods):
        f[t] = f[t-1]
    
    #Creo df
    df = pd.DataFrame({'Demand': d,
                       'Forecast':f, 
                       'Error': d-f})
    
    return(df)


#%%





