# coding: utf-8
import numpy as np 
import scipy.special as sc 
import statistics as st 
import random 
import pandas as pd 
import sys
  

df_key = 'SOBBH'
df_nm = sys.argv[1]
tau_cut = sys.argv[2]

print('We start by loading the dataframe...')

BHCat = pd.read_hdf(df_nm, df_key)

print('We are now applying the cut on the coalescence time')

NewBHCat = BHCat[BHCat.CoalTime <= float(tau_cut)]

print('Let me save the dataframe and we are done !')

NewBHCat.to_hdf('TauCutted'+tau_cut+df_nm, df_key, mode='w')
