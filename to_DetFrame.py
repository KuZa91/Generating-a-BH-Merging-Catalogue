import numpy as np 
import scipy.special as sc 
import statistics as st 
import random 
import pandas as pd 
import sys
#from LISAhdf5 import LISAhdf5,ParsUnits 
#%matplotlib inline 
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid') 

df_key = 'SOBBH'
df_nm = sys.argv[1]


BHCat = pd.read_hdf(df_nm, df_key)

BHCat['Mass1'] = BHCat['Mass1']*(1. + BHCat.Redshift)

BHCat['Mass2'] = BHCat['Mass2']*(1. + BHCat.Redshift)

BHCat['CoalTime'] = BHCat['CoalTime']*(1. + BHCat.Redshift)

BHCat.to_hdf('DetFrame'+df_nm, df_key, mode='w')
