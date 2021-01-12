# This script merge the dataframe for a catalogue given by the first argument of the script call with the 4th row of the SNR txt given by the second argument of the call

import numpy as np
import sys
import pandas as pd

def ChirpMass(m1,m2):  
    return ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))   



df_key = 'SOBBH'
df_nm = sys.argv[1]

SNR_nm = sys.argv[2]

BHCat = pd.read_hdf(df_nm, df_key)

BHCat['ChirpMass'] = ChirpMass(BHCat.Mass1, BHCat.Mass2) 

SNR = pd.read_csv(SNR_nm, sep = ' ', header = None, names = ['Nid', 'SNR_A', 'SNR_B', 'SNR'])

SNR = SNR.set_index('Nid')

BHCat = pd.concat([BHCat,SNR.SNR], axis = 1)
BHCat.SNR = BHCat.SNR.fillna(0)

BHCat.to_hdf('SNR'+df_nm, df_key, mode='w')