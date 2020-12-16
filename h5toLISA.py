# This script convert the first .h5 dataframe, passed in the command line, in a LISA type dataframe with name given by the first argument name starting with LISA

import numpy as np
import sys
import pandas as pd
from LISAhdf5 import LISAhdf5,ParsUnits


SOBBHsunits = {

'Redshift': 'Unit',\
    
'Mass1': 'SolarMass',\
    
'Mass2': 'SolarMass',\

'InitialFrequency' : 'Hertz',\
    
'InBandTime' : 'Years',\

'EclipticLongitude' : 'Radian',\
    
'EclipticLatitude' : 'Radian',\
    
'Inclination' : 'Radian',\

'Polarization' : 'Radian',\
    
'InitialPhase' : 'Radian',\
    
'CoalTime' : 'Years',\
    
'Distance' : 'GigaParsec',\

'Spin1' : 'Unit',\
    
'Spin2' : 'Unit',\
    
'AzimuthalAngleOfSpin1' : 'Radian',\
    
'AzimuthalAngleOfSpin2' : 'Radian'    

}

df_key = 'SOBBH'
df_nm = sys.argv[1]

BHCat = pd.read_hdf(df_nm, df_key)

LH = LISAhdf5('LISA'+df_nm)
pr = ParsUnits()

for p in list(SOBBHsunits.keys()):
    pr.addPar(p,BHCat[p],SOBBHsunits[p])

pr.addPar("SourceType",df_key, "name")    
LH.addSource('SOBBH',pr, overwrite=True)  