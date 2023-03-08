import numpy as np 
import scipy.special as sc 
import statistics as st 
import random 
import pandas as pd 
import sys
#from LISAhdf5 import LISAhdf5,ParsUnits 
#%matplotlib inline 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy import interpolate
from scipy.interpolate import interp1d
plt.style.use('seaborn-whitegrid') 

# Parameter defintion

df_nm = sys.argv[1]     # first parameter to be passed on calling, name of the .h5 catalogue file
df_key = sys.argv[2]    # second parameter to be passed on calling, key of the .h5 catalogue file

# Numerical simulation settings

span_prec = 50000

# Spin tilt PDF parameters

norm_const = 0.
sigma_1 = 1.5 # + 2.0 - 0.8
sigma_2 = 1.5 # + 2.0 - 0.8
zeta = 0.66 # + 0.31 - 0.52

# Estimating the ICDF for a Gaussian spin tilt distribution

def PDF_Gauss(t, sigma, norm_const):
    return norm_const*np.exp(-((1 - t)**2)/(2.*sigma**2))


ran_cost = np.linspace(-1., 1., span_prec)
int_range = 0.5*(ran_cost[1::] + ran_cost[:-1:])

print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')
print('We start by estimating the normalization constant for the gaussian spin distribution')

for i in tqdm(range(len(int_range))):
    norm_const += (ran_cost[i + 1] - ran_cost[i])*PDF_Gauss(int_range[i], sigma_1, 1.)

print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')
print('We can now proceed by estimating the cumulative distribution function for the gaussian spin distribution')

Gauss_CumDist = 0.*int_range

for i in tqdm(range(len(int_range))):
    if(i == 0):
        Gauss_CumDist[i] += (ran_cost[i + 1] - ran_cost[i])*PDF_Gauss(int_range[i], sigma_1, (1./norm_const))
    else:
        Gauss_CumDist[i] = Gauss_CumDist[i -1] + (ran_cost[i + 1] - ran_cost[i])*PDF_Gauss(int_range[i], sigma_1, (1./norm_const))

# To 0-1 range
Gauss_CumDist = (Gauss_CumDist - min(Gauss_CumDist)) / (max(Gauss_CumDist) - min(Gauss_CumDist))
# Invert
spl_inv_spintilt = interpolate.splrep(Gauss_CumDist, int_range)

def spintilt_sample(N):
    return interpolate.splev(np.random.random(N), spl_inv_spintilt)

# The function to correct the spins can hence be defined as

def SpinTiltFunct(zeta): 
    x = random.uniform(0.,1.)
    if(x >= zeta):
        cos_t1, cos_t2 = random.uniform(-1., 1.), random.uniform(-1., 1.)
    else:
        cos_t1, cos_t2 = spintilt_sample(1), spintilt_sample(1)

    return np.arccos(cos_t1), np.arccos(cos_t2)




print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')
print('We can now load the dataframe...')

BHCat = pd.read_hdf(df_nm, df_key)


print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')


print('And correct the spin tilts !')

for i in tqdm(range(len(BHCat.AzimuthalAngleOfSpin1))):
    BHCat['AzimuthalAngleOfSpin1'][i], BHCat['AzimuthalAngleOfSpin2'][i] = SpinTiltFunct(zeta)

print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')

print('Let me save the dataframe and we are done !')

BHCat.to_hdf('CorrST'+df_nm, df_key, mode='w')

