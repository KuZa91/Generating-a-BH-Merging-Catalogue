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


c = 299792.46 # speed of light in Km/sec
G = 6.674*(10.**(-11.)) # Gravitational constant in m^3⋅kg^−1⋅s^−2
sol_mass = 1.988e30 # Value of the Solar Mass in Kg
MPc = 3.086e+22 # MPc to m conversion factor
H_0 = 67.8 # Hubble constant in Km/(s*MPc)
year = 365.25*24*60*60 # Years in second 
f_min = 1.e-4
f_max = 0.5
T_obs = 4.
arm_lenght = 2.5e9 # value of the arms lenght 
A = 2 / np.pi**(2 / 3) * np.sqrt(5 / 96) # Constant factor needed to compute anal SNR
span_prec = 50000


def ChirpMass(m1,m2): 
   return ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))

def GetInitialFrequency(m1,m2,coal_T):
    M = m1 + m2
    ni = (m1*m2)/(M*M)
    res = ((256.*ni)/(5.*np.power((c*(10.**3.)),5.)))*np.power((G*M*sol_mass),(5./3.))*coal_T
    return (np.power(res,(-(3./8.)))/np.pi)

def S_oms(freq):
    omega = (2.*np.pi*freq)/(c*1000)
    res = (15*1e-12)*omega*np.sqrt(1. + ((2.*1.e-3)/freq)**4.)
    return res**2.

def S_acc(freq):
    res = ((3.*1.e-15)/(2.*np.pi*freq*c*1000))*np.sqrt(1. + ((0.4*1.e-3)/(freq))**2.)\
    *np.sqrt(1. + (freq/(8.*1.e-3))**4.)
    return res**2.

def Snx1p5(freq, arm_lenght):
    omega = (2.*np.pi*freq)/(c*1000)
    res = 16.*(np.sin(omega*arm_lenght)**2.)*(S_oms(freq) + \
          (3. + np.cos(2.*omega*arm_lenght))*S_acc(freq)) 
    return res

def S_Hx(freq, arm_lenght):
    omega = (2.*np.pi*freq)/(c*1000)
    return (20./3.)*(1. + 0.6*(omega*arm_lenght)**2.)*\
    ((Snx1p5(freq, arm_lenght))/((np.sin(omega*arm_lenght)**2.)*(4.*omega*arm_lenght)**2))


df_key = 'SOBBH'
df_nm = sys.argv[1]

print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')
print('We start by loading the dataframe...')

BHCat = pd.read_hdf(df_nm, df_key)
print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~')
print('First of all we have to compute the frequency at the end of the observation time for the various events')

BHCat['fend'] = GetInitialFrequency(BHCat.m1, BHCat.m2, (BHCat.Coal_t - T_obs)*year)
BHCat[BHCat.fend > f_max] = f_max
BHCat['ChM'] = ChirpMass(BHCat.m1,BHCat.m2)

print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')

print('We will now estimate the cumulative distribution for the integral factor of the SNR estimator')

ran_frq = np.linspace(f_min,f_max, span_prec)
ran_cmi = np.linspace(f_min,f_max, span_prec - 1)*0.
ran_mfrq = np.linspace(f_min,f_max, span_prec - 1)*0.

for i in tqdm(range(len(ran_frq) - 1)):

    ran_mfrq[i] = 0.5*(ran_frq[i] + ran_frq[i + 1])
    
    if (i == 0):
        ran_cmi[i] = (ran_frq[i + 1] - ran_frq[i])*\
        ((ran_mfrq[i]**(-7./3.))/(S_Hx(ran_mfrq[i], arm_lenght)))
    else:
        ran_cmi[i] = (ran_frq[i + 1] - ran_frq[i])*\
        ((ran_mfrq[i]**(-7./3.))/(S_Hx(ran_mfrq[i], arm_lenght)))\
        + ran_cmi[i - 1]

print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')

print('Given the cumulative distribution, we can just define an interpolator to fastly compute that for any range')

IntFac = interp1d(ran_mfrq, ran_cmi)

print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')

print('We can finally estimate the analytical SNR for the events of the catalogue')

BHCat['AnalSNR'] = np.sqrt(16.*((A*((BHCat.ChM*sol_mass*G)**(5./6.))/(BHCat.dist*(1. + BHCat.z)*MPc))**2.)*\
    (1./(c*1000)**3.)*(IntFac(BHCat.fend) - IntFac(BHCat.fstart)))

print('-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~~-~-~-~-~-~-~-~-~-~-~-~')

print('Let me save the dataframe and we are done !')

BHCat.to_hdf('An'+df_nm, df_key, mode='w')
