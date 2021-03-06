# Generating-a-BH-Merging-Catalogue
**Paolo Marcoccia<sup>1</sup>**

<sub>1. University of Stavanger, Institutt for Matematikk og Fysikk, Kjølv Egelands hus, 5.etg, E-blokk, 4021 Stavanger, Norway </sub> 
## Introduction ##

The main purpose of this ipython notebook, is to generate a catalogue of _Stellar Origin Binary Black Hole Merging events_ (_SOBBHm_) inside a given volume of space and detectable time.
In order to do so, I'll use equation **1)** and **2)** of the paper of [A. Sesana](https://arxiv.org/abs/1602.06951), in the same form described in the paper by [LIGO and Virgo Scientific Collaboration](https://arxiv.org/abs/1811.12940).
The previous article infer the properties of the parameters distribution for the _SOBBHm_ population detected by the _LIGO and VIRGO Collaboration_, in order to predict a reliable distribution for the future detected events, and in particular, the inference was done adopting 3 different distribution for the mass.
I decided to adopt the _Model B_ distribution given by equation **2)** for the mass in this notebook, as it allows the distribution to consider the effects of proportionality among the 2 masses without getting as complicated as the _Model C_.
For what concerns _spin amplitudes_ and _spin tilts angles_, equations **4)** and **6)** were used respectively, while all the others parameters generated will be assumed uniform in their value range.
To conclude, equation **11 a)** of the paper by [S. Marsat and J. Baker](https://arxiv.org/abs/1806.10734) was used both to esteem the _initial frequency_ and the time in which the merging event gets to a maximum frequency that may be assumed as the maximum sensitivity of a given detector.
The minimum and maximum sensitivity of said detector (that in this notebook will be setted by default to the [LISA](https://www.elisascience.org/) sensitivity values), will be used to know the time in years that each merging events will spend in the detector band, as well as to kick out of the dataframe all the events with _initial frequency_ outside the detector band.
The _initial frequency_ value, however, even though depends partially from the masses of the considered event, will also depend on the _residual time to merging_ of the event, which in this simulation will be given by a uniform distribution

## Analysis Details ##

The main equation that will be integrated during this simulation is equation **8)** of the paper by [LIGO and Virgo Scientific Collaboration](https://arxiv.org/abs/1811.12940), where all the parameters for the distribution functions were taken from the same paper and are reported together with their uncertainty in <code>In[16]</code>
This equation however in order to be properly solved require to span over lot of parameters, which are :

- The _redshift_, which may be considered as an analougous of the distance from the detector ;

- The _masses_ of the 2 events;

- The _spin amplitudes_ of the 2 events;

- The _spin tilts angles_ of the 2 events;

To increase the computational speed, the function <code>Bin_and_Gen</code> was parallelized over the first cycle, resulting in a performance increase up to the setted _mass precision_ for a computer with said number of cores.
Furthermore, the simmetry of the _spin amplitudes_ and _spin tilts angles_ distributions were used to span over only half of the elements that needed to be considered.
This may be done as long as the number of generated events outside the diagonal are multiplied by 2, and a parameter flip among the 2 values is introduced.
For each of the considered bin, the function <code>Gen_Events_Parameters</code> generated a number of events equal to the number predicted in the considered bin, and saved that (if inside the detector frequency sensitivity) in a partial list where all the non-uniform parameters were saved.
All the missing parameters, were added to the final dataframe later by using vectorized operation in order to increase computational speed.
The partial list, moreover, by being quite heavy could be saved to the final dataframe in multiple steps of a fixed percentage in order to avoid ram duplication problem caused by the function <code>pd.concat</code>.
Three different modalities were implemented to test out the notebook :

- The _standard mode_ will only saves certain events in each bin, and will be usefull to understimate the real number of events ;

- The _fast montecarlo mode_ will generate in each bin the certain number, and will flip a random number between 0 and 1 for the residual value of the bin, if the random number will be smaller of equal than the residual value of the bin, a new event will be added to the dataframe (this is the most reliable mode) ;

- The _mode ex_ will randomly add a value between 0 and 0.49 to each of the considered bin, if the bin residual value will be bigger than 0.5 then an events would be generated, this allows for low probability bins to appear easy on the dataframe and tend to overstimate the final result !

The random seed during the simulation may be choosen in order to have reproducible results, furthermore at the end of the notebook many plots will be shown for observing the properties of the generated dataframe !

## Additional Material ##

This notebook, doesn't require any additional material except the one already present in a standard _Anaconda_ installation.
The _Anaconda 3_ distribution may be downloaded for free at the following [link](https://www.anaconda.com/products/individual)

