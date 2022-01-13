# Generating-a-BH-Merging-Catalogue
**Paolo Marcoccia<sup>1</sup>**

<sub>1. University of Stavanger, Institutt for Matematikk og Fysikk, Kjølv Egelands hus, 5.etg, E-blokk, 4021 Stavanger, Norway </sub> 
## Introduction ##

The main purpose of this ipython notebook, is to generate a catalogue of _Stellar Origin Binary Black Hole Merging events_ (_SOBBHm_) inside a given volume of space and detectable time.
In order to do so, I'll use equation **1)** and **2)** of the paper of [A. Sesana](https://arxiv.org/abs/1602.06951), in the same form described in the paper by [LIGO and Virgo Scientific Collaboration [1]](https://arxiv.org/abs/2111.03634)[[2]](https://arxiv.org/abs/2010.14533)[[3]](https://arxiv.org/abs/1811.12940).
The previous articles infer the properties of the parameters distribution for the _SOBBHm_ population detected by the _LIGO and VIRGO Collaboration_, in order to predict a reliable distribution for the future detected events, and in particular, the inference was done adopting several different distributions for the masses.
In the [notebook](https://github.com/KuZa91/Generating-a-BH-Merging-Catalogue/blob/master/BHCatalogV6.0.ipynb), we implemented both the _Power law + Peak model_ taken by [[1]](https://arxiv.org/abs/2111.03634), and the _Model B_ distribution taken from [[3]](https://arxiv.org/abs/1811.12940).
Furthermore, in the latest version _V 6.0_ of the notebook, in order to describe _Primordial Black Hole(PBH)_ perturbations, we implemented a standard _Gaussian_ mass function as well as a _Log-Normal Probability Distribution Function(PDF)_ taken from [B. Carr et al.](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.023514). 
For what concerns _spin amplitudes_ and _spin tilts angles_, we implemented the _Default Spin model_ taken from the paper [[1]](https://arxiv.org/abs/2111.03634), the parameters for the latters were described in the paper [[2]](https://arxiv.org/abs/2010.14533).
The merging rate is taken in agreement with the latest results of [[1]](https://arxiv.org/abs/2111.03634), in particular, it was implemented both a redshift evolving merging rate and a constant one over the volume.
In order to generate perturbation catalogues at a given z, the mode _R Spike_ was implemented which will generate events only at a particular bin of z given a certain width and intensity for the perturbation.
Most of the other parameters were taken uniform in their prior, as no real information on their distribution could still be inferred from the observational data.
To conclude, equation **11 a)** of the paper by [S. Marsat and J. Baker](https://arxiv.org/abs/1806.10734) was used both to esteem the _initial frequency_ and the time in which the merging event gets to a maximum frequency that may be assumed as the maximum sensitivity of a given detector.
The minimum and maximum sensitivity of said detector (that in this notebook will be setted by default to the [LISA](https://www.elisascience.org/) sensitivity values), will be used to know the time in years that each merging events will spend in the detector band, as well as to kick out of the dataframe all the events with _initial frequency_ outside the detector band.
The _initial frequency_ value, however, even though depends partially from the masses of the considered event, will also depend on the _residual time to merging_ of the event, which in this simulation will be given by a uniform distribution up to a maximum cutoff value.
The dataset would be generated in the Detector frame for all variables but the masses, which are kept in the source frame in order to directly compare with the expected mass distribution functions, they can however be converted in the detector frame by running the [script](https://github.com/KuZa91/Generating-a-BH-Merging-Catalogue/blob/master/to_DetFrameV3.py).

## Analysis Details ##

The main equation that will be integrated during this simulation is equation **8)** of the paper by [LIGO and Virgo Scientific Collaboration](https://arxiv.org/abs/1811.12940), where all the parameters for the distribution functions were taken from the previously cited papers and are reported together with their uncertainties in <code>In[11]-In[27]</code>
This equation, however, in order to be properly solved require to span over some parameters, which are :

- The _redshift_, which may be considered as an analougous of the distance from the detector ;

- The _masses_ of the 2 events;

- The _spin amplitudes_ of the 2 events;

- The _spin tilts angles_ of the 2 events;

To increase the computational speed, the function <code>Bin_and_Gen</code> was parallelized over the first cycle, resulting in a performance increase up to the setted _mass precision_ for a computer with said number of cores.
Furthermore, the simmetry of the _spin amplitudes_ and _spin tilts angles_ distributions were used to implement an _Inverse cumulative distribution function(ICDF)_ for the latters and generate them a posteriori without having to span over the phase space.
For each of the considered bin, the function <code>Gen_Events_Parameters</code> generated a number of events equal to the number predicted in the considered bin, this number can be interpreted in several different ways by activating the corrispective flag for the mode. We have the following modes available for the run :

- _Default mode_ : the number of events generated would be just the predicted number of events rounded up (e.g 18.7 ~ 19, 0.4 ~0.);

- _Mode Ex_ : allows the generation of exotic events with non-zero probability by summing a random number between 0 and 0.5 before rounding up as in _Default mode_;

- _Mode FastMc_ : The decimal part of the predicted number of events would be treated in a Monte Carlo approach by generating a random number between 0 and 1, only if the generated number will be smaller or equal than the decimal part the latter would be rounded up to 1;

- _Mode Poisson_ : The predicted number of events would be used as the parameter describing a Poissonian distribution and the effective number of generated events per bin would be obtained by reverse-sampling the latter. 

All the uniform parameters, that are not described by any _PDF_, are added to the final dataframe later by using vectorized operation in order to increase computational speed.
The partial list, moreover, can be saved to the final dataframe in multiple steps of a fixed percentage in order to avoid ram duplication problem caused by the function <code>pd.concat</code>.

Initialization of the random seed during the simulation may be choosen manually by turning on the _sel_rs_ flag in order to have reproducible results, at the same time the flag _Check_Plot_ can be turned on and off depending on wether or not the plots describing the generated population have to be computed or not.

## Additional Material ##

This notebook, doesn't require any additional material except the one already present in a standard _Anaconda_ installation.
The _Anaconda 3_ distribution may be downloaded for free at the following [link](https://www.anaconda.com/products/individual)

