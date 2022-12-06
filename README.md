# Generating-a-BH-Merging-Catalogue
**Paolo Marcoccia<sup>1</sup>**

<sub>1. University of Stavanger, Institutt for Matematikk og Fysikk, Kjølv Egelands hus, 5.etg, E-blokk, 4021 Stavanger, Norway </sub> 
## Introduction ##

The main purpose of this ipython notebook, is to generate a catalogue of _Stellar Origin Binary Black Hole Merging events_ (_SOBBHm_) inside a given volume of space and detectable time.
In order to do so, I'll use equation **1)** and **2)** of the paper of [A. Sesana](https://arxiv.org/abs/1602.06951), in the same form described in the paper by [LIGO and Virgo Scientific Collaboration [1]](https://arxiv.org/abs/2111.03634)[[2]](https://arxiv.org/abs/2010.14533)[[3]](https://arxiv.org/abs/1811.12940).
The previous articles infer the properties of the parameters distribution for the _SOBBHm_ population detected by the _LIGO and VIRGO Collaboration_, in order to predict a reliable description of the _Black Holes_ (_BHS_).
The code can also be used to generate subdominant channels of the main population described by the _LIGO-Virgo-Kagra_ (_LVK_) collaboration, as the _Primordial Black Holes_ (_PBHs_) population described in [B. Carr](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.023514), [V. De Luca](https://arxiv.org/abs/2106.13769).

## Description of the code ##

In the paper by [LIGO and Virgo Scientific Collaboration [1]](https://arxiv.org/abs/2111.03634)[[2]](https://arxiv.org/abs/2010.14533)[[3]](https://arxiv.org/abs/1811.12940), the inference was done adopting several different distributions for the masses.
In this notebook [notebook](https://github.com/KuZa91/Generating-a-BH-Merging-Catalogue/blob/master/BHCatalogV6.0.ipynb), we implemented both the _Power law + Peak model_ taken by [[1]](https://arxiv.org/abs/2111.03634), and the _Model B_ distribution taken from [[3]](https://arxiv.org/abs/1811.12940).
Furthermore, in the latest version _V 6.0_ of the notebook, in order to describe _PBH_ perturbations we implemented a standard _Gaussian_ mass function as well as a _Log-Normal Probability Distribution Function(PDF)_ taken from [B. Carr et al.](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.023514). 
For what concerns _spin amplitudes_ and _spin tilts angles_, we only implemented the _Default Spin model_ taken from the paper [[1]](https://arxiv.org/abs/2111.03634) as there is no evidence in literature for other relevant models, the parameters for the latters were described in the paper [[2]](https://arxiv.org/abs/2010.14533).
The merger rates implemented in the paper are as follows: 

- A constant merger rate as proposed in [LIGO and Virgo Scientific Collaboration [2]](https://arxiv.org/abs/2010.14533);
- A redshift evolving merger rate which follows a power law dependency in redshift as in [[1]](https://arxiv.org/abs/2111.03634);
- A redshift evolving merger rate in the form of [P. Madau](https://arxiv.org/abs/1403.0007);
- A broken power law for _PBH_ population as presented by [V. Atal](https://arxiv.org/abs/2201.12218);
- A merger rate in function of the universal time as proposed in [S.S. Bavera](2109.05836).

In order to generate perturbation catalogues at a given z, the mode _R Spike_ was implemented which will generate events only at a particular bin of z given a certain width and intensity for the perturbation.
Most of the other parameters were taken uniform in their prior, as no real information on their distribution could still be inferred from the observed data.
To conclude, when considering a population for the [LISA](https://www.elisascience.org/) detector equation **11 a)** of the paper by [S. Marsat and J. Baker](https://arxiv.org/abs/1806.10734) can be used both to calculate the _initial frequency_, and the time in which the event gets to a choosen maximum value  of frequency that may be set as the maximum sensitivity of a given detector, in order to obtain a first approximation for the _In-Band Time_.
The dataset would be generated in the Detector frame for all variables but the masses, which are kept in the source frame in order to directly compare with the expected mass distribution functions, they can however be converted in the detector frame by running the script [ToDetFrame.py](https://github.com/KuZa91/Generating-a-BH-Merging-Catalogue/blob/master/to_DetFrameV3.py). Analogously, the _Signal to noise Ratio_ (_SNR_) of the generated events can be computed, both for the [aLIGO](https://dcc.ligo.org/public/0149/T1800042/004/T1800042-v4.pdf) and [LISA](https://www.elisascience.org/) detector, using the script [AddAnalSNR.py](https://github.com/KuZa91/Generating-a-BH-Merging-Catalogue/blob/master/AddAnalSNRV2.py) 

## Analysis Details ##

The main equation that will be integrated during this simulation is equation **8)** of the paper by [LIGO and Virgo Scientific Collaboration](https://arxiv.org/abs/1811.12940), the merger rate and mass _PDF_ can be selected trough the flags at the start of the notebook, whereby the parameters for these function can be set on their function definition.
The equation, in order to be properly solved require to integrate over the followings parameter :

- The _redshift_, which may be considered as an analougous of the distance from the detector ;

- The _masses_ of the 2 events;

- The _spin amplitudes_ of the 2 events;

- The _spin tilts angles_ of the 2 events;

To increase the computational speed, the function <code>Bin_and_Gen</code> was parallelized over the first cycle, resulting in a performance increase up to the choosen _mass precision_ for a computer with a number of cores greater than this value.
Furthermore, the simmetry of the _spin amplitudes_ and _spin tilts angles_ distributions were used to implement an _Inverse cumulative distribution function(ICDF)_ for the latters and generate them a posteriori without having to span over their phase space.
For each of the considered bin, the function <code>Gen_Events_Parameters</code> generates estimate the number of events in the considered bin, this number can be interpreted in several different ways by activating the corrispective flag for the mode. We have the following modes available for the run :

- _Default mode_ : the number of events generated would be just the predicted number of events in each binrounded up (e.g 18.7 ~ 19, 0.4 ~0.), this mode can be used to understand the properties dominant events that will appear in the catalogue;

- _Mode Ex_ : allows the generation of exotic events with non-zero probability by summing a random number between 0 and 0.5 before rounding up as in _Default mode_;

- _Mode FastMc_ : The decimal part of the predicted number of events would be treated in a Monte Carlo approach by generating a random number between 0 and 1, only if the generated number will be smaller or equal than the decimal part the latter would be rounded up to 1. This mode works well when the number of generated events is not high enough to give multiple events on each considered bin (e.g. when using _LIGO_ observation time), in the other case can overestimate the number of generated events;

- _Mode Poisson_ : The predicted number of events would be used as the parameter $\lambda$ describing a Poissonian distribution and the effective number of generated events per bin would be obtained by reverse-sampling the latter. This is generally the most reliable approach to generate events. 

All the uniform parameters, that are not described by any _PDF_, are added to the final dataframe later by using vectorized operation in order to increase computational speed.
The partial list, moreover, can be saved to the final dataframe in multiple steps of a fixed percentage in order to avoid ram duplication problem caused by the function <code>pd.concat</code>. The percentage of dataframe copied on each step can be defined trough the _cp_perc_ variable, and is particularly usefull to set small when computing a _LISA_ observation catalogue.

To conclude, for what concerns the randomization procedure, the random seed can be inizialized to a choosen value during the simulation manually by turning on the _sel_rs_ flag in order to have reproducible results, at the same time the flag _Check_Plot_ can be turned on and off depending on wether or not the plots describing the generated population have to be computed or not.

## Additional Material ##

This notebook, doesn't require any additional material except the one already present in a standard _Anaconda_ installation.
The _Anaconda 3_ distribution may be downloaded for free at the following [link](https://www.anaconda.com/products/individual)

