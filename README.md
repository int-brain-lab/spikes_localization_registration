# Localizing NP detected spikes and registering raw data

## This repository provides code for localizing the spikes detected in Neuropixels recordings, estimating motion and registering raw data, and tools for visualizing and evaluating the output of any spike sorter.

### Localization works as follow : 
 - It takes as input the detected spikes and filtered + standardized data
 - Spikes are read from the data and denoised by a Neural Network denoiser. Pre-trained weights are available on github. 
 The Pre-trained NN Denoiser model expects the spikes to be of temporal length 121 and aligned so that their minimum is reached at timestep 42. 
 - Spikes are then localized for each batch of data (for example each second of data), obtained positions and features are stored into the desired repository before being merged to give final results. 
 
Localization code is designed to be fully self-contained. Code to read data and denoise data is written following the YASS pipeline (YASS: Yet Another Spike Sorter applied to large-scale multi-electrode array recordings in primate retina, Lee et al., 2020) and github repository (https://github.com/paninski-lab/yass)
Instructions to train and obtain a new NN-Denoiser can be found on YASS github repository. 


### Motion estimate and registration work as follow : 



### Visualisation : 

The repository contain a script that provides a 3d interactive visualization of the spike train and its localization features. 
Code requires Datoviz (https://github.com/datoviz/datoviz.git, see documentation here https://datoviz.org) 


