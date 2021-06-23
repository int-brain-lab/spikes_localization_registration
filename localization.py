import numpy as np
from tqdm.notebook import tqdm
import numpy as np
import torch
import pickle
from tqdm import tqdm
import os
import shutil
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gridspec
import scipy.optimize as optim
import matplotlib.cm as cm
from functools import reduce
import yass

from yass import read_config, set_config
from yass.reader import READER
from yass.config import Config

from yass.noise import get_noise_covariance

import scipy
from scipy.stats import chi2
import cycler

from yass.neuralnetwork import Denoise ### Can be replaced - see YASS repo : yass/src/yass/neuralnetwork/model_denoiser.py

import os
import sys
from collections import Counter

from sklearn.decomposition import PCA

### Set CONFIG to read waveforms + load NN - can be replaced not to use YASS environment
set_config('drift.yaml', 'tmp/')
CONFIG = read_config()
TMP_FOLDER = CONFIG.path_to_output_directory


### Load spike train + templates -> Can be replaced if integrated to detection + deduplication

spike_train = np.load("spike_train.npy") 
templates = np.load('templates.npy')

standardized_path = 'standardized.bin' ## Needs the data to be standardized
standardized_dtype = 'float32'
reader= READER(standardized_path, 'float32', CONFIG, n_sec_chunk = 1)

##### Load Denoiser 

def denoise_wf_nn_tmp(wf, denoiser, device):
    denoiser = denoiser.to(device)
    n_data, n_times, n_chans = wf.shape
    if wf.shape[0]>0:
        wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
        wf_torch = torch.FloatTensor(wf_reshaped).to(device)
        denoised_wf = denoiser(wf_torch)[0].data
        denoised_wf = denoised_wf.reshape(
            n_data, n_chans, n_times)
        denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)

        del wf_torch
    else:
        denoised_wf = np.zeros((wf.shape[0], wf.shape[1]*wf.shape[2]),'float32')

    return denoised_wf


### Uses YASS env -> Update to load the denoiser without yass
denoiser = Denoise(CONFIG.neuralnetwork.denoise.n_filters,
                   CONFIG.neuralnetwork.denoise.filter_sizes,
                   CONFIG.spike_size_nn, CONFIG)
denoiser.load(CONFIG.neuralnetwork.denoise.filename)
denoiser = denoiser.cuda()

### Example NN takes minimum (or abs value maximum) at time step 41 
### Make sure Spike train uses similar alignment (+/- 1), otherwise shift spike train to get correct alignemnt before inputing spike train into denoiser


### Function we optimize
def minimize_ls(vec, wfs_0, CONFIG, z_initial, channels):
    return wfs_0.ptp(1)-vec[3]/(((CONFIG.geom[channels] - [vec[0], z_initial+vec[1]])**2).sum(1) + vec[2]**2)**0.5 # vec[0]


def get_estimate(batch_id, reader, spike_train, sampling_rate, threshold = 6, output_directory = 'position_results_files'):
    '''
    get_estimate() takes as input spike train, reads waveforms, denoises, select channels and perform localization optimization
    Saves .npy features for waveforms of each batch. arrays entries are 0 if ptp < threshold
    '''
    spike_times_batch = spike_train[np.logical_and(spike_train[:, 0] >= batch_id*fs, spike_train[:, 0] < (batch_id+1)*fs), 0]
    spike_units_batch = spike_train[np.logical_and(spike_train[:, 0] >= batch_id*fs, spike_train[:, 0] < (batch_id+1)*fs), 1]

    wfs, skipped_idx = reader.read_waveforms(spike_times_batch) ### USES YASS ENV
        
    results_x = np.zeros(wfs.shape[0])
    results_y = np.zeros(wfs.shape[0])
    results_z = np.zeros(wfs.shape[0])
    results_alpha = np.zeros(wfs.shape[0])
    results_max_ptp = np.zeros(wfs.shape[0])

    # These two features are useful for clustering, not localization
    results_spread = np.zeros(wfs.shape[0])
    time_width = np.zeros(wfs.shape[0])

    for i in (range(wfs.shape[0])):
        unit = spike_units_batch[i]
        

        ### This step allows to select a first batch of at least 20 channels for "gross localization" 
        ### Equivalent to removing far away channels. Can be done more efficiently if integrated in deduplication
        mc_chan = templates[unit].ptp(0).argmax()
        channels = np.where(templates[unit].ptp(0) >= 2)[0]
        channels = np.arange(channels.min() - (channels.min()%2), channels.max() + 2 - (channels.max()%2))
        if channels.shape[0] < 20:
            channels = np.arange(mc_chan - 10, mc_chan + 10)

        ### Denoise on first batch of channels before selecting 10 channels for optimization
        ### If deduplication gives a first depth estimate, can select surrounding 10 channels instead
        wfs_0 = wfs[i, :, channels].T
        wfs_0 = denoise_wf_nn_tmp(wfs_0.reshape((1, wfs_0.shape[0], wfs_0.shape[1])), denoiser, device)[0]
        mc = wfs_0.ptp(0).argmax()
        if wfs_0.ptp(0).max() > threshold:
            time_width[i] = np.abs(wfs_0[:, mc].argmax() - wfs_0[:, mc].argmin())
            if mc <= 4: 
                channels_wfs = np.arange(0, 10)
            elif  mc >= channels.shape[0]-5:
                channels_wfs = np.arange(channels.shape[0]-10, channels.shape[0])
            elif mc % 2 == 0:
                channels_wfs = np.arange(mc - 4, mc + 6)
            else:
                channels_wfs = np.arange(mc - 5, mc + 5)


            ##### Run optimization
            z_init = (wfs_0.ptp(0)[channels_wfs]*CONFIG.geom[channels[channels_wfs], 1]).sum()/wfs_0.ptp(0)[channels_wfs].sum()
            x_init = (wfs_0.ptp(0)[channels_wfs]*CONFIG.geom[channels[channels_wfs], 0]).sum()/wfs_0.ptp(0)[channels_wfs].sum()

            results_max_ptp[i] = wfs_0.ptp(0).max()

            output = optim.least_squares(minimize_ls, x0=[x_init, 0, 16, 1000], bounds = ([-200, -200, 0, 0], [232, 200, 250, 10000]), args=(wfs_0[:, channels_wfs].T, CONFIG, results_z_mean[i], channels[channels_wfs]))['x']

            results_x[i] = output[0]
            results_z[i] = CONFIG.geom[channels[mc], 1] + output[1] 
            results_alpha[i] = output[3]
            results_y[i] = np.abs(output[2]) #max(25, (output[2]/wfs_0.ptp(0)[channels_wfs].max() - ((CONFIG.geom[channels[mc]] - [output[0] , CONFIG.geom[channels[mc], 1] + output[1]])**2).sum()).mean())
            results_spread[i] = (wfs_0.ptp(0)[channels_wfs]*((CONFIG.geom[channels[channels_wfs]] - [results_x[i], results_z[i]])**2).sum(1)).sum()/wfs_0.ptp(0)[channels_wfs].sum()


    fname_time_width = os.path.join(output_directory, 'results_width_{}.npy'.format(str(batch_id).zfill(6)))
    np.save(fname_time_width, time_width)

    fname_disp = os.path.join(output_directory, 'results_disp_{}.npy'.format(str(batch_id).zfill(6)))
    fname_z = os.path.join(output_directory, 'results_z_{}.npy'.format(str(batch_id).zfill(6)))     
    fname_x = os.path.join(output_directory, 'results_x_{}.npy'.format(str(batch_id).zfill(6)))
    fname_z_mean = os.path.join(output_directory, 'results_z_mean_{}.npy'.format(str(batch_id).zfill(6)))     
    fname_x_mean = os.path.join(output_directory, 'results_x_mean_{}.npy'.format(str(batch_id).zfill(6)))
    fname_spread = os.path.join(output_directory, 'results_spread_{}.npy'.format(str(batch_id).zfill(6)))
    fname_max_ptp = os.path.join(output_directory, 'results_max_ptp_{}.npy'.format(str(batch_id).zfill(6)))
    fname_y = os.path.join(output_directory, 'results_y_{}.npy'.format(str(batch_id).zfill(6)))
    fname_alpha = os.path.join(output_directory, 'results_alpha_{}.npy'.format(str(batch_id).zfill(6)))
    fname_max_channels = os.path.join(output_directory, 'results_max_channels_{}.npy'.format(str(batch_id).zfill(6)))


    np.save(fname_z, results_z)
    np.save(fname_x, results_x)
    np.save(fname_z_mean, results_z_mean)
    np.save(fname_x_mean, results_x_mean)

    np.save(fname_max_channels, max_channels)
    np.save(fname_disp, disp)
    np.save(fname_max_ptp, results_max_ptp)
    np.save(fname_spread, results_spread)
    np.save(fname_alpha, results_alpha)
    np.save(fname_y, results_y)


### Parallelized on CPU

output_directory = 'registration_position_files'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
        
sampling_rate = 30000
threshold = 6
device = "cpu"

parmap.map(
        get_estimate,
        [i for i in range(1000)], 
        reader,
        reader_registered, spike_train, displacement,
        sampling_rate,
        threshold,
        output_directory,
        processes=5,
        pm_pbar=True)


##### Functions to merge arrays 

def get_total_len(output_directory):
    len_total = 0
    len_batch = np.zeros(1000)
    for batch_id in tqdm(range(1000)):
        fname_z = os.path.join(output_directory, 'results_z_mean_{}.npy'.format(str(batch_id).zfill(6)))
        len_batch[batch_id] = np.load(fname_z).shape[0]
        len_total += len_batch[batch_id]
    return int(len_total), len_batch.astype('int')


def get_merged_arrays(output_directory):
    len_total, len_batch = get_total_len(output_directory)
    
    merged_z_array = np.zeros(len_total)
    merged_x_array = np.zeros(len_total)
    merged_max_ptp_array = np.zeros(len_total)
    merged_alpha_array = np.zeros(len_total)
    merged_spread_array = np.zeros(len_total)
    merged_y_array = np.zeros(len_total)
    
    cmp = 0
    for batch_id in tqdm(range(1000)): #tqdm(range(1000)):

        fname_z = os.path.join(output_directory, 'results_z_{}.npy'.format(str(batch_id).zfill(6)))     
        fname_x = os.path.join(output_directory, 'results_x_{}.npy'.format(str(batch_id).zfill(6)))
        fname_spread = os.path.join(output_directory, 'results_spread_{}.npy'.format(str(batch_id).zfill(6)))
        fname_max_ptp = os.path.join(output_directory, 'results_max_ptp_{}.npy'.format(str(batch_id).zfill(6)))
        fname_alpha = os.path.join(output_directory, 'results_alpha_{}.npy'.format(str(batch_id).zfill(6)))
        fname_y = os.path.join(output_directory, 'results_y_{}.npy'.format(str(batch_id).zfill(6)))

        merged_z_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_z)
        merged_x_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_x)
        merged_max_ptp_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_max_ptp)
        merged_spread_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_spread)
        merged_alpha_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_alpha)
        merged_y_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_y_squared)

        cmp+=len_batch[batch_id] 
        
    print(cmp)  
    fname_z_merged = os.path.join(output_directory, 'results_z_merged.npy')
    fname_x_merged = os.path.join(output_directory, 'results_x_merged.npy')
    
    fname_alpha_merged = os.path.join(output_directory, 'results_alpha_merged.npy')
    fname_y_merged = os.path.join(output_directory, 'results_y_merged.npy')
    
    fname_max_ptp_merged = os.path.join(output_directory, 'results_max_ptp_merged.npy')
    fname_spread_merged = os.path.join(output_directory, 'results_spread_merged.npy')

    np.save(fname_z_merged, merged_z_array)
    np.save(fname_x_merged, merged_x_array)
    np.save(fname_alpha_merged, merged_alpha_array)
    np.save(fname_y_merged, merged_y_squared_array)
    np.save(fname_max_ptp_merged, merged_max_ptp_array)
    np.save(fname_spread_merged, merged_spread_array)


#### Run to get final position arrays 
output_directory = 'registration_position_files'
get_merged_arrays(output_directory)


output_directory = 'LS_registration_position_files_parmap'

fname_z = os.path.join(output_directory, 'results_z_merged.npy')     
fname_x = os.path.join(output_directory, 'results_x_merged.npy')
fname_y = os.path.join(output_directory, 'results_y_merged.npy')
fname_alpha = os.path.join(output_directory, 'results_alpha_merged.npy')
fname_max_ptp = os.path.join(output_directory, 'results_max_ptp_merged.npy')


alpha_ls = np.load(fname_alpha)
x_ls = np.load(fname_x)
z_ls = np.load(fname_z)
y_ls = np.load(fname_y)
max_ptp_array = np.load(fname_max_ptp)

ptp_array = max_ptp_array[z_ls != 0]
spike_times = spike_train[z_ls != 0, 0]
y_ls = y_ls[z_ls != 0]
alpha_ls = alpha_ls[alpha_ls != 0]
x_ls = x_ls[x_ls != 0]
z_ls = z_ls[z_ls != 0]


