import os
import numpy as np
from tqdm import tqdm
from residual import RESIDUAL
from localizer import LOCALIZER
from denoiser import Denoise
from merge_results import get_merged_arrays

bin_file = '/media/cat/julien/nick_drift/different_preprocessings/data_standardized_registered/standardized_unregistered.bin'
residual_file = '/media/cat/julien/test_final_loc_code/residuals/residual.bin'
dtype_input = 'float32'
fname_spike_train = "/media/cat/julien/nick_drift/spt_yass.npy"
fname_templates = "/media/cat/julien/nick_drift/templates_yass.npy"
geom_path = "/media/cat/julien/nick_drift/geom.npy"
denoiser_weights = '/media/cat/julien/nn_np2/denoise.pt'
denoiser_min = 42 ## Goes with the weights
n_batches = 1000
len_recording = 1000
sampling_rate = 30000
n_channels = 384
fname_out = '/media/cat/julien/test_final_loc_code/residuals/residual.bin'
dtype_out = 'float32'
dtype_in = 'float32'
n_batches = 1000

##### COMPUTE RESIDUALS 
'''
This is necessary for denoising spikes and removing collisions
It needs to be ran only once per datasets, as residual.bin file can be stored and reused 
'''
residual_obj = RESIDUAL(bin_file,
                 fname_templates,
                 fname_spike_train,
                 n_batches,
                 len_recording,
                 sampling_rate,
                 n_channels,
                 fname_out,
                 dtype_in,
                 dtype_out)
        

residual_obj.compute_residual('/media/cat/julien/test_final_loc_code/residuals/')
residual_obj.save_residual()


##### LOCALIZE SPIKES 

localizer_obj = LOCALIZER(bin_file, residual_file, dtype_input, fname_spike_train, fname_templates, geom_path, denoiser_weights, denoiser_min)
localizer_obj.get_offsets()
localizer_obj.compute_aligned_templates()
localizer_obj.load_denoiser()

for i in tqdm(range(n_batches)):
    localizer_obj.get_estimate(i, threshold = 6, output_directory = '/media/cat/julien/test_final_loc_code/position_results_files')


##### Merge results 

input_directory = '/media/cat/julien/test_final_loc_code/position_results_files'
output_directory = '/media/cat/julien/test_final_loc_code/final_results/'

get_merged_arrays('/media/cat/julien/test_final_loc_code/position_results_files', '/media/cat/julien/test_final_loc_code/final_results/', n_batches)






