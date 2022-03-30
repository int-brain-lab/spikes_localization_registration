"""
This module if for basic util functions for visualization and quality control of the spike sorting.
It also includes wrappers to perform detections and denoising on waveforms
"""

"""
Detects and Denoise spikes within a numpy array of voltage traces
The goal is to benchmark the spike detector
"""
from pathlib import Path

import numpy as np
import scipy
import torch
import pandas as pd
import matplotlib.pyplot as plt

from ibllib.dsp import voltage
from ibllib.ephys.neuropixel import trace_header
from ibllib.io import spikeglx
import ibllib.ephys.spikes as spikes
from brainbox.io.one import SpikeSortingLoader

from viewephys.gui import viewephys

from detect.run import find_channel_neighbors, make_channel_index
from detect.deduplication import deduplicate_gpu
import detect.detector
from localization_pipeline.denoiser import Denoise


def plot_spikes_view_ephys(spikes, clusters, t0, nsecs, fs, rgb, label, eqcs):
    """
    Add spikes to a raw data viewer
    """
    stimes = spikes['samples'] / fs
    # stimes = spikes['times']
    slice_spikes = slice(np.searchsorted(stimes, t0), np.searchsorted(stimes, t0 + nsecs))
    t = (stimes[slice_spikes] - t0) * 1e3
    c = clusters.channels[spikes.clusters[slice_spikes]]
    for k in eqcs:
        eqcs[k].ctrl.add_scatter(t, c, rgb, label=label)


def init_detector(cbin_file):
    sr = spikeglx.Reader(cbin_file)
    REPO_PATH = Path(detect.detector.__file__).parents[1]
    APPLY_NN = True
    BATCH_SIZE_SECS = 1
    DETECT_THRESHOLD = -4  # in normalized units for once
    params = dict(
        apply_nn=APPLY_NN,  # If set to false, run voltage threshold instead of NN detector,
        detect_threshold=.56 if APPLY_NN else 6,  # 0.5 if apply NN, 4/5/6 otherwise,
        filter_sizes_denoise=[5, 11, 21],
        geom_array=np.c_[sr.geometry['x'], sr.geometry['y']],
        len_recording=sr.rl, n_batches=sr.rl / 2,
        n_filters_denoise=[16, 8, 4],
        n_filters_detect=[16, 8, 8],
        n_processors=4,
        n_sec_chunk=BATCH_SIZE_SECS,
        n_sec_chunk_gpu_detect=.1,
        output_directory=cbin_file.parent.joinpath("detection"),
        path_nn_denoiser=REPO_PATH.joinpath('pretrained_denoiser/denoise.pt'),
        path_nn_detector=REPO_PATH.joinpath('pretrained_detector/detect_np1.pt'),
        run_chunk_sec='full',
        sampling_rate=sr.fs,
        spatial_radius=70,
        spike_size_nn=121,
        standardized_dtype='float32',
        standardized_path=None,
    )

    neigh_channels = find_channel_neighbors(params['geom_array'], params['spatial_radius'])
    channel_index = make_channel_index(neigh_channels, params['geom_array'])

    # need to run by small batches of 10000 samples
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = detect.detector.Detect(params['n_filters_detect'], params['spike_size_nn'], channel_index)
    detector.load(params['path_nn_detector'])
    detector.to(device)

    denoiser = Denoise(params['n_filters_denoise'], params['filter_sizes_denoise'], params['spike_size_nn'])
    denoiser.load(params['path_nn_denoiser'])
    denoiser.to(device)


    return detector, denoiser, params, channel_index


def detect_nn(data, detector, denoiser, channel_index, params):
    """Apply the detector to a numpy array [nsamples, ntraces]"""
    # TODO: we need overlaps here
    DETECT_BATCH = 10000  # need to run by small size data in order not to run out of GPU memory
    nbatches = data.shape[0] / DETECT_BATCH
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert nbatches % 1 == 0
    all_detects = []
    for m in np.arange(nbatches):
        first = int(m * DETECT_BATCH)
        last = int((m + 1) * DETECT_BATCH)
        data_ = torch.FloatTensor(data[first: last, :]).to(device)
        spike_index, wfs = detector.get_spike_times(data_, threshold=params['detect_threshold'])

        wfs_denoised = denoiser(wfs)[0].data
        energy = (torch.max(wfs_denoised, 1)[0] - torch.min(wfs_denoised, 1)[0])

        # deduplicate
        spike_index_dedup = deduplicate_gpu(spike_index, energy, data_.shape, channel_index)

        detects = spike_index_dedup.detach().cpu().numpy()
        detects[:, 0] = detects[:, 0] + m * DETECT_BATCH
        all_detects.append(detects)

        del data_
        del wfs
        del wfs_denoised
        del energy
        del spike_index
        del spike_index_dedup

        torch.cuda.empty_cache()

    all_detects = np.concatenate(all_detects)

    # plt.plot(mmap[1:10000, 55])
    npz_batches = np.load(
        '/datadisk/Data/spike_sorting/benchmark/raw/8ca1a850-26ef-42be-8b28-c2e2d12f06d6/detection/batch/detect_00000.npz')
    # npz_batches.files ['spike_index', 'spike_index_dedup', 'minibatch_loc']
    spikes_thresh = npz_batches['spike_index_dedup'][0]

    return all_detects




def apply_ks2_whitening(raw, kwm, sr, channels):
    if 'rawInd' not in channels:
        _, iraw, _ = np.intersect1d(sr.geometry['x'] * 1e4 + sr.geometry['y'],
                       channels['lateral_um'] * 1e4 + channels['axial_um'], return_indices=True)
    else:
        iraw = channels['rawInd']
    iraw = np.sort(iraw)
    carbutt  = raw - np.mean(raw, axis=0)
    butter_kwargs = {'N': 3, 'Wn': np.array([300, 8000]) / sr.fs * 2, 'btype': 'bandpass'}
    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    carbutt = scipy.signal.sosfiltfilt(sos, carbutt)
    ks2 = np.zeros_like(raw)
    ks2[iraw, :] = np.matmul(kwm, carbutt[iraw, :])
    scaling = np.std(carbutt)  # choose and apply a constant scaling throughout
    ks2 = ks2 * np.std(carbutt) / np.std(ks2)
    return ks2



