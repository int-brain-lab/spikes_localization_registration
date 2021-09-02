"""
Detection pipeline
"""
import logging
import os
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import parmap

from detect import Detect
from localization_pipeline.denoise import Denoise

from yass.util import file_loader
from yass.threshold.detect import voltage_threshold
from yass.detect.deduplication import deduplicate_gpu, deduplicate
from yass.detect.output import gather_result

### If set to false, run threshold instead
## ADD ARGUMENTS
apply_nn = True
geom_array = 
spatial_radius


#### ADD ARGUMENTS AND CHANGE ARGUMENTS




def read_data(bin_file, dtype_str, data_start, data_end, geom_array, channels=None):
    dtype = np.dtype(dtype)
    if channels is None:
        n_channels = geom_array.shape[0]
    else:
        n_channels = len(channels)
    with open(bin_file, "rb") as fin:

        fin.seek(int((data_start)*dtype.itemsize*n_channels), os.SEEK_SET)
        data = np.fromfile(
            fin, dtype=dtype,
            count=int((data_end - data_start)*n_channels))
    fin.close()
    
    data = data.reshape(-1, n_channels)
    if channels is not None:
        data = data[:, channels]

    return data

def read_data_batch(batch_id, rec_len, sampling_rate, n_sec_chunk, len_recording, geom_array, dtype_str, add_buffer=False, channels=None):
    dtype = np.dtype(dtype_str)
    batch_size = sampling_rate*n_sec_chunk
    indexes = np.arange(0, len_recording*sampling_rate, batch_size)
    indexes = np.hstack((indexes, len_recording*sampling_rate))

    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([indexes[k], indexes[k + 1]])
    idx_list = np.int64(np.vstack(idx_list))


    # batch start and end
    data_start, data_end = idx_list[batch_id]
    # add buffer if asked
    if add_buffer:
        data_start -= buffer_templates
        data_end += buffer_templates

        # if start is below zero, put it back to 0 and and zeros buffer
        if data_start < 0:
            left_buffer_size = - data_start
            data_start = 0
        else:
            left_buffer_size = 0

        # if end is above rec_len, put it back to rec_len and and zeros buffer
        if data_end > rec_len + offset:
            right_buffer_size = data_end - rec_len
            data_end = rec_len
        else:
            right_buffer_size = 0

    #data_start= int(data_start)
    #data_end = int(data_end)
    # read data
    data = read_data(data_start, data_end, channels)
    # add leftover buffer with zeros if necessary
    if channels is None:
        n_channels = geom_array.shape[0]
    else:
        n_channels = len(channels)

    if add_buffer:
        left_buffer = np.zeros(
            (left_buffer_size, n_channels),
            dtype=dtype)
        right_buffer = np.zeros(
            (right_buffer_size, n_channels),
            dtype=dtype)
        if channels is not None:
            left_buffer = left_buffer[:, channels]
            right_buffer = right_buffer[:, channels]
        data = np.concatenate((left_buffer, data, right_buffer), axis=0)

    return data

def read_data_batch_batch(batch_id, n_sec_chunk_small, sampling_rate, buffer, dtype_str, add_buffer=False, channels=None):
    '''
    this is for nn detection using gpu
    get a batch and then make smaller batches
    '''
    dtype = np.dtype(dtype_str)
    data = read_data_batch(batch_id, add_buffer, channels) ## ADD ARGUMENTS
    
    T, C = data.shape
    T_mini = int(sampling_rate*n_sec_chunk_small)

    if add_buffer:
        T = T - 2*buffer
    else:
        buffer = 0

    # # batch sizes
    # indexes = np.arange(0, T, T_mini)
    # indexes = np.hstack((indexes, T))
    # indexes += buffer


    indexes = np.arange(0, T, T_mini)
    indexes = np.hstack((indexes, indexes[-1]+T_mini))
    indexes += buffer
   
   
    n_mini_batches = len(indexes) - 1
    # add addtional buffer if needed
    if n_mini_batches*T_mini > T:
        T_extra = n_mini_batches*T_mini - T

        pad_zeros = np.zeros((T_extra, C),
            dtype=dtype)

        data = np.concatenate((data, pad_zeros), axis=0)
    data_loc = np.zeros((n_mini_batches, 2), 'int32')
    data_batched = np.zeros((n_mini_batches, T_mini + 2*buffer, C), 'float32')
    for k in range(n_mini_batches):
        data_batched[k] = data[indexes[k]-buffer:indexes[k+1]+buffer]
        data_loc[k] = [indexes[k], indexes[k+1]]
    return data_batched, data_loc

def make_channel_index(neighbors, channel_geometry, steps=1):
    """
    Compute an array whose whose ith row contains the ordered
    (by distance) neighbors for the ith channel
    """
    C, C2 = neighbors.shape

    if C != C2:
        raise ValueError('neighbors is not a square matrix, verify')

    # get neighbors matrix
    neighbors = n_steps_neigh_channels(neighbors, steps=steps)

    # max number of neighbors for all channels
    n_neighbors = np.max(np.sum(neighbors, 0))

    # FIXME: we are using C as a dummy value which is confusing, it may
    # be better to use something else, maybe np.nan
    # initialize channel index, initially with a dummy C value (a channel)
    # that does not exists
    channel_index = np.ones((C, n_neighbors), 'int32') * C

    # fill every row in the matrix (one per channel)
    for current in range(C):

        # indexes of current channel neighbors
        neighbor_channels = np.where(neighbors[current])[0]

        # sort them by distance
        ch_idx, _ = order_channels_by_distance(current, neighbor_channels,
                                               channel_geometry)

        # fill entries with the sorted neighbor indexes
        channel_index[current, :ch_idx.shape[0]] = ch_idx

    return channel_index

def find_channel_neighbors(geom, radius):
    """Compute a neighbors matrix by using a radius
    Parameters
    ----------
    geom: np.array
        Array with the cartesian coordinates for the channels
    radius: float
        Maximum radius for the channels to be considered neighbors
    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        Symmetric boolean matrix with the i, j as True if the ith and jth
        channels are considered neighbors
    """
    return (squareform(pdist(geom)) <= radius)


def run(standardized_path, standardized_dtype,
        output_directory, run_chunk_sec='full'):
    ## CHANGE ARGUMENTS
           
    """Execute detect step
    Parameters
    ----------
    standardized_path: str or pathlib.Path
        Path to standardized data binary file
    standardized_dtype: string
        data type of standardized data
    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/
    Returns
    -------
    clear_scores: numpy.ndarray (n_spikes, n_features, n_channels)
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels
    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)
    spike_index_call: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for all spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)
    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/ (if save_results is
    True):
    * ``spike_index_clear.npy`` - Same as spike_index_clear returned
    * ``spike_index_all.npy`` - Same as spike_index_collision returned
    * ``rotation.npy`` - Rotation matrix for dimensionality reduction
    * ``scores_clear.npy`` - Scores for clear spikes
    Threshold detector runs on CPU, neural network detector runs CPU and GPU,
    depending on how pytorch is configured.
    Examples
    --------
    .. literalinclude:: ../../examples/pipeline/detect.py
    """
    
    # make output directory if not exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    fname_spike_index = os.path.join(
        output_directory, 'spike_index.npy')
    if os.path.exists(fname_spike_index):
        print("Detection already done!")
        return fname_spike_index

    ##### detection #####
    # save directory for temp files
    output_temp_files = os.path.join(
        output_directory, 'batch')
    if not os.path.exists(output_temp_files):
        os.mkdir(output_temp_files)

    # neighboring channels
    neigh_channels = geom.find_channel_neighbors(geom_array, spatial_radius)

    # run detection
    if apply_nn:
        ## CHANGE ARGUMENTS
        run_neural_network(
            standardized_path,
            standardized_dtype,
            output_temp_files,
            run_chunk_sec=run_chunk_sec)

    else:
        run_voltage_treshold(standardized_path,
                             standardized_dtype,
                             output_temp_files,
                             n_filters, 
                             spike_size_nn, 
                             channel_index, 
                             run_chunk_sec=run_chunk_sec)

    ##### gather results #####
    gather_result(fname_spike_index,
                  output_temp_files)

    return fname_spike_index


def run_neural_network(standardized_path, standardized_dtype, output_directory, n_sec_chunk, n_processors, n_sec_chunk_gpu_detect, detect_threshold,
    path_nn_detector, n_filters_detect, spike_size_nn, path_nn_denoiser, n_filters_denoise, filter_sizes_denoise, run_chunk_sec='full'):
                           
    """Run neural network detection
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = 0 
    # load NN detector
    detector = Detect(n_filters_detect, spike_size_nn, channel_index)
    detector.load(path_nn_detector)

    # load NN denoiser
    denoiser = Denoise(n_filters_denoise,
                       filter_sizes_denoise,
                       spike_size_nn)
    denoiser.load(path_nn_denoiser)

    # get data reader
    batch_length = n_sec_chunk*n_processors
    n_sec_chunk = n_sec_chunk_gpu_detect
    print ("   batch length to (sec): ", batch_length, 
           " (longer increase speed a bit)")
    print ("   length of each seg (sec): ", n_sec_chunk)
    buffer = spike_size_nn
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec
    
    # neighboring channels

    channel_index_dedup = make_channel_index(
        neigh_channels, geom_array, steps=2)

    # loop over each chunk
    batch_ids = np.arange(n_batches)

    if False:
        batch_ids_split = np.split_array(batch_ids, len(CONFIG.torch_devices))
        processes = []
        for ii, device in enumerate([torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]): ### SEVERAL DEVICES??
            p = mp.Process(target=run_nn_detection_batch,
                           args=(batch_ids_split[ii], output_directory, reader, n_sec_chunk,
                                 detector, denoiser, channel_index_dedup,
                                 detect_threshold, device))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        run_nn_detction_batch(batch_ids, output_directory, reader, n_sec_chunk,
                                 detector, denoiser, channel_index_dedup,
                                 detect_threshold, device=0) #CONFIG.resources.gpu_id?


def run_nn_detection_batch(batch_ids, output_directory,
                          reader, n_sec_chunk,
                          detector, denoiser,
                          channel_index_dedup,
                          detect_threshold,
                          device):

    detector = detector.to(device)
    denoiser = denoiser.to(device)

    for batch_id in batch_ids:
        # skip if the file exists
        fname = os.path.join(
            output_directory,
            "detect_" + str(batch_id).zfill(5) + '.npz')

        if os.path.exists(fname):
            continue


        # get a bach of size n_sec_chunk
        # but partioned into smaller minibatches of 
        # size n_sec_chunk_gpu
        ### UPDATE ARGUMENTS
        batched_recordings, minibatch_loc_rel = read_data_batch_batch(
            batch_id,
            #CONFIG.detect.n_sec_chunk,
            n_sec_chunk,
            add_buffer=True)

        # offset for big batch
        ### GET IDX LIST + BUFFER BEFORE
        batch_offset = idx_list[batch_id, 0] - buffer
        # location of each minibatch (excluding buffer)
        minibatch_loc = minibatch_loc_rel + batch_offset
        spike_index_list = []
        spike_index_dedup_list = []
        for j in range(batched_recordings.shape[0]):
            # detect spikes and get wfs
            spike_index, wfs = detector.get_spike_times(
                torch.FloatTensor(batched_recordings[j]).to(device),
                threshold=detect_threshold)

            # denoise and take ptp as energy
            if len(spike_index) == 0:
                del spike_index, wfs
                continue

            wfs_denoised = denoiser(wfs)[0].data
            energy = (torch.max(wfs_denoised, 1)[0] - torch.min(wfs_denoised, 1)[0])

            # deduplicate
            spike_index_dedup = deduplicate_gpu(
                spike_index, energy,
                batched_recordings[j].shape,
                channel_index_dedup)

            # convert to numpy
            spike_index_cpu = spike_index.cpu().data.numpy()
            spike_index_dedup_cpu = spike_index_dedup.cpu().data.numpy()

            # update the location relative to the whole recording
            spike_index_cpu[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
            spike_index_dedup_cpu[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
            spike_index_list.append(spike_index_cpu)
            spike_index_dedup_list.append(spike_index_dedup_cpu)

            del wfs
            del wfs_denoised
            del energy
            del spike_index
            del spike_index_dedup

            torch.cuda.empty_cache()

        #if processing_ctr%100==0:
        print('batch : {}'.format(batch_id))

        # save result
        np.savez(fname,
                 spike_index=spike_index_list,
                 spike_index_dedup=spike_index_dedup_list,
                 minibatch_loc=minibatch_loc)
        
    del detector
    del denoiser


def run_voltage_treshold(standardized_path, standardized_dtype,
                         output_directory, run_chunk_sec='full'):
                           
    """Run detection that thresholds on amplitude
    """
    logger = logging.getLogger(__name__)

    # get data reader
    #n_sec_chunk = CONFIG.resources.n_sec_chunk*CONFIG.resources.n_processors
    ### ADD Arguments
    batch_length = n_sec_chunk
    n_sec_chunk = 0.5
    print ("   batch length to (sec): ", batch_length, 
           " (longer increase speed a bit)")
    print ("   length of each seg (sec): ", n_sec_chunk)
    buffer = CONFIG.spike_size
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    # reader = READER(standardized_path,
    #                 standardized_dtype,
    #                 CONFIG,
    #                 batch_length,
    #                 buffer,
    #                 chunk_sec)

    # number of processed chunks
    n_mini_per_big_batch = int(np.ceil(batch_length/n_sec_chunk))    
    total_processing = int(n_batches*n_mini_per_big_batch)

    # neighboring channels
    channel_index = make_channel_index(
        CONFIG.neigh_channels, CONFIG.geom, steps=2)
    
    ## ADD ARGUMENTS
    if multi_processing_flag:
        parmap.starmap(run_voltage_threshold_parallel, 
                       list(zip(np.arange(reader.n_batches))),
                       reader,
                       n_sec_chunk,
                       threshold,
                       channel_index,
                       output_directory,
                       processes=n_processors,
                       pm_pbar=True)                
    else:
        for batch_id in range(reader.n_batches):
            run_voltage_threshold_parallel(
                batch_id,
                reader,
                n_sec_chunk,
                CONFIG.detect.threshold,
                channel_index,
                output_directory)


def run_voltage_threshold_parallel(batch_id, reader, n_sec_chunk,
                                   threshold, channel_index,
                                   output_directory):

    # skip if the file exists
    fname = os.path.join(
        output_directory,
        "detect_" + str(batch_id).zfill(5) + '.npz')

    if os.path.exists(fname):
        return

    # get a bach of size n_sec_chunk
    # but partioned into smaller minibatches of 
    # size n_sec_chunk_gpu
    batched_recordings, minibatch_loc_rel = read_data_batch_batch(
        batch_id,
        n_sec_chunk,
        add_buffer=True)

    # offset for big batch
    batch_offset = idx_list[batch_id, 0] - buffer
    # location of each minibatch (excluding buffer)
    minibatch_loc = minibatch_loc_rel + batch_offset
    spike_index_list = []
    spike_index_dedup_list = []
    for j in range(batched_recordings.shape[0]):
        spike_index, energy = voltage_threshold(
            batched_recordings[j], 
            threshold)

        # move to gpu
        spike_index = torch.from_numpy(spike_index)
        energy = torch.from_numpy(energy)

        # deduplicate
        spike_index_dedup = deduplicate_gpu(
            spike_index, energy,
            batched_recordings[j].shape,
            channel_index)

        # convert to numpy
        spike_index = spike_index.cpu().data.numpy()
        spike_index_dedup = spike_index_dedup.cpu().data.numpy()
        
        # update the location relative to the whole recording
        spike_index[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
        spike_index_dedup[:, 0] += (minibatch_loc[j, 0] - reader.buffer)
        spike_index_list.append(spike_index)
        spike_index_dedup_list.append(spike_index_dedup)

    # save result
    np.savez(fname,
             spike_index=spike_index_list,
             spike_index_dedup=spike_index_dedup_list,
             minibatch_loc=minibatch_loc)