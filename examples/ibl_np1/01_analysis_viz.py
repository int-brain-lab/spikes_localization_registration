from ibllib.io import spikeglx
import numpy as np
import h5py
from visualization.matplotlib import plotlocs


h5_file = '/home/olivier/scratch/8ca1a850-26ef-42be-8b28-c2e2d12f06d6/subtraction__spikeglx_ephysData_g0_t0.imec0.ap.normalized_t_100_150.h5'
h5 = h5py.File(h5_file, 'r')
plotlocs(*np.hsplit(h5['localizations'], 5), h5['geom'])


h5.keys()
h5['spike_index'][:, 0]
h5['spike_index'][:, 1]

