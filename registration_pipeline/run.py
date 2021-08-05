import motion_estimate as me
import numpy as np

### change paths to your localization files (e.g. z localizations), and specify the direction
locs = 'z_results.npy'
times = 'spike_times.npy'
amps = 'max_ptp.npy'
direction = 'z'

### change path to geometry array
geomarray = np.load('geom.npy') # (num of channels, 2)

### set registration params
reg_win_num = 10 # set to 1 for rigid registration
reg_block_num = 100 # set to 1 for rigid registration
iteration_num = 2

### get motion estimate
total_shift, raster, registered_raster = me.estimate_motion(locs, times, amps, geomarray, direction)

### save results
np.save('total_shift.npy', total_shift) # motion estimate
np.save('raster.npy', raster) # original raster
np.save('registered_raster.npy', registered_raster) # registered raster
