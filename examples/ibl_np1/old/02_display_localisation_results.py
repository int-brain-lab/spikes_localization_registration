from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gridspec
import matplotlib.cm as cm
from one.api import ONE
import one.alf.io as alfio

from ibllib.ephys import neuropixel
from brainbox.io.one import load_spike_sorting_fast
from brainbox.plot import driftmap

h = neuropixel.trace_header(version=1)

pids = ['8ca1a850-26ef-42be-8b28-c2e2d12f06d6', '8413c5c6-b42b-4ec6-b751-881a54413628', 'ce24bbe9-ae70-4659-9e9c-564d1a865de8', 'ce397420-3cd2-4a55-8fd1-5e28321981f4']
pid = pids[3]


standardized_file = next(Path(f"/datadisk/Data/spike_sorting/benchmark/raw/{pid}/").glob("*.ap.normalized.bin"))
alf_path = Path(f"/datadisk/Data/spike_sorting/benchmark/sorters/ibl_1.1.0a02/{pid}/alf")
directory_localisation_merged = standardized_file.parent.joinpath('localisation_merged')


# if we're desperate for channels locations
# one = ONE()
# eid, pname = one.pid2eid(pid)
# _, _, channels = load_spike_sorting_fast(eid=eid, probe=pname, nested=False, dataset_types=['spikes.depths'])

clusters = alfio.load_object(alf_path, 'clusters')
spikes = alfio.load_object(alf_path, 'spikes')

# TODO this would be amecable to ALF format with easy loaders
ptp_array = np.load(directory_localisation_merged.joinpath('results_max_ptp_merged.npy'))
idx_good = np.where(ptp_array != 0)[0]
x_results = np.load(directory_localisation_merged.joinpath('results_x_merged.npy'))[idx_good]
y_results = np.load(directory_localisation_merged.joinpath('results_y_merged.npy'))[idx_good]
z_results = np.load(directory_localisation_merged.joinpath('results_z_merged.npy'))[idx_good]
times_array = np.load(directory_localisation_merged.joinpath('times_read.npy'))[idx_good]

plt.figure()
plt.hist(ptp_array[idx_good], bins = 50)
plt.show()

ptp_rescaled = ptp_array[idx_good] - ptp_array[idx_good].min()
ptp_rescaled = ptp_rescaled/ptp_rescaled.max()
ptp_rescaled[ptp_rescaled >= 0.4] = 0.4
# plt.figure()
# plt.hist(ptp_rescaled, bins = 50)
# plt.show()


fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [.15, .85]}, figsize=(20, 10), sharey=True)
driftmap(spikes['times'], spikes['depths'], t_bin=0.1, d_bin=5, ax=axs[1])
f_ax = axs[0]
vir = cm.get_cmap('viridis')

f_ax.set_xlim((-40, 80))
f_ax.scatter(x_results, z_results, s = 2, color = vir(ptp_rescaled), alpha = 0.005) #vir(ptp_scaled_high_units) 0.05
f_ax.scatter(h['x'], h['y'], c = 'orange', label = "NP channels", marker='s', s= 10)

# f_ax.set_yticks([])
f_ax.set_ylim(min(h['y']), max(h['y']))

f_ax.set_xlabel("X", fontsize = 22)

f_ax.set_title("LS position \n by max-ptp", fontsize = 18)


axs[1].set_title(pid)

fig.savefig(standardized_file.parent.joinpath('localisation.png'))