import logging

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from iblutil.numerical import ismember2d

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from spikeutils import load_npy_yasap
from ibllib.ephys.neuropixel import trace_header

from visualization.matplotlib import plotlocs, driftmaps, displacement_map
from visualization.pyqt import raw_data

h = trace_header(version=1)
SCRATCH_DIR = Path.home().joinpath('scratch')
V_T0 = [60 * 10, 60 * 30, 60 * 50]  # raw data samples at 10, 30, 50 min in
FIG_DIR = Path(f"/datadisk/team_drives/WG-Neural-Analysis/Spike-Sorting-Analysis/re_datasets")
logger = logging.getLogger('ibllib')
AWS_ROOT_PATH = Path('data')

one = ONE()
# pids = list(np.load("/home/olivier/Documents/PYTHON/00_IBL/paper-reproducible-ephys/repeated_site_pids.npy"))
root_path = Path("/mnt/s0/Data")
pids = ['ce397420-3cd2-4a55-8fd1-5e28321981f4',  #  0 ok
        'e31b4e39-e350-47a9-aca4-72496d99ff2a',  #  2 ok
        '1e176f17-d00f-49bb-87ff-26d237b525f1',  #  4 ok
        'b25799a5-09e8-4656-9c1b-44bc9cbb5279',  #  5 ok
        'c17772a9-21b5-49df-ab31-3017addea12e',  #  6 ok
]


IMIN = 0
IMAX = 500
for i, pid in enumerate(pids):
    plt.close('all')
    if i < IMIN or i > IMAX:
        continue
    plt.close('all')
    fs = 30000

    # load YAS Data
    YAS_DIR = Path(f"/datadisk/Data/spike_sorting/re_datasets/{pid}")  # contains npy output of localisation / registrtaio
    ss = {'yas': {}, 'yasreg': {}}
    ss['yas']['spikes'], ss['yas']['channels'] = load_npy_yasap(YAS_DIR, fs, registration=False)
    ss['yasreg']['spikes'], ss['yasreg']['channels'] = load_npy_yasap(YAS_DIR, fs, registration=True)

    dm = np.load(YAS_DIR.joinpath("dispmap.npy"))  # dm shape ndephts, ntimes
    loc = np.load(YAS_DIR.joinpath("localizations.npy"))
    geom = np.load(YAS_DIR.joinpath("geom.npy"))

    # load spike sorting data from IBL
    ssl = SpikeSortingLoader(pid=pid, one=one)
    for coll in ssl.collections:
        if coll.endswith('ks2_preproc_tests'):
            continue
        sorter_name = coll.replace(f"alf/{ssl.pname}", "").replace('/', "") or 'ks2'
        spikes, clusters, channels = ssl.load_spike_sorting(collection=coll, dataset_types=['spikes.samples'])
        # TODO Noam: run your QC metrics here
        ss[sorter_name] = {}
        ss[sorter_name]['spikes'], ss[sorter_name]['clusters'], ss[sorter_name]['channels'] = (spikes, clusters, channels)
        spikes['times'] = spikes['samples'] / fs  # NB: the clock is the self probe clock, so careful if trying to put some behaviour data !
        # get the raw data filechannelindices
        _, ic = ismember2d(np.c_[channels['lateral_um'], channels['axial_um']],
                           np.c_[h['x'], h['y']])
        spikes['raw_channels'] = ic[clusters['channels'][spikes['clusters']]]

    for i, k in enumerate(ss):
        spikes = ss[k]['spikes']
        print(f"{pid} {k} {spikes['times'].size / spikes['times'][-1]} spikes/h, {spikes['times'].size}")

    # load pykilosort drift
    pyks_drift = one.load_object(ssl.eid, 'drift', collection=f'alf/{ssl.pname}/pykilosort')
    d = one.load_object(ssl.eid, 'drift_depths', collection=f'alf/{ssl.pname}/pykilosort')
    pyks_drift['depths'] = d['um'][0]


    figs = driftmaps(ss, output_dir=FIG_DIR, pid=pid)
    ax = figs[0].axes[0]
    fig = displacement_map(
        displacement=dm, label='yass', xlim=ax.get_xlim(), ylim=ax.get_ylim(),
        output_file=FIG_DIR.joinpath(f"{pid}_drift_yass.png")
    )
    fig = displacement_map(
        displacement=pyks_drift['um'].transpose(), label='pykilosort', xlim=ax.get_xlim(), ylim=ax.get_ylim(),
        output_file=FIG_DIR.joinpath(f"{pid}_drift_pykilosort.png"),
        extent=np.r_[pyks_drift['times'][[0, -1]], pyks_drift['depths'][[0, -1]]],
    )

    raw_data(pid, times=V_T0, channels=ss['pykilosort']['channels'], one=one,
             output_dir=FIG_DIR.joinpath('raw'), ss=ss)
    #
    # plotlocs(*np.hsplit(loc, 5), geom)
    # sel = np.sort(np.floor(np.random.random(100000) * loc.shape[0])).astype(np.int32)
    # fig = plotlocs(*np.hsplit(loc, 5), geom, which=sel)
    # break




