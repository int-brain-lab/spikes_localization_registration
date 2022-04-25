import logging

from pathlib import Path
from spikeutils import run_cbin_ibl, detections_f1_score

import numpy as np
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from spikeutils import load_npy_yasap

SCRATCH_DIR = Path.home().joinpath('scratch')
logger = logging.getLogger('ibllib')
AWS_ROOT_PATH = Path('data')

one = ONE()
# pids = list(np.load("/home/olivier/Documents/PYTHON/00_IBL/paper-reproducible-ephys/repeated_site_pids.npy"))
root_path = Path("/mnt/s0/Data")
pids = ['ce397420-3cd2-4a55-8fd1-5e28321981f4',  #  0 ok
        # 'f4bd76a6-66c9-41f3-9311-6962315f8fc8',  #  1 aws
        'e31b4e39-e350-47a9-aca4-72496d99ff2a',  #  2 ok
        '6fc4d73c-2071-43ec-a756-c6c6d8322c8b',  #  3 no registration
        '1e176f17-d00f-49bb-87ff-26d237b525f1',  #  4 ok
        'b25799a5-09e8-4656-9c1b-44bc9cbb5279',  #  5 ok
        'c17772a9-21b5-49df-ab31-3017addea12e',  #  6 ok
        # '0851db85-2889-4070-ac18-a40e8ebd96ba',  #  7 ERROR: processed killed
        # 'eeb27b45-5b85-4e5c-b6ff-f639ca5687de',  #  8 aws
        # '69f42a9c-095d-4a25-bca8-61a9869871d3',  #  9 aws
        ]


for pid in pids:
    fs = 30000
    ss = {'yasap': {}}
    YAS_DIR = Path(f"/datadisk/Data/spike_sorting/re_datasets/{pid}")
    ss['yasap']['spikes'], ss['yasap']['channels'] = load_npy_yasap(YAS_DIR, fs)

    loc = np.load(YAS_DIR.joinpath("localizations.npy"))



    ssl = SpikeSortingLoader(pid=pid, one=one)
    for coll in ssl.collections:
        sorter_name = coll.replace(f"alf/{ssl.pname}", "").replace('/', "") or 'ks2'
        spikes, clusters, channels = ssl.load_spike_sorting(collection=coll, dataset_types=['spikes.samples'])
        ss[sorter_name] = {}
        ss[sorter_name]['spikes'], ss[sorter_name]['clusters'], ss[sorter_name]['channels'] = (spikes, clusters, channels)
        spikes['times'] = spikes['samples'] / fs  # NB: the clock is the self probe clock, so careful if trying to put some behaviour data !


    def get_txy(spikes, clusters, channels):
        xy = (channels['axial_um'] + 1j * channels['lateral_um'])  #
        xy = xy[clusters['channels'][spikes['clusters']]]
        return spikes['times'], xy


    ta, xya = get_txy(ss['ks2']['spikes'], ss['ks2']['clusters'], ss['ks2']['channels'])
    tc, xyc = get_txy(ss['pykilosort']['spikes'], ss['pykilosort']['clusters'], ss['pykilosort']['channels'])
    tb = ss['yasap']['spikes']['times']
    xyb = loc[:, 2] + 1j * loc[:, 0]


    import yaml
    f1_file = YAS_DIR.joinpath('f1.yaml')
    if not f1_file.exists():
        f1 = {}
        f1['ks2_yasap'] = detections_f1_score(ta, xya, tb, xyb)
        f1['pyks_yasap'] = detections_f1_score(tc, xyc, tb, xyb)
        f1['ks2_pyks2'] = detections_f1_score(ta, xya, tc, xyc)
        with f1_file.open('w+') as fid:
            yaml.dump(f1, fid)
