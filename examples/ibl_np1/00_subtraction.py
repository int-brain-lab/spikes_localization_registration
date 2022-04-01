"""Detection, subtraction, denoising and localization script

This script is a CLI for the function `subtraction` in subtract.py,
which runs the subtraction and localization pass.

This will also run registration and add the registered z coordinates
and displacement map to the output HDF5 file from subtraction.

See the documentation of `subtract.subtraction` for lots of detail.
"""

from pathlib import Path
from spikeutils import run_cbin_ibl
SCRATCH_DIR = Path.home().joinpath('scratch')


# benchmarking datasets on ferret
pid = "8ca1a850-26ef-42be-8b28-c2e2d12f06d6"
cbin_file = Path(f"/datadisk/Data/spike_sorting/benchmark/raw/{pid}/_spikeglx_ephysData_g0_t0.imec0.ap.cbin")
# RS datasets
scratch_dir = SCRATCH_DIR.joinpath(cbin_file.parts[-2])
standardized_file = scratch_dir.joinpath(f"{cbin_file.stem}.normalized.bin")

h5_file = run_cbin_ibl(cbin_file, standardized_file, n_jobs=8, t_start=100, t_end=150, save_residual=False)

# import h5py
# import numpy as np
# h5 = h5py.File(h5_file, "r")
# [k for k in h5.keys()]
# h5['cleaned_waveforms']
# to_keep = ['channel_index', 'dispmap', 'end_sample', 'first_channels', 'geom', 'localizations', 'maxptps', 'spike_index',
#  'start_sample', 'tpca_components', 'tpca_mean', 'z_reg']
# for k in h5.keys():
#     if k not in to_keep:
#         continue
#     np.save(scratch_dir.joinpath(f"{k}.npy"), h5[k])
