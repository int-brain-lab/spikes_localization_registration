"""Detection, subtraction, denoising and localization script

This script is a CLI for the function `subtraction` in subtract.py,
which runs the subtraction and localization pass.

This will also run registration and add the registered z coordinates
and displacement map to the output HDF5 file from subtraction.

See the documentation of `subtract.subtraction` for lots of detail.
"""

from pathlib import Path
import h5py
from spikeutils import run_cbin_ibl
SCRATCH_DIR = Path.home().joinpath('scratch')


# benchmarking datasets on ferret
cbin_file = Path("/datadisk/Data/spike_sorting/benchmark/raw/8ca1a850-26ef-42be-8b28-c2e2d12f06d6/_spikeglx_ephysData_g0_t0.imec0.ap.cbin")
scratch_dir = SCRATCH_DIR.joinpath(cbin_file.parts[-2])
standardized_file = scratch_dir.joinpath(f"{cbin_file.stem}.normalized.bin")
h5_file = run_cbin_ibl(cbin_file, standardized_file, n_jobs=8, t_start=100, t_end=150, save_residual=False)
