"""Detection, subtraction, denoising and localization script

This script is a CLI for the function `subtraction` in subtract.py,
which runs the subtraction and localization pass.

This will also run registration and add the registered z coordinates
and displacement map to the output HDF5 file from subtraction.

See the documentation of `subtract.subtraction` for lots of detail.
"""

from pathlib import Path
from spikeutils import run_cbin_ibl, h5_to_npy
SCRATCH_DIR = Path.home().joinpath('scratch')
SCRATCH_DIR = Path("/media/olivier/Seagate Expansion Drive/scratch")
# SCRATCH_DIR = Path("/datadisk/scratch")
# benchmarking datasets on ferret
pid = "8ca1a850-26ef-42be-8b28-c2e2d12f06d6"  # done !
pid = "ce397420-3cd2-4a55-8fd1-5e28321981f4"  # done !
pid = "8413c5c6-b42b-4ec6-b751-881a54413628"
pid = "ce24bbe9-ae70-4659-9e9c-564d1a865de8"

for pid in ["8413c5c6-b42b-4ec6-b751-881a54413628", "ce24bbe9-ae70-4659-9e9c-564d1a865de8"]:
    OUTPUT_DIR = Path(f"/datadisk/Data/spike_sorting/benchmark/sorters/yasap/{pid}")

    cbin_file = next(Path(f"/datadisk/Data/spike_sorting/benchmark/raw/{pid}").glob("*.ap.cbin"))
    # RS datasets
    scratch_dir = SCRATCH_DIR.joinpath(cbin_file.parts[-2])
    standardized_file = scratch_dir.joinpath(f"{cbin_file.stem}.normalized.bin")

    h5_file = run_cbin_ibl(cbin_file, standardized_file, n_jobs=8, t_start=0, t_end=None, save_residual=False)

    h5_to_npy(h5_file, output_dir=OUTPUT_DIR)
