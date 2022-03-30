from pathlib import Path
import numpy as np
import tqdm
from ibllib.dsp import voltage
from ibllib.io import spikeglx
import shutil
SCRATCH_DIR = Path.home().joinpath('scratch')

cbin_file = Path("/datadisk/Data/spike_sorting/benchmark/raw/8ca1a850-26ef-42be-8b28-c2e2d12f06d6/_spikeglx_ephysData_g0_t0.imec0.ap.cbin")
standardized_file = SCRATCH_DIR.joinpath(f"{cbin_file.stem}.normalized.bin")

# h = neuropixel.trace_header(version=1)
sr = spikeglx.Reader(cbin_file)

shutil.copy(sr.file_meta_data, SCRATCH_DIR.joinpath(f"{sr.file_meta_data.stem}.normalized.meta"))

h = sr.geometry
if not standardized_file.exists():
    batch_size_secs = 1
    batch_intervals_secs = 50
    # scans the file at constant interval, with a demi batch starting offset
    nbatches = int(np.floor((sr.rl - batch_size_secs) / batch_intervals_secs - 0.5))
    wrots = np.zeros((nbatches, sr.nc - sr.nsync, sr.nc - sr.nsync))
    for ibatch in tqdm.trange(nbatches, desc="destripe batches"):
        ifirst = int((ibatch + 0.5) * batch_intervals_secs * sr.fs + batch_intervals_secs)
        ilast = ifirst + int(batch_size_secs * sr.fs)
        sample = voltage.destripe(sr[ifirst:ilast, :-sr.nsync].T, fs=sr.fs, neuropixel_version=1)
        np.fill_diagonal(wrots[ibatch, :, :], 1 / voltage.rms(sample) * sr.sample2volts[:-sr.nsync] )

    wrot = np.median(wrots, axis=0)
    voltage.decompress_destripe_cbin(
        sr.file_bin, h=h, wrot=wrot, output_file=standardized_file, dtype=np.float32, nc_out=sr.nc - sr.nsync)


print(f"python scripts/subtract.py {standardized_file} {SCRATCH_DIR} --t_start 250 --t_end 350 --n_jobs 2 --noresidual")