#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script processes fMRI data from the HCP2021 dataset for the Working Memory task,
curated for the NMA-CN 2021 session. It uses functions from psy_process_lib.py for
individual steps of processing.

Specifically, the script loads BOLD activation data for all subjects and computes
first-level GLM estimates, adds identifier information and writes out to disk

@author: Pranay S. Yadav
"""
# Import modules
import psy_process_lib as ppl
import pandas as pd
from pathlib import Path

# Initialize paths for root dir
HCP_DIR = Path("/home/pranay/Projects/NMA-CN/Project/dataset/hcp_task_2021/")
regions_file = Path(
    "/home/pranay/Projects/NMA-CN/Project/dataset/hcp_task_2021/regions.npy"
)

# Directory for storing betas
outdir = HCP_DIR / "betas"
if not outdir.exists():
    outdir.mkdir()

# Load region info
regions = ppl.load_regions(regions_file)

# Iterate over all subjects and accumulate
accumulator = []
for subject in range(339):

    # Iterate over both runs
    for run in ["LR", "RL"]:

        # Fetch EVs
        EVs = ppl.load_single_EVs(HCP_DIR, subject, run)

        # Prepare timeseries
        timeseries = ppl.load_single_timeseries(HCP_DIR, subject, run, regions)

        # Fit GLM and get betas
        betas = ppl.compute_betas(timeseries, EVs)

        # Save betas
        outpath = HCP_DIR / "betas" / f"subject_{subject:03}_run_{run}_trial_betas.h5"
        betas.to_hdf(outpath, key="betas")

        # Accumulate
        accumulator.append(betas)

# Concatenate all betas and write to disk
df = pd.concat(accumulator).reset_index(drop=True)
df.to_hdf(HCP_DIR / "betas" / "all_subjects_and_runs_trial_betas.h5", key="betas")
