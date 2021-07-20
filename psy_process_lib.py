#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains functions for processing fMRI data from the HCP2021 dataset for the
Working Memory task, curated for the NMA-CN 2021 session. The functions are to be used
as individual steps of processing in a script - see corresponding psy_process_runner.py
for such a pipeline.

@author: Pranay S. Yadav
"""

# %% Import libraries
import numpy as np
import pandas as pd
from pathlib import Path

# %% Data-specific environment variables
# Voxel data has already been aggregated into ROIs from the Glasser parcellation
N_PARCELS = 360

# The acquisition parameters for all tasks were identical
TR = 0.72  # Time resolution, in seconds

# The parcels are matched across hemispheres with the same order
HEMIS = ["Right", "Left"]

# Each experiment was repeated twice in each subject
RUNS = {"LR": 7, "RL": 8}
N_RUNS = 2

# Task conditions for WM experiment
EXPERIMENTS = {
    "WM": {
        "cond": [
            "0bk_body",
            "0bk_faces",
            "0bk_places",
            "0bk_tools",
            "2bk_body",
            "2bk_faces",
            "2bk_places",
            "2bk_tools",
        ]
    },
}

# %% Function definitions
def load_regions(fname):
    """
    Load identifier information for all 360 ROIs - name, network, hemisphere for each

    Parameters
    ----------
    fname : str or Path object
        Full path to file containing region data stored in npy container.

    Returns
    -------
    region_info : pd.DataFrame
        Dataframe of shape (360, 3) with identifiers for each ROI.

    """
    # Convert to Path object if necessary
    if not isinstance(fname, Path):
        fname = Path(fname)

    # Check if file exists and has correct extension
    assert fname.exists(), "File doesn't exist"
    assert fname.suffix == ".npy", "File doesn't have .npy extension"

    # Load data, convert to dataframe with appropriate labels and return
    regions = np.load(fname).T
    region_info = pd.DataFrame(
        dict(
            name=regions[0].tolist(),
            network=regions[1],
            hemi=["Right"] * int(N_PARCELS / 2) + ["Left"] * int(N_PARCELS / 2),
        )
    )

    return region_info


def load_single_timeseries(HCP_DIR, subject, run, regions, remove_mean=True):
    """
    Load timeseries data for a single subject and single run.

    Parameters
    ----------
    HCP_DIR : str or Path object
        Full path to root directory containing dataset.
    subject : int
        Subject ID to load.
    run : str
        Run to load, LR: 7, RL: 8.
    regions: pd.DataFrame
        Identifiers for all 360 parcels obtained from load_regions()
    remove_mean : bool, optional
        Subtract parcel-wise mean BOLD signal. The default is True.

    Returns
    -------
    ts : pd.DataFrame
        Dataframe of shape (n_frames, n_parcels) containing BOLD values.
        Full data per subject per run has size (405, 360)

    """
    # Prepare filename from input arguments
    bold_run = RUNS[run]
    bold_path = f"{HCP_DIR}/subjects/{subject}/timeseries/"
    bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"

    # Load data and remove mean if requested
    ts = np.load(f"{bold_path}/{bold_file}")
    if remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)

    # Convert to Dataframe, add identifiers and return
    ts = pd.DataFrame(ts, index=[regions["name"], regions["network"]]).T
    ts["Subject"] = subject
    ts["Run"] = run

    return ts
