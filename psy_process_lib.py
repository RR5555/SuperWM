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
RUNS = ["LR", "RL"]
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
