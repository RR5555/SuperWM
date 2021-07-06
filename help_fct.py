from configuration import HCP_DIR,\
	N_PARCELS, EXPERIMENTS, TR

import numpy as np

# Page 49-50: https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf

def get_region_info():
	"""Return the region name and network assignment for each parcel in dict.

	Returns:
		* (dict): keys: ('name', 'network', 'hemi')
	"""
	regions = np.load(f"{HCP_DIR}/regions.npy").T
	return dict(name=regions[0].tolist(), network=regions[1],
		hemi=['Right']*int(N_PARCELS/2) + ['Left']*int(N_PARCELS/2),)

def load_single_timeseries(subject, experiment, run, remove_mean=True):
	"""Load timeseries data for a single subject and single run.
	
	Args:
		subject (int):      0-based subject ID to load
		experiment (str):   Name of experiment 
		run (int):          0-based run index, across all tasks
		remove_mean (bool): If True, subtract the parcel-wise mean (typically the mean BOLD signal is not of interest)

	Returns
		ts (n_parcel x n_timepoint np.array): Array of BOLD data values

	"""
	bold_run  = EXPERIMENTS[experiment]['runs'][run]
	bold_path = f"{HCP_DIR}/subjects/{subject}/timeseries"
	bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
	ts = np.load(f"{bold_path}/{bold_file}")
	# print('ts shape:{}'.format(ts.shape))
	if remove_mean:
		ts -= ts.mean(axis=1, keepdims=True)
	return ts


def load_evs(subject, experiment, run):
	"""Load EVs (explanatory variables) data for one task experiment.

	Args:
		subject (int): 0-based subject ID to load
		experiment (str) : Name of experiment

	Returns
		evs (list of lists): A list of frames associated with each condition

	"""
	frames_list = []
	task_key = 'tfMRI_'+experiment+'_'+['RL','LR'][run]
	for cond in EXPERIMENTS[experiment]['cond']:    
		ev_file  = f"{HCP_DIR}/subjects/{subject}/EVs/{task_key}/{cond}.txt"
		ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
		ev       = dict(zip(["onset", "duration", "amplitude"], ev_array))
		# Determine when trial starts, rounded down
		start = np.floor(ev["onset"] / TR).astype(int)
		# Use trial duration to determine how many frames to include for trial
		duration = np.ceil(ev["duration"] / TR).astype(int)
		# Take the range of frames that correspond to this specific trial
		frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
		frames_list.append(frames)

	return frames_list

# we need a little function that averages all frames from any given condition

def average_frames(data, evs, experiment, cond):
	"""Averages all frames from any given condition.

	Args:
		* **data** (:xref:[type]): [description]
		* **evs** (:xref:[type]): [description]
		* **experiment** (:xref:[type]): [description]
		* **cond** (:xref:[type]): [description]

	Returns:
		* ( np.array): [description]
	"""
	idx = EXPERIMENTS[experiment]['cond'].index(cond)
	# print(evs[idx])
	return np.mean(np.concatenate([np.mean(data[:,evs[idx][i]],axis=1,keepdims=True) for i in range(len(evs[idx]))],axis=-1),axis=1)