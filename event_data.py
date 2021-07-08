from configuration import EXPERIMENTS,\
    HCP_DIR, N_SUBJECTS, RESULT_DIR,\
	TR
from help_fct import load_single_timeseries
from save_N_load import save_obj, load_obj

import numpy as np
import pandas as pd
import os

def get_shape_evs(subject, experiment, run, _dict):
	"""Get the shapes for the given subject, experiment, run, and conditions-the one present in the **_dict** dict - of the EV. 

	Args:
		* **subject** (int): index in range(N_SUBJECTS)-see configuration.py-, range(339)
		* **experiment** (str): 'WM'
		* **run** (str): 'RL' or 'LR'
		* **_dict** (dict): dict with nested keys of type: int (subject), 'RL' or 'LR' (run), str (cond)
	"""
	task_key = 'tfMRI_'+experiment+'_'+['RL','LR'][run]
	for cond in EXPERIMENTS[experiment]['cond']:
		ev_file  = f"{HCP_DIR}/subjects/{subject}/EVs/{task_key}/{cond}.txt"
		ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
		if cond in _dict[subject][['RL','LR'][run]].keys():
			_dict[subject][['RL','LR'][run]][cond]=ev_array.shape

def create_EVENT_cond_shape_pd_df():
	"""Create a pd.DataFrame containing all the shapes for the EVENT condition of the WM experiment for all runs and subjects.

	Returns:
		* (pd.DataFrame): with columns: (index, 'subject', '0bk_cor', '0bk_err',  '0bk_nlr', '2bk_cor', '2bk_err',  '2bk_nlr', 'all_bk_cor', 'all_bk_err', 'Sync')
	"""
	EVENT_cond=('0bk_cor', '0bk_err',  '0bk_nlr', '2bk_cor', '2bk_err',  '2bk_nlr', 'all_bk_cor', 'all_bk_err', 'Sync')
	_dict = {_subj:{_run:{_ev:0 for _ev in EVENT_cond} for _run in ('RL','LR')} for _subj in range(N_SUBJECTS)}
	for _subj in range(N_SUBJECTS):
		for _run in (0,1):
			get_shape_evs(subject=_subj, experiment='WM',run=_run, _dict=_dict)
	pd_df = pd.concat({k: pd.DataFrame(v).T for k, v in _dict.items()}, axis=0)
	pd_df = pd_df.reset_index()
	# print(pd_df.head(5))
	pd_df.columns = ['subject', 'run']+list(pd_df.columns[2:])
	# print(pd_df.head(5))
	return pd_df

def create_N_save_EVENT_cond_shape_pd_df(csv_name):
	"""Save the pd.DataFrame created by the create_EVENT_cond_shape_pd_df() fct in the **csv_name** csv file.

	Args:
		* **csv_name** (str): path for saving the file
	"""
	pd_df = create_EVENT_cond_shape_pd_df()
	pd_df.to_csv(csv_name)

def check_EVENT_cond_shape_pd_df(shape_file_path):
	"""Create and save the EVENT cond pd.DataFrame if it does not already exist.

	Args:
		* **shape_file_path** (str): path to the saving file of the EVENT cond pd.DataFrame
	"""
	if not os.path.exists(shape_file_path):
		create_N_save_EVENT_cond_shape_pd_df(shape_file_path)

def load_EVENT_cond_shape_pd_df(shape_file_path):
	"""Load the EVENT cond pd.DataFrame given by the **shape_file_path** path.

	Args:
		* **shape_file_path** (str): path to the saved file of the EVENT cond pd.DataFrame

	Returns:
		* (pd.DataFrame): with columns: (index, 'subject', '0bk_cor', '0bk_err',  '0bk_nlr', '2bk_cor', '2bk_err',  '2bk_nlr', 'all_bk_cor', 'all_bk_err', 'Sync')
	"""
	pd_df = pd.read_csv(shape_file_path)
	return pd_df.drop(pd_df.columns[0], axis=1)

def get_ev_value(subject, experiment, run, cond):
	"""The specific EV values for a given subject, experiment, run, and condition.

	Args:
		* **subject** (int): index in range(N_SUBJECTS)-see configuration.py-, range(339)
		* **experiment** (str): 'WM'
		* **run** (str): 'RL' or 'LR'
		* **cond** (str): see configuration.py file '_ord_conds'

	Returns:
		* (np.array): the loaded EV (a (1,0) shape indicate an empty file)
	"""
	task_key = 'tfMRI_'+experiment+'_'+run
	ev_file  = f"{HCP_DIR}/subjects/{subject}/EVs/{task_key}/{cond}.txt"
	return np.loadtxt(ev_file, ndmin=2, unpack=True)

def get_onset_N_duration(subject, run, cond):
	_array = get_ev_value(subject=subject, experiment='WM', run=run, cond=cond).tolist()
	# print(_array)
	if len(_array) == 3:
		_onset = _array[0]
		_duration = _array[1]
		return _onset, _duration
	return [], None

def dict_timestamps():
	block_cond = ('0bk_body', '0bk_faces', '0bk_places', '0bk_tools',
		'2bk_body', '2bk_faces', '2bk_places', '2bk_tools')
	run_type = ('LR', 'RL')
	restricted_event_cond = ('0bk_cor', '0bk_err',  '0bk_nlr', '2bk_cor', '2bk_err',  '2bk_nlr')

	_dict = {subj:{run:{cond:None for cond in block_cond} for run in run_type} for subj in range(N_SUBJECTS)}
	for subj in range(N_SUBJECTS):
		for run in run_type:
			for cond in block_cond:
				block_onset, block_duration = get_onset_N_duration(subject=subj, run=run, cond=cond)
				event_onsets = []
				event_onset_type = []
				event_durations = []
				for event_cond in restricted_event_cond:
					event_onset, event_duration = get_onset_N_duration(subject=subj, run=run, cond=event_cond)
					if event_duration is not None:
						for event, duration in zip(event_onset, event_duration):
							if block_onset[0] <= event <= block_onset[0]+block_duration[0]:
								event_onsets.append(event)
								event_durations.append(duration)
								event_onset_type.append(event_cond[-3:])
				_dict[subj][run][cond]=(block_onset[0], block_duration[0], event_onsets, event_onset_type, event_durations)
	return _dict

def get_dict_timestamps():
	print('Getting dict timestamps.')
	if not os.path.exists(RESULT_DIR+'/timestamps.pkl'):
		print('Creating the dict:')
		_dict = dict_timestamps()
		save_obj(_dict, RESULT_DIR+'/timestamps')
		print('\t Dict created and saved')
	else:
		print('Loading the dict:')
		_dict = load_obj(RESULT_DIR+'/timestamps')
		print('\t Dict loaded')
	return _dict

def get_frames(_onset, _duration):
	start = np.floor(_onset / TR).astype(int)
	duration = np.ceil(_duration / TR).astype(int)
	frames = start + np.arange(0, duration) 
	return frames

def dict_timeframes(_dict_timestamps):
	block_cond = ('0bk_body', '0bk_faces', '0bk_places', '0bk_tools',
		'2bk_body', '2bk_faces', '2bk_places', '2bk_tools')
	run_type = ('LR', 'RL')
	abridged_event_cond = ('cor', 'err',  'nlr')

	_dict = {subj:{run:{cond:{event_cond:None for event_cond in list(abridged_event_cond)+['block', 'cue']} for cond in block_cond} for run in run_type} for subj in range(N_SUBJECTS)}
	for subj in range(N_SUBJECTS):
		for run in run_type:
			for cond in block_cond:
				block_onset, block_duration, event_onsets, event_onset_type, event_durations = _dict_timestamps[subj][run][cond]
				_dict[subj][run][cond]['block']= get_frames(_onset=block_onset, _duration=block_duration)
				_dict[subj][run][cond]['cue']= get_frames(_onset=block_onset, _duration=2.5)
				for event_cond in abridged_event_cond:
					_index = [_i for _i, _type in enumerate(event_onset_type) if _type==event_cond]
					_dict[subj][run][cond][event_cond]= [get_frames(_onset=event_onsets[_i], _duration=event_durations[_i]) for _i in _index]
	return _dict

def get_dict_timeframes():
	print('Getting dict timeframes.')
	if not os.path.exists(RESULT_DIR+'/timeframes.pkl'):
		print('Creating the dict:')
		_dict = dict_timeframes(_dict_timestamps=get_dict_timestamps())
		save_obj(_dict, RESULT_DIR+'/timeframes')
		print('\t Dict created and saved')
	else:
		print('Loading the dict:')
		_dict = load_obj(RESULT_DIR+'/timeframes')
		print('\t Dict loaded')
	return _dict

_dict_timeframes = get_dict_timeframes()
def get_timeseries(subject, run, cond, event):
	data =load_single_timeseries(subject=subject,experiment='WM',run= 0 if run=='RL' else 1,remove_mean=True)
	timeseries = [data[:,frames] for frames in _dict_timeframes[subject][run][cond][event]]
	return timeseries