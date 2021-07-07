from configuration import EXPERIMENTS,\
    HCP_DIR, N_SUBJECTS

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
