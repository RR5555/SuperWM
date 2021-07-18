from numpy.lib.npyio import load
from pandas.core import frame
from shared_configuration import HCP_DIR, N_PARCELS, RESULT_DIR, ATLAS_FILE, INIT_CONDS, TR, RUNS, ATLAS_FILE, TASK_KEY
from configuration import N_SUBJECTS, SUM_PATH
from beh_configuration import BEH_SUBJECT_LOC, BEH_SUM_PATH, BEH_N_SUBJECTS
from help_fct import load_single_timeseries,\
	load_evs, average_frames, get_region_info,\
	beh_load_single_timeseries
from event_data import check_EVENT_cond_shape_pd_df,\
	load_EVENT_cond_shape_pd_df, get_ev_value,\
	get_dict_timestamps, get_onset_N_duration,\
	get_dict_timeframes, get_timeseries
from save_N_load import save_obj, load_obj

import os
import matplotlib
import matplotlib.pyplot as plt
# import mpld3
import numpy as np
import pandas as pd
import seaborn as sns

from nilearn import plotting, datasets

if os.path.exists(SUM_PATH):
	os.system(f'rm {SUM_PATH}')

if os.path.exists(BEH_SUM_PATH):
	os.system(f'rm {BEH_SUM_PATH}')

sum_file = open(SUM_PATH, 'w')
beh_sum_file = open(BEH_SUM_PATH, 'w')

def print_in(_msg, _file):
	print(_msg, file=_file)

subjects = range(N_SUBJECTS)
beh_subjects = np.loadtxt(BEH_SUBJECT_LOC,dtype='str')

_tmp = f'Atlas path (the datasets share a same atlas downloaded in {ATLAS_FILE})'
# print(''.join(['\t']*i) for i in range(1,7))
# print(tuple(''.join(['\t']*i for i in range(1,7))))
# print(*tuple(''.join(['\t']*i for i in range(1,7))))

print_in("{0}:\n- hcp{1}- atlas.npz (Atlas file)".format(_tmp, *tuple('\n'+''.join(['\t']*i) for i in range(1,7))), _file=sum_file)
print_in("{0}:\n- hcp{1}- atlas.npz (Atlas file)".format(_tmp, *tuple('\n'+''.join(['\t']*i) for i in range(1,7))), _file=beh_sum_file)

_tmp = 'Folder organisation'

print_in("{0}:\n- hcp{1}- regions.npy (information on the brain parcellation){1}- subjects (main data folder){2}- [subjectID] (subject-specific subfolder){3}- timeseries (){4}- bold{{['7','8'] for WM ('LR', 'RL') resp.}}_Atlas_MSMAll_Glasser360Cortical.npy{3}- EVs (EVs folder){4}- tfMRI_{{'WM' for WM}}_{{('LR', 'RL') for ('LR', 'RL') resp.}}{5}- (3 rows, 1col; onset time, duration, amplitude){6}- 0bk_body.txt{6}- 0bk_faces.txt{6}- 0bk_places.txt{6}- 0bk_tools.txt{6}- 2bk_body.txt{6}- 2bk_faces.txt{6}- 2bk_places.txt{6}- 2bk_tools.txt{5}- (3 rows, k cols; onset time, duration, amplitude){6}- 0bk_cor.txt{6}- 0bk_err.txt{6}- 0bk_nlr.txt{6}- 2bk_cor.txt{6}- 2bk_err.txt{6}- 2bk_nlr.txt{6}- all_bk_cor.txt{6}- all_bk_err.txt{5} - Sync.txt (ignore this file)".format(_tmp, *('\n'+''.join(['\t']*i) for i in range(1,7))), _file=sum_file)
print_in("{0}:\n- hcp_beh{1}- regions.npy (information on the brain parcellation){1}- subjects_list.txt (list of subject IDs){1}- subjects (main data folder){2}- [subjectID] (subject-specific subfolder){3}- EXPERIMENT (one folder per experiment){4}- RUN (one folder per run){5}- data.npy (the parcellated time series data){5}- EVs (EVs folder){6}- [ev1.txt] (one file per condition){6}- [ev2.txt]{6}- Stats.txt (behavioural data [where available] - averaged per run){6}- Sync.txt (ignore this file)".format(_tmp, *('\n'+''.join(['\t']*i) for i in range(1,7))), _file=beh_sum_file)

_tmp = 'Number of subjects:'

print_in(f'{_tmp} {N_SUBJECTS}', _file=sum_file)
print_in(f'{_tmp} {BEH_N_SUBJECTS}', _file=beh_sum_file)

_cue_duration = 2.5
_trial_duration = 2.5
_ITF_duration = 15
nb_cue_per_block = 1
nb_trial_per_block = 10
nb_ITF_per_block = 4
nb_block = 8
run_duration = (nb_cue_per_block*_cue_duration+nb_trial_per_block*_trial_duration)*nb_block+nb_ITF_per_block*_ITF_duration

def print_same_in(_tmp, _var):
	print_in(f"{_tmp}{_var}", _file=sum_file)
	print_in(f"{_tmp}{_var}", _file=beh_sum_file)

print_same_in(_tmp='Exp setting:', _var='')
print_same_in(_tmp='\tTR: ', _var=f'{TR}')
print_same_in(_tmp='\t_cue_duration: ', _var=f'{_cue_duration}')
print_same_in(_tmp='\t_trial_duration: ', _var=f'{_trial_duration}')
print_same_in(_tmp='\t_ITF_duration: ', _var=f'{_ITF_duration}')
print_same_in(_tmp='\tnb_cue_per_block: ', _var=f'{nb_cue_per_block}')
print_same_in(_tmp='\tnb_trial_per_block: ', _var=f'{nb_trial_per_block}')
print_same_in(_tmp='\tnb_ITF_per_block: ', _var=f'{nb_ITF_per_block}')
print_same_in(_tmp='\tnb_block: ', _var=f'{nb_block}')

print_same_in(_tmp='\tSupposed run duration in s (run_duration): ', _var=f"(nb_cue_per_block*_cue_duration+nb_trial_per_block*_trial_duration)*nb_block+nb_ITF_per_block*_ITF_duration=({nb_cue_per_block}*{_cue_duration}+{nb_trial_per_block}*{_trial_duration})*{nb_block}+{nb_ITF_per_block}*{_ITF_duration}={(nb_cue_per_block*_cue_duration+nb_trial_per_block*_trial_duration)*nb_block+nb_ITF_per_block*_ITF_duration}")

print_same_in(_tmp='\tSupposed run duration in frames: ', _var=f"run_duration/TR={run_duration}/{TR}={run_duration/TR}")

_tmp = 'Format of the timeseries data (for subject 0, subject 0 might be different in the two datasets):'
print_in(f"{_tmp} {load_single_timeseries(subject=0, experiment='WM', run=0, remove_mean=False).shape}", _file=sum_file)
print_in(f"{_tmp} {beh_load_single_timeseries(subject=beh_subjects[0], experiment='WM', run=0, remove_mean=False).shape}", _file=beh_sum_file)

_tmp = 'Run duration in s (for subject 0, subject 0 might be different in the two datasets)):'
print_in(f"{_tmp} {load_single_timeseries(subject=0, experiment='WM', run=0, remove_mean=False).shape[1]*TR}", _file=sum_file)
print_in(f"{_tmp} {beh_load_single_timeseries(subject=beh_subjects[0], experiment='WM', run=0, remove_mean=False).shape[1]*TR}", _file=beh_sum_file)

def get_range_per_parcel(_loading_fct, _subjects, remove_mean):
	max = np.zeros((N_PARCELS,))
	min = np.zeros((N_PARCELS,))
	mean = np.zeros((N_PARCELS,))
	accumulated = 0
	for subject in _subjects:
		for run in (0, 1):
			# print(np.max(_loading_fct(subject=subject, experiment='WM', run=run, remove_mean=remove_mean), axis=1).shape)
			# exit()
			_tmp = _loading_fct(subject=subject, experiment='WM', run=run, remove_mean=remove_mean)
			max = np.maximum(max,np.max(_tmp, axis=1))
			min = np.minimum(min,np.min(_tmp, axis=1))
			mean = mean + np.sum(_tmp, axis=1)
			assert _tmp.shape[1] == 405, 'Shape of temporal dat has changed:{}!=405'.format(_tmp.shape[1])
			accumulated += _tmp.shape[1]
	return max, min, mean/accumulated

def get_summary_on_all_parcels(_title, _remove_mean, _path):
	print_same_in(f"\n{''.join(['#']*20)}\n{_title}", '')
	_max, _min, _mean = get_range_per_parcel(_loading_fct=load_single_timeseries, _subjects=range(N_SUBJECTS), remove_mean=_remove_mean)
	# print(_min.shape, _max.shape, _mean.shape)
	df = pd.DataFrame(data=np.stack((_min, _max, _max-_min, _mean), axis=1), columns=('min', 'max', 'range', 'mean'))
	df_path = RESULT_DIR+'/'+_path+'.fth'
	df.to_feather(df_path)
	assert df.equals(pd.read_feather(df_path))
	print_in(f'Stats stored at {df_path}:\n{df}', _file=sum_file)
	print_in(f"Marginal stat over all parcels:\n\tmin:{df.loc[:,'min'].min()}\n\tmax:{df.loc[:,'max'].max()}\n\tmax range:{df.loc[:,'range'].max()}", _file=sum_file)

	_max, _min, _mean = get_range_per_parcel(_loading_fct=beh_load_single_timeseries, _subjects=beh_subjects, remove_mean=_remove_mean)
	# print(_min.shape, _max.shape, _mean.shape)
	df = pd.DataFrame(data=np.stack((_min, _max, _max-_min, _mean), axis=1), columns=('min', 'max', 'range', 'mean'))
	df_path = RESULT_DIR+'/beh_'+_path+'.fth'
	df.to_feather(df_path)
	print_in(f'Stats stored at {df_path}:\n{df}', _file=beh_sum_file)
	print_in(f"Marginal stat over all parcels:\n\tmin:{df.loc[:,'min'].min()}\n\tmax:{df.loc[:,'max'].max()}\n\tmax range:{df.loc[:,'range'].max()}", _file=beh_sum_file)

get_summary_on_all_parcels(_title='Without mean substraction:', _remove_mean=False, _path='summary_all_parcels')
get_summary_on_all_parcels(_title='With mean substraction:', _remove_mean=True, _path='summary_all_parcels_w_mean_sub')

def find_correspondence(remove_mean=False):
	_index_in_original = {}
	for beh_subject in beh_subjects:
		beh_data_LR = beh_load_single_timeseries(subject=beh_subject, experiment='WM', run=0, remove_mean=remove_mean)
		beh_data_RL = beh_load_single_timeseries(subject=beh_subject, experiment='WM', run=1, remove_mean=remove_mean)
		_tmp = []
		for original_subject in range(N_SUBJECTS):
			# The normal and coherent check
			if np.array_equal(beh_data_LR, load_single_timeseries(subject=original_subject, experiment='WM', run=0, remove_mean=remove_mean))\
					and np.array_equal(beh_data_RL, load_single_timeseries(subject=original_subject, experiment='WM', run=1, remove_mean=remove_mean)):
				_tmp.append(original_subject)
			# In case, there had been a messing-up between LR and RL in the two datasets
			if np.array_equal(beh_data_LR, load_single_timeseries(subject=original_subject, experiment='WM', run=1, remove_mean=remove_mean))\
					and np.array_equal(beh_data_RL, load_single_timeseries(subject=original_subject, experiment='WM', run=0, remove_mean=remove_mean)):
				_tmp.append(original_subject)
		# if len(_tmp) != 1:
		# 	raise ValueError(f'For beh subject {beh_subject}, corresponding index in original is:{_tmp}')
		_index_in_original[beh_subject]=_tmp
	return _index_in_original
print('Without mean substraction:')
print(find_correspondence(remove_mean=False))
print('With mean substraction:')
print(find_correspondence(remove_mean=False))
# df.equals(exactly_equal)
# df.compare(df2, keep_shape=True)

sum_file.close()
beh_sum_file.close()
