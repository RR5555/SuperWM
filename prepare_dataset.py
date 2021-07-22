from base_fct import print_N_log
from help_fct import load_single_timeseries,\
	beh_load_single_timeseries
from event_data import get_timeseries
from configuration import N_SUBJECTS
from beh_configuration import BEH_SUBJECT_LOC
from shared_configuration import INIT_CONDS, RUNS, ACC_EVENT_COND

import numpy as np

DATASET_DICT= {'HCP2021WM':{'subjects':range(N_SUBJECTS), 'load_fct':load_single_timeseries},
'HCP2021WM_BEH':{'subjects': np.loadtxt(BEH_SUBJECT_LOC,dtype='str'), 'load_fct':beh_load_single_timeseries}}
#TODO: make a get_timeseries for beh data 

# make a raw dataset for n-back differentiation
def get_raw_dataset_for_n_back_diff_agg(dataset_desc_dict, preproc_fct=lambda x: x, include_cue=False, include_bias=False, _remove_mean=False, _preproc=lambda x:x):
	_data = []
	_labels = []
	ev_conds = ('cor', 'err') if not include_cue else ('cue', 'cor', 'err')
	for _subject in dataset_desc_dict['subjects']:
		for run in RUNS:
			for ev_cond in ev_conds:
				for cond in INIT_CONDS:
					_tmp = [preproc_fct(ts) for ts in get_timeseries(subject=_subject, run=run, cond=cond, event=ev_cond, _remove_mean=_remove_mean)]
					_data.extend(_tmp)
					_labels.extend([0]*len(_tmp) if cond.find('0bk')>=0 else [1]*len(_tmp))
	assert len(_data) == len(_labels), f"{len(_data)}!={len(_labels)}"
	# print(_data[0].shape)
	# print(_labels)
	# exit()
	print_N_log(f"{np.ones((len(_labels),1)).shape}, {np.stack((x.mean(axis=1, keepdims=False) for x in _data), axis=0).shape}", log_dst='debug')
	_tmp = np.stack((x.mean(axis=1, keepdims=False) for x in _data), axis=0)
	_tmp = _preproc(_tmp)
	return _tmp if not include_bias else  np.concatenate((np.ones((len(_labels),1)), _tmp), axis=1), np.array(_labels)

def get_raw_dataset_for_n_back_diff(dataset_desc_dict, preproc_fct=lambda x: x, include_cue=False, include_bias=False, _remove_mean=False, _preproc=lambda x:x):
	_data = []
	_labels = []
	ev_conds = ('cor', 'err') if not include_cue else ('cue', 'cor', 'err')
	for _subject in dataset_desc_dict['subjects']:
		for run in RUNS:
			for ev_cond in ev_conds:
				for cond in INIT_CONDS:
					_tmp = [preproc_fct(ts) for ts in get_timeseries(subject=_subject, run=run, cond=cond, event=ev_cond, _remove_mean=_remove_mean)]
					_data.extend(_tmp)
					_labels.extend([0]*len(_tmp) if cond.find('0bk')>=0 else [1]*len(_tmp))
	assert len(_data) == len(_labels), f"{len(_data)}!={len(_labels)}"
	# print(_data[0].shape)
	# print(_labels)
	# exit()
	print_N_log(f"{np.ones((len(_labels),1)).shape}, {np.stack((x.flatten() for x in _data), axis=0).shape}", log_dst='debug')
	_tmp = np.stack((x.flatten() for x in _data), axis=0)
	_tmp = _preproc(_tmp)
	return _tmp if not include_bias else  np.concatenate((np.ones((len(_labels),1)), _tmp), axis=1), np.array(_labels)

# make a raw dataset for accuracy differentiation
# make a HRF-processed dataset for n-back differentiation
# make a HRF-processed for accuracy differentiation
# make an extended-time raw dataset for n-back differentiation
# make an extended-time raw for accuracy differentiation
