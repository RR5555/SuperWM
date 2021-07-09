from numpy.lib.npyio import load
from pandas.core import frame
from configuration import HCP_DIR, N_SUBJECTS, TASK_KEY, RESULT_DIR, ATLAS_FILE, INIT_CONDS, TR, RUNS
from help_fct import load_single_timeseries,\
	load_evs, average_frames, get_region_info
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

subject=0
run=1
task_key = TASK_KEY[run]
os.system(f"ls {HCP_DIR}/subjects/{subject}/EVs/{task_key}/")


my_exp  = 'WM'
my_subj = 0
my_run  = 0 #0 or 1

data = load_single_timeseries(subject=my_subj,experiment=my_exp,run=my_run,remove_mean=True)
print(data.shape)
plt.plot(data)
plt.savefig(RESULT_DIR+'/test_single_timeseries.pdf', backend='pdf')
plt.close()

## Test if all subjects over all runs in WM have an ev_array of shape (3,1)
# for _subj in range(N_SUBJECTS):
# 	for _run in (0,1):
# 		load_evs(subject=_subj, experiment=my_exp,run=_run)
# exit()
#All subjects have a (3,1) ev_array for the init_conds but not for ord_conds

###### On EVENT cond ###############
EVENT_cond=('0bk_cor', '0bk_err',  '0bk_nlr', '2bk_cor', '2bk_err',  '2bk_nlr', 'all_bk_cor', 'all_bk_err', 'Sync')

# shape_file_path = RESULT_DIR+'/file_csv.csv'
# check_EVENT_cond_shape_pd_df(shape_file_path)
# pd_df = load_EVENT_cond_shape_pd_df(shape_file_path)
# print(pd_df.head(40))

# print(get_ev_value(subject=0, experiment='WM', run='LR', cond='2bk_err'))
# print(f"{''.join(['#']*40)}")
# for cond in ('0bk_cor', '0bk_err',  '0bk_nlr', '0bk_body','0bk_faces','0bk_places','0bk_tools'):
# 	print(f"{cond}:{get_ev_value(subject=0, experiment='WM', run='LR', cond=cond)}")



# _dict_timestamps = get_dict_timestamps()
# for cond in INIT_CONDS:
# 	for run in ('RL', 'LR'):
# 		print(f"{cond},{run}:{_dict_timestamps[1][run][cond]}")

# _dict_timeframes = get_dict_timeframes()
# print(_dict_timeframes[0]['LR']['2bk_body']['cor'])

# the possible events for the 'event' argument: ('block', 'cue', 'cor', 'err',  'nlr')
ts = get_timeseries(subject=0, run='LR', cond='2bk_body', event='cor')[0]
plt.plot(ts)
plt.legend([f'{_i}' for _i in range(len(ts))])
plt.savefig(RESULT_DIR+'/test_event_single_timeseries.pdf', backend='pdf')
plt.close()

plt.matshow(ts)
plt.ylabel('ROI')
plt.xlabel('time')
plt.savefig(RESULT_DIR+'/test_event_single_timeseries_mat.pdf', backend='pdf')
plt.close()

# Exp1: contrast 2-back vs 0-back averaged on all conditions
_0bk_data = []
_2bk_data = []
for subject in range(N_SUBJECTS):
	for run in RUNS:
		for cond in INIT_CONDS:
			if '0bk' in cond:
				_0bk_data.extend(get_timeseries(subject=subject, run=run, cond=cond, event='block'))
			if '2bk' in cond:
				_2bk_data.extend(get_timeseries(subject=subject, run=run, cond=cond, event='block'))
np_0bk_data = np.array(_0bk_data)
np_2bk_data = np.array(_2bk_data)
# save_obj(np_0k_back, RESULT_DIR+'/np_0k_back')
# save_obj(np_2k_back, RESULT_DIR+'/np_2k_back')
np.savez_compressed(file=RESULT_DIR+'/np_0bk_data',np_0k_back=np_0bk_data)
np.savez_compressed(file=RESULT_DIR+'/np_2bk_data',np_2k_back=np_2bk_data)
print(f'np_0k_back:{np_0bk_data.shape}')
print(f'np_2k_back:{np_2bk_data.shape}')

# print(_dict[0]['LR']['2bk_places'])
exit()
####################################

evs = load_evs(subject=my_subj, experiment=my_exp,run=my_run)

body_activity = average_frames(data, evs, my_exp, '0bk_body')
# nlr_activity = average_frames(data, evs, my_exp, '0bk_nlr') #TRIAL
faces_activity = average_frames(data, evs, my_exp, '0bk_faces')
contrast = body_activity-faces_activity   # difference between left and right hand movement
print(f'contrast.shape:{contrast.shape}')

# Plot activity level in each ROI for both conditions
plt.plot(body_activity,label='0bk_body')
plt.plot(faces_activity,label='0bk_faces')
plt.xlabel('ROI')
plt.ylabel('activity')
plt.legend()
plt.savefig(RESULT_DIR+'/test_activity_lvl.pdf', backend='pdf')
plt.close()

region_info = get_region_info()


df = pd.DataFrame({'body_activity' : body_activity,
					'faces_activity' : faces_activity,
					'network'     : region_info['network'],
					'hemi'        : region_info['hemi']})

fig,(ax1,ax2) = plt.subplots(1,2)
sns.barplot(y='network', x='body_activity', data=df, hue='hemi',ax=ax1)
sns.barplot(y='network', x='faces_activity', data=df, hue='hemi',ax=ax2)
plt.savefig(RESULT_DIR+'/test_hist.pdf', backend='pdf')
plt.close()

subjects = range(N_SUBJECTS)
group_contrast = 0
for s in subjects:
	for r in [0,1]:
		data = load_single_timeseries(subject=s,experiment=my_exp,run=r,remove_mean=True)
		evs = load_evs(subject=s, experiment=my_exp,run=r)

		body_activity = average_frames(data, evs, my_exp, '0bk_body')
		faces_activity = average_frames(data, evs, my_exp, '0bk_faces')

		contrast    = body_activity-faces_activity
		group_contrast        += contrast

group_contrast /= (len(subjects)*2)  # remember: 2 sessions per subject

df = pd.DataFrame({'contrast':group_contrast,'network':region_info['network'],'hemi':region_info['hemi']})
# we will plot the left foot minus right foot contrast so we only need one plot
sns.barplot(y='network', x='contrast', data=df, hue='hemi')
plt.savefig(RESULT_DIR+'/test_contrast.pdf', backend='pdf')
plt.close()


with np.load(ATLAS_FILE) as dobj:
	atlas = dict(**dobj)

# Try both hemispheres (L->R and left->right)
fsaverage = datasets.fetch_surf_fsaverage(data_dir=RESULT_DIR)
surf_contrast = group_contrast[atlas["labels_L"]]
plotting.view_surf(fsaverage['infl_left'],
					surf_contrast,
					vmax=20).save_as_html(RESULT_DIR+'/test_surf.html')
