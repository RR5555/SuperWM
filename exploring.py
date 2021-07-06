from configuration import HCP_DIR, N_SUBJECTS, TASK_KEY, RESULT_DIR, ATLAS_FILE
from help_fct import load_single_timeseries,\
    load_evs, average_frames, get_region_info

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

evs = load_evs(subject=my_subj, experiment=my_exp,run=my_run)

body_activity = average_frames(data, evs, my_exp, '0bk_body')
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
