
HCP_DIR = "./hcp"
ATLAS_FILE = f"{HCP_DIR}/atlas.npz"
RESULT_DIR = './results'

# The data have already been aggregated into ROIs from the Glasser parcellation
N_PARCELS = 360

# The acquisition parameters for all tasks were identical
TR = 0.72  # Time resolution, in seconds

# The parcels are matched across hemispheres with the same order
HEMIS = ["Right", "Left"]

# Each experiment was repeated twice in each subject
N_RUNS = 2
RUNS = ('LR', 'RL')

INIT_CONDS = ['0bk_body','0bk_faces','0bk_places','0bk_tools',
				'2bk_body','2bk_faces','2bk_places','2bk_tools']

TASK_KEY = ['tfMRI_WM_'+'RL','tfMRI_WM_'+'LR']

ACC_EVENT_COND = ('cor', 'err',  'nlr')
ALL_EVENT_COND = ['block', 'cue'].extend(ACC_EVENT_COND)